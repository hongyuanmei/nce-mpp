import random
import numpy as np
import torch
import torch.nn as nn


class NCE(nn.Module): 

	def __init__(self, *, 
		noise_process_num, noise_type_num, over_rate, redraw_prob, device=None): 
		super(NCE, self).__init__()
		"""
		:param noise_process_num: M1, # of noise processes in parallel 
		:param noise_type_num: M2, # of noise types per time t
		:param over_rate: over-sampling rate for noise times
		"""
		device = device or 'cpu'
		self.device = torch.device(device)
		self.noise_process_num = noise_process_num
		self.noise_type_num = noise_type_num
		self.over_rate = over_rate
		self.redraw_prob = redraw_prob
		self.eps = np.finfo(float).eps
		assert noise_process_num >= 1, f"invalid noise process # : {noise_process_num} "
		assert noise_type_num >= 1, f"invalid noise type # : {noise_type_num}"
		assert over_rate >= 1.0, f"invalid over rate : {over_rate} "

		"""
		reparametric normalizer for binary-classification NCE (Guo et al 2018)
		"""
		self.normalizer = None 


	def cuda(self, device=None):
		device = device or 'cuda'
		self.device = torch.device(device)
		assert self.device.type == 'cuda'
		super().cuda(self.device)

	def cpu(self):
		self.device = torch.device('cpu')
		super().cuda(self.device)

	#@profile
	def forward(self, method, model, noise, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor, 
		noise_tensor_pack): 
		"""
		input : 
			method [str] : which NCE method to use -- frac or async or sync?
			model [object of GNHP] : p_theta or simply p 
			noise [object of GNHP/GPP] : p_N or simply q (q is a convention for proposal dist)
			event_tensor [B x T+2] : a batch of event types with length T+2
			dtime_tensor [B x T+2] : a batch of dtimes with length T+2
			token_num_tensor [B] : # of tokens per seq 
			duration_tensor [B] : duration per seq
			noise_tensor_pack : list of tensors related to noise samples
		"""
		"""
		NOTE : for now, we only use NCE to train GNHP, not GPP
		GPP can only be used as a noise distribution 
		"""
		"""
		get cells, gates & states for actual events from model
		"""
		target_tensor, mask_tensor = model.get_target(event_tensor)
		# B x T+1 , starting from 1st actual event 

		have_noise = noise_tensor_pack is not None
		prob_small = self.redraw_prob <= 1e-6
		random_draw = random.uniform(0.0, 1.0) <= self.redraw_prob 
		
		if not have_noise or (not prob_small and random_draw):
			"""
			no noise samples yet or roll a dice and decide to draw
			"""
			with torch.no_grad(): 
				"""
				get noise times and types and their q intensities 
				"""
				"""
				_time : _noise == at noise times, _actual == at actual times
				_time_type : _actual_noise == at actual times, of noise types
				"""
				all_type_actual_noise, all_type_noise_noise, \
				all_dtime_noise, all_mask_noise, \
				inten_q_actual_both, inten_q_noise_noise, \
				all_accept_prob_noise, all_S_noise, \
				all_inten_num_noise = \
				noise.get_noise_samples(
					method, 
					event_tensor, dtime_tensor, target_tensor, mask_tensor, duration_tensor, 
					self.noise_process_num, self.noise_type_num, self.over_rate
				)
				# _noise : B x T' 
				# _actual : B x (T+1)
				# _actual_noise : B x (T+1) x noise_type_num
				# _noise_noise : B x T' x noise_type_num
				# S : <T' 
				count_noise = 1.0
				new_noise_pack = all_type_actual_noise, all_type_noise_noise, \
				all_dtime_noise, all_mask_noise, \
				inten_q_actual_both, inten_q_noise_noise, \
				all_accept_prob_noise, all_S_noise, \
				all_inten_num_noise
		elif have_noise and (prob_small or not random_draw): 
			"""
			use pre-saved noise pack
			"""
			all_type_actual_noise, all_type_noise_noise, \
			all_dtime_noise, all_mask_noise, \
			inten_q_actual_both, inten_q_noise_noise, \
			all_accept_prob_noise, all_S_noise, \
			all_inten_num_noise = noise_tensor_pack

			count_noise = 0.0
			new_noise_pack = None # don't update it
		else: 
			raise Exception(f"Unknown case : do you want to sample at all or not?!")
		
		"""
		given all of K (fine) intensities of noise dist q 
		at both noise and actual times 
		find the intensities of actual and noise event types 
		_both : of both actual and noise event types 
		"""
		all_type_actual_both = torch.cat(
			[ target_tensor.unsqueeze(-1), all_type_actual_noise ], dim=-1
		) # B x (T + 1) x (1 + noise_type_num) # 0-th == actual
		"""
		compute cells and gates for model p
		"""
		all_c_p_actual, all_cb_p_actual, all_d_p_actual, all_o_p_actual, _, _ = \
		model.get_cells_gates_states(event_tensor, dtime_tensor )
		# B x T+1 x D 
		"""
		interpolate cells and gates at noise times
		"""
		all_c_p_noise, all_cb_p_noise, \
		all_d_p_noise, all_o_p_noise = model.get_interpolated_cells(
			all_c_p_actual, all_cb_p_actual, all_d_p_actual, all_o_p_actual, 
			all_S_noise
		) # B x T' x D 
		"""
		compute intensities of model distribution p 
		"""
		inten_p_actual_both = model.get_intensities_given_types(
			all_c_p_actual, all_cb_p_actual, all_d_p_actual, all_o_p_actual, 
			all_type_actual_both, dtime_tensor[:, 1:]
		) # B x (T + 1) x (1 + noise_type_num)
		inten_p_noise_noise = model.get_intensities_given_types(
			all_c_p_noise, all_cb_p_noise, all_d_p_noise, all_o_p_noise, 
			all_type_noise_noise, all_dtime_noise
		) # B x T' x noise_type_num 
		if method == 'nce_frac': 
			nce_obj = self.compute_obj_frac(
				inten_p_actual_both, inten_p_noise_noise, 
				inten_q_actual_both, inten_q_noise_noise, 
				mask_tensor, all_mask_noise, all_accept_prob_noise
			)
		elif method == 'nce_async': 
			nce_obj = self.compute_obj_async(
				inten_p_actual_both, inten_p_noise_noise, 
				inten_q_actual_both, inten_q_noise_noise, 
				mask_tensor, all_mask_noise
			)
		elif method == 'nce_sync': 
			nce_obj = self.compute_obj_sync(
				inten_p_actual_both, inten_p_noise_noise, 
				inten_q_actual_both, inten_q_noise_noise, 
				mask_tensor, all_mask_noise
			)
		elif method == 'nce_binary': 

			if not self.normalizer: 
				self.normalizer = nn.Sequential(
					nn.Linear(model.hidden_dim, 1), nn.Softplus()
				)
				self.normalizer.to(self.device)
			
			norm_actual = self.normalizer(all_c_p_actual).squeeze(-1)
			# B x (T + 1)
			norm_noise = self.normalizer(all_c_p_noise).squeeze(-1)
			# B x T'

			nce_obj = self.compute_obj_binary(
				inten_p_actual_both, inten_p_noise_noise, 
				norm_actual, norm_noise, 
				mask_tensor, all_mask_noise
			)

		else: 
			raise Exception(f"Unknown NCE method : {method}")
		"""
		count # of intensities that we computed for this objective
		"""
		inten_num = self.count_inten_num(
			type(noise).__name__, 
			mask_tensor, all_mask_noise, all_inten_num_noise, count_noise
		)
		return nce_obj, inten_num, new_noise_pack

	def count_inten_num(self, noise_name, 
		mask_tensor, all_mask_noise, all_inten_num_noise, count_noise): 
		"""
		count # of intensities that we computed for this objective
		"""
		inten_num = float(torch.sum(mask_tensor))
		# intensity under model p per actual event 
		inten_num += float(torch.sum(all_mask_noise))
		# intensity under model p per noise event 
		if noise_name == 'GNHP': 
			inten_num += float(torch.sum(mask_tensor)) * 1.0 * count_noise
			# intensity under noise dist q per actual event 
			inten_num += all_inten_num_noise * 1.0 * count_noise
			# all_inten_num_noise : # of intensities computed in order to get noise samples
			# including, e.g., # of intensities computed in thinning algorithm
			# this number already considers intensity under noise dist q per noise event 
			# so we shouldn't double-count it
		elif noise_name == 'GPP': 
			inten_num += 0.0 
			# it is super cheap to compute intensities in GPP 
			# we only count intensities that are expensive to compute 
			# e.g., neural intensities 
		else: 
			raise Exception(f"Unknown noise name : {noise_name}")
		return inten_num

	#@profile
	def compute_obj_frac(self, 
		inten_p_actual_both, inten_p_noise_noise, 
		inten_q_actual_both, inten_q_noise_noise, 
		mask_tensor, all_mask_noise, all_accept_prob_noise): 
		"""
		fractional version of NCE (fractional version of async) : 
		"""
		assert self.noise_type_num == 1, \
			"more than 1 noise type for frac (i.e. frac of async)?!"
		"""
		largely copy _async : may need to revise or merge
		"""
		nume_actual = inten_p_actual_both[:, :, 0] 
		deno_actual = inten_p_actual_both[:, :, 0] + \
			inten_q_actual_both[:, :, 0] * (self.noise_process_num * 1.0)
		log_posterior_actual = \
			torch.log(nume_actual + self.eps) - torch.log(deno_actual + self.eps)
		log_posterior_actual = log_posterior_actual * mask_tensor
		# B x (T + 1)
		deno_noise = inten_p_noise_noise.sum(-1) + \
			inten_q_noise_noise.sum(-1) * (self.noise_process_num * 1.0)
		log_posterior_noise = - torch.log(deno_noise + self.eps)
		"""
		NOTE : USE ACCEPT PROB that fall in [0, 1]
		such that each proposed noise time is counted with some probability
		this increases sample efficiency
		"""
		log_posterior_noise = log_posterior_noise * all_mask_noise * all_accept_prob_noise
		# B x T' 
		nce_obj = torch.sum(log_posterior_actual) + torch.sum(log_posterior_noise)
		return nce_obj

	#@profile
	def compute_obj_async(self, 
		inten_p_actual_both, inten_p_noise_noise, 
		inten_q_actual_both, inten_q_noise_noise, 
		mask_tensor, all_mask_noise): 
		"""
		async version of NCE : only one not-NULL noise types per time
		"""
		assert self.noise_type_num == 1, "more than 1 noise type for async?!"
		"""
		compute NCE objective 
		"""
		nume_actual = inten_p_actual_both[:, :, 0]
		deno_actual = inten_p_actual_both[:, :, 0] + \
			inten_q_actual_both[:, :, 0] * (self.noise_process_num * 1.0)
		log_posterior_actual = \
			torch.log(nume_actual + self.eps) - torch.log(deno_actual + self.eps)
		log_posterior_actual = log_posterior_actual * mask_tensor
		# B x (T + 1)
		deno_noise = inten_p_noise_noise.sum(-1) + \
			inten_q_noise_noise.sum(-1) * (self.noise_process_num * 1.0)
		log_posterior_noise = - torch.log(deno_noise + self.eps)
		log_posterior_noise = log_posterior_noise * all_mask_noise
		# B x T' 
		nce_obj = torch.sum(log_posterior_actual) + torch.sum(log_posterior_noise)
		return nce_obj

	
	def compute_obj_binary(self, 
		inten_p_actual_both, inten_p_noise_noise, 
		norm_actual, norm_noise, 
		mask_tensor, all_mask_noise): 
		"""
		binary-classification NCE : real next event or noise next event?
		"""
		"""
		neural normalizer or constant 1? 
		Guo et al used both
		I tried both but constant = 1.0 makes better
		if one wants to use the neural normalizer
		just simply comment out the 2 lines below
		"""
		norm_actual = 1.0 # comment out to use neural
		norm_noise = 1.0 # comment out to use neural

		nume_actual = inten_p_actual_both[:, :, 0] * norm_actual
		deno_actual = inten_p_actual_both[:, :, 0] * norm_actual + 1.0 
		# reparametrization trick proposed by Mnih & Teh 2012
		# used by Guo et al 2018
		log_posterior_actual = \
			torch.log(nume_actual + self.eps) - torch.log(deno_actual + self.eps)
		log_posterior_actual = log_posterior_actual * mask_tensor
		# B x (T + 1)
		deno_noise = inten_p_noise_noise.sum(-1) * norm_noise + 1.0 
		log_posterior_noise = - torch.log(deno_noise + self.eps)
		log_posterior_noise = log_posterior_noise * all_mask_noise
		# B x T' 
		nce_obj = torch.sum(log_posterior_actual) + torch.sum(log_posterior_noise)
		return nce_obj

	"""
	ISSUE : derivation of this objective seems wrong to me 
	so currently it is not being used 
	"""

	def compute_obj_sync(self, 
		inten_p_actual_both, inten_p_noise_noise, 
		inten_q_actual_both, inten_q_noise_noise, 
		mask_tensor, all_mask_noise): 
		"""
		sync version of NCE : multiple not-NULL noise types per time
		"""
		assert self.noise_process_num == 1, "more then 1 process for sync?!"
		"""
		compute NCE objective 
		"""
		ratio_actual_both = \
			inten_p_actual_both / (inten_q_actual_both + self.eps)
		# B x (T + 1) x (1 + noise_type_num)
		ratio_noise_noise = \
			inten_p_noise_noise / (inten_q_noise_noise + self.eps)
		# B x T' x noise_type_num
		nume_actual = ratio_actual_both[:, :, 0]
		deno_actual = ratio_actual_both.sum(-1)
		log_posterior_actual = \
			torch.log(nume_actual + self.eps) - torch.log(deno_actual + self.eps)
		log_posterior_actual = log_posterior_actual * mask_tensor
		# B x (T + 1)
		deno_noise = ratio_noise_noise.sum(-1) + 1.0
		log_posterior_noise = - torch.log(deno_noise + self.eps)
		log_posterior_noise = log_posterior_noise * all_mask_noise
		# B x T'
		nce_obj = torch.sum(log_posterior_actual) + torch.sum(log_posterior_noise)
		return nce_obj