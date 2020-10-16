import os 
import psutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nce_point_process.models.layers import CF, LinearEmbedding
from nce_point_process.models.utils import sample_noise_types

class GPP(nn.Module): 

	"""
	generalized Poisson Process : 
	a Poisson process + a c->k distribution on top 
	when C = K and c->k is one-to-one correspondence 
	generalized PP falls back to PP 
	"""
	def __init__(self, *, 
		coarse_num, event_num, fine_to_coarse, 
		beta=1.0, noise_mode='multinomial', device=None): 
		super(GPP, self).__init__()
		"""
		input 
			coarse_num : # of coarse event types 
			event_num : # of fine event types  
			fine_to_coarse : dictionary k -> c
			noise_mode : how to sample noise types when used as noise dist q
		"""

		device = device or 'cpu'
		self.device = torch.device(device)
		self.eps = np.finfo(float).eps
		self.max = np.finfo(float).max 
		self.noise_mode = noise_mode

		self.coarse_num = coarse_num # # of coarse event types 
		self.event_num = event_num # # of fine event types 
		self.beta = beta

		self.idx_BOS = self.event_num
		self.idx_EOS = self.event_num + 1
		self.idx_PAD = self.event_num + 2

		self.out_emb = LinearEmbedding( 
			num_embeddings = self.coarse_num, 
			embedding_dim = 1, 
			device=self.device 
		)
		self.coarse_to_fine = CF(
			coarse_num = coarse_num, event_num = event_num, 
			fine_to_coarse = fine_to_coarse,
			device=self.device
		)

	def cuda(self, device=None):
		device = device or 'cuda:0'
		self.device = torch.device(device)
		assert self.device.type == 'cuda'
		super().cuda(self.device)

	def cpu(self):
		self.device = torch.device('cpu')
		super().cuda(self.device)

	def get_inten_num(self): 
		"""
		return # of intensities to be computed in this model
		"""
		return self.coarse_num

	def get_target(self, event_tensor): 
		"""
		make target variables and masks 
		i.e., set >= event_num to 0, also mask them out 
		"""
		batch_size, T_plus_2 = event_tensor.size()
		mask = torch.ones((batch_size, T_plus_2-1), dtype=torch.float32, device=self.device)
		target_data = event_tensor[:, 1:].clone().detach()
		mask[target_data >= self.event_num] = 0.0
		target_data[target_data >= self.event_num] = 0 # PAD to be 0
		return target_data, mask

	def get_intensities_given_types(self, event_tensor): 
		"""
		get intensities given types 
		e.g., for MLE, compute intensities for the sum term in log-likelihood
		e.g., for NCE, compute intensities for times drawn from p or q 
		note that Poisson intensities do NOT change with time
		"""
		coarse_types = self.coarse_to_fine.get_coarse_for_given_fine(event_tensor)
		# batch_size x T+1 x N (N can be 1)
		coarse_intensities_signed = self.out_emb.get_embeddings_given_types(coarse_types)
		# batch_size x (T+1) x N x 1
		coarse_intensities = F.softplus(
			coarse_intensities_signed, beta=self.beta
		).squeeze(-1) # batch_size x T+1 x N 
		fine_intensities = self.coarse_to_fine.get_fine_probs_given_types(
			coarse_intensities, coarse_types, event_tensor)
		# batch_size x T+1 x N 
		return fine_intensities

	def get_intensities_all_coarse_types(self): 
		"""
		get intensities for all coarse types 
		e.g., compute total intensity in thinning algorithm
		total_coarse_intensity == total_fine_intensity
		"""
		coarse_intensities_signed = self.out_emb.get_embeddings_all_types().squeeze(-1)
		# C x 1 -> C 
		coarse_intensities = F.softplus(
			coarse_intensities_signed, beta=self.beta) 
		# C
		return coarse_intensities

	def get_intensities_all_fine_types(self): 
		"""
		get intensities for all types 
		e.g., compute all intensities for integral term in log-likelihood
		"""
		coarse_intensities = self.get_intensities_all_coarse_types()
		# C 
		expand_coarse_intensities = coarse_intensities.unsqueeze(0).unsqueeze(0)
		# 1 x 1 x C
		# CF implementation assumes dimensions for speedup
		expand_fine_intensities = self.coarse_to_fine.get_fine_probs_all_types(
			expand_coarse_intensities )
		return expand_fine_intensities.squeeze(0).squeeze(0)
	
	def get_noise_samples(self, method, 
		event_tensor, dtime_tensor, target_tensor, mask_tensor, duration_tensor, 
		noise_process_num, noise_type_num, over): 
		"""
		draw noise samples from q and evaluate these samples
		NOTE : we keep the format and method names consistent with GNHP
		"""
		return self.get_noise_samples_given_states(
			method, 
			target_tensor, mask_tensor, dtime_tensor[:, 1:], duration_tensor, 
			noise_process_num, noise_type_num, over
		)


	#@profile
	def get_noise_samples_given_states(self, nce_method, 
		target_tensor, mask_tensor, 
		dtime_tensor, duration_tensor, 
		noise_process_num, noise_type_num, over): 
		"""
		for NCE method, sample noise times and event types 
		when object of this class is used as a noise distribution q 
		"""
		"""
		input 
			nce_method : nce_frac or nce_async or nce_sync
			noise_process_num [0] : # noise processes in parallel 
			noise_type_num [0] : # noise types per noise and actual time
			duration_tensor [B] : duration of each sequence
			mask_tensor [B x T+1] : 1.0/0.0 mask of each token of each seq
		"""
		"""
		_time : _noise == at noise times 
		_time_type : _actual_noise == at actual times, of noise types 
		_both : at both times (or of both actual and noise types, subj. to context)
		"""
		all_type_actual_noise = []
		all_type_noise_noise = []
		all_dtime_noise = []
		all_mask_noise = []
		inten_q_actual_both = [] # these intensities no need to compute again outside
		inten_q_noise_noise = []
		all_accept_prob_noise = []
		all_S = []
		all_inten_num_noise = 0 
		# count # of intensities computed in this sampling algorithm

		_, T_plus_1 = target_tensor.size()
		for i in range(T_plus_1): 
			"""
			draw noise times and noise types 
			"""
			type_both_noise, dtime_noise, mask_noise, \
			fine_inten_actual_both, fine_inten_noise_noise, \
			accept_prob_noise, S, inten_num_noise = \
			self.draw_noise_samples_per_interval(
				nce_method, 
				target_tensor[:, i], dtime_tensor[:, i], 
				noise_process_num, noise_type_num, over
			)
			"""
			type_both_noise : B x (S + 1) x noise_type_num # -1 th is actual time
			dtime_noise, mask_noise, accept_prob : B x S
			fine_inten_both : B x (S + 1) x K
			"""
			all_type_noise_noise.append(type_both_noise[:, :-1, :])
			all_type_actual_noise.append(type_both_noise[:, -1, :])
			all_dtime_noise.append(dtime_noise)
			inten_q_actual_both.append(fine_inten_actual_both)
			inten_q_noise_noise.append(fine_inten_noise_noise)
			all_accept_prob_noise.append(accept_prob_noise)
			all_S.append(S)
			"""
			mask_noise all set to 0.0 if this token is masked out 
			"""
			mask_noise = mask_noise * mask_tensor[:, i].unsqueeze(1)
			all_mask_noise.append(mask_noise)
			"""
			properly mask out fake # of inten and sum them
			"""
			inten_num_noise = inten_num_noise * mask_tensor[:, i].unsqueeze(1)
			inten_num_noise = float(torch.sum(inten_num_noise))
			all_inten_num_noise += inten_num_noise
		
		all_type_actual_noise = torch.stack(all_type_actual_noise, dim=1)
		all_type_noise_noise = torch.cat(all_type_noise_noise, dim=1)
		all_dtime_noise = torch.cat(all_dtime_noise, dim=1)
		all_mask_noise = torch.cat(all_mask_noise, dim=1)
		inten_q_actual_both = torch.stack(inten_q_actual_both, dim=1)
		inten_q_noise_noise = torch.cat(inten_q_noise_noise, dim=1)
		all_accept_prob_noise = torch.cat(all_accept_prob_noise, dim=1)

		return all_type_actual_noise, all_type_noise_noise, \
			all_dtime_noise, all_mask_noise, \
			inten_q_actual_both, inten_q_noise_noise, \
			all_accept_prob_noise, all_S, all_inten_num_noise

	#@profile
	def draw_noise_samples_per_interval(self, nce_method, 
		target, dtime, M1, M2, over, type_mask=None): 
		"""
		input 
			nce_method [str] : nce_frac or nce_async or nce_sync
			target [B] : actual event type 
			dtime [B] : actual time intervals between last and next actual event 
			M1 [0] : # of noise processes in parallel 
			M2 [0] : # of noise TYPEs per atual/noise time
			over [0] : over-sampling rate
			type_mask [B x K] : 1.0/0.0 mask for possible/valid event types of this interval
		"""
		"""
		NOTE : why type_mask? 
		for some complex model p, the valid event types may change over time 
		then we only want to propose the possible types 
		so we use the type_mask to maks out the impossible ones 
		why we only use it here but NOT in get_noise_samples? 
		because for such complex model p, 
		we would draw noise samples on-the-fly of training
		meaing that get_noise_samples won't be called in that case 
		so we leave it out of get_noise_samples 
		which also makes code run faster
		"""
		if nce_method == 'nce_frac': 
			"""
			fractional thinning algorithm : all noise times are used and reweighted by prob
			"""
			fractional = True 
		elif nce_method == 'nce_sync' or nce_method == 'nce_async': 
			"""
			thinning algorithm : noise times might be rejected 
			"""
			fractional = False
		elif nce_method == 'nce_binary': 
			"""
			binary-classification-NCE: noise times might be rejected
			"""
			fractional = False 
		else: 
			raise Exception(f"Unknown NCE method : {nce_method}")
		"""
		NOTE : drawing from Poisson process with constant rates 
		very easy and fast --- no thinning at all 
		"""
		coarse_inten = self.get_intensities_all_coarse_types() # C
		total_coarse_inten = coarse_inten.sum() # scalar 
		sample_rate = total_coarse_inten

		sample_num_per_seq = int(over * M1 + self.eps)
		sample_num_per_seq = 1 if sample_num_per_seq < 1 else sample_num_per_seq
		sample_num_max = sample_num_per_seq
		batch_size = target.size(0)
		Exp_numbers = torch.empty(
			size=[batch_size, sample_num_max], dtype=torch.float32, device=self.device )
		Exp_numbers.exponential_(1.0)
		sampled_dt = Exp_numbers / (sample_rate * M1).unsqueeze(-1)
		sampled_dt = sampled_dt.cumsum(dim=-1) # batch_size x sample_num_max
		"""
		since no thinning is used, we use all sampled times as noise times 
		meaning that we ``accept'' all of them so all accept probs are 1.0
		"""
		dtime_noise = sampled_dt
		to_collect = sampled_dt < dtime.unsqueeze(-1)
		mask_noise = to_collect.float()
		accept_prob_noise = torch.ones_like(dtime_noise)
		S = sample_num_max
		"""
		count # of intensities to be computed for sampling noise times and types
		for this model, it is 0 because nothing is neural!!!
		non-neural intensities are very cheap to compute!!!
		"""
		inten_num_noise = 0
		"""
		sample event types at noise times and actual time 
		"""
		"""
		NOTE : coarse intensities are same across sequences and intervals
		we can leverage this fact to speed up code 
		SLOW and naive : for each noise time, have a prob vector and sample types 
		FAST : only one prob vector, (over-)sample types and then reshape 
		"""
		expand_coarse_inten = coarse_inten.unsqueeze(0).unsqueeze(0)
		# 1 x 1 x C
		fine_inten = self.coarse_to_fine.get_fine_probs_all_types(expand_coarse_inten)
		# 1 x 1 x K 
		if type_mask is not None: 
			fine_inten = fine_inten * type_mask

		sampled_types = sample_noise_types(
			fine_inten, 1, 1, batch_size * (S + 1) * M2, 
			self.event_num, self.noise_mode, self.device
		) # 1 x 1 x ( batch_size x (S+1) x M2 )
		type_both_noise = sampled_types.view(batch_size, S + 1, M2)

		fine_inten_both = fine_inten.expand(batch_size, S + 1, self.event_num)

		"""
		gather intensities at actual and noise times 
		of actual and noise types 
		"""
		type_actual_both = torch.cat(
			[ target.unsqueeze(-1), type_both_noise[:, -1, :] ], dim=-1
		) # batch_size x (1 + M2)
		fine_inten_actual_both = torch.gather(
			fine_inten_both[:, -1, :], # batch_size x K 
			1, type_actual_both
		) # batch_size x (1 + M2)
		fine_inten_noise_noise = torch.gather(
			fine_inten_both[:, :-1, :], # batch_size x S x K
			2, type_both_noise[:, :-1, :]
		) # batch_size x S x M2
		return type_both_noise, dtime_noise, mask_noise, \
			fine_inten_actual_both, fine_inten_noise_noise, \
			accept_prob_noise, S, inten_num_noise

	"""
	NOTE : BELOW ARE USED FOR DATA-SAMPLING AS SAMPLER DIST PSTAR
	NOT REVISED YET 
	CODE FOR GNHP HERE FOR REFERENCE
	"""

	def draw_seq(self, num): 
		raise NotImplementedError

	def update(self): 
		raise NotImplementedError

	def draw_next(self): 
		raise NotImplementedError

	# def draw_seq(self, num): 
	# 	rst = []
	# 	# use BOS to init
	# 	k = self.idx_BOS
	# 	dt = 0.0 
	# 	c = self.init_c.unsqueeze(0).clone() # 1 x D
	# 	cb = c.clone() # 1 x D
	# 	d = c.clone()
	# 	o = c.clone().fill_(1.0)
	# 	for i in range(num): 
	# 		# update using last event
	# 		c, cb, d, o = self.update(k, dt, c, cb, d, o)
	# 		# then draw the next event
	# 		dt, k = self.draw_next(c, cb, d, o)
	# 		rst.append((dt, k))
	# 	return rst
	
	# def update(self, k, dt, c, cb, d, o): 
	# 	"""
	# 	k : event type (idx) (maybe BOS)
	# 	dt : event dtime since last event (or init)
	# 	c, cb, d, o : gates after LAST update (or init)
	# 	"""
	# 	event_tensor = torch.zeros(
	# 		size=[1], dtype=torch.long, device=self.device).fill_(k)
	# 	dtime_i = torch.zeros(
	# 		size=[1], dtype=torch.float32, device=self.device).fill_(dt)
	# 	emb_i = self.in_emb(event_tensor) # 1 x D

	# 	c_t_minus, h_t_minus = self.rnn_cell.decay(c, cb, d, o, dtime_i)
	# 	c, cb, d, o = self.rnn_cell(emb_i, h_t_minus, c_t_minus, cb)
	# 	return c, cb, d, o

	# def draw_next(self, c, cb, d, o): 
	# 	"""
	# 	draw next event dtime and type using thinning algorithm
	# 	NOTE : different from the thinning method in this calss 
	# 	that thinning : draw noise samples given h(t)
	# 	this thinning : draw next event given h(t_i)
	# 	there is similar code 
	# 	but we decide to separate them to not mess up each other
	# 	"""
	# 	over = 10.0
	# 	N = 500 
	# 	"""
	# 	find upper bound (a conservative estimate)
	# 	"""
	# 	coarse_inten = self.get_intensities_all_coarse_types(
	# 		c, cb, d, o, 
	# 		torch.zeros(size=[1], dtype=torch.float32, device=self.device)
	# 	) # 1 x C
	# 	total_coarse_inten = coarse_inten.sum() # 0
	# 	sample_rate = total_coarse_inten * over # 0
	# 	"""
	# 	rejection sampling for next event dtime and type
	# 	"""
	# 	Exp_numbers = torch.empty(
	# 		size=[1, N], dtype=torch.float32, device=self.device )
	# 	Unif_numbers = torch.empty(
	# 		size=[1, N], dtype=torch.float32, device=self.device )
	# 	Exp_numbers.exponential_(1.0)
	# 	sampled_dt = Exp_numbers / sample_rate
	# 	sampled_dt = sampled_dt.cumsum(dim=-1) # 1 x N
	# 	"""
	# 	compute intensities at sampled times
	# 	"""
	# 	D = c.size(-1) # hidden dimension
	# 	c_exp = c.unsqueeze(1).expand(1, N, D)
	# 	cb_exp = cb.unsqueeze(1).expand(1, N, D)
	# 	d_exp = d.unsqueeze(1).expand(1, N, D)
	# 	o_exp = o.unsqueeze(1).expand(1, N, D)
	# 	coarse_inten = self.get_intensities_all_coarse_types(
	# 		c_exp, cb_exp, d_exp, o_exp, sampled_dt 
	# 	) # 1 x N x C
	# 	total_coarse_inten = coarse_inten.sum(-1) # 1 x N 
	# 	accept_prob = total_coarse_inten / (sample_rate + self.eps)
	# 	# 1 x N
	# 	Unif_numbers.uniform_(0.0, 1.0)
	# 	"""
	# 	randomly accept
	# 	"""
	# 	accept_idx = Unif_numbers <= accept_prob # accept : 1 x ?
	# 	accept_dt = sampled_dt[accept_idx] # ?
	# 	#print()
	# 	#print(accept_dt.size())
	# 	accept_coarse_inten = coarse_inten[accept_idx, :] # ? x C
	# 	#print(accept_coarse_inten.size())
	# 	assert accept_idx.size(-1) > 0, "no accept?"
	# 	dt, min_i = accept_dt.min(dim=-1) # 1 
	# 	min_i = int(min_i.sum())
	# 	dt = float(dt.sum())
	# 	accept_coarse_inten = accept_coarse_inten[min_i, :] # C
	# 	"""
	# 	sample event type
	# 	"""
	# 	"""
	# 	NOTE : most robust # of dimension is 3
	# 	cuz that is for which coarse_to_fine is optimized
	# 	"""
	# 	accept_coarse_inten_exp = accept_coarse_inten.unsqueeze(0).unsqueeze(0)
	# 	fine_inten = self.coarse_to_fine.get_fine_probs_all_types(
	# 		accept_coarse_inten_exp ) # 1 x 1 x K
	# 	#fine_inten = fine_inten.unsqueeze(0) # 1 x 1 x K
	# 	"""
	# 	continue 
	# 	"""
	# 	sampled_k = self.sample_noise_types(fine_inten, 1, 1, 1) # 1 x 1 x 1 
	# 	sampled_k = int(sampled_k.sum())
	# 	return dt, sampled_k