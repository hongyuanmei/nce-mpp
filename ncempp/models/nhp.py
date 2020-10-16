import os 
import psutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nce_point_process.models.cont_time_cell import CTLSTMCell
from nce_point_process.models.layers import CF, LinearEmbedding
from nce_point_process.models.utils import draw_mc_samples, sample_noise_types

class GNHP(nn.Module): 

	"""
	generalized NHP : 
	a NHP (Mei & Eisner 2017) + a c->k distribution on top 
	when C = K and c->k is one-to-one correspondence 
	generalized NHP falls back to NHP 
	"""
	def __init__(self, *, 
		coarse_num, event_num, fine_to_coarse, 
		hidden_dim=8, beta=1.0, noise_mode='multinomial', device=None): 
		super(GNHP, self).__init__()
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
		self.hidden_dim = hidden_dim
		self.beta = beta 

		self.idx_BOS = self.event_num
		self.idx_EOS = self.event_num + 1
		self.idx_PAD = self.event_num + 2

		self.in_emb = nn.Embedding(
			self.event_num + 3, self.hidden_dim )
		self.rnn_cell = CTLSTMCell(
			self.hidden_dim, device=self.device )
		self.out_emb = LinearEmbedding( 
			num_embeddings = self.coarse_num, 
			embedding_dim = self.hidden_dim, 
			device=self.device 
		)
		self.coarse_to_fine = CF(
			coarse_num = coarse_num, event_num = event_num, 
			fine_to_coarse = fine_to_coarse,
			device=self.device
		)

		self.init_h = torch.zeros(
			size=[hidden_dim], dtype=torch.float32, device=self.device)
		self.init_c = torch.zeros(
			size=[hidden_dim], dtype=torch.float32, device=self.device)
		self.init_cb = torch.zeros(
			size=[hidden_dim], dtype=torch.float32, device=self.device)


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

	def get_cells_gates_states(self, event_tensor, dtime_tensor): 
		"""
		input 
			event_tensor [B x T+2] : tensor of event types 
			dtime_tensor [B x T+2] : tensor of dtimes
		return
			cells, gates, and states [B x T+1]
		"""
		batch_size, T_plus_2 = event_tensor.size()
		cell_t_i_minus = self.init_c.unsqueeze(0).expand(
			batch_size, self.hidden_dim)
		cell_bar_im1 = self.init_cb.unsqueeze(0).expand(
			batch_size, self.hidden_dim)
		hidden_t_i_minus = self.init_h.unsqueeze(0).expand(
			batch_size, self.hidden_dim)

		all_cell, all_cell_bar = [], []
		all_gate_output, all_gate_decay = [], []
		all_hidden = []
		all_hidden_after_update = []

		for i in range(T_plus_2 - 1):
			# only T+1 events update LSTM 
			# BOS, k1t1, ..., kItI
			emb_i = self.in_emb(event_tensor[:, i ])
			dtime_i = dtime_tensor[:, i + 1 ] # need to carefully check here

			cell_i, cell_bar_i, gate_decay_i, gate_output_i = self.rnn_cell(
				emb_i, hidden_t_i_minus, cell_t_i_minus, cell_bar_im1
			)
			_, hidden_t_i_plus = self.rnn_cell.decay(
				cell_i, cell_bar_i, gate_decay_i, gate_output_i,
				torch.zeros(dtime_i.size(), device=self.device)
			)
			cell_t_ip1_minus, hidden_t_ip1_minus = self.rnn_cell.decay(
				cell_i, cell_bar_i, gate_decay_i, gate_output_i,
				dtime_i
			)
			all_cell.append(cell_i)
			all_cell_bar.append(cell_bar_i)
			all_gate_decay.append(gate_decay_i)
			all_gate_output.append(gate_output_i)
			all_hidden.append(hidden_t_ip1_minus)
			all_hidden_after_update.append(hidden_t_i_plus)
			cell_t_i_minus = cell_t_ip1_minus
			cell_bar_im1 = cell_bar_i
			hidden_t_i_minus = hidden_t_ip1_minus
			
		# these tensors shape : batch_size, T+1, hidden_dim
		# cells and gates right after BOS, 1st event, ..., I-th event
		# hidden right before 1st event, ..., I-th event, End event (PAD)
		all_cell = torch.stack( all_cell, dim=1)
		all_cell_bar = torch.stack( all_cell_bar, dim=1)
		all_gate_decay = torch.stack( all_gate_decay, dim=1)
		all_gate_output = torch.stack( all_gate_output, dim=1)
		all_hidden = torch.stack( all_hidden, dim=1 )
		all_hidden_after_update = torch.stack( all_hidden_after_update, dim=1)
		#assert all_gate_decay.data.cpu().numpy().all() >= 0.0, "Decay > 0"
		return all_cell, all_cell_bar, all_gate_decay, all_gate_output, \
		all_hidden, all_hidden_after_update

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

	def get_intensities_given_types(self, 
		all_cell, all_cell_bar, all_gate_decay, all_gate_output, 
		event_tensor, dtime_tensor): 
		"""
		get intensities given types 
		e.g., for MLE, compute intensities for the sum term in log-likelihood
		e.g., for NCE, compute intensities for times drawn from p or q 
		note that these cells, gates, event types and dtimes 
		may be either at the oberservation times when actual events happened 
		or at the sampled times 
		e.g., for MLE, they are sampled times for Monte-Carlo approx 
		e.g., for NCE, they are sampled times from noise dist q 
		"""
		coarse_types = self.coarse_to_fine.get_coarse_for_given_fine(event_tensor)
		# batch_size x T+1 x N (N can be 1)
		coarse_embs = self.out_emb.get_embeddings_given_types(coarse_types)
		# batch_size x T+1 x N x D
		_, all_h_t = self.rnn_cell.decay(
			all_cell, all_cell_bar, all_gate_decay, all_gate_output, dtime_tensor)
		# batch_size x T+1 x D
		coarse_intensities = F.softplus(
			torch.sum(
				coarse_embs * all_h_t.unsqueeze(-2), dim=-1
			), beta=self.beta
		) # batch_size x T+1 x N (N can be 1)
		fine_intensities = self.coarse_to_fine.get_fine_probs_given_types(
			coarse_intensities, coarse_types, event_tensor)
		# batch_size x T+1 x N (N can be 1)
		return fine_intensities

	def get_intensities_all_coarse_types(self, 
		all_cell, all_cell_bar, all_gate_decay, all_gate_output, 
		dtime_tensor): 
		"""
		get intensities for all coarse types 
		e.g., compute total intensity in thinning algorithm
		total_coarse_intensity == total_fine_intensity
		NOTE : although comments use 3D as example 
		the coce is actually robust to any # of dimensions
		"""
		coarse_embs = self.out_emb.get_embeddings_all_types()
		# C x D 
		_, all_h_t = self.rnn_cell.decay(
			all_cell, all_cell_bar, all_gate_decay, all_gate_output, dtime_tensor)
		# batch_size x T+1 x D 
		coarse_intensities = F.softplus(
			torch.matmul(all_h_t, coarse_embs.t()), beta=self.beta) 
		# batch_size x T+1 x C 
		return coarse_intensities

	def get_intensities_all_fine_types(self, 
		all_cell, all_cell_bar, all_gate_decay, all_gate_output, 
		dtime_tensor): 
		"""
		get intensities for all types 
		e.g., compute all intensities for integral term in log-likelihood
		note that these cells, gates, and dtimes 
		may be either at the oberservation times when actual events happened 
		or at the sampled times 
		e.g., for MLE, they are sampled times for Monte-Carlo approx 
		e.g., for NCE, they are sampled times from noise dist q 
		"""
		coarse_intensities = self.get_intensities_all_coarse_types( 
			all_cell, all_cell_bar, all_gate_decay, all_gate_output, dtime_tensor )
		# return : batch_size x T+1 x K
		return self.coarse_to_fine.get_fine_probs_all_types(coarse_intensities)

	def get_mc_samples(self, 
		all_cell, all_cell_bar, all_gate_decay, all_gate_output, dtime_tensor, 
		mc_sample_num_tensor, duration_tensor, mask_tensor): 
		"""
		for MLE, sample time points for each interval 
		for Monte-Carlo approximation of the integral in log-likelihood
		"""
		"""
		input 
			mc_sample_num_tensor [B] : # of MC samples per sequence 
			duration_tensor [B] : duration per sequence 
			mask_tensor [B x T+1] : 1.0/0.0 mask of each token of each sequence
		"""
		all_c_inter, all_cb_inter = [], []
		all_d_inter, all_o_inter = [], []
		all_dtime_inter = []
		all_mask_inter = []

		batch_size, T_plus_1, hidden_dim = all_cell.size()
		"""
		draw MC time samles 
		TODO : use randomized rounding when rho * I is not integer !!!
		"""
		mc_max = torch.max(mc_sample_num_tensor)
		mc_max = mc_max if mc_max > 1 else 1
		u = torch.ones(size=[batch_size, mc_max], dtype=torch.float32, device=self.device)
		u, _ = torch.sort(u.uniform_(0.0, 1.0)) # batch_size x mc_max 
		sampled_time = u * duration_tensor.unsqueeze(-1)

		last_time = torch.zeros(size=[batch_size], dtype=torch.float32, device=self.device)

		for i in range(T_plus_1): 
			"""
			starting from the 1st (non-BOS) event 
			find mc samples in this interval
			"""
			dtime_i = dtime_tensor[:, i] # batch_size 
			curr_time = last_time + dtime_i 
			fallin = (sampled_time > last_time.unsqueeze(-1)) \
				& (sampled_time <= curr_time.unsqueeze(-1))
			# 0/1 unit 8 : batch_size x mc_max
			"""
			find the min rectangle covering all 1 
			"""
			fallin_idx = fallin.sum(0) > 0.5

			mask_inter = fallin[:, fallin_idx].float() 
			mask_inter *= mask_tensor[:, i].unsqueeze(1)

			chosen_time = sampled_time[:, fallin_idx]
			sampled_dt = chosen_time - last_time.unsqueeze(-1)
			"""
			chosen time may < past time : they are chosen cuz that col is chosen 
			and they will eventually be masked out in the end 
			"""
			sampled_dt[sampled_dt < 0.0] = 0.0
			# batch_size x S (S may be 0)
			_, S = sampled_dt.size()

			c_inter = all_cell[:, i, :].unsqueeze(1).expand(batch_size, S, hidden_dim)
			cb_inter = all_cell_bar[:, i, :].unsqueeze(1).expand(batch_size, S, hidden_dim)
			d_inter = all_gate_decay[:, i, :].unsqueeze(1).expand(batch_size, S, hidden_dim)
			o_inter = all_gate_output[:, i, :].unsqueeze(1).expand(batch_size, S, hidden_dim)

			last_time = curr_time

			all_c_inter.append(c_inter)
			all_cb_inter.append(cb_inter)
			all_d_inter.append(d_inter)
			all_o_inter.append(o_inter)

			all_dtime_inter.append(sampled_dt)
			all_mask_inter.append(mask_inter)

		all_c_inter = torch.cat(all_c_inter, dim=1)
		all_cb_inter = torch.cat(all_cb_inter, dim=1)
		all_d_inter = torch.cat(all_d_inter, dim=1)
		all_o_inter = torch.cat(all_o_inter, dim=1)
		all_dtime_inter = torch.cat(all_dtime_inter, dim=1)
		all_mask_inter = torch.cat(all_mask_inter, dim=1)

		return all_c_inter, all_cb_inter, all_d_inter, all_o_inter, \
			all_dtime_inter, all_mask_inter	


	def get_interpolated_cells(self, 
		all_cell, all_cell_bar, all_gate_decay, all_gate_output, all_S_inter): 
		"""
		given a tensor of cells & gates 
		interpolate the cells & gates between them 
		# of interpolated points is in S
		"""
		all_c_inter, all_cb_inter = [], []
		all_d_inter, all_o_inter = [], []
		batch_size, T_plus_1, D = all_cell.size()
		for i in range(T_plus_1): 
			c_inter = all_cell[:,i,:].unsqueeze(1).expand(
				batch_size, all_S_inter[i], D)
			cb_inter = all_cell_bar[:,i,:].unsqueeze(1).expand(
				batch_size, all_S_inter[i], D)
			d_inter = all_gate_decay[:,i,:].unsqueeze(1).expand(
				batch_size, all_S_inter[i], D)
			o_inter = all_gate_output[:,i,:].unsqueeze(1).expand(
				batch_size, all_S_inter[i], D)
			all_c_inter.append(c_inter)
			all_cb_inter.append(cb_inter)
			all_d_inter.append(d_inter)
			all_o_inter.append(o_inter)
		all_c_inter = torch.cat(all_c_inter, dim=1)
		all_cb_inter = torch.cat(all_cb_inter, dim=1)
		all_d_inter = torch.cat(all_d_inter, dim=1)
		all_o_inter = torch.cat(all_o_inter, dim=1)
		return all_c_inter, all_cb_inter, all_d_inter, all_o_inter


	def get_noise_samples(self, method, 
		event_tensor, dtime_tensor, target_tensor, mask_tensor, duration_tensor, 
		noise_process_num, noise_type_num, over): 
		"""
		draw noise samples from q and evaluate these samples
		"""
		all_c_q_actual, all_cb_q_actual, all_d_q_actual, all_o_q_actual, _, _ = \
		self.get_cells_gates_states(event_tensor, dtime_tensor)
		# B x T+1 x D 
		return self.get_noise_samples_given_states(
			method, all_c_q_actual, all_cb_q_actual, all_d_q_actual, all_o_q_actual, 
			target_tensor, mask_tensor, dtime_tensor[:, 1:], duration_tensor, 
			noise_process_num, noise_type_num, over
		)

	#@profile
	def get_noise_samples_given_states(self, nce_method, 
		all_cell, all_cell_bar, all_gate_decay, all_gate_output, 
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

		_, T_plus_1, _ = all_cell.size()
		for i in range(T_plus_1): 
			"""
			draw noise times and noise types 
			"""
			type_both_noise, dtime_noise, mask_noise, \
			fine_inten_actual_both, fine_inten_noise_noise, \
			accept_prob_noise, S, inten_num_noise = \
			self.draw_noise_samples_per_interval(
				nce_method, 
				all_cell[:, i, :], all_cell_bar[:, i, :], 
				all_gate_decay[:, i, :], all_gate_output[:, i, :], 
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
		c, cb, d, o, target, dtime, M1, M2, over, type_mask=None): 
		"""
		input 
			nce_method [str] : nce_frac or nce_async or nce_sync
			c [B x D] : c_i in NHP
			cb [B x D] : \bar{c}_i in NHP
			d [B x D] : d_i in NHP, how fast c(t) goes from c_i to \bar{c}_i
			o [B x D] :  o_i in NHP
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
		NOTE : try best to speed it up---AS FAST AS IT CAN BE
		"""
		"""
		choose (over-)sampling rate for thinning algorithm
		why we construct (upper bound) sample rate this way ? 
		well, in principle, we should find an upper bound intensity
		so that thinning algorithm is exact 
		however, for a misspecified model, this may often be bad: 
		e.g., if it is too high, we got too many noise times which is expensive 
		fortunately, this sampling is used for training, not testing! 
		that being said, we can leverage the training data!!!
		with constructed 'sample_rate', 
		there will be 'over' # of tokens in expectation!!!
		when sample_rate is actually an upper bound, algorithm is exact 
		when it is not, all proposed times are accepted
		this acts as a correction to a mis-specified noise model
		such that we can always get some noise times :-)
		"""
		"""
		NOTE : if we ever run into super-long run-time or out-of-memory
		check if any / ends up in large # to slow things down!!!
		e.g., / 0.0 will end up a very large number 
		if this number is used as a dimension size, then we got HUGE overhead
		it may also have other bad effect 
		e.g., dtime too small, sample rate too high 
		for given sample_num_per_seq, it doesn't cover the interval enough, etc 
		so we need to carefully revisit this part...
		"""
		sample_rate = over * 1.0 / (dtime + self.eps)
		"""
		compute # of noise samples per interval per seq 
		"""
		"""
		thinning algorithm to sample # of events (1 or 0)
		at any time t given the (actual) history up to time t 
		equivalent to having M noise processes in parallel 
		we set only one process with rate = rate x M 
		equivalent to union of samples drawn from M processes, but less overhead
		"""
		sample_num_per_seq = int(over * M1 + self.eps)
		sample_num_per_seq = 1 if sample_num_per_seq < 1 else sample_num_per_seq
		sample_num_max = sample_num_per_seq
		batch_size, D = c.size()
		"""
		prepare tensors to host samples
		"""
		Exp_numbers = torch.empty(
			size=[batch_size, sample_num_max], dtype=torch.float32, device=self.device )
		Unif_numbers = torch.empty(
			size=[batch_size, sample_num_max], dtype=torch.float32, device=self.device )
		"""
		rejection sampling for # of noise events at each t 
		"""
		Exp_numbers.exponential_(1.0)
		sampled_dt = Exp_numbers / (sample_rate * M1).unsqueeze(-1)
		sampled_dt = sampled_dt.cumsum(dim=-1) # batch_size x sample_num_max
		"""
		sample noise types for noise times + actual time 
		"""
		sampled_and_actual_dt = torch.cat([sampled_dt, dtime.unsqueeze(-1)], dim=-1)
		# batch_size x (sample_num_max + 1)
		"""
		compute intensities at noise AND actual time points 
		"""
		c_both = c.unsqueeze(1).expand(batch_size, sample_num_max + 1, D)
		cb_both = cb.unsqueeze(1).expand(batch_size, sample_num_max + 1, D)
		d_both = d.unsqueeze(1).expand(batch_size, sample_num_max + 1, D)
		o_both = o.unsqueeze(1).expand(batch_size, sample_num_max + 1, D)
		# batch_size x (sample_num_max + 1) x D
		coarse_inten_both = self.get_intensities_all_coarse_types(
			c_both, cb_both, d_both, o_both, sampled_and_actual_dt 
		) # batch_size x (sample_num_max + 1) x coarse_num
		total_coarse_inten_both = coarse_inten_both.sum(-1) 
		# batch_size x (sample_num_max + 1)
		"""
		only collect time points that not exceed dtime
		"""
		to_collect = sampled_dt < dtime.unsqueeze(-1)
		"""
		count # of intensities to be computed for sampling noise times and types
		this MUST be done before to_collect is used and pruned (cuz of rejection)
		BUT we shouldn't sum them here : some of them need to be properly masked out
		"""
		inten_num_noise = to_collect.float() * self.coarse_num
		# batch_size x sample_num_max
		# total # of intensities per proposed noise sample (for thinning)
		"""
		thinning algorithm to get noise times and intensities at noise times
		"""
		dtime_noise, mask_noise, accept_prob_noise, coarse_inten_noise, S = self.thinning(
			Unif_numbers, sample_rate, total_coarse_inten_both[:, :-1], 
			sampled_dt, to_collect, coarse_inten_both[:, :-1, :], 
			batch_size, sample_num_max, fractional
		)
		"""
		sample event types at noise times and actual time 
		"""
		coarse_inten_both = torch.cat(
			[coarse_inten_noise, coarse_inten_both[:, -1, :].unsqueeze(1)], dim=1
		) # batch_size x (S + 1) x C # discard useless (sample_num_max - S) samples
		fine_inten_both = self.coarse_to_fine.get_fine_probs_all_types(
			coarse_inten_both )
		# batch_size x (S + 1) x K 

		if type_mask is not None: 
			fine_inten_both = fine_inten_both * type_mask

		type_both_noise = sample_noise_types(
			fine_inten_both, batch_size, S + 1, M2, 
			self.event_num, self.noise_mode, self.device )
		# batch_size x (S + 1) x M2
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

	#@profile
	def thinning(self, 
		Unif_numbers, sample_rate, total_coarse_inten_noise, 
		sampled_dt, to_collect, coarse_inten_noise, batch_size, sample_num_max, fractional): 
		"""
		rejection sampling (only for proposed noise times)
		input 
			fractional [boolean] : 
				true : keep all proposed noise times and their accept probs 
				false : only keep the (stochastically-)accepted noise times 
		"""
		accept_prob = total_coarse_inten_noise / (sample_rate.unsqueeze(-1) + self.eps)
		accept_prob[accept_prob > 1.0] = 1.0 # scale prob back to (0, 1)
		Unif_numbers.uniform_(0.0, 1.0)
		if fractional: 
			"""
			NOTE : we may adaptively change threshold in the future
			"""
			threshold = 0.01
			"""
			when accept prob >= threshold, we keep it 
			when accept prob < threshold, we stochastically accept it 
			"""
			id1 = accept_prob < threshold # accep prob too small
			id2 = Unif_numbers < accept_prob # accept some noise times
			to_collect[id1 * (~id2)] = 0
			"""
			adjust accept prob to 1 for id2 
			because accept prob will be used to scale log posterior 
			and samples indexed by id2 are not fractional
			"""
			accept_prob[id1 * id2] = 1.0
			"""
			in fractional thinning, quite most will be collected
			so there won't be many zeros to discard 
			so we don't find those zeros and save time
			"""
			dtime_noise = sampled_dt
			mask_noise = to_collect.float()
			accept_prob_noise = accept_prob
			S = sample_num_max
			#coarse_inten_noise = coarse_inten_noise
		else: 
			to_collect[Unif_numbers > accept_prob] = 0
			"""
			for efficiency (when these noise samples are actually used), 
			for each column, we check if all rows are 0
			if yes, we discard this column (all seqs in a batch)
			if no, we keep this column (all seqs in a batch)
			"""
			cnt_each_col = to_collect.sum(dim=0, keepdim=True)
			keep_col = cnt_each_col > 0 
			if keep_col.sum() < 1: 
				# nothing to collect, we force an accepted time
				# this is why over=1 may be good enough : 
				# when we don't accept, we force one!!!
				# this acts as a correction to a mis-specified model
				keep_col[:, 0] = 1
			keep_col = keep_col.expand(batch_size, sample_num_max)
			"""
			collect sample times, coarse intensities and construct mask 
			"""
			dtime_noise = sampled_dt[keep_col].view(batch_size, -1)
			mask_noise = to_collect[keep_col].view(batch_size, -1).float()
			accept_prob_noise = accept_prob[keep_col].view(batch_size, -1)
			# batch_size x S
			_, S = dtime_noise.size()
			coarse_inten_noise = coarse_inten_noise[keep_col, :].view(
				batch_size, S, self.coarse_num)
		return dtime_noise, mask_noise, accept_prob_noise, coarse_inten_noise, S


	"""
	FOR PREDICTING FUTURE TIME & TYPE 
	"""

	def get_next_events_given_states(self, 
		all_cell, all_cell_bar, all_gate_decay, all_gate_output, 
		dtime_tensor, over): 
		"""
		predict next event by aggregating many draws
		"""
		
		_, T_plus_1, _ = all_cell.size()

		all_dtime_predict = []
		all_type_predict = []

		for i in range(T_plus_1): 
			"""
			draw times and types (given times)
			"""
			dtime_predict, type_predict = \
			self.mbr_time_type(
				all_cell[:, i, :], all_cell_bar[:, i, :], 
				all_gate_decay[:, i, :], all_gate_output[:, i, :], 
				dtime_tensor[:, i], over
			)
			all_dtime_predict += [dtime_predict]
			all_type_predict += [type_predict]

		all_dtime_predict = torch.stack(all_dtime_predict, dim=1)
		all_type_predict = torch.stack(all_type_predict, dim=1)

		return all_dtime_predict, all_type_predict

	
	def mbr_time_type(self, c, cb, d, o, dtime, over): 
		"""
		numerical approximation to MBR time and type by importance sampling
		similar to draw_noise_samples_per_interval
		not couple code to avoid mess up
		"""
		sample_rate = over * 1.0 / (dtime + self.eps) # B 
		batch_size, D = c.size()
		m = 10 # may make it hyper-param
		"""
		sample from q
		"""
		Exp_numbers = torch.empty(
			size=[batch_size, m], dtype=torch.float32, device=self.device
		)
		Exp_numbers.exponential_(1.0)
		sampled_dt = Exp_numbers / (sample_rate + self.eps).unsqueeze(-1)
		# B x M 
		"""
		compute q
		"""
		q_prob = sample_rate.unsqueeze(-1) * torch.exp(
			- sample_rate.unsqueeze(-1) * sampled_dt )
		# B x M 
		"""
		compute p
		"""
		sampled_and_actual_dt = torch.cat([sampled_dt, dtime.unsqueeze(-1)], dim=-1)
		c_both = c.unsqueeze(1).expand(batch_size, m + 1, D)
		cb_both = cb.unsqueeze(1).expand(batch_size, m + 1, D)
		d_both = d.unsqueeze(1).expand(batch_size, m + 1, D)
		o_both = o.unsqueeze(1).expand(batch_size, m + 1, D)
		# batch_size x (m + 1) x D
		coarse_inten_both = self.get_intensities_all_coarse_types(
			c_both, cb_both, d_both, o_both, sampled_and_actual_dt 
		) 
		total_coarse_inten_both = coarse_inten_both.sum(-1) 
		# batch_size x (m + 1)
		total_coarse_inten_sample = total_coarse_inten_both[:, :-1]
		avg_total_inten = torch.mean(total_coarse_inten_sample, dim=-1, keepdim=True)
		p_prob = total_coarse_inten_sample * torch.exp(
			- avg_total_inten * sampled_dt )
		"""
		compute weights and predcited time
		"""
		weights = p_prob / q_prob + self.eps
		weights /= torch.sum(weights, dim=-1, keepdim=True)
		time_predict = torch.sum( sampled_dt * weights, dim=-1 ) 
		"""
		compute predicted type
		"""
		fine_inten_actual = self.coarse_to_fine.get_fine_probs_all_types(
			coarse_inten_both )[:, -1, :] # B x K 
		type_predict = torch.argmax(fine_inten_actual, dim=-1) # B 

		return time_predict, type_predict
	


	"""
	FOR DRAWING SEQUENCES
	"""

	def draw_seq(self, num): 
		rst = []
		# use BOS to init
		k = self.idx_BOS
		dt = 0.0 
		c = self.init_c.unsqueeze(0).clone() # 1 x D
		cb = c.clone() # 1 x D
		d = c.clone()
		o = c.clone().fill_(1.0)
		for i in range(num): 
			# update using last event
			c, cb, d, o = self.update(k, dt, c, cb, d, o)
			# then draw the next event
			dt, k = self.draw_next(c, cb, d, o)
			rst.append((dt, k))
		return rst
	
	def update(self, k, dt, c, cb, d, o): 
		"""
		k : event type (idx) (maybe BOS)
		dt : event dtime since last event (or init)
		c, cb, d, o : gates after LAST update (or init)
		"""
		event_tensor = torch.zeros(
			size=[1], dtype=torch.long, device=self.device).fill_(k)
		dtime_i = torch.zeros(
			size=[1], dtype=torch.float32, device=self.device).fill_(dt)
		emb_i = self.in_emb(event_tensor) # 1 x D

		c_t_minus, h_t_minus = self.rnn_cell.decay(c, cb, d, o, dtime_i)
		c, cb, d, o = self.rnn_cell(emb_i, h_t_minus, c_t_minus, cb)
		return c, cb, d, o

	def draw_next(self, c, cb, d, o): 
		"""
		draw next event dtime and type using thinning algorithm
		NOTE : different from the thinning method in this calss 
		that thinning : draw noise samples given h(t)
		this thinning : draw next event given h(t_i)
		there is similar code 
		but we decide to separate them to not mess up each other
		"""
		over = 10.0
		N = 500 
		"""
		find upper bound (a conservative estimate)
		"""
		coarse_inten = self.get_intensities_all_coarse_types(
			c, cb, d, o, 
			torch.zeros(size=[1], dtype=torch.float32, device=self.device)
		) # 1 x C
		total_coarse_inten = coarse_inten.sum() # 0
		sample_rate = total_coarse_inten * over # 0
		"""
		rejection sampling for next event dtime and type
		"""
		Exp_numbers = torch.empty(
			size=[1, N], dtype=torch.float32, device=self.device )
		Unif_numbers = torch.empty(
			size=[1, N], dtype=torch.float32, device=self.device )
		Exp_numbers.exponential_(1.0)
		sampled_dt = Exp_numbers / sample_rate
		sampled_dt = sampled_dt.cumsum(dim=-1) # 1 x N
		"""
		compute intensities at sampled times
		"""
		D = c.size(-1) # hidden dimension
		c_exp = c.unsqueeze(1).expand(1, N, D)
		cb_exp = cb.unsqueeze(1).expand(1, N, D)
		d_exp = d.unsqueeze(1).expand(1, N, D)
		o_exp = o.unsqueeze(1).expand(1, N, D)
		coarse_inten = self.get_intensities_all_coarse_types(
			c_exp, cb_exp, d_exp, o_exp, sampled_dt 
		) # 1 x N x C
		total_coarse_inten = coarse_inten.sum(-1) # 1 x N 
		accept_prob = total_coarse_inten / (sample_rate + self.eps)
		# 1 x N
		Unif_numbers.uniform_(0.0, 1.0)
		"""
		randomly accept
		"""
		accept_idx = Unif_numbers <= accept_prob # accept : 1 x ?
		accept_dt = sampled_dt[accept_idx] # ?
		#print()
		#print(accept_dt.size())
		accept_coarse_inten = coarse_inten[accept_idx, :] # ? x C
		#print(accept_coarse_inten.size())
		assert accept_idx.size(-1) > 0, "no accept?"
		dt, min_i = accept_dt.min(dim=-1) # 1 
		min_i = int(min_i.sum())
		dt = float(dt.sum())
		accept_coarse_inten = accept_coarse_inten[min_i, :] # C
		"""
		sample event type
		"""
		"""
		NOTE : most robust # of dimension is 3
		cuz that is for which coarse_to_fine is optimized
		"""
		accept_coarse_inten_exp = accept_coarse_inten.unsqueeze(0).unsqueeze(0)
		fine_inten = self.coarse_to_fine.get_fine_probs_all_types(
			accept_coarse_inten_exp ) # 1 x 1 x K
		#fine_inten = fine_inten.unsqueeze(0) # 1 x 1 x K
		"""
		continue 
		"""
		sampled_k = sample_noise_types(
			fine_inten, 1, 1, 1, 
			self.event_num, self.noise_mode, self.device
		) # 1 x 1 x 1 

		sampled_k = int(sampled_k.sum())
		return dt, sampled_k