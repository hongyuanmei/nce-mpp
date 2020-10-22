import os 
import psutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncempp.models.layers import CF, LinearEmbedding
from ncempp.models.utils import sample_noise_types

class GHP(nn.Module): 

	"""
	generalized Hawkes Process : 
	a Poisson process + a c->k distribution on top 
	when C = K and c->k is one-to-one correspondence 
	generalized PP falls back to PP 
	"""
	def __init__(self, *, 
		coarse_num, event_num, fine_to_coarse, 
		beta=1.0, noise_mode='multinomial', device=None): 
		super(GHP, self).__init__()
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

		self.mu = nn.Parameter(
			torch.zeros(
				[self.coarse_num], 
				dtype=torch.float32, device=self.device
			)
		)

		"""
		alpha + delta : [k, k'] -- effect of k to k'
		"""
		self.alpha = nn.Parameter(
			torch.zeros(
				[self.coarse_num, self.coarse_num], 
				dtype=torch.float32, device=self.device
			)
		)

		self.delta = nn.Parameter(
			torch.zeros(
				[self.coarse_num, self.coarse_num], 
				dtype=torch.float32, device=self.device
			)
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
	
	def get_states(self, event_tensor, dtime_tensor): 
		"""
		input 
			event_tensor [B x T+2] : tensor of event types 
			dtime_tensor [B x T+2] : tensor of dtimes
		return 
			past_event [B x T]
			past_time [B x T]
		"""
		past_event = event_tensor[:, 1:-1] # discard BOS EOS 
		past_event[past_event >= self.event_num] = 0
		past_event = self.coarse_to_fine.get_coarse_for_given_fine(past_event)
		past_time = torch.cumsum(dtime_tensor, dim=-1)[:, 1:-1]
		return past_event, past_time


	def get_intensities(self, past_event, past_time, time_tensor): 
		"""
		get intensities given types 
		e.g., for MLE, compute intensities for the sum term in log-likelihood
		e.g., for NCE, compute intensities for times drawn from p or q 
		"""
		batch_size, Tp = time_tensor.size()
		_, T = past_time.size()

		past_alpha = F.softplus(self.alpha[past_event, :])
		past_delta = F.softplus(self.delta[past_event, :]) + self.eps
		# B x T x C

		all_inten = torch.ones(
			[batch_size, Tp, self.coarse_num], dtype=torch.float32, device=self.device
		)
		# B x T' x C

		for i in range(T): 
			"""
			T is just seq length that any model (e.g., NHP) needs to loop over
			T' can be arbitrarily large so we parallelize that
			"""
			past_alpha_i = past_alpha[:, i, :] # B x C
			past_delta_i = past_delta[:, i, :] # B x C
			past_time_i = past_time[:, i] # B

			"""
			compute effect to accumulate to any future event
			"""
			use = torch.zeros(
				[batch_size, Tp], dtype=torch.float32, device=self.device
			)
			elapsed_time = time_tensor - past_time_i.unsqueeze(-1)
			use[elapsed_time > self.eps] = 1.0
			elapsed_time[elapsed_time < self.eps] = 0.0 

			# B x T' 
			effect = past_alpha_i.unsqueeze(1) * torch.exp(
				- past_delta_i.unsqueeze(1) * elapsed_time.unsqueeze(-1)
			)
			# B x T' x C
			
			effect = effect * use.unsqueeze(-1)

			all_inten += effect

		all_inten = effect + F.softplus(self.mu).unsqueeze(0).unsqueeze(0)
		# B x T' x C
		fine_inten = self.coarse_to_fine.get_fine_probs_all_types(all_inten)
		# B x T' x K 

		return fine_inten


	def get_mc_samples(self, dtime_tensor, 
		mc_sample_num_tensor, duration_tensor, mask_tensor): 
		"""
		similar to get_mc_samples in GNHP
		"""
		all_time_inter = []
		all_mask_inter = []

		batch_size, T_plus_1 = dtime_tensor.size()

		mc_max = torch.max(mc_sample_num_tensor)
		mc_max = mc_max if mc_max > 1 else 1
		u = torch.ones(size=[batch_size, mc_max], dtype=torch.float32, device=self.device)
		u, _ = torch.sort(u.uniform_(0.0, 1.0)) # batch_size x mc_max 
		sampled_time = u * duration_tensor.unsqueeze(-1)

		all_time_inter = sampled_time
		all_mask_inter = torch.ones_like(all_time_inter)

		return all_time_inter, all_mask_inter

	"""
	NOTE : BELOW ARE USED FOR PROPOSAL DISTRIBUTIONS
	NOT REVISED YET 
	CODE FOR GNHP HERE FOR REFERENCE
	"""

	def get_noise_samples(self, method, 
		event_tensor, dtime_tensor, target_tensor, mask_tensor, duration_tensor, 
		noise_process_num, noise_type_num, over): 
		raise NotImplementedError

	#@profile
	def get_noise_samples_given_states(self, nce_method, 
		target_tensor, mask_tensor, 
		dtime_tensor, duration_tensor, 
		noise_process_num, noise_type_num, over): 
		raise NotImplementedError

	#@profile
	def draw_noise_samples_per_interval(self, nce_method, 
		target, dtime, M1, M2, over, type_mask=None): 
		raise NotImplementedError

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
