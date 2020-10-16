import numpy as np
import torch
import torch.nn as nn


class MLE(nn.Module): 

	def __init__(self, *, mc_sample_num, device=None): 
		super(MLE, self).__init__()
		device = device or 'cpu'
		self.device = torch.device(device)
		self.mc_sample_num = mc_sample_num
		self.mc_sample_num_eval = max(1.0, mc_sample_num)
		self.eps = np.finfo(float).eps

	def cuda(self, device=None):
		device = device or 'cuda'
		self.device = torch.device(device)
		assert self.device.type == 'cuda'
		super().cuda(self.device)

	def cpu(self):
		self.device = torch.device('cpu')
		super().cuda(self.device)

	def forward(self, model, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor, eval_tag=False): 
		"""
		model : GNHP or GPP or GHP ... 
		choose the right MLE function based on model type
		"""
		if type(model).__name__ == 'GNHP': 
			return self.mle_gnhp(
				model, event_tensor, dtime_tensor, token_num_tensor, duration_tensor, 
				eval_tag=False
			)
		elif type(model).__name__ == 'GHP': 
			return self.mle_ghp(
				model, event_tensor, dtime_tensor, token_num_tensor, duration_tensor, 
				eval_tag=False
			)
		elif type(model).__name__ == 'GPP': 
			return self.mle_gpp(
				model, event_tensor, dtime_tensor, token_num_tensor, duration_tensor, 
				eval_tag=False
			)
		else: 
			raise Exception(f"Unknown model type : {type(model).__name__}")

	#@profile
	def mle_gnhp(self, model, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor, eval_tag=False): 
		"""
		input
			model [object of GNHP] : p_theta or simply p 
			event_tensor [B x T+2] : a batch of event types with length T+2
			dtime_tensor [B x T+2] : a batch of dtimes with length T+2
			token_num_tensor [B] : # of tokens per seq 
			duration_tensor [B] : duration per seq
		"""
		target_tensor, mask_tensor = model.get_target(event_tensor)
		# B x T+1 , starting from 1st actual event 
		all_c_p_actual, all_cb_p_actual, all_d_p_actual, all_o_p_actual, _, _ = \
		model.get_cells_gates_states(event_tensor, dtime_tensor)
		# B x T+1 x D 
		inten_p_actual = model.get_intensities_given_types(
			all_c_p_actual, all_cb_p_actual, all_d_p_actual, all_o_p_actual, 
			target_tensor.unsqueeze(-1), dtime_tensor[:, 1:]
		) # B x (T + 1) x 1
		"""
		NOTE : we sometimes use very small mc sample to train for speed-up
		but we need >= 1 for stable eval
		"""
		mc_sample_num = self.mc_sample_num_eval if eval_tag else self.mc_sample_num
		mc_sample_num_tensor = (token_num_tensor.float() * mc_sample_num).long()
		all_c_p_noise, all_cb_p_noise, all_d_p_noise, all_o_p_noise, \
		all_dtime_noise, all_mask_noise = model.get_mc_samples(
			all_c_p_actual, all_cb_p_actual, 
			all_d_p_actual, all_o_p_actual, 
			dtime_tensor[:, 1:], mc_sample_num_tensor, 
			duration_tensor, mask_tensor
		)
		# B x T' (x D)
		inten_p_noise = model.get_intensities_all_fine_types(
			all_c_p_noise, all_cb_p_noise, all_d_p_noise, all_o_p_noise, 
			all_dtime_noise
		)
		# B x T' x K
		log_inten = torch.log(inten_p_actual.sum(-1) + self.eps) * mask_tensor
		# B x (T + 1)
		integral = torch.sum(inten_p_noise, dim=-1) * all_mask_noise
		# B x T'
		actual_counts = torch.sum(all_mask_noise, dim=-1) # B 
		integral = torch.sum(integral, dim=-1) / actual_counts
		integral = duration_tensor * integral # B
		log_likelihood = torch.sum(log_inten) - torch.sum(integral)
		"""
		count # of intensities that we computed for this log-likelihood
		"""
		inten_num = float(torch.sum(mask_tensor)) 
		# one intensity per actual event 
		inten_num += float(torch.sum(all_mask_noise)) * model.get_inten_num()
		# total # of intensities per noise sample
		return log_likelihood, inten_num
	

	def mle_ghp(self, model, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor, eval_tag=False): 
		"""
		input
			model [object of GNHP] : p_theta or simply p 
			event_tensor [B x T+2] : a batch of event types with length T+2
			dtime_tensor [B x T+2] : a batch of dtimes with length T+2
			token_num_tensor [B] : # of tokens per seq 
			duration_tensor [B] : duration per seq
		"""
		target_tensor, mask_tensor = model.get_target(event_tensor)
		# B x T+1
		past_event, past_time = model.get_states(event_tensor, dtime_tensor)
		# B x T
		"""
		NOTE : we sometimes use very small mc sample to train for speed-up
		but we need >= 1 for stable eval
		"""
		mc_sample_num = self.mc_sample_num_eval if eval_tag else self.mc_sample_num
		mc_sample_num_tensor = (token_num_tensor.float() * mc_sample_num).long()
		
		all_time_noise, all_mask_noise = model.get_mc_samples(
			dtime_tensor[:, 1:], mc_sample_num_tensor, duration_tensor, mask_tensor
		)
		# B x T' 
		inten_p_actual = model.get_intensities(
			past_event, past_time, torch.cumsum(dtime_tensor, dim=-1)[:, 1:]
		)
		inten_p_actual = torch.gather(
			inten_p_actual, # B x T+1 x K 
			-1, target_tensor.unsqueeze(-1) # B x T+1 x 1
		).squeeze(-1)
		# B x T+1
		inten_p_noise = model.get_intensities(
			past_event, past_time, all_time_noise
		)
		# B x T' x K 
		log_inten = torch.log(inten_p_actual + self.eps) * mask_tensor
		# B x T+1
		integral = torch.sum(inten_p_noise, dim=-1) * all_mask_noise
		# B x T' 
		actual_counts = torch.sum(all_mask_noise, dim=-1) # B 
		integral = torch.sum(integral, dim=-1) / actual_counts
		integral = duration_tensor * integral # B 
		log_likelihood = torch.sum(log_inten) - torch.sum(integral)
		"""
		count # of intensities that we computed for this log-likelihood
		"""
		inten_num = float(torch.sum(mask_tensor))
		inten_num += float(torch.sum(all_mask_noise)) * model.get_inten_num()

		return log_likelihood, inten_num



	def mle_gpp(self, model, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor, eval_tag=False): 
		"""
		input
			model [object of GPP] : p_theta or simply p 
			event_tensor [B x T+2] : a batch of event types with length T+2
			dtime_tensor [B x T+2] : a batch of dtimes with length T+2
			token_num_tensor [B] : # of tokens per seq 
			duration_tensor [B] : duration per seq
		"""
		target_tensor, mask_tensor = model.get_target(event_tensor)
		# B x T+1 , starting from 1st actual event 
		inten_p_actual = model.get_intensities_given_types(
			target_tensor.unsqueeze(-1)
		)
		# B x (T+1) x 1
		inten_p_noise = model.get_intensities_all_fine_types()
		# K
		log_inten = torch.log(inten_p_actual.sum(-1) + self.eps) * mask_tensor
		# B x (T + 1)
		integral = torch.sum(inten_p_noise) * duration_tensor # B
		log_likelihood = torch.sum(log_inten) - torch.sum(integral)
		"""
		count # of intensities that we computed for this log-likelihood
		for GPP (generalized Poisson process), intensities are stationary 
		so we only need to make K computation
		the cost stays the same when we have more sequences 
		because intensities are history-independent
		actually, # of intensities evaluated means nothing in this model
		arguably, it should be 0 because we only care 
		# of ``neural'' intensity evaluations
		"""
		inten_num = float(model.event_num)
		return log_likelihood, inten_num