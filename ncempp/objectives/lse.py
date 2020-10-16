import numpy as np
import torch
import torch.nn as nn


class LSE(nn.Module): 

	def __init__(self, *, mc_sample_num, device=None): 
		super(LSE, self).__init__()
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
		choose the right LS function based on model type 
		least-square training objective 
		sum_k ( integral(lambda_k(t)^2) - 2 sum lambda_k(t) )
		"""
		if type(model).__name__ == 'GNHP': 
			return self.ls_gnhp(
				model, event_tensor, dtime_tensor, token_num_tensor, duration_tensor, 
				eval_tag=False
			)
		elif type(model).__name__ == 'GHP': 
			return self.ls_ghp(
				model, event_tensor, dtime_tensor, token_num_tensor, duration_tensor, 
				eval_tag=False
			)
		elif type(model).__name__ == 'GPP': 
			return self.ls_gpp(
				model, event_tensor, dtime_tensor, token_num_tensor, duration_tensor, 
				eval_tag=False
			)
		else: 
			raise Exception(f"Unknown model type : {type(model).__name__}")

	def ls_gnhp(self, model, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor, eval_tag=False): 
		"""
		very similar to mle_gnhp in MLE 
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
		inten_p_actual = inten_p_actual.sum(-1) * mask_tensor
		# B x (T + 1)

		# B x T' (x D)
		inten_p_noise = model.get_intensities_all_fine_types(
			all_c_p_noise, all_cb_p_noise, all_d_p_noise, all_o_p_noise, 
			all_dtime_noise
		)
		# B x T' x K
		quad_inten_p_noise = inten_p_noise ** 2
		# B x T' x K 
		integral = torch.sum(quad_inten_p_noise, dim=-1) * all_mask_noise
		# B x T' 
		actual_counts = torch.sum(all_mask_noise, dim=-1) # B 
		integral = torch.sum(integral, dim=-1) / actual_counts
		integral = duration_tensor * integral # B 

		minus_square = 2 * torch.sum(inten_p_actual) - torch.sum(integral)

		inten_num = float(torch.sum(mask_tensor))
		inten_num += float(torch.sum(all_mask_noise)) * model.get_inten_num()

		return minus_square, inten_num

	def ls_ghp(self, model, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor, eval_tag=False): 
		raise NotImplementedError

	def ls_gpp(self, model, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor, eval_tag=False): 
		raise NotImplementedError