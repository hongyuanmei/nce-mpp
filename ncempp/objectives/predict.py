import random
import numpy as np
import torch
import torch.nn as nn

class Predict(nn.Module): 

	def __init__(self, *, over_rate, device=None): 
		super(Predict, self).__init__()

		device = device or 'cpu'
		self.device = torch.device(device)
		self.over_rate = over_rate
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
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor): 
		"""
		model : GNHP or GPP or GHP ... 
		"""
		if type(model).__name__ == 'GNHP': 
			return self.predict_gnhp(
				model, event_tensor, dtime_tensor, token_num_tensor, duration_tensor
			)
		elif type(model).__name__ == 'GHP': 
			return self.predict_ghp(
				model, event_tensor, dtime_tensor, token_num_tensor, duration_tensor
			)
		elif type(model).__name__ == 'GPP': 
			return self.predict_gpp(
				model, event_tensor, dtime_tensor, token_num_tensor, duration_tensor 
			)
		else: 
			raise Exception(f"Unknown model type : {type(model).__name__}")

	def predict_gnhp(self, model, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor): 
		target_tensor, mask_tensor = model.get_target(event_tensor)
		all_c_p_actual, all_cb_p_actual, all_d_p_actual, all_o_p_actual, _, _ = \
		model.get_cells_gates_states(event_tensor, dtime_tensor)
		all_dtime_predict, all_type_predict = \
			model.get_next_events_given_states(
				all_c_p_actual, all_cb_p_actual, all_d_p_actual, all_o_p_actual, 
				dtime_tensor[:, 1:], self.over_rate
			)
		se = (all_dtime_predict - dtime_tensor[:, 1:]) ** 2
		se = torch.sum(se * mask_tensor)
		en = torch.abs(all_type_predict - target_tensor).float() * mask_tensor
		en = torch.sum(en > 0.5)
		return se, en 
	
	def predict_gpp(self, model, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor): 
		raise NotImplementedError

	def predict_ghp(self, model, 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor): 
		raise NotImplementedError
