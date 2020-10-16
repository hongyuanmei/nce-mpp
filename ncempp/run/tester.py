import numpy as np
import random
import torch
import torch.nn as nn
from torch import autograd
import pickle
import os 
import psutil

from torch import optim

from nce_point_process.models.nhp import GNHP
from nce_point_process.models.hp import GHP
from nce_point_process.models.pp import GPP
from nce_point_process.objectives.mle import MLE
from nce_point_process.objectives.nce import NCE
from nce_point_process.objectives.predict import Predict
from nce_point_process.io.data import DataProcessor
from nce_point_process.io.log import LogReader
from nce_point_process.run.manager import Manager

class Tester(Manager): 

	def __init__(self, *, args): 
		self.args = args
		"""
		set random numbers 
		"""
		random.seed(args['Seed'])
		np.random.seed(args['Seed'])
		torch.manual_seed(args['Seed'])

		"""
		create model p 
		"""
		log_path = os.path.join(
			self.args['PathSave'], f"{self.args['ModelFolder']}/log.txt" )
		saved_args = self.load_log(log_path)
		config_path = os.path.join(
			self.args['PathData'], f"{saved_args['Config']}.config")
		coarse_num, event_num, fine_to_coarse = self.load_config(config_path)
		device = 'cuda' if args['UseGPU'] else 'cpu'

		if saved_args['Model'] == 'gnhp': 
			self.model = GNHP(
				coarse_num = coarse_num, 
				event_num = event_num, 
				fine_to_coarse = fine_to_coarse, 
				hidden_dim = saved_args['DimLSTM'], 
				device = device
			)
		elif saved_args['Model'] == 'gpp': 
			self.model = GPP(
				coarse_num = coarse_num, 
				event_num = event_num, 
				fine_to_coarse = fine_to_coarse, 
				device = device
			)
		elif saved_args['Model'] == 'ghp': 
			assert False, f"Hawkes can't be used as testing model yet..."
			self.model = GHP(
				coarse_num = coarse_num, 
				event_num = event_num, 
				fine_to_coarse = fine_to_coarse, 
				device = device
			)
		else: 
			raise Exception(f"Unknown model : {args['Model']}")

		model_path = os.path.join(
			self.args['PathSave'], f"{self.args['ModelFolder']}/saved_model"
		)
		if '.pkl' in saved_args['PathModel']: 
			model_path += '.pkl'
		self.load_model(self.model, model_path)
		if args['UseGPU']: 
			self.model.cuda()

		self.proc = DataProcessor(
			idx_BOS = self.model.idx_BOS, 
			idx_EOS = self.model.idx_EOS, 
			idx_PAD = self.model.idx_PAD, 
			device = 'cuda' if args['UseGPU'] else 'cpu'
		)
		
		self.mle = MLE(
			mc_sample_num = self.args['MCSample'], 
			device = 'cuda' if self.args['UseGPU'] else 'cpu'
		)
		self.predict = Predict(
			over_rate = self.args['OverRate'], 
			device = 'cuda' if self.args['UseGPU'] else 'cpu'
		)

		self.device = 'cuda' if args['UseGPU'] else 'cpu'
		self.write_buffer = []
		self.use_buffer = True if args['WriteBuffer'] == 'buffer' else False

	def compute_predict_error(self, split, i_batch_comp): 
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor = \
			self.data_tensor[split][i_batch_comp]
		se, en = self.predict(
			self.model, event_tensor, dtime_tensor, token_num_tensor, duration_tensor
		)
		return float(se), float(en), float(torch.sum(token_num_tensor))

	def eval_predict_batch(self, split, i_batch_comp): 
		with torch.no_grad(): 
			se, en, token_num = \
				self.compute_predict_error(split, i_batch_comp)
		return se, en, token_num

	def eval_predict_epoch(self, split): 
		total_se = 0.0 
		total_en = 0.0 
		total_num = 0.0 
		for i_batch_comp in range(len(self.data_tensor[split])): 
			se_batch, en_batch, token_num_batch = \
				self.eval_predict_batch(split, i_batch_comp)
			total_se += float(se_batch)
			total_en += float(en_batch)
			total_num += float(token_num_batch)
		return total_se, total_en, total_num