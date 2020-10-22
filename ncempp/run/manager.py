import numpy as np
import random
import torch
import torch.nn as nn
from torch import autograd
import pickle
import os 
import psutil

from torch import optim

from ncempp.models.nhp import GNHP
from ncempp.models.hp import GHP
from ncempp.models.pp import GPP
from ncempp.objectives.mle import MLE
from ncempp.objectives.nce import NCE
from ncempp.objectives.lse import LSE
from ncempp.io.data import DataProcessor
from ncempp.io.log import LogReader
from ncempp.io.log import LogWriter


class Manager(object): 

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
		config_path = os.path.join(args['PathData'], f"{args['Config']}.config")
		coarse_num, event_num, fine_to_coarse = self.load_config(config_path)

		if args['Model'] == 'gnhp': 
			self.model = GNHP(
				coarse_num = coarse_num, 
				event_num = event_num, 
				fine_to_coarse = fine_to_coarse, 
				hidden_dim = args['DimLSTM'], 
				device = 'cuda' if args['UseGPU'] else 'cpu'
			)
		elif args['Model'] == 'gpp': 
			self.model = GPP(
				coarse_num = coarse_num, 
				event_num = event_num, 
				fine_to_coarse = fine_to_coarse, 
				device = 'cuda' if args['UseGPU'] else 'cpu'
			)
		elif args['Model'] == 'ghp': 
			self.model = GHP(
				coarse_num = coarse_num, 
				event_num = event_num, 
				fine_to_coarse = fine_to_coarse, 
				device = 'cuda' if args['UseGPU'] else 'cpu'
			)
		else: 
			raise Exception(f"Unknown model : {args['Model']}")
		
		self.proc = DataProcessor(
			idx_BOS = self.model.idx_BOS, 
			idx_EOS = self.model.idx_EOS, 
			idx_PAD = self.model.idx_PAD, 
			device = 'cuda' if args['UseGPU'] else 'cpu'
		)
		if args['UseGPU']: 
			self.model.cuda()
		
		self.device = 'cuda' if args['UseGPU'] else 'cpu'
		self.write_buffer = []
		self.use_buffer = True if args['WriteBuffer'] == 'buffer' else False

	def prepare_training(self):
		"""
		prepare training-related stuff
		"""
		self.log_writer = LogWriter(self.args['PathLog'], self.args)
		self.optimizer = optim.Adam(
			self.model.parameters(), lr = self.args['LearnRate']
		)
		self.optimizer.zero_grad()
		"""
		create training obj
		"""
		self.mle = MLE(
			mc_sample_num = self.args['MCSample'], 
			device = 'cuda' if self.args['UseGPU'] else 'cpu'
		)
		if self.args['TrainMethod'] == 'mle': 
			self.noise = None 
			self.nce = None 
		elif self.args['TrainMethod'] == 'lse': 
			self.lse = LSE(
				mc_sample_num = self.args['MCSample'], 
				device = 'cuda' if self.args['UseGPU'] else 'cpu'
			)
			self.noise = None 
			self.nce = None 
		elif 'nce' in self.args['TrainMethod']: # nce_frac or _async or _sync
			log_path = os.path.join(
				self.args['PathSave'], f"{self.args['NoiseFolder']}/log.txt" )
			noise_args = self.load_log(log_path)
			config_path = os.path.join(
				self.args['PathData'], f"{noise_args['Config']}.config")
			coarse_num, event_num, fine_to_coarse = self.load_config(config_path)

			if 'Model' not in noise_args or noise_args['Model'] == 'gnhp': 
				self.noise = GNHP(
					coarse_num = coarse_num, 
					event_num = event_num, 
					fine_to_coarse = fine_to_coarse, 
					hidden_dim = noise_args['DimLSTM'], 
					noise_mode = self.args['NoiseMode'], 
					device = 'cuda' if self.args['UseGPU'] else 'cpu'
				)
			elif noise_args['Model'] == 'gpp': 
				self.noise = GPP(
					coarse_num = coarse_num, 
					event_num = event_num, 
					fine_to_coarse = fine_to_coarse, 
					noise_mode = self.args['NoiseMode'], 
					device = 'cuda' if self.args['UseGPU'] else 'cpu'
				)
			elif noise_args['Model'] == 'ghp': 
				raise Exception(f"Hawkes can't be used as proposal yet!")
			else: 
				raise Exception(f"Unknown model : {noise_args['Model']}")

			"""
			NOTE : model may be trained and saved on another machine
			so its abs path may be not compitable
			but its rel path must be compitable 
			so we make the hybrid/combined abs path here
			"""
			model_path = os.path.join(
				self.args['PathSave'], f"{self.args['NoiseFolder']}/saved_model"
			)
			if '.pkl' in noise_args['PathModel']: 
				model_path += '.pkl'
			self.load_model( self.noise, model_path )
			self.nce = NCE(
				noise_process_num = self.args['NoiseProcess'], 
				noise_type_num = self.args['NoiseType'], 
				over_rate = self.args['OverRate'], 
				redraw_prob = self.args['ReDrawProb'], 
				device = 'cuda' if self.args['UseGPU'] else 'cpu'
			) 
			if self.args['UseGPU']: 
				self.noise.cuda()
		else: 
			raise Exception(f"Unknow method : {self.args['TrainMethod']}")

	def load_config(self, config_path): 
		with open(config_path, 'r') as f: 
			lines = f.read().split('\n')
		if lines[-1] == '': 
			lines.pop(-1)
		rst = dict()
		for l in lines: 
			k, v = l.split(' ')
			rst[int(k)] = int(v)
		K = max(rst.keys()) + 1 
		C = max(rst.values()) + 1
		return C, K, rst

	def load_log(self, log_path): 
		log_reader = LogReader(log_path)
		saved_args = log_reader.get_args()
		return saved_args

	def load_data(self, splits): 
		"""
		load data for each split, e.g., train, dev, ...
		"""
		self.data = dict()
		for s in splits: 
			with open(os.path.join(self.args['PathData'], f'{s}.pkl'), 'rb') as f: 
				self.data[s] = pickle.load(f)
		"""
		for training and dev data, need to adjust # of sequences
		(usually for quick debugging and learning curves)
		NOTE : this may trigger bug in test stage
		but we don't need to test, right? we only compare learning curves!!!???
		"""
		if 'train' in splits and 'TrainRatio' in self.args: 
			total_num = len(self.data['train'])
			cur_num = int(total_num * self.args['TrainRatio'])
			cur_num = 1 if cur_num < 1 else cur_num
			assert cur_num >= 1 and cur_num <= total_num, f"# of seqs wrong : {cur_num}"
			self.data['train'] = self.data['train'][:cur_num]
		if 'dev' in splits and 'DevRatio' in self.args: 
			total_num = len(self.data['dev'])
			cur_num = int(total_num * self.args['DevRatio'])
			cur_num = 1 if cur_num < 1 else cur_num
			assert cur_num >= 1 and cur_num <= total_num, f"# of seqs wrong : {cur_num}"
			self.data['dev'] = self.data['dev'][:cur_num]

	def load_model(self, model, path_saved_state): 
		"""
		load saved_state and assign it to model
		model can be either p or noise dist q 
		"""
		model.load_state_dict(
			torch.load(
				path_saved_state, 
				map_location=torch.device('cpu')
			)
		)
	
	def save_model(self): 
		torch.save(self.model.state_dict(), self.args['PathModel'])

	def process_data(self): 
		"""
		NOTE : 
		another way is to only process data into tensors when needed
		it will save memory, esp for large datasets
		but it will be slower, cuz of re-processing
		but for the datasets we have used 
		we don't need to do this because they are not that large 
		actually, for a dataset with # seqs == 200K and avg. len == 10
		the memory cost is only < 100 MB --- very minor 
		so we don't need to worry about it now 
		however, in the future, if we indeed work with super-large data 
		we may have to make this change
		"""
		"""
		NOTE : 
		data tensors organized in BatchCompute
		data tensors for noise samples (init as None)
		"""
		self.data_tensor = dict()
		self.noise_tensor_pack = dict()
		for s in self.data: 
			self.data_tensor[s], self.noise_tensor_pack[s] = self.proc.process_batch(
				self.data[s], self.args['SizeBatchCompute'] )
		"""
		NOTE : 
		track list of batch compute for each batch update
		"""
		self.upd_to_comp = dict()
		for s in self.data: 
			self.upd_to_comp[s] = dict()
			i_upd, i_comp = 0, 0
			num_seqs = len(self.data[s])
			self.upd_to_comp[s][i_upd] = list()
			for i_seq in range(num_seqs): 
				batch_comp_end = \
					i_seq % self.args['SizeBatchCompute'] == self.args['SizeBatchCompute'] - 1
				data_end = i_seq == num_seqs - 1 
				if batch_comp_end or data_end : 
					self.upd_to_comp[s][i_upd].append(i_comp)
					i_comp += 1 
				batch_upd_end = \
					i_seq % self.args['SizeBatchUpdate'] == self.args['SizeBatchUpdate'] - 1
				if batch_upd_end and not data_end: 
					i_upd += 1
					self.upd_to_comp[s][i_upd] = list()
		assert i_comp == len(self.data_tensor[s]), "# of compute batches NOT match???!!!"

	def compute_mle(self, split, i_batch_comp): 
		"""
		compute mle on i_batch-th batch of data of split 
		input 
			split : train, dev, or test 
			i_batch_comp : i-th batch for computing
		"""
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor = \
			self.data_tensor[split][i_batch_comp]
		"""
		inten_num : # of intensities computed for this batch of data
		"""
		if 'train' in split: 
			eval_tag = False 
		else: 
			eval_tag = True

		log_likelihood, inten_num = self.mle(
			self.model, 
			event_tensor, dtime_tensor, token_num_tensor, duration_tensor, 
			eval_tag )
		return log_likelihood, float(torch.sum(token_num_tensor)), inten_num

	def compute_lse(self, split, i_batch_comp): 
		"""
		compute ls on i_batch-th batch of data of split 
		"""
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor = \
			self.data_tensor[split][i_batch_comp]
		if 'train' in split: 
			eval_tag = False 
		else: 
			eval_tag = True 
		
		minus_square, inten_num = self.lse(
			self.model, 
			event_tensor, dtime_tensor, token_num_tensor, duration_tensor, 
			eval_tag )
		
		return minus_square, float(torch.sum(token_num_tensor)), inten_num

	def compute_nce(self, split, i_batch_comp): 
		"""
		compute nce on i_batch-th batch of data of split 
		input 
			split : train, dev, or test 
			i_batch_comp : i-th batch for computing
		"""
		event_tensor, dtime_tensor, token_num_tensor, duration_tensor = \
			self.data_tensor[split][i_batch_comp]
		noise_tensor_pack = self.get_nosie_pack(split, i_batch_comp)
		"""
		inten_num : # of intensities computed for this batch of data
		"""
		nce_obj, inten_num, new_noise_pack = self.nce(
			self.args['TrainMethod'], # nce_frac or _async or _sync
			self.model, self.noise, 
			event_tensor, dtime_tensor, token_num_tensor, duration_tensor, 
			noise_tensor_pack
		)
		self.save_noise_pack(split, i_batch_comp, new_noise_pack)
		return nce_obj, float(torch.sum(token_num_tensor)), inten_num
	
	def get_nosie_pack(self, split, i_batch_comp): 
		rst = self.noise_tensor_pack[split][i_batch_comp]
		if rst is not None and self.device == 'cuda': 
			# move it from CPU to GPU 
			for i in range(6): 
				rst[i].to('cuda')
		return rst 

	def save_noise_pack(self, split, i_batch_comp, new_noise_pack): 
		if new_noise_pack is not None: 
			self.noise_tensor_pack[split][i_batch_comp] = new_noise_pack
		if self.device == 'cuda': 
			# new pack comes from GPU, move it to CPU 
			# or old pack moved to GPU already, move it back
			for i in range(6): 
				self.noise_tensor_pack[split][i_batch_comp][i].to('cpu')

	def train_batch(self, i_batch): 
		# i_batch : for updating
		#print(f"i batch = {i_batch}")
		self.optimizer.zero_grad() # clear grads
		token_num, inten_num = 0.0, 0.0
		batch_indices = self.upd_to_comp['train'][i_batch]
		for i_batch_comp in batch_indices: 
			# iterate each batch-compute responsible for this update
			if self.args['TrainMethod'] == 'mle': 
				to_maximize, token_num_i, inten_num_i = \
					self.compute_mle('train', i_batch_comp)
			elif self.args['TrainMethod'] == 'lse': 
				to_maximize, token_num_i, inten_num_i = \
					self.compute_lse('train', i_batch_comp)
			elif 'nce' in self.args['TrainMethod']: # nce_frac or _async or _sync
				to_maximize, token_num_i, inten_num_i = \
					self.compute_nce('train', i_batch_comp)
			else: 
				raise Exception(f"Unknow method : {self.args['TrainMethod']}")
			(-to_maximize).backward()
			token_num += token_num_i
			inten_num += inten_num_i
		"""
		NOTE : when we do backward() many times and then step()
		does it take sum or average??? sum!
		"""
		self.optimizer.step()
		return float(to_maximize), token_num, inten_num

	def eval_mle_batch(self, split, i_batch_comp): 
		with torch.no_grad(): 
			log_likelihood, token_num, inten_num = \
				self.compute_mle(split, i_batch_comp)
		return log_likelihood, token_num
	
	def eval_lse_batch(self, split, i_batch_comp): 
		with torch.no_grad(): 
			minus_square, token_num, inten_num = \
				self.compute_lse(split, i_batch_comp)
		return minus_square, token_num

	def eval_nce_batch(self, split, i_batch_comp): 
		with torch.no_grad(): 
			nce_obj, token_num, inten_num = self.compute_nce(split, i_batch_comp)
		return nce_obj, token_num

	def eval_mle_epoch(self, split): 
		#print(f"eval MLE for one epoch")
		#process = psutil.Process(os.getpid())
		total_log_likelihood = 0.0 
		total_num = 0.0 
		for i_batch_comp in range(len(self.data_tensor[split])): 
			#print(f"eval {i_batch_comp}-th compute batch")
			log_likelihood_batch, token_num_batch = \
				self.eval_mle_batch(split, i_batch_comp)
			total_log_likelihood += float(log_likelihood_batch)
			total_num += token_num_batch
			#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')
		return total_log_likelihood, total_num

	def prepare_gen(self): 
		"""
		prepare data-generating stuff
		"""
		self.log_writer = LogWriter(self.args['PathLog'], self.args)
	
	def gen_seq(self): 
		"""
		draw an i.i.d. sample sequence from model p
		"""
		with torch.no_grad(): 
			seq = self.model.draw_seq(self.args['NumToken'])
		last_time = 0.0 
		rst = []
		for dt, k in seq: 
			token = {
				'name': k, 'time': last_time + dt, 
				'time_since_last_event': dt
			}
			last_time = token['time']
			rst.append(token)
		return rst

	
	def get_batchcompute_num(self, split): 
		# return # of batches for computing loss 
		return len(self.data_tensor[split])

	def get_batchupdate_num(self, split): 
		# return # of batches for updating params
		return len(self.upd_to_comp[split])

	def get_seq_num(self, split): 
		return len(self.data[split])

	def checkpoint(self, to_write): 
		self.log_writer.checkpoint(to_write)

	def print_or_buffer(self, message):
		if self.use_buffer:
			self.write_buffer.append(message)
		else:
			print(message)

	def flush_buffer(self):
		for message in self.write_buffer:
			print(message)
		self.write_buffer = []