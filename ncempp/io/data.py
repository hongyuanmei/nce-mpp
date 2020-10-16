import numpy as np
import torch


class DataProcessor(object):
	
	def __init__(self,
		idx_BOS, idx_EOS, idx_PAD, device=None):
		self.idx_BOS = idx_BOS
		self.idx_EOS = idx_EOS
		self.idx_PAD = idx_PAD

		device = device or 'cpu'
		self.device = torch.device(device)
		self.eps = np.finfo(float).eps

	def process_seq(self, seq):
		"""
		The process seq function is moved to the class.
		:param list seq:
		"""
		# including BOS and EOS
		len_seq = len(seq)+2
		event = torch.zeros(
			size=[len_seq], device=self.device, dtype=torch.int64)\
			.long()
		event[-1] = self.idx_EOS
		event[0] = self.idx_BOS

		dtime = torch.zeros(
			size=[len_seq], device=self.device, dtype=torch.float32)\

		for token_idx, token in enumerate(seq):
			event[token_idx+1] = int(token['name'])
			dtime[token_idx+1] = float(token['time_since_last_event'])

		dtime[-1] = 0.1
		duration = seq[-1]['time'] + self.eps

		return event, dtime, len_seq-2, duration

	def process_batch(self, data, batch_size):
		data_size = len(data)
		idx_seq = 0

		batched_data = []
		batched_lens = []
		batched_noise = []

		input = []
		input_lens = []

		while idx_seq < data_size:
			one_seq = data[idx_seq]
			event, dtime, len_seq, duration = self.process_seq(one_seq)
			input.append((event, dtime, len_seq, duration))
			input_lens.append(len_seq)

			if len(input) >= batch_size or idx_seq == data_size - 1:
				batched_lens.append(
					torch.tensor(
						input_lens, device=self.device).view(len(input), 1))
				input_lens = []

				batchdata_seqs = self.org_batch(input)
				batched_data.append(batchdata_seqs)
				batched_noise.append(None)
				input = []

			idx_seq = idx_seq + 1
		return batched_data, batched_noise #, batched_lens


	def org_batch(self, batch_of_seqs):

		batch_size = len(batch_of_seqs)
		max_len = -1

		for event_dtime_tuple in batch_of_seqs:
			seq_len = event_dtime_tuple[0].size(0)
			if seq_len > max_len:
				max_len = seq_len

		event_tensor = torch.zeros(size=[batch_size, max_len], device=self.device)\
			.fill_(self.idx_PAD)\
			.long()
		dtime_tensor = torch.zeros(size=[batch_size, max_len], device=self.device)\
			.fill_(0.1) #may need to change
		token_num_tensor = torch.zeros(size=[batch_size], device=self.device)\
			.long()
		duration_tensor = torch.zeros(size=[batch_size], device=self.device)

		for i_batch, event_dtime_tuple in enumerate(batch_of_seqs):
			seq_len = event_dtime_tuple[0].size(0)

			event_tensor[i_batch, :seq_len] = event_dtime_tuple[0]
			dtime_tensor[i_batch, :seq_len] = event_dtime_tuple[1]
			token_num_tensor[i_batch] = event_dtime_tuple[2]
			duration_tensor[i_batch] = event_dtime_tuple[3]

		return event_tensor, dtime_tensor, token_num_tensor, duration_tensor
