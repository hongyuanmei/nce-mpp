import numpy as np
import torch
import torch.nn as nn


class CF(nn.Module): 
	
	"""
	a layer that maps coarse prob dist to fine prob dist
	"""
	def __init__(self, *, coarse_num, event_num, fine_to_coarse, device=None): 
		super(CF, self).__init__()

		device = device or 'cpu'
		self.device = torch.device(device)
		self.eps = np.finfo(float).eps

		self.coarse_num = coarse_num # # of coarse event types 
		self.event_num = event_num # # of fine event types 
		"""
		for each coarse type c, track its fine types
		"""
		aux = dict()
		for k, c in fine_to_coarse.items(): 
			if c not in aux: 
				aux[c] = list()
			aux[c].append(k)
		max_Kc = -1
		for c, ck in aux.items(): 
			if len(ck) > max_Kc: 
				max_Kc = len(ck) 
		assert max_Kc > 0, "c has no k???!!!"
		"""
		group k into coarse-grained c : 
		coarse_to_fine : tensor of c -> k (many k each c)
		fine_to_coarse : tensor of k -> c (one c each k)
		mask : mask out padding for each c 
		indices : k-th = (c,i) : k-th type is c,i-th of coarse_to_fine
		probs : for c, probs over its fine types
		"""
		self.coarse_to_fine = torch.zeros( 
			self.coarse_num, max_Kc, dtype=torch.long, device=self.device )
		self.fine_to_coarse = torch.zeros(
			self.event_num, dtype=torch.long, device=self.device )
		self.mask = torch.zeros(
			self.coarse_num, max_Kc, dtype=torch.float32, device=self.device )
		self.indices = torch.zeros(
			self.event_num, 2, dtype=torch.long, device=self.device )
		for c, ck in aux.items(): 
			# c : ck = [ many fine types ]
			for i, k in enumerate(ck): 
				self.coarse_to_fine[c,i] = k
				self.fine_to_coarse[k] = c
				self.mask[c, i] = 1.0 
				self.indices[k, 0] = c
				self.indices[k, 1] = i
		probs_data = torch.zeros(
			self.coarse_num, max_Kc, dtype=torch.float32, device=self.device )
		self.probs = nn.Parameter(probs_data) # parameter probs 
	
	def cuda(self, device=None):
		device = device or 'cuda:0'
		self.device = torch.device(device)
		assert self.device.type == 'cuda'
		super().cuda(self.device)

	def cpu(self):
		self.device = torch.device('cpu')
		super().cuda(self.device)

	def get_probs(self): 
		# when each c has only one k, learning this parameters makes no difference 
		positive_mass = torch.exp( -torch.abs(self.probs) )
		# exp(-|x|) : no underflow or overflow 
		masked_probs = self.mask * positive_mass + self.eps # mask c->k structure
		norm_probs = masked_probs / masked_probs.sum(1, keepdim=True) # normalize
		# C x Kmax
		return norm_probs

	def get_coarse_for_given_fine(self, fine_types): 
		"""
		for a tensor of fine types ids, find their coarse types 
		e.g., for a GNHP, we only want intensity of a given fine type 
		then we have to find the corresponding coarse type first 
		"""
		return self.fine_to_coarse[fine_types]
	
	def get_fine_probs_all_types(self, coarse_probs): 
		"""
		input
			coarse_probs : unnormalized probs 
			a tensor of D_1 x ... x D_N x coarse_num
		return 
			fine_probs : unormalized probs 
			a tensor of D_1 x ... x D_N x event_num
		"""
		if self.coarse_num == 1 : 
			return self.get_fine_probs_all_types_Ceq1(coarse_probs)
		else: 
			return self.get_fine_probs_all_types_Cneq1(coarse_probs)

	def get_fine_probs_all_types_Cneq1(self, coarse_probs): 
		"""
		C != 1 : need a general method
		"""
		coarse_probs_divided = coarse_probs.unsqueeze(-1) * self.get_probs()
		# D_1 x ... x D_N x coarse_num * Kmax
		"""
		for efficiency, we only consider N == 2 
		D1 = batch_size, D2 = # times
		other cases need clever transpose or indexing case by case 
		which might be slower 
		"""
		fine_probs = coarse_probs_divided[:, :, self.indices[:,0], self.indices[:,1]]
		# D_1 x ... x D_N x K
		return fine_probs

	def get_fine_probs_all_types_Ceq1(self, coarse_probs): 
		"""
		C == 1 : has a fast special implementation
		NOTE : having this special case is very important 
		because C=1 is often used as noise distribution
		so its speed is very important to get a good wall-clock time
		"""
		return coarse_probs * self.get_probs()[0, :]
		#return torch.matmul(coarse_probs, self.get_probs() )


	def get_fine_probs_given_types(self, coarse_probs, coarse_types, fine_types): 
		"""
		input 
			coarse_probs : unnormalized probs, D_1 x ... x D_N
			coarse_types : D_1 x ... x D_N
			fine_types : D_1 x ... x D_N
		"""
		all_cf_probs = self.get_probs() # C x Kmax
		col_idx = self.indices[fine_types, 1]
		# D_1 x ... x D_N 
		cf_probs = all_cf_probs[coarse_types, col_idx]
		# D_1 x ... x D_N
		return cf_probs * coarse_probs


class LinearEmbedding(nn.Module): 

	"""
	a layer that maps hidden states to (coarse) intensities 
	it a wrapper for nn.Embedding that is easy for indexing 
	e.g., only compute intensities for given types 
	it also has a linear-alike function that is easy for multiplication 
	e.g., compute intensities for all types 
	"""
	def __init__(self, *, num_embeddings, embedding_dim, device=None): 
		super(LinearEmbedding, self).__init__()

		device = device or 'cpu'
		self.device = torch.device(device)
		self.eps = np.finfo(float).eps

		self.num_embeddings = num_embeddings # # of embeddings
		self.embedding_dim = embedding_dim # dimension
		self.ranger = torch.arange(
			0, num_embeddings, dtype=torch.long, device=self.device)
		
		self.emb = nn.Embedding(num_embeddings, embedding_dim)
	
	def cuda(self, device=None):
		device = device or 'cuda:0'
		self.device = torch.device(device)
		assert self.device.type == 'cuda'
		super().cuda(self.device)

	def cpu(self):
		self.device = torch.device('cpu')
		super().cuda(self.device)

	def get_embeddings_all_types(self): 
		return self.emb(self.ranger) # num_embeddings x embedding_dim
	
	def get_embeddings_given_types(self, event_tensor): 
		# event_tensor : D1 x ... x DN
		return self.emb(event_tensor) # D1 x ... x DN x embedding_dim