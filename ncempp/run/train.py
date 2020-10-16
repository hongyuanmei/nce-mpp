import argparse
import datetime
import os
import psutil
import time

import torch
#from tqdm import tqdm
# Hongyuan: I have to disable it because it is not MARCC-friendly, sorry

from nce_point_process.run.manager import Manager

#@profile
def train(args):
	# track memory
	#process = psutil.Process(os.getpid())
	#print(f'memory starts at : {process.memory_info().rss/1000000:.3f} MB')
	"""
	train model with chosen method
	"""
	manager = Manager(args=args)
	print('prepare model and training method')
	manager.prepare_training()
	#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')
	print('load train and dev data')
	manager.load_data( ['train', 'dev'] )
	#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')
	print('process data (into tensors) ')
	manager.process_data()
	#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')

	print('start training ...')

	best_obj = -1e6
	best_epoch = -1
	best_batch = -1

	report_gap = args['TrackPeriod'] // args['SizeBatchUpdate']
	report_gap = 1 if report_gap == 0 else report_gap

	train_times = list()
	dev_times = list()
	inten_nums = list()

	num_reports = 0
	max_epochs = args['MaxEpoch']
	train_batch_num = manager.get_batchupdate_num('train')

	for epoch in range(max_epochs):

		for i_batch in range(train_batch_num):
			# i_batch for updating params

			iteration = epoch * train_batch_num + i_batch

			"""
			Train and optimize
			"""
			tic = time.time()
			obj, num, inten_num = manager.train_batch(i_batch)
			toc = time.time()
			train_times.append(toc - tic)
			inten_nums.append(inten_num)

			#message = f"iteration {iteration} loglik (per token) is {(obj/num):.4f}"
			#manager.checkpoint(message) # too many checkpoint messages
			#message = f"training on {i_batch}-th batch of {epoch}-th epoch"
			#print(message)
			#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')
			"""
			Eval on dev data
			"""
			if iteration % report_gap == report_gap - 1:
				"""
				old way : (iteration // report_gap > num_reports)
				this can be buggy 
				e.g., if batch_size = 1 and track_period = 1
				then report_gap = 1 and we should eval after every batch
				however, at beginning, iter = 0 and num_reports = 0
				then iter // report_gap = 0 not > 0 = num_reports
				so it doesn't eval after the first batch!!!
				new way is exact and doesn't have this issue
				e.g., when iter = 0 and report_gap = 1
				iter % report_gap = 0 == 0 = report_gap - 1
				"""
				print(f"validating after {i_batch}-th update batch of {epoch}-th epoch")

				tic = time.time()
				obj, num = manager.eval_mle_epoch('dev')
				avg_obj = obj / num
				toc = time.time()
				dev_times.append(toc - tic)
				#print(f"after validating one epoch")
				#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')

				message = f"eval: {iteration} iteration, {i_batch}-th batch of {epoch}-th epoch, loglik (per token) is {avg_obj:.4f}"
				manager.print_or_buffer(message)
				manager.checkpoint(message)
				"""
				track total time for training on {report_gap} batches
				"""
				total_train_time = sum(train_times[-report_gap:])
				message = f'{total_train_time:.4f} seconds for training since last validation'
				manager.print_or_buffer(message)
				manager.checkpoint(message)
				# for train time
				total_inten_num = sum(inten_nums[-report_gap:])
				message = f'# of intensities computed = {total_inten_num:.0f} since last validation'
				manager.print_or_buffer(message)
				manager.checkpoint(message)
				# for # of intensities computed
				message = f'{dev_times[-1]:.4f} seconds for eval' # useful to track :-) 
				manager.print_or_buffer(message)
				manager.checkpoint(message)
				# for dev time
				if avg_obj > best_obj:
					best_obj = avg_obj
					updated = True
					best_batch = i_batch
					best_epoch = epoch
				else:
					updated = False
				message = f"Current best loglik is {best_obj:.4f} (updated at {best_batch}-th batch of {best_epoch}-th epoch)"
				if updated:
					message += ", best updated at this iteration"
					manager.save_model()
				#print(message)
				manager.print_or_buffer(message)
				manager.checkpoint(message)
				num_reports += 1
	
	message = f'training finished' 
	# important indicator for if training finished
	# on a server like MARCC, 
	# sometimes training terminated but not finished
	manager.print_or_buffer(message)
	manager.checkpoint(message)
	manager.flush_buffer()


def get_args():
	"""
	Retrieves arguments from command line
	"""
	parser = argparse.ArgumentParser(description='Training model ...')
	parser.add_argument(
		'-ds', '--Dataset', required=True, type=str,
		help='e.g. smallnhp'
	)
	parser.add_argument(
		'-md', '--Model', default='gnhp', type=str, 
		choices=[ 'gnhp', 'gpp' , 'ghp' ],
		help='model to train : gnhp (default) or gpp or ghp?'
	)
	parser.add_argument(
		'-c', '--Config', required=True, type=str,
		help='name of config file used by model, e.g., onetoone'
	)
	parser.add_argument(
		'-d', '--DimLSTM', default=8, type=int,
		help='dimension of LSTM'
	)
	parser.add_argument(
		'-sbc', '--SizeBatchCompute', default=1, type=int,
		help='size of mini-batch (# seqs) to compute loss'
	)
	parser.add_argument(
		'-sbu', '--SizeBatchUpdate', default=1, type=int,
		help='size of mini-batch (# seqs) to update params---ideal : N x sbc'
	)
	parser.add_argument(
		'-tp', '--TrackPeriod', default=10, type=int,
		help='# of sequences for each checkpoint'
	)
	parser.add_argument(
		'-tr', '--TrainRatio', default=1.0, type=float, 
		help='ratio of training data to use for this run'
	)
	parser.add_argument(
		'-dr', '--DevRatio', default=1.0, type=float, 
		help='ratio of dev data to use for this run'
	)
	parser.add_argument(
		'-me', '--MaxEpoch', default=10, type=int,
		help='max # of epochs of training'
	)
	parser.add_argument(
		'-tm', '--TrainMethod', required=True, type=str, 
		choices=[ 
			'mle', # MLE method
			'lse', # least-square estimation
			'nce_frac', 'nce_async', 'nce_sync', # ranking-based NCE
			'nce_binary' # classification-based NCE
		],
		help='training method : mle or nce? for nce, frac or async or sync method? (frac and async are in paper)'
	)
	parser.add_argument(
		'-mc', '--MCSample', default=1, type=float,
		help='# of Monte-Carlo samples per actual event in MLE'
	)
	parser.add_argument(
		'-nf', '--NoiseFolder', type=str,
		help='folder name of saved noise model and log containing useful info like config file name'
	)
	parser.add_argument(
		'-rdp', '--ReDrawProb', type=float, default=0.0, 
		help='default 0.0 : probability to redraw noise samples for each real sequence'
	)
	parser.add_argument(
		'-np', '--NoiseProcess',default=1, type=int, 
		help='# of noise processes in parallel to sample noise times'
	)
	parser.add_argument(
		'-nt', '--NoiseType', default=1, type=int, 
		help='# of noise event types for each noise process'
	)
	parser.add_argument(
		'-nm', '--NoiseMode', type=str, 
		default='multinomial', choices=['multinomial', 'uniform'], 
		help='way to sample noise event types (from q) : '+\
			'multinomial--use computed intensities; uniform--uniformly (for speed-up)'
		# why not use action='store_true'? current way makes MARCC script simple
	)
	parser.add_argument(
		'-or', '--OverRate', default=1.0, type=float,
		help='over sampling rate in thinning algorithm (currently dummy)'
	)
	parser.add_argument(
		'-lr', '--LearnRate', default=1e-3, type=float,
		help='(starting) learning rate of the training algorithm'
	)
	parser.add_argument(
		'-sd', '--Seed', default=12345, type=int,
		help='random seed. e.g. 12345'
	)
	parser.add_argument(
		'-gpu', '--UseGPU', default=0, type=int, choices=[0,1],
		help='use GPU or not : 0--not, 1--use'
		# why not use action='store_true'? current way makes MARCC script simple
	)
	parser.add_argument(
		'-rp', '--RootPath', type=str, default='../../',
		help='root path of project'
	)
	parser.add_argument(
		'-wb', '--WriteBuffer', type=str, default='print', choices=['buffer', 'print'],
		help='print method: buffer or print?'
	)
	args = parser.parse_args()
	return vars(args)

def aug_args_with_log(dict_args):
	"""
	create the path to folder for saving logs, models, and plots
	"""
	id_process = os.getpid()
	time_current = datetime.datetime.now().isoformat()

	root_path = os.path.abspath(dict_args['RootPath'])
	dict_args['PathData'] = os.path.join(root_path, f"data/{dict_args['Dataset']}")
	dict_args['Version'] = torch.__version__
	dict_args['ID'] = id_process
	dict_args['TIME'] = time_current

	folder_name = get_foldername(dict_args, id_process)
	#print(folder_name)

	path = os.path.join(root_path, 'logs', dict_args['Dataset'])
	path_log = os.path.join(path, folder_name)
	os.makedirs(path_log)

	file_log = os.path.join(path_log, 'log.txt')
	file_model = os.path.join(path_log, 'saved_model.pkl') 
	# some sys may think a file with no extension is an executable : bad

	dict_args['PathLog'] = file_log
	dict_args['PathModel'] = file_model
	dict_args['PathSave'] = path
	dict_args['PathRun'] = path_log

def get_foldername(dict_args, id_process):
	"""
	create folder name for current run
	"""
	# format: [arg name, name used in path]
	args_used_in_name = [
		['TrainMethod', 'meth'],
		['Model', 'md'], 
		['Config', 'conf'],
		['MCSample', 'mc'], 
		['ReDrawProb', 'rdp'], 
		['NoiseProcess', 'np'], 
		['NoiseType', 'nt'], 
		['DimLSTM', 'dim'],
		['UseGPU', 'gpu'], 
		['SizeBatchCompute', 'batchcomp'],
		['SizeBatchUpdate', 'batchupd'],
		['TrainRatio', 'tr'], 
		['NoiseMode', 'nm'], 
		['Seed', 'seed'],
		#['LearnRate', 'lr'],
	]
	folder_name = list()
	for arg_name, rename in args_used_in_name:
		folder_name.append('{}-{}'.format(rename, dict_args[arg_name]))
	folder_name = '_'.join(folder_name)
	folder_name = f'{folder_name}_{id_process}'
	return folder_name

def main():

	dict_args = get_args()
	aug_args_with_log(dict_args)
	if '' in dict_args:
		del dict_args['']
	train(dict_args)


if __name__ == "__main__": main()