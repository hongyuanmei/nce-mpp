import argparse
import datetime
import os
import psutil
import time
import pickle

import torch

from ncempp.run.manager import Manager

#@profile
def gen(args):
	"""
	gen data from randomly init model
	"""
	manager = Manager(args=args)
	print('prepare model and training method')
	manager.prepare_gen()

	print('start generating ...')

	report_gap = args['TrackPeriod'] 
	report_gap = 1 if report_gap == 0 else report_gap

	for s in ['Train', 'Dev', 'Test', 'Future']: 
		
		arg_name = f'Num{s}'
		num_seqs = args[arg_name]

		message = f"generating {num_seqs} seqs for {s}"
		manager.print_or_buffer(message)
		manager.checkpoint(message)

		seqs = []

		for i in range(num_seqs): 

			"""
			generating seqs
			"""
			seq = manager.gen_seq()
			seqs.append(seq)

			if i % report_gap == report_gap - 1: 
				
				message = f'finish {i+1} sequences'
				manager.print_or_buffer(message)
				manager.checkpoint(message)

		message = f"finish generating for {s}"
		manager.print_or_buffer(message)
		manager.checkpoint(message)

		message = f"one sample sequence is {seqs[0]}"
		manager.print_or_buffer(message)
		manager.checkpoint(message)

		path_data = os.path.join(args['PathData'], f'{s.lower()}.pkl')
		with open(path_data, 'wb') as f: 
			pickle.dump(seqs, f)
	
	manager.save_model()

	message = f"generating finished"
	manager.print_or_buffer(message)
	manager.checkpoint(message)
	manager.flush_buffer()


def get_args():
	"""
	Retrieves arguments from command line
	"""
	"""
	NOTE : some args will be saved in log file 
	so that the data-generating model can be read and used as noise model
	"""
	parser = argparse.ArgumentParser(description='generating data ...')
	parser.add_argument(
		'-ds', '--Dataset', required=True, type=str,
		help='e.g. gnhp10k (10k means 10,000), name of the dataset we generate'
	)
	parser.add_argument(
		'-d', '--DimLSTM', default=8, type=int,
		help='dimension of LSTM of the data-generating model'
	)
	parser.add_argument(
		'-c', '--Config', required=True, type=str,
		help='name of config file used by model, e.g., onetoone'
	)
	parser.add_argument(
		'-ntoken', '--NumToken', default=10, type=int, 
		help="# of tokens in each sequence"
	)
	parser.add_argument(
		'-ntrain', '--NumTrain', default=10000, type=int, 
		help="# of training seqs"
	)
	parser.add_argument(
		'-ndev', '--NumDev', default=1000, type=int, 
		help="# of dev seqs"
	)
	parser.add_argument(
		'-ntest', '--NumTest', default=1000, type=int, 
		help="# of test seqs"
	)
	parser.add_argument(
		'-nfuture', '--NumFuture', default=1000, type=int, 
		help="# of seqs for future use"
	)
	parser.add_argument(
		'-tp', '--TrackPeriod', default=10, type=int,
		help='# of sequences for each checkpoint'
	)
	parser.add_argument(
		'-nm', '--NoiseMode', type=str, 
		default='multinomial', choices=['multinomial'], 
		help='way to sample event types, must be multinomial for exact sampling '
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
		['Config', 'conf'],
		['DimLSTM', 'dim'],
		['UseGPU', 'gpu'], 
		['Seed', 'seed'],
	]
	folder_name = list()
	for arg_name, rename in args_used_in_name:
		folder_name.append('{}-{}'.format(rename, dict_args[arg_name]))
	folder_name = '_'.join(folder_name)
	folder_name = f'{folder_name}_datagen_{id_process}'
	return folder_name

def main():

	dict_args = get_args()
	aug_args_with_log(dict_args)
	if '' in dict_args:
		del dict_args['']
	gen(dict_args)


if __name__ == "__main__": main()