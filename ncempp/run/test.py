import argparse
import datetime
import os
import psutil
import time

import numpy
import torch

from nce_point_process.run.tester import Tester

#@profile
def test(args):
	# track memory
	#process = psutil.Process(os.getpid())
	#print(f'memory starts at : {process.memory_info().rss/1000000:.3f} MB')
	"""
	train model with chosen method
	"""
	manager = Tester(args=args)
	#print('prepare model and testing method')
	#manager.prepare_testing()
	#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')
	print(f"load {args['Split']} data")
	manager.load_data( [ args['Split'] ] )
	#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')
	print('process data (into tensors) ')
	manager.process_data()
	#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')

	print('start testing ...')

	dev_times = list()

	print(f"\ntesting on log-likelihood")
	tic = time.time()
	obj, num = manager.eval_mle_epoch(args['Split'])
	avg_obj = obj / num
	toc = time.time()
	dev_times.append(toc - tic)
	#print(f"after validating one epoch")
	#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')

	message = f"eval: loglik (per token) is {avg_obj:.4f}"
	manager.print_or_buffer(message)
	message = f'{dev_times[-1]:.4f} seconds for eval' # useful to track :-) 
	manager.print_or_buffer(message)
	# for dev time

	print(f"\ntesting on prediction accuracy")
	tic = time.time()
	se, en, num = manager.eval_predict_epoch(args['Split'])
	mse = se / num
	er = en / num 
	toc = time.time()
	dev_times.append(toc - tic)
	#print(f"after validating one epoch")
	#print(f'memory afterwards : {process.memory_info().rss/1000000:.3f} MB')

	message = f"eval: mse (per token) is {mse:.4f}"
	message += f"\neval: rmse (per token) is {numpy.sqrt(mse):.4f}"
	message += f"\neval: error rate (per token) is {100*er:.2f} %"
	manager.print_or_buffer(message)
	message = f'{dev_times[-1]:.4f} seconds for eval' # useful to track :-) 
	manager.print_or_buffer(message)

	message = f'\ntesting finished' 
	# important indicator for if training finished
	# on a server like MARCC, 
	# sometimes training terminated but not finished
	manager.print_or_buffer(message)
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
		'-mf', '--ModelFolder', type=str,
		help='folder name of saved model and log containing useful info like config file name'
	)
	parser.add_argument(
		'-sp', '--Split', type=str, 
		help='data split used to eval'
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
		'-r', '--Ratio', default=1.0, type=float, 
		help='ratio of dev/test data to use for this run'
	)
	parser.add_argument(
		'-mc', '--MCSample', default=1, type=float,
		help='# of Monte-Carlo samples per actual event in MLE'
	)
	parser.add_argument(
		'-or', '--OverRate', default=1.0, type=float,
		help='over sampling rate in thinning algorithm (currently dummy)'
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

	capsplit = dict_args['Split'][0].upper() + dict_args['Split'][1:]
	dict_args[f"{capsplit}Ratio"] = dict_args['Ratio']

	#folder_name = get_foldername(dict_args, id_process)
	#print(folder_name)

	path = os.path.join(root_path, 'logs', dict_args['Dataset'])
	#path_log = os.path.join(path, folder_name)
	#os.makedirs(path_log)

	#file_log = os.path.join(path_log, 'log.txt')
	#file_model = os.path.join(path_log, 'saved_model.pkl') 
	# some sys may think a file with no extension is an executable : bad

	#dict_args['PathLog'] = file_log
	#dict_args['PathModel'] = file_model
	dict_args['PathSave'] = path
	#dict_args['PathRun'] = path_log

def main():

	dict_args = get_args()
	aug_args_with_log(dict_args)
	if '' in dict_args:
		del dict_args['']
	test(dict_args)


if __name__ == "__main__": main()