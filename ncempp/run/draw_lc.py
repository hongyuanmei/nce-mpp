import argparse
import datetime
import os
import time

import numpy
import torch
from tqdm import tqdm

from nce_point_process.eval.draw import Drawer


def draw_lc(args):
	"""
	draw learning curves given restored results 
	"""
	drawer = Drawer(args=args)
	drawer.draw_lc()

def get_args():
	"""
	Retrieves arguments from command line
	"""
	parser = argparse.ArgumentParser(description='draw figures ...')
	parser.add_argument(
		'-ds', '--Dataset', required=True, type=str, help='e.g. smallnhp'
	)
	parser.add_argument(
		'-cn', '--CSVName', required=True, type=str, 
		help='prefix name of meta and log csv'
	)
	parser.add_argument(
		'-yl', '--YLow', type=float, default=-numpy.inf, 
		help='y value on graph can not be lowr than this given value'
	)
	parser.add_argument(
		'-ih', '--IntenHigh', type=float, default=numpy.inf, 
		help='# inten on graph can not be higher than this given value'
	)
	parser.add_argument(
		'-th', '--TimeHigh', type=float, default=numpy.inf, 
		help='wall-clock time on graph can not be higher than this given value'
	)
	parser.add_argument(
		'-ct', '--ContinualTrain', type=int, default=0, choices=[0,1], 
		help='0==False(default);1==True : draw continual training strategy?'
	)
	parser.add_argument(
		'-hm', '--HybridMultiplier', type=int, default=1, 
		help='M of the MLE run that we want to use for continual hybrid training'
	)
	parser.add_argument(
		'-wp', '--WaitPeriod', type=int, default=10, 
		help='# of tracked validations before we see it as converged'
	)
	#parser.add_argument(
	#	'-ann', '--Annotate', type=str, default='2', 
	#	help="where to annotate? 1/ann of the curve..., can be 3,2,3,..."
	#)
	parser.add_argument(
		'-ub', '--UseBatch', type=int, default=1, choices=[0,1], 
		help="1==True(default);0==False : show batchsize in curve annotation?"
	)
	parser.add_argument(
		'-prefix', '--Prefix', type=str, default='', 
		help='prefix of the figure pdf name'
	)
	parser.add_argument(
		'-inter', '--Inter', type=int, default=10, 
		help="# of points to interpolate for smoothing"
	)
	parser.add_argument(
		'-loc', '--Location', type=int, default=3, 
		help="loc of legends"
	)
	parser.add_argument(
		'-tes', '--TextSize', type=int, default=10, 
		help="font size of text"
	)
	parser.add_argument(
		'-tis', '--TickSize', type=int, default=10, 
		help="font size of tick"
	)
	parser.add_argument(
		'-les', '--LegendSize', type=int, default=10, 
		help="font size of legend"
	)
	parser.add_argument(
		'-las', '--LabelSize', type=int, default=10, 
		help="font size of label"
	)
	parser.add_argument(
		'-rp', '--RootPath', type=str, default='../../',
		help='root path of project'
	)
	args = parser.parse_args()
	return vars(args)

def aug_args_with_log(dict_args):
	"""
	compose path to folder of logs
	"""
	root_path = os.path.abspath(dict_args['RootPath'])
	path = os.path.join(root_path, 'logs', dict_args['Dataset'])
	dict_args['PathSave'] = path
	dict_args['PathMetaCSV'] = os.path.join(path, f"{dict_args['CSVName']}_meta.csv")
	dict_args['PathLogCSV'] = os.path.join(path, f"{dict_args['CSVName']}_log.csv")

def main():

	dict_args = get_args()
	aug_args_with_log(dict_args)
	if '' in dict_args:
		del dict_args['']
	draw_lc(dict_args)


if __name__ == "__main__": main()