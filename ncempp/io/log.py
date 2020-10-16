import torch
import numpy
import os
import fnmatch
import csv
import heapq

class LogWriter(object):

	def __init__(self, path, args):
		if '' in args:
			del args['']
		self.path = path
		self.args = args
		with open(self.path, 'w') as f:
			f.write("Training Log\n")
			f.write("Specifications\n")
			for argname in self.args:
				f.write("{} : {}\n".format(argname, self.args[argname]))
			f.write("Checkpoints:\n")

	def checkpoint(self, to_write):
		with open(self.path, 'a') as f:
			f.write(to_write+'\n')


class LogReader(object):

	def __init__(self, path):
		self.path = path
		with open(self.path, 'r') as f:
			self.doc = f.read()

	def isfloat(self, str):
		try:
			float(str)
			return True
		except ValueError:
			return False

	def cast_type(self, str):
		res = None
		if str.isdigit():
			res = int(str)
		elif self.isfloat(str):
			res = float(str)
		elif str == 'True' or str == 'False':
			res = True if str == 'True' else False
		else:
			res = str
		return res

	def finished(self): 
		return 'training finished' in self.doc

	def get_args(self):
		block_args = self.doc.split('Specifications\n')[-1]
		block_args = block_args.split('Checkpoints:\n')[0]
		lines_args = block_args.split('\n')
		res = dict()
		for line in lines_args:
			items = line.split(' : ')
			res[items[0]] = self.cast_type(items[-1])
		return res

	def get_best(self):
		block_score = self.doc.split('Checkpoints:\n')[-1]
		lines_score = block_score.split('\n')
		best_score = ''
		best_batch = ''
		best_epoch = ''
		for line in lines_score:
			if 'Current best loglik is' in line:
				best_score = line.split('Current best loglik is ')[-1]
				best_score = best_score.split(' (updated at')[0]
				best_batch = line.split('(updated at ')[-1]
				best_batch = best_batch.split('-th batch')[0]
				best_epoch = line.split('-th batch of ')[1]
				best_epoch = best_epoch.split('-th epoch')[0]
				best_score = self.cast_type(best_score)
				best_batch = self.cast_type(best_batch)
				best_epoch = self.cast_type(best_epoch)
		return best_score, best_batch, best_epoch

	def get_times_and_iteration(self):
		block_times = self.doc.split('Checkpoints:\n')[1]
		line_times = block_times.split('\n')
		log_liks = []
		iterations = []
		times = [0]
		for i_line, line in enumerate(line_times):
			if 'eval:' in line:
				line = line.split(' ')
				loglik = self.cast_type(line[-1])
				iteration = self.cast_type(line[1])

				if i_line < len(line_times) - 1:
					line_time = line_times[i_line + 1]

				time = line_time.split(' ')[0]
				time = self.cast_type(time)

				log_liks.append(loglik)
				iterations.append(iteration)
				times.append(time + times[-1])
		return log_liks, iterations, times[1:]

	def get_vals(self): 
		block_vals = self.doc.split('Checkpoints:\n')[-1]
		lines = block_vals.split('\n')
		total_len = len(lines)
		rst = list()
		for i, line in enumerate(lines): 
			if 'eval: ' in line and i < total_len - 2: 
				"""
				track loglik, train time, and # of intensities computed
				if any of them is missing, we discard all of them
				"""
				line = line.split(' ')
				loglik = self.cast_type(line[-1])
				i_iter = self.cast_type(line[1])
				# i : eval : ...
				assert 'seconds for training since last validation' in lines[i + 1]
				t = self.cast_type(lines[i + 1].split(' ')[0]) 
				assert '# of intensities computed = ' in lines[i + 2]
				c = self.cast_type(lines[i + 2].split(' ')[-4])
				rst.append( f'iter{i_iter}|{t}|{c}|{loglik}' )
		return rst

	def get_all(self):
		rst = self.get_args()
		i = len(rst)
		rst['Values'] = self.get_vals()
		best_score, best_batch, best_epoch = self.get_best()
		rst['_best_score'] = best_score
		rst['_best_batch'] = best_batch
		rst['_best_epoch'] = best_epoch
		return rst

class LogBatchReader(object):

	def __init__(self, path, name):
		self.path = path
		self.name = name
		"""
		given domain path, find all the log folders and get their results
		"""
		self.all_readers = list()
		for root, dirs, files in self.walklevel(path): 
			for file_name in fnmatch.filter(files, 'log.txt'): 
				full_path = os.path.join(root, file_name)
				self.all_readers.append(LogReader(full_path))

	def walklevel(self, some_dir, level=1):
		some_dir = some_dir.rstrip(os.path.sep)
		assert os.path.isdir(some_dir)
		num_sep = some_dir.count(os.path.sep)	
		for root, dirs, files in os.walk(some_dir):
			yield root, dirs, files
			num_sep_this = root.count(os.path.sep)
			if num_sep + level <= num_sep_this:
				del dirs[:]

	def write_csv(self):
		"""
		NOTE : it is possible that the Values field is TOO long 
		so microsoft excel may not show it 
		in that case, we can create two sheets: 
		(1) each row is a run, each col is a field, not including Values 
		(2) each col is a run---all its values...
		"""
		path_save_meta = os.path.join(self.path, f"{self.name}_meta.csv")
		print(f"writing {path_save_meta}")
		names_field = self.make_header()
		c = 0
		i_log_values = dict()
		max_log = -1
		"""
		write meta info
		"""
		with open(path_save_meta, 'w') as file_csv:
			writer_csv = csv.DictWriter( file_csv, fieldnames = names_field )
			writer_csv.writeheader()
			for i_log, log_reader in enumerate(self.all_readers):
				dict_rst = log_reader.get_all()
				i_log_values[i_log] = dict_rst['Values']
				if len(dict_rst['Values']) > max_log: 
					max_log = len(dict_rst['Values'])
				self.process_info(dict_rst, i_log)
				writer_csv.writerow(dict_rst)
				if log_reader.finished(): 
					c += 1 
		print(f"meta csv finished")
		"""
		write log details
		"""
		names_field = sorted(i_log_values.keys())
		path_save_log = os.path.join(self.path, f"{self.name}_log.csv")
		print(f"writing {path_save_log}")
		with open(path_save_log, 'w') as file_csv: 
			writer_csv = csv.DictWriter( file_csv, fieldnames = names_field )
			writer_csv.writeheader()
			for i_row in range(max_log): 
				dict_row = dict()
				for i_log in names_field: 
					if i_row < len(i_log_values[i_log]): 
						dict_row[i_log] = i_log_values[i_log][i_row]
					else: 
						dict_row[i_log] = ''
				writer_csv.writerow(dict_row)
		print(f"log csv finished")
		print(f"{c} out of {len(self.all_readers)} organized logs finished training")

	def make_header(self):
		names_field = set()
		for i_log, log_reader in enumerate(self.all_readers):
			dict_rst = log_reader.get_all()
			self.process_info(dict_rst, i_log)
			for k in dict_rst:
				names_field.add(k)
		names_field.add('_index')
		names_field.add('_keep')
		names_field = list(names_field)
		hq = []
		to_show = self.get_to_show()
		"""
		header is sorted based on to_show value
		"""
		rst = []
		for nf in names_field: 
			heapq.heappush(hq, ( to_show[nf] , nf ))
		while hq: 
			rst.append(heapq.heappop(hq)[1])
		return rst

	def process_info(self, x, i):
		# or to delete? 
		to_show = self.get_to_show()
		to_del = []
		for k in x: 
			if k not in to_show: 
				to_del.append(k)
		for k in to_del: 
			del x[k]
		x['_index'] = i
		x['_keep'] = 1

	def get_to_show(self): 
		to_show = {
			'TrainMethod' : 0, 
			'Model': 1, 
			'DimLSTM' : 2, 'Config' : 3, 
			'SizeBatchCompute' : 4, 'SizeBatchUpdate': 5, 
			'TrackPeriod' : 6, 
			'TrainRatio' : 7, 'DevRatio' : 8, 
			'MaxEpoch' : 9, 'MCSample' : 10, 'NoiseFolder' : 11, 
			'ReDrawProb': 12, 
			'NoiseProcess' : 13, 'NoiseType' : 14, 'NoiseMode' : 15, 
			'OverRate' : 16, #'LearnRate', 
			'Seed' : 17, 'UseGPU' : 18, 'ID' : 19, 
			'_best_score' : 20, '_best_batch' : 21, '_best_epoch' : 22, 
			'_index' : 23, 
			#'Values' : 21, 
			'_keep': 24, 
		}
		return to_show
