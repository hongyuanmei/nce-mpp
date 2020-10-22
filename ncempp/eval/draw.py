import os
import csv
import numpy

from ncempp.io.log import LogReader
import matplotlib.pyplot as plt
#from scipy.interpolate import make_interp_spline, BSpline
from scipy import interpolate


class Drawer(object): 
	"""
	draw figures given stored results 
	"""
	def __init__(self, *, args): 
		self.args = args
		self.figure_tags = ['inten', 'time']
		self.figure_labels = {
			'inten': ['# of intensities computed', 'log-likelihood'], 
			'time': ['wall-clock time', 'log-likelihood'], 
		}
		self.index = {
			'inten': 2, 'time': 1, 'll': 3 # ll : loglikelihood
		}
		self.color = {
			'mle': 'red', 
			'lse': 'orange', 
			'nce_frac': 'blue', 'nce_async': 'blue', 'nce_sync': 'blue', 
			'nce_binary': 'green', 
			'hybrid': 'blueviolet', 
			'high': 'red', 
		}
		self.range = {
			'y_low': args['YLow'], 
			'inten_high': args['IntenHigh'], 
			'time_high': args['TimeHigh']
		}
		"""
		collect annotation positions
		"""
		with open(os.path.join(self.args['PathSave'], 'anns.txt'), 'r') as f: 
			anns = f.read().split('\n')
			if anns[-1] == '': 
				anns.pop(-1)
		self.ann = {}
		for l in anns: 
			method_m_b, ann_frac = l.split(' ')
			ann_frac = float(ann_frac)
			self.ann[method_m_b] = ann_frac

	def draw_lc(self): 
		"""
		get all results
		"""
		all_runs = list()	
		"""
		read meta data
		"""
		with open(self.args['PathMetaCSV']) as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			for row in readCSV:
				all_runs.append(row)
		"""
		read header 
		"""
		header = all_runs.pop(0)
		method_i = header.index('TrainMethod')
		mc_i = header.index('MCSample')
		batch_i = header.index('SizeBatchUpdate')
		np_i = header.index('NoiseProcess')
		nt_i = header.index('NoiseType')
		logidx_i = header.index('_index')
		keep_i = header.index('_keep')
		vals_i = -1
		"""
		read log data
		"""
		log_rows = list()
		with open(self.args['PathLogCSV']) as csvfile: 
			readCSV = csv.reader(csvfile, delimiter=',')
			for row in readCSV: 
				log_rows.append(row)
		indices = log_rows.pop(0) # a list 
		vals = list()
		for idx in indices: 
			vals.append( list() )
		for row in log_rows: 
			for i, v in enumerate(row): 
				if v != '': 
					assert '|' in v, "not a track?"
					vals[i].append(v)
		"""
		match logs with runs
		"""
		for run in all_runs: 
			logidx = run[logidx_i]
			run.append( vals[indices.index(logidx)] ) 
			# that's why its index is -1
		"""
		prune runs with keep=0
		"""
		draw_runs = list()
		for run in all_runs: 
			if int(run[keep_i]) == 1: 
				draw_runs.append(run)
		"""
		draw curves
		"""
		for figure_tag in self.figure_tags: 
			"""
			for each tag (i.e., inten, time, etc...) : 
			# inten : vs. # of intensities computed 
			# time : vs. wall-clock time 
			find x_i and y_i : index of x and y in 'Values'
			"""
			x_i = self.index[figure_tag]
			y_i = self.index['ll']
			xlabel = self.figure_labels[figure_tag][0]
			ylabel = self.figure_labels[figure_tag][1]
			x_high = self.range[f'{figure_tag}_high']
			y_low = self.range['y_low']
			fig_name = f"lc_tag={figure_tag}_csv={self.args['CSVName']}_xhigh={x_high:.0f}_ylow={y_low:.0f}.pdf"
			if self.args['Prefix'] != '': 
				fig_name = self.args['Prefix'] + '_' + fig_name
			self.draw_one_figure(
				draw_runs, method_i, mc_i, batch_i, 
				np_i, nt_i, vals_i, x_i, y_i, 
				x_high, y_low, xlabel, ylabel, fig_name
			)

	def draw_one_figure(self, 
		all_runs, method_i, mc_i, batch_i, 
		np_i, nt_i, vals_i, x_i, y_i, 
		x_high, y_low, xlabel, ylabel, fig_name): 
		"""
		NOTE : same hyperparam may have different runs with different random seed 
		"""
		curve_with_band = dict()
		
		for run in all_runs: 
			"""
			for this curve (i.e. this run of training method)
			find its color and label based on training method
			"""
			#print(run)
			method = run[method_i]
			color = self.color[method]
			label = self.get_prefix(method)
			m = run[self.get_mi(method, [mc_i, np_i, nt_i])]
			batch = run[batch_i]
			annotate = self.get_annotate(method, m, batch)
			vals = run[vals_i]
			# get curve
			x, y = list(), list()
			for v in vals: 
				v = v.split('|')
				x.append(float(v[x_i]))
				y.append(float(v[y_i]))
			x = numpy.cumsum(x)
			"""
			smooth y : 
			rigid and wavy y doesn't make sense---why? 
			y is dev-performance---if we find it drop, we'll just use prev saved model 
			so actual performance of entire model never goes down
			"""
			for i in range(len(y) - 1): 
				if y[i+1] < y[i]: 
					y[i+1] = y[i]
			y = numpy.array(y)
			"""
			save curves to draw
			"""
			if f'{method}_{m}_{batch}' not in curve_with_band:  
				curve_with_band[f'{method}_{m}_{batch}'] = {
					'method': method, 'm': m, 'batch': batch, 
					'color': color, 'label': label, 
					'annotate': annotate, 
					'x_all': [], 'y_all': []
				}
			
			curve_with_band[f'{method}_{m}_{batch}']['x_all'].append(x)
			curve_with_band[f'{method}_{m}_{batch}']['y_all'].append(y)

		"""
		NOTE : aggregate random runs for each hyper-param
		"""
		inter_num = self.args['Inter'] # for interpolation, may be a hyper-param
		smooth_kind = 'linear'
		highest = dict() # track highest value achieved by each method
		x_cat, y_cat, y_min_cat, y_max_cat = \
			None, None, None, None # to save the extrapolated MLE curve
		
		for method_m_b, info in curve_with_band.items(): 

			x = numpy.mean(self.array(info['x_all']), axis=0)
			y = numpy.mean(self.array(info['y_all']), axis=0)
			y_min = numpy.min(self.array(info['y_all']), axis=0)
			y_max = numpy.max(self.array(info['y_all']), axis=0)

			"""
			further smooth : use poly
			first make a smoother function using current x and y
			"""
			"""
			make new x
			"""
			x_new = numpy.linspace(x.min(), x.max(), inter_num * len(x))
			smooth = interpolate.interp1d(x, y, kind=smooth_kind)
			#spl = make_interp_spline(x, y, k=3)  # type: BSpline
			#y_smooth = spl(x_new)
			y_new = smooth(x_new)
			smooth = interpolate.interp1d(x, y_min, kind=smooth_kind)
			#spl = make_interp_spline(x, y, k=3)  # type: BSpline
			#y_smooth = spl(x_new)
			y_min_new = smooth(x_new)
			smooth = interpolate.interp1d(x, y_max, kind=smooth_kind)
			#spl = make_interp_spline(x, y, k=3)  # type: BSpline
			#y_smooth = spl(x_new)
			y_max_new = smooth(x_new)

			"""
			only draw values within range
			"""
			x, y, y_min, y_max = list(), list(), list(), list()
			for i in range(len(x_new)): 
				if x_new[i] <= x_high and y_new[i] >= y_low - 0.1 * numpy.abs(y_low): 
					"""
					NOTE : we over-draw to make band look better
					in the end, we will crop out the over-drawn curve
					"""
					x.append(x_new[i])
					y.append(y_new[i])
					y_min.append(y_min_new[i])
					y_max.append(y_max_new[i])

			info['x'] = x
			info['y'], info['y_min'], info['y_max'] = y, y_min, y_max
			
			info['x_entire'] = x_new
			info['y_entire'], info['y_min_entire'], info['y_max_entire'] = \
				y_new, y_min_new, y_max_new
			
			"""
			track highest (mean) y for this method 
			"""
			method = info['method']
			if method_m_b not in highest: 
				highest[method_m_b] = []
			highest[method_m_b] += [numpy.max(y_new)]
			"""
			track a MLE curve for concatenation
			track the entire curve 
			even though some parts may not be shown on its own curve in this window
			those parts may be shown on the corresponding NCE curves!!!
			"""
			if info['method'] == 'mle' and abs(float(info['m']) - float(self.args['HybridMultiplier'])) < 1e-3: 
				x_cat, y_cat, y_min_cat, y_max_cat = \
					info['x_entire'], info['y_entire'], info['y_min_entire'], info['y_max_entire']
		
		"""
		NOTE : okay, we have all the curves to draw, now we draw them
		"""
		fig, ax = plt.subplots()
		used = set() # track used labels
		drawn = dict() # track drawn part of each curve... useful for concat MLE to NCE
		alpha = 0.2
		linewidth = 0.5
		linewidth2 = 1.0
	
		for method_m_b, info in curve_with_band.items(): 

			method = info['method']
			m = info['m']
			batch = info['batch']
			color = info['color']
			label = info['label']
			annotate = info['annotate']
			"""
			plot 
			"""
			x, y, y_min, y_max = info['x'], info['y'], info['y_min'], info['y_max']
			#print(f"\n\n check len")
			#for xlist in info['x']: 
			#	print(f"len = {len(xlist)}")
			plt.plot(
				x, y, ls='-', lw=linewidth, 
				color=color, label=label if label not in used else '')
			plt.fill_between(
				x, y_min, y_max, color=color, alpha=alpha)
			used.add(label) # track used labels
			
			extrapolated = False 
			if len(x) > 0 and x[-1] < x_high: 
				extrapolated = True 
				"""
				extrapolate to the end
				"""
				x_extra = numpy.linspace(x[-1], x_high, inter_num)
				y_extra = numpy.linspace(max(y), max(y), inter_num)
				y_min_extra = numpy.linspace(max(y_min), max(y_min), inter_num)
				y_max_extra = numpy.linspace(max(y_max), max(y_max), inter_num)
				plt.plot(x_extra, y_extra, ls='-', lw=linewidth, color=color)
				plt.fill_between(x_extra, y_min_extra, y_max_extra, color=color, alpha=alpha)
			"""
			concat MLE : choose where curve stops increasing ... 
			if it keeps increasing within this window, then choose boundary
			a good place to concat MLE curve is where NCE started converging
			"""
			i = 0 
			i_cat = None 
			while i_cat is None and i < len(x): 
				i_giveup = i + self.args['WaitPeriod'] * inter_num
				if i_giveup >= len(x): 
					i_giveup = len(x) - 1
				if abs(y[i] - y[i_giveup]) < 1e-6: 
					i_cat = i 
				i += 1
			if i_cat is None: 
				i_cat = len(x) - 1
			"""
			annotate : manually define how to place annotations
			"""
			x_ann, y_ann = list(x), list(y) 
			if extrapolated: 
				inter_ann = int( len(x) * (x_high - x[-1]) / (x[-1] - x[0]) )
				x_ann += list(numpy.linspace(x[-1], x_high, inter_ann))
				y_ann += list(numpy.linspace(max(y), max(y), inter_ann))
			if len(x) > 0: 
				x_idx, y_idx = self.find_where_to_annotate(x_ann, y_ann, method_m_b)
				plt.text(
					x_ann[x_idx], y_ann[y_idx], annotate, color='black', 
					fontsize=self.args['TextSize'], 
					horizontalalignment='center', 
					verticalalignment='center'
				)
				# OLD way: annotate where we concate MLE curve
				#plt.text(x[i_cat], y[i_cat], annotate, color='black')
			"""
			track drawn part of this curve
			"""
			drawn[f'{method}||{m}||{annotate}'] = {
				'xy': [x, y, y_min, y_max], 
				'converge': i_cat
			}
		
		"""
		track the highest value 
		to check if we need to concat NCE curves
		"""
		highest_method = None
		highest_y = -numpy.inf
		for method_m_b in highest: 
			if numpy.mean(highest[method_m_b]) > highest_y: 
				highest_y = numpy.mean(highest[method_m_b])
				highest_method = method
		"""
		concat NCE curves with MLE curves : to show the hybrid training strategies 
		"""
		if highest_method == 'mle' and self.args['ContinualTrain']: 
			label = 'NCE->MLE'
			"""
			when NCE hasn't improved yet for several rounds of evaluation
			"""
			assert x_cat is not None and y_cat is not None, f"no MLE with rho={self.args['HybridMultiplier']}?!"
			for method_m_annotate in drawn: 
				method, m, annotate = method_m_annotate.split('||')
				x, y, y_min, y_max = drawn[method_m_annotate]['xy']
				if 'nce' in method and len(x) > 0:
					"""
					for this curve, we find its steady point
					if no steady point (i.e., keep increasing), then its right most point
					if this point is at right boundary -> ignore it 
					if not, concat MLE curve to it
					""" 
					"""
					then what's a steady point? 
					it is where training stops improving 
					precisely, not improving fast enough, 
					it may be wavy, but not worth to wait longer cuz it is too slow...
					technically, it is the first point where 
					waiting for the wait period doesn't really help!
					"""
					i_cat = drawn[method_m_annotate]['converge']
					y_converge = y[i_cat]
					if x[i_cat] < x_high: 
						"""
						still within show range 
						find the concat part of MLE
						1. find y in MLE that matches y_max 
						2. find its index i 
						3. adjust x values to shift segment
						"""
						y_keep = y_cat >= y_converge
						y_seg = y_cat[y_keep]
						y_min_seg = y_min_cat[y_keep]
						y_max_seq = y_max_cat[y_keep]
						x_seg = x_cat[y_keep]
						x_seg -= x_seg[0]
						x_seg += x[i_cat]
						x_keep = x_seg <= x_high
						x_seg = x_seg[x_keep]
						y_seg = y_seg[x_keep]
						y_min_seg = y_min_seg[x_keep]
						y_max_seq = y_max_seq[x_keep]

						plt.plot(
							x_seg, y_seg, ls='-', lw=linewidth, 
							color=self.color['hybrid'], alpha=1.0, 
							label=label if label not in used else ''
						)
						plt.fill_between(
							x_seg, y_min_seg, y_max_seq, 
							color=self.color['hybrid'], alpha=alpha
						)
						used.add(label) # track used labels
		"""
		plot the highest value : dashed line - - - 
		color indicates achieved by which method 
		this has to be done in the end s.t. x_min and x_max are final
		"""
		#highest_color = self.color[highest_method]
		highest_color = self.color['high']
		x_min, x_max = plt.xlim() 
		plt.hlines(
			highest_y, x_min, x_max, 
			colors=highest_color, linestyles='dashed', linewidth=linewidth2)
		"""
		label and legend
		"""
		"""
		crop out-of-view curves
		"""
		bottom, _ = plt.ylim() 
		if bottom < y_low: 
			plt.ylim(bottom=y_low)
		plt.xlabel(xlabel, fontsize=self.args['LabelSize'])
		plt.ylabel(ylabel, fontsize=self.args['LabelSize'])
		plt.xticks(fontsize=self.args['TickSize'])
		plt.yticks(fontsize=self.args['TickSize'], rotation=0)
		leg = plt.legend(
			loc=self.args['Location'], 
			prop={'size': self.args['LegendSize']}
		) # or 'best'?
		for legobj in leg.legendHandles: 
			legobj.set_linewidth(4.0)
			legobj.set_alpha(0.5)
		plt.tight_layout()
		plt.savefig(
			os.path.join(self.args['PathSave'], fig_name), 
			format='pdf'
		)

	def get_prefix(self, method): 
		if method == 'mle': 
			return 'MLE'
		elif method == 'lse': 
			return 'LSE'
		elif method == 'nce_binary': 
			return 'b-NCE'
		elif 'nce' in method: 
			return 'NCE'
		else: 
			raise Exception(f'Unknown method : {method}')

	def get_mi(self, method, x): 
		# items in x : in order of : mc_i, np_i, nt_i
		if method == 'mle': 
			return x[0]
		elif method == 'lse': 
			return x[0]
		elif method == 'nce_frac' or method == 'nce_async' or method == 'nce_binary': 
			return x[1]
		elif method == 'nce_sync': 
			return x[2]
		else: 
			raise Exception(f'Unknown method : {method}')

	def get_annotate(self, method, m, batch): 
		rst = ''
		if method == 'mle': 
			rst += f'\u03C1={m}'
		elif method == 'lse': 
			rst += f'\u03C1={m}'
		elif method == 'nce_frac' or method == 'nce_async' or method == 'nce_sync' or method == 'nce_binary': 
			rst += f'M={m}'
		else: 
			raise Exception(f'Unknown method : {method}')
		if self.args['UseBatch']: 
			rst += f'\nB={batch}'
		return rst

	def find_where_to_annotate(self, x, y, method_m_b):
		idx = int( len(x) / self.ann[method_m_b] )
		if idx >= len(x): 
			idx = len(x) - 1
		return idx, idx

	def array(self, x): 
		# make array with a list of vectors with diff lens
		maxlen = -1 
		maxy = None
		for y in x: 
			if len(y) > maxlen: 
				maxlen = len(y)
				maxy = y 
		for i, y in enumerate(x): 
			if len(y) < maxlen: 
				newy = numpy.zeros(maxlen)
				newy[:len(y)] = y[:]
				dy = y[-1] - maxy[len(y)-1]
				newy[len(y):] = maxy[len(y):] + dy
				x[i] = newy
		return numpy.array(x)
				