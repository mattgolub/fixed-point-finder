'''
Timer.py
Written for Python 3.6.9
@ Matt Golub, August 2018
Please direct correspondence to mgolub@cs.washington.edu
'''

import numpy as np
import time

class Timer(object):
	'''Class for profiling computation time.

	Example usage, bare-bones:

		t = Timer()
		t.start()
		# Do your business
		t.print()

		-->	Total time: 0.57s:

		# This will work regardless of do_retrospective.

	Example usage, using prospective splits (default):

		# Build a timer object to profile three tasks.
		t = Timer(3, do_retrospective=False)

		t.start()           	# Start the timer (optional).

		t.split('Task 1')   	# Measure time taken for task 1.
		run_task_1()        	# Run task 1 of 3.

		t.split('Task 2')   	# Measure time taken for task 2.
		run_task_2()        	# Run task 2 of 3.

		t.split('Task 3')   	# Measure time taken for task 3.
		run_task_3()        	# Run task 3 of 3.

		t.stop()				# Stop the timer
								# (required to see split 3 time)

		t.print()            	# Print profile of timing.

		--> Total time: 16.00s:
		-->     Task 1: 2.00s (12.5%)
		-->     Task 2: 8.00s (50.0%)
		-->     Task 3: 6.00s (37.5%)

		# Note that split() returns None with prospective splits, as it
		# cannot yet know the timing of the upcoming task.

	Example usage, using retrospective splits:

		# Build a timer object to profile three tasks.
		t = Timer(3, do_retrospective=True)

		t.start()           	# Start the timer (required).

		run_task_1()        	# Run task 1 of 3.
		t1 = t.split('Task 1')	# Measure time taken for task 1.

		run_task_2()        	# Run task 2 of 3.
		t2 = t.split('Task 2')	# Measure time taken for task 2.

		run_task_3()        	# Run task 3 of 3.
		t3 = t.split('Task 3')	# Measure time taken for task 3.

		t.print()            	# Print profile of timing.

		--> Total time: 16.00s:
		-->     Task 1: 2.00s (12.5%)
		-->     Task 2: 8.00s (50.0%)
		-->     Task 3: 6.00s (37.5%)

	Outward facing usage is identical when the number of tasks to be timed is
	not specified from the outset. However, this prevents Timer from
	preallocating all required memory ahead of start(). Thus, Timer needs to
	allocate (very small) amounts of memory during timing, which may bias total
	time in unpredictable ways when total time is fast relative to memory
	allocations.
	'''
	_s_per_min = 60
	_s_per_hr = _s_per_min*60
	_s_per_day = _s_per_hr*24
	_s_per_yr = _s_per_day*365

	def __init__(self,
		n_splits=10,
		do_retrospective=False,
		do_print_single_line=False,
		n_indent=0,
		name='Total',
		verbose=False):
		'''Builds a timer object.

		Args:
			n_splits (optional): int specifying the total number of splits to
			be timed. If provided, memory can be preallocated, which may
			prevent unpredictable allocation artifacts during profiling.
			Over-allocating (i.e., requesting more splits than ultimately are
			invoked) will not impact profiling (though may have memory
			implications in extreme applications). Default: 10.

			do_retrospective (optional): bool. Default: False.

			do_print_single_line (optional): bool. Default: False.

			n_indent (optional): int specifying the number of indentation
			to prefix into print statements. Useful when utilizing multiple
			timer objects for profile nested code. Default: 0.

			name (optional): string specifying name for this timer, used only
			when printing updates. Default: 'Total'.

			verbose (optional): Bool indicating whether to print when
			allocating new splits beyond the initial n_splits (which may
			indicate biased timing for very fast splits). Default: False.

		Returns:
			None.
		'''

		assert n_splits >= 0, ('n_splits must be >= 0, but was %d' % n_splits)

		assert n_indent >= 0,('n_indent must be >= 0, but was %d' % n_indent)

		self.name = name
		self.do_retrospective = do_retrospective
		self.verbose = verbose

		'''Pre-allocate to avoid having to call append after starting the timer
		(which might incur non-uniform overhead, biasing timing splits).
		If more times are recorded than pre-allocated, lists will append.

		Note that self.times has n+1 elements (to include a start value) while
		self.task_names has n elements.
		'''
		self._empty_split_val = -1.0

		# Preallocate memory
		self._split_starts = [self._empty_split_val for idx in range(n_splits)]
		self._split_stops = [self._empty_split_val for idx in range(n_splits)]
		self._split_names = ['Task %d' % (idx+1) for idx in range(n_splits)]

		self._print_prefix = self._generate_print_prefix(n_indent)
		self._do_print_single_line = do_print_single_line

		self._is_started = False
		self._is_stopped = False

		''' Strategy for memory management:

		n: always represents the current length of
			_split_starts
			_split_stops
			_task_names
		'''
		self._alloc_len = n_splits
		self._idx = -1

	def __call__(self):
		'''Returns the time elapsed since the timer was started.
		   If start() has not yet been called, returns 0.
		'''

		if self._is_stopped:
			return self._stop_time - self._start_time
		if self._is_started:
			return time.time() - self._start_time
		else:
			return 0.0

	def start(self):
		'''Starts the timer. '''

		assert not self._is_stopped, 'Cannot restart a stopped Timer.'

		if self._is_started:
			self._print('Timer has already been started. '
				'Ignoring call to Timer.start()')
		else:
			self._is_started = True
			self._start_time = time.time()
			assert self._idx == -1, 'Inconsistent Timer state.'

			if self.do_retrospective:
				self._start_split()

	def stop(self):
		''' Stops the timer, freezing total and split times. '''



		if self._is_stopped:
			self._print('Timer has already been stopped. '
				'Ignoring call to Timer.stop()')
		else:
			# Fill in final split if it had been started.
			if not self.do_retrospective:
				self._stop_split()

			# Record stop time.S
			self._stop_time = time.time()

	def split(self, name=None, stop=False):
		'''Records and returns the time elapsed for the most recent task.

		Args:
			name (optional): A string describing the most recent task.

		Returns:
			float indicating the split time in seconds.
		'''

		assert not self._is_stopped, \
			'Cannot take a split on a stopped Timer.'

		idx = self._idx # get idx before it's incremented

		if self.do_retrospective:
			'''
			Record split stop.
			Prepare and start next split.
			Return split time.
			'''

			assert self._is_running,\
				'Cannot record split time because Timer was not started.'

			assert self._split_starts[idx] != self._empty_split_val, \
				('Attempting to record split stop with no split start.')

			self._stop_split(name)

			if stop:
				# Avoid allocating a new split if known to be unnecessary.
				self.stop()
			else:
				self._start_split()

			return self._get_split_time(idx)

		else:
			'''
			Record split stop.

			'''

			if self._is_started:
				self._stop_split()
			else:
				self.start()

			self._start_split(name)

			return None

	def get_split(self, name):
		''' Retrieves a previously recorded split time.

		Args:
			name: the string name used to record the split, as previously
			provided in the call: split(name).

		Returns:
			float indicating the split time in seconds.
		'''

		idx = self.split_names.index(name)
		return self._get_split_time(idx)

	def get_splits(self):
		''' Returns a list of all completed split times.
		'''
		idx = 0
		splits = []
		while self._is_split_complete(idx):

			split_name = self._split_names[idx]
			split_time = self._get_split_time(idx)
			splits.append(split_time)
			idx += 1

		return splits

	def disp(self, *args, **kwargs):

		print('Timer.disp() is deprecated and '
		      'will be removed in a future version of Timer.py. '
		      'Use Timer.print(...) instead.')

		self.print(*args, **kwargs)

	def print(self, n_indent=None, do_single_line=None):
		'''Prints the profile of the tasks that have been timed thus far.

		Args:
			None.

		Returns:
			None.
		'''

		if self._is_started:

			total_time = self.print_total_time(
				do_single_line=do_single_line,
				n_indent=n_indent)

			self.print_split_times(total_time, 
				do_single_line=do_single_line,
				n_indent=n_indent)
		else:
			self._print('Timer has not been started.')

	def print_total_time(self, n_indent=None, do_single_line=None):
		# Print total time

		if n_indent is None:
			prefix = self._print_prefix
		else:
			prefix = self._generate_print_prefix(n_indent)

		if do_single_line is None:
			do_single_line = self._do_print_single_line

		total_time = self.__call__()
		print_data = (prefix, self.name, self._format_time(total_time))
		end = '' if do_single_line else '\n'
		print('%s%s time: %s. ' % print_data, end=end)

		return total_time

	def print_split_times(self, total_time,
		n_indent=None,
		do_single_line=None):
		# Print split times for all completed splits

		if n_indent is None:
			prefix = self._print_prefix
		else:
			prefix = self._generate_print_prefix(n_indent)

		if do_single_line is None:
			do_single_line = self._do_print_single_line

		if do_single_line:
			print('[', end='')

		idx = 0
		pct_scale = 100./total_time # for converting to percent of total time

		while self._is_split_complete(idx):

			split_name = self._split_names[idx]
			split_time = self._get_split_time(idx)

			if do_single_line:
				print(' %s: %.1f%% (%s);' %
					(split_name,
					split_time*pct_scale,
					self._format_time(split_time)),
					end='')
			else:
				print('%s\t%.1f%% (%s): %s' %
					(prefix,
					split_time*pct_scale,
					self._format_time(split_time),
					split_name),
					end='\n')

			idx += 1

		if do_single_line:
			print(' ]', end='\n')

	@property
	def total_time(self):
		return self.__call__()

	# ************************************************************************
	# Internal support *******************************************************
	# ************************************************************************

	def _start_split(self, name=None):

		idx = self._prepare_next_split()

		if not self.do_retrospective and name is not None:
			self._split_names[idx] = name

		self._split_starts[idx] = time.time()

	def _stop_split(self, name=None):

		idx = self._idx

		self._split_stops[idx] = time.time()

		if self.do_retrospective and name is not None:
			self._split_names[idx] = name

	@property
	def _is_running(self):
		'''Returns a bool indicating whether or not the timer has been started.
		'''
		return self._is_started and not self._is_stopped

	def _is_split_complete(self, idx):
		''' Returns True if split[idx] has been completed, meaning it has a
		recorded start time and stop time.
		'''

		return idx < self._alloc_len and \
			self._split_starts[idx] != self._empty_split_val and \
			self._split_stops[idx] != self._empty_split_val

	def _get_split_time(self, idx):

		assert self._is_split_complete(idx), \
			('split[%d] is not complete.' % idx)

		return self._split_stops[idx] - self._split_starts[idx]

	def _prepare_next_split(self):
		# This is the only place that _idx and _alloc_len are ever changed.

		assert self._idx == -1 or self._is_split_complete(self._idx),\
			('Cannot prepare split %d because split %d is not complete.' %
			(self._idx+1, self._idx))

		self._idx += 1

		# Ensure safe to write to _split_times[idx], etc.
		if self._idx == self._alloc_len:

			if self.verbose:
				self._print('Appending Timer lists. '
					'This may cause biased time profiling.')

			self._split_starts.append(self._empty_split_val)
			self._split_stops.append(self._empty_split_val)
			self._split_names.append('Task %d' % (self._idx+1))
			self._alloc_len += 1

		return self._idx

	@classmethod
	def _format_time(cls, t_seconds, do_abbreviate=True, n_sig_figs=3):
		''' Builds a string representation of a timing measurement, converted
		to the appropriate units (from nanoseconds to years) and desired number
		of significant figures.

		Currently times < 1 nansecond are printed in scientific notation with
		fixed 3 significant figures. This is typically irrelevant because the
		timing overhead tends to be around 1-10 microseconds.

		This is not the most lightweight printing function. In some use cases,
		a more lightweight variant may be needed if frequent printing is
		required and corresponding time measurements are on the order of
		microseconds.
		'''

		def time2str(t):

			# Reduces to n significant figures for printing
			s = np.format_float_positional(t,
				precision=n_sig_figs,
				fractional=False,
				unique=False,
				trim='k',
				sign=False)

			# Drop trailing decimal point (no 'keep' arg does this without
			# impacting trailing zeros).
			if s[-1] == '.':
				return s[:-1]
			else:
				return s

		def str_sci_notation(t_seconds, do_abbreviate):
			str_units = 's' if do_abbreviate else ' seconds'
			return '%1.2.e%s' % (t_seconds, str_units)

		def str_nanoseconds(t_seconds, do_abbreviate):
			str_units = 'ns' if do_abbreviate else ' nanoseconds'
			return '%s%s' % (time2str(t_seconds*1e9), str_units)

		def str_microseconds(t_seconds, do_abbreviate):
			str_units = 'us' if do_abbreviate else ' microseconds'
			return '%s%s' % (time2str(t_seconds*1e6), str_units)

		def str_milliseconds(t_seconds, do_abbreviate):
			str_units = 'ms' if do_abbreviate else ' milliseconds'
			return '%s%s' % (time2str(t_seconds*1e3), str_units)

		def str_seconds(t_seconds, do_abbreviate):
			str_units = 's' if do_abbreviate else ' seconds'
			return '%s%s' % (time2str(t_seconds), str_units)

		def str_minutes(t_seconds, do_abbreviate):
			str_units = 'mins' if do_abbreviate else 'minutes'
			return '%s %s' % (time2str(t_seconds/cls._s_per_min), str_units)

		def str_hours(t_seconds, do_abbreviate):
			str_units = 'hrs' if do_abbreviate else 'hours'
			return '%s %s' % (time2str(t_seconds/cls._s_per_hr), str_units)

		def str_days(t_seconds, do_abbreviate):
			str_units = 'd' if do_abbreviate else 'days'
			return '%s %s' % (time2str(t_seconds/cls._s_per_day), str_units)

		def str_years(t_seconds, do_abbreviate):
			str_units = 'yrs' if do_abbreviate else 'years'
			return '%s %s' % (time2str(t_seconds/cls._s_per_yr), str_units)

		if t_seconds<0:
			return '-' + cls._format_time(-t_seconds, do_abbreviate)
		elif t_seconds==0:
			return '0 s'
		elif t_seconds<1e-9:
			return str_sci_notation(t_seconds, do_abbreviate)
		elif t_seconds<1e-6:
			return str_nanoseconds(t_seconds, do_abbreviate)
		elif t_seconds<1e-3:
			return str_microseconds(t_seconds, do_abbreviate)
		elif t_seconds<1:
			return str_milliseconds(t_seconds, do_abbreviate)
		elif t_seconds<=cls._s_per_min:
			return str_seconds(t_seconds, do_abbreviate)
		elif t_seconds<=cls._s_per_hr:
			return str_minutes(t_seconds, do_abbreviate)
		elif t_seconds<=cls._s_per_day:
			return str_hours(t_seconds, do_abbreviate)
		else:
			return str_years(t_seconds, do_abbreviate)

	def _print(self, str, n_indent=None):
		'''Prints string after prefixing with the desired number of
		indentations.

		Args:
			str: The string to be printed.

		Returns:
			None.
		'''

		if n_indent is None:
			prefix = self._print_prefix
		else:
			prefix = self._generate_print_prefix(n_indent)

		print('%s%s' % (print_prefix, str))

	def _generate_print_prefix(self, n_indent):

		return '\t' * n_indent

	# ************************************************************************
	# Testing ****************************************************************
	# ************************************************************************

	@classmethod
	def run_tests(cls):
		''' See docstring to test(...).

		Args:
			None.

		Returns:
			None.
		'''

		times = [.123, .234, .345, .456]
		cls.test(times, do_retrospective=True, verbose=True)
		cls.test(times, do_retrospective=False, verbose=True)

		times = np.random.rand(200)/100.;
		cls.test(times, do_retrospective=True, verbose=False)
		cls.test(times, do_retrospective=False, verbose=False)

	@classmethod
	def test(cls, times, 
		do_retrospective=False, 
		verbose=False, 
		tol_seconds=1e-3,
		tol_stddev_seconds=5e-4):
		''' Test the Timer class by timing calls to time.sleep(times[i]).

		Args:
			times: list of (nonnegative) times to measure.

		Returns: 
			the Timer object used in the timing simultion.
		'''
		N = len(times)
		n_tests = 3
		n_passed = 0

		print('\n***********************************************************')
		print('Testing Timer class (do_retrospective=%s)' % do_retrospective)
		print('Simulating %d sequential processes as calls to time.sleep().' 
			% N)
		print('Expected total time for this test: %.3fs.' % np.sum(times))

		if do_retrospective:
			tmr = cls._test_retrospective(times)
		else:
			tmr = cls._test_prospective(times)

		if verbose:
			print('')
			tmr.print()
			print('')
		else:
			print('Done (%.3fs).' % tmr.total_time)

		t_meas = tmr.get_splits()

		# These should be nonnegative (realistically, they should be strictly 
		# positive), indicating consistent and very small amounts of overhead 
		# involved with the timing scaffolding.
		diffs = np.subtract(t_meas, times)

		print('\nAverage error per task (i.e., overhead): %.3es\n' 
			% np.mean(diffs))

		# *********************************************************************
		# 1. Make sure all overhead times are nonnegative.
		# *********************************************************************
		passed = np.all(diffs >= 0)
		pass_or_fail = 'PASSED' if passed else 'FAILED'
		print('%s test 1 of %d:' % (pass_or_fail, n_tests))
		print('\tMin error: %.3es (must be >= 0s)' 
			% np.min(diffs))

		if passed:
			n_passed += 1
		else:
			print('\tDetected measured times that were '
				'shorter than the known time.')

		# *********************************************************************
		# 2. Make sure all overhead times are less than set tolerance.
		# *********************************************************************
		passed = np.all(diffs <= tol_seconds)
		pass_or_fail = 'PASSED' if passed else 'FAILED'
		print('%s test 2 of %d:' % (pass_or_fail, n_tests))
		print('\tMax error: %.3es (must be <= %.3es)' 
			% (np.max(diffs), tol_seconds))

		if passed:
			n_passed += 1
		else:
			print('\tDetected overhead times that exceeded '
				'tolerance of %.3es.' % tol_seconds)

		# *********************************************************************
		# 3. Make sure overhead times have less variability than set tolerance.
		# *********************************************************************
		passed = np.std(diffs, ddof=0) < tol_stddev_seconds
		pass_or_fail = 'PASSED' if passed else 'FAILED'
		print('%s test 3 of %d:' % (pass_or_fail, n_tests))
		print('\tStandard deviation of errors: %.3es (must be <= %.3es)' 
			% (np.std(diffs, ddof=0), tol_stddev_seconds))

		if passed:
			n_passed += 1
		else:
			print('\tDetected overhead standard deviation '
				'in excess of tolerance.')

		# *********************************************************************
		# Print all actual and measured times.
		# *********************************************************************
		if verbose:
			print('\nActual time --> measured time:')
			for n in range(N):
				print('%f-->%f: (error = %fs)' % 
					(times[n], t_meas[n], diffs[n]))
			print ('')

		# *********************************************************************
		# Print testing summary.
		# *********************************************************************
		if n_passed == n_tests:
			print('PASSED ALL TESTS.')
		else:
			raise AssertionError(
				'FAILED %d OF %d TESTS.' % (n_tests-n_passed, n_passed))

		print('***********************************************************\n')

		return tmr

	@classmethod
	def _test_prospective(cls, times):

		N = len(times)
		tmr = cls(N, do_retrospective=False)

		tmr.start() # optional

		for n in range(N):

			# Start split for "task" n.
			tmr.split()

			# Run "task" n
			time.sleep(times[n])

		tmr.stop()

		return tmr

	@classmethod
	def _test_retrospective(cls, times):

		N = len(times)
		tmr = cls(N, do_retrospective=True)
		tmr.start()

		for n in range(N):

			# Run "task" n
			time.sleep(times[n])

			# Retrospective time measurement for "task" n
			t = tmr.split()

		# This shouldn't be needed, not in documentation, doesn't help.
		# tmr.split() 

		tmr.stop() # Optional

		return tmr