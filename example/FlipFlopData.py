'''
flipflop.py
Written for Python 3.6.9
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import os
import sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

class FlipFlopData(object):
	def __init__(self, n_batch=512, n_time=64, n_bits=3, p_flip=0.5, random_seed=0):

		self.rng = npr.RandomState(random_seed)
		self.n_time = n_time
		self.n_bits = n_bits
		self.p_flip = p_flip

	def generate_data(self, n_trials):

		n_time = self.n_time
		n_bits = self.n_bits
		p_flip = self.p_flip

		# Randomly generate unsigned input pulses
		unsigned_inputs = self.rng.binomial(
			1, p_flip, [n_trials, n_time, n_bits])

		# Ensure every trial is initialized with a pulse at time 0
		unsigned_inputs[:, 0, :] = 1

		# Generate random signs {-1, +1}
		random_signs = 2*self.rng.binomial(
			1, 0.5, [n_trials, n_time, n_bits]) - 1

		# Apply random signs to input pulses
		inputs = np.multiply(unsigned_inputs, random_signs)

		# Allocate targets
		targets = np.zeros([n_trials, n_time, n_bits])

		# Update inputs (zero-out random start holds) & compute targets
		for trial_idx in range(n_trials):
			for bit_idx in range(n_bits):
				input_ = np.squeeze(inputs[trial_idx, :, bit_idx])
				t_flip = np.where(input_ != 0)
				for flip_idx in range(np.size(t_flip)):
					# Get the time of the next flip
					t_flip_i = t_flip[0][flip_idx]

					'''Set the targets to the sign of the flip for the
					remainder of the trial. Future flips will overwrite future
					targets'''
					targets[trial_idx, t_flip_i:, bit_idx] = \
						inputs[trial_idx, t_flip_i, bit_idx]

		return {'inputs': inputs, 'targets': targets}

	@classmethod
	def plot_trials(cls, data, pred, start_time=0, stop_time=None, 
		n_trials_plot=1,
		fig=None):
		'''Plots example trials, complete with input pulses, correct outputs,
		and RNN-predicted outputs.

		Args:
			data: dict as returned by generate_data.

			start_time (optional): int specifying the first timestep to plot.
			Default: 0.

			stop_time (optional): int specifying the last timestep to plot.
			Default: n_time.

		Returns:
			None.
		'''

		FIG_WIDTH = 6 # inches
		FIG_HEIGHT = 3 # inches

		if fig is None:
			fig = plt.figure(
				figsize=(FIG_WIDTH, FIG_HEIGHT),
				tight_layout=True)
		else:
			fig.clf()

		inputs = data['inputs']
		targets = data['targets']
		pred_output = pred['output']

		n_batch, n_time, _ = inputs.shape
		n_plot = np.min([n_trials_plot, n_batch])

		if stop_time is None:
			stop_time = n_time

		time_idx = list(range(start_time, stop_time))

		for trial_idx in range(n_plot):
			ax = plt.subplot(n_plot, 1, trial_idx+1)
			if n_plot == 1:
				plt.title('Example trial', fontweight='bold')
			else:
				plt.title('Example trial %d' % (trial_idx + 1),
						  fontweight='bold')

			cls.plot_single_trial(
				inputs[trial_idx, time_idx, :],
				targets[trial_idx, time_idx, :],
				pred_output[trial_idx, time_idx, :])

			# Only plot x-axis ticks and labels on the bottom subplot
			if trial_idx < (n_plot-1):
				plt.xticks([])
			else:
				plt.xlabel('Timestep', fontweight='bold')

		cls._refresh_figs()

		return fig

	@classmethod
	def plot_single_trial(cls, input_txd, output_txd, pred_output_txd):

		VERTICAL_SPACING = 2.5
		[n_time, n_bits] = input_txd.shape
		tt = list(range(n_time))

		y_ticks = [VERTICAL_SPACING*bit_idx for bit_idx in range(n_bits)]
		y_tick_labels = \
			['Bit %d' % (n_bits-bit_idx) for bit_idx in range(n_bits)]

		plt.yticks(y_ticks, y_tick_labels, fontweight='bold')
		for bit_idx in range(n_bits):

			vertical_offset = VERTICAL_SPACING*bit_idx

			# Input pulses
			plt.fill_between(
				tt,
				vertical_offset + input_txd[:, bit_idx],
				vertical_offset,
				step='mid',
				color='gray')

			# Correct outputs
			plt.step(
				tt,
				vertical_offset + output_txd[:, bit_idx],
				where='mid',
				linewidth=2,
				color='cyan')

			# RNN outputs
			plt.step(
				tt,
				vertical_offset + pred_output_txd[:, bit_idx],
				where='mid',
				color='purple',
				linewidth=1.5,
				linestyle='--')

		plt.xlim(-1, n_time)

	@classmethod
	def _refresh_figs(cls):
		''' Refreshes all matplotlib figures.

		Args:
			None.

		Returns:
			None.
		'''
		plt.ion()
		plt.show()
		plt.pause(1e-10)