'''
flipflop.py
Written for Python 3.6.9 and TensorFlow 2.8.0
@ Matt Golub, August 2018
Please direct correspondence to mgolub@cs.washington.edu
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf1 = tf.compat.v1
tf1.disable_eager_execution()
# tf1.disable_v2_behavior()

from RecurrentWhisperer import RecurrentWhisperer
import tf_utils

class FlipFlop(RecurrentWhisperer):
    ''' Class for training an RNN to implement an N-bit memory, a.k.a. "the
    flip-flop  task" as described in Sussillo & Barak, Neural Computation,
    2013.

    Task:
        Briefly, a set of inputs carry transient pulses (-1 or +1) to set the
        state of a set of binary outputs (also -1 or +1). Each input drives
        exactly one output. If the sign of an input pulse opposes the sign
        currently held at the corresponding output, the sign of the output
        flips. If an input pulse's sign matches that currently held at the
        corresponding output, the output does not change.

        This class generates synthetic data for the flip-flop task via
        generate_flipflop_trials(...).

    Usage:
        This class trains an RNN to generate the correct outputs given the
        inputs of the flip-flop task. All that is needed to get started is to
        construct a flipflop object and to call .train on that object:

        # dict of hyperparameter key/value pairs
        # (see 'Hyperparameters' section below)
        hps = {...}

        ff = FlipFlop(**hps)
        ff.train()

    Hyperparameters:
        rnn_type: string specifying the architecture of the RNN. Currently
        must be one of {'vanilla', 'gru', 'lstm'}. Default: 'vanilla'.

        n_hidden: int specifying the number of hidden units in the RNN.
        Default: 24.

        data_hps: dict containing hyperparameters for generating synthetic
        data. Contains the following keys:

            'n_batch': int specifying the number of synthetic trials to use
            per training batch (i.e., for one gradient step). Default: 128.

            'n_time': int specifying the duration of each synthetic trial
            (measured in timesteps). Default: 256.

            'n_bits': int specifying the number of input channels into the
            FlipFlop device (which will also be the number of output channels).
            Default: 3.

            'p_flip': float between 0.0 and 1.0 specifying the probability
            that a particular input channel at a particular timestep will
            contain a pulse (-1 or +1) on top of its steady-state value (0).
            Pulse signs are chosen by fair coin flips, and pulses are produced
            with the same statistics across all input channels and across all
            timesteps (i.e., there are no history effects, there are no
            interactions across input channels). Default: 0.2.

        log_dir: string specifying the top-level directory for saving various
        training runs (where each training run is specified by a different set
        of hyperparameter settings). When tuning hyperparameters, log_dir is
        meant to be constant across models. Default: '/tmp/flipflop_logs/'.

        n_trials_plot: int specifying the number of synthetic trials to plot
        per visualization update. Default: 4.
    '''

    @staticmethod
    def _default_hash_hyperparameters():
        '''Defines default hyperparameters, specific to FlipFlop, for the set
        of hyperparameters that are hashed to define a directory structure for
        easily managing multiple runs of the RNN training (i.e., using
        different hyperparameter settings). Additional default hyperparameters
        are defined in RecurrentWhisperer (from which FlipFlop inherits).

        Args:
            None.

        Returns:
            dict of hyperparameters.
        '''
        return {
            'rnn_type': 'vanilla',
            'n_hidden': 24,
            'data_hps': {
                'n_batch': 128,
                'n_time': 256,
                'n_bits': 3,
                'p_flip': 0.2}
            }

    @staticmethod
    def _default_non_hash_hyperparameters():
        '''Defines default hyperparameters, specific to FlipFlop, for the set
        of hyperparameters that are NOT hashed. Additional default
        hyperparameters are defined in RecurrentWhisperer (from which FlipFlop
        inherits).

        Args:
            None.

        Returns:
            dict of hyperparameters.
        '''
        return {
            'log_dir': '/tmp/flipflop_logs/',
            'n_trials_plot': 1,
            }

    def _setup_model(self):
        '''Defines an RNN in Tensorflow.

        See docstring in RecurrentWhisperer.
        '''
        hps = self.hps
        n_hidden = hps.n_hidden

        data_hps = hps.data_hps
        n_batch = data_hps['n_batch']
        n_time = data_hps['n_time']
        n_inputs = data_hps['n_bits']
        n_output = n_inputs

        # Data handling
        self.inputs_bxtxd = tf1.placeholder(tf.float32,
            [n_batch, n_time, n_inputs])
        self.output_bxtxd = tf1.placeholder(tf.float32,
            [n_batch, n_time, n_output])

        # RNN
        if hps.rnn_type == 'vanilla':
            self.rnn_cell = tf1.nn.rnn_cell.BasicRNNCell(n_hidden)
        elif hps.rnn_type == 'gru':
            self.rnn_cell = tf1.nn.rnn_cell.GRUCell(n_hidden)
        elif hps.rnn_type == 'lstm':
            self.rnn_cell = tf1.nn.rnn_cell.LSTMCell(n_hidden)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of '
                '[vanilla, gru, lstm] but was %s' % hps.rnn_type)

        initial_state = self.rnn_cell.zero_state(n_batch, dtype=tf.float32)

        if hps.rnn_type == 'lstm':
            self.state_bxtxd = tf_utils.unroll_LSTM(
                self.rnn_cell,
                inputs=self.inputs_bxtxd,
                initial_state=initial_state)

            self.hidden_bxtxd = self.state_bxtxd.h

        else:
            self.state_bxtxd, _ = tf1.nn.dynamic_rnn(
                self.rnn_cell,
                inputs=self.inputs_bxtxd,
                initial_state=initial_state)

            self.hidden_bxtxd = self.state_bxtxd

        # Readout from RNN
        np_W_out, np_b_out = self._np_init_weight_matrix(n_hidden, n_output)
        self.W_out = tf.Variable(np_W_out, dtype=tf.float32, name='W_out')
        self.b_out = tf.Variable(np_b_out, dtype=tf.float32, name='b_out')
        self.pred_output_bxtxd = tf.tensordot(self.hidden_bxtxd,
            self.W_out, axes=1) + self.b_out

        # Loss
        self.loss = tf.reduce_mean(
            input_tensor=tf.math.squared_difference(self.output_bxtxd, self.pred_output_bxtxd))

    def _np_init_weight_matrix(self, input_size, output_size):
        '''Randomly initializes a weight matrix W and bias vector b.

        For use with input data matrix X [n x input_size] and output data
        matrix Y [n x output_size], such that Y = X*W + b (with broadcast
        addition). This is the typical required usage for TF dynamic_rnn.

        Weights drawn from a standard normal distribution and are then
        rescaled to preserve input-output variance.

        Args:
            input_size: non-negative int specifying the number of input
            dimensions of the linear mapping.

            output_size: non-negative int specifying the number of output
            dimensions of the linear mapping.

        Returns:
            W: numpy array of shape [input_size x output_size] containing
            randomly initialized weights.

            b: numpy array of shape [output_size,] containing all zeros.
        '''
        if input_size == 0:
            scale = 1.0 # This avoids divide by zero error
        else:
            scale = 1.0 / np.sqrt(input_size)
        W = np.multiply(scale,self.rng.randn(input_size, output_size))
        b = np.zeros(output_size)
        return W, b

    def _build_data_feed_dict(self, data, **kwargs):
        '''Performs a training step over a single batch of data.

        Args:
            data: dict containing one epoch of data. Contains the
            following key/value pairs:

                'inputs': [n_batch x n_time x n_bits] numpy array specifying
                the inputs to the RNN.

                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct output responses to the 'inputs.'

        Returns:
            dict with (TF placeholder, feed value) as (key, value) pairs.
        '''
        feed_dict = dict()
        feed_dict[self.inputs_bxtxd] = data['inputs']
        feed_dict[self.output_bxtxd] = data['output']
        return feed_dict

    def _get_pred_ops(self):
        ''' See docstring in RecurrentWhisperer._get_pred_ops()
        '''

        return {
            'state': self.state_bxtxd,
            'output': self.pred_output_bxtxd
            }

    def _get_batch_size(self, batch_data):
        '''See docstring in RecurrentWhisperer.'''
        return batch_data['inputs'].shape[0]

    def generate_data(self, train_or_valid_str=None):
        '''Generates synthetic data (i.e., ground truth trials) for the
        FlipFlop task. See comments following FlipFlop class definition for a
        description of the input-output relationship in the task.

        Args:
            None (RecurrentWhisperer option train_or_valid_str is ignored).

        Returns:
            dict containing 'inputs' and 'outputs'.

                'inputs': [n_batch x n_time x n_bits] numpy array containing
                input pulses.

                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct behavior of the FlipFlop memory device.
        '''

        data_hps = self.hps.data_hps
        n_batch = data_hps['n_batch']
        n_time = data_hps['n_time']
        n_bits = data_hps['n_bits']
        p_flip = data_hps['p_flip']

        # Randomly generate unsigned input pulses
        unsigned_inputs = self.rng.binomial(
            1, p_flip, [n_batch, n_time, n_bits])

        # Ensure every trial is initialized with a pulse at time 0
        unsigned_inputs[:, 0, :] = 1

        # Generate random signs {-1, +1}
        random_signs = 2*self.rng.binomial(
            1, 0.5, [n_batch, n_time, n_bits]) - 1

        # Apply random signs to input pulses
        inputs = np.multiply(unsigned_inputs, random_signs)

        # Allocate output
        output = np.zeros([n_batch, n_time, n_bits])

        # Update inputs (zero-out random start holds) & compute output
        for trial_idx in range(n_batch):
            for bit_idx in range(n_bits):
                input_ = np.squeeze(inputs[trial_idx, :, bit_idx])
                t_flip = np.where(input_ != 0)
                for flip_idx in range(np.size(t_flip)):
                    # Get the time of the next flip
                    t_flip_i = t_flip[0][flip_idx]

                    '''Set the output to the sign of the flip for the
                    remainder of the trial. Future flips will overwrite future
                    output'''
                    output[trial_idx, t_flip_i:, bit_idx] = \
                        inputs[trial_idx, t_flip_i, bit_idx]

        return {'inputs': inputs, 'output': output}

    def _split_data_into_batches(self, data):
        '''See docstring in RecurrentWhisperer.'''

        # Just use a single batch in this simple example.
        return [data], None

    def _combine_prediction_batches(self, pred_list, summary_list, idx_list):
        '''See docstring in RecurrentWhisperer.'''

        # Just use a single batch in this simple example.

        assert (len(pred_list)==1),\
            ('FlipFlop only supports single batches, but found %d batches.'
             % len(pred_list))

        assert (len(summary_list)==1),\
            ('FlipFlop only supports single batches, but found %d batches.'
             % len(summary_list))

        return pred_list[0], summary_list[0]

    def _update_visualizations(self, data, pred,
        train_or_valid_str=None,
        version=None):
        '''See docstring in RecurrentWhisperer.'''

        self.plot_trials(data, pred)
        self.refresh_figs()

    def plot_trials(self, data, pred, start_time=0, stop_time=None):
        '''Plots example trials, complete with input pulses, correct outputs,
        and RNN-predicted outputs.

        Args:
            data: dict as returned by generate_flipflop_trials.

            start_time (optional): int specifying the first timestep to plot.
            Default: 0.

            stop_time (optional): int specifying the last timestep to plot.
            Default: n_time.

        Returns:
            None.
        '''

        FIG_WIDTH = 6 # inches
        FIG_HEIGHT = 3 # inches

        fig = self._get_fig('example_trials',
            width=FIG_WIDTH,
            height=FIG_HEIGHT)

        hps = self.hps
        n_batch = self.hps.data_hps['n_batch']
        n_time = self.hps.data_hps['n_time']
        n_plot = np.min([hps.n_trials_plot, n_batch])

        inputs = data['inputs']
        output = data['output']
        pred_output = pred['output']

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

            self._plot_single_trial(
                inputs[trial_idx, time_idx, :],
                output[trial_idx, time_idx, :],
                pred_output[trial_idx, time_idx, :])

            # Only plot x-axis ticks and labels on the bottom subplot
            if trial_idx < (n_plot-1):
                plt.xticks([])
            else:
                plt.xlabel('Timestep', fontweight='bold')

    @staticmethod
    def _plot_single_trial(input_txd, output_txd, pred_output_txd):

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