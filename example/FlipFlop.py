'''
flipflop.py
Version 1.0
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

PATH_TO_RECURRENT_WHISPERER = '../../recurrent-whisperer/'
sys.path.insert(0, PATH_TO_RECURRENT_WHISPERER)
from RecurrentWhisperer import RecurrentWhisperer

import tensorflow as tf
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

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

        hps = {...} # dictionary of hyperparameter key/value pairs
        ff = FlipFlop(**hps)
        ff.train()
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
            A dict of hyperparameters.
        '''
        return {
            'rnn_type': 'vanilla', # 'vanilla', 'gru' or 'lstm'
            'n_hidden': 24,
            'data_hps': {
                'n_batch':  128,
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
            A dict of hyperparameters.
        '''
        return {
            'log_dir': '/tmp/flipflop_logs/',
            'n_epochs_per_validation_update': -1, #overrides RecurrentWhisperer
            'n_trials_plot': 4,
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
        self.inputs_bxtxd = tf.placeholder(tf.float32,
            [n_batch, n_time, n_inputs])
        self.output_bxtxd = tf.placeholder(tf.float32,
            [n_batch, n_time, n_output])

        # RNN
        if hps.rnn_type == 'vanilla':
            self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
        elif hps.rnn_type == 'gru':
            self.rnn_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
        elif hps.rnn_type == 'lstm':
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of '
                '[vanilla, gru, lstm] but was %s' % hps.rnn_type)

        initial_state = self.rnn_cell.zero_state(n_batch, dtype=tf.float32)
        self.hidden_bxtxd, _ = tf.nn.dynamic_rnn(self.rnn_cell,
            self.inputs_bxtxd, initial_state=initial_state)

        # Readout from RNN
        np_W_out, np_b_out = self._np_init_weight_matrix(n_hidden, n_output)
        self.W_out = tf.Variable(np_W_out, dtype=tf.float32)
        self.b_out = tf.Variable(np_b_out, dtype=tf.float32)
        self.pred_output_bxtxd = tf.tensordot(self.hidden_bxtxd,
            self.W_out, axes=1) + self.b_out

        # Loss
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.output_bxtxd, self.pred_output_bxtxd))

    def _setup_saver(self):
        '''See docstring in RecurrentWhisperer.'''

        self.saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=self.hps.max_ckpt_to_keep)

    def _setup_training(self, train_data, valid_data):
        '''Does nothing. Required by RecurrentWhisperer.'''
        pass

    def _train_batch(self, batch_data):
        '''See docstring in RecurrentWhisperer.'''


        ops_to_eval = [self.train_op,
            self.grad_global_norm,
            self.loss,
            self.merged_opt_summaries]

        feed_dict = dict()
        feed_dict[self.inputs_bxtxd] = batch_data['inputs']
        feed_dict[self.output_bxtxd] = batch_data['output']
        feed_dict[self.learning_rate] = self.adaptive_learning_rate()
        feed_dict[self.grad_norm_clip_val] = self.adaptive_grad_norm_clip()

        [ev_train_op,
            ev_grad_global_norm,
            ev_loss,
            ev_merged_opt_summaries] = \
                self.session.run(ops_to_eval, feed_dict=feed_dict)

        self.writer.add_summary(ev_merged_opt_summaries, self._step())

        summary = {'loss': ev_loss, 'grad_global_norm': ev_grad_global_norm}

        return summary

    def predict(self, batch_data):
        '''See docstring in RecurrentWhisperer.'''

        ops_to_eval = [self.hidden_bxtxd, self.pred_output_bxtxd]
        feed_dict = {self.inputs_bxtxd: batch_data['inputs']}
        ev_hidden_bxtxd, ev_pred_output_bxtxd = \
            self.session.run(ops_to_eval, feed_dict=feed_dict)

        predictions = {
            'hidden': ev_hidden_bxtxd,
            'output': ev_pred_output_bxtxd
            }

        return predictions

    def _get_data_batches(self, train_data):
        '''See docstring in RecurrentWhisperer.'''
        return [self.generate_flipflop_trials()]

    def _get_batch_size(self, batch_data):
        '''See docstring in RecurrentWhisperer.'''
        return batch_data['inputs'].shape[0]

    def generate_flipflop_trials(self):
        '''Generates synthetic data (i.e., ground truth trials) for the
        FlipFlop task. See comments following FlipFlop class definition for a
        description of the input-output relationship in the task.

        Args:
            None.

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
        unsigned_inputs = npr.binomial(1, p_flip, [n_batch, n_time, n_bits])

        # Ensure every trial is initialized with a pulse at time 0
        unsigned_inputs[:, 0, :] = 1

        # Generate random signs {-1, +1}
        random_signs = 2*npr.binomial(1, 0.5, [n_batch, n_time, n_bits]) - 1

        # Apply random signs to input pulses
        inputs = np.multiply(unsigned_inputs, random_signs)

        # Allocate output
        output = np.zeros([n_batch, n_time, n_bits])

        # Update inputs (zero-out random start holds) & compute output
        for trial_idx in range(n_batch):
            for bit_idx in range(n_bits):
                _input = np.squeeze(inputs[trial_idx, :, bit_idx])
                t_flip = np.where(_input != 0)
                for flip_idx in range(np.size(t_flip)):
                    # Get the time of the next flip
                    t_flip_i = t_flip[0][flip_idx]

                    '''Set the output to the sign of the flip for the
                    remainder of the trial. Future flips will overwrite future
                    output'''
                    output[trial_idx, t_flip_i:, bit_idx] = \
                        inputs[trial_idx, t_flip_i, bit_idx]

        return {'inputs': inputs, 'output': output}

    def _setup_visualization(self):
        '''See docstring in RecurrentWhisperer.'''
        self.fig = plt.figure()

    def _update_visualization(self, train_data=None, valid_data=None):
        '''See docstring in RecurrentWhisperer.'''
        data = self.generate_flipflop_trials()
        self.plot_trials(data)

    def plot_trials(self, data):
        '''Plots example trials, complete with input pulses, correct outputs,
        and RNN-predicted outputs.

        Args:
            data: dict as returned by generate_flipflop_trials.

        Returns:
            None.
        '''
        fig = plt.figure(self.fig.number)
        plt.clf()

        predictions = self.predict(data)

        inputs = data['inputs']
        output = data['output']
        pred_output = predictions['output']

        [n_batch, n_time, n_bits] = np.shape(inputs)
        n_plot = np.min([self.hps.n_trials_plot, n_batch])

        for trial_idx in range(n_plot):
            ax = plt.subplot(n_plot, 1, trial_idx+1)
            for bit_idx in range(n_bits):
                vertical_offset = 2.5*bit_idx
                ax.plot(vertical_offset + inputs[trial_idx, :, bit_idx],
                    color='purple')
                ax.plot(vertical_offset + output[trial_idx, :, bit_idx],
                    color='green')
                ax.plot(vertical_offset + pred_output[trial_idx, :, bit_idx],
                    color='cyan', linestyle='--')
            plt.yticks([])
            if trial_idx < (n_plot-1):
                plt.xticks([])

        plt.ion()
        plt.show()
        plt.pause(1e-10)
