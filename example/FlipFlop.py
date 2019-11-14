'''
flipflop.py
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np

if os.environ.get('DISPLAY','') == '':
    # Ensures smooth running across environments, including servers without
    # graphical backends.
    print('No display found. Using non-interactive Agg backend.')
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from RecurrentWhisperer import RecurrentWhisperer

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

            # DO NOT OVERWRITE THESE VALUES.
            # See docstrings in recurrent_whisperer.py for definitions. The
            # values given here reflect that FlipFlop does not use (or require)
            # validation data (because all trials are generated independently).
            'do_generate_lvl_visualizations': False,
            'do_save_lvl_visualizations': False,
            'do_save_lvl_ckpt': False,
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

    def _setup_training(self, train_data, valid_data):
        '''Does nothing. Required by RecurrentWhisperer.'''
        pass

    def _train_batch(self, batch_data):
        '''Performs a training step over a single batch of data.

        Args:
            batch_data: dict containing one training batch of data. Contains
            the following key/value pairs:

                'inputs': [n_batch x n_time x n_bits] numpy array specifying
                the inputs to the RNN.

                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct output responses to the 'inputs.'

        Returns:
            summary: dict containing the following summary key/value pairs
            from the training step:

                'loss': scalar float evalutaion of the loss function over the
                data batch.

                'grad_global_norm': scalar float evaluation of the norm of the gradient of the loss function with respect to all trainable variables, taken over the data batch.
        '''

        ops_to_eval = [self.train_op,
            self.grad_global_norm,
            self.loss,
            self.tensorboard['merged_opt_summary']]

        feed_dict = dict()
        feed_dict[self.inputs_bxtxd] = batch_data['inputs']
        feed_dict[self.output_bxtxd] = batch_data['output']
        feed_dict[self.learning_rate] = self.adaptive_learning_rate()
        feed_dict[self.grad_norm_clip_val] = self.adaptive_grad_norm_clip()

        [ev_train_op,
         ev_grad_global_norm,
         ev_loss,
         ev_merged_opt_summary] = \
                self.session.run(ops_to_eval, feed_dict=feed_dict)

        if self.hps.do_save_tensorboard_summaries:

            if self._epoch()==0:
                '''Hack to prevent throwing the vertical axis on the
                Tensorboard figure for grad_norm_clip_val (grad_norm_clip val
                is initialized to an enormous number to prevent clipping
                before we know the scale of the gradients).'''
                feed_dict[self.grad_norm_clip_val] = np.nan
                ev_merged_opt_summary = \
                    self.session.run(
                        self.tensorboard['merged_opt_summary'],
                        feed_dict)

            self.tensorboard['writer'].add_summary(
                ev_merged_opt_summary, self._step())

        summary = {'loss': ev_loss, 'grad_global_norm': ev_grad_global_norm}

        return summary

    def predict(self, batch_data, do_predict_full_LSTM_state=False):
        '''Runs the RNN given its inputs.

        Args:
            batch_data:
                dict containing the key 'inputs': [n_batch x n_time x n_bits]
                numpy array specifying the inputs to the RNN.

            do_predict_full_LSTM_state (optional): bool indicating, if the RNN
            is an LSTM, whether to return the concatenated hidden and cell
            states (True) or simply the hidden states (False). Default: False.

        Returns:
            predictions: dict containing the following key/value pairs:

                'state': [n_batch x n_time x n_states] numpy array containing
                the activations of the RNN units in response to the inputs.
                Here, n_states is the dimensionality of the hidden state,
                which, depending on the RNN architecture and
                do_predict_full_LSTM_state, may or may not include LSTM cell
                states.

                'output': [n_batch x n_time x n_bits] numpy array containing
                the readouts from the RNN.

        '''

        if do_predict_full_LSTM_state:
            return self._predict_with_LSTM_cell_states(batch_data)
        else:
            ops_to_eval = [self.hidden_bxtxd, self.pred_output_bxtxd]
            feed_dict = {self.inputs_bxtxd: batch_data['inputs']}
            ev_hidden_bxtxd, ev_pred_output_bxtxd = \
                self.session.run(ops_to_eval, feed_dict=feed_dict)

            predictions = {
                'state': ev_hidden_bxtxd,
                'output': ev_pred_output_bxtxd
                }

            return predictions

    def _predict_with_LSTM_cell_states(self, batch_data):
        '''Runs the RNN given its inputs.

        The following is added for execution only when LSTM predictions are
        needed for both the hidden and cell states. Tensorflow does not make
        it easy to access the cell states via dynamic_rnn.

        Args:
            batch_data: as specified by predict.

        Returns:
            predictions: as specified by predict.

        '''

        hps = self.hps
        if hps.rnn_type != 'lstm':
            return self.predict(batch_data)

        n_hidden = hps.n_hidden
        [n_batch, n_time, n_bits] = batch_data['inputs'].shape
        initial_state = self.rnn_cell.zero_state(n_batch, dtype=tf.float32)

        ''' Add ops to the graph for getting the complete LSTM state
        (i.e., hidden and cell) at every timestep.'''
        self.full_state_list = []
        for t in range(n_time):
            input_ = self.inputs_bxtxd[:,t,:]
            if t == 0:
                full_state_t_minus_1 = initial_state
            else:
                full_state_t_minus_1 = self.full_state_list[-1]
            _, full_state_bxd = self.rnn_cell(input_, full_state_t_minus_1)
            self.full_state_list.append(full_state_bxd)

        '''Evaluate those ops'''
        ops_to_eval = [self.full_state_list, self.pred_output_bxtxd]
        feed_dict = {self.inputs_bxtxd: batch_data['inputs']}
        ev_full_state_list, ev_pred_output_bxtxd = \
            self.session.run(ops_to_eval, feed_dict=feed_dict)

        '''Package the results'''
        h = np.zeros([n_batch, n_time, n_hidden]) # hidden states: bxtxd
        c = np.zeros([n_batch, n_time, n_hidden]) # cell states: bxtxd
        for t in range(n_time):
            h[:,t,:] = ev_full_state_list[t].h
            c[:,t,:] = ev_full_state_list[t].c

        ev_LSTMCellState = tf.nn.rnn_cell.LSTMStateTuple(h=h, c=c)

        predictions = {
            'state': ev_LSTMCellState,
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

    def _setup_visualizations(self):
        '''See docstring in RecurrentWhisperer.'''
        self.figs = {}

    def _update_visualizations(self, train_data=None, valid_data=None):
        '''See docstring in RecurrentWhisperer.'''
        data = self.generate_flipflop_trials()
        self.plot_trials(data)

    def plot_trials(self, data, start_time=0, stop_time=None):
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

        hps = self.hps
        n_batch = self.hps.data_hps['n_batch']
        n_time = self.hps.data_hps['n_time']
        n_plot = np.min([hps.n_trials_plot, n_batch])

        if 'example_trials' not in self.figs:
            self.figs['example_trials'] = plt.figure(
                figsize=(FIG_WIDTH, FIG_HEIGHT),
                tight_layout=True)

        fig = self.figs['example_trials']
        plt.clf()

        inputs = data['inputs']
        output = data['output']
        predictions = self.predict(data)
        pred_output = predictions['output']

        if stop_time is None:
            stop_time = n_time

        time_idx = range(start_time, stop_time)

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

        self.refresh_figs()

    @staticmethod
    def _plot_single_trial(input_txd, output_txd, pred_output_txd):

        VERTICAL_SPACING = 2.5
        [n_time, n_bits] = input_txd.shape
        tt = range(n_time)

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

    @staticmethod
    def refresh_figs():
        plt.ion()
        plt.show()
        plt.pause(1e-10)
