'''
run_flipflop.py
Version 1.0
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
import tensorflow as tf
import sys

PATH_TO_FIXED_POINT_FINDER = '../'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FlipFlop import FlipFlop

# *****************************************************************************
# STEP 1: Train an RNN to solve the N-bit memory task *************************
# *****************************************************************************

# Hyperparameters for AdaptiveLearningRate
alr_hps = {'initial_rate': 0.05}

# Hyperparameters for FlipFlop
# See FlipFlop.py for detailed descriptions.
hps = {
    'rnn_type': 'gru',
    'n_hidden': 24,
    'min_loss': 1e-3,
    'min_learning_rate': 1e-5,
    'log_dir': './logs/',
    'alr_hps': alr_hps
    }

ff = FlipFlop(**hps)
ff.train()

# *****************************************************************************
# STEP 2: Find, analyze, and visualize the fixed points of the trained RNN ****
# *****************************************************************************

import sys
sys.path.insert(0, '../FixedPointFinder/')
from FixedPointFinder import FixedPointFinder

'''For reproducibility, use the seeded random number generator from the
FlipFlop object.'''
rng = ff.rng

'''Initial states are sampled from states observed during realistic behavior
of the network. Because a well-trained network transitions instantaneously
from one stable state to another, observed networks states are never near the
unstable fixed points. In order to identify ALL fixed points, noise must be
added to the initial states before handing them to the fixed point finder.'''
NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states

N_INITS = 256 # The number of initial states to provide

# Study the system in the absence of input pulses (e.g., all inputs are 0)
n_bits = ff.hps.data_hps['n_bits']
inputs = np.zeros([1,n_bits])

'''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
descriptions of available hyperparameters.'''
fpf_hps = {}

# Get initial states from realistic runs of the network
example_trials = ff.generate_flipflop_trials()
example_predictions = ff.predict(example_trials)

initial_states = np.zeros([N_INITS, hps['n_hidden']])
for init_idx in range(N_INITS):
    trial_idx = rng.randint(ff.hps.data_hps['n_batch'])
    time_idx = rng.randint(ff.hps.data_hps['n_time'])
    initial_states[init_idx,:] = \
        example_predictions['hidden'][trial_idx,time_idx,:]

# Add noise to the network states
initial_states += NOISE_SCALE * rng.randn(N_INITS, hps['n_hidden'])

'''Tensorflow doesn't make it easy to access LSTM cell states. If RNN is an
LSTM, initialize cell states to be zeros for the purposes of fixed point
finding, recognizing that this might have negative effects on the fixed point
finding (since the initial states might not be perfectly representative of
realistic network operation).'''
is_lstm = isinstance(ff.rnn_cell.state_size, tf.nn.rnn_cell.LSTMStateTuple)
if is_lstm:
    initial_states = tf.nn.rnn_cell.LSTMStateTuple(
        h=initial_states, c=np.zeros([N_INITS, hps['n_hidden']]))

# Setup the fixed point finder and run it.
fpf = FixedPointFinder(ff.rnn_cell, ff.session,
                       initial_states, inputs, **fpf_hps)
fp_dict = fpf.find_fixed_points()

# Visualize example trials
ff.plot_trials(example_trials)

# Visualize identified fixed points
fpf.plot_summary()

print('Entering debug mode to allow interaction with objects and figures.')
pdb.set_trace()