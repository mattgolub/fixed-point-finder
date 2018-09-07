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
    'rnn_type': 'lstm',
    'n_hidden': 16,
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
from one stable state to another, observed networks states spend little if any
time near the unstable fixed points. In order to identify ALL fixed points,
noise must be added to the initial states before handing them to the fixed
point finder.'''
NOISE_SCALE = 2.5 # Standard deviation of noise added to initial states

N_INITS = 256 # The number of initial states to provide

n_batch = ff.hps.data_hps['n_batch']
n_time = ff.hps.data_hps['n_time']
n_bits = ff.hps.data_hps['n_bits']
n_hidden = ff.hps.n_hidden
is_lstm = ff.hps.rnn_type == 'lstm'

# Study the system in the absence of input pulses (e.g., all inputs are 0)
inputs = np.zeros([1,n_bits])

'''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
descriptions of available hyperparameters.'''
fpf_hps = {}

# Get example state trajectories from the network
example_trials = ff.generate_flipflop_trials()
example_predictions = ff.predict(example_trials,
                                 do_predict_full_LSTM_state=is_lstm)
if is_lstm:
    example_states = FixedPointFinder._convert_from_LSTMStateTuple(
        example_predictions['state'])
else:
    example_states = example_predictions['state']

# if lstm, this reflects the concatenated hidden and cell states
n_states = example_states.shape[2]

'''Draw random samples of those state trajectories to use as initial states
for the fixed point optimizations.'''
initial_states = np.zeros([N_INITS, n_states])
for init_idx in range(N_INITS):
    trial_idx = rng.randint(n_batch)
    time_idx = rng.randint(n_time)
    initial_states[init_idx,:] = example_states[trial_idx,time_idx,:]

# Add noise to the network states
initial_states += NOISE_SCALE * rng.randn(N_INITS, n_states)

if is_lstm:
    initial_states = FixedPointFinder._convert_to_LSTMStateTuple(
        initial_states)

# Setup the fixed point finder
fpf = FixedPointFinder(ff.rnn_cell,
                       ff.session,
                       initial_states,
                       inputs,
                       **fpf_hps)

# Run the fixed point finder
fp_dict = fpf.find_fixed_points()

# Visualize inputs, outputs, and RNN predictions from example trials
ff.plot_trials(example_trials)

# Visualize identified fixed points
# To do: replace with call to RNNTools
example_state_trajectories = example_states[0:5,:,:]
if is_lstm:
    example_state_trajectories = FixedPointFinder._convert_to_LSTMStateTuple(
        example_state_trajectories)

fpf.plot_summary(example_state_trajectories)

print('Entering debug mode to allow interaction with objects and figures.')
pdb.set_trace()