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
alr_hps = {'initial_rate': 0.1}

# Hyperparameters for FlipFlop
# See FlipFlop.py for detailed descriptions.
hps = {
    'rnn_type': 'lstm',
    'n_hidden': 16,
    'min_loss': 5e-4,
    'min_learning_rate': 1e-5,
    'log_dir': './logs/',
    'data_hps': {
        'n_batch': 128,
        'n_time': 256,
        'n_bits': 3,
        'p_flip': 0.5},
    'alr_hps': alr_hps
    }

ff = FlipFlop(**hps)
ff.train()

# Get example state trajectories from the network
# Visualize inputs, outputs, and RNN predictions from example trials
example_trials = ff.generate_flipflop_trials()
ff.plot_trials(example_trials)

# *****************************************************************************
# STEP 2: Find, analyze, and visualize the fixed points of the trained RNN ****
# *****************************************************************************

import sys
sys.path.insert(0, '../FixedPointFinder/')
from FixedPointFinder import FixedPointFinder

'''Initial states are sampled from states observed during realistic behavior
of the network. Because a well-trained network transitions instantaneously
from one stable state to another, observed networks states spend little if any
time near the unstable fixed points. In order to identify ALL fixed points,
noise must be added to the initial states before handing them to the fixed
point finder.'''
NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
N_INITS = 1024 # The number of initial states to provide

n_bits = ff.hps.data_hps['n_bits']
is_lstm = ff.hps.rnn_type == 'lstm'

'''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
descriptions of available hyperparameters.'''
fpf_hps = {}

# Setup the fixed point finder
fpf = FixedPointFinder(ff.rnn_cell,
                       ff.session,
                       **fpf_hps)

# Study the system in the absence of input pulses (e.g., all inputs are 0)
inputs = np.zeros([1,n_bits])

'''Draw random, noise corrupted samples of those state trajectories
to use as initial states for the fixed point optimizations.'''
example_predictions = ff.predict(example_trials,
                                 do_predict_full_LSTM_state=is_lstm)
initial_states = fpf.sample_states(example_predictions['state'],
                                   n_inits=N_INITS,
                                   rng=ff.rng,
                                   noise_scale=NOISE_SCALE)

# Run the fixed point finder
fp_dict = fpf.find_fixed_points(initial_states, inputs)

# Visualize identified fixed points with overlaid RNN state trajectories
# All visualized in the 3D PCA space fit the the example RNN states.
fpf.plot_summary(example_predictions['state'], range(5))

print('Entering debug mode to allow interaction with objects and figures.')
pdb.set_trace()