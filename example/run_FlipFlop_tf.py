'''
run_flipflop.py
Written for Python 3.6.9 and TensorFlow 2.8.0
@ Matt Golub, October 2018
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb
import sys
import argparse
import numpy as np

PATH_TO_FIXED_POINT_FINDER = '../'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FlipFlop import FlipFlop
from FixedPointFinder import FixedPointFinder
from FixedPoints import FixedPoints
from plot_utils import plot_fps

def train_FlipFlop(train_mode):
    ''' Train an RNN to solve the N-bit memory task.

        Args:
            train_mode: 1, 2, or 3.

                1.  Generate on-the-fly training data (new data for each
                    gradient step)
                2.  Provide a single, fixed set of training data.
                3.  Provide, single, fixed set of training data (as in 2) and
                    a single, fixed set of validation data.

                (see docstring to RecurrentWhisperer.train() for more detail)

        Returns:
            model: FlipFlop object.

                The trained RNN model.

            valid_predictions: dict.

                The model's predictions on a set of held-out validation trials.
    '''

    assert train_mode in [1, 2, 3], \
        ('train_mode must be 1, 2, or 3, but was %s' % str(train_mode))

    # Hyperparameters for FlipFlop
    # See FlipFlop.py for detailed descriptions.
    hps = {
            'rnn_type': 'lstm',
            'n_hidden': 16,
            'min_loss': 1e-4,
            'log_dir': './logs/',
            'do_generate_pretraining_visualizations': True,

            'data_hps': {
                'n_batch': 512,
                'n_time': 64,
                'n_bits': 3,
                'p_flip': 0.5
                },

            # Hyperparameters for AdaptiveLearningRate
            'alr_hps': {
                'initial_rate': 1.0,
                'min_rate': 1e-5
                }
            }

    model = FlipFlop(**hps)

    train_data = model.generate_data(n_trials=hps['data_hps']['n_batch'])
    valid_data = model.generate_data(n_trials=hps['data_hps']['n_batch'])

    if train_mode == 1:
        model.train()
    elif train_mode == 2:
        # This runs much faster at the expense of overfitting potential
        model.train(train_data)
    elif train_mode == 3:
        # This requires some changes to hps to fully leverage validation
        model.train(train_data, valid_data)

    # Get example state trajectories from the network
    # Visualize inputs, outputs, and RNN predictions from example trials
    valid_predictions, valid_summary = model.predict(valid_data)
    model.plot_trials(valid_data, valid_predictions)

    return model, valid_predictions

def find_fixed_points(model, valid_predictions):
    ''' Find, analyze, and visualize the fixed points of the trained RNN.

    Args:
        model: FlipFlop object.

            Trained RNN model, as returned by train_FlipFlop().

        valid_predictions: dict.

            Model predictions on validation trials, as returned by
            train_FlipFlop().

    Returns:
        None.
    '''

    '''Initial states are sampled from states observed during realistic
    behavior of the network. Because a well-trained network transitions
    instantaneously from one stable state to another, observed networks states
    spend little if any time near the unstable fixed points. In order to
    identify ALL fixed points, noise must be added to the initial states
    before handing them to the fixed point finder. In this example, the noise
    needed is rather large, which can lead to identifying fixed points well
    outside of the domain of states observed in realistic behavior of the
    network--such fixed points can be safely ignored when interpreting the
    dynamical landscape (but can throw visualizations).'''

    NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
    N_INITS = 1024 # The number of initial states to provide

    n_bits = model.hps.data_hps['n_bits']

    '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
    descriptions of available hyperparameters.'''
    fpf_hps = {}

    # Setup the fixed point finder
    fpf = FixedPointFinder(model.rnn_cell, model.session, **fpf_hps)

    # Study the system in the absence of input pulses (e.g., all inputs are 0)
    inputs = np.zeros([1,n_bits])

    '''Draw random, noise corrupted samples of those state trajectories
    to use as initial states for the fixed point optimizations.'''
    initial_states = fpf.sample_states(valid_predictions['state'],
        n_inits=N_INITS,
        noise_scale=NOISE_SCALE)

    # Run the fixed point finder
    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

    # Visualize identified fixed points with overlaid RNN state trajectories
    # All visualized in the 3D PCA space fit the the example RNN states.
    fig = plot_fps(unique_fps, valid_predictions['state'],
        plot_batch_idx=list(range(30)),
        plot_start_time=10)

    model.save_visualizations(figs={'fixed_points': fig})

    print('Entering debug mode to allow interaction with objects and figures.')
    print('You should see a figure with:')
    print('\tMany blue lines approximately outlining a cube')
    print('\tStable fixed points (black dots) at corners of the cube')
    print('\tUnstable fixed points (red lines or crosses) '
        'on edges, surfaces and center of the cube')
    print('Enter q to quit.\n')
    pdb.set_trace()

def main():

    parser = argparse.ArgumentParser(
        description='FixedPointFinder: Flip Flop example')
    parser.add_argument('--train_mode', default=1, type=int)
    args = vars(parser.parse_args())
    train_mode = args['train_mode']

    # Step 1: Train an RNN to solve the N-bit memory task
    model, valid_predictions = train_FlipFlop(train_mode)

    # STEP 2: Find, analyze, and visualize the fixed points of the trained RNN
    find_fixed_points(model, valid_predictions)

if __name__ == '__main__':
    main()
