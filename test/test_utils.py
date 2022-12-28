'''
test_utils.py
Written for Python 3.6.9 and TensorFlow 2.8.0
@ Matt Golub, October 2018
Please direct correspondence to mgolub@cs.washington.edu
'''

import numpy as np
import tensorflow as tf
tf1 = tf.compat.v1
import matplotlib.pyplot as plt

def build_test_rnn(n_hidden, n_inputs, session):
    '''Build a "vanilla" RNNCell and deterministically set its weights. The
    RNN's fixed points are thus also deterministic, allowing ground truth
    values to be compared against those found by FixedPointFinder.

    Args:
        n_hidden:
            Non-negative int specifying the number of hidden units in the RNN.

        n_inputs:
            Non-negative int specifying the number of inputs to the RNN.
            Inputs are ignored by these test RNN's, so this value can be set
            arbitrarily without affecting results. This is included only to
            ensure input dimensionality is consistent across the various objects used in a test.

        session:
            A Tensorflow session within which to initialize the RNNCell.
    '''
    rnn_cell = tf1.nn.rnn_cell.BasicRNNCell(n_hidden)
    # rnn_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    # rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

    # These will stay as 0's
    W_np_input = np.zeros([n_inputs, n_hidden], dtype=np.float32)
    b_np = np.zeros([n_hidden], dtype=np.float32)

    # This will be set deterministically
    W_np_state = np.zeros([n_hidden, n_hidden], dtype=np.float32)

    '''The following was chosen to such that resulting networks have some non-
    origin fixed points.'''
    row_vals = np.linspace(-1, 1, n_hidden)
    for in_idx in range(n_hidden):
        W_np_state[in_idx, :] = -np.roll(row_vals, -in_idx)
    # print('\nW=' + str(W_np_state))

    W_np = np.concatenate((W_np_input, W_np_state), axis=0)

    input_data = tf.Variable(np.zeros([1, n_inputs]), dtype=tf.float32)
    state = tf.Variable(np.zeros([1, n_hidden]), dtype=tf.float32)

    output, final_state = rnn_cell(input_data, state)
    W_tf, b_tf = rnn_cell.variables

    assign_W = W_tf.assign(W_np)
    assign_b = b_tf.assign(b_np)

    session.run(tf1.global_variables_initializer())
    session.run([assign_W, assign_b])

    return rnn_cell

def generate_initial_states_and_inputs(n_hidden, n_inputs,
                                       n_inits_per_state_dim=3,
                                       min_val_per_state_dim=-1.0,
                                       max_val_per_state_dim=1.0,
                                       debug=False):
    '''Generates a grid of initial states and zeroed-out inputs in a format
    compatible with RNNCell.

    Args:
        n_hidden:
            Non-negative int specifying the number of hidden units in the RNN.

        n_inputs:
            Non-negative int specifying the number of inputs to the RNN.
            Inputs are ignored by these test RNN's, so this value can be set
            arbitrarily without affecting results. This is included only to
            ensure input dimensionality is consistent across the various
            objects used in a test.

        n_inits_per_state_dim (optional):
            Non-negative int specifying the grid resolution for each
            dimension. The total number of grid states will be
            pow(n_inits_per_state_dim, n_hidden). Default: 3.

        min_val_per_state_dim (optional):
            Float specifying the minimum value for each state dimension.
            Default: -1.0.

        max_val_per_state_dim (optional):
            Float specifying the maximum value for each state dimension.
            Default: 1.0.

        debug (optional):
            Bool indicating whether to visualize the grid of states for the
            first two state dimensions (first two hidden units). Default:
            False.

    Returns:
        grid_states:
            [n, n_hidden] numpy array with grid_states[i, :] representing one
            initial state.

        inputs:
            [1, n_inputs] numpy array of zeros.
    '''

    # All inputs are set to zero.
    inputs = np.zeros([1,n_inputs])

    grid_coords_1D = (np.linspace(min_val_per_state_dim, max_val_per_state_dim, n_inits_per_state_dim) for hidden_idx in range(n_hidden))
    grid_states_list = np.meshgrid(*grid_coords_1D)

    n_inits = pow(n_inits_per_state_dim, n_hidden)
    grid_states = np.transpose(
        np.reshape(np.array(grid_states_list), [n_hidden, n_inits]))

    # Visualize the grid of initial states (first 2 dims only)
    if debug and n_hidden > 1:
        import matplotlib.pyplot as plt
        plt.figure()
        for state_idx in range(n_inits):
            plt.plot(grid_states[state_idx, 0],
                     grid_states[state_idx, 1],
                     'rx')
        plt.show()

    return grid_states, inputs

def get_ground_truth_path(test_path, n_hidden):
    '''Returns the path of the file containing the ground truth FixedPoints.

    Args:
        test_path:
            String indicating the path of the directory containing all ground
            truth files.

        n_hidden:
            Number of hidden units in the test RNN for which ground truth
            FixedPoints are sought.

    Returns:
        String containing the path to the file containing the ground truth
        FixedPoints.
    '''
    return test_path + str('ground-truth/n_hidden=%02d.fps' % n_hidden)