'''
Tensorflow Utilities
Supports FixedPointFinder
Written for Python 3.6.9 and TensorFlow 2.8.0
@ Matt Golub, October 2018
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb
import numpy as np

import tensorflow as tf
tf1 = tf.compat.v1
tf1.disable_eager_execution()
# tf1.disable_v2_behavior()

'''
These utility functions are primarily for robustly managing TFs different
state representations depending on RNNCell types--specifically the
LSTMStateTuple.
'''

def unroll_LSTM(lstm_cell, inputs, initial_state):
    ''' Unroll an LSTM.

    If only the hidden states (but not the cell states) are needed,
    tf.static_rnn or tf.dynamic_rnn will do the trick. However, those don't
    provide access to the LSTM cell state. This does.

    Args:
        lstm_cell: tf.nn.rnn_cell.LSTMCell object.

        inputs: TF tensor with shape (# trials, # timesteps, # features).
        Contains timestep-by-timestep inputs to the LSTM.

        initial_state: LSTMStateTuple with .h and .c (hidden and cell states,
        respectively) as TF tensors with shape (# trials, # units).

    Returns:
        LSTMStateTuple with .h and .c (hidden and cell states, respectively),
        having shape (# trials, # timesteps, # units).
    '''

    assert (is_lstm(lstm_cell)),('lstm_cell is not an LSTM.')
    assert (is_lstm(initial_state)),('initial_state is not an LSTMStateTuple.')

    ''' Add ops to the graph for getting the complete LSTM state
    (i.e., hidden and cell) at every timestep.'''
    n_time = inputs.shape[1]
    hidden_list = []
    cell_list = []

    prev_state = initial_state

    for t in range(n_time):

        input_ = inputs[:,t,:]

        _, state = lstm_cell(input_, prev_state)

        hidden_list.append(state.h)
        cell_list.append(state.c)
        prev_state = state

    c = tf.stack(cell_list, axis=1)
    h = tf.stack(hidden_list, axis=1)

    return tf1.nn.rnn_cell.LSTMStateTuple(c=c, h=h)

def is_tf_object(x):
    '''Determine whether x is a Tensorflow object.

    Args:
        x: Any object

    Returns:
        A bool indicating whether x is any type of TF object (e.g.,
        tf.Variable, tf.Tensor, tf.placeholder, or any TF op)
    '''
    return tf.debugging.is_numeric_tensor(x) or isinstance(x, tf.Variable)

def is_lstm(x):
    '''Determine whether x is an LSTMCell or any object derived from one.

    Args:
        x: Any object

    Returns:
        A bool indicating whether x is an LSTMCell or any object derived from
        one.
    '''
    if isinstance(x, tf1.nn.rnn_cell.LSTMCell):
        return True

    if isinstance(x, tf1.nn.rnn_cell.LSTMStateTuple):
        return True

    return False

def maybe_convert_from_LSTMStateTuple(x):
    '''Returns a numpy array representation of the RNN states in x.

    Args:
        x: RNN state representation, either as an LSTMStateTuple or a numpy
        array.

    Returns:
        A numpy array representation of x (e.g., concatenated hidden and cell
        states in the case of x as LSTMStateTuple).
    '''
    if is_lstm(x):
        return convert_from_LSTMStateTuple(x)
    else:
        return x

def safe_shape(states):
    '''Returns shape of states robustly regardless of the TF representation.
    If the state TF representation is an LSTMStateTuple, the shape of a
    concatenated (non-tuple) state representation is returned.

    Args:
        states: Either a numpy array, a TF tensor, or an LSTMStateTuple.

    Returns:
        tuple shape of states, directly if states is a numpy array or TF
        tensor, or shape of convert_from_LSTMStateTuple(states) if states is
        an LSTMStateTuple.

    '''
    if is_lstm(states):
        shape_tuple = states.h.shape
        shape_list = list(shape_tuple)
        shape_list[-1] *= 2
        return tuple(shape_list)
    else:
        return states.shape

def safe_index(states, index):
    '''Safely index into RNN states, regardless of the TF representation.

    Args:
        states: Either a numpy array, a TF tensor, or an LSTMStateTuple.

        index: a slice object for indexing into states.

    Returns:
        The data from states indexed by index. Type is preserved from states.
    '''

    if is_lstm(states):
        c = states.c[index]
        h = states.h[index]
        return tf1.nn.rnn_cell.LSTMStateTuple(c=c, h=h)
    else:
        return states[index]

def safe_concat(states):
    '''
    Returns a [b x t x d] Tensor or Numpy array

    states is either:
    1) a [b x t x d] Tensor
    2) an LSTMStateTuple with .c and .h as [b x t x d] Tensors
    3) a tuple or list consisting of two instances of 1) or two
       instances of 2), which correspond to forward and backward
       passes of a bidirectional RNN.
    '''

    if isinstance(states, tf1.nn.rnn_cell.LSTMStateTuple):
        return convert_from_LSTMStateTuple(states)
    elif isinstance(states, tuple) or isinstance(states, list):
        return tf.concat(
            [safe_concat(item) for item in states], axis=2)
    elif isinstance(states, np.ndarray):
        return states
    else:
    	raise ValueError('Unsupported type: %s' % str(type(states)))