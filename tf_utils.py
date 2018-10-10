'''
Tensorflow Utilities
Supports FixedPointFinder
Written using Python 2.7.12 and TensorFlow 1.10.
@ Matt Golub, October 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def convert_from_LSTMStateTuple(lstm_state):
    '''Concatenates the representations of LSTM hidden and cell states.

    Args:
        lstm_state: an LSTMStateTuple, with .c and .h as
        [n_batch, n_dims/2] or [n_batch, n_time, n_dims/2] numpy or tf
        objects.

    Returns:
        A numpy or tf object with shape [n_batch, n_dims] or
        [n_batch, n_time, n_dims] containing the concatenated hidden and
        cell states (type is preserved from lstm_state).
    '''
    c = lstm_state.c
    h = lstm_state.h

    rank = len(lstm_state.c.shape) # works for tf and np objects
    axis = rank - 1

    if is_tf_object(c):
        return tf.concat((c, h), axis=axis)
    else:
        return np.concatenate((c, h), axis=axis)

def convert_to_LSTMStateTuple(x):
    '''Converts a concatenated representation of LSTMT hidden and cell
    states to tf's LSTMStateTuple representation.

    Args:
        x: [n_batch, n_dims] or [n_batch, n_time, n_dims] numpy or tf
        object containing concatenated hidden and cell states.

    Returns:
        An LSTMStateTuple containing the de-concatenated hidden and cell
        states from x. Resultant .c and .h are either [n_batch , n_dims/2]
        or [n_batch, n_time, n_dims/2] numpy or tf objects (type and rank
        preserved from x).
    '''
    rank = len(x.shape) # works for tf and np objects
    n_concat_dims = x.shape[rank - 1]
    if np.mod(n_concat_dims, 2) != 0:
        raise ValueError('x must have an even length'
                         ' along its last dimension,'
                         ' but was length %d' % n_concat_dims)

    n_dims = n_concat_dims//2 # floor division returns an int
    if rank == 2:
        c = x[0:, :n_dims] # [n_batch x n_dims]
        h = x[0:, n_dims:] # [n_batch x n_dims]
    elif rank == 3:
        c = x[0:, 0:, :n_dims] # [n_batch x n_time x n_dims]
        h = x[0:, 0:, n_dims:] # [n_batch x n_time x n_dims]
    else:
        raise ValueError('x must be rank 2 or 3, but was rank %d' % rank)

    return tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)

def is_tf_object(x):
    '''Determine whether x is a Tensorflow object.

    Args:
        x: Any object

    Returns:
        A bool indicating whether x is any type of TF object (e.g.,
        tf.Variable, tf.Tensor, tf.placeholder, or any TF op)
    '''
    return tf.is_numeric_tensor(x) or isinstance(x, tf.Variable)

def is_lstm(x):
    '''Determine whether x is an LSTMCell or any object derived from one.

    Args:
        x: Any object

    Returns:
        A bool indicating whether x is an LSTMCell or any object derived from
        one.
    '''
    if isinstance(x, tf.nn.rnn_cell.LSTMCell):
        return True

    if isinstance(x, tf.nn.rnn_cell.LSTMStateTuple):
        return True

    return False
