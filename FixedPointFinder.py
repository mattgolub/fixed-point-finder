'''
TensorFlow FixedPointFinder
Written using Python 2.7.12 and TensorFlow 1.10.
@ Matt Golub, October 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.ops import parallel_for as pfor
import absl

from FixedPoints import FixedPoints
from AdaptiveLearningRate import AdaptiveLearningRate
from AdaptiveGradNormClip import AdaptiveGradNormClip
from Timer import Timer
import tf_utils

class FixedPointFinder(object):

    default_hps = {
        'tol_q': 1e-12,
        'tol_dq': 1e-20,
        'max_iters': 5000,
        'method': 'joint',
        'do_rerun_q_outliers': False,
        'outlier_q_scale': 10.0,
        'do_exclude_distance_outliers': True,
        'outlier_distance_scale': 10.0,
        'tol_unique': 1e-3,
        'max_n_unique': np.inf,
        'do_compute_jacobians': True,
        'do_decompose_jacobians': True,

        # keep as string (rather than tf.float32)
        # for better argparse handling, yaml writing
        'tf_dtype': 'float32',
        'random_seed': 0,
        'verbose': True,
        'super_verbose': False,
        'n_iters_per_print_update': 100,
        'alr_hps': {},
        'agnc_hps': {},
        'adam_hps': {'epsilon': 0.01},
        'rnn_cell_feed_dict': {},
        }

    def __init__(self, rnn_cell, sess,
        tol_q=default_hps['tol_q'],
        tol_dq=default_hps['tol_dq'],
        max_iters=default_hps['max_iters'],
        method=default_hps['method'],
        do_rerun_q_outliers=default_hps['do_rerun_q_outliers'],
        outlier_q_scale=default_hps['outlier_q_scale'],
        do_exclude_distance_outliers=\
            default_hps['do_exclude_distance_outliers'],
        outlier_distance_scale=default_hps['outlier_distance_scale'],
        tol_unique=default_hps['tol_unique'],
        max_n_unique=default_hps['max_n_unique'],
        do_compute_jacobians=default_hps['do_compute_jacobians'],
        do_decompose_jacobians=default_hps['do_decompose_jacobians'],
        tf_dtype=default_hps['tf_dtype'],
        random_seed=default_hps['random_seed'],
        verbose=default_hps['verbose'],
        super_verbose=default_hps['super_verbose'],
        n_iters_per_print_update=default_hps['n_iters_per_print_update'],
        alr_hps=default_hps['alr_hps'],
        agnc_hps=default_hps['agnc_hps'],
        adam_hps=default_hps['adam_hps'],
        rnn_cell_feed_dict=default_hps['rnn_cell_feed_dict']):
        '''Creates a FixedPointFinder object.

        Optimization terminates once every initialization satisfies one or
        both of the following criteria:
            1. q < tol_q
            2. dq < tol_dq * learning_rate

        Args:
            rnn_cell: A Tensorflow RNN cell, which has been initialized or
            restored in the Tensorflow session, 'sess'.

            tol_q (optional): A positive scalar specifying the optimization
            termination criteria on each q-value. Default: 1e-12.

            tol_dq (optional): A positive scalar specifying the optimization
            termination criteria on the improvement of each q-value (i.e.,
            "dq") from one optimization iteration to the next. Default: 1e-20.

            max_iters (optional): A non-negative integer specifying the
            maximum number of gradient descent iterations allowed.
            Optimization terminates upon reaching this iteration count, even
            if 'tol' has not been reached. Default: 5000.

            method (optional): Either 'joint' or 'sequential' indicating
            whether to find each fixed point individually, or to optimize
            them all jointly. Further testing is required to understand pros
            and cons. Empirically, 'joint' runs faster (potentially making
            better use of GPUs, fewer python for loops), but may be
            susceptible to pathological conditions. Default: 'joint'.

            do_rerun_q_outliers (optional): A bool indicating whether or not
            to run additional optimization iterations on putative outlier
            states, identified as states with large q values relative to the
            median q value across all identified fixed points (i.e., after
            the initial optimization ran to termination). These additional
            optimizations are run sequentially (even if method is 'joint').
            Default: False.

            outlier_q_scale (optional): A positive float specifying the q
            value for putative outlier fixed points, relative to the median q
            value across all identified fixed points. Default: 10.

            do_exclude_distance_outliers (optional): A bool indicating
            whether or not to discard states that are far away from the set
            of initial states, as measured by a normalized euclidean
            distance. If true, states are evaluated and possibly discarded
            after the initial optimization runs to termination.
            Default: True.

            outlier_distance_scale (optional): A positive float specifying a
            normalized distance cutoff used to exclude distance outliers. All
            distances are calculated relative to the centroid of the
            initial_states and are normalized by the average distance-to-
            centroid of the initial_states. Default: 10.

            tol_unique (optional): A positive scalar specifying the numerical
            precision required to label two fixed points as being unique from
            one another. Two fixed points will be considered unique if they
            differ by this amount (or more) along any dimension. This
            tolerance is used to discard numerically similar fixed points.
            Default: 1e-3.

            max_n_unique (optional): A positive integer indicating the max
            number of unique fixed points to keep. If the number of unique
            fixed points identified exceeds this value, points are randomly
            dropped. Default: np.inf.

            do_compute_jacobians (optional): A bool specifying whether or not
            to compute the Jacobian at each fixed point. Default: True.

            do_decompose_jacobians (optional): A bool specifying whether or not
            to eigen-decompose each fixed point's Jacobian. Default: True.

            tf_dtype: string indicating the Tensorflow data type to use for
            all Tensorflow ops and objects. The corresponding numpy data type
            is used for numpy objects and operations. Default: 'float32' -->
            tf.float32.

            random_seed: Seed for numpy random number generator. Default: 0.

            verbose (optional): A bool indicating whether to print high-level
            status updates. Default: True.

            super_verbose (optional): A bool indicating whether or not to
            print per-iteration updates during each optimization. Default:
            False.

            n_iters_per_print_update (optional): An int specifying how often
            to print updates during the fixed point optimizations. Default:
            100.

            alr_hps (optional): A dict containing hyperparameters governing
            an adaptive learning rate. Default: Set by AdaptiveLearningRate.
            See AdaptiveLearningRate.py for more information on these
            hyperparameters and their default values.

            agnc_hps (optional): A dict containing hyperparameters governing
            an adaptive gradient norm clipper. Default: Set by
            AdaptiveGradientNormClip. See AdaptiveGradientNormClip.py
            for more information on these hyperparameters and their default
            values.

            adam_hps (optional): A dict containing hyperparameters governing
            Tensorflow's Adam Optimizer. Default: 'epsilon'=0.01, all other
            hyperparameter defaults set by AdamOptimizer.
        '''

        self.rnn_cell = rnn_cell
        self.rnn_cell_feed_dict = rnn_cell_feed_dict
        self.session = sess
        self.tf_dtype = getattr(tf, tf_dtype)
        self.np_dtype = self.tf_dtype.as_numpy_dtype

        # Make random sequences reproducible
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        self.is_lstm = isinstance(
            rnn_cell.state_size, tf.nn.rnn_cell.LSTMStateTuple)

        # *********************************************************************
        # Optimization hyperparameters ****************************************
        # *********************************************************************

        self.tol_q = tol_q
        self.tol_dq = tol_dq
        self.method = method
        self.max_iters = max_iters
        self.do_rerun_q_outliers = do_rerun_q_outliers
        self.outlier_q_scale = outlier_q_scale
        self.do_exclude_distance_outliers = do_exclude_distance_outliers
        self.outlier_distance_scale = outlier_distance_scale
        self.tol_unique = tol_unique
        self.max_n_unique = max_n_unique
        self.do_compute_jacobians = do_compute_jacobians
        self.do_decompose_jacobians = do_decompose_jacobians
        self.verbose = verbose
        self.super_verbose = super_verbose
        self.n_iters_per_print_update = n_iters_per_print_update

        self.adaptive_learning_rate_hps = alr_hps
        self.grad_norm_clip_hps = agnc_hps
        self.adam_optimizer_hps = adam_hps

    # *************************************************************************
    # Primary exposed functions ***********************************************
    # *************************************************************************

    def sample_inputs_and_states(self, inputs, state_traj, n_inits,
        valid_bxt=None,
        noise_scale=0.0):
        '''Draws random paired samples from the RNN's inputs and hidden-state
        trajectories. Sampled states (but not inputs) can optionally be
        corrupted by independent and identically distributed (IID) Gaussian
        noise. These samples are intended to be used as initial states for
        fixed point optimizations.

        Args:
            inputs: [n_batch x n_time x n_inputs] numpy array containing input
            sequences to the RNN.

            state_traj: [n_batch x n_time x n_states] numpy array or
            LSTMStateTuple with .c and .h as [n_batch x n_time x n_states]
            numpy arrays. Contains state trajectories of the RNN, given inputs.

            n_inits: int specifying the number of sampled states to return.

            valid_bxt (optional): [n_batch x n_time] boolean mask indicated
            the set of trials and timesteps from which to sample. Default: all
            trials and timesteps are assumed valid.

            noise_scale (optional): non-negative float specifying the standard
            deviation of IID Gaussian noise samples added to the sampled
            states. Default: 0.0.

        Returns:
            initial_states: Sampled RNN states as a [n_inits x n_states] numpy
            array or as an LSTMStateTuple with .c and .h as
            [n_inits x n_states] numpy arrays (type matches than of
            state_traj).

        Raises:
            ValueError if noise_scale is negative.
        '''
        if self.is_lstm:
            state_traj_bxtxd = \
                tf_utils.convert_from_LSTMStateTuple(state_traj)
        else:
            state_traj_bxtxd = state_traj

        [n_batch, n_time, n_states] = state_traj_bxtxd.shape
        n_inputs = inputs.shape[2]

        valid_bxt = self._get_valid_mask(n_batch, n_time, valid_bxt=valid_bxt)
        trial_indices, time_indices = \
            self._sample_trial_and_time_indices(valid_bxt, n_inits)

        # Draw random samples from inputs and state trajectories
        input_samples = np.zeros([n_inits, n_inputs])
        state_samples = np.zeros([n_inits, n_states])
        for init_idx in range(n_inits):
            trial_idx = trial_indices[init_idx]
            time_idx = time_indices[init_idx]
            input_samples[init_idx,:] = inputs[trial_idx,time_idx,:]
            state_samples[init_idx,:] = state_traj_bxtxd[trial_idx,time_idx,:]

        # Add IID Gaussian noise to the sampled states
        state_samples = self._add_gaussian_noise(
            state_samples, noise_scale)

        if self.is_lstm:
            return input_samples, tf_utils.convert_to_LSTMStateTuple(states)
        else:
            return input_samples, state_samples

    def sample_states(self, state_traj, n_inits,
                      valid_bxt=None,
                      noise_scale=0.0):
        '''Draws random samples from trajectories of the RNN state. Samples
        can optionally be corrupted by independent and identically distributed
        (IID) Gaussian noise. These samples are intended to be used as initial
        states for fixed point optimizations.

        Args:
            state_traj: [n_batch x n_time x n_states] numpy array or
            LSTMStateTuple with .c and .h as [n_batch x n_time x n_states]
            numpy arrays. Contains example trajectories of the RNN state.

            n_inits: int specifying the number of sampled states to return.

            valid_bxt (optional): [n_batch x n_time] boolean mask indicated
            the set of trials and timesteps from which to sample. Default: all
            trials and timesteps are assumed valid.

            noise_scale (optional): non-negative float specifying the standard
            deviation of IID Gaussian noise samples added to the sampled
            states.

        Returns:
            initial_states: Sampled RNN states as a [n_inits x n_states] numpy
            array or as an LSTMStateTuple with .c and .h as [n_inits x
            n_states] numpy arrays (type matches than of state_traj).

        Raises:
            ValueError if noise_scale is negative.
        '''

        if self.is_lstm:
            state_traj_bxtxd = \
                tf_utils.convert_from_LSTMStateTuple(state_traj)
        else:
            state_traj_bxtxd = state_traj

        [n_batch, n_time, n_states] = state_traj_bxtxd.shape

        valid_bxt = self._get_valid_mask(n_batch, n_time, valid_bxt=valid_bxt)
        trial_indices, time_indices = \
            self._sample_trial_and_time_indices(valid_bxt, n_inits)

        # Draw random samples from state trajectories
        states = np.zeros([n_inits, n_states])
        for init_idx in range(n_inits):
            trial_idx = trial_indices[init_idx]
            time_idx = time_indices[init_idx]
            states[init_idx,:] = state_traj_bxtxd[trial_idx,time_idx,:]

        # Add IID Gaussian noise to the sampled states
        states = self._add_gaussian_noise(states, noise_scale)

        if self.is_lstm:
            return tf_utils.convert_to_LSTMStateTuple(states)
        else:
            return states

    def find_fixed_points(self, initial_states, inputs):
        '''Finds RNN fixed points and the Jacobians at the fixed points.

        Args:
            initial_states: Either an [n x n_dims] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n x n_dims] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.

            inputs: Either a [1 x n_inputs] numpy array specifying a set of
            constant inputs into the RNN to be used for all optimization
            initializations, or an [n x n_inputs] numpy array specifying
            potentially different inputs for each initialization.

        Returns:
            unique_fps: A FixedPoints object containing the set of unique
            fixed points after optimizing from all initial_states. Two fixed
            points are considered unique if all absolute element-wise
            differences are less than tol_unique AND the corresponding inputs
            are unique following the same criteria. See FixedPoints.py for
            additional detail.

            all_fps: A FixedPoints object containing the likely redundant set
            of fixed points (and associated metadata) resulting from ALL
            initializations in initial_states (i.e., the full set of fixed
            points before filtering out putative duplicates to yield
            unique_fps).
        '''
        n = tf_utils.safe_shape(initial_states)[0]

        self._print_if_verbose('\nSearching for fixed points '
                               'from %d initial states.\n' % n)

        if inputs.shape[0] == 1:
            inputs_nxd = np.tile(inputs, [n, 1]) # safe, even if n == 1.
        elif inputs.shape[0] == n:
            inputs_nxd = inputs
        else:
            raise ValueError('Incompatible inputs shape: %s.' %
                str(inputs.shape))

        if self.method == 'sequential':
            all_fps = self._run_sequential_optimizations(
                initial_states, inputs_nxd)
        elif self.method == 'joint':
            all_fps = self._run_joint_optimization(
                initial_states, inputs_nxd)
        else:
            raise ValueError('Unsupported optimization method. Must be either \
                \'joint\' or \'sequential\', but was  \'%s\'' % self.method)

        # Filter out duplicates after from the first optimization round
        unique_fps = all_fps.get_unique()

        self._print_if_verbose('\tIdentified %d unique fixed points.' %
            unique_fps.n)

        if self.do_exclude_distance_outliers:
            unique_fps = \
                self._exclude_distance_outliers(unique_fps, initial_states)

        # Optionally run additional optimization iterations on identified
        # fixed points with q values on the large side of the q-distribution.
        if self.do_rerun_q_outliers:
            unique_fps = \
                self._run_additional_iterations_on_outliers(unique_fps)

            # Filter out duplicates after from the second optimization round
            unique_fps = unique_fps.get_unique()

        # Optionally subselect from the unique fixed points (e.g., for
        # computational savings when not all are needed.)
        if unique_fps.n > self.max_n_unique:
            self._print_if_verbose('\tRandomly selecting %d unique '
                'fixed points to keep.' % self.max_n_unique)
            max_n_unique = int(self.max_n_unique)
            idx_keep = self.rng.choice(
                unique_fps.n, max_n_unique, replace=False)
            unique_fps = unique_fps[idx_keep]

        if self.do_compute_jacobians and unique_fps.n > 0:
            self._print_if_verbose('\tComputing Jacobian at %d '
                                   'unique fixed points.' % unique_fps.n)
            J_np, J_tf = self._compute_multiple_jacobians_np(unique_fps)
            unique_fps.J_xstar = J_np

            if self.do_decompose_jacobians:
                # self._test_decompose_jacobians(unique_fps, J_np, J_tf)
                unique_fps.decompose_jacobians(str_prefix='\t')

        self._print_if_verbose('\tFixed point finding complete.\n')

        return unique_fps, all_fps

    # *************************************************************************
    # Tensorflow Core (these functions will be updated in next major revision)
    # *************************************************************************

    def _build_state_vars(self, initial_states):
        ''' Creates state variables over which to optimize during fixed point
        finding. State variables are setup to be initialized to the
        user-specified initial_states (although formal TF initialization is
        not done here).

        Args:
            initial_states: Either an [n x n_states] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n x n_states/2] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.

        Returns:
            x: An [n x n_states] tf.Variable (the optimization variable)
            representing RNN states, initialized to the values in
            initial_states. If the RNN is an LSTM, n_states represents the
            concatenated hidden and cell states.

            x_rnncell: A potentially reformatted variant of x that serves as
            the second argument to self.rnn_cell. If rnn_cell is an LSTM,
            x_rnncell is formatted as an LSTMStateTuple. Otherwise, it's just
            a reference to x.
        '''

        if self.is_lstm:
            c_h_init = tf_utils.convert_from_LSTMStateTuple(initial_states)
            x = tf.Variable(c_h_init, dtype=self.tf_dtype)
            x_rnncell = tf_utils.convert_to_LSTMStateTuple(x)
        else:
            x = tf.Variable(initial_states, dtype=self.tf_dtype)
            x_rnncell = x

        return x, x_rnncell

    def _grab_RNN(self, initial_states, inputs):
        '''Creates objects for interfacing with the RNN.

        These objects include 1) the optimization variables (initialized to
        the user-specified initial_states) which will, after optimization,
        contain fixed points of the RNN, and 2) hooks into those optimization
        variables that are required for building the TF graph.

        Args:
            initial_states: Either an [n x n_states] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n x n_states/2] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.

            inputs: A [n x n_inputs] numpy array specifying the inputs to the
            RNN for this fixed point optimization.

        Returns:
            x: An [n x n_states] tf.Variable (the optimization variable)
            representing RNN states, initialized to the values in
            initial_states. If the RNN is an LSTM, n_states represents the
            concatenated hidden and cell states.

            F: An [n x n_states] tf op representing the state transition
            function of the RNN applied to x.
        '''

        x, x_rnncell = self._build_state_vars(initial_states)

        inputs_tf = tf.constant(inputs, dtype=self.tf_dtype)

        output, F_rnncell = self.rnn_cell(inputs_tf, x_rnncell)

        if self.is_lstm:
            F = tf_utils.convert_from_LSTMStateTuple(F_rnncell)
        else:
            F = F_rnncell

        init = tf.variables_initializer(var_list=[x])
        self.session.run(init)

        return x, F

    def _run_joint_optimization(self, initial_states, inputs):
        '''Finds multiple fixed points via a joint optimization over multiple
        state vectors.

        Args:
            initial_states: Either an [n x n_states] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n_inits x n_states] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.

            inputs: A [n x n_inputs] numpy array specifying a set of constant
            inputs into the RNN.

        Returns:
            fps: A FixedPoints object containing the optimized fixed points
            and associated metadata.
        '''
        self._print_if_verbose('\tFinding fixed points '
                               'via joint optimization.')

        n, _ = tf_utils.safe_shape(initial_states)

        x, F = self._grab_RNN(initial_states, inputs)

        # A shape [n,] TF Tensor of objectives (one per initial state) to be
        # combined in _run_optimization_loop.
        q = 0.5 * tf.reduce_sum(tf.square(F - x), axis=1)

        xstar, F_xstar, qstar, dq, n_iters = \
            self._run_optimization_loop(q, x, F)

        fps = FixedPoints(
            xstar=xstar,
            x_init=tf_utils.maybe_convert_from_LSTMStateTuple(initial_states),
            inputs=inputs,
            F_xstar=F_xstar,
            qstar=qstar,
            dq=dq,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.np_dtype)

        return fps

    def _run_sequential_optimizations(self, initial_states, inputs,
                                      q_prior=None):
        '''Finds fixed points sequentially, running an optimization from one
        initial state at a time.

        Args:
            initial_states: Either an [n x n_states] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n_inits x n_states] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.

            inputs: An [n x n_inputs] numpy array specifying a set of constant
            inputs into the RNN.

            q_prior (optional): An [n,] numpy array containing q values from a
            previous optimization round. Provide these if performing
            additional optimization iterations on a subset of outlier
            candidate fixed points. Default: None.

        Returns:
            fps: A FixedPoints object containing the optimized fixed points
            and associated metadata.

        '''

        is_fresh_start = q_prior is None

        if is_fresh_start:
            self._print_if_verbose('\tFinding fixed points via '
                                   'sequential optimizations...')

        n_inits, n_states = tf_utils.safe_shape(initial_states)
        n_inputs = inputs.shape[1]

        # Allocate memory for storing results
        fps = FixedPoints(do_alloc_nan=True,
                          n=n_inits,
                          n_states=n_states,
                          n_inputs=n_inputs,
                          dtype=self.np_dtype)

        for init_idx in range(n_inits):

            index = slice(init_idx, init_idx+1)

            initial_states_i = tf_utils.safe_index(initial_states, index)
            inputs_i = inputs[index, :]

            if is_fresh_start:
                self._print_if_verbose('\n\tInitialization %d of %d:' %
                    (init_idx+1, n_inits))
            else:
                self._print_if_verbose('\n\tOutlier %d of %d (q=%.2e):' %
                    (init_idx+1, n_inits, q_prior[init_idx]))

            fps[init_idx] = self._run_single_optimization(
                initial_states_i, inputs_i)

        return fps

    def _run_single_optimization(self, initial_state, inputs):
        '''Finds a single fixed point from a single initial state.

        Args:
            initial_state: A [1 x n_states] numpy array or an
            LSTMStateTuple with initial_state.c and initial_state.h as
            [1 x n_states/2] numpy arrays. These data specify an initial
            state of the RNN, from which the optimization will search for
            a single fixed point. The choice of type must be consistent with
            state type of rnn_cell.

            inputs: A [1 x n_inputs] numpy array specifying the inputs to the
            RNN for this fixed point optimization.

        Returns:
            A FixedPoints object containing the optimized fixed point and
            associated metadata.
        '''

        x, F = self._grab_RNN(initial_state, inputs)
        q = 0.5 * tf.reduce_sum(tf.square(F - x))

        xstar, F_xstar, qstar, dq, n_iters = \
            self._run_optimization_loop(q, x, F)

        fp = FixedPoints(
            xstar=xstar,
            x_init=tf_utils.maybe_convert_from_LSTMStateTuple(initial_state),
            inputs=inputs,
            F_xstar=F_xstar,
            qstar=qstar,
            dq=dq,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.np_dtype)

        return fp

    def _run_optimization_loop(self, q, x, F):
        '''Minimize the scalar function q with respect to the tf.Variable x.

        Args:
            q: An [n_inits,] TF op representing the collection of
            optimization objectives to be minimized. When n_inits > 1, the
            actual optimization objective minimized is a combination of these
            values.

            x: An [n_inits x n_dims] tf.Variable (the optimization variable)
            representing RNN states, initialized to the values in
            initial_states. If the RNN is an LSTM, n_dims represents the
            concatenated hidden and cell states.

            F: An [n_inits x n_dims] tf op representing the state transition
            function of the RNN applied to x.

        Returns:
            ev_x: An [n_inits x n_dims] numpy array containing the optimized
            fixed points, i.e., the RNN states that minimize q.

            ev_F: An [n_inits x n_dims] numpy array containing the values in
            ev_x after transitioning through one step of the RNN.

            ev_q: A scalar numpy float specifying the value of the objective
            function upon termination of the optimization.

            ev_dq: A scalar numpy float specifying the absolute change in the
            objective function across the final optimization iteration.

            iter_count: An int specifying the number of iterations completed
            before the optimization terminated.
        '''
        def print_update(iter_count, q, dq, lr, is_final=False):

            t = time.time()
            t_elapsed = t - t_start
            avg_iter_time = t_elapsed / iter_count

            if is_final:
                delimiter = '\n\t\t'
                print('\t\t%d iters%s' % (iter_count, delimiter), end='')
            else:
                delimiter = ', '
                print('\tIter: %d%s' % (iter_count, delimiter), end='')

            if q.size == 1:
                print('q = %.2e%sdq = %.2e%s' %
                      (q, delimiter, dq, delimiter), end='')
            else:
                mean_q = np.mean(q)
                std_q = np.std(q)

                mean_dq = np.mean(dq)
                std_dq = np.std(dq)

                print('q = %.2e +/- %.2e%s'
                      'dq = %.2e +/- %.2e%s' %
                      (mean_q, std_q, delimiter, mean_dq, std_dq, delimiter),
                      end='')

            print('learning rate = %.2e%s' % (lr, delimiter), end='')

            print('avg iter time = %.2e sec' % avg_iter_time, end='')

            if is_final:
                print('') # Just for the endline
            else:
                print('.')

        '''There are two obvious choices of how to combine multiple
        minimization objectives:

            1--minimize the maximum value.
            2--minimize the mean value.

        While the former allows for direct checks for convergence for each
        fixed point, the latter is empirically much more efficient (more
        progress made in fewer gradient steps).

        max: This should have nonzero gradients only for the state with the
        largest q. If so, in effect, this will wind up doing a sequential
        optimization.

        mean: This should make progress on many of the states at each step,
        which likely speeds things up. However, one could imagine pathological
        situations arising where the objective continues to improve due to
        improvements in some fixed points but not others.'''

        q_scalar = tf.reduce_mean(q)
        grads = tf.gradients(q_scalar, [x])

        q_prev_tf = tf.placeholder(self.tf_dtype,
                                   shape=q.shape.as_list(),
                                   name='q_prev')

        # when (q-q_prev) is negative, optimization is making progress
        dq = tf.abs(q - q_prev_tf)

        # Optimizer
        adaptive_learning_rate = AdaptiveLearningRate(
            **self.adaptive_learning_rate_hps)
        learning_rate = tf.placeholder(self.tf_dtype, name='learning_rate')

        adaptive_grad_norm_clip = AdaptiveGradNormClip(
            **self.grad_norm_clip_hps)
        grad_norm_clip_val = tf.placeholder(self.tf_dtype,
                                            name='grad_norm_clip_val')

        # Gradient clipping
        clipped_grads, grad_global_norm = tf.clip_by_global_norm(
            grads, grad_norm_clip_val)
        clipped_grad_global_norm = tf.global_norm(clipped_grads)
        clipped_grad_norm_diff = grad_global_norm - clipped_grad_global_norm
        grads_to_apply = clipped_grads

        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate, **self.adam_optimizer_hps)
        train = optimizer.apply_gradients(zip(grads_to_apply, [x]))

        # Initialize x and AdamOptimizer's auxiliary variables
        # (very careful not to reinitialize RNN parameters)
        uninitialized_vars = optimizer.variables()
        init = tf.variables_initializer(var_list=uninitialized_vars)
        self.session.run(init)

        ops_to_eval = [train, x, F, q_scalar, q, dq, grad_global_norm]

        iter_count = 1
        t_start = time.time()
        q_prev = np.tile(np.nan, q.shape.as_list())
        while True:

            iter_learning_rate = adaptive_learning_rate()
            iter_clip_val = adaptive_grad_norm_clip()

            feed_dict = {learning_rate: iter_learning_rate,
                         grad_norm_clip_val: iter_clip_val,
                         q_prev_tf: q_prev}
            feed_dict.update(self.rnn_cell_feed_dict)

            (ev_train,
            ev_x,
            ev_F,
            ev_q_scalar,
            ev_q,
            ev_dq,
            ev_grad_norm) = self.session.run(ops_to_eval, feed_dict)

            if self.super_verbose and \
                np.mod(iter_count, self.n_iters_per_print_update)==0:
                print_update(iter_count, ev_q, ev_dq, iter_learning_rate)

            if iter_count > 1 and \
                np.all(np.logical_or(
                    ev_dq < self.tol_dq*iter_learning_rate,
                    ev_q < self.tol_q)):
                '''Here dq is scaled by the learning rate. Otherwise very
                small steps due to very small learning rates would spuriously
                indicate convergence. This scaling is roughly equivalent to
                measuring the gradient norm.'''
                self._print_if_verbose('\tOptimization complete '
                                       'to desired tolerance.')
                break

            if iter_count + 1 > self.max_iters:
                self._print_if_verbose('\tMaximum iteration count reached. '
                                       'Terminating.')
                break

            q_prev = ev_q
            adaptive_learning_rate.update(ev_q_scalar)
            adaptive_grad_norm_clip.update(ev_grad_norm)
            iter_count += 1

        if self.verbose:
            print_update(iter_count,
                         ev_q, ev_dq,
                         iter_learning_rate,
                         is_final=True)

        iter_count = np.tile(iter_count, ev_q.shape)
        return ev_x, ev_F, ev_q, ev_dq, iter_count

    def _compute_multiple_jacobians_np(self, fps):
        '''Computes the Jacobian of the RNN state transition function at the
        specified fixed points (i.e., partial derivatives with respect to the
        hidden states).

        Args:
            fps: A FixedPoints object containing the RNN states (fps.xstar)
            and inputs (fps.inputs) at which to compute the Jacobians.

        Returns:
            J_np: An [n x n_states x n_states] numpy array containing the
            Jacobian of the RNN state transition function at the states
            specified in fps, given the inputs in fps.

            J_tf: The TF op representing the Jacobians.

        '''
        inputs_np = fps.inputs

        if self.is_lstm:
            states_np = tf_utils.convert_to_LSTMStateTuple(fps.xstar)
        else:
            states_np = fps.xstar

        x_tf, F_tf = self._grab_RNN(states_np, inputs_np)

        try:
           J_tf = pfor.batch_jacobian(F_tf, x_tf)
        except absl.flags._exceptions.UnparsedFlagAccessError:
           J_tf = pfor.batch_jacobian(F_tf, x_tf, use_pfor=False)

        J_np = self.session.run(J_tf)

        return J_np, J_tf

    # *************************************************************************
    # Helper functions, no interaction with Tensorflow graph ******************
    # *************************************************************************

    @property
    def _input_size(self):
        ''' Gets the input size of the RNN. This is a bit of a hack to get
        around the fact that rnn_cell.input_size is often undefined, even after
        training and predicting from the model. A fine quirk of Tensorflow. Perhaps this is fixed in more recent versions of Tensorflow.

        Args:
            None.

        Returns:
            An int specifying the size of the RNN's inputs.
        '''
        if self.is_lstm:
            input_size = \
                self.rnn_cell.variables[0].shape[0].value - \
                self.rnn_cell.state_size[0]
        else:
            input_size = \
                self.rnn_cell.variables[0].shape[0].value - \
                self.rnn_cell.state_size

        return input_size

    @property
    def _state_size(self):
        ''' Gets the state size of the RNN. For an LSTM, here state size is
        defined by the concatenated (hidden, cell) size.

        Args:
            None

        Returns:
            An int specifying the size of the RNN's state.
        '''

        if self.is_lstm:
            return self.rnn_cell.state_size[0] + self.rnn_cell.state_size[1]
        else:
            return self.rnn_cell.state_size

    def _sample_trial_and_time_indices(self, valid_bxt, n):
        ''' Generate n random indices corresponding to True entries in
        valid_bxt. Sampling is performed without replacement.

        Args:
            valid_bxt: [n_batch x n_time] bool numpy array.

            n: integer specifying the number of samples to draw.

        returns:
            (trial_indices, time_indices): tuple containing random indices
            into valid_bxt such that valid_bxt[i, j] is True for every
            (i=trial_indices[k], j=time_indices[k])
        '''

        (trial_idx, time_idx) = np.nonzero(valid_bxt)
        sample_indices = self.rng.randint(n, size=n)
        return trial_idx[sample_indices], time_idx[sample_indices]

    @staticmethod
    def _get_valid_mask(n_batch, n_time, valid_bxt=None):
        ''' Returns an appropriately sized boolean mask.

        Args:
            (n_batch, n_time) is the shape of the desired mask.

            valid_bxt: (optional) proposed boolean mask.

        Returns:
            A shape (n_batch, n_time) boolean mask.

        Raises:
            AssertionError if valid_bxt does not have shape (n_batch, n_time)
        '''

        if valid_bxt is None:
            valid_bxt = np.ones((n_batch, n_time), dtype=np.bool)
        else:

            assert (valid_bxt.shape[0] == n_batch and
                valid_bxt.shape[1] == n_time),\
                ('valid_bxt.shape should be %s, but is %s'
                 % ((n_batch, n_time), valid_bxt.shape))

            if not valid_bxt.dtype == np.bool:
                valid_bxt = valid_bxt.astype(np.bool)

        return valid_bxt

    def _add_gaussian_noise(self, data, noise_scale=0.0):
        ''' Adds IID Gaussian noise to Numpy data.

        Args:
            data: Numpy array.

            noise_scale: (Optional) non-negative scalar indicating the
            standard deviation of the Gaussian noise samples to be generated.
            Default: 0.0.

        Returns:
            Numpy array with shape matching that of data.

        Raises:
            ValueError if noise_scale is negative.
        '''

        # Add IID Gaussian noise
        if noise_scale == 0.0:
            return data # no noise to add
        if noise_scale > 0.0:
            return data + noise_scale * self.rng.randn(*data.shape)
        elif noise_scale < 0.0:
            raise ValueError('noise_scale must be non-negative,'
                             ' but was %f' % noise_scale)

    @staticmethod
    def identify_q_outliers(fps, q_thresh):
        '''Identify fixed points with optimized q values that exceed a
        specified threshold.

        Args:
            fps: A FixedPoints object containing optimized fixed points and
            associated metadata.

            q_thresh: A scalar float indicating the threshold on fixed
            points' q values.

        Returns:
            A numpy array containing the indices into fps corresponding to
            the fixed points with q values exceeding the threshold.

        Usage:
            idx = identify_q_outliers(fps, q_thresh)
            outlier_fps = fps[idx]
        '''
        return np.where(fps.qstar > q_thresh)[0]

    @staticmethod
    def identify_q_non_outliers(fps, q_thresh):
        '''Identify fixed points with optimized q values that do not exceed a
        specified threshold.

        Args:
            fps: A FixedPoints object containing optimized fixed points and
            associated metadata.

            q_thresh: A scalar float indicating the threshold on fixed points'
            q values.

        Returns:
            A numpy array containing the indices into fps corresponding to the
            fixed points with q values that do not exceed the threshold.

        Usage:
            idx = identify_q_non_outliers(fps, q_thresh)
            non_outlier_fps = fps[idx]
        '''
        return np.where(fps.qstar <= q_thresh)[0]

    @staticmethod
    def identify_distance_non_outliers(fps, initial_states, dist_thresh):
        ''' Identify fixed points that are "far" from the initial states used
        to seed the fixed point optimization. Here, "far" means a normalized
        Euclidean distance from the centroid of the initial states that
        exceeds a specified threshold. Distances are normalized by the average
        distances between the initial states and their centroid.

        Empirically this works, but not perfectly. Future work: replace
        [distance to centroid of initial states] with [nearest neighbors
        distance to initial states or to other fixed points].

        Args:
            fps: A FixedPoints object containing optimized fixed points and
            associated metadata.

            initial_states: Either an [n x n_states] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n_inits x n_states] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.

            dist_thresh: A scalar float indicating the threshold of fixed
            points' normalized distance from the centroid of the
            initial_states. Fixed points with normalized distances greater
            than this value are deemed putative outliers.

        Returns:
            A numpy array containing the indices into fps corresponding to the
            non-outlier fixed points.
        '''

        if tf_utils.is_lstm(initial_states):
            initial_states = \
                tf_utils.convert_from_LSTMStateTuple(initial_states)

        n_inits = initial_states.shape[0]
        n_fps = fps.n

        # Centroid of initial_states, shape (n_states,)
        centroid = np.mean(initial_states, axis=0)

        # Distance of each initial state from the centroid, shape (n,)
        init_dists = np.linalg.norm(initial_states - centroid, axis=1)
        avg_init_dist = np.mean(init_dists)

        # Normalized distances of initial states to the centroid, shape: (n,)
        scaled_init_dists = np.true_divide(init_dists, avg_init_dist)

        # Distance of each FP from the initial_states centroid
        fps_dists = np.linalg.norm(fps.xstar - centroid, axis=1)

        # Normalized
        scaled_fps_dists = np.true_divide(fps_dists, avg_init_dist)

        init_non_outlier_idx = np.where(scaled_init_dists < dist_thresh)[0]
        n_init_non_outliers = init_non_outlier_idx.size
        print('\t\tinitial_states: %d outliers detected (of %d).'
            % (n_inits - n_init_non_outliers, n_inits))

        fps_non_outlier_idx = np.where(scaled_fps_dists < dist_thresh)[0]
        n_fps_non_outliers = fps_non_outlier_idx.size
        print('\t\tfixed points: %d outliers detected (of %d).'
            % (n_fps - n_fps_non_outliers, n_fps))

        return fps_non_outlier_idx

    def _exclude_distance_outliers(self, fps, initial_states):
        ''' Removes putative distance outliers from a set of fixed points.
        See docstring for identify_distance_non_outliers(...).
        '''

        idx_keep = self.identify_distance_non_outliers(
            fps,
            initial_states,
            self.outlier_distance_scale)
        return fps[idx_keep]

    def _run_additional_iterations_on_outliers(self, fps):
        '''Detects outlier states with respect to the q function and runs
        additional optimization iterations on those states This should only be
        used after calling either _run_joint_optimization or
        _run_sequential_optimizations.

        Args:
            A FixedPoints object containing (partially) optimized fixed points
            and associated metadata.

        Returns:
            A FixedPoints object containing the further-optimized fixed points
            and associated metadata.
        '''

        '''
        Known issue:
            Additional iterations do not always reduce q! This may have to do
            with learning rate schedules restarting from values that are too
            large.
        '''

        def perform_outlier_optimization(fps, method):

            idx_outliers = self.identify_q_outliers(fps, outlier_min_q)
            n_outliers = len(idx_outliers)

            outlier_fps = fps[idx_outliers]
            n_prev_iters = outlier_fps.n_iters
            inputs = outlier_fps.inputs
            initial_states = self._get_rnncell_compatible_states(
                outlier_fps.xstar)

            if method == 'joint':

                self._print_if_verbose('\tPerforming another round of '
                                       'joint optimization, '
                                       'over outlier states only.')

                updated_outlier_fps = self._run_joint_optimization(
                    initial_states, inputs)

            elif method == 'sequential':

                self._print_if_verbose('\tPerforming a round of sequential '
                                       'optimizations, over outlier '
                                       'states only.')

                updated_outlier_fps = self._run_sequential_optimizations(
                    initial_states, inputs, q_prior=outlier_fps.qstar)

            else:
                raise ValueError('Unsupported method: %s.' % method)

            updated_outlier_fps.n_iters += n_prev_iters
            fps[idx_outliers] = updated_outlier_fps

            return fps

        def outlier_update(fps):

            idx_outliers = self.identify_q_outliers(fps, outlier_min_q)
            n_outliers = len(idx_outliers)

            self._print_if_verbose('\n\tDetected %d putative outliers '
                                   '(q>%.2e).' % (n_outliers, outlier_min_q))

            return idx_outliers

        outlier_min_q = np.median(fps.qstar)*self.outlier_q_scale
        idx_outliers = outlier_update(fps)

        if len(idx_outliers) == 0:
            return fps

        '''
        Experimental: Additional rounds of joint optimization. This code
        currently runs, but does not appear to be very helpful in eliminating
        outliers.
        '''
        if self.method == 'joint':
            N_ROUNDS = 0 # consider making this a hyperparameter
            for round in range(N_ROUNDS):

                fps = perform_outlier_optimization(fps, 'joint')

                idx_outliers = outlier_update(fps)
                if len(idx_outliers) == 0:
                    return fps

        # Always perform a round of sequential optimizations on any (remaining)
        # "outliers".
        fps = perform_outlier_optimization(fps, 'sequential')
        outlier_update(fps) # For print output only

        return fps

    def _get_rnncell_compatible_states(self, states):
        '''Converts RNN states if necessary to be compatible with
        self.rnn_cell.

        Args:
            states:
                Either a numpy array or LSTMStateTuple.

        Returns:
            A representation of states that is compatible with self.rnn_cell.
            If self.rnn_cell is an LSTMCell, the representation is as an
            LSTMStateTuple. Otherwise, the representation is a numpy array.
        '''
        if self.is_lstm:
            return tf_utils.convert_to_LSTMStateTuple(states)
        else:
            return states

    def _print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    # *************************************************************************
    # In development: *********************************************************
    # *************************************************************************

    def _compute_dFdu(self, fps):
        ''' Computes the partial derivatives of the RNN state transition
        function with respect to the RNN's inputs.

        Args:
            fps: A FixedPoints object containing the RNN states (fps.xstar)
            and inputs (fps.inputs) at which to compute the Jacobians.

        Returns:
            J_np: An [n x n_states x n_inputs] numpy array containing the
            partial derivatives of the RNN state transition function at the
            inputs specified in fps, given the states in fps.

            J_tf: The TF op representing the partial derivatives.
        '''

        def grab_RNN_for_dFdu(initial_states, inputs):
            # Modified variant of _grab_RNN(), repurposed for dFdu

            # Same as in _grab_RNN()
            x, x_rnncell = self._build_state_vars(initial_states)

            # Different from _grab_RNN(), which builds inputs as tf.constant
            inputs_tf = tf.Variable(inputs, dtype=self.tf_dtype)

            output, F_rnncell = self.rnn_cell(inputs_tf, x_rnncell)

            if self.is_lstm:
                F = tf_utils.convert_from_LSTMStateTuple(F_rnncell)
            else:
                F = F_rnncell

            init = tf.variables_initializer(var_list=[x, inputs_tf])
            self.session.run(init)

            return inputs_tf, F

        inputs_np = fps.inputs

        if self.is_lstm:
            states_np = tf_utils.convert_to_LSTMStateTuple(fps.xstar)
        else:
            states_np = fps.xstar

        inputs_tf, F_tf = grab_RNN_for_dFdu(states_np, inputs_np)

        try:
           J_tf = pfor.batch_jacobian(F_tf, inputs_tf)
        except absl.flags._exceptions.UnparsedFlagAccessError:
           J_tf = pfor.batch_jacobian(F_tf, inputs_tf, use_pfor=False)

        J_np = self.session.run(J_tf)

        return J_np, J_tf

    '''Playing around to see if Jacobian decomposition can be done faster if
    done in Tensorflow (currently it is done in numpy--see FixedPoints.py).

    Answer: empirically TF might be a lot faster, but as of Dec 2019 there does
    not appear to be a full eigendecomposition available in Tensorflow. What is
    available only supports self adjoint matrices, which in general, will not
    match results from numpy.linalg.eig.

    What I've seen:
        TF is ~4x faster on ~100-128D GRU states. Comparison is with single
        1080-TI GPU compared to 32-core I9 CPU (that seems to be fully utilized
        by numpy).
    '''
    def _test_decompose_jacobians(self, unique_fps, J_np, J_tf):

        def decompose_J1(J_tf):
            e_tf, v_tf = tf.linalg.eigh(J_tf)
            e_np, v_np = self.session.run([e_tf, v_tf])
            return e_np, v_np

        def decompose_J2(J_np):
            J_tf = tf.constant(J_np, dtype=tf.complex64)
            return decompose_J1(J_tf)

        def decompose_J3(J_np):
            J_tf = tf.Variable(np.complex64(J_np))

            init = tf.variables_initializer(var_list=[J_tf])
            self.session.run(init)

            return decompose_J1(J_tf)

        timer_eigs = Timer(3)
        timer_eigs.start()

        self._print_if_verbose(
            '\tDecomposing Jacobians in Tensorflow....')

        # This gives real result (not complex)
        # evals, evecs = decompose_J1(J_tf)

        # This returns complex data type, but imaginary components are all 0!
        e2, v2 = decompose_J2(J_np)
        timer_eigs.split('TF v2')

        # This returns complex data type, but imaginary components are all 0!
        e3, v3 = decompose_J3(J_np)
        timer_eigs.split('TF v3')

        self._print_if_verbose('\t\tDone.')

        unique_fps.decompose_jacobians(str_prefix='\t')
        timer_eigs.split('NP eigs')

        timer_eigs.disp()

        '''Look at differences in leading eigenvalue:'''

        # TF sorts in ascending order using REAL PART ONLY. This is a fair
        # comparison since TF is used for both (and thus sorting is the same).
        tf2_vs_tf3 = np.mean(np.abs(e2[:,-1] - e3[:,-1]))
        print('mean abs difference between leading eigenval '
              'using TF methods 2 and 3: %.3e' % tf2_vs_tf3)

        # PROBLEM: FixedPoints sorts in descending order by magnitude. Thus,
        # this is not guaranteed to compare the same eigenvalues, even if both
        # computations are correct. Likely ok for rough comparison since the
        # eigenvalue with the largest real component is typically the one with
        # the largest magnitude--but this is rough. In the current setting, TF
        # computations are returning 0 imaginary component, so real = mag.
        np_vs_tf = np.mean(np.abs(unique_fps.eigval_J_xstar[:,0] - e3[:,-1]))

        pdb.set_trace()
