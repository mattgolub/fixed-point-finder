'''
TensorFlow FixedPointFinder
Version 1.1.1
Written using Python 2.7.12 and TensorFlow 1.10.
@ Matt Golub, September 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import sys
import numpy as np
import numpy.random as npr
import tensorflow as tf
from tensorflow.python.ops import parallel_for as pfor
import absl

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PATH_TO_RECURRENT_WHISPERER = '../../recurrent-whisperer/'
sys.path.insert(0, PATH_TO_RECURRENT_WHISPERER)
import tf_utils
from AdaptiveLearningRate import AdaptiveLearningRate
from AdaptiveGradNormClip import AdaptiveGradNormClip

class FixedPointFinder(object):

    def __init__(self, rnn_cell, sess,
        tol=1e-20,
        max_iters=5000,
        method='joint',
        do_rerun_outliers=False,
        outlier_q_scale=1000,
        tol_unique=1e-3,
        do_compute_jacobians=True,
        verbose=False,
        alr_hps=dict(),
        agnc_hps=dict(),
        adam_hps={'epsilon': 0.01}):
        '''Creates a FixedPointFinder object.

        Args:
            rnn_cell: A Tensorflow RNN cell, which has been initialized or
            restored in the Tensorflow session, 'sess'.

            tol (optional): A positive scalar specifying the optimization
            termination criteria on improvement in the objective function.
            Default: 1e-20.

            max_iters (optional): A non-negative integer specifying the
            maximum number of gradient descent iterations allowed.
            Optimization terminates upon reaching this iteration count, even
            if 'tol' has not been reached. Default: 5000.

            method (optional): Either 'joint' or 'sequential' indicating
            whether to find each fixed point individually, or to optimize them
            all jointly. Further testing is required to understand pros and
            cons. Empirically, 'joint' runs faster (potentially making better
            use of GPUs, fewer python for loops), but may be susceptible to
            pathological conditions. Default: 'joint'.

            do_rerun_outliers (optional): A bool indicating whether or not to
            run additional optimization iterations on putative outlier states,
            identified as states with large q values relative to the median q
            value across all identified fixed points (i.e., after the initial
            optimization ran to termination). These additional optimizations
            are run sequentially (even if method=='joint'). Default: False.

            outlier_q_scale (optional): A positive float specifying the q
            value for putative outlier fixed points, relative to the median q
            value across all identified fixed points. Default: 10.

            tol_unique (optional): A positive scalar specifying the numerical
            precision required to label two fixed points as being unique from
            one another. Two fixed points will be considered unique if they
            differ by this amount (or more) along any dimension. This
            tolerance is used to discard numerically similar fixed points.
            Default: 1e-3.

            do_compute_jacobians (optional): A bool specifying whether or not
            to compute the Jacobian at each fixed point. Default: True.

            verbose (optional): A bool specifying whether or not to print
            per-iteration updates during each optimization. Default: False.

            alr_hps (optional): A dict containing hyperparameters governing an
            adaptive learning rate. Default: Set by AdaptiveLearningRate. See
            AdaptiveLearningRate.py for more information on these
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
        self.session = sess

        self.is_lstm = isinstance(
            rnn_cell.state_size, tf.nn.rnn_cell.LSTMStateTuple)

        # These class variables are set by find_fixed_points
        self.initial_states = None
        self.inputs = None
        self.n_inits = None
        self.n_dims = None

        # *********************************************************************
        # Optimization hyperparameters ****************************************
        # *********************************************************************
        self.tol = tol
        self.method = method
        self.max_iters = max_iters
        self.do_rerun_outliers = do_rerun_outliers
        self.outlier_q_scale = outlier_q_scale
        self.tol_unique = tol_unique
        self.do_compute_jacobians = do_compute_jacobians
        self.verbose = verbose

        self.adaptive_learning_rate_hps = alr_hps
        self.grad_norm_clip_hps = agnc_hps
        self.adam_optimizer_hps = adam_hps

        ''' These class variables are set either by _run_joint_optimization or
        _run_sequential_optimizations'''
        self.xstar = None
        self.F_xstar = None
        self.qstar = None
        self.dq = None

        ''' These class variables are set by _identify_unique_fixed_points and
        _compute_jacobians_at_unique_fixed_points'''
        self.unique_xstar = None
        self.unique_Fxstar = None
        self.unique_J_xstar = None
        self.unique_q = None
        self.unique_dq = None

    def sample_states(self, state_traj, n_inits,
                      noise_scale=0.0, rng=npr.RandomState(0)):
        '''Draws random samples from trajectories of the RNN state. Samples
        can optionally be corrupted by independent and identically distributed
        (IID) Gaussian noise. These samples are intended to be used as initial
        states for fixed point optimizations.

        Args:
            state_traj: [n_batch x n_time x n_dims] numpy array or
            LSTMStateTuple with .c and .h as [n_batch x n_time x n_dims] numpy
            arrays. Contains example trajectories of the RNN state.

            n_inits: int specifying the number of sampled states to return.

            noise_scale (optional): non-negative float specifying the standard
            deviation of IID Gaussian noise samples added to the sampled
            states.

        Returns:
            initial_states: Sampled RNN states as a [n_inits x n_dims] numpy
            array or as an LSTMStateTuple with .c and .h as [n_inits x n_dims]
            numpy arrays (type matches than of state_traj).

        Raises:
            ValueError if noise_scale is negative.
        '''
        if self.is_lstm:
            state_traj_bxtxd = tf_utils.convert_from_LSTMStateTuple(
                state_traj)
        else:
            state_traj_bxtxd = state_traj

        [n_batch, n_time, n_states] = state_traj_bxtxd.shape

        # Draw random samples from state trajectories
        states = np.zeros([n_inits, n_states])
        for init_idx in range(n_inits):
            trial_idx = rng.randint(n_batch)
            time_idx = rng.randint(n_time)
            states[init_idx,:] = state_traj_bxtxd[trial_idx,time_idx,:]

        # Add IID Gaussian noise to the sampled states
        if noise_scale > 0.0:
            states += noise_scale * rng.randn(n_inits, n_states)
        elif noise_scale < 0.0:
            raise ValueError('noise_scale must be non-negative,'
                             ' but was %f' % noise_scale)
        else: # noise_scale == 0 --> don't add noise
            pass

        if self.is_lstm:
            return tf_utils.convert_to_LSTMStateTuple(states)
        else:
            return states

    def find_fixed_points(self, initial_states, inputs):
        '''Finds RNN fixed points and the Jacobians at the fixed points.

        Args:
            initial_states: Either an [n_inits x n_dims] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n_inits x n_dims] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.

            inputs: A [1 x n_inputs] numpy array specifying a set of constant
            inputs into the RNN.

        Returns:
            None

        Raises:
            ValueError: Unsupported optimization method. Must be either
            'joint' or 'sequential', but was %s.

        After running, the following class variables contain results of the
        fixed point finding. The general procedure is to (1) run the fixed
        point optimization initialized at each provided initial state, (2)
        identify the unique fixed points, (3) further refine the set of unique
        fixed points by checking for pathological conditions and running
        additional optimization iterations as needed, and (4) find the
        Jacobian of the RNN state transition function at the unique fixed
        points.

        xstar, F_xstar, qstar, dq, n_iters: Numpy arrays containing results
        from step 1) above, the optimization from all of the initial_states
        provided to __init__. Descriptions of the data contained in each of
        these variables are provided below. For each variable, the first
        dimension indexes the initialization, e.g., xstar[i, :] corresponds to
        initial_states[i, :]. Each of these variables has .shape[0] =
        initial_states.shape[0].

        unique_xstar, unique_F_xstar, unique_J_xstar, unique_qstar, unique_dq,
        unique_n_iters: Numpy arrays containing the results from (2-4) above.
        Each of these variables has .shape[0] = n_unique, where n_unique is an
        int specifying the number of unique fixed points identified

        ***********************************************************************

        xstar: An [n_inits x n_dims] numpy array with row xstar[i, :]
        specifying an the fixed point identified from initial_states[i, :].

        F_xstar: An [n_inits x n_dims] numpy array with F_xstar[i, :]
        specifying RNN state after transitioning from the fixed point in xstar[
        i, :]. If the optimization succeeded (e.g., to 'tol') and identified a
        stable fixed point, the state should not move substantially from the
        fixed point (i.e., xstar[i, :] should be very close to F_xstar[i, :]).

        qstar: An [n_inits,] numpy array with qstar[i] containing the
        optimized objective (1/2)(x-F(x))^T(x-F(x)), where
        x = xstar[i, :]^T and F is the RNN transition function (with the
        specified constant inputs).

        dq: An [n_inits,] numpy array with dq[i] containing the absolute
        difference in the objective function after (i.e., qstar[i]) vs before
        the final gradient descent step of the optimization of xstar[i, :].

        n_iters: An [n_inits,] numpy array with n_iters[i] as the number of
        gradient descent iterations completed to yield xstar[i, :].

        ***********************************************************************

        unique_xstar: An [n_unique x n_dims] numpy array, analogous to xstar,
        but containing only the unique fixed points identified.

        unique_J_xstar: An [n_unique x n_dims x n_dims] numpy array with
        unique_J_xstar[i, :, :] containing the Jacobian of the RNN state
        transition function at fixed point unique_xstar[i, :.

        unique_F_xstar: An [n_unique x n_dims] numpy array, analogous to
        F_xstar, but corresponding only to the unique fixed points identified.

        unique_qstar: An [n_unique,] numpy array, analogous to qstar, but
        corresponding only to the unique fixed points identified.

        unique_dq: An [n_unique,] numpy array, analogous to dq, but
        corresponding only to the unique fixed points identified.

        ***********************************************************************

        Note that xstar, F_xstar, unique_xstar, unique_F_xstar and
        unique_J_xstar are all numpy arrays, regardless of whether that type
        is consistent with the state type of rnn_cell (i.e., whether or not
        rnn_cell is an LSTM). This design decision reflects that a Jacobian is
        most naturally expressed as a single matrix (as opposed to a
        collection of matrices representing interactions between LSTM hidden
        and cell states). If one requires state representations as type
        LSTMStateCell, use _convert_to_LSTMStateTuple(...).
        '''
        self.initial_states = initial_states
        self.inputs = inputs

        if self.is_lstm:
            self.n_inits, self.n_dims = self.initial_states.h.shape
        else:
            self.n_inits, self.n_dims = self.initial_states.shape

        if self.method == 'sequential':
            self._run_sequential_optimizations()
        elif self.method == 'joint':
            self._run_joint_optimization()
        else:
            raise ValueError('Unsupported optimization method. Must be either \
                \'joint\' or \'sequential\', but was  \'%s\'' % self.method)

        if self.do_rerun_outliers:
            self._run_additional_iterations_on_outliers()

        self._identify_unique_fixed_points()

        self._compute_jacobians_at_unique_fixed_points()

    def print_summary(self, unique=True):
        '''Prints a summary of the fixed-point-finding optimization.

        Args:
            unique (optional): A bool specifying whether to only print a
            summary relating to the unique fixed points identified. If False,
            the summary will reflect all fixed points identified from all
            initial_states.
        '''
        if unique:
            unique_str = 'unique_'
        else:
            unique_str = ''

        print('\nThe q function at the fixed points:')
        print(getattr(self, unique_str + 'qstar'))

        print('\nChange in the q function from the final iteration \
              of each optimization:')
        print(getattr(self, unique_str + 'dq'))

        print('\nNumber of iterations completed for each optimization:')
        print(getattr(self, unique_str + 'n_iters'))

        print('\nThe fixed points:')
        print(getattr(self, unique_str + 'xstar'))

        print('\nThe fixed points after one state transition:')
        print(getattr(self, unique_str + 'F_xstar'))
        print('(these should be very close to the fixed points)')

        if unique:
            print('\nThe Jacobians at the fixed points:')
            print(self.unique_J_xstar)

    def plot_summary(self,
                     state_traj=None,
                     plot_batch_idx=None,
                     plot_start_time=0,
                     plot_stop_time=None,
                     mode_scale=0.25):
        '''Plots a visualization and analysis of the unique fixed points.

        1) Finds a low-dimensional subspace for visualization via PCA. If
        state_traj is provided, PCA is fit to [all of] those RNN state
        trajectories. Otherwise, PCA is fit to the identified unique fixed
        points. This subspace is 3-dimensional if the RNN state dimensionality
        is >= 3.

        2) Plots the PCA representation of the stable unique fixed points as
        black dots.

        3) Plots the PCA representation of the unstable unique fixed points as
        red dots.

        4) Plots the PCA representation of the modes of the Jacobian at each
        fixed point. By default, only unstable modes are plotted.

        5) (optional) Plots example RNN state trajectories as blue lines.

        Args:
            state_traj (optional): [n_batch x n_time x n_dims] numpy
            array or LSTMStateTuple with .c and .h as
            [n_batch x n_time x n_dims/2] numpy arrays. Contains example
            trials of RNN state trajectories.

            plot_batch_idx (optional): Indices specifying which trials in
            state_traj to plot on top of the fixed points. Default: plot all
            trials.

            plot_start_time (optional): int specifying the first timestep to
            plot in the example trials of state_traj. Default: 0.

            plot_stop_time (optional): int specifying the last timestep to
            plot in the example trials of stat_traj. Default: n_time.

            stop_time (optional):

            mode_scale (optional): Non-negative float specifying the scaling
            of the plotted eigenmodes. A value of 1.0 results in each mode
            plotted as a set of diametrically opposed line segments
            originating at a fixed point, with each segment's length specified
            by the magnitude of the corresponding eigenvalue.

        Returns:
            None.
        '''
        FIG_WIDTH = 6 # inches
        FIG_HEIGHT = 6 # inches
        FONT_WEIGHT = 'bold'

        xstar = self.unique_xstar
        J_xstar = self.unique_J_xstar

        if state_traj is not None:
            # Use LSTMTools here
            if self.is_lstm:
                state_traj_bxtxd = tf_utils.convert_from_LSTMStateTuple(
                    state_traj)
            else:
                state_traj_bxtxd = state_traj

            [n_batch, n_time, n_dims] = state_traj_bxtxd.shape

            # Ensure plot_start_time >= 0
            plot_start_time = np.max([plot_start_time, 0])

            if plot_stop_time is None:
                plot_stop_time = n_time
            else:
                # Ensure plot_stop_time <= n_time
                plot_stop_time = np.min([plot_stop_time, n_time])

            plot_time_idx = range(plot_start_time, plot_stop_time)

        n_inits, n_dims = np.shape(xstar)

        fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT),
                         tight_layout=True)
        if n_dims >= 3:
            pca = PCA(n_components=3)

            if state_traj is not None:
                state_traj_btxd = np.reshape(state_traj_bxtxd,
                    (n_batch*n_time, n_dims))
                pca.fit(state_traj_btxd)
            else:
                pca.fit(xstar)

            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('PC 1', fontweight=FONT_WEIGHT)
            ax.set_zlabel('PC 3', fontweight=FONT_WEIGHT)
            ax.set_ylabel('PC 2', fontweight=FONT_WEIGHT)

            # For generating figure in paper.md
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-1, 0, 1])
            ax.set_zticks([-1, 0, 1])
        else:
            # For 1D or 0D networks (i.e., never)
            pca = None
            ax = fig.add_subplot(111)
            ax.xlabel('Hidden 1', fontweight=FONT_WEIGHT)
            if n_dims == 2:
                ax.ylabel('Hidden 2', fontweight=FONT_WEIGHT)

        if state_traj is not None:
            if plot_batch_idx is None:
                plot_batch_idx = range(n_batch)

            for batch_idx in plot_batch_idx:
                x_idx = state_traj_bxtxd[batch_idx]

                if n_dims >= 3:
                    z_idx = pca.transform(x_idx[plot_time_idx, :])
                else:
                    z_idx = x_idx[plot_time_idx, :]
                self._plot_123d(ax, z_idx, color='b', linewidth=0.2)

        for init_idx in range(n_inits):
            self._plot_fixed_point(
                ax,
                xstar[init_idx:(init_idx+1)],
                J_xstar[init_idx],
                pca,
                scale=mode_scale)

        plt.ion()
        plt.show()
        plt.pause(1e-10)

    def _run_joint_optimization(self):
        '''Finds multiple fixed points via a joint optimization over multiple
        state vectors.

        After running, the following class variables contain the results of
        the optimization (see find_fixed_points for detailed descriptions):

        xstar, F_xstar, qstar, dq, n_iters

        Args:
            None.

        Returns:
            None.

        '''
        x, F, states, new_states = \
            self._grab_RNN(self.initial_states)

        # an array of objectives (one per initial state) to be combined below
        q_1xn = 0.5 * tf.reduce_sum(tf.square(F - x), axis=1)

        '''There are two obvious choices of how to combine multiple objectives
        here: minimizing the maximum value; or minimizing the mean value.
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

        q = tf.reduce_mean(q_1xn)

        print('\nFinding fixed points via joint optimization...')
        self.xstar, self.F_xstar, _, dq, n_iters = \
            self._run_optimization_loop(q, x, F, states, new_states)

        '''Replace mean qstar (the scalar loss function) with the individual q
        values at each fixed point'''
        self.qstar = self.session.run(q_1xn)

        '''Note, there is no meaningful way to get individual dq values
        without following up with at least one gradient step from the
        sequential optimizer for each fixed point. This is likely a good idea
        to do anyways, since it would allow fine tuning at slow points that
        are not local minima, but do not prevent the joint optimization from
        reaching termination criteria.'''

        self.dq = dq * np.ones([self.n_inits])
        self.n_iters = n_iters * np.ones([self.n_inits])

    def _run_sequential_optimizations(self):
        '''Finds fixed points sequentially, running an optimization from one
        initial state at a time.

        After running, the following class variables contain the results of
        the optimization (see find_fixed_points for detailed descriptions):

        xstar, F_xstar, qstar, dq, n_iters

        Args:
            None.

        Returns:
            None.

        '''

        # *********************************************************************
        # Allocate memory for storing results *********************************
        # *********************************************************************

        print('\nFinding fixed points via sequential optimizations...')

        n_dims = self.n_dims
        n_inits = self.n_inits

        if self.is_lstm:
            self.xstar = np.zeros((n_inits, 2*n_dims))
            self.F_xstar = np.zeros((n_inits, 2*n_dims))
        else:
            self.xstar = np.zeros((n_inits, n_dims))
            self.F_xstar = np.zeros((n_inits, n_dims))

        self.qstar = np.zeros(n_inits)
        self.dq = np.zeros(n_inits)
        self.n_iters = np.zeros(n_inits)

        for init_idx in range(self.n_inits):

            # *****************************************************************
            # Prepare initial state *******************************************
            # *****************************************************************
            if self.is_lstm:
                c = self.initial_states.c[init_idx:(init_idx+1), :]
                h = self.initial_states.h[init_idx:(init_idx+1), :]
                initial_state = tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)
            else:
                initial_state = self.initial_states[init_idx:(init_idx+1), :]

            # *****************************************************************
            # Solve for a single fixed point given an initial state ***********
            # *****************************************************************

            print('Initialization %d of %d:' % (init_idx+1, self.n_inits))

            xstar, F_xstar, qstar, dq, n_iters = \
                self._run_single_optimization(initial_state)

            # *****************************************************************
            # Package results *************************************************
            # *****************************************************************

            self.xstar[init_idx, :] = xstar
            self.F_xstar[init_idx, :] = F_xstar
            self.qstar[init_idx] = qstar
            self.dq[init_idx] = dq
            self.n_iters[init_idx] = n_iters

    def _run_additional_iterations_on_outliers(self):
        '''Detects outlier states with respect to the q function and runs
        additional optimization iterations on those states, using the (slow)
        sequential optimization procedure. This should only be used after
        calling either _run_joint_optimization or
        _run_sequential_optimizations. Updates class variables containing
        optimization results.

        Args:
            None.

        Returns:
            None.
        '''
        outlier_min_q = np.median(self.qstar)*self.outlier_q_scale
        is_outlier = self.qstar > outlier_min_q
        idx_outliers = np.where(is_outlier)[0]
        n_outliers = len(idx_outliers)

        print('Detected %d \"outliers.\"' % n_outliers)

        if n_outliers == 0:
            return

        print('Performing additional optimization iterations.')
        for counter, idx in enumerate(idx_outliers):
            print('\n\tOutlier %d of %d (q=%.3e).'
                % (counter+1, n_outliers, self.qstar[idx]))

            initial_state = np.expand_dims(self.xstar[idx],axis=0)
            if self.is_lstm:
                initial_state = tf_utils.convert_to_LSTMStateTuple(
                    initial_state)
            xstar, F_xstar, qstar, dq, n_iters = self._run_single_optimization(
                initial_state)
            self.xstar[idx] = xstar
            self.F_xstar[idx] = F_xstar
            self.qstar[idx] = qstar
            self.dq[idx] = dq
            self.n_iters[idx] += n_iters

    def _grab_RNN(self, initial_states):
        '''Creates objects for interfacing with the RNN.

        These objects include 1) the optimization variables (initialized to
        the user-specified initial_states) which will, after optimization,
        contain fixed points of the RNN, and 2) hooks into those optimization
        variables that are required for building the TF graph.

        Args:
            initial_states: Either an [n_inits x n_dims] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n_inits x n_dims/2] numpy arrays. These data specify the initial
            states of the RNN, from which the optimization will search for
            fixed points. The choice of type must be consistent with state
            type of rnn_cell.

        Returns:
            x: An [n_inits x n_dims] tf.Variable (the optimization variable)
            representing RNN states, initialized to the values in
            initial_states. If the RNN is an LSTM, n_dims represents the
            concatenated hidden and cell states.

            F: An [n_inits x n_dims] tf op representing the state transition
            function of the RNN applied to x.

            states: Contains the same data as in x, but formatted to interface
            with self.rnn_cell (e.g., formatted as LSTMStateTuple if rnn_cell
            is a LSTMCell)

            new_states: Contains the same data as in F, but formatted to
            interface with self.rnn_cell
        '''
        if self.is_lstm:
            # [1 x (2*n_dims)]
            c_h_init = tf_utils.convert_from_LSTMStateTuple(initial_states)

            # [1 x (2*n_dims)]
            x = tf.Variable(c_h_init, dtype=tf.float32)

            states = tf_utils.convert_to_LSTMStateTuple(x)
        else:
            x = tf.Variable(initial_states, dtype=tf.float32)
            states = x

        n_inits = x.shape[0]
        tiled_inputs = np.tile(self.inputs, [n_inits, 1])
        inputs_tf = tf.constant(tiled_inputs, dtype=tf.float32)

        output, new_states = self.rnn_cell(inputs_tf, states)

        if self.is_lstm:
            # [1 x (2*n_dims)]
            F = tf_utils.convert_from_LSTMStateTuple(new_states)
        else:
            F = new_states

        init = tf.variables_initializer(var_list=[x])
        self.session.run(init)

        return x, F, states, new_states

    def _run_single_optimization(self, initial_state):
        '''Finds a single fixed point from a single initial state.

        Args:
            initial_state: A [1 x n_dims] numpy array or an
            LSTMStateTuple with initial_state.c and initial_state.h as
            [1 x n_dims/2] numpy arrays. These data specify an initial
            state of the RNN, from which the optimization will search for
            a single fixed point. The choice of type must be consistent with
            state type of rnn_cell.

        Returns:
            All arguments returned by _run_optimization_loop (with
            n_inits=1)
        '''
        x, F, state, new_state = self._grab_RNN(initial_state)
        q = 0.5 * tf.reduce_sum(tf.square(F - x))
        return self._run_optimization_loop(q, x, F, state, new_state)

    def _run_optimization_loop(self, q, x, F, state, new_state):
        '''Minimize the scalar function q with respect to the tf.Variable x.

        Args:
            q: A scalar TF op representing the optimization objective to be
            minimized.

            x: An [n_inits x n_dims] tf.Variable (the optimization variable)
            representing RNN states, initialized to the values in
            initial_states. If the RNN is an LSTM, n_dims represents the
            concatenated hidden and cell states.

            F: An [n_inits x n_dims] tf op representing the state transition
            function of the RNN applied to x.

            states: Contains the same data as in x, but formatted to interface
            with self.rnn_cell (e.g., formatted as LSTMStateTuple if rnn_cell
            is a LSTMCell)

            new_states: Contains the same data as in F, but formatted to
            interface with self.rnn_cell

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
        def print_update(iter_count, q, dq, lr):
            print('\tIter: %d, q = %.3e, diff(q) = %.3e, learning rate = %.3e.'
                % (iter_count, q, dq, lr))

        def print_final_summary(iter_count, q, dq, lr):
            print('\t%d iters, q = %.3e, diff(q) = %.3e, learning rate = %.3e.'
                % (iter_count, q, dq, lr))

        def print_values(x, F):
            print('\t\tx = \t\t' + str(x))
            print('\t\tF(x) = \t\t' + str(F))

        # If LSTM, here n_dims reflects the concatenated state
        n_inits, n_dims = x.shape

        q_prev_tf = tf.placeholder(tf.float32, name='q_prev')

        # when (q-q_prev) is negative, optimization is making progress
        dq = tf.abs(q-q_prev_tf)

        # Optimizer
        adaptive_learning_rate = AdaptiveLearningRate(
            **self.adaptive_learning_rate_hps)
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        adaptive_grad_norm_clip = AdaptiveGradNormClip(
            **self.grad_norm_clip_hps)
        grad_norm_clip_val = tf.placeholder(tf.float32,
                                            name='grad_norm_clip_val')

        grads = tf.gradients(q, [x])

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

        ops_to_eval = [train, x, F, q, dq, grad_global_norm]

        iter_count = 1
        q_prev = np.nan
        while True:

            iter_learning_rate = adaptive_learning_rate()
            iter_clip_val = adaptive_grad_norm_clip()

            feed_dict = {learning_rate: iter_learning_rate,
                         grad_norm_clip_val: iter_clip_val,
                         q_prev_tf: q_prev}
            ev_train, ev_x, ev_F, ev_q, ev_dq, ev_grad_norm = \
                self.session.run(ops_to_eval, feed_dict)

            if self.verbose:
                print_update(iter_count, ev_q, ev_dq, iter_learning_rate)
                # print_values(ev_x, ev_F)

            if iter_count > 1 and ev_dq/iter_learning_rate < self.tol:
                '''Here dq is scaled by the learning rate. Otherwise very
                small steps due to very small learning rates would spuriously
                indicate convergence. This scaling is roughly equivalent to
                measuring the gradient norm.'''
                print('\tOptimization complete to desired tolerance.')
                break

            if iter_count+1 > self.max_iters:
                print('\tMaximum iteration count reached. Terminating.')
                break

            q_prev = ev_q
            adaptive_learning_rate.update(ev_q)
            adaptive_grad_norm_clip.update(ev_grad_norm)
            iter_count += 1

        print_final_summary(iter_count, ev_q, ev_dq, iter_learning_rate)

        return ev_x, ev_F, ev_q, ev_dq, iter_count

    def _identify_unique_fixed_points(self):
        '''Identifies the unique fixed points found after optimizing from all
        initiali_states

        After running, the following class variables contain the data
        corresponding to the unique fixed points (see find_fixed_points for
        detailed descriptions):

        unique_xstar, unique_F_xstar, unique_qstar, unique_dq, unique_n_iters

        Args:
            None.

        Returns:
            None.

        '''
        def unique_rows(x, approx_tol):
            # Quick and dirty. Can update using pdist if necessary
            d = int(np.round(np.max([0 -np.log10(approx_tol)])))
            ux, idx = np.unique(x.round(decimals=d),
                                axis=0,
                                return_index=True)
            return ux, idx

        self.unique_xstar, idx = unique_rows(self.xstar, self.tol_unique)

        self.n_unique = len(idx)
        self.unique_F_xstar = self.F_xstar[idx]

        if self.is_lstm:
            self.unique_states = \
                tf_utils.convert_to_LSTMStateTuple(self.unique_xstar)
            self.unique_new_states = \
                tf_utils.convert_to_LSTMStateTuple(self.unique_F_xstar)
        else:
            self.unique_states = self.unique_xstar
            self.unique_new_states = self.unique_F_xstar

        self.unique_qstar = self.qstar[idx]
        self.unique_dq = self.dq[idx]
        self.n_iters = self.n_iters[idx]
        '''In normal operation, Jacobians haven't yet been computed, so don't
        bother indexing into those.'''

    def _compute_jacobians_at_unique_fixed_points(self):
        '''Computes Jacobians at the fixed points identified as being unique.

        After running, the class variable unique_J_xstar contains the
        Jacobians of the fixed points in self.unique_xstar (see
        find_fixed_points for detailed variable descriptions).

        Args:
            None.

        Returns:
            None.
        '''
        print('Computing Jacobians at unique %d fixed points' % self.n_unique)

        self.unique_J_xstar = self._compute_multiple_jacobians_np(
            self.unique_states)

    def _compute_multiple_jacobians_np(self, states_np):
        '''Computes the Jacobian of the RNN state transition function.

        Args:
            states_np: An [n_inits x n_dims] numpy array containing the states
            at which to compute the Jacobian.

        Returns:
            J_np: An [n_inits x n_dims x n_dims] numpy array containing the
            Jacobian of the RNN state transition function at the states
            specified in states_np.

        '''
        x, F, states, new_states = self._grab_RNN(states_np)
        try:
           J_tf = pfor.batch_jacobian(F, x)
        except absl.flags._exceptions.UnparsedFlagAccessError:
           J_tf = pfor.batch_jacobian(F, x, use_pfor=False)

        J_np = self.session.run(J_tf)

        return J_np

    @staticmethod
    def _plot_fixed_point(ax, xstar, J, pca,
        scale=1.0, max_n_modes=3, do_plot_stable_modes=False):
        '''Plots a single fixed point and its dominant eigenmodes.

        Args:
            ax: Matplotlib figure axis on which to plot everything.

            xstar: [1 x n_dims] numpy array representing the fixed point to be
            plotted.

            J: [n_dims x n_dims] numpy array containing the Jacobian of the
            RNN transition function at fixed point xstar.

            pca: PCA object as returned by sklearn.decomposition.PCA. This is
            used to transform the high-d state space representations into 3-d
            for visualization.

            scale (optional): Scale factor for stretching (>1) or shrinking
            (<1) lines representing eigenmodes of the Jacobian. Default: 1.0 (
            unity).

            max_n_modes (optional): Maximum number of eigenmodes to plot. Default: 3.

            do_plot_stable_modes (optional): bool indicating whether or not to plot lines representing stable modes (i.e., eigenvectors of the Jacobian whose eigenvalue magnitude is less than one).

        Returns:
            None.
        '''
        n_dims = xstar.shape[1]
        e_vals, e_vecs = np.linalg.eig(J)
        sorted_e_val_idx = np.argsort(np.abs(e_vals))

        if max_n_modes > len(e_vals):
            max_n_modes = e_vals

        for mode_idx in range(max_n_modes):
            idx = sorted_e_val_idx[-(mode_idx+1)] # -[1, 2, ..., max_n_modes]

            # Magnitude of complex eigenvalue
            e_val_mag = np.abs(e_vals[idx])

            if e_val_mag > 1.0 or do_plot_stable_modes:

                # Already real. Cast to avoid warning.
                e_vec = np.real(e_vecs[:,idx])

                # [1 x d] numpy arrays
                xstar_plus = xstar + scale*e_val_mag*e_vec
                xstar_minus = xstar - scale*e_val_mag*e_vec

                # [3 x d] numpy array
                xstar_mode = np.vstack((xstar_minus, xstar, xstar_plus))

                if e_val_mag < 1.0:
                    color = 'k'
                else:
                    color = 'r'

                if n_dims >= 3:
                    # [3 x 3] numpy array
                    zstar_mode = pca.transform(xstar_mode)
                else:
                    zstar_mode = x_star_mode

                FixedPointFinder._plot_123d(ax, zstar_mode, color=color)

        is_stable = all(np.abs(e_vals) < 1.0)
        if is_stable:
            color = 'k'
        else:
            color = 'r'

        if n_dims >= 3:
            zstar = pca.transform(xstar)
        else:
            zstar = xstar

        FixedPointFinder._plot_123d(
            ax, zstar, color=color, marker='.', markersize=12)

    @staticmethod
    def _plot_123d(ax, z, **kwargs):
        '''Plots in 1D, 2D, or 3D.

        Args:
            ax: Matplotlib figure axis on which to plot everything.

            z: [n x n_dims] numpy array containing data to be plotted,
            where n_dims is 1, 2, or 3.

            any keyword arguments that can be passed to ax.plot(...).

        Returns:
            None.
        '''
        n_dims = z.shape[1]
        if n_dims ==3:
            ax.plot(z[:, 0], z[:, 1], z[:, 2], **kwargs)
        elif n_dims == 2:
            ax.plot(z[:, 0], z[:, 1], **kwargs)
        elif n_dims == 1:
            ax.plot(z, **kwargs)

    # ******************************************************************
    # Previous Jacobian implementations, now obsolete w/ TF 1.10 *******
    # ***** Can be re-integrated by users who require using earlier ****
    # ***** versions of TensorFlow *************************************
    # ******************************************************************

    def _compute_single_jacobian_np_old(self, state_np):
        '''Computes the Jacobian of the RNN state transition function at a
        single point.

        Uses 'brute force' assembling of tf.gradient calls. Very slow.

        Args:
            state_np: A [1 x n_dims] numpy array representing the point at
            which to compute the Jacobian.

        Returns:
            An [n_dims x n_dims] numpy array containing the Jacobian of the
            RNN state transition function at state_np.
        '''

        x, F, state, new_state = self._grab_RNN(state_np, self.inputs)
        J = self._compute_single_jacobian_tf(F, x)

        return J

    def _compute_single_jacobian_np_new(self, state_np):
        '''Computes the Jacobian of the RNN state transition function at a
        single point.

        Uses TF implementation of Jacobian computation, and is somewhat faster
        than _compute_single_jacobian_np_old (about 50% faster on small toy
        problems).

        Args:
            state_np: A [1 x n_dims] numpy array representing the point at
            which to compute the Jacobian.

        Returns:
            An [n_dims x n_dims] numpy array containing the Jacobian of the
            RNN state transition function at state_np.
        '''

        x, F, state, new_state = self._grab_RNN(state_np, self.inputs)
        J_tf = tf.reshape(pfor.jacobian(F, x), [state_np.size, state_np.size])

        return self.session.run(J_tf)

    def _compute_single_jacobian_tf(self, F, x):
        '''Computes the Jacobian of F at a single point, x.

        Args:
            F: A [1 x n_dims] tf.Tensor
            x: A [1 x n_dims] tf.Variable.

        Returns:
            An [n_dims x n_dims] numpy array containing the Jacobian of F at x.

        Raises:
            ValueError: You should be calling _compute_multiple_jacobians_tf.
        '''

        n_inits, n_dims = x.shape

        if n_inits > 1:
            raise ValueError('You should be calling \
                             _compute_multiple_jacobians_tf.')

        print('\tComputing Jacobian at fixed point.')

        # Assemble list of TF ops that will each compute the gradient
        # of one dimension of the RNN function w.r.t. each dimension
        # of the state
        J_tf = []
        for i in range(n_dims):
            J_tf.append(tf.gradients(F[0, i], x)[0])

        # Run that list of TF ops
        J_tf_eval = self.session.run(J_tf)

        # Assemble the evaluated TF ops into a numpy array
        J_np = np.zeros([n_dims, n_dims])
        for dim_idx in range(n_dims):
            J_np[dim_idx, :] = J_tf_eval[dim_idx]

        return J_np

    def _compute_multiple_jacobians_tf(self, F, x):
        '''Computes the Jacobian of F at multiple points, x.

        Args:
            F: An [n_inits x n_dims] tf.Tensor
            x: An [n_inits x n_dims] tf.Variable

        Returns:
            An [n_inits x n_dims x n_dims] numpy array containing the Jacobian
            of F at the points in x.
        '''

        '''Ideally, this could just call
        _compute_single_jacobian_tf(F[i,j],x[i, :])

        Unfortunately, x[i, :] is a TF op (no longer a TF variable, like x is),
        and as a result tf.gradients throws an error. I see two possible ways
        around this. Need to test to see which is faster, but I'm guessing #1
        is faster.

        1) Use tf.gradients(F[i,j],x), but only run the ops for row i.
            Advantage: Doesn't require adding new (x,F) elements to the graph.
            Disadvantage: asking TF to compute a bunch of gradients we already
            know a priori to be 0.
        2) Create new tf.Variables x_i and corresponding new F_xi. Then send
        to _compute_single_jacobian_tf.
            Advantage, better code reuse and readability.
            Disadvantage: considerable back-end overhead (adding additional
            (x,F) elements to the graph).

        Here is approach 1.'''

        n_inits, n_dims = x.shape
        J_np = np.zeros((n_inits, n_dims, n_dims))

        for init_idx in range(n_inits):
            print('\tComputing Jacobians at fixed point %d of %d.' \
                  % (init_idx+1, n_inits))

            J_tf = []

            '''Assemble list of TF ops that will each compute the gradient
            of one dimension of the RNN function w.r.t. each element of
            each state (knowing ahead of time that gradients can only be
            non-zero w.r.t. x[init_idx, :]).'''
            for dim_idx in range(n_dims):
                grad = tf.gradients(F[init_idx, dim_idx], x)[0]

                # Only run those non-zero gradients
                J_tf.append(grad[init_idx, :])

            # Run that list of TF ops
            J_tf_eval = self.session.run(J_tf)

            # Assemble the evaluated TF ops into a numpy array
            for dim_idx in range(n_dims):
                J_np[init_idx, dim_idx, :] = np.squeeze(J_tf_eval[dim_idx])

        return J_np
