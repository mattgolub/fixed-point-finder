'''
FixedPointFinderBase
Written for Python 3.6.9 and TensorFlow 2.8.0
@ Matt Golub, 2018-2023.

If you are using FixedPointFinder in research to be published, 
please cite our accompanying paper in your publication:

Golub and Sussillo (2018), "FixedPointFinder: A Tensorflow toolbox for 
identifying and characterizing fixed points in recurrent neural networks," 
Journal of Open Source Software, 3(31), 1003.
https://doi.org/10.21105/joss.01003

Please direct correspondence to mgolub@cs.washington.edu
'''

import numpy as np
import time
from copy import deepcopy
import pdb

from FixedPoints import FixedPoints
# from AdaptiveLearningRate import AdaptiveLearningRate
# from AdaptiveGradNormClip import AdaptiveGradNormClip

class FixedPointFinderBase(object):

    _default_hps = {
        'feed_dict': {},
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
        'dtype': 'float32',
        'random_seed': 0,
        'verbose': True,
        'super_verbose': False,
        'n_iters_per_print_update': 100,
        'alr_hps': {}, # Note: ALR's termination criteria not currently used.
        'agnc_hps': {},
        'adam_hps': {'epsilon': 0.01},
        }

    @staticmethod
    def default_hps():
        ''' Returns a deep copy of the default hyperparameters dict.

        The deep copy protects against external updates to the defaults, which
        in turn protects against unintended interactions with the hashing done
        by the Hyperparameters class.

        Args:
            None.

        Returns:
            dict of hyperparameters.


        '''
        return deepcopy(FixedPointFinder._default_hps)

    def __init__(self, rnn_cell,
        feed_dict=_default_hps['feed_dict'],
        tol_q=_default_hps['tol_q'],
        tol_dq=_default_hps['tol_dq'],
        max_iters=_default_hps['max_iters'],
        method=_default_hps['method'],
        do_rerun_q_outliers=_default_hps['do_rerun_q_outliers'],
        outlier_q_scale=_default_hps['outlier_q_scale'],
        do_exclude_distance_outliers=\
            _default_hps['do_exclude_distance_outliers'],
        outlier_distance_scale=_default_hps['outlier_distance_scale'],
        tol_unique=_default_hps['tol_unique'],
        max_n_unique=_default_hps['max_n_unique'],
        do_compute_jacobians=_default_hps['do_compute_jacobians'],
        do_decompose_jacobians=_default_hps['do_decompose_jacobians'],
        dtype=_default_hps['dtype'],
        random_seed=_default_hps['random_seed'],
        verbose=_default_hps['verbose'],
        super_verbose=_default_hps['super_verbose'],
        n_iters_per_print_update=_default_hps['n_iters_per_print_update'],
        alr_hps=_default_hps['alr_hps'],
        agnc_hps=_default_hps['agnc_hps'],
        adam_hps=_default_hps['adam_hps']):
        '''Creates a FixedPointFinder object.

        Optimization terminates once every initialization satisfies one or
        both of the following criteria:
            1. q < tol_q
            2. dq < tol_dq * learning_rate

        Args:
            rnn_cell: A Pytorch RNN or a Tensorflow RNN cell.

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

            dtype: string indicating the data type to use for all numerical ops
            and objects. Default: 'float32'

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
            hyperparameters and their default values. NOTE: although
            AdaptiveLearningRate can manage termination criteria, this
            functionality is not currently used by FixedPointFinder.

            agnc_hps (optional): A dict containing hyperparameters governing
            an adaptive gradient norm clipper. Default: Set by
            AdaptiveGradientNormClip. See AdaptiveGradientNormClip.py
            for more information on these hyperparameters and their default
            values.

            adam_hps (optional): A dict containing hyperparameters governing
            Tensorflow's Adam Optimizer. Default: 'epsilon'=0.01, all other
            hyperparameter defaults set by AdamOptimizer.
        '''

        self.feed_dict = feed_dict
        self.dtype = dtype
        self.np_dtype = np.dtype(dtype)

        # Make random sequences reproducible
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)


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

            inputs: Sampled RNN inputs as a [n_inits x n_inputs] numpy array.
            These are paired with the states in initial_states (below).

            initial_states: Sampled RNN states as a [n_inits x n_states] numpy
            array or as an LSTMStateTuple with .c and .h as
            [n_inits x n_states] numpy arrays (type matches that of
            state_traj).

        Raises:
            ValueError if noise_scale is negative.
        '''
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

        assert not np.any(np.isnan(state_samples)),\
            'Detected NaNs in sampled states. Check state_traj and valid_bxt.'

        assert not np.any(np.isnan(input_samples)),\
            'Detected NaNs in sampled inputs. Check inputs and valid_bxt.'

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

        state_traj_bxtxd = state_traj

        [n_batch, n_time, n_states] = state_traj_bxtxd.shape

        valid_bxt = self._get_valid_mask(n_batch, n_time, valid_bxt=valid_bxt)
        trial_indices, time_indices = self._sample_trial_and_time_indices(
            valid_bxt, n_inits)

        # Draw random samples from state trajectories
        states = np.zeros([n_inits, n_states])
        for init_idx in range(n_inits):
            trial_idx = trial_indices[init_idx]
            time_idx = time_indices[init_idx]
            states[init_idx,:] = state_traj_bxtxd[trial_idx, time_idx]

        # Add IID Gaussian noise to the sampled states
        states = self._add_gaussian_noise(states, noise_scale)

        assert not np.any(np.isnan(states)),\
            'Detected NaNs in sampled states. Check state_traj and valid_bxt.'

        return states

    def find_fixed_points(self, initial_states, inputs, cond_ids=None):
        '''Finds RNN fixed points and the Jacobians at the fixed points.

        Args:
            initial_states: Either an [n x n_states] numpy array or an
            LSTMStateTuple with initial_states.c and initial_states.h as
            [n x n_states] numpy arrays. These data specify the initial
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
        n = initial_states.shape[0]

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
                initial_states, inputs_nxd, cond_ids=cond_ids)
        elif self.method == 'joint':
            all_fps = self._run_joint_optimization(
                initial_states, inputs_nxd, cond_ids=cond_ids)
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

        if self.do_compute_jacobians:
            if unique_fps.n > 0:

                self._print_if_verbose('\tComputing recurrent Jacobian at %d '
                    'unique fixed points.' % unique_fps.n)
                dFdx = self._compute_recurrent_jacobians(unique_fps)
                unique_fps.J_xstar = dFdx

                self._print_if_verbose('\tComputing input Jacobian at %d '
                    'unique fixed points.' % unique_fps.n)
                dFdu = self._compute_input_jacobians(unique_fps)
                unique_fps.dFdu = dFdu

            else:
                # Allocate empty arrays, needed for robust concatenation
                n_states = unique_fps.n_states
                n_inputs = unique_fps.n_inputs

                shape_dFdx = (0, n_states, n_states)
                shape_dFdu = (0, n_states, n_inputs)

                unique_fps.J_xstar = unique_fps._alloc_nan(shape_dFdx)
                unique_fps.dFdu = unique_fps._alloc_nan(shape_dFdu)
            
            if self.do_decompose_jacobians:
                # self._test_decompose_jacobians(unique_fps, J_np, J_tf)
                unique_fps.decompose_jacobians(str_prefix='\t')

        self._print_if_verbose('\tFixed point finding complete.\n')

        return unique_fps, all_fps

    # *************************************************************************
    # Helper functions ********************************************************
    # *************************************************************************

    def _run_sequential_optimizations(self, initial_states, inputs,
                                      cond_ids=None,
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

        n_inits, n_states = initial_states.shape
        n_inputs = inputs.shape[1]

        # Allocate memory for storing results
        fps = FixedPoints(do_alloc_nan=True,
                          n=n_inits,
                          n_states=n_states,
                          n_inputs=n_inputs,
                          dtype=self.np_dtype)

        for init_idx in range(n_inits):

            initial_states_i = initial_states[init_idx:(init_idx+1)]
            inputs_i = inputs[index]

            if cond_ids is None:
                colors_i = None
            else:
                colors_i = cond_ids[index]

            if is_fresh_start:
                self._print_if_verbose('\n\tInitialization %d of %d:' %
                    (init_idx+1, n_inits))
            else:
                self._print_if_verbose('\n\tOutlier %d of %d (q=%.2e):' %
                    (init_idx+1, n_inits, q_prior[init_idx]))

            fps[init_idx] = self._run_single_optimization(
                initial_states_i, inputs_i, cond_id=colors_i)

        return fps

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
        max_sample_index = len(trial_idx) # same as len(time_idx)
        sample_indices = self.rng.randint(max_sample_index, size=n)

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
            valid_bxt = np.ones((n_batch, n_time), dtype=bool)
        else:

            assert (valid_bxt.shape[0] == n_batch and
                valid_bxt.shape[1] == n_time),\
                ('valid_bxt.shape should be %s, but is %s'
                 % ((n_batch, n_time), valid_bxt.shape))

            if not valid_bxt.dtype == bool:
                valid_bxt = valid_bxt.astype(bool)

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
            initial_states = outlier_fps.xstar
            cond_ids = outlier_fps.cond_id

            if method == 'joint':

                self._print_if_verbose('\tPerforming another round of '
                                       'joint optimization, '
                                       'over outlier states only.')

                updated_outlier_fps = self._run_joint_optimization(
                    initial_states, inputs,
                    cond_ids=cond_ids)

            elif method == 'sequential':

                self._print_if_verbose('\tPerforming a round of sequential '
                                       'optimizations, over outlier '
                                       'states only.')

                updated_outlier_fps = self._run_sequential_optimizations(
                    initial_states, inputs,
                    cond_ids=cond_ids,
                    q_prior=outlier_fps.qstar)

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

    def _print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    @classmethod
    def _print_iter_update(cls, iter_count, t_start, q, dq, lr, is_final=False):

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
