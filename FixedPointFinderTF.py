'''
TensorFlow FixedPointFinder
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

import tensorflow as tf
tf1 = tf.compat.v1
tf1.disable_eager_execution()
# tf1.disable_v2_behavior()

from FixedPoints import FixedPoints
from AdaptiveLearningRate import AdaptiveLearningRate
from AdaptiveGradNormClip import AdaptiveGradNormClip
from Timer import Timer

class FixedPointFinderTF(FixedPointFinderBase):

    def __init__(self,rnn_cell, sess, **kwargs):
        '''Creates a FixedPointFinder object.

        Args:
            rnn_cell: A Tensorflow RNN cell, which has been initialized or
            restored in the Tensorflow session, 'sess'.

            See FixedPointFinderBase.py for additional keyword arguments.
        '''
        self.session = sess
        self.tf_dtype = getattr(tf, tf_dtype)
        self.np_dtype = self.tf_dtype.as_numpy_dtype
        super().__init__(rnn_cell, **kwargs)

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

        F = F_rnncell

        init = tf1.variables_initializer(var_list=[x])
        self.session.run(init)

        return x, F

    def _run_joint_optimization(self, initial_states, inputs, cond_ids=None):
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

        n = initial_states.shape[0]

        x, F = self._grab_RNN(initial_states, inputs)
        
        # A shape [n,] TF Tensor of objectives (one per initial state) to be
        # combined in _run_optimization_loop.
        q = 0.5 * tf.reduce_sum(input_tensor=tf.square(F - x), axis=1)

        xstar, F_xstar, qstar, dq, n_iters = \
            self._run_optimization_loop(q, x, F)

        fps = FixedPoints(
            xstar=xstar,
            x_init=initial_states,
            inputs=inputs,
            cond_id=cond_ids,
            F_xstar=F_xstar,
            qstar=qstar,
            dq=dq,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.np_dtype)

        return fps

    def _run_single_optimization(self, initial_state, inputs, cond_id=None):
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
        q = 0.5 * tf.reduce_sum(input_tensor=tf.square(F - x))

        xstar, F_xstar, qstar, dq, n_iters = \
            self._run_optimization_loop(q, x, F)

        fp = FixedPoints(
            xstar=xstar,
            x_init=initial_state,
            inputs=inputs,
            cond_id=cond_id,
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

            x: An [n_inits x n_states] tf.Variable (the optimization variable)
            representing RNN states, initialized to the values in
            initial_states. If the RNN is an LSTM, n_states represents the
            concatenated hidden and cell states.

            F: An [n_inits x n_states] tf op representing the state transition
            function of the RNN applied to x.

        Returns:
            ev_x: An [n_inits x n_states] numpy array containing the optimized
            fixed points, i.e., the RNN states that minimize q.

            ev_F: An [n_inits x n_states] numpy array containing the values in
            ev_x after transitioning through one step of the RNN.

            ev_q: A scalar numpy float specifying the value of the objective
            function upon termination of the optimization.

            ev_dq: A scalar numpy float specifying the absolute change in the
            objective function across the final optimization iteration.

            iter_count: An int specifying the number of iterations completed
            before the optimization terminated.
        '''

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

        q_scalar = tf.reduce_mean(input_tensor=q)
        grads = tf.gradients(ys=q_scalar, xs=[x])

        q_prev_tf = tf1.placeholder(self.tf_dtype,
                                   shape=q.shape.as_list(),
                                   name='q_prev')

        # when (q-q_prev) is negative, optimization is making progress
        dq = tf.abs(q - q_prev_tf)

        # Optimizer
        adaptive_learning_rate = AdaptiveLearningRate(
            **self.adaptive_learning_rate_hps)
        learning_rate = tf1.placeholder(self.tf_dtype, name='learning_rate')

        adaptive_grad_norm_clip = AdaptiveGradNormClip(
            **self.grad_norm_clip_hps)
        grad_norm_clip_val = tf1.placeholder(self.tf_dtype,
                                            name='grad_norm_clip_val')

        # Gradient clipping
        clipped_grads, grad_global_norm = tf.clip_by_global_norm(
            grads, grad_norm_clip_val)
        clipped_grad_global_norm = tf.linalg.global_norm(clipped_grads)
        clipped_grad_norm_diff = grad_global_norm - clipped_grad_global_norm
        grads_to_apply = clipped_grads

        # Migrating to TF2 usage: tf.keras.optimizers.Adam
        # Currently, updating to the TF2 usage fails when FPF is used in 
        # conjunction with RecurrentWhisperer (RW), because TF2 doesn't allow 
        # multiple Adam instances, and RW already instantiated one.
        optimizer = tf1.train.AdamOptimizer(
            learning_rate=learning_rate, **self.adam_optimizer_hps)

        train = optimizer.apply_gradients(list(zip(grads_to_apply, [x])))

        # Initialize x and AdamOptimizer's auxiliary variables
        # (very careful not to reinitialize RNN parameters)
        uninitialized_vars = optimizer.variables()
        init = tf1.variables_initializer(var_list=uninitialized_vars)
        self.session.run(init)

        ops_to_eval = [train, x, F, q_scalar, q, dq, grad_global_norm]

        iter_count = 1
        t_start = time.time()
        q_prev = np.tile(np.nan, q.shape.as_list())
        rnn_cell_feed_dict = self.feed_dict

        while True:

            iter_learning_rate = adaptive_learning_rate()
            iter_clip_val = adaptive_grad_norm_clip()

            feed_dict = {learning_rate: iter_learning_rate,
                         grad_norm_clip_val: iter_clip_val,
                         q_prev_tf: q_prev}
            feed_dict.update(rnn_cell_feed_dict)

            (ev_train,
            ev_x,
            ev_F,
            ev_q_scalar,
            ev_q,
            ev_dq,
            ev_grad_norm) = self.session.run(ops_to_eval, feed_dict)

            if self.super_verbose and \
                np.mod(iter_count, self.n_iters_per_print_update)==0:
                self._print_update(iter_count, ev_q, ev_dq, iter_learning_rate)

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
            self._print_update(iter_count, ev_q, ev_dq, iter_learning_rate, is_final=True)

        iter_count = np.tile(iter_count, ev_q.shape)
        return ev_x, ev_F, ev_q, ev_dq, iter_count

    def _compute_recurrent_jacobians(self, fps):
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

        states_np = fps.xstar

        with tf.GradientTape(persistent=True) as tape:
            
            x_tf, F_tf = self._grab_RNN(states_np, inputs_np)

        J_tf = tape.batch_jacobian(F_tf, x_tf)
            
        J_np = self.session.run(J_tf)

        return J_np, J_tf

    def _compute_input_jacobians(self, fps):
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

            F = F_rnncell

            init = tf1.variables_initializer(var_list=[x, inputs_tf])
            self.session.run(init)

            return inputs_tf, F

        inputs_np = fps.inputs

        states_np = fps.xstar

        with tf.GradientTape(persistent=True) as tape:

            inputs_tf, F_tf = grab_RNN_for_dFdu(states_np, inputs_np)

        J_tf = tape.batch_jacobian(F_tf, inputs_tf)

        J_np = self.session.run(J_tf)

        return J_np, J_tf

    # *************************************************************************
    # In development: *********************************************************
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
        input_size = \
            self.rnn_cell.variables[0].shape[0] - \
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
        return self.rnn_cell.state_size

    def approximate_updates(self, states, inputs, fps,
        do_compute_exact_update=True):
        ''' Computes approximate one-step updates based on linearized dynamics
        around fixed points. See _compute_approx_one_step_update() docstring
        for the underlying math.

        This function computes an approximate based on every pair (states[i],
        inputs[i]) and based on the linearized dynamics about every fixed
        point fps[j].

        Args:
            states: numpy array with shape (n, n_states) of RNN states for
            which approximate updates will be computed. In math, each row
            states[i] corresponds to an x(t) as described above.

            inputs: numpy array with shape (n, n_inputs) of RNN inputs. In
            math, each row inputs[i] corresponds to a u(t+1) as described
            above. Inputs are paired with states, such that an update is
            approximated for each pair (states[i], inputs[i]). Alternatively,
            inputs can be a shape (n_inputs,) or (1, n_inputs) numpy array
            specifying a single set of inputs to apply to all state updates.

            fps: A FixedPoints object containing the (possible many) fixed
            points about which to compute linearized dynamics.

            do_compute_exact_update (optional): Bool indicating whether to
            compute the exact one-step updates via the RNN itself
            (Default: True).

        Returns:
            approx_states: shape (k, n, n_states) numpy array containing the
            approximate one-step updated states. Here, k is the number of fixed
            points in fps, and n is the number of state-input pairs in states
            and inputs.

            exact_states (optional): shape (n, n_states) numpy array containing
            the exact one-step updates dates (i.e., using the full RNN). Only
            returned if do_compute_exact_update is True.
        '''

        # This version, all computation done in numpy
        # To do: consider converting to TF for GPU acceleration.
        if hasattr(fps, 'dFdu'):
            dFdu = fps.dFdu
        else:
            print('Computing input Jacobians...', end='')
            dFdu, dFdu_tf = self._compute_input_jacobians(fps)
            print('done.')

        n, n_states = states.shape
        approx_states = np.zeros((fps.n, n, n_states))

        for idx in range(fps.n):
            A = fps.J_xstar[idx] # shape (n_states, n_states)
            B = dFdu[idx] # shape (n_states, n_inputs)

            xstar = fps.xstar[idx] # shape (n_states,)
            u = fps.inputs[idx] # shape (n_inputs,)

            approx_states[idx] = self._compute_approx_one_step_update(
                states, inputs, A, xstar, B, u)

        if not do_compute_exact_update:
            return approx_states
        else:
            x, F = self._grab_RNN(states, inputs)
            true_states = self.session.run(F, feed_dict=self.feed_dict)
            return approx_states, true_states

    def _compute_approx_one_step_update(self, states, inputs, dFdx, xstar, dFdu, u):
        ''' Approximate one-step updates based on linearized dynamics around
        a fixed point.

        In general, the RNN update is:

            x(t+1) = F(x(t), u(t+1)),

        for states x and inputs u. In the strict definition, a state x* is
        considered a fixed point with input u when:

            x* =  F(x*, u)

        (in practice this can be less string to include slow points that can
        also provide meaningful insight into the RNN's dynamics). Near a fixed
        point, the dynamics can be well-approximated by the linearized dynamics
        about that fixed point. I.e., for x(t) near x*, F(x(t), u(t+1)) is
        well-approximated by the first-order Taylor expansion:

            x(t+1) ~ F(x*, u) + A (x(t) - x*) + B (u(t+1) - u)
                   =    x*    + A (x(t) - x*) + B (u(t+1) - u)

        where A is the recurrent Jacobian dF/dx evaluated at (x*, u), B is the
        input Jacobian dF/du evaluated at (x*, u), and ~ denotes approximate
        equivalence. If the RNN is actually a linear dynamical system, then
        this equivalence is exact for any x(t), u (this can be helpful for
        proofs and testing).

        This function computes this approximate update for every state (x(t)),
        input (u(t+1)) pair based on the linearized dynamics about a single
        fixed point.

        Args:
            states: numpy array with shape (n, n_states) of RNN states for
            which approximate updates will be computed. In math, each row
            states[i] corresponds to an x(t) as described above.

            inputs: numpy array with shape (n, n_inputs) of RNN inputs. In
            math, each row inputs[i] corresponds to a u(t+1) as described
            above. Inputs are paired with states, such that an update is
            approximated for each pair (states[i], inputs[i]). Alternatively,
            inputs can be a shape (n_inputs,) or (1, n_inputs) numpy array
            specifying a single set of inputs to apply to all state updates.

            dFdx: numpy array with shape (n_states, n_states) containing the
            Jacobian of the RNN's recurrent dynamics (i.e., with respect to
            the RNN state) at fixed point x* and input u.

            xstar: numpy array with shape (n_states,) containing the fixed
            point (x* in the math above) about which the dynamics were
            linearized.

            dFdu: numpy array with shape (n_states, n_inputs) containing the
            Jacobian of the RNN's input dynamics (i.e., with respect to the
            RNN inputs) at fixed point x* and input u.

            u: numpy array with shape (n_inputs,) containing the input for
            which x* is a fixed point (also u in the math above).

        Returns:
            approx_states: shape (k, n, n_states) numpy array containing the
            approximate one-step updated states.
        '''

        # This version, all computation done in numpy
        # To do: consider converting to TF for GPU acceleration.

        n_states = self._state_size
        n_inputs = self._input_size

        assert (xstar.shape == (n_states,))
        assert (u.shape == (n_inputs,))
        assert (dFdx.shape == (n_states, n_states))
        assert (dFdu.shape == (n_states, n_inputs))
        assert (states.shape[-1] == n_states)
        assert (inputs.shape[-1] == n_inputs)

        term1 = xstar # shape (n_states,)

        # shape (n, n_states) or (1, n_states) or (n_states,)
        term2 = np.matmul(dFdx, np.transpose(states - xstar)).T

        # shape (n, n_states) or (1, n_states) or (n_states,)
        term3 = np.matmul(dFdu, np.transpose(inputs - u)).T

        return term1 + term2 + term3 # shape (n, n_states)

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

            init = tf1.variables_initializer(var_list=[J_tf])
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