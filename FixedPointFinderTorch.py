'''
Pytorch FixedPointFinder
Written for Python 3.6.9 and Pytorch (1.12.1)
@ Matt Golub, 2018-2023.
@ Alexander Ladd, 2022

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

import torch
from torch.autograd.functional import jacobian

from FixedPointFinderBase import FixedPointFinderBase
from FixedPoints import FixedPoints

class FixedPointFinderTorch(FixedPointFinderBase):

    def __init__(self, rnn, 
        lr_init=1.0,
        lr_patience=5,
        lr_factor=0.95,
        lr_cooldown=0,
        **kwargs):
        '''Creates a FixedPointFinder object.

        Args:
            rnn: A Pytorch RNN object.

            lr_init: Scalar, initial learning rate. Default: 1.0.

            lr_patience: The 'patience' arg provided to ReduceLROnPlateau().
            Default: 5.

            lr_factor: The 'factor' arg provided to ReduceLROnPlateau().
            Default: 0.95.

            lr_cooldown: The 'cooldown' arg provided to ReduceLROnPlateau().
            Default: 0.

            See FixedPointFinderBase.py for additional keyword arguments.
        '''
        self.rnn = rnn
        self.device = next(rnn.parameters()).device

        self.lr_init = lr_init
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_cooldown = lr_cooldown

        super().__init__(rnn, **kwargs)
        self.torch_dtype = getattr(torch, self.dtype)

        # Naming conventions assume batch_first==True.
        self._time_dim = 1 if rnn.batch_first else 0
        
    def _run_joint_optimization(self, initial_states, inputs, cond_ids=None):
        '''Finds multiple fixed points via a joint optimization over multiple
        state vectors.

        Args:
            initial_states: An [n x n_states] numpy array specifying the initial
            states of the RNN, from which the optimization will search for
            fixed points.

            inputs: A [n x n_inputs] numpy array specifying a set of constant
            inputs into the RNN.

        Returns:
            fps: A FixedPoints object containing the optimized fixed points
            and associated metadata.
        '''

        n_batch = inputs.shape[0]
        TIME_DIM = self._time_dim

        # Ensure that fixed point optimization does not alter RNN parameters.
        for p in self.rnn.parameters():
            p.requires_grad = False

        self._print_if_verbose('\tFinding fixed points via joint optimization.')

        # Unsqueeze to build in time dimension (a single timestep)
        inputs_bx1xd = torch.from_numpy(inputs).unsqueeze(TIME_DIM)
        inputs_bx1xd = inputs_bx1xd.to(self.torch_dtype)
        inputs_bx1xd = inputs_bx1xd.to(self.device)

        # Unsqueeze to promote appropriate broadcasting
        x_1xbxd = torch.from_numpy(initial_states).unsqueeze(0)
        x_1xbxd = x_1xbxd.to(self.torch_dtype)
        x_1xbxd = x_1xbxd.to(self.device)

        inputs_bx1xd.requires_grad = False
        x_1xbxd.requires_grad = True

        init_lr = 0.05
        optimizer = torch.optim.Adam([x_1xbxd], lr=self.lr_init)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        #     step_size=500, 
        #     gamma=0.7)

        # Ideally would use ReduceLROnPlateau, as that is closest to 
        # AdaptiveLearningRate, but RLROP doesn't give ready external access
        # to the current LR, so it's difficult to monitor.
        # 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=.95,
            patience=2,
            cooldown=0)

        iter_count = 1
        iter_learning_rate = init_lr
        t_start = time.time()
        q_prev_b = torch.full((n_batch,), float('nan'), device=self.device)

        while True:
            
            optimizer.zero_grad()
            
            _, F_x_1xbxd = self.rnn(inputs_bx1xd, x_1xbxd)

            dx_bxd = torch.squeeze(x_1xbxd - F_x_1xbxd)
            q_b = 0.5 * torch.sum(torch.square(dx_bxd), axis=1)
            q_scalar = torch.mean(q_b)
            dq_b = torch.abs(q_b - q_prev_b)
            q_scalar.backward()
            
            optimizer.step()
            scheduler.step(metrics=q_scalar)

            iter_learning_rate = scheduler.state_dict()['_last_lr'][0]

            ev_q_scalar = q_scalar.detach().cpu().numpy()
            ev_q_b = q_b.detach().cpu().numpy()
            ev_dq_b = dq_b.detach().cpu().numpy()

            if self.super_verbose and \
                np.mod(iter_count, self.n_iters_per_print_update)==0:
                self._print_iter_update(
                    iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate)

            if iter_count > 1 and \
                np.all(np.logical_or(
                    ev_dq_b < self.tol_dq*iter_learning_rate,
                    ev_q_b < self.tol_q)):
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

            q_prev_b = q_b
            iter_count += 1

        if self.verbose:
            self._print_iter_update(
                iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate, 
                is_final=True)

        # remove extra dims
        xstar = x_1xbxd.squeeze(0)
        xstar = xstar.detach().cpu().numpy()
        
        F_xstar = F_x_1xbxd.squeeze(0)
        F_xstar = F_xstar.detach().cpu().numpy()

        # Indicate same n_iters for each initialization (i.e., joint optimization)        
        n_iters = np.tile(iter_count, reps=F_xstar.shape[0])

        fps = FixedPoints(
            xstar=xstar,
            x_init=initial_states,
            inputs=inputs,
            cond_id=cond_ids,
            F_xstar=F_xstar,
            qstar=ev_q_b,
            dq=ev_dq_b,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.np_dtype)

        return fps

    def _run_single_optimization(self, initial_state, inputs, cond_id=None):
        '''Finds a single fixed point from a single initial state.

        Args:
            initial_state: A [1 x n_states] numpy array specifying an initial
            state of the RNN, from which the optimization will search for
            a single fixed point. 

            inputs: A [1 x n_inputs] numpy array specifying the inputs to the
            RNN for this fixed point optimization.

        Returns:
            A FixedPoints object containing the optimized fixed point and
            associated metadata.
        '''
        
        return self._run_joint_optimization(initial_state, inputs, cond_id=None)

    def _compute_recurrent_jacobians(self, fps):
        '''Computes the Jacobian of the RNN state transition function at the
        specified fixed points assuming fixed inputs for each fixed point
        (i.e., dF/dx, the partial derivatives with respect to the hidden 
        states).

        Args:
            fps: A FixedPoints object containing the RNN states (fps.xstar)
            and inputs (fps.inputs) at which to compute the Jacobians.

        Returns:
            J_np: An [n x n_states x n_states] numpy array containing the
            Jacobian of the RNN state transition function at the states
            specified in fps, given the inputs in fps.
        '''

        TIME_DIM = self._time_dim
        inputs_np = fps.inputs
        states_np = fps.xstar

        n_batch, n_states = states_np.shape
        n_batch, n_inputs = inputs_np.shape

        # For debugging and timing.
        #
        # n_batch = 128
        # inputs_np = np.random.randn(n_batch,n_inputs)
        # states_np = np.random.randn(n_batch, n_states)

        # Unsqueeze to build in time dimension (a single timestep)
        inputs_bxd = torch.from_numpy(inputs_np).to(self.torch_dtype).to(self.device)
        
        x_bxd = torch.from_numpy(states_np)
        x_bxd = x_bxd.to(self.torch_dtype)
        x_bxd = x_bxd.to(self.device)

        def efficient_jacobian(x_bxd, inputs_bxd):
            '''' 
            This numerically matches excessive_jacobian() for all i:
            J_bxdxd[i] <--> J_bxdxbxd[i, :, i, :]
            https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/6
            
            Empirically, computation time is nearly constant wrt batch size! 
    
            i) This function.
            ii) Computing all bxbxdxd elements where all off-dig elements are 0 (excessive_jacobian).
            iii) Sequential computation using for loop along batch dim (sequential_jacobian).

            b=128, d=4:
            i) Total time: 7.37ms. 
            ii) Total time: 254ms. 
            iii) Total time: 324ms. 

            Same for b=27, d=4:
            i) Total time: 7.12ms. 
            ii) Total time: 66.4ms. 
            iii) Total time: 84.2ms.

            Same for b=1, d=4 (EDGE CASE):
            i) Total time: 22.6ms. 
            ii) Total time: 3.19ms. 
            iii) Total time: 5.57ms.
            '''
            
            inputs_bx1xd = inputs_bxd.unsqueeze(TIME_DIM) # Used locally--ugly but necessary.

            def forward_fn(x_bxd):
                ''' Computes x(t+1) as a function of x(t) under fixed inputs. 
                Both x(t) and x(t+1) have shape (n, n_states).
                '''

                # Unsqueeze to promote appropriate broadcasting
                x_1xbxd = x_bxd.unsqueeze(0)

                _, F_x_1xbxd = self.rnn(inputs_bx1xd, x_1xbxd)

                F_x_bxd = F_x_1xbxd.squeeze(0)
                return F_x_bxd

            def batch_jacobian(f, x):
                ''' Computes dF/dx.

                Args:
                    x: shape (n, n_states).
                    f: Torch function that computes x(t+1) from x(t).

                Returns dF/dx, shape (n, n_states, n_states).
                '''

                # This works because dF[i,:]/dx[j,:] is 0 for i!=j. 
                # Here x and F have shape [n, n_states].
                # I'm not sure why this is so much faster in practice, but something in the backend
                # is smarter for this implementation than for excessive_jacobian.

                f_sum = lambda x: torch.sum(f(x), axis=0)
                J_dxbxd = jacobian(f_sum, x)
                J_bxdxd = J_dxbxd.permute(1,0,2)
                return J_bxdxd

            J_bxdxd = batch_jacobian(forward_fn, x_bxd)
            return J_bxdxd

        def excessive_jacobian(x_bxd, inputs_bxd):
            # This uses excess computation to compute derivatives across the batch,
            # which are always 0.
            # To confirm this, see that J_bxbxdxd[i, j, :, :]==0 for all i != j.

            inputs_bx1xd = inputs_bxd.unsqueeze(TIME_DIM) # Used locally--ugly but necessary.

            def forward_fn(x_bxd):
                # Unsqueeze to promote appropriate broadcasting
                x_1xbxd = x_bxd.unsqueeze(0)

                _, F_x_1xbxd = self.rnn(inputs_bx1xd, x_1xbxd)

                F_x_bxd = F_x_1xbxd.squeeze(0)
                return F_x_bxd

            J_bxdxbxd = jacobian(forward_fn, x_bxd, create_graph=False)
            J_bxbxdxd = J_bxdxbxd.permute(0,2,1,3)

            return J_bxbxdxd

        def sequential_jacobian(x_bxd, inputs_bxd):

            def forward_fn(x_d):
                # Unsqueeze to promote appropriate broadcasting
                x_1xbxd = x_d.unsqueeze(0).unsqueeze(1)
                inputs_bx1xd = inputs_d.unsqueeze(0).unsqueeze(1)

                _, F_x_1xbxd = self.rnn(inputs_bx1xd, x_1xbxd)

                F_x_d = F_x_1xbxd.squeeze()
                return F_x_d

            J_list = []

            for i in range(n_batch):
                x_d = x_bxd[i]
                inputs_d = inputs_bxd[i]
                J_i = jacobian(forward_fn, x_d)
                J_list.append(J_i)

            return J_list

        J_bxdxd = efficient_jacobian(x_bxd, inputs_bxd)


        # J_bxbxdxd = excessive_jacobian(x_bxd, inputs_bxd)
        J_list = sequential_jacobian(x_bxd, inputs_bxd)

        J_np = J_bxdxd.detach().cpu().numpy()

        return J_np
        
    def _compute_input_jacobians(self, fps):
        ''' Computes the partial derivatives of the RNN state transition
        function with respect to the RNN's inputs, assuming fixed hidden states.

        Args:
            fps: A FixedPoints object containing the RNN states (fps.xstar)
            and inputs (fps.inputs) at which to compute the Jacobians.

        Returns:
            J_np: An [n x n_states x n_inputs] numpy array containing the
            partial derivatives of the RNN state transition function at the
            inputs specified in fps, given the states in fps.
        '''

        return None