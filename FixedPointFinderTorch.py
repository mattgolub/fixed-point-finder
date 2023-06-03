'''
Pytorch FixedPointFinder
Written for Python 3.6.9 and Pytorch (version??)
@ Matt Golub, 2018-2023.
@ Alexander Ladd, 202

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

import torch
from torch.autograd.functional import jacobian

from FixedPointFinderBase import FixedPointFinderBase
from FixedPoints import FixedPoints

class FixedPointFinderTorch(FixedPointFinderBase):

    def __init__(self, rnn, **kwargs):
        '''Creates a FixedPointFinder object.

        Args:
            rnn: A Pytorch RNN object.

            See FixedPointFinderBase.py for additional keyword arguments.
        '''
        self.rnn = rnn
        self.device = next(rnn.parameters()).device
        super().__init__(rnn, **kwargs)
        self.torch_dtype = getattr(torch, self.dtype)

        # Naming conventions assume batch_first==True.
        self._time_dim = 1 if rnn.batch_first else 0
        
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

        n_batch = inputs.shape[0]
        TIME_DIM = self._time_dim

        for p in self.rnn.parameters():
            p.requires_grad = False

        self._print_if_verbose('\tFinding fixed points '
                            'via joint optimization.')

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

        optimizer = torch.optim.Adam([x_1xbxd], lr=0.05)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

        iter_count = 1
        t_start = time.time()
        q_prev_b = torch.full((n_batch,), float('nan'), device=self.device)

        while True:

            # iter_learning_rate = adaptive_learning_rate()
            iter_learning_rate = 1
            # iter_clip_val = adaptive_grad_norm_clip()
            
            optimizer.zero_grad()
            
            _, F_x_1xbxd = self.rnn(inputs_bx1xd, x_1xbxd)

            dx_bxd = torch.squeeze(x_1xbxd - F_x_1xbxd)
            q_b = 0.5 * torch.sum(torch.square(dx_bxd), axis=1)
            q_scalar = torch.mean(q_b)
            dq_b = torch.abs(q_b - q_prev_b)
            q_scalar.backward()
            
            optimizer.step()
            scheduler.step()

            ev_q_scalar = q_scalar.detach().cpu().numpy()
            ev_q_b = q_b.detach().cpu().numpy()
            ev_dq_b = dq_b.detach().cpu().numpy()

            if self.super_verbose and \
                np.mod(iter_count, self.n_iters_per_print_update)==0:
                self._print_iter_update(iter_count, t_start, ev_q, ev_dq, iter_learning_rate)

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
            # adaptive_learning_rate.update(ev_q_scalar)
            # adaptive_grad_norm_clip.update(ev_grad_norm)
            iter_count += 1

        if self.verbose:
            self._print_iter_update(iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate, is_final=True)

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
        raise NotImplementedError()

    def _compute_recurrent_jacobians(self, fps):
        '''Computes the Jacobian of the RNN state transition function at the
        specified fixed points (i.e., partial derivatives with respect to the
        hidden states) assuming constant inputs.

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

        n_batch = 27

        inputs_np = np.random.randn(n_batch,n_inputs)
        states_np = np.random.randn(n_batch, n_states)

        # Unsqueeze to build in time dimension (a single timestep)
        inputs_bxd = torch.from_numpy(inputs_np).to(self.torch_dtype).to(self.device)
        
        x_bxd = torch.from_numpy(states_np)
        x_bxd = x_bxd.to(self.torch_dtype)
        x_bxd = x_bxd.to(self.device)

        def excessive_jacobian(x_bxd, inputs_bxd):
            # This uses excess computation to compute derivatives across the batch,
            # which are always 0).
            # To confirm this, see J_bxdxbxd[i, :, j, :] for all i != j.

            inputs_bx1xd = inputs_bxd.unsqueeze(TIME_DIM) # Used locally--ugly but necessary.

            def forward_fn(x_bxd):
                # Unsqueeze to promote appropriate broadcasting
                x_1xbxd = x_bxd.unsqueeze(0)

                _, F_x_1xbxd = self.rnn(inputs_bx1xd, x_1xbxd)

                F_x_bxd = F_x_1xbxd.squeeze(0)
                return F_x_bxd

            J_bxdxbxd = jacobian(forward_fn, x_bxd, create_graph=False)
            return J_bxdxbxd

            J_bxdxbxd = jacobian(forward_fn, x_bxd, create_graph=False)

        def efficient_jacobian(x_bxd, inputs_bxd):
            # I don't really understand how this works, but it does!
            # # This numerically matches excessive_jacobian() for all i:
            # J_bxdxd[i] <--> J_bxdxbxd[i, :, i, :]
            # https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/6
            
            inputs_bx1xd = inputs_bxd.unsqueeze(TIME_DIM) # Used locally--ugly but necessary.

            def forward_fn(x_bxd):
                # Unsqueeze to promote appropriate broadcasting
                x_1xbxd = x_bxd.unsqueeze(0)

                _, F_x_1xbxd = self.rnn(inputs_bx1xd, x_1xbxd)

                F_x_bxd = F_x_1xbxd.squeeze(0)
                return F_x_bxd

            def batch_jacobian(f, x):
                f_sum = lambda x: torch.sum(f(x), axis=0)
                J_dxbxd = jacobian(f_sum, x)
                J_bxdxd = J_dxbxd.permute(1,0,2)
                return J_bxdxd

            J_bxdxd = batch_jacobian(forward_fn, x_bxd)
            return J_bxdxd

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

        # J1 = excessive_jacobian(x_bxd, inputs_bxd)
        J2 = efficient_jacobian(x_bxd, inputs_bxd)
        # J3 = sequential_jacobian(x_bxd, inputs_bxd)

        J_np = J2.detach().cpu().numpy()

        return J_np
        
    def _compute_input_jacobians(self,fps):
        inputs_np = fps.inputs

        # TODO: inputs shape problem
        if len(inputs_np.shape) == 2:
            inputs_np = inputs_np.unsqueeze(-2)
            
        states_np = fps.xstar

        x = torch.from_numpy(states_np).unsqueeze(0).to(torch.float32)
        states_torch = torch.from_numpy(states_np).unsqueeze(0)
        inputs_torch = fps.inputs.unsqueeze(-2)

        n_inputs, _, input_dim = inputs_torch.shape
        _, n_fixed_points, hidden_dim = states_torch.shape

        J = torch.zeros ((n_inputs, hidden_dim, input_dim))   # loop will fill in Jacobian
        inputs_torch.requires_grad = True

        preds, states = self.rnn_cell(fps.inputs.unsqueeze(-2),states_torch)
        for  i in range(hidden_dim):
            grd = torch.ones(n_inputs,1,input_dim)   # same shape as preds
            inputs_torch.backward(gradient = grd, retain_graph = True)
            J[:,i,:] = inputs_torch.grad.squeeze(-2)   # remove time dim
            inputs_torch.grad.zero_()   # .backward() accumulates gradients, so reset to zero
        dFdu = J.detach().numpy()
        # return None since tf compute graph is not applicable
        return dFdu, None

    def rnn_wrapper(self, inputs,states):
        _, h = self.rnn_cell(inputs, states)
        return h