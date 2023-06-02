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

from FixedPointFinderBase import FixedPointFinderBase
from FixedPoints import FixedPoints
from Timer import Timer

class FixedPointFinderTorch(FixedPointFinderBase):

    def __init__(self,rnn, **kwargs):
        '''Creates a FixedPointFinder object.

        Args:
            rnn: A Pytorch RNN object.

            See FixedPointFinderBase.py for additional keyword arguments.
        '''
        self.np_dtype = np.float32
        super().__init__(rnn, **kwargs)

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
        TIME_DIM = 1 # (batch_first=True)

        self._print_if_verbose('\tFinding fixed points '
                            'via joint optimization.')

        # Unsqueeze to build in time dimension (a single timestep)
        x = torch.from_numpy(initial_states).unsqueeze(TIME_DIM).to(torch.float32)
        inputs = torch.from_numpy(inputs).unsqueeze(TIME_DIM).to(torch.float32)
        
        x.requires_grad = True

        optimizer = torch.optim.Adam([x], lr=0.05)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

        # COPIED FROM FPF-TF
        iter_count = 1
        t_start = time.time()
        q_prev = np.nan

        pdb.set_trace()
        # To Do: fix input arguments to rnn_cell. In run_FF_torch.py, args are just (inputs)
        # no previous state.

        while True:

            # iter_learning_rate = adaptive_learning_rate()
            # iter_clip_val = adaptive_grad_norm_clip()
            
            optimizer.zero_grad()
            _, F_x = self.rnn_cell(inputs,x)
            q = torch.mean(0.5 * torch.sum(torch.square(x - F_x), axis=2))
            dq = torch.abs(q - q_prev)
            q.backward()
            
            optimizer.step()
            scheduler.step()

            ev_q = q.cpu()
            ev_dq = dq.cpu()

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
            # adaptive_learning_rate.update(ev_q)
            # adaptive_grad_norm_clip.update(ev_grad_norm)
            iter_count += 1

        if self.verbose:
            self._print_update(iter_count, ev_q, ev_dq, iter_learning_rate, is_final=True)

        # remove extra dims
        inputs = inputs.squeeze(TIME_DIM)
        inputs = inputs.detach().numpy()

        xstar = x.squeeze(TIME_DIM)
        xstar = xstar.detach().numpy()
        
        F_xstar = F_x.squeeze(TIME_DIM)
        F_xstar = F_xstar.detach().numpy()

        qstar = qstar.detach().numpy()
        qstar, dq, n_iters, x_traj = q, dq, iter_count, torch.vstack(x_traj)
        

        # Indicate same n_iters for each initialization (i.e., joint optimization)        
        n_iters = np.tile(n_iters, reps=F_xstar.shape[0])


        # TODO: input shape problem removes sequence dim from inputs 
        if  len(inputs.shape) > 2:
            inputs = inputs[:,0,:]

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
        raise NotImplementedError()

    def _compute_recurrent_jacobians(self,fps):
        inputs_np = fps.inputs

        # TODO: inputs shape problem
        if len(inputs_np.shape) == 2:
            inputs_np = inputs_np.unsqueeze(-2)
            
        states_np = fps.xstar

        x = torch.from_numpy(states_np).unsqueeze(0).to(torch.float32)
        states_torch = torch.from_numpy(states_np).unsqueeze(0)
        _, n_fixed_points, hidden_dim = states_torch.shape
        J = torch.zeros ((n_fixed_points, hidden_dim, hidden_dim))   # loop will fill in Jacobian
        states_torch.requires_grad = True
        preds, states = self.rnn_cell(fps.inputs.unsqueeze(-2),states_torch)
                
        for  i in range(hidden_dim):
            grd = torch.zeros(size=(1,n_fixed_points, hidden_dim))   # same shape as preds
            grd[:, :, i] = 1    # column of Jacobian to compute
            states.backward(gradient = grd, retain_graph = True)
            J[:,:,i] = states_torch.grad   # fill in one column of Jacobian
            states_torch.grad.zero_()   # .backward() accumulates gradients, so reset to zero
        
        dFdx = J.detach().numpy()
        # return None since tf compute graph is not applicable
        return dFdx, None
        
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