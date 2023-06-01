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
import absl
import pdb

from FixedPointFinderBase import FixedPointFinderBase
from FixedPoints import FixedPoints
import torch

class FixedPointFinderTorch(FixedPointFinderBase):

    def __init__(self,rnn, **kwargs):
        '''Creates a FixedPointFinder object.

        Args:
            rnn: A Pytorch RNN object.

            See FixedPointFinderBase.py for additional keyword arguments.
        '''
        super().__init__(rnn, **kwargs)

    @staticmethod
    def compute_q(x, x_1):
        return  (0.5 * torch.sum(torch.square(x_1 - x), axis=2)).squeeze(0)

    @staticmethod
    def compute_q_scalar(x, x_1):
        return  torch.mean(0.5 * torch.sum(torch.square(x_1 - x), axis=2))#.unsqueeze(-1)


    def _run_optimization_loop(self, q, x, F_x, inputs, rnn):        
        x.requires_grad = True
        optimizer = torch.optim.Adam([x], lr=0.05)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

        # q = self.compute_q(x,F_x)
        q_prev = torch.zeros(size=[x.shape[1]])
        q_scalar = 10000
        iter_count = 0
        x_traj = []
        x_traj.append(x.clone())

        # TODO: remove this , just ensures RNN params are frozen
        # optimizer.param_groups[0]['params'][0].shape
        rnn.eval()
        # self.tol_q = 5e-14
        while True and iter_count < 2000:
            optimizer.zero_grad()
            _, F_x = rnn(inputs,x)
            q = self.compute_q(x,F_x)
            dq = torch.abs(q - q_prev)
            q_scalar = self.compute_q_scalar(x,F_x)#torch.mean(q)
            q_scalar.backward()
            
            optimizer.step()
            scheduler.step()
        
            # shows that RNN weights aren't updating
            # print(torch.mean(rnn.weight_hh_l0), torch.mean(rnn.weight_ih_l0))
            print("q scalar : ", q_scalar.item(), " iter: ", iter_count)

            if iter_count > 1 and np.all(np.logical_or( dq.detach().numpy() < self.tol_dq,q.detach().numpy() < self.tol_q)): # TODO: add learning rate scale
                break
            else: # update q, x
                x_traj.append(x.clone())
                q_prev = q.clone()
                iter_count += 1
        # remove extra dims
        xstar, F_xstar, qstar, dq, n_iters, x_traj = x.squeeze(0), F_x.squeeze(0), q, dq, iter_count, torch.vstack(x_traj)
        n_iters = np.tile(n_iters, reps=F_xstar.shape[0])
        xstar, F_xstar, qstar = xstar.detach().numpy(), F_xstar.detach().numpy(), qstar.detach().numpy()
        return xstar, F_xstar, qstar, dq, n_iters, x_traj
    
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

        x = torch.from_numpy(initial_states).unsqueeze(0).to(torch.float32)
        F_x = torch.zeros(size=x.shape)
        q = self.compute_q(x,F_x)

        # A shape [n,] TF Tensor of objectives (one per initial state) to be
        # combined in _run_optimization_loop.
        xstar, F_xstar, qstar, dq, n_iters, x_traj  = \
            self._run_optimization_loop(q,x, F_x, inputs, self.rnn_cell)
        
        # TODO: input shape problem removes sequence dim from inputs 
        if  len(inputs.shape) > 2:
            inputs = inputs[:,0,:]

        fps = FixedPoints(
            xstar=xstar,
            # x_init=tf_utils.maybe_convert_from_LSTMStateTuple(initial_states),
            inputs=inputs,
            cond_id=cond_ids,
            F_xstar=F_xstar,
            qstar=qstar,
            dq=dq,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.np_dtype)

        return fps

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
        
    def sample_states(self, state_traj, n_inits,valid_bxt=None,noise_scale=0.0):
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
