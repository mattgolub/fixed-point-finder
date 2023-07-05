'''
examples/torch/FlipFlop.py
Written for Python 3.8.17 and Pytorch 2.0.1
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb

import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PATH_TO_FIXED_POINT_FINDER = '../../'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from FixedPoints import FixedPoints

from FlipFlopData import FlipFlopData

class FlipFlopDataset(Dataset):

	def __init__(self, data, device='cpu'):
		'''
		Args:
			data:
				Numpy data dict as returned by FlipFlopData.generate_data()

		Returns:
			None.
		'''
		
		super().__init__()
		self.device = device
		self.data = data

	def __len__(self):
		''' Returns the total number of trials contained in the dataset.
		'''
		return self.data['inputs'].shape[0]
	
	def __getitem__(self, idx):
		''' 
		Args:
			idx: slice indices for indexing into the batch dimension of data 
			tensors.

		Returns:
			Dict of indexed torch.tensor objects, with key/value pairs 
			corresponding to those in self.data.

		'''
		
		inputs_bxtxd = torch.tensor(
			self.data['inputs'][idx], 
			device=self.device)

		targets_bxtxd = torch.tensor(
			self.data['targets'][idx], 
			device=self.device)

		return {
			'inputs': inputs_bxtxd, 
			'targets': targets_bxtxd
			}

class FlipFlop(nn.Module):

	def __init__(self, n_inputs, n_hidden, n_outputs,
		rnn_type='tanh'):

		super().__init__()

		self.n_inputs = n_inputs
		self.n_hidden = n_hidden
		self.n_outputs = n_outputs
		self.rnn_type = rnn_type.lower()
		self.device = self._get_device()

		zeros_1xd = torch.zeros(1, n_hidden, device=self.device)

		self.initial_hiddens_1xd = nn.Parameter(zeros_1xd)

		self._is_lstm = False

		if self.rnn_type in ['tanh', 'relu']:
		
			self.rnn = nn.RNN(n_inputs, n_hidden,
				nonlinearity=self.rnn_type,
				batch_first=True, 
				device=self.device)

		elif self.rnn_type=='gru':

			self.rnn = nn.GRU(n_inputs, n_hidden, 
				batch_first=True, 
				device=self.device)

		elif self.rnn_type=='lstm':

			self._is_lstm = True
			self.initial_cell_1xd = nn.Parameter(zeros_1xd)
			self.rnn = nn.LSTM(n_inputs, n_hidden, 
				batch_first=True, 
				device=self.device)

		else:
			raise ValueError('Unsupported rnn_type: \'%s\'' % rnn_type)

		self.readout = nn.Linear(n_hidden, n_outputs, device=self.device)

		# Create the loss function
		self._loss_fn = nn.MSELoss()
		
	def forward(self, data):
		'''
		Args:
			data: dict of torch.tensor as returned by 
			FlipFlopDataset.__getitem__()

		Returns:
			dict containing the following key/value pairs:
				
				'output': shape (n_batch, n_time, n_bits) torch.tensor 
				containing the outputs of the FlipFlop.

				'hidden': shape (n_batch, n_time, n_hidden) torch.tensor
				containing the hidden unit activitys of the FlipFlop RNN.
		'''

		inputs_bxtxd = data['inputs']
		batch_size = inputs_bxtxd.shape[0]

		# Expand initial hidden state to match batch size. This creates a new 
		# view without actually creating a new copy of it in memory.
		initial_hiddens_1xbxd = self.initial_hiddens_1xd.expand(
			1, batch_size, self.n_hidden)

		if self._is_lstm:
			initial_cell_1xbxd = self.initial_cell_1xd.expand(
				1, batch_size, self.n_hidden)

			initial_lstm_state = (initial_hiddens_1xbxd, initial_cell_1xbxd)

			# Pass the input through the RNN layer
			hiddens_bxtxd, _ = self.rnn(inputs_bxtxd, initial_lstm_state)    
		
		else:
			# Pass the input through the RNN layer
			hiddens_bxtxd, _ = self.rnn(inputs_bxtxd, initial_hiddens_1xbxd)        

		outputs_bxtxd = self.readout(hiddens_bxtxd)

		return {
			'output': outputs_bxtxd, 
			'hidden': hiddens_bxtxd,
			}

	def predict(self, data):
		''' Runs a forward pass through the model, starting with Numpy data and
		returning Numpy data.

		Args:
			data:
				Numpy data dict as returned by FlipFlopData.generate_data()

		Returns:
			dict matching that returned by forward(), but with all tensors as
			detached numpy arrays on cpu memory.

		'''
		dataset = FlipFlopDataset(data, device=self.device)
		pred_np = self._forward_np(dataset[:len(dataset)])

		return pred_np

	def _tensor2numpy(self, data):

		np_data = {}

		for key, val in data.items():
			np_data[key] = data[key].cpu().numpy()

		return np_data

	def _forward_np(self, data):

		with torch.no_grad():
			pred = self.forward(data)

		pred_np = self._tensor2numpy(pred)

		return pred_np

	def _loss(self, data, pred):

		return self._loss_fn(pred['output'], data['targets'])

	def train(self, train_data, valid_data, 
		learning_rate=1.0,
		batch_size=128,
		min_loss=1e-4, 
		disp_every=1, 
		plot_every=5, 
		max_norm=1.):

		train_dataset = FlipFlopDataset(train_data, device=self.device)
		valid_dataset = FlipFlopDataset(valid_data, device=self.device)

		dataloader = DataLoader(train_dataset, 
			shuffle=True,
			batch_size=batch_size)

		# Create the optimizer
		optimizer = optim.Adam(self.parameters(), 
			lr=learning_rate,
			eps=0.001,
			betas=(0.9, 0.999))

		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, 
			mode='min',
			factor=.95,
			patience=1,
			cooldown=0)

		epoch = 0
		losses = []
		grad_norms = []
		fig = None
		
		while True:

			t_start = time.time()

			if epoch % plot_every == 0:
				valid_pred = self._forward_np(valid_dataset[0:1])
				fig = FlipFlopData.plot_trials(valid_data, valid_pred, fig=fig)

			avg_loss, avg_norm = self._train_epoch(dataloader, optimizer)

			scheduler.step(metrics=avg_loss)
			iter_learning_rate = scheduler.state_dict()['_last_lr'][0]
				
			# Store the loss
			losses.append(avg_loss)
			grad_norms.append(avg_norm)

			t_epoch = time.time() - t_start
				
			if epoch % disp_every == 0: 
				print('Epoch %d; loss: %.2e; grad norm: %.2e; learning rate: %.2e; time: %.2es' %
					(epoch, losses[-1], grad_norms[-1], iter_learning_rate, t_epoch))

			if avg_loss < min_loss:
				break

			epoch += 1

		valid_pred = self._forward_np(valid_dataset[0:1])
		fig = FlipFlopData.plot_trials(valid_data, valid_pred, fig=fig)

		return losses, grad_norms

	def _train_epoch(self, dataloader, optimizer, verbose=False):

		n_trials = len(dataloader)
		avg_loss = 0; 
		avg_norm = 0

		for batch_idx, batch_data in enumerate(dataloader):

			step_summary = self._train_step(batch_data, optimizer)
			
			# Add to the running loss average
			avg_loss += step_summary['loss']/n_trials
			
			# Add to the running gradient norm average
			avg_norm += step_summary['grad_norm']/n_trials

			if verbose:
				print('\tStep %d; loss: %.2e; grad norm: %.2e; time: %.2es' %
					(batch_idx, 
					step_summary['loss'], 
					step_summary['grad_norm'], 
					step_summary['time']))

		return avg_loss, avg_norm

	def _train_step(self, batch_data, optimizer):
		'''
		Returns:

		'''

		t_start = time.time()


		# Run the model and compute loss
		batch_pred = self.forward(batch_data)
		loss = self._loss(batch_data, batch_pred)
		
		# Run the backward pass and gradient descent step
		optimizer.zero_grad()
		loss.backward()
		# nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
		optimizer.step()

		grad_norms = [p.grad.norm().cpu() for p in self.parameters()]

		loss_np = loss.item()
		grad_norm_np = np.mean(grad_norms)

		t_step = time.time() - t_start

		summary = {
			'loss': loss_np,
			'grad_norm': grad_norm_np,
			'time': t_step
		}

		return summary

	@classmethod
	def _get_device(cls, verbose=False):
		"""
		Set the device. CUDA if available, else MPS if available (Apple Silicon), CPU otherwise.

		Args:
			None.

		Returns:
			Device string ("cuda", "mps" or "cpu").
		"""
		if torch.backends.cuda.is_built() and torch.cuda.is_available():
			device = "cuda"
			if verbose: 
				print("CUDA GPU enabled.")
		else:
			device = "cpu"
			if verbose:
				print("No GPU found. Running on CPU.")

		# I'm overriding here because of performance and correctness issues with 
		# Apple Silicon MPS: https://github.com/pytorch/pytorch/issues/94691
		#
		# elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
		# 	device = "mps"
		# 	if verbose:
		# 		print("Apple Silicon GPU enabled.")

		return device
