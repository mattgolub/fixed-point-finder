'''
run_flipflop_torch.py
Written for Python 3.6.9 and Pytorch (version?)
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# PATH_TO_FIXED_POINT_FINDER = '../'
# sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
# from FlipFlop import FlipFlop
# from FixedPointFinder import FixedPointFinder
# from FixedPoints import FixedPoints
# from plot_utils import plot_fps

from FlipFlopData import FlipFlopData
from torch_utils import get_device

class FlipFlopDataset(Dataset):
	def __init__(self, inputs_bxtxd, targets_bxtxd, device='cpu'):
		super().__init__()

		self.device = device
		self.inputs_bxtxd = inputs_bxtxd.astype(np.float32)
		self.targets_bxtxd = targets_bxtxd.astype(np.float32)
		
	def __len__(self):
		return self.inputs_bxtxd.shape[0]
	
	def __getitem__(self, idx):
		inputs_bxtxd = torch.tensor(self.inputs_bxtxd[idx], device=self.device)
		targets_bxtxd = torch.tensor(self.targets_bxtxd[idx], device=self.device)
		return inputs_bxtxd, targets_bxtxd

class RNN(nn.Module):
	def __init__(self, in_size, hidden_size, out_size, nonlinearity='tanh', device='cpu'):
		super().__init__()
		
		self.rnn = nn.RNN(in_size, hidden_size, 
			batch_first=True, 
			nonlinearity=nonlinearity,
			device=device)

		# self.rnn = nn.GRU(in_size, hidden_size, 
		# 	batch_first=True, 
		# 	device=device)

		self.decoder = nn.Linear(hidden_size, out_size, device=device)
		
	def forward(self, x):
		hiddens_bxtxd, _ = self.rnn(x)        
		outputs_bxtxd = self.decoder(hiddens_bxtxd)
		return outputs_bxtxd, hiddens_bxtxd

def train(model, dataloader, optimizer, loss_fn, num_epochs=1, disp_every=100, max_norm=1.):
	losses = []; grad_norms = []
	
	for epoch in range(num_epochs):
		avg_loss = 0; avg_norm = 0
		for batch_idx, batch in enumerate(dataloader):
			
			# Run the model and compute loss
			inputs, targets = batch

			outputs, hiddens = model(inputs)
			loss = loss_fn(outputs, targets)
			
			# Run the backward pass and gradient descent step
			optimizer.zero_grad()
			loss.backward()
			# nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
			optimizer.step()
			
			# Add to the running loss average
			avg_loss += loss/len(dataloader)
			
			# Add to the running gradient norm average
			grad_norms = [p.grad.norm().cpu() for p in model.parameters()]
			grad_norm = np.mean(grad_norms)
			avg_norm += grad_norm/len(dataloader)
			
		# Store the loss
		losses.append(avg_loss.item())
		grad_norms.append(avg_norm.item())
			
		# Compute the average epoch loss
		if epoch % disp_every == 0: 
			print(f'Epoch: {epoch}, Average Loss: {losses[-1]}, Average Gradient Norm: {grad_norms[-1]}')
	
	return losses, grad_norms

def main():

	parser = argparse.ArgumentParser(
		description='FixedPointFinder: Flip Flop example')
	parser.add_argument('--train_mode', default=1, type=int)
	args = vars(parser.parse_args())
	train_mode = args['train_mode']

	in_size = 3
	out_size = 3
	hidden_size = 32
	lr = 1e-2

	n_train = 512
	n_valid = 128
	batch_size = 128

	device = get_device()

	# Step 0: Generate data
	data_gen = FlipFlopData()
	train_data = data_gen.generate_data(n_trials=n_train)
	train_dataset = FlipFlopDataset(train_data['inputs'], train_data['targets'], device=device)
	
	valid_data = data_gen.generate_data(n_trials=n_valid)
	valid_dataset = FlipFlopDataset(valid_data['inputs'], train_data['targets'], device=device)

	dataloader = DataLoader(train_dataset, batch_size=batch_size)

	# Step 1: Train an RNN to solve the N-bit memory task
	# Model and training settings

	# Create the model
	model = RNN(in_size, hidden_size, out_size, device=device)

	# Create the optimizer
	optimizer = optim.Adam(model.parameters(), lr=lr)

	# Create the loss function
	loss_fn = nn.MSELoss()

	with torch.no_grad():
		valid_inputs, valid_targets = valid_dataset[:n_valid]
		valid_pred, valid_hidden = model(valid_inputs)
		pred = {'output': valid_pred.cpu()}
		fig = FlipFlopData.plot_trials(valid_data, pred)

	# Train the model
	for i in range(30):
		num_epochs = 10
		losses, grad_norms = train(model, dataloader, optimizer, loss_fn, num_epochs=num_epochs, disp_every=10)
		
		with torch.no_grad():
			valid_inputs, valid_targets = valid_dataset[:n_valid]
			valid_pred, valid_hidden = model(valid_inputs)
			pred = {'output': valid_pred.cpu()}
			FlipFlopData.plot_trials(valid_data, pred, fig=fig)


	# model, valid_predictions = train_FlipFlop(train_mode)

	# STEP 2: Find, analyze, and visualize the fixed points of the trained RNN
	# find_fixed_points(model, valid_predictions)

if __name__ == '__main__':
	main()