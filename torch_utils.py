'''
torch_utils.py
Written for Python 3.8.17 and Pytorch 2.0.1
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb
import sys
import numpy as np
import torch

def get_device(verbose=False):
	"""
	Set the device. CUDA if available, else MPS if available (Apple Silicon), CPU otherwise

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