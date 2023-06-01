'''
torch_utils.py
Written for Python 3.6.9 and Pytorch (version?)
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb
import sys
import numpy as np
import torch

def get_device():
	"""
	Set the device. CUDA if available, else MPS if available (Apple Silicon), CPU otherwise

	Args:
		None.

	Returns:
		Device string ("cuda", "mps" or "cpu").
	"""
	if torch.backends.cuda.is_built() and torch.cuda.is_available():
		device = "cuda"
		print("CUDA GPU enabled.")
	elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
		device = "mps"
		print("Apple Silicon GPU enabled.")
	else:
		device = "cpu"
		print("No GPU found. Running on CPU.")

	return device