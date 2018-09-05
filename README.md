# FixedPointFinder - Finds fixed points of a recurrent neural network

This code finds and analyzes the fixed points of recurrent neural networks that have been built using Tensorflow. The approach follows that outlined in Sussillo and Barak, "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks", Neural Computation 2013 (https://doi.org/10.1162/NECO_a_00409).


## Prerequisites

The code is written in Python 2.7.6. You will also need:

* **TensorFlow** version 1.10 ([install](https://www.tensorflow.org/install/))
* **NumPy, SciPy, Matplotlib** ([install SciPy stack](https://www.scipy.org/install.html), contains all of them)
* **Scikit-learn** ([install](http://scikit-learn.org/))
* **RecurrentWhisperer** ([install](https://github.com/mattgolub/recurrent-whisperer/))

## Getting started

Check out ./example/run_FlipFlop.py. This is an end-to-end example that 1) trains an RNN to solve the flip-flop task, 2) finds the fixed points of the trained RNN, and 3) visualizes and analyzes the fixed points identified.
