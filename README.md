# FixedPointFinder - A Tensorflow toolbox for finding fixed points and linearized dynamics in recurrent neural networks.

This code finds and analyzes the fixed points of recurrent neural networks that have been built using Tensorflow. The approach follows that outlined in Sussillo and Barak, "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks", Neural Computation 2013 (https://doi.org/10.1162/NECO_a_00409).


## Prerequisites

The code is written in Python 2.7.6. You will also need:

* **TensorFlow** version 1.10 ([install](https://www.tensorflow.org/install/)).
* **NumPy, SciPy, Matplotlib** ([install SciPy stack](https://www.scipy.org/install.html), contains all of them).
* **Scikit-learn** ([install](http://scikit-learn.org/)).

To run the included example, you will additionally need:
* **RecurrentWhisperer** ([install](https://github.com/mattgolub/recurrent-whisperer/)); see 'Installation' below.
* **PyYaml** ([install](https://pyyaml.org/)).

## Installation

1. [Clone or download](https://help.github.com/articles/cloning-a-repository/) this repository into a directory of your choice.
2. To run the example, you must also clone or download [RecurrentWhisperer](https://github.com/mattgolub/recurrent-whisperer/) into the same directory (do not place recurrent-whisperer/ inside fixed-point-finder/; rather, both directories should share the same parent directory).
3. Install the prerequisite packages listed above.

## End-to-end example

Included is an end-to-end example that 1) trains an LSTM RNN to implement a 3-bit memory system (aka the 'flip-flop' task), 2) finds the fixed points of the trained RNN, and 3) visualizes and analyzes the fixed points identified. Running the example requires only a few minutes on a modern machine.

To run the example, descend into the example directory: `fixed-point-finder/example/` and execute:

```bash
>>> python run_FlipFlop.py
```

## General Usage

1. Start by building, and if desired, training an RNN. ```FixedPointFinder``` works with any arbitrary RNN that conforms to Tensorflow's `RNNCell` API.
2. Build a ```FixedPointFinder``` object:
  ```python
  >>> fpf = FixedPointFinder(your_rnn_cell, tf_session, **hyperparams)
  ```
  using `your_rnn_cell`, the `RNNCell` that specifies the single-timestep transitions in your RNN, `tf_session`, the Tensorflow session in which your model has been instantiated, and `hyperparams`, a python dict of optional hyperparameters for the fixed-point optimizations.
3. Specify the `initial_states` from which you'd like to initialize the local optimizations implemented by ```FixedPointFinder```. These states should conform to type expected by `your_rnn_cell` (e.g., `LSTMStateTuple` if `your_rnn_cell` is an `LSTMCell`).
4. Specify the `inputs` under which you'd like to study your RNN. Currently, ```FixedPointFinder``` only supports static inputs. Thus `inputs` should be a numpy array with shape `(1, n_inputs)` where `n_inputs` is an int specifying the depth of the inputs expected by `your_rnn_cell`.
5. Run the local optimizations that find the fixed points:
```python
>>> fp_dict = fpf.find_fixed_points(initial_states, inputs)
```
The fixed points identified, along with the Jacobian of your RNN state transition function at those points, will be stored within the ```FixedPointFinder``` object, and are additionally optionally returned in the python dict `fp_dict`.
6. Finally, visualize the identified fixed points:
```python
>>> fpf.plot_summary()
```
You can also visualize these fixed points amongst state trajectories from your RNN (see the docstring for `FixedPointFinder.plot_summary()` and the example in `fixed-point-finder/example/run_FlipFlop.py`).
