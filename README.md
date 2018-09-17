# FixedPointFinder - A Tensorflow toolbox for finding fixed points and linearized dynamics in recurrent neural networks

This code finds and analyzes the fixed points of recurrent neural networks that have been built using Tensorflow. The approach follows that outlined in Sussillo and Barak, "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks", Neural Computation 2013 ([link](https://doi.org/10.1162/NECO_a_00409)).


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
2. To run the example, you must also clone or download [```RecurrentWhisperer```](https://github.com/mattgolub/recurrent-whisperer/) into the same directory (do not place `recurrent-whisperer/` inside `fixed-point-finder/`; rather, both directories should share the same parent directory).
3. Install the prerequisite packages listed above.

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

## Example

``FixedPointFinder`` includes an end-to-end example that trains a Tensorflow RNN to solve a task and then identifies and visualizes the fixed points of the trained RNN. To run the example, descend into the example directory: `fixed-point-finder/example/` and execute:

```bash
>>> python run_FlipFlop.py
```

The task is the "flip-flop" task previously described in @SussilloBarak2013. Briefly, the task is to implement a 3-bit binary memory, in which each of 3 input channels delivers signed transient pulses (-1 or +1) to a corresponding bit of the memory, and an input pulse flips the state of that memory bit (also -1 or +1) whenever a pulse's sign is opposite of the current state of the bit. The example trains a 16-unit LSTM RNN to solve this task (Fig. 1). Once the RNN is trained, the example uses ``FixedPointFinder`` to identify and characterize the trained RNN's fixed points. Finally, the example produces a visualization of these results (Fig. 2). In addition to demonstrating a working use of ``FixedPointFinder``, this example provides a testbed for experimenting with different RNN architectures (e.g., numbers of recurrent units, LSTMs vs. GRUs vs. vanilla RNNs) and characterizing how these lower-level model design choices manifest in the higher-level dynamical implementation used to solve a task.

---
![Figure 1](paper/task_example.png)

##### Figure 1. Inputs (gray), target outputs (cyan), and outputs of a trained LSTM RNN (purple) from an example trial of the flip-flop task. Signed input pulses (gray) flip the corresponding bit's state (green) whenever an input pulse has the opposite sign of the current bit state (e.g., if gray goes high when green is low). The RNN has been trained to nearly perfectly reproduce the target memory state (purple closely overlaps cyan).
---
![Figure 2](paper/fixed_points.png)

##### Figure 2. Fixed-point structure of an LSTM RNN trained to solve the flip-flop task. ``FixedPointFinder`` identified 8 stable fixed points (black points), each of which corresponds to a unique state of the 3-bit memory. ``FixedPointFinder`` also identified a number of unstable fixed points (red points) along with their unstable modes (red lines), which mediate the set of state transitions trained into the RNN's dynamics. Here, each unstable fixed point is a "saddle" in the RNN's dynamical flow field, and the corresponding unstable modes indicate the directions that nearby states are repelled from the fixed point. State trajectories from example trials (blue) traverse about these fixed points. All quantities are visualized in the 3-dimensional space determined by the top 3 principal components computed across 128 example trials.
