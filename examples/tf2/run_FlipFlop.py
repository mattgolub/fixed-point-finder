'''
In development: Porting the TF stack to true TF2
(from tf.compat.v1 band-aid solution)
'''

import sys
import pdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PATH_TO_HELPER = '../helper/'
sys.path.insert(0, PATH_TO_HELPER)
from FlipFlopData import FlipFlopData

# Data specifications
n_bits = 3
n_time = 64
n_batch = 32
n_train = 512
n_valid = 128

data_gen = FlipFlopData(n_time=n_time, n_bits=n_bits)
train_data = data_gen.generate_data(n_trials=n_train)
valid_data = data_gen.generate_data(n_trials=n_valid)

model = keras.Sequential()

# Add a RNN layer with 16 internal units.
model.add(layers.SimpleRNN(16, return_sequences=True))

# Or other standard RNN layers:
# model.add(layers.LSTM(16, return_sequences=True))
# model.add(layers.GRU(16, return_sequences=True))

model.add(layers.Dense(n_bits))
model.compile(optimizer='adam', loss='mse')
model.build(input_shape=(n_batch, n_time, n_bits))
model.summary()

history = model.fit(
	x=train_data['inputs'], 
	y=train_data['targets'],
	validation_data=(valid_data['inputs'], valid_data['targets']),
	epochs=200, 
	batch_size=n_batch)

valid_pred = {'output': model.predict(valid_data['inputs'])}

FlipFlopData.plot_trials(valid_data, valid_pred, n_trials_plot=4)

model.layers[0]

pdb.set_trace()