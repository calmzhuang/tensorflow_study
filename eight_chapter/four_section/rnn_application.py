from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')

num_units = 30
num_layers = 2

time_steps = 10
training_steps = 10000
batch_size = 32

training_examples = 10000
testing_examples = 1000
sample_gap = 0.01


def generate_data(seq):
    x = []
    y = []
    for i in range(len(seq) - time_steps):
        x.append([seq[i: i + time_steps]])
        y.append([seq[i: i + time_steps]])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def lstm_model(x, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(num_units) for _ in range(num_layers)]
    )
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    outputs = outputs[:, -1, :]