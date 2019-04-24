import numpy as np
import tensorflow as tf

# import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt

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
        y.append([seq[i + time_steps]])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def lstm_model(x, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(num_units) for _ in range(num_layers)]
    )
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = outputs[:, -1, :]
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    if not is_training:
        return predictions, None, None

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1
    )
    return predictions, loss, train_op


def train(sess, train_x, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    x, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(x, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(training_steps):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print("train step:" + str(i) + ", loss:" + str(l))


def run_eval(sess, test_x, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1)
    x, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(x, [0.0], False)

    predictions = []
    labels = []
    for i in range(testing_examples):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Mean Square Error is : {}".format(rmse))

    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


test_start = (training_examples + time_steps) * sample_gap
test_end = test_start + (testing_examples + time_steps) * sample_gap
train_x, train_y = generate_data(np.sin(np.linspace(
    0, test_start, training_examples + time_steps, dtype=np.float32
)))
test_x, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, testing_examples + time_steps, dtype=np.float32
)))

with tf.Session() as sess:
    train(sess, train_x, train_y)
    run_eval(sess, test_x, test_y)