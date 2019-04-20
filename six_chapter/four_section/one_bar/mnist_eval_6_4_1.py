import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference_6_4_1
import mnist_train_6_4_1

eval_interval_secs = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference_6_4_1.image_size, mnist_inference_6_4_1.image_size, 1], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, mnist_inference_6_4_1.output_node], name="y-input")
        validate_feed = {x: np.reshape(mnist.validation.images, [-1, mnist_inference_6_4_1.image_size, mnist_inference_6_4_1.image_size, 1]), y_: mnist.validation.labels}

        y = mnist_inference_6_4_1.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train_6_4_1.moving_average_decay)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train_6_4_1.model_save_path)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After {} training step(s), validation accuracy = {}".format(global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(eval_interval_secs)


def main(argv=None):
    mnist = input_data.read_data_sets("../../../tmp/data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()