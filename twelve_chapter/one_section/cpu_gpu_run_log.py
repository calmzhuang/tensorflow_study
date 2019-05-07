import tensorflow as tf

with tf.device('/cpu:0'):
    a = tf.constant([0.1, 2.3, 4.0], shape=[3], name='a')
    b = tf.constant([0.4, 2.6, 4.8], shape=[3], name='b')

with tf.device('/gpu:0'):
    c = a + b

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
