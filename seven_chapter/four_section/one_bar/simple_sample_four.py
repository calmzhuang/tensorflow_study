import tensorflow as tf


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'feat1': tf.FixedLenFeature([], tf.int64),
            'feat2': tf.FixedLenFeature([], tf.int64),
        }
    )
    return features['feat1'], features['feat2']


input_files = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parser)

iterator = dataset.make_one_shot_iterator()
feat1, feat2 = iterator.get_next()


with tf.Session() as sess:
    sess.run(iterator.initializer,
             feed_dict={
                 ["../../../path/to/input_file1",
                  "../../../path/to/input_file2"]
             })
    while True:
        try:
            sess.run([feat1, feat2])
        except tf.errors.OutOfRangeError:
            break