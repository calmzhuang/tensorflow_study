import tensorflow as tf

input_files = ["../../../path/to/input_file1", "../../../path/to/input_file2"]
dataset = tf.data.TextLineDataset(input_files)

iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()
with tf.Session() as sess:
    for i in range(3):
        print(sess.run(x))