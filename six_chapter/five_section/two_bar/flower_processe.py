import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-one_section"


input_data = '../../../path/to/flower_photos'
output_file = '../../../path/to/flower_processed_data.npy'

validation_percentage = 10
test_percentage = 10


def create_image_lists(sess, testing_percentage, validationing_percentage):
    sub_dirs = [x[0] for x in os.walk(input_data)]
    is_root_dir = True

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(input_data, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))

        if not file_list:
            continue

        file_list = file_list[0: 400]

        for file_name in file_list:
            image_raw_data = gfile.GFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)

            chance = np.random.randint(100)
            if chance < validationing_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validationing_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
        current_label += 1

    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])


def main():
    with tf.Session() as sess:
        proccessed_data = create_image_lists(
            sess, test_percentage, validation_percentage
        )
        np.save(output_file, proccessed_data)


if __name__ == '__main__':
    main()