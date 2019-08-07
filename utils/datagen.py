from pathlib import Path
import tensorflow as tf
import numpy as np


def image_generator(data_dir, batch_size=128, image_shape=(192, 192), channels=3):

    def preprocess_image(image, channels=image_shape, image_shape=image_shape):
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.image.resize(image, image_shape)
        image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(image_path, image_label, channels=channels, image_shape=image_shape):
        image = tf.io.read_file(image_path)
        return preprocess_image(image, channels=channels, image_shape=image_shape), image_label

    class_names = sorted([x.name for x in Path(data_dir).glob('*') if x.is_dir()])

    # list of paths to images
    images = [str(x) for x in Path(data_dir).rglob('*.JPEG')]

    # one-hot encoded labels
    classes = [class_names.index(x.split('/')[-2]) for x in images]
    labels = tf.keras.utils.to_categorical(classes, len(class_names))

    # tensorflow dataset
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.shuffle(buffer_size=len(images))
    data = data.map(load_and_preprocess_image)
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    return data


def mnist_train(batch_size=64, multi_input=False):

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1)).astype(np.float32)

    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0

    train_labels = tf.keras.utils.to_categorical(train_labels, len(np.unique(train_labels)))

    if multi_input:
        data = tf.data.Dataset.from_tensor_slices((train_images, train_images, train_labels))
    else:
        data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

    data = data.shuffle(buffer_size=len(train_images))
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    return data


def mnist_test(batch_size=64, multi_input=False):

    (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    test_images = test_images.reshape((10000, 28, 28, 1)).astype(np.float32)

    # Normalize pixel values to be between 0 and 1
    test_images = test_images / 255.0

    test_labels = tf.keras.utils.to_categorical(test_labels, len(np.unique(test_labels)))

    if multi_input:
        data = tf.data.Dataset.from_tensor_slices((test_images, test_images, test_labels))
    else:
        data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    data = data.shuffle(buffer_size=len(test_images))
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    return data


