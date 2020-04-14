from pathlib import Path
import tensorflow as tf
import numpy as np


def image_generator(data_dir, batch_size=128, image_shape=(192, 192), channels=3):
    """
    Generator that loads images and processes images from the disk. Requires the data to be structured as in the keras
    ImageDataGenerator (ImageNet format), i.e. data_dir should contain subdirectories of the classes, each containing
    the images belonging to its respective class. Also resizes the images to the desired dimension.
    Currently, only works for JPEG images.
    :param data_dir: Directory containing class subdirectories with the images.
    :param batch_size: Desired batch size.
    :param image_shape: A tuple containing the resolution the images will be resized to.
    :param channels: Number of channels we want to load from the images (3 for RGB, 1 for grayscale).
    :return: a tf.data.Dataset that loads and processes the images in data_dir.
    """

    def preprocess_image(image, channels=channels, image_shape=image_shape):
        """
        Function that opens a JPEG image, resizes it and rescales it to [0, 1]
        :param image: A path for a JPEG image.
        :param channels: The number of channels in the image (3 for RGB, 1 for grayscale)
        :param image_shape: The desired resolution that we're going to resize the image to
        :return: a np.array containing the processed image
        """
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.image.resize(image, image_shape)
        image /= 255.0
        return image

    def load_and_preprocess_image(image_path, image_label, channels=channels, image_shape=image_shape):
        """
        Function that loads and processes a JPEG image. Label is not modified at all.
        :param image: A path for a JPEG image.
        :param channels: The number of channels in the image (3 for RGB, 1 for grayscale)
        :param image_shape: The desired resolution that we're going to resize the image to
        :return: The loaded and processed image in a np.array and the label
        """
        image = tf.io.read_file(image_path)
        return preprocess_image(image, channels=channels, image_shape=image_shape), image_label

    # List of classes
    class_names = sorted([x.name for x in Path(data_dir).glob('*') if x.is_dir()])

    # List of paths to images
    images = [str(x) for x in Path(data_dir).rglob('*.JPEG')]

    # One-hot encoded labels
    classes = [class_names.index(x.split('/')[-2]) for x in images]
    labels = tf.keras.utils.to_categorical(classes, len(class_names))

    # Tensorflow dataset
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.shuffle(buffer_size=len(images))
    data = data.map(load_and_preprocess_image)
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    return data


def mnist(batch_size=64, split='train'):
    """
    Generator that loads the mnist dataset.
    :param batch_size: The desired batch size.
    :param split: Which set to load, "train" or "test".
    :return: a tf.data.Dataset that generates mnist images
    """

    # Load train/test set images
    if split == 'train':
        (images, labels), (_, _) = tf.keras.datasets.mnist.load_data()
        images = images.reshape((60000, 28, 28, 1))
    elif split == 'test':
        (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
        images = images.reshape((10000, 28, 28, 1))
    else:
        raise ValueError('Invalid value for argument "set". Should be either "train" or "test".')

    # Normalize pixel values to be between 0 and 1
    images = images.astype(np.float32) / 255.0

    # One-hot encode the labels
    labels = tf.keras.utils.to_categorical(labels, 10)

    # Create tf.Dataset
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.shuffle(buffer_size=len(images))
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    return data


def fashion(batch_size=64, split='train'):
    """
    Generator that loads the fashion mnist dataset.
    :param batch_size: The desired batch size.
    :param split: Which set to load, "train" or "test".
    :return: a tf.data.Dataset that generates mnist images
    """

    # Load train/test set images
    if split == 'train':
        (images, labels), (_, _) = tf.keras.datasets.mnist.load_data()
        images = images.reshape((60000, 28, 28, 1))
    elif split == 'test':
        (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
        images = images.reshape((10000, 28, 28, 1))
    else:
        raise ValueError('Invalid value for argument "set". Should be either "train" or "test".')

    # Normalize pixel values to be between 0 and 1
    images = images.astype(np.float32) / 255.0

    # One-hot encode the labels
    labels = tf.keras.utils.to_categorical(labels, 10)

    # Create tf.Dataset
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.shuffle(buffer_size=len(images))
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    return data


def cifar10(batch_size=64, split='train', channels=3):
    """
    Generator that loads the cifar10 dataset.
    :param batch_size: The desired batch size.
    :param split: Which set to load, "train" or "test".
    :param channels: How many channels do we want the image to have (3 for RGB, 1 for grayscale)
    :return: a tf.data.Dataset that generates cifar10 images
    """

    # Load train/test set images
    if split == 'train':
        (images, labels), (_, _) = tf.keras.datasets.cifar10.load_data()
    elif split == 'test':
        (_, _), (images, labels) = tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError('Invalid value for argument "set". Should be either "train" or "test".')

    if channels == 1:
        images = np.expand_dims(images.mean(axis=-1), axis=-1)

    # Normalize pixel values to be between 0 and 1
    images = images.astype(np.float32) / 255.

    # One-hot encode the labels
    labels = tf.keras.utils.to_categorical(labels, 10)

    # Create tf.Dataset
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.shuffle(buffer_size=len(images))
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    return data


def cifar100(batch_size=64, split='train', channels=3):
    """
    Generator that loads the cifar100 dataset.
    :param batch_size: The desired batch size.
    :param split: Which set to load, "train" or "test".
    :param channels: How many channels do we want the image to have (3 for RGB, 1 for grayscale)
    :return: a tf.data.Dataset that generates cifar10 images
    """

    # Load train/test set images
    if split == 'train':
        (images, labels), (_, _) = tf.keras.datasets.cifar100.load_data()
    elif split == 'test':
        (_, _), (images, labels) = tf.keras.datasets.cifar100.load_data()
    else:
        raise ValueError('Invalid value for argument "set". Should be either "train" or "test".')

    if channels == 1:
        images = np.expand_dims(images.mean(axis=-1), axis=-1)

    # Normalize pixel values to be between 0 and 1
    images = images.astype(np.float32) / 255.

    # One-hot encode the labels
    labels = tf.keras.utils.to_categorical(labels, 100)

    # Create tf.Dataset
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.shuffle(buffer_size=len(images))
    data = data.repeat()
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=1)

    return data
