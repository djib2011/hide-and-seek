import tensorflow as tf
from utils.custom_layers import BinaryDeterministic


def hider_small(input_shape=(28, 28, 1)):
    # encoder
    inp = tf.keras.layers.Input(shape=input_shape)
    en0 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(inp)
    en1 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(en0)
    flat = tf.keras.layers.Flatten()(en1)
    encoded = tf.keras.layers.Dense(input_shape[0] * 5)(flat)
    # decoder
    de0 = tf.keras.layers.Dense((input_shape[0] // 4)**2 * 16)(encoded)
    res = tf.keras.layers.Reshape((input_shape[0] // 4, input_shape[0] // 4, 16))(de0)
    de1 = tf.keras.layers.Conv2DTranspose(8, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(res)
    de2 = tf.keras.layers.Conv2DTranspose(4, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(de1)
    decoded = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(de2)
    model = tf.keras.models.Model(inputs=[inp], outputs=[decoded])
    return model


def autoencoder_small_old(input_shape=(28, 28, 1)):

    # encoder
    inp = tf.keras.layers.Input(shape=input_shape)
    en0 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(inp)
    en1 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(en0)
    flat = tf.keras.layers.Flatten()(en1)
    encoded = tf.keras.layers.Dense(128)(flat)

    # decoder
    de0 = tf.keras.layers.Dense(784)(encoded)
    res = tf.keras.layers.Reshape((7, 7, 16))(de0)
    de1 = tf.keras.layers.Conv2DTranspose(8, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(res)
    de2 = tf.keras.layers.Conv2DTranspose(4, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(de1)
    decoded = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(de2)

    model = tf.keras.models.Model(inputs=[inp], outputs=[decoded])

    return model


def hider_large(input_shape=(28, 28, 1)):
    # encoder
    inp = tf.keras.layers.Input(shape=input_shape)
    en0 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(inp)
    en1 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(en0)
    en2 = tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(en1)
    en3 = tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(en2)
    en4 = tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(en3)
    flat = tf.keras.layers.Flatten()(en4)
    encoded = tf.keras.layers.Dense(input_shape[0] * 5)(flat)
    # decoder
    de0 = tf.keras.layers.Dense((input_shape[0] // 32)**2 * 64)(encoded)
    res = tf.keras.layers.Reshape((input_shape[0] // 32, input_shape[0] // 32, 64))(de0)
    de1 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(res)
    de2 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(de1)
    de3 = tf.keras.layers.Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(de2)
    de4 = tf.keras.layers.Conv2DTranspose(8, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(de3)
    de5 = tf.keras.layers.Conv2DTranspose(4, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(de4)
    decoded = tf.keras.layers.Conv2D(input_shape[-1], (1, 1), activation='sigmoid')(de5)
    model = tf.keras.models.Model(inputs=[inp], outputs=[decoded])
    return model


def mask_model(input_shape):

    ae = hider_small(input_shape)
    bd = BinaryDeterministic()(ae.output)

    model = tf.keras.models.Model(inputs=[ae.input], outputs=[bd])

    return model


available_models = {'hns_small': hider_small,
                    'hns_large': hider_large,
                    'hns_resnet': hider_large}
