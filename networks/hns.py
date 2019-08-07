import tensorflow as tf
from utils.custom_layers import *


def hide_and_seek_small(input_shape, num_classes, binary_type='deterministic', stochastic_estimator='reinforce',
                        slope_increase_rate=0.000001):

    # Input layer
    inp = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    en0 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(inp)
    en1 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(en0)
    flat = tf.keras.layers.Flatten()(en1)
    encoded = tf.keras.layers.Dense(input_shape[0] * 5)(flat)

    # Decoder
    de0 = tf.keras.layers.Dense((input_shape[0] // 4)**2 * 16)(encoded)
    res = tf.keras.layers.Reshape((input_shape[0] // 4, input_shape[0] // 4, 16))(de0)
    de1 = tf.keras.layers.Conv2DTranspose(8, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(res)
    de2 = tf.keras.layers.Conv2DTranspose(4, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(de1)
    decoded = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(de2)

    # Binary layer
    if binary_type == 'deterministic':
        binary = BinaryDeterministic(name='hider_output')(decoded)
    elif binary_type == 'stochastic':
        binary = BinaryStochastic(estimator=stochastic_estimator, name='hider_output',
                                  slope_increase_rate=slope_increase_rate)(decoded)
    else:
        raise ValueError("'binary_type' can either be 'stochastic' or 'deterministic'")

    # Connection layer
    masked_img = inp * binary

    # CNN
    c1 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(masked_img)
    c2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c1)
    c3 = tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c2)
    flat = tf.keras.layers.Flatten()(c3)
    fc = tf.keras.layers.Dense(num_classes, activation='softmax', name='seeker_output')(flat)

    # Model
    model = tf.keras.models.Model(inputs=[inp], outputs={'seeker_output': fc,
                                                         'hider_output': binary})

    model.compile(optimizer='adam',
                  loss={'seeker_output': 'categorical_crossentropy', 'hider_output': 'binary_crossentropy'},
                  loss_weights={'seeker_output': 1., 'hider_output': 0},
                  metrics={'seeker_output': ['accuracy']})

    return model


def hide_and_seek_large(input_shape, num_classes, binary_type='deterministic', stochastic_estimator='reinforce',
                        slope_increase_rate=0.000001):

    # Input layer
    inp = tf.keras.layers.Input(shape=input_shape)

    # encoder
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

    # Binary layer
    if binary_type == 'deterministic':
        binary = BinaryDeterministic(name='hider_output')(decoded)
    elif binary_type == 'stochastic':
        binary = BinaryStochastic(estimator=stochastic_estimator, name='hider_output',
                                  slope_increase_rate=slope_increase_rate)(decoded)
    else:
        raise ValueError("'binary_type' can either be 'stochastic' or 'deterministic'")

    # Connection layer
    masked_img = inp * binary

    # CNN
    c1 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(masked_img)
    c2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c1)
    c3 = tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c2)
    c4 = tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c3)
    c5 = tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c4)

    # FC
    flat = tf.keras.layers.Flatten()(c5)
    fc = tf.keras.layers.Dense(num_classes, activation='softmax', name='seeker_output')(flat)

    # Model
    model = tf.keras.models.Model(inputs=[inp], outputs={'seeker_output': fc,
                                                         'hider_output': bd})

    model.compile(optimizer='adam',
                  loss={'seeker_output': 'categorical_crossentropy', 'hider_output': 'binary_crossentropy'},
                  loss_weights={'seeker_output': 1., 'hider_output': 0},
                  metrics={'seeker_output': ['accuracy']})

    return model


def hide_and_seek_resnet(input_shape, num_classes, binary_type='deterministic', stochastic_estimator='reinforce',
                         slope_increase_rate=0.000001):

    # Input layer
    inp = tf.keras.layers.Input(shape=input_shape)

    # encoder
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

    # Binary layer
    if binary_type == 'deterministic':
        binary = BinaryDeterministic(name='hider_output')(decoded)
    elif binary_type == 'stochastic':
        binary = BinaryStochastic(estimator=stochastic_estimator, name='hider_output',
                                  slope_increase_rate=slope_increase_rate)(decoded)
    else:
        raise ValueError("'binary_type' can either be 'stochastic' or 'deterministic'")

    # Connection layer
    masked_img = inp * binary

    res = tf.keras.applications.resnet50.ResNet50(input_shape=input_shape)(masked_img)
    out = tf.keras.layers.Dense(num_classes, activation='softmax', name='seeker_output')(res.output[-2])

    model = tf.keras.models.Model(inputs=[inp], outputs={'seeker_output': out,
                                                         'hider_output': binary})

    model.compile(optimizer='adam',
                  loss={'seeker_output': 'categorical_crossentropy', 'hider_output': 'binary_crossentropy'},
                  loss_weights={'seeker_output': 1., 'hider_output': 0},
                  metrics={'seeker_output': ['accuracy']})

    return model


available_models = {'hns_small': hide_and_seek_small,
                    'hns_large': hide_and_seek_large,
                    'hns_resnet': hide_and_seek_resnet}


if __name__ == '__main__':

    # for mnist
    # input_shape = (28, 28, 1)
    # num_classes = 10
    # model = hide_and_seek_small(input_shape, num_classes)

    # for animals
    input_shape = (192, 192, 3)
    num_classes = 398
    model = hide_and_seek_large(input_shape, num_classes)
