import tensorflow as tf


def seeker(input_shape=(28, 28, 1), num_classes=10):

    inp = tf.keras.layers.Input(shape=input_shape)
    c1 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(inp)
    c2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c1)
    c3 = tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c2)
    flat = tf.keras.layers.Flatten()(c3)
    fc = tf.keras.layers.Dense(num_classes, activation='softmax')(flat)

    model = tf.keras.models.Model(inputs=[inp], outputs=[fc])

    return model


def seeker_v2(input_shape=(28, 28, 1), num_classes=10):

    inp = tf.keras.layers.Input(shape=input_shape)
    c1 = tf.keras.layers.Conv2D(32, kernel_size=(6, 6), strides=(1, 1), activation='relu', padding='same')(inp)
    m1 = tf.keras.layers.MaxPool2D((2, 2), strides=(1, 1))(c1)
    c2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(m1)
    m2 = tf.keras.layers.MaxPool2D((2, 2), strides=(1, 1))(c2)
    flat = tf.keras.layers.Flatten()(m2)
    fc1 = tf.keras.layers.Dense(1024, activation='relu')(flat)
    fc2 = tf.keras.layers.Dense(num_classes, activation='softmax')(fc1)

    model = tf.keras.models.Model(inputs=[inp], outputs=[fc2])

    return model


def seeker_large(input_shape, num_classes):
    inp = tf.keras.layers.Input(input_shape)
    c1 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(inp)
    c2 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c1)
    c3 = tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c2)
    c4 = tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c3)
    c5 = tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same')(c4)
    # FC
    flat = tf.keras.layers.Flatten()(c5)
    fc = tf.keras.layers.Dense(num_classes, activation='softmax', name='seeker_output')(flat)
    # Model
    model = tf.keras.models.Model(inputs=[inp], outputs=[fc])
    return model


def seeker_resnet(input_shape, num_classes):
    res = tf.keras.applications.resnet50.ResNet50(input_shape=input_shape, include_top=False)
    flat = tf.keras.layers.Flatten()(res.output)
    out = tf.keras.layers.Dense(num_classes, activation='softmax', name='seeker_output')(flat)
    model = tf.keras.models.Model(inputs=[res.input], outputs=[out])
    return model


available_models = {'hns_small': seeker,
                    'hns_large': seeker_large,
                    'hns_resnet': seeker_resnet}
