import tensorflow as tf
import numpy as np
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
import matplotlib.pyplot as plt
from pathlib import Path
import os

import utils
from utils.options import config
import networks


class HiderTrainer:

    def __init__(self, model, weight_dir='weights', log_dir='logs', optimizer=None, loss_function=None, debug=False):

        self.model = model

        if not debug:
            self.weight_dir = weight_dir
            if not Path(weight_dir).is_dir():
                os.makedirs(weight_dir)

            self.log_dir = log_dir
            if not Path(log_dir).is_dir():
                os.makedirs(log_dir)
            self.train_summary_writer = tf.summary.create_file_writer(log_dir + '/batch')
            self.epoch_summary_writer = tf.summary.create_file_writer(log_dir + '/epoch')
        self.debug = debug

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = adam_v2.Adam()

        if loss_function:
            self.loss_function = loss_function
        else:
            self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.iteration = 0
        self.epoch_loss = []
        self.total_train_loss = []
        self.validation_loss = []

    def loss(self, x):

        y_ = self.model(x)

        reconstruction_loss = self.loss_function(y_true=x, y_pred=y_)

        if not self.debug:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('reconstruction loss', reconstruction_loss, step=self.iteration)

        return reconstruction_loss

    def train_batch(self, x):

        with tf.GradientTape() as tape:
            loss_value = self.loss(x)

        grads = tape.gradient(loss_value, self.model.trainable_variables)

        total_grad = np.abs(np.concatenate([g.numpy().flatten() for g in grads], axis=0)).sum()

        if not self.debug:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('gradients', total_grad, step=self.iteration)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss_value

    def train(self, train_data, training_steps, max_epochs=10, test_data=None, validation_steps=None):

        for epoch in range(max_epochs):

            epoch_loss_avg = tf.keras.metrics.Mean()

            for i, (x, _) in enumerate(train_data):

                self.iteration += 1

                batch_loss = self.train_batch(x)

                epoch_loss_avg(batch_loss)
                self.total_train_loss.append(batch_loss)

                if i == training_steps:
                    break

            avg = epoch_loss_avg.result()

            if not self.debug:
                with self.epoch_summary_writer.as_default():
                    tf.summary.scalar('Average loss per epoch', avg, step=epoch+1)
            self.epoch_loss.append(avg)

            print('Epoch {}: Training Loss: {:.3f}'.format(epoch+1, avg))

            if test_data:
                val_loss = self.evaluate(test_data, validation_steps)
                if not self.debug:
                    with self.epoch_summary_writer.as_default():
                        tf.summary.scalar('Validation loss', val_loss, step=epoch+1)
                self.validation_loss.append(val_loss)
                print('Validation loss: {:.3f}%'.format(val_loss))

    def evaluate(self, data, steps):

        mean_loss = tf.keras.metrics.Mean()

        for i, (x, _) in enumerate(data):

            y_ = self.model(x)

            reconstruction_loss = self.loss_function(y_true=x, y_pred=y_)

            mean_loss(reconstruction_loss)

            if i == steps:
                break

        return mean_loss.result().numpy()

    def save_sample_images(self, x, directory='/tmp/sample_images'):

        y_ = self.model(x)

        if not Path(directory).is_dir():
            os.makedirs(directory)

        for i in range(len(x)):
            plt.imsave(directory + '/real_' + str(i + 1) + '.png', x[i, ..., 0])
            plt.imsave(directory + '/reconstruction_' + str(i + 1) + '.png', y_[i, ..., 0])


if __name__ == '__main__':

    # Experiment identifier
    identifier = config['identifier']

    # Model configurations
    model_id = config['model']
    pretrained_hider_weights = None
    if config['hider_weights']:
        p = Path(config['hider_weights']).absolute()
        if p.exists():
            pretrained_hider_weights = config['hider_weights']
        else:
            print('No valid hider weights found at:', config['hider_weights'])
            print('Will train hider from skratch.')

    # training configurations
    max_epochs = config['max_epochs']
    batch_size = config['batch_size']
    debug = config['debug']

    # Data configurations
    image_shape = (config['image_size'], ) * 2
    train_images = config['train_images']
    test_images = config['test_images']
    channels = config['channels']
    input_shape = image_shape + (channels,)
    num_classes = config['num_classes']

    if config['config'] == 'mnist':
        train_set = utils.datagen.mnist_train(batch_size=batch_size, split='train')
        test_set = utils.datagen.mnist_test(batch_size=batch_size, split='test')
    elif config['config'] == 'cifar10':
        train_set = utils.datagen.cifar10(batch_size=batch_size, split='train', channels=channels)
        test_set = utils.datagen.cifar10(batch_size=batch_size, split='test', channels=channels)
    elif config['config'] == 'cifar100':
        train_set = utils.datagen.cifar100(batch_size=batch_size, split='train', channels=channels)
        test_set = utils.datagen.cifar100(batch_size=batch_size, split='test', channels=channels)
    else:
        data_dir = Path(config['data_dir'])
        train_set = utils.datagen.image_generator(data_dir / 'train',batch_size=batch_size, image_shape=image_shape, channels=channels)
        test_set = utils.datagen.image_generator(data_dir / 'test', batch_size=batch_size, image_shape=image_shape, channels=channels)

    log_dir = str(Path('logs') / config['config'] / 'hider' / identifier)
    weight_dir = str(Path('weights') / config['config'] / 'hider' / identifier)

    print('Initializing Hider')
    hider = networks.hide.available_models[model_id](input_shape)

    if pretrained_hider_weights:
        print('Loading pre-trained weights')
        hider.load_weights(pretrained_hider_weights)

    hider_trainer = HiderTrainer(model=hider, weight_dir=weight_dir, log_dir=log_dir, debug=debug)

    print('Training hider for {} epochs.'.format(max_epochs))
    with utils.training.WeightFailsafe(weight_dir, hider, debug=debug):
        hider_trainer.train(train_set, training_steps=train_images//batch_size,  max_epochs=max_epochs,
                            test_data=test_set, validation_steps=test_images//batch_size)

    print('Test set loss: {:.3f}%'.format(hider_trainer.evaluate(test_set, test_images//batch_size)))

    if not debug:
        print('Saving model to:', weight_dir)
        hider.save_weights(weight_dir + '/final_weights.h5')
