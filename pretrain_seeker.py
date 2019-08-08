import tensorflow as tf
import numpy as np
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from sklearn.metrics import accuracy_score
from pathlib import Path
import os
import time

import utils
from utils.options import config
import networks


class SeekerTrainer:

    def __init__(self, model, weight_dir='weights', log_dir='logs', optimizer=None, loss_function=None, debug=False):


        self.model = model
        self.debug = debug

        if not debug:
            self.weight_dir = weight_dir
            if not Path(weight_dir).is_dir():
                os.makedirs(weight_dir)

            self.log_dir = log_dir
            if not Path(log_dir).is_dir():
                os.makedirs(log_dir)
            self.train_summary_writer = tf.summary.create_file_writer(log_dir + '/batch')
            self.epoch_summary_writer = tf.summary.create_file_writer(log_dir + '/epoch')

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = adam_v2.Adam()

        if loss_function:
            self.loss_function = loss_function
        else:
            self.loss_function = tf.keras.losses.CategoricalCrossentropy()

        self.iteration = 0
        self.epoch_loss = []
        self.total_train_loss = []
        self.validation_acc = []
        self.current_classification_loss = 0.

    def loss(self, x, y):

        y_ = self.model(x)

        classification_loss = self.loss_function(y_true=y, y_pred=y_)

        if not self.debug:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('training loss', classification_loss, step=self.iteration)

        return classification_loss

    def train_batch(self, x, y):

        with tf.GradientTape() as tape:
            loss_value = self.loss(x, y)

        grads = tape.gradient(loss_value, self.model.trainable_variables)

        total_grad = np.abs(np.concatenate([g.numpy().flatten() for g in grads], axis=0)).sum()

        if not debug:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('gradients', total_grad, step=self.iteration)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss_value

    def train(self, train_data, training_steps, max_epochs=10, test_data=None, validation_steps=None, update_every=6):

        last_update = start_time = time.time()

        for epoch in range(max_epochs):

            epoch_loss_avg = tf.keras.metrics.Mean()

            for i, (x, y) in enumerate(train_data):

                self.iteration += 1

                batch_loss = self.train_batch(x, y)

                epoch_loss_avg(batch_loss)
                self.total_train_loss.append(batch_loss)

                if i == training_steps:
                    break

                if (time.time() - last_update) >= update_every * 3600:
                    last_update = time.time()
                    print('\n  [Model has been successfully training for {:.1f} hours.'
                          'Currently at step {} of {}, in epoch {}]\n'.format((last_update - start_time) / 3600,
                                                                            i+1, training_steps, epoch+1))

            avg = epoch_loss_avg.result()

            if not debug:
                with self.epoch_summary_writer.as_default():
                    tf.summary.scalar('Average loss per epoch', avg, step=epoch+1)
            self.epoch_loss.append(avg)

            print('End of epoch {}. Average training Loss: {:.3f}'.format(epoch+1, avg))

            if test_data:
                acc = self.evaluate(test_data, validation_steps)
                if not debug:
                    with self.epoch_summary_writer.as_default():
                        tf.summary.scalar('Validation accuracy', acc, step=epoch+1)
                self.validation_acc.append(acc)
                print('Validation accuracy: {:.2f}%'.format(acc*100))

        time_elapsed = (time.time() - start_time)
        if time_elapsed < 3600:
            print('Training finished after {} mimutes.\n'.format(int(time_elapsed / 60)))
        else:
            print('Training finished after {:.1f} hours.\n'.format(time_elapsed / 3600))

    def evaluate(self, data, steps):

        accuracy = tf.keras.metrics.Mean()

        for i, (x, y) in enumerate(data):

            preds = self.model(x)

            y_pred = [np.argmax(p) for p in preds]
            y_true = [np.argmax(p) for p in y]
            accuracy(accuracy_score(y_true, y_pred))

            if i == steps:
                break

        return accuracy.result().numpy()


if __name__ == '__main__':

    # Experiment identifier
    identifier = config['identifier']

    # Model configurations
    model_id = config['model']
    pretrained_seeker_weights = None

    if config['seeker_weights']:
        p = Path(config['seeker_weights']).absolute()
        if p.exists():
            pretrained_seeker_weights = config['seeker_weights']
        else:
            print('No valid seeker weights found at:', config['seeker_weights'])
            print('Will train seeker from skratch.')

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
        train_set = utils.datagen.mnist(batch_size=batch_size, set='train')
        test_set = utils.datagen.mnist(batch_size=batch_size, set='test')
    elif config['config'] == 'cifar10':
        train_set = utils.datagen.cifar10(batch_size=batch_size, set='train', channels=channels)
        test_set = utils.datagen.cifar10(batch_size=batch_size, set='test', channels=channels)
    else:
        data_dir = Path(config['data_dir'])
        train_set = utils.datagen.image_generator(data_dir / 'train', batch_size=batch_size, image_shape=image_shape,
                                                  channels=channels)
        test_set = utils.datagen.image_generator(data_dir / 'test', batch_size=batch_size, image_shape=image_shape,
                                                 channels=channels)

    log_dir = str(Path('logs') / config['config'] / 'seeker' / identifier)
    weight_dir = str(Path('weights') / config['config'] / 'seeker' / identifier)

    print('Initializing seeker model')
    seeker = networks.seek.available_models[model_id](input_shape, num_classes)

    if pretrained_seeker_weights:
        print('Loading pre-trained seeker weights')
        seeker.load_weights(pretrained_seeker_weights)

    seeker_trainer = SeekerTrainer(model=seeker, weight_dir=weight_dir, log_dir=log_dir, debug=debug)

    with utils.training.WeightFailsafe(weight_dir, seeker, debug=debug):
        seeker_trainer.train(train_set, training_steps=train_images//batch_size,  max_epochs=max_epochs,
                          test_data=test_set, validation_steps=test_images//batch_size, update_every=6)

    print('Test set accuracy: {:.2f}%'.format(seeker_trainer.evaluate(test_set, test_images//batch_size) * 100))

    if not debug:
        final_weights = str(Path(weight_dir) / 'final_weights.h5')
        print('Saving model to:', final_weights)
        seeker.save_weights(final_weights)
