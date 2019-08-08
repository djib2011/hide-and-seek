import tensorflow as tf
import numpy as np
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time

import utils
from utils.options import config
import networks


class HNSTrainer:

    def __init__(self, model, weight_dir='weights', log_dir='logs', optimizer=None, loss_function=None, debug=debug):

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
        ys = y_['seeker_output']
        yh = y_['hider_output']

        classification_loss = self.loss_function(y_true=y, y_pred=ys)

        mask_loss = tf.reduce_mean(yh)
        total_loss = self.a * classification_loss + (1 - self.a) * mask_loss

        pixels_kept = mask_loss * tf.cast(tf.size(mask_loss), 'float')
        pixels_hidden = tf.cast(tf.size(mask_loss), 'float') - pixels_kept
        percentage_hidden = (1 - mask_loss) * 100

        self.current_classification_loss = classification_loss  # store for monitor

        if not self.debug:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('pixels hidden', pixels_kept, step=self.iteration)
                tf.summary.scalar('pixels kept', pixels_hidden, step=self.iteration)
                tf.summary.scalar('percentage hidden', percentage_hidden, step=self.iteration)
                tf.summary.scalar('classification loss', classification_loss, step=self.iteration)
                tf.summary.scalar('mask loss', mask_loss, step=self.iteration)
                tf.summary.scalar('loss regulator', self.a, step=self.iteration)
                tf.summary.scalar('total loss', total_loss, step=self.iteration)

        return total_loss

    def train_batch(self, x, y):

        with tf.GradientTape() as tape:
            loss_value = self.loss(x, y)

        grads = tape.gradient(loss_value, self.model.trainable_variables)

        total_grad = np.abs(np.concatenate([g.numpy().flatten() for g in grads], axis=0)).sum()

        if not self.debug:
            with self.train_summary_writer.as_default():
                tf.summary.scalar('gradients', total_grad, step=self.iteration)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss_value

    def train(self, train_data, training_steps, max_epochs=10, test_data=None, validation_steps=None,
              adaptive_weighting=True, alpha=1., a_patience=100, loss_to_monitor='total', update_every=6):

        last_update = start_time = time.time()

        terminate = False

        if adaptive_weighting:
            self.a = 1.
        else:
            self.a = alpha

        monitor = utils.training.MetricMonitor(steps=a_patience)

        print('\n' * 5)

        for epoch in range(max_epochs):

            epoch_loss_avg = tf.keras.metrics.Mean()

            for i, (x, y) in enumerate(train_data):

                self.iteration += 1

                batch_loss = self.train_batch(x, y)

                if loss_to_monitor == 'total':
                    monitor(batch_loss)
                elif loss_to_monitor == 'classification':
                    monitor(self.current_classification_loss)
                else:
                    raise ValueError('Invalid "loss_to_monitor" value. Can either be "total" or "classification".')

                epoch_loss_avg(batch_loss)
                self.total_train_loss.append(batch_loss)

                if not self.debug:
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss monitor', monitor.average(), step=self.iteration)

                if i == training_steps:
                    break

                if adaptive_weighting and monitor.no_significant_change():

                    monitor.flush()

                    self.model.save_weights(weight_dir + '/interm_weights_a_{:.2f}.h5'.format(self.a))
                    self.a -= 0.05

                    if self.a <= 0.05:
                        print('\nAlpha reduction complete. Terminating training loop.')
                        terminate = True
                        break

                    print('\n-----> Training stagnated at iteration {} (epoch {}, batch {}).\n'
                          '   |\n'
                          '   -----> Decreased alpha to {:.2f}'.format(self.iteration+1, epoch+1, i+1, self.a))

                if (time.time() - last_update) >= update_every * 3600:
                    last_update = time.time()
                    print('\n  [Model has been successfully training for {:.1f} hours.'
                          'Currently at step {} of {}, in epoch {}]'.format((last_update - start_time) / 3600,
                                                                            i+1, training_steps, epoch+1))

            avg = epoch_loss_avg.result()

            if not self.debug:
                with self.epoch_summary_writer.as_default():
                    tf.summary.scalar('Average loss per epoch', avg, step=epoch+1)
            self.epoch_loss.append(avg)

            if terminate:
                break

            print('\n' * 2 + '-' * 65)
            print('-' * 65)
            print('          End of epoch {}. Average training Loss: {:.3f}'.format(epoch+1, avg))

            if test_data:
                acc = self.evaluate(test_data, validation_steps)
                if not self.debug:
                    with self.epoch_summary_writer.as_default():
                        tf.summary.scalar('Validation accuracy', acc, step=epoch+1)
                self.validation_acc.append(acc)
                print('               Validation accuracy: {:.2f}%'.format(acc*100))

            print('-' * 65)
            print('-' * 65 + '\n')

        time_elapsed = (time.time() - start_time)
        if time_elapsed < 3600:
            print('Training finished after {} mimutes.\n'.format(int(time_elapsed / 60)))
        else:
            print('Training finished after {:.1f} hours.\n'.format(time_elapsed / 3600))

    def evaluate(self, data, steps):

        accuracy = tf.keras.metrics.Mean()

        for i, (x, y) in enumerate(data):

            preds = self.model(x)['seeker_output']

            y_pred = [np.argmax(p) for p in preds]
            y_true = [np.argmax(p) for p in y]
            accuracy(accuracy_score(y_true, y_pred))

            if i == steps:
                break

        return accuracy.result().numpy()

    def save_sample_images(self, x, y, directory='/tmp/sample_images'):

        y_ = self.model(x)
        ys = y_['seeker_output']
        yh = y_['hider_output']

        accuracy = np.where(np.argmax(ys, axis=1) == np.argmax(y, axis=1), '_hit', '_miss')

        if not Path(directory).is_dir():
            os.makedirs(directory)

        for i in range(len(x)):
            plt.imsave(directory + '/x' + str(i + 1) + '.png', x[i, ..., 0])
            plt.imsave(directory + '/y' + str(i + 1) + accuracy[i] + '.png', yh[i, ..., 0])


if __name__ == '__main__':

    # Experiment identifier
    identifier = config['identifier']

    # Model configurations
    model_id = config['model']
    pretrained_seeker_weights = None
    pretrained_hider_weights = None

    if config['seeker_weights']:
        p = Path(config['seeker_weights']).absolute()
        if p.exists():
            pretrained_seeker_weights = config['seeker_weights']
        else:
            print('No valid seeker weights found at:', config['seeker_weights'])
            print('Will train seeker from skratch.')
    if config['hider_weights']:
        p = Path(config['hider_weights']).absolute()
        if p.exists():
            pretrained_hider_weights = config['hider_weights']
        else:
            print('No valid hider weights found at:', config['hider_weights'])
            print('Will train hider from skratch.')

    # Experimental configurations
    binary_type = config['binary_type']
    stochastic_estimator = config['estimator']
    alpha = config['alpha']
    monitor = config['monitor']
    patience = config['patience']
    slope_increase_rate = config['rate_per_iteration']
    adaptive = alpha is None

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
        train_set = utils.datagen.image_generator(data_dir / 'train',batch_size=batch_size, image_shape=image_shape,
                                                  channels=channels)
        test_set = utils.datagen.image_generator(data_dir / 'test', batch_size=batch_size, image_shape=image_shape,
                                                 channels=channels)

    sub_dirs = Path(config['config']) / 'hns' / binary_type

    if binary_type == 'stochastic':
        sub_dirs = sub_dirs / stochastic_estimator
        if stochastic_estimator == 'sa':
            sub_dirs = sub_dirs / 'rate_{}'.format(str(config['rate']))

    log_dir = str(Path('logs') / sub_dirs / identifier)
    weight_dir = str(Path('weights') / sub_dirs / identifier)

    print('Initializing Hide-and-Seek model')
    hns_model = networks.hns.available_models[model_id](input_shape, num_classes, binary_type=binary_type,
                                                        stochastic_estimator=stochastic_estimator,
                                                        slope_increase_rate=slope_increase_rate)

    if pretrained_seeker_weights:
        print('Loading pre-trained seeker')
        seeker = networks.seek.available_models[model_id](input_shape, num_classes)
        seeker.load_weights(pretrained_seeker_weights)
    else:
        seeker = None

    if pretrained_hider_weights:
        print('Loading pre-trained hider')
        hider = networks.hide.available_models[model_id](input_shape)
        hider.load_weights(pretrained_hider_weights)
        print('Transfering weights')
        utils.training.transfer_weights(hns_model, hider, seeker)

        del hider, seeker

    hns_trainer = HNSTrainer(model=hns_model, weight_dir=weight_dir, log_dir=log_dir, debug=debug)

    if alpha:
        print('Training model with constant loss weighting. alpha set to {}.'.format(alpha))
    else:
        print('Training model with adaptive loss weighting. Initial alpha set to 1.0.')

    with utils.training.WeightFailsafe(weight_dir, hns_model, debug=debug):
        hns_trainer.train(train_set, training_steps=train_images//batch_size,  max_epochs=max_epochs,
                          test_data=test_set, validation_steps=test_images//batch_size, adaptive_weighting=adaptive,
                          alpha=alpha, a_patience=patience, loss_to_monitor=monitor, update_every=6)

    print('Test set accuracy: {:.2f}%'.format(hns_trainer.evaluate(test_set, test_images//batch_size) * 100))

    if not debug:
        final_weights = str(Path(weight_dir) / 'final_weights.h5')
        print('Saving model to:', final_weights)
        hns_model.save_weights(final_weights)
