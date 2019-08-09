import tensorflow as tf
from pathlib import Path
import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import utils
from utils.options import config
import networks


class HNSEvaluator:

    def __init__(self, model, weight_dir, debug=False):

        self.weight_dir = Path(weight_dir)
        self.weights = self.get_weights()
        self.target_dir = Path(str(self.weight_dir).replace('weights/', 'results/'))
        self.model = model
        self.results = {}
        self.debug = debug

    def complete_evaluation(self, data, steps, batches_to_save=1):

        print('Beginning evaluation...')
        self.evaluate(data, steps)

        print('Saving results...')
        if not self.debug and not self.target_dir.is_dir():
            os.makedirs(str(self.target_dir))
        self.save_results()

        print('Generating sample images...')
        self.generate_sample_images(data, batches_to_save)

        print('Evaluation complete. Results stored in: {}'.format(str(self.target_dir)))

    def get_weights(self):

        weights = {a: str(self.weight_dir / 'interm_weights_a_{:.2f}.h5'.format(a))
                   for a in np.arange(0.1, 1.01, 0.05)}
        weights['final'] = self.weight_dir / 'final_weights.h5'
        return weights

    def evaluate(self, data, steps):

        self.results = {}

        for identifier, weight in self.weights.items():

            if identifier != 'final':
                print('Evaluating model for alpha = {:.2f}'.format(identifier))
                self.model.load_weights(weight)
                accuracy = self.evaluate_model(data, steps)
                self.results[identifier] = accuracy
                print('Accuracy: {:.2f}%'.format(accuracy * 100))

        return self.results

    def evaluate_model(self, data, steps):

        accuracy = tf.keras.metrics.Mean()

        for i, (x, y) in enumerate(data):

            preds = self.model(x)['seeker_output']

            y_pred = [np.argmax(p) for p in preds]
            y_true = [np.argmax(p) for p in y]
            accuracy(accuracy_score(y_true, y_pred))

            if i == steps:
                break

        return accuracy.result().numpy()

    def save_results(self):

        target_file = str(self.target_dir / 'results.pkl')
        if not self.debug:
            pkl.dump(self.results, open(target_file, 'wb'))

    def generate_sample_images(self, data, batches=1):

        for a in np.arange(0.1, 1.01, 0.05):

            directory = str(self.target_dir / 'a_{:.2f}'.format(a))

            if not self.debug and not Path(directory).is_dir():
                os.makedirs(directory)
            self.model.load_weights(self.weights[a])

            for b, (x, y) in enumerate(data):
                y_ = self.model(x)
                ys = y_['seeker_output']
                yh = y_['hider_output']

                accuracy = np.where(np.argmax(ys, axis=1) == np.argmax(y, axis=1), '_hit', '_miss')

                for j in range(len(x)):
                    i = (b + 1) * j
                    if not self.debug:
                        plt.imsave(directory + '/x' + str(i + 1) + '.png', x[i, ..., 0])
                        plt.imsave(directory + '/y' + str(i + 1) + accuracy[i] + '.png', yh[i, ..., 0])

                if (b + 1) == batches:
                    break


if __name__ == '__main__':

    # Experiment identifier
    identifier = config['identifier']

    # Model configurations
    model_id = config['model']

    # Debug mode
    debug = config['debug']

    # Find weight dir
    binary_type = config['binary_type']
    stochastic_estimator = config['estimator']
    slope_increase_rate = config['rate_per_iteration']

    sub_dirs = Path(config['config']) / 'hns' / binary_type

    if binary_type == 'stochastic':
        sub_dirs = sub_dirs / stochastic_estimator
        if stochastic_estimator == 'sa':
            sub_dirs = sub_dirs / 'rate_{}'.format(str(config['rate']))

    weight_dir = str(Path('weights') / sub_dirs / identifier)


    # Load dataset
    batch_size = config['batch_size']
    image_shape = (config['image_size'], ) * 2
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

    # Load model
    hns_model = networks.hns.available_models[model_id](input_shape, num_classes, binary_type=binary_type,
                                                        stochastic_estimator=stochastic_estimator,
                                                        slope_increase_rate=slope_increase_rate)

    # Run evaluator
    evaluator = HNSEvaluator(model=hns_model, weight_dir=weight_dir, debug=debug)
    evaluator.complete_evaluation(test_set, steps=test_images//batch_size, batches_to_save=1)
