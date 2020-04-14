import tensorflow as tf
import pickle as pkl
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import accuracy_score

import utils
from utils.options import config
import networks


def evaluate(model, data, steps):
    accuracy = tf.keras.metrics.Mean()

    for i, (x, y) in enumerate(data):

        preds = model(x)

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

    # Debug mode
    debug = config['debug']

    # Find weight dir
    weight_dir = Path('weights') / config['config'] / 'seeker' / identifier
    weights = str(weight_dir / 'final_weights.h5')

    # Load dataset
    batch_size = config['batch_size']
    image_shape = (config['image_size'], ) * 2
    test_images = config['test_images']
    channels = config['channels']
    input_shape = image_shape + (channels,)
    num_classes = config['num_classes']

    if config['config'] == 'mnist':
        test_set = utils.datagen.mnist(batch_size=batch_size, set='test')
    elif config['config'] == 'fashion':
        test_set = utils.datagen.fashion(batch_size=batch_size, set='test', channels=channels)
    elif config['config'] == 'cifar10':
        test_set = utils.datagen.cifar10(batch_size=batch_size, set='test', channels=channels)
    elif config['config'] == 'cifar100':
        test_set = utils.datagen.cifar100(batch_size=batch_size, split='test', channels=channels)
    else:
        data_dir = Path(config['data_dir'])
        test_set = utils.datagen.image_generator(data_dir / 'test', batch_size=batch_size, image_shape=image_shape,
                                                 channels=channels)

    # Load model
    seeker = networks.seek.available_models[model_id](input_shape, num_classes)
    seeker.load_weights(weights)

    # Evaluate model
    result = evaluate(model=seeker, data=test_set, steps=test_images//batch_size)
    print('Accuracy: {:.2f}%'.format(result * 100))

    if not debug:
        results_dir = Path(str(weight_dir).replace('weights/', 'results/'))
        if not results_dir.is_dir():
            os.makedirs(str(results_dir))
        with open(str(results_dir / 'baseline.txt'), 'w') as f:
            f.write(str(result))
