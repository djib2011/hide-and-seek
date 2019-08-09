from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import __main__

from pathlib import Path
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser()

# Name of the current "experiment"
parser.add_argument('--identifier', type=str, default=None, help='Name of the current experiment, will be used to name '
                                                                 'the folders containing the logs and the weights')

# Preset configuration name
parser.add_argument('--config', type=str, default='animals', help='Name of a valid configuration from "config.txt"')

# Number of times to train the model
parser.add_argument('--num_trainings', type=int, default=None, help='How many times to train the model')

# Experimental options
parser.add_argument('--stochastic', action='store_true', default=None, help='Select if you want to use Binary '
                                                                            'Stochastic Neurons, instead of '
                                                                            'Deterministic ones.')
parser.add_argument('--estimator', type=str, default=None, help='Name of the gradient estimator. only relevant for '
                                                                'stochastic neurons')
parser.add_argument('--rate', type=float, default=None, help='Slope increase rate of Slope-Annealing estimator (only '
                                                             'relevant for this estimator). How much the slope '
                                                             'increases per epoch. E.g. "0.5" means that at the end of '
                                                             'the first epoch the slope will be 50% larger than what it'
                                                             ' started.')
parser.add_argument('--monitor', type=str, default=None, help='What loss to monitor: "classification" or "total".'
                                                              'Only relevant for adaptive loss weighting.')
parser.add_argument('--patience', type=int, default=None, help='How many iterations to check for a significant change'
                                                               'in classification loss before reducing a. Only '
                                                               'relevant for adaptive loss weighting.')
parser.add_argument('--alpha', type=float, default=None, help='Value for alpha. Should be between 0 and 1. Higher '
                                                              'values cause the "classification loss" to contribute '
                                                              'more to the total loss, while lower values cause the '
                                                              '"mask loss" to contribute more. If not specified, an '
                                                              'adaptive loss weighting will occur.')

# Only relevant for custom configurations, i.e. experiment==None
parser.add_argument('--data_dir', type=str, default=None, help='Directory where training data is located')
parser.add_argument('--image_size', type=int, default=None, help='Image dimensions')
parser.add_argument('--channels', type=int, default=None, help='Number of channels (3 for rgb, 1 for grayscale)')
parser.add_argument('--num_classes', type=int, default=None, help='Number of classes in the dataset')
parser.add_argument('--train_images', type=int, default=None, help='How many images you want to train on. Usually set '
                                                                   'to the training set size. (optional)')
parser.add_argument('--test_images', type=int, default=None, help='How many images you want to test on; usually set '
                                                                  'to the test set size. (optional)')

# Pretrained weights
parser.add_argument('--model', type=str, default=None, help='Type of model to use. Available: "hns_small", "hns_large" '
                                                            'and "hns_resnet"')
parser.add_argument('--hider_weights', type=str, default=None, help='Location of a valid pretrained "hider" weights '
                                                                    'file (optional but recommended)')
parser.add_argument('--seeker_weights', type=str, default=None, help='Location of a valid pretrained "seeker" weights '
                                                                     'file (optional)')

# Training parameters
parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
parser.add_argument('--max_epochs', type=int, default=None, help='Maximum number of epochs. Adaptive loss weighting may'
                                                                 ' cause the network to converge faster.')
parser.add_argument('--gpu', type=str, default=None, help='Which gpu to use. Only relevant for multi-gpu enviromnemts.')
parser.add_argument('--debug', action='store_true', default=False, help='If set to True, no weights or logs will be '
                                                                        'stored for the models. It is intended for '
                                                                        'seeing if a script runs properly, without '
                                                                        'generating empty logs or useless weights.')

if hasattr(__main__, '__file__'):
    opt = parser.parse_args()
else:
    opt = parser.parse_args(args=[])

args = vars(opt)


# Parse config file
defaults = {'batch_size': 64, 'max_epochs': 10, 'gpu': 0, 'model': 'hns_small', 'config': None, 'data_dir': None,
            'image_size': None, 'channels': None, 'num_classes': None, 'hider_weights': None, 'seeker_weights': None,
            'train_images': None, 'test_images': None, 'stochastic': False, 'estimator': 'st1', 'patience': 100,
            'alpha': None, 'monitor': 'classification', 'rate': 0.5, 'debug': False, 'num_trainings': 1}


def parse_config():
    with open('config', 'r') as config:
        line = True
        i = 0
        configs = {}
        while line:
            i += 1
            line = config.readline()
            if not line or line.startswith('#') or line in ['\n', '\r\n']:
                continue
            line = line.split('#')[0].rstrip().strip('\n').strip('\r')
            if line.startswith('[') and line.endswith(']'):
                current_config = line[1:-1]
                configs[current_config] = defaults.copy()
            elif (line.startswith(' ') or line.startswith('\t')) and '=' in line:
                k, v = line.split('=')
                k = k.strip()
                v = v.strip()
                if v.isdigit():
                    v = int(v)
                if not current_config:
                    raise ValueError('Invalid "config" file. Cannot identify first configuration.')
                configs[current_config][k] = v
            else:
                print('Invalid line detected in "config" file, in line {}. Will skip line.'.format(i))
    return configs


configs = parse_config()

# Merge config file with command line arguments
print('\n'*20)

if not args['identifier']:
    print('Warning! Identifier was not set, using the identifier "default"')
    args['identifier'] = 'default'
else:
    print('Identifier: {}'.format(args['identifier']))

if args['config'] == 'example':
    raise ValueError('Preset configuration "example" is only for demonstration. Please select another configuration.')

if args['config']:
    if args['config'] not in configs:
        l = list(configs.keys())
        if 'example' in l:
            l.remove('example')
        raise KeyError('Invalid preset configuration. Please enter one of:\n', l)
    print('Using preset configuration "{}"'.format(args['config']))
    args = {k: args[k] if args[k] is not None else configs[args['config']][k] for k in args}
else:
    print('Using custom configuration')
    args = {k: args[k] if args[k] is not None else defaults[k] for k in args}

# Check for valid arguemnts
config = args.copy()
if config['data_dir']:
    p = Path(config['data_dir']).absolute()
    if not p.is_dir():
        raise OSError('Specified data_dir doesn\'t exist:', data_dir)
    if not config['num_classes']:
        config['num_classes'] = len([x for x in p.glob('train/*') if x.is_dir()])


# Find number of train/test set images
def find_images(directory):
    train = Path(directory)
    return len([x for x in train.rglob('*') if x.suffix.lower() in ('.jpeg', '.png', '.gif', '.jpg')])


if not config['train_images']:
    print('Searching for training set images...')
    config['train_images'] = find_images(Path(config['data_dir']).absolute() / 'train')
if not config['test_images']:
    print('Searching for test set images...')
    config['test_images'] = find_images(Path(config['data_dir']).absolute() / 'test')

if config['debug']:
    print('\n' + '-'*55)
    print('            WARNING! "debug" mode is set.')
    print('            won\'t store weights or logs.')
    print('-'*55 + '\n')

# Print final form of configuration file 
print('\n{:<20} | {}'.format('Argument', 'Value'))
print('-'*55)
for k, v in sorted(config.items()):
    if not k in('identifier', 'config'):
        print('{:<20} | {}'.format(k, v))

# Make another variable dependent on "stochastic"
if config['stochastic']:
    config['binary_type'] = 'stochastic'
else:
    config['binary_type'] = 'deterministic'

# Re-adjust the slope from the human-friendly "per epoch", to the "per iteration" that is needed
config['rate_per_iteration'] = config['rate'] / config['train_images']

print('\n'*5)
