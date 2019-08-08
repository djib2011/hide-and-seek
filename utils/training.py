from pathlib import Path
import numpy as np

loss_tensor = 1.
slope = 1.


class MetricMonitor:

    def __init__(self, steps=10, threshold=0.1):
        self.n = steps
        self.metric = []
        self.full = False
        self.threshold = threshold

    def __call__(self, val):

        self.metric.append(val)

        if self.full:
            self.metric.pop(0)
        else:
            self.full = self.check_full()

    def check_full(self):
        return len(self.metric) >= self.n

    def average(self):
        return np.mean(self.metric)

    def flush(self):
        self.metric = []
        self.full = False

    def no_significant_change(self, threshold=None):

        if not self.full:
            return

        if not threshold:
            threshold = self.threshold

        threshold = self.average() * (1 + threshold)

        return (np.max(np.abs(self.metric)) - np.min(np.abs(self.metric))) >= threshold


class WeightFailsafe:

    def __init__(self, weight_dir=None, model=None, debug=False):

        if not weight_dir:
            weight_dir = '.'

        if not model:
            if 'model' in globals():
                model = globals()['model']
            else:
                raise ValueError('Model not found, please add a model.')

        self.file_name = Path(weight_dir) / 'checkpoint.h5'
        self.model = model
        self.debug = debug

    def __enter__(self):
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if not self.debug:
            print('Training loop terminated. Saving weights to:', self.file_name)
            self.model.save_weights(str(self.file_name))


def transfer_weights(hns, pretrained_hider, pretrained_seeker=None):
    """
    Function meant to transfer the weights from a pretrained Hider and Seeker to a full HideAndSeek model
    :param hns: The HNS model, whose weights we want to update.
    :param pretrained_hider: The Hider model which will give its weights to the first layers of the HNS model.
    :param pretrained_seeker: The Seeker model which will give its weights to the last layers of the HNS model.
                             (This model is optional)
    :return: True, if the transfer is successful.
    """

    # As long as the models' architectures match, transfer the weights
    i = 0

    if pretrained_hider.input_shape != hns.input_shape:
        raise ValueError('Models should have matching input shapes.')

    while i < len(pretrained_hider.layers) and pretrained_hider.layers[i].output_shape == hns.layers[i].output_shape:

        hns.layers[i].set_weights(pretrained_hider.layers[i].get_weights())
        i += 1

    if pretrained_seeker:
        j = 0

        if hns.layers[i].name == 'hider_output':
            i += 1  # skip binary layer (not image masking layer)

        while j < len(pretrained_seeker.layers) and \
                pretrained_seeker.layers[j].output_shape == hns.layers[i].output_shape:

            hns.layers[i].set_weights(pretrained_hider.layers[j].get_weights())
            i += 1
            j += 1

        print('Successfully transferred weights from {} hider and {} seeker layers.'.format(i-j-1, j))

    else:
        print('Successfully transferred weights from the first {} layers.'.format(i))

    return True
