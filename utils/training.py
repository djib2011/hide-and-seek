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


def transfer_weights(hns, pretrained_hider=None, pretrained_seeker=None, debug=False):
    """
    Function meant to transfer the weights from a pretrained Hider and Seeker to a full HideAndSeek model
    :param hns: The HNS model, whose weights we want to update.
    :param pretrained_hider: The Hider model which will give its weights to the first layers of the HNS model.
    :param pretrained_seeker: The Seeker model which will give its weights to the last layers of the HNS model.
                             (This model is optional)
    :return: True, if the transfer is successful.
    """

    def transfer_weights_starting_from_layer(layer_index, pretrained_model):

        # As long as the models' architectures match, transfer the weights
        i = layer_index
        j = 0

        if pretrained_model.input_shape != hns.input_shape:
            raise ValueError('Models should have matching input shapes.')

        while j < len(pretrained_model.layers) and \
                pretrained_model.layers[j].output_shape == hns.layers[i].output_shape:

            if debug:
                print('Pretrained:', pretrained_model.layers[j].name)
                print('HNS:       ', hns.layers[i].name)

            hns.layers[i].set_weights(pretrained_model.layers[j].get_weights())
            i += 1
            j += 1

        return i - layer_index

    h = s = 0

    if pretrained_hider:
        # Start from the beginning for the hider
        start_from_layer = 0
        h = transfer_weights_starting_from_layer(start_from_layer, pretrained_hider)

    if pretrained_seeker:
        # Start from the maksing layer for the seeker
        start_from_layer = hns.layers.index(hns.get_layer('hider_output')) + 1
        s = transfer_weights_starting_from_layer(start_from_layer, pretrained_seeker)

    print('Transferred weights from {} hider and {} seeker layers.'.format(h, s))

