import tensorflow as tf
from utils.training import slope
from utils.custom_ops import *


class BinaryDeterministic(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BinaryDeterministic, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return binary_det(x)


class BinaryStochastic(tf.keras.layers.Layer):

    def __init__(self, estimator='st1', scale=True, slope_increase_rate=0.000001, baseline=True, seed=None, **kwargs):

        self.estimator = self.check_estimator(estimator)
        self.slope = 1.
        self.slope_increase_rate = slope_increase_rate
        self.scale = scale
        self.baseline = baseline
        self.seed = seed

        super(BinaryStochastic, self).__init__(**kwargs)

    def call(self, x, **kwargs):

        if self.estimator == 'st1':
            return binary_stochastic_st_1(x) #, scale=self.scale, seed=self.seed)
        elif self.estimator == 'st2':
            return binary_stochastic_st_2(x) # , scale=self.scale, seed=self.seed)
        elif self.estimator == 'sa':
            self.slope *= (1 + self.slope_increase_rate)
            return binary_stochastic_sa(x, self.slope) #, slope=self.slope, scale=self.scale, seed=self.seed)
        elif self.estimator == 'reinforce':
            return binary_stochastic_reinforce(x)# , scale=self.scale, baseline=self.baseline, seed=self.seed)

    @staticmethod
    def check_estimator(estimator):

        error = ValueError("estimator must be one of:\n 'st1', 'st2', 'sa', 'reinforce'")

        if not isinstance(estimator, str):
            raise error

        estimator = estimator.lower()

        if 'reinforce' in estimator or estimator == 'r':
            return 'reinforce'

        elif any([word in estimator for word in ('sa', 'slope', 'anneal')]):
            return 'sa'

        elif any([word in estimator for word in ('st', 'straight', 'through')]):
            return 'st2' if '2' in estimator else 'st1'

        else:
            raise error
