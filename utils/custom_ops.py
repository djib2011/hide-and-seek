import tensorflow as tf
from utils.training import loss_tensor, slope


@tf.custom_gradient
def binary_det(x, name=None):
    """
    Converts a real valued tensor, whose values are in [0, 1] to a tensor with binary values (0 and 1).
    Does this by invoking tf.round (which rounds the values).
    This operation is set to have no effect during the gradient computation.
    """
    y = tf.round(x, name=name)

    return y, lambda dy: dy


@tf.custom_gradient
def binary_stochastic_st_1(x, scale=True, seed=None):
    s = tf.math.sigmoid(x)

    if scale:
        minval = tf.reduce_min(s)
        maxval = tf.reduce_max(s)
    else:
        minval = 0
        maxval = 1

    tf.random.set_seed(seed)
    z = tf.random.uniform(tf.shape(s), minval=minval, maxval=maxval)
    y = tf.math.ceil(s - z)

    return y, lambda dy: dy


@tf.custom_gradient
def binary_stochastic_st_2(x, scale=True, seed=None):
    s = tf.math.sigmoid(x)

    if scale:
        minval = tf.reduce_min(s)
        maxval = tf.reduce_max(s)
    else:
        minval = 0
        maxval = 1

    tf.random.set_seed(seed)
    z = tf.random.uniform(tf.shape(s), minval=minval, maxval=maxval)
    y = tf.math.ceil(s - z)

    return y, lambda dy: dy * x * (1 - x)


@tf.custom_gradient
def binary_stochastic_sa(x, slope=1, scale=True, seed=None):
    s = tf.math.sigmoid(slope * x)

    if scale:
        minval = tf.reduce_min(s)
        maxval = tf.reduce_max(s)
    else:
        minval = 0
        maxval = 1

    tf.random.set_seed(seed)
    z = tf.random.uniform(tf.shape(s), minval=minval, maxval=maxval)
    y = tf.math.ceil(s - z)

    return y, lambda dy: (dy * x * (1 - x) * slope, dy * x * (1 - x) * slope)
    # I'm not sure how this custom_gradient works exactly. I guess it takes 2 input tensors so it wants 2 gradient
    # functions to be returned, as indicated by the "expected 2 gradients got 1..." error message.


@tf.custom_gradient
def binary_stochastic_reinforce(x, baseline=True, scale=True, seed=None):
    s = tf.math.sigmoid(x)

    if scale:
        minval = tf.reduce_min(s)
        maxval = tf.reduce_max(s)
    else:
        minval = 0
        maxval = 1

    tf.random.set_seed(seed)
    z = tf.random.uniform(tf.shape(s), minval=minval, maxval=maxval)  # seed argument doens't work
    y = tf.math.ceil(s - z)


    global loss_tensor
    # gradient computation
    diff = y - s
    if baseline:
        baseline = tf.math.reduce_mean(diff ** 2 * loss_tensor) / tf.math.reduce_mean(diff ** 2)
    estimator = diff * (loss_tensor - baseline)

    '''
    def grad_fn(dy, baseline=baseline):
        diff = y - s
        if baseline:
            baseline = tf.math.reduce_mean(diff ** 2 * loss_tensor) / tf.math.reduce_mean(diff ** 2)
        estimator = diff * (loss_tensor - baseline)
        return dy * estimator
    '''

    return y, lambda dy: estimator * dy
