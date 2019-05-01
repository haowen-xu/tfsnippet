import numpy as np
import tensorflow as tf

from .base import InvertibleActivation

__all__ = ['LeakyReLU']


class LeakyReLU(InvertibleActivation):
    """
    Leaky ReLU activation function.

    `y = x if x >= 0 else alpha * x`
    """

    def __init__(self, alpha=0.2):
        alpha = float(alpha)
        if alpha <= 0 or alpha >= 1:
            raise ValueError('`alpha` must be a float number, and 0 < alpha < '
                             '1: got {}'.format(alpha))

        self._alpha = alpha
        self._inv_alpha = 1. / alpha
        self._log_alpha = float(np.log(alpha))

    def _transform_or_inverse_transform(self, x, compute_y, compute_log_det,
                                        reverse=False):
        y = None
        if compute_y:
            if reverse:
                y = tf.minimum(x * self._inv_alpha, x)
            else:
                y = tf.maximum(x * self._alpha, x)

        log_det = None
        if compute_log_det:
            log_det = tf.cast(tf.less(x, 0), dtype=tf.float32)
            if reverse:
                log_det *= -self._log_alpha
            else:
                log_det *= self._log_alpha

        return y, log_det

    def _transform(self, x, compute_y, compute_log_det):
        return self._transform_or_inverse_transform(
            x=x, compute_y=compute_y, compute_log_det=compute_log_det,
            reverse=False
        )

    def _inverse_transform(self, y, compute_x, compute_log_det):
        return self._transform_or_inverse_transform(
            x=y, compute_y=compute_x, compute_log_det=compute_log_det,
            reverse=True
        )
