import tensorflow as tf

from tfsnippet.utils import (flatten, unflatten, add_name_and_scope_arg_doc,
                             reopen_variable_scope)
from .base import MultiLayerFlow

__all__ = ['PlanarNormalizingFlow']


class PlanarNormalizingFlow(MultiLayerFlow):
    """
    Planar Normalizing Flow with activation function `tanh` as well as the
    invertibility trick from (Danilo 2016).  `x` is assumed to be a 1-D
    random variable.

    .. math::

        \\begin{aligned}
            \\mathbf{y} &= \\mathbf{x} +
                \\mathbf{\\hat{u}} \\tanh(\\mathbf{w}^\\top\\mathbf{x} + b) \\\\
            \\mathbf{\\hat{u}} &= \\mathbf{u} +
                \\left[m(\\mathbf{w}^\\top \\mathbf{u}) -
                       (\\mathbf{w}^\\top \\mathbf{u})\\right]
                \\cdot \\frac{\\mathbf{w}}{\\|\\mathbf{w}\\|_2^2} \\\\
            m(a) &= -1 + \\log(1+\\exp(a))
        \\end{aligned}
    """

    @add_name_and_scope_arg_doc
    def __init__(self,
                 n_units,
                 n_layers=1,
                 w_initializer=tf.random_normal_initializer(0., 0.01),
                 b_initializer=tf.zeros_initializer(),
                 u_initializer=tf.random_normal_initializer(0., 0.01),
                 w_regularizer=None,
                 b_regularizer=None,
                 u_regularizer=None,
                 trainable=True,
                 dtype=tf.float32,
                 name=None,
                 scope=None):
        """
        Construct a new :class:`PlanarNormalizingFlow`.

        Args:
            n_units (int): The size of the last axis of `x`.
            n_layers (int): The number of normalizing flow layers.
                (default 1)
            w_initializer: The initializer for parameter `w`.
            b_initializer: The initializer for parameter `b`.
            u_initializer: The initializer for parameter `u`.
            w_regularizer: The regularizer for parameter `w`.
            b_regularizer: The regularizer for parameter `b`.
            u_regularizer: The regularizer for parameter `u`.
            trainable (bool): Whether or not the parameters are trainable?
                (default :obj:`True`)
            dtype: The data type of the transformed `y`.
        """
        n_units = int(n_units)
        super(PlanarNormalizingFlow, self).__init__(
            n_layers=n_layers,
            value_ndims=1,
            dtype=dtype,
            name=name,
            scope=scope
        )

        self._n_units = n_units
        self._layer_params = []

        with reopen_variable_scope(self.variable_scope):
            for layer_id in range(self.n_layers):
                with tf.variable_scope('_{}'.format(layer_id)):
                    w = tf.get_variable(
                        'w',
                        shape=[1, n_units],
                        dtype=self._dtype,
                        initializer=w_initializer,
                        regularizer=w_regularizer,
                        trainable=trainable
                    )
                    b = tf.get_variable(
                        'b',
                        shape=[1],
                        dtype=self._dtype,
                        initializer=b_initializer,
                        regularizer=b_regularizer,
                        trainable=trainable
                    )
                    u = tf.get_variable(
                        'u',
                        shape=[1, n_units],
                        dtype=self._dtype,
                        initializer=u_initializer,
                        regularizer=u_regularizer,
                        trainable=trainable
                    )
                    wu = tf.matmul(w, u, transpose_b=True)  # wu.shape == [1]
                    u_hat = u + (-1 + tf.nn.softplus(wu) - wu) * \
                        w / tf.reduce_sum(tf.square(w))  # shape == [1, n_units]
                    self._layer_params.append((w, b, u, u_hat))

    @property
    def n_units(self):
        """
        Get the size of the last axis of `x`.

        Returns:
            int: The size of the last axis of `x`.
        """
        return self._n_units

    def _transform_layer(self, layer_id, x, compute_y, compute_log_det):
        w, b, u, u_hat = self._layer_params[layer_id]

        # flatten x for better performance
        x, s1, s2 = flatten(x, 2)  # x.shape == [?, n_units]
        wxb = tf.matmul(x, w, transpose_b=True) + b  # shape == [?, 1]
        tanh_wxb = tf.tanh(wxb)  # shape == [?, 1]

        # compute y = f(x)
        y = None
        if compute_y:
            y = x + u_hat * tanh_wxb  # shape == [?, n_units]
            y = unflatten(y, s1, s2)

        # compute log(det|df/dz|)
        log_det = None
        if compute_log_det:
            grad = 1. - tf.square(tanh_wxb)  # dtanh(x)/dx = 1 - tanh^2(x)
            phi = grad * w  # shape == [?, n_units]
            u_phi = tf.matmul(phi, u_hat, transpose_b=True)  # shape == [?, 1]
            det_jac = 1. + u_phi  # shape == [?, 1]
            log_det = tf.log(tf.abs(det_jac))  # shape == [?, 1]
            log_det = unflatten(tf.squeeze(log_det, -1), s1, s2)

        # now returns the transformed sample and log-determinant
        return y, log_det

    @property
    def explicitly_invertible(self):
        return False

    # provide this method to avoid abstract class warning
    def _inverse_transform_layer(self, layer_id, y, compute_x, compute_log_det):
        pass  # pragma: no cover
