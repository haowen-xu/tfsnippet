import tensorflow as tf
from tfsnippet.examples.utils import int_shape, flatten, unflatten

__all__ = ['planar_normalizing_flow']


def planar_normalizing_flow(
        z,
        log_qz,
        w_initializer=tf.random_normal_initializer(0., 0.01),
        b_initializer=tf.zeros_initializer(),
        u_initializer=tf.random_normal_initializer(0., 0.01),
        w_regularizer=None,
        b_regularizer=None,
        u_regularizer=None,
        trainable=True,
        name='planar_normalizing_flow',
        scope=None):
    """
    Apply Planar Normalizing Flow transformation along the last axis of `z`.

    .. math ::
        f(z_t) = z_{t-1} + h(z_{t-1} * w_t + b_t) * u_t

    with activation function `tanh` as well as the invertibility trick
    from (Danilo 2016).

    Args:
        z: A N-D (N>=2) `float32` Tensor, the samples to be transformed.
        log_qz: A (N-1)-D `float32` Tensor, the log-probabilities of the
            samples.  The shape should be the same as the first (N-1)
            dimensions of `z`.
        w_initializer: The initializer for parameter `w`.
        b_initializer: The initializer for parameter `b`.
        u_initializer: The initializer for parameter `u`.
        w_regularizer: The regularizer for parameter `w`, optional.
        b_regularizer: The regularizer for parameter `b`, optional.
        u_regularizer: The regularizer for parameter `u`, optional.
        trainable (bool): Whether or not the parameters are trainable?
            (default :obj:`True`)
        name: The default name for the variable scope.
            (default "planar_normalizing_flow")
        scope: The variable scope, will override `name`.

    Returns:
        (tf.Tensor, tf.Tensor): The transformed samples, and the transformed
            log-probability.
    """
    # check `z` and `log_qz`
    z = tf.convert_to_tensor(z)
    log_qz = tf.convert_to_tensor(log_qz)
    dtype = z.dtype.base_dtype
    if not dtype.is_floating:
        raise TypeError('`z` is expected to be a float tensor, but got '
                        'dtype {}.'.format(z.dtype))

    if z.get_shape() is None or len(z.get_shape()) < 2:
        raise ValueError('The rank of `z` must be fixed and must be at least '
                         '2-dimensional.')
    n_units = int_shape(z)[-1]
    if n_units is None:
        raise ValueError('The last dimension of `z` must be deterministic.')
    if int_shape(z)[:-1] != int_shape(log_qz):
        raise ValueError(
            'The static shape mismatch between `z` and `log_qz`: {} vs {}'.
            format(z.get_shape()[:-1], log_qz.get_shape())
        )

    # derive the normalizing flow
    with tf.variable_scope(scope, default_name=name):
        # create variables
        w = tf.get_variable(
            'w',
            shape=[1, n_units],
            dtype=dtype,
            initializer=w_initializer,
            regularizer=w_regularizer,
            trainable=trainable
        )
        b = tf.get_variable(
            'b',
            shape=[1],
            dtype=dtype,
            initializer=b_initializer,
            regularizer=b_regularizer,
            trainable=trainable
        )
        u = tf.get_variable(
            'u',
            shape=[1, n_units],
            dtype=dtype,
            initializer=u_initializer,
            regularizer=u_regularizer,
            trainable=trainable
        )

        # flatten z for better performance
        z, s1, s2 = flatten(z, 2)  # z.shape == [?, n_units]

        # enforce invertible mapping
        wu = tf.matmul(w, u, transpose_b=True)  # shape == [1]
        u_hat = u + (-1 + tf.nn.softplus(wu) - wu) * \
            w / tf.reduce_sum(tf.square(w))  # shape == [1, n_units]

        # compute f(z)
        wzb = tf.matmul(z, w, transpose_b=True) + b  # shape == [?, 1]
        tanh_wzb = tf.tanh(wzb)  # shape == [?, 1]
        fz = z + u_hat * tanh_wzb  # shape == [?, n_units]
        fz = unflatten(fz, s1, s2)

        # compute log(det|df/dz|)
        grad = 1. - tf.square(tanh_wzb)  # dtanh(x)/dx = 1 - tanh^2(x)
        phi = grad * w  # shape == [?, n_units]
        u_phi = tf.matmul(phi, u_hat, transpose_b=True)  # shape == [?, 1]
        det_jac = 1. + u_phi  # shape == [?, 1]
        log_det_jac = tf.log(tf.abs(det_jac))  # shape == [?, 1]

        # compute log q(f(z))
        log_q_fz = log_qz - \
            unflatten(tf.squeeze(log_det_jac, -1), s1, s2)

        # now returns the transformed sample and log-prob
        return fz, log_q_fz
