import functools

import tensorflow as tf
from tensorflow import keras as K

from tfsnippet.distributions import Distribution
from tfsnippet.modules import Module, Sequential
from tfsnippet.utils import global_reuse, instance_reuse

__all__ = ['wgan_gp_cost']

K.layers.Input


def wgan_gp_cost(real_data, fake_data, discriminator, lambda_):
    # derive the real and fake outputs
    disc_real = discriminator(real_data)
    disc_fake = discriminator(fake_data)

    # assemble the loss for generator
    g_loss = -tf.reduce_mean(disc_fake)

    # assemble the loss for discriminator
    alpha = tf.random_uniform(
        shape=tf.shape(fake_data),
        minval=0.,
        maxval=1.,
    )
    interpolates = real_data + (alpha * (fake_data - real_data))
    interpolates_cost = discriminator(interpolates)
    interpolates_grad = tf.gradients(interpolates_cost, interpolates)
    slopes = tf.sqrt(
        tf.reduce_sum(
            tf.square(interpolates_grad),
            axis=-1  # TODO: is this parameter always right?
        )
    )
    # tf.reduce_mean corresponds to E_{p_g}
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    d_loss = (tf.reduce_mean(disc_fake) -
              tf.reduce_mean(disc_real) +
              lambda_ * gradient_penalty)

    return g_loss, d_loss


class WGAN_GP(Module):

    def __init__(self, p_z, g_net, name=None, scope=None):
        super(WGAN_GP, self).__init__(name=name, scope=scope)
        self.p_z = p_z  # type: Distribution
        self.g_net = g_net  # type: Module or (*) -> tf.Tensor

    @instance_reuse
    def generator(self, z=None, n_samples=None):
        if z is not None:
            if n_samples is None:
                raise ValueError('`n_samples` must not be None if `z` is '
                                 'not specified.')
            z = self.p_z.sample(n_samples)
        return self.g_net(z)


if __name__ == '__main__':
    model_scale = 64

    with tf.variable_scope('Generator'):
        g_net = Sequential([
            K.layers.Dense(model_scale * 4 * 4 * 4, name='Generator.Input',
                           activation=tf.nn.relu),
            K.layers.Reshape([-1, 4 * model_scale, 4, 4]),
            K.layers.Convolution2DTranspose
        ])
