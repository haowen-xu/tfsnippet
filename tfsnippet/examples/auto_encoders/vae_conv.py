# -*- coding: utf-8 -*-
import functools

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from tfsnippet.modules import VAE
from tfsnippet.dataflow import DataFlow
from tfsnippet.distributions import Normal, Bernoulli
from tfsnippet.examples.nn import (resnet_block,
                                   deconv_resnet_block,
                                   reshape_conv2d_to_flat,
                                   l2_regularizer,
                                   regularization_loss)
from tfsnippet.examples.utils import (load_mnist,
                                      create_session,
                                      Config,
                                      anneal_after, save_images_collection)
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import AnnealingDynamicValue, LossTrainer, Evaluator
from tfsnippet.utils import global_reuse, get_default_session_or_error, makedirs


class ExpConfig(Config):
    # model parameters
    z_dim = 32

    # training parameters
    max_epoch = 3000
    batch_size = 128
    l2_reg = 0.0001
    initial_lr = 0.0001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 300
    lr_anneal_step_freq = None

    # evaluation parameters
    test_n_z = 100


config = ExpConfig()


@global_reuse
def h_for_q_z(x, is_training):
    with arg_scope([resnet_block],
                   activation_fn=tf.nn.relu,
                   normalizer_fn=functools.partial(
                       tf.layers.batch_normalization,
                       axis=1,
                       training=is_training
                   ),
                   dropout_fn=functools.partial(
                       tf.layers.dropout,
                       training=is_training
                   ),
                   kernel_regularizer=l2_regularizer(config.l2_reg)):
        x = tf.to_float(x)
        x = tf.reshape(x, [-1, 1, 28, 28])
        x = resnet_block(x, 16)  # output: (16, 28, 28)
        x = resnet_block(x, 32, strides=2)  # output: (32, 14, 14)
        x = resnet_block(x, 32)  # output: (32, 14, 14)
        x = resnet_block(x, 64, strides=2)  # output: (64, 7, 7)
        x = resnet_block(x, 64)  # output: (64, 7, 7)
    x = reshape_conv2d_to_flat(x)
    return {
        'mean': tf.layers.dense(x, config.z_dim, name='z_mean'),
        'logstd': tf.layers.dense(x, config.z_dim, name='z_logstd'),
    }


@global_reuse
def h_for_p_x(z, is_training):
    with arg_scope([deconv_resnet_block],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=l2_regularizer(config.l2_reg)):
        origin_shape = tf.shape(z)[:-1]  # might be (n_z, n_batch)
        z = tf.reshape(z, [-1, config.z_dim])
        z = tf.reshape(tf.layers.dense(z, 64 * 7 * 7), [-1, 64, 7, 7])
        z = deconv_resnet_block(z, 64)  # output: (64, 7, 7)
        z = deconv_resnet_block(z, 32, strides=2)  # output: (32, 14, 14)
        z = deconv_resnet_block(z, 32)  # output: (32, 14, 14)
        z = deconv_resnet_block(z, 16, strides=2)  # output: (16, 28, 28)
        z = tf.layers.conv2d(
            z, 1, (1, 1), padding='same', name='feature_map_to_pixel',
            data_format='channels_first')  # output: (1, 28, 28)
    x_logits = tf.reshape(z, tf.concat([origin_shape, [784]], axis=0))
    return {'logits': x_logits}


def main():
    # load mnist data
    (x_train, y_train), (x_test, y_test) = \
        load_mnist(shape=[784], dtype=np.float32, normalize=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None,) + x_train.shape[1:], name='input_x')
    is_training = tf.placeholder(
        dtype=tf.bool, shape=(), name='is_training')

    # build the sampled x
    sample_input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + x_train.shape[1:],
        name='sample_input_x'
    )
    with tf.name_scope('sample_x'):
        uniform_samples = tf.random_uniform(
            shape=tf.shape(sample_input_x), minval=0., maxval=1.,
            dtype=sample_input_x.dtype
        )
        sampled_x = tf.cast(
            tf.less(uniform_samples, sample_input_x), dtype=tf.int32)

    # build the model
    vae = VAE(
        p_z=Normal(mean=tf.zeros([config.z_dim]),
                   std=tf.ones([config.z_dim])),
        p_x_given_z=Bernoulli,
        q_z_given_x=Normal,
        h_for_p_x=functools.partial(h_for_p_x, is_training=is_training),
        h_for_q_z=functools.partial(h_for_q_z, is_training=is_training),
    )
    vae_loss = vae.get_training_loss(input_x)
    loss = vae_loss + regularization_loss()
    lower_bound = -vae_loss
    test_chain = vae.chain(input_x, n_z=config.test_n_z)
    test_nll = tf.reduce_mean(
        test_chain.vi.evaluation.importance_sampling_log_likelihood())

    # derive the optimizer
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
    learning_rate_var = AnnealingDynamicValue(config.initial_lr,
                                              config.lr_anneal_factor)
    params = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.control_dependencies(
            tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, var_list=params)

    # derive the plotting function
    def plot_samples(loop):
        with loop.timeit('plot_time'):
            session = get_default_session_or_error()
            images = session.run(x_plots, feed_dict={is_training: False})
            makedirs('vae_conv/plotting', exist_ok=True)
            save_images_collection(
                images=images,
                filename='vae_conv/plotting/{}.png'.format(loop.epoch),
                grid_size=(10, 10)
            )

    with tf.name_scope('plot_x'):
        x_plots = tf.reshape(
            tf.cast(
                255 * tf.sigmoid(vae.model(n_z=100)['x'].distribution.logits),
                dtype=tf.uint8
            ),
            [-1, 28, 28]
        )

    # prepare for training and testing data
    def input_sampler(x):
        session = get_default_session_or_error()
        return [session.run(sampled_x, feed_dict={sample_input_x: x})]

    train_flow = DataFlow.arrays([x_train], config.batch_size, shuffle=True,
                                 skip_incomplete=True).map(input_sampler)
    # eval_flow = DataFlow.arrays([x_train], config.batch_size).map(x_sampler)
    test_flow = DataFlow.arrays([x_test], config.batch_size).map(input_sampler)

    with create_session().as_default():
        # # fix the samples used by test_flow
        # test_flow = test_flow.to_arrays_flow(config.batch_size)

        # train the network
        with TrainLoop(params,
                       max_epoch=config.max_epoch,
                       summary_dir='vae_conv/train_summary',
                       early_stopping=False) as loop:
            trainer = LossTrainer(
                loop, loss, train_op, [input_x], train_flow,
                feed_dict={learning_rate: learning_rate_var, is_training: True}
            )
            anneal_after(
                trainer, learning_rate_var, epochs=config.lr_anneal_epoch_freq,
                steps=config.lr_anneal_step_freq
            )
            trainer.evaluate_after_epochs(
                Evaluator(
                    loop,
                    metrics={'test_nll': test_nll, 'test_lb': lower_bound},
                    inputs=[input_x],
                    data_flow=test_flow,
                    feed_dict={is_training: False},
                    time_metric_name='test_time'
                ),
                freq=10
            )
            trainer.evaluate_after_epochs(
                functools.partial(plot_samples, loop), freq=10)
            trainer.log_after_epochs(freq=1)
            trainer.run()


if __name__ == '__main__':
    main()
