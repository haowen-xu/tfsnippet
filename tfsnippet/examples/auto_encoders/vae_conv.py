# -*- coding: utf-8 -*-
import functools

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope, add_arg_scope

from tfsnippet.modules import VAE
from tfsnippet.dataflow import DataFlow
from tfsnippet.distributions import Normal, Bernoulli
from tfsnippet.examples.nn import (dense,
                                   resnet_block,
                                   deconv_resnet_block,
                                   reshape_conv2d_to_flat,
                                   l2_regularizer,
                                   regularization_loss,
                                   conv2d,
                                   batch_norm_2d)
from tfsnippet.examples.utils import (load_mnist,
                                      create_session,
                                      Config,
                                      anneal_after,
                                      save_images_collection,
                                      Results,
                                      MultiGPU,
                                      get_batch_size,
                                      flatten,
                                      unflatten)
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import AnnealingDynamicValue, Trainer, Evaluator
from tfsnippet.utils import global_reuse


class ExpConfig(Config):
    # model parameters
    z_dim = 40
    x_dim = 784
    batch_norm = True
    dropout = False
    l2_reg = 0.0001
    shortcut_kernel_size = 1

    # training parameters
    max_epoch = 3000
    batch_size = 128
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 300
    lr_anneal_step_freq = None

    # evaluation parameters
    test_n_z = 500
    test_batch_size = 64


@global_reuse
@add_arg_scope
def h_for_q_z(x, is_training, channels_last):
    normalizer_fn = None if not config.batch_norm else functools.partial(
        batch_norm_2d,
        channels_last=channels_last,
        training=is_training,
    )
    dropout_fn = None if not config.dropout else functools.partial(
        tf.layers.dropout,
        training=is_training
    )

    with arg_scope([resnet_block],
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   dropout_fn=dropout_fn,
                   kernel_regularizer=l2_regularizer(config.l2_reg),
                   channels_last=channels_last):
        h_x = tf.to_float(x)
        h_x = tf.reshape(
            h_x, [-1, 28, 28, 1] if channels_last else [-1, 1, 28, 28])
        h_x = resnet_block(h_x, 16)  # output: (16, 28, 28)
        h_x = resnet_block(h_x, 32, strides=2)  # output: (32, 14, 14)
        h_x = resnet_block(h_x, 32)  # output: (32, 14, 14)
        h_x = resnet_block(h_x, 64, strides=2)  # output: (64, 7, 7)
        h_x = resnet_block(h_x, 64)  # output: (64, 7, 7)
    h_x = reshape_conv2d_to_flat(h_x)
    return {
        'mean': dense(h_x, config.z_dim, name='z_mean'),
        'logstd': dense(h_x, config.z_dim, name='z_logstd'),
    }


@global_reuse
@add_arg_scope
def h_for_p_x(z, is_training, channels_last):
    with arg_scope([deconv_resnet_block],
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=l2_regularizer(config.l2_reg),
                   channels_last=channels_last):
        h_z, s1, s2 = flatten(z, 2)
        h_z = tf.reshape(dense(h_z, 64 * 7 * 7),
                         [-1, 7, 7, 64] if channels_last else [-1, 64, 7, 7])
        h_z = deconv_resnet_block(h_z, 64)  # output: (64, 7, 7)
        h_z = deconv_resnet_block(h_z, 32, strides=2)  # output: (32, 14, 14)
        h_z = deconv_resnet_block(h_z, 32)  # output: (32, 14, 14)
        h_z = deconv_resnet_block(h_z, 16, strides=2)  # output: (16, 28, 28)
    h_z = conv2d(
        h_z, 1, (1, 1), padding='same', name='feature_map_to_pixel',
        channels_last=channels_last)  # output: (1, 28, 28)
    h_z = tf.reshape(h_z, [-1, config.x_dim])
    x_logits = unflatten(h_z, s1, s2)
    return {'logits': x_logits}


def sample_from_probs(x):
    uniform_samples = tf.random_uniform(
        shape=tf.shape(x), minval=0., maxval=1.,
        dtype=x.dtype
    )
    return tf.cast(tf.less(uniform_samples, x), dtype=tf.int32)


def main():
    # load mnist data
    (x_train, y_train), (x_test, y_test) = \
        load_mnist(shape=[784], dtype=np.float32, normalize=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None,) + x_train.shape[1:], name='input_x')
    is_training = tf.placeholder(
        dtype=tf.bool, shape=(), name='is_training')
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
    learning_rate_var = AnnealingDynamicValue(config.initial_lr,
                                              config.lr_anneal_factor)
    multi_gpu = MultiGPU(disable_prebuild=False)

    # build the model
    vae = VAE(
        p_z=Normal(mean=tf.zeros([1, config.z_dim]),
                   logstd=tf.zeros([1, config.z_dim])),
        p_x_given_z=Bernoulli,
        q_z_given_x=Normal,
        h_for_p_x=functools.partial(h_for_p_x, is_training=is_training),
        h_for_q_z=functools.partial(h_for_q_z, is_training=is_training),
    )

    grads = []
    losses = []
    lower_bounds = []
    test_nlls = []
    batch_size = get_batch_size(input_x)
    params = None
    optimizer = tf.train.AdamOptimizer(learning_rate)

    for dev, pre_build, [dev_input_x] in multi_gpu.data_parallel(
            batch_size, [input_x]):
        with tf.device(dev), multi_gpu.maybe_name_scope(dev):
            if pre_build:
                with arg_scope([h_for_q_z, h_for_p_x], channels_last=True):
                    _ = vae.chain(dev_input_x)

            else:
                with arg_scope([h_for_q_z, h_for_p_x],
                               channels_last=multi_gpu.channels_last(dev)):
                    # derive the loss and lower-bound for training
                    dev_vae_loss = vae.get_training_loss(dev_input_x)
                    dev_loss = dev_vae_loss + regularization_loss()
                    dev_lower_bound = -dev_vae_loss
                    losses.append(dev_loss)
                    lower_bounds.append(dev_lower_bound)

                    # derive the nll and logits output for testing
                    test_chain = vae.chain(dev_input_x, n_z=config.test_n_z)
                    dev_test_nll = -tf.reduce_mean(
                        test_chain.vi.evaluation.is_loglikelihood())
                    test_nlls.append(dev_test_nll)

                    # derive the optimizer
                    params = tf.trainable_variables()
                    grads.append(
                        optimizer.compute_gradients(dev_loss, var_list=params))

    # merge multi-gpu outputs and operations
    [loss, lower_bound, test_nll] = \
        multi_gpu.average([losses, lower_bounds, test_nlls], batch_size)
    train_op = multi_gpu.apply_grads(
        grads=multi_gpu.average_grads(grads),
        optimizer=optimizer,
        control_inputs=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    )

    # derive the plotting function
    work_dev = multi_gpu.work_devices[0]
    with tf.device(work_dev), tf.name_scope('plot_x'), \
            arg_scope([h_for_q_z, h_for_p_x],
                      channels_last=multi_gpu.channels_last(work_dev)):
        x_plots = tf.reshape(
            tf.cast(
                255 * tf.sigmoid(vae.model(n_z=100)['x'].distribution.logits),
                dtype=tf.uint8
            ),
            [-1, 28, 28]
        )

    def plot_samples(loop):
        with loop.timeit('plot_time'):
            images = session.run(x_plots, feed_dict={is_training: False})
            save_images_collection(
                images=images,
                filename=results.prepare_parent('plotting/{}.png'.
                                                format(loop.epoch)),
                grid_size=(10, 10)
            )

    # prepare for training and testing data
    def input_x_sampler(x):
        return session.run([sampled_x], feed_dict={sample_input_x: x})

    with tf.device('/device:CPU:0'):
        sample_input_x = tf.placeholder(
            dtype=tf.float32, shape=(None, config.x_dim), name='sample_input_x')
        sampled_x = sample_from_probs(sample_input_x)

    train_flow = DataFlow.arrays([x_train], config.batch_size, shuffle=True,
                                 skip_incomplete=True).map(input_x_sampler)
    test_flow = DataFlow.arrays([x_test], config.test_batch_size). \
        map(input_x_sampler)

    with create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        # fix the testing flow, reducing the testing time
        test_flow = test_flow.to_arrays_flow(batch_size=config.test_batch_size)

        # train the network
        with TrainLoop(params,
                       var_groups=['h_for_q_z', 'h_for_p_x'],
                       max_epoch=config.max_epoch,
                       summary_dir=results.make_dir('train_summary'),
                       summary_graph=tf.get_default_graph(),
                       summary_commit_freqs={'loss': 10},
                       early_stopping=False) as loop:
            trainer = Trainer(
                loop, train_op, [input_x], train_flow,
                feed_dict={learning_rate: learning_rate_var, is_training: True},
                metrics={'loss': loss}
            )
            anneal_after(
                trainer, learning_rate_var, epochs=config.lr_anneal_epoch_freq,
                steps=config.lr_anneal_step_freq
            )
            evaluator = Evaluator(
                loop,
                metrics={'test_nll': test_nll, 'test_lb': lower_bound},
                inputs=[input_x],
                data_flow=test_flow,
                feed_dict={is_training: False},
                time_metric_name='test_time'
            )
            evaluator.after_run.add_hook(
                lambda: results.commit(evaluator.last_metrics_dict))
            trainer.evaluate_after_epochs(evaluator, freq=10)
            trainer.evaluate_after_epochs(
                functools.partial(plot_samples, loop), freq=10)
            trainer.log_after_epochs(freq=1)
            trainer.run()

    # write the final test_nll and test_lb
    results.commit_and_print(evaluator.last_metrics_dict)


if __name__ == '__main__':
    config = ExpConfig()
    results = Results()
    main()
