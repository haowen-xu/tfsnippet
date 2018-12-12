# -*- coding: utf-8 -*-
import codecs
import functools
import logging
import warnings

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.contrib.framework import arg_scope, add_arg_scope

from tfsnippet.bayes import BayesianNet
from tfsnippet.dataflow import DataFlow
from tfsnippet.distributions import Normal, Bernoulli, Categorical
from tfsnippet.examples.nn import (l2_regularizer,
                                   regularization_loss,
                                   dense)
from tfsnippet.examples.utils import (load_mnist,
                                      create_session,
                                      Config,
                                      anneal_after,
                                      save_images_collection,
                                      Results,
                                      MultiGPU,
                                      collect_outputs,
                                      ClusteringClassifier)
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import AnnealingDynamicValue, Trainer, Evaluator
from tfsnippet.utils import global_reuse, get_batch_size, flatten, unflatten


class ExpConfig(Config):
    # model parameters
    x_dim = 784
    z_dim = 16
    n_clusters = 16
    l2_reg = 0.0001
    p_z_given_y_std = 'unbound_logstd'
    # {'one', 'one_plus_softplus_std', 'softplus_logstd', 'unbound_logstd'}
    mean_field_assumption_for_q = False

    # training parameters
    max_epoch = 3000
    batch_size = 128
    train_n_samples = 25  # use "reinforce" if None, otherwise "vimco"

    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 300
    lr_anneal_step_freq = None

    # evaluation parameters
    test_n_samples = 500
    test_batch_size = 128


@global_reuse
def gaussian_mixture_prior(y, z_dim, n_clusters):
    # derive the learnt z_mean
    prior_mean = tf.get_variable(
        'z_prior_mean', dtype=tf.float32, shape=[n_clusters, z_dim],
        initializer=tf.random_normal_initializer()
    )
    z_mean = tf.nn.embedding_lookup(prior_mean, y)

    # derive the learnt z_std
    z_logstd = z_std = None
    if config.p_z_given_y_std == 'one':
        z_logstd = tf.zeros_like(z_mean)
    else:
        prior_std_or_logstd = tf.get_variable(
            'z_prior_std_or_logstd',
            dtype=tf.float32,
            shape=[n_clusters, z_dim],
            initializer=tf.zeros_initializer()
        )
        z_std_or_logstd = tf.nn.embedding_lookup(prior_std_or_logstd, y)

        if config.p_z_given_y_std == 'one_plus_softplus_std':
            z_std = 1. + tf.nn.softplus(z_std_or_logstd)
        elif config.p_z_given_y_std == 'softplus_logstd':
            z_logstd = tf.nn.softplus(z_std_or_logstd)
        elif config.p_z_given_y_std == 'unbound_logstd':
            z_logstd = z_std_or_logstd
        else:
            raise ValueError(
                'Unexpected value for config `p_z_given_y_std`: {}'.
                format(config.p_z_given_y_std)
            )

    return Normal(mean=z_mean, std=z_std, logstd=z_logstd)


@global_reuse
@add_arg_scope
def q_net(x, observed=None, n_samples=None, is_training=True):
    logging.info('q_net builder: %r', locals())

    net = BayesianNet(observed=observed)

    # compute the hidden features
    with arg_scope([dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = dense(h_x, 500)
        h_x = dense(h_x, 500)

    # sample y ~ q(y|x)
    y_logits = dense(h_x, config.n_clusters, name='y_logits')
    y = net.add('y', Categorical(y_logits), n_samples=n_samples)
    y_one_hot = tf.one_hot(y, config.n_clusters, dtype=tf.float32)

    # sample z ~ q(z|y,x)
    with arg_scope([dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=l2_regularizer(config.l2_reg)):
        if config.mean_field_assumption_for_q:
            # by mean-field-assumption we let q(z|y,x) = q(z|x)
            h_z, s1, s2 = flatten(h_x, 2)
            z_n_samples = n_samples
        else:
            if n_samples is not None:
                h_z = tf.concat(
                    [
                        tf.tile(tf.reshape(h_x, [1, -1, 500]),
                                tf.stack([n_samples, 1, 1])),
                        y_one_hot
                    ],
                    axis=-1
                )
            else:
                h_z = tf.concat([h_x, y_one_hot], axis=-1)
            h_z, s1, s2 = flatten(h_z, 2)
            h_z = dense(h_z, 500)
            z_n_samples = None

    z_mean = dense(h_z, config.z_dim, name='z_mean')
    z_logstd = dense(h_z, config.z_dim, name='z_logstd')
    z = net.add('z',
                Normal(mean=unflatten(z_mean, s1, s2),
                       logstd=unflatten(z_logstd, s1, s2),
                       is_reparameterized=False),
                n_samples=z_n_samples, group_ndims=1)

    return net


@global_reuse
@add_arg_scope
def p_net(observed=None, n_y=None, n_z=None, is_training=True, n_samples=None):
    if n_samples is not None:
        warnings.warn('`n_samples` is deprecated, use `n_y` instead.')
        n_y = n_samples

    logging.info('p_net builder: %r', locals())

    net = BayesianNet(observed=observed)

    # sample y
    y = net.add('y',
                Categorical(tf.zeros([1, config.n_clusters])),
                n_samples=n_y)

    # sample z ~ p(z|y)
    z = net.add('z',
                gaussian_mixture_prior(y, config.z_dim, config.n_clusters),
                group_ndims=1,
                n_samples=n_z,
                is_reparameterized=False)

    # compute the hidden features for x
    with arg_scope([dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=l2_regularizer(config.l2_reg)):
        h_x, s1, s2 = flatten(z, 2)
        h_x = dense(h_x, 500)
        h_x = dense(h_x, 500)

    # sample x ~ p(x|z)
    x_logits = unflatten(dense(h_x, config.x_dim, name='x_logits'), s1, s2)
    x = net.add('x', Bernoulli(logits=x_logits), group_ndims=1)

    return net


@global_reuse
def reinforce_baseline_net(x):
    x, s1, s2 = flatten(tf.to_float(x), 2)
    with arg_scope([dense],
                   kernel_regularizer=l2_regularizer(config.l2_reg),
                   activation_fn=tf.nn.leaky_relu):
        h_x = dense(x, 500)
    h_x = unflatten(tf.reshape(dense(h_x, 1), [-1]), s1, s2)
    return h_x


def sample_from_probs(x):
    uniform_samples = tf.random_uniform(
        shape=tf.shape(x), minval=0., maxval=1.,
        dtype=x.dtype
    )
    return tf.cast(tf.less(uniform_samples, x), dtype=tf.int32)


def main():
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # load mnist data
    (x_train, y_train), (x_test, y_test) = \
        load_mnist(shape=[config.x_dim], dtype=np.float32, normalize=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None,) + x_train.shape[1:], name='input_x')
    is_training = tf.placeholder(
        dtype=tf.bool, shape=(), name='is_training')
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32,
                                   name='learning_rate')
    learning_rate_var = AnnealingDynamicValue(config.initial_lr,
                                              config.lr_anneal_factor)
    multi_gpu = MultiGPU(disable_prebuild=False)

    # build the model
    grads = []
    losses = []
    test_nlls = []
    y_given_x_list = []
    batch_size = get_batch_size(input_x)
    params = None
    optimizer = tf.train.AdamOptimizer(learning_rate)

    for dev, pre_build, [dev_input_x] in multi_gpu.data_parallel(
            batch_size, [input_x]):
        with tf.device(dev), multi_gpu.maybe_name_scope(dev):
            if pre_build:
                with arg_scope([q_net, p_net], is_training=is_training):
                    _ = q_net(dev_input_x).chain(
                        p_net,
                        latent_names=['y', 'z'],
                        observed={'x': dev_input_x}
                    )

            else:
                with arg_scope([q_net, p_net], is_training=is_training):
                    # derive the loss and lower-bound for training
                    train_q_net = q_net(
                        dev_input_x, n_samples=config.train_n_samples
                    )
                    train_chain = train_q_net.chain(
                        p_net, latent_names=['y', 'z'], latent_axis=0,
                        observed={'x': dev_input_x}
                    )

                    if config.train_n_samples is None:
                        dev_baseline = reinforce_baseline_net(dev_input_x)
                        dev_vae_loss = tf.reduce_mean(
                            train_chain.vi.training.reinforce(
                                baseline=dev_baseline
                            )
                        )
                    else:
                        dev_vae_loss = tf.reduce_mean(
                            train_chain.vi.training.vimco())
                    dev_loss = dev_vae_loss + regularization_loss()
                    losses.append(dev_loss)

                    # derive the nll and logits output for testing
                    test_q_net = q_net(
                        dev_input_x, n_samples=config.test_n_samples
                    )
                    test_chain = test_q_net.chain(
                        p_net, latent_names=['y', 'z'], latent_axis=0,
                        observed={'x': dev_input_x}
                    )
                    dev_test_nll = -tf.reduce_mean(
                        test_chain.vi.evaluation.is_loglikelihood())
                    test_nlls.append(dev_test_nll)

                    # derive the classifier via q(y|x)
                    dev_q_y_given_x = tf.argmax(
                        test_q_net['y'].distribution.logits, axis=-1)
                    y_given_x_list.append(dev_q_y_given_x)

                    # derive the optimizer
                    params = tf.trainable_variables()
                    grads.append(
                        optimizer.compute_gradients(dev_loss, var_list=params))

    # merge multi-gpu outputs and operations
    [loss, test_nll] = \
        multi_gpu.average([losses, test_nlls], batch_size)
    [y_given_x] = multi_gpu.concat([y_given_x_list])

    train_op = multi_gpu.apply_grads(
        grads=multi_gpu.average_grads(grads),
        optimizer=optimizer,
        control_inputs=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    )

    # derive the plotting function
    work_dev = multi_gpu.work_devices[0]
    with tf.device(work_dev), tf.name_scope('plot_x'):
        plot_p_net = p_net(
            observed={'y': tf.range(config.n_clusters, dtype=tf.int32)},
            n_z=10,
            is_training=is_training
        )
        x = tf.cast(
            255 * tf.sigmoid(plot_p_net['x'].distribution.logits),
            dtype=tf.uint8
        )
        x_plots = tf.reshape(tf.transpose(x, [1, 0, 2]), [-1, 28, 28])

    def plot_samples(loop):
        with loop.timeit('plot_time'):
            images = session.run(x_plots, feed_dict={is_training: False})
            save_images_collection(
                images=images,
                filename=results.prepare_parent('plotting/{}.png'.
                                                format(loop.epoch)),
                grid_size=(config.n_clusters, 10)
            )

    # derive the final un-supervised classifier
    c_classifier = ClusteringClassifier(config.n_clusters, 10)

    def train_classifier(loop):
        df = DataFlow.arrays([x_train], batch_size=config.batch_size). \
            map(input_x_sampler)
        with loop.timeit('cls_train_time'):
            [c_pred] = collect_outputs(
                outputs=[y_given_x],
                inputs=[input_x],
                data_flow=df,
                feed_dict={is_training: False}
            )
            c_classifier.fit(c_pred, y_train)
            print(c_classifier.describe())

    def evaluate_classifier(loop):
        with loop.timeit('cls_test_time'):
            [c_pred] = collect_outputs(
                outputs=[y_given_x],
                inputs=[input_x],
                data_flow=test_flow,
                feed_dict={is_training: False}
            )
            y_pred = c_classifier.predict(c_pred)
            cls_metrics = {'test_acc': accuracy_score(y_test, y_pred)}
            loop.collect_metrics(cls_metrics)
            results.commit(cls_metrics)

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
                       var_groups=['p_net', 'q_net', 'gaussian_mixture_prior'],
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
                metrics={'test_nll': test_nll},
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
            trainer.evaluate_after_epochs(
                functools.partial(train_classifier, loop), freq=10)
            trainer.evaluate_after_epochs(
                functools.partial(evaluate_classifier, loop), freq=10)

            trainer.log_after_epochs(freq=1)
            trainer.run()

    # write the final results
    with codecs.open('cluster_classifier.txt', 'wb', 'utf-8') as f:
        f.write(c_classifier.describe())


if __name__ == '__main__':
    config = ExpConfig()
    results = Results()
    main()
