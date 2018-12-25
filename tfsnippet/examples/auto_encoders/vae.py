# -*- coding: utf-8 -*-
import functools

import click
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope, add_arg_scope

from tfsnippet.bayes import BayesianNet
from tfsnippet.distributions import Normal, Bernoulli
from tfsnippet.layers import dense, act_norm
from tfsnippet.examples.datasets import load_mnist, bernoulli_flow
from tfsnippet.examples.nn import (l2_regularizer,
                                   regularization_loss)
from tfsnippet.examples.utils import (MLConfig,
                                      MLResults,
                                      save_images_collection,
                                      config_options,
                                      global_config as config,
                                      bernoulli_as_pixel,
                                      print_with_title)
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import AnnealingDynamicValue, Trainer, Evaluator
from tfsnippet.utils import (global_reuse, create_session,
                             ensure_variables_initialized)


class ExpConfig(MLConfig):
    # model parameters
    z_dim = 40
    x_dim = 784

    # training parameters
    write_summary = False
    max_epoch = 3000
    max_step = None
    batch_size = 128
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 300
    lr_anneal_step_freq = None

    # evaluation parameters
    test_n_z = 500
    test_batch_size = 128


@global_reuse
@add_arg_scope
def q_net(x, observed=None, n_z=None, is_training=False, is_initializing=False):
    net = BayesianNet(observed=observed)
    normalizer_fn = functools.partial(act_norm, initializing=is_initializing)

    # compute the hidden features
    with arg_scope([dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=True,
                   kernel_regularizer=l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = dense(h_x, 500)
        h_x = dense(h_x, 500)

    # sample z ~ q(z|x)
    z_mean = dense(h_x, config.z_dim, name='z_mean')
    z_logstd = dense(h_x, config.z_dim, name='z_logstd')
    z = net.add('z', Normal(mean=z_mean, logstd=z_logstd), n_samples=n_z,
                group_ndims=1)

    return net


@global_reuse
@add_arg_scope
def p_net(observed=None, n_z=None, is_training=False, is_initializing=False):
    net = BayesianNet(observed=observed)
    normalizer_fn = functools.partial(act_norm, initializing=is_initializing)

    # sample z ~ p(z)
    z = net.add('z', Normal(mean=tf.zeros([1, config.z_dim]),
                            logstd=tf.zeros([1, config.z_dim])),
                group_ndims=1, n_samples=n_z)

    # compute the hidden features
    with arg_scope([dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=True,
                   kernel_regularizer=l2_regularizer(config.l2_reg)):
        h_z = z
        h_z = dense(h_z, 500)
        h_z = dense(h_z, 500)

    # sample x ~ p(x|z)
    x_logits = dense(h_z, config.x_dim, name='x_logits')
    x = net.add('x', Bernoulli(logits=x_logits), group_ndims=1)

    return net


@click.command()
@click.option('--result-dir', help='The result directory.', metavar='PATH',
              required=False, type=str)
@config_options(ExpConfig)
def main(result_dir):
    # print the config
    print_with_title('Configurations', config.format_config(), after='\n')

    # open the result object and prepare for result directories
    results = MLResults(result_dir)
    results.make_dirs('plotting', exist_ok=True)
    results.make_dirs('train_summary', exist_ok=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None, config.x_dim), name='input_x')
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
    learning_rate_var = AnnealingDynamicValue(config.initial_lr,
                                              config.lr_anneal_factor)

    # derive the output for initialization
    with tf.name_scope('initialization'):
        init_q_net = q_net(input_x, is_initializing=True)
        init_chain = init_q_net.chain(
            p_net, latent_names=['z'], latent_axis=0, observed={'x': input_x},
            is_initializing=True
        )
        init_lb = tf.reduce_mean(init_chain.vi.lower_bound.elbo())

    # derive the loss and lower-bound for training
    with tf.name_scope('training'):
        train_q_net = q_net(input_x, is_training=True)
        train_chain = train_q_net.chain(
            p_net, latent_names=['z'], latent_axis=0, observed={'x': input_x},
            is_training=True
        )
        vae_loss = tf.reduce_mean(train_chain.vi.training.sgvb())
        loss = vae_loss + regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, n_z=config.test_n_z)
        test_chain = test_q_net.chain(
            p_net, latent_names=['z'], latent_axis=0, observed={'x': input_x})
        test_nll = -tf.reduce_mean(test_chain.vi.evaluation.is_loglikelihood())
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())

    # derive the optimizer
    with tf.name_scope('optimizing'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        grads = optimizer.compute_gradients(loss, var_list=params)
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.apply_gradients(grads)

    # derive the plotting function
    with tf.name_scope('plotting'):
        plot_p_net = p_net(n_z=100)
        x_plots = tf.reshape(bernoulli_as_pixel(plot_p_net['x']), (-1, 28, 28))

    def plot_samples(loop):
        with loop.timeit('plot_time'):
            images = session.run(x_plots)
            save_images_collection(
                images=images,
                filename='plotting/{}.png'.format(loop.epoch),
                grid_size=(10, 10),
                results=results
            )

    # prepare for training and testing data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    train_flow = bernoulli_flow(
        x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    test_flow = bernoulli_flow(
        x_test, config.test_batch_size, sample_now=True)

    with create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        ensure_variables_initialized()

        # initialize the network
        for [x] in train_flow:
            lb_out = session.run(init_lb, feed_dict={input_x: x})
            print('Network initialized, first-batch loss is {:.2g}.\n'.
                  format(lb_out))
            break

        # train the network
        with TrainLoop(params,
                       var_groups=['q_net', 'p_net'],
                       max_epoch=config.max_epoch,
                       max_step=config.max_step,
                       summary_dir=(results.system_path('train_summary')
                                    if config.write_summary else None),
                       summary_graph=tf.get_default_graph(),
                       early_stopping=False) as loop:
            trainer = Trainer(
                loop, train_op, [input_x], train_flow,
                feed_dict={learning_rate: learning_rate_var},
                metrics={'loss': loss}
            )
            trainer.anneal_after(
                learning_rate_var,
                epochs=config.lr_anneal_epoch_freq,
                steps=config.lr_anneal_step_freq
            )
            evaluator = Evaluator(
                loop,
                metrics={'test_nll': test_nll, 'test_lb': test_lb},
                inputs=[input_x],
                data_flow=test_flow,
                time_metric_name='test_time'
            )
            evaluator.after_run.add_hook(
                lambda: results.update_metrics(evaluator.last_metrics_dict))
            trainer.evaluate_after_epochs(evaluator, freq=10)
            trainer.evaluate_after_epochs(
                functools.partial(plot_samples, loop), freq=10)
            trainer.log_after_epochs(freq=1)
            trainer.run()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
