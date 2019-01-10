# -*- coding: utf-8 -*-
import functools

import click
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

import tfsnippet as sn
from tfsnippet.examples.utils import (MLConfig,
                                      MLResults,
                                      MultiGPU,
                                      global_config as config,
                                      config_options,
                                      print_with_title)


class ExpConfig(MLConfig):
    # model parameters
    x_dim = 784
    l2_reg = 0.0001
    kernel_size = 3

    # training parameters
    write_summary = False
    max_epoch = 1000
    max_step = None
    batch_size = 64
    test_batch_size = 256
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 100
    lr_anneal_step_freq = None


@sn.global_reuse
def model(x, is_training, channels_last):
    with arg_scope([sn.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=functools.partial(
                       tf.layers.batch_normalization,
                       axis=-1 if channels_last else -3,
                       training=is_training,
                   ),
                   dropout_fn=functools.partial(
                       tf.layers.dropout,
                       training=is_training
                   ),
                   kernel_regularizer=sn.layers.l2_regularizer(config.l2_reg),
                   channels_last=channels_last):
        h_x = tf.reshape(
            x, [-1, 28, 28, 1] if channels_last else [-1, 1, 28, 28])
        h_x = sn.layers.resnet_conv2d_block(h_x, 16)  # output: (16, 28, 28)
        h_x = sn.layers.resnet_conv2d_block(h_x, 32, strides=2)  # output: (32, 14, 14)
        h_x = sn.layers.resnet_conv2d_block(h_x, 32)  # output: (32, 14, 14)
        h_x = sn.layers.resnet_conv2d_block(h_x, 64, strides=2)  # output: (64, 7, 7)
        h_x = sn.layers.resnet_conv2d_block(h_x, 64)  # output: (64, 7, 7)
        h_x = sn.layers.global_avg_pool2d(
            h_x, channels_last=channels_last)  # output: (64,)
    logits = sn.layers.dense(h_x, 10, name='logits')
    return logits


@click.command()
@click.option('--result-dir', help='The result directory.', metavar='PATH',
              required=False, type=str)
@config_options(ExpConfig)
def main(result_dir):
    # print the config
    print_with_title('Configurations', config.format_config(), after='\n')

    # open the result object and prepare for result directories
    results = MLResults(result_dir)
    results.make_dirs('train_summary', exist_ok=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None, config.x_dim), name='input_x')
    input_y = tf.placeholder(
        dtype=tf.int32, shape=[None], name='input_y')
    is_training = tf.placeholder(
        dtype=tf.bool, shape=(), name='is_training')
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
    learning_rate_var = sn.AnnealingDynamicValue(config.initial_lr,
                                                 config.lr_anneal_factor)
    multi_gpu = MultiGPU()

    # build the model
    grads = []
    losses = []
    y_list = []
    acc_list = []
    batch_size = sn.utils.get_batch_size(input_x)
    params = None
    optimizer = tf.train.AdamOptimizer(learning_rate)

    for dev, pre_build, [dev_input_x, dev_input_y] in multi_gpu.data_parallel(
            batch_size, [input_x, input_y]):
        with tf.device(dev), multi_gpu.maybe_name_scope(dev):
            if pre_build:
                _ = model(dev_input_x, is_training, channels_last=True)

            else:
                # derive the loss, output and accuracy
                dev_logits = model(
                    dev_input_x,
                    is_training=is_training,
                    channels_last=multi_gpu.channels_last(dev)
                )
                dev_cls_loss = tf.losses.sparse_softmax_cross_entropy(
                    dev_input_y, dev_logits
                )
                dev_loss = dev_cls_loss + tf.losses.get_regularization_loss()
                dev_y = sn.ops.softmax_classification_output(dev_logits)
                dev_acc = sn.ops.classification_accuracy(dev_y, dev_input_y)
                losses.append(dev_loss)
                y_list.append(dev_y)
                acc_list.append(dev_acc)

                # derive the optimizer
                params = tf.trainable_variables()
                grads.append(
                    optimizer.compute_gradients(dev_loss, var_list=params))

    # merge multi-gpu outputs and operations
    [loss, acc] = multi_gpu.average([losses, acc_list], batch_size)
    [y] = multi_gpu.concat([y_list])
    train_op = multi_gpu.apply_grads(
        grads=multi_gpu.average_grads(grads),
        optimizer=optimizer,
        control_inputs=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    )

    # prepare for training and testing data
    (x_train, y_train), (x_test, y_test) = \
        sn.datasets.load_mnist(normalize_x=True)
    train_flow = sn.DataFlow.arrays([x_train, y_train], config.batch_size,
                                    shuffle=True, skip_incomplete=True)
    test_flow = sn.DataFlow.arrays([x_test, y_test], config.test_batch_size)

    with sn.utils.create_session().as_default(), \
            train_flow.threaded(5) as train_flow:
        # train the network
        with sn.TrainLoop(params,
                          max_epoch=config.max_epoch,
                          max_step=config.max_step,
                          summary_dir=(results.system_path('train_summary')
                                       if config.write_summary else None),
                          summary_graph=tf.get_default_graph(),
                          early_stopping=False) as loop:
            trainer = sn.Trainer(
                loop, train_op, [input_x, input_y], train_flow,
                feed_dict={learning_rate: learning_rate_var, is_training: True},
                metrics={'loss': loss, 'acc': acc}
            )
            trainer.anneal_after(
                learning_rate_var,
                epochs=config.lr_anneal_epoch_freq,
                steps=config.lr_anneal_step_freq
            )
            evaluator = sn.Evaluator(
                loop,
                metrics={'test_acc': acc},
                inputs=[input_x, input_y],
                data_flow=test_flow,
                feed_dict={is_training: False},
                time_metric_name='test_time'
            )
            evaluator.after_run.add_hook(
                lambda: results.update_metrics(evaluator.last_metrics_dict))
            trainer.evaluate_after_epochs(evaluator, freq=5)
            trainer.log_after_epochs(freq=1)
            trainer.run()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
