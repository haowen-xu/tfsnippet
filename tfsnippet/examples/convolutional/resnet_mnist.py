# -*- coding: utf-8 -*-
import functools

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.contrib.framework import arg_scope

from tfsnippet.dataflow import DataFlow
from tfsnippet.examples.nn import (resnet_block,
                                   global_average_pooling,
                                   softmax_classification_loss,
                                   softmax_classification_output,
                                   l2_regularizer,
                                   regularization_loss,
                                   classification_accuracy)
from tfsnippet.examples.utils import (load_mnist,
                                      create_session,
                                      Config,
                                      write_result,
                                      collect_outputs,
                                      anneal_after)
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import AnnealingDynamicValue, LossTrainer, Evaluator
from tfsnippet.utils import global_reuse


class ExpConfig(Config):
    max_epoch = 200
    batch_size = 64
    l2_reg = 0.0001

    # learning rate and scheduler
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None


config = ExpConfig()


@global_reuse
def model(x, is_training):
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
        x = resnet_block(x, 16)  # output: (16, 28, 28)
        x = resnet_block(x, 32, strides=2)  # output: (32, 14, 14)
        # x = resnet_block(x, 32)  # output: (32, 14, 14)
        x = resnet_block(x, 64, strides=2)  # output: (64, 7, 7)
        # x = resnet_block(x, 64)  # output: (64, 7, 7)
        x = global_average_pooling(x)  # output: (64, 1, 1)
    x = tf.layers.dense(tf.layers.flatten(x), 10, name='logits')
    return x


def main():
    # load mnist data
    (x_train, y_train), (x_test, y_test) = \
        load_mnist(shape=[1, 28, 28], dtype=np.float32, normalize=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + x_train.shape[1:], name='input_x')
    input_y = tf.placeholder(
        dtype=tf.int32, shape=[None], name='input_y')
    is_training = tf.placeholder(
        dtype=tf.bool, shape=(), name='is_training')

    # build the model
    logits = model(input_x, is_training)
    loss = (softmax_classification_loss(logits, input_y) +
            regularization_loss())
    y = softmax_classification_output(logits)
    acc = classification_accuracy(y, input_y)

    # derive the optimizer
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
    learning_rate_var = AnnealingDynamicValue(config.initial_lr,
                                              config.lr_anneal_factor)
    params = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.control_dependencies(
            tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, var_list=params)

    # prepare for training and testing data
    train_flow = DataFlow.arrays(
        [x_train, y_train], config.batch_size, shuffle=True,
        skip_incomplete=True
    )
    eval_flow = DataFlow.arrays([x_train, y_train], config.batch_size)
    test_eval_flow = DataFlow.arrays([x_test, y_test], config.batch_size)
    test_flow = DataFlow.arrays([x_test], config.batch_size)

    with create_session().as_default():
        # train the network
        with TrainLoop(params,
                       max_epoch=config.max_epoch,
                       summary_dir='train_summary',
                       early_stopping=False) as loop:
            trainer = LossTrainer(
                loop, loss, train_op, [input_x, input_y], train_flow,
                feed_dict={learning_rate: learning_rate_var, is_training: True}
            )
            anneal_after(
                trainer, learning_rate_var, epochs=config.lr_anneal_epoch_freq,
                steps=config.lr_anneal_step_freq
            )
            trainer.evaluate_after_epochs(
                Evaluator(
                    loop,
                    metrics={'train_acc': acc},
                    inputs=[input_x, input_y],
                    data_flow=eval_flow,
                    feed_dict={is_training: False},
                    time_metric_name='eval_time'
                ),
                freq=1
            )
            trainer.evaluate_after_epochs(
                Evaluator(
                    loop,
                    metrics={'test_acc': acc},
                    inputs=[input_x, input_y],
                    data_flow=test_eval_flow,
                    feed_dict={is_training: False},
                    time_metric_name='test_time'
                ),
                freq=1
            )
            trainer.log_after_epochs(freq=1)
            trainer.run()

        # do final evaluation
        [y_pred] = collect_outputs(
            [y], [input_x], test_flow, feed_dict={is_training: False})

        # save test result
        pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred},
                     columns=['y_test', 'y_pred']). \
            to_csv('test.csv', index=False)
        write_result({
            'test_acc': accuracy_score(y_test, y_pred)
        })


if __name__ == '__main__':
    main()
