# -*- coding: utf-8 -*-
import functools

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from sklearn.metrics import accuracy_score

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
from tfsnippet.utils import global_reuse, split_numpy_arrays


class ExpConfig(Config):
    max_epoch = 500
    batch_size = 64
    valid_portion = 0.4
    l2_reg = 0.001

    # learning rate and scheduler
    initial_lr = 0.01
    lr_anneal_factor = 0.75
    lr_anneal_epoch_freq = 10
    lr_anneal_step_freq = None


config = ExpConfig()
config.from_env()


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
    return tf.layers.dense(tf.layers.flatten(x), 10, name='logits')

    # x = tf.layers.flatten(x)
    # x = tf.layers.dense(x, 500, activation=tf.nn.relu,
    #                     kernel_regularizer=l2_regularizer(config.l2_reg))
    # x = tf.layers.dense(x, 500, activation=tf.nn.relu,
    #                     kernel_regularizer=l2_regularizer(config.l2_reg))
    # return tf.layers.dense(x, 10, name='logits')


def train(loss, acc, is_training, input_x, input_y, x_data, y_data):
    (x_train, y_train), (x_valid, y_valid) = \
        split_numpy_arrays([x_data, y_data], portion=config.valid_portion)
    train_flow = DataFlow.arrays(
        [x_train, y_train], batch_size=config.batch_size, shuffle=True,
        skip_incomplete=True
    )
    valid_flow = DataFlow.arrays(
        [x_valid, y_valid], batch_size=config.batch_size)

    # derive the optimizer
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
    learning_rate_var = AnnealingDynamicValue(config.initial_lr,
                                              config.lr_anneal_factor)
    params = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, var_list=params)

    # run the training loop
    with TrainLoop(params,
                   max_epoch=config.max_epoch,
                   early_stopping=True) as loop:
        trainer = LossTrainer(
            loop, loss, train_op, [input_x, input_y], train_flow,
            feed_dict={learning_rate: learning_rate_var, is_training: True}
        )
        anneal_after(
            trainer, learning_rate_var, epochs=config.lr_anneal_epoch_freq,
            steps=config.lr_anneal_step_freq
        )
        trainer.validate_after_epochs(
            Evaluator(
                loop,
                metrics={'valid_loss': loss, 'valid_acc': acc},
                inputs=[input_x, input_y],
                data_flow=valid_flow,
                feed_dict={is_training: False},
                time_metric_name='valid_time'
            ),
            freq=1
        )
        trainer.after_epochs.add_hook(
            lambda: trainer.loop.collect_metrics(lr=learning_rate_var),
            freq=1
        )
        trainer.log_after_epochs(freq=1)
        trainer.run()


def main():
    # load mnist data
    (x_train, y_train), (x_test, y_test) = \
        load_mnist(shape=[1, 28, 28], dtype=np.float32, normalize=True)

    # construct the graph
    with tf.Graph().as_default(), create_session() as session:
        input_x = tf.placeholder(
            dtype=tf.float32, shape=(None,) + x_train.shape[1:], name='input_x')
        input_y = tf.placeholder(
            dtype=tf.int32, shape=[None], name='input_y')
        is_training = tf.placeholder(
            dtype=tf.bool, shape=(), name='is_training')

        # derive the classifier
        logits = model(input_x, is_training)
        loss = (softmax_classification_loss(logits, input_y) +
                regularization_loss())
        y = softmax_classification_output(logits)
        acc = classification_accuracy(y, input_y)

        # train the network
        train(loss, acc, is_training, input_x, input_y, x_train, y_train)

        # do the test
        [y_pred] = collect_outputs(
            [y], [input_x], DataFlow.arrays([x_test], config.batch_size),
            feed_dict={is_training: False}, session=session
        )
        print(y_test)
        print(y_pred)
        test_acc = accuracy_score(y_test, y_pred)
        write_result({
            'test_acc': test_acc
        })


if __name__ == '__main__':
    main()
