# -*- coding: utf-8 -*-
import functools

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from tfsnippet.dataflow import DataFlow
from tfsnippet.examples.nn import (dense,
                                   softmax_classification_loss,
                                   softmax_classification_output,
                                   l2_regularizer,
                                   regularization_loss,
                                   classification_accuracy)
from tfsnippet.examples.utils import (load_mnist,
                                      create_session,
                                      Config,
                                      Results,
                                      anneal_after)
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import AnnealingDynamicValue, Trainer, Evaluator
from tfsnippet.utils import global_reuse


class ExpConfig(Config):
    # model parameters
    l2_reg = 0.0001

    # training parameters
    max_epoch = 500
    batch_size = 64

    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 50
    lr_anneal_step_freq = None


@global_reuse
def model(x, is_training):
    with arg_scope([dense],
                   activation_fn=tf.nn.leaky_relu,
                   kernel_regularizer=l2_regularizer(config.l2_reg)):
        h_x = x
        h_x = dense(h_x, 500)
        h_x = dense(h_x, 500)
    logits = dense(h_x, 10, name='logits')
    return logits


def main():
    # load mnist data
    (x_train, y_train), (x_test, y_test) = \
        load_mnist(shape=[784], dtype=np.float32, normalize=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + x_train.shape[1:], name='input_x')
    input_y = tf.placeholder(
        dtype=tf.int32, shape=[None], name='input_y')
    is_training = tf.placeholder(
        dtype=tf.bool, shape=(), name='is_training')
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
    learning_rate_var = AnnealingDynamicValue(config.initial_lr,
                                              config.lr_anneal_factor)

    # build the model
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # derive the loss, output and accuracy
    logits = model(input_x, is_training=is_training)
    softmax_loss = softmax_classification_loss(logits, input_y)
    loss = softmax_loss + regularization_loss()
    y = softmax_classification_output(logits)
    acc = classification_accuracy(y, input_y)

    # derive the optimizer
    params = tf.trainable_variables()
    grads = optimizer.compute_gradients(loss, var_list=params)
    with tf.control_dependencies(
            tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.apply_gradients(grads)

    # prepare for training and testing data
    train_flow = DataFlow.arrays(
        [x_train, y_train], config.batch_size, shuffle=True,
        skip_incomplete=True
    )
    test_flow = DataFlow.arrays([x_test, y_test], config.batch_size)

    with create_session().as_default():
        # train the network
        with TrainLoop(params,
                       max_epoch=config.max_epoch,
                       summary_dir=results.make_dir('train_summary'),
                       summary_graph=tf.get_default_graph(),
                       summary_commit_freqs={'loss': 10, 'acc': 10},
                       early_stopping=False) as loop:
            trainer = Trainer(
                loop, train_op, [input_x, input_y], train_flow,
                feed_dict={learning_rate: learning_rate_var, is_training: True},
                metrics={'loss': loss, 'acc': acc}
            )
            anneal_after(
                trainer, learning_rate_var, epochs=config.lr_anneal_epoch_freq,
                steps=config.lr_anneal_step_freq
            )
            evaluator = Evaluator(
                loop,
                metrics={'test_acc': acc},
                inputs=[input_x, input_y],
                data_flow=test_flow,
                feed_dict={is_training: False},
                time_metric_name='test_time'
            )
            evaluator.after_run.add_hook(
                lambda: results.commit(evaluator.last_metrics_dict))
            trainer.evaluate_after_epochs(evaluator, freq=5)
            trainer.log_after_epochs(freq=1)
            trainer.run()

        # save test result
        results.commit_and_print(evaluator.last_metrics_dict)


if __name__ == '__main__':
    config = ExpConfig()
    results = Results()
    main()
