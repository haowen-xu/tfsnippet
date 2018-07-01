# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from matplotlib import pyplot as plt

from tfsnippet.dataflow import DataFlow
from tfsnippet.distributions import Normal, Bernoulli
from tfsnippet.modules import Sequential, DictMapper, VAE
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import LossTrainer, Validator, AnnealingDynamicValue
from tfsnippet.utils import split_numpy_array


def sample_input_x(input_x, dtype=tf.int32):
    with tf.name_scope('sample_input_x'):
        uniform_samples = tf.random_uniform(
            shape=tf.shape(input_x), minval=0., maxval=1., dtype=input_x.dtype)
        return tf.cast(tf.less(uniform_samples, input_x), dtype=dtype)


def train(loss, input_x, x_data, max_epoch, batch_size, valid_portion):
    train_x, valid_x = split_numpy_array(x_data, portion=valid_portion)
    train_flow = DataFlow.arrays(
        [train_x], batch_size=batch_size, shuffle=True, skip_incomplete=True)
    valid_flow = DataFlow.arrays([valid_x], batch_size=batch_size)

    # derive the optimizer
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
    learning_rate_var = AnnealingDynamicValue(0.001, 0.99995)
    params = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, var_list=params)

    # run the training loop
    with TrainLoop(params, max_epoch=max_epoch, early_stopping=True) as loop:
        trainer = LossTrainer(loop, loss, train_op, [input_x], train_flow)
        trainer.anneal_after_steps(learning_rate_var, freq=1)
        trainer.validate_after_epochs(
            Validator(loop, loss, [input_x], valid_flow), freq=1)
        trainer.after_epochs.add_hook(
            lambda: trainer.loop.collect_metrics(lr=learning_rate_var),
            freq=1
        )
        trainer.log_after_epochs(freq=1)
        trainer.run(feed_dict={learning_rate: learning_rate_var})


def main():
    # load mnist data
    (train_x, train_y), (test_x, test_y) = K.datasets.mnist.load_data()

    # the parameters of this experiment
    x_dim = train_x.shape[1]
    z_dim = 2
    max_epoch = 10
    batch_size = 256
    valid_portion = 0.2

    # construct the graph
    with tf.Graph().as_default(), tf.Session().as_default() as session:
        input_x = tf.placeholder(
            dtype=tf.float32, shape=(None, x_dim), name='input_x')
        x_binarized = tf.stop_gradient(sample_input_x(input_x))
        batch_size_tensor = tf.shape(input_x)[0]

        # derive the VAE
        z_shape = tf.stack([batch_size_tensor, z_dim])
        vae = VAE(
            p_z=Normal(mean=tf.zeros(z_shape), std=tf.ones(z_shape)),
            p_x_given_z=Bernoulli,
            q_z_given_x=Normal,
            h_for_p_x=Sequential([
                K.layers.Dense(100, activation=tf.nn.relu),
                K.layers.Dense(100, activation=tf.nn.relu),
                DictMapper({
                    'logits': K.layers.Dense(x_dim, name='x_logits')
                })
            ]),
            h_for_q_z=Sequential([
                tf.to_float,
                K.layers.Dense(100, activation=tf.nn.relu),
                K.layers.Dense(100, activation=tf.nn.relu),
                DictMapper({
                    'mean': K.layers.Dense(z_dim, name='z_mean'),
                    'logstd': K.layers.Dense(z_dim, name='z_logstd'),
                })
            ])
        )

        # train the network
        train(vae.get_training_loss(x_binarized), input_x, train_x,
              max_epoch, batch_size, valid_portion)

        # plot the latent space
        q_net = vae.variational(x_binarized)
        z_posterior = q_net['z']
        z_predict = []

        for [batch_x] in DataFlow.arrays([test_x], batch_size=batch_size):
            z_predict.append(session.run(
                z_posterior,
                feed_dict={input_x: batch_x}
            ))

        z_predict = np.concatenate(z_predict, axis=0)
        plt.figure(figsize=(8, 6))
        plt.scatter(z_predict[:, 0], z_predict[:, 1], c=test_y)
        plt.colorbar()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    main()
