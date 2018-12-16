TFSnippet
=========

+---------+-----------------+-----------------+---------------+
| Master  | |master_build|  | |master_cover|  | |master_doc|  |
+---------+-----------------+-----------------+---------------+
| Develop | |develop_build| | |develop_cover| |               |
+---------+-----------------+-----------------+---------------+

.. |master_build| image:: https://travis-ci.org/haowen-xu/tfsnippet.svg?branch=master
    :target: https://travis-ci.org/haowen-xu/tfsnippet
.. |master_cover| image:: https://coveralls.io/repos/github/haowen-xu/tfsnippet/badge.svg?branch=master
    :target: https://coveralls.io/github/haowen-xu/tfsnippet?branch=master
.. |master_doc| image:: https://readthedocs.org/projects/tfsnippet/badge/?version=latest
    :target: http://tfsnippet.readthedocs.io/en/latest/?badge=latest
.. |develop_build| image:: https://travis-ci.org/haowen-xu/tfsnippet.svg?branch=develop
    :target: https://travis-ci.org/haowen-xu/tfsnippet
.. |develop_cover| image:: https://coveralls.io/repos/github/haowen-xu/tfsnippet/badge.svg?branch=develop
    :target: https://coveralls.io/github/haowen-xu/tfsnippet?branch=develop

TFSnippet is a set of utilities for writing and testing TensorFlow models.

The design philosophy of TFSnippet is `non-interfering`.  It aims to provide a
set of useful utilities, possible to be used along with any other TensorFlow
libraries and frameworks.

Dependencies
------------

TensorFlow >= 1.5

Installation
------------

.. code-block:: bash

    pip install git+https://github.com/thu-ml/zhusuan.git
    pip install git+https://github.com/haowen-xu/tfsnippet.git

Documentation
-------------

* `Tutorials and API docs <http://tfsnippet.readthedocs.io/>`_

Examples
--------

* Classification:
    `MNIST <tfsnippet/examples/classification/mnist.py>`_,
    `Convolutional MNIST <tfsnippet/examples/classification/mnist_conv.py>`_.
* Auto Encoders:
    `VAE <tfsnippet/examples/auto_encoders/vae.py>`_,
    `Convolutional VAE <tfsnippet/examples/auto_encoders/vae_conv.py>`_,
    `Bernoulli Latent VAE <tfsnippet/examples/auto_encoders/bernoulli_latent_vae.py>`_,
    `Gaussian Mixture VAE <tfsnippet/examples/auto_encoders/gm_vae.py>`_.
* Normalizing Flows:
    `Planar Normalizing Flow <tfsnippet/examples/normalizing_flows/planar_nf.py>`_.

Quick Tutorial
--------------

Distributions
~~~~~~~~~~~~~

If you use ``tfsnippet.distributions`` to obtain random samples, you
shall get enhanced tensor objects, from which you may compute the
log-likelihood by simply calling ``log_prob()``.

.. code-block:: python

    from tfsnippet.distributions import Normal

    normal = Normal(0., 1.)
    # The type of `samples` is :class:`tfsnippet.stochastic.StochasticTensor`.
    samples = normal.sample(n_samples=100)
    # You may obtain the log-likelhood of `samples` under `normal` by:
    log_prob = samples.log_prob()
    # You may also obtain the distribution instance back from the samples,
    # such that you may fire-and-forget the distribution instance!
    distribution = samples.distribution

The distributions from `ZhuSuan <https://github.com/thu-ml/zhusuan.git>`_ can
be casted into a ``tfsnippet.distributions.Distribution``, in case we
haven't provided a wrapper for a certain ZhuSuan distribution:

.. code-block:: python

    from tfsnippet.distributions import as_distribution

    uniform = as_distribution(zhusuan.distributions.Uniform())
    # The type of `samples` is :class:`tfsnippet.stochastic.StochasticTensor`.
    samples = uniform.sample(n_samples=100)

Data Flows
~~~~~~~~~~

It is a common practice to iterate through a dataset by mini-batches.
The ``tfsnippet.dataflow`` provides a unified interface for assembling
the mini-batch iterators.

.. code-block:: python

    from tfsnippet.dataflow import DataFlow

    # Obtain a shuffled, two-array data flow, with batch-size 64.
    # Any batch with samples fewer than 64 would be discarded.
    flow = DataFlow.arrays(
        [x, y], batch_size=64, shuffle=True, skip_incomplete=True)
    for batch_x, batch_y in flow:
        ...  # Do something with batch_x and batch_y

    # You may use a threaded data flow to prefetch the mini-batches
    # in a background thread.  The threaded flow is a context object,
    # where exiting the context would destroy the background thread.
    with flow.threaded(prefetch=5) as threaded_flow:
        for batch_x, batch_y in threaded_flow:
            ...  # Do something with batch_x and batch_y

    # If you use `MLSnippet <https://github.com/haowen-xu/mlsnippet>`_,
    # you can even load data from a MongoDB via data flow.  Suppose you
    # have stored all images from ImageNet into a GridFS (of MongoDB),
    # along with the labels stored as ``metadata.y``.
    # You may iterate through the ImageNet in batches by:
    from mlsnippet.datafs import MongoFS

    fs = MongoFS('mongodb://localhost', 'imagenet', 'train')
    with fs.as_flow(batch_size=64, with_names=False, meta_keys=['y'],
                    shuffle=True, skip_incomplete=True) as flow:
        for batch_x, batch_y in flow:
            ...  # Do something with batch_x and batch_y.  batch_x is the
                 # raw content of images you stored into the GridFS.

Training
~~~~~~~~

After you've build the model and obtained the training operation, you may
quickly run a training-loop by using utilities from ``tfsnippet.scaffold``
and ``tfsnippet.trainer``.

.. code-block:: python

    from tfsnippet.dataflow import DataFlow
    from tfsnippet.scaffold import TrainLoop
    from tfsnippet.trainer import Trainer, Evaluator, AnnealingDynamicValue

    input_x = ...  # the input x placeholder
    input_y = ...  # the input y placeholder
    loss = ...  # the training loss
    params = tf.trainable_variables()  # the trainable parameters

    # We shall adopt learning-rate annealing, the initial learning rate is
    # 0.001, and we would anneal it by a factor of 0.99995 after every step.
    learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
    learning_rate_var = AnnealingDynamicValue(0.001, 0.99995)

    # Build the training operation by AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, var_list=params)

    # Build the training data-flow
    train_flow = DataFlow.arrays(
        [train_x, train_y], batch_size=64, shuffle=True, skip_incomplete=True)
    # Build the validation data-flow
    valid_flow = DataFlow.arrays([valid_x, valid_y], batch_size=256)

    with TrainLoop(params, max_epoch=max_epoch, early_stopping=True) as loop:
        trainer = Trainer(loop, train_op, [input_x, input_y], train_flow,
                          metrics={'loss': loss})
        # Anneal the learning-rate after every step by 0.99995.
        trainer.anneal_after_steps(learning_rate_var, freq=1)
        # Do validation and apply early-stopping after every epoch.
        trainer.evaluate_after_epochs(
            Evaluator(loop, loss, [input_x, input_y], valid_flow),
            freq=1
        )
        # You may log the learning-rate after every epoch by adding a callback
        # hook.  Surely you may also add any other callbacks.
        trainer.after_epochs.add_hook(
            lambda: trainer.loop.collect_metrics(lr=learning_rate_var),
            freq=1
        )
        # Print training metrics after every epoch.
        trainer.log_after_epochs(freq=1)
        # Run all the training epochs and steps.
        trainer.run(feed_dict={learning_rate: learning_rate_var})


Pre-trained Models
~~~~~~~~~~~~~~~~~~

The ``tfsnippet.applications`` package provides useful utilities to load
and use pre-trained models (among which most are third-party models).

.. code-block:: python

    from tfsnippet.applications import InceptionV3

    # Model from https://www.tensorflow.org/tutorials/image_recognition
    inception_v3 = InceptionV3()
    image_data = imageio.imread('path-to-image.jpg')
    class_proba = inception_v3.predict_proba([image_data])[0]


Math Operations
~~~~~~~~~~~~~~~

The ``tfsnippet.mathops`` package provides numerical stable implementations for
lots of advanced neural network math operations.  Also, it supports both
NumPy and TensorFlow.  You may obtain the math operation for a particular
backend by passing ``tfsnippet.mathops.npyos`` (for NumPy) or
``tfsnippet.mathops.tfops`` (for TensorFlow) as the first argument ``ops``
of every math operation function.

.. code-block:: python

    from tfsnippet.mathops import npyops, tfops
    from tfsnippet.mathops import log_sum_exp, log_softmax

    # Compute :math:`\log \sum_{k=1}^K \exp(x_k)` by TensorFlow
    log_sum_exp(tfops, x, axis=-1)
    # Compute :math:`\log \frac{\exp(x_k)}{\sum_i \exp(x_i)}` by NumPy
    log_softmax(npyops, logits)
