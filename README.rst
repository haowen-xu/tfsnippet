TFSnippet
=========

+---------+-----------------+-----------------+---------------+
| Stable  | |stable_build|  | |stable_cover|  | |stable_doc|  |
+---------+-----------------+-----------------+---------------+
| Develop | |develop_build| | |develop_cover| | |develop_doc| |
+---------+-----------------+-----------------+---------------+

.. |stable_build| image:: https://travis-ci.org/haowen-xu/tfsnippet.svg?branch=stable
    :target: https://travis-ci.org/haowen-xu/tfsnippet
.. |stable_cover| image:: https://coveralls.io/repos/github/haowen-xu/tfsnippet/badge.svg?branch=stable
    :target: https://coveralls.io/github/haowen-xu/tfsnippet?branch=stable
.. |stable_doc| image:: https://readthedocs.org/projects/tfsnippet/badge/?version=stable
    :target: http://tfsnippet.readthedocs.io/en/stable/
.. |develop_build| image:: https://travis-ci.org/haowen-xu/tfsnippet.svg?branch=develop
    :target: https://travis-ci.org/haowen-xu/tfsnippet
.. |develop_cover| image:: https://coveralls.io/repos/github/haowen-xu/tfsnippet/badge.svg?branch=develop
    :target: https://coveralls.io/github/haowen-xu/tfsnippet?branch=develop
.. |develop_doc| image:: https://readthedocs.org/projects/tfsnippet/badge/?version=latest
    :target: http://tfsnippet.readthedocs.io/en/latest/

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

    pip install git+https://github.com/haowen-xu/tfsnippet.git

Documentation
-------------

* `Tutorials and API docs <http://tfsnippet.readthedocs.io/>`_

Examples
--------

* Classification:
   - `MNIST <tfsnippet/examples/classification/mnist.py>`_.
   - `Convolutional MNIST <tfsnippet/examples/classification/mnist_conv.py>`_.

* Auto Encoders:
   - `VAE <tfsnippet/examples/auto_encoders/vae.py>`_.
   - `Convolutional VAE <tfsnippet/examples/auto_encoders/conv_vae.py>`_: VAE with convolutional layers.
   - `Bernoulli Latent VAE <tfsnippet/examples/auto_encoders/bernoulli_latent_vae.py>`_: VAE with `q(z|x)` and `p(z)` being Bernoulli distribution.
   - `Mixture Prior VAE <tfsnippet/examples/auto_encoders/mixture_vae.py>`_: VAE with `p(z)` being a mixture of Gaussian.
   - `Gaussian Mixture VAE <tfsnippet/examples/auto_encoders/gm_vae.py>`_: GM-VAE, with auxiliary categorical variable `y`.
   - `Planar Normalizing Flow <tfsnippet/examples/auto_encoders/planar_nf.py>`_: VAE with `q(z|x)` derived by a planar normalizing flow.
   - `Dense Real NVP <tfsnippet/examples/auto_encoders/dense_real_nvp.py>`_: VAE with `q(z|x)` derived by a Real NVP (with dense layers).

Quick Tutorial
--------------

From the very beginning, you might import the TFSnippet as:

.. code-block:: python

    import tfsnippet as spt

Distributions
~~~~~~~~~~~~~

If you use TFSnippet distribution classes to obtain random samples, you
shall get enhanced tensor objects, from which you may compute the
log-likelihood by simply calling ``log_prob()``.

.. code-block:: python

    normal = spt.Normal(0., 1.)
    # The type of `samples` is :class:`tfsnippet.stochastic.StochasticTensor`.
    samples = normal.sample(n_samples=100)
    # You may obtain the log-likelhood of `samples` under `normal` by:
    log_prob = samples.log_prob()
    # You may also obtain the distribution instance back from the samples,
    # such that you may fire-and-forget the distribution instance!
    distribution = samples.distribution

The distributions from `ZhuSuan <https://github.com/thu-ml/zhusuan.git>`_ can
be casted into a TFSnippet distribution class, in case we
haven't provided a wrapper for a certain ZhuSuan distribution:

.. code-block:: python

    import zhusuan as zs

    uniform = spt.as_distribution(zs.distributions.Uniform())
    # The type of `samples` is :class:`tfsnippet.stochastic.StochasticTensor`.
    samples = uniform.sample(n_samples=100)

Data Flows
~~~~~~~~~~

It is a common practice to iterate through a dataset by mini-batches.
The ``tfsnippet.DataFlow`` provides a unified interface for assembling
the mini-batch iterators.

.. code-block:: python

    # Obtain a shuffled, two-array data flow, with batch-size 64.
    # Any batch with samples fewer than 64 would be discarded.
    flow = spt.DataFlow.arrays(
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
quickly run a training-loop by using utilities from TFSnippet:

.. code-block:: python

    input_x = ...  # the input x placeholder
    input_y = ...  # the input y placeholder
    loss = ...  # the training loss
    params = tf.trainable_variables()  # the trainable parameters

    # We shall adopt learning-rate annealing, the initial learning rate is
    # 0.001, and we would anneal it by a factor of 0.99995 after every step.
    learning_rate = spt.AnnealingVariable('learning_rate', 0.001, 0.99995)

    # Build the training operation by AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, var_list=params)

    # Build the training data-flow
    train_flow = spt.DataFlow.arrays(
        [train_x, train_y], batch_size=64, shuffle=True, skip_incomplete=True)
    # Build the validation data-flow
    valid_flow = spt.DataFlow.arrays([valid_x, valid_y], batch_size=256)

    with spt.TrainLoop(params, max_epoch=max_epoch, early_stopping=True) as loop:
        trainer = spt.Trainer(loop, train_op, [input_x, input_y], train_flow,
                              metrics={'loss': loss})
        # Anneal the learning-rate after every step by 0.99995.
        trainer.anneal_after_steps(learning_rate, freq=1)
        # Do validation and apply early-stopping after every epoch.
        trainer.evaluate_after_epochs(
            spt.Evaluator(loop, loss, [input_x, input_y], valid_flow),
            freq=1
        )
        # You may log the learning-rate after every epoch registering an
        # event handler.  Surely you may also add any other handlers.
        trainer.events.on(
            EventKeys.AFTER_EPOCH,
            lambda epoch: trainer.loop.collect_metrics(lr=learning_rate),
        )
        # Print training metrics after every epoch.
        trainer.log_after_epochs(freq=1)
        # Run all the training epochs and steps.
        trainer.run()

