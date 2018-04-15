TFSnippet
=========

.. image:: https://travis-ci.org/korepwx/tfsnippet.svg?branch=master
    :target: https://travis-ci.org/korepwx/tfsnippet
.. image:: https://coveralls.io/repos/github/korepwx/tfsnippet/badge.svg?branch=master
    :target: https://coveralls.io/github/korepwx/tfsnippet?branch=master
.. image:: https://readthedocs.org/projects/tfsnippet/badge/?version=latest
    :target: http://tfsnippet.readthedocs.io/en/latest/?badge=latest

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
    pip install git+https://github.com/korepwx/tfsnippet.git

Documentation
-------------

* `Tutorials and API docs <http://tfsnippet.readthedocs.io/>`_

Quick Tutorial
--------------

Use the flexible training loop context, with nice logging and early-stopping:

.. code-block:: python

    from tfsnippet.scaffold import TrainLoop
    from tfsnippet.utils import ensure_variables_initialized

    loss = ...  # build your own training loss here
    params = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.get_variable('global_step', dtype=tf.int32, initializer=0)
    grads = optimizer.compute_gradients(loss, var_list=params)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Initialize all un-initialized TensorFlow variables
    ensure_variables_initialized()

    # Start a training loop, with early-stopping enabled on `params`.
    # By default, `params` w.r.t. best `valid_loss` will be restored when
    # exiting the `loop` context.
    with TrainLoop(params, max_epoch=max_epoch, early_stopping=True) as loop:
        # print the summary of training, e.g., the `params` to be trained
        loop.print_training_summary()

        for epoch in loop.iter_epochs():  # `epoch` is the epoch counter
            # build your own training mini-batches iterator here
            train_iterator = ...

            # run mini-batches of current `epoch`
            for step, (batch_x, batch_y) in loop.iter_steps(train_iterator):
                step_loss, _ = session.run(
                    [loss, train_op],
                    feed_dict={
                        input_x: batch_x,
                        input_y: batch_y,
                    }
                )
                loop.collect_metrics(loss=step_loss)

            # build your own validation mini-batches iterator here
            valid_iterator = ...

            # run validation of current `epoch`
            with loop.timeit('valid_time'), \
                    loop.metric_collector('valid_loss') as collector:
                for batch_x, batch_y in valid_iterator:
                    valid_loss = session.run(
                        loss,
                        feed_dict={
                            input_x: batch_x,
                            input_y: batch_y,
                        }
                    )
                    collector.collect(valid_loss=valid_loss, weight=len(batch_x))

            # print the logs at the end of every epoch
            loop.print_logs()

Or use the early-stopping context directly, without emitting a training loop:

.. code-block:: python

    from tfsnippet.scaffold import EarlyStopping

    with EarlyStopping(params) as es:
        ...
        es.update(loss)  # This will update the loss being monitored.
                         # It can be called for arbitrary times, and
                         # `param` will be restored w.r.t. the best loss
                         # when exiting the `es` context.
        ...
