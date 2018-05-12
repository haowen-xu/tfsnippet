from .base_trainer import BaseTrainer

__all__ = ['LossTrainer']


class LossTrainer(BaseTrainer):
    """
    A subclass of :class:`BaseTrainer`, which optimizes a single loss.

    This class does not derive the training operation that minimizes or
    maximizes the loss.  Instead, the caller must derive the training
    operation and pass it to the :class:`LossTrainer`.  For example::

        from tfsnippet.scaffold import TrainLoop
        from tfsnippet.trainer import (LossTrainer,
                                       Validator,
                                       AnnealingDynamicValue)

        # build the model
        input_x = tf.placeholder(...)
        input_y = tf.placeholder(...)
        learning_rate = tf.placeholder(...)  # learning rate annealing

        # prepare for the data and
        train_data = DataFlow.arrays(
            [train_x, train_y], batch_size=128, shuffle=True,
            skip_incomplete=True
        )
        valid_data = DataFlow.arrays(
            [valid_x, valid_y], batch_size=512)
        ...

        # derive the training operation
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

        # run the trainer
        learning_rate_var = AnnealingDynamicValue(0.001, ratio=0.75)

        with TrainLoop(param_vars,
                       max_epoch=10,
                       early_stopping=True) as loop:
            trainer = LossTrainer(
                loop, loss, train_op, [input_x, input_y], train_data,
                feed_dict={learning_rate: learning_rate_var}
            )
            validator = Validator(
                loop, loss, [input_x, input_y], valid_data)

            # validate after every epoch
            trainer.validate_after_epochs(validator, freq=1)

            # log after every epoch (and after validation, since
            # ``HookPriority.VALIDATION < HookPriority.LOGGING``)
            trainer.log_after_epochs(freq=1)

            # anneal the learning rate after every 10 epochs
            trainer.anneal_after_epochs(learning_rate_var, freq=10)

            # run the main training loop
            trainer.run()
    """

    def __init__(self, loop, loss, train_op, inputs, data_flow, feed_dict=None,
                 metric_name='loss'):
        """
        Construct a new :class:`LossTrainer`.

        Args:
            loop (TrainLoop): The training loop object.
            loss (tf.Tensor): The training loss.
            train_op (tf.Operation): The training operation.
            inputs (list[tf.Tensor]): The input placeholders.
                The number of tensors, and the order of tensors, should
                both match the arrays of each mini-batch data, provided
                by `data_flow`.
            data_flow (DataFlow): The training data flow.
            feed_dict (dict[tf.Tensor, any]): The fixed feed dict for
                training.  It will be merged with `inputs` and the
                argument of ``run(feed_dict)``. (default :obj:`None`)
            metric_name (str): The metric name for collecting training loss.
        """
        super(LossTrainer, self).__init__(
            loop=loop, inputs=inputs, data_flow=data_flow, feed_dict=feed_dict)

        # memorize the variables
        self._loss = loss
        self._train_op = train_op
        self._metric_name = metric_name

    @property
    def loss(self):
        """Get the training loss."""
        return self._loss

    @property
    def train_op(self):
        """Get the training operation."""
        return self._train_op

    @property
    def metric_name(self):
        """Get the metric name for collecting training loss."""
        return self._metric_name

    def _fit_step(self, session, feed_dict):
        _, loss = session.run([self._train_op, self.loss], feed_dict=feed_dict)
        self.loop.collect_metrics({self._metric_name: loss})
