import six

from tfsnippet.scaffold import TrainLoop
from .base_trainer import BaseTrainer
from .feed_dict import resolve_feed_dict, merge_feed_dict


__all__ = ['Trainer']


class Trainer(BaseTrainer):
    """
    A subclass of :class:`BaseTrainer`, executing a training operation per step.
    This might be the most commonly used :class:`Trainer`.  Code example::

        from tfsnippet.scaffold import TrainLoop
        from tfsnippet.trainer import (LossTrainer,
                                       Evaluator,
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
            trainer = Trainer(
                loop, train_op, [input_x, input_y], train_data,
                feed_dict={learning_rate: learning_rate_var},
                metrics={'loss': loss'}
            )
            evaluator = Evaluator(
                loop, {'loss': loss}, [input_x, input_y], valid_data)

            # validate after every epoch
            trainer.evaluate_after_epochs(evaluator, freq=1)

            # log after every epoch (and after validation, since
            # ``HookPriority.VALIDATION < HookPriority.LOGGING``)
            trainer.log_after_epochs(freq=1)

            # anneal the learning rate after every 10 epochs
            trainer.anneal_after_epochs(learning_rate_var, freq=10)

            # run the main training loop
            trainer.run()
    """

    def __init__(self, loop, train_op, inputs, data_flow, feed_dict=None,
                 metrics=None):
        """

        Args:
            loop (TrainLoop): The training loop object.
            train_op (tf.Operation): The training operation.
            inputs (list[tf.Tensor]): The input placeholders.
                The number of tensors, and the order of tensors, should
                both match the arrays of each mini-batch data, provided
                by `data_flow`.
            data_flow (DataFlow): The training data flow.
                Each mini-batch must contain one array for each placeholder
                in `inputs`.
            feed_dict: The feed dict for training.  It will be merged with
                the arrays provided by `data_flow` in each step.
                (default :obj:`None`)
            metrics (dict[str, tf.Tensor]): Metrics to be computed along with
                `train_op`.  The keys are the names of metrics.
        """
        if loop.max_epoch is None and loop.max_step is None:
            raise ValueError('At least one of `max_epoch`, `max_step` should '
                             'be configured for `loop`.')
        super(Trainer, self).__init__(loop=loop)

        # memorize the arguments
        self._inputs = tuple(inputs or ())
        self._data_flow = data_flow
        self._feed_dict = dict(feed_dict or ())
        self._train_op = train_op
        self._metrics = dict(metrics or ())

    @property
    def inputs(self):
        """
        Get the input placeholders.

        Returns:
            list[tf.Tensor]: The input placeholders.
        """
        return self._inputs

    @property
    def data_flow(self):
        """
        Get the training data flow.

        Returns:
            DataFlow: The training data flow.
        """
        return self._data_flow

    @property
    def feed_dict(self):
        """
        Get the feed dict for training.

        Returns:
            dict[tf.Tensor, any]: The feed dict for training.
        """
        return self._feed_dict

    @property
    def train_op(self):
        """Get the training operation."""
        return self._train_op

    @property
    def metrics(self):
        """Get the metrics to be computed along with `train_op`."""
        return self._metrics

    def _iter_steps(self):
        return self.loop.iter_steps(self.data_flow)

    def _run_step(self, session, payload):
        # prepare for the feed dict of this step
        step, batch_data = payload
        feed_dict = resolve_feed_dict(
            merge_feed_dict(
                self.feed_dict,
                zip(self.inputs, batch_data)
            )
        )

        # run the training operation
        metric_names = list(six.iterkeys(self.metrics))
        metric_tensors = [self.metrics[k] for k in metric_names]
        metric_values = session.run(
            [self._train_op] + metric_tensors, feed_dict=feed_dict)[1:]
        self.loop.collect_metrics(
            {n: v for n, v in zip(metric_names, metric_values)})
