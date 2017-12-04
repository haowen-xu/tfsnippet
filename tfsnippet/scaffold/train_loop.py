# -*- coding: utf-8 -*-
import os
import time
from collections import OrderedDict
from contextlib import contextmanager

import six
import tensorflow as tf

from tfsnippet.utils import StatisticsCollector
from .logging import summarize_variables, DefaultMetricFormatter, MetricLogger
from .validation import (EarlyStoppingContext, 
                         early_stopping as open_early_stopping)

__all__ = [
    'train_loop', 'TrainLoopContext',
]

EPOCH_TIME_METRIC = 'epoch_time'
STEP_TIME_METRIC = 'step_time'


if six.PY2:
    def print_function(message):
        print(message)
else:
    print_function = print


class TrainLoopContext(object):
    """
    Training loop context object.

    Args:
        param_vars (list[tf.Variable] or dict[str, tf.Variable]):
            List or dict of variables, optimized during training.
        print_func ((str) -> None): Function for printing log messages.
        summary_writer: TensorFlow summary writer for writing metrics onto disk.
        metric_formatter (MetricFormatter): The training metrics formatter.
        valid_metric_name (str): Name of the validation metric.
        initial_valid_metric (float):
            Initial value of the validation metric for early-stopping.
        valid_metric_smaller_is_better (bool):
            Whether or not the smaller value is better for validation metric?
        early_stopping (EarlyStoppingContext): The early-stopping context.
        initial_epoch (int): The initial epoch.
        initial_step (int): The initial step.
        max_epoch (int or None): The maximum epoch to run.
        max_step (int or None): The maximum step to run.
    """

    def __init__(self,
                 param_vars,
                 print_func,
                 summary_writer,
                 metric_formatter,
                 valid_metric_name,
                 initial_valid_metric,
                 valid_metric_smaller_is_better,
                 early_stopping,
                 initial_epoch,
                 initial_step,
                 max_epoch,
                 max_step):
        self._param_vars = param_vars
        self._print_func = print_func
        self._metric_formatter = metric_formatter
        self._early_stopping = early_stopping
        self._valid_metric_name = valid_metric_name
        self._valid_metric_smaller_is_better = valid_metric_smaller_is_better
        self._summary_writer = summary_writer

        # metric accumulators
        self._step_metrics = MetricLogger(formatter=self._metric_formatter)
        self._epoch_metrics = MetricLogger(summary_writer=self._summary_writer,
                                           formatter=self._metric_formatter)

        # flag to track the context
        self._within_epoch = False
        self._within_step = False

        # epoch and step counter
        self._epoch = initial_epoch
        self._step = initial_step
        self._max_epoch = max_epoch
        self._max_step = max_step

        # epoch and step context flags
        self._best_valid_metric = initial_valid_metric
        self._is_best_valid_metric = False
        self._epoch_start_time = None
        self._step_start_time = None

    def _commit_epoch_start_time(self):
        if self._epoch_start_time is not None:
            duration = time.time() - self._epoch_start_time
            self.collect_metrics(metrics={EPOCH_TIME_METRIC: duration})
            self._epoch_start_time = None

    def _commit_step_start_time(self):
        if self._step_start_time is not None:
            duration = time.time() - self._step_start_time
            self.collect_metrics(metrics={STEP_TIME_METRIC: duration})
            self._step_start_time = None

    @property
    def summary_writer(self):
        """Get the summary writer instance."""
        return self._summary_writer

    @property
    def param_vars(self):
        """Get the trainable parameter variables."""
        return self._param_vars

    @property
    def epoch(self):
        """Get the epoch counter (starting from 1)."""
        return self._epoch

    @property
    def max_epoch(self):
        """Get or set the max value for epoch counter."""
        return self._max_epoch

    @max_epoch.setter
    def max_epoch(self, value):
        self._max_epoch = int(value)

    @property
    def step(self):
        """Get the global step counter (starting from 1)."""
        return self._step

    @property
    def max_step(self):
        """Get or set the max value for global step counter."""
        return self._max_step

    @max_step.setter
    def max_step(self, value):
        self._max_step = int(value)

    @property
    def best_valid_metric(self):
        """Get the best valid metric."""
        return self._best_valid_metric

    def iter_epochs(self):
        """
        Iterate through the epochs.

        This method can only be called when there's no other epoch loop
        is being iterated.  Furthermore, after exiting this loop, both
        the epoch metrics as well as the step metrics will be cleared.

        If `max_epoch` is configured, it will stop at it.

        Yields:
            int: The epoch counter (starting from 1).
        """
        def loop_condition():
            return (
                (self._max_epoch is None or self._epoch < self._max_epoch) and
                (self._max_step is None or self._step < self._max_step)
            )

        if self._within_epoch:
            raise RuntimeError('Another epoch loop has been opened')
        try:
            while loop_condition():
                self._epoch += 1
                self._within_epoch = True
                self._epoch_start_time = time.time()
                yield self._epoch
                self._commit_epoch_start_time()
        finally:
            self._within_epoch = False
            self._epoch_start_time = None
            self._step_metrics.clear()
            self._epoch_metrics.clear()
            self._is_best_valid_metric = False

    def iter_steps(self, data_generator=None):
        """
        Iterate through the steps.

        This method can only be called when there's no other step loop
        is being iterated, and an epoch loop is active.

        Args:
            data_generator:
                Optional iterable data to be yielded at every step.
                This is required if `max_step` is not configured, so as to
                prevent an infinite step loop.

        Yields:
            int or (int, any):
                The global step counter (starting from 1), or the tuple of
                ``(step counter, batch data)`` if `data_generator` is specified.
        """
        def loop_condition():
            return self._max_step is None or self._step < self._max_step

        if not self._within_epoch:
            raise RuntimeError('Step loop must be opened within active epoch '
                               'loop')
        if self._within_step:
            raise RuntimeError('Another step loop has been opened')
        if self._max_step is None and data_generator is None:
            raise RuntimeError('`data_generator` is required when `max_step` '
                               'is not configured, so as to prevent an '
                               'unstoppable step loop')

        try:
            if data_generator is not None:
                data_generator = iter(data_generator)

            while loop_condition():
                # prepare for the step data
                if data_generator is None:
                    yield_obj = self._step + 1
                else:
                    try:
                        step_data = next(data_generator)
                    except StopIteration:
                        break
                    yield_obj = self._step + 1, step_data

                # yield this step
                self._step += 1
                self._within_step = True
                self._step_start_time = time.time()
                yield yield_obj
                self._commit_step_start_time()
        finally:
            self._within_step = False
            self._step_start_time = None

    def _require_context(self):
        if not self._within_epoch and not self._within_step:
            raise RuntimeError('An epoch or a step loop is expected, but '
                               'neither has been opened')

    @contextmanager
    def timeit(self, metric_name):
        """
        Open a context for timing.

        Args:
            metric_name (str):
                Store the timing result in metric of this name.
                Note that `metric_name` must end with ``time`` or ``timer``,
                otherwise by default the time values will not be formatted
                as human readable strings.
        """
        self._require_context()
        start_time = time.time()
        yield
        duration = time.time() - start_time
        self.collect_metrics(metrics={metric_name: duration})

    @contextmanager
    def metric_collector(self, metric_name):
        """
        Get a :class:`~tfsnippet.utils.StatisticsCollector` for metric.

        The mean value of the collected metrics will be added to summary
        after exiting the context.  Other statistics will be discarded.

        Args:
            metric_name (str): The name of this metric.

        Yields:
            StatisticsCollector: The collector for metric values.
        """
        self._require_context()
        acc = StatisticsCollector()
        yield acc
        if acc.has_value:
            self.collect_metrics(metrics={metric_name: acc.mean})

    def collect_metrics(self, metrics=None, **kwargs):
        """
        Add metric values.

        This method must be called when there's at least an active epoch
        loop.  It will add metrics to the epoch metrics collector, and if
        there's an active step loop, it will also add metrics to the step
        metrics collector.

        If `summary_writer` is configured, it will also write the metrics
        as summaries onto disk.  Furthermore, if `valid_metric_name`
        is configured, it will also perform early-stopping.

        Args:
            metrics (dict[str, float or np.ndarray]): Metric values as dict.
            **kwargs: Metric values, specified as named arguments.
        """
        self._require_context()

        if metrics is None:
            metrics = {}
        elif metrics is not None and not isinstance(metrics, dict):
            raise TypeError('`metrics` should be a dict')
        metrics.update(kwargs)

        self._epoch_metrics.collect_metrics(metrics, global_step=self.step)
        if self._within_step:
            self._step_metrics.collect_metrics(metrics, global_step=self.step)

        def update_valid_metric(d):
            v = d.get(self._valid_metric_name)
            if v is not None:
                if self._best_valid_metric is None or \
                        (self._valid_metric_smaller_is_better and
                         v < self._best_valid_metric) or \
                        (not self._valid_metric_smaller_is_better and
                         v > self._best_valid_metric):
                    self._best_valid_metric = v
                    self._is_best_valid_metric = True
                else:
                    self._is_best_valid_metric = False
                if self._early_stopping:
                     self._early_stopping.update(v, self.step)
        if self._valid_metric_name:
            if metrics:
                update_valid_metric(metrics)

    def add_summary(self, summary):
        """
        Add a summary object, with ``self.step`` as `global_step`.

        Args:
            summary (tf.summary.Summary or byes):
                TensorFlow summary object, or serialized summary.
        """
        self._summary_writer.add_summary(summary, global_step=self.step)

    def println(self, message, with_tag=False):
        """
        Print `message` via `print_function`.

        Args:
            message (str): Message to be printed.
            with_tag (bool): Whether or not to add the epoch & step tag?
                             (default :obj:`False`)
        """
        if with_tag:
            def format_tag(v, max_v, name):
                if max_v is not None:
                    return '{} {}/{}'.format(name, v, max_v)
                else:
                    return '{} {}'.format(name, v)

            tag = ''
            if self._within_step:
                tag = ', '.join((
                    format_tag(self._epoch, self._max_epoch, 'Epoch'),
                    format_tag(self._step, self._max_step, 'Step'),
                ))
            elif self._within_epoch:
                tag = format_tag(self._epoch, self._max_epoch, 'Epoch')
            else:
                self._require_context()
            message = '[{}] {}'.format(tag, message)
        self._print_func(message)

    def print_training_summary(self):
        """
        Print the training summary.

        The training summary include the following content:

        1.   Execution environment.
        2.   Parameters to be optimized during training.
        """
        self.println(summarize_variables(variables=self._param_vars,
                                         title='Trainable Parameters'))
        self.println('')

    def print_logs(self):
        """
        Print the training logs.

        This method will print the collected metrics.  If there's an
        active step loop, it will print metrics from the step metrics
        collector.  Otherwise if there's only an epoch loop, it will
        print metrics from the epoch metrics accumulator.

        Note it must be called at the end of an epoch or a step.
        This is because the metrics of corresponding loop context will be
        cleared after the logs are printed.
        Moreover, the epoch or step timer will be committed as metric
        immediately when this method is called, before printing the logs.
        """
        metrics = None
        if self._within_step:
            self._commit_step_start_time()
            metrics = self._step_metrics
        elif self._within_epoch:
            self._commit_epoch_start_time()
            metrics = self._epoch_metrics
        else:
            self._require_context()

        best_mark = ' (*)' if self._is_best_valid_metric else ''
        self.println(metrics.format_logs() + best_mark, with_tag=True)
        self._is_best_valid_metric = False
        metrics.clear()


@contextmanager
def train_loop(param_vars,
               print_func=print_function,
               summary_dir=None,
               summary_writer=None,
               metric_formatter=DefaultMetricFormatter(),
               valid_metric_name='valid_loss',
               initial_valid_metric=None,
               valid_metric_smaller_is_better=None,
               early_stopping=False,
               initial_epoch=0,
               initial_step=0,
               max_epoch=None,
               max_step=None):
    """
    Open a training loop context.

    This method should open a context for training loop, and provide an object
    shipped with a set of convenient methods for common routines in training.
    Basically, it should be useful for maintaining epoch and step counters,
    logging training metrics, memorizing best parameters via early-stopping,
    and many more.  An example of using the :func:`train_loop`:

    .. code-block:: python

        with train_loop(param_vars, max_epoch=10, early_stopping=True) as loop:
            loop.print_training_summary()
            for epoch in loop.iter_epochs():
                data_iterator = zip(
                    minibatch_iterator(data_x, batch_size),
                    minibatch_iterator(data_y, batch_size),
                )
                for step, (x, y) in loop.iter_steps(data_iterator):
                    step_loss = session.run(
                        [loss, train_op],
                        feed_dict={input_x: x, input_y: y}
                    )
                    loop.collect_metrics(loss=step_loss)
                with loop.timeit('valid_time'):
                    valid_loss = session.run(
                        loss, feed_dict={input_x: test_x, input_y: test_y})
                    loop.collect_metrics(valid_loss=valid_loss)
                loop.print_logs()

    Args:
        param_vars (list[tf.Variable] or dict[str, tf.Variable]):
            List or dict of variables, optimized during training.
        print_func ((str) -> None):
            Function for printing log messages (calling ``print`` by default).
            An alternative of this argument may be ``getLogger(__name__).info``,
            such that the log messages will be printed via logging facilities.
        summary_dir (str):
            Directory for writing TensorFlow summaries.
            Ignored if `summary_writer` is specified.
        summary_writer: TensorFlow summary writer for writing metrics onto disk.
        metric_formatter (MetricFormatter): The training metrics formatter.
        valid_metric_name (str): Name of the validation metric.
        initial_valid_metric (float or tf.Tensor or tf.Variable):
            Initial value of the validation metric for early-stopping.
        valid_metric_smaller_is_better (bool):
            Whether or not the smaller value is better for validation metric?
            If not specified, it will be inferred according to
            `valid_metric_name`: metric names with ``acc`` or ``accuracy`` as
            suffix imply :obj:`True`, while other names imply :obj:`False`.
        early_stopping (bool):
            Whether or not to do early-stopping? (default :obj:`False`)
            If :obj:`True`, early-stopping will be applied on `param_vars`,
            according to the validation metric.
        initial_epoch (int or tf.Tensor or tf.Variable):
            The initial epoch (default 0).
            Should be one less than the actual first epoch.
        initial_step (int or tf.Tensor or tf.Variable):
            The initial step (default 0).
            Should be one less than the actual first step.
        max_epoch (int or tf.Tensor or tf.Variable): The maximum epoch to run.
        max_step (int or tf.Tensor or tf.Variable): The maximum step to run.

    Yields:
        TrainLoopContext: The training loop context object.
    """
    if not isinstance(param_vars, (dict, OrderedDict)):
        param_vars = list(param_vars)

    if isinstance(initial_valid_metric, (tf.Variable, tf.Tensor)):
        initial_valid_metric = initial_valid_metric.eval()
    if isinstance(initial_epoch, (tf.Variable, tf.Tensor)):
        initial_epoch = int(initial_epoch.eval())
    if isinstance(initial_step, (tf.Variable, tf.Tensor)):
        initial_step = int(initial_step.eval())
    if isinstance(max_epoch, (tf.Variable, tf.Tensor)):
        max_epoch = int(max_epoch.eval())
    if isinstance(max_step, (tf.Variable, tf.Tensor)):
        max_step = int(max_step.eval())

    smaller_is_better = valid_metric_smaller_is_better
    if smaller_is_better is None:
        smaller_is_better = not (
            valid_metric_name.endswith('acc') or
            valid_metric_name.endswith('accuracy')
        )

    close_summary_writer = False
    if summary_writer is None and summary_dir is not None:
        summary_dir = os.path.abspath(summary_dir)
        summary_writer = tf.summary.FileWriter(summary_dir)
        close_summary_writer = True

    def mkctx(es=None):
        return TrainLoopContext(
            param_vars=param_vars,
            print_func=print_func,
            metric_formatter=metric_formatter,
            valid_metric_name=valid_metric_name,
            initial_valid_metric=initial_valid_metric,
            valid_metric_smaller_is_better=smaller_is_better,
            early_stopping=es,
            summary_writer=summary_writer,
            initial_epoch=initial_epoch,
            initial_step=initial_step,
            max_epoch=max_epoch,
            max_step=max_step,
        )

    try:
        if early_stopping and len(param_vars) > 0:
            with open_early_stopping(param_vars,
                                     initial_metric=initial_valid_metric,
                                     smaller_is_better=smaller_is_better) as es:
                yield mkctx(es)
        else:
            yield mkctx(None)
    finally:
        if close_summary_writer:
            summary_writer.close()
