from __future__ import print_function

import copy
import os
import re
import time
from collections import OrderedDict
from contextlib import contextmanager
from logging import getLogger

import six
import tensorflow as tf

from tfsnippet.dataflows import DataFlow
from tfsnippet.utils import (StatisticsCollector, DisposableContext,
                             humanize_duration, ETA, deprecated,
                             EventSource)
from .checkpoint import CheckpointSavableObject, CheckpointSaver
from .early_stopping_ import EarlyStopping
from .event_keys import EventKeys
from .logging_ import summarize_variables, DefaultMetricFormatter, MetricLogger

__all__ = [
    'TrainLoop', 'TrainLoopContext', 'train_loop',
]

EPOCH_TIME_METRIC = 'epoch_time'
STEP_TIME_METRIC = 'step_time'


class TrainLoopStates(CheckpointSavableObject):
    """
    Internal states of a :class:`TrainLoop`, which can be saved via a
    :class:`CheckpointSaver`.
    """

    def __init__(self, epoch=0, step=0):
        self.epoch = epoch
        self.step = step

    def get_state(self):
        return {
            'epoch': self.epoch,
            'step': self.step
        }

    def set_state(self, state):
        self.epoch = state['epoch']
        self.step = state['step']


class TrainLoop(DisposableContext):
    """
    Training loop object.

    This class provides a set of convenient methods for writing training loop.
    It is useful for maintaining epoch and step counters, logging training
    metrics, memorizing best parameters for early-stopping, etc.  An
    example of using the :class:`TrainLoop`::

        import tfsnippet as spt

        with spt.TrainLoop(param_vars,
                           max_epoch=10,
                           early_stopping=True) as loop:
            loop.print_training_summary()
            train_flow = spt.DataFlow.arrays([x, y], batch_size, shuffle=True)

            for epoch in loop.iter_epochs():
                for step, (x, y) in loop.iter_steps(train_flow):
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

    The event schedule of a :class:`TrainLoop` can be briefly described as::

        # the main training loop
        events.fire(EventKeys.ENTER_LOOP, self)

        for epoch in self.iter_epochs():
            events.fire(EventKeys.BEFORE_EPOCH, self)

            for step in self.iter_steps(...):
                events.fire(EventKeys.BEFORE_STEP, self)

                ...  # execute the step

                events.reverse_fire(EventKeys.AFTER_STEP, self)

            events.reverse_fire(EventKeys.AFTER_EPOCH, self)

        events.fire(EventKeys.EXIT_LOOP, self)

        # when metrics are fed into the loop by :meth:`collect_metrics`
        def collect_metrics(self, metrics_dict=None, **kwargs):
            metrics_dict = merge(metrics_dict, kwargs)
            events.fire(EventKeys.METRICS_COLLECTED, self, metrics_dict)

        # when summaries are fed into the loop by :meth:`add_summary`
        def add_summary(self, summary):
            events.fire(EventKeys.SUMMARY_ADDED, self, summary)
    """

    def __init__(self,
                 param_vars,
                 var_groups=None,
                 show_eta=True,
                 print_func=print,
                 max_epoch=None,
                 max_step=None,
                 metric_formatter=DefaultMetricFormatter(),

                 # checkpoint related arguments
                 checkpoint_dir=None,
                 checkpoint_epoch_freq=None,
                 checkpoint_max_to_keep=None,
                 checkpoint_save_objects=None,
                 restore_checkpoint=True,

                 # summary related arguments
                 summary_dir=None,
                 summary_writer=None,
                 summary_graph=None,
                 summary_metric_prefix='metrics/',
                 summary_skip_pattern=re.compile(r'.*(time|timer)$'),
                 summary_commit_freqs=None,

                 # validation and early-stopping related arguments
                 valid_metric_name='valid_loss',
                 initial_valid_metric=None,
                 valid_metric_smaller_is_better=None,
                 early_stopping=False):
        """
        Construct the :class:`TrainLoop`.

        Args:
            param_vars (list[tf.Variable] or dict[str, tf.Variable]): List or
                dict of variables, optimized during training.
            var_groups (None or list[str]): Variable groups, the prefixes of
                variable scopes.  A hint for printing the variables summary.
                (default :obj:`None`)
            show_eta (bool): Whether or not to show ETA? (default :obj:`True`)
            print_func ((str) -> None): Function for printing log messages
                (calling ``print`` by default). An alternative of this argument
                may be ``getLogger(__name__).info``, such that the log messages
                will be printed via logging facilities.
            max_epoch (None or int or tf.Tensor or tf.Variable):
                The maximum epoch to run.  If :obj:`None`, will run for
                infinite epochs.  If ``1``, the epoch counter will be
                discarded in the output logs. (default :obj:`None`)
            max_step (None or int or tf.Tensor or tf.Variable):
                The maximum step to run.  If :obj:`None`, will run for
                infinite steps.  Note this limit applies for the total
                step counter, rather than the epoch-wise step counter.
                (default :obj:`None`)
            metric_formatter (MetricFormatter): The training metrics formatter.

            checkpoint_dir (str): If specified, will save checkpoint files to
                this directory, when :meth:`make_checkpoint()` is called.
            checkpoint_epoch_freq (int or None): If specified, will make
                checkpoint every this number of epochs.  If not specified,
                you must call :meth:`make_checkpoint()` manually.
            checkpoint_max_to_keep (int or None): Maximum number of checkpoint
                versions to keep. If :obj:`None` or `0`, keep all versions.
            checkpoint_save_objects (Iterable[CheckpointSavableObject]): If
                specified, will save and restore the states of these objects.
            restore_checkpoint (bool or str): If :obj:`True`, will restore
                the latest checkpoint.  If a str, it should be the path of
                a checkpoint file, and will restore from this checkpoint.
                If :obj:`False`, will not restore the from the checkpoint
                files (but will still save new checkpoints if `checkpoint_dir`
                if specified).

            summary_dir (str): Directory for writing TensorFlow summaries.
                Ignored if `summary_writer` is specified.
            summary_writer: TensorFlow summary writer for writing metrics.
            summary_metric_prefix (str): The prefix for the metrics committed
                to `summary_writer`.  This will not affect the summaries
                added via :meth:`add_summary`. (default "")
            summary_graph: If specified, log the graph via `summary_writer`.
            summary_skip_pattern (str or regex): Metrics matching this pattern
                will be excluded from `summary_writer`.
                (default ".*(time|timer)$")
            summary_commit_freqs (dict[str, int] or None): If specified,
                a metric will be committed to `summary_writer` no more frequent
                than ``summary_commit_freqs[metric]``. (default :obj:`None`)

            valid_metric_name (str): Name of the validation metric.
            initial_valid_metric (float or tf.Tensor or tf.Variable): Initial
                value of the validation metric for early-stopping.
            valid_metric_smaller_is_better (bool): Whether or not the smaller
                value is better for validation metric? If not specified, it
                will be inferred according to `valid_metric_name`: metric names
                with ``acc`` or ``accuracy`` as suffix imply :obj:`True`, while
                other names imply :obj:`False`.
            early_stopping (bool): Whether or not to do early-stopping?
                (default :obj:`False`)  If :obj:`True`, early-stopping will be
                applied on `param_vars`, according to the validation metric.
                This argument cannot be used toegether with `checkpoint_dir`.
        """
        # regularize the parameters
        if not isinstance(param_vars, (dict, OrderedDict)):
            param_vars = list(param_vars)
        if isinstance(max_epoch, (tf.Variable, tf.Tensor)):
            max_epoch = int(max_epoch.eval())
        if isinstance(max_step, (tf.Variable, tf.Tensor)):
            max_step = int(max_step.eval())

        if checkpoint_dir is not None:
            checkpoint_dir = os.path.abspath(checkpoint_dir)
        if checkpoint_epoch_freq is not None:
            checkpoint_epoch_freq = int(checkpoint_epoch_freq)
            if checkpoint_epoch_freq < 1:
                raise ValueError(
                    '`checkpoint_epoch_freq` must be a positive integer: '
                    'got {}'.format(checkpoint_epoch_freq)
                )
        if isinstance(restore_checkpoint, six.string_types):
            restore_checkpoint = os.path.abspath(restore_checkpoint)
        if checkpoint_dir is not None and early_stopping:
            raise ValueError(
                'Currently `early_stopping = True` is not supported when '
                '`checkpoint_dir` is specified.'
            )
        save_objects = list(checkpoint_save_objects or ())

        if summary_writer is not None:
            summary_dir = None
            own_summary_writer = False
        elif summary_dir is not None:
            summary_dir = os.path.abspath(summary_dir)
            own_summary_writer = True
        else:
            own_summary_writer = False

        if isinstance(initial_valid_metric, (tf.Variable, tf.Tensor)):
            initial_valid_metric = initial_valid_metric.eval()
        smaller_is_better = valid_metric_smaller_is_better
        if smaller_is_better is None:
            smaller_is_better = not (
                    valid_metric_name.endswith('acc') or
                    valid_metric_name.endswith('accuracy')
            )

        # memorize the parameters
        self._param_vars = copy.copy(param_vars)
        self._var_groups = list(var_groups) if var_groups else None
        self._print_func = print_func
        self._show_eta = show_eta
        self._max_epoch = max_epoch
        self._max_step = max_step
        self._metric_formatter = metric_formatter

        self._summary_dir = summary_dir
        self._summary_writer = summary_writer
        self._summary_metric_prefix = summary_metric_prefix
        self._summary_graph = summary_graph
        self._summary_skip_pattern = summary_skip_pattern
        self._summary_commit_freqs = dict(summary_commit_freqs or ())
        self._own_summary_writer = own_summary_writer

        self._use_early_stopping = early_stopping
        self._valid_metric_name = valid_metric_name
        self._initial_valid_metric = initial_valid_metric
        self._valid_metric_smaller_is_better = smaller_is_better

        # the event source
        self._events = EventSource([
            EventKeys.ENTER_LOOP,
            EventKeys.EXIT_LOOP,
            EventKeys.BEFORE_EPOCH,
            EventKeys.AFTER_EPOCH,
            EventKeys.BEFORE_STEP,
            EventKeys.AFTER_STEP,
            EventKeys.METRICS_COLLECTED,
            EventKeys.TIME_METRICS_COLLECTED,
            EventKeys.SUMMARY_ADDED,
        ])

        # the restorable train loop states
        self._states = TrainLoopStates()

        # initialize the checkpoint saver
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_epoch_freq = checkpoint_epoch_freq
        self._restore_checkpoint = restore_checkpoint

        self._checkpoint_saver = None
        if checkpoint_dir:
            getLogger(__name__).debug(
                'Global variables to save at checkpoints: %s',
                tf.global_variables()
            )
            self._checkpoint_saver = CheckpointSaver(
                tf.global_variables() + [self._states] + save_objects,
                save_dir=checkpoint_dir,
                max_to_keep=checkpoint_max_to_keep,
                save_meta=False
            )

        # euphemeral train loop states
        self._eta = None
        self._step_metrics = None  # type: MetricLogger
        self._epoch_metrics = None  # type: MetricLogger
        self._early_stopping = None  # type: EarlyStopping
        self._within_epoch = False
        self._within_step = False
        self._steps_per_epoch = None  # average steps per epoch
        self._best_valid_metric = initial_valid_metric
        self._is_best_valid_metric = False
        self._epoch_start_time = None
        self._step_start_time = None

        # the active data flow of current epoch
        self._data_flow = None  # type: DataFlow
        self._step_data = None  # the data of the current step

    def _enter(self):
        # open the summary writer if required
        if self._summary_dir is not None:
            self._summary_writer = tf.summary.FileWriter(
                self._summary_dir, graph=self._summary_graph)

        # create the metric accumulators
        self._step_metrics = MetricLogger(formatter=self._metric_formatter)
        self._epoch_metrics = MetricLogger(
            summary_writer=self._summary_writer,
            summary_metric_prefix=self._summary_metric_prefix,
            summary_skip_pattern=self._summary_skip_pattern,
            summary_commit_freqs=self._summary_commit_freqs,
            formatter=self._metric_formatter
        )

        # open the early-stopping if required
        if self._use_early_stopping:
            self._early_stopping = EarlyStopping(
                self._param_vars,
                initial_metric=self._initial_valid_metric,
                smaller_is_better=self._valid_metric_smaller_is_better
            )
            self._early_stopping.__enter__()

        # restore the checkpoint
        if self._checkpoint_saver is not None:
            checkpoint_file = None
            if isinstance(self._restore_checkpoint, six.string_types):
                checkpoint_file = str(self._restore_checkpoint)
            elif self._restore_checkpoint:
                checkpoint_file = self._checkpoint_saver.latest_checkpoint()
            if checkpoint_file:
                self.println('Resume training from checkpoint: {}'.
                             format(checkpoint_file))
                self._checkpoint_saver.restore(checkpoint_file)

        # initialize the eta flags
        self._eta = ETA()

        # trigger the event
        self.events.on(EventKeys.ENTER_LOOP, self)

        # return self as the context object
        return self

    def _exit(self, exc_type, exc_val, exc_tb):
        try:
            # close the summary writer
            if self._own_summary_writer:
                self._summary_writer.close()
                self._summary_writer = None
                self._own_summary_writer = False

            # close the early-stopping context
            if self._early_stopping is not None:
                self._early_stopping.__exit__(exc_type, exc_val, exc_tb)
                self._early_stopping = None

        finally:
            # clear status
            self._steps_per_epoch = None
            self._eta = None

            # trigger the event
            self.events.on(EventKeys.EXIT_LOOP, self)

    def _commit_epoch_stop_time(self):
        if self._epoch_start_time is not None:
            duration = time.time() - self._epoch_start_time
            self.collect_metrics(metrics={EPOCH_TIME_METRIC: duration})
            self._epoch_start_time = None

    def _commit_step_stop_time(self):
        if self._step_start_time is not None:
            duration = time.time() - self._step_start_time
            self.collect_metrics(metrics={STEP_TIME_METRIC: duration})
            self._step_start_time = None

    def get_progress(self):
        """
        Get the progress of training.

        Returns:
            float or None: The progress in range ``[0, 1]``, or None if
                the progress cannot be estimated.
        """
        max_step = self.max_step
        if max_step is None and self.max_epoch is not None and \
                self._steps_per_epoch is not None:
            max_step = self.max_epoch * self._steps_per_epoch

        if max_step:
            if self._within_step and self._step_start_time is not None:
                # _step_start_time != None, indicating the step not finished
                return (self.step - 1.) / max_step
            else:
                return float(self.step) / max_step
        elif self.max_epoch is not None:
            if self._within_epoch and self._epoch_start_time is not None:
                # _epoch_start_time != None, indicating the epoch not finished
                return (self.epoch - 1.) / self.max_epoch
            else:
                return float(self.epoch) / self.max_epoch

    @property
    def param_vars(self):
        """Get the trainable parameter variables."""
        return self._param_vars

    @property
    def var_groups(self):
        """Get the variable groups."""
        return self._var_groups

    @property
    def max_epoch(self):
        """Get or set the max value for epoch counter."""
        return self._max_epoch

    @max_epoch.setter
    def max_epoch(self, value):
        self._max_epoch = int(value)

    @property
    def max_step(self):
        """Get or set the max value for global step counter."""
        return self._max_step

    @max_step.setter
    def max_step(self, value):
        self._max_step = int(value)

    @property
    def summary_writer(self):
        """Get the summary writer instance."""
        return self._summary_writer

    @property
    def events(self):
        """
        Get the event source.

        Returns:
            EventSource: The event source.
        """
        return self._events

    @property
    def epoch(self):
        """Get the epoch counter (starting from 1)."""
        return self._states.epoch

    @property
    def step(self):
        """Get the global step counter (starting from 1)."""
        return self._states.step

    @property
    def step_data(self):
        """Get the data of current step."""
        return self._step_data

    @property
    def use_early_stopping(self):
        """Whether or not to adopt early-stopping?"""
        return self._use_early_stopping

    @property
    def valid_metric_name(self):
        """Get the name of the validation metric."""
        return self._valid_metric_name

    @property
    def valid_metric_smaller_is_better(self):
        """Whether or not the smaller value is better for validation metric?"""
        return self._valid_metric_smaller_is_better

    @property
    def best_valid_metric(self):
        """Get the best valid metric."""
        return self._best_valid_metric

    def make_checkpoint(self):
        """
        Make a checkpoint.

        This method must be called within an eopch or a step context.
        For example::

            for epoch in loop.iter_epochs():
                for [x] in loop.iter_steps(train_data):
                    ...

                if epoch % 100 == 0:
                    loop.make_checkpoint()
        """
        if not self._checkpoint_saver:
            raise RuntimeError('Checkpoint directory is not configured.')
        self._checkpoint_saver.save(self._states.step)

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
                (self._max_epoch is None or self.epoch < self._max_epoch) and
                (self._max_step is None or self.step < self._max_step)
            )

        self._require_entered()
        if self._within_epoch:
            raise RuntimeError('Another epoch loop has been opened')
        try:
            while loop_condition():
                self._states.epoch += 1
                self._within_epoch = True
                self._epoch_start_time = time.time()

                self.events.fire(EventKeys.BEFORE_EPOCH, self)
                yield self.epoch
                self.events.reverse_fire(EventKeys.AFTER_EPOCH, self)

                self._commit_epoch_stop_time()
                self._steps_per_epoch = float(self.step) / self.epoch

                # do checkpoint if configured
                if self._checkpoint_epoch_freq is not None and \
                        self.epoch % self._checkpoint_epoch_freq == 0:
                    self.make_checkpoint()
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
            data_generator: Optional iterable data to be yielded at every step.
                This is required if `max_step` is not configured, so as to
                prevent an infinite step loop.

        Yields:
            int or (int, any): The global step counter (starting from 1), or
                the tuple of ``(step counter, batch data)`` if `data_generator`
                is specified.
        """
        def loop_condition():
            return self._max_step is None or self.step < self._max_step

        self._require_entered()
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
                if isinstance(data_generator, DataFlow):
                    data_flow = data_generator
                else:
                    def iter_factory():
                        if data_gen[0] is not None:
                            for batch in data_gen[0]:
                                yield batch
                        data_gen[0] = None  # force to use data_generator once

                    data_gen = [data_generator]
                    data_flow = DataFlow.iterator_factory(iter_factory)
                self._data_flow = data_flow

            while loop_condition():
                # prepare for the step data
                if self._data_flow is None:
                    yield_obj = self.step + 1
                    step_data = None
                else:
                    try:
                        step_data = self._data_flow.next_batch()
                    except StopIteration:
                        break
                    yield_obj = self.step + 1, step_data

                # yield this step
                self._states.step += 1
                self._within_step = True
                self._step_data = step_data
                self._step_start_time = time.time()

                self.events.fire(EventKeys.BEFORE_STEP, self)
                try:
                    yield yield_obj
                except StopIteration:  # pragma: no cover
                    # might be caused by call to ``data_flow.next_batch()``
                    break
                self.events.reverse_fire(EventKeys.AFTER_STEP, self)

                self._commit_step_stop_time()
        finally:
            self._within_step = False
            self._step_start_time = None
            self._data_flow = None
            self._step_data = None

    def _require_context(self):
        self._require_entered()
        if not self._within_epoch and not self._within_step:
            raise RuntimeError('An epoch or a step loop is expected, but '
                               'neither has been opened')

    @contextmanager
    def timeit(self, metric_name):
        """
        Open a context for timing.

        Args:
            metric_name (str): Store the timing result in metric of this name.
                Note that `metric_name` must end with ``time`` or ``timer``,
                otherwise by default the time values will not be formatted as
                human readable strings.
        """
        self._require_context()
        start_time = time.time()
        yield
        duration = time.time() - start_time
        self._collect_metrics(
            {metric_name: duration}, EventKeys.TIME_METRICS_COLLECTED)

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

    def _collect_metrics(self, metrics, event_key):
        self._require_context()

        # update the metrics
        self._epoch_metrics.collect_metrics(metrics, global_step=self.step)
        if self._within_step:
            self._step_metrics.collect_metrics(metrics, global_step=self.step)
        self.events.fire(event_key, self, metrics)

        # update the validation metric
        def update_valid_metric(d):
            v = d.get(self.valid_metric_name)
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

        if self.valid_metric_name:
            if metrics:
                update_valid_metric(metrics)

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
        if metrics is None:
            metrics = {}
        elif metrics is not None and not isinstance(metrics, dict):
            raise TypeError('`metrics` should be a dict')
        else:
            metrics = dict(metrics)
        metrics.update(kwargs)
        self._collect_metrics(metrics, EventKeys.METRICS_COLLECTED)

    def add_summary(self, summary):
        """
        Add a summary object, with ``self.step`` as `global_step`.

        Args:
            summary (tf.summary.Summary or bytes): TensorFlow summary object,
                or serialized summary.
        """
        self._require_entered()
        self._summary_writer.add_summary(summary, global_step=self.step)
        self.events.fire(EventKeys.SUMMARY_ADDED, self, summary)

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

            if not self._within_step and not self._within_epoch:
                self._require_context()
            tags = []
            if self._max_epoch != 1:
                tags.append(format_tag(self.epoch, self._max_epoch, 'Epoch'))
            tags.append(format_tag(self.step, self._max_step, 'Step'))
            if self._show_eta:
                progress = self.get_progress()
                if progress is not None:
                    eta = self._eta.get_eta(progress)
                    if eta is not None:
                        tags.append('ETA {}'.format(humanize_duration(eta)))
            message = '[{}] {}'.format(', '.join(tags), message)
        self._print_func(message)

    def print_training_summary(self):
        """
        Print the training summary.

        The training summary include the following content:

        1.   Execution environment.
        2.   Parameters to be optimized during training.
        """
        self._require_entered()
        self.println(summarize_variables(
            variables=self._param_vars,
            title='Trainable Parameters',
            other_variables_title='Other Parameters',
            groups=self.var_groups
        ))
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
        self._require_entered()
        metrics = None
        if self._within_step:
            self._commit_step_stop_time()
            metrics = self._step_metrics
        elif self._within_epoch:
            self._commit_epoch_stop_time()
            metrics = self._epoch_metrics
        else:
            self._require_context()

        best_mark = ' (*)' if self._is_best_valid_metric else ''
        self.println(metrics.format_logs() + best_mark, with_tag=True)
        self._is_best_valid_metric = False
        metrics.clear()


TrainLoopContext = TrainLoop  # legacy alias for TrainLoop


@deprecated('use :class:`TrainLoop` instead.', version='0.1')
def train_loop(*args, **kwargs):  # pragma: no cover
    return TrainLoop(*args, **kwargs)
