# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
import six
import tensorflow as tf

from tfsnippet.utils import (humanize_duration,
                             StatisticsCollector,
                             get_default_session_or_error,
                             DocInherit)

__all__ = [
    'MetricFormatter',
    'DefaultMetricFormatter',
    'MetricLogger',
    'summarize_variables',
]


@DocInherit
class MetricFormatter(object):
    """
    Base class for a training metrics formatter.

    A training metric formatter determines the order of metrics, and the way
    to display the values of these metrics, in :class:`MetricLogger`.
    """

    def sort_metrics(self, names):
        """
        Sort the names of metrics.

        Args:
            names: Iterable metric names.

        Returns:
            list[str]: Sorted metric names.
        """
        raise NotImplementedError()

    def format_metric(self, name, value):
        """
        Format the value of specified metric.

        Args:
            name: Name of the metric.
            value: Value of the metric.

        Returns:
            str: Human readable string representation of the metric value.
        """
        raise NotImplementedError()


class DefaultMetricFormatter(MetricFormatter):
    """
    Default training metric formatter.

    This class sorts the metrics as follows:

    1.  The metrics are first divided into groups according to the suffices
        of their names as follows:

        1.  names ending with "time" or "timer" should come first
        2.  names ending with "loss" should come the second
        3.  names ending with "acc" or "accuracy" should come the third
        4.  other names should all come afterwards

    2.  The metrics are then sorted according to their names, within each group.

    The values of the metrics would be formatted into 6-digit real numbers,
    except for metrics with "time" or "timer" as suffices in their names,
    which would be formatted using :func:`~tfsnippet.utils.humanize_duration`.
    """

    def sort_metrics(self, names):
        def sort_key(name):
            if name.endswith('time') or name.endswith('timer'):
                return -3, name
            elif name.endswith('loss'):
                return -2, name
            elif name.endswith('acc') or name.endswith('accuracy'):
                return -1, name
            return 0, name

        return sorted(names, key=sort_key)

    def format_metric(self, name, value):
        if name.endswith('time') or name.endswith('timer'):
            return humanize_duration(float(value))
        else:
            return '{:.6g}'.format(float(value))


class MetricLogger(object):
    """
    Logger for the training metrics.

    This class provides convenient methods for logging training metrics,
    and for writing metrics onto disk via TensorFlow summary writer.
    The statistics of the metrics could be formatted into human readable
    strings via :meth:`format_logs`.

    An example of using this logger is:

    .. code-block:: python

        logger = MetricLogger(tf.summary.FileWriter(log_dir))
        global_step = 1

        for epoch in range(1, max_epoch+1):
            for batch in DataFlow.arrays(...):
                loss, _ = session.run([loss, train_op], ...)
                logger.collect_metrics({'loss': loss}, global_step)
                global_step += 1

            valid_loss = session.run([loss], ...)
            logger.collect_metrics({'valid_loss': valid_loss}, global_step)
            print('Epoch {}, step {}: {}'.format(
                epoch, global_step, logger.format_logs()))
            logger.clear()
    """

    def __init__(self, summary_writer=None, formatter=None):
        """
        Construct the :class:`MetricLogger`.

        Args:
            summary_writer: TensorFlow summary writer.
            formatter (MetricFormatter): Metric formatter for this logger.
                If not specified, will use an instance of
                :class:`DefaultMetricFormatter`.
        """
        if formatter is None:
            formatter = DefaultMetricFormatter()
        self._formatter = formatter
        self._summary_writer = summary_writer

        # accumulators for various metrics
        self._metrics = defaultdict(StatisticsCollector)

    def clear(self):
        """Clear all the metric statistics."""
        # Instead of calling ``self._metrics.clear()``, we reset every
        # collector object (so that they can be reused).
        # This may help reduce the time cost on GC.
        for k, v in six.iteritems(self._metrics):
            v.reset()

    def collect_metrics(self, metrics, global_step=None):
        """
        Collect the statistics of metrics.

        Args:
            metrics (dict[str, float or np.ndarray]): Dict from metrics names
                to their values.  For :meth:`format_logs`, there is no
                difference between calling :meth:`collect_metrics` only once,
                with an array of metric values; or calling
                :meth:`collect_metrics` multiple times, with one value at
                each time.  However, for the TensorFlow summary writer, only
                the mean of the metric values would be recorded, if calling
                :meth:`collect_metrics` with an array.
            global_step (int or tf.Variable or tf.Tensor): The global step
                counter. (optional)
        """
        tf_summary_values = []
        for k, v in six.iteritems(metrics):
            v = np.asarray(v)
            self._metrics[k].collect(v)

            if self._summary_writer is not None:
                mean_value = v.mean()
                tf_summary_values.append(
                    tf.summary.Summary.Value(tag=k, simple_value=mean_value))

        if tf_summary_values:
            summary = tf.summary.Summary(value=tf_summary_values)
            if global_step is not None and \
                    isinstance(global_step, (tf.Variable, tf.Tensor)):
                global_step = get_default_session_or_error().run(global_step)
            self._summary_writer.add_summary(summary, global_step=global_step)

    def format_logs(self):
        """
        Format the metric statistics as human readable strings.

        Returns:
            str: The formatted metric statistics.
        """
        buf = []
        for key in self._formatter.sort_metrics(six.iterkeys(self._metrics)):
            metric = self._metrics[key]
            if metric.has_value:
                name = key.replace('_', ' ')
                val = self._formatter.format_metric(key, metric.mean)
                if metric.counter > 1:
                    std = ' (±{})'.format(
                        self._formatter.format_metric(key, metric.stddev))
                else:
                    std = ''
                buf.append('{}: {}{}'.format(name, val, std))
        return '; '.join(buf)


def summarize_variables(variables, title='Variables Summary'):
    """
    Get a formatted summary about the variables.

    Args:
        variables (list[tf.Variable] or dict[str, tf.Variable]): List or
            dict of variables to be summarized.
        title (str): Optional title of this summary.

    Returns:
        str: Formatted summary about the variables.
    """
    if isinstance(variables, dict):
        var_name, var_shape = list(zip(*sorted(six.iteritems(variables))))
        var_shape = [s.get_shape() for s in var_shape]
    else:
        variables = sorted(variables, key=lambda v: v.name)
        var_name = [v.name.rsplit(':', 1)[0] for v in variables]
        var_shape = [v.get_shape() for v in variables]

    var_count = [int(np.prod(s.as_list(), dtype=np.int32)) for s in var_shape]
    var_count_total = sum(var_count)
    var_shape = [str(s) for s in var_shape]
    var_count = [str(s) for s in var_count]

    buf = []
    if len(var_count) > 0:
        var_title = '{} ({:d} in total)'.format(title, var_count_total)

        var_name_len = max(map(len, var_name))
        var_shape_len = max(map(len, var_shape))
        var_count_len = max(map(len, var_count))
        var_table = []

        for name, shape, count in zip(var_name, var_shape, var_count):
            var_table.append(
                '{name:<{namelen}}  {shape:<{shapelen}}  '
                '{count}'.format(
                    name=name, shape=shape, count=count,
                    namelen=var_name_len, shapelen=var_shape_len,
                    countlen=var_count_len
                )
            )

        max_line_length = max(
            var_name_len + var_shape_len + var_count_len + 4,
            len(var_title)
        )
        buf.append(var_title)
        buf.append('-' * max_line_length)
        buf.extend(var_table)

    return '\n'.join(buf)
