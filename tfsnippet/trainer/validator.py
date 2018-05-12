from tfsnippet.dataflow import DataFlow
from tfsnippet.utils import get_default_session_or_error
from tfsnippet.scaffold import TrainLoop

from .feed_dict import resolve_feed_dict

__all__ = ['auto_loss_weight', 'Validator']


def auto_loss_weight(*batch_arrays):
    """
    Automatically inspect the loss weight for a validation mini-batch.

    Args:
        *batch_arrays: Mini-batch arrays.  The ``.size`` of the first array
            will be used as the weight.

    Returns:
        The inspected loss weight, or 1. if any error occurs during inspection.
    """
    try:
        return batch_arrays[0].size
    except Exception:
        return 1.


class Validator(object):
    """
    Class to run validation by loss.
    """

    def __init__(self, loop, loss, inputs, data_flow, feed_dict=None,
                 time_metric_name='valid_time', loss_metric_name=None,
                 loss_weight_func=auto_loss_weight):
        """
        Construct a new :class:`Validator`.

        Args:
            loop (TrainLoop): The training loop object.
            loss (tf.Tensor): The validation loss.
            inputs (list[tf.Tensor]): The input placeholders.
                The number of tensors, and the order of tensors, should
                both match the arrays of each mini-batch data, provided
                by `data_flow`.
            data_flow (DataFlow): The validation data flow.
            feed_dict (dict[tf.Tensor, any]): The fixed feed dict for
                validation.  It will be merged with `inputs` and the
                argument of ``run(feed_dict)``. (default :obj:`None`)
            time_metric_name (str): The metric name for collecting
                validation time usage. (default "valid_time")
            loss_metric_name (str): The metric name for collecting
                validation loss. (default ``loop.valid_metric_name``)
            loss_weight_func ((*arrays) -> float or None): Specify how
                to compute the loss weight for each mini-batch.  If
                :obj:`None`, will use 1. as the loss weight.
                (default :func:`auto_loss_weight`)
        """
        self._loop = loop
        self._loss = loss
        self._inputs = list(inputs or ())
        self._data_flow = data_flow
        self._feed_dict = dict(feed_dict or ())
        self._time_metric_name = time_metric_name
        self._loss_metric_name = loss_metric_name or loop.valid_metric_name
        self._loss_weight_func = loss_weight_func

    @property
    def loop(self):
        """
        Get the training loop object.

        Returns:
            TrainLoop: The training loop object.
        """
        return self._loop

    @property
    def loss(self):
        """Get the validation loss."""
        return self._loss

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
        Get the validation data flow.

        Returns:
            DataFlow: The validation data flow.
        """
        return self._data_flow

    @property
    def feed_dict(self):
        """
        Get the fixed feed dict.

        Returns:
            dict[tf.Tensor, any]: The fixed feed dict.
        """
        return self._feed_dict

    @property
    def time_metric_name(self):
        """Get the metric name for collecting validation time usage."""
        return self._time_metric_name

    @property
    def loss_metric_name(self):
        """Get the metric name for collecting validation loss."""
        return self._loss_metric_name

    @property
    def loss_weight_func(self):
        """Get the function to compute the loss weight for each mini-batch."""
        return self._loss_weight_func

    def _run_batch(self, session, feed_dict):
        return session.run(self.loss, feed_dict=feed_dict)

    def run(self, feed_dict=None):
        """
        Run validation.

        Args:
            feed_dict (dict[tf.Tensor, any]): The extra feed dict to be
                merged with the already configured dict.  (default :obj:`None`)
        """
        session = get_default_session_or_error()
        merged_feed_dict = {}

        with self.loop.timeit(self._time_metric_name), \
                self.loop.metric_collector(self._loss_metric_name) as mc:
            for batch_data in self.data_flow:
                # prepare for the batch feed dict
                merged_feed_dict.clear()
                merged_feed_dict.update(self.feed_dict)
                if feed_dict is not None:
                    merged_feed_dict.update(feed_dict)
                for ph, val in zip(self.inputs, batch_data):
                    merged_feed_dict[ph] = val

                # run the mini-batch
                loss = self._run_batch(
                    session, resolve_feed_dict(merged_feed_dict))
                if self._loss_weight_func is not None:
                    loss_weight = self._loss_weight_func(*batch_data)
                else:
                    loss_weight = 1.
                mc.collect(loss, weight=loss_weight)
