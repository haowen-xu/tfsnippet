from tfsnippet.utils import get_default_session_or_error
from .feed_dict import resolve_feed_dict

__all__ = ['auto_loss_weight', 'Validator']


def auto_loss_weight(batch_data):
    try:
        return batch_data[0].size()
    except Exception:
        return 1.


class Validator(object):
    """Class to run validation in training."""

    def __init__(self, loop, loss, inputs, data_flow, feed_dict=None,
                 time_metric_name='valid_time', loss_metric_name=None,
                 loss_weight_func=auto_loss_weight):
        self._loop = loop
        self._loss = loss
        self._inputs = list(inputs or ())
        self._data_flow = data_flow
        self._feed_dict = dict(feed_dict or ())
        self._time_metric_name = time_metric_name
        self._loss_metric_name = loss_metric_name or loop._valid_metric_name
        self._loss_weight_func = loss_weight_func

    @property
    def loop(self):
        return self._loop

    @property
    def loss(self):
        return self._loss

    @property
    def inputs(self):
        return self._inputs

    @property
    def data_flow(self):
        return self._data_flow

    @property
    def feed_dict(self):
        return self._feed_dict

    def _run_batch(self, session, feed_dict):
        return session.run(self.loss, feed_dict=feed_dict)

    def run(self, feed_dict=None):
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
                resolve_feed_dict(merged_feed_dict, inplace=True)

                # run the mini-batch
                loss = self._run_batch(session, merged_feed_dict)
                if self._loss_weight_func is not None:
                    loss_weight = self._loss_weight_func(batch_data)
                else:
                    loss_weight = 1.
                mc.collect(loss, weight=loss_weight)
