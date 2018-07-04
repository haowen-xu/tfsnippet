import warnings

from .evaluator import Evaluator, auto_batch_weight

__all__ = ['Validator']


class Validator(Evaluator):
    """
    Class to compute validation loss and other metrics.

    This class is a legacy class, which inherits :class:`Evaluator`.
    Use :class:`Evaluator` instead if you're writing new code.
    """

    def __init__(self, loop, metrics, inputs, data_flow, feed_dict=None,
                 time_metric_name='valid_time',
                 batch_weight_func=auto_batch_weight):  # pragma: no cover
        warnings.warn(
            'Validator is deprecated, use tfsnippet.trainer.Evaluator instead.',
            DeprecationWarning
        )
        super(Validator, self).__init__(
            loop=loop, metrics=metrics, inputs=inputs, data_flow=data_flow,
            feed_dict=feed_dict, time_metric_name=time_metric_name,
            batch_weight_func=batch_weight_func
        )
