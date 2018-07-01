import numpy as np

from tfsnippet.trainer import merge_feed_dict
from tfsnippet.utils import get_default_session_or_error

__all__ = ['collect_outputs']


def collect_outputs(outputs, inputs, data_flow, feed_dict=None, session=None):
    """
    Run TensorFlow graph by mini-batch and concat outputs from each batch.

    Args:
        outputs (Iterable[tf.Tensor]): Output tensors to be computed.
        inputs (Iterable[tf.Tensor]): Input placeholders.
        data_flow (DataFlow): Data flow to feed the input placeholders.
        feed_dict: Optional, additional feed dict.
        session: The TensorFlow session.  If not specified, use the
            default session.

    Returns:
        tuple[np.ndarray]: The concatenated outputs.
    """
    outputs = list(outputs)
    inputs = list(inputs)
    session = session or get_default_session_or_error()

    collected = [[] for _ in range(len(outputs))]
    for batch in data_flow:
        batch_feed_dict = merge_feed_dict(
            feed_dict,
            {k: v for (k, v) in zip(inputs, batch)}
        )
        for i, o in enumerate(session.run(outputs, feed_dict=batch_feed_dict)):
            collected[i].append(o)

    for i, batches in enumerate(collected):
        collected[i] = np.concatenate(batches, axis=0)
    return tuple(collected)
