from collections import OrderedDict

import numpy as np
import tensorflow as tf

from tfsnippet.trainer import resolve_feed_dict, merge_feed_dict
from tfsnippet.utils import validate_enum_arg, get_default_session_or_error

__all__ = ['collect_outputs']


def collect_outputs(outputs, inputs, data_flow, mode='concat', axis=0,
                    feed_dict=None, session=None):
    """
    Run TensorFlow nodes by mini-batch and collect outputs from each batch.

    Args:
        outputs (Iterable[tf.Tensor] or dict[str, tf.Tensor]): The output
            tensors to be computed.
        inputs (Iterable[tf.Tensor]): Input placeholders.
        data_flow (DataFlow): Data flow to feed the input placeholders.
        mode ({'concat', 'average'}): If "concat", will concatenate the outputs
            from each mini-batch.  If "average", the output from each batch
            must be a scalar, and if so, this method will take average of the
            outputs from each mini-batch, weighted according to the batch size.
        axis (int): The axis for concatenation.
        feed_dict: Optional, additional feed dict.
        session: The TensorFlow session.  If not specified, use the
            default session.

    Returns:
        tuple[np.ndarray] or dict[str, tf.Tensor]: The collected outputs.
            Returns a dict if `outputs` is a dict, or a tuple otherwise.
    """
    mode = validate_enum_arg('mode', mode, ['concat', 'average'])
    session = session or get_default_session_or_error()

    if isinstance(outputs, (dict, OrderedDict)):
        output_keys = list(outputs)
        outputs = [tf.convert_to_tensor(outputs[k]) for k in output_keys]
    else:
        output_keys = None
        outputs = [tf.convert_to_tensor(o) for o in outputs]
    inputs = [tf.convert_to_tensor(i) for i in inputs]

    # check the shape of output tensors
    for i, o in enumerate(outputs):
        o_shape = o.get_shape()
        if mode == 'concat':
            if o_shape.ndims is not None and o_shape.ndims < 1:
                raise ValueError('`mode` is "concat", but the {}-th output '
                                 'is a scalar: {!r}'.format(i, o))
        else:
            if o_shape.ndims is not None and o_shape.ndims > 0:
                raise ValueError('`mode` is "average", but the {}-th output '
                                 'is not a scalar: {!r}'.format(i, o))

    collected = [[] for _ in range(len(outputs))]
    weights = []

    for batch in data_flow:
        weights.append(len(batch[0]))
        batch_feed_dict = merge_feed_dict(
            feed_dict,
            {k: v for (k, v) in zip(inputs, batch)}
        )
        batch_feed_dict = resolve_feed_dict(batch_feed_dict)
        for i, o in enumerate(session.run(outputs, feed_dict=batch_feed_dict)):
            collected[i].append(o)

    weights = np.asarray(weights, dtype=np.float32)
    for i, batches in enumerate(collected):
        if mode == 'average':
            stacked = np.stack(batches, axis=0)
            assert(len(stacked.shape) == 1)
            collected[i] = np.average(stacked, axis=0, weights=weights)
        else:
            collected[i] = np.concatenate(batches, axis=axis)

    if output_keys is not None:
        collected = dict(zip(output_keys, collected))
    else:
        collected = tuple(collected)
    return collected
