import tensorflow as tf

from tfsnippet.utils import is_tensor_object

__all__ = [
    'strides_tuple_to_channels_last',
    'input_maybe_to_channels_last',
    'output_maybe_to_channels_first',
]


def strides_tuple_to_channels_last(stride_tuples, channels_last=None,
                                   data_format=None):
    def to_tensor(x):
        x = list(x)
        if any(is_tensor_object(t) for t in x):
            return tf.stack(list(x))
        else:
            return tuple(x)

    if channels_last is None and data_format is None:
        raise ValueError('At least one of `channels_last` and `data_format` '
                         'should be specified.')

    if channels_last is False or data_format == 'NCHW':
        stride_tuples = tuple(
            to_tensor(strides[i] for i in (0, 2, 3, 1))
            for strides in stride_tuples
        )

    return stride_tuples


def input_maybe_to_channels_last(input, channels_last=None, data_format=None):
    if channels_last is None and data_format is None:
        raise ValueError('At least one of `channels_last` and `data_format` '
                         'should be specified.')

    if channels_last is False or data_format == 'NCHW':
        return tf.transpose(input, (0, 2, 3, 1))
    return input


def output_maybe_to_channels_first(output, channels_last=None,
                                   data_format=None):
    if channels_last is None and data_format is None:
        raise ValueError('At least one of `channels_last` and `data_format` '
                         'should be specified.')

    if channels_last is False or data_format == 'NCHW':
        return tf.transpose(output, (0, 3, 1, 2))
    return output
