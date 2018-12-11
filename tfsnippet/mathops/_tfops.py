from contextlib import contextmanager

import tensorflow as tf

name_scope = tf.name_scope


def ensure_axis_arg(axis):
    """
    Ensure the type of ``axis`` can be accepted by other math operations.
    """
    return tf.convert_to_tensor(axis, dtype=tf.int32)


int32 = tf.int32
identity = tf.identity

abs = tf.abs
sign = tf.sign
log = tf.log
log1p = tf.log1p
exp = tf.exp
clip_by_value = tf.clip_by_value

convert_to_tensor = tf.convert_to_tensor
reduce_mean = tf.reduce_mean
reduce_sum = tf.reduce_sum
reduce_max = tf.reduce_max
reduce_min = tf.reduce_min
squeeze = tf.squeeze

range = tf.range
shape = tf.shape
rank = tf.rank
reshape = tf.reshape


@contextmanager
def assert_rank_at_least(x, rank, message=None):
    with tf.control_dependencies(
            [tf.assert_rank_at_least(x, rank, message=message)]):
        yield
