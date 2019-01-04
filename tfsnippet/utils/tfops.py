import tensorflow as tf

from .doc_utils import add_name_arg_doc
from .shape_utils import int_shape
from .type_utils import is_tensor_object

__all__ = [
    'add_n_broadcast', 'smart_cond', 'get_variable_ddi',
    'assert_scalar_equal', 'assert_rank', 'assert_rank_at_least',
]


@add_name_arg_doc
def add_n_broadcast(tensors, name=None):
    """
    Add zero or many tensors with broadcasting.

    Args:
        tensors (Iterable[Tensor]): A list of tensors to be summed.

    Returns:
        tf.Tensor: The summed tensor.
    """
    tensors = [tf.convert_to_tensor(t) for t in tensors]
    if not tensors:
        raise ValueError('`tensors` must not be empty.')
    with tf.name_scope(name, default_name='add_n_broadcast', values=tensors):
        ret = tensors[0]
        for t in tensors[1:]:
            ret += t
        return ret


@add_name_arg_doc
def smart_cond(cond, true_fn, false_fn, name=None):
    """
    Execute `true_fn` or `false_fn` according to `cond`.

    Args:
        cond (bool or tf.Tensor): A bool constant or a tensor.
        true_fn (() -> tf.Tensor): The function of the true branch.
        false_fn (() -> tf.Tensor): The function of the false branch.

    Returns:
        tf.Tensor: The output tensor.
    """
    if is_tensor_object(cond):
        return tf.cond(cond, true_fn, false_fn, name=name)
    else:
        if cond:
            return true_fn()
        else:
            return false_fn()


def get_variable_ddi(name,
                     initial_value,
                     shape=None,
                     dtype=tf.float32,
                     initializing=False,
                     regularizer=None,
                     constraint=None,
                     trainable=True,
                     **kwargs):
    """
    Wraps :func:`tf.get_variable` to support data-dependent initialization.

    Args:
        name: Name of the variable.
        initial_value: The data-dependent initial value of the variable.
        shape: Shape of the variable.
        dtype: Data type of the variable.
        initializing (bool): Whether or not it is building the graph for
            data-dependent initialization? Ignored if `initial_value` is absent.
        regularizer: Regularizer of the variable.
        constraint: Constraint of the variable.
        trainable (bool): Whether or not to the variable is trainable?
        \\**kwargs: Other named parameters passed to :func:`tf.get_variable`.

    Returns:
        tf.Variable or tf.Tensor: The variable or the tensor.
    """
    # TODO: detect shape from `initial_value` if not specified
    v = tf.get_variable(
        name, shape=shape, dtype=dtype, regularizer=regularizer,
        constraint=constraint, trainable=trainable,
        **kwargs
    )
    if initializing:
        v = v.assign(initial_value)
    return v


def _make_assertion_error(expected, actual, message=None):
    ret = 'Assertion failed for {}: {}'.format(expected, actual)
    if message:
        ret += '; {}'.format(message)
    return AssertionError(ret)


@add_name_arg_doc
def assert_scalar_equal(a, b, message=None, name=None):
    """
    Assert 0-d scalar `a` == `b`.

    Args:
        a: A 0-d tensor.
        b: A 0-d tensor.
        message: Message to display when assertion failed.

    Returns:
        tf.Operation or None: The TensorFlow assertion operation,
            or None if can be statically asserted.
    """
    if not is_tensor_object(a) and not is_tensor_object(b):
        if a != b:
            raise _make_assertion_error(
                'a == b', '{!r} != {!r}'.format(a, b), message)
    else:
        return tf.assert_equal(a, b, message=message, name=name)


@add_name_arg_doc
def assert_rank(x, ndims, message=None, name=None):
    """
    Assert the rank of `x` is `ndims`.

    Args:
        x: A tensor.
        ndims (int or tf.Tensor): An integer, or a 0-d integer tensor.
        message: Message to display when assertion failed.

    Returns:
        tf.Operation or None: The TensorFlow assertion operation,
            or None if can be statically asserted.
    """
    if not is_tensor_object(ndims) and int_shape(x) is not None:
        ndims = int(ndims)
        x_ndims = len(int_shape(x))
        if x_ndims != ndims:
            raise _make_assertion_error(
                'rank(x) == ndims', '{!r} != {!r}'.format(x_ndims, ndims),
                message
            )
    else:
        return tf.assert_rank(x, ndims, message=message, name=name)


@add_name_arg_doc
def assert_rank_at_least(x, ndims, message=None, name=None):
    """
    Assert the rank of `x` is at least `ndims`.

    Args:
        x: A tensor.
        ndims (int or tf.Tensor): An integer, or a 0-d integer tensor.
        message: Message to display when assertion failed.

    Returns:
        tf.Operation or None: The TensorFlow assertion operation,
            or None if can be statically asserted.
    """
    x = tf.convert_to_tensor(x)
    if not is_tensor_object(ndims) and int_shape(x) is not None:
        ndims = int(ndims)
        x_ndims = len(int_shape(x))
        if x_ndims < ndims:
            raise _make_assertion_error(
                'rank(x) >= ndims', '{!r} < {!r}'.format(x_ndims, ndims),
                message
            )
    else:
        return tf.assert_rank_at_least(x, ndims, message=message, name=name)
