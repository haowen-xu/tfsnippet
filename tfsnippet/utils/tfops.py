from contextlib import contextmanager

import tensorflow as tf

from .debugging import maybe_assert
from .doc_utils import add_name_arg_doc
from .type_utils import is_tensor_object

__all__ = [
    'add_n_broadcast', 'smart_cond', 'control_deps', 'get_variable',
    'assert_scalar_equal',
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


@contextmanager
def control_deps(control_inputs):
    """
    A wrapper of :func:`tensorflow.control_dependencies`, where the :obj:`None`
    in specified `control_inputs` are filtered out.

    Args:
        control_inputs (Iterable[tf.Operation or None]): The operations to
            be executed, or :obj:`None`.

    Yields:
        bool: :obj:`True` if the `control_inputs` is not empty,
            :obj:`False` otherwise.
    """
    control_inputs = [o for o in control_inputs if o is not None]
    if control_inputs:
        with tf.control_dependencies(control_inputs):
            yield True
    else:
        yield False


def get_variable(name,
                 shape=None,
                 dtype=tf.float32,
                 initial_value=None,
                 initializing=False,
                 initializer=None,
                 regularizer=None,
                 constraint=None,
                 trainable=True,
                 **kwargs):
    """
    Wraps :func:`tf.get_variable` to support data-dependent initialization.

    Args:
        name: Name of the variable.
        shape: Shape of the variable.
        dtype: Data type of the variable.
        initial_value: The data-dependent initial value of the variable.
            Only one of `initial_value` and `initializer` can be specified.
        initializing (bool): Whether or not it is building the graph for
            data-dependent initialization? Ignored if `initial_value` is absent.
        initializer: Initializer of the variable
            Only one of `initial_value` and `initializer` can be specified.
        regularizer: Regularizer of the variable.
        constraint: Constraint of the variable.
        trainable (bool): Whether or not to the variable is trainable?
        \\**kwargs: Other named parameters passed to :func:`tf.get_variable`.

    Returns:
        tf.Variable or tf.Tensor: The variable or the tensor.
    """
    if initial_value is not None and initializer is not None:
        raise TypeError('`initial_value` and `initializer` cannot be both '
                        'specified.')

    v = tf.get_variable(
        name, shape=shape, dtype=dtype, initializer=initializer,
        regularizer=regularizer, constraint=constraint, trainable=trainable,
        **kwargs
    )

    if initial_value is not None and initializing:
        v = v.assign(initial_value)

    return v


@add_name_arg_doc
def assert_scalar_equal(a, b, message=None, name=None):
    """
    Assert 0-d scalar `a` == `b`.

    Args:
        a: A 0-d tensor.
        b: A 0-d tensor.
        message: Operational message when assertion failed.

    Returns:
        tf.Operation or None: The TensorFlow assertion operation,
            or None if `a` == `b` can be statically asserted.
    """
    if not is_tensor_object(a) and not is_tensor_object(b):
        if a != b:
            msg = 'Assertion failed for a == b: {!r} != {!r}'.format(a, b)
            if message:
                msg += '; {}'.format(message)
            raise ValueError(msg)
    else:
        return maybe_assert(
            tf.assert_equal,
            a, b, message=message, name=name
        )
