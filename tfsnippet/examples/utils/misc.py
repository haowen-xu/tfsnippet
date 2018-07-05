import imageio
import numpy as np
import six
import tensorflow as tf
import zhusuan as zs

from tfsnippet.stochastic import StochasticTensor
from tfsnippet.trainer import BaseTrainer, Evaluator, AnnealingDynamicValue
from tfsnippet.utils import is_integer

__all__ = [
    'validate_strides_or_kernel_size',
    'check_epochs_and_steps_arg',
    'validate_after',
    'anneal_after',
    'save_images_collection',
    'isolate_graph',
    'get_batch_size',
    'int_shape',
    'is_dynamic_tensor',
    'smart_apply',
]


def validate_strides_or_kernel_size(arg_name, arg_value):
    """
    Validate the `strides` or `filter` arg, to ensure it is a tuple of
    two integers.

    Args:
        arg_name (str): The name of the argument, for formatting error.
        arg_value: The value of the argument.

    Returns:
        (int, int): The validated argument.
    """

    if not is_integer(arg_value) and (not isinstance(arg_value, tuple) or
                                      len(arg_value) != 2 or
                                      not is_integer(arg_value[0]) or
                                      not is_integer(arg_value[1])):
        raise TypeError('`{}` must be a int or a tuple (int, int).'.
                        format(arg_name))
    if not isinstance(arg_value, tuple):
        arg_value = (arg_value, arg_value)
    arg_value = tuple(int(v) for v in arg_value)
    return arg_value


def check_epochs_and_steps_arg(epochs=None, steps=None):
    """
    Check the argument `epochs` and `steps` to ensure one and only one
    of them is specified.
    """
    if (epochs is not None and steps is not None) or \
            (epochs is None and steps is None):
        raise ValueError('One and only one of `epochs` and `steps` should '
                         'be specified.')


def validate_after(trainer, evaluator, epochs=None, steps=None):
    """
    Run validation after every `epochs` or every `steps`.

    Args:
        trainer (BaseTrainer): The trainer object.
        evaluator (Evaluator): The evaluator object.
        epochs (None or int): Run validation after every `epochs`.
        steps (None or int): Run validation after every `steps`.

    Raises:
         ValueError: If both `epochs` and `steps` are specified, or
            neither is specified.
    """
    check_epochs_and_steps_arg(epochs, steps)
    if epochs is not None:
        return trainer.evaluate_after_epochs(evaluator, freq=epochs)
    else:
        return trainer.evaluate_after_steps(evaluator, freq=steps)


def anneal_after(trainer, value, epochs=None, steps=None):
    """
    Anneal dynamic value after every `epochs` or every `steps`.

    Args:
        trainer (BaseTrainer): The trainer object.
        value (AnnealingDynamicValue): The value to be annealed.
        epochs (None or int): Run validation after every `epochs`.
        steps (None or int): Run validation after every `steps`.

    Raises:
         ValueError: If both `epochs` and `steps` are specified, or
            neither is specified.
    """
    check_epochs_and_steps_arg(epochs, steps)
    if epochs is not None:
        return trainer.anneal_after_epochs(value, freq=epochs)
    else:
        return trainer.anneal_after_steps(value, freq=steps)


def save_images_collection(images, filename, grid_size, border_size=0,
                           channels_last=False):
    """
    Save a collection of images as a large image, arranged in grid.

    Args:
        images: The images collection.  Each element should be a Numpy array,
            in the shape of ``(H, W)``, ``(H, W, C)`` (if `channels_last` is
            :obj:`True`) or ``(C, H, W)``.
        filename (str): The target filename.
        grid_size ((int, int)): The ``(rows, columns)`` of the grid.
        border_size (int): Size of the border, for separating images.
            (default 0, no border)
        channels_last (bool): Whether or not the channel dimension is at last?
            (default :obj:`False`)
    """
    # check the arguments
    def validate_image(img):
        if len(img.shape) == 2:
            img = np.reshape(img, img.shape + (1,))
        elif len(images[0].shape) == 3:
            if img.shape[2 if channels_last else 0] not in (1, 3, 4):
                raise ValueError('Unexpected image shape: {!r}'.format(img))
            if not channels_last:
                img = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError('Unexpected image shape: {!r}'.format(img.shape))
        return img

    images = [validate_image(img) for img in images]
    h, w = images[0].shape[:2]
    rows, cols = grid_size[0], grid_size[1]
    buf_h = rows * h + (rows - 1) * border_size
    buf_w = cols * w + (cols - 1) * border_size

    # copy the images to canvas
    n_channels = images[0].shape[2]
    buf = np.zeros((buf_h, buf_w, n_channels), dtype=images[0].dtype)
    for j in range(rows):
        for i in range(cols):
            img = images[j * cols + i]
            buf[j * (h + border_size): (j + 1) * h + j * border_size,
                i * (w + border_size): (i + 1) * w + i * border_size,
                :] = img[:, :, :]

    # save the image
    if n_channels == 1:
        buf = np.reshape(buf, (buf_h, buf_w))
    imageio.imsave(filename, buf)


def isolate_graph(method):
    """
    Create an isolated :class:`tf.Graph` for the `method`.

    Args:
        method: The method to decorate.

    Returns:
        The decorated method.
    """
    @six.wraps(method)
    def wrapper(*args, **kwargs):
        with tf.Graph().as_default():
            return method(*args, **kwargs)
    return wrapper


def get_batch_size(input):
    """
    Infer the mini-batch size according to `input`.

    Args:
        input (tf.Tensor): The input placeholder.

    Returns:
        int or tf.Tensor: The batch size.
    """
    if input.get_shape() is None:
        batch_size = tf.shape(input)[0]
    else:
        batch_size = int_shape(input)[0]
        if batch_size is None:
            batch_size = tf.shape(input)[0]
    return batch_size


def int_shape(tensor):
    """
    Get the int shape tuple of specified `tensor`.

    Args:
        tensor: The tensor object.

    Returns:
        tuple[int or None]: The int shape tuple.
    """
    shape = tensor.get_shape().as_list()
    return tuple((int(v) if v is not None else None) for v in shape)


def is_dynamic_tensor(tensor):
    """
    Check whether or not `tensor` is a dynamic tensor.

    Args:
        tensor: The tensor to be checked.

    Returns:
        bool: Whether the tensor is a dynamic tensor.
    """
    return isinstance(tensor, (tf.Tensor, tf.Variable, StochasticTensor,
                               zs.StochasticTensor))


def smart_apply(tensor, static_fn, dynamic_fn):
    """
    Apply transformation on `tensor`, with either `static_fn` for static
    tensors (e.g., Numpy arrays, numbers) or `dynamic_fn` for dynamic
    tensors.

    Args:
        tensor: The tensor to be transformed.
        static_fn: Static transformation function.
        dynamic_fn: Dynamic transformation function.

    Returns:
        Tensor: The transformed tensor.
    """
    if isinstance(tensor, (tf.Tensor, tf.Variable, StochasticTensor,
                           zs.StochasticTensor)):
        return dynamic_fn(tensor)
    else:
        return static_fn(tensor)
