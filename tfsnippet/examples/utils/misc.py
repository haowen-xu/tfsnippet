import imageio
import numpy as np
import six

from tfsnippet.utils import is_integer

__all__ = [
    'validate_strides_or_kernel_size',
    'save_images_collection',
    'cached',
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
                raise ValueError('Unexpected image shape: {!r}'.
                                 format(img.shape))
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


def cached(method):
    """
    Decorate `method`, to cache its result.

    Args:
        method: The method whose result should be cached.

    Returns:
        The decorated method.
    """
    results = {}

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        cache_key = (args, tuple((k, kwargs[k]) for k, v in sorted(kwargs)))
        if cache_key not in results:
            results[cache_key] = method(*args, **kwargs)
        return results[cache_key]

    return wrapper
