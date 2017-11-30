import numpy as np

__all__ = ['humanize_duration', 'docstring_inherit']


def humanize_duration(seconds):
    """
    Format specified time duration as human readable text.

    Args:
        seconds: Number of seconds of the time duration.

    Returns:
        str: The formatted time duration.
    """
    if seconds < 0:
        seconds = -seconds
        suffix = ' ago'
    else:
        suffix = ''

    pieces = []
    for uvalue, uname in [(86400, 'day'),
                          (3600, 'hr'),
                          (60, 'min')]:
        if seconds >= uvalue:
            val = int(seconds // uvalue)
            if val > 0:
                if val > 1:
                    uname += 's'
                pieces.append('{:d} {}'.format(val, uname))
            seconds %= uvalue
    if seconds > np.finfo(np.float64).eps:
        pieces.append('{:.4g} sec{}'.format(
            seconds, 's' if seconds > 1 else ''))
    elif not pieces:
        pieces.append('0 sec')

    return ' '.join(pieces) + suffix


def docstring_inherit(from_method):
    """
    Inhert docstrings from specified `from_method`.

    Args:
        from_method: The original method, where docstring comes from.

    Returns:
        Decorated method.
    """
    def wrapper(method):
        method.__doc__ = from_method.__doc__
        return method
    return wrapper
