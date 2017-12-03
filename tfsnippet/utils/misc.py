import re

import numpy as np

__all__ = ['humanize_duration', 'docstring_inherit', 'camel_to_underscore']


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
        Decorator for a method.
    """
    def wrapper(method):
        method.__doc__ = from_method.__doc__
        return method
    return wrapper


def camel_to_underscore(name):
    """Convert a camel-case name to underscore."""
    s1 = re.sub(CAMEL_TO_UNDERSCORE_S1, r'\1_\2', name)
    return re.sub(CAMEL_TO_UNDERSCORE_S2, r'\1_\2', s1).lower()


CAMEL_TO_UNDERSCORE_S1 = re.compile('([^_])([A-Z][a-z]+)')
CAMEL_TO_UNDERSCORE_S2 = re.compile('([a-z0-9])([A-Z])')
