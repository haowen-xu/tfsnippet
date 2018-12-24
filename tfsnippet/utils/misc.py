import os
import re
import threading
import time
from contextlib import contextmanager

import numpy as np

from .type_utils import is_integer

__all__ = [
    'humanize_duration', 'camel_to_underscore', 'maybe_close',
    'iter_files', 'ETA', 'ContextStack', 'validate_enum_arg',
    'validate_int_or_int_tuple_arg',
]


def humanize_duration(seconds, short_units=True):
    """
    Format specified time duration as human readable text.

    Args:
        seconds: Number of seconds of the time duration.
        short_units (bool): Whether or not to use short units
            ("d", "h", "m", "s") instead of long units
            ("day", "hour", "minute", "second")? (default :obj:`False`)

    Returns:
        str: The formatted time duration.
    """
    if short_units:
        units = [(86400, 'd', 'd'), (3600, 'h', 'h'),
                 (60, 'm', 'm'), (1, 's', 's')]
    else:
        units = [(86400, ' day', ' days'), (3600, ' hour', ' hours'),
                 (60, ' minute', ' minutes'), (1, ' second', ' seconds')]

    if seconds < 0:
        seconds = -seconds
        suffix = ' ago'
    else:
        suffix = ''

    pieces = []
    for uvalue, uname, uname_plural in units[:-1]:
        if seconds >= uvalue:
            val = int(seconds // uvalue)
            if val > 0:
                pieces.append(
                    '{:d}{}'.format(val, uname_plural if val > 1 else uname))
            seconds %= uvalue
    uname, uname_plural = units[-1][1:]
    if seconds > np.finfo(np.float64).eps:
        pieces.append('{:.4g}{}'.format(
            seconds, uname_plural if seconds > 1 else uname))
    elif not pieces:
        pieces.append('0{}'.format(uname))

    return ' '.join(pieces) + suffix


def camel_to_underscore(name):
    """Convert a camel-case name to underscore."""
    s1 = re.sub(CAMEL_TO_UNDERSCORE_S1, r'\1_\2', name)
    return re.sub(CAMEL_TO_UNDERSCORE_S2, r'\1_\2', s1).lower()


CAMEL_TO_UNDERSCORE_S1 = re.compile('([^_])([A-Z][a-z]+)')
CAMEL_TO_UNDERSCORE_S2 = re.compile('([a-z0-9])([A-Z])')


@contextmanager
def maybe_close(obj):
    """
    Enter a context, and if `obj` has ``.close()`` method, close it
    when exiting the context.

    Args:
        obj: The object maybe to close.

    Yields:
        The specified `obj`.
    """
    try:
        yield obj
    finally:
        if hasattr(obj, 'close'):
            obj.close()


def iter_files(root_dir, sep='/'):
    """
    Iterate through all files in `root_dir`, returning the relative paths
    of each file.  The sub-directories will not be yielded.
    Args:
        root_dir (str): The root directory, from which to iterate.
        sep (str): The separator for the relative paths.
    Yields:
        str: The relative paths of each file.
    """
    def f(parent_path, parent_name):
        for f_name in os.listdir(parent_path):
            f_child_path = parent_path + os.sep + f_name
            f_child_name = parent_name + sep + f_name
            if os.path.isdir(f_child_path):
                for s in f(f_child_path, f_child_name):
                    yield s
            else:
                yield f_child_name

    for name in os.listdir(root_dir):
        child_path = root_dir + os.sep + name
        if os.path.isdir(child_path):
            for x in f(child_path, name):
                yield x
        else:
            yield name


class ETA(object):
    """Class to help compute the Estimated Time Ahead (ETA)."""

    def __init__(self, take_initial_snapshot=True):
        """
        Construct a new :class:`ETA`.

        Args:
            take_initial_snapshot (bool): Whether or not to take the initial
                snapshot ``(0., time.time())``? (default :obj:`True`)
        """
        self._times = []
        self._progresses = []
        if take_initial_snapshot:
            self.take_snapshot(0.)

    def take_snapshot(self, progress, now=None):
        """
        Take a snapshot of ``(progress, now)``, for later computing ETA.

        Args:
            progress: The current progress, range in ``[0, 1]``.
            now: The current timestamp in seconds.  If not specified, use
                ``time.time()``.
        """
        if not self._progresses or progress - self._progresses[-1] > .001:
            # we only record the time and corresponding progress if the
            # progress has been advanced by 0.1%
            if now is None:
                now = time.time()
            self._progresses.append(progress)
            self._times.append(now)

    def get_eta(self, progress, now=None, take_snapshot=True):
        """
        Get the Estimated Time Ahead (ETA).

        Args:
            progress: The current progress, range in ``[0, 1]``.
            now: The current timestamp in seconds.  If not specified, use
                ``time.time()``.
            take_snapshot (bool): Whether or not to take a snapshot of
                the specified ``(progress, now)``? (default :obj:`True`)

        Returns:
            float or None: The remaining seconds, or :obj:`None` if
                the ETA cannot be estimated.
        """
        # TODO: Maybe we can have a better estimation algorithm here!
        if now is None:
            now = time.time()

        if self._progresses:
            time_delta = now - self._times[0]
            progress_delta = progress - self._progresses[0]
            progress_left = 1. - progress
            if progress_delta < 1e-7:
                return None
            eta = time_delta / progress_delta * progress_left
        else:
            eta = None

        if take_snapshot:
            self.take_snapshot(progress, now)

        return eta


class ContextStack(object):
    """
    A thread-local context stack for general purpose.

    Usage::

        stack = ContextStack()
        stack.push(dict())  # push an object to the top of thread local stack
        stack.top()[key] = value  # use the top object
        stack.pop()  # pop an object from the top of thread local stack
    """

    def __init__(self, initial_factory=None):
        """
        Construct a new instance of :class:`ContextStack`.

        Args:
            initial_factory (() -> any): If specified, fill the context stack
                with an initial object generated by this factory.
        """
        self._thread_local = threading.local()
        self._initial_factory = initial_factory

    @property
    def items(self):
        if not hasattr(self._thread_local, 'items'):
            items = []
            if self._initial_factory is not None:
                items.append(self._initial_factory())
            setattr(self._thread_local, 'items', items)
        return self._thread_local.items

    def push(self, context):
        self.items.append(context)

    def pop(self):
        self.items.pop()

    def top(self):
        items = self.items
        if items:
            return items[-1]


def validate_enum_arg(arg_name, arg_value, choices, nullable=False):
    """
    Validate the value of a enumeration argument.

    Args:
        arg_name: Name of the argument.
        arg_value: Value of the argument.
        choices: Valid choices of the argument value.
        nullable: Whether or not the argument can be None?

    Returns:
        The validated argument value.

    Raises:
        ValueError: If `arg_value` is not valid.
    """
    choices = tuple(choices)

    if not (nullable and arg_value is None) and (arg_value not in choices):
        raise ValueError('Invalid value for argument `{}`: expected to be one '
                         'of {!r}, but got {!r}.'.
                         format(arg_name, choices, arg_value))

    return arg_value


def validate_int_or_int_tuple_arg(arg_name, arg_value):
    """
    Validate an integer or a tuple of integers.

    Args:
        arg_name (str): Name of the argument.
        arg_value (int or Iterable[int]): An integer, or an iterable collection
            of integers, to be casted into tuples of integers.

    Returns:
        tuple[int]: The tuple of integers.
    """
    if is_integer(arg_value):
        arg_value = (arg_value,)
    else:
        try:
            arg_value = tuple(int(v) for v in arg_value)
        except (ValueError, TypeError):
            raise ValueError('Invalid value for argument `{}`: expected to be '
                             'a tuple of integers, but got {!r}.'.
                             format(arg_name, arg_value))
    return arg_value
