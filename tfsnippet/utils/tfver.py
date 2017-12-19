from distutils.version import StrictVersion

import tensorflow as tf

__all__ = ['is_tensorflow_version_higher_or_equal']


def is_tensorflow_version_higher_or_equal(version):
    """
    Check whether the version of TensorFlow is higher than `version`.

    Args:
        version (str): Expected version of TensorFlow.

    Returns:
        bool: True if higher or equal to, False if not.
    """
    return StrictVersion(version) <= StrictVersion(tf.__version__)
