"""
Package for neural network math operations.

This package contains advanced math operations for training and evaluating
neural networks.  Most of the operations accept an ``ops`` as their first
argument.

You may pass :mod:`tfsnippet.mathops.npyops` or :mod:`tfsnippet.mathops.tfops`
as the ``ops`` argument to obtain a NumPy or TensorFlow math operation.
"""


from . import _npyops as npyops, _tfops as tfops
from . import inception_score, kld, log_exp, softmax

__all__ = ['npyops', 'tfops'] + sum(
    [m.__all__ for m in [
        inception_score, kld, log_exp, softmax
    ]],
    []
)

from .inception_score import *
from .kld import *
from .log_exp import *
from .softmax import *
