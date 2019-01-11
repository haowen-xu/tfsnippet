from tfsnippet.utils import InvertibleMatrix
from .base import BaseFlow

__all__ = ['InvertibleConv2d']


class InvertibleConv2d(BaseFlow):
    """
    Invertible 1x1 2D convolution proposed in (Kingma & Dhariwal, 2018).
    """

    def __init__(self,
                 channels_last=True,
                 strict_invertible=False,
                 random_state=None,
                 trainable=True,
                 name=None,
                 scope=None):
        self._channels_last = bool(channels_last)
        self._strict_invertible = bool(strict_invertible)
        self._random_state = random_state
        self._trainable = trainable

        super(InvertibleConv2d, self).__init__(
            value_ndims=3,
            name=name,
            scope=scope,
        )
