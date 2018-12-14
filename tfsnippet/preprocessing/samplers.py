import numpy as np

from tfsnippet.dataflow import DataMapper

__all__ = ['BernoulliSampler', 'UniformNoiseSampler']


class BernoulliSampler(DataMapper):
    """
    A :class:`DataMapper` which can sample 0/1 integers according to the
    input probability.  The input is assumed to be float numbers range within
    [0, 1) or [0, 1].
    """

    def __init__(self, dtype=np.int32, random_state=None):
        """
        Construct a new :class:`BernoulliSampler`.

        Args:
            dtype: The data type of the sampled array.  Default `np.int32`.
            random_state (RandomState): Optional numpy RandomState for sampling.
                (default :obj:`None`, use the global :class:`RandomState`).
        """
        self._dtype = dtype
        self._random_state = random_state

    @property
    def dtype(self):
        """Get the data type of the sampled array."""
        return self._dtype

    def _transform(self, x):
        rng = self._random_state or np.random
        sampled = np.asarray(
            rng.uniform(0., 1., size=x.shape) < x, dtype=self._dtype)
        return sampled,


class UniformNoiseSampler(DataMapper):
    """
    A :class:`DataMapper` which can add uniform noise onto the input array.
    The data type of the returned array will be the same as the input array,
    unless `dtype` is specified at construction.
    """

    def __init__(self, minval=0., maxval=1., dtype=None, random_state=None):
        """
        Construct a new :class:`UniformNoiseSampler`.

        Args:
            minval: The lower bound of the uniform noise (included).
            maxval: The upper bound of the uniform noise (excluded).
            dtype: The data type of the sampled array.  Default `np.int32`.
            random_state (RandomState): Optional numpy RandomState for sampling.
                (default :obj:`None`, use the global :class:`RandomState`).
        """
        self._minval = minval
        self._maxval = maxval
        self._dtype = np.dtype(dtype) if dtype is not None else None
        self._random_state = random_state

    @property
    def minval(self):
        """Get the lower bound of the uniform noise (included)."""
        return self._minval

    @property
    def maxval(self):
        """Get the upper bound of the uniform noise (excluded)."""
        return self._maxval

    @property
    def dtype(self):
        """Get the data type of the sampled array."""
        return self._dtype

    def _transform(self, x):
        rng = self._random_state or np.random
        dtype = self._dtype or x.dtype
        noise = rng.uniform(self._minval, self._maxval, size=x.shape)
        return np.asarray(x + noise, dtype=dtype),
