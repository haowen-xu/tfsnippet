from .base import Distribution
from .wrapper import as_distribution

__all__ = ['BatchToValueDistribution']


class BatchToValueDistribution(Distribution):
    """
    Distribution that converts the last few `batch_ndims` into `values_ndims`.
    See :meth:`Distribution.batch_ndims_to_value` for more details.
    """

    def __init__(self, distribution, ndims):
        """
        Construct a new :class:`BatchToValueDistribution`.

        Args:
            distribution (Distribution): The source distribution.
            ndims (int): The last few `batch_ndims` to be converted
                into `value_ndims`.  Must be non-negative.
        """

        distribution = as_distribution(distribution)
        ndims = int(ndims)
        if ndims < 0:
            raise ValueError('`ndims` must be non-negative integers: '
                             'got {!r}'.format(ndims))

        self._distribution = distribution
        self._ndims = ndims
        self._value_ndims = ndims + distribution.value_ndims

    @property
    def base_distribution(self):
        """
        Get the base distribution.

        Returns:
            Distribution: The base distribution.
        """
        return self._distribution

    @property
    def value_ndims(self):
        return self._distribution.value_ndims + self._ndims

    def expand_value_ndims(self, ndims):
        ndims = int(ndims)
        if ndims == 0:
            return self
        return BatchToValueDistribution(
            self._distribution, ndims + self._ndims)

    batch_ndims_to_value = expand_value_ndims

    @property
    def dtype(self):
        return self._distribution.dtype

    @property
    def is_continuous(self):
        return self._distribution.is_continuous

    @property
    def is_reparameterized(self):
        return self._distribution.is_reparameterized

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        from tfsnippet.bayes import StochasticTensor
        group_ndims = int(group_ndims)
        t = self._distribution.sample(
            n_samples=n_samples,
            group_ndims=group_ndims + self._ndims,
            is_reparameterized=is_reparameterized,
            compute_density=compute_density,
            name=name
        )
        ret = StochasticTensor(
            distribution=self,
            tensor=t.tensor,
            n_samples=n_samples,
            group_ndims=group_ndims,
            is_reparameterized=t.is_reparameterized,
            log_prob=t._self_log_prob
        )
        ret._self_prob = t._self_prob
        return ret

    def log_prob(self, given, group_ndims=0, name=None):
        group_ndims = int(group_ndims)
        return self._distribution.log_prob(
            given=given,
            group_ndims=group_ndims + self._ndims,
            name=name
        )
