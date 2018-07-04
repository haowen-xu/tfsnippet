import six
from zhusuan import distributions as zd

from .base import Distribution
from .wrapper import ZhuSuanDistribution

__all__ = ['DistributionFactory']


class DistributionFactory(object):
    """A factory for constructing a :class:`Distribution` instance."""

    def __init__(self, distribution_class, default_args=None):
        """
        Construct the :class:`DistributionFactory`.

        Args:
            distribution_class: The class of the distribution to be constructed,
                a subclass of :class:`Distribution`.
            default_args (dict[str, any]): Dict of default named arguments for
                constructing the distribution.
        """
        if not isinstance(distribution_class, six.class_types) or \
                (not issubclass(distribution_class, Distribution) and
                 not issubclass(distribution_class, zd.Distribution)):
            raise TypeError('`distribution_class` must be a subclass of '
                            '`Distribution` or '
                            '`zhusuan.distributions.Distribution`')
        self.distribution_class = distribution_class
        self.default_args = dict(default_args or ())

    def __call__(self, distribution_params=None, **kwargs):
        """
        Construct a :class:`Distribution`.

        Args:
            distribution_params (dict[str, any]): Dict of distribution
                parameters for constructing the distribution.  Usually used for
                consuming the output of a :class:`tfsnippet.modules.DictMapper`.
                Will override the default arguments in `default_args`.
            \**kwargs: Other named arguments for constructing the distribution.
                Will override the arguments in `default_args` and in
                `distribution_params`.

        Returns:
            Distribution: The constructed distribution instance.
        """
        merged_kwargs = dict(self.default_args)
        if distribution_params:
            merged_kwargs.update(distribution_params)
        if kwargs:
            merged_kwargs.update(kwargs)
        dist = self.distribution_class(**merged_kwargs)
        if issubclass(self.distribution_class, zd.Distribution):
            dist = ZhuSuanDistribution(dist)
        return dist
