import six
from zhusuan import distributions as zd

from tfsnippet.distributions import Distribution, DistributionFactory

__all__ = ['validate_distribution_factory']


def validate_distribution_factory(factory, name):
    """
    Validate the specified distribution `factory` argument.

    Args:
        factory: A class object which is subclass of :class:`Distribution`
            or :class:`zhusuan.distributions.Distribution`,
            or an instance of :class:`DistributionFactory`.
        name (str): Name of the argument (in error message).

    Returns:
        DistributionFactory: A distribution factory, for constructing the
            desired distribution.

    Raises:
        TypeError: If neither of the above conditions is satisfied.
    """
    if isinstance(factory, six.class_types):
        if issubclass(factory, Distribution):
            factory = factory.factory()
        elif issubclass(factory, zd.Distribution):
            factory = DistributionFactory(factory)
    if not isinstance(factory, DistributionFactory):
        raise TypeError('{} must be a subclass of `Distribution` or '
                        '`zhusuan.distributions.Distribution`, or '
                        'an instance of `DistributionFactory`'.format(name))
    return factory
