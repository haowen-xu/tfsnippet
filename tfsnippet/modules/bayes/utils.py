import six

from tfsnippet.distributions import Distribution, DistributionFactory

__all__ = ['validate_distribution_factory']


def validate_distribution_factory(factory, name):
    """
    Validate the specified distribution `factory` argument.

    Args:
        factory: A class object which is subclass of :class:`Distribution`,
                 or an instance of :class:`DistributionFactory`.
        name (str): Name of the argument (in error message).

    Returns:
        DistributionFactory:
            ``factory.factory()`` if `factory` is a subclass of
            :class:`Distribution`, or the `factory` itself if it is
            already an instance of :class:`DistributionFactory`.

    Raises:
        TypeError: If neither of the above conditions is satisfied.
    """
    if isinstance(factory, six.class_types) and \
            issubclass(factory, Distribution):
        factory = factory.factory()
    if not isinstance(factory, DistributionFactory):
        raise TypeError('{} must be a subclass of `Distribution`, or '
                        'an instance of `DistributionFactory`'.format(name))
    return factory
