import pytest
import six
import tensorflow as tf

from tfsnippet.distributions import Normal, DistributionFactory
from tfsnippet.modules.bayes.utils import validate_distribution_factory

if six.PY2:
    LONG_MAX = long(1) << 63 - long(1)
else:
    LONG_MAX = 1 << 63 - 1


class ValidateDistributionFactoryTestCase(tf.test.TestCase):

    def test_validate_distribution_factory(self):
        # test distribution class
        factory = validate_distribution_factory(Normal, 'xyz')
        self.assertIsInstance(factory, DistributionFactory)
        self.assertIs(factory.distribution_class, Normal)
        self.assertEqual(factory.default_args, {})

        # test distribution factory
        factory_0 = Normal.factory(mean=0.)
        factory = validate_distribution_factory(factory_0, 'xyz')
        self.assertIs(factory, factory_0)

        # test error
        with pytest.raises(
                TypeError, match='xyz must be a subclass of `Distribution`, or '
                                 'an instance of `DistributionFactory`'):
            _ = validate_distribution_factory(object, 'xyz')
        with pytest.raises(
                TypeError, match='xyz must be a subclass of `Distribution`, or '
                                 'an instance of `DistributionFactory`'):
            _ = validate_distribution_factory(object(), 'xyz')
