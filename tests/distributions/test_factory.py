import pytest
import tensorflow as tf

from tfsnippet.distributions import DistributionFactory, Distribution


class _MyDistribution(Distribution):

    def __init__(self, *args, **kwargs):
        self.call_args = (args, kwargs)


class DistributionFactoryTestCase(tf.test.TestCase):

    def test_distribution_factory(self):
        # test type check
        with pytest.raises(
                TypeError, match='`distribution_class` must be a subclass of '
                                 '`Distribution`'):
            _ = DistributionFactory(1)
        with pytest.raises(
                TypeError, match='`distribution_class` must be a subclass of '
                                 '`Distribution`'):
            _ = DistributionFactory(str)

        # test no default args
        factory = DistributionFactory(_MyDistribution)
        self.assertEqual(factory().call_args, ((), {}))
        self.assertEqual(factory(a=1).call_args, ((), {'a': 1}))
        self.assertEqual(factory({'a': 1}).call_args, ((), {'a': 1}))
        self.assertEqual(factory({'a': 1}, a=2).call_args, ((), {'a': 2}))

        # test with default args
        factory = DistributionFactory(_MyDistribution, {'a': 3})
        self.assertEqual(factory().call_args, ((), {'a': 3}))
        self.assertEqual(factory(a=1).call_args, ((), {'a': 1}))
        self.assertEqual(factory({'a': 1}).call_args, ((), {'a': 1}))
        self.assertEqual(factory({'a': 1}, a=2).call_args, ((), {'a': 2}))
