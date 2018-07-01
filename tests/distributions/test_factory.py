import unittest

import numpy as np
import pytest
import tensorflow as tf
from zhusuan import distributions as zd

from tfsnippet.distributions import DistributionFactory, Distribution
from tfsnippet.distributions.wrapper import ZhuSuanDistribution


class _MyDistribution(Distribution):

    def __init__(self, *args, **kwargs):
        self.call_args = (args, kwargs)


class DistributionFactoryTestCase(tf.test.TestCase):

    def test_distribution_factory(self):
        # test type check
        with pytest.raises(
                TypeError, match='`distribution_class` must be a subclass of '
                                 '`Distribution` or '
                                 '`zhusuan.distributions.Distribution`'):
            _ = DistributionFactory(1)
        with pytest.raises(
                TypeError, match='`distribution_class` must be a subclass of '
                                 '`Distribution` or '
                                 '`zhusuan.distributions.Distribution`'):
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

        # test zhusuan distribution
        factory = DistributionFactory(zd.Normal, {'mean': 1.})
        dist = factory({'std': 2.})
        self.assertIsInstance(dist, ZhuSuanDistribution)
        self.assertIsInstance(dist._distribution, zd.Normal)
        with self.test_session() as sess:
            np.testing.assert_equal(
                [1., 2.],
                sess.run([dist._distribution.mean, dist._distribution.std])
            )


if __name__ == '__main__':
    unittest.main()
