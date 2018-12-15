import unittest

import numpy as np

from tfsnippet.preprocessing import *


class BaseSamplerTestCase(unittest.TestCase):

    def test_sample(self):
        class _MySampler(BaseSampler):
            def sample(self, x):
                return x

        sampler = _MySampler()
        x = np.arange(12).reshape([3, 4])
        self.assertIs(sampler.sample(x), x)
        self.assertEqual(sampler(x), (x,))


class BernoulliSamplerTestCase(unittest.TestCase):

    def test_property(self):
        self.assertEqual(BernoulliSampler().dtype, np.int32)
        self.assertEqual(BernoulliSampler(np.float64).dtype, np.float64)

    def test_sample(self):
        np.random.seed(1234)
        x = np.linspace(0, 1, 1001, dtype=np.float32)

        # test output is int32 arrays
        sampler = BernoulliSampler()
        y = sampler.sample(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.int32)
        self.assertLessEqual(np.max(y), 1)
        self.assertGreaterEqual(np.min(y), 0)

        # test output is float32 arrays
        sampler = BernoulliSampler(dtype=np.float32)
        y = sampler.sample(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)
        self.assertLessEqual(np.max(y), 1 + 1e-5)
        self.assertGreaterEqual(np.min(y), 0 - 1e-5)


class UniformNoiseSamplerTestCase(unittest.TestCase):

    def test_property(self):
        sampler = UniformNoiseSampler()
        self.assertIsNone(sampler.dtype)

        sampler = UniformNoiseSampler(minval=-2., maxval=2., dtype=np.float64)
        self.assertEqual(sampler.minval, -2.)
        self.assertEqual(sampler.maxval, 2.)
        self.assertEqual(sampler.dtype, np.float64)

    def test_sample(self):
        np.random.seed(1234)
        x = np.arange(0, 1000, dtype=np.float64)

        # test output dtype equals to input
        sampler = UniformNoiseSampler()
        y = sampler.sample(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float64)
        self.assertLess(np.max(y - x), 1.)
        self.assertGreaterEqual(np.min(y - x), 0.)

        # test output is float32 arrays, and min&max val is not 0.&1.
        x = x * 4
        sampler = UniformNoiseSampler(minval=-2., maxval=2., dtype=np.float32)
        y = sampler.sample(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)
        self.assertLess(np.max(y - x), 2.)
        self.assertGreaterEqual(np.min(y - x), -2.)
