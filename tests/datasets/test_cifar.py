import unittest

import numpy as np
import pytest

from tests.datasets.helper import skipUnlessRunDatasetsTests
from tfsnippet.datasets import *


class CifarTestCase(unittest.TestCase):

    @skipUnlessRunDatasetsTests()
    def test_fetch_cifar10(self):
        # test channels_last = True, normalize_x = False
        (train_x, train_y), (test_x, test_y) = load_cifar10()
        self.assertTupleEqual(train_x.shape, (50000, 32, 32, 3))
        self.assertTupleEqual(train_y.shape, (50000,))
        self.assertTupleEqual(test_x.shape, (10000, 32, 32, 3))
        self.assertTupleEqual(test_y.shape, (10000,))

        self.assertGreater(np.max(train_x), 128.)
        self.assertEqual(np.max(train_y), 9)

        # test channels_last = False, normalize_x = True
        (train_x, train_y), (test_x, test_y) = load_cifar10(channels_last=False,
                                                            normalize_x=True)
        self.assertTupleEqual(train_x.shape, (50000, 3, 32, 32))
        self.assertTupleEqual(train_y.shape, (50000,))
        self.assertTupleEqual(test_x.shape, (10000, 3, 32, 32))
        self.assertTupleEqual(test_y.shape, (10000,))

        self.assertLess(np.max(train_x), 1. + 1e-5)

        # test x_shape
        (train_x, train_y), (test_x, test_y) = load_cifar10(x_shape=(1024, 3))
        self.assertTupleEqual(train_x.shape, (50000, 1024, 3))
        self.assertTupleEqual(test_x.shape, (10000, 1024, 3))

        with pytest.raises(ValueError,
                           match='`x_shape` does not product to 3072'):
            _ = load_cifar10(x_shape=(1, 2, 3))

    @skipUnlessRunDatasetsTests()
    def test_fetch_cifar100(self):
        # test channels_last = True, normalize_x = False
        (train_x, train_y), (test_x, test_y) = load_cifar100()
        self.assertTupleEqual(train_x.shape, (50000, 32, 32, 3))
        self.assertTupleEqual(train_y.shape, (50000,))
        self.assertTupleEqual(test_x.shape, (10000, 32, 32, 3))
        self.assertTupleEqual(test_y.shape, (10000,))

        self.assertGreater(np.max(train_x), 128.)
        self.assertEqual(np.max(train_y), 99)

        # test channels_last = False, normalize_x = True
        (train_x, train_y), (test_x, test_y) = load_cifar100(
            label_mode='coarse', channels_last=False, normalize_x=True)
        self.assertTupleEqual(train_x.shape, (50000, 3, 32, 32))
        self.assertTupleEqual(train_y.shape, (50000,))
        self.assertTupleEqual(test_x.shape, (10000, 3, 32, 32))
        self.assertTupleEqual(test_y.shape, (10000,))

        self.assertLess(np.max(train_x), 1. + 1e-5)
        self.assertEqual(np.max(train_y), 19)

        # test x_shape
        (train_x, train_y), (test_x, test_y) = load_cifar100(x_shape=(1024, 3))
        self.assertTupleEqual(train_x.shape, (50000, 1024, 3))
        self.assertTupleEqual(test_x.shape, (10000, 1024, 3))

        with pytest.raises(ValueError,
                           match='`x_shape` does not product to 3072'):
            _ = load_cifar100(x_shape=(1, 2, 3))
