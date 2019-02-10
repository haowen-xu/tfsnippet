import unittest

import numpy as np
import pytest

from tests.datasets.helper import skipUnlessRunDatasetsTests
from tfsnippet.datasets import *


class MnistTestCase(unittest.TestCase):

    @skipUnlessRunDatasetsTests()
    def test_fetch_mnist(self):
        # test normalize_x = False
        (train_x, train_y), (test_x, test_y) = load_mnist()
        self.assertTupleEqual(train_x.shape, (60000, 28, 28))
        self.assertTupleEqual(train_y.shape, (60000,))
        self.assertTupleEqual(test_x.shape, (10000, 28, 28))
        self.assertTupleEqual(test_y.shape, (10000,))

        self.assertGreater(np.max(train_x), 128.)

        # test normalize_x = True
        (train_x, train_y), (test_x, test_y) = load_mnist(normalize_x=True)
        self.assertTupleEqual(train_x.shape, (60000, 28, 28))
        self.assertTupleEqual(train_y.shape, (60000,))
        self.assertTupleEqual(test_x.shape, (10000, 28, 28))
        self.assertTupleEqual(test_y.shape, (10000,))

        self.assertLess(np.max(train_x), 1. + 1e-5)

        # test x_shape
        (train_x, train_y), (test_x, test_y) = load_mnist(x_shape=(784,))
        self.assertTupleEqual(train_x.shape, (60000, 784))
        self.assertTupleEqual(test_x.shape, (10000, 784))

        with pytest.raises(ValueError,
                           match='`x_shape` does not product to 784'):
            _ = load_mnist(x_shape=(1, 2, 3))
