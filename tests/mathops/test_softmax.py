import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.mathops import *
from .utils import assert_op_result, naive_softmax


class LogOpsTestCase(tf.test.TestCase):

    def test_softmax(self):
        with self.test_session():
            x = np.linspace(0, 10, 1000).reshape([5, 10, 20])
            result = naive_softmax(x)
            assert_op_result(result, softmax, x)

    def test_log_softmax(self):
        with self.test_session():
            x = np.linspace(0, 10, 1000).reshape([5, 10, 20])
            result = naive_softmax(x)
            assert_op_result(np.log(result), log_softmax, x)


if __name__ == '__main__':
    unittest.main()
