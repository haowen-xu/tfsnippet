import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.mathops import *
from .utils import assert_op_result


class LogExpTestCase(tf.test.TestCase):

    def test_log_sum_exp(self):
        with self.test_session():
            x = np.linspace(0, 10, 1000).reshape([5, 10, 20])
            assert_op_result(
                np.log(np.sum(np.exp(x), axis=-1, keepdims=False)),
                log_sum_exp, x, axis=-1
            )
            assert_op_result(
                np.log(np.sum(np.exp(x), axis=(0, 2), keepdims=False)),
                log_sum_exp, x, axis=(0, 2), keepdims=False
            )
            assert_op_result(
                np.log(np.sum(np.exp(x), axis=(0, 2), keepdims=True)),
                log_sum_exp, x, axis=(0, 2), keepdims=True
            )
            assert_op_result(
                np.log(np.sum(np.exp(x), axis=None, keepdims=False)),
                log_sum_exp, x, axis=None, keepdims=False
            )
            assert_op_result(
                np.log(np.sum(np.exp(x), axis=None, keepdims=True)),
                log_sum_exp, x, axis=None, keepdims=True
            )

    def test_log_mean_exp(self):
        with self.test_session():
            x = np.linspace(0, 10, 1000).reshape([5, 10, 20])
            assert_op_result(
                np.log(np.mean(np.exp(x), axis=-1, keepdims=False)),
                log_mean_exp, x, axis=-1
            )
            assert_op_result(
                np.log(np.mean(np.exp(x), axis=(0, 2), keepdims=False)),
                log_mean_exp, x, axis=(0, 2), keepdims=False
            )
            assert_op_result(
                np.log(np.mean(np.exp(x), axis=(0, 2), keepdims=True)),
                log_mean_exp, x, axis=(0, 2), keepdims=True
            )
            assert_op_result(
                np.log(np.mean(np.exp(x), axis=None, keepdims=False)),
                log_mean_exp, x, axis=None, keepdims=False
            )
            assert_op_result(
                np.log(np.mean(np.exp(x), axis=None, keepdims=True)),
                log_mean_exp, x, axis=None, keepdims=True
            )


if __name__ == '__main__':
    unittest.main()
