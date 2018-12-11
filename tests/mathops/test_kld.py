import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.mathops import *
from .utils import assert_op_result, naive_softmax


class KLDTestCase(tf.test.TestCase):

    def prepare_data(self):
        np.random.seed(1234)
        p_logits = np.random.uniform(0., 1., 1000).reshape([5, 10, 20])
        q_logits = np.random.uniform(0., 1., 1000).reshape([5, 10, 20])
        p_probs = naive_softmax(p_logits)
        q_probs = naive_softmax(q_logits)
        return (p_logits, p_probs), (q_logits, q_probs)

    def test_softmax_logits_kld(self):
        with self.test_session():
            (p_logits, p_probs), (q_logits, q_probs) = self.prepare_data()
            kld_keep = np.sum(p_probs * (np.log(p_probs) - np.log(q_probs)),
                              axis=-1, keepdims=True)
            kld = np.squeeze(kld_keep, axis=-1)

            assert_op_result(
                kld,
                softmax_logits_kld, p_logits, q_logits
            )
            assert_op_result(
                kld,
                softmax_logits_kld, p_logits, q_logits, keepdims=False
            )
            assert_op_result(
                kld_keep,
                softmax_logits_kld, p_logits, q_logits, keepdims=True
            )

    def test_softmax_probs_kld(self):
        with self.test_session():
            (p_logits, p_probs), (q_logits, q_probs) = self.prepare_data()
            kld_keep = np.sum(p_probs * (np.log(p_probs) - np.log(q_probs)),
                              axis=-1, keepdims=True)
            kld = np.squeeze(kld_keep, axis=-1)

            assert_op_result(
                kld,
                softmax_probs_kld, p_probs, q_probs
            )
            assert_op_result(
                kld,
                softmax_probs_kld, p_probs, q_probs, keepdims=False
            )
            assert_op_result(
                kld_keep,
                softmax_probs_kld, p_probs, q_probs, keepdims=True
            )


if __name__ == '__main__':
    unittest.main()
