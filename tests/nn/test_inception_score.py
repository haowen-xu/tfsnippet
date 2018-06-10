import unittest

import numpy as np
import tensorflow as tf

from tfsnippet.nn import *
from .utils import assert_op_result, naive_softmax


class InceptionScoreTestCase(tf.test.TestCase):

    def prepare_data(self):
        np.random.seed(1234)
        logits = np.random.uniform(0., 1., 1000).reshape([5, 10, 20])
        probs = naive_softmax(logits)
        return logits, probs

    def naive_inception_score(self, probs, reduce_dims=None):
        if reduce_dims is None:
            reduce_dims = tuple(range(len(probs.shape) - 1))
        p_y_given_x = probs
        p_y = np.mean(p_y_given_x, axis=reduce_dims, keepdims=True)
        kld = np.sum(
            p_y_given_x * (np.log(p_y_given_x) - np.log(p_y)),
            axis=-1,
            keepdims=True
        )
        score = np.squeeze(np.mean(kld, axis=reduce_dims), axis=-1)
        return np.exp(score)

    def test_inception_score(self):
        with self.test_session():
            logits, probs = self.prepare_data()

            # by logits
            assert_op_result(
                self.naive_inception_score(probs),
                inception_score, logits=logits
            )
            assert_op_result(
                self.naive_inception_score(probs, reduce_dims=-2),
                inception_score, logits=logits, reduce_dims=-2
            )
            assert_op_result(
                self.naive_inception_score(probs, reduce_dims=(0, 1)),
                inception_score, logits=logits, reduce_dims=(0, 1)
            )

            # by probs
            assert_op_result(
                self.naive_inception_score(probs),
                inception_score, probs=probs
            )
            assert_op_result(
                self.naive_inception_score(probs, reduce_dims=-2),
                inception_score, probs=probs, reduce_dims=-2
            )
            assert_op_result(
                self.naive_inception_score(probs, reduce_dims=(0, 1)),
                inception_score, probs=probs, reduce_dims=(0, 1)
            )



if __name__ == '__main__':
    unittest.main()
