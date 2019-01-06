import numpy as np
import tensorflow as tf

from tfsnippet.ops import *


class ClassificationAccuracyTestCase(tf.test.TestCase):

    def test_classification_accuracy(self):
        y_pred = np.asarray([0, 1, 3, 3, 2, 5, 4])
        y_true = np.asarray([0, 2, 3, 1, 3, 5, 5])
        acc = np.mean(y_pred == y_true)

        with self.test_session() as sess:
            self.assertAllClose(
                acc,
                sess.run(classification_accuracy(y_pred, y_true))
            )

    def test_softmax_classification_output(self):
        with self.test_session() as sess:
            np.random.seed(1234)

            # test 2d input, 1 class
            logits = np.random.random(size=[100, 1])
            ans = np.argmax(logits, axis=-1)
            np.testing.assert_equal(
                sess.run(softmax_classification_output(logits)),
                ans
            )

            # test 2d input, 5 classes
            logits = np.random.random(size=[100, 5])
            ans = np.argmax(logits, axis=-1)
            np.testing.assert_equal(
                sess.run(softmax_classification_output(logits)),
                ans
            )

            # test 3d input, 7 classes
            logits = np.random.random(size=[10, 100, 7])
            ans = np.argmax(logits, axis=-1)
            np.testing.assert_equal(
                sess.run(softmax_classification_output(logits)),
                ans
            )

