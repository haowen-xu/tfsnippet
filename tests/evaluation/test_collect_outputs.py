import numpy as np
import pytest
import tensorflow as tf

from tfsnippet import DataFlow
from tfsnippet.evaluation import collect_outputs


class CollectOutputsTestCase(tf.test.TestCase):

    def test_collect_outputs_concat(self):
        with self.test_session() as sess:
            ph0 = tf.placeholder(dtype=tf.float32, shape=[])
            ph1 = tf.placeholder(dtype=tf.float32, shape=[None, 2])
            ph2 = tf.placeholder(dtype=tf.float32, shape=[None, 3])

            offset = np.random.normal(size=[]).astype(np.float32)
            arr1 = np.random.normal(size=[10, 2]).astype(np.float32)
            arr2 = np.random.normal(size=[10, 3]).astype(np.float32)

            # test 1 array, tuple
            df = DataFlow.arrays([arr1], batch_size=3)
            outputs = collect_outputs(
                [ph1 + offset], [ph1], df,
                mode='concat', feed_dict={ph0: offset}
            )
            self.assertIsInstance(outputs, tuple)
            self.assertEqual(len(outputs), 1)
            np.testing.assert_allclose(outputs[0], arr1 + offset)

            # test 1 array, dict
            df = DataFlow.arrays([arr1], batch_size=3)
            outputs = collect_outputs(
                {'output': ph1 + offset}, [ph1], df,
                mode='concat', feed_dict={ph0: offset}
            )
            self.assertIsInstance(outputs, dict)
            self.assertEqual(len(outputs), 1)
            np.testing.assert_allclose(outputs['output'], arr1 + offset)

            # test 2 arrays, tuple
            df = DataFlow.arrays([arr1, arr2], batch_size=3)
            outputs = collect_outputs(
                [ph1 + offset, ph2 + offset * 2], [ph1, ph2], df,
                mode='concat', feed_dict={ph0: offset}
            )
            self.assertIsInstance(outputs, tuple)
            self.assertEqual(len(outputs), 2)
            np.testing.assert_allclose(outputs[0], arr1 + offset)
            np.testing.assert_allclose(outputs[1], arr2 + offset * 2)

            # test 2 arrays, dict
            df = DataFlow.arrays([arr1, arr2], batch_size=3)
            outputs = collect_outputs(
                {'output1': ph1 + offset, 'output2': ph2 + offset * 2},
                [ph1, ph2], df,
                mode='concat', feed_dict={ph0: offset}
            )
            self.assertIsInstance(outputs, dict)
            self.assertEqual(len(outputs), 2)
            np.testing.assert_allclose(outputs['output1'], arr1 + offset)
            np.testing.assert_allclose(outputs['output2'], arr2 + offset * 2)

            with pytest.raises(ValueError,
                               match='`mode` is "concat", but the 0-th '
                                     'output is a scalar'):
                _ = collect_outputs(
                    [tf.reduce_mean(ph1)], [ph1], df, mode='concat')

            # test concat on axis 1
            df = DataFlow.arrays([arr1, arr2], batch_size=3)
            outputs = collect_outputs(
                [tf.concat(
                    [
                        tf.transpose(ph1, [1, 0]),
                        tf.transpose(ph2, [1, 0])
                    ],
                    axis=0
                )],
                [ph1, ph2], df, mode='concat', axis=1
            )
            ans = np.concatenate(
                [
                    np.transpose(arr1, [1, 0]),
                    np.transpose(arr2, [1, 0])
                ],
                axis=0
            )
            self.assertIsInstance(outputs, tuple)
            self.assertEqual(len(outputs), 1)
            np.testing.assert_allclose(outputs[0], ans)

    def test_collect_outputs_average(self):
        with self.test_session() as sess:
            ph0 = tf.placeholder(dtype=tf.float32, shape=[])
            ph1 = tf.placeholder(dtype=tf.float32, shape=[None])
            ph2 = tf.placeholder(dtype=tf.float32, shape=[None])

            offset = np.random.normal(size=[]).astype(np.float32)
            arr1 = np.random.normal(size=[10]).astype(np.float32)
            arr2 = np.random.normal(size=[10]).astype(np.float32)

            # test 1 array, tuple
            df = DataFlow.arrays([arr1], batch_size=3)
            outputs = collect_outputs(
                [tf.reduce_mean(ph1 + offset)], [ph1], df,
                mode='average', feed_dict={ph0: offset}
            )
            self.assertIsInstance(outputs, tuple)
            self.assertEqual(len(outputs), 1)
            np.testing.assert_allclose(outputs[0], np.mean(arr1 + offset))

            # test 2 arrays, dict
            df = DataFlow.arrays([arr1, arr2], batch_size=3)
            outputs = collect_outputs(
                {'output1': tf.reduce_mean(ph1 + offset),
                 'output2': tf.reduce_mean(ph2 + offset * 2)},
                [ph1, ph2], df,
                mode='average', feed_dict={ph0: offset}
            )
            self.assertIsInstance(outputs, dict)
            self.assertEqual(len(outputs), 2)
            np.testing.assert_allclose(
                outputs['output1'], np.mean(arr1 + offset))
            np.testing.assert_allclose(
                outputs['output2'], np.mean(arr2 + offset * 2))

            with pytest.raises(ValueError,
                               match='`mode` is "average", but the 0-th '
                                     'output is not a scalar'):
                _ = collect_outputs([ph1], [ph1], df, mode='average')
