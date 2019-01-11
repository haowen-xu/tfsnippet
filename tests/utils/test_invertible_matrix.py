import itertools

import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.utils import PermutationMatrix


class PermutationMatrixTestCase(tf.test.TestCase):

    def test_permutation(self):
        np.random.seed(1234)

        for size in (1, 2, 3, 5):
            for seq in itertools.permutations(range(size)):
                row_perm = tuple(seq)
                m = np.zeros(shape=[size, size], dtype=np.float32)
                m[range(size), row_perm] = 1.
                inv_m = np.transpose(m)

                # test construct from matrix
                p = PermutationMatrix(m)
                self.assertEqual(p.shape, (size, size))
                self.assertEqual(repr(p),
                                 'PermutationMatrix({!r})'.format(row_perm))
                np.testing.assert_equal(p.get_numpy_matrix(m.dtype), m)
                self.assertTupleEqual(p.row_permutation, row_perm)

                inv_p = PermutationMatrix(inv_m)
                self.assertEqual(inv_p.shape, (size, size))
                np.testing.assert_equal(inv_p.get_numpy_matrix(m.dtype), inv_m)
                self.assertTupleEqual(inv_p.col_permutation, row_perm)
                self.assertTupleEqual(p.col_permutation, inv_p.row_permutation)

                # test construct from row permutation indices
                p = PermutationMatrix(row_perm)
                self.assertEqual(p.shape, (size, size))
                self.assertEqual(repr(p),
                                 'PermutationMatrix({!r})'.format(row_perm))
                np.testing.assert_equal(p.get_numpy_matrix(m.dtype), m)
                self.assertTupleEqual(p.row_permutation, row_perm)
                self.assertTupleEqual(p.col_permutation, inv_p.row_permutation)

                # test left multiplication
                x = np.random.normal(size=[size, size * 2])
                np.testing.assert_allclose(p.left_mult(x), np.dot(m, x))

                with self.test_session() as sess:
                    np.testing.assert_allclose(
                        sess.run(p.left_mult(tf.convert_to_tensor(x))),
                        np.dot(m, x)
                    )

                # test right multiplication
                x = np.random.normal(size=[size * 2, size])
                np.testing.assert_allclose(p.right_mult(x), np.dot(x, m))

                with self.test_session() as sess:
                    np.testing.assert_allclose(
                        sess.run(p.right_mult(tf.convert_to_tensor(x))),
                        np.dot(x, m)
                    )

                # test inverse of the permutation matrix
                x = np.random.normal(size=[size, size])
                np.testing.assert_allclose(
                    p.inv().get_numpy_matrix(np.float32),
                    inv_m
                )
                np.testing.assert_allclose(inv_p.left_mult(p.left_mult(x)), x)
                np.testing.assert_allclose(p.left_mult(inv_p.left_mult(x)), x)
                np.testing.assert_allclose(inv_p.right_mult(p.right_mult(x)), x)
                np.testing.assert_allclose(p.right_mult(inv_p.right_mult(x)), x)

                # test determinant
                p = PermutationMatrix(m)
                self.assertEqual(p.det(), p.inv().det())
                self.assertEqual(p.det(), np.linalg.det(m))
                self.assertEqual(p.inv().det(), np.linalg.det(np.transpose(m)))

    def test_errors(self):
        # test construction error
        def check_construct_error(data):
            with pytest.raises(ValueError, match='`data` is not a valid '
                                                 'permutation matrix or row '
                                                 'permutation indices'):
                _ = PermutationMatrix(data)

        check_construct_error([])
        check_construct_error([[0]])
        check_construct_error([2])

        check_construct_error(np.eye(3, 2))
        check_construct_error(np.eye(3, 2))
        check_construct_error(np.zeros([3, 3]))
        check_construct_error(np.ones([3, 3], dtype=np.int32))
        check_construct_error(np.ones([2, 2]) * .5)

        # test left & right multiplication error
        p = PermutationMatrix(np.eye(3, 3))
        with pytest.raises(ValueError,
                           match=r'Cannot compute matmul\(self, input\)'):
            p.left_mult(np.random.normal(size=(4, 4)))
        with pytest.raises(ValueError,
                           match=r'Cannot compute matmul\(input, self\)'):
            p.right_mult(np.random.normal(size=(4, 4)))
