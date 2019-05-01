import functools
import itertools

import numpy as np
import pytest
import tensorflow as tf

from tfsnippet.utils import *
from tests.helper import assert_variables


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


class InvertibleMatrixTestCase(tf.test.TestCase):

    def test_strict_mode(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, atol=1e-6, rtol=1e-5)
        tf.set_random_seed(1234)
        np.random.seed(1234)
        VarScopeRandomState.set_global_seed(0)

        with self.test_session() as sess:
            shape = (5, 5)
            m = InvertibleMatrix(shape, strict=True)
            self.assertTupleEqual(m.shape, (5, 5))
            assert_variables(['matrix'], exist=False, scope='invertible_matrix')
            assert_variables(['pre_L', 'pre_U', 'log_s'], trainable=True,
                             scope='invertible_matrix',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])
            assert_variables(['P', 'sign'], trainable=False,
                             scope='invertible_matrix',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

            ensure_variables_initialized()

            # check whether `P` is a permutation matrix
            P = sess.run(m._P)
            _ = PermutationMatrix(P)

            # check `L` is a lower triangular matrix and has unit diags
            pre_L, L = sess.run([m._pre_L, m._L])
            assert_allclose(
                pre_L * np.tril(np.ones(shape), k=-1) + np.eye(*shape),
                L
            )

            # check `U` is an upper triangular matrix and has `exp(s)` diags
            pre_U, sign, log_s, U = sess.run(
                [m._pre_U, m._sign, m._log_s, m._U])
            assert_allclose(
                (pre_U * np.triu(np.ones(shape), k=1) +
                 np.diag(sign * np.exp(log_s))),
                U
            )

            # check `matrix`, `inv_matrix` and `log_det`
            matrix, inv_matrix, log_det = \
                sess.run([m.matrix, m.inv_matrix, m.log_det])
            assert_allclose(matrix, np.dot(P, np.dot(L, U)))
            assert_allclose(inv_matrix, np.linalg.inv(matrix))
            assert_allclose(log_det, np.sum(log_s))

            # check whether or not `matrix` is orthogonal
            assert_allclose(np.transpose(matrix), inv_matrix)

        with tf.Graph().as_default():
            # test non-trainable
            _ = InvertibleMatrix(shape, strict=True, trainable=False)
            assert_variables(['pre_L', 'pre_U', 'log_s', 'P', 'sign'],
                             trainable=False, scope='invertible_matrix',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

    def test_non_strict_mode(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, atol=1e-6, rtol=1e-5)
        tf.set_random_seed(1234)
        np.random.seed(1234)
        VarScopeRandomState.set_global_seed(0)

        with self.test_session() as sess:
            m = InvertibleMatrix(5, strict=False)
            self.assertTupleEqual(m.shape, (5, 5))
            assert_variables(['matrix'], trainable=True,
                             scope='invertible_matrix',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])
            assert_variables(['pre_L', 'pre_U', 'log_s', 'P', 'sign'],
                             exist=False, scope='invertible_matrix')
            ensure_variables_initialized()

            # check `matrix`, `inv_matrix` and `log_det`
            matrix, inv_matrix, log_det = \
                sess.run([m.matrix, m.inv_matrix, m.log_det])
            assert_allclose(inv_matrix, np.linalg.inv(matrix))
            assert_allclose(log_det, np.linalg.slogdet(matrix)[1])

            # check whether or not `matrix` is orthogonal
            assert_allclose(np.transpose(matrix), inv_matrix)

            # ensure m.log_det can compute grad
            _ = tf.gradients(m.log_det, m.matrix)

        with tf.Graph().as_default():
            # test non-trainable
            _ = InvertibleMatrix(5, strict=False, trainable=False)
            assert_variables(['matrix'],
                             trainable=False, scope='invertible_matrix',
                             collections=[tf.GraphKeys.MODEL_VARIABLES])

    def test_errors(self):
        def check_shape_error(shape):
            with pytest.raises(ValueError,
                               match='`size` is not valid for a square matrix'):
                _ = InvertibleMatrix(shape)

        check_shape_error('')
        check_shape_error(0)
        check_shape_error((2, 3))
        check_shape_error((2, 2, 2))

    def test_reuse(self):
        @global_reuse
        def f():
            m = InvertibleMatrix((5, 5))
            ensure_variables_initialized()
            return sess.run(m.matrix), m._random_state.randint(10000)

        @global_reuse
        def g():
            m = InvertibleMatrix((5, 5))
            ensure_variables_initialized()
            return sess.run(m.matrix), m._random_state.randint(10000)

        tf.set_random_seed(1234)
        np.random.seed(1234)

        with self.test_session() as sess:
            m1, i1 = f()
            m2, i2 = f()
            m3, i3 = g()
            np.testing.assert_allclose(m2, m1)
            self.assertEqual(i2, i1)
            self.assertGreater(np.max(np.abs(m3 - m1)), 1e-4)
            self.assertNotEqual(i3, i1)
