import numpy as np
import tensorflow as tf
from scipy import linalg as la

from .doc_utils import add_name_arg_doc, add_name_and_scope_arg_doc
from .random import VarScopeRandomState
from .reuse import VarScopeObject
from .scope import reopen_variable_scope
from .tensor_spec import InputSpec
from .type_utils import is_integer

__all__ = ['PermutationMatrix', 'InvertibleMatrix']


class PermutationMatrix(object):
    """A non-trainable permutation matrix."""

    def __init__(self, data):
        """
        Construct a new :class:`PermutationMatrix`.

        Args:
            data (np.ndarray or Iterable[int]): A 2-d permutation matrix,
                or a list of integers as the row permutation indices.
        """
        def validate_data(data):
            if isinstance(data, np.ndarray) and len(data.shape) == 2:
                try:
                    epsilon = np.finfo(data.dtype).eps
                except ValueError:
                    epsilon = 0

                # check whether or not `data` is a permutation matrix
                if data.shape[0] != data.shape[1] or data.shape[0] < 1:
                    raise ValueError()

                for axis in (0, 1):
                    axis_sum = np.sum(data, axis=axis)
                    if not np.all(np.abs(1 - axis_sum) <= epsilon):
                        raise ValueError()
                    axis_max = np.max(data, axis=axis)
                    if not np.all(np.abs(1 - axis_max) <= epsilon):
                        raise ValueError()

                # compute the row & col permutation indices
                row_perm = np.argmax(data, axis=1).astype(np.int32)
                col_perm = np.argmax(data, axis=0).astype(np.int32)
            else:
                # check whether or not `data` is row permutation indices
                data = np.asarray(data, dtype=np.int32)
                if len(data.shape) != 1 or len(data) < 1:
                    raise ValueError()
                if np.max(data) != len(data) - 1 or np.min(data) != 0 or \
                        len(np.unique(data)) != len(data):
                    raise ValueError()

                # compute the row & col permutation indices
                row_perm = data
                col_perm = [0] * len(data)
                for i, j in enumerate(row_perm):
                    col_perm[j] = i
                col_perm = np.asarray(col_perm, dtype=np.int32)

            return tuple(row_perm), tuple(col_perm)

        try:
            self._row_perm, self._col_perm = validate_data(data)
        except ValueError:
            raise ValueError('`data` is not a valid permutation matrix or '
                             'row permutation indices: {!r}'.format(data))
        self._shape = (len(self._row_perm),) * 2

        # compute the determinant
        det = 1
        for i in range(len(self._row_perm) - 1):
            for j in range(i+1, len(self._row_perm)):
                if self._row_perm[i] > self._row_perm[j]:
                    det = -det
        self._det = float(det)

    def __repr__(self):
        return 'PermutationMatrix({!r})'.format(self._row_perm)

    @property
    def shape(self):
        """
        Get the shape of this permutation matrix.

        Returns:
            (int, int): The shape of this permutation matrix.
        """
        return self._shape

    def det(self):
        """
        Get the determinant of this permutation matrix.

        Returns:
            float: The determinant of this permutation matrix.
        """
        return self._det

    @property
    def row_permutation(self):
        """
        Get the row permutation indices.

        Returns:
            tuple[int]: The row permutation indices.
        """
        return self._row_perm

    @property
    def col_permutation(self):
        """
        Get the column permutation indices.

        Returns:
            tuple[int]: The column permutation indices.
        """
        return self._col_perm

    def get_numpy_matrix(self, dtype=np.int32):
        """
        Get the numpy permutation matrix.

        Args:
            dtype: The data type of the returned matrix.

        Returns:
            np.ndarray: A 2-d numpy matrix.
        """
        m = np.zeros(self.shape, dtype=dtype)
        m[range(self.shape[0]), self._row_perm] = 1
        return m

    @add_name_arg_doc
    def left_mult(self, input, name=None):
        """
        Left multiply to `input` matrix.

        `output = matmul(self, input)`

        Args:
            input (np.ndarray or tf.Tensor): The input matrix, whose
                shape must be ``(self.shape[1], ?)``.

        Returns:
            np.ndarray or tf.Tensor: The result of multiplication.
        """
        # fast routine: left multiply to a numpy matrix
        if isinstance(input, np.ndarray):
            if len(input.shape) != 2 or input.shape[0] != self.shape[1]:
                raise ValueError(
                    'Cannot compute matmul(self, input): shape mismatch; '
                    'self {!r} vs input {!r}'.format(self, input)
                )
            return input[self._row_perm, :]

        # slow routine: left multiply to a TensorFlow matrix
        input = InputSpec(shape=(self.shape[1], '?')).validate(input)
        return tf.gather(input, indices=self._row_perm, axis=0,
                         name=name or 'left_mult')

    @add_name_arg_doc
    def right_mult(self, input, name=None):
        """
        Right multiply to `input` matrix.

        `output = matmul(input, self)`

        Args:
            input (np.ndarray or tf.Tensor): The input matrix, whose
                shape must be ``(?, self.shape[0])``.

        Returns:
            np.ndarray or tf.Tensor: The result of multiplication.
        """
        # fast routine: right multiply to a numpy matrix
        if isinstance(input, np.ndarray):
            if len(input.shape) != 2 or input.shape[1] != self.shape[0]:
                raise ValueError(
                    'Cannot compute matmul(input, self): shape mismatch; '
                    'input {!r} vs self {!r}'.format(input, self)
                )
            return input[:, self._col_perm]

        # slow routine: right multiply to a TensorFlow matrix
        input = InputSpec(shape=('?', self.shape[0])).validate(input)
        return tf.gather(input, indices=self._col_perm, axis=1,
                         name=name or 'right_mult')

    def inv(self):
        """
        Get the inverse permutation matrix of this matrix.

        Returns:
            PermutationMatrix: The inverse permutation matrix.
        """
        return PermutationMatrix(self._col_perm)


class InvertibleMatrix(VarScopeObject):
    """
    A partially trainable invertible matrix.

    This class composes the invertible matrix by a variant of PLU
    decomposition, proposed in (Kingma & Dhariwal, 2018), as follows:

    .. math::

        \\mathbf{M} = \\mathbf{P} \\mathbf{L} (\\mathbf{U} +
            \\mathrm{diag}(\\mathbf{sign} \\odot \\exp(\\mathbf{s})))

    where `P` is a permutation matrix, `L` is a lower triangular matrix
    with all its diagonal elements equal to one, `U` is an upper triangular
    matrix with all its diagonal elements equal to zero, `sign` is a vector
    of `{-1, 1}`, and `s` is a vector.

    `P` and `sign` are fixed variables, while `L`, `U`, `s` are trainable
    variables.  They are initialized via the following method:

    .. code-block:: python

        Q, _ = scipy.linalg.qr(
            random_state.normal(loc=0., scale=1., size=shape))
        P, L, U = scipy.linalg.lu(Q)
        sign = np.sign(np.diag(U))
        s = np.log(np.abs(np.diag(U)))
        U = np.triu(U, k=1)

    The `random_state` can be specified via the constructor.  If it is not
    specified, an instance of :class:`VarScopeRandomState` will be created
    according to the variable scope name of the object.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, size, epsilon=1e-6, dtype=tf.float32, random_state=None,
                 name=None, scope=None):
        # validate the shape
        if is_integer(size):
            shape = (int(size),) * 2
        else:
            h, w = size
            shape = (int(h), int(w))
        self._shape = shape

        # initialize the variable scope and the random state
        super(InvertibleMatrix, self).__init__(name=name, scope=scope)
        if random_state is None:
            random_state = VarScopeRandomState(self.variable_scope)
        self._random_state = random_state

        # generate random matrix
        np_Q, _ = la.qr(random_state.normal(loc=0., scale=1., size=shape))
        np_P, np_L, np_U = la.lu(np_Q)
        np_s = np.diag(np_U)
        np_sign = np.sign(np_s)
        np_log_s = np.log(np.maximum(np.abs(np_s), epsilon))
        np_U = np.triu(np_U, k=1)

        # create the variables and random initialize the matrix
        with reopen_variable_scope(self.variable_scope):
            self._P = PermutationMatrix(np_P)
            self._L = tf.get_variable('L', initializer=np_L, dtype=dtype)
            self._U = tf.get_variable('U', initializer=np_U, dtype=dtype)
            self._sign = tf.get_variable('sign', initializer=np_sign,
                                         dtype=dtype, trainable=False)
            self._log_s = tf.get_variable('log_s', initializer=np_log_s,
                                          dtype=dtype)


