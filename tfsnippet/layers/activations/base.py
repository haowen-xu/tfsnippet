import tensorflow as tf

from tfsnippet.utils import (get_default_scope_name, camel_to_underscore,
                             add_name_and_scope_arg_doc)
from ..flows import BaseFlow

__all__ = ['InvertibleActivation', 'InvertibleActivationFlow']


class InvertibleActivation(object):
    """
    Base class for intertible activation functions.

    An invertible activation function is an element-wise transformation
    :math:`y = f(x)`, where its inverse function :math:`x = f^{-1}(y)`
    exists and can be explicitly computed.
    """

    def __call__(self, x):
        y, _ = self.transform(
            x=x, compute_y=True, compute_log_det=False,
            name=get_default_scope_name(
                camel_to_underscore(self.__class__.__name__))
        )
        return y

    def _transform(self, x, compute_y, compute_log_det):
        raise NotImplementedError()

    def transform(self, x, compute_y=True, compute_log_det=True,
                  value_ndims=0, name=None):
        """
        Transform `x` into `y`, and compute the log-determinant of `f` at `x`
        (i.e., :math:`\\log \\det \\frac{\\partial f(x)}{\\partial x}`).

        Args:
            x (Tensor): The samples of `x`.
            compute_y (bool): Whether or not to compute :math:`y = f(x)`?
                Default :obj:`True`.
            compute_log_det (bool): Whether or not to compute the
                log-determinant?  Default :obj:`True`.
            value_ndims (int): Number of value dimensions.
                `log_det.ndims == x.ndims - value_ndims`.
            name (str): If specified, will use this name as the TensorFlow
                operational name scope.

        Returns:
            (tf.Tensor, tf.Tensor): `y` and the (maybe summed) log-determinant.
                The items in the returned tuple might be :obj:`None`
                if corresponding `compute_?` argument is set to :obj:`False`.

        Raises:
            RuntimeError: If both `compute_y` and `compute_log_det` are set
                to :obj:`False`.
        """
        if not compute_y and not compute_log_det:
            raise ValueError('At least one of `compute_y` and '
                             '`compute_log_det` should be True.')

        value_ndims = int(value_ndims)
        if value_ndims < 0:
            raise ValueError('`value_ndims` must be >= 0: got {}'.
                             format(value_ndims))

        x = tf.convert_to_tensor(x)

        with tf.name_scope(
                name,
                default_name=get_default_scope_name('transform', self),
                values=[x]):
            y, log_det = self._transform(
                x=x, compute_y=compute_y, compute_log_det=compute_log_det)
            if log_det is not None and value_ndims > 0:
                log_det = tf.reduce_sum(
                    log_det, axis=list(range(-value_ndims, 0)))

            return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        raise NotImplementedError()

    def inverse_transform(self, y, compute_x=True, compute_log_det=True,
                          value_ndims=0, name=None):
        """
        Transform `y` into `x`, and compute the log-determinant of `f^{-1}` at
        `y` (i.e.,
        :math:`\\log \\det \\frac{\\partial f^{-1}(y)}{\\partial y}`).

        Args:
            y (Tensor): The samples of `y`.
            compute_x (bool): Whether or not to compute :math:`x = f^{-1}(y)`?
                Default :obj:`True`.
            compute_log_det (bool): Whether or not to compute the
                log-determinant?  Default :obj:`True`.
            value_ndims (int): Number of value dimensions.
                `log_det.ndims == y.ndims - value_ndims`.
            name (str): If specified, will use this name as the TensorFlow
                operational name scope.

        Returns:
            (tf.Tensor, tf.Tensor): `x` and the (maybe summed) log-determinant.
                The items in the returned tuple might be :obj:`None`
                if corresponding `compute_?` argument is set to :obj:`False`.

        Raises:
            RuntimeError: If both `compute_x` and `compute_log_det` are set
                to :obj:`False`.
            RuntimeError: If the flow is not explicitly invertible.
        """
        if not compute_x and not compute_log_det:
            raise ValueError('At least one of `compute_x` and '
                             '`compute_log_det` should be True.')

        value_ndims = int(value_ndims)
        if value_ndims < 0:
            raise ValueError('`value_ndims` must be >= 0: got {}'.
                             format(value_ndims))

        y = tf.convert_to_tensor(y)

        with tf.name_scope(
                name,
                default_name=get_default_scope_name('inverse_transform', self),
                values=[y]):
            x, log_det = self._inverse_transform(
                y=y, compute_x=compute_x, compute_log_det=compute_log_det)
            if log_det is not None and value_ndims > 0:
                log_det = tf.reduce_sum(
                    log_det, axis=list(range(-value_ndims, 0)))

            return x, log_det

    @add_name_and_scope_arg_doc
    def as_flow(self, value_ndims, name=None, scope=None):
        """
        Convert this activation object into a :class:`BaseFlow`.

        Args:
            value_ndims (int): Number of value dimensions in both `x` and `y`.
                `x.ndims - value_ndims == log_det.ndims` and
                `y.ndims - value_ndims == log_det.ndims`.

        Returns:
            InvertibleActivationFlow: The flow.
        """
        return InvertibleActivationFlow(
            activation=self, value_ndims=value_ndims, name=name, scope=scope)


class InvertibleActivationFlow(BaseFlow):
    """
    A flow that converts a :class:`InvertibleActivation` into a flow.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, activation, value_ndims, name=None, scope=None):
        """
        Construct a new :class:`InvertibleActivationFlow`.

        Args:
            activation (InvertibleActivation): The invertible activation object.
            value_ndims (int): Number of value dimensions in both `x` and `y`.
                `x.ndims - value_ndims == log_det.ndims` and
                `y.ndims - value_ndims == log_det.ndims`.
        """
        if not isinstance(activation, InvertibleActivation):
            raise TypeError('`activation` must be an instance of '
                            '`InvertibleActivation`: got {}'.format(activation))

        super(InvertibleActivationFlow, self).__init__(
            x_value_ndims=value_ndims,
            y_value_ndims=value_ndims,
            require_batch_dims=False,
            name=name,
            scope=scope,
        )
        self._activation = activation

    @property
    def value_ndims(self):
        """
        Get the number of value dimensions.

        Returns:
            int: The number of value dimensions.
        """
        assert(self.y_value_ndims == self.x_value_ndims)
        return self.x_value_ndims

    @property
    def activation(self):
        """
        Get the invertible activation object.

        Returns:
            InvertibleActivation: The invertible activation object.
        """
        return self._activation

    @property
    def explicitly_invertible(self):
        return True

    def _transform(self, x, compute_y, compute_log_det):
        return self._activation.transform(
            x=x, compute_y=compute_y, compute_log_det=compute_log_det,
            value_ndims=self.value_ndims,
        )

    def _inverse_transform(self, y, compute_x, compute_log_det):
        return self._activation.inverse_transform(
            y=y, compute_x=compute_x, compute_log_det=compute_log_det,
            value_ndims=self.value_ndims,
        )

    def _build(self, input=None):
        pass
