import tensorflow as tf

from tfsnippet.layers.flows.utils import assert_log_det_shape_matches_input
from tfsnippet.ops import add_n_broadcast
from tfsnippet.utils import (DocInherit, add_name_and_scope_arg_doc,
                             get_default_scope_name, assert_deps,
                             get_static_shape)
from ..base import BaseLayer

__all__ = ['BaseFlow', 'MultiLayerFlow']


@DocInherit
class BaseFlow(BaseLayer):
    """
    The basic class for normalizing flows.

    A normalizing flow transforms a random variable `x` into `y` by an
    (implicitly) invertible mapping :math:`y = f(x)`, whose Jaccobian matrix
    determinant :math:`\\det \\frac{\\partial f(x)}{\\partial x} \\neq 0`, thus
    can derive :math:`\\log p(y)` from given :math:`\\log p(x)`.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, value_ndims=0, name=None, scope=None):
        """
        Construct a new :class:`Flow`.

        Args:
            value_ndims (int): Number of dimensions to be considered as the
                value dimensions.  `x.ndims - value_ndims == log_det.ndims`.
        """
        super(BaseFlow, self).__init__(name=name, scope=scope)
        self._value_ndims = int(value_ndims)

    @property
    def value_ndims(self):
        """
        Get the number of dimensions to be considered as the value of each
        sample of `x`.
        """
        return self._value_ndims

    @property
    def explicitly_invertible(self):
        """
        Whether or not this flow is explicitly invertible?

        If a flow is not explicitly invertible, then it only supports to
        transform `x` into `y`, and corresponding :math:`\\log p(x)` into
        :math:`\\log p(y)`.  It cannot compute :math:`\\log p(y)` directly
        without knowing `x`, nor can it transform `x` back into `y`.

        Returns:
            bool: A boolean indicating whether or not the flow is explicitly
                invertible.
        """
        raise NotImplementedError()

    def _transform(self, x, compute_y, compute_log_det):
        raise NotImplementedError()

    def build(self, input=None):
        if input is not None:
            input = tf.convert_to_tensor(input)
            shape = get_static_shape(input)
            if shape is None or len(shape) < self._value_ndims:
                raise ValueError('`input.ndims` must be known and >= '
                                 '`value_ndims`: input {} vs value_ndims {}.'.
                                 format(input, self._value_ndims))
        return super(BaseFlow, self).build(input)

    def transform(self, x, compute_y=True, compute_log_det=True, name=None):
        """
        Transform `x` into `y`, and the log-determinant of `f` at `x`, i.e.,
        :math:`\\log \\det \\frac{\\partial f(x)}{\\partial x}`.

        Args:
            x (Tensor): The samples of `x`.
            compute_y (bool): Whether or not to compute :math:`y = f(x)`?
                Default :obj:`True`.
            compute_log_det (bool): Whether or not to compute the
                log-determinant?  Default :obj:`True`.
            name (str): If specified, will use this name as the TensorFlow
                operational name scope.
            \\**kwargs: Other named arguments.

        Returns:
            (tf.Tensor, tf.Tensor): `y` and the log-determinant.
                The items in the returned tuple might be :obj:`None`
                if corresponding `compute_?` argument is set to :obj:`False`.

        Raises:
            RuntimeError: If both `compute_y` and `compute_log_det` are set
                to :obj:`False`.
        """
        if not compute_y and not compute_log_det:
            raise RuntimeError('At least one of `compute_y` and '
                               '`compute_log_det` should be True.')

        x = tf.convert_to_tensor(x)
        x_shape = get_static_shape(x)
        if x_shape is None or len(x_shape) < self._value_ndims:
            raise ValueError('`x.ndims` must be known and >= `value_ndims`: '
                             'x {} vs value_ndims {}.'.
                             format(x, self._value_ndims))

        if not self._has_built:
            self.build(x)

        with tf.name_scope(
                name,
                default_name=get_default_scope_name('transform', self),
                values=[x]):
            y, log_det = self._transform(x, compute_y, compute_log_det)

            if log_det is not None:
                with assert_deps([
                        assert_log_det_shape_matches_input(
                            log_det=log_det,
                            input=x,
                            value_ndims=self.value_ndims
                        )]) as asserted:
                    if asserted:  # pragma: no cover
                        log_det = tf.identity(log_det)

            return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        raise NotImplementedError()

    def inverse_transform(self, y, compute_x=True, compute_log_det=True,
                          name=None):
        """
        Transform `y` into `x`, and the log-determinant of `f^{-1}` at `y`,
        i.e., :math:`\\log \\det \\frac{\\partial f^{-1}(y)}{\\partial y}`.

        Args:
            y (Tensor): The samples of `y`.
            compute_x (bool): Whether or not to compute :math:`x = f^{-1}(y)`?
                Default :obj:`True`.
            compute_log_det (bool): Whether or not to compute the
                log-determinant?  Default :obj:`True`.
            name (str): If specified, will use this name as the TensorFlow
                operational name scope.

        Returns:
            (tf.Tensor, tf.Tensor): `x` and the log-determinant.
                The items in the returned tuple might be :obj:`None`
                if corresponding `compute_?` argument is set to :obj:`False`.

        Raises:
            RuntimeError: If both `compute_x` and `compute_log_det` are set
                to :obj:`False`.
            RuntimeError: If the flow is not explicitly invertible.
        """
        if not self.explicitly_invertible:
            raise RuntimeError('The flow is not explicitly invertible: {!r}'.
                               format(self))
        if not compute_x and not compute_log_det:
            raise RuntimeError('At least one of `compute_x` and '
                               '`compute_log_det` should be True.')
        if not self._has_built:
            raise RuntimeError('`inverse_transform` cannot be called before '
                               'the flow has been built; it can be built by '
                               'calling `build`, `apply` or `transform`: '
                               '{!r}'.format(self))

        y = tf.convert_to_tensor(y)
        y_shape = get_static_shape(y)
        if y_shape is None or len(y_shape) < self._value_ndims:
            raise ValueError('`y.ndims` must be known and >= `value_ndims`: '
                             'y {} vs value_ndims {}.'.
                             format(y, self._value_ndims))

        with tf.name_scope(
                name,
                default_name=get_default_scope_name('inverse_transform', self),
                values=[y]):
            x, log_det = self._inverse_transform(y, compute_x, compute_log_det)

            if log_det is not None:
                with assert_deps([
                        assert_log_det_shape_matches_input(
                            log_det=log_det,
                            input=y,
                            value_ndims=self.value_ndims
                        )]) as asserted:
                    if asserted:  # pragma: no cover
                        log_det = tf.identity(log_det)

            return x, log_det

    def _apply(self, x):
        y, _ = self.transform(x, compute_y=True, compute_log_det=False)
        return y


class MultiLayerFlow(BaseFlow):
    """Base class for multi-layer normalizing flows."""

    @add_name_and_scope_arg_doc
    def __init__(self, n_layers, value_ndims=0, name=None, scope=None):
        """
        Construct a new :class:`MultiLayerFlow`.

        Args:
            value_ndims (int): Number of dimensions to be considered as the
                value dimensions.  `x.ndims - value_ndims == log_det.ndims`.
            n_layers (int): Number of flow layers.
        """
        n_layers = int(n_layers)
        if n_layers < 1:
            raise ValueError('`n_layers` must be larger than 0.')
        self._n_layers = n_layers
        self._layer_params = []

        super(MultiLayerFlow, self).__init__(
            value_ndims=value_ndims, name=name, scope=scope)

    @property
    def n_layers(self):
        """
        Get the number of flow layers.

        Returns:
            int: The number of flow layers.
        """
        return self._n_layers

    def _transform_layer(self, layer_id, x, compute_y, compute_log_det):
        raise NotImplementedError()

    def _transform(self, x, compute_y, compute_log_det):
        log_det_list = []

        # apply transformation of each layer
        for i in range(self._n_layers):
            with tf.name_scope('_{}'.format(i)):
                x, log_det = self._transform_layer(
                    layer_id=i,
                    x=x,
                    compute_y=True if i < self._n_layers - 1 else compute_y,
                    compute_log_det=compute_log_det
                )
                log_det_list.append(log_det)

        # merge the log-determinants
        log_det = None
        if compute_log_det:
            log_det = add_n_broadcast(log_det_list)

        y = x if compute_y else None
        return y, log_det

    def _inverse_transform_layer(self, layer_id, y, compute_x, compute_log_det):
        raise NotImplementedError()

    def _inverse_transform(self, y, compute_x, compute_log_det):
        log_det_list = []

        # apply transformation of each layer
        for i in range(self._n_layers - 1, -1, -1):
            with tf.name_scope('_{}'.format(i)):
                y, log_det = self._inverse_transform_layer(
                    layer_id=i,
                    y=y,
                    compute_x=True if i > 0 else compute_x,
                    compute_log_det=compute_log_det
                )
                log_det_list.append(log_det)

        # merge the log-determinants
        log_det = None
        if compute_log_det:
            log_det = add_n_broadcast(log_det_list)

        x = y if compute_x else None
        return x, log_det
