import tensorflow as tf

from tfsnippet.utils import DocInherit, VarScopeObject, reopen_variable_scope

__all__ = ['Flow', 'MultiLayerFlow']


@DocInherit
class Flow(VarScopeObject):
    """
    The basic class for normalizing flows.

    A normalizing flow transforms a random variable `x` into `y` by an
    (implicitly) invertible mapping :math:`y = f(x)`, whose Jaccobian matrix
    determinant :math:`\\det \\frac{\\partial f(x)}{\\partial x} \\neq 0`, thus
    can derive :math:`\\log p(y)` from given :math:`\\log p(x)`.
    """

    def __init__(self, dtype=tf.float32, name=None, scope=None):
        """
        Construct a new :class:`Flow`.

        Args:
            dtype: The data type of the transformed `y`.
            name (str): Optional name of this :class:`Flow`
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
            scope (str): Optional scope of this :class:`Flow`
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
        """
        dtype = tf.as_dtype(dtype)
        if not dtype.is_floating:
            raise TypeError('Expected a float dtype, but got {}.'.format(dtype))
        self._dtype = dtype
        super(Flow, self).__init__(name=name, scope=scope)

    @property
    def dtype(self):
        """
        The data type of `y`.

        Returns:
            tf.DType: The data type of `y`.
        """
        return self._dtype

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
        if x.dtype != self.dtype:
            x = tf.cast(x, self.dtype)
        def_name = '{}.transform'.format(self.__class__.__name__)
        with tf.name_scope(name, default_name=def_name, values=[x]):
            return self._transform(x, compute_y, compute_log_det)

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
        y = tf.convert_to_tensor(y)
        if y.dtype != self.dtype:
            y = tf.cast(y, self.dtype)
        def_name = '{}.inverse_transform'.format(self.__class__.__name__)
        with tf.name_scope(name, default_name=def_name, values=[y]):
            return self._inverse_transform(y, compute_x, compute_log_det)


class MultiLayerFlow(Flow):
    """Base class for multi-layer normalizing flows."""

    def __init__(self, n_layers, dtype=tf.float32, name=None, scope=None):
        """
        Construct a new :class:`MultiLayerFlow`.

        Args:
            n_layers (int): Number of flow layers.
            dtype: The data type of the transformed `y`.
            name (str): Optional name of this :class:`VariableSaver`
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
            scope (str): Optional scope of this :class:`VariableSaver`
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
        """
        super(MultiLayerFlow, self).__init__(
            dtype=dtype, name=name, scope=scope)

        n_layers = int(n_layers)
        if n_layers < 1:
            raise ValueError('`n_layers` must be larger than 0.')
        self._n_layers = n_layers
        self._layer_params = []

        with reopen_variable_scope(self.variable_scope):
            for i in range(self._n_layers):
                with tf.variable_scope('_{}'.format(i)):
                    self._layer_params.append(self._create_layer_params(i))

    @property
    def n_layers(self):
        """
        Get the number of flow layers.

        Returns:
            int: The number of flow layers.
        """
        return self._n_layers

    def _create_layer_params(self, layer_id):
        """
        Create layer

        Args:
            layer_id (int): The integer ID of the layer.

        Returns:
            dict[str, tf.Variable or tf.Tensor]: The layer parameters.
        """
        raise NotImplementedError()

    def get_layer_params(self, layer_id, names):
        """
        Get layer parameters.

        Args:
            layer_id (int): The integer ID of the layer.
            names (Iterable[str]): The names of the parameters to get.

        Returns:
            list[tf.Variable or tf.Tensor]: The layer parameters.
        """
        layer_params = self._layer_params[layer_id]
        return [layer_params[n] for n in names]

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
            log_det = tf.add_n(log_det_list)

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
            log_det = tf.add_n(log_det_list)

        x = y if compute_x else None
        return x, log_det
