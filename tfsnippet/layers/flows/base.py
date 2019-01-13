import tensorflow as tf

from tfsnippet.ops import assert_rank_at_least
from tfsnippet.utils import (DocInherit, add_name_and_scope_arg_doc,
                             get_default_scope_name, assert_deps,
                             get_static_shape, InputSpec, is_integer)
from ..base import BaseLayer

__all__ = ['BaseFlow', 'MultiLayerFlow', 'FeatureMappingFlow']


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
    def __init__(self, value_ndims=0, require_batch_dims=False,
                 name=None, scope=None):
        """
        Construct a new :class:`Flow`.

        Args:
            value_ndims (int): Number of dimensions to be considered as the
                value dimensions.  `x.ndims - value_ndims == log_det.ndims`.
            require_batch_dims (bool): If :obj:`True`, the `input` tensors
                are required to have at least `value_ndims + 1` dimensions.
                If :obj:`False`, the `input` tensors are required to have
                at least `value_ndims` dimensions.
        """
        super(BaseFlow, self).__init__(name=name, scope=scope)
        self._value_ndims = int(value_ndims)
        self._require_batch_dims = bool(require_batch_dims)

        # derived classes may set this attribute to let BaseFlow validate
        # the input tensors in `transform` and `inverse_transform`.
        self._input_spec = None  # type: InputSpec

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

    def _transform(self, x, compute_y, compute_log_det, previous_log_det):
        raise NotImplementedError()

    def build(self, input=None):
        if input is None:
            raise ValueError('`input` is required to build {}.'.
                             format(self.__class__.__name__))

        input = tf.convert_to_tensor(input)
        shape = get_static_shape(input)
        required_ndims = self._value_ndims
        required_ndims_text = 'value_ndims'
        if self._require_batch_dims:
            required_ndims += 1
            required_ndims_text += ' + 1'

        if shape is None or len(shape) < required_ndims:
            raise ValueError('`input.ndims` must be known and >= '
                             '`{}`: input {} vs value_ndims {}.'.
                             format(required_ndims_text, input, required_ndims))

        return super(BaseFlow, self).build(input)

    def transform(self, x, compute_y=True, compute_log_det=True,
                  previous_log_det=None, name=None):
        """
        Transform `x` into `y`, and the log-determinant of `f` at `x`, i.e.,
        :math:`\\log \\det \\frac{\\partial f(x)}{\\partial x}`.

        Args:
            x (Tensor): The samples of `x`.
            compute_y (bool): Whether or not to compute :math:`y = f(x)`?
                Default :obj:`True`.
            compute_log_det (bool): Whether or not to compute the
                log-determinant?  Default :obj:`True`.
            previous_log_det (Tensor): If specified, add the log-determinant
                of this flow to the log-determinants computed from previous
                flows, and return the summed log-determinant.
            name (str): If specified, will use this name as the TensorFlow
                operational name scope.
            \\**kwargs: Other named arguments.

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
        if previous_log_det is not None and not compute_log_det:
            raise ValueError('`previous_log_det` is specified but '
                             '`compute_log_det` is False.')

        x = tf.convert_to_tensor(x)

        # validate x.ndims
        required_ndims = self._value_ndims
        required_ndims_text = 'value_ndims'
        if self._require_batch_dims:
            required_ndims += 1
            required_ndims_text += ' + 1'
        with assert_deps([
                    assert_rank_at_least(
                        x, required_ndims,
                        message='`x.ndims` must be known and >= `{}`'.
                                format(required_ndims_text)
                    )
                ]) as flag:
            if flag:  # pragma: no cover
                x = tf.identity(x)

        # validate via InputSpec
        if self._input_spec is not None:
            x = self._input_spec.validate('x', x)

        if not self._has_built:
            self.build(x)

        with tf.name_scope(
                name,
                default_name=get_default_scope_name('transform', self),
                values=[x]):
            y, log_det = self._transform(
                x, compute_y, compute_log_det, previous_log_det)

            return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det,
                           previous_log_det):
        raise NotImplementedError()

    def inverse_transform(self, y, compute_x=True, compute_log_det=True,
                          previous_log_det=None, name=None):
        """
        Transform `y` into `x`, and the log-determinant of `f^{-1}` at `y`,
        i.e., :math:`\\log \\det \\frac{\\partial f^{-1}(y)}{\\partial y}`.

        Args:
            y (Tensor): The samples of `y`.
            compute_x (bool): Whether or not to compute :math:`x = f^{-1}(y)`?
                Default :obj:`True`.
            compute_log_det (bool): Whether or not to compute the
                log-determinant?  Default :obj:`True`.
            previous_log_det (Tensor): If specified, add the log-determinant
                of this flow to the log-determinants computed from previous
                flows, and return the summed log-determinant.
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
        if not self.explicitly_invertible:
            raise RuntimeError('The flow is not explicitly invertible: {!r}'.
                               format(self))
        if not compute_x and not compute_log_det:
            raise ValueError('At least one of `compute_x` and '
                             '`compute_log_det` should be True.')
        if previous_log_det is not None and not compute_log_det:
            raise ValueError('`previous_log_det` is specified but '
                             '`compute_log_det` is False.')
        if not self._has_built:
            raise RuntimeError('`inverse_transform` cannot be called before '
                               'the flow has been built; it can be built by '
                               'calling `build`, `apply` or `transform`: '
                               '{!r}'.format(self))

        y = tf.convert_to_tensor(y)

        # validate y.ndims
        required_ndims = self._value_ndims
        required_ndims_text = 'value_ndims'
        if self._require_batch_dims:
            required_ndims += 1
            required_ndims_text += ' + 1'
        with assert_deps([
                    assert_rank_at_least(
                        y, required_ndims,
                        message='`y.ndims` must be known and >= `{}`'.
                                format(required_ndims_text)
                    )
                ]) as flag:
            if flag:  # pragma: no cover
                y = tf.identity(y)

        # validate via InputSpec
        if self._input_spec is not None:
            y = self._input_spec.validate('y', y)

        with tf.name_scope(
                name,
                default_name=get_default_scope_name('inverse_transform', self),
                values=[y]):
            x, log_det = self._inverse_transform(
                y, compute_x, compute_log_det, previous_log_det)

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

    def _transform_layer(self, layer_id, x, compute_y, compute_log_det,
                         previous_log_det):
        raise NotImplementedError()

    def _transform(self, x, compute_y, compute_log_det,
                   previous_log_det):
        # apply transformation of each layer
        log_det = previous_log_det
        for i in range(self._n_layers):
            with tf.name_scope('_{}'.format(i)):
                x, log_det = self._transform_layer(
                    layer_id=i,
                    x=x,
                    compute_y=True if i < self._n_layers - 1 else compute_y,
                    compute_log_det=compute_log_det,
                    previous_log_det=log_det
                )

        y = x if compute_y else None
        return y, log_det

    def _inverse_transform_layer(self, layer_id, y, compute_x, compute_log_det,
                                 previous_log_det):
        raise NotImplementedError()

    def _inverse_transform(self, y, compute_x, compute_log_det,
                           previous_log_det):
        # apply transformation of each layer
        log_det = previous_log_det
        for i in range(self._n_layers - 1, -1, -1):
            with tf.name_scope('_{}'.format(i)):
                y, log_det = self._inverse_transform_layer(
                    layer_id=i,
                    y=y,
                    compute_x=True if i > 0 else compute_x,
                    compute_log_det=compute_log_det,
                    previous_log_det=log_det
                )

        x = y if compute_x else None
        return x, log_det


class FeatureMappingFlow(BaseFlow):
    """
    Base class for flows mapping input features to output features.

    In the :class:`FeatureMappingFlow`, the specified `axis` (may be just a
    single axis or a list of axes) of the input tensors is/are considered to
    be the features axis/axes.  The feature axis/axes must be covered by
    `value_ndims`.

    Also, the `input` tensors are required to have at least `value_ndims`
    dimensions.  If `require_batch_axis` is :obj:`True`, the input tensors must
    have at least `value_ndims + 1` dimensions.

    This base class performs all the validation in `_build`, and constructs
    the corresponding `_input_spec`.  Derived classes should remember to call
    `FeatureMappingFlow._build` in their overrided `_build`, for example::

        class YourFlow(FeatureMappingFlow):

            def _build(input=None):
                super(FeatureMappingFlow, self)._build(input)

                # your build code begins here
                ...
    """

    @add_name_and_scope_arg_doc
    def __init__(self,
                 axis=-1,
                 value_ndims=1,
                 require_batch_dims=False,
                 name=None,
                 scope=None):
        """
        Construct a new :class:`FeatureMappingFlow`.

        Args:
            axis (int or Iterable[int]): The feature axis/axes, on which to
                apply the transformation.
            value_ndims (int): Number of dimensions to be considered as the
                value dimensions.  `x.ndims - value_ndims == log_det.ndims`.
            require_batch_dims (bool): If :obj:`True`, the `input` tensors
                are required to have at least `value_ndims + 1` dimensions.
                If :obj:`False`, the `input` tensors are required to have
                at least `value_ndims` dimensions.
        """
        if is_integer(axis):
            axis = int(axis)
        else:
            axis = tuple(int(a) for a in axis)
            if not axis:
                raise ValueError('`axis` must not be empty.')

        value_ndims = int(value_ndims)
        self._axis = axis

        super(FeatureMappingFlow, self).__init__(
            value_ndims=value_ndims,
            require_batch_dims=require_batch_dims,
            name=name,
            scope=scope,
        )

    @property
    def axis(self):
        """
        Get the feature axis/axes.

        Returns:
            int or tuple[int]: The feature axis/axes, as is specified
                in the constructor.
        """
        return self._axis

    def _build(self, input=None):
        # check the input.
        input = tf.convert_to_tensor(input)
        dtype = input.dtype.base_dtype
        shape = get_static_shape(input)

        # These facts should have been checked in `BaseFlow.build`.
        assert (shape is not None)
        assert (len(shape) >= self.value_ndims)

        # validate the feature axis, ensure it is covered by `value_ndims`.
        axis = self._axis
        axis_is_int = is_integer(axis)
        if axis_is_int:
            axis = [axis]
        else:
            axis = list(axis)

        for i, a in enumerate(axis):
            if a < 0:
                a += len(shape)
            if a < 0 or a < len(shape) - self.value_ndims:
                raise ValueError('`axis` out of range, or not covered by '
                                 '`value_ndims`: axis {}, value_ndims {}, '
                                 'input {}'.
                                 format(self._axis, self.value_ndims, input))
            if shape[a] is None:
                raise ValueError('The feature axis of `input` is not '
                                 'deterministic: input {}, axis {}'.
                                 format(input, self._axis))

            # Store the negative axis, such that when new inputs can have more
            # dimensions than this `input`, the axis can still be correctly
            # resolved.
            axis[i] = a - len(shape)

        if axis_is_int:
            assert(len(axis) == 1)
            self._axis = axis[0]
        else:
            axis_len = len(axis)
            axis = tuple(sorted(set(axis)))
            if len(axis) != axis_len:
                raise ValueError('Duplicated elements after resolving negative '
                                 '`axis` with respect to the `input`: '
                                 'input {}, axis {}'.format(input, self._axis))
            self._axis = tuple(axis)

        # build the input spec
        shape_spec = ['?'] * self.value_ndims
        if self._require_batch_dims:
            shape_spec = ['?'] + shape_spec
        shape_spec = ['...'] + shape_spec

        for a in axis:
            shape_spec[a] = shape[a]

        self._input_spec = InputSpec(shape=shape_spec, dtype=dtype)
        self._input_spec.validate('input', input)
