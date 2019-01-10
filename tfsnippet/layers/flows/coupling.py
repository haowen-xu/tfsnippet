import tensorflow as tf

from tfsnippet.ops import assert_shape_equal
from tfsnippet.utils import (add_name_and_scope_arg_doc, get_static_shape,
                             validate_enum_arg, InputSpec, assert_deps,
                             get_shape, broadcast_to_shape)
from .base import BaseFlow
from .utils import SigmoidScale, ExpScale, LinearScale

__all__ = ['BaseCouplingLayer', 'CouplingLayer']


class BaseCouplingLayer(BaseFlow):
    """
    The base class of :class:`CouplingLayer`.

    The only situation this class is preferred to :class:`CouplingLayer`,
    is that `shift_and_scale_fn` is provided as the class method, instead
    of an argument of :class:`CouplingLayer`.

    See Also:
        :class:`tfsnippet.layers.CouplingLayer`
    """

    @add_name_and_scope_arg_doc
    def __init__(self,
                 axis=-1,
                 value_ndims=1,
                 secondary=False,
                 scale_type='linear',
                 sigmoid_scale_bias=2.,
                 epsilon=1e-6,
                 name=None,
                 scope=None):
        """
        Construct a new :class:`BaseCouplingLayer`.

        Args:
            axis (int): The feature axis, to apply the transformation.
            value_ndims (int): Number of dimensions to be considered as the
                value dimensions.  `x.ndims - value_ndims == log_det.ndims`.
            secondary (bool): Whether or not this layer is a secondary layer?
                See :class:`tfsnippet.layers.CouplingLayer`.
            scale_type: One of {"exp", "sigmoid", "linear", None}.
                See :class:`tfsnippet.layers.CouplingLayer`.
            sigmoid_scale_bias (float or Tensor): Add this bias to the `scale`
                if ``scale_type == 'sigmoid'``.  See the reason of adopting
                this in :class:`tfsnippet.layers.CouplingLayer`.
            epsilon: Small float number to avoid dividing by zero or taking
                logarithm of zero.
        """
        self._axis = int(axis)
        self._secondary = bool(secondary)
        self._scale_type = validate_enum_arg(
            'scale_type', scale_type, ['exp', 'sigmoid', 'linear', None])
        self._sigmoid_scale_bias = sigmoid_scale_bias
        self._epsilon = epsilon

        super(BaseCouplingLayer, self).__init__(
            value_ndims=value_ndims, name=name, scope=scope)

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
        if axis < 0:
            axis += len(shape)
        if axis < 0 or axis < len(shape) - self.value_ndims:
            raise ValueError('`axis` out of range, or not covered by '
                             '`value_ndims`: axis {}, value_ndims {}, input {}'.
                             format(self._axis, self.value_ndims, input))
        if shape[axis] is None:
            raise ValueError('The feature axis of `input` is not deterministic'
                             ': input {}, axis {}'.format(input, self._axis))
        if shape[axis] < 2:
            raise ValueError('The feature axis of `input` must be at least 2: '
                             'got {}, input {}, axis {}.'.
                             format(shape[axis], input, self._axis))
        self._n_features = shape[axis]

        # store the negative axis, such that new inputs can have more dimensions
        # than this input.
        self._axis = axis - len(shape)

        # build the input spec
        shape_spec = ['...'] + (['?'] * self.value_ndims)
        shape_spec[self._axis] = self._n_features
        self._input_spec = InputSpec(shape=shape_spec, dtype=dtype)

        # validate the input
        self._input_spec.validate(input)

    @property
    def explicitly_invertible(self):
        return True

    def _split(self, x):
        assert(get_static_shape(x)[self._axis] == self._n_features)
        n1 = self._n_features // 2
        n2 = self._n_features - n1
        x1, x2 = tf.split(x, [n1, n2], self._axis)
        if self._secondary:
            return x2, x1, n1
        else:
            return x1, x2, n2

    def _unsplit(self, x1, x2):
        n1 = self._n_features // 2
        n2 = self._n_features - n1
        if self._secondary:
            x1, x2 = x2, x1
        assert(get_static_shape(x1)[self._axis] == n1)
        assert(get_static_shape(x2)[self._axis] == n2)
        return tf.concat([x1, x2], axis=self._axis)

    def _compute_shift_and_scale(self, x1, n2):
        raise NotImplementedError()

    def _check_scale_or_shift_shape(self, name, tensor, x2):
        assert_op = assert_shape_equal(
            tensor, x2,
            message='`{}.shape` expected to be {}, but got {}'.format(
                name,
                get_static_shape(x2),
                get_static_shape(tensor)
            )
        )
        with assert_deps([assert_op]) as asserted:
            if asserted:
                tensor = tf.identity(tensor)
        return tensor

    def _transform_or_inverse_transform(self, x, compute_y, compute_log_det,
                                        previous_log_det, reverse=False):
        # Since the transform and inverse_transform are too similar, we
        # just implement these two methods by one super method, controlled
        # by `reverse == True/False`.

        # check the argument
        x = self._input_spec.validate(x)
        shape = get_static_shape(x)
        assert (len(shape) >= self.value_ndims)  # checked in `BaseFlow`

        # split the tensor
        x1, x2, n2 = self._split(x)

        # compute the scale and shift
        shift, pre_scale = self._compute_shift_and_scale(x1, n2)
        if self._scale_type is not None and pre_scale is None:
            raise RuntimeError('`scale_type` != None, but no scale is '
                               'computed.')
        elif self._scale_type is None and pre_scale is not None:
            raise RuntimeError('`scale_type` == None, but scale is computed.')

        if pre_scale is not None:
            pre_scale = self._check_scale_or_shift_shape('scale', pre_scale, x2)
        shift = self._check_scale_or_shift_shape('shift', shift, x2)

        # derive the scale class
        if self._scale_type == 'sigmoid':
            scale = SigmoidScale(
                pre_scale + self._sigmoid_scale_bias, self._epsilon)
        elif self._scale_type == 'exp':
            scale = ExpScale(pre_scale, self._epsilon)
        elif self._scale_type == 'linear':
            scale = LinearScale(pre_scale, self._epsilon)
        else:
            assert (self._scale_type is None)
            scale = None

        # compute y
        y = None
        if compute_y:
            y1 = x1
            if reverse:
                y2 = x2
                if scale is not None:
                    y2 = y2 / scale
                y2 -= shift
            else:
                y2 = x2 + shift
                if scale is not None:
                    y2 = y2 * scale
            y = self._unsplit(y1, y2)

        # compute log_det
        log_det = None
        if compute_log_det:
            assert (self.value_ndims >= 0)  # checked in `_build`
            if scale is not None:
                log_det = tf.reduce_sum(
                    scale.neg_log_scale if reverse else scale.log_scale,
                    list(range(-self.value_ndims, 0))
                )
                if previous_log_det is not None:
                    log_det = previous_log_det + log_det
            else:
                dst_shape = get_shape(x)[:-self.value_ndims]
                if previous_log_det is not None:
                    log_det = broadcast_to_shape(previous_log_det, dst_shape)
                else:
                    log_det = tf.zeros(dst_shape, dtype=x.dtype.base_dtype)

        return y, log_det

    def _transform(self, x, compute_y, compute_log_det, previous_log_det):
        return self._transform_or_inverse_transform(
            x=x, compute_y=compute_y, compute_log_det=compute_log_det,
            previous_log_det=previous_log_det, reverse=False
        )

    def _inverse_transform(self, y, compute_x, compute_log_det,
                           previous_log_det):
        return self._transform_or_inverse_transform(
            x=y, compute_y=compute_x, compute_log_det=compute_log_det,
            previous_log_det=previous_log_det, reverse=True
        )


class CouplingLayer(BaseCouplingLayer):
    """
    A general implementation of the coupling layer (Dinh et al., 2016)

    Basically, a :class:`CouplingLayer` does the following transformation::

        x1, x2 = split(x)
        if secondary:
            x1, x2 = x2, x1

        y1 = x1

        shift, scale = shift_and_scale_fn(x1, x2.shape[axis])
        if scale_type == 'exp':
            y2 = (x2 + shift) * exp(scale)
        elif scale_type == 'sigmoid':
            y2 = (x2 + shift) * sigmoid(scale + sigmoid_scale_bias)
        elif scale_type == 'linear':
            y2 = (x2 + shift) * scale
        else:
            y2 = x2 + shift

        if secondary:
            y1, y2 = y2, y1
        y = tf.concat([y1, y2], axis=axis)

    The inverse transformation, and the log-determinants are computed
    according to the above transformation, respectively.
    """

    @add_name_and_scope_arg_doc
    def __init__(self,
                 shift_and_scale_fn,
                 axis=-1,
                 value_ndims=1,
                 secondary=False,
                 scale_type='linear',
                 sigmoid_scale_bias=2.,
                 epsilon=1e-6,
                 name=None,
                 scope=None):
        """

        Args:
            shift_and_scale_fn:
            axis (int): The feature axis, to apply the transformation.
                See above.
            value_ndims (int): Number of dimensions to be considered as the
                value dimensions.  `x.ndims - value_ndims == log_det.ndims`.
            secondary (bool): Whether or not this layer is a secondary layer?
                See above.
            scale_type: One of {"exp", "sigmoid", "linear", None}.
                See above.
            sigmoid_scale_bias (float or Tensor): Add this bias to
                the `scale` if ``scale_type == 'sigmoid'``.
                Here is the reason of adopting this: many random initializers
                will cause the activation of a linear layer to have zero
                mean, thus `sigmoid(scale)` will distribute around `0.5`.
                A positive bias will shift `sigmoid(scale)` towards `1.`,
                while a negative bias will shift `sigmoid(scale)` towards `0.`.
            epsilon: Small float number to avoid dividing by zero or taking
                logarithm of zero.
        """
        self._shift_and_scale_fn = shift_and_scale_fn
        super(CouplingLayer, self).__init__(
            axis=axis,
            value_ndims=value_ndims,
            secondary=secondary,
            scale_type=scale_type,
            sigmoid_scale_bias=sigmoid_scale_bias,
            epsilon=epsilon,
            name=name,
            scope=scope,
        )

    def _compute_shift_and_scale(self, x1, n2):
        shift, pre_scale = self._shift_and_scale_fn(x1, n2)
        return shift, pre_scale
