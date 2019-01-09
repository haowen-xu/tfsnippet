import tensorflow as tf

from tfsnippet.ops import assert_shape_equal
from tfsnippet.utils import (add_name_and_scope_arg_doc, get_static_shape,
                             validate_enum_arg, InputSpec, assert_deps,
                             get_shape)
from .base import BaseFlow
from .utils import SigmoidScale, ExpScale, LinearScale

__all__ = ['BaseCouplingLayer', 'CouplingLayer']


class BaseCouplingLayer(BaseFlow):

    @add_name_and_scope_arg_doc
    def __init__(self,
                 axis=-1,
                 value_ndims=1,
                 secondary=False,
                 scale_type='linear',
                 name=None,
                 scope=None):
        self._axis = int(axis)
        self._secondary = bool(secondary)
        self._scale_type = validate_enum_arg(
            'scale_type', scale_type, ['linear', 'exp', 'sigmoid', None])

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
            raise RuntimeError('`scale_type` != None, but scale is computed.')

        if pre_scale is not None:
            pre_scale = self._check_scale_or_shift_shape('scale', pre_scale, x2)
        shift = self._check_scale_or_shift_shape('shift', shift, x2)

        # derive the scale class
        if self._scale_type == 'sigmoid':
            scale_obj = SigmoidScale(pre_scale)
        elif self._scale_type == 'exp':
            scale_obj = ExpScale(pre_scale)
        elif self._scale_type == 'linear':
            scale_obj = LinearScale(pre_scale)
        else:
            assert (self._scale_type is None)
            scale_obj = None

        # compute y
        y = None
        if compute_y:
            y1 = x1

            if reverse:
                y2 = x2
                if scale_obj is not None:
                    y2 = scale_obj.apply(y2, reverse=reverse)
                y2 += shift
            else:
                y2 = x2 + shift
                if scale_obj is not None:
                    y2 = scale_obj.apply(y2, reverse=reverse)

            y = self._unsplit(y1, y2)

        # compute log_det
        log_det = None
        if compute_log_det:
            assert (self.value_ndims >= 0)  # checked in `_build`

            if scale_obj is not None:
                log_det = tf.reduce_sum(
                    scale_obj.log_scale(reverse=reverse),
                    list(range(-self.value_ndims, 0))
                )
                if previous_log_det:
                    log_det = previous_log_det + log_det

            else:
                log_det = tf.zeros(get_shape(x)[:-self.value_ndims])
                if previous_log_det:
                    # zeros is required here, to enforce a broadcast
                    # between the x shape and the previous_log_det.
                    # TODO: use broadcast function instead of adding zeros
                    log_det = previous_log_det + log_det

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

    @add_name_and_scope_arg_doc
    def __init__(self,
                 shift_and_scale_fn,
                 axis=-1,
                 value_ndims=1,
                 secondary=False,
                 scale_type='linear',
                 name=None,
                 scope=None):
        self._shift_and_scale_fn = shift_and_scale_fn
        super(CouplingLayer, self).__init__(
            axis=axis,
            value_ndims=value_ndims,
            secondary=secondary,
            scale_type=scale_type,
            name=name,
            scope=scope,
        )

    def _compute_shift_and_scale(self, x1, n2):
        shift, pre_scale = self._shift_and_scale_fn(x1, n2)
        return shift, pre_scale
