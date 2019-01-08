import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.layers.flows.utils import broadcast_log_det_against_input
from tfsnippet.utils import (InputSpec, ParamSpec, add_name_and_scope_arg_doc,
                             get_static_shape, maybe_check_numerics,
                             validate_int_tuple_arg, resolve_negative_axis,
                             get_dimensions_size, validate_enum_arg)
from ..flows import BaseFlow


__all__ = ['ActNorm', 'act_norm']


class ActNorm(BaseFlow):
    """
    ActNorm proposed by (Kingma & Dhariwal, 2018).

    `y = (x + bias) * scale; log_det = y / scale - bias`

    `bias` and `scale` are initialized such that `y` will have zero mean and
    unit variance for the initial mini-batch of `x`.
    It can be initialized only through the forward pass.  You may need to use
    :meth:`invert()` to get a inverted flow if you need to initialize the
    parameters via the opposite direction.
    """

    @add_name_and_scope_arg_doc
    def __init__(self,
                 axis=-1,
                 value_ndims=1,
                 initializing=False,
                 scale_type='exp',
                 bias_regularizer=None,
                 bias_constraint=None,
                 log_scale_regularizer=None,
                 log_scale_constraint=None,
                 scale_regularizer=None,
                 scale_constraint=None,
                 trainable=True,
                 epsilon=1e-6,
                 name=None,
                 scope=None):
        """
        Construct a new :class:`ActNorm` instance.

        Args:
            axis (int or Iterable[int]): The axis to apply ActNorm.
                Dimensions not in `axis` will be averaged out when computing
                the mean of activations. Default `-1`, the last dimension.
                All items of the `axis` should be covered by `value_ndims`.
            value_ndims (int): Number of dimensions to be considered as the
                value dimensions.  `x.ndims - value_ndims == log_det.ndims`.
            initializing (bool): Whether or not to use the first input `x`
                in the forward pass to initialize the layer parameters?
                (default :obj:`True`)
            scale_type: One of {"exp", "linear"}.
                If "exp", ``y = (x + bias) * tf.exp(log_scale)``.
                If "linear", ``y = (x + bias) * scale``.
                Default is "exp".
            bias_regularizer: The regularizer for `bias`.
            bias_constraint: The constraint for `bias`.
            log_scale_regularizer: The regularizer for `log_scale`.
            log_scale_constraint: The constraint for `log_scale`.
            scale_regularizer: The regularizer for `scale`.
            scale_constraint: The constraint for `scale`.
            trainable (bool): Whether or not the variables are trainable?
            epsilon: Small float added to variance to avoid dividing by zero.
        """
        self._axis = validate_int_tuple_arg('axis', axis)
        if not self._axis:
            raise ValueError('`axis` must not be empty: got {!r}'.
                             format(self._axis))
        self._value_ndims = int(value_ndims)
        self._scale_type = validate_enum_arg(
            'scale_type', scale_type, ['exp', 'linear'])
        self._initialized = not bool(initializing)
        self._bias_regularizer = bias_regularizer
        self._bias_constraint = bias_constraint
        self._log_scale_regularizer = log_scale_regularizer
        self._log_scale_constraint = log_scale_constraint
        self._scale_regularizer = scale_regularizer
        self._scale_constraint = scale_constraint
        self._trainable = bool(trainable)
        self._epsilon = epsilon

        BaseFlow.__init__(self, value_ndims=value_ndims, name=name, scope=scope)

    def _build(self, input=None):
        # check the input.
        if input is None:
            raise ValueError('`ActNorm` requires `input` to build.')
        input = tf.convert_to_tensor(input)
        dtype = input.dtype.base_dtype
        shape = get_static_shape(input)

        # These facts should have been checked in `BaseFlow.build`.
        assert(shape is not None)
        assert(len(shape) >= self.value_ndims)

        # compute the negative indices of `axis`, and store it in `self._axis`
        axis = resolve_negative_axis(len(shape), self._axis)
        axis = tuple(sorted(set(axis)))
        min_axis = axis[0]
        assert(not not axis)  # already checked in constructor
        self._axis = tuple(a - len(shape) for a in axis)

        # compute var spec and input spec
        shape_spec = [None] * len(shape)
        for a in axis:
            shape_spec[a] = shape[a]
        shape_spec = shape_spec[min_axis:]
        assert(not not shape_spec)

        self._var_shape = tuple(s or 1 for s in shape_spec)
        self._input_spec = InputSpec(
            shape=(('...',) +
                   ('?',) * max(0, self._value_ndims - len(shape_spec)) +
                   tuple(shape_spec)),
            dtype=dtype
        )
        self._var_spec = ParamSpec(self._var_shape)

        # validate the input
        self._input_spec.validate(input)

        # build the variables
        self._bias = tf.get_variable(
            'bias',
            dtype=dtype,
            shape=self._var_shape,
            regularizer=self._bias_regularizer,
            constraint=self._bias_constraint,
            trainable=self._trainable
        )
        if self._scale_type == 'exp':
            self._log_scale = tf.get_variable(
                'log_scale',
                dtype=dtype,
                shape=self._var_shape,
                regularizer=self._log_scale_regularizer,
                constraint=self._log_scale_constraint,
                trainable=self._trainable
            )
            self._scale = None
        else:
            self._log_scale = None
            self._scale = tf.get_variable(
                'scale',
                dtype=dtype,
                shape=self._var_shape,
                regularizer=self._scale_regularizer,
                constraint=self._scale_constraint,
                trainable=self._trainable
            )

    @property
    def explicitly_invertible(self):
        return True

    def _transform(self, x, compute_y, compute_log_det, previous_log_det):
        # check the argument
        x = self._input_spec.validate(x)
        dtype = x.dtype.base_dtype
        shape = get_static_shape(x)
        assert(len(shape) >= self.value_ndims)  # checked in `BaseFlow`
        assert(-len(shape) <= min(self._axis))
        reduce_axis = tuple(sorted(
            set(range(-len(shape), 0)).difference(self._axis)))

        # prepare for the parameters
        if not self._initialized:
            if len(shape) == len(self._var_shape):
                raise ValueError('Initializing ActNorm requires multiple '
                                 '`x` samples, thus `x` must have at least '
                                 'one more dimension than `var_shape`: '
                                 'x {} vs var_shape {}.'.
                                format(x, self._var_shape))

            with tf.name_scope('initialization'):
                x_mean, x_var = tf.nn.moments(x, reduce_axis, keep_dims=True)
                x_mean = tf.reshape(x_mean, self._var_shape)
                x_var = tf.reshape(x_var, self._var_shape)

                bias = self._bias.assign(-x_mean)
                if self._scale_type == 'exp':
                    scale = None
                    log_scale = self._log_scale.assign(
                        -tf.constant(.5, dtype=dtype) *
                        tf.log(x_var + self._epsilon)
                    )
                    log_scale = maybe_check_numerics(log_scale, 'log_scale')
                else:
                    scale = self._scale.assign(
                        tf.constant(1., dtype=dtype) /
                        tf.sqrt(x_var + self._epsilon)
                    )
                    scale = maybe_check_numerics(scale, 'scale')
                    log_scale = None
            self._initialized = True
        else:
            bias = self._bias
            scale = self._scale
            log_scale = self._log_scale

        # compute y
        y = None
        if compute_y:
            if log_scale is not None:
                y = (x + bias) * tf.exp(log_scale, name='scale')
            else:
                y = (x + bias) * scale

        # compute log_det
        log_det = None
        if compute_log_det:
            with tf.name_scope('log_det'):
                if log_scale is None:
                    log_scale = tf.log(tf.abs(scale), name='log_scale')
                log_det = log_scale
                reduce_ndims1 = min(self.value_ndims, len(self._var_shape))
                reduce_ndims2 = self.value_ndims - reduce_ndims1

                # reduce the last `min(value_ndims, len(var_shape))` dimensions
                if reduce_ndims1 > 0:
                    log_det = tf.reduce_sum(
                        log_det, axis=list(range(-reduce_ndims1, 0)))

                    # the following axis have been averaged out during
                    # computation, and will be directly summed up without
                    # getting broadcasted. Thus we need to multiply a factor
                    # to the log_det by the count of reduced elements.
                    reduce_axis1 = tuple(filter(
                        lambda a: (a >= -reduce_ndims1),
                        reduce_axis
                    ))
                    reduce_shape1 = get_dimensions_size(x, reduce_axis1)
                    if isinstance(reduce_shape1, tuple):
                        log_det *= np.prod(reduce_shape1, dtype=np.float32)
                    else:
                        log_det *= tf.cast(
                            tf.reduce_prod(reduce_shape1),
                            dtype=log_scale.dtype
                        )

                # we need to broadcast `log_det` to match the shape of `x`
                log_det = broadcast_log_det_against_input(
                    log_det, x, value_ndims=reduce_ndims1)

                # reduce the remaining dimensions
                if reduce_ndims2 > 0:
                    log_det = tf.reduce_sum(
                        log_det, axis=list(range(-reduce_ndims2, 0)))

                # merge with previous log det if specified
                if previous_log_det is not None:
                    log_det = previous_log_det + log_det

        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det,
                           previous_log_det):
        # `BaseFlow` ensures `build` is called before `inverse_transform`.
        # In `ActNorm`, `build` can only be called by `apply` or `transform`.
        # Thus it should always have been initialized.
        assert(self._initialized)

        # check the argument
        y = self._input_spec.validate(y)
        shape = get_static_shape(y)
        assert(len(shape) >= self.value_ndims)  # checked in `BaseFlow`
        assert(-len(shape) <= min(self._axis))
        reduce_axis = tuple(sorted(
            set(range(-len(shape), 0)).difference(self._axis)))

        # compute x
        x = None
        if compute_x:
            if self._scale_type == 'exp':
                x = y * tf.exp(-self._log_scale) - self._bias
            else:
                x = y / self._scale - self._bias

        # compute log_det
        log_det = None
        if compute_log_det:
            with tf.name_scope('log_det'):
                if self._scale_type == 'exp':
                    log_scale = self._log_scale
                else:
                    log_scale = tf.log(tf.abs(self._scale), name='log_scale')
                log_det = -log_scale
                reduce_ndims1 = min(self.value_ndims, len(self._var_shape))
                reduce_ndims2 = self.value_ndims - reduce_ndims1

                # reduce the last `min(value_ndims, len(var_shape))` dimensions
                if reduce_ndims1 > 0:
                    log_det = tf.reduce_sum(
                        log_det, axis=list(range(-reduce_ndims1, 0)))

                    # the following axis have been averaged out during
                    # computation, and will be directly summed up without
                    # getting broadcasted. Thus we need to multiply a factor
                    # to the log_det by the count of reduced elements.
                    reduce_axis1 = tuple(filter(
                        lambda a: (a >= -reduce_ndims1),
                        reduce_axis
                    ))
                    reduce_shape1 = get_dimensions_size(y, reduce_axis1)
                    if isinstance(reduce_shape1, tuple):
                        log_det *= np.prod(reduce_shape1, dtype=np.float32)
                    else:
                        log_det *= tf.cast(
                            tf.reduce_prod(reduce_shape1),
                            dtype=log_scale.dtype
                        )

                # we need to broadcast `log_det` to match the shape of `y`
                log_det = broadcast_log_det_against_input(
                    log_det, y, value_ndims=reduce_ndims1)

                # reduce the remaining dimensions
                if reduce_ndims2 > 0:
                    log_det = tf.reduce_sum(
                        log_det, axis=list(range(-reduce_ndims2, 0)))

                # merge with previous log det if specified
                if previous_log_det is not None:
                    log_det = previous_log_det + log_det

        return x, log_det


@add_arg_scope
def act_norm(input, **kwargs):
    """
    ActNorm proposed by (Kingma & Dhariwal, 2018).

    Args:
        input (tf.Tensor): The input tensor.
        \\**kwargs: Other arguments passed to :class:`ActNorm`.

    Returns:
        tf.Tensor: The output after the ActNorm has been applied.
    """
    return ActNorm(**kwargs).apply(input)
