import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import (InputSpec, ParamSpec, add_name_and_scope_arg_doc,
                             int_shape, reopen_variable_scope,
                             maybe_check_numerics, get_dimensions_size,
                             get_shape, concat_shapes, validate_enum_arg,
                             validate_int_or_int_tuple_arg)
from ..flows import BaseFlow


__all__ = ['ActNorm', 'act_norm', 'act_norm_conv2d']


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
                 var_shape,
                 initializing=False,
                 scale_type='log_scale',
                 bias_regularizer=None,
                 bias_constraint=None,
                 log_scale_regularizer=None,
                 log_scale_constraint=None,
                 scale_regularizer=None,
                 scale_constraint=None,
                 trainable=True,
                 epsilon=1e-6,
                 dtype=tf.float32,
                 name=None,
                 scope=None):
        """
        Construct a new :class:`ActNorm` instance.

        Args:
            var_shape (Iterable[int]): The shape of the normalization variables.
                Typically, the `1`s in the `var_shape` should indicate an
                axis to be reduced out when normalizing the input, and the
                non-`1`s should indicate an axis of the individual features
                (e.g., channels in a convolutional layer).

                In addition to this information, ``len(var_shape)`` will be
                used as ``value_ndims`` of the flow.  Thus the leading `1`s
                in `var_shape` should not be omitted.  (i.e., when `var_shape`
                is aligned with the input shape at right, the first dimension
                at the left of `var_shape` should typically be the mini-batch)
            initializing (bool): Whether or not to use the first input `x`
                in the forward pass to initialize the layer parameters?
                (default :obj:`True`)
            scale_type: One of {"log_scale", "scale"}.
                If "log_scale", ``y = (x + bias) * tf.exp(log_scale)``.
                If "scale", ``y = (x + bias) * scale``.
                Default is "log_scale".
            bias_regularizer: The regularizer for `bias`.
            bias_constraint: The constraint for `bias`.
            log_scale_regularizer: The regularizer for `log_scale`.
            log_scale_constraint: The constraint for `log_scale`.
            scale_regularizer: The regularizer for `scale`.
            scale_constraint: The constraint for `scale`.
            trainable (bool): Whether or not the variables are trainable?
            epsilon: Small float added to variance to avoid dividing by zero.
            dtype: The data type of the transformed `y`.
        """
        scale_type = validate_enum_arg(
            'scale_type', scale_type, ['scale', 'log_scale'])

        var_shape = validate_int_or_int_tuple_arg('var_shape', var_shape)
        var_spec = ParamSpec(var_shape)

        # for each dimension in var_shape, generate its negative index
        var_indices = tuple(i - len(var_shape) for i in range(len(var_shape)))
        # all `1`s should be reduced out
        reduce_axis = tuple(i for i, s in zip(var_indices, var_shape) if s == 1)
        # the input shape should be ('?') for `1`s
        input_spec = InputSpec(
            shape=('...',) + tuple(a if a != 1 else '?' for a in var_shape),
            dtype=dtype
        )

        self._var_shape = var_shape
        self._var_spec = var_spec
        self._input_spec = input_spec
        self._reduce_axis = reduce_axis
        self._scale_type = scale_type
        self._epsilon = epsilon
        self._initialized = not initializing

        BaseFlow.__init__(
            self,
            value_ndims=len(var_shape),
            dtype=dtype,
            name=name,
            scope=scope
        )

        with reopen_variable_scope(self.variable_scope):
            self._bias = tf.get_variable(
                'bias',
                dtype=self.dtype,
                shape=var_shape,
                regularizer=bias_regularizer,
                constraint=bias_constraint,
                trainable=trainable
            )
            if self._scale_type == 'log_scale':
                self._log_scale = tf.get_variable(
                    'log_scale',
                    dtype=self.dtype,
                    shape=var_shape,
                    regularizer=log_scale_regularizer,
                    constraint=log_scale_constraint,
                    trainable=trainable
                )
                self._scale = None
            else:
                self._log_scale = None
                self._scale = tf.get_variable(
                    'scale',
                    dtype=self.dtype,
                    shape=var_shape,
                    regularizer=scale_regularizer,
                    constraint=scale_constraint,
                    trainable=trainable
                )

    @property
    def explicitly_invertible(self):
        return True

    @property
    def var_shape(self):
        """
        Get the shape of the normalization variables.

        Returns:
            tuple[int]: The shape of the normalization variables.
        """
        return self._var_shape

    def _check_shape(self, x):
        x = self._input_spec.validate(x)
        shape = int_shape(x)
        reduce_axis = self._reduce_axis
        if len(shape) > len(self.var_shape):
            front_axis = tuple(
                -i - len(self.var_shape) - 1
                for i in range(len(shape) - len(self.var_shape))
            )
        else:
            front_axis = ()
        reduce_axis += front_axis
        return x, shape, reduce_axis

    def _transform(self, x, compute_y, compute_log_det):
        # check the argument
        x, shape, reduce_axis = self._check_shape(x)

        # prepare for the parameters
        if not self._initialized:
            if len(shape) == len(self.var_shape):
                raise TypeError('Initializing ActNorm requires multiple '
                                '`x` samples, thus `x` must have at least '
                                'one more dimension than `var_shape`: '
                                'x.shape {} vs var_shape {}.'.
                                format(shape, self.var_shape))

            with tf.name_scope('initialization'):
                x_mean = tf.reshape(
                    tf.reduce_mean(x, axis=reduce_axis, keepdims=True),
                    self.var_shape
                )
                x_var = tf.reshape(
                    tf.reduce_mean((x - x_mean) ** 2, axis=reduce_axis,
                                   keepdims=True),
                    self.var_shape
                )
                bias = self._bias.assign(-x_mean)
                if self._scale_type == 'log_scale':
                    scale = None
                    log_scale = self._log_scale.assign(
                        -tf.constant(.5, dtype=self.dtype) *
                        tf.log(x_var + self._epsilon)
                    )
                    log_scale = maybe_check_numerics(log_scale, 'log_scale')
                else:
                    scale = self._scale.assign(
                        tf.constant(1., dtype=self.dtype) /
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
            if log_scale is None:
                log_scale = tf.log(tf.abs(scale), name='log_scale')
            # the reduce axis has been averaged out, thus the log-det computed
            # from log_scale is smaller than the real log-det by this factor.
            local_reduce_shape = get_dimensions_size(x, self._reduce_axis)
            if isinstance(local_reduce_shape, tuple):
                log_det_factor = np.prod(local_reduce_shape, dtype=np.float32)
            else:
                log_det_factor = tf.cast(
                    tf.reduce_prod(local_reduce_shape), dtype=log_scale.dtype)
            log_det = tf.reduce_sum(log_scale) * log_det_factor

            # `rank(log_det) == len([s for s in var_shape if s != 1])`,
            # but this is not expected by the Flow, which requires
            # `rank(log_det) == rank(x) - value_ndims`.
            # since `value_ndims == len([s for s in var_shape if s != 1])`,
            # we need to expand the dimension of `log_det` at the front
            if len(shape) > len(self.var_shape):
                log_det_shape = concat_shapes([
                    [1] * (len(shape) - len(self.var_shape)),
                    get_shape(log_det)
                ])
                log_det = tf.reshape(log_det, log_det_shape)

        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        if not self._initialized:
            raise RuntimeError('{!r} has not been initialized by data.'.
                               format(self))

        # check the argument
        y, shape, reduce_axis = self._check_shape(y)

        # compute x
        x = None
        if compute_x:
            if self._scale_type == 'log_scale':
                x = y * tf.exp(-self._log_scale) - self._bias
            else:
                x = y / self._scale - self._bias

        # compute log_det
        log_det = None
        if compute_log_det:
            if self._scale_type == 'log_scale':
                log_scale = self._log_scale
            else:
                log_scale = tf.log(tf.abs(self._scale), name='log_scale')

            # see `_transform`
            local_reduce_shape = get_dimensions_size(x, self._reduce_axis)
            if isinstance(local_reduce_shape, tuple):
                log_det_factor = np.prod(local_reduce_shape, dtype=np.float32)
            else:
                log_det_factor = tf.cast(
                    tf.reduce_prod(local_reduce_shape), dtype=log_scale.dtype)
            log_det = -tf.reduce_sum(log_scale) * log_det_factor

            # see `_transform`
            if len(shape) > len(self.var_shape):
                log_det_shape = concat_shapes([
                    [1] * (len(shape) - len(self.var_shape)),
                    get_shape(log_det)
                ])
                log_det = tf.reshape(log_det, log_det_shape)

        return x, log_det


@add_arg_scope
def act_norm(input, axis=-1, value_ndims=1, **kwargs):
    """
    ActNorm proposed by (Kingma & Dhariwal, 2018).

    Args:
        input (tf.Tensor): The input tensor.
        axis (int or Iterable[int]): The axis to apply ActNorm.
            Dimensions not in `axis` will be averaged out when computing
            the mean of activations. Default `-1`, the last dimension.
        value_ndims (int): The number of dimensions for each input sample.
            The other dimensions will be regarded as mini-batch dimension
            and sampling dimensions.  Default `1`, i.e., only the last
            dimension will be regarded as value dimension.
        \\**kwargs: Other arguments passed to :class:`ActNorm`.

    Returns:
        tf.Tensor: The output after the ActNorm has been applied.
    """
    # check the arguments.
    axis = validate_int_or_int_tuple_arg('axis', axis)
    input = tf.convert_to_tensor(input)
    shape = int_shape(input)
    if shape is None:
        raise TypeError('The ndims of `input` must be known.')
    if len(shape) <= value_ndims:
        raise TypeError('The `input` ndims must be larger than '
                        '`value_ndims`: input shape {} vs value_ndims {}'.
                        format(shape, value_ndims))

    # compute the var_shape according to axis and value_ndims
    if value_ndims > 0:
        prepare_spec = [1] * len(shape)
        for a in axis:
            prepare_spec[a] = shape[a]
        var_shape = tuple(prepare_spec[-value_ndims:])
    else:
        var_shape = ()

    layer = ActNorm(var_shape=var_shape, **kwargs)
    return layer(input)


@add_arg_scope
def act_norm_conv2d(input, channels_last=False, **kwargs):
    """
    ActNorm proposed by (Kingma & Dhariwal, 2018), specialized for 2d
    convolutional layers.  See :func:`act_norm` for more details.

    Args:
        input (tf.Tensor): The input tensor.  Must be at least 4-d tensor,
            where the last 3 dimensions are the image pixels.
        channels_last (bool): Whether or not the last dimension is the channel?
    """
    axis = -1 if bool(channels_last) else -3
    return act_norm(input, axis=axis, value_ndims=3, **kwargs)
