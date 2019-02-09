import tensorflow as tf

from tfsnippet.stochastic import StochasticTensor
from tfsnippet.utils import (convert_to_tensor_and_cast, is_tensor_object,
                             maybe_check_numerics, broadcast_to_shape,
                             get_shape)
from .base import Distribution
from .utils import (compute_density_immediately, maybe_clip_value,
                    reduce_group_ndims)

__all__ = ['DiscretizedLogistic']


def is_integer_number(n):
    return abs(n - int(n)) < 1e-7


class DiscretizedLogistic(Distribution):
    """
    Discretized logistic distribution (Kingma et. al, 2016).

    For discrete value `x` with equal intervals::

        p(x) = sigmoid((x - mean + bin_size * 0.5) / scale) -
            sigmoid((x - mean - bin_size * 0.5) / scale)

    where `delta` is the interval between two possible values of `x`.

    The `min_val` and `max_val` specifies the minimum and maximum possible
    value of `x`.  It should constraint the generated samples, and if
    `biased_edges` is True, then::

        p(x_min) = sigmoid((x_min - mean + bin_size * 0.5) / scale)
        p(x_max) = 1 - sigmoid((x_max - mean - bin_size * 0.5) / scale)
    """

    def __init__(self, mean, log_scale, bin_size, min_val=None, max_val=None,
                 dtype=tf.float32, biased_edges=True, epsilon=1e-7):
        """
        Construct a new :class:`DiscretizedLogistic`.

        Args:
            mean: A Tensor, the `mean`.
            log_scale: A Tensor, the `log(scale)`.
            bin_size: A scalar, the `bin_size`.
            min_val: A scalar, the minimum possible value of `x`.
            max_val: A scalar, the maximum possible value of `x`.
            dtype: The data type of `x`.
            biased_edges: Whether or not to use bias density for edge values?
                See above.
            epsilon: Small float to avoid dividing by zero or taking
                logarithm of zero.
        """
        # check the arguments
        mean = tf.convert_to_tensor(mean)
        param_dtype = mean.dtype
        log_scale = tf.convert_to_tensor(log_scale)
        dtype = tf.as_dtype(dtype)

        if not is_integer_number(bin_size) and not dtype.is_floating:
            raise ValueError(
                '`bin_size` is a float number, but `dtype` is not a float '
                'number type: {}'.format(dtype)
            )

        if min_val is not None:
            if not is_integer_number(min_val / bin_size):
                raise ValueError(
                    '`min_val` must be multiples of `bin_size`: '
                    'min_val {} vs bin_size {}'.format(min_val, bin_size)
                )

        if max_val is not None:
            if not is_integer_number(max_val / bin_size):
                raise ValueError(
                    '`max_val` must be multiples of `bin_size`: '
                    'max_val {} vs bin_size {}'.format(max_val, bin_size)
                )

        # infer the batch shape
        try:
            batch_static_shape = tf.broadcast_static_shape(
                mean.get_shape(), log_scale.get_shape())
        except ValueError:
            raise ValueError('The shape of `mean` and `log_scale` cannot '
                             'be broadcasted: mean {} vs log_scale {}'.
                             format(mean, log_scale))

        with tf.name_scope('DiscretizedLogistic.init'):
            batch_shape = tf.broadcast_dynamic_shape(tf.shape(mean),
                                                     tf.shape(log_scale))

        # memorize the arguments and call parent constructor
        bin_size = convert_to_tensor_and_cast(bin_size, param_dtype)
        if min_val is not None:
            min_val = convert_to_tensor_and_cast(min_val, param_dtype)
        if max_val is not None:
            max_val = convert_to_tensor_and_cast(max_val, param_dtype)

        self._mean = mean
        self._log_scale = log_scale
        self._param_dtype = param_dtype
        self._bin_size = bin_size
        self._min_val = min_val
        self._max_val = max_val
        self._biased_edges = bool(biased_edges)
        self._epsilon = epsilon

        super(DiscretizedLogistic, self).__init__(
            dtype=dtype,
            is_continuous=False,
            is_reparameterized=False,
            batch_shape=batch_shape,
            batch_static_shape=batch_static_shape,
            value_ndims=0
        )

    @property
    def mean(self):
        """Get the mean."""
        return self._mean

    @property
    def log_scale(self):
        """Get the log-scale."""
        return self._log_scale

    @property
    def bin_size(self):
        """Get the bin size."""
        return self._bin_size

    @property
    def min_val(self):
        """Get the minimum value."""
        return self._min_val

    @property
    def max_val(self):
        """Get the maximum value."""
        return self._max_val

    @property
    def biased_edges(self):
        """Whether or not to use biased density for edge values?"""
        return self._biased_edges

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        self._validate_sample_is_reparameterized_arg(is_reparameterized)

        with tf.name_scope(name, default_name='DiscretizedLogistic.sample'):
            # sample from uniform distribution
            sample_shape = self.batch_shape
            static_sample_shape = self.get_batch_shape()
            if n_samples is not None:
                sample_shape = tf.concat([[n_samples], sample_shape], 0)
                static_sample_shape = tf.TensorShape(
                    [None if is_tensor_object(n_samples) else n_samples]). \
                    concatenate(static_sample_shape)

            u = tf.random_uniform(
                shape=sample_shape, minval=self._epsilon,
                maxval=1. - self._epsilon, dtype=self._param_dtype)
            u.set_shape(static_sample_shape)

            # inverse CDF of the logistic
            inverse_logistic_cdf = maybe_check_numerics(
                tf.log(u) - tf.log(1. - u), 'inverse_logistic_cdf')

            # obtain the actual sample
            scale = maybe_check_numerics(
                tf.exp(self.log_scale, name='scale'), 'scale')
            sample = self.mean + scale * inverse_logistic_cdf
            sample = tf.floor(sample / self.bin_size + .5) * self.bin_size
            sample = maybe_check_numerics(sample, 'sample')
            sample = maybe_clip_value(sample, self.min_val, self.max_val)
            sample = convert_to_tensor_and_cast(sample, self.dtype)

            # tf.floor is non-parameterized until TF 1.12.
            # to ensure future compatibility, we use `stop_gradient`.
            sample = tf.stop_gradient(sample)

            t = StochasticTensor(
                distribution=self,
                tensor=sample,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=False
            )

            # compute the density
            if compute_density:
                compute_density_immediately(t)

            return t

    def log_prob(self, given, group_ndims=0, name=None):
        given = tf.convert_to_tensor(given)

        with tf.name_scope('DiscretizedLogistic.log_prob', values=[given]):
            # inv_scale = 1. / scale
            inv_scale = maybe_check_numerics(
                tf.exp(-self.log_scale, name='inv_scale'), 'inv_scale')
            # half_bin = bin_size / 2
            half_bin = self._bin_size * .5
            # delta = bin_size / scale, half_delta = delta / 2
            half_delta = half_bin * inv_scale
            # log(delta) = log(bin_size) - log(scale)
            log_delta = tf.log(self._bin_size) - self.log_scale

            x_mid = (given - self.mean) * inv_scale
            x_low = x_mid - half_delta
            x_high = x_mid + half_delta

            cdf_low = tf.sigmoid(x_low, name='cdf_low')
            cdf_high = tf.sigmoid(x_high, name='cdf_high')

            # the middle bins cases:
            #   log(sigmoid(x_high) - sigmoid(x_low))
            # but in extreme cases where `sigmoid(x_high) - sigmoid(x_low)`
            # is very small, we use an alternative form, as in PixelCNN++.
            cdf_delta = cdf_high - cdf_low
            middle_bins_pdf = tf.where(
                cdf_delta > self._epsilon,
                # to avoid NaNs pollute the select statement, we have to use
                # `maximum(cdf_delta, 1e-12)`
                tf.log(tf.maximum(cdf_delta, 1e-12)),
                # the alternative form.  basically it can be derived by using
                # the mean value theorem for integration.
                x_mid + log_delta - 2. * tf.nn.softplus(x_mid)
            )
            log_prob = maybe_check_numerics(middle_bins_pdf, 'middle_bins_pdf')

            # broadcasted given, shape == x_mid
            broadcast_given = broadcast_to_shape(given, get_shape(x_mid))

            # the left-edge bin case
            #   log(sigmoid(x_high) - sigmoid(-infinity))
            if self._biased_edges and self.min_val is not None:
                left_edge = self._min_val + half_bin
                left_edge_pdf = maybe_check_numerics(
                    -tf.nn.softplus(-x_high), 'left_edge_pdf')
                log_prob = tf.where(
                    broadcast_given < left_edge, left_edge_pdf, log_prob)

            # the right-edge bin case
            #   log(sigmoid(infinity) - sigmoid(x_low))
            if self._biased_edges and self.max_val is not None:
                right_edge = self._max_val - half_bin
                right_edge_pdf = maybe_check_numerics(
                    -tf.nn.softplus(x_low), 'right_edge_pdf')
                log_prob = tf.where(
                    broadcast_given >= right_edge, right_edge_pdf, log_prob)

            # now reduce the group_ndims
            log_prob = reduce_group_ndims(tf.reduce_sum, log_prob, group_ndims)

        return log_prob
