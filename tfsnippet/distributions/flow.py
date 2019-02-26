import tensorflow as tf

from tfsnippet.stochastic import StochasticTensor
from tfsnippet.layers import BaseFlow
from tfsnippet.utils import (validate_group_ndims_arg,
                             get_default_scope_name,
                             TensorWrapper,
                             register_tensor_wrapper_class)
from .base import Distribution
from .utils import reduce_group_ndims
from .wrapper import as_distribution

__all__ = ['FlowDistributionDerivedTensor', 'FlowDistribution']


class FlowDistributionDerivedTensor(TensorWrapper):
    """
    A combination of a :class:`FlowDistribution` derived tensor, and its
    original stochastic tensor from the base distribution.
    """

    def __init__(self, tensor, flow_origin):
        """
        Construct a new :class:`FlowDistributionDerivedTensor`.

        Args:
            tensor (tf.Tensor): The :class:`FlowDistribution` derived tensor.
            flow_origin (StochasticTensor): The original stochastic tensor
                from the base distribution.
        """
        self._self_tensor = tensor
        self._self_flow_origin = flow_origin

    @property
    def flow_origin(self):
        """
        Get the original stochastic tensor from the base distribution.

        Returns:
            StochasticTensor: The original stochastic tensor.
        """
        return self._self_flow_origin

    @property
    def tensor(self):
        return self._self_tensor


register_tensor_wrapper_class(FlowDistributionDerivedTensor)


class FlowDistribution(Distribution):
    """
    Transform a :class:`Distribution` by a :class:`BaseFlow`, as a new
    distribution.
    """

    def __init__(self, distribution, flow):
        """
        Construct a new :class:`FlowDistribution` from the given `distribution`.

        Args:
            distribution (Distribution): The distribution to transform from.
                It must be continuous,
            flow (BaseFlow): A normalizing flow to transform the `distribution`.
        """
        if not isinstance(flow, BaseFlow):
            raise TypeError('`flow` is not an instance of `BaseFlow`: {!r}'.
                            format(flow))
        distribution = as_distribution(distribution)
        if not distribution.is_continuous:
            raise ValueError('Distribution {!r} cannot be transformed by a '
                             'flow, because it is not continuous.'.
                             format(distribution))
        if not distribution.dtype.is_floating:
            raise ValueError('Distribution {!r} cannot be transformed by a '
                             'flow, because its data type is not float.'.
                             format(distribution))
        if distribution.value_ndims > flow.x_value_ndims:
            raise ValueError('Distribution {!r} cannot be transformed by flow '
                             '{!r}, because distribution.value_ndims is larger '
                             'than flow.x_value_ndims.'.
                             format(distribution, flow))

        self._flow = flow
        self._distribution = distribution

        tmp_distrib = distribution.expand_value_ndims(
            flow.x_value_ndims - distribution.value_ndims)
        super(FlowDistribution, self).__init__(
            dtype=distribution.dtype,
            is_continuous=distribution.is_continuous,
            is_reparameterized=distribution.is_reparameterized,
            batch_shape=tmp_distrib.batch_shape,
            batch_static_shape=tmp_distrib.get_batch_shape(),
            value_ndims=flow.y_value_ndims,
        )

    @property
    def flow(self):
        """
        Get the transformation flow.

        Returns:
            BaseFlow: The transformation flow.
        """
        return self._flow

    @property
    def base_distribution(self):
        """
        Get the base distribution.

        Returns:
            Distribution: The base distribution to transform from.
        """
        return self._distribution

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        group_ndims = validate_group_ndims_arg(group_ndims)
        if not compute_density and compute_density is not None:
            raise RuntimeError('`FlowDistribution` requires `compute_prob` '
                               'not to be False.')

        with tf.name_scope(
                name, default_name='FlowDistribution.sample'):
            # x and log p(x)
            ndims_diff = (self.flow.x_value_ndims -
                          self.base_distribution.value_ndims)
            x = self._distribution.sample(
                n_samples=n_samples,
                group_ndims=ndims_diff,
                is_reparameterized=is_reparameterized,
                compute_density=True
            )
            log_px = x.log_prob()

            # y, log |dy/dx|
            is_reparameterized = x.is_reparameterized
            y, log_det = self._flow.transform(x)
            if not is_reparameterized:
                y = tf.stop_gradient(y)  # important!

            # compute log p(y) = log p(x) - log |dy/dx|
            # and then apply `group_ndims` on log p(y)
            log_py = reduce_group_ndims(
                tf.reduce_sum, log_px - log_det, group_ndims)

            # compose the transformed tensor
            return StochasticTensor(
                distribution=self,
                tensor=y,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized,
                log_prob=FlowDistributionDerivedTensor(
                    tensor=log_py,
                    flow_origin=x
                ),
                flow_origin=x
            )

    def log_prob(self, given, group_ndims=0, name=None):
        given = tf.convert_to_tensor(given)
        with tf.name_scope(
                name,
                default_name='FlowDistribution.log_prob',
                values=[given]):
            # x, log |dx/dy|
            x, log_det = self._flow.inverse_transform(given)

            # log p(x)
            ndims_diff = (self.flow.x_value_ndims -
                          self.base_distribution.value_ndims)
            log_px = self._distribution.log_prob(x, group_ndims=ndims_diff)

            # compute log p(y) = log p(x) + log |dx/dy|,
            # and then apply `group_ndims` on log p(x)
            log_py = reduce_group_ndims(
                tf.reduce_sum, log_px + log_det, group_ndims)

        return FlowDistributionDerivedTensor(
            tensor=log_py,
            flow_origin=StochasticTensor(
                distribution=self.base_distribution, tensor=x)
        )

    def prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(
                name, default_name=get_default_scope_name('prob', self)):
            log_p = self.log_prob(given, group_ndims=group_ndims)
            p = tf.exp(log_p)
        return FlowDistributionDerivedTensor(
            tensor=p, flow_origin=log_p.flow_origin)
