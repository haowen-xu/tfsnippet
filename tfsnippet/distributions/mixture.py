import tensorflow as tf

from tfsnippet.ops import log_sum_exp
from tfsnippet.stochastic import StochasticTensor
from tfsnippet.utils import (is_tensor_object, concat_shapes, get_shape,
                             settings, assert_deps)
from .base import Distribution
from .univariate import Categorical
from .utils import reduce_group_ndims, compute_density_immediately
from .wrapper import as_distribution

__all__ = ['Mixture']


class Mixture(Distribution):
    """
    Mixture distribution.

    Given a categorical distribution, and corresponding component distributions,
    this class derives a mixture distribution, formulated as follows:

    .. math::

        p(x) = \\sum_{k=1}^{K} \\pi(k) p_k(x)

    where :math:`\\pi(k)` is the probability of taking the k-th component,
    derived by the categorical distribution, and :math:`p_k(x)` is the density
    of the k-th component distribution.
    """

    def __init__(self, categorical, components, is_reparameterized=False):
        """
        Construct a new :class:`Mixture`.

        Args:
            categorical (Categorical): The categorical distribution,
                indicating the probabilities of the mixture components.
            components (Iterable[Distribution]): The component distributions
                of the mixture.
            is_reparameterized (bool): Whether or not this mixture distribution
                is re-parameterized?  If :obj:`True`, the `components` must
                all be re-parameterized.  The `categorical` will be treated
                as constant, and the mixture samples will be composed by
                `one_hot(categorical samples) * stack([component samples])`,
                such that the gradients can be propagated back directly
                through these samples.  If :obj:`False`, `tf.stop_gradient`
                will be applied on the mixture samples, such that no gradient
                will be propagated back through these samples.
        """
        components = tuple(as_distribution(c) for c in components)
        is_reparameterized = bool(is_reparameterized)

        if not isinstance(categorical, Categorical):
            raise TypeError(
                '`categorical` must be a Categorical distribution: got {}'.
                format(categorical)
            )
        if is_tensor_object(categorical.n_categories):
            raise ValueError(
                'Dynamic `categorical.n_categories` is not supported.')

        if not components:
            raise ValueError('`components` must not be empty.')
        if len(components) != categorical.n_categories:
            raise ValueError(
                '`len(components)` != `categorical.n_categories`: {} vs {}'.
                format(len(components), categorical.n_categories)
            )
        for i, c in enumerate(components):
            if is_reparameterized and not c.is_reparameterized:
                raise ValueError(
                    '`is_reparameterized` is True, but the {}-th component '
                    'is not re-parameterized: {}'.format(i, c)
                )

        for attr in ('dtype', 'is_continuous', 'value_ndims'):
            first_val = getattr(components[0], attr)
            for i, c in enumerate(components[1:], 1):
                c_val = getattr(c, attr)
                if c_val != first_val:
                    raise ValueError(
                        '`{}` of the {}-th component does not agree with the '
                        'first component: {} vs {}'.
                        format(attr, i, c_val, first_val)
                    )

        # check the batch_shape of components, ensure they are equal
        batch_shape = components[0].batch_shape
        batch_static_shape = components[0].get_batch_shape()

        def is_static_batch_shape_match(c, batch_static_shape):
            batch_static_shape = batch_static_shape.as_list()
            c_batch_static_shape = c.get_batch_shape().as_list()
            equal = True

            if len(batch_static_shape) != len(c_batch_static_shape):
                equal = False
            else:
                for a, b in zip(batch_static_shape, c_batch_static_shape):
                    if a is not None and b is not None and a != b:
                        equal = False
                        break

            return equal

        if not is_static_batch_shape_match(categorical, batch_static_shape):
            raise ValueError(
                'Batch shape of `categorical` does not agree with '
                'the first component: {} vs {}'.
                format(categorical.get_batch_shape(), batch_static_shape)
            )

        for i, c in enumerate(components[1:], 1):
            if not is_static_batch_shape_match(c, batch_static_shape):
                raise ValueError(
                    'Batch shape of the {}-th component does not agree with '
                    'the first component: {} vs {}'.
                    format(i, c.get_batch_shape(), batch_static_shape)
                )

        def assert_batch_shape(c, batch_shape):
            c_batch_shape = c.batch_shape
            with assert_deps([
                        tf.assert_equal(
                            tf.reduce_all(
                                tf.equal(
                                    tf.concat([batch_shape, c_batch_shape], 0),
                                    tf.concat([c_batch_shape, batch_shape], 0)
                                )
                            ),
                            True
                        )
                    ]) as asserted:
                if asserted:  # pragma: no cover
                    batch_shape = tf.identity(batch_shape)
            return batch_shape

        if settings.enable_assertions:
            with tf.name_scope('Mixture.init'):
                batch_shape = assert_batch_shape(categorical, batch_shape)
                for c in components[1:]:
                    batch_shape = assert_batch_shape(c, batch_shape)

        self._categorical = categorical
        self._components = components

        super(Mixture, self).__init__(
            dtype=components[0].dtype,
            is_continuous=components[0].is_continuous,
            is_reparameterized=is_reparameterized,
            batch_shape=components[0].batch_shape,
            batch_static_shape=components[0].get_batch_shape(),
            value_ndims=components[0].value_ndims,
        )

    @property
    def categorical(self):
        """
        Get the categorical distribution of this mixture.

        Returns:
            Categorical: The categorical distribution.
        """
        return self._categorical

    @property
    def components(self):
        """
        Get the mixture components of this distribution.

        Returns:
            tuple[Distribution]: The mixture components.
        """
        return self._components

    @property
    def n_components(self):
        """
        Get the number of mixture components.

        Returns:
            int: The number of mixture components.
        """
        return len(self._components)

    def _cat_prob(self, log_softmax):
        softmax_fn = tf.nn.log_softmax if log_softmax else tf.nn.softmax
        probs = softmax_fn(self._categorical.logits, axis=-1, name='cat_prob')
        return tf.unstack(probs, num=self.n_components, axis=-1)

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        self._validate_sample_is_reparameterized_arg(is_reparameterized)

        #######################################################################
        # slow routine: generate the mixture by one_hot * stack([c.sample()]) #
        #######################################################################
        with tf.name_scope(name or 'Mixture.sample'):
            cat = self.categorical.sample(n_samples, group_ndims=0)
            mask = tf.one_hot(cat, self.n_components, dtype=self.dtype, axis=-1)
            if self.value_ndims > 0:
                static_shape = (mask.get_shape().as_list() +
                                [1] * self.value_ndims)
                dynamic_shape = concat_shapes([get_shape(mask),
                                               [1] * self.value_ndims])
                mask = tf.reshape(mask, dynamic_shape)
                mask.set_shape(static_shape)
            mask = tf.stop_gradient(mask)

            # derive the mixture samples
            c_samples = [
                c.sample(n_samples, group_ndims=0)
                for c in self.components
            ]
            samples = tf.reduce_sum(
                mask * tf.stack(c_samples, axis=-self.value_ndims - 1),
                axis=-self.value_ndims - 1
            )

            if not self.is_reparameterized:
                samples = tf.stop_gradient(samples)

            t = StochasticTensor(
                distribution=self,
                tensor=samples,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized
            )

            if compute_density:
                compute_density_immediately(t)

            return t

    def log_prob(self, given, group_ndims=0, name=None):
        given = tf.convert_to_tensor(given)

        with tf.name_scope(name or 'Mixture.log_prob', values=[given]):
            cat_log_probs = self._cat_prob(log_softmax=True)
            c_log_probs = [
                c.log_prob(given, group_ndims=0)
                for c in self.components
            ]
            log_probs = tf.stack(
                [cat + c for cat, c in zip(cat_log_probs, c_log_probs)],
                axis=0
            )
            log_prob = log_sum_exp(log_probs, axis=0)
            log_prob = reduce_group_ndims(
                tf.reduce_sum, log_prob, group_ndims=group_ndims)
            return log_prob
