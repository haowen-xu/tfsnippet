# -*- coding: utf-8 -*-
import tensorflow as tf

from tfsnippet.utils import DocInherit, get_default_scope_name

__all__ = ['Distribution']


@DocInherit
class Distribution(object):
    """
    Base class for probability distributions.

    A :class:`Distribution` object receives inputs as distribution parameters,
    generating samples and computing densities according to these inputs.
    The shape of the inputs can have more dimensions than the nature shape
    of the distribution parameters, since :class:`Distribution` is designed
    to work with batch parameters, samples and densities.

    The shape of the parameters of a :class:`Distribution` object would be
    decomposed into ``batch_shape + param_shape``, with `param_shape` being
    the nature shape of the parameter.  For example, a 5-class
    :class:`Categorical` distribution with class probabilities of shape
    ``(3, 4, 5)`` would have ``(3, 4)`` as the `batch_shape`, with ``(5,)``
    as the `param_shape`, corresponding to the probabilities of 5 classes.

    Generating `n` samples from a :class:`Distribution` object would result
    in tensors with shape ``[n] (sample_shape) + batch_shape + value_shape``,
    with ``value_shape`` being the nature shape of an individual sample from
    the distribution.  For example, the `value_shape` of a :class:`Categorical`
    is ``()``, such that the sample shape would be ``(3, 4)``, provided the
    shape of class probabilities is ``(3, 4, 5)``.

    Computing the densities (i.e., `prob(x)` or `log_prob(x)`) of samples
    involves broadcasting these samples against the distribution parameters.
    These samples should be broadcastable against ``batch_shape + value_shape``.
    Suppose the shape of the samples can be decomposed into
    ``sample_shape + batch_shape + value_shape``, then by default, the shape of
    the densities should be ``sample_shape + batch_shape``, i.e., each
    individual sample resulting in an individual density value.
    """

    def __init__(self, dtype, is_continuous, is_reparameterized, batch_shape,
                 batch_static_shape, value_ndims):
        assert(isinstance(batch_static_shape, tf.TensorShape))
        # currently we do not support non-deterministic ndims
        assert(batch_static_shape.ndims is not None)
        # assert(isinstance(value_static_shape, tf.TensorShape))
        # value_ndims = value_static_shape.ndims
        assert(value_ndims is not None)

        self._dtype = dtype
        self._is_continuous = is_continuous
        self._is_reparameterized = is_reparameterized
        self._batch_shape = batch_shape
        self._batch_static_shape = batch_static_shape
        # self._value_shape = value_shape
        # self._value_static_shape = value_static_shape
        self._value_ndims = value_ndims

    @property
    def dtype(self):
        """
        Get the data type of samples.

        Returns:
            tf.DType: Data type of the samples.
        """
        return self._dtype

    @property
    def is_continuous(self):
        """
        Whether or not the distribution is continuous?

        Returns:
            bool: A boolean indicating whether it is continuous.
        """
        return self._is_continuous

    @property
    def is_reparameterized(self):
        """
        Whether or not the distribution is re-parameterized?

        The re-parameterization trick is proposed in "Auto-Encoding Variational
        Bayes" (Kingma, D.P. and Welling), allowing the gradients to be
        propagated back along the samples.  Note that the re-parameterization
        can be disabled by specifying ``is_reparameterized = False`` as an
        argument of :meth:`sample`.

        Returns:
            bool: A boolean indicating whether it is re-parameterized.
        """
        return self._is_reparameterized

    @property
    def value_ndims(self):
        """
        Get the number of value dimensions in samples.

        Returns:
            int: The number of value dimensions in samples.
        """
        return self._value_ndims

    @property
    def base_distribution(self):
        """
        Get the base distribution of this distribution.

        For distribution other than :class:`tfsnippet.BatchToValueDistribution`,
        this property should return this distribution itself.

        Returns:
            Distribution: The base distribution.
        """
        return self

    def expand_value_ndims(self, ndims):
        """
        Convert the last few `batch_ndims` into `value_ndims`.

        For a particular :class:`Distribution`, the number of dimensions
        between the samples and the log-probability of the samples should
        satisfy::

            samples.ndims - distribution.value_ndims == log_det.ndims

        We denote `samples.ndims - distribution.value_ndims` by `batch_ndims`.
        This method thus wraps the current distribution, converts the last
        few `batch_ndims` into `value_ndims`.

        Args:
            ndims (int): The last few `batch_ndims` to be converted into
                `value_ndims`.  Must be non-negative.

        Returns:
            Distribution: The converted distribution.
        """
        from .batch_to_value import BatchToValueDistribution
        ndims = int(ndims)
        if ndims == 0:
            return self
        return BatchToValueDistribution(self, ndims)

    batch_ndims_to_value = expand_value_ndims

    # TODO: implement value_shape and get_value_shape()

    # @property
    # def value_shape(self):
    #     """
    #     Get the value shape of an individual sample.
    #
    #     Returns:
    #         tf.Tensor: The value shape as tensor.
    #     """
    #     return self._value_shape
    #
    # def get_value_shape(self):
    #     """
    #     Get the static value shape of an individual sample.
    #
    #     Returns:
    #         tf.TensorShape: The static value shape.
    #     """
    #     return self._value_static_shape

    @property
    def batch_shape(self):
        """
        Get the batch shape of the samples.

        Returns:
            tf.Tensor: The batch shape as tensor.
        """
        return self._batch_shape

    def get_batch_shape(self):
        """
        Get the static batch shape of the samples.

        Returns:
            tf.TensorShape: The batch shape.
        """
        return self._batch_static_shape

    def _validate_sample_is_reparameterized_arg(self, is_reparameterized):
        if is_reparameterized and not self.is_reparameterized:
            raise RuntimeError('{} is not re-parameterized.'.format(self))

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        """
        Generate samples from the distribution.

        Args:
            n_samples (int or tf.Tensor or None): A 0-D `int32` Tensor or None.
                How many independent samples to draw from the distribution.
                The samples will have shape ``[n_samples] + batch_shape +
                value_shape``, or ``batch_shape + value_shape`` if `n_samples`
                is :obj:`None`.
            group_ndims (int or tf.Tensor): Number of dimensions at the end of
                ``[n_samples] + batch_shape`` to be considered as events group.
                This will effect the behavior of :meth:`log_prob` and
                :meth:`prob`. (default 0)
            is_reparameterized (bool): If :obj:`True`, raises
                :class:`RuntimeError` if the distribution is not
                re-parameterized.  If :obj:`False`, disable re-parameterization
                even if the distribution is re-parameterized.
                (default :obj:`None`, following the setting of distribution)
            compute_density (bool): Whether or not to immediately compute the
                log-density for the samples? (default :obj:`None`, determine by
                the distribution class itself)
            name: TensorFlow name scope of the graph nodes.
                (default "sample").

        Returns:
            tfsnippet.stochastic.StochasticTensor: The samples as
                :class:`~tfsnippet.stochastic.StochasticTensor`.
        """
        raise NotImplementedError()

    def log_prob(self, given, group_ndims=0, name=None):
        """
        Compute the log-densities of `x` against the distribution.

        Args:
            given (Tensor): The samples to be tested.
            group_ndims (int or tf.Tensor): If specified, the last `group_ndims`
                dimensions of the log-densities will be summed up.
                (default 0)
            name: TensorFlow name scope of the graph nodes.
                (default "log_prob").

        Returns:
            tf.Tensor: The log-densities of `given`.
        """
        raise NotImplementedError()

    def prob(self, given, group_ndims=0, name=None):
        """
        Compute the densities of `x` against the distribution.

        Args:
            given (Tensor): The samples to be tested.
            group_ndims (int or tf.Tensor): If specified, the last `group_ndims`
                dimensions of the log-densities will be summed up. (default 0)
            name: TensorFlow name scope of the graph nodes.
                (default "prob").

        Returns:
            tf.Tensor: The densities of `given`.
        """
        with tf.name_scope(
                name, default_name=get_default_scope_name('prob', self)):
            return tf.exp(self.log_prob(given, group_ndims=group_ndims))
