from collections import OrderedDict

import six
import zhusuan
import tensorflow as tf
from frozendict import frozendict

from tfsnippet.distributions import Distribution, as_distribution
from tfsnippet.stochastic import StochasticTensor

__all__ = ['BayesianNet']


class TransformedDistribution(Distribution):
    """
    Mimic the :class:`Distribution` interface for a transformed variable.

    The :class:`BayesianNet` interface requires each :class:`StochasticTensor`
    to be bound with a :class:`Distribution`, however, for the transformed
    variables, the distribution is not fully defined.  Thus we provide
    this class to mimic the basic interface of a :class:`Distribution`,
    which is required for :class:`BayesianNet`.
    """

    def __init__(self, origin, transformed, transformed_log_p,
                 is_reparameterized, is_continuous):
        """
        Construct a :class:`TransformedDistribution`.

        Args:
            origin (StochasticTensor): The original sample or observation.
            transformed (tf.Tensor): The transformed sample or observation.
            transformed_log_p (tf.Tensor): The transformed log-likelihood.
            is_reparameterized (bool): Whether the transformed distribution
                is is_reparameterized?
            is_continuous (bool): Whether the transformed distribution
                is continuous?
        """
        self._origin = origin
        self._transformed = transformed
        self._transformed_log_p = transformed_log_p
        self._is_reparameterized = is_reparameterized
        self._is_continuous = is_continuous

    @property
    def origin(self):
        """
        Get the original sample or observation.

        Returns:
            StochasticTensor: The original sample or observation.
        """
        return self._origin

    @property
    def transformed(self):
        """
        Get the transformed sample or observation.

        Returns:
            tf.Tensor: The transformed sample or observation.
        """
        return self._transformed

    @property
    def transformed_log_p(self):
        """
        Get the transformed log-likelihood.

        Returns:
            The transformed log-likelihood.
        """
        return self._transformed_log_p

    @property
    def dtype(self):
        return self.transformed.dtype

    @property
    def is_reparameterized(self):
        return self._is_reparameterized

    @property
    def is_continuous(self):
        return self._is_continuous

    def _check_given(self, given, group_ndims):
        if given is not self.transformed or \
                group_ndims != self.origin.group_ndims:
            raise ValueError('`given` must be `self.transformed` and '
                             '`group_ndims` must be `self.origin.group_ndims`.')

    def log_prob(self, given, group_ndims=0, name=None):
        self._check_given(given, group_ndims)
        return self.transformed_log_p

    def prob(self, given, group_ndims=0, name=None):
        self._check_given(given, group_ndims)
        with tf.name_scope(name=name, default_name='prob'):
            return tf.exp(self.log_prob(given, group_ndims))


class BayesianNet(object):
    """
    Bayesian networks.

    :class:`BayesianNet` is a class which helps to construct Bayesian
    networks and to derive the variational lower-bounds.
    It re-implements similar interfaces as :class:`zhusuan.BayesianNet`,
    but is more friendly to :class:`~tfsnippet.modules.Module` interface.

    Due to the expressive limitations of TensorFlow, it is hard to build
    :class:`BayesianNet` with the concept of `random variables`.
    Instead, we only collect :class:`StochasticTensor` objects, i.e.,
    tensors sampled from the distributions of these random variables.
    Thus :class:`BayesianNet` is actually a collection of (multiple)
    ancestral samples from the random variables.
    Fortunately, we can approximate most interested statistics of the desired
    random variables with these samples, by using Monte Carlo methods.
    For example, obtaining the expectation of a random variable by averaging
    over multiple samples from it.
    The :class:`StochasticTensor` objects are called `stochastic nodes`
    within the context of :class:`BayesianNet`.

    To build a Bayesian network, first obtain a :class:`BayesianNet`:

    .. code-block:: python

        net = tfsnippet.bayes.BayesianNet()

    Then add stochastic nodes into the network:

    A Bayesian Linear Regression example, as of :class:`zhusuan.BayesianNet`:

    .. math::

        w \\sim N(0, \\alpha^2 I)

        y \\sim N(w^Tx, \\beta^2)

    .. code-block:: python

        from tfsnippet.bayes import BayesianNet()
        from tfsnippet.distributions import Normal

        def bayesian_linear_regression(x, alpha, beta, observed=None):
            net = BayesianNet(observed)
            w = net.add('w', Normal(mean=0., logstd=tf.log(alpha)))
            y_mean = tf.reduce_sum(tf.expand_dims(w, 0) * x, 1)
            y = net.add('y', Normal(mean=y_mean, logstd=tf.log(beta)))
            return net

    To observe any stochastic nodes in the network, pass a dictionary mapping
    of ``(name, Tensor)`` as `observed` when constructing :class:`BayesianNet`.
    For example:

    .. code-block:: python

        model = bayesian_linear_regression(..., observed={'w': w_obs})

    After construction, :class:`BayesianNet` supports queries on the network.

    .. code-block:: python

        # get samples of random variable y following generative process
        # in the network
        model.output('y')

        # because w is observed in this case, its observed value will be
        # returned
        model.output('w')

        # also multiple outputs can be fetched together
        model.outputs(['y', 'w'])

        # get local log probability values of w and y, which returns
        # log p(w) and log p(y|w, x)
        model.local_log_probs(['w', 'y'])

        # query many quantities at the same time
        model.query(['w', 'y'])

    See Also:
        :class:`zhusuan.BayesianNet`
    """

    def __init__(self, observed=None):
        """
        Construct the :class:`BayesianNet`.

        Args:
            observed: Dict of ``(str, tf.Tensor)``, the names of stochastic
                nodes and their observations.
        """
        super(BayesianNet, self).__init__()
        self._observed = frozendict([
            (name, tf.convert_to_tensor(tensor))
            for name, tensor in (six.iteritems(observed) if observed else ())
        ])
        self._stochastic_tensors = OrderedDict()

    @property
    def observed(self):
        """
        Get the read-only dict of observations.

        Returns:
            collections.Mapping[str, tf.Tensor]: The read-only dict of
                observations.
        """
        return self._observed

    def _check_names_exist(self, names):
        names = tuple(names)
        for name in names:
            if not isinstance(name, six.string_types):
                raise TypeError('`names` is not a list of str')
            if name not in self._stochastic_tensors:
                raise KeyError('StochasticTensor with name {!r} does not exist'.
                               format(name))
        return names

    def add(self, name, distribution, n_samples=None, group_ndims=0,
            is_reparameterized=None, transform=None):
        """
        Add a stochastic node to the network.

        A :class:`StochasticTensor` will be created for this node.
        If `name` exists in `observed` dict, its value will be used as the
        observation of this node.  Otherwise samples will be taken from
        `distribution`.

        Args:
            name (str): Name of the stochastic node.
            distribution (Distribution or zhusuan.distributions.Distribution):
                Distribution where the samples should be taken from.
            n_samples (int or tf.Tensor): Number of samples to take.
                If specified, `n_samples` will be taken, with a dedicated
                sampling dimension ``[n_samples]`` at the front.
                If not specified, just one sample will be taken, without the
                dedicated dimension.
            group_ndims (int or tf.Tensor): Number of dimensions at the end of
                ``[n_samples] + batch_shape`` to be considered as events group.
                (default 0)
            is_reparameterized: Whether or not the re-parameterization trick
                should be applied? (default :obj:`None`, following the setting
                of `distribution`)
            transform ((Tensor, Tensor) -> (tf.Tensor, tf.Tensor)):
                The function to transform (x, log_p) to (x', log_p').
                If specified, a :class:`StochasticTensor` will be sampled,
                then transformed, then wrapped by a :class:`StochasticTensor`
                with :class:`TransformedDistribution`.

        Returns:
            StochasticTensor: The sampled stochastic tensor.

        Raises:
            TypeError: If `name` is not a str, or `distribution` is a
                :class:`TransformedDistribution`.
            KeyError: If :class:`StochasticTensor` with `name` already exists.
            ValueError: If `transform` cannot be applied.

        See Also:
            :meth:`tfsnippet.distributions.Distribution.sample`
        """
        if not isinstance(name, six.string_types):
            raise TypeError('`name` must be a str')
        if name in self._stochastic_tensors:
            raise KeyError('StochasticTensor with name {!r} already exists in '
                           'the BayesianNet.  Names must be unique.'.
                           format(name))
        if isinstance(distribution, TransformedDistribution):
            raise TypeError('Cannot add `TransformedDistribution`.')
        if transform is not None and \
                (not distribution.is_continuous or
                 not distribution.is_reparameterized or
                 is_reparameterized is False):
            raise ValueError('`transform` can only be applied on continuous, '
                             're-parameterized variables.')
        if transform is not None and name in self._observed:
            raise ValueError('`observed` variable cannot be transformed.')

        distribution = as_distribution(distribution)
        if name in self._observed:
            t = StochasticTensor(
                distribution=distribution,
                tensor=self._observed[name],
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized,
            )
        else:
            t = distribution.sample(
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized,
            )
            assert(isinstance(t, StochasticTensor))

            # do transformation
            if transform is not None:
                t_log_p = t.log_prob()
                ft, ft_log_p = transform(t, t_log_p)
                ft = tf.convert_to_tensor(ft)
                ft_log_p = tf.convert_to_tensor(ft_log_p)
                if not ft.dtype.is_floating:
                    raise ValueError('The transformed samples must be '
                                     'continuous: got {!r}'.format(ft))
                t = StochasticTensor(
                    distribution=TransformedDistribution(
                        origin=t,
                        transformed=ft,
                        transformed_log_p=ft_log_p,
                        is_reparameterized=t.is_reparameterized,
                        is_continuous=True
                    ),
                    tensor=ft,
                    n_samples=t.n_samples,
                    group_ndims=t.group_ndims,
                    is_reparameterized=t.is_reparameterized
                )

        self._stochastic_tensors[name] = t
        return t

    def get(self, name):
        """
        Get :class:`StochasticTensor` of a stochastic node.

        Args:
            name (str): Name of the queried stochastic node.

        Returns:
            StochasticTensor: :class:`StochasticTensor` of the queried node,
                or :obj:`None` if no node exists with `name`.
        """
        return self._stochastic_tensors.get(name)

    def __getitem__(self, name):
        """
        Get :class:`StochasticTensor` of a stochastic node.

        Args:
            name (str): Name of the queried stochastic node.

        Returns:
            StochasticTensor: :class:`StochasticTensor` of the queried node.

        Raises:
            KeyError: If non-exist name is queried.
        """
        self._check_names_exist((name,))
        return self._stochastic_tensors[name]

    def __contains__(self, name):
        """Test whether or not a stochastic node with `name` exists."""
        return name in self._stochastic_tensors

    def __iter__(self):
        """Get an iterator of the stochastic node names."""
        return iter(self._stochastic_tensors)

    def outputs(self, names):
        """
        Get the outputs of stochastic nodes.
        The output of a stochastic node is its :attr:`StochasticTensor.tensor`.

        Args:
            names (Iterable[str]): Names of the queried stochastic nodes.

        Returns:
            list[tf.Tensor]: Outputs of the queried stochastic nodes.

        Raises:
            KeyError: If non-exist name is queried.
        """
        names = self._check_names_exist(names)
        return [self._stochastic_tensors[n].tensor for n in names]

    def output(self, name):
        """
        Get the output of a stochastic node.
        The output of a stochastic node is its :attr:`StochasticTensor.tensor`.

        Args:
            name (str): Name of the queried stochastic node.

        Returns:
            tf.Tensor: Output of the queried stochastic node.

        Raises:
            KeyError: If non-exist name is queried.
        """
        return self.outputs((name,))[0]

    def local_log_probs(self, names):
        """
        Get the log-probability densities of stochastic nodes.

        Args:
            names (Iterable[str]): Names of the queried stochastic nodes.

        Returns:
            list[tf.Tensor]: Log-probability densities of the queried stochastic
                nodes.

        Raises:
            KeyError: If non-exist name is queried.
        """
        names = self._check_names_exist(names)
        return [self._stochastic_tensors[n].log_prob() for n in names]

    def local_log_prob(self, name):
        """
        Get the log-probability density of a stochastic node.

        Args:
            name (str): Name of the queried stochastic node.

        Returns:
            tf.Tensor: Log-probability density of the queried stochastic node.

        Raises:
            KeyError: If non-exist name is queried.
        """
        return self.local_log_probs((name,))[0]

    def query(self, names):
        """
        Get the outputs and log-probability densities of stochastic node(s).

        Args:
            names (Iterable[str]): Names of the queried stochastic nodes.

        Returns:
            list[(tf.Tensor, tf.Tensor)]: Tuples of `(output, log-prob)` of the
                queried stochastic nodes.

        Raises:
            KeyError: If non-exist name is queried.
        """
        names = self._check_names_exist(names)
        return list(zip(self.outputs(names), self.local_log_probs(names)))

    def variational_chain(self, model_builder, latent_names=None,
                          latent_axis=None, observed=None, **kwargs):
        """
        Treat this :class:`BayesianNet` as variational, and build the model
        net chained after this variational net.

        Args:
            model_builder: Function which receives the `observed` dict, and
                produce the model :class:`BayesianNet` or a tuple of the model
                :class:`BayesianNet` and the log-joint of the model.
            latent_names (Iterable[str]): Names of the nodes to be considered
                as latent variables in this :class:`BayesianNet`.  All these
                variables will be fed into `model_builder` as observed
                variables, overriding the observations in `observed`.
                (default all the variables in this :class:`BayesianNet`)
            latent_axis: The axis or axes to be considered as the sampling
                dimensions of latent variables.  The specified axes will be
                summed up in the variational lower-bounds or training
                objectives. (default :obj:`None`)
            observed: Dict of ``(name, observation)`` fed into `model_builder`.
                (default :obj:`None`)
            \**kwargs: Additional named arguments passed to `model_builder`.

        Returns:
            tfsnippet.variational.VariationalChain: The object that holds this
                :class:`BayesianNet` as the `variational` net, the constructed
                `model` net, and the
                :class:`~tfsnippet.variational.VariationalInference` object
                for obtaining the variational lower-bounds and training
                objectives.

        See Also:
            :class:`tfsnippet.variational.VariationalChain`
        """
        from tfsnippet.variational.chain import VariationalChain

        # build the observed dict: observed + latent samples
        merged_obs = {}
        # add the user-provided observed dict
        if observed:
            merged_obs.update(observed)
        # add the latent samples
        if latent_names is None:
            latent_names = tuple(self)
        else:
            latent_names = tuple(latent_names)
        merged_obs.update({
            n: t
            for n, t in zip(latent_names, self.outputs(latent_names))
        })

        # build the model and its log-joint
        model_and_log_joint = model_builder(merged_obs, **kwargs)
        if isinstance(model_and_log_joint, tuple):
            model, log_joint = model_and_log_joint
        else:
            model, log_joint = model_and_log_joint, None

        # build the VariationalModelChain
        return VariationalChain(
            variational=self,
            model=model,
            log_joint=log_joint,
            latent_names=latent_names,
            latent_axis=latent_axis,
        )

    chain = variational_chain
    """Alias for :meth:`variational_chain`."""
