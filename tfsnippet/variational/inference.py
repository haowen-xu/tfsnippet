import tensorflow as tf
import zhusuan as zs

__all__ = [
    'VariationalInference',
    'VariationalLowerBounds',
    'VariationalTrainingObjectives',
    'VariationalEvaluation'
]


def _require_multi_samples(vi, name):
    if vi.axis is None:
        raise ValueError('{} requires multi-samples of latent variables, '
                         'thus the `axis` argument must be specified'.
                         format(name))


class VariationalInference(object):
    """
    Class for variational inference.

    The interface of `ZhuSuan`_ for variational inference tightly binds
    the construction of model and the deriving of lower-bounds / training
    objectives together.  This causes the following two problems.

    First, :class:`zhusuan.variational.VariationalObjective` requires a
    `log_joint` function, which constructs the computation graph of the model,
    only for obtaining the log-joint.  The :class:`BayesianNet` instance of
    the model constructed in `log_joint` cannot be obtained by outside caller.
    If we also need other statistics from the model, we shall have to build
    the model again, which is very wasteful.

    Second, such interface is not very friendly for implementation of a
    Bayesian network as :class:`~tfsnippet.module.Module`.

    We thus provide this class to separate the construction of model and the
    variational inference stage.  It wraps the variational inference algorithms
    from `ZhuSuan`_, providing a virtual `log_joint` function, which simply
    returns the pre-computed log-joint :class:`tf.Tensor`.

    .. _`ZhuSuan`: https://github.com/thu-ml/zhusuan
    """

    def __init__(self, log_joint, latent_log_probs, axis=None):
        """
        Construct the :class:`VariationalInference`.

        Args:
            log_joint (tf.Tensor): The log-joint of model.
            latent_log_probs (Iterable[tf.Tensor]): The log-probability
                densities of latent variables from the variational net.
            axis: The axis or axes to be considered as the sampling dimensions
                of latent variables.  The specified axes will be summed up in
                the variational lower-bounds or training objectives.
                (default :obj:`None`)
        """
        self._log_joint = tf.convert_to_tensor(log_joint)
        self._latent_log_probs = tuple(tf.convert_to_tensor(t)
                                       for t in latent_log_probs)
        self._axis = axis
        self._lower_bound = VariationalLowerBounds(self)
        self._training = VariationalTrainingObjectives(self)
        self._evaluation = VariationalEvaluation(self)

    @property
    def log_joint(self):
        """
        Get the log-joint of the model.

        Returns:
            tf.Tensor: The log-joint of the model.
        """
        return self._log_joint

    @property
    def latent_log_probs(self):
        """
        Get the log-probability densities of latent variables.

        Returns:
            tuple[tf.Tensor]: The log-probability densities of latent variables.
        """
        return self._latent_log_probs

    @property
    def axis(self):
        """
        Get the axis or axes to be considered as the sampling dimensions
        of latent variables.
        """
        return self._axis

    def zs_objective(self, func, **kwargs):
        """
        Create a :class:`zhusuan.variational.VariationalObjective` with
        pre-computed log-joint, by specified algorithm.

        Args:
            func: The variational algorithm from `ZhuSuan`_. Supported
                functions are: 1. :func:`zhusuan.variational.elbo`
                2. :func:`zhusuan.variational.importance_weighted_objective`
                3. :func:`zhusuan.variational.klpq`
            \**kwargs: Named arguments passed to `func`.

        Returns:
            zhusuan.variational.VariationalObjective: The constructed
                per-data variational objective.
        """
        return func(
            log_joint=lambda observed: self._log_joint,
            observed={},
            latent={i: (None, log_prob)
                    for i, log_prob in enumerate(self._latent_log_probs)},
            axis=self._axis,
            **kwargs
        )

    def zs_elbo(self):
        """
        Create a :class:`zhusuan.variational.EvidenceLowerBoundObjective`,
        with pre-computed log-joint.

        Returns:
            zhusuan.variational.EvidenceLowerBoundObjective: The constructed
                per-data ELBO objective.
        """
        return self.zs_objective(zs.variational.elbo)

    def zs_importance_weighted_objective(self):
        """
        Create a :class:`zhusuan.variational.ImportanceWeightedObjective`,
        with pre-computed log-joint.

        Returns:
            zhusuan.variational.ImportanceWeightedObjective: The constructed
                per-data importance weighted objective.
        """
        return self.zs_objective(zs.variational.importance_weighted_objective)

    def zs_klpq(self):
        """
        Create a :class:`zhusuan.variational.InclusiveKLObjective`,
        with pre-computed log-joint.

        Returns:
            zhusuan.variational.InclusiveKLObjective: The constructed
                per-data inclusive KL objective.
        """
        return self.zs_objective(zs.variational.klpq)

    @property
    def lower_bound(self):
        """
        Get the factory for variational lower-bounds.

        Returns:
            VariationalLowerBounds: The factory for variational lower-bounds.
        """
        return self._lower_bound

    @property
    def training(self):
        """
        Get the factory for training objectives.

        Returns:
            VariationalTrainingObjectives: The factory for training objectives.
        """
        return self._training

    @property
    def evaluation(self):
        """
        Get the factory for evaluation outputs.

        Returns:
            VariationalEvaluation: The factory for evaluation outputs.
        """
        return self._evaluation


class VariationalLowerBounds(object):
    """Factory for variational lower-bounds."""

    def __init__(self, vi):
        """
        Construct a new :class:`VariationalEvaluation`.

        Args:
            vi (VariationalInference): The variational inference object.
        """
        self._vi = vi

    def elbo(self, name=None):
        """
        Get the evidence lower-bound.

        Args:
            name (str): Name of this operation in TensorFlow graph.
                (default "elbo")

        Returns:
            tf.Tensor: The evidence lower-bound.

        See Also:
            :class:`zhusuan.variational.EvidenceLowerBoundObjective`
        """
        with tf.name_scope(name, default_name='elbo'):
            return self._vi.zs_elbo().tensor

    def importance_weighted_objective(self, name=None):
        """
        Get the importance weighted lower-bound.

        Args:
            name (str): Name of this operation in TensorFlow graph.
                (default "importance_weighted_objective")

        Returns:
            tf.Tensor: The per-data importance weighted lower-bound.

        See Also:
            :class:`zhusuan.variational.ImportanceWeightedObjective`
        """
        _require_multi_samples(self._vi, 'importance weighted lower-bound')
        with tf.name_scope(name, default_name='importance_weighted_objective'):
            return self._vi.zs_importance_weighted_objective().tensor


class VariationalTrainingObjectives(object):
    """Factory for variational training objectives."""

    def __init__(self, vi):
        """
        Construct a new :class:`VariationalEvaluation`.

        Args:
            vi (VariationalInference): The variational inference object.
        """
        self._vi = vi

    def sgvb(self, name=None):
        """
        Get the SGVB training objective.

        Args:
            name (str): Name of this operation in TensorFlow graph.
                (default "sgvb")

        Returns:
            tf.Tensor: The per-data SGVB training objective.

        See Also:
            :meth:`zhusuan.variational.EvidenceLowerBoundObjective.sgvb`
        """
        with tf.name_scope(name, default_name='sgvb'):
            return self._vi.zs_elbo().sgvb()

    def reinforce(self, variance_reduction=True, baseline=None, decay=0.8,
                  name=None):
        """
        Get the REINFORCE training objective.

        Args:
            variance_reduction (bool): Whether to use variance reduction.
            baseline (tf.Tensor): A trainable estimation for the scale of
                the elbo value.
            decay (float): The moving average decay for variance normalization.
            name (str): Name of this operation in TensorFlow graph.
                (default "reinforce")

        Returns:
            tf.Tensor: The per-data REINFORCE training objective.

        See Also:
            :meth:`zhusuan.variational.EvidenceLowerBoundObjective.reinforce`
        """
        # reinforce requires extra variables to collect the moving average
        # statistics, so we need to generate a variable scope
        with tf.variable_scope(name, default_name='reinforce'):
            return self._vi.zs_elbo().reinforce(
                variance_reduction=variance_reduction,
                baseline=baseline,
                decay=decay,
            )

    def iwae(self, name=None):
        """
        Get the SGVB training objective for importance weighted objective.

        Args:
            name (str): Name of this operation in TensorFlow graph.
                (default "iwae")

        Returns:
            tf.Tensor: The per-data SGVB training objective for importance
                weighted objective.

        See Also:
            :meth:`zhusuan.variational.ImportanceWeightedObjective.sgvb`
        """
        _require_multi_samples(self._vi, 'iwae training objective')
        with tf.name_scope(name, default_name='iwae'):
            return self._vi.zs_importance_weighted_objective().sgvb()

    def vimco(self, name=None):
        """
        Get the VIMCO training objective.

        Args:
            name (str): Name of this operation in TensorFlow graph.
                (default "vimco")

        Returns:
            tf.Tensor: The per-data VIMCO training objective.

        See Also:
            :meth:`zhusuan.variational.ImportanceWeightedObjective.vimco`
        """
        _require_multi_samples(self._vi, 'vimco training objective')
        with tf.name_scope(name, default_name='vimco'):
            return self._vi.zs_importance_weighted_objective().vimco()

    def rws_wake(self, name=None):
        """
        Get the wake-phase Reweighted Wake-Sleep (RWS) training objective.

        Args:
            name (str): Name of this operation in TensorFlow graph.
                (default "rws_wake")

        Returns:
            tf.Tensor: The per-data wake-phase RWS training objective.

        See Also:
            :meth:`zhusuan.variational.InclusiveKLObjective.rws`
        """
        _require_multi_samples(
            self._vi, 'reweighted wake-sleep training objective')
        with tf.name_scope(name, default_name='rws_wake'):
            return self._vi.zs_klpq().rws()


class VariationalEvaluation(object):
    """Factory for variational evaluation outputs."""

    def __init__(self, vi):
        """
        Construct a new :class:`VariationalEvaluation`.

        Args:
            vi (VariationalInference): The variational inference object.
        """
        self._vi = vi

    def importance_sampling_log_likelihood(self, name=None):
        """
        Compute :math:`log p(x)` by importance sampling.

        Args:
            name (str): Name of this operation in TensorFlow graph.
                (default "sgvb")

        Returns:
            tf.Tensor: The per-data :math:`log p(x)`.

        See Also:
            :meth:`zhusuan.evaluation.is_loglikelihood`
        """
        _require_multi_samples(
            self._vi, 'importance sampling log-likelihood')
        with tf.name_scope(
                name, default_name='importance_sampling_log_likelihood'):
            return self._vi.zs_objective(zs.evaluation.is_loglikelihood)

    is_loglikelihood = importance_sampling_log_likelihood
    """Short-cut for :meth:`importance_sampling_log_likelihood`."""
