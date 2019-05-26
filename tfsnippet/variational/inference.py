import tensorflow as tf

from tfsnippet.ops import add_n_broadcast
from tfsnippet.utils import add_name_arg_doc
from .estimators import *
from .evaluation import *
from .objectives import *
from .utils import _require_multi_samples

__all__ = [
    'VariationalInference',
    'VariationalLowerBounds',
    'VariationalTrainingObjectives',
    'VariationalEvaluation'
]


class VariationalInference(object):
    """Class for variational inference."""

    def __init__(self, log_joint, latent_log_probs, axis=None):
        """
        Construct the :class:`VariationalInference`.

        Args:
            log_joint (tf.Tensor): The log-joint of model.
            latent_log_probs (Iterable[tf.Tensor]): The log-densities
                of latent variables from the variational net.
            axis: The axis or axes to be considered as the sampling dimensions
                of latent variables.  The specified axes will be summed up in
                the variational lower-bounds or training objectives.
                (default :obj:`None`)
        """
        self._log_joint = tf.convert_to_tensor(log_joint)
        self._latent_log_probs = tuple(tf.convert_to_tensor(t)
                                       for t in latent_log_probs)
        self._latent_log_prob = add_n_broadcast(
            self._latent_log_probs, name='latent_log_prob')
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
        Get the log-densities of latent variables.

        Returns:
            tuple[tf.Tensor]: The log-densities of latent variables.
        """
        return self._latent_log_probs

    @property
    def latent_log_prob(self):
        """
        Get the summed log-density of latent variables.

        Returns:
            tf.Tensor: The summed log-density of latent variables.
        """
        return self._latent_log_prob

    @property
    def axis(self):
        """
        Get the axis or axes to be considered as the sampling dimensions
        of latent variables.
        """
        return self._axis

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

    @add_name_arg_doc
    def elbo(self, name=None):
        """
        Get the evidence lower-bound.

        Returns:
            tf.Tensor: The evidence lower-bound.

        See Also:
            :func:`tfsnippet.variational.elbo_objective`
        """
        return elbo_objective(
            log_joint=self._vi.log_joint,
            latent_log_prob=self._vi.latent_log_prob,
            axis=self._vi.axis,
            name=name or 'elbo'
        )

    @add_name_arg_doc
    def monte_carlo_objective(self, name=None):
        """
        Get the importance weighted lower-bound.

        Returns:
            tf.Tensor: The per-data importance weighted lower-bound.

        See Also:
            :func:`tfsnippet.variational.monte_carlo_objective`
        """
        _require_multi_samples(self._vi.axis, 'monte carlo objective')
        return monte_carlo_objective(
            log_joint=self._vi.log_joint,
            latent_log_prob=self._vi.latent_log_prob,
            axis=self._vi.axis,
            name=name or 'monte_carlo_objective'
        )

    importance_weighted_objective = monte_carlo_objective  # Legacy name


class VariationalTrainingObjectives(object):
    """Factory for variational training objectives."""

    def __init__(self, vi):
        """
        Construct a new :class:`VariationalEvaluation`.

        Args:
            vi (VariationalInference): The variational inference object.
        """
        self._vi = vi

    @add_name_arg_doc
    def sgvb(self, name=None):
        """
        Get the SGVB training objective.

        Returns:
            tf.Tensor: The per-data SGVB training objective.
                It is the negative of ELBO, which should directly be minimized.

        See Also:
            :func:`tfsnippet.variational.sgvb_estimator`
        """
        with tf.name_scope(name, default_name='sgvb'):
            return sgvb_estimator(
                # -(log p(x,z) - log q(z|x))
                values=self._vi.latent_log_prob - self._vi.log_joint,
                axis=self._vi.axis
            )

    @add_name_arg_doc
    def nvil(self, baseline=None, center_by_moving_average=True, decay=0.8,
             baseline_cost_weight=1., name=None):
        """
        Get the NVIL training objective.

        Args:
            baseline: Values of the baseline function
                math:`C_{\\psi}(\\mathbf{x})` given input `x`.
                If this is not specified, then this method will
                degenerate to the REINFORCE algorithm, with only a moving
                average estimated constant baseline `c`.
            center_by_moving_average (bool): Whether or not to use the moving
                average to maintain an estimation of `c` in above equations?
            decay: The decaying factor for moving average.
            baseline_cost_weight: Weight of the baseline cost.

        Returns:
            tf.Tensor: The per-data NVIL training objective.
        """
        with tf.name_scope(name, default_name='nvil'):
            cost, baseline_cost = nvil_estimator(
                values=self._vi.log_joint - self._vi.latent_log_prob,
                latent_log_joint=self._vi.latent_log_prob,
                axis=self._vi.axis,
                baseline=baseline,
                center_by_moving_average=center_by_moving_average,
                decay=decay
            )

            if baseline_cost is not None:
                return baseline_cost_weight * baseline_cost - cost
            else:
                return -cost

    reinforce = nvil

    @add_name_arg_doc
    def iwae(self, name=None):
        """
        Get the SGVB training objective for importance weighted objective.

        Returns:
            tf.Tensor: The per-data SGVB training objective for importance
                weighted objective.

        See Also:
            :func:`tfsnippet.variational.iwae_estimator`
        """
        _require_multi_samples(self._vi.axis, 'iwae training objective')
        with tf.name_scope(name, default_name='iwae'):
            return -iwae_estimator(
                log_values=self._vi.log_joint - self._vi.latent_log_prob,
                axis=self._vi.axis
            )

    @add_name_arg_doc
    def vimco(self, name=None):
        """
        Get the VIMCO training objective.

        Returns:
            tf.Tensor: The per-data VIMCO training objective.

        See Also:
            :meth:`tfsnippet.variational.vimco_estimator`
        """
        _require_multi_samples(self._vi.axis, 'vimco training objective')
        with tf.name_scope(name, default_name='vimco'):
            return -vimco_estimator(
                log_values=self._vi.log_joint - self._vi.latent_log_prob,
                latent_log_joint=self._vi.latent_log_prob,
                axis=self._vi.axis
            )


class VariationalEvaluation(object):
    """Factory for variational evaluation outputs."""

    def __init__(self, vi):
        """
        Construct a new :class:`VariationalEvaluation`.

        Args:
            vi (VariationalInference): The variational inference object.
        """
        self._vi = vi

    @add_name_arg_doc
    def importance_sampling_log_likelihood(self, name=None):
        """
        Compute :math:`log p(x)` by importance sampling.

        Returns:
            tf.Tensor: The per-data :math:`log p(x)`.

        See Also:
            :meth:`tfsnippet.variational.importance_sampling_log_likelihood`
        """
        _require_multi_samples(
            self._vi.axis, 'importance sampling log-likelihood')
        return importance_sampling_log_likelihood(
            log_joint=self._vi.log_joint,
            latent_log_prob=self._vi.latent_log_prob,
            axis=self._vi.axis,
            name=name or 'importance_sampling_log_likelihood'
        )

    is_loglikelihood = importance_sampling_log_likelihood
    """Short-cut for :meth:`importance_sampling_log_likelihood`."""
