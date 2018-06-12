from . import tfops, npyops
from .log_exp import log_mean_exp
from .softmax import log_softmax

__all__ = ['inception_score']


def inception_score(ops, logits=None, probs=None, reduce_dims=None,
                    clip_eps=1e-7):
    """
    Compute the Inception score ("Improved techniques for training gans",
    Salimans, T. et al. 2016.) from given softmax `logits` or `probs`.

    .. math::

        \\begin{align*}
            \\text{Inception score} &= \\exp\\left\\{
                \\operatorname{\\mathbb{E}}_{x}\\Big[
                    \\operatorname{\\text{D}}_{KL}\\left(p(y|x)
                        \\,\\big\\|\\,p(y)\\Big)\\right]
                \\right\\} \\\\
            p(y) &= \\operatorname{\\mathbb{E}}_{x}\\left[p(y|x)\\right]
        \\end{align*}

    Args:
        ops (tfops or npyops): The math operations module.
        logits: The softmax logits for :math:`p(y|x)`.
            The last dimension will be treated as the softmax dimension.
        probs: The softmax probs for :math:`p(y|x)`.
            The last dimension will be treated as the softmax dimension.
            Ignored if `logits` is specified.
        reduce_dims: If specified, only these dimension will be treated as
            the data dimensions, and reduced for computing the Inception score.
            If not specified, all dimensions except the softmax dimension will
            be treated as the data dimensions.  (default :obj:`None`)
        clip_eps: The epsilon value for clipping `probs`, in order to avoid
            numerical issues. (default ``1e-7``)

    Returns:
        The computed Inception score, with data dimension reduced.
    """
    if logits is None and probs is None:
        raise ValueError('At least one of `logits` and `probs` should be '
                         'specified.')

    def make_inception_score():
        kld = ops.reduce_sum(
            p_y_given_x * (log_p_y_given_x - log_p_y),
            axis=-1,
            keepdims=True  # We do not squeeze the dimension here, in case
                           # `reduce_dims` uses negative indices
        )
        score = ops.squeeze(
            ops.reduce_mean(kld, axis=reduce_dims),  # Expectation over data
            axis=-1  # Squeeze the softmax dimension here
        )
        return ops.exp(score)

    def log_probs(probs, lbound=clip_eps, ubound=1. - clip_eps):
        return ops.log(ops.clip_by_value(probs, lbound, ubound))

    if logits is not None:
        logits = ops.convert_to_tensor(logits)
        with ops.assert_rank_at_least(logits, 1):
            with ops.name_scope('inception_score', values=[logits]):
                log_p_y_given_x = log_softmax(ops, logits)
                if reduce_dims is None:
                    reduce_dims = ops.ensure_axis_arg(
                        ops.range(ops.rank(logits) - 1, dtype=ops.int32))
                log_p_y = log_mean_exp(
                    ops, log_p_y_given_x, axis=reduce_dims, keepdims=True)
                p_y_given_x = ops.exp(log_p_y_given_x)
                return make_inception_score()

    else:
        probs = ops.convert_to_tensor(probs)
        with ops.assert_rank_at_least(probs, 1):
            with ops.name_scope('inception_score', values=[probs]):
                log_p_y_given_x = log_probs(probs)
                if reduce_dims is None:
                    reduce_dims = ops.ensure_axis_arg(
                        ops.range(ops.rank(probs) - 1, dtype=ops.int32))
                log_p_y = log_probs(
                    ops.reduce_mean(probs, axis=reduce_dims, keepdims=True))
                p_y_given_x = probs
                return make_inception_score()
