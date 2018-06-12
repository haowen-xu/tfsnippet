from .softmax import log_softmax, softmax

__all__ = ['softmax_logits_kld', 'softmax_probs_kld']


def softmax_logits_kld(ops, p_logits, q_logits, keepdims=False):
    """
    Compute the KL-divergence between two softmax categorical distributions
    via logits.  The last dimension of `p` and `q` are treated as the
    softmax dimension, and will be reduced for computing KL-divergence.

    .. math::

        \\operatorname{D}_{KL}(p(y)\\|q(y)) =
            \\sum_y p(y) \\left(\\log p(y) - \\log q(y)\\right)

    Args:
        ops (npyops or tfops): The math operations module.
        p_logits: Logits of softmax categorical :math:`p(y)`.
        q_logits: Logits of softmax categorical :math:`q(y)`.
        keepdims (bool): Whether or not to keep the reduced dimension?
            (default :obj:`False`)

    Returns:
        The computed softmax categorical distributions KL-divergence.
    """
    p_logits = ops.convert_to_tensor(p_logits)
    q_logits = ops.convert_to_tensor(q_logits)
    with ops.name_scope('softmax_logits_kld', values=[p_logits, q_logits]):
        log_p = log_softmax(ops, p_logits)
        log_q = log_softmax(ops, q_logits)
        p = softmax(ops, p_logits)
        # TODO: can we reduce time consumption by ``np.exp(log_p)``?
        # p = ops.exp(log_p)
        return ops.reduce_sum(p * (log_p - log_q), axis=-1, keepdims=keepdims)


def softmax_probs_kld(ops, p_probs, q_probs, keepdims=False, clip_eps=1e-7):
    """
    Compute the KL-divergence between two softmax categorical distributions
    via probs.  The last dimension of `p` and `q` are treated as the
    softmax dimension, and will be reduced for computing KL-divergence.

    .. math::

        \\operatorname{D}_{KL}(p(y)\\|q(y)) =
            \\sum_y p(y) \\left(\\log p(y) - \\log q(y)\\right)

    Args:
        ops (npyops or tfops): The math operations module.
        p_probs: Probabilities of softmax categorical :math:`p(y)`.
        q_probs: Probabilities of softmax categorical :math:`q(y)`.
        keepdims (bool): Whether or not to keep the reduced dimension?
            (default :obj:`False`)
        clip_eps: The epsilon value for clipping `p_probs` and `q_probs`,
            in order to avoid numerical issues. (default ``1e-7``)

    Returns:
        The computed softmax categorical distributions KL-divergence.
    """
    p_probs = ops.convert_to_tensor(p_probs)
    q_probs = ops.convert_to_tensor(q_probs)
    with ops.name_scope('softmax_probs_kld', values=[p_probs, q_probs]):
        # clip the probabilities to avoid nans
        log_p = ops.log(ops.clip_by_value(p_probs, clip_eps, 1. - clip_eps))
        log_q = ops.log(ops.clip_by_value(q_probs, clip_eps, 1. - clip_eps))
        return ops.reduce_sum(p_probs * (log_p - log_q), axis=-1,
                              keepdims=keepdims)
