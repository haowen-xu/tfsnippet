from . import npyops, tfops
from .log_exp import log_sum_exp

__all__ = ['softmax', 'log_softmax']


def softmax(ops, logits):
    """
    Compute softmax from logits :math:`\\alpha_k`.
    Note the corresponding multinomial distribution is defined as
    :math:`\\pi_k = \\exp(\\alpha_k) / \\sum_{k=1}^K \\exp(\\alpha_i)`.

    .. math::

        \\mathop{\\text{softmax}}(x) =
            \\frac{\\exp \\alpha_k}
                  {\\sum_{k=1}^K \\exp\\left(\\alpha_i\\right)} =
            \\frac{\\exp \\left(\\alpha_k - \\alpha_{max}\\right)}
                  {\\sum_{k=1}^K \\exp\\left(\\alpha_i - \\alpha_{max}\\right)}

    Args:
        ops (npyops or tfops): The math operations module.
        logits: The un-normalized logits :math:`\\alpha_k` of :math:`p(x)`.
            The last dimension will be treated as the softmax dimension.

    Returns:
        The softmax outputs.
    """
    logits = ops.convert_to_tensor(logits)
    with ops.name_scope('softmax', values=[logits]):
        logits_max = ops.reduce_max(logits, axis=-1, keepdims=True)
        logits_exp = ops.exp(logits - logits_max)
        return logits_exp / ops.reduce_sum(logits_exp, axis=-1, keepdims=True)


def log_softmax(ops, logits):
    """
    Compute log-softmax from logits :math:`\\alpha_k`.
    Note the corresponding multinomial distribution is defined as
    :math:`\\pi_k = \\exp(\\alpha_k) / \\sum_{k=1}^K \\exp(\\alpha_i)`.

    .. math::

        \\log\\mathop{\\text{softmax}}(x) =
            \\log \\frac{\\exp \\alpha_k}{\\sum_{k=1}^K \\exp(\\alpha_i)} =
            \\alpha_k - \\log \\sum_{k=1}^K \\exp(\\alpha_i)

    Args:
        ops (npyops or tfops): The math operations module.
        logits: The un-normalized logits :math:`\\alpha_k` of :math:`p(x)`.
            The last dimension will be treated as the softmax dimension.

    Returns:
        The log-softmax outputs.
    """
    logits = ops.convert_to_tensor(logits)
    with ops.name_scope('log_softmax', values=[logits]):
        return logits - log_sum_exp(ops, logits, axis=-1, keepdims=True)
