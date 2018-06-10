from . import npyops, tfops

__all__ = ['log_sum_exp', 'log_mean_exp', 'log_softmax']


def log_sum_exp(ops, x, axis=None, keepdims=False):
    """
    Compute :math:`\\log \\sum_{k=1}^K \\exp(x_k)`.

    .. math::

        \\begin{align*}
            \\log \\sum_{k=1}^K \\exp(x_k)
                &= \\log \\left[\\exp(x_{max})
                    \\sum_{k=1}^K \\exp(x_k - x_{max})\\right] \\\\
                &= x_{max} + \\log
                    \\sum_{k=1}^K \\exp(x_k - x_{max}) \\\\
            x_{max} &= \\max x_k
        \\end{align*}

    Args:
        ops (npyops or tfops): The math operations module.
        x: The input `x`.
        axis: The dimension to sum.
            Default :obj:`None`, all dimensions.
        keepdims (bool): Whether or not to keep the summed dimensions?
            (default :obj:`False`)

    Returns:
        The computed value.
    """
    x = ops.convert_to_tensor(x)
    with ops.name_scope('log_sum_exp', values=[x]):
        x_max_keepdims = ops.reduce_max(x, axis=axis, keepdims=True)
        if not keepdims:
            x_max = ops.squeeze(x_max_keepdims, axis=axis)
        else:
            x_max = x_max_keepdims
        sum_exp = ops.reduce_sum(ops.exp(x - x_max_keepdims), axis=axis,
                                 keepdims=keepdims)
        return x_max + ops.log(sum_exp)


def log_mean_exp(ops, x, axis=None, keepdims=False):
    """
    Compute :math:`\\log \\frac{1}{K} \\sum_{k=1}^K \\exp(x_k)`.

    .. math::

        \\begin{align*}
            \\log \\frac{1}{K} \\sum_{k=1}^K \\exp(x_k)
                &= \\log \\left[\\exp(x_{max}) \\frac{1}{K}
                    \\sum_{k=1}^K \\exp(x_k - x_{max})\\right] \\\\
                &= x_{max} + \\log \\frac{1}{K}
                    \\sum_{k=1}^K \\exp(x_k - x_{max}) \\\\
            x_{max} &= \\max x_k
        \\end{align*}

    Args:
        ops (npyops or tfops): The math operations module.
        x: The input `x`.
        axis: The dimension to take average.
            Default :obj:`None`, all dimensions.
        keepdims (bool): Whether or not to keep the summed dimensions?
            (default :obj:`False`)

    Returns:
        The computed value.
    """
    x = ops.convert_to_tensor(x)
    with ops.name_scope('log_mean_exp', values=[x]):
        x = ops.convert_to_tensor(x)
        x_max_keepdims = ops.reduce_max(x, axis=axis, keepdims=True)
        if not keepdims:
            x_max = ops.squeeze(x_max_keepdims, axis=axis)
        else:
            x_max = x_max_keepdims
        mean_exp = ops.reduce_mean(ops.exp(x - x_max_keepdims), axis=axis,
                                   keepdims=keepdims)
        return x_max + ops.log(mean_exp)


def log_softmax(ops, logits):
    """
    Compute log-softmax from softmax logits :math:`\\alpha_k`.
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

