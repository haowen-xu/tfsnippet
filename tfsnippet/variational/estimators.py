import tensorflow as tf

from tfsnippet.nn import tfops, log_mean_exp
from .utils import _require_multi_samples

__all__ = [
    'sgvb_estimator', 'iwae_estimator', 'nvil_estimator', 'vimco_estimator'
]


def sgvb_estimator(values, axis=None, keepdims=False, name=None):
    """
    Derive the gradient estimator for
    :math:`\\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\big[f(\\mathbf{x},\\mathbf{z})\\big]`,
    by SGVB (Kingma, D.P. and Welling, M., 2013) algorithm.

    .. math::

        \\nabla \\, \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\big[f(\\mathbf{x},\\mathbf{z})\\big] = \\nabla \\, \\mathbb{E}_{q(\\mathbf{\\epsilon})}\\big[f(\\mathbf{x},\\mathbf{z}(\\mathbf{\\epsilon}))\\big] = \\mathbb{E}_{q(\\mathbf{\\epsilon})}\\big[\\nabla f(\\mathbf{x},\\mathbf{z}(\\mathbf{\\epsilon}))\\big]

    Args:
        values: Values of the target function given `z` and `x`, i.e.,
            :math:`f(\\mathbf{z},\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
            If :obj:`None`, no dimensions will be averaged out.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the averaged dimensions?  (default :obj:`False`)
        name (str): Name of this operation in TensorFlow graph.
            (default "sgvb_estimator")

    Returns:
        tf.Tensor: The surrogate for optimizing the target function
            with SGVB gradient estimator.
    """
    values = tf.convert_to_tensor(values)
    with tf.name_scope(name, default_name='sgvb_estimator', values=[values]):
        estimator = values
        if axis is not None:
            estimator = tf.reduce_mean(estimator, axis=axis, keepdims=keepdims)
        return estimator


def iwae_estimator(log_values, axis, keepdims=False, name=None):
    """
    Derive the gradient estimator for
    :math:`\\mathbb{E}_{q(\\mathbf{z}^{(1:K)}|\\mathbf{x})}\\Big[\\log \\frac{1}{K} \\sum_{k=1}^K f\\big(\\mathbf{x},\\mathbf{z}^{(k)}\\big)\\Big]`,
    by IWAE (Burda, Y., Grosse, R. and Salakhutdinov, R., 2015) algorithm.

    .. math::

        \\begin{aligned}
            &\\nabla\\,\\mathbb{E}_{q(\\mathbf{z}^{(1:K)}|\\mathbf{x})}\\Big[\\log \\frac{1}{K} \\sum_{k=1}^K f\\big(\\mathbf{x},\\mathbf{z}^{(k)}\\big)\\Big]
                = \\nabla \\, \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\log \\frac{1}{K} \\sum_{k=1}^K w_k\\Bigg]
                = \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\nabla \\log \\frac{1}{K} \\sum_{k=1}^K w_k\\Bigg] = \\\\
                & \\quad \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\frac{\\nabla \\frac{1}{K} \\sum_{k=1}^K w_k}{\\frac{1}{K} \\sum_{i=1}^K w_i}\\Bigg]
                = \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\frac{\\sum_{k=1}^K w_k \\nabla \\log w_k}{\\sum_{i=1}^K w_i}\\Bigg]
                = \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\sum_{k=1}^K \\widetilde{w}_k \\nabla \\log w_k\\Bigg]
        \\end{aligned}

    Args:
        log_values: Log values of the target function given `z` and `x`, i.e.,
            :math:`\\log f(\\mathbf{z},\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the averaged dimensions?  (default :obj:`False`)
        name (str): Name of this operation in TensorFlow graph.
            (default "iwae_estimator")

    Returns:
        tf.Tensor: The surrogate for optimizing the target function
            with IWAE gradient estimator.
    """
    _require_multi_samples(axis, 'iwae estimator')
    log_values = tf.convert_to_tensor(log_values)
    with tf.name_scope(name, default_name='iwae_estimator',
                       values=[log_values]):
        estimator = log_mean_exp(
            tfops, log_values, axis=axis, keepdims=keepdims)
        return estimator


def nvil_estimator(values, baseline=None, variance_reduction=True, decay=.8,
                   axis=None, keepdims=False, name=None):
    """
    Derive the gradient estimator for
    :math:`\\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\big[f(\\mathbf{x},\\mathbf{z})\\big]`,
    by NVIL (Mnih, A. and Gregor, K., 2014) algorithm.

    The gradient estimator for optimizing :math:`\\phi` and :math:`\\theta` is:

    .. math::

        \\begin{aligned}
        \\nabla \\, \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})} \\big[f(\\mathbf{x},\\mathbf{z})\\big]
            &= \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\Big[
                \\nabla f(\\mathbf{x},\\mathbf{z}) + f(\\mathbf{x},\\mathbf{z})\\,\\nabla\\log q(\\mathbf{z}|\\mathbf{x})\\Big] \\\\
            &= \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\Big[
                \\nabla f(\\mathbf{x},\\mathbf{z}) + \\big(f(\\mathbf{x},\\mathbf{z}) - C_{\\psi}(\\mathbf{x})-c\\big)\\,\\nabla\\log q(\\mathbf{z}|\\mathbf{x})\\Big]
        \\end{aligned}

    The gradient estimator for optimizing :math:`\\psi`, the parameter of the
    baseline network is:

    .. math::

        \\nabla_{\\psi} \\, \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\Big[\\big(f(\\mathbf{x},\\mathbf{z})-C_{\\psi}(\\mathbf{x})-c\\big)^2\\Big]
                = \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\Big[-2\\, \\big(f(\\mathbf{x},\\mathbf{z})-C_{\\psi}(\\mathbf{x})-c\\big) \\, \\nabla_{\\psi} \\, C_{\\psi}(\\mathbf{x})\\Big]

    Args:
        values: Values of the target function given `z` and `x`, i.e.,
            :math:`f(\\mathbf{z},\\mathbf{x})`.
        baseline: Values of the baseline function given `x`, i.e.,
            :math:`C_{\\psi}(\\mathbf{x})-c\\big)`.
        variance_reduction (bool): Whether to use
            :math:`C_{\\psi}(\\mathbf{x})-c\\big)` (if specified) and a
            moving average estimation of :math:`c` to reduce variance?
            (default :obj:`True`)
        decay (float): The moving average decay for variance normalization.
            (default 0.8)
        axis: The sampling dimensions to be averaged out.
            If :obj:`None`, no dimensions will be averaged out.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the averaged dimensions?  (default :obj:`False`)
        name (str): Name of this operation in TensorFlow graph.
            (default "sgvb_estimator")

    Returns:
        tf.Tensor or (tf.Tensor, tf.Tensor):
            If `baseline` is not given, return a Tensor, which is the surrogate
            for optimizing :math:`\\phi` and :math:`\\theta`.

            Otherwise if `baseline` is given, return a tuple of Tensor, where
            the first term is the surrogate for optimizing :math:`\\phi` and
            :math:`\\theta`, and the second term is the surrogate for optimizing
            :math:`\\psi`.  One may get the final training objective by gluing
            them together with a proper scaling factor on the second term.

    Notes:
        All the reparameteriable latent variables must be constructed/added
        with `is_reparameterized` set to :obj:`False`.
    """
    if not variance_reduction and baseline is not None:
        raise ValueError('`variance_reduction` must be True when `baseline` '
                         'is specified.')

    values = tf.convert_to_tensor(values)
    ns_args = [values]
    if baseline is not None:
        baseline = tf.convert_to_tensor(baseline)
        ns_args.append(baseline)

    with tf.name_scope(name, default_name='nvil_estimator', values=ns_args):
        raise NotImplementedError()


def vimco_estimator(log_values, axis, keepdims=False, name=None):
    """
    Derive the gradient estimator for
    :math:`\\mathbb{E}_{q(\\mathbf{z}^{(1:K)}|\\mathbf{x})}\\Big[\\log \\frac{1}{K} \\sum_{k=1}^K f\\big(\\mathbf{x},\\mathbf{z}^{(k)}\\big)\\Big]`,
    by IWAE (Burda, Y., Grosse, R. and Salakhutdinov, R., 2015) algorithm.

    .. math::

        \\begin{aligned}
        &\\nabla\\,\\mathbb{E}_{q(\\mathbf{z}^{(1:K)}|\\mathbf{x})}\\Big[\\log \\frac{1}{K} \\sum_{k=1}^K f\\big(\\mathbf{x},\\mathbf{z}^{(k)}\\big)\\Big] \\\\
                &\\quad =  \\mathbb{E}_{q(\\mathbf{z}^{(1:K)}|\\mathbf{x})}\\bigg[{\\sum_{k=1}^K \\hat{L}(\\mathbf{z}^{(k)}|\\mathbf{z}^{(-k)}) \\, \\nabla \\log q(\\mathbf{z}^{(k)}|\\mathbf{x})}\\bigg] +
                 \\mathbb{E}_{q(\\mathbf{z}^{(1:K)}|\\mathbf{x})}\\bigg[{\\sum_{k=1}^K \\widetilde{w}_k\\,\\nabla\\log f(\\mathbf{x},\\mathbf{z}^{(k)})}\\bigg]
        \\end{aligned}

    where :math:`w_k = f\\big(\\mathbf{x},\\mathbf{z}^{(k)}\\big)`,
    :math:`\\widetilde{w}_k = w_k / \\sum_{i=1}^K w_i`, and:

    .. math::

        \\begin{aligned}
            \\hat{L}(\\mathbf{z}^{(k)}|\\mathbf{z}^{(-k)})
                &= \\hat{L}(\\mathbf{z}^{(1:K)}) - \\log \\frac{1}{K} \\bigg(\\hat{f}(\\mathbf{x},\\mathbf{z}^{(-k)})+\\sum_{i \\neq k} f(\\mathbf{x},\\mathbf{z}^{(i)})\\bigg) \\\\
            \\hat{L}(\\mathbf{z}^{(1:K)}) &= \\log \\frac{1}{K} \\sum_{k=1}^K f(\\mathbf{x},\\mathbf{z}^{(k)}) \\\\
                \\hat{f}(\\mathbf{x},\\mathbf{z}^{(-k)}) &= \\exp\\big(\\frac{1}{K-1} \\sum_{i \\neq k} \\log f(\\mathbf{x},\\mathbf{z}^{(i)})\\big)
        \\end{aligned}

    Args:
        log_values: Log values of the target function given `z` and `x`, i.e.,
            :math:`\\log f(\\mathbf{z},\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the averaged dimensions?  (default :obj:`False`)
        name (str): Name of this operation in TensorFlow graph.
            (default "iwae_estimator")

    Returns:
        tf.Tensor: The surrogate for optimizing the target function
            with VIMCO gradient estimator.

    Notes:
        All the reparameteriable latent variables must be constructed/added
        with `is_reparameterized` set to :obj:`False`.
    """
    _require_multi_samples(axis, 'vimco estimator')
    log_values = tf.convert_to_tensor(log_values)
    with tf.name_scope(name, default_name='vimco_estimator',
                       values=[log_values]):
        raise NotImplementedError()
