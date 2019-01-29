from contextlib import contextmanager

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

from tfsnippet.ops import log_mean_exp
from tfsnippet.utils import add_name_arg_doc, get_static_shape
from .utils import _require_multi_samples

__all__ = [
    'sgvb_estimator', 'iwae_estimator', 'nvil_estimator',
]


@add_name_arg_doc
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
        axis: The sampling axes to be reduced in outputs.
            If not specified, no axis will be reduced.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the reduced axes?  (default :obj:`False`)

    Returns:
        tf.Tensor: The surrogate for optimizing the original target.
            Maximizing/minimizing this surrogate via gradient descent will
            effectively maximize/minimize the original target.
    """
    values = tf.convert_to_tensor(values)
    with tf.name_scope(name, default_name='sgvb_estimator', values=[values]):
        estimator = values
        if axis is not None:
            estimator = tf.reduce_mean(estimator, axis=axis, keepdims=keepdims)
        return estimator


@add_name_arg_doc
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
        axis: The sampling axes to be reduced in outputs.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the reduced axes?  (default :obj:`False`)

    Returns:
        tf.Tensor: The surrogate for optimizing the original target.
            Maximizing/minimizing this surrogate via gradient descent will
            effectively maximize/minimize the original target.
    """
    _require_multi_samples(axis, 'iwae estimator')
    log_values = tf.convert_to_tensor(log_values)
    with tf.name_scope(name, default_name='iwae_estimator',
                       values=[log_values]):
        estimator = log_mean_exp(log_values, axis=axis, keepdims=keepdims)
        return estimator


@add_name_arg_doc
def nvil_estimator(values, latent_log_joint, baseline=None,
                   center_by_moving_average=True, decay=0.8,
                   axis=None, keepdims=False, batch_axis=None,
                   alt_values=None, name=None):
    """
    Derive the gradient estimator for
    :math:`\\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\big[f(\\mathbf{x},\\mathbf{z})\\big]`,
    by NVIL (Mnih and Gregor, 2014) algorithm.

    .. math::

        \\begin{aligned}
        \\nabla \\, \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})} \\big[f(\\mathbf{x},\\mathbf{z})\\big]
            &= \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\Big[
                \\nabla f(\\mathbf{x},\\mathbf{z}) + f(\\mathbf{x},\\mathbf{z})\\,\\nabla\\log q(\\mathbf{z}|\\mathbf{x})\\Big] \\\\
            &= \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\Big[
                \\nabla f(\\mathbf{x},\\mathbf{z}) + \\big(f(\\mathbf{x},\\mathbf{z}) - C_{\\psi}(\\mathbf{x})-c\\big)\\,\\nabla\\log q(\\mathbf{z}|\\mathbf{x})\\Big]
        \\end{aligned}

    where :math:`C_{\\psi}(\\mathbf{x})` is a learnable network with parameter
    :math:`\\psi`, and `c` is a learnable constant.  They would be learnt by
    minimizing :math:`\\mathbb{E}_{ q(\\mathbf{z}|\\mathbf{x}) }\\Big[\\big(f(\\mathbf{x},\\mathbf{z}) - C_{\\psi}(\\mathbf{x})-c\\big)^2 \\Big]`.

    Args:
        values: Values of the target function given `z` and `x`, i.e.,
            :math:`f(\\mathbf{z},\\mathbf{x})`.
        latent_log_joint: Values of :math:`\\log q(\\mathbf{z}|\\mathbf{x})`.
        baseline: Values of the baseline function :math:`C_{\\psi}(\\mathbf{x})`
            given input `x`.  If this is not specified, then this method will
            degenerate to the REINFORCE algorithm, with only a moving
            average estimated constant baseline `c`.
        center_by_moving_average (bool): Whether or not to use the moving
            average to maintain an estimation of `c` in above equations?
        decay: The decaying factor for moving average.
        axis: The sampling axes to be reduced in outputs.
            If not specified, no axis will be reduced.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the reduced axes?  (default :obj:`False`)
        batch_axis: The batch axes to be reduced when computing
            expectation over `x`.  If not specified, all axes will be
            treated as batch axes, except the sampling axes.
        alt_values: Alternative `values` for computing the gradient of
            :math:`f(x,z)`.  This argument is originally designed for a
            slight optimization in :meth:`tfsnippet.VariationalInference.nvil`.
            USE THIS ARGUMENT ONLY IF YOU KNOW WHAT YOU ARE DOING!

    Returns:
        (tf.Tensor, tf.Tensor): The `(surrogate, baseline cost)`.

            `surrogate` is the surrogate for optimizing the original target.
            Maximizing/minimizing this surrogate via gradient descent will
            effectively maximize/minimize the original target.

            `baseline cost` is the cost to be minimized for training baseline.
            It will be :obj:`None` if `baseline` is :obj:`None`.
    """
    if baseline is None and not center_by_moving_average:
        raise ValueError('`baseline` is not specified, thus '
                         '`center_by_moving_average` must be False.')

    values = tf.convert_to_tensor(values)  # f(x,z)
    latent_log_joint = tf.convert_to_tensor(latent_log_joint)  # log q(z|x)
    if baseline is not None:
        baseline = tf.convert_to_tensor(baseline)
    if alt_values is not None:
        alt_values = tf.convert_to_tensor(alt_values)

    @contextmanager
    def mk_scope():
        if center_by_moving_average:
            with tf.variable_scope(None, default_name=name or 'nvil_estimator'):
                yield
        else:
            ns_values = [values, latent_log_joint]
            if baseline is not None:
                ns_values += [baseline]
            if alt_values is not None:
                ns_values += [alt_values]
            with tf.name_scope(name or 'nvil_estimator', values=ns_values):
                yield

    with mk_scope():
        l_signal = values
        baseline_cost = None
        if alt_values is None:
            alt_values = values

        # compute the baseline cost
        if baseline is not None:
            # baseline_cost = E[(f(x,z)-C(x)-c)^2]
            with tf.name_scope('baseline_cost'):
                baseline_cost = tf.square(
                    tf.stop_gradient(l_signal) - baseline)
                if axis is not None:
                    baseline_cost = tf.reduce_mean(
                        baseline_cost, axis, keepdims=keepdims)

            l_signal = l_signal - baseline

        # estimate `c` by moving average
        if center_by_moving_average:
            with tf.name_scope('center_by_moving_average'):
                batch_center = tf.reduce_mean(
                    l_signal, axis=batch_axis, keepdims=True)
                moving_mean_shape = get_static_shape(batch_center)
                if None in moving_mean_shape:
                    raise ValueError(
                        'The shape of `values` or `alt_values` after '
                        '`batch_axis` having been reduced must be static: '
                        'values or alt_values {}, batch_axis {}'.
                        format(values, batch_axis)
                    )
                moving_mean = tf.get_variable(
                    'moving_mean', shape=moving_mean_shape,
                    initializer=tf.constant_initializer(0.),
                    trainable=False
                )

                update_mean_op = assign_moving_average(
                    moving_mean, batch_center, decay=decay)
                l_signal = l_signal - moving_mean
                with tf.control_dependencies([update_mean_op]):
                    l_signal = tf.identity(l_signal)

        # compute the nvil cost
        with tf.name_scope('cost'):
            cost = tf.stop_gradient(l_signal) * latent_log_joint + alt_values
            if axis is not None:
                cost = tf.reduce_mean(cost, axis, keepdims=keepdims)

        # return alt_values, None
        return cost, baseline_cost
