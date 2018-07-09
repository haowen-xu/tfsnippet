import tensorflow as tf

__all__ = ['create_session']


def create_session(lock_memory=None,
                   log_device_placement=False,
                   allow_soft_placement=True):
    """
    Create a TensorFlow session.

    Args:
        lock_memory (None or False or float):
            * If :obj:`False`, set `allow_growth` to True.
            * If :obj:`None`, lock all free memory.
            * If float, lock this portion of memory.
            (default :obj:`None`)
        log_device_placement (bool): Whether to log the placement of graph
            nodes.   (default :obj:`False`)
        allow_soft_placement (bool): Whether or not to allow soft placement?
            (default :obj:`True`)

    Returns:
        tf.Session: The TensorFlow session.
    """
    config = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement)
    if lock_memory is False:
        config.gpu_options.allow_growth = True
    elif isinstance(lock_memory, float):
        config.gpu_options.per_process_gpu_memory_fraction = lock_memory
    elif lock_memory is not None:
        raise TypeError('`lock_memory` must be None, False or float.')
    session = tf.Session(config=config)
    return session
