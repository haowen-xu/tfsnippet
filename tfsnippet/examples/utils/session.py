import tensorflow as tf

__all__ = ['create_session']


def create_session(lock_memory=None,
                   log_device_placement=False,
                   allow_soft_placement=True):
    """
    Create a TensorFlow session.

    Args:
        lock_memory (None or float): If not specified, set ``allow_growth``
            flag to :obj:`True`.  Otherwise set the memory portion to this.
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
    if lock_memory is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = lock_memory
    session = tf.Session(config=config)
    return session
