import tensorflow as tf

__all__ = ['create_session']


def create_session(allow_growth=True, log_device_placement=False):
    """
    Create a TensorFlow session.

    Args:
        allow_growth (bool): Allow GPU memory to grow, instead of locking
            on all the memory.  (default :obj:`True`)
        log_device_placement (bool): Whether to log the placement of graph
            nodes.   (default :obj:`False`)

    Returns:
        tf.Session: The TensorFlow session.
    """
    config = tf.ConfigProto(log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    session = tf.Session(config=config)
    return session
