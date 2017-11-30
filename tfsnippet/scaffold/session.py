import tensorflow as tf

__all__ = ['get_default_session_or_error']


def get_default_session_or_error():
    """
    Get the default session.

    Raises:
        RuntimeError: If there's no active session.
    """
    ret = tf.get_default_session()
    if ret is None:
        raise RuntimeError('No session is active')
    return ret
