__all__ = ['detect_devices']


def detect_devices():
    """

    Returns:

    """
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
