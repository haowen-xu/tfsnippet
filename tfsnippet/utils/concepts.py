__all__ = ['OpenCloseContext']


class OpenCloseContext(object):
    """
    Abstract context object with ``open`` and ``close`` methods.

    This class provides a base for all context objects exposing a pair of
    ``open`` and ``close`` methods.  These methods can only be called once
    each.  Also, entering an instance of the context object will automatically
    call the ``open`` method, while exiting it will call ``close``.
    """

    _has_opened = False  # indicating whether the class has been opened
    _has_closed = False  # indicating whether the class has been closed

    def _open(self):
        raise NotImplementedError()

    def _close(self, exc_info):
        raise NotImplementedError()

    def _require_alive(self):
        """Require the context object to be opened, and not closed."""
        if not self._has_opened:
            raise RuntimeError(
                'The {} has not been opened'.format(self.__class__.__name__))
        if self._has_closed:
            raise RuntimeError(
                'The {} has been closed'.format(self.__class__.__name__))

    def open(self):
        """
        Open the context object.
        """
        if self._has_opened:
            raise RuntimeError(
                'The {} has been opened'.format(self.__class__.__name__))

        self._open()
        self._has_opened = True

    def close(self, exc_info=None):
        """
        Close the context object.

        Args:
            exc_info: The tuple of (exc_type, exc_val, exc_tb), if available.
        """
        if self._has_opened and not self._has_closed:
            self._close(exc_info)
            self._has_closed = True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None or exc_val is not None or exc_tb is not None:
            exc_info = (exc_type, exc_val, exc_tb)
        else:
            exc_info = None
        self.close(exc_info)
