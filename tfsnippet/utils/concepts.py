__all__ = [
    'NoReentrantContext',
    'OneTimeContext',
]


class NoReentrantContext(object):
    """
    Base class for contexts which are not reentrant (i.e., if there is
    a context opened by ``__enter__``, and it has not called ``__exit__``,
    the ``__enter__`` cannot be called again).
    """

    _is_entered = False

    def _enter(self):
        """
        Enter the context.  Subclasses should override this instead of
        the true ``__enter__`` method.
        """
        raise NotImplementedError()

    def _exit(self, exc_type, exc_val, exc_tb):
        """
        Exit the context.  Subclasses should override this instead of
        the true ``__exit__`` method.
        """
        raise NotImplementedError()

    def _require_entered(self):
        """
        Require the context to be entered.

        Raises:
            RuntimeError: If the context is not entered.
        """
        if not self._is_entered:
            raise RuntimeError(
                '{} is not currently entered.'.format(self.__class__.__name__))

    def __enter__(self):
        if self._is_entered:
            raise RuntimeError(
                '{} is not reentrant.'.format(self.__class__.__name__))
        ret = self._enter()
        self._is_entered = True
        return ret

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_entered:
            self._is_entered = False
            return self._exit(exc_type, exc_val, exc_tb)


class OneTimeContext(NoReentrantContext):
    """
    Base class for contexts which can only be entered once.
    """

    _has_entered = False

    def __enter__(self):
        if self._has_entered:
            raise RuntimeError(
                'The one-time context {} has already been entered, thus cannot '
                'be entered again.'.format(self.__class__.__name__))
        ret = super(OneTimeContext, self).__enter__()
        self._has_entered = True
        return ret
