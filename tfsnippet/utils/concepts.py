__all__ = [
    'LazyInit',
    'LazyInitAndDestroyable',
    'Disposable',
    'NoReentrantContext',
    'DisposableContext',
]


class LazyInit(object):
    """
    Classes with lazy initialized internal states.
    """

    _initialized = False

    def _init(self):
        """Override this method to initialize the internal states."""
        raise NotImplementedError()

    def ensure_init(self):
        """Ensure the internal states are initialized."""
        if not self._initialized:
            self._init()
            self._initialized = True


class LazyInitAndDestroyable(LazyInit):
    """
    Classes with lazy initialized internal states, which are also destroyable
    by calling :meth:`destroy()`.  A destroyed object may be initialized again,
    depending on the implementation of specific class.
    """

    def _destroy(self):
        """Override this method to destroy the internal states."""
        raise NotImplementedError()

    def destroy(self):
        """Destroy the internal states."""
        if self._initialized:
            try:
                self._destroy()
            finally:
                self._initialized = False


class Disposable(object):
    """
    Classes which can only be used once.
    """

    _already_used = False

    def _check_usage_and_set_used(self):
        """
        Check whether the usage flag, ensure the object has not been used,
        and then set it to be used.
        """
        if self._already_used:
            raise RuntimeError('Disposable object cannot be used twice: {!r}.'.
                               format(self))
        self._already_used = True


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
            raise RuntimeError('Context is required be entered: {!r}.'.
                               format(self))

    def __enter__(self):
        if self._is_entered:
            raise RuntimeError('Context is not reentrant: {!r}.'.
                               format(self))
        ret = self._enter()
        self._is_entered = True
        return ret

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_entered:
            self._is_entered = False
            return self._exit(exc_type, exc_val, exc_tb)


class DisposableContext(NoReentrantContext):
    """
    Base class for contexts which can only be entered once.
    """

    _has_entered = False

    def __enter__(self):
        if self._has_entered:
            raise RuntimeError(
                'A disposable context cannot be entered twice: {!r}.'.
                format(self))
        ret = super(DisposableContext, self).__enter__()
        self._has_entered = True
        return ret
