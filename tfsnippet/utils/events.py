__all__ = ['EventSource']


class EventSource(object):
    """
    An object that may trigger events.

    This class is designed to either be the parent of another class,
    or be a member of another object.  For example::

        def event_handler(**kwargs):
            print('event triggered: args {}, kwargs {}'.format(args, kwargs))

        # use alone
        class SomeObject(EventSource):

            def func(self, **kwargs):
                self.fire('some_event', **kwargs)

        obj = SomeObject()
        obj.on('some_event', event_handler)

        # use as a member
        class SomeObject(object):

            def __init__(self):
                self.events = EventSource()

            def func(self, **kwargs):
                self.events.fire('some_event', **kwargs)

        obj = SomeObject()
        obj.events.on('some_event', event_handler)
    """

    def __init__(self, allowed_event_keys=None):
        """
        Construct a new :class:`EventSource`.

        Args:
            allowed_event_keys (Iterable[str]): The allowed event keys
                in :meth:`on` and :meth:`fire`.  If not specified, all
                names are allowed.
        """
        if allowed_event_keys is not None:
            allowed_event_keys = tuple(filter(str, allowed_event_keys))
        self._event_handlers_map = {}  # type: dict[str, list]
        self._allowed_event_keys = allowed_event_keys

    def on(self, event_key, handler):
        """
        Register a new event handler.

        Args:
            event_key (str): The event key.
            handler ((*args, **kwargs) -> any): The event handler.

        Raises:
            KeyError: If `event_key` is not allowed.
        """
        event_key = str(event_key)
        if self._allowed_event_keys is not None and \
                event_key not in self._allowed_event_keys:
            raise KeyError('`event_key` is not allowed: {}'.format(event_key))
        if event_key not in self._event_handlers_map:
            self._event_handlers_map[event_key] = []
        self._event_handlers_map[event_key].append(handler)

    def off(self, event_key, handler):
        """
        De-register an event handler.

        Args:
            event_key (str): The event key.
            handler ((*args, **kwargs) -> any): The event handler.

        Raises:
            ValueError: If `handler` is not a registered event handler of
                the specified event `event_key`.
        """
        event_key = str(event_key)
        try:
            self._event_handlers_map[event_key].remove(handler)
        except (KeyError, ValueError):
            raise ValueError('`handler` is not a registered event handler of '
                             'event `{}`: {}'.format(event_key, handler))

    def _fire(self, event_key, args, kwargs, reverse=False):
        event_key = str(event_key)
        if self._allowed_event_keys is not None and \
                event_key not in self._allowed_event_keys:
            raise KeyError('`event_key` is not allowed: {}'.format(event_key))
        event_handlers = self._event_handlers_map.get(event_key, None)
        if event_handlers:
            for h in (reversed(event_handlers) if reverse else event_handlers):
                h(*args, **kwargs)

    def fire(self, event_key, *args, **kwargs):
        """
        Fire an event.

        Args:
            event_key (str): The event key.
            *args: Arguments to be passed to the event handler.
            \\**kwargs: Named arguments to be passed to the event handler.

        Raises:
            KeyError: If `event_key` is not allowed.
        """
        return self._fire(event_key, args, kwargs, reverse=False)

    def reverse_fire(self, event_key, *args, **kwargs):
        """
        Fire an event, call event handlers in reversed order of registration.

        Args:
            event_key (str): The event key.
            *args: Arguments to be passed to the event handler.
            \\**kwargs: Named arguments to be passed to the event handler.

        Raises:
            KeyError: If `event_key` is not allowed.
        """
        return self._fire(event_key, args, kwargs, reverse=True)

    def clear_event_handlers(self, event_key=None):
        """
        Clear all event handlers.

        Args:
            event_key (str or None): If specified, clear all event handlers
                of this name.  Otherwise clear all event handlers.
        """
        if event_key is not None:
            event_key = str(event_key)
            self._event_handlers_map.pop(event_key, None)
        else:
            self._event_handlers_map.clear()
