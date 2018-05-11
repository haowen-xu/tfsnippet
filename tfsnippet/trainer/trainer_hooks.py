__all__ = [
    'HookPriority', 'HookEntry', 'HookList', 'TrainerHooks',
]


class HookPriority(object):
    """
    Hook priorities defined for :class:`~tfsnippet.trainer.BaseTrainer`.
    Smaller values take higher priorities.
    """

    VALIDATION = 50
    DEFAULT = 100
    ANNEALING = 200
    LOGGING = 999


class HookEntry(object):
    """Configurations of a hook entry in :class:`HookList`."""

    def __init__(self, callback, freq, priority, birth):
        """
        Construct a new :class:`HookEntry`.

        Args:
            callback (() -> any): The callable object, as the hook callback.
            freq (int): The frequency for this callback to be called.
            priority (int): The hook priority number.
            birth (int): The counter of birth, as an additional key for
                sorting the hook entries, such that old hooks will be
                placed in front of newly added hooks, if they have the
                same priority.
        """
        self.callback = callback
        self.freq = freq
        self.priority = priority
        self.counter = freq
        self.birth = birth

    def reset_counter(self):
        """Reset the `counter` to `freq`, its initial value."""
        self.counter = self.freq

    def maybe_call(self):
        """
        Decrease the `counter`, and call the `callback` if `counter` is less
        than 1.  The counter will be reset to `freq` after then.
        """
        self.counter -= 1
        if self.counter < 1:
            # put this statement before calling the callback, such that
            # the remaining counter would be correctly updated even if
            # any error occurs
            self.counter = self.freq
            self.callback()

    def sort_key(self):
        """Get the key for sorting this hook entry."""
        return self.priority, self.birth


class HookList(object):
    """Epoch, step and validation hooks list."""

    def __init__(self):
        self._callbacks = []  # type: list[HookEntry]
        self._birth_counter = 0  # to enforce stable ordering

    def add(self, callback, freq=1, priority=HookPriority.DEFAULT):
        freq = int(freq)
        if freq < 1:
            raise ValueError('`freq` must be at least 1.')
        self._birth_counter += 1
        self._callbacks.append(HookEntry(
            callback=callback, freq=freq, priority=priority,
            birth=self._birth_counter
        ))
        self._callbacks.sort(key=lambda e: e.sort_key())

    def call(self):
        for e in self._callbacks:
            e.maybe_call()

    def reset_counter(self):
        for e in self._callbacks:
            e.reset_counter()

    def remove_if(self, condition):
        self._callbacks = [
            e for e in self._callbacks
            if not condition(e.callback, e.freq, e.priority)
        ]

    def remove_all(self, callback):
        return self.remove_if(lambda c, f, p: c == callback)

    def __repr__(self):
        return 'Callbacks[{}]'.format(
            '; '.join(
                '{!r},freq={}'.format(e.callback, e.freq)
                for e in self._callbacks
            )
        )


class TrainerHooks(object):

    def __init__(self):
        self._before_list = HookList()
        self._after_list = HookList()

    @property
    def before_list(self):
        return self._before_list

    @property
    def after_list(self):
        return self._after_list

    def call_before(self):
        self.before_list.call()

    def call_after(self):
        self.after_list.call()

    def add_before(self, callback, freq=1, priority=HookPriority.DEFAULT):
        self.before_list.add(callback, freq=freq, priority=priority)

    def add_after(self, callback, freq=1, priority=HookPriority.DEFAULT):
        self.after_list.add(callback, freq=freq, priority=priority)

    def reset_counter(self):
        self.before_list.reset_counter()
        self.after_list.reset_counter()

    def remove_if(self, condition):
        self.before_list.remove_if(condition)
        self.after_list.remove_if(condition)

    def remove_all(self, callback):
        self.before_list.remove_all(callback)
        self.after_list.remove_all(callback)
