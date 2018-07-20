from tfsnippet.utils import DocInherit

__all__ = [
    'DynamicValue', 'SimpleDynamicValue', 'AnnealingDynamicValue',
]


@DocInherit
class DynamicValue(object):
    """
    Dynamic values fed into trainers.

    It is sometimes necessary to feed a dynamic value into a trainer,
    e.g., an annealing learning rate.  This class provides such a base
    class for all dynamic values.
    """

    def get(self):
        """Get the current value of this :class:`DynamicValue` object."""
        raise NotImplementedError()


class SimpleDynamicValue(DynamicValue):
    """
    A simple :class:`DynamicValue`, which stores the value in its internal
    attribute, and can be changed by :meth:`set`.
    """

    def __init__(self, value):
        """
        Construct a new :class:`SimpleDynamicValue`.

        Args:
            value: Any value to be set.  It can even be another instance
                of :class:`DynamicValue`.
        """
        self._value = None
        self.set(value)

    def get(self):
        if isinstance(self._value, DynamicValue):
            return self._value.get()
        else:
            return self._value

    def set(self, value):
        """
        Set the value of this :class:`SimpleDynamicValue` instance.

        Args:
            value: Any value to be set.  It can even be another instance
                of :class:`DynamicValue`.
        """
        if value is self:
            raise ValueError('Cannot set the value to `self`.')
        self._value = value


class AnnealingDynamicValue(SimpleDynamicValue):
    """
    A :class:`DynamicValue` whose value is annealed (scaled) each time
    :meth:`anneal` is called.
    """

    def __init__(self, initial_value, ratio, min_value=None):
        """
        Construct a new :class:`AnnealingDynamicValue`.

        Args:
            initial_value: A number, the initial value.
            ratio: A number, the ratio of annealing at each time.
            min_value: Optional, a number, the minimum value.
        """
        if min_value is not None:
            initial_value = max(initial_value, min_value)
        super(AnnealingDynamicValue, self).__init__(initial_value)
        self.min_value = min_value
        self.ratio = ratio

    def anneal(self):
        """Anneal the value."""
        if self.min_value is not None:
            self._value = max(self.min_value, self._value * self.ratio)
        else:
            self._value *= self.ratio
