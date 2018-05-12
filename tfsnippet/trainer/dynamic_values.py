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

    def __init__(self, initial_value, ratio):
        """
        Construct a new :class:`AnnealingDynamicValue`.

        Args:
            initial_value: A number, the initial value.
            ratio: A number, the ratio of annealing at each time.
        """
        super(AnnealingDynamicValue, self).__init__(initial_value)
        self._ratio = None
        self.ratio = ratio

    @property
    def ratio(self):
        """Get the ratio of annealing at each time."""
        return self._ratio

    @ratio.setter
    def ratio(self, ratio):
        self._ratio = ratio

    def anneal(self):
        """Anneal the value."""
        self._value *= self._ratio
