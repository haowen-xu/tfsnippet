from .base import DataFlow

__all__ = ['MapperFlow']


class MapperFlow(DataFlow):
    """
    Data flow which transforms the mini-batch arrays from source flow
    by a specified mapper function.

    Usage::

        source_flow = Data.arrays([x, y], batch_size=256)
        mapper_flow = source_flow.map(lambda x, y: (x + y,))
    """

    def __init__(self, source, mapper):
        """
        Construct a :class:`MapperFlow`.

        Args:
            source (DataFlow): The source data flow.
            mapper ((\*np.ndarray) -> tuple[np.ndarray])): The mapper
                function, which transforms numpy arrays into a tuple
                of other numpy arrays.
        """
        self._source = source
        self._mapper = mapper

    @property
    def source(self):
        """Get the source data flow."""
        return self._source

    def _minibatch_iterator(self):
        for b in self._source:
            mapped_b = self._mapper(*b)
            if isinstance(mapped_b, list):
                mapped_b = tuple(mapped_b)
            elif not isinstance(mapped_b, tuple):
                raise TypeError('The output of the ``mapper`` is expected to '
                                'be a tuple or a list, but got a {}.'.
                                format(mapped_b.__class__.__name__))
            yield mapped_b
