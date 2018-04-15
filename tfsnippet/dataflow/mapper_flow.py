from .data_flow import DataFlow

__all__ = ['MapperFlow']


class MapperFlow(DataFlow):
    """
    Data flow which transforms the mini-batch arrays from source flow
    by a specified mapper function.

    Usage::

        source_flow = Data.from_arrays([x, y], batch_size=256)
        mapper_flow = source_flow.map(lambda arr: (arr[0] + arr[1],))
    """

    def __init__(self, source, mapper):
        """
        Construct a :class:`MapperFlow`.

        Args:
            source (DataFlow): The source data flow.
            mapper ((tuple[np.ndarray]) -> tuple[np.ndarray])): The mapper
                function, which transforms a tuple of numpy arrays into
                another tuple of numpy arrays.
        """
        self._source = source
        self._mapper = mapper

    def _minibatch_iterator(self):
        for b in self._source:
            yield self._mapper(tuple(b))
