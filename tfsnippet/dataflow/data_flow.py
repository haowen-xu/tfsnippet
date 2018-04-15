__all__ = ['DataFlow']


class DataFlow(object):
    """
    Data flows are objects for constructing mini-batch iterators.

    There are two major types of :class:`DataFlow` classes: data sources
    and data transformers.  Data sources, like the :class:`ArrayFlow`,
    produce mini-batches from underlying data sources.  Data transformers,
    like :class:`MapperFlow`, produce mini-batches by transforming arrays
    from the source.

    All :class:`DataFlow` subclasses shipped by :mod:`tfsnippet.dataflow`
    can be constructed by factory methods of this base class.  For example::

        # :class:`ArrayFlow` from arrays
        DataFlow.from_array([x, y], batch_size=256, shuffle=True)
    """

    _is_iter_entered = False

    def _minibatch_iterator(self):
        """
        Get the mini-batch iterator.  Subclasses should override this to
        implement the data flow.

        Yields:
            tuple[np.ndarray]: Mini-batches of tuples of numpy arrays.
                The arrays might be read-only.
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        Iterate through the mini-batches.  Not reentrant.

        Some subclasses may also inherit from :class:`NoReentrantContext`,
        thus a context must be firstly entered before using such data flows
        as iterators, for example::

            with DataFlow.from_database(...) as df:
                for epoch in epochs:
                    for batch_x, batch_y in df:
                        ...

        Yields:
            tuple[np.ndarray]: Mini-batches of tuples of numpy arrays.
                The arrays might be read-only.
        """
        if self._is_iter_entered:
            raise RuntimeError('{}.__iter__ is not reentrant.'.
                               format(self.__class__.__name__))
        self._is_iter_entered = True
        try:
            for b in self._minibatch_iterator():
                yield b
        finally:
            self._is_iter_entered = False

    # -------- here starts the transforming methods --------
    def map(self, mapper):
        """
        Construct a :class:`MapperFlow`, from this flow and the ``mapper``.

        Args:
            mapper ((tuple[np.ndarray]) -> tuple[np.ndarray])): The mapper
                function, which transforms a tuple of numpy arrays into
                another tuple of numpy arrays.

        Returns:
            tfsnippet.dataflow.DataFlow: The data flow with ``mapper`` applied.
        """
        from .mapper_flow import MapperFlow
        return MapperFlow(self, mapper)

    # -------- here starts the factory methods for data flows --------
    @staticmethod
    def from_arrays(arrays, batch_size, shuffle=False, skip_incomplete=False):
        """
        Construct an :class:`ArrayFlow`.

        Args:
            arrays: List of numpy-like arrays, to be iterated through
                mini-batches.  These arrays should be at least 1-d,
                with identical first dimension.
            batch_size (int): Size of each mini-batch.
            shuffle (bool): Whether or not to shuffle data before iterating?
                (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude the last
                mini-batch if it is incomplete? (default :obj:`False`)

        Returns:
            tfsnippet.dataflow.ArrayFlow: The data flow from arrays.
        """
        from .array_flow import ArrayFlow
        return ArrayFlow(arrays, batch_size=batch_size, shuffle=shuffle,
                         skip_incomplete=skip_incomplete)
