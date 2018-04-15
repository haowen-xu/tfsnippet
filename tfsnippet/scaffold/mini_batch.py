import numpy as np

__all__ = [
    'minibatch_slices_iterator', 'StaticMiniBatch', 'MiniBatch',
    'ArrayMiniBatch'
]


def minibatch_slices_iterator(length, batch_size,
                              ignore_incomplete_batch=False):
    """
    Iterate through all the mini-batch slices.

    Args:
        length (int): Total length of data in an epoch.
        batch_size (int): Size of each mini-batch.
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)

    Yields
        slice: Slices of each mini-batch.  The last mini-batch may contain
               less indices than `batch_size`.
    """
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class MiniBatch(object):
    """
    Base class of mini-batches generators for training, validation and testing.

    Each instance of a :class:`MiniBatch` is designed with fixed data source.
    Also, :method:`get_iterator` is not designed to be re-entrant, since the
    :class:`MiniBatch` instances may have internal states for an iterator.
    """

    @property
    def batch_size(self):
        """
        Get the size of each mini-batch.

        Returns:
            int: The size of each mini-batch.
        """
        raise NotImplementedError()

    @property
    def ignore_incomplete_batch(self):
        """
        Whether or not to exclude the last mini-batch if it is incomplete?
        """
        raise NotImplementedError()

    @property
    def is_shuffled(self):
        """
        Whether or not the data are first shuffled before iterated through
        mini-batches?
        """
        raise NotImplementedError()

    def _get_iterator(self):
        raise NotImplementedError()

    _GET_ITERATOR_REENTRANT_FLAG_FIELD = '__get_iterator_entered'

    def get_iterator(self):
        """
        Get the iterator of mini-batches.

        Yields:
            tuple[np.ndarray]: Yields tuples of numpy arrays in each mini-batch.

        Raises:
            RuntimeError: If this method is re-entered.
        """
        if getattr(self, self._GET_ITERATOR_REENTRANT_FLAG_FIELD, False):
            raise RuntimeError('get_iterator of MiniBatch is not re-entrant.')
        setattr(self, self._GET_ITERATOR_REENTRANT_FLAG_FIELD, True)
        try:
            for b in self._get_iterator():
                yield b
        finally:
            delattr(self, self._GET_ITERATOR_REENTRANT_FLAG_FIELD)

    def __iter__(self):
        return self.get_iterator()


class StaticMiniBatch(MiniBatch):
    """
    Additional constraints for sub-classes of :class:`MiniBatch` whose
    ``array_count``, ``data_length`` and ``data_shapes`` are known.
    """

    @property
    def array_count(self):
        """
        Get the count of arrays in each mini-batch.

        Returns:
            int: The count of arrays in each mini-batch.
        """
        raise NotImplementedError()

    @property
    def data_length(self):
        """
        Get the total length of the data.

        Returns:
            int or None: The total length of the data.
        """
        raise NotImplementedError()

    @property
    def data_shapes(self):
        """
        Get the shapes of the data in each mini-batch.

        Returns:
            tuple[tuple[int]]: The shapes of data in a mini-batch.
                The batch dimension is not included.
        """
        raise NotImplementedError()


class ArrayMiniBatch(StaticMiniBatch):
    """
    Mini-batches generator constructed from numpy-like arrays.

    Usage::

        for batch_x, batch_y in ArrayMiniBatch([x, y], batch_size=256):
            ...
    """

    def __init__(self, arrays, batch_size,
                 shuffle=False, ignore_incomplete_batch=False):
        """
        Construct a :class:`ArrayMiniBatch`.

        Args:
            arrays: List of numpy-like arrays, to be iterated through
                mini-batches.  These arrays should be at least 1-d,
                with identical first dimension.
            batch_size (int): Size of each mini-batch.
            shuffle (bool): Whether or not to shuffle data before iterating?
                (default :obj:`False`)
            ignore_incomplete_batch (bool): Whether or not to exclude the
                last mini-batch if it is incomplete? (default :obj:`False`)
        """
        # validate parameters
        arrays = tuple(arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty.')
        for a in arrays:
            if not hasattr(a, 'shape'):
                raise ValueError('`arrays` must be numpy-like arrays.')
            if len(a.shape) < 1:
                raise ValueError('`arrays` must be at least 1-d arrays.')
        data_length = len(arrays[0])
        for a in arrays[1:]:
            if len(a) != data_length:
                raise ValueError('`arrays` must have the same data length.')

        # memorize the parameters
        self._arrays = arrays
        self._array_count = len(arrays)
        self._data_length = data_length
        self._data_shapes = tuple(a.shape[1:] for a in arrays)
        self._batch_size = batch_size
        self._is_shuffled = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

        # internal indices buffer
        self._indices_buffer = None

    @property
    def array_count(self):
        return self._array_count

    @property
    def data_length(self):
        return self._data_length

    @property
    def data_shapes(self):
        return self._data_shapes

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def ignore_incomplete_batch(self):
        return self._ignore_incomplete_batch

    @property
    def is_shuffled(self):
        return self._is_shuffled

    def get_iterator(self):
        # shuffle the source arrays if necessary
        if self._is_shuffled:
            if self._indices_buffer is None:
                t = np.int32 if self._data_length < (1 << 31) else np.int64
                self._indices_buffer = np.arange(self._data_length, dtype=t)
            np.random.shuffle(self._indices_buffer)

            def get_slice(s):
                return tuple(a[self._indices_buffer[s]] for a in self._arrays)
        else:
            def get_slice(s):
                return tuple(a[s] for a in self._arrays)

        # now iterator through the mini-batches
        for batch_s in minibatch_slices_iterator(
                length=self._data_length,
                batch_size=self._batch_size,
                ignore_incomplete_batch=self._ignore_incomplete_batch):
            yield get_slice(batch_s)
