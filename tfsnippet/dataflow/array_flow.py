import numpy as np

from tfsnippet.utils import minibatch_slices_iterator
from .data_flow import DataFlow

__all__ = ['ArrayFlow']


def _make_readonly(arr):
    arr = np.asarray(arr)
    arr.setflags(write=False)
    return arr


class ArrayFlow(DataFlow):
    """
    Data source flow from numpy-like arrays.

    Usage::

        array_flow = DataFlow.from_arrays([x, y], batch_size=256, shuffle=True,
                                          skip_incomplete=True)
        for batch_x, batch_y in array_flow:
            ...
    """

    def __init__(self, arrays, batch_size,
                 shuffle=False, skip_incomplete=False):
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
        self._batch_size = batch_size
        self._data_shapes = tuple(a.shape[1:] for a in arrays)
        self._shuffled = shuffle
        self._skip_incomplete = skip_incomplete

        # internal indices buffer
        self._indices_buffer = None

    def _minibatch_iterator(self):
        # shuffle the source arrays if necessary
        if self._shuffled:
            if self._indices_buffer is None:
                t = np.int32 if self._data_length < (1 << 31) else np.int64
                self._indices_buffer = np.arange(self._data_length, dtype=t)
            np.random.shuffle(self._indices_buffer)

            def get_slice(s):
                return tuple(
                    _make_readonly(a[self._indices_buffer[s]])
                    for a in self._arrays
                )
        else:
            def get_slice(s):
                return tuple(_make_readonly(a[s]) for a in self._arrays)

        # now iterator through the mini-batches
        for batch_s in minibatch_slices_iterator(
                length=self._data_length,
                batch_size=self._batch_size,
                skip_incomplete=self._skip_incomplete):
            yield get_slice(batch_s)

    @property
    def array_count(self):
        """
        Get the count of arrays in each mini-batch.

        Returns:
            int: The count of arrays in each mini-batch.
        """
        return self._array_count

    @property
    def data_length(self):
        """
        Get the total length of the data.

        Returns:
            int: The total length of the data.
        """
        return self._data_length

    @property
    def data_shapes(self):
        """
        Get the shapes of the data in each mini-batch.

        Returns:
            tuple[tuple[int]]: The shapes of data in a mini-batch.
                The batch dimension is not included.
        """
        return self._data_shapes

    @property
    def batch_size(self):
        """
        Get the size of each mini-batch.

        Returns:
            int: The size of each mini-batch.
        """
        return self._batch_size

    @property
    def skip_incomplete(self):
        """
        Whether or not to exclude the last mini-batch if it is incomplete?
        """
        return self._skip_incomplete

    @property
    def is_shuffled(self):
        """
        Whether or not the data are first shuffled before iterated through
        mini-batches?
        """
        return self._shuffled
