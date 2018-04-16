import numpy as np

from tfsnippet.utils import minibatch_slices_iterator
from .base import ExtraInfoDataFlow

__all__ = ['SeqFlow']


class SeqFlow(ExtraInfoDataFlow):
    """
    Using number sequence as data source flow.

    This :class:`SeqFlow` is particularly used for generating the `seed`
    number indices, then fetch the actual data by :class:`MapperFlow`
    according to the seed numbers.

    Usage::

        seq_flow = DataFlow.seq(0, len(x), batch_size=256)
        mapper_flow = seq_flow.map(lambda idx: np.stack(
            [fetch_data_by_index(i) for i in idx]
        ))
    """

    def __init__(self, start, stop, step=1, batch_size=None, shuffle=False,
                 skip_incomplete=False, dtype=np.int32):
        """
        Construct a :class:`SeqFlow`.

        Args:
            start: The starting number of the sequence.
            stop: The ending number of the sequence.
            step: The step of the sequence. (default ``1``)
            batch_size: Batch size of the data flow. Required.
            shuffle (bool): Whether or not to shuffle the numbers before
                iterating? (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude the last
                mini-batch if it is incomplete? (default :obj:`False`)
            dtype: Data type of the numbers. (default ``np.int32``)
        """
        # check the parameters
        if batch_size is None:
            raise ValueError('`batch_size` is required.')

        # generate the numbers
        numbers = np.arange(start, stop, step, dtype=dtype)

        # memorize the parameters
        super(SeqFlow, self).__init__(
            array_count=1,
            data_length=len(numbers),
            data_shapes=((),),
            batch_size=batch_size,
            skip_incomplete=skip_incomplete,
            is_shuffled=shuffle
        )
        self._numbers = numbers
        self._start = start
        self._stop = stop
        self._step = step

    @property
    def start(self):
        """Get the starting number of the sequence."""
        return self._start

    @property
    def stop(self):
        """Get the ending number of the sequence."""
        return self._stop

    @property
    def step(self):
        """Get the step of the sequence."""
        return self._step

    def _minibatch_iterator(self):
        if self.is_shuffled:
            np.random.shuffle(self._numbers)

        for batch_s in minibatch_slices_iterator(
                length=self.data_length,
                batch_size=self.batch_size,
                skip_incomplete=self.skip_incomplete):
            yield (self._numbers[batch_s],)
