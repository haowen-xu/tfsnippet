from tfsnippet.scaffold import TrainLoop
from tfsnippet.utils import DocInherit

__all__ = [
    'DynamicValue', 'AnnealingScalar',
]


@DocInherit
class DynamicValue(object):
    """
    Dynamic values to be fed into trainers and evaluators.

    For example, if you want to feed a learning rate into trainer, which
    shrinks into half every 100 epochs, you may use the following code::

        class MyLearningRate(spt.DynamicValue):

            def __init__(self, loop):
                self.loop = loop

            def get(self):
                return 0.001 * int(self.loop.epoch // 100) * 0.5

        learning_rate = tf.placeholder(dtype=tf.float32, shape=())
        ...

        with spt.TrainLoop(...) as loop:
            trainer = spt.Trainer(
                ...,
                feed_dict={learning_rate: MyLearningRate(loop)}
            )
            trainer.run()

    Or you may also use :class:`AnnealingScalar`, a class that has already
    implemented such behaviour.
    """

    def get(self):
        """Get the current value of this :class:`DynamicValue` object."""
        raise NotImplementedError()


class AnnealingScalar(DynamicValue):
    """
    A :class:`DynamicValue` scalar, which anneals every few epochs or steps.

    For example, to anneal the learning rate every 100 epochs::

        learning_rate = tf.placeholder(dtype=tf.float32, shape=())
        ...

        with spt.TrainLoop(...) as loop:
            trainer = spt.Trainer(
                ...,
                feed_dict={learning_rate: spt.AnnealingScalar(
                    loop, initial=0.001, ratio=0.5, epochs=100)}
            )
    """

    def __init__(self, loop, initial_value, ratio, epochs=None, steps=None,
                 min_value=None, max_value=None):
        """
        Construct a new :class:`AnnealingScalar`.

        Args:
            loop (TrainLoop): The training loop object.
            initial_value (float): A float number, the initial value.
            ratio (float): A float number, the ratio of annealing at each time.
            epochs (int): Anneal every this number of epochs.
                One and only one of `epochs` and `steps` should be specified.
            steps (int): Anneal every this number of steps.
                One and only one of `epochs` and `steps` should be specified.
            min_value (float): Optional, a float number, the minimum value.
            max_value (float): Optional, a float number, the maximum value.
        """
        initial_value = float(initial_value)
        ratio = float(ratio)
        if min_value is not None:
            min_value = float(min_value)
            if initial_value < min_value:
                raise ValueError('`initial_value` must >= `min_value`: '
                                 'initial_value {} vs min_value {}'.
                                 format(initial_value, min_value))

        if max_value is not None:
            max_value = float(max_value)
            if min_value is not None and max_value < min_value:
                raise ValueError('`min_value` must <= `max_value`: '
                                 'min_value {} vs max_value {}'.
                                 format(min_value, max_value))
            if initial_value > max_value:
                raise ValueError('`initial_value` must <= `max_value`: '
                                 'initial_value {} vs max_value {}'.
                                 format(initial_value, max_value))

        if (epochs is None and steps is None) or \
                (epochs is not None and steps is not None):
            raise ValueError('One and only one of `epochs` and `steps` '
                             'should be specified.')

        if epochs is not None:
            epochs = int(epochs)
            if epochs < 1:
                raise ValueError('`epochs` must be positive: {}'.format(epochs))

        if steps is not None:
            steps = int(steps)
            if steps < 1:
                raise ValueError('`steps` must be positive: {}'.format(steps))

        self._loop = loop
        self._initial_value = initial_value
        self._ratio = ratio
        self._epochs = epochs
        self._steps = steps
        self._min_value = min_value
        self._max_value = max_value

        self._cache_value = None
        self._cache_epoch = None
        self._cache_step = None

    def get(self):
        if (self._epochs is not None and
            self._cache_epoch != self._loop.epoch) or \
                (self._steps is not None and
                 self._cache_step != self._loop.step):
            if self._epochs is not None:
                freq_count = int(max(self._loop.epoch - 1, 0) // self._epochs)
            else:
                freq_count = int(max(self._loop.step - 1, 0) // self._steps)

            scale = self._ratio ** freq_count
            value = self._initial_value * scale

            if self._max_value is not None:
                value = min(self._max_value, value)
            if self._min_value is not None:
                value = max(self._min_value, value)

            self._cache_value = value
            self._cache_epoch = self._loop.epoch
            self._cache_step = self._loop.step

        return self._cache_value
