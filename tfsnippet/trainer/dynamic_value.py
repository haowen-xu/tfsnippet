from tfsnippet.utils import DocInherit

__all__ = [
    'DynamicValue',
]


@DocInherit
class DynamicValue(object):
    """
    Dynamic values fed into trainers and evaluators.

    For example, if you want to feed a learning rate into trainer, which
    shrinks into half every 100 epochs, you may use the following code::

        class MyLearningRate(object):

            def __init__(self, loop):
                self.loop = loop

            def get(self):
                return 0.001 * int(self.loop.epoch // 100) * 0.5

        learning_rate = tf.placeholder(dtype=tf.float32, shape=())
        ...

        with TrainLoop(...) as loop:
            trainer = Trainer(
                ...,
                feed_dict={learning_rate: MyLearningRate(loop)}
            )
            trainer.run()
    """

    def get(self):
        """Get the current value of this :class:`DynamicValue` object."""
        raise NotImplementedError()
