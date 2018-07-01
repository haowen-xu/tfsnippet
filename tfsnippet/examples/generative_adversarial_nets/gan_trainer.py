from tfsnippet.dataflow import DataFlow
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer.base_trainer import BaseTrainer
from tfsnippet.trainer.feed_dict import merge_feed_dict, resolve_feed_dict

__all__ = ['GANTrainer']


class GANTrainer(BaseTrainer):
    """
    A subclass of :class:`BaseTrainer`, which optimizes generator and
    discriminator losses for generative adversarial nets.

    This class does not derive the training operations that minimizes or
    maximizes the losses.  Instead, the caller must derive the training
    operations and pass it to the :class:`GANTrainer`.
    """

    def __init__(self, loop, g_loss, g_train_op, d_iters, d_loss, d_train_op,
                 inputs, data_flow, feed_dict=None, g_feed_dict=None,
                 d_feed_dict=None, g_metric_name='g_loss',
                 d_metric_name='d_loss'):
        """
        Construct a new :class:`GANTrainer`.

        Args:
            loop (TrainLoop): The training loop object.
            g_loss (tf.Tensor): The generator training loss.
            g_train_op (tf.Operation): The generator training operation.
            d_iters (int): Number of iterations to train the discriminator,
                before each iteration to train the generator.
                Should be at least 1.
            d_loss (tf.Tensor): The discriminator training loss.
            d_train_op (tf.Operation): The discriminator training operation.
            inputs (list[tf.Tensor]): The input placeholders for training
                the discriminator.
            data_flow (DataFlow): The data flow for training the discriminator.
                Each mini-batch must contain one array for each placeholder
                in `inputs`.
            feed_dict: The feed dict for training both the generator and the
                discriminator.  (default :obj:`None`)
            g_feed_dict: The feed dict for training the generator.
                It will override `feed_dict` for training the generator.
                (default :obj:`None`)
            d_feed_dict: The feed dict for training the discriminator.
                It will override `feed_dict` for training the discriminator.
                (default :obj:`None`)
            g_metric_name (str): The metric name for collecting the
                generator training loss. (default "g_loss")
            d_metric_name (str): The metric name for collecting the
                discriminator training loss. (default "d_loss")
        """
        if loop.max_epoch != 1:
            raise ValueError('`max_epoch` must be set to 1 for `loop`.')
        if loop.max_step is None:
            raise ValueError('`max_step` must be configured for `loop`.')
        if d_iters < 1:
            raise ValueError('`d_iters` must be at least 1.')
        super(GANTrainer, self).__init__(loop=loop)

        # memorize the arguments
        self._g_loss = g_loss
        self._g_train_op = g_train_op
        self._d_iters = d_iters
        self._d_loss = d_loss
        self._d_train_op = d_train_op
        self._inputs = tuple(inputs or ())
        self._data_flow = data_flow
        self._feed_dict = dict(feed_dict or ())
        self._g_feed_dict = dict(g_feed_dict or ())
        self._d_feed_dict = dict(d_feed_dict or ())
        self._g_metric_name = g_metric_name
        self._d_metric_name = d_metric_name
        self._data_iterator = None

    @property
    def g_loss(self):
        """Get the generator training loss."""
        return self._g_loss

    @property
    def g_train_op(self):
        """Get the generator training operation."""
        return self._g_train_op

    @property
    def d_iters(self):
        """
        Number of iterations to train the discriminator, before each iteration
        to train the generator.
        """
        return self._d_iters

    @property
    def d_loss(self):
        """Get the discriminator training loss."""
        return self._d_loss

    @property
    def d_train_op(self):
        """Get the discriminator training operation."""
        return self._d_train_op

    @property
    def inputs(self):
        """
        Get the discriminator training input placeholders.

        Returns:
            tuple[tf.Tensor]: The list of training input placeholders.
        """
        return self._inputs

    @property
    def data_flow(self):
        """
        Get the discriminator training data flow.

        Returns:
            DataFlow: The discriminator training data flow.
        """
        return self._data_flow

    @property
    def feed_dict(self):
        """
        Get the feed dict for training generator and discriminator.

        Returns:
            dict[tf.Tensor, any]: The feed dict for generator and discriminator.
        """
        return self._feed_dict

    @property
    def g_feed_dict(self):
        """
        Get the feed dict for training generator only.

        Returns:
            dict[tf.Tensor, any]: The feed dict for generator only.
        """
        return self._g_feed_dict

    @property
    def d_feed_dict(self):
        """
        Get the feed dict for training discriminator only.

        Returns:
            dict[tf.Tensor, any]: The feed dict for discriminator only.
        """
        return self._d_feed_dict

    @property
    def g_metric_name(self):
        """
        Get the metric name for collecting the generator training loss.
        """
        return self._g_metric_name

    @property
    def d_metric_name(self):
        """
        Get the metric name for collecting the discriminator training loss.
        """
        return self._d_metric_name

    def run(self):
        def infinite_data_iterator():
            while iterator_alive[0]:
                for payload in self.data_flow:
                    yield payload
                    if not iterator_alive[0]:
                        break

        iterator_alive = [True]
        old_data_iterator = self._data_iterator
        data_iterator = infinite_data_iterator()
        try:
            self._data_iterator = data_iterator
            super(GANTrainer, self).run()
        finally:
            self._data_iterator = old_data_iterator
            # drain the infinite data iterator
            iterator_alive[0] = False
            try:
                while True:
                    _ = next(data_iterator)
            except StopIteration:
                pass

    def _iter_steps(self):
        return self.loop.iter_steps()

    def _run_step(self, session, step):
        # train the generator for one time
        if step > 1:
            g_feed_dict = resolve_feed_dict(
                merge_feed_dict(
                    self.feed_dict,
                    self.g_feed_dict
                )
            )
            _, g_loss = session.run(
                [self.g_train_op, self.g_loss],
                feed_dict=g_feed_dict
            )
            self.loop.collect_metrics({
                self.g_metric_name: g_loss
            })

        # train the discriminator for `d_iters` times
        for i in range(self.d_iters):
            d_batch_array = next(self._data_iterator)
            d_feed_dict = resolve_feed_dict(
                merge_feed_dict(
                    self.feed_dict,
                    self.d_feed_dict,
                    zip(self.inputs, d_batch_array)
                )
            )
            _, d_loss = session.run(
                [self.d_train_op, self.d_loss],
                feed_dict=d_feed_dict
            )

        self.loop.collect_metrics({
            self.d_metric_name: d_loss
        })
