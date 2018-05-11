from .base_trainer import BaseTrainer
from .validator import Validator, auto_loss_weight

__all__ = ['LossTrainer']


class LossTrainer(BaseTrainer):

    def __init__(self, loop, loss, train_op, inputs, feed_dict=None,
                 metric_name='loss'):
        super(LossTrainer, self).__init__(
            loop=loop, inputs=inputs, feed_dict=feed_dict)

        # memorize the variables
        self._loss = loss
        self._train_op = train_op
        self._metric_name = metric_name

    @property
    def loss(self):
        return self._loss

    def _fit_step(self, session, feed_dict):
        _, loss = session.run([self._train_op, self.loss], feed_dict=feed_dict)
        self.loop.collect_metrics({self._metric_name: loss})

    def add_default_validator(self, data_flow, feed_dict=None,
                              epochs=None, steps=None,
                              time_metric_name='valid_time',
                              loss_metric_name=None,
                              loss_weight_func=auto_loss_weight):
        merged_feed_dict = {}
        merged_feed_dict.update(self.feed_dict)
        if feed_dict is not None:
            merged_feed_dict.update(feed_dict)
        validator = Validator(
            loop=self.loop,
            loss=self.loss,
            inputs=self.inputs,
            data_flow=data_flow,
            feed_dict=merged_feed_dict,
            time_metric_name=time_metric_name,
            loss_metric_name=loss_metric_name,
            loss_weight_func=loss_weight_func,
        )
        self.add_validation(validator, epochs=epochs, steps=steps)
