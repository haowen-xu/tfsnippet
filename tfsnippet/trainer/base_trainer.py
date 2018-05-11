from tfsnippet.scaffold import TrainLoop
from tfsnippet.utils import (ensure_variables_initialized,
                             get_default_session_or_error)

from .feed_dict import resolve_feed_dict
from .trainer_hooks import TrainerHooks, HookPriority

__all__ = ['BaseTrainer']


class BaseTrainer(object):
    """Base class for all trainers."""

    def __init__(self, loop, inputs=None, feed_dict=None):
        """
        Construct a new :class:`BaseTrainer`.

        Args:
            loop (TrainLoop): The main loop for this trainer.
        """
        self._loop = loop
        self._inputs = list(inputs or ())
        self._feed_dict = dict(feed_dict or ())

        self._epoch_hooks = TrainerHooks()
        self._step_hooks = TrainerHooks()
        self._is_fitting = False

    @property
    def loop(self):
        """Get the main loop of this trainer."""
        return self._loop

    @property
    def inputs(self):
        return self._inputs

    @property
    def feed_dict(self):
        return self._feed_dict

    @property
    def epoch_hooks(self):
        return self._epoch_hooks

    @property
    def step_hooks(self):
        return self._step_hooks

    def fit(self, data_flow, feed_dict=None):
        if self._is_fitting:
            raise RuntimeError('`fit` is not re-entrant.')
        self._is_fitting = True
        try:
            # initialize global training status
            session = get_default_session_or_error()
            ensure_variables_initialized()
            self.loop.print_training_summary()

            # initialize internal status
            self.epoch_hooks.reset_counter()
            self.step_hooks.reset_counter()
            merged_feed_dict = {}

            for epoch in self.loop.iter_epochs():
                # run before epoch hook
                self.epoch_hooks.call_before()

                # run steps of this epoch
                for step, batch_data in self.loop.iter_steps(data_flow):
                    # run before step hook
                    self.step_hooks.call_before()

                    # prepare for the feed dict of this step
                    merged_feed_dict.clear()
                    merged_feed_dict.update(self.feed_dict)
                    if feed_dict is not None:
                        merged_feed_dict.update(feed_dict)
                    for ph, val in zip(self.inputs, batch_data):
                        merged_feed_dict[ph] = val
                    resolve_feed_dict(merged_feed_dict, inplace=True)

                    # run the step
                    self._fit_step(session, merged_feed_dict)

                    # run after step hook
                    self.step_hooks.call_after()

                # run after epoch hook
                self.epoch_hooks.call_after()
        finally:
            self._is_fitting = False

    def _fit_step(self, session, feed_dict):
        raise NotImplementedError()

    def add_epoch_or_step_hook(self, callback, epochs=None, steps=None,
                               priority=HookPriority.DEFAULT):
        if (epochs is not None and steps is not None) or \
                (epochs is None and steps is None):
            raise ValueError('One and only one of `epochs`, `steps` '
                             'should be specified.')
        if epochs is not None:
            if epochs <= 0:
                raise ValueError('`epochs` must be positive.')
            self.epoch_hooks.add_after(callback, freq=epochs, priority=priority)
        if steps is not None:
            if steps <= 0:
                raise ValueError('`steps` must be positive.')
            self.step_hooks.add_after(callback, freq=steps, priority=property)

    def clear_hooks_at_priority(self, priority):
        condition = lambda c, f, p: p == priority
        self.epoch_hooks.remove_if(condition)
        self.step_hooks.remove_if(condition)

    def add_logging(self, epochs=None, steps=None):
        self.add_epoch_or_step_hook(
            self.loop.print_logs, epochs=epochs, steps=steps,
            priority=HookPriority.LOGGING
        )

    def clear_logging(self):
        self.clear_hooks_at_priority(HookPriority.LOGGING)

    def add_validation(self, validator, epochs=None, steps=None):
        self.add_epoch_or_step_hook(
            validator.run, epochs=epochs, steps=steps,
            priority=HookPriority.VALIDATION
        )

    def clear_validation(self):
        self.clear_hooks_at_priority(HookPriority.VALIDATION)

    def add_annealing(self, value, epochs=None, steps=None):
        self.add_epoch_or_step_hook(
            value.anneal, epochs=epochs, steps=steps,
            priority=HookPriority.ANNEALING
        )

    def clear_annealing(self):
        self.clear_hooks_at_priority(HookPriority.ANNEALING)
