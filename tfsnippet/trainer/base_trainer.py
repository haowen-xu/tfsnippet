from tfsnippet.scaffold import TrainLoop
from tfsnippet.utils import (ensure_variables_initialized,
                             get_default_session_or_error,
                             DocInherit)

from .dynamic_values import AnnealingDynamicValue
from .feed_dict import resolve_feed_dict
from .hooks import HookPriority, HookList
from .validator import Validator

__all__ = ['BaseTrainer']


@DocInherit
class BaseTrainer(object):
    """
    Base class for all trainers.

    All the trainers provided in :mod:`tfsnippet.trainer` are not
    designed to take control of the training totally, which is often
    assumed in other libraries such as Keras.  Instead, it just takes
    responsibility of assembling different steps of a training process
    together, and run the main training loop.  So it is usually the caller's
    responsibility to derive his training operation from a certain TensorFlow
    optimizer, and pass it to a proper trainer.

    You may see an example of usage in :class:`~tfsnippet.trainer.LossTrainer`.
    """

    def __init__(self, loop, inputs, data_flow, feed_dict=None):
        """
        Construct a new :class:`BaseTrainer`.

        Args:
            loop (TrainLoop): The training loop object.
            inputs (list[tf.Tensor]): The input placeholders.
                The number of tensors, and the order of tensors, should
                both match the arrays of each mini-batch data, provided
                by `data_flow`.
            data_flow (DataFlow): The training data flow.
            feed_dict (dict[tf.Tensor, any]): The fixed feed dict for
                training.  It will be merged with `inputs` and the
                argument of ``run(feed_dict)``. (default :obj:`None`)
        """
        self._loop = loop
        self._inputs = list(inputs or ())
        self._data_flow = data_flow
        self._feed_dict = dict(feed_dict or ())

        self._before_epochs = HookList()
        self._after_epochs = HookList()
        self._before_steps = HookList()
        self._after_steps = HookList()
        self._hook_lists = (
            self._before_epochs, self._before_steps, self._after_steps,
            self._after_epochs
        )

        self._is_fitting = False

    @property
    def loop(self):
        """
        Get the training loop object.

        Returns:
            TrainLoop: The training loop object.
        """
        return self._loop

    @property
    def inputs(self):
        """
        Get the input placeholders.

        Returns:
            list[tf.Tensor]: The input placeholders.
        """
        return self._inputs

    @property
    def data_flow(self):
        """
        Get the training data flow.

        Returns:
            DataFlow: The training data flow.
        """
        return self._data_flow

    @property
    def feed_dict(self):
        """
        Get the fixed feed dict.

        Returns:
            dict[tf.Tensor, any]: The fixed feed dict.
        """
        return self._feed_dict

    def run(self, feed_dict=None):
        """
        Run training.

        Args:
            feed_dict (dict[tf.Tensor, any]): The extra feed dict to be
                merged with the already configured dict.  (default :obj:`None`)
        """
        if self._is_fitting:
            raise RuntimeError('`run()` is not re-entrant.')
        self._is_fitting = True
        try:
            # initialize global training status
            session = get_default_session_or_error()
            ensure_variables_initialized()
            self.loop.print_training_summary()

            # initialize internal status
            for hook_list in self.hook_lists:
                hook_list.reset()
            merged_feed_dict = {}

            for epoch in self.loop.iter_epochs():
                # run before epoch hook
                self.before_epochs.call_hooks()

                # run steps of this epoch
                for step, batch_data in self.loop.iter_steps(self.data_flow):
                    # run before step hook
                    self.before_steps.call_hooks()

                    # prepare for the feed dict of this step
                    merged_feed_dict.clear()
                    merged_feed_dict.update(self.feed_dict)
                    if feed_dict is not None:
                        merged_feed_dict.update(feed_dict)
                    for ph, val in zip(self.inputs, batch_data):
                        merged_feed_dict[ph] = val

                    # run the step
                    self._fit_step(session, resolve_feed_dict(merged_feed_dict))

                    # run after step hook
                    self.after_steps.call_hooks()

                # run after epoch hook
                self.after_epochs.call_hooks()
        finally:
            self._is_fitting = False

    def _fit_step(self, session, feed_dict):
        raise NotImplementedError()

    @property
    def before_epochs(self):
        """
        Get the hooks run before epochs.

        Returns:
            HookList: The hook list.
        """
        return self._before_epochs

    @property
    def after_epochs(self):
        """
        Get the hooks run after epochs.

        Returns:
            HookList: The hook list.
        """
        return self._after_epochs

    @property
    def before_steps(self):
        """
        Get the hooks run before steps.

        Returns:
            HookList: The hook list.
        """
        return self._before_steps

    @property
    def after_steps(self):
        """
        Get the hooks run after steps.

        Returns:
            HookList: The hook list.
        """
        return self._after_steps

    @property
    def hook_lists(self):
        """
        Get all the hook lists.

        Returns:
            tuple[HookList]: The tuple (self.before_epochs, self.before_steps,
                self.after_steps, self.after_epochs).
        """
        return self._hook_lists

    def remove_by_priority(self, priority):
        """
        Remove hooks having the specified `priority` from all lists.

        Args:
            priority: The priority of the hooks to be removed.

        Returns:
            int: The number of removed hooks.
        """
        ret = 0
        for hook_list in self.hook_lists:
            ret += hook_list.remove_by_priority(priority)
        return ret

    def log_after_steps(self, freq):
        """
        Add a logging hook to run after every few steps.

        Args:
            freq (int): The frequency for this logging hook to run.
        """
        self.after_steps.add_hook(
            self.loop.print_logs, freq=freq, priority=HookPriority.LOGGING)

    def log_after_epochs(self, freq):
        """
        Add a logging hook to run after every few epochs.

        Args:
            freq (int): The frequency for this logging hook to run.
        """
        self.after_epochs.add_hook(
            self.loop.print_logs, freq=freq, priority=HookPriority.LOGGING)

    def remove_log_hooks(self):
        """
        Remove logging hooks from all lists.

        Returns:
            int: The number of removed hooks.
        """
        return self.remove_by_priority(HookPriority.LOGGING)

    def validate_after_steps(self, validator, freq):
        """
        Add a validation hook to run after every few steps.

        Args:
            validator (Validator or () -> any): A validator object
                (which has ``.run()``), or any callable object.
            freq (int): The frequency for this validation hook to run.
        """
        callback = validator if callable(validator) else validator.run
        self.after_steps.add_hook(
            callback, freq=freq, priority=HookPriority.VALIDATION)

    def validate_after_epochs(self, validator, freq):
        """
        Add a validation hook to run after every few epochs.

        Args:
            validator (Validator or () -> any): A validator object
                (which has ``.run()``), or any callable object.
            freq (int): The frequency for this validation hook to run.
        """
        callback = validator if callable(validator) else validator.run
        self.after_epochs.add_hook(
            callback, freq=freq, priority=HookPriority.VALIDATION)

    def remove_validation_hooks(self):
        """
        Remove validation hooks from all lists.

        Returns:
            int: The number of removed hooks.
        """
        return self.remove_by_priority(HookPriority.VALIDATION)

    def anneal_after_steps(self, value, freq):
        """
        Add an annealing hook to run after every few steps.

        Args:
            value (AnnealingDynamicValue or () -> any): An annealing dynamic
                value (which has ``.anneal()``), or any callable object.
            freq (int): The frequency for this annealing hook to run.
        """
        callback = value if callable(value) else value.anneal
        self.after_steps.add_hook(
            callback, freq=freq, priority=HookPriority.ANNEALING)

    def anneal_after_epochs(self, value, freq):
        """
        Add an annealing hook to run after every few epochs.

        Args:
            value (AnnealingDynamicValue or () -> any): An annealing dynamic
                value (which has ``.anneal()``), or any callable object.
            freq (int): The frequency for this annealing hook to run.
        """
        callback = value if callable(value) else value.anneal
        self.after_epochs.add_hook(
            callback, freq=freq, priority=HookPriority.ANNEALING)

    def remove_annealing_hooks(self):
        """
        Remove annealing hooks from all lists.

        Returns:
            int: The number of removed hooks.
        """
        return self.remove_by_priority(HookPriority.ANNEALING)
