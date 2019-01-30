from tfsnippet.scaffold import TrainLoop, EventKeys
from tfsnippet.utils import (ensure_variables_initialized,
                             get_default_session_or_error,
                             DocInherit, EventSource)

from .evaluator import Evaluator

__all__ = ['BaseTrainer']


def check_epochs_and_steps_arg(epochs=None, steps=None):
    if (epochs is not None and steps is not None) or \
            (epochs is None and steps is None):
        raise ValueError('One and only one of `epochs` and `steps` should '
                         'be specified.')


class OnEveryFewCalls(object):
    def __init__(self, freq, callback):
        assert(callable(callback))
        self.freq = freq
        self.callback = callback

    def __call__(self, counter__, *args, **kwargs):
        if counter__ % self.freq == 0:
            return self.callback(*args, **kwargs)

    def __repr__(self):  # for `test_base_trainer.py`
        return '{}:{}'.format(self.callback, self.freq)


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

    See Also:
        :class:`tfsnippet.trainer.LossTrainer`
    """

    def __init__(self, loop):
        """
        Initialize the internal states of :class:`BaseTrainer`.

        Args:
            loop (TrainLoop): The training loop object.
        """
        self._loop = loop
        self._events = EventSource([
            EventKeys.BEFORE_EPOCH,
            EventKeys.AFTER_EPOCH_EVAL,
            EventKeys.AFTER_EPOCH_ANNEAL,
            EventKeys.AFTER_EPOCH_LOG,
            EventKeys.AFTER_EPOCH,
            EventKeys.BEFORE_STEP,
            EventKeys.AFTER_STEP_EVAL,
            EventKeys.AFTER_STEP_ANNEAL,
            EventKeys.AFTER_STEP_LOG,
            EventKeys.AFTER_STEP,
        ])
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
    def events(self):
        """
        Get the event source object.

        Returns:
            EventSource: The event source object.
        """
        return self._events

    def run(self):
        """Run training loop."""
        if self._is_fitting:
            raise RuntimeError('`run()` is not re-entrant.')
        self._is_fitting = True
        try:
            # initialize global training status
            session = get_default_session_or_error()
            ensure_variables_initialized()
            self.loop.print_training_summary()

            for _ in self.loop.iter_epochs():
                epoch = int(self.loop.epoch)

                # trigger before epoch event
                self.events.fire(EventKeys.BEFORE_EPOCH, epoch)

                # run steps of this epoch
                for payload in self._iter_steps():
                    step = int(self.loop.step)

                    # trigger before step event
                    self.events.fire(EventKeys.BEFORE_STEP, step)

                    # run the step
                    self._run_step(session, payload)

                    # trigger after step events
                    self.events.fire(EventKeys.AFTER_STEP_EVAL, step)
                    self.events.fire(EventKeys.AFTER_STEP_ANNEAL, step)
                    self.events.fire(EventKeys.AFTER_STEP_LOG, step)
                    self.events.fire(EventKeys.AFTER_STEP, step)

                # trigger after epoch events
                self.events.fire(EventKeys.AFTER_EPOCH_EVAL, epoch)
                self.events.fire(EventKeys.AFTER_EPOCH_ANNEAL, epoch)
                self.events.fire(EventKeys.AFTER_EPOCH_LOG, epoch)
                self.events.fire(EventKeys.AFTER_EPOCH, epoch)
        finally:
            self._is_fitting = False

    def _iter_steps(self):
        """
        Subclasses should override this to iterate through steps.

        A common implementation of :meth:`_iter_steps()` might be::

            def _iter_steps(self):
                return self.loop.iter_steps(training_data)

        Yields:
            int or (int, tuple[np.ndarray]): The step counter, or the step
                counter as well as the step training data.  Will be directly
                given to :meth:`_fit_step` as the `payload` argument.
        """
        raise NotImplementedError()

    def _run_step(self, session, payload):
        """
        Subclasses should override this to run a training step.

        Args:
            session: The TensorFlow session.
            payload: The step payload generated by :meth:`_iter_steps`.
        """
        raise NotImplementedError()

    def log_after_steps(self, freq):
        """
        Add a logging hook to run after every few steps.

        Args:
            freq (int): The frequency for this logging hook to run.
        """
        self.events.on(
            EventKeys.AFTER_STEP_LOG,
            OnEveryFewCalls(freq, self.loop.print_logs)
        )

    def log_after_epochs(self, freq):
        """
        Add a logging hook to run after every few epochs.

        Args:
            freq (int): The frequency for this logging hook to run.
        """
        self.events.on(
            EventKeys.AFTER_EPOCH_LOG,
            OnEveryFewCalls(freq, self.loop.print_logs)
        )

    def log_after(self, epochs=None, steps=None):
        """
        Add a logging hook to run after every few epochs or steps.

        Args:
            epochs (None or int): Run validation after every this few `epochs`.
            steps (None or int): Run validation after every this few `steps`.

        Raises:
            ValueError: If both `epochs` and `steps` are specified, or neither
                is specified.
        """
        check_epochs_and_steps_arg(epochs, steps)
        if epochs is not None:
            return self.log_after_epochs(epochs)
        else:
            return self.log_after_steps(steps)

    def remove_log_hooks(self):
        """
        Remove logging hooks from all lists.

        Returns:
            int: The number of removed hooks.
        """
        self.events.clear_event_handlers(EventKeys.AFTER_STEP_LOG)
        self.events.clear_event_handlers(EventKeys.AFTER_EPOCH_LOG)

    def evaluate_after_steps(self, evaluator, freq):
        """
        Add an evaluation hook to run after every few steps.

        Args:
            evaluator (Evaluator or () -> any): A evaluator object
                (which has ``.run()``), or any callable object.
            freq (int): The frequency for this evaluation hook to run.
        """
        callback = evaluator if callable(evaluator) else evaluator.run
        self.events.on(
            EventKeys.AFTER_STEP_EVAL, OnEveryFewCalls(freq, callback))

    def evaluate_after_epochs(self, evaluator, freq):
        """
        Add an evaluation hook to run after every few epochs.

        Args:
            evaluator (Evaluator or () -> any): A evaluator object
                (which has ``.run()``), or any callable object.
            freq (int): The frequency for this evaluation hook to run.
        """
        callback = evaluator if callable(evaluator) else evaluator.run
        self.events.on(
            EventKeys.AFTER_EPOCH_EVAL, OnEveryFewCalls(freq, callback))

    def evaluate_after(self, evaluator, epochs=None, steps=None):
        """
        Add an evaluation hook to run after every few epochs or steps.

        Args:
            evaluator (Evaluator or () -> any): A evaluator object
                (which has ``.run()``), or any callable object.
            epochs (None or int): Run validation after every this few `epochs`.
            steps (None or int): Run validation after every this few `steps`.

        Raises:
            ValueError: If both `epochs` and `steps` are specified, or neither
                is specified.
        """
        check_epochs_and_steps_arg(epochs, steps)
        if epochs is not None:
            return self.evaluate_after_epochs(evaluator, freq=epochs)
        else:
            return self.evaluate_after_steps(evaluator, freq=steps)

    def remove_evaluation_hooks(self):
        """
        Remove evaluation hooks from all lists.

        Returns:
            int: The number of removed hooks.
        """
        self.events.clear_event_handlers(EventKeys.AFTER_STEP_EVAL)
        self.events.clear_event_handlers(EventKeys.AFTER_EPOCH_EVAL)

    # legacy names for evaluation
    validate_after_steps = evaluate_after_steps
    validate_after_epochs = evaluate_after_epochs
    validate_after = evaluate_after
    remove_validation_hooks = remove_evaluation_hooks

    def anneal_after_steps(self, value, freq):
        """
        Add an annealing hook to run after every few steps.

        Args:
            value (AnnealingVariable or () -> any): An annealing variable
                (which has ``.anneal()``), or any callable object.
            freq (int): The frequency for this annealing hook to run.
        """
        callback = value if callable(value) else value.anneal
        self.events.on(
            EventKeys.AFTER_STEP_ANNEAL, OnEveryFewCalls(freq, callback))

    def anneal_after_epochs(self, value, freq):
        """
        Add an annealing hook to run after every few epochs.

        Args:
            value (AnnealingVariable or () -> any): An annealing variable
                (which has ``.anneal()``), or any callable object.
            freq (int): The frequency for this annealing hook to run.
        """
        callback = value if callable(value) else value.anneal
        self.events.on(
            EventKeys.AFTER_EPOCH_ANNEAL, OnEveryFewCalls(freq, callback))

    def anneal_after(self, value, epochs=None, steps=None):
        """
        Add an annealing hook to run after every few epochs or steps.

        Args:
            value (AnnealingVariable or () -> any): An annealing variable
                (which has ``.anneal()``), or any callable object.
            epochs (None or int): Run validation after every this few `epochs`.
            steps (None or int): Run validation after every this few `steps`.

        Raises:
            ValueError: If both `epochs` and `steps` are specified, or neither
                is specified.
        """
        check_epochs_and_steps_arg(epochs, steps)
        if epochs is not None:
            return self.anneal_after_epochs(value, freq=epochs)
        else:
            return self.anneal_after_steps(value, freq=steps)

    def remove_annealing_hooks(self):
        """
        Remove annealing hooks from all lists.

        Returns:
            int: The number of removed hooks.
        """
        self.events.clear_event_handlers(EventKeys.AFTER_STEP_ANNEAL)
        self.events.clear_event_handlers(EventKeys.AFTER_EPOCH_ANNEAL)
