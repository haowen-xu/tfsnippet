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
    def __init__(self, key, freq, callback):
        assert(callable(callback))
        self.key = key
        self.freq = freq
        self.callback = callback

    def __call__(self, trainer):
        if getattr(trainer.loop, self.key) % self.freq == 0:
            return self.callback()

    def __repr__(self):  # for `test_base_trainer.py`
        return '{}:{}:{}'.format(self.callback, self.key, self.freq)


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

    The event schedule of a :class:`BaseTrainer` can be briefly described as::

        events.fire(EventKeys.BEFORE_EXECUTION, self)

        for epoch in epochs:
            events.fire(EventKeys.BEFORE_EPOCH, self)

            for step in steps:
                events.fire(EventKeys.BEFORE_STEP, self)

                ...  # actually train for a step

                events.fire(EventKeys.STEP_EVALUATION, self)
                events.fire(EventKeys.STEP_ANNEALING, self)
                events.fire(EventKeys.STEP_LOGGING, self)
                events.reverse_fire(EventKeys.AFTER_STEP, self)

            events.fire(EventKeys.EPOCH_EVALUATION, self)
            events.fire(EventKeys.EPOCH_ANNEALING, self)
            events.fire(EventKeys.EPOCH_LOGGING, self)
            events.reverse_fire(EventKeys.AFTER_EPOCH, self)

        events.reverse_fire(EventKeys.AFTER_EXECUTION, self)

    Using `trainer.events.on(EventKeys.AFTER_EPOCH, lambda trainer: ...)` can
    register an after-epoch event handler.  Handlers for other events can be
    registered in a similar way.

    To make things even simpler, we provide several methods to register
    callbacks that will run every few epochs/steps, e.g.::

        trainer.evaluate_after_epochs(
            lambda: print('after epoch callback'), 10)  # run every 10 epochs
        trainer.log_after_steps(1000)  # call `loop.print_logs` every 1000 steps
    """

    def __init__(self, loop, ensure_variables_initialized=True):
        """
        Initialize the internal states of :class:`BaseTrainer`.

        Args:
            loop (TrainLoop): The training loop object.
            ensure_variables_initialized (bool): Whether or not to ensure
                the variables are initialized in :meth:`run()`?
        """
        self._loop = loop
        self._ensure_variables_initialized = ensure_variables_initialized
        self._events = EventSource([
            EventKeys.BEFORE_EXECUTION,
            EventKeys.AFTER_EXECUTION,
            EventKeys.BEFORE_EPOCH,
            EventKeys.EPOCH_EVALUATION,
            EventKeys.EPOCH_ANNEALING,
            EventKeys.EPOCH_LOGGING,
            EventKeys.AFTER_EPOCH,
            EventKeys.BEFORE_STEP,
            EventKeys.STEP_EVALUATION,
            EventKeys.STEP_ANNEALING,
            EventKeys.STEP_LOGGING,
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
            # trigger the before execution event
            self.events.fire(EventKeys.BEFORE_EXECUTION, self)

            # initialize global training status
            session = get_default_session_or_error()
            if self._ensure_variables_initialized:
                ensure_variables_initialized()
            self.loop.print_training_summary()

            for _ in self.loop.iter_epochs():
                # trigger before epoch event
                self.events.fire(EventKeys.BEFORE_EPOCH, self)

                # run steps of this epoch
                for payload in self._iter_steps():
                    # trigger before step event
                    self.events.fire(EventKeys.BEFORE_STEP, self)

                    # run the step
                    self._run_step(session, payload)

                    # trigger after step events
                    self.events.fire(EventKeys.STEP_EVALUATION, self)
                    self.events.fire(EventKeys.STEP_ANNEALING, self)
                    self.events.fire(EventKeys.STEP_LOGGING, self)
                    self.events.reverse_fire(EventKeys.AFTER_STEP, self)

                # trigger after epoch events
                self.events.fire(EventKeys.EPOCH_EVALUATION, self)
                self.events.fire(EventKeys.EPOCH_ANNEALING, self)
                self.events.fire(EventKeys.EPOCH_LOGGING, self)
                self.events.reverse_fire(EventKeys.AFTER_EPOCH, self)

            # trigger the after execution event
            self.events.reverse_fire(EventKeys.AFTER_EXECUTION, self)
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
            EventKeys.STEP_LOGGING,
            OnEveryFewCalls('step', freq, self.loop.print_logs)
        )

    def log_after_epochs(self, freq):
        """
        Add a logging hook to run after every few epochs.

        Args:
            freq (int): The frequency for this logging hook to run.
        """
        self.events.on(
            EventKeys.EPOCH_LOGGING,
            OnEveryFewCalls('epoch', freq, self.loop.print_logs)
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
        self.events.clear_event_handlers(EventKeys.STEP_LOGGING)
        self.events.clear_event_handlers(EventKeys.EPOCH_LOGGING)

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
            EventKeys.STEP_EVALUATION,
            OnEveryFewCalls('step', freq, callback)
        )

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
            EventKeys.EPOCH_EVALUATION,
            OnEveryFewCalls('epoch', freq, callback)
        )

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
        self.events.clear_event_handlers(EventKeys.STEP_EVALUATION)
        self.events.clear_event_handlers(EventKeys.EPOCH_EVALUATION)

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
            EventKeys.STEP_ANNEALING,
            OnEveryFewCalls('step', freq, callback)
        )

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
            EventKeys.EPOCH_ANNEALING,
            OnEveryFewCalls('epoch', freq, callback)
        )

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
        self.events.clear_event_handlers(EventKeys.STEP_ANNEALING)
        self.events.clear_event_handlers(EventKeys.EPOCH_ANNEALING)
