from tfsnippet.trainer import BaseTrainer, Evaluator, AnnealingDynamicValue
from tfsnippet.utils import is_integer

__all__ = [
    'validate_strides_or_filter_arg',
    'check_epochs_and_steps_arg',
    'validate_after',
    'anneal_after',
]


def validate_strides_or_filter_arg(arg_name, arg_value):
    """
    Validate the `strides` or `filter` arg, to ensure it is a tuple of
    two integers.

    Args:
        arg_name (str): The name of the argument, for formatting error.
        arg_value: The value of the argument.

    Returns:
        (int, int): The validated argument.
    """

    if not is_integer(arg_value) and (not isinstance(arg_value, tuple) or
                                      len(arg_value) != 2 or
                                      not is_integer(arg_value[0]) or
                                      not is_integer(arg_value[1])):
        raise TypeError('`{}` must be a int or a tuple (int, int).'.
                        format(arg_name))
    if not isinstance(arg_value, tuple):
        arg_value = (arg_value, arg_value)
    arg_value = tuple(int(v) for v in arg_value)
    return arg_value


def check_epochs_and_steps_arg(epochs=None, steps=None):
    """
    Check the argument `epochs` and `steps` to ensure one and only one
    of them is specified.
    """
    if (epochs is not None and steps is not None) or \
            (epochs is None and steps is None):
        raise ValueError('One and only one of `epochs` and `steps` should '
                         'be specified.')


def validate_after(trainer, evaluator, epochs=None, steps=None):
    """
    Run validation after every `epochs` or every `steps`.

    Args:
        trainer (BaseTrainer): The trainer object.
        evaluator (Evaluator): The evaluator object.
        epochs (None or int): Run validation after every `epochs`.
        steps (None or int): Run validation after every `steps`.

    Raises:
         ValueError: If both `epochs` and `steps` are specified, or
            neither is specified.
    """
    check_epochs_and_steps_arg(epochs, steps)
    if epochs is not None:
        return trainer.evaluate_after_epochs(evaluator, freq=epochs)
    else:
        return trainer.evaluate_after_steps(evaluator, freq=steps)


def anneal_after(trainer, value, epochs=None, steps=None):
    """
    Anneal dynamic value after every `epochs` or every `steps`.

    Args:
        trainer (BaseTrainer): The trainer object.
        value (AnnealingDynamicValue): The value to be annealed.
        epochs (None or int): Run validation after every `epochs`.
        steps (None or int): Run validation after every `steps`.

    Raises:
         ValueError: If both `epochs` and `steps` are specified, or
            neither is specified.
    """
    check_epochs_and_steps_arg(epochs, steps)
    if epochs is not None:
        return trainer.anneal_after_epochs(value, freq=epochs)
    else:
        return trainer.anneal_after_steps(value, freq=steps)
