__all__ = ['EventKeys']


class EventKeys(object):
    """Defines event keys for TFSnippet."""
    # (TrainLoop) Enter the train loop.
    ENTER_LOOP = 'enter'

    # (TrainLoop) Exit the train loop.
    EXIT_LOOP = 'exit'

    # (TrainLoop) When metrics (except time metrics) have been collected.
    METRICS_COLLECTED = 'metrics_collected'

    # (TrainLoop) When time metrics have been collected.
    TIME_METRICS_COLLECTED = 'time_metrics_collected'

    # (TrainLoop) When TensorFlow summary has been added.
    SUMMARY_ADDED = 'summary_added'

    # (TrainLoop, Trainer) Before executing an epoch.
    BEFORE_EPOCH = 'before_epoch'

    # (Trainer) Run evaluation after an epoch.
    EPOCH_EVALUATION = 'after_epoch:eval'

    # (Trainer) Anneal after an epoch.
    EPOCH_ANNEALING = 'after_epoch:anneal'

    # (Trainer) Log after an epoch.
    EPOCH_LOGGING = 'after_epoch:log'

    # (TrainLoop, Trainer) After executing an epoch.
    AFTER_EPOCH = 'after_epoch'

    # (TrainLoop, Trainer) Before executing a step.
    BEFORE_STEP = 'before_step'

    # (Trainer) Run evaluation after a step.
    STEP_EVALUATION = 'after_step:eval'

    # (Trainer) Anneal after a step.
    STEP_ANNEALING = 'after_step:anneal'

    # (Trainer) Log after a step.
    STEP_LOGGING = 'after_step:log'

    # (TrainLoop, Trainer) After executing a step.
    AFTER_STEP = 'after_step'

    # (Trainer, Evaluator) Before execution.
    BEFORE_EXECUTION = 'before_execution'

    # (Evaluator) After execution.
    AFTER_EXECUTION = 'after_execution'
