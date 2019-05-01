__all__ = ['EventKeys']


class EventKeys(object):
    """Defines event keys for TFSnippet."""
    # (TrainLoop) Enter the train loop.
    ENTER_LOOP = 'enter_loop'

    # (TrainLoop) Exit the train loop.
    EXIT_LOOP = 'exit_loop'

    # (TrainLoop) When metrics (except time metrics) have been collected.
    METRICS_COLLECTED = 'metrics_collected'

    # (TrainLoop) When time metrics have been collected.
    TIME_METRICS_COLLECTED = 'time_metrics_collected'

    # (TrainLoop) When metric statistics have been printed.
    METRIC_STATS_PRINTED = 'metric_stats_printed'

    # (TrainLoop) When time metric statistics have been printed.
    TIME_METRIC_STATS_PRINTED = 'time_metric_stats_printed'

    # (TrainLoop) When TensorFlow summary has been added.
    SUMMARY_ADDED = 'summary_added'

    # (TrainLoop, Trainer) Before executing an epoch.
    BEFORE_EPOCH = 'before_epoch'

    # (Trainer) Run evaluation after an epoch.
    EPOCH_EVALUATION = 'epoch_evaluation'

    # (Trainer) Anneal after an epoch.
    EPOCH_ANNEALING = 'epoch_annealing'

    # (Trainer) Log after an epoch.
    EPOCH_LOGGING = 'epoch_logging'

    # (TrainLoop, Trainer) After executing an epoch.
    AFTER_EPOCH = 'after_epoch'

    # (TrainLoop, Trainer) Before executing a step.
    BEFORE_STEP = 'before_step'

    # (Trainer) Run evaluation after a step.
    STEP_EVALUATION = 'step_evaluation'

    # (Trainer) Anneal after a step.
    STEP_ANNEALING = 'step_annealing'

    # (Trainer) Log after a step.
    STEP_LOGGING = 'step_logging'

    # (TrainLoop, Trainer) After executing a step.
    AFTER_STEP = 'after_step'

    # (Trainer, Evaluator) Before execution.
    BEFORE_EXECUTION = 'before_execution'

    # (Evaluator) After execution.
    AFTER_EXECUTION = 'after_execution'
