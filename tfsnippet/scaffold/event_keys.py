__all__ = ['EventKeys']


class EventKeys(object):
    """Defines event keys for TFSnippet."""
    # (TrainLoop, Trainer) Before executing an epoch.
    BEFORE_EPOCH = 'before_epoch'

    # (Trainer) Run evaluation after an epoch.
    AFTER_EPOCH_EVAL = 'after_epoch:eval'

    # (Trainer) Anneal after an epoch.
    AFTER_EPOCH_ANNEAL = 'after_epoch:anneal'

    # (Trainer) Log after an epoch.
    AFTER_EPOCH_LOG = 'after_epoch:log'

    # (TrainLoop, Trainer) After executing an epoch.
    AFTER_EPOCH = 'after_epoch'

    # (TrainLoop, Trainer) Before executing a step.
    BEFORE_STEP = 'before_step'

    # (Trainer) Run evaluation after a step.
    AFTER_STEP_EVAL = 'after_step:eval'

    # (Trainer) Anneal after a step.
    AFTER_STEP_ANNEAL = 'after_step:anneal'

    # (Trainer) Log after a step.
    AFTER_STEP_LOG = 'after_step:log'

    # (TrainLoop, Trainer) After executing a step.
    AFTER_STEP = 'after_step'

    # (Evaluator) Before executing the evaluator.
    BEFORE_EVALUATION = 'before_evaluation'

    # (Evaluator) After executing the evaluator.
    AFTER_EVALUATION = 'after_evaluation'
