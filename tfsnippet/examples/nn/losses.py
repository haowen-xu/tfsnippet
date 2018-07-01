import tensorflow as tf

__all__ = [
    'classification_accuracy',
    'softmax_classification_loss',
    'softmax_classification_output',
]


def classification_accuracy(y_pred, y_true):
    """
    Compute the classification accuracy for `y_pred` and `y_true`.

    Args:
        y_pred: The predicted labels.
        y_true: The ground truth labels.

    Returns:
        tf.Tensor: The accuracy.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true)
    with tf.name_scope('classification_accuracy', values=[y_pred, y_true]):
        return tf.reduce_mean(
            tf.cast(tf.equal(y_pred, y_true), dtype=tf.float32))


def softmax_classification_loss(logits, y):
    """
    Get softmax classification training loss for given `logits` and label `y`.

    Args:
        logits: The softmax logits.
        y: The int32 labels for each logits.

    Returns:
        tf.Tensor: tf.float32 0-d tensor, the training loss.
    """
    logits = tf.convert_to_tensor(logits)
    y = tf.convert_to_tensor(y)
    with tf.name_scope('softmax_classification_loss', values=[logits, y]):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                           logits=logits)
        )


def softmax_classification_output(logits):
    """
    Get the most possible softmax classification output for each logit.

    Args:
        logits: The softmax logits.

    Returns:
        tf.Tensor: tf.int32 tensor, the class label for each logit.
    """
    logits = tf.convert_to_tensor(logits)
    with tf.name_scope('softmax_classification_output', values=[logits]):
        return tf.argmax(logits, axis=-1, output_type=tf.int32)
