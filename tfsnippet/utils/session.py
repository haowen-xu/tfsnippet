import os
from logging import getLogger

import six
import tensorflow as tf

from .imported import makedirs
from .scope import VarScopeObject

__all__ = [
    'create_session',
    'get_default_session_or_error',
    'get_variables_as_dict',
    'VariableSaver',
    'get_uninitialized_variables',
    'ensure_variables_initialized',
]


def create_session(lock_memory=True,
                   log_device_placement=False,
                   allow_soft_placement=True,
                   **kwargs):
    """
    A convenient method to create a TensorFlow session.

    Args:
        lock_memory (True or False or float):

            * If :obj:`True`, lock all free memory.

            * If :obj:`False`, set `allow_growth` to True, i.e., not to lock
                all free memory.

            * If float, lock this portion of memory.

            (default :obj:`None`)

        log_device_placement (bool): Whether to log the placement of graph
            nodes.   (default :obj:`False`)
        allow_soft_placement (bool): Whether or not to allow soft placement?
            (default :obj:`True`)
        \**kwargs: Other named parameters to be passed to `tf.ConfigProto`.

    Returns:
        tf.Session: The TensorFlow session.
    """
    config = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement,
                            **kwargs)
    if lock_memory is False:
        config.gpu_options.allow_growth = True
    elif isinstance(lock_memory, float):
        config.gpu_options.per_process_gpu_memory_fraction = lock_memory
    elif lock_memory is not True:
        raise TypeError('`lock_memory` must be True, False or float.')
    session = tf.Session(config=config)
    return session


def get_default_session_or_error():
    """
    Get the default session.

    Returns:
        tf.Session: The default session.

    Raises:
        RuntimeError: If there's no active session.
    """
    ret = tf.get_default_session()
    if ret is None:
        raise RuntimeError('No session is active')
    return ret


def get_variables_as_dict(scope=None, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    """
    Get TensorFlow variables as dict.

    Args:
        scope (str or tf.VariableScope or None): If :obj:`None`, will collect
            all the variables within current graph.  If a :class:`str` or a
            :class:`tf.VariableScope`, will collect the variables only from
            this scope. (default :obj:`None`)
        collection (str): Collect the variables only from this collection.
            (default ``tf.GraphKeys.GLOBAL_VARIABLES``)

    Returns:
        dict[str, tf.Variable]: Dict which maps from names to TensorFlow
            variables.  The names will be the full names of variables if
            `scope` is not specified, or the `relative names` within the
            `scope` otherwise. By `relative names` we mean the variable names
            without the common scope name prefix.
    """
    # get the common prefix to be stripped
    if isinstance(scope, tf.VariableScope):
        scope_name = scope.name
    else:
        scope_name = scope
    if scope_name and not scope_name.endswith('/'):
        scope_name += '/'
    scope_name_len = len(scope_name) if scope_name else 0

    # get the variables and strip the prefix
    variables = tf.get_collection(collection, scope_name)
    return {
        var.name[scope_name_len:].rsplit(':', 1)[0]: var
        for var in variables
    }


class VariableSaver(VarScopeObject):
    """Version controlled saving and restoring TensorFlow variables."""

    def __init__(self, variables, save_dir, max_versions=2,
                 filename='variables.dat', latest_file='latest',
                 save_meta=True, name=None, scope=None):
        """
        Construct the :class:`VariableSaver`.

        Args:
            variables (collections.Iterable[tf.Variable] or dict[str, any]):
                List of variables, or dict of variables with explicit keys,
                which should be saved and restored.
            save_dir (str): Directory where to place the saved variables.
            max_versions (int): Maximum versions to keep in the directory
                (Default is 2). At least 2 versions should be kept, in order to
                prevent corrupted checkpoint files caused by IO failure.
            filename (str): Name of the files of variable values (default is
                ``variables.dat``).
            latest_file (str): Name of the file which organizes the checkpoint
                versions (default is ``latest``).
            save_meta (bool): Whether or not to save meta graph (default
                is :obj:`True`).
            name (str): Optional name of this :class:`VariableSaver`
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
            scope (str): Optional scope of this :class:`VariableSaver`
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
        """
        if not isinstance(variables, dict):
            variables = list(variables)
        if max_versions < 2:
            raise ValueError('At least 2 versions should be kept')
        super(VariableSaver, self).__init__(scope, name)
        self.variables = variables
        self.save_dir = os.path.abspath(save_dir)
        self.filename = filename
        self.max_versions = max_versions
        self.latest_file = latest_file
        self.save_meta = save_meta
        with tf.variable_scope(self.variable_scope):
            self._saver = tf.train.Saver(
                var_list=self.variables, max_to_keep=self.max_versions,
                name='saver'
            )

    def get_latest_file(self):
        """Get the latest available checkpoint file."""
        return tf.train.latest_checkpoint(self.save_dir, self.latest_file)

    def save(self, global_step=None):
        """
        Save the checkpoint to file.

        Args:
            global_step (int or tf.Tensor): The global step counter.
        """
        sess = get_default_session_or_error()
        makedirs(self.save_dir, exist_ok=True)
        self._saver.save(
            sess,
            os.path.join(self.save_dir, self.filename),
            global_step=global_step,
            latest_filename=self.latest_file,
            write_meta_graph=self.save_meta
        )

    def restore(self, ignore_non_exist=False):
        """
        Restore the checkpoint from file if it exists.

        Args:
            ignore_non_exist (bool): Whether or not to ignore error if the
                checkpoint file does not exist? (default :obj:`False`)

        Raises:
            IOError: If the checkpoint files do not exist, and
                `ignore_non_exist` is not :obj:`True`.
        """
        file_path = self.get_latest_file()
        if file_path:
            sess = get_default_session_or_error()
            self._saver.restore(sess, file_path)
            getLogger(__name__).debug(
                'Restored from checkpoint file %r.', file_path)
        elif not ignore_non_exist:
            raise IOError('Checkpoint file does not exist in directory {}'.
                          format(self.save_dir))


def get_uninitialized_variables(variables=None, name=None):
    """
    Get uninitialized variables as a list.

    Args:
        variables (list[tf.Variable]): Collect only uninitialized variables
            within this list. If not specified, will collect all uninitialized
            variables within ``tf.GraphKeys.GLOBAL_VARIABLES`` collection.
        name (str): Name of this operation in TensorFlow graph.

    Returns:
        list[tf.Variable]: Uninitialized variables.
    """
    sess = get_default_session_or_error()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)
    with tf.name_scope(name, default_name='get_uninitialized_variables'):
        init_flag = sess.run(tf.stack(
            [tf.is_variable_initialized(v) for v in variables]
        ))
    return [v for v, f in zip(variables, init_flag) if not f]


def ensure_variables_initialized(variables=None, name=None):
    """
    Ensure variables are initialized.

    Args:
        variables (list[tf.Variable] or dict[str, tf.Variable]): Ensure only
            the variables within this collection to be initialized. If not
            specified, will ensure all variables within the collection
            `tf.GraphKeys.GLOBAL_VARIABLES` to be initialized.
        name (str): Name of this operation in TensorFlow graph. (default
            `ensure_variables_initialized`)
    """
    with tf.name_scope(name, default_name='ensure_variables_initialized'):
        if isinstance(variables, dict):
            variables = list(six.itervalues(variables))
        uninitialized = get_uninitialized_variables(variables)
        if uninitialized:
            sess = get_default_session_or_error()
            sess.run(tf.variables_initializer(uninitialized))
