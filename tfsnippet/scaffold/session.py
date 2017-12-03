import os
from logging import getLogger

import tensorflow as tf

from tfsnippet.utils import makedirs
from .scope import VarScopeObject

__all__ = [
    'get_default_session_or_error',
    'get_variables_as_dict',
    'VariableSaver',
]


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
        scope (str or tf.VariableScope):
            Name of a variable scope, or a variable scope object.
            Only the variables within this scope would be returned if specified.

        collection (str):
            Only the variables belonging to this collection would be returned.
            (default is ``tf.GraphKeys.GLOBAL_VARIABLES``)

    Returns:
        dict[str, tf.Variable]:
            Dict which maps from names to TensorFlow variables.

            The names will be the full names of variables if `scope` is not
            specified, or the `relative names` within the `scope` otherwise.
            By `relative names` we mean the variable names without the common
            scope name prefix.
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
    """
    Version controlled saving and restoring TensorFlow variables.

    Args:
        variables (collections.Iterable[tf.Variable] or dict[str, any]):
            List of variables, or dict of variables with explicit keys,
            which should be saved and restored.

        save_dir (str):
            Directory where to place the saved variables.

        max_versions (int):
            Maximum versions to keep in the directory (Default is 2).
            At least 2 versions should be kept, in order to prevent corrupted
            checkpoint files caused by IO failure.

        filename (str):
            Name of the files of variable values (default is ``variables.dat``).

        latest_file (str):
            Name of the file which organizes the checkpoint versions
            (default is ``latest``).

        save_meta (bool):
            Whether or not to save meta graph (default is :obj:`True`).

        name (str):
            Optional name of this :class:`VariableSaver`.

        scope (str):
            Optional scope of this :class:`VariableSaver`.
    """

    def __init__(self, variables, save_dir, max_versions=2,
                 filename='variables.dat', latest_file='latest',
                 save_meta=True, name=None, scope=None):
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
            ignore_non_exist (bool):
                Whether or not to ignore error if the checkpoint file does not
                exist? (default :obj:`False`)
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
