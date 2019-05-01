import copy
import os
from collections import OrderedDict

import six
import tensorflow as tf

from tfsnippet.utils import (VarScopeObject, add_name_and_scope_arg_doc,
                             reopen_variable_scope, makedirs,
                             get_default_session_or_error)
from .scheduled_var import ScheduledVariable

if six.PY2:
    import cPickle as pkl
else:
    import pickle as pkl

__all__ = ['CheckpointSavableObject', 'CheckpointSaver']

CHECKPOINT_VAR_NAME = 'tfsnippet_checkpoint_pickle_variable_' \
                      'd2a4b5a2c0ca48b9855bce2953bc11d5'


class CheckpointSavableObject(object):
    """
    Base class for all objects that can be saved via :class:`CheckpointSaver`.
    """

    def get_state(self):
        """
        Get the internal states of the object.

        The returned state dict must be pickle-able.

        Returns:
            dict: The internal states dict.
        """
        raise NotImplementedError()

    def set_state(self, state):
        """
        Set the internal states of the object.

        Args:
            state: The internal states dict.
        """
        raise NotImplementedError()


class CheckpointSerialVar(object):

    def __init__(self):
        self._variable = tf.get_variable(
            CHECKPOINT_VAR_NAME, initializer='', dtype=tf.string,
            trainable=False, collections=[]
        )
        self._read_op = tf.convert_to_tensor(self._variable)
        self._assign_ph = tf.placeholder(dtype=tf.string, shape=())
        self._assign_op = tf.assign(self._variable, self._assign_ph)

    @property
    def variable(self):
        return self._variable

    def get(self, session=None):
        session = session or get_default_session_or_error()
        return session.run(self._read_op)

    def set(self, value, session=None):
        session = session or get_default_session_or_error()
        session.run(self._assign_op, feed_dict={self._assign_ph: value})


class CheckpointSaver(VarScopeObject):
    """
    Save and restore :class:`tf.Variable`, :class:`ScheduledVariable` and
    :class:`CheckpointSavableObject` with :class:`tf.train.Saver`.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, variables, save_dir, objects=None,
                 filename='checkpoint.dat', max_to_keep=None, save_meta=True,
                 name=None, scope=None):
        """
        Construct a new :class:`CheckpointSaver`.

        Args:
            variables: A list of variables, or a dict `(name -> variable)`.
                A variable might be a :class:`tf.Variable` or a
                :class:`ScheduledVariable`.
            save_dir (str): The directory, where to place the checkpoint files.
                This directory must be solely owned by this saver.
            objects (dict[str, CheckpointSavableObject]): A dict
                `(name -> savable object)`.
            filename (str): Name of the checkpoint files.
            max_to_keep (int or None): Maximum number of versions to keep.
                If :obj:`None` or `0`, keep all versions.
            save_meta (bool): Whether or not to save the graph meta in
                 checkpoint files?
        """
        # check the argument `variables`
        def check_var(var):
            if not isinstance(var, (tf.Variable, ScheduledVariable)):
                raise TypeError('Not a variable: {!r}'.format(var))
            if isinstance(var, ScheduledVariable):
                var = var.variable
            return var

        def normalize_var_name(var):
            name = var.name
            if name.endswith(':0'):
                name = name[:-2]
            return name

        if isinstance(variables, (dict, OrderedDict)):
            variables = {
                k: check_var(v)
                for k, v in six.iteritems(variables)
            }
        else:
            variables = {
                normalize_var_name(v): v
                for v in map(check_var, variables)
            }
        if CHECKPOINT_VAR_NAME in variables:
            raise KeyError('Name is reserved for `variables`: {}'.
                           format(CHECKPOINT_VAR_NAME))

        # check the arguments `objects`
        def check_obj(obj):
            if not isinstance(obj, CheckpointSavableObject):
                raise TypeError('Not a savable object: {!r}'.format(obj))
            return obj

        objects = {k: check_obj(v) for k, v in six.iteritems(objects or {})}
        if CHECKPOINT_VAR_NAME in objects:
            raise KeyError('Name is reserved for `objects`: {}'.
                           format(CHECKPOINT_VAR_NAME))

        self._variables = variables
        self._objects = objects
        self._save_dir = os.path.abspath(save_dir)
        self._filename = str(filename)
        self._save_meta = bool(save_meta)

        super(CheckpointSaver, self).__init__(name=name, scope=scope)

        with reopen_variable_scope(self.variable_scope):
            # build the variable for serialization
            self._serial_var = None
            if self._objects:
                self._serial_var = CheckpointSerialVar()

            # add the serial var to var_dict
            var_dict = copy.copy(variables)
            if self._objects:
                var_dict[CHECKPOINT_VAR_NAME] = self._serial_var.variable
            self._var_dict = var_dict

            # now build the saver
            self._saver = tf.train.Saver(
                var_list=var_dict,
                max_to_keep=max_to_keep
            )

        # recover the internal states
        self.recover_internal_states()

    @property
    def save_dir(self):
        """Get the checkpoint directory."""
        return self._save_dir

    @property
    def filename(self):
        """Get the filename of checkpoint files."""
        return self._filename

    @property
    def save_meta(self):
        """Whether or not to save graph meta?"""
        return self._save_meta

    @property
    def saver(self):
        """
        Get the TensorFlow saver object.

        Returns:
            tf.train.Saver: The TensorFlow saver object.
        """
        return self._saver

    def recover_internal_states(self):
        """Restore the internal states of this saver."""
        checkpoint_state = tf.train.get_checkpoint_state(self._save_dir)
        if checkpoint_state is not None:
            self._saver.recover_last_checkpoints(
                checkpoint_state.all_model_checkpoint_paths)

    def latest_checkpoint(self):
        """
        Get the path of the latest checkpoint file.

        Returns:
            str or None: The path of the latest checkpoint file, or
                :obj:`None` if no checkpoint file is found.
        """
        return tf.train.latest_checkpoint(self._save_dir)

    def restore_latest(self, ignore_non_exist=False, session=None):
        """
        Restore the latest checkpoint file.
        Args:
            ignore_non_exist (bool): Whether or not to ignore error if the
                latest checkpoint file does not exist?
            session (tf.Session): Restore the variables into this session.
                If not specified, restore into the default session.

        Raises:
            IOError: If no checkpoint file is found.
        """
        latest_checkpoint = self.latest_checkpoint()
        if latest_checkpoint is None:
            if not ignore_non_exist:
                raise IOError('No checkpoint file is found.')
        else:
            self.restore(latest_checkpoint, session=session)

    def restore(self, save_path, session=None):
        """
        Restore from a checkpoint file.

        Args:
            save_path (str): Restore from this checkpoint file.
            session (tf.Session): Restore the variables into this session.
                If not specified, restore into the default session.
        """
        session = session or get_default_session_or_error()

        # restore the variables
        self._saver.restore(session, save_path)

        # restore the states of savable objects
        if self._objects:
            object_states = pkl.loads(self._serial_var.get(session))
            assert(isinstance(object_states, dict))

            for key, obj in six.iteritems(self._objects):
                if key not in object_states:
                    raise KeyError('Object `{}` not found in the checkpoint: '
                                   '{}'.format(key, save_path))
                obj.set_state(object_states[key])

    def save(self, global_step=None, session=None):
        """
        Save the session to a checkpoint file.

        Args:
            global_step (int or tf.Tensor): The global step counter.
            session (tf.Session): The session to save.
                If not specified, select the default session.

        Returns:
            str: The path of the saved checkpoint file.
        """
        session = session or get_default_session_or_error()

        # save the states of savable objects into serial var
        if self._objects:
            object_states = {}
            for key, obj in six.iteritems(self._objects):
                object_states[key] = obj.get_state()

            serialized_states = pkl.dumps(
                object_states, protocol=pkl.HIGHEST_PROTOCOL)
            self._serial_var.set(serialized_states)

        # now save the variables to checkpoint file
        if not os.path.isdir(self.save_dir):
            makedirs(self.save_dir, exist_ok=True)
        return self._saver.save(
            session,
            os.path.join(self.save_dir, self.filename),
            global_step=global_step,
            write_meta_graph=self.save_meta
        )
