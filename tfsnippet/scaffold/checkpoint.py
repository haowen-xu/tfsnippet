import os
from collections import OrderedDict, namedtuple

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

__all__ = ['CheckpointSaver']


CheckpointSaverSerialVar = namedtuple(
    'CheckpointSaverSerialVar',
    ['variable', 'read_op', 'assign_op', 'assign_ph']
)


class CheckpointSaver(VarScopeObject):
    """
    Save and restore :class:`tf.Variable` and :class:`CheckpointSavableObject`
    with :class:`tf.train.Saver`.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, objects, save_dir, filename='checkpoint.dat',
                 max_to_keep=None, save_meta=True, name=None, scope=None):
        """
        Construct a new :class:`CheckpointSaver`.

        Args:
            objects: A list of savable objects, or a dict of
                `(name -> savable object)`.  A :class:`tf.Variable` is savable,
                a :class:`ScheduledVariable` is savable, and any object that
                supports the pickle protocol is savable.
            save_dir (str): The directory, where to place the checkpoint files.
                This directory must be solely owned by this saver.
            filename (str): Name of the checkpoint files.
            max_to_keep (int or None): Maximum number of versions to keep.
                If :obj:`None` or `0`, keep all versions.
            save_meta (bool): Whether or not to save the graph meta in
                 checkpoint files?
        """
        def check_obj(obj):
            if not isinstance(obj, (tf.Variable, ScheduledVariable)) and \
                    (not hasattr(obj, '__getstate__') or
                     not hasattr(obj, '__setstate__')):
                raise TypeError('{!r} is not a savable object.'.format(obj))

        if isinstance(objects, (dict, OrderedDict)):
            objects = dict(objects)
            for key, value in six.iteritems(objects):
                check_obj(value)
        else:
            objects = list(objects)
            for value in objects:
                check_obj(value)

        self._save_dir = os.path.abspath(save_dir)
        self._filename = str(filename)
        self._save_meta = bool(save_meta)

        super(CheckpointSaver, self).__init__(name=name, scope=scope)

        with reopen_variable_scope(self.variable_scope):
            serial_dict = {}  # type: dict[CheckpointSavableObject, CheckpointSaverSerialVar]

            # dict[str, tf.Variable]: the variable dict
            var_dict = {}

            # build the above two dict from `objects`
            def build_var_op_ph(obj, var_idx):
                var_name = 'serial_var_{}'.format(var_idx)
                serial_var = tf.get_variable(
                    name=var_name, initializer='', dtype=tf.string,
                    trainable=False, collections=[]
                )
                read_op = tf.identity(serial_var,
                                      name='read_{}'.format(var_idx))
                assign_ph = tf.placeholder(dtype=tf.string, shape=(),
                                           name='assign_ph_{}'.format(var_idx))
                assign_op = tf.assign(serial_var, assign_ph,
                                      name='assign_op_{}'.format(var_idx))
                serial_dict[obj] = CheckpointSaverSerialVar(
                    variable=serial_var, read_op=read_op, assign_op=assign_op,
                    assign_ph=assign_ph
                )
                return serial_var

            def normalize_var_name(var):
                name = var.name
                if name.endswith(':0'):
                    name = name[:-2]
                return name

            if isinstance(objects, dict):
                for key, value in six.iteritems(objects):
                    if isinstance(value, ScheduledVariable):
                        var_dict[key] = value.variable
                    elif isinstance(value, tf.Variable):
                        var_dict[key] = value
                    else:
                        var_idx = len(serial_dict)
                        var_dict[key] = build_var_op_ph(value, var_idx)
            else:
                for value in objects:
                    if isinstance(value, ScheduledVariable):
                        key = 'vars/' + normalize_var_name(value.variable)
                        var_dict[key] = value.variable
                    elif isinstance(value, tf.Variable):
                        key = 'vars/' + normalize_var_name(value)
                        var_dict[key] = value
                    else:
                        var_idx = len(serial_dict)
                        key = 'serial_var_{}'.format(var_idx)
                        var_dict[key] = build_var_op_ph(value, var_idx)

            self._serial_dict = serial_dict
            self._var_dict = var_dict

            # now build the saver
            self._saver = tf.train.Saver(
                var_list=self._var_dict,
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
        if session is None:
            session = get_default_session_or_error()

        # restore the variables
        self._saver.restore(session, save_path)

        # restore the states of savable objects
        objects = list(self._serial_dict)
        obj_values = session.run(
            [self._serial_dict[o].read_op for o in objects])
        for obj, val in zip(objects, obj_values):
            obj.__setstate__(pkl.loads(val))

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
        if session is None:
            session = get_default_session_or_error()

        # save the states of savable objects into variables
        objects = list(self._serial_dict)
        op_list, feed_dict = [], {}

        for obj in objects:
            obj_serial = self._serial_dict[obj]
            op_list.append(obj_serial.assign_op)
            feed_dict[obj_serial.assign_ph] = pkl.dumps(
                obj.__getstate__(), protocol=pkl.HIGHEST_PROTOCOL)

        session.run(op_list, feed_dict=feed_dict)

        # now save the variables to checkpoint file
        if not os.path.isdir(self.save_dir):
            makedirs(self.save_dir, exist_ok=True)
        return self._saver.save(
            session,
            os.path.join(self.save_dir, self.filename),
            global_step=global_step,
            write_meta_graph=self.save_meta
        )
