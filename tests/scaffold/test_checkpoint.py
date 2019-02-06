import os

import pytest
import tensorflow as tf
from mock import Mock

from tfsnippet.scaffold import *
from tfsnippet.scaffold.checkpoint import CHECKPOINT_VAR_NAME
from tfsnippet.utils import ensure_variables_initialized, TemporaryDirectory


class CheckpointSaverTestCase(tf.test.TestCase):

    def test_constructor(self):
        with TemporaryDirectory() as tmpdir:
            v1 = tf.get_variable('v1', dtype=tf.int32, shape=())
            with tf.variable_scope('parent'):
                v2 = tf.get_variable('v2', dtype=tf.int32, shape=())
            sv = ScheduledVariable('sv', dtype=tf.float32, initial_value=123)
            obj = Mock(
                spec=CheckpointSavableObject,
                get_state=Mock(return_value={'value': 123}),
                set_state=Mock()
            )
            obj2 = Mock(
                spec=CheckpointSavableObject,
                get_state=Mock(return_value={'value': 456}),
                set_state=Mock()
            )

            saver = CheckpointSaver([v1, sv, v2], tmpdir + '/1',
                                    objects={'obj': obj, 'obj2': obj2})
            self.assertEqual(saver.save_dir, tmpdir + '/1')
            self.assertIsNone(saver._saver._max_to_keep)
            self.assertTrue(saver.save_meta)
            self.assertDictEqual(
                saver._var_dict,
                {'v1': v1, 'parent/v2': v2, 'sv': sv.variable,
                 CHECKPOINT_VAR_NAME: saver._serial_var.variable}
            )
            self.assertIsInstance(saver.saver, tf.train.Saver)

            saver = CheckpointSaver(
                {'vv1': v1, 'v22': v2, 'svv': sv},
                tmpdir + '/2',
                objects={'oobj': obj, 'obj2': obj2},
                filename='variables.dat',
                max_to_keep=3, save_meta=False
            )
            self.assertEqual(saver.save_dir, tmpdir + '/2')
            self.assertEqual(saver._saver._max_to_keep, 3)
            self.assertFalse(saver.save_meta)
            self.assertDictEqual(
                saver._var_dict,
                {'vv1': v1, 'v22': v2, 'svv': sv.variable,
                 CHECKPOINT_VAR_NAME: saver._serial_var.variable}
            )
            self.assertIsInstance(saver.saver, tf.train.Saver)

            with pytest.raises(TypeError, match='Not a variable'):
                _ = CheckpointSaver([object()], tmpdir)
            with pytest.raises(TypeError, match='Not a variable'):
                _ = CheckpointSaver([tf.constant(123.)], tmpdir)

            with pytest.raises(TypeError, match='Not a savable object'):
                _ = CheckpointSaver([], tmpdir, {'obj': object()})
            with pytest.raises(TypeError, match='Not a savable object'):
                _ = CheckpointSaver([], tmpdir, {'obj': tf.constant(0.)})

            with pytest.raises(KeyError,
                               match='Name is reserved for `variables`'):
                _ = CheckpointSaver(
                    [tf.get_variable(CHECKPOINT_VAR_NAME, dtype=tf.int32,
                                     initializer=0)],
                    tmpdir
                )
            with pytest.raises(KeyError,
                               match='Name is reserved for `variables`'):
                _ = CheckpointSaver(
                    {CHECKPOINT_VAR_NAME: tf.get_variable(
                        'a', dtype=tf.int32, initializer=0)},
                    tmpdir
                )

            with pytest.raises(KeyError,
                               match='Name is reserved for `objects`'):
                _ = CheckpointSaver([], tmpdir, {CHECKPOINT_VAR_NAME: obj})

    def test_save_restore(self):
        class MyObject(CheckpointSavableObject):
            def __init__(self, value):
                self.value = value

            def get_state(self):
                return self.__dict__

            def set_state(self, state):
                keys = list(self.__dict__)
                for k in keys:
                    if k not in state:
                        self.__dict__.pop(k)
                for k in state:
                    self.__dict__[k] = state[k]

        with TemporaryDirectory() as tmpdir, \
                self.test_session() as sess:
            save_dir = os.path.join(tmpdir, 'saves')
            v = tf.get_variable('v', dtype=tf.int32, initializer=12)
            sv = ScheduledVariable('sv', dtype=tf.float32, initial_value=34)
            obj = MyObject(56)
            obj2 = MyObject(90)
            ensure_variables_initialized()

            # test construct a saver upon empty directory
            saver = CheckpointSaver([v, sv], save_dir,
                                    objects={'obj': obj, 'obj2': obj2})
            self.assertIsNone(saver.latest_checkpoint())

            with pytest.raises(IOError, match='No checkpoint file is found'):
                saver.restore_latest()
            saver.restore_latest(ignore_non_exist=True, session=sess)

            # save the first checkpoint
            ckpt_0 = saver.save(0)
            self.assertEqual(saver.latest_checkpoint(), ckpt_0)

            # now we change the states
            sess.run(tf.assign(v, 1212))
            sv.set(3434)
            obj.value = 5656
            obj.value2 = 7878
            obj2.value = 9090
            ckpt_1 = saver.save(1, session=sess)
            self.assertEqual(saver.latest_checkpoint(), ckpt_1)

            # construct a saver on existing checkpoint directory
            saver = CheckpointSaver([v, sv], save_dir,
                                    objects={'obj': obj, 'obj2': obj2})
            self.assertEqual(saver.latest_checkpoint(), ckpt_1)

            # restore the latest checkpoint
            saver.restore_latest()
            self.assertListEqual(sess.run([v, sv]), [1212, 3434])
            self.assertEqual(obj.value, 5656)
            self.assertEqual(obj.value2, 7878)
            self.assertEqual(obj2.value, 9090)

            # restore a previous checkpoint
            saver.restore(ckpt_0, sess)
            self.assertListEqual(sess.run([v, sv]), [12, 34])
            self.assertEqual(obj.value, 56)
            self.assertFalse(hasattr(obj, 'value2'))
            self.assertEqual(obj2.value, 90)

            # try to restore only a partial of the variables and objects
            saver = CheckpointSaver([v], save_dir, objects={'obj': obj})
            saver.restore_latest()
            self.assertListEqual(sess.run([v, sv]), [1212, 34])
            self.assertEqual(obj.value, 5656)
            self.assertEqual(obj.value2, 7878)
            self.assertEqual(obj2.value, 90)

            # try to restore a non-exist object
            saver = CheckpointSaver([v], save_dir, objects={'obj3': obj})
            with pytest.raises(KeyError, match='Object `obj3` not found in the '
                                               'checkpoint'):
                saver.restore_latest()
