import os
from tempfile import TemporaryDirectory

import pytest
import tensorflow as tf
from mock import Mock

from tfsnippet.scaffold import *
from tfsnippet.utils import ensure_variables_initialized


class CheckpointSaverTestCase(tf.test.TestCase):

    def test_constructor(self):
        with TemporaryDirectory() as tmpdir:
            v1 = tf.get_variable('v1', dtype=tf.int32, shape=())
            with tf.variable_scope('parent'):
                v2 = tf.get_variable('v2', dtype=tf.int32, shape=())
            sv = ScheduledVariable('sv', dtype=tf.float32, initial_value=123)
            obj = Mock(
                __getstate__=Mock(return_value={'value': 123}),
                __setstate__=Mock()
            )
            obj2 = Mock(
                __getstate__=Mock(return_value={'value': 456}),
                __setstate__=Mock()
            )

            saver = CheckpointSaver([v1, obj, sv, obj2, v2], tmpdir + '/1')
            self.assertEqual(saver.save_dir, tmpdir + '/1')
            self.assertIsNone(saver._saver._max_to_keep)
            self.assertTrue(saver.save_meta)
            self.assertDictEqual(
                saver._var_dict,
                {'vars/v1': v1, 'vars/parent/v2': v2, 'vars/sv': sv.variable,
                 'serial_var_0': saver._serial_dict[obj].variable,
                 'serial_var_1': saver._serial_dict[obj2].variable}
            )
            self.assertIsInstance(saver.saver, tf.train.Saver)

            saver = CheckpointSaver(
                {'vv1': v1, 'v22': v2, 'svv': sv, 'oobj': obj, 'obj2': obj2},
                tmpdir + '/2', 'variables.dat', max_to_keep=3, save_meta=False
            )
            self.assertEqual(saver.save_dir, tmpdir + '/2')
            self.assertEqual(saver._saver._max_to_keep, 3)
            self.assertFalse(saver.save_meta)
            self.assertDictEqual(
                saver._var_dict,
                {'vv1': v1, 'v22': v2, 'svv': sv.variable,
                 'oobj': saver._serial_dict[obj].variable,
                 'obj2': saver._serial_dict[obj2].variable}
            )
            self.assertIsInstance(saver.saver, tf.train.Saver)

            with pytest.raises(TypeError, match='is not a savable object'):
                _ = CheckpointSaver(
                    [Mock(__getstate__=Mock(return_value={'value': 123}))],
                    tmpdir
                )
            with pytest.raises(TypeError, match='is not a savable object'):
                _ = CheckpointSaver({'obj': Mock(__setstate__=Mock())}, tmpdir)
            with pytest.raises(TypeError, match='is not a savable object'):
                _ = CheckpointSaver([tf.constant(123.)], tmpdir)

    def test_save_restore(self):
        class MyObject(object):
            def __init__(self, value):
                self.value = value

            def __getstate__(self):
                return self.__dict__

            def __setstate__(self, state):
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
            ensure_variables_initialized()

            # test construct a saver upon empty directory
            saver = CheckpointSaver([v, sv, obj], save_dir)
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
            ckpt_1 = saver.save(1, session=sess)
            self.assertEqual(saver.latest_checkpoint(), ckpt_1)

            # construct a saver on existing checkpoint directory
            saver = CheckpointSaver([v, sv, obj], save_dir)
            self.assertEqual(saver.latest_checkpoint(), ckpt_1)

            # restore the latest checkpoint
            saver.restore_latest()
            self.assertListEqual(sess.run([v, sv]), [1212, 3434])
            self.assertEqual(obj.value, 5656)
            self.assertEqual(obj.value2, 7878)

            # restore a previous checkpoint
            saver.restore(ckpt_0, sess)
            self.assertListEqual(sess.run([v, sv]), [12, 34])
            self.assertEqual(obj.value, 56)
            self.assertFalse(hasattr(obj, 'value2'))
