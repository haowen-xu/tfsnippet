# -*- coding: utf-8 -*-
import os
import re
import time
from collections import OrderedDict

import pytest
import numpy as np
import tensorflow as tf
from mock import Mock

from tfsnippet.dataflows import DataFlow
from tfsnippet.scaffold import (TrainLoop, CheckpointSavableObject,
                                ScheduledVariable)
from tfsnippet.scaffold.train_loop_ import (TRAIN_LOOP_STATES_CKPT_NAME,
                                            EARLY_STOPPING_STATES_CKPT_NAME)
from tfsnippet.utils import (TemporaryDirectory,
                             ensure_variables_initialized,
                             get_default_session_or_error)


def get_variable_values(variables):
    sess = get_default_session_or_error()
    return sess.run(variables)


def set_variable_values(variables, values):
    sess = get_default_session_or_error()
    sess.run([tf.assign(v, a) for v, a in zip(variables, values)])


class TrainLoopTestCase(tf.test.TestCase):

    def assertMatches(self, a, b):
        self.assertTrue(
            not not re.match(b, a),
            msg='{!r} should match {!r}'.format(b, a)
        )

    def test_counter_attributes(self):
        with TrainLoop([]) as loop:
            self.assertEqual(loop.epoch, 0)
            self.assertEqual(loop.step, 0)
            self.assertIsNone(loop.max_epoch)
            self.assertIsNone(loop.max_step)
            self.assertFalse(loop.within_epoch)
            self.assertFalse(loop.within_step)

        with TrainLoop([], max_epoch=2, max_step=10,
                       summary_metric_prefix='123/') as loop:
            self.assertEqual(loop.max_epoch, 2)
            self.assertEqual(loop.max_step, 10)
            self.assertEqual(loop._summary_metric_prefix, '123/')
            loop.max_epoch = 20
            loop.max_step = 100
            self.assertEqual(loop.max_epoch, 20)
            self.assertEqual(loop.max_step, 100)

    def test_counters(self):
        # test loop with configured `max_epoch`
        with TrainLoop([], max_epoch=2) as loop:
            epoch_counter = 0
            step_counter = 0
            for epoch in loop.iter_epochs():
                epoch_counter += 1
                self.assertEqual(epoch, epoch_counter)
                self.assertTrue(loop.within_epoch)
                self.assertFalse(loop.within_step)
                x_ans = 0
                for step, [x] in \
                        loop.iter_steps(DataFlow.arrays([np.arange(4)], 1)):
                    self.assertTrue(loop.within_step)
                    self.assertEqual(step, loop.step)
                    self.assertEqual(epoch, loop.epoch)
                    self.assertEqual(x, x_ans)
                    self.assertIsInstance(loop.step_data, tuple)
                    self.assertEqual(len(loop.step_data), 1)
                    np.testing.assert_equal(loop.step_data[0], x_ans)
                    x_ans += 1
                    step_counter += 1
                    self.assertEqual(step, step_counter)
                self.assertEqual(step_counter, loop.step)
                self.assertEqual(epoch, loop.epoch)
            self.assertEqual(epoch_counter, 2)
            self.assertEqual(step_counter, 8)

        # test loop with configured `max_step`
        with TrainLoop([], max_step=10) as loop:
            epoch_counter = 0
            step_counter = 0
            for epoch in loop.iter_epochs():
                epoch_counter += 1
                self.assertEqual(epoch, epoch_counter)
                for step in loop.iter_steps():
                    step_counter += 1
                    self.assertEqual(step, step_counter)
            self.assertEqual(epoch_counter, 1)
            self.assertEqual(step_counter, 10)

        # test loop with configured `max_step` with payload
        with TrainLoop([], max_step=10) as loop:
            epoch_counter = 0
            step_counter = 0
            for epoch in loop.iter_epochs():
                epoch_counter += 1
                self.assertEqual(epoch, epoch_counter)
                x_ans = 0
                for step, x in loop.iter_steps(np.arange(4)):
                    self.assertEqual(x, x_ans)
                    x_ans += 1
                    step_counter += 1
                    self.assertEqual(step, step_counter)
            self.assertEqual(epoch_counter, 3)
            self.assertEqual(step_counter, 10)

        # test loop with configured `max_step` and `max_epoch`,
        # while `max_epoch` finishes first
        with TrainLoop([], max_step=10, max_epoch=2) as loop:
            epoch_counter = 0
            step_counter = 0
            for epoch in loop.iter_epochs():
                epoch_counter += 1
                self.assertEqual(epoch, epoch_counter)
                for step, _ in loop.iter_steps(np.arange(4)):
                    step_counter += 1
                    self.assertEqual(step, step_counter)
            self.assertEqual(epoch_counter, 2)
            self.assertEqual(step_counter, 8)

        # test loop with configured `max_step` and `max_epoch`,
        # while `max_step` finishes first
        with TrainLoop([], max_step=10, max_epoch=3) as loop:
            epoch_counter = 0
            step_counter = 0
            for epoch in loop.iter_epochs():
                epoch_counter += 1
                self.assertEqual(epoch, epoch_counter)
                for step, _ in loop.iter_steps(np.arange(4)):
                    step_counter += 1
                    self.assertEqual(step, step_counter)
            self.assertEqual(epoch_counter, 3)
            self.assertEqual(step_counter, 10)

    def test_get_progress(self):
        null_print = lambda x: None

        # test no progress
        with TrainLoop([], max_epoch=None, max_step=None) as loop:
            self.assertIsNone(loop.get_progress())

        # test infer progress from epoch
        with TrainLoop([], max_epoch=10, max_step=None,
                       print_func=null_print) as loop:
            np.testing.assert_allclose(0., loop.get_progress())
            for i in loop.iter_epochs():
                np.testing.assert_allclose((i - 1) * .1, loop.get_progress())
                loop.print_logs()
                np.testing.assert_allclose(i * .1, loop.get_progress())
            np.testing.assert_allclose(1., loop.get_progress())

        # test infer progress from step
        with TrainLoop([], max_epoch=None, max_step=100,
                       print_func=null_print) as loop:
            np.testing.assert_allclose(0., loop.get_progress())
            for _ in loop.iter_epochs():
                for _ in loop.iter_steps([0, 1, 2]):
                    step = loop.step
                    np.testing.assert_allclose(
                        (step - 1) * .01, loop.get_progress())
                    loop.print_logs()
                    np.testing.assert_allclose(
                        step * .01, loop.get_progress())
            np.testing.assert_allclose(1., loop.get_progress())

        # test infer progress from epoch & steps_per_epoch
        with TrainLoop([], max_epoch=10, print_func=null_print) as loop:
            np.testing.assert_allclose(0., loop.get_progress())
            for i in loop.iter_epochs():
                np.testing.assert_allclose((i - 1) * .1, loop.get_progress())
                for _, j in loop.iter_steps([0, 1, 2, 3, 4]):
                    if i == 1:
                        np.testing.assert_allclose(0., loop.get_progress())
                        loop.print_logs()
                        np.testing.assert_allclose(0., loop.get_progress())
                    else:
                        np.testing.assert_allclose(
                            (i - 1) * .1 + j * .02, loop.get_progress())
                        loop.print_logs()
                        np.testing.assert_allclose(
                            (i - 1) * .1 + (j + 1) * .02, loop.get_progress())
                if i == 1:
                    np.testing.assert_allclose(0., loop.get_progress())
                    loop.print_logs()
                    np.testing.assert_allclose(.1, loop.get_progress())
                else:
                    np.testing.assert_allclose(i * .1, loop.get_progress())
                    loop.print_logs()
                    np.testing.assert_allclose(i * .1, loop.get_progress())
            np.testing.assert_allclose(1., loop.get_progress())

    def test_logs(self):
        logs = []
        with TrainLoop([], max_step=6, print_func=logs.append) as loop:
            for epoch in loop.iter_epochs():
                for step, x in loop.iter_steps(np.arange(4)):
                    time.sleep(0.01)
                    loop.collect_metrics(x=x)
                    if step % 2 == 0:
                        loop.print_logs()
                loop.collect_metrics(y=epoch)
                loop.print_logs()
        self.assertMatches('\n'.join(logs), re.compile(
            r'^'
            r'\[Epoch 1, Step 2/6, ETA \S+\] step time: 0\.01\d*s \(±[^ ]+s\); '
            r'x: 0\.5 \(±0\.5\)\n'
            r'\[Epoch 1, Step 4/6, ETA \S+\] step time: 0\.01\d*s \(±[^ ]+s\); '
            r'x: 2\.5 \(±0\.5\)\n'
            r'\[Epoch 1, Step 4/6, ETA \S+\] epoch time: 0\.0[456]\d*s; '
            r'step time: 0\.01\d*s \(±[^ ]+s\); x: 1\.5 \(±1\.11803\); '
            r'y: 1\n'
            r'\[Epoch 2, Step 6/6, ETA \S+\] step time: 0\.01\d*s \(±[^ ]+s\); '
            r'x: 0\.5 \(±0\.5\)\n'
            r'\[Epoch 2, Step 6/6, ETA \S+\] epoch time: 0\.0[23]\d*s; '
            r'step time: 0\.01\d*s \(±[^ ]+s\); x: 0\.5 \(±0\.5\); y: 2'
            r'$'
        ))

    def test_single_epoch_logs(self):
        logs = []
        with TrainLoop([], max_epoch=1, print_func=logs.append,
                       show_eta=False) as loop:
            for epoch in loop.iter_epochs():
                for step, x in loop.iter_steps(np.arange(4)):
                    time.sleep(0.01)
                    loop.collect_metrics(x=x)
                    if step % 2 == 0:
                        loop.print_logs()
                loop.collect_metrics(y=epoch)
                loop.print_logs()
        self.assertMatches('\n'.join(logs), re.compile(
            r'^'
            r'\[Step 2\] step time: 0\.01\d*s \(±[^ ]+s\); '
            r'x: 0\.5 \(±0\.5\)\n'
            r'\[Step 4\] step time: 0\.01\d*s \(±[^ ]+s\); '
            r'x: 2\.5 \(±0\.5\)\n'
            r'\[Step 4\] epoch time: 0\.0[456]\d*s; '
            r'step time: 0\.01\d*s \(±[^ ]+s\); x: 1\.5 \(±1\.11803\); '
            r'y: 1'
            r'$'
        ))

    def test_valid_metric_default_settings(self):
        logs = []
        with TrainLoop([], print_func=logs.append, show_eta=False) as loop:
            self.assertEqual(loop.valid_metric_name, 'valid_loss')
            self.assertTrue(loop.valid_metric_smaller_is_better)
            self.assertFalse(loop.use_early_stopping)
            for _ in loop.iter_epochs():
                best_metric = 1.
                for _, valid_loss in loop.iter_steps([0.8, 0.6, 0.7]):
                    loop.collect_metrics(valid_loss=valid_loss)
                    best_metric = min(best_metric, valid_loss)
                    self.assertAlmostEqual(loop.best_valid_metric, best_metric)
                    loop.print_logs()
                loop.print_logs()
                break
        self.assertAlmostEqual(loop.best_valid_metric, 0.6)
        self.assertMatches('\n'.join(logs), re.compile(
            r'^'
            r'\[Epoch 1, Step 1\] step time: [^ ]+s; '
            r'valid loss: 0\.8 \(\*\)\n'
            r'\[Epoch 1, Step 2\] step time: [^ ]+s; '
            r'valid loss: 0\.6 \(\*\)\n'
            r'\[Epoch 1, Step 3\] step time: [^ ]+s; '
            r'valid loss: 0\.7\n'
            r'\[Epoch 1, Step 3\] epoch time: [^ ]+s; step time: [^ ]+s '
            r'\(±[^ ]+s\); valid loss: 0\.7 \(±0\.0816497\)'
            r'$'
        ))

    def test_valid_metric_with_custom_settings(self):
        logs = []
        v = tf.get_variable('a', shape=[1], dtype=tf.int32)
        with TrainLoop([v], print_func=logs.append, show_eta=False,
                       valid_metric_name='y',
                       valid_metric_smaller_is_better=False) as loop:
            self.assertEqual(loop.valid_metric_name, 'y')
            self.assertFalse(loop.valid_metric_smaller_is_better)
            for _ in loop.iter_epochs():
                best_metric = 0.
                for _, y in loop.iter_steps([0.7, 0.6, 0.8]):
                    loop.collect_metrics(y=y)
                    best_metric = max(best_metric, y)
                    self.assertAlmostEqual(loop.best_valid_metric, best_metric)
                    loop.print_logs()
                loop.print_logs()
                break
        self.assertAlmostEqual(loop.best_valid_metric, 0.8)
        self.assertMatches('\n'.join(logs), re.compile(
            r'^'
            r'\[Epoch 1, Step 1\] step time: [^ ]+s; '
            r'y: 0\.7 \(\*\)\n'
            r'\[Epoch 1, Step 2\] step time: [^ ]+s; '
            r'y: 0\.6\n'
            r'\[Epoch 1, Step 3\] step time: [^ ]+s; '
            r'y: 0\.8 \(\*\)\n'
            r'\[Epoch 1, Step 3\] epoch time: [^ ]+s; step time: [^ ]+s '
            r'\(±[^ ]+s\); y: 0\.7 \(±0\.0816497\)'
            r'$'
        ))

    def test_valid_metric_with_valid_acc(self):
        with TrainLoop([], valid_metric_name='valid_acc') as loop:
            self.assertEqual(loop.valid_metric_name, 'valid_acc')
            self.assertFalse(loop.valid_metric_smaller_is_better)

    def test_valid_metric_with_y_as_name(self):
        with TrainLoop([], valid_metric_name='y') as loop:
            self.assertEqual(loop.valid_metric_name, 'y')
            self.assertTrue(loop.valid_metric_smaller_is_better)

    def test_training_summary(self):
        a = tf.get_variable('a', dtype=tf.float32, shape=(2, 3))
        b = tf.get_variable('b', dtype=tf.float32, shape=(4,))

        # test param variables in list
        logs = []
        with TrainLoop([a, b], print_func=logs.append) as loop:
            self.assertEqual(loop.param_vars, [a, b])
            loop.print_training_summary()
        self.assertEqual('\n'.join(logs), (
            'Trainable Parameters (10 in total)\n'
            '----------------------------------\n'
            'a                        (2, 3)  6\n'
            'b                        (4,)    4\n'
        ))

        # test param variables in dict
        logs = []
        with TrainLoop(OrderedDict([('aa', a), ('bb', b)]),
                       print_func=logs.append) as loop:
            self.assertEqual(loop.param_vars, {'aa': a, 'bb': b})
            loop.print_training_summary()
        self.assertEqual('\n'.join(logs), (
            'Trainable Parameters (10 in total)\n'
            '----------------------------------\n'
            'aa                       (2, 3)  6\n'
            'bb                       (4,)    4\n'
        ))

    def test_timeit(self):
        logs = []
        with TrainLoop([], max_epoch=1, print_func=logs.append,
                       show_eta=False) as loop:
            for _ in loop.iter_epochs():
                with loop.timeit('x_timer'):
                    time.sleep(0.01)
                with loop.timeit('y_time'):
                    time.sleep(0.02)
                loop.print_logs()
        self.assertMatches('\n'.join(logs), re.compile(
            r'^'
            r'\[Step 0\] epoch time: 0\.0[345]\d*s; '
            r'x timer: 0\.01\d*s; y time: 0\.0[23]\d*s'
            r'$'
        ))

    def test_metric_collector(self):
        logs = []
        with TrainLoop([], max_epoch=1, print_func=logs.append,
                       show_eta=False) as loop:
            for _ in loop.iter_epochs():
                with loop.metric_collector('x') as acc:
                    acc.collect(2)
                    acc.collect(3, weight=3)
                loop.print_logs()
        self.assertMatches('\n'.join(logs), re.compile(
            r'^'
            r'\[Step 0\] epoch time: [^ ]+s; x: 2\.75'
            r'$'
        ))

    def test_summary_writer(self):
        def read_summary(summary_dir):
            # read the metric summary
            loss_steps = []
            loss_values = []
            valid_loss_steps = []
            valid_loss_values = []
            x_steps = []
            x_values = []
            tags = set()

            event_file_path = os.path.join(
                summary_dir, os.listdir(summary_dir)[0])
            for e in tf.train.summary_iterator(event_file_path):
                for v in e.summary.value:
                    tags.add(v.tag)
                    if v.tag == 'metrics/loss':
                        loss_steps.append(e.step)
                        loss_values.append(v.simple_value)
                    elif v.tag == 'metrics/valid_loss':
                        valid_loss_steps.append(e.step)
                        valid_loss_values.append(v.simple_value)
                    elif v.tag == 'x':
                        x_steps.append(e.step)
                        x_values.append(v.simple_value)

            return (tags, loss_steps, loss_values, valid_loss_steps,
                    valid_loss_values, x_steps, x_values)

        # test enable summary with `summary_dir`
        with TemporaryDirectory() as tempdir:
            with TrainLoop([], max_epoch=2, summary_dir=tempdir,
                           summary_graph=tf.get_default_graph()) as loop:
                self.assertIsInstance(loop.summary_writer,
                                      tf.summary.FileWriter)
                for epoch in loop.iter_epochs():
                    for _, loss in loop.iter_steps([0.7, 0.6, 0.8]):
                        loop.collect_metrics(loss=epoch + loss)
                    loop.collect_metrics(valid_loss=epoch)

                with self.test_session():
                    summary_op = tf.summary.scalar('x', tf.constant(1.23))
                    loop.add_summary(summary_op.eval())

            obj = read_summary(tempdir)
            self.assertEqual(
                ['metrics/loss', 'metrics/valid_loss', 'x'],
                sorted(obj[0])
            )
            np.testing.assert_equal(obj[1], [1, 2, 3, 4, 5, 6])
            np.testing.assert_almost_equal(
                obj[2],
                [1.7, 1.6, 1.8, 2.7, 2.6, 2.8]
            )
            np.testing.assert_equal(obj[3], [3, 6])
            np.testing.assert_almost_equal(obj[4], [1, 2])
            np.testing.assert_equal(obj[5], [6])
            np.testing.assert_almost_equal(obj[6], [1.23])

        # test enable summary with `summary_writer`
        with TemporaryDirectory() as tempdir:
            sw = tf.summary.FileWriter(tempdir)
            with TrainLoop([], max_epoch=2, summary_writer=sw) as loop:
                self.assertIs(loop.summary_writer, sw)
                self.assertIs(loop._summary_writer, sw)
                for epoch in loop.iter_epochs():
                    for _, loss in loop.iter_steps([0.7, 0.6, 0.8]):
                        loop.collect_metrics(loss=epoch + loss)
                    loop.collect_metrics(valid_loss=epoch)
            sw.close()
            self.assertEqual(
                sorted(read_summary(tempdir)[0]),
                ['metrics/loss', 'metrics/valid_loss']
            )

        with TemporaryDirectory() as tempdir:
            sw = tf.summary.FileWriter(tempdir)
            with TrainLoop([], max_epoch=2, summary_writer=sw) as loop:
                self.assertIs(loop._summary_writer, sw)
                for epoch in loop.iter_epochs():
                    for _, loss in loop.iter_steps([0.7, 0.6, 0.8]):
                        loop.collect_metrics(loss=epoch + loss)
                    loop.collect_metrics(valid_loss=epoch)
            sw.close()
            self.assertEqual(
                sorted(read_summary(tempdir)[0]),
                ['metrics/loss', 'metrics/valid_loss']
            )

    def test_early_stopping(self):
        with self.test_session():
            a = tf.get_variable('a', shape=(), dtype=tf.int32)
            b = tf.get_variable('b', shape=(), dtype=tf.int32)

            # test early-stopping with no valid metric committed
            set_variable_values([a, b], [1, 2])
            self.assertEqual(get_variable_values([a, b]), [1, 2])
            with TrainLoop([a], early_stopping=True) as loop:
                self.assertTrue(loop.use_early_stopping)
                set_variable_values([a, b], [10, 20])
            self.assertEqual(get_variable_values([a, b]), [10, 20])

            # test early-stopping with smaller-better metric
            set_variable_values([a, b], [1, 2])
            self.assertEqual(get_variable_values([a, b]), [1, 2])
            with TrainLoop([a], max_epoch=1, early_stopping=True) as loop:
                for _ in loop.iter_epochs():
                    for step, valid_loss in loop.iter_steps([0.7, 0.6, 0.8]):
                        set_variable_values([a, b], [10 + step, 20 + step])
                        loop.collect_metrics(valid_loss=valid_loss)
            self.assertAlmostEqual(loop.best_valid_metric, 0.6)
            self.assertEqual(get_variable_values([a, b]), [12, 23])

            # test early-stopping with larger-better metric
            set_variable_values([a, b], [1, 2])
            self.assertEqual(get_variable_values([a, b]), [1, 2])
            with TrainLoop([a],
                           max_epoch=1,
                           valid_metric_name='y',
                           valid_metric_smaller_is_better=False,
                           early_stopping=True) as loop:
                for _ in loop.iter_epochs():
                    for step, y in loop.iter_steps([0.7, 0.6, 0.8]):
                        set_variable_values([a, b], [10 + step, 20 + step])
                        loop.collect_metrics(y=y)
            self.assertAlmostEqual(loop.best_valid_metric, 0.8)
            self.assertEqual(get_variable_values([a, b]), [13, 23])

    def test_checkpoint(self):
        class MyObject(CheckpointSavableObject):
            def __init__(self):
                self.value = 123

            def get_state(self):
                return {'value': self.value}

            def set_state(self, state):
                self.value = state['value']

        o = MyObject()
        var = ScheduledVariable('var', initial_value=456, dtype=tf.int32)

        with self.test_session() as sess, \
                TemporaryDirectory() as tempdir:
            ensure_variables_initialized()

            with TrainLoop([var.variable],
                           checkpoint_dir=tempdir,
                           checkpoint_save_objects={'o': o}) as loop:
                loop.make_checkpoint()

            # test restore_checkpoint == True
            o.value = 1234
            var.set(4567)
            self.assertEqual(o.value, 1234)
            self.assertEqual(var.get(), 4567)

            with TrainLoop([var.variable],
                           checkpoint_dir=tempdir,
                           checkpoint_save_objects={'o': o}):
                self.assertEqual(loop.epoch, 0)
                self.assertEqual(loop.step, 0)
                self.assertEqual(o.value, 123)
                self.assertEqual(var.get(), 456)

            # test restore_checkpoint == False, and generate new checkpoints
            o.value = 1234
            var.set(4567)

            with TrainLoop([var.variable],
                           checkpoint_dir=tempdir,
                           checkpoint_save_objects={'o': o},
                           checkpoint_epoch_freq=2,
                           restore_checkpoint=False,
                           max_epoch=8) as loop:
                self.assertEqual(loop.epoch, 0)
                self.assertEqual(loop.step, 0)
                self.assertEqual(o.value, 1234)
                self.assertEqual(var.get(), 4567)

                for epoch in loop.iter_epochs():
                    for _ in loop.iter_steps([1, 1]):
                        pass

                    o.value = 9120 + epoch
                    var.set(9450 + epoch)

            # restore from latest
            with TrainLoop([var.variable],
                           checkpoint_dir=tempdir,
                           checkpoint_save_objects={'o': o}) as loop:
                self.assertEqual(loop.epoch, 8)
                self.assertEqual(loop.step, 16)
                self.assertEqual(o.value, 9128)
                self.assertEqual(var.get(), 9458)

            # restore from specified file
            for epoch in [2, 4, 6, 8]:
                restore_checkpoint = os.path.join(
                    tempdir, 'checkpoint/checkpoint.dat-{}'.format(epoch * 2))

                with TrainLoop([var.variable],
                               checkpoint_dir=tempdir,
                               checkpoint_save_objects={'o': o},
                               restore_checkpoint=restore_checkpoint) as loop:
                    self.assertEqual(loop.epoch, epoch)
                    self.assertEqual(loop.step, epoch * 2)
                    self.assertEqual(o.value, 9120 + epoch)
                    self.assertEqual(var.get(), 9450 + epoch)

    def test_checkpoint_and_early_stopping(self):
        with self.test_session(), TemporaryDirectory() as tempdir:
            a = tf.get_variable('a', shape=(), dtype=tf.int32)
            b = tf.get_variable('b', shape=(), dtype=tf.int32)

            # test early-stopping with no valid metric committed
            set_variable_values([a, b], [1, 2])
            self.assertEqual(get_variable_values([a, b]), [1, 2])
            with TrainLoop([a],
                           checkpoint_dir=tempdir,
                           early_stopping=True) as loop:
                self.assertTrue(loop.use_early_stopping)
                set_variable_values([a, b], [10, 20])
                loop.make_checkpoint()
            self.assertEqual(get_variable_values([a, b]), [10, 20])

            # test early-stopping with smaller-better metric, 1st loop
            set_variable_values([a, b], [1, 2])
            with pytest.raises(KeyboardInterrupt):
                with TrainLoop([a],
                               max_epoch=2,
                               checkpoint_dir=tempdir,
                               early_stopping=True) as loop:
                    self.assertIsNone(loop.best_valid_metric)
                    self.assertEqual(get_variable_values([a, b]), [10, 20])

                    for i, epoch in enumerate(loop.iter_epochs(), 1):
                        self.assertEqual(epoch, i)
                        for j, (step, valid_loss) in \
                                enumerate(loop.iter_steps([0.7, 0.6, 0.8]), 1):
                            self.assertEqual(step, j)
                            set_variable_values([a, b], [10 + step, 20 + step])
                            loop.collect_metrics(valid_loss=valid_loss)
                            loop.make_checkpoint()
                        raise KeyboardInterrupt()

            # because the loop is interrupted, the early-stopping should not
            # restore the variables to the best state
            self.assertEqual(get_variable_values([a, b]), [13, 23])

            # test early-stopping with smaller-better metric, 2nd loop
            with TrainLoop([a],
                           max_epoch=2,
                           checkpoint_dir=tempdir,
                           early_stopping=True) as loop:
                self.assertEqual(loop.best_valid_metric, 0.6)
                self.assertEqual(get_variable_values([a, b]), [13, 23])

                for i, epoch in enumerate(loop.iter_epochs(), 2):
                    self.assertEqual(epoch, i)
                    for j, (step, valid_loss) in \
                            enumerate(loop.iter_steps([0.9]), 4):
                        self.assertEqual(step, j)
                        set_variable_values([a, b], [10 + step, 20 + step])
                        loop.collect_metrics(valid_loss=valid_loss)

            self.assertAlmostEqual(loop.best_valid_metric, 0.6)
            self.assertEqual(get_variable_values([a, b]), [12, 24])

    def test_tensor_arguments(self):
        with self.test_session():
            a = tf.get_variable('a', initializer=0, dtype=tf.int32)
            ensure_variables_initialized()
            with TrainLoop([a],
                           early_stopping=True,
                           max_epoch=tf.constant(6),
                           max_step=tf.constant(7)) as loop:
                self.assertEqual(loop.max_epoch, 6)
                self.assertEqual(loop.max_step, 7)

    def test_errors(self):
        with TemporaryDirectory() as tempdir:
            with pytest.raises(ValueError, match='`checkpoint_epoch_freq` must '
                                                 'be a positive integer'):
                with TrainLoop([], checkpoint_dir=tempdir,
                               checkpoint_epoch_freq=0):
                    pass

            with pytest.raises(ValueError,
                               match='Currently `early_stopping = True` is not '
                                     'supported when a file path is '
                                     'specified for `restore_checkpoint`'):
                with TrainLoop([],
                               checkpoint_dir=tempdir,
                               early_stopping=True,
                               restore_checkpoint=os.path.join(
                                   tempdir, 'checkpoint.dat')):
                    pass

            with pytest.raises(RuntimeError, match='Checkpoint directory is '
                                                   'not configured'):
                with TrainLoop([]) as loop:
                    loop.make_checkpoint()

            obj = Mock(
                spec=CheckpointSavableObject,
                get_state=Mock(return_value={}),
                set_state=Mock()
            )

            with pytest.raises(KeyError, match='Name is reserved for '
                                               '`checkpoint_save_objects`'):
                with TrainLoop([], checkpoint_dir=tempdir,
                               checkpoint_save_objects={
                                   TRAIN_LOOP_STATES_CKPT_NAME: obj
                               }):
                    pass

            with pytest.raises(KeyError, match='Name is reserved for '
                                               '`checkpoint_save_objects`'):
                with TrainLoop([], checkpoint_dir=tempdir,
                               checkpoint_save_objects={
                                   EARLY_STOPPING_STATES_CKPT_NAME: obj
                               }):
                    pass

        with pytest.raises(
                RuntimeError, match='Another epoch loop has been opened'):
            with TrainLoop([], max_epoch=10) as loop:
                for _ in loop.iter_epochs():
                    for _ in loop.iter_epochs():
                        pass

        with pytest.raises(
                RuntimeError, match='Step loop must be opened within active '
                                    'epoch loop'):
            with TrainLoop([], max_step=10) as loop:
                for _ in loop.iter_steps():
                    pass

        with pytest.raises(
                RuntimeError, match='Another step loop has been opened'):
            with TrainLoop([], max_epoch=10, max_step=10) as loop:
                for _ in loop.iter_epochs():
                    for _ in loop.iter_steps():
                        for _ in loop.iter_steps():
                            pass

        def require_context():
            return pytest.raises(
                RuntimeError, match='An epoch or a step loop is expected, '
                                    'but neither has been opened')

        with require_context():
            with TrainLoop([]) as loop:
                with loop.timeit('timer'):
                    pass

        with require_context():
            with TrainLoop([]) as loop:
                with loop.metric_collector('metric'):
                    pass

        with require_context():
            with TrainLoop([]) as loop:
                loop.collect_metrics(loss=1.)

        with require_context():
            with TrainLoop([]) as loop:
                loop.println('', with_tag=True)

        with require_context():
            with TrainLoop([]) as loop:
                loop.print_logs()

        with pytest.raises(
                RuntimeError, match='`data_generator` is required when '
                                    '`max_step` is not configured, so as to '
                                    'prevent an unstoppable step loop'):
            with TrainLoop([], max_epoch=10) as loop:
                for _ in loop.iter_epochs():
                    for _ in loop.iter_steps():
                        pass

        with pytest.raises(
                TypeError, match='`metrics` should be a dict'):
            with TrainLoop([], max_epoch=10) as loop:
                for _ in loop.iter_epochs():
                    loop.collect_metrics(())
