# -*- coding: utf-8 -*-
import contextlib
import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from tfsnippet.scaffold import (summarize_variables, MetricLogger,
                                ScheduledVariable)
from tfsnippet.utils import TemporaryDirectory, ensure_variables_initialized


class LoggingUtilsTestCase(tf.test.TestCase):

    def test_summarize_variables(self):
        def strip_summary(cnt):
            return '\n'.join(l.lstrip() for l in cnt.strip().split('\n'))

        a = tf.get_variable('a', dtype=tf.int32, shape=[2])
        with tf.variable_scope('nested'):
            b = tf.get_variable('b', dtype=tf.float32, shape=(3, 4, 5))
        c = tf.get_variable('c', dtype=tf.float32, shape=(30000,))

        self.assertEqual(summarize_variables([]), '')
        self.assertEqual(summarize_variables([a]), (
            'Variables Summary (2 in total)\n'
            '------------------------------\n'
            'a                      (2,)  2'
        ))
        self.assertEqual(summarize_variables([a, b, c]), (
            'Variables Summary (30,062 in total)\n'
            '-----------------------------------\n'
            'a                 (2,)            2\n'
            'nested/b          (3, 4, 5)      60\n'
            'c                 (30000,)   30,000'
        ))
        self.assertEqual(summarize_variables([a, b, c], sort_by_names=True), (
            'Variables Summary (30,062 in total)\n'
            '-----------------------------------\n'
            'a                 (2,)            2\n'
            'c                 (30000,)   30,000\n'
            'nested/b          (3, 4, 5)      60'
        ))
        var_dict = OrderedDict([('a', a), ('c', c), ('b', b)])
        self.assertEqual(summarize_variables(var_dict), (
            'Variables Summary (30,062 in total)\n'
            '-----------------------------------\n'
            'a                 (2,)            2\n'
            'c                 (30000,)   30,000\n'
            'b                 (3, 4, 5)      60'
        ))
        self.assertEqual(
            summarize_variables(var_dict, groups=['pfx1', 'pfx2'],
                                sort_by_names=True), (
                'Variables Summary (30,062 in total)\n'
                '-----------------------------------\n'
                'a                 (2,)            2\n'
                'b                 (3, 4, 5)      60\n'
                'c                 (30000,)   30,000'
            )
        )
        self.assertEqual(
            summarize_variables(
                {'pfx1/a': a, 'pfx1/b': b, 'pfx2/a': a, 'pfx2/b': b, 'c': c},
                groups=['pfx1', 'pfx2'], sort_by_names=True
            ),
            strip_summary("""
                Variables Summary (30,124 in total)
                ===================================
                pfx1/                 (62 in total)
                -----------------------------------
                a                     (2,)        2
                b                     (3, 4, 5)  60
                
                pfx2/                 (62 in total)
                -----------------------------------
                a                     (2,)        2
                b                     (3, 4, 5)  60
                
                Other Variables   (30,000 in total)
                -----------------------------------
                c                  (30000,)  30,000
            """)
        )
        self.assertEqual(
            summarize_variables(
                {'pfx1/a': a, 'pfx1/b': b, 'pfx2/a': a, 'pfx2/b': b},
                groups=['pfx1', 'pfx2'], sort_by_names=True
            ),
            strip_summary("""
                Variables Summary (124 in total)
                ================================
                pfx1/              (62 in total)
                --------------------------------
                a                  (2,)        2
                b                  (3, 4, 5)  60
                
                pfx2/              (62 in total)
                --------------------------------
                a                  (2,)        2
                b                  (3, 4, 5)  60
            """)
        )


class MetricLoggerTestCase(tf.test.TestCase):

    def test_basic_logging(self):
        v = ScheduledVariable('v', 1.)

        logger = MetricLogger()
        self.assertIsInstance(logger.metrics, dict)
        self.assertEqual(logger.format_logs(), '')

        with self.test_session() as sess:
            ensure_variables_initialized()
            logger.collect_metrics(dict(loss=v))
        logger.collect_metrics(dict(loss=2., valid_loss=3., valid_timer=0.1))
        logger.collect_metrics(dict(loss=4., valid_acc=5., train_time=0.2))
        logger.collect_metrics(dict(loss=6., valid_acc=7., train_time=0.3))
        logger.collect_metrics(dict(other_metric=5.))
        # self.assertEqual(
        #     logger.format_logs(),
        #     'train time: 0.25s (±0.05s); '
        #     'valid timer: 0.1s; '
        #     'other metric: 5; '
        #     'loss: 3.25 (±1.92029); '
        #     'valid loss: 3; '
        #     'valid acc: 6 (±1)'
        # )
        self.assertEqual(
            logger.format_logs(),
            'train time: 0.25s (±0.05s); '
            'valid timer: 0.1s; '
            'loss: 3.25 (±1.92029); '
            'other metric: 5; '
            'valid acc: 6 (±1); '
            'valid loss: 3'
        )

        logger.clear()
        self.assertEqual(logger.format_logs(), '')

        logger.collect_metrics({'loss': 1.})
        self.assertEqual(logger.format_logs(), 'loss: 1')

    def test_summary_writer(self):
        with TemporaryDirectory() as tempdir:
            # generate the metric summary
            with contextlib.closing(tf.summary.FileWriter(tempdir)) as sw:
                logger = MetricLogger(
                    sw,
                    summary_skip_pattern=r'.*(time|timer)$',
                    summary_commit_freqs={'every_two': 2}
                )
                step = 0
                for epoch in range(1, 3):
                    for data in range(10):
                        step += 1
                        logger.collect_metrics({'acc': step * 100 + data}, step)
                        logger.collect_metrics({'time': epoch}, step)
                        logger.collect_metrics({'every_two': step * 2}, step)

                    with self.test_session(use_gpu=False):
                        logger.collect_metrics(
                            {'valid_loss': -epoch}, tf.constant(step))

            # read the metric summary
            acc_steps = []
            acc_values = []
            valid_loss_steps = []
            valid_loss_values = []
            every_two_steps = []
            every_two_values = []
            tags = set()

            event_file_path = os.path.join(tempdir, os.listdir(tempdir)[0])
            for e in tf.train.summary_iterator(event_file_path):
                for v in e.summary.value:
                    tags.add(v.tag)
                    if v.tag == 'acc':
                        acc_steps.append(e.step)
                        acc_values.append(v.simple_value)
                    elif v.tag == 'valid_loss':
                        valid_loss_steps.append(e.step)
                        valid_loss_values.append(v.simple_value)
                    elif v.tag == 'every_two':
                        every_two_steps.append(e.step)
                        every_two_values.append(v.simple_value)

            self.assertEqual(sorted(tags), ['acc', 'every_two', 'valid_loss'])
            np.testing.assert_equal(acc_steps, np.arange(1, 21))
            np.testing.assert_almost_equal(
                acc_values,
                np.arange(1, 21) * 100 + np.concatenate([
                    np.arange(10), np.arange(10)
                ])
            )
            np.testing.assert_equal(every_two_steps, np.arange(1, 21, 2))
            np.testing.assert_almost_equal(
                every_two_values,
                np.arange(1, 21, 2) * 2
            )
            np.testing.assert_equal(valid_loss_steps, [10, 20])
            np.testing.assert_almost_equal(
                valid_loss_values,
                [-1, -2]
            )
