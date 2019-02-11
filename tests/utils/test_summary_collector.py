import tensorflow as tf
from mock import mock, Mock

from tfsnippet import SummaryCollector, default_summary_collector, settings, \
    add_summary, add_histogram
from tfsnippet.utils import scoped_set_config


class SummaryCollectorTestCase(tf.test.TestCase):

    def test_props(self):
        c = SummaryCollector()
        self.assertEqual(c.collections, ())
        self.assertEqual(c.summary_list, [])
        self.assertFalse(c._no_add_to_collections)

        c = SummaryCollector(
            iter([tf.GraphKeys.SUMMARIES]),
            no_add_to_collections=True
        )
        self.assertEqual(c.collections, (tf.GraphKeys.SUMMARIES,))
        self.assertTrue(c._no_add_to_collections)

    def test_context_stack(self):
        def_collector = default_summary_collector()
        self.assertIsNotNone(def_collector)

        c = SummaryCollector()
        self.assertIsNot(c, def_collector)

        with scoped_set_config(settings, auto_histogram=False):
            with c.as_default() as cc:
                self.assertIs(cc, c)
                self.assertIs(default_summary_collector(), c)
                self.assertFalse(settings.auto_histogram)

            with c.as_default(auto_histogram=True) as cc:
                self.assertIs(cc, c)
                self.assertIs(default_summary_collector(), c)
                self.assertTrue(settings.auto_histogram)

        with scoped_set_config(settings, auto_histogram=True):
            with c.as_default() as cc:
                self.assertIs(cc, c)
                self.assertIs(default_summary_collector(), c)
                self.assertTrue(settings.auto_histogram)

            with c.as_default(auto_histogram=False) as cc:
                self.assertIs(cc, c)
                self.assertIs(default_summary_collector(), c)
                self.assertFalse(settings.auto_histogram)

        self.assertIs(default_summary_collector(), def_collector)

    def test_add_summary(self):
        graph_key = 'my_summaries'
        graph_key_2 = 'my_summaries_2'

        # test no_add_to_collections = False
        with tf.Graph().as_default():
            c = SummaryCollector(collections=[graph_key])

            with mock.patch('tensorflow.summary.merge') as m:
                self.assertIsNone(c.merge_summary())
            self.assertFalse(m.called)

            summary_op = tf.summary.scalar('v1', tf.constant(123.))
            summary_op_2 = tf.summary.scalar('v2', tf.constant(456.))
            summary_op_3 = tf.summary.scalar('v3', tf.constant(789.))

            self.assertIs(c.add_summary(summary_op), summary_op)
            self.assertEqual(c.summary_list, [summary_op])
            self.assertIn(summary_op, tf.get_collection(graph_key))

            self.assertIs(
                c.add_summary(summary_op_2, collections=[graph_key_2]),
                summary_op_2)
            self.assertEqual(c.summary_list, [summary_op, summary_op_2])
            self.assertNotIn(summary_op_2, tf.get_collection(graph_key))
            self.assertIn(summary_op_2, tf.get_collection(graph_key_2))

            self.assertIs(
                c.add_summary(summary_op_3, collections=[]),
                summary_op_3)
            self.assertEqual(
                c.summary_list,
                [summary_op, summary_op_2, summary_op_3])
            self.assertNotIn(summary_op_3, tf.get_collection(graph_key))
            self.assertNotIn(summary_op_3, tf.get_collection(graph_key_2))

            with mock.patch('tensorflow.summary.merge') as m:
                self.assertIsNotNone(c.merge_summary())
            self.assertEqual(m.call_args, ((c.summary_list,), {}))

        # test no_add_to_collections = True
        with tf.Graph().as_default():
            c = SummaryCollector(
                collections=[graph_key], no_add_to_collections=True)
            summary_op = tf.summary.scalar('v1', tf.constant(123.))
            summary_op_2 = tf.summary.scalar('v2', tf.constant(456.))
            summary_op_3 = tf.summary.scalar('v3', tf.constant(789.))

            self.assertIs(c.add_summary(summary_op), summary_op)
            self.assertEqual(c.summary_list, [summary_op])
            self.assertNotIn(summary_op, tf.get_collection(graph_key))

            self.assertIs(
                c.add_summary(summary_op_2, collections=[graph_key_2]),
                summary_op_2)
            self.assertEqual(c.summary_list, [summary_op, summary_op_2])
            self.assertNotIn(summary_op_2, tf.get_collection(graph_key))
            self.assertNotIn(summary_op_2, tf.get_collection(graph_key_2))

            self.assertIs(
                c.add_summary(summary_op_3, collections=[]),
                summary_op_3)
            self.assertEqual(
                c.summary_list,
                [summary_op, summary_op_2, summary_op_3])
            self.assertNotIn(summary_op_3, tf.get_collection(graph_key))
            self.assertNotIn(summary_op_3, tf.get_collection(graph_key_2))

            with mock.patch('tensorflow.summary.merge') as m:
                self.assertIsNotNone(c.merge_summary())
            self.assertEqual(m.call_args, ((c.summary_list,), {}))

    def test_global_add_summary(self):
        c = SummaryCollector()
        c.add_summary = Mock(wraps=c.add_summary)
        summary_op = tf.summary.scalar('v1', tf.constant(123.))

        with c.as_default():
            self.assertIs(
                add_summary(summary_op, collections=['my_summaries']),
                summary_op
            )

        self.assertEqual(c.add_summary.call_args, (
            (summary_op,), {'collections': ['my_summaries']}
        ))

    def test_global_add_histogram(self):
        c = SummaryCollector()
        c.add_histogram = Mock(wraps=c.add_histogram)
        v = tf.get_variable('v', shape=[2, 3], dtype=tf.float32,
                            initializer=tf.zeros_initializer())

        with c.as_default():
            self.assertIsNotNone(
                add_histogram(v, 'var', strip_scope=True, name='ns',
                              collections=['my_summaries'])
            )

        self.assertEqual(c.add_histogram.call_args, (
            (), {
                'tensor': v,
                'summary_name': 'var',
                'strip_scope': True,
                'name': 'ns',
                'collections': ['my_summaries']
            }
        ))
