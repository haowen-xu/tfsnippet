import pytest
import tensorflow as tf

from tfsnippet.shortcuts import global_reuse
from tfsnippet.utils import (get_default_session_or_error,
                             get_variables_as_dict,
                             get_uninitialized_variables,
                             ensure_variables_initialized,
                             create_session,
                             get_variable_ddi)


class CreateSessionTestCase(tf.test.TestCase):

    def test_create_session(self):
        with pytest.raises(TypeError, match='`lock_memory` must be True, '
                                            'False or float'):
            _ = create_session(lock_memory='')

        # test with default options
        session = create_session()
        self.assertFalse(session._config.gpu_options.allow_growth)
        self.assertFalse(session._config.log_device_placement)
        self.assertTrue(session._config.allow_soft_placement)

        # test with various options
        session = create_session(lock_memory=0.5, log_device_placement=True,
                                 allow_soft_placement=False)
        self.assertEqual(
            session._config.gpu_options.per_process_gpu_memory_fraction, .5)
        self.assertTrue(session._config.log_device_placement)
        self.assertFalse(session._config.allow_soft_placement)

        # test with lock_memory = False
        session = create_session(lock_memory=False)
        self.assertTrue(session._config.gpu_options.allow_growth)


class GetDefaultSessionOrErrorTestCase(tf.test.TestCase):

    def test_get_default_session_or_error(self):
        with pytest.raises(RuntimeError, match='No session is active'):
            get_default_session_or_error()
        with self.test_session(use_gpu=False) as sess:
            self.assertIs(sess, get_default_session_or_error())
        with pytest.raises(RuntimeError, match='No session is active'):
            get_default_session_or_error()


class GetVariablesTestCase(tf.test.TestCase):

    def test_get_variables_as_dict(self):
        GLOBAL_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
        MODEL_VARIABLES = tf.GraphKeys.MODEL_VARIABLES
        LOCAL_VARIABLES = tf.GraphKeys.LOCAL_VARIABLES

        # create the variables to be checked
        a = tf.get_variable(
            'a', shape=(), collections=[GLOBAL_VARIABLES, MODEL_VARIABLES])
        b = tf.get_variable(
            'b', shape=(), collections=[GLOBAL_VARIABLES])
        c = tf.get_variable(
            'c', shape=(), collections=[MODEL_VARIABLES])

        with tf.variable_scope('child') as child:
            child_a = tf.get_variable(
                'a', shape=(),
                collections=[GLOBAL_VARIABLES, MODEL_VARIABLES])
            child_b = tf.get_variable(
                'b', shape=(), collections=[GLOBAL_VARIABLES])
            child_c = tf.get_variable(
                'c', shape=(), collections=[MODEL_VARIABLES])

        # test to get variables as dict
        self.assertEqual(
            get_variables_as_dict(),
            {'a': a, 'b': b, 'child/a': child_a, 'child/b': child_b}
        )
        self.assertEqual(
            get_variables_as_dict(collection=MODEL_VARIABLES),
            {'a': a, 'c': c, 'child/a': child_a, 'child/c': child_c}
        )
        self.assertEqual(
            get_variables_as_dict(collection=LOCAL_VARIABLES),
            {}
        )
        self.assertEqual(
            get_variables_as_dict(''),
            {'a': a, 'b': b, 'child/a': child_a, 'child/b': child_b}
        )
        self.assertEqual(
            get_variables_as_dict('child'),
            {'a': child_a, 'b': child_b}
        )
        self.assertEqual(
            get_variables_as_dict('child/'),
            {'a': child_a, 'b': child_b}
        )
        self.assertEqual(
            get_variables_as_dict(child),
            {'a': child_a, 'b': child_b}
        )
        self.assertEqual(
            get_variables_as_dict('child', collection=MODEL_VARIABLES),
            {'a': child_a, 'c': child_c}
        )
        self.assertEqual(
            get_variables_as_dict('child', collection=LOCAL_VARIABLES),
            {}
        )
        self.assertEqual(
            get_variables_as_dict('non_exist'),
            {}
        )


class GetUninitializedVariablesTestCase(tf.test.TestCase):

    def test_get_uninitialized_variables(self):
        with self.test_session() as sess:
            a = tf.get_variable('a', dtype=tf.int32, initializer=1)
            b = tf.get_variable('b', dtype=tf.int32, initializer=2)
            c = tf.get_variable('c', dtype=tf.int32, initializer=3,
                                collections=[tf.GraphKeys.MODEL_VARIABLES])
            d = tf.get_variable('d', dtype=tf.int32, initializer=4,
                                collections=[tf.GraphKeys.MODEL_VARIABLES])
            self.assertEqual(
                get_uninitialized_variables(),
                [a, b]
            )
            self.assertEqual(
                get_uninitialized_variables([a, b, c, d]),
                [a, b, c, d]
            )
            sess.run(tf.variables_initializer([a, c]))
            self.assertEqual(
                get_uninitialized_variables(),
                [b]
            )
            self.assertEqual(
                get_uninitialized_variables([a, b, c, d]),
                [b, d]
            )
            sess.run(tf.variables_initializer([b, d]))
            self.assertEqual(
                get_uninitialized_variables(),
                []
            )
            self.assertEqual(
                get_uninitialized_variables([a, b, c, d]),
                []
            )


class EnsureVariablesInitializedTestCase(tf.test.TestCase):

    def test_ensure_variables_initialized(self):
        a = tf.get_variable('a', dtype=tf.int32, initializer=1)
        b = tf.get_variable('b', dtype=tf.int32, initializer=2)
        c = tf.get_variable('c', dtype=tf.int32, initializer=3,
                            collections=[tf.GraphKeys.MODEL_VARIABLES])
        d = tf.get_variable('d', dtype=tf.int32, initializer=4,
                            collections=[tf.GraphKeys.MODEL_VARIABLES])

        # test using list
        with self.test_session():
            self.assertEqual(
                get_uninitialized_variables([a, b, c, d]),
                [a, b, c, d]
            )
            ensure_variables_initialized()
            self.assertEqual(
                get_uninitialized_variables([a, b, c, d]),
                [c, d]
            )
            ensure_variables_initialized([a, b, c, d])
            self.assertEqual(
                get_uninitialized_variables([a, b, c, d]),
                []
            )

    def test_ensure_variables_initialized_using_dict(self):
        a = tf.get_variable('a', dtype=tf.int32, initializer=1)
        b = tf.get_variable('b', dtype=tf.int32, initializer=2)

        # test using dict
        with self.test_session():
            ensure_variables_initialized({'a': a})
            self.assertEqual(
                get_uninitialized_variables([a, b]),
                [b]
            )


class GetVariableDDITestCase(tf.test.TestCase):

    def test_get_variable_ddi(self):
        # test collections
        def g(name, initial_value, initializing=False, collections=None):
            v = get_variable_ddi(
                name, initial_value, shape=(), initializing=initializing,
                collections=collections
            )
            # ensure `get_variable_ddi` will add the variable to collections
            for coll in (collections or [tf.GraphKeys.GLOBAL_VARIABLES]):
                self.assertEqual(
                    tf.get_collection(coll)[-1].name.rsplit('/', 1)[-1],
                    name + ':0'
                )
            return v

        _ = g('var', 0., initializing=True,
              collections=[tf.GraphKeys.MODEL_VARIABLES])

        # test reuse
        @global_reuse
        def f(initial_value, initializing=False):
            return g('x', initial_value, initializing=initializing)

        with self.test_session() as sess:
            x_in = tf.placeholder(dtype=tf.float32, shape=())
            x = f(x_in, initializing=True)
            self.assertEqual(sess.run(x, feed_dict={x_in: 123.}), 123.)
            x = f(x_in, initializing=False)
            self.assertEqual(sess.run(x, feed_dict={x_in: 456.}), 123.)
