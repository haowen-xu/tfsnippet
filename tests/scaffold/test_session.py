import pytest
import tensorflow as tf

from tfsnippet.scaffold import get_default_session_or_error


class SessionTestCase(tf.test.TestCase):

    def test_get_default_session_or_error(self):
        with pytest.raises(RuntimeError, message='No session is active'):
            get_default_session_or_error()
        with self.test_session(use_gpu=False) as sess:
            self.assertIs(sess, get_default_session_or_error())
        with pytest.raises(RuntimeError, message='No session is active'):
            get_default_session_or_error()
