import unittest

import tensorflow as tf

from tfsnippet.utils import is_tensorflow_version_higher_or_equal


class IsTensorflowVersionHigherOrEqualTestCase(unittest.TestCase):

    def test_is_tensorflow_version_higher_or_equal(self):
        # test compatibility with current version
        tf_version = tf.__version__
        self.assertTrue(is_tensorflow_version_higher_or_equal(tf_version),
                        msg='{} >= {} not hold'.format(tf_version, tf_version))

        # test various cases
        try:
            versions = [
                '0.1.0', '0.9.0', '0.12.0', '0.12.1',
                '1.0.0-rc0', '1.0.0-rc1', '1.0.0', '1.0.1',
            ]
            for i, v0 in enumerate(versions):
                tf.__version__ = v0
                for v in versions[:i+1]:
                    self.assertTrue(is_tensorflow_version_higher_or_equal(v),
                                    msg='{} >= {} not hold'.format(v0, v))
                for v in versions[i+1:]:
                    self.assertFalse(is_tensorflow_version_higher_or_equal(v),
                                     msg='{} < {} not hold'.format(v0, v))
        finally:
            tf.__version__ = tf_version
