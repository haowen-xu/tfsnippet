import unittest

import pytest
import six

from tfsnippet import config


class ConfigTestCase(unittest.TestCase):

    def test_default_config_values(self):
        self.assertEqual(config.float_x, 'float32')

    def test_to_dict(self):
        config_dict = config.to_dict()
        for k, v in six.iteritems(config_dict):
            self.assertTrue(not k.startswith('_'))
            self.assertEqual(v, getattr(config, k))

    def test_set_unknown_attribute(self):
        with pytest.raises(
                AttributeError,
                match='Unknown config attribute: unknown_attribute'):
            config.unknown_attribute = 'value'

        with pytest.raises(
                AttributeError,
                match='Unknown config attribute: unknown_attribute'):
            config.from_dict({'unknown_attribute': 'value'})

    def test_float_x(self):
        config.float_x = 'float16'
        self.assertEqual(config.float_x, 'float16')
        config.from_dict({'float_x': 'float64'})
        self.assertEqual(config.float_x, 'float64')

        with pytest.raises(
                ValueError, match='Unknown floatx type: invalid value'):
            config.float_x = 'invalid value'
        with pytest.raises(
                ValueError, match='Unknown floatx type: invalid value'):
            config.from_dict({'float_x': 'invalid value'})
