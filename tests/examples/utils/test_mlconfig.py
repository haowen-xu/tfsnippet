import unittest

import pytest

from tfsnippet.examples.utils import MLConfig


class MLConfigTestCase(unittest.TestCase):

    def test_assign(self):
        class MyConfig(MLConfig):
            a = 123

        config = MyConfig()
        self.assertEqual(config.a, 123)
        config.a = 234
        self.assertEqual(config.a, 234)

        with pytest.raises(AttributeError, match='Config key \'non_exist\' '
                                                 'does not exist'):
            config.non_exist = 12345

    def test_defaults_and_to_dict(self):
        self.assertDictEqual(MLConfig.defaults(), {})
        self.assertDictEqual(MLConfig().to_dict(), {})

        class MyConfig(MLConfig):
            a = 123
            b = 456

        self.assertDictEqual(MyConfig.defaults(), {'a': 123, 'b': 456})
        config = MyConfig()
        self.assertDictEqual(config.to_dict(), {'a': 123, 'b': 456})
        config.a = 333
        self.assertDictEqual(config.to_dict(), {'a': 333, 'b': 456})

        class MyConfig2(MyConfig):
            a = 234
            c = 1234

        self.assertDictEqual(MyConfig2.defaults(),
                             {'a': 234, 'b': 456, 'c': 1234})
        config = MyConfig2()
        self.assertDictEqual(config.to_dict(), {'a': 234, 'b': 456, 'c': 1234})
        config.a = 333
        config.c = 444
        self.assertDictEqual(config.to_dict(), {'a': 333, 'b': 456, 'c': 444})

