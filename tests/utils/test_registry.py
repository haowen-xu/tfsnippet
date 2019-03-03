import unittest

import pytest

from tfsnippet.utils import BaseRegistry, ClassRegistry


class RegistryTestCase(unittest.TestCase):

    def test_base_registry(self):
        a = object()
        b = object()

        # test not ignore case
        r = BaseRegistry(ignore_case=False)
        self.assertFalse(r.ignore_case)

        r.register('a', a)
        self.assertIs(r.get('a'), a)
        with pytest.raises(KeyError, match='Object not registered: \'A\''):
            _ = r.get('A')
        self.assertListEqual(list(r), ['a'])

        with pytest.raises(KeyError, match='Object already registered: \'a\''):
            _ = r.register('a', a)
        with pytest.raises(KeyError, match='Object not registered: \'b\''):
            _ = r.get('b')

        r.register('A', b)
        self.assertIs(r.get('A'), b)
        self.assertListEqual(list(r), ['a', 'A'])

        # test ignore case
        r = BaseRegistry(ignore_case=True)
        self.assertTrue(r.ignore_case)

        r.register('a', a)
        self.assertIs(r.get('a'), a)
        self.assertIs(r.get('A'), a)
        self.assertListEqual(list(r), ['a'])

        with pytest.raises(KeyError, match='Object already registered: \'A\''):
            _ = r.register('A', a)
        with pytest.raises(KeyError, match='Object not registered: \'b\''):
            _ = r.get('b')

        r.register('B', b)
        self.assertIs(r.get('b'), b)
        self.assertIs(r.get('B'), b)
        self.assertListEqual(list(r), ['a', 'B'])

    def test_class_registry(self):
        r = ClassRegistry()

        with pytest.raises(TypeError, match='`obj` is not a class: 123'):
            r.register('int', 123)

        class MyClass(object):
            def __init__(self, value, message):
                self.value = value
                self.message = message

        r.register('MyClass', MyClass)
        self.assertIs(r.get('MyClass'), MyClass)
        o = r.construct('MyClass', 123, message='message')
        self.assertIsInstance(o, MyClass)
        self.assertEqual(o.value, 123)
        self.assertEqual(o.message, 'message')
