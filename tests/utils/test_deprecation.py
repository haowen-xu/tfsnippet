import unittest
import warnings

import pytest

from tfsnippet.utils import *


class DeprecationTestCase(unittest.TestCase):

    def test_deprecated_class_and_method(self):
        @deprecated('use `YourClass` instead.', version='1.2.3')
        class MyClass(object):
            def __init__(self, value):
                self.value = value

            @deprecated('use `g` instead.', version='1.2.4')
            def f(self):
                return self.value

            @deprecated()
            def g(self):
                return self.value + 1

        @deprecated()
        class YourClass(object):
            pass

        # test the deprecation of the class
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            o = MyClass(1234)
            self.assertEqual(o.value, 1234)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn('Class `MyClass` is deprecated; '
                          'use `YourClass` instead.', str(w[-1].message))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            _ = YourClass()

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn('Class `YourClass` is deprecated.',
                          str(w[-1].message))

        # test the deprecation of the method
        o = MyClass(1234)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            ret = o.f()
            self.assertEqual(ret, 1234)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn('Function `f` is deprecated; '
                          'use `g` instead.', str(w[-1].message))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            ret = o.g()
            self.assertEqual(ret, 1235)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn('Function `g` is deprecated.',
                          str(w[-1].message))

    def test_deprecated_function(self):
        @deprecated('use `your_func` instead.', version='1.2.3')
        def my_func(value):
            return value

        @deprecated()
        def your_func(value):
            return value + 1

        # test the deprecation of the function
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            ret = my_func(1234)
            self.assertEqual(ret, 1234)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn('Function `my_func` is deprecated; '
                          'use `your_func` instead.', str(w[-1].message))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            ret = your_func(1234)
            self.assertEqual(ret, 1235)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn('Function `your_func` is deprecated.',
                          str(w[-1].message))

    def test_dreprecated_arg(self):
        @deprecated_arg('a', version='1.2.3')
        def my_func(a=None, b=None, c=None):
            return a, b, c

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(my_func(b=1, c=2), (None, 1, 2))
            self.assertEqual(len(w), 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(my_func(a=1, b=1.5, c=2), (1, 1.5, 2))
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn('In function `my_func`: argument `a` is deprecated '
                          'since 1.2.3', str(w[-1].message))

        @deprecated_arg('a')
        def no_version(a=None, b=None, c=None):
            return a, b, c

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(no_version(a=1, c=2), (1, None, 2))
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn('In function `no_version`: argument `a` is '
                          'deprecated', str(w[-1].message))

    def test_deprecated_arg_with_new_arg(self):
        @deprecated_arg('a', 'b', version='1.2.3')
        def my_func(a=None, b=None, c=None):
            return a, b, c

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(my_func(b=1, c=2), (None, 1, 2))
            self.assertEqual(len(w), 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(my_func(a=1, c=2), (1, None, 2))
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn('In function `my_func`: argument `a` is deprecated '
                          'since 1.2.3, use `b` instead', str(w[-1].message))

        with pytest.raises(
                TypeError, match='You should not specify the deprecated '
                                 'argument `a` and its replacement `b` '
                                 'at the same time.'):
            _ = my_func(a=1, b=2, c=3)

        @deprecated_arg('a', 'b')
        def no_version(a=None, b=None, c=None):
            return a, b, c

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(no_version(a=1, c=2), (1, None, 2))
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn('In function `no_version`: argument `a` is '
                          'deprecated, use `b` instead', str(w[-1].message))
