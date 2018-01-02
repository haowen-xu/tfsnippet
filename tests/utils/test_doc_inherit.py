import unittest

from tfsnippet.utils import DocInherit


class DocInheritTestCase(unittest.TestCase):

    def test_typing(self):
        @DocInherit
        class Parent(object):
            """Doc of parent."""

        class Child(Parent):
            pass

        self.assertIsInstance(Parent(), Parent)
        self.assertIsInstance(Child(), Parent)
        self.assertNotIsInstance(Parent(), Child)
        self.assertIsInstance(Child(), Child)
        self.assertIs(Parent().__class__, Parent)
        self.assertIs(Child().__class__, Child)
        self.assertIs(type(Parent()), Parent)
        self.assertIs(type(Child()), Child)

    def test_class_docstring(self):
        @DocInherit
        class Parent(object):
            """Doc of parent."""

        class ChildA(Parent):
            pass

        class ChildB(Parent):
            """Doc of child."""

        class GrandChildA(ChildA):
            pass

        class GrandChildB(ChildB):
            pass

        self.assertEqual(Parent.__doc__, 'Doc of parent.')
        self.assertEqual(ChildA.__doc__, 'Doc of parent.')
        self.assertEqual(ChildB.__doc__, 'Doc of child.')
        self.assertEqual(GrandChildA.__doc__, 'Doc of parent.')
        self.assertEqual(GrandChildB.__doc__, 'Doc of child.')

    def test_method_docstring(self):
        @DocInherit
        class Parent(object):
            def some_method(self):
                """Doc of parent."""

        class ChildA(Parent):
            def some_method(self):
                pass

        class ChildB(Parent):
            def some_method(self):
                """Doc of child."""

        class GrandChildA(ChildA):
            def some_method(self):
                pass

        class GrandChildB(ChildB):
            def some_method(self):
                pass

        self.assertEqual(Parent.some_method.__doc__, 'Doc of parent.')
        self.assertEqual(ChildA.some_method.__doc__, 'Doc of parent.')
        self.assertEqual(ChildB.some_method.__doc__, 'Doc of child.')
        self.assertEqual(GrandChildA.some_method.__doc__, 'Doc of parent.')
        self.assertEqual(GrandChildB.some_method.__doc__, 'Doc of child.')

    def test_property_docstring(self):
        @DocInherit
        class Parent(object):
            @property
            def some_property(self):
                """Doc of parent."""
                return 'parent'

        class ChildA(Parent):
            @property
            def some_property(self):
                return 'childA'

        class ChildB(Parent):
            @property
            def some_property(self):
                """Doc of child."""
                return 'childB'

        class GrandChildA(ChildA):
            @property
            def some_property(self):
                return 'grandChildA'

        class GrandChildB(ChildB):
            @property
            def some_property(self):
                return 'grandChildB'

        self.assertEqual(Parent.some_property.__doc__, 'Doc of parent.')
        self.assertEqual(ChildA.some_property.__doc__, 'Doc of parent.')
        self.assertEqual(ChildB.some_property.__doc__, 'Doc of child.')
        self.assertEqual(GrandChildA.some_property.__doc__, 'Doc of parent.')
        self.assertEqual(GrandChildB.some_property.__doc__, 'Doc of child.')

        self.assertEqual(Parent().some_property, 'parent')
        self.assertEqual(ChildA().some_property, 'childA')
        self.assertEqual(ChildB().some_property, 'childB')
        self.assertEqual(GrandChildA().some_property, 'grandChildA')
        self.assertEqual(GrandChildB().some_property, 'grandChildB')
