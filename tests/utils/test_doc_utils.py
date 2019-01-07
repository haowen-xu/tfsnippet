import unittest

from tfsnippet.utils import *


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


class AppendToDocTestCase(unittest.TestCase):

    def test_append_single_line_to_empty_doc(self):
        old_doc = None
        content = 'new content'
        expected = '\nnew content\n'
        self.assertEqual(append_to_doc(old_doc, content), expected)

    def test_append_block_to_empty_doc(self):
        old_doc = None
        content = '''
Args:
    a: this is an argument
'''
        expected = '''
Args:
    a: this is an argument
'''
        self.assertEqual(append_to_doc(old_doc, content), expected)

    def test_append_single_line_to_single_line_doc(self):
        old_doc = '''This is a doc string.'''
        content = 'new content'
        expected = '''This is a doc string.

new content
'''
        doc = append_to_doc(old_doc, content)
        self.assertEqual(doc, expected)

    def test_append_block_to_single_line_doc(self):
        old_doc = '''This is a doc string.'''
        content = '''
Args:
    a: this is an argument.

Notes:
    This is the note.
'''
        expected = '''This is a doc string.

Args:
    a: this is an argument.

Notes:
    This is the note.
'''
        doc = append_to_doc(old_doc, content)
        self.assertEqual(doc, expected)

    def test_append_single_line_to_doc(self):
        old_doc = '''
    This is a doc string.
'''
        content = 'new content'
        expected = '''
    This is a doc string.

    new content
'''
        doc = append_to_doc(old_doc, content)
        self.assertEqual(doc, expected)

    def test_append_block_to_doc(self):
        old_doc = '''
    This is a doc string.
'''
        content = '''
Args:
    a: this is an argument.

Notes:
    This is the note.
'''
        expected = '''
    This is a doc string.

    Args:
        a: this is an argument.

    Notes:
        This is the note.
'''
        doc = append_to_doc(old_doc, content)
        self.assertEqual(doc, expected)


class AppendArgToDocTestCase(unittest.TestCase):

    maxDiff = None
    arg_doc = """
name (str): Default name of the variable scope.  Will be uniquified.
    If not specified, generate one according to the class name.
scope (str): The name of the variable scope."""

    def test_add_to_empty(self):
        old_doc = """
    Header line.
"""
        new_doc = """
    Header line.

    Args:
        name (str): Default name of the variable scope.  Will be uniquified.
            If not specified, generate one according to the class name.
        scope (str): The name of the variable scope.
"""
        self.assertEqual(
            append_arg_to_doc(old_doc, self.arg_doc),
            new_doc,
        )

    def test_add(self):
        old_doc = """
    Header line.

    Args:
        a: 1
            this is multi-line
        b: 2
"""
        new_doc = """
    Header line.

    Args:
        a: 1
            this is multi-line
        b: 2
        name (str): Default name of the variable scope.  Will be uniquified.
            If not specified, generate one according to the class name.
        scope (str): The name of the variable scope.
"""
        self.assertEqual(
            append_arg_to_doc(old_doc, self.arg_doc),
            new_doc,
        )

    def test_add_with_arg(self):
        old_doc = """
    Header line.
    
    Args:
        a: 1
            this is multi-line
        b: 2
        \\*args: arguments
"""
        new_doc = """
    Header line.
    
    Args:
        a: 1
            this is multi-line
        b: 2
        name (str): Default name of the variable scope.  Will be uniquified.
            If not specified, generate one according to the class name.
        scope (str): The name of the variable scope.
        \\*args: arguments
"""
        self.assertEqual(
            append_arg_to_doc(old_doc, self.arg_doc),
            new_doc,
        )

    def test_add_with_kwargs(self):
        old_doc = """
Header line.

Args:
    a: 1
        this is multi-line
    b: 2
    \\**kwargs: the keyword arguments
"""
        new_doc = """
Header line.

Args:
    a: 1
        this is multi-line
    b: 2
    name (str): Default name of the variable scope.  Will be uniquified.
        If not specified, generate one according to the class name.
    scope (str): The name of the variable scope.
    \\**kwargs: the keyword arguments
"""
        self.assertEqual(
            append_arg_to_doc(old_doc, self.arg_doc),
            new_doc,
        )

    def test_add_with_next_section(self):
        old_doc = """
    Header line.
    
    Args:
        a: 1
            this is multi-line
        b: 2
    
    Notes:
        This is a note
"""
        new_doc = """
    Header line.
    
    Args:
        a: 1
            this is multi-line
        b: 2
        name (str): Default name of the variable scope.  Will be uniquified.
            If not specified, generate one according to the class name.
        scope (str): The name of the variable scope.
    
    Notes:
        This is a note
"""
        self.assertEqual(
            append_arg_to_doc(old_doc, self.arg_doc),
            new_doc,
        )


class AddArgDocDecoratorsTestCase(unittest.TestCase):

    maxDiff = None

    def test_add_name_arg_doc(self):
        @add_name_arg_doc
        def f(first_arg):
            """
            Header line.

            Args:
                first_arg: this is the first arg

                    First arg has a blank line.
            """

        expected_doc = """
            Header line.

            Args:
                first_arg: this is the first arg

                    First arg has a blank line.
                name (str): Default name of the name scope.
                    If not specified, generate one according to the method name.
            \n"""
        self.assertEqual(f.__doc__, expected_doc)

    def test_add_name_and_scope_arg_doc(self):
        @add_name_and_scope_arg_doc
        def f(first_arg):
            """
            Header line.

            Args:
                first_arg: this is the first arg

                    First arg has a blank line.
            """

        expected_doc = """
            Header line.

            Args:
                first_arg: this is the first arg

                    First arg has a blank line.
                name (str): Default name of the variable scope.  Will be uniquified.
                    If not specified, generate one according to the class name.
                scope (str): The name of the variable scope.
            \n"""
        self.assertEqual(f.__doc__, expected_doc)
