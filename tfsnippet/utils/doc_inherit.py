import six

__all__ = ['DocInherit']


class DocStringInheritor(type):
    """
    Meta-class for automatically inherit docstrings from base classes.
    """

    def __new__(kclass, name, bases, dct):
        def iter_mro():
            for base in bases:
                for mro in base.__mro__:
                    yield mro

        # inherit the class docstring
        if not dct.get('__doc__', None):
            for cls in iter_mro():
                cls_doc = getattr(cls, '__doc__', None)
                if cls_doc:
                    dct['__doc__'] = cls_doc
                    break

        # inherit the method docstrings
        for key in dct:
            attr = dct[key]
            if attr is not None and not getattr(attr, '__doc__', None):
                for cls in iter_mro():
                    cls_attr = getattr(cls, key, None)
                    if cls_attr:
                        cls_doc = getattr(cls_attr, '__doc__', None)
                        if cls_doc:
                            if isinstance(attr, property) and six.PY2:
                                # In Python 2.x, "__doc__" of a property
                                # is read-only.  We choose to wrap the
                                # original property in a new property.
                                dct[key] = property(
                                    fget=attr.fget,
                                    fset=attr.fset,
                                    fdel=attr.fdel,
                                    doc=cls_doc
                                )
                            else:
                                attr.__doc__ = cls_doc
                            break

        return super(DocStringInheritor, kclass). \
            __new__(kclass, name, bases, dct)


def DocInherit(kclass):
    """
    Class decorator to enable `kclass` and all its sub-classes to
    automatically inherit docstrings from base classes.

    Usage:

    .. code-block:: python

        import six


        @DocInherit
        class Parent(object):
            \"""Docstring of the parent class.\"""

            def some_method(self):
                \"""Docstring of the method.\"""
                ...

        class Child(Parent):
            # inherits the docstring of :meth:`Parent`

            def some_method(self):
                # inherits the docstring of :meth:`Parent.some_method`
                ...

    Args:
        kclass (Type): The class to decorate.

    Returns:
        The decorated class.
    """
    return six.add_metaclass(DocStringInheritor)(kclass)
