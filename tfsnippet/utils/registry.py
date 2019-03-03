import six

from .doc_utils import DocInherit

__all__ = ['BaseRegistry', 'ClassRegistry']


@DocInherit
class BaseRegistry(object):
    """
    A base class for implement a type or object registry.

    Usage::

        registry = BaseRegistry()
        registry.register('MNIST', spt.datasets.MNIST())
    """

    def __init__(self, ignore_case=False):
        """
        Construct a new :class:`BaseRegistry`.

        Args:
            ignore_case (bool): Whether or not to ignore case in object names?
        """
        ignore_case = bool(ignore_case)

        if ignore_case:
            self._norm_key = lambda s: str(s).lower()
        else:
            self._norm_key = lambda s: str(s)
        self._ignore_case = ignore_case
        self._name_and_objects = []
        self._key_to_object = {}

    @property
    def ignore_case(self):
        """Whether or not to ignore the case?"""
        return self._ignore_case

    def __iter__(self):
        return (n for n, o in self._name_and_objects)

    def register(self, name, obj):
        """
        Register an object.

        Args:
            name (str): Name of the object.
            obj: The object.
        """
        key = self._norm_key(name)
        if key in self._key_to_object:
            raise KeyError('Object already registered: {!r}'.format(name))
        self._key_to_object[key] = obj
        self._name_and_objects.append((name, obj))

    def get(self, name):
        """
        Get an object.

        Args:
            name (str): Name of the object.

        Returns:
            The retrieved object.

        Raises:
            KeyError: If `name` is not registered.
        """
        key = self._norm_key(name)
        if key not in self._key_to_object:
            raise KeyError('Object not registered: {!r}'.format(name))
        return self._key_to_object[key]


class ClassRegistry(BaseRegistry):
    """
    A subclass of :class:`BaseRegistry`, dedicated for classes.

    Usage::

        Class MyClass(object):

            def __init__(self, value, message):
                ...

        registry = ClassRegistry()
        registry.register('MyClass', MyClass)

        obj = registry.create_object('MyClass', 123, message='message')
    """

    def register(self, name, obj):
        if not isinstance(obj, six.class_types):
            raise TypeError('`obj` is not a class: {!r}'.format(obj))
        return super(ClassRegistry, self).register(name, obj)

    def construct(self, name, *args, **kwargs):
        """
        Construct an object according to class `name` and arguments.

        Args:
            name (str): Name of the class.
            *args: Arguments passed to the class constructor.
            \\**kwargs: Named arguments passed to the class constructor.

        Returns:
            The constructed object.

        Raises:
            KeyError: If `name` is not registered.
        """
        return self.get(name)(*args, **kwargs)
