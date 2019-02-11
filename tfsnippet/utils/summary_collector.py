from contextlib import contextmanager

import tensorflow as tf

from .config_utils import scoped_set_config
from .doc_utils import add_name_arg_doc
from .misc import ContextStack
from .settings_ import settings

__all__ = [
    'SummaryCollector', 'default_summary_collector',
    'add_summary', 'add_histogram',
]


class SummaryCollector(object):
    """
    Collecting summaries and histograms added by :func:`tfsnippet.add_summary`
    and :func:`tfsnippet.add_histogram`.  For example::

        collector = SummaryCollector()
        with collector.as_default():
            spt.add_summary(...)
        summary_op = collector.merge_summary()

    You may also use this collector to capture the auto histogram generated
    by layers from `tfsnippet.layers`, for example::

        collector = SummaryCollector()
        with collector.as_default(auto_histogram=True):
            y = spt.layers.dense(x, ...)
        summary_op = collector.merge_summary()
    """

    def __init__(self, collections=None, no_add_to_collections=False):
        """
        Construct a new :class:`SummaryCollector`.

        Args:
            collections (Iterable[str]): Also add the captured summaries to
                these TensorFlow graph collections.  If not specified, will
                not add the summaries to any collection.
            no_add_to_collections (bool): If :obj:`True`, will not add the
                captured to any collection, regardless of the `collections`
                argument in the constructor, or in :meth:`add_summary` and
                :meth:`add_histogram` method.
        """
        collections = tuple(collections or ())
        no_add_to_collections = bool(no_add_to_collections)

        self._collections = collections
        self._no_add_to_collections = no_add_to_collections
        self._summary_list = []

    @property
    def collections(self):
        """Get the summary collections."""
        return self._collections

    @property
    def summary_list(self):
        """
        Get the list of captured summaries.
        """
        return self._summary_list

    def merge_summary(self):
        """
        Merge all the captured summaries into one operation.

        Returns:
            tf.Operation or None: The merged operation, or None if no
                summary has been captured.
        """
        if self._summary_list:
            return tf.summary.merge(self._summary_list)

    @contextmanager
    def as_default(self, auto_histogram=None):
        """
        Push this :class:`SummaryCollector` to the top of context stack,
        and enter a scoped context.

        Args:
            auto_histogram (bool): If specified, set the config value of
                `tfsnippet.settings.auto_histogram` within the context.

        Yields:
            This summary collector object.
        """
        @contextmanager
        def set_auto_histogram():
            if auto_histogram is not None:
                with scoped_set_config(settings, auto_histogram=auto_histogram):
                    yield
            else:
                yield

        _summary_collect_stack.push(self)
        try:
            with set_auto_histogram():
                yield self
        finally:
            _summary_collect_stack.pop()

    def add_summary(self, summary, collections=None):
        """
        Add the summary to this collector and to `collections`.

        Args:
            summary: TensorFlow summary tensor.
            collections: Also add the summary to these collections.
                Defaults to `self.collections`.

        Returns:
            The `summary` tensor.
        """
        self._summary_list.append(summary)
        if not self._no_add_to_collections:
            collections = (self._collections if collections is None
                           else tuple(collections))
            for c in collections:
                tf.add_to_collection(c, summary)
        return summary

    @add_name_arg_doc
    def add_histogram(self, tensor, summary_name=None, strip_scope=False,
                      collections=None, name=None):
        """
        Add the histogram of `tensor` to this collector and to `collections`.

        Args:
            tensor: Take histogram of this tensor.
            summary_name: Specify the summary name for `tensor`.
            strip_scope: If :obj:`True`, strip the name scope from `tensor.name`
                when adding the histogram.
            collections: Also add the histogram to these collections.
                Defaults to `self.collections`.

        Returns:
            The serialized histogram tensor of `tensor`.
        """
        tensor = tf.convert_to_tensor(tensor)

        with tf.name_scope(name, default_name='SummaryCollector.add_histogram',
                           values=[tensor]):
            if summary_name is None:
                summary_name = tensor.name
                summary_name = summary_name.replace(':', '_')
                if summary_name.endswith('_0'):
                    summary_name = summary_name[:-2]
                if strip_scope:
                    summary_name = summary_name.rsplit('/', 1)[-1]
            histogram = tf.summary.histogram(
                summary_name, tensor, collections=[])

        self.add_summary(histogram, collections)
        return histogram


def default_summary_collector():
    """
    Get the :class:`SummaryCollector` object at the top of context stack.

    Returns:
        SummaryCollector: The summary collector.
    """
    return _summary_collect_stack.top()


def add_summary(summary, collections=None):
    """
    Add the summary to the default summary collector, and to `collections`.

    Args:
        summary: TensorFlow summary tensor.
        collections: Also add the summary to these collections.
            Defaults to `self.collections`.

    Returns:
        The `summary` tensor.
    """
    c = default_summary_collector()
    return c.add_summary(summary, collections=collections)


def add_histogram(tensor, summary_name=None, strip_scope=False,
                  collections=None, name=None):
    """
    Add the histogram of `tensor` to the default summary collector,
    and to `collections`.

    Args:
        tensor: Take histogram of this tensor.
        summary_name: Specify the summary name for `tensor`.
        strip_scope: If :obj:`True`, strip the name scope from `tensor.name`
            when adding the histogram.
        collections: Also add the histogram to these collections.
            Defaults to `self.collections`.

    Returns:
        The serialized histogram tensor of `tensor`.
    """
    return default_summary_collector().add_histogram(
        tensor=tensor, summary_name=summary_name, strip_scope=strip_scope,
        name=name, collections=collections
    )


_summary_collect_stack = ContextStack(
    initial_factory=lambda: SummaryCollector(
        # the default summary collector adds summaries to SUMMARIES collection
        collections=(tf.GraphKeys.SUMMARIES,),
        no_add_to_collections=False
    )
)
