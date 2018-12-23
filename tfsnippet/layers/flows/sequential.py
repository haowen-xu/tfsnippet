from tfsnippet.utils import add_name_and_scope_arg_doc
from .base import BaseFlow, MultiLayerFlow

__all__ = ['SequentialFlow']


class SequentialFlow(MultiLayerFlow):
    """
    Manage a sequential list of :class:`Flow` instances.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, flows, name=None, scope=None):
        """
        Construct a new :class:`SequentialFlow`.

        Args:
            flows (Iterable[Flow]): The flow list.
        """
        flows = list(flows)  # type: list[BaseFlow]
        if not flows:
            raise TypeError('`flows` must not be empty.')

        print(flows)
        for i, flow in enumerate(flows):
            if not isinstance(flow, BaseFlow):
                raise TypeError('The {}-th flow in `flows` is not an instance '
                                'of `Flow`: {!r}'.format(i, flow))
        value_ndims = flows[0].value_ndims
        for i, flow in enumerate(flows[1:], 1):
            if flow.value_ndims != value_ndims:
                raise TypeError('`value_ndims` of the {}-th flow in `flows` '
                                'mismatch with the first flow: {} vs {}.'.
                                format(i, flow.value_ndims, value_ndims))

        super(SequentialFlow, self).__init__(
            n_layers=len(flows), value_ndims=value_ndims,
            dtype=flows[-1].dtype, name=name, scope=scope
        )
        self._flows = flows
        self._explicitly_invertible = all(
            flow.explicitly_invertible for flow in self._flows)

    @property
    def flows(self):
        """
        Get the immutable flow list.

        Returns:
            tuple[BaseFlow]: The immutable flow list.
        """
        return tuple(self._flows)

    @property
    def explicitly_invertible(self):
        return self._explicitly_invertible

    def _transform_layer(self, layer_id, x, compute_y, compute_log_det):
        flow = self._flows[layer_id]
        return flow.transform(x, compute_y, compute_log_det)

    def _inverse_transform_layer(self, layer_id, y, compute_x, compute_log_det):
        flow = self._flows[layer_id]
        return flow.inverse_transform(y, compute_x, compute_log_det)
