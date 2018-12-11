from .base import Flow, MultiLayerFlow

__all__ = ['SequentialFlow']


class SequentialFlow(MultiLayerFlow):
    """
    Manage a sequential list of :class:`Flow` instances.
    """

    def __init__(self, flows, name=None, scope=None):
        """
        Construct a new :class:`SequentialFlow`.

        Args:
            flows (Iterable[Flow]): The flow list.
        """
        flows = list(flows)  # type: list[Flow]
        if not flows:
            raise TypeError('`flows` must not be empty.')
        for i, flow in enumerate(flows):
            if not isinstance(flow, Flow):
                raise TypeError('The {}-th item in `flows` is not an instance '
                                'of `Flow`: {!r}'.format(i, flow))
        super(SequentialFlow, self).__init__(
            n_layers=len(flows), dtype=flows[-1].dtype, name=name, scope=scope)
        self._flows = flows
        self._explicitly_invertible = all(
            flow.explicitly_invertible for flow in self._flows)

    @property
    def flows(self):
        """
        Get the immutable flow list.

        Returns:
            tuple[Flow]: The immutable flow list.
        """
        return tuple(self._flows)

    @property
    def explicitly_invertible(self):
        return self._explicitly_invertible

    def _create_layer_params(self, layer_id):
        pass

    def _transform_layer(self, layer_id, x, compute_y, compute_log_det):
        flow = self._flows[layer_id]
        return flow.transform(x, compute_y, compute_log_det)

    def _inverse_transform_layer(self, layer_id, y, compute_x, compute_log_det):
        flow = self._flows[layer_id]
        return flow.inverse_transform(y, compute_x, compute_log_det)
