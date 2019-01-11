import random

import numpy as np
import tensorflow as tf

__all__ = ['set_random_seed', 'VarScopeRandomState']


def set_random_seed(seed):
    """
    Initialize the random modules of Python standard lib, NumPy, TensorFlow
    and TFSnippet, with a single seed.

    Args:
        seed (int): The seed used to generate the separated seeds for
            all concerning modules.
    """
    def next_seed():
        return np.random.randint(0xffffffff)

    np.random.seed(seed)
    random.seed(next_seed())
    tf.set_random_seed(next_seed())
    VarScopeRandomState.set_global_seed(next_seed())
    np.random.seed(next_seed())


class VarScopeRandomState(np.random.RandomState):
    """
    A sub-class of :class:`np.random.RandomState`, which uses a variable-scope
    dependent seed.  It is guaranteed for a :class:`VarScopeRandomState`
    initialized with the same global seed and variable scopes with the same
    name to produce exactly the same random sequence.
    """

    _global_seed = 0

    def __init__(self, variable_scope):
        vs_name = variable_scope.name
        seed = (self._global_seed & 0xfffffff) ^ (hash(vs_name) & 0xffffffff)
        super(VarScopeRandomState, self).__init__(seed=seed)

    @classmethod
    def set_global_seed(cls, seed):
        """
        Set the global random seed for all new :class:`VarScopeRandomState`.

        If not set, the default global random seed is `0`.

        Args:
            seed (int): The global random seed.
        """
        cls._global_seed = int(seed)
