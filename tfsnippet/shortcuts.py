"""
This package provides shortcuts to utilities from second-level packages.
"""

from .dataflows import *
from .utils.reuse import *

__all__ = [
    # from tfsnippet.dataflows
    'DataFlow', 'DataMapper', 'SlidingWindow',

    # from tfsnippet.utils.reuse
    'get_reuse_stack_top', 'instance_reuse', 'global_reuse',
    'VarScopeObject'
]
