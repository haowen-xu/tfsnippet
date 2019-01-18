"""
This package provides shortcuts to utilities from second-level packages.
"""

from .dataflows import *
from .utils.reuse import *
from .utils.config_utils import *

__all__ = [
    # from tfsnippet.dataflows
    'DataFlow', 'DataMapper', 'SlidingWindow',

    # from tfsnippet.utils.reuse
    'get_reuse_stack_top', 'instance_reuse', 'global_reuse',
    'VarScopeObject',

    # from tfsnippet.utils.config
    'Config', 'ConfigField',
    'config_defaults', 'register_config_arguments',
]
