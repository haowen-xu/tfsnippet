"""
This package provides shortcuts to utilities from second-level packages.
"""

from .dataflows import DataFlow, DataMapper, SlidingWindow
from .utils.config_utils import (Config, ConfigField, get_config_defaults,
                                 register_config_arguments)
from .utils.graph_keys import GraphKeys
from .utils.model_vars import model_variable, get_model_variables
from .utils.reuse import instance_reuse, global_reuse, VarScopeObject
from .utils.settings_ import settings

__all__ = [
    # from tfsnippet.dataflows
    'DataFlow', 'DataMapper', 'SlidingWindow',

    # from tfsnippet.utils.config_utils
    'Config', 'ConfigField',
    'get_config_defaults', 'register_config_arguments',

    # from tfsnippet.utils.graph_keys
    'GraphKeys',

    # from tfsnippet.utils.model_vars
    'model_variable', 'get_model_variables',

    # from tfsnippet.utils.reuse
    'instance_reuse', 'global_reuse', 'VarScopeObject',

    # from tfsnippet.utils.settings_
    'settings',
]
