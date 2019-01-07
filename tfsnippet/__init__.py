__version__ = '0.2.0-alpha.1'


from . import (dataflows, datasets, distributions, layers, ops,
               preprocessing, scaffold, trainer, utils, variational,
               bayes, reuse, stochastic)
from .dataflows import *
from .distributions import *
from .scaffold import *
from .trainer import *
from .variational import *
from .bayes import *
from .reuse import *
from .stochastic import *


def _exports():
    exports = [
        # export modules
        'dataflows', 'datasets', 'distributions', 'layers', 'ops',
        'preprocessing', 'scaffold', 'trainer', 'utils', 'variational',
        'bayes', 'reuse', 'stochastic',

        # export classes and functions
        'DataFlow',
    ]

    # recursively export classes and functions
    for pkg in (distributions, scaffold, trainer, variational, bayes,
                reuse, stochastic):
        exports += list(pkg.__all__)

    # remove `_exports` from root namespace
    import sys
    del sys.modules[__name__]._exports

    return exports


__all__ = _exports()
