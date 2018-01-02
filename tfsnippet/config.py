from collections import OrderedDict

import six

__all__ = ['Config']


class Config(object):
    """Class of configuration values."""

    _config_initialized = False

    def __init__(self):
        """Construct the :class:`Config`."""
        self._float_x = 'float32'
        self._config_initialized = True

    def __setattr__(self, key, value):
        if not key.startswith('_') and not hasattr(self, key):
            raise AttributeError('Unknown config attribute: {}'.format(key))
        super(Config, self).__setattr__(key, value)

    def to_dict(self):
        """Get all config values as an ordered dict."""
        return OrderedDict([
            (k[1:], getattr(self, k))
            for k in sorted(six.iterkeys(self.__dict__))
            if k.startswith('_') and k != '_config_initialized'
        ])

    def from_dict(self, config_dict):
        """
        Set the config values via a dict.

        Args:
            config_dict: Config values dict.

        Raises:
            AttributeError: If an unknown config attribute is specified in
                `config_dict`.
        """
        for k, v in six.iteritems(config_dict):
            if k.startswith('_') or not hasattr(self, k):
                raise AttributeError('Unknown config attribute: {}'.format(k))
            setattr(self, k, v)

    @property
    def float_x(self):
        """Default data type of floating numbers."""
        return self._float_x

    @float_x.setter
    def float_x(self, value):
        if value not in {'float16', 'float32', 'float64'}:
            raise ValueError('Unknown floatx type: {}'.format(value))
        self._float_x = value


config = Config()
