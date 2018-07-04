import json
import os

import six

__all__ = ['Config']


class Config(object):
    """
    Base class for defining a set of configuration.

    One usage pattern of :class:`Config` is to initialize it via the
    constructor::

        config = Config(
            max_epoch=10,
            initial_learning_rate=0.001
        )

    Another usage pattern is to derive a subclass of :class:`Config`::

        class ExpConfig(Config):
            max_epoch = 10
            initial_learning_rate = 0.001

        config = ExpConfig()
    """

    def __init__(self, **kwargs):
        """
        Initialize the configuration values.

        Args:
            \**kwargs: The initial configuration values to set.
        """
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)

        # Load config from ``env["MLTOOLKIT_EXPERIMENT_CONFIG"]`` if presents.
        # The ``env["MLTOOLKIT_EXPERIMENT_CONFIG"]`` would be set if the
        # program is run via `mlrun` from MLToolkit.
        # See `MLToolkit <https://github.com/haowen-xu/mltoolkit>`_ for details.
        self.from_env('MLTOOLKIT_EXPERIMENT_CONFIG')

    def to_dict(self):
        """
        Seal all configuration values into a dict.

        All public, non-callable attributes and properties of this object
        will be regarded as configuration values.

        Returns:
            dict: The configuration dict.
        """
        ret = {}
        for k in dir(self):
            if not k.startswith('_'):
                v = getattr(self, k)
                if not callable(v):
                    ret[k] = v
        return ret

    def from_dict(self, config_dict, reject_new_keys=True):
        """
        Load configuration values from `config_dict`.

        Args:
            config_dict (dict): The source configuration dict.
            reject_new_keys (bool): Whether new configuration keys should
                be rejected? (default :obj:`True`)

        Raises:
            KeyError: If new configuration key presents.
        """
        for k, v in six.iteritems(config_dict):
            if reject_new_keys and not hasattr(self, k):
                raise KeyError('Unexpected configuration key: {!r}'.
                               format(k))
            setattr(self, k, v)

    def from_env(self, env_name, reject_new_keys=True):
        """
        Load configuration values from ``env[env_key]``.

        Args:
            env_name (str): Name of the environmental variable to load.
            reject_new_keys (bool): Whether new configuration keys should
                be rejected? (default :obj:`True`)

        Raises:
            ValueError: If ``env[env_key]`` presents but is not a JSON object.
            KeyError: If new configuration key presents.
        """
        config_json = os.environ.get(env_name, None)
        if config_json:
            config_dict = json.loads(config_json)
            if not isinstance(config_dict, dict):
                raise ValueError('env[{!r}] is not a JSON object: {!r}'.
                                 format(env_name, config_json))
            self.from_dict(config_dict, reject_new_keys=reject_new_keys)
