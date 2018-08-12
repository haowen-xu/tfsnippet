import codecs
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

        # The ``env["MLSTORAGE_EXPERIMENT_ID"]`` would be set if the program
        # is run via `mlrun` from MLStorage.
        # See `MLStorage <https://github.com/haowen-xu/mlstorage>`_ for details.
        if os.environ.get('MLSTORAGE_EXPERIMENT_ID'):
            # save default config values to "config.defaults.json"
            default_config_path = os.path.abspath(
                os.path.join(os.getcwd(), 'config.defaults.json'))
            default_config_json = json.dumps(self.to_dict())
            with codecs.open(default_config_path, 'wb', 'utf-8') as f:
                f.write(default_config_json)

            # load user specified config from "config.json"
            config_path = os.path.abspath(
                os.path.join(os.getcwd(), 'config.json'))
            if os.path.isfile(config_path):
                with codecs.open(config_path, 'rb', 'utf-8') as f:
                    config_dict = json.load(f)
                if not isinstance(config_dict, dict):
                    raise ValueError('%s: expected a config dict, but got '
                                     '%r'.format(config_path, config_dict))
                self.from_dict(config_dict)

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
