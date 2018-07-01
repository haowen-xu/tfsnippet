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
        config.from_env()

    Another usage pattern is to derive a subclass of :class:`Config`::

        class ExpConfig(Config):
            max_epoch = 10
            initial_learning_rate = 0.001

        config = ExpConfig()
        config.from_env()
    """

    def __init__(self, **kwargs):
        """
        Initialize the configuration values.

        Args:
            \**kwargs: The initial configuration values to set.
        """
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)

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
                raise KeyError('Unexpected configuration key from '
                               'env["MLTOOLKIT_EXPERIMENT_CONFIG"]: {!r}'.
                               format(k))
            setattr(self, k, v)

    def from_env(self, reject_new_keys=True):
        """
        Load configuration values from ``env["MLTOOLKIT_EXPERIMENT_CONFIG"]``.

        The ``env["MLTOOLKIT_EXPERIMENT_CONFIG"]`` would be set if the program
        is run via `mlrun` from MLToolkit. See
        `MLToolkit <https://github.com/haowen-xu/mltoolkit>`_ for details.

        Args:
            reject_new_keys (bool): Whether new configuration keys should
                be rejected? (default :obj:`True`)

        Raises:
            ValueError: If ``env["MLTOOLKIT_EXPERIMENT_CONFIG"]`` presents
                but is not a JSON object.
            KeyError: If new configuration key presents.
        """
        config_json = os.environ.get('MLTOOLKIT_EXPERIMENT_CONFIG', None)
        if config_json:
            config_dict = json.loads(config_json)
            if not isinstance(config_dict, dict):
                raise ValueError('env["MLTOOLKIT_EXPERIMENT_CONFIG"] is not a '
                                 'JSON object: {!r}'.format(config_json))
            self.from_dict(config_dict, reject_new_keys=reject_new_keys)
