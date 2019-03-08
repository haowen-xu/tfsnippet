from .config_utils import Config, ConfigField

__all__ = ['TFSnippetConfig', 'settings']


class TFSnippetConfig(Config):
    """Global configurations of TFSnippet."""

    enable_assertions = ConfigField(
        bool, default=True,
        description='Whether or not to enable assertions operations?'
    )
    check_numerics = ConfigField(
        bool, default=False,
        description='Whether or not to check numeric issues?'
    )
    auto_histogram = ConfigField(
        bool, default=False,
        description='Whether or not to automatically add histograms of layer '
                    'parameters and outputs to the collection '
                    '`tfsnippet.GraphKeys.AUTO_HISTOGRAM`?'
    )
    file_cache_checksum = ConfigField(
        bool, default=False,
        description='Whether or not to validate the checksum of cached files?'
    )


settings = TFSnippetConfig()
"""The TFSnippet global configuration object."""
