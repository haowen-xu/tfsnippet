from tfsnippet.utils import CacheDir

__all__ = ['TFInception']


class TFInception(object):
    """

    """

    def __init__(self):
        self._model_dir = CacheDir('inception-2015-12-05')

    def _fetch_model(self):
        return self._model_dir.download_and_extract(
            'http://download.tensorflow.org/models/image/imagenet/'
            'inception-2015-12-05.tgz'
        )
