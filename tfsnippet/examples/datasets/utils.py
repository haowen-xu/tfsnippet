import os

import six

if six.PY2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

__all__ = [
    'get_cache_dir',
    'cached_download',
]


def get_cache_dir(name, root_dir=None):
    """
    Get the cache directory for downloading a particular data set.

    Args:
        name (str): Name of the data set.
        root_dir (str): Root directory for cache.
            If not specified, will automatically choose one according to OS.

    Returns:
        str: Path of the data set cache directory.
    """
    if root_dir is None:
        root_dir = os.path.expanduser('~/.tfsnippet/datasets')
    return os.path.join(root_dir, name)


def cached_download(uri, cache_file):
    """
    Download ``uri`` with caching.

    Args:
        uri (str): URI to be downloaded.
        cache_file (str): Path of the cache file.

    Returns:
        str: The full path of the downloaded file.
    """
    cache_file = os.path.abspath(cache_file)
    if not os.path.isfile(cache_file):
        parent_dir = os.path.split(cache_file)[0]
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)

        tmp_file = '%s~' % cache_file
        try:
            urlretrieve(uri, tmp_file)
            os.rename(tmp_file, cache_file)
        finally:
            if os.path.isfile(tmp_file):
                os.remove(tmp_file)
    return cache_file
