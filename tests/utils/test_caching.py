import hashlib
import mimetypes
import os
import shutil
import socket
import unittest
from contextlib import contextmanager
from threading import Thread

import six
import pytest
from mock import mock

from tfsnippet.utils import *

if six.PY2:
    from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
    from io import BytesIO as LogIO
else:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from io import StringIO as LogIO


def get_asset_path(name):
    return os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        'assets',
        name
    )


def summarize_dir(path):
    def read_file(file_path):
        with open(file_path, 'rb') as f:
            return f.read()
    return sorted([
        (name, read_file(os.path.join(path, name)))
        for name in iter_files(path)
    ])


PAYLOAD_CONTENT = [
    ('a/1.txt', b'a/1.txt'),
    ('b/2.txt', b'b/2.txt'),
    ('c.txt', b'c.txt'),
]


class AssetsHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        asset_file = get_asset_path(self.path.lstrip('/'))
        if not os.path.isfile(asset_file):
            self.send_error(404, 'Not Found')
        else:
            self.send_response(200)
            self.send_header('Content-type', mimetypes.guess_type(asset_file))
            self.send_header('Content-Length', os.stat(asset_file).st_size)
            self.send_header('Connection', 'close')
            self.end_headers()
            with open(asset_file, 'rb') as f:
                shutil.copyfileobj(f, self.wfile)
            self.server.counter[0] += 1
        return


@contextmanager
def set_cache_root_var(value):
    import tfsnippet.utils.caching
    old_value = tfsnippet.utils.caching._cache_root
    try:
        if value is None:
            tfsnippet.utils.caching._cache_root = None
        else:
            set_cache_root(value)
        yield
    finally:
        tfsnippet.utils.caching._cache_root = old_value


@contextmanager
def set_cache_root_env(value):
    key = 'TFSNIPPET_CACHE_ROOT'
    old_value = os.environ.get(key)
    try:
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value
        yield
    finally:
        if old_value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = old_value


def get_free_port():
    s = socket.socket(socket.AF_INET, type=socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    address, port = s.getsockname()
    s.close()
    return port


@contextmanager
def assets_server():
    port = get_free_port()
    server = HTTPServer(('127.0.0.1', port), AssetsHTTPRequestHandler)
    server.counter = [0]
    background_thread = Thread(target=server.serve_forever)
    background_thread.daemon = True
    background_thread.start()
    try:
        yield server, 'http://127.0.0.1:{}/'.format(port)
    finally:
        server.server_close()


# patch the extractor for tests
class PatchedExtractor(Extractor):
    call_count = 0

    @staticmethod
    def open(file_path):
        PatchedExtractor.call_count += 1
        return Extractor.open(file_path)


class CacheRootSettingsTestCase(unittest.TestCase):

    def test_get_cache_root(self):
        with TemporaryDirectory() as tmpdir, set_cache_root_env(None):
            self.assertEqual(
                os.path.expanduser('~/.tfsnippet/cache'),
                get_cache_root()
            )
            with set_cache_root_env(tmpdir):
                self.assertEqual(tmpdir, get_cache_root())

    def test_set_cache_root(self):
        with TemporaryDirectory() as tmpdir, \
                set_cache_root_env(None), set_cache_root_var(None):
            self.assertNotEqual(tmpdir, get_cache_root())
            with set_cache_root_var(tmpdir):
                self.assertEqual(tmpdir, get_cache_root())
            self.assertNotEqual(tmpdir, get_cache_root())


class CacheDirTestCase(unittest.TestCase):

    def test_basic(self):
        cache_dir = CacheDir('sub-dir/sub-sub-dir', cache_root=None)
        self.assertEqual('sub-dir/sub-sub-dir', cache_dir.name)
        self.assertEqual(os.path.abspath(get_cache_root()),
                         cache_dir.cache_root)
        self.assertEqual(os.path.abspath(os.path.join(get_cache_root(),
                                                      'sub-dir/sub-sub-dir')),
                         cache_dir.path)

        with TemporaryDirectory() as tmpdir:
            cache_dir = CacheDir('sub-dir/sub-sub-dir', cache_root=tmpdir)

            self.assertEqual('sub-dir/sub-sub-dir', cache_dir.name)
            self.assertEqual(tmpdir, cache_dir.cache_root)
            self.assertEqual(os.path.join(tmpdir, 'sub-dir/sub-sub-dir'),
                             cache_dir.path)
            self.assertEqual(os.path.join(cache_dir.path, 'a/b/c'),
                             cache_dir.resolve('a/b/c'))

        with pytest.raises(ValueError, match='`name` is required'):
            _ = CacheDir('')

    def test_download(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = CacheDir('sub-dir', cache_root=tmpdir)
            with assets_server() as (server, url):
                self.assertEqual(0, server.counter[0])

                # no cache
                path = cache_dir.download(url + 'payload.zip',
                                          show_progress=True)
                self.assertEqual(
                    os.path.join(cache_dir.path, 'payload.zip'), path)
                self.assertTrue(os.path.isfile(path))
                self.assertFalse(os.path.isfile(path + '._downloading_'))
                self.assertEqual(1, server.counter[0])

                # having cache
                path = cache_dir.download(url + 'payload.zip',
                                          show_progress=True)
                self.assertEqual(
                    os.path.join(cache_dir.path, 'payload.zip'), path)
                self.assertTrue(os.path.isfile(path))
                self.assertFalse(os.path.isfile(path + '._downloading_'))
                self.assertEqual(1, server.counter[0])

                # no cache, because of filename mismatch
                out_file = LogIO()
                path = cache_dir.download(url + 'payload.zip',
                                          filename='sub-dir/payload2.zip',
                                          show_progress=False,
                                          progress_file=out_file)
                self.assertEqual(
                    os.path.join(cache_dir.path, 'sub-dir/payload2.zip'), path)
                self.assertTrue(os.path.isfile(path))
                self.assertFalse(os.path.isfile(path + '._downloading_'))
                self.assertEqual(2, server.counter[0])
                self.assertEqual(
                    out_file.getvalue(),
                    'Downloading ' + url + 'payload.zip ... ok\n'
                )

                # test download error
                out_file = LogIO()
                with pytest.raises(Exception, match='404'):
                    _ = cache_dir.download(url + 'not-exist.zip',
                                           progress_file=out_file)
                path = os.path.join(cache_dir.path, 'not-exist.zip')
                self.assertFalse(os.path.isfile(path))
                self.assertFalse(os.path.isfile(path + '._downloading_'))
                self.assertEqual(
                    out_file.getvalue(),
                    'Downloading ' + url + 'not-exist.zip ... error\n'
                )

    def test_download_validate_hash(self):
        def compute_hash(hasher, path):
            with open(path, 'rb') as f:
                hasher.update(f.read())
            return hasher.hexdigest()

        with TemporaryDirectory() as tmpdir:
            cache_dir = CacheDir('sub-dir', cache_root=tmpdir)
            payload_zip_md5 = compute_hash(
                hashlib.md5(), get_asset_path('payload.zip'))
            payload_tar_sha1 = compute_hash(
                hashlib.sha1(), get_asset_path('payload.tar'))

            with assets_server() as (server, url):
                # test md5 okay
                path = cache_dir.download(url + 'payload.zip',
                                          show_progress=True,
                                          hasher=hashlib.md5(),
                                          expected_hash=payload_zip_md5)
                cache_path = os.path.join(cache_dir.path, 'payload.zip')
                self.assertEqual(path, cache_path)
                self.assertTrue(os.path.isfile(cache_path))
                self.assertEqual(compute_hash(hashlib.md5(), cache_path),
                                 payload_zip_md5)

                # test md5 mismatch
                with pytest.raises(IOError,
                                   match='Hash not match for file downloaded '
                                         'from {}payload.tar'.format(url)):
                    _ = cache_dir.download(url + 'payload.tar',
                                           show_progress=True,
                                           hasher=hashlib.md5(),
                                           expected_hash=payload_tar_sha1)
                cache_path = os.path.join(cache_dir.path, 'payload.tar')
                self.assertFalse(os.path.isfile(cache_path))

                # test sha1 okay
                path = cache_dir.download(url + 'payload.tar',
                                          show_progress=True,
                                          hasher=hashlib.sha1(),
                                          expected_hash=payload_tar_sha1)
                self.assertEqual(path, cache_path)
                self.assertTrue(os.path.isfile(cache_path))
                self.assertEqual(compute_hash(hashlib.sha1(), cache_path),
                                 payload_tar_sha1)

                # test validate cached file
                with scoped_set_config(settings, file_cache_checksum=True):
                    path = cache_dir.download(url + 'payload.tar',
                                              hasher=hashlib.sha1(),
                                              expected_hash=payload_tar_sha1)

                    with open(path, 'wb') as f:
                        f.write(b'12345')

                    with pytest.raises(IOError, match='Hash not match for '
                                                      'cached file'):
                        _ = cache_dir.download(url + 'payload.tar',
                                               hasher=hashlib.sha1(),
                                               expected_hash=payload_tar_sha1)

    @mock.patch('tfsnippet.utils.caching.Extractor', PatchedExtractor)
    def test_extract_file(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = CacheDir('sub-dir', cache_root=tmpdir)
            # old_open = Extractor.open
            # Extractor.open = Mock(wraps=old_open)
            self.assertEqual(0, PatchedExtractor.call_count)

            # no cache
            log_file = LogIO()
            path = cache_dir.extract_file(get_asset_path('payload.tar.gz'),
                                          progress_file=log_file,
                                          show_progress=True)
            self.assertEqual(1, PatchedExtractor.call_count)
            self.assertEqual(
                os.path.join(cache_dir.path, 'payload'), path)
            self.assertListEqual(PAYLOAD_CONTENT, summarize_dir(path))
            self.assertFalse(os.path.isdir(path + '._extracting_'))
            self.assertEqual(
                'Extracting {} ... ok\n'.format(
                    get_asset_path('payload.tar.gz')),
                log_file.getvalue()
            )

            # having cache
            log_file = LogIO()
            path = cache_dir.extract_file(get_asset_path('payload.tgz'),
                                          progress_file=log_file,
                                          show_progress=True)
            self.assertEqual(1, PatchedExtractor.call_count)
            self.assertEqual(
                os.path.join(cache_dir.path, 'payload'), path)
            self.assertListEqual(PAYLOAD_CONTENT, summarize_dir(path))
            self.assertFalse(os.path.isdir(path + '._extracting_'))
            self.assertEqual('', log_file.getvalue())

            # no cache, because of extract_dir mismatch
            log_file = LogIO()
            path = cache_dir.extract_file(get_asset_path('payload.tar.bz2'),
                                          extract_dir='sub-dir/payload2',
                                          show_progress=False,
                                          progress_file=log_file)
            self.assertEqual(2, PatchedExtractor.call_count)
            self.assertEqual(
                os.path.join(cache_dir.path, 'sub-dir/payload2'), path)
            self.assertListEqual(PAYLOAD_CONTENT, summarize_dir(path))
            self.assertFalse(os.path.isdir(path + '._extracting_'))
            self.assertEqual(
                'Extracting {} ... ok\n'.format(
                    get_asset_path('payload.tar.bz2')),
                log_file.getvalue()
            )

            # test extracting error
            err_archive = os.path.join(cache_dir.path, 'invalid.txt')
            with open(err_archive, 'wb') as f:
                f.write(b'not a valid archive')
            log_file = LogIO()
            with pytest.raises(Exception):
                _ = cache_dir.extract_file(
                    err_archive, progress_file=log_file, show_progress=True)
            path = os.path.join(cache_dir.path, 'invalid')
            self.assertFalse(os.path.isdir(path))
            self.assertFalse(os.path.isdir(path + '._extracting_'))
            self.assertEqual(
                'Extracting {} ... error\n'.format(err_archive),
                log_file.getvalue()
            )

    def test_download_and_extract_and_purge_all(self):
        with TemporaryDirectory() as tmpdir:
            cache_dir = CacheDir('sub-dir', cache_root=tmpdir)
            with assets_server() as (server, url):
                # download and extract
                path = cache_dir.download_and_extract(url + 'payload.zip')
                self.assertEqual(
                    os.path.join(cache_dir.path, 'payload'), path)
                self.assertFalse(os.path.isfile(path + '.zip'))
                self.assertFalse(os.path.isfile(path + '._downloading_'))
                self.assertTrue(os.path.isdir(path))
                self.assertFalse(os.path.isdir(path + '._extracting_'))
                self.assertListEqual(PAYLOAD_CONTENT, summarize_dir(path))

                # purge all
                self.assertTrue(os.path.isdir(cache_dir.path))
                cache_dir.purge_all()
                self.assertFalse(os.path.isdir(cache_dir.path))
