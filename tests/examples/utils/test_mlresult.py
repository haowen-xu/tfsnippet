import os
import unittest

import numpy as np

from tfsnippet.examples.utils import MLResults
from tfsnippet.utils import TemporaryDirectory


def head_of_file(path, n):
    with open(path, 'rb') as f:
        return f.read(n)


class MLResultTestCase(unittest.TestCase):

    def test_imwrite(self):
        with TemporaryDirectory() as tmpdir:
            results = MLResults(tmpdir)
            im = np.zeros([32, 32], dtype=np.uint8)
            im[16:, ...] = 255

            results.imwrite('test.bmp', im)
            file_path = os.path.join(tmpdir, 'test.bmp')
            self.assertTrue(os.path.isfile(file_path))
            self.assertEqual(head_of_file(file_path, 2), b'\x42\x4d')

            results.imwrite('test.png', im)
            file_path = os.path.join(tmpdir, 'test.png')
            self.assertTrue(os.path.isfile(file_path))
            self.assertEqual(head_of_file(file_path, 8),
                             b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a')

            results.imwrite('test.jpg', im)
            file_path = os.path.join(tmpdir, 'test.jpg')
            self.assertTrue(os.path.isfile(file_path))
            self.assertEqual(head_of_file(file_path, 3), b'\xff\xd8\xff')
