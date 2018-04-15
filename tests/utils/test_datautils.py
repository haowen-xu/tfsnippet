import unittest

from tfsnippet.utils import minibatch_slices_iterator


class MiniBatchSlicesIteratorTestCase(unittest.TestCase):

    def test_minibatch_slices_iterator(self):
        self.assertEqual(
            list(minibatch_slices_iterator(0, 10, False)),
            []
        )
        self.assertEqual(
            list(minibatch_slices_iterator(9, 10, False)),
            [slice(0, 9, 1)]
        )
        self.assertEqual(
            list(minibatch_slices_iterator(10, 10, False)),
            [slice(0, 10, 1)]
        )
        self.assertEqual(
            list(minibatch_slices_iterator(10, 9, False)),
            [slice(0, 9, 1), slice(9, 10, 1)]
        )
        self.assertEqual(
            list(minibatch_slices_iterator(10, 9, True)),
            [slice(0, 9, 1)]
        )


if __name__ == '__main__':
    unittest.main()
