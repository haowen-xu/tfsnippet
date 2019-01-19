import unittest

import tfsnippet as spt


class TFSnippetConfigTestCase(unittest.TestCase):

    def test_tfsnippet_config(self):
        self.assertTrue(spt.settings.enable_assertions)
        self.assertFalse(spt.settings.check_numerics)
