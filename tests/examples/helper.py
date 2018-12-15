import os
import unittest


def skipUnlessRunExamplesTests():
    return unittest.skipUnless(
        os.environ.get('RUN_EXAMPLES_TEST_CASE') == '1',
        'RUN_EXAMPLES_TEST_CASE is not set to 1, thus skipped'
    )
