import os
import unittest


def skipUnlessRunDatasetsTests():
    return unittest.skipUnless(
        os.environ.get('RUN_DATASETS_TEST_CASE') == '1',
        'RUN_DATASETS_TEST_CASE is not set to 1, thus skipped'
    )
