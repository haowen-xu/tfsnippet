import unittest

from tfsnippet.examples.datasets import *
from tests.examples.helper import skipUnlessRunExamplesTests


class ExamplesDatasetsTestCase(unittest.TestCase):

    @skipUnlessRunExamplesTests()
    def test_examples_datasets_fetching(self):
        _ = load_mnist()
        _ = load_cifar10()
        _ = load_cifar100()
