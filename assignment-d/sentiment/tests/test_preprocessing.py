import unittest
import numpy.testing as nptest
import numpy as np

from .. import preprocessing


class TestPreprocessing(unittest.TestCase):

    def test_add_padding(self):
        
        inp = np.array([
            ['a', 'b'],
            ['c'],
            ['d', 'e', 'f', 'g'],
            ['g']
        ])

        expected = np.array([
            ['a', 'b', '', ''],
            ['c', '', '', ''],
            ['d', 'e', 'f', 'g'],
            ['g', '', '', '']
        ])

        nptest.assert_array_equal(preprocessing.add_padding(inp, dtype=expected.dtype), expected)

    def test_tokenize(self):

        inp = np.array(['a b', 'c d e'])

        expected = np.array([
            ['a', 'b'],
            ['c', 'd', 'e']
        ])

        nptest.assert_array_equal(preprocessing.tokenize(inp), expected)
