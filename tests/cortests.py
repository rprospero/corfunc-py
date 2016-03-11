#!/usr/bin/python

import unittest
from corfunc import porod


class TestStringMethods(unittest.TestCase):
    def test_porod(self):
        self.assertEqual(porod(1,1,0), 1)

if __name__ == '__main__':
    unittest.main()
