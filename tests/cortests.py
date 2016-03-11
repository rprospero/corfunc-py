#!/usr/bin/python

import unittest
import numpy as np
from corfunc import porod, guinier, fitguinier


class TestStringMethods(unittest.TestCase):
    def test_porod(self):
        self.assertEqual(porod(1, 1, 0), 1)

    def test_guinier(self):
        self.assertEqual(guinier(1, 1, 0), 1)

    def test_sane_fit(self):
        A = np.pi
        B = -np.sqrt(2)
        x = np.linspace(0, 1, 71)
        y = guinier(x, A, B)

        g = fitguinier(x, y)[0]

        self.assertAlmostEqual(B, g[0])
        self.assertAlmostEqual(A, np.exp(g[1]))

if __name__ == '__main__':
    unittest.main()
