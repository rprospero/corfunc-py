#!/usr/bin/python

import unittest
from hypothesis import given
from hypothesis.strategies import floats
import numpy as np
from corfunc import porod, guinier, fitguinier, smooth


class TestStringMethods(unittest.TestCase):
    def test_porod(self):
        self.assertEqual(porod(1, 1, 0), 1)

    @given(q=floats(min_value=0, max_value=1e3),
           A=floats(min_value=0))
    def test_guinier(self, q, A):
        self.assertEqual(guinier(q, A, 0), A)

    def test_sane_fit(self):
        A = np.pi
        B = -np.sqrt(2)
        x = np.linspace(0, 1, 71)
        y = guinier(x, A, B)

        g = fitguinier(x, y)[0]

        self.assertAlmostEqual(B, g[0])
        self.assertAlmostEqual(A, np.exp(g[1]))

    def test_smooth(self):
        f = lambda x: np.sqrt(x)*np.sin(x/10)
        g = lambda x: np.log(1+x)
        s = smooth(f, g, 25, 75)

        x = np.linspace(0, 1, 100)
        fg = np.vstack([f(x), g(x)])
        small = np.min(fg, axis=0)
        large = np.max(fg, axis=0)

        x = np.linspace(0, 1, 100)

        self.assertTrue(np.all(small <= s(x)))
        self.assertTrue(np.all(s(x) <= large))
        self.assertEqual(s(0), f(0))
        self.assertEqual(s(25), f(25))
        self.assertEqual(s(75), g(75))
        self.assertEqual(s(100), g(100))


if __name__ == '__main__':
    unittest.main()
