#!/usr/bin/python

import unittest
from hypothesis import given, assume
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

    @given(A=floats(min_value=1e-100, max_value=1e30),
           B=floats(min_value=-1e2, max_value=-1e-5))
    def test_sane_fit(self, A, B):
        x = np.linspace(0, 1, 71)
        y = guinier(x, A, B)

        g = fitguinier(x, y)[0]

        self.assertAlmostEqual(B/g[0], 1.0)
        self.assertAlmostEqual(A/np.exp(g[1]), 1.0)

    @given(start=floats(min_value=0,max_value=1e50),
           stop=floats(max_value=1e50))
    def test_smooth(self, start, stop):
        assume(stop > start + 1e-50)
        f = lambda x: np.sqrt(x)*np.sin(x/10)
        g = lambda x: np.log(1+x)
        s = smooth(f, g, start, stop)

        x = np.linspace(start/2, stop*2, 100)
        fg = np.vstack([f(x), g(x)])
        small = np.min(fg, axis=0)
        large = np.max(fg, axis=0)

        self.assertTrue(np.all(small <= s(x)))
        self.assertTrue(np.all(s(x) <= large))
        self.assertEqual(s(start/2), f(start/2))
        self.assertEqual(s(start), f(start))
        self.assertEqual(s(stop), g(stop))
        self.assertEqual(s(stop*2), g(stop*2))


if __name__ == '__main__':
    unittest.main()
