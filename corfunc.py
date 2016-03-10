#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.fftpack import dct
from scipy.signal import argrelextrema
from numpy.linalg import lstsq


def porod(q, K, sigma):
    return (K*q**(-4))*np.exp(-q**2*sigma**2)


def guinier(q, A, B):
    return A*np.exp(B*q**2)


def fitguinier(q, iq):
    A = np.vstack([q**2, np.ones(q.shape)]).T
    return lstsq(A, np.log(iq))


def smooth(f, g, start, stop):
    def result_array(x):
        ys = np.zeros(x.shape)
        ys[x <= start] = f(x[x <= start])
        ys[x >= stop] = g(x[x >= stop])
        h = 1/(1+(x-stop)**2/(start-x)**2)
        mask = np.logical_and(x > start, x < stop)
        ys[mask] = h[mask]*g(x[mask])+(1-h[mask])*f(x[mask])
        return ys

    def result_scalar(x):
        return result_array(np.array([x]))[0]

    def result(x):
        if type(x) is np.ndarray:
            return result_array(x)
        else:
            return result_scalar(x)
    return result


def fit_data(q, iq):

    maxq = 0.04

    mask = q > maxq

    fitp = curve_fit(lambda q, k, sig: porod(q, k, sig)*q**2,
                     q[mask], iq[mask]*q[mask]**2)[0]

    data = interp1d(q, iq)
    s1 = smooth(data, lambda x: porod(x, fitp[0], fitp[1]), maxq, q[-1])

    minq = 0.0065*3
    mask = np.logical_and(q < minq, minq*0 < q)
    mask[0:6] = False

    g = fitguinier(q[mask], iq[mask])[0]

    s2 = smooth(lambda x: (np.exp(g[1]+g[0]*x**2)), s1, q[0], minq)

    return s2


def corr(f, background=None):
    orig = np.loadtxt(f, skiprows=1, dtype=np.float32)
    if background is None:
        back = np.zeros(orig.shape)[:,1]
    else:
        back = np.loadtxt(background, skiprows=1, dtype=np.float32)[:, 1]
    q = orig[:480, 0]
    iq = orig[:480, 1]
    iq -= back[:480]
    s2 = fit_data(q, iq)
    qs = np.arange(0, q[-1]*100, (q[1]-q[0]))
    iqs = s2(qs)*qs**2
    transform = dct(iqs)
    xs = np.pi*np.arange(len(qs))/(q[1]-q[0])/len(qs)
    return (xs, transform)


def extract(x, y):
    maxs = argrelextrema(y, np.greater)[0]  # A list of the maxima
    mins = argrelextrema(y, np.less)[0]  # A list of the minima

    # If there are no maxima, return NaN
    if len(maxs) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    GammaMin = y[mins[0]]  # The value at the first minimum

    ddy = (y[:-2]+y[2:]-2*y[1:-1])/(x[2:]-x[:-2])**2  # Second derivative of y
    dy = (y[2:]-y[:-2])/(x[2:]-x[:-2])  # First derivative of y
    # Find where the second derivative goes to zero
    zeros = argrelextrema(np.abs(ddy), np.less)[0]
    # locate the first inflection point
    linear_point = zeros[0]
    linear_point = int(mins[0]/10)

    m = np.mean(dy[linear_point-40:linear_point+40])  # Linear slope
    b = y[1:-1][linear_point]-m*x[1:-1][linear_point]  # Linear intercept

    Lc = (GammaMin-b)/m  # Hard block thickness

    # Create a fitted line through the linear section
    xs = np.linspace(0, x[mins[0]], 30)
    ys = m * xs + b

    #Find the data points where the graph is linear to within 1%
    mask = np.where(np.abs((y-(m*x+b))/y) < 0.01)[0]
    dtr = x[mask[0]]  # Beginning of Linear Section
    d0 = x[mask[-1]]  # End of Linear Section
    GammaMax = y[mask[-1]]
    A = -GammaMin/GammaMax  # Normalized depth of minimum

    return (x[mins[0]], x[maxs[0]], dtr, Lc, d0, A)

values = []
specs = []


def main(files, background=None, export=None, plot=False, save=None):
    import os.path

    for f in files:
        x, y = corr(f, background)
        plt.plot(x, y, label=os.path.basename(f))
        values.append(extract(x, y))
        specs.append(y)
        x0 = x
    plt.xlabel("Distance [nm]")
    plt.ylabel("Correlation")
    plt.legend()

    if plot:
        plt.show()
    elif save:
        plt.savefig(save)

    from math import isnan

    maxs = np.array([v[1] for v in values if not isnan(v[0])])
    dtrs = np.array([v[2] for v in values if not isnan(v[0])])
    lcs = np.array([v[3] for v in values if not isnan(v[0])])
    qs = np.array([v[4] for v in values if not isnan(v[0])])
    As = np.array([v[5] for v in values if not isnan(v[0])])

    print("Long Period")
    print("%f ± %f" % (np.median(maxs), np.max(np.abs(maxs-np.median(maxs)))))
    print("Average Hard Block Thickness")
    print("%f ± %f" % (np.median(lcs), np.max(np.abs(lcs-np.median(lcs)))))
    print("Average Interface Thickness")
    print("%f ± %f" % (np.median(dtrs), np.max(np.abs(dtrs-np.median(dtrs)))))
    print("Average Core Thickness ")
    print("%f ± %f" % (np.median(qs), np.max(np.abs(qs-np.median(qs)))))
    print("PolyDispersity")
    print("%f ± %f" % (np.median(As), np.max(np.abs(As-np.median(As)))))
    print("Filling Fraction")
    print("%f ± %f" % (np.median(lcs/maxs),
                       np.max(np.abs(lcs/maxs-np.median(lcs/maxs)))))

    if export:
        np.savetxt(export,
                   np.vstack([x0, specs]).T)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Perform correlation function analysis on scattering data')
    parser.add_argument('--background', action='store',
                        help='A background measurement for subtraction')
    parser.add_argument('--export', action='store',
                        help='Export the extracted real space data to a file')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--plot', action='store_true',
                       help='Display a plot of the correlation functions.')
    group.add_argument('--saveImage', action='store',
                       help='Save a plot to an image file.')

    parser.add_argument('FILE', nargs="+",
                        help='Scattering data in two column ascii format')
    args = parser.parse_args()

    main(args.FILE, args.background, args.export,
         args.plot, args.saveImage)
