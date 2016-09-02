#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.fftpack import dct
from scipy.signal import argrelextrema
from numpy.linalg import lstsq


# Pretend Python allows for anonymous classes
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def porod(q, K, sigma):
    """Calculate the Porod region of a curve"""
    return (K*q**(-4))*np.exp(-q**2*sigma**2)


def guinier(q, A, B):
    """Calculate the Guinier region of a curve"""
    return A*np.exp(B*q**2)


def fitguinier(q, iq):
    """Fit the Guinier region of a curve"""
    A = np.vstack([q**2, np.ones(q.shape)]).T
    return lstsq(A, np.log(iq))

def fitcylinder(q, iq):
    """Fit to an infinite cylinder at low q"""
    A = np.vstack([1/q]).T
    return lstsq(A,iq)


# FIXME: result_array is recalculated on every call, when it should
# just be cached on creation.  Closure is a poor man's object and all
# that.
def smooth(f, g, start, stop):
    """Interpolate from curve f to curve g over the range from start to stop"""
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


def fit_data(model, q, iq, qrange):
    """Given a data set, extrapolate out to large q with Porod
    and to q=0 with Guinier"""

    minq, maxq = qrange

    mask = q > maxq

    fitp = curve_fit(lambda q, k, sig: porod(q, k, sig)*q**2,
                     q[mask], iq[mask]*q[mask]**2)[0]

    data = interp1d(q, iq)
    s1 = smooth(data, lambda x: porod(x, fitp[0], fitp[1]), maxq, q[-1])

    mask = np.logical_and(q < minq, 0 < q)
    # mask[0:6] = False

    if model=="guinier":
        g = fitguinier(q[mask], iq[mask])[0]
        s2 = smooth(lambda x: (np.exp(g[1]+g[0]*x**2)), s1, q[0], minq)
    elif model=="cylinder":
        g = fitcylinder(q[mask],iq[mask])[0]
        s2 = smooth(lambda x: g[0] / x, s1, q[0], minq)
    else:
        print("lowq-model must either be guinier or cylinder")

    return s2


def corr(model, f, qrange, background=None):
    """Transform a scattering curve into a correlation function"""
    orig = np.loadtxt(f, skiprows=1, dtype=np.float32)
    if background is None:
        back = np.zeros(orig. shape)[:, 1]
    else:
        back = np.loadtxt(background, skiprows=1, dtype=np.float32)[:, 1]
    q = orig[:480, 0]
    iq = orig[:480, 1]
    iq -= back[:480]
    s2 = fit_data(q, iq, qrange)
    qs = np.arange(0, q[-1]*100, (q[1]-q[0]))
    iqs = s2(qs)*qs**2
    transform = dct(iqs)
    xs = np.pi*np.arange(len(qs))/(q[1]-q[0])/len(qs)
    return (xs, transform)


def extract(x, y):
    """Extract the interesting measurements from a correlation function"""
    maxs = argrelextrema(y, np.greater)[0]  # A list of the maxima
    mins = argrelextrema(y, np.less)[0]  # A list of the minima

    # If there are no maxima, return NaN
    garbage = Struct(minimum=np.nan,
                     maximum=np.nan,
                     dtr=np.nan,
                     Lc=np.nan,
                     d0=np.nan,
                     A=np.nan)
    if len(maxs) == 0:
        return garbage
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

    # Find the data points where the graph is linear to within 1%
    mask = np.where(np.abs((y-(m*x+b))/y) < 0.01)[0]
    if len(mask) == 0:  # Return garbage for bad fits
        return garbage
    dtr = x[mask[0]]  # Beginning of Linear Section
    d0 = x[mask[-1]]  # End of Linear Section
    GammaMax = y[mask[-1]]
    A = -GammaMin/GammaMax  # Normalized depth of minimum

    return Struct(minimum=x[mins[0]],
                  maximum=x[maxs[0]],
                  dtr=dtr,
                  Lc=Lc,
                  d0=d0,
                  A=A)

values = []
specs = []


def main(files, qrange, model="guinier", background=None, export=None, plot=False, save=None):
    """Load a set of intensity curves and gathers the relevant statistics"""
    import os.path

    for f in files:
        x, y = corr(model, f, qrange, background)
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

    maxs = np.array([v.maximum for v in values if not isnan(v.minimum)])
    dtrs = np.array([v.dtr for v in values if not isnan(v.minimum)])
    lcs = np.array([v.Lc for v in values if not isnan(v.minimum)])
    qs = np.array([v.d0 for v in values if not isnan(v.minimum)])
    As = np.array([v.A for v in values if not isnan(v.minimum)])

    def printWithError(title, values):
        print(title)
        print("%f ± %f" % (np.median(values),
                            np.max(np.abs(values-np.median(values)))))

    printWithError("Long Period", maxs)
    printWithError("Average Hard Block Thickness", lcs)
    printWithError("Average Interface Thickness", dtrs)
    printWithError("Average Core Thickness ", qs)
    printWithError("PolyDispersity", As)
    printWithError("Filling Fraction", lcs/maxs)

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
    parser.add_argument('--minq', default  = 0.0065 * 3, type=float,
                        help="Minimum Q")
    parser.add_argument('--maxq', default  = 0.04, type=float,
                        help="Maximum Q")
    parser.add_argument('--lowq-model', type=str, default="guinier",
                        help="What model to use for low q.  The current options are guinier and cylinder")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--plot', action='store_true',
                       help='Display a plot of the correlation functions.')
    group.add_argument('--saveImage', action='store',
                       help='Save a plot to an image file.')

    parser.add_argument('FILE', nargs="+",
                        help='Scattering data in two column ascii format')
    args = parser.parse_args()

    main(args.FILE, (args.minq, args.maxq),
         args.lowq_model,
         args.background, args.export,
         args.plot, args.saveImage)
