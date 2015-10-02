#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import quad,romb
from scipy.fftpack import dct,dst
from scipy.signal import argrelextrema
from numpy.linalg import lstsq

def porod(q,K,sigma):
    return (K*q**(-4))*np.exp(-q**2*sigma**2)
def guinier(q,A,B):
    return A*np.exp(B*q**2)
def vonk(q,A,B):
    temp = A-B*q**(-2)
    temp[np.logical_not(np.isfinite(temp))] = A
    #print(temp)
    return temp
def fitguinier(q,iq):
    A = np.vstack([q**2,np.ones(q.shape)]).T
    return lstsq(A,np.log(iq))
def fitvonk(q,iq):
    A = np.vstack([q**2,np.ones(q.shape)]).T
    return lstsq(A,iq*q**2)[0]
def smooth(f,g,start,stop):
    def result_array(x):
        ys = np.zeros(x.shape)
        ys[x<=start] = f(x[x<=start])
        ys[x>=stop] = g(x[x>=stop])
        #k = (x-start)/(stop-start)
        #h = np.exp(-1/k)/(np.exp(-1/k)+np.exp(-1/(1-k)))
        h = 1/(1+(x-stop)**2/(start-x)**2)
        mask = np.logical_and(x>start,x<stop)
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



def fit_data(q,iq):

    maxq = 0.04

    mask = q>maxq

    fitp = curve_fit(lambda q,k,sig: porod(q,k,sig)*q**2,q[mask],iq[mask]*q[mask]**2)[0]
    fitq = np.arange(maxq,10*maxq,q[1]-q[0])

    data=interp1d(q,iq)
    s1 = smooth(data,lambda x:porod(x,fitp[0],fitp[1]),maxq,q[-1])

    minq = 0.0065*3
    mask = np.logical_and(q<minq,minq*0<q)
    mask[0:6] = False

    fitg = curve_fit(lambda q,a,b:vonk(q,a,b)*q**2,q[mask],iq[mask]*q[mask]**2)[0]
    print(fitg)
    v = fitvonk(q[mask],iq[mask])
    g = fitguinier(q[mask],iq[mask])[0]
    print(g)
    fitq = np.arange(0,minq,q[1]-q[0])

    s2 = smooth(lambda x:(np.exp(g[1]+g[0]*x**2)),s1,q[0],minq)
    qs = np.arange(0,q[-1]*5,(q[1]-q[0]))

    # plt.plot(q,iq*q**2,"r.")
    # plt.plot(qs,s2(qs)*qs**2)

    # plt.yscale('log')
    # plt.show()

    return s2

def corr(f,background=None):
    orig = np.loadtxt(f,skiprows=1,dtype=np.float32)
    if background is None:
        back=np.zeros(orig.shape)
    else:
        back = np.loadtxt(background,skiprows=1,dtype=np.float32)
    q = orig[:480,0]
    iq = orig[:480,1]#-back[:480,1]
    s2 = fit_data(q,iq)
    qs = np.arange(0,q[-1]*100,(q[1]-q[0]))
    iqs = s2(qs)*qs**2
    transform = dct(iqs)
    xs = np.pi*np.arange(len(qs))/(q[1]-q[0])/len(qs)
    return (xs,transform)

def sq(f):
    orig = np.loadtxt(f,skiprows=1,dtype=np.float32)
    q = orig[:240,0]
    iq = orig[:240,1]
    s2 = fit_data(q,iq)
    qs = np.arange(0,q[-1]*100,(q[1]-q[0]))
    iqs = s2(qs)*qs**2
    return (q,iq)

def extract(x,y):
    maxs = argrelextrema(y,np.greater)[0]
    mins = argrelextrema(y,np.less)[0]

    if len(maxs) == 0:
        return (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
    print(maxs)
    Lp = x[maxs[0]] # First maximum
    GammaMin = y[mins[0]]

    ddy = (y[:-2]+y[2:]-2*y[1:-1])/(x[2:]-x[:-2])**2
    dy = (y[2:]-y[:-2])/(x[2:]-x[:-2])
    zeros = argrelextrema(np.abs(ddy),np.less)[0]
    linear_point = zeros[0]
    linear_point = int(mins[0]/10)
    #plt.plot(ddy)
    #plt.show()

    print(linear_point*100)
    m = np.mean(dy[linear_point-40:linear_point+40])
    b = y[1:-1][linear_point]-m*x[1:-1][linear_point]

    Lc = (GammaMin-b)/m
    Q = b
    plt.axhline(GammaMin)
    plt.axhline(0)

    xs = np.linspace(0,x[mins[0]],30)
    ys = m*xs+b
    
    plt.plot(xs,ys)

    mask = np.where(np.abs((y-(m*x+b))/y) < 0.01)[0]
    dtr = x[mask[0]]
    d0 = x[mask[-1]]
    GammaMax = y[mask[-1]]
    A = -GammaMin/GammaMax
    wc = -A / (-A + Q)

    return (x[mins[0]], x[maxs[0]], dtr, Lc, d0, A)

values = []
specs = []

# for i in [22, 23, 27]:  # 9, 13, 14, 22, 23, 27, 36, 37, 38, 41, 42, 43]:
for i in [18]: #LS Beetle
# for i in range(313,322): # LS in capillary
# for i in range(333,342): # Lactator Beetle in capillary
    if i == 139:
        continue
    # x, y = corr("/home/adam/Science/sax-data/Cyphochilus/" +
    #             "CY15_saxs_%05d_0001.dat" % i,
    #             "/home/adam/Science/sax-data/Cyphochilus/" +
    #             "CY15_saxs_%05d_0001.dat" % 15)
    x, y = corr("/home/adam/Science/sax-data/lsbeetle/" +
                "LSB_saxs_%05d_0001.dat" % i,
                "/home/adam/Science/sax-data/lsbeetle/" +
                "LSB_saxs_%05d_0001.dat" % 3)
    # x, y = corr("/home/adam/Science/sax-data/static/" +
    #             "JL_saxs_%05d_0001.dat" % i,
    #             "/home/adam/Science/sax-data/static/" +
    #             "JL_saxs_%05d_0001.dat" % 1)
    style = "k-"
    leg = "Black"
    if i % 10 != 0:
        leg = None
    plt.plot(x, y, style, label=leg)
    values.append(extract(x, y))
    specs.append(y/y[0])
    print(len(y))
# plt.yscale("log")
plt.xlabel("Distance [nm]")
plt.ylabel("Correlation")
plt.xlim(0, 750)
plt.ylim(-500, 950)
plt.legend()
plt.show()

from math import isnan

mins = np.array([x[0] for x in values if not(isnan(x[0]))])
maxs = np.array([x[1] for x in values if not(isnan(x[0]))])
dtrs = np.array([x[2] for x in values if not(isnan(x[0]))])
lcs  = np.array([x[3] for x in values if not(isnan(x[0]))])
qs  = np.array([x[4] for x in values if not(isnan(x[0]))])
As = np.array([x[5] for x in values if not(isnan(x[0]))])
#gmins= np.array([x[6] for x in values])
#gmaxs= np.array([x[7] for x in values])

maxs[maxs>1000] = np.nan
#lcs[mins>300] = np.nan
#mins[mins>300] = np.nan

x = np.linspace(1,22,len(maxs))
x -= 1

#plt.axvline(x[45],c="purple")
#plt.axvline(x[80],c="blue")

plt.xlabel("Feather Position (mm)")

#plt.subplot(1,2,2)
#plt.plot(mins, label="First Minimum")
p1=plt.plot(x,maxs,"-b",lw=4, label="First Maximum")
print(maxs)
plt.ylabel("Long Period (nm)")

ax2 = plt.twinx()
ax2.set_ylabel("Width (nm)")

p2=ax2.plot(x,lcs,"-r",lw=4, label="lc")
print(lcs)
#ax2.plot(lcs, label="lc")
#plt.plot(qs, label="Q")
#plt.plot(As, label="A")
#plt.plot(gmins, label="gmin")
#plt.plot(gmaxs, label="gmax")

plt.legend(p1+p2,["Long Period","Hard Block Thickness"],loc=2)
#plt.subplot(1,3,3)
#plt.imshow(np.vstack(specs).T)

plt.xlim(0,21)

plt.show()

print("Minimum")
print("%f ± %f" % (np.median(mins), np.max(np.abs(mins-np.median(mins)))))
print("Long Period")
print("%f ± %f" % (np.median(maxs), np.max(np.abs(maxs-np.median(maxs)))))
print("Average Interface Thickness")
print("%f ± %f" % (np.median(dtrs), np.max(np.abs(dtrs-np.median(dtrs)))))
print("Average Hard Block Thickness")
print("%f ± %f" % (np.median(lcs), np.max(np.abs(lcs-np.median(lcs)))))
print("Average Core Thickness ")
print("%f ± %f" % (np.median(qs), np.max(np.abs(qs-np.median(qs)))))
print("PolyDispersity")
print("%f ± %f" % (np.median(As), np.max(np.abs(As-np.median(As)))))
print("Filling Fraction")
print("%f ± %f" % (np.median(lcs/maxs), np.max(np.abs(lcs/maxs-np.median(lcs/maxs)))))
