from __future__ import division
import scipy.special as sf
import numpy
import itertools as iteration
from deco import *

"""
This module serves as the main component for computing the multivariate Fox-H function (MFHF). 
The implementation builds upon the foundational code presented in [1], which employs a simple 
rectangular approximation of the MFHF integrals, with significant enhancements. 
Specifically, the current version of the code enables the evaluation of MFHF when 
either 'm' or 'n' is included in the definition. Additionally, the implementation 
extends functionality to evaluate MFHF even when one or more of (a_p^(i), alpha_p^(i)) 
or (b_q^(i), beta_q^(i)) are empty. An example of using this module is provided 
in the script Hybrid_Capacity.py.


[1] H. R. Alhennawi, M. M. El Ayadi, M. H. Ismail, and H.-A. M. Mourad,
"Closed-form exact and asymptotic expressions for the symbol error
rate and capacity of the H-function fading channel," IEEE Transactions
on Vehicular Technology, vol. 65, no. 4, pp. 1957–1974, 2015.
"""


# Note: you need to adjust no_divisions, and tolerance, in case you don't get accurate results.

#@concurrent
def MultiFoxH_main(params, no_divisions, tolerance=0.000001):
    '''This module estimates a multivariate integral using simple rectangule
    quadrature. In most practical applications, 20 points per dimension provide
    sufficient accuracy.
    Inputs:
    'params': list containing z, mn, pq, a, b, c, d.
    'no_divisions': the number of divisions taken along each dimension. Note
    that the total number of points will be no_divisions**dim.
    'tolerance': tolerance used for determining the limits
    Output:
    'estim_output': the estimated value of the multivariate Fox H function...'''
    limits = rect_limits(params, tolerance)
    dim = limits.shape[0]
    signs = list(iteration.product([1, -1], repeat=dim))
    code = list(iteration.product(range(int(no_divisions / 2)), repeat=dim))
    quad = 0
    for sign in signs:
        points = numpy.array(sign) * (numpy.array(code) + 0.5) * limits * 2 / no_divisions
        quad += numpy.sum(Integrand(points, params))
    volume = numpy.prod(2 * limits / no_divisions)
    estim_output = quad * volume
    return estim_output


def Integrand(Y, params):
    ''' This module computes the complex integrand of the multivariate Fox-H
    function at the points given by the rows of the matrix Y.'''
    z, mn, pq, a, b, c, d = params
    m, n = zip(*mn)
    p, q = zip(*pq)
    no_points, r = Y.shape
    s = 1j * Y

    # Estimating sigma[i]
    lower_sig = numpy.zeros(r)
    upper_sig = numpy.zeros(r)

    ''' This part was modified to make the code is capable of evaluating multiFoxH (MFHFs) when one or
    more of (a_p^(i), alpha_p^(i)) or (b_q^(i), beta_q^(i)) in Eq.8 is empty.'''
    i = 0
    while i < r:
        if d[i] and m[i + 1] > 0:
            dj, Dj = zip(*d[i])
            dj = numpy.array(dj[:m[i + 1]])
            Dj = numpy.array(Dj[:m[i + 1]])
            lower_sig[i] = -numpy.min(dj / Dj)
        else:
            lower_sig[i] = -100
        if c[i] and n[i + 1] > 0:
            cj, Cj = zip(*c[i])
            cj = numpy.array(cj[:n[i + 1]])
            Cj = numpy.array(Cj[:n[i + 1]])
            upper_sig[i] = numpy.min((1 - cj) / Cj)
        else:
            upper_sig[i] = 1
        i += 1
    mindist = numpy.linalg.norm(upper_sig - lower_sig)
    sigs = 0.5 * (upper_sig + lower_sig)

    '''This part was added to evaluate MFHFs for 'm' not equal zero in the main integral solution Eq.8'''

    if m[0] > 0:
        for j in range(m[0]):
            num = b[j][0] + numpy.sum(b[j][1:] * upper_sig)
            bnorm = numpy.linalg.norm(b[j][1:])
            newdist = numpy.abs(num) / (bnorm + numpy.finfo(float).eps)
            if newdist < mindist:
                mindist = newdist
                sigs = upper_sig - 0.5 * num * numpy.array(b[j][1:]) / (bnorm * bnorm)
    else:
        for j in range(n[0]):
            num = 1 - a[j][0] - numpy.sum(a[j][1:] * lower_sig)
            anorm = numpy.linalg.norm(a[j][1:])
            newdist = numpy.abs(num) / (anorm + numpy.finfo(float).eps)
            if newdist < mindist:
                mindist = newdist
                sigs = lower_sig + 0.5 * num * numpy.array(a[j][1:]) / (anorm * anorm)

    s += sigs

    # Computing products of Gamma factors on both numerators and denomerator
    '''This part is updated to include the products of Gamma factors in case 'm' not equal zero Eq.8'''
    sum_s = numpy.c_[numpy.ones((no_points, 1)), s]
    num = denom = 1 + 0j
    for j in range(n[0]):
        num *= sf.gamma(1 - numpy.dot(sum_s, a[j]))
    for j in range(m[0]):
        num *= sf.gamma(numpy.dot(sum_s, b[j]))
    for j in range(m[0], q[0]):
        denom *= sf.gamma(1 - numpy.dot(sum_s, b[j]))
    for j in range(n[0], p[0]):
        denom *= sf.gamma(numpy.dot(sum_s, a[j]))
    for i in range(r):
        for j in range(n[i + 1]):
            num *= sf.gamma(1 - c[i][j][0] - c[i][j][1] * s[:, i])
        for j in range(m[i + 1]):
            num *= sf.gamma(d[i][j][0] + d[i][j][1] * s[:, i])
        for j in range(n[i + 1], p[i + 1]):
            denom *= sf.gamma(c[i][j][0] + c[i][j][1] * s[:, i])
        for j in range(m[i + 1], q[i + 1]):
            denom *= sf.gamma(1 - d[i][j][0] - d[i][j][1] * s[:, i])

    #integrand
    zs = numpy.power(z, -s)
    f = (num / denom) * numpy.prod(zs, axis=1) / (2 * numpy.pi) ** r
    # the complex j is not forgotten
    return f


def rect_limits(params, tolerance):
    '''This modules attempts to determine an appropriate  rectangular
    limits of the integration region of the multivariate Fox H function.'''
    limit_range = numpy.arange(0, 50, 0.05)
    r = len(params[0])
    limits = numpy.zeros(r)
    for i in range(r):
        points = numpy.zeros((limit_range.shape[0], r))
        points[:, i] = limit_range
        abs_integrand = numpy.abs(Integrand(points, params))
        index_max = numpy.max(numpy.nonzero(abs_integrand > tolerance * abs_integrand[0]))
        limits[i] = limit_range[index_max]
    return limits



