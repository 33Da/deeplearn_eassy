'''
Module for generating normalized tight frames.
'''

import numpy as NP
import pylab, paddle

def _get_CS_even(d, n):
    assert n%2 == 0, n
    CS = NP.zeros((n, n), NP.float32)
    CS[0,:] = 1/NP.sqrt(2)
    for k in xrange(1, n/2):
        CS[2*k-1,:] = NP.cos(2*NP.pi*k*NP.arange(n)/n)
        CS[2*k,:] = NP.sin(2*NP.pi*k*NP.arange(n)/n)
    CS[n-1,:] = 1/NP.sqrt(2)
    CS[n-1,1::2] *= -1
    return CS

def _get_CS_odd(d, n):
    assert n%2 == 1, n
    CS = NP.zeros((n, n), NP.float32)
    CS[0,:] = 1/NP.sqrt(2)
    for k in xrange(n/2):
        CS[2*k+1,:] = NP.cos(2*NP.pi*(k+1)*NP.arange(n)/n)
        CS[2*k+2,:] = NP.sin(2*NP.pi*(k+1)*NP.arange(n)/n)
    return CS

def get(d, n, alpha=1.):
    '''
    Builds an alpha-tight frame with n elements for R^d,
    according to:
    G. Zimmermann, "Normalized Tight Frames in Finite Dimensions",
    in Recent Progress in Multivariate Approximation, 2001
    W. Haussmann, K. Jetter, and M. Reimer (eds.)
    International Series of Numerical Mathematics Vol. 17, pp. 249-252
    '''
    assert n >= d, 'n must be >= d (here n=%d and d=%d)' % (n, d)
    if n%2 == 0:
        CS = _get_CS_even(d, n)
    else:
        CS = _get_CS_odd(d, n)
    if d == n:
        B = CS
    elif d%2 == 0:
        B = CS[1:d+1]
    elif d%2 == 1:
        B = CS[:d]
    else:
        raise RuntimeError, 'something is amiss with dimensions'
    return NP.sqrt(2*alpha/float(n)) * B

def get_ntf(d, n):
    '''
    Builds a normalized tight frame with n elements for R^d.

    The NTF is generated as an alpha-tight frame with alpha=n/d,
    calling the function get.
    '''
    return get(d, n, alpha=float(n)/d)

if __name__=='__main__':

    A = get(12*12, 12*12)
    paddle._saveDict(A, None, Nrows = 12, Ncols = 12, path = './tightframe.png', sorted = False)
        
