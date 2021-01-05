'''
Recovery experiment with learned dual dictionaries.
'''

import sys
import numpy as NP
import numpy.random as RA
import pylab

if __name__=='__main__':

    # parse the arguments
    if len(sys.argv) < 2:
        print 'usage: python %s path-to-dictionary-file' % sys.argv[0]
        sys.exit(0)

    # load the dictionaries
    npz = NP.load(sys.argv[1])
    D, C = npz['D'], npz['C']

    # generate some data with varying sparsity
    d, K = D.shape
    n = 50 # number of samples for each level of sparsity
    N = n*(K-1) # total number of samples
    U0 = RA.normal(size=(K, N))
    X = NP.zeros((d, N))
    for i in xrange(1, K):
        for j in xrange(n):
            U0[RA.permutation(K)[i:],j+(i-1)*n] = 0
    S = NP.where(NP.abs(U0)>0, 1, 0)
    s = NP.sum(S, 0)
    assert s.min() == 1, s
    assert s.max() == K-1, s

    # exact recovery
    X = NP.dot(D, U0)
    U = NP.dot(C, X)
    I = NP.argsort(NP.abs(U), 0)[::-1,:]
    r = []
    for i in xrange(1, K):
        cols = NP.indices((i, n))[1]
        r.append(S[I[:i,(i-1)*n:i*n], cols].astype(NP.float).mean())

    pylab.plot(NP.arange(1, K), r)
    pylab.show()

    # stable recovery
    
