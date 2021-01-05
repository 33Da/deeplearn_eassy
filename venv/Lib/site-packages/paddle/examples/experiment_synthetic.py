'''
This is the script to run the experiments reported in section 4.1 of
the techical report:

C.Basso, M.Santoro, A.Verri and S.Villa. "PADDLE: Proximal Algorithm for
Dual Dictionaries LEarning", DISI-TR-2010-XX, 2010.
'''

#import numpy as NP
import scipy as sp
from scipy import linalg as la
from paddle import dual, tframes, common
import pylab, os

def principal_angle(A, B):
    """
    Returns the largest angle between the linear subspaces spanned by
    the columns of 'A' and 'B'.
    'A' and 'B' must already be orthogonal matrices.
    """
    svd = la.svd(sp.dot(sp.transpose(A), B))
    return sp.arccos(min(svd[1].min(), 1.0))


def add_noise(X, level):
    noise = sp.random.normal(0, level, size=X.shape)
    SNR = la.norm(X) / la.norm(noise)
    print 'SNR (dB) = %d' % (20*sp.log10(SNR),)
    return X + noise

def run_experiment_1(d, d0, N, K, noise, exp1fn):

    # generates a normalized basis with d0 elements
    B = sp.random.uniform(size=(d, d0))
    B /= sp.sqrt(sp.sum(B**2, 0)).reshape((1, -1))
    # generates data (zero centered)
    X = sp.random.normal(size=(d0, N))
    X -= sp.mean(X, 1).reshape((-1, 1))
    X = sp.dot(B, X)
    print 'Generated %d examples with dimension %d and using %d basis' % (N, d, d0)
    # add noise
    X = add_noise(X, noise)
    # computes principal directions
    ew, ev = la.eigh(sp.dot(X, X.T)/N)
    # checks they are sorted
    assert sp.all(sp.argsort(ew) == sp.arange(d)), ew
    # picks the first K
    ev = ev[:,-K:]
    ew = ew[-K:]
    # PCA reconstruction error
    Erec_pca = la.norm(X - sp.dot(ev, sp.dot(ev.T, X)))/la.norm(X)

    # this is a function called at each iteration to compute the
    # metrics of interest
    def test_callback(D, C, U, X):
        # (relative) reconstruction/synthesis error
        Erec = la.norm(X - sp.dot(D, sp.dot(C, X)))/la.norm(X)
        # (relative) encoding/analysis error
        Ecod = la.norm(U - sp.dot(C, X))/la.norm(U)
        # ...
        nC = sp.sqrt(sp.sum(C*C, 1))
        nD = sp.sqrt(sp.sum(D*D, 0))
        Etrn = (1-sp.absolute(sp.sum(D*C.T, 0)/(nD*nC))).mean()
        # principal angle between the true subspace and the recovered dictionary
        angle = principal_angle(la.qr(D, econ=True)[0], ev)
        return Erec, Ecod, Etrn, angle

    pars = {
        'tau' : 0, # sparsity coefficient
        'mu'  : 1e-5, # l2 regularization
        'eta' : 1, # coding/decoding weight
        'maxiter' : 500,
        'minused' : 1,
        'verbose': False,
        'rtol': 1.e-8,
        }

    # initialize the variables
    D0, C0, U0 = dual.init(X, K, det=False)
    # learn the dictionaries
    D, C, U, full_out = dual.learn(X, D0, C0, U0, callable=test_callback, **pars)
    # here we take the results of the callback and put them in an array
    call_res = sp.array(full_out['call_res']).T
    # saves the result in a file
    sp.savez(exp1fn, X=X, D=D, C=C, U=U,
             call_res=call_res, Erec_pca=Erec_pca)

def plot_figure_1(call_res, figfn, Erec_pca):

    Erec, Etrn = call_res[0], call_res[2]

    fonts = {'fontsize': 20}

    fig = pylab.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(Erec, 'k-', lw=1.5, label='Rec. error')
    ax1.plot(Etrn, 'k--', lw=1.5, label='Duality')
    ax1.hlines([Erec_pca,], 0, len(Erec)-1, lw=2., linestyle='dashdot', label='PCA error')
    ax1.set_xlabel('Iteration', fontdict=fonts)
    ax1.set_ylabel('Error', fontdict=fonts)
    ax1.semilogy()
    ax1.set_yticks((0.01, 0.02, 0.1, 1))
    ax1.set_yticklabels(('1%', '2%', '10%', '100%'), fontdict=fonts)
    pylab.legend()

    pylab.savefig(figfn, dpi=300, transparent=True, bbox_inches='tight')

def run_experiment_2(d, k, N, K, noise, exp2fn):

    # build a normalized tight frame
    A = tframes.get(d, K)
    # generate the coefficients of the atoms combinations
    Utrue = sp.zeros((K, N), sp.float32)
    for i in xrange(N):
        Utrue[sp.random.permutation(K)[:k],i] = sp.random.uniform(-1, 1, size=(k,))
    # generates the data
    X = sp.dot(A, Utrue)
    # add noise
    X = add_noise(X, noise)

    # this is a function called at each iteration to compute the
    # metrics of interest
    def test_callback(D, C, U, X):
        # reconstruction error
        Erec = la.norm(X - sp.dot(D, sp.dot(C, X))) / la.norm(X)
        # duality error 
        Einv = la.norm(sp.identity(d) - sp.dot(D, C))/d
        # ....
        nC = sp.sqrt(sp.sum(C*C, 1))
        nD = sp.sqrt(sp.sum(D*D, 0))
        Etrn = (1 - sp.absolute(sp.sum(D*C.T, 0)/(nD*nC))).mean()
        return Erec, Einv, Etrn

    pars = {
        'tau' : .5, # sparsity coefficient
        'mu'  : 1.e-8, # l2 regularization
        'eta' : 1, # coding/decoding weight
        'maxiter' : 50,
        'minused' : 10,
        'verbose': False,
        'rtol': 1.e-8,
        }

    # recovery by IST with true dictionary
    U0 = sp.zeros((K, N), sp.float32)
    Uopt, full_out = dual._ist(X, U0, A, A.T, pars, maxiter=1000)
    # this is the optimal reconstruction that can be achieved
    Erec_ntf = la.norm(X - sp.dot(A, Uopt)) / la.norm(X)

    # initialize the variables
    D0, C0, U0 = dual.init(X, K, det=False)
    # learn the dictionaries
    D, C, U, full_out = dual.learn(X, D0, C0, U0, callable=test_callback, **pars)
    # here we take the results of the callback and put them in an array
    call_res = sp.array(full_out['call_res']).T
    # saves the result in a file
    sp.savez(exp2fn, X=X, D=D, C=C, U=U, A=A,
             call_res=call_res, Erec_ntf=Erec_ntf)

def plot_figure_2(call_res, figfn, Erec_opt):

    Erec, Etrn = call_res[0], call_res[2]

    fonts = {'fontsize': 20}

    fig = pylab.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(Erec, 'k-', lw=1.5, label='Rec. error')
    ax1.plot(Etrn, 'k--', lw=1.5, label='Duality')
    ax1.hlines([Erec_opt,], 0, len(Erec)-1, linestyle='dashdot', label='Optimal')
    ax1.set_xlabel('Iteration', fontdict=fonts)
    ax1.set_ylabel('Error', fontdict=fonts)
    ax1.semilogy()
    ax1.set_xlim((0, 100))
    ax1.set_ylim((0.001, 1))

    ax1.set_yticks((0.001, 0.01, 0.1, 1))
    ax1.set_yticklabels(('0.1%', '1%', '10%', '100%'), fontdict=fonts)

    pylab.legend()
    pylab.savefig(figfn, dpi=300, transparent=True, bbox_inches='tight')
    
if __name__=='__main__':

    d = 25       # dimensionality of the data
    N = 10000    # number of training vectors
    k1 = 15      # number of non-zero components for each generated vector, first experiment
    K1 = 15      # number of dictionary atoms in the first experiment
    k2 = 3       # number of non-zero components for each generated vector, second experiment
    K2 = 2*d     # number of dictionary atoms in the second experiment
    noise = 0.02 # noise level

    runExp1 = True
    runExp2 = True

    ########### synthetic experiment 1

    print 

    exp1fn = 'exp_synth_fig1.npz' # file storing the results of the first experiment

    if runExp1:

        if not os.access(exp1fn, os.R_OK):

            run_experiment_1(d, k1, N, K1, noise, exp1fn)

        else:

            print '[!!] loading results of experiment #1 from file %s' % exp1fn
            print '[!!] delete the file is you want to repeat the experiment'

        assert os.access(exp1fn, os.R_OK), exp1fn
        npz = sp.load(exp1fn)
        figfn = 'exp_synth_fig1.png'
        plot_figure_1(npz['call_res'], figfn, npz['Erec_pca'])
                
    ########### synthetic experiment 2

    print

    exp2fn = 'exp_synth_fig2.npz' # file storing the results of the first experiment

    if runExp2:

        if not os.access(exp2fn, os.R_OK):

            run_experiment_2(d, k2, N, K2, noise, exp2fn)

        else:

            print '[!!] loading results of experiment #2 from file %s' % exp2fn
            print '[!!] delete the file is you want to repeat the experiment'

        assert os.access(exp2fn, os.R_OK), exp2fn
        npz = sp.load(exp2fn)
        figfn = 'exp_synth_fig2.png'
        plot_figure_2(npz['call_res'], figfn, npz['Erec_ntf'])

        # saves the original and recovered dictionaries
        A = npz['A']
        D = npz['D']
        C = npz['C']

        # try to re-arrange D and C before saving
        for i in xrange(D.shape[1]):
            diff = sp.absolute(sp.sum(A[:,i,sp.newaxis]*D[:,i:], 0))
            order = sp.argsort(diff)[::-1] + i
            D = sp.concatenate((D[:,:i], D[:,order]), 1)
            C = sp.concatenate((C[:i], C[order]), 0)
            if sp.sum(A[:,i]*D[:,i]) < 0:
                D[:,i] *= -1
                C[i] *= -1
                
        common._saveDict(A, None, Nrows = 5, Ncols = 10, path = './tightframeA.png', sorted = False)
        common._saveDict(D, None, Nrows = 5, Ncols = 10, path = './tightframeD.png', sorted = False)
        common._saveDict(C.T, None, Nrows = 5, Ncols = 10, path = './tightframeCt.png', sorted = False)
        pylab.matshow(sp.absolute(sp.sum(A[:,:,sp.newaxis]*D[:,sp.newaxis,:], 0)))


