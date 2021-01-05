'''
Implementation of tight-frames (TF) version of PADDLE.
'''

import time
import numpy as NP
import numpy.random as RA
from scipy import linalg as LA
from scipy import stats
import scipy
import pylab
from common import _cost_rec, _replaceAtoms, _saveDict, print_frame_assessment
from prox import _st
    
def _frame_potential(D, pars):
    #return ((U - NP.dot(C, X))**2).sum()
    #return pars['eta']*(NP.dot(D.T, D)**2).mean()
    d, K = D.shape
    f = 2*float(K)/d
    return pars['eta']*((NP.dot(D, D.T)-f*NP.identity(d))**2).mean()

def _cost(X, U, D, pars):
    full_out = {
        'rec_err': _cost_rec(D, X, U, pars),
        'fp': _frame_potential(D, pars),
        'l1_pen': 2*pars['tau'] * NP.abs(U).mean(),
        'l2_pen': pars['mu'] * (U**2).mean(),
        }
    E = full_out['rec_err'] + full_out['fp'] + full_out['l1_pen'] + full_out['l2_pen']
    return E, full_out

def _cost_fast(X, U, D, pars):
    '''
    Computes the same cost function as in _cost, but exploits a precomputed
    CX and returns just the value of the whole cost function.
    '''
    #return ((X - NP.dot(D, U))**2).mean() + pars['eta']*(NP.dot(D.T, D)**2).mean() + 2*pars['tau'] * NP.abs(U).mean() + pars['mu'] * (U**2).mean()
    d, K = D.shape
    f = 2*float(K)/d
    return ((X - NP.dot(D, U))**2).mean() + pars['eta']*((NP.dot(D, D.T)-f*NP.identity(d))**2).mean() + 2*pars['tau'] * NP.abs(U).mean() + pars['mu'] * (U**2).mean()

def _ist(X, U0, D, pars, maxiter=1000):
    '''
    Iterative soft-thresholding with FISTA acceleration.

    Minimization of :math:`\\frac{1}{d}\|X-DU\|_F^2+\\frac{2\\tau}{K}\|U\|_1` wrt :math:`U`, that is the well-known LASSO.

    Although nearly equivalent to calling :func:`paddle.dual._ist` with :math:`\eta=0`, the cost function is different because of the (constant) term from
    the frame potential.

    The function is used by :func:`paddle.tight.learn` for the optimization
    wrt ``U``.


    Parameters
    ----------
    X : (d, N) ndarray
        Data matrix
    U0 : (K, N) ndarray
        Initial value of the unknown 
    D : (d, K) ndarray
        Dictionary
    pars : dict
        Optional parameters
    maxiter : int
        Maximum number of iterations allowed (default 500)
        
    Returns
    -------
    U : (K, N) ndarray
        Optimal value of the unknown
    full_out : dict
        Full output
    '''
    ######## gets the dimensions
    d, N = X.shape
    K = D.shape[1]
    ######## initial evaluation of the cost function
    E, full_out = _cost(X, U0, D, pars)
    print '   initial energy = %.5e' % E
    if pars['verbose']:
        print '   iter   energy   avg|upd|'
    ######## fixed-step sigma
    DtD = NP.dot(D.T, D)
    # computes only the largest eigenvalue
    ewmax = LA.eigvalsh(DtD, eigvals=(K-1, K-1)) 
    sigma0 = ewmax/(N*d) + (pars['mu'] + pars['eta'])/(N*K)
    ######## FISTA initialization
    U = U0.copy()
    t = 1 # used in FISTA
    Y = U.copy() # needed by FISTA
    ####### pre-computes what is needed in the loop
    DtD /= (N*d*sigma0)
    DtX = NP.dot(D.T, X) / (N*d*sigma0)
    # constant part of the gradient descent step
    U0 = DtX
    # first-order part of the gradient descent step
    f = (1 - pars['mu']/(N*K*sigma0))
    A = DtD - f*NP.identity(K)
    ####### begin loop
    for i in xrange(maxiter):
        # compute gradient step and soft-thresholding
        Unew = _st(U0 - NP.dot(A, Y), pars['tau']/(N*K*sigma0))
        # evaluates the cost function
        E_new = _cost_fast(X, Unew, D, pars)
        if i%1 == 0 and pars['verbose']:
            upd = NP.sum((Unew-U)**2, 0)/NP.sum(U**2, 0)
            upd = upd[NP.isfinite(upd)]
            print '   %4d  %8.5e     %4.2f' % (i, E_new, upd.mean())
        # FISTA from Beck and Teboulle, SIAM J. Imaging Science, 2009
        tnew = (1 + NP.sqrt(1 + 4*(t**2))) / 2
        Y = Unew + ((t-1)/tnew)*(Unew - U)
        t = tnew
        U = Unew
        # check convergence
        if NP.abs(E-E_new)/E < pars['rtol']:
            break
        E = E_new
    ####### final evaluation of the cost function
    print '   energy after %d iter. = %.5e' % (i+1, E)
    dummy, full_out = _cost(X, Unew, D, pars)
    return U, full_out

def _pgd(D0, X, U, XUt, UUt, pars, maxiter=500, bound=1):
    '''
    Projected gradient descent with FISTA acceleration.

    Minimization of :math:`\|X-DU\|_F^2+\\frac{\eta}{d}\|DD^T-\\frac{2K}{d} I\|_F^2` wrt :math:`D`, under additional constraints on the norms of the columns of :math:`D`.
    The minimization is performed by alternatively descending along the gradient direction and projecting the columns (rows) of :math:`D` on the ball with given radius.

    The function is used by :func:`paddle.tight.learn` for the optimization
    wrt ``D``.

    Parameters
    ----------
    D0 : (d, K) ndarray
        Initial value of the unknown
    X : (d, N) ndarray
        Input data (fixed)
    U : (K, N) ndarray
        Current encodings (fixed)
    XUt : (d, K) ndarray
        Part of the gradient
    UUt : (K, K) ndarray
        Part of the gradient
    maxiter : int
        Maximum number of iterations allowed (default 500)
    bound : float
        Value of the constraint on the norms of the columns/rows of ``Y`` (default is 1)
    verbose : bool
        If True displays the value of the cost function at each iteration (default is False)
    rtol : float
        Relative tolerance for convergence criterion
        
    Returns
    -------
    D : (d, K) ndarray
        Optimal value of the dictionary
    j : int
        Number of interations performed
    '''
    # initial value of the cost function
    E = _cost_fast(X, U, D0, pars)
    # fixed step-size used by FISTA
    d, K = D0.shape
    N = U.shape[1]
    sigma = (2./N)*(U**2).sum() + ((4*pars['eta']*(1+K+d))/(d**2))*(D0**2).sum()
    ####### initialization
    D = D0.copy()
    t = 1 # used in FISTA
    Y = D.copy() # needed by FISTA
    ###### pre-computing constant used in the loop
    f1 = (2.*N*pars['eta'])/d
    f2 = (4*pars['eta']*K*N)/(d**2)
    IK = NP.identity(K)
    ###### begins the optimization loop
    for j in xrange(maxiter):
        step = UUt + f1*NP.dot(Y.T, Y) - f2*IK
        # gradient descent step
        Dnew = Y + (1/sigma) * (XUt/(d*N) - NP.dot(Y/(N*d), step))
        ######### projections onto the ball
        if bound > 0:
            # computes the norms
            n = NP.sqrt(NP.sum(Dnew**2, 0))
            n.shape = (1, -1)
            # projects onto the ball with specified radius 
            n = NP.where(n > bound, n/bound, 1)
            Dnew /= n
        ###### recompute cost function
        Enew = _cost_fast(X, U, Dnew, pars)
        if pars['verbose']:
            print '   energy =', Enew
        if abs(Enew-E)/E < pars['rtol']:
            break
        ###### FISTA acceleration
        tnew = (1 + NP.sqrt(1 + 4*(t**2))) / 2
        Y = Dnew + ((t-1)/tnew)*(Dnew - D)
        t = tnew
        D = Dnew
        E = Enew
    return D, j


def learn(X, D0, U0, callable=None, **kwargs):
    '''
    Runs the PADDLE-TF algorithm.

    The function takes as input the data matrix ``X``, and the initial
    values for the unknowns ``D0`` and ``U0``.

    A function that will be called at each iteration might also be passed as
    optional argument.

    All other optional parameters are passed as keyword arguments.

    Parameters
    ----------
    X : (d, N) ndarray
        Input data
    D0 : (d, K) ndarray
        The initial dictionary
    U0 : (K, N) ndarray
        The initial encodings
    callable : function of type foo(D, U, X)
        If not None, it gets called at each iteration and the result is
        appended in an item of full_output
    tau : float, optional
          Weight of the sparsity penalty (default 0.1)
    eta : float, optional
          Weight of the frame potential (default 1.0)
    mu : float, optional
          Weight of the l2 penalty on the coefficients (default 0.0)
    maxiter : int, optional
          Maximum number of outer iterations (default 10)
    minused : integer, optional
          Minimum number of times an atom as to be used (default 1)
    verbose : bool, optional
          Enables verbose output (default False)
    rtol : float, optional
          Relative tolerance checking convergence (default 1.e-4)
    save_dict : bool, optional
          If true, the dictionary is saved after each outer iteration (default False)
    save_path : str, optional
          The path in which to save the dictionary (relevant only if save_dict is True, default "./")
    save_sorted : bool, optional
          If true and if save_dict is also True, the atoms of dictionary are sorted wrt the usage in the sparse coding before being displayed and saved (default False)
    save_shape: integer pair, optional
          Numbers of (rows,cols) used to display the atoms of the dictionary (default (10,10))

    Returns
    -------

    D : (d, K) ndarray
        The final dictionary
    U : (K, N) ndarray
        The final encodings
    full_out : dict
        Full output
    '''

    # start the timer
    time0 = time.time()

    # default parameters
    pars = {
        'maxiter': 10,
        'minused': 1,
        'tau'    : 0.1,
        'eta'    : 1.0,
        'mu'     : 0.0,
        'verbose': False,
        'rtol'   : 1.e-4,
        'save_dict': False,
        'save_path': './',
        'save_sorted': False,
        'save_shape': (10,10),
        }
    # check that all user-defined parameters are existing
    for key in kwargs:
        if key not in pars:
            raise ValueError, "User-defined parameter '%s' is not known" % key
        if key in pars:
            # additional checks
            if isinstance(pars[key], float):
                # cast to float if required
                kwargs[key] = float(kwargs[key])
    # update the parameters with the user-defined values
    pars.update(kwargs)

    # DOES NOT make a copy of the init values
    D, U = D0, U0
    d, K = D.shape

    # check the value of the cost function
    E, full_out0 = _cost(X, U, D, pars)
    print
    print ' Initial energy = %.5e' % E

    # calls the function if present
    if callable != None:
        call_res = [callable(D, U, X), ]

    Ehist = [E,] # keeps track of the energy values
    timing = []  # keeps track of the times
    # start the optimization loop
    for i in xrange(pars['maxiter']):

        print '  iter %d ------------' % i

        # possibly saves figures of the dictionary and its dual
        if pars['save_dict']:
            rows, cols = pars['save_shape']
            # save the dictionary
            _saveDict(D, U, rows, cols, path = pars['save_path']+'dictD_'+str(i), sorted = pars['save_sorted'])

        # 1 ------- sparse coding step
        print '  optimizing U'
        start = time.time()
        U, full_out = _ist(X, U, D, pars, maxiter=1000)
        timeA = time.time() - start

        # checks the sparsity of the solution
        used = NP.where(NP.abs(U) > 0, 1, 0)
        used_per_example = NP.sum(used, 0)
        print '  %.1f / %d non-zero coefficients per example (on avg)' \
            % (used_per_example.mean(), U.shape[0])

        # check the coding and see if there are atoms not being
        # used enough, or not being used at all
        notused = NP.where(NP.sum(used, 1) < pars['minused'])[0]
        if len(notused) > 0:
            print '  %d atoms not used by at least %d examples' \
                % (len(notused), pars['minused'])
            U = _replaceAtoms(X, U, D, notused)

        # 2 ------------ dictionary update

        # pre-compute some matrix we need at this step
        UUt = NP.dot(U, U.T)
        XUt = NP.dot(X, U.T)

        # optimization of the decoding matrix
        print '  Optimizing D'
        print '     reconstruction error = %.5e' % full_out['rec_err']
        print '     frame potential      = %.5e' % full_out['fp']
        start = time.time()
        D, iters = _pgd(D, X, U, XUt, UUt, pars)
        timeB = time.time() - start
        # check the value of the cost function
        E, full_out = _cost(X, U, D, pars)
        print '     final reconstruction error = %.5e' % full_out['rec_err']
        print '     final frame potential      = %.5e' % full_out['fp']
        print '   energy after %d iter. = %.5e' % (iters+1, E)

        # 3 ------------- stopping condition
        if callable != None:
            call_res.append(callable(D, U, X))

        Em = NP.mean(Ehist[-min(len(Ehist), 3):])
        if abs(E-Em)/Em < 10.*pars['rtol']:
            break
        Ehist.append(E)
        full_out0 = full_out
        timing.append((timeA, timeB))

    # collect and print some final stats
    print '  final --------------' 
    used = NP.where(NP.abs(U) > 0, 1, 0)
    l0 = used.flatten().sum()
    print '  %d / %d non-zero coefficients' % (l0, U.size)
    print '  average atom usage = %.1f' % NP.sum(used, 1).mean()
    
    timing.append(time.time() - time0)
    full_out['time'] = timing
    if callable != None:
        full_out['call_res'] = call_res

    return D, U, full_out

def init(X, K, det=False):
    '''
    Initializes the variables.

    Initializes the matrices ``D`` and ``U`` from the data matrix
    ``X``.
    The dimension of the dictionary is ``K``.
    If ``det`` is True the first ``K`` columns of ``X`` are chosen as
    atoms (deterministic initialization), otherwise they are picked 
    randomly.
    The atoms are normalized to one.
    The matrix ``U`` is chosen as to minimize the reconstruction error.

    Parameters
    ----------
    X : (d, N) ndarray
        Input data
    K : integer
        Size of the dictionary
    det : bool
        If True, the choice of atoms is deterministic
    
    Returns
    -------
    D0 : (d, K) ndarray
        The initial dictionary
    U0 : (K, N) ndarray
        The initial encodings
    '''
    d, N = X.shape
    # D is the decoding matrix
    # it is initialized by choosing a random sample of
    # non-zero examples
    nonzero = NP.where(NP.sum(X**2, 0) > 0)[0]
    if det:
        sample = nonzero[:K]
    else:
        sample = NP.random.permutation(len(nonzero))[:K]
        sample = nonzero[sample]
    D0 = X[:, sample]
    # the atoms are normalized
    D0 /= NP.sqrt(NP.sum(D0**2, 0)).reshape((1, -1))
    # the coefficients are initialized with the optimal ones
    # in an l2-sense
    pinv = LA.pinv(D0)
    U0 = NP.dot(pinv, X)
    return D0, U0


if __name__=='__main__':

    N = 1500 # number of data
    K = 50  # dimension of the dictionary
    k = 3   # number of relevant atoms for each datum
    d = 20  # dimension of the data
    #dic_reg = 1.

    ############ synthetic experiment
    
    # generating dictionary, all atoms are centered on 0
    # and with 2-norm one
    D_true = RA.uniform(-1, 1, size=(d, K)).astype(NP.float32)
    #D_true -= NP.mean(D_true, 0) 
    D_true /= NP.sqrt(NP.sum(D_true**2, 0)).reshape((1, -1))
    #assert NP.allclose(NP.mean(D_true, 0), 0, atol=1.e-7), NP.mean(D_true, 0)

    # generating coefficients
    U_true = NP.zeros((K, N), NP.float32)
    for i in xrange(N):
        U_true[RA.permutation(K)[:k],i] = RA.uniform(-1, 1, size=(k,))
    assert NP.all(NP.sum(NP.where(NP.abs(U_true) > 0, 1, 0), 0) == k)
    #m = NP.mean(U_true, 0)
    #U_true -= m.reshape((1, -1))
    #assert NP.all(NP.abs(NP.mean(U_true, 0)) < 1.e-6)

    # generate data
    X = NP.dot(D_true, U_true)
    assert X.shape == (d, N)
    #assert NP.all(NP.abs(NP.mean(X, 0)) < 1.e-6), NP.mean(X, 0)
    #assert NP.allclose(NP.mean(X, 0), 0, atol=1.e-7), NP.mean(X, 0)

    # add some Gaussian noise
    noise = RA.normal(0, 0.007, size=(d, N))
    SNR = LA.norm(X) / LA.norm(noise)
    print 'SNR (dB) = %d' % (20*NP.log10(SNR),)
    X += noise

    D0, U0 = init(X, K)

    pars = {
        'tau' : .2, # sparsity coefficient
        'mu'  : 1.e-8, # l2 regularization
        'eta' : 0., # coding/decoding weight
        'maxiter' : 80,
        'minused' : 1,
        'verbose' : False,
        'rtol': 1.e-8,
        }
    D, U, full_out = learn(X, D0, U0, None, **pars)

    # check the tightness of the frame
    print_frame_assessment(D)
    #pylab.matshow(NP.dot(D, D.T))
    #pylab.show()
    1/0
    
    # comparison wih true dictionary
    print
    print NP.sum(D**2, 0).mean()
    D /= NP.sqrt(NP.sum(D**2, 0)).reshape((1, -1))
    print NP.sum(D**2, 0).mean()
    print NP.sum(D_true**2, 0).mean()
    d2 = NP.sum((D[:,:,NP.newaxis] - D_true[:,NP.newaxis,:])**2, 0)
    closest = NP.argmin(d2, 0)
    dist = 1-NP.abs(NP.sum(D[:,closest] * D_true, 0))
    print dist
    print NP.where(dist < 0.01)
    print len(NP.where(dist < 0.01)[0])
    
