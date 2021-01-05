'''
Implementation of PADDLE.

The two main functions are learn(), to actually run the algorithm, and init(),
to initialize the algorithm variables.

All other functions are internal, used in the implementation of the
above-mentioned two.
As such, their names begin with an underscore.
'''

import time
from scipy import linalg as la
from scipy import stats
import scipy as sp
import pylab
from common import _cost_rec, _cost_cod, _replaceAtoms, _saveDict
from prox import _st
ra = sp.random

    
def _cost(X, U, D, C, pars):
    '''
    Evaluation of the cost function.
    '''
    full_out = {
        'rec_err': _cost_rec(D, X, U, pars),
        'l1_pen': 2*pars['tau'] * sp.absolute(U).mean(),
        'l2_pen': pars['mu'] * (U**2).mean(),
        }
    E = full_out['rec_err'] + full_out['l1_pen'] + full_out['l2_pen']
    if pars['eta'] > 0:
        full_out['cod_err'] = _cost_cod(C, X, U, pars)
        E += full_out['cod_err']
    return E, full_out

def _cost_fast(X, U, D, CX, pars):
    '''
    Fast evaluation of the cost function.
    
    Computes the same cost function as in _cost, but exploits a precomputed
    CX and returns just the value of the whole cost function.
    '''
    return ((X - sp.dot(D, U))**2).mean() + pars['eta']*((U - CX)**2).mean() + 2*pars['tau'] * sp.absolute(U).mean() + pars['mu'] * (U**2).mean()

def _ist_fixed(X, U0, D, C, pars, maxiter=1000):
    '''
    Iterative soft-thresholding with fixed step-size.

    Minimization of :math:`\\frac{1}{d}\|X-DU\|_F^2+\\frac{\eta}{K}\|U-CX\|_F^2+\\frac{2\\tau}{K}\|U\|_1` wrt :math:`U`.
    When :math:`\eta=0` the functional reduces to the well-known LASSO.

    This function is curently noy used. The main function
    :func:`paddle.dual.learn` uses its FISTA-accelerated counterpart
    :func:`paddle.dual._pgd`.

    Parameters
    ----------
    X : (d, N) ndarray
        Data matrix
    U0 : (K, N) ndarray
        Initial value of the unknown 
    D : (d, K) ndarray
        Dictionary
    C : (K, d) ndarray
        Dual of the dictionary
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
    # gets the dimensions
    d, N = X.shape
    K = D.shape[1]

    U = U0.copy()
    DtD = sp.dot(D.T, D)
    ew = la.eigvalsh(DtD) 
    sigma0 = (ew.min() + ew.max()) / (2*N*d) + (pars['mu'] + pars['eta'])/(N*K)
    Y = U # no need to copy here
    sigma = sigma0
    E, full_out = _cost(X, U, D, C, pars)
    print '   initial energy = %.5e' % E
    if pars['verbose']:
        print '   iter   energy   avg|upd|'
    # pre-computes what is needed in the loop
    DtD /= (N*d*sigma)
    DtX = sp.dot(D.T, X) / (N*d*sigma)
    CX = sp.dot(C, X)
    U0 = pars['eta']*CX/(N*K*sigma) + DtX
    f = (1 - (pars['mu']+pars['eta'])/(N*K*sigma))
    A = DtD - f*sp.identity(K)
    for i in xrange(maxiter):
        Unew = _st(U0 - sp.dot(A, Y), pars['tau']/(N*K*sigma))
        E_new = _cost_fast(X, Unew, D, CX, pars)
        if i%1 == 0 and pars['verbose']:
            upd = sp.sum((Unew-U)**2, 0)/sp.sum(U**2, 0)
            upd = upd[sp.isfinite(upd)]
            print '   %4d  %8.5e     %4.2f' % (i, E_new, upd.mean())
        Y = Unew
        U = Unew
        if sp.absolute(E-E_new)/E < pars['rtol']:
            break
        E = E_new
    print '   energy after %d iter. = %.5e' % (i+1, E)
    dummy, full_out = _cost(X, Unew, D, C, pars)
    return U, full_out

def _ist(X, U0, D, C, pars, maxiter=1000):
    '''
    Iterative soft-thresholding with FISTA acceleration.

    Minimization of :math:`\\frac{1}{d}\|X-DU\|_F^2+\\frac{\eta}{K}\|U-CX\|_F^2+\\frac{2\\tau}{K}\|U\|_1` wrt :math:`U`.
    When :math:`\eta=0` the functional reduces to the well-known LASSO.

    The function is used by :func:`paddle.dual.learn` for the optimization
    wrt ``U``.

    Parameters
    ----------
    X : (d, N) ndarray
        Data matrix
    U0 : (K, N) ndarray
        Initial value of the unknown 
    D : (d, K) ndarray
        Dictionary
    C : (K, d) ndarray
        Dual of the dictionary
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
    E, full_out = _cost(X, U0, D, C, pars)
    print '   initial energy = %.5e' % E
    if pars['verbose']:
        print '   iter   energy   avg|upd|'
    ######## fixed-step sigma
    DtD = sp.dot(D.T, D)
    # computes only the largest eigenvalue
    ewmax = la.eigvalsh(DtD, eigvals=(K-1, K-1)) 
    sigma0 = ewmax/(N*d) + (pars['mu'] + pars['eta'])/(N*K)
    ######## FISTA initialization
    U = U0.copy()
    t = 1 # used in FISTA
    Y = U.copy() # needed by FISTA
    ####### pre-computes what is needed in the loop
    DtD /= (N*d*sigma0)
    DtX = sp.dot(D.T, X) / (N*d*sigma0)
    if C != None:
        CX = sp.dot(C, X)
    else:
        CX = sp.zeros(U.shape)
    # constant part of the gradient descent step
    U0 = pars['eta']*CX/(N*K*sigma0) + DtX
    # first-order part of the gradient descent step
    f = (1 - (pars['mu']+pars['eta'])/(N*K*sigma0))
    A = DtD - f*sp.identity(K)
    ####### begin loop
    for i in xrange(maxiter):
        # compute gradient step and soft-thresholding
        Unew = _st(U0 - sp.dot(A, Y), pars['tau']/(N*K*sigma0))
        # non-negativity
        if pars['nnU']:
            Unew = sp.clip(Unew, 0, sp.inf)
        # evaluates the cost function
        E_new = _cost_fast(X, Unew, D, CX, pars)
        if i%1 == 0 and pars['verbose']:
            upd = sp.sum((Unew-U)**2, 0)/sp.sum(U**2, 0)
            upd = upd[sp.isfinite(upd)]
            print '   %4d  %8.5e     %4.2f' % (i, E_new, upd.mean())
        # FISTA from Beck and Teboulle, SIAM J. Imaging Science, 2009
        tnew = (1 + sp.sqrt(1 + 4*(t**2))) / 2
        Y = Unew + ((t-1)/tnew)*(Unew - U)
        t = tnew
        U = Unew
        # check convergence
        if sp.absolute(E-E_new)/E < pars['rtol']:
            break
        E = E_new
    ####### final evaluation of the cost function
    print '   energy after %d iter. = %.5e' % (i+1, E)
    dummy, full_out = _cost(X, Unew, D, C, pars)
    return U, full_out

def _pgd_fixed(A0, X, U, B, G2, cost, maxiter=500, pars=None, sigma=None, axis=0, bound=1):
    '''
    Projected gradient descent with fixed stepsize.
    '''
    A = A0.copy()
    if sigma == None:
        # eigenvalues of the second derivatives matrix
        ew = la.eigvalsh(G2)
        # (inverse) step length
        sigma = (ew.min() + ew.max())/2
    E = cost(A, X, U, pars)
    for j in xrange(maxiter):
        # gradient descent step
        Anew = A + (1/sigma) * (B - sp.dot(A, G2))
        # projections onto the ball
        n = sp.sqrt(sp.sum(Anew**2, axis))
        newshape = [-1, -1]
        newshape[axis] = 1
        n.shape = newshape
        n = sp.where(n > bound, n, 1)
        Anew /= n
        #E, full_out = cost(X, U, Dnew, C, tau, mu, eta)
        #print '   energy =', E
        Enew = cost(Anew, X, U, pars)
        if abs(Enew-E)/E < pars['rtol']:
            break
        A = Anew
        E = Enew
    return A, j

def _pgd(Y0, ABt, BBt, cost, maxiter=500, axis=0, bound=1, verbose=False, rtol=1.e-4, nn=False):
    '''
    Projected gradient descent with FISTA acceleration.

    Minimization of :math:`\|A-YB\|_F^2` wrt :math:`Y`, under additional
    constraints on the norms of the columns (or the rows) of :math:`Y`.
    The minimization is performed by alternatively descending along the
    gradient direction :math:`AB^T-YBB^T` and projecting the columns (rows)
    of :math:`Y` on the ball with given radius.

    The function is used by :func:`paddle.dual.learn` for the optimization
    wrt ``D`` and ``C``.
    In the former case, ``A`` and ``B`` are ``X`` and ``U``, respectively,
    while in the latter the roles are swapped.

    Parameters
    ----------
    Y0 : (a1, a2) ndarray
        Initial value of the unknown
    ABt : (a1, a2) ndarray
        Part of the gradient
    BBt : (a2, a2) ndarray
        Part of the gradient
    cost : function of type ``foo(Y)``
        Evaluates the cost function 
    maxiter : int
        Maximum number of iterations allowed (default 500)
    axis : int
        Dimension of ``Y`` along which the constraint is active (0 for cols, 1 for rows, default is 0)
    bound : float
        Value of the constraint on the norms of the columns/rows of ``Y`` (default is 1)
    verbose : bool
        If True displays the value of the cost function at each iteration (default is False)
    rtol : float
        Relative tolerance for convergence criterion
        
    Returns
    -------
    Y : () ndarray
        Optimal value of the unknown
    j : int
        Number of interations performed
    '''
    # initial value of the cost function
    E = cost(Y0)
    # fixed step-size used by FISTA
    sigma = 2*la.norm(BBt)
    # initialization
    Y = Y0.copy()
    t = 1 # used in FISTA
    Z = Y.copy() # needed by FISTA
    for j in xrange(maxiter):
        ###### gradient descent step
        Ynew = Z + (1/sigma) * (ABt - sp.dot(Z, BBt))
        ###### non-negativity constraint
        if nn:
            Ynew = sp.clip(Ynew, 0, sp.inf)
        ###### projections onto the ball
        if bound > 0:
            # computes the norms
            n = sp.sqrt(sp.sum(Ynew**2, axis))
            # reshapes the norms depending on the direction of the constraint
            newshape = [-1, -1]
            newshape[axis] = 1
            n.shape = newshape
            # projects onto the ball with specified radius 
            n = sp.where(n > bound, n/bound, 1)
            Ynew /= n
        ###### recompute cost function
        Enew = cost(Ynew)
        if verbose:
            print '   energy =', Enew
        if abs(Enew-E)/E < rtol:
            break
        ###### FISTA acceleration
        tnew = (1 + sp.sqrt(1 + 4*(t**2))) / 2
        Z = Ynew + ((t-1)/tnew)*(Ynew - Y)
        t = tnew
        Y = Ynew
        E = Enew
    return Y, j

#def learn(X, D0, C0, U0, _pars=dict(), callable=None):
def learn(X, D0, C0, U0, callable=None, **kwargs):
    '''
    Runs the PADDLE algorithm.

    The function takes as input the data matrix ``X``, the initial
    values for the three unknowns, ``D0`` ``C0`` and ``U0``, and a dictionary
    of parameters.

    The optional parameters are passed as keyword arguments.

    Parameters
    ----------
    X : (d, N) ndarray
        Input data
    D0 : (d, K) ndarray
        The initial dictionary
    C0 : (K, d) ndarray
        The initial dual
    U0 : (K, N) ndarray
        The initial encodings
    callable : function of type foo(D, C, U, X)
        If not None, it gets called at each iteration and the result is
        appended in an item of full_output
    tau : float, optional
          Weight of the sparsity penalty (default 0.1)
    eta : float, optional
          Weight of the coding error (default 1.0)
    mu : float, optional
          Weight of the l2 penalty on the coefficients (default 0.0)
    nnU : bool, optional
          Adds a non-negativity constraint on U (default False)
    nnD : bool, optional
          Adds a non-negativity constraint on U (default False)
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
    C : (K, d) ndarray
        The final dual
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
        'nnU'    : False,
        'nnD'    : False,
        'Cbound' : 1.0,
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
    C, D, U = C0, D0, U0
    d, N = X.shape
    K = D.shape[1]

    # the covariance matrix is used at each iteration
    XXt = sp.dot(X, X.T)

    # check the value of the cost function
    E, full_out0 = _cost(X, U, D, C, pars)
    print
    print ' Initial energy = %.5e' % E

    # calls the function if present
    if callable != None:
        call_res = [callable(D, C, U, X), ]

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
            # save the dual
            _saveDict(C.T, U, rows, cols, path = pars['save_path']+'dictC_'+str(i), sorted = pars['save_sorted'])

        # 1 ------- sparse coding step
        print '  optimizing U'
        start = time.time()
        U, full_out = _ist(X, U, D, C, pars, maxiter=1000)
        timeA = time.time() - start

        # checks the sparsity of the solution
        used = sp.where(sp.absolute(U) > 0, 1, 0)
        used_per_example = sp.sum(used, 0)
        print '  %.1f / %d non-zero coefficients per example (on avg)' \
            % (used_per_example.mean(), U.shape[0])

        # check the coding and see if there are atoms not being
        # used enough, or not being used at all
        notused = sp.where(sp.sum(used, 1) < pars['minused'])[0]
        if len(notused) > 0:
            print '  %d atoms not used by at least %d examples' \
                % (len(notused), pars['minused'])
            U = _replaceAtoms(X, U, D, notused)

        # 2 ------------ dictionary update

        # pre-compute some matrix we need at this step
        UUt = sp.dot(U, U.T)
        XUt = sp.dot(X, U.T)

        # optimization of the decoding matrix
        print '  Optimizing D'
        print '   reconstruction error = %.2e' % full_out['rec_err']
        start = time.time()
        def _costD(Y):
            return _cost_rec(Y, X, U)
        f = 1./(N*d)
        D, iters = _pgd(D, f*XUt, f*UUt, _costD, verbose=pars['verbose'], rtol=pars['rtol'], nn=pars['nnD'])
        timeB = time.time() - start
        # check the value of the cost function
        E, full_out = _cost(X, U, D, C, pars)
        print '   final reconstruction error = %.2e' % full_out['rec_err']
        print '   energy after %d iter. = %.5e' % (iters+1, E)

        if pars['eta'] > 0:

            # optimization of the coding matrix
            print '  Optimizing C'
            print '   coding error = %.2e' % full_out['cod_err']
            start = time.time()
            def _costC(Y):
                return _cost_cod(Y, X, U, pars)
            f = pars['eta']/(N*K)
            C, iters = _pgd(C, f*XUt.T, f*XXt, _costC, axis=1, bound=pars['Cbound'], verbose=pars['verbose'], rtol=pars['rtol'])
            timeC = time.time() - start
            # check the value of the cost function
            E, full_out = _cost(X, U, D, C, pars)
            print '   final coding error = %.2e' % full_out['cod_err']
            print '   energy after %d iter. = %.5e' % (iters+1, E)
        else:
            timeC = 0

        # 3 ------------- stopping condition
        if callable != None:
            call_res.append(callable(D, C, U, X))

        Em = sp.mean(Ehist[-min(len(Ehist), 3):])
        if abs(E-Em)/Em < 10.*pars['rtol']:
            break
        Ehist.append(E)
        full_out0 = full_out
        timing.append((timeA, timeB, timeC))

    # collect and print some final stats
    print '  final --------------' 
    used = sp.where(sp.absolute(U) > 0, 1, 0)
    l0 = used.flatten().sum()
    print '  %d / %d non-zero coefficients' % (l0, U.size)
    print '  average atom usage = %.1f' % sp.sum(used, 1).mean()
    timing.append(time.time() - time0)
    full_out['time'] = timing
    if callable != None:
        full_out['call_res'] = call_res
    full_out['iters'] = i
    
    return D, C, U, full_out

def init(X, K, det=False, rnd=False):
    '''
    Initializes the variables.

    Initializes the matrices ``D``, ``C`` and ``U`` from the data matrix
    ``X``.
    The dimension of the dictionary is ``K``.
    If ``det`` is True the first ``K`` columns of ``X`` are chosen as
    atoms (deterministic initialization), otherwise they are picked 
    randomly.
    The atoms are normalized to one.
    The matrix ``U`` is chosen as to minimize the reconstruction error.
    The matrix ``C`` is chosen as the pseudo-inverse of ``D``, with
    rows normalized to one.

    Parameters
    ----------
    X : (d, N) ndarray
        Input data
    K : integer
        Size of the dictionary
    det : bool
        If True, the choice of atoms is deterministic
    rnd : bool
        If True, the atoms are not sampled from the examples, but have random values
    
    Returns
    -------
    D0 : (d, K) ndarray
        The initial dictionary
    C0 : (K, d) ndarray
        The initial dual
    U0 : (K, N) ndarray
        The initial encodings
    '''
    d, N = X.shape
    # D is the decoding matrix
    # if rnd is True, we chose random atoms
    if rnd:
        D0 = sp.random.rand(d, K)
    # otherwise, it is initialized by choosing a random sample of
    # non-zero examples
    else:
        nonzero = sp.where(sp.sum(X**2, 0) > 0)[0]
        if det:
            sample = nonzero[:K]
        else:
            sample = sp.random.permutation(len(nonzero))[:K]
            sample = nonzero[sample]
        D0 = X[:, sample]
    # the atoms are normalized
    D0 /= sp.sqrt(sp.sum(D0**2, 0)).reshape((1, -1))
    # the coefficients are initialized with the optimal ones
    # in an l2-sense
    pinv = la.pinv(D0)
    U0 = sp.dot(pinv, X)
    # C is the coding matrix
    # it is initialized with the pseudoinverse of D, but 
    # after normalizing the rows
    C0 = pinv/sp.sqrt(sp.sum(pinv**2, 1)).reshape((-1, 1))
    return D0, C0, U0

if __name__=='__main__':

    N = 1500 # number of data
    K = 50  # dimension of the dictionary
    k = 3   # number of relevant atoms for each datum
    d = 20  # dimension of the data

    ############ synthetic experiment

    NMF = True
    
    # generating dictionary, all atoms are centered on 0
    # and with 2-norm one
    if NMF:
        D_true = ra.uniform(0, 1, size=(d, K)).astype(sp.float32)
    else:
        D_true = ra.uniform(-1, 1, size=(d, K)).astype(sp.float32)
    #D_true -= sp.mean(D_true, 0) 
    D_true /= sp.sqrt(sp.sum(D_true**2, 0)).reshape((1, -1))
    #assert sp.allclose(sp.mean(D_true, 0), 0, atol=1.e-7), sp.mean(D_true, 0)

    # generating coefficients
    U_true = sp.zeros((K, N), sp.float32)
    for i in xrange(N):
        if NMF:
            U_true[ra.permutation(K)[:k],i] = ra.uniform(0, 1, size=(k,))
        else:
            U_true[ra.permutation(K)[:k],i] = ra.uniform(-1, 1, size=(k,))
    assert sp.all(sp.sum(sp.where(sp.absolute(U_true) > 0, 1, 0), 0) == k)
    #m = sp.mean(U_true, 0)
    #U_true -= m.reshape((1, -1))
    #assert sp.all(sp.absolute(sp.mean(U_true, 0)) < 1.e-6)

    # generate data
    X = sp.dot(D_true, U_true)
    assert X.shape == (d, N)
    #assert sp.all(sp.absolute(sp.mean(X, 0)) < 1.e-6), sp.mean(X, 0)
    #assert sp.allclose(sp.mean(X, 0), 0, atol=1.e-7), sp.mean(X, 0)

    # add some Gaussian noise
    noise = ra.normal(0, 0.007, size=(d, N))
    SNR = la.norm(X) / la.norm(noise)
    print 'SNR (dB) = %d' % (20*sp.log10(SNR),)
    X += noise

    D0, C0, U0 = init(X, K)

    #D0 = D_true + ra.normal(0, 0.2, size=(d, K))
    tau = .1
    mu = 1.e-8
    eta = 0.
    maxiter = 80
    D, C, U, full_out = learn(X, D0, C0, U0, tau=tau, mu=mu, eta=eta, maxiter=maxiter)

    # comparison wih true dictionary
    print
    print sp.sum(D**2, 0).mean()
    D /= sp.sqrt(sp.sum(D**2, 0)).reshape((1, -1))
    print sp.sum(D**2, 0).mean()
    print sp.sum(D_true**2, 0).mean()
    d2 = sp.sum((D[:,:,sp.newaxis] - D_true[:,sp.newaxis,:])**2, 0)
    closest = sp.argmin(d2, 0)
    dist = 1-sp.absolute(sp.sum(D[:,closest] * D_true, 0))
    print dist
    print sp.where(dist < 0.01)
    print len(sp.where(dist < 0.01)[0])

    if NMF:
        print sp.where(U < 0), sp.where(D < 0)
        D, C, U, full_out = learn(X, D0, C0, U0, tau=tau, mu=mu, eta=eta, maxiter=maxiter, nnU=True, nnD=True, verbose=False)
        print sp.where(U < 0), sp.where(D < 0)
        d2 = sp.sum((D[:,:,sp.newaxis] - D_true[:,sp.newaxis,:])**2, 0)
        closest = sp.argmin(d2, 0)
        dist = 1-sp.absolute(sp.sum(D[:,closest] * D_true, 0))
        print dist
        print sp.where(dist < 0.01)
        print len(sp.where(dist < 0.01)[0])
