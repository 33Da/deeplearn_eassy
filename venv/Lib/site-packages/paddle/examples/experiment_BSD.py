'''
This is the script to run the experiment on the BSd dataset, reported in
section 4.2 of the techical report:

C.Basso, M.Santoro, A.Verri and S.Villa. "PADDLE: Proximal Algorithm for
Dual Dictionaries LEarning", DISI-TR-2010-XX, 2010.
'''

import sys, os, glob, pylab
import scipy.linalg as la
import scipy.stats
import scipy as sp
from paddle import dual, common, data

if __name__=='__main__':

    W = 12   # patch width
    N = 1e4  # number of patches to sample
    tau = 1. # sparsity coefficient
    eta = 1. # coding/decoding weight
    K = 200  # size of dictionary
    R = 1    # number of repetitions
    pca = True

    # set the input arguments
    if len(sys.argv) < 2:
        print 'usage: python %s BSD_root_path [tau eta K]' % sys.argv[0]
        sys.exit(0)

    if len(sys.argv) > 2:
        if len(sys.argv) != 5:
            print 'usage: python %s BSD_root_path [tau eta K]' % sys.argv[0]
            sys.exit(0)
        tau = float(sys.argv[2])
        eta = float(sys.argv[3])
        K = int(sys.argv[4])

    # checks the directory holding the Berkeley segmentation dataset
    path = sys.argv[1]
    testdir, traindir = data.checkBSD(path)

    # randomly draw the patches for training images, or load them from file
    Xtrn, Xtst = data.draw_patches(traindir, W, N)

    # recenter the patches
    Xtrn -= sp.mean(Xtrn, 0).reshape((1, -1))
    Xtst -= sp.mean(Xtst, 0).reshape((1, -1))

    # start dictionary learning
    pars = {
        'tau' : tau, # sparsity coefficient
        'mu'  : 0, # l2 regularization
        'eta' : eta, # coding/decoding weight
        'maxiter' : 200,
        'minused' : 1,
        'verbose': False,
        'rtol': 1.e-5,
        }
    # file where the results are stored
    dicfn = 'BSD_dict_%dx%d_%dk_tau%.1e_eta%d_K%d.npz' % (W, W, N/1e3, pars['tau'], pars['eta'], K)
    if not os.access(dicfn, os.R_OK):
        # initialize the variables
        D0, C0, U0 = dual.init(Xtrn, K, det=False)
        # learn the dictionary
        D, C, U, full_out = dual.learn(Xtrn, D0, C0, U0, **pars)
        print
        print ' whole computation took %.1f secs' % full_out['time'][-1]
        timing = sp.sum(sp.array(full_out['time'][:-1]), 0)
        print ' ... time spent optimizing U = %6.1f secs (%.1f%%)' % (timing[0], 100*timing[0]/full_out['time'][-1])
        print ' ... time spent optimizing D = %6.1f secs (%.1f%%)' % (timing[1], 100*timing[1]/full_out['time'][-1])
        print ' ... time spent optimizing C = %6.1f secs (%.1f%%)' % (timing[2], 100*timing[2]/full_out['time'][-1])
        # save the results
        sp.savez(dicfn, D=D, C=C, U=U)

    # compute PCA for comparison
    if pca:
        pcafn = 'BSD_pca_%dx%d_%dk.npz' % (W, W, N/1e3)
        if not os.access(pcafn, os.R_OK):
            Cov = sp.dot(Xtrn, Xtrn.T)/(Xtrn.shape[1]-1)
            ew, ev = la.eigh(Cov)
            order = sp.argsort(ew)[::-1]
            ew = ew[order]
            ev = ev[:,order]
            assert sp.allclose(sp.sum(ev**2, 0), 1)
            Erec_pca, Erec_pca_tst = [], []
            for i in xrange(1, ev.shape[1]-1):
                Xr = sp.dot(ev[:,:i], sp.dot(ev[:,:i].T, Xtrn))
                erec = la.norm(Xtrn - Xr)/la.norm(Xtrn)
                Erec_pca.append(erec)
                Xr = sp.dot(ev[:,:i], sp.dot(ev[:,:i].T, Xtst))
                erec = la.norm(Xtst - Xr)/la.norm(Xtst)
                Erec_pca_tst.append(erec)
            Erec_pca, Erec_pca_tst = sp.array(Erec_pca), sp.array(Erec_pca_tst)
            sp.savez(pcafn, Erec_pca=Erec_pca, Erec_pca_tst=Erec_pca_tst)
        else:
            npz = sp.load(pcafn)
            Erec_pca = npz['Erec_pca']
            Erec_pca_tst = npz['Erec_pca_tst']

    # draw the atoms
    dicfn = 'BSD_dict_%dx%d_%dk_tau%.1e_eta%d_K%d.npz' % (W, W, N/1e3, tau, eta, K)
    assert os.access(dicfn, os.R_OK), 'output dictionary file %s not found' % dicfn
    npz = sp.load(dicfn)
    figfn = 'BSD_atoms_%dx%d_%dk_tau%.0e_eta%d_K%d' % (W, W, N/1e3, tau, eta, K)
    Nrows, Ncols = 20, 10
    assert Nrows*Ncols <= K, 'reduce the number of rows or columns'
    U, D, C = npz['U'], npz['D'], npz['C']
    common._saveDict(D, U, Nrows, Ncols, path = figfn + '_D.png', sorted = True)
    common._saveDict(C.T, U, Nrows, Ncols, path = figfn + '_C.png', sorted = True)
        
