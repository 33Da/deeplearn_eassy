'''
Experiment on the MNIST dataset for the talk at CBCL.
Note that the following code extends that used for the 
submission to NIPS 2010.
'''

import sys, os, glob, scipy, pylab
import numpy as NP
import scipy.stats
import cPickle, gzip
import paddle

if __name__=='__main__':

    if len(sys.argv) < 2:
        print 'usage: python %s MNIST_dataset_path (output_path)' % sys.argv[0]
        sys.exit(0)

    W = 28 # image size for the MNIST dataset
    K = 64 # size of dictionary
    M = 10000 # number of patches for each repetition
    R = 1 # number of repetitions

    path = sys.argv[1]
    if len(sys.argv) < 3:
        outpath = './results/'
    else:
        outpath = sys.argv[2]
    
    # check if a previously drawn sample is already there
    patchesfn = 'MNIST_patches_%dx%d_%dk.npy' % (W, W, M)
    if os.access(patchesfn, os.R_OK):
        print 'Loading a previously drawn sample from', patchesfn
        print 'REMOVE the file IF you want a NEW SAMPLE'
        X = NP.load(patchesfn)
    else:
        # if not there, load the MNIST dataset and draw a new sample
        train_set, validation_set, test_set = paddle.data.loadMNIST(path)
        X = train_set[0].T
        N = X.shape[1]
        # sampling M samples from the entire dataset
        # X = X[:,NP.arange(0,M)]
        X = X[:,NP.random.permutation(int(N))[:M]]
        # map the intensities of the input images to the range [0,1]
        X /=255.
        # center the patches
        m = NP.mean(X, 0).reshape((1, -1))
        X -= m
        # save the patches for subsequent runs of the experiment
        NP.save(patchesfn, X)

    N = X.shape[1] # total number of digits
    d = X.shape[0] # dimensionality of the input space
    assert W == NP.sqrt(d)
    
    # loop over the repetitions
    for rep in xrange(R):

        # randomly scramble the patches for this repetition
        # X = X[:,NP.random.permutation(int(M))]

        # start dictionary learning
        dicfn = 'MNIST_dict_%dx%d_%dk.npz' % (W, W, M)
        if os.access(dicfn, os.R_OK):
            data = NP.load(dicfn)
            D = data['arr_0']
            C = data['arr_1']
            U = data['arr_2']
        else:
            pars = {
                'tau' : 5e-4, # sparsity coefficient
                'mu'  : 1.e-4, # l2 regularization
                'eta' : 1., # coding/decoding weight
                'maxiter' : 15,
                'minused' : 1,
                'verbose': True,
                'rtol': 1.e-3,
                'save_dict': True, 
                'save_path': outpath,
                'save_sorted': True,
                'save_shape': (8,8)
                }
            D0, C0, U0 = paddle.dual.init(X, K)
            # WARNING: MODIFICA MATTEO
            #D0 = NP.random.rand(D0.shape[0], D0.shape[1])
            #pinv = NP.linalg.pinv(D0)
            #U0 = NP.dot(pinv, X)
            #C0 = pinv/NP.sqrt(NP.sum(pinv**2, 1)).reshape((-1, 1))
            D, C, U , full_out = paddle.dual.learn(X, D0, C0, U0, callable=None, **pars)
            NP.savez(dicfn, D, C, U)

        # display some of the encoder/decoder pairs
        usage = NP.sum(NP.where(NP.abs(U) > 1.e-6, 1, 0), 1)
        order = NP.argsort(usage)[::-1]

        
        #indexOfDigit = NP.round(NP.random.rand()*X.shape[1])
        indexOfDigit = 5017
        print indexOfDigit
        Ncols = len(NP.where(U[:,indexOfDigit]>0)[0])
        nonzeroindices = NP.where(U[:,indexOfDigit]>0)[0]
        print nonzeroindices
        print U[nonzeroindices,indexOfDigit]
        
        m = 2 # margin between the atoms
        imgCOMP = NP.ones((2*W+m, W*(4+Ncols)+m*(4+Ncols)))
        vmaxCOMP = scipy.stats.scoreatpercentile(D.flatten(), 98)
        vminCOMP = scipy.stats.scoreatpercentile(D.flatten(), 2)
        imgCOMP *= vmaxCOMP
        #imgCOMP[0:W,0:W] = 50*X[:,indexOfDigit].reshape((W,W))
        imgCOMP[W+m:2*W+m,0:W] = 50*X[:,indexOfDigit].reshape((W,W))
        
        for i in NP.arange(2,Ncols+2):
            x = i*(W+m) # coordinates of the top-left corner
            imgCOMP[0:W,x:x+W] = D[:,nonzeroindices[i-2]].reshape((W,W))
            imgCOMP[W+m:2*W+m,x:x+W] = D[:,nonzeroindices[i-2]].reshape((W,W))*U[nonzeroindices[i-2],indexOfDigit]*150

        x = (3+Ncols)*(W+m)      
        #imgCOMP[0:W,x:x+W] = 100*NP.dot(D[:,nonzeroindices],U[nonzeroindices,indexOfDigit]).reshape((W,W))
        imgCOMP[W+m:2*W+m,x:x+W] = 100*NP.dot(D[:,nonzeroindices],U[nonzeroindices,indexOfDigit]).reshape((W,W))
        dpi = 50.
        pylab.figure(figsize=(imgCOMP.shape[1]/dpi, imgCOMP.shape[0]/dpi), dpi=dpi)
        pylab.imshow(imgCOMP, interpolation='nearest', vmin=vminCOMP, vmax=vmaxCOMP)
        pylab.gray()
        pylab.xticks(())
        pylab.yticks(())
        #pylab.show()
        pylab.savefig('mnistCOMP.png', dpi=300, bbox_inches='tight', transparent=True)
        
    #     Nrows, Ncols = 8, 25
    #     assert Nrows*Ncols == D.shape[1]
    #     subset = order[:Nrows*Ncols]

    #     # build an image with all atoms
    #     m = 2 # margin between the atoms
    #     imgD = NP.ones((W*Nrows+m*(Nrows-1), W*Ncols+m*(Ncols-1)))
    #     vminD = scipy.stats.scoreatpercentile(D.flatten(), 2)
    #     vmaxD = scipy.stats.scoreatpercentile(D.flatten(), 98)
    #     imgD *= vmaxD
    #     imgC = NP.ones((W*Nrows+m*(Nrows-1), W*Ncols+m*(Ncols-1)))
    #     vminC = scipy.stats.scoreatpercentile(C.flatten(), 2)
    #     vmaxC = scipy.stats.scoreatpercentile(C.flatten(), 98)
    #     imgC *= vmaxC
    #     for i in xrange(Ncols):
    #         for j in xrange(Nrows):
    #             k = i*Nrows+j # atom index
    #             y, x = j*(W+m), i*(W+m) # coordinates of the top-left corner
    #             imgD[y:y+W,x:x+W] = D[:,subset[k]].reshape((W,W))
    #             imgC[y:y+W,x:x+W] = C[subset[k],:].reshape((W,W))
    #     dpi = 50.
    #     pylab.figure(figsize=(imgD.shape[1]/dpi, imgD.shape[0]/dpi), dpi=dpi)
    #     pylab.imshow(imgD, interpolation='nearest', vmin=vminD, vmax=vmaxD)
    #     pylab.gray()
    #     pylab.xticks(())
    #     pylab.yticks(())
    #     pylab.savefig('mnist_testD.png', dpi=300, bbox_inches='tight', transparent=True)
    #     pylab.figure(figsize=(imgC.shape[1]/dpi, imgC.shape[0]/dpi), dpi=dpi)
    #     pylab.imshow(imgC, interpolation='nearest', vmin=vminC, vmax=vmaxC)
    #     pylab.gray()
    #     pylab.xticks(())
    #     pylab.yticks(())
    #     pylab.savefig('mnist_testC.png', dpi=300, bbox_inches='tight', transparent=True)
    #     #pylab.show()
    #     #1/0

    #     pylab.matshow(NP.dot(D, C))
    #     pylab.title('DC')
    #     pylab.matshow(NP.dot(C, D))
    #     pylab.title('CD')
    #     pylab.matshow(NP.dot(D.T, D))
    #     pylab.title('DtD')
    # #     pylab.matshow(NP.dot(D, D.T))
    # #     pylab.title('DDt')
