
import os, glob, gzip, cPickle
import scipy as sp

def checkBSD(path):
    '''
    Cheks that the is the root directory of the standard distribution
    of the Berkeley segmentation dataset, as downloaded from
    http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz
    It returns the absolute paths of the test and training directories.
    '''
    print 'Looking for Berkeley segmentation dataset in', path
    print ' ...',
    if not os.access(path, os.X_OK):
        print 'the path is not accessible. Exiting.'
        sys.exit(0)
    # sets the path holding all images
    imagesdir = os.path.join(path, 'images')
    if not os.access(imagesdir, os.X_OK):
        print 'the images directory (%s) is not accessible. Exiting.' % imagesdir
        sys.exit(0)
    # sets the path holding test images
    testdir = os.path.join(imagesdir, 'test')
    if not os.access(testdir, os.X_OK):
        print 'the test directory (%s) is not accessible. Exiting.' % testdir
        sys.exit(0)
    # sets the path holding training images
    traindir = os.path.join(imagesdir, 'train')
    if not os.access(traindir, os.X_OK):
        print 'the train directory (%s) is not accessible. Exiting.' % traindir
        sys.exit(0)
    print 'OK!'
    return testdir, traindir


def draw_sample(path, W, N):
    '''
    Draws randomly N patches with size WxW from the images in path.
    It returns an ndarray with shape (W*W, N).
    '''
    # list of training images
    trainlist = glob.glob(os.path.join(path, '*.jpg'))
    # number of training images (should be 200)
    Nfiles = len(trainlist)
    if Nfiles != 200:
        print 'WARNING: number of images in train dir is %d (!= 200)' % Nfiles
    # number of patches to sample from each image
    n = N/Nfiles
    if n*Nfiles < N:
        n += 1
    print 'Sampling %d patches each from %d images, for a total of %d patches' % (n, Nfiles, n*Nfiles)
    bytesize = (n*Nfiles*W*W*sp.dtype('f').itemsize)
    print 'Total size about %d MB' % (bytesize/(1024**2))
    # creates the sampling stencil
    stencil = sp.indices((W,W))
    stencil.shape = (2,1,-1)
    # loops over the images
    patches = []
    for fn in trainlist:
        #print ' ...', os.path.basename(fn)
        # load the images
        img = sp.misc.imread(fn)
        assert img.ndim == 3, img.ndim
        assert img.shape[2] == 3, img.shape
        # convert to gray scale
        img = sp.mean(img, 2)
        # recenter
        img -= img.flatten().mean()
        # normalize
        img /= 125.
        # pick random positions for upper left corner
        y = sp.random.randint(0, img.shape[0]-W, size=(1, n))
        x = sp.random.randint(0, img.shape[1]-W, size=(1, n))
        yx = sp.concatenate((y,x), 0)
        assert yx.shape == (2,n), yx.shape
        yx.shape = (2,n,1)
        # create an array with the positions of all pixels in the pacthes
        s = stencil + yx
        assert s.shape == (2,n,W*W), s.shape
        # sample from the image
        p = img[s[0], s[1]]
        assert p.shape == (n,W*W), p.shape
        # append to the list
        patches.append(p)
    # build matrix
    patches = sp.concatenate(patches, 0).T
    return patches

def draw_patches(traindir, W, N):
    # check if a previously drawn sample is already there
    patchesfn = 'BSD_patches_%dx%d_%dk.npz' % (W, W, N/1e3)
    if os.access(patchesfn, os.R_OK):
        print 'Loading a previously drawn sample from', patchesfn
        print 'REMOVE the file IF you want a NEW SAMPLE'
        npz = sp.load(patchesfn)
        Xtrn = npz['Xtrn']
        Xtst = npz['Xtst']
    else:
        # if not there, draw a new sample
        Xtrn = draw_sample(traindir, W, N)
        Xtst = draw_sample(traindir, W, N)
        sp.savez(patchesfn, Xtrn=Xtrn, Xtst=Xtst)
    return Xtrn, Xtst

def loadMNIST(path):
    '''
    Loads the MNIST dataset ...
    http://www.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz
    It returns the absolute paths of the test and training directories.
    '''

    print 'Looking for MNIST dataset in', path
    print ' ...',
    if not os.access(path, os.F_OK):
        print 'the file does not exist. Exiting.'
        sys.exit(0)

    # Load the dataset
    f = gzip.open(path,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    print 'OK!'
    return train_set, valid_set, test_set
