import os, sys
ISM_DIRECTORY = os.path.expanduser('~/GITHUB')
try:
    ISM_DIRECTORY = os.environ(['ISM_DIRECTORY'])
except:
    pass
sys.path.append(ISM_DIRECTORY)
from    ISM.Defs import *
from    ISM.FITS.FITS import MADFilterCL
from    scipy.ndimage import generic_filter

import  time
import  numpy as np


def fun(x):
    return np.median(np.abs(x-np.median(x)))


NN = np.asarray(np.logspace(1.5, 4.01, 10), np.int32)
nn = [ 3, 7, 15, 31, 63, 127 ]


# First test -- constant filter size, scale the image size
n = 5
RES = zeros((len(NN), 3), float32)
fp = open("test_MAD_timings.dat", "w")
for i in range(len(NN)):
    N        =  NN[i]
    X        =  clip(2.0+randn(N, N), 1.0, 10.0)
    print("%2d/%2d -- map size %d" % (i, len(NN), N))    
    if (N<3000):
        t        =  time.time()
        A        =  zeros((N, N), float32)
        generic_filter(X.copy(), fun, size=(2*n+1,2*n+1), output=A)
        RES[i,0] =  time.time()-t
        print("   Scipy        %.2f seconds" % RES[i,0])    
    t        =  time.time()
    B        =  MADFilterCL(X.copy(), R1=0, R2=n, circle=0, GPU=0)
    RES[i,1] =  time.time()-t
    print("   OpenCL/CPU   %.2f seconds" % RES[i,1])
    t        =  time.time()
    B        =  MADFilterCL(X.copy(), R1=0, R2=n, circle=0, GPU=1, platforms=[1,])
    RES[i,2] =  time.time()-t
    print("   OpenCL/GPU   %.2f seconds" % RES[i,2])
    fp.write("%6d   %10.3f %10.3f %10.3f\n" % (N, RES[i,0], RES[i,1], RES[i,2]))
    fp.flush()
fp.close()    
    
clf()
m = nonzero(RES[:,0]>0.0)
loglog(NN[m], RES[:,0][m], 'ks-', label='Scipy')
loglog(NN, RES[:,1], 'bo-', label='OpenCL/CPU')
loglog(NN, RES[:,2], 'ro-', label='OpenCL/GPU')
legend(loc='upper left')
xlabel("Map size")
ylabel("Time (s)")

savefig("test_MAD_timings.png")

show()


