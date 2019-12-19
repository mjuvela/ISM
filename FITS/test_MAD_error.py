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


N, n    =  128, 5

X       =  clip(2.0+randn(N, N), 1.0, 10.0)
A       =  zeros((N, N), float32)
generic_filter(X.copy(), fun, size=(2*n+1,2*n+1), output=A)

B       =  MADFilterCL(X.copy(), R1=0, R2=n, circle=0, GPU=0)

# the borders will be different => cut the result images down
A       =  A[n:(-n), n:(-n)]
B       =  B[n:(-n), n:(-n)]

rc('font', size=9)
subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.94, wspace=0.3, hspace=0.27)

subplot(221)
imshow(A)
title("Scipy")
colorbar()

subplot(222)
imshow(B)
title("OpenCL")
colorbar()

subplot(223)
imshow(B-A)
title("OpenCL-Scipy")
colorbar()

subplot(224)
imshow((B-A)/A)
title("(OpenCL-Scipy)/Scipy")
colorbar()

savefig("test_MAD_error.png")

show()


