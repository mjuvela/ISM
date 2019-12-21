import os, sys
ISM_DIRECTORY = os.path.expanduser('~/GITHUB')
try:
    ISM_DIRECTORY = os.environ(['ISM_DIRECTORY'])
except:
    pass
sys.path.append(ISM_DIRECTORY)
from    ISM.Defs import *
from    ISM.FITS.FITS import CopyFits, MakeEmptyFitsDim
from    ISM.MBB.MBB_MCMC import *
import  matplotlib.ticker as ticker
import  time
import  numpy as np



# Generate some synthetic surface brightness observations
N, M    =  54, 55
J, I    =  indices((N,M))
I250    =  10.0+2.0*cos(0.05*I)+2.0*sin(0.05*J)
T       =  15.0+2.0*sin(0.1*I)
B       =  1.8 + 0.2*sin(0.1*J)

um      =  asarray([160.0, 250.0, 350.0, 500.0], float32)  # wavelengths [um]
freq    =  um2f(um)
FF, dFF =  [], []

for iband in range(len(um)):    
    y    = (I250* ModifiedBlackbody_250(freq[iband], T, B)).reshape(N,M)    
    dy   =  0.03*y
    y   +=  dy*randn(N,M)

    F    =  MakeEmptyFitsDim(1.0, 0.0, 10.0*ARCSEC_TO_RADIAN, N, M)
    dF   =  MakeEmptyFitsDim(1.0, 0.0, 10.0*ARCSEC_TO_RADIAN, N, M)
    F[0].data = y.reshape(N, M)
    dF[0].data = dy.reshape(N, M)
    
    FF.append(F)
    dFF.append(dF)
    
    

# Fits the MBB spectra with python
PI, PT, PB = FitModifiedBlackbody_simple(um, FF, dFF, FF[1][0].data.copy(), 15.0, 1.8)

# Fit the MBB spectra with the OpenCL routine
CI, CT, CB = MBB_fit_CL_FITS(freq, FF, dFF, GPU=0, TMIN=7.0, TMAX=40.0, BMIN=0.5, BMAX=3.5)


figure(1, figsize=(8,7))
rc('font', size=9)
subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.94, wspace=0.29, hspace=0.27)

subplot(331)
imshow(I250)
title(r'$I_0$')
colorbar()

subplot(332)
imshow(T)
title(r'$T_0$')
colorbar()

subplot(333)
imshow(B)
title(r'$\beta_0$')
colorbar()


# second row -- residuals of the Python fits
subplot(334)
X =  PI[0].data - I250
imshow(X)
title(r'$I(Py)-I_0$')
colorbar()

subplot(335)
X =  PT[0].data - T
imshow(X)
title(r'$T(Py)-T_0 \/ (K)$')
colorbar()

subplot(336)
X =  PB[0].data - B
imshow(X)
title(r'$\beta(Py)-\beta_0$')
colorbar()


# third row -- residuals of the OpenCL fits
subplot(337)
X =  CI[0].data - PI[0].data
imshow(X)
title(r'$I(CL)-I(Py)$')
colorbar()

subplot(338)
X =  CT[0].data - PT[0].data
imshow(X)
title(r'$T(CL)-T(Py) \/ (K)$')
colorbar()

subplot(339)
X =  CB[0].data - PB[0].data
imshow(X)
title(r'$\beta(CL)-\beta(Py)$')
colorbar()

savefig('test_MBB_errors.png')

show()






