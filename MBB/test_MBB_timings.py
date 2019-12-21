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



NN     =  asarray(logspace(0.5, 3.0, 7), int32)
RES    =  zeros((len(NN), 4), float32)


for itest in range(0, len(NN)):
    N       =  NN[itest]
    M       =  NN[itest]    
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
    # MBB fits
    # (1) python
    t0 = time.time()
    PI, PT, PB = FitModifiedBlackbody_simple(um, FF, dFF, FF[1][0].data.copy(), 15.0, 1.8)
    RES[itest, 0] = time.time()-t0
    # (2) OpenCL on CPU
    t0 = time.time()
    CI, CT, CB = MBB_fit_CL_FITS(freq, FF, dFF, GPU=0)
    RES[itest, 1] = time.time()-t0
    # (3) OpenCL on GPU1
    t0 = time.time()
    CI, CT, CB = MBB_fit_CL_FITS(freq, FF, dFF, GPU=1, platforms=[0,])
    RES[itest, 2] = time.time()-t0
    # (3) OpenCL on GPU2
    t0 = time.time()
    CI, CT, CB = MBB_fit_CL_FITS(freq, FF, dFF, GPU=1, platforms=[1,])
    RES[itest, 3] = time.time()-t0
    
    if (0): #
        PT.writeto('PT.fits', overwrite=True)
        PB.writeto('PB.fits', overwrite=True)
        CT.writeto('CT.fits', overwrite=True)
        CB.writeto('CB.fits', overwrite=True)
        sys.exit()
    
    
figure(1, figsize=(8,7))
rc('font', size=9)

loglog(NN, RES[:,0], 'ks-', label='Python')
loglog(NN, RES[:,1], 'bo-', label='OpenCL/CPU')
loglog(NN, RES[:,2], 'go-', label='OpenCL/GPU1')
loglog(NN, RES[:,3], 'ro-', label='OpenCL/GPU2')
legend(loc='upper left')
xlabel('Image size')
ylabel('Time (s)')
savefig('test_MBB_timings.png')

show()


