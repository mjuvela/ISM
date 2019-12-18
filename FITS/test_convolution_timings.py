import os, sys
ISM_DIRECTORY = os.path.expanduser('~/GITHUB')
try:
    ISM_DIRECTORY = os.environ(['ISM_DIRECTORY'])
except:
    pass
sys.path.append(ISM_DIRECTORY)
from    ISM.Defs import *
from    ISM.FITS.FITS import MakeEmptyFitsDim, Reproject, convolve_map_fast, ConvolveFitsPyCL, ConvolveFitsBeamPyCL
import  montage_wrapper as montage
import  matplotlib.ticker as ticker
from    astropy.convolution import convolve as astropy_convolve
from    astropy.convolution import Gaussian2DKernel
from    scipy.signal import convolve as scipy_convolve
import  time
import  numpy as np
 
SLEEP        =  10     # optionally sleep between calls (reduce CPU throttling)
N            =  1280   # map size
pix          =  10.0*ARCSEC_TO_RADIAN
FITS         =  MakeEmptyFitsDim(1.0, 0.0, pix, N, N)
FITS[0].data =  asarray(3.0+randn(N,N), float32)
FITS[0].data =  clip(FITS[0].data, 1.0, 10.0)


fwhm_values = asarray([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], float32)
# fwhm_values = asarray([2.0, 4.0, 8.0], float32)
TIME = zeros((len(fwhm_values), 6), float32)

if (1): # redo the computations
    for i in range(len(fwhm_values)):
        print(fwhm_values[i])
    
        fwhm_pix  =  fwhm_values[i]
        fwhm_rad  =  fwhm_pix*pix
        sigma_pix =  fwhm_pix/sqrt(8.0*log(2.0))
        kernel    =  Gaussian2DKernel(x_stddev=sigma_pix)
        if (kernel.shape[0]%2==0): 
            print("kernel dimensions must be odd") 
            sys.exit()
            
        for iprog in range(6):
            do_it = 1
            if (i>0):
                if (TIME[i-1,iprog]>10.0):
                    do_it = 0  # it was already taking too long
                    TIME[i, iprog] = 1e10
            if (1): # we have problem on the Intel GPU -- i915 bug prevents long runs ??
                if ((iprog==4)&(fwhm_pix>32)): do_it = False
            if (do_it):
                time.sleep(SLEEP)
                t0 = time.time()
                if (iprog==0):
                    A = convolve_map_fast(FITS[0].data.copy(), fwhm_pix, radius=1.7*fwhm_pix)
                elif (iprog==1):
                    B = scipy_convolve(FITS[0].data.copy(), kernel, mode='same', method='direct')                
                elif (iprog==2):
                    C = astropy_convolve(FITS[0].data.copy(), kernel)
                elif (iprog==3):
                    # ConvolveFitsPyCL(F, fwhm_rad, fwhm_orig=0.0, RinFWHM=1.7, GPU=0, platforms=[2,])
                    D = ConvolveFitsBeamPyCL(FITS, kernel, GPU=0, platforms=[2,])[0].data.copy()
                elif (iprog==4):  # on my system Intel GPU
                    # ConvolveFitsPyCL(F, fwhm_rad, fwhm_orig=0.0, RinFWHM=1.7, GPU=1, platforms=[0,])
                    E = ConvolveFitsBeamPyCL(FITS, kernel, GPU=1, platforms=[0,])[0].data.copy()
                elif (iprog==5):  # on my system NVidia GPU
                    # ConvolveFitsPyCL(F, fwhm_rad, fwhm_orig=0.0, RinFWHM=1.7, GPU=1, platforms=[1,])
                    F = ConvolveFitsBeamPyCL(FITS, kernel, GPU=1, platforms=[1,])[0].data.copy()
                TIME[i, iprog] = time.time()-t0
        if (i==-1):
            clf()
            subplot(221)
            imshow(C)
            colorbar()
            subplot(222)
            imshow(E)
            colorbar()
            subplot(223)
            imshow(E-C)
            colorbar()
            subplot(224)
            imshow((E-C)/C)
            colorbar()
            show()
        print("   ", TIME[i,:])
    
    asarray(TIME, float32).tofile('test_convolution_timings.dat')
else:  # read saved file for the run times
    TIME = fromfile('test_convolution_timings.dat', float32).reshape(  len(fwhm_values), 6 )

    
clf()
m = nonzero((TIME[:,0]<9999.0)&(TIME[:,0]>0.001))
loglog(fwhm_values[m], TIME[:,0][m], 'ko-', label='FFT')
m = nonzero((TIME[:,1]<9999.0)&(TIME[:,1]>0.001))
loglog(fwhm_values[m], TIME[:,1][m], 'bo-', label='Scipy')
m = nonzero((TIME[:,2]<9999.0)&(TIME[:,2]>0.001))
loglog(fwhm_values[m], TIME[:,2][m], 'go-', label='astropy')
m = nonzero((TIME[:,3]<9999.0)&(TIME[:,3]>0.001))
loglog(fwhm_values[m], TIME[:,3][m], 'co-', label='CL/CPU')
m = nonzero((TIME[:,4]<9999.0)&(TIME[:,4]>0.001))
loglog(fwhm_values[m], TIME[:,4][m], 'mo-', label='CL/GPU1')
m = nonzero((TIME[:,5]<9999.0)&(TIME[:,5]>0.001))
loglog(fwhm_values[m], TIME[:,5][m], 'ro-', label='CL/GPU2')
legend(loc='upper left')
xlabel("FWHM [pixels]")
ylabel("Time (s)")
savefig('test_convolution_timings.png')

show()

    
