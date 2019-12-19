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
 

N            =  256   # map size
pix          =  10.0*ARCSEC_TO_RADIAN
FITS         =  MakeEmptyFitsDim(1.0, 0.0, pix, N, N)
FITS[0].data =  asarray(3.0+randn(N,N), float32)
FITS[0].data =  clip(FITS[0].data, 1.0, 10.0)

fwhm_pix  =  7.3
fwhm_rad  =  fwhm_pix*pix
sigma_pix =  fwhm_pix/sqrt(8.0*log(2.0))
kernel    =  Gaussian2DKernel(x_stddev=sigma_pix)
if (kernel.shape[0]%2==0): 
    print("kernel dimensions must be odd") 
    sys.exit()
            
AP = astropy_convolve(FITS[0].data.copy(), kernel, boundary='extend')
if (0):
    CL = ConvolveFitsBeamPyCL(FITS, kernel, border='ignore')[0].data.copy()
else:
    CL = ConvolveFitsBeamPyCL(FITS, kernel, border='nearest')[0].data.copy()

    
figure(1, figsize=(8,6))
rc('font', size=9)
subplots_adjust(left=0.09, right=0.94, bottom=0.09, top=0.95, wspace=0.26, hspace=0.3)
    
subplot(221)
imshow(AP)
title("AP=aplpy.convolve")
colorbar()

subplot(222)
imshow(CL)
title("CL=ConvolveFitsBeamPyCL")
colorbar()

subplot(223)
X = CL-AP
imshow(X)
title("CL-AP")
colorbar()

subplot(224)
X = (CL-AP)/AP
imshow(X)
title("(CL-AP)/AP")
colorbar()


savefig('test_convolution_errors.png')

show()

    
