import os, sys
ISM_DIRECTORY = os.path.expanduser('~/GITHUB')
try:
    ISM_DIRECTORY = os.environ(['ISM_DIRECTORY'])
except:
    pass
sys.path.append(ISM_DIRECTORY)
import ISM.Defs
from   ISM.FITS.FITS import *    
from HOG_Aux import *

# set the directory where code from https://github.com/solerjuan/astroHOG is installed
HOG_DIRECTORY = '/home/mika/Python/astroHOG-master/'

sys.path.append(HOG_DIRECTORY)
from astrohog   import *
from astrohog2d import *
import time
from scipy.ndimage import zoom


F1   =  pyfits.open(HOG_DIRECTORY+'/data/image1.fits')
F2   =  pyfits.open(HOG_DIRECTORY+'/data/image2.fits')


# Our version -- CPU
PHI, circstats  =  HOG_images(F1[0].data, F2[0].data, sigma=1.0, GPU=0)

# JS version
image1 = F1[0].data
image2 = F2[0].data
circstats, corrframe, smoothframe1, smoothframe2 = HOGcorr_frame(image1, image2)

rc('font', size=9)
subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.93, wspace=0.25, hspace=0.3)


# plot the angle maps .... in degrees
corrframe *= RADIAN_TO_DEGREE
PHI       *= RADIAN_TO_DEGREE

ax = subplot(221)
imshow(PHI)
title('OpenCL')
colorbar()
text(1.34, 0.5, r'$\phi \/ \/ \rm (degrees)$', ha='center', va='center', transform=ax.transAxes, rotation=90)

ax = subplot(222)
imshow(corrframe)
title('Python')
colorbar()
text(1.34, 0.5, r'$\phi \/ \/ \rm (degrees)$', ha='center', va='center', transform=ax.transAxes, rotation=90)

ax = subplot(223)
X  = PHI-corrframe
imshow(PHI-corrframe, vmin=-10, vmax=10)
title("OpenCL-Python")
colorbar()
text(1.34, 0.5, r'$\Delta \phi \/ \/ \rm (degrees)$', ha='center', va='center', transform=ax.transAxes, rotation=90)

ax = subplot(224)
X = (PHI-corrframe)/(corrframe+0.1) # add 0.1 degrees to avoid division by zero
imshow(X, vmin=-0.1, vmax=0.1)
title("(OpenCL-Python)/Python")
colorbar()
text(1.42, 0.5, r'$\rm Relative \/ \/ error$', ha='center', va='center', transform=ax.transAxes, rotation=90)

show()



