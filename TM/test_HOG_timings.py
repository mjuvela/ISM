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


if (1):  # redo the computations, python code of Juan Soler and OpenCL version on CPU and GPU
    zzz = 5.0  # sleep between runs
    fp = open('HOG_timings.txt', 'w')
    for SCALE in [0.5, 1.0, 2, 4, 8, 16]: # loop over different image sizes
        print("*** SCALE %.2f ***" % SCALE)
        F1   =  pyfits.open(HOG_DIRECTORY+'/data/image1.fits')
        F2   =  pyfits.open(HOG_DIRECTORY+'/data/image2.fits')
        F1[0].data  =  zoom(F1[0].data, SCALE)
        F2[0].data  =  zoom(F2[0].data, SCALE)
        F1.writeto('1.fits', overwrite=True)
        F2.writeto('2.fits', overwrite=True)    
        # Our version -- CPU
        t0   =  time.time()
        PHI, circstats  =  HOG_images(F1[0].data, F2[0].data, sigma=1.0, GPU=0)
        t0   =  time.time()-t0
        print("CPU: %.3f seconds" % t0)
        time.sleep(zzz)
        # Our version -- GPU  --- our external GPU is on platform=1
        t1   =  time.time()
        PHI, circstats  =  HOG_images(F1[0].data, F2[0].data, sigma=1.0, GPU=1, platforms=[1,])
        t1   =  time.time()-t1
        print("GPU: %.3f seconds" % t1)
        time.sleep(zzz)
        # JS version
        if (SCALE<=16):  # running out of memory for larger maps?
            image1 = F1[0].data
            image2 = F2[0].data
            t2 = time.time()
            circstats, corrframe, smoothframe1, smoothframe2 = HOGcorr_frame(image1, image2)
            t2   =  time.time()-t2    
            print("Python: %.3f seconds" % t2) 
            time.sleep(zzz)
        else:
            t2 = 1e10
        fp.write('%7.2f     %8.3f  %8.3f  %8.3f\n' % (SCALE, t2, t0, t1))    
    fp.close()


plt.rcParams['axes.formatter.min_exponent'] = 3

figure(1)
ax = axes()
d = loadtxt('HOG_timings.txt')
xlabel('Image size x  [512x512]', size=13)
ylabel('Time (seconds)', size=13)
title('HOG', size=13, weight='bold')

m = nonzero(d[:,1]<9999.0)
loglog(d[:,0][m], d[:,1][m], 'k^-', label='Python')
loglog(d[:,0], d[:,2], 'bs-', label='OpenCL/CPU')
loglog(d[:,0], d[:,3], 'ro-', label='OpenCL/GPU')

if (0):
    text(2.3, 320,  'Python',      rotation=20)
    text(2.3, 1.8,  'OpenCL / CPU', rotation=16)
    text(2.3, 0.23, 'OpenCL / GPU',  rotation=10)

legend(loc='lower right')
# grid()
savefig('HOC_timings.png')

    
show()
