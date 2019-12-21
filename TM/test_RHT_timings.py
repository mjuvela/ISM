import os, sys
ISM_DIRECTORY = os.path.expanduser('~/GITHUB')
try:
    ISM_DIRECTORY = os.environ(['ISM_DIRECTORY'])
except:
    pass
sys.path.append(ISM_DIRECTORY)
import ISM.Defs
from   ISM.FITS.FITS import *    
from   scipy.ndimage import zoom
from   ISM.TM.Pattern import RollingHoughTransformBasic

# set the directory where code from http://seclark.github.io/RHT/ is installed
RHT_DIRECTORY = '/home/mika/IN/seclark-RHT-f8b1f3e'

DK = 11
DW = 55
TH = 0.70

SIZES = logspace(log10(100.0), log10(2000.0), 7)
# SIZES = logspace(log10(100.0), log10(200.0), 3)
TIME  = zeros((len(SIZES), 4), float32)

for isize in range(len(SIZES)):    
    F          =  pyfits.open('PSW.fits')
    kzoom      =  SIZES[isize]/float(F[0].data.shape[0])
    F[0].data  =  zoom(F[0].data.copy(), kzoom)
    F.verify('fix')
    F.writeto('test.fits', overwrite=True)
    #
    t0 = time.time()
    os.system('python %s/rht.py -f -w %.0f -s %.0f -t %.3f test.fits' % (RHT_DIRECTORY, DW, DK, TH))
    TIME[isize,0] = time.time()-t0
    #
    t0 = time.time()
    RollingHoughTransformBasic(F, DK, DW, TH, GPU=0)
    TIME[isize,1] = time.time()-t0
    #
    t0 = time.time()
    RollingHoughTransformBasic(F, DK, DW, TH, GPU=1, platforms=[0,])
    TIME[isize,2] = time.time()-t0
    #
    t0 = time.time()
    RollingHoughTransformBasic(F, DK, DW, TH, GPU=1, platforms=[1,])
    TIME[isize,3] = time.time()-t0
    
    
loglog(SIZES, TIME[:,0], 'ks-', label='Python')    
loglog(SIZES, TIME[:,1], 'bo-', label='OpenCL/CPU')    
loglog(SIZES, TIME[:,2], 'go-', label='OpenCL/GPU1')
loglog(SIZES, TIME[:,3], 'ro-', label='OpenCL/GPU2')
legend(loc='upper left')
xlabel('Size (pixels)')
ylabel('Time (s)')

savefig('test_RHT_timings.png')

show()

