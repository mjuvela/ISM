import os, sys
ISM_DIRECTORY = os.path.expanduser('~/GITHUB')
try:
    ISM_DIRECTORY = os.environ(['ISM_DIRECTORY'])
except:
    pass
sys.path.append(ISM_DIRECTORY)
from    ISM.Defs import *
from    ISM.FITS.FITS import MakeEmptyFitsDim, Reproject
import  montage_wrapper as montage


"""
Resample an image onto the pixels of another image using Montage reproject()  and
compare run times with the  OpenCL routine that is run on CPU or on GPU. In this
script we assume that we have two GPUSs -- we check the run times with one and then
with the other (see the platform parameters of the OpenCL Reproject routine).
   speedup   CPU         =  61.804 /  14.50   =    4.3
   speedup   GPU/Intel   =  61.804 /   4.403  =   14.0
   speedup   GPU/NVidia  =  61.804 /   1.299  =   47.6
"""

CSTEP    =  10
SLEEP    =  2   # we may sleep before each calculation, to reduce throttling (?)
np.random.seed(12345)
offx     =  -1.3001*ARCMIN_TO_RADIAN
offy     =  -1.2000*ARCMIN_TO_RADIAN
scale    =  1.0
factor   =  1.0
threads  =  1

# DIMS =  np.asarray(np.logspace(np.log10(32.0), np.log10(16384.2), 10),  int32)
# DIMS =  np.asarray(np.logspace(np.log10(32.0), np.log10( 8192.2),  9),  int32)
# DIMS =  np.asarray(np.logspace(np.log10(32.0), np.log10( 4096.2),  8),  int32)
# DIMS =  np.asarray(np.logspace(np.log10(32.0), np.log10(1024.1), 6), int32)
DIMS =  np.asarray(np.logspace(np.log10(32.0), np.log10(512.1), 5), int32)
TA   =  np.zeros(len(DIMS), np.float32)
TC   =  np.zeros((len(DIMS), 4), np.float32)   # tP, tB, tK, TC
TG1  =  np.zeros((len(DIMS), 4), np.float32)   # tP, tB, tK, TC
TG2  =  np.zeros((len(DIMS), 4), np.float32)   # tP, tB, tK, TC
for III in range(len(DIMS)):  # loop over different map sizes
    # Generate the input map
    N, M = int(DIMS[III]), int(DIMS[III])
    print("--- %d %d---" % (M, M))
    A = MakeEmptyFitsDim(1.0,      0.0,      10.0*DEGREE_TO_RADIAN/N      , N, M)
    B = MakeEmptyFitsDim(1.0,      0.0,      10.0*DEGREE_TO_RADIAN/N      , N, M)
    G = MakeEmptyFitsDim(1.0+offx, 0.0+offy, 10.0*DEGREE_TO_RADIAN/N*scale, N, M)
    G[0].data = np.asarray(randn(N, M), np.float32)+5.0
    J, I      = indices((N, M), np.float32)
    for i in range(30):
        K          =  4.0*log(2.0)/(clip(2.0+1.0*randn(), 2.0, 99)**2.0)
        G[0].data +=  100.0*rand()*exp(-K*((J-N*rand())**2.0+(I-M*rand())**2.0))
    G[0].data[0:3,:] = 0.0
    G[0].data[-3:,:] = 0.0
    G[0].data[:,0:3] = 0.0
    G[0].data[:,-3:] = 0.0
    if (1):  # rotate the input image
        rot = 30*DEGREE_TO_RADIAN
        G[0].header['CD1_1'] =  G[0].header['CDELT1']*cos(rot)
        G[0].header['CD1_2'] = -G[0].header['CDELT2']*sin(rot)
        G[0].header['CD2_1'] =  G[0].header['CDELT1']*sin(rot)
        G[0].header['CD2_2'] =  G[0].header['CDELT2']*cos(rot)
    G.writeto('g.fits', overwrite=True)
    
    # Run montage.reproject
    time.sleep(SLEEP)
    t0 = time.time()
    A[0].header.totextfile('ref.header', overwrite=True)
    montage.reproject('g.fits', 'A.fits', 'ref.header', exact_size=True, factor=factor)
    TA[III] = time.time()-t0

    # Run the OpenCL routine on CPU
    time.sleep(SLEEP)    
    t0 = time.time()
    tP, tB, tK  = Reproject(G, B, GPU=0, platforms=[0,1,2,3,4], cstep=CSTEP, threads=threads)  # CPU
    TC[III, 0 ] = time.time()-t0
    TC[III, 1:] = [tP, tB, tK]
    
    # Run the OpenCL routine on the first GPU --- assuming that is platform 0
    time.sleep(SLEEP)    
    t0 = time.time()
    tP, tB, tK  = Reproject(G, B, GPU=1, platforms=[0,], cstep=CSTEP, threads=threads)  # Intel HD Graphics
    TG1[III, 0 ] = time.time()-t0
    TG1[III, 1:] = [tP, tB, tK]            
    
    if (0):
        # Run the OpenCL routine on the second GPU --- assuming that is platform 1
        time.sleep(SLEEP)    
        t0 = time.time()
        tP, tB, tK  = Reproject(G, B, GPU=1, platforms=[1,], cstep=CSTEP, threads=threads)  # NVidia GTX 1080 Ti
        TG2[III, 0 ] = time.time()-t0
        TG2[III, 1:] = [tP, tB, tK]            
    
res          =  np.zeros((len(DIMS), 13), np.float32)
res[:,0]     =  TA
res[:,1:5]   =  TC
res[:,5:9]   =  TG1
res[:,9:13]  =  TG2
np.asarray(res, np.float32).tofile('drizzle_timings.dat')


figure(1, figsize=(8,6))
rc('font', size=9)
loglog(DIMS, TA, 'ks-', lw=3, ms=10, label='Montage')

loglog(DIMS, TC[:,0], 'bo-', lw=3, ms=10, label='CL/CPU')
loglog(DIMS, TC[:,1], 'b:')
loglog(DIMS, TC[:,2], 'b-.')
loglog(DIMS, TC[:,3], 'b--')

loglog(DIMS, TG1[:,0], 'ro-', lw=3, ms=10, label='CL/GPU1')
loglog(DIMS, TG1[:,1], 'r:',  label='Python')
loglog(DIMS, TG1[:,2], 'r-.', label='Build')
loglog(DIMS, TG1[:,3], 'r--', label='Kernel')

if (0):
    loglog(DIMS, TG2[:,0], 'go-', lw=3, ms=10, label='CL/GPU2')
    loglog(DIMS, TG2[:,1], 'g:',  label='Python')
    loglog(DIMS, TG2[:,2], 'g-.', label='Build')
    loglog(DIMS, TG2[:,3], 'g--', label='Kernel')

xlabel(r'$\rm Map \/ \/ dimension \/ \/ \/ (pixels)$')
ylabel(r'$\rm Time \/ \/ \/ (s)$')

legend(loc='lower right')
savefig('test_drizzle.png')

show()

