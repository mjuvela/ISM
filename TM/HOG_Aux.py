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

import pyopencl as cl
import time
import pycircstat as circ
import scipy
from   scipy import stats


def HOG_PRS(phi):
    """
    Calculates the projected Rayleigh statistic of the distributions of angles phi.
    Input:
         phi      - angles between -pi/2 and pi/2
    Output:
        Zx       - value of the projected Rayleigh statistic   
        s_Zx     - 
        meanPhi  -
    """
    angles  =  phi #2.*phi
    Zx      =  np.sum(np.cos(angles)) / np.sqrt(np.size(angles)/2.)
    temp    =  np.sum(np.cos(angles)*np.cos(angles))
    s_Zx    =  np.sqrt((2.*temp-Zx*Zx)/np.size(angles))    

    Zy      =  np.sum(np.sin(angles)) / np.sqrt(np.size(angles)/2.)
    temp    =  np.sum(np.sin(angles)*np.sin(angles))
    s_Zx    =  np.sqrt((2.*temp-Zy*Zy)/np.size(angles))
    meanPhi =  0.5*np.arctan2(Zy, Zx)    
    return Zx, s_Zx, meanPhi


def HOG_AM(phi):
    """
    Calculate the alignment measure.
    Input:
        phi      - angles between -pi/2 and pi/2
    Output:
        AM       - value of the alignment measure
    """
    angles = phi
    ami    = 2.*np.cos(phi)-1.
    am     = np.mean(ami)
    return am
                                                                                       
                                                                                       
def HOG_auto(S, sigma, GPU=0, platforms=[0,1,2,3,4,5]):
    """
    Calculate gradients and the v and r arrays.
    Input:
        S     =  input cube
        sigma =  stdev of the convolving Gaussian
        GPU   =  if>0, use GPU
    """
    t0           =   time.time()
    # Run kernel Gradient to get gradient vectors for all images (l,m).
    # Run kernel Sums to calculate V[l,m] and r[l,m].
    NV, NY, NX   =  S.shape                    # first axis is the velocity
    Gx           =  zeros((NV, NY, NX), float32)
    Gy           =  zeros((NV, NY, NX), float32)    
    platform, device, context, queue, mf = InitCL(GPU=0, platforms=platforms)
    LOCAL = [ 8, 64][GPU>0]
    OPT       = " -D N=%d -D M=%d -D SIGMA=%.3ff -cl-fast-relaxed-math" % (NY, NX, sigma)
    source    =  open(ISM_DIRECTORY+"/ISM/TM/kernel_HOG.c").read()
    program   =  cl.Program(context, source).build(OPT)
    S_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*NX*NY)
    Gx_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NX*NY)
    Gy_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NX*NY)
    Hx_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NX*NY)
    Hy_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NX*NY)
    # First kernel: gradients
    IGRAD     =  program.Gradient  # kernel to calculate gradient vectors for one 2d image
    IGRAD.set_scalar_arg_dtypes([None, None, None])
    GLOBAL    =  (int((NX*NY)/64)+1)*64
    S         = asarray(S, float32)
    # Second kernel: sums
    SUMS      =  program.Sums      # kernel to calculate sums of W, W**2, W*cos, W*sin
    SUMS.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None])
    GLOBAL2   =  2048              # number of work items for the latter Sums kernel
    CSUM_buf  =  cl.Buffer(context, mf.WRITE_ONLY, 4*GLOBAL)
    SSUM_buf  =  cl.Buffer(context, mf.WRITE_ONLY, 4*GLOBAL)
    WSUM_buf  =  cl.Buffer(context, mf.WRITE_ONLY, 4*GLOBAL)
    W2SUM_buf =  cl.Buffer(context, mf.WRITE_ONLY, 4*GLOBAL)
    CSUM      =  zeros(GLOBAL2, float32)
    SSUM      =  zeros(GLOBAL2, float32)
    WSUM      =  zeros(GLOBAL2, float32)
    W2SUM     =  zeros(GLOBAL2, float32)
    for k in range(NV):            # Calculate gradient images for each image plane
        cl.enqueue_copy(queue, S_buf, S[k,:,:])
        IGRAD(queue, [GLOBAL,], [LOCAL,], S_buf, Gx_buf, Gy_buf)
        cl.enqueue_copy(queue, Gx[k,:,:], Gx_buf)
        cl.enqueue_copy(queue, Gy[k,:,:], Gy_buf)
    # Calculate V and r for each combination of planes
    VLM = zeros((NV, NV), float32)
    RLM = zeros((NV, NV), float32)
    for i in range(NV):
        cl.enqueue_copy(queue, Gx_buf, Gx[i,:,:])            # gradients for the first image plane
        cl.enqueue_copy(queue, Gy_buf, Gy[i,:,:])
        for j in range(NV):                                  # for auto-HOG, could skip half of (i,j) calculations...
            cl.enqueue_copy(queue, Hx_buf, Gx[j,:,:])        # gradients for the second image plane
            cl.enqueue_copy(queue, Hy_buf, Gy[j,:,:])
            SUMS(queue, [GLOBAL2,], [LOCAL,], Gx_buf, Gy_buf, Hx_buf, Hy_buf, CSUM_buf, SSUM_buf, WSUM_buf, W2SUM_buf)
            cl.enqueue_copy(queue, CSUM,  CSUM_buf)
            cl.enqueue_copy(queue, SSUM,  SSUM_buf)
            cl.enqueue_copy(queue, WSUM,  WSUM_buf)
            cl.enqueue_copy(queue, W2SUM, W2SUM_buf)        
            res       =  sum(CSUM) / sqrt(0.5*sum(W2SUM))    # V[l,m]
            VLM[i,j]  =  res
            res       =  sqrt( (sum(CSUM))**2.0 + (sum(SSUM))**2.0 ) / sum(WSUM)  # r[l,m]
            RLM[i,j]  =  res
    # asarray(VLM, float32).tofile('VLM.bin')
    # asarray(RLM, float32).tofile('RLM.bin')
    t0   = time.time()-t0
    print("HOG: %.3f seconds" % t0)
    return VLM, RLM




def HOG_images(A, B, sigma, gthresh1=1.0e-12, gthresh2=1.0e-12, 
               intmask1=asarray([],int32), intmask2=asarray([],int32), 
               GPU=0, platforms=[0,1,2,3,4,5]):
    """
    Run HOG for a pair of 2D images.
    Input:
        A        =   first image (2D numpy array)
        B        =   second image (2D numpy array)
                     images must be already resampled onto same pixels
        gthresh1 =   gradient threshold for the first image
        gthresh2 =   gradient threshold for the second image
    Return:
        phi      =   array of the relative angles (gradients in two images)
        circstat =   [rvl, Z, V, pz, pv, myV, s_myV, meanphi, am] (see the code!)
    """
    print("")
    t0           =  time.time()
    NY, NX       =  A.shape
    # OpenCL initialisations
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms)    
    LOCAL     =  [ 8, 64][GPU>0]
    OPT       = " -D N=%d -D M=%d -D SIGMA=%.3ff -cl-fast-relaxed-math" % (NY, NX, sigma)
    source    =  open(ISM_DIRECTORY+"/ISM/TM/kernel_HOG.c").read()
    program   =  cl.Program(context, source).build(OPT)
    # Prepare buffers to transfer data between host and device
    I1_buf    =  cl.Buffer(context, mf.READ_ONLY,   4*NX*NY)
    I2_buf    =  cl.Buffer(context, mf.READ_ONLY,   4*NX*NY)
    PHI_buf   =  cl.Buffer(context, mf.READ_WRITE,  4*NX*NY)
    # The kernel is called CFRAME -- calculates both the gradients and the various sums
    GRAD      =  program.HOG_gradients
    GRAD.set_scalar_arg_dtypes([None, None, None, np.float32, np.float32])
    GLOBAL    =  (int((NX*NY)/64)+1)*64    
    print("--------   HOG_images -- initialisation took      ---- %.3f seconds" % (time.time()-t0))
    t1        =  time.time()
    cl.enqueue_copy(queue, I1_buf, asarray(A, float32))
    cl.enqueue_copy(queue, I2_buf, asarray(B, float32))
    GRAD(queue, [GLOBAL,], [LOCAL,], I1_buf, I2_buf, PHI_buf, gthresh1, gthresh2)
    phi       =  zeros((NY, NX), float32)
    cl.enqueue_copy(queue, phi, PHI_buf)
    t1        =  time.time()-t1
    print("--------   HOG_images computation took            ---- %.3f seconds" % t1)
    
    # Note: masking for small gradients is done inside the kernel
    # Apply external masks at this point to phi array.
    if (A.shape == intmask1.shape):
        phi[nonzero(intmask1==0.0)] = np.nan
        if (B.shape == intmask2.shape):
            phi[nonzero(intmask2==0.0)] = np.nan
    good  = np.nonzero(np.isfinite(phi))
    ngood = len(good[0])
    # print("NPIX %d, GOOD %d" % (size(phi), ngood))
    
    weights  =  ones((NY, NX), float32) *  1.0/sigma**2.0        
    
    # In the HOG_sums kernel we want to limit number of workers because that
    # affects the data transfer and the size of the arrays that the host needs to process.
    t2          =  time.time()
    GLOBAL      =  4096
    NPIX        =  size(phi)
    # This kernel replaces HO_PRS() routine
    HOG_PRS_CL  =  program.HOG_sums
    HOG_PRS_CL.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, None, None])
    scos_buf    =  cl.Buffer(context, mf.READ_WRITE,  4*GLOBAL)
    ssin_buf    =  cl.Buffer(context, mf.READ_WRITE,  4*GLOBAL)
    scos2_buf   =  cl.Buffer(context, mf.READ_WRITE,  4*GLOBAL)
    ssin2_buf   =  cl.Buffer(context, mf.READ_WRITE,  4*GLOBAL)
    wscos_buf   =  cl.Buffer(context, mf.READ_WRITE,  4*GLOBAL)
    wssin_buf   =  cl.Buffer(context, mf.READ_WRITE,  4*GLOBAL)
    wscos2_buf  =  cl.Buffer(context, mf.READ_WRITE,  4*GLOBAL)
    wssin2_buf  =  cl.Buffer(context, mf.READ_WRITE,  4*GLOBAL)
    weight_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NX*NY)
    scos        =  zeros(GLOBAL, float32)
    ssin        =  zeros(GLOBAL, float32)
    scos2       =  zeros(GLOBAL, float32)
    ssin2       =  zeros(GLOBAL, float32)
    wscos       =  zeros(GLOBAL, float32)
    wssin       =  zeros(GLOBAL, float32)
    wscos2      =  zeros(GLOBAL, float32)
    wssin2      =  zeros(GLOBAL, float32)
    t2 = time.time()-t2
    print("--------   preparations for second kernel         ---- %.3f seconds" % t2)
    t3 = time.time()
    # phi is already on the device... but perhaps we masked something on the host side -> copy again to device
    cl.enqueue_copy(queue, PHI_buf,  phi)
    cl.enqueue_copy(queue, weight_buf, weights)
    # use kernel to calculate partial sums
    HOG_PRS_CL(queue, [GLOBAL,], [LOCAL,], PHI_buf, 
    scos_buf,   ssin_buf,  scos2_buf, ssin2_buf, weight_buf, wscos_buf, wssin_buf, wscos2_buf, wssin2_buf)
    queue.finish()
    cl.enqueue_copy(queue, scos,   scos_buf)
    cl.enqueue_copy(queue, ssin,   scos_buf)
    cl.enqueue_copy(queue, scos2,  scos_buf)
    cl.enqueue_copy(queue, ssin2,  scos_buf)
    cl.enqueue_copy(queue, wscos,  scos_buf)
    cl.enqueue_copy(queue, wssin,  scos_buf)
    cl.enqueue_copy(queue, wscos2, scos_buf)
    cl.enqueue_copy(queue, wssin2, scos_buf)
    queue.finish()
    t3 = time.time()-t3
    print("--------   second kernel, get results             ---- %.3f seconds" % t3)
    t4 = time.time()
    # Even in sums calculated without weighting only good elements are counted = ngood elements.
    #   --> denominator should be ngood rather than npix.
    # am ==  < 2*cos(phi)-1>  =  2<cos(phi)>-1, can be calculated using scos
    am       =  2.0 * sum(scos)/ngood  - 1.0
    # final total sums
    Scos     =  sum(scos) / sqrt(NPIX/2.0)   # Zx  -- why this denominator
    Ssin     =  sum(ssin) / sqrt(NPIX/2.0)   # Zy
    Scos2    =  sum(scos2)
    Ssin2    =  sum(ssin2)
    #print("am = %.3e,     (ZX,ZY) = (%.4e, %.4e)" % (am, Zx, Zy))
    ### s_Zx     =  np.sqrt((2.0*temp1-Zx*Zx)/NPIX)  <--- unnecessary!
    s_Zx     =  np.sqrt((2.0*Ssin2-Ssin*Ssin)/NPIX)
    meanPhi  =  0.5*np.arctan2(Ssin, Scos)
    #print('* HOG_PRS_CL   %.3f seconds:  %10.3e %10.3e %10.3e' % (time.time()-t01, Zx, s_Zx, meanPhi))
    # HOC_AM() is also replaced by the above  --  AM =  <2*cos(phi)-1>
    # ...........................
    # The rest is the calculation of circstat  statistics.
    # ***circ.descriptive.resultant_vector_length***
    WScos    =  sum(wscos)/ngood
    WSsin    =  sum(wssin)/ngood
    t4  = time.time()-t4
    print("--------   postprocessing data from second kernel ---- %.3f seconds" % t4)

    t5 = time.time()
    # ***HOG_PRS(2*phi)*** -- as above but with argument twice phi
    myV, s_myV, meanphi = None, None, None
    # NaN pixels are already ignored ---- masked pixels should also be set to NaN above
    cl.enqueue_copy(queue, PHI_buf,  2.0*phi)
    # ---
    HOG_PRS_CL(queue, [GLOBAL,], [LOCAL,], PHI_buf, 
    scos_buf,   ssin_buf,  scos2_buf, ssin2_buf, weight_buf, wscos_buf, wssin_buf, wscos2_buf, wssin2_buf)
    # ---
    cl.enqueue_copy(queue,   scos,     scos_buf)
    cl.enqueue_copy(queue,   ssin,     ssin_buf)
    cl.enqueue_copy(queue,   scos2,    scos2_buf)
    cl.enqueue_copy(queue,   ssin2,    ssin2_buf)
    cl.enqueue_copy(queue,   wscos,    wscos_buf)
    cl.enqueue_copy(queue,   wssin,    wssin_buf)
    cl.enqueue_copy(queue,   wscos2,   wscos2_buf)
    cl.enqueue_copy(queue,   wssin2,   wssin2_buf)
    Scos     =  sum(scos)
    Ssin     =  sum(ssin)
    Scos2    =  sum(scos2)
    Ssin2    =  sum(ssin2)
    WScos    =  sum(wscos)
    WSsin    =  sum(wssin)
    Zy       =  Ssin / sqrt(NPIX/2.0)    
    myV      =  Scos / sqrt(NPIX/2.0)
    s_myV    =  np.sqrt((2.0*Ssin2-Zy*Zy)/NPIX)
    meanphi  =  0.5*np.arctan2(Zy, myV)
    
    rvl      =  sqrt( WScos*WScos + WSsin*WSsin ) / sum(weights)  # weighted, only good pixels
    # rvl      =  sqrt(  Scos* Scos +  Ssin* Ssin ) / NPIX
    
    #  can    =  circ.descriptive.mean(2.*phi[good], w=wghts)/2.  -- needed in vtest
    can      =  arctan2(Ssin, Scos)
    t5  = time.time()-t5
    print("--------   second call to second kernel           ---- %.3f seconds" % t5)
    
    # ***circ.tests.rayleigh***
    t6     = time.time()
    # pz, Z  =  circ.tests.rayleigh(2.*phi[good],  w=wghts)
    """
    r    =  descriptive.resultant_vector_length(alpha, w=w, d=d, axis=axis)
    n    =  np.sum(w, axis=axis)
    R    =  n * r
    z    =  R ** 2 / n
    pval =  np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    """
    n    =  sum(weights)   # weight of Nan elements should be zero!
    R    =  n * rvl
    Z    =  R**2.0 / n
    pz   =  np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    # ->   pz, Z
    t6   = time.time()-t6
    print("--------   some python stuff                      ---- %.3f seconds" % t6)

    
        
    # ***circ.tests.vtest***
    t7     = time.time()
    # pv, V  =  circ.tests.vtest(2.*phi[good], 0., w=wghts)
    """
    r = descriptive.resultant_vector_length(alpha, w=w, d=d, axis=axis)
    m = descriptive.mean(alpha, w=w, d=d, axis=axis)
    n = np.sum(w, axis=axis)    
    R = n * r
    V = R * np.cos(m - mu)
    u = V * np.sqrt(2 / n)
    pval = 1 - stats.norm.cdf(u)
    return pval, V
    """                                     
    V  =  R * np.cos(can)
    u  =  V * np.sqrt(2 / n)
    pv = 1 - stats.norm.cdf(u)
    #  -->  pv, V
    print("%.3f s --- pv = %.4e  V = %8.4f" % (time.time()-t0, pv, V))

    t7  = time.time()-t7
    print("--------   postprocessing data from second kernel ---- %.3f seconds" % t7)
    
    print("--------   TOTAL   ----------------------------------- %.3f seconds" % (time.time()-t0))
    print("")
        
    #          0    1  2  3   4   5    6      7        8 
    circstats=[rvl, Z, V, pz, pv, myV, s_myV, meanphi, am]


    if (0):
        print("-"*80)
        print('Mean resultant vector (r)        ', circstats[0])
        print('Rayleigh statistic (Z)           ', circstats[1])
        print('Projected Rayleigh statistic (V) ', circstats[2])
        print('Rayleigh statistic (ii)          ', circstats[5], '+/-', circstats[6])
        print('Mean angle                       ', circstats[7])
        print('Alignment measure (AM)           ', circstats[8])
        print("-"*80)
        sys.exit()
    
    
    return phi, circstats


