import os, sys
import pyopencl as cl
from astropy.io import fits as pyfits
import numpy as np
import matplotlib.pylab as pl
from scipy.ndimage import label
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))

DEGREE_TO_RADIAN =  0.0174532925199432958
ARCMIN_TO_RADIAN =  (2.9088820e-4)
ARCSEC_TO_RADIAN =  (4.8481368e-6)
RADIAN_TO_DEGREE =  57.2957795130823208768
RADIAN_TO_ARCMIN =  3437.746771
RADIAN_TO_ARCSEC =  206264.8063


def GetFitsPixelSize(A, hdu=0, radians=True):
    """
    Return the pixel size [degrees] in given FITS file.
        A       =   FITS object
        hdu     =   number of the selected header unit
        radians =   if True, return size in radians; default is degrees
    Note: 
        Routine assumes that the pixels are square!
    """
    Kscale = 1.0
    if (radians): Kscale = DEGREE_TO_RADIAN
    # By default take pixel height based on CD1_2 and CD2_2, use CDELT2 only if CDs are not defined.
    if ('CD1_1' in A[hdu].header.keys()):
        if ('CD1_2' in A[hdu].header.keys()):
            return Kscale*np.sqrt(float(A[hdu].header['CD1_1'])**2.0+float(A[hdu].header['CD1_2'])**2.0)
        else:
            return Kscale*abs(float(A[hdu].header['CD1_1']))
    else:
        return Kscale*abs(float(A[hdu].header['CDELT1']))

       
def CopyFits(F, input_hdu=0):
    hdu      = pyfits.PrimaryHDU(np.asarray(F[input_hdu].data.copy(), np.float32))
    hdulist  = pyfits.HDUList([hdu])
    hdulist[0].header = F[input_hdu].header.copy()
    hdulist[0].data   = 1.0*F[input_hdu].data
    return hdulist


def MakeEmptyFits(lon, lat, radius, pix, sys_req):
    """
    Make an empty fits object.
        lon, lat  = centre coordinates of the field [radians]
        radius    = map radius [radians]
        pix       = pixel size [radians]
        sys_req   = coordinate system, WCS_GALACTIC or WCS_J2000
    """
    npix      = int(2.0*radius/pix)+1
    A         = np.zeros((npix, npix), np.float32)
    hdu       = pyfits.PrimaryHDU(A)
    F         = pyfits.HDUList([hdu])
    F[0].header.update(CRVAL1 =  lon*RADIAN_TO_DEGREE)
    F[0].header.update(CRVAL2 =  lat*RADIAN_TO_DEGREE)
    F[0].header.update(CDELT1 = -pix*RADIAN_TO_DEGREE)
    F[0].header.update(CDELT2 =  pix*RADIAN_TO_DEGREE)
    F[0].header.update(CRPIX1 =  0.5*(npix+1))
    F[0].header.update(CRPIX2 =  0.5*(npix+1))
    if (sys_req=='galactic'):
        F[0].header.update(CTYPE1   = 'GLON-TAN')
        F[0].header.update(CTYPE2   = 'GLAT-TAN')
        F[0].header.update(COORDSYS = 'GALACTIC')
    else:
        F[0].header.update(CTYPE1   = 'RA---TAN')
        F[0].header.update(CTYPE2   = 'DEC--TAN')
        F[0].header.update(COORDSYS = 'EQUATORIAL')
        F[0].header.update(EQUINOX  = 2000.0)
    return F


def InitCL(GPU=0, platforms=[], sub=0, verbose=False):
    """
    Usage:
        platform, device, context, queue, mf = InitCL(GPU=0, platforms=[], sub=0)
    Input:
        GPU       =  if >0, try to return a GPU device instead of CPU
        platforms =  optional array of possible platform numbers
        sub       =  optional number of threads for a subdevice (first returned)
    """
    platform, device, context, queue = None, None, None, None
    possible_platforms = range(6)
    if (len(platforms)>0):
        possible_platforms = platforms
    device = []
    for iplatform in possible_platforms:
        if (verbose): print("try platform %d... for GPU=%d" % (iplatform, GPU))
        try:
            platform     = cl.get_platforms()[iplatform]
            if (GPU>0):
                device   = platform.get_devices(cl.device_type.GPU)
            else:
                device   = platform.get_devices(cl.device_type.CPU)
            if (sub>0):
                # try to make subdevices with sub threads, return the first one
                dpp       =  cl.device_partition_property
                device    =  [device[0].create_sub_devices( [dpp.EQUALLY, sub] )[0],]
            context   =  cl.Context(device)
            queue     =  cl.CommandQueue(context)
            break
        except:
            pass
    if (verbose):
        print(device)
    return platform, device, context, queue,  cl.mem_flags



def ConvolveFitsPyCL(F, fwhm, fwhm_orig=None, hdu=0, dF=None, TAG=0, GPU=False, platforms=[], 
                     RinFWHM=2.5, masked_value=0.0):
    """
    Usage:
        G = ConvolveFitsGPU(F, fwhm, fwhm_orig=None, hdu=0, dF=dF, RinFWHM=2.5)
    Input:
        F         =  original pyfits object
        fwhm      =  desired final resolution [radians]
        fwhm_orig =  original resolution of the map [radians]
        hdu       =  which image to use (default main one, hdu=0)
        dF        =  optional error map, 2d array
        TAG       =  0=default kernel, 1=beam in constant memory, 2=beam in constant memory, no pixel weighting
                     3=beam in constant memory, no pixel weighting, zero values not ignored
        GPU       =  if True, use GPU instead of CPU
        platforms =  if not [], list of possible OpenCL devices
        RinFWHM   =  maximum distance from beam centre considered on convolution
        masked_value = floating point number considered as missing value
    NOTE: 
        Apart from TAG==3, pixel values 0.0 are interpreted as missing values
        Convolve given fits F to resolution fwhm [radians]. If original resolution fwhm_orig 
        is given, convolve only with sqrt(fwhm^2-fwhm_orig^2). If fwhm_orig==None, look for 
        keywords FWHM or BEAM in the header [degrees!!].
    """
    FWHM = fwhm
    if (fwhm_orig==None):
        for key in ['FWHM', 'BEAM']:
            try:
                fwhm_orig = F[hdu].header[key]*DEGREE_TO_RADIAN
                break
            except:
                print('FITS has no keyword "%s"...' % key)
        if (fwhm_orig!=None):
            print('ConvolveFits uses from header %.3f arcmin as original fwhm !!' % \
            (fwhm_orig*RADIAN_TO_ARCMIN))
    if (fwhm_orig!=None):
        if (fwhm_orig>FWHM):
            print('*** WARNING *** ConvolveFits FWHM %12.4e < FWHM_ORIG %12.4e => NO CONVOLUTION !' % (fwhm, fwhm_orig))
            FWHM = 0.0
        else:
            FWHM = np.sqrt(FWHM**2.0-fwhm_orig**2.0)                
    # Check that FWHM is valid
    pix = GetFitsPixelSize(F, hdu, radians=True)
    if (FWHM<0.1*pix):
        print('FWHM (%.1f arcsec) less than 1/10 of pixel size (%.1f arcsec) => skip convolution' % (
                            FWHM*RADIAN_TO_ARCSEC, pix*RADIAN_TO_ARCSEC))
        return CopyFits(F)
    if (FWHM>(1000*pix)):
        print('FWHM more than 1000 pixels => REFUSE TO DO CONVOLUTION')
        return CopyFits(F)
    N     =  F[hdu].data.shape[0]
    M     =  F[hdu].data.shape[1]
    X     =  np.asarray(np.ravel(F[hdu].data.copy()), np.float32)
    W     =  None
    if (dF==None):
        W    = np.ones(N*M, np.float32)
    else:
        W    = np.ravel(dF.copy())
        m    = np.nonzero(w>1e-10)
        W[m] = 1.0/(W[m]*W[m])     # weight, not std !!! -> GPUconvolution
    K     =  4.0*np.log(2.0)/((FWHM/pix)**2.0)
    WDIM  =  max(1, int(RinFWHM*FWHM/pix))    # up to distances RinFWHM times FWHM
    I, J  =  np.indices((2*WDIM+1,2*WDIM+1), np.float32)
    I    -=  WDIM
    J    -=  WDIM
    P     =  np.exp(-K*(I*I+J*J))
    P     =  np.asarray(np.ravel(P), np.float32)   # beam pattern
    platform, device, context, queue, mf = InitCL(GPU, platforms=platforms)
    LOCAL     =  [ 8, 32 ][GPU>0]
    GLOBAL    =  int((np.floor((N*M)/64)+1)*64)
    OPT       =  '-D N=%d -D M=%d -D NPIX=%d -D K=%.5ef -D WDIM=%d -D GLOBAL=%d -D MASKED_VALUE=%.4ef' % \
                  (N, M, N*M, K, WDIM, GLOBAL, masked_value)
    source    =  open(INSTALL_DIR+"/kernel_convolve.c").read()
    program   =  cl.Program(context, source).build(OPT)
    Con       =  None
    if (TAG==0):
        Con   =  program.Con
        Con.set_scalar_arg_dtypes([None,  None,  None,  None])  # X, W, P, Z
    elif (TAG==1):
        Con   =  program.ConC
        Con.set_scalar_arg_dtypes([None,  None,  None,  None])  # X, W, P, Z
    elif (TAG==2):
        Con   =  program.ConCU
        Con.set_scalar_arg_dtypes([None,  None,  None])         # X, P, Z
    elif (TAG==3):
        Con   =  program.ConCUZ
        Con.set_scalar_arg_dtypes([None,  None,  None])         # X, P, Z
    X_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*N*M)
    W_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*N*M)
    P_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*(2*WDIM+1)*(2*WDIM+1))
    Z_buf     =  cl.Buffer(context, mf.WRITE_ONLY, 4*N*M)    
    cl.enqueue_copy(queue, X_buf, X)
    cl.enqueue_copy(queue, P_buf, P)
    if (TAG<2):
        cl.enqueue_copy(queue, W_buf, W)
        Con(queue, [GLOBAL,], [LOCAL,], X_buf, W_buf, P_buf, Z_buf)
    else:
        Con(queue, [GLOBAL,], [LOCAL,], X_buf, P_buf, Z_buf)
    cl.enqueue_copy(queue, X, Z_buf)
    # Copy result to new FITS image
    G                   =  CopyFits(F)
    G[0].data           =  X.reshape(G[0].data.shape)
    G[0].header['FWHM'] =  fwhm*RADIAN_TO_DEGREE
    G[0].header['BMAJ'] =  fwhm*RADIAN_TO_DEGREE
    G[0].header['BMIN'] =  fwhm*RADIAN_TO_DEGREE
    G[0].header['PA']   =  0.0
    return G

