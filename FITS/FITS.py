from ISM.Defs import *
import scipy



def CopyFits(F, input_hdu=0, new_data=[]):
    """
    Return a copy of the provided FITS object, with only a single header unit
    Usage:
        G = CopyFits(F, input_hdu=0)
    Input:
        F          = pyfits object
        input_hdu  = number of the header unit to be copied to the return FITS
    Return:
        G = a new FITS object containing a copy of F[input_hdu]
    """
    hdu      = pyfits.PrimaryHDU(np.asarray(F[input_hdu].data.copy(), np.float32))
    hdulist  = pyfits.HDUList([hdu])
    hdulist[0].header = F[input_hdu].header.copy()
    if (len(new_data)>0):
        hdulist[0].data   = new_data.copy()
    else:
        hdulist[0].data   = 1.0*F[input_hdu].data
    hdulist.verify('fix')
    return hdulist



def GetFitsPixelSize(A, hdu=0, radians=True):
    """
    Return the pixel size in given FITS file.
    Usage:
        pixel_size = GetFitsPixelSize(F, hdu=0, radians=True)
    Input:
        A       =   FITS object
        hdu     =   number of the selected header unit
        radians =   if True, return size in radians (the default),
                    otherwise return it in degrees
    Return:
        the pixel size in the image A[hdu]
    Note: 
        Pixels are assumed to be square-
    """
    Kscale = 1.0
    if (radians): Kscale = DEGREE_TO_RADIAN
    if (1):
        # By default take pixel height based on CD1_2 and CD2_2,
        # use CDELT2 only if CDs are not defined.
        if ('CD1_1' in A[hdu].header.keys()):
            if ('CD1_2' in A[hdu].header.keys()):
                return Kscale*np.sqrt(np.float32(A[hdu].header['CD1_1'])**2.0+np.float32(A[hdu].header['CD1_2'])**2.0)
            else:
                return Kscale*abs(np.float32(A[hdu].header['CD1_1']))
        else:
            # print 'CDELT', A[hdu].header['CDELT1']
            return Kscale*abs(np.float32(A[hdu].header['CDELT1']))
    else:
        # use WCS package
        a1, b1 = PIX2WCS(A[hdu].header,   0, 0)
        a2, b2 = PIX2WCS(A[hdu].header, 100, 0)
        a1     = a1*DEGREE_TO_RADIAN
        b1     = b1*DEGREE_TO_RADIAN
        a2     = a2*DEGREE_TO_RADIAN
        b2     = b2*DEGREE_TO_RADIAN
        d      = distance_on_sphere(a1,b1,a2,b2) * RADIAN_TO_DEGREE
        pix    = d / 100.0
        return Kscale*pix

    

def MakeEmptyFitsDim(lon, lat, pix, m, n, sys_req='fk5'):
    """
    Make an empty fits object.
    Inputs:
        lon, lat  = centre coordinates of the field [radians]
        pix       = pixel size [radians]
        m, n      = width and height in pixels
        sys_req   = coordinate system, 'fk5' (default) or 'galactic'
    """
    A         = np.zeros((n, m), np.float32)
    hdu       = pyfits.PrimaryHDU(A)
    F         = pyfits.HDUList([hdu])
    F[0].header.update(CRVAL1 =  lon*RADIAN_TO_DEGREE)
    F[0].header.update(CRVAL2 =  lat*RADIAN_TO_DEGREE)
    F[0].header.update(CDELT1 = -pix*RADIAN_TO_DEGREE)
    F[0].header.update(CDELT2 =  pix*RADIAN_TO_DEGREE)
    F[0].header.update(CRPIX1 =  0.5*(m+1))
    F[0].header.update(CRPIX2 = 0.5*(n+1))
    if ((sys_req=='galactic')):
        F[0].header.update(CTYPE1   = 'GLON-TAN')
        F[0].header.update(CTYPE2   = 'GLAT-TAN')
        F[0].header.update(COORDSYS = 'GALACTIC')
    elif (sys_req=='fk5'):
        F[0].header.update(CTYPE1   = 'RA---TAN')
        F[0].header.update(CTYPE2   = 'DEC--TAN')
        F[0].header.update(COORDSYS = 'EQUATORIAL')
        F[0].header.update(EQUINOX  = 2000.0)
    else:
        print("MakeEmptyFitsDim: unknown coordinate system")
        return None
    return F



# ------------------------------------------------------------------------------------------


    
def InitCL(GPU=0, platforms=[], sub=0, verbose=False):
    """
    Initialise OpenCL environment
    Usage:
        platform, device, context, queue, mf = InitCL(GPU=0, platforms=[], sub=0)
    Input:
        GPU       =  if >0, try to return a GPU device instead of CPU
        platforms =  optional array of possible platform numbers
        sub       =  optional number of threads for a subdevice (first returned)
    Return:
        platform, device, context, queue,  cl.mem_flags
    """
    platform, device, context, queue = None, None, None, None
    possible_platforms = range(6)
    if (len(platforms)>0):
        possible_platforms = platforms
    device = []
    for iplatform in possible_platforms:
        if (verbose): print("try platform %d..." % iplatform)
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



def ReprojectOld(A, B, GPU=0, platforms=[0,1,2,3,4,5], cstep=1, shrink=1.0):
    """
    Reproject pyfits image A onto the pixels in pyfits image B.
    Input:
        A, B   =   source and target pyfits objects
        GPU    =   if >0, use GPU instead of CPU
        platforms = array of potential OpenCL platforms, default is [0, 1, 2, 3, 4, 5]
        cstep  =   calculate coordinates for the input image only at intervals of
                   cstep pixels -- kernel will use linear interpolation for the other pixels
        shrink =   factor by which to skrink the input pixels before overlap calculation
    """    
    # (Xin,Yin) = positions of image A pixels, given in pixel coordinates of the image B system
    N,  M    =  A[0].data.shape
    NN, MM   =  B[0].data.shape
    Mc, Nc   =  (M-1)//cstep+2, (N-1)//cstep+2   # number of points on coordinate grid
    Yin, Xin =  np.indices((Nc, Mc), np.float32)
    tmp      =  wcs.WCS(header=A[0].header)
    ra, de   =  tmp.wcs_pix2world(Xin*cstep, Yin*cstep, 0)
    tmp      =  wcs.WCS(header=B[0].header)
    Xin, Yin =  tmp.wcs_world2pix(ra, de, 0)  # pixel coordinates on target image
    Xin, Yin =  np.asarray(np.ravel(Xin), np.float32), np.asarray(np.ravel(Yin), np.float32)
    # OpenCL initialisations
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=0)
    LOCAL   =  [4, 32][GPU>0]
    GLOBAL  =  (1+(NN*MM)//LOCAL)*LOCAL
    OPT     =  "-D N=%d -D M=%d -D NN=%d -D MM=%d -D LOCAL=%d -D STEP=%d -D Mc=%d -D Nc=%d -D Q=%.4ef" % \
    (N, M, NN, MM, LOCAL, cstep, Mc, Nc, shrink)
    source  =  open(ISM_DIRECTORY+"/ISM/FITS/kernel_resample_image.c").read()
    program =  cl.Program(context, source).build(OPT)
    Sampler =  program.Sampler
    Sampler.set_scalar_arg_dtypes([None, None, None, None])        
    S       =  np.asarray(A[0].data.copy(), np.float32)
    SS      =  np.zeros(NN*MM, np.float32)    
    S_buf   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf  =  cl.Buffer(context, mf.WRITE_ONLY,  4*NN*MM)    
    Xin_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Xin) # Xin[N,M]
    Yin_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Yin) # Yin[N,M]
    queue.finish()
    Sampler(queue, [GLOBAL,], [LOCAL,], Xin_buf, Yin_buf, S_buf, SS_buf)
    cl.enqueue_copy(queue, SS, SS_buf)
    queue.finish()
    B[0].data = SS.reshape(NN,MM)
    B[0].data[np.nonzero(B[0].data==0.0)] = np.NaN # undefined pixels were set to 0.0, now NaN



def ConvolveFitsPyCL(F, fwhm, fwhm_orig=None, hdu=0, dF=None, tag=0,
                     RinFWHM=2.5, masked_value=0.0, GPU=0, platforms=[0,1,2,3,4,5], border='ignore'):
    """
    Convolve a FITS image with a Gaussian beam.
    Usage:
        G = ConvolveFitsGPU(F, fwhm)
        or with all parameters
        G = ConvolveFitsGPU(F, fwhm, fwhm_orig=None, hdu=0, dF=dF, tag=0, RinFWHM=2.5)
    Input:
        F         =  original pyfits object
        fwhm      =  desired final resolution [radians]
        fwhm_orig =  original resolution of the map [radians] (see notes below)
        hdu       =  header unit to use, default is hdu=0
        dF        =  optional error map, 2d array with the same dimensions as F[hdu].data
        tag       =  0=default kernel, 1=beam in constant memory, 2=beam in constant memory, no pixel weighting
                     3=beam in constant memory, no pixel weighting, zero values not ignored
        RinFWHM   =  maximum distance from the beam centre considered on convolution, in units of FWHM
        masked_value = floating point number interpreted as missing values
        GPU       =  if True, use GPU instead of CPU
        platforms =  list of possible OpenCL platforms, default is platforms=[0,1,2,3,4,5]
        border    =  either 'ignore' (default, values outside the map boundary are ignored) or
                     'nearest' (values outside the map replaced by the nearest map pixel), 
                     used only for the default method tag=0 !
    Return:
        G, a pyfits object containing the convolved image
    Note: 
        * apart from tag==3, pixel values equal to masked_value are interpreted as missing values
        * if original resolution fwhm_orig is given, convolve only with a beam with FWHM=sqrt(fwhm^2-fwhm_orig^2),
          if fwhm_orig==None, look for keywords FWHM or BEAM [degrees!!] in the FITS header;
          if no such keyword is found, convolve  with fwhm
    """
    t0 = time.time()
    FWHM = fwhm
    if (fwhm_orig==None): # user gave no original resolution -> check the FITS header
        for key in ['FWHM', 'BEAM']:  # ... for keywords FWHM and BEAM (with values in degrees)
            try:
                fwhm_orig = F[hdu].header[key]*DEGREE_TO_RADIAN
                break
            except:
                print('FITS has no keyword "%s"...' % key)
        if (fwhm_orig!=None):
            print('ConvolveFits uses from header %.3f arcmin as the original fwhm' % \
            (fwhm_orig*RADIAN_TO_ARCMIN))
    if (fwhm_orig!=None):
        if (fwhm_orig>FWHM):
            print('*** WARNING *** ConvolveFits FWHM %12.4e < original FWHM %12.4e => no convolution done!' % \
            (fwhm, fwhm_orig))
        else:
            FWHM = np.sqrt(FWHM**2.0-fwhm_orig**2.0)
    border_options = { 'ignore':0, 'nearest':1}
    border_tag = 0
    if (border in border_options.keys()): border_tag = border_options[border]
    # check that FWHM is valid
    pix =  GetFitsPixelSize(F, hdu)
    G   =  CopyFits(F)
    if (FWHM<0.1*pix):
        print('FWHM (%.1f arcsec) less than 1/10 of pixel size (%.1f arcsec) => skip convolution' % (
                            FWHM*RADIAN_TO_ARCSEC, pix*RADIAN_TO_ARCSEC))
        return G
    if (FWHM>(1000*pix)):
        print('FWHM more than 1000 pixels => refuse to do convolution!')
        return G
    # write data for the OpenCL kernel
    N, M  =  F[hdu].data.shape[0], F[hdu].data.shape[1]    
    X     =  np.asarray(np.ravel(F[hdu].data.copy()), np.float32)
    W     =  None
    if (dF==None):
        W    = np.ones(N*M, np.float32)  # default weights = 1.0
    else:
        W    = np.ravel(dF.copy())
        m    = np.nonzero(w>1e-10)
        W[m] = 1.0/(W[m]*W[m])     # weight ~ 1/var,  and ~ zero for data with dF<=1e-10
    K     =  4.0*np.log(2.0)/((FWHM/pix)**2.0)
    WDIM  =  max(1, int(RinFWHM*FWHM/pix))    # compute up to a distance RinFWHM times the FWHM
    I, J  =  np.indices((2*WDIM+1,2*WDIM+1), np.float32)
    I    -=  WDIM
    J    -=  WDIM
    P     =  np.exp(-K*(I*I+J*J))
    P     =  np.asarray(np.ravel(P), np.float32)   # convolving beam as 2d array
    # OpenCL part...
    platform, device, context, queue, mf = InitCL(GPU, platforms=platforms)
    LOCAL     =  [ 8, 32 ][GPU>0]
    GLOBAL    =  int((np.floor((N*M)/64)+1)*64)
    OPT       =  '-D N=%d -D M=%d -D NPIX=%d -D K=%.5ef -D WDIM=%d -D GLOBAL=%d -D MASKED_VALUE=%.2ef -D BORDER=%d' % \
                  (N, M, N*M, K, WDIM, GLOBAL, masked_value, border_tag)
    source    =  open(ISM_DIRECTORY+"/ISM/FITS/kernel_convolve.c").read()
    program   =  cl.Program(context, source).build(OPT)
    Con       =  None
    # Choose the kernel version
    if (tag==0):
        Con   =  program.Con
        Con.set_scalar_arg_dtypes([None,  None,  None,  None])  # X, W, P, Z
    elif (tag==1):
        Con   =  program.ConC
        Con.set_scalar_arg_dtypes([None,  None,  None,  None])  # X, W, P, Z
    elif (tag==2):
        Con   =  program.ConCU
        Con.set_scalar_arg_dtypes([None,  None,  None])         # X, P, Z
    elif (tag==3):
        Con   =  program.ConCUZ
        Con.set_scalar_arg_dtypes([None,  None,  None])         # X, P, Z
    X_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*N*M)
    W_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*N*M)
    P_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*(2*WDIM+1)*(2*WDIM+1))
    Z_buf     =  cl.Buffer(context, mf.WRITE_ONLY, 4*N*M)    
    cl.enqueue_copy(queue, X_buf, X)
    cl.enqueue_copy(queue, P_buf, P)
    if (tag<2):
        cl.enqueue_copy(queue, W_buf, W)
        Con(queue, [GLOBAL,], [LOCAL,], X_buf, W_buf, P_buf, Z_buf)
    else:  # tag>=2 -> we ignore the weight array
        Con(queue, [GLOBAL,], [LOCAL,], X_buf, P_buf, Z_buf)
    cl.enqueue_copy(queue, X, Z_buf)
    G[0].data           =  X.reshape(G[0].data.shape)
    G[0].header['FWHM'] =  fwhm*RADIAN_TO_DEGREE  # the final FWHM is updated to the header
    return G




def ConvolveFitsBeamPyCL(F, P, hdu=0, dF=None, masked_value=0.0, GPU=0, platforms=[0,1,2,3,4,5], border='ignore'):
    """
    Convolve Fits image with a kernel P provided as 2D array
    Usage:
        G = ConvolveFitsBeamPyCL(F, P)
        or with all parameters
        G = ConvolveFitsBeamPyCL(F, P, hdu=0, dF=dF, masked_value=0.0, GPU=0, platforms=[0,1,2,3,4,5])
    Input:
        F         =  original pyfits object
        P         =  the convolving beam, a square 2d array with odd dimensions
        hdu       =  header unit from which input image is read, default is main one hdu=0
        dF        =  optional error map, 2d array with the same dimensions as F[hdu].data
        masked_value = floating point number standing for missing values, default is 0.0
        GPU       =  if True, use GPU instead of CPU
        platforms =  list of possible OpenCL platforms, default is [0, 1, 2, 3, 4, 5]
        border    =  default is 'ignore' = pixels outside the map are ignored, other option
                     is 'nearest', replacing data outside the map with the nearest map pixel
    Return: 
        G, a new FITS object containing the convolved image
    """
    t0    =  time.time()
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
    border_options = { 'ignore':0, 'nearest':1}
    border_tag = 0
    if (border in border_options.keys()): border_tag = border_options[border]
    WDIM  =  int((P.shape[0]-1)/2)
    if (((2*WDIM+1)!=P.shape[0])|((2*WDIM+1)!=P.shape[1])):
        print('Error in ConvolveFitsBeamPyCL: check the provided beam')
        print('P %d x %d, must be (2*n+1) times (2*n+1), with n an integer' % (P.shape[0], P.shape[1]))
        sys.exit()
    P         =  np.asarray(np.ravel(P), np.float32)   # beam pattern
    # OpenCL part...
    platform, device, context, queue, mf = InitCL(GPU, platforms=platforms, verbose=True)
    LOCAL     =  [ 8, 64 ][GPU>0]
    GLOBAL    =  int((np.floor((N*M)/64)+1)*64)
    OPT       =  '-D N=%d -D M=%d -D NPIX=%d -D K=0.0f -D WDIM=%d -D GLOBAL=%d -D MASKED_VALUE=%.3ef -D BORDER=%d' % \
                  (N, M, N*M, WDIM, GLOBAL, masked_value, border_tag)
    source    =  open(ISM_DIRECTORY+"/ISM/FITS/kernel_convolve.c").read()
    program   =  cl.Program(context, source).build(OPT)
    Con       =  None
    Con       =  program.Con
    Con.set_scalar_arg_dtypes([None,  None,  None,  None])  # X, W, P, Z
    X_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*N*M)
    W_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*N*M)
    P_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*(2*WDIM+1)*(2*WDIM+1))
    Z_buf     =  cl.Buffer(context, mf.WRITE_ONLY, 4*N*M)    
    cl.enqueue_copy(queue, X_buf, X)
    cl.enqueue_copy(queue, P_buf, P)
    cl.enqueue_copy(queue, W_buf, W)
    Con(queue, [GLOBAL,], [LOCAL,], X_buf, W_buf, P_buf, Z_buf)
    cl.enqueue_copy(queue, X, Z_buf)
    # Put the result to pyfits object G
    G          =  CopyFits(F)
    G[0].data  =  X.reshape(G[0].data.shape)
    print('ConvolveFitsBeamPyCL: %.3f seconds' % (time.time()-t0))
    return G




def ConvolveMapBeamPyCL(F, P, dF=None, masked_value=0.0, GPU=0, platforms=[0,1,2,3,4,5]):
    """
    Convolve 2D array with a kernel provided as a 2D array
    Usage:
        F = ConvolveMapBeamPyCL(F, P, dF=dF, masked_value=0.0, GPU=0, platforms=[0,1,2,3,4,5])
    Input:
        F         =  original 2D image (numpy array)
        P         =  2D array, the beam give at the same pixelisation
        dF        =  optional error map, 2d array
        masked_value = floating point number standing for missing values (default = 0.0)
        GPU       =  if True, use GPU instead of CPU
        platforms =  array of possible OpenCL platforms, default is [0,1,2,3,4,5]
    Note: 
        The original image F is not changed. The pixel size of the image and the
        beam must be the same. The beam P must be a square array with odd dimensions.
    """
    t0    =  time.time()
    N     =  F.shape[0]
    M     =  F.shape[1]
    X     =  np.asarray(np.ravel(F.copy()), np.float32)
    W     =  None
    if (dF==None):
        W    = np.ones(N*M, np.float32)
    else:
        W    = np.ravel(dF.copy())
        m    = np.nonzero(w>1e-10)
        W[m] = 1.0/(W[m]*W[m])     # weight, not std !!! -> GPUconvolution
    WDIM  =  int((P.shape[0]-1)/2)
    if (((2*WDIM+1)!=P.shape[0])|((2*WDIM+1)!=P.shape[1])):
        print('Error in ConvolveMapBeamPyCL: check the provided beam')
        print('P %d x %d, must be (2*n+1) times (2*n+1), with n an integer' % (P.shape[0], P.shape[1]))
        sys.exit()
    P         =  np.asarray(np.ravel(P), np.float32)   # beam pattern
    platform, device, context, queue, mf = InitCL(GPU, platforms=platforms, verbose=False)
    LOCAL     =  [ 8, 64 ][GPU>0]
    GLOBAL    =  int((np.floor((N*M)/64)+1)*64)
    OPT       =  '-D N=%d -D M=%d -D NPIX=%d -D K=0.0f -D WDIM=%d -D GLOBAL=%d -D MASKED_VALUE=%.4ef' % \
                  (N, M, N*M, WDIM, GLOBAL, masked_value)
    source    =  open(ISM_DIRECTORY+"/ISM/FITS/kernel_convolve.c").read()
    program   =  cl.Program(context, source).build(OPT)
    Con       =  None
    Con       =  program.Con
    Con.set_scalar_arg_dtypes([None,  None,  None,  None])  # X, W, P, Z
    X_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*N*M)
    W_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*N*M)
    P_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*(2*WDIM+1)*(2*WDIM+1))
    Z_buf     =  cl.Buffer(context, mf.READ_ONLY, 4*N*M)
    t1 = time.time()
    cl.enqueue_copy(queue, X_buf, X)
    cl.enqueue_copy(queue, P_buf, P)
    cl.enqueue_copy(queue, W_buf, W)
    Con(queue, [GLOBAL,], [LOCAL,], X_buf, W_buf, P_buf, Z_buf)
    cl.enqueue_copy(queue, X, Z_buf)
    X.shape = F.shape
    print('ConvolveMapBeamPyCL: %.3f seconds,  kernel %.3f seconds' % (time.time()-t0, time.time()-t1))
    return X



def Reproject(A, B, GPU=0, platforms=[0,1,2,3,4,5], cstep=5, threads=1, shrink=1.0, hdu_A=0, hdu_B=0):
    """
    Reproject pyfits image A onto the pixels in pyfits image B.
    Usage:
        Reproject(A, B)
        or with all parameters
        Reproject(A, B, GPU=0, platforms=[0,1,2,3,4,5], cstep=5, threads=1, shrink=1.0):
    Input:
        A, B    =   source and target pyfits objects
        GPU     =   if >0, use GPU instead of CPU
        platforms = array of potential OpenCL platforms, default is [0, 1, 2, 3, 4, 5]
        cstep   =   calculate coordinates for the input image explicitly only at intervals of 
                    cstep [integer] pixels, kernel using linear interpolation between these points
        threads =   if >1, calculate coordinate transformations with multiprocessing, this number of threads
        shrink  =   factor by which the input image pixels are shrunk before calculating pixel overlap,
                    default value is 1.0, possible values between zero and one
        hdu_A, hdu_B = header units to be used with pyfits objects A and B, respectively
    Return:
        reprojected image is copied to the pyfits object B
    """    
    DO_TIMINGS =  1
    if (DO_TIMINGS):
        tP, tB, tK = 0.0, 0.0, 0.0
        t0 = time.time()
    # (Xin,Yin) = positions of image A pixels, given in pixel coordinates of the image B system
    N,  M    =  A[hdu_A].data.shape
    NN, MM   =  B[hdu_B].data.shape
    Mc, Nc   =  (M-1)//cstep+2, (N-1)//cstep+2   # number of points on coordinate grid
    Yin, Xin =  np.indices((Nc, Mc), np.float32)
    tmp1     =  wcs.WCS(header=A[hdu_A].header)
    tmp2     =  wcs.WCS(header=B[hdu_B].header)
    if (threads<2):
        ra, de   =  tmp1.wcs_pix2world(Xin*cstep, Yin*cstep, 0)
        tmp      =  wcs.WCS(header=B[hdu_B].header)
        Xin, Yin =  tmp2.wcs_world2pix(ra, de, 0)  # pixel coordinates on target image
        Xin, Yin =  np.asarray(np.ravel(Xin), np.float32), np.asarray(np.ravel(Yin), np.float32)
    else:
        npix     =  Nc*Mc
        manager  =  mp.Manager()
        yin      =  mp.Array('f', npix)
        xin      =  mp.Array('f', npix)
        no       =  npix//threads
        ###
        def fun(x, y, tmp1, tmp2, a, b):
            ra, de          =  tmp1.wcs_pix2world(x[a:b], y[a:b], 0)
            x[a:b], y[a:b]  =  tmp2.wcs_world2pix(ra, de, 0)
        ###
        PROC = []
        xin[:]   =  cstep*np.ravel(Xin)
        yin[:]   =  cstep*np.ravel(Yin)
        for i in range(threads):
            a, b =  i*no, (i+1)*no+1
            if (i==(threads-1)): b = npix
            p = mp.Process(target=fun, args=(xin, yin, tmp1, tmp2, a, b))
            PROC.append(p)
            p.start()
        for i in range(threads):
            PROC[i].join()    
        Xin, Yin =  np.asarray(np.ravel(xin), np.float32), np.asarray(np.ravel(yin), np.float32)
    if (DO_TIMINGS):        
        tP = time.time()-t0
        t0 = time.time()
    # OpenCL initialisations
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=0)
    LOCAL   =  [4, 32][GPU>0]
    GLOBAL  =  (1+(NN*MM)//LOCAL)*LOCAL
    OPT     =  "-D N=%d -D M=%d -D NN=%d -D MM=%d -D LOCAL=%d -D STEP=%d -D Mc=%d -D Nc=%d -D Q=%.4ef" % \
    (N, M, NN, MM, LOCAL, cstep, Mc, Nc, shrink)
    source  =  open(ISM_DIRECTORY+"/ISM/FITS/kernel_resample_image.c").read()
    program =  cl.Program(context, source).build(OPT)
    queue.finish()
    if (DO_TIMINGS):
        tB = time.time()-t0
        t0 = time.time()
    Sampler =  program.Sampler
    Sampler.set_scalar_arg_dtypes([None, None, None, None])        
    S       =  np.asarray(A[hdu_A].data.copy(), np.float32)
    SS      =  np.zeros(NN*MM, np.float32)    
    S_buf   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf  =  cl.Buffer(context, mf.WRITE_ONLY,  4*NN*MM)    
    Xin_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Xin) # Xin[N,M]
    Yin_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Yin) # Yin[N,M]
    queue.finish()
    Sampler(queue, [GLOBAL,], [LOCAL,], Xin_buf, Yin_buf, S_buf, SS_buf)
    cl.enqueue_copy(queue, SS, SS_buf)
    queue.finish()
    B[hdu_B].data = SS.reshape(NN,MM)
    B[hdu_B].data[np.nonzero(B[hdu_B].data==0.0)] = np.NaN # undefined pixels were set to 0.0, now NaN
    if (DO_TIMINGS):
        tK = time.time()-t0
        return tP, tB, tK  # python, build, and kernel timings

    
    
def convolve_map(A, fwhm, mode='reflect', radius=3.6):
    # For comparison convolution with a Gaussian kernel using scipy.ndimage.convolve
    n  = int(2+fwhm*radius)
    n  = min(n, A.shape[0]-1)
    if (n%2==0):
        n += 1   # make sure it is odd (?)
    kernel = np.zeros((n,n), np.float32)
    K      = 4.0*np.log(2.0)/(fwhm*fwhm) 
    D2     = max(1.7, (1.5*fwhm))**2.0   # outer radius in pixels
    for i in range(n):                   # loop over kernel pixels
        dx  = (n-1.0)/2.0 - i
        for j in range(n):
            dy  = (n-1.0)/2.0 - j
            d2  = dx*dx+dy*dy
            if (d2<D2):
                kernel[i,j] =  np.exp(-K*d2)
    # normalize: integral over kernel equals one
    kernel /= np.sum(kernel.flat)
    tmp     = np.asarray(scipy.ndimage.convolve(np.asarray(A,np.float32), kernel, mode=mode), float32)
    return tmp



def convolve_map_fast(A, fwhm, radius=5.0):
    """
    Image convolution with FFT.
    Usage:
        B = convolve_map_fast(A, fwhm, radius=5.0)
    Input:
        A      = image as 2d array
        fwhm   = FWHM of the convolving beam, in units of pixels
        radius = maximum distance from beam centre [pixels] used to construct the beam
    Return:
        convolved image as 2d array
    """
    N, M   = A.shape
    n      = int(1+radius*fwhm)  # size of the beam area
    n      = min(N-1, n)
    if (n%2==0):
        n += 1   # make sure it is odd (?)
    kernel = zeros((N,M), float32)
    K      = 4.0*log(2.0)/(fwhm*fwhm) 
    dx, dy = indices((N,M), float32)
    dx    -= 0.5*(N-1.0)+0.5
    dy    -= 0.5*(M-1.0)+0.5
    if (N%2==1): dx -= 0.5
    if (M%2==1): dy -= 0.5
    d2     = dx*dx+dy*dy        
    mask   = nonzero(d2<(n*n))
    kernel[mask] = exp(-K*d2[mask])        
    # normalize the integral over the kernel to one
    kernel /= sum(kernel.flat)
    kernel  = fftshift(kernel.copy())
    tmp     = ifft2(fft2(kernel)*fft2(A))
    # keep the sign of real part, use amplitude of the complex number
    tmp     = sign(real(tmp))*abs(tmp)
    return tmp



def MADFilterCL(X, R1, R2, circle=0, GPU=0, platforms=[0,1,2,3,4,5]):
    """
    Median absolute deviation filter on a full  2D image.
    Input:
        X      = 2D numpy array
        R1, R2 = inner and outer radius of the annulus [int32], if circle=True,
                 otherwise only R2 is used and the footprint is 1+2*R2 pixels squared
        circle = if >0, footprint is circular instead of a full square
        GPU    = if >0, use GPU instead of CPU
        platforms = optional list of possible OpenCL platforms
    Return:
        image of MAD-filtered values
    """
    platform, device, context, queue, mf = InitCL(GPU, platforms=platforms)
    LOCAL    =  [ 8, 64 ][GPU>0]
    N, M     =  X.shape
    NFOOT    =  int((1+2*R2)**2.0)    # maximum number of pixels under the footprint
    OPT      =  " -D N=%d -D M=%d -D R1=%d -D R2=%d -D CIRCLE=%d -D NFOOT=%d -D LOCAL=%d" % \
    (N, M, R1, R2, circle, NFOOT, LOCAL)
    # print(OPT)
    source   =  open(ISM_DIRECTORY+'/ISM/FITS/kernel_Median.c').read()
    program  =  cl.Program(context, source).build(OPT)    
    t0       =  time.time()
    S        =  asarray(X, float32)
    SS       =  np.empty_like(S)
    S_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf   =  cl.Buffer(context, mf.READ_WRITE, S.nbytes)
    ###
    D        =  asarray([701, 301, 132, 57, 23, 10, 4, 1], int32)
    ###
    t0       =  time.time()
    GLOBAL   = (int((N*M)/64)+1)*64    
    D        =  asarray(D[nonzero(D<100)], int32)  # truncate the gap sequence to values < (1+2*D)**2
    D_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=D)
    MAD      =  program.MAD
    MAD.set_scalar_arg_dtypes([None, None, None])
    MAD(queue, [GLOBAL,], [LOCAL,], S_buf, D_buf, SS_buf)        
    cl.enqueue_copy(queue, SS, SS_buf)
    t2       =  time.time()-t0
    return SS


