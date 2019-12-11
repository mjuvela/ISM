import os, sys, time
import numpy as np
from   astropy import wcs
from   astropy.io import fits as pyfits
import pyopencl as cl


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


def Reproject(A, B, GPU=0, platforms=[0,1,2,3,4,5], cstep=1):
    """
    Reproject pyfits image A onto the pixels in pyfits image B.
    Input:
        A, B   =   source and target pyfits objects
        GPU    =   if >0, use GPU instead of CPU
        platforms = array of potential OpenCL platforms, default is [0, 1, 2, 3, 4, 5]
        cstep  =   calculate coordinates for the input image only at intervals of 
                   cstep pixels -- kernel will use linear interpolation for the other pixels
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
    OPT     =  "-D N=%d -D M=%d -D NN=%d -D MM=%d -D LOCAL=%d -D STEP=%d -D Mc=%d -D Nc=%d" % (N, M, NN, MM, LOCAL, cstep, Mc, Nc)
    source  =  open(os.path.dirname(os.path.realpath(__file__))+"/kernel_resample_image.c").read()
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


if (len(sys.argv)<4):
    print("\nUsage: ResampleImage A.fits B.fits C.fits" )
    print(" Resample A.fits onto the pixels defined by B.fits, write result to C.fits\n")
    print("*** THIS IS BETA SOFTWARE, NO GUARANTEES OF CORRECT RESULTS ***") 
    sys.exit()
A = pyfits.open(sys.argv[1])
B = pyfits.open(sys.argv[2])
Reproject(A, B)
B.writeto(sys.argv[3], overwrite=True)


