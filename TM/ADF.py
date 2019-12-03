from   MJ.mjDefs import *
from   MJ.WCS import get_fits_pixel_size
import time
import numpy as np
import pyopencl as cl
from   MJ.Aux.FITS import CopyFits
from   MJ.WCS import WCS2PIX, PIX2WCS_ALL, PIX2WCS


def AbsolutePA(F):
    """
    Input:
        F   = pyfits image containing position angles (east of north)
              determined in pixel space
    Output:
        A new pyfits image with corrected position angles = PA with respect
        to the true north, taking into account the used projection.
    Note:
        The procedure:
            - for a grid of positions, calculate north vector in pixels
              => difference between true north and pixel north
            - interpolate the correction to all pixels and return
              corrected map
    """
    N, M      =  F[0].data.shape
    LON, LAT  =  PIX2WCS_ALL(F[0].header, radians=True)
    # initial points in pixel coordinates
    x0, y0    =  WCS2PIX(F[0].header, ravel(LON), ravel(LAT), radians=True)
    # points towards north
    x1, y1    =  WCS2PIX(F[0].header, ravel(LON), ravel(LAT)+ARCMIN_TO_RADIAN, radians=True)
    print(x0[0:10])
    print(y0[0:10])
    print(x1[0:10])
    print(y1[0:10])
    # position angle in pixel coordinates
    dPA       =  arctan2(y1-y0, x1-x0) - 0.5*pi
    print(dPA[0:10])
    # if true north is to left, dPA>0;  subtract dPA from original PA values
    G         =  CopyFits(F)
    G[0].data =  F[0].data - dPA.reshape(F[0].data.shape)
    return G



def AngleDispersionFunction(F, R1, R2, GPU=0, FAST_DISTANCE=0):
    """
    Calculate Angle Dispersion Function (ADF).
    Input:
        F        = pyfits image with position angles in radians
                   values < -99 are considered missing values
        R1, R2   = angular separation (scalar)  [radians], inner and outer radius
        GPU      = if >0, use GPU instead of CPU
        FAST_DISTANCE
                 = if >0, use approximate distances (ok for small fields, away from poles)
    Return:
        A pyfits object where the image contains the ADF map
    Note:
        Takes into account the true distance between pixels.
        Should take into account the difference between the pixel north and
        the true north. We assume that the angles in F are determined
        in pixel coordinates, i.e., in the projected system. In ADF,
        one should calculate angle difference by first correcting for
        the projection effect (especially near the poles).
        *** THE CORRECTION OF ANGLE DIFFERENCES IS NOT IMPLEMENTED ***
        ***     use Absolute_PA() routine                          ***
        In tangential projection the error in angles is less than 1 degree
        for fields of the size of one degree...
        in v31_0097 and equatorial coordinates it is max ~three degrees
    """
    t0          = time.time()
    lon, lat    = PIX2WCS_ALL(F[0].header, radians=True)
    lon, lat    = asarray(lon, float32), asarray(lat, float32)
    platform    = cl.get_platforms()[0]
    if (GPU>0): device, LOCAL = platform.get_devices(cl.device_type.GPU), 64
    else:       device, LOCAL = platform.get_devices(cl.device_type.CPU),  8
    N, M        = F[0].data.shape
    NPIX        = N*M
    context     = cl.Context(device)
    queue       = cl.CommandQueue(context)
    mf          = cl.mem_flags
    OPT         = "-D NPIX=%d -D R1=%.5e -D R2=%.5e -D FAST_DISTANCE=%d" % (NPIX, R1, R2, FAST_DISTANCE)
    source      = file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_ADF.c').read()
    program     = cl.Program(context, source).build(OPT)
    print('--- initialisation  %5.2f seconds' % (time.time()-t0))
    #
    P           = asarray(ravel(F[0].data), float32)
    S           = np.empty_like(P)
    X_buf       = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lon)
    Y_buf       = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lat)
    P_buf       = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P)
    S_buf       = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    ADF         = program.ADF
    ADF.set_scalar_arg_dtypes([None, None, None, None])
    t0          = time.time()
    GLOBAL      = (int(NPIX/64)+1)*64
    ADF(queue, [GLOBAL,], [LOCAL,], X_buf, Y_buf, P_buf, S_buf)
    cl.enqueue_copy(queue, S, S_buf)
    FS          = CopyFits(F)
    FS[0].data  = S.reshape(FS[0].data.shape)
    return FS
