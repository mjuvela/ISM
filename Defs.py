import os, sys, time
import numpy as np
from   astropy import wcs
from   astropy.io import fits as pyfits
import pyopencl as cl
import multiprocessing as mp
from   matplotlib.pylab import *


# this is required in order to locate the kernel codes *.c
ISM_DIRECTORY = os.path.expanduser('~/GITHUB')
try:
    ISM_DIRECTORY = os.environ(['ISM_DIRECTORY'])
except:
    pass


# the following in cgs system !!
C_LIGHT          = 2.99792458E10
C_LIGHT_SI       = 2.99792458E8
AMU              = 1.6605E-24 
H_K              = 4.799243348e-11 ## 4.7995074E-11 
BOLTZMANN        = 1.3806488e-16
BOLTZMANN_SI     = 1.3806488e-23
STEFAN_BOLTZMANN = 5.670373e-5
SB_SI            = 5.670373e-8
CGS_TO_JY_SR     = 1e23          # erg/cm2/sr/Hz = CGS_TO_JY_SR * Jy/sr
PLANCK           = 6.62606957e-27 
PLANCK_SI        = 6.62606957e-34
M0               = 1.99e33
MJupiter         = 1.9e30        # [g]
GRAV             = 6.67e-8
GRAV_SI          = 6.673e-11
PARSEC           = 3.0857E18
LIGHTYEAR        = 9.4607e17
LIGHTYEAR_SI     = 9.4607e15
ELECTRONVOLT     = 1.6022e-12
AU               = 149.597871e11
RSUN             = 6.955e10
RSUN_SI          = 6.955e8
DSUN             = 1.496e13  # cm
DSUN_SI          = 1.496e11  # 1.496e8 km
MSUN             = 1.9891e33
MSUN_SI          = 1.9891e30
M_EARTH          = 5.972e27
LSUN             = 3.839e33
LSUN_SI          = 3.839e26
TSUN             = 5778.0
MJUPITER         = 1.9e30
H_C2             = PLANCK/(C_LIGHT*C_LIGHT)
H_C2_GHz         = PLANCK/(C_LIGHT*C_LIGHT)*1.0e27


ARCSEC_TO_DEGREE =  (1.0/3600.0)
DEGREE_TO_RADIAN =  0.0174532925199432958
ARCMIN_TO_RADIAN =  (2.9088820e-4)
ARCSEC_TO_RADIAN =  (4.8481368e-6)
HOUR_TO_RADIAN   =  (0.261799387)
MINUTE_TO_RADIAN =  (4.3633231e-3)
SECOND_TO_RADIAN =  (7.2722052e-5)

RADIAN_TO_DEGREE =  57.2957795130823208768
RADIAN_TO_ARCMIN =  3437.746771
RADIAN_TO_ARCSEC =  206264.8063
RADIAN_TO_HOUR   =  3.819718634
RADIAN_TO_MINUTE =  229.1831181
RADIAN_TO_SECOND =  13750.98708

ARCMIN_TO_DEGREE =   (1.0/60.0)
DEGREE_TO_ARCMIN =   60.0
DEGREE_TO_ARCSEC =   3600.0


def HMS2RAD(h, m, s):
    # Convert hour, minute, and second to radians
    if (h>=0): 
        return  (h+(1.0*m)/60.0+(1.0*s)/3600.0)*HOUR_TO_RADIAN
    else:    # someone uses negative hours ?
        return  -(-(1.0*h)+(1.0*m)/60.0+(1.0*s)/3600.0)*HOUR_TO_RADIAN

    
def DMS2RAD(d, m, s):
    # Convert degree, arcmin, arcsec to radians
    # If  -1<DEC<0.0, d must be float so that we have it as -0.0
    # ok as long as -0.0 has STRING REPRESENTATION WITH MINUS SIGN INCLUDED
    if (d==0.0):
        sss = '%.1f' % d
        if (sss.find('-')>=0):  # d was -0.0 !!
            return -(-(1.0*d)+(1.0*m)/60.0+(1.0*s)/3600.0)*DEGREE_TO_RADIAN            
    if (d>=0):
        return ((1.0*d)+(1.0*m)/60.0+(1.0*s)/3600.0)*DEGREE_TO_RADIAN
    else:
        return -(-(1.0*d)+(1.0*m)/60.0+(1.0*s)/3600.0)*DEGREE_TO_RADIAN

    
def RAD2HMS(x):
    xx  =  abs(x*RADIAN_TO_SECOND)
    h   =  int(xx/3600)
    xx -=  h*3600.0 
    h   =  h % 24
    m   =  int(xx/60) 
    s   =  xx-m*60.0 
    if (abs(s-60.0)<0.1):
        s  = 0.0 
        m += 1 
        if (m==60):
            m  = 0 
            h += 1 
        if (h==24):
            h  = 0
    return (h, m, s)


def RAD2DMS(x):
    xx  =  abs(x*RADIAN_TO_ARCSEC) 
    d   =  int(xx/3600)
    xx -=  d*3600.0 
    m   =  int(xx/60) 
    s   =  xx-m*60.0 
    if (abs(s-60.0)<0.01):
        s  = 0.0 
        m += 1 
        if (m==60):
            m  = 0 
            d += 1
    if (x<0.0):
        d = -d 
    return (d, m, s)


def STR_HMS2RAD(s):
    """
    Convert string 'h m s' into radians
    """
    ss = s.lower().replace(':', ' ')
    ss = ss.replace('h',' ').replace('m', ' ').replace('s', ' ').replace(':', ' ').split()
    h, m, s = float(ss[0]), float(ss[1]), float(ss[2])
    return HMS2RAD(h,m,s)


def STR_DMS2RAD(s):
    """
    Convert string 'd am as' into radians    
    """
    ss = s.lower().replace(':', ' ')
    ss = ss.replace('d',' ').replace('m', ' ').replace('s', ' ').replace('\'',' ').replace('"', ' ').replace(':', ' ').split()
    d, am, ars = float(ss[0]), float(ss[1]), float(ss[2])
    # if -1.0<DEC<0.0, d=-0.0 => sign should be correctly interpreted in DMS2RAD
    res = DMS2RAD(d,am,ars)
    if (ss[0].find('-')>=0): 
        if (res>0.0):
            print('*'*80)
            print('ERROR IN STR_DMS2RAD !!!!   %s --> %.7f' % (s, res))
            print('*'*80)
        res = -abs(res)
    return res





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



def um2f(um):
    """
    Convert wavelength [um] to frequency [Hz]
    """
    return C_LIGHT/(um*1.0e-4)


def f2um(f):
    """
    Convert frequency [Hz] to wavelenth [um]
    """
    return 1.0e4*C_LIGHT/f


