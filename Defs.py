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

