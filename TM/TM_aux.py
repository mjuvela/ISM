import os, sys
import time
import numpy as np
import pyopencl as cl

from astropy.io import fits
from astropy import wcs
from astropy import units
from astropy.coordinates import SkyCoord

from matplotlib.pylab import *

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


def PIX2WCS(header, X, Y, radians=True):
    """
    Returns coordinates for pixels (X,Y) in the FITS image
    Usage:
        ra, de = PIX2WCS(header, X, Y, radians=False)
    Input:
        X, Y     =  vectors of column and row indices (floats)
        radians  =  if true, return in radians, otherwise in degrees (default=True)
    Note:
        Pixel coordinates have origin at 0
        We assume that NAXIS1 (first coordinate) is the longitude (x) coordinate == (X,Y) == (lon, lat)
    """
    tmp   = wcs.WCS(header=header)
    ra,de = tmp.wcs_pix2world(X,Y,0)  # (NAXIS1,NAXIS2) = (x,y) = (longitude, latitude)
    if (radians):
        return ra*DEGREE_TO_RADIAN, de*DEGREE_TO_RADIAN
    return ra, de
    


def WCS2PIX(header, lon, lat, radians=True):
    """
    Converts sky coordinates to FITS pixel coordinates (0-offset).
    Usage:
        i, j = WCS2PIX(header, lon, lat, radians=False)
    Input:
        header    =   FITS header
        lon, lat  =   sky coordinate vectors
        radians   =   if True, (lon,lat) are assumed to be in radians
        if False, (lon, lat) are in degrees (default radians=True)
    Returns:
        i, j  = vectors of column and row indices (floats)
    Note:
        We assume (lon,lat) = (NAXIS1, NAXIS2)
    """
    tmp   = wcs.WCS(header=header)
    if (radians):
        return tmp.wcs_world2pix(lon*RADIAN_TO_DEGREE, lat*RADIAN_TO_DEGREE, 0)
    else:
        # return tmp.wcs_world2pix(lon, lat, origin=0)
        return tmp.wcs_world2pix(lon, lat, 0)
        
    

def get_fits_pixel_size(header, radians=True):
    """
    Return pixel size of the FITS image described by the header
    Input:
        header   =  FITS header
        radians  =  if True, result returned in radians (default),
                    otherwise it is returned in degrees
    Note:
        if pixels are not square, returns the mean of the two dimensions
    """
    # Measure explicitly the distance corresponding to ten pixels, using wcs transformations
    lon, lat = PIX2WCS(header, [1, 1], [1, 11], radians=True)
    cosy     = cos(0.5*(lat[0]+lat[1]))
    d        = sqrt(((lon[1]-lon[0])*cosy)**2.0 + (lat[1]-lat[0])**2.0) / 10.0
    if (radians):
        return d
    else:
        return d*RADIAN_TO_DEGREE
        
        
