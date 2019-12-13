# ISM_DIRECTORY must have been defined and added to the path
from   ISM.Defs import *
import ISM.FITS


print("*** IMPORT Nicer_aux.py ***")


# from old WCS  !!
# WCS_J2000    = 1  #    J2000(FK5) right ascension and declination   
# WCS_B1950    = 2  #    B1950(FK4) right ascension and declination   
# WCS_GALACTIC = 3  #    Galactic longitude and latitude   
# WCS_ECLIPTIC = 4  #    Ecliptic longitude and latitude   



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


def PIX2WCS_ALL(header, radians=True):
    """
    Returns coordinate vectors for all pixels in the FITS image
    Usage:
        ra, de = PIX2WCS_ALL(header, radians=False)
    Input:
        radians  =  if true, return in radians, otherwise in degrees (default=True)
    Output:
        X, Y  =  coordinate vectors in degrees or radians
    Note:
        Pixel coordinates have origin 0
    """
    M     = int(header['NAXIS1'])  # columns  =  longitude = X
    N     = int(header['NAXIS2'])  # rows     =  latitude  = Y
    Y, X  = np.indices((N,M), np.float32)
    return PIX2WCS(header, X.copy(), Y.copy(), radians=radians) # arguments ~ (NAXIS1, NAXIS2)
    
    
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
    
    
    
def WCSExtent(header, radians=True):
    """
    Return [lon_min, lon_max, lat_min, lat_max]
    Usage:
        E = WCSExtent(header, radians=True)   
    Input:
        header   = FITS image header
        radians  = if True, result in radians, otherwise in degrees
                   (default=True)
    Output:
        E = [ lon1, lon2, lat1, lat2 ]
    """
    N     = int(header['NAXIS2'])  # rows
    M     = int(header['NAXIS1'])  # columns
    x     = asarray([0,   0, M-1, M-1], float32)
    y     = asarray([0, N-1,   0, N-1], float32)
    lon, lat = PIX2WCS(header, x, y, radians=False)
    E     = asarray([min(lon), max(lon), min(lat), max(lat)], float32)
    if (radians): return E*DEGREE_TO_RADIAN
    else:         return E
    
    
def WCSCentre(header, radians=True):
    """
    Return centre coordinates for a FITS image.
    Usage:
        lon, lat = WCSCentre(header, radians=False)
    """
    y        = 0.5*(int(header['NAXIS2'])-1.0)  # rows
    x        = 0.5*(int(header['NAXIS1'])-1.0)  # columns
    lon, lat =  PIX2WCS(header, [x,], [y,], radians=False)
    if (radians):
        return lon[0]*DEGREE_TO_RADIAN, lat[0]*DEGREE_TO_RADIAN
    else:
        return lon[0], lat[0]
    
    
                    
def GetFitsValues(F, lon, lat, hdu=0, radians=True):
    """
    Read values from a FITS image based on sky coordinates.
    Usage:
        GetFitsValues(F, lon, lat, hdu=0, radians=True)
    Input:
        F        =   FITS image
        lon,lat  =   sky coordinates
        hdu      =   number of the header unit
        radians  =   if True, (lon,lat) assumed to be in radians, otherwise in degrees
    Return:
        Values read from the image, -1e10 for coordinates falling outside the image.
    """
    S      =  lon.shape
    I, J   =  WCS2PIX(F[hdu].header, ravel(lon), ravel(lat), radians=radians)
    print(I)
    print(J)
    M, N   =  F[hdu].header['NAXIS1'], F[hdu].header['NAXIS2']
    res    =  -1e10*ones(len(I))
    m      =  nonzero((I>=0.0)&(I<=M)&(J>=0)&(J<=N))
    i      =  asarray(floor(I[m]), int32)
    j      =  asarray(floor(J[m]), int32)
    res[m] =  F[hdu].data[j,i]
    print(res.shape, res[m].shape)
    return res.reshape(lon.shape)


def wcscon(lon, lat, sys1='fk5', sys2='galactic', unit='rad'):
    """
    Convert coordinates to another system.
    Input:
        lon, lat = vectors of input coordinates in units unit
        sys1     = system of input coordinates
        sys2     = system of output coordinates
        unit     = unit of both inputs and outputs, deg or rad
    Return:
        new vectors (lon2, lat2) in the new system, in units unit,
        the same shape as for the input data
    """
    frame_from = sys1.lower()
    frame_to   = sys2.lower()
    if (frame_from in ['J2000', 'J2000.0']):
        frame_from = 'fk5'
    elif (frame_from in ['B1950', 'B1950.0']):
        frame_from = 'fk4'
    elif (frame_from=='ecliptic'):
        frame_from = 'heliocentrictrueecliptic'
    ###
    if (frame_to in ['J2000', 'J2000.0']):
        frame_to = 'fk5'
    elif (frame_to in ['B1950', 'B1950.0']):
        frame_to = 'fk4'
    elif (frame_to=='ecliptic'):
        frame_to = 'heliocentrictrueecliptic'
    ##
    S = lon.shape   # remember the shape of the input array
    lon, lat = ravel(lon), ravel(lat)
    if (frame_from in ['fk4', 'fk5']):
        A = SkyCoord(ra=lon, dec=lat, frame=frame_from, unit=unit)
    elif (frame_from in ['heliocentrictrueecliptic']):
        A = SkyCoord(lon=lon, lat=lat, frame=frame_from, unit=unit)
    else:
        A = SkyCoord(l=lon, b=lat, frame=frame_from, unit=unit)
    B = A.transform_to(frame_to)
    ##
    if (frame_to in ['galactic']):
        return asarray(B.l.to(unit), float32).reshape(S), asarray(B.b.to(unit), float32).reshape(S)
    if (frame_to in ['heliocentrictrueecliptic']):
        return asarray(B.lon.to(unit), float32).reshape(S), asarray(B.lat.to(unit), float32).reshape(S)
    else:
        return asarray(B.ra.to(unit), float32).reshape(S), asarray(B.dec.to(unit), float32).reshape(S)
    

    
def wcscon_deg(lon, lat, sys1='fk5', sys2='galactic'):
    """
    Usage:
        lon, lat = wcscon_deg(lon, lat, sys1='fk5', sys2='galactic')
    """
    return wcscon(lon, lat, sys1, sys2, unit='deg')



def wcscon_rad(lon, lat, sys1='fk5', sys2='galactic'):
    """
    Usage:
        lon, lat = wcscon_rad(lon, lat, sys1='fk5', sys2='galactic')
    """
    return wcscon(lon, lat, sys1, sys2, unit='rad')



def get_fits_pixel_size(header, radians=True):
    """
    Return pixel size of the FITS image described by the header
    Input:
        header   =  FITS header
        radians  =  if True, result returned in radians (default),
                    otherwise it is returned in degrees
    """    
    # Measure the distance corresponding to ten pixels
    lon, lat = PIX2WCS(header, [1, 1], [1, 11], radians=True)
    cosy     = cos(0.5*(lat[0]+lat[1]))
    d        = sqrt(((lon[1]-lon[0])*cosy)**2.0 + (lat[1]-lat[0])**2.0) / 10.0
    if (radians):
        return d
    else:
        return d*RADIAN_TO_DEGREE

    

#==========================================================================================
# Av-related
#==========================================================================================



NIR_WVL = { 'U' : 0.365, 'B' : 0.440, 'V' : 0.556, 'I' : 0.90, \
            'R' : 0.70, 'J' : 1.25, 'H' : 1.65, 'Ks' : 2.16, 'K' : 2.20,  \
            'L' : 3.25, 'H-alpha' : 0.6563 ,
            'W1' : 3.4, 'W2' : 4.6, 'W3' : 12.0, 'W4' : 22.0 }
                                    

def CardelliA(band, Rv):
    """
    Usage:
        A_Av = CardelliA(band, Rv)
    Input:
        band = name(s) of bands or wavelength(s) in micrometers
        Rv   = Av/E(B-V) value defining the extinction curve shape
    Note:
        Returns A(band)/A(V) according to Cardelli -89 parametrization.
        Band is U, B, V, R, I, J, H, K, Ks, L or the wavelength in micrometers.
        Routine accepts as band a string, a scalar um, or an array even with mixed 
        elements of strings and floating point numbers.
    """
    band       = asarray(band)
    band.shape = (size(band),)                 # force an 1d array
    um         = zeros(size(band), float32)
    res        = zeros(size(band), float32)
    for i in range(len(um)):
        try:
            um[i] = float(NIR_WVL[band[i]])    # either name of a band
        except:
            um[i] = float(band[i])             # ... or wavelength in um
    x      = 1.0/um
    ##
    m      = nonzero((x>=0.3)&(x<=1.1))
    res[m] = (0.574*x[m]**1.61) - (0.527*x[m]**1.61)/Rv
    ##
    m      = nonzero((x>=1.1)&(x<=3.3))
    y      = x[m]-1.82
    res[m] = \
        (1.0+0.17699*y-0.50447*y**2.0-0.02427*y**3.0+0.72085*y**4.0 \
               +0.01979*y**5.0-0.77530*y**6.0+0.32999*y**7.0)       \
        + (1.41338*y+2.28305*y**2.0+1.07233*y**3.0-5.38434*y**4.0 \
               -0.62251*y**5.0+5.30260*y**6.0-2.09002*y**7.0) / Rv
    if (len(res)==1):
        return res[0]
    else:
        return res
    


def CardelliRatio(Rv):
    # return (J-H)/(H-K) for Cardelli extinction curve
    tmp = (CardelliA('J',Rv)-CardelliA('H',Rv))/(CardelliA('H',Rv)-CardelliA('K',Rv))
    # print 'CardelliRatio(%5.2f) = %8.4f' % (Rv, tmp)
    return tmp


def A_Av_Flaherty(um):
    """
    Return extinction in magnitudes at the given wavelength, relative to V-band.
    """
    print('*** WARNING *** A_Av_Flaherty() is very approximate beyond 4.6 microns!!')
    # Values at 3.6, 4.5, 5.8, and 8.0 from Flaherty et al. 2007, Table 3.
    # This routine is used in Koenig() to de-redden 2MASS and WISE data.
    #    2MASS => one uses actually Cardelli curve with Rv=3.1
    #    WISE  => 3.4, 4.6, 12, 22  ~ the table below contains approximate wavelengths.
    # AT 12UM ONE SHOULD INTEGRATE OVER FILTER TO TAKE INTO ACCOUNT THE SHAPE OF THE
    # SILICATE PEAK ... see also Rosenthal et al. (2000).
    # ...but Koenig() needs reddening only at 3.4 and 4.6 um!!
    UM    = [ 1.99, 2.2, 3.00,  3.6,   4.5,   5.8,   8.0,  12.0,  24.0 ]
    A_Ak  = [ 1.00, 1.0, 0.75,  0.63, 0.53,  0.48,  0.49,  1.2,   0.54 ]
    ip    = interp1d(UM, A_Ak)
    # return value from Cardelli ??
    if (um==1.25):  return CardelliA('J', Rv=3.1)
    if (um==1.65):  return CardelliA('H', Rv=3.1)
    if (um==2.2):   return CardelliA('K', Rv=3.1)
    if (um==2.25):  return CardelliA('K', Rv=3.1)
    return  ip(um) * CardelliA('K', Rv=3.1) #   (AX/AK) * (AK/AV) =  AX/AV

                        

def get_AX_AV(bands, Rv=3.1):
    """
    Return A(band)/AV for the default extinction curve (Rv=3.1), 'band' is array of band names, e.g., ['J', 'H', 'K']
    2013-01-26 => use the Cardelli extinction curves.
    """
    N    = len(bands)
    K    = zeros(N, float32)
    for i in range(N):  K[i] = CardelliA(bands[i], Rv)
    return K



def get_excess_Av_ratios(bands, Rv=3.1, old_curve=False):
    """
    Given array of band names ('U', 'B', ..., 'K'), return ratios between
    colour excess (0-1), (1-2), (2-3), ... and the Av.
    Assumes normal Cardelli extinction curve
    """
    A_AV = { 'U': 1.52, 'B': 1.33, 'V': 1.00, 'R': 0.74, 'I': 0.48, 'J': 0.26, 'H': 0.15, 'K': 0.087 }
    colours = len(bands)-1
    K   = zeros(colours, float32)
    for i in range(colours):
        if ((Rv==3.1)&(old_curve==True)):
            K[i] = A_AV[bands[i]]-A_AV[bands[i+1]]
        else:
            K[i] = CardelliA(bands[i], Rv) - CardelliA(bands[i+1], Rv)
    return K



def get_excess_Av_ratios_dustfile(bands, filename):
    """
    Return ratios of colour excess and Av for given CRT dust file.
    """
    d = pylab.load(filename, skiprows=10)
    F        = d[:,0]
    X        = d[:,2]+d[:,3]
    ip       = interp1d(F, X)
    colours  = len(bands)-1
    K        = zeros(colours, float32)
    fV       = C_LIGHT/0.55e-4
    for i in range(colours):
        um1  = NIR_WVL[bands[i]]
        um2  = NIR_WVL[bands[i+1]]
        f1   = C_LIGHT/(1.0e-4*um1)
        f2   = C_LIGHT/(1.0e-4*um2)
        K[i] = (ip(f1)-ip(f2))/ip(fV)
    return K    
    

#==========================================================================================



def InitCL(GPU=0, platforms=[], sub=0):
    """
    OpenCL initialisation, searching for the requested CPU or GPU device.
    Usage:
        platform, device, context, queue, mf = InitCL(GPU=0, platforms=[], sub=0)
    Input:
        GPU       =  if >0, try to return a GPU device instead of CPU
        platforms =  optional array of possible platform numbers, default [0, 1, 2, 3, 4, 5]
        sub       =  optional number of threads for a subdevice (CPU only, first subdevice returned)
    """
    platform, device, context, queue = None, None, None, None
    possible_platforms = range(6)
    if (len(platforms)>0):
        possible_platforms = platforms
    device = []
    for iplatform in possible_platforms:
        print("try platform %d..." % iplatform)
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
    print(".... device ....")
    print(device)
    return platform, device, context, queue,  cl.mem_flags
                
