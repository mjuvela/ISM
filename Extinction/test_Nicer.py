import os, sys

# INSTALL_DIR is needed to find the kernel files that must be in the same directory as this script
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__)) 
sys.path.append(INSTALL_DIR)
from Nicer import *


# Select a target cloud
ra0  = HMS2RAD( 15, 39, 42.0)
de0  = DMS2RAD( -7, 10,  0.0)
box  = 60.0*ARCMIN_TO_RADIAN   # map size in radians
pix  = 0.5*ARCMIN_TO_RADIAN    # pixel size
fwhm = 3.0*ARCMIN_TO_RADIAN    # fwhm of the Av map
npix = int(box/pix)            # map size as number of pixels

# Read 2Mass stars for a reference field (OFF field)
coo, mag, dmag = read_2mass_www(ra0, de0+box, box_size=0.5*box, filename='OFF.2Mass')

# Read 2Mass stars for the ON field
COO, MAG, DMAG = read_2mass_www(ra0, de0    , box_size=box,     filename='ON.2Mass')

# Make a template FITS image for the extinction map and the error map
F  = MakeEmptyFitsDim(ra0, de0, 0.5*ARCMIN_TO_RADIAN, npix, npix)
dF = MakeEmptyFitsDim(ra0, de0, 0.5*ARCMIN_TO_RADIAN, npix, npix)

# Choose extinction curve
EX_AV = get_AX_AV(['J', 'H', 'Ks'], Rv=3.1)

# Calculate extinction and save the results to FITS files
NICER_with_OpenCL(F, COO, MAG, DMAG, mag, dmag, EX_AV, FWHM=3.0*ARCMIN_TO_RADIAN, CLIP=3.0, GPU=0, dF=dF)
F.writeto( 'test_Av.fits' , overwrite=True)
dF.writeto('test_dAv.fits', overwrite=True)

