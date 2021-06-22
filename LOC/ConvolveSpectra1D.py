#!/usr/bin/env python
import os, sys
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
from   LOC_aux import *

if (len(sys.argv)<2):
    print("Usage:")
    print("    ConvolveSpectra1D.py   loc_spectrum_file   fwhm_in_arcsec [ angle_as [ samples ] ]") 
    print("")
    sys.exit()
    
filename  =  sys.argv[1]
fwhm_as   =  float(sys.argv[2])
angle_as  =  -1
samples   =  201
if (len(sys.argv)>3):  
    angle_as = float(sys.argv[3])
if (len(sys.argv)>4):  
    samples  = int(sys.argv[4])
    if (samples<0):
        samples = 201
    
if  (angle_as<0):
    print("Convolve %s, FWHM=%.2f arcsec, %dx%d samples" % (filename, fwhm_as, samples, samples))
else:
    print("Convolve %s, FWHM=%.2f arcsec, model %.2f arcsec, %d x %d samples" % (filename, fwhm_as, angle_as, samples, samples))
    
ConvolveSpectra1D(filename, fwhm_as, GPU=0, platforms=[0,1,2,3,4], angle_as=angle_as, samples=samples)
