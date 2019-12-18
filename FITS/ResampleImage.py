import os, sys
ISM_DIRECTORY = os.path.expanduser('~/GITHUB')
try:
    ISM_DIRECTORY = os.environ(['ISM_DIRECTORY'])
except:
    pass
sys.path.append(ISM_DIRECTORY)
from ISM.Defs      import *
from ISM.FITS.FITS import *


if (len(sys.argv)<4):
    print("\nUsage: ResampleImage g.fits A.fits B.fits" )
    print(" Resample g.fits onto the pixels defined by A.fits, write result to B.fits\n")
    print("*** THIS IS BETA SOFTWARE, NO GUARANTEES OF CORRECT RESULTS ***") 
    sys.exit()
A = pyfits.open(sys.argv[1])
B = pyfits.open(sys.argv[2])
Reproject(A, B, threads=4, cstep=4)
B.writeto(sys.argv[3], overwrite=True)


