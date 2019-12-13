import os, sys
ISM_DIRECTORY = os.path.expanduser('~/GITHUB')
try:
    ISM_DIRECTORY = os.environ(['ISM_DIRECTORY'])
except:
    pass
sys.path.append(ISM_DIRECTORY)
import ISM.Defs
from   ISM.FITS.FITS import *



# Make a FITS image
N, M  =  256, 212
F     =  MakeEmptyFitsDim(0.0, 0.0, pix=ARCMIN_TO_RADIAN, m=M, n=N, sys_req='fk5')
# add something to the image
J, I = indices((N,M), float32)
for i in range(20):
    x    =  M*rand()
    y    =  N*rand()
    fwhm =  2.0+10.0*rand()
    K    =  4.0*log(2.0)/fwhm**2
    F[0].data += rand()*exp(-K*( (I-x)**2.0 + (J-y)**2.0 ))

print(GetFitsPixelSize(F)*RADIAN_TO_ARCMIN)

fwhm_pix  =  3.0
fwhm_rad  =  fwhm_pix*ARCMIN_TO_RADIAN
# calculate beam into a 2D array
J, I      =  indices((21,21), float32)
K         =  4.0*np.log(2.0)/fwhm_pix**2
beam      =  np.exp(-K*( (I-10.0)**2 + (J-10.0)**2))
beam     /=  np.sum(np.ravel(beam))


# convolve using  ConvolveFitsPyCL
G1  =  ConvolveFitsPyCL(F, fwhm_rad, fwhm_orig=0.0, RinFWHM=3.0)
# convolve using the beam
G2  =  ConvolveFitsBeamPyCL(F, beam)
# convolve directly the 2D array
X   =  ConvolveMapBeamPyCL(F[0].data, beam)


figure(1, figsize=(8,6))
subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.94, wspace=0.28, hspace=0.28)

subplot(2,2,1)
title("Original")
imshow(F[0].data)
colorbar()

subplot(2,2,2)
title("G1")
imshow(G1[0].data)
colorbar()

subplot(2,2,3)
title("G2-G1")
imshow(G2[0].data-G1[0].data)
colorbar()

subplot(2,2,4)
title("G3-G1")
imshow(X-G1[0].data)
colorbar()


show()
