INSTALL_DIR = '/home/mika/GITHUB/ISM/TM'
import os, sys
sys.path.append(INSTALL_DIR)
from Pattern import *

if (1):
    GPU    =   1
    LOCAL  =  32
else:
    GPU    =   0
    LOCAL  =   8
    
F = fits.open("PSW.fits")

figure(1, figsize=(12,4.5))
subplots_adjust(left=0.05, right=0.95, bottom=0.12, top=0.92, wspace=0.25, hspace=0.25)

subplot(251)
imshow(F[0].data, vmin=0, vmax=120)
title("Input map")
colorbar()
# First RHT
t0 = time.time()
R1, T1, dT1 = RollingHoughTransformBasic(F, DK=20, DW=30, FRAC=0.8, GPU=GPU, local=LOCAL)
print("========== RollingHoughTransformBasic: %.2f seconds" % (time.time()-t0))
m = nonzero(R1<=0.0)
R1[m], T1[m] = NaN, NaN  # better for the plotting
subplot(252)
imshow(R1)
title(r'$RHT_1$')
colorbar()
subplot(257)
imshow(T1*RADIAN_TO_DEGREE)
title(r'$\theta_1$')
colorbar()
# Second RHT
#  we select a larger scale to make the result different from the above
t0 = time.time()
R2, T2, dT2 =  RollingHoughTransformOversampled(F, DK=30, DW=45, NW=2, STEP=0.5, FRAC=0.8, GPU=GPU, local=LOCAL)
print("========== RollingHoughTransformOversampled: %.2f seconds" % (time.time()-t0))
m = nonzero(R2<=0.0)
R2[m], T2[m] = NaN, NaN  # better for the plotting
subplot(253)
imshow(R2)
title(r'$RHT_2$')
colorbar()
subplot(258)
imshow(T2*RADIAN_TO_DEGREE)
title(r'$\theta_2$')
colorbar()
# First template matching
t0 = time.time()
SIG1, PA1, SS1 = Centipede(F, FWHM_PIX=10.0, LEGS=3, STUDENT=1, NDIR=17, K_HPF=2.0, GPU=GPU, local=LOCAL)
print("========== Centipede: %.2f seconds" % (time.time()-t0))
# lets add some thresholding also to Centipede results
m = nonzero(SIG1<percentile(SIG1[nonzero(SIG1>0.0)], (80,)))
SIG1[m], PA1[m] = NaN, NaN
subplot(254)
imshow(SIG1)
title(r'$TM_1$')
colorbar()
subplot(259)
imshow(PA1*RADIAN_TO_DEGREE)
title(r'$PA_1$')
colorbar()
# Second template matching
# note --- with same parameters the result would be identical to the above run
#          here we omit the normalisation (STUDENT=0) and select a larger spatial scale
PAT = zeros((3,3), float32)
for i in range(3): PAT[i,:] = [ -0.5, +1.0, -0.5 ]
t0 = time.time()
SIG2, PA2, SS2 = PatternMatch(F, FWHM_PIX=15.0, FILTER=1, STUDENT=0, PAT=PAT, THRESHOLD=0.9, GPU=GPU, NDIR=21,
                           K_HPF=2.0, SYMMETRIC=True, STEP=1.0, AAVE=0)                          
print("========== PatternMatch: %.2f seconds" % (time.time()-t0))
# lets add some thresholding also to template matching results
m = nonzero(SIG2<percentile(SIG2[nonzero(SIG2>0.0)], (80,)))
SIG2[m], PA2[m] = NaN, NaN
subplot(2,5,5)
imshow(SIG2)
title(r'$TM_2$')
colorbar()
subplot(2,5,10)
imshow(PA2*RADIAN_TO_DEGREE)
title(r'$PA_2$')
colorbar()

savefig("test_TM.png")

show()



