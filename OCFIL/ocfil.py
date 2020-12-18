import os, sys
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)
from ocfil_aux import *

if (len(sys.argv)<2):
    print("")
    print("Usage: ocfil  fits_file  [ threshold  [ scale  [ FWHM] ] ]")
    print("     fits_file =  input image (2D FITS image in hdu=0)")
    print("     threshold =  probability threshold, e.g. 0.3")
    print("     scale     =  step between stencil points in arcsec, ~ scale of filament width")
    print("     FWHM      =  convolve input map with FWHM [arcsec] before analysis")
    print("""
    ocfil.py is a program to identify and extract 2D images of filaments in a figure:
        - optionally start by convolving the input image
        - calculate filament propability and position angle at each pixel location separately
        - label connected regions (pixels with probability above the given threshold)
        - trace the spine of the filaments
        - extract 2D images of the filaments in the figure\n    
""")
    sys.exit()

GPU       =  0                         # use GPU instead of CPU?
LEGS      =  3                         # points along the filament (for initial probability); stencil = 3xLEGS points
NDIR      =  11                        # number of position angles tested
PLIM      =  0.3                       # probability threshold (to identify connected regions)
PIX_LIMIT =  30                        # lower limit for the number of pixels in a filament
STEP      =  ARCMIN_TO_RADIAN          # characteristic scale (step between points in the stencil)
FWHM      =  -1.0

F         =  pyfits.open(sys.argv[1])  # open the FITS image to be analysed
if (len(sys.argv)>2): PLIM      =  float(sys.argv[2])                   # probability threshold
if (len(sys.argv)>3): STEP      =  float(sys.argv[3])*ARCSEC_TO_RADIAN  # step length [arcsec] -> [radian]
if (len(sys.argv)>4): FWHM      =  float(sys.argv[4])*ARCSEC_TO_RADIAN  # FWHM for convolution [arcsec] -> [radian]

N, M      =  F[0].data.shape            # image dimentions
PIX       =  GetFitsPixelSize(F)        # FITS image pixel size [radians]
m         =  np.nonzero( (F[0].data!=0.0) & (np.isfinite(F[0].data)) )   # valid pixels
F[0].data[np.nonzero(~np.isfinite(F[0].data))] = 0.0  # all missing pixels set to zero
if (FWHM>0):
    F     =  ConvolveFitsPyCL(F, FWHM, fwhm_orig=0.0)
    
# OpenCL initialisations
PLF       =  [0, 1, 2, 3 ]              # allowed OpenCL platforms
platform, device, context, queue, mf = InitCL(GPU, PLF) # OpenCL environment
LOCAL     =  [4, 32][GPU>0]             # local work group size
GLOBAL    =  ((N*M)//32+1)*32           # total number of work items >= number of pixels
OPT       =  '-D N=%d -D M=%d -D STEP=%.4ff -D LEGS=%d -D NDIR=%d' % (N, M, STEP/PIX, LEGS, NDIR)
source    =  open('kernel_ocfil.c').read()
program   =  cl.Program(context, source).build(OPT)
S         =  np.asarray(F[0].data, np.float32)
S_buf     =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)  # buffer for the input image
P_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*N*M)                        # filament probability (per pixel)
T_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*N*M)                        # best position angle
s_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*1024)                       # used for sigma estimates

# Calculate SIGMA value for the value change at the scale SCALE
Sigma     =  program.Sigma
Sigma.set_scalar_arg_dtypes([None, None])
Sigma(queue, [1024,], [LOCAL,], S_buf, s_buf)
s         =  np.zeros(1024, np.float32)
cl.enqueue_copy(queue, s,  s_buf)
SIGMA     =  np.mean(np.sqrt(s))
print("SIGMA = %.3f MJy/sr" % SIGMA)

# Calculate pixel-by-pixel probability that the individual pixel is on a filament
Prob      =  program.Probability
Prob.set_scalar_arg_dtypes([np.float32, None, None, None])       # S, P, T
Prob(queue, [GLOBAL,], [LOCAL,], SIGMA, S_buf, P_buf, T_buf)
P         =  np.empty_like(S)
T         =  np.empty_like(S)
cl.enqueue_copy(queue, P,  P_buf)
cl.enqueue_copy(queue, T,  T_buf)
P         =  np.exp(P)                  # convert log-probability ln(P) to probability P
print("Probability done")

pl.figure(1, figsize=(5.5, 7))
pl.rc('font', size=9)
pl.subplots_adjust(left=0.13, right=0.94, bottom=0.09, top=0.95, wspace=0.23, hspace=0.32)
ax = pl.subplot(321)
pl.imshow(S, cmap=pl.cm.gist_stern)
pl.title('Input image', size=9)
pl.colorbar()
ax = pl.subplot(322)
pl.imshow(P, cmap=pl.cm.gist_stern)
pl.title('Probability (per position)', size=9)
pl.colorbar()

# Label connected areas
m         =  np.nonzero(P>PLIM)
L         =  np.zeros(S.shape, np.int32)
L[m]      =  1
struc     =  [[1,1,1], [1,1,1], [1,1,1]]     # accept connections also in diagonal directions
L, NO     =  label(L, struc)                 # label() finds the connected regions
L        -=  1                               # -1 for no filemant, filaments labelled 0, 1, 2, ...
L         =  np.asarray(L, np.int32)
print("Label done, %d filaments" % NO)

# Remove filaments with less than PIX_LIMIT pixels
i = 0
while(i<NO):
    m = np.nonzero(L==i)
    if (len(m[0])<PIX_LIMIT):
        L[m] = -1
        L[np.nonzero(L>i)] -= 1
        NO -= 1
    else:
        i += 1
print("Prune done based on number of pixels")

# Simplify calculations by saying that no filament touches the image borders
L[0,:], L[-1,:], L[:,0], L[:,-1] = -1, -1, -1, -1 
L_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*N*M)
cl.enqueue_copy(queue, L_buf, L)

ax = pl.subplot(323)
LL = np.asarray(L, np.float32)
LL[np.nonzero(LL<0)] = np.NaN
pl.imshow(LL, cmap=pl.cm.tab20, interpolation='nearest') # try cm.hsv, cm.prism, cm.Set1, cm.tab20, ...
pl.title("%d filament candidates" % NO, size=9)
pl.colorbar()

# XY = pixel coordinates for the maximum probability in each filament
XY = np.zeros((NO, 2), np.float32)   # start positions for each filament
for i in range(NO):
    m    =  np.nonzero(L==i)
    j    =  np.argmax(P[m])
    y, x =  m[0][j], m[1][j]
    XY[i,:] = [x, y]
    pl.plot(x, y, 'ko', ms=2)
    pl.text(1.03*x, 0.99*y, '%d' % i, size=7, color='k')
XY_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=XY) # start positions for tracing the spines
print("Maxima selected")

# Trace the filaments
R         =  np.zeros((NO, 1000, 3), np.float32)             # coordinates for traces of the filament spines
R_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*NO*1000*3)  # at most 1000 points along the filament (x, y, p)
cl.enqueue_copy(queue, P_buf,  P)                            # updated to probability instead of lnP
Trace     =  program.Trace
#                                 NO        PLIM        XY      S      P      T      L,     RW   
Trace.set_scalar_arg_dtypes([     np.int32, np.float32, None,   None,  None,  None,  None,  None])
Trace(queue, [GLOBAL,], [LOCAL,], NO,       PLIM,       XY_buf, S_buf, P_buf, T_buf, L_buf, R_buf)   # R = 'route'
cl.enqueue_copy(queue, R,  R_buf)
print("Trace done")

ax = pl.subplot(324)
pl.imshow(F[0].data)
pl.title("Traces", size=9)
pl.colorbar()
NP  = np.zeros(NO, np.int32)  # number of points along the filament
fp  = open('spine.dat', 'wb') # dump all filament spine traces to a single binary file
for i in range(NO):           # check the length of each filament (how many steps)
    m  =  np.nonzero(R[i,:,0]<0.0)
    if (len(m[0])<1): continue
    j  =  m[0][0]       # j positions along the filament
    pl.plot(R[i, 0:j, 0], R[i, 0:j, 1], 'w-', ms=1)
    NP[i] = j           # number of (x, y) points along the filament
    print("  Filament %3d has %3d points" % (i, j))
    np.asarray([NP[i],], np.int32).tofile(fp)        # number of points for the current filament
    np.asarray(R[i, 0:j, :], np.float32).tofile(fp)  # (x,y) values
fp.close()

# Extract 2D images for the filaments -- HERE JUST THE LONGEST ONE AS AN EXAMPLE
ID    =  np.argmax(NP)  # select the longest filament
ns    =  NP[ID]         # points along the filament
NR    =  40             # 2*NR+1 steps across the filament (the profile)
F_buf =  cl.Buffer(context, mf.WRITE_ONLY, 4*1000*(2*NR+1))
TD    =  program.Filament2D
TD.set_scalar_arg_dtypes([ np.int32, np.int32, np.int32, None, None, None, None])   
TD(queue, [GLOBAL,], [LOCAL,], ID, ns, NR,  R_buf, S_buf, T_buf, F_buf) 
F     =  np.zeros((ns, 2*NR+1), np.float32)
cl.enqueue_copy(queue, F, F_buf)
# Save 2d filament image to a FITS file
G     =  MakeEmptyFits(0.0, 0.0, 10.0, 1.0, 'fk5')
G[0].header['CDELT2'] = 0.25*STEP*RADIAN_TO_DEGREE  # filament stepped in steps of 0.25*STEP
G[0].header['CDELT1'] = 0.25*STEP*RADIAN_TO_DEGREE  # profile sampled at 0.25*STEP steps
G[0].data = F
G.writeto('filament.fits', overwrite=True)

ax = pl.subplot(325)
pl.title('Sample filament image', size=9)
pl.imshow(F, aspect='auto', cmap=pl.cm.gist_stern, 
   extent=[-20.0*STEP*RADIAN_TO_ARCMIN, +20.0*STEP*RADIAN_TO_ARCMIN,  0.0, NP[ID]*0.5*STEP*RADIAN_TO_ARCMIN])
pl.text(0.1, 0.9, "Filament %d" % ID, transform=ax.transAxes, color='w')
pl.xlabel(r'$\rm Profile \/ \/ (arcmin)$')
pl.ylabel(r'$\rm Length \/ \/ (arcmin)$')
pl.colorbar()
pl.subplot(326)
p  =  R[ID, 0:NP[ID], 2]
# modified probability p=1-(1-p)*(1-p(-)/2)*(1-p(+)/2), using probabilities of previous and next filament position
pp =  1.0*p
for i in range(2, NP[ID]-2):  # use p(i-2) and p(i+2) that are almost independent of p(i)
    pp[i] = 1.0 - (1.0-p[i])*(1.0-p[i-2]/2.0)*(1.0-p[i-2]/2.0)
pl.plot(p,  np.arange(NP[ID])*0.5*STEP*RADIAN_TO_ARCMIN, 'o', lw=2, mfc='none', ms=3)
pl.plot(pp, np.arange(NP[ID])*0.5*STEP*RADIAN_TO_ARCMIN, 'bo-', lw=1, ms=3)
pl.xlabel('Probability')
pl.title('Probability along filament', size=9)
pl.show()
