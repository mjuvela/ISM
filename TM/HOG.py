from MJ.mjDefs import *
import pyopencl as cl
import time

t0           =   time.time()
filename     =   '12CO_Tmb_small.fits'   # file containing a FITS cube
FWHM         =   32.0                     # FWHM in pixels (convolving kernel)
GPU          =   0                       # optionally use a GPU
# Run kernel Gradient to get gradient vectors for all images (l,m).
# Run kernel Sums to calculate V[l,m] and r[l,m].
F            =  pyfits.open(filename)
NV, NY, NX   =  F[0].data.shape           # first axis is the velocity
Gx           =  zeros((NV, NY, NX), float32)
Gy           =  zeros((NV, NY, NX), float32)
SIGMA        =  FWHM/sqrt(8.0*log(2.0))      # in pixels
platform = cl.get_platforms()[ [1,0][GPU] ]
if (GPU>0):
    device   =  platform.get_devices(cl.device_type.GPU)
    LOCAL    =  64
else:
    device   =  platform.get_devices(cl.device_type.CPU)
    LOCAL    =  8
context   =  cl.Context(device)
queue     =  cl.CommandQueue(context)
mf        =  cl.mem_flags
OPT       = " -D N=%d -D M=%d -D SIGMA=%.3ff -cl-fast-relaxed-math" % (NY, NX, SIGMA)
source    =  file('kernel_HOG.c').read()
program   =  cl.Program(context, source).build(OPT)
S_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*NX*NY)
Gx_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NX*NY)
Gy_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NX*NY)
Hx_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NX*NY)
Hy_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NX*NY)
IGRAD     =  program.Gradient  # kernel to calculate gradient vectors for one 2d image
IGRAD.set_scalar_arg_dtypes([None, None, None])
GLOBAL    =  (int((NX*NY)/64)+1)*64
F[0].data = asarray(F[0].data, float32)
SUMS      =  program.Sums      # kernel to calculate sums of W, W**2, W*cos, W*sin
SUMS.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None])
GLOBAL2   =  2048              # number of work items for the latter Sums kernel
CSUM_buf  =  cl.Buffer(context, mf.WRITE_ONLY, 4*GLOBAL)
SSUM_buf  =  cl.Buffer(context, mf.WRITE_ONLY, 4*GLOBAL)
WSUM_buf  =  cl.Buffer(context, mf.WRITE_ONLY, 4*GLOBAL)
W2SUM_buf =  cl.Buffer(context, mf.WRITE_ONLY, 4*GLOBAL)
CSUM      =  zeros(GLOBAL2, float32)
SSUM      =  zeros(GLOBAL2, float32)
WSUM      =  zeros(GLOBAL2, float32)
W2SUM     =  zeros(GLOBAL2, float32)
for k in range(NV):            # Calculate gradient images for each image plane
    cl.enqueue_copy(queue, S_buf, F[0].data[k,:,:])
    IGRAD(queue, [GLOBAL,], [LOCAL,], S_buf, Gx_buf, Gy_buf)
    cl.enqueue_copy(queue, Gx[k,:,:], Gx_buf)
    cl.enqueue_copy(queue, Gy[k,:,:], Gy_buf)
# Calculate V and r for each combination of planes
VLM = zeros((NV, NV), float32)
RLM = zeros((NV, NV), float32)
for i in range(NV):
    cl.enqueue_copy(queue, Gx_buf, Gx[i,:,:])            # gradients for the first image plane
    cl.enqueue_copy(queue, Gy_buf, Gy[i,:,:])
    for j in range(NV):                                  # for auto-HOG, could skip half of (i,j) calculations...
        cl.enqueue_copy(queue, Hx_buf, Gx[j,:,:])        # gradients for the second image plane
        cl.enqueue_copy(queue, Hy_buf, Gy[j,:,:])
        SUMS(queue, [GLOBAL2,], [LOCAL,], Gx_buf, Gy_buf, Hx_buf, Hy_buf, CSUM_buf, SSUM_buf, WSUM_buf, W2SUM_buf)
        cl.enqueue_copy(queue, CSUM,  CSUM_buf)
        cl.enqueue_copy(queue, SSUM,  SSUM_buf)
        cl.enqueue_copy(queue, WSUM,  WSUM_buf)
        cl.enqueue_copy(queue, W2SUM, W2SUM_buf)        
        res       =  sum(CSUM) / sqrt(0.5*sum(W2SUM))    # V[l,m]
        VLM[i,j]  =  res
        res       =  sqrt( (sum(CSUM))**2.0 + (sum(SSUM))**2.0 ) / sum(WSUM)  # r[l,m]
        RLM[i,j]  =  res
asarray(VLM, float32).tofile('VLM.bin')
asarray(RLM, float32).tofile('RLM.bin')
t0   = time.time()-t0

CM   = cm.hot
figure(1, figsize=(8.5,3.5))
subplots_adjust(left=0.1, right=0.94, bottom=0.13, top=0.9, wspace=0.27, hspace=0.27)
clf()
ax = subplot(121)
title(r'$V(l,m)$')
a, b =  percentile(ravel(VLM), (5.0, 98.9))
imshow(VLM, interpolation='nearest', vmin=a, vmax=b, cmap=CM)
text(0.0, -0.19, r'$t=%.2f \rm \/ \/ seconds \/ \/ NX=%d, NY=%d, NV=%d, FHWM=%.1f, GPU=%d$' % (t0, NX, NY, NV, FWHM, GPU), transform=ax.transAxes)
colorbar()
subplot(122)
title(r'$r(l,m)$')
a, b =  percentile(ravel(RLM), (5.0, 98.9))
imshow(RLM, interpolation='nearest', vmin=a, vmax=b, cmap=CM)
colorbar()
savefig('HOG_test_GPU%d_FWHM%02d.png' % (GPU, int(FWHM)))
SHOW()
