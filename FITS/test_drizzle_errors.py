import os, sys
ISM_DIRECTORY = os.path.expanduser('~/GITHUB')
try:
    ISM_DIRECTORY = os.environ(['ISM_DIRECTORY'])
except:
    pass
sys.path.append(ISM_DIRECTORY)
from    ISM.Defs import *
from    ISM.FITS.FITS import MakeEmptyFitsDim, Reproject
import  montage_wrapper as montage
import  scipy
from    scipy import ndimage
import matplotlib.ticker as ticker


"""
Compare results to Montage.reproject, for different latitude, different pixel size.
"""

np.random.seed(12345)

GPU      =  1    # set >0 to use GPU
scale    =  1.0
factor   =  1.0
threads  =  1

CSTEPS       = [    1,   2,   4,   8 ]
PIXEL_SIZES  = [  1.0, 10.0, 60.0, 300.0 ]                # arcsec
N, M         =  256, 256
LAT          =   10.0


def convolve_map(A, fwhm, mode='reflect', radius=3.6):
    # fwhm = fwhm of the beam,  in pixels
    # make the kernel with up to ?? fwhm
    n      = int(2+fwhm*radius)
    n = min(n, A.shape[0]-1)
    if (n%2==0):
        n += 1   # make sure it is odd (?)
    kernel = zeros((n,n), float32)
    K      = 4.0*log(2.0)/(fwhm*fwhm) 
    D2     = max(1.7, (1.5*fwhm))**2.0   # outer radius in pixels
    for i in range(n):                   # loop over kernel pixels
        dx  = (n-1.0)/2.0 - i
        for j in range(n):
            dy  = (n-1.0)/2.0 - j
            d2  = dx*dx+dy*dy
            if (d2<D2):
                kernel[i,j] =  exp(-K*d2)
    # normalize: integral over kernel equals one
    kernel /= sum(kernel.flat)
    tmp     = np.asarray(scipy.ndimage.convolve(np.asarray(A, np.float32), kernel, mode=mode), float32)
    return tmp




figure(1, figsize=(9,7))
rc('font', size=9)
subplots_adjust(left=0.07, right=0.97, bottom=0.05, top=0.94, wspace=0.26, hspace=0.38)

for row in range(4):
    cstep  = CSTEPS[row]

    for col in range(4):
        pix = PIXEL_SIZES[col]*ARCSEC_TO_RADIAN
        
        # Generate the input map
        offx =  1.3*pix
        offy = -0.8*pix
        A = MakeEmptyFitsDim(1.0,      LAT*DEGREE_TO_RADIAN,      pix,   N, M)
        B = MakeEmptyFitsDim(1.0,      LAT*DEGREE_TO_RADIAN,      pix,   N, M)
        G = MakeEmptyFitsDim(1.0+offx, LAT*DEGREE_TO_RADIAN+offy, pix,   N, M)
        G[0].data = np.asarray(randn(N, M), np.float32)+2.0
        J, I      = indices((N, M), np.float32)
        for i in range(1000):
            K          =  4.0*log(2.0)/(clip(2.0+1.0*randn(), 2.0, 99)**2.0)
            G[0].data +=  98.0*rand()*exp(-K*((J-N*rand())**2.0+(I-M*rand())**2.0))
        G[0].data        = clip(G[0].data, 1.0, +100.0)
        G[0].data[0:3,:] = 0.0
        G[0].data[-3:,:] = 0.0
        G[0].data[:,0:3] = 0.0
        G[0].data[:,-3:] = 0.0
        if (1):  # rotate the input image
            rot = 30*DEGREE_TO_RADIAN
            G[0].header['CD1_1'] =  G[0].header['CDELT1']*cos(rot)
            G[0].header['CD1_2'] = -G[0].header['CDELT2']*sin(rot)
            G[0].header['CD2_1'] =  G[0].header['CDELT1']*sin(rot)
            G[0].header['CD2_2'] =  G[0].header['CDELT2']*cos(rot)
        G.writeto('g.fits', overwrite=True)
    
        # Run montage.reproject
        A[0].header.totextfile('ref.header', overwrite=True)
        montage.reproject('g.fits', 'A.fits', 'ref.header', exact_size=True, factor=factor)
        A = pyfits.open('A.fits')

        # Run the OpenCL routine
        Reproject(G, B, GPU=GPU, cstep=cstep, shrink=1.0)
        

        # ignore the borders .... note that pixel values may be very small for the last pixels
        #  on the edges of the area covered by the input data
        MASK =  ones((N,M), float32)
        MASK[nonzero((A[0].data<0.5)|(~isfinite(A[0].data)))] = 0.0  # all map data >=1.0
        MASK[0:3,:] = 0.0
        MASK[-3:,:] = 0.0
        MASK[:,0:3] = 0.0
        MASK[:,-3:] = 0.0
        MASK =  convolve_map(MASK, 4)
        m    =  nonzero(MASK>0.98)   # ignoring anything close to map *and* coverage edges

        ax   =  subplot(4, 4, 1+col+4*row)
        if (0):  # plot absolute error
            X    =  B[0].data[m]-A[0].data[m]
        else:    # plot relative error
            X    =  (B[0].data[m]-A[0].data[m])/(A[0].data[m])
        a, b =  min(X), max(X)
        d    =  0.1*(b-a)
        hist(asarray(X, float32), bins=linspace(a, b, 40), log=True)
        xlim(a-d, b+d)
        err  = max(abs(X))
        text(0.06, 0.73, r'$\rm max(\Delta)=$', transform=ax.transAxes)
        text(0.06, 0.58, '%.1e' % err, transform=ax.transAxes)

        X    =  (B[0].data[m]-A[0].data[m]) / B[0].data[m]
        err  =   max(abs(X))
        if (cstep==8):
            merr = nonzero( abs((B[0].data-A[0].data)/B[0].data) == err)
            print("MAXIMUM ERROR AT:", merr, " rerr = ", err)
            #sys.exit()
        
        m    =  nonzero( abs((B[0].data-A[0].data) / B[0].data)==err )        
        text(0.65, 0.73, r'$\rm max(r)=$', transform=ax.transAxes)
        text(0.65, 0.58, '%.1e' % err, transform=ax.transAxes)
        title("cstep=%d, pix=%.0f''" % (cstep, PIXEL_SIZES[col]), size=9)
        if (0):
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useOffset=False, useMathText=True)
        if (0):
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
            xticks(rotation=30, ha='right')
        if (1):
            f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
            g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
            plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(g))
        if (0):
            A.writeto('A.fits', overwrite=True)
            B.writeto('B.fits', overwrite=True)
            sys.exit()
            
savefig("drizzle_errors.png")
show()
