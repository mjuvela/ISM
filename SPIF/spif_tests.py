import sys, os
sys.path.append('./')
from SPIF_aux import *

"""
This script:
    - creates synthetic spectral line observations and saves those
      as input files for SPIF
    - creates INI files for SPIF
    - runs SPIF
    - produces some plots on the results
    
The cases are:
    1 = spectra with a single Gaussian, chi2 minimisation
    2 = spectra with a single Gaussian, Monte Carlo error estimates
    3 = spectra with a single Gaussian, MCMC error estimates
    4 = spectra with two Gaussians (chi2 minimisation)
    5 = spectra with two Gaussians (chi2 minimisation) + penalty functions
    6 = two sets of spectra, both with a single Gaussian (chi2 minimisation)
    7 = hyperfine spectra with a single velocity component (chi2 minimisation)
    8 = hyperfine spectra with a single velocity component, MCMC error estimates
    9 = hyperfine spectra with a single velocity component, MCMC error estimates with priors
        
With # being the case number above:
    - the input files will be E#.fits, and dE#.fits
    - the written INI files will be E#.ini
    - the command lines will be saved to E#.sh
with the exception of cases 1-3 and 4-5 use the same input files.    
"""
CASES = arange(10)                # array of cases to recompute

M, NX, NY = 100, 64, 64           # velocity channels + number of spectra on the sky
N = NY*NX                         # total number of spectra
V = linspace(-5.0, +5.0, 100)     # chosen channel velocities
NOISE = linspace(0.01, 0.3, N)    # relative noise, varies over the map

Q = 4.0*log(2.0)          
def Gauss(p, v):  # return Gaussian profile for parameters p = [ T, v, fwhm ], velocities v
    return p[0] * exp(-Q*((v-p[1])/p[2])**2)

def make_fits(D, dv=0.1):
    hdu       = pyfits.PrimaryHDU(D)
    F         = pyfits.HDUList([hdu])
    if (len(D.shape)==3): # 3D => add some keywords for NAXIS3
        F[0].header['CRPIX3'] = 0.5*(D.shape[0]+1.0)
        F[0].header['CRVAL3'] = 0.0
        F[0].header['CDELT3'] = dv
    return F


if (1 in CASES): # single Gaussian, chi2 minimisation
    p0 = asarray([1.0, 0.5, 1.5], float32)   # true parameters of the Gaussian
    y0 = Gauss(p0, V)                        # template noiseless profile
    ymax = max(y0)        # error set relative to peak value ymax
    Y1 = zeros((M, N), float32)
    np.random.seed(123)   # for reproducibility
    for k in range(N): Y1[:,k] = y0 + NOISE[k]*ymax*randn(M) # noisy spectra
    # Write observations and their error estimates to FITS files
    Y1.shape = (M, NY, NX)
    F        = make_fits(Y1)
    F.writeto('E1.fits', overwrite=True)
    # errors = 2D FITS images
    err      = asarray(NOISE*ymax, float32).reshape(NY, NX)
    dF       = make_fits(err)
    dF.writeto('dE1.fits', overwrite=True)
    #--------------------------------------------------------------------------------
    # write the INI file for this fit
    INI = """
    fits1  = E1.fits
    dfits1 = dE1.fits
    y1     = GAUSS(v1, x[0], x[1], x[2])
    init   = y1:tmax  y1:vmax  y1:fwhm   # ... not necessarily needed
    prefix = E1
    """
    with open('E1.ini', 'w') as fp: fp.write(INI)
    #--------------------------------------------------------------------------------
    # run SPIF
    CMD = 'python SPIF.py -ini E1.ini -iter 1 -method 0'
    with open('E1.sh', 'w') as fp: fp.write(CMD+'\n') # save a copy of the command line
    os.system(CMD)
    #--------------------------------------------------------------------------------
    # plot 10x10 observed spectra with the fit
    P = pyfits.open('E1_res.fits')  # fitted parameters, P[NP, NY, NX]
    figure(1, figsize=(9,7))
    subplots_adjust(left=0.1, right=0.94, bottom=0.12, top=0.95, wspace=0, hspace=0)
    for j in range(10):
        jj = int(NY-1-j*(NY-1.0)/9.5)
        for i in range(10):
            ii = int(i*(NX-1.0)/9.5)
            ax = subplot(10, 10, 1+i+10*j)
            if (i>0): setp(ax, yticklabels=[])
            if (j<9): setp(ax, xticklabels=[])
            plot(V, F[0].data[:,jj,ii], 'k-', ds='steps', lw=0.8) # observation
            p = P[0].data[:,jj,ii]  # parameter vector for the spectrum (j,i) = [T, v, fwhm, chi2]
            plot(V, Gauss(p, V), 'r-')
    savefig('E1.png')
    savefig('E1.pdf', bbox_inches='tight')
    SHOW()
    
    
    
if (2 in CASES): # as above but calculating Monte Carlo error estimates
    # we reuse the E1.fits and dE1.fits data from CASE=1... even the INI file is the same
    #--------------------------------------------------------------------------------
    # run SPIF
    CMD = 'python SPIF.py -ini E1.ini -iter 1 -method 0 -mc 100'
    with open('E1.sh', 'w') as fp: fp.write(CMD+'\n') # save a copy of the command line
    os.system(CMD)
    # not that without "-fullsave 1", the Monte Carlo samples are not saved, only their
    # mean values and standard deviations will end up in E1_mc.fits
    #--------------------------------------------------------------------------------
    # plot parameter errors as 2d images
    figure(1, figsize=(10, 2.5))
    subplots_adjust(left=0.1, right=0.94, bottom=0.15, top=0.92, wspace=0.4)
    E = pyfits.open('E1_mc.fits') #  data [NP, NY, NX] = stdev values
    L = [r'$\sigma(T)$', r'$\sigma(V)$', r'$\sigma(FWHM)$'] 
    for i in range(3):
        ax = subplot(1,3,1+i)
        imshow(E[0].data[i,:,:])
        title(L[i])
        colorbar()
    savefig('E2.png')
    savefig('E2.pdf', bbox_inches='tight')
    SHOW()
    

    
if (3 in CASES): # as above but calculating MCMC error estimates
    # we reuse the E1.fits, dE1.fits, and E1.ini
    #--------------------------------------------------------------------------------
    # run SPIF, Metropolis MCMC, including pre-optimisation 
    CMD  = 'python SPIF.py -ini E1.ini -iter 1 -method 0 -preopt 1'
    CMD += ' -mcmc 1 -burning 5000 -samples 2000 -thin 20 -fullsave 1'
    with open('E1.sh', 'w') as fp: fp.write(CMD+'\n') # save a copy of the command line
    os.system(CMD)
    #  without "-fullsave 1", one saves the mean and std values into E1_mcmc.fits
    #  but also the individual MCMC samples into E1_mcmc_samples.fits.
    #--------------------------------------------------------------------------------
    # plot parameter errors as 2d images
    figure(1, figsize=(10, 2.5))
    subplots_adjust(left=0.1, right=0.94, bottom=0.15, top=0.92, wspace=0.4)
    E = pyfits.open('E1_mcmc.fits') #  data [2*NP, NY, NX] = mean and stdev values over MCMC samples
    L = [r'$\sigma(T)$', r'$\sigma(V)$', r'$\sigma(FWHM)$'] 
    for i in range(3):
        ax = subplot(1,3,1+i)
        imshow(E[0].data[2*i+1,:,:])  # std values
        title(L[i])
        colorbar()
    savefig('E3a.png')
    savefig('E3a.pdf', bbox_inches='tight')
    # plot MCMC chains for selected spectra
    C = pyfits.open('E1_mcmc_samples.fits')  #  [SAMPLES, NP, NY, NX]
    figure(2, figsize=(9, 7))
    subplots_adjust(left=0.06, right=0.97, bottom=0.06, top=0.96, wspace=0.0, hspace=0.0)
    for j in range(10):
        jj = int(NY-1-j*(NY-1.0)/9.5)
        for i in range(10):
            ii = int(i*(NX-1.0)/9.5)
            ax = subplot(10, 10, 1+i+10*j)
            if (i>0): setp(ax, yticklabels=[])
            if (j<9): setp(ax, xticklabels=[])
            for ipar in range(3): # for the three free parameters
                plot(C[0].data[:, ipar, jj, ii], color=['k', 'b', 'r'][ipar], lw=0.8)
    savefig('E3b.png')                
    savefig('E3b.pdf', bbox_inches='tight')
    if (0): 
        # there was one spectrum where MCMC has some trouble
        j0, i0 = 55, 16
        figure(3)
        subplot(121)
        for ipar in range(3):
            plot(C[0].data[:, ipar, j0, i0], color=['k', 'b', 'r'][ipar])
        subplot(122)
        F = pyfits.open('E1.fits')
        plot(V, F[0].data[:, j0, i0], 'k-', ds='steps')
        P = pyfits.open('E1_res.fits')
        p = P[0].data[:, j0, i0]
        y = Gauss(p, V)
        plot(V, y, 'r-')
        # that was because the ML solution was very far from the correct,
        # which in turn was due to wrong initial values... fwhm not being finite
        # This has been since corrected, but similar types of problems might be fixed by 
        #     (1) better initial values for the chi2 fit (as was the case here)
        #     (2) larger -iter ...  to ensure optimal ML solution as the starting point
        #     (3) some penalties for unphysical parameter values
        #     (4) and/or a longer burnin phase
    SHOW()
    
    
    


if (4 in CASES): # two Gaussians in a single spectrum, each fit repeated 10 times
    p0 = asarray([1.0, -0.5, 1.5,  0.5, 1.0, 1.2], float32)   # true parameters of the Gaussians
    y0 = Gauss(p0, V) + Gauss(p0[3:], V)                        # template noiseless profile
    ymax = max(y0)        # error set relative to peak value ymax
    Y1 = zeros((M, N), float32)
    np.random.seed(123)   # for reproducibility
    for k in range(N): Y1[:,k] = y0 + NOISE[k]*ymax*randn(M) # noisy spectra
    # Write observations and their error estimates to FITS files
    Y1.shape = (M, NY, NX)
    F        = make_fits(Y1)
    F.writeto('E4.fits', overwrite=True)
    # errors = 2D FITS images
    err      = asarray(NOISE*ymax, float32).reshape(NY, NX)
    dF       = make_fits(err)
    dF.writeto('dE4.fits', overwrite=True)
    #--------------------------------------------------------------------------------
    # write the INI file for this fit
    INI = """
    fits1  = E4.fits
    dfits1 = dE4.fits
    y1     = GAUSS(v1, x[0], x[1], x[2])   +   GAUSS(v1, x[3], x[4], x[5])
    init   = y1:tmax  y1:vmax  1.0    y1:tmax*0.8  y1:vmax+1.0  1.0
    prefix = E4
    """
    with open('E4.ini', 'w') as fp: fp.write(INI)
    #--------------------------------------------------------------------------------
    # run SPIF
    CMD = 'python SPIF.py -ini E4.ini -iter 10 -method 0'
    with open('E4.sh', 'w') as fp: fp.write(CMD+'\n') # save a copy of the command line
    os.system(CMD)
    #--------------------------------------------------------------------------------
    # plot 10x10 observed spectra with the fit
    P = pyfits.open('E4_res.fits')  # fitted parameters, P[NP, NY, NX]
    figure(1, figsize=(9,7))
    subplots_adjust(left=0.1, right=0.94, bottom=0.12, top=0.95, wspace=0, hspace=0)
    for j in range(10):
        jj = int(NY-1-j*(NY-1.0)/9.5)
        for i in range(10):
            ii = int(i*(NX-1.0)/9.5)
            ax = subplot(10, 10, 1+i+10*j)
            if (i>0): setp(ax, yticklabels=[])
            if (j<9): setp(ax, xticklabels=[])
            plot(V, F[0].data[:,jj,ii], 'k-', ds='steps', lw=0.8) # observation
            p = P[0].data[:,jj,ii]  # parameter vector for the spectrum (j,i) = [T, v, fwhm, chi2]
            plot(V, Gauss(p, V), 'r-')
            plot(V, Gauss(p[3:], V), 'b-')
    savefig('E4.png')
    savefig('E4.pdf', bbox_inches='tight')
    SHOW()


    
if (5 in CASES): # two Gaussians in a single spectrum, each fit repeated 10 times
    # this one uses penalty functions, reusing the E4.fits and dE4.fits files
    #--------------------------------------------------------------------------------
    # write the INI file for this fit
    INI = """
    fits1   = E4.fits
    dfits1  = dE4.fits
    y1      = GAUSS(v1, x[0], x[1], x[2])   +   GAUSS(v1, x[3], x[4], x[5])
    init    = y1:tmax  y1:vmax  1.0    y1:tmax*0.8  y1:vmax+1.0  1.0
    penalty = POS(x[0], 0.01) + POS(x[3], 0.01)
    ## prior   = NORMAL(x[2]-1.5, 2.0) + NORMAL(x[4]-1.5, 2.0)
    prefix = E5
    """
    with open('E5.ini', 'w') as fp: fp.write(INI)
    #--------------------------------------------------------------------------------
    # run SPIF
    CMD = 'python SPIF.py -ini E5.ini -iter 10 -method 0'
    with open('E5.sh', 'w') as fp: fp.write(CMD+'\n') # save a copy of the command line
    os.system(CMD)
    #--------------------------------------------------------------------------------
    # plot 10x10 observed spectra with the fit
    P = pyfits.open('E5_res.fits')  # fitted parameters, P[NP, NY, NX]
    figure(1, figsize=(9,7))
    subplots_adjust(left=0.1, right=0.94, bottom=0.12, top=0.95, wspace=0, hspace=0)
    F = pyfits.open('E4.fits')
    for j in range(10):
        jj = int(NY-1-j*(NY-1.0)/9.5)
        for i in range(10):
            ii = int(i*(NX-1.0)/9.5)
            ax = subplot(10, 10, 1+i+10*j)
            if (i>0): setp(ax, yticklabels=[])
            if (j<9): setp(ax, xticklabels=[])
            plot(V, F[0].data[:,jj,ii], 'k-', ds='steps', lw=0.8) # observation
            p = P[0].data[:,jj,ii]  # parameter vector for the spectrum (j,i) = [T, v, fwhm, chi2]
            plot(V, Gauss(p, V), 'r-')
            plot(V, Gauss(p[3:], V), 'b-')
    savefig('E5.png')
    savefig('E5.pdf', bbox_inches='tight')
    SHOW()
    


if (6 in CASES): # two sets of spectra, one Gaussians in each
    p1 = asarray([ 1.0, +0.5, 1.5 ], float32)   # true parameters of the Gaussians "y1"
    p2 = asarray([ 0.5, +0.5, 1.2 ], float32)   # true parameters of the Gaussians "y2"
    y1 = Gauss(p1, V)
    y2 = Gauss(p2, V)
    Y1 = zeros((M, N), float32)
    np.random.seed(123)   # for reproducibility
    for k in range(N): Y1[:,k] = y1 + NOISE[k]*randn(M) # noisy spectra
    Y1.shape = (M, NY, NX)   
    F1       = make_fits(Y1)                  # first set of spectra
    F1.writeto('E61.fits', overwrite=True)
    err      = asarray(NOISE, float32).reshape(NY, NX)
    dF1      = make_fits(err)                 # error estimates
    dF1.writeto('dE61.fits', overwrite=True)
    #==========
    Y2 = zeros((M, N), float32)           
    for k in range(N): Y2[:,k] = y2 + NOISE[k]*randn(M) # noisy spectra
    Y2.shape = (M, NY, NX)  
    F2       = make_fits(Y2)                  # second set of spectra
    F2.writeto('E62.fits', overwrite=True)
    err      = asarray(NOISE, float32).reshape(NY, NX)
    dF2      = make_fits(err)                 # error estimates
    dF2.writeto('dE62.fits', overwrite=True)    
    #--------------------------------------------------------------------------------
    # write the INI file for this fit
    INI = """
    fits1   =   E61.fits
    dfits1  =  dE61.fits
    y1      =  GAUSS(v1, x[0], x[1], x[2]) 
    fits2   =   E62.fits
    dfits2  =  dE62.fits
    y2      =  GAUSS(v2, x[3], x[4], x[5])
    init    =  y1:tmax*2.5  y1:vmax  1.0    y2:tmax*1.5  y2:vmax  1.0
    # instead of using x[1] as the velocity for both, we include a weaker link via a prior
    prior   =  NORMAL(x[1]-x[4], 1.0)
    prefix  =  E6
    """
    with open('E6.ini', 'w') as fp: fp.write(INI)
    #--------------------------------------------------------------------------------
    # run SPIF
    CMD = 'python SPIF.py -ini E6.ini -iter 10 -method 1'
    with open('E6.sh', 'w') as fp: fp.write(CMD+'\n') # save a copy of the command line
    os.system(CMD)
    #--------------------------------------------------------------------------------
    # plot 10x10 observed spectra with the fit, fits1 and fits2 separately
    P = pyfits.open('E6_res.fits')  # fitted parameters, P[NP, NY, NX]
    # first set of spectra
    figure(1, figsize=(9,7))
    subplots_adjust(left=0.1, right=0.94, bottom=0.12, top=0.95, wspace=0, hspace=0)
    for j in range(10):
        jj = int(NY-1-j*(NY-1.0)/9.5)
        for i in range(10):
            ii = int(i*(NX-1.0)/9.5)
            ax = subplot(10, 10, 1+i+10*j)
            if (i>0): setp(ax, yticklabels=[])
            if (j<9): setp(ax, xticklabels=[])
            plot(V, F1[0].data[:,jj,ii], 'k-', ds='steps', lw=0.8) # observation
            p = P[0].data[0:3,jj,ii]  # parameter vector => p[0:3] for first set of spectra
            plot(V, Gauss(p, V), 'r-')
    savefig('E6a.png')
    savefig('E6a.pdf', bbox_inches='tight')            
    # second set of spectra
    figure(2, figsize=(9,7))
    subplots_adjust(left=0.1, right=0.94, bottom=0.12, top=0.95, wspace=0, hspace=0)
    for j in range(10):
        jj = int(NY-1-j*(NY-1.0)/9.5)
        for i in range(10):
            ii = int(i*(NX-1.0)/9.5)
            ax = subplot(10, 10, 1+i+10*j)
            if (i>0): setp(ax, yticklabels=[])
            if (j<9): setp(ax, xticklabels=[])
            plot(V, F2[0].data[:,jj,ii], 'k-', ds='steps', lw=0.8) # observation
            p = P[0].data[3:6,jj,ii]  # parameter vector => p[3:6] for second set of spectra
            plot(V, Gauss(p, V), 'r-')
    savefig('E6b.png')
    savefig('E6b.pdf', bbox_inches='tight')
    SHOW()



if (7 in CASES): # hyperfine structure line, single velocity component
    # write file for the velocities and intensities of the hyperfine components
    s = """    93.171905e9
    5.98       0.03703704
    5.0282     0.18518519
    4.5892     0.11111111
    0.0        0.18518519
    -0.956      0.25925926
    -1.5669     0.11111111
    -8.9624     0.11111111    
    """
    with open('E7.hfs','w') as fp: fp.write(s)
    # read the hyperfine parameters
    HFK1 =  4.799243348e-11 * float(open('E7.hfs').readline().split()[0])
    d    =  loadtxt('E7.hfs', skiprows=1)
    VHF1 =  asarray(d[:,0], float32)  # velocities of hyperfine components
    IHF1 =  asarray(d[:,1], float32)  # relative opacities of the components
    NHF1 =  len(VHF1)                 # number of components
    # we need a wider band than in the Gaussian fits
    M = 120
    V = linspace(-12.0, +12.0, M)     # chosen channel velocities    
    #
    def Hfs(p, V):                    # return spectrum for parameters p, velocities v
        Tbg = 2.73
        tau = zeros(len(V))
        for j in range(NHF1):
            tau +=  p[3]*IHF1[j]*exp(-Q*((p[1]+VHF1[j]-V)/p[2])**2)
        return (Jfun(p[0], HFK1)-Jfun(Tbg, HFK1))*(1-exp(-tau))
    #
    p0   = asarray([10.0, 0.5, 1.5, 0.5], float32)   # true parameters: Tex, v, fwhm, tau
    y0   = Hfs(p0, V)     # template noiseless profile
    ymax = max(y0)        # error will be set relative to peak value ymax
    Y1   = zeros((M, N), float32)
    np.random.seed(123)   # for reproducibility
    for k in range(N):  Y1[:,k] = y0 + NOISE[k]*ymax*randn(M) # noisy spectra
    # Write observations and their error estimates to FITS files
    Y1.shape = (M, NY, NX)
    F        = make_fits(Y1, dv=V[1]-V[0])
    F.writeto('E7.fits', overwrite=True)
    # errors = 2D FITS images
    err      = asarray(NOISE*ymax, float32).reshape(NY, NX)
    dF       = make_fits(err)
    dF.writeto('dE7.fits', overwrite=True)
    #--------------------------------------------------------------------------------
    # write the INI file for this fit
    INI = """
    fits1   =  E7.fits
    dfits1  =  dE7.fits
    y1      =  HFS(v1, x[0], x[1], x[2], x[3])
    hfs1    =  E7.hfs
    init    =  7.0  0.1  1.0  1.0        #  Tex, v, fwhm, tau
    delta   =  r:0.3 a:0.5 r:0.3 r:0.3   #  dispersion for repeated fits
    prefix  =  E7
    """
    with open('E7.ini', 'w') as fp: fp.write(INI)
    #--------------------------------------------------------------------------------
    # run SPIF, ten repeats
    CMD = 'python SPIF.py -ini E7.ini -iter 10 -method 1'
    with open('E7.sh', 'w') as fp: fp.write(CMD+'\n') # save a copy of the command line
    os.system(CMD)
    #--------------------------------------------------------------------------------
    # plot 10x10 observed spectra with the fit
    P = pyfits.open('E7_res.fits')  # fitted parameters, P[NP, NY, NX]
    figure(1, figsize=(9,7))
    subplots_adjust(left=0.1, right=0.94, bottom=0.12, top=0.95, wspace=0, hspace=0)
    for j in range(10):
        jj = int(NY-1-j*(NY-1.0)/9.5)
        for i in range(10):
            ii = int(i*(NX-1.0)/9.5)
            ax = subplot(10, 10, 1+i+10*j)
            if (i>0): setp(ax, yticklabels=[])
            if (j<9): setp(ax, xticklabels=[])
            plot(V, F[0].data[:,jj,ii], 'k-', ds='steps', lw=0.8) # observation
            p  = P[0].data[:,jj,ii]  # parameter vector for the spectrum (j,i) = [T, v, fwhm, chi2]
            yp = Hfs(p, V)           # fitted model
            plot(V, yp, 'r-')
    savefig('E7.png')
    savefig('E7.pdf', bbox_inches='tight')
    SHOW()


    
if (8 in CASES): # hyperfine structure line, single velocity component, MCMC error estimates
    # reusing E7.fits, dE7.fits, E7.hfs
    # read the hyperfine parameters
    HFK1 =  4.799243348e-11 * float(open('E7.hfs').readline().split()[0])
    d    =  loadtxt('E7.hfs', skiprows=1)
    VHF1 =  asarray(d[:,0], float32)  # velocities of hyperfine components
    IHF1 =  asarray(d[:,1], float32)  # relative opacities of the components
    NHF1 =  len(VHF1)                 # number of components
    def Hfs(p, V):                    # return spectrum for parameters p, velocities v
        Tbg = 2.73
        tau = zeros(len(V))
        for j in range(NHF1):
            tau +=  p[3]*IHF1[j]*exp(-Q*((p[1]+VHF1[j]-V)/p[2])**2)
        return (Jfun(p[0], HFK1)-Jfun(Tbg, HFK1))*(1-exp(-tau))
    #--------------------------------------------------------------------------------
    # write the INI file for this fit
    INI = """
    fits1   =  E7.fits
    dfits1  =  dE7.fits
    y1      =  HFS(v1, x[0], x[1], x[2], x[3])
    hfs1    =  E7.hfs
    init    =  7.0  0.1  1.0  1.0        #  Tex, v, fwhm, tau
    delta   =  r:0.3 a:0.5 r:0.3 r:0.3   #  dispersion for repeated fits
    prefix  =  E8
    """
    with open('E8.ini', 'w') as fp: fp.write(INI)
    #--------------------------------------------------------------------------------
    # run SPIF, ML fit + MCMC
    CMD  = 'python SPIF.py -ini E8.ini -iter 10 -method 1 -preopt 1 '
    CMD += ' -mcmc 1 -burnin 4000 -samples 2000 -thin 50'
    with open('E8.sh', 'w') as fp: fp.write(CMD+'\n') # save a copy of the command line
    os.system(CMD)
    #--------------------------------------------------------------------------------
    # plot 10x10 observed spectra with the fit
    P = pyfits.open('E8_mcmc.fits')  # fitted parameters, P[2*NP, NY, NX]
    # take absolute value of FWHM (optimisation considers FWHM==-FWHM ...)
    P[0].data[4,:,:] = abs(P[0].data[4,:,:])
    figure(1, figsize=(9.9, 4))
    subplots_adjust(left=0.06, right=0.96, bottom=0.09, top=0.94, wspace=0.3, hspace=0.3)
    L = [ r'$ T_{\rm ex}$', r'$ v$', r'$ FWHM$', r'$ \tau$']
    for ipar in range(4): # one frame per parameter
        subplot(2,4,1+ipar)
        imshow(P[0].data[2*ipar,:,:], cmap=cm.gist_stern)      # mean over MCMC samples
        title(L[ipar])
        colorbar()
        subplot(2,4,5+ipar)
        imshow(P[0].data[2*ipar+1,:,:], cmap=cm.gist_stern)    # std of MCMC samples
        title(L[ipar].replace('$ ', '$\sigma-'))
        colorbar()
    savefig('E8.png')
    savefig('E8.pdf', bbox_inches='tight')
    SHOW()

    
    
if (9 in CASES): # hyperfine structure line, single velocity component, MCMC error estimates with priors
    # reusing E7.fits, dE7.fits, E7.hfs
    # read the hyperfine parameters
    HFK1 =  4.799243348e-11 * float(open('E7.hfs').readline().split()[0])
    d    =  loadtxt('E7.hfs', skiprows=1)
    VHF1 =  asarray(d[:,0], float32)  # velocities of hyperfine components
    IHF1 =  asarray(d[:,1], float32)  # relative opacities of the components
    NHF1 =  len(VHF1)                 # number of components
    def Hfs(p, V):                    # return spectrum for parameters p, velocities v
        Tbg = 2.73
        tau = zeros(len(V))
        for j in range(NHF1):
            tau +=  p[3]*IHF1[j]*exp(-Q*((p[1]+VHF1[j]-V)/p[2])**2)
        return (Jfun(p[0], HFK1)-Jfun(Tbg, HFK1))*(1-exp(-tau))
    #--------------------------------------------------------------------------------
    # write the INI file for this fit
    INI = """
    fits1   =  E7.fits
    dfits1  =  dE7.fits
    y1      =  HFS(v1, x[0], x[1], x[2], x[3])
    hfs1    =  E7.hfs
    init    =  7.0  0.1  1.0  1.0        #  Tex, v, fwhm, tau
    delta   =  r:0.3 a:0.5 r:0.3 r:0.3   #  dispersion for repeated fits
    prior   =  NORMAL(x[0]-10.0, 3.0)
    prefix  =  E9
    """
    with open('E9.ini', 'w') as fp: fp.write(INI)
    #--------------------------------------------------------------------------------
    # run SPIF, ML fit + MCMC
    CMD  = 'python SPIF.py -ini E9.ini -iter 10 -method 1 -preopt 1 '
    CMD += ' -mcmc 1 -burnin 4000 -samples 2000 -thin 50'
    with open('E9.sh', 'w') as fp: fp.write(CMD+'\n') # save a copy of the command line
    os.system(CMD)
    #--------------------------------------------------------------------------------
    # plot 10x10 observed spectra with the fit
    P = pyfits.open('E9_mcmc.fits')  # fitted parameters, P[2*NP, NY, NX]
    # take absolute value of FWHM (optimisation considers FWHM==-FWHM ...)
    P[0].data[4,:,:] = abs(P[0].data[4,:,:])
    figure(1, figsize=(9.9, 4))
    subplots_adjust(left=0.06, right=0.96, bottom=0.09, top=0.94, wspace=0.3, hspace=0.3)
    L = [ r'$ T_{\rm ex}$', r'$ v$', r'$ FWHM$', r'$ \tau$']
    for ipar in range(4): # one frame per parameter
        subplot(2,4,1+ipar)
        imshow(P[0].data[2*ipar,:,:], cmap=cm.gist_stern)      # mean over MCMC samples
        title(L[ipar])
        colorbar()
        subplot(2,4,5+ipar)
        imshow(P[0].data[2*ipar+1,:,:], cmap=cm.gist_stern)    # std of MCMC samples
        title(L[ipar].replace('$ ', '$\sigma-'))
        colorbar()
    savefig('E9.png')
    savefig('E9.pdf', bbox_inches='tight')
    SHOW()
    
