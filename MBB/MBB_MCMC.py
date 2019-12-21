
# we must already have ISM_DIRECTORY defined and included in sys.path
from    ISM.Defs import *
from    ISM.FITS.FITS import CopyFits
import  threading



def MBB_fit_CL(F, S, dS, FIXED_BETA=-1.0, TMIN=5.0, TMAX=40.0, BMIN=0.4, BMAX=4.5,
               FILTERS=[], CCDATA=[], get_CC=False, TOL=0.0001, GPU=0, platforms=[0,1,2,3,4,5]):
    """
    Do normal chi2 fitting to estimate intensity, temperature, spectral index.
    Usage:
        I250, T, B  =  MBB_fit_CL(F, S, dS, GPU=False, FIXED_BETA=-1.0, TMIN=5.0, TMAX=40.0, BMIN=0.4, BMAX=4.5, FILTERS=[])
    Input:        
        F          =   vector of frequencies [Hz]
        S          =   observed brightness, S[Npix, Nfreq]
        dS         =   error estimates of S values
        FIXED_BETA =   if >0, keep spectral index fixed beta==FIXED_BETA
        TMIN, TMAX =   limits of acceptable temperature values
        BMIN, BMAX =   limits of acceptable spectral index values
        FILTERS    =   optional array with filenames for the response curves
        CCDATA     =   previously calculated colour correction factors,
                       if given, these are used instead of recalculating from FILTERS
                       Note -- one must still give FILTERS if colour corrections are required
        get_CC     =   if True, only CCDATA calculated and returned
        TOL        =    tolerance, default is 0.0001
        GPU        =   if True, use GPU, default is False
        platforms  =   possible OpenCL platforms, default [0,1,2,3,4,5]
    Return:
        I, T, B    =   vectors of 250um intensity, colour temperature, opacity spectral index
        If get_CC=True, return (C, None, None) where CC s an array of colour correction factors, 
                       to be reused as long as bands and parameter limits remain the same
                       
    Note:
        Input data in S must correspond to same spatial resolution for each frequency.
    Todo:
        Add option for colour correction on the fly:
            host calculates a grid over [TMIN,TMAX], [BMIN,BMAX] and
            kernel interpolates from the 2d array
    """    
    print("MBB_fit_CL NPIX=%d, NF=%d" % (S.shape[0], S.shape[1]))
    S, dS      = asarray(S, float32), asarray(dS, float32)
    N          = S.shape[0]
    NF         = S.shape[1]
    if (NF!=len(F)):
        print('MBB_fit_CL: %d frequencies but %d maps ???' % (len(F), NF))
        print("F  ", F)
        print("S  ", S.shape)
        print("dS ", dS.shape)
        sys.exit()
    # Pre-calculate colour corrections for [TMIN, TMAX], [BMIN, BMAX]
    NCC = 200
    if (len(FILTERS)>0):
        if (len(CCDATA)>0): # colour correction factors given
            CCDATA = asarray(CCDATA, float32)
        else:               # recalculate colour corrections
            # beta runs faster
            CCDATA = zeros(NCC*NCC*NF, float32)
            t      = linspace(TMIN, TMAX, NCC)
            b      = linspace(BMIN, BMAX, NCC)
            b, t   = meshgrid(b.copy(), t.copy())
            b, t   = asarray(ravel(b), float32), asarray(ravel(t), float32)
            for i in range(NF):
                CC   = asarray(CalculateColourCorrection(FILTERS[i], F[i], b, t), float32)
                # CC is now NCC x NCC values for T=TMIN...TMAX, B=BMIN...BMAX, beta running faster
                CCDATA[(i*NCC*NCC):((i+1)*NCC*NCC)] = CC
    else:   # no FILTERS == no colour corrections
        NCC     =  1
        CCDATA  =  ones(NCC*NCC*NF, float32)
    if (get_CC):
        return CCDATA, None, None
    # OpenCL initialisations
    t00         =  time.time()
    platform, device, context, queue, mf = InitCL(GPU, platforms=platforms)
    LOCAL       =  [ 8, 64 ][GPU>0]
    ###
    # Initial step size 10%, after each refinement smaller by a factor of K,
    # Convergence good if K < 0.001 ?
    KK         =  0.6
    KMIN       =  TOL
    OPT        = \
    " -D N=%d -D NF=%d -D FIXED_BETA=%d -D TMIN=%.3ff -D TMAX=%.3ff -D BMIN=%.3ff -D BMAX=%.3ff -D KK=%.6ff -D KMIN=%.6ff -D NCC=%d -D DO_CC=%d" % \
    (N, NF, FIXED_BETA, TMIN, TMAX, BMIN, BMAX, KK, KMIN, NCC, len(FILTERS)>0)
    source     = open(ISM_DIRECTORY+"/ISM/MBB/kernel_fit_MBB.c").read()
    program    = cl.Program(context, source).build(OPT)
    # Allocate buffers
    F_buf      = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=F)
    S_buf      = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    dS_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dS)
    CC_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=CCDATA)
    # Initial values and results of the computation
    I_buf      = cl.Buffer(context, mf.READ_WRITE, 4*N)
    T_buf      = cl.Buffer(context, mf.READ_WRITE, 4*N)
    B_buf      = cl.Buffer(context, mf.READ_WRITE, 4*N)
    # Figure out suitable initial values
    ifreq      = argmin(abs(F-um2f(250.0)))
    T          = asarray(15.0*ones(N), float32)
    B          = asarray( 1.8*ones(N), float32)
    if (FIXED_BETA>0.0): 
        B[:] = FIXED_BETA
    I          = asarray(S[:,ifreq]*ModifiedBlackbody_250(um2f(250.0),T,B)/ModifiedBlackbody_250(F[ifreq],T,B), float32)
    cl.enqueue_copy(queue, I_buf, I)
    cl.enqueue_copy(queue, T_buf, T)
    cl.enqueue_copy(queue, B_buf, B)    
    # Set up the kernel with parameters
    FitMBB   = program.FitMBB    
    GLOBAL   = (int(N/64)+1)*64       # one work item per source           
    #                             F,    CC,   S,    dS,   I,    T,    B    
    FitMBB.set_scalar_arg_dtypes([None, None, None, None, None, None, None])
    FitMBB(queue, [GLOBAL,], [LOCAL,], F_buf, CC_buf, S_buf, dS_buf, I_buf, T_buf, B_buf)
    cl.enqueue_copy(queue, I, I_buf)
    cl.enqueue_copy(queue, T, T_buf)
    cl.enqueue_copy(queue, B, B_buf)
    #
    print('MBB_fit_CL  %.2f seconds' % (time.time()-t00))
    return I, T, B
    



def MBB_fit_CL_FITS(FREQ, FITS, dFITS, TMIN=7.0, TMAX=40.0, BMIN=0.5, BMAX=3.5,
                    FIXED_BETA=-1.6, FILTERS=[], CCDATA=[], get_CC=False, TOL=0.0001, GPU=0, platforms=[0,1,2,3,4,5]):
    """
    Wrapper for MBB_fit_CL, using FITS images as the input instead of data vectors.
    Also returns I(250um), T, and beta as FITS images.
    Usage:
        FI, FT, FB = MBB_fit_CL_FITS(FREQ, FITS, dFITS, GPU=0, 
                                     TMIN=7.0, TMAX=40.0, BMIN=0.5, BMAX=3.5, 
                                     FIXED_BETA=-1.6, FILTERS=[], CCDATA=[], get_CC=False)
    Input:        
        FREQ       =   vector of frequencies [Hz]
        FITS       =   list of FITS images
        dFITS      =   list of FITS images with error estimates
        FIXED_BETA =   if >0, keep spectral index fixed beta==FIXED_BETA
        TMIN, TMAX =   limits of acceptable temperature values
        BMIN, BMAX =   limits of acceptable spectral index values
        FILTERS    =   optional array with filenames for the response curves
        CCDATA     =   previously calculated colour correction factors,
                       if given, these are used instead of recalculating from FILTERS
                       Note -- one must still give FILTERS if colour corrections are required
        get_CC     =   if True, only CCDATA calculated => the returned values are (CC, None, None) !
        GPU        =   if True, use GPU (default=False)
        platforms  =   possuible OpenCL platforms, default is [0, 1, 2, 3, 4, 5]
    Return:
        FI, FT, FB  =   pyfits images of 250um intensity, colour temperature, and opacity spectral index
        If get_CC==True, returned values are
        CC, None, None  =  CC being an array of colour correction factors, 
                           can be reused as long as bands and parameter limits remain the same        
    """
    n, m = FITS[0][0].data.shape
    nf   = len(FREQ)
    S    = zeros((n*m, nf), float32)
    dS   = 0.0*S
    for i in range(nf):
        S[:,i]  = ravel(FITS[i][0].data)
        dS[:,i] = ravel(dFITS[i][0].data)
    ###
    I, T, B = MBB_fit_CL(FREQ, S, dS, TMIN=TMIN, TMAX=TMAX, BMIN=BMIN, BMAX=BMAX, 
                         FIXED_BETA=FIXED_BETA, FILTERS=FILTERS, CCDATA=CCDATA, get_CC=get_CC, TOL=TOL,
                         GPU=GPU, platforms=platforms)
    if (get_CC): return I, None, None   # I = colour correction factors, return immediately ...
    # ... otherwise (I, T, B) are the fitted values for each pixel
    FI = CopyFits(FITS[0])
    FT = CopyFits(FITS[0])
    FB = CopyFits(FITS[0])
    FI[0].data = I.reshape(n,m)
    FT[0].data = T.reshape(n,m)
    FB[0].data = B.reshape(n,m)
    return FI, FT, FB
    





###########################################################################################################
###########################################################################################################



def RunMCMC(F, S, dS, GPU=False, BURNIN=1000, SAMPLES=10000, THIN=20, WIN=200, \
            USE_HD=True, FIXED_BETA=-1.0, RETURN_SAMPLES=False, ML=False,
            TMIN=5.0, TMAX=40.0, BMIN=0.4, BMAX=4.5, COV=[], METHOD=0, INI=[], SUMMARY=0,
            MP=1):
    """
    Fit modified blackbody curves to provided fluxes using MCMC.
    Input:
        F          =  vector of band frequencies, F[NF], [Hz]
        S          =  fluxes, S[N, NF] where N is the number of sources
        dS         =  flux uncertainties, dS[N, NF]
        GPU        =  use gpu (default False)
        BURNIN     =  length of burn-in phase [samples] (default 1000)
        SAMPLES    =  number of returned samples (default 10000)
                      note - total number of MCMC steps is BURNIN+THIN*SAMPLES
        THIN       =  every THIN:th sample returned
        WIN        =  length of the window (in MCMC steps) used to estimate
                      parameter covariance matrix (for step generation)
        USE_HD     =  whether to use HD instead of MWC64X random number
                      generator (default True)
        FIXED_BETA =  if >0, use that as a fixed beta value (default -1.0)
        RETURN_SAMPLES = if true, return extracted samples (see below)
        ML         =  if True, use maximum likelihood estimates as starting position
        TMIN, TMAX =  hard limits for temperature
        BMIN, BMAX =  hard limits for spectral index
        COV        =  optionally covariance matrix for errors
                      if S = S[N, NF],  COV = COV[N, NF, NF]
        METHOD     =  0 - default Metropolis
                      1 - update covariance matrix for the proposition distribution (Haario 2001)
                      2 - Hamiltonian MCMC
                      3 - RAM (Robust Adaptive Metropolis, Vihola 2011)
        INI        =  optional array of initial values INI[N,3] = { I, T, beta }
        SUMMARY    =  if >0, calculate only summary statistics { mean, stdev }
                      for  { I250, T, beta, tau250 }
        MP         =  if >0 (default=1), use multiprocessing in the postprocessing of MCMC samples
                      (calculation of percentiles)
    Output with SUMMARY==0:
        MI       =  MI[N, 5], 250um intensity { mean, stdev, Q1, Q2, Q3 }
        MT       =  MT[N, 5], temperature     { mean, stdev, Q1, Q2, Q3 }
        MB       =  MB[N, 5], beta values     { mean, stdev, Q1, Q2, Q3 }
        Mtau     =  MB[N, 5], tau250 values   { mean, stdev, Q1, Q2, Q3 }
        RES      =  returned only if RETURN_SAMPLES==True (and SUMMARY==0),
                    RES[N, SAMPLES, 3] = samples of {I, T, beta}
                    *** NOTE *** this is a memory limitation,
                    16000 pixels/sources and 20000 samples => close to 4GB array
        X        =  returned only if ML=True, X[N,3] = [ I, T, beta ] for the
                    chi2 minimum position (also used as initial values
                    for MCMC!)
    Output with SUMMARY>0:
        MI       =  MI[N, 5], 250um intensity { mean, stdev }
        MT       =  MT[N, 5], temperature     { mean, stdev }
        MB       =  MB[N, 5], beta values     { mean, stdev }    
        Mtau     =  MB[N, 5], tau250 values   { mean, stdev }
        X        =  ML results returned only if ML=True
    Note:
        Optical depth values assume that the inputs are in units of MJy/sr.
    """
    print('-'*80)    
    print("RunMCMC NPIX=%d, NF=%d, ML=%d" % (S.shape[0], S.shape[1], ML))
    NF         = len(F)       # number of frequencies
    N          = S.shape[0]   # number of sources
    BINS       = 100          # bins for (T,beta) proposition distribution
    # Calculate covariance matrices on the host side
    USE_COV    = (len(COV)>0)
    if (USE_COV):  # calculate the inverses of the covariance matrices
        print('>>>>>> USING CONVARIANCE MATRICES IN RunMCMC <<<<<<')
        COVI   = zeros((N, NF, NF), float32)
        for i in range(N): # over sources
            COVI[i,:,:] = inv(COV[i,:,:])
    else:
        COVI   = zeros(2, float32) # dummy
    # Set initial values
    if (len(INI)<1):  # user did not provide initial values
        # print("Default initial values for MCMC !!")
        INI        = zeros((N,3), float32)
        INI[:,1]   = 15.0
        if (FIXED_BETA>0.0):
            INI[:,2] = FIXED_BETA
        else:
            INI[:,2] = 1.8            
        # Estimate 250um intensity based on the closest frequency point X -->  S250 = SX * MBB(F250) / MBB(FX)
        ifreq =  argmin(abs(um2f(250.0)-F))
        INI[:,0] = S[:,ifreq] * ModifiedBlackbody_250(um2f(250.0),INI[:,1],INI[:,2]) / ModifiedBlackbody_250(F[ifreq], INI[:,1], INI[:,2])
        
    X = None
    if (ML): # calculate chi2 minimum solution
        # print("   ... RunMCMC ML\n")
        t0 = time.time()
        X  = zeros((N,3), float32)
        cov = []
        um = f2um(F)
        ###
        I0, T0, B0 = MBB_fit_CL(F, S, dS, GPU=GPU, FIXED_BETA=FIXED_BETA,
                                TMIN=TMIN, TMAX=TMAX, BMIN=BMIN, BMAX=BMAX,
                                FILTERS=[], CCDATA=[], get_CC=False)
        # ML estimates will override the initial values
        X[:,0]   =  I0
        X[:,1]   =  T0
        X[:,2]   =  B0
        INI[:,:] = X[:,:]
        # print('   ... RunMCMC ML %.2f seconds' % (time.time()-t0))
        
    S, dS, INI = ravel(S), ravel(dS), ravel(INI)  #  kernel uses 1D vectors
    
    # OpenCL initialisations
    t0         = time.time()
    platform, device, context, queue,  mf = InitCL(GPU)
    LOCAL       = [ 8, 32 ][GPU>0]
    ADD        = "-cl-fast-relaxed-math"
    ADD        = ""
    OPT        = \
    " -D N=%d -D NF=%d -D SAMPLES=%d -D THIN=%d -D BURNIN=%d -D FIXED_BETA=%d -D WIN=%d -D USE_HD=%d \
    -D TMIN=%.3ff -D TMAX=%.3ff -D BMIN=%.3ff -D BMAX=%.3ff -I ./ -D METHOD=%d -D USE_COV=%d %s -D SUMMARY=%d" % \
    (N, NF, SAMPLES, THIN, BURNIN, (FIXED_BETA>0.0), WIN, USE_HD, TMIN, TMAX, BMIN, BMAX, 
    METHOD, USE_COV, ADD, SUMMARY)  
    source     = open(ISM_DIRECTORY+"/ISM/MBB/kernel_MBB_MCMC.c").read()
    program    = cl.Program(context, source).build(OPT)
    # Allocate buffers
    F_buf      = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=F)
    S_buf      = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    dS_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dS)
    INI_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=INI)
    # RES ==  [I,T,beta]*SAMPLES*N
    # if SUMMARY>0,  RES=RES[N,8] and contain { mean, stdev } estimated on the devie.
    if (SUMMARY):
        # RES =  mean(I), std(I),  mean(T), std(T),  mean(B), std(B),  mean(tau250), std(tau250)
        RES_buf  = cl.Buffer(context, mf.READ_WRITE, 4 * 8*N)
        RES      = zeros(8*N, float32)  #  [N, 6]
        print("  ... RES 8 x %d entries = %.2f MB" % (N, (32*N)/(1024.0*1024.0) ))
    else:
        print("XXX  N=%d, SAMPLES=%d" % (N, SAMPLES))
        RES_buf  = cl.Buffer(context, mf.READ_WRITE, 4 * 3*N*SAMPLES)
        RES      = zeros(3*N*SAMPLES, float32)  #  [N, SAMPLES, 3]
        print("  ... RES 3 x %d pix x %d samples = %.2f MB" % \
        (N, SAMPLES, (4.0*3.0*N*SAMPLES)/(1024.0*1024.0) ))
    COVI_buf   = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=COVI)
    # Set up the kernel with parameters
    if (METHOD in [0, 1]):      # Metropolis with or without proposition covariances
        MCMC  = program.MCMC
    elif (METHOD==2):           # Hamiltonian MCMC
        MCMC  = program.HMCMC
    elif (METHOD==3):           # RAM
        MCMC  = program.RAM
        
    GLOBAL   = (int(N/LOCAL)+1)*LOCAL       # one work item per source

    MCMC.set_scalar_arg_dtypes([np.float32, None, None, None, None, None, None])
    MCMC(queue, [GLOBAL,], [LOCAL,], float(rand()), F_buf, S_buf, dS_buf, INI_buf, RES_buf, COVI_buf)

    # Read the samples --- if SUMMARY=True, these are the {mean, stdev} for each pixel, for I, T, and B
    cl.enqueue_copy(queue, RES, RES_buf)
        
    if (SUMMARY):
        RES.shape = (N, 8)
        if (ML):
            return RES[:,0:2], RES[:,2:4], RES[:,4:6], RES[:,6:8],  X
        else:
            return RES[:,0:2], RES[:,2:4], RES[:,4:6], RES[:,6:8]

    # If we come here, RES contains MCMC samples
    # RES fasters running indices (1) I,T, B, (2) sample, (3) source
    RES.shape = (N, SAMPLES, 3)

    print('METHOD %d took %.3f seconds ...' % (METHOD, time.time()-t0))
    
    t0 = time.time()
    
    # Calculate mean and std -- I, T, beta, tau250
    MI         = zeros((N, 5), float32)  #  [mean, std, Q1, Q2, Q3]
    MT         = zeros((N, 5), float32)
    MB         = zeros((N, 5), float32)
    MO         = zeros((N, 5), float32)  # 250um optical depth
    # RES = RES[npix,samples, ifield],   ifield=I, T, B
    tmp        = mean(RES, axis=1)       # [N, 3]
    MI[:,0]    = tmp[:,0]
    MT[:,0]    = tmp[:,1]
    MB[:,0]    = tmp[:,2]    
    tmp        = std(RES, axis=1)
    MI[:,1]    = tmp[:,0]
    MT[:,1]    = tmp[:,1]
    MB[:,1]    = tmp[:,2]
    
    # optical depth samples -- all pixels, all samples  --- tmp = tmp[npix, samples]
    # assumes data in units of MJy/sr !!!
    tmp        = RES[:,:,0] / (PlanckFunction(um2f(250.0), RES[:,:,0])*intensity_conversion('cgs','MJy/sr'))
    # tau[npix, samples]
    MO[:,0]    = mean(tmp, axis=1)
    MO[:,1]    = std( tmp, axis=1)

    t0 = time.time()
    if (MP==0):
        print("SERIAL")
        MI[:, 2:] = transpose(percentile(RES[:,:,0], (25.0, 50.0, 75.0), axis=1))
        MT[:, 2:] = transpose(percentile(RES[:,:,1], (25.0, 50.0, 75.0), axis=1))
        MB[:, 2:] = transpose(percentile(RES[:,:,2], (25.0, 50.0, 75.0), axis=1))
        MO[:, 2:] = transpose(percentile(tmp[:,:],   (25.0, 50.0, 75.0), axis=1))
        if (0):
            clf()
            subplot(221)
            hist(MI[:,3], bins=100, color='b', alpha=0.3, hatch='//')
            subplot(222)
            hist(MT[:,3], bins=100, color='b', alpha=0.3, hatch='//')
            subplot(223)
            hist(MB[:,3], bins=100, color='b', alpha=0.3, hatch='//')
            subplot(224)
            hist(MO[:,3], bins=100, color='b', alpha=0.3, hatch='//')        
    else:
        # Threaded
        print("THREADED")
        def worker(ind, A, outs):
            outs[:,2:] = transpose(percentile(A[:,:], (25.0, 50.0, 75.0), axis=1))
        ##
        t1        =  threading.Thread(target=worker, args=(0, RES[:,:,0], MI))
        t1.start()
        t2        =  threading.Thread(target=worker, args=(1, RES[:,:,1], MT))
        t2.start()
        t3        =  threading.Thread(target=worker, args=(2, RES[:,:,2], MB))
        t3.start()
        t4        =  threading.Thread(target=worker, args=(3, tmp[:,:  ], MO))
        t4.start()
        t1.join(),  t2.join(),  t3.join(),  t4.join()
        if (0):
            subplot(221)
            hist(MT[:,0], bins=100, alpha=0.4, hatch='\\')
            subplot(222)
            hist(MT[:,1], bins=100, alpha=0.4, hatch='\\')
            subplot(223)
            hist(MT[:,2], bins=100, alpha=0.4, hatch='\\')
            subplot(224)
            hist(MT[:,3], bins=100, alpha=0.4, hatch='\\')
            SHOW()
            sys.exit()
        
    
    print('===== Post analysis %.3f seconds =====' % (time.time()-t0))    
    print('-'*80)
    
    if (ML):
        if (RETURN_SAMPLES):
            return MI, MT, MB, MO, RES, X
        else:
            return MI, MT, MB, MO, X
    else:
        if (RETURN_SAMPLES):
            return MI, MT, MB, MO, RES
        else:
            return MI, MT, MB, MO


        
        
        
def RunMCMC_FITS(FREQ, FITS, dFITS, GPU=False, BURNIN=1000, SAMPLES=10000, THIN=20, WIN=200, \
                 USE_HD=True, FIXED_BETA=-1.0, ML=True,
                 TMIN=5.0, TMAX=40.0, BMIN=0.4, BMAX=4.5, COV=[], METHOD=0, INI=[],
                 SUMMARY=0):
    """
    Wrapper to RunMCMC() using FITS images for inputs and outputs.
    Fit modified blackbody curves to provided fluxes using MCMC.
    Input:
        FREQ    = vector of band frequencies, FREQ[NF], [Hz]
        FITS    = list of FITS images of the brightness
        dFITS   = list of FITS images for the uncertainties
        GPU     = use gpu (default False)
        BURNIN  = length of burn-in phase [samples] (default 1000)
        SAMPLES = number of returned samples (default 10000)
                  note - total number of MCMC steps is BURNIN+THIN*SAMPLES
        THIN    = every THIN:th sample returned
        WIN     = length of the window (in MCMC steps) used to estimate
                  parameter covariance matrix (for step generation)
        USE_HD  = whether to use HD instead of MWC64X random number
                  generator (default True)
        FIXED_BETA = if >0, use that as a fixed beta value (default -1.0)
        ML      = if True, initialise with maximum likelihood estimates
        TMIN, TMAX =  hard limits for temperature
        BMIN, BMAX =  hard limits for spectral index
        COV        =  optionally covariance matrix for errors
                      if S = S[N, NF],  COV = COV[N, NF, NF]
        METHOD  =  0 - default Metropolis
                   1 - update covariance matrix for the proposition distribution (Haario 2001)
                   2 - Hamiltonian MCMC
                   3 - RAM (Robust Adaptive Metropolis, Vihola 2011)
        INI     = optional array of initial values INI[N,3] = { I, T, beta }
        SUMMARY = if >0, calculate only summary statistics (see below)
    Output with SUMMARY==0:
        FI       =  FITS image  [:,:, 5], 250um intensity { mean, stdev, Q1, Q2, Q3 }
        FT       =  FITS image  [:,:, 5], temperature     { mean, stdev, Q1, Q2, Q3 }
        FB       =  FITS image  [:,:, 5], beta values     { mean, stdev, Q1, Q2, Q3 }
        X        =  for ML=1 only, a single FITS image [:,:,3], last index ~ { I, T, beta }
    Output with SUMMARY>0:
        FI       =  FITS image  [:,:, 2], 250um intensity      { mean, stdev }
        FT       =  FITS image  [:,:, 2], temperature          { mean, stdev }
        FB       =  FITS image  [:,:, 2], beta values          { mean, stdev }
        Ftau     =  FITS image  [:,:, 2], 250um optical depth  { mean, stdev }
        X        =  for ML=1 only, a single FITS image [:,:,3], last index ~ { I, T, beta }
    Note:
        If you need to examine the raw MCMC samples, you need to use the RunMCMC() routine
        directly.
    """
    n, m = FITS[0][0].data.shape
    nf   = len(FREQ)
    ok   = ones((n,m), int32)
    # Figure out valid pixels
    for i in range(nf): # note - this bails if any of the data is missing (any band)
        ok[nonzero( (~isfinite(FITS[i][0].data)) | (FITS[i][0].data<=0.0) ) ] = 0
##    if (SPARSE):
##        tmp = zeros((n,m), int32)
##        tmp[0:-1:2,0:-1:2] = 1
##        ok[nonzero(tmp<1)] = 0
    mok2  = nonzero(ok>0)        # only these pixels worth calculating
    mok1 = nonzero(ravel(ok)>0)  # same mask but for 1d vector
    no   = len(mok2[0])
    print('mok %d,  fraction %.3f of all pixels' % (no, no/float(n*m)))
    S    = zeros((no, nf), float32)
    dS   = 0.0*S
    for i in range(nf):
        S[:,i]  =  FITS[i][0].data[mok2]
        dS[:,i] = dFITS[i][0].data[mok2]
    ###
    print('******', S.shape, SAMPLES)
    RES =  RunMCMC(FREQ, S, dS, GPU=GPU, BURNIN=BURNIN, SAMPLES=SAMPLES, THIN=THIN, WIN=WIN, \
                    USE_HD=USE_HD, FIXED_BETA=FIXED_BETA, RETURN_SAMPLES=False, ML=ML,
                    TMIN=TMIN, TMAX=TMAX, BMIN=BMIN, BMAX=BMAX, COV=COV, METHOD=METHOD, INI=INI, 
                    SUMMARY=SUMMARY)
    # SUMMARY=0, ML=0  -->    RES =  [ I, T,  B         ]
    # SUMMARY=0, ML=1  -->    RES =  [ I, T,  B, X      ], where X are the results from the chi2 fit
    # SUMMARY=1, ML=0  -->    RES =  [ I, T,  B, tau    ]
    # SUMMARY=1, ML=1  -->    RES =  [ I, T,  B, tau, X ], where X are the results from the chi2 fit
    
    # ... otherwise (I, T, B) are the fitted values for each pixel
    if (SUMMARY==0):  ncol=5  #   { mean, stdev, Q1, Q2, Q3 }
    else:             ncol=2  #   { mean, stdev }
    FI   =   CopyEmptyFits(FITS[0], data=zeros((n*m,ncol), float32))  # 250um intensity
    FT   =   CopyEmptyFits(FITS[0], data=zeros((n*m,ncol), float32))  # temperature 
    FB   =   CopyEmptyFits(FITS[0], data=zeros((n*m,ncol), float32))  # spectral index
    FI[0].data[mok1[0],:]  =  RES[0].reshape(no, ncol)
    FT[0].data[mok1[0],:]  =  RES[1].reshape(no, ncol)
    FB[0].data[mok1[0],:]  =  RES[2].reshape(no, ncol)
    FI[0].data.shape      = (n, m, ncol)
    FT[0].data.shape      = (n, m, ncol)
    FB[0].data.shape      = (n, m, ncol)
    FI.verify('fix')
    FT.verify('fix')
    FB.verify('fix')
    Ft, FML = None, None
    if (SUMMARY):
        Ft   =   CopyEmptyFits(FITS[0], data=zeros((n*m,ncol), float32))  # 250um optical depth
        Ft[0].data[mok1[0],:]  =  RES[3].reshape(no, ncol)
        Ft[0].data.shape       = (n, m, ncol)
        Ft.verify('fix')
    if (ML):
        FML                         =  CopyEmptyFits(FITS[0], data=zeros((n*m,3), float32))
        if (SUMMARY):
            FML[0].data[mok1[0],:]  =  RES[4].reshape(no,3)   # "X" 
        else:
            FML[0].data[mok1[0],:]  =  RES[3].reshape(no,3)
        FML[0].data.shape           =  (n,m,3)
        FML.verify('fix')
        if (SUMMARY):
            return FI, FT, FB, Ft, FML
        else:            
            return FI, FT, FB, FML
    if (SUMMARY):
        return FI, FT, FB, Ft
    return FI, FT, FB




def MBB_HMCMC_RAM_CL(F, S, dS, GPU=0, TMIN=5.0, TMAX=40.0, BMIN=0.4, BMAX=4.5,
                     SAMPLES=2000, BURNIN=1000, THIN=20, INI=[],HIER=1, STUDENT=0, HRAM=1, 
                     LOCAL=-8, GLOBAL=-32, platforms=[0,1,2,3,4]):
    """
    Fit a set of observations with modified blackbody function, including a 
    hierarchical part for the overall (T, beta) or (I, T, beta) distribution.
    Usage:
        GS, LS  = MBB_HMCMC_RAM_CL(F, S, dS, GPU=0, TMIN=5.0, TMAX=40.0, BMIN=0.4, BMAX=4.5, 
                                   SAMPLES=2000, BURNIN=1000, THIN=20, INI=[], NGW=10, HIER=1, STUDENT=0)
    Input:
        F          =   vector of frequencies
        S          =   S[NPIX, NFREQ] of flux density values
        dS         =   dS[NPIX, NFREQ] of flux density uncertainty estimates
        TMIN, TMAX =   hard limits for the temperatures
        BMIN, BMAX =   hard limits for the spectral index values
        SAMPLES    =   number of samples returned (after thinning)
        BURNIN     =   number of initial samples (before thinning) rejected
        THIN       =   return every THIN:th values only
        INI        =   optional initial values INI[N, 3], with columns [I(250um), T, beta];
                       if not given, start with the ML solution
        LOCAL      =   number of threads per work group (default 4 for CPU, 32 for GPU)
        GLOBAL     =   total number of threads (default 16 for CPU, 128 for GPU)
        HIER       =   0 = no hierarchical part, 
                       1 = hierarchical part for (T, beta),
                       2 = hierarchical part for (I, T, beta)
        STUDENT    =   if >0, use Student t distributions for hierarchical and the per-source
                       parts; default is the normal distribution
        HRAM       =   HRAM=0 -->  use RAM only for per-source (I, T, beta), not for hyperparameters
                       HRAM=1 -->  use RAM also for hyperparameters but in separate groups of 2-3
                                   parameters each
                       HRAM=2 -->  one RAM for per-source parameters, one for all hyperparameters together
    Return:
        GS[NWG, 5, SAMPLES], samples of [ <T>, <Beta>, dT, dBeta, rho(T, Beta) ] for the 
                             hierarchical (global) part, NWG is the number of work groups
        LS[NWG, N, 3, SAMPLES], samples of (I, T, Beta) for individual sources
    Notes:
        - Each work group has its own set of hyperparameters and calculates likelihood values for
          every source/pixel. Host receives these independent samples and the final solution can 
          be combined from the results of different work groups.
        - HIER=1 implies a set of 5 hyperparameters [ T, B, dT, dB, rho(T,B) ]  (B=beta)
          HIER=2 implies a set of 9 hyperparameters [ I, T, B, dI, dT, dB, rIT, rIB, rTB ]
    """
    if (LOCAL<0):   LOCAL   =  [4,    16][GPU>0]
    if (GLOBAL<0):  GLOBAL  =  [24,  768][GPU>0]
    NWG      =   int(GLOBAL/LOCAL)  # number of work groups
    print("NWG=%d" % NWG);
    NF       =   len(F)             # number of frequencies
    N        =   S.shape[0]         # number of sources
    np.random.seed(123456)
    if (len(INI)<1):                # Use ML estimates as the initial values
        I0, T0, B0 = MBB_fit_CL(F, S, dS, GPU=GPU, FIXED_BETA=-1, TMIN=TMIN, TMAX=TMAX, BMIN=BMIN, BMAX=BMAX)
        INI = zeros((N,3), np.float32)
        INI[:,0], INI[:,1], INI[:,2] = I0, T0, B0    
        if (1):
            INI[:,1] = 20.0 + 1.0+np.random.randn(N)
            INI[:,2] =  1.3 + 0.2*np.random.randn(N)
    if (1):
        print("  I = %10.3e +- %10.3e" % (mean(INI[:,0]), std(INI[:,0]))) ;
        print("  T = %10.3e +- %10.3e" % (mean(INI[:,1]), std(INI[:,1]))) ;
        print("  B = %10.3e +- %10.3e" % (mean(INI[:,2]), std(INI[:,2]))) ;
    # OpenCL initialisation
    platform, device, context, queue,  mf = InitCL(GPU, platforms)
    if (not(HIER in [0, 1, 2])):
        print("*** MBB_HMCMC_RAM_CL: parameter HIER must be 0, 1, or 2")
        sys.exit()
    NG         =  [ 5, 9 ][HIER>1]    # number of global parameters
    source     = open(ISM_DIRECTORY+"/ISM/MBB/kernel_MBB_HMCMC_RAM.c").read()
    OPT     =  " -D N=%d -D NF=%d -D SAMPLES=%d -D THIN=%d -D BURNIN=%d -D STUDENT=%d -D HIER=%d -D HRAM=%d \
    -D TMIN=%.3ff -D TMAX=%.3ff -D BMIN=%.3ff -D BMAX=%.3ff -I ./ -D LOCAL=%d -D NG=%d" % \
    (N, NF, SAMPLES, THIN, BURNIN, STUDENT, HIER, HRAM, TMIN, TMAX, BMIN, BMAX, LOCAL, NG)
    if (1):
        OPT += " -cl-fast-relaxed-math"
        # OPT += ' -cl-opt-disable '
    program  = cl.Program(context, source).build(OPT)
    ###
    F        =  asarray(F,  np.float32)
    S        =  asarray(S,  np.float32)
    dS       =  asarray(dS, np.float32)
    # Data for the global and local states, including their initial values
    G        =  zeros((NWG, NG),   np.float32)            #  (I, T, B, dI, ...) or (T, B, dT, ...) for each work group
    L        =  zeros((NWG, N, 3), np.float32)            #  (I, T, beta), for each work group and each pixel/source
    GS       =  zeros(NWG*NG*SAMPLES, np.float32)         # storage for hyperparameters samples
    LS       =  zeros(NWG*N*3*SAMPLES, np.float32)        # storage for the (I, T, beta) values of N sources
    # Buffers for input data
    F_buf    =  cl.Buffer(context, mf.READ_ONLY, 4*NF)
    S_buf    =  cl.Buffer(context, mf.READ_ONLY, 4*N*NF)
    dS_buf   =  cl.Buffer(context, mf.READ_ONLY, 4*N*NF) 
    cl.enqueue_copy(queue, F_buf,  F)
    cl.enqueue_copy(queue, S_buf,  ravel(S))
    cl.enqueue_copy(queue, dS_buf, ravel(dS))
    # Buffers for samples
    GS_buf   =  cl.Buffer(context, mf.WRITE_ONLY, GS.nbytes)    # global samples
    LS_buf   =  cl.Buffer(context, mf.WRITE_ONLY, LS.nbytes)    # local samples
    if (1):
        # Compress initial values --- to improve convergence ?
        for i in [0,1,2]:  # T and Beta
            mu       =  mean(INI[:,i])
            INI[:,i] = mu + 0.3*(INI[:,i]-mu)
    if (NG==9):      # 3D Student t or 3D Gaussian hierarchical part
        G[:,0]   =  mean(INI[:,0])    # I    
        G[:,1]   =  mean(INI[:,1])    # T    
        G[:,2]   =  mean(INI[:,2])    # beta 
        # Initialise the rest using the covariance matrix of the observations
        SIG      =  np.cov(transpose(INI))  # cov needs parameter vectors as rows
        G[:,3]   =  sqrt(SIG[0,0])    #  std(I)
        G[:,4]   =  sqrt(SIG[1,1])    #  std(T)
        G[:,5]   =  sqrt(SIG[2,2])    #  std(beta)
        G[:,6]   =  SIG[0,1] / (sqrt(SIG[0,0]*SIG[1,1]))  #  rho(I,T)
        G[:,7]   =  SIG[0,2] / (sqrt(SIG[0,0]*SIG[2,2]))  #  rho(I,beta)
        G[:,8]   =  SIG[1,2] / (sqrt(SIG[1,1]*SIG[2,2]))  #  rho(T,beta)
    else:           # just 2D Gaussian as the hierarchical part ... or no hierarchical level at all
        G[:,0]   =  mean(INI[:,1])    #  T  
        G[:,1]   =  mean(INI[:,2])    #  B  
        G[:,2]   =  std( INI[:,1])    # dT   will change fast, more important to have initial (T,beta) ~correct
        G[:,3]   =  std( INI[:,2])    # dB  
        G[:,4]   =  -0.7              # rho 
    #================================================================================
    for i in range(NWG):
        L[i,:,0], L[i,:,1], L[i,:,2]  =  INI[:,0], INI[:,1], INI[:,2]
    # Buffers for the current state
    G_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NWG*NG)     # global state [NWG, NG]
    L_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NWG*N*3)    # local state [NWG, N, 3]
    WRK_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*NWG*3*N)    # space for per-wg local propositions (I, T, beta)
    cl.enqueue_copy(queue, G_buf, ravel(G))
    cl.enqueue_copy(queue, L_buf, ravel(L))
    queue.finish()
    # The kernel
    MCMC    =  program.MCMC_RAM
    MCMC.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, None])
    RAM_buf = cl.Buffer(context, mf.READ_WRITE, 4*NWG*N*6)  # RAM... each work group, each source, 6 matrix elem.
    MCMC(queue, [GLOBAL,], [LOCAL,], F_buf, S_buf, dS_buf, G_buf, L_buf, GS_buf, LS_buf, WRK_buf, RAM_buf)
    queue.finish()
    # Read back the samples
    cl.enqueue_copy(queue, GS, GS_buf)    # GS[NWG, NG  , SAMPLES]
    cl.enqueue_copy(queue, LS, LS_buf)    # LS[NWG, N, 3, SAMPLES]
    queue.finish()
    GS.shape = (NWG, NG,   SAMPLES) ;
    LS.shape = (NWG, N, 3, SAMPLES) ;
    ####
    return GS, LS




def MBB_HMCMC_GIBBS_CL(F, S, dS, GPU=0, TMIN=5.0, TMAX=40.0, BMIN=0.4, BMAX=4.5, STEP=[],
                       SAMPLES=2000, BURNIN=1000, LOCAL_INI=[], GLOBAL_INI=[], HIER=1, STUDENT=0, DOUBLE=0,
                       GITER=5, LITER=50, SITER=1, platforms=[0,1,2,3,4]):
    """
    Fit a set of observations with modified blackbody function, including a 
    hierarchical part for the overall (T, beta) or (I, T, beta) distribution.
    Usage:
        GS, LS, ML = MBB_HMCMC_RAM_CL(F, S, dS, GPU=0, TMIN=5.0, TMAX=40.0, BMIN=0.4, BMAX=4.5, 
                                   SAMPLES=2000, BURNIN=1000, INI=[], NGW=10, HIER=1, STUDENT=0)
    Input:
        F          =   vector of frequencies
        S          =   S[NPIX, NFREQ] of flux density values
        dS         =   dS[NPIX, NFREQ] of flux density uncertainty estimates
        TMIN, TMAX =   hard limits for the temperatures
        BMIN, BMAX =   hard limits for the spectral index values
        SAMPLES    =   number of samples returned (after thinning)
        BURNIN     =   number of initial samples (before thinning) rejected
        LOCAL_INI  =   optional initial values LOCAL_INI[N, 3], with columns [I(250um), T, beta];
                       if not given, start with the ML solution
        GLOBAL_INI =   optional initial values for global parameters [NG]
        HIER       =   0 = no hierarchical part, 
                       1 = hierarchical part for (T, beta),
                       2 = hierarchical part for (I, T, beta)
        STUDENT    =   if >0, use Student t distributions for hierarchical and the per-source
                       parts; default is the normal distribution
        DOUBLE     =   for DOUBLE=1, use double preceision in kernel
        GITER      =   iterations over hyperparameters within the main loop
        LITER      =   iterations over per-source parameters within the main loop
        SITER      =   subiterations GITER+LITER before saving one sample
    Return:
        GS[NWG, 5, SAMPLES], samples of [ <T>, <Beta>, dT, dBeta, rho(T, Beta) ] for the 
                             hierarchical (global) part, NWG is the number of work groups
        LS[NWG, N, 3, SAMPLES], samples of (I, T, Beta) for individual sources
        GINI[NG]   =   last global parameters
        LINI[N*3]  =   last per-source parameters
        lnP0       =   lnP of the last accepted point
    Notes:
        - Separate calls for the hyperparameters and for the parameters of the individual sources.
        - HIER=1 implies a set of 5 hyperparameters [ T, B, dT, dB, rho(T,B) ]  (B=beta)
          HIER=2 implies a set of 9 hyperparameters [ I, T, B, dI, dT, dB, rIT, rIB, rTB ]
    """
    # print("MBB_HMCMC_GIBBS_CL !!")
    if (DOUBLE==1):   
        real, NB = np.float64, 8
    else:             
        real, NB = np.float32, 4
    NF       =   len(F)             # number of frequencies
    N        =   S.shape[0]         # number of sources
    if (not(HIER in [0, 1, 2])):
        print("*** MBB_HMCMC_GIBBS_CL: parameter HIER must be 0, 1, or 2")
        sys.exit()
    NG       =  [5, 5, 9 ][HIER]    # number of global parameters
    # print("NF=%d  N=%d" % (NF, N))
    if (0):
        np.random.seed(1234567)
    LINI = []
    if (len(LOCAL_INI)<1):                # Use ML estimates as the initial values
        I0, T0, B0 = MBB_fit_CL(F, S, dS, GPU=GPU, FIXED_BETA=-1, TMIN=TMIN, TMAX=TMAX, BMIN=BMIN, BMAX=BMAX)
        LINI = zeros((N,3), real)
        LINI[:,0], LINI[:,1], LINI[:,2] = I0, T0, B0    
        print("  I = %10.3e +- %10.3e" % (mean(LINI[:,0]), std(LINI[:,0]))) ;
        print("  T = %10.3e +- %10.3e" % (mean(LINI[:,1]), std(LINI[:,1]))) ;
        print("  B = %10.3e +- %10.3e" % (mean(LINI[:,2]), std(LINI[:,2]))) ;
        if (1):
            LINI[:,1] = 20.0 + 1.5*np.random.randn(N)
            LINI[:,2] =  2.0 + 0.4*np.random.randn(N)
    else:
        # print("LOCAL_INI ", LOCAL_INI.shape)
        LINI = asarray(1.0*LOCAL_INI, real)
        # print("INIT LINI   %10.3e %10.3e %10.3e" % (mean(LINI[:,0]), mean(LINI[:,1]), mean(LINI[:,2])))
    SSTEP = 0.001*ones(NG+3*N, float32)
    if (len(STEP)<1):
        for i in range(N):
            SSTEP[NG+3*i+0] = 0.003 ;
            SSTEP[NG+3*i+1] = 0.003 ;
            SSTEP[NG+3*i+2] = 0.002 ;
    else:
        SSTEP[:] = 1.0*STEP
    # OpenCL initialisation
    LOCAL    =  [ 1, 32][GPU>0]
    GLOBAL   =  [ N, (N//64+1)*64 ][GPU>0]
    platform, device, context, queue, mf = InitCL(GPU, platforms)
    DETMIN   =  1.0e-29  # was 1e-29  1e-30 was working, test 1e-29 again
    DETMIN   =  1.0e-32
    source   =  open(ISM_DIRECTORY+"/ISM/MBB/kernel_MBB_HMCMC_GIBBS.c").read()
    OPT      =  " -D N=%d -D NF=%d -D SAMPLES=%d -D STUDENT=%d -D HIER=%d -D DETMIN=%.3ef -D DOUBLE=%d  \
    -D TMIN=%.5ff -D TMAX=%.5ff -D BMIN=%.5ff -D BMAX=%.5ff -I ./ -D NG=%d -D LOCAL=%d -D LITER=%d \
    -I%s -D SEED=%.6ff" % \
    (N, NF, SAMPLES, STUDENT, HIER, DETMIN, DOUBLE, TMIN, TMAX, BMIN, BMAX, NG, LOCAL, LITER,    
    HOMEDIR+'starformation/Python/MJ/MJ/Aux/', rand())
    if (0):
        # OPT += " -cl-fast-relaxed-math"
        OPT += ' -cl-opt-disable '
    program  = cl.Program(context, source).build(OPT)
    ###
    F        =  asarray(F,  np.float32)
    S        =  asarray(S,  np.float32)
    dS       =  asarray(dS, np.float32)
    # Data for the global and local states, including their initial values
    G        =  zeros(NG+8,   real)           #  (I, T, B, dI, ...) or (T, B, dT, ...)
    L        =  zeros((N, 3), real)           #  (I, T, beta), for each work group and each pixel/source
    # GS       =  zeros((1,NG,SAMPLES),  np.float32)   # storage for hyperparameters samples
    # LS       =  zeros((1,N,3,SAMPLES), np.float32)   # storage for the (I, T, beta) values of N sources
    # Buffers for input data
    F_buf    =  cl.Buffer(context, mf.READ_ONLY, 4*NF)
    S_buf    =  cl.Buffer(context, mf.READ_ONLY, 4*N*NF)
    dS_buf   =  cl.Buffer(context, mf.READ_ONLY, 4*N*NF) 
    STEP_buf =  cl.Buffer(context, mf.READ_ONLY, 4*(NG+3*N))
    cl.enqueue_copy(queue, F_buf,  F)
    cl.enqueue_copy(queue, S_buf,  S)
    cl.enqueue_copy(queue, dS_buf, dS)
    cl.enqueue_copy(queue, STEP_buf,  SSTEP)

    # Buffers for samples
    if (len(GLOBAL_INI)>0):
        G[0:NG] = 1.0*GLOBAL_INI   # G has additional space for cov. SI[0:7] and SI[7] == priors
    else:
        if (HIER==2):          # 3D Student t or 3D Gaussian hierarchical part
            #  G[1, NG]   ---- NG=9
            G[0]   =  mean(LINI[:,0])*(1.0+0.1*np.random.randn())      # I
            G[1]   =  mean(LINI[:,1])*(1.0+0.1*np.random.randn())      # T    
            G[2]   =  mean(LINI[:,2])*(1.0+0.1*np.random.randn())      # beta 
            # Initialise the rest using the covariance matrix of the observations
            SIG      =  np.cov(transpose(LINI))  # cov needs parameter vectors as rows
            G[3]   =  sqrt(SIG[0,0]*(1.0+0.1*np.random.randn()) )     #  std(I)
            G[4]   =  sqrt(SIG[1,1]*(1.0+0.1*np.random.randn()) )     #  std(T)
            G[5]   =  sqrt(SIG[2,2]*(1.0+0.1*np.random.randn()) )     #  std(beta)
            G[6]   =  SIG[0,1] / (sqrt(SIG[0,0]*SIG[1,1]))            #  rho(I,T)
            G[7]   =  SIG[0,2] / (sqrt(SIG[0,0]*SIG[2,2]))            #  rho(I,beta)
            G[8]   =  SIG[1,2] / (sqrt(SIG[1,1]*SIG[2,2]))            #  rho(T,beta)
        else:  # just 2D Gaussian as the hierarchical part ... or no hyperparameters at all
            # G[NG]  ---   NG=5
            G[0]   =  mean(LINI[:,1]) * (1.0+0.0001*np.random.randn())   #  T  
            G[1]   =  mean(LINI[:,2]) * (1.0+0.0001*np.random.randn())   #  B  
            G[2]   =  std( LINI[:,1]) * (1.0+0.0001*np.random.randn())   # dT   will change fast, more important to have initial (T,beta) ~correct
            G[3]   =  std( LINI[:,2]) * (1.0+0.0001*np.random.randn())   # dB  
            G[4]   =  -0.5*(1.0+0.0001*np.random.randn())               # rho 
    #================================================================================
    L[:, 0] =  LINI[:, 0] #* (1.0+0.0001*np.random.randn(N))        # L[ N, 3]
    L[:, 1] =  LINI[:, 1] #* (1.0+0.0001*np.random.randn(N))
    L[:, 2] =  LINI[:, 2] #* (1.0+0.0001*np.random.randn(N))
    # Buffers for the current state
    G        =  asarray(ravel(G), real)
    G_buf    =  cl.Buffer(context, mf.READ_WRITE, NB*(NG+8))     # global state [NG]
    GP_buf   =  cl.Buffer(context, mf.READ_WRITE, NB*(NG+8))     # space for per-source parameter propositions
    L        =  asarray(ravel(L), real)
    L_buf    =  cl.Buffer(context, mf.READ_WRITE, NB*N*3)        # local state  [N, 3]
    lnP_buf  =  cl.Buffer(context, mf.READ_WRITE, NB*(N+1))      # N sources + 1 lnP from H.P. priors
    GS_buf   =  cl.Buffer(context, mf.WRITE_ONLY, NB*NG*SAMPLES)
    LS_buf   =  cl.Buffer(context, mf.WRITE_ONLY, NB*N*3*SAMPLES)
    cl.enqueue_copy(queue, G_buf,  G)   # NG+8
    cl.enqueue_copy(queue, GP_buf, G)   # NG+8
    cl.enqueue_copy(queue, L_buf,  L)
    rng_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*N*2)         # 2 uint per *active* work item
    queue.finish()
    # The kernel
    GPROP    =  program.GPROP
    GSTEP    =  program.GSTEP
    LSTEP    =  program.LSTEP
    #                            tag       G     GP    rng0  step
    GPROP.set_scalar_arg_dtypes([np.int32, None, None, None, None])
    #                            F     S     dS    G     L     lnP  
    GSTEP.set_scalar_arg_dtypes([None, None, None, None, None, None])
    #                            sample    F     S     dS    G     L     rng   lnP   GS    LS    step
    LSTEP.set_scalar_arg_dtypes([np.int32, None, None, None, None, None, None, None, None, None, None])


    HOST_RNG = 0
    rng = zeros(2*N, np.uint32)
    
    
    # Initialise random numbers --- G is not changed but SI is calculated
    GPROP(queue, [GLOBAL,], [LOCAL,], -1, GP_buf, G_buf, rng_buf, STEP_buf)   # initialise rng and G[]
    # First call to GSTEP, just to get lnP for the initial state --- for the original (G, L)
    GSTEP(queue, [GLOBAL,], [LOCAL,], F_buf, S_buf, dS_buf, G_buf, L_buf, lnP_buf)    
    # lnP[N+1] ---  lnP of each source and lnP[N] == priors of the hyperparameters
    lnP_all, lnP0_all = zeros(N+1, real), zeros(N+1, real)
    cl.enqueue_copy(queue, lnP0_all, lnP_buf)    # lnP for original G and  L
    queue.finish()
    lnP0      =  sum(lnP0_all)                   # lnP of the initial state -- includes all lnP components
    #print("********************************************************************************")
    print("INITIAL      lnP0 %12.4e   global priors [%12.4e]" % (lnP0, lnP0_all[N]))
    #print("********************************************************************************")
    accept = 0                                   # will again start with the G[]
    
    for sample in range(BURNIN*SITER):
        if (sample%10000==0): print("BURNIN  %.3f" % (sample/float(BURNIN*SITER)))
        for i in range(GITER):
            GPROP(queue,[GLOBAL,],[LOCAL,], 1, G_buf, GP_buf, rng_buf, STEP_buf)          # G -> GP
            GSTEP(queue,[GLOBAL,],[LOCAL,], F_buf, S_buf, dS_buf, GP_buf, L_buf, lnP_buf) # ** lnP_buf filled **
            cl.enqueue_copy(queue, lnP_all, lnP_buf)                                      # lnP for proposal GP
            #  lnP_all[N] is the additional prior probability of hyperparemeters
            queue.finish()
            lnP    =  sum(lnP_all)                                                        # lnP of proposed step 
            accept =  int( isfinite(lnP) & ((lnP-lnP0) > log(rand())) )
            if (accept): 
                lnP0        =  lnP
                lnP0_all[:] =  1.0*lnP_all[:]   # last accepted per-source probabilities + HP prior probability
                cl.enqueue_copy(queue, G,     GP_buf)   # NG+8
                cl.enqueue_copy(queue, G_buf, G     )
                queue.finish()

        # LSTEP needs the up-to-date lnP0_all values.... lnP0_all[N] = HP prior probability NOT used
        cl.enqueue_copy(queue, lnP_buf, lnP0_all) # kernel has lnP[N] == pb for HP priors
        queue.finish()
        LSTEP(queue,[GLOBAL,],[LOCAL,], -1, F_buf, S_buf, dS_buf, G_buf, L_buf, rng_buf, lnP_buf, GS_buf, LS_buf, STEP_buf)
        cl.enqueue_copy(queue, lnP0_all, lnP_buf) # lnP for proposal G2, L unchanged
        lnP0  =  sum(lnP0_all)                    # LSTEP returns lnP for the last *accepted* step
        # LSTEP does not touch lnP0_all[N], lnP0 is now total probability, priors of HP included
        if (sample%10000==0): print("G   sample=%6d   lnP=%12.5e [%12.4e]  lnP0=%12.5e [%12.4e]" % (sample, lnP, lnP_all[N], lnP0, lnP0_all[N]))

        
        
        
    if (HOST_RNG<0):  # use host random numbers to start kernel random number chains
        rng =  asarray(np.random.randint(0, 4294967295, size=2*N, dtype=np.uint32), np.uint32)
        cl.enqueue_copy(queue, rng_buf, rng)
        queue.finish()                    

    g_accept = 0
    for sample in range(SAMPLES):
        for siter in range(SITER):  # subiterations of GITER+LITER steps
            # Loop over a number of propositions for the hyperparameters
            for i in range(GITER):
                if (HOST_RNG<-1):  # use host random numbers to re-start kernel random number chains
                    rng =  asarray(np.random.randint(0, 4294967295, size=2*N, dtype=np.uint32), np.uint32)
                    cl.enqueue_copy(queue, rng_buf, rng)
                    queue.finish()                    
                GPROP(queue,[GLOBAL,],[LOCAL,], accept, G_buf, GP_buf, rng_buf, STEP_buf)        # new GP[]
                queue.finish()                    
                GSTEP(queue,[GLOBAL,],[LOCAL,], F_buf, S_buf, dS_buf, GP_buf, L_buf, lnP_buf)    # new lnP
                queue.finish()                    
                cl.enqueue_copy(queue, lnP_all, lnP_buf)   # this is still just a proposition == GP
                queue.finish()
                lnP    =  sum(lnP_all)  # includes lnP for individual sources + hyperparameter priors
                # if ((sample%(SAMPLES//4)==0)&(i%100==0)): print("      ..... GSTEP   lnP=%12.5e  lnP0=%12.5e" % (lnP, lnP0))
                accept =  int( isfinite(lnP) & ((lnP-lnP0) > log(rand())) )
                # print(accept)
                if (accept): 
                    lnP0         =  lnP
                    lnP0_all[:]  =  1.0*lnP_all[:]
                    cl.enqueue_copy(queue, G,     GP_buf)
                    cl.enqueue_copy(queue, G_buf, G     )
                    queue.finish()
                    g_accept += 1
                if (i%10==-1):
                    print("------------------------------------------------------------------------------------------")
                    print(G)
                    print("------------------------------------------------------------------------------------------")
            if (GITER>0):
                if (sample%200==0): print("GLOBAL   lnP0 %12.6e   global priors [%12.6e]   lnP %12.6e" % (lnP0, lnP0_all[N], lnP))
                # LSTEP must have in lnP_buf the probabilities from the last **accepted** GSTEP
                cl.enqueue_copy(queue, lnP_buf, lnP0_all)  # current lnP of last accepted step
                queue.finish()
            else:
                lnP = lnP0
            # ====================================================================================================
            if (HOST_RNG<-2):  # use host random numbers to re-start kernel random number chains
                rng =  asarray(np.random.randint(0, 4294967295, size=2*N, dtype=np.uint32), np.uint32)
                cl.enqueue_copy(queue, rng_buf, rng)
                # queue.finish()            
            queue.finish()                
            LSTEP(queue,[GLOBAL,],[LOCAL,], int([-1, sample][siter==(SITER-1)]),
            F_buf, S_buf, dS_buf, G_buf, L_buf, rng_buf, lnP_buf, GS_buf, LS_buf, STEP_buf)
            cl.enqueue_copy(queue, lnP0_all, lnP_buf)   # last accepted step probabilities
            queue.finish()
            lnP0  =  sum(lnP0_all)                      # LSTEP returns lnP for the last *accepted* step
            if (sample%99999==999999): 
                print("LOCAL    lnP0 %12.6e   global priors [%12.6e]" % (lnP0, lnP0_all[N]))
            # ====================================================================================================

                
        if ((sample*SITER)%10000==999999):
            print("   SAMPLE  %.3f    %12.5e" % (sample*SITER/float(SAMPLES*SITER), lnP0))

            
    print("FINAL        lnP0 %12.4e   global priors [%12.4e], g_accept %d" % (lnP0, lnP0_all[N], g_accept))
            
    GS  =  zeros((NG,SAMPLES),  real)   # storage for hyperparameters samples
    LS  =  zeros((N,3,SAMPLES), real)   # storage for the (I, T, beta) values of N sources
    cl.enqueue_copy(queue, GS, GS_buf)
    cl.enqueue_copy(queue, LS, LS_buf)
    GS.shape = (1, NG, SAMPLES)
    LS.shape = (1, N, 3, SAMPLES)
    #print("G: min(T)=%.3f  min(B)=%.3f" % (min(ravel(GS[0,0,:])), min(ravel(GS[0,1,:]))))
    #print("L: min(T)=%.3f  min(B)=%.3f" % (min(ravel(LS[0,:,1,:])), min(ravel(LS[0,:,2,:]))))
    ####
    # The global and local parameters from the last accepted step
    GINI = zeros(NG, real)
    cl.enqueue_copy(queue, GINI, G_buf)  # ONLY NG VALUES
    cl.enqueue_copy(queue, LINI, L_buf)
    LINI.shape = (N, 3)
    ####
    print("EXIT LINI   %10.3e %10.3e %10.3e" % (mean(LINI[:,0]), mean(LINI[:,1]), mean(LINI[:,2])))
    print("EXIT GINI ", GINI)    
    return GS, LS, GINI, LINI, lnP0





def MBB_HMCMC_BASIC_CL(F, S, dS, GPU=0, TMIN=5.0, TMAX=40.0, BMIN=0.4, BMAX=4.5,
                     SAMPLES=2000, BURNIN=1000, THIN=20, INI=[], HIER=1, STEP_SCALE=1.0, STUDENT=0, 
                     platforms=[0,1,2,3,4], STEP=[], DOUBLE=0):
    """
    Fit a set of observations with modified blackbody function, including a 
    hierarchical part for the overall (T, beta) or (I, T, beta) distribution.
    This one more basic, no affine, no robust adaptive!
    Usage:
        GS, LS  = MBB_HMCMC_BASIC_CL(F, S, dS, GPU=0, TMIN=5.0, TMAX=40.0, BMIN=0.4, BMAX=4.5, 
                                   SAMPLES=2000, BURNIN=1000, THIN=20, INI=[], HIER=1, STUDENT=0)
    Input:
        F          =   vector of frequencies
        S          =   S[NPIX, NFREQ] of flux density values
        dS         =   dS[NPIX, NFREQ] of flux density uncertainty estimates
        TMIN, TMAX =   hard limits for the temperatures
        BMIN, BMAX =   hard limits for the spectral index values
        SAMPLES    =   number of samples returned (after thinning)
        BURNIN     =   number of initial samples (before thinning) rejected
        THIN       =   return every THIN:th values only
        INI        =   optional initial values INI[N, 3], with columns [I(250um), T, beta];
                       **or** a single vector with NG+3*N elements;
                       if not given, start with the ML solution
        LOCAL      =   number of threads per work group (default 4 for CPU, 32 for GPU)
        GLOBAL     =   total number of threads (default 16 for CPU, 128 for GPU)
        HIER       =   0 = no hierarchical part, 
                       1 = hierarchical part for (T, beta),
                       2 = hierarchical part for (I, T, beta)
        STEP_SCALE =   scalar scaling of all step sizes (default 1.0)
        STUDENT    =   if >0, use Student t distributions for hierarchical and the per-source
                       parts; default is the normal distribution
        platforms  =   optional list of possible OpenCL platorms, e.g. [0, 1, 2]
        STEP       =   optional vector of step lengths, NG+3*N parameters
                       (NG=5 for HIER==1 and NG=9 for HIER==2)
    Return:
        PS, STEP, lnP 
              PS   =   PS[NG+3*N, samples]
              STEP =   estimated good step sizes (actually std of samples)
              lnP  =   final value of lnP
    Note:
        - HIER=0 => no hyperparameters
          HIER=1 implies a set of NG=5 hyperparameters [ T, B, dT, dB, rho(T,B) ]  (B=beta)
          HIER=2 implies a set of NG=9 hyperparameters [ I, T, B, dI, dT, dB, rIT, rIB, rTB ]
          *** HIER==2 actually not implemented !!! ***
    """
    NF       =   len(F)             # number of frequencies
    N        =   S.shape[0]         # number of sources
    if (GPU):
        LOCAL = 32
    else:
        LOCAL = 4
    GLOBAL   =   (1+N//32)*32
    ## np.random.seed(123456)
    if (len(INI)<1):                # Use ML estimates as the initial values
        print("*** INI from ML estimates !!!")
        I0, T0, B0 = MBB_fit_CL(F, S, dS, GPU=GPU, FIXED_BETA=-1, TMIN=TMIN, TMAX=TMAX, BMIN=BMIN, BMAX=BMAX)
        INI = zeros((N,3), np.float32)
        INI[:,0], INI[:,1], INI[:,2] = I0, T0, B0    
        if (0):
            INI[:,1] = 18.0 + 1.0+np.random.randn(N)
            INI[:,2] =  1.3 + 0.2*np.random.randn(N)            
        if (1):
            print("  I = %10.3e +- %10.3e" % (mean(INI[:,0]), std(INI[:,0]))) ;
            print("  T = %10.3e +- %10.3e" % (mean(INI[:,1]), std(INI[:,1]))) ;
            print("  B = %10.3e +- %10.3e" % (mean(INI[:,2]), std(INI[:,2]))) ;
    # OpenCL initialisation
    platform, device, context, queue,  mf = InitCL(GPU, platforms)
    if (not(HIER in [0, 1, 2])):
        print("*** MBB_HMCMC_BASIC_CL: parameter HIER must be 0, 1, or 2")
        sys.exit()
    NG = [0, 5, 9][HIER]  # number of global parameters
    source     = open(ISM_DIRECTORY+"/ISM/MBB/kernel_MBB_HMCMC_BASIC.c").read()
    OPT     =  " -D N=%d -D NF=%d -D NG=%d -D NP=%d -D SAMPLES=%d -D THIN=%d -D BURNIN=%d -D STUDENT=%d -D HIER=%d -D STEP_SCALE=%.5ef \
    -D TMIN=%.5ff -D TMAX=%.5ff -D BMIN=%.5ff -D BMAX=%.5ff -I /home/mika/starformation/Python/MJ/MJ/Aux -D LOCAL=%d -D GLOBAL=%d -D DOUBLE=%d" % \
    (N, NF, NG, NG+3*N, SAMPLES, THIN, BURNIN, STUDENT, HIER, STEP_SCALE, TMIN, TMAX, BMIN, BMAX, LOCAL, GLOBAL, (DOUBLE>0))
    program  = cl.Program(context, source).build(OPT)
    ###
    F        =  asarray(F,  np.float32)
    S        =  asarray(S,  np.float32)
    dS       =  asarray(dS, np.float32)
    # Data for the global and local states, including their initial values
    NP       =  NG+3*N                                    # total number of parameters
    P        =  zeros(NP,   np.float32)                   # current vector
    PS       =  zeros((NP, SAMPLES), np.float32)          # storage for samples
    # Buffers for input data
    F_buf    =  cl.Buffer(context, mf.READ_ONLY, 4*NF)
    S_buf    =  cl.Buffer(context, mf.READ_ONLY, 4*N*NF)
    dS_buf   =  cl.Buffer(context, mf.READ_ONLY, 4*N*NF) 
    # Two alternative buffers for the current state
    P1_buf   =  cl.Buffer(context, mf.READ_WRITE, P.nbytes)     # current state
    P2_buf   =  cl.Buffer(context, mf.READ_WRITE, P.nbytes)     # proposal
    DX_buf   =  cl.Buffer(context, mf.READ_WRITE, P.nbytes)     # step sizes
    lnP_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*N)
    # random number state = 2 uints per work item
    rng_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL*2)
    
    lnP0 = -1e32
    if (INI.shape[0]==(NG+3*N+1)):  # last element is the lnP value !
        # user gave the full vector, including the hyperparameters
        P[:]   = 1.0*INI[0:NP]
        lnP0   = INI[NP]
        print("*** USING USER_PROVIDED FULL STATE ==>  %12.8f" % lnP0)
    else:
        # user has given or we have computed INI[N,3] initial values
        print("*** USING USER-PROVIDED PER-SOURCE STATES ONLY ***")
        if (NG==9):      # 3D Student t or 3D Gaussian hierarchical part
            P[0]   =  mean(INI[:,0])    # I    
            P[1]   =  mean(INI[:,1])    # T    
            P[2]   =  mean(INI[:,2])    # beta 
            # Initialise the rest using the covariance matrix of the observations
            SIG    =  np.cov(transpose(INI))  # cov needs parameter vectors as rows
            P[3]   =  sqrt(SIG[0,0])    #  std(I)
            P[4]   =  sqrt(SIG[1,1])    #  std(T)
            P[5]   =  sqrt(SIG[2,2])    #  std(beta)
            P[6]   =  SIG[0,1] / (sqrt(SIG[0,0]*SIG[1,1]))  #  rho(I,T)
            P[7]   =  SIG[0,2] / (sqrt(SIG[0,0]*SIG[2,2]))  #  rho(I,beta)
            P[8]   =  SIG[1,2] / (sqrt(SIG[1,1]*SIG[2,2]))  #  rho(T,beta)
        elif(NG==5):      # just 2D Gaussian as the hierarchical part ... or no hierarchical level at all
            P[0]   =  mean(INI[:,1])    #  T  
            P[1]   =  mean(INI[:,2])    #  B  
            P[2]   =  std( INI[:,1])    # dT   will change fast, more important to have initial (T,beta) ~correct
            P[3]   =  std( INI[:,2])    # dB  
            P[4]   =  -0.5              # rho 
        for i in range(N):
            P[(NG+3*i):(NG+3*i+3)] =   INI[i,:]
        if (1):
            tmp = P[NG:].copy().reshape(N,3)
            print("<INI>  =  %10.3e  %10.3e  %10.3e" % (mean(tmp[:,0]), mean(tmp[:,1]), mean(tmp[:,2])))
    # The kernels
    SEED    =  program.Seed
    SEED.set_scalar_arg_dtypes([np.float32, None,]) # seed, rng_state
    
    PROPO   =  program.MakeProposal
    PROPO.set_scalar_arg_dtypes([None, None, None, None ])    # rng_state, P1, P2, DX
    LNP     =  program.LNP
    LNP.set_scalar_arg_dtypes([None, None, None, None, None]) # F, S, dS, P, lnP
    if (len(STEP)<2):   DX  =  0.001*ones(NG+3*N, float32)
    else:               DX  =  asarray(1.0*STEP, float32)
    lnP_all =  zeros(N, float32)
    
    cl.enqueue_copy(queue, F_buf,  F)
    cl.enqueue_copy(queue, S_buf,  ravel(S))
    cl.enqueue_copy(queue, dS_buf, ravel(dS))
    cl.enqueue_copy(queue, P1_buf, ravel(P))
    cl.enqueue_copy(queue, DX_buf, DX)
    
    TIK  =  0  # P1 is the current state
    SEED(queue, [GLOBAL,], [LOCAL,], float(rand()), rng_buf)
    queue.finish()
    
    
    lnP = 0.0
    print("* %10.3e %10.3e %10.3e %10.3e %10.3e  lnP0 %10.3e  lnP %10.3e" % (P[0], P[1], P[2], P[3], P[4], lnP0, lnP))
        
    for i in range(0):
        if (TIK==0):  # P1 is the current, P2 is the proposal
            PROPO(queue, [GLOBAL,], [LOCAL,],  rng_buf, P1_buf, P2_buf, DX_buf)
            LNP(queue, [GLOBAL,], [LOCAL,],    F_buf,   S_buf,  dS_buf, P2_buf, lnP_buf)
        else:         # P2 is the current, P2 is the proposal
            PROPO(queue, [GLOBAL,], [LOCAL,],  rng_buf, P2_buf, P1_buf, DX_buf)
            LNP(queue, [GLOBAL,], [LOCAL,],    F_buf,   S_buf,  dS_buf, P1_buf, lnP_buf)
        cl.enqueue_copy(queue, lnP_all, lnP_buf)
        queue.finish()
        lnP = sum(asarray(lnP_all, float64))
        if ((lnP-lnP0)>log(rand())): 
            # accept the step, if TIK=0, accept P2 as the current state and vice versa
            if (TIK==0): TIK = 1
            else:        TIK = 0
            lnP0  =  lnP
        cl.enqueue_copy(queue, P, [P1_buf, P2_buf][TIK])
        if (i%100==101):
            print("  %10.3e %10.3e %10.3e %10.3e %10.3e   lnP0 %10.3e  lnP %10.3e" \
            % (P[0], P[1], P[2], P[3], P[4], lnP0, lnP))

            
            
            
    for i in range(SAMPLES):
        for j in range(THIN):
            if (TIK==0):  # P1 is the current, P2 is the proposal
                PROPO(queue, [GLOBAL,], [LOCAL,], rng_buf, P1_buf, P2_buf, DX_buf)
                LNP(queue, [GLOBAL,], [LOCAL,], F_buf, S_buf, dS_buf, P2_buf, lnP_buf)
            else:         # P2 is the current, P1 is the proposal
                PROPO(queue, [GLOBAL,], [LOCAL,], rng_buf, P2_buf, P1_buf, DX_buf)
                LNP(queue, [GLOBAL,], [LOCAL,], F_buf, S_buf, dS_buf, P1_buf, lnP_buf)
            cl.enqueue_copy(queue, lnP_all, lnP_buf)
            queue.finish()
            lnP = sum(asarray(lnP_all, float64))
            if (   ((lnP-lnP0)>log(rand()))   ):
                if (TIK==0): TIK = 1    #  P2 accepted
                else:        TIK = 0    #  P1 accepted
                lnP0 = lnP
            if ((i%499==0)&(j==0)):
                print("%10.6f %10.6f %10.6f %10.6f %10.6f  lnP0 %12.6f  lnP %12.6f" % (P[0], P[1], P[2], P[3], P[4], lnP0, lnP))
                # sys.exit()
        # after THIN steps, add one sample
        if (TIK==0):   cl.enqueue_copy(queue, P, P1_buf)
        else:          cl.enqueue_copy(queue, P, P2_buf)
        queue.finish()
        PS[:, i]  =  P    # PS[NP, SAMPLES]
    print("%10.6f %10.6f %10.6f %10.6f %10.6f  lnP0 %12.6f  lnP %12.6f" % (P[0], P[1], P[2], P[3], P[4], lnP0, lnP))
    # estimate good step ???
    STEP = std(PS, axis=1)
    print("*** FINAL STATE => %12.8f" % sum(PS[:,SAMPLES-1]))
    return PS, STEP, lnP0   #  lnP0 is the probability for the last sample in PS




####################################################################################################

####################################################################################################

####################################################################################################


from scipy.optimize import leastsq


def ModifiedBlackbody_250(f, T, beta):
    """
    Return value of modified black body B(T)*f^beta at given frequency f [Hz],
    the returned values are normalized with the value at 250um.
    """
    return ((f/1.19917e+12)**(3.0E0+beta)) * (exp(H_K*1.19917e+12/T)-1.0) / (exp(H_K*f/T)-1.0)



def Delta_fix_beta(p, f, S, dS, beta):
    """
    Given a modified blackbody curve with parameters p = [ I, T ],
    observed frequencies f and intensities S+-dS, return normalised residuals.
    p = [ I, T], emissivity index beta is kept constant.
    Note:
        Reference wavelength 250um = as defined in ModifiedBlackbody_250() routine !
    """
    # print 'p f S dS beta', p, f, S, dS, beta
    I  = p[0]*ModifiedBlackbody_250(f, p[1], beta)   # model predictions
    I  = (I-S)/dS
    return I/len(I)


def Deriv_fix_beta(p, f, S, dS, beta):
    """
    Return derivatives - NOT TESTED
    """
    D  = zeros((len(S), 2), float32)
    for i in range(len(S)):
        D[i,0] = ModifiedBlackbody_250(f, p[1], beta)/dS[i]
        D[i,1] = D[i,0] * H_K/(p[1]*p[1]) / (1.0-exp(-H_K*f[i]/p[1]))
    return D



def Delta_fix_T(p, f, S, dS, fixed_T):
    """
    Given a modified blackbody curve with parameters p = [ I200, beta ],
    observed frequencies f and intensities S+-dS, return chi2 error.
    p = [ I, T], emissivity index beta is kept constant.
    """
    I  = p[0]*ModifiedBlackbody_250(f, fixed_T, p[1])
    I  = (I-S)/dS
    return I/len(I)


def Deriv_fix_T(p, f, S, dS, fixed_T):
    """
    Return derivatives - NOT TESTED
    """
    f0 = um2f(200.0)
    D  = zeros((len(S), 2), float32)
    for i in range(len(S)):
        D[i,0] = ModifiedBlackbody_250(f[i], fixed_T, p[1])/dS[i]   #  d/dI
        D[i,1] = D[i,0] *  p[1] * (f0/f[i])                     #  d/dbeta
    return D



def Delta_free(p, f, S, dS, beta):
    """
    Given a modified blackbody curve with parameters p = [ I250, T, beta ],
    observed frequencies f and intensities S+-dS, return normalised residuals.
    Note:
        Reference wavelength 250um = as defined in ModifiedBlackbody_250() routine !
    """
    I   =  p[0]*ModifiedBlackbody_250(f, p[1], p[2])   # model predictions
    I   =  (I-S)/dS
    return I/len(I)





def FitModifiedBlackbody_simple(um, FF, dFF, I250, T, beta, fix_T=False, fix_beta=False,
                        Tmin=3.0, Tmax=35.0, beta_min=0.5, beta_max=4.0, 
                        xtol=1.0e-5, ftol=1.0e-9):
    """
    Usage:
        p     = FitModifiedBlackbody(um, S, dS, I200, T, beta, fix_T=False, fix_beta=False)
    Input:
        um       = vector of wavelenths [um]
        S        = intensity values
        FF       = pyfits images at different wavelengths
        I250     = initial value for 250um intensity (input, array)
        T        = initial value for temperature [K], scalar
        beta     = initial value for spectral index, scalar
        fix_T    = if True, keep temperature fixed
        fix_beta = if True, keep spectral index fixed
        Tmin, Tmax         = temperature values cut to this interval
        beta_min, beta_max = spectral indices cut to this interval
    Returns:
        Three pyfits images (even when T or beta is kept fixed):
            250um intensity, colour temperature, spectral index
    """
    f   = C_LIGHT/(1.0e-4*asarray(um))
    NF  = len(f)
    tmp = None 
    N, M = FF[0][0].data.shape
    SS   = zeros((N*M, NF), float32)
    dSS  = zeros((N*M, NF), float32)
    for iband in range(NF):
        SS[:, iband] = ravel(FF[iband][0].data)
        if (dFF):
            dSS[:, iband] = ravel(dFF[iband][0].data)
    I250 = ravel(I250)
    beta = 1.8
    # Results will go to FITS images, one for 250um intensity, one for colour temperature, one fore spectral index
    res = zeros((N*M, 3), float32)
    for ipix in range(N*M):
        S    = SS[ipix, :]
        dS   = dSS[ipix,:]
        if (fix_T):        #  T fixed
            p0  = [ I250[ipix], max(beta_min, min(beta_max, beta)) ]
            if (0):
                p1  = fmin(BBChi2_fix_T, p0, args=(f, S, dS, T), disp=0)
            else:
                # p1, p1Con = leastsq(Delta_fix_T, p0, args=(f, SS, dS, beta), Dfun=Deriv_fix_T)
                p1, p1Con = leastsq(Delta_fix_T, p0, args=(f, S, dS, beta), Dfun=None)
            res[ipix, :] = [ p1[0], T, beta ]                
        elif (fix_beta):   # BETA fixed
            p0  = [ I250[ipix], max(Tmin, min(Tmax, T)) ]
            if (0):
                p1  = fmin(BBChi2_fix_beta, p0, args=(f, S, dS, beta), disp=0)
            else:         
                # p1, p1Con = leastsq(Delta_fix_beta, p0, args=(f, S, dS, beta), Dfun=Deriv_fix_beta, disp=0)
                p1, p1Con = leastsq(Delta_fix_beta, p0, args=(f, S, dS, beta), Dfun=None, disp=0)
            res[ipix, :] = [ p1[0], p1[1], beta ]
        else:              # T and BETA both free
            p0  = [ I250[ipix], max(Tmin, min(Tmax, T)), max(beta_min, min(beta_max, beta)) ]
            p1, p1Con = leastsq(Delta_free, p0, args=(f, S, dS, beta), Dfun=None)
            res[ipix, :] = p1
    ###
    FI, FT, FB = CopyFits(FF[0]), CopyFits(FF[0]), CopyFits(FF[0])
    FI[0].data = res[:,0].reshape(N, M)
    FT[0].data = res[:,1].reshape(N, M)
    FB[0].data = res[:,2].reshape(N, M)
    return FI, FT, FB
    
    
