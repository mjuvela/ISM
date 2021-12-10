
if (0):
    from MJ.mjDefs import *
else:
    import os, sys
    import numpy as np    
    import pyopencl as cl
    from scipy.interpolate import interp1d 




def PlanckSafe(f, T):  # Planck function
    # Add clip to get rid of warnings
    return 2.0*H_CC*f*f*f / (np.exp(np.clip(H_K*f/T,-100,+100))-1.0)



def SolveEquilibrium(FREQ, ABSORBED, EMITTED, SK_ABS, ISIZE, AF):
    
    cells, nfreq = ABSORBED.shape
    #
    ABS    =  SK_ABS[ISIZE,:]       # absorption cross sections for this dust
    # ABSORBED must be scaled by the fraction of absorptions due to the current size
    # absorption fraction between dust components does not appear -->
    #        A2E_pyCL is called with ABSORBED that already the fraction for this dust population
    # fraction  of absorptions due to current size is in AF = SK_ABS / [sum(SK_ABS)*S_FRAC*GD] = per grain!
    K      =  ones(NFREQ, float64)  # fraction
    K      =  ABS/SUM(SK_ABS[
    # Prepare lookup table between energy and temperature
    NE     = 30000        # number of interpolation points
    t1     = time.time()  # this is fast (<1sec) and can be done by host
    TSTEP  = 1600.0/NE    # hardcoded upper limit 1600K for the maximum dust temperatures
    TT     = zeros(NE, float64)
    Eout   = zeros(NE, float64)
    DF     = FREQ[2:] - FREQ[:(-2)]  #  x[i+1] - x[i-1], lengths of intervals for Trapezoid rule
    for i in range(NE):
        TT[i]  =  1.0+TSTEP*i
        TMP    =  ABS * PlanckSafe(asarray(FREQ, float64), TT[i])
        # Trapezoid integration TMP over FREQ frequencies
        res     = TMP[0]*(FREQ[1]-FREQ[0]) + TMP[-1]*(FREQ[-1]-FREQ[-2]) # first and last step
        res    += sum(TMP[1:(-1)]*DF)  # the sum over the rest of TMP*DF
        Eout[i] = (4.0*PI*1e20/(USER.GL*PARSEC)) * 0.5 * res  # energy corresponding to TT[i]
    # Calculate the inverse mapping    Eout -> TTT
    Emin, Emax = Eout[0], Eout[NE-1]*0.9999
    # E ~ T^4  => use logarithmic sampling
    kE     = (Emax/Emin)**(1.0/(NE-1.0))  # E[i] = Emin*pow(kE, i)
    # oplgkE = 1.0/log10(kE)
    ip     = interp1d(Eout, TT)           # (linear) interpolation from energy to temperature
    TTT    = asarray(ip(Emin * kE**arange(NE)), float32)
    print("Mapping E -> T calculated on host: %.3f seconds" % (time.time()-t1))
    
    
    # Calculate temperatures on device
    #   ABSORBED * AF  integrated over frequency -> Ein   !!!
    #   = sum ABSORBED * AF * TW, TW = integration weight
    #   kernel will get ABSORBED, will do integration and table lookup to get T
    #   Because of the large amount of data, kernel calls for GLOBAL cells at a time
    GLOBAL   =  2048
    TTT_buf  =  cl.Buffer(context[0], mf.READ_ONLY,   TTT.nbytes)
    ABS_buf  =  cl.Buffer(context[0], mf.READ_ONLY,   4*GLOBAL*NFREQ)
    T_buf    =  cl.Buffer(context[0], mf.READ_WRITE,  4*GLOBAL)
    kernel_T =  program.EqTemperature
    TNEW     =  zeros(CELLS, float32)
    #                               icell     kE          oplogkE     Emin        NE         TTT    ABS    T   
    kernel_T.set_scalar_arg_dtypes([np.int32, np.float32, np.float32, np.float32, np.int32,  None,  None,  None])
    cl.enqueue_copy(commands, TTT_buf,  TTT)
    for icell in range(0, CELLS, GLOBAL):
        # T_buf is being updated for GLOBAL cells
        b    =  min(icell+GLOBAL, CELLS)
        cl.enqueue_copy(commands, ABS_buf,  ABSORBED[icell, b])
        kernel_T(commands[0], [GLOBAL,], [LOCAL,], icell, kE, oplgkE, Emin, NE,  TTT_buf, ABS_buf, T_buf)
    
        cl.enqueue_copy(commands[0], EMIT, EMIT_buf) ;  # emission for <= GLOBAL cells
        
        # Add emission to the final array
        for i in range(icell,b):
            EMITTED[icell:b, :] += EMIT[i-icell,:]
        
