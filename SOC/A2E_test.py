#!/usr/bin/python3

# 2019-04-24 -- New version of A2E_pyCL.py - stripped away the iterative solvers and plotting

import os, sys

# we assume that the Python scripts and *.c kernel files are in this directory
# HOMEDIR = os.path.expanduser('~/')
# sys.path.append(HOMEDIR+'/starformation/SOC/')
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)


import pyopencl as cl
from   scipy.interpolate import interp1d
from   matplotlib.pylab import *
from   SOC_aux import *

"""
Usage:   A2E.py  dump absorbed.data emitted.data
Using solver.data dumped from A2E.cpp or A2E_pre.py, convert absorbed.data to emitted.data.

Notes 2019-10-28:
    DustLib reading DustEM dust:
        - GRAIN_DENSITY == 1e-7 dummy value -- NOT USED !!!!
        - CRT_SRAC == includes normalisation, sum(CRT_SFRAC) == true number of grains
    DustLib write_A2E_dustfile:
        - writes the gs_* dust files used by A2E_pre.py
        - file has GRAIN_DENSITY == sum(CRT_FRAC) == true total number of grains
        - CRT_SFRAC column has sum == 1.0   == ok
        - THIS DOES NOT USE DUST.GRAIN_DENSITY but assumes that it is in CRT_SFRAC --- true for DustEmDust !!!
    DustLib write_simple_dust:
        - Abs == Kabs(FREQ) == should be the total absorption cross section,
          METHOD='CRT' => KabsCRT  == sum(CRT_SFRAC*pi*SIZE_A**2) --- ok, GRAIN_DENSITY not used !!!
        - output file has GRAIN_DENSITY==1e-7, abs column /= (GRAIN_DENSITY*pi*GRAIN_SIZE**2)
        - different from write_A2E_dustfile but the end result also ok
    DustLib read GSETDust:
        - as noted above, file has sum of CRT_SFRAC column == 1, GRAIN_DENSITY is true grain density
        - does read GRAIN_DENSITY == true grain density written by write_A2E_dustfile
        - explicitly normalises sum(CRT_SFRAC)==1.0  --- should be already true, as written by write_A2E_dustfile
        - multiplies GRAIN_DENSITY into CRT_SFRAC --- thereafter GRAIN_DENSITY should not be used !!!
    A2E_pre.py:
        - dust data read using GSETDust
        - assumes that CRT_SFRAC in file contains GRAIN_DENSITY -- true, as read by GSETDust
        
    We had a problem with the CRT_SFRAC normalisation.... but only because we manually dropped
    a number of size bins, thereby making sum(CRT_FRAC)<1.
    A2E.cpp explicitly normalised sum(CRT_SFRAC)==1, thereby avoiding the problem.
    Explicit normalisation is now also in DustLib.py, in GSETDust::Init.
"""

ICELL     =  0
GPU       =  0.0      #     GPU.PLATFORM
WITH_X    =  0
        
if (len(sys.argv)<4):
    print("\n  A2E_pyCL  dump  absorbed.data emitted.data [GPU [nstoch]]\n")
    sys.exit()

if (not(os.path.exists(sys.argv[1]))):
    print("????????????????????????????????????????????????????????????????????????????????")
    print("The solver file %s does not exist !!!!!!!!!!!!!!!!!!!!!!!" % sys.argv[1])
    print("????????????????????????????????????????????????????????????????????????????????")
    sys.exit(1)
    
if (not(os.path.exists(sys.argv[2]))):
    print("????????????????????????????????????????????????????????????????????????????????")
    print("The file of absorptionse %s does not exist !!!!!!!!!!!!!!" % sys.argv[2])
    print("????????????????????????????????????????????????????????????????????????????????")
    sys.exit(1)

    
# since 2018-12-29 also SOC absorbed.data starts with [ cells, nfreq ] only
fp     =  open(sys.argv[2], 'rb')      # absorbed
CELLS, NFREQ = np.fromfile(fp, np.int32, 2)
fp.close()
    
NSTOCH = 999
if (len(sys.argv)>4):
    GPU  = float(sys.argv[4])
    if (len(sys.argv)>5):
        NSTOCH = int(sys.argv[5])

        
        
platform, device, context, queue = None, None, None, None
LOCAL = 16 
platforms = arange(4)
if (fmod(GPU,1.0)>0): platforms = [ int(10*fmod(GPU,1)), ]
for iplatform in platforms:
    try:
        platform = cl.get_platforms()[iplatform]
        if (GPU>=1.0):
            device   = platform.get_devices(cl.device_type.GPU)
            LOCAL    = 32  #  64 -> 32, TS test = no effect
        else:
            device   = platform.get_devices(cl.device_type.CPU)
            LOCAL    =  8
        context  = cl.Context(device)
        queue    = cl.CommandQueue(context)
        break
    except:
        pass
GLOBAL      =  max([4096, 64*LOCAL])
if (GLOBAL%64!=0):
    GLOBAL  = (GLOBAL/64+1)*64
context     =  cl.Context(device)
queue       =  cl.CommandQueue(context)
mf          =  cl.mem_flags


#BATCH =  8192     # RESTRICTED BY MEMORY USAGE !!!
BATCH  = GLOBAL
    
# Read solver dump file
FP     =  open(sys.argv[1], 'rb')      # dump
NFREQ  =  np.fromfile(FP, np.int32, 1)[0]                                # NFREQ
FREQ   =  np.fromfile(FP, np.float32, NFREQ)                             # FREQ[NFREQ]
GD     =  np.fromfile(FP, np.float32, 1)[0]                              # GRAIN_DENSITY
NSIZE  =  np.fromfile(FP, np.int32, 1)[0]                                # NSIZE
S_FRAC =  clip(np.fromfile(FP, np.float32, NSIZE), 1.0e-32, 1.0e30)      # S_FRAC
NE     =  np.fromfile(FP, np.int32, 1)[0]                                # NE
SK_ABS =  np.fromfile(FP, np.float32, NSIZE*NFREQ).reshape(NSIZE, NFREQ) # SK_ABS[NSIZE, NFREQ]
K_ABS  =  sum(SK_ABS, axis=0)

    
# Emitted has always 2 int header
fp    =  open(sys.argv[3], 'wb')      # emitted
asarray([CELLS, NFREQ], np.int32).tofile(fp)  # 2018-12-29 -- dropped levels from the python files !!
fp.close()

ABSORBED  = np.memmap(sys.argv[2], dtype='float32', mode='r+',  shape=(CELLS, NFREQ), offset=8)
"""
2019-10-27
Actual problem was in cpp version, the last two highest frequency getting large integration 
weight Iw values, one positive, one negative !!!!
As long as this python version uses the copy of those formulas, it produces similar results to
the cpp version ... but those might be incorrect/inaccurate depending on the absorptions in the
first two channels!!!
  The julia version in Dust.jl calculates the integration weights differently
(Euler instead of 3rd order accurate). With the same absorptions file it produced results
similar to those obtained by setting absorptions in the last two channels to zero. When
two last channels were set to zero, the results of the python and cpp versions changed and were
similar to the julia results.
   ==> it is very likely that the difference *is* produced by cpp/py handling those two frequency
bins incorrectly  ==> one should use the julia version to create the solver file and also 
implement in A2E_pre.py also that same version of Iw calculation !!!!
   The actual error depends on the significance of absorptions in the last two frequency bins
but might also on how the enthalpy limits W1-W4 happend to reside in relation to those frequencies.
"""
if (1):  # Necessary to avoid huge emission in rare cells ???
    ABSORBED[:, NFREQ-1] = np.clip(ABSORBED[:, NFREQ-1], 0.0, 0.2*ABSORBED[:, NFREQ-2])
EMITTED   = np.memmap(sys.argv[3], dtype='float32', mode='r+',  shape=(CELLS, NFREQ), offset=8)
EMITTED[:,:] = 0.0


def PlanckSafe(f, T):  # Planck function
    # Add clip to get rid of warnings
    H_CC    =  7.372496678e-48    #   PLANCK/C_LIGHT^2
    return 2.0*H_CC*f*f*f / (exp(np.clip(H_K*f/T,-100,+100))-1.0)
    

t0 = time.time()
NIP         =  30000  # number of interpolation points for the lookup tables (equilibrium dust)
ARGS        =  "-D NE=%d -D LOCAL=%d -D NFREQ=%d -D CELLS=%d -D NIP=%d -D FACTOR=%.4ef -D WITH_X=%d -D GLOBAL=%d" \
               % (NE, LOCAL, NFREQ, CELLS, NIP, FACTOR, WITH_X, GLOBAL)
if (0): # no effect on run times
    ARGS   +=  "-cl-fast-relaxed-math -cl-single-precision-constant -cl-mad-enable"
if (0):
    # ARGS  += " -cl-opt-disable"  # slower by x3 on CPU, x8 on GPU !!
    ARGS   += ' -cl-nv-opt-level 1'

source      =  open(INSTALL_DIR+"/kernel_A2E_new.c").read()
program     =  cl.Program(context, source).build(ARGS)
Iw_buf      =  cl.Buffer(context, mf.READ_ONLY,  NE*NE*NFREQ*4)
L1_buf      =  cl.Buffer(context, mf.READ_ONLY,  NE*NE*4)
L2_buf      =  cl.Buffer(context, mf.READ_ONLY,  NE*NE*4)
Tdown_buf   =  cl.Buffer(context, mf.READ_ONLY,  NE*4)
EA_buf      =  cl.Buffer(context, mf.READ_ONLY,  NE*NFREQ*4)
Ibeg_buf    =  cl.Buffer(context, mf.READ_ONLY,  NFREQ*4)
AF_buf      =  cl.Buffer(context, mf.READ_ONLY,  NFREQ*4)
ABS_buf     =  cl.Buffer(context, mf.READ_ONLY,  CELLS*NFREQ*4)
EMIT_buf    =  cl.Buffer(context, mf.READ_WRITE, CELLS*NFREQ*4)
A_buf       =  cl.Buffer(context, mf.READ_WRITE, GLOBAL*(int((NE*NE-NE)/2))*4) # lower triangle only



if (NSTOCH<NSIZE):    # Prepare to solve equilibrium temperature emission for larger grains
    TTT_buf   =  cl.Buffer(context, mf.READ_ONLY,   4*NIP)
    T_buf     =  cl.Buffer(context, mf.READ_WRITE,  4*BATCH)
    KABS_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NFREQ)
    FREQ_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NFREQ)
    # we can use EMIT_buf and ABS_buf, which are correct size for processing BATCH cells !!
    kernel_T  =  program.EqTemperature
    #                               icell     kE          oplogkE     Emin       
    kernel_T.set_scalar_arg_dtypes([np.int32, np.float32, np.float32, np.float32,
    #  FREQ   KABS    TTT    ABS    T      EMIT
    None,     None,   None,  None,  None,  None   ])
    cl.enqueue_copy(queue,   FREQ_buf,  FREQ)
    EMIT      =  zeros((BATCH,NFREQ), np.float32)
            
DoSolve =  program.DoSolve

if (WITH_X):
    X_buf     =  cl.Buffer(context, mf.WRITE_ONLY, BATCH*NE*4)    # no initial values -> write only
    X         =  zeros((BATCH, NE), np.float32)
    DoSolve.set_scalar_arg_dtypes([np.int32, np.int32, 
    None, None, None, None, None, None, None, None, None, None, None ])
else:
    DoSolve.set_scalar_arg_dtypes([np.int32, np.int32, 
    None, None, None, None, None, None, None, None, None, None ])


# A2E.py strips the option of iterative solvers -- no worries about initial values.
emit = zeros((CELLS, NFREQ), np.float32)

t0 = time.time()    
for isize in range(NSIZE):        

    
    # AF = fraction of absorptions due to the current size
    AF    = asarray(SK_ABS[isize,:], float64) / asarray(K_ABS[:], float64)  # => E per grain
    AF   /= S_FRAC[isize]*GD  # "invalid value encountered in divide"
    AF    = asarray(np.clip(AF, 1.0e-32, 1.0e+100), np.float32)
    cl.enqueue_copy(queue, AF_buf,    AF)
    batch = CELLS
    
    if (isize>NSTOCH):  # this size treated with equilibrium temperature approximation
        if (S_FRAC[isize]<1.0e-30): continue  # empty size bin
        KABS   =  SK_ABS[isize,:] / (GD*S_FRAC[isize])
        # Prepare lookup table between energy and temperature
        t1     =  time.time()  # this is fast (<1sec) and can be done by host
        TSTEP  =  1600.0/NIP    # hardcoded upper limit 1600K for the maximum dust temperatures
        TT     =  zeros(NIP, float64)
        Eout   =  zeros(NIP, float64)
        DF     =  FREQ[2:] - FREQ[:(-2)]  #  x[i+1] - x[i-1], lengths of intervals for Trapezoid rule
        for i in range(NIP):
            TT[i]   =  4.0 + TSTEP*i
            TMP     =  FACTOR * KABS * PlanckSafe(asarray(FREQ, float64), TT[i]) # FACTOR * energy
            # Trapezoid integration TMP over FREQ frequencies
            res     =  TMP[0]*(FREQ[1]-FREQ[0]) + TMP[-1]*(FREQ[-1]-FREQ[-2]) # first and last step
            res    +=  sum(TMP[1:(-1)]*DF)  # the sum over the rest of TMP*DF
            Eout[i] =  4.0*np.pi * 0.5 * res   # energy corresponding to TT[i]
        # Calculate the inverse mapping    Eout -> TTT
        Emin, Emax = Eout[0], Eout[NIP-1]*0.9999
        # E ~ T^4  => use logarithmic sampling
        kE     = (Emax/Emin)**(1.0/(NIP-1.0))  # E[i] = Emin*pow(kE, i)
        oplgkE = 1.0/log10(kE)
        ip     = interp1d(Eout, TT)           # (linear) interpolation from energy to temperature
        print('Eout = %10.3e ... %10.3e' % (Emin, Emax))
        TTT    = asarray(ip(Emin * kE**arange(NIP)), np.float32)
        print("Mapping E -> T calculated on host: %.3f seconds" % (time.time()-t1))
        if (0):
            loglog(TTT, Emin * kE**arange(NIP), 'k-')
            SHOW()
            sys.exit()            
        # Calculate temperatures on device
        #   ABSORBED * AF  integrated in the kernel over frequency -> Ein
        #   kernel will get ABSORBED, will do integration and table lookup to get T
        #   Because of the large amount of data, kernel calls for GLOBAL cells at a time ...
        #   the emission will be also calculated already for the BATCH cells
        cl.enqueue_copy(queue, TTT_buf,  TTT)    # NIP elements
        cl.enqueue_copy(queue, KABS_buf, KABS)   # NFREQ elements
        T = zeros(BATCH, np.float32)        
        for icell in range(0, CELLS, BATCH):     # T_buf is being updated for GLOBAL cells
            b       =  min(icell+BATCH, CELLS)
            tmp_abs = ABSORBED[icell:b, :]*AF
            tmp_abs = asarray(tmp_abs, np.float32)
            cl.enqueue_copy(queue, ABS_buf,  tmp_abs)  # BATCH*NFREQ elements
            kernel_T(queue, [BATCH,], [LOCAL,], icell, kE, oplgkE, Emin,
            FREQ_buf, KABS_buf, TTT_buf, ABS_buf, T_buf, EMIT_buf)
            # Add emission to the final array
            cl.enqueue_copy(queue, EMIT, EMIT_buf) ;  # emission for <= GLOBAL cells
            for i in range(icell,b):
                EMITTED[i, :] += EMIT[i-icell,:] * GD * S_FRAC[isize]        
        continue
    

    if (isize==0):
        # The rest is for stochastically heated grains
        noIw  = np.fromfile(FP, np.int32, 1)[0]
        Iw    = np.fromfile(FP, np.float32, noIw)   # [ windex ], loop l=[0,NE-1[, u=[l+1,NE[
        L1    = np.fromfile(FP, np.int32,   NE*NE)  # [ l*NE + u ]
        L2    = np.fromfile(FP, np.int32,   NE*NE)
        Tdown = np.fromfile(FP, np.float32, NE)
        EA    = np.fromfile(FP, np.float32, NE*NFREQ)
        Ibeg  = np.fromfile(FP, np.int32,   NFREQ)    

    cl.enqueue_copy(queue, Iw_buf,    Iw)
    cl.enqueue_copy(queue, L1_buf,    L1)
    cl.enqueue_copy(queue, L2_buf,    L2)
    cl.enqueue_copy(queue, Tdown_buf, Tdown)
    cl.enqueue_copy(queue, EA_buf,    EA)
    cl.enqueue_copy(queue, Ibeg_buf,  Ibeg)        
        
    t00 = time.time()            
    cl.enqueue_copy(queue, ABS_buf,  ABSORBED[0:CELLS,:])
    queue.finish()            
    if (WITH_X):
        DoSolve(queue, [GLOBAL,], [LOCAL,], 
        batch,     isize,
        Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
        Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf,   X_buf)
    else:
        DoSolve(queue, [GLOBAL,], [LOCAL,], 
        batch,     isize,
        Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
        Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf)

        
    if (isize<(NSIZE-1)): # read already the data for the next size, calculate new AF == next size
        noIw  = np.fromfile(FP, np.int32, 1)[0]
        Iw    = np.fromfile(FP, np.float32, noIw)   # [ windex ], loop l=[0,NE-1[, u=[l+1,NE[
        L1    = np.fromfile(FP, np.int32,   NE*NE)  # [ l*NE + u ]
        L2    = np.fromfile(FP, np.int32,   NE*NE)
        Tdown = np.fromfile(FP, np.float32, NE)
        EA    = np.fromfile(FP, np.float32, NE*NFREQ)
        Ibeg  = np.fromfile(FP, np.int32,   NFREQ)    
        
    # now get the results from the kernel, for the current size
    cl.enqueue_copy(queue, emit, EMIT_buf)  # batch*NFREQ
    EMITTED[0:CELLS,:] += emit[0:CELLS,:]        # contribution of current dust, current size
    
    ##   A2E_new.py ==  64.4 seconds
    ##   A2E.py     ==  55.2 seconds !!   faster in batches than in kernel loops !!
    
    print('   SIZE %2d/%2d  %.3e s/cell/size' % (isize, NSIZE, (time.time()-t00)/CELLS))
            
DT = time.time() - t0
print('  %.3f SECONDS' % DT)
REF_RATE = 1520.0
REF_RATE =  100.7
REF_RATE =  1.0/1.054e-2
print('  %4d  -- %.3e SECONDS PER CELL  -- %8.3f CELLS PER SECOND -- x %5.2f' % (CELLS, DT/CELLS,  CELLS/DT, (CELLS/DT)/REF_RATE))

