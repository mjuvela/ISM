#!/usr/bin/python3

import os, sys
HOMEDIR = os.path.expanduser('~/')
sys.path.append(HOMEDIR+'/starformation/SOC/')

import pyopencl as cl
from   scipy.interpolate import interp1d
from   matplotlib.pylab import *
from   ASOC_aux import *

"""
Usage:
    A2E_pyCL  dump absorbed.data emitted.data
Using solver.data dumped from A2E, convert absorbed.data to emitted.data.

2018-12-13:
    one dump file per dust --- A2E_pyCL IS FOR A SINGLE DUST !!
    relative Kabs for different dusts and different sizes is to be calculated
    based on data directly in the dump file

solver.data:
    NFREQ
    FREQ[NFREQ]
    GRAIN_DENSITY
    NSIZE
    S_FRAC[NSIZE]
    NE
    SK_ABS[NSIZE, NFREQ]
    --- for each size ...:
        num_Iw
        Iw[NE*NE*NFREQ]
        L1[NE*NE]
        L2[NE*NE]
        Tdown[NE]
        EA[NE*NFREQ]
        Ibeg[NFREQ]
        ---- this dropped ABS_FRAC[NFREQ] =  SK_ABS[]/ (K_ABS[ifreq]*S_FRAC[size]*GRAIN_DENSITY)
    -1
absorbed.data, emitted.data:
    CELLS, NFREQ, { values }

"""

ICELL     =  0
GPU       =  0
        
        
if (len(sys.argv)<4):
    print("")
    print("A2E_pyCL  dump  absorbed.data emitted.data [GPU [nstoch]]\n")
    sys.exit()
    
#BATCH =   8192
BATCH =  4096
BATCH =  8192     # RESTRICTED BY MEMORY USAGE !!!
#BATCH  = 2560*2
#BATCH  = 16384
#BATCH =  4096
#BATCH =  1024
#BATCH  = 2048
#BATCH =  512
#BATCH =  128
#BATCH =  64*64   # each work item takes one column of cells in 64x64 cloud
NSTOCH = 999
if (len(sys.argv)>4):
    GPU  = int(sys.argv[4])
    if (len(sys.argv)>5):
        NSTOCH = int(sys.argv[5])


    
# Read solver dump file
FP     =  open(sys.argv[1], 'rb')      # dump
NFREQ  =  np.fromfile(FP, np.int32, 1)[0]                                # NFREQ
FREQ   =  np.fromfile(FP, np.float32, NFREQ)                             # FREQ[NFREQ]
GD     =  np.fromfile(FP, np.float32, 1)[0]                              # GRAIN_DENSITY
NSIZE  =  np.fromfile(FP, np.int32, 1)[0]                                # NSIZE
S_FRAC =  np.fromfile(FP, np.float32, NSIZE)                             # S_FRAC
NE     =  np.fromfile(FP, np.int32, 1)[0]                                # NE
SK_ABS =  np.fromfile(FP, np.float32, NSIZE*NFREQ).reshape(NSIZE, NFREQ) # SK_ABS[NSIZE, NFREQ]
## SK_ABS =  asarray(SK_ABS, float64)
K_ABS  =  sum(SK_ABS, axis=0)

print('S_FRAC', S_FRAC)
print('NE    ', NE)

if (0):
    clf()
    subplot(221)
    plot(S_FRAC)
    title('S_FRAC')
    subplot(222)
    imshow(log10(SK_ABS), aspect='auto')
    title('lg SK_ABS')
    colorbar()
    subplot(223)
    plot(log10(sum(SK_ABS, axis=0)), 'x')
    plot(arange(NFREQ), -24*ones(NFREQ), 'r+')
    title('lg sum(SK_ABS)')
    print(SK_ABS)
    SHOW()
    sys.exit()

# since 2018-12-29 also SOC absorbed.data starts with [ cells, nfreq ] only
fp     =  open(sys.argv[2], 'rb')      # absorbed
CELLS, NFREQ = np.fromfile(fp, np.int32, 2)
fp.close()
    
# Emitted has always 2 int header
fp    =  open(sys.argv[3], 'wb')      # emitted
asarray([CELLS, NFREQ], np.int32).tofile(fp)  # 2018-12-29 -- dropped levels from the python files !!
fp.close()
   
print("CELLS %d, NFREQ %d" % (CELLS, NFREQ))

# print 'data %s, from %s to %s' % (sys.argv[1], sys.argv[2], sys.argv[3])
#  2018-10-15 ADDED LEVELS AS THE FIRST INTEGER IN THE ASORBED AND EMITTED FILES
ABSORBED  = np.memmap(sys.argv[2], dtype='float32', mode='r+',  shape=(CELLS, NFREQ), offset=8)
if (1):  # NECESSARY TO AVOID HUGE EMISSION IN RARE CELLS ???
    ABSORBED[:, NFREQ-1] = np.clip(ABSORBED[:, NFREQ-1], 0.0, 0.2*ABSORBED[:, NFREQ-2])
EMITTED   = np.memmap(sys.argv[3], dtype='float32', mode='r+',  shape=(CELLS, NFREQ), offset=8)
EMITTED[:,:] = 0.0


t0 = time.time()
platform, device, context, queue = None, None, None, None
LOCAL = 16 
for iplatform in range(3):
    try:
        platform = cl.get_platforms()[iplatform]
        if (GPU>0):
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
GLOBAL      =  max([BATCH, 32*LOCAL])
if (GLOBAL%32!=0):
    GLOBAL  = (GLOBAL/32+1)*32
context     =  cl.Context(device)
queue       =  cl.CommandQueue(context)
mf          =  cl.mem_flags
NIP         =  30000  # number of interpolation points for the lookup tables (equilibrium dust)
OPT         =  "-D NE=%d -D LOCAL=%d -D NFREQ=%d -D CELLS=%d -D NIP=%d -D FACTOR=%.4ef" \
               % (NE, LOCAL, NFREQ, CELLS, NIP, FACTOR)
if (1):
    OPT += ' -cl-fast-relaxed-math'  # this is fastest for both CPU and GPU
    # OPT  += " -cl-opt-disable"  # slower by x3 on CPU, x8 on GPU !!
               
source      =  open(HOMEDIR+"/starformation/SOC/kernel_A2E_pyCL_2.c").read()
program     =  cl.Program(context, source).build(OPT)
Iw_buf      =  cl.Buffer(context, mf.READ_ONLY,  NE*NE*NFREQ*4)
L1_buf      =  cl.Buffer(context, mf.READ_ONLY,  NE*NE*4)
L2_buf      =  cl.Buffer(context, mf.READ_ONLY,  NE*NE*4)
Tdown_buf   =  cl.Buffer(context, mf.READ_ONLY,  NE*4)
EA_buf      =  cl.Buffer(context, mf.READ_ONLY,  NE*NFREQ*4)
Ibeg_buf    =  cl.Buffer(context, mf.READ_ONLY,  NFREQ*4)
AF_buf      =  cl.Buffer(context, mf.READ_ONLY,  NFREQ*4)
ABS_buf     =  cl.Buffer(context, mf.READ_ONLY,  BATCH*NFREQ*4)
EMIT_buf    =  cl.Buffer(context, mf.READ_WRITE, BATCH*NFREQ*4)
A_buf       =  cl.Buffer(context, mf.READ_WRITE, BATCH*(int((NE*NE-NE)/2))*4) # lower triangle only
X_buf       =  cl.Buffer(context, mf.READ_WRITE, BATCH*NE*4)


if (NSTOCH<NSIZE):    # Prepare to solve equilibrium temperature emission
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
            
            
DoSolve =  program.DoSolve4
#                              batch     isize     Iw    L1    L2    Tdown EA    Ibeg  AF    ABS   EMIT  LL  
DoSolve.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None, None, None, None, None, None, None, None])

emit = zeros((BATCH, NFREQ), np.float32)


if (0): # TESTING
    print("***** TESTING *****") 
    CELLS = 128



t0 = time.time()    
for isize in range(NSIZE):        

    thost, tdevice = 0.0, 0.0
    ts    = time.time()

    AF    = asarray(SK_ABS[isize,:], float64) / asarray(K_ABS[:], float64)  # => E per grain
    AF   /= S_FRAC[isize]*GD  # "invalid value encountered in divide"
    AF    = asarray(np.clip(AF, 1.0e-32, 1.0e+100), np.float32)
    if (1):
        mm = np.nonzero(~isfinite(AF))
        AF[mm] = 1.0e-30

        
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
            TMP     =  FACTOR * KABS * PlanckSafe(asarray(FREQ, float64), TT[i])
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
        # Upload data for the current grain size
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
    
        
    
    # The rest is for stochastically heated grains
    #  ... interleaving read and transfer => no effect on run time
    cl.enqueue_copy(queue, AF_buf,    AF)
    noIw  = np.fromfile(FP, np.int32, 1)[0]
    Iw    = np.fromfile(FP, np.float32, noIw)   # [ windex ], loop l=[0,NE-1[, u=[l+1,NE[
    cl.enqueue_copy(queue, Iw_buf,    Iw)
    L1    = np.fromfile(FP, np.int32,   NE*NE)  # [ l*NE + u ]
    cl.enqueue_copy(queue, L1_buf,    L1)
    L2    = np.fromfile(FP, np.int32,   NE*NE)
    cl.enqueue_copy(queue, L2_buf,    L2)
    Tdown = np.fromfile(FP, np.float32, NE)
    cl.enqueue_copy(queue, Tdown_buf, Tdown)
    if (0):
        Tdown = np.clip(Tdown, 1.0e-32, 1.0e30)    
    EA    = np.fromfile(FP, np.float32, NE*NFREQ)
    cl.enqueue_copy(queue, EA_buf,    EA)
    Ibeg  = np.fromfile(FP, np.int32,   NFREQ)    
    cl.enqueue_copy(queue, Ibeg_buf,  Ibeg)        
    queue.finish()

    thost += time.time() - ts
    
    # Loop over the cells, BATCH cells per kernel call
    for icell in range(0, CELLS, BATCH):        
        t00   = time.time()            
        batch = min([BATCH, CELLS-icell])  # actual number of cells
        cl.enqueue_copy(queue, ABS_buf,  ABSORBED[icell:(icell+BATCH),:])
        queue.finish()            
        ts      =  time.time()
        thost  +=  ts - t00
        DoSolve(queue, [GLOBAL,], [LOCAL,], 
        batch, isize, Iw_buf, L1_buf, L2_buf, Tdown_buf, EA_buf, Ibeg_buf, AF_buf, ABS_buf, EMIT_buf, A_buf)
        queue.finish()            
        tdevice += time.time() - ts
        ts = time.time()
        cl.enqueue_copy(queue, emit, EMIT_buf)  # batch*NFREQ
        EMITTED[icell:(icell+batch),:] += emit[0:batch,:]        # contribution of current dust, current size

        if (icell==0):
            print('   SIZE %2d/%2d  icell %6d/%6d  %7.2f s/%d %.3e s/cell/size  %.2f cells/second' % \
            (isize, NSIZE, icell, CELLS, time.time()-t00, batch, (time.time()-t00)/batch, batch/(time.time()-t00)))
        thost += time.time()- ts
                
    print("HOST %.3f seconds,  DEVICE %.3f seconds" % (thost, tdevice))

    
DT = time.time() - t0
print('  %.3f SECONDS' % DT)
print('  %4d  -- %.3e SECONDS PER CELL  -- %8.3f CELLS PER SECOND' % (CELLS, DT/CELLS,  CELLS/DT))

