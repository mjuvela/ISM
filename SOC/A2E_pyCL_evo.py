#!/usr/bin/python3

import os, sys
HOMEDIR = os.path.expanduser('~/')
sys.path.append(HOMEDIR+'/starformation/SOC/')

import pyopencl as cl
from   scipy.interpolate import interp1d
from   matplotlib.pylab import *
from   SOC_aux import *

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
EVO2      =  2
        
if (len(sys.argv)<4):
    print("")
    print("A2E_pyCL  dump  absorbed.data emitted.data [GPU [nstoch]]\n")
    sys.exit()
    

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
K_ABS  =  sum(SK_ABS, axis=0)


# since 2018-12-29 also SOC absorbed.data starts with [ cells, nfreq ] only
fp     =  open(sys.argv[2], 'rb')      # absorbed
CELLS, NFREQ = np.fromfile(fp, np.int32, 2)
fp.close()
    
# Emitted has always 2 int header
fp    =  open(sys.argv[3], 'wb')      # emitted
asarray([CELLS, NFREQ], np.int32).tofile(fp)  # 2018-12-29 -- dropped levels from the python files !!
fp.close()
print("CELLS %d, NFREQ %d" % (CELLS, NFREQ))


#  2018-10-15 ADDED LEVELS AS THE FIRST INTEGER IN THE ASORBED AND EMITTED FILES
ABSORBED  = np.memmap(sys.argv[2], dtype='float32', mode='r+',  shape=(CELLS, NFREQ), offset=8)
if (1):  # NECESSARY TO AVOID HUGE EMISSION IN RARE CELLS ???
    ABSORBED[:, NFREQ-1] = np.clip(ABSORBED[:, NFREQ-1], 0.0, 0.2*ABSORBED[:, NFREQ-2])
EMITTED   = np.memmap(sys.argv[3], dtype='float32', mode='r+',  shape=(CELLS, NFREQ), offset=8)
EMITTED[:,:] = 0.0


platform, device, context, queue = None, None, None, None
LOCAL = 8
for iplatform in range(3):
    try:
        platform = cl.get_platforms()[iplatform]
        if (GPU>0):
            device   = platform.get_devices(cl.device_type.GPU)
            LOCAL    = 32
        else:
            device   = platform.get_devices(cl.device_type.CPU)
            LOCAL    =  8
        context  = cl.Context(device)
        queue    = cl.CommandQueue(context)
        break
    except:
        pass
    
# one work group per cell each cell in the batch
if (EVO2):
    BATCH   =  4096
    GLOBAL  =  NSIZE*LOCAL*8          # one work group per size, can have integer multiple of NSIZE*LOCAL
    #   GLOBAL = 10*64*4 = 2560,  4*LOCAL = 256 work items per size
    # multiplier 8 =>  8*64 = 512 work items per size, covering BATCH=1024 cells
else:    
    BATCH   =  128                    #  == number of work groups
    GLOBAL  =  LOCAL*BATCH            # one work group per cell, local work items loop over sizes
context     =  cl.Context(device)
queue       =  cl.CommandQueue(context)
mf          =  cl.mem_flags
NIP         =  30000  # number of interpolation points for the lookup tables (equilibrium dust)
OPT         =  "-D NE=%d -D NFREQ=%d -D CELLS=%d -D NIP=%d -D NSIZE=%d" % (NE, NFREQ, CELLS, NIP, NSIZE)
OPT        +=  " -D GLOBAL=%d -D LOCAL=%d" % (GLOBAL, LOCAL)
OPT        +=  " -cl-fast-relaxed-math"
source      =  open(HOMEDIR+"/starformation/SOC/kernel_A2E_pyCL_evo.c").read()
program     =  cl.Program(context, source).build(OPT)

"""
For ***simplicity*** we allocate Iw_buf for the full NSIZE*NE*NE*NFREQ although in
the input file noIw << NE*NE*NFREQ for each size ... the array can be easily ~1 GB in size!
"""

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
            
            
if (EVO2):
    DoSolve =  program.DoSolveEvo2
else:
    DoSolve =  program.DoSolveEvo
    
#                              0         1     2     3     4     5     6     7     8     9     10  
#                              batch     Iw    L1    L2    Tdown EA    Ibeg  AF    ABS   EMIT  L   
DoSolve.set_scalar_arg_dtypes([np.int32, None, None, None, None, None, None, None, None, None, None ])
                                
# Send to kernel Iw, L1, L2, Tdown, EA, Ibeg, AF ---- FOR ALL SIZES

Iw_buf       =  cl.Buffer(context, mf.READ_ONLY,  NSIZE*NE*NE*NFREQ*4)
L1_buf       =  cl.Buffer(context, mf.READ_ONLY,  NSIZE*NE*NE*4)
L2_buf       =  cl.Buffer(context, mf.READ_ONLY,  NSIZE*NE*NE*4)
Tdown_buf    =  cl.Buffer(context, mf.READ_ONLY,  NSIZE*NE*4)
EA_buf       =  cl.Buffer(context, mf.READ_ONLY,  NSIZE*NE*NFREQ*4)
Ibeg_buf     =  cl.Buffer(context, mf.READ_ONLY,  NSIZE*NFREQ*4)
AF_buf       =  cl.Buffer(context, mf.READ_ONLY,  NSIZE*NFREQ*4)

ABS_buf      =  cl.Buffer(context, mf.READ_ONLY,  BATCH*NFREQ*4)

if (EVO2):
    EMIT_buf =  cl.Buffer(context, mf.READ_WRITE, BATCH*NFREQ*NSIZE*4)
    emit     =  zeros((BATCH, NFREQ, NSIZE), np.float32)    
else:
    EMIT_buf =  cl.Buffer(context, mf.READ_WRITE, BATCH*NFREQ*NSIZE*4)
    emit     =  zeros((BATCH, NFREQ), np.float32)
    
A_buf        =  cl.Buffer(context, mf.READ_WRITE, GLOBAL*(int((NE*NE-NE)/2))*4) # lower triangle only

Iw_all       =  zeros((NSIZE, NE*NE*NFREQ), float32)
L1_all       =  zeros((NSIZE, NE*NE),       int32)
L2_all       =  zeros((NSIZE, NE*NE),       int32)
Tdown_all    =  zeros((NSIZE, NE),          float32)
EA_all       =  zeros((NSIZE, NE*NFREQ),    float32)
Ibeg_all     =  zeros((NSIZE, NFREQ),       int32)
AF_all       =  zeros((NSIZE, NFREQ),       float32)



print("Iw_all %.3f GB" % (Iw_all.nbytes/1024.0/1024.0/1024.0))

for isize in range(NSIZE):        
    AF    = asarray(SK_ABS[isize,:], float64) / asarray(K_ABS[:], float64)  # => E per grain
    AF   /= (S_FRAC[isize]*GD+1.0e-50)  # "invalid value encountered in divide"
    AF    = asarray(np.clip(AF, 1.0e-32, 1.0e+100), np.float32)
    if (1):
        mm = np.nonzero(~isfinite(AF))
        AF[mm] = 1.0e-30
    noIw  = np.fromfile(FP, np.int32, 1)[0]
    Iw    = np.fromfile(FP, np.float32, noIw)   # [ windex ], loop l=[0,NE-1[, u=[l+1,NE[
    L1    = np.fromfile(FP, np.int32,   NE*NE)  # [ l*NE + u ]
    L2    = np.fromfile(FP, np.int32,   NE*NE)
    Tdown = np.fromfile(FP, np.float32, NE)
    EA    = np.fromfile(FP, np.float32, NE*NFREQ)
    Ibeg  = np.fromfile(FP, np.int32,   NFREQ)    
    #
    Iw_all[isize, 0:noIw]  =  Iw
    L1_all[isize, :]       =  L1
    L2_all[isize, :]       =  L2
    Tdown_all[isize,:]     =  Tdown
    EA_all[isize,:]        =  EA
    Ibeg_all[isize,:]      =  Ibeg
    AF_all[isize,:]        =  AF
    
cl.enqueue_copy(queue, Iw_buf, Iw_all)
cl.enqueue_copy(queue, L1_buf, L1_all)
cl.enqueue_copy(queue, L2_buf, L2_all)
cl.enqueue_copy(queue, Tdown_buf, Tdown_all)
cl.enqueue_copy(queue, EA_buf, EA_all)
cl.enqueue_copy(queue, Ibeg_buf, Ibeg_all)
cl.enqueue_copy(queue, AF_buf, AF_all)


CELLS = 32768
# CELLS = 128
        
# Loop over the cells, BATCH cells per kernel call
t0 = time.time()
for icell in range(0, CELLS, BATCH):        
    batch = min([BATCH, CELLS-icell])  # actual number of cells <= BATCH
    cl.enqueue_copy(queue, ABS_buf,  ABSORBED[icell:(icell+BATCH),:])
    DoSolve(queue, [GLOBAL,], [LOCAL,], batch,
        Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf)
    queue.finish()
    cl.enqueue_copy(queue, emit, EMIT_buf)  # batch*NFREQ
    if (EVO2):
        EMITTED[icell:(icell+batch),:] += sum(emit[0:batch,:,:], axis=2)
    else:
        EMITTED[icell:(icell+batch),:] += emit[0:batch,:]
            
DT = time.time() - t0
print('  %.3f SECONDS' % DT)
print('  %7d CELLS, %.3e SECONDS PER CELL, %8.3f CELLS PER SECOND' % (CELLS, DT/CELLS,  CELLS/DT))
