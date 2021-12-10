#!/usr/bin/python3

import os, sys

# HOMEDIR = os.path.expanduser('~/')
# sys.path.append(HOMEDIR+'/starformation/SOC/')
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)

if (0):
    from   MJ.mjDefs import *
else:
    import os, sys
    import numpy as np
    
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

    
2019-10-28:
    - python == A2E.cpp
    - drop two largest frequencies (ABSORBED=0): python  == julia (largest grains remain bin 0)
    - GD, zero none, like the A2E.cpp and python without zeroing the highest frequency bins !!
      GD, zero two, GD remains the same !! 
      if zero six first: GD vs. CPP =  
    - are python and A2E.cpp wrong if largest frequency bins are included ?
    - why largest size grains remain in bin 0 whenever using A2E.py, whether using the solver
      file from Cpp, from python, or from julia == something in A2E.py script ??
      or rounding error caused by the use of the file ?? try double precision file ??
      
    ... there are thus THREE different solutions
    (1) cpp, python, all frequencies  AND GD irrespective of the zeroing of highest frequency bins 
         == N(E) moves towards lower bins for larger sizes, ends up in bin 0 for the largest size
    (2) julia always and .... cpp and python *without* two highest frequencies
         == N(E) moves to higher bins for larger grains (*BUT* A2E.py largest size = bin 0 when S_FRAC=1e-42)
         
    *** something in the handling of the TWO HIGHEST FREQUENCY BINS  (julia vs. python and cpp)
        One could assume that A2E.py and cpp are wrong because of the large Iw values given for those last bins
        AND because their results are the same as with julia as soon as absorptions in those two frequency bins 
        are set to zero.
        But why shoud Guhathagurta-Draine agree with that ~erroneous solution, irrespective of the
        zeroing of absorptions ??
        Try GD again with full 25 sizes... because it assumes narrow size bins
        - nothing zeroed         -- as default cpp (without anything zerod), N(E) down with size
        - zero 2 absorption bins -- idem
        - zero 4 absorption bins -- idem
        - zero 6 absorption bin  -- idem

        
    Make new dust with ADDED TWO HIGHER FREQUENCIES and set absorptions there to zero -- should be
    the same as previous cases without zeroing any absorptions!!!!!!!!!
    - python  N(E) increase with size == suggests that julia was correct !!
      cpp     N(E) increase with size == suggests that julia was correct !!
      julia   N(E) increase with size
      .... it is only GD that is still N(E) decreasing with size --- perhaps THAT IS ALSO INCORRECT ???
      force DE in to Ef range => no change
      no interpolation of two points => no change
        
"""

ICELL     =  9
GPU       =  0
t_beg     =  time.time()        
        
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
    show()
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
if (0):
    ABSORBED  = np.memmap(sys.argv[2], dtype='float32', mode='r+',  shape=(CELLS, NFREQ), offset=8)
else:
    ABSORBED = fromfile(sys.argv[2], float32)[2:].reshape(CELLS, NFREQ)

if (0):  # NECESSARY TO AVOID HUGE EMISSION IN RARE CELLS ???
    ABSORBED[:, NFREQ-1] = np.clip(ABSORBED[:, NFREQ-1], 0.0, 0.2*ABSORBED[:, NFREQ-2])
"""
Actually, the A2E.cpp and A2E_pre.py solver files results in large positive and negative Iw values 
for the last two frequency bins!! That must be a bug => cannot use last two nabs bins !!!
"""
if (0):
    ABSORBED[:, NFREQ-1] = 0.0
    ABSORBED[:, NFREQ-2] = 0.0
    ABSORBED[:, NFREQ-3] = 0.0
    ABSORBED[:, NFREQ-4] = 0.0
    ABSORBED[:, NFREQ-5] = 0.0
    ABSORBED[:, NFREQ-6] = 0.0
    
EMITTED   = np.memmap(sys.argv[3], dtype='float32', mode='r+',  shape=(CELLS, NFREQ), offset=8)
EMITTED[:,:] = 0.0



def PlanckSafe(f, T):  # Planck function
    # Add clip to get rid of warnings
    H_CC    =  7.372496678e-48    #   PLANCK/C_LIGHT^2
    return 2.0*H_CC*f*f*f / (exp(np.clip(H_K*f/T,-100,+100))-1.0)



def AnalyseMatrix(iW, L1, L2, Tdown, EA, ABSORBED, AF):
    """
    Make the linear equation matrix on the host side and analyse it...
    perhaps to compute a preconditioning matrix.
    """
    NE    = len(Tdown)
    CELLS = ABSORBED.shape[0]
    NFREQ = ABSORBED.shape[1]
    # Make the matrix for the mean ABSORBED vector
    if (1):
        MABS  = mean(ABSORBED, axis=0)
    else:
        icell = [ 0, 4210 ][1]
        MABS = ABSORBED[icell,:]
    A     = zeros((NE, NE), float64)
    TMP   = zeros(NFREQ, float64)
    for i in range(0, NFREQ):
        TMP[i] = np.clip(AF[i] * MABS[i], 0.0, 1.0e30)
    TMP[NFREQ-1] = np.clip(TMP[NFREQ-1], 0.0, 0.3*TMP[NFREQ-2])
    iw_index = 0
    for l in range(0, NE-1):
        for u in range(l+1, NE):
            I = 0.0
            for i in range(L1[l*NE+u], L2[l*NE+u]+1):
                I         +=  TMP[i] * Iw[iw_index]
                iw_index  +=  1
            A[u,l] = np.clip(I, 1.0e-30, 1.0e30)
            # A[u,l] = I
    # Cooling - first elements above the diagonal
    for j in range(1, NE):
        A[j-1, j] = Tdown[j]
    # Diagonal values = -sum(column)
    for i in range(NE):
        s = 0.0
        for j in range(i+1, NE):  # loop over lower triangle
            s += A[j,i]
        if (i>0):
            s += Tdown[i]         # off-diagonal
        if (0):
            # clip at 0.0  N(E) has plateau at 1e-25, cond 1.8e8, 1.8e8 after normalisation of diag elements
            # clip at 1.0e-20,      plateau at 1e-16, cond 1.8e8, 8e20 !! ... could clip A[i,i] even later
            # clip at 1.0e-15       plateau at 5e-12, cond 1.8e8, 8e15
            # clip only A[0,0] at 1e-15 plateau at 1e-11, cond    8e15
            # clip only A[0,0] at 1e-20 plateau at 5e-17, cond    8e20
            A[i,i] = -np.clip(s, 1.0e-20, 1e30)
        else:
            A[i,i] = -s
    
    if (1):  # Replace last equation
        A[NE-1,:] = 1.0
    # Calculate the solution
    b       =  zeros(NE, float64)
    b[NE-1] =  1.0
    x       =  np.linalg.solve(A, b)
    n       =  A.shape[0]
    if (0):
        # is A diagonal dominant??
        DD  =   zeros(n, float64)
        for i in range(n): 
            DD[i] = abs(A[i,i]) / ( sum(abs(A[i,:])) - abs(A[i,i]) )
        semilogy(arange(n), DD, 'x')
        show()
        sys.exit()
    subplot(221)
    imshow(A, origin='upper')
    colorbar()
    ax3     =  subplot(223)
    loglog(np.linspace(1, n+1, n), x, 'b--')
    #########################################
    w       =  0.5
    # A, b    =  A[0:-5, 0:-5], b[0:-5]
    # A, b    =  A[5:, 5:], b[5:]
    # A, b    =  A[0:256, 0:256], b[0:256]
    n       =  A.shape[0]
    # condition number of the original matrix
    c1      =  cond(A)
    # iteration matrix is T = -(L+D)^-1 * U
    L       =  tril(A, k=-1)
    D       =  diag(diag(A))
    U       =  triu(A, k=1)
    T       =  matmul(inv(w*L+D), (1.0-w)*D-w*U)
    lambda1 =  max(ravel(eig(T)[0]))
    print("Original matrix cond = %.3e, spectral radius %10.3e" % (c1, lambda1))
    # convert the matrix to one with ones on the diagonal
    for i in range(A.shape[0]):
        A[i,:] /=  A[i,i]
    c2      =  cond(A)
    L       =  tril(A, k=-1)
    D       =  diag(diag(A))
    U       =  triu(A, k=1)
    T       =  matmul(inv(w*L+D), (1.0-w)*D-w*U)
    lambda2 =  max(ravel(eig(T)[0]))
    print("Jacobi conditioned  matrix cond = %.3e, spectral radius %10.3e" % (c2, lambda2))
    x       = np.linalg.solve(A, b)
    loglog(np.linspace(1, n+1, n), x, 'r:', lw=2)
    if (1):
        # is A diagonal dominant??
        clf()
        DD  =   zeros(n, float64)
        for i in range(n): 
            DD[i] = abs(A[i,i]) / ( sum(abs(A[i,:])) - abs(A[i,i]) )
        semilogy(arange(n), DD, 'x')
        show()
        sys.exit()
    # Calculate inverse matrix
    AI  = inv(A)
    sys.exit()
    ax4 = subplot(224)
    imshow(AI, origin='upper')
    colorbar()
    show()
    sys.exit()
    

    
    
    
if (0):
    print("***** TESTING !!!! ****") 
    CELLS = 128

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
ARGS        =  "-D NE=%d -D LOCAL=%d -D NFREQ=%d -D CELLS=%d -D NIP=%d -D FACTOR=%.4ef" \
               % (NE, LOCAL, NFREQ, CELLS, NIP, FACTOR)
if (1):
    ARGS   += ' -cl-fast-relaxed-math'  # this is fastest for both CPU and GPU
    # ARGS  += " -cl-opt-disable"  # slower by x3 on CPU, x8 on GPU !!
    # ARGS   += ' -cl-nv-opt-level 2'
               
source      =  open(INSTALL_DIR+"/kernel_A2E_pyCL_L2.c").read()
program     =  cl.Program(context, source).build(ARGS)
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
            
            

### SOLVER 
# DoSolve a=  program.DoSolveGS  # Gauss-Seidel iterative
# DoSolve =  program.DoSolve4   # Explicit solver
DoSolve =  program.DoSolve4X  # more stable == A2E version
    
#                               batch     NFREQ     NE
DoSolve.set_scalar_arg_dtypes([np.int32, np.int32, 
                                None, None, None, None, None, None, None, None, None, None, None ])

X0_GOOD_A = zeros((BATCH, NE), np.float32)    
X0_GOOD_V = zeros(NE, np.float32)    


A2E_ARRAY  = []   # Saved by A2E.cpp  --- [NSIZE,NE], single cell !!
PREV_ARRAY = []   # Saved by this script, previous run.



ARRAY0    = zeros((NSIZE, NE), np.float32)
ARRAY1    = zeros((NSIZE, NE), np.float32)
X         = zeros((BATCH, NE), np.float32)

# Initial values, if better are not read from a file
X0        = 1.0 / (0.1*((NE/3)-arange(NE))**2.0+1.0)
X0_BAD    = 0.0*X0

try:
    print("READ dump.ARRAY[%d,%d]" % (NSIZE, NE))
    A2E_ARRAY = np.fromfile('dump.ARRAY', np.float32)
    A2E_ARRAY.shape = (NSIZE,NE)  # single cell, all NE, all sizes
    print("READ dump.ARRAY !!!!")
    A2E_ARRAY = np.clip(A2E_ARRAY, 1.0e-35, 1.1)
    print("READ dump.ARRAY -- RESHAPED !!!!")
    if (0):
        for isize in range(NSIZE):
            A2E_ARRAY[isize,:] /= sum(A2E_ARRAY[isize,:])
    X0        = A2E_ARRAY[0      ,:].copy()        # good initial guess --- exact for isize==0
    X0_BAD    = A2E_ARRAY[NSIZE//2,:].copy()  # bad initial guess
except:
    print("Did not find dump.ARRAY = old results !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    A2E_ARRAY = zeros((NSIZE,NE), np.float32)
    for ii in range(NSIZE):
        A2E_ARRAY[ii,:] = X0
    
try:
    # Previous run
    PREV_ARRAY = np.fromfile('previous.ARRAY', np.float32).reshape(NSIZE,NE)
    print("previous.ARRAY  ok")
    # X0      = PREV_ARRAY[0,:]
    # X0_BAD  = PREV_ARRAY[NSIZE/2,:]  # bad initial guess
except:
    print("Did not find previous.ARRAY")
    PREV_ARRAY = zeros((NSIZE,NE), np.float32)
    for ii in range(NSIZE):
        PREV_ARRAY[ii,:] = X0

X[:,:] = X0     
cl.enqueue_copy(queue, X_buf,  X)
queue.finish()


emit = zeros((BATCH, NFREQ), np.float32)

t0 = time.time()    
for isize in range(NSIZE):        

    thost, tdevice = 0.0, 0.0
    ts    = time.time()
    # print("isize %d, S_FRAC = %.3e" % (isize, S_FRAC[isize]))
    
    ### AF    = np.fromfile(FP, float32, NFREQ)  -->  AF == SK_ABS / (K_ABS*GD*S_FRAC)
    AF    = asarray(SK_ABS[isize,:], float64) / asarray(K_ABS[:], float64)  # => E per grain
    AF   /= S_FRAC[isize]*GD  # "invalid value encountered in divide"
    AF    = asarray(np.clip(AF, 1.0e-32, 1.0e+100), np.float32)
    if (1):
        mm = np.nonzero(~isfinite(AF))
        AF[mm] = 1.0e-30
        
    """
    ABSORBED = number of absorbed photons .... NOT YET ENERGY
    """
    
    # if (isize!=23): continue
    
    
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
            show()
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

    
    if (isize==-10):
        AnalyseMatrix(Iw, L1, L2, Tdown, EA, ABSORBED, AF)

    thost += time.time() - ts
    
    # Loop over the cells, BATCH cells per kernel call
    for icell in range(0, CELLS, BATCH):        
        t00   = time.time()            
        batch = min([BATCH, CELLS-icell])  # actual number of cells
        if (0): # old things for iterative solver
            # @@@
            if (0):    # BAD start -- same init for all sizes
                if (isize==0):
                    cl.enqueue_copy(queue, X_buf,  X0_BAD)
            if (0):    # GOOD start -- vector for all
                # WAS ALL isize; ALL icell
                # vector or full array... no difference?
                if ((isize==0)&(icell<1)):
                    X0_GOOD_V[:] = A2E_ARRAY[isize,:] / sum(A2E_ARRAY[isize,:])
                    cl.enqueue_copy(queue, X_buf,  X0_GOOD_V)
            if (0):   # GOOD start -- full array
                # since first work item will overwrite X ??
                # WORKS IF A2E USES THE SAME ENTHALPY DISCRETISATION ???
                if (icell==0):  # useful when cells in several batches? --- no real effect
                    X0_GOOD_A[:,:] = A2E_ARRAY[isize,:].copy()
                    cl.enqueue_copy(queue, X_buf,  X0_GOOD_A)
            if (0):   # start with previous run, first cell
                X0_GOOD_A[:,:] = PREV_ARRAY[isize,:].copy()
                cl.enqueue_copy(queue, X_buf,  X0_GOOD_A)

        cl.enqueue_copy(queue, ABS_buf,  ABSORBED[icell:(icell+BATCH),:])
        queue.finish()            
        ts      =  time.time()
        thost  +=  ts - t00
        DoSolve(queue, [GLOBAL,], [LOCAL,], 
            batch,     isize,
            Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
            Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf,   X_buf)
        queue.finish()            
        tdevice += time.time() - ts
        ts = time.time()
        cl.enqueue_copy(queue, emit, EMIT_buf)  # batch*NFREQ
        EMITTED[icell:(icell+batch),:] += emit[0:batch,:]        # contribution of current dust, current size

        if (icell==0):
            print('   SIZE %2d/%2d  icell %6d/%6d  %7.2f s/%d %.3e s/cell/size  %.2f cells/second' % \
            (isize, NSIZE, icell, CELLS, time.time()-t00, batch, (time.time()-t00)/batch, batch/(time.time()-t00)))
                    
        if (1):
            if ((ICELL>=icell)&((icell+batch)>ICELL)):
                cl.enqueue_copy(queue, X, X_buf)   # NE
                ARRAY0[isize,:] = X[ICELL-icell,  0:NE]  # first cell [NSIZE, NE]

        thost += time.time()- ts
                
    print("    HOST %.3f seconds,  DEVICE %.3f seconds" % (thost, tdevice))

    
DT = time.time() - t0
print('  %.3f SECONDS' % DT)
REF_RATE = 1520.0
REF_RATE =  100.7
REF_RATE =  1.0/1.054e-2
print('  %4d  -- %.3e SECONDS PER CELL  -- %8.3f CELLS PER SECOND -- x %5.2f' % (CELLS, DT/CELLS,  CELLS/DT, (CELLS/DT)/REF_RATE))
print('  Total time from the start of A2E_pyCL.py = %.3f seconds' % (time.time()-t_beg))
if (len(sys.argv)<=5):
    sys.exit() # skip plotting
    
if (1):
    asarray(ARRAY0, np.float32).tofile('previous.ARRAY')  # first cell
    asarray(ARRAY0, np.float32).tofile('ARRAY0.dump')     # first cell

        
#=================================================================================================================================


        
FREQ = np.loadtxt('freq.dat')

    
if (1):
    #x1 = np.fromfile('BlackBody_T_06000.emit.REF', float32)[2:]  # this with many binnings ??
    #x2 = np.fromfile('BlackBody_T_06000.emit.GPU', float32)[2:]  # single binning A2E
    HEADER   = 2

    filename     = 'cpp.emitted'       # this written by A2E.cpp
    
    cells, nfreq = np.fromfile(filename, np.int32, 2)
    EM_REF       = np.fromfile(filename, np.float32)[2:]
    EM_REF.shape = (CELLS, NFREQ)    

    
    figure(1, figsize=(7,6))
    subplots_adjust(left=0.1, right=0.93, bottom=0.07, top=0.94, wspace=0.25, hspace=0.25)
    
    clf()
    subplot(221)
    # *** SPECTRA FROM A2E.cpp AND FROM A2E_pyCL.py ***
    icell    = 0
    em_ref   = EM_REF[ icell,:]              # reference A2E.cpp solution for first cell
    em_py    = EMITTED[icell,:]             # A2E_pyCL.py spectrum for first cell
    loglog(f2um(FREQ), em_ref,  'r-',  alpha=0.6, label='A2E, cell=%d' % icell)
    loglog(f2um(FREQ), em_py,   'b--', lw=2, alpha=0.6, label='Current, cell=%d' % icell )    
    if (1): # some other cell = ICELL
        em2_ref   = EM_REF[ ICELL,:]         # reference A2E.cpp solution
        em2_py    = EMITTED[ICELL,:]        # A2E_pyCL.py spectrum
        loglog(f2um(FREQ), em2_ref,  'm--',  alpha=0.6,      label='A2E, cell=%d' % ICELL)
        loglog(f2um(FREQ), em2_py,   'c.-', lw=2, alpha=0.6, label='Current, cell=%d' % ICELL)
        
    
    legend(loc='lower right')
    if (0):
        for icell in arange(1, min([CELLS,BATCH]), max([1,BATCH/10])):
            semilogy(EMITTED[icell,:],  'g-', lw=3, alpha=0.3)
    ylim(1e-23, 99.0)
    

    s1, s2, S1, S2 = 0.0, 0.0, 0.0, 0.0
    for i in range(ARRAY0.shape[0]):
        s1  = sum(A2E_ARRAY[i,:])
        s2  = sum(ARRAY0[i,:])
        S1 += s1
        S2 += s2
        print("i=%2d    CPP=%12.4e    PY=%12.4e" % (i, s1, s2))
    print("SUM CPP %12.4e, PY %12.4e" % (S1, S2))
    
    
    subplot(222)
    ## plot(transpose(ARRAY), alpha=0.3)  # ARRAY[size, NE]
    IE = np.linspace(1.0, NE+1.0, NE)
    ie = 3
    if (1):
        loglog(IE, ARRAY0[1,   :], 'r-', lw=1, label='Current, size=1')    # cell ICELL
        #plot(IE,   ARRAY0[ie,  :], 'g-', lw=1, label='Current, size=%d' % ie)
        plot(IE,   ARRAY0[-2,  :], 'b-', lw=1, label='Current, size=-2')    
    if (1):
        # A2E_ARRAY == DUMP.array from A2E.cpp, first cell [NSIZE, NE]
        semilogy(IE, A2E_ARRAY[1,   :], 'm--', lw=2, alpha=0.5, label='A2E, size=1')    # cell ICELL
        #plot(IE,     A2E_ARRAY[ie,  :], 'k--', lw=2, alpha=0.5, label='Cpp, size=%d' % ie)        
        plot(IE,     A2E_ARRAY[-2,  :], 'c--', lw=2, alpha=0.5, label='A2E, size=-2')
    else:
        # PREV_ARRAY == result of previous call to this script
        semilogy(IE, PREV_ARRAY[1,   :], 'm-', lw=2, alpha=0.5)
        #plot(IE, PREV_ARRAY[ie,      :], 'y-', lw=2, alpha=0.5)
        plot(IE, PREV_ARRAY[-2,      :], 'c-', lw=2, alpha=0.5)
    ####
    legend(loc='lower left', labelspacing=0.01)
    title("CELL %d" % ICELL)
    
    
    subplot(223)    
    imshow(log10(ARRAY0), aspect='auto')
    title("CURRENT, cell %s" % ICELL)
    colorbar()

    
    subplot(224)
    if (1):
        imshow(log10(A2E_ARRAY), aspect='auto')
        title("CPP A2E ARRAY")
    else:
        Z = 1.0e6*(clip(ARRAY0,1e-25,1e30)-clip(A2E_ARRAY,1e-25,1e30))
        title("RES-A2E (1e-6), %.2e" % max(ravel(Z)), size=9)
        imshow(Z, aspect='auto')
        if (0):
            print('REF SOLUTION')
            print(A2E_ARRAY[0, :])
            print('NEW SOLUTION')
            print(ARRAY[0, :])
    colorbar()
    
    
    
    show()
    
            
