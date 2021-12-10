#!/bin/python
import os, sys
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)
from ASOC_aux import *


if (len(sys.argv)<3):
    print()
    print("Usage:")
    print("   EQ_solver.py  dust  absorbed.dat  emitted.dat  [GPU]")
    print("   dust           =  name of the dust file (simple dust)")
    print("   absorbed.data  =  file with absorptions")
    print("   emitted.data   =  file where emission will be written")
    print("   GPU            =  integer, values >0 tell program to use CPU instead of GPU")
    print()
    sys.exit(0)
    
    
dust        =  sys.argv[1]
f_absorbed  =  sys.argv[2]
f_emitted   =  sys.argv[3]
GPU         =  0
if (len(sys.argv)>4):
    GPU     =  int(sys.argv[4])
use_mmap    =  False

C_LIGHT =  2.99792458e10  
PLANCK  =  6.62606957e-27 
H_K     =  4.79924335e-11 
D2R     =  0.0174532925       # degree to radian
PARSEC  =  3.08567758e+18 
H_CC    =  7.372496678e-48 

def PlanckSafe(f, T):  # Planck function
    # Add clip to get rid of warnings
    return 2.0*H_CC*f*f*f / (np.exp(np.clip(H_K*f/T,-100,+100))-1.0)

    
def opencl_init(GPU, platforms, verbose=0):
    """
    Initialise OpenCL environment.
    """
    platform, device, context, queue = None, None, None, None
    ok = False
    # print("........... platforms ======", platforms)
    for iii in range(2):
        for iplatform in platforms:
            tmp = cl.get_platforms()
            if (verbose):
                print("      --------------------------------------------------------------------------------")
                print("      GPU=%d,  TRY PLATFORM %d" % (GPU, iplatform))
                print("      NUMBER OF PLATFORMS: %d" % len(tmp))
                print("      PLATFORM %d = " % iplatform, tmp[iplatform])
                print("      DEVICE ",         tmp[iplatform].get_devices())
                print("      --------------------------------------------------------------------------------")
            try:
                platform  = cl.get_platforms()[iplatform]
                if (GPU):
                    device  = platform.get_devices(cl.device_type.GPU)
                else:
                    device  = platform.get_devices(cl.device_type.CPU)
                context  = cl.Context(device)
                queue    = cl.CommandQueue(context)
                ok       = True
                if (verbose):
                    print("     ===>     DEVICE ", device, " ACCEPTED !!!")
                break
            except: 
                if (verbose):
                    print("     ===>     DEVICE ", device, " REJECTED !!!")                
                pass
        if (ok):
            return context, queue, cl.mem_flags
        else:
            if (iii==0):
                platforms = arange(4)  # try without specific choise of platform
            else:
                print("*** EQ_solver.py => opencl_ini could not find valid OpenCL device *** ABORT ***")
                time.sleep(10)
                sys.exit()
            

                

UM_MIN     = 0.0001
UM_MAX     = 99999.0
platforms  = [0,1,2,3,4]


# Read dust data
print("      SolveEquilibriumDust(%s)" % dust)
lines  =  open(dust).readlines()
gd     =  float(lines[1].split()[0])
gr     =  float(lines[2].split()[0])
d      =  np.loadtxt(dust, skiprows=4)
FREQ   =  np.asarray(d[:,0].copy(), np.float32)
KABS   =  np.asarray(d[:,2] * gd * np.pi*gr**2.0, np.float32)   # cross section PER UNIT DENSITY
# Start by making a mapping between temperature and energy
NE     =  30000
TSTEP  =  1600.0/NE    # hardcoded upper limit 1600K for the maximum dust temperatures
TT     =  np.zeros(NE, np.float64)
Eout   =  np.zeros(NE, np.float64)
DF     =  FREQ[2:] - FREQ[:(-2)]  #  x[i+1] - x[i-1], lengths of intervals for Trapezoid rule
# Calculate EMITTED ENERGY per UNIT DENSITY, scaled by 1e20 -> FACTOR
for i in range(NE):
    TT[i]   =  1.0+TSTEP*i
    TMP     =  KABS * PlanckSafe(np.asarray(FREQ, np.float64), TT[i])
    # Trapezoid integration TMP over freq frequencies
    res     =  TMP[0]*(FREQ[1]-FREQ[0]) + TMP[-1]*(FREQ[-1]-FREQ[-2]) # first and last step
    res    +=  sum(TMP[1:(-1)]*DF)          # the sum over the rest of TMP*DF
    Eout[i] =  (4.0*np.pi*FACTOR) * 0.5 * res  # energy corresponding to TT[i] * 1e20, per unit density
# Calculate the inverse mapping    Eout -> TTT
Emin, Emax  =  Eout[0], Eout[NE-1]*0.9999
# print("=== Mapping EOUT ==>   Emin %12.4e, Emax %12.4e\n" % (Emin, Emax))
# E ~ T^4  => use logarithmic sampling
kE          =  (Emax/Emin)**(1.0/(NE-1.0))  # E[i] = Emin*pow(kE, i)
oplgkE      =  1.0/np.log10(kE)
# oplgkE = 1.0/log10(kE)
ip          =  interp1d(Eout, TT)           # (linear) interpolation from energy to temperature
TTT         =  np.asarray(ip(Emin * kE**np.arange(NE)), np.float32)
# Set up kernels
CELLS, NFREQ=  np.fromfile(f_absorbed, np.int32, 2)
context, commands, mf = opencl_init(GPU, platforms)
source      =  open(INSTALL_DIR+"/kernel_eqsolver.c").read()
ARGS        =  "-D CELLS=%d -D NFREQ=%d -D FACTOR=%.4ef" % (CELLS, NFREQ, FACTOR)
if (0):
    ARGS += ' -cl-fast-relaxed-math'
    ARGS += ' -cl-opt-disable'    
program     =  cl.Program(context, source).build(ARGS)    
# Use the E<->T  mapping to calculate ***TEMPERATURES** on the device
GLOBAL      =  32768
LOCAL       =  [8, 32][GPU>0]
kernel_T    =  program.EqTemperature
#                               icell     kE          oplgE       Emin        NE         FREQ   TTT   ABS   T
kernel_T.set_scalar_arg_dtypes([np.int32, np.float32, np.float32, np.float32, np.int32 , None,  None, None, None])
FREQ_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=FREQ)
TTT_buf     =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=TTT)
KABS_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=KABS)
ABS_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*GLOBAL*NFREQ)  # batch of GLOBAL cells
T_buf       =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)
EMIT_buf    =  cl.Buffer(context, mf.WRITE_ONLY, 4*CELLS)
# Solve temperature GLOBAL cells at a time
TNEW        =  np.zeros(CELLS, np.float32)
tmp         =  np.zeros(GLOBAL*NFREQ, np.float32)
# print("=== EQ_solver.py Solve temperatures")

# Open file containing absorptions
FP_ABSORBED =  open(f_absorbed, 'rb')
CELLS, NFREQ=  np.fromfile(FP_ABSORBED, np.int32, 2) # get rid of the header
print("      SolveEquilibriumDust(%s): CELLS %d, NFREQ %d" % (f_absorbed, CELLS, NFREQ))
t0          =  time.time()
for ibatch in range(int(CELLS/GLOBAL+1)):
    # print("BATCH %d" % ibatch) 
    a   =  ibatch*GLOBAL
    b   =  min([a+GLOBAL, CELLS])  # interval is [a,b[
    # no need to use mmap file for absorptions - values are read in order
    # print("      Solving  eqdust [%6d, %6d[ out of %6d" % (a, b, CELLS))
    # print("    READING FP_ABSORBED: ", FP_ABSORBED)
    tmp[0:((b-a)*NFREQ)] =  np.fromfile(FP_ABSORBED, np.float32, (b-a)*NFREQ)
    cl.enqueue_copy(commands, ABS_buf, tmp)
    kernel_T(commands, [GLOBAL,], [LOCAL,], a, kE, oplgkE, Emin, NE, FREQ_buf, TTT_buf, ABS_buf, T_buf)
cl.enqueue_copy(commands, TNEW, T_buf)    
a, b, c  = np.percentile(TNEW, (10.0, 50.0, 90.0))
print("      Solve temperatures: %.2f seconds =>  %10.3e %10.3e %10.3e" % (time.time()-t0, a,b,c))
FP_ABSORBED.close()

np.asarray(TNEW, np.float32).tofile('%s.T' % dust)
        
# Use another kernel to calculate ***EMISSIONS*** -- per unit density
# 2019-02-24 --- this can be restricted to output frequencies [UM_MIN, UM_MAX]
kernel_emission = program.Emission
#                                      FREQ        KABS         T     EMIT
kernel_emission.set_scalar_arg_dtypes([np.float32, np.float32,  None, None ])
GLOBAL   =  int((CELLS/LOCAL+1))*LOCAL
if ((GLOBAL%64)!=0): GLOBAL = int((GLOBAL/64+1))*64
# figure out the actual output frequencies
MOUT     =  np.nonzero((FREQ<=um2f(UM_MIN))&(FREQ>=um2f(UM_MAX)))
OFREQ    =  FREQ[MOUT]
NOFREQ   =  len(OFREQ)
# Solve emission one frequency at a time, all cells on a single call
print("      EQ_solver.py  Solve emitted for NOFREQ = %d frequencies" % NOFREQ)
t0       =  time.time()
if (use_mmap):
    np.asarray([CELLS, NOFREQ], np.int32).tofile(f_emitted)
    EMITTED      =  np.memmap(f_emitted, dtype='float32', mode='r+', offset=8, shape=(CELLS,NOFREQ))
    EMITTED[:,:] = 0.0
else:
    EMITTED      = np.zeros((CELLS, NOFREQ), np.float32)        
# **** BAD ****  ----   update has outer loop over FREQ,  inner over CELLS
#                storage order has outer loop over CELLS, inner over FREQ 
for ifreq in MOUT[0]:            # ifreq = selected frequencies, index to full list of frequencies
    oifreq = ifreq-MOUT[0][0]    # index to the set of output frequencies
    kernel_emission(commands, [GLOBAL,], [LOCAL,], FREQ[ifreq], KABS[ifreq], T_buf, EMIT_buf)
    cl.enqueue_copy(commands, TNEW, EMIT_buf)
    commands.finish()
    EMITTED[:, oifreq] = TNEW    # OUT OF ORDER UPDATE --- HOPE FOR SMALL NOFREQ!
    ## print("ifreq %3d   ofreq %3d   %10.3e" % (ifreq, oifreq, mean(TNEW)))
print("      EQ_solver.py  Solve emitted: %.2f seconds" % (time.time()-t0))
if (use_mmap):
    del EMITTED
else:
    fp = open(f_emitted, 'wb')
    np.asarray([CELLS, NOFREQ], np.int32).tofile(fp)
    EMITTED.tofile(fp)  #  B(T)*kappa for a single dust component
    fp.close()

