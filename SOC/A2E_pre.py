#!/usr/bin/python3

import os, sys

# We assume that the Python scripts and *.c kernel files are in this directory
# sys.path.append(HOMEDIR+'/starformation/SOC/')
# HOMEDIR = os.path.expanduser('~/')
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)
  

from   ASOC_aux import *
from   DustLib import *
import pyopencl as cl
import numpy as np

"""
Write solver.data using OpenCL.
"""

if (len(sys.argv)<4):
    print("Usage:   A2E_pre.py <gset-dustname>  <frequencyfile>  <solver.data file> [NE]")
    sys.exit()
    

DUST   =  GSETDust(sys.argv[1])
FREQ   =  asarray(loadtxt(sys.argv[2]), float32)
NFREQ  =  len(FREQ)
NSIZE  =  DUST.NSIZE
##TDOWN  =  zeros((NSIZE, NFREQ), float32)
## SCALE = 1.0e20  ---- replaced by FACTOR from ASOC_aux.py
Ef     =  PLANCK*FREQ
NE     =  128
NE     =  256
if (len(sys.argv)>4):
    NE =  int(sys.argv[4])
NEPO   =  NE+1
LOCAL  =  4
GLOBAL =  int((NE/64)+1)*64

# we do not want any zeros in CRT_SFRAC !!
DUST.CRT_SFRAC = clip(DUST.CRT_SFRAC, 1.0e-25, 1.0e30)

if (0):
    isize, freq = 3, 1.0e12
    print("SKabs(isize=3, f=1e12)     = %12.4e" %  DUST.SKabs(3, 1.0e12))
    print("SKabs_Int(isize=3, f=1e12) = %12.4e" %  DUST.SKabs_Int(3, 1.0e12))
    print("SKabs*SFRAC*GRAIN_DENSITY  = %12.4e" % (DUST.SKabs(3, 1.0e12)*DUST.GRAIN_DENSITY*DUST.CRT_SFRAC[isize]))
    sys.exit()


def PlanckIntensity(f, T):
    res = (2.0*PLANCK*(f/C_LIGHT)**2.0*f) / (exp(H_K*f/T) - 1.0)
    ## print(" B(%.1f) = %12.4e %12.4e %12.4e" % (T, res[1], res[20], res[39]))
    return res
    
    
# Define the temperature grid based on Tmin and Tmax values for 
# the smallest and largest sizes, given in the dust file as
#      tlimits   tmin  tmax1   tmax2
Tmin, Tmax1, Tmax2 = 4.0, 2500.0, 150.0
dfile = sys.argv[1]
for line in open(dfile, 'r').readlines():
    s = line.split()
    if (len(s)<4): continue
    if (s[0].find('tlimit')==0):
        Tmin, Tmax1, Tmax2 = float(s[1]), float(s[2]), float(s[3])
print("Temperature limits in %s: Tmin %.1f, Tmax1 %.1f, Tmax2 %.1f" % (dfile, Tmin, Tmax1, Tmax2))
                
# Calculate temperature limits for each size bin
TMIN  = Tmin*ones(NSIZE, float32)
TMAX  = logspace(log10(Tmax1), log10(Tmax2), NSIZE)       

if (1): # 2019-10-25
    TMIN = DUST.TMIN
    TMAX = DUST.TMAX

Ibeg  = zeros(NFREQ, int32)
L     = zeros(NE*NE, int32)

# start by calculating SKABS[NSIZE, NFREQ]
SKABS = zeros((NSIZE, NFREQ), float64)
for isize in range(NSIZE):
    SKABS[isize,:] =  DUST.SKabs_Int(isize, FREQ)  # ==  pi*a^2*Q*S_FRAC, including GRAIN_DENSITY

if (0):
    print('-'*50)
    print(SKABS[-1,:])
    print('-'*50)
    
# OpenCL initialisation --- kernel probably efficient only on CPU
context, queue = None, None
for iplatform in range(3):
    try:
        platform  = cl.get_platforms()[iplatform]
        device  = platform.get_devices(cl.device_type.CPU)
        context = cl.Context(device)
        queue   = cl.CommandQueue(context)
        break
    except:
        pass
src       =  open(INSTALL_DIR+"/kernel_A2E_pre.c").read()
program   =  cl.Program(context, src).build(" -D FACTOR=%.4ef " % FACTOR)
mf        =  cl.mem_flags 
FREQ_buf  =  cl.Buffer(context, mf.READ_ONLY,  4*NFREQ)
Ef_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*NFREQ)
SKABS_buf =  cl.Buffer(context, mf.READ_ONLY,  4*NFREQ)  # one size!
E_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*NEPO)
T_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*NEPO)
Tdown_buf =  cl.Buffer(context, mf.READ_WRITE, 4*NE)
L1_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NE*NE)
L2_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NE*NE)
Iw_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NE*NE*NFREQ)      # may be ~100 MB !
wrk_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*(NE*NFREQ+NE*(NFREQ+4)))
noIw_buf  =  cl.Buffer(context, mf.WRITE_ONLY, 4*(NE-1))     # for each l = lower

Tdown     =  zeros(NE, float32)
wrk       =  zeros(NE*NFREQ+NE*(NFREQ+4), float32)
Iw        =  zeros(NE*NE*NFREQ, float32)
noIw      =  zeros(NE-1, int32)
EA        =  zeros((NFREQ, NE), float32)


cl.enqueue_copy(queue, FREQ_buf, FREQ)                    # NFREQ
cl.enqueue_copy(queue, Ef_buf,   asarray(Ef, float32))    # just for convenience

PrepareTdown =  program.PrepareTdown
PrepareTdown.set_scalar_arg_dtypes([np.int32, None, None, None, np.int32, None, None, None])

if (0):
    DoNotUse()  # ... or only for truly stochastically heated grains!
    PrepareIw    =  program.PrepareIntegrationWeights
if (0):
    DoNotUse()  # ... or only for truly stochastically heated grains!
    PrepareIw    =  program.PrepareIntegrationWeightsGD
if (1):         # ... seems to work also for large equilibrium-temperature grains
    PrepareIw    =  program.PrepareIntegrationWeightsEuler
    
PrepareIw.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None, None, None, None, None])

# Start writing the solver.data file
fp = open(sys.argv[3], "wb")
asarray( [NFREQ,    ], int32   ).tofile(fp)
asarray(  FREQ,        float32 ).tofile(fp)
asarray( [DUST.GRAIN_DENSITY,] , float32).tofile(fp)
asarray( [NSIZE,],     int32   ).tofile(fp)
# 2021-04-26 -- add DUST.SIZE_A to the solver file @@
asarray( DUST.SIZE_A, float32  ).tofile(fp)
# Dustlib CRT_SFRAC is the fraction multiplied with GRAIN_DENSITY
asarray(  DUST.CRT_SFRAC/DUST.GRAIN_DENSITY, float32 ).tofile(fp)  # this SFRAC is one with sum(SFRAC)==1, not GRAIN_DENSITY
asarray( [NE,       ], int32   ).tofile(fp)
asarray(  SKABS,       float32 ).tofile(fp)   # SKAbs_Int()    ~     pi*a^2*Qabs * SFRAC, including GRAIN_DENSITY

# Below PrepareTdown uses SKabs() instead of SKAbs_Int() => divide by SFRAC  (including GRAIN_DENSITY)


# in the following Tdown needs SKABS ... but per grain => need to divide CRT_SFRAC

print("NSIZE %d, NFREQ %d, NE %d" % (NSIZE, NFREQ, NE))

for isize in range(NSIZE):    
    
    # A2ELIB used logarithmically spaced energies
    print("--------------------------------------------------------------------------------")
    emin  =  DUST.T2E(isize, TMIN[isize])
    emax  =  DUST.T2E(isize, TMAX[isize])
    print("nsize [%3d] %9.4f um  E %.3e - %.3e  T = %5.2f - %7.2f K" % \
    (isize, DUST.SIZE_A[isize]*1.0e4, emin, emax, TMIN[isize], TMAX[isize]))
    
    if (0): # 
        E     =  exp(log(emin)+(arange(NEPO)/float(NE))*(log(emax)-log(emin))) # NEPO elements
        T     =  DUST.E2T(isize, E)       # NEPO !
    else:
        # GSETDustO.cpp has --  e = TemperatureToEnergy_Int(s, TMIN[s]+(TMAX[s]-TMIN[s])* pow(ie/(NEPO-1.0), 2.0)) ;
        T   =  TMIN[isize]+(TMAX[isize]-TMIN[isize])* (arange(NEPO)/(NEPO-1.0))**2.0
        E   =  DUST.T2E(isize, T)
    cl.enqueue_copy(queue, SKABS_buf, asarray((SKABS[isize,:]/DUST.CRT_SFRAC[isize]),float32))
    cl.enqueue_copy(queue, E_buf, asarray(E, float32))  # E[NEPO]
    cl.enqueue_copy(queue, T_buf, asarray(T, float32))  # T[NEPO]  --- NEPO elements for interpolation in PrepareTdown

    # PrepareIntegrationWeights() kernel
    PrepareIw(queue, [GLOBAL,], [LOCAL,], NFREQ, NE, Ef_buf, E_buf, L1_buf, L2_buf, Iw_buf, wrk_buf, noIw_buf)
    cl.enqueue_copy(queue, Iw,   Iw_buf)
    cl.enqueue_copy(queue, noIw, noIw_buf)      # one worker = one l=lower bin ~ at most NE*NFREQ Iw weights
    sum_noIw = sum(noIw)                        # number of actual integration weights, for each l = lower bin
    asarray([sum_noIw,], int32).tofile(fp)      # --> noIw
    for l in range(0, NE-1):                    # loop over lower bins = results of each kernel worker
        ind = l*NE*NFREQ                        # start of the array reserved for each l = each worker
        asarray(Iw[ind:(ind+noIw[l])], float32).tofile(fp)     # --> Iw, for l
    cl.enqueue_copy(queue, L,  L1_buf)
    L[0] = -2
    asarray(L, int32).tofile(fp)              # --> L1
    cl.enqueue_copy(queue, L,  L2_buf)
    L[0] = -2
    asarray(L, int32).tofile(fp)              # --> L2
    
    # PrepareTdown() kernel
    PrepareTdown(queue, [GLOBAL,], [LOCAL,], NFREQ, FREQ_buf, Ef_buf, SKABS_buf, NE, E_buf, T_buf, Tdown_buf)
    cl.enqueue_copy(queue, Tdown, Tdown_buf)    # --> Tdown
    ### Tdown /= DUST.CRT_SFRAC[isize]   # GRAIN_DENSITY already in CRT_SFRAC !! --- division already in SKABS
    Tdown.tofile(fp)
    print('T %.3e - %.3f  E %.3e - %.3e   Ef %.3e - %.3e' % (T[0], T[-1], E[0], E[-1], Ef[0], Ef[-1]))
    # print('SKABS ', SKABS[isize, 0:5], SKABS[isize, -5:])
    print(Tdown[0:4])
    
    # Prepare EA[ifreq, iE]  <---- storage order!
    TC  =  DUST.E2T(isize, 0.5*(E[0:NE]+E[1:])) # NE temperatures for the *centre* of each energy bin
    for iE in range(NE):                        # EA[ifreq, iE], here SKABS is still pi*a^2*Qabs*GD*SFRAC
        EA[:, iE] =  SKABS[isize,:] * (PlanckIntensity(asarray(FREQ, float64), TC[iE]) / (PLANCK*FREQ))
    EA *= FACTOR*4.0*np.pi
    asarray(EA, float32).tofile(fp)             # --> EA

    # Prepare Ibeg array
    for ifreq in range(NFREQ):
        startind = 1 
        while((0.5*(E[startind-1]+E[startind])<Ef[ifreq]) & (startind<NEPO-1)): startind += 1
        Ibeg[ifreq] = startind 
    Ibeg.tofile(fp)                             # --> Ibeg
                                                       
fp.close()    

