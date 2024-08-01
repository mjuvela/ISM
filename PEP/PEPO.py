#!/usr/bin/env python

import time
t00 = time.time()

import os, sys
import pyopencl as cl
import numpy as np

INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)

if (len(sys.argv)<2):
    print("Usage:   PEPO.py my.ini [-d]")
    sys.exit()
    
DOUBLE = 0    
if (len(sys.argv)>2):
    if (sys.argv[2]=='-d'):
        DOUBLE = 1
    
if (DOUBLE):
    FLOAT = np.float64
    NBYTE = 8
else:
    FLOAT = np.float32
    NBYTE = 4

import PEPO_Aux    
PEPO_Aux.FLOAT = FLOAT
from PEPO_Aux import *

INI    =  ReadIni(sys.argv[1])    
TBG    =  2.725
MOL    =  INI['mol']
TLIM   =  INI['tlim']
TKIN   =  INI['tkin']
GPU    =  INI['gpu']
rho    =  INI['density']
cd     =  INI['colden']
RHO    =  asarray(logspace(log10(rho[0]), log10(rho[1]),  int(rho[2])),  FLOAT)
CD0    =  asarray(logspace(log10(cd[0]),  log10(cd[1]),   int(cd[2])),   FLOAT)
FWHM   =  INI['fwhm']
CD     =  (CD0/(1.0e5*FWHM)) * c0 * 0.939437279    # only CD/FWHM matters
NRHO, NCD = len(RHO), len(CD)
NW     =  NRHO*NCD

LOCAL  = [ 4, 64 ][INI['gpu']>0]
GLOBAL = (NW//LOCAL+1)*LOCAL
ESC    =  INI['escape']

# DownloadMoleculeData(MOL)      # ... not useful ?

t0 = time.time()
# Read molecule, limited to energy levels below given temperatures  (=0.003 seconds)
NL, E, G,   UL, Aul, F,   TCOL, CC = ReadMolecule(MOL)

CABU = ones(10, FLOAT)/len(CC)
nc  =  len(INI['cabu'])
if (nc>0):
    CABU[:] = 0.0
    for i in range(nc): CABU[i] = INI['cabu'][i]
if (nc!=len(CC)):
    print("*** Warning: Ini-file specified abundances for %d collisional partners out of %d???" % (nc, len(CC)))
    
# Drop energy levels E > Elimit
# ... this does not work if levels are not in increasing order of energy
NL  =  len(nonzero(E<(TLIM*k0))[0])
E   =  E[0:NL].copy()                      # should be in increasing order of energy
G   =  G[0:NL].copy()
# The transitions remaining between the selected levels
m   =  nonzero((UL[:,0]<NL)&(UL[:,1]<NL))  # remaining transitions
UL  =  UL[m[0],:].copy()
Aul =  Aul[m].copy()
F   =  F[m].copy()
NTR =  len(Aul)
# Combine collisional coefficients into a single matrix, RHO not yet multiplied in
NTCOL  =  CC[0].shape[0]
CX  =  zeros((NTCOL,NL,NL), FLOAT)
for i in range(len(CC)):
    CX[:,:,:] += CABU[i] * CC[i][:, 0:NL, 0:NL]
            
if (0):
    for i in range(NL):
        print("   level %2d  E %7.2f cm-1   G %.1f" % (i, E[i]/(h0*c0), G[i]))
    for i in range(NTR):
        print("   tran  %2d  %d->%d  F %10.3e  Aul %10.3e" % (i, UL[i,0],UL[i,1], F[i], Aul[i]))
    for u in range(NL):
        for l in range(u):
            print("   C(%d,%d) =  %12.4e" % (u, l, CX[1,l,u]))
    sys.exit()
    
if (0):  # Recompute frequencies based on the energies?
    for tr in range(NTR):
        u, l  = UL[tr]
        F[tr] = (E[u]-E[l])/h0

# Reduce further to a single matrix for the selected Tkin (=0.009 seconds)
C   =  zeros((NL,NL), FLOAT)
for u in range(NL):
    for l in range(u):
        ip     = interp1d(TCOL, CX[:, l, u], kind='linear', bounds_error=False, fill_value=(CX[0,l,u],CX[-1,l,u]))
        C[l,u] = ip(TKIN)

# Fill in missing collisional coefficients --- C[l,u] = C(u->l)  (=0.001 seconds)
for u in range(NL):
    for l in range(u):
        # fill in C[:, u, l] for transitions l->u
        # Clu = Cul * Gu/Gl * exp(-(Eu-El)/(k0*T))
        C[u, l] = C[l, u] * (G[u]/G[l]) * exp(-(E[u]-E[l])/(k0*TKIN))

# EPF =>  coefficient matrix will have replacements  Aul -> beta*Aul,  J*B -> beta*Bul*Ibg
Ibg = PlanckFunction(F, TBG)       # background intensity for transition frequencies
# initial level populations from LTE
n   =   ones(NL, FLOAT)
for i in range(1, NL):
    n[i] = n[0] * (G[i]/G[0]) * exp(-(E[i]-E[0])/(k0*TKIN))
n /= sum(n)    

# UL is mapping  transition -> {u,l}
# Make TR[] matrix for faster reverse lookup {u,l} -> transition
TR  =  -99*ones((NL, NL), int32)
for i in range(NTR):  # faster lookup levels -> transition
    TR[UL[i][0], UL[i][1]] = i    #  TR[u,l] = tr

# Potential hyperfine transition
HFSTR = -1            
HFSN, HFS0, HFSK, HFSC, HFSMAX  = 1, -1, -1, -1, -1
if (len(INI['hfs'])>0):
    # add hfs to one transition
    u, l, freq, hfs = ReadHFS(INI['hfs'])
    HFSTR = TR[u,l]
    HFSN, HFS0, HFSK, HFSC, HFSMAX  =  get_HFS_beta(hfs, FWHM)

print("Molecule preparation (read, cut levels, update Clu): %.2f seconds" % (time.time()-t0))
print("So far %.2f seconds" % (time.time()-t00)) 
    

# OpenCL initialisation -----------------------------------------------------------------
t0  =  time.time()
PLF = [0, 1, 2, 3, 4]
platform, device, context, queue, mf = InitCL(GPU, platforms=PLF, sdevice=INI['sdevice'])

OPT  = "-D NL=%d -D NT=%d -D TKIN=%.4ef -D NRHO=%d -D NCD=%d -D NW=%d " % (NL, NTR, TKIN, NRHO, NCD, NW)
OPT += "-D DOUBLE=%d -D ESC=%d -D LOCAL=%d -D ALFA=%.3ef " % (DOUBLE, ESC, LOCAL, INI['alfa'])
OPT += "-D ATOL=%.3ef -D RTOL=%.3ef -D MAXITER=%d " % (INI['tolerance'][0], INI['tolerance'][1], int(INI['tolerance'][2]))
OPT += "-D HFSTR=%d -D HFSN=%d -D HFS0=%.4ef -D HFSK=%.4ef -D HFSMAX=%.4ef" % (HFSTR, HFSN, HFS0, HFSK, HFSMAX)
if (1): # default kernel
    src  = open(INSTALL_DIR+"/kernel_PEPO.c").read()
else:   # a faster version... but GPU may run out of local memory for a large number of levels
    src  = open(INSTALL_DIR+"/kernel_PEPO_evo.c").read()
    
program   =  cl.Program(context, src).build(OPT)

# Buffer objects
E_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=E)
G_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G)
F_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=F)
A_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Aul)
Ibg_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Ibg)
C_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C)
UL_buf   = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=UL)
TR_buf   = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=TR)
RHO_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=RHO) # RHO[NRHO]
CD_buf   = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=CD)  # CD[NCD]
X_buf    = cl.Buffer(context, mf.READ_WRITE, NBYTE*NW*NL*NL)                # X[NCD,NRHO,NL,NL]
b_buf    = cl.Buffer(context, mf.READ_WRITE, NBYTE*NW*NL)                   # b[NCD,NRHO,NL]
n_buf    = cl.Buffer(context, mf.READ_WRITE, NBYTE*NW*NL)
TEX_buf  = cl.Buffer(context, mf.READ_WRITE, NBYTE*NW*NTR)
TAU_buf  = cl.Buffer(context, mf.READ_WRITE, NBYTE*NW*NTR)
TRAD_buf = cl.Buffer(context, mf.READ_WRITE, NBYTE*NW*NTR)
V_buf    = cl.Buffer(context, mf.READ_WRITE, NBYTE*NW*NL*2)
P_buf    = cl.Buffer(context, mf.READ_WRITE, NBYTE*NW*NL)
HFSC_buf = cl.Buffer(context, mf.READ_ONLY,  NBYTE*HFSN)
if (HFSTR>=0):
    cl.enqueue_copy(queue, HFSC_buf, HFSC)
# Initially same level populations for all parameter cases  (0.004 seconds)

# copy initial level populations to all (=0.004 seconds)
nn      = zeros((NCD, NRHO, NL), FLOAT)
for i in range(NCD):
    for j in range(NRHO):
        nn[i,j,:] = n        
cl.enqueue_copy(queue, n_buf, nn)

print("OpenCL initialisations: %.2f seconds" % (time.time()-t0))
print("So far: %.2f seconds" % (time.time()-t00))
# OpenCL initialisations:  0.81 seconds out of 1.28 second run time
# ---------------------------------------------------------------------------------------

t0 = time.time()
STEP    = program.Step
STEP(queue, [GLOBAL,], [LOCAL,],     E_buf, G_buf,     F_buf, A_buf, Ibg_buf,
     C_buf, UL_buf, TR_buf,     RHO_buf, CD_buf,  X_buf, b_buf, n_buf, 
     TEX_buf, TAU_buf, TRAD_buf,    V_buf, P_buf,   HFSC_buf)
cl.enqueue_copy(queue, nn, n_buf)

print("KERNEL CALL: %.3f seconds" % (time.time()-t0))
print("So far %.2f seconds" % (time.time()-t0))

"""
./PEPO.py pep.ini 128x128 parameters
 - RTX-9700 XTX ... 0.10 seconds
 - 
"""

TEX  = zeros((NCD, NRHO, NTR), FLOAT)
TAU  = zeros((NCD, NRHO, NTR), FLOAT)
TRAD = zeros((NCD, NRHO, NTR), FLOAT)
cl.enqueue_copy(queue, TEX,  TEX_buf)
cl.enqueue_copy(queue, TAU,  TAU_buf)
cl.enqueue_copy(queue, TRAD, TRAD_buf)


# Save outputs to a binary file  (=0.005 seconds)
fp = open(INI['output'], 'wb')
asarray([NCD, NRHO, NTR], int32).tofile(fp)  # number of values for:  N(molecule), n(H2), transitions
asarray(F,    FLOAT).tofile(fp)  # transition frequencies  F[NTR]
asarray(RHO,  FLOAT).tofile(fp)  # density values          RHO[NRHO]
asarray(CD0,  FLOAT).tofile(fp)  # column densities        CD0[ NCD]
asarray(TEX,  FLOAT).tofile(fp)  # Tex values              TEX[ NCD, NRHO, NTR]
asarray(TAU,  FLOAT).tofile(fp)  # optical depths          TAU[ NCD, NRHO, NTR]
asarray(TRAD, FLOAT).tofile(fp)  # Trad intensities        TRAD[NCD, NRHO, NTR]
U = []
NU  =  nn[:,:,UL[:,0]].copy()    # upper level populations  NU[NCD, NRHO, NTR]
asarray(NU, FLOAT).tofile(fp)
fp.close()

if (0):
    # Print something also directly to terminal... results for the first parameter combination
    for tr in range(NTR):
        u, l = UL[tr]
        tex  = TEX[0,0,tr]
        tau  = TAU[0,0,tr]
        trad = TRAD[0,0,tr]
        nu   = NU[0,0,tr]
        if (1):
            print("t=%2d %2d-%2d  Tex %6.2f  tau %11.4e  Trad %11.4e  Aul %11.4e  F %11.4e nu %11.4e" % \
            (tr, u, l,  tex, tau, trad, Aul[tr], F[tr], nu))

print("PEPO.py finished, run time %.2f seconds" % (time.time()-t00))
# Beware of GPU (and CPU?)  powersaving settings.
# Example with an external AMD GPU, a with "normal" run time 0.7 seconds, 
# Once computer was idel for 5 seconds, next run took over five seconds.
# The "normal" run time is consistent only with several powersettings (GPU, possibly thunderbolt) disabled.





    
