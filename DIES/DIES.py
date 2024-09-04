#!/usr/bin/env python
import os, sys

INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)
from DIES_Aux import *

IPO = interp1d
IPO = interpL

if (len(sys.argv)<2):
    print("Usage:   DIES.py dies.ini")
    sys.exit()

t00    =  time.time()    
INI    =  ReadIni(sys.argv[1])
OFFS   =  INI['offsets']
BATCH  =  INI['batch']    # now the number of batches when using two absorption buffers

# DUST MODEL
FF, G, KABS, KSCA  = ReadDust(INI['dust'])
NF     =  len(FF)   
ipKABS =  IPO(FF, KABS, fill_value=1.0e-30, bounds_error=False, kind='linear') # tau per H * PARSEC
ipKSCA =  IPO(FF, KSCA, fill_value=1.0e-30, bounds_error=False, kind='linear') # tau per H * PARSEC
ipG    =  IPO(FF, G,    fill_value=0.001,   bounds_error=False)
if (0):
    clf()
    loglog(f2um(FF), KABS, 'r-', label='KABS/H * pc')
    loglog(f2um(FF), KSCA, 'b-', label='KSCA/H * PC') 
    loglog(f2um(FF), G,    'g-', label='G')
    legend()
    SHOW()
    
# DENSITY MODEL
RC, RHO = ReadCloud(INI['cloud'])
try:
    RC  *= INI['gridlength']   # ini file has potential *multiplier*
except:
    pass   # file already contains radiae in [pc]
if (INI['density']>0.0):
    RHO *= INI['density']

NC     =  len(RC)
VC     =  zeros(NC, float32)
VC[0]  =  (4.0/3.0)*pi*RC[0]**3                   # cell volume V [pc^3]
for i in range(1, NC):
    VC[i]   = (4.0/3.0)*pi*(RC[i]**3-RC[i-1]**3)  # cell volume V [pc^3]
ABS    =  zeros(NC, float32)                      # absorbed energy in each cell
N_BG   =  INI['bgpackets']                        # number of photon packages from the background
N_PS   =  INI['pspackets']                        # number of photon packages from the central source

GPU        =   INI['gpu']
PLF        =  [0,1,2,3]
if (INI['platform']>0): PLF = [INI['platform'],]

LOCAL      =  [4, 32][GPU>0]
if (N_BG<1): 
    GLOBAL_BG  =   0
else:     
    GLOBAL_BG  =  (N_BG//LOCAL)*LOCAL
    if (BATCH>1):
        while ( ((GLOBAL_BG//BATCH) % LOCAL) !=0 ): 
            GLOBAL_BG += LOCAL
    N_BG       =   GLOBAL_BG
if (N_PS<1): 
    GLOBAL_PS  =   0
else:
    GLOBAL_PS  =  (N_PS//LOCAL+1)*LOCAL
    if (BATCH>1):
        while ( ((GLOBAL_PS//BATCH) % LOCAL) !=0): 
            GLOBAL_PS += LOCAL
    N_PS       =   GLOBAL_PS
print("GLOBAL_BG %9d, GLOBAL_PS %9d" % (GLOBAL_BG, GLOBAL_PS))
print( (GLOBAL_BG//10) % LOCAL)
print( (GLOBAL_PS//10) % LOCAL)

    
#  Iff radiation at um>100um do not affect dust temperature... ignore that part of emission
UM_LIMIT = 100.0

# Total background luminosity divided into N_BG photon packages, random frequencies
DE_BG   =  0.0
ipISRF  =  None
if (N_BG>0):
    d      =  loadtxt(INI['background'][0], skiprows=1)    # just test file { freq, intensity }
    kISRF  =  INI['background'][1]
    x, y   =  d[:,0]*1.0, d[:,1]*kISRF
    if (0):
        mm = nonzero(d[:,0]<um2f(UM_LIMIT))               # ignore Ibg for wavelengths > UM_LIMIT
        d[m[0],1] = 0.0
    else:
        m   = nonzero(d[:,1]>1.0e-5*d[:,1])
        x, y = x[m], y[m]
    ipISRF =  interp1d(x, y, bounds_error=False, kind='quadratic', fill_value=1.0e-30)  #  [Hz] -> [cgs]
    # tolerances 1e-4 ... 1e-8 have no effect
    DE_BG  =  quad(ipISRF, x[0], x[-1], epsabs=1.0e-6, epsrel=1.0e-6, limit=100)[0]   # integral of intensity
    DE_BG *=  pi  *  4.0*pi*RC[-1]**2  /  N_BG             # true energy / pc^2 / per photon package
    
    
# Total point source luminosity divided into N_PS photon packages
DE_PS   =  0.0
kPS     =  INI['pointsource'][1]                            # scaling factor
RPS     =  INI['pointsource'][2]                            # source radius
if (N_PS>0):
    d      =  loadtxt(INI['pointsource'][0], skiprows=0)    # { freq, intensity }
    x, y   =  d[:,0]*1.0, d[:,1]*kPS
    # whether one uses one of the following or not, no effect on the result nor the run time
    if (1):  #  PC=1e6  => kernel 0.0152 seconds ... no, inequality the wrong way.... run time now 3.0 seconds !!!!
        mum  = nonzero(x>um2f(UM_LIMIT))                     # ignore wavelengths > UM_LIMIT
        x, y = x[mum], y[mum]
    else:  #  PC=1e6  => kernel 3.1 seconds !!!!!
        mum = nonzero(y>1.0e-6*max(y))                      # @@@ ???
        x, y = x[mum], y[mum]
    ipPS   =  interp1d(x, y, bounds_error=False, kind='quadratic', fill_value=0.0)  #  [Hz] -> [cgs] ... luminosity
    # tolerances 1e-4 ... 1e-8 have no effect
    DE_PS  =  quad(ipPS, x[0]*1.0001, x[-1]*0.9999, epsrel=1.0e-6, limit=80)[0]    # integral of luminosity
    DE_PS /=  PARSEC**2
    DE_PS /=  N_PS                                          # energy per per photon package
    if (0):
        clf()
        loglog(f2um(x), y, 'm-')
        show(block=True)
        sys.exit()

# ipFS_BG[NFF] = cumulative probability for background source emission, ipFS[u] = freq;   ipFS_BG[u] = random frequency
# ipKABS, ipKSCA, ipQ  =  parameters for frequencies F = F0*KF**ifreq =>   ipKABS[ifreq] etc.
# ipT = solve temperatures, energies E = E0*KE**iE, iE=0, ..., NE-1   ipT[iE] = corresponding temperature
# IPF = for each T, generate random frequency for reemission; IPF[NT*NF]
#            IPF[NT][u] = frequency

# fixed logarithmic frequency grid
NF     =  512
F0, F1 =  um2f(3000.0), um2f(0.091)
KF     =  exp(log(F1/F0)/(NF-1))
FF     =  F0*KF**arange(NF)             # fixed frequency grid
IPKABS =  asarray(ipKABS(FF), float32)  # IPKABS[NF]    ---  tau / H * pc
IPKSCA =  asarray(ipKSCA(FF), float32)  # IPSCA[NF]     ---  tau / H * pc
IPG    =  asarray(ipG(FF), float32)     # IPG[NF]

if (N_BG>0):
    # IPFS_BG to generate random frequencies for background packages  IPFS[NF]
    #    packages of equal energy => should be fewer long-wavelength packages
    IBG    =  ipISRF(FF)                    # IBG[NF]
    #  mask  >30 vs. >100 vs. >300 um   =>   T ~ 13 K ->  10 K  ->  8 K  ????????  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #  IBG = Ibg[FF] 
    IBG[nonzero(FF<um2f(UM_LIMIT))] = 1e-30 # remove FIR photons... that will not be absorbed
    # need integrals of energy E(f<FF[i]), cumsum does not work since FF is not linear...
    CI =  zeros(NF, float32)
    t0 =  time.time()
    if (0):
        # scipy integration  --  6.4 seconds
        for i in range(NF): 
            CI[i] = quad(ipISRF, FF[0], FF[i])[0]
    else:
        # Euler integration  --  0.05 seconds
        CI[0] = 0.0
        for i in range(1,NF):
            CI[i] = CI[i-1] + 0.5*(ipISRF(FF[i-1])+ipISRF(FF[i]))*(FF[i]-FF[i-1])
    print("... integration %.2f seconds" % (time.time()-t0))
    CI       +=  linspace(0.0, 1.0e-6, NF)
    CI        =  CI-CI[0]                      # for FF frequency grid, cumulative intensity
    CI       /=  CI[-1]                        #  ... [0,1] 
    ipF       =  interp1d(CI, FF)              #  u in [0,1]  ->  frequency
    IPFS_BG   =  asarray(ipF(linspace(1e-6, 0.999999, NF)), float32)  #  ipFS[NF*rand()] = freq,  random frequency for bg photons
    # IPFS_BG  frequencies for P, P has NF equal steps from 0 to 1
else:
    IPFS_BG  = zeros(NF, float32)
    
if (N_PS>0):
    # IPFS_PS to generate random frequencies for packages from the central source, IPFS_PS[NF]
    IPS    =  ipPS(FF)                         # luminosity, not intensity
    IPS[nonzero(FF<um2f(UM_LIMIT))] = 1e-30    # remove FIR photons... that will not be absorbed
    CI     =  zeros(NF, float32)
    t0     =  time.time()
    if (0):   # scipy integration  --  6.4 seconds
        for i in range(NF): 
            CI[i] = quad(ipPS, FF[0], FF[i])[0]
    else:     # Euler integration  --  0.05 seconds
        CI[0] = 0.0
        for i in range(1,NF):
            CI[i] = CI[i-1] + 0.5*(ipPS(FF[i-1])+ipPS(FF[i]))*(FF[i]-FF[i-1])
    print("... integration %.2f seconds" % (time.time()-t0))
    # ensure monotonously increasing cumulative probabilities
    CI       +=  linspace(0.0, 1.0e-6, NF)
    CI        =  CI-CI[0]                      # for FF frequency grid, cumulative intensity
    CI       /=  CI[-1]                        #  ... [0,1]
    ipF       =  interp1d(CI, FF)              #  u in [0,1]  ->  frequency
    IPFS_PS   =  asarray(ipF(linspace(0.0, 1.0, NF)), float32)   #  ipFS[NF*rand()] = freq,  random frequency for bg photons
else:
    IPFS_PS   =  zeros(NF, float32)
        
# limits of temperature and energy .... IPKABS[FF] etc.
#     y= 4*pi*a^2*Q * pi * B(T),  IPKABS = pi*a^2*Q   =>    y = 4*IPKABS * pi*B(T) = 4*pi*IPKABS*B(T)
T0     =  3.0
y      =  4.0*pi * IPKABS * PlanckFunction(FF, T0)
E0     =  trapezoid(y, FF)            #  E / H * pc
T1     =  1000.0
y      =  4.0*pi * IPKABS * PlanckFunction(FF, T1)
E1     =  trapezoid(y, FF)            #  E / H * pc

# solve temperature... first mapping T -> E
#  energy is emitted energy per H, multiplied by pc  ==  energy for unit density, pc^3 volume, divided by constant pc^2
NT     =  256
KT     =  exp(log(T1/T0)/(NT-1))
TT     =  T0*KT**arange(NT)
EE     =  zeros(NT, float32)
for iT in range(NT):
    y      =  4.0*pi * IPKABS * PlanckFunction(FF, TT[iT])     # Y[NF]
    EE[iT] =  trapezoid(y, FF)   # emission for T=T0*KE**iT    # EE[NT]  =   E / H * pc
ipET = IPO(EE, TT, bounds_error=False, fill_value=1.0) # ip for E -> T

# make reverse mapping          E = E0*KE**iE --> T   ... discrete version
KE     =  exp(log(E1/E0)/(NT-1))
EE2    =  E0*KE**arange(NT)             # EE2[NT]
IPET   =  asarray(ipET(EE2), float32)   # IPET[NT],   IPT[iE] = T for energy E0*KE**iE

# IPFR[NT, NF] =>  IPFR[iT, NF*rand()] = freq  ==> random frequency for dust reemission
y0   = zeros(NF, float32)
IPFR = zeros((NT, NF), float32)
for iT in range(NT):  # based on deterministic temperature grid
    y1  =   IPKABS * PlanckFunction(FF, TT[iT])
    Y   =   cumsum(y1-y0)                               #  Y[NF]
    if (iT==-10):
        clf()
        loglog(f2um(FF), y0, 'b-')
        loglog(f2um(FF), y1, 'r-')
        loglog(f2um(FF), Y, 'g-')
        SHOW()
    y0  =   1.0*y1
    Y  -=   Y[0]
    ip  =   interp1d(Y/Y[-1], FF)                       #  mapping [0,1] -> frequency
    IPFR[iT,:] = ip(linspace(0.0, 1.0, NF))             #  IPFR[T,u] ->  freq
    
print("----- initialisations:     %.2f seconds" % (time.time()-t00))

platform, device, context, queue, mf = InitCL(GPU, platforms=PLF, sdevice=INI['sdevice'])
# ok, prefer these cards, if available
NVIDIA =  ('3090' in device[0].name) | ('4070' in device[0].name) | ('NVIDIA' in device[0].name)

tCL = time.time()
OPT  = " -D NC=%d -D NVIDIA=%d"  % (NC, NVIDIA)
OPT += " -D NT=%d -D KT=%.4ef -D T0=%.4ef"   % (NT, KT, T0)
OPT += " -D NF=%d -D KF=%.4ef -D F0=%.4ef"   % (NF, KF, F0)
OPT += "          -D KE=%.4ef -D E0=%.4ef"   % (    KE, E0)
OPT += " -D DE_BG=%.4ef -D DE_PS=%.4ef -D OFFS=%d -D RPS=%.4ef"    % (DE_BG, DE_PS, OFFS, RPS)
# parameters for weighted sampling
OPT += ' -D WEIGHT_BG=%d       -D K_WEIGHT_BG=%.3ef'       % (INI['weightbg']>-1.1,       INI['weightbg'])
OPT += ' -D WEIGHT_EMIT=%d     -D K_WEIGHT_EMIT=%.3ef'     % (INI['weightemit']>-1.1,     FixG(INI['weightemit']))
OPT += ' -D WEIGHT_SCA=%d      -D K_WEIGHT_SCA=%.3ef'      % (INI['weightsca']>-1.1,      FixG(INI['weightsca']))
OPT += ' -D WEIGHT_EMIT_PS=%d  -D K_WEIGHT_EMIT_PS=%.3ef'  % (INI['weightemitps']>-1.1,   FixG(INI['weightemitps']))
OPT += ' -D WEIGHT_SCA_PS=%d   -D K_WEIGHT_SCA_PS=%.3ef'   % (INI['weightscaps']>-1.1,    FixG(INI['weightscaps']))
OPT += ' -D WEIGHT_STEP_SCA=%d -D K_WEIGHT_STEP_SCA=%.3ef' % (INI['weightstepsca']>-1.1,  INI['weightstepsca'])
OPT += ' -D FORCED=%d -D LOCAL=%d -D BATCH=%d' % (INI['forced'], LOCAL, BATCH)
tmp  =    0                               # no adjustment for the free path for absorptions
if (INI['weightstepabs']>-1.0): tmp = 1   # fixed free-path change
if (INI['weightstepabs']>+8.0): tmp = 2   # adaptive free-path change
OPT += ' -D WEIGHT_STEP_ABS=%d -D K_WEIGHT_STEP_ABS=%.3ef' % (tmp,  INI['weightstepabs'])
OPT += ' -I ./ '
# 
print(OPT)
#
src  = open(INSTALL_DIR+"/kernel_DIES.c").read()

t0 = time.time()
program   =  cl.Program(context, src).build(OPT)
print("----- build:               %.2f seconds" % (time.time()-t0))

RC_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=RC)       # RC[NC]
VC_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=VC)       # VC[NC]
RHO_buf     =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=RHO)      # RHO[NC]
IPFS_BG_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=IPFS_BG)  # IPFS_BG[NF]  ***
IPFS_PS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=IPFS_PS)  # IPFS_PS[NF]  ***
IPET_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=IPET)     # IPET[NT]
IPFR_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=IPFR)     # IPFR[NT, NF]
IPKABS_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=IPKABS)   # IPKABS[NF]
IPKSCA_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=IPKSCA)   # IPSCA[NF]
IPG_buf     =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=IPG)      # ABS[NC]
ABS_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*NC)
DABS_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NC)

SIM_BG      =  program.SimulateBG
SIM_BG.set_scalar_arg_dtypes([np.float32, None,None,None,None,None, None,None,None,None,None,None])

SIM_PS      =  program.SimulatePS
SIM_PS.set_scalar_arg_dtypes([np.float32, None,None,None,None,None, None,None,None,None,None,None])

MAP         =  program.Map
UPDATE      =  program.UpdateABS
print("----- OpenCL setup:        %.2f seconds" % (time.time()-tCL))

# Simulate
ABS         =  zeros(NC, float32)
cl.enqueue_copy(queue,  ABS_buf, ABS)   #  ABS =  E / H * pc
cl.enqueue_copy(queue, DABS_buf, ABS)   #  ABS =  E / H * pc

SEED    = INI['seed']

t0   = time.time()

if (N_BG>0):
    if (BATCH<=1):
        SIM_BG(queue, [GLOBAL_BG,], [LOCAL,], SEED, RC_buf, VC_buf, RHO_buf, IPFS_BG_buf, IPFR_buf, IPET_buf,
                                              IPKABS_buf, IPKSCA_buf, IPG_buf, ABS_buf, DABS_buf)
    else:
        GLOB        =  GLOBAL_BG//BATCH
        TMP_GLOBAL  =  max([32, (NC//LOCAL+1)*LOCAL])
        for ibatch in range(BATCH):
            SEED = rand()
            SIM_BG(queue, [GLOB,], [LOCAL,], SEED, RC_buf, VC_buf, RHO_buf, IPFS_BG_buf, IPFR_buf, IPET_buf,
                                              IPKABS_buf, IPKSCA_buf, IPG_buf, ABS_buf, DABS_buf)
            UPDATE(queue, [TMP_GLOBAL,], [LOCAL,], ABS_buf, DABS_buf)
if (N_PS>0):                                
    if (BATCH<=1):
        SIM_PS(queue, [GLOBAL_PS,], [LOCAL,], SEED, RC_buf, VC_buf, RHO_buf, IPFS_PS_buf, IPFR_buf, IPET_buf,
                                              IPKABS_buf, IPKSCA_buf, IPG_buf, ABS_buf, DABS_buf)
    else:
        GLOB        =  GLOBAL_PS//BATCH
        TMP_GLOBAL  =  max([32, (NC//LOCAL+1)*LOCAL])
        for ibatch in range(BATCH):
            SEED = rand()            
            SIM_PS(queue, [GLOB,], [LOCAL,], SEED, RC_buf, VC_buf, RHO_buf, IPFS_PS_buf, IPFR_buf, IPET_buf,
                                              IPKABS_buf, IPKSCA_buf, IPG_buf, ABS_buf, DABS_buf)
            UPDATE(queue, [TMP_GLOBAL,], [LOCAL,], ABS_buf, DABS_buf)
            if (ibatch%10==0):
                sys.stdout.write(' %d' % ibatch)
                sys.stdout.flush()
        sys.stdout.write('\n')
                                
cl.enqueue_copy(queue, ABS, ABS_buf)
queue.finish()
t0 = time.time()-t0
print("----- kernel call:         %.4f seconds" % t0)


t0 = time.time()

# save and plot final temperatures
#  E / H * pc   =  ABS*pc^2 / (VC*pc^3*RHO) * pc =  ABS / VC / RHO
Eabs  =  ABS / VC / RHO
TT    =  ipET(Eabs)

# Calculate intensity, frequencies FF, GLOBAL impact parameters
GLOBAL  = ((OFFS*NF)//LOCAL+1)*LOCAL
RES_buf =  cl.Buffer(context, mf.READ_WRITE, 4*OFFS*NF)
FF_buf  =  cl.Buffer(context, mf.READ_ONLY,  4*NF)
cl.enqueue_copy(queue, ABS_buf, asarray(TT, float32))
cl.enqueue_copy(queue, FF_buf,  asarray(FF, float32))

print("Save spectra...")
MAP(queue, [GLOBAL,], [LOCAL,], RC_buf, RHO_buf, FF_buf, IPKABS_buf, IPKSCA_buf, ABS_buf, RES_buf)
RES = zeros((OFFS, NF), float32)
cl.enqueue_copy(queue, RES, RES_buf)
print("... spectra done")

fp  = open('%s.spe' % INI['prefix'], 'w')
asarray([NF, OFFS], int32).tofile(fp)
asarray(FF,  float32).tofile(fp)
asarray(RES, float32).tofile(fp)
fp.close()

print("----- solve + spectra:         %.2f seconds" % (time.time()-t0))

tfile = '%s.T' % INI['prefix']
fp = open(tfile, 'w')
for i in range(NC):
    fp.write('%12.5e %12.5e\n' % (RC[i], TT[i]))
fp.close()
print("T = ", percentile(TT, (0,50,100)))
    
