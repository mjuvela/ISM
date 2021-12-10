#!/usr/bin/env python
import os, sys
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)
from   LOC_aux import *
from matplotlib.pylab import *

"""
Faster (?) methodx version for the basic calculation.
- The idea was to move the loop over transitions in the simulation step and the
  loop over cells in the equilibrium-equation step completely to the kernel side.
  This would avoid transfer of data between host and device and avoid the loops
  on the host side. In practice, there was no change in the run times, even when
  measuring just those parts of the program.
  Also tested simulations either with each work item updating separate SIJ and ESC,
  later summed up in the solver, or using single table with atomic operations.
  There was no clear difference (the latter was slightly slower?).
  
                                             With atomics                        No atomics
                                ------------------------------------------       -----------------------
  
                                                              Constant memory
                                ------------------------------------------------------------------------
         default      methodx    UPPER/LOWER   ALL EXCEPT GAU      ALL+GAU       ALL+GAU
  CPU     1.28         1.25        1.21          1.21               1.20         1.22
  GPU     1.15         1.21        1.20          1.22               1.19         1.16...1.15
  
  The use of constant memory does not appear to have any effect, not even on 
  the GPU (everything cached anyway?). CPU and GPU appear equally fast, which
  suggests that the kernel call overheads are insignificant. The only measurable
  difference is that on GPU the version without atomics is lightly faster.

  Use LIM also for the normal case:    
      GPU run time  1.15 -> 1.12 seconds
      CPU run time  1.22 -> 1.08 seconds !

"""

NI_LIMIT    = 1.0e-28     # limit to ignore low NI values

if (len(sys.argv)<2):
    print("Usage:  LOC.py  <ini-file>")
    sys.exit()    
INI         =  ReadIni(sys.argv[1])
MOL         =  ReadMolecule(INI['molecule'])
HFS         =  len(INI['hfsfile'])>0              # HFS with LTE components
OVERLAP     =  0
if (not(HFS)):
    OVERLAP =  len(INI['overlap'])>0              # individual overlapping lines
NFREQ       =  len(MOL.F)
LEVELS      =  INI['levels']
TRANSITIONS =  MOL.Transitions(LEVELS)            # how many transitions among LEVELS levels
CHANNELS    =  INI['channels']
WIDTH       =  INI['bandwidth'] / CHANNELS        # km/s, remains unchanged, even for HFS
TRANSITIONS =  MOL.TRANSITIONS
COOLING     =  INI['cooling']
if (COOLING & HFS):
    print("*** Cooling not implemented for HFS => cooling will not be calculated!")
    COOLING =  0
DOUBLE_COOL =  0

if (INI['angle']>0.0):  # cloud size defined by ini file
    GL         =  INI['angle']*ARCSEC_TO_RADIAN * INI['distance'] *PARSEC   # [cm] = 1D cloud radius
             
RADIUS, VOLUME, RHO, TKIN, CLOUD, ABU = ReadCloud1D(INI, MOL)

CELLS         =  len(RHO)   
CLOUD_AREA    =  4.0*pi          # total cloud area [GL^2]
CLOUD_VOLUME  = (4.0/3.0)*pi     # cloud volume, in units GL^3
min_sigma, max_sigma = np.min(CLOUD[:]['w']), np.max(CLOUD[:]['w'])
MAXCMP        =  1

if (HFS):
    BAND, CHANNELS, MAXCMP = ReadHFS(INI, MOL)     # CHANNELS becomes the maximum over all transitions
    print("HFS revised =>  CHANNELS %d" % (CHANNELS))
if (OVERLAP):
    OLBAND, OLTRAN, OLOFF, MAXCMP = ReadOverlap(INI['overlap'], MOL, WIDTH, TRANSITIONS, CHANNELS)
    
ESC_ARRAY     =  zeros((CELLS, TRANSITIONS), float32)
SIJ_ARRAY     =  zeros((CELLS, TRANSITIONS), float32)
    
WITH_CRT      =  INI['with_crt']
CRT_TAU       =  []
CRT_EMI       =  []
if (WITH_CRT):
    # CRT_TAU[CELLS, TRANSITIONS]  = optical depth / pc
    CRT_TAU   =  ReadDustTau('crt.opacity', GL, CELLS, TRANSITIONS)  # CRT_TAU[CELLS, TRANSITIONS]
    # CRT_EMI in the file       =  photons / s / Hz      / H  
    # ReadDustEmission returns  =  photons / s / channel / H  
    # rescale still to units    =  photons / s / channel / cm3  == multiply by density
    CRT_EMI   =  ReadDustEmission('crt.emission', CELLS, TRANSITIONS, WIDTH, MOL)
    for t in range(TRANSITIONS):  
        CRT_EMI[:,t] *=  RHO[:]          # CRT_EMI[CELLS, TRANSITIONS]


# Basic version: impact parameters follow the ~r probability and all rays have the same weight
# (emitted photons are divided between rays purely based on path lengths)
NRAY   =  INI['nray']

if (INI['GPU']):  LOCAL = 32
else:             LOCAL =  4
GLOBAL        =  IRound(NRAY,  64)
GLOBAL_SUM    =  IRound(CELLS, 64)  # kernel sum uses CELLS work items
NWG           =  1
NRAY_SPE      =  INI['nray_spe']
if (NRAY_SPE<1):
    NRAY_SPE  = NRAY
if (OVERLAP):
    GLOBAL    =  256
    NWG       =  GLOBAL//LOCAL
    NRAY      =  NWG*max([1, NRAY//NWG])
    NRAY_SPE  =  NWG*max([1, NRAY_SPE//NWG])
print("NRAY %d, NRAY_SPE %d" % (NRAY, NRAY_SPE))
    
DIRWEI =  zeros(NRAY, float32)
IP     =  zeros(NRAY, float32)
for i in range(NRAY):
    IP[i]      =  ((i+0.4999)/NRAY)**0.5
    DIRWEI[i]  =  1.0 

# Weighted version --- p(IP) = (k+1) * r^k,  P(IP) = r^(k+1)
#   IP      =   u^(1/(k+1))
#   weight  =   2/(k+1)  * r^(1-k)
for i in range(NRAY):
    IP[i]      =  ((i+0.5)/NRAY)**(1.0/(1.0+INI['alpha']))
    DIRWEI[i]  =  (2.0/(1.0+INI['alpha'])) * (IP[i])**(1.0-INI['alpha'])

# Precalculate the distance each ray travels in each of the shells = STEP[NRAY,CELLS]
# calculate the average length within each cell (for emission weighting)  = APL
# APL[icell] used to divide emitted ~ STEP/APL
# Before GetSteps1d, one must update NRAY, IP[NRAY], DIRWEI[NRAY]
STEP, APL        =  GetSteps1D(CELLS, RADIUS,    NRAY, IP, DIRWEI)
# try to correct sij using the VOLUME/APL ratio
PATH_CORRECTION  =  zeros(CELLS, float32)
PATH_CORRECTION  =  VOLUME/APL
PATH_CORRECTION /=  mean(PATH_CORRECTION)

SINGLE  =  []
MAXCHN  =  CHANNELS
if (OVERLAP):
    SINGLE  =  ones(TRANSITIONS, int32)  # by default all transitions single (case overlap)
    MAXCHN  =  CHANNELS
    for iband in range(OLBAND.Bands()):
        for icmp in range(OLBAND.Components(iband)):
            SINGLE[OLBAND.GetTransition(iband, icmp)] = 0  # not solitary transition
        MAXCHN = max([MAXCHN, CHANNELS+OLBAND.Channels(iband)])
    print("CHANNELS %d, MAXCHN %d" % (CHANNELS, MAXCHN))
    for i in range(TRANSITIONS): 
        print("   SINGLE[%2d] = %d" % (i, SINGLE[i]))

    
LIM = None        
if (HFS):        
    # HFS implies GAU[TRANSITIONS*CELLS*TRANSITIONS] !!
    GAU =  zeros((TRANSITIONS, CELLS, CHANNELS), float32)
    LIM =  zeros((TRANSITIONS, CELLS), cl.cltypes.int2)
    for tran in range(TRANSITIONS):
        # print("TRANSITION %3d,  COMPONENTS %d" % (tran, BAND[tran].N))
        channels =  BAND[tran].Channels()   # given transition, channels <= CHANNELS
        vel      =  (arange(channels)-0.5*(channels-1.0))*WIDTH
        for icell in range(CELLS):
            GAU[tran,icell,:] = 0.0             # CHANNELS elements
            s   =  CLOUD[icell]['w']            # sigma  [km/s]
            for icmp in range(BAND[tran].N):    # loop over components
                v0 = BAND[tran].VELOCITY[icmp]  # velocity shift (km/s) of this component
                w  = BAND[tran].WEIGHT[icmp]
                dv = vel-v0
                GAU[tran,icell,0:channels] +=  w*exp(-dv*dv/(s*s))
            # normalisation
            GAU[tran,icell,:] /= sum(GAU[tran,icell,:])
            # figure out limits of significant channels
            m  =  nonzero(GAU[tran,icell,:]>1.0e-5)
            LIM[tran,icell]['x'] = m[0][0]
            LIM[tran,icell]['y'] = m[0][-1]
            # print("  %3d - %3d" % (m[0][0], m[0][-1]))            
            if (tran==-1):
                clf()
                plot(GAU[tran,icell,:])
                show()
                sys.exit()
else:  # overlap and normal calculatoin =>  GAU[CELLS,CHANNELS]
    GAU    =  zeros((CELLS, CHANNELS), float32)
    v      =  (arange(CHANNELS)-0.5*(CHANNELS-1.0)+0.75)*WIDTH  # 0.75 FUDGE TO MATCH CPPSIMU!!
    for icell in range(CELLS):
        s             =  CLOUD[icell]['w']
        GAU[icell,:]  =  exp(-v*v/(s*s))
        GAU[icell,:] /=  sum(GAU[icell,:])
    # overlap will also use LIM .... 2020-06-22 LIM now used also in the default case
    LIM    =   zeros(CELLS, cl.cltypes.int2)
    for icell in range(CELLS):
        m  =  nonzero(GAU[icell,:]>1.0e-5)
        LIM[icell]['x'] = m[0][ 0]
        LIM[icell]['y'] = m[0][-1]


GNO, SIGMA0, SIGMAX =  55, 0.0, 0.0
        
# If cabfile is specified, read the relative abundances of the collisional partners
# The file is plain floating point numbers, the cell index runs slow, partner fast.
# If cabfile is NOT provided, set default abundances based on the name of the collision
# partner -- these can be overridden by settings in the ini file
PARTNERS = MOL.PARTNERS
print("Molecule has %d collisional partners" % PARTNERS)
CAB  =  zeros((CELLS, PARTNERS), float32)
if (len(INI['cabfile'])>1):
    # We provide arrays for the abundance of each collisional partner
    try:
        CAB = fromfile(INI['cabfile'], float32).reshape(CELLS, PARTNERS)
    except:
        print("Error reading abundances of collisional partners: %s" % INI['cabfile'])
        sys.exit()
else: # default abundances from MOL
    for i in range(PARTNERS):
        # Note: sum of PartnerAbundance() over all collisional partners is 1.0
        cab = MOL.CABU[i]
        print("   COLLISIONAL PARTNER ABUNDANCE: %5s  %10.3e" % (MOL.PNAME[i], MOL.CABU[i]))
        CAB[:,i] = cab

for i in range(PARTNERS):
    print("  1-0   partner %d  %10s  ---  C[0] = %10.3e   ABU[0] = %10.3e" % \
           (i, MOL.PNAME[i], MOL.C(1, 0, 10.0, i), CAB[0*PARTNERS,i]))


TIME = 0.0
def Seconds():
    global TIME
    t = time.time()-TIME
    TIME = time.time()
    return t
    

platform, device, context, queue,  mf = InitCL(INI['GPU'], INI['platforms'])

OPT = "-D NRAY=%d -D CHANNELS=%d -D WIDTH=%.5f -D CELLS=%d -D LOCAL=%d -D GLOBAL=%d -D LEVELS=%d \
-D TRANSITIONS=%d -D GNO=%d -D SIGMA0=%.5f -D SIGMAX=%.4f -D GL=%.4e -D COOLING=%d -D NRAY_SPE=%d \
-D DOUBLE_COOL=%d -D WITH_CRT=%d -D NWG=%d -D MAXCMP=%d -D MAXCHN=%d -D BGSUB=1 -D LOWMEM=%d \
-D GLOBAL=%d -D LOCAL=%d" % \
(NRAY, CHANNELS, WIDTH, CELLS, LOCAL, GLOBAL, LEVELS, 
TRANSITIONS, GNO, SIGMA0, SIGMAX, GL, COOLING, NRAY_SPE, 
DOUBLE_COOL, WITH_CRT, NWG, MAXCMP, MAXCHN, INI['lowmem'], GLOBAL, LOCAL)

print("GLOBAL %d, GLOBAL_SUM %d, LOCAL %d, CELLS %d" % (GLOBAL, GLOBAL_SUM, LOCAL, CELLS))

METHOD_X = INI['method_x']

if (HFS):
    source  = open(INSTALL_DIR+"/kernel_update_1d_hfs_py.c").read()
elif (OVERLAP):
    source  = open(INSTALL_DIR+"/kernel_update_1d_ol_py.c").read()
else:
    if (METHOD_X):
        source  = open(INSTALL_DIR+"/kernel_update_1d_pyx.c").read()
    else:
        source  = open(INSTALL_DIR+"/kernel_update_1d_py.c").read()

    
program = cl.Program(context, source).build(OPT)


if (METHOD_X):
    # using a kernel that loops over transitions
    # CLOUD ---
    # UPPER 
    X_UPPER_buf =  cl.Buffer(context, mf.READ_ONLY,   4*TRANSITIONS)
    cl.enqueue_copy(queue, X_UPPER_buf, asarray(MOL.TRANSITION[:,0].copy(), int32))
    # GAU ---
    # Aul
    X_Aul_buf   =  cl.Buffer(context, mf.READ_ONLY,   4*TRANSITIONS)
    cl.enqueue_copy(queue, X_Aul_buf, asarray(MOL.A, float32))
    # Ab
    X_Ab_buf    =  cl.Buffer(context, mf.READ_ONLY,   4*TRANSITIONS)
    cl.enqueue_copy(queue, X_Ab_buf, asarray(MOL.BB, float32))
    # GG
    X_GG_buf    =  cl.Buffer(context, mf.READ_ONLY,   4*TRANSITIONS)
    cl.enqueue_copy(queue, X_GG_buf, asarray(MOL.GG, float32))
    # G
    X_G_buf     =  cl.Buffer(context, mf.READ_ONLY,   4*LEVELS)
    cl.enqueue_copy(queue, X_G_buf, asarray(MOL.G, float32))
    # GN
    X_GN_buf    =  cl.Buffer(context, mf.READ_ONLY,   4*TRANSITIONS)
    GN          =  asarray((C_LIGHT/(1.0e5*WIDTH*MOL.F)) * GL, float32)
    cl.enqueue_copy(queue, X_GN_buf, GN)
    # DIRWEI ---> WEI_buf 
    # VOLUME ---> VOL_buf
    # STEP   ---> STEP_buf
    # APL    ---> APL_buf 
    # BG
    X_BG_buf    =  cl.Buffer(context, mf.READ_ONLY,   4*TRANSITIONS)
    BGPHOT      =  Planck(MOL.F, INI['Tbg'])*pi*CLOUD_AREA/(PLANCK*C_LIGHT)*(1.0e5*WIDTH) / (CLOUD_VOLUME*GL)
    BG          =  asarray(BGPHOT/NRAY, float32)
    cl.enqueue_copy(queue, X_BG_buf, BG)
    # IP    ---> IP_buf 
    # NI
    X_NI_buf    =  cl.Buffer(context, mf.READ_WRITE,  4*CELLS*LEVELS)
    # NBNB
    X_NBNB_buf  =  cl.Buffer(context, mf.READ_WRITE,  4*CELLS*TRANSITIONS)
    # NTRUE
    X_NTRUE_buf =  cl.Buffer(context, mf.READ_WRITE,  4*NRAY*TRANSITIONS*CHANNELS)    
    # SIJ
    X_SIJ_buf   =  cl.Buffer(context, mf.READ_WRITE,  4*NRAY*CELLS*TRANSITIONS)
    # ESC
    X_ESC_buf   =  cl.Buffer(context, mf.READ_WRITE,  4*NRAY*CELLS*TRANSITIONS)

    # For Clear_X
    # LOWER 
    X_LOWER_buf =  cl.Buffer(context,   mf.READ_ONLY,   4*TRANSITIONS)
    cl.enqueue_copy(queue, X_LOWER_buf, asarray(MOL.TRANSITION[:,1].copy(), int32))
    # FREQ
    X_FREQ_buf  =  cl.Buffer(context,   mf.READ_ONLY,   4*TRANSITIONS)
    cl.enqueue_copy(queue, X_FREQ_buf,  asarray(MOL.F, float32))
    # E
    X_E_buf     =  cl.Buffer(context,   mf.READ_ONLY,   4*LEVELS)
    cl.enqueue_copy(queue, X_E_buf,     asarray(MOL.E, float32))
    
    # For Solve_X
    # TKIN
    X_TKIN_buf =  cl.Buffer(context,    mf.READ_ONLY,   4*CELLS)
    cl.enqueue_copy(queue, X_TKIN_buf,  asarray(TKIN, float32))
    # ABU
    X_ABU_buf  =  cl.Buffer(context,    mf.READ_ONLY,   4*CELLS)
    cl.enqueue_copy(queue, X_ABU_buf,   asarray(ABU, float32))
    # WRK
    X_WRK_buf  =  cl.Buffer(context,    mf.READ_WRITE,  4*GLOBAL*LEVELS*(LEVELS+1))
    # ERR
    X_ERR_buf  =  cl.Buffer(context,    mf.WRITE_ONLY,  4*CELLS)
    # PC  <- PATH_CORRECTION
    X_PC_buf   =  cl.Buffer(context,    mf.READ_ONLY,   4*CELLS)
    cl.enqueue_copy(queue, X_PC_buf,    asarray(PATH_CORRECTION, float32))    
    
    
    kernel_clear_x   =   program.Clear_X
    #                                     0     1     2     3     4     5     6      7       8   
    #                                     SIJ   ESC   NI    NBNB  Aul   GG    UPPER  LOWER   FREQ
    kernel_clear_x.set_scalar_arg_dtypes([None, None, None, None, None, None, None,  None,   None])    
    
    kernel_sim_x  =   program.Update_X
    #                                      0      1      2     3     4     5     6     7        8        9   
    #                                      CLOUD  UPPER  GAU   Aul   Ab    GG    GN    DIRWEI   VOLUME   STEP
    kernel_sim_x.set_scalar_arg_dtypes([None,  None,  None, None, None, None, None,    None,    None, None,
    # 10   11    12    13    14     15     16    17    18  
    # APL  BG    IP    NI    NBNB   NTRUE  SIJ   ESC   LIM 
    None,  None, None, None, None,  None,  None, None, None])
    
    kernel_solve_x   =   program.Solve_X
    #                                     0     1      2      3     4      5         6         7       
    #                                     A     UPPER  LOWER  E     GG     PARTNERS  NTKIN     NCUL    
    kernel_solve_x.set_scalar_arg_dtypes([None, None,  None,  None, None,  np.int32, np.int32, np.int32,
    # 8         9     10    11    12    13    14    15    16    17    18    19    20    21
    # MOL_TKIN  CUL   C     CABU  RHO   TKIN  ABU   NI    SIJ   ESC   WRK   PC    ERR   VOLUME
    None,       None, None, None, None, None, None, None, None, None, None, None, None, None ])

    
else:
    # old methods .... here used only for HFS and OVERLAP
           
    # Create the kernels
    kernel_sim    =   program.Update
    kernel_sum    =   program.Sum
    kernel_clear  =   program.Clear
    kernel_solve  =   program.SolveCL
    if (HFS):
        if (WITH_CRT):
            kernel_sim.set_scalar_arg_dtypes([None,None,np.float32,np.float32,np.float32,None,None,None,None,
            np.float32,None,None,None,np.int32,np.int32,None,None,None,None])
        else:
            kernel_sim.set_scalar_arg_dtypes([None,None,np.float32,np.float32,np.float32,None,None,None,None,
            np.float32,None,None,None,np.int32,np.int32,None,None])
    elif (OVERLAP):
        if (WITH_CRT):
            #                                 CLOUD VOL   Aul    Blu   GAU   LIM   GN    DIRWEI
            kernel_sim.set_scalar_arg_dtypes([None, None, None,  None, None, None, None, None,
            # STEP  APL   IP    BG    NU    NBNB   ARES  SINGLE   NTRUE  CRT_TAU  CRT_EMI
            None,   None, None, None, None, None,  None, None,    None,  None,    None])
            ####
            kernel_ol = program.Overlap
            #                                 CLOUD VOL   Aul    Blu   GAU   LIM   GN    DIRWEI
            kernel_ol.set_scalar_arg_dtypes([ None, None, None,  None, None, None, None, None,
            # STEP  APL   IP    BG    NU    NBNB   ARES  NCMP      NCHN      TRAN  OFF   NTRUE  CRT_TAU CRT_EMI
            None,   None, None, None, None, None,  None, np.int32, np.int32, None, None, None,  None,   None])
        else:
            #                                 CLOUD VOL   Aul    Blu   GAU   LIM   GN    DIRWEI
            kernel_sim.set_scalar_arg_dtypes([None, None, None,  None, None, None, None, None,
            # STEP  APL   IP    BG    NU    NBNB   ARES  SINGLE   NTRUE
            None,   None, None, None, None, None,  None, None,    None])
            ####
            kernel_ol = program.Overlap
            #                                 CLOUD VOL   Aul    Blu   GAU   LIM   GN    DIRWEI
            kernel_ol.set_scalar_arg_dtypes([ None, None, None,  None, None, None, None, None,
            # STEP  APL   IP    BG    NU    NBNB   ARES  NCMP      NCHN      TRAN  OFF   NTRUE
            None,   None, None, None, None, None,  None, np.int32, np.int32, None, None, None])
    else:    
        if (COOLING):
            kernel_sim.set_scalar_arg_dtypes([None,None,np.float32,np.float32,np.float32,None,None,None,None,np.float32,
            None,None,None,None])
        else:
            if (WITH_CRT):
                #                                 CLOUD, GAU,  LIM,  Aul,        A_b,        GN          LIM  
                kernel_sim.set_scalar_arg_dtypes([None,  None, None, np.float32, np.float32, np.float32, None,
                # DIRWEI  VOLUME   STEP   APL    BG          IP    NI    NTRUES  ARES 
                None,     None,    None,  None,  np.float32, None, None, None,   None,
                #  TRAN   CRT_TAU   CRT_EMI
                np.int32,  None,    None])
            else:
                #                                 CLOUD, GAU,  LIM   Aul,        A_b,        GN         
                kernel_sim.set_scalar_arg_dtypes([None,  None, None, np.float32, np.float32, np.float32,
                # DIRWEI, VOLUME, STEP, APL,  BG,         IP,   NI,   NTRUES ARES
                None,     None,   None, None, np.float32, None, None, None,  None])
            
    kernel_sum.set_scalar_arg_dtypes([None,None,None])
    kernel_clear.set_scalar_arg_dtypes([None,])    
            
    kernel_solve.set_scalar_arg_dtypes(
    # 0        1     2     3     4     5         6         7         8     9     10    11   
    [np.int32, None, None, None, None, np.int32, np.int32, np.int32, None, None, None, None,
    # 12  13    14    15    16    17    18    19    20 
    None, None, None, None, None, None, None, None, None ])
        
    if (OVERLAP):
        maxi       =  max([max([NRAY, NRAY_SPE])*MAXCHN, NWG*MAXCHN*max([4, TRANSITIONS])])
        NTRUE_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*maxi)
        STAU_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*maxi)
        TAU_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NWG*TRANSITIONS*MAXCHN)
        NU_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*CELLS*TRANSITIONS)
        NBNB_buf   =  cl.Buffer(context, mf.READ_ONLY,  4*CELLS*TRANSITIONS)
        ARES_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*2*NWG*TRANSITIONS*CELLS)
        RES_buf    =  cl.Buffer(context, mf.WRITE_ONLY, 4*2*CELLS*TRANSITIONS)
        Aul_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*TRANSITIONS)
        Blu_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*TRANSITIONS)
        GN_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*TRANSITIONS)
        SINGLE_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SINGLE)
        BG_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*TRANSITIONS)
        TRAN_buf   =  cl.Buffer(context, mf.READ_ONLY,  4*TRANSITIONS)
        OFF_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*TRANSITIONS)
    else:
        ARES_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*2*CELLS*NRAY)
        RES_buf    =  cl.Buffer(context, mf.WRITE_ONLY, 4*2*CELLS)    
        NTRUE_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*NRAY*CHANNELS)

        
#    if (HFS | OVERLAP): # HFS has [transitions,cells], OL has [cells]
LIM_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=LIM) 


kernel_spe    =   program.Spectra
if (HFS):
    if (WITH_CRT):
        #                                 CLOUD  GAU   GN          NI    BG
        kernel_spe.set_scalar_arg_dtypes([None,  None, np.float32, None, np.float32,
        # emissivity  IP     STEP   NTRUE_SPE  STAU 
        np.float32,   None,  None,  None,      None, 
        # tran    NCHN      CRT_TAU  CRT_EMI
        np.int32, np.int32, None,    None])
    else:
        kernel_spe.set_scalar_arg_dtypes([None,None,np.float32,None,np.float32,np.float32,None,
        None,None,None,np.int32,np.int32])
elif (OVERLAP):
    # Individual transitions -- kernel_spe the same as without OVERLAP ????
    if (WITH_CRT):
        #                                 CLOUD  GAU   GN          NI    BG        
        kernel_spe.set_scalar_arg_dtypes([None,  None, np.float32, None, np.float32,
        # emissivity  IP     STEP   NTRUE_SPE  STAU   RHO   COLDEN  TRAN      CRT_TAU  CRT_EMI
        np.float32,   None,  None,  None,      None,  None, None,   np.int32, None,    None])
    else:
        #                                 CLOUD GAU   GN          NI    BG          emis0       IP    STEP
        kernel_spe.set_scalar_arg_dtypes([None, None, np.float32, None, np.float32, np.float32, None, None,
        # NTRUE SUM_TAU  RHO   COLDEN
        None,   None,    None, None ])
    # Bands of overlapping lines
    if (WITH_CRT):
        # overlapping spectra
        kernel_spe_ol = program.Spectra_OL
        #                                   FREQ         CLOUD GAU   LIM   GN    NU    Aul   NBNB  STEP  IP  
        kernel_spe_ol.set_scalar_arg_dtypes([np.float32, None, None, None, None, None, None, None, None, None,
        #  BG NTRUE  STAU   NCMP      NCHN      TRAN  OFF   CRT_TAU   CRT_EMI
        None, None,  None,  np.int32, np.int32, None, None, None,     None ])
    else:
        # overlapping spectra
        kernel_spe_ol = program.Spectra_OL
        #                                    FREQ        CLOUD GAU   LIM   GN    NU    Aul   NBNB  STEP  IP  
        kernel_spe_ol.set_scalar_arg_dtypes([np.float32, None, None, None, None, None, None, None, None, None,
        #  BG NTRUE  STAU   NCMP      NCHN      TRAN  OFF
        None, None,  None,  np.int32, np.int32, None, None])
else:
    if (WITH_CRT):
        kernel_spe.set_scalar_arg_dtypes([None,None,np.float32,None,np.float32,np.float32,
        None,None,None,None,None,None,     np.int32, None, None])
    else:
        kernel_spe.set_scalar_arg_dtypes([None,None,np.float32,None,np.float32,np.float32,
        None,None,None,None,None,None, np.int32])
        
            
# things common with METHOD_X
NI_buf         =  cl.Buffer(context, mf.READ_ONLY,  4*2*CELLS)
CLOUD_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=CLOUD) # [ Vrad, Rc, dummy, sigma ]
GAU_buf        =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=GAU)   #  depends on HFS
WEI_buf        =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=DIRWEI)
VOL_buf        =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=VOLUME)
APL_buf        =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=APL)
# the following may also be used to write spectra, with NRAY_SPE>NRAY
IP_buf         =  cl.Buffer(context, mf.READ_ONLY, 4*max([NRAY, NRAY_SPE]))
STEP_buf       =  cl.Buffer(context, mf.READ_ONLY, 4*max([NRAY, NRAY_SPE])*CELLS)
PL_buf         =  cl.Buffer(context, mf.READ_ONLY,  4*CELLS)
RHO_buf        =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=RHO)
COLDEN_buf     =  cl.Buffer(context, mf.WRITE_ONLY, 4*4*NRAY_SPE)   # [NRAY_SPE] float4
NTRUE_SPE_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*NRAY_SPE*CHANNELS)      # [NRAY, CHANNELS]

if (WITH_CRT):    
    CRT_TAU_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=CRT_TAU)
    CRT_EMI_buf = cl.Buffer(context, mf.READ_ONLY, 4*CELLS*TRANSITIONS)
    cl.enqueue_copy(queue, CRT_EMI_buf, CRT_EMI)
        
cl.enqueue_copy(queue, STEP_buf,   STEP)  # will change when we write spectra !!
cl.enqueue_copy(queue, IP_buf, IP)        # will change when we write spectra !!

if (COOLING):
    if (DOUBLE_COOL):
        COOL_buf  = cl.Buffer(context, mf.READ_WRITE, 8*CELLS*GLOBAL)
    else:
        COOL_buf  = cl.Buffer(context, mf.READ_WRITE, 4*CELLS)

WRK  = zeros(CELLS, cl.cltypes.float2)




# For SolveCL, completely separate buffers 
BATCH      =  min([16384, IRound(CELLS,32)])
S_WRK_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*BATCH*LEVELS*(LEVELS+1))
S_RHO_buf  =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
S_TKIN_buf =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
S_ABU_buf  =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
S_SIJ_buf  =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH*TRANSITIONS)
S_ESC_buf  =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH*TRANSITIONS)    
S_PC_buf   =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
S_NI_buf   =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH*LEVELS)
if (0):
    S_RES_buf  =  cl.Buffer(context, mf.WRITE_ONLY, 4*BATCH*LEVELS)
else:
    S_RES_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*BATCH*LEVELS)
# molecule basic data
MOL_A_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.A)
MOL_UL_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.TRANSITION)
MOL_E_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.E)
MOL_G_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.G)
# TKIN
NTKIN      =  len(MOL.TKIN[0])
for i in range(1, PARTNERS):
    if (len(MOL.TKIN[i])!=NTKIN):
        print("SolveCL assumes the same number of Tkin for each collisional partner!!"), sys.exit()
MOL_TKIN = zeros((PARTNERS, NTKIN), float32)
for i in range(PARTNERS):
    MOL_TKIN[i, :]  =  MOL.TKIN[i][:]
MOL_TKIN_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL_TKIN)
# CUL  -- levels are included separately for each partner... must have the same number of rows!
#         KERNEL ASSUMES IT IS THE SAME TRANSITIONS, IN THE SAME ORDER
NCUL  =  MOL.CUL[0].shape[0]
if (MOL.CUL[i].shape[0]!=NCUL):
    print("SolveCL assumes the same number of C rows for each collisional partner!!"),  sys.exit()
CUL   =  zeros((PARTNERS,NCUL,2), int32)  # specifies the (u,l) levels for row of collisional coefficients
for i in range(PARTNERS):
    CUL[i, :, :]  =  MOL.CUL[i][:,:]
    # KERNEL USES ONLY THE CUL ARRAY FOR THE FIRST PARTNER -- CHECK THAT TRANSITIONS ARE IN THE SAME ORDER
    delta =   np.max(ravel(MOL.CUL[i]-MOL.CUL[0]))
    if (delta>0):
        print("*** ERROR: SolveCL assumes all partners have C in the same order of transitions!!"), sys.exit()
MOL_CUL_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=CUL)
# C
C    =  zeros((PARTNERS, NCUL, NTKIN), float32)
for i in range(PARTNERS):   
    C[i, :, :]  =  MOL.CC[i][:, :]
MOL_C_buf  = cl.Buffer(context, mf.READ_ONLY, C.nbytes)
cl.enqueue_copy(queue, MOL_C_buf, C)
# abundance of collisional partners
MOL_CABU_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.CABU)    
  

# kernel_clear -- to zero the SIJ and ESC arrays on the device side
if (OVERLAP):
    WRK2  = zeros((CELLS, TRANSITIONS), np.float32)
    RES   = zeros((CELLS, TRANSITIONS), cl.cltypes.float2)
    # Update buffers:  Aul, Blu, GN, BG
    tmp   =  asarray(MOL.A,   float32)
    cl.enqueue_copy(queue, Aul_buf, tmp)
    tmp   =  asarray(MOL.BB,  float32)
    cl.enqueue_copy(queue, Blu_buf, tmp)
    tmp   =  asarray(C_LIGHT/(1.0e5*WIDTH*MOL.F), float32)
    cl.enqueue_copy(queue, GN_buf, tmp)
    #print("Aul ", MOL.A)
    #print("BB  ", MOL.BB)
    #print("GN  ", tmp)    
    for i in range(TRANSITIONS):
        tmp[i] = Planck(MOL.F[i], INI['Tbg'])*pi*CLOUD_AREA/(PLANCK*C_LIGHT)*(1.0e5*WIDTH) / (CLOUD_VOLUME*GL*NRAY)
    cl.enqueue_copy(queue, BG_buf, tmp)
    #print("BG  ", tmp)    
    #sys.exit()
    
    
# Read or generate NI_ARRAY
NI_ARRAY  =  zeros((CELLS, LEVELS), float32)
ok = 0
if (len(INI['load'])>1):
    try:
        fp            = open(INI['load'], "rb") 
        cells, levels = fromfile(fp, int32, 2)
        if (cells!=CELLS):
            print("File %s has wrong number of cells" % INI['load'])
            sys.exit()
        if (levels!=LEVELS):
            print("  Reading save with %d levels to current array with %d levels" % (levels, LEVELS))
            ml         =  max([levels, LEVELS])
            buff       =  zeros(ml, float32)
            buff[0:ml] =  NI_LIMIT
            for icell in range(CELLS):
                NI[icell, 0:levels] =  fromfile(fp, float32, levels)
        else:   # same number of levels in the file and in current calculation
            NI_ARRAY = fromfile(fp, float32).reshape(CELLS, LEVELS)
        NI_ARRAY = clip(NI_ARRAY, NI_LIMIT, 1e10)
        fp.close()
        ok = 1
    except:
        ok = 0
if (ok):
    print("*** Old level populations read ok")
else:
    print("*** Level populations reset to LTE")
    for icell in range(CELLS):
        NI_ARRAY[icell,:] = RHO[icell]*ABU[icell] * MOL.Partition(arange(LEVELS), TKIN[icell])
    # save level populations -- format not compatible with Cppsimu
    if (0):
        fp = open(INI['save'], "wb") 
        asarray([CELLS, LEVELS], int32).tofile(fp)
        asarray(NI_ARRAY, float32).tofile(fp)
        fp.close()

        
        
##########################################################################################




def Simulate_X(INI, MOL, NITER):
    global X_SIJ_buf, X_ESC_buf, X_NI_buf, X_NBNB_buf, X_Aul_buf, X_GG_buf, X_UPPER_buf, X_LOWER_buf, X_FREQ_buf
    global CLOUD_buf, GAU_buf, X_Ab_buf, X_GN_buf, WEI_buf, VOL_buf, STEP_buf, APL_buf, X_BG_buf, IP_buf, X_NTRUE_buf
    global X_E_buf, X_TKIN_buf, X_ABU_buf, X_WRK_buf, X_PC_buf, X_ERR_BUF, MOL_TKIN_buf, MOL_CUL_buf, MOL_C_buf
    global MOL_CABU_buf, RHO_buf
    global PARTNERS, NTKIN, NCUL, GLOBAL, LOCAL
    # ---- Upload the NI[upper] and NB_NB[tran] values
    cl.enqueue_copy(queue, X_NI_buf, asarray(NI_ARRAY, float32))    
    t1, Tsim, Tsol = time.time(), 0.0, 0.0    
    for III in range(NITER):
        t0 = time.time()
        # ---- Clear SIJ, ESC, and update NBNB
        #                                          0          1          2         3           4          5       
        kernel_clear_x(queue, [GLOBAL,], [LOCAL,], X_SIJ_buf, X_ESC_buf, X_NI_buf, X_NBNB_buf, X_Aul_buf, X_GG_buf,
        # 6          7            8          
        X_UPPER_buf, X_LOWER_buf, X_FREQ_buf)    
        # ---- Do simulation => SIJ, ESC values
        #                                      0          1            2        3        4          5       
        kernel_sim_x(queue,[GLOBAL,],[LOCAL,], CLOUD_buf, X_UPPER_buf, GAU_buf, LIM_buf, X_Aul_buf, X_Ab_buf, 
        #  6      7          8        9        10        11       12        13      14        15         
        X_GG_buf, X_GN_buf,  WEI_buf, VOL_buf, STEP_buf, APL_buf, X_BG_buf, IP_buf, X_NI_buf, X_NBNB_buf, 
        #  16        17         18       
        X_NTRUE_buf, X_SIJ_buf, X_ESC_buf)
        queue.finish()
        Tsim += time.time()-t0
        t0    = time.time()
        # ---  Solve equilibrium equations... kernel will sum SIJ and ESC over NRAY elements
        kernel_solve_x(queue, [GLOBAL,], [LOCAL,], X_Aul_buf, X_UPPER_buf, X_LOWER_buf, X_E_buf, X_G_buf,
        PARTNERS, NTKIN, NCUL, MOL_TKIN_buf, MOL_CUL_buf,  MOL_C_buf, MOL_CABU_buf,
        RHO_buf, X_TKIN_buf, X_ABU_buf,  X_NI_buf, X_SIJ_buf, X_ESC_buf, X_WRK_buf, X_PC_buf, X_ERR_buf, VOL_buf)
        queue.finish()
        Tsol += time.time()-t0
    t1 = time.time()-t1
    print("--------------------------------------------------------------------------------")
    print("Simulate_X()")
    print("       simulation  %.4f    solve %.4f  seconds per iteration" % (Tsim/NITER, Tsol/NITER))
    print("       total run time %.4f seconds, %.4f seconds per iteration" % (t1, t1/NITER))
    print("--------------------------------------------------------------------------------")
    

    
    


def Simulate(INI, MOL):
    global WIDTH, CLOUD_AREA, CLOUD_VOLUME, GL, SIJ_ARRAY
    tmp_1  =  C_LIGHT*C_LIGHT/(8.0*pi)
    Tbg    =  INI['Tbg']
    SUM_COOL, LEVL_COOL, HF = [], [], []
    if (COOLING):
        SUM_COOL = zeros(CELLS,        float32)
        HF       = zeros(TRANSITIONS,  float32)
        LEV_COOL = zeros(CELLS*GLOBAL, [float32,float64][DOUBLE_COOL>0])
        for t in range(TRANSITIONS):
            # photon counts = true photons in/through cell, divided by Vcloud
            # true_in_cell/Vcloud *= Vcloud/Vcell    ==     /= VOLUME
            HF[t]  = MOL.F[t]*PLANCK   #     / VOLUME # energy per unit density
    for tran in range(TRANSITIONS):
        if (COOLING>0):
            LEV_COOL[:] = 0.0
            cl.enqueue_copy(queue, COOL_buf, LEV_COOL)  # float32 or float64, depending on DOUBLE_COOL
        upper, lower  =  MOL.T2L(tran)
        A_b           =  MOL.BB[tran]
        A_Aul         =  MOL.A[tran]
        freq          =  MOL.F[tran]
        gg            =  MOL.G[upper]/MOL.G[lower]
        # all photon numbers = real number divided by cloud volume
        BGPHOT        =  Planck(freq, Tbg)*pi*CLOUD_AREA/(PLANCK*C_LIGHT)*(1.0e5*WIDTH) / (CLOUD_VOLUME*GL)
        BG            =  BGPHOT / NRAY                       # photons per ray, before possible DIRWEI!=1
        A_gauss_norm  = (C_LIGHT/(1.0e5*WIDTH*freq)) * GL    # GRID_LENGTH multiplied to gauss norm
        # print("A_b %.3e, A_Aul %.3e, freq %.3e, gg %.3f, GN %.3e, BG %.3e" % (A_b, A_Aul, freq, gg, A_gauss_norm, BG))
        if (tran==0): sys.stdout.write('   %d' % tran)
        else:         sys.stdout.write(", %d" % tran)
        if (tran==(TRANSITIONS-1)): sys.stdout.write("\n") 
        sys.stdout.flush()      
        # Clear ARES work area on the device side
        kernel_clear(queue, [GLOBAL,], [LOCAL,], ARES_buf)      
        # Upload the NI[upper] and NB_NB[tran] values for the current transition
        nu   =  NI_ARRAY[:,upper]
        nl   =  NI_ARRAY[:,lower]
        tmp  =  (tmp_1 * A_Aul * (nl*gg-nu)) / (freq*freq)
        tmp[nonzero(abs(tmp)<1.0e-25)] =  2.0e-25
        tmp[nonzero(tmp<(-1.0e-3))]    = -1.0e-3
        WRK[:]['x']  = nu      #  NI(upper)
        WRK[:]['y']  = tmp     #  NB_NB(cell, tran)
        cl.enqueue_copy(queue, NI_buf, WRK)      
        # A single kernel call does all the rays
        Seconds() 
        if (HFS):
            if (WITH_CRT):
                kernel_sim(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, GAU_buf, A_Aul, A_b, A_gauss_norm, 
                WEI_buf, VOL_buf, STEP_buf, APL_buf, BG, IP_buf, NI_buf, ARES_buf,
                tran, BAND[tran].Channels(), NTRUE_buf, LIM_buf, CRT_TAU_buf, CRT_EMI_buf)
            else:
                kernel_sim(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, GAU_buf, A_Aul, A_b, A_gauss_norm, 
                WEI_buf, VOL_buf, STEP_buf, APL_buf, BG, IP_buf, NI_buf, ARES_buf,
                tran, BAND[tran].Channels(), NTRUE_buf, LIM_buf)
        elif (OVERLAP):
            if (WITH_CRT):
                kernel_sim(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, VOL_buf, Blu_buf, GAU_buff, LIM_buf,
                GN_buf, WEI_buf, STEP_buf, APL_buf, IP_buf, BG_buf, NU_buf, NBNB_buf, ARES_buf, SINGLE_buf,
                NTRUE_buf, TRAN_buf, CRT_TAU_buf, CRT_EMI_buf)                
            else:
                kernel_sim(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, VOL_buf, Blu_buf, GAU_buff, LIM_buf,
                GN_buf, WEI_buf, STEP_buf, APL_buf, IP_buf, BG_buf, NU_buf, NBNB_buf, ARES_buf, SINGLE_buf,
                NTRUE_buf)
        else:
            if (COOLING):
                kernel_sim(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, GAU_buf, LIM_buf, A_Aul, A_b, A_gauss_norm, 
                WEI_buf, VOL_buf, STEP_buf, APL_buf, BG, IP_buf, NI_buf, NTRUE_buf, ARES_buf, COOL_buf)
                c.enqueue_copy(queue, LEV_COOL, COOL_buf)   # float32 or float64  [CELLS*GLOBAL]
                LEV_COOL.shape = (GLOBAL, CELLS)
                for icell in range(CELLS):
                    SUM_COOL[icell] += sum(LEV_COOL[:,icell]) * HF[tran] / VOLUME[icell]    # per cm3 !!
            else:
                if (WITH_CRT):
                    kernel_sim(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, GAU_buf, LIM_buf, A_Aul, A_b, A_gauss_norm, 
                    WEI_buf, VOL_buf, STEP_buf, APL_buf, BG, IP_buf, NI_buf, NTRUE_buf, ARES_buf, 
                    tran, CRT_TAU_buf, CRT_EMI_buf)
                else:
                    kernel_sim(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, GAU_buf, LIM_buf, A_Aul, A_b, A_gauss_norm, 
                    WEI_buf, VOL_buf, STEP_buf, APL_buf, BG, IP_buf, NI_buf, NTRUE_buf, ARES_buf)
        # Run second kernel to calculate sum of SIJ and ESC for each cell, ... and to divided each SIJ by volume!!
        kernel_sum(queue, [GLOBAL_SUM,], [LOCAL,], ARES_buf, RES_buf, VOL_buf)
        #/ Read the result for the current transition == SIJ, ESC
        cl.enqueue_copy(queue, WRK, RES_buf)
        SIJ_ARRAY[:, tran]  =  WRK[:]['x']   #  [CELLS, TRANSITIONS]
        ESC_ARRAY[:, tran]  =  WRK[:]['y']
    #  for tran
    if (COOLING):
        print("BRUTE_COOL: %10.3e", mean(asarray(SUM_COOL, float64)))
        fpb = open(COOLFILE, "wb") 
        asarray(SUM_COOL, float32).tofile(fpb)
        fpb.close()
    m  = nonzero(~isfinite(SIJ_ARRAY))
    if (len(m[0])>0):  
        print("       *** SIJ_ARRAY: NUMBER OF BAD VALUES = %d" % len(m[0]))
    m  = nonzero(SIJ_ARRAY<-1.0e-12)
    if (len(m[0])>0):   
        print("       *** SIJ_ARRAY: NUMBER OF NEGATIVE VALUES = %d" % len(m[0]))
        for i in range(len(m[0])):
            print("  CELL %3d   TRAN %3d" % (m[0][i], m[1][i]))
    if (0):
        for i in range(CELLS):
            if (min(SIJ_ARRAY[i,:])<0.0):
                print("CELL %d\n" % i, SIJ_ARRAY[i,:])
    # --- end of Simulate
    # print("SIJ_ARRAY ", SIJ_ARRAY[0,:])
    # sys.exit()
   
   

    
def SimulateOL(INI, MOL):
    """
    Simulation in case of generic line overlap.
    """
    global WIDTH, CLOUD_AREA, CLOUD_VOLUME, GL, TRANSITIONS, CELLS, RES
    # print("SimulateOL")
    tmp_1  =  C_LIGHT*C_LIGHT/(8.0*pi)
    Tbg    =  INI['Tbg']
    SUM_COOL, LEVL_COOL, HF = [], [], []
    # Upload NU and NBNB arrays
    for itran in range(TRANSITIONS):
        u, l   =  MOL.T2L(itran)        
        gg     =  MOL.GG[itran]
        coef   =  (C_LIGHT*C_LIGHT/(8.0*pi)) * MOL.A[itran] / (MOL.F[itran]**2.0)
        tmp    =  coef * (NI_ARRAY[:,l]*gg-NI_ARRAY[:,u])
        tmp[nonzero(abs(tmp)<1.0e-25)] = 2.0e-25
        WRK2[:,itran] = tmp
    cl.enqueue_copy(queue, NBNB_buf, WRK2)   #  [CELLS, TRANSITIONS]
    for itran in range(TRANSITIONS):
        u, l          =  MOL.T2L(itran)
        WRK2[:,itran] =  NI_ARRAY[:,u]
    cl.enqueue_copy(queue, NU_buf, WRK2)     #  [CELLS, TRANSITIONS]
    # Clear ARES
    kernel_clear(queue, [GLOBAL,], [LOCAL,], ARES_buf)
    if (WITH_CRT):
        kernel_sim(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, VOL_buf, Aul_buf, Blu_buf, GAU_buf, LIM_buf,
        GN_buf, WEI_buf, STEP_buf, APL_buf, IP_buf, BG_buf, NU_buf, NBNB_buf, ARES_buf, SINGLE_buf, NTRUE_buf,
        CRT_TAU_buf, CRT_EMI_buf)
    else:
        kernel_sim(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, VOL_buf, Aul_buf, Blu_buf, GAU_buf, LIM_buf,
        GN_buf, WEI_buf, STEP_buf, APL_buf, IP_buf, BG_buf, NU_buf, NBNB_buf, ARES_buf, SINGLE_buf, NTRUE_buf)
    for iband in range(OLBAND.Bands()):
        print("Simulate overlapping lines, band %d / %d" % (iband, OLBAND.Bands()))
        ncmp =  OLBAND.Components(iband)
        nchn =  OLBAND.Channels(iband) + CHANNELS
        # print("#### HOST NCMP %d  NCHN %d" % (ncmp, nchn))
        # push new vectors OLTRAN, OLOFF
        f0   =  MOL.F[OLBAND.GetTransition(iband,0)]
        for icmp in range(ncmp):
            OLTRAN[icmp]  =  OLBAND.GetTransition(iband, icmp)
            f             =  MOL.F[OLTRAN[icmp]]
            OLOFF[icmp]   =  0.5*(nchn-1.0)-0.5*(CHANNELS-1.0)-(f-f0)*(C_LIGHT/f0)*1.0e-5/WIDTH
        # TRAN, OFF  to device
        cl.enqueue_copy(queue, TRAN_buf, OLTRAN)
        cl.enqueue_copy(queue, OFF_buf,  OLOFF)
        if (WITH_CRT):
            kernel_ol(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, VOL_buf, Aul_buf, Blu_buf, GAU_buf,
            LIM_buf, GN_buf, WEI_buf, STEP_buf, APL_buf, IP_buf, BG_buf, NU_buf, NBNB_buf,
            ARES_buf, ncmp, nchn, TRAN_buf, OFF_buf, NTRUE_buf, CRT_TAU_buf, CRT_EMI_buf)            
        else:
            kernel_ol(queue, [GLOBAL,], [LOCAL,], CLOUD_buf, VOL_buf, Aul_buf, Blu_buf, GAU_buf,
            LIM_buf, GN_buf, WEI_buf, STEP_buf, APL_buf, IP_buf, BG_buf, NU_buf, NBNB_buf,
            ARES_buf, ncmp, nchn, TRAN_buf, OFF_buf, NTRUE_buf)            
    # Add results of all work groups
    GLOBAL_SUM = IRound(CELLS, LOCAL)
    kernel_sum(queue, [GLOBAL_SUM,], [LOCAL,], ARES_buf, RES_buf, VOL_buf)
    cl.enqueue_copy(queue, RES, RES_buf)  # RES = RES[CELLS, TRANSITIONS].xy
    SIJ_ARRAY[:,:] =  (RES[:,:]['x'])
    ESC_ARRAY[:,:] =  (RES[:,:]['y'])
    # --- end of SimulateOL
   
   
    

def WriteTex(tran, filename):
    # Save Tex values to binary file
    global INI
    freq   =  MOL.F[tran]
    u, l   =  MOL.T2L(tran)
    gg     =  MOL.G[u]/MOL.G[l]
    fp     =  open(filename, "wb")
    asarray([CELLS,], int32).tofile(fp)    
    tex    =  BOLTZMANN * log((NI_ARRAY[:,l]*gg)/NI_ARRAY[:,u])
    m      =  nonzero(abs(tex)<1.0e-35)
    tex    =  PLANCK * freq / tex
    tex[m] =  0.0
    asarray(tex, float32).tofile(fp)
    if (pickle):
        pickle.dump(INI,fp)
    fp.close()
    for i in range(3):
        print("       Tex(%02d-%02d) =  %.3e  cell %d" % (u, l, tex[i], i))
    print("       Average Tex(%d->%d) = %.3f" % (u, l, mean(tex)))

    
    
    
def WriteSpectra(tran, prefix, fwhm=-1.0):
    # Use separate kernel to calculate spectra, one work group per ray
    global INI
    tmp_1         =  C_LIGHT*C_LIGHT/(8.0*pi)
    upper, lower  =  MOL.T2L(tran)
    A_Aul         =  MOL.A[tran]
    freq          =  MOL.F[tran]
    gg            =  MOL.G[upper] / MOL.G[lower]
    A_gauss_norm  =  (C_LIGHT/(1.0e5*WIDTH*freq))     # GRID_LENGTH **NOT** multiplied in
    channels      =  CHANNELS
    if (HFS): channels = BAND[tran].Channels()        # CHANNELS is the allocation, channels for current transition
    # this time BG is BG intensity (not photon number)
    int2temp      =  C_LIGHT*C_LIGHT/(2.0*BOLTZMANN*freq*freq)
    print("       WriteSpectra, tran=%02d=%02d-%02d, F = %.3e, I2T=%.3e, fwhm=%.1f" % (tran, upper, lower, freq, int2temp, fwhm))
    BG            =  int2temp * Planck(freq, INI['Tbg'])   
    GLOBAL_SPE    =  IRound(NRAY_SPE, 64)
    STAU_buf      =  cl.Buffer(context, mf.READ_WRITE,  4*NRAY_SPE*CHANNELS)
    NTRUE         =  zeros((NRAY_SPE, channels), float32) # HFS => CHANNELS >= channels
    emissivity    =  (PLANCK/(4.0*pi))*freq*A_Aul*int2temp    
    # need to push NI to device
    nu            =  NI_ARRAY[:, upper] 
    nl            =  NI_ARRAY[:, lower] 
    tmp           =  (tmp_1 * A_Aul * (nl*gg-nu)) / (freq*freq) 
    tmp[nonzero(abs(tmp)<1.0e-25)] = 2.0e-25
    tmp[nonzero(tmp<(-1.0e-3))]    = -1.0e-3
    WRK[:]['x'] = nu     #  NI(upper)
    WRK[:]['y'] = tmp    #  NB_NB(cell, tran)
    cl.enqueue_copy(queue, NI_buf, WRK)    
    # one kernel call does all the spectra for the current transition
    if (HFS):
        if (WITH_CRT):
            kernel_spe(queue, [GLOBAL_SPE,], [LOCAL,], CLOUD_buf, GAU_buf, A_gauss_norm, NI_buf, BG, 
            emissivity, IP_buf, STEP_buf, NTRUE_SPE_buf, STAU_buf, np.int32(tran), np.int32(channels),
            CRT_TAU_buf, CRT_EMI_buf)
        else:
            kernel_spe(queue, [GLOBAL_SPE,], [LOCAL,], CLOUD_buf, GAU_buf, A_gauss_norm, NI_buf, BG, 
            emissivity, IP_buf, STEP_buf, NTRUE_SPE_buf, STAU_buf, np.int32(tran), np.int32(channels))
    else:  # including OVERLAP ... spectra for individual transitions
        if (WITH_CRT):
            kernel_spe(queue, [GLOBAL_SPE,], [LOCAL,], CLOUD_buf, GAU_buf, A_gauss_norm, NI_buf, BG, 
            emissivity, IP_buf, STEP_buf, NTRUE_SPE_buf, STAU_buf, RHO_buf, COLDEN_buf,
            tran, CRT_TAU_buf, CRT_EMI_buf)
        else:
            kernel_spe(queue, [GLOBAL_SPE,], [LOCAL,], CLOUD_buf, GAU_buf, A_gauss_norm, NI_buf, BG, 
            emissivity, IP_buf, STEP_buf, NTRUE_SPE_buf, STAU_buf, RHO_buf, COLDEN_buf, -1)
    # extract spectra and write to file
    cl.enqueue_copy(queue, NTRUE, NTRUE_SPE_buf)      #  buffer [NRAY_SPE, CHANNELS] -> NTRUE[NRAY_SPE, channels]
    #
    # dust emission and absorption -- CRT_TAU_buf and CRT_EMI_buf already contain data for all transitions
    filename = "%s.spe" % prefix
    fp       = open(filename, "wb")
    # File format:  NRAY_SPE, CHANNELS, V0, DV   { T[CHANNELS] }
    asarray([NRAY_SPE, channels], int32).tofile(fp)
    asarray([-0.5*(channels-1.0)*WIDTH, WIDTH], float32).tofile(fp)    #  [ v0, dv ]
    # NTRUE.shape = (NRAY_SPE, channels)
    asarray(NTRUE, float32).tofile(fp)  # in HFS case, channels<=CHANNELS
    if (pickle):
        pickle.dump(INI,fp)
    fp.close()
    if (0):
        print(" ==== TRAN %d  = %d - %d  =>  NTRUE = %.3e K  ....  B(T)*I2T = %.3e" % 
        (tran, upper, lower, NTRUE[0,2], Planck(freq, 10.0)*int2temp))
    if (fwhm>0):     # calculate spectrum convolved with INI['fwhm'] arcsec, towards the centre only!
        fwhm_gl =  fwhm / INI['angle']             # fwhm [GL]
        K       = -4.0*log(2.0)/fwhm_gl**2.0       # beam weighting, K to be multiplied with r^2, [r]=GL
        dd      =  0.5*(IP[1]-IP[0])               # half of the annulus width, is constant!
        # centre LOS
        W       =  exp(K*0.0) * dd**2.0            # weight = beam * area,  area ~ (0.5*IP[1])**2  [GL^2]
        S       =  NTRUE[0, :] * W                 # S will be the beam-averaged spectrum
        # off-centre samples
        for i in range(1, NRAY_SPE):
            w   =  exp(K*IP[i]**2.0) * ((IP[i]+dd)**2.0-(IP[i]-dd)**2.0)   # weight = beam * area
            W  +=  w
            S  +=  NTRUE[i,:] * w
        S /= W
        fp = open(filename.replace('.spe', '_con.spe'), 'wb')
        asarray([1, channels], int32).tofile(fp)
        asarray([-0.5*(channels-1.0)*WIDTH, WIDTH], float32).tofile(fp)    #  [ v0, dv ]
        asarray(S[0:channels], float32).tofile(fp)
        if (pickle):
            pickle.dump(INI,fp)
        fp.close()
    # write another file:   IP, N, Nmol, taumax
    if (not(HFS)):  # colden was not (yet) added as option in HFS spectrum routine
        tmp3      =  zeros(NRAY_SPE, cl.cltypes.float4)
        cl.enqueue_copy(queue, tmp3, COLDEN_buf)       # COLDEN_buf ~ [NRAY_SPE] float4 !!!
        filename  =  "%s.colden" % prefix
        fp        =  open(filename, "wb")
        asarray(NRAY_SPE, float32).tofile(fp)    
        asarray(IP[:]*GL, float32).tofile(fp)          # offsets [cm]
        # print(GL)
        # print(tmp3[:]['x'])
        asarray(tmp3[:]['x']*GL, float32).tofile(fp)   # total column densities
        asarray(tmp3[:]['y']*GL, float32).tofile(fp)   # column density of the molecule
        asarray(tmp3[:]['z'],    float32).tofile(fp)   # maximum optical depth
        if (pickle):
            pickle.dump(INI,fp)
        fp.close()

        


def WriteSpectraOL(iband, prefix, fwhm=-1.0):
    # Use separate kernel to calculate spectra for bands of overlapping lines
    ncmp          =  OLBAND.Components(iband)
    nchn          =  OLBAND.Channels(iband) + CHANNELS
    NTRUE         =  zeros((NRAY_SPE, nchn), float32)
    TRAN          =  zeros(ncmp, int32)
    OFF           =  zeros(ncmp, float32)
    f0            =  MOL.F[OLBAND.GetTransition(iband,0)]
    print("       WriteSpectraOL, band %d, fwhm=%.1f" % (iband, fwhm))
    # Find out transitions and their relative offsets
    for icmp in range(ncmp):
        TRAN[icmp]  =  OLBAND.GetTransition(iband, icmp)
        f           =  MOL.F[TRAN[icmp]]
        OFF[icmp]   =  0.5*(nchn-1.0)-0.5*(CHANNELS-1.0)-(f-f0)*(C_LIGHT/f0)*1.0e-5/WIDTH # channels
        # print("   icmp %d/%d:  TRAN %d OFF %4.1f" % (icmp, ncmp, TRAN[icmp], OFF[icmp]))
    cl.enqueue_copy(queue, TRAN_buf, TRAN)
    cl.enqueue_copy(queue, OFF_buf,  OFF)
    # one kernel call does all the spectra for the current transition; one work item per ray !!
    if (WITH_CRT):
        kernel_spe_ol(queue, [GLOBAL,], [LOCAL,], f0, CLOUD_buf, GAU_buf, LIM_buf, GN_buf, NU_buf, 
        Aul_buf, NBNB_buf, STEP_buf, IP_buf, BG_buf, NTRUE_buf, STAU_buf, ncmp, nchn, TRAN_buf, OFF_buf,
        CRT_TAU_buf, CRT_EMI_buf)
    else:
        kernel_spe_ol(queue, [GLOBAL,], [LOCAL,], f0, CLOUD_buf, GAU_buf, LIM_buf, GN_buf, NU_buf, 
        Aul_buf, NBNB_buf, STEP_buf, IP_buf, BG_buf, NTRUE_buf, STAU_buf, ncmp, nchn, TRAN_buf, OFF_buf)
    # extract spectra and write to file
    cl.enqueue_copy(queue, NTRUE, NTRUE_buf)      #  buffers [NRAY_SPE, nchn]
    # Write the file -------------------------
    filename = "%s.spe" % prefix
    fp       = open(filename, "wb")
    asarray([NRAY_SPE, nchn], int32).tofile(fp)
    asarray([-0.5*(nchn-1.0)*WIDTH, WIDTH], float32).tofile(fp)    #  [ v0, dv ]
    asarray(NTRUE[:,0:nchn], float32).tofile(fp)
    if (pickle):
        pickle.dump(INI,fp)
    fp.close()
    # ----------------------------------------
    if (fwhm>0):     # calculate spectrum convolved with INI['fwhm'] arcsec, towards the centre only!
        fwhm_gl =  fwhm / INI['angle']             # fwhm [GL]
        K       = -4.0*log(2.0)/fwhm_gl**2.0       # beam weighting, K to be multiplied with r^2, [r]=GL
        dd      =  0.5*(IP[1]-IP[0])               # half of the annulus width, is constant!
        # centre LOS
        W       =  exp(K*0.0) * dd**2.0            # weight = beam * area,  area ~ (0.5*IP[1])**2  [GL^2]
        S       =  NTRUE[0, :] * W                 # S will be the beam-averaged spectrum
        # off-centre samples
        for i in range(1, NRAY_SPE):
            w   =  exp(K*IP[i]**2.0) * ((IP[i]+dd)**2.0-(IP[i]-dd)**2.0)   # weight = beam * area
            W  +=  w
            S  +=  NTRUE[i,:] * w
        S /= W
        fp = open(filename.replace('.spe', '_con.spe'), 'wb')
        asarray([1, nchn], int32).tofile(fp)
        asarray([-0.5*(nchn-1.0)*WIDTH, WIDTH], float32).tofile(fp)    #  [ v0, dv ]
        asarray(S, float32).tofile(fp)
        if (pickle):
            pickle.dump(INI,fp)
        fp.close()

    
    
def Solve(INI, MOL):
    #  3d  had ESCAPE/VOLUME,  in 1d VOLUME does not appear !!
    #  instead we have SIJ scaled with PATH_CORRECTION
    global LEVELS, TKIN, RHO, ABU, ESC_ARRAY, PATH_CORRECTION, SIJ_ARRAY, NI_ARRAY
    CHECK     = min([INI['uppermost']+1, LEVELS])  # check this many lowest energylevels
    cab       = ones(10, float32)                  # scalar abundances of different collision partners
    for i in range(PARTNERS):
        cab[i] = MOL.CABU[i]                       # default values == 1.0
    # possible abundance file for abundances of all collisional partners
    CABFP = None
    if (len(INI['cabfile'])>0): # we have a file with abundances for each collisional partner
        CABFP = open(INI['cabfile'], 'rb')
        tmp   = fromfile(CABFP, int32, 4)
        if ((tmp[0]!=X)|(tmp[1]!=Y)|(tmp[2]!=Z)|(tmp[3]!=PARTNERS)):
            print("*** ERROR: CABFILE has dimensions %d x %d x %d, for %d partners" % (tmp[0], tmp[1], tmp[2], tmp[3]))
            sys.exit()            
    real      =  float64
    MATRIX    =  zeros((LEVELS, LEVELS), real)
    VECTOR    =  zeros(LEVELS, real)
    COMATRIX  =  []
    ave_max_change, global_max_change = 0.0, 0.0    
    if (INI['constant_tkin']): # Tkin same for all cells => precalculate collisional part
        print("========== Tkin assumed to be constant !! ==========")
        constant_tkin = True
    else:
        constant_tkin = False
    if (constant_tkin):
        if (CABFP):
            print("Cannot have variable CAB if Tkin is assumed to be constant")
        COMATRIX = zeros((LEVELS, LEVELS), real)
        tkin     = TKIN[1]
        for iii in range(LEVELS):
            for jjj in range(LEVELS):
                if (iii==jjj):
                    COMATRIX[iii,jjj] = 0.0
                else:
                    if (PARTNERS==1):
                        gamma = MOL.C(iii,jjj,tkin,0)  # rate iii -> jjj
                    else:
                        gamma = 0.0
                        for ip in range(PARTNERS):
                            gamma += cab[ip] * MOL.C(iii, jjj, tkin, ip)
                    COMATRIX[jjj, iii] = gamma                    
    for icell in range(CELLS):
        if (icell%10==11):
            print("          solve   %7d / %7d  .... %3.0f%%" % (icell, CELLS, 100.0*icell/float(CELLS)))
        tkin, rho, chi  =  TKIN[icell], RHO[icell], ABU[icell]
        if (rho<0.1):     continue
        if (chi<1.0e-20): continue
        if (constant_tkin):
            MATRIX[:,:] = COMATRIX[:,:] * rho 
        else:
            if (CABFP):
                cab = fromfile(CABFP, float32, PARTNERS)   # abundances for current cell, cab[PARTNERS]
            for iii in range(LEVELS):
                for jjj in range(LEVELS):
                    if (iii==jjj):
                        MATRIX[iii,jjj] = 0.0
                    else:
                        if (PARTNERS==1):
                            gamma = MOL.C(iii,jjj,tkin,0)  # rate iii -> jjj
                        else:
                            gamma = 0.0
                            for ip in range(PARTNERS):
                                gamma += cab[ip] * MOL.C(iii, jjj, tkin, ip)
                        MATRIX[jjj, iii] = gamma*rho
        if (len(ESC_ARRAY)>0):
            for t in range(TRANSITIONS):
                u, l          =  MOL.T2L(t)
                MATRIX[l,u]  +=  ESC_ARRAY[icell, t] / (NI_ARRAY[icell,u]) # 1D => no 1/VOLUME !!
        else:
            for t in range(TRANSITIONS):
                u,l           =  MOL.T2L(t)
                MATRIX[l,u]  +=  MOL.A[t]                
        if (len(PATH_CORRECTION)>0):
            for t in range(TRANSITIONS):
                u, l           =  MOL.T2L(t)   # radiative transitions only !!
                MATRIX[u, l]  +=  SIJ_ARRAY[icell, t]             * PATH_CORRECTION[icell]
                MATRIX[l, u]  +=  SIJ_ARRAY[icell, t] / MOL.GG[t] * PATH_CORRECTION[icell]
        else:
            for t in range(TRANSITIONS):
                u, l           =  MOL.T2L(t)   # radiative transitions only !!
                MATRIX[u, l]  +=  SIJ_ARRAY[icell, t]               #  u <-- l
                MATRIX[l, u]  +=  SIJ_ARRAY[icell, t] / MOL.GG[t]   #  l <-- u
        for u in range(LEVELS-1): # diagonal = -sum of the column
            tmp            = sum(MATRIX[:,u])    # MATRIX[i,i] was still == 0
            MATRIX[u,u]    = -tmp            
        MATRIX[LEVELS-1, :]   =  -MATRIX[0,0]    # replace last equation = last row
        VECTOR[:]             =   0.0
        VECTOR[LEVELS-1]      =  -(rho*chi) * MATRIX[0,0]  # ???
        if (0):
            if (icell==10):
                for j in range(LEVELS):
                    for i in range(LEVELS):
                        sys.stdout.write(' %10.3e' % MATRIX[j,i])
                    sys.stdout.write('\n')
                for j in range(TRANSITIONS):
                    sys.stdout.write(' %10.3e' % SIJ_ARRAY[icell, j]) 
                sys.stdout.write('\n')
                sys.exit()
        VECTOR  =  np.linalg.solve(MATRIX, VECTOR)        
        VECTOR *=  rho*chi / sum(VECTOR)        
        VECTOR  =  clip(VECTOR, NI_LIMIT, 1e99)
        if (0):
            print("CO_ARRAY")
            for j in range(LEVELS):
                for i in range(LEVELS):
                    sys.stdout.write('%9.2e ' % (MATRIX[j,i]))
                sys.stdout.write('   %9.2e\n' % (VECTOR[j]))
            print('')
            print("SIJ")
            for j in range(TRANSITIONS):
                sys.stdout.write(' %10.2e' % SIJ_ARRAY[icell,j])
            sys.stdout.write('\n')
            print("ESC")
            for j in range(TRANSITIONS):
                sys.stdout.write(' %10.2e' % ESC_ARRAY[icell,j])
            sys.stdout.write('\n')
            print("VECTOR")
            for j in range(LEVELS):
                sys.stdout.write(' %10.2e' % VECTOR[j])
            sys.stdout.write('\n')
            sys.exit()            
        max_relative_change =  max((NI_ARRAY[icell, 0:CHECK]-VECTOR[0:CHECK])/(NI_ARRAY[icell, 0:CHECK]))        
        NI_ARRAY[icell,:]   =  VECTOR        
        ave_max_change     +=  max_relative_change
        global_max_change   =  max([global_max_change, max_relative_change])    
    # <--- for icell
    ave_max_change /= CELLS
    print("       ave %10.3e    max %10.3e" % (ave_max_change, global_max_change))
    return ave_max_change
    



def SolveCL(INI, MOL):
    """
    Solve equilibrium equations on the device. We do this is batches, perhaps 10000 cells
    at a time => could be up to GB of device memory. 
    """
    global NI_buf, PL_buf, CELLS, queue, kernel_solve, PL_buf, RES_buf, CELLS, LEVELS, TRANSITIONS
    global RHO, TKIN, ABU, PARTNERS, NTKIN, NCUL, SIJ_ARRAY, ESC_ARRAY
    global S_WRK_buf, S_RHO_buf, S_TKIN_buf, S_ABU_buf, S_SIJ_buf, S_ESC_buf, S_PC_buf, S_NI_buf, S_RES_buf
    global MOL_A_buf, MOL_UL_buf, MOL_E_buf, MOL_G_buf, MOL_TKIN_buf, MOL_CUL_buf, MOL_C_buf, MOL_CABU_buf
    # new buffer for matrices and the right side of the equilibriumm equations
    CHECK              =  min([INI['uppermost']+1, LEVELS])  # check this many lowest energylevels
    GLOBAL_SOLVE       =  IRound(BATCH, LOCAL)
    tmp                =  zeros((BATCH, 2, TRANSITIONS), float32)
    res                =  zeros((BATCH, LEVELS), float32)
    ave_max_change     =  0.0
    global_max_change  =  0.0
    for ibatch in range(1+CELLS//BATCH):
        a     = ibatch*BATCH
        b     = min([a+BATCH, CELLS])
        batch = b-a
        if (batch<1): break
        # print(" SolveCL, batch %5d,  [%7d, %7d[,  %4d cells, BATCH %4d" % (ibatch, a, b, batch, BATCH))
        # copy RHO, TKIN, ABU
        cl.enqueue_copy(queue, S_RHO_buf,  RHO[a:b].copy())  # without copy() "ndarray is not contiguous"
        cl.enqueue_copy(queue, S_TKIN_buf, TKIN[a:b].copy())
        cl.enqueue_copy(queue, S_ABU_buf,  ABU[a:b].copy())
        cl.enqueue_copy(queue, S_NI_buf,   NI_ARRAY[a:b,:].copy())   # PL[CELLS] ~ NI[BATCH, LEVELS]
        cl.enqueue_copy(queue, S_SIJ_buf,  SIJ_ARRAY[a:b,:].copy())
        cl.enqueue_copy(queue, S_ESC_buf,  ESC_ARRAY[a:b,:].copy())
        cl.enqueue_copy(queue, S_PC_buf,   PATH_CORRECTION[a:b].copy())
        # solve
        kernel_solve(queue, [GLOBAL_SOLVE,], [LOCAL,], batch, 
        MOL_A_buf, MOL_UL_buf,  MOL_E_buf, MOL_G_buf, PARTNERS, NTKIN, NCUL,   
        MOL_TKIN_buf, MOL_CUL_buf,  MOL_C_buf, MOL_CABU_buf,
        S_RHO_buf, S_TKIN_buf, S_ABU_buf,  S_NI_buf, S_SIJ_buf, S_ESC_buf,  S_RES_buf,  S_WRK_buf, S_PC_buf)
        cl.enqueue_copy(queue, res, S_RES_buf)
        # delta = for each cell, the maximum level populations change among levels 0:CHECK
        delta             =  np.max((res[0:batch,0:CHECK] - NI_ARRAY[a:b,0:CHECK]) / NI_ARRAY[a:b,0:CHECK], axis=1)
        global_max_change =  max([global_max_change, max(delta)])
        ave_max_change   +=  sum(delta)
        NI_ARRAY[a:b,:]   =  res[0:batch,:]                        
    ave_max_change /= CELLS
    print("       ave %10.3e    max %10.3e" % (ave_max_change, global_max_change))        
    if (0):
        print("NI ", NI_ARRAY[0,:])
        sys,exit()
    return ave_max_change


##########################################################################################
        
        
        
        
# SIMULATIONS
NITER = INI['iterations']
if (METHOD_X):
    Simulate_X(INI, MOL, NITER)
    # Download the final level populations
    cl.enqueue_copy(queue, NI_ARRAY, X_NI_buf)
    if (0):
        print("NI ", NI_ARRAY[0,:])
        sys.exit()
        
else:    
    t0 = time.time()
    max_change, Tsim, Tsol, Tsav = 0.0, 0.0, 0.0, 0.0
    for ITER in range(NITER):
        sys.stdout.write("ITER %2d / %2d: " % (ITER+1, NITER))
        Seconds()
        if (OVERLAP):  SimulateOL(INI, MOL) 
        else:          Simulate(INI, MOL) 
        Tsim = Seconds() 
        if (1):
            SIJ_ARRAY = clip(SIJ_ARRAY, 0.0, 1.0)
        # max_change = SolveParallel()   #  64^3  0.18 seconds, 256^3  10.76 seconds
        if (INI['clsolve']):
            max_change =  SolveCL(INI, MOL)
        else:
            max_change =    Solve(INI, MOL)
        Tsol = Seconds() 
        if (1):
            NI_ARRAY = clip(NI_ARRAY, 1.0e-20, 1.0)
        if (0):
            print('*********************************************************************')
            for icell in range(CELLS):
                for itran in range(TRANSITIONS):  sys.stdout.write('%10.3e ' % SIJ_ARRAY[icell, itran])
                sys.stdout.write('\n')
            print('*********************************************************************')
            for icell in range(CELLS):
                for ilev in range(LEVELS):  sys.stdout.write('%10.3e ' % NI_ARRAY[icell, ilev])
                sys.stdout.write('\n')
            print('*********************************************************************')
            #sys.exit()    
        if (((ITER%11==10)|(ITER==(NITER-1)))&(len(INI['save'])>0)): # save level populations
            fp = open(INI['save'], "wb")
            asarray([CELLS, LEVELS], int32).tofile(fp)
            asarray(NI_ARRAY, float32).tofile(fp)
            fp.close()
            Tsav = Seconds()
            print("       simulation  %7.3f    solve %7.3f   save %7.3f  seconds" % (Tsim, Tsol, Tsav))
        if (max_change<INI['stop']): break
    t0 = time.time()-t0
    print("ITERATIONS, SIMULATION + SOLUTION: %.4f seconds, %.3f seconds per iteration" % (t0, t0/NITER))

# EXCITATION TEMPERATURES
Seconds()
for i in range(len(INI['Tex'])//2):                # save Tex
    u, l =  INI['Tex'][2*i], INI['Tex'][2*i+1]     # upper and lower level
    t    = MOL.L2T(u, l)                           # transition
    if (t<0): continue
    filename  =  "%s.%s_%02d-%02d.tex" % (INI['prefix'], MOL.NAME, u, l)
    WriteTex(t, filename) 
# print("       TEX      %7.3f seconds" % Seconds())

      
   
# SPECTRA
# We use NRAY_SPE equidistant rays, not the original NRAY that were used in the simulation
# Because of device allocations, NRAY_SPE <= NRAY, recompute STEP[], APL[] is not used
# NO --- NRAY_SPE will be allowed to be larger  (STEP_buf and IP_buf allocated for the maximum of NRAY and NRAY_SPE)
IP     =  asarray(arange(NRAY_SPE)*1.0/(NRAY_SPE-1.0), float32)  # first at the centre, the last on the outer border
STEP   =  GetSteps1D(CELLS, RADIUS, NRAY_SPE, IP, [])  # DIRWEI and APL not touched, not needed for spectra
cl.enqueue_copy(queue, STEP_buf, STEP)
cl.enqueue_copy(queue, IP_buf,   IP)

if (0):
    # double check if we still have correct dust source function
    # originally   S =  (CRT_EMI[0,t]*RHO[0]*PLANCK*MOL.F[t]/(4.0*pi)) / (CRT_TAU[0,t]/PARSEC)
    # but CRT_TAU was scaled from tau/pc  to tau/gl  ==  *= GL/PARSEC
    # and CRT_EMI was scaled   *= RHO * (f*1e5*WIDTH)/C_LIGHT == photons per channel, not photons / Hz
    print("GL = %.4f pc" % (GL/PARSEC))
    for t in range(TRANSITIONS):
        u, l = MOL.T2L(t)
        NOM   =   CRT_EMI[0,t]*RHO[0]*PLANCK*MOL.F[t]/(4.0*pi)
        DEN   =   CRT_TAU[0,t]/PARSEC
        # undo CRT_EMI scalings
        NOM  /=  MOL.F[t] * 1e5*WIDTH/C_LIGHT  
        NOM  /=  RHO[0]
        # undo CRT_TAU scaling
        DEN  /=  GL/PARSEC
        ####
        S     =  NOM/DEN
        print("   ***** TRANSITIONS %d  = %d - %d   =>   S = %.3e" % (t, u, l, S))
    

if (WITH_CRT>0):
    #  CRT_EMIT was   photons/s/channel/cm3,  here converted to emissivity
    #  GN is included == conversion from 1/channel to 1/Hz
    for t in range(TRANSITIONS):
        CRT_EMI[:,t]  *=   H_K * ((C_LIGHT/MOL.F[t])**2.0) * C_LIGHT / (1.0e5*WIDTH*8.0*pi)
    cl.enqueue_copy(queue, CRT_EMI_buf, CRT_EMI)

    
    
    
if (OVERLAP):
    # Upload NU and NBNB
    for itran in range(TRANSITIONS):
        u, l   =  MOL.T2L(itran)        
        gg     =  MOL.GG[itran]
        coef   =  (C_LIGHT*C_LIGHT/(8.0*pi)) * MOL.A[itran] / (MOL.F[itran]**2.0)
        tmp    =  coef * (NI_ARRAY[:,l]*gg-NI_ARRAY[:,u])
        tmp[nonzero(abs(tmp)<1.0e-25)] = 2.0e-25
        WRK2[:,itran] = tmp
    cl.enqueue_copy(queue, NBNB_buf, WRK2)   #  [CELLS, TRANSITIONS]
    ###
    for itran in range(TRANSITIONS):
        u, l          =  MOL.T2L(itran)
        WRK2[:,itran] =  NI_ARRAY[:,u]       #  WRK2[CELLS, TRANSITIONS]
    cl.enqueue_copy(queue, NU_buf, WRK2)     #  WRK2[CELLS, TRANSITIONS]
    # write the bands of overlapping lines
    tmp = zeros(TRANSITIONS, float32)
    for i in range(TRANSITIONS):
        freq    =  MOL.F[i]
        tmp[i]  =  Planck(freq, INI['Tbg'])  * (C_LIGHT**2.0)/(2.0*BOLTZMANN*freq*freq)  # B()*I2T
    cl.enqueue_copy(queue, BG_buf, tmp)
    for iband in range(OLBAND.Bands()):
        # check if one has specified fwhm... for the transitions in INI['spectra'], if that 
        # transition is now part of the band
        fwhm = -1.0
        if (len(INI['fwhm'])>0):
            for i in range(len(INI['spectra'])//2):  # loop over transitions specified with "spectra"
                u, l  =  INI['spectra'][2*i], INI['spectra'][2*i+1]
                t     =  MOL.L2T(u, l)
                if (t in OLBAND.TRAN[iband]):        # if that transition is part of the current band....
                    fwhm = INI['fwhm'][min([i, len(INI['fwhm'])-1])]  # see if fwhm was specified for t
        prefix  =  "%s.band%d" % (INI['prefix'], iband)
        WriteSpectraOL(iband, prefix, fwhm)

        
# Normal spectra, including separate transitions in the OVERLAP case
for i in range(len(INI['spectra'])//2):
    u, l  =  INI['spectra'][2*i], INI['spectra'][2*i+1]
    t     =  MOL.L2T(u, l)
    if (t<0): 
        print("*** Error: requested transitions does not exist: %d -> %d !!" % (u,l))
        continue
    prefix = '%s.%s_%02d-%02d' % (INI['prefix'], MOL.NAME, u, l)
    fwhm   = -1.0
    # if (len(INI['fwhm'])>0):  fwhm = INI['fwhm'][min([i, len(INI['fwhm'])-1])]
    WriteSpectra(t, prefix, fwhm)
        
# print("     SPECTRA  %7.3f seconds" % Seconds())
   

