#!/usr/bin/env python

"""
LOC.py copied to LOC_OT.py  2020-06-15
because Paths and Update started to have several additional parameters not needed 
in case of regular cartesian grid. LOC.py was already using separate kernel_update_py_OT.c for OT.
And OT uses work group per ray while in LOC.py it was still one work item per ray.
"""


import os, sys
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)   
from   LOC_aux import *
t000 = time.time()


if (0):
    nside = 12
    tmp = zeros((12*nside*nside, 4), float32)
    for idir in range(12*nside*nside):
        theta, phi     =  Pixel2AnglesRing(nside, idir)
        tmp[idir, 0:2] = [theta, phi]
        theta, phi     =  healpy.pix2ang(nside, idir)
        tmp[idir, 2:4] = [theta, phi]
    clf()
    plot(tmp[:,0], tmp[:,1], 'r+')
    plot(tmp[:,2], tmp[:,3], 'bx')
    show()
    sys.exit()
    
"""
x = cl.cltypes.make_float2() 
x['x'], x['y']
x = np.zeros((3,3), cl.cltypes.float2) 
x[0,0]['x']
"""

if (len(sys.argv)<2):
    print("Usage:  LOC.py  <ini-file>")
    sys.exit()
    
INI         =  ReadIni(sys.argv[1])

MOL         =  ReadMolecule(INI['molecule'])
HFS         =  len(INI['hfsfile'])>0               # HFS with LTE components
LEVELS      =  INI['levels']                       # energy levels in the species
TRANSITIONS =  MOL.Transitions(LEVELS)             # how many transitions among LEVELS levels
CHANNELS    =  INI['channels']
OCTREE      =  INI['octree']                       # 0 = regular Cartesian, 1 = octree, 2 = octree with ray splitting
WITH_ALI    =  (INI['WITH_ALI']>0)
DOUBLE_POS  =  ((OCTREE==4)|(OCTREE==40))          # 2020-08-01 this was the working combination (not very fast though)
MAX_NBUF    =  40
MAX_NBUF    =  INI['maxbuf']
PLWEIGHT    =  INI['plweight']
WITH_HALF   =  INI['WITH_HALF']>0

REAL, REAL3 = None, None
if (DOUBLE_POS>0):
    REAL, REAL3 = np.float64, cl.cltypes.double3
else:
    REAL, REAL3 = np.float32, cl.cltypes.float3
FLOAT3 = cl.cltypes.float3
print("LOC_OT.py, OCTREE=%d, WITH_ALI=%d, DOUBLE_POS=%d, WITH_HALF=%d, PLEIGHT=%d, MAX_NBUF=%d" % \
(OCTREE, WITH_ALI, DOUBLE_POS, WITH_HALF, PLWEIGHT, MAX_NBUF))



FIX_PATHS   =  0
if (OCTREE>0):
    FIX_PATHS = 0     # does not work for octrees, especially OT4
# FIX_PATHS does not work for OCTREE>0 cases (yet)
#   Should work for OCTREE=0 => dispersion on PL is decreased. While this effect is very small
#   and dispersion is already in the 5th significant digit and the mean PL remains the same to
#   the fifth significant digit, in test case the minimum Tex increases from 2.69K to 2.81K !!
# 2020-01-10 ... FIX_PATHS *not* used


if (OCTREE):
    # RHO[CELLS], TKIN[CELLS], CLOUD[CELLS]{vx,vy,vz,sigma}, ABU[CELLS], OFF[OTL], OTL = octree levels
    RHO, TKIN, CLOUD, ABU, CELLS, OTL, LCELLS, OFF,  NX, NY, NZ =  ReadCloudOT(INI, MOL)
else:
    print("*** Cartesian grid cloud ***")
    RHO, TKIN, CLOUD, ABU, NX, NY, NZ  =  ReadCloud3D(INI, MOL)
    CELLS  =  NX*NY*NZ
    OTL    =  -1

ONESHOT     =  INI['oneshot']    # no loop over ray offsets, kernel lops over all the rays
NSIDE       =  int(INI['nside'])
NDIR        =  max([6, 12*NSIDE*NSIDE])           # NSIDE=0 => 6 directions, cardinal directions only
WIDTH       =  INI['bandwidth']/INI['channels']   # channel width [km/s], even for HFS calculations
AREA        =  2.0*(NX*NY+NY*NZ+NZ*NX) 
# NRAY should be enough for the largest side of the cloud
if ((NX<=NY)&(NX<=NZ)):  NRAY =  ((NY+1)//2) * ((NZ+1)//2) 
if ((NY<=NX)&(NY<=NZ)):  NRAY =  ((NX+1)//2) * ((NZ+1)//2) 
if ((NZ<=NX)&(NZ<=NY)):  NRAY =  ((NX+1)//2) * ((NY+1)//2) 
if (ONESHOT):
    if ((NX<=NY)&(NX<=NZ)):  NRAY =  NY * NZ 
    if ((NY<=NX)&(NY<=NZ)):  NRAY =  NX * NZ 
    if ((NZ<=NX)&(NZ<=NY)):  NRAY =  NX * NY 
    if (not(OCTREE in [0, 4, 40])):
        print("Option ONESHOT applies to method OCTREE=0, OCTREE=4 and OCTREE=40 only!!")
        sys.exit()
    
VOLUME      =  1.0/(NX*NY*NZ)         #   Vcell / Vcloud ... for octree it is volume of root-grid cell
GL          =  INI['angle'] * ARCSEC_TO_RADIAN * INI['distance'] * PARSEC
APL         =  0.0
# APL_WEIGHT  =  1.0

print("================================================================================")
if (OCTREE>0):
    print("CLOUD %s, ROOT TGRID %d %d %d, OTL %d, LCELLS " % (INI['cloud'], NX, NY, NZ, OTL), LCELLS)
m = nonzero(RHO>0.0)  # only leaf nodes
print("    CELLS    %d" % CELLS)
if (WITH_HALF==0):
    print("    density  %10.3e  %10.3e" % (min(RHO[m]), max(RHO[m])))
    print("    Tkin     %10.3e  %10.3e" % (min(TKIN[m]), max(TKIN[m])))
    print("    Sigma    %10.3e  %10.3e" % (min(CLOUD['w'][m]), max(CLOUD['w'][m])))
    print("    vx       %10.3e  %10.3e" % (min(CLOUD['x'][m]), max(CLOUD['x'][m])))
    print("    vy       %10.3e  %10.3e" % (min(CLOUD['y'][m]), max(CLOUD['y'][m])))
    print("    vz       %10.3e  %10.3e" % (min(CLOUD['z'][m]), max(CLOUD['z'][m])))
    print("    chi      %10.3e  %10.3e" % (min(ABU[m]),  max(ABU[m])))
    if ((min(TKIN[m])<0.0)|(min(ABU[m])<0.0)|(min(CLOUD['w'][m])<0.0)):
        print("*** Check the cloud parameters: Tkin, abundance, sigma must all be non-negative")
        sys.exit()
else:
    print("    density  %10.3e  %10.3e" % (min(RHO[m]), max(RHO[m])))
    print("    Tkin     %10.3e  %10.3e" % (min(TKIN[m]), max(TKIN[m])))
    print("    Sigma    %10.3e  %10.3e" % (min(CLOUD[:,3][m]), max(CLOUD[:,3][m])))
    print("    vx       %10.3e  %10.3e" % (min(CLOUD[:,0][m]), max(CLOUD[:,0][m])))
    print("    vy       %10.3e  %10.3e" % (min(CLOUD[:,1][m]), max(CLOUD[:,1][m])))
    print("    vz       %10.3e  %10.3e" % (min(CLOUD[:,2][m]), max(CLOUD[:,2][m])))
    print("    chi      %10.3e  %10.3e" % (min(ABU[m]),  max(ABU[m])))
    if ((min(TKIN[m])<0.0)|(min(ABU[m])<0.0)|(min(CLOUD[:,3][m])<0.0)):
        print("*** Check the cloud parameters: Tkin, abundance, sigma must all be non-negative")
        sys.exit()
print("GL %.3e, NSIDE %d, NDIR %d, NRAY %d" % (GL, NSIDE, NDIR, NRAY))
print("================================================================================")


if (0):
    for i in [ 837288, 837233 ]:
        print()
        print("CELL %d" % i)
        print("    density  %10.4e" % RHO[i])
        print("    Tkin     %10.4e" % TKIN[i])
        print("    Sigma    %10.4e" % CLOUD['w'][i])
        print("    vx       %10.4e" % CLOUD['x'][i])
        print("    vy       %10.4e" % CLOUD['y'][i])
        print("    vz       %10.4e" % CLOUD['z'][i])
        print("    chi      %10.4e" % ABU[i])
        # peak optical depth for sigma(V) = 1 km/s line
        s          =  250.0/64.0  * PARSEC * 0.25
        GN         =  C_LIGHT/(1.0e5*WIDTH*MOL.F[0])
        taumax     =  (RHO[i]*ABU[i]  *   MOL.A[0]  *   C_LIGHT**2 / (8.0*pi*MOL.F[0]**2)) * s * GN        
        print("    taumax   %10.3e" % taumax)
        print("    GN = %10.3e" % GN)
        print()
    sys.exit()


LOWMEM      =  INI['lowmem']
COOLING     =  INI['cooling']
MAXCHN      =  INI['channels']

if (COOLING & HFS):
    print("*** Cooling not implemented for HFS => cooling will not be calculated!")
    COOLING =  0
if (HFS):
    BAND, MAXCHN, MAXCMP = ReadHFS(INI, MOL)     # CHANNELS becomes the maximum over all transitions
    print("HFS revised =>  CHANNELS %d,  MAXCMP = %d" % (CHANNELS, MAXCMP))
    HF      =  zeros(MAXCMP, cl.cltypes.float2)
    
    
print("TRANSITIONS %d, CELLS %d = %d x %d x %d" % (TRANSITIONS, CELLS, NX, NY, NZ))
SIJ_ARRAY, ESC_ARRAY = None, None
if (LOWMEM>1): #  NI_ARRAY, SIJ_ARRAY and (for ALI) ESC_ARRAY are mmap files
    SIJ_ARRAY = np.memmap('LOC_SIJ.mmap', dtype='float32', mode='w+', offset=0, shape=(CELLS, TRANSITIONS))
    if (WITH_ALI==0):  
        ESC_ARRAY =  zeros(1, float32)  # dummy array
    else:              
        ESC_ARRAY =  np.memmap('LOC_ESC.mmap', dtype='float32', mode='w+', offset=0, shape=(CELLS, TRANSITIONS))
else:  # SIJ_ARRAY and ESC_ARRAY normal in-memory arrays
    SIJ_ARRAY   =  zeros((CELLS, TRANSITIONS), float32)
    if (WITH_ALI==0):  ESC_ARRAY   =  zeros(1, float32)  # dummy array
    else:              ESC_ARRAY   =  zeros((CELLS, TRANSITIONS), float32)

WITH_CRT    =  INI['with_crt']
CRT_TAU     =  []
CRT_EMI     =  []
TMP         =  []
if (WITH_CRT):
    # Also in case of octree, CRT_TAU[CELLS] and CRT_EMI[CELLS] are simple contiguous vectors
    TMP     =  zeros(CELLS, float32)
    print(INI['crttau'])
    CRT_TAU =  ReadDustTau(INI['crttau'], GL, CELLS, TRANSITIONS)                # [CELLS, TRANSITIONS]
    CRT_EMI =  ReadDustEmission(INI['crtemit'], CELLS, TRANSITIONS, WIDTH, MOL)  # [CELLS, TRANSITIONS]
    # conversion from photons / s / channel / H    -->   photons / s / channel / cm3
    for t in range(TRANSITIONS):
        CRT_EMI[:,t] *=  RHO

    
# unlike for LOC1D.py which has GAU[TRANSITIONS, CELLS, CHANNELS], LOC.py still has
# GAU[GNO, CHANNELS] ... we probably should have this in global memory, not to restrict GNO.
GAUSTORE = '__global'
GNO      =  100      # number of precalculated Gaussians
G0, GX, GAU, LIM =  GaussianProfiles(INI['min_sigma'], INI['max_sigma'], GNO, CHANNELS, WIDTH)

if (INI['sdevice']==''):
    # We use ini['GPU'], INI['platforms'], INI['idevice'] to select the platform and device
    if (INI['GPU']):  LOCAL = 32
    else:             LOCAL =  1
    if (INI['LOCAL']>0): LOCAL = INI['LOCAL']
    platform, device, context, queue,  mf = InitCL(INI['GPU'], INI['platforms'], INI['idevice'])
else:
    # we use INI['GPU'] and optionally INI['sdevice'] to select the device
    platform, device, context, queue, mf = InitCL_string(INI)
    if (INI['GPU']):  LOCAL = 32
    else:             LOCAL =  1
    if (INI['LOCAL']>0): LOCAL = INI['LOCAL']
    

if (OCTREE<2):
    NWG    =  -1
    GLOBAL =  IRound(NRAY, 32)
elif (OCTREE in [2,3,4,5]):   # one work group per ray
    NWG    =  NRAY
    # for memory reasons, we need to limit NWG
    # BUFFER requires possibly 20 kB per work group, limit that to ~100MB => maximum of ~10000 work groups
    #  each level can add one fron-ray and one side-ray entry to the buffer,
    #  each entry is for OCTREE4  equal to 26+CHANNELS floats
    #  MAXL=7 ==   7*2*(26+CHANNELS) =  4 kB if CHANNELS=256
    #  NWG=16384  =>  buffer allocation 61.7 MB
    ########################
    #  BUFFER  ~  8*NWG*(26+CHANNELS)*MAX_NBUF
    NWG   = min([NRAY,    int(1+200.0e6/(8.0*(26.0+CHANNELS)*MAX_NBUF))   ])
    if (NWG>16384):
        NWG = 16384
    
    ## NWG = 1024
    
    GLOBAL =  IRound(NWG*LOCAL, 32)
    print("*** NWG SET TO %d   -->   GLOBAL %d --- NWG=%d == GLOBAL/LOCAL = %.3f" % \
    (NWG, GLOBAL, NWG, GLOBAL/float(LOCAL)))
elif (OCTREE in [40,]):  # one work item per ray
    NWG    =  -1
    GLOBAL =  IRound(NRAY, 32)  # for ONESHOT, this is ~NX*NY
    
    
    

OPT = " -D NX=%d -D NY=%d -D NZ=%d -D NRAY=%d -D CHANNELS=%d -D WIDTH=%.5ff -D ONESHOT=%d \
-D VOLUME=%.5ef -D CELLS=%d -D LOCAL=%d -D GLOBAL=%d -D GNO=%d -D SIGMA0=%.5ff -D SIGMAX=%.4ff \
-D GL=%.4ef -D MAXCHN=%d -D WITH_HALF=%d -D PLWEIGHT=%d -D LOC_LOWMEM=%d -D CLIP=%.3ef -D BRUTE_COOLING=%d \
-D LEVELS=%d -D TRANSITIONS=%d -D WITH_HFS=%d -D WITH_CRT=%d -D DOUBLE_POS=%d -D MAX_NBUF=%d \
-I%s -D OTLEVELS=%d -D NWG=%d -D WITH_OCTREE=%d -D GAUSTORE=%s -D WITH_ALI=%d -D FIX_PATHS=%d \
-D MINMAPLEVEL=%d -D MAP_INTERPOLATION=%d " % \
(NX, NY, NZ, NRAY, CHANNELS, WIDTH, ONESHOT, VOLUME, CELLS, LOCAL, GLOBAL, GNO, G0, GX,
GL, MAXCHN, WITH_HALF, PLWEIGHT, LOWMEM, INI['clip'], (COOLING==2),
LEVELS, TRANSITIONS, HFS, WITH_CRT, DOUBLE_POS, MAX_NBUF, 
INSTALL_DIR, OTL, NWG, OCTREE, GAUSTORE, WITH_ALI, FIX_PATHS, 
int(INI['minmaplevel']), INI['MAP_INTERPOLATION'])

if (0):
    # -cl-fast-relaxed-math == -cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only
    # -cl-unsafe-math-optimizations is ***DANGEROUS*** ON NVIDIA
    OPT += "-cl-mad-enable -cl-no-signed-zeros -cl-finite-math-only"  # this seems ok on NVidia !!
    
if (0):
    OPT  += " -cl-std=CL1.1"
if (INI['GPU']==99):  # Intel fails .... does optimisation work for POCL? Yes - with no visible effect...
    OPT  += " -cl-opt-disable"  # opt-disable faster??? at least 3D up to 64^3
if (1):
    print("Kernel options:")
    print(OPT)

# Note: unlike in 1d versions, both SIJ and ESC are divided by VOLUME only in the solver
#       3d and octree use kernel_LOC_aux.c, where the solver is incompatible with the 1d routines !!!
source    =  open(INSTALL_DIR+"/kernel_update_py_OT.c").read()
# @ MAXL3_P20.0 WITH_HALF   10.7/9.6   s1   65/7.5  68645100 7834372
# @ gid   54746 ------  NBUF = 50 !!!!!!!!!!!!!!!!!!!!!!
# @ during run 68646124 7834372
# @ MAXL3_P20.0 WITH_HALF   10.7/9.6    65.5/7.5
#print("--- Create program -------------------------------------------------------------")
program       =  cl.Program(context, source).build(OPT, cache_dir=None)
#print("--------------------------------------------------------------------------------")

# Set up kernels
kernel_clear   =  program.Clear
kernel_paths   =  None   # only if PLWEIGHT in use... no we need this also to get correct PACKETS !!
if (OCTREE>0):
    if (OCTREE==1): 
        print("USING  program.UpdateOT1: offs %d" % INI['offsets'])
        kernel_sim   =  program.UpdateOT1
        kernel_paths =  program.PathsOT1
    elif (OCTREE==2):           
        print("USING  program.UpdateOT2: offs %d, ALI %d" % (INI['offsets'], WITH_ALI))
        kernel_sim   =  program.UpdateOT2
        kernel_paths =  program.PathsOT2
    elif (OCTREE==3):
        print("USING  program.UpdateOT3:  ALI %d" % ( WITH_ALI))
        kernel_sim   =  program.UpdateOT3
        kernel_paths =  program.PathsOT3
    elif (OCTREE==4):
        print("USING  program.UpdateOT4:  ALI %d" % ( WITH_ALI))
        kernel_sim   =  program.UpdateOT4
        kernel_paths =  program.PathsOT4
    elif (OCTREE==5):
        print("USING  program.UpdateOT4:  ALI %d" % ( WITH_ALI))
        kernel_sim   =  program.UpdateOT5
        kernel_paths =  program.PathsOT5
    elif (OCTREE==40):
        print("USING  program.UpdateOT40:  ALI %d" % ( WITH_ALI))
        kernel_sim   =  program.UpdateOT40
        kernel_paths =  program.PathsOT40        
else:
    print("USING  program.Update")
    kernel_sim       =  program.Update
    if (PLWEIGHT):  kernel_paths     =  program.Paths

kernel_solve         =  program.SolveCL
kernel_parents       =  program.Parents


# 2020-05-31:  WITH_CRT implemented only for the default case (no HFS)
if (WITH_CRT):
    kernel_sim.set_scalar_arg_dtypes([
    # 0     1     2     3           4           5           6           7      8           9           10        
    # id0   CLOUD GAU   LIM         Aul         A_b         GN          PL     APL         BG          DIRWEI    
    None,   None, None, np.float32, np.float32, np.float32, np.float32, None,  np.float32, np.float32, np.float32,
    #  11       12       13     14      15     16     17     18       19      
    #  EWEI     LEADING  POS0   DIR     NI     RES    NTRUE  CRT_TAU  CRT_EMIT
    np.float32, np.int32,  REAL3, FLOAT3, None,  None,  None,  None,    None  ])
else:
    if (OCTREE<1):
        kernel_sim.set_scalar_arg_dtypes([
        # 0        1      2     3     4           5           6           7     8           9           
        # id0      CLOUD  GAU   LIM   Aul         A_b         GN          PL    APL         BG          
        np.int32,  None,  None, None, np.float32, np.float32, np.float32, None, np.float32, np.float32,
        # 10        11          12         13      14      15      16     17    
        # DIRWEI    EWEI        LEADING    POS0    DIR     NI      RES    NTRUE 
        np.float32, np.float32, np.int32,  REAL3,  FLOAT3, None,  None,   None  ])
    elif (OCTREE==1):
        if (PLWEIGHT>0):
            kernel_sim.set_scalar_arg_dtypes([
            # 0   1     2     3           4           5           6           7           8     9           10
            # PL  CLOUD GAU   LIM         Aul         A_b         GN          APL         BG    DIRWEI      EWEI
            None, None, None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
            #   11     12     13       14     15     16   
            # LEADING  POS0   DIR      NI     RES    NTRUE
            np.int32,  REAL3, FLOAT3,  None,  None,  None ,
            #  17      18     19     20 
            # LCELLS   OFF    PAR    RHO
            None,      None,  None,  None])
        else:
            kernel_sim.set_scalar_arg_dtypes([
            # 0     1     2     3           4           5           6           7           8           9    
            # CLOUD GAU   LIM   Aul         A_b         GN          APL         BG          DIRWEI      EWEI,
            None,   None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
            #  10      11     12       13     14     15   
            # LEADING  POS0   DIR      NI     RES    NTRUE
            np.int32,  REAL3, FLOAT3,  None,  None,  None ,
            #  16      17     18     19    
            # LCELLS   OFF    PAR    RHO   
            None,      None,  None,  None])
    elif (OCTREE in [2,3]): # OCTREE=2,3
        kernel_sim.set_scalar_arg_dtypes([
        # 0    1      2     3     4           5           6           7           8           9           10        
        # PL   CLOUD  GAU   LIM   Aul         A_b         GN          APL         BG          DIRWEI      EWEI      
        None,  None,  None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
        #  11      12      13       14     15     16    
        # LEADING  POS0    DIR      NI     RES    NTRUE 
        np.int32,  REAL3,  FLOAT3,  None,  None,  None,
        #  17     18     19     20     21     
        # LCELLS, OFF,   PAR,   RHO    BUFFER 
        None,     None,  None,  None,  None ])
    elif (OCTREE>=4):
        if (PLWEIGHT>0):
            kernel_sim.set_scalar_arg_dtypes([
            # 0       1      2      3     4     5           6           7           8           9           
            # gid0    PL     CLOUD  GAU   LIM   Aul         A_b         GN          APL         BG          
            np.int32, None,  None,  None, None, np.float32, np.float32, np.float32, np.float32, np.float32, 
            #  10       11          12         13      14       15     16     17    
            #  DIRWEI   EWEI        LEADING    POS0    DIR      NI     RES    NTRUE 
            np.float32, np.float32, np.int32,  REAL3,  FLOAT3,  None,  None,  None,
            #  18     19     20     21     22     
            # LCELLS, OFF,   PAR,   RHO    BUFFER 
            None,     None,  None,  None,  None ])
        else:
            kernel_sim.set_scalar_arg_dtypes([
            # 0       1      2     3     4      5           6           7           8              
            # gid0    CLOUD  GAU   LIM   Aul    A_b         GN          APL         BG             
            np.int32, None,  None, None, np.float32, np.float32, np.float32, np.float32, np.float32, 
            #  9        10          11         12      13       14     15     16    
            #  DIRWEI   EWEI        LEADING    POS0    DIR      NI     RES    NTRUE 
            np.float32, np.float32, np.int32,  REAL3,  FLOAT3,  None,  None,  None,
            #  17     18     19     20     21    
            # LCELLS, OFF,   PAR,   RHO    BUFFER 
            None,     None,  None,  None,  None ])
   
#                                   RES[CELLS].xy
kernel_clear.set_scalar_arg_dtypes([None, ])

if (OCTREE==0):  ##  Cartesian
    if (PLWEIGHT>0):
        kernel_paths.set_scalar_arg_dtypes([np.int32, None, None, None,  np.int32, REAL3, FLOAT3])
    else:
        kernel_paths.set_scalar_arg_dtypes([np.int32, None, None,  np.int32, REAL3, FLOAT3])
elif (OCTREE==1):
    kernel_paths.set_scalar_arg_dtypes([None, None, None, np.int32, REAL3, FLOAT3,
    None, None, None, None])
elif (OCTREE in [2, 3]):
    kernel_paths.set_scalar_arg_dtypes([None, None, None,  np.int32, REAL3, FLOAT3, 
    None, None, None, None, None])
elif (OCTREE>=4):
    kernel_paths.set_scalar_arg_dtypes([np.int32, None, None, None,  np.int32, REAL3, FLOAT3,
    None, None, None, None, None])    


if (HFS):   
    if (OCTREE==0):
        kernel_hf    =  program.UpdateHF
        kernel_hf.set_scalar_arg_dtypes([
        # 0     1     2     3           4           5           6           7           8           9         
        # CLOUD GAU   LIM   Aul         A_b,        GN          APL         BG          DIRWEI      EWEI      
        None,   None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
        # 10       11       12     13    14    15        16        17     18      19 
        # LEADING  POS      DIR    NI    RES   NCHN      NCOMP     HF     NTRUES  PROFILE
        np.int32,  REAL3,  FLOAT3, None, None, np.int32, np.int32, None,  None,   None])
    else:
        if (OCTREE==4):
            kernel_hf    =  program.UpdateHF4
            kernel_hf.set_scalar_arg_dtypes([
            # 0       1     2      3     4     5           6           7           8           9         
            # gid0    PL,   CLOUD  GAU   LIM   Aul         A_b,        GN          APL         BG        
            np.int32, None, None,  None, None, np.float32, np.float32, np.float32, np.float32, np.float32,
            # 10          11        12        13     14      15    16        17        17    
            # DIRWEI      EWEI      LEADING   POS    DIR     NI    NCHN      NCOMP     HF    
            np.float32, np.float32, np.int32, REAL3, FLOAT3, None, np.int32, np.int32, None,
            # 18   19      20      21    22    23    24    
            # RES  NTRUES  LCELLS  OFF   PAR   RHO   BUFFER
            None,  None,   None,   None, None, None, None  ])
        else:
            print("HFS not implemented for OCTREE=%d" % OCTREE), sys.exit()
else:       
    kernel_hf    =  None


if (OCTREE>0):
    kernel_solve.set_scalar_arg_dtypes(
    # 0        1         2     3     4     5     6         7         8         9         10    11    12   
    # OTL      BATCH     A     UL    E     G     PARTNERS  NTKIN     NCUL      MOL_TKIN  CUL   C     CABU 
    [np.int32, np.int32, None, None, None, None, np.int32, np.int32, np.int32, None,     None, None, None,
    # 13  14    15    16    17    18    19    20   
    # RHO TKIN  ABU   NI    SIJ   ESC   RES   WRK   
    None, None, None, None, None, None, None, None, np.int32 ])
else:
    kernel_solve.set_scalar_arg_dtypes(
    # 0        1     2     3     4     5         6         7         8         9     10    11   
    # BATCH    A     UL    E     G     PARTNERS  NTKIN     NCUL      MOL_TKIN  CUL   C     CABU 
    [np.int32, None, None, None, None, np.int32, np.int32, np.int32, None,     None, None, None,
    # 12  13    14    15    16    17    18    19    
    # RHO TKIN  ABU   NI    SIJ   ESC   RES   WRK   
    None, None, None, None, None, None, None, None, np.int32 ])


# Set up input and output arrays
# print("Set up input arrays")
if (PLWEIGHT):
    PL_buf =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)   # could be half??
else:
    PL_buf =  cl.Buffer(context, mf.READ_WRITE, 4)         # dummy
GAU_buf    =  cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=GAU)
LIM_buf    =  cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=LIM)
TPL_buf    =  cl.Buffer(context, mf.READ_WRITE,  4*NRAY)
COUNT_buf  =  cl.Buffer(context, mf.READ_WRITE,  4*NRAY)
if (0):
    CLOUD_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=CLOUD)   # vx, vy, vz, sigma
else:
    # On the host side, CLOUD is the second largest array after NI_ARRAY.
    # Here we drop CLOUD before the NI_ARRAY is allocated.
    if (WITH_HALF==0):
        CLOUD_buf =  cl.Buffer(context, mf.READ_ONLY, 4*4*CELLS)   # vx, vy, vz, sigma
    else:
        CLOUD_buf =  cl.Buffer(context, mf.READ_ONLY, 2*4*CELLS)   # vx, vy, vz, sigma
    cl.enqueue_copy(queue, CLOUD_buf, CLOUD)
    CLOUD = None

    
NI_buf    =  cl.Buffer(context, mf.READ_ONLY,   8*CELLS)                          # nupper, nb_nb
if (WITH_ALI>0):
    RES_buf   =  cl.Buffer(context, mf.READ_WRITE,  8*CELLS)                      # SIJ, ESC
else:
    RES_buf   =  cl.Buffer(context, mf.READ_WRITE,  4*CELLS)                      # SIJ only
HF_buf    =  None
if (HFS):
    HF_buf  =  cl.Buffer(context, mf.READ_ONLY,   8*MAXCMP)

if (COOLING==2):
    COOL_buf = cl.Buffer(context,mf.WRITE_ONLY, 4*CELLS)
    
NTRUE_buf =  cl.Buffer(context, mf.READ_WRITE, 4*max([INI['points'][0], NRAY])*MAXCHN)
STAU_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*max([INI['points'][0], NRAY])*MAXCHN)
WRK       =  np.zeros((CELLS,2), np.float32)     #  NI=(nu, nb_nb)  and  RES=(SIJ, ESC)

# Buffers for SolveCL
NTKIN        =  len(MOL.TKIN[0])
PARTNERS     =  MOL.PARTNERS
NCUL         =  MOL.CUL[0].shape[0] 
MOL_A_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.A)
MOL_UL_buf   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.TRANSITION)
MOL_E_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.E)
MOL_G_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.G)
MOL_TKIN_buf =  cl.Buffer(context, mf.READ_ONLY, 4*PARTNERS*NTKIN)
MOL_TKIN     =  zeros((PARTNERS, NTKIN), float32)
for i in range(PARTNERS):
    MOL_TKIN[i, :]  =  MOL.TKIN[i][:]
cl.enqueue_copy(queue, MOL_TKIN_buf, MOL_TKIN)        
# CUL  -- levels are included separately for each partner... must have the same number of rows!
#         KERNEL ASSUMES IT IS THE SAME TRANSITIONS, IN THE SAME ORDER
if (MOL.CUL[i].shape[0]!=NCUL):
    print("SolveCL assumes the same number of C rows for each collisional partner!!"),  sys.exit()
CUL   =  zeros((PARTNERS,NCUL,2), int32)
for i in range(PARTNERS):
    CUL[i, :, :]  =  MOL.CUL[i][:,:]
    # KERNEL USES ONLY THE CUL ARRAY FOR THE FIRST PARTNER -- CHECK THAT TRANSITIONS ARE IN THE SAME ORDER
    delta =   np.max(ravel(MOL.CUL[i]-MOL.CUL[0]))
    if (delta>0):
        print("*** ERROR: SolveCL assumes all partners have C in the same order of transitions!!"), sys.exit()
MOL_CUL_buf  = cl.Buffer(context, mf.READ_ONLY, 4*PARTNERS*NCUL*2)
cl.enqueue_copy(queue, MOL_CUL_buf, CUL)        
# C
C    =  zeros((PARTNERS, NCUL, NTKIN), float32)
for i in range(PARTNERS):   
    C[i, :, :]  =  MOL.CC[i][:, :]
MOL_C_buf  = cl.Buffer(context, mf.READ_ONLY, 4*PARTNERS*NCUL*NTKIN)
cl.enqueue_copy(queue, MOL_C_buf, C)        
# abundance of collisional partners
MOL_CABU_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.CABU)    
# new buffer for matrices and the right side of the equilibriumm equations
BATCH        =  max([1,CELLS//max([LEVELS, TRANSITIONS])]) # now ESC, SIJ fit in NI_buf
BATCH        =  min([BATCH, CELLS, 16384])        #  16384*100**2 = 0.6 GB

SOL_WRK_buf  = cl.Buffer(context, mf.READ_WRITE, 4*BATCH*LEVELS*(LEVELS+1))
SOL_RHO_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
SOL_TKIN_buf = cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
SOL_ABU_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
SOL_SIJ_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH*TRANSITIONS)
SOL_ESC_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH*TRANSITIONS)
SOL_NI_buf   = cl.Buffer(context, mf.READ_ONLY,  4*BATCH*LEVELS)


if (WITH_CRT):
    # unlike in LOC1D.py, buffers are for single transition at a time
    CRT_TAU_buf = cl.Buffer(context, mf.READ_ONLY, 4*CELLS)
    CRT_EMI_buf = cl.Buffer(context, mf.READ_ONLY, 4*CELLS)
                

    
if (OCTREE>0):
    # Octree-hierarchy-specific buffers
    LCELLS_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=LCELLS)
    OFF_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=OFF)
    PAR_buf     = cl.Buffer(context, mf.READ_WRITE, 4*max([1, (CELLS-NX*NY*NZ)]))  # no space for root cells!
    RHO_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=RHO)
    BUFFER_buf  = None
    if (OCTREE==2):
        BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*NWG*(14+CHANNELS)*512)    # 1024 ???
    elif (OCTREE==3):  #  OTL, OTI,  {x, y, z, OTLRAY} [CHANNELS]  = 2+12*4+CHANNELS
        BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*NWG*(50+CHANNELS)*512)
    elif (OCTREE in [4,5]):  #  OTL, OTI,  {x, y, z, OTLRAY} [CHANNELS]  = 2+12*4+CHANNELS
        #  for each additionl level of hierarchy, original ray + 2 new rays to buffer, one new continues
        #  => BUFFER must be allocated to for just 3*MAXL rays... actually there are also siderays
        #  leading-edge rays go to one slot (26+CHANNELS) includes space for 4 rays),
        #  siderays go to another slot with (26+CHANNELS) numbers
        #  => each increase of refinement requires two slots (slot = 26+CHANNELS)
        #  for MAXL=7 that is 14 slots
        #  if root grid is 512^2, NWG~256^2 and the memory requirement for BUFFER is
        #  256^2*14*(26+CHANNELS~256)*4B  ~ 1 GB, 2 GB for DOUBLE_POS ....        
        #  => we limit NWG (OCTREE==4) and loop over the rays using several kernel calls
        #  Also, if we are computing hyperfine structure lines (HFS), we need storage for 
        #  not CHANNELS but for MAXCHN channels!!
        #  Combining HFS and ONESHOT options then means that 512^3 root grid and MAXCHN=512 lead to
        #     NWG=65536  x   MAXCHN=1024  x  4B  = 320 MB  ... still ok
        if (HFS):
            if (DOUBLE_POS>0): BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 8*NWG*(26+MAXCHN)*MAX_NBUF)
            else:              BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*NWG*(26+MAXCHN)*MAX_NBUF)
        else:
            if (DOUBLE_POS>0): BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 8*NWG*(26+CHANNELS)*MAX_NBUF)
            else:              BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*NWG*(26+CHANNELS)*MAX_NBUF)
            print("BUFFER ALLOCATION  %.3e MB" % (float(8.0*NWG*(26+MAXCHN)*MAX_NBUF)/(1024.0*1024.0)))
            ## sys.exit()
            
    elif (OCTREE in [40,]):
        # We allocate in BUFFER also room for GLOBAL*CHANNELS, the NTRUE vectors in the update kernel
        print("OCTREE=40, ALLOCATE BUFFER = %.3f MB" % ( (GLOBAL*(26+CHANNELS)*MAX_NBUF + 4*GLOBAL*CHANNELS)/(1024.0*1024.0)))
        BUFFER_buf  = cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL*(26+CHANNELS)*MAX_NBUF + 4*GLOBAL*CHANNELS)
        
    # update links from cells to parents:  DENS  LCELLS  OFF   PAR
    kernel_parents.set_scalar_arg_dtypes([ None, None,   None, None])
    program.Parents(queue, [GLOBAL,], [LOCAL,], RHO_buf, LCELLS_buf, OFF_buf, PAR_buf)
    

PROFILE_buf = cl.Buffer(context, mf.READ_WRITE, 4)  # dummy
if (LOWMEM & HFS):
    # the writing of spectra needs GLOBAL*MAXCHN for the profile vectors
    print("PROFILE BUFFER ALLOCATED FOR %d x %d = %.3e FLOATS" % (GLOBAL, MAXCHN, GLOBAL*MAXCHN))
    PROFILE_buf = cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL*MAXCHN)
else:
    PROFILE_buf = cl.Buffer(context, mf.READ_WRITE, 4)  # dummy
    
    
    
if (0):
    print("Stopping before calculations...")
    sys.exit()
    

# 2020-06-26 using  DIRWEI ==  normalisation factor <cos theta>, calculated for the NDIR directions
#                   DIRWEI may still be different for the six sides, depending on how the directions are
PACKETS     =   0
TPL, COUNT  =  [], []
DIRWEI      =  zeros(6, float32)  #  <cos(theta)> = <DIR.normal_component> for each six sides
EWEI        =  0.0 
RCOUNT      =  zeros(6, int32)
TRUE_APL    =  0.0
if (PLWEIGHT):
    PL      =  1.0e-30*ones(CELLS, float32)
else:
    PL      =  None
if (INI['iterations']>0):
    TPL     =  zeros(NRAY, float32)
    COUNT   =  zeros(NRAY, int32)
    offs    =  INI['offsets']          # default = 1, one ray per root grid surface element
    print("Paths...  for NRAY=%d, CELLS=%d, offs=%d" % (NRAY, CELLS, offs))
    if (PLWEIGHT):
        cl.enqueue_copy(queue, PL_buf,  PL)
        cl.enqueue_copy(queue, TPL_buf, TPL)
        queue.finish()
    t00     =  time.time()  
    inner_loop = 4*offs*offs

    for idir in range(NDIR):
        # offsets=1  =>   for ioff in range(4),    offsets=2 => for ioff in range(16)
        
        
        if (ONESHOT): inner_loop = 1
        
        for ioff in range(inner_loop):

            if (0):
                theta0, phi0  =   30.0, 10.0    # +Z
                theta0, phi0  =   170.0, 10.0   # -Z
                theta0, phi0  =   60.0,  60.0   # +Y
                theta0, phi0  =   60.0, -50.0   # -Y
                theta0, phi0  =   55.0,  25.0   # +X
                theta0, phi0  =   55.0,  155.0  # -X  ???
                theta0, phi0  =   60.0,  170.0  # -X  ???
                theta0, phi0  =   60.0,  160.0  # -X  ???
                theta0, phi0  =   theta0*pi/180.0, phi0*pi/180.0
                
            # if (idir!=13): continue
            
            queue.finish()
            POS, DIR, LEADING = GetHealpixDirection(NSIDE, ioff, idir, NX, NY, NZ, offs, DOUBLE_POS) ## , theta0, phi0)

            if (0):
                print("IDIR %2d / %2d     %8.5f %8.5f %8.5f  %7.4f %7.4f %7.4f   %d" % 
                (idir, NDIR, POS['x'], POS['y'], POS['z'], DIR['x'], DIR['y'], DIR['z'], LEADING))
            
            
            # Calculate DIRWEI part
            #    DIRWEI is calculated taking into account all rays that enter a given side
            #    However, kernel will use DIRWEI[i] only for rays with leading edge i.
            #    Does that mean that this direction weighting is less useful?
            #    or DIRWEI should be calculated based on leading edge cos(theta) only???
            #    .... because rays entering the sides are further apart in surface elements.
            if    (LEADING==0):  
                DIRWEI[0] += DIR['x']        # summing cos(theta) of the rays hitting given side
                RCOUNT[0] += 1
            elif  (LEADING==1):                
                DIRWEI[1] -= DIR['x'] 
                RCOUNT[1] += 1
            elif  (LEADING==2):
                DIRWEI[2] += DIR['y'] 
                RCOUNT[2] += 1                
            elif  (LEADING==3):
                DIRWEI[3] -= DIR['y'] 
                RCOUNT[3] += 1
            elif  (LEADING==4):
                DIRWEI[4] += DIR['z'] 
                RCOUNT[4] += 1
            elif  (LEADING==5):
                DIRWEI[5] -= DIR['z'] 
                RCOUNT[5] += 1
            ####
            if (LEADING   in [0,1]):
                EWEI +=  1.0/abs(DIR['x'])
            elif (LEADING in [2,3]):
                EWEI +=  1.0/abs(DIR['y'])
            else:
                EWEI +=  1.0/abs(DIR['z'])

            # we have to process NRAY with NWG workgroups, possibly NWG<NRAY
            if (OCTREE==0):
                if (NRAY%GLOBAL==0): niter = NRAY//GLOBAL
                else:                niter = NRAY//GLOBAL+1
                for ibatch in range(niter):
                    if (PLWEIGHT>0):
                        kernel_paths(queue, [GLOBAL,], [LOCAL], 
                        ibatch*GLOBAL, PL_buf, TPL_buf, COUNT_buf, LEADING, POS, DIR)
                    else:
                        kernel_paths(queue, [GLOBAL,], [LOCAL], 
                        ibatch*GLOBAL,         TPL_buf, COUNT_buf, LEADING, POS, DIR)                
            elif (OCTREE==1):
                kernel_paths(queue, [GLOBAL,], [LOCAL], PL_buf, TPL_buf, COUNT_buf, LEADING, POS, DIR, 
                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf)
            elif ((OCTREE==2)|(OCTREE==3)):
                kernel_paths(queue, [GLOBAL,], [LOCAL], PL_buf, TPL_buf, COUNT_buf, LEADING, POS, DIR, 
                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, BUFFER_buf)            
            elif (OCTREE in [4,5]): # one work group per ray
                if (NRAY%NWG==0): niter = NRAY//NWG     # ONESHOT => NRAY ~ NX*NY, not ~(NX/2)*(NY/2)
                else:             niter = (NRAY//NWG)+1
                if (PLWEIGHT<1):  niter = 0   # just skip the path calculation
                
                for ibatch in range(niter):
                    # print("paths, idir %3d, ioff %3d, batch %d/%d, dir %7.3f %7.3f %7.3f -- PLWEIGHT=%d, NWG=%d" % (idir, ioff, ibatch, niter, DIR['x'], DIR['y'], DIR['z'], PLWEIGHT, NWG))                        
                    kernel_paths(queue, [GLOBAL,], [LOCAL],
                    #gid0       [CELLS]  [NRAY]    [NRAY]      np.int32
                    ibatch*NWG, PL_buf,   TPL_buf, COUNT_buf,  LEADING,
                    # REAL3  float3  [LEVELS]    [LEVELS]   [CELLS-NX*NY*NZ]  CELLS     [4|8]*NWG*(26+CHANNELS)*MAX_NBUF
                    POS,     DIR,    LCELLS_buf, OFF_buf,   PAR_buf,          RHO_buf,  BUFFER_buf)
                    queue.finish()
                    
            elif (OCTREE in [40,]):  # one work item per ray
                # with option ONESHOT, kernel will loop over all ray offsets, not only every other
                if (NRAY%GLOBAL==0): niter = NRAY//GLOBAL
                else:                niter = NRAY//GLOBAL+1
                if (PLWEIGHT<1): niter = 0   # just skip the path calculation
                for ibatch in range(niter):
                    kernel_paths(queue, [GLOBAL,], [LOCAL], ibatch*GLOBAL, PL_buf,  TPL_buf,  COUNT_buf,  LEADING,
                    POS,     DIR,    LCELLS_buf, OFF_buf,   PAR_buf,       RHO_buf,  BUFFER_buf)            
                
            queue.finish()
            cl.enqueue_copy(queue, COUNT, COUNT_buf)
            queue.finish()
            # PACKETS = total number of packets from the background, including rays continued to model sides
            for i in range(NRAY):  PACKETS  += COUNT[i]      

            if (LEADING in [0,1]):
                TRUE_APL          +=  1.0/fabs(DIR['x'])     # expected path length per root grid cell
            elif (LEADING in [2,3]):
                TRUE_APL          +=  1.0/fabs(DIR['y'])
            else:
                TRUE_APL          +=  1.0/fabs(DIR['z'])            
    

            if (PLWEIGHT):
                cl.enqueue_copy(queue, PL, PL_buf)
                queue.finish()
                m = nonzero(RHO[0:NX*NY*NZ]>0.0)
                if (0):
                    if (len(m[0])>0):
                        print("  <PL> root grid: %.3e +- %.3e" % (mean(PL[0:NX*NY*NZ][m[0]]), std(PL[0:NX*NY*NZ][m[0]])))
             
                    
        ### break
    ### end for idir
    
    if (PLWEIGHT):
        cl.enqueue_copy(queue, TPL,   TPL_buf)
        # print("PL[20444] = %8.4f" % PL[20444])
        if (min(TPL)<0.0):
            print("PATH KERNEL RAN OUT OF BUFFER SPACE !!!")
            sys.exit()
        
    #   BG is calculated as the total number of photons entering the clouds divided by the number of
    #      rays == AVERAGE NUMBER OF PHOTONS PER PACKAGE
    #   Individual rays are weighted ~   cos(theta) / <cos(theta)>, where the denominator should be 0.5
    #   We provide kernel the weight factors  =  cos(theta) / <cos(theta)>
    if (1):
        # 2020-01-13 -- revised weighting, BGPAC * cos(theta) / DIRWEI, where DIRWEI
        #               is the sum of cos(theta) for each side separately
        #               *and* the weighting of photon packages does not depend on PACKETS variable
        print("NEW DIRWEI ", DIRWEI)    # nside=0  => DIRWEI[:] ~ 1.0
        # sys.exit()
    else:
        DIRWEI  /=  RCOUNT         #  now DIRWEI is <cos(theta)> for the rays entering each of the six sides
    EWEI    /=  inner_loop ;       #  <1/cosT>
    EWEI     =  1.0/(EWEI*NDIR)    #  emission from a cell ~  (1/cosT) * EWEI, larger fraction when LOS longer 
    if (ONESHOT<1):
        TRUE_APL =  TRUE_APL/4.0   # no division by offs*offs !!   -- fixed 2020-07-20
    APL      =  TRUE_APL

    # average path length APL/(inner_loop*NDIR) through a cell
    # random rays + cubic shape => average should be 1.222
    # possible additional weighting   1.222*inner_loop*NDIR/APL ?
    # APL_WEIGHT  =  1.222*inner_loop*NDIR/APL
    # APL_WEIGHT *=  0.8
    # print("APL_WEIGHT = %8.5f" % APL_WEIGHT)
        
    if (PLWEIGHT):
        cl.enqueue_copy(queue, PL, PL_buf)
        m = nonzero(RHO>0.0)
        print('SUM OF PL = %.3e, APL %.3e,  <PL> %.3e, TRUE_APL %.3e, PACKETS=%d' %
        (sum(PL), APL, mean(PL[m[0]]), TRUE_APL, PACKETS))
        m = nonzero(RHO[0:NX*NY*NZ]>0.0)
        if (len(m[0])>0):
            print("<PL> for root grid cells: %.3e +- %.3e" % (mean(PL[0:NX*NY*NZ][m[0]]), std(PL[0:NX*NY*NZ][m[0]])))
        # print("PL[20444] = %8.4f" % PL[20444])
    print("Paths kernel: %.3f seconds" % (time.time()-t00))
    # @   WITH_HALF=1   68646124 7840992   65.5/7.7 GB
    # print('DIRWEI     ', DIRWEI)
    
    # @@ 
    if (0):
        print("SAVING PL%d.dat" % OCTREE)
        PL.tofile('PL%d.dat' % OCTREE)
        sys.exit()
        
    if (OCTREE>999):
        tmp  = PL[0:NX*NY*NZ].copy()
        tmp2 = tmp[nonzero(RHO[0:NX*NY*NZ]>0.0)].copy()
        print("FIRST OCTREE LEVEL PL", percentile(tmp2, (0.0, 1.0, 10.0, 50.0, 90.0, 99.0, 100.0)))
        tmp.shape = (NX,NY,NZ)
        clf()
        for ii in range(4):
            subplot(2,2,1+ii)
            title("i=%d" % (NX//2-2+ii))
            imshow(tmp[NX//2-2+ii,:,:])
            colorbar()
        show()
        sys.exit()
    ## sys.exit()    
    if (0): # cartesian
        PL.shape = (NX, NY, NZ)
        subplot(221)
        imshow(PL[NX//2-2,:,:])
        colorbar()
        subplot(222)
        imshow(PL[NX//2-1,:,:])
        colorbar()
        subplot(223)
        imshow(PL[NX//2-0,:,:])
        colorbar()
        subplot(224)
        imshow(sum(PL, axis=0))
        colorbar()
        show()
        sys.exit()
    
    if (PLWEIGHT):
        m = nonzero(RHO>0.0)
        print('PL  ', percentile(PL[m], (0.0, 1.0, 10.0, 50.0, 90.0, 99.0, 100.0)))
    # sys.exit()
    if (0):
        print("APL = %.3e    ---    <PL> = %.3e" % (APL, mean(PL))) # yes, they are ~ the same
        clf()
        for i in range(OTL):
            m = nonzero(RHO[OFF[i]:(OFF[i]+LCELLS[i])]>0.0)
            plot(PL[OFF[i]:(OFF[i]+LCELLS[i])][m], '.', label='LEVEL %d' % i)
        legend()
        show()
        sys.exit()
        
    if (0):
        x =   PL[OFF[1]:(OFF[1]+LCELLS[1])]
        print("PL, LEVEL1", percentile(x, (0.0, 1.0, 10.0, 50.0, 90.0, 99.0, 100.0)))
        # PL[OFF[1]:(OFF[1]+LCELLS[1])]  = APL/2.0
        # sys.exit()
    
    
# Read or generate NI_ARRAY
if (LOWMEM>1): # not only is kernel using more global arrays, also NI_ARRAY, SIJ_ARRAY, ESC_ARRAY are memmap
    # do we keep this separate from load/save... in case one does not want to load/save the populations....
    fp = open('LOC_NI.mmap', 'wb')
    asarray([NX, NY, NZ, LEVELS], int32).tofile(fp)
    fp.close()
    NI_ARRAY = np.memmap('LOC_NI.mmap', dtype='float32', mode='r+', offset=16, shape=(CELLS, LEVELS))
    NI_ARRAY[:,:] = 1.0
else:
    NI_ARRAY = ones((CELLS, LEVELS), float32)
ok = False
if (len(INI['load'])>0):  # load saved level populations
    try:
        fp = open(INI['load'], 'rb')
        nx, ny, nz, lev = fromfile(fp, int32, 4)
        #print(nx, ny, nz, lev)
        #print(NX, NY, NZ, LEVELS)
        if ((nx!=NX)|(ny!=NY)|(nz!=NZ)|(lev!=LEVELS)):
            print("Reading %s => %d x %d x %d cells, %d levels" % (nx, ny, nz, lev))
            print("but we have now %d x %d x %d cells, %d levels ?? "  % (NX, NY, NZ, LEVELS))
            sys.exit()
        NI_ARRAY[:,:] = fromfile(fp, float32).reshape(CELLS, LEVELS)
        fp.close()
        ok = True
        print("Level populations read from: %s" % INI['load'])
    except:
        print("Failed to load level populations from: %s" % INI['load'])
        pass
if (not(ok)): # reset LTE populations
    print("***** Resetting level populations to LTE values !!! *****")
    J   =  asarray(arange(LEVELS), int32)
    m   =  nonzero(RHO>0.0)
    t0  =  time.time()
    if (0):  # this one took  9    seconds
        for icell in m[0]:
            NI_ARRAY[icell,:] = RHO[icell] * ABU[icell] * MOL.Partition(J, TKIN[icell])
    else:    # this one took  0.01 seconds
        LTE = program.LTE
        #                          BATCH     E      G     TKIN   RHO    NI   
        LTE.set_scalar_arg_dtypes([np.int32, None,  None, None,  None,  None])
        for i in range(CELLS//BATCH+1):
            a     = i*BATCH                    # first index included
            b     = min([CELLS, a+BATCH])      # last index included + 1
            cells = b-a
            if (cells<1): break
            cl.enqueue_copy(queue, SOL_TKIN_buf,  TKIN[a:b].copy())
            cl.enqueue_copy(queue, SOL_RHO_buf,   ABU[a:b]*RHO[a:b])                        
            LTE(queue, [GLOBAL,], [LOCAL,], BATCH, MOL_E_buf, MOL_G_buf, SOL_TKIN_buf, SOL_RHO_buf, SOL_NI_buf)
            cl.enqueue_copy(queue, NI_ARRAY[a:b,:], SOL_NI_buf)
            # print("%7d - %7d    %10.3e %10.3e" % (a, b, NI_ARRAY[a,0], NI_ARRAY[b-1,0])) ;
    print("LTE populations set in %.3f seconds" % (time.time()-t0))
    # write also directly to disk
    if (len(INI['save'])>0):
        fp = open(INI['save'], 'wb')
        asarray([NX, NY, NZ, LEVELS], int32).tofile(fp)
        asarray(NI_ARRAY, float32).tofile(fp)
        fp.close()
        print("Level populations (LTE) saved to: %s" % INI['save'])
        if (0):
            for tr in range(TRANSITIONS):
                u, l = MOL.T2L(tr)
                print("  <ni[%02d]*gg-ni[%02d]> = %11.3e" % (l, u, mean(NI_ARRAY[:,l]*MOL.GG[tr]-NI_ARRAY[:,u])))
            Tkin = 10.0
            for tr in range(TRANSITIONS):
                u, l = MOL.T2L(tr)
                print("  %4.1f==%4.1f <ni[%02d]*gg-ni[%02d]> = %11.3e" % (
                MOL.GG[tr],  (2.0*u+1.0)/(2.0*l+1.0),       l, u,
                MOL.GG[tr]*MOL.Partition(l, Tkin) - MOL.Partition(u, Tkin)) )
            Tkin = 20.0
            for tr in range(TRANSITIONS):
                u, l = MOL.T2L(tr)
                print("  %4.1f==%4.1f <ni[%02d]*gg-ni[%02d]> = %11.3e" % (
                MOL.GG[tr],  (2.0*u+1.0)/(2.0*l+1.0),       l, u,
                MOL.GG[tr]*MOL.Partition(l, Tkin) - MOL.Partition(u, Tkin)) )
            sys.exit()
    if (0):
        print(NI_ARRAY[0,:])
        print(NI_ARRAY[1000,:])
        print(NI_ARRAY[10000,:])
        sys.exit()

            
LTE_10_pop = zeros(LEVELS, float32)
for i in range(LEVELS):
    # print(MOL.G[i], MOL.E[i])
    LTE_10_pop[i] = MOL.G[i] * exp(-MOL.E[i]*PLANCK/(BOLTZMANN*10.0))
LTE_10_pop /= sum(LTE_10_pop)


if (1):
    m = nonzero(~isfinite(sum(NI_ARRAY, axis=1)))
    if (len(m[0])>0):
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        print("NI_ARRAY READ ---- %d WITH DATA NOT FINITE !!!!!!!" % (len(m[0])))
        for i in m[0]:
            NI_ARRAY[i,:] = LTE_10_pop * RHO[i]*ABU[i]
        print("???? FIXED ????")
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        

        
#================================================================================
#================================================================================
#================================================================================
#================================================================================


# @ 65.5/7.7 GB   68646124 7840992
# [  +0.000000] amdgpu 0000:3e:00.0: VM_L2_PROTECTION_FAULT_STATUS:0x00301031



def Simulate():
    global INI, MOL, queue, LOCAL, GLOBAL, WIDTH, VOLUME, GL, COOLING, NSIDE, HFS
    global RES_buf, GAU_buf, CLOUD_buf, NI_buf, LIM_buf, PL, EWEI, PL_buf
    global ESC_ARRAY, SIJ_ARRAY, PACKETS
    ncmp    =  1
    tmp_1   =  C_LIGHT*C_LIGHT/(8.0*pi)
    Tbg     =  INI['Tbg']
    SUM_COOL, LEV_COOL, hf = [], [], []
    if (COOLING==2):
        SUM_COOL = zeros(CELLS, float32)
        LEV_COOL = zeros(CELLS, float32)
        hf       = MOL.F*PLANCK/VOLUME        
    sys.stdout.write('      ')
    for tran in range(MOL.TRANSITIONS): # ------>
        upper, lower  =  MOL.T2L(tran)
        Ab            =  MOL.BB[tran]
        Aul           =  MOL.A[tran]
        freq          =  MOL.F[tran]
        gg            =  MOL.GG[tran]
        BG            =  1.0
        if (1):
            # 2021-01-13 -- weighting directly by cos(theta)/DIRWEI where DIRWEI = sum(cos(theta)) for each side
            #  BGPHOT = number of photons per a single surface element, not the whole cloud
            #  ***AND*** only photons into the solid angle where the largest vector component is 
            #            perpendicular to the surface element !!!!!
            #            instead of pi, integral is 1.74080 +-   0.00016   ==> total number of photons per LEADING
            BGPHOT        =  Planck(freq, Tbg)* 1.74080         /(PLANCK*C_LIGHT) * (1.0e5*WIDTH)*VOLUME/GL
        else:
            #  BGPHOT = total number of photons entering the cloud via area == AREA
            BGPHOT        =  Planck(freq, Tbg)* pi      *  AREA /(PLANCK*C_LIGHT) * (1.0e5*WIDTH)*VOLUME/GL
            # the number of photons that should be assigned an individual packages ON AVERAGE
            # individual packages have additional relative weights, dirwei/DIRWEI == cos(theta) / <cos(theta)>
            BG        =  BGPHOT/PACKETS    

        # print("TRAN=%2d  ==>  BGPHOT = BG*PACKETS = %12.4e" % (tran, BGPHOT))
        
        if (HFS):
            nchn = BAND[tran].Channels()
            ncmp = BAND[tran].N
            for i in range(ncmp):
                HF[i]['x']  =  round(BAND[tran].VELOCITY[i]/WIDTH) # offset in channels
                HF[i]['y']  =  BAND[tran].WEIGHT[i]
            HF[0:ncmp]['y']  /= sum(HF[0:ncmp]['y'])
            cl.enqueue_copy(queue, HF_buf, HF)
        if (WITH_CRT):
            cl.enqueue_copy(queue, CRT_TAU_buf, asarray(CRT_TAU[:,tran].copy(), float32))
            cl.enqueue_copy(queue, CRT_EMI_buf, asarray(CRT_EMI[:,tran].copy(), float32))
        ###
        GNORM         = (C_LIGHT/(1.0e5*WIDTH*freq)) * GL  # GRID_LENGTH multiplied to gauss norm
        sys.stdout.write(' %2d' % tran)
        sys.stdout.flush()
        kernel_clear(queue, [GLOBAL,], [LOCAL,], RES_buf)

        # Upload NI[upper] and NB_NB[tran] values
        tmp  =  tmp_1 * Aul * (NI_ARRAY[:,lower]*gg-NI_ARRAY[:,upper]) / (freq*freq)  # [CELLS]

        if (0):
            tmp  = np.clip(tmp, -1.0e-12, 1.0e32)      # kernel still may have clamp on tau
            tmp[nonzero(abs(tmp)<1.0e-32)] = 1.0e-32   # nb_nb -> one must not divide by zero  $$$
        else:
            tmp = np.clip(tmp, 1.0e-25, 1.0e32)        # KILL ALL MASERS  $$$
            
        WRK[:,0]  = NI_ARRAY[:, upper]   # ni
        WRK[:,1]  = tmp                  # nb_nb
        cl.enqueue_copy(queue, NI_buf, WRK)
        # the next loop is 99% of the Simulate() routine run time
        offs  =  INI['offsets']  # default was 1, one ray per cell
        t000  =  time.time()

        if (0):
            print("  A_b     %12.4e" % Ab)
            print("  GL      %12.4e" % GL)
            print("  GN      %12.4e" % GNORM)
            print("  Aul     %12.4e" % Aul)
            print("  freq    %12.4e" % freq)
            print("  gg      %12.4e" % gg)
            print("  BGPHOT  %12.4e" % BGPHOT)
            print("  PACKETS %d"     % PACKETS)
            print("  BG      %12.4e" % Planck(freq, Tbg))


        SUM_DIRWEI = 0.0   # weight ~ cos(theta)/sum(cos(theta)) ... should sum to 1.0 for each side, 6.0 total
        
        for idir in range(NDIR):
            
            inner_loop = 4*offs*offs
            if (ONESHOT): inner_loop = 1    # for  OCTREE=4, OCTREE=40
            
            for ioff in range(inner_loop):  # 4 staggered initial positions over 2x2 cells -- if ONESHOT==0
                
                POS, DIR, LEADING  =  GetHealpixDirection(NSIDE, ioff, idir, NX, NY, NZ, offs, DOUBLE_POS) # < 0.001 seconds !
                
                # print("IDIR %3d/%3d   IOFF %2d/%2d   %7.4f %7.4f %7.4f" % (idir+1, NDIR, ioff, 4*offs*offs, DIR['x'], DIR['y'], DIR['z']))
                dirwei, ewei = 1.0, 1.0
                
                if (1):
                    # 2021-01-13 --- BGPHOT * cos(theta)/DIRWEI, DIRWEI = sum(cos(theta)) for each side separately
                    # there is no change at this point, except that DIRWEI is sum, not the average cos(theta)
                    # *AND* BG is computed without dependence on the PACKETS variable
                    if (LEADING in   [0, 1]):    
                        dirwei   =  fabs(DIR['x']) / DIRWEI[LEADING]   # cos(theta)/<cos(theta)>
                        ewei     =  float32(EWEI / abs(DIR['x']))      # (1/cosT) / <1/cosT> / NDIR ... not used !!
                    elif (LEADING in [2, 3]):  
                        dirwei   =  fabs(DIR['y']) / DIRWEI[LEADING]
                        ewei     =  float32(EWEI / abs(DIR['y']))
                    else:                     
                        dirwei   =  fabs(DIR['z']) / DIRWEI[LEADING]
                        ewei     =  float32(EWEI / abs(DIR['z']))                    
                    BG     = BGPHOT * dirwei
                    SUM_DIRWEI += dirwei
                    dirwei = 1.0        # this is the weight factor that goes to kernel... now not used
                else:
                    # kernel gets  WEI = dirwei/DIRWEI ==  cos(theta)/<cos(theta>)>, to weight BG
                    #   One needs a weighting according to the angle between the ray and the surface.
                    #   That could be taken care by the density of rays hitting a surface, if the rays 
                    #   were equidistant in 3d and therefore density of hits on the surface would get 
                    #   lower by ~1/cos(theta) for more obliques angles.
                    # *However*, rays are created equidistant on the leading edge, irrespective of cos(theta).
                    #    Therefore, we do need to include the weighting 1/cos(theta) because we have rays too densely for
                    #    oblique angles. On the other hand, simulated rays correspond exactly to the true number
                    #    of photons that should enter the model volume  => it is only a relative weighting
                    #    and <dirwei/DIRWEI> == 1.0   =>  PHOTONS ~ sum{   (BGPHOT/PACKETS) * (dirwei/DIRWEI)  }
                    # DIRWEI was computed for each of the six sides separately
                    if (LEADING in [0, 1]):    
                        dirwei   =  fabs(DIR['x']) / DIRWEI[LEADING]   # cos(theta)/<cos(theta)>
                        ewei     =  float32(EWEI / abs(DIR['x']))      # (1/cosT) / <1/cosT> / NDIR ... not used !!
                    elif (LEADING in [2, 3]):  
                        dirwei   =  fabs(DIR['y']) / DIRWEI[LEADING]
                        ewei     =  float32(EWEI / abs(DIR['y']))
                    else:                     
                        dirwei   =  fabs(DIR['z']) / DIRWEI[LEADING]
                        ewei     =  float32(EWEI / abs(DIR['z']))
                        
                        
                # print("         idir %3d  =>  dirwei %8.5f" % (idir, dirwei))
                               
                if (ncmp==1):
                    if (WITH_CRT):
                        niter = NRAY//GLOBAL
                        if (NRAY%GLOBAL!=0): niter += 1
                        for ibatch in range(niter):                                
                            kernel_sim(queue, [GLOBAL,], [LOCAL,],  
                            # 0             1          2        3        4    5   6      7     8  
                            ibatch*GLOBAL,  CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab, GNORM, None, APL,
                            # 9   10      11    12       13   14   15      16       17        
                            BG,   dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf,
                            # 18         19         
                            CRT_TAU_buf, CRT_EMI_buf)
                    else:
                        if (OCTREE<1):
                            niter = NRAY//GLOBAL
                            if (NRAY%GLOBAL!=0): niter += 1
                            # print("kernel_sim, niter=%d, NRAY=%d, GLOBAL=%d" % (niter, NRAY, GLOBAL))
                            for ibatch in range(niter):
                                kernel_sim(queue, [GLOBAL,], [LOCAL,],
                                # 0            1          2        3        4    5 
                                ibatch*GLOBAL, CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                                # 6    7       8    9   10      11    12       13   14   15      16       17      
                                GNORM, PL_buf, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf)
                        elif (OCTREE==1):
                            # OCTREE==1 does always calculate paths but we choose not to use those??
                            if (PLWEIGHT<1):
                                #                                       0          1        2        3    4  
                                kernel_sim(queue, [GLOBAL,], [LOCAL,],  CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                                # 5    6    7   8       9     10       11   12   13      14       15       
                                GNORM, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf, 
                                #  16       17       18       19      
                                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf)
                            else:
                                #                                      0       1          2        3        4    5  
                                kernel_sim(queue, [GLOBAL,], [LOCAL,], PL_buf, CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                                # 5    6    7   8       9     10       11   12   13      14       15       
                                GNORM, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf, 
                                #  16       17       18       19    
                                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf)                                
                        elif (OCTREE in [2,3]): # OCTREE=2, OCTREE=3
                            #                                       0       1          2        3        4    5   
                            kernel_sim(queue, [GLOBAL,], [LOCAL,],  PL_buf, CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab,
                            # 6    7    8   9       10    11       12   13   14      15       16        
                            GNORM, APL, BG, dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf, 
                            #  17       18       19       20        21        
                            LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)
                        elif (OCTREE in [4,5]):
                            niter = NRAY//NWG
                            if (NRAY%NWG!=0): niter += 1
                            for ibatch in range(niter):                                
                                # print("idir=%2d, kernel_sim %d/%d" %(idir, 1+ibatch, niter))
                                if (PLWEIGHT>0):
                                    kernel_sim(queue, [GLOBAL,], [LOCAL,],  ibatch*NWG, PL_buf, CLOUD_buf, 
                                    GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, BG, dirwei, ewei,
                                    LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf,
                                    LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)
                                else:
                                    kernel_sim(queue, [GLOBAL,], [LOCAL,],  ibatch*NWG, CLOUD_buf,
                                    GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, BG, dirwei, ewei,
                                    LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf,
                                    LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)
                                
                            # sys.exit()
                                
                        elif (OCTREE in [40,]):
                            # with option ONESHOT, kernel should loop over all ray position, not only every other
                            niter = NRAY//GLOBAL
                            if (NRAY%GLOBAL!=0): niter += 1
                            for ibatch in range(niter):
                                # print("kernel_sim batch %d/%d" %(1+ibatch, niter))
                                kernel_sim(queue, [GLOBAL,], [LOCAL,],  ibatch*GLOBAL, PL_buf, CLOUD_buf, 
                                GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, BG, dirwei, ewei,
                                LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf,
                                LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,  BUFFER_buf)
                            
                else:
                    if (OCTREE<1): #  GLOBAL >= NRAY
                        #                                       0          1        2        3    4   5      6   
                        kernel_hf(queue, [GLOBAL,], [LOCAL,],   CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, 
                        # 7  8       9     10       11   12   13      14       15    16    17      18      
                        BG,  dirwei, ewei, LEADING, POS, DIR, NI_buf, RES_buf, nchn, ncmp, HF_buf, NTRUE_buf,
                        PROFILE_buf)
                    else:
                        if (OCTREE==4):
                            niter = NRAY//NWG
                            if (NRAY%NWG!=0): niter += 1
                            for ibatch in range(niter):
                                kernel_hf(queue, [GLOBAL,], [LOCAL,],  ibatch*NWG, PL_buf, CLOUD_buf, 
                                GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, BG, dirwei, ewei,
                                LEADING, POS, DIR, NI_buf,  nchn, ncmp, HF_buf, 
                                RES_buf, NTRUE_buf, LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, BUFFER_buf)
                        else:
                            print("HFS + OCTREE%d not yet implemented" % OCTREE), sys.exit()
                        
                queue.finish()
        
        # end of loop over NDIR
        # print("--- tran=%2d   ncmp=%2d  %.3f seconds" % (tran, ncmp, time.time()-t000))
        # print("--- SUM_DIRWEI %8.5f" % SUM_DIRWEI)
        
        # print("**** DIRWEI AVERAGE VALUE %8.4f ***" % (dwsum/dwcount))
        # average weight was == 1.000 but this reduced Tex ... the weighting is not ok?
        
        # @@ for plot_pl.py
        if (0):  # debugging ... see that Update has the same PL as Path
            print("SAVING PL.dat")
            PL.tofile('PL.dat')
            cl.enqueue_copy(queue, PL, PL_buf)
            PL.tofile('PL_zero.dat')
            sys.exit()


        # PLWEIGHT
        # - OCTREE=0 -- PLWEIGHT can be 1 (for testing). Update kernel can take PL_buf but only for testing
        #   and we do not use PL here.
        # - OCTREE>0, we assume PL is calculated and it is not used only for OCTREE=4, if INI['plweight']==0
        # post weighting
        if (WITH_ALI>0):
            cl.enqueue_copy(queue, WRK, RES_buf)       # pull both SIJ and ESC, RES[CELLS,2] => WRK[CELLS,2]
            if (PLWEIGHT==0):    # assume PLWEIGHT implies OCTREE==0
                #  Cartesian grid -- no PL weighting, no cell-volume weighting
                SIJ_ARRAY[:, tran]                    =  WRK[:,0]  
                ESC_ARRAY[:, tran]                    =  WRK[:,1]
            else:
                if (OCTREE==0):   # regular Cartesian but PLWEIGHT=1 ==> use PL weighting
                    SIJ_ARRAY[:, tran]                =  WRK[:,0] * APL/PL[:]
                    ESC_ARRAY[:, tran]                =  WRK[:,1] * APL/PL[:]
                elif (OCTREE in [1,2]):   #  OCTREE=1,2, weight with  (APL/PL) * f(level)
                    a, b  =  0, LCELLS[0]
                    SIJ_ARRAY[a:b, tran]              =  WRK[a:b, 0] * (APL/PL[a:b])
                    ESC_ARRAY[a:b, tran]              =  WRK[a:b, 1] * (APL/PL[a:b])
                    for l in range(1, OTL):
                        a, b                          =  OFF[l], OFF[l]+LCELLS[l]                    
                        if (OCTREE==1):             k =  8.0**l  # OCTREE1   PL should be  APL/8^l
                        else:                       k =  2.0**l  # OCTREE2   PL should be  APL/2^l
                        SIJ_ARRAY[a:b, tran]          =  WRK[a:b, 0] * (1.0/k)  * (APL/PL[a:b])
                        ESC_ARRAY[a:b, tran]          =  WRK[a:b, 1] * (1.0/k)  * (APL/PL[a:b])
                else:  # OCTREE 3,4,5,40   --- assuming  it should be  PL[] == APL/2^l ;
                    if (PLWEIGHT):
                        a, b                          =  0, LCELLS[0]
                        SIJ_ARRAY[a:b, tran]          =  WRK[a:b, 0] * APL/PL[a:b] 
                        ESC_ARRAY[a:b, tran]          =  WRK[a:b, 1] * APL/PL[a:b]
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            SIJ_ARRAY[a:b, tran]      =  WRK[a:b, 0] * (APL/2.0**l) /  PL[a:b]
                            ESC_ARRAY[a:b, tran]      =  WRK[a:b, 1] * (APL/2.0**l) /  PL[a:b]
                    else:  # if we rely on PL being correct ~ 0.5^level
                        # above scaling  APL/2**l/PL translates to 1.0 
                        a, b                          =  0, LCELLS[0]
                        SIJ_ARRAY[a:b, tran]          =  WRK[a:b, 0]
                        ESC_ARRAY[a:b, tran]          =  WRK[a:b, 1]
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            SIJ_ARRAY[a:b, tran]      =  WRK[a:b, 0] #### / (2.0**l)
                            ESC_ARRAY[a:b, tran]      =  WRK[a:b, 1] #### / (2.0**l)
        else:  # no ALI, SIJ only
            WRK.shape = (2, CELLS)         # trickery... we use only first CELLS elements of WRK
            cl.enqueue_copy(queue, WRK[0,:], RES_buf)    # SIJ only .... RES is only RES[CELLS]
            if (PLWEIGHT==0):              # PLWEIGHT==0 implies OCTREE=0, Cartesian grid without PL weighting
                SIJ_ARRAY[:, tran]                    =  WRK[0,:]
            else:
                if (OCTREE==0):            # Cartesian grid with PL weihgting
                    SIJ_ARRAY[:, tran]                =  WRK[0,:]  * APL/PL[:]
                elif (OCTREE in [1,2]):    #  OCTREE=1,2, weight with  (APL/PL) * f(level)
                    a, b                              =  0, LCELLS[0]
                    SIJ_ARRAY[a:b, tran]              =  WRK[0, a:b] * APL/PL[a:b] 
                    for l in range(1, OTL):
                        a, b                          =  OFF[l], OFF[l]+LCELLS[l]                    
                        if (OCTREE==1):            k  =  8.0**l    # OCTREE1   PL should be  APL/8^l
                        else:                      k  =  2.0**l    # OCTREE2   PL should be  APL/2^l
                        SIJ_ARRAY[a:b, tran]          =  WRK[0,a:b] * (APL/k) / PL[a:b]
                else:  # OCTREE 3,4,5,40   weight 2**-l  --- except OCTREE=3 is not exact
                    if (PLWEIGHT):  # include APL/PL weighting
                        # print("*** PLWEIGHT***")
                        a, b                          =  0, LCELLS[0]
                        SIJ_ARRAY[a:b, tran]          =  WRK[0, a:b] * APL/PL[a:b] 
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]                    
                            SIJ_ARRAY[a:b, tran]      =  WRK[0, a:b] * (APL/2.0**l) /  PL[a:b]
                    else:  # if we rely on PL being correct  - only volume weighting
                        a, b                          =  0, LCELLS[0]
                        SIJ_ARRAY[a:b, tran]          =  WRK[0, a:b]
                        m = nonzero(RHO[a:b]>0.0)
                        yyy  =  APL/PL[a:b]
                        print("*** L=0, OMIT  APL/PL = %.4f +- %.4f" % (mean(yyy[m]), std(yyy[m])))
                        for l in range(1, OTL):
                            a, b                      =  OFF[l], OFF[l]+LCELLS[l]      
                            m  =  nonzero(RHO[a:b]>0.0)
                            yyy  =  APL / (2.0**l) / PL[a:b]
                            print("*** L=%d, OMIT  APL/PL = %.4f +- %.4f" % (l, mean(yyy[m]), std(yyy[m])))
                            SIJ_ARRAY[a:b, tran]      =  WRK[0, a:b] ### / (2.0**l)
            WRK.shape = (CELLS, 2)
                        
                        
                            
        m = nonzero(RHO>0.0)
        if (WITH_ALI>0):
            print(" TRANSITION %2d  <SIJ> %12.5e   <ESC> %12.5e" % 
            (tran, mean(SIJ_ARRAY[m[0],tran]), mean(ESC_ARRAY[m[0],tran])))
            
        if (1):
            m1  =  nonzero(~isfinite(SIJ_ARRAY[m[0],0]))
            # m1  =  ([6083,],)
            # m1  =  ([5083,],)
            if (len(m1[0])>0):
                print("\nSIJ NOT FINITE:", len(m1[0]))
                print('SIJ  ', SIJ_ARRAY[m[0],0][m1[0]])
                print('RHO  ',       RHO[m[0]][m1[0]])
                print('TKIN ',      TKIN[m[0]][m1[0]])
                print('ABU  ',       ABU[m[0]][m1[0]])
                if (len(m1[0])>0): sys.exit()
            
        if (tran==-1):
            clf()
            ax = axes()
            m = nonzero(RHO>0.0)
            print("CELLS %d" % len(m[0]))
            print("SIJ_ARRAY[%d] " % tran,  percentile(SIJ_ARRAY[m[0],tran], (1.0, 50.0, 99.0)))
            plot(SIJ_ARRAY[m[0],tran], 'r.')
            text(0.2, 0.5, r'$%.4e, \/ \/ \/ \sigma(r) = %.6f$' % (mean(SIJ_ARRAY[m[0],tran]), std(SIJ_ARRAY[m[0],tran])/mean(SIJ_ARRAY[m[0],tran])), transform=ax.transAxes, color='r', size=15)
            if (WITH_ALI>0):
                print("ESC_ARRAY[%d] " % tran,  percentile(ESC_ARRAY[m[0],tran], (1.0, 50.0, 99.0)))
                plot(ESC_ARRAY[m[0],tran], 'b.')
                text(0.2, 0.4, r'$%.4e, \/ \/ \/ \sigma(r) = %.6f$' % (mean(ESC_ARRAY[m[0],tran]), std(ESC_ARRAY[m[0],tran])/mean(ESC_ARRAY[m[0],tran])), transform=ax.transAxes, color='b', size=15)
            title("SIJ and ESC directly from kernel")
            show()
            sys.exit()

        if (0):
            subplot(221)
            x = PL[0:NX*NY*NZ].reshape(NX,NY,NZ)
            a, b = percentile(ravel(x), (10, 99.9))
            imshow(x[NX//2,:,:], vmin=a, vmax=b)
            title("PL, root level")
            colorbar()
            subplot(222)
            x = SIJ_ARRAY[0:NX*NY*NZ,tran].reshape(NX,NY,NZ)
            a, b = percentile(ravel(x), (10, 99.9))
            imshow(x[NX//2,:,:], vmin=2.0e-12, vmax=2.4e-12)
            title("SIJ, root level")
            colorbar()
            subplot(223)
            x = ESC_ARRAY[0:NX*NY*NZ,tran].reshape(NX,NY,NZ)
            a, b = percentile(ravel(x), (10, 99))
            imshow(x[NX//2,:,:], vmin=a, vmax=b)
            title("ESC, root level")
            colorbar()
            subplot(224)
            x = RHO[0:NX*NY*NZ].reshape(NX,NY,NZ)
            a, b = percentile(ravel(x), (10, 99))
            imshow(x[NX//2,:,:], vmin=a, vmax=b)
            title("RHO, root level")
            colorbar()
            show()
            sys.exit()
            
        if (0):
            print("")
            print("SIJ_ARRAY[%d] " % tran,  percentile(SIJ_ARRAY[:,tran], (1.0, 50.0, 99.0)))
            print("ESC_ARRAY[%d] " % tran,  percentile(ESC_ARRAY[:,tran], (1.0, 50.0, 99.0)))
        if (COOLING==2):
            cl.enqueue_copy(queue, LEV_COOL, COOL_buf)
            SUM_COOL[:]   +=   LEV_COOL[:] * hf[tran]    # per cm3
        if (0):
            print("       tran = %3d  = %2d - %2d  => <SIJ> = %.3e   <ESC> = %.3e" % 
            (tran, upper, lower, mean(WRK[:,0]), mean(WRK[:,1])))
    sys.stdout.write('\n')



    if (0): # check SIJ vs cell
        close(1)
        figure("NoFocus")
        a  =      SIJ_ARRAY[0:LCELLS[0],               0]
        b  =  8.0*SIJ_ARRAY[OFF[1]:(OFF[1]+LCELLS[1]), 0]
        plot(a, 'b+')
        plot(b, 'r+')
        title("2/1 = %.3f" % (mean(b[nonzero(b>0.0)])/mean(a)))
        plot(    SIJ_ARRAY[0:LCELLS[0]              , 1], 'bx')
        plot(8.0*SIJ_ARRAY[OFF[1]:(OFF[1]+LCELLS[1]), 1], 'rx')
        show()
        sys.exit()
            
        
    
    
    if (0):
        print("------------------------------------------------------------------------------------------")
        for i in range(6):
            for j in range(7):
                sys.stdout.write(" %12.5e" % SIJ_ARRAY[i,j])
            sys.stdout.write('\n')
        print("------------------------------------------------------------------------------------------")
        for i in range(6):
            for j in range(7):
                sys.stdout.write(" %12.5e" % ESC_ARRAY[i,j])
            sys.stdout.write('\n')
        print("------------------------------------------------------------------------------------------")
        sys.exit()
        

            
            
    # <--- for tran ---
    if (COOLING==2):
        print("BRUTE COOLING: %10.3e" % (sum(SUM_COOL)/CELLS))
        fpb = open('brute.cooling', 'wb')
        asarray(SUM_COOL, float32).tofile(fpb)
        fpb.close()
        SUM_COOL = []
        LEV_COOL = []
        

        
            
def Cooling():
    """
    cell emits             n_u*Aul  photons / cm3
    escaping photons       ESC
    all absorbed in        SIJ
       => enough information to calculate net cooling
    COOL  =  2*ESC - SIJ*NI[lower] - n[upper]*Aul
    """
    global CELLS, TRANSITIONS, MOL, INI, SIJ_ARRAY, ESC_ARRAY, NI_ARRAY
    COOL  =  zeros(CELLS, float32)
    Aul   =  MOL.A
    hf    =  MOL.F*PLANCK
    AVE   =  0.0
    U, L  =  zeros(TRANSITIONS, int32), zeros(TRANSITIONS, int32)
    for tr in range(TRANSITIONS):
        u, l         =  MOL.T2L(tr)
        U[tr], L[tr] =  u, l   
    for icell in range(CELLS):
        # COOL[icell] +=  sum(hf[:] * (  NI_ARRAY[icell,[U[:]]*Aul[:] - SIJ_ARRAY[icell,:]*NI[L[:]] ))
        COOL[icell] +=  sum(hf[:] * (  ESC_ARRAY[icell,:]/VOLUME - SIJ_ARRAY[icell,:]*NI_ARRAY[icell, L[:]] ))
    print("COOLING: AVERAGE %12.4e" % (mean(COOL)))
    fp = open('cooling.bin', 'wb')
    asarray(COOL, float32).tofile(fp)
    fp.close()

    
    
    
def Solve(CELLS, MOL, INI, LEVELS, TKIN, RHO, ABU, ESC_ARRAY):
    NI_LIMIT  =  1.0e-28
    CHECK     = min([INI['uppermost']+1, LEVELS])  # check this many lowest energylevels
    cab       = ones(10, float32)                  # scalar abundances of different collision partners
    for i in range(PARTNERS):
        cab[i] = MOL.CABU[i]                       # default values == 1.0
    # possible abundance file for abundances of all collisional partners
    CABFP = None
    if (len(INI['cabfile'])>0): # we have a file with abundances of each collisional partner
        CABFP = open(INI['cabfile'], 'rb')
        tmp   = np.fromfile(CABFP, int32, 4)
        if ((tmp[0]!=NX)|(tmp[1]!=NY)|(tmp[2]!=NZ)|(tmp[3]!=PARTNERS)):
            print("*** ERROR: CABFILE has dimensions %d x %d x %d, for %d partners" % (tmp[0], tmp[1], tmp[2], tmp[3]))
            sys.exit()            
    MATRIX    =  np.zeros((LEVELS, LEVELS), float32)
    VECTOR    =  np.zeros(LEVELS, float32)
    COMATRIX  =  []
    ave_max_change, global_max_change = 0.0, 0.0
    
    if (INI['constant_tkin']): # Tkin same for all cells => precalculate collisional part
        print("Tkin assumed to be constant !")
        constant_tkin = True
    else:
        constant_tkin = False
    if (constant_tkin):
        if (CABFP):
            print("Cannot have variable CAB if Tkin is assumed to be constant")
        COMATRIX = zeros((LEVELS, LEVELS), float32)
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

    uu, ll  = zeros(TRANSITIONS, int32), zeros(TRANSITIONS, int32)
    for t in range(TRANSITIONS):
        u, l  = MOL.T2L(t)
        uu[t], ll[t] = u, l
    # all_cab = fromfile(CABFP, float32, PARTNERS*CELLS).reshape(CELLS, PARTNERS)
    for icell in range(CELLS):
        if (icell%(CELLS//20)==0):
            print("  solve   %7d / %7d  .... %3.0f%%" % (icell, CELLS, 100.0*icell/float(CELLS)))
        tkin, rho, chi  =  TKIN[icell], RHO[icell], ABU[icell]
        if (rho<1.0e-2):  continue
        if (chi<1.0e-20): continue
        
        #print("rho = %.3e" % rho)
        #rho = 1.0e-5
        
        if (constant_tkin):
            MATRIX[:,:] = COMATRIX[:,:] * rho 
        else:
            if (CABFP):
                cab = np.fromfile(CABFP, float32, PARTNERS)   # abundances for current cell, cab[PARTNERS]
            if (PARTNERS==1):
                for iii in range(LEVELS):
                    for jjj in range(LEVELS):
                        if (iii==jjj):
                            MATRIX[iii,jjj] = 0.0
                        else:
                            gamma = MOL.C(iii,jjj,tkin,0)  # rate iii -> jjj
                            MATRIX[jjj, iii] = gamma*rho
            else:
                for iii in range(LEVELS):
                    for jjj in range(LEVELS):
                        if (iii==jjj):
                            MATRIX[iii,jjj] = 0.0
                        else:
                            gamma = 0.0
                            for ip in range(PARTNERS):
                                gamma += cab[ip] * MOL.C(iii, jjj, tkin, ip)
                            MATRIX[jjj, iii] = gamma*rho
        if (len(ESC_ARRAY)>1):
            MATRIX[ll,uu]    +=  ESC_ARRAY[icell, :] / (VOLUME*NI_ARRAY[icell, uu])
        else:
            for t in range(TRANSITIONS):
                u,l           =  MOL.T2L(t)
                MATRIX[l,u]  +=  MOL.A[t]

        #   X[l,u] = Aul + Bul*I       =  Aul + Blu*gl/gu*I = Aul + Slu/GG
        #   X[u,l] = Blu = Bul*gu/gl
        
        MATRIX[uu, ll]    +=  SIJ_ARRAY[icell, :] /  VOLUME
        MATRIX[ll, uu]    +=  SIJ_ARRAY[icell, :] / (VOLUME * MOL.GG[:])

        for u in range(LEVELS-1): # diagonal = -sum of the column
            tmp            = sum(MATRIX[:,u])    # MATRIX[i,i] was still == 0
            MATRIX[u,u]    = -tmp
            
        MATRIX[LEVELS-1, :]   =  -MATRIX[0,0]    # replace last equation = last row
        VECTOR[:]             =   0.0
        VECTOR[LEVELS-1]      =  -(rho*chi) * MATRIX[0,0]  # ???

        VECTOR  =  np.linalg.solve(MATRIX, VECTOR)        
        VECTOR  =  np.clip(VECTOR, NI_LIMIT, 1e99)
        VECTOR *=  rho*chi / sum(VECTOR)        
        
        if (0):
            print("F= %12.4e  Gu = %.3f  Gl = %3f" % (MOL.F[0], MOL.G[1], MOL.G[0]))
            tex  =  -H_K*MOL.F[0] / log(MOL.G[0]*VECTOR[1]/(MOL.G[1]*VECTOR[0]))
            print("Tex(1-0) = %.4f" % tex)
            print("CO_ARRAY")
            for j in range(LEVELS):
                for i in range(LEVELS):
                    sys.stdout.write('%10.4e ' % (MATRIX[j,i]))
                sys.stdout.write('   %10.4e\n' % (VECTOR[j]))
            print('')
            print("VECTOR")
            for j in range(LEVELS):
                sys.stdout.write(' %10.2e' % VECTOR[j])
            sys.stdout.write('\n')
            print("")
            print("SIJ")
            for j in range(TRANSITIONS):
                sys.stdout.write(' %10.2e' % SIJ_ARRAY[icell,j])
            sys.stdout.write('\n')
            if (WITH_ALI):
                print("ESC")
                for j in range(TRANSITIONS):
                    sys.stdout.write(' %10.2e' % ESC_ARRAY[icell,j])
                sys.stdout.write('\n')
            sys.exit()
            
        max_relative_change =  max(abs((NI_ARRAY[icell, 0:CHECK]-VECTOR[0:CHECK])/(NI_ARRAY[icell, 0:CHECK])))
        NI_ARRAY[icell,:]   =  VECTOR        
        ave_max_change     +=  max_relative_change
        global_max_change   =  max([global_max_change, max_relative_change])    
    # <--- for icell
    ave_max_change /= CELLS
    print("     AVE %10.3e    MAX %10.3e" % (ave_max_change, global_max_change))
    return ave_max_change
    



def SolveCL():
    """
    Solve equilibrium equations on the device. We do this is batches, perhaps 10000 cells
    at a time => could be up to GB of device memory. 
    Note:
        In case of octree, SIJ and ESC values must be scaled by 2^level
    """
    global NI_buf, CELLS, queue, kernel_solve, RES_buf, VOLUME, CELLS, LEVELS, TRANSITIONS, MOL
    global RHO, TKIN, ABU
    global MOL_A_buf, MOL_UL_buf, MOL_E_buf, MOL_G_buf
    global MOL_TKIN_buf, MOL_CUL_buf, MOL_C_buf, MOL_CABU_buf   
    global SOL_WRK_buf, SOL_RHO_buf, SOL_TKIN_buf, SOL_ABU_buf, SOL_NI_buf
    global SOL_SIJ_buf, SOL_ESC_buf, OCTREE, BATCH
    # TKIN
    for i in range(1, PARTNERS):
        if (len(MOL.TKIN[i])!=NTKIN):
            print("SolveCL assumes the same number of Tkin for each collisional partner!!"), sys.exit()
        
    CHECK = min([INI['uppermost']+1, LEVELS])  # check this many lowest energylevels
    GLOBAL_SOLVE = IRound(BATCH, LOCAL)
    tmp   = zeros((BATCH, 2, TRANSITIONS), float32)
    res   = zeros((BATCH, LEVELS), float32)
    ave_max_change     = 0.0
    global_max_change  = 0.0
    
    if (OCTREE>0):
        # follow_ind = 6083
        follow_ind =   -1
        for ilevel in range(OTL):
            # print("OCTREE LEVEL %d" % ilevel)
            debug_i   = -1
            for ibatch in range(LCELLS[ilevel]//BATCH+1):
                a     = OFF[ilevel] + ibatch*BATCH
                b     = min([a+BATCH, OFF[ilevel]+LCELLS[ilevel]])
                batch = b-a
                if (batch<1): break  #   division CELLS//BATCH went even...
                if (ibatch%5==-1):
                    print(" SolveCL, batch %5d,  [%7d, %7d[, batch %4d, BATCH %4d - LCELLS %d" % 
                    (ibatch, a, b, batch, BATCH, LCELLS[ilevel]))
                debug_i = -1
                if (0):
                    if ((follow_ind>=a)&(follow_ind<b)):
                        debug_i = follow_ind-a  # cell of interest the debug-i:th in the current batch
                        print("--------------------------------------------------------------------------------") 
                        print("*** BATCH with COI %d  ... entry %d in batch=%d***" % (follow_ind, follow_ind-a, batch))
                        print("    RHO  %12.4e" %  RHO[follow_ind])
                        print("    TKIN %12.4e" % TKIN[follow_ind])
                        print("    ABU  %12.4e" %  ABU[follow_ind])
                        print("    NI   ",    NI_ARRAY[follow_ind,:])
                        print("    SIJ  ",   SIJ_ARRAY[follow_ind,:])
                        if (WITH_ALI>0):
                            print("    ESC  ",   ESC_ARRAY[follow_ind,:])
                        tex  =  -H_K*MOL.F[0] / log(MOL.G[0]*NI_ARRAY[follow_ind,1]/(MOL.G[1]*NI_ARRAY[follow_ind,0]))
                        tmp_nbnb  =  NI_ARRAY[follow_ind, 0]*MOL.GG[0]-NI_ARRAY[follow_ind, 1]
                        tmp_nbnb *=  (C_LIGHT*C_LIGHT/(8.0*pi)) * MOL.A[0]  /  (MOL.F[0]**2.0)
                        print("    TEX10= %+8.4f,   nb_nb= %.3e,   ni= %10.3e %10.3e %10.3e" % \
                        (tex, tmp_nbnb, NI_ARRAY[follow_ind,0], NI_ARRAY[follow_ind,1], NI_ARRAY[follow_ind,2]))
                        print("--------------------------------------------------------------------------------")
                    else:
                        debug_i = -1
                # copy RHO, TKIN, ABU
                cl.enqueue_copy(queue, SOL_RHO_buf,  RHO[a:b].copy())  # without copy() "ndarray is not contiguous"
                cl.enqueue_copy(queue, SOL_TKIN_buf, TKIN[a:b].copy())
                cl.enqueue_copy(queue, SOL_ABU_buf,  ABU[a:b].copy())
                cl.enqueue_copy(queue, SOL_NI_buf,   NI_ARRAY[a:b,:].copy())   # PL[CELLS] ~ NI[BATCH, LEVELS]
                cl.enqueue_copy(queue, SOL_SIJ_buf,  SIJ_ARRAY[a:b,:].copy())
                if (WITH_ALI>0):
                    cl.enqueue_copy(queue, SOL_ESC_buf,  ESC_ARRAY[a:b,:].copy())
                # solve
                kernel_solve(queue, [GLOBAL_SOLVE,], [LOCAL,], ilevel, batch, 
                MOL_A_buf, MOL_UL_buf,  MOL_E_buf, MOL_G_buf, PARTNERS, NTKIN, NCUL,   
                MOL_TKIN_buf, MOL_CUL_buf,  MOL_C_buf, MOL_CABU_buf,
                SOL_RHO_buf, SOL_TKIN_buf, SOL_ABU_buf,  SOL_NI_buf, SOL_SIJ_buf, SOL_ESC_buf,  
                RES_buf, SOL_WRK_buf, debug_i)   # was follow_ind, should be debug_i ???
                cl.enqueue_copy(queue, res, RES_buf)
                if (INI['dnlimit']>0.0): # Limit maximum change in NI
                    # if change is above dnlimit, use average of old and new level populations (plus damping below)
                    m            =  nonzero(abs((res[0:batch,0]-NI_ARRAY[a:b,0])/NI_ARRAY[a:b,0])>INI['dnlimit'])
                    if (len(m[0])>0):
                        print("*** DNLIMIT APPLIED TO %d CELLS, dnlimit=%.3f" % (len(m[0]), INI['dnlimit']))
                        res[m[0],:]  =  0.5*res[m[0],:] + 0.5*NI_ARRAY[a+m[0],:]
                if (INI['damping']>0.0): #  Dampen change
                    # print("*** DAMPING APPLIED WITH damping=%.3f" % (INI['damping']))
                    res[0:batch,:] =  INI['damping']*NI_ARRAY[a:b,:] + (1.0-INI['damping'])*res[0:batch,:]
                # delta = for each cell, the maximum level populations change among levels 0:CHECK
                delta             =  np.max((res[0:batch,0:CHECK] - NI_ARRAY[a:b,0:CHECK]) / (1.0e-22+NI_ARRAY[a:b,0:CHECK]), axis=1)
                global_max_change =  max([global_max_change, max(delta)])
                ave_max_change   +=  sum(delta)
                NI_ARRAY[a:b,:]   =  res[0:batch,:]
                ####
                if (0):
                    if (debug_i>=0):
                        print("================================================================================")
                        print("*** BATCH with COI %d ***" % follow_ind)
                        print("    RHO  %12.4e" %  RHO[follow_ind])
                        print("    TKIN %12.4e" % TKIN[follow_ind])
                        print("    ABU  %12.4e" %  ABU[follow_ind])
                        print("    NI   ",    NI_ARRAY[follow_ind,:])
                        print("    SIJ  ",   SIJ_ARRAY[follow_ind,:])
                        print("    CLOUD ", CLOUD[follow_ind])
                        if (WITH_ALI>0):
                            print("    ESC  ",   ESC_ARRAY[follow_ind,:])
                        tex   =  -H_K*MOL.F[0] / log(MOL.G[0]*NI_ARRAY[follow_ind,1]/(MOL.G[1]*NI_ARRAY[follow_ind,0]))
                        tmp_nbnb  =  NI_ARRAY[follow_ind, 0]*MOL.GG[0]-NI_ARRAY[follow_ind, 1]
                        tmp_nbnb *=  (C_LIGHT*C_LIGHT/(8.0*pi)) * MOL.A[0]  /  (MOL.F[0]**2.0)
                        print("    TEX10= %+8.4f,   nb_nb= %.3e,   ni= %10.3e %10.3e %10.3e" % \
                        (tex, tmp_nbnb, NI_ARRAY[follow_ind,0], NI_ARRAY[follow_ind,1], NI_ARRAY[follow_ind,2]))
                        print("================================================================================")
                    else:
                        debug_i = -1
                
            if (0):   # SIJ LOOK OK ACROSS LEVELS !!!
                a, b =  OFF[ilevel], OFF[ilevel]+LCELLS[ilevel]
                subplot(2,2,1+ilevel)
                plot(SIJ_ARRAY[a:b], 'k.')
                plot(ESC_ARRAY[a:b], 'r.')
                subplot(2,2,3+ilevel)
                plot(NI_ARRAY[a:b,1]/NI_ARRAY[a:b,0], 'k.')            
        #show()
        #sys.exit()
            
            
            
    else:
        
        for ibatch in range(CELLS//BATCH+1):
            a     = ibatch*BATCH
            b     = min([a+BATCH, CELLS])
            batch = b-a
            if (batch<1): break  #   division CELLS//BATCH went even...
            #print(" SolveCL, batch %5d,  [%7d, %7d[,  %4d cells, BATCH %4d" % (ibatch, a, b, batch, BATCH))
            # copy RHO, TKIN, ABU
            cl.enqueue_copy(queue, SOL_RHO_buf,  RHO[a:b].copy())  # without copy() "ndarray is not contiguous"
            cl.enqueue_copy(queue, SOL_TKIN_buf, TKIN[a:b].copy())
            cl.enqueue_copy(queue, SOL_ABU_buf,  ABU[a:b].copy())
            cl.enqueue_copy(queue, SOL_NI_buf,   NI_ARRAY[a:b,:].copy())   # PL[CELLS] ~ NI[BATCH, LEVELS]
            cl.enqueue_copy(queue, SOL_SIJ_buf,  SIJ_ARRAY[a:b,:].copy())
            if (WITH_ALI>0):
                cl.enqueue_copy(queue, SOL_ESC_buf,  ESC_ARRAY[a:b,:].copy())
            # solve
            kernel_solve(queue, [GLOBAL_SOLVE,], [LOCAL,], batch, 
            MOL_A_buf, MOL_UL_buf,  MOL_E_buf, MOL_G_buf,        PARTNERS, NTKIN, NCUL,   
            MOL_TKIN_buf, MOL_CUL_buf,  MOL_C_buf, MOL_CABU_buf,
            SOL_RHO_buf, SOL_TKIN_buf, SOL_ABU_buf,  SOL_NI_buf, SOL_SIJ_buf, SOL_ESC_buf,  RES_buf, SOL_WRK_buf, -1)
            cl.enqueue_copy(queue, res, RES_buf)
            # delta = for each cell, the maximum level populations change amog levels 0:CHECK
            delta             =  np.max((res[0:batch,0:CHECK] - NI_ARRAY[a:b,0:CHECK]) / NI_ARRAY[a:b,0:CHECK], axis=1)
            global_max_change =  max([global_max_change, max(delta)])
            ave_max_change   +=  sum(delta)
            NI_ARRAY[a:b,:]   =  res[0:batch]
    ave_max_change /= CELLS
    print("      SolveCL    AVE %10.3e    MAX %10.3e" % (ave_max_change, global_max_change))
    
    if (1):
        mbad = nonzero(~isfinite(sum(SIJ_ARRAY, axis=1)))
        print('      *** SIJ NOT FINITE: %d' % len(mbad[0]))
        if (WITH_ALI):
            mbad = nonzero(~isfinite(sum(ESC_ARRAY, axis=1)))
            print('      *** ESC NOT FINITE: %d' % len(mbad[0]))
        mbad = nonzero(~isfinite(sum(NI_ARRAY,  axis=1)))
        print('      *** NI  NOT FINITE: %d' % len(mbad[0]))
        for i in mbad[0]:
            NI_ARRAY[i,:] =  LTE_10_pop *RHO[i]*ABU[i]
    return ave_max_change

    


def WriteSpectra(INI, u, l):
    global MOL, program, queue, WIDTH, LOCAL, NI_ARRAY, WRK, NI_buf, HFS, CHANNELS, HFS
    global NTRUE_buf, STAU_buf, NI_buf, CLOUD_buf, GAU_buf, PROFILE_buf
    tmp_1       =  C_LIGHT*C_LIGHT/(8.0*pi)
    tran        =  MOL.L2T(u, l)
    if (tran<0):
        print("*** ERROR:  WriteSpectra  %2d -> %2d not valid transition" % (u, l))
        return
    if (HFS):
        ncmp    =  BAND[tran].N
        nchn    =  BAND[tran].Channels()
        print("     .... WriteSpectra, tran=%d, %d->%d: %d components, %d channels" % (tran, u, l, ncmp, nchn))
    else:
        nchn    =  CHANNELS     #  it is the original INI['channels']
        ncmp    =  1
    Aul         =  MOL.A[tran]
    freq        =  MOL.F[tran]
    gg          =  MOL.G[u]/MOL.G[l]
    GNORM       =  (C_LIGHT/(1.0e5*WIDTH*freq))    #  GRID_LENGTH **NOT** multiplied in    
    int2temp    =  C_LIGHT*C_LIGHT/(2.0*BOLTZMANN*freq*freq)
    BG          =  int2temp * Planck(freq, INI['Tbg'])
    NRA, NDE    =  INI['points']
    DE          =  0.0    
    GLOBAL      =  IRound(NRA, LOCAL)    
    ##NTRUE       =  zeros(NRA*INI['channels'], float32)
    STEP        =  INI['grid'] / INI['angle']
    emissivity  =  (PLANCK/(4.0*pi))*freq*Aul*int2temp    
    direction   =  cl.cltypes.make_float2()
    direction['x'], direction['y'] = INI['direction']                   # theta, phi = observer direction
    centre       =  cl.cltypes.make_float3()
    centre['x'],  centre['y'], centre['z'] =  0.5*NX, 0.5*NY, 0.5*NZ    # map centre in root grid coordinates
    if (isfinite(sum(INI['map_centre']))):  
        centre['x']  =   INI['map_centre'][0]
        centre['y']  =   INI['map_centre'][1]
        centre['z']  =   INI['map_centre'][2]
        print("MAP CENTRE: ", INI['map_centre'])
 
    if (HFS): # note -- GAU is for CHANNELS channels = maximum over all bands!!
        for i in range(ncmp):
            HF[i]['x']  =  round(BAND[tran].VELOCITY[i]/WIDTH) # offset in channels (from centre of the spectrum)
            HF[i]['y']  =  BAND[tran].WEIGHT[i]
            print("       offset  %5.2f channels, weight %5.3f" % (HF[i]['x'], HF[i]['y']))
        HF[0:ncmp]['y']  /= sum(HF[0:ncmp]['y'])
        cl.enqueue_copy(queue, HF_buf, HF)

    if (WITH_CRT):
        TMP[:] = CRT_EMI[:, tran] * H_K * ((C_LIGHT/MOL.F[tran])**2.0)*C_LIGHT/(1.0e5*WIDTH*8.0*pi)
        cl.enqueue_copy(queue, CRT_EMI_buf, TMP)
        cl.enqueue_copy(queue, CRT_TAU_buf, asarray(CRT_TAU[:,tran].copy(), float32))

    if (WITH_CRT):
        kernel_spe  = program.Spectra        
        #                                 0     1     2     3           4                  5
        kernel_spe.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
        # 6         7         8           9           10          11    12    13    14    15                
        np.float32, np.int32, np.float32, np.float32, np.float32, None, None, None, None, cl.cltypes.float3])
    else:
        if (OCTREE):
            kernel_spe  = program.Spectra        
            #                                 0     1     2     3           4                  5
            kernel_spe.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
            # 6         7         8           9           10          11    12  
            np.float32, np.int32, np.float32, np.float32, np.float32, None, None,
            # 13   14     15    16   17        18                 
            None,  None, None, None, np.int32, cl.cltypes.float3])
        else:
            kernel_spe  = program.Spectra        
            #                                 0     1     2     3           4                  5
            kernel_spe.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
            # 6         7         8           9           10          11    12    14        15                
            np.float32, np.int32, np.float32, np.float32, np.float32, None, None, np.int32, cl.cltypes.float3])
        
    if (HFS):
        # Same kernel used with both OCTREE==0 and OCTREE==4, argument list differs
        if (OCTREE==0):
            kernel_spe_hf  = program.SpectraHF
            #                                    0     1     2     3           4                  5      
            #                                    CLOUD GAU   LIM   GN          D                  NI     
            kernel_spe_hf.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
            #  6        7         8           9           10          11      12     
            #  DE       NRA       STEP        BG          emis        NTRUE   SUM_TAU
            np.float32, np.int32, np.float32, np.float32, np.float32, None,   None,
            # 13      14        15     16       17                 
            # NCHN    NCOMP     HF     PROFILE  MAP_CENTRE
            np.int32, np.int32, None,  None,    cl.cltypes.float3 ])
        elif (OCTREE==4):
            kernel_spe_hf  = program.SpectraHF
            #                                    0     1     2     3           4                  5      
            #                                    CLOUD GAU   LIM   GN          D                  NI     
            kernel_spe_hf.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
            #  6        7         8           9           10          11      12     
            #  DE       NRA       STEP        BG          emis        NTRUE   SUM_TAU
            np.float32, np.int32, np.float32, np.float32, np.float32, None,   None,
            # 13      14        15      16   
            # LCELLS  OFF       PAR     RHO  
            None,     None,     None,   None,
            # 17      18        19     20       21
            # NCHN    NCOMP     HF     PROFILE  MAP_CENTRE
            np.int32, np.int32, None,  None,    cl.cltypes.float3 ])
        else:
            print("SpectraHF has not been defined for OCTREE=%d" % OCTREE), sys.exit()
        
    wrk         =  (tmp_1 * Aul * (NI_ARRAY[:,l]*gg-NI_ARRAY[:,u])) / (freq*freq)
    # wrk was clipped to 1e-25 ... and this produced spikes in spectra ??... not the reason....
    if (0):
        wrk     =  np.clip(wrk, -1.0e-12, 1.0e10)    
        wrk[nonzero(abs(wrk)<1.0e-30)] = 1.0e-30
    else:
        wrk     =  clip(wrk, 1.0e-25, 1e10)          #  KILL ALL MASERS  $$$
        
    WRK[:,0]    =  NI_ARRAY[:, u]    # ni
    WRK[:,1]    =  wrk               # nb_nb
    wrk         =  []
    cl.enqueue_copy(queue, NI_buf, WRK)

    if (INI['FITS']==0):
        fp          =  open('%s_%s_%02d-%02d.spe' % (INI['prefix'], MOL.NAME, u, l), 'wb')
        asarray([NRA, NDE, nchn], int32).tofile(fp)
        asarray([-0.5*(nchn-1.0)*WIDTH, WIDTH], float32).tofile(fp)
        fptau       =  open('%s_%s_%02d-%02d.tau' % (INI['prefix'], MOL.NAME, u, l), 'wb')
    else:
        # pix  =   INI['grid']
        fp    =  MakeEmptyFitsDim(0.0, 0.0, INI['grid']*ARCSEC_TO_RADIAN, NRA, NDE, WIDTH, nchn)
        fptau =  MakeEmptyFitsDim(0.0, 0.0, INI['grid']*ARCSEC_TO_RADIAN, NRA, NDE, WIDTH, nchn)
        
    NTRUE       =  zeros((NRA, nchn), float32)
    ANGLE       =  INI['angle']
    ave_tau     =  0.0
    tau         =  zeros(NRA, float32)
    follow      =  -1
    for de in range(NDE):
        DE      =  de-0.5*(NDE-1.0)
        ## DE      =  +de
        if (HFS): # since CHANNELS has been changed, all transitions written using this kernel ???
            if (OCTREE==0):
                print("HFS:  OCTREE==0   GLOBAL %d  LOCAL %d" % (GLOBAL, LOCAL))
                kernel_spe_hf(queue, [GLOBAL,], [LOCAL,],
                # 0        1        2         3     4          5       6   7    8     9   10         
                CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity,
                # 11        12       13    14    15      16           17     
                NTRUE_buf, STAU_buf, nchn, ncmp, HF_buf, PROFILE_buf, centre)
            elif (OCTREE==4):
                kernel_spe_hf(queue, [GLOBAL,], [LOCAL,],
                # 0        1        2         3     4          5       6   7    8     9
                CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, 
                # 10        11         12        13          14       15       16      
                emissivity, NTRUE_buf, STAU_buf, LCELLS_buf, OFF_buf, PAR_buf, RHO_buf,
                #  17  18    19      20           21     
                nchn,  ncmp, HF_buf, PROFILE_buf, centre)
            else:
                print("kernel_spe_hfs exists only for OCTREE==0 and OCTREE==4"), sys.exit()
        else:
            # print("---------- kernel_spe ----------")
            if (WITH_CRT):
                kernel_spe(queue, [GLOBAL,], [LOCAL,],
                # 0        1        2         3     4          5       6   7    8     9   10          11         12       
                CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity, NTRUE_buf, STAU_buf,
                # 13         14           15     
                CRT_TAU_buf, CRT_EMI_buf, centre)
            else:
                if (OCTREE):
                    follow = -1
                    #   (x,y) = (ra, de)
                    # if (de==80):   follow=202
                    # if (de==78):   follow=167
                    # if (de==56):   follow=205     #    837288
                    # if (de==155):  follow=36
                    # if (de==65):   follow=192
                    # if (de==170):  follow=24
                    # if (de==90):   follow=170
                    kernel_spe(queue, [GLOBAL,], [LOCAL,],
                    # 0        1        2         3     4          5       6   7    8     9   10         
                    CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity, 
                    # 11       12        13          14       15       16       17      18     
                    NTRUE_buf, STAU_buf, LCELLS_buf, OFF_buf, PAR_buf, RHO_buf, follow, centre)
                else:
                    kernel_spe(queue, [GLOBAL,], [LOCAL,],
                    # 0        1        2         3     4          5       6   7    8     9   10          11         12        13  14
                    CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity, NTRUE_buf, STAU_buf, -1, centre)
                    
                    
        # save spectrum
        cl.enqueue_copy(queue, NTRUE, NTRUE_buf)
        
        WWW  = sum(NTRUE, axis=1)
        ira  = argmax(WWW)
        # if (WWW[ira]>300): print("de=%3d  max(W)=%7.2f for ra=%d" % (de, WWW[ira], ira))
                    
        if (INI['FITS']==0):
            for ra in range(NRA):
                asarray([(ra-0.5*(NRA-1.0))*ANGLE, (de-0.5*(NDE-1.0))*ANGLE], float32).tofile(fp) # offsets
                NTRUE[ra,:].tofile(fp)       # spectrum
            # save optical depth
            cl.enqueue_copy(queue, NTRUE, STAU_buf)
            for ra in range(NRA):
                tau[ra]  =  np.max(NTRUE[ra,:])
            ave_tau +=  sum(tau)   # sum of the peak tau values of the individual spectra
            tau.tofile(fptau)      # file containing peak tau for each spectrum
        else:
            for ra in range(NRA):
                fp[0].data[:, de, ra] = NTRUE[ra,:]
            # save optical depth
            cl.enqueue_copy(queue, NTRUE, STAU_buf)
            for ra in range(NRA):
                fptau[0].data[:, de, ra]  = NTRUE[ra,:]
    # --- for de
    if (INI['FITS']==0):
        fp.close()
        fptau.close()
    else:
        fp.writeto('%s_%s_%02d-%02d.fits'        % (INI['prefix'], MOL.NAME, u, l), overwrite=True)
        fptau.writeto('%s_%s_%02d-%02d_tau.fits' % (INI['prefix'], MOL.NAME, u, l), overwrite=True)
        del fp, fptau
    print("  SPECTRUM %3d  = %2d -> %2d,  <tau_peak> = %.3e" % (tran, u, l, ave_tau/(NRA*NDE)))
    
    

    
    
    
#================================================================================
#================================================================================
#================================================================================
#================================================================================

    
    
        
# Main loop -- simulation and updates to level populations
max_change, Tsin, Tsol, Tsav = 0.0, 0.0, 0.0, 0.0
print("================================================================================")
for ITER in range(INI['iterations']):
    print('   ITERATION %d/%d' % (1+ITER, INI['iterations']))
    t0    =  time.time()
    Simulate()
    Tsim  =  time.time()-t0
    t0    =  time.time()
    if (INI['clsolve']):
        ave_max_change = SolveCL()
    else:
        ave_max_change = Solve(CELLS, MOL, INI, LEVELS, TKIN, RHO, ABU, ESC_ARRAY)
    Tsol  =  time.time()-t0
    t0    =  time.time()
    if (((ITER%4==3)|(ITER==(INI['iterations']-1))) & (len(INI['save'])>0)): # save level populations
        print("      ... save level populations") 
        fp = open(INI['save'], 'wb')
        asarray([NX, NY, NZ, LEVELS], int32).tofile(fp)
        asarray(NI_ARRAY, float32).tofile(fp)
        fp.close()
    Tsave = time.time()-t0
    print("      SIMULATION %7.2f    SOLVE %7.2f    SAVE %7.2f" % (Tsim, Tsol, Tsav))
    if (ave_max_change<INI['stop']):  break
print("================================================================================")
if ((INI['iterations']>0)&(COOLING)):
    Cooling()        

    
mleaf = nonzero(RHO<=0.0)
NI_ARRAY[mleaf[0], :] = NaN

if (0):    
    clf()    
    plot(NI_ARRAY[:,0], 'b.')    
    plot(NI_ARRAY[:,1], 'r.')
    show()
    sys.exit()
    
# Save Tex files
ul = INI['Tex']                # upper and lower level for each transition
## print(ul)
for i in range(len(ul)//2):    # loop over transitions
    u, l  =  ul[2*i], ul[2*i+1]
    tr    =  MOL.L2T(u,l)
    if (tr<0):
        print("*** Error:  Tex %2d -> %2d  is not a valid transition" % (u, l))
        continue
    gg    =  MOL.G[u]/MOL.G[l]
    fp    =  open('%s_%s_%02d-%02d.tex' % (INI['prefix'], MOL.NAME, u, l), 'wb')
    asarray([NX, NY, NZ, LEVELS], int32).tofile(fp)
    tex   =  BOLTZMANN * log(NI_ARRAY[:, l]*gg/NI_ARRAY[:, u])
    if (1):
        m = nonzero((RHO>0.0)&(~isfinite(tex)))
        print("NaNs: %d" % len(m[0]))
        if (len(m[0])>10):
            print("Problem RHO percentiles:", percentile(RHO[m], (0.0, 10.0, 50.0, 90.0, 100.0)))
        elif (len(m[0])>0):
            print("Problem RHO all:", RHO[m])
        nnn = min([len(m[0]), 10])
        for i in range(nnn):
            icell = m[0][i]
            print("--- CELL %8d  RHO %.3e   n[%d] = %11.3e      n[%d] = %11.3e" % (icell, RHO[icell], l, NI_ARRAY[icell, l],   u, NI_ARRAY[icell, u]))
            sys.stdout.write("       SIJ ")
            for j in range(TRANSITIONS):
                sys.stdout.write(' %11.3e' % (SIJ_ARRAY[icell, j]))
            sys.stdout.write('\n')
            ###
            #icell = m[0][i]+1001
            # print("+++ CELL %8d  RHO %.3e   n[%d] = %11.3e      n[%d] = %11.3e" % (icell, RHO[icell], l, NI_ARRAY[icell, l],   u, NI_ARRAY[icell, u]))
            #sys.stdout.write("       SIJ ")
            #for j in range(TRANSITIONS):
            #    sys.stdout.write(' %11.3e' % (SIJ_ARRAY[icell, j]))
            #sys.stdout.write('\n')
    m     =  nonzero(abs(tex)>1.0e-35)
    tex   =  PLANCK * MOL.F[tr] / tex
    asarray(tex, float32).tofile(fp)
    fp.close()
    if (OCTREE):
        m = nonzero(RHO>0.0)
        print("  TEX      %3d  = %2d -> %2d,  [%.3f,%.3f]  %.3f K" % (tr, u, l, min(tex[m]), max(tex[m]), mean(tex[m])))
    else:
        print("  TEX      %3d  = %2d -> %2d,  [%.3f,%.3f]  %.3f K" % (tr, u, l, min(tex), max(tex), mean(tex)))

    if (0):
        clf()
        plot(PL, tex, 'r.')
        xlabel(r'$\rm Path \/ \/ length$')
        ylabel(r'$T\rm_{ex} \/ \/ (K)$')
        title(r'$%.4f \pm %.4f$' % (mean(tex), std(tex)))
        show()
        sys.exit()
        
    #subplot(1,2,1+i)
    #plot(tex, 'r.')
#show()
#sys.exit()
        

# Save spectra
ul = INI['spectra']
for i in range(len(ul)//2):
    WriteSpectra(INI, ul[2*i], ul[2*i+1])

print("================================================================================")

print("LOC_OT.py TOTAL TIME: %.3f SECONDS" % (time.time()-t000))
    
print(type(NI_ARRAY))    
print(type(SIJ_ARRAY))    
print(type(ESC_ARRAY))    
    
