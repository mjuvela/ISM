#!/bin/python2

import os, sys
HOMEDIR = os.path.expanduser('~/')
sys.path.append(HOMEDIR+'/starformation/SOC/')
from   SOC_aux import *
import multiprocessing as mp

GLOBAL_0 = 32768
# GLOBAL_0 = 2048
# GLOBAL_0 = 65536

if (len(sys.argv)<2):  
    print(" pySOCS input_file"), sys.exit()
USER = User(sys.argv[1])
if (not(USER.Validate())):   
    print("Check the inifile... exiting!"),  sys.exit()

# Read optical data for the dust model, updates USER.NFREQ
FFREQ,AFG,AFABS,AFSCA  =  read_dust(USER)
NFREQ               =  USER.NFREQ
NDUST               =  len(AFABS)
# Read scattering functions:  FDSC[NFREQ, BINS], FCSC[NFREQ,BINS]
FDSC, FCSC          =  read_scattering_functions(USER)
WITH_MSF            =  len(FDSC)>1  # only with WITH_ABU, also using multiple scattering functions
if (WITH_MSF==0):
    DSC, CSC        =  None, None
else:
    DSC, CSC        =  zeros((NDUST, USER.DSC_BINS), np.float32), zeros((NDUST, USER.DSC_BINS), np.float32)    
# Read intensities of the background and of point source
IBG                 =  read_background_intensity(USER)
LPS                 =  read_source_luminosities(USER)   # LPS[USER.NO_PS, USER.NFREQ]
# Read cloud file and scale densities; also updates USER.AREA/AXY/AXZ/AXZ
NX, NY, NZ, LEVELS, CELLS, LCELLS, OFF, DENS = read_cloud(USER)
print('--- DENS --- %d, %d' % (NX*NY*NZ, len(DENS)))
# Open file containing diffuse emission per cell
DIFFUSERAD          =  mmap_diffuserad(USER, CELLS)
DEVICES             =  len(USER.DEVICES)
KDEV                =  1.0  # here only one device !!
# Parameters related to the observer directions
NODIR, ODIR, RA, DE =  set_observer_directions(USER)        #  ODIR[NODIR, clfloat3]
ABU                 =  read_abundances(CELLS, NDUST, USER)  # [CELLS, NDUST] or []
WITH_ABU            =  len(ABU.shape)>1
OPT                 =  zeros(3, np.float32)  # WITH_ABU => large abundance vectors for all dust species
if (WITH_ABU): # ABU + SCA vectors in case of non-constant dust abundances
    OPT   = zeros((CELLS, 2), np.float32)
# We do not allow multiple scattering functions unless we have variable abundances
# => if abundances are fixed, one should calculate combined scattering function outside SOC
if ((WITH_ABU==0)&(WITH_MSF)):
    print("Multiple scattering functions given but abundances do not vary =>")
    print("Calculate a single scattering function outside SOC!")
    sys.exit()

    
# User may limit output (intensities, spectra) to a range between two frequencies
m                  = nonzero((FFREQ>=USER.REMIT_F[0])&(FFREQ<=USER.REMIT_F[1]))
REMIT_I1, REMIT_I2 = m[0][0], m[0][-1]
REMIT_NFREQ        = len(m[0])   #   REMIT_NFREQ <= NFREQ
USER.REMIT_NFREQ, USER.REMIT_I1, USER.REMIT_I2 = REMIT_NFREQ, REMIT_I1, REMIT_I2 
print("Emitted data for channels [%d, %d], %d out of %d channels" % (REMIT_I1, REMIT_I2, REMIT_NFREQ, NFREQ))
print("     %.3e - %.3e Hz" % (USER.REMIT_F[0], USER.REMIT_F[1]))

# We can have REMIT_NFREQ<NFREQ only if emission is not used in the simulation
#  ... that is, require full frequency grid for temperature calculations
if ((REMIT_NFREQ<NFREQ) & ((USER.ITERATIONS>0) & (USER.CLPAC>0))):
    print("NFREQ=%d, REMIT_NFREQ=%d -- cannot be if ITERATIONS=%d, CLPAC=%d>0" %
    (NFREQ, REMIT_NFREQ, USER.ITERATIONS, CLPAC))

LOCAL = 8
if (USER.DEVICES.find('c')<0): 
    LOCAL = 32
if ('local' in USER.KEYS):
    print(USER.KEYS['local'])
    LOCAL = int(USER.KEYS['local'][0])
GLOBAL_0 = Fix(GLOBAL_0, 4*LOCAL)
    
PSPAC = Fix(USER.PSPAC, LOCAL)                    # any number ... but multiple of LOCAL
BGPAC = Fix(Fix(USER.BGPAC, USER.AREA), LOCAL)    # multiple of AREA and LOCAL
DFPAC = 0
if (USER.USE_EMWEIGHT>0):
    CLPAC = Fix(USER.CLPAC, LOCAL)                # multiple of LOCAL
    DFPAC = Fix(USER.DFPAC, LOCAL)                # multiple of LOCAL
else:
    CLPAC = Fix(Fix(USER.CLPAC, CELLS), LOCAL)    # multiple of CELLS and LOCAL
    DFPAC = Fix(Fix(USER.DFPAC, CELLS), LOCAL)
# Either CLPAC==DFPAC or CLPAC=0 and DFPAC>=0
print('*'*80)
print('PACKETS: PSPAC %d   BGPAC %d  CLPAC %d  DFPAC %d' % (PSPAC, BGPAC, CLPAC, DFPAC))
print('*'*80)
asarray([BGPAC, PSPAC, DFPAC, CLPAC], np.int32).tofile('packet.info')

# Precalculate some things for point sources located outside the model volume
#  XPS_NSIDE[NO_PS], XPS_SIDE[3*NO_PS], XPS_AREA[3*NO_PS]  ---- but PSPOS[MAXPS]
XPS_NSIDE, XPS_SIDE, XPS_AREA = AnalyseExternalPointSources(NX, NY, NZ, USER.PSPOS, int(USER.NO_PS), int(USER.PS_METHOD))

HPBG = []
if (len(USER.file_hpbg)>2):
    # We use healpix maps for the background sky intensity... 
    #   currently *fixed* at NSIDE=64 which gives ~55 arcmin pixels, 49152 pixels on the sky
    HPBG = np.fromfile(USER.file_hpbg, np.float32).reshape(NFREQ, 49152)
    ### 2019-04-18:  TEST_ORIENTATIONS.py -> TO_scattering_hp2*.png -> SOCS.py + Healpix not working
    print("\n\n")
    print("SOCS.py + Healpix background does not work.... use ASOCS.py instead !!!")
    print("\n\n")
    time.sleep(5)
    # sys.exit()
    
    
# for better sampling of emission from external point sources
XPS_NSIDE_buf, XPS_SIDE_buf, XPS_AREA_buf = [], [], []
# 2018-12-27 -- for -D WITH_MSF make ABS, SCA, G potentially vectors => need buffers
ABS_buf, SCA_buf, ABU_buf = [], [], []

# for point sources we use exactly PSPAC packages
# for background it is     int(BGPAC/AREA)*AREA
# for cell emission it is  (CLPAC/CELLS) packages per cell
# for HPBG (Healpix backgound) BGPAC can be anything
WPS = 0.0
WBG = 0.0

print('-'*90)
print("PS_METHOD=%d, WITH_ABU=%d, WITH_MSF=%d" % (USER.PS_METHOD, WITH_ABU, WITH_MSF))
print('-'*90)

ARGS = "-D NX=%d -D NY=%d -D NZ=%d -D BINS=%d -D WITH_ALI=%d -D PS_METHOD=%d \
-D CELLS=%d -D AREA=%.0f -D NO_PS=%d  %s -D WITH_ABU=%d -D FACTOR=%.4ef \
-D AXY=%.5ff -D AXZ=%.5ff -D AYZ=%.5ff -D LEVELS=%d -D LENGTH=%.5ef \
-I%s/starformation/SOC/ \
-D POLSTAT=%d -D SW_A=%.3ef -D SW_B=%.3ef -D STEP_WEIGHT=%d -D DIR_WEIGHT=%d -D DW_A=%.3ef \
-D LEVEL_THRESHOLD=%d -D POLRED=%d -D WITH_COLDEN=%d -D MINLOS=%.3ef -D MAXLOS=%.3ef \
-D FFS=%d -D NODIR=%d -D METHOD=%d -D USE_EMWEIGHT=%d -D HPBG_WEIGHTED=%d -D WITH_MSF=%d -D NDUST=%d \
-D OPT_IS_HALF=%d" % \
(  NX, NY, NZ, USER.DSC_BINS, USER.WITH_ALI, USER.PS_METHOD,
   CELLS, int(USER.AREA), max([1,int(USER.NO_PS)]), USER.kernel_defs, WITH_ABU, FACTOR, 
   USER.AXY, USER.AXZ, USER.AYZ, LEVELS, USER.GL*PARSEC, 
   os.getenv("HOME"), #####  NSIDE==USER.NPIX['x'] ???
   USER.POLSTAT, int(USER.STEP_WEIGHT[0]), USER.STEP_WEIGHT[1], int(USER.STEP_WEIGHT[2]), 
   int(USER.DIR_WEIGHT[0]), USER.DIR_WEIGHT[1],
   USER.LEVEL_THRESHOLD, len(USER.file_polred)>0, len(USER.file_colden)>1, USER.MINLOS, USER.MAXLOS,
   USER.FFS, NODIR, USER.METHOD, USER.USE_EMWEIGHT, USER.HPBG_WEIGHTED, WITH_MSF, NDUST,
   USER.OPT_IS_HALF)
NVARGS  = " -cl-fast-relaxed-math"
ARGS   += NVARGS
print(ARGS)

# Create contexts, command queue, and program = kernels for the simulation step
source            =  os.getenv("HOME")+"/starformation/SOC/kernel_SOC_sca.c"        
context, commands =  opencl_init(USER)   # context and commands are both arrays [DEVICES]
program           =  get_program(context, commands, source, ARGS) # program in also an array [DEVICES]
mf                =  cl.mem_flags
    
# A set of buffers needed by emission calculations
LCELLS_buf, OFF_buf, PAR_buf, DENS_buf, EMIT_buf, DSC_buf, CSC_buf = [], [], [], [], [], [], []
TABS_buf, RA_buf, DE_buf, ODIR_buf, PS_buf, PSPOS_buf = [], [], [], [], [], []
EMINDEX_buf, OUT_buf = [], []
OPT_buf, HPBG_buf, HPBGP_buf = [], [], []
OUT = zeros((NODIR, USER.NPIX['y'], USER.NPIX['x']), np.float32)
print("OUT HAS THE SIZE OF:", OUT.shape)

for ID in range(DEVICES):
    # print("ID %d,  HPBG %d  x %d" % (ID, len(HPBG), len(HPBG[0])))
    LCELLS_buf.append( cl.Buffer(context[ID], mf.READ_ONLY,  LCELLS.nbytes))
    OFF_buf.append(    cl.Buffer(context[ID], mf.READ_ONLY,  OFF.nbytes))
    DENS_buf.append(   cl.Buffer(context[ID], mf.READ_ONLY,  DENS.nbytes))
    ODIR_buf.append(   cl.Buffer(context[ID], mf.READ_ONLY,  ODIR.nbytes))
    RA_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  RA.nbytes))
    DE_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  DE.nbytes))
    PAR_buf.append(    cl.Buffer(context[ID], mf.READ_WRITE, 4*CELLS))
    EMIT_buf.append(   cl.Buffer(context[ID], mf.READ_ONLY,  4*CELLS))
    PS_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  4*max([1,USER.NO_PS])))
    PSPOS_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  USER.PSPOS.nbytes))  # always MAXPS float3 !!
    OUT_buf.append(    cl.Buffer(context[ID], mf.READ_WRITE, OUT.nbytes))
    if (0):
        if (USER.USE_EMWEIGHT==2):
            EMINDEX_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*CELLS))
        else:
            EMINDEX_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4))  # dummy
    if (WITH_ABU):
        if (USER.OPT_IS_HALF):
            OPT_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 2*CELLS*2))  # ABS, SCA
        else:
            OPT_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*CELLS*2))  # ABS, SCA
    else:
        OPT_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY, 4*3))  # DUMMY
    if (size(HPBG)>0):
        print("========== HPBG init ==========")
        HPBG_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*49152))  # NSIDE==64
        if (USER.HPBG_WEIGHTED): # use weighting for emission from Healpix background
            print("========== HPBGP_buf allocated ==========")
            HPBGP_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*49152 ))
        else:
            HPBGP_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*3))  # DUMMY
    else:
        HPBG_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY, 4*3))  # DUMMY
        HPBGP_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*3))  # DUMMY        
    if (1):
        # data on external point sources...
        XPS_NSIDE_buf.append( cl.Buffer(context[ID], mf.READ_ONLY,  4*max([1,   USER.NO_PS]))) # [NO_PS  ] int
        XPS_SIDE_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*max([1, 3*USER.NO_PS]))) # [3*NO_PS] int
        XPS_AREA_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*max([1, 3*USER.NO_PS]))) # [3*NO_PS] float
        ### sys.exit()
    if (WITH_MSF==0):  # multiple scattering functions
        ABS_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4)   )
        SCA_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4)   )
        ABU_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4)   )  # dummy
        DSC_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*USER.DSC_BINS))
        CSC_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*USER.DSC_BINS))
        ABS, SCA = zeros(1, np.float32), zeros(1, np.float32)
    else:
        ABS_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*NDUST)  )
        SCA_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*NDUST)  )
        ABU_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*NDUST*CELLS)   )  
        DSC_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*USER.DSC_BINS*NDUST))
        CSC_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*USER.DSC_BINS*NDUST))
        ABS, SCA = zeros(NDUST, np.float32), zeros(NDUST, np.float32)


for ID in range(DEVICES):        
    cl.enqueue_copy(commands[ID], LCELLS_buf[ID], LCELLS)
    cl.enqueue_copy(commands[ID], OFF_buf[ID],    OFF)
    cl.enqueue_copy(commands[ID], DENS_buf[ID],   DENS)
    cl.enqueue_copy(commands[ID], ODIR_buf[ID],   ODIR)
    cl.enqueue_copy(commands[ID], RA_buf[ID],     RA)
    cl.enqueue_copy(commands[ID], DE_buf[ID],     DE)
    if (USER.NO_PS>0):
        cl.enqueue_copy(commands[ID],     PSPOS_buf[ID],  USER.PSPOS) #  MAXPS float3
        cl.enqueue_copy(commands[ID], XPS_NSIDE_buf[ID],  XPS_NSIDE)  #  NO_PS          int
        cl.enqueue_copy(commands[ID],  XPS_SIDE_buf[ID],  XPS_SIDE)   #  NO_PS * 3      int
        cl.enqueue_copy(commands[ID],  XPS_AREA_buf[ID],  XPS_AREA)   #  NO_PS * 3      float
    if (WITH_MSF):
        cl.enqueue_copy(commands[ID], ABU_buf[ID], ABU)  # ABU[NDUST,CELLS]
    commands[ID].finish()                


EMWEI, EMWEI_buf = [], []
if (USER.USE_EMWEIGHT>0):
    pac       = max([CLPAC, DFPAC])                        #  equal or CLPAC=0 and DFPAC>0
    EMWEI     = ones(CELLS, np.float32)  * (pac/CELLS)     # could be ushort, 0-65k packages per cell?
    for ID in range(DEVICES):        
        EMWEI_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*CELLS) )
else:    
    for ID in range(DEVICES):        
        EMWEI_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4) )  # dummy buffer


# Update the parent links on devices = PAR_buf
kernel_parents = []
GLOBAL  = GLOBAL_0
for ID in range(DEVICES):
    kernel_parents.append(program[ID].Parents)  # initialisation of links to cell parents
    kernel_parents[ID](commands[ID], [GLOBAL,], [LOCAL,], DENS_buf[ID], LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID])
    commands[ID].finish()

    
# EMITTED will be mmap array  EMITTED[icell, ifreq], possibly REMIT_NFREQ frequencies
#  REMIT_NFREQ, indices [REMIT_I1, REMIT_I2]  => EMITTED is limited to this
#     should work even even if cell emission is included
EMITTED  = mmap_emitted(USER, CELLS, LEVELS, LCELLS, REMIT_NFREQ, OFF, DENS)  # EMITTED[CELLS, REMIT_NFREQ]

Tkernel, Tpush, Tpull, Tsolve, Tmap = 0.0, 0.0, 0.0, 0.0, 0.0

# New host buffers
ID = 0
EMIT            = zeros(CELLS, np.float32)   # true emission per cell
kernel_PB       = program[ID].SimRAM_PB
kernel_CL       = program[ID].SimRAM_CL
kernel_HP       = program[ID].SimRAM_HP
kernel_zero_out = program[ID].zero_out


kernel_PB.set_scalar_arg_dtypes(
#   0      1          2         3           4           5           
# SOURCE   PACKETS    BATCH     SEED        ABS         SCA         
[np.int32, np.int32,  np.int32, np.float32, None,       None,
# 6         7            8    
# BG        PSPOS        PS   
np.float32, None,        None,
# 9       10    11    12     13     14     
# LCELLS  OFF   PAR   DENS   DSC    CSC    
None,     None, None, None,  None,  None,  
# 15         16      17                    18           19                      20    21    22   
# NODIR      ODIR,   NPIX                  MAP_DX       CENTRE                  ORA   ODE   OUT  
  np.int32,  None,   clarray.cltypes.int2, np.float32,  clarray.cltypes.float3, None, None, None,
# 23      24      25           26          27      
# ABU     OPT     XPS_NSIDE    XPS_SIDE    XPS_AREA
None,     None,   None,        None,       None    ])
    

kernel_HP.set_scalar_arg_dtypes(
#   0        1          2          3     4    
#   PACKETS  BATCH     SEED        ABS   SCA  
[np.int32,   np.int32, np.float32, None, None,
# 5       6     7     8      9     10    
# LCELLS  OFF   PAR   DENS   DSC   CSC   
None,     None, None, None,  None, None, 
#  11     12      13                     14           15                        16    17    18  
#  NDIR   ODIRS   NPIX                   MAP_DX       CENTRE                    ORA   ODE   OUT 
np.int32, None,   clarray.cltypes.int2,  np.float32,  clarray.cltypes.float3,   None, None, None,
#  19     20      21     22   
#  ABU    OPT     BG     HPBGP
None,     None,   None,  None ])    


kernel_CL.set_scalar_arg_dtypes(
#   0      1          2         3           4           5          
# SOURCE   PACKETS    BATCH     SEED        ABS         SCA        
[np.int32, np.int32,  np.int32, np.float32, None,       None,
# 6       7     8     9      10     11     12    13        14    
# LCELLS  OFF   PAR   DENS   EMIT   DSC   CSC    NODIR     ODIRS 
None,     None, None, None,  None,  None, None,  np.int32, None,    
# 15                  16           17                      18    19    20  
# NPIX                MAP_DX       MAPCENTRE               ORA   ODE   OUT 
clarray.cltypes.int2, np.float32,  clarray.cltypes.float3, None, None, None,
#  21    22 
#  OPT   ABU
None, None ])

kernel_zero_out.set_scalar_arg_dtypes([np.int32, clarray.cltypes.int2, None])
    
    
# Initialise memory mapped file for outcoming photons
fp = open('outcoming.socs', 'w')
asarray([USER.NPIX['y'], USER.NPIX['x'], USER.NFREQ], np.int32).tofile(fp)
asarray(FFREQ, np.float32).tofile(fp)
fp.close()
OUTCOMING  = \
  np.memmap('outcoming.socs',dtype='float32',mode="r+",
  shape=(USER.NFREQ, NODIR, USER.NPIX['y'], USER.NPIX['x']),offset=4*(3+NFREQ))
OUTCOMING[:,:,:,:] = 0.0   #  [NFREQ, NODIR, NPIX.y, NPIX.x]
print("OUTCOMING[NFREQ=%d,NDIR=%d,NPIX.y=%d,NPIX.x=%d]" % (USER.NFREQ, NODIR, USER.NPIX['x'], USER.NPIX['y']))


BG, PS = 0.0, 0.0  # photons per package for background and point source
# Simulate the constant radiation sources with PSPAC and BGPAC packages
# the resulting TABS array is stored to CTABS and saved to disk
# CLPAC = constant diffuse emission
#    do DFPAC here, separate of CLPAC, to avoid complications with ALI
skip  = 2
for II in range(3):  # loop over PSPAC, BGPAC and DFPAC==DIFFUSERAD simulations
    print("CONSTANT SOURCES II=%d" % II)
    
    if (II==0):   # point source
        # 2017-12-25 -- each work item does BATCH packages, looping over the sources
        #   => BATCH is multiple of USER.NO_PS
        GLOBAL   =  GLOBAL_0
        if ((PSPAC<1)|(USER.NO_PS<1)): 
            continue   # point sources not simulated
        BATCH    =  int(max([1, PSPAC / GLOBAL]))  # each work item does this many per source
        PSPAC    =  GLOBAL * BATCH                 # actual number of packages per source
        WPS      =  1.0 / (PLANCK*PSPAC*((USER.GL*PARSEC)**2.0))
        BATCH   *=  USER.NO_PS   # because each work item loops over ALL point sources
        PSPAC   *=  USER.NO_PS   # total number of packages (including all sources)
        PACKETS  =  PSPAC
        print("=== PS  GLOBAL %d x BATCH %d = %d" % (GLOBAL, BATCH, PSPAC))
    elif (II==1): # background -- GLOBAL == surface area, BATCH>=1
        if (BGPAC<1): continue
        if (size(HPBG)<1): # truly isotropic background
            # We may not have many surface elements -> GLOBAL remains too small
            # calculate BATCH assuming that GLOBAL is 8 x AREA
            BATCH    =  max([1, int(round(BGPAC/(8*USER.AREA)))])
            BGPAC    =  int(8*USER.AREA*BATCH)   # GLOBAL >= 8*AREA !!
            PACKETS  =  int(BGPAC)
            WBG      =  np.pi/(PLANCK*8*BATCH)
            GLOBAL   =  Fix(int(8*USER.AREA), 64)
            if (LOCAL==12):     GLOBAL  =  Fix(int(8*USER.AREA), 32*LOCAL)
            assert(GLOBAL>=(8*USER.AREA))
        else:
            # Healpix background
            # Calculate WBG for a sphere R^2= 0.25*(NX^2+NY^2+NZ^2) and include rejections in the kernel
            # The number of work items is not tied to the number of cloud or sky elements.
            print("************ HEALPIX ***************")
            BATCH    =  1
            GLOBAL   =  int(BGPAC/BATCH)
            GLOBAL   =  Fix(GLOBAL, 64)
            BGPAC    =  GLOBAL*BATCH
            PACKETS  =  GLOBAL*BATCH                   # *every* work item does BATCH packages
            Rout     =  0.5*sqrt(NX*NX+NY*NY+NZ*NZ)    # radius of a sphere containing the cloud
            if (0): # sending only packages that hit the cloud 
                # this is not correct because it does not take into account the the project size
                # of the cloud depends on the direction (cloud is not a sphere)
                WBG  =  np.pi * USER.AREA            /  (PLANCK*BGPAC)
            else:   # packages sent towards a sphere with radius Rout [GL]
                WBG  =  np.pi * 4.0*np.pi*Rout**2.0  /  (PLANCK*BGPAC)
            print("HPBG: BATCH %d, GLOBAL %d, BGPAC %d" % (BATCH, GLOBAL, BGPAC))
        print("=== BGPAC %d, BATCH %d ===" % (BGPAC, BATCH))
    else:
        # RAM2 version --- each work item loops over cells =>  GLOBAL<CELLS, BATCH==packets per cell
        GLOBAL   =  GLOBAL_0
        if (len(DIFFUSERAD)<1): continue  # no diffuse radiation sources            
        if (DFPAC<1): continue            # ??? THIS MAY BE >0 EVEN WHEN NO DIFFUSE FIELD USED ??
        BATCH    =  int(DFPAC/CELLS)
        PACKETS  =  DFPAC                
        print("=== DFPAC %d, GLOBAL %d, BATCH %d, LOCAL %d" % (DFPAC, GLOBAL, BATCH, LOCAL))
        
        
        
    for IFREQ in range(NFREQ):
        
        FREQ  =  FFREQ[IFREQ]
        # print('IFREQ %d / %d' % (IFREQ, NFREQ))
        kernel_zero_out(commands[ID], [GLOBAL,], [LOCAL,], NODIR, USER.NPIX, OUT_buf[ID])
        
        
        ABS[0], SCA[0]  = AFABS[0][IFREQ], AFSCA[0][IFREQ]        
        if (WITH_ABU>0): # ABS, SCA, G are vectors
            OPT[:,:] = 0.0
            for idust in range(NDUST):
                OPT[:,0] +=  ABU[:,idust] * AFABS[idust][IFREQ]
                OPT[:,1] +=  ABU[:,idust] * AFSCA[idust][IFREQ]
            if (USER.OPT_IS_HALF):
                cl.enqueue_copy(commands[ID], OPT_buf[ID], asarray(ravel(OPT), np.float16))
            else:
                cl.enqueue_copy(commands[ID], OPT_buf[ID], asarray(ravel(OPT), np.float32))
            if (WITH_MSF):  # ABS[], SCA[] actually not needed !!
                for idust in range(NDUST):
                    ABS[idust] = AFABS[idust][IFREQ]
                    SCA[idust] = AFSCA[idust][IFREQ]
        else:               # constant abundance ->  ABS, SCA, G are scalars
            if (WITH_MSF):
                print("WITH_ABU=0 and WITH_MSF=1  -- no such combination!!")
                sys.exit(0)
            ABS[0], SCA[0] = 0.0, 0.0
            for idust in range(NDUST):
                ABS[0] += AFABS[idust][IFREQ]
                SCA[0] += AFSCA[idust][IFREQ]                
        cl.enqueue_copy(commands[ID], ABS_buf[ID], ABS)
        cl.enqueue_copy(commands[ID], SCA_buf[ID], SCA)
                
        BG  =  0.0
        if (len(IBG)==NFREQ):  BG = IBG[IFREQ] * WBG/FREQ    # background photons per package
        if (II==0): # update point source luminosities for the current frequency
            PS = LPS[:,IFREQ] * WPS/FREQ  # source photons per package
            cl.enqueue_copy(commands[ID], PS_buf[ID], PS)
        else:
            PS = [0.0,]

            
        if ((II==1)&(size(HPBG)>0)):
            # push the Healpix background map for the current frequency to device
            if (USER.HPBG_WEIGHTED):  # using weighted emission from the Healpix background
                tmp    =  asarray(HPBG[IFREQ,:], float64)
                # convert into probablity
                tmp   /=  mean(tmp)
                tmp    =  clip(tmp, 1.0e-2, 1.0e4) # clip very low and high probabilities
                tmp   /=  sum(tmp)                 # 1/49152  ->  tmp    == probability per Healpix pixel
                print("HPBGp", percentile(tmp, (0, 10, 50, 90, 100)))                    
                HPBGW  =  (1.0/49152.0)/tmp        # weight of each Healpix pixel, relative to unweighted case
                HPBGP  =  cumsum(tmp)     
                HPBGP[-1] = 1.00001                # avoid any chance of rounding error here
                # Include the additional weighting for relative pixel probability directly in HPBG !!
                cl.enqueue_copy(commands[ID], HPBG_buf[ID],  asarray((WBG/FREQ)*HPBG[IFREQ,:]*HPBGW, np.float32))
                cl.enqueue_copy(commands[ID], HPBGP_buf[ID], asarray(HPBGP, np.float32))
            else:
                # no relative weighting between Healpix pixels
                cl.enqueue_copy(commands[ID], HPBG_buf[ID], asarray((WBG/FREQ)*HPBG[IFREQ,:], np.float32))

                
        print("  FREQ %3d/%3d  %10.3e --  ABS %.3e  SCA %.3e  BG %10.3e  PS %10.3e..." % (IFREQ+1, NFREQ, FREQ,    ABS[0], SCA[0], BG, PS[0]))
        # Upload to device new DSC, CSC, possibly also EMIT
        t0 = time.time()
        if (WITH_MSF==0):
            cl.enqueue_copy(commands[ID], DSC_buf[ID], FDSC[0,IFREQ,:])  # FDSC[idust,ifreq,ibin]
            cl.enqueue_copy(commands[ID], CSC_buf[ID], FCSC[0,IFREQ,:])
        else:
            for idust in range(NDUST):
                DSC[idust,:] = FDSC[idust,IFREQ,:]
                CSC[idust,:] = FCSC[idust,IFREQ,:]
            cl.enqueue_copy(commands[ID], DSC_buf[ID], DSC)
            cl.enqueue_copy(commands[ID], CSC_buf[ID], CSC)
        commands[ID].finish()
        Tpush += time.time()-t0    
        if (USER.SEED>0): seed = fmod(USER.SEED+SEED0+(DEVICES*IFREQ+ID)*SEED1, 1.0)
        else:             seed = rand()
        
        if (II==2):  # emission from the volume (but not yet the dust)
            if (IFREQ>=DIFFUSERAD.shape[1]): # may contain fewer frequencies !!
                EMIT[:] = 0.0
                continue
            # Note -- DIFFUSERAD is emission as photons / 1Hz / cm3
            #         EMIT is number of photons for volume (GL*pc)^3 divided by (GL*pc)^2
            #         => EMIT = DIFFUSERAD * (GL*pc)
            for level in range(LEVELS):
                coeff      =  USER.GL*PARSEC / (8.0**level)    # cell volume in units of GL^3
                a, b       =  OFF[level], OFF[level]+LCELLS[level]
                EMIT[a:b]  =  DIFFUSERAD[a:b, IFREQ] * coeff   # DIFFUSERAD = photons/cm3
            EMIT[nonzero(DENS<1.0e-10)] = 0.0    # empty cells emit nothing
            cl.enqueue_copy(commands[ID], EMIT_buf[ID], EMIT)
        commands[ID].finish()
        t0 = time.time()            
        if (II==2): # DFPAC --- the same kernel as for CLPAC !!
            kernel_CL(commands[ID], [GLOBAL,], [LOCAL,],
            np.int32(II), np.int32(PACKETS), np.int32(BATCH), np.float32(seed), 
            ABS, SCA, LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], 
            EMIT_buf[ID], DSC_buf[ID], CSC_buf[ID], NODIR, ODIR_buf[ID], 
            USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, RA_buf[ID], DE_buf[ID], 
            OUT_buf[ID], OPT_buf[ID], ABU_buf[ID])
        else:       # PSPAC, BGPAC
            if ((II==1)&(size(HPBG)>0)): # using a healpix map for the background
                # print(ID, GLOBAL, LOCAL)
                # print(PACKETS, BATCH, seed)
                # print(ABS_buf[ID], SCA_buf[ID])
                # print(LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], DSC_buf[ID])
                kernel_HP(commands[ID], [GLOBAL,], [LOCAL,],
                PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], 
                LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], DSC_buf[ID], CSC_buf[ID], 
                NODIR, ODIR_buf[ID], USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, 
                RA_buf[ID], DE_buf[ID], OUT_buf[ID], ABU_buf[ID], OPT_buf[ID], 
                HPBG_buf[ID], HPBGP_buf[ID])
            else:
                kernel_PB(commands[ID], [GLOBAL,], [LOCAL,], np.int32(II), 
                np.int32(PACKETS), np.int32(BATCH), seed, ABS_buf[ID], SCA_buf[ID], 
                BG, PSPOS_buf[ID], PS_buf[ID], 
                LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], DSC_buf[ID], CSC_buf[ID], 
                np.int32(NODIR), ODIR_buf[ID], USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, 
                RA_buf[ID], DE_buf[ID], OUT_buf[ID], ABU_buf[ID], OPT_buf[ID], 
                XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID])
        commands[ID].finish()                
        Tkernel += time.time()-t0 # REALLY ONLY THE KERNEL EXECUTION TIME
        
        # pull OUT and add to mmap file
        cl.enqueue_copy(commands[ID], OUT, OUT_buf[ID])
        commands[ID].finish()
        OUTCOMING[IFREQ,:,:,:] += OUT    # OUTCOMING[NFREQ, NODIR, NPIX.y, NPIX.x]
    # end of -- for IFREQ
    
    
# Next simulate the emission from the dust within the model volume, possibly over many iterations
# Keep updating the escape probability on each iteration, because temperatures change and
# (with reference field) to make the estimates more precise.

BATCH    = int(CLPAC/CELLS)
EMDONE   = None


    
# RAM2 version --- each work item loops over cells =>  GLOBAL<CELLS, BATCH==packets per cell
GLOBAL, BATCH  =  GLOBAL_0,  max([1,int(CLPAC/CELLS)])
print('=== CLPAC %d, GLOBAL %d, BATCH %d' % (CELLS*BATCH, GLOBAL, BATCH))

if (CLPAC>0):
    for IFREQ in range(NFREQ):     # loop over single frequencies (all the frequencies)

        kernel_zero_out(commands[ID],[GLOBAL,],[LOCAL,], NODIR, USER.NPIX, OUT_buf[ID])
        commands[ID].finish()
        # Parameters for the current frequency
        FREQ  =  FFREQ[IFREQ]
        if ((FREQ<USER.SIM_F[0])|(FREQ>USER.SIM_F[1])): continue # FREQ not simulayted!!
        print("    FREQ %3d/%3d   %12.4e --  ABS %.3e  SCA %.3e" % (IFREQ+1, NFREQ, FREQ, ABS, SCA))
        
        ABS[0], SCA[0]  = AFABS[0][IFREQ], AFSCA[0][IFREQ]        
        if (WITH_ABU>0): # ABS, SCA, G are vectors
            OPT[:,:] = 0.0
            for idust in range(NDUST):
                OPT[:,0] +=  ABU[:,idust] * AFABS[idust][IFREQ]
                OPT[:,1] +=  ABU[:,idust] * AFSCA[idust][IFREQ]
            if (USER.OPT_IS_HALF):
                cl.enqueue_copy(commands[ID], OPT_buf, asarray(OPT, np.float16))
            else:
                cl.enqueue_copy(commands[ID], OPT_buf, OPT)
            if (WITH_MSF):  # ABS[], SCA[] actually not needed !!
                for idust in range(NDUST):
                    ABS[idust] = AFABS[idust][IFREQ]
                    SCA[idust] = AFSCA[idust][IFREQ]
        else:               # constant abundance ->  ABS, SCA, G are scalars
            ABS[0], SCA[0] = 0.0, 0.0
            for idust in range(NDUST):
                ABS[0] += AFABS[idust][IFREQ]
                SCA[0] += AFSCA[idust][IFREQ]
        cl.enqueue_copy(commands[ID], ABS_buf[ID], ABS)
        cl.enqueue_copy(commands[ID], SCA_buf[ID], SCA)
        
        # Upload to device new DSC, CSC, possibly also EMIT
        t0 = time.time()
        commands[ID].finish()
        if (WITH_MSF==0):
            cl.enqueue_copy(commands[ID], DSC_buf[ID], FDSC[0,IFREQ,:])  
            cl.enqueue_copy(commands[ID], CSC_buf[ID], FCSC[0,IFREQ,:])
        else:
            for idust in range(NDUST):
                DSC[idust,:] = FDSC[idust, IFREQ, :]   # FDSC[idust,ifreq,bins]
                CSC[idust,:] = FCSC[idust, IFREQ, :]
            cl.enqueue_copy(commands[ID], DSC_buf[ID], DSC)
            cl.enqueue_copy(commands[ID], CSC_buf[ID], CSC)
        commands[ID].finish()                
        Tpush += time.time()-t0
        EMIT[:]  =  EMITTED[:, IFREQ-REMIT_I1]    # total emission
        for level in range(LEVELS):
            coeff      = 1.0e-20*USER.GL*PARSEC/(8.0**level)  #  cell volume in units of GL^3
            a, b       = OFF[level], OFF[level]+LCELLS[level]
            EMIT[a:b] *= coeff*DENS[a:b]
        EMIT[nonzero(DENS<1.0e-10)] = 0.0    # empty cells emit nothing
        # EMIT <-   TRUE - OEMITTED
        cl.enqueue_copy(commands[ID], EMIT_buf[ID], EMIT)
        if (USER.SEED>0): seed = fmod(USER.SEED+(DEVICES*IFREQ+ID)*SEED1, 1.0)
        else:             seed = rand()
        commands[ID].finish()
        t0 = time.time()        
        # A single call to the kernel
        kernel_CL(commands[ID], [GLOBAL,], [LOCAL,],
        np.int32(2),  CLPAC, BATCH, seed, ABS_buf[ID], SCA_buf[ID],
        LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
        DSC_buf[ID], CSC_buf[ID], 
        NODIR, ODIR_buf[ID], USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, RA_buf[ID], DE_buf[ID], OUT_buf[ID],
        OPT_buf[ID], ABU_buf[ID])
        commands[ID].finish()
        Tkernel += time.time()-t0

        # pull OUT and add to mmap file
        cl.enqueue_copy(commands[ID], OUT, OUT_buf[ID])
        commands[ID].finish()
        OUTCOMING[IFREQ,:,:,:] += OUT    # OUTCOMING[NFREQ, NODIR, NPIX.y, NPIX.x]
        
    # end of -- for IFREQ
    
    

# Scale final OUTCOMING into surface brightness
#   *= frequency
#   *= 1.0e23 * PLANCK/(DX*DX*SCALE)
SCALE = 1.0
for IFREQ in range(NFREQ):
    k =  FFREQ[IFREQ] * 1.0e23 * PLANCK / ( USER.MAP_DX*USER.MAP_DX*SCALE )
    OUTCOMING[IFREQ, :, :, :] *= k
        
del OUTCOMING


