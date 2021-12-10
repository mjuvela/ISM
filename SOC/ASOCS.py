#!/usr/bin/env python
import os, sys

#  HOMEDIR = os.path.expanduser('~/')
#  sys.path.append(HOMEDIR+'/starformation/SOC/')
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)

from   ASOC_aux import *
import multiprocessing as mp


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


if (USER.INTOBS['s0']>-10000.0):  # output map is a healpix map
    ODIR    =   zeros(1, clarray.cltypes.float3)    
    ODIR[0] =   USER.INTOBS 
    NDIR    =  -USER.OUT_NSIDE     # output scattered map is healpix map with NSIDE == - NODIR
    RA, DE  =   zeros(2, float32), zeros(2, float32)  # dummy
else:
    NDIR, ODIR, RA, DE =  set_observer_directions(USER)        #  ODIR[NDIR, clfloat3]
ABU                 =  read_abundances(CELLS, NDUST, USER)     # [CELLS, NDUST] or []
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

LOCAL    = 8
GLOBAL_0 = 65536
if (USER.DEVICES.find('c')<0): LOCAL    = 32
if ('local'  in USER.KEYS):    LOCAL    = int(USER.KEYS['local'][0])
if ('global' in USER.KEYS):    GLOBAL_0 = int(USER.KEYS['global'][0])
GLOBAL_0 = Fix(GLOBAL_0, 32*LOCAL)
    
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
print('PACKETS: PSPAC %d   BGPAC %d  CLPAC %d  DFPAC %d  --- GLOBAL %d LOCAL %d' % (PSPAC, BGPAC, CLPAC, DFPAC, GLOBAL_0, LOCAL))
print('*'*80)
asarray([BGPAC, PSPAC, DFPAC, CLPAC], np.int32).tofile('packet.info')

# Precalculate some things for point sources located outside the model volume
XPS_NSIDE, XPS_SIDE, XPS_AREA = AnalyseExternalPointSources(NX, NY, NZ, USER.PSPOS, int(USER.NO_PS), int(USER.PS_METHOD))

HPBG = []
if (len(USER.file_hpbg)>2):
    # We use healpix maps for the background sky intensity... 
    #   currently *fixed* at NSIDE=64 which gives ~55 arcmin pixels, 49152 pixels on the sky
    HPBG = np.fromfile(USER.file_hpbg, np.float32).reshape(NFREQ, 49152)
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
-D AXY=%.5ff -D AXZ=%.5ff -D AYZ=%.5ff -D LEVELS=%d -D LENGTH=%.5ef -I%s \
-D POLSTAT=%d -D SW_A=%.3ef -D SW_B=%.3ef -D STEP_WEIGHT=%d -D DIR_WEIGHT=%d -D DW_A=%.3ef \
-D LEVEL_THRESHOLD=%d -D POLRED=%d -D WITH_COLDEN=%d -D MINLOS=%.3ef -D MAXLOS=%.3ef \
-D FFS=%d -D BG_METHOD=%d -D USE_EMWEIGHT=%d -D HPBG_WEIGHTED=%d -D WITH_MSF=%d -D NDUST=%d \
-D OPT_IS_HALF=%d -D WITH_ROI_LOAD=%d -D ROI_NSIDE=%d" % \
(  NX, NY, NZ, USER.DSC_BINS, USER.WITH_ALI, USER.PS_METHOD,
   CELLS, int(USER.AREA), max([1,int(USER.NO_PS)]), USER.kernel_defs, WITH_ABU, FACTOR, 
   USER.AXY, USER.AXZ, USER.AYZ, LEVELS, USER.GL*PARSEC, INSTALL_DIR,
   USER.POLSTAT, int(USER.STEP_WEIGHT[0]), USER.STEP_WEIGHT[1], int(USER.STEP_WEIGHT[2]), 
   int(USER.DIR_WEIGHT[0]), USER.DIR_WEIGHT[1],
   USER.LEVEL_THRESHOLD, len(USER.file_polred)>0, len(USER.file_savetau)>1, USER.MINLOS, USER.MAXLOS,
   USER.FFS, USER.BG_METHOD, USER.USE_EMWEIGHT, USER.HPBG_WEIGHTED, WITH_MSF, NDUST,
   USER.OPT_IS_HALF, USER.WITH_ROI_LOAD, USER.ROI_NSIDE)
   
### NVARGS  = " -cl-fast-relaxed-math"  ---- THIS WILL NOT WORK WITH NVIDIA, SMALL FLOAT ROUNDED TO ZERO !!!!
### ARGS   += " -cl-mad-enable -cl-no-signed-zeros -cl-finite-math-only"  # this seems ok on NVidia !!
### note: do we  need signed zeros (-0 is the link to first subcell on next level but we test n<eps, not n<=0)
ARGS   += " -cl-mad-enable -cl-finite-math-only"
print(ARGS)

# Create contexts, command queue, and program = kernels for the simulation step
source            =  INSTALL_DIR+"/kernel_ASOC_sca.c"        
context, commands =  opencl_init(USER, 1)   # context and commands are both arrays [DEVICES]
program           =  get_program(context, commands, source, ARGS) # program in also an array [DEVICES]
mf                =  cl.mem_flags
    
# A set of buffers needed by emission calculations
LCELLS_buf, OFF_buf, PAR_buf, DENS_buf, EMIT_buf, DSC_buf, CSC_buf = [], [], [], [], [], [], []
TABS_buf, RA_buf, DE_buf, ODIR_buf, PS_buf, PSPOS_buf = [], [], [], [], [], []
EMINDEX_buf, OUT_buf = [], []
OPT_buf, HPBG_buf, HPBGP_buf = [], [], []
if (NDIR<0):  # healpix output map
    print("Healpix OUT .... NDIR=%d" % NDIR)
    OUT = zeros(12*NDIR*NDIR, np.float32)   #  NDIR == -NSIDE of the output map
else:
    OUT = zeros((NDIR, USER.NPIX['y'], USER.NPIX['x']), np.float32)
print("OUT HAS THE SIZE OF:", OUT.shape)




ROI_DIM_buf  = None  # ROI_LOAD
ROI_LOAD_buf = None  # ROI_LOAD
ROI_LOAD     = []
if (USER.WITH_ROI_LOAD):   # ****  WITH_ROI_LOAD   -->    (ROI_DIM, ROI_SAVE)
    tmp             =  fromfile(USER.FILE_ROI_LOAD, int32, 5)  # (nx, ny, nz, nside, nfreq)
    if (tmp[3]!=USER.ROI_NSIDE):
        print("ROI file %s has nside %d, ini-file has %d" % (USER.FILE_ROI_LOAD, tmp[3], USER.ROI_NSIDE))
        sys.exit()
    if (tmp[4]!=USER.NFREQ):
        print("ROI file %s has %d, current run %d frequencies" % (USER.FILE_ROI_LOAD, tmp[3], USER.NFREQ))
        sys.exit()
    ROI_LOAD_NELEM  =  tmp[0]*tmp[1] + tmp[1]*tmp[2] + tmp[2]*tmp[0]  # number of surface elements
    ROI_LOAD        =  np.memmap(USER.FILE_ROI_LOAD, dtype='float32', mode='r', offset=20,
    shape=(USER.NFREQ, ROI_LOAD_NELEM * 12*USER.ROI_NSIDE*USER.ROI_NSIDE))
    ROI_DIM         =  asarray(tmp[0:3], int32) # nx, ny, nz in the ROI file
    ROI_DIM_buf     =  cl.Buffer(context[0], mf.READ_ONLY, 4*3)        
    cl.enqueue_copy(commands[0], ROI_DIM_buf, ROI_DIM)             # ---- NX, NY, NZ ---
    ROI_LOAD_NPIX   =  ROI_LOAD_NELEM * 12*USER.ROI_NSIDE*USER.ROI_NSIDE
    ROI_LOAD_buf    =  cl.Buffer(context[0], mf.READ_ONLY, 4*ROI_LOAD_NPIX)
else:
    ROI_DIM_buf     =  cl.Buffer(context[0], mf.READ_ONLY, 4)  # dummy
    ROI_LOAD_buf    =  cl.Buffer(context[0], mf.READ_ONLY, 4)  # dummy



for ID in range(DEVICES):
    # print("ID %d,  HPBG %d  x %d" % (ID, len(HPBG), len(HPBG[0])))
    LCELLS_buf.append( cl.Buffer(context[ID], mf.READ_ONLY,  LCELLS.nbytes))
    OFF_buf.append(    cl.Buffer(context[ID], mf.READ_ONLY,  OFF.nbytes))
    DENS_buf.append(   cl.Buffer(context[ID], mf.READ_ONLY,  DENS.nbytes))
    ODIR_buf.append(   cl.Buffer(context[ID], mf.READ_ONLY,  ODIR.nbytes)) # [NDIR] or single float3 for healpix
    RA_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  RA.nbytes))
    DE_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  DE.nbytes))
    PAR_buf.append(    cl.Buffer(context[ID], mf.READ_WRITE, 4*CELLS))
    EMIT_buf.append(   cl.Buffer(context[ID], mf.READ_ONLY,  4*CELLS))
    PS_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  4*max([1,USER.NO_PS])))
    PSPOS_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  USER.PSPOS.nbytes))
    OUT_buf.append(    cl.Buffer(context[ID], mf.READ_WRITE, OUT.nbytes))   # normal or healpix outputs
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
        OPT_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*3))  # DUMMY
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
        XPS_NSIDE_buf.append(    cl.Buffer(context[ID], mf.READ_ONLY,  4*max([1,  USER.NO_PS])))
        XPS_SIDE_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  4*max([1,3*USER.NO_PS])))
        XPS_AREA_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  4*max([1,3*USER.NO_PS])))
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
        cl.enqueue_copy(commands[ID], PSPOS_buf[ID],     USER.PSPOS)
        cl.enqueue_copy(commands[ID], XPS_NSIDE_buf[ID], XPS_NSIDE)  #  NO_PS
        cl.enqueue_copy(commands[ID], XPS_SIDE_buf[ID],  XPS_SIDE)   #  NO_PS * 3
        cl.enqueue_copy(commands[ID], XPS_AREA_buf[ID],  XPS_AREA)   #  NO_PS * 3
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

kernel_PS       = program[ID].SimRAM_PS     # 2021-04-22


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
# NDIR       ODIR,   NPIX                  MAP_DX       CENTRE                  ORA   ODE   OUT  
  np.int32,  None,   clarray.cltypes.int2, np.float32,  clarray.cltypes.float3, None, None, None,
# 23      24      25           26          27      
# ABU     OPT     XPS_NSIDE    XPS_SIDE    XPS_AREA
None,     None,   None,        None,       None,   
# 28          29       
# ROI_DIM     ROI_LOAD 
None,         None
])
    

kernel_PS.set_scalar_arg_dtypes(
#   0      1          2         3           4         
# PACKETS    BATCH     SEED        ABS         SCA    
[ np.int32,  np.int32, np.float32, None,       None,
# 5         6            7    
# BG        PSPOS        PS   
np.float32, None,        None,
# 8       9     10    11     12     13  
# LCELLS  OFF   PAR   DENS   DSC    CSC 
None,     None, None, None,  None,  None,  
# 14         15      16                    17           18                      19    20    21   
# NDIR       ODIR,   NPIX                  MAP_DX       CENTRE                  ORA   ODE   OUT  
  np.int32,  None,   clarray.cltypes.int2, np.float32,  clarray.cltypes.float3, None, None, None,
# 22      23      24           25          26      
# ABU     OPT     XPS_NSIDE    XPS_SIDE    XPS_AREA
None,     None,   None,        None,       None   
])
    

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
# LCELLS  OFF   PAR   DENS   EMIT   DSC   CSC    NDIR      ODIRS 
None,     None, None, None,  None,  None, None,  np.int32, None,    
# 15                  16           17                      18    19    20  
# NPIX                MAP_DX       MAPCENTRE               ORA   ODE   OUT 
clarray.cltypes.int2, np.float32,  clarray.cltypes.float3, None, None, None,
#  21    22     23   
#  OPT   ABU    EMWEI
None,    None,  None ])

kernel_zero_out.set_scalar_arg_dtypes([np.int32, clarray.cltypes.int2, None])
    
    
# Initialise memory mapped file for outcoming photons
if (NDIR>0):  # normal orthographic maps
    if ((USER.FITS>0)&(NDIR==1)): # write FITS instead of plain binary file
        OUTCOMING = zeros((USER.NFREQ, NDIR, USER.NPIX['y'], USER.NPIX['x']), float32)
    else:
        fp = open('outcoming.socs', 'w')
        asarray([USER.NPIX['y'], USER.NPIX['x'], USER.NFREQ], np.int32).tofile(fp)
        asarray(FFREQ, np.float32).tofile(fp)
        fp.close()
        OUTCOMING  =  np.memmap('outcoming.socs',dtype='float32',mode="r+",
                      shape=(USER.NFREQ, NDIR, USER.NPIX['y'], USER.NPIX['x']),offset=4*(3+NFREQ))
        OUTCOMING[:,:,:,:] = 0.0   #  [NFREQ, NDIR, NPIX.y, NPIX.x]
    print("OUTCOMING[NFREQ=%d,NDIR=%d,NPIX.y=%d,NPIX.x=%d]" % (USER.NFREQ, NDIR, USER.NPIX['x'], USER.NPIX['y']))
else:
    fp = open('outcoming.socs', 'w')
    asarray([-NDIR, USER.NFREQ], np.int32).tofile(fp)
    asarray(FFREQ, np.float32).tofile(fp)
    fp.close()
    OUTCOMING  = \
    np.memmap('outcoming.socs',dtype='float32',mode="r+",
    shape=(USER.NFREQ, 12*NDIR*NDIR),offset=4*(2+NFREQ))
    OUTCOMING[:,:] = 0.0   #  [NFREQ, IPIX]
    print("OUTCOMING[NFREQ=%d,NPIX=%d]" % (USER.NFREQ, 12*NDIR*NDIR))


BG, PS = 0.0, 0.0  # photons per package for background and point source
# Simulate the constant radiation sources with PSPAC and BGPAC packages
# the resulting TABS array is stored to CTABS and saved to disk
# CLPAC = constant diffuse emission
#    do DFPAC here, separate of CLPAC, to avoid complications with ALI
for II in range(4):  # loop over PSPAC, BGPAC, DFPAC, ROIPAC simulations
    print("CONSTANT SOURCES II=%d" % II)
    
    if (II==0):   # point source
        GLOBAL   =  GLOBAL_0
        if ((PSPAC<1)|(USER.NO_PS<1)): 
            continue   # point sources not simulated
        if (0):  # [2021-10-30] each work item works on one source ... slower than the alternative below !!
            #  PSPAC ~ total number of packages per sources
            #  each work item does only one source
            #  BATCH == 256 and number of work items is set    GLOBAL/NO_PS*BATCH ~ PSPAC
            BATCH   =  256                          # photon packages per work item
            PSPAC   =  Fix(USER.PSPAC, BATCH)       # packages per source
            GLOBAL  =  USER.NO_PS*PSPAC//BATCH      # number of work items used, PSPAC//BATCH wi needed per source
            WPS     =  1.0 / (PLANCK*PSPAC*((USER.GL*PARSEC)**2.0)) # weight using packages per source = PSPAC
            PACKETS =  GLOBAL*BATCH                 # total number of photon packages == NO_PS*PSPAC
            GLOBAL  =  Fix(GLOBAL, 32)              # actual number of work items used is PACKETS/BATCH
            print("%d = %d" % (GLOBAL*BATCH, USER.NO_PS*PSPAC))
        else:
            # every work item loops over all sources, simulating BATCH packages per source
            #   PSPAC = packages per source = GLOBAL*BATCH, since EVERY work item does EVERY source
            BATCH    =  int(max([1, PSPAC / GLOBAL]))  # packages per source, per work item
            PACKETS  =  GLOBAL * BATCH                 # packages per source -- kernel ignores PACKETS
            WPS      =  1.0 / (PLANCK*PACKETS*((USER.GL*PARSEC)**2.0))
            # above BATCH = packages per source and per work item
            # change that to total number of packages per work item => multiply by number of sources
            BATCH   *=  USER.NO_PS        # one work item does BATCH packages, looping over all sources
            PACKETS  =  GLOBAL*BATCH      # total number of photon packages simulated
        print("=== PS: GLOBAL %d, LOCAL %d, NO_PS %d, BATCH %d, PSPAC %.2e, PACKETS %.2e" % 
        (GLOBAL, LOCAL, USER.NO_PS, BATCH, PSPAC, PACKETS))
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
    elif (II==2):
        # RAM2 version --- each work item loops over cells =>  GLOBAL<CELLS, BATCH==packets per cell
        GLOBAL   =  GLOBAL_0
        if (len(DIFFUSERAD)<1): continue  # no diffuse radiation sources            
        if (DFPAC<1): continue            # ??? THIS MAY BE >0 EVEN WHEN NO DIFFUSE FIELD USED ??
        BATCH    =  int(DFPAC/CELLS)
        PACKETS  =  DFPAC                
        print("=== DFPAC %d, GLOBAL %d, BATCH %d, LOCAL %d" % (DFPAC, GLOBAL, BATCH, LOCAL))
    elif (II==3): # ROI_LOAD
        # *** ROIPAC ***
        #   one work item per surface element = max ROIPAC work items
        #   each work item loops BATCH times over the full healpix map
        #   PHOTONS weighted by 1/BATCH
        if (USER.ROIPAC<1): continue                      # continue only for ROI_LOAD
        # each work item loops over all surface elements => BATCH=number of healpix pixels
        GLOBAL  =  100*ROI_LOAD_NELEM                     # threads = 100 x  surface elements
        BATCH   =  12*USER.ROI_NSIDE*USER.ROI_NSIDE       # this many healpix pixels
        BATCH   =  max([1, int(USER.ROIPAC/(100.0*BATCH*ROI_LOAD_NELEM))]) # work item does healpix map this many times
        BATCH  *=  12*USER.ROI_NSIDE*USER.ROI_NSIDE       # kernel gets BATCH == number of rays per surf. element = per work item
        # i.e. BATCH must be multiple >=1 of the number of healpix pixels
        GLOBAL  =  Fix(GLOBAL, LOCAL)                     # possibly increase to make divisible by LOCAL
        PACKETS =  ROI_LOAD_NELEM                         # use PACKETS only to store number of surface elements
        print("================================================================================")
        print("ROI --- GLOBAL %d, BATCH %d, ROI_LOAD_NELEM %d, ROI_LOAD_NPIX %d, PACKETS %d" % (GLOBAL, BATCH, ROI_LOAD_NELEM, ROI_LOAD_NPIX, PACKETS))
        print("================================================================================")
        

    skip = 2 
    for IFREQ in range(NFREQ):

        t00   =  time.time()
        FREQ  =  FFREQ[IFREQ]

        if ((FREQ<USER.SIM_F[0])|(FREQ>USER.SIM_F[1])):         
            print(" ... skip frequency %12.4e Hz = %8.2f um" % (FREQ, f2um(FREQ)))
            continue
        
        # print('IFREQ %d / %d' % (IFREQ, NFREQ))
        kernel_zero_out(commands[ID], [GLOBAL,], [LOCAL,], NDIR, USER.NPIX, OUT_buf[ID])
        
        
        ABS[0], SCA[0]  = AFABS[0][IFREQ], AFSCA[0][IFREQ]        
        if (WITH_ABU>0): # ABS, SCA, G are vectors,  OPT[CELLS, 2]
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
            # OPT not needed... unless kernel uses OPT instead of ABS and SCA....
            # !!! OPT not defined unless WITH_ABU>0
        cl.enqueue_copy(commands[ID], ABS_buf[ID], ABS)
        cl.enqueue_copy(commands[ID], SCA_buf[ID], SCA)
                
        BG  =  0.0
        if (len(IBG)==NFREQ):  BG = IBG[IFREQ] * WBG/FREQ    # background photons per package
        if (II==0): # update point source luminosities for the current frequency
            PS = LPS[:,IFREQ] * WPS/FREQ  # source photons per package
            cl.enqueue_copy(commands[ID], PS_buf[ID], PS)
        else:
            PS = [0.0,]

            
        if ((II==2)&(USER.USE_EMWEIGHT>0)):
            print("UPDATE EMWEI")
            skip += 1 
            if (skip==3):
                skip = 0
                tmp      =  asarray(EMITTED[:,IFREQ-REMIT_I1].copy(), float64)
                if (1):
                    mmm      =  nonzero(~isfinite(tmp))
                    print('='*80)
                    print('EMWEI --- EMISSION NOT FINITE: %d'% len(mmm[0]))
                    tmp[mmm] = 0.0
                    print('='*80)
                tmp[:]   =  CLPAC*tmp/(sum(tmp)+1.0e-65)  # ~ number of packages
                ## print 'EMWEI   ', np.percentile(tmp, (0.0, 10.0, 90.0, 100.0))
                EMWEI[:] =  clip(tmp, USER.EMWEIGHT_LIM[0], USER.EMWEIGHT_LIM[1])
                # any EMWEI<1.0 means it has probability EMWEI of being simulated batch=1
                #   which means the weight is 1/p = 1/EMWEI
                EMWEI[nonzero(rand(CELLS)>EMWEI)] = 0.0 # Russian roulette
                ## print 'REMAIN ', len(nonzero(EMWEI>1e-7)[0])
                if (USER.EMWEIGHT_LIM[2]>0.0):
                    EMWEI[nonzero(EMWEI<USER.EMWEIGHT_LIM[2])] = 0.0 # ignore completely
                    ## print 'REMAIN ', len(nonzero(EMWEI>1e-7)[0])
                if (USER.USE_EMWEIGHT==-2):
                    # Host provides weights in EMWEI and the indices of emitting
                    #    cells in EMINDEX
                    # Host makes number of packages divisible by 3 and makes several
                    # calls, host keeps count of the simulated packages while kernel
                    # always simulates 3 packages for any cell in the current EMINDEX
                    EMPAC    = asarray(EMWEI2_STEP*np.round(tmp/EMWEI2_STEP), int32)  # packages per cell
                    EMWEI[:] = 1.0/(EMPAC[:]+1e-10)                      # packet weigth
                    ## print 'EMWEI', EMWEI
                    # below host loops over kernel calls
                    #  * make EMINDEX of cells with ENCOUNT>EMDONE
                    #  * send kernel EMINDEX
                    #     - kernel uses EMWEI to weight the packages, simulating 3
                    #       packages for every cell in EMINDEX
                    #  * host increases EMDONE[EMINDEX] += 3
                    #  ... until no cells left
                commands[ID].finish()
                cl.enqueue_copy(commands[ID], EMWEI_buf[ID], EMWEI)
                commands[ID].finish()
            
            
            
            
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
                print("      <HPBG> %12.4e" % mean(ravel(HPBG[IFREQ,:])))
                
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
                print("??????????   DIFFUSERAD HAS FEWER FREQUENCIES THAN OUR SIMULATION ???????????????")
                continue
            # Note -- DIFFUSERAD is emission as photons / 1Hz / cm3
            #         EMIT is number of photons for volume (GL*pc)^3 divided by (GL*pc)^2
            #         => EMIT = DIFFUSERAD * (GL*pc)
            for level in range(LEVELS):
                coeff      =  USER.GL*PARSEC / (8.0**level)    # cell volume in units of GL^3
                coeff     *=  USER.K_DIFFUSE                   # user-provided scaling of the field
                a, b       =  OFF[level], OFF[level]+LCELLS[level]
                EMIT[a:b]  =  DIFFUSERAD[a:b, IFREQ] * coeff   # DIFFUSERAD = photons/cm3
            EMIT[nonzero(DENS<1.0e-10)] = 0.0    # empty cells emit nothing
            cl.enqueue_copy(commands[ID], EMIT_buf[ID], EMIT)
            print(" EMIT_buf UPDATED !!!")
            time.sleep(5)
        commands[ID].finish()
        t0 = time.time()            
        if (II==0):      # 0=PSPAC
            if (0):
                kernel_PB(commands[ID], [GLOBAL,], [LOCAL,], np.int32(II), 
                PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], 
                BG, PSPOS_buf[ID], PS_buf[ID], 
                LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], DSC_buf[ID], CSC_buf[ID], 
                NDIR, ODIR_buf[ID], USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, 
                RA_buf[ID], DE_buf[ID], OUT_buf[ID], ABU_buf[ID], OPT_buf[ID], 
                XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID], ROI_DIM_buf, ROI_LOAD_buf)
            else:  #  2021-04-22 -- separate kernel for point source photons
                kernel_PS(commands[ID], [GLOBAL,], [LOCAL,], 
                PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], 
                BG, PSPOS_buf[ID], PS_buf[ID], 
                LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], DSC_buf[ID], CSC_buf[ID], 
                NDIR, ODIR_buf[ID], USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, 
                RA_buf[ID], DE_buf[ID], OUT_buf[ID], ABU_buf[ID], OPT_buf[ID], 
                XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID])
        elif (II==1):    # 1=BGPAC
            # we do not come here unless BGPAC>1 --- either isotropic or healpix background
            if (size(HPBG)>0):         # using a healpix map for the background
                kernel_HP(commands[ID], [GLOBAL,], [LOCAL,],
                PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], 
                LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], DSC_buf[ID], CSC_buf[ID], 
                NDIR, ODIR_buf[ID], USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, 
                RA_buf[ID], DE_buf[ID], OUT_buf[ID], ABU_buf[ID], OPT_buf[ID], 
                HPBG_buf[ID], HPBGP_buf[ID])
            else:                      #  isotropic background
                kernel_PB(commands[ID], [GLOBAL,], [LOCAL,], np.int32(II), 
                PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], 
                BG, PSPOS_buf[ID], PS_buf[ID], 
                LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], DSC_buf[ID], CSC_buf[ID], 
                NDIR, ODIR_buf[ID], USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, 
                RA_buf[ID], DE_buf[ID], OUT_buf[ID], ABU_buf[ID], OPT_buf[ID], 
                XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID], ROI_DIM_buf, ROI_LOAD_buf)
        elif (II==2):    # 2=DFPAC     --- the same kernel as for CLPAC !!
            print(" DFPACK KERNEL CALL !!!")
            time.sleep(5)
            kernel_CL(commands[ID], [GLOBAL,], [LOCAL,],
            np.int32(II), PACKETS, BATCH, seed, ABS_buf[ID],
            SCA_buf[ID], LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], 
            EMIT_buf[ID], DSC_buf[ID], CSC_buf[ID], NDIR, ODIR_buf[ID], 
            USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, RA_buf[ID], DE_buf[ID], 
            OUT_buf[ID], OPT_buf[ID], ABU_buf[ID], EMWEI_buf[ID])
        elif (II==3):    # ROI_LOAD
            if (USER.WITH_ROI_LOAD<=0): continue
            cl.enqueue_copy(commands[0], ROI_LOAD_buf, asarray(ROI_LOAD[IFREQ,:]*USER.ROI_LOAD_SCALE/(USER.GL*USER.GL), float32))
            ## ROI_LOAD
            kernel_PB(commands[ID], [GLOBAL,], [LOCAL,], np.int32(II), 
            PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], 
            BG, PSPOS_buf[ID], PS_buf[ID], 
            LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], DSC_buf[ID], CSC_buf[ID], 
            NDIR, ODIR_buf[ID], USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, 
            RA_buf[ID], DE_buf[ID], OUT_buf[ID], ABU_buf[ID], OPT_buf[ID], 
            XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID], ROI_DIM_buf, ROI_LOAD_buf)
            
            
        commands[ID].finish()                
        Tkernel += time.time()-t0 # REALLY ONLY THE KERNEL EXECUTION TIME
        
        # pull OUT and add to mmap file
        cl.enqueue_copy(commands[ID], OUT, OUT_buf[ID])
        commands[ID].finish()
        if (NDIR>0):
            OUTCOMING[IFREQ,:,:,:] += OUT    # OUTCOMING[NFREQ, NDIR, NPIX.y, NPIX.x]
        else:
            OUTCOMING[IFREQ,:]     += OUT    # healpix
            
        sys.stdout.write("  == %.1f sec\n" % (time.time()-t00))
        # print
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
    print(" CLPACK KERNEL CALL !!!")
    time.sleep(5)
    skip = 2
    for IFREQ in range(NFREQ):     # loop over single frequencies (all the frequencies)

        kernel_zero_out(commands[ID],[GLOBAL,],[LOCAL,], NDIR, USER.NPIX, OUT_buf[ID])
        commands[ID].finish()
        # Parameters for the current frequency
        FREQ  =  FFREQ[IFREQ]
        if ((FREQ<USER.SIM_F[0])|(FREQ>USER.SIM_F[1])): 
            print(" ... skip frequency %12.4e Hz = %8.2f um" % (FREQ, f2um(FREQ)))
            continue # FREQ not simulayted!!
        
        ABS[0], SCA[0]  = AFABS[0][IFREQ], AFSCA[0][IFREQ]        
        print("    FREQ %3d/%3d   %12.4e --  ABS %.3e  SCA %.3e" % (IFREQ+1, NFREQ, FREQ, ABS[0], SCA[0]))
        if (WITH_ABU>0): # ABS, SCA, G are vectors
            OPT[:,:] = 0.0
            for idust in range(NDUST):
                OPT[:,0] +=  ABU[:,idust] * AFABS[idust][IFREQ]
                OPT[:,1] +=  ABU[:,idust] * AFSCA[idust][IFREQ]
            if (USER.OPT_IS_HALF):
                cl.enqueue_copy(commands[ID], OPT_buf[ID], asarray(OPT, np.float16))
            else:
                cl.enqueue_copy(commands[ID], OPT_buf[ID], OPT)
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
        
        
        if (USER.USE_EMWEIGHT>0):
            skip += 1 
            if (skip==3):
                print("UPDATE EMWEI")
                skip = 0
                tmp      =  asarray(EMITTED[:,IFREQ-REMIT_I1].copy(), float64)
                if (1):
                    mmm      =  nonzero(~isfinite(tmp))
                    print('='*80)
                    print('EMWEI --- EMISSION NOT FINITE: %d'% len(mmm[0]))
                    tmp[mmm] = 0.0
                    print('='*80)
                tmp[:]   =  CLPAC*tmp/(sum(tmp)+1.0e-65)  # ~ number of packages
                ## print 'EMWEI   ', np.percentile(tmp, (0.0, 10.0, 90.0, 100.0))
                EMWEI[:] =  clip(tmp, USER.EMWEIGHT_LIM[0], USER.EMWEIGHT_LIM[1])
                # any EMWEI<1.0 means it has probability EMWEI of being simulated batch=1
                #   which means the weight is 1/p = 1/EMWEI
                EMWEI[nonzero(rand(CELLS)>EMWEI)] = 0.0 # Russian roulette
                ## print 'REMAIN ', len(nonzero(EMWEI>1e-7)[0])
                if (USER.EMWEIGHT_LIM[2]>0.0):
                    EMWEI[nonzero(EMWEI<USER.EMWEIGHT_LIM[2])] = 0.0 # ignore completely
                    ## print 'REMAIN ', len(nonzero(EMWEI>1e-7)[0])
                if (USER.USE_EMWEIGHT==-2):
                    # Host provides weights in EMWEI and the indices of emitting
                    #    cells in EMINDEX
                    # Host makes number of packages divisible by 3 and makes several
                    # calls, host keeps count of the simulated packages while kernel
                    # always simulates 3 packages for any cell in the current EMINDEX
                    EMPAC    = asarray(EMWEI2_STEP*np.round(tmp/EMWEI2_STEP), int32)  # packages per cell
                    EMWEI[:] = 1.0/(EMPAC[:]+1e-10)                      # packet weigth
                    ## print 'EMWEI', EMWEI
                    # below host loops over kernel calls
                    #  * make EMINDEX of cells with ENCOUNT>EMDONE
                    #  * send kernel EMINDEX
                    #     - kernel uses EMWEI to weight the packages, simulating 3
                    #       packages for every cell in EMINDEX
                    #  * host increases EMDONE[EMINDEX] += 3
                    #  ... until no cells left
                commands[ID].finish()
                cl.enqueue_copy(commands[ID], EMWEI_buf[ID], EMWEI)
                commands[ID].finish()
        
        
        if (USER.SEED>0): seed = fmod(USER.SEED+(DEVICES*IFREQ+ID)*SEED1, 1.0)
        else:             seed = rand()
        commands[ID].finish()
        t0 = time.time()        
        # A single call to the kernel
        kernel_CL(commands[ID], [GLOBAL,], [LOCAL,],
        np.int32(2),  CLPAC, BATCH, seed, ABS_buf[ID], SCA_buf[ID],
        LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
        DSC_buf[ID], CSC_buf[ID], 
        NDIR, ODIR_buf[ID], USER.NPIX, USER.MAP_DX, USER.MAPCENTRE, RA_buf[ID], DE_buf[ID], OUT_buf[ID],
        OPT_buf[ID], ABU_buf[ID], EMWEI_buf[ID])
        commands[ID].finish()
        Tkernel += time.time()-t0

        # pull OUT and add to mmap file
        cl.enqueue_copy(commands[ID], OUT, OUT_buf[ID])
        commands[ID].finish()
        if (NDIR>0):
            OUTCOMING[IFREQ,:,:,:] += OUT    # OUTCOMING[NFREQ, NDIR, NPIX.y, NPIX.x]
        else:
            OUTCOMING[IFREQ,:]     += OUT    # healpix
        
    # end of -- for IFREQ
    
    

# Scale final OUTCOMING into surface brightness
#   *= frequency
#   *= 1.0e23 * PLANCK/(DX*DX*SCALE)
SCALE = 1.0
for IFREQ in range(NFREQ):
    if (NDIR>0):   # orthographic map
        k =  FFREQ[IFREQ] * 1.0e23 * PLANCK / ( USER.MAP_DX*USER.MAP_DX*SCALE )
        OUTCOMING[IFREQ, :, :, :] *= k
        print("< OUTCOMING[%3d] > = %10.3e" % (IFREQ, mean(OUTCOMING[IFREQ, :, :, :])))
    else:          # healpix map
        k =  FFREQ[IFREQ] * 1.0e23 * PLANCK  / ( 4.0*pi/(12.0*NDIR*NDIR) )        
        OUTCOMING[IFREQ, :]       *= k  ## /home/mika/bin/ASOCS.py:706: RuntimeWarning: overflow encountered in multiply

if ((NDIR==1)&(USER.FITS>0)):
    # OUTCOMING is a regular array [nfreq, ndir, y, x], save it as a FITS file
    # use nominal 1 kpc distance to set the pixel size... unless USER.DISTANCE is set
    # FITS      =  MakeFits(USER.FITS_RA, USER.FITS_DE, USER.GL/[1000.0, USER.DISTANCE][USER.DISTANCE>0.0], USER.NPIX['x'], USER.NPIX['y'], FFREQ)
    # pixel size  = GL*MAP_DX/distance
    FITS      =  MakeFits(USER.FITS_RA, USER.FITS_DE, USER.GL*USER.MAP_DX/[1000.0, USER.DISTANCE][USER.DISTANCE>0.0], USER.NPIX['x'], USER.NPIX['y'], FFREQ)
    for ifreq in range(USER.NFREQ):
        FITS[0].data[ifreq,:,:] =  OUTCOMING[ifreq, 0, :, :] # NDIR==1
    FITS.verify('fix')
    FITS.writeto('%s.fits' % USER.file_scattering, overwrite=True)
    del FITS
    
del OUTCOMING  # this saves mmap file, if exists



