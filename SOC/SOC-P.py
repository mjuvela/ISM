#!/usr/bin/env python

import os, sys

# HOMEDIR      = os.path.expanduser('~/')
# sys.path.append(HOMEDIR+'/starformation/SOC/')
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)
    
from   ASOC_aux import *    
import multiprocessing as mp


"""
Single dust species:
    -  OPT and ABU not used ... *** OPT is used if an abundance file is specified
    -  kernel has scalar ABS, SCA
    -  single scattering function defined by DSC and CSC arrays
    
Multiple dust species with variable abundances but a single scattering function:
    -D WITH_ABU:
        -  OPT used instead of scalar ABS, SCA; precalculated by the host for all cells
        -  ABU not used => single scattering function as defined by DSC and CSC

Multiple dust species with variable abundances, different scattering functions:
    -D WITH_ABU:
        - cross sections still precalculated by the host into the kernel parameter OPT
    -D WITH_MSF:
        - ABS, SCA are vectors = separate values for each dust component (needed ONLY for scattering events)
        - ABU[CELLS,NDUST] = abundance for each species (needed ONLY for scattering events)
        
Memory requirements for the kernel:
    - basic run           ->   PAR[CELLS], DENS[CELLS], EMIT[CELLS], TABS[CELLS]
    - SAVE_INTENSITY==1   ->   INT[CELLS],
      SAVE_INTENSITY==2   ->   INT[CELLS], INTX[CELLS], INTY[CELLS], INTZ[CELLS]
    - WITH_ABU>0          ->   OPT[2*CELLS]
    - WITH_MSF            ->   ABU[CELLS, NDUST]
    
Basic run with 480e6 cells
      PAR, DENSE, EMIT, TABS      =  4 x CELLS  =   7.2 GB
The same with variable abundances:      
      PAR, DENSE, EMIT, TABS, OPT =  6 x CELLS  =  10.7 GB
      
2019-02-19:
    Drop root grid cells from PAR_buf (top level cells have no parents), ~0.5 GB for 400e6 cell model!
    
2019-03-20:
    Consider (not implemented!) replacing simum, remit, wavelengths with masks specified in the optical file:
        - simum = wavelengths used in the simulation
        - remit = wavelengths used in simulations of dust emission = wavelengths saved in emitted file
        - wavelengths = wavelengths included in the output spectrum files
    Also, all files (emitted, spectra) should include the list of the actual wavelengths used.
    
2019-04-14:
    Added ROI_MAP option to Mapping and HealpixMapping = to include in the maps only the emission from ROI
2019-04-15:
    Added USER.FILE_ROI_SAVE,  USER.ROI_STEP  to save the photons entering ROI
2019-04-18:
    Added FACTOR, replacing the constant scaling by 1e20
2019-08-27:
    Added fselect keyword
2019-10-01:
    Added new options for work with A2E_LIB (dust emission solved with the library method).
    libabs  => save absorptions for a subset of frequencies and stop there
    libmaps => load smaller emitted file with a subset of frequencies, write maps for those
"""



EMWEI2_STEP = 100
#if 0
ADHOC       = 1.0e-10
#else
ADHOC       = 1.0
#endif


# TODO -- looks like in PS simulation the single CPU thread is working at 100%
#         -> need to keep batch large enough to lighten the load on the host side?
GLOBAL_0    = 32768
# GLOBAL_0 = 65536

if (len(sys.argv)<2):  
    print("Usage:  SOC ini-file")
    if (os.path.exists('soc_example.ini')):
        print("A sample inifile already exists in soc_example.ini")
    else:
        WriteSampleInit('soc_example.ini')
        print("A sample inifile has been written to soc_example.ini")
    sys.exit()
USER = User(sys.argv[1])
if (not(USER.Validate())):   
    print("Check the inifile... exiting!"),  sys.exit()

# Read optical data for the dust model, updates USER.NFREQ
FFREQ, AFG, AFABS, AFSCA = read_dust(USER) # arrays containing parameters possibly for multiple dust species
NFREQ             =  USER.NFREQ
NDUST             =  len(AFABS)
# Use ONFREQ for the actual number of frequencies in the absorption and emission files
# In case of library method, ONFREQ<NFREQ and ONFREQ contains the reference frequencies (LIB_ABS==True)
# or the emitted/map frequencies (LIB_MAPS=True) -- only one of those can be true
# Note: F_SIM can be used to limit the number of the simulated frequencies but that does not affect the
#       number of frequencies in the absorption and emission arrays
ONFREQ            =  NFREQ
if ((USER.LIB_ABS)|(USER.LIB_MAPS)):
    ONFREQ        =  len(USER.FSELECT)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!  ONFREQ = %3d   <   NFREQ = %3d  !!!!!" % (ONFREQ, NFREQ))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
if (USER.LIB_MAPS):
    USER.ITERATIONS  = 0
    USER.NOSOLVE     = 1
    USER.FAST_MAP    = 0
    USER.MAP_FREQ[0] = 1.0e-10
    USER.MAP_FREQ[1] = 1.0e30
    
# Read scattering functions:  FDSC[ndust, NFREQ, BINS], FCSC[ndust, NFREQ,BINS]
#  ... where ndust == 1 or NDUST, the latter implies WITH_MSF 
FDSC, FCSC        =  read_scattering_functions(USER)
WITH_MSF          =  len(FDSC)>1
if (WITH_MSF==0):
    DSC, CSC      =  None, None
else:
    DSC, CSC      =  zeros((NDUST, USER.DSC_BINS), np.float32), zeros((NDUST, USER.DSC_BINS), np.float32)    
# Read intensities of the background and of point source
IBG               =  read_background_intensity(USER)
LPS               =  read_source_luminosities(USER)   # LPS[USER.NO_PS, USER.NFREQ]
# Read cloud file and scale densities; also updates USER.AREA/AXY/AXZ/AXZ
NX, NY, NZ, LEVELS, CELLS, LCELLS, OFF, DENS = read_cloud(USER)
# ROI dimensions (if any)
# ROI_NX, ROI_NY, ROI_NZ   =  (ROI[1]-ROI[0]+1), (ROI[3]-ROI[2]+1), (ROI[5]-ROI[4]+1)
# Read dust population abundances, either [] or ABU[CELLS,NDUST] for WITH_MSF
ABU               =  read_abundances(CELLS, NDUST, USER)

WITH_ABU          =  ABU.shape[0]>0  # even single dust with abundance file => use OPT array

if (WITH_ABU & USER.SINGLE_ABU):
    if (NDUST!=2):
        print("Option USER.SINGLE_ABU assumes exactly two dust components !!")
        sys.exit(0)
    if (WITH_MSF):
        print("Option USER.SINGLE_ABU cannot be used with WITH_MSF !!")
        sys.exit(0)
    ABU = ravel(ABU[:,0])  #  second dust has implicitly abundance 1-ABU !!

    
if (WITH_ABU):    # ABU + SCA vectors in case of non-constant dust abundances
    OPT   = zeros((CELLS, 2), float32)
else:
    if (WITH_MSF):
        print("Cannot have multiple scattering functions without multiple dusts with variable abundances")
        sys.exit()

        
if (len(sys.argv)>2): # command line overrides devices defined in the ini file
    if (int(sys.argv[2])>0):  USER.DEVICES = 'g'
    else:                     USER.DEVICES = 'c'
        
# Open file containing diffuse emission per cell
DIFFUSERAD        =  mmap_diffuserad(USER, CELLS)   # memory mapped => do not scale the array itself with K_diffuse
DEVICES           =  len(USER.DEVICES)
KDEV              =  1.0/DEVICES
KDEV              =  1.0  # here only one device !!


if (0): # testing
    print("********************************************************************************")
    USER.OBS_THETA[0], USER.OBS_PHI[0] =  0.0*DEGREE_TO_RADIAN,   0.0*DEGREE_TO_RADIAN  # ODIR=[0,0,1]
    USER.OBS_THETA[0], USER.OBS_PHI[0] =  90.0*DEGREE_TO_RADIAN,  0.0*DEGREE_TO_RADIAN  # ODIR=[1,0,0]
    USER.OBS_THETA[0], USER.OBS_PHI[0] =  90.0*DEGREE_TO_RADIAN, 90.0*DEGREE_TO_RADIAN  # ODIR=[0,1,0]
    USER.OBS_THETA[0], USER.OBS_PHI[0] =  80.0*DEGREE_TO_RADIAN, 30.0*DEGREE_TO_RADIAN  # ODIR=[0.85,0.49,0.17]
    NODIR, ODIR, RA, DE =  set_observer_directions(USER)
    print("********************************************************************************")
    sys.exit()
    

# Parameters related to the observer directions
NODIR, ODIR, RA, DE =  set_observer_directions(USER)

# User may limit output (intensities, spectra) between two frequencies
m                  = nonzero((FFREQ>=USER.REMIT_F[0])&(FFREQ<=USER.REMIT_F[1]))
REMIT_I1, REMIT_I2 = m[0][0], m[0][-1]
REMIT_NFREQ        = len(m[0])   #   REMIT_NFREQ <= NFREQ
if (USER.LIB_MAPS): # always maps for all the FSELECT frequencies (with LIB_MAPS)
    REMIT_I1, REMIT_I2 = 0, NFREQ-1
    REMIT_NFREQ        = NFREQ    # only FSELECT frequencies in the final map loop
USER.REMIT_NFREQ, USER.REMIT_I1, USER.REMIT_I2 = REMIT_NFREQ, REMIT_I1, REMIT_I2
print("Emitted data for channels [%d, %d], %d out of %d channels" % (REMIT_I1, REMIT_I2, REMIT_NFREQ, NFREQ))
print("     %.3e - %.3e Hz" % (USER.REMIT_F[0], USER.REMIT_F[1]))
# We can have REMIT_NFREQ<NFREQ only if cell emission is not included in the calculation
if ((REMIT_NFREQ<NFREQ) & ((USER.ITERATIONS>0) & (USER.CLPAC>0))):
    print("NFREQ=%d, REMIT_NFREQ=%d -- cannot be if ITERATIONS=%d, CLPAC=%d>0" %
    (NFREQ, REMIT_NFREQ, USER.ITERATIONS, CLPAC))

beta_interpoler = None
if (USER.WITH_ALI): # estimate effective escape probability beta(T, tau)
    # 2018-12-08 -- ALI is restricted to runs with a single dust (not stochastic!)
    if (len(AFABS)>1):
        print("ALI can ne used only with a single, non-stochastically-heated dust")
        sys.exit()
    beta_interpoler = calculate_beta_vs_tau_T(FFREQ, AFABS[0])

LOCAL = 8
NCPUS = 4
if (USER.DEVICES.find('c')<0): 
    LOCAL = 32

if ('local' in USER.KEYS):
    LOCAL = int(USER.KEYS['local'][0])
if (LOCAL==12): 
    NCPUS = 6
    # we must make GLOBAL0 multiple of both LOCAL and 32
    GLOBAL_0 = Fix(GLOBAL_0, 32*LOCAL)

    
PSPAC = Fix(USER.PSPAC, LOCAL)                     # any number ... but multiple of LOCAL
BGPAC = Fix(Fix(USER.BGPAC, USER.AREA), LOCAL)     # multiple of AREA and LOCAL
DFPAC = 0
if (USER.USE_EMWEIGHT>0):
    CLPAC = Fix(USER.CLPAC, LOCAL)                 # multiple of LOCAL
    if (USER.DFPAC>0):  # 2020-10-04  -- cannot have DFPAC<CELLS  ---  even when EMWEIGHT is used?
        DFPAC = Fix(USER.DFPAC, LOCAL)            # multiple of LOCAL
        if (0):
            DFPAC = Fix(max([USER.DFPAC, CELLS]), LOCAL)  # multiple of LOCAL ... AND >=CELLS ???
else:
    CLPAC = Fix(Fix(USER.CLPAC, CELLS), LOCAL)     # multiple of CELLS and LOCAL
    if (USER.DFPAC>0):
        DFPAC = Fix(Fix(USER.DFPAC, CELLS), LOCAL) # multiple of both LOCAL and CELLS
# Either CLPAC==DFPAC or CLPAC=0 and DFPAC>=0
print('PACKETS: PSPAC %d   BGPAC %d  CLPAC %d  DFPAC %d' % (PSPAC, BGPAC, CLPAC, DFPAC))
asarray([BGPAC, PSPAC, DFPAC, CLPAC], int32).tofile('packet.info')

if (USER.ITERATIONS<1):
    # If we solved emission outside SOC and here only write spectra without new
    # simulations, we do not want to clear the old absorption file !!
    USER.NOABSORBED = True

    
if ((NDUST>1)&(USER.ITERATIONS>0)&(USER.NOABSORBED)&(not(USER.NOSOLVE))):
    print('*'*80)
    print("One cannot have NDUST>1, ITERATIONS>0, NOABSORBED, and NOSOLVE=False!")
    print("  Internal solver in SOC can only deal with a single equilibrium-temperature dust.")
    print("  In case of multiple dust components, do not use the keyword noabsorbed but instead use")
    print("  a separate A2E run or A2E_MABU.py to calculate the emission, and then another SOC run")
    print("  to write the maps.")
    print('*'*80)
    sys.exit()
    

if ((USER.NOABSORBED==False)&(USER.NOSOLVE==False)):
    print('*'*80)
    print("*** Error: noabsorbed is not set and nosolve is not set ??")
    print("***    SOC cannot solve temperatures or calculate emission if noabsorbed is not used")
    print("***    If there is a single equilibrium dust component, add keyword noabsorbed to ini file;")
    print("***    otherwise do not use noabsorbed but call A2E or A2E_MABU.py to compute the emissions")
    print("***    based on the absorptions that SOC writes to a file")
    print('*'*80)
    sys.exit()
    
    
    
# Precalculate some things for point sources located outside the model volume

# For PS_METHOD==2
XPS_NSIDE, XPS_SIDE, XPS_AREA = AnalyseExternalPointSources(NX, NY, NZ, USER.PSPOS, int(USER.NO_PS), int(USER.PS_METHOD))

# For PS_METHOD==3
#    XPS_NSIDE, XPS_SIDE, XPS_AREA = AnalyseExternalPointSourcesHealpix(NX, NY, NZ, USER.PSPOS, int(USER.NO_PS))
   
HPBG = []
if (len(USER.file_hpbg)>2):
    # We use healpix maps for the background sky intensity... 
    #   currently *fixed* at NSIDE=64 which gives ~55 arcmin pixels, 49152 pixels on the sky
    #   1000 frequencies ~ 188 MB => just read all in
    print("*** Healpix background %s, user scaling %.3e ***" % (USER.file_hpbg, USER.scale_background))
    HPBG = fromfile(USER.file_hpbg, float32).reshape(NFREQ, 49152) * USER.scale_background
    
    
# for point sources we use exactly PSPAC packages
# for background it is     int(BGPAC/AREA)*AREA
# for cell emission it is  (CLPAC/CELLS) packages per cell
WPS   = 0.0
WBG   = 0.0
SELEM = 0         # for DO_SPLIT runs only
GLOBAL_SPLIT = 0  # different for BGPAC + split packages run
# Device initialisations
# -cl-fast-relaxed-math  -cl-unsafe-math-optimizations
# Intel OpenCL, Nvidia  ... do not recognise option -O ??

if (USER.DO_SPLIT):
    GLOBAL_SPLIT  =  GLOBAL_0
    SELEM         =  max([1, (USER.AREA//GLOBAL_SPLIT)])   # each work item does SELEM surface elements
    # change  GLOBAL_SPLIT so that SELEM*GLOBAL >= AREA but not more
    GLOBAL_SPLIT  =  Fix((USER.AREA//SELEM)+1, LOCAL)

print("PS_METHOD=%d, WITH_ABU=%d, WITH_MSF=%d" % (USER.PS_METHOD, WITH_ABU, WITH_MSF))


ARGS = "-D NX=%d -D NY=%d -D NZ=%d -D BINS=%d -D WITH_ALI=%d -D PS_METHOD=%d -D FACTOR=%.4ef \
-D CELLS=%d -D AREA=%.0f -D NO_PS=%d -D WITH_ABU=%d -D ROI_MAP=%d -D MAX_SPLIT=%d -D SELEM=%d \
-D ROI_STEP=%d -D ROI_NSIDE=%d -D WITH_ROI_LOAD=%d -D WITH_ROI_SAVE=%d \
-D AXY=%.5ff -D AXZ=%.5ff -D AYZ=%.5ff -D LEVELS=%d -D LENGTH=%.5ef -I%s -D DO_SPLIT=%d \
-D POLSTAT=%d -D SW_A=%.3ef -D SW_B=%.3ef -D STEP_WEIGHT=%d -D DIR_WEIGHT=%d -D DW_A=%.3ef \
-D LEVEL_THRESHOLD=%d -D POLRED=%d -D p00=%.4ff -D SAVE_TAU=%.4ef -D MINLOS=%.3ef -D MAXLOS=%.3ef \
-D FFS=%d -D NODIR=%d -D USE_EMWEIGHT=%d -D SAVE_INTENSITY=%d -D NOABSORBED=%d -D INTERPOLATE=%d \
-D ADHOC=%.5ef %s -D HPBG_WEIGHTED=%d -D WITH_MSF=%d -D NDUST=%d -D OPT_IS_HALF=%d -D POL_RHO_WEIGHT=%d \
-D MAP_INTERPOLATION=%d" % \
(  NX, NY, NZ, USER.DSC_BINS, USER.WITH_ALI, USER.PS_METHOD, FACTOR,
   CELLS, int(USER.AREA), max([1,int(USER.NO_PS)]), WITH_ABU, USER.ROI_MAP, MAX_SPLIT, SELEM,
   USER.ROI_STEP, USER.ROI_NSIDE, USER.WITH_ROI_LOAD, USER.WITH_ROI_SAVE,
   USER.AXY, USER.AXZ, USER.AYZ, LEVELS, USER.GL*PARSEC, INSTALL_DIR, USER.DO_SPLIT,
   USER.POLSTAT, int(USER.STEP_WEIGHT[0]), USER.STEP_WEIGHT[1], int(USER.STEP_WEIGHT[2]),
   int(USER.DIR_WEIGHT[0]), USER.DIR_WEIGHT[1],
   USER.LEVEL_THRESHOLD, len(USER.file_polred)>0, USER.p0, USER.savetau_freq, USER.MINLOS, USER.MAXLOS,
   USER.FFS, NODIR, USER.USE_EMWEIGHT, USER.SAVE_INTENSITY,  USER.NOABSORBED, USER.INTERPOLATE, 
   ADHOC, USER.kernel_defs, USER.HPBG_WEIGHTED, WITH_MSF, NDUST, USER.OPT_IS_HALF, USER.POL_RHO_WEIGHT,
   USER.MAP_INTERPOLATION)
# print(ARGS)
VARGS = ""
# VARGS += " -cl-nv-cstd=CL1.1 -cl-nv-arch sm_20 -cl-single-precision-constant -cl-mad-enable"
# VARGS += " -cl-fast-relaxed-math -cl-nv-opt-level=2"
# VARGS  = " -cl-fast-relaxed-math"
# 2020-07-23 --- usually one needs to add -cl-opt-disable for compilation to finish on Intel
#                ... but with Intel GPU one needs to have optimisation on lest the calculation
#                    freezes with Asynchronous wait on fence dma-fence (kernel 5.7) or
#                    i915 reset (kernel 4.x) ... is it just the computation finished before it has
#                    time to freeze? 
if (USER.DEVICES=='c'): # F***ing Intel compiler...
    VARGS += " -cl-opt-disable "
# VARGS += "-cl-fast-relaxed-math =  -cl-unsafe-math-optimizations -cl-finite-math-only"
# VARGS += "-cl-fast-relaxed-math =  -cl-unsafe-math-optimizations"
# VARGS += "-cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -cl-uniform-work-group-size"
# VARGS += "-cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math"
# VARGS += "-cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros"
# On NVidia -cl-fast-relaxed-math causes some small array values (such as used for cell links!!)
#    to appear as zeros on the device !!!!!!!!
# -cl-fast-relaxed-math = -cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only
# => DO NOT USE -cl-unsafe-math-optimizations ON NVidia !!!!!!!!!!!
VARGS += "-cl-mad-enable -cl-no-signed-zeros -cl-finite-math-only"  # this seems ok on NVidia !!

ARGS   += VARGS

# Create contexts, command queu, and program = kernels for the simulation step
# source  =  os.getenv("HOME")+"/starformation/SOC/kernel_ASOC.c"        
source  =  INSTALL_DIR+"/kernel_ASOC.c"        


# NOTE -- kernel_SOC now uses Healpix for pointsource and background simulation
#         and this requires NSIDE to be defined
#         The mapping kernels use different NSIDE and therefore their ARGS is different !!
context, commands =  opencl_init(USER)
# fixed NSIDE=128 for the PS and BG simulations?
print("get_program ...")
program           =  get_program(context, commands, source, ARGS + " -D NSIDE=%d" % 128)
mf                =  cl.mem_flags
print("get_program ... ok")

# A set of buffers needed by emission calculations
LCELLS_buf, OFF_buf, PAR_buf, DENS_buf, EMIT_buf, DSC_buf, CSC_buf = [], [], [], [], [], [], []
TABS_buf, RA_buf, DE_buf, XAB_buf, PS_buf, PSPOS_buf = [], [], [], [], [], []
EMINDEX_buf, OPT_buf, HPBG_buf, HPBGP_buf = [], [], [], []
# for better sampling of emission from external point sources
XPS_NSIDE_buf, XPS_SIDE_buf, XPS_AREA_buf = [], [], []
# 2018-12-27 -- for -D WITH_MSF make ABS, SCA, G potentially vectors => need buffers
ABS_buf, SCA_buf, ABU_buf = [], [], []
# 2020-07-16 -- try package splitting for simulation of the isotropic background
BUFFER_buf = []


ROI_dim_buf  = None  # ROI_LOAD
ROI_LOAD_buf = None  # ROI_LOAD
ROI_buf      = None  # ROI_SAVE
ROI_SAVE_buf = None  # ROI_SAVE
ROI_LIM_buf  = None

# We do not always need EMIT_buf on the device, for example
# when we are only calculating absorptions, without emission from the medium
#  400e6 cells => saves 1.5 GB device memory !!
NEED_EMIT_BUF = (USER.LOAD_TEMPERATURE) | (DFPAC>0) | (CLPAC>0) | (not(USER.NOSOLVE)) \
              | ('SUBITERATIONS' in USER.KEYS) | (not(USER.NOMAP)) | (USER.POLMAP>0)
# print("NEED_EMIT_BUF = %d" % NEED_EMIT_BUF)

for ID in range(DEVICES):
    LCELLS_buf.append( cl.Buffer(context[ID], mf.READ_ONLY,  LCELLS.nbytes))
    OFF_buf.append(    cl.Buffer(context[ID], mf.READ_ONLY,  OFF.nbytes))
    DENS_buf.append(   cl.Buffer(context[ID], mf.READ_ONLY,  DENS.nbytes))  # float
    RA_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  RA.nbytes))
    DE_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  DE.nbytes))
    if (WITH_MSF==0):
        DSC_buf.append(    cl.Buffer(context[ID], mf.READ_ONLY,  4*USER.DSC_BINS))
        CSC_buf.append(    cl.Buffer(context[ID], mf.READ_ONLY,  4*USER.DSC_BINS))
    else:
        DSC_buf.append(    cl.Buffer(context[ID], mf.READ_ONLY,  4*USER.DSC_BINS*NDUST))
        CSC_buf.append(    cl.Buffer(context[ID], mf.READ_ONLY,  4*USER.DSC_BINS*NDUST))
    PAR_buf.append(    cl.Buffer(context[ID], mf.READ_WRITE, 4*max([1, (CELLS-NX*NY*NZ)])) )  # 2019-02-19
    if (NEED_EMIT_BUF):
        # print("EMIT_buf allocated for CELLS")
        EMIT_buf.append(   cl.Buffer(context[ID], mf.READ_ONLY,  4*CELLS))
    else:
        EMIT_buf.append(   cl.Buffer(context[ID], mf.READ_ONLY,  4))   # dummy buffer !!

    # print("TABS allocated for CELLS")
    TABS_buf.append(   cl.Buffer(context[ID], mf.READ_WRITE, 4*CELLS))
    PS_buf.append(     cl.Buffer(context[ID], mf.READ_ONLY,  4*max([1,USER.NO_PS])))
    PSPOS_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  USER.PSPOS.nbytes))
    if (USER.WITH_ALI):
        XAB_buf.append(    cl.Buffer(context[ID], mf.READ_WRITE, 4*CELLS))
    else:
        XAB_buf.append(    cl.Buffer(context[ID], mf.READ_WRITE, 4))  # dummy
    if (USER.USE_EMWEIGHT==2):
        EMINDEX_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*CELLS))
    else:
        EMINDEX_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4))  # dummy
        
    if (WITH_ABU>0):
        if (USER.OPT_IS_HALF):
            OPT_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 2*CELLS*2))  # ABS, SCA
        else:
            OPT_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*CELLS*2))  # ABS, SCA
    else:
        OPT_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*3))  # DUMMY
    if (len(HPBG)>0):
        HPBG_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*49152))  # NSIDE==64
        if (USER.HPBG_WEIGHTED): # use weighting for emission from Healpix background
            HPBGP_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*49152))
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
    # OPT is needed whenever abundances are variable but ABU only with multiple scattering functions
    if (WITH_MSF==0):
        ABS_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4)   )
        SCA_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4)   )
        ABU_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4)   )  # dummy
        ABS, SCA = zeros(1, float32), zeros(1, float32)
    else:
        ABS_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*NDUST)  )
        SCA_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*NDUST)  )
        ABU_buf.append(  cl.Buffer(context[ID], mf.READ_ONLY,  4*NDUST*CELLS)   )  
        ABS, SCA = zeros(NDUST, float32), zeros(NDUST, float32)
    # Buffers for split-packages kernel,  we have work item per 100 surface elements
    # 512^3 root grid, MAXL=7 => (6*512**2/100.0)*(3**6)*12*4B = 0.51 GB
    if (USER.DO_SPLIT):
        # now we need to allocate BUFFER only based on work items (not surface elements)
        BUFFER_buf.append(   cl.Buffer(context[ID], mf.READ_WRITE,  4*GLOBAL_SPLIT*12*MAX_SPLIT)  )

    
# for split packages
# SID_NUMBERS is the list of four SID numbers for each of the six sides (X low & hi, Y low & hi, Z low & hi)
# SID_NUMBERS =  asarray([  0, 2, 4, 6,   1, 3, 5, 7,    0, 1, 4, 5,   2, 3, 6, 7,   4, 5, 6, 7 ], int32)
# SIDPOS  =  relative coordinates of each of the eight SID cells, basically (0,0) and offsets [0|1]
# SIDPOS = asarray([  0.0, 0.0, 0.0,    1.0, 0.0, 0.0,   0.0, 1.0, 0.0,   1.0, 1.0, 0.0,
#                     0.0, 0.0, 1.0,    1.0, 0.0, 1.0,   0.0, 1.0, 1.0,   1.0, 1.0, 1.0 ], float32)

for ID in range(DEVICES):        
    cl.enqueue_copy(commands[ID], LCELLS_buf[ID], LCELLS)
    cl.enqueue_copy(commands[ID], OFF_buf[ID],    OFF)
    cl.enqueue_copy(commands[ID], DENS_buf[ID],   DENS)      # density... ABS and SCA may be updated later
    ## cl.enqueue_copy(commands[ID], ODIR_buf[ID],   ODIR)
    cl.enqueue_copy(commands[ID], RA_buf[ID],     RA)
    cl.enqueue_copy(commands[ID], DE_buf[ID],     DE)
    if (USER.NO_PS>0):
        cl.enqueue_copy(commands[ID], PSPOS_buf[ID],     USER.PSPOS)
        cl.enqueue_copy(commands[ID], XPS_NSIDE_buf[ID], XPS_NSIDE)  #  NO_PS
        cl.enqueue_copy(commands[ID], XPS_SIDE_buf[ID],  XPS_SIDE)   #  NO_PS * 3
        cl.enqueue_copy(commands[ID], XPS_AREA_buf[ID],  XPS_AREA)   #  NO_PS * 3
    if (WITH_MSF):
        cl.enqueue_copy(commands[ID], ABU_buf[ID],   ABU)  # only when WITH_ABU and WITH_MSF
        
    commands[ID].finish()                

        

print('EMWEI %d, SAVE_INTENSITY %d' % (USER.USE_EMWEIGHT, USER.SAVE_INTENSITY))


# Emission weighting
# if CLPAC>0, DFPAC==CLPAC and the same weighting can be used for both
# if CLPAC==0, can still use emission weighting for DFPAC
EMWEI, EMWEI_buf = [], []
if (USER.USE_EMWEIGHT>0):
    pac       = max([CLPAC, DFPAC])                     #  equal or CLPAC=0 and DFPAC>0
    EMWEI     = ones(CELLS, float32)  * (pac/CELLS)     # could be ushort, 0-65k packages per cell?
    EMWEI_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4*CELLS) )
else:    
    EMWEI_buf.append( cl.Buffer(context[ID], mf.READ_ONLY, 4) )  # dummy buffer
EMWEIGHT_IGNORE = 0.0

    
    
INT_buf, INTX_buf, INTY_buf, INTZ_buf = [], [], [], []
if (USER.SAVE_INTENSITY==2): # save total intensity and anisotropy
    for ID in range(DEVICES):
        INT_buf.append(  cl.Buffer(context[ID], mf.WRITE_ONLY, 4*CELLS) )
        INTX_buf.append( cl.Buffer(context[ID], mf.WRITE_ONLY, 4*CELLS) )
        INTY_buf.append( cl.Buffer(context[ID], mf.WRITE_ONLY, 4*CELLS) )
        INTZ_buf.append( cl.Buffer(context[ID], mf.WRITE_ONLY, 4*CELLS) )
else:
    for ID in range(DEVICES):
        # NOTE -- if absorptions are saved to file, we use the same INT_buf to pull the data
        if ((USER.SAVE_INTENSITY in [1,2])|(not(USER.NOABSORBED))):
            INT_buf.append(  cl.Buffer(context[ID], mf.WRITE_ONLY, 4*CELLS) )
        else:
            INT_buf.append(  cl.Buffer(context[ID], mf.WRITE_ONLY, 4) )  # dummy
        INTX_buf.append( cl.Buffer(context[ID], mf.WRITE_ONLY, 4) )      # dummy
        INTY_buf.append( cl.Buffer(context[ID], mf.WRITE_ONLY, 4) )
        INTZ_buf.append( cl.Buffer(context[ID], mf.WRITE_ONLY, 4) )

    
# Update the parent links on devices = PAR_buf
kernel_parents = []
GLOBAL  = GLOBAL_0
for ID in range(DEVICES):
    kernel_parents.append(program[ID].Parents)  # initialisation of links to cell parents
    print("LOCAL %d, GLOBAL %d" % (LOCAL, GLOBAL))
    kernel_parents[ID](commands[ID], [GLOBAL,], [LOCAL,], DENS_buf[ID], LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID])
    commands[ID].finish()
    
if (0): # -cl-unsafe-math-optimizations caused some DENS values to appear as zeros on the device !!!
    print("HOST        DENS[886247] = %.4e   DENS[1886248] = %.4e   DENS[2886249] = %.4e\n" % (DENS[886247],DENS[1886248],DENS[2886249]))
    cl.enqueue_copy(commands[ID], DENS, DENS_buf[0])
    commands[ID].finish()    
    print("FROM DEVICE DENS[886247] = %.4e   DENS[1886248] = %.4e   DENS[2886249] = %.4e\n" % (DENS[886247],DENS[1886248],DENS[2886249]))
    sys.exit()    
    
# EMITTED will be mmap array  EMITTED[icell, ifreq], possibly REMIT_NFREQ frequencies
# assert(REMIT_NFREQ==NFREQ)
#  REMIT_NFREQ, indices [REMIT_I1, REMIT_I2]  => EMITTED is limited to this
#     should work even even if cell emission is included
# With LIB_MAPS, emitted file contains only a subset of frequencies == USER.FSELECT
# => make sure EMITTED is still read and not zeroed
if (USER.LIB_MAPS):
    print("### ASOC WITH LIB_MAPS =>  ASSUME EMITTED %d CELLS x  %d FREQUENCIES" % (CELLS, len(USER.FSELECT)))    
    EMITTED  = mmap_emitted(USER, CELLS, LEVELS, LCELLS, len(USER.FSELECT), OFF, DENS)  # EMITTED[CELLS, REMIT_NFREQ]
elif (USER.LIB_ABS): #  only emission at USER.FSLECT frequencies = reference frequencies
    print("### ASOC WITH LIB_ABS =>  EMITTED %d CELLS  x   %d FREQUENCIES" % (CELLS, len(USER.FSELECT)))
    # IN FACT THIS RUN WILL NOT DO NOTHING WITH EMITTED... APART FROM AS DIFFUSE RADIATION SOURCE ?
    EMITTED  = mmap_emitted(USER, CELLS, LEVELS, LCELLS, len(USER.FSELECT), OFF, DENS)
else:
    EMITTED  = mmap_emitted(USER, CELLS, LEVELS, LCELLS, REMIT_NFREQ, OFF, DENS)  # EMITTED[CELLS, REMIT_NFREQ]


Tkernel, Tpush, Tpull, Tsolve, Tmap = 0.0, 0.0, 0.0, 0.0, 0.0
        

## ABSORBED  = zeros(CELLS, float32)  ---- 2019-02-19  NOT USED !?
FABSORBED = None
if (not(USER.NOABSORBED)): # we store absorptions at all frequencies, use mmap array FABSORBED
    # This is required by A2E ... DustEm uses the file_intensity instead
    # If LIB_ABS is true, absorption file will contain only frequencies USER.FSELECT => ONFREQ frequencies
    print("**** OPEN ABSORPTION FILE ****\n")
    fp  =  open(USER.file_absorbed, "wb")
    asarray([CELLS, ONFREQ], int32).tofile(fp)  # 2018-12-29 dropped "levels" from the absorbed/emitted files
    fp.close()
    FABSORBED = np.memmap(USER.file_absorbed, dtype='float32', mode='r+', offset=8, shape=(CELLS, ONFREQ))
    FABSORBED[:,:] = 0.0
    
    
Emin, kE, TTT, TTT_buf = None, None, None, None
NE = 30000  # number of energies when interpolating Tdust for non-stochastic grains
if (not(USER.NOSOLVE)):  # Calculate mapping T <-> E
    # 2018-12-08 -- at the moment require that if we are solving dust emission
    #               inside SOC, we are dealing with non-stochastically heated grains
    #               and therefore there must be only a single dust component
    if (NDUST>1):
        print("*** Error: If emission is solved inside SOC, there must be only a single dust population!")
        """
        Note:
            In principle we could use *noabsorbed* options for non-stochastically heated grains,
            calculate integral of absorbed energy on-the-fly and avoid storing the large
            absorbed.data file to disk. However, that would not work for cases where the dust
            abundances vary and, when solving dust temperatures, we would need to divide the
            absorbed energy (frequency by frequency!) between the different dust populations.
            => whenever there are several dust components, just save the huge absorbed.data 
            file and use A2E_MABU.py to calculate the emission.
        """
        sys.exit()

    t1     = time.time()  # this is fast (<1sec) and can be done by host
    TSTEP  = 1600.0/NE    # hardcoded upper limit 1600K for the maximum dust temperatures
    TT     = zeros(NE, float64)
    Eout   = zeros(NE, float64)
    DF     = FFREQ[2:] - FFREQ[:(-2)]  #  x[i+1] - x[i-1], lengths of intervals for Trapezoid rule
    for i in range(NE):
        # Eout = integral  FABS * B(TT)
        #    FABS = optical depth for GL distance and unit density
        #    energy per H   =   this / (GL*PARSEC)
        TT[i]  =  1.0+TSTEP*i
        TMP    =  AFABS[0] * PlanckSafe(asarray(FFREQ, float64), TT[i])
        # Trapezoid integration TMP over FFREQ frequencies
        res     = TMP[0]*(FFREQ[1]-FFREQ[0]) + TMP[-1]*(FFREQ[-1]-FFREQ[-2]) # first and last step
        res    += sum(TMP[1:(-1)]*DF)  # the sum over the rest of TMP*DF
        Eout[i] = (4.0*np.pi*FACTOR/(USER.GL*PARSEC)) * 0.5 * res  # energy corresponding to TT[i]
    # Calculate the inverse mapping    Eout -> TTT   ---- energy includes FACTOR scaling
    Emin, Emax = Eout[0], Eout[NE-1]*0.9999
    # E ~ T^4  => use logarithmic sampling
    kE     = (Emax/Emin)**(1.0/(NE-1.0))  # E[i] = Emin*pow(kE, i)
    oplgkE = 1.0/log10(kE)
    ip     = interp1d(Eout, TT)           # (linear) interpolation from energy to temperature
    TTT    = asarray(ip(Emin * kE**arange(NE)), float32)
    print("Mapping E -> T calculated on host: %.3f seconds" % (time.time()-t1))
    print("Emin= %.3e, Emax= %.3e, kE= %.6f,  NE= %d, kE**(NE-1)= %.3e" % (Emin, Emax, kE, NE, kE**(NE-1.0)))
    print("TTT %.3e - %.3e" % (TTT[0], TTT[NE-1]))
    ## sys.exit()

    
"""
This script makes separete kernel calls for PSPAC, BGPAC, diffuse sources (CLPAC) and
finally the emission from dust (CLPAC packages). This last step includes both the use 
of a reference field and the ALI part with the escape probabilities extracted from 
the simulation.
"""

# Try to read old temperatures
TNEW         = None
if (USER.LOAD_TEMPERATURE):
    print("Trying to read old temperatures: %s" % (USER.file_temperature))
    try:
        TNEW     = read_otfile(USER.file_temperature)
        if (len(TNEW)<CELLS):
            print("Old temperature file contained %d < %d temperature values" % (len(TNEW),CELLS))
            # One may have added another hierarchy level...
            if (len(TNEW)!=(CELLS-LCELLS[LEVELS-1])):            
                throw(1)   # no -- we cannot use the old file
    except:        
        print("*"*80)
        print("*** Failed to read old temperatures !!")
        if (USER.WITH_REFERENCE):
            print("*** Without a temperature file the results will be wrong!!")
            print("*** ... not a problem if this is the first iteration of the first run.")
        print("*"*80)
        time.sleep(3)
        TNEW     = 15.0*ones(CELLS, float32)
        TNEW[nonzero(DENS<1e-7)] = 0.0  # link cells have no temperature
                
    if (len(TNEW)==(CELLS-LCELLS[LEVELS-1])):
        # presumably hierarchy has an additional level -> copy values from 
        # LEVELS-2 to LEVELS-1
        print("Copy temperatures level %d -> level %d" % (LEVELS-2, LEVELS-1))
        if (0):
            clf()
            plot(TNEW, 'mx')    
        TNEW = concatenate((TNEW, zeros(LCELLS[LEVELS-1])))
        values_down_in_hierarchy(OFF, LCELLS, DENS, TNEW, LEVELS-2)
        if (0):
            plot(1.0e-4*DENS, 'k.')
            plot(TNEW, 'r.')
            show()
            sys.exit()


        
EMIT         = zeros(CELLS, float32)   # true emission per cell
# we cannot skip EMIT even if emission is not simulated because EMIT[] is
# used as work space for absorptions returned from the device


t0 = time.time()
if ((USER.LOAD_TEMPERATURE)&(USER.ITERATIONS<1)):  
    """
    Optionally, recalculate EMITTED using stored temperature file (equilibrium dust only!)
    This can be combined with iterations=0 to directly write the resulting spectra.
    Normally emission is just read from a file that was written in a previous calculation.
    """
    # temperature_to_emitted(USER, EMITTED)  # updates EMITTED, values will be written to file
    if (0):
        print('*'*80)
        print("temperature_to_emitted_inmem")
        temperature_to_emitted_inmem(USER, TNEW, EMITTED)  # updates EMITTED, values will be written to file
        print('*'*80)
    else:
        # 2019-01-15
        # use Emission() kernel routine
        print('*'*80)
        print("temperature_to_emitted --- in kernel")
        kernel_emission = program[0].Emission
        kernel_emission.set_scalar_arg_dtypes([np.float32, np.float32, None, None, None])
        cl.enqueue_copy(commands[0], EMIT_buf[0], TNEW)
        a, b = REMIT_I1, REMIT_I2+1
        for ifreq in range(a, b):
            kernel_emission(commands[0], [GLOBAL,], [LOCAL,], float(FFREQ[ifreq]), float(AFABS[0][ifreq]),
                 DENS_buf[0], EMIT_buf[0],  TABS_buf[0])
            cl.enqueue_copy(commands[0], EMIT, TABS_buf[0])
            EMITTED[:, ifreq-REMIT_I1] = EMIT
        ####

Tsolve += time.time()-t0    


    
    
    
    
ID = 0  # not all parts work for multiple devices ?





# New host buffers
## REMIT        = zeros(CELLS, float32)   # emission for the reference case
XEM, XAB     = None, None
if (USER.WITH_ALI):
    XEM      = zeros(CELLS, float32)   # emitted energy from each cell
    XAB      = zeros(CELLS, float32)   # absorbed energy in each cell, not counting self-absorptions
    

# These arrays are needed because of the reference field
CTABS        = zeros(CELLS, float32)   # stores absorptions due to sources other than dust itself 
# OEMITTED is the large array, emission in the reference case per cell and per frequency
OEMITTED, OTABS = None, None
if (USER.WITH_REFERENCE>0):
    assert(NFREQ==REMIT_NFREQ)           # perhaps could have REMIT_NFREQ<NFREQ if no reference field ??
    OEMITTED     = np.memmap("emitted.ref", dtype='float32', mode='w+', shape=(CELLS, REMIT_NFREQ))
    OTABS        = zeros(CELLS, float32) # absorbed energy [CELLS] cause by the reference field
    
    if (USER.WITH_REFERENCE>1): # trying to continue old run -- read last OEMITTED and OTABS
        wr_fir        = (USER.WITH_REFERENCE % 100)
        if (wr_fir>0):  
            # first iteration of this run == iteration >1 of all the runs ==  SOC continues from an old run
            OEMITTED[:,:] = fromfile('OEMITTED.save', float32).reshape(CELLS, REMIT_NFREQ)
            OTABS[:]      = fromfile('OTABS.save', float32)
            print('1'*80)
            print('1   OEMITTED %12.4e, OTABS %12.4e' % (mean(ravel(OEMITTED)), mean(OTABS)))
            print('1'*80)
OXAB, OXEM   = None, None
if ((USER.WITH_ALI)&(USER.WITH_REFERENCE)):  # 2019-02-19 added the second condition
    OXAB     = zeros(CELLS, float32)   # self-absorptions for the reference simulation
    OXEM     = zeros(CELLS, float32)   # emitted energy per cell for the reference case


kernel_zero    = program[ID].ZeroAMC
kernel_zero.set_scalar_arg_dtypes([np.int32, None, None, None, None, None, None])

kernel_ram_pb    = program[ID].SimRAM_PB   # point sources, isotropic background, ROI background
kernel_ram_hp    = program[ID].SimRAM_HP   # healpix background
kernel_ram_cl    = program[ID].SimRAM_CL   # cell emission

if (USER.DO_SPLIT):
    # Isotropic background + photon splitting
    kernel_bg_split  = program[ID].SimBgSplit
    kernel_bg_split.set_scalar_arg_dtypes([
    # 0          1         2           3         4     5      6           7           8      9 
    # PACKETS,   BATCH,    SEED,       ABS,   SCA,  BG,         TW,         LCELLS, OFF,   PAR 
    np.int32,    np.int32, np.float32, None, None,  np.float32, np.float32, None,   None, None,
    # 10    11    12    13    14    15    16    17    18    19    20     21
    # DENS, EMIT, TABS, DSC,  CSC,  INT,  INTX, INTY, INTZ, OPT,  ABU,   BUFFER
    None,   None, None, None, None, None, None, None, None, None, None,  None])
    # 2021-01-31 Healpix background + photon splitting
    kernel_hp_split  = program[ID].SimHpSplit
    kernel_hp_split.set_scalar_arg_dtypes([
    # 0          1          2            3      4      5           6           7      8   
    # PACKETS,   BATCH,     SEED,        ABS,   SCA,   TW,         LCELLS,     OFF,   PAR 
    np.int32,    np.int32,  np.float32,  None,  None,  np.float32, None,       None,  None,
    # 9,    10    11    12    13    14    15    16    17    18    19    20     21     22    
    # DENS, EMIT, TABS, DSC,  CSC,  INT,  INTX, INTY, INTZ, OPT,  BG,   HPBGP, ABU,   BUFFER
    None,   None, None, None, None, None, None, None, None, None, None, None,  None,  None])
    
    


if ((USER.WITH_ROI_LOAD<=0)&(USER.WITH_ROI_SAVE<=0)):  # normal case
    kernel_ram_pb.set_scalar_arg_dtypes(
    #   0      1          2         3           4      5     
    # SOURCE   PACKETS    BATCH     SEED        ABS    SCA   
    [np.int32, np.int32,  np.int32, np.float32, None,  None, 
    # 6         7            8                        9      
    # BG        PSPOS        PS                       TW     
    np.float32, None,        None,                    np.float32,
    # 10      11    12    13     14     15     16    17    18    19     20    21    22    23    24 
    # LCELLS  OFF   PAR   DENS   EMIT   TABS   DSC   CSC   XAB   EMWEI  INT   INTX  INTY  INTZ  OPT
    None,     None, None, None,  None,  None,  None, None, None, None,  None, None, None, None, None,
    # 25          26         27         28      
    # ABU         XPS_NSIDE  XPS_SIDE   XPS_AREA
    None,         None,      None,      None])
elif ((USER.WITH_ROI_LOAD>0)&(USER.WITH_ROI_SAVE>0)): # both ROI_LOAD and ROI_SAVE !!
    kernel_ram_pb.set_scalar_arg_dtypes(
    #   0      1          2         3           4      5     
    # SOURCE   PACKETS    BATCH     SEED        ABS    SCA   
    [np.int32, np.int32,  np.int32, np.float32, None,  None, 
    # 6         7            8                        9      
    # BG        PSPOS        PS                       TW     
    np.float32, None,        None,                    np.float32,
    # 10      11    12    13     14     15     16    17    18    19     20    21    22    23    24 
    # LCELLS  OFF   PAR   DENS   EMIT   TABS   DSC   CSC   XAB   EMWEI  INT   INTX  INTY  INTZ  OPT
    None,     None, None, None,  None,  None,  None, None, None, None,  None, None, None, None, None,
    # 25          26         27         28         29         30        31     32
    # ABU         XPS_NSIDE  XPS_SIDE   XPS_AREA   ROI_DIM    ROI_LOAD  ROI    ROI_SAVE
    None,         None,      None,      None,      None,      None,     None,  None    ])
elif (USER.WITH_ROI_LOAD>0):   # only ROI_LOAD
    kernel_ram_pb.set_scalar_arg_dtypes(
    #   0      1          2         3           4      5     
    # SOURCE   PACKETS    BATCH     SEED        ABS    SCA   
    [np.int32, np.int32,  np.int32, np.float32, None,  None, 
    # 6         7            8                        9      
    # BG        PSPOS        PS                       TW     
    np.float32, None,        None,                    np.float32,
    # 10      11    12    13     14     15     16    17    18    19     20    21    22    23    24 
    # LCELLS  OFF   PAR   DENS   EMIT   TABS   DSC   CSC   XAB   EMWEI  INT   INTX  INTY  INTZ  OPT
    None,     None, None, None,  None,  None,  None, None, None, None,  None, None, None, None, None,
    # 25          26         27         28         29         30       
    # ABU         XPS_NSIDE  XPS_SIDE   XPS_AREA   ROI_DIM    ROI_LOAD 
    None,         None,      None,      None,      None,      None     ])
else:   # only ROI_SAVE
    kernel_ram_pb.set_scalar_arg_dtypes(
    #   0      1          2         3           4      5     
    # SOURCE   PACKETS    BATCH     SEED        ABS    SCA   
    [np.int32, np.int32,  np.int32, np.float32, None,  None, 
    # 6         7            8                        9      
    # BG        PSPOS        PS                       TW     
    np.float32, None,        None,                    np.float32,
    # 10      11    12    13     14     15     16    17    18    19     20    21    22    23    24 
    # LCELLS  OFF   PAR   DENS   EMIT   TABS   DSC   CSC   XAB   EMWEI  INT   INTX  INTY  INTZ  OPT
    None,     None, None, None,  None,  None,  None, None, None, None,  None, None, None, None, None,
    # 25          26         27         28         29         30       
    # ABU         XPS_NSIDE  XPS_SIDE   XPS_AREA   ROI        ROI_SAVE 
    None,         None,      None,      None,      None,      None     ])
    
    
    
    
    
# Open the save file to mmap and send to device the dimensions of the ROI array
ROI_LOAD, ROI_SAVE = [], []

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

    
if (USER.WITH_ROI_SAVE):   # **** WITH_ROI_SAVE   -->   (ROI, ROI_SAVE)
    tmp    = zeros(5, int32)
    tmp[0] = (USER.ROI[1]-USER.ROI[0]+1) * USER.ROI_STEP
    tmp[1] = (USER.ROI[3]-USER.ROI[2]+1) * USER.ROI_STEP
    tmp[2] = (USER.ROI[5]-USER.ROI[4]+1) * USER.ROI_STEP
    tmp[3] =  USER.ROI_NSIDE
    tmp[4] =  USER.NFREQ    
    ### sys.exit()
    asarray(tmp, int32).tofile(USER.FILE_ROI_SAVE)                   # header (nx, ny, nz, nside, nfreq)
    ROI_SAVE_NELEM =  tmp[0]*tmp[1] + tmp[1]*tmp[2] + tmp[2]*tmp[0]  # number of surfaces elements
    ROI_SAVE_NPIX  =  ROI_SAVE_NELEM * 12*USER.ROI_NSIDE*USER.ROI_NSIDE
    ROI_SAVE       =  np.memmap(USER.FILE_ROI_SAVE, dtype='float32', mode='r+', offset=20,
                               shape=(USER.NFREQ, ROI_SAVE_NPIX))
    ROI_SAVE[:,:] =  0.0  # constant sources added once, dust emission on last iteration
    ##  ROI_buf = ROI = 6 elements = [x0, x1, y0, y1, z0, z1 ]
    ROI_buf       =  cl.Buffer(context[0], mf.READ_ONLY, 4*6)
    cl.enqueue_copy(commands[0], ROI_buf, asarray(USER.ROI, int32)) # --- x0, x1, y0, y1, z0, z1 ---
    ROI_SAVE_buf  =  cl.Buffer(context[0], mf.READ_WRITE, 4*ROI_SAVE_NPIX)
    print("***  nfreq %d, npix %d, nelem %d = %d" % (USER.NFREQ,12*USER.ROI_NSIDE**2,ROI_SAVE_NELEM,USER.NFREQ*ROI_SAVE_NPIX))
    # sys.exit()
    

kernel_ram_hp.set_scalar_arg_dtypes(
#   0        1          2          3     4     6          
#   PACKETS  BATCH     SEED        ABS   SCA   TW         
[np.int32,   np.int32, np.float32, None, None, np.float32,
# 7       8     9     10     11     12     13    14    15    16    17    18    19    20   
# LCELLS  OFF   PAR   DENS   EMIT   TABS   DSC   CSC   XAB   INT   INTX  INTY  INTZ  OPT  
None,     None, None, None,  None,  None,  None, None, None, None, None, None, None, None,
# 21  22     23   
# BG  HPBGP  ABU  
None, None,  None  ])


# For the kernel_ram_cl  has nothing for ROI_LOAD
if (USER.WITH_ROI_SAVE<=0):  # normal case, no ROI photons saved
    kernel_ram_cl.set_scalar_arg_dtypes(
    #   0      1          2         3           4     5       6          
    # SOURCE   PACKETS    BATCH     SEED        ABS   SCA     TW         
    [np.int32, np.int32,  np.int32, np.float32, None, None,   np.float32,
    # 7       8     9     10     11     12     13    14    15    16     17    18    19    20  
    # LCELLS  OFF   PAR   DENS   EMIT   TABS   DSC   CSC   XAB   EMWEI  INT   INTX  INTY  INTZ
    None,     None, None, None,  None,  None,  None, None, None, None,  None, None, None, None,
    # 21        22     22  
    # EMINDEX   OPT    ABU 
    None,       None,  None
    ])
else:
    # *** USER.WITH_ROI_SAVE ***
    # for MWM, we have the option to store the photons that enter ROI from outside 
    # (to be used for further simulations of the subgrid that is defined by ROI)
    kernel_ram_cl.set_scalar_arg_dtypes(
    #   0      1          2         3           4     5      6          
    # SOURCE   PACKETS    BATCH     SEED        ABS   SCA    TW         
    [np.int32, np.int32,  np.int32, np.float32, None, None,  np.float32,
    # 7       8     9     10     11     12     13    14    15    16     17    18    19    20  
    # LCELLS  OFF   PAR   DENS   EMIT   TABS   DSC   CSC   XAB   EMWEI  INT   INTX  INTY  INTZ
    None,     None, None, None,  None,  None,  None, None, None, None,  None, None, None, None,
    # 21        22     22        23     24       
    # EMINDEX   OPT    ABU       ROI    ROI_SAVE
    None,       None,  None,     None,  None
    ])
    
    
# Initialise memory mapped file ISRF.DAT to save intensity values
INTENSITY = []
if (USER.SAVE_INTENSITY>0):
    assert(USER.WITH_REFERENCE<1) # save intensity only if the kernel has the total field
if (USER.SAVE_INTENSITY==1):      # save scalar intensity
    INTENSITY = np.memmap('ISRF.DAT',dtype='float32',mode="w+",shape=(CELLS, USER.NFREQ),offset=8)
    INTENSITY[:,:] = 0.0
if (USER.SAVE_INTENSITY==2):      # save vector --- header is { CELLS, NFREQ, 4 }
    INTENSITY = np.memmap('ISRF.DAT',dtype='float32',mode="w+",shape=(CELLS,USER.NFREQ,4),offset=12)
    INTENSITY[:,:,:] = 0.0        # [CELLS, NFREQ, 4]

TMP      = zeros(CELLS, float32)






# First a loop to simulate all photons from "constant" radiation sources
#   = all except the dust medium itself

BG, PS = 0.0, 0.0  # photons per package for background and point source
if (len(USER.file_constant_load)>0):
    # Absorbed energy per cell for the constant sources (all except the dust emissions)
    # CLOAD => CONSTANT SOURCES NOT SIMULATED
    # WE HAVE SEPARATE LOOP FOR CELL EMISSION.... WHICH ALSO INCLUDES THE OPTIONAL T SOLUTION !!
    CTABS = fromfile(USER.file_constant_load, float32, CELLS)
    if (USER.SAVE_INTENSITY>0):
        print("*** WARNING: USER.file_constant_load + USER.SAVE_INTENSITY ???")
        print("             The intensity file will contain only emission from the medium !!!")
else:
    # Simulate the constant radiation sources with PSPAC and BGPAC packages
    # the resulting TABS array is stored to CTABS and saved to disk
    # CLPAC = constant diffuse emission
    #    do DFPAC here, separate of CLPAC, to avoid complications with ALI
    skip  =  USER.EMWEIGHT_SKIP-1
    for II in range(4):  # PSPAC, BGPAC, DFPAC, ROI-background
        print("CONSTANT SOURCES II=%d" % II)
        if (USER.ITERATIONS<1): continue
        
        if (II==0):   # point sources
            # 2017-12-25 -- each work item does BATCH packages, looping over the sources
            # 2018-12-22 -- also working on external point sources (see METHOD_PS in kernel_SOC.c)
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
            # We may not have many surface elements -> GLOBAL remains too small
            # calculate BATCH assuming that GLOBAL is 8 x AREA
            if (len(HPBG)>0):
                # no systematic traversal of surface elements => use all GLOBAL work items
                BATCH    =  100
                GLOBAL   =  Fix(BGPAC/BATCH, 64)
                if (USER.DO_SPLIT>0):    # however, if we split packages, for memory reasons a smaller GLOBAL_SPLIT
                    GLOBAL = Fix(GLOBAL_SPLIT, 64)
                BGPAC    =  GLOBAL*BATCH
                WBG      =  np.pi/PLANCK
                WBG     /=  (GLOBAL*BATCH)/(2*(NX*NY+NX*NZ+NY*NZ))    # /= packages per element
                print("=== HPBG: BGPAC %d, BATCH %d, GLOBAL %d, LOCAL %d ===" % (BGPAC, BATCH, GLOBAL, LOCAL))
            else:
                BATCH    =  max([1, int(round(BGPAC/(8*USER.AREA)))]) # packets per 
                BGPAC    =  int(8*USER.AREA*BATCH)     # GLOBAL >= 8*AREA !!
                WBG      =  np.pi/(PLANCK*8*BATCH)     # 8*BATCH rays per surface element
                GLOBAL   =  Fix(int(8*USER.AREA), 64)
                if (LOCAL==12):     GLOBAL  =  Fix(int(8*USER.AREA), 96)  # GLOBAL >= 8*USER.AREA
                assert(GLOBAL>=(8*USER.AREA))
                if (USER.DO_SPLIT):
                    # AREA*BATCH == BGPAC,    GLOBAL_SPLIT*SELEM >= AREA
                    # work item loops over SELEM surface elements, sends BATCH packages per surface element
                    print("*** DO SPLIT ***")
                    print("%d / %d = %d" % (USER.BGPAC, USER.AREA, max([1, int(USER.BGPAC/USER.AREA)])))
                    BATCH   =  max([1, int(USER.BGPAC/USER.AREA)])  # rays per element == BATCH
                    BGPAC   =  int(USER.AREA*BATCH)                 # corrected number of rays
                    WBG     =  np.pi/(PLANCK*BATCH)                 # BATCH rays per surface element
                    print("===== SPLIT: GLOBAL_SPLIT %d x SELEM %d = %d >= AREA %d x BATCH %d = %d = BGPAC %d =====\n" %
                    (GLOBAL_SPLIT, SELEM, GLOBAL_SPLIT*SELEM, USER.AREA, BATCH, USER.AREA*BATCH, BGPAC))
                else:
                    print("=== BG: BGPAC %d, BATCH %d, GLOBAL %d, LOCAL %d ===" % (BGPAC, BATCH, GLOBAL, LOCAL))
            PACKETS  =  BGPAC            
            # sys.exit()
        elif (II==2):      # diffuse emission
            # RAM2 version --- each work item loops over cells =>  GLOBAL<CELLS, BATCH==packets per cell
            GLOBAL   =  GLOBAL_0
            if (len(DIFFUSERAD)<1): continue  # no diffuse radiation sources            
            if (DFPAC<1): continue            # ??? THIS MAY BE >0 EVEN WHEN NO DIFFUSE FIELD USED ??
            BATCH    =  int(DFPAC/CELLS)
            PACKETS  =  DFPAC                
            print("=== DFPAC %d, GLOBAL %d, BATCH %d, LOCAL %d" % (DFPAC, GLOBAL, BATCH, LOCAL))
        elif (II==3):     # II==3  ----- ROI background --- ROI should contain real photons
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
            
        # This will set TABS == 0.0 --- in this loop over II, TABS is always 
        #   just for one radiation field component
        kernel_zero(commands[ID],[GLOBAL,],[LOCAL,],0,TABS_buf[ID],XAB_buf[ID],INT_buf[ID],INTX_buf[ID],INTY_buf[ID],INTZ_buf[ID])
                

        OIFREQ = -1   # true index of the saved frequency, in case we have LIB_ABS==True
        for IFREQ in range(NFREQ):

            Tpre = time.time()            
            FREQ = FFREQ[IFREQ]
                        
            if (USER.LIB_ABS): 
                # Simulation for the library method, FSELECT contains the reference frequencies
                if (np.min(abs((FREQ-USER.FSELECT)/FREQ))>0.001): continue
            OIFREQ += 1 

            # Must increase OIFREQ before SIM_F test.... SIM_F limits the number of simulated frequencies
            # but the absorption arrays still contain all NFREQ frequencies.... unless frequencies were
            # dropped using FSELECT above --- OIFREQ is index to FABSORBED array
            
            # INTENSITY has been zeroed -> skip some unimportant frequencies that will remain 0
            if ((FREQ<USER.SIM_F[0])|(FREQ>USER.SIM_F[1])):
                continue            
            
            # single dust            => we use ABS[1] and SCA[1]  
            # WITH_ABU               => we use OPT instead of ABS[1] and SCA[1]
            # WITH_ABU and WITH_MSF  => we use OPT **and** ABS[NDUST] and SCA[NDUST]
            if (WITH_ABU>0): # ABS, SCA, G are precalculated on host side -> OPT
                OPT[:,:] = 0.0
                if (USER.SINGLE_ABU):
                    # we have exactly two dust components with abundances ABU and 1.0-ABU
                    OPT[:,0] +=       ABU  * AFABS[0][IFREQ]  # IFREQ -> index to full NFREQ vectors
                    OPT[:,1] +=       ABU  * AFSCA[0][IFREQ]
                    OPT[:,0] +=  (1.0-ABU) * AFABS[1][IFREQ]
                    OPT[:,1] +=  (1.0-ABU) * AFSCA[1][IFREQ]
                else:
                    for idust in range(NDUST):
                        OPT[:,0] +=  ABU[:,idust] * AFABS[idust][IFREQ]
                        OPT[:,1] +=  ABU[:,idust] * AFSCA[idust][IFREQ]
                if (USER.OPT_IS_HALF):
                    cl.enqueue_copy(commands[ID], OPT_buf[ID], asarray(OPT, np.float16))
                else:
                    cl.enqueue_copy(commands[ID], OPT_buf[ID], OPT)
                if (WITH_MSF):  # additionally put in ABS, SCA vectors of values for each species
                    for idust in range(NDUST):
                        ABS[idust] = AFABS[idust][IFREQ]
                        SCA[idust] = AFSCA[idust][IFREQ]
            else:               # constant abundance, ABS, SCA, G are scalar (but possibly still sum over species)
                if (WITH_MSF):
                    print("WITH_ABU=0 and WITH_MSF=1  -- no such combination!!")
                    sys.exit(0)
                ABS[0], SCA[0] = 0.0, 0.0
                for idust in range(NDUST):
                    ABS[0] += AFABS[idust][IFREQ]
                    SCA[0] += AFSCA[idust][IFREQ]                    
            cl.enqueue_copy(commands[ID], ABS_buf[ID], ABS) 
            cl.enqueue_copy(commands[ID], SCA_buf[ID], SCA)
       

            sys.stdout.write("  FREQ %3d/%3d  %10.3e" % (IFREQ+1, NFREQ, FREQ))
            sys.stdout.flush()
            
            kernel_zero(commands[ID],[GLOBAL,],[LOCAL,],1,TABS_buf[ID],XAB_buf[ID],INT_buf[ID],INTX_buf[ID],INTY_buf[ID],INTZ_buf[ID])
            
            if (II==0): # update point source luminosities for the current frequency
                PS = LPS[:,IFREQ] * WPS/FREQ  # source photons per package
                cl.enqueue_copy(commands[ID], PS_buf[ID], PS)
            else:
                PS = [0.0,]

                
            BG                 =  0.0
            if (len(IBG)==NFREQ):  BG = IBG[IFREQ] * WBG/FREQ    # background photons per package

            if ((II==1)&(len(HPBG)>0)):
                # push the Healpix background map for the current frequency to device
                if (USER.HPBG_WEIGHTED):  # using weighted emission from the Healpix background
                    tmp    =  asarray(HPBG[IFREQ,:], float64)
                    if (max(tmp)<1.0e-40): continue # empty sky
                    # convert into probablity
                    tmp   /=  mean(tmp)
                    tmp    =  clip(tmp, 1.0e-3, 1.0e4) # clip very low and high probabilities
                    tmp   /=  sum(tmp)                 # 1/49152  ->  tmp    == probability per Healpix pixel
                    # print("HPBGp", percentile(tmp, (0, 10, 50, 90, 100)))                    
                    HPBGW  =  (1.0/49152.0)/tmp        # weight of each Healpix pixel, relative to unweighted case
                    HPBGP  =  cumsum(tmp)     
                    HPBGP[-1] = 1.00001                # avoid any chance of rounding error here
                    # Include the additional weighting for relative pixel probability directly in HPBG !!
                    cl.enqueue_copy(commands[ID], HPBG_buf[ID],  asarray((WBG/FREQ)*HPBG[IFREQ,:]*HPBGW, float32))
                    cl.enqueue_copy(commands[ID], HPBGP_buf[ID], asarray(HPBGP, float32))
                else:
                    # no relative weighting between Healpix pixels
                    cl.enqueue_copy(commands[ID], HPBG_buf[ID], (WBG/FREQ)*HPBG[IFREQ,:])
                # print("    ---  min(HPBG)=%12.4e, max(HPBG)=%12.4e" % (min(HPBG[IFREQ,:]), max(HPBG[IFREQ,:])))
                    
            
            # The weight for on-the-fly trapezoid integration
            FF = FREQ 
            if (IFREQ==0):              FF *= 0.5*(FFREQ[1]-FFREQ[0])
            else:
                if (IFREQ==(NFREQ-1)):  FF *= 0.5*(FFREQ[NFREQ-1]-FFREQ[NFREQ-2])
                else:                   FF *= 0.5*(FFREQ[IFREQ+1]-FFREQ[IFREQ-1])
            
                
            Tpre = time.time()-Tpre
                
            sys.stdout.write("   BG %12.4e  PS %12.4e   TW %10.3e" % (BG, PS[0], FF))
            sys.stdout.flush()
            T000 = time.time()
            
            # Upload to device new DSC, CSC, possibly also EMIT
            Tpush = time.time()
            if (WITH_MSF==0): # we have single DSC and CSC arrays, pick values for the current frequency
                cl.enqueue_copy(commands[ID], DSC_buf[ID], FDSC[0,IFREQ,:])
                cl.enqueue_copy(commands[ID], CSC_buf[ID], FCSC[0,IFREQ,:])
            else:             # kernel needs DSC and CSC for each dust separately
                for idust in range(NDUST):
                    DSC[idust,:] = FDSC[idust, IFREQ, :]
                    CSC[idust,:] = FCSC[idust, IFREQ, :]
                cl.enqueue_copy(commands[ID], DSC_buf[ID], DSC)
                cl.enqueue_copy(commands[ID], CSC_buf[ID], CSC)
            commands[ID].finish()
            Tpush = time.time()-Tpush    
            if (USER.SEED>0): seed = fmod(USER.SEED+SEED0+(DEVICES*IFREQ+ID)*SEED1, 1.0)
            else:             seed = rand()
            
            if (II==2):     # emission from the volume (but not yet the dust)
                # 2019-02-20 added condition on DFPAC -> EMIT not needed here unless DFPACK>0
                #            ... except that it is used as work memory for returned absorptions!!
                ## print("DFPAC = %d" % DFPAC) 
                if (DFPAC<1): continue
                # 2019-08-16: DIFFUSERAD may contain fewer frequencies than NFREQ but they must
                #             be the *highest* frequencies = last frequencies in the arrays
                # 2020-10-04: was fixed again (at some point the dr_ind formula was again incorrect...)
                dr_ind  =   IFREQ+(DIFFUSERAD.shape[1]-NFREQ)  # actual index to DIFFUSERAD
                if (dr_ind>=DIFFUSERAD.shape[1]): # may contain fewer frequencies !!
                    EMIT[:] = 0.0
                    ## print("dr_ind>=DIFFUSERAD.shape  %d>=%d  .... skip" % (dr_ind, DIFFUSERAD.shape[1]))
                    continue
                # Note -- DIFFUSERAD is emission as photons / 1Hz / cm3
                #         EMIT is number of photons per cell volume (GL*pc)^3/8^level, divided by (GL*pc)^2
                #         =>  EMIT = DIFFUSERAD * (GL*pc)
                # @@ file values DIFFUSERAD not scaled with FACTOR -- they go directly to kernel
                #    (cf. EMIT ~ EMITTED / SCALE)
                for level in range(LEVELS):
                    coeff      =  USER.GL*PARSEC / (8.0**level)    # cell volume in units of GL^3
                    coeff     *=  USER.K_DIFFUSE                   # user-provided scaling of the field
                    a, b       =  OFF[level], OFF[level]+LCELLS[level]
                    EMIT[a:b]  =  DIFFUSERAD[a:b, dr_ind] * coeff  # DIFFUSERAD = photons/Hz/cm3  @@
                if (0):
                    EMIT[nonzero(DENS<1.0e-10)] = 0.0              # empty cells emit nothing
                cl.enqueue_copy(commands[ID], EMIT_buf[ID], EMIT)
                # Emission weighting ---- THIS IS NEVER CALLED vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                if (USER.USE_EMWEIGHT>0):
                    skip += 1
                    if (skip % USER.EMWEIGHT_SKIP==0):
                        tmp      =  asarray(EMIT, float64)
                        if (1):
                            mmm      =  nonzero(~isfinite(tmp))
                            if (len(mmm[0])>0):
                                print('\n     ===========================================================================')
                                print('     EMWEI --- EMISSION NOT FINITE: %d'% len(mmm[0]))
                                tmp[mmm] = 0.0
                                print('     ===========================================================================')
                        tmp[:]   =  DFPAC*tmp/(sum(tmp)+1.0e-32)  # ~ number of packages
                        EMWEI[:] =  clip(tmp, USER.EMWEIGHT_LIM[0], USER.EMWEIGHT_LIM[1])
                        # any EMWEI<1.0 means it has probability EMWEI of being simulated with
                        #   one package, the package weight = 1/p = 1/EMWEI
                        EMWEI[nonzero(rand(CELLS)>EMWEI)] = 0.0 # Russian roulette
                        cl.enqueue_copy(commands[ID], EMWEI_buf[ID], EMWEI)
                        m        = nonzero(EMWEI>0.0)
                        # print('     Update EMWEI -> DFPAC=%d' % len(m[0]))
                        print(' DFPAC=%d ' % len(m[0]))
                        # print('EMWEIGHT', prctile(EMWEI, (0, 10, 50, 90, 100)))
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                
            if (USER.WITH_ROI_SAVE): # clear ROI_SAVE_buf for the next frequency
                cl.enqueue_copy(commands[0], ROI_SAVE_buf, zeros(ROI_SAVE_NPIX, float32))
                
                
            ## t0 = time.time()            
            commands[ID].finish()

            
            #  II==0  =   point sources, PSPAC
            #  II==1  =   background (isotropic of Healpix), BGPAC
            #  II==2  =   constant diffuse emission from the medium (not linked to dust temperature), DFPAC
            #  II==3  =   region-of-interest (ROI) emission 
            
            t0 = time.time()            
            ##   2018-12-08 OPT_buf added for (ABS, SCA) in case of abundance variations
            if (II==2):    #  DFPAC  ---  kernel_ram_cl  ---  [ normal | WITH_ROI_SAVE ]
                if (USER.WITH_ROI_SAVE<=0):
                    kernel_ram_cl(commands[ID], [GLOBAL,], [LOCAL,],
                    np.int32(II), PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], FF,
                    LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                    TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID],
                    EMWEI_buf[ID], INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], EMINDEX_buf[ID],
                    OPT_buf[ID], ABU_buf[ID])
                else:
                    # *** DFPAC + WITH_ROI_SAVE ***
                    print("               WITH_ROI_SAVE --- kernel_ram_cl")
                    kernel_ram_cl(commands[ID], [GLOBAL,], [LOCAL,],
                    np.int32(II), PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], FF,
                    LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                    TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID],
                    EMWEI_buf[ID], INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], EMINDEX_buf[ID],
                    OPT_buf[ID], ABU_buf[ID], ROI_buf, ROI_SAVE_buf)
            elif (II<2):  #  PSPAC, BGPAC  --- either isotropic or Healpix, with photon splitting or not
                if ((II==1)&(USER.DO_SPLIT)):  # BGPAC + split
                    if (len(HPBG)>0):          # using Healpix background sky
                        # 2021-01-31 added this missing combination Healpix background + photon splitting
                        kernel_hp_split(commands[ID], [GLOBAL_SPLIT,], [LOCAL,],
                        PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], FF, LCELLS_buf[ID], OFF_buf[ID], 
                        PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID], TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], 
                        INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], OPT_buf[ID],
                        HPBG_buf[ID], HPBGP_buf[ID], ABU_buf[ID], BUFFER_buf[ID])
                    else:                      # using fully isotropic background sky
                        # 2020-07-16 --- trying package splitting for simulation of background packages
                        kernel_bg_split(commands[ID], [GLOBAL_SPLIT,], [LOCAL,],
                        PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], BG, FF, LCELLS_buf[ID], OFF_buf[ID], 
                        PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID], TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], 
                        INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], OPT_buf[ID], ABU_buf[ID],
                        BUFFER_buf[ID])
                else:   # PSPAC or BGPAC but without photon splitting
                    if ((II==1)&(len(HPBG)>0)):   # BGPAC + Healpix, without photon splitting
                        kernel_ram_hp(commands[ID], [GLOBAL,], [LOCAL,],
                        PACKETS, BATCH, seed, ABS_buf[ID], SCA_buf[ID], FF, LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], 
                        DENS_buf[ID], EMIT_buf[ID], TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID],
                        INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], OPT_buf[ID], 
                        HPBG_buf[ID], HPBGP_buf[ID], ABU_buf[ID])
                    else:
                        # The arguments of kernel_ram_pb are defined by the ROI_LOAD and ROI_SAVE options
                        # although it is only for source ii==2 that ROI_LOAD is actually used
                        #   BGPAC/PSPAC --- kernel_ram_pb  --- [ normal | WITH_ROI_SAVE | WITH_ROI_LOAD | both ]
                        if ((USER.WITH_ROI_SAVE<=0)&(USER.WITH_ROI_LOAD<=0)):   # normal case without ROI
                            kernel_ram_pb(commands[ID], [GLOBAL,], [LOCAL,],
                            # SOURCE      PACKETS  BATCH  SEED  ABS
                            np.int32(II), PACKETS, BATCH, seed, ABS_buf[ID], 
                            # SCA        BG  PSPOS          PS          TW
                            SCA_buf[ID], BG, PSPOS_buf[ID], PS_buf[ID], FF,
                            # LCELLS        OFF          PAR          DENS          EMIT         
                            LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                            # TABS        DSC          CSC          XAB          EMWEI           
                            TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID], EMWEI_buf[ID], 
                            # INT        INTX          INTY          INTZ          OPT           
                            INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], OPT_buf[ID],
                            # ABU        XPS_NSIDE          XPS_SIDE          XPS_AREA           
                            ABU_buf[ID], XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID])
                        elif ((USER.WITH_ROI_SAVE>0)&(USER.WITH_ROI_LOAD>0)):   # both ROI_LOAD and ROI_SAVE
                            kernel_ram_pb(commands[ID], [GLOBAL,], [LOCAL,],
                            # SOURCE      PACKETS  BATCH  SEED  ABS
                            np.int32(II), PACKETS, BATCH, seed, ABS_buf[ID], 
                            # SCA        BG  PSPOS          PS          TW
                            SCA_buf[ID], BG, PSPOS_buf[ID], PS_buf[ID], FF,
                            # LCELLS        OFF          PAR          DENS          EMIT         
                            LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                            # TABS        DSC          CSC          XAB          EMWEI           
                            TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID], EMWEI_buf[ID], 
                            # INT        INTX          INTY          INTZ          OPT           
                            INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], OPT_buf[ID],
                            # ABU        XPS_NSIDE          XPS_SIDE          XPS_AREA           
                            ABU_buf[ID], XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID],
                            ROI_DIM_buf, ROI_LOAD_buf, ROI_buf, ROI_SAVE_buf)
                        elif (USER.WITH_ROI_SAVE>0):                            # only ROI_SAVE
                            #print("               kernel_ram_pb -- WITH_ROI_SAVE %d, WITH_ROI_LOAD %d" % (USER.WITH_ROI_SAVE, USER.WITH_ROI_LOAD))            
                            kernel_ram_pb(commands[ID], [GLOBAL,], [LOCAL,],
                            # SOURCE      PACKETS  BATCH  SEED  ABS
                            np.int32(II), PACKETS, BATCH, seed, ABS_buf[ID], 
                            # SCA        BG  PSPOS          PS          TW
                            SCA_buf[ID], BG, PSPOS_buf[ID], PS_buf[ID], FF,
                            # LCELLS        OFF          PAR          DENS          EMIT         
                            LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                            # TABS        DSC          CSC          XAB          EMWEI           
                            TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID], EMWEI_buf[ID], 
                            # INT        INTX          INTY          INTZ          OPT           
                            INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], OPT_buf[ID],
                            # ABU        XPS_NSIDE          XPS_SIDE          XPS_AREA           
                            ABU_buf[ID], XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID],
                            ROI_buf, ROI_SAVE_buf)
                        else:                                                   # only ROI_LOAD
                            #print("               kernel_ram_pb -- WITH_ROI_SAVE %d, WITH_ROI_LOAD %d" % (USER.WITH_ROI_SAVE, USER.WITH_ROI_LOAD))
                            kernel_ram_pb(commands[ID], [GLOBAL,], [LOCAL,],
                            # SOURCE      PACKETS  BATCH  SEED  ABS
                            np.int32(II), PACKETS, BATCH, seed, ABS_buf[ID], 
                            # SCA        BG  PSPOS          PS          TW
                            SCA_buf[ID], BG, PSPOS_buf[ID], PS_buf[ID], FF,
                            # LCELLS        OFF          PAR          DENS          EMIT         
                            LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                            # TABS        DSC          CSC          XAB          EMWEI           
                            TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID], EMWEI_buf[ID], 
                            # INT        INTX          INTY          INTZ          OPT           
                            INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], OPT_buf[ID],
                            # ABU        XPS_NSIDE          XPS_SIDE          XPS_AREA           
                            ABU_buf[ID], XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID],
                            ROI_DIM_buf, ROI_LOAD_buf)
                            
            elif (II==3):  # II==3  == ROI_LOAD source
                ## print("II=3 -- ROI_LOAD -- USER.WITH_ROI_LOAD=%d" % USER.WITH_ROI_LOAD)
                if (USER.WITH_ROI_LOAD<=0): continue  # no ROI_LOAD photons simulated
                # scale in again the dependence on the grid length
                cl.enqueue_copy(commands[0], ROI_LOAD_buf, asarray(ROI_LOAD[IFREQ,:]*USER.ROI_LOAD_SCALE/(USER.GL*USER.GL), float32))
                commands[ID].finish()
                if (USER.WITH_ROI_SAVE): # argument list has both ROI_LOAD and ROI_SAVE buffers
                    #print("    kernel_ram_pb  --- ROI_LOAD + ROI_SAVE")
                    kernel_ram_pb(commands[ID], [GLOBAL,], [LOCAL,],
                    # SOURCE      PACKETS  BATCH  SEED  ABS
                    np.int32(II), PACKETS, BATCH, seed, ABS_buf[ID], 
                    # SCA        BG  PSPOS          PS          TW
                    SCA_buf[ID], BG, PSPOS_buf[ID], PS_buf[ID], FF,
                    # LCELLS        OFF          PAR          DENS          EMIT         
                    LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                    # TABS        DSC          CSC          XAB          EMWEI           
                    TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID], EMWEI_buf[ID], 
                    # INT        INTX          INTY          INTZ          OPT           
                    INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], OPT_buf[ID],
                    # ABU        XPS_NSIDE          XPS_SIDE          XPS_AREA           
                    ABU_buf[ID], XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID],
                    ROI_DIM_buf, ROI_LOAD_buf, ROI_buf, ROI_SAVE_buf)
                else:                 # only ROI_LOAD
                    #print("    kernel_ram_pb  --- ROI_LOAD only")
                    kernel_ram_pb(commands[ID], [GLOBAL,], [LOCAL,],
                    # SOURCE      PACKETS  BATCH  SEED  ABS
                    np.int32(II), PACKETS, BATCH, seed, ABS_buf[ID], 
                    # SCA        BG  PSPOS          PS          TW
                    SCA_buf[ID], BG, PSPOS_buf[ID], PS_buf[ID], FF,
                    # LCELLS        OFF          PAR          DENS          EMIT         
                    LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                    # TABS        DSC          CSC          XAB          EMWEI           
                    TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID], EMWEI_buf[ID], 
                    # INT        INTX          INTY          INTZ          OPT           
                    INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], OPT_buf[ID],
                    # ABU        XPS_NSIDE          XPS_SIDE          XPS_AREA           
                    ABU_buf[ID], XPS_NSIDE_buf[ID], XPS_SIDE_buf[ID], XPS_AREA_buf[ID],
                    ROI_DIM_buf, ROI_LOAD_buf) 
                
            commands[ID].finish()                
            Tkernel += time.time()-t0 # REALLY ONLY THE KERNEL EXECUTION TIME
                
            Tpost = time.time()
            
            # #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
            if (0): # @@
                print("DONE ONE FREQUENCY: %.2f seconds" % (time.time()-t0))
                cl.enqueue_copy(commands[ID], EMIT, TABS_buf[ID])
                asarray(EMIT, float32).tofile('tabs.1')
                sys.exit()
            # #=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
            
            
                
            if (USER.WITH_ROI_SAVE>0):
                # Add the photons that entered ROI for the current IFREQ into the global array
                tmp= zeros(ROI_SAVE_NPIX, float32)
                cl.enqueue_copy(commands[ID], tmp, ROI_SAVE_buf)
                # scale away the dependence on the current GL
                tmp  *= USER.GL*USER.GL
                ROI_SAVE[IFREQ, :] += tmp   # += because we may have PS and BG and even ROI in this loop
                print("ROI +=  %12.4e\n" % sum(tmp))
                tmp = None

                
            # Save intensity 
            if (USER.SAVE_INTENSITY>0):
                assert(USER.WITH_REFERENCE==0) ; # we want TABS_buf to contain the total field
            if ((USER.SAVE_INTENSITY==1)|(not(USER.NOABSORBED))): # scalar intensity in each cell
                cl.enqueue_copy(commands[ID], TMP, INT_buf[ID])   # ~ total field
                commands[ID].finish()
                if (not(USER.NOABSORBED)):
                    FABSORBED[:, OIFREQ] += TMP ;  # all scalings later; using OIFREQ <= IFREQ (in case LIB_ABS)
                if (USER.SAVE_INTENSITY==1):
                    for level in range(LEVELS):
                        coeff                  =  KDEV * (PLANCK*FREQ/ABS) * (8.0**level)
                        a, b                   =  OFF[level], OFF[level]+LCELLS[level]
                        # += because we have PS, BG, diffuse emission separately
                        INTENSITY[a:b, IFREQ] +=  coeff * TMP[a:b] / DENS[a:b]
            if (USER.SAVE_INTENSITY==2):
                for icomp in range(4):
                    BUFS = [INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID]]
                    cl.enqueue_copy(commands[ID], TMP, BUFS[icomp])
                    for level in range(LEVELS):
                        coeff =  KDEV * (PLANCK*FREQ/ABS) * (8.0**level)
                        a, b  =  OFF[level], OFF[level]+LCELLS[level]
                        if (1):
                            print("  <INTENSITY[k=%d]> = %11.3e  *= %10.3e" % (icomp, mean(TMP[a:b]), coeff ))
                        INTENSITY[a:b, IFREQ, icomp] += coeff * TMP[a:b]  / DENS[a:b]
                        # this will be final for INTENSITY[:,:,0] 
                        #   (except for added cell emission below!)
                        # others will be later divided by INTENSITY[:,:,0]
                
            Tpost = time.time()-Tpost
            sys.stdout.write('   %7.2f\n' % (time.time()-T000))
            #sys.stdout.write(' push %7.2f pre %7.2f\n' % (Tpush, Tpre))
            sys.stdout.flush()
                
            if (0): sys.exit()    # debugging...
            
        # FOR FREQUENCY ======
        
                                
        # pull TABS and store to CTABS  --- TABS = integrated energy [CELLS] for this II component!!
        cl.enqueue_copy(commands[ID], EMIT, TABS_buf[ID])
        commands[ID].finish()
        # end of -- for IFREQ
        CTABS  +=   EMIT   # integrated energy -- sum of all constant components

        # @@
        if (len(CTABS)<1e8):
            print('='*80)
            print("******  CONSTANT   %10s   CTABS -> %12.4e" % (['PS', 'BG', 'DE', 'ROI'][II], mean(CTABS)))
            print('='*80)
        
    # end of -- for II = three kinds of constant sources
    # CTABS contains the absorbed energy due to sources all except the dust in the medium
    if (len(USER.file_constant_save)>0): 
        # asarray(CTABS, float32).tofile(USER.file_constant_save)
        CTABS.tofile(USER.file_constant_save)

    
    ### sys.exit()
    
    
        
if (len(CTABS)<1e8):
    print("FINAL CTABS  = %.3e" % mean(CTABS))
        
        
# Next simulate the emission from the dust within the model volume, possibly over many iterations
# Keep updating the escape probability on each iteration, because temperatures change and
# (with reference field) to make the estimates more precise.

BATCH    = int(CLPAC/CELLS)
EMDONE   = None




if (USER.WITH_REFERENCE==1):
    # We would not want to zero these in case one is continuing on a previous run
    # .... now reference field is effective only on later iterations of a single run
    print('*'*80)
    print("WITH_REFERENCE...")
    OEMITTED[:,:] = 0.0    #  [CELLS, REMIT_NFREQ] ... zero on first iteration
    OTABS[:]      = 0.0    #  [CELLS,]
else:
    if ((USER.WITH_ALI)&(USER.WITH_REFERENCE%100>0)):  # restore old OXAB, OXEM
        try:
            OXAB[:] = fromfile('OXAB.save', float32)
            OXEM[:] = fromfile('OXEM.save', float32)
        except:
            pass


        
"""
Note --- WITH_ROI_SAVE not implemented (yet) for dust reemission !!
"""
    
if (not('SUBITERATIONS' in USER.KEYS)):
    for iteration in range(USER.ITERATIONS):  # iteratio over simulation <-> T-update cycles
        print("ITERATION %d/%d" % (iteration+1, USER.ITERATIONS))
        
        
        commands[ID].finish()
        # zero TABS, XABS, INT, INTX, INTY, INTZ
        kernel_zero(commands[ID],[GLOBAL,],[LOCAL,],np.int32(0),TABS_buf[ID],XAB_buf[ID],INT_buf[ID],INTX_buf[ID],INTY_buf[ID],INTZ_buf[ID])
        commands[ID].finish()
        if (USER.WITH_ALI): 
            XEM[:] = 1.0e-32
    
        if (USER.WITH_REFERENCE):
            # simulation for  true - old, this should decrease noise so there is no
            # need for explicit weighting of OEMITTED !
            if (1):
                print(" [22]   OEMITTED  %12.4e   OTABS %12.4e" % (mean(OEMITTED[22,:]), mean(OTABS)))
                if (1):
                    # This weighting means that reference field is nulled at first iteration
                    # --> noise suppressed only on later iterations of a single run
                    if (USER.WITH_REFERENCE==1):
                        k           = iteration/float(0.0+USER.ITERATIONS)
                    else:
                        # encode in WITH_ITERATIONS  AABB where
                        # AA is the total number of iterations and the first iteration of this run is BB
                        # -> makes possible to continue iterations without resetting the effect of the 
                        # reference field??
                        wr_fir =  int(USER.WITH_REFERENCE%100)
                        wr_tot =  int(floor(0.01*USER.WITH_REFERENCE))
                        print('*'*80)
                        print('WITH_REFERENCE %d  --- TOTAL %d, CURRENT = %d + %d' % \
                        (USER.WITH_REFERENCE, wr_tot, wr_fir, iteration))
                        print('*'*80)
                        k           = (iteration+wr_fir) / float(wr_tot)
                else:
                    k           = 0.5
                OEMITTED[:,:]  *= k   # [CELLS, REMIT_NFREQ]
                OTABS[:]       *= k   # [CELLS],  not including CTABS
    
        if ('OLDKERNEL' in USER.KEYS):
            # RAM version --- one work item per cell
            GLOBAL, BATCH  =  Fix(CELLS, LOCAL),    int(CLPAC/CELLS)
        else:
            # RAM2 version --- each work item loops over cells =>  GLOBAL<CELLS, BATCH==packets per cell
            GLOBAL, BATCH  =  GLOBAL_0,  max([1,int(CLPAC/CELLS)])
            print('=== CLPAC %d (%d), GLOBAL %d, BATCH %d' % (CLPAC, CELLS*BATCH, GLOBAL, BATCH))
                
        skip = USER.EMWEIGHT_SKIP-1
        if (CLPAC>0):
            OIFREQ = -1
            for IFREQ in range(NFREQ):     # loop over single frequencies (all the frequencies)
                FREQ = FFREQ[IFREQ]
                
                commands[ID].finish()
                kernel_zero(commands[ID],[GLOBAL,],[LOCAL,],np.int32(1),TABS_buf[ID],XAB_buf[ID],INT_buf[ID],INTX_buf[ID],INTY_buf[ID],INTZ_buf[ID])
                commands[ID].finish()
                
                if (USER.LIB_ABS): # simulate only reference frequencies in FSELECT
                    if (np.min(abs((FREQ-USER.FSELECT)/FREQ))>0.001): continue
                OIFREQ += 1
                
                # Parameters for the current frequency
                if ((FREQ<USER.SIM_F[0])|(FREQ>USER.SIM_F[1])): continue # FREQ not simulated!!
                
                print("  FREQ %3d/%3d  %10.3e" % (IFREQ+1, NFREQ, FREQ))
            
                G = 0.0  # not used !
                if (WITH_ABU>0): # ABS, SCA vectors
                    OPT[:,:] = 0.0
                    if (USER.SINGLE_ABU):
                        OPT[:,0]  +=       ABU  * AFABS[0][IFREQ]
                        OPT[:,1]  +=       ABU  * AFSCA[0][IFREQ]
                        OPT[:,0]  +=  (1.0-ABU) * AFABS[1][IFREQ]
                        OPT[:,1]  +=  (1.0-ABU) * AFSCA[1][IFREQ]
                    else:
                        for idust in range(1, NDUST):
                            OPT[:,0]  +=  ABU[:,idust] * AFABS[idust][IFREQ]
                            OPT[:,1]  +=  ABU[:,idust] * AFSCA[idust][IFREQ]
                    if (USER.OPT_IS_HALF):
                        cl.enqueue_copy(commands[ID], OPT_buf[ID], asarray(OPT, np.float16))
                    else:
                        cl.enqueue_copy(commands[ID], OPT_buf[ID], OPT)
                    if (WITH_MSF):
                        for idust in range(NDUST):
                            ABS[idust] = AFABS[idust][IFREQ]
                            SCA[idust] = AFSCA[idust][IFREQ]                        
                else:               # constant abundance, ABS, SCA, G are scalar
                    ABS[0], SCA[0] = 0.0, 0.0
                    for idust in range(NDUST):
                        ABS[0] += AFABS[idust][IFREQ]
                        SCA[0] += AFSCA[idust][IFREQ]
                cl.enqueue_copy(commands[ID], ABS_buf[ID], ABS)                
                cl.enqueue_copy(commands[ID], SCA_buf[ID], SCA)

                
                if (WITH_ROI_SAVE): # clear ROI_SAVE_buf for the next frequency
                    cl.enqueue_copy(commmands[0], ROI_SAVE_buf, zeros(ROI_SAVE_NPIX, float32))
                                    
                    
                ## BG, PS             =  0.0, 0.0
                ## if (len(IBG)==NFREQ):  BG = IBG[IFREQ] * WBG/FREQ  # background photons per package
                ## if (len(LPS)==NFREQ):  PS = LPS[IFREQ] * WPS/FREQ  # source photons per package
                # The weight for on-the-fly trapezoid integration
                FF = FREQ 
                if (IFREQ==0):              FF *= 0.5*(FFREQ[1]-FFREQ[0])
                else:
                    if (IFREQ==(NFREQ-1)):  FF *= 0.5*(FFREQ[NFREQ-1]-FFREQ[NFREQ-2])
                    else:                   FF *= 0.5*(FFREQ[IFREQ+1]-FFREQ[IFREQ-1])

                # Upload to device new DSC, CSC
                t0 = time.time()
                commands[ID].finish()
                if (WITH_MSF==0):
                    cl.enqueue_copy(commands[ID], DSC_buf[ID], FDSC[0,IFREQ,:])
                    cl.enqueue_copy(commands[ID], CSC_buf[ID], FCSC[0,IFREQ,:])
                else:
                    for idust in range(NDUST):
                        DSC[idust,:] = FDSC[idust, IFREQ, :]
                        CSC[idust,:] = FCSC[idust, IFREQ, :]
                    cl.enqueue_copy(commands[ID], DSC_buf[ID], DSC)
                    cl.enqueue_copy(commands[ID], CSC_buf[ID], CSC)
                commands[ID].finish()
                Tpush += time.time()-t0
                ###
                if ((IFREQ<REMIT_I1)|(IFREQ>REMIT_I2)): # this frequency not in EMITTED !!
                    sys.stdout.write('\n')
                    continue  # cannot simulate emission that is not in EMITTED
                ###
                if (USER.WITH_REFERENCE):
                    EMIT[:]  =  EMITTED[:, IFREQ-REMIT_I1] - OEMITTED[:, IFREQ-REMIT_I1]
                    
                    if (IFREQ==-22): print(" [22]  EMIT  %12.4e - %12.4e = %12.4e" %
                     (mean(EMITTED[:, IFREQ-REMIT_I1]),
                     mean(OEMITTED[:, IFREQ-REMIT_I1]),
                     mean(EMIT)))

                    OEMITTED[:, IFREQ-REMIT_I1] = 1.0*EMITTED[:,IFREQ-REMIT_I1] # -> OTABS
                else:
                    EMIT[:]  =  EMITTED[:, IFREQ-REMIT_I1]    # total emission
                    
                    if (IFREQ==-22): print(" [22]  EMIT  %12.4e" % (mean(EMIT)))
                    
                # EMITTED / FACTOR  =>  EMIT  .... file has unit FACTOR*photons/Hz/cm3
                # kernel uses "actual" values, after division with FACTOR
                for level in range(LEVELS):
                    coeff      = USER.GL*PARSEC/(8.0**level)/FACTOR  #  cell volume in units of GL^3
                    a, b       = OFF[level], OFF[level]+LCELLS[level]
                    EMIT[a:b] *= coeff*DENS[a:b]
                EMIT[nonzero(DENS<1.0e-10)] = 0.0    # empty cells emit nothing
                
                # XEM = integral of emitted energy
                if (USER.WITH_ALI):   XEM  += EMIT*FF
                
                if (USER.USE_EMWEIGHT>0):
                    skip += 1
                    if (skip % USER.EMWEIGHT_SKIP==0):
                        # weighting according to total emission, even if reference field used!
                        tmp      =  asarray(EMITTED[:,IFREQ-REMIT_I1].copy(), float64)
                        # print 'EMITTED ', np.percentile(tmp, (0.0, 10.0, 90.0, 100.0))
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
                        if (USER.USE_EMWEIGHT==2):
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
                        # print('EMWEIGHT', prctile(EMWEI, (0, 10, 50, 90, 100)))

                        if (iteration==-1):
                            if (IFREQ>=24):
                                clf()
                                hist(log10(EMWEI+0.1), 100)
                                title('IFREQ %d' % IFREQ)
                                savefig('EMWEI_IFREQ_%02d.png' % IFREQ)
                                sys.exit()                                                    
                        
                commands[ID].finish()                
                
                # EMIT <-   TRUE - OEMITTED    ...... EMIT ~ EMITTED / FACTOR !!!
                cl.enqueue_copy(commands[ID], EMIT_buf[ID], EMIT)
                #  out comes difference TABS = TRUE - OTABS
                if (USER.SEED>0): seed = fmod(USER.SEED+(DEVICES*IFREQ+ID)*SEED1, 1.0)
                else:             seed = rand()
                
                commands[ID].finish()
                t0 = time.time()
                
                #----------------------------------------------------------------------------------------
                if (USER.USE_EMWEIGHT==2):
                    # Host loops until EMDONE[CELLS]==EMPAC[CELLS], kernel simulates packages
                    # for cells listed in EMINDEX[<=CELLS], package weights already in EMWEI[CELLS]
                    EMDONE  = zeros(CELLS, int32)
                    EMINDEX = zeros(CELLS, int32)
                    loops = 0
                    while 1:
                        m     = nonzero(EMDONE<EMPAC) # cells that should send packages
                        no    = len(m[0])
                        loops += 1
                        # if (loops%10==0): print ' loops ', loops
                        if (no<1): break  # all done
                        # print '.... EMITTING CELLS: %d' % (len(m[0]))
                        # print '.... 3789 has %d packets' % (EMPAC[3789]-EMDONE[3789])
                        # print '.... EMINDEX[0] = %d' EMINDEX[0]
                        EMINDEX[0:no]     =  m[0]
                        EMINDEX[no:CELLS] = -1
                        cl.enqueue_copy(commands[ID], EMINDEX_buf[ID], EMINDEX)
                        kernel_ram_cl(commands[ID], [8192,], [LOCAL,],
                        np.int32(2),  CLPAC, BATCH, seed, ABS_buf[ID], SCA_buf[ID],
                        FF,
                        LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                        TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID], EMWEI_buf[ID],
                        INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID],
                        EMINDEX_buf[ID], OPT_buf[ID])  # <-- one additional parameter in case of USE_EMWEIGHT==2
                        commands[ID].finish()
                        EMDONE[m] += EMWEI2_STEP  # kernel did () packages for each cell in EMINDEX
                        
                else:
                    
                    if (USER.WITH_ROI_SAVE<=0):
                        print(" **** CALLING kernel_ram_cl() !!!!!!")
                        # A single call to the kernel
                        kernel_ram_cl(commands[ID], [GLOBAL,], [LOCAL,],
                        # 0           1      2      3     4            5             6  
                        np.int32(2),  CLPAC, BATCH, seed, ABS_buf[ID], SCA_buf[ID],  FF,
                        # 11            12           13           14            15          
                        LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                        # 16          17           18           19           20             
                        TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID], EMWEI_buf[ID],
                        # 21         22            23            24      
                        INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], EMINDEX_buf[ID], OPT_buf[ID],
                        ABU_buf[ID])
                    else:
                        # *** DFPAC + WITH_ROI_SAVE ***
                        print("WITH_ROI_SAVE --- kernel_ram_cl")                    
                        # A single call to the kernel
                        kernel_ram_cl(commands[ID], [GLOBAL,], [LOCAL,],
                        # 0           1      2      3     4            5             6  
                        np.int32(2),  CLPAC, BATCH, seed, ABS_buf[ID], SCA_buf[ID],  FF,
                        # 11            12           13           14            15          
                        LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
                        # 16          17           18           19           20             
                        TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID], EMWEI_buf[ID],
                        # 21         22            23            24      
                        INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID], EMINDEX_buf[ID], OPT_buf[ID],
                        ABU_buf[ID],  ROI_buf, ROI_SAVE_buf)
                    
                commands[ID].finish()
                #----------------------------------------------------------------------------------------
                Tkernel += time.time()-t0
    
                if (0):
                    if ((time.time()-t0)>7200): # something wrong ??
                        print('EXIT !!!!!!!!!!!!!')
                        sys.exit()
                                                
                # ------------------------------------------------------------------------------
                # Intensity
                if (iteration==(USER.ITERATIONS-1)):
                    
                    if (USER.SAVE_INTENSITY>0):
                        assert(USER.WITH_REFERENCE==0) ; # we want TABS_buf to contain the total field
                    if ((USER.SAVE_INTENSITY==1)|(not(USER.NOABSORBED))):  # scalar intensity in each cell
                        cl.enqueue_copy(commands[ID], TMP, INT_buf[ID])    # ~ total field
                        commands[ID].finish()                    
                        if (not(USER.NOABSORBED)):
                            FABSORBED[:,OIFREQ] += TMP
                        if (USER.SAVE_INTENSITY==1):
                            for level in range(LEVELS):
                                coeff =  KDEV * (PLANCK*FREQ/ABS) * (8.0**level)
                                a, b  =  OFF[level], OFF[level]+LCELLS[level]
                                INTENSITY[a:b, IFREQ] += coeff * TMP[a:b] / DENS[a:b]
                    if (USER.SAVE_INTENSITY==2):
                        for icomp in range(4):
                            BUFS = [INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID]]
                            cl.enqueue_copy(commands[ID], TMP, BUFS[icomp])  # ~ total field
                            commands[ID].finish()                        
                            for level in range(LEVELS):
                                coeff =  KDEV * (PLANCK*FREQ/ABS) * (8.0**level)
                                a, b  =  OFF[level], OFF[level]+LCELLS[level]
                                INTENSITY[a:b, IFREQ, icomp]  += coeff * TMP[a:b]  / DENS[a:b]
                                
                    if (USER.WITH_ROI_SAVE>0):
                        # Add the photons that entered ROI into the global array
                        tmp= zeros(ROI_SAVE_NPIX, float32)
                        cl.enqueue_copy(commands[ID], tmp, ROI_SAVE_buf)
                        ROI_SAVE[IFREQ, :] += tmp
                        tmp = None
                                
                # ------------------------------------------------------------------------------
            # end of -- for IFREQ
            
            
            # TABS is the integrated absorbed energy for emission from the medium --- IF USER_NOABSORBED==True
            commands[ID].finish()
            
            
            if (USER.WITH_REFERENCE):
                #   OEMITTED has been updated
                #   Calculate escape probability beta = (XEM+OXEM-(XAB+OXAB))/(XEM+OXEM), 
                #   that is, we can start by updating OXEM and OXAB
                #   XAB == is on device the DELTA = TRUE - REFERENCE reabsorbed energy
                if (USER.WITH_ALI):
                    cl.enqueue_copy(commands[ID], EMIT, XAB_buf[ID])  # DELTA XAB
                    commands[ID].finish()
                    OXAB[:]  += EMIT       # @    OXAB + DELTA XAB
                    OXEM[:]  += XEM        # @    OXEM + DELTA XEM
                    TMP[:]    = (OXEM-OXAB)/OXEM  # escape probability
                # all information of the reference field has been updated
                #  OEMITTED, OXAB, OXEM, OTABS
            else:
                if (USER.WITH_ALI):
                    cl.enqueue_copy(commands[ID], EMIT, XAB_buf[ID])  # full XAB
                    commands[ID].finish()                
                    TMP[:]  = (XEM-EMIT)/XEM  # escape probability
                
            
                    
            if (USER.NOABSORBED): # WE SOLVE EMISSION WITHIN SOC
                # TABS_buf == ***integrated*** absorbed energy, total (REFERENCE=0) or delta (REFERENCE=1)
                #            possibly ignoring absorptions in emitting cell (WITH_ALI)
                #            ... only for CLPAC part of the simulation
                commands[ID].finish()
                cl.enqueue_copy(commands[ID], EMIT, TABS_buf[ID]) # could be negative for reference field
                commands[ID].finish()
            
                if (iteration==-1):
                    clf()
                    subplot(221)
                    x  = DENS
                    y1 = OTABS
                    y2 = EMIT
                    m = nonzero(DENS>0.0)
                    plot(x[m], y1[m], 'bx')
                    plot(x[m], y2[m], 'r+')

            
                if (USER.WITH_REFERENCE):
                    # OEMITTED lead to OTABS,  TABS corresponds to EMITTED-OEMITTED = delta
                    #  (but OEMITTED has been updated in the mean time)
                
                    print(" TABS  = %12.4e = %12.4e + %12.4e" % (mean(EMIT+OTABS), mean(EMIT), mean(OTABS)))
                    # kernel calculated TABS from EMITTED-OEMITTED, add back the OTABS ~ OEMITTED
                    EMIT[:]  +=  OTABS[:]  # true TABS == delta TABS + OTABS
                    OTABS[:]  =  1.0*EMIT  # OTABS = current TABS for current (updated) OEMITTED
                    EMIT[:]  +=  CTABS     # TRUE = TABS + OTABS + CTABS
                    # EMIT re-used for absorptions --- at this point 
                    # EMIT  ==  sum of TABS components ==  OLD + DELTA + CONSTANT  absorptions
                else:
                    # Here EMIT becomes the sum of all absorptions due to constant sources plus the medium                
                    print("  TABS  = %12.4e  ... ADD CTABS % 12.4e" % (mean(EMIT), mean(CTABS)))                
                    EMIT[:]  +=  CTABS   # add absorbed energy due to constant radiation sources

            else:
                # USER.NOABSORBED==False -- we are only saving FABSORBED to a file
                pass
                    
        else:  # else CLPAC==0 .... EMIT == CTABS
            if (USER.NOABSORBED):   # NOABSORBED == SOLVE IN SOC
                EMIT[:] = 1.0*CTABS[:]
            else:
                pass     # absorptions saved to file ... do nothing here

            
            
        scale   = (6.62607e-27*FACTOR)/(USER.GL*PARSEC)     # 1.0e20f*PLANCK / (GL*PARSEC)
        #if (not(USER.NOSOLVE)):
        #    # Solve temperatures on the host side -- total absorbed energy in array EMIT[]
        #    # NOTE -- can solve within SOC only when calculation includes a single non-stochastic dust !!
        #    oplgkE  = 1.0/log10(kE)
        beta   = 1.0
    
        if (0):  # 2020-07-09 commented out -- to save memory !!
            mmm   = nonzero(~isfinite(EMIT))
            if (len(mmm[0]>0)):
                print('******* EMIT NOT FINITE *******', len(mmm[0]))
            EMIT  = clip(EMIT, 1.0e-25, 1e32)
    
        ###if (len(EMIT)<1e6):
        print("====================================================================================================")
        print(" TABS=EMIT  %.3e %.3e %.3e %.3e %.3e" % tuple(np.percentile(EMIT, (0.0, 10.0, 50.0, 90.0, 100.0))))
        print("====================================================================================================")
        
        
        

        if (not(USER.NOSOLVE)):
            # This works only with USER.NOABSORBED=True, when we have array EMIT that 
            # at this point already contains the integrated absorbed enery
            print('Calculate temperatures')
            TNEW = zeros(CELLS, float32)
            if (NDUST>1):
                print("Internal solver in SOC can only deal with a single equilibrium-temperature dust.")
                print("In case of multiple dust components, use A2E to calculate the emission and then")
                print("again SOC to write maps.")
                sys.exit()
            t0 = time.time()
            if (('CLT' in USER.KEYS)&(not(USER.WITH_ALI))):    # calculate temperatures on device
                kernel_T = program[0].EqTemperature
                # EMIT + DENS  ->  TABS
                #                               level     adhoc       kE          Emin        NE        OFF    LCELLS  TTT    DENS   EMIT   TNEW
                kernel_T.set_scalar_arg_dtypes([np.int32, np.float32, np.float32, np.float32, np.int32, None,  None,   None,  None,  None,  None ])
                TTT_buf  = cl.Buffer(context[0], mf.READ_ONLY,  TTT.nbytes)
                print("DENS_buf", DENS_buf[0])
                print("TABS_buf", TABS_buf[0])
                print("EMIT_buf", EMIT_buf[0])
                print("push TTT ...", len(TTT), NE)
                cl.enqueue_copy(commands[0], TTT_buf, TTT)
                print("push EMIT ...")
                cl.enqueue_copy(commands[0], EMIT_buf[0], EMIT)
                print("Calling kernel_T...")
                for l in range(LEVELS):
                    print("Level %d ..." % l)
                    kernel_T(commands[0], [GLOBAL,], [LOCAL,], l, ADHOC, kE, Emin, NE,
                    OFF_buf[0], LCELLS_buf[0], TTT_buf, DENS_buf[0], EMIT_buf[0], TABS_buf[0])
                    print("   ... call ready")
                    print("Finish...")
                    commands[0].finish()
                    print(" ... Finished")
                print("Copy TNEW to host ....")    
                cl.enqueue_copy(commands[0], TNEW, TABS_buf[0])
                print("   ... ready")
                
            elif (not('MPT' in USER.KEYS)):
                for level in range(LEVELS):
                    print('LEVEL %d, LCELLS %d, IND [%d, %d[' % (level, LCELLS[level], OFF[level], OFF[level]+LCELLS[level]))
                    for i in range(LCELLS[level]):
                        ind     =  OFF[level]+i
                        if (DENS[ind]<1.0e-10): 
                            TNEW[ind] = 0.0
                            continue # skip empty cells and parent cells
                        told    =  TNEW[ind]
                        # @@ EMIT scaled by scale i.e. FACTOR, because TTT calculation also inlcuded FACTOR in Ein
                        Ein     = (scale/ADHOC)*EMIT[ind]*(8.0**level)/DENS[ind] # "EMIT" == absorbed energy
                        # if (1): print("%6d   DENS %10.3e-- Ein %10.3e" % (ind, DENS[ind], Ein))
                        #  Ein = beta*Eout  =>   Ein/beta = Eout
                        if (USER.WITH_ALI):
                            beta    =  TMP[ind]   
                        # print Emin, beta, oplgkE
                        iE      =  int(clip(floor(oplgkE * log10((Ein/beta)/Emin)), 0, NE-2))
                        wi      = (Emin*pow(kE,iE+1)-(Ein/beta)) / (Emin*pow(kE,(iE+1))-pow(kE,iE))
                        tnew    =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1]                
                        if (0):
                            if (USER.WITH_ALI): # optional iteration with updated beta values
                                tau     =  AFABS[0][-1]*DENS[ind]
                                for j in range(2): # iterate by making correction for beta=beta(T)
                                    # re-estimate escape probability after the temperature update
                                    # if T increases ->  beta decreases -> T increase further 
                                    beta  *=  beta_interpoler(tnew, tau) / beta_interpoler(told, tau)
                                    iE     =  clip(int(floor(oplgkE * log10((Ein/beta)/Emin))), 0, NE-2)
                                    wi     = (Emin*pow(kE,iE+1)-(Ein/beta)) / (Emin*(pow(kE,(iE+1))-pow(kE,iE))) ;
                                    told   =  tnew
                                    tnew   =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1] ;    
                        TNEW[ind]  = tnew
                        # print("Ein %10.3e   Tnew %10.3e" % (Ein, TNEW[ind]))
            else:
                # Calculate temperatures in parallel
                def MP_temp(a0, a, b, EIN, DENS, kk, MPTNEW, TMP=[]):  # TNEW = single level
                    if (len(TMP)<1): # no ali
                        for i in range(a, b):
                            if (DENS[i]<=0.0): 
                                MPTNEW[i-a0] =  0.0
                                continue
                            Ein       = (EIN[i]/DENS[i])*kk
                            iE        =  int(clip(floor(oplgkE * log10(Ein/Emin)), 0, NE-2))
                            wi        = (Emin*pow(kE,iE+1)-Ein) / (Emin*pow(kE,(iE+1))-pow(kE,iE))
                            MPTNEW[i-a0] =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1]
                    else:
                        for i in range(a, b):
                            if (DENS[i]<=0.0):
                                MPTNEW[i-a0] = 0.0
                                continue
                            Ein       = (EIN[i]/DENS[i])*kk
                            beta      = TMP[i]
                            iE        =  int(clip(floor(oplgkE * log10((Ein/beta)/Emin)), 0, NE-2))
                            wi        = (Emin*pow(kE,iE+1)-(Ein/beta)) / (Emin*pow(kE,(iE+1))-pow(kE,iE))
                            MPTNEW[i-a0] =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1]
                    return
                
                for level in range(LEVELS):
                    a, b    = OFF[level], OFF[level]+LCELLS[level]  #  [a, b[
                    kk      = (scale/ADHOC)*(8.0**level)
                    manager = mp.Manager()
                    PROC    = []
                    n       = (LCELLS[level]+NCPUS)/NCPUS + 1   # cells per single thread
                    MPTNEW  = mp.Array('f', int(LCELLS[level])) # T for cells of a single level
                    for i in range(NCPUS):
                        if (USER.WITH_ALI):
                            p   = mp.Process(target=MP_temp, 
                            args=(a, a+i*n, min([b, a+(i+1)*n]), EMIT, DENS, kk, MPTNEW, TMP))
                            # args=(a, a+i*n, min([b, a+(i+1)*n]), EMIT, DENS, kk, TMP))
                        else:
                            p   = mp.Process(target=MP_temp, 
                            args=(a, a+i*n, min([b, a+(i+1)*n]), EMIT, DENS, kk, MPTNEW, []))
                            # args=(a, a+i*n, min([b, a+(i+1)*n]), EMIT, DENS, kk, []))
                        PROC.append(p)
                        p.start()
                    for i in range(NCPUS):
                        PROC[i].join()                
                    TNEW[a:b] = MPTNEW

                    
            if (len(TNEW)<1e8):
                print('CHECKING TEMPERATURES')
                mok  = nonzero(DENS>1.0e-8)
                mbad = nonzero(~isfinite(TNEW))
                print('    TNEW not finite: %d' % len(mbad[0]))
                for iii in mbad[0]:
                    print('   .... bad %8d  n=%10.3e  T=%.3e  EMIT=ABS=%.3e' % (iii, DENS[iii], TNEW[iii], EMIT[iii]))
                TNEW[mbad] = 10.0
                print('    T < 3K   : %d' % (len(nonzero(TNEW[mok]<3.0)[0])))
                print('    T >1599K : %d' % (len(nonzero(TNEW[mok]>1599.0)[0])))
                TNEW[mok] = clip(TNEW[mok], 3.0, 1600.0)
                if (len(TNEW)<2e7):
                    print('    TNEW: %.3e %.3e %.3e  ' % tuple(np.percentile(TNEW[mok], (0, 50, 100))))
            
                    
            # Save temperatures -- file format is the same as for density (including links!)
            print('Save temperatures')
            if (len(USER.file_temperature)>1):
                fp = open(USER.file_temperature, "wb")
                asarray([NX, NY, NZ, LEVELS, CELLS], int32).tofile(fp)
                for level in range(LEVELS):
                    a, b = OFF[level], OFF[level]+LCELLS[level]
                    asarray([LCELLS[level],], int32).tofile(fp)
                    TNEW[a:b].tofile(fp)
                fp.close()

                
            
        # Calculate emission
        if (not(USER.NOSOLVE)):
            # If we have NOSOLVE==False, we have calculate TNEW above
            FTMP = zeros(REMIT_NFREQ, float64)
            if ('CLE' in USER.KEYS):                 # do emission calculation on device
                kernel_emission = program[0].Emission
                # FREQ, FABS,   DENS, T, EMIT
                kernel_emission.set_scalar_arg_dtypes([np.float32, np.float32, None, None, None])
                cl.enqueue_copy(commands[0], EMIT_buf[0], TNEW)
                # Note:  EMIT_buf is reused for temperature, TABS_buf for the resulting emission
                #        ... because EMIT_buf is read only!
                a, b = REMIT_I1, REMIT_I2+1
                for ifreq in range(a, b):
                    kernel_emission(commands[0], [GLOBAL,], [LOCAL,], float(FFREQ[ifreq]), float(AFABS[0][ifreq]),
                    #         *** T ***  *** emission ***
                    DENS_buf[0], EMIT_buf[0],  TABS_buf[0])
                    cl.enqueue_copy(commands[0], EMIT, TABS_buf[0])
                    # EMITTED ~ directly EMIT, no scaling with FACTOR --- *= FACTOR IS ALREADY IN THE KERNEL !!!
                    EMITTED[:, ifreq-REMIT_I1] = EMIT  #  FACTOR x PHOTONS / Hz / cm3
                ####
            elif (not('MPE' in USER.KEYS)):          # emission with single host thread
                for icell in range(CELLS):          # loop over the GLOBAL large array
                    if (DENS[icell]<1.0e-10):
                        FTMP[:] = 0.0
                    else:
                        a, b  =  REMIT_I1, REMIT_I2+1
                        # @@ EMITTED is again the value scaled with FACTOR (= as stored the file)
                        FTMP[0:REMIT_NFREQ] =  (FACTOR*4.0*np.pi/(PLANCK*FFREQ[a:b])) * \
                        AFABS[0][a:b]*Planck(FFREQ[a:b], TNEW[icell])/(USER.GL*PARSEC) #  1e20*photons/H -> FACTOR
                    EMITTED[icell,:] = FTMP[0:REMIT_NFREQ]  # EMITTED[CELLS, REMIT_NFREQ]
            else:                                    # emission multithreaded on host
                def MP_emit(ifreq, ithread, ncpu, TNEW, MPA):
                    ii   =  int(REMIT_I1 + ifreq + ithread)  # index of the frequency
                    if (ii>REMIT_I2):  # no such channel
                        return
                    kk   =  (FACTOR*4.0*np.pi/(PLANCK*FFREQ[ii])) * AFABS[0][ii] / (USER.GL*PARSEC)
                    freq =  FFREQ[ii]
                    MPA[ithread][:] = kk*PlanckTest(freq, TNEW) #  1e20*photons/H -> FACTOR photons/H
                ###
                manager = mp.Manager()
                TNEW[nonzero(TNEW<3.0)] = 10.0  # avoid warnings about parent cells
                MPA     = [ mp.Array('f', int(CELLS)), mp.Array('f', int(CELLS)), mp.Array('f', int(CELLS)), mp.Array('f', int(CELLS))]
                for ifreq in range(REMIT_I1, REMIT_I2+1, NCPUS):
                    PROC      = []
                    for i in range(NCPUS):
                        p   = mp.Process(target=MP_emit,args=(ifreq, i, NCPUS, TNEW, MPA))
                        PROC.append(p)
                        p.start()
                    for i in range(NCPUS):
                        PROC[i].join()
                    for i in range(NCPUS):
                        if ((ifreq+i)<=REMIT_I2):
                            EMITTED[:,ifreq+i] = MPA[i][:]
                    
            del FTMP            
            Tsolve = time.time()-t0
        
        # end of -- for iteration
        if (0):
            clf()
            hist(TNEW[nonzero(TNEW>0.0)], linspace(3.0, 50.0, 100))
            show()
            sys.exit()

    # Save OTABS, OEMITTED??
    if ((USER.WITH_REFERENCE>1)&(not(USER.NOSOLVE))):
        asarray(OEMITTED, float32).tofile('OEMITTED.save')
        asarray(OTABS, float32).tofile('OTABS.save')
        print('2'*80)
        print('2 OEMITTED %12.4e, OTABS %12.4e' % (mean(ravel(OEMITTED)), mean(OTABS)))
        print('2'*80)


        
if ('SUBITERATIONS' in USER.KEYS):
    """
    $$$
    Sub-iterations with reference field. This is for simplicity separate from the
    not(USER.SUBITERATIONS) alternative above:
        only with reference field, only without ALI, only with subiterations,
        only with emission weighting, only for CLPAC>0
        
    Simple version:
        - assume that cold cells can more or less be ignored
        - iteration=0, full simulation -> (T, dT) specify active cells
        - continue with normal REFERENCE field method
        - finish with a single iteration again with all cells
    """
    assert((USER.WITH_REFERENCE)&(not(USER.WITH_ALI))&(USER.USE_EMWEIGHT>0)&(CLPAC>0))
    assert(USER.USE_EMWEIGHT!=2)  # not yet implemented in this subiteration branch
    assert(USER.NOSOLVE==False)   # does not make sense to make iterations if temperatures are not updated

    
    HOT_LIMIT = 30.0          # limit between "hot" and "cold" cells
    
    TOLD  = 0.0*TNEW
    PTABS = 0.0*OTABS         # for absorptions from cells not included in emission simulation
    if (USER.file_external_mask!=""): # use provides the mask for cells that are emitting packages
        # in the file 1=emitting cell, here mask is for ignored cells
        external_mask = fromfile(USER.file_external_mask, int32) # 1 = emitting cells
    else:
        external_mask = []
    mask  = nonzero(TOLD>1.0) # first iteration, no cells masked
    
    for iteration in range(USER.ITERATIONS):  # iteratio over simulation <-> T-update cycles
        prc = 100.0*len(mask[0])/float(len(TOLD))
        print("ITERATION %d/%d -- ignore emission %.2f %% of cells" % (iteration+1, USER.ITERATIONS, prc))
        commands[ID].finish()    
        kernel_zero(commands[ID],[GLOBAL,],[LOCAL,],np.int32(0),TABS_buf[ID],XAB_buf[ID],INT_buf[ID],INTX_buf[ID],INTY_buf[ID],INTZ_buf[ID])
        commands[ID].finish()

        # Note -- for the inactive (cold cells) this causes error because their contribution
        #         to OTABS is scaled down while their emission is not being simulated !!
        
        # This weighting applies to iterations = [2, ITERATIONS-2]
        k               = (iteration-2.0)/float(USER.ITERATIONS-3.0)
        OEMITTED[:,:]  *= k   # [CELLS, REMIT_NFREQ]
        OTABS[:]       *= k   # [CELLS],  not including CTABS
        
        if (iteration==0):
            CLPAC = Fix(USER.CLPAC, LOCAL)
            mask  = nonzero(TOLD>1.0)     # mask nonoe of the cells
            OEMITTED[:,:] = 0.0
            OTABS[:]      = 0.0
        elif (iteration==1):
            CLPAC = Fix(USER.CLPAC/2, LOCAL)     # afford to reduce the number of packages??
            if (len(external_mask)>0):
                mask = nonzero(external_mask>0)  # simulate user-excluded cells
            else:
                mask = nonzero(TOLD>=HOT_LIMIT)  # simulate cold cells = mask hot cells
            OEMITTED[:,:] = 0.0
            OTABS[:]      = 0.0
            # this iteration produces PTABS, T is not solved !!
        elif (iteration==2):             # entering loop where hot cells are simulated
            if (len(external_mask)>0):
                mask = nonzero(external_mask<1)  # simulate user-excluded cells
            else:
                mask = nonzero(TOLD<HOT_LIMIT)   # = ignore cold cells
            OEMITTED[:,:] = 0.0
            OTABS[:]      = 0.0
            # in this loop TABS = TABS + OTABS + PTABS, PTABS coming from iteration==1
        elif (iteration==(USER.ITERATIONS-1)):
            # last iteration is again full
            CLPAC = Fix(USER.CLPAC, LOCAL)  # again full USER.CLPAC cell packages
            # OEMITTED, OTABS are consistent but correspond to emission/absorption only of
            #  photons from hot cells
            OEMITTED[mask[0],:] = 0.0  # cold cells should have nothing in OEMITTED
            mask = nonzero(TOLD<0.0)   
            # this single iteration again includes cold cells - without reference = more noisy
            #   TABS = TABS + OTABS
            # must not add PTABS because cold cell emission not in OEMITTED either
            pass

        
        if ('OLDKERNEL' in USER.KEYS):
            # RAM version --- one work item per cell
            GLOBAL, BATCH  =  Fix(CELLS, LOCAL),    int(CLPAC/CELLS)
        else:
            # RAM2 version --- each work item loops over cells =>  GLOBAL<CELLS, BATCH==packets per cell
            GLOBAL, BATCH  =  GLOBAL_0,  max([1,int(CLPAC/CELLS)])
            print('=== CLPAC %d, GLOBAL %d, BATCH %d' % (CELLS*BATCH, GLOBAL, BATCH))
                
        assert(CLPAC>0)
        
        OIFREQ = -1
        for IFREQ in range(NFREQ):     # loop over single frequencies
            commands[ID].finish()
            # Among other things, set TABS==0.0
            kernel_zero(commands[ID],[GLOBAL,],[LOCAL,],np.int32(1),TABS_buf[ID],XAB_buf[ID],INT_buf[ID],INTX_buf[ID],INTY_buf[ID],INTZ_buf[ID])
            commands[ID].finish()
            # Parameters for the current frequency
            FREQ   =  FFREQ[IFREQ]

            if (len(USER.FSELECT)>0): 
                if (np.min(abs((FREQ-USER.FSELECT)/FREQ))>0.001): continue
            OIFREQ += 1  # OIFREQ<=IFREQ, in case we use LIB_ABS (although that makes no sense in subiterations?)
            
            if ((FREQ<USER.SIM_F[0])|(FREQ>USER.SIM_F[1])): continue            
            
            # print("    FREQ %3d/%3d   %12.4e --  ABS %.3e  SCA %.3e" % (IFREQ+1, NFREQ, FREQ, ABS, SCA))
            print("  FREQ %3d/%3d  %10.3e" % (IFREQ+1, NFREQ, FREQ))
            
            G = 0.0
            if (WITH_ABU>0): # ABS, SCA, G are vectors
                OPT[:,:] = 0.0
                if (USER.SINGLE_ABU):
                    # we have exactly two dust components with abundances ABU and 1.0-ABU
                    OPT[:,0] +=       ABU  * AFABS[0][IFREQ]
                    OPT[:,1] +=       ABU  * AFSCA[0][IFREQ]
                    OPT[:,0] +=  (1.0-ABU) * AFABS[1][IFREQ]
                    OPT[:,1] +=  (1.0-ABU) * AFSCA[1][IFREQ]
                else:
                    for idust in range(NDUST):
                        OPT[:,0]  +=  ABU[:,idust] * AFABS[idust][IFREQ]
                        OPT[:,1]  +=  ABU[:,idust] * AFSCA[idust][IFREQ]
                if (USER.OPT_IS_HALF):
                    cl.enqueue_copy(commands[ID], OPT_buf[ID], asarray(OPT, np.float16))
                else:
                    cl.enqueue_copy(commands[ID], OPT_buf[ID], OPT)
                if (WITH_MSF):
                    for idust in range(NDUST):
                        ABS[idust] = AFABS[idust][IFREQ]
                        SCA[idust] = AFSCA[idust][IFREQ]                    
            else:               # constant abundance, ABS, SCA, G are scalar
                ABS[0], SCA[0] = 0.0, 0.0
                for idust in range(NDUST):
                    ABS[0] += AFABS[idust][IFREQ]
                    SCA[0] += AFSCA[idust][IFREQ]
            cl.enqueue_copy(commands[ID], ABS_buf[ID], ABS)
            cl.enqueue_copy(commands[ID], SCA_buf[ID], SCA)
            
            ## BG, PS             =  0.0, 0.0
            ## if (len(IBG)==NFREQ):  BG = IBG[IFREQ] * WBG/FREQ  # background photons per package
            ## if (len(LPS)==NFREQ):  PS = LPS[IFREQ] * WPS/FREQ  # source photons per package
            # The weight for on-the-fly trapezoid integration
            FF = FREQ 
            if (IFREQ==0):              FF *= 0.5*(FFREQ[1]-FFREQ[0])
            else:
                if (IFREQ==(NFREQ-1)):  FF *= 0.5*(FFREQ[NFREQ-1]-FFREQ[NFREQ-2])
                else:                   FF *= 0.5*(FFREQ[IFREQ+1]-FFREQ[IFREQ-1])
            # Upload to device new DSC, CSC, possibly also EMIT
            t0 = time.time()
            commands[ID].finish()
            if (WITH_MSF==0):
                cl.enqueue_copy(commands[ID], DSC_buf[ID], FDSC[0,IFREQ,:])
                cl.enqueue_copy(commands[ID], CSC_buf[ID], FCSC[0,IFREQ,:])
            else:
                for idust in range(NDUST):
                    DSC[idust,:] = FDSC[idust, IFREQ, :]
                    CSC[idust,:] = FCSC[idust, IFREQ, :]
                cl.enqueue_copy(commands[ID], DSC_buf[ID], DSC)
                cl.enqueue_copy(commands[ID], CSC_buf[ID], CSC)
            commands[ID].finish()                
            Tpush += time.time()-t0
            ###
            if ((IFREQ<REMIT_I1)|(IFREQ>REMIT_I2)): # this frequency not in EMITTED !!
                continue  # cannot simulate emission that is not in EMITTED
            ###
            EMIT[:]  =  EMITTED[:, IFREQ-REMIT_I1] - OEMITTED[:, IFREQ-REMIT_I1]  
            OEMITTED[:, IFREQ-REMIT_I1] = 1.0*EMITTED[:, IFREQ-REMIT_I1]
            
            for level in range(LEVELS):
                coeff      = USER.GL*(PARSEC/FACTOR)(8.0**level)  #  cell volume in units of GL^3
                a, b       = OFF[level], OFF[level]+LCELLS[level]
                EMIT[a:b] *= coeff*DENS[a:b]
            EMIT[nonzero(DENS<1.0e-10)] = 0.0    # empty cells emit nothing
            
            cl.enqueue_copy(commands[ID], EMIT_buf[ID], EMIT)

            
            # Weighting using the total emission, masked cells ignored
            tmp         =  asarray(EMITTED[:, IFREQ-REMIT_I1].copy(), float64)
            tmp[mask]   =  0.0  # cells in mask are not emitting any packages
            tmp[:]      =  CLPAC*tmp/(sum(tmp)+1.0e-32)  # ~ number of packages
            EMWEI[:]    =  clip(tmp, USER.EMWEIGHT_LIM[0], USER.EMWEIGHT_LIM[1])

            # Note: this will be used with SimRAM_CL() that generates one ray per
            #       work item == not ideal when the fraction of photon-emitting 
            #       cells is small == potentially small occupancy. For this 
            #       application one could/should again write a separate kernel that 
            #       only gets a list of cells that will have simulated photon packages.
            
            p  =  100.0*len(nonzero(EMWEI>0.0)[0])/float(CELLS)
            print('ITERATION %2d, EMISSION FROM %5.2f %% OF CELLS' % (iteration, p))
            
            # any EMWEI<1.0 means it has probability EMWEI of being simulated batch=1
            #   which means the weight is 1/p = 1/EMWEI
            EMWEI[nonzero(rand(CELLS)>EMWEI)] = 0.0 # Russian roulette
            if (USER.EMWEIGHT_LIM[2]>0.0):
                EMWEI[nonzero(EMWEI<USER.EMWEIGHT_LIM[2])] = 0.0 # ignore completely
            commands[ID].finish()
            cl.enqueue_copy(commands[ID], EMWEI_buf[ID], EMWEI)
            commands[ID].finish()
            print('=== Update EMWEI -> CLPAC=%d' % sum(EMWEI))
            # this limited only the re-simulation of emission from some cells
            # ... EMIT will still be updated for all cells, even if they are not
            #     included in the simulation on this iteration
            # Also, this does not limit TABS, so the temperatures can/will still 
            #     change for all cells on all iterations
            
            if (USER.SEED>0): seed = fmod(USER.SEED+(DEVICES*IFREQ+ID)*SEED1, 1.0)
            else:             seed = rand()
            
            commands[ID].finish()
            t0 = time.time()                                 
            kernel_ram_cl(commands[ID], [GLOBAL,], [LOCAL,],
            # 0           1      2      3     4            5          
            np.int32(2),  CLPAC, BATCH, seed, ABS_buf[ID], SCA_buf[ID], 
            FF,
            # 11            12           13           14            15          
            LCELLS_buf[ID], OFF_buf[ID], PAR_buf[ID], DENS_buf[ID], EMIT_buf[ID],
            # 16          17           18           19           20             
            TABS_buf[ID], DSC_buf[ID], CSC_buf[ID], XAB_buf[ID], EMWEI_buf[ID],
            # 21         22            23            24      
            INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID])            
            commands[ID].finish()                
            Tkernel += time.time()-t0

            if (0):
                if ((time.time()-t0)>7200): # something wrong ??
                    print('EXIT !!!!!!!!!!!!!')
                    sys.exit()
            
            # ------------------------------------------------------------------------------
            # Intensity
            if (iteration==(USER.ITERATIONS-1)):
                if (USER.SAVE_INTENSITY>0):
                    assert(USER.WITH_REFERENCE==0) ; # we want TABS_buf to contain the total field
                if ((USER.SAVE_INTENSITY==1)|(not(USER.NOABSORBED))):     # scalar intensity in each cell
                    cl.enqueue_copy(commands[ID], TMP, INT_buf[ID])       # ~ total field
                    commands[ID].finish()                    
                    if (not(USER.NOABSORBED)):
                        FABSORBED[:, OIFREQ] += TMP
                    if (USER.SAVE_INTENSITY==1):
                        for level in range(LEVELS):
                            coeff =  KDEV * (PLANCK*FREQ/ABS) * (8.0**level)
                            a, b  =  OFF[level], OFF[level]+LCELLS[level]
                            INTENSITY[a:b, IFREQ] += coeff * TMP[a:b] / DENS[a:b]
                if (USER.SAVE_INTENSITY==2):
                    for icomp in range(4):
                        BUFS = [INT_buf[ID], INTX_buf[ID], INTY_buf[ID], INTZ_buf[ID]]
                        cl.enqueue_copy(commands[ID], TMP, BUFS[icomp])  # ~ total field
                        commands[ID].finish()                        
                        for level in range(LEVELS):
                            coeff =  KDEV * (PLANCK*FREQ/ABS) * (8.0**level)
                            a, b  =  OFF[level], OFF[level]+LCELLS[level]
                            INTENSITY[a:b, IFREQ, icomp]  += coeff * TMP[a:b]  / DENS[a:b]
            # ------------------------------------------------------------------------------
            
            
        # end of -- for IFREQ
        # TABS is the integrated absorbed energy for emission from the medium
        commands[ID].finish()
        cl.enqueue_copy(commands[ID], EMIT, TABS_buf[ID])
        commands[ID].finish()

        
        if (iteration==1):
            # we are doing cold cell simulation -> result is just PTABS
            # = absorptions caused by emission from cold cells
            PTABS[:] = 1.0*EMIT[:]
            continue
                  
        
        EMIT[:]  +=  OTABS     # TRUE =  delta TABS + old TABS + energy from constant sources
        OTABS[:]  =  1.0*EMIT  # updated for the next iteration
        EMIT[:]  +=  CTABS
        # EMIT re-used for absorptions --- at this point 
        # EMIT  ==  sum of TABS components ==  OLD + DELTA + CONSTANT  absorptions

        # during iterations 2, ..., USER.ITERATIONS-2 we are simulating only hot cells
        # so we need to add to TABS also cold cell contribution from PTABS
        # on ITERATIONS-2, OTABS and OEMITTED both still corresponds to hot cells only
        # ==> we do not add PTABS (cold dust emission not in OEMITTED either)
        if ((iteration>1)&(iteration<=(USER.ITERATIONS-2))):
            EMIT[:] += PTABS[:]
        
        
        # Solve temperatures on the host side -- total absorbed energy in array EMIT[]
        # scale  = 6.62607e-07/(USER.GL*PARSEC)     # 1.0e20f*PLANCK / (GL*PARSEC)
        scale  = (6.62607e-27*FACTOR)/(USER.GL*PARSEC)     # 1.0e20f*PLANCK / (GL*PARSEC)
        ## oplgkE = 1.0/log10(kE)
        ## ADHOC  = 1.0e-10         # *** SAME AS IN THE KERNEL !! ***
        beta   = 1.0
    
        if (1):
            mmm   = nonzero(~isfinite(EMIT))
            if (len(mmm[0]>0)):
                print('EMIT NOT FINITE', len(mmm[0]))
            EMIT  = clip(EMIT, 1.0e-25, 1e32)


            
            
        print('Calculate temperatures')
        t0 = time.time()
        if (not('MPT' in USER.KEYS)):
            for level in range(LEVELS):
                print('LEVEL %d, LCELLS %d, IND [%d, %d[' % (level, LCELLS[level], OFF[level], OFF[level]+LCELLS[level]))
                for i in range(LCELLS[level]):
                    ind     =  OFF[level]+i
                    if (DENS[ind]<1.0e-10):
                        continue # skip empty cells and parent cells
                    told    =  TNEW[ind]
                    Ein     = (scale/ADHOC)*EMIT[ind]*(8.0**level)/DENS[ind] # "EMIT" == absorbed energy
                    # if (1): print("%6d   DENS %10.3e-- Ein %10.3e" % (ind, DENS[ind], Ein))
                    #  Ein = beta*Eout  =>   Ein/beta = Eout
                    if (USER.WITH_ALI):
                        beta    =  TMP[ind]   
                    # print Emin, beta, oplgkE
                    iE      =  int(clip(floor(oplgkE * log10((Ein/beta)/Emin)), 0, NE-2))
                    wi      = (Emin*pow(kE,iE+1)-(Ein/beta)) / (Emin*pow(kE,(iE+1))-pow(kE,iE))
                    tnew    =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1]                
                    if (0):
                        if (USER.WITH_ALI): # optional iteration with updated beta values
                            tau     =  AFABS[0][-1]*DENS[ind]
                            for j in range(2): # iterate by making correction for beta=beta(T)
                                # re-estimate escape probability after the temperature update
                                # if T increases ->  beta decreases -> T increase further 
                                beta  *=  beta_interpoler(tnew, tau) / beta_interpoler(told, tau)
                                iE     =  clip(int(floor(oplgkE * log10((Ein/beta)/Emin))), 0, NE-2)
                                wi     = (Emin*pow(kE,iE+1)-(Ein/beta)) / (Emin*pow(kE,(iE+1))-pow(kE,iE)) ;
                                told   =  tnew
                                tnew   =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1] ;    
                    TNEW[ind]  = tnew
                    # print("Ein %10.3e   Tnew %10.3e" % (Ein, TNEW[ind]))
        else:
            # Calculate temperatures in parallel
            def MP_temp(a0, a, b, EIN, DENS, kk, MPTNEW, TMP=[]):  # TNEW = single level
                if (len(TMP)<1): # no ali
                    for i in range(a, b):
                        if (DENS[i]<=0.0): 
                            continue
                        Ein       = (EIN[i]/DENS[i])*kk
                        iE        =  int(clip(floor(oplgkE * log10(Ein/Emin)), 0, NE-2))
                        wi        = (Emin*pow(kE,iE+1)-Ein) / (Emin*pow(kE,(iE+1))-pow(kE,iE))
                        MPTNEW[i-a0] =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1]
                else:
                    for i in range(a, b):
                        if (DENS[i]<=0.0): continue
                        Ein       = (EIN[i]/DENS[i])*kk
                        beta      = TMP[i]
                        iE        =  int(clip(floor(oplgkE * log10((Ein/beta)/Emin)), 0, NE-2))
                        wi        = (Emin*pow(kE,iE+1)-(Ein/beta)) / (Emin*pow(kE,(iE+1))-pow(kE,iE))
                        MPTNEW[i-a0] =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1]
                return
            
            for level in range(LEVELS):
                a, b    = OFF[level], OFF[level]+LCELLS[level]  #  [a, b[
                kk      = (scale/ADHOC)*(8.0**level)
                manager = mp.Manager()
                PROC    = []
                n       = (LCELLS[level]+NCPUS)/NCPUS + 1  # cells per single thread
                MPTNEW  = mp.Array('f', int(LCELLS[level])) # T for cells of a single level
                for i in range(NCPUS):
                    if (USER.WITH_ALI):
                        p   = mp.Process(target=MP_temp, 
                        args=(a, a+i*n, min([b, a+(i+1)*n]), EMIT, DENS, kk, MPTNEW, TMP))
                        # args=(a, a+i*n, min([b, a+(i+1)*n]), EMIT, DENS, kk, TMP))
                    else:
                        p   = mp.Process(target=MP_temp, 
                        args=(a, a+i*n, min([b, a+(i+1)*n]), EMIT, DENS, kk, MPTNEW, []))
                        # args=(a, a+i*n, min([b, a+(i+1)*n]), EMIT, DENS, kk, []))
                    PROC.append(p)
                    p.start()
                for i in range(NCPUS):
                    PROC[i].join()                
                TNEW[a:b] = MPTNEW
            
                
        if (len(TNEW)<1e6):
            m       = nonzero(DENS>1e-7)
            a, b, c = np.percentile(TNEW[m], (0, 50.0, 100.0))
            print('TEMPERATURES %.3f %.3f %.3f' % (a, b, c))

            
        # Save temperatures -- file format is the same as for density (including links!)
        print('Save temperatures')
        if (len(USER.file_temperature)>1):
            fp = open(USER.file_temperature, "wb")
            asarray([NX, NY, NZ, LEVELS, CELLS], int32).tofile(fp)
            for level in range(LEVELS):
                a, b = OFF[level], OFF[level]+LCELLS[level]
                asarray([LCELLS[level],], int32).tofile(fp)
                TNEW[a:b].tofile(fp)
            fp.close()
            
            
        # Calculate emission
        print('Calculate emission')
        FTMP = zeros(REMIT_NFREQ, float64)
        if ('CLE' in USER.KEYS): # do emission calculation on device
            kernel_emission = program[0].Emission
            # FREQ, FABS,   DENS, T, EMIT
            kernel_emission.set_scalar_arg_dtypes([np.float32, np.float32, None, None, None])
            cl.enqueue_copy(commands[0], EMIT_buf[0], TNEW)
            # Note:  EMIT_buf is reused for temperature, TABS_buf for the resulting emission
            #        ... because EMIT_buf is read only!
            a, b = REMIT_I1, REMIT_I2+1
            for ifreq in range(a, b):
                kernel_emission(commands[0], [GLOBAL,], [LOCAL,], float(FFREQ[ifreq]), float(AFABS[0][ifreq]),
                #         *** T ***  *** emission ***
                DENS_buf[0], EMIT_buf[0],  TABS_buf[0])
                cl.enqueue_copy(commands[0], EMIT, TABS_buf[0])
                EMITTED[:, ifreq-REMIT_I1] = EMIT
            ####
        elif (not('MPE' in USER.KEYS)):  # single thread on host
            for icell in range(CELLS):  # loop over the GLOBAL large array
                if (DENS[icell]<1.0e-10):
                    FTMP[:] = 0.0
                else:
                    a, b  =  REMIT_I1, REMIT_I2+1
                    FTMP[0:REMIT_NFREQ] =  (FACTOR*4.0*np.pi/(PLANCK*FFREQ[a:b])) * \
                    AFABS[0][a:b]*Planck(FFREQ[a:b], TNEW[icell])/(USER.GL*PARSEC) #  1e20*photons/H -> FACTOR
                EMITTED[icell,:] = FTMP[0:REMIT_NFREQ]  # EMITTED[CELLS, REMIT_NFREQ]
        else:                            # multithreaded on host
            def MP_emit(ifreq, ithread, ncpu, TNEW, MPA):
                ii   =  int(REMIT_I1 + ifreq + ithread)  # index of the frequency
                if (ii>REMIT_I2):  # no such channel
                    return
                kk   =  (FACTOR*4.0*np.pi/(PLANCK*FFREQ[ii])) * AFABS[0][ii] / (USER.GL*PARSEC)
                freq =  FFREQ[ii]
                MPA[ithread][:] = kk*PlanckTest(freq, TNEW) #  1e20*photons/H -> A FACTOR photons/H
            ###
            manager = mp.Manager()
            TNEW[nonzero(TNEW<3.0)] = 10.0  # avoid warnings about parent cells
            MPA     = [ mp.Array('f', int(CELLS)), mp.Array('f', int(CELLS)), mp.Array('f', int(CELLS)), mp.Array('f', int(CELLS))]
            for ifreq in range(REMIT_I1, REMIT_I2+1, NCPUS):
                PROC      = []
                for i in range(NCPUS):
                    p   = mp.Process(target=MP_emit,args=(ifreq, i, NCPUS, TNEW, MPA))
                    PROC.append(p)
                    p.start()
                for i in range(NCPUS):
                    PROC[i].join()
                for i in range(NCPUS):
                    if ((ifreq+i)<=REMIT_I2):
                        EMITTED[:,ifreq+i] = MPA[i][:]
                
        del FTMP            
        Tsolve = time.time()-t0

        
        TOLD  = 1.0*TNEW
        
        
    # end of -- for iteration

    
    
    
    
    
    
    
    
    
    

if (USER.SAVE_INTENSITY in [1,2]):
    # run_RAT.py with OCTREE=True assumes IX etc. were divided by total I
    if (USER.SAVE_INTENSITY==2):
        # in the file IX, IY, IZ should already be normalised with the total intensity
        for k in [1,2,3]:
            INTENSITY[:,:,k] /= (INTENSITY[:,:,0]+1.0e-33)
        for ifreq in range(NFREQ):
            print("ifreq %3d   %10.3e   %10.3e %10.3e %10.4e" % (ifreq,  mean(INTENSITY[:,ifreq,0]),
            mean(INTENSITY[:,ifreq,1]),mean(INTENSITY[:,ifreq,2]),mean(INTENSITY[:,ifreq,3])))
    else:
        for ifreq in range(NFREQ):
            print("ifreq %3d   %10.3e" % (ifreq,  mean(INTENSITY[:,ifreq])))
            
    del INTENSITY  # INTENSITY to mmap file
    
    if (1):
        # add the missing header to the intensity file
        fp = open('ISRF.DAT', 'r+')
        if (USER.SAVE_INTENSITY==1):
            asarray([CELLS, NFREQ   ], int32).tofile(fp)
        elif (USER.SAVE_INTENSITY==2):
            asarray([CELLS, NFREQ, 4], int32).tofile(fp)
        fp.close()

        
        
if ((not(USER.NOSOLVE))&(USER.WITH_ALI)):
    asarray(OXAB, float32).tofile('OXAB.save')
    asarray(OXEM, float32).tofile('OXEM.save')
        
del XEM
del OXEM
del OXAB
del OEMITTED





# ===============================================================================================
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# ===============================================================================================




# We do not need ABSORBED after this point, free some memory before spectrum calculation.
if (not(USER.NOABSORBED)):
    print('*'*80)
    
    #  FABSORBED[:,:] *=  1.0e20 / (USER.GL*PARSEC)
    # for level in range(1,LEVELS): FABSORBED[(OFF[level]) :  (OFF[level]+LCELLS[level])]  *=  8.0**level
    # for icell in range(CELLS):    FABSORBED[icell,:] /= DENS[icell]
    # Faster alternative
    # FABSORBED[CELLS, NFREQ]
    print("Update absorbed (mmap)") 
    for level in range(LEVELS):
        a, b      =   OFF[level], OFF[level]+LCELLS[level]   # index limits for cells on current level
        if (level==0):
            coeff =                   FACTOR / (USER.GL*PARSEC)
        else:
            coeff =   (8.0**level) * (FACTOR / (USER.GL*PARSEC))
        FABSORBED[a:b]  *=  coeff                            # volume scaling
        # frequency runs faster... update one cell at a time, all frequencies 
        if (1):
            # This must be a slow loop... but everything processed in the storage order, in minimal amount of memory
            for icell in range(a, b):
                if (DENS[icell]>0.0):   
                    FABSORBED[icell,:] /= DENS[icell]
                else:
                    FABSORBED[icell,:] = -1.0  # tell A2E to skip this one !!
        else: # 2019-02-26
            # This is faster if FABSORBED is in memory
            # However, in a large model there may be 100e6 cells per level = tens of GB.
            # This forces one to read and write the array twice (sparse write on the second round)
            # -> possibly slower if one FABSORBED data for a single hierarchy level does not fit in the main memory
            FABSORBED[a:b,:] /= DENS[a:b].reshape(b-a,1)
            m                 = nonzero(DENS[a:b]<=0.0)
            FABSORBED[m[0],0] = -1.0   

            
    if (USER.SAVE_INTENSITY==3):
        # Save intensity based on the ABSORBED array
        #  I  =  n_phot * (h*nu)/(4*pi)  and   n_phot_absorbed ~ n_phot * kappa
        #  I  =  (h*nu)/(4*pi) *  n_phot_absorbed_per_cm3 / kappa_per_cm
        # 2019-03-23 -- to be consistent with SAVE_INTENSITY=1,2 options, 
        #               we now save I *** multiplied by 4*pi ***
        print("Save intensities")
        fp = open("ISRF.DAT", "wb") ;
        if (NDUST>1): 
            print("*** ERROR: SAVE_INTENSITY=3 does not work for multiple dusts (fix it!)")
            sys.exit()
        if (USER.LIB_ABS):
            print("*** ERROR: using LIB_ABS, cannot compute intensity for all frequencies!")
            sys.exit()
        asarray([CELLS, NFREQ], int32).tofile(fp)        
        for ifreq in range(NFREQ):            
            kappa  =  AFABS[0][ifreq] / (USER.GL*PARSEC)
            TMP    =  FABSORBED[:,ifreq]  * ((PLANCK*FFREQ[ifreq])/(4.0*pi)) / kappa
            TMP   *=  4.0*pi/FACTOR        # intensity saved in cgs units ... 4*pi * true intensity
            # print("  %3d  n=%10.3e   ABS: %.3e   I=%10.3e" % (ifreq, DENS[99], FABSORBED[99,ifreq], TMP[99]))
            TMP.tofile(fp)
        fp.close()
        
            
    ### delete = save and delete
    del FABSORBED   # writes the rest of updates to USER.file_absorbed
    print("FABSORBED saved and deleted")
    print('*'*80)
## del ABSORBED





"""
There only remains the writing of different emission-related maps,
with orthographic, perspective, or Healpix projection (but only one).
Polarisation maps are treated separately. Only one device will be used.

When maps are written, one is using either scalar ABS, SCA or,
in case of variable abundances, data in OPT array.
There is no need for ABU array or to have ABS and SCA as vectors!!
"""

print('Write maps')

t0            = time.time()
MAP_SLOW      = (not(USER.NOMAP)) & (USER.FAST_MAP<2)
MAP_FAST      = (not(USER.NOMAP)) & (USER.FAST_MAP>1) & (USER.FAST_MAP<999)
MAP_HIER      = (not(USER.NOMAP)) & (USER.FAST_MAP>=999)
NDIR          = len(USER.OBS_THETA)  # number of directions towards observers
MAP           = []

if (not(USER.NOMAP)):
    if (MAP_HIER):             # concatenate maps from different hierarchy levels
        if (USER.NPIX['y']<=0):   #  make a Healpix map
            MAP  = zeros(12*(USER.NPIX['x']**2)*LEVELS, float32)   # NPIX.x ==  NSIDE
            NDIR = 1
        else:
            MAP  = zeros(USER.NPIX['x']*USER.NPIX['y']*LEVELS, float32) # flat map
    else:                         # normal maps (no separation of level)
        if (USER.NPIX['y']<=0):   # Healpix map
            MAP  = zeros(12*USER.NPIX['x']*USER.NPIX['x'], float32) # NPIX.x == NSIDE
            NDIR = 1
        else:
            MAP  = zeros(USER.NPIX['x']*USER.NPIX['y'], float32)    # normal flat map

            
            
            
if ((MAP_SLOW)&(USER.NPIX['y']>0)): # make maps one frequency at a time
    # MAP has already been allocated for the correct size
    # USER.LIB_MAPS =>  emission and maps will both be only for the USER.FSELECT frequencies
    #                   then REMIT_I1==0, REMIT_I2==ONFREQ-1
    # print("MAP_SLOW  --- USER.NPIX.x = %d" % USER.NPIX['x'])
    MAP_buf      = cl.Buffer(context[0], mf.WRITE_ONLY, MAP.nbytes)
    SAVETAU_buf  = cl.Buffer(context[0], mf.WRITE_ONLY, MAP.nbytes)
    # source      = open(os.getenv("HOME")+ "/starformation/SOC/kernel_ASOC_map.c").read()
    source      = open(INSTALL_DIR+"/kernel_ASOC_map.c").read()
    program_map = cl.Program(context[0], source).build(ARGS+' -D NSIDE=%d' % (USER.NPIX['x']))
    kernel_map  = None
    try:
        if (USER.NPIX['y']<=0):  kernel_map = program.HealpixMapping
        else:                    kernel_map = program_map.Mapping
    except:
        print("Error: Failed to create mapping kernel!")
        sys.exit()
    if (USER.ROI_MAP>0): # write map using only emission within ROI
        kernel_map.set_scalar_arg_dtypes([
        # 0         1                     2     3   
        # MAP_DX    NPIX                  MAP   EMIT
        np.float32, clarray.cltypes.int2, None, None,
        # 4                     5                       6                     
        # DIR                   RA                      DE                    
        clarray.cltypes.float3, clarray.cltypes.float3, clarray.cltypes.float3,
        # 7       8     9     10    11            12 
        # LCELLS  OFF   PAR   DENS  ABS           SCA
        None,     None, None, None, np.float32,   np.float32,
        # 13                    14                      15     16       17
        # CENTRE                INTOBS                  OPT    SAVETAU  ROI
        clarray.cltypes.float3, clarray.cltypes.float3, None,  None,    None])
        ROI_LIM_buf = cl.Buffer(context[0], mf.READ_ONLY, 4*6)  # ROI[6], int32 limits
        cl.enqueue_copy(commands[0], ROI_LIM_buf, asarray(USER.ROI, int32))
    else:        # the same but without column density calculation
        kernel_map.set_scalar_arg_dtypes([        
        # 0          1                      2      3   
        # MAP_DX     NPIX                   MAP    EMIT
        np.float32,  clarray.cltypes.int2,  None,  None,        
        # 4                      5                        6                     
        # DIR                    RA                       DE                    
        clarray.cltypes.float3,  clarray.cltypes.float3,  clarray.cltypes.float3,        
        # 7        8      9      10     11           12     
        # LCELLS   OFF    PAR    DENS   ABS          SCA    
        None,      None,  None,  None,  np.float32,  np.float32,
        # 13                     14                       15     16    
        # CENTRE                 INTOBS                   OPT    SAVETAU
        clarray.cltypes.float3,  clarray.cltypes.float3,  None,  None  ])
    if (USER.NPIX['y']<=0):   # Healpix map
        GLOBAL = int((1+floor((12*USER.NPIX['x']**2)/LOCAL))*LOCAL)
    else:
        GLOBAL = int((1+floor((USER.NPIX['x']*USER.NPIX['y'])/LOCAL))*LOCAL)
    commands[0].finish()                
    fpmap = []                   # file handles for NDIR maps
    if (USER.NPIX['y']<=0):      # only one Healpix map
        fpmap.append( open("map.healpix", "wb") )
    else:                        # flat maps
        for idir in range(len(USER.OBS_THETA)):
            filename =  "map_dir_%02d.bin" % idir
            fpmap.append( open(filename, "wb") )
            asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fpmap[idir])        
    KK  = (1.0e23/FACTOR) * PLANCK / (4.0*np.pi) #  1e3 = 1e23/1e20 = remove 1e20 scaling and convert to Jy/sr
    KK *= USER.GL*PARSEC
    first_loop = True
    OIFREQ = -1
    print("*** MAPS FOR IFREQ=[%d,%d]" % (REMIT_I1, REMIT_I2))
    for IFREQ in range(REMIT_I1, REMIT_I2+1):  # loop over frequencies [REMIT_I1, REMIT_I2]
        # with LIB_MAPS, the loop is always over all NFREQ frequencies but only frequencies in FSELECT processed
        FREQ = FFREQ[IFREQ] 
        
        if (USER.LIB_MAPS):  # Simulation for the library method, FSELECT contains the reference frequencies
            if (np.min(abs((FREQ-USER.FSELECT)/FREQ))>0.001):
                iiii = argmin(abs((FREQ-USER.FSELECT)))
                # print("  FREQ %12.4e  ... closest FSELECT %12.4e\n" % (FREQ, USER.FSELECT[iiii]))
                continue                    
        # print("FREQ %.3e Hz = %8.2f um   ---   MAP_FREQ  [%.3e, %.3e] Hz  =  [%7.2f, %7.2f] um" % (FREQ, f2um(FREQ), USER.MAP_FREQ[0], USER.MAP_FREQ[1],  f2um(USER.MAP_FREQ[1]), f2um(USER.MAP_FREQ[0])))
            
        # OIFREQ is   [0,NFREQ[ or [REMIT_I1,REMIT_I2] and with LIB_MAPS it is [0, ONFREQ[
        #  => it is always directly the index to the EMITTED array
        OIFREQ += 1         
        if ((FREQ<USER.MAP_FREQ[0])|(FREQ>USER.MAP_FREQ[1])): continue # user may limit frequencies further
        
        if (USER.LIB_MAPS):
            print(" LIB_MAPS ---- map %d, frequency %d in the full frequency grid" % (OIFREQ, IFREQ))


        if (len(USER.SINGLE_MAP_FREQ)>0): # user picks individual map frequencies
            ii = argmin(abs(FREQ-USER.SINGLE_MAP_FREQ))
            if ((abs(FREQ-USER.SINGLE_MAP_FREQ[ii])/FREQ)>0.005):
                print("      UM %6.2f ---NOT--- IN SINGLE_MAP_FREQ, CLOSEST %.2f um" % (f2um(FREQ), f2um(USER.SINGLE_MAP_FREQ[ii])))
                continue
            print("   *** DOING SINGLE_MAP_FREQ[%d] = %.3e = %6.1f um" % (ii, FREQ, f2um(FREQ)))

            
        # optical parameters
        if (WITH_ABU>0): # ABS, SCA, G are vectors... precalculated into OPT
            OPT[:,:] = 0.0
            if (USER.SINGLE_ABU):
                # we have exactly two dust components with abundances ABU and 1.0-ABU
                OPT[:,0] +=       ABU  * AFABS[0][IFREQ]
                OPT[:,1] +=       ABU  * AFSCA[0][IFREQ]
                OPT[:,0] +=  (1.0-ABU) * AFABS[1][IFREQ]
                OPT[:,1] +=  (1.0-ABU) * AFSCA[1][IFREQ]
            else:
                for idust in range(NDUST):
                    OPT[:,0]  +=  ABU[:,idust] * AFABS[idust][IFREQ]
                    OPT[:,1]  +=  ABU[:,idust] * AFSCA[idust][IFREQ]
            if (USER.OPT_IS_HALF):
                cl.enqueue_copy(commands[0], OPT_buf[0], asarray(OPT, np.float16))
            else:
                cl.enqueue_copy(commands[0], OPT_buf[0], OPT)
        else:          # constant abundance, ABS, SCA, G are scalar
            ABS[0], SCA[0] = 0.0, 0.0
            for idust in range(NDUST):
                ABS[0] += AFABS[idust][IFREQ]
                SCA[0] += AFSCA[idust][IFREQ]
        # in map-making ABS and SCA are scalar kernel arguments
        if (USER.LIB_MAPS):
            EMIT[:] = KK * FREQ * EMITTED[:, OIFREQ]
        else:
            ii = IFREQ
            if (REMIT_NFREQ<NFREQ):   # EMITTED has only REMIT_NFREQ<=NFREQ frequencies
                # print("File has %d < %d frequencies" % (REMIT_NFREQ, NFREQ))
                ii = IFREQ-REMIT_I1   # frequency index to elements in EMITTED
            EMIT[:] = KK * FREQ * EMITTED[:, ii]
            
        # Use kernel to do the LOS integration:  EMIT -> MAP
        # DENSITY is already on device, EMIT becomes number of emitted photons * 4*pi/(h*f)
        cl.enqueue_copy(commands[0], EMIT_buf[0], EMIT)
        for idir in range(NDIR): #  loop over directions
            # print("IFREQ %d, IDIR %d" % (IFREQ, idir))
            if (USER.ROI_MAP):
                kernel_map(commands[0], [GLOBAL,], [LOCAL,],
                # 0          1          2        3            4           5         6       
                USER.MAP_DX, USER.NPIX, MAP_buf, EMIT_buf[0], ODIR[idir], RA[idir], DE[idir],
                # 7            8           9           10           11      12       13            
                LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABS[0], SCA[0],  USER.MAPCENTRE,
                # 14         15          16          17      
                USER.INTOBS, OPT_buf[0], SAVETAU_buf, ROI_LIM_buf)
            else:  # the same without ROI
                # print("  *** kernel_map ***") 
                kernel_map(commands[0], [GLOBAL,], [LOCAL,],
                USER.MAP_DX, USER.NPIX, MAP_buf, EMIT_buf[0], ODIR[idir], RA[idir], DE[idir],
                LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABS[0], SCA[0], USER.MAPCENTRE,
                USER.INTOBS, OPT_buf[0], SAVETAU_buf)
            cl.enqueue_copy(commands[0], MAP, MAP_buf)  # copy result to MAP
            # write the frequency maps to the files (whatever the size of MAP)
            asarray(MAP,float32).tofile(fpmap[idir])  # directly all the selected frequencies
            if ((first_loop)&(USER.savetau_freq<=0.0)):
                # Save column density only after the calculation of the first frequency
                # print("*** WRITING THE COLUMN DENSITY MAP f=%.3e Hz" % USER.savetau_freq)                
                fp       = open("%s.%d" %  (USER.file_savetau, idir), "wb")
                asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fp)
                cl.enqueue_copy(commands[0], MAP, SAVETAU_buf) # same number of pixels as MAP
                asarray(MAP, float32).tofile(fp)
                # print("*** AVERAGE COLUMN DENSITY %12.4e" % mean(ravel(MAP)))
                fp.close()
            if (USER.savetau_freq>0.0):  # save optical depth for frequency USER.savetau_freq
                if (abs((USER.savetau_freq-FREQ)/FREQ) < 0.01 ): # save computed tau value for this frequency
                    # print("*** WRITING THE OPTICAL DEPTH MAP f=%.3e Hz" % USER.savetau_freq)
                    fp       = open("%s.%d" %  (USER.file_savetau, idir), "wb")
                    asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fp)
                    cl.enqueue_copy(commands[0], MAP, SAVETAU_buf) # same number of pixels as MAP
                    asarray(MAP, float32).tofile(fp)
                    fp.close()            
            
        first_loop = False
        # end of -- for idir
    # end of -- for IFREQ
    for idir in range(NDIR):  fpmap[idir].close()
# end of -- if (USER.FAST_MAP<2) = slow mapping






if ((not(MAP_HIER))&(USER.NPIX['y']<0)): # Healpix map
    # Normal Healpix map
    # 2019-04-10: column density map is now always saved
    print("MAP_HIER=%d, NPIX[y]=%d" % (MAP_HIER, USER.NPIX['y']))
    # source      = open(os.getenv("HOME")+"/starformation/SOC/kernel_ASOC_map.c").read()
    source      = open(INSTALL_DIR+"/kernel_ASOC_map.c").read()
    program_map = cl.Program(context[0], source).build(ARGS+' -D NSIDE=%d' % (USER.NPIX['x']))
    kernel_map  = None
    print(USER.NPIX['y'], USER.ROI_MAP)
    if (USER.NPIX['y']<=0):
        print("program_map.HealpixMapping")
        kernel_map = program_map.HealpixMapping
    else:                  
        print("program_map.Mapping")
        kernel_map = program_map.Mapping
    print("USER.ROI_MAP=%d" % USER.ROI_MAP) 
    if (USER.ROI_MAP>0): # write map using only emission within ROI
        kernel_map.set_scalar_arg_dtypes([
        # 0       1                     2     3   
        np.float32, clarray.cltypes.int2, None, None,
        # 4                     5                       6                     
        clarray.cltypes.float3, clarray.cltypes.float3, clarray.cltypes.float3,
        # 7   8     9     10    11          12        
        None, None, None, None, np.float32, np.float32,
        # 13                    14                      15     16    17  
        clarray.cltypes.float3, clarray.cltypes.float3, None,  None, None])
        ##
        ROI_LIM_buf = cl.Buffer(context[0], mf.READ_ONLY, 4*6)  # ROI[6], int32 limits
        cl.enqueue_copy(commands[0], ROI_LIM_buf, asarray(USER.ROI, int32))
    else:
        kernel_map.set_scalar_arg_dtypes([
        # 0         1                     2     3   
        np.float32, clarray.cltypes.int2, None, None,
        # 4                     5                       6                     
        clarray.cltypes.float3, clarray.cltypes.float3, clarray.cltypes.float3,
        # 7   8     9     10    11          12        
        None, None, None, None, np.float32, np.float32,
        # 13                    14                      15    16   
        clarray.cltypes.float3, clarray.cltypes.float3, None, None ])
    
    if (USER.NPIX['y']<=0): GLOBAL = int((1+floor((12*USER.NPIX['x']*USER.NPIX['x'])/LOCAL))*LOCAL) # Healpix
    else:                   GLOBAL = int((1+floor((USER.NPIX['x']*USER.NPIX['y'])/LOCAL))*LOCAL)    # flat map
    # Open output files for NDIR files, each will contain LEVELS maps
    fpmap = []
    for idir in range(NDIR):
        fpmap.append( open("map_dir_%02d_H.bin" % idir, "wb") )
        asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fpmap[idir])            
        itmp = 0
        for IFREQ in range(REMIT_I1, REMIT_I2+1):
            FREQ = FFREQ[IFREQ]            
            if (USER.LIB_MAPS): # Simulation for the library method, FSELECT contains the reference frequencies
                if (np.min(abs((FREQ-USER.FSELECT)/FREQ))>0.001): continue
            if ((FREQ<USER.MAP_FREQ[0])|(FREQ>USER.MAP_FREQ[1])): 
                continue   # this will not happen with LIB_MAPS
            itmp += 1      # actual number of frequencies in the result files, also in case in LIB_MAPS
        asarray([itmp, LEVELS], int32).tofile(fpmap[idir])
    KK  = (1.0e23/FACTOR) * PLANCK / (4.0*np.pi)   #  1e3 = 1e23/1e20 = removing scaling, convert to Jy/sr
    KK *= USER.GL*PARSEC
    MAP_buf     = cl.Buffer(context[0], mf.WRITE_ONLY, MAP.nbytes) # MAP is already for LEVELS
    SAVETAU_buf = cl.Buffer(context[0], mf.WRITE_ONLY, MAP.nbytes)

    OIFREQ = -1
    for IFREQ in range(REMIT_I1, REMIT_I2+1):    #/ loop over individual frequencies
        FREQ = FFREQ[IFREQ]
        if (USER.LIB_MAPS): # Simulation for the library method, FSELECT contains the reference frequencies
            if (np.min(abs((FREQ-USER.FSELECT)/FREQ))>0.001): continue
        OIFREQ += 1   # the running index in case of LIB_MAPS

        if ((FREQ<USER.MAP_FREQ[0])|(FREQ>USER.MAP_FREQ[1])): continue # never with LIB_MAPS
        
        # optical parameters
        ABS[0], SCA[0]  = AFABS[0][IFREQ], AFSCA[0][IFREQ]
        if (WITH_ABU>0): # ABS, SCA, G are vectors
            OPT[:,:] = 0.0
            if (USER.SINGLE_ABU):
                # we have exactly two dust components with abundances ABU and 1.0-ABU
                OPT[:,0] +=       ABU  * AFABS[0][IFREQ]
                OPT[:,1] +=       ABU  * AFSCA[0][IFREQ]
                OPT[:,0] +=  (1.0-ABU) * AFABS[1][IFREQ]
                OPT[:,1] +=  (1.0-ABU) * AFSCA[1][IFREQ]
            else:
                for idust in range(NDUST):
                    OPT[:,0]  +=  ABU[:,idust] * AFABS[idust][IFREQ]
                    OPT[:,1]  +=  ABU[:,idust] * AFSCA[idust][IFREQ]
            if (USER.OPT_IS_HALF):
                cl.enqueue_copy(commands[ID], OPT_buf[0], asarray(OPT, np.float16))
            else:
                cl.enqueue_copy(commands[ID], OPT_buf[0], OPT)
        else:               # constant abundance, ABS, SCA, G are scalar
            ABS[0], SCA[0] = 0.0, 0.0
            for idust in range(NDUST):
                ABS[0] += AFABS[idust][IFREQ]
                SCA[0] += AFSCA[idust][IFREQ]
        
        # copy EMITTED for this wavelengths
        if (USER.LIB_MAPS):
            EMIT[:] = EMITTED[:, OIFREQ] * KK * FREQ   # EMITTED and maps will have the frequencies from FSELECT
        else:
            EMIT[:] = EMITTED[:, IFREQ-REMIT_I1] * KK * FREQ
        print("EMIT = %12.4e" % mean(EMIT))
        # use kernel to do the LOS integration:  EMIT -> MAP
        cl.enqueue_copy(commands[0], EMIT_buf[0], EMIT)
        for idir in range(NDIR):        #  loop over observer directions
            if (USER.ROI_MAP>0):
                kernel_map(commands[0], [GLOBAL,], [LOCAL,],
                USER.MAP_DX, USER.NPIX, MAP_buf, EMIT_buf[0], ODIR[idir], RA[idir], DE[idir],
                LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABS[0],  SCA[0],  USER.MAPCENTRE,
                USER.INTOBS, OPT_buf[0], SAVETAU_buf, ROI_LIM_buf)
                #### cl.enqueue_copy(commands[0], ROI_LIM_buf, USER.ROI)
            else:
                kernel_map(commands[0], [GLOBAL,], [LOCAL,],
                USER.MAP_DX, USER.NPIX, MAP_buf, EMIT_buf[0], ODIR[idir], RA[idir], DE[idir],
                LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABS[0],  SCA[0],  USER.MAPCENTRE,
                USER.INTOBS, OPT_buf[0], SAVETAU_buf)
            cl.enqueue_copy(commands[0], MAP, MAP_buf)
            # write the frequency maps to file
            print(" ---- MAP %12.4e" % mean(MAP)) ;
            asarray(MAP, float32).tofile(fpmap[idir])
            if ((IFREQ==REMIT_I1)&(USER.savetau_freq==0.0)):
                # save column density only after the calculation of the first frequency
                fp   = open("%s.%d" % (USER.file_savetau, idir), "wb")
                asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fp)
                cl.enqueue_copy(commands[0], MAP, SAVETAU_buf)
                asarray(MAP, float32).tofile(fp)
                fp.close()
            if (USER.savetau_freq>0.0):
                if (abs((USER.savetau_freq-FREQ)/FREQ)<0.001): # save optical depth
                    fp   = open("%s.%d" % (USER.file_savetau, idir), "wb")
                    asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fp)
                    cl.enqueue_copy(commands[0], MAP, SAVETAU_buf)
                    asarray(MAP, float32).tofile(fp)
                    fp.close()
        # for -- idir
    # for -- IFREQ
    for idir in range(NDIR): fpmap[idir].close()

    
    

if (MAP_HIER):
    # One frequency at a time, images of individual hierarchy levels, only one direction!
    if (USER.LIB_MAPS):
        print("*** ERROR: MAP_HIER not implemented together with LIB_MAPS !!!"), sys.exit()
    print("MAP_HIER=%d, NPIX[y]=%d" % (MAP_HIER, USER.NPIX['y']))
    # source      = open(os.getenv("HOME")+"/starformation/SOC/kernel_ASOC_map_H.c").read()
    source      = open(INSTALL_DIR+"/kernel_ASOC_map_H.c").read()
    program_map = cl.Program(context[0], source).build(ARGS+' -D NSIDE=%d' % (USER.NPIX['x']))
    kernel_map  = None
    if (USER.NPIX['y']<=0):  kernel_map = program_map.HealpixMapping
    else:                    kernel_map = program_map.Mapping
    if (USER.savetau_freq>=0.0):
        kernel_map.set_scalar_arg_dtypes([
        # 0       1                     2     3   
        np.float, clarray.cltypes.int2, None, None,
        # 4                     5                       6                     
        clarray.cltypes.float3, clarray.cltypes.float3, clarray.cltypes.float3,
        # 7   8     9     10    11          12        
        None, None, None, None, np.float32, np.float32,
        # 13                    14                      15    16   
        clarray.cltypes.float3, clarray.cltypes.float3, None, None])                        
    else:    # the same without column density map
        kernel_map.set_scalar_arg_dtypes([
        np.float32, clarray.cltypes.int2, None, None,
        clarray.cltypes.float3, clarray.cltypes.float3, clarray.cltypes.float3,
        None, None, None, None, np.float32, np.float32,
        clarray.cltypes.float3, clarray.cltypes.float3, None])
        
    if (USER.NPIX['y']<=0): GLOBAL = int((1+floor((12*USER.NPIX['x']*USER.NPIX['x'])/LOCAL))*LOCAL) # Healpix
    else:                   GLOBAL = int((1+floor((USER.NPIX['x']*USER.NPIX['y'])/LOCAL))*LOCAL)    # flat map
    # Open output files for NDIR files, each will contain LEVELS maps
    fpmap = []
    for idir in range(NDIR):
        fpmap.append( open("map_dir_%02d_H.bin" % idir, "wb") )
        asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fpmap[idir])            
        itmp = 0
        for IFREQ in range(REMIT_I1, REMIT_I2+1):
            FREQ = FFREQ[IFREQ]
            if ((FREQ<USER.MAP_FREQ[0])|(FREQ>USER.MAP_FREQ[1])): 
                continue
            itmp += 1   # actual number of frequencies in the result files
        asarray([itmp, LEVELS], int32).tofile(fpmap[idir])
    KK  = (1.0e23/FACTOR) * PLANCK / (4.0*np.pi)   #  1e3 = 1e23/1e20 = removing scaling, convert to Jy/sr
    KK *= USER.GL*PARSEC
    MAP_buf     = cl.Buffer(context[0], mf.WRITE_ONLY, MAP.nbytes) # MAP is already for LEVELS
    SAVETAU_buf = cl.Buffer(context[0], mf.WRITE_ONLY, MAP.nbytes)
    for IFREQ in range(REMIT_I1, REMIT_I2+1):    #/ loop over individual frequencies
        FREQ = FFREQ[IFREQ]
        if ((FREQ<USER.MAP_FREQ[0])|(FREQ>USER.MAP_FREQ[1])): continue 

        # optical parameters
        ABS[0], SCA[0] = AFABS[0][IFREQ], AFSCA[0][IFREQ]        
        # optical parameters
        if (WITH_ABU>0): # ABS, SCA, G are vectors
            OPT[:,:] = 0.0
            if (USER.SINGLE_ABU):
                # we have exactly two dust components with abundances ABU and 1.0-ABU
                OPT[:,0] +=       ABU  * AFABS[0][IFREQ]
                OPT[:,1] +=       ABU  * AFSCA[0][IFREQ]
                OPT[:,0] +=  (1.0-ABU) * AFABS[1][IFREQ]
                OPT[:,1] +=  (1.0-ABU) * AFSCA[1][IFREQ]
            else:
                for idust in range(NDUST):
                    OPT[:,0]  +=  ABU[:,idust] * AFABS[idust][IFREQ]
                    OPT[:,1]  +=  ABU[:,idust] * AFSCA[idust][IFREQ]
            if (USER.OPT_IS_HALF):
                cl.enqueue_copy(commands[0], OPT_buf[0], asarray(OPT, np.float16))
            else:
                cl.enqueue_copy(commands[0], OPT_buf[0], OPT)
        else:               # constant abundance, ABS, SCA, G are scalar
            ABS[0], SCA[0] = 0.0, 0.0
            for idust in range(NDUST):
                ABS[0] += AFABS[idust][IFREQ]
                SCA[0] += AFSCA[idust][IFREQ]

        
        # copy EMITTED for this wavelengths
        EMIT[:] = EMITTED[:, IFREQ-REMIT_I1] * KK * FREQ
        # use kernel to do the LOS integration:  EMIT -> MAP
        cl.enqueue_copy(commands[0], EMIT_buf[0], EMIT)
        for idir in range(NDIR):        #  loop over observer directions
            if (USER.savetau_freq>=0.0):
                kernel_map(commands[0], [GLOBAL,], [LOCAL,],
                USER.MAP_DX, USER.NPIX, MAP_buf, EMIT_buf[0], ODIR[idir], RA[idir], DE[idir],
                LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABS[0],  SCA[0],  USER.MAPCENTRE,
                USER.INTOBS, OPT_buf[0], SAVETAU_buf)                    
            else:
                kernel_map(commands[0], [GLOBAL,], [LOCAL,],
                USER.MAP_DX, USER.NPIX, MAP_buf, EMIT_buf[0], ODIR[idir], RA[idir], DE[idir],
                LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABS[0],  SCA[0],  USER.MAPCENTRE,
                USER.INTOBS, OPT_buf[0])
            cl.enqueue_copy(commands[0], MAP, MAP_buf)
            # write the frequency maps to file
            asarray(MAP, float32).tofile(fpmap[idir])
            if ((USER.savetau_freq==0.0)&(IFREQ==REMIT_I1)):
                # save column density only after the calculation of the first frequency
                fp   = open("%s.%d" % (USER.file_savetau, idir), "wb")
                asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fp)
                cl.enqueue_copy(commands[0], MAP, SAVETAU_buf)
                asarray(MAP, float32).tofile(fp)
                fp.close()
            if (USER.savetau_freq>0.0): # save optical depth
                if (abs((USER.savetau_freq-FREQ)/FREQ)<0.001):
                    fp   = open("%s.%d" % (USER.file_savetau, idir), "wb")
                    asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fp)
                    cl.enqueue_copy(commands[0], MAP, SAVETAU_buf)
                    asarray(MAP, float32).tofile(fp)
                    fp.close()
        # for -- idir
    # for -- IFREQ
    for idir in range(NDIR): fpmap[idir].close()
# end of -- MAP_HIER

    
    
if (MAP_FAST):
    if (1):
        print("*** kernel_ASOC_map_X.c (fast mapping) does not yet exist for ASOC.py -- use slow mapping instead")        
    if (WITH_ABU>0):
        print("*** Error:  MAP_FAST currently possible only when dust abundances are constant!"), sys.exit()
    if (USER.LIB_MAPS):
        print("*** Error: MAP_FAST not implemented together with LIB_MAPS"), sys.exit()
    # Write maps using a faster alternative = several frequencies at one kernel call.
    NF         =  USER.FAST_MAP     # this many frequencies per kernel call
    # ... check the actual number of frequencies, the intersection of MAP_FREQ and REMIT_F limits
    # EMITTED always has only frequency indices REMIT_I1... REMIT_I2
    m          =  nonzero((FFREQ>USER.MAP_FREQ[0])&(FFREQ>USER.REMIT_F[0])&(FFREQ<USER.MAP_FREQ[1])&(FFREQ<USER.REMIT_F[1]))
    NF         =  min([NF, len(m[0])])   # actual number of frequencies
    I1, I2     =  m[0][0], m[0][-1]      # first and last frequency index
    ##print NF, I1, I2
    ##print USER.MAP_FREQ, USER.REMIT_F
    # filename    = os.getenv("HOME")+"/starformation/SOC/kernel_ASOC_map_X.c"
    filename    = INSTALL_DIR+"/kernel_ASOC_map_X.c"
    source      = open(filename).read()
    program_map = cl.Program(context[0], source).build(ARGS+' -D NSIDE=%d -D NF=%d' % (USER.NPIX['x'], NF))
    kernel_map  = None
    if (USER.NPIX['y']<=0):   # healpix map
        npix   = 12*USER.NPIX['x']*USRT.NPIX['x']
        GLOBAL = int((1+floor(npix/LOCAL))*LOCAL)
    else:
        npix   = USER.NPIX['x'] * USER.NPIX['y']
        GLOBAL = int((1+floor(npix/LOCAL))*LOCAL)
    # Note -- MAP has already been allocated MAP[npix]
    EMITX      = zeros(CELLS*NF, float32)
    MAPX       = zeros(npix*NF, float32)
    ABSX       = zeros(NF, float32)
    SCAX       = zeros(NF, float32)        
    EMITX_buf  = cl.Buffer(context[0], mf.READ_ONLY,  EMITX.nbytes) # [CELLS, ifreq]
    ABSX_buf   = cl.Buffer(context[0], mf.READ_ONLY,  ABSX.nbytes)  # [NF]
    SCAX_buf   = cl.Buffer(context[0], mf.READ_ONLY,  SCAX.nbytes)  # [NF]
    MAPX_buf   = cl.Buffer(context[0], mf.WRITE_ONLY, MAPX.nbytes)  # [npix, NF]
    SAVETAU_buf = cl.Buffer(context[0], mf.WRITE_ONLY, MAP.nbytes)                
    if (USER.NPIX['y']<=0):  kernel_map = program_map.HealpixMappingX
    else:                    kernel_map = program_map.MappingX
    # Note -- combination column density + healpix map is not implemented !!
    #         WITH_ABU = variable-abundance case not implemented for MappingX() kernel
    if (USER.savetau_freq>=0.0):
        kernel_map.set_scalar_arg_dtypes([
        np.float32, clarray.cltypes.int2, None, None,
        clarray.cltypes.float3, clarray.cltypes.float3, clarray.cltypes.float3,
        None, None, None, None, None,  None,
        clarray.cltypes.float3, clarray.cltypes.float3, None])
    else:
        kernel_map.set_scalar_arg_dtypes([
        # 0         1                     2     3    
        # DX        NPIX                  MAPX  EMITX
        np.float32, clarray.cltypes.int2, None, None,
        # 4                     5                       6                    
        # DIR                   RA                      DE                   
        clarray.cltypes.float3, clarray.cltypes.float3, clarray.cltypes.float3, 
        # 7        8     9     10    11     12   
        # LCELLS   OFF   PAR   DENS  ABS    SCA  
        None,      None, None, None, None,  None,
        # 13                    14                    
        # CENTRE                INTOBS                
        clarray.cltypes.float3, clarray.cltypes.float3])
    # Open output files for NDIR maps
    fpmap = []
    if (USER.NPIX['y']<=0):      #  there is only one Healpix map
        fpmap.append( open("map.healpix", "wb") )
        NDIR = 1 
    else:
        for idir in range(NDIR):
            fpmap.append( open("map_dir_%02d.bin" % idir, "wb") )
            asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fpmap[idir])
    KK  = (1.0e23/FACTOR) * PLANCK / (4.0*np.pi)   # 1e3 = 1e23/1e20 = removing scaling, convert to Jy/sr
    KK *= USER.GL*PARSEC
    for IFREQ in range(I1, I2, NF):   # loop over batches of NF frequencies
        sys.stdout.write(" %d" % IFREQ)
        freq        = FFREQ[IFREQ]
        EMITX.shape = (CELLS, NF)
        # Mappingx ... only when abundances are constant !!
        for ioff in range(NF):        # update parameters for the current set of NF frequencies
            if ((IFREQ+ioff)<=I2):
                freq           =  FFREQ[IFREQ+ioff] 
                ABSX[ioff]     =  AFABS[0][IFREQ+ioff] 
                SCAX[ioff]     =  AFSCA[0][IFREQ+ioff]
                EMITX[:,ioff]  =  EMITTED[:, IFREQ+ioff-REMIT_I1] * KK * freq
        EMITX = ravel(EMITX)            
        cl.enqueue_copy(commands[0], EMITX_buf, EMITX)
        cl.enqueue_copy(commands[0], ABSX_buf,  ABSX)
        cl.enqueue_copy(commands[0], SCAX_buf,  SCAX)            
        for idir in range(NDIR):    # loop over directions
            if (USER.savetau_freq>=0.0):
                kernel_map(commands[0], [GLOBAL,], [LOCAL,],
                USER.MAP_DX, USER.NPIX, MAPX_buf, EMITX_buf, ODIR[idir], RA[idir], DE[idir],
                LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABSX_buf,  SCAX_buf,
                USER.MAPCENTRE, USER.INTOBS, SAVETAU_buf)
            else:
                kernel_map(commands[0], [GLOBAL,], [LOCAL,],
                USER.MAP_DX, USER.NPIX, MAPX_buf, EMITX_buf, ODIR[idir], RA[idir], DE[idir],
                LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABSX_buf,  SCAX_buf,
                USER.MAPCENTRE, USER.INTOBS)
            cl.enqueue_copy(commands[0], MAPX, MAPX_buf) # maps for next NF frequencies
            MAPX.shape = (npix, NF)          # reshape to 2D
            for ifreq in range(NF):          # save maps to file
                if ((IFREQ+ifreq)<=I2):
                    MAP[:] = MAPX[:, ifreq]  # could be flat image or healpix map
                    asarray(MAP, float32).tofile(fpmap[idir])
            MAPX = ravel(MAPX)               # return to 1D vector
            if ((USER.savetau_freq==0.0)&(IFREQ==I1)):
                fp   = open("%s.%d" % (USER.file_savetau, idir), "wb")
                asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fp)
                cl.enqueue_copy(commands[0], MAP, SAVETAU_buf)
                asarray(MAP, float32).tofile(fp)
                fp.close()
            if (USER.savetau_freq>0.0):  # save optical depth instead
                if (abs((USER.savetau_freq-FREQ)/FREQ)<0.001):
                    fp   = open("%s.%d" % (USER.file_savetau, idir), "wb")
                    asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fp)
                    cl.enqueue_copy(commands[0], MAP, SAVETAU_buf)
                    asarray(MAP, float32).tofile(fp)
                    fp.close()

    # end of  -- for IFREQ
    sys.stdout.write("\n")
    sys.stdout.flush()
    for idir in range(NDIR):  fpmap[idir].close()
    del EMITX
    del MAPX
    del ABSX
    del SCAX
# end of -- MAP_FAST
del MAP
Tmap = time.time()-t0  # total time spent on calculating maps





if ((USER.NO_PS>0)&(USER.pssavetau_freq>0.0)&(USER.NPIX['y']>0)):
    # Calculate column density or tau from each point source to the observer
    #                   then REMIT_I1==0, REMIT_I2==ONFREQ-1
    print("PS_SAVETAU  --- USER.NPIX.x = %d" % USER.NPIX['x'])

    PSCOLDEN_buf   = cl.Buffer(context[0], mf.WRITE_ONLY, 4*USER.NO_PS) # colden towards each pointsource
    PSTAU_buf      = cl.Buffer(context[0], mf.WRITE_ONLY, 4*USER.NO_PS) # tau towards each pointsource
    kernel_pstau   = program_map.PSTau

    kernel_pstau.set_scalar_arg_dtypes([        
    # 0         1        2                        3                        4                      
    # nops      PSPOS    DIR                      RA                       DE                     
    np.int32,   None,    clarray.cltypes.float3,  clarray.cltypes.float3,  clarray.cltypes.float3,        
    # 5        6      7      8      9            10     
    # LCELLS   OFF    PAR    DENS   ABS          SCA    
    None,      None,  None,  None,  np.float32,  np.float32,
    # 11     12         13   
    # OPT    pscolden   pstau
    None,    None,      None  ])
        
    GLOBAL =  Fix(USER.NO_PS, 32)
    IFREQ  =  argmin(abs(FFREQ-USER.pssavetau_freq))
    if (abs(FFREQ[IFREQ]-USER.pssavetau_freq)>0.001*FFREQ[IFREQ]):
        print("*** Requested frequency for PSSAVETAU is not in the frequency grid")
        print("***      =>   f = %.3e  ==  %.3f um" % (USER.pssavetau_freq, um2f(USER.pssavetau_freq)))
    FREQ = FFREQ[IFREQ]
    # optical parameters
    if (WITH_ABU>0): # ABS, SCA, G are vectors... precalculated into OPT
        OPT[:,:] = 0.0
        if (USER.SINGLE_ABU):
            # we have exactly two dust components with abundances ABU and 1.0-ABU
            OPT[:,0] +=       ABU  * AFABS[0][IFREQ]
            OPT[:,1] +=       ABU  * AFSCA[0][IFREQ]
            OPT[:,0] +=  (1.0-ABU) * AFABS[1][IFREQ]
            OPT[:,1] +=  (1.0-ABU) * AFSCA[1][IFREQ]
        else:
            for idust in range(NDUST):
                OPT[:,0]  +=  ABU[:,idust] * AFABS[idust][IFREQ]
                OPT[:,1]  +=  ABU[:,idust] * AFSCA[idust][IFREQ]
        if (USER.OPT_IS_HALF):
            cl.enqueue_copy(commands[0], OPT_buf[0], asarray(OPT, np.float16))
        else:
            cl.enqueue_copy(commands[0], OPT_buf[0], OPT)
    else:          # constant abundance, ABS, SCA, G are scalar
        ABS[0], SCA[0] = 0.0, 0.0
        for idust in range(NDUST):
            ABS[0] += AFABS[idust][IFREQ]
            SCA[0] += AFSCA[idust][IFREQ]
    # in map-making ABS and SCA are scalar kernel arguments
    # Use kernel to do the LOS integration:  column density and optical depth
    # DENSITY is already on device
    pscolden = zeros(USER.NO_PS, float32)
    pstau    = zeros(USER.NO_PS, float32)
    nops     = USER.NO_PS
    for idir in range(NDIR): #  loop over directions   0     1           
        kernel_pstau(commands[0], [GLOBAL,], [LOCAL,], nops, PSPOS_buf[0],
        # 2         3         4       
        ODIR[idir], RA[idir], DE[idir],
        # 5            6           7           8            9       10     
        LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABS[0], SCA[0],
        # 11        12             13       
        OPT_buf[0], PSCOLDEN_buf,  PSTAU_buf)
        print("*** Writing the ps column density")
        cl.enqueue_copy(commands[0], pscolden, PSCOLDEN_buf)
        cl.enqueue_copy(commands[0], pstau,    PSTAU_buf)
        fp       = open("%s_%d.dat" % (USER.file_pssavetau, idir), "w")
        for i in range(nops):
            fp.write('%6d  %12.4e  %12.4e\n' % (i, pscolden[i], pstau[i]))
        fp.close()
        
    
    



print("USER.POLMAP %d, USER.NOMAP %d, USER.NPIX[y] %d" % (USER.POLMAP, USER.NOMAP, USER.NPIX['y']))


t0 = time.time()
if ((USER.POLMAP>0)&(USER.NPIX['y']>0)):  # NORMAL POLARISATION MAPS
    """
    Keep polarisation map completelety separate from the normal map
    can have NOMAP (no regular maps) but still write POLMAP (I, Q, U or other polarisation stats)
    POLSTAT==0   -->  I,Q,U,N 
    POLSTAT==1   -->  statistics  rT, rI, jT, jI  
                      r=rho weighted, j=emission weighted, T=tangling, I=inclination
    POLSTAT==2   -->  I,Q,U,N but non-orthographic, with cube replication and arbitrary LOS length
    """
    print("=== WRITE POLARISATION MAPS ===") 
    if (USER.LIB_MAPS):
        print("*** ERROR: POLMAP not implemented together with LIB_MAPS"), sys.exit()
    TNEW        = zeros(CELLS, float32)        
    # source      = open(os.getenv("HOME")+ "/starformation/SOC/kernel_ASOC_map.c").read()
    source      = open(INSTALL_DIR+ "/kernel_ASOC_map.c").read()
    program_map = cl.Program(context[0], source).build(ARGS+' -D NSIDE=%d' % (USER.NPIX['x']))
    kernel_map  = program_map.PolMapping
    # Push the magnetic fields. B files have the same format as density
    #     NX, NY, NZ, LEVELS, CELLS
    #    {  LCELLS, { values } }
    BB = []
    for ii in range(3):  
        print(USER.BFILES[ii])
        BB.append( read_otfile(USER.BFILES[ii]) )
    if (len(USER.file_polred)>0):
        # Polarisation reduction factor R is encoded to the length of the B vectors
        R = None  # the polarisation reduction factor
        if (USER.file_polred=='adhoc'):
            # As an example, we use ad hoc R calculated based on local dust temperature
            R   = read_otfile(USER.file_temperature) # returns a single vector
            # Encode polarisation reduction factor to the norm of B
            #   x =  (T/11.0-1)/1.0,  R =  exp(x) / (exp(x) + exp(-x))
            R   =  (R-13.3)/2.0 + 1.0e-4
            R   =   exp(R) / ( exp(R)+exp(-R) )
        elif (USER.file_polred.find('rhofun')>=0):
            # Use semilogx(x, 0.5*(1.0+tanh( (log10(1e4)-log10(x))/0.5 )))
            # encoding in filename rhofun_1e4_0.5 -- step and logarithmic width of the rise
            s       =   USER.file_polred.split('_')
            th, sw  =   float(s[1]), float(s[2]) # threshold and logarithmic step width
            R       =   read_otfile(USER.file_cloud)
            if (USER.KDENSITY!=1.0): R *= USER.KDENSITY
            
            aa      =   R[0:-1:10].copy()
            R       =   clip(R, 0.1, 1e10)
            R       =   0.5*(1.0+tanh((log10(th)-log10(R))/sw))
            if (0):
                clf()
                semilogx(aa, R[0:-1:10], 'x')
                show()
                sys.exit()
            
            
        else:     # argument is a file to file with R values
            if (0):
                R   =  read_otfile(USER.file_polred)
            else:  # run_RAT.py writes R to a plain file == [ cells, { R } ]
                R   = fromfile(USER.file_polred, float32)[1:]
        # encode 0<R<1 to the length of the polarisation vector
        R      =  R  /  sqrt(BB[0]**2 + BB[1]**2 + BB[2]**2)
        BB[0] *=  R
        BB[1] *=  R 
        BB[2] *=  R
        del R
    # push B  to device
    Bx_buf = cl.Buffer(context[0], mf.READ_ONLY, BB[0].nbytes)
    By_buf = cl.Buffer(context[0], mf.READ_ONLY, BB[1].nbytes)
    Bz_buf = cl.Buffer(context[0], mf.READ_ONLY, BB[2].nbytes)
    cl.enqueue_copy(commands[0], Bx_buf, BB[0])
    cl.enqueue_copy(commands[0], By_buf, BB[1])
    cl.enqueue_copy(commands[0], Bz_buf, BB[2])
    print("    --------- Bx  %7.3f %7.3f %7.3f %7.3f %7.3f" % (BB[0][0], BB[0][1], BB[0][2], BB[0][3], BB[0][4]))
    print("    --------- By  %7.3f %7.3f %7.3f %7.3f %7.3f" % (BB[1][0], BB[1][1], BB[1][2], BB[1][3], BB[1][4]))
    print("    --------- Bz  %7.3f %7.3f %7.3f %7.3f %7.3f" % (BB[2][0], BB[2][1], BB[2][2], BB[2][3], BB[2][4]))
    commands[0].finish()
    ###
    del BB[2]
    del BB[1]
    del BB[0]
    ###
    MAP_buf, kernel_map, fpmap = None, None, []
    MAP  = None

    MAP        = zeros(4*USER.NPIX['x']*USER.NPIX['y'], float32)  # [ I, Q, U, N ] maps
    MAP_buf    = cl.Buffer(context[0], mf.WRITE_ONLY,  4*4*USER.NPIX['x']*USER.NPIX['y'])
    kernel_map = program_map.PolMapping
    GLOBAL     = int((1+floor((USER.NPIX['x']*USER.NPIX['y'])/LOCAL))*LOCAL)
    for idir in range(NDIR):
        fpmap.append( open("polmap_dir_%02d.bin" % idir, "wb") )
        asarray([USER.NPIX['x'], USER.NPIX['y']], int32).tofile(fpmap[idir])
        
    kernel_map.set_scalar_arg_dtypes(
    [ np.float32, clarray.cltypes.int2, None, None,
    clarray.cltypes.float3, clarray.cltypes.float3, clarray.cltypes.float3,
    None, None, None, None,
    np.float32,  np.float32, clarray.cltypes.float3, clarray.cltypes.float3,
    None, None, None, None ])
            
    KK  = (1.0e23/FACTOR) * PLANCK / (4.0*np.pi)  #  1e3 = 1e23/1e20 = removing scaling, convert to Jy/sr
    KK *= USER.GL*PARSEC    
    for IFREQ in range(REMIT_I1, REMIT_I2+1):    #  loop over frequencies
        FREQ = FFREQ[IFREQ]
        if ((FREQ<USER.MAP_FREQ[0])|(FREQ>USER.MAP_FREQ[1])): 
            continue  # we are writing only frequencies limited by USER.MAP_FREQ
        print("POLMAP, IFREQ =  %d" % IFREQ)
        ABS[0], SCA[0]  = AFABS[0][IFREQ], AFSCA[0][IFREQ]        
        if (WITH_ABU>0): # ABS, SCA, G are vectors
            OPT[:,:] = 0.0
            if (USER.SINGLE_ABU):
                # we have exactly two dust components with abundances ABU and 1.0-ABU
                OPT[:,0] +=       ABU  * AFABS[0][IFREQ]
                OPT[:,1] +=       ABU  * AFSCA[0][IFREQ]
                OPT[:,0] +=  (1.0-ABU) * AFABS[1][IFREQ]
                OPT[:,1] +=  (1.0-ABU) * AFSCA[1][IFREQ]
            else:
                for idust in range(NDUST):
                    OPT[:,0] +=  ABU[:,idust] * AFABS[idust][IFREQ]
                    OPT[:,1] +=  ABU[:,idust] * AFSCA[idust][IFREQ]
            if (USER.OPT_IS_HALF):
                cl.enqueue_copy(commands[0], OPT_buf[0], asarray(OPT, np.float16))
            else:
                cl.enqueue_copy(commands[0], OPT_buf[0], OPT)
        else:               # constant abundance ->  ABS, SCA, G are scalars
            ABS[0], SCA[0] = 0.0, 0.0
            for idust in range(NDUST):
                ABS[0] += AFABS[idust][IFREQ]
                SCA[0] += AFSCA[idust][IFREQ]
                        
        # use kernel to do the LOS integration:  EMIT -> MAP
        EMIT[:] = KK * FREQ * EMITTED[:, IFREQ-REMIT_I1]
        cl.enqueue_copy(commands[0], EMIT_buf[0], EMIT)
        for idir in range(NDIR):                
            kernel_map(commands[0], [GLOBAL,], [LOCAL,],
                USER.MAP_DX, USER.NPIX, MAP_buf, EMIT_buf[0], ODIR[idir], RA[idir], DE[idir],
                LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABS[0], SCA[0],
                USER.MAPCENTRE, USER.INTOBS, Bx_buf, By_buf, Bz_buf, OPT_buf[0])
            # MAP is either a single Healpix map or a flat map of (I, Q, U, N)
            cl.enqueue_copy(commands[0], MAP, MAP_buf)
            # write this frequency to file -- I, Q, U, N
            asarray(MAP, float32).tofile(fpmap[idir])
    for idir in range(NDIR):
        fpmap[idir].close()
# end of -- USER.POLMAP>0






if ((USER.POLMAP>0)&(USER.NOMAP==0)&(USER.NPIX['y']<0)):  # HEALPIX POLARISATION MAPS
    """
    Keep polarisation map completelety separate from the normal map
    can have NOMAP (no regular maps) but still write POLMAP (I, Q, U or other polarisation stats)
    POLSTAT==0   -->  I,Q,U,N 
    POLSTAT==1   -->  statistics  rT, rI, jT, jI  
                      r=rho weighted, j=emission weighted, T=tangling, I=inclination
    POLSTAT==2   -->  I,Q,U,N but non-orthographic, with cube replication and arbitrary LOS length
    """
    if (USER.LIB_MAPS):
        print("*** ERROR: POLMAP (healpix maps) not implemented together with LIB_MAPS"), sys.exit()
    import healpy
    column_names = [ 'I_STOKES', 'Q_STOKES', 'U_STOKES', 'N' ]
    if (USER.POLSTAT>0):
        column_names = [ 'rhoTheta', 'rhoGamma', 'jTheta', 'jGamma' ]
    #### TNEW = zeros(CELLS, float32)        
    program_map, kernel_map = None, None
    # source = open(os.getenv("HOME")+ "/starformation/SOC/kernel_ASOC_map_H.c").read()
    source = open(INSTALL_DIR+ "/kernel_ASOC_map_H.c").read()
    program_map = cl.Program(context[0], source).build(ARGS+' -D NSIDE=%d' % (USER.NPIX['x']))
    kernel_map = program_map.PolHealpixMapping
    # Push the magnetic fields. B files have the same format as density
    #     NX, NY, NZ, LEVELS, CELLS
    #    {  LCELLS, { values } }
    BB = []
    for ii in range(3):  
        print(USER.BFILES[ii])
        BB.append( read_otfile(USER.BFILES[ii]) )
    if (len(USER.file_polred)>0):
        # Polarisation reduction factor R is encoded to the length of the B vectors
        R = None  # the polarisation reduction factor
        if (USER.file_polred=='adhoc'):
            # As an example, we use ad hoc R calculated based on local dust temperature
            R   = read_otfile(USER.file_temperature) # returns a single vector
            # Encode polarisation reduction factor to the norm of B
            #   x =  (T/11.0-1)/1.0,  R =  exp(x) / (exp(x) + exp(-x))
            R   =  (R-13.3)/2.0 + 1.0e-4
            R   =   exp(R) / ( exp(R)+exp(-R) )
        elif (USER.file_polred.find('rhofun')>=0):
            # Use semilogx(x, 0.5*(1.0+tanh( (log10(1e4)-log10(x))/0.5 )))
            # encoding in filename rhofun_1e4_0.5 -- step and logarithmic width of the rise
            s       =   USER.file_polred.split('_')
            th, sw  =   float(s[1]), float(s[2]) # threshold and logarithmic step width
            R       =   read_otfile(USER.file_cloud)
            if (USER.KDENSITY!=1.0): R *= USER.KDENSITY
            
            aa      =   R[0:-1:10].copy()
            R       =   clip(R, 0.1, 1e10)
            R       =   0.5*(1.0+tanh((log10(th)-log10(R))/sw))
            if (0):
                clf()
                semilogx(aa, R[0:-1:10], 'x')
                show()
                sys.exit()            
            
        else:     # argument is a file to file with R values
            if (0):
                R   =  read_otfile(USER.file_polred)
            else:  # run_RAT.py writes R to a plain file == [ cells, { R } ]
                R   = fromfile(USER.file_polred, float32)[1:]
        # encode 0<R<1 to the length of the polarisation vector
        R      =  R  /  sqrt(BB[0]**2 + BB[1]**2 + BB[2]**2)
        BB[0] *=  R
        BB[1] *=  R 
        BB[2] *=  R
        del R
    # push B  to device
    Bx_buf = cl.Buffer(context[0], mf.READ_ONLY, BB[0].nbytes)
    By_buf = cl.Buffer(context[0], mf.READ_ONLY, BB[1].nbytes)
    Bz_buf = cl.Buffer(context[0], mf.READ_ONLY, BB[2].nbytes)
    cl.enqueue_copy(commands[0], Bx_buf, BB[0])
    cl.enqueue_copy(commands[0], By_buf, BB[1])
    cl.enqueue_copy(commands[0], Bz_buf, BB[2])
    commands[0].finish()
    ###
    del BB[2]
    del BB[1]
    del BB[0]
    BB = None
    ###################################################################################
    MAP_buf, kernel_map = None, None
    MAP  = None
    # Healpix map
    npix       = 12*USER.NPIX['x']*USER.NPIX['x']
    MAP        = zeros(4*npix, float32)  #  NPIX[0] == NSIDE
    MAP_buf    = cl.Buffer(context[0], mf.WRITE_ONLY, 4*4*npix)
    kernel_map = program_map.PolHealpixMapping
    GLOBAL     = int((1+floor((npix)/LOCAL))*LOCAL)
    NDIR       = 1
    kernel_map.set_scalar_arg_dtypes(
    # DX          NPIX                  MAP   EMIT
    [ np.float32, clarray.cltypes.int2, None, None,
    # DIR                   RA                      DE
    clarray.cltypes.float3, clarray.cltypes.float3, clarray.cltypes.float3,
    # LCELLS OFF   PAR    DENS
    None,    None, None,  None,
    # ABS        SCA         CENTRE                  INTOBS                
    np.float32,  np.float32, clarray.cltypes.float3, clarray.cltypes.float3,
    # Bx  By    Bz    OPT    Y_SHEAR
    None, None, None, None,  np.float32 ])
            
    KK  = (1.0e23/FACTOR) * PLANCK / (4.0*np.pi)  #  1e3 = 1e23/1e20 = removing scaling, convert to Jy/sr
    KK *= USER.GL*PARSEC    
    for IFREQ in range(REMIT_I1, REMIT_I2+1):    #  loop over frequencies
        if (IFREQ%5==0): 
            sys.stdout.write(" %d" % IFREQ)
            sys.stdout.flush()
        FREQ = FFREQ[IFREQ]
        if ((FREQ<USER.MAP_FREQ[0])|(FREQ>USER.MAP_FREQ[1])): 
            continue  # we are writing only frequencies limited by USER.MAP_FREQ

        ABS[0], SCA[0]  = AFABS[0][IFREQ], AFSCA[0][IFREQ]        
        if (WITH_ABU>0): # ABS, SCA, G are vectors
            OPT[:,:] = 0.0
            if (USER.SINGLE_ABU):
                # we have exactly two dust components with abundances ABU and 1.0-ABU
                OPT[:,0] +=       ABU  * AFABS[0][IFREQ]
                OPT[:,1] +=       ABU  * AFSCA[0][IFREQ]
                OPT[:,0] +=  (1.0-ABU) * AFABS[1][IFREQ]
                OPT[:,1] +=  (1.0-ABU) * AFSCA[1][IFREQ]
            else:
                for idust in range(NDUST):
                    OPT[:,0] +=  ABU[:,idust] * AFABS[idust][IFREQ]
                    OPT[:,1] +=  ABU[:,idust] * AFSCA[idust][IFREQ]
            if (USER.OPT_IS_HALF):
                cl.enqueue_copy(commands[0], OPT_buf[0], asarray(OPT, np.float16))
            else:
                cl.enqueue_copy(commands[0], OPT_buf[0], OPT)
        else:               # constant abundance ->  ABS, SCA, G are scalars
            ABS[0], SCA[0] = 0.0, 0.0
            for idust in range(NDUST):
                ABS[0] += AFABS[idust][IFREQ]
                SCA[0] += AFSCA[idust][IFREQ]
        
        # use kernel to do the LOS integration:  EMIT -> MAP
        EMIT[:] = KK * FREQ * EMITTED[:, IFREQ-REMIT_I1]
        cl.enqueue_copy(commands[0], EMIT_buf[0], EMIT)
        kernel_map(commands[0], [GLOBAL,], [LOCAL,],
        USER.MAP_DX, USER.NPIX, MAP_buf, EMIT_buf[0], ODIR[idir], RA[idir], DE[idir],
        LCELLS_buf[0], OFF_buf[0], PAR_buf[0], DENS_buf[0], ABS[0], SCA[0],
        USER.MAPCENTRE, USER.INTOBS, Bx_buf, By_buf, Bz_buf, OPT_buf[0], USER.Y_SHEAR)
        # MAP is Healpix map
        commands[0].finish()
        cl.enqueue_copy(commands[0], MAP, MAP_buf)
        # write this frequency to file -- I, Q, U, N
        # asarray(MAP, float32).tofile(fpmap[idir])
        healpy.write_map('pol_healpix.fits.%d' % IFREQ, 
        (MAP[(0*npix):(1*npix)], MAP[(1*npix):(2*npix)], MAP[(2*npix):(3*npix)], MAP[(3*npix):(4*npix)]),
        fits_IDL=False, coord='G', 
        column_names=column_names,  overwrite=True)

    #for idir in range(NDIR):
    #    fpmap[idir].close()

        
if (USER.WITH_ROI_SAVE):    del ROI_SAVE
if (USER.WITH_ROI_LOAD):    del ROI_LOAD


###################################################################################
Tpolmap = time.time()-t0


del EMITTED   # this also writes the rest of EMITTED to disk

print("        Tpush     %9.4f seconds" % Tpush)
print("        Tkernel   %9.4f seconds" % Tkernel)
print("        Tpull     %9.4f seconds" % Tpull)
print("        Tsolve    %9.4f seconds" % Tsolve)
print("        Tmap      %9.4f seconds" % Tmap)
    
