#!/bin/python3

import os, sys, time

# we assume that the Python scripts and *.c kernel files are in this directory
# HOMEDIR = os.path.expanduser('~/')
# sys.path.append(HOMEDIR+'/starformation/SOC/')
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)

from SOC_aux import *


SHAREDIR = '/dev/shm'
## SHAREDIR = '/HDD/mika/tt/HIMASS/'
## SHAREDIR = './'

USE_MMAP = False  # False, unless CELLS*NOFREQ does not fit to main memory !!

if (len(sys.argv)<2):
    print("Usage:")
    print("      A2E_MABU.py  ini absorbed.data emitted.data [uselib|makelib|ofreq.dat] ")
    print("Input:")
    print("  soc.ini        =  SOC ini file (listing the dusts and abundances etc.)")
    print("  absorbed.data  =  absorptions  [CELLS, nfreq] or [CELLS, nlfreq]")
    print("  emitted.data   =  solved emissions [CELLS, nfreq]")
    print("  uselib         =  if given, solve using existing libraries (one per dust population)")
    print("  makelib        =  if given, create new libraries")
    print("Note:")
    print("   When using the library method, files freq.dat and lfreq.dat should contain")
    print("   all frequencies and the reference frequencies, respectively. Above nfreq and nlfreq")
    print("   are the number of frequencies in those files, respectively.\n")
    print("Note on the dust names:")
    print("   If this is called by A2E_driver.py, the ini file may contain dust names")
    print("   like gs_aSilx.dust. However, the solver file will not have the gs_ prefix")
    print("   => any gs_ prefixes of dust names will be dropped when reading the ini.\n")
    sys.exit()

    
USELIB, MAKELIB, F_OFREQ =  0, 0, ""
for arg in sys.argv[4:]:
    if (arg=='uselib'):      
        USELIB  = 1
    elif (arg=='makelib'):  
        MAKELIB = 1
    else:
        F_OFREQ = arg

F_LFREQ = "lfreq.dat"

fplog = open('A2E_MABU.log', 'w')
fplog.write("[%s]\n\n" % time.asctime())
fplog.write("A2E_MABU.py")
for i in range(len(sys.argv)): fplog.write(" %s" % sys.argv[i])
fplog.write("\n")


# find dusts and potential abundance file names from the ini
# with USELIB, these are still dust files for the full NFREQ
# DUST[] will contain only names of simple dusts !!
DUST, AFILE, EQDUST = [], [], []
UM_MIN, UM_MAX      = 0.00001, 999999.0
GPU  = 0
LIB_BINS = ''  # bins when making a library with A2E_LIB.py
platforms = []
fp  = open(sys.argv[1], 'r')
for line in fp.readlines():
    s = line.split()
    if (len(s)<2): continue
    if (s[0].find('remit')==0):
        UM_MIN = float(s[1])
        UM_MAX = float(s[2])
    # by default the library reference frequencies are in lfreq.dat and output frequencies in ofreq.dat 
    #  --- these can be overwritten with the libabs and libmap keywords 
    if (MAKELIB & (s[0]=='libabs')):
        F_LFREQ  =  s[1]
    if (MAKELIB & (s[0]=='libbins')):
        LIB_BINS =  s[1]
    if (USELIB  & (s[0]=='libmap')):
        F_OFREQ  =  s[1]
    if (s[0].find('device')>=0):
        if (s[1].find('g')>=0): GPU = 1
        else:                   GPU = 0
    if (s[0].find('platform')>=0):
        platforms = [ int(s[1]), ]    # user requested a specific OpenCL platform
    if (s[0][0:6]=='optica'):
        dustname = s[1]
        tag      = open(dustname).readline().split()[0]
        if (tag=='eqdust'):
            DUST.append(dustname)  # full filename for equilibrium dusts
            EQDUST.append(1)
        else:                      # else SHG = gset dust
            dustname = s[1].replace('_simple.dust','')
            # drop "gs_" from the dust name -- the solver file will be named without "gs_"
            if (dustname[0:3]=='gs_'):                  
                dustname = dustname[3:]
            DUST.append(dustname.replace('.dust', ''))  # dust basename
            EQDUST.append(0)
        if (len(s)>2):      # we have abundance file
            if (s[2]=="#"):
                AFILE.append("")
            else:
                AFILE.append(s[2])
        else:
            AFILE.append("")
fp.close()
fplog.write("%s ->  GPU=%d\n" % (sys.argv[1], GPU))


# GPU is passed onto A2E.py on command line .... 1.3 would mean GPU and platform 3 !!!
GPU_TAG = 0
if (GPU>0):                 # GPU was specified on command line or in the ini file
    if (len(platforms)>0):  #  [] or one value read from the ini file
        GPU_TAG = 1.0 + 0.1*platforms[0]  # encoded into float for the A2E.py command line argument
    else:
        GPU_TAG = 1         # platform not specified
# we need platforms also in this script (solving of equilibrium temperature dust)
if (len(platforms)<1):
    platforms = arange(5)
    


print("A2E_MABU.py   makelib %d, uselib %d, ini %s" % (MAKELIB, USELIB, sys.argv[1]))
fplog.write("\nMAKELIB %d, USELIB %d\n" % (MAKELIB, USELIB))
fplog.write("DUST\n")
for x in DUST:
    fplog.write("    %s\n" % x)   # this will be aSilx ... not gs_aSilx or aSilx_simple
fplog.write("EQDUST\n")
for x in EQDUST:
    fplog.write("    %s\n" % x)


            
# read KABS for each dust from the solver files
NDUST = len(DUST)
GD    = np.zeros(NDUST, np.float32)  # grain density
RABS  = []
FREQ  = []
NFREQ = 0
for idust in range(NDUST):
    print("=== dust %d: %s" % (idust, DUST[idust]))
    # Note: with USELIB, we will use *_simple.dust for the full original NFREQ frequency grid
    if (EQDUST[idust]):
        print("=== EQDUST")
        lines      =  open(DUST[idust]).readlines()
        GD[idust]  =  float(lines[1].split()[0])
        radius     =  float(lines[2].split()[0])
        d          =  np.loadtxt(DUST[idust], skiprows=4)
        NFREQ      =  d.shape[0]
        FREQ       =  d[:,0]
        kabs       =  np.pi*radius**2.0*GD[idust] * d[:,2]
        if (len(RABS)<2):
            RABS = np.zeros((NFREQ, NDUST), np.float64)  # total cross section per unit density
        RABS[:,idust] = kabs
    else:
        # Stochastically heated grains
        print("=== SHG")
        fp     = open('%s.solver' % DUST[idust], 'rb') # solver will be called aSilx.solver, not gs_aSilx.solver...
        NFREQ  = np.fromfile(fp, np.int32, 1)[0]
        FREQ   = np.fromfile(fp, np.float32, NFREQ)
        GD     = np.fromfile(fp, np.float32, 1)[0]
        NSIZE  = np.fromfile(fp, np.int32, 1)[0]
        S_FRAC = np.fromfile(fp, np.float32, NSIZE)
        NE     = np.fromfile(fp, np.int32, 1)[0]
        SK_ABS = np.fromfile(fp, np.float32, NSIZE*NFREQ).reshape(NSIZE,NFREQ)  # Q*pi*a^2 * GD*S_FRAC
        SK_ABS = np.asarray(SK_ABS, np.float64)
        fp.close()
        if (len(RABS)<2):
            RABS = np.zeros((NFREQ, NDUST), np.float64)    # relative absorption per grain population
        RABS[:,idust] = sum(SK_ABS, axis=0)          # total cross section as sum over sizes
        fp.close()

        
NLFREQ  = 0
if (USELIB | MAKELIB):
    LFREQ  = loadtxt(F_LFREQ)           # read lfreq.dat == reference frequencies
    NLFREQ = len(LFREQ)                 # at this point NFREQ = number of absorption channels
    IFREQ  = zeros(len(LFREQ), int32)
    for i in range(len(LFREQ)):
        IFREQ[i] = argmin(abs(LFREQ[i]-FREQ))
    print("xxx  IFREQ = ", IFREQ)
    ## sys.exit()
    
NOFREQ = NFREQ
if (USELIB):  # we may limit output to frequencies on OFREQ
    if (len(F_OFREQ)>0):
        OFREQ  = loadtxt(F_OFREQ)
        if (size(OFREQ)==1): OFREQ = asarray(OFREQ,float32).reshape(1,)
    else:
        OFREQ = FREQ
    NOFREQ = len(OFREQ)
print(NOFREQ)    
# sys.exit()

# file of absorptions should start with three integers
H             =  np.fromfile(sys.argv[2], np.int32, 2)
CELLS, NFFREQ = H[0], H[1]
print("=== Absorption file:  CELLS %d, NFFREQ %d, NFREQ %d, NDUST %d" % (CELLS, NFFREQ, NFREQ, NDUST))

if (USELIB & (NFFREQ==NLFREQ)): 
    # Absorption file has already been truncate to reference frequencies only
    # => RABS is converted from RABS[NFREQ, NDUST] to RABS[NLFREQ, NDUST]
    RABS       =  RABS[IFREQ,:]        # RABS[NLFREQ, NDUST]
    RABS.shape = (len(LFREQ), NDUST)   # make sure it stays a 2d array
    ## USELIB .... absorption file may have 3 or NFREQ frequencies !!!
    

    
# Convert RABS[NFREQ, NDUST] to better normalised relative cross section
RABS    = clip(RABS, 1.0e-40, 1.0e30)   # must not be zero 
for ifreq in range(RABS.shape[0]):
    RABS[ifreq,:]  /= (1.0e-40+sum(RABS[ifreq,:]))
RABS    =  np.clip(RABS, 1.0e-30, 1.0)                       # RABS[freq, dust]

    
print("=== A2E_MABU.py .... NFREQ %d" % NFREQ)
fplog.write("NFREQ = %d, NLFREQ = %d\n" % (NFREQ, NLFREQ))


C_LIGHT =  2.99792458e10  
PLANCK  =  6.62606957e-27 
H_K     =  4.79924335e-11 
D2R     =  0.0174532925       # degree to radian
PARSEC  =  3.08567758e+18 
H_CC    =  7.372496678e-48 


def PlanckSafe(f, T):  # Planck function
    # Add clip to get rid of warnings
    return 2.0*H_CC*f*f*f / (np.exp(np.clip(H_K*f/T,-100,+100))-1.0)


def opencl_init(GPU, platforms, verbose=False):
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
                print("--------------------------------------------------------------------------------")
                print("GPU=%d,  TRY PLATFORM %d" % (GPU, iplatform))
                print("NUMBER OF PLATFORMS: %d" % len(tmp))
                print("PLATFORM %d = " % iplatform, tmp[iplatform])
                print("DEVICE ",         tmp[iplatform].get_devices())
                print("--------------------------------------------------------------------------------")
            try:
                platform  = cl.get_platforms()[iplatform]
                if (GPU):
                    device  = platform.get_devices(cl.device_type.GPU)
                else:
                    device  = platform.get_devices(cl.device_type.CPU)
                context  = cl.Context(device)
                queue    = cl.CommandQueue(context)
                ok       = True
                if (verbose): print("    ===>     DEVICE ", device, " ACCEPTED !!!")
                break
            except:                
                if (verbose): print("    ===>     DEVICE ", device, " REJECTED !!!")                
                pass
        if (ok):
            return context, queue, cl.mem_flags
        else:
            if (iii==0):
                platforms = arange(4)  # try without specific choise of platform
            else:
                print("*** A2E_MABU => opencl_ini could not find valid OpenCL device *** ABORT ***")
                time.sleep(10)
                sys.exit()
            



def SolveEquilibriumDust(dust, f_absorbed, f_emitted, UM_MIN=0.0001, UM_MAX=99999.0, GPU=False, platforms=[0,1,2,3,4]):
    """
    Calculate equilibrium temperature dust emission based on absorptions.
    Input:
        dust            =   name of the dust file (type eqdust)
        f_absorbed      =   file name for absorptions = CELLS, NFREQ, floats[CELLS*NFREQ]
        f_emitted       =   file name for emissions   = CELLS, NFREQ, floats[CELLS*NFREQ]
        UM_MIN, UM_MAX  =   limits output frequencies
        GPU             =   if True, use GPU instead of CPU
    """
    # Read dust data
    print("      SolveEquilibriumDust(%s)" % dust)
    # fplog.write("      SolveEquilibriumDust(%s)\n" % dust)
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
    print("      Mapping EOUT ==>   Emin %12.4e, Emax %12.4e\n" % (Emin, Emax))
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
    print("      A2E_MABU.py Solve temperatures")
    # Open file containing absorptions
    FP_ABSORBED =  open(f_absorbed, 'rb')
    CELLS, NFREQ=  np.fromfile(FP_ABSORBED, np.int32, 2) # get rid of the header
    print("      SolveEquilibriumDust(%s): CELLS %d, NFREQ %d" % (f_absorbed, CELLS, NFREQ))
    t0          =  time.time()
    for ibatch in range(int(CELLS/GLOBAL+1)):
        a   =  ibatch*GLOBAL
        b   =  min([a+GLOBAL, CELLS])  # interval is [a,b[
        # no need to use mmap file for absorptions - values are read in order
        print("      Solving  eqdust [%6d, %6d[ OUT OF %6d" % (a, b, CELLS))
        tmp[0:((b-a)*NFREQ)] =  np.fromfile(FP_ABSORBED, np.float32, (b-a)*NFREQ)
        cl.enqueue_copy(commands, ABS_buf, tmp)
        kernel_T(commands, [GLOBAL,], [LOCAL,], a, kE, oplgkE, Emin, NE, FREQ_buf, TTT_buf, ABS_buf, T_buf)
    cl.enqueue_copy(commands, TNEW, T_buf)    
    a, b, c  = np.percentile(TNEW, (10.0, 50.0, 90.0))
    print("      Solve temperatures: %.2f seconds =>  %10.3e %10.3e %10.3e" % (time.time()-t0, a,b,c))
    FP_ABSORBED.close()
    
    if (0):
        np.asarray(TNEW, np.float32).tofile(SHAREDIR+'/TNEW.bin')
    else:
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
    ofreq    =  FREQ[MOUT]
    nofreq   =  len(ofreq)
    # Solve emission one frequency at a time, all cells on a single call
    print("      A2E_MABU.py  Solve emitted for nofreq = %d frequencies" % nofreq)
    t0       =  time.time()
    if (USE_MMAP):
        np.asarray([CELLS, nofreq], np.int32).tofile(f_emitted)
        EMITTED      =  np.memmap(f_emitted, dtype='float32', mode='r+', offset=8, shape=(CELLS,nofreq))
        EMITTED[:,:] = 0.0
    else:
        EMITTED      = np.zeros((CELLS, nofreq), np.float32)        
    # **** BAD ****  ----   update has outer loop over FREQ,  inner over CELLS
    #                storage order has outer loop over CELLS, inner over FREQ 
    for ifreq in MOUT[0]:            # ifreq = selected frequencies, index to full list of frequencies
        oifreq = ifreq-MOUT[0][0]    # index to the set of output frequencies
        kernel_emission(commands, [GLOBAL,], [LOCAL,], FREQ[ifreq], KABS[ifreq], T_buf, EMIT_buf)
        cl.enqueue_copy(commands, TNEW, EMIT_buf)
        commands.finish()
        EMITTED[:, oifreq] = TNEW    # OUT OF ORDER UPDATE --- HOPE FOR SMALL nofreq!
        ## print("ifreq %3d   ofreq %3d   %10.3e" % (ifreq, oifreq, mean(TNEW)))
    print("      A2E_MABU.py  Solve emitted: %.2f seconds" % (time.time()-t0))
    if (USE_MMAP):
        del EMITTED
    else:
        fp = open(f_emitted, 'wb')
        np.asarray([CELLS, nofreq], np.int32).tofile(fp)
        EMITTED.tofile(fp)  #  B(T)*kappa for a single dust component
        fp.close()
    return nofreq
        
        


FPE = None
    
fplog.write("\nAbsorption file: CELLS %d, NFREQ %d, NDUST %d\n" % (CELLS, NFREQ, NDUST))


# Read abundance files... we must have enough memory for that
ABU = np.ones((CELLS, NDUST), np.float32)
for idust in range(NDUST):
    if (len(AFILE[idust])>1): # we have a file for the abundance of the current dust species
        ABU[:,idust] = np.fromfile(AFILE[idust], np.float32, CELLS)
    

# Initialise OpenCL to split the absorptions
# First check the number of frequencies in the absorption file...
#    muist be NFREQ or possibly NLFREQ in case of USELIB
fp_absorbed   = open(sys.argv[2], 'rb')
CELLS, NFFREQ = np.fromfile(fp_absorbed, np.int32, 2)  # USELIB => FNFREQ is either NFREQ or LNFREQ
fp_absorbed.close()
nkfreq        = NFFREQ

## No - if absorption file contains NFREQ frequencies, use the kernel to 
##      split those for all the frequencies ==> if cell cannot be solved
##      with library look-up, we can still solve the emission directly !!!!
if (0):
    if (USELIB & (NFFREQ!=NLFREQ)):  # USELIB => only reference frequencies to kernel
        nkfreq    = NLFREQ


# at this point nkfreq is the number of frequencies passed to the split kernel
if (0):
    print("Initialize OpenCL for splitting absorptions => ALWAYS ON CPU !!")
    context, queue, mf = opencl_init(GPU=0, platforms=platforms)
else:
    print("Initialize OpenCL for splitting absorptions => GPU=%d" % GPU)
    context, queue, mf = opencl_init(GPU, platforms=platforms)
source      =  open(INSTALL_DIR+"/kernel_A2E_MABU_aux.c").read()
OPTS        =  '-D NFREQ=%d -D NDUST=%d' % (nkfreq, NDUST)  # USELIB => NFREQ==3
program     =  cl.Program(context, source).build(OPTS)
Split       =  program.split_absorbed
Split.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None, None])
BATCH       =  32768
GLOBAL, LOCAL = BATCH, 16
ABS_IN_buf  =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH*nkfreq)
ABS_OUT_buf =  cl.Buffer(context, mf.WRITE_ONLY, 4*BATCH*nkfreq)
RABS_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=RABS) # RABS[ NFREQ|NLFREQ, NDUST]
ABU_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH*NDUST)                  # ABU[CELLS, NDUST]


# loop over dust populations
### NOFREQ = NFREQ    # normally output is all simulated frequencies
fplog.write("\nLoop over dust components\n")
for IDUST in range(NDUST):
    t0 = time.time()
    # write a new file for absorptions, the photons absorbed by this dust component
    fp1 = open(SHAREDIR+'/tmp.absorbed', 'wb')
    # nkfreq is now whatever was in the input absorption file, hopefully NFREQ
    asarray([CELLS, nkfreq], int32).tofile(fp1)
    fp_absorbed  = open(sys.argv[2], 'rb')
    CELLS, FNFREQ = np.fromfile(fp_absorbed, np.int32, 2)  # USELIB => FNFREQ is either NFREQ or LNFREQ
    print("DUST %d/%d -- CELLS %d, FNFREQ %d" % (1+IDUST, NDUST, CELLS, NFFREQ))
    fplog.write("  DUST %d/%d, CELLS %d, NFREQ %d ... split absorbed\n" % (IDUST, NDUST, CELLS, NFREQ))
    if (USELIB):
        if ((NFFREQ!=NFREQ)&(NFFREQ!=NLFREQ)):
            print("ABSORPTIONS IN THE FILE: NFFREQ=%d NOT NFREQ=%d AND NOT NLFREQ=%d" % (NFFREQ, NFREQ, NLFREQ))
            sys.exit()
    else:
        if (NFFREQ!=NFREQ):
            print("ABSORPTIONS IN THE FILE: NFFREQ=%d NOT NFREQ=%d" % (NFFREQ, NFREQ))
            sys.exit()
    # OpenCL used to split the absorptions => absorptions by the current dust species, 
    #    absorbed[icell] =  absorbed[icell] * RABS[ifreq,idust] / K
    #    K               =  sum[   ABU[icell, idust] * RABS[ifreq, idust]  ]
    #  process BATCH cells at a time
    # USELIB  =>  RABS was already limited to NLFREQ frequencies
    #             below also ABS_IN_buf gets only absortions for NLFREQ, not NFREQ frequencies
    tmp_F       =  zeros((BATCH, NFFREQ), float32)  #  allocation to read the file, == NFREQ or NLFREQ
    tmp         =  zeros((BATCH, nkfreq), float32)  #  whatever kernel gives, NFREQ or NLFREQ
    a = 0
    while(a<CELLS):
        b                 =  min(CELLS, a+BATCH)                             # cells [a,b[
        tmp_F[0:(b-a),:]  =  fromfile(fp_absorbed, np.float32, (b-a)*NFFREQ).reshape((b-a), NFFREQ)   # tmp[icell, ifreq]
        if (nkfreq==NFFREQ):    # "normal", all read frequencies passed to the kernel
            cl.enqueue_copy(queue, ABS_IN_buf, tmp_F)                        # ABS_IN[batch, nkfreq], nkfreq=NFREQ or NLFREQ
        else:                   # USELIB -- pick reference frequencies NLFREQ < NFREQ
            tmp[0:(b-a), :]  =  tmp_F[0:(b-a), IFREQ]
            cl.enqueue_copy(queue, ABS_IN_buf, tmp)                          # ABS_IN[batch, nkfreq], nkfreq=NLFREQ
        # sys.stdout.write(" %10.3e %10.3e %10.3e  " % (tmp[0, 10], tmp[0,20], tmp[0,42]))
        cl.enqueue_copy(queue, ABU_buf, ABU[a:b, :])                   # ABU[batch, ndust]
        # USELIB => RABS and ABS_IN both for the reference frequencies only
        Split(queue, [GLOBAL,], [LOCAL,], IDUST, b-a, RABS_buf, ABU_buf, ABS_IN_buf, ABS_OUT_buf)
        cl.enqueue_copy(queue, tmp[0:(b-a), :], ABS_OUT_buf)           # tmp[BATCH, nkfreq]
        tmp[0:(b-a),:].tofile(fp1)                                     # absorption file possibly inly reference frequencies
        # sys.stdout.write(" -> %10.3e %10.3e %10.3e   NFREQ=%d\n" % (tmp[0,10], tmp[0,20], tmp[0,42], NFREQ))
        a             +=  BATCH            
    # --------------------------------------------------------------------------------
    fp_absorbed.close()
    fp1.close() # 8 + 4*CELLS*NFREQ bytes
    print("=== Split absorbed: %.2f seconds" % (time.time()-t0))
    fplog.write("      Split absorbed: %.2f seconds\n" % (time.time()-t0))
    
    # the written absorption file has either all NFREQ frequencies or, in the case of USELIB,
    # possibly only the reference frequencies => but soc_library.py can deal with both
    
    
    fplog.write("      Solve emission, NOFREQ %d, MAKELIB %d, USELIB %d, NFREQ %d, nkfreq %d\n" % (NOFREQ, MAKELIB, USELIB, NFREQ, nkfreq))
    print("NOFREQ = NFREQ = %d .... dust %s" % (NOFREQ, DUST[IDUST]))
    print("")
    
    t0 = time.time()
    if (EQDUST[IDUST] & (USELIB==0)):
        # Equilibrium dust, also calculating emission to SHAREDIR/tmp.emitted
        # MAY INCLUDE ONLY FREQUENCIES [UM_MIN, UM_MAX]
        print("EQUILIBRIUM DUST %s  --  MAKELIB %d" % (DUST[IDUST], MAKELIB))
        if (MAKELIB):  
            # 2021-01-23 -- make the library using soc_library.py ... even for equilibrium dust
            fplog.write('      soc_library.py  %.1f %s %s/tmp.absorbed\n' % (GPU_TAG, DUST[IDUST], SHAREDIR))
            os.system(        'soc_library.py  %.1f %s %s/tmp.absorbed'   % (GPU_TAG, DUST[IDUST], SHAREDIR))
            continue  # skip the writing of emission file --- only libraries are produced
        else:
            # normal case - solve without library
            nofreq = SolveEquilibriumDust(DUST[IDUST], SHAREDIR+'/tmp.absorbed', SHAREDIR+'/tmp.emitted', UM_MIN, UM_MAX, GPU, platforms)
            fplog.write("      SolveEquilibriumDust(%s) => nofreq %d, NOFREQ %d\n" % (DUST[IDUST], nofreq, NOFREQ))
    else:        
        if (USELIB):    # solve using existing library
            print("      STOCHASTIC DUST--- USELIB=%d, MAKELIB=%d, DUST %s" % (USELIB, MAKELIB, DUST[IDUST]))
            if (0):
                # emitted will contain values only for frequencies selected in OFREQ = map frequencies
                dname = DUST[IDUST].replace('_simple.dust', '')  #  uselib.ini had simple dusts
                print("================================================================================")
                print('      A2E_LIB.py %s.solver %s.lib freq.dat %s %s/tmp.absorbed %s/tmp.emitted %s %s' % \
                (dname, dname, F_LFREQ, SHAREDIR, SHAREDIR, ["", "GPU"][GPU], F_OFREQ))
                print("================================================================================")
                fplog.write('      A2E_LIB.py %s.solver %s.lib  freq.dat  %s  %s/tmp.absorbed %s/tmp.emitted %s %s\n' % \
                (dname, dname, F_LFREQ, SHAREDIR, SHAREDIR, ["", "GPU"][GPU], F_OFREQ))            
                os.system('A2E_LIB.py %s.solver   %s.lib  freq.dat  %s  %s/tmp.absorbed %s/tmp.emitted %s %s' % \
                (dname, dname, F_LFREQ, SHAREDIR, SHAREDIR, ["", "GPU"][GPU], F_OFREQ))
            else:       # MAKELIB - can be equilibrium dust or stochastically heated grains
                if (EQDUST[IDUST]):
                    fplog.write('      soc_library.py  %.1f %s %s/tmp.absorbed %s/tmp.emitted %s\n' % (GPU_TAG, DUST[IDUST], SHAREDIR, SHAREDIR, F_OFREQ))
                    os.system(        'soc_library.py  %.1f %s %s/tmp.absorbed %s/tmp.emitted %s'   % (GPU_TAG, DUST[IDUST], SHAREDIR, SHAREDIR, F_OFREQ))
                else:
                    fplog.write('      soc_library.py  %.1f %s.solver %s/tmp.absorbed %s/tmp.emitted %s\n' % (GPU_TAG, DUST[IDUST], SHAREDIR, SHAREDIR, F_OFREQ))
                    os.system(        'soc_library.py  %.1f %s.solver %s/tmp.absorbed %s/tmp.emitted %s'   % (GPU_TAG, DUST[IDUST], SHAREDIR, SHAREDIR, F_OFREQ))
        elif (MAKELIB): # solve every cell and also build the library
            # Note --- file names freq.dat and lfreq.dat hardcoded!
            # Library is always for full NFREQ output frequencies, USELIB can/will limit output with ofreq.dat
            if (0):
                """
                A2E_LIB.py PAH0_MC10_1.solver PAH0_MC10_1.lib freq.dat lfreq.dat ./tmp.absorbed ./tmp.emitted GPU makelib
                """
                print("--------------------------------------------------------------------------------")
                print('A2E_LIB.py %s.solver %s.lib freq.dat %s %s/tmp.absorbed %s/tmp.emitted %s makelib %s' % \
                (DUST[IDUST], DUST[IDUST], F_LFREQ, SHAREDIR, SHAREDIR, ["", "GPU"][GPU], LIB_BINS))
                print("--------------------------------------------------------------------------------")
                fplog.write('      A2E_LIB.py %s.solver %s.lib freq.dat %s %s/tmp.absorbed %s/tmp.emitted %s makelib %s\n' % \
                (DUST[IDUST], DUST[IDUST], F_LFREQ, SHAREDIR, SHAREDIR, ["", "GPU"][GPU], LIB_BINS))
                os.system(  'A2E_LIB.py %s.solver %s.lib freq.dat %s %s/tmp.absorbed %s/tmp.emitted %s makelib %s' % \
                (DUST[IDUST], DUST[IDUST], F_LFREQ, SHAREDIR, SHAREDIR, ["", "GPU"][GPU], LIB_BINS))
            else:  # 2021-01-23 -- make the library using soc_library.py
                fplog.write('      soc_library.py  %.1f %s.solver %s/tmp.absorbed\n' % (GPU_TAG, DUST[IDUST], SHAREDIR))
                os.system(        'soc_library.py  %.1f %s.solver %s/tmp.absorbed' % (GPU_TAG, DUST[IDUST], SHAREDIR))
            continue    # skip the writing of emission file --- only library is being made
        else:           # stochastic heating, solve with A2E.py in the normal way, no libraries are involved
            # note -- GPU can be  e.g. 0.1 for CPU/device=1  or 1.3 for GPU/device=3
            print("================================================================================")
            print('      A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f' % (DUST[IDUST], SHAREDIR, SHAREDIR, GPU))
            print("================================================================================")
            fplog.write('      A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f\n' % (DUST[IDUST], SHAREDIR, SHAREDIR, GPU))
            os.system('A2E.py  %s.solver %s/tmp.absorbed %s/tmp.emitted  %.1f' % (DUST[IDUST], SHAREDIR, SHAREDIR, GPU))
            
        if ((USELIB==0)&(NOFREQ!=NFREQ)):  # so far only equilibrium dust files can have NOFREQ<NFREQ !!
            print("=== A2E_MABU.py --- stochastically heated grains with equilibrium grains")
            print("    different number of frequencies... fix A2E.py to use a2e_wavelength parameter??")
            sys.exit()

        
    # read the emissions and add to FPE
    if (IDUST==0): # FPE opened only here, once we know the number of output frequencies
        if (USE_MMAP):
            fp  =  open(sys.argv[3], 'wb')
            np.asarray([CELLS, NOFREQ], np.int32).tofile(fp)
            fp.close()
            FPE =  np.memmap(sys.argv[3], dtype='float32', mode='r+', shape=(CELLS, NOFREQ), offset=8)
            FPE[:,:] = 0.0
        else:
            print(CELLS, NOFREQ)
            FPE = np.zeros((CELLS, NOFREQ), np.float32)

    print("=== Add emitted to sum file: %s  DUST %s" % (sys.argv[3], DUST[IDUST]))
    t0 = time.time()
    filename =  SHAREDIR+'/tmp.emitted'
    print("    READING EMISSION FROM %s .... processing %s, with NOFREQ=%d" % (filename, DUST[IDUST], NOFREQ))
    fplog.write('      Add emission from %s .... processing %s with NOFREQ=%d\n' % (filename, DUST[IDUST], NOFREQ))
    
    fp2 = open(filename, 'rb')
    cells_e, nfreq_e = np.fromfile(fp2, np.int32, 2)  # get rid of the header (CELLS, NFREQ)
    print("    emitted file for %s, cells %d, CELLS %d, nfreq %d, NOFREQ %d" % \
    (DUST[IDUST], cells_e, CELLS, nfreq_e, NOFREQ))
    # **** GOOD **** ---- single dust emission added to the total in the file order

    t00 = time.time()
    if (0):
        # SLOW !
        for ICELL in range(CELLS):
            # file = B(T)*KABS for a single dust component, total is sum of B(T)*KABS*ABU
            # FPE[ICELL,:] += np.fromfile(fp2, np.float32, NOFREQ) * ABU[ICELL, IDUST]
            xxxx = np.fromfile(fp2, np.float32, NOFREQ)
            if (ICELL%1000000==0): 
                print("=== A2E_MABU %8d/%8d, add emission, NOFREQ %d" % (ICELL, CELLS, len(xxxx)))
            FPE[ICELL,:] += xxxx * ABU[ICELL, IDUST]
    else:
        # MUCH FASTER
        a = 0
        while(a<CELLS):
            b = min(a+1024, CELLS)
            FPE[a:b,:] += np.fromfile(fp2, np.float32, (b-a)*NOFREQ).reshape(b-a, NOFREQ) * \
                          ABU[a:b, IDUST].reshape(b-a,1)
            a += 1024
    print("*** ADD EMITTED TO FPE: %.2f SECONDS" % (time.time()-t00))
    fplog.write('      Added emitted to FPE: %.2f seconds\n' % (time.time()-t00))
    
    fp2.close()
    print("    Add emitted to sum file: %.2f seconds =====" % (time.time()-t0))
    # os.system('rm /dev/shm/tmp.absorbed /dev/shm/tmp.emitted')


fplog.write('Loop over dusts completed\n')
fplog.write('\n[%s]\n' % time.asctime())

if (MAKELIB==False):
    if (USE_MMAP):    
        del FPE  # file should now contain combined emission from all dust components
    else:
        fp  =  open(sys.argv[3], 'wb')
        np.asarray([CELLS, NOFREQ], np.int32).tofile(fp)
        FPE.tofile(fp)
        fp.close()
    

        
