#!/bin/python3

import os, sys
HOMEDIR = os.path.expanduser('~/')

# we assume that the Python scripts and *.c kernel files are in this directory
sys.path.append(HOMEDIR+'/starformation/SOC/')
from SOC_aux import *


"""
Given a SOC ini file, calculate total emission for 
a run with potentially several dust components with
variable abundances.

If SOC ini has a dust aSilx_simple.dust, we assume 
that the corresponding GSETDustO format dust file
exists and is named aSilx.dust.

In the ini, each dust is specified with a line like

  optical  aSilx_simple.dust  my.abundances
  
where the second argument is an optional abundance
file.

We also need for the solver pre-calculated data that
are stored in files like aSilx.solver.

This script:
    (1) reads KABS from the *solver files
    (2) calculates the absorptions for each
        individual dust component 
    (3) calls A2E_pyCL.py to solve emission
        from each individual dust population
    (4) adds the emissions to a single
        output file
        
2018-12-28:
    correct results when ran on two dust components with sum(ABU)=1.0
    
2019-02-15:
    option for using equilibrium temperature dust
    
2019-02-20:
    no longer reads wavelength parameter from the ini file
    in SOC/CRT that affects only the writing of the spectra, not
    absorbed and emitted files !!
    
2019-02-24:
    - use remit parameter, because of slow performance
      for very large models with many output frequencies
    - absorptions are read from file in the storage order, no large in-memory arrays
    - USE_MMAP=False is default: assumes that emissions can be stored
      to an in-memory array[CELLS*ONFREQ], ok even for HIMASS models when
      ONFREQ is limited to 8
      if [CELLS*ONFREQ] does not fit to main memory, must use USE_MMAP = slow
      
2019-09-28:
    USELIB -> ini contains dusts <dust>_lib_simple because dust must only contain data
    for the reference frequencies and the actual lib files already exist to compute the emission
    ->  output full set of frequencies ***OR*** only what is given in the ini with ofreq !!!!!!!
"""


## SHAREDIR = '/dev/shm/'
## SHAREDIR = '/HDD/mika/tt/HIMASS/'
SHAREDIR = './'


USE_MMAP = False  # False, unless CELLS*ONFREQ does not fit to main memory !!

if (len(sys.argv)<2):
    print("Usage:")
    print("      A2E_MABU.py  ini absorbed.data emitted.data [GPU] [uselib|makelib]")
    print("Input:")
    print("  soc.ini        =  SOC ini file (listing the dusts and abundances etc.)")
    print("  absorbed.data  =  absorptions  [CELLS, nfreq] or [CELLS, nlfreq]")
    print("  emitted.data   =  solved emissions [CELLS, nfreq]")
    print("  GPU            =  if given, use GPU instead of CPU")
    print("  uselib         =  if given, solve using existing libraries (one per dust population)")
    print("  makelib        =  if given, create new libraries")
    print("Note:")
    print("   When using the library method, files freq.dat and lfreq.dat should contain")
    print("   all frequencies and the reference frequencies, respectively. Above nfreq and nlfreq")
    print("   are the number of frequencies in those files, respectively.")
    sys.exit()

    
GPU, USELIB, MAKELIB = 0, 0, 0
for arg in sys.argv[4:]:
    if (arg.lower()=='gpu'): GPU     = 1
    if (arg=='uselib'):      USELIB  = 1
    if (arg=='makelib'):     MAKELIB = 1


# find dusts and potential abundance file names from the ini
DUST, AFILE, EQDUST = [], [], []
UM_MIN, UM_MAX      = 0.00001, 999999.0
fp  = open(sys.argv[1], 'r')
OFREQ = ""
for line in fp.readlines():
    s = line.split()
    if (len(s)<2): continue
    if (s[0].find('remit')==0):
        UM_MIN = float(s[1])
        UM_MAX = float(s[2])
    if (s[0][0:4]=='ofreq'): OFREQ = s[1] # USELIB + OFREQ => EMISSION FOR THESE FREQUENCIES ONLY
    if (s[0][0:6]=='optica'):
        # ini should have "gs_aSilx.dust" or "gs_aSilx_simple.dust" !!
        dustname = s[1]
        tag      = open(dustname).readline().split()[0]
        if (tag=='eqdust'):
            DUST.append(dustname)  # full filename for equilibrium dusts
            EQDUST.append(1)
        else:
            dustname = s[1].replace('_simple.dust','')
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


        
# Dust initialisations            
            
# read KABS for each dust from the solver files
NDUST = len(DUST)
GD    = np.zeros(NDUST, np.float32)  # grain density
RABS  = []
FREQ  = []
NFREQ = 0
for idust in range(NDUST):
    print("dust %d: %s" % (idust, DUST[idust]))
    if (EQDUST[idust]):
        lines     =  open(DUST[idust]).readlines()
        GD[idust] =  float(lines[1].split()[0])
        radius    =  float(lines[2].split()[0])
        d         =  np.loadtxt(DUST[idust], skiprows=4)
        NFREQ     =  d.shape[0]
        FREQ      =  d[:,0]
        kabs      =  np.pi*radius**2.0*GD[idust] * d[:,2]
        if (len(RABS)<2):
            RABS = np.zeros((NFREQ, NDUST), np.float64)  # total cross section per unit density
        RABS[:,idust] = kabs
    else:
        # Stochastically heated grains
        fp = open('%s.solver' % DUST[idust], 'rb')
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

        
# Convert RABS to better normalised relative cross section
for ifreq in range(NFREQ):
    RABS[ifreq,:]  /= (1.0e-30+sum(RABS[ifreq,:]))
RABST   =  np.clip(np.transpose(RABS.copy()), 1.0e-35, 1e30)

print("A2E_MABU.py .... NFREQ %d" % NFREQ)


C_LIGHT =  2.99792458e10  
PLANCK  =  6.62606957e-27 
H_K     =  4.79924335e-11 
D2R     =  0.0174532925       # degree to radian
PARSEC  =  3.08567758e+18 
H_CC    =  7.372496678e-48 


def PlanckSafe(f, T):  # Planck function
    # Add clip to get rid of warnings
    return 2.0*H_CC*f*f*f / (np.exp(np.clip(H_K*f/T,-100,+100))-1.0)


def opencl_init(GPU, requested_platform=-1):
    """
    Initialise OpenCL environment.
    """
    try_platforms    = np.arange(4)
    if (requested_platform>=0): 
        try_platforms = [requested_platform,]
    platform, device, context, queue = None, None, None, None
    for iplatform in try_platforms:
        print("TRY PLATFORM %d" % (iplatform))
        try:
            platform  = cl.get_platforms()[iplatform]
            if (GPU):
                device  = platform.get_devices(cl.device_type.GPU)
            else:
                device  = platform.get_devices(cl.device_type.CPU)
            context  = cl.Context(device)
            queue   = cl.CommandQueue(context)
            break
        except:
            pass
    return context, queue, cl.mem_flags
            



def SolveEquilibriumDust(dust, f_absorbed, f_emitted, UM_MIN=0.0001, UM_MAX=99999.0, GPU=False):
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
    print("SolveEquilibriumDust(%s)" % dust)
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
    print("Mapping EOUT ==>   Emin %12.4e, Emax %12.4e\n" % (Emin, Emax))
    # E ~ T^4  => use logarithmic sampling
    kE          =  (Emax/Emin)**(1.0/(NE-1.0))  # E[i] = Emin*pow(kE, i)
    oplgkE      =  1.0/np.log10(kE)
    # oplgkE = 1.0/log10(kE)
    ip          =  interp1d(Eout, TT)           # (linear) interpolation from energy to temperature
    TTT         =  np.asarray(ip(Emin * kE**np.arange(NE)), np.float32)
    # Set up kernels
    CELLS, NFREQ=  np.fromfile(f_absorbed, np.int32, 2)
    context, commands, mf = opencl_init(GPU)
    source      =  open(os.getenv("HOME")+ "/starformation/SOC/kernel_eqsolver.c").read()
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
    print("===== Solve temperatures =====")
    # Open file containing absorptions
    FP_ABSORBED =  open(f_absorbed, 'rb')
    print("OPENED FP_ABSORBED: ", FP_ABSORBED)    
    CELLS, NFREQ=  np.fromfile(FP_ABSORBED, np.int32, 2) # get rid of the header
    print("SolveEquilibriumDust(%s): CELLS %d, NFREQ %d" % (f_absorbed, CELLS, NFREQ))
    print("FP_ABSORBED: ", FP_ABSORBED)    
    t0          =  time.time()
    for ibatch in range(int(CELLS/GLOBAL+1)):
        # print("BATCH %d" % ibatch) 
        a   =  ibatch*GLOBAL
        b   =  min([a+GLOBAL, CELLS])  # interval is [a,b[
        # no need to use mmap file for absorptions - values are read in order
        print(" SOLVING  [%6d, %6d[ OUT OF %6d" % (a, b, CELLS))
        print("READING FP_ABSORBED: ", FP_ABSORBED)
        tmp[0:((b-a)*NFREQ)] =  np.fromfile(FP_ABSORBED, np.float32, (b-a)*NFREQ)
        cl.enqueue_copy(commands, ABS_buf, tmp)
        kernel_T(commands, [GLOBAL,], [LOCAL,], a, kE, oplgkE, Emin, NE, FREQ_buf, TTT_buf, ABS_buf, T_buf)
    cl.enqueue_copy(commands, TNEW, T_buf)    
    a, b, c  = np.percentile(TNEW, (10.0, 50.0, 90.0))
    print("===== Solve temperatures: %.2f seconds =>  %10.3e %10.3e %10.3e =====" % (time.time()-t0, a,b,c))
    FP_ABSORBED.close()
    
    np.asarray(TNEW, np.float32).tofile(SHAREDIR+'TNEW.bin')
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
    ONFREQ   =  len(OFREQ)
    # Solve emission one frequency at a time, all cells on a single call
    print("===== Solve emitted for ONFREQ = %d frequencies =====" % ONFREQ)
    t0       =  time.time()
    if (USE_MMAP):
        np.asarray([CELLS, ONFREQ], np.int32).tofile(f_emitted)
        EMITTED      =  np.memmap(f_emitted, dtype='float32', mode='r+', offset=8, shape=(CELLS,ONFREQ))
        EMITTED[:,:] = 0.0
    else:
        EMITTED      = np.zeros((CELLS, ONFREQ), np.float32)        
    # **** BAD ****  ----   update has outer loop over FREQ,  inner over CELLS
    #                storage order has outer loop over CELLS, inner over FREQ 
    for ifreq in MOUT[0]:            # ifreq = selected frequencies, index to full list of frequencies
        oifreq = ifreq-MOUT[0][0]    # index to the set of output frequencies
        kernel_emission(commands, [GLOBAL,], [LOCAL,], FREQ[ifreq], KABS[ifreq], T_buf, EMIT_buf)
        cl.enqueue_copy(commands, TNEW, EMIT_buf)
        commands.finish()
        EMITTED[:, oifreq] = TNEW    # OUT OF ORDER UPDATE --- HOPE FOR SMALL ONFREQ !!!
        ## print("ifreq %3d   ofreq %3d   %10.3e" % (ifreq, oifreq, mean(TNEW)))
    print("===== Solve emitted: %.2f seconds =====" % (time.time()-t0))
    if (USE_MMAP):
        del EMITTED
    else:
        fp = open(f_emitted, 'wb')
        np.asarray([CELLS, ONFREQ], np.int32).tofile(fp)
        EMITTED.tofile(fp)  #  B(T)*kappa for a single dust component
        fp.close()
    return ONFREQ
        
        


# file of absorptions should start with three integers
H            =  np.fromfile(sys.argv[2], np.int32, 2)
CELLS, NFREQ = H[0], H[1]
print("Absorption file:  CELLS %d, NFREQ %d, NDUST %d" % (CELLS, NFREQ, NDUST))
FPE = None
    

# Read abundance files... we must have enough memory for that
ABU = np.ones((CELLS, NDUST), np.float32)
for idust in range(NDUST):
    if (len(AFILE[idust])>1): # we have a file for the abundance of the current dust species
        ABU[:,idust] = np.fromfile(AFILE[idust], np.float32, CELLS)
    

# Initialise OpenCL to split the absorptions
context, queue, mf = opencl_init(GPU=0)
source      =  open(HOMEDIR+"starformation/SOC/kernel_A2E_MABU_aux.c").read()
OPTS        =  '-D NFREQ=%d -D NDUST=%d' % (NFREQ, NDUST)
program     =  cl.Program(context, source).build(OPTS)
Split       =  program.split_absorbed
Split.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None, None])
BATCH       =  32768
GLOBAL, LOCAL = BATCH, 16
ABS_IN_buf  =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH*NFREQ)
ABS_OUT_buf =  cl.Buffer(context, mf.WRITE_ONLY, 4*BATCH*NFREQ)
RABS_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=RABS) # RABS[NFREQ, NDUST]
ABU_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH*NDUST)                  # ABU[CELLS, NDUST]
        
# loop over dust populations
ONFREQ = -1.0
for IDUST in range(NDUST):
    t0 = time.time()
    # write a new file for absorptions, the photons absorbed by this dust component
    fp1 = open(SHAREDIR+'tmp.absorbed', 'wb')
    H.tofile(fp1)  # header = CELLS, NFREQ
    fp_absorbed  = open(sys.argv[2], 'rb')
    CELLS, NFREQ = np.fromfile(fp_absorbed, np.int32, 2)        
    print("DUST %d/%d -- CELLS %d, NFREQ %d" % (1+IDUST, NDUST, CELLS, NFREQ))
    print("===== Split absorbed =====")
    if (0):
        # loop over cells
        tmp = np.zeros(NFREQ, np.float32)
        K   = np.zeros(NFREQ, np.float64)
        for ICELL in range(CELLS):
            ## K     =  matmul(ABU[ICELL, :], RABST)  # sum over idust ->  [NFREQ]
            ## tmp   =  (RABS[:,IDUST]/K) *  np.fromfile(fp_absorbed, float32, NFREQ)
            tmp   =  (RABS[:,IDUST]/(np.matmul(ABU[ICELL, :], RABST))) *  np.fromfile(fp_absorbed, np.float32, NFREQ)
            np.asarray(tmp, np.float32).tofile(fp1)   # NFREQ values
    else:
        # Use OpenCL to split the absorptions => absorptions by the current dust species
        #  RABS[ifreq, idust]
        #  ABU[icell, idust]
        #    absorbed[icell] =  absorbed[icell] * RABS[ifreq,idust] / K
        #    K               =  sum[   ABU[icell, idust] * RABS[ifreq, idust]  ]
        #  process BATCH cells at a time
        tmp   = zeros((BATCH, NFREQ), float32)
        a = 0
        while(a<CELLS):
            b           =  min(CELLS, a+BATCH)                             # cells [a,b[
            print(a,b)
            tmp[0:(b-a),:]  =  fromfile(fp_absorbed, np.float32, (b-a)*NFREQ).reshape((b-a),NFREQ)   # tmp[icell, ifreq]
            cl.enqueue_copy(queue, ABS_IN_buf, tmp)                        # ABS_IN[batch, nfreq]
            cl.enqueue_copy(queue, ABU_buf, ABU[a:b, :])                   # ABU[batch, ndust]
            Split(queue, [GLOBAL,], [LOCAL,], IDUST, b-a, RABS_buf, ABU_buf, ABS_IN_buf, ABS_OUT_buf)
            cl.enqueue_copy(queue, tmp[0:(b-a), :], ABS_OUT_buf)           # tmp[batch, nfreq]
            tmp[0:(b-a),:].tofile(fp1)
            a += BATCH
    # --------------------------------------------------------------------------------
    fp_absorbed.close()
    fp1.close() # 8 + 4*CELLS*NFREQ bytes
    print("===== Split absorbed: %.2f seconds =====" % (time.time()-t0))
    # Call A2E_pyCL to calculate emissions for the current dust component
    ONFREQ = NFREQ
    t0 = time.time()
    if (EQDUST[IDUST]):
        # Equilibrium dust, also calculating emission to SHAREDIR/tmp.emitted
        #  MAY INCLUDE ONLY FREQUENCIES [UM_MIN, UM_MAX]
        ONFREQ = SolveEquilibriumDust(DUST[IDUST], SHAREDIR+'tmp.absorbed', SHAREDIR+'tmp.emitted', UM_MIN, UM_MAX, GPU)
    else:
        # Stochastically heated -- produces file with ALL FREQUENCIES !!
        if (USELIB):    # solve using existing library
            print("================================================================================")
            print('A2E_LIB.py %s.solver %s.lib freq.dat lfreq.dat %stmp.absorbed %stmp.emitted %s %s' % \
            (DUST[IDUST], DUST[IDUST], SHAREDIR, SHAREDIR, ["", "GPU"][GPU], OFREQ))
            print("================================================================================")
            os.system('A2E_LIB.py %s.solver %s.lib freq.dat lfreq.dat %stmp.absorbed %stmp.emitted %s %s' % \
            (DUST[IDUST], DUST[IDUST], SHAREDIR, SHAREDIR, ["", "GPU"][GPU], OFREQ))
        elif (MAKELIB): # solve every cell and also build the library
            # Note --- file names freq.dat and lfreq.dat hardcoded !!!
            print("--------------------------------------------------------------------------------")
            print('A2E_LIB.py %s.solver %s.lib freq.dat lfreq.dat %stmp.absorbed %stmp.emitted %s makelib' % \
            (DUST[IDUST], DUST[IDUST], SHAREDIR, SHAREDIR, ["", "GPU"][GPU]))
            print("--------------------------------------------------------------------------------")
            os.system('A2E_LIB.py %s.solver %s.lib freq.dat lfreq.dat %stmp.absorbed %stmp.emitted %s makelib' % \
            (DUST[IDUST], DUST[IDUST], SHAREDIR, SHAREDIR, ["", "GPU"][GPU]))            
        else:           # just solve using A2E.py
            print("================================================================================")
            print('    A2E.py  %s.solver %stmp.absorbed %stmp.emitted  %d' % (DUST[IDUST], SHAREDIR, SHAREDIR, GPU))
            print("================================================================================")
            os.system('A2E.py  %s.solver %stmp.absorbed %stmp.emitted  %d' % (DUST[IDUST], SHAREDIR, SHAREDIR, GPU))
        if (ONFREQ!=NFREQ):  # so far only equilibrium dust files can have ONFREQ<NFREQ !!
            print("A2E_MABU.py --- stochastically heated grains with equilibrium grains")
            print("                different number of frequencies... fix A2E.py to use a2e_wavelength parameter??")
            sys.exit()            
        ### os.system('cp %s %d.emitted' % (SHAREDIR+'tmp.emitted', IDUST))            
    # read the emissions and add to FPE
    if (IDUST==0): # FPE opened only here, once we know the number of output frequencies
        if (USE_MMAP):
            fp  =  open(sys.argv[3], 'wb')
            np.asarray([CELLS, ONFREQ], np.int32).tofile(fp)
            fp.close()
            FPE =  np.memmap(sys.argv[3], dtype='float32', mode='r+', shape=(CELLS, ONFREQ), offset=8)
            FPE[:,:] = 0.0
        else:
            FPE = np.zeros((CELLS, ONFREQ), np.float32)
    print("===== Add emitted to sum file: %s =====" % sys.argv[3])
    t0 = time.time()
    fp2 = open(SHAREDIR+'tmp.emitted', 'rb')
    np.fromfile(fp2, np.int32, 2)  # get rid of the header (CELLS, NFREQ)
    # **** GOOD **** ---- single dust emission added to the total in the file order
    if (1):
        for ICELL in range(CELLS):
            # file = B(T)*KABS for a single dust component, total is sum of B(T)*KABS*ABU
            FPE[ICELL,:] += np.fromfile(fp2, np.float32, ONFREQ) * ABU[ICELL, IDUST]
    else:
        a = 0
        while(a<CELLS):
            b = min(a+1024, CELLS)
            FPE[a:b,:] += np.fromfile(fp2, np.float32, (b-a)*ONFREQ).reshape(b-a, ONFREQ) * \
                          ABU[a:b, IDUST].reshape(b-a,1)
            a += 1024
    fp2.close()
    print("===== Add emitted to sum file: %.2f seconds =====" % (time.time()-t0))
    # os.system('rm /dev/shm/tmp.absorbed /dev/shm/tmp.emitted')

    
if (USE_MMAP):    
    del FPE  # file should now contain combined emission from all dust components
else:
    fp  =  open(sys.argv[3], 'wb')
    np.asarray([CELLS, ONFREQ], np.int32).tofile(fp)
    FPE.tofile(fp)
    fp.close()
    

        
