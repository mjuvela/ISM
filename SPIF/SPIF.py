#!/usr/bin/env python
import os, sys

INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)

from SPIF_aux import *

# list of command line parameters with default values
PAR = { '-gpu':'1', '-mcmc': '0', '-samples':'1000', '-burnin':'1000', '-thin':'20', \
        '-platforms': '[0,1,2,3]', '-iter':'1', '-adapt':'0', '-plot':'0', \
        '-ini': 'test.ini', '-preopt':'0', '-mc':'0', '-method':'0',
        '-polak':'0', '-mciter':'1', '-fullsave':'0', '-nan':'0'  }

if (len(sys.argv)<2):
    print()
    print("Usage:")
    print("  SPIF.py [optional flags] -ini <ini-file>")
    print("Flags:")
    print("   -gpu #        for #>0, use GPU instead of cpu (default %s)" % PAR['-gpu'])
    print("   -platforms s  list of OpenCL platforms to choose from (default %s)" % PAR['-platforms'])
    print("   -iter #       number of iterations (not for MCMC)  (default %s)" % PAR['-iter'])
    print("   -method #     optimiser, 0=Simplex, 1=gradient descent, 2=conjugate gradient (default %s)" % PAR['-method'])
    print("   -polak #      for congugate gradient method, use Polak-Ribiere (default %s)" % PAR['-polak'])
    print("   -plot #       1 = show graphics window (default %s)" % PAR['-plot'])
    print("") 
    print("   -mc #         Monte Carlo simulation based on observational noise (default %s)" % PAR['-mc'])
    print("   -mciter #     iterate # times each fit with different initial values (default %s)" % PAR['-mciter'])
    print("   -fullsave #   for #=1, save also Monte Carlo / MCMC samples (default %s)" % PAR['-fullsave'])
    print("")
    print("   -mcmc #       use MCMC (1=Metropolis, 2=Hamiltonian, 3=RAM) (default %s)" % PAR['-mcmc'])
    print("   -samples #    number of MCMC samples (default %s)" % PAR['-samples'])
    print("   -burnin #     length of burn-in phase in MCMC steps (default %s)" % PAR['-burnin'])
    print("   -thin #       register every #:th sample (default %s)" % PAR['-thin'])
    print("   -preopt #     1 = optimisation as initial state for MCMC (default %s)" % PAR['-preopt'])
    print("   -nan #        0 = do not check for Nan, 1 = ignore channels with NaN values")
    print("")
    print("Examples of possible lines in the ini-file (lines are not mutually consistent)")
    print("-------------------------------")
    print("  fits1     =  file1.fits                        # input file for 1. spectra")
    print("  dfits1    =  file1_error.fits                  # error map for 1. spectra")
    print("  fits2     =  file2.fits                        # input file for 2. spectra")
    print("  dfits2    =  file1_error.fits                  # error map for 2. spectra") 
    print("  y1        =  GAUSS(v1,x[0],x[1],x[2])              # 1. model: single Gaussian")
    print("  y1        =  HFS(v1,x[0],x[1],x[2],x[3],VHF1,IHF1) # 1. model: hyperfine structure")    
    print("  hfs1      =  n2h+.hfs                          # data for the 1. spectrum HFS model")
    print("  aux       =  aux.fits                          # auxiliary FITS image")
    print("  y2        =  GAUSS(v1,x[0],aux,x[2])           # 2. model: Gaussian with velocity from aux")
    print("  init      =  y1:5.0 y1:vmean y1:fwhm y2:vmax   # set initial values for all x[:]")
    print("  init      =  init.fits                         # read initial values from FITS file")
    print("  penalty   =  (x[3]<2) ? ((2-x[3])/0.1) : 0.0   # penalties for x[:]")
    print("  prior     =  NORMAL(x[1], 1.0)                 # prior for a model parameter")
    print("  delta     =  r:0.3 a:0.5 a:0.3  r:0.3 a:0.5 a:0.3  # perturbation in initial values (with -iter)")
    print("  prefix    =  test                              # prefix for output files")
    print("  step      =  0.17068787 0.20831834 0.2675255  0.30431524  # modify MCMC step sizes")
    print("  mask      =  some_2d.fits                      # pixel value <=0 => pixel not fitted")
    print("  threshold =  0.1                               # fit only pixels with Tmax>threshold")
    print("  ")
    print("  -------------------------------")
    print()
    sys.exit()

"""
2025-05-14:
    - FWHM is now always returned as non-negative values
    - added ini-file options MASK and THRESHOLD (to exclude some pixels from the fit)
"""

for k in PAR.keys():
    if (k in sys.argv):
        ind    = sys.argv.index(k)
        PAR[k] = sys.argv[ind+1]
        
GPU        =  int(PAR['-gpu'])
PLF        =  []
for x in PAR['-platforms'].replace('[','').replace(']','').split(','):
    PLF.append(int(x))
POLAK      =  int(PAR['-polak'])        
METHOD     =  int(PAR['-method'])    
USE_MCMC   =  int(PAR['-mcmc'])
MC_ITER    =  int(PAR['-mciter'])
FULLSAVE   =  int(PAR['-fullsave'])
DO_MC      =  int(PAR['-mc'])    
BURNIN     =  int(PAR['-burnin'])
SAMPLES    =  int(PAR['-samples'])
THIN       =  int(PAR['-thin'])
NITER      =  int(PAR['-iter'])
ADAPT      =  int(PAR['-adapt'])
PLOT       =  int(PAR['-plot'])
PARFILE    =  PAR['-ini']
PREOPT     =  int(PAR['-preopt'])
TBG        =  2.73    
STEP       =  ones(100, float32)
IGNORE_NAN =  int(PAR['-nan'])

FITS1, FITS2, dFITS1, dFITS2, INI, MASK = None, None, None, None, [], None
Y1, Y2, hfs1, hfs2, PEN, PRIOR, AUX, DELTA = "", "", "", "", "", "", [], []
THRESHOLD = -999.0
PREFIX = 'res'
for l in open(PARFILE).readlines():
    if (l[0:1]=='#'): continue
    if (l.find('#')>0):
        s = l[0:l.find('#')].split()
    else:
        s = l.split()
    if (len(s)<2): continue
    if (s[0]=='#'): continue
    if (s[0].lower()=='fits1'):    FITS1  = s[2]
    if (s[0].lower()=='dfits1'):   dFITS1 = s[2]
    if (s[0].lower()=='fits2'):    FITS2  = s[2]
    if (s[0].lower()=='dfits2'):   dFITS2 = s[2]
    if (s[0]=='y1'):               Y1     = l[0:-1]
    if (s[0]=='y2'):               Y2     = l[0:-1]
    if (s[0].lower()=='hfs1'):     hfs1   = s[2]
    if (s[1].lower()=='hfs2'):     hfs2   = s[2]
    if (s[0].lower()=='init'):     INI    = s[2:]
    if (s[0].lower()=='penalty'):
        ind = l.find('=')
        if (ind>0):  PEN = l[(ind+1):-1]
    if (s[0].lower().find('prior')>=0):
        ind = l.find('=')
        if (ind>0):  PRIOR = l[(ind+1):-1]
    if (s[0].lower()=='aux'):      AUX.append(s[2])
    if (s[0].lower()=='prefix'):   PREFIX = s[2]
    if (s[0].lower()=='delta'):    DELTA  = s[2:]
    if (s[0].lower()=='step'):
        for i in range(2, len(s)): STEP[i-2] = float(s[i])
    if (s[0].lower()=='mask'):      
        tmp = pyfits.open(s[2])
        MASK = nonzero(tmp[0].data>0.0)  # fit only these pixels
    if (s[0].lower()=='threshold'):      
        THRESHOLD = float(s[2])
            
    
NHF1, NHF2, HFK1, HFK2 = 0, 0, 0.0, 0.0
if (len(hfs1)>0):   
    HFK1 =  4.799243348e-11 * float(open(hfs1).readline().split()[0])
    d    =  loadtxt(hfs1, skiprows=1)
    VHF1 =  asarray(d[:,0], float32)  # velocities of hyperfine components
    IHF1 =  asarray(d[:,1], float32)  # relative opacities of the components
    NHF1 =  len(VHF1)                 # number of components
if (len(hfs2)>0):   
    HFK2 =  H_K * float(open(hfs2).readline().split()[0])
    d    =  loadtxt(hfs2, skiprows=1)
    VHF2 =  asarray(d[:,0], float32)
    IHF2 =  asarray(d[:,1], float32)
    NHF2 =  len(VHF2)


# replace HFS() with HFS1() or HFS2() and add parameters VHF and IHF
if (Y1.find('HFSX(')>=0):
    Y1 = Y1.replace('HFSX(', 'HFSX1(').replace(')', ', VHF1, IHF1)')
else:
    if (Y1.find('HFS(')>=0):
        Y1 = Y1.replace('HFS(', 'HFS1(').replace(')', ', VHF1, IHF1)')
    
if (Y2.find('HFS(')>=0):
    Y2 = Y2.replace('HFS(', 'HFS2(').replace(')', ', VHF2, IHF2)')

    
    
# Y1 may also have a reference to auxiliary FITS file, such as  GAUSS(x1, p[0], aux, p[1])
PENALTIES = len(PEN)>0
PRIORS    = len(PRIOR)>0
src = open(INSTALL_DIR+"/kernel_SPIF.c").read().replace('@y1',Y1).replace('@y2',Y2)
if (PENALTIES>0): src = src.replace('@pen',   PEN)
if (PRIORS>0):    src = src.replace('@prior', PRIOR)

# find the parameter indices for FWHM parameters (at the end absolute value is returned)
FWHM_INDICES = []
for s in Y1.split(')'):
    if (s.find('GAUSS')>=0):
        #  GAUSS(v, x[0], x[1], x[2]
        tmp = s.split(',')[-1].replace(']','').split('[')[1]
        FWHM_INDICES.append(int(tmp))
    elif (s.find('HFS')>=0):
        # HFS(x1,p[0],p[1],p[2],p[3],VHF1,IHF1
        tmp = s.split(',')[3].replace(']','').split('[')[1]
        FWHM_INDICES.append(int(tmp))
print("FWHM_INDICES = ", FWHM_INDICES)

if (USE_MCMC==2):
    src_grad = WriteGradient(Y1, Y2)
    src = src.replace('//@GRAD', src_grad)
    
fp = open('/dev/shm/src.c', 'w')
fp.write(src)
fp.close()
# sys.exit()

# print(src)
# sys.exit()

print(Y1)
OK = zeros(20, int32)
for i in range(20):
    if (('[%d]' % i) in Y1): OK[i] = 1
    if (('[%d]' % i) in Y2): OK[i] = 1
NP = len(nonzero(OK>0)[0])
if (min(OK[0:NP])<1):
    print("One must use consecutive elements from p[] !!"), sys.exit()
if (len(DELTA)<1):
    for i in range(NP): DELTA.append('r:0.3')   # perturbation = relative 0.3
print(DELTA)    
STEP = asarray(STEP[0:NP].copy(), float32)
    
def SHOW():
    show(block=True)
    sys.exit()
    
    
# Read input FITS files
print("GOPT.py reading file ", FITS1)
F          =  pyfits.open(FITS1)
M1, NY, NX =  F[0].data.shape 
if (THRESHOLD>0.0): # overrides previous MASK
    MASK   =  np.nonzero(np.max(F[0].data, axis=0)>THRESHOLD)
    
    
if (0): # test NaNs
    F[0].data[40:50, 30:40, 30:40] = NaN
    
N          =  NY*NX
V1 =  asarray(F[0].header['CRVAL3']+(arange(1,M1+1)-F[0].header['CRPIX3'])*F[0].header['CDELT3'], float32)
Y1 =  asarray(F[0].data.copy(), float32)        # Y1[M1, NY, NX]
if (F[0].header['CUNIT3'].strip()=='m/s'): V1 /= 1000.0
if (MASK):
    Y1 = Y1[:, MASK[0], MASK[1]]
    N  = len(MASK[0])
Y1 =  Y1.reshape(M1, N).transpose().copy()      # Y1[N, M1]
# error dFITS1 is either file name or floating point constant
if (dFITS1==None): 
    print("One should specify dfits1! Continuing with dT=1.0.")
    dY1 =  ones(N, float32)
else:
    try:
        F   =  pyfits.open(dFITS1)
        dY1 =  asarray(F[0].data.copy(), float32)
        if (MASK): dY1 =  dY1[MASK]
        dY1 =  dY1.reshape(N).copy()      # Y1[N]
    except:
        dY1 =  float(dFITS1)*ones(N, float32)
        
TWIN = 0    
M2   = 0
if (FITS2!=None):
    TWIN =  1
    F    =  pyfits.open(FITS2)
    M2, ny, nx =  F[0].data.shape
    if ((ny!=NY)|(nx!=NX)): print("Inconsisten dimensions in fits2"), sys.exit()
    V2   =  asarray(F[0].header['CRVAL3']+(arange(1,M1+1)-F[0].header['CRPIX3'])*F[0].header['CDELT3'], float32)
    Y2   =  asarray(F[0].data.copy(), float32)
    if (MASK):
        Y2 =  Y2[:, MASK[0], MASK[1]]
    Y2   =  Y2.reshape(M2, N).transpose().copy()     # Y2[N, M2]
    if (dFITS2==None): 
        print("One should  specify dfits2 ... using dT=1.0...!")
        dY2 =  ones(N, float32)
    try:
        F    =  pyfits.open(dFITS2)
        dY2  =  asarray(F[0].data.copy(), float32)
        if (MASK): dY2 = dY2[MASK]
        dY2  =  dY2.reshape(N).copy()  # dY2[N, M1]
    except:
        dY2  =  float(dFITS1)*ones(N, float32)

        
        

P = zeros((N, NP), float32)
if (len(INI)>0):
    if (INI[0].find('.fits')>0): # one gave a FITS file for the initial values
        F0 = pyfits.open(INI[0])
        n, ny, nx = F0[0].data.shape
        if (n<NP):
            print("Error in ini %s: not planes for each free paramater (%d<%d)" % (INI[0], n, NP))
            sys.exit(0)
        if ((ny!=NY)|(nx!=NX)):
            print("Error in ini %s: images %d x %d, not %d x %d pixels" % (ny, nx, NY, NX))
            sys.exit()
        for i in range(NP):
            P[:,i] = ravel(F0[0].data[i,:,:])   #  P[NY*NX, NP]
        if (MASK):
            tmp = F0[0].data[:, MASK[0], MASK[1]].copy().reshape(NP, N)
            for i in range(NP): P[:,i] = tmp[i,:]
        del F0        
    else: # INI contains some instructions for setting initial values
        """
        For each free parameter, constant or calculated from y1 or y2, for example
                  y1:tmax      y2:vmax    y1:vmean   y1:fwhm   2.1
                  y1:fwhm*1.3  y1:vmax-1
        
        """
        print("------------------------------------------------------------------------------------------")
        for i in range(len(INI)): # each term in INI is something like "mean-y1"
            s    =  INI[i].split(':')
            tag = ''
            if (len(s)==1):  # one value given?
                try:
                    P[:,i] = float(s[0])
                except:
                    print("initialisation of p[%d] unknown: %s" % (i, s[0]))
                    pass
            else:   # one probably has something like y1:tmax
                if (s[0]=='y1'):    X, Y, tag = V1, Y1, s[1]
                elif (s[0]=='y2'):  X, Y, tag = V2, Y2, s[1]
                else:   
                    print("initialisation should be a constant or y1:... or y2:...: %s ?" % INI[i])
                    continue
                if (tag.lower().find('tmax')>=0):
                    P[:,i] = np.nanmax(Y, axis=1)        # maximum value
                elif (tag.lower().find('vmax')>=0):
                    P[:,i] = X[np.nanargmax(Y, axis=1)]  # velocity of maximum intensity
                elif (tag.lower().find('vmean')>=0):  
                    P[:,i] = nansum(Y*X)/nansum(Y)       # intensity-weighted mean velocity
                elif (tag.lower().find('fwhm')>=0):
                    z  =  np.nansum(Y*X, axis=1)
                    z2 =  np.nansum(Y*X*X, axis=1)
                    w  =  np.nansum(Y, axis=1)
                    P[:,i] = 2.355*sqrt((z2-z*z/w)/w)
                    # there is no guanrantee that this would be positive, if spectrum has
                    # many negative values due to noise
                    m = nonzero(~isfinite(P[:,i]))
                    P[m[0],i] = 1.0   # fall back to FWHM=1.0?
                # tag may also contain operations *x /x +x -x
                if (tag.find('*')>0):     P[:,i] *=  float(tag[1+tag.find('*')])
                elif (tag.find('/')>0):   P[:,i] /=  float(tag[1+tag.find('/')])
                elif (tag.find('+')>0):   P[:,i] +=  float(tag[1+tag.find('+')])
                elif (tag.find('-')>0):   P[:,i] -=  float(tag[1+tag.find('-')])
            ###
            a,b,c,d,e = percentile(P[:,i], (0, 10, 50, 90, 100))
            print("INI[%2d] %15s  ->  %11.3e %10.3e %11.3e %11.3e %11.3e" % (i, INI[i],a,b,c,d,e))
        print("------------------------------------------------------------------------------------------")
    
            
            

if (0):
    clf()
    for i in range(NP):
        subplot(2,2,1+i)
        imshow(P[:,i].reshape(NY, NX))
        colorbar()
    SHOW()
    
# ==========================================================================================

platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=PLF, verbose=True)
if (device==[]): print("No OpenCL device found - exiting."), sys.exit()
print("device", device)

print(type(V1), V1.shape)
V1_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V1)        # V1[M1]
Y1_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Y1)        # Y1[N, NCHN]
dY1_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dY1)       # dY1[N]
if (M2>0):
    V2_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V2)    # V2[M1]
    Y2_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Y2)    # Y2[N, NCHN]
    dY2_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dY2)   # dY2[N]
else:
    V2_buf  =  cl.Buffer(context, mf.READ_ONLY,  4)    # V2[M1]
    Y2_buf  =  cl.Buffer(context, mf.READ_ONLY,  4)    # Y2[N, NCHN]
    dY2_buf =  cl.Buffer(context, mf.READ_ONLY,  4)    # dY2[N, NCHN]
    
P_buf       =  cl.Buffer(context, mf.READ_WRITE, 4*N*NP)   # P[N, NP]
C2_buf      =  cl.Buffer(context, mf.WRITE_ONLY, 4*N)      # C2[N]   ... chi2 values
STEP_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*NP)     # STEP[NP], MCMC step size modifier


if (NHF1>0): # for hyperfine fits, one needs HFK = h*f/k
    IHF1_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=IHF1)
    VHF1_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=VHF1)
else:
    IHF1_buf = cl.Buffer(context, mf.READ_ONLY, 4)
    VHF1_buf = cl.Buffer(context, mf.READ_ONLY, 4)

if (NHF2>0):
    IHF2_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=IHF2)
    VHF2_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=VHF2)
else:
    IHF2_buf = cl.Buffer(context, mf.READ_ONLY, 4)
    VHF2_buf = cl.Buffer(context, mf.READ_ONLY, 4)

    
if (USE_MCMC>0):
    RES_buf =  cl.Buffer(context, mf.READ_WRITE, 4*N*SAMPLES*(NP+1)) # RES[N, SAMPLES, NP+1]
else:
    RES_buf =  cl.Buffer(context, mf.READ_WRITE, 4)
    
            


AUX_DATA = [] 
N_AUX    = len(AUX)  # this many Aux images
AUX_DATA = zeros((NY, NX, N_AUX), float32)  # N_AUX auxiliary images
for k in range(N_AUX):
    s       =  AUX[k].split(':')
    if (len(s)==1):
        AUX_DATA[:,:,k]  =  asarray(pyfits.open(s[0])[0].data, float32)  # [NY, NX, N_AUX]
    else:
        j   =  int(s[1])
        AUX_DATA[:,:,k]  =  asarray(pyfits.open(s[0])[0].data[j,:,:], float32)
if (N_AUX>0):
    if (MASK):  AUX_DATA = AUX_DATA[MASK[0], MASK[1], :].copy()
    AUX_DATA = ravel(AUX_DATA)
    AUX_buf  = cl.Buffer(context, mf.READ_ONLY, 4*N_AUX*N)
    cl.enqueue_copy(queue, AUX_buf, AUX_DATA)
else:
    AUX_buf =  cl.Buffer(context, mf.READ_ONLY, 4*NP)  # dummy small buffer


LOCAL  =  [1, 32][GPU>0]
    
OPT    =  "-D N=%d -D M1=%d -D M2=%d -D NP=%d -D TWIN=%d " % (N, M1, M2, NP, TWIN)
OPT   +=  "-D NHF1=%d -D NHF2=%d -D TBG=%.4ff " % (NHF1, NHF2, TBG)
OPT   +=  "-D HFK1=%.4ef -D HFK2=%.4ef " % (HFK1, HFK2)
OPT   +=  "-D BURNIN=%d -D SAMPLES=%d -D THIN=%d " % (BURNIN, SAMPLES, THIN)
OPT   +=  "-D PENALTIES=%d -D PRIORS=%d -D N_AUX=%d -D ADAPT=%d " % (PENALTIES, PRIORS, N_AUX, ADAPT)
OPT   +=  "-I %s -D LOCAL=%d -D POLAK=%d " % (INSTALL_DIR, LOCAL, POLAK)
OPT   +=  "-D USE_MCMC=%d -D IGNORE_NAN=%d" % (USE_MCMC, IGNORE_NAN)
print(OPT)


program   =  cl.Program(context, src).build(OPT)

if (METHOD==0):
    BF = program.Simplex
elif (METHOD==1):
    BF = program.BF    
else:
    BF = program.CG
    
    
    
#                         V1    Y1    dY1   V2    Y2    dY2   VHF1  IHF1  VHF2  IHF2  P     C2
#BF.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, None, None, None, None])

# Choose MCMC routine based on the value oof USE_MCMC (= option -mcmc #)
if (USE_MCMC==1):
    MCMC = program.MCMC
    #                           seed        V1    YY1   dYY1  V2    YY2   dYY2  VHF1  IHF1  VHF2  IHF2 
    MCMC.set_scalar_arg_dtypes([np.float32, None, None, None, None, None, None, None, None, None, None,
    #                           P     RES   AUX   STEP
                                None, None, None, None])
elif (USE_MCMC==2):
    MCMC = program.MCMC_RAM
    #                           seed        V1    YY1   dYY1  V2    YY2   dYY2  VHF1  IHF1  VHF2  IHF2 
    MCMC.set_scalar_arg_dtypes([np.float32, None, None, None, None, None, None, None, None, None, None,
    #                           P     RES   AUX
                                None, None, None, None])
elif (USE_MCMC==3):
    MCMC = program.HMCMC
    #                           seed        V1    YY1   dYY1  V2    YY2   dYY2  VHF1  IHF1  VHF2  IHF2 
    MCMC.set_scalar_arg_dtypes([np.float32, None, None, None, None, None, None, None, None, None, None,
    #                           P     RES   AUX   STEP
                                None, None, None, None])
                            
GLOBAL    = (N//32+1)*32

cl.enqueue_copy(queue, P_buf, P)
C2 = zeros(N, float32)
np.random.seed(123)

if (NITER>1): 
    P0    = P.copy()   # keep a copy of the original paramaters
    BEST  = 1e30*ones((N,NP+1), float32)



# Maximum likelihood solution. Skipped only for MCMC run with PREOPT=0
if (not((USE_MCMC>0)&(PREOPT==0))):
    t0 = time.time()    
    for ITER in range(NITER):
        print("ITER %d/%d" % (ITER+1, NITER))
        if (ITER>0):       # perturb original input parameters
            for i in range(NP):
                d = DELTA[i].split(':')
                z = float(d[1])
                if (d[0]=='r'):  P[:,i] = P0[:,i] * (1.0+z*randn(N))
                else:            P[:,i] = P0[:,i] + z*randn(N)
            cl.enqueue_copy(queue, P_buf, P)
        BF(queue, [GLOBAL,], [LOCAL,],   
        V1_buf, Y1_buf, dY1_buf,    V2_buf, Y2_buf, dY2_buf,
        VHF1_buf, IHF1_buf,    VHF2_buf, IHF2_buf,    P_buf, C2_buf, AUX_buf)
        cl.enqueue_copy(queue, P,   P_buf)    # P_buf[N, NP]  .... P[N,NP]
        cl.enqueue_copy(queue, C2,  C2_buf)
        if (NITER>1):     # iterations done
            a = max(BEST[:,NP])/M1
            b = max(C2)/M1
            print(" max chi2    was %7.3f    new %7.3f " % (a, b))
            if (ITER==0):
                BEST[:,0:NP] = P[:,:].copy()
                BEST[:,NP]   = C2[:].copy()
            else:
                m               = nonzero(C2<BEST[:,NP]) # improved fits
                BEST[m[0],0:NP] = P[m[0],0:NP]
                BEST[m[0],NP]   = C2[m[0]]
                print("   ..... %d improved" % len(m[0]))
            print("    ==>  %7.3f " % (max(BEST[:,NP])/M1))
            if (ITER==(NITER-1)):  # copy BEST results back to (P, C2)
                P  = BEST[:,0:NP].copy()
                C2 = BEST[:,NP]*1.0
    # save the result... even when one continues with MCMC
    F[0].data = zeros((NP+1, NY, NX), float32)
    for i in range(NP):
        if (MASK):
            F[0].data[i,MASK[0],MASK[1]] = P[:,i]
        else:
            F[0].data[i,:,:] = P[:,i].reshape(NY, NX)
    if (MASK):
        F[0].data[NP,MASK[0],MASK[1]] = C2
    else:
        F[0].data[NP,:,:] = C2.reshape(NY,NX)   
    for i in FWHM_INDICES: F[0].data[i,:,:] = np.abs(F[0].data[i,:,:])
    F.writeto(f'{PREFIX}_res.fits', overwrite=True)
    ##
    t0 = time.time()-t0
    print("ML run in total %.2f seconds, %.2e spectra/second" % (t0, N/t0))


# Monte Carlo simulation    
if (DO_MC):
    # Using current P0 as initial values, do DO_MC Monte Carlo samples 
    # by adding noise to the input spectra. Use the samples to estimate
    # 1d errors for each of the free parameters
    P0      =  zeros((N, NP), float32)
    P0[:,:] =  1.0*P[:,:]   # should be the parameters estimated above
    P       =  zeros((N, NP), float32)  # otherwise "array not contiguous"?
    ###
    RES     = zeros((N, NP, DO_MC), float32)
    for ITER in range(DO_MC):
        print(" MC %d" % ITER)
        tmp = Y1.copy()
        for i in range(N): tmp[i,:] +=  dY1[i]*randn(M1)
        cl.enqueue_copy(queue, Y1_buf, tmp)
        if (M2>0):  # random realisation of the observations
            tmp = Y2.copy()
            for i in range(N): tmp[i,:] +=  dY2[i]*randn(M2)
            cl.enqueue_copy(queue, Y2_buf, tmp)
        ###
        cl.enqueue_copy(queue, P_buf, P0) # using the parameters of pre-MC fit as initial values
        BF(queue, [GLOBAL,], [LOCAL,],   
        V1_buf, Y1_buf, dY1_buf,    V2_buf, Y2_buf, dY2_buf,
        VHF1_buf, IHF1_buf,    VHF2_buf, IHF2_buf,    P_buf, C2_buf, AUX_buf)
        ###
        cl.enqueue_copy(queue, P, P_buf)
        if (MC_ITER>1): # repeat the fit with perturbed initial values
            BEST = zeros((N,NP+1), float32)
            BEST[:,0:NP] = P
            BEST[:,NP]   = C2
            for III in range(1,MC_ITER):
                print("   ... realisation -> fit repeated %d" % III)
                for i in range(NP):
                    d = DELTA[i].split(':')
                    z = float(d[1])
                    if (d[0]=='r'):  P[:,i] = P0[:,i] * (1.0+z*randn(N))
                    else:            P[:,i] = P0[:,i] + z*randn(N)
                cl.enqueue_copy(queue, P_buf, P)
                #
                BF(queue, [GLOBAL,], [LOCAL,],   
                V1_buf, Y1_buf, dY1_buf,    V2_buf, Y2_buf, dY2_buf,
                VHF1_buf, IHF1_buf,    VHF2_buf, IHF2_buf,    P_buf, C2_buf, AUX_buf)
                #
                cl.enqueue_copy(queue, P,   P_buf)    # P_buf[N, NP]  .... P[N,NP]
                cl.enqueue_copy(queue, C2,  C2_buf)
                m = nonzero(C2<BEST[:,NP])   # copy improved fits to BEST
                BEST[m[0],0:NP] = P[m[0],:]
                BEST[m[0], NP]  = C2[m[0]]
            ###
            P[:,:] = BEST[:,0:NP]  # copy best fits back to P
        #####
        RES[:,:,ITER]  =  1.0*P    #  RES[N, NP,DO_MC]
    # make FWHM non-negative
    for i in FWHM_INDICES: RES[:,i,:] = np.abs(RES[:,i,:]) 
    # Save results to a fits file
    if (FULLSAVE): # save MC samples ... from RES[N, NP, DO_MC]
        G  =  pyfits.open(FITS1)
        G[0].data = zero((DO_MC, NP, NY, NX), float32)
        if (MASK):
            for ido in range(DO_MC):
                for ipar in range(NP):
                    G[0].data[ido, ipar, MASK[0], MASK[1]] = RES[:, ipar, ido]
        else:
            for ipar in range(NP):
                G[0].data[:, ipar, :, :] = transpose(RES[:, ipar,:]).reshape(DO_MC, NY, NX)
        G.writeto(f'{PREFIX}_mc_samples.fits', overwrite=True)
    G    =  pyfits.open(FITS1)
    G[0].data = zeros((NP, NY, NX), float32)
    if (MASK):
        RES  =  np.std(RES, axis=2).reshape(N, NP)   #  RES[N, NP, DO_MC] -> std  RES[N, NP]
        for ipar in range(NP): 
            G[0].data[ipar, MASK[0], MASK[1]] = RES[:,ipar]
    else:
        RES  =  np.std(RES, axis=2).reshape(NY, NX, NP)   # RES[N, NP, DO_MC] -> std RES[NY, NX, NP]
        for ipar in range(NP): 
            G[0].data[ipar,:,:] = RES[:,:,ipar]           #  RES[NY, NX, NP] -> G[NP, NY, NX]
    G.writeto(f'{PREFIX}_mc.fits', overwrite=True)  #  [NP, NY, NX]
    

# Markov chain Monte Carlo
if (USE_MCMC>0):
    # for PREOPT>0, the maxmum likelyhood solution is already in P, not yet in P_buf
    t0   = time.time()
    RES  = zeros((N,SAMPLES,NP+1), float32)
    SEED = np.random.rand()
    cl.enqueue_copy(queue, STEP_buf, STEP)            
    cl.enqueue_copy(queue, P_buf, P)
    print("MCMC ...")
    MCMC(queue, [GLOBAL,], [LOCAL,], SEED, 
         V1_buf, Y1_buf, dY1_buf,    V2_buf, Y2_buf, dY2_buf,
         VHF1_buf, IHF1_buf,    VHF2_buf, IHF2_buf,    P_buf, RES_buf, AUX_buf, STEP_buf)
    queue.finish()
    print("MCMC ... done")
    # put mean values of RES[N, SAMPLES, 0:NP] to to P[:, 0:NP]
    cl.enqueue_copy(queue, RES, RES_buf)    
    for i in FWHM_INDICES: RES[:,:,i] = np.abs(RES[:,:,i])
    for i in range(NP):
        P[:,i] = np.mean(RES[:,:,i], axis=1)   # replace P with the mean over samples
    C2[:] = np.mean(RES[:,:,NP], axis=1).reshape(N)  # last is the chi2 value
    if (FULLSAVE):
        G         = pyfits.open(FITS1)
        if (MASK):
            G[0].data = zeros((SAMPLES, NP, NY, NX), float32)
            for isam in range(SAMPLES):
                for ipar in range(NP):
                    G[0].data[isam, ipar, :, :] = RES[:, isam, ipar]
        else:
            G[0].data = zeros((SAMPLES, NP, NY*NX), float32)
            for ipix in range(N):
                for ipar in range(NP):
                    G[0].data[:, ipar, ipix] = RES[ipix, :, ipar]
            G[0].data.shape = (SAMPLES, NP, NY, NX)
        G.writeto(f'{PREFIX}_mcmc_samples.fits', overwrite=True)   # all samples
    # save another file with just the mean and the std of each parameter
    G         =  pyfits.open(FITS1)  #   [M, NY, NX]
    if (MASK):
        G[0].data =  zeros((2*NP, NY, NX), float32)
        for ipar in range(NP):
            G[0].data[2*ipar  , MASK[0], MASK[1]] = np.mean(RES[:,:,ipar], axis=1)  # RES[N, SAMPLES, NP+1]
            G[0].data[2*ipar+1, MASK[0], MASK[1]] = np.std( RES[:,:,ipar], axis=1)
    else:
        G[0].data =  zeros((2*NP, NY*NX), float32)
        for i in range(NP):
            G[0].data[2*i  ,:] = np.mean(RES[:,:,i], axis=1)  # RES[N, SAMPLES, NP+1]
            G[0].data[2*i+1,:] = np.std( RES[:,:,i], axis=1)
        G[0].data.shape = (2*NP, NY, NX)
    G.verify('fix')
    G.writeto(f'{PREFIX}_mcmc.fits', overwrite=True)   # all samples    
    ###
    t0 = time.time()-t0
    print("MCMC run in total %.2f seconds, %.2e spectra/second" % (t0, N/t0))

    


if (PLOT):
    prefix='%s_NP%d_MCMC%d_%s' % (FITS1.replace('.fits',''), NP, USE_MCMC, PARFILE.replace('.ini',''))
    close(1)
    figure(1, figsize=(10,8))
    P.shape = (N, NP)
    Q = 4.0*log(2.0)
    if (NHF1<1):
        # fitted Gaussians
        for i in range(10):
            for j in range(10):
                ax = axes([0.0+i*0.1, 0.9-j*0.1, 0.1, 0.1])
                x = int(((i+0.5)/10.0)*NX)
                y = int(((j+0.5)/10.0)*NY)
                ind = x+NX*y
                plot(V1, Y1[ind,:], 'k-', lw=1.2)
                # plot(0.0, dY1[ind], 'rx', ms=6)
                if (NP==2): # assume velocity given in AUX_DATA[0,:,:]
                    y1 = P[ind,0]*exp(-Q*((V1-  AUX_DATA[ind] )/P[ind,1])**2)
                else:
                    y1 = P[ind,0]*exp(-Q*((V1-P[ind,1])/P[ind,2])**2)
                plot(V1, y1, 'm--', lw=0.8)
                yp = 1.0*y1
                if (NP==6): # two Gaussians
                    y2 = P[ind,3]*exp(-Q*((V1-P[ind,4])/P[ind,5])**2)
                    yp += y2
                    plot(V1, y2, 'b:', lw=0.8)
                    plot(V1, y1+y2, 'r--', lw=0.8)
                if (1): # recompute reduced chi2
                    chi2        = sum(((Y1[ind,:]-yp)/dY1[ind])**2)   / (M1-NP)
                    chi2_kernel = F[0].data[NP,y,x] / (M1-NP)
                    text(0.1, 0.8, '%.2f' % chi2_kernel, color='b', transform=ax.transAxes)
                    text(0.6, 0.8, '%.2f' % chi2,        color='r', transform=ax.transAxes)
                setp(ax, xticks=[], yticks=[])

    else:
        # fitted HFS
        def Hfs(x, p):
            # for testing only ... assumes HFS parameters in p[0]...p[3]
            tau = zeros(len(x))
            for j in range(NHF1):
                tau +=  p[3]*IHF1[j]*exp(-Q*((p[1]+VHF1[j]-x)/p[2])**2)
            return (Jfun(p[0], HFK1)-Jfun(TBG, HFK1))*(1-exp(-tau))
        #        
        for i in range(10):
            for j in range(10):
                ax   =  axes([0.0+i*0.1, 0.9-j*0.1, 0.1, 0.1])
                x    =  int(((i+0.5)/10.0)*NX)
                y    =  int(((j+0.5)/10.0)*NY)
                ind  =  x+NX*y
                plot(V1, Y1[ind,:], 'k-', lw=1.2)
                y1   =  Hfs(V1, P[ind,:])
                plot(V1, y1, 'm--', lw=0.8)
                text(0.1, 0.7, '%d' % ind, transform=ax.transAxes, size=7)
                setp(ax, xticks=[], yticks=[])
    if (USE_MCMC>0):        
        savefig('./FIG/%s_spe_po%d_ad%d.png' % (prefix, PREOPT, ADAPT))
    else:
        savefig('./FIG/%s_spe.png' % prefix.replace('/dev/shm/',''))
                
    if (USE_MCMC>0): # plot chains
        close(2)
        figure(2, figsize=(10,8))
        for i in range(10):
            for j in range(10):
                ax = axes([0.0+i*0.1, 0.9-j*0.1, 0.1, 0.1])
                x = int(((i+0.5)/9.5)*NX)
                y = int(((j+0.5)/9.5)*NY)
                ind = x+NX*j
                # plot sample chains
                for k in range(NP-1,-1,-1):   #     Tex  v   fwhm   tau
                    plot(RES[ind,:,k] /mean(RES[ind,:,k]),
                         '-', lw=0.8, color=['k', 'b', 'g', 'r', 'c','m','y'][k%7])
                if (1):
                    # chi2 values
                    tmp = RES[ind,0:-1:100,NP]
                    # plot(arange(0,SAMPLES,100), tmp/mean(tmp), 'cx', ms=3)
                    plot(arange(0,SAMPLES,100), tmp, 'cx', ms=3)
                text(0.1, 0.7, '%d' % ind, transform=ax.transAxes, size=7)
                setp(ax, xticks=[], yticks=[])
        savefig('./FIG/%s_chain_po%d_ad%d.png' % (prefix, PREOPT, ADAPT))
        
    if (PLOT):
        SHOW()

        
