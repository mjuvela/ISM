from   matplotlib.pylab import *
from   scipy.interpolate import interp1d
from   scipy.integrate import   quad
import time
import os, sys
import pyopencl as cl
# from   numba import njit

FLOAT = np.float32

h0    =  6.62606957e-27
k0    =  1.3806488e-16
c0    =  29979245800.0
hk0   =  4.799243348e-11

ESCNAMES = {'lvg': 0, 'simple': 1, 'slab': 2, 'sphere' : 3}

    
def SHOW():
    show(block=True)
    sys.exit()

    
def ReadIni(filename):
    INI = {}
    INI.update({'mol':          'co'   })
    INI.update({'tkin':          10.0  })
    INI.update({'tlim':          200.0 })
    INI.update({'alfa':          0.7   })
    INI.update({'colden':       [1e10, 1e15, 20] })
    INI.update({'fwhm':         1.0 })
    INI.update({'density':      [1e2, 1e6,   20] })
    INI.update({'gpu':          0 })
    INI.update({'sdevice':      ''  })
    INI.update({'cabu':         []})
    INI.update({'output':       'pepo.output'})
    INI.update({'escape':       0 })  #      tmp = {'lvg': 0, 'simple': 1, 'slab': 2, 'sphere' : 3}
    INI.update({'tolerance':    [ 1.0e-8, 1.0e-5, 100] })  # abs + rel tolerances
    INI.update({'hfs':          ''})
    for L in open(filename).readlines():
        if (len(L)<2): continue
        s = L.split()
        if (len(s)<2): continue
        if (s[0]=='mol'):       INI.update({'mol':       s[1]})
        if (s[0]=='sdevice'):   INI.update({'sdevice':   s[1]})
        if (s[0]=='output'):    INI.update({'output':    s[1]})        
        if (s[0]=='hfs'):       INI.update({'hfs':        s[1]})
        if (s[0]=='escape'):
            try:
                if (s[1] in ESCNAMES.keys()):
                    INI.update({'escape': ESCNAMES[s[1]]})
            except:
                try:
                    INI.update({'escape':    int(s[1])})
                except:
                    print("Error in escape prescription: %s" % s[1])
                    sys.exit()
        if (s[0]=='alfa'):      INI.update({'alfa':      float(s[1])})
        if (s[0]=='tkin'):      INI.update({'tkin':      float(s[1])})
        if (s[0]=='fwhm'):      INI.update({'fwhm':      float(s[1])})
        if (s[0]=='tlim'):      INI.update({'tlim':      float(s[1])})
        if (s[0]=='gpu'):       INI.update({'gpu':       int(s[1])})
        if (s[0]=='colden'):    INI.update({'colden':   [float(s[1]), float(s[2]), int(s[3])]})
        if (s[0]=='density'):   INI.update({'density':  [float(s[1]), float(s[2]), int(s[3])]})
        if (s[0]=='tolerance'): 
            if (len(s)>3): # also maxiter given
                INI.update({'tolerance': [float(s[1]), float(s[2]), int(s[3])]})
            else:
                maxiter = int(INI['tolerance'][2])   # keep the default value
                INI.update({'tolerance': [float(s[1]), float(s[2]), maxiter]})
        if (s[0]=='cabu'):
            tmp = []
            for x in s[1:]:
                if (x=='#'): break
                tmp.append(float(x))
            INI.update({'cabu': tmp})
    return INI


def InitCL(gpu, platforms=[0,1,2,3,4], sdevice=''):
    """
    Initialise OpenCL environment.
    Input:
        gpu       =   if >0, use GPU instead of CPU, must be given
        platform  =   array of potential platforms, optional parameter
        sdevice   =   overrides platform, selecting platform based on a string, optional parameter
    Return:
        context   =  compute context
        commands  =  command queue
    """
    context          = []  # for each device
    commands         = []
    print("==============================================================================")
    for iplatform in platforms:            
        platform    = cl.get_platforms()[iplatform]
        if (gpu==0):  devices  = platform.get_devices(cl.device_type.CPU)
        else:         devices  = platform.get_devices(cl.device_type.GPU)
        device = []
        for idevice in range(len(devices)):
            if ((len(sdevice)<1)|(sdevice in devices[idevice].name)):
                device = [ devices[idevice] ]
                break
        if (len(device)>0): break
    if (len(device)<1):
        print("InitCL_string: could not find any device matching string: %s" % INI['sdevice'])
        sys.exit()
    print(sdevice)
    try:
        context   =  cl.Context(device)
        queue     =  cl.CommandQueue(context)
    except:
        print("Failed to create OpenCL context and quee for device: ", device[0])
        sys.exit()
    if (1):
        print("Selected:")
        print("   Platform: ", platform)
        print("   Device:   ", device)
        print("================================================================================")
    return platform, device, context, queue,  cl.mem_flags


def DownloadMoleculeData(mol):
    """
    Download molecular data from Leiden LAMDA database, unless the file already
    exists in the current directory.
    ... not very useful, unless one already knows the name of the file
    """
    if (os.path.exists(f'{mol}.dat')): return
    os.system(f'wget https://home.strw.leidenuniv.nl/~moldata/datafiles/{mol}.dat')
    
    
def ReadMolecule(mol):
    """
    Read molecular data from LAMDA datafile.
    Usage:
        NL, E, G, UL, AUL, F, TCOL, CC = ReadMolecule(mol)
    Input:
        mol   =   name of the molecule (with LAMDA data in {mol}.dat)
    Return:
        NL    =   number of energy levels
        E     =   energy for each level [cgs], E[NL]
        G     =   statistical weights, G[NL]
        UL    =   UL[i]=[u,l], transition i corresponding to levels u->l (0-offset indices)
        AUL   =   Einstein A-coefficients [1/s]
        F     =   transition frequencies [Hz]
        TCOL  =   vector of Tkin values for collisional coefficients
        CC    =   list of NL*NL matrixes, collisional coefficients for each partner separately        
    """
    fname = f'{mol}.dat'
    L     = open(fname).readlines()
    i     = 0
    while (L[i][0:1]=='!'): i+= 1
    name  = L[i].split()[0]         # molecule name in the file
    i    += 1
    while (L[i][0:1]=='!'): i+= 1
    M     = float(L[i].split()[0])  # molecular weight
    i    += 1
    while (L[i][0:1]=='!'): i+= 1
    NL    = int(L[i].split()[0])    # number of energy levels
    i    += 1                       
    while (L[i][0:1]=='!'): i+= 1   # first line with level, energy, g, "J"
    d     = np.loadtxt(fname, skiprows=i, max_rows=NL, usecols=(1,2))
    E     = asarray(h0*c0*d[:,0].copy(), FLOAT)         # E [cgs]
    G     = asarray(d[:,1].copy(), FLOAT)
    i    += NL
    while (L[i][0:1]=='!'): i+= 1   
    NT    = int(L[i].split()[0])    # number of transitions
    i    += 1
    while (L[i][0:1]=='!'): i+= 1   # first line with tran, upper, lower, Aul, freq, Eupper
    d     = loadtxt(fname, skiprows=i, max_rows=NT)
    UL    = asarray(d[:,1:3]-1, int32)            # transitions UL[i]={upper, lower} 0-offset level indices
    AUL   = asarray(d[:,3].copy(), FLOAT)         # must be double ? ... not necessary?
    F     = asarray(d[:,4].copy(), FLOAT) * 1e9   # preferably double ? ... not necessary?
    i    += NT
    while (L[i][0:1]=='!'): i+= 1   
    NCP   = int(L[i].split()[0])    # number of collisional partners
    i    += 1
    CP    = []                      # names of collisional partners
    CC    = []
    for j in range(NCP):
        while (L[i][0:1]=='!'): i+= 1  
        CP.append(L[i].split()[1])  # name of this collisional partner
        i  +=   1
        while (L[i][0:1]=='!'): i+= 1
        ntran =  int(L[i].split()[0])   # collisional transitions for this partner
        i  +=   1
        while (L[i][0:1]=='!'): i+= 1
        ntkin =  int(L[i].split()[0])   # number of collisional temperatures for this partner
        i  +=   1
        while (L[i][0:1]=='!'): i+= 1
        tcol = loadtxt(fname, skiprows=i, max_rows=1)
        if (j==0): TCOL = 1.0*tcol
        if (len(tcol)!=len(TCOL)):
            print(f'Molecule {mol}:  different number of Tkin for collisional partners')
            sys.exit()
        if (np.max(np.abs(tcol-TCOL)>0.1)):
            print(f'Molecule {mol}:  different Tkin for collisional partners')
            sys.exit()
        i +=    1
        while (L[i][0:1]=='!'): i+= 1  
        d    =  loadtxt(fname, skiprows=i, max_rows=ntran)
        i   +=  ntran
        # we asssume that the temperature grid is the same for all partners
        C    = zeros((ntkin, NL, NL), FLOAT)  # ok with float
        for ii in range(ntran):
            u, l       =  int(d[ii,1])-1, int(d[ii,2])-1
            C[:, l, u] = d[ii,3:]
            # print("%2d -> %2d   %10.3e %10.3e" % (u, l, C[1, l, u], C[2, l, u]))
        CC.append(C.copy())
    ####
    if (0):
        print(NL)
        print('E', E)
        print('G', G)
        print('UL', UL)
        print('F', F)
        print('TCOL', TCOL)
        sys.exit()        
    return NL, E, G, UL, AUL, F, TCOL, CC


# @njit
def PlanckFunction(f, T):
    # return 2.0*h0*f**3/c0**2 / (exp(clip(hk0*f/T, 1e-30, 40.0))-1.0+1e-30)
    return 2.0*h0*(f/c0)**2*f / (exp(hk0*f/T)-1.0+1e-30)


def ReadHFS(filename):
    # For the moment only a single HFS line allowed !!!
    s = open(filename).readline().split()
    u, l, freq  =  int(s[1]), int(s[2]), float(s[3])
    hfs = loadtxt(filename)
    return u, l, freq, hfs

# @njit
def Gauss(x, s):
    return exp(-0.5*(x/s)**2) / sqrt(2.0*pi*s*s)


def get_HFS_beta(hfs, fwhm):
    # mapping   beta(single)  ->   beta(HFS)
    # hfs[nc, 2] = {  velocity, component strength }
    v1     =  asarray(arange(-2*fwhm, 2*fwhm, fwhm/50.0), float32) 
    v2     =  asarray(arange(min(hfs[:,0])-2*fwhm, max(hfs[:,0])+2*fwhm, fwhm/50.0), float32)
    n      =  100                     # tau values
    TAU    =  logspace(-2.0, 3.0, n)
    RES    =  zeros((n,2), float32)
    s      =  fwhm/sqrt(8.0*log(2.0))
    emit1  =  Gauss(v1, s)
    nc     =  hfs.shape[0]  # number of HF components
    emit2  =  zeros(len(v2), float32)
    for ic in range(nc):
        emit2 += hfs[ic,1]*Gauss(v2-hfs[ic,0], s)
    ####
    for it in range(n):
        tau    =  TAU[it]*Gauss(v1, s)
        # escape probability for a single Gaussian
        betaG  =  sum(emit1*exp(-tau))  /  sum(emit1)
        # same for a combination of HS components
        tau    =  zeros(len(v2), float32)
        for ic in range(nc):
            tau +=  hfs[ic,1]*TAU[it]*Gauss(v2-hfs[ic,0], s)
        betaH   =   sum(emit2*exp(-tau)) / sum(emit2)
        RES[it,:] =   [betaG, betaH]
    # interpolate onto new grid of betaG values
    r      =  RES[:,1]/RES[:,0]             #  beta(HFS) / beta(Gaussian)
    ip     =  interp1d(RES[:,0], r, bounds_error=False, fill_value=(r[0], r[-1])) # betaG ->  betaHFS/betaG
    HFSN   =  200
    HFS0   =  1.0e-3                        #  betaG = 10**[-3, 0]
    HFSK   =  10**(3/HFSN)
    betaG  =  HFS0*HFSK**arange(HFSN)       # new grid of betaG values
    HFSC   =  asarray(ip(betaG), FLOAT)
    HFSMAX =  max(emit2)/max(emit1)         # how much lower is the peak optical depth because of HFS?
    return HFSN, HFS0, HFSK, HFSC, HFSMAX
