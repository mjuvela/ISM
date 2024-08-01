from matplotlib.pylab import *
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid, quad
import time
import scipy
import pyopencl as cl
## from MJ.Aux.mjGPU import *


PARSEC, STEFAN_BOLTZMANN  =  3.0856775e+18, 5.670373e-05
C_LIGHT, H_K, PLANCK      =  29979245800.0, 4.799243348e-11, 6.62606957e-27
delta, DELTA              =  1.0e-6, 4.0e-6
EPS                       =  1.0e-5
LSUN, RSUN  =  3.839e+33, 69550000000.0   # Solar luminosity [cgs], Sun's radius [cm]

def SHOW():
    show(block=True)
    sys.exit()

def ReadIni(filename):
    print("ReadIni....")
    INI = {}
    INI.update({'weightbg':      -2.0 })
    INI.update({'weightemit':    -2.0 })
    INI.update({'weightsca':     -2.0 })
    INI.update({'weightemitps':  -2.0 })
    INI.update({'weightscaps':   -2.0 })
    INI.update({'weightstepsca': -2.0 })
    INI.update({'weightstepabs': -2.0 })
    INI.update({'background':    ['none', 1] })
    INI.update({'pointsource':   ['none', 1, 0.0] })
    INI.update({'bgpackets':     0 })
    INI.update({'pspackets':     0 })
    INI.update({'gpu':           0 })
    INI.update({'sdevice':       ''})
    INI.update({'platform':      -1 })
    INI.update({'seed':          rand()})
    INI.update({'gridlength':    1.0})
    INI.update({'offsets':       10})
    INI.update({'prefix':       'prefix'})    
    INI.update({'forced':        0})
    INI.update({'batch':         0})
    INI.update({'density':      -1.0})
    for L in open(filename).readlines():
        if (len(L)<2): continue
        s = L.split()
        if (len(s)<2): continue
        if (s[0]=='cloud'):         INI.update({'cloud':         s[1]})
        if (s[0]=='dust'):          INI.update({'dust':          s[1]})
        if (s[0]=='gridlength'):    INI.update({'gridlength' :   float(s[1])})
        if (s[0]=='density'):       INI.update({'density' :      float(s[1])})
        if (s[0]=='background'):    INI.update({'background' :   [s[1], float(s[2])]})               # file k
        if (s[0]=='pointsource'):   INI.update({'pointsource':   [s[1], float(s[2]), float(s[3])]})  # file k  R
        if (s[0]=='bgpackets'):     INI.update({'bgpackets'  :   int(s[1])})
        if (s[0]=='pspackets'):     INI.update({'pspackets'  :   int(s[1])})
        if (s[0]=='prefix'):        INI.update({'prefix':        s[1]})
        if (s[0]=='offsets'):       INI.update({'offsets':       int(s[1])})
        if (s[0]=='weightbg'):      INI.update({'weightbg':      float(s[1])})
        if (s[0]=='weightemit'):    INI.update({'weightemit':    float(s[1])})
        if (s[0]=='weightsca'):     INI.update({'weightsca':     float(s[1])})
        if (s[0]=='weightemitps'):  INI.update({'weightemitps':  float(s[1])})
        if (s[0]=='weightscaps'):   INI.update({'weightscaps':   float(s[1])})
        if (s[0]=='weightstepsca'): INI.update({'weightstepsca': float(s[1])})
        if (s[0]=='weightstepabs'): INI.update({'weightstepabs': float(s[1])})
        if (s[0]=='gpu'):           INI.update({'gpu':           int(s[1])})
        if (s[0]=='platform'):      INI.update({'platform':      int(s[1])})        
        if (s[0]=='forced'):        INI.update({'forced':        int(s[1])})        
        if (s[0]=='batch'):         INI.update({'batch':         int(s[1])})        
        if (s[0]=='sdevice'):       INI.update({'sdevice':       s[1]})
        if (s[0]=='seed'):
            tmp = float(s[1])
            if (tmp>0.0):         INI.update({'seed':        tmp})
        if (s[0]=='mapum'):
            um = []
            for x in s[1:]:  um.append(float(x))
            INI.update({'mapum': asarray(um, float32).copy()})
    print(INI)
    return INI


def ReadCloud(filename):
    d   = loadtxt(filename, skiprows=1)
    R   = asarray(d[:,0], float32)
    RHO = asarray(d[:,1], float32)
    return R, RHO


def ReadDust(filename):
    L      = open(filename).readlines()
    l      = 1
    while (L[l][0:1]=='#'): l += 1
    d2g    = float(L[l].split()[0].strip())
    l     += 1
    while (L[l][0:1]=='#'): l += 1
    a      = float(L[l].split()[0].strip())
    l     += 1
    while (L[l][0:1]=='#'): l += 1
    d      = loadtxt(filename, skiprows=l+1)
    ###
    FF     = d[:,0].copy()
    KABS   = clip(d2g * pi*a**2 * d[:,2] * PARSEC, 1.0e-30, 1.0e30) #  tau / H * PARSEC
    KSCA   = clip(d2g * pi*a**2 * d[:,3] * PARSEC, 1.0e-30, 1.0e30) #  tau / H * PARSEC
    G      = d[:,1].copy()
    G[nonzero(abs(G)<1e-3)] = 1.0e-3
    return FF, G, KABS, KSCA


def um2f(um):
    return 1e4*C_LIGHT/um

def f2um(f):
    return 1e4*C_LIGHT/f

def PlanckFunction(f, T):
    # return 2.0*PLANCK*f**3/C_LIGHT**2 / (exp(H_K*f/T)-1.0)
    return 2.0*PLANCK*f**3/C_LIGHT**2 / (exp(clip(H_K*f/T, 1e-30, 60.0))-1.0)

def HenyeyGreenstein(g):  # return random cos(theta) for scattering angle
    return (1+g*g-((1-g*g)/(1-g+2*g*rand()))**2) / (2.0*g)

def Deflect(DIR, ct):
    # Deflect direction by scattering angle theta, given ct=cos(theta)
    phi       =  2.0*pi*rand()
    sin_theta =  sqrt(1.0-ct*ct)
    ox        =  sin_theta*cos(phi)
    oy        =  sin_theta*sin(phi)
    oz        =  ct
    # compute direction of the old vector - rotate in the opposite direction: theta0, phi0
    cx, cy, cz = DIR
    theta0    =  arccos(cz/sqrt(cx*cx+cy*cy+cz*cz+EPS))
    phi0      =  arccos(cx/sqrt(cx*cx+cy*cy+DELTA))
    if (cz<0.0): phi0 = (2.0*pi-phi0)
    theta0    = -theta0 ;
    phi0      = -phi0 ;
    # rotate (ox,oy,oz) with angles theta0 and phi0
    # 1. rotate around z angle phi0,   2. rotate around x (or y?) angle theta0
    sin_theta =  sin(theta0)
    cos_theta =  cos(theta0)
    sin_phi   =  sin(phi0)
    cos_phi   =  cos(phi0)
    DIR[0]    = +ox*cos_theta*cos_phi   + oy*sin_phi   -  oz*sin_theta*cos_phi
    DIR[1]    = -ox*cos_theta*sin_phi   + oy*cos_phi   +  oz*sin_theta*sin_phi
    DIR[2]    = +ox*sin_theta                          +  oz*cos_theta
    return

def Print(POS, DIR, IND, TAG):
    print("%s -- %2d:   POS= %8.4f %8.4f %8.4f     DIR= %8.4f %8.4f %8.4f" % (TAG, IND, POS[0], POS[1], POS[2], DIR[0], DIR[1], DIR[2]))
   
def interpL(xx, yy, kind='cubic',bounds_error=False, fill_value=1.0e-30):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = scipy.interpolate.interp1d(logx, logy, kind=kind, bounds_error=bounds_error, fill_value=log10(fill_value))
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp
                                        
def FixG(g):
    # avoid |g|<0.005
    if (fabs(g)>0.005): return g
    if (g<0.0): return -0.005
    return 0.005


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

