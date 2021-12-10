
from   matplotlib.pylab import *
from   scipy.interpolate import interp1d
import pyopencl as cl
from   pyopencl import array as clarray
import os, sys, scipy, time
from   scipy.interpolate import RectBivariateSpline
import ctypes
import numpy as np
import time
import mmap


def I2F(i):
    # Convert cell index (int32) to float value
    return ctypes.c_float.from_buffer(ctypes.c_int(i)).value
    
def F2I(x):
    # Convert float value x into integer value (int32)
    return ctypes.c_int.from_buffer(ctypes.c_float(x)).value
        
def EXIT(txt):
    print(txt)
    sys.exit()
    

# MUST BE THE SAME AS IN SOC_aux.py, AS LONG AS A2E_pyCL.py AND A2E_MABU.py READ FACTOR FROM THERE !!!
FACTOR  =  1.0e20  
    
"""
Python version of the SOCAMOAMOF program.
"""

C_LIGHT =  2.99792458e10  
PLANCK  =  6.62606957e-27 
H_K     =  4.79924335e-11 
D2R     =  0.0174532925       # degree to radian
PARSEC  =  3.08567758e+18 
H_CC    =  7.372496678e-48 

SEED0   =  0.8150982470475214
SEED1   =  0.1393378751427912
MAXPS   =  3000   # maximum number of point sources

DEGREE_TO_RADIAN =  0.0174532925199432958
RADIAN_TO_DEGREE =  57.2957795130823208768


MAX_SPLIT = 2560   # for MWM
MAX_SPLIT = 3560   # 2021-03-06 -- needed for IMF256 snap 176 ... and not enough !!
MAX_SPLIT = 3800   # 2021-03-06 -- needed for IMF256 snap 176, even IRDC, close to Tux GPU memory limit
# MAX_SPLIT = 5120   # 2020-12-05 .... again 2021-03-06

def Planck(f, T):      # Planck function
    return 2.0*H_CC*f*f*f / (exp(H_K*f/T)-1.0) 

def PlanckSafe(f, T):  # Planck function
    # Add clip to get rid of warnings
    return 2.0*H_CC*f*f*f / (exp(clip(H_K*f/T,-100,+100))-1.0)

def PlanckTest(f, T):  # Planck function
    return 2.0*H_CC*f * (f*f / (exp(clip(H_K*f/T,-80,+80))-1.0))

def um2f(um):  # convert wavelength [um] to frequency [Hz]
    return C_LIGHT/(1.0e-4*um)

def f2um(f):   # convert frequency [Hz] to wavelength [um]
    return 1.0e4*C_LIGHT/f

def Trapezoid(x, y):
    dx     = x[2:] - x[:(-2)]   #  x[i+1] - x[i-1], lengths of intervals for Trapezoid rule    
    res    = y[0]*(x[1]-x[0]) + y[-1]*(x[-1]-x[-2]) # first and last step
    res   += sum(y[1:(-1)]*dx)  # the sum over the rest of y*dx
    return res

class User:    
    def __init__(self, filename):
        """
        Read ini-file and store run parameters
        """
        # input files
        self.file_cloud       = ''   # file to read cloud description from
        self.file_diffuse     = ''   # file to read diffuse emission [photons/cm3]
        self.file_background  = ''   # file to read isotropic backgroun intensity [cgs]
        self.file_polred      = ''   # file for polarisation reduction factor R
        self.file_constant_load = '' # file to load heating by constant sources, with reference field
        self.file_external_mask = '' # mask for emitting cells (==1)
        self.file_optical     = []   # file to read dust parameters from (one per dust)
        self.file_scafunc     = []   # files to read scattering function from (one per dust)
        self.file_abundance   = []   # files to read abundances from (one per dust)
        self.file_hpbg        = ''   # Healpix file for the background intensity
        self.HPBG_WEIGHTED    = False
        
        # output files 
        self.file_absorbed    = 'default.absorbed'   # file to save number of absorptions per cell
        self.file_emitted     = 'soc.emitted'   # file to save emission per cell
        self.file_sourcemap   = ''   # file to save map of attenuated sources
        self.file_temperature = ''   # file to save dust temperatures
        self.file_savetau     = ''   # file to save column density map or tau map
        self.file_pssavetau   = ''   # file to save column density and optical depth towards point sources
        self.file_scattering  = 'scattering'   # file to save images of scattered light
        self.file_constant_save = '' # file to save heating by constant sources, with reference field
        self.kernel_defs      = ''   # additional defs passed onto kernel compilation

        # run parameters
        self.GL           = 0.0      # grid size parameter [pc]
        self.MAP_DX       = 1.0      # pixel size in root grid cell length units
        self.KDENSITY     = 1.0      # scaling of density read from tile
        self.DISTANCE     = 0.0      # distance of the model [pc]
        self.ITERATIONS   = 1        #  number of iterations (simulation / T calculation)
        self.STEP_WEIGHT  = [-1,0,0] #  parameters of step-length weighting
        self.DIR_WEIGHT   = [-1,0,0] # parameters of direction weighting
        self.NPIX         = cl.cltypes.make_int2(10,10)  # number of map pixels
        self.FAST_MAP     = -1       # fast map calculation (>1 == several frequencies in parallel)
        self.REMIT_F      = [0.0, 1e30] # save emission and spectra only for this frequency range
        ## self.SIM_F        = [1.2e12, 3.3e15 ] # limit simulated frequencies
        self.SIM_F        = [1.0e8, 1.0e17 ] # limit simulated frequencies
        self.LEVEL_THRESHOLD = 0         # if >0, save (polarisation) ignore large cells, level<LEVEL_THRESHOLD
        self.INTOBS       = cl.cltypes.make_float3(-1e12,0,0) # location of an internal observer
        self.MAPCENTRE    = cl.cltypes.make_float3(-1e12,0.0) # position towards centre of the maps
        self.DEVICES      = 'c'      # computing devices to be used
        self.FISSION      =  0       # optional, use a subdevice with FISSION threads
        self.DSC_BINS     = 0        # angle bins in scattering function
        self.LOCAL        = -1       # is user request specific value of local
        self.GLOBAL       = -1       # overrides GLOBAL in ASOCS.py
        self.BATCH        = 30       # sub-batch size of rays per kernel call
        self.OBS_THETA    = []       # direction of the observer, angle from +Z [deg]
        self.OBS_PHI      = []       # direction of the observed, agnle from +X [deg]
        self.PSPAC        = 0        # number of photon packages from EACH point source
        self.PS_METHOD    = 0 
        self.BGPAC        = 0        # number of photon packages from background
        self.CLPAC        = 0        # number of photon packages from medium
        self.DFPAC        = 0        # number of photons packages for diffuse emission (if CLPAC=0)
        # List of point sources
        self.NO_PS        = 0        # number of point sources
        self.file_pointsource = []   # files to read point source intensities [cgs]
        self.PS_SCALING   = ones(MAXPS, float32)  # scaling of the file values
        self.PSPOS        = zeros(MAXPS, cl.cltypes.float3)  # locations of point source
        for i in range(MAXPS): self.PSPOS[i][0] = -1e10
        
        self.DO_SPLIT     = 0        # include splitting of photon packages
        self.POLMAP       = 0        # if >0, write maps of I, Q, U
        self.BFILES       = []       # names of Bx, By, and Bz files
        self.POLSTAT      = 0        # write maps of polarisation statistics: S, <cos_theta>, ...
        self.NOSOLVE      = 0        # if >0, do not solve dust temperatures in SOC
        self.LOAD_TEMPERATURE = 0    # if >0, load old temperature file and recalculate emission
        self.NOMAP        = 0        # if >0, do not write any maps
        self.NOABSORBED   = 0        # if >0, calculate Ein on the fly (non-stochastic grains only) 
        self.SAVE_INTENSITY = 0      # 1 = save intensity (for DustEM), 2 = save [I, Ix, Iy, Iz], 3 = based on ABSORBED
        self.SAVE_INTENSITY_FILE = 'ISRF.DAT'
        self.USE_EMWEIGHT = 0        # if >0, weight package generation with emission
        self.EMWEIGHT_SKIP = 3       # re-evaluate for every skip:th frequency
        self.EMWEIGHT_LIM = [0.0,1e10,0.0] # min/max number of packages per cell, threshold to ignore
        self.p0           = 0.2      # maximum polarisation
        self.MAXLOS       = 1e10     # maximum length of the LOS [GL]
        self.MINLOS       = -1.0
        self.Y_SHEAR      =  0.0     # for shearing box simulation, implies periodicity in xy plane
        self.INTERPOLATE  =  0       # if *healpix* map-making uses interpolation...
        self.SEED         = pi/4.0   # seed value for random number generators
        self.MAP_FREQ     = [1.0e6, 1e18] # frequency limits for the maps to be saved [Hz]
        self.SINGLE_MAP_FREQ = []    # individual frequencies for which only to write the maps
        self.SOLVE_ON_DEVICE = 0     # if >0, solve temperatures on device
        self.FFS          = 1        # use forced first scattering
        self.BG_METHOD    = 0        # version of the BGPAC simulation part
        self.WITH_ALI     = 0        # use ALI (if the method supports it)
        self.WITH_REFERENCE = 0      # use reference field (if method supports it)
        self.scale_background = 1.0
        # These are not read from ini file but are written here for convenience
        self.AREA         = 0.0      # surface area of the model [GL^2]
        self.AXY          = 0.0      # area of one side of the model [GL^2]
        self.AXZ          = 0.0      # ... these updated here once model has been read
        self.AYZ          = 0.0
        self.FFREQ        = []
        self.FABS         = []
        self.FSCA         = []
        self.LEVELS       = 999      # maximum number of levels included
        self.CELLS        = 0
        self.LCELLS       = 0
        self.OFF          = None
        self.DENS         = None
        self.REMIT_NFREQ  = 0
        self.REMIT_I1     = 0
        self.REMIT_I2     = 0
        self.KEYS         = {}
        self.PLATFORM     = -1       # number of the OpenCL platform, <0 = figure it out
        self.IDEVICE      = 0        # index of the device within the selected platform (index over CPUs or GPUs only)
        self.K_DIFFUSE    = 1.0      # scaling of the input diffuse radiation field
        self.SINGLE_ABU   = 0
        self.OPT_IS_HALF  = 0
        self.POL_RHO_WEIGHT = 0 
        self.savetau_freq =  []       # if >0, save tau map instead of column density map
        self.pssavetau_freq = -1.0     # if >0, save tau map instead of column density map
        
        self.ROI           = zeros(6, int32)   # ROI limits in root grid cells [x0,x1,y0,y1,z0,z1] = [x0:(x1+1), ...]
        self.ROI_STEP      = 0       # for ROI_SAVE, subsampling of GL
        self.ROI_MAP       = 0       # True if maps are made using only emission within ROI
        self.ROI_NSIDE     = 16      # healpix map resolution for storing photons that enter ROI
        self.WITH_ROI_SAVE = 0
        self.WITH_ROI_LOAD = 0
        self.ROI_LOAD_SCALE = 1.0
        self.FILE_ROI_SAVE = ''      # filename to save packages that enter ROI
        self.FILE_ROI_LOAD = ''      # filename to load packages that enter from outside
        self.ROIPAC        = 0
        
        self.OUT_NSIDE     = 128     # NSIDE for scattering maps (healpix)
        
        self.FSELECT       = []      # selected frequencies (simulation using library)
        self.LIB_ABS       = False   # only calculate absorptions at FSELECT frequencies
        self.LIB_MAPS      = False   # only calculate maps from library-solved emission

        self.MAP_INTERPOLATION = 0   # if normal mapmaking (not healpix) uses interpolation
        self.FITS          = 0       # 2021-12-09 changed to default 1... and back to 0
        self.FITS_RA       = 0.0     # optional  FITS centre coordinates in degrees
        self.FITS_DE       = 0.0
        
        # read inifile
        for line in open(filename).readlines():    
            # assert((DIFFUSE==1)&(WITHDUST==1)&(DISTANCE<1.0))
            s = line.split('#')[0].split()
            if (len(s)<1): continue

            if (0):
                if (s[0]=='EXEC'):
                    cmd = line[5:]
                    print('### EXEC %s' % cmd)
                    os.system(cmd)

            if (s[0]=='DEFS'):
                self.kernel_defs = line[4:]
                # drop tailing comments
                self.kernel_defs = self.kernel_defs[0:self.kernel_defs.index('#')]
                
            if (s[0].find('mapum')==0):
                for ss in s[1:]:
                    self.SINGLE_MAP_FREQ.append( um2f(float(ss)) )
                print("ASOC: mapum ", self.SINGLE_MAP_FREQ)

            if (s[0]=='singleabu'):
                self.SINGLE_ABU = 1
            if (s[0]=='optishalf'):
                self.OPT_IS_HALF = 1
                
                
            self.KEYS.update({s[0]: s[1:]})
            
            # keywords without arguments
            key = s[0].lower()
            if (key.find('nosolve')==0):    self.NOSOLVE  = 1
            if (key.find('loadtemp')==0):   self.LOAD_TEMPERATURE  = 1
            if (key.find('nomap')==0):      self.NOMAP  = 1
            if (key.find('noabs')==0):      self.NOABSORBED  = 1
            # if (key.find('saveint')==0):    self.SAVE_INTENSITY = 1
            if (key.find('dustem')==0):
                self.NOABSORBED     =  1
                self.SAVE_INTENSITY =  1
            if (key.find('solveondev')==0):     self.SOLVE_ON_DEVICE  = 1
            if (key.find('xemonhost')==0):      self.XEM_ON_HOST  = 1
            if (key.find('polrhoweight')==0):   self.POL_RHO_WEIGHT  = 1            
            if (key.lower().find('roimap')==0): self.ROI_MAP = 1
            
            if ((key.find('savetau')==0)&(len(s)>2)):
                #     savetau   filename   um1 um2 um3 ....
                #  if wavelength == -um[i], that means saving of column density
                self.file_savetau  = s[1]
                for x in s[2:]:
                    if (abs(float(x))<1):  self.savetau_freq.append(0.0)  # meaning column density
                    else:                  self.savetau_freq.append(um2f(float(x)))
                print("self.savetau_freq ", self.savetau_freq)
                    
            if (key.find('pssavetau')==0):
                self.file_pssavetau  = s[1]
                self.pssavetau_freq  = um2f(float(s[2]))
            
            if ((key.find('FITS')==0)|(key.find('fits')==0)): 
                self.FITS  =  1
                if (len(s)>=3):  # set also centre coordinates
                    self.FITS_RA = float(s[1]) 
                    self.FITS_DE = float(s[2])
                
                
                
                
            if (len(s)<2): continue
            # keywords with a single argument
            key, a  =  s[0], s[1]
            if (key.find('device')==0):      self.DEVICES           = a
            if (key.find('fission')==0):     self.FISSION           = int(a)
            if (key.find('sourcemap')==0):   self.file_sourcemap    = a
            if (key.find('tempera')==0):     self.file_temperature  = a
            if (key.find('cloud')==0):       self.file_cloud        = a
            if (key.find('absorb')==0):      self.file_absorbed     = a
            if (key.find('scatter')==0):     self.file_scattering   = a
            if (key.find('emit')==0):        self.file_emitted      = a
            if (key.find('emit')==0):        self.file_emitted      = a
            if (key.find('split')==0):       self.DO_SPLIT          = int(a)
            if (key.find('mapint')==0):      self.MAP_INTERPOLATION = int(a)
            if (key.find('polstat')==0):     self.POLSTAT           = int(a)
            
            if (key.find('platform')==0):    
                self.PLATFORM   = int(a)
                if (len(s)>2):
                    try:
                        self.IDEVICE = int(s[2])
                    except:
                        self.IDEVICE = 0
            if (key.find('libabs')==0): 
                self.FSELECT = loadtxt(a)
                if (size(self.FSELECT)==1): self.FSELECT = asarray(self.FSELECT, float32).reshape(1,)
                self.LIB_ABS = True
                print("Simulating only selected frequencies:")
                print(self.FSELECT)
            if (key.find('libmap')==0): 
                self.FSELECT = loadtxt(a)
                if (size(self.FSELECT)==1): self.FSELECT = asarray(self.FSELECT, float32).reshape(1,)                
                self.LIB_MAPS = True
                print("Writing maps only for selected frequencies:")
                print(self.FSELECT)
            if (key.find('diffus')==0):   
                self.file_diffuse   = a
                if (len(s)>2):
                    self.K_DIFFUSE = float(s[2])
            if (key.find('optic')==0):     
                self.file_optical.append(a)
                # optional second argument = abundance file
                if (len(s)>2):
                    self.file_abundance.append(s[2])
                else:
                    self.file_abundance.append('#')
            if (key.find('externalm')==0):   self.file_external_mask= a            
            if (key.find('backg')==0):  # background intensity for each frequency
                self.file_background   = a
                if (len(s)>2): self.scale_background = float(s[2])
            if (key.find('hpbg')==0):   # Healpix map for intensity at each frequency
                self.file_hpbg = a
                if (len(s)>2): self.scale_background = float(s[2])
                if (len(s)>3): self.HPBG_WEIGHTED = int(s[3])
            if (key.find('polred')==0):      self.file_polred       = a
            if (key.find('cload')==0):       self.file_constant_load = a
            if (key.find('csave')==0):       self.file_constant_save = a
            if (key.find('iterations')==0):  self.ITERATIONS      =  int(a)
            if (key.find('threshold')==0):   self.LEVEL_THRESHOLD =  int(a)
            if (key.find('gridlen')==0):     self.GL              =  float(a)
            if (key.find('p0')==0):          self.p0              =  float(a)
            if (key.find('distance')==0):    self.DISTANCE        =  float(a)
            if (key.find('bgpac')==0):       self.BGPAC           =  int(float(a))
            if (key.find('pspac')==0):       self.PSPAC           =  int(float(a))
            if (key.find('psmetho')==0):     self.PS_METHOD       =  int(a)
            if (key.find('cellpac')==0):     self.CLPAC           =  int(round(float(a)))
            if (key.find('roipac')==0):      self.ROIPAC          =  int(round(float(a)))
            if (key.find('roinside')==0):    self.ROI_NSIDE       =  int(round(float(a)))
            if (key.find('diffpac')==0):     self.DFPAC           =  int(a)
            if (key.find('seed')==0):        self.SEED            =  clip(float(a), -1.0, 1.0)
            if (key.find('dens')==0):        self.KDENSITY        =  float(a)
            if (key.find('batch')==0):       self.BATCH           =  int(a)
            if (key.find('local')==0):       self.LOCAL           =  int(a)
            if (key.find('global')==0):      self.GLOBAL          =  int(a)
            if (key.find('forcedfirst')==0): self.FFS             =  int(a)
            if (key.find('ffs')==0):         self.FFS             =  int(a)
            if (key.find('bgmethod')==0):    self.BG_METHOD       =  int(a)
            if (key.find('ali')==0):         self.WITH_ALI        =  int(a)
            if (key.find('reference')==0):   self.WITH_REFERENCE  =  int(a)
            if (key.find('saveint')==0):     
                self.SAVE_INTENSITY  =  int(a)
                if (len(s)>2): self.SAVE_INTENSITY_FILE = s[2]
                # should still be ok to write maps even if we had the addition intensity vector comp.
                # if (self.SAVE_INTENSITY==2): self.NOMAP = 1
            if (key.find('levels')==0):      self.LEVELS          =  int(a)    # hierarchy levels !!
            if (key.find('yshear')==0):      self.Y_SHEAR         =  float(a)
            if (key.find('interpol')==0):    self.INTERPOLATE     =  float(a)
            if (key.find('outnside')==0):    self.OUT_NSIDE       =  int(a)
            if (key.find('emwei')==0):
                self.USE_EMWEIGHT = int(a)   # 0/1 = no weighting / precalculated    a == s[1]
                if (len(s)>3):
                    #  MIN(packets_per_cell), MAX(packets per cell) [, threshold:ignore cells]
                    self.EMWEIGHT_LIM = [ float(s[2]), float(s[3]), 0.0 ]
                    if (len(s)>4):
                        self.EMWEIGHT_LIM[2]   = float(s[4])
                        if (len(s)>5):
                            self.EMWEIGHT_SKIP = int(s[5])
                print("emwei %6.4f %6.4f %6.4f\n" % (self.EMWEIGHT_LIM[0], self.EMWEIGHT_LIM[1], self.EMWEIGHT_LIM[2]))
            if (len(s)<3): continue
            key, a, b = s[0], s[1], s[2]            
            # keywords with two arguments
            if (key.find('remit')==0):       self.REMIT_F = [ um2f(float(b)), um2f(float(a)) ]
            if (key.find('simum')==0):       self.SIM_F   = [ um2f(float(b)), um2f(float(a)) ]
            if (key.find('dsc')==0):         
                self.file_scafunc.append(s[1])
                if (len(self.file_scafunc)==1):
                    self.DSC_BINS     = int(s[2])
                else:
                    if (self.DSC_BINS!=int(s[2])):
                        print("*** Error in scattering functions: number of bins must be the same for all dusts")
                        sys.exit()                        
            if (key.find('direwei')==0):
                self.DIR_WEIGHT = [ int(a), float(b) ]
            if (key.find('direct')==0):
                if (len(self.OBS_THETA)>=10):
                    print("** ERROR - cannot have more than 10 directions -- ABORT !!")
                    sys.exit()
                self.OBS_THETA.append(float(a)*D2R)
                self.OBS_PHI.append(  float(b)*D2R)
            if (key.find('wavelen')==0):
                self.MAP_FREQ = [ um2f(float(b)), um2f(float(a)) ]
            if (key.find('roisave')==0):
                self.WITH_ROI_SAVE  = 1
                self.FILE_ROI_SAVE  = a
                self.ROI_STEP       = int(b)  # GL divided to ROI_STEP smaller surface elements
            if (key.find('roiload')==0):
                self.WITH_ROI_LOAD  = 1
                self.FILE_ROI_LOAD  = a
                self.ROI_LOAD_SCALE = float(b)  # ROI_LOAD does not use ROI_STEP !!
                                
            if (len(s)<4): continue
            # keywords with three arguments
            key, a, b, c = s[0], s[1], s[2], s[3]            
            if (key.find('polmap')==0):
                self.POLMAP  = 1
                self.BFILES = [ a, b, c ]
                if (len(s)==5):
                    self.MAXLOS  = float(s[4]) # maximum length of the LOS [GL]
                    print("MAXLOS %.3f" % self.MAXLOS)
                if (len(s)>5):
                    self.MINLOS = float(s[4])
                    self.MAXLOS = float(s[5])
                    print("MINLOS %.3f, MAXLOS %.3f" % (self.MINLOS, self.MAXLOS))
            if (key.find('perspec')==0):
                self.INTOBS = cl.cltypes.make_float3(float(a), float(b), float(c))
            if (key.find('stepwei')==0):
                # stepweight=0      p = exp(-x)
                # stepweight=1      p = A*exp(-A*x)
                # stepweight=2      p ~   exp(-B*x) + A*exp(-2*B*x)
                self.STEP_WEIGHT = [ int(a), float(b), float(c) ]
            if (key.find('mapping')==0):
                # mapping:  nx, ny,  dx[grid units]
                self.NPIX   = cl.cltypes.make_int2(int(a), int(b))
                self.MAP_DX = float(c)                
                if (len(s)>=4):
                    try:
                        self.FAST_MAP = int(s[4])
                    except:
                        pass
            if (key.find('mapcent')==0):
                self.MAPCENTRE = cl.cltypes.make_float3(float(a), float(b), float(c))
                                
            if (len(s)<5): continue
            if (key.find('pointsou')==0):
                if (self.NO_PS<MAXPS):  # there is room for another point source
                    self.PSPOS[self.NO_PS][0] = float(s[1])
                    self.PSPOS[self.NO_PS][1] = float(s[2])
                    self.PSPOS[self.NO_PS][2] = float(s[3])
                    self.file_pointsource.append(s[4])
                    if (len(s)>5):
                        if (s[5]!='#'):
                            self.PS_SCALING[self.NO_PS] = float(s[5])
                            # print(" PS [%d] -- SCALING %.1f" % (self.NO_PS, float(s[5])))
                    self.NO_PS += 1
                else:
                    print("Reached maximum number of point sources = %d", MAXPS)
                    sys.exit()
                    
            if ((key=='roi')&(len(s)>=6)):
                self.ROI = asarray([ int(s[1]), int(s[2]), int(s[3]), int(s[4]), int(s[5]), int(s[6]) ], int32)
                print("ROI !!!", self.ROI)
                
        ## end of loop over inifile lines 
        
        # DFPAC==CLPAC or DFPAC>0 when CLPAC==0
        if (self.CLPAC>0): self.DFPAC = self.CLPAC

    
    ## end of __init__
    
    def Validate(self):
        ok = True
        if (len(self.file_cloud)<1):
            print("*** Cloud model not definied: keyword cloud")
            ok = False
        if ((self.CLPAC<1)&(self.WITH_ALI>0)):
            print("*** WARNING:  CLPAC=0 and WITH_ALI=%d" % self.WITH_ALI)
            print("***           Cannot use ALI, we set WITH_ALI=0")
            self.WITH_ALI = 0
        if (self.PSPAC<1):
            self.NO_PS = 0  # no point sources
        return ok
            
            
## end of User definition



def read_dust(USER):
    """
    Read optical data from the dust file(s).
    Input:
        USER    =  User object
    Return:
        FFREQ, FG, FABS, FSCA:
            FFREQ  =  frequencies
            AFG    =  values of asymmetry parameter (array of vectors)
            AFABS  =  optical depth / n / GL  for absorption (array of vectors)
            AFSCA  =  optical depth / n / GL  for scattering (array of vector)
    Note:
        USER should have the following parameters:
            USER.file_optical  =  file with the dust parameters
            USER.GL            =  grid length == GL == size of root grid cells [pc]
        USER.NFREQ is updated to correspond to the number of frequencies
    """
    # 2018-12-08 --- FG, FABS, FSCA changed into arrays of vectors to allow multiple dust species
    AFG, AFABS, AFSCA = [], [], []
    USER.NFREQ = -1
    for filename in USER.file_optical:
        print("read_dust: %s" % filename) 
        tmp           = open(filename).readlines()
        GRAIN_DENSITY = float(tmp[1].split()[0])
        GRAIN_SIZE    = float(tmp[2].split()[0])
        coeff         = GRAIN_DENSITY*pi*GRAIN_SIZE**2.0 * USER.GL*PARSEC  # tau / n / pc
        d             = loadtxt(filename, skiprows=4)
        FFREQ         = d[:,0]
        FG            = d[:,1]
        FABS          = d[:,2] * coeff
        FSCA          = d[:,3] * coeff
        if ((USER.NFREQ>0)&(USER.NFREQ!=len(FFREQ))):
            print("*** Error in optical parameters: dusts must have same frequency grid")
            sys.exit()
        USER.NFREQ    = len(FFREQ)
        AFG.append(FG)
        AFABS.append(FABS)
        AFSCA.append(FSCA)
    USER.FFREQ, USER.FABS, USER.FSCA = FFREQ, AFABS, AFSCA
    return FFREQ, AFG, AFABS, AFSCA



def read_abundances(cells, ndust, USER):
    """
    Read abundances for the dusts. 
    If length of abundance file <=2, interpret this as constant abundance.
    """
    count = 0
    for idust in range(len(USER.file_abundance)):
        if (USER.file_abundance[idust][0]=='#'): continue
        count += 1
    if (count<1): return asarray([], float32)
    # some abundance file specified -> abundances are not constant between cells
    ABU = ones((cells, ndust), float32)
    for idust in range(len(USER.file_abundance)):
        if (USER.file_abundance[idust][0]=='#'): continue
        ABU[:,idust] =  fromfile(USER.file_abundance[idust], float32)
    return ABU
            


def read_scattering_functions(USER):
    """
    Read discrete and cumulative scattering functions.
    Input:
        USER   =  User object, should have USER.file_scafunc set
    Return:
        FDSC   =  Discrete scattering function FDSC[USER.NFREQ, USER.BINS]
        FDSC   =  Cumulative scattering function FCSC[USER.NFREQ, USER.BINS]
    Note:
        Even with abundance variations, one *can* use a single scattering function
        assuming that it is not significantly affected by the abundance variations
        -> kernel_SOC.c
        Iff multiple scattering functions are provided, 
        kernel needs to have abundance arrays for all dust species !
    """
    ndust =  len(USER.file_scafunc)   #  ndust>1 implies -D WITH_MSF = multiple scattering functions
    print("ndust=%d" % ndust)
    if ((ndust!=len(USER.file_optical))&(ndust!=1)):
        print("Must have either a single scattering function (DSC file) or one for each dust!")
        sys.exit()
    FDSC  =  zeros((ndust, USER.NFREQ, USER.DSC_BINS), float32)
    FCSC  =  zeros((ndust, USER.NFREQ, USER.DSC_BINS), float32)
    for i in range(ndust):
        print("--- SCATTERING FUNCTION: %s" % USER.file_scafunc[i])
        fp           =  open(USER.file_scafunc[i], 'rb')
        FDSC[i,:,:]  =  fromfile(fp, float32, USER.NFREQ*USER.DSC_BINS).reshape(USER.NFREQ, USER.DSC_BINS)
        FCSC[i,:,:]  =  fromfile(fp, float32).reshape(USER.NFREQ, USER.DSC_BINS)
        fp.close()
    return FDSC, FCSC
    


def OT_cut_levels(infile, outfile, maxlevel, platform=-1):
    """
    Write a new SOCAMO cloud, cutting levels>maxlevel. maxlevel=0,1,2,...
    Parent cells (old links) are replaced with average density.
    Input:
        infile   = old cloud or magnetic field file
        outfile  = new file
        maxlevel = new maximum level (0,1,2...)
        patform  = if >=0, try to use that OpenCL platform
    """
    fp = open(infile, 'rb')
    NX, NY, NZ, LEVELS, CELLS = fromfile(fp, int32, 5)
    maxlevel = min([LEVELS-1, maxlevel])
    LCELLS = zeros(LEVELS, int32)
    H = []
    for i in range(LEVELS):
        cells     = fromfile(fp, int32, 1)
        LCELLS[i] = cells
        H.append(fromfile(fp, float32, cells))
    fp.close()    
    print('OT_cut_levels: reading hierarchy with %d levels -> use %d levels' % (LEVELS, maxlevel+1))
    # Create OpenCL program to average child values into parent node
    platform, device, context, queue = None, None, None, None
    try_platforms = arange(3) 
    if (platform>=0): try_platforms = [platform,]
    for iplatform in try_platforms: 
        try:
            platform = cl.get_platforms()[iplatform]
            device   = platform.get_devices([cl.device_type.CPU, cl.device_type.GPU][USER.DEVICES=='g'])
            context  = cl.Context(device)
            queue    = cl.CommandQueue(context)
            break
        except:
            pass
    mf       = cl.mem_flags
    LOCAL    =  8
    # source   =  open(os.getenv("HOME")+"/starformation/SOC/kernel_OT_tools.c").read()
    source   =  open(os.path.dirname(os.path.realpath(__file__))+"/kernel_OT_tools.c").read()
    OPT      =  '' 
    program  =  cl.Program(context, source).build(OPT)
    AverageParent = program.AverageParent
    #                                    NP        NC        P     C
    AverageParent.set_scalar_arg_dtypes([np.int32, np.int32, None, None])
    GLOBAL   =  max(LCELLS)   # at most this many parent cells, at most this many vector elements
    P_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL)
    C_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*GLOBAL)
    #
    for i in arange(LEVELS-2, maxlevel-1, -1):  # loop over the parent levels, including maxlevel
        #  i = level of parent cells
        cl.enqueue_copy(queue, P_buf, H[i])     # parent cells -- all cells on the level
        cl.enqueue_copy(queue, C_buf, H[i+1])   # child  cells
        AverageParent(queue, [GLOBAL,], [LOCAL,], len(H[i]), len(H[i+1]), P_buf, C_buf)
        cl.enqueue_copy(queue, H[i], P_buf)     # upated values at the parent level
    # write the new file
    fp = open(outfile, 'wb')
    tmp = asarray([NX, NY, NZ, maxlevel+1, sum(LCELLS[0:(maxlevel+1)])], int32)
    tmp.tofile(fp)
    for i in range(maxlevel+1):
        print('   ---> write level %2d' % i)
        asarray([LCELLS[i],], int32).tofile(fp)
        asarray(H[i], float32).tofile(fp)
    fp.close()
    
    

def read_cloud(USER):
    """
    Read the cloud (density field).
    Input:
        USER       = User object with USER.file_cloud and USER.KDENSITY
    Return:
        NX, NY, NZ, LEVELS, CELLS, LCELLS, DENSITY
            NX, NY, NZ = root grid dimensions
            LEVELS     = levels in the hierarchy
            CELLS      = total number of levels
            LCELLS     = number of cells per hierarchy level, LCELLS[LEVELS]
            OFF        = index of first cell on each level, OFF[LEVELS]
            DENSITY    = density values of each cell, DENSITY[CELLS]
            AREA       = cloud surface area [GL^2]
            AXY, AXZ, AYZ = surface areas towards main axes
    Note:
        Densities are scaled by USER.KDENSITY.
        Field AREA, AXY, AXZ, AYZ are also updated inside USER.
    File format:
        NX, NY, NZ     --  dimensions of the root grid (float32)
        LEVELS         --  maximum number of levels (int32)
        CELLS          --  total number of cells (int32)
        for each hierarchy level:
            LCELLS     --  cells on the level (int32)
            densities  --  density values for LCELLS[level] cells on this level
        Note:
            values density<0 => interpret as index to an octet one level below
            = -(float*)&cell_index
    """
    fp = open(USER.file_cloud, 'rb')
    # Dimensions
    NX, NY, NZ, LEVELS, CELLS = fromfile(fp, int32, 5)
    print("NX %d, NY %d, NZ %d LEVELS %d, CELLS %d" % (NX, NY, NZ, LEVELS, CELLS))
    if (LEVELS>USER.LEVELS): 
        # we cut the number of hierarchy levels and write a new file for that model
        print('************************************************************')
        print('CUT LEVELS')
        print('************************************************************')
        newname = '%s.MAX%d' % (USER.file_cloud, USER.LEVELS)
        ## if (not(os.path.exists(newname))):
        OT_cut_levels(USER.file_cloud, newname, USER.LEVELS-1, USER.PLATFORM) # argument is max level
        fp.close()
        USER.file_cloud = newname
        fp = open(USER.file_cloud, 'rb')
        NX, NY, NZ, LEVELS, CELLS = fromfile(fp, int32, 5)
        print("NX %d, NY %d, NZ %d LEVELS %d, CELLS %d" % (NX, NY, NZ, LEVELS, CELLS))
    #
    LCELLS = zeros(LEVELS, int32)
    OFF    = zeros(LEVELS, int32)
    DENS   = zeros(CELLS, float32)
    TRUE_CELLS = 0
    cells      = 0
    kdensity   = USER.KDENSITY    # scaling requested in the ini file
    for level in range(LEVELS):    
        if (level>0):
            OFF[level] = OFF[level-1] + cells   # index to [CELLS] array, first on this level
        cells = fromfile(fp, int32, 1)[0]       # cells on this level of hierarchy
        print(" level %2d  cells %6d" % (level, cells))
        if (cells<0):
            break                               # the lowest level already read
        LCELLS[level] = cells    
        tmp = fromfile(fp, float32, cells)
        if (kdensity!=1.0):                     # apply density scaling
            m  = nonzero(tmp>0.0)               # not a link
            tmp[m] = clip(kdensity*tmp[m], 1.0e-6, 1e20)
        DENS[(OFF[level]):(OFF[level]+cells)] = tmp
        TRUE_CELLS += cells                     # true cells (leafs) on this level
        if (level>0):
            TRUE_CELLS -= cells/8               # this many dummy parent cells to be subtracted
    fp.close()
    USER.AREA = 2.0*(NX*NY+NY*NZ+NZ*NX) 
    USER.AXY  = (NX*NY)/float(USER.AREA)
    USER.AXZ  = (NX*NZ)/float(USER.AREA)
    USER.AYZ  = (NY*NZ)/float(USER.AREA)
    # Make sure USER.MAPCENTRE is set
    if (USER.MAPCENTRE['x']<-1e7): 
        USER.MAPCENTRE = cl.cltypes.make_float3(0.5*NX, 0.5*NY, 0.5*NZ)
    # for convenience make these accessible also via USER
    USER.LEVELS, USER.CELLS, USER.LCELLS, USER.OFF, USER.DENS = \
    LEVELS, CELLS, LCELLS, OFF, DENS
    #
    print("CLOUD READ: %d %d %d   %d  %d" % (NX, NY, NZ, LEVELS, CELLS))
    if (0): # Fix malformed cloud file
        LEVELS = len(nonzero(LCELLS>0)[0])
        LCELLS = LCELLS[0:LEVELS]
        OFF    = OFF[0:LEVELS]
    return NX, NY, NZ, LEVELS, CELLS, LCELLS, OFF, DENS



def mmap_intensity(USER, CELLS, vector=False):
    """
    Create a memory mapped array that contains output intensities (for DustEM).
    Input:
        USER     =  User object
        CELLS    = total number of cells
    Output:
        INTENSITY  = memory mapped array for intensities [cgs],
                     INTENSITY[NFREQ*CELLS], cell index runs faster
    Note:
        name of the intensity hardcoded to ISRF.DAT
    """
    if (not(USER.SAVE_INTENSITY in [1,2])): return []        # no intensity file used
    fp =  open("ISRF.DAT", "wb")
    asarray([CELLS, USER.NFREQ], int32).tofile(fp)  # put dimensions at the beginning
    # Fill with something - necessary for seeking ??
    tmp  = zeros(CELLS, float32)
    if (vector):
        for j in range(4):
            for i in range(NFREQ): 
                tmp.tofile(FP_INTENSITY)
    else:
        for i in range(NFREQ): tmp.tofile(FP_INTENSITY)
    del tmp
    if (vector):
        INTENSITY = np.memmap(filename, dtype='float32', mode="r+", shape=(USER.NFREQ, CELLS, 4), offset=8)
    else:
        INTENSITY = np.memmap(filename, dtype='float32', mode="r+", shape=(USER.NFREQ, CELLS), offset=8)
    return INTENSITY



def mmap_diffuserad(USER, CELLS):
    """
    Return a memory mapped array for input diffuse radiation field.
    Input:
        USER       =  User object
        CELLS      =  number of cells in the model
    Return:
        DIFFUSERAD = reference to the memory mapped array, DIFFUSERED[CELLS*nfreq]
                     frequency runs faster .. allow nfreq <= NFREQ
    Note:
        The file contains data in units  photons / 1 Hz / 1 cm3
    """
    # Check the dimensions
    if (len(USER.file_diffuse)<1): return []        # no diffuse radiation field
    dims = fromfile(USER.file_diffuse, int32, 2)    # should begin with CELLS, NFREQ
    if (dims[0]!=CELLS):   # allow dims[1]<=USER.NFREQ !!
        EXIT("DIFFUSERAD has %d cells but the cloud has %d cells ??" % (CELLS, dims[0]))
    if (dims[1]<USER.NFREQ):
        print("--------------------------------------------------------------------------------")
        print("DIFFUSERAD has %d frequencies <  %d of the simulation ??" % (dims[1], USER.NFREQ))
        print("   this is ok - diffuse emission simulated for %d ***highest*** frequencies" % dims[1])
        print("--------------------------------------------------------------------------------")
    # Make mmap object to access the data... frequency index runs faster
    DIFFUSERAD = np.memmap(USER.file_diffuse, dtype='float32', mode='r', shape=(CELLS, dims[1]), offset=8)
    return DIFFUSERAD
    



    
def mmap_emitted(USER, CELLS, LEVELS, LCELLS, REMIT_NFREQ, OFF, DENS):
    """
    Return memory mapped array for emitted photons.
    Input:
        USER         =  User object
        CELLS        =  number of cells in the model
        REMIT_NFREQ  =  number of frequencies (could be < USER.NFREQ)
    Note:
        2019-10-06:
            LIB_MAPS means that emitted file has only ofreq frequencies and values have 
            been calculated with A2E_LIB.py
            In that case the file should have len(USER.FSELECT) frequencies, not REMIT_NFREQ.
            The caller should set argument REMIT_NFREQ to len(USER.FSELECT) !!
    """
    EMITTED = []
    if (len(USER.file_emitted)>0): # prepare file to store emitted photons
        ok         = 1
        old_levels = 0
        old_cells  = 0
        try:
            # os.system('ls -l emitted.data')
            # os.system('ls -l /dev/shm/emitted.data')
            print("Reading emitted file %s" % USER.file_emitted)
            old_cells, old_nfreq = fromfile(USER.file_emitted, int32, 2)
            print("Emitted file has cells %d, freq %d -> now %d cells, %d freq" % (old_cells, old_nfreq,  CELLS, REMIT_NFREQ))
            if (old_nfreq!=REMIT_NFREQ):   # must have the same number of frequencies
                print("mmap_emitted %s -- number of frequencies changed %d -> %d ???" % (USER.file_emitted, old_nfreq, REMIT_NFREQ))
                ok = 0   
            if (old_cells!=CELLS):
                print("mmap_emitted %s -- number of cells changed %d -> %d ???" % \
                          (USER.file_emitted, old_cells, CELLS))
                ok = 0   
            #if (old_cells<LCELLS[0]):      # must have data at least for the first hierarchy level
            #    print("mmap_emitted %s -- number of cells %d < LCELLS[0]=%d ???" % \
            #              (USER.file_emitted, old_cells, LCELLS[0]))
            #    ok = 0
        except:
            print('============================================================================')
            print("mmap_emitted -- failed to read old file %s" % USER.file_emitted) 
            print('============================================================================')
            ok = 0

        if (ok==0): #  created/truncated
            print('*'*80)
            print('*'*80)
            print('*'*80)
            print("Emitted file TRUNCATED:  %s !!" % USER.file_emitted)
            print(" LIB_MAPS", USER.LIB_MAPS)
            print(" FSELECT",  USER.FSELECT)
            print('*'*80)
            print('*'*80)
            print('*'*80)
            fp = open(USER.file_emitted, "w+")
            fp.seek(0)
            asarray([CELLS, REMIT_NFREQ], int32).tofile(fp)
            tmp  = zeros(CELLS, float32)
            for i in range(REMIT_NFREQ):
                tmp.tofile(fp)
            del tmp
            fp.close()
        else: # in every case, make sure the header is correct
            fp = open(USER.file_emitted, "r+")
            fp.seek(0)
            asarray([CELLS, REMIT_NFREQ], int32).tofile(fp)
            fp.close()
            
        # mmap object, frequency runs faster --- note that offset is in bytes
        print("EMITTED ~ %s" % USER.file_emitted)
        EMITTED=np.memmap(USER.file_emitted, dtype='float32',mode='r+', shape=(CELLS, REMIT_NFREQ),offset=8) # header only CELLS, NFREQ
        
        # print 'EMITTED FULL RANGE', prctile(ravel(EMITTED), (0,50.0,100.0))
        # We might need to copy data to a deeper hierarchy levels (increase depth)
        if (0):
            if (old_cells<CELLS): # we did have a smaller file, now adding levels to the hierarchy ????
                print('*'*80)
                print('*'*80)
                print('EMITTED FILE CONTAINED %d CELLS, NOW %d CELLS' % (old_cells, CELLS))
                print('*'*80)
                print('*'*80)
                # assuming that file was for a smaller number of levels, count the number of old hierarchy levels
                for old_levels in range(LEVELS):
                    old_cells -= LCELLS[old_levels]
                    if (old_cells<=0): break
                print("We think that the old file had data for %d hierarchy levels" % old_levels)
                ##
                if (old_levels>0):
                    for level in range(old_levels-1, LEVELS-1): # level -> level+1
                        print('COPY EMITTED FROM LEVEL %d TO LEVEL %d' % (level, level+1))
                        a, b  =  OFF[level], OFF[level]+LCELLS[level]
                        for i in range(a, b):
                            if (DENS[i]<1e-7):                       # should have a child octet
                                j = OFF[level+1] + F2I(-DENS[i])    # index of the first child node
                                for jj in range(j, j+8):
                                    EMITTED[jj,:] = EMITTED[i,:]
                print('*'*80)

        if (0):  # commented out 2019-01-15
            # We might need to copy data to a deeper hierarchy levels (increase depth)
            if (old_cells<CELLS): # we did have a smaller file, now adding levels to the hierarchy ????
                print('*'*80)
                print('*'*80)
                print('EMITTED FILE CONTAINED %d CELLS, NOW %d CELLS' % (old_cells, CELLS))
                print('*'*80)
                print('*'*80)
                # assuming that file was for a smaller number of levels, count the number of old hierarchy levels
                for old_levels in range(LEVELS):
                    old_cells -= LCELLS[old_levels]
                    if (old_cells<=0): break
                print("We think that the old file had data for %d hierarchy levels" % old_levels)
                ##
                for level in range(old_levels-1, LEVELS-1): # level -> level+1
                    print('COPY EMITTED FROM LEVEL %d TO LEVEL %d' % (level, level+1))
                    a, b  =  OFF[level], OFF[level]+LCELLS[level]
                    for i in range(a, b):
                        if (DENS[i]<1e-7):                       # should have a child octet
                            j = OFF[level+1] + F2I(-DENS[i])     # index of the first child node
                            for jj in range(j, j+8):
                                EMITTED[jj,:] = EMITTED[i,:]
                print('*'*80)

    return EMITTED



def set_wg_sizes(USER, CELLS):
    """
    Figure out the actual number of work groups allocated for each radiation
    source and an updated number of photon packages.
    Input:
        USER   =  reference to an object of User class
        CELLS  =  total number of cells in the model
    Return:
        BGWRG, CLWRG, PSWRG:
            number of work groups for emission from background, cells, and point source(s)
        BGPAC, CLPAC, PSPAC, GLOBAL:
            number of photon packages from background, cells, and point source(s)
        LOCAL, GLOBAL:
            new values for the local and global work group sizes
        NITER:
            number of kernel calls per frequency (>=1)
    Note:
        This will modify the package numbers originally requested by the observer,
        especially if emission weighting is NOT used. Also value of GLOBAL can change.
    """
    # Set initial values for LOCAL, GLOBAL, DEVICES
    LOCAL   =     32
    GLOBAL  =  16384
    if (USER.DEVICES.find('g')<0): # only CPUs
        LOCAL  = 8     
    if (USER.LOCAL>0):             # user asked for a specific number of LOCAL
        LOCAL  = USER.LOCAL
        GLOBAL = (int)(GLOBAL/LOCAL) * LOCAL    
    if (USER.DO_SPLIT>0):
        USER.BATCH = 1
    # Start with the requested photon numbers
    BGPAC, CLPAC, PSPAC  =  USER.BGPAC, USER.CLPAC, USER.PSPAC
    PACKETS =  USER.BGPAC + USER.CLPAC + USER.PSPAC
    NITER   =  max([1, int(PACKETS/(GLOBAL*USER.BATCH))])
    print(" --- NITER %d" % NITER)
    print(" --- REQUEST  BG %9d  PS %9d  CELL %9d    =  %9d" % (BGPAC, PSPAC, CLPAC, PACKETS))
    # Calculate work group sizes under the assumption that one work group handles
    # only packages of certain origin.
    #   packets = GLOBAL*NITER*BATCH =  sum of  WRG*LOCAL*NITER*BATCH
    PSWRG   =  max([1, int(round((GLOBAL/LOCAL)*PSPAC/float(PACKETS)))])
    BGWRG   =  max([1, int(round((GLOBAL/LOCAL)*BGPAC/float(PACKETS)))])
    print(" --- BGPAC %d, PACKETS %d, ratio %.2f, GLOBAL/LOCAL %d, BGWRG %d" % \
       (BGPAC, PACKETS, BGPAC/float(PACKETS), int(GLOBAL/LOCAL), BGWRG))
    if (PSPAC<=0):  PSWRG = 0
    if (BGPAC<=0):  BGWRG = 0
    CLWRG   =  (GLOBAL/LOCAL) - PSWRG - BGWRG
    BGPAC   =  NITER*BGWRG*LOCAL*USER.BATCH
    CLPAC   =  NITER*CLWRG*LOCAL*USER.BATCH
    PSPAC   =  NITER*PSWRG*LOCAL*USER.BATCH        
    if (USER.USE_EMWEIGHT>0):
        # one can make sum(EMWEIGHT) == CLPAC for any value of CLPAC
        pass
    else:
        # decrease CLPAC until it is multiple of CELLS -- required for unweighted case
        CLPAC   =  (int(CLPAC/CELLS)) * CELLS        
    # PACKETS variable is actually not used...
    PACKETS   =  BGPAC + CLPAC + PSPAC  # only CLWRG may simulate less than full CLWRG*LOCAL*NITER*BATCH        
    print(" --- PACKETS  BG %9d  PS %9d  CELL %9d    =  %9d" % (BGPAC, PSPAC, CLPAC, PACKETS))
    print("     OUT OF      %9d     %9d       %9d    =  %9d" % (
    (BGWRG*LOCAL*NITER*USER.BATCH), (PSWRG*LOCAL*NITER*USER.BATCH),
    (CLWRG*LOCAL*NITER*USER.BATCH), (GLOBAL*NITER*USER.BATCH)))        
    print("     WG       BG %9d  PS %9d  CELL %9d    =  %9d" % (BGWRG, PSWRG, CLWRG, int(GLOBAL/LOCAL)))
    print("     GLOBAL %d, LOCAL %d, BATCH %d, NITER %d" % (GLOBAL, LOCAL, USER.BATCH, NITER))
    # Because CLPAC was made multiple of CELLS   ==>   CLPAC <= NITER*CLWRG*LOCAL*USER.BATCH
    GLOBAL = (BGWRG+PSWRG+CLWRG)*LOCAL
    print(" --- *** NOTE: GLOBAL changed to %d" % (GLOBAL))
    # assert(GLOBAL==((BGWRG+PSWRG+CLWRG)*LOCAL))
    # assert(BGPAC==(int)(BGWRG*LOCAL*NITER*BATCH))
    # assert(CLPAC<=(int)(CLWRG*LOCAL*NITER*BATCH))
    # assert(PSPAC==(int)(PSWRG*LOCAL*NITER*BATCH))
    ##if (USE_EMWEIGHT>0):
    ##    assert(CLPAC==(int)(CLWRG*LOCAL*NITER*BATCH))
    ##    assert((BGWRG+CLWRG+PSWRG)==(int)(GLOBAL/LOCAL))
    ##    assert((BGPAC+CLPAC+PSPAC)==(int)(NITER*GLOBAL*BATCH))
    # Weight for background and point source packages
    # AREA = cloud area in units GL^2  ==> effective division by GL^2
    WPS, WBG = 0.0, 0.0
    if (BGPAC>0):
        WBG   =  3.141592653589793 * float(USER.AREA) / (BGPAC*PLANCK) # divided by GL^2
    if (PSPAC>0):  
        WPS   =  1.0 / (PLANCK*PSPAC*pow(USER.GL*PARSEC, 2.0))  # divided by GL^2
    return BGWRG, CLWRG, PSWRG, BGPAC,  CLPAC, PSPAC, LOCAL, GLOBAL, NITER, WBG, WPS



def read_background_intensity(USER):
    """
    Read background intensities at the simulated frequencies.
    Input:
        USER   = User object
    Return:
        IBG    = array of background intensities or []
    """
    IBG = []
    if (USER.BGPAC>0):
        try:
            IBG  = fromfile(USER.file_background, float32, USER.NFREQ)
        except:
            print('Not reading file USER.file_BG=[%s]' % USER.file_background)
            IBG  = zeros(USER.NFREQ, float32)  # probably because we will use HPBG instead
        if (len(IBG)!=USER.NFREQ):
            EXIT('Optical data for %d, background intensity for %d frequencies ??' % (USER.NFREQ, len(IBG)))
        IBG *= USER.scale_background
    return IBG




def read_source_luminosities(USER):
    """
    Read source luminosities.
    Input:
        USER  =  User object with file names in USER.file_pointsource (if USER.PSPAC>0)
    Return:
        LPS   =  source luminosities LPS[source, USER.NFREQ] or [] if no point sources
    """
    print("read_source_luminosities: %d point sources" % USER.NO_PS)
    if (USER.NO_PS<1): return []  # no point sources
    LPS = zeros((USER.NO_PS, USER.NFREQ), float32)
    for i in range(USER.NO_PS):
        tmp = fromfile(USER.file_pointsource[i], float32, USER.NFREQ)
        if (len(tmp)!=USER.NFREQ):
            EXIT('Source %d: optical data for %d, intensity for %d frequencies ??' % 
            (i, USER.NFREQ, len(LPS)))
        LPS[i,:] = tmp * USER.PS_SCALING[i]
    return LPS




def set_observer_directions(USER, new_order=True):
    """
    Calculate basic vectors used to calculate position -> map pixel.
    Input:
        USER   =  User object with necessary input parameters
    Output:
        NDIR  = number of observer directions and
        ODIR, RA, DE = arrays of float3 with vectors towards the 
                       observers and orthogonal vectors in the map plane.
    """
    NDIR = len(USER.OBS_THETA)
    if (NDIR==0):
        USER.OBS_THETA = [ 0.5*pi, ]
        USER.OBS_PHI   = [ 0.0,    ]
        NDIR           = 1
    ODIR  = zeros(NDIR, clarray.cltypes.float3)
    RA    = zeros(NDIR, clarray.cltypes.float3)
    DE    = zeros(NDIR, clarray.cltypes.float3)
    # 2019-04-12 the old system of making (RA,DE) axes could not be made consistant
    #            with healpix background
    #  new system makes sure we have consistent definitions of the model cube,
    #  background Healpix sky, and the projected maps.
    #  (ODIR, RA, DE) are a rotation of (x, y, z)  ---- we keep variable name RA but that is
    #  now a coordinate that in the final maps increases towards the  **right**
    #  ini-file specifies THETA, PHI for the direction towards the observer
    #  such that (THETA,PHI)=(0,0) means that the observer is in the direction of the +Z axis
    #  and (THETA,PHI)=(90,90) means that the observer is in the direction of the +Y axis
    #  =>   ODIR = Ry(LAT)*Rz(LON)*[1,0,0]
    #  (odir, ra, de) =  R_Z(lon) R_z(phi) (x, y, z)
    R = zeros((3,3), float32)
    for i in range(NDIR):
        b, a   =   0.5*pi-USER.OBS_THETA[i], USER.OBS_PHI[i] # b is latitude, not angle from +Z
        R[0,:] = [ cos(a)*cos(b), -sin(a), -cos(a)*sin(b) ]
        R[1,:] = [ sin(a)*cos(b),  cos(a), -sin(a)*sin(b) ]
        R[2,:] = [ sin(b),         0.0,     cos(b)        ]
        x      = matmul(R, [1,0,0])
        ODIR[i][0], ODIR[i][1], ODIR[i][2] = x[0], x[1], x[2]
        x      = matmul(R, [0,1,0])
        RA[i][0], RA[i][1], RA[i][2]  = x[0], x[1], x[2]
        x      = matmul(R, [0,0,1])
        DE[i][0], DE[i][1], DE[i][2]  = x[0], x[1], x[2]

        if (fabs(ODIR[i][0])<1.0e-5): ODIR[i][0]=1.0e-5
        if (fabs(ODIR[i][1])<1.0e-5): ODIR[i][1]=1.0e-5
        if (fabs(ODIR[i][2])<1.0e-5): ODIR[i][2]=1.0e-5
           
    if (0):
        for i in range(NDIR):
            print("==========================================================================================")
            print("LON   %.0f,    LAT  %.0f" % (USER.OBS_THETA[i]*RADIAN_TO_DEGREE, USER.OBS_PHI[i]*RADIAN_TO_DEGREE))
            print("ODIR  %8.4f %8.4f %8.4f" % (ODIR[i][0], ODIR[i][1], ODIR[i][2]))
            print("RA    %8.4f %8.4f %8.4f" % (RA[i][0], RA[i][1], RA[i][2]))
            print("DE    %8.4f %8.4f %8.4f" % (DE[i][0], DE[i][1], DE[i][2]))
            print("==========================================================================================")
    return NDIR, ODIR, RA, DE




def opencl_init(USER, verbose=False):
    """
    Initialise OpenCL environment.
    Input:
        USER      =  User object, including USER.DEVICES ('c'=CPU, 'g'=GPU)
        platform  =  index of selected OpenCL platform (default=0)
    Return:
        context   =  array of compute contexts, one per device
        commands  =  array of command queues, one per device
    """
    platform         = []
    devi             = []
    context          = []  # for each device
    commands         = []
    try_platforms    = arange(5)
    if (USER.PLATFORM>=0): try_platforms = [USER.PLATFORM,]
    for dc in USER.DEVICES:
        plat, devi, cont, queue = None, None, None, None
        for iplatform in try_platforms:
            if (verbose):  print("DEVICE %s, TRY PLATFORM %d" % (dc, iplatform))
            try:
                platform  = cl.get_platforms()[iplatform]
                if (dc=='g'):
                    devi  = [ platform.get_devices(cl.device_type.GPU)[USER.IDEVICE] ]
                else:
                    devi  = [ platform.get_devices(cl.device_type.CPU)[USER.IDEVICE] ]
                ###
                if (USER.FISSION>0):
                    print("FISSION %d !" % USER.FISSION)
                    print(devi[0])
                    dpp   =  cl.device_partition_property
                    devi  =  [devi[0].create_sub_devices( [dpp.EQUALLY, USER.FISSION] )[0],]
                    print('  --> ')
                    print(devi[0])                    
                ###
                cont  = cl.Context(devi)
                queue = cl.CommandQueue(cont)
                if (verbose): print("context ", cont)
                break
            except:
                pass
        # for iplatform
        context.append(cont)
        commands.append(queue)
    # --- end creating kernels
    if (verbose):
        print("opencl_init:")
        print(" platform ", platform)
        print(" device   ", devi)
        print(" context  ", context)
    return context, commands



def get_program(context, commands, sourcefile, options):
    """
    Build kernel codes and return programs for all devices.
    Input:
        context    = OpenCL context
        commands   = OpenCL command queue
        sourcefile = kernel source code
        options    = compiler options 
    """
    source  = open(sourcefile).read() 
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    program = []
    for ID in range(len(context)):
        if (1): #try:
            program.append( cl.Program(context[ID], source).build(options) )
        else:
            print("*** Error -- failed to create program (1)")
            sys.exit()        
    return program
    


def temperature_to_emitted(USER, EMITTED):
    """
    Use previously stored temperature to calculate values to EMITTED.
    Input:
        USER     =  User object
        EMITTED  =  array of emitted photons [CELLS*REMIT_FREQ]
    Return:
        EMITTED updated in place.
    Note:
        This assumes that there is only a single dust component
        => USER.FABS[0] is being used
    """
    print("*** temperature_to_emitted ***")
    if (len(USER.FABS)>1):
        print("*** Error: temperature_to_emitted requires that one has a single dust component")
        sys.exit(0)
    TNEW  =  zeros(USER.CELLS, float32)
    fp    =  open(USER.file_temperature, "rb")
    fromfile(fp, int32, 4)  # skip header ...
    for level in range(USER.LEVELS):
        print("level %d --- lcells %d" % (level, itmp))
        itmp      =  fromfile(fp, int32, 1)
        a, b      =  USER.OFF[level], USER.OFF[level]+USER.LCELLS[level]
        TNEW[a:b] =  fromfile(fp, float32, USER.LCELLS[level])
    fp.close()
    # sanitize temperature values
    m       = nonzero(USER.DENS>1.0e-10)
    TNEW[m] = clip(TNEW[m], 6.0, 35.0)
    for icell in m[0]:       # loop over real cells
        a, b = USER.REMIT_I1, USER.REMIT_I2+1
        # EMITTED[CELLS, REMIT_NFREQ]
        EMITTED[icell,:] =  (FACTOR*4.0*np.pi/(PLANCK*USER.FFREQ[a:b])) *  \
        USER.FABS[0][a:b]*Planck(USER.FFREQ[a:b], TNEW[icell])/(USER.GL*PARSEC)   # 1e20*photons/H -> FACTOR photons/H
    del TNEW
    del m
    

    
def temperature_to_emitted_inmem(USER, TNEW, EMITTED):
    """
    Use temperatures in TNEW to calculate values to EMITTED.
    Input:
        USER     =  User object
        TNEW     =  array of dust temperatures
        EMITTED  =  array of emitted photons [CELLS*REMIT_FREQ]
    Return:
        EMITTED updated in place.
    Note:
        This assumes that there is only a single equilibrium dust component.
    """
    print("*** temperature_to_emitted_inmem ***")
    if (len(USER.FABS)>1):
        print("*** Error: temperature_to_emitted_inmem requires that one has a single dust component")
        sys.exit(0)
    m       = nonzero(USER.DENS>1.0e-10)
    print("NUMBER OF CELLS TO COMPUTE: %d" % len(m[0]))
    print("TNEW HAS NUMBER OF CELLS:   %d" % len(TNEW))
    print("REMIT HAS FREQUENCIES:      %d - %d = %d freq" % (USER.REMIT_I1, USER.REMIT_I2, USER.REMIT_I2 - USER.REMIT_I1 + 1))
    print("USER.FFREQ has %d frequencies " % len(USER.FFREQ))
    print("USER.FABS[0] has %d frequencies  " % len(USER.FABS[0]))
    for icell in m[0]:       # loop over real cells
        a, b              =  USER.REMIT_I1, USER.REMIT_I2+1
        EMITTED[icell,:]  =  (FACTOR*4.0*np.pi/(PLANCK*USER.FFREQ[a:b])) *  \
        USER.FABS[0][a:b]*Planck(USER.FFREQ[a:b], TNEW[icell])/(USER.GL*PARSEC)   # 1e20*photons/H -> FACTOR
    del TNEW
    del m
    print("*** temperature_to_emitted_inmem - done ! ***")
    
    
    
    
    
def read_otfile(filename):
    """
    Read a file in the hierarchical data format and return the values as a single vector.
    Input:
        filename   = name of a file in OT file format
    Return:
        vector of values val[cells]
    Note:
        Most files (density, B, temperature) use this format.
        starts with:   NX, NY, NZ, LEVELS, CELLS   (all int32)
        each level:    LCELLS (int32) VAL[LCELLS] (float32)
    """
    fp  = open(filename, "rb")
    nx, ny, nz, levels, cells = fromfile(fp, int32, 5)
    print("read_otfile %s --- dims %d %d %d --- levels %d cells %d" % (filename, nx, ny, nz, levels, cells))
    a  = 0
    val = zeros(cells, float32)
    for level in range(levels):
        lcells = fromfile(fp, int32, 1)[0]   # cells on this level
        val[a:(a+lcells)] = fromfile(fp, float32, lcells) # cell values, current level
        a     += lcells
    fp.close()
    return val
    


def escape_probability(tau):
    """
    Escape probability can be approximated with
    beta =   A*exp(-B*tau) + (1.0-A)*exp(-C*tau) with
    [A, B, C ] = [ 0.41960922  0.11793479  0.66852746]    
    """
    return 0.41960922*exp(-0.11793479*tau) + (1.0-0.41960922)*exp(-0.66852746*tau)



def calculate_beta_vs_tau_T(FREQ, ABS):
    """
    Calculate effective beta averaged over modified blackbody,
    for the given dust model and a grid of (tau, T) values.
    Return interpolation object.
    """
    T     = logspace(log10(7.0), log10(1600.0), 59)              # along columns
    TAU   = logspace(-2, 2.01, 91)              # along rows
    TAU  -= 0.01
    BETA  = zeros((len(T), len(TAU)),float32)  # BETA[iT, itau]
    for iT in range(len(T)):
        for itau in range(len(TAU)):
            # beta =  Int [ beta * B ] df / Int [ B ] df
            tau     =  TAU[itau] * ABS / ABS[-1]   # optical depth in each channel, tau==last channel
            A       =  Planck(FREQ, T[iT])
            B       =  escape_probability(tau) * A
            BETA[iT, itau]  =  Trapezoid(FREQ, B) / Trapezoid(FREQ, A)
    ip =  RectBivariateSpline(T, TAU, BETA, kx=3, ky=3, s=0)    
    ###
    if (0):
        clf()
        subplot(221)
        imshow(BETA, vmin=0.90)
        colorbar()            
        tmp = meshgrid(T, TAU)
        BBETA = 0.0*BETA
        for iT in range(len(T)):
            for itau in range(len(TAU)):
                BBETA[iT, itau] = ip(T[iT]*1.03, TAU[itau]*0.97)    
        subplot(222)
        imshow(BBETA, vmin=0.90)
        colorbar()    
        subplot(223)
        for i in range(100):
            t    =  exp(0.01 + rand()*3.5)
            tau  =  exp(0.01 + rand()*3.0)
            taus =  tau*ABS/ABS[-1]
            A    =  ABS * Planck(FREQ, t)
            B    =  escape_probability(taus) * A
            tmp  =  Trapezoid(FREQ, B) / Trapezoid(FREQ, A)
            plot(tmp, ip(t, tau), 'x')            
        show()
        sys.exit()
    ###
    return ip

    

def Fix(n, l):
    # Return smallest integer >=n that is divisible with l
    return int(int(floor((n+l-1)/l))*l)



def values_down_in_hierarchy(OFF, LCELLS, H, X, Lparent):
    """
    Copy values down in the hierarchy structure.
    Input:
        OFF     =  offsets, first cell on each hierarchy level
        LCELLS  =  array with the number of cells on each level
        H       =  the hierarchy (vector of variable values and links to child cells;
                   used here only to provide link information, X is a different array)
        X       =  the hierarchy to be modified; values for parent level Lparent will 
                   be copied to child cells on the level Lparent+1
        Lparent =  the hierarchy level corresponding to the X vector
    Return:
        Updated X array, values on Lparent+1 are copies of the parent values
        on the hierarchy level Lparent
    """
    print("values_down_in_hierarchy from parent level %d" % Lparent)
    print("H has %d cells, X has %d cells, LCELLS has %d levels" % (len(H), len(X), len(LCELLS)))

    for i in range(LCELLS[Lparent]):
        pind = OFF[Lparent]+i                # index of the parent
        if (H[pind]<0.0):                    # this is a link
            cind = OFF[Lparent+1]+ F2I(-H[pind])             # index of first child cell
            X[cind:(cind+8)] = X[pind]       # copy values
            # 'print 'parent %6d has child %6d  -- value %.3e' % (pind, cind, X[pind])

            

            
def AnalyseExternalPointSources(NX, NY, NZ, PSPOS, NO_PS, PS_METHOD):
    """
    For PS_METHOD==2
    Analyse external point sources and precalculate things like the visible projected areas
    of the cloud sides:
        XPS_NSIDE, XPS_SIDE, XPS_AREA = AnalyseExternalPointSources(NX, NY, NZ, USER.PSPOS)
    Input:
        NX, NY, NZ    =  cloud dimensions (in units of the root grid cell size)
        PSPOS         =  positions of the point sources (float3),
        NO_PS         =  number of point sources
    Returns:
        XPS_NSIDE     =  number of visible cloud surfaces (1-3), zero fro internal sources
        XPS_SIDE      =  indices of the visible sides, +X, -X, +Y, -Y, +Z, -Z ~ values 0-5
                         XPS_NSIDE[3*NO_PS]
        XPS_AREA      =  corresponding to XPS_SIDE, the projected visible surface area of
                         each of the visible cloud sides, XPS_AREA[3*NO_PS]
    Todo:
        Real XPS_AREA are not yet calculated!
    """
    XPS_NSIDE = zeros(NO_PS, int32)       # number of cloud sides visible from the source
    XPS_SIDE  = zeros(3*NO_PS, int32)     # indices[3] of the cloud sides
    XPS_AREA  = zeros(3*NO_PS, float32)   # relative solid angle, each side visible from the source
    axis      = zeros(3, float32)         # centre axis for the cone for PS_METHOD==5
    for i in range(NO_PS):
        if ((PSPOS[i][0]>=0.0)&(PSPOS[i][0]<=NX)&(PSPOS[i][1]>=0.0)&(PSPOS[i][1]<=NY)&(PSPOS[i][2]>=0.0)&(PSPOS[i][2]<=NZ)):
            # this one is inside the model volume
            continue
        # check which sides are visible from the source... 1-3 sides are possible
        no = 0
        # +X ?
        if (PSPOS[i][0]>NX):   # +X cloud side is illuminated
            XPS_SIDE[3*i+no] =   0
            XPS_AREA[3*i+no] = 1.0                # === REAL PROJECTED AREAS NOT YET CALCULATED ===
            axis             = array([-1.0, 0.0, 0.0], float32)
            no += 1
        if (PSPOS[i][0]<0.0):  # -X side is illuminated
            XPS_SIDE[3*i+no] =   1
            XPS_AREA[3*i+no] = 1.0
            axis             = array([+1.0, 0.0, 0.0], float32)
            no += 1
        if (PSPOS[i][1]>NY):   # +Y cloud side is illuminated
            XPS_SIDE[3*i+no] =   2
            XPS_AREA[3*i+no] = 1.0
            axis             = array([0.0, -1.0, 0.0], float32)
            no += 1
        if (PSPOS[i][1]<0.0):  # -Y side is illuminated
            XPS_SIDE[3*i+no] =   3
            XPS_AREA[3*i+no] = 1.0
            axis             = array([0.0, +1.0, 0.0], float32)
            no += 1
        if (PSPOS[i][2]>NZ):   # +Z cloud side is illuminated
            XPS_SIDE[3*i+no] =   4
            XPS_AREA[3*i+no] = 1.0
            axis             = array([0.0, 0.0, -1.0], float32)
            no += 1
        if (PSPOS[i][2]<0.0):  # -Z side is illuminated
            XPS_SIDE[3*i+no] =   5
            XPS_AREA[3*i+no] = 1.0
            axis             = array([0.0, 0.0, +1.0], float32)
            no += 1
        ###
        XPS_NSIDE[i] = no
        XPS_AREA[(3*i):(3*i+3)] /= XPS_NSIDE[i]   # === REAL PROJECTED AREAS NOT YET CALCULATED ===
    ###
    if (PS_METHOD==5):
        # Use XPS_SIDE[3*i] to indicate the main illuminated side and put to XPS_AREA the
        #    opening angle in radians for the cone that contains the cloud.
        # At the moment we leave XPS_SIDE[3*i] to the value set above -- should be ok but may not
        #    be optimal, i.e. the cone might be smaller if one selected another axis for it.
        #    Current value is already optimal if only one side is visible from the source.
        # Find the maximum angle between the selected cone centre axis *axis* and the vector from the point 
        #   source to any surface position of the cloud. One can check the eight corner positions of 
        #   the cloud -- the largest angle will cover all of the cloud. Opening angle in radians
        #   is put into XPS_AREA[3*i]
        for i in range(NO_PS):
            cos_theta = 0.5*pi
            for ii in range(8):  # loop over the eight corner positions
                vec    =  zeros(3, float32)
                vec[0] =  NX*(ii%2==0)        - PSPOS[i][0]
                vec[1] =  NY*((ii/2)%2==0)    - PSPOS[i][1]
                vec[2] =  NZ*((ii/4)%2==0)    - PSPOS[i][2]
                tmp    =  fabs(dot(axis, vec)) / np.linalg.norm(vec)
                print("    CORNER %d:   COS_THETA %.2f" % (ii, tmp))
                cos_theta  =  min([cos_theta, tmp])
            XPS_AREA[3*i] = cos_theta
            print("Point source %d, side %d, cos_theta = %.3f" % (i, XPS_SIDE[3*i], XPS_AREA[3*i]))
            
    if (0):
        for i in range(NO_PS):
            print('Point source: %d, XPS_NSIDE %d'% (i, XPS_NSIDE[i]))
            print("   SIDE %d,  AREA %8.5f" % (XPS_SIDE[3*i+0], XPS_AREA[3*i+0]))
            print("   SIDE %d,  AREA %8.5f" % (XPS_SIDE[3*i+1], XPS_AREA[3*i+1]))
            print("   SIDE %d,  AREA %8.5f" % (XPS_SIDE[3*i+2], XPS_AREA[3*i+2]))
        #sys.exit()
    return XPS_NSIDE, XPS_SIDE, XPS_AREA




def AnalyseExternalPointSourcesHealpix(NX, NY, NZ, PSPOS, NO_PS):
    """
    For PS_METHOD==3
    Prepare for each external point source a Healpix map that shows which part of the sky
    the cloud fills -> send only packages in the cloud direction:    
        XPS_NSIDE, XPS_SIDE, XPS_AREA = AnalyseExternalPointSourcesHealpix(NX, NY, NZ, USER.PSPOS, int(USER.NO_PS))
    Input:
        NX, NY, NZ    =  cloud dimensions (in units of the root grid cell size)
        PSPOS         =  positions of the point sources (float3),
        NO_PS         =  number of point sources
    Returns:
        XPS_NSIDE     =  [NSIDE, ], resolution of the map and the look-up table on cumulative probability
        XPS_SIDE      =  ipix[12*NSIDE*NSIDE] = lookup table to sample the Healpix pixels
        XPS_AREA      =  relative weight given for each Healpix pixel, <>==1.0
    Note:
        Currently using fixed values: NSIDE=64, NPIX=12*NSIDE*NSIDE=49152
    NOT IMPLEMENTED:
        the problem is how to find the healpix pixels that cover the cloud and, on the other hand,
        how to efficiently select random positions that uniformly cover a Healpix pixel
    NOTE:
        One could do this automatically by having in the kernel an array of hits per Healpix pixel.
        First frequency would have to assume isotropic emission but for the second frequency one can
        already use some weighting according to the hits. Importance sampling would use
        the cumulative probability to select the next Healpix pixel to be targeted.
        The weighting array could even be stored to disc for further simulations with the same
        geometric configuration.
    """
    print("*** Error: AnalyseExternalPointSourcesHealpix() not implemented !!!") 
    sys.exit()
    
    

    
def WriteSampleIni(filename):
    fp = open(filename, 'w')
    fp.write('cloud         ambi_soc.cloud   # --> file containing the cloud model\n')
    fp.write('optical       my.dust          # --> file containing the dust optical properties\n')
    fp.write('dsc           my.dsc 2500      # --> file containing the scattering function\n')
    fp.write('background    bg_intensity.bin # --> file containing background intensity for simulated frequencies\n')
    fp.write('gridlength    0.01             # size of the root-grid cell in the cloud model [pc]\n')
    fp.write('density       1.0              # scaling of the density values read from the cloud file\n')
    fp.write('iterations    1                # number of iterations\n')
    fp.write('seed         -1.0              # seed for the random number generator (<0 mean random seed)\n')
    fp.write('absorbed      absorbed.dat     # file to store computed absorbed energy (all cells and frequencies)\n')
    fp.write('emitted       emitted.dat      # file to store the dust emission (all cells and frequencies)\n')
    fp.write('bgpackets     999999           # number of photon packages sent from the background per freq. \n')
    fp.write('mapping       64 64 1.0        # pixels NX and NY in the output map and pixel size in root-grid units\n')
    fp.write('directions    0.01  0.01       # direction (theta, phi) towards the observer [deg] [deg]\n')
    fp.write('prefix        soc              # prefix for output files \n')
    fp.write('CLT                            # solve dust temperatures on device (recommended!)\n')
    fp.write('CLE                            # solve dust emission on device (recommended!)\n')
    fp.write('# singleabu                    # for case with two dust populations with abundances x and 1-x (saves space)\n')
    fp.write('# emweight    1                # use weighted sampling for the volume emission (value 1 or 2)\n')
    fp.write('# remit       10.0  1000.0     # limit simulation of re-emitted photons to some wavelength range [um]\n')
    fp.write('# ffs         1                # turn forced first scattering on (1=default) or off (=0)\n')
    fp.write('# cellpackets 999999           # number of photon packages simulated from the medium\n')
    fp.write('# diffuse     diffuse.bin      # read additional diffuse emission from the given file\n')
    fp.write('# diffpack    999999           # number of photon packages simulated from a diffuse source filling the cloud\n')
    fp.write('# hpbg        hp.bin 1.0 1     # Healpix file for background radiation, optional scaling, optional weighted sampling\n')
    fp.write('# cload       some.bin         # load additional absorptions from an external file\n')
    fp.write('# pointsources 64.0 64.0 64.0  ps_intensity.bin 1.0 # add a point source to the model\n')
    fp.write('# pspackets   999999           # number of photon packages to simulate from point sources\n')
    fp.write('# loadtemp    T.save           # load dust temperatures (e.g. non-stochastic, to recompute emission based on smaller T file\n')
    fp.write('# csave       some.bin         # save absorptions due to constant sources (not re-emission) to a file\n')
    fp.write('# saveint     1                # save intensities to a file, argument 1, 2, or 3 (see online documentation)\n')
    fp.write('# savetau     tau.bin 0.55     # save optical-depth map, second argument wavelength [um], if missing, save column density\n')
    fp.write('# nosolve                      # do not solve dust temperatures and emission (only save absorptions)\n')
    fp.write('# simum       0.1 5000.0       # limit radiation field simulation to wavelength range [um]\n')
    fp.write('# wavelength  0.1 5000.0       # limit wavelengths in the output spectrum files [um]\n')
    fp.write('# noabsorbed                   # do not save absorptions (e.g. if emission solved within SOC)\n')
    fp.write('# nomap                        # skip the writing of the map files\n')
    fp.write('# nosolve                      # skip the solving of dust emission\n')
    fp.write('# temperature T.soc            # save temperatures to the specified file (non-stochastic heating only)\n')    
    fp.write('# device      g                # select OpenCL device, c for CPU, g for GPU  \n')
    fp.write('# platform    0                # select one OpenCL platform (check the order with clinfo)\n')
    fp.write('# local       1                # override the default OpenCL local work grounp size\n')
    fp.write('\n')
    fp.write('# --> The first four lines refer to files that must be prepared *before* running the SOC program')
    fp.write('# --> see http://www.interstellarmedium.org/radiative_transfer/soc/')
    fp.close()
    print("--------------------------------------------------------------------------------")
    print("A sample SOC ini file has been written to: %s" % filename)
    print("--------------------------------------------------------------------------------")

    
    

def MakeFits(lon, lat, pix, m, n, freq=[], sys_req='fk5'):
    """
    Make an empty fits object.
    Inputs:
        lon, lat  =  centre coordinates of the field [degrees]
        pix       =  pixel size [radians]
        m, n      =  width and height in pixels
        freq      =  vector of frequencies, len(freq)>1  => 3d instead of 2d image
        sys_req   =  coordinate system, 'fk5' or 'galactic'
    """
    import astropy.io.fits as pyfits
    nchn      =  max([1, len(freq)])
    if (nchn>1):
        print("*** CUBE ***")
        A     =  zeros((nchn, n, m), float32)
    else:
        print("*** 2D ***")
        A     =  zeros((n, m), float32)
    hdu       =  pyfits.PrimaryHDU(A)
    F         =  pyfits.HDUList([hdu])
    F[0].header.update(CRVAL1 =  lon)
    F[0].header.update(CRVAL2 =  lat)
    F[0].header.update(CDELT1 = -pix*180.0/pi)
    F[0].header.update(CDELT2 =  pix*180.0/pi)
    F[0].header.update(CRPIX1 =  0.5*(m+1)+0.5) # offset 0.5 => ds9 map centre == lon
    F[0].header.update(CRPIX2 =  0.5*(n+1)+0.5) # offset 0.5 => ds9 map centre == lat
    if (sys_req=='galactic'):
        F[0].header.update(CTYPE1   = 'GLON-TAN')
        F[0].header.update(CTYPE2   = 'GLAT-TAN')
        F[0].header.update(COORDSYS = 'GALACTIC')
    else:
        F[0].header.update(CTYPE1   = 'RA---TAN')
        F[0].header.update(CTYPE2   = 'DEC--TAN')
        F[0].header.update(COORDSYS = 'EQUATORIAL')
        F[0].header.update(EQUINOX  = 2000.0)
    if (nchn>1):
        F[0].data = zeros((nchn, n, m), float32)
        F[0].header['NAXIS' ] =  3
        F[0].header['NAXIS3'] =  nchn
        F[0].header['CRPIX3'] =  1
        F[0].header['CRVAL3'] =  0.0
        F[0].header['CDELT3'] =  1
        F[0].header['CTYPE3'] = 'channel'
        # add frequency values as comments
        for i in range(nchn):
            F[0].header.add_comment('F[ %3d ] = %.4e' % (i, freq[i]))
    else:
        F[0].data = zeros((n, m), float32)
    return F
    
