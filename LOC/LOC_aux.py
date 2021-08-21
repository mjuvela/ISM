import  os, sys
import  numpy as np
from    matplotlib.pylab import *
import  pyopencl as cl
import  pyopencl.array as cl_array
#import  healpy
from    scipy.interpolate import interp1d
import  pickle
## from   MoleculeO import *


C_LIGHT           =  2.99792458e10
AMU               =  1.6605e-24 
H_K               =  4.799243348e-11
BOLTZMANN         =  1.3806488e-16
STEFAN_BOLTZMANN  =  5.670373e-5
PLANCK            =  6.62606957e-27 
GRAV              =  6.673e-8
PARSEC            =  3.0857e18
ELECTRONVOLT      =  1.6022e-12
AU                =  149.597871e11
RSUN              =  6.955e10
MSUN              =  1.9891e33
ARCSEC_TO_RADIAN  =  4.8481368e-06




class MoleculeO:    
    WEIGHT       =  0.0
    TRANSITIONS  =  0
    LEVELS       =  0
    A            =  []
    B            =  []        
    BB           =  []
    E            =  []
    F            =  []
    G            =  []
    GG           =  []
    TRANSITION   =  []
    LEVEL        =  []
    NAME         =  ''
    PARTNERS     =  0

    
    def L2T(self, upper, lower):
        """
        Returns the index of the transition
        returns value<0 if not a valid transition
        NOTE: nothing can be assumed about the order in which transitions are
              stored in TRANSITION-array                 
        """
        for tr in range(self.TRANSITIONS):
            if ((self.TRANSITION[tr,0]==upper)&(self.TRANSITION[tr,1]==lower)):
                return tr
        return -1    
    
    
    def T2L(self, tr):
        """
        Return level indices (upper, lower) for the transition number tr
        Returns -1 if transition not found.
        """
        if ((tr<0)|(tr>=self.TRANSITIONS)): return -1
        return self.TRANSITION[tr,:]
    
    
    def C(self, upper, lower, Tkin, partner):
        """
        Return collision coefficient upper -> lower for given kinetic temperature Tkin and
        collision partner with index partner. Note, Tkin can be a vector.
        """
        if (upper==lower): return 0.0
        if (partner>=self.PARTNERS):
            print("MoleculeO.C() called for partner %d, only %d partners exist" % (partner, self.PARTNERS))
        c   = self.CC[  partner]    # collision coefficients upwards [:, ntkin]
        ul  = self.CUL[ partner]    # each row = (upper, lower), already with 0-offset values
        T   = self.TKIN[partner]    # Tkin array for this partner
        # the input file contains rates upper -> lower
        u, l =  max([upper, lower]), min([upper,lower])  # real upper and lower
        m    =  nonzero((ul[:,0]==u)&(ul[:,1]==l))       # find the row in C array
        if (len(m[0])<1): return 0.0
        ip   =  interp1d(T, c[m[0][0],:])                # interpolation over temperature
        try:
            res  =  ip(clip(Tkin, T[0], T[-1]))
        except:
            print("MOL.C failed --- TKIN defined %.1f-%.1f" % (T[0], T[-1]))
            print(Tkin)
            sys.exit()
        if (upper<lower):  # actually upward transition => calculate based on stored downward rates
            #  Clu = Cul  gu/gl  exp(-(Eu-El)/kT),  l="upper", u="lower" !
            res *= (self.G[lower]/self.G[upper]) * np.exp(-H_K*(self.E[lower]-self.E[upper])/Tkin)
        return res
    
    
    def Read(self, filename):
        """
        Read Lamda description of the molecule.
        """
        fp           = open(filename, 'r')
        fp.readline()                                        # comment
        self.NAME    =  fp.readline().split()[0].replace(',','')    # name
        fp.readline()                                        # comment
        self.WEIGHT  =  float(fp.readline().split()[0])      # weight
        fp.readline()                                        # comment
        self.LEVELS  =  int(fp.readline().split()[0])        # number of energy levels
        tmp = fp.readline()                                  # comment
        d            =  loadtxt(fp, max_rows=self.LEVELS, usecols=(0,1,2))    # energy table
        self.G       =  asarray(d[:,2].copy(), float32)
        self.E       =  asarray(C_LIGHT*d[:,1], float32)     # [Hz], for each level
        fp.readline()                                        # comment
        self.TRANSITIONS = int(fp.readline().split()[0])     # number of transitions
        fp.readline()                                        # comment
        d            =  loadtxt(fp, max_rows=self.TRANSITIONS, usecols=(0,1,2,3,4)) # Einstein A array
        d.shape      =  (self.TRANSITIONS, 5)
        self.TRANSITION = asarray(d[:,1:3], np.int32)-1
        self.A       =  asarray(d[:,3], float32)
        self.F       =  asarray(d[:,4]*1.0e9, float32)
        fp.readline()                                        # comment
        self.PARTNERS = int(fp.readline().split()[0])        # number of collisional partners
        self.PNAME    = []
        self.TKIN     = []
        self.CC       = []
        self.CUL      = []    # for C array, upper and lower levels
        self.CABU     = []    # abundances of collisional partners
        for ipartner in range(self.PARTNERS):
            # one partner =  one entry in  { PNAME, TKIN, CUL, CC }
            fp.readline()                                    # !COLLISIONS BETWEEN
            self.PNAME.append(fp.readline().split()[1])      # name ?
            fp.readline()                                    # comment
            nc            = int(fp.readline().split()[0])    # number of transitions
            fp.readline()                                    # comment
            line          = fp.readline()
            nt            = int(line.split()[0])             # number of temperatures
            # can have optional abundance OF COLLISIONAL PARTNER after the temperature value !!!
            try:
                cabu = float(line.split()[1])
                self.CABU.append(cabu)
            except:
                self.CABU.append(1.0/self.PARTNERS)
                print("*** WARNING -- molecule file did not contain abundances of collisional")
                print("               partners, abundance set to %.3f !!!!!!!!!!!!!!!!!!!!!!!" % (1.0/self.PARTNERS))
            fp.readline()                                      # comment
            T             = loadtxt(fp, max_rows=1)            # Tkin vector
            self.TKIN.append(ravel(asarray(T, np.float32)))   
            fp.readline()                                      # comment
            d             = loadtxt(fp, max_rows=nc)           # the array
            if (len(d.shape)==1): d.shape = (1, len(d))
            self.CUL.append( asarray(d[:,1:3], np.int32)-1 )   # (upper, lower) --- zero offset values
            self.CC.append(  asarray(d[:,3: ], np.float32))    # collisional coefficients
            # next line is again "!COLLISIONS BETWEEN"
        ###
        self.CABU = asarray(self.CABU, np.float32)
        self.TKIN = asarray(self.TKIN, np.float32)
        self.Init()
        if (0):
            for t in range(self.TRANSITIONS):
                u, l = self.T2L(t)
                print("  %2d -> %2d   F= %12.4e   E= %12.4e   A= %12.4e" % (u, l, self.F[t], self.E[u], self.A[t]))
            print(self.CUL[0])
            print(self.CC[0])
            sys.exit()
            
        
    def Partition(self, u, T):
        """
        Return partition function value
        """
        # if ((u<0)|(u>=self.LEVELS)): return 0.0
        return (self.G[u]*np.exp(-H_K*self.E[u]/T))  /  sum(self.G[:]*np.exp(-H_K*self.E[:]/T))
    
    
    def Init(self):
        """
        Precalculate GG, B, BB
        """
        # GG
        self.GG = np.zeros(self.TRANSITIONS, np.float32)
        for tr in range(self.TRANSITIONS):
            u, l        = self.T2L(tr)
            self.GG[tr] = self.G[u]/self.G[l]
        # B
        self.B  =   asarray(  self.A*C_LIGHT*C_LIGHT/(2.0*PLANCK*double(self.F)**3.0) , float32 )
        # BB
        self.BB = self.GG * self.A * (C_LIGHT/self.F)**2.0 / (8.0*pi)


        
    def Transitions(self, levels):
        """
        Update self.TRANSITION array so that it contains only transitions between levels < levels.
        """
        print("Transitions(%d) from TRANSITIONS=%d" % (levels, self.TRANSITIONS))
        print(self.TRANSITION)
        ok  =  np.zeros(self.TRANSITIONS, np.int32)
        for i in range(self.TRANSITIONS):  # loop over original array
            a, b = self.TRANSITION[i,:]
            if ((a<levels)&(b<levels)):
                ok[i] = 1
        m = nonzero(ok>0)                  # only these transitions remain
        # TRANSITION[], A[], and F[] may be truncated = higher levels dropped
        self.TRANSITION   =  asarray(self.TRANSITION[m[0],:], np.int32)
        self.A            =  asarray(self.A[m[0]],  np.float32)
        self.B            =  asarray(self.B[m[0]],  np.float32)
        self.BB           =  asarray(self.BB[m[0]], np.float32)
        self.F            =  asarray(self.F[m[0]],  np.float32)
        self.TRANSITIONS  =  len(m[0])                               # number of radiative transitions
        self.GG           =  np.zeros(self.TRANSITIONS, np.float32)  # G indexed with level, GG with transition!
        for t in range(self.TRANSITIONS):
            u, l          =  self.T2L(t)
            self.GG[t]    =  self.G[u]/self.G[l]
            # print("    %2d -> %2d   F= %12.4e   A= %12.4e" % (u, l, self.F[t], self.A[t]))
        # Drop also higher levels... that would otherwise be included in partition function
        if (levels<len(self.G)):
            self.G        =  asarray(self.G[0:levels], np.float32)*1.0
            self.E        =  asarray(self.E[0:levels], np.float32)*1.0
        self.LEVELS = levels        
        if (1): # dump the remaining data
            fp = open('%s.dump' % self.NAME, 'w')
            fp.write('%s\n' % self.NAME)
            fp.write('%.3f\n' % self.WEIGHT)
            fp.write('%d\n'   % self.LEVELS)
            fp.write("Energies [Hz] --  g\n")
            for i in range(self.LEVELS):
                fp.write(' %3d    %12.5e    %.2f \n' % (i, self.E[i], self.G[i]))
            fp.write('%d\n' % self.TRANSITIONS)
            fp.write("Aul [Hz]\n")
            for i in range(self.TRANSITIONS):
                u, l = self.T2L(i)
                fp.write("  %3d   %3d %3d   %12.4e   %12.4e\n" % (i, u, l, self.A[i], self.F[i]))
            fp.write("Cul\n")
            for i in range(self.LEVELS):
                for j in range(self.LEVELS):
                    fp.write("  %3d %3d  %12.4e\n" % (i, j, self.C(i,j,25.0,0)))
            fp.close()
        print("Transitions -> %d" % self.TRANSITIONS)
        return self.TRANSITIONS
    
        
    
##########################################################################################
##########################################################################################
        

def Planck(F, T):
    return 2.0*PLANCK*((F/C_LIGHT)**2.0)*F / (np.exp((H_K*F/T))-1.0)



def ReadIni(filename):
    global PLANCK
    INI = {
    'nside'           : 1,                   #  Healpix NSIDE parameter, to generate ray directions
    'Tex'             : [],                  #  save excitation temperatures for listed transitions
    'spectra'         : [],                  #  save spectra for listed transitions
    'direction'       : [0.0, 0.0],          #  (theta, phi), the direction towards the observer
    'points'          : [10,10],             #  number pixels in the output maps
    'cooling'         : 0,                   #  save cooling rates 
    'coolfile'        : 'brute.cool',
    'mapview'         : [],                  #  theta, phi, nx, ny,  (x,y,z) map centre
    'GPU'             : 0 ,                  #  use GPU instead of CPU
    'platforms'       : [0,1,2,3,4],         #  OpenCL platforms to try
    'idevice'         : 0,                   #  selected device within the platform (for given device type)
    'sdevice'         : '',                  #  string used to select the OpenCL device
    'load'            : '',                  #  file to load saved level populations
    'save'            : '' ,                 #  file to save calculated level populations
    'iterations'      : 1,                   #  number of iterations (field simulation + level population updates)
    'stop'            : 1.0e-5,              #  stopping condition, based on relative change in level populations
    'uppermost'       : 999,                 #  uppermost level to check in connection with 'stop'
    'cabfile'         : '',                  #  abundance file for collisional partners
    'constant_tkin'   : 0,                   #  assume constant Tkin for the whole model
    'nray'            : 64,                  #  number of rays (1D models)
    'alpha'           : 1.0,                 #  parameter to adjust placement of rays (1D models)
    'nray_spe'        : -1 ,                 #  number of rays to calculate spectra for (1D models)
    'Tbg'             : 2.72548,             #  background sky temperature
    'hfsfile'         : '',                  #  file describing HFS line structure (HF components in LTE)
    'with_crt'        : 0 ,                  #  include CRT files (dust continuum absorption and emission)
    'overlap'         : '' ,                 #  include spectral overlap between lines
    'lowmem'          : 1,                   #  choose some memory saving options for the kernels
    'min_sigma'       : 1e30,                #  minimum turbulent linewidth in the model
    'max_sigma'       : 0.0,                 #  maximum turbulent linewidth in the model
    'method_x'        : 0,                   #  not used
    'LOCAL'           : -1,                  #  local work group size (overrides the program defaults)
    'crttau'          : 'crt.opacity',       #  name of the file for dust opacity
    'crtemit'         : 'crt.emission',      #  name of the file for dust emission
    'cloud'           : '',                  #  name of the 'cloud', the file containing densities etc.
    'octree'          : 0,                   #  the octree ray-tracing method chosen
    'clsolve'         : 1,                   #  solve equilibrium equations on device instead on host
    'offsets'         : 1,                   #  number of spatial ray offsets per surface element (default=1)
    'pickle'          : 1,                   #  save this INI structure to all output files (1D models)
    'WITH_ALI'        : 1,                   #  whether to use ALI (can be 0 for octree>=2)
    'savetau'         : 0,                   #  save optical depths (currently 1D models only)
    'plweight'        : 1,                   #  include path-length weighting for octree4 (affects OT4 only)
    'clip'            : 0.0,                 #  skip calculations when density below this threshold
    'damping'         : -1.0,                #  with ALI, dampen iterations is damping*old+(1-damping)*new
    'dnlimit'         : -1.0,                #  ni relative change > dnlimit => new = 0.5*old+0.5*new
    'oneshot'         :  0,                  #  OCTREE=40, single kernel call to cover all rays (given side)
    'thermaldv'       :  1,                  #  if >0, add thermal broadening, otherwise use file values as such
    'kdensity'        :  1.0,                #  scale volume densities
    'ktemperature'    :  1.0,                #  scale Tkin
    'kabundance'      :  1.0,                #  scale fractional abundance
    'kvelocity'       :  1.0,                #  scale macroscopic velocity
    'ksigma'          :  1.0,                #  scale microturbulence
    'maxbuf'          :  40,                 #  maximum allocation of rays per root-grid ray
    'WITH_HALF'       :  0,                  #  whether CLOUD is stored in half precision (vx, vy, vz, sigma)
    'KILL_EMISSION'   :  999999,             #  write spectra ignoring emission from cells >= KILL_EMISSION, 1D models only!!
    'minmaplevel'     : -1,                  #  only hierarky levels level>minmaplevel used in map calculation
    'MAP_INTERPOLATION': -1,                 #  spatial interpolation in map making
    'FITS'            :  0,                  #  if >0, save spectra and tau as FITS images
    'verbose'         :  1
    }
    lines = open(filename, 'r').readlines()
    for line in lines:        
        s = line.split()
        if (len(s)<1): continue
        if ((line[0:1]=='#')|(s[0]=='#')): continue

        if ((s[0].find('mapview')>=0)&(len(s)>=5)): # at least  theta, phi, nx, ny  -- optionally (xc, yc, zc)
            print("mapview, len(s)=%d" % (len(s)), s)
            tmp =  [ float(s[1])*pi/180.0, float(s[2])*pi/180.0,  int(s[3]), int(s[4]) ]  #   theta, phi, NX, NY map parameters
            try:
                mc = [ float(s[5]), float(s[6]), float(s[7]) ]          #   map centre (xc, yc, zc)
            except:
                mc = [ NaN, NaN, NaN ]  # these will be replaced by the default, the cloud centre
                pass
            #                      theta   phi       NX      NY        xc     yc     zc    
            INI['mapview'].append([tmp[0], tmp[1],   tmp[2], tmp[3],   mc[0], mc[1], mc[2]])
            
        if (len(s)>2): # two float arguments
            try:
                a, b = float(s[1]), float(s[2])
                if (s[0].find('points')>=0):   INI.update({'points':      [int(a), int(b)]})
                if (s[0].find('directi')>=0):  INI.update({'direction':   [a*pi/180.0, b*pi/180.0]})
            except:
                pass        
        if (len(s)>1): # keywords with one argument
            # spectra and transitions have several int arguments
            if ((s[0].find('spectra')>=0)|(s[0].lower().find('tex')==0)|(s[0].find('transition')>=0)):
                x = []
                for i in range(1, len(s)):
                    try:
                        a = int(s[i])
                        x.append(a)
                    except:
                        break
                if (len(x)>0):
                    x = asarray(x, np.int32)
                    if (s[0].find('spectra')>=0):      INI.update({'spectra':     x})
                    if (s[0].lower().find('tex')==0):  INI.update({'Tex':         x})            
                    if (s[0].find('transition')>=0):   INI.update({'Tex':         x})
            if (s[0].find('octree')>=0):  
                INI.update({'cloud':    s[1]})
                if   (s[0].find('40')>0): INI['octree'] = 40
                elif (s[0].find('5')>0):  INI['octree'] = 5
                elif (s[0].find('4')>0):  INI['octree'] = 4
                elif (s[0].find('3')>0):  INI['octree'] = 3
                elif (s[0].find('2')>0):  INI['octree'] = 2
                elif (s[0].find('1')>0):  INI['octree'] = 1
                else:                     INI['octree'] = 4 #  0 -> 4, default changed 2021-03-14
                print("*** OCTREE %d ***" % INI['octree'])
            if (s[0].find('cloud')>=0):   INI.update({'cloud':    s[1]})
            if (s[0].find('molec')>=0):   INI.update({'molecule': s[1]})
            if (s[0].find('load')==0):    INI.update({'load':     s[1]})
            if (s[0].find('save')==0):    INI.update({'save':     s[1]})  # make sure tausave and save and not confused
            if (s[0].find('prefix')>=0):  INI.update({'prefix':   s[1]})
            if (s[0].find('cabfile')>=0): INI.update({'cabfile':  s[1]})
            if (s[0].find('hfsfile')>=0): INI.update({'hfsfile':  s[1]})
            if (s[0].find('overlap')>=0): INI.update({'overlap':  s[1]})            
            if (s[0].find('crttau')>=0):  INI.update({'crttau':   s[1]})
            if (s[0].find('crtemit')>=0): INI.update({'crtemit':  s[1]})
            if (s[0].find('device')>=0):  INI.update({'sdevice':  s[1]})
            ### 18-08-2021: now specifies a file name if the "cooling" keyword exists
            if (s[0].find('cooling')>=0):
                INI.update({'cooling':  1})
                INI.update({'coolfile':  s[1]})
            # float argument
            try:
                x = float(s[1])
                if (s[0].find("isotropic")>=0):    INI.update({'Tbg':         x})
                if (s[0].find("bandwidth")>=0):    INI.update({'bandwidth':   x})
                if (s[0].find("temperature")>=0):  INI.update({'ktemperature':x})
                if (s[0].find("density")>=0):      INI.update({'kdensity':    x})
                if (s[0].find("fraction")>=0):     INI.update({'kabundance':  x})
                if (s[0].find("abundance")>=0):    INI.update({'kabundance':  x})
                if (s[0].find("velocity")>=0):     INI.update({'kvelocity':   x})
                if (s[0].find("sigma")>=0):        INI.update({'ksigma':      x})
                if (s[0].find("distance")>=0):     INI.update({'distance':    x})
                if (s[0].find("angle")>=0):        INI.update({'angle':       x})
                if (s[0].find("grid")>=0):         INI.update({'grid':        x})
                if (s[0].find("stop")>=0):         INI.update({'stop':        x})                
                if (s[0].find("alpha")>=0):        INI.update({'alpha':       x})
                if (s[0].find("clip")>=0):         INI.update({'clip':        x})
                if (s[0].find("damp")>=0):         INI.update({'damping':     x})
                if (s[0].find("dnlim")>=0):        INI.update({'dnlimit':     x})
            except:
                pass
            # int argument
            try:
                x = int(s[1])
                if (s[0].find("channels")>=0):     INI.update({'channels':    x})
                if (s[0].find("iterations")>=0):   INI.update({'iterations':  x})
                if (s[0].find("uppermost")>=0):    INI.update({'uppermost':   x})
                if (s[0].find("nside")>=0):        INI.update({'nside':       x})
                if (s[0].find("gpu")>=0):          INI.update({'GPU':         x})
                if (s[0].find("GPU")>=0):          INI.update({'GPU':         x})
                if (s[0].find("levels")>=0):       INI.update({'levels':      x})
                if (s[0].find("speray")>=0):       INI.update({'nray_spe':    x})
                if (s[0].find("nray")>=0):         INI.update({'nray':        x})                
                if (s[0].find('offsets')>=0):      INI.update({'offsets':     x})
                if (s[0].find('lowmem')>=0):       INI.update({'lowmem':      x})
                if (s[0].find('clsolve')>=0):      INI.update({'clsolve':     x})                        
                if (s[0].find('LOCAL')>=0):        INI.update({'LOCAL':       x})
                if (s[0].find('local')>=0):        INI.update({'LOCAL':       x})
                if (s[0].find("ALI")==0):          INI.update({'WITH_ALI':    x})
                if (s[0].find("ali")==0):          INI.update({'WITH_ALI':    x})
                if (s[0].find("FITS")==0):         INI.update({'FITS':        x})
                if (s[0].find("verbose")==0):      INI.update({'verbose':     x})
                if (s[0].find("tausave")>=0):      INI.update({'savetau':     x})
                if (s[0].find("plweight")>=0):     INI.update({'plweight':    x})
                if (s[0].find("oneshot")>=0):      INI.update({'oneshot':     x})
                if (s[0].find("thermaldv")>=0):    INI.update({'thermaldv':   x})
                if (s[0].find("maxbuf")>=0):       INI.update({'maxbuf':      x})
                if (s[0].find("half")>=0):         INI.update({'WITH_HALF':   x})
                if (s[0].find("killemi")>=0):      INI.update({'KILL_EMISSION': x})  # kill all emission from cells>x, 1D models only!!
                if (s[0].find('minmaplevel')>=0):  INI.update({'minmaplevel'  : x})
                if (s[0].find("mapint")>=0):       INI.update({'MAP_INTERPOLATION':   x})
                if (s[0].find("platform")>=0):  
                    INI.update({'platforms':   [x,]})
                    if (len(s)>2): # user also specifies the device within the platform
                        try:
                            INI.update({'idevice': int(s[2])})
                        except:
                            idevice = 0
            except:
                pass                                    
        # keywords without arguments
        if (s[0].find('cool')>=0):             INI.update({'cooling':       1})
        if (s[0].find('constant_tkin')>=0):    INI.update({'constant_tkin': 1})
        if (s[0].find('crtdust')>=0):          INI.update({'with_crt':      1})
        if (s[0].find('pickle')>=0):           INI.update({'pickle':        1})
        if (s[0].find('methodx')>=0):          INI.update({'method_x':      1})
        
    # one can use "direction" and "points", map centred on the cloud centre
    # if mapview is given, direction and points will be ignored
    if (len(INI['mapview'])<1):
        #                  theta               phi                 nx               ny                xc   yc   zc 
        INI['mapview'].append(
        [INI['direction'][0],INI['direction'][1],INI['points'][0],INI['points'][1], NaN, NaN, NaN])
    # some allocations depend on map size => update INI['points'] with the maximum values
    max_nra, max_nde = 0, 0
    for i in range(len(INI['mapview'])):
        max_nra = max(max_nra, INI['mapview'][i][2])
        max_nra = max(max_nra, INI['mapview'][i][2])
    INI['points'] = [ max_nra, max_nde]
    # INI['direction'] is no longer needed
    INI['direction'] = []
    return INI



def ReadMolecule(molname):
    """
    Read molecule
    """
    MOL = MoleculeO()
    if (os.path.exists(molname)):
        MOL.Read(molname)
    else:
        MOL.Read('%s/%s' % ('/home/mika/tt/MOL/', molname))
    return MOL



def ReadCloudOT(INI, MOL):
    """
    Read and rescale an octree cloud.
    Usage:
        RHO, TKIN, CLOUD, ABU, CELLS, OTL, LCELLS, OFF, NX, NY, NZ = ReadCloudOT(INI, MOL)
    Return:
        RHO[cells],  vector of volume densities
        TKIN[cells], vector of kinetic temperatures
        CLOUD[cells], float4 ~ [vx, vy, vz, sigma], sigma = line width
        ABU[cells], vector of abundances
        CELLS, total number of cells
        OTL,  number of levels in the octree hierarchy
        OFF,  offsets within a parameter vector to the first entry on a level of hierarchy
        NX, NY, NZ, dimensions of the root grid
    File format:
        NX, NY, NZ, OTL, CELLS = dimensions (x, y, z), number of levels (OTL), number of cells
        this is followed by data for rho[], T[], sigma[], vx[], vy[], vz[], ABU[] ;
        each of these vectors consist of:
            lcells0, vector0, lcells1, vector1, ...
        i.e. number of cells on the hierarchy level and the values for the cells on that
        hierarchy level.
        The cloud hierarchy is defined by rho[], values <=0.0 are links to child cells.
        For sigma, use convention of having values <-1e10 for cells other than leaf nodes
        Could use the same for (vx, vy, vz).
        In addition tho WHO, we could have actual links as part of the parameters vectors only 
        for TKIN and sigma...
    """
    fp         =  open(INI['cloud'], 'rb')
    NX, NY, NZ, OTL, CELLS = fromfile(fp, int32, 5) # OTL = octree levels
    print("ReadCloudOT(%s): " % INI['cloud'], NX, NY, NZ, OTL, CELLS)
    LCELLS     =  zeros(OTL, int32)
    OFF        =  zeros(OTL, int32)
    #  rho, tkin, sigma, vx, vy, vz, abu = 6*4 bytes per cell
    #  100 million cells = 2.2 GB
    RHO        =  zeros(CELLS,  float32)
    TKIN       =  zeros(CELLS,  float32)
    WITH_HALF  =  [0, 1][INI['WITH_HALF']>0]
    if (WITH_HALF==0):
        CLOUD      =  zeros(CELLS,  cl.cltypes.float4)         # vx, vy, vz, sigma
    else:
        CLOUD      =  np.empty((CELLS,4),  cl.cltypes.half)    # vx, vy, vz, sigma
    ABU        =  zeros(CELLS,  float32)
    """
    For density, values <= 0 correspond to links to child cells.
    For the other quantities, use the convention val<-1e10 to indicate that the cell is not a leaf node.
    """
    # density
    cells      = 0
    for level in range(OTL):
        if (level>0): OFF[level] = OFF[level-1] + cells  # cells = cells on the previous level
        cells = fromfile(fp, int32, 1)[0]
        print(" level %2d  cells %6d" % (level, cells))
        if (cells<1): break
        LCELLS[level] = cells
        tmp           = fromfile(fp, float32, cells)
        if (INI['kdensity']!=1.0):     tmp[nonzero(tmp>0.0)] *= INI['kdensity']   # no scaling if cell contains a link
        RHO[(OFF[level]):(OFF[level]+cells)] = tmp
    # temperature
    for level in range(OTL):
        cells   = fromfile(fp, int32, 1)[0]
        if (cells!=LCELLS[level]):       print("Error in the hierarchy file !"),  sys.exit()
        tmp     = fromfile(fp, float32, LCELLS[level])
        if (INI['ktemperature']!=1.0): tmp[nonzero(tmp>0.0)] *= INI['ktemperature']                        
        TKIN[(OFF[level]):(OFF[level]+cells)] = tmp
    # sigma = turbulent linewidth
    print("================================================================================")
    for level in range(OTL):
        # Note: any NaN values make np.min() np.max() equal to NaN !
        cells   = fromfile(fp, int32, 1)[0]
        if (cells!=LCELLS[level]):    print("Error in the hierarchy file, in sigma !"),  sys.exit()
        tmp  = fromfile(fp, float32, cells)
        m    = nonzero(RHO[OFF[level]:(OFF[level]+cells)]>0.0) # must ignore sigma for links
        if (len(m[0])>0):
            #print("tmp-1 %12.4e %12.4e,   ksigma %10.3e" % (np.min(tmp[m]), np.max(tmp[m]), INI['ksigma']))
            if (INI['ksigma']!=1.0):   tmp[m] *= INI['ksigma']
            #print("tmp-2 %12.4e %12.4e" % (np.min(tmp[m]), np.max(tmp[m])))
            #print("Tkin %12.4e %12.4e" % (np.min(TKIN[(OFF[level]):(OFF[level]+cells)][m]), np.max(TKIN[(OFF[level]):(OFF[level]+cells)][m])))
            if (INI['thermaldv']>0):                
                tmp[m]  = (np.sqrt(tmp**2.0+2.0e-10*BOLTZMANN*TKIN[(OFF[level]):(OFF[level]+cells)]/(AMU*MOL.WEIGHT)))[m]
                # print("tmp-3 %12.4e %12.4e" % (np.min(tmp[m]), np.max(tmp[m])))
            INI['min_sigma'] = min([ INI['min_sigma'],  np.min(tmp[m]) ])
            INI['max_sigma'] = max([ INI['max_sigma'],  np.max(tmp[m]) ])
        print("min_sigma %12.4e, max_sigma %12.4e" % (INI['min_sigma'], INI['max_sigma']))
        if (WITH_HALF==0):
            CLOUD[(OFF[level]):(OFF[level]+cells)]['w'] = tmp
        else:
            CLOUD[(OFF[level]):(OFF[level]+cells),3] = tmp
    print("================================================================================")
    # vx, vy, vz
    for level in range(OTL):
        cells   = fromfile(fp, int32, 1)[0]
        if (cells!=LCELLS[level]):      print("Error in the hierarchy file, in vx !"),  sys.exit()
        tmp     = fromfile(fp, float32, LCELLS[level])
        if (INI['kvelocity']!=1.0):     tmp  *= INI['kvelocity']
        if (WITH_HALF==0):
            CLOUD[(OFF[level]):(OFF[level]+cells)]['x'] = tmp
        else:
            CLOUD[(OFF[level]):(OFF[level]+cells), 0] = tmp
    for level in range(OTL):
        cells   = fromfile(fp, int32, 1)[0]
        if (cells!=LCELLS[level]):      print("Error in the hierarchy file, in vy !"),  sys.exit()
        tmp     = fromfile(fp, float32, LCELLS[level])
        if (INI['kvelocity']!=1.0):     tmp  *= INI['kvelocity']
        if (WITH_HALF==0):
            CLOUD[(OFF[level]):(OFF[level]+cells)]['y'] = tmp
        else:
            CLOUD[(OFF[level]):(OFF[level]+cells), 1] = tmp
    for level in range(OTL):
        cells   = fromfile(fp, int32, 1)[0]
        if (cells!=LCELLS[level]):      print("Error in the hierarchy file, in vz !"),  sys.exit()
        tmp     = fromfile(fp, float32, LCELLS[level])
        if (INI['kvelocity']!=1.0):     tmp *= INI['kvelocity']                        
        if (WITH_HALF==0):
            CLOUD[(OFF[level]):(OFF[level]+cells)]['z'] = tmp
        else:
            CLOUD[(OFF[level]):(OFF[level]+cells), 2] = tmp
    # abundance
    for level in range(OTL):
        cells   = fromfile(fp, int32, 1)[0]
        if (cells!=LCELLS[level]):      print("Error in the hierarchy file, in abundance !"),  sys.exit()
        tmp     = fromfile(fp, float32, LCELLS[level])
        if (INI['kabundance']!=1.0):    tmp *= INI['kabundance']
        if (0):
            tmp[:] = 1.0e-4    # FORCE CONSTANT ABUNDANCE EVERYWHERE
        ABU[(OFF[level]):(OFF[level]+cells)] = tmp
    fp.close()
    #
    return RHO, TKIN, CLOUD, ABU, CELLS, OTL, LCELLS, OFF, NX, NY, NZ




def ReadCloud3D(INI, MOL):
    """
    Read and rescale the cloud.
    """
    fp         = open(INI['cloud'], 'rb')
    nx, ny, nz = fromfile(fp, np.int32, 3)
    cells      = nx*ny*nz
    #    0  1  2      3   4   5   6
    #    n, T, sigma, vx, vy, vz, x
    try:
        C       =  fromfile(fp, np.float32).reshape(nz*ny*nx,7)
    except:
        # perhaps cloud is in octree format but just with one hierarchy level ...
        fp.close()
        print("Trying to read plain cartesian cloud from octree file....")
        fp         = open(INI['cloud'], 'rb')
        nx, ny, nz, otl, cells = fromfile(fp, np.int32, 5)
        if (otl!=1):
            print("Trying to read cartesian grid cloud but file has %d levels of hierarchy!" % otl)
            sys.exit(0)
        C = transpose(fromfile(fp, np.float32).reshape(7, 1+cells)[:, 1:].reshape(7, cells))        
    ###
    C[:,0]  =  clip(C[:,0]*INI['kdensity'],     1.0e-4, 1e15)    # density
    C[:,1]  =  clip(C[:,1]*INI['ktemperature'], 2.0,    2900.0)  # Tkin
    C[:,2]  =  clip(C[:,2]*INI['ksigma'],       1e-10,  1e3)     # sigma, nonthermal
    if (INI['thermaldv']>0):
        C[:,2]  =  np.sqrt(C[:,2]**2.0 + 2.0e-10*BOLTZMANN*C[:,1]/(AMU*MOL.WEIGHT)) # add thermal broadening
    C[:,3] *=  INI['kvelocity']
    C[:,4] *=  INI['kvelocity']
    C[:,5] *=  INI['kvelocity']
    C[:,6] *=  INI['kabundance']
    #
    INI['min_sigma'] = np.min(C[:,2])
    INI['max_sigma'] = np.max(C[:,2])
    #
    RHO           =  C[:,0]   # density
    TKIN          =  C[:,1]   # Tkin
    CLOUD         =  np.zeros(cells, cl.cltypes.float4)
    CLOUD[:]['x'] =  C[:,3]   #  vx
    CLOUD[:]['y'] =  C[:,4]   #  vy
    CLOUD[:]['z'] =  C[:,5]   #  vz
    CLOUD[:]['w'] =  C[:,2]   # sigma
    ABU           =  C[:,6]   # left as 1d vecto
    return RHO, TKIN, CLOUD, ABU, nx, ny, nz
    


def ReadCloud1D(INI, MOL):
    """
    Read and rescale 1D cloud
    Input:
        INI  =  initialisation parameter dictionary
        MOL  =  Molecule
    Return:
        RADIUS, VOLUME, RHO, TKIN, CLOUD, ABU  =  cloud data, where
        CLOUD is  [ Vrad, Rc, dummy, sigma ]
    Note:
        File has data in the order of:   rho, Tkin, sigma, abu, vrad
    """
    fp      =  open(INI['cloud'], 'rb')
    CELLS   =  fromfile(fp, np.int32, 1)[0]
    VOLUME  =  np.zeros(CELLS, np.float32)
    RADIUS  =  fromfile(fp, np.float32, CELLS)
    # cloud file may be normalised or it may be absolute values [cm]
    GL_IN_CLOUD_FILE = RADIUS[CELLS-1] ;
    for i in range(CELLS): RADIUS[i] /= GL_IN_CLOUD_FILE     # RADIUS is normalised !!
    VOLUME[0]  =  RADIUS[0]**3.0
    for i in range(1, CELLS): VOLUME[i] = RADIUS[i]**3.0 - RADIUS[i-1]**3.0
    RHO   = np.zeros(CELLS, np.float32)
    TKIN  = np.zeros(CELLS, np.float32)
    ABU   = np.zeros(CELLS, np.float32)
    CLOUD             =  np.zeros(CELLS, cl.cltypes.float4)
    # CLOUD.x/y/z/w = Vrad, Rc, dummy, sigma
    molwei            =  MOL.WEIGHT
    for i in range(CELLS):
        buf           =  fromfile(fp, np.float32, 5)   # rho, Tkin, sigma, abundance, Vrad
        RHO[i]        =  max(1.0e-5,  buf[0] * INI['kdensity'])
        TKIN[i]       =  min(2900.0,  buf[1] * INI['ktemperature'])
        ABU[i]        =  max(1.0e-20, buf[3] * INI['kabundance'])
        CLOUD[i]['z'] =  ABU[i]                     # z=abundance
        CLOUD[i]['x'] =  buf[4] * INI['kvelocity']  # x=Vrad
        sigma         =  buf[2] * INI['ksigma']     # multiplication applied to nonthermal component only
        if (INI['thermaldv']>0):
            sigma     =  np.sqrt(sigma*sigma + 2.0e-10*BOLTZMANN*TKIN[i] / (AMU*MOL.WEIGHT))
        CLOUD[i]['w'] =  sigma                      #  w = sigma
        if (not(isfinite(sigma))):
            print(" ??? TKIN %10.3e, buf[2] %.3e, ksigma %.3e\n" % (TKIN[i], buf[2], INI['ksigma']))
    fp.close()
    RHO         =  clip(RHO,  1.0e-5,    1e20)
    TKIN        =  clip(TKIN, 2.0,     2900.0)
    ABU         =  clip(ABU,  1.0e-20,    2.0)
    #
    if (0):
        CLOUD[0]['y'] =  0.5*RADIUS[0]    # y=Rc
        for icell in range(1, CELLS):     # radius weighted by volume
            CLOUD[icell]['y'] = 0.5*(RADIUS[icell-1]+RADIUS[icell])
    if (1):
        # CLOUD[].y = effective shell radius
        CLOUD[0]['y']  = 0.5*RADIUS[0]    # y=Rc
        for icell in range(1,CELLS):   # radius weighted by volume
            CLOUD[icell]['y'] = np.sqrt( 0.5*RADIUS[icell-1]**2.0 + 0.5*RADIUS[icell]**2.0 )
    if (0):
        # check the effect of replacing Vrad with the average over inner and outer boundary values
        # ... effect minimal (in test case)
        for icell in range(CELLS-1, -1, -1):
            CLOUD[icell]['x'] = 0.5*(CLOUD[icell-1]['x']+CLOUD[icell]['x'])
    print("================================================================")
    print("CELLS           %d" % CELLS)
    print("DENSITY         %10.3e to %10.3e, average_vol %10.3e" % (np.min(RHO),  np.max(RHO),  sum(VOLUME*RHO)/sum(VOLUME)))
    print("TKIN            %10.3e to %10.3e, average_vol %10.3e" % (np.min(TKIN), np.max(TKIN), sum(VOLUME*TKIN)/sum(VOLUME)))
    print("THERMAL SIGMA   %10.3e to %10.3e" % (sqrt(2.0e-10*BOLTZMANN*np.min(TKIN) / (AMU*molwei)),
    sqrt(2.0e-10*BOLTZMANN*max(TKIN) / (AMU*molwei))))
    sigma = CLOUD[:]['w']
    print("TOTAL SIGMA     %10.3e to %10.3e, average_vol %10.3e" % (np.min(sigma), np.max(sigma), sum(VOLUME*sigma)/sum(VOLUME)))
    print("ABUNDANCE       %10.3e to %10.3e" % (np.min(ABU), np.max(ABU)))
    print("VRAD            %10.3e to %10.3e" % (np.min(CLOUD[:]['x']), np.max(CLOUD[:]['x'])))
    print("================================================================")
    
    if (INI['angle']>0.0):                # cloud size defines by ini file
        INI['GL']  =  INI['angle'] * ARCSEC_TO_RADIAN * INI['distance'] * PARSEC   # [cm] = 1D cloud radius
        if (GL_IN_CLOUD_FILE>1.0001):     # but also cloud file had values in [cm]
            if (fabs(INI['GL']-GL_IN_CLOUD_FILE)>(0.01*INI['GL'])):
                print("**** WARNING: CLOUD FILE SPECIFIED RADIUS %.3e BUT INI FILE SCALED IT TO %.3e\n" % 
                (GL_IN_CLOUD_FILE, INI['GL']))
    else:
        INI['GL']     =  GL_IN_CLOUD_FILE
        INI['angle']  =  INI['GL'] / (ARCSEC_TO_RADIAN*INI['distance']*PARSEC)
    #
    if (0):
        print("________________________________________________________________________________")
        print('RADIUS')
        print(RADIUS)
        print('VOLUME')
        print(VOLUME) 
        print("TKIN")
        print(TKIN)
        print("SIGMA")
        print(CLOUD[:,]['w'])
        print("VRAD")
        print(CLOUD[:,]['x'])
        print("ABU")
        print(CLOUD[:,]['z'])
        print("________________________________________________________________________________")
        sys.exit()
    return RADIUS, VOLUME, RHO, TKIN, CLOUD, ABU



def GaussianProfiles(s0, s1, ng, nchn, dv):
    """
    Prepare array of n Gaussian profiles between sigma = s0 and s1 
    """
    # print("Gaussian profiles, nchn=%d" % nchn) 
    a = clip(s0, 0.05, 1.0)
    b = clip(s1, 0.05, 100.0)
    a = 0.999*a
    b = 1.001*b
    SIGMA0 =  a
    SIGMAX =  10.0**(  np.log10(b/a)/(ng-1.0)  )
    ## SIGMAX =  10.0**(np.log10(max([a, 1.01*b])/b) / (ng-1.0))
    if (SIGMAX>1.06): print("*** WARNING: SIGMAX = %.4f > 1.06 !!!" % SIGMAX)
    print("GAUSS(%.3e, %.3e) : [%.3e, %.3e], SIGMA0 %.3e, SIGMAX %.3e" % 
    (s0, s1, SIGMA0, SIGMA0*SIGMAX**(ng-1.0), SIGMA0, SIGMAX))
    GAU = np.zeros((ng, nchn), np.float32)
    v   = (-0.5*(nchn-1.0)+arange(nchn))*dv
    # integration limits, LIM = first and last nonzero channel
    LIM = np.zeros(ng, cl.cltypes.int2)
    for i in range(ng):
        s           =  SIGMA0 * SIGMAX**i
        GAU[i,:]    =  clip(np.exp(-v*v/(s*s)), 1.0e-30, 1.0)      # doppler width ??
        GAU[i,:]   /=  sum(GAU[i,:])
        m           =  nonzero(GAU[i,:]>2.0e-5)
        if (1):
            LIM[i]['x'] =  m[0][ 0]
            LIM[i]['y'] =  m[0][-1]
        else:
            LIM[i]['x'] = 0
            LIM[i]['y'] = nchn-1        
    asarray(GAU, np.float32).tofile('gauss_py.dat')
    return SIGMA0, SIGMAX, GAU, LIM
        
        
        
def InitCL(GPU=0, platforms=[], idevice=0, sub=0, verbose=True):
    """
    Usage:
        platform, device, context, queue, mf = InitCL(GPU=0, platforms=[], sub=0, idevice=0, verbose=True)
    Input:
        GPU       =  if >0, try to return a GPU device instead of CPU
        platforms =  optional array of possible platform numbers
        idevice   =  index of the device within the selected platform (default idevice=0)
        sub       =  optional number of threads for a subdevice (first returned)
        verbose   =  if True, print out the names of the platforms
    """
    platform, device, context, queue = None, None, None, None
    possible_platforms = range(6)
    if ((len(platforms)>0)&(platforms[0]>=0)):
        possible_platforms = platforms
    device = []
    for iplatform in possible_platforms:
        if (verbose): print("try platform %d, idevice=%d, request GPU=%d" % (iplatform, idevice, GPU))
        try:
            platform     = cl.get_platforms()[iplatform]
            if (GPU>0):
                device   = [ platform.get_devices(cl.device_type.GPU)[idevice] ]
            else:
                device   = [ platform.get_devices(cl.device_type.CPU)[idevice] ]
            if (sub>0):
                # try to make subdevices with sub threads, return the first one
                dpp       =  cl.device_partition_property
                device    =  [device[0].create_sub_devices( [dpp.EQUALLY, sub] )[0],]
            context   =  cl.Context(device)
            queue     =  cl.CommandQueue(context)
            break
        except:
            pass
    # print("***InitCL completed***")
    if (verbose):
        print("  Platform: ", platform)
        print("  Device:   ", device)
    return platform, device, context, queue,  cl.mem_flags
        


def InitCL_string(INI, verbose=True):
    """
    Usage:
        platform, device, context, queue, mf = InitCL(INI, verbose=True)
    Input:
        INI       =  structure built based on the initialisation file
                     we use INI['sdevice'] string to identify the requested device
                     and only set INI['GPU'] to indicate whether that was a CPU or a GPU
        verbose   =  if True, print out the names of the platforms
    """
    platforms    = cl.get_platforms()
    if (1): # print out platform.version, device.version for all devices
        print("================================================================================")
        for iplatform in range(len(platforms)):
            print('  Platform [%d]:   %s' % (iplatform, platforms[iplatform].name))
            devices     = platforms[iplatform].get_devices(cl.device_type.CPU)
            for idevice in range(len(devices)):
                print('       CPU [%d]:   %s' % (idevice, devices[idevice].name))
            devices     = platforms[iplatform].get_devices(cl.device_type.GPU)
            for idevice in range(len(devices)):
                print('       GPU [%d]:   %s' % (idevice, devices[idevice].name))
        print("================================================================================")
    ###
    platform, device, context, queue = None, None, None, None
    device = []
    for iplatform in range(len(platforms)):
        platform    = cl.get_platforms()[iplatform]
        devices     = platform.get_devices(cl.device_type.GPU)
        for idevice in range(len(devices)):
            if (INI['sdevice'] in devices[idevice].name):                
                device = [ devices[idevice] ]
                INI['GPU'] = 1
                break
        if (len(device)>0): break
        devices   = platform.get_devices(cl.device_type.CPU)
        for idevice in range(len(devices)):
            if (INI['sdevice'] in devices[idevice].name):
                device = [ devices[idevice] ]
                INI['GPU'] = 0
                break
        if (len(device)>0): break
    if (len(device)<1):
        print("InitCL_string: could not find any device matching string: %s" % INI['sdevice'])
        sys.exit()
    # try to make subdevices with sub threads, return the first one
    try:
        context   =  cl.Context(device)
        queue     =  cl.CommandQueue(context)
    except:
        print("Failed to create OpenCL context and quee for device: ", device[0])
        sys.exit()
    if (verbose):
        print("Selected:")
        print("   Platform: ", platform)
        print("   Device:   ", device)
        print("================================================================================")        
    return platform, device, context, queue,  cl.mem_flags
        


def IRound(a, b):
    if (a%b==0):
        return a
    return  (a//b+1)*b



def Pixel2AnglesRing(nside, ipix):
    # Convert Healpix pixel index to angles (phi, theta), theta=0.5*pi-lat, phi=lon
    # Uses formulas for maps in RING order.
    #  int    nl2, nl4, npix, ncap, iring, iphi, ip, ipix1 ;
    #  float  fact1, fact2, fodd, hip, fihip ;
    npix  = 12*nside*nside      # total number of points
    theta, phi = -999.0, -999.0
    # if ((ipix<0)|(ipix>=npix)): return -999.0, -999.0
    ipix1 = ipix + 1     #  in {1, npix}
    nl2   = 2*nside 
    nl4   = 4*nside 
    ncap  = 2*nside*(nside-1)   # points in each polar cap, =0 for nside =1
    fact1 = 1.5*nside 
    fact2 = 3.0*nside*nside  
    if (ipix1<=ncap):                               # North Polar cap
        hip   = ipix1/2.0
        fihip = int(hip)
        iring = int(sqrt(hip-sqrt(fihip))) + 1      # counted from North pole
        iphi  = ipix1 - 2*iring*(iring - 1) 
        theta = arccos(1.0-iring*iring / fact2)
        phi   = (iphi - 0.5) * pi/(2.0*iring)
    else:
        if (ipix1<=nl2*(5*nside+1)):                # Equatorial region ------
            ip    = ipix1 - ncap - 1
            iring = int(ip/nl4) + nside             # counted from North pole
            iphi  = (ip%nl4) + 1 
            fodd  = 0.5 * (1 + (iring+nside)%2)     #  1 if iring+nside is odd, 1/2 otherwise
            theta = arccos( (nl2 - iring) / fact1 )
            phi   = (iphi - fodd) * pi /(2.0*nside)
        else:                                      # South Polar cap
            ip    = npix - ipix1 + 1
            hip   = ip/2.0 
            fihip = int(hip) 
            iring = int(sqrt( hip - sqrt(fihip) )) + 1    # counted from South pole
            iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1))
            theta = arccos( -1.0 + iring*iring / fact2 ) 
            phi   = (iphi - 0.5) * pi/(2.0*iring) 
    return theta, phi





def GetHealpixDirection(nside, ioff, idir, X, Y, Z, offs=1,  DOUBLE_POS=False, theta0=-99.0, phi0=-99.0, with_healpy=False):
    """
    Return ray direction and position based on Healpix angular discretisation.
    Input:
        nside    =  resolution parameter of the Healpix map
        ioff     =  index for the offset of the initial position
        idir     =  index of the Healpix pixel (=direction)
        X, Y, Z  =  root dimensions of the model
        offs     =  number of positions per dimension per cell
                    Default is offs=1, one ray per surface element, ioff=0-3 cover all rays
                    For offs=2, 2x2 rays per cell, ioff=0-15 covers all the ray start positions over 2x2 cell.
                    Kernel steps the whole surface at steps of 2 cells, starting from the offsets provided here.
                    offs=1 => ioff=0:4,    offs=2 => ioff=0:16
    """
    if (DOUBLE_POS):
        POS   =  cl.cltypes.make_double3()
        ## DIR   =  cl.cltypes.make_double3()
    else:
        POS   =  cl.cltypes.make_float3()
    DIR   =  cl.cltypes.make_float3()
    
    
    if (nside==0):  # mostly for debugging -- only the six cardinal directions
        a     =  ioff //  4        #  offsets within a cell
        b     =  ioff %   4        #  offsetss between 2x2 cells
        d1    =  (0.01171875+(a//offs)) / offs   +  (b//2)
        d2    =  (0.91015625+(a% offs)) / offs   +  (b% 2)
        d1    =  (0.531+(a//offs)) / offs   +  (b//2)
        d2    =  (0.53+(a% offs)) / offs   +  (b% 2)
        b     =  1.0e-5
        #   WHY IS TEX DEPENDENT ON THIS ANGLE ????????????????????????????????
        #   b=0.3 gives expected Tex when collisions are weak... <PL> = 1.086  
        b     =  0.3     
        b     =  1.0e-4
        a     =  sqrt(1.0-b*b-b*b)
        if (idir in [0,1]):
            DIR['x'], DIR['y'], DIR['z']  =  a*[1.0,-1.0][idir % 2], b, b
            POS['x'], POS['y'], POS['z']  =  0.0, d1, d2
        elif (idir in [2,3]):
            DIR['y'], DIR['x'], DIR['z']  =  a*[1.0,-1.0][idir % 2], b, b
            POS['y'], POS['x'], POS['z']  =  0.0, d1, d2
        else:
            DIR['z'], DIR['x'], DIR['y']  =  a*[1.0,-1.0][idir % 2], b, b
            POS['z'], POS['x'], POS['y']  =  0.0, d1, d2
        return POS, DIR, idir
    
    if (with_healpy==False):
        theta, phi =  Pixel2AnglesRing(nside, idir)
    else:
        theta, phi =  healpy.pix2ang(nside, idir)  # theta from the pole
    if ((theta0>-2.0*pi)&(phi>-4.0*pi)):
        theta, phi = theta0, phi0
    else:
        phi   +=  0.137534213            # to avoid exact (+-0.666667,+-0.6666667, 0.3333333) directions
        # theta  =  0.00005+theta*0.99995
        pass

    DIR['x']   =  sin(theta)*cos(phi)
    DIR['y']   =  sin(theta)*sin(phi)
    DIR['z']   =  cos(theta)
    if (1):
        if (abs(DIR['x'])<1.0e-4): DIR['x'] = 1.0e-4 
        if (abs(DIR['y'])<1.0e-4): DIR['y'] = 1.0e-4 
        if (abs(DIR['z'])<1.0e-4): DIR['z'] = 1.0e-4 
        tmp        =  1.0/sqrt(DIR['x']*DIR['x']+DIR['y']*DIR['y']+DIR['z']*DIR['z']) 
        DIR['x']   =  tmp*DIR['x']
        DIR['y']   =  tmp*DIR['y']
        DIR['z']   =  tmp*DIR['z']
    # Which is the LEADING edge
    if (1):
        tmp        =  asarray([DIR['x'], DIR['y'], DIR['z']], float64)
        imax       =  argmax(abs(tmp))
        tmp[imax] *= 1.00001
        tmp       /=  sqrt(sum(tmp**2.0))
        DIR['x'], DIR['y'], DIR['z'] =  tmp
        LEADING    =  2*imax + [0,1][tmp[imax]<0.0]
    else:
        if (abs(DIR['x'])>abs(DIR['y'])):
            if (abs(DIR['x'])>abs(DIR['z'])):
                if (DIR['x']>0.0): LEADING = 0   # lower X
                else:              LEADING = 1   # upper X
            else:
                if (DIR['z']>0.0): LEADING = 4   # lower Z
                else:              LEADING = 5   # upper Z
        else:  #  Y or Z
            if (abs(DIR['y'])>abs(DIR['z'])):
                if (DIR['y']>0.0): LEADING = 2   # lower Y
                else:              LEADING = 3   # upper Y
            else:
                if (DIR['z']>0.0): LEADING = 4   # lower Z
                else:              LEADING = 5   # upper Z
    if (0): # one ray per cell, rays at steps of two cells, ioff=0-3
        d1    =  0.5 + int(ioff/2)
        d2    =  0.5 + int(ioff%2)
        # print(d1, d2)
    else:
        # index a for offs*offs position inside a cell, index b to cover 2x2 cells
        # (each call to simulation kernel will do rays at steps of two cells)
        a     =  ioff //  4        #  index for offsets inside a cell
        b     =  ioff %   4        #  index for offsets among 2x2 cells, ioff = [0, 4*offs*offs[
        #        .... inside cell ....       ... 2x2 cells ...
        # For some reason PL is sensitive to offsets ??
        # set offsets to 0.49 and 0.49 and PL has up to 10% variations
        # this worked for 25^3 test up to MAXL=4
        d1    =  (0.403714567+(a//offs)) / offs   +  (b//2)
        d2    =  (0.513975432+(a% offs)) / offs   +  (b% 2)
        d1    =  (0.01171875+(a//offs)) / offs   +  (b//2)
        d2    =  (0.91015625+(a% offs)) / offs   +  (b% 2)
                
    if (LEADING in[0,1]):     # lower or upper X-axis face
        POS['x'], POS['y'], POS['z']  =  0, d1, d2 
    elif (LEADING in[2,3]):  
        POS['y'], POS['x'], POS['z']  =  0, d1, d2
    else:  
        POS['z'], POS['y'], POS['x']  =  0, d1, d2
    ###
    return POS, DIR, LEADING
        
        
        
def GetSteps1D(CELLS, RADIUS, NRAY, IP, DIRWEI):
    """
    For strictly 1D models:
        Precalculate for each ray the distance it travels in each of the shells = STEP
        calculate the average length within each cell (for emission weighting)  = APL
        Input:
            CELLS    = number of cells
            RADIUS   = shell radiae [0,1]
            NRAY     = number of rays
            IP       = impact parameters of the rays [0,1]
            DIRWEI   = weight of each ray [NRAY]
                       called with DIRWEI=[] before writing of spectra, when STEP is updated
                       but DIRWEI and APL are not used
        Return:
            STEP     = length of each ray within each of the cells [NRAY, CELLS]
            APL      = total length of ray paths within each cell  [CELLS]
    """
    STEP =  np.zeros((NRAY, CELLS), np.float32)
    APL  =  np.zeros(CELLS, np.float32)
    # innermost cell
    for iray in range(NRAY):
        if (IP[iray]<RADIUS[0]):
            tmp            =  2.0*sqrt(RADIUS[0]**2.0-(IP[iray]**2.0))
            STEP[iray, 0]  =  tmp
            if (len(DIRWEI)>0):  APL[0]  +=  tmp*DIRWEI[iray]
        else:
            STEP[iray, 0]  =  -1.0
    # the rest of the cells:   IP<R0,  R0<IP<R1, IP>R1
    for icell in range(1, CELLS):
        for iray in range(NRAY):
            if (IP[iray]>RADIUS[icell]):         # shell is not hit at all
                STEP[iray, icell] = -1.0 
            else:
                if (IP[iray]>RADIUS[icell-1]):   #  almost tangential
                    tmp                =  2.0*sqrt((RADIUS[icell]**2.0)-(IP[iray]**2.0)) + 1.0e-6 
                    STEP[iray, icell]  =  tmp 
                    if (len(DIRWEI)>0):   APL[icell]  +=  tmp*DIRWEI[iray] 
                else:                        # two steps, incoming and outcoming ray are both:
                    tmp =  sqrt((RADIUS[icell  ]**2.0)-(IP[iray]**2.0)) -            \
                           sqrt((RADIUS[icell-1]**2.0)-(IP[iray]**2.0)) + 1.0e-6  
                    STEP[iray, icell]  =  tmp 
                    if (len(DIRWEI)>0):   APL[icell]  +=  2.0*tmp*DIRWEI[iray] 
    if (STEP[0, 0]<1e-10):
        print(" *** INNERMOST CELL IS NOT HIT BY ANY RAYS -> increase nray and/or adjust alpha !!!") 
        sys.exit(0)
    if (len(DIRWEI)<1): return STEP   # this one when called before writing the spectra
    return STEP, APL  # this one before simulations (DIRWEI used, APL updated)
        
    

class BandO:
    # Bands of LTE hfs
    N          =  0      # number of components
    WIDTH      =  0.0    # velocity channel
    BANDWIDTH  =  0.0    # bandwidth without distance between components
    VMIN       =  0.0
    VMAX       =  0.0
    VELOCITY   =  asarray([], float32)     # line offsets in km/s for each component
    WEIGHT     =  asarray([], float32)     # relative weights
    
    def Init(self, bandwidth, width):
        self.N          = 0   
        self.BANDWIDTH  = bandwidth  
        self.WIDTH      = width 
        self.VMIN       = 0.0  
        self.VMAX       = 0.0 

    def Add(self, velocity, weight): # add another line component to the band
        self.VELOCITY = concatenate((self.VELOCITY, [velocity,]))
        self.WEIGHT   = concatenate((self.WEIGHT,   [weight,]))
        self.N += 1
        self.VMAX  =  np.max(self.VELOCITY)
        self.VMIN  =  np.min(self.VELOCITY)
        
    def Channels(self):
        # return the number of channels needed for this band
        return (int)((self.BANDWIDTH+self.VMAX-self.VMIN)/self.WIDTH) ;


    
class OLBandO:
    BANDS  =    0   # number of frequency bands (with possibly multiple components)
    NCMP   =    []  # number of components in each band
    TRAN   =    []  # list of transitions in each band
    FMIN   =    []  # minimum transition frequency in the band
    FMAX   =    []  # maximum transition frequency in the band
    DV     =    0.0 # channel width

    def Init(self, dv):
        self.BANDS = 0
        self.DV    = dv
        
    def Bands(self):
        return self.BANDS
    
    def Components(self, iband):
        return self.NCMP[iband] 
    
    def AddBand(self, components):
        self.TRAN.append( zeros(components, int32) )
        self.NCMP.append( 0 )
        self.FMIN.append( 1.0e30 )
        self.FMAX.append( 0.0    )
        self.BANDS += 1
        
    def AddTransition(self, tran, freq):
        iband = self.BANDS-1
        self.TRAN[iband][self.NCMP[iband]] = tran
        self.NCMP[iband] += 1
        if (freq<self.FMIN[iband]):  self.FMIN[iband] = freq
        if (freq>self.FMAX[iband]):  self.FMAX[iband] = freq        
        
    def GetTransition(self, iband, icmp):
        return self.TRAN[iband][icmp]
    
    def Channels(self, iband):
        # Return NCHN = the number of extra channels needed
        return 0


    
def ReadHFS(INI, MOL):
    """
    Read decription of HS structure, for calculations assuming LTE between HFS components.
    Input:
        filename  = name of the HFS structure description
    Return:
        BAND      = structure containing the band information
        channels  = maximum number of channels needed for any of the transitions 
    """
    BAND   =  []    # BandO for each transition
    width  =  INI['bandwidth'] / INI['channels']   # this remains unchanged
    for tran in range(MOL.TRANSITIONS):
        BAND.append(BandO())
        BAND[tran].Init(INI['bandwidth'], width)
    # First set up transitions mentioned in the hfs file
    lines = open(INI['hfsfile']).readlines()
    iline = 0
    while (iline<len(lines)):
        s       = lines[iline].split()
        iline  += 1
        if (len(s)<3):  continue
        if (s[0][0:1]=='#'): continue
        # line contained [upper, lower, number_of_components]
        upper, lower, nc  =  int(s[0]), int(s[1]), int(s[2])
        tran              =  MOL.L2T(upper, lower)
        for i in range(nc):
            s       = lines[iline].split()
            iline  += 1
            BAND[tran].Add(float(s[0]), float(s[1])) # add velocity, weight
    # Initialise the rest of BAND and find out the maximum bandwidth needed
    MAXCHN  = 0
    MAXCMP  = 1
    for tran in range(MOL.TRANSITIONS):
        if (BAND[tran].N==0):           # we did not add this one from the hfs file...
            BAND[tran].Add(0.0, 1.0)    # ... has only the single component
        MAXCHN  = max([MAXCHN, BAND[tran].Channels()])
        upper, lower  =  MOL.T2L(tran)
        print(" Tran  %3d = %3d -> %3d = %d components" % (tran, upper, lower, BAND[tran].N))
        if (BAND[tran].N>1):
            for i in range(BAND[tran].N):
                print("   off %7.3f   weight %7.3f" % (BAND[tran].VELOCITY[i], BAND[tran].WEIGHT[i]))
        MAXCMP = max([MAXCMP, BAND[tran].N])
    bandwidth = MAXCHN*width     # BAND.WIDTH ==  width == INI['bandwidth'] / INI['channels']
    return BAND, MAXCHN, MAXCMP
    
    
    
def ReadDustTau(filename, gl, cells, transitions):
    """
    Read dust optical depths from file =>  tau[cells, transitions]
    File contains optical depths  [1/pc], values are here converted to [1/GL]
    Input:
        filename    = name of the optical depth file saved from CRT
        gl          = GL for the current model [cm]
        cells       = number of cells in the model 
        transitions = number of transitions in the calculation
    """
    fp   = open(filename, 'rb')
    a, b = fromfile(fp, int32, 2)
    if ((a!=cells)|(b!=transitions)):
        print("*** Error in ReadDustTau: (CELLS,TRANSITIONS)=(%d,%d), file has (%d,%d)" % (cells,transitions,a,b))
        sys.exit()
    tau  = fromfile(fp, float32).reshape(cells, transitions)
    fp.close()
    tau[:,:] *= gl/PARSEC    # optical depth per GL
    return asarray(tau, float32)



def ReadDustEmission(filename, cells, transitions, width, mol):
    """
    Read dust emission from file =>  emit[cells, transitions]
    File contains values =  photons / s / Hz / H2
    Returned values      =  photons / s / channel / H2
    Input:
        filename    =   name of the dust emission file written by CRT
        cells       =   number of cells in the model
        transitions =   number of transitions 
        width       =   channel width [km/s]
        mol         =   molecule object
    Return:
        dust emission [cells, transitions] in units  photons / s / channel / H2
    Note:
        the caller will do scaling with density so that the final array should be in 
        units   photons / s / channel / cm3
    """
    fp   = open(filename, 'rb')
    a, b = fromfile(fp, int32, 2)
    if ((a!=cells)|(b!=transitions)):
        print("*** Error in ReadDustEmission: (CELLS,TRANSITIONS)=(%d,%d), file has (%d,%d)" % (cells,transitions,a,b))
        sys.exit()
    emi  =   fromfile(fp, float32).reshape(cells, transitions)
    fp.close()
    for t in range(transitions):
        emi[:,t] *=  mol.F[t] * (1.0e5*width/C_LIGHT)  # converted to photons / s / channel / H2
    return asarray(emi, float32)
        


def ReadOverlap(filename, mol, width, transitions, channels):
    """
    Read file describing overlapping transitions:
        # components
        { upper lower}
    """
    lines  = open(filename, 'r').readlines()
    OLBAND = OLBandO()
    OLBAND.Init(width)
    iline  = 0
    while(iline<len(lines)):
        s      =  lines[iline].split()
        iline +=  1
        if (len(s)<1):  continue
        if (s[0]=='#'): continue
        components = int(s[0])
        OLBAND.AddBand(components)
        for i in range(components):
            s      =  lines[iline].split()
            iline +=  1
            u, l   =  int(s[0]), int(s[1])
            itran  =  mol.L2T(u, l)
            OLBAND.AddTransition(itran, mol.F[itran])
    #
    MAXCMP = 0
    OLTRAN  =  zeros(transitions, int32)
    OLOFF   =  zeros(transitions, float32)
    for iband in range(OLBAND.Bands()):
        nchn   = OLBAND.Channels(iband) + channels
        ncmp   = OLBAND.Components(iband)
        MAXCMP = max([MAXCMP, ncmp])
        print("iband %d/%d,  %d components, %d channels" % (iband, OLBAND.Bands(), ncmp, nchn))
        f0     = mol.F[OLBAND.GetTransition(iband, 0)]
        for icmp in range(ncmp):  #   ncmp <= transitions
            OLTRAN[icmp]  =  OLBAND.GetTransition(iband, icmp)
            freq          =  mol.F[OLBAND.GetTransition(iband, icmp)]
            off           =  0.5*(nchn-1.0)-0.5*(channels-1.0)-(freq-f0)*(C_LIGHT/f0)*1.0e-5/width
            print("  %.1f %.1f %.1f" % (0.5*(nchn-1.0), 0.5*(channels-1.0), (freq-f0)*(C_LIGHT/f0)*1.0e-5/width))
            print("   icmp %2d, transition %3d, offset %5.1f" % (icmp, OLBAND.GetTransition(iband, icmp), off))
    ####
    return OLBAND, OLTRAN, OLOFF, MAXCMP




def ConvolveSpectra1D(filename, fwhm_as, GPU=0, platforms=[0,1,2,3,4], angle_as=-1.0, samples=201):
    """
    Convolve spectra from LOC1D.py with Gaussian beam.
    Input:
        filename  =  original spectrum file written by LOC1D.py
        fwhm_as   =  FWHM value of the Gaussian beam [arcsec]
        GPU       =  if >0, use GPU instead of CPU
        platforms =  potential OpenCL platforms to use, default [0,1,2,3,4]
        angle_as  =  model cloud radius in arcsec; if the parameter is not specified
                     or the value is negative, try to read pickled INI information from 
                     the provided spectrum file and read the angle from there
        samples   =  optional, number of samples per one dimension (default 201)
    """
    fp           =  open(filename, 'rb')
    NSPE, NCHN   =  fromfile(fp, int32, 2)
    V0, DV       =  fromfile(fp, float32, 2)
    SPE          =  fromfile(fp, float32, NSPE*NCHN).reshape(NSPE, NCHN)
    INI          =  None
    if (angle_as<0.0): # angle not given as parameter, read pickled data from the spectrum file
        try:
            INI          =  pickle.load(fp)
            fp.close()
            fwhm     =  (fwhm_as/INI['angle']) * (NSPE-1.0)   # fwhm, in units where [0, NSPE-1] is the cloud radius
        except:
            print("*** ConvolveSpectra1D fails: angle_as not given as argument and not found in the spectrum file")
            sys.exit(0)
    else:
        fwhm     =  (fwhm_as/angle_as) * (NSPE-1.0)  # fwhm [number of offset steps]
    ###
    platform, device, context, queue, mf = InitCL(GPU, platforms)
    SPE_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*NSPE*NCHN)
    CON_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*NSPE*NCHN)
    INSTALL_DIR  =  os.path.dirname(os.path.realpath(__file__))    
    source       =  open(INSTALL_DIR+"/kernel_convolve_spectra_1d.c").read()
    program      =  cl.Program(context, source).build()
    kernel_con   =  program.Convolve
    kernel_con.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, np.float32, None, None])
    LOCAL        =  [ 1, 32 ][GPU>0]
    GLOBAL       =  (NSPE//LOCAL+1)*LOCAL
    cl.enqueue_copy(queue, SPE_buf, SPE)
    kernel_con(queue, [GLOBAL,], [LOCAL,], NSPE, NCHN, int(samples)//2, fwhm, SPE_buf, CON_buf)
    cl.enqueue_copy(queue, SPE, CON_buf)
    ###
    ofilename    =  filename.replace('.spe','')+'_convolved.spe'
    fp           =  open(ofilename, 'wb')
    asarray([NSPE, NCHN], int32).tofile(fp)
    asarray([V0, DV],float32).tofile(fp)
    asarray(SPE, float32).tofile(fp)
    if (INI!=None):
        pickle.dump(INI, fp)
    fp.close()
    return V0+arange(NCHN)*DV, SPE



def MakeEmptyFitsDim(lon, lat, pix, m, n, dv=0.0, nchn=0, sys_req='fk5'):
    """
    Make an empty fits object.
    Inputs:
        lon, lat  = centre coordinates of the field [radians]
        pix       = pixel size [radians]
        m, n      = width and height in pixels
        sys_req   = coordinate system, 'fk5' or 'galactic'
    """
    import astropy.io.fits as pyfits
    A         = zeros((n, m), float32)
    hdu       = pyfits.PrimaryHDU(A)
    F         = pyfits.HDUList([hdu])
    F[0].header.update(CRVAL1 =  lon*180.0/pi)
    F[0].header.update(CRVAL2 =  lat*180.0/pi)
    F[0].header.update(CDELT1 = -pix*180.0/pi)
    F[0].header.update(CDELT2 =  pix*180.0/pi)
    F[0].header.update(CRPIX1 =  0.5*(m+1))
    F[0].header.update(CRPIX2 =  0.5*(n+1))
    if (sys_req=='galactic'):
        F[0].header.update(CTYPE1   = 'GLON-TAN')
        F[0].header.update(CTYPE2   = 'GLAT-TAN')
        F[0].header.update(COORDSYS = 'GALACTIC')
    else:
        F[0].header.update(CTYPE1   = 'RA---TAN')
        F[0].header.update(CTYPE2   = 'DEC--TAN')
        F[0].header.update(COORDSYS = 'EQUATORIAL')
        F[0].header.update(EQUINOX  = 2000.0)
    if (nchn>0):
        F[0].data = zeros((nchn, n, m), float32)
        F[0].header['NAXIS' ] =  3
        F[0].header['NAXIS3'] =  nchn
        F[0].header['CRPIX3'] =  0.5*(nchn+1.0)
        F[0].header['CRVAL3'] =  0.0
        F[0].header['CDELT3'] =  dv
        F[0].header['CTYPE3'] = 'velocity'
    else:
        F[0].data = zeros((n, m), float32)
    return F

