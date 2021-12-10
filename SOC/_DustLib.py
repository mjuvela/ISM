import os, sys
import numpy as np
from numpy import *
from scipy.integrate import quad
from scipy.interpolate import interp1d

METHOD      = 'CRT'
# METHOD      = 'native'

DUSTEM_DIR  = '/home/mika/tt/dustem4.0_web'
# DUSTEM_DIR  = '/dev/shm/dustem4.0_web'


# EPSABS 1e-25 ok for DustEM 0.5%,  1e-23 ok within 1%
EPSABS  = 1.0e-25  # absolute tolerance of integrals
EPSABS2 = 1.0e-19

# for testing only !!
# EPSABS  = 1.0e-23  # absolute tolerance of integrals
# EPSABS2 = 1.0e-19

import time

try:
    import aplpy.io.fits as pyfits
except:
    pass
try:
    import pyfits
except:
    pass

AMU = 1.6605e-24
mH  = 1.0079*AMU
C_LIGHT = 29979245800.0


def um2f(um):
    """
    Convert wavelength [um] to frequency [Hz]
    """
    return C_LIGHT/(um*1.0e-4)

def f2um(f):
    """
    Convert frequency [Hz] to wavelenth [um]
    """
    return 1.0e4*C_LIGHT/f

def isscalar(x):
    return (0==len(asarray(x).shape))
    

if (0): # LOGLOG INTERPOLATION, OK FOR KABS (==CRT), KSCA LOW???
    def IP(x, y):
        return interp1d(log(x), log(y), kind='linear')
    def get_IP(x, ip):
        return exp(ip(log(x)))
if (1): # LINEAR INTERPOLATION
    def IP(x, y):
        return interp1d(x, y, kind='linear')
    def get_IP(x, ip):
        return ip(x)
if (0): # ARGUMENT ON LOG SCALE
    def IP(x, y):
        return interp1d(log(x), y, kind='linear')
    def get_IP(x, ip):
        return ip(log(x))
if (0): # RESULT ON LOG SCALE  -- BEST FOR KABS, KSCA LOWER!
    def IP(x, y):
        return interp1d(x, log(y), kind='linear')
    def get_IP(x, ip):
        return exp(ip(x))

    
def ProductIntegrand(x, F1, F2):
    return F1(x)*F2(x)

def ProductIntegrandA(x, F1, F2):
    return F1(x)*F2(x) * pi*x*x

def ProductIntegrand_LILO(x, F1, F2):
    return F1(x)*( exp(F2(log(x))) )

def ProductIntegrand_LILOA(x, F1, F2):
    return F1(x)*( exp(F2(log(x))) ) * pi*x*x

def ProductIntegrand_LOLO(x, F1, F2):
    return exp(F1(log(x))) * ( exp(F2(log(x))))

def ProductIntegrand_LOLOA(x, F1, F2):
    return exp(F1(log(x))) * exp(F2(log(x))) * pi*x*x


def HGRealisation(g):
    # generate random value of theta angle
    # of course includes the sin(theta) scaling of solid angle
    tmp        =  (1.0-g*g)/(1.0-g+2.0*g*u) 
    cos_theta  =  ((1.0+g*g) - tmp*tmp)/(2.0*g) 
    return arccos(theta)

def HenyeyGreenstein(theta, g):
    # probability per solid angle (not per dtheta!)
    #   INTEGRAL 0...PI OF  [ HenyeyGreenstein x sin(theta) ] dtheta dphi == 1.0
    p = (1.0/(4.0*pi)) * (1.0-g*g) / (1.0+g*g-2.0*g*cos(theta))**1.5
    return p
    
def HG_per_theta(theta, g):
    # return probability per dtheta
    #   INTEGRAL 0...PI of  HG_per_theta dtheta == 1.0
    return 2.0*pi * sin(theta) * HenyeyGreenstein(theta, g)


def fun_exp(x, ip):
    return 10.0**(ip(log10(x)))
    

class DustO:
    

    def __init__(self, Foptical, Fsizes=None):
        # eqdust + GRAIN_DENSITY + a + rows + data
        #  if Fsizes==None, assume standard CRT format
        #  otherwise assume TRUST format
        self.EFF = False
        # else -- read optical properties and size distributions from different files
        self.GRAIN_DENSITY = 1.0e-7  # DUMMY !!
        self.GRAIN_SIZE    = 1.0e-4  # DUMMY !!
        # read the size distribution
        # we assume that linear interpolation is good enough (we will integrate over this function)
        d           = np.loadtxt(Fsizes)           # first row = nsizes, nebins
        self.SIZE_A = d[1:,0].copy() * 1.0e-4   # [cm]  -- SIZE DISTRIBUTION USED
        self.SIZE_F = d[1:,1].copy()            # 1/cm/H   = dn/da/H -- SIZE DISTRIBUTION USED
        # ipFRAC gives d(grains)/da / H,  [a]=cm
        # extend definition of ipFRAC to cover all possible values -- zero outside input file range
        xx, yy = asarray(self.SIZE_A, float64), asarray(self.SIZE_F, float64)
            
        # define also bin limits for the size distribution - assuming logarithmic bins
        self.SIZE_BO = zeros(self.NSIZE+1, float32)
        beta  =  self.SIZE_A[1] / self.SIZE_A[0]  # log step
        self.SIZE_BO[0]   =  self.SIZE_A[0] / sqrt(beta)
        self.SIZE_BO[1:]  =  self.SIZE_A * sqrt(beta)
        
        self.ipFRAC     = interp1d(xx, yy, bounds_error=True, fill_value=0.0)
        self.ipFRAC_LL  = interp1d(log(xx), log(yy), bounds_error=True, fill_value=0.0)
        # read optical data
        lines       = open(Foptical).readlines()
        s           = lines[1].split()
        self.QNSIZE = int(s[0])                   # sizes for which optical data given
        self.QNFREQ = int(s[1])                   # frequencies for which optical data given
        self.QSIZE  = zeros(self.QNSIZE, float64) # sizes for which Q defined
        self.QFREQ  = zeros(self.QNFREQ, float64)
        self.OPT    = zeros((self.QNSIZE, self.QNFREQ, 4), float64)     #  um, Q_abs, Q_sca, g
        row         = 3
        for isize in range(self.QNSIZE):
            self.QSIZE[isize] = float(lines[row].split()[0]) * 1.0e-4  # [cm]
            A  = pi*(self.QSIZE[isize])**2.0     # grain area [cm2]
            # DO NOT MULTIPLY WITH GRAIN SIZE BEFORE Q IS INTERPOLATED OVER GRAIN SIZES !!!
            row += 2
            for ifreq in range(self.QNFREQ):
                s = lines[row].split()
                # OPT[isize, ifreq, 4] = [ um, Kabs, Ksca, g ], grain density 1/H included later
                self.OPT[isize, ifreq, :] = [float(s[1]), float(s[2]), float(s[3]), float(s[5]) ]
                row += 1
            row += 1        
        self.QFREQ = um2f(self.OPT[0,:,0])  # DECREASING order of frequency
        # in self.OPT, data already converted from from Qabs, Qsca  to  Kabs, Kscat
        self.AMIN = self.QSIZE_A[0]
        self.AMAX = self.QSIZE_A[-1]
        #
        print('  --- SIZE DISTRIBUTION SIZE_A  %12.4e - %12.4e' % (self.SIZE_A[0], self.SIZE_A[-1]))
        print('  --- OPTICAL DATA      SIZE    %12.4e - %12.4e' % (self.QSIZE[0], self.QSIZE[-1]))
        if (0):
            print('*** WARNING **** AMIN %12.4e < OPTICAL DATA AMIN %12.4e' % (self.AMIN, self.QSIZE[0]))
            print('MOVING AMIN %12.4e UP TO %12.4e' % (self.AMIN, self.QSIZE[0]))
            self.AMIN = self.QSIZE[0]
        else:
            if (self.AMIN<self.QSIZE[0]):
                print('*** WARNING **** AMIN %12.4e < OPTICAL DATA AMIN %12.4e' % (self.AMIN, self.QSIZE[0]))
                scale            = (self.AMIN/self.QSIZE[0])**2.0
                self.OPT[0,:,1] *= scale       # Kabs --- NOT YET SCALED WITH pi*a^2
                self.OPT[0,:,2] *= scale       # Ksca --- NOT YET SCALED WITH pi*a^2
                self.QSIZE[0]    = self.AMIN   # optical data "extrapolated" to self.AMIN
        if (self.AMAX>self.QSIZE[-1]):
            print('*** ERROR *** AMAX %12.4e > OPTICAL DATA AMAX %12.4e' % (self.AMAX, self.QSIZE[-1]))
            sys.exit()

        
        
    def Ksca(self, freq):
        global METHOD
        print("*** Ksca ***", METHOD)
        if (METHOD=='CRT'):
            FUN = self.KscaCRT
        else:
            FUN = self.KscaTRUST
        if (isscalar(freq)):
            return clip(FUN(freq), 1.0e-99, 1.0e10)
        else:
            res = zeros(len(freq), float64)
            for i in range(len(freq)):
                res[i] = FUN(freq[i])
        return clip(res, 1.0e-99, 1.0e10)

    
    def Kabs(self, freq):
        global METHOD
        if (METHOD=='CRT'):            
            FUN = self.KabsCRT
        else:
            FUN = self.KabsTRUST
        if (isscalar(freq)):
            return FUN(freq)
        else:
            res = zeros(len(freq), float64)
            for i in range(len(freq)):
                res[i] = FUN(freq[i])
        return res

    
    def Gsca(self, freq):
        if (isscalar(freq)):
            return self.GTRUST(freq)
        else:
            res = zeros(len(freq), float64)
            for i in range(len(freq)):
                res[i] = self.GTRUST(freq[i])
        return res
    
    
    def GetScatteringFunctionInterpoler(self, freq):
        # probability dp/dtheta, including sin(theta)
        g      = self.Gsca(freq)
        the    = linspace(0.0, pi, 3000)
        the[0] = 1.0e-8
        y      = (1.0/(4.0*pi)) * sin(the) * (1.0-g*g) * pow(1.0+g*g-2.0*g * cos(the), -1.5)
        the[0] = 0.0
        return interp1d(the, y)

    
    
    def KabsTRUST(self, freq, amin=-1, amax=-1):
        # Return KABS integrated over the size distribution
        #  ... more similar to CRT/DUSTEM
        # find frequency indices surrounding freq
        if (amin<0.0): amin = self.AMIN
        if (amax<0.0): amax = self.AMAX
        i          = argmin(abs(self.QFREQ-freq))  # note -- self.QFREQ is in decreasing order
        if (self.QFREQ[0]<self.QFREQ[1]):
            print('KabsTRUST FREQUENCIES IN INCREASING ORDER -- MUST BE DECREASING !!!')
            sys.exit()
        if (self.QFREQ[i]<freq): i-=1    #  i -= 1  is step to higher frequency!
        j          = i+1                 #  j>i,  lower frequency
        # requested frequency between indices i and j
        if (i<0): i=0
        if (j>=self.QNFREQ): j= self.QNFREQ-1
        if (i==j):
            wj = 0.5
        else:
            wj     = (self.QFREQ[i]-freq)/(self.QFREQ[i]-self.QFREQ[j])
            # wj     = (log(self.QFREQ[i])-log(freq))/(log(self.QFREQ[i])-log(self.QFREQ[j]))
        wi     = 1.0-wj
        #   OPT[isize, ifreq, 0:4] = [ um, Kabs, Ksca, g ]
        tmp        = ravel(self.OPT[:, i, 1])        # OPT[isize, ifreq, 4] -> frequency i, Qabs
        ipK        = interp1d(self.QSIZE, tmp)       # Qabs for given size
        # integrate over sizes = self.ipFRAC * ipK
        resi, dres = quad(ProductIntegrandA, amin, amax, args=(self.ipFRAC, ipK), 
        epsrel=1.0e-12, epsabs=EPSABS)
        tmp        = ravel(self.OPT[:, j, 1])
        ipK        = interp1d(self.QSIZE, tmp)
        resj, dres = quad(ProductIntegrandA, amin, amax, args=(self.ipFRAC, ipK), 
        epsrel=1.0e-12, epsabs=EPSABS)
        if (0):
            print('      res  %12.5e  +/-  %12.5e' % (resj, dres))
        # kabs       = exp(wi*log(resi) + wj*log(resj))
        kabs       = wi*resi + wj*resj
        return kabs

    
    def KscaTRUST(self, freq):
        # Return KSCA integrated over the size distribution
        print("*** KscaTRUST")
        i          = argmin(abs(self.QFREQ-freq))
        if (self.QFREQ[i]<freq): i-=1
        j          = i+1
        i          = clip(i, 0, self.QNFREQ-1)
        j          = clip(j, 0, self.QNFREQ-1)
        if (i==j):
            wj     = 0.5
        else:
            # wj     = (log(self.QFREQ[i])-log(freq))/(log(self.QFREQ[i])-log(self.QFREQ[j]))
            wj     = (self.QFREQ[i]-freq) / (self.QFREQ[i]-self.QFREQ[j])
        wi         = 1.0-wj
        ## OPT[isize, ifreq, 4] = [ um, Qabs, Qsca, g ]
        tmp        = asarray(ravel(self.OPT[:, i, 2]), float64)   # OPT[isize, ifreq, 4] -> frequency i, Qsca
        ipKi       = interp1d(self.QSIZE, tmp, kind='linear')                 # Qabs for given size
        ipKi_LL    = interp1d(log(self.QSIZE), log(tmp), kind='linear')   # Qabs for given size
        tmp        = ravel(self.OPT[:, j, 2])
        ipKj       = interp1d(self.QSIZE, tmp, kind='linear')
        ipKj_LL    = interp1d(log(self.QSIZE), log(tmp), kind='linear')
        #
        if (1): # LILI -- interpolation over sizes
            resi, dres = quad(ProductIntegrandA, self.AMIN, self.AMAX, args=(self.ipFRAC, ipKi), 
            epsrel=1.0e-10, epsabs=EPSABS)
            resj, dres = quad(ProductIntegrandA, self.AMIN, self.AMAX, args=(self.ipFRAC, ipKj), 
            epsrel=1.0e-10, epsabs=EPSABS)
        if (0): # LOLO -- interpolation over sizes --- further away from DUSTEM curves
            resi, dres = quad(ProductIntegrand_LOLOA, self.AMIN, self.AMAX, args=(self.ipFRAC_LL, ipKi_LL), 
            epsrel=1.0e-10, epsabs=EPSABS)
            resj, dres = quad(ProductIntegrand_LOLOA, self.AMIN, self.AMAX, args=(self.ipFRAC_LL, ipKj_LL), 
            epsrel=1.0e-20, epsabs=EPSABS)
            print('        KscaTrust  %12.5e +/- %12.5e' % (resj, dres))
        #    ksca       = exp(wi*log(resi) + wj*log(resj))
        ksca       = wi*resi + wj*resj
        return ksca


    
    

    def KabsTRUST_X(self, freq):
        # Return KABS integrated over the size distribution
        # find frequency indices surrounding freq
        i          = argmin(abs(self.QFREQ-freq))  # note -- self.QFREQ is in decreasing order
        if (self.QFREQ[0]<self.QFREQ[1]):
            print('KabsTRUST FREQUENCIES IN INCREASING ORDER -- MUST BE DECREASING !!!')
            sys.exit()
        if (self.QFREQ[i]<freq): i-=1    #  i -= 1  is step to higher frequency!
        j          = i+1                #  j>i,  lower frequency
        # requested frequency between indices i and j
        if (i<0): i=0
        if (j>=self.QNFREQ): j= self.QNFREQ-1
        if (i==j):
            wi, wj = 0.5, 0.5
        else:
            # wj     = (log(self.QFREQ[i])-log(freq))/(log(self.QFREQ[i])-log(self.QFREQ[j]))
            wj     = (self.QFREQ[i]-freq)/(self.QFREQ[i]-self.QFREQ[j])
            wi     = 1.0-wj
        # extract kabs for frequency i, as function of grain size -- self.OPT[ um, Qabs, Qsca, g ]
        #   OPT[ isize, ifreq, 0:4] = [ um, Qabs, Qsca, g ]
        # integrate over sizes = self.ipFRAC * ipK
        #  --- first frequency
        tmp        = ravel(self.OPT[:, i, 1])       # OPT[isize, ifreq, 4] -> frequency i, Qabs
        ipK_LL     = interp1d(log(self.QSIZE), log(tmp))       # Qabs for given size
        res, dres  = quad(ProductIntegrand_LOLOA, self.AMIN, self.AMAX, args=(self.ipFRAC_LL, ipK_LL),
        epsrel=1e-10, epsabs=EPSABS)
        kabs       = wi*log(res)
        #  --- second frequency
        tmp        = ravel(self.OPT[:, j, 1])
        ipK_LL     = interp1d(log(self.QSIZE), log(tmp))
        res, dres  = quad(ProductIntegrand_LOLOA, self.AMIN, self.AMAX, args=(self.ipFRAC_LL, ipK_LL), 
        epsrel=1e-10, epsabs=EPSABS)
        kabs      += wj*log(res)
        return 10.0**kabs
    
    
    
    
    def KscaTRUST_X(self, freq):
        # Return KSCA integrated over the size distribution
        print("*** KscaTRUST_X")
        i          = argmin(abs(self.QFREQ-freq))
        if (self.QFREQ[i]<freq): i-=1
        j          = i+1
        if (i<0): i=0
        if (j>=self.QNFREQ): j= self.QNFREQ-1
        if (i==j):
            wi, wj = 0.5, 0.5
        else:
            wj     = (log(self.QFREQ[i])-log(freq))/(log(self.QFREQ[i])-log(self.QFREQ[j]))
            wi     = 1.0-wj
        ## OPT[isize, ifreq, 4] = [ um, Qabs, Qsca, g ]
        tmp        = ravel(self.OPT[:, i, 2])    # OPT[isize, ifreq, 4] -> frequency i, Qsca
        ipKi_LL    = interp1d(log(self.QSIZE), log(tmp))   # Kabs for given size
        tmp        = ravel(self.OPT[:, j, 2])
        ipKj_LL    = interp1d(log(self.QSIZE), log(tmp))
        # LOLO -- interpolation over sizes
        resi, dres = quad(ProductIntegrand_LOLOA, self.AMIN, self.AMAX, args=(self.ipFRAC_LL, ipKi_LL), 
        epsrel=1.0e-10, epsabs=EPSABS)
        resj, dres = quad(ProductIntegrand_LOLOA, self.AMIN, self.AMAX, args=(self.ipFRAC_LL, ipKj_LL), 
        epsrel=1.0e-10, epsabs=EPSABS)
        ksca       = exp(wi*log(resi) + wj*log(resj))
        
        return ksca
    

    
    
    
    
    def KabsCRT(self, freq):
        # Return KABS .... USE SUMMATION WITH LOG-SPACED SIZE GRID AND self.CRT_SFRAC
        # find frequency indices surrounding freq   @@@
        i          = argmin(abs(self.QFREQ-freq))  # note -- self.QFREQ is in decreasing order
        if (self.QFREQ[i]<freq): i-=1              #  i -= 1  is step to higher frequency!
        j          = i+1                           #  j>i,  lower frequency
        # requested frequency between indices i and j
        i   = clip(i, 0, self.QNFREQ-1)
        j   = clip(j, 0, self.QNFREQ-1)
        if (i==j):
            wj   = 0.5
        else:
            # wj     = (log(self.QFREQ[i])-log(freq))/(log(self.QFREQ[i])-log(self.QFREQ[j]))
            wj   = (self.QFREQ[i]-freq)  / (self.QFREQ[i]-self.QFREQ[j])
        wi    = 1.0-wj
        # we have to interpolate between the sizes given in Q files == in the OPT array
        #   from grid QSIZE[NQZIE] to grid   SIZE_A[NSIZE]
        # To be consistent with CRT (and dustem) need to interpolate
        # Q ***BEFORE*** scaling that with a^2 !!!!!!!!!!!!!!!!!!!!!
        y1    =  self.OPT[:, i, 1]
        y2    =  self.OPT[:, j, 1]
        ip1   =  IP(self.QSIZE, y1.copy())
        ip2   =  IP(self.QSIZE, y2.copy())
        ## print('ip1 %11.4e ... %11.4e  --- ask for %11.4e ... %11.4e' % (min(self.QSIZE), max(self.QSIZE), min(self.SIZE_A), max(self.SIZE_A)))
        y1    =  get_IP(self.SIZE_A, ip1)
        y2    =  get_IP(self.SIZE_A, ip2)
        y     =  wi*y1 + wj*y2
        kabs  =  sum(self.CRT_SFRAC*y* (pi*self.SIZE_A**2.0) ) # INCLUDES GRAIN_DENSITY IN CRT_SFRAC
        return   kabs
    
    
    
    def KscaCRT(self, freq):
        # Return KSCA integrated over the size distribution @@@
        print("*** KscaCRT")
        i          = argmin(abs(self.QFREQ-freq))  # note -- self.QFREQ is in decreasing order
        if (self.QFREQ[i]<freq): i-=1              #  i -= 1  is a step to higher frequency!
        j          = i+1
        #             i   <       <       j   
        #       QFREQ[i]  >  freq > QFREQ[j]  
        # requested frequency is between indices i and j
        i   = clip(i, 0, self.QNFREQ-1)
        j   = clip(j, 0, self.QNFREQ-1)
        if (i==j):
            wj = 0.5
        else:
            # wj     = (log(self.QFREQ[i])-log(freq))/(log(self.QFREQ[i])-log(self.QFREQ[j]))
            wj     = (self.QFREQ[i]-freq)  / (self.QFREQ[i]-self.QFREQ[j])            
        wi    = 1.0-wj
        # we have to interpolate between the sizes given in Q files == in the OPT array
        #  =============  OPT[size, freq, 4] = [um, Kabs, Kaca, g]
        #  note -- this is interpolation between size and Q*pi*a**2, not size and Q (no effect!)
        y1    =  self.OPT[:, i, 2] 
        y2    =  self.OPT[:, j, 2] 
        ip1   =  IP(self.QSIZE, y1.copy())
        ip2   =  IP(self.QSIZE, y2.copy())
        y1    =  get_IP(self.SIZE_A, ip1)
        y2    =  get_IP(self.SIZE_A, ip2)
        y     =  wi*y1 + wj*y2   # interpolated Qsca
        # the integral as a sum, CRT_SFRAC includes scaling by mass ratio
        ksca  =  sum(self.CRT_SFRAC*y*(pi*self.SIZE_A**2.0))  # CRT_SFRAC INCLUDES GRAIN_DENSITY
        print("      KscaCRT   %.3e x %.3e    + %.3e x %.3e   .... %12.4e" % (wi, y1[1], wj, y2[1], ksca))
        return   ksca
    
    
    
    def GTRUST(self, freq):
        # Effective value of g as <Kscat*g> over the size distribution
        i        = argmin(abs(self.QFREQ-freq))
        if (self.QFREQ[i]<freq): i-=1
        j        = i+1
        i        = clip(i, 0, self.QNFREQ-1)
        j        = clip(j, 0, self.QNFREQ-1)
        if (i==j):
            wj   = 0.5
        else:
            wj   = (self.QFREQ[i]-freq)/(self.QFREQ[i]-self.QFREQ[j])
        wi         = 1.0-wj
        tmp        = ravel(self.OPT[:, i, 3] * self.OPT[:,i,2])  # g * Qsca
        ipK        = interp1d(self.QSIZE, tmp, kind='linear')
        # ProductIntegrandA includes the scaling by grain area !!
        #     integral of ---  SIZE_F * Ksca * g * a^2
        res, dres  = quad(ProductIntegrandA, self.AMIN, self.AMAX, args=(self.ipFRAC, ipK), epsrel=1.0e-10, epsabs=EPSABS2)
        ipK        = interp1d(self.QSIZE, self.OPT[:, i, 2], kind='linear')     # Qsca only
        #     integral of --- SIZE_F * Ksca * a^2
        norm, dres = quad(ProductIntegrandA, self.AMIN, self.AMAX, args=(self.ipFRAC, ipK), epsrel=1.0e-10, epsabs=EPSABS2)
        # print(' %.2e %.2e cm =>  res %10.3e, norm %10.3e' % (self.AMIN, self.AMAX, res, norm))
        g          = wi * res/(norm+1.0e-40)
        #
        tmp        = ravel(self.OPT[:, j, 3]*self.OPT[:,j,2])  # g * Qsca
        ipK        = interp1d(self.QSIZE, tmp)
        res, dres  = quad(ProductIntegrandA, self.AMIN, self.AMAX, args=(self.ipFRAC, ipK), 
        epsrel=1.0e-10, epsabs=EPSABS2)
        ipK        = interp1d(self.QSIZE, self.OPT[:, j, 2])
        norm, dres = quad(ProductIntegrandA, self.AMIN, self.AMAX, args=(self.ipFRAC, ipK), 
        epsrel=1.0e-10, epsabs=EPSABS2)
        g         += wj * res/(norm+1.0e-40)
        if (0):
            print('norm w %10.3e %10.3e ... res %10.3e norm %10.3e  g = %10.3e' % (wi, wj, res, norm, g))
        return g

    
    
    def DSF(self, freq, theta, SIN_WEIGHT, size_sub_bins=500):
        # Return discretised scattering function {theta, SF}
        # calculate Kscat weighted average scattering function over all sizes
        # note -- this is SF(theta) (i.e., per dtheta, not per solid angle)
        i          = argmin(abs(self.QFREQ-freq))
        if (self.QFREQ[i]<freq): i-=1    #  i = higher frequency
        j          = i+1                #  j = lower frequency
        if (i<0): i=0
        if (j>=self.QNFREQ): j= self.QNFREQ-1
        if (i==j):
            wi, wj = 0.5, 0.5
        else:
            wj     = (self.QFREQ[i]-freq)/(self.QFREQ[i]-self.QFREQ[j])
            wi     = 1.0-wj
        #  -- weight frequencies i and j with wi and wj
        #  -- the scattering function is weighted by SFRAC*KSCAT
        # OPT[isize, ifreq, 4] = { um, Kabs, Ksca, g }
        total      = zeros(len(theta), float64)
        # data on the lower frequency = i
        # print('ipFRAC CALLED ', self.QSIZE[0], self.QSIZE[-1])
        W          =  wi * self.OPT[:,i,2] * self.ipFRAC(self.QSIZE) * self.QSIZE**2.0  # wi * FRAC * Kscat = weight
        WG         =   W * self.OPT[:,i,3]                             # weight * g
        ipW        = interp1d(self.QSIZE, W,  bounds_error=True, fill_value=0.0)
        ipWG       = interp1d(self.QSIZE, WG, bounds_error=True, fill_value=0.0)
        # calculate Integral(ipWG)/Integral(ipW) for narrow size bins
        A          = logspace(log10(self.AMIN), log10(self.AMAX), size_sub_bins+1)
        WEIGHT     = 0.0
        for k in range(size_sub_bins):  # loop over narrow grain size intervals
            # for each bin, calculate its relative weight
            w, dr  = quad(ipW,  A[k], A[k+1], epsrel=1.0e-10, epsabs=EPSABS2)    #  weight = integral of (wi * Kscat * FRAC)
            # the value of g averaged over the bin
            r1, dr = quad(ipWG, A[k], A[k+1], epsrel=1.0e-10, epsabs=EPSABS2)
            g      = r1/w      # <g> within a narrow bin
            # add to total  w*SF(g)
            if (SIN_WEIGHT):
                total += w * HG_per_theta(theta, g)     # sin(theta) IS included !!
            else:
                total += w * HenyeyGreenstein(theta, g) # sin(theta) NOT included !!
            WEIGHT += w
        # the same for the higher frequency
        W          = wj * self.OPT[:,j,2] * self.ipFRAC(self.QSIZE) * self.QSIZE**2.0   # wi * FRAC * Kscat = weight
        WG         =  W * self.OPT[:,j,3]                            # weight * g
        ipW        = interp1d(self.QSIZE, W,  bounds_error=True, fill_value=0.0)
        ipWG       = interp1d(self.QSIZE, WG, bounds_error=True, fill_value=0.0)
        for k in range(size_sub_bins):  # loop over narrow grain size intervals
            w, dr  = quad(ipW,  A[k], A[k+1], epsrel=1.0e-10, epsabs=EPSABS2)
            r1, dr = quad(ipWG, A[k], A[k+1], epsrel=1.0e-10, epsabs=EPSABS2)
            g      = r1/w      # <g> within a narrow bin
            if (SIN_WEIGHT):
                total += w * HG_per_theta(theta, g)     # sin(theta) is included !!
            else:
                total += w * HenyeyGreenstein(theta, g) # sin(theta) NOT included !!
            WEIGHT += w
        total /= WEIGHT        # retain the normalisation of HG_per_theta and HenyeyGreenstein
        if (1):
            # double check normalisation
            if (SIN_WEIGHT):
                ip = interp1d(theta, total)
                I  = quad(ip, 0.0, pi, epsrel=1.0e-10, epsabs=EPSABS2)[0]
                total *= 1.0/I
            else:
                ip  = interp1d(theta, total*sin(theta))
                I   = 2.0*pi * quad(ip, 0.0, pi, epsrel=1.0e-10, epsabs=EPSABS2)[0]
                total *= 1.0/I
        #
        return total
        
        

    def DSF2(self, freq, cos_theta, SIN_WEIGHT, size_sub_bins=500):
        # Return discretised scattering function {theta, SF}
        # calculate Kscat weighted average scattering function over all sizes
        # note -- this is SF(theta) (i.e., per dtheta, not per solid angle)
        i          = argmin(abs(self.QFREQ-freq))
        if (self.QFREQ[i]<freq): i-=1    #  i = higher frequency
        j          = i+1                #  j = lower frequency
        if (i<0): i=0
        if (j>=self.QNFREQ): j= self.QNFREQ-1
        if (i==j):
            wi, wj = 0.5, 0.5
        else:
            wj     = (self.QFREQ[i]-freq)/(self.QFREQ[i]-self.QFREQ[j])
            wi     = 1.0-wj
        #  -- weight frequencies i and j with wi and wj
        #  -- the scattering function is weighted by SFRAC*KSCAT
        # OPT[isize, ifreq, 4] = { um, Kabs, Ksca, g }
        total      = zeros(len(cos_theta), float64)
        # data on the lower frequency = i
        W          = wi * self.OPT[:,i,2] * self.ipFRAC(self.QSIZE) * self.QSIZE**2   # wi * FRAC * Kscat = weight
        WG         =  W * self.OPT[:,i,3]                            # weight * g
        ipW        = interp1d(self.QSIZE, W,  bounds_error=True, fill_value=0.0)
        ipWG       = interp1d(self.QSIZE, WG, bounds_error=True, fill_value=0.0)
        # calculate Integral(ipWG)/Integral(ipW) for narrow size bins
        A          = logspace(log10(self.AMIN), log10(self.AMAX), size_sub_bins+1)
        WEIGHT     = 0.0
        for k in range(size_sub_bins):  # loop over narrow grain size intervals
            # for each bin, calculate its relative weight
            w, dr  = quad(ipW,  A[k], A[k+1], epsrel=1.0e-10, epsabs=EPSABS2)    #  weight = integral of (wi * Kscat * FRAC)
            # the value of g averaged over the bin
            r1, dr = quad(ipWG, A[k], A[k+1], epsrel=1.0e-10, epsabs=EPSABS2)
            g      = r1/w      # <g> within a narrow bin
            # add to total  w*SF(g)
            if (SIN_WEIGHT):
                total += w * HG_per_theta(arccos(cos_theta), g)     # sin(theta) IS included !!
            else:
                total += w * HenyeyGreenstein(arccos(cos_theta), g) # sin(theta) NOT included !!
            WEIGHT += w
        # the same for the higher frequency
        W          = wj * self.OPT[:,j,2] * self.ipFRAC(self.QSIZE) * self.QSIZE**2   # wi * FRAC * Kscat = weight
        WG         =  W * self.OPT[:,j,3]                             # weight * g
        ipW        = interp1d(self.QSIZE, W,  bounds_error=True, fill_value=0.0)
        ipWG       = interp1d(self.QSIZE, WG, bounds_error=True, fill_value=0.0)
        for k in range(size_sub_bins):  # loop over narrow grain size intervals
            w, dr  = quad(ipW,  A[k], A[k+1], epsrel=1.0e-10, epsabs=EPSABS2)
            r1, dr = quad(ipWG, A[k], A[k+1], epsrel=1.0e-10, epsabs=EPSABS2)
            g      = r1/w      # <g> within a narrow bin
            if (SIN_WEIGHT):
                total += w * HG_per_theta(arccos(cos_theta), g)     # sin(theta) is included !!
            else:
                total += w * HenyeyGreenstein(arccos(cos_theta), g) # sin(theta) NOT included !!
            WEIGHT += w
        total /= WEIGHT        # retain the normalisation of HG_per_theta and HenyeyGreenstein
        if (1):
            # double check normalisation
            if (SIN_WEIGHT):
                ip = interp1d(arccos(cos_theta), total)
                I  = quad(ip, 0.0, pi, epsrel=1.0e-10, epsabs=EPSABS2)[0]
                total *= 1.0/I
            else:
                theta = arccos(cos_theta)
                ip  = interp1d(theta, total*sin(theta))
                I   = 2.0*pi * quad(ip, 0.0, p, epsrel=1.0e-10, epsabs=EPSABS2)[0]
                total *= 1.0/I
        #
        return total
        
        
    def DSF_simple(self, freq, theta, SIN_WEIGHT):
        total      = zeros(len(theta), float64)
        # data on the lower frequency = i
        g  = self.Gsca(freq)  # now defined also for GSETDust
        if (SIN_WEIGHT):
            total = HG_per_theta(theta, g)     # sin(theta) IS included !!
        else:
            total = HenyeyGreenstein(theta, g) # sin(theta) NOT included !!
        if (1):
            # double check normalisation
            if (SIN_WEIGHT):
                ip = interp1d(theta, total)
                I  = quad(ip, 0.0, pi, epsrel=1.0e-10, epsabs=EPSABS2)[0]
                total *= 1.0/I
            else:
                ip  = interp1d(theta, total*sin(theta))
                I   = 2.0*pi * quad(ip, 0.0, pi, epsrel=1.0e-10, epsabs=EPSABS2)[0]
                total *= 1.0/I
        #
        return total
        
        

    def DSF2_simple(self, freq, cos_theta, SIN_WEIGHT):
        # Return discretised scattering function {theta, SF}
        # calculate Kscat weighted average scattering function over all sizes
        # note -- this is SF(theta) (i.e., per dtheta, not per solid angle)
        total      = zeros(len(cos_theta), float64)
        g          = self.Gsca(freq)
        if (SIN_WEIGHT):
            total  = HG_per_theta(arccos(cos_theta), g)     # sin(theta) IS included !!
        else:
            total  = HenyeyGreenstein(arccos(cos_theta), g) # sin(theta) NOT included !!
        if (1):
            # double check normalisation
            if (SIN_WEIGHT):
                ip = interp1d(arccos(cos_theta), total)
                I  = quad(ip, 0.0, pi, epsrel=1.0e-10, epsabs=EPSABS2)[0]
                total *= 1.0/I
            else:
                theta = arccos(cos_theta)
                ip    = interp1d(theta, total*sin(theta))
                I     = 2.0*pi * quad(ip, 0.0, pi, epsrel=1.0e-10, epsabs=EPSABS2)[0]
                total *= 1.0/I
        #
        return total

    
    
    
    
    


class EffDustO(DustO):
    """
    Effective dust (no size distributions, no stochastic heating)
    """
    def __init__(self, Foptical, Fsizes=None):
        self. EFF   = True
        d           = open(Foptical).readlines()
        self.GRAIN_DENSITY = float(d[1].split()[0])
        self.GRAIN_SIZE    = float(d[2].split()[0])
        d           = loadtxt(Foptical, skiprows=4)
        self.QFREQ   = d[:,0]
        self.QNFREQ  = len(self.QFREQ)
        self.G      = d[:,1]
        self.QABS   = d[:,2]
        self.QSCA   = d[:,3]
        k           = self.GRAIN_DENSITY * pi*(self.GRAIN_SIZE)**2.0
        self.KABS   = k * self.QABS
        self.KSCA   = k * self.QSCA
        self.ipKABS = interp1d(self.QFREQ, self.KABS)
        self.ipKSCA = interp1d(self.QFREQ, self.KSCA)
        self.ipGSCA = interp1d(self.QFREQ, self.G)
    
    def Ksca(self, freq):
        print("EffDustO::Ksca")
        return clip(self.ipKSCA(freq), 1.0e-99, 1.0e10)
    
    def Kabs(self, freq):
        return self.ipKABS(freq)

    def Gsca(self, freq):
        return self.ipGSCA(freq)

    
    
    
    
    
    

    
    
class DustemDustO(DustO):
        
    def __init__(self, filename, force_nsize=-1):
        """
        Input:
            filename  =  the name of dust file used by CRT/SOC
            including for example:
                optical         /home/mika/tt/dustem4.0_web/oprop/Q_Gra.DAT
                phase_function  /home/mika/tt/dustem4.0_web/oprop/G_Gra.DAT
                sizes           /home/mika/tt/dustem4.0_web/data/GRAIN_DL07.DAT
        """
        self.EFF           = False
        self.DUSTEM        = True
        # else -- read optical properties and size distributions from different files
        self.GRAIN_DENSITY = 1.0e-7  # DUMMY !!
        self.GRAIN_SIZE    = 1.0e-4  # DUMMY !!
        self.DUSTNAME      = ""      # name of a single dust component
        self.NSIZE         = 0       # size bins in calculation (in size distribution, in CRT_SFRAC)
        self.QNFREQ        = 0       # size bins in Q files (preferably NSIZE==QNSIZE)
        self.QSIZE         = None
        self.NQSIZE        = 0
        
        # Read CRT/SOC file and find out the names of the files
        # The file should contain a line "sizes filename" that specifies the name
        #   of the DustEM file with the definition of the size distribution
        file_optical = ""
        file_phase   = ""
        file_sizes   = ""
        file_lambda  = DUSTEM_DIR+'/oprop/LAMBDA.DAT'
        for line in open(filename).readlines():
            s = line.split()
            if (len(s)<1): continue
            if (s[0]=='optical'):
                file_optical = s[1]
            if (s[0]=='phase_function'):
                file_phase   = s[1]
            if (s[0]=='sizes'):
                file_sizes   = s[1]
            if (s[0]=='lambda'):
                file_lambda  = s[1]
            if (s[0]=='prefix'):
                self.DUSTNAME = s[1]
        if (file_optical==""):
            print("DustemDust - file for optical properties not defined")
            sys.exit()            
        if (file_phase==""):
            print("DustemDust - file for phase functions not defined")
            sys.exit()
        if (file_lambda==""):
            print("DustemDust - file for wavelength grid not defined")
            sys.exit()
        if (file_sizes==""):
            print("DustemDust - file for grain sizes not defined")
            sys.exit()
        if (self.DUSTNAME==""):
            print("DustemDust - missing prefix == name of the dust component")
            sys.exit()

            
        # Read the sizes
        print("Reading: %s" % file_sizes)
        lines = open(file_sizes).readlines()
        dusts_found  = 0
        # arrays to store parameters of the size distribution of the current species
        #   (SIZE_A, SIZE_F) determine size distribution --- may be different from QSIZE !!
        #   when using CRT_SFRAC, this has as many elements as given in DustEM input file
        #   == self.NSIZE
        self.SIZE_A,   self.SIZE_F = None, None  # these define the size distribution
        self.CRT_SFRAC = None                    # will have self.NSIZE elements
        for i in range(len(lines)):
            s          = lines[i].split()
            if (len(s)<6): continue
            if (s[0]!=self.DUSTNAME): continue # DustO == a single dust species !!
            # We found this dust !!
            s          = lines[i].split()
            nsize      =   int(s[1])   # number of log bins in DustEM
            typ        =       s[2]    # mix-logn, logn, plaw-ed-cv etc.
            rmass      = float(s[3])   # Mdust/MH
            rho        = float(s[4])   # g/cm3
            self.RHO   = rho
            self.AMIN  = float(s[5])   #  [cm]
            self.AMAX  = float(s[6])   #  [cm]
            if (METHOD=='CRT'):
                self.NSIZE = nsize     # DustEM size bins in definition != in Q files !!!
            else:
                if (force_nsize>0):
                    self.NSIZE = force_nsize
                else:
                    self.NSIZE = 500       # finer grid, more interpolation
            
            # s =  grain type, nsize, type keywords, Mdust/MH, rho, amin, amax, <parameters>
            #      0           1      2              3         4    5     6     7...
            # types
            #   logn   dn/dloga = exp(-(log(a/a0)/sigma)^2)           :: a0, sigma
            #   plaw   dn/da    = a^alpha                             :: alpha
            #   ed     pl *=  1, for a<at,   exp(-((a-at)/ac)^gamma)  :: alpha, at, ac, gamma 
            #   cv     pl *=  [ 1 + |z| (a/au)^eta ]^(sign(z))        :: alpha, au, z, eta
            #                                                         :: alp...gamma, au, z, eta
            # Create big enough arrays self.SIZE_A and self.SIZE_F so that these can be interpolated
            if (typ=='size'): # read file data/SIZE_tt.DAT
                lines = open(DUSTEM_DIR+'/data/SIZE_%s.DAT' % self.DUSTNAME).readlines()
                # 1. line ==  shape, eps
                # 2. line == number of bulk materials
                # 3. line == name of bulks
                # 4. line == bulk densities [g/cm3]
                # 5. line == mass fraction of each bulk
                # 6. line == number of size bins
                #    rest ==  a[cm], dlog(a), a**4dn/da, rho_eff, volume fraction
                # Find the first line with those five columns
                for i in range(len(lines)):
                    if (lines[i][0:1]=='#'): continue
                    if (len(lines[i].split())<5): continue
                # read the rest of the file
                d = loadtxt(DUSTEM_DIR+'/data/SIZE_%s.DAT' % self.DUSTNAME, skiprows=i-1)
                self.SIZE_A = d[:,0].copy()                        #  a [cm]
                self.SIZE_F = d[:,3].copy() / (self.SIZE_A)**4.0   # dn/da [1/cm2/H]
            else:
                # analytical formulas => discretised representation that can be interpolated
                # we use our own size grid with many bins...
                # NOT IF ONE IS USING CRT_SFRAC => must be equal to number of bins in DustEM files
                self.SIZE_A = asarray(logspace(log10(self.AMIN), log10(self.AMAX), self.NSIZE), float64)
                if (0):
                    # size grid ***is*** identical to CRT
                    for i in range(len(self.SIZE_A)):
                        print('%12.4e' % self.SIZE_A[i])
                    sys.exit()
                # the first should generate the distribution, others can modify it
                i = 7   # next parameter 
                for ss in typ.split('-'):   # type can be complex such as "plaw-ed-cv"
                    # plaw and logn create a distribution, ed and cv modify existing one
                    if   (ss=='logn'):
                        # dn/dloga ~  exp(-( log(a/a0)/sigma )**2)
                        a0, sigma, i = float(s[i]), float(s[i+1]), i+2
                        print('*** logn ***  a0 %10.3e, sigma %10.3e' % (a0, sigma))
                        # factor 0.5 was missing from the documentation ??
                        self.SIZE_F  = exp(- 0.5*(log(self.SIZE_A/a0)/sigma)**2.0 )
                        # convert from dn/dloga to dn/da
                        #     *=  dloga/da = 1/a
                        self.SIZE_F /= self.SIZE_A
                    elif (ss=='plaw'):
                        # dn/da ~ a^alpha
                        print('*** plaw ***')
                        alpha, i     =  float(s[i]), i+1
                        self.SIZE_F  =  self.SIZE_A**alpha
                    elif (ss=='ed'):
                        #  *= exp(-( (a-at)/ac )**gamma), for a>at
                        a_t, a_c, gamma, i  =  float(s[i]), float(s[i+1]), float(s[i+2]), i+3
                        print('*** ed   *** at %10.3e ac %10.3e gamma %10.3e' % (a_t, a_c, gamma))
                        m                =  nonzero(self.SIZE_A>a_t)
                        self.SIZE_F[m]  *=  exp(-((self.SIZE_A[m]-a_t)/a_c)**gamma)
                    elif (ss=='cv'):
                        #  *=  (1+|z|*(a/au)**eta)**sign(z)
                        print('*** cv   ***')
                        a_u, z, eta, i = float(s[i]), float(s[i+1]), float(s[i+2]), i+3
                        self.SIZE_F  *=  (1.0 + abs(z)*(self.SIZE_A/a_u)**eta)**(sign(z))
                #

                
                # we need normalisation with respect to H
                #   volume integral * rho  =  Hydrogen mass * rmass
                # integal of mass_integrand over [amin, amax] == 1.00794*AMU*rmass
                #        => self.SIZE_F  becomes  1/H/cm
                mass_integrand = asarray(self.SIZE_F * (4.0*pi/3.0)*(self.SIZE_A**3.0) * rho, float64)
                ip             =  interp1d(self.SIZE_A, mass_integrand)
                res, dres      =  quad(ip, self.AMIN, self.AMAX, epsrel=1.0e-10, epsabs=EPSABS2)  # total dust mass...
                ##
                self.SIZE_F   *=  mH*rmass / res        # ... should equal AMU*rmass
                                
                    
                # CRT-type summation instead of integrals over size distribution
                # must use the logarithmic size grid specified in the DustEM file
                self.CRT_SFRAC   =  self.SIZE_F  * self.SIZE_A
                vol              =  (4.0*pi/3.0) * sum( self.CRT_SFRAC * self.SIZE_A**3.0 )
                print('rmass %10.3e, rho %10.3e, vol %10.3e' % (rmass, rho, vol))
                # grain_density  =  mH*ratio / (vol*rho)
                self.CRT_SFRAC  *=  mH*rmass / (rho*vol)  # grain density included in self.SFRAC
                # now sum(SFRAC*pi*a**2*Q) should be tau/H !!
                if (1):
                    vol = sum(self.CRT_SFRAC * (4.0*pi/3.0) * self.SIZE_A**3.0)
                    print('Dust mass %10.3e, H mass %10.3e' % (vol*rho, mH))
            
                    
                    
                # apply mix, if given --- AFTER the initial normalisation
                mix = None
                if (typ.find('mix')>=0):
                    print('*** MIX ***')
                    # number of grains at each size multiplied with a factor from /data/MIX_tt.dat
                    # this is specified for points logspace(amin, amax, nsize)
                    mix = ravel(loadtxt(DUSTEM_DIR+'/data/MIX_%s.DAT' % self.DUSTNAME))
                    print('MIX ', mix)
                    if (nsize==self.NSIZE): # using original dustem grid !!
                        if (len(mix)!=len(self.SIZE_F)):
                            print('??????????')
                            sys.exit()
                        ### self.SIZE_F *= mix
                    else:                   # need to interpolate !!
                        print('*** Interpolating MIX !!! nsize %d, NSIZE %d' % (nsize, self.NSIZE))
                        x = logspace(log10(self.AMIN), log10(self.AMAX), nsize)  # the DustEM size grid !!
                        if (len(x)!=len(mix)):
                            print('DustLib, read MIX with %d size points != %d' % (self.NSIZElen(mix), mix))
                        # apply these factors to our (probably more dense) size grid
                        if (0):
                            ip = interp1d(x, mix, fill_value=1.0, bounds_error=False)
                            mix = ip(self.SIZE_A)
                        else:
                            ip  = interp1d(log(x), mix, fill_value=1.0, bounds_error=False)
                            mix = ip(log(self.SIZE_A))
                        print('AVERAGE SCALING %.2f' % mean(ip(self.SIZE_A)))
                        print('MIX defined %10.3e - %10.3e' % (self.AMIN, self.AMAX))
                        print('SIZE_A      %10.3e - %10.3e' % (self.SIZE_A[0], self.SIZE_A[-1]))
                    ####
                    self.SIZE_F    *= mix
                    self.CRT_SFRAC *= mix
                    
                
            ## 
            break  # A SINGLE DUST COMPONENT !!
        
        ## put the total into self.SIZE_A and self.SIZE_F
        ## interpolated function ipFRAC = dn/da/H
        print('SIZE_A', self.SIZE_A.shape, '   SIZE_F', self.SIZE_F.shape)
        self.ipFRAC    = interp1d(self.SIZE_A, self.SIZE_F, bounds_error=True, fill_value=0.0)
        self.ipFRAC_LL = interp1d(log(self.SIZE_A), log(self.SIZE_F), bounds_error=True, fill_value=0.0)
        print('SIZE DEFINED FOR %10.3e %10.3e' % (self.AMIN, self.AMAX))
        print('SIZE DEFINED FOR %10.3e %10.3e' % (min(self.SIZE_A), max(self.SIZE_A)))
                    
        # Read the optical data
        #  dustem directory /oprop/G_<dustname>.DAT
        # (1) read Qabs and Qsca
        #       self.QNSIZE    =  number of sizes for which optical data specified
        #       self.QNFREQ    =  number of frequencies
        #       self.QSIZE     =  the sizes               [um] -> [cm]
        #       self.QFREQ     =  the frequencies IN DECREASING ORDER
        #       self.OPT[QNSIZE, NFREQ, 4] = um, Qabs, Qsca, g
        # Read the common wavelength grid
        DUM          = loadtxt(file_lambda, skiprows=4)   # [um]
        self.QFREQ   = um2f(DUM)          # [Hz], decreasing order of frequency!
        self.QNFREQ  = len(self.QFREQ)    # from DustEM LAMBDA file !!
        # Read the Q data
        lines = open(DUSTEM_DIR+'/oprop/Q_%s.DAT' % self.DUSTNAME).readlines()
        for iline in range(20):
            if (lines[iline][0:1]=='#'): continue
            if (len(lines[iline])<2): continue
            break
        print('LINE ', iline)  # 4
        self.QNSIZE  = int(lines[iline].split()[0]) # 50 sizes?
        self.QSIZE   = zeros(self.QNSIZE, float64)
        s            = lines[iline+1].split()   # line 600 characters
        if (len(s)!=self.QNSIZE):
            print('[1] ??????????????????')
        for i in range(self.QNSIZE):
            self.QSIZE[i] = float(s[i]) # this size grid not necessarily logarithmic?
        self.QSIZE *= 1.0e-4   # file has [um], we need [cm]
        # read Qabs and Qsca data (increasing order of wavelength)
        x      =  loadtxt(DUSTEM_DIR+'/oprop/Q_%s.DAT' % self.DUSTNAME, skiprows=iline+3) # 7
        qabs   =  x[0:self.QNFREQ,:]   # rows=wavelenghts, columns = sizes
        qsca   =  x[self.QNFREQ:, :]
        if ((qabs.shape[0]!=self.QNFREQ)|(qsca.shape[0]!=self.QNFREQ) | (qabs.shape[1]!=self.QNSIZE)):
            print('????!!!!!')
            sys.exit()
        # read the g parameters -- assume the grid is the same as for Q
        gHG    =  loadtxt(DUSTEM_DIR+'/oprop/G_%s.DAT' % self.DUSTNAME, skiprows=9)
        if ((gHG.shape[0]!=qabs.shape[0])|(gHG.shape[1]!=qabs.shape[1])):
            print('DustLib: mismatch between Q and g files')
            sys.exit()
        # Put data into self.OPT, still in order of increasing wavelength
        # OPT[size, freq, 4] = [ um, Kabs, Ksca, g ], including multiplication by grain area
        # TOBE CONSISTENT WITH CRT AND DUSTEM, Q INTERPOLATED OVER SIZES BEFORE *= pi*a^2
        self.OPT  = zeros((self.QNSIZE, self.QNFREQ, 4), float64)
        for isize in range(self.QNSIZE):
            ### A  = pi*(self.QSIZE[isize])**2.0     # grain area [cm2]
            for ium in range(self.QNFREQ):
                self.OPT[isize, ium, :] = asarray(\
                [ DUM[ium], qabs[ium, isize], qsca[ium, isize], gHG[ium, isize] ], float64)
                
        self.OPT = asarray(self.OPT, float64)
        #
        self.QFREQ = asarray(um2f(DUM), float64)  # LAMBDA file => DECREASING order of frequency!
        #
        print('  --- SIZE DISTRIBUTION SIZE_A  %12.4e - %12.4e' % (self.SIZE_A[0], self.SIZE_A[-1]))
        print('  --- OPTICAL DATA      SIZE    %12.4e - %12.4e' % (self.QSIZE[0], self.QSIZE[-1]))
        if (self.AMIN<self.QSIZE[0]):
            print('*** WARNING **** AMIN %12.4e < OPTICAL DATA AMIN %12.4e' % (self.AMIN, self.QSIZE[0]))
            scale            = (self.AMIN/self.SIZE[0])**2.0
            self.OPT[0,:,1] *= scale       # Kabs
            self.OPT[0,:,2] *= scale       # Ksca
            self.SIZE[0]     = self.AMIN   # optical data "extrapolated" to self.AMIN
        if (self.AMAX>self.QSIZE[-1]):
            print('*** ERROR *** AMAX %12.4e > OPTICAL DATA AMAX %12.4e' % (self.AMAX, self.QSIZE[-1]))
            sys.exit()            
            
        print('AMIN - AMAX      %10.3e  %10.3e' % (self.AMIN, self.AMAX))
        print('self.SIZE_A      %10.3e  %10.3e' % (min(self.SIZE_A), max(self.SIZE_A)))
        print('Q defined        %10.3e  %10.3e' % (min(self.QSIZE), max(self.QSIZE)))
        
        
        # Read the file of heat capacities
        lines = open(DUSTEM_DIR+'/hcap/C_%s.DAT' % self.DUSTNAME).readlines()
        for i in range(len(lines)):
            if (lines[i][0:1]=='#'): 
                continue
            break
        # this line has the number of sizes
        self.CNSIZE = int(lines[i].split()[0])
        i += 1
        # this line has the sizes... should be the same as self.QSIZE ??
        self.CSIZE = zeros(self.CNSIZE, float32)
        s = lines[i].split()
        for isize in range(self.CNSIZE):
            self.CSIZE[isize] = float(s[isize]) * 1.0e-4   # file [um], CSIZE [cm]
        i += 1
        # this line has the number of T values
        self.CNT  = int(lines[i].split()[0])
        i += 1
        # the rest, first column = log(T), other columns = C [erg/K/cm3] for each size
        #   self.ClgC[iT, iSize]
        d         = loadtxt(DUSTEM_DIR+'/hcap/C_%s.DAT' % self.DUSTNAME, skiprows=i)
        self.ClgT = d[:,0]
        self.ClgC = d[:,1:]   #  [ index_T,  index_size ]
        if ((self.ClgC.shape[0]!=self.CNT)|(self.ClgC.shape[1]!=self.CNSIZE)):
            print('Error reading enthalpy file !!')
            sys.exit()
        ##
        self.CT = 10.0**self.ClgT   # [K]
        self.CC = 10.0**self.ClgC   # [erg/K/cm3]
        ## END OF __init__    


        
        
            
    
def combined_scattering_function(DUST, um, theta, SIN_WEIGHT, size_sub_bins=500):
    """
    We have several dust components. Calculate combined, discretised scattering function.
    Input:
        DUST  = list of DustO
        um    = selected single wavelength
        theta = array of theta values [0.0, pi]
    Note:
        For packet generation, SIN_WEIGHT must be true
        For packet weighting, SIN_WEIGHT must be false
    Note:
        For the sum of Henyey-Greenstein functions (no sin(theta)),
        the value of <cos(theta)> is the same as the weighted average of g values
        -- this is true for combined_scattering_function(DUST, um, theta, False)!
        500 bins should be enough for accuracy ~1e-4
    """
    n   = len(DUST)
    f   = um2f(um)
    sca = zeros(n, float64)
    res = zeros(len(theta), float64)
    W   = 0.0
    for i in range(n):
        w     =  DUST[i].Ksca(f)   # Ksca = weight factor
        W    +=  w
        res  +=  w * DUST[i].DSF(f, theta, SIN_WEIGHT, size_sub_bins)
    res /= W
    # must have proper normalisation (not only ratios)
    return res  



def combined_scattering_function2(DUST, um, cos_theta, SIN_WEIGHT, size_sub_bins=500):
    n   = len(DUST)
    f   = um2f(um)
    sca = zeros(n, float64)
    res = zeros(len(cos_theta), float64)
    W   = 0.0
    for i in range(n):
        w     =  DUST[i].Ksca(f)   # Ksca = weight factor
        W    +=  w
        res  +=  w * DUST[i].DSF2(f, cos_theta, SIN_WEIGHT, size_sub_bins)
    res /= W
    # must have proper normalisation (not only ratios)
    return res  


def SFlookup(DUST, um, BINS, SIN_WEIGHT, size_sub_bins=500):
    """
    Make a look-up table with N elements, u*N maps to theta distribution
    Note:
        For packet generation, SIN_WEIGHT must be true
        For packet weighting, SIN_WEIGHT must be false
    Note:
        BINS should be ~5000 !
        kind must be 'linear'
    """
    theta =  linspace(0, pi, BINS)    # ... was 180 
    Y     =  combined_scattering_function(DUST, um, theta, SIN_WEIGHT, size_sub_bins)
    Y     =  cumsum(Y) + 1e-7*cumsum(ones(len(Y), float64))
    Y    -=  Y[0]
    Y    /=  Y[-1]                 # [0,1]
    Y[0]  = -1.0e-7
    Y[-1] =  1.0+1.0e-7
    ip    =  interp1d(Y, theta, kind='linear')    # Prob -> theta
    res   =  ip(linspace(0.0, 1.0, len(theta)))   # theta values for equidistant prob
    return res  # theta values


def SFlookupCT(DUST, um, BINS, SIN_WEIGHT, size_sub_bins=500):
    """
    Make a look-up table with N elements, u*N maps to theta distribution
    Note:
        For packet generation, SIN_WEIGHT must be true
        For packet weighting, SIN_WEIGHT must be false
    Note:
        return cos(theta)
    """
    # internally better discretisation over theta 
    # theta =  linspace(0, pi, 4*BINS)
    theta =  linspace(0, pi, 5*BINS)
    Y     =  combined_scattering_function(DUST, um, theta, SIN_WEIGHT, size_sub_bins)
    P     =  cumsum(Y) + 1e-7*cumsum(ones(len(Y), float64))
    P    -=  P[0]
    P    /=  P[-1]                 # [0,1]
    P[0]  = -1.0e-7
    P[-1] =  1.0+1.0e-7
    ip    =  interp1d(P, cos(theta), kind='linear')    # mapping [0,1] -> [-1.0, +1.0] of cos(theta)
    res   =  ip(linspace(0.0, 1.0, BINS))              # shorter arrays = BINS elements
    return res




def combined_scattering_function2_simple(DUST, um, cos_theta, SIN_WEIGHT):
    # @@@
    n   = len(DUST)
    f   = um2f(um)
    sca = zeros(n, float64)
    res = zeros(len(cos_theta), float64)
    W   = 0.0
    for i in range(n):
        w     =  DUST[i].Ksca(f)   # Ksca = weight factor
        W    +=  w
        res  +=  w * DUST[i].DSF2_simple(f, cos_theta, SIN_WEIGHT)
    res /= W
    # must have proper normalisation (not only ratios)
    return res  


    
def combined_scattering_function_simple(DUST, um, theta, SIN_WEIGHT, size_sub_bins=500):
    # @@@
    n   = len(DUST)
    f   = um2f(um)
    sca = zeros(n, float64)
    res = zeros(len(theta), float64)
    W   = 0.0
    for i in range(n):
        print("A")
        w     =  DUST[i].Ksca(f)   # Ksca = weight factor
        print("um = %7.3f     dust %d   w =  %.3e" % (um, i, w))
        W    +=  w
        res  +=  w * DUST[i].DSF_simple(f, theta, SIN_WEIGHT)
    res /= W
    # must have proper normalisation (not only ratios)
    return res  

    

def SFlookup_simple(DUST, um, BINS, SIN_WEIGHT):
    """
    Make a look-up table with N elements, u*N maps to theta distribution
    Note:
        For packet generation, SIN_WEIGHT must be true
        For packet weighting, SIN_WEIGHT must be false
    Note:
        BINS should be ~5000 !
        kind must be 'linear'
    """
    theta =  linspace(0, pi, BINS)    # ... was 180 
    Y     =  combined_scattering_function_simple(DUST, um, theta, SIN_WEIGHT)
    Y     =  cumsum(Y) + 1e-7*cumsum(ones(len(Y), float64))
    Y    -=  Y[0]
    Y    /=  Y[-1]                 # [0,1]
    Y[0]  = -1.0e-7
    Y[-1] =  1.0+1.0e-7
    ip    =  interp1d(Y, theta, kind='linear')    # Prob -> theta
    res   =  ip(linspace(0.0, 1.0, len(theta)))   # theta values for equidistant prob
    return res  # theta values


def SFlookupCT_simple(DUST, um, BINS, SIN_WEIGHT):
    """
    Make a look-up table with N elements, u*N maps to theta distribution
    Note:
        For packet generation, SIN_WEIGHT must be true
        For packet weighting, SIN_WEIGHT must be false
    Note:
        return cos(theta)
    """
    theta =  linspace(0, pi, 5*BINS)
    Y     =  combined_scattering_function_simple(DUST, um, theta, SIN_WEIGHT)
    P     =  cumsum(Y) + 1e-7*cumsum(ones(len(Y), float64))
    P    -=  P[0]
    P    /=  P[-1]                 # [0,1]
    P[0]  = -1.0e-7
    P[-1] =  1.0+1.0e-7
    ip    =  interp1d(P, cos(theta), kind='linear')    # mapping [0,1] -> [-1.0, +1.0] of cos(theta)
    res   =  ip(linspace(0.0, 1.0, BINS))              # shorter arrays = BINS elements
    return res




class DustsO:
    # if DUSTS has only one element, assume it is of simple dust type
    # if there are multiple, they are assumed to be of type TRUST (with size distributions etc.)
    def __init__(self, dusts):
        self.DUSTS = []
        for dust in dusts:
            self.DUSTS.append(dust)
        self.N = len(self.DUSTS)
        if (self.N==1):
            self.GRAIN_DENSITY = self.DUSTS[0].GRAIN_DENSITY
            self.GRAIN_SIZE    = self.DUSTS[0].GRAIN_SIZE
        else:
            # these are dummy values --- GRAIN_DENSITY*pi*GRAIN_SIZE**2 == 1.0
            self.GRAIN_DENSITY = 1.0/pi
            self.GRAIN_SIZE    = 1.0
        
    def Kabs(self, freq):
        res = 0.0
        if (isscalar(freq)):
            for dust in self.DUSTS:
                res += dust.Kabs(freq)
        else:
            res = zeros(len(freq), float64)
            for i in range(len(freq)):
                for j in range(len(self.DUSTS)):
                    res[i] += self.DUSTS[j].Kabs(freq[i])                    
        return res

    def Ksca(self, freq):
        print("DustsO::Ksca")
        res = 0.0
        if (isscalar(freq)):
            for dust in self.DUSTS:
                res += dust.Ksca(freq)
        else:
            res = zeros(len(freq), float64)
            for i in range(len(freq)):
                for j in range(len(self.DUSTS)):
                    res[i] += self.DUSTS[j].Ksca(freq[i])                    
        return res
    
    def Gsca(self, freq):
        res = 0.0
        # this is approximate - usually one will use full scattering function...
        if (isscalar(freq)):
            w, res = 0.0, 0.0
            for i in range(self.N):
                x    = self.DUSTS[i].Ksca(freq)
                w   += x
                res += x * self.DUSTS[i].Gsca(freq)
            res /= w
        else:
            res = zeros(len(freq), float64)
            for j in range(len(freq)):
                w, s = 0.0, 0.0
                for i in range(self.N):
                    x    = self.DUSTS[i].Ksca(freq[j])
                    w   += x
                    s   += x * self.DUSTS[i].Gsca(freq[j])
                res[j] = s/w
        return res
    
    


def write_simple_dust(DUST, FREQ, BINS=2500, filename='tmp.dust', dscfile='tmp.dsc', force_isotropic=False):
    """
    Write dust description for CRT and SOC, based on DustEM but
    not assuming that DustEM will be used to calculate emission.
    This is a simple dust definition [freq, Qabs, Qsca, g] that is sufficient 
    for the radiative transfer part of the calculation.
    Input:
        DUST      = array of DustO objects
        FREQ      = list of frequencies (many for CRT, which does interpolation;
                    should be less for SOC, which directly uses the listed frequencies)
        BINS      = number of angle bins in the calculation of the 
                    discretised scattering functions
        filename  = name of the written dust file
        dscfile   = name of the file written for discretised scattering functions 
    Output:
        tmp.dust  = simple dust definition sufficient for radiative 
                    transport part of the calculation with CRT or with SOC
        tmp.dsc   = scattering functions in the format expected by CRT and SOC
    """
    if (force_isotropic):
        for i in range(len(DUST)):
            DUST[i].OPT[:,:,3] = 0.01
    NFREQ     = len(FREQ)
    NDUST     = len(DUST)
    ABS, SCA, GSUM = zeros(NFREQ, float64), zeros(NFREQ, float64), zeros(NFREQ, float64)
    for i in range(NDUST):
        Abs   =  DUST[i].Kabs(FREQ)
        Sca   =  DUST[i].Ksca(FREQ)
        g     =  DUST[i].Gsca(FREQ)
        ABS   += Abs
        SCA   += Sca
        GSUM  += Sca*g
    G = GSUM / (SCA+1.0e-40)   # this will not be used if one uses discretised scattering functions
    # write tmp.dust = simple dust description
    GRAIN_DENSITY = 1.0e-7  # ad hoc
    GRAIN_SIZE    = 1.0e-4  # ad hoc
    K     =  GRAIN_DENSITY * pi*GRAIN_SIZE**2.0  # CRT will multipy Q factors using these
    fp = open(filename, 'w')
    fp.write('eqdust\n')
    fp.write('%12.5e\n' % GRAIN_DENSITY)
    fp.write('%12.5e\n' % GRAIN_SIZE)
    fp.write('%d\n' % NFREQ)
    for i in range(NFREQ):
        fp.write('%12.5e  %8.5f  %12.5e %12.5e\n' % (FREQ[i], G[i], ABS[i]/K, SCA[i]/K))
    fp.close()
    if (dscfile!=''):
        # write the scattering function
        cos_theta = linspace(-1.0, 1.0, BINS)
        fs  = open(dscfile, 'w')
        for i in range(NFREQ):  # all wavelengths!
            X   =  combined_scattering_function2_simple(DUST, f2um(FREQ[i]), cos_theta, SIN_WEIGHT=False)
            X   =  clip(X, 1e-5*max(X), 1e20) # should not have zeros...
            asarray(X, float32).tofile(fs)
        for i in range(NFREQ):
            X   =  SFlookupCT_simple(DUST, f2um(FREQ[i]), BINS=BINS, SIN_WEIGHT=True)
            asarray(X, float32).tofile(fs)
        fs.close()
          


    
def write_DUSTEM_files(dustem_file=''):
    """
    Write dust files needed by the CRT + DustEM runs.
    If necessary, rename dusts so that each size distribution is associated to a 
    unique dust name. The corresponding files (symbolic links) are created
    in the DustEM directories.
    Input:
        Name of the DustEM GRAIN.DAT type file
    Output:
        writes crt_dusts.txt = lines to be added to CRT ini, lines [ dust < dustname> ]
        makes symbolic links in DustEM directories for the renamed dust components
        writes CRT dust files <dustname>.dust for each component
        return list of dust components, after renaming
    """
    ORINAME = []  # dust names from GRAIN.DAT
    NEWNAME = []  # the same, if dust on several GRAIN.DAT lines, rename <dust>_copy1... etc.
    fp  = open('crt_dusts.txt', 'w')                         # list here dust names used by CRT
    fpD = open('%s/data/GRAIN_TMP.DAT' % (DUSTEM_DIR), 'w')  # open the DustEM file with size distributions
    for line in open(DUSTEM_DIR+'/data/'+dustem_file):       # extract dust names from DustEM file
        s = line.split()
        if (len(s)<1): continue
        if (line[0:1]=='#'): 
            fpD.write(line)
            continue
        if (len(s)<8):  # other line than a dust component
            fpD.write(line)
            continue
        orig = s[0]     # original dust name in GRAIN.DAT
        for i in range(10):  # make sure repeated entries get unique names
            if (i==0): dust = '%s' % orig
            else:      dust = '%s_copy%d' % (orig, i)      # rename so that same name appears only once
            if (not(dust in ORINAME)): break    # ok, first line with this dust name
        ORINAME.append(orig)
        NEWNAME.append(dust)
        # write out the CRT file for this dust component
        fpc = open('dustem_%s.dust' % dust, 'w')
        fpc.write('dustem\n')
        fpc.write('# dust from %s\n' % dustem_file)
        fpc.write('prefix          %s\n' % dust)   # dust name, possibly renamed
        fpc.write('optical         %s/oprop/Q_%s.DAT\n' % (DUSTEM_DIR, dust))
        fpc.write('phase_function  %s/oprop/G_%s.DAT\n' % (DUSTEM_DIR, dust))
        fpc.write('sizes           %s/data/GRAIN_TMP.DAT\n' % (DUSTEM_DIR))  # this is now being written
        fpc.write('lambda          %s/oprop/LAMBDA.DAT\n' % (DUSTEM_DIR))
        fpc.write('heat            %s/hcap/C_%s.DAT\n'  % (DUSTEM_DIR, dust))
        fpc.close()
        fp.write('dust   dustem_%s.dust\n' % dust)     # these should be included in CRT ini
        # copy the line from GRAIN.DAT
        fpD.write(line.replace(orig, dust))
        # make sure Q, G, and C files (symbolic links) exist for the renamed dusts
        if (orig!=dust):
            os.system('ln -s %s/oprop/Q_%s.DAT %s/oprop/Q_%s.DAT' % (DUSTEM_DIR, orig, DUSTEM_DIR, dust))
            os.system('ln -s %s/oprop/G_%s.DAT %s/oprop/G_%s.DAT' % (DUSTEM_DIR, orig, DUSTEM_DIR, dust))
            os.system('ln -s %s/hcap/C_%s.DAT %s/hcap/C_%s.DAT'   % (DUSTEM_DIR, orig, DUSTEM_DIR, dust))
    fp.close()    
    fpD.close()    
    return NEWNAME

    


def write_A2E_dustfiles(DNAME, DUST, NE_ARRAY=[]):
    """
    Write the dust files for A2E  *** NATIVE CRT FORMAT ***
    Input:
        DNAME = list of dust names (e.g. PAH0_MC10, aSilx)
        DUST  = list of DustemDustO objects
        NE    = list of number of energy bins, each dust
    Output:
        dust files sg<dustname>.dust
        and the corresponding files
        *.opt  = optical properties
        *.ent  = specific heats
        *.size = size distribution
    """
    NDUST      = len(DUST)
    if (len(NE_ARRAY)<1): 
        NE_ARRAY = 512*ones(NDUST, int32)
    ##
    for idust in range(NDUST):
        dust  =  DUST[idust]
        NE    =  NE_ARRAY[idust]
        name  =  DNAME[idust]   # PAH0_MC10 etc.
        fp = open('gs_%s.dust' % name, 'w')
        fp.write('gsetdust\n')
        fp.write('prefix     %s\n' % name)
        #fp.write('eqlimit    0.0025\n')
        #fp.write('binnings   1e-3 1e-1 1e0 1e1 1e2 1e4\n')
        fp.write('nstoch     -1\n')
        # fp.write('nstoch     999\n')
        fp.write('optical    gs_%s.opt\n' % name)
        fp.write('enthalpies gs_%s.ent\n' % name)
        fp.write('sizes      gs_%s.size\n' % name)
        fp.close()
        
        
        # Copy the data on optical properties, original size and frequency grid
        #   === CRT format ===
        #     NSIZE, NFREQ
        #     {  size [um]
        #          { freq, Qabs, Qsca, g }
        #     }
        fp = open('gs_%s.opt' % name, 'w')
        fp.write('%d %d  # NSIZE, NFREQ\n' % (dust.QNSIZE, dust.QNFREQ))
        for isize in range(dust.QNSIZE):
            fp.write('%12.5e   # SIZE [um]\n' % (1e4*dust.QSIZE[isize]))
            fp.write('# FREQ      Qabs        Qsca        g\n')
            a  = dust.QSIZE[isize]
            for ifreq in range(dust.QNFREQ):   # INCREASING order of FREQUENCY
                ii    = dust.QNFREQ-1-ifreq
                freq  = um2f(dust.OPT[isize, ii, 0])
                qabs  =      dust.OPT[isize, ii, 1]
                qsca  =      dust.OPT[isize, ii, 2]
                g     =      dust.OPT[isize, ii, 3]
                fp.write('%12.5e %12.5e %12.5e %12.5e\n' %  (freq, qabs, qsca, g))
        fp.close()
        
        
        # Copy the data on grain sizes
        #      GRAIN_DENSITY   = total number of grains per H
        #      NSIZE    NE
        #      {   size[um]   S_FRAC    Tmin    Tmax   }
        # CRT_SFRAC = dn/da*da => sum(CRT_FRAC) is GRAIN_DENSITY
        # NE  =  256
        # NE  =  512
        # NE  =  2048
        fp  =  open('gs_%s.size' % name, 'w')
        fp.write('%12.5e   # GRAIN_DENSITY\n' % sum(dust.CRT_SFRAC))
        fp.write('%d %d    # NSIZE NE\n' % (dust.NSIZE, NE))    #  NE not used ????
        
        tmp  =  dust.CRT_SFRAC * 1.0
        tmp  =  tmp / sum(tmp)

        # Tmax =  logspace(log10(2500.0), log10(150.0), dust.NSIZE)
        # Tmax =  logspace(log10(2600.0), log10(200.0), dust.NSIZE)
        # Tmax =  logspace(log10(3000.0), log10(200.0), dust.NSIZE)
        Tmax =  logspace(log10(2500.0), log10(150.0), dust.NSIZE)

        fp.write('#  SIZE [um]    S_FRAC      Tmin [K]   Tmax [K]\n')
        for isize in range(dust.NSIZE):
            #  Tmin and Tmax not used by A2E ???
            fp.write('  %12.5e %12.5e  %10.3e %10.3e\n' % 
            (1.0e4*dust.SIZE_A[isize], tmp[isize], 4.0, Tmax[isize]))
        fp.close()

        
        # Copy the data on enthalpies,  dust.CC = erg/K/cm3
        fp   = open('gs_%s.ent' % name, 'w')
        fp.write('# NUMBER OF SIZE\n')
        fp.write('#    { SIZES [um] }\n')
        fp.write('# NUMBER OF TEMPERATURES\n')
        fp.write('#    { T }\n')
        fp.write('# E[SIZES, NT]  #  each row = one size, each column = one temperature !!\n')
        fp.write('#\n')
        fp.write('%d   #  NSIZE\n' % (dust.CNSIZE))
        for isize in range(dust.CNSIZE):
            fp.write('   %12.5e\n' % (1.0e4*dust.CSIZE[isize])) # write in [um]
        fp.write('%d   #  NTEMP\n' % (dust.CNT))
        for iT    in range(dust.CNT):
            fp.write('   %12.5e\n' % (dust.CT[iT]))
        # file should contain enthalpy,   C(grain) = (4*pi/3)*a^3*rho*(C/g)
        # DustEM files contain C [erg/K/cm3] => multiply by volume and integrate to T
        #    dust.CC[iT, iSize]  =   erg/K/cm3 
        for isize in range(dust.CNSIZE): # one row per size
            # A2E needs E(T), not C(T) !!
            x  = concatenate(([0.0,], dust.CT))
            #  C(grain) = dust.CC * Vol
            y  = (4.0*pi/3.0)*(dust.CSIZE[isize])**3.0 * dust.CC[:, isize]  # erg/K  == integrand
            y  = concatenate(([0.0,], y.copy()))
            for iT in range(dust.CNT):                 # each row = single size, different temperatures
                ent = trapz(y[0:(iT+2)], x[0:(iT+2)])  # integral of C(T)*dT = E(T)
                fp.write(' %12.5e' % ent)
            fp.write('\n')
        fp.close()
    

        
        
class GSETDust(DustO):
    

    def __init__(self, filename):
        # Dust following GSTEDustO format
        # the input filename is the text file with "optical", "enthalpies", and "sizes" keywords
        self.EFF = False
        # read optical properties and size distributions from different files
        fopt, fent, fsize = None, None, None
        for line in open(filename).readlines():
            s = line.split()
            if (len(s)<2): continue
            if (s[0]=='optical'):    fopt  = s[1]
            if (s[0]=='enthalpies'): fent  = s[1]
            if (s[0]=='sizes'):      fsize = s[1]
        ##
        ## self.GRAIN_DENSITY = 1.0e-7  # DUMMY !!
        ## self.GRAIN_SIZE    = 1.0e-4  # DUMMY !!
        # read the size distribution -- ASSUME LOGARITHMIC BINS SO WE ONLY NEED CRT_SFRAC
        # => must use KabsCRT()
        self.GRAIN_DENSITY = float(open(fsize).readline().split()[0])
        d               =  loadtxt(fsize, skiprows=3)
        self.NSIZE      =  d.shape[0]
        self.SIZE_A     =  d[:,0].copy() * 1.0e-4     # [cm] -- the actual size bins used
        self.CRT_SFRAC  =  d[:,1].copy()              # IN THE FILE THIS IS sum()==1.0
        self.CRT_SFRAC *=  self.GRAIN_DENSITY         # WITHIN THIS SCRIPT CRT_SFRAC INCLUDES GRAIN_DENSITY !!
        self.SIZE_F     =  d[:,1].copy() * 0.0        # 1/cm/H   = dn/da/H --- we do not have this !!
        # the idea with GSETDustO is that size file already lists the fraction of grains per log size interval
        if (0):
            # ipFRAC gives d(grains)/da / H,  [a]=cm
            # extend definition of ipFRAC to cover all possible values -- zero outside input file range
            xx, yy = asarray(self.SIZE_A, float64), asarray(self.SIZE_F, float64)            
            # define also bin limits for the size distribution - assuming logarithmic bins
            self.SIZE_BO      =  zeros(self.NSIZE+1, float32)
            beta              =  self.SIZE_A[1] / self.SIZE_A[0]  # log step
            self.SIZE_BO[0]   =  self.SIZE_A[0] / sqrt(beta)
            self.SIZE_BO[1:]  =  self.SIZE_A * sqrt(beta)        
            self.ipFRAC     = interp1d(xx, yy, bounds_error=True, fill_value=0.0)
            self.ipFRAC_LL  = interp1d(log(xx), log(yy), bounds_error=True, fill_value=0.0)
        # read optical data
        lines       = open(fopt).readlines()
        s           = lines[0].split()
        self.QNSIZE = int(s[0])                   # sizes for which optical data given
        self.QNFREQ = int(s[1])                   # frequencies for which optical data given
        self.QSIZE  = zeros(self.QNSIZE, float64) # sizes for which Q defined
        self.QFREQ  = zeros(self.QNFREQ, float64)
        self.OPT    = zeros((self.QNSIZE, self.QNFREQ, 4), float64)     #  freq, Q_abs, Q_sca, g
        row         = 1   # we are on line containing size [um]
        for isize in range(self.QNSIZE):
            self.QSIZE[isize] = float(lines[row].split()[0]) * 1.0e-4  # [cm]
            A                 = pi*(self.QSIZE[isize])**2.0     # grain area [cm2]
            # DO NOT MULTIPLY WITH GRAIN SIZE BEFORE Q IS INTERPOLATED OVER GRAIN SIZES !!!
            row += 2   # skip over header line
            for ifreq in range(self.QNFREQ):
                s = lines[row].split()
                # OPT[isize, ifreq, 4] = [ um -> freq, Kabs, Ksca, g ], grain density 1/H included later
                self.OPT[isize, ifreq, :] = [float(s[0]), float(s[1]), float(s[2]), float(s[3]) ]
                row += 1
        self.QFREQ   =  self.OPT[0,:,0]  # already Hz, already in order of increasing frequency
        self.AMIN    =  self.SIZE_A[0]   # limits from the size file
        self.AMAX    =  self.SIZE_A[-1]
        print('  --- SIZE DISTRIBUTION SIZE_A  %12.4e - %12.4e' % (self.SIZE_A[0], self.SIZE_A[-1]))
        print('  --- OPTICAL DATA      SIZE    %12.4e - %12.4e' % (self.QSIZE[0], self.QSIZE[-1]))
        if (0):
            print('*** WARNING **** AMIN %12.4e < OPTICAL DATA AMIN %12.4e' % (self.AMIN, self.QSIZE[0]))
            print('MOVING AMIN %12.4e UP TO %12.4e' % (self.AMIN, self.QSIZE[0]))
            self.AMIN = self.QSIZE[0]
        else:
            if (self.AMIN<self.QSIZE[0]):
                print('*** WARNING **** AMIN %12.4e < OPTICAL DATA AMIN %12.4e' % (self.AMIN, self.QSIZE[0]))
                scale            = (self.AMIN/self.QSIZE[0])**2.0
                self.OPT[0,:,1] *= scale       # Kabs --- NOT YET SCALED WITH pi*a^2
                self.OPT[0,:,2] *= scale       # Ksca --- NOT YET SCALED WITH pi*a^2
                self.QSIZE[0]    = self.AMIN   # optical data "extrapolated" to self.AMIN
        if (self.AMAX>self.QSIZE[-1]):
            print('*** ERROR *** AMAX %12.4e > OPTICAL DATA AMAX %12.4e' % (self.AMAX, self.QSIZE[-1]))
            sys.exit()
        # -------------------------------------------------------------------------------------------
        # Read enthalpies ->  self.CT, self.CE, GSETDust file already has array for E, some size and 
        #                     some temperature grid
        lines  = open(fent, 'r').readlines()
        i = 0
        while(lines[i][0]=='#'):
            i += 1
        # number if grain sizes in the enthalpy table
        self.C_NSIZE = int(lines[i].split()[0])
        print("C_NSIZE %d" % self.C_NSIZE)
        self.C_SIZE  = zeros(self.C_NSIZE, float32)
        i += 1
        for j in range(self.C_NSIZE):
            self.C_SIZE[j] = float(lines[i].split()[0])
            i += 1
        self.C_SIZE *= 1.0e-4   # um -> cm
        # number of temperatures in the enthalpy table
        self.C_NTEMP = int(lines[i].split()[0])
        print("C_NTEMP %d" % self.C_NSIZE)
        self.C_TEMP  = zeros(self.C_NTEMP, float32)
        i += 1
        for j in range(self.C_NTEMP):
            self.C_TEMP[j] = float(lines[i].split()[0])
            i += 1
        # the rest of the file is the array E[iT, isize]
        self.C_E = loadtxt(fent, skiprows=i)
        if (self.C_E.shape[0]!=self.C_NSIZE):
            print("File %s: incorrect number of rows in self.C_E: %d != %d" % (fent, self.C_E.shape[0], self.C_NSIZE))
            sys.exit()
        if (self.C_E.shape[1]!=self.C_NTEMP):
            print("File %s: incorrect number of columns in self.C_E: %d != %d" % (fent, self.C_E.shape[0], self.C_NTEMP))
            sys.exit()
        
            
    def Kabs(self, freq_in):
        """
        Return KABS .... USE SUMMATION WITH LOG-SPACED SIZE GRID AND self.CRT_SFRAC
        Kabs() =  SUM [  pi*a^2  * Qabs * S_FRAC  ] 
        *** NOTE:  A2ELIB/SETDustO.cpp is similarly  sum(SKabs_Int) / GRAIN_DENSITY
        ***        A2ELIB/SETDustO.cpp                   SKabs_Int == pi*a^2*Q * S_FRAC*GRAIN_DENSITY
        ***        result includes sum over S_FRAC, does not include scaling with GRAIN_DENSITY
                   == sum( pi*a^2*Q * S_FRAC)
        """
        # find frequency indices surrounding freq
        if (isscalar(freq_in)):
            freqs = [freq_in,]
        else:
            freqs =  freq_in
        nfreq = len(freqs)
        res   = zeros(nfreq, float32)
        for ifreq in range(nfreq):
            freq   =  freqs[ifreq]
            i      =  argmin(abs(self.QFREQ-freq))  # note -- here self.QFREQ is in INCREASING order
            if (self.QFREQ[i]>freq): i-=1           #  i -= 1  is step to lower frequency!
            j      =  i+1                           #  j>i,  lower frequency
            # requested frequency between indices i and j
            i      =  clip(i, 0, self.QNFREQ-1)
            j      =  clip(j, 0, self.QNFREQ-1)
            if (i==j):
                wj = 0.5   # weight for frequency interpolation
            else:
                wj = (freq-self.QFREQ[i])  / (self.QFREQ[j]-self.QFREQ[i])
            wi     = 1.0-wj
            # we have to interpolate between the sizes given in Q files == in the OPT array
            #   from grid QSIZE[NQZIE] to grid   SIZE_A[NSIZE]
            # To be consistent with CRT (and dustem) need to interpolate Q ***BEFORE*** scaling that with a^2
            y1     =  self.OPT[:, i, 1]          # OPT[isize, ifreq, 4]  { freq, Kabs, Ksca, g }
            y2     =  self.OPT[:, j, 1]
            ip1    =  IP(self.QSIZE, y1.copy())  # Q1 = first frequency, interpolation over size
            ip2    =  IP(self.QSIZE, y2.copy())  # Q2 = second frequency, interpolation over size
            y1     =  get_IP(self.SIZE_A, ip1)
            y2     =  get_IP(self.SIZE_A, ip2)
            y      =  wi*y1 + wj*y2              # final Q
            # geometrical cross section multiplied in AFTER interpolation of Q factors
            kabs   =  sum(self.CRT_SFRAC*y* (pi*self.SIZE_A**2.0) ) # @@@ GRAIN_DENSITY INCLUDED IN CRT_SFRAC
            res[ifreq] = kabs
        if (isscalar(freq_in)): return res[0]
        return  res
    

    
    def SKabs_Int(self, isize, freq_in):
        """
        Return pi*a^2*Q * S_FRAC*GRAIN_DENSITY,  single grain size == identical to A2ELIB/GSETDustO.cpp
        Size is self.SIZE_A[isize], one of the final grain size bins.
        """
        # find frequency indices surrounding freq
        if (isscalar(freq_in)):
            freqs = [freq_in,]
        else:
            freqs =  freq_in
        nfreq = len(freqs)
        res   = zeros(nfreq, float64)
        for ifreq in range(nfreq):
            freq  = freqs[ifreq]
            i     = argmin(abs(self.QFREQ-freq))  # note -- here self.QFREQ is in INCREASING order
            if (self.QFREQ[i]>freq): i-=1         #  i -= 1  is step to lower frequency!
            j     = i+1                           #  j>i,  lower frequency
            # requested frequency between frequency indices i and j
            i     = clip(i, 0, self.QNFREQ-1)
            j     = clip(j, 0, self.QNFREQ-1)
            if (i==j):
                wj   = 0.5      # wj = weight for the higher frequency bin
            else:
                wj   = (freq-self.QFREQ[i])  / (self.QFREQ[j]-self.QFREQ[i])
            wi    = 1.0-wj
            if (ifreq==10): 
                 print("wj = %.3e, SIZE_A %10.3e, CRT_SFRAC %10.3e" % (wj, self.SIZE_A[isize], self.CRT_SFRAC[isize]))
            # we have to interpolate between the sizes given in Q files == in the OPT array
            #   from grid QSIZE[NQZIE] to grid   SIZE_A[NSIZE]
            # To be consistent with CRT (and dustem) need to interpolate Q ***BEFORE*** scaling that with a^2
            y1    =  self.OPT[:, i, 1]          # OPT[isize, ifreq, 4]  { freq, Kabs, Ksca, g }
            y2    =  self.OPT[:, j, 1]
            ip1   =  IP(self.QSIZE, y1.copy())  # first frequency, interpolation over size
            ip2   =  IP(self.QSIZE, y2.copy())  # second frequency, interpolation over size
            y1    =  get_IP(self.SIZE_A[isize], ip1)   # Q interpolated to current size, one frequency
            y2    =  get_IP(self.SIZE_A[isize], ip2)   # Q interpolated to current size, other frequency
            y     =  wi*y1 + wj*y2              # Q interpolated to current frequency
            # geometrical cross section multiplied in AFTER interpolation of Q factors
            kabs       =  y * (pi*self.SIZE_A[isize]**2.0)
            res[ifreq] = kabs
        res *=   self.CRT_SFRAC[isize] # @@@ GRAIN_DENSITY included in CRT_SFRAC     ## * self.GRAIN_DENSITY
        if (isscalar(freq_in)): return res[0]
        return  res

    
    def SKabs(self, isize, freq_in):
        """
        Return   pi*a^2*Qabs
        Note:  
            SKabs_Int() == SKabs() * S_FRAC[isize]*GRAIN_DENSITY
        """
        # find frequency indices surrounding freq
        if (isscalar(freq_in)):
            freqs = [freq_in,]
        else:
            freqs = freq_in
        nfreq = len(freqs)
        res   = zeros(nfreq, float32)
        for ifreq in range(nfreq):
            freq =  freqs[ifreq]
            i    =  argmin(abs(self.QFREQ-freq))  # note -- here self.QFREQ is in INCREASING order
            if (self.QFREQ[i]>freq): i-=1         #  i -= 1  is step to lower frequency!
            j    =  i+1                           #  j>i,  lower frequency
            # requested frequency between indices i and j
            i    = clip(i, 0, self.QNFREQ-1)
            j    = clip(j, 0, self.QNFREQ-1)
            if (i==j):   wj   = 0.5
            else:        wj   = (freq-self.QFREQ[i])  / (self.QFREQ[j]-self.QFREQ[i])
            wi   = 1.0-wj
            # we have to interpolate between the sizes given in Q files == in the OPT array
            #   from grid QSIZE[NQZIE] to grid   SIZE_A[NSIZE]
            # To be consistent with CRT (and dustem) need to interpolate Q ***BEFORE*** scaling that with a^2
            y1   =  self.OPT[:, i, 1]             # OPT[isize, ifreq, 4]  { freq, Kabs, Ksca, g }
            y2   =  self.OPT[:, j, 1]
            ip1  =  IP(self.QSIZE, y1.copy())     # first frequency, interpolation over size
            ip2  =  IP(self.QSIZE, y2.copy())     # second frequency, interpolation over size
            y1   =  get_IP(self.SIZE_A[isize], ip1)
            y2   =  get_IP(self.SIZE_A[isize], ip2)
            y    =  wi*y1 + wj*y2                 # Q interpolated over frequency
            # geometrical cross section multiplied in AFTER interpolation of Q factors
            kabs       =  y * (pi*self.SIZE_A[isize]**2.0) # no S_FRAC, no GRAIN_DENSITY
            res[ifreq] = kabs
        if (isscalar(freq_in)): return res[0]
        return  res
        

    def Ksca(self, freq_in):
        # Return KSCA .... USE SUMMATION WITH LOG-SPACED SIZE GRID AND self.CRT_SFRAC
        print("GSETDust::Ksca")
        if (isscalar(freq_in)):
            freqs = [freq_in,]
        else:
            freqs = freq_in
        nfreq = len(freqs)
        res   = zeros(nfreq, float32)
        for ifreq in range(nfreq):
            freq  = freqs[ifreq]
            i     = argmin(abs(self.QFREQ-freq))  # note -- here self.QFREQ is in INCREASING order
            if (self.QFREQ[i]>freq): i-=1              #  i -= 1  is step to lower frequency!
            j     = i+1                           #  j>i,  lower frequency
            # requested frequency between indices i and j
            i     = clip(i, 0, self.QNFREQ-1)
            j     = clip(j, 0, self.QNFREQ-1)
            if (i==j):
                wj   = 0.5
            else:
                wj   = (freq-self.QFREQ[i])  / (self.QFREQ[j]-self.QFREQ[i])
            wi    = 1.0-wj
            # we have to interpolate between the sizes given in Q files == in the OPT array
            #   from grid QSIZE[NQZIE] to grid   SIZE_A[NSIZE]
            # To be consistent with CRT (and dustem) need to interpolate Q ***BEFORE*** scaling that with a^2
            y1    =  self.OPT[:, i, 2]          # OPT[isize, ifreq, 4]  { freq, Kabs, Ksca, g }
            y2    =  self.OPT[:, j, 2]
            ip1   =  IP(self.QSIZE, y1.copy())  # first frequency, interpolation over size
            ip2   =  IP(self.QSIZE, y2.copy())  # second frequency, interpolation over size
            y1    =  get_IP(self.SIZE_A, ip1)
            y2    =  get_IP(self.SIZE_A, ip2)
            y     =  wi*y1 + wj*y2
            # geometrical cross section multiplied in AFTER interpolation of Q factors
            kabs  =  sum(self.CRT_SFRAC*y* (pi*self.SIZE_A**2.0) )
            res[ifreq] = kabs
        if (isscalar(freq_in)): return res[0]
        return  res

    
    
    def Gsca(self, freq_in):
        # Return effective g parameter by summation over the size distribution
        #  sum(g[isize]*Ksca(isize)) / sum(Ksca(isize))
        if (isscalar(freq_in)):
            freqs = [freq_in,]
        else:
            freqs = freq_in
        nfreq = len(freqs)
        res   = zeros(nfreq, float32)
        for ifreq in range(nfreq):
            freq  = freqs[ifreq]
            i     = argmin(abs(self.QFREQ-freq))  # note -- here self.QFREQ is in INCREASING order
            if (self.QFREQ[i]>freq): i-=1              #  i -= 1  is step to lower frequency!
            j     = i+1                           #  j>i,  lower frequency
            # requested frequency between indices i and j
            i     = clip(i, 0, self.QNFREQ-1)
            j     = clip(j, 0, self.QNFREQ-1)
            if (i==j):
                wj   = 0.5
            else:
                wj   = (freq-self.QFREQ[i])  / (self.QFREQ[j]-self.QFREQ[i])
            wi    = 1.0-wj
            # we have to interpolate between the sizes given in Q files == in the OPT array
            #   from grid QSIZE[NQZIE] to grid   SIZE_A[NSIZE]
            # To be consistent with CRT (and dustem) need to interpolate Q ***BEFORE*** scaling that with a^2
            y1    =  self.OPT[:, i, 2]          # OPT[isize, ifreq, 4]  { freq, Kabs, Ksca, g }
            y2    =  self.OPT[:, j, 2]
            ip1   =  IP(self.QSIZE, y1.copy())  # first frequency, interpolation over size
            ip2   =  IP(self.QSIZE, y2.copy())  # second frequency, interpolation over size
            y1    =  get_IP(self.SIZE_A, ip1)
            y2    =  get_IP(self.SIZE_A, ip2)
            y     =  wi*y1 + wj*y2              # Ksca for size grid, interpolated in frequency
            # interpolate similarly the g values
            y1    =  self.OPT[:, i, 3]          # OPT[isize, ifreq, 4]  { freq, Kabs, Ksca, g }
            y2    =  self.OPT[:, j, 3]
            ip1   =  IP(self.QSIZE, y1.copy())  # first frequency, interpolation over size
            ip2   =  IP(self.QSIZE, y2.copy())  # second frequency, interpolation over size
            y1    =  get_IP(self.SIZE_A, ip1)
            y2    =  get_IP(self.SIZE_A, ip2)
            g     =  wi*y1 + wj*y2              # g for entire size grid, interpolated in frequency
            # geometrical cross section multiplied in AFTER interpolation of Q factors
            res[ifreq] = sum(self.CRT_SFRAC*y*(pi*self.SIZE_A**2.0)*g) / \
                         sum(self.CRT_SFRAC*y*(pi*self.SIZE_A**2.0))
        if (isscalar(freq_in)): return res[0]
        return  res
    
    
    def E2T(self, isize, E):
        """
        Convert grain energy to temperature. 
        Input:
            isize  =  index to the main size discretisation self.SIZE_A
            E      =  grain energy
        Return:
            temperature value interpolated from self.C_SIZE and self.C_E
            these grids are different from the actual size discretisation used
            It seems that E(a)/a^3 is in practice constant - use iw for that interpolation
            It seems that E(T) is less linear - but is nearly linear on log-log scale
        """
        # Find closest size bins
        a  = self.SIZE_A[isize]
        for i in range(1, self.C_NSIZE):
            if (a<self.C_SIZE[i]): break
        i -= 1   #  current size should be between isize and isize+1 bins in the C_E array
        # Interpolate the two vectors in size -->  E(T) vector for the given size
        iw     = (self.C_SIZE[isize+1]-a) / (self.C_SIZE[isize+1]-self.C_SIZE[isize])
        Etmp   =  iw*(self.C_E[i,:]/self.C_SIZE[i]**3.0) + (1.0-iw)*(self.C_E[i+1,:]/self.C_SIZE[i+1]**3.0)
        Etmp  *=  a**3   # for the actual grain size, after interpolating E/a^3
        # Interpolate along this vector = not E(T) but T(E)
        if (1): # linear interpolation along the energy axis
            ip     =  interp1d(Etmp, self.C_TEMP)
            return ip(E)
        else:   # linear interpolation on logarithmic scale
            ip     =  interp1d(log(Etmp), log(self.C_TEMP))
            return exp(ip(log(E)))
            
        
    def T2E(self, isize, T):
        """
        Convert temperature to grain energy.
        Input:
            isize  =  index to the main size discretisation self.SIZE_A
            T      =  grain temperature
        Return:
            temperature value interpolated from self.C_SIZE and self.C_E
        Note:
            2018-12-24 -- tested E2T() and T2E() - results ok.
        """
        # Find closest size bins
        a  = self.SIZE_A[isize]
        for i in range(1, self.C_NSIZE):
            if (a<self.C_SIZE[i]): break
        i -= 1   #  current size should be between isize and isize+1 bins in the C_E array
        # Interpolate the two vectors for the two sizes in C_SIZE[]
        iw     = (self.C_SIZE[isize+1]-a) / (self.C_SIZE[isize+1]-self.C_SIZE[isize])
        Etmp   =  iw*(self.C_E[i,:]/self.C_SIZE[i]**3.0) + (1.0-iw)*(self.C_E[i+1,:]/self.C_SIZE[i+1]**3.0)
        Etmp  *=  a**3   # for the actual grain size, after interpolating E/a^3
        # Interpolate along this vector = along temperature axis
        if (1): # linear interpolation along the temperature axis
            ip     =  interp1d(self.C_TEMP, Etmp)
            return ip(T)
        else:   # linear interpolation on logarithmic scale
            ip     =  interp1d(log(self.C_TEMP), log(Etmp))
            return exp(ip(log(T)))
            
        
        
        
        
        
