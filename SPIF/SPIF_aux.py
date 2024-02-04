from   matplotlib.pylab import *
import numpy as np
import pyopencl as cl
from   astropy.io import fits as pyfits

def FixSize(n, L):
    # Return n fixed to a the smallest integer nn>=n that is multiple of L
    nn = (n//L)*L
    if ((nn*L)<n): nn += L
    return nn


def InitCL(GPU=0, platforms=[], sub=0, verbose=False):
    """
    Usage:
        platform, device, context, queue, mf = InitCL(GPU=0, platforms=[], sub=0)
    Input:
        GPU       =  if >0, try to return a GPU device instead of CPU
        platforms =  optional array of possible platform numbers
        sub       =  optional number of threads for a subdevice (first returned)
    """
    platform, device, context, queue = None, None, None, None
    possible_platforms = range(6)
    if (len(platforms)>0):
        possible_platforms = platforms
    device = []
    for iplatform in possible_platforms:
        if (verbose): print("try platform %d... for GPU=%d" % (iplatform, GPU))
        try:
            # print(cl.get_platforms())
            platform     = cl.get_platforms()[iplatform]
            # print(platform)
            if (GPU>0):
                device   = platform.get_devices(cl.device_type.GPU)
            else:
                device   = platform.get_devices(cl.device_type.CPU)
            #print(device)
            if (sub>0):
                # try to make subdevices with sub threads, return the first one
                dpp       =  cl.device_partition_property
                device    =  [device[0].create_sub_devices( [dpp.EQUALLY, sub] )[0],]
            context   =  cl.Context(device)
            queue     =  cl.CommandQueue(context)
            break
        except:
            pass
    if (verbose):
        print(device)
    return platform, device, context, queue,  cl.mem_flags



def Jfun(T, T0):
    return T0/(exp(T0/T)-1)



def WriteGradient(T1, T2=""):
    """
    Write kernel code to calculate dU/dp given the formulas in the string T, e.g.
        y1 = GAUSS(x1, x[0], x[1], x[2]).
    One can assume that @y1 is replaced by this formula to calculate y1.
        dC2/dp =  1/dY^2  sum  (y1-Y1[i])  * dy1/dp,
    for y = GAUSS(v, x[0], x[1], x[1]):
        dy/dx[0]  =  y/x[0] ;
        dy/dx[1]  =  2*Q * (v-x[1])    / x[2]^2   *  y
        dy/dx[2]  =  2*Q * (v-x[1])^2  / c^3      *  y
    for y = HFS(v, x0, x1, x2, x3)
        q         =  exp(-Q*((HFV+x1-v)/x2)^2)
        z         =  exp(-HFI*x3*q)
        r         =  c/(exp(c/x0)-J0)
        dy/dx0    =  c^2 * (1-z) * exp(c/x0) / [ x0^2 * (exp(c/x0)-1)^2 ]
        dy/dx1    =  2*HFI*Q*x3*r*(HFV+x1-v)    *z*q / x2^2
        dy/dx2    =  2*HFI*Q*x3*r*(HFV+x1-v)^2 * z*q / x2^3
        dy/dx3    =    HFI     *r*               z*q
    """
    CODE = ""
    for T in [T1, T2]:
        if (len(T)<2): continue
        ind  = T.find('y1')
        if (ind>=0):
            SPE = 1
        else:       
            ind = T.find('y2')
            SPE = 2
        TT = T[ind:]
        ind = T.find('GAUSS')
        if (ind<0): continue
        CODE +=   """
         for(int i=0; i<M%d; i++) {
            v     =  V%d[i] ;
            y     =  @yx ;
            tmp   =  2.0f * (y-Y%d[i])/(dY%d*dY%d) ;
        """ % (SPE, SPE, SPE, SPE, SPE)
        # added above 2.0f to tmp from the ()^2 derivation....
        CODE = CODE.replace('@yx', T[ind:].replace('v1','v').replace('v2',''))
        # TODO: multiple gaussians => below should process each GAUSS() separately!!
        while (1):
            k = TT.find('GAUSS')
            if (k<0): break
            TT = TT[(k+6):]     # drop "GAUSS"
            k  = TT.find(')')
            CURRENT = TT[0:k+1] # only one term GAUSS(...)
            s  = CURRENT[0:k].split(',')[1:] # list of the arguments 
            a  = s[0].strip()   # could be x[0], could be something else, including a constant?
            b  = s[1].strip()
            c  = s[2].strip()
            C  = "    y = GAUSS(v,%s,%s,%s) ;\n" % (a,b,c)
            C += """        
            g@00 +=  tmp * (y/@00) ;
            g@01 +=  tmp * 2.0f*Q*(v-@01)/(@02*@02) * y ;
            g@02 +=  tmp * 2.0f*Q*(v-@01)*(v-@01)*y/pown(@02,3) ;"""
            C = C.replace('@00', a)
            C = C.replace('@01', b)
            C = C.replace('@02', c)
            CODE += C
        CODE += '\n         }\n'
    for T in [T1, T2]:
        if (len(T)<2): continue
        ind  = T.find('y1')
        if (ind>=0):
            SPE = 1
        else:       
            ind = T.find('y2')
            SPE = 2
        TT = T[ind:]
        if (T.find('HFSX')): continue
        ind = T.find('HFS')
        if (ind<0): continue
        CODE +=   """
         for(int i=0; i<M%d; i++) {
            v     =  V%d[i] ;
            y     =  @yx ;
            tmp   =  2.0f * (y-Y%d[i])/(dY%d*dY%d) ;
        """ % (SPE, SPE, SPE, SPE, SPE)
        # added above 2.0f to tmp from the ()^2 derivation....
        CODE = CODE.replace('@yx', T[ind:].replace('v1','v').replace('v2',''))        
        while (1):
            k = TT.find('HFS')
            if (k<0): break
            TT = TT[(k+6):]
            k  = TT.find(')')
            s  = TT[0:k].split(',')[1:] # list of the arguments 
            x0 = s[0].strip() 
            x1 = s[1].strip()
            x2 = s[2].strip()
            x3 = s[3].strip()
            C = """        
            REAL tau = ZERO ;
            for(int j=0; j<NHF1; j++) tau += GAUSS(v, @03*IHF%d[j], @01+VHF%d[j], @02) ;            
            g@00 +=  tmp  * pown(HFK%d/(@00*(exp(HFK%d/@00)-ONE)), 2) * exp(HFK%d/@00) * (ONE-exp(-tau)) ;
            REAL r = (Jfun(@00, HFK%d)-Jfun(TBG, HFK%d)) * exp(-tau) ;        
            for(int j=0; j<NHF1; j++) {               
               REAL z = exp(-Q*pown( (@01+VHF%d[j]-v)/@02, 2)) ;
               REAL q = TWO*IHF%d[j]*Q*@03*(v-@01-VHF%d[j])*z/(@02*@02) ; 
               g@01 +=  tmp * r * q ;
               g@02 +=  tmp * r * q * (v-@01-VHF%d[j])  / @02 ;
               g@03 +=  tmp * r * IHF%d[j]*z ;
            }
            """ % (SPE, SPE,  SPE, SPE,  SPE, SPE,  SPE, SPE, SPE, SPE, SPE, SPE)
            C = C.replace('@00', x0)
            C = C.replace('@01', x1)
            C = C.replace('@02', x2)
            C = C.replace('@03', x3)
            CODE += C
        CODE += '\n         }\n'
    for T in [T1, T2]:
        if (len(T)<2): continue
        ind  = T.find('y1')
        if (ind>=0):
            SPE = 1
        else:       
            ind = T.find('y2')
            SPE = 2
        TT = T[ind:]
        ind = T.find('HFSX')
        if (ind<0): continue
        CODE +=   """
         for(int i=0; i<M%d; i++) {
            v     =  V%d[i] ;
            y     =  @yx ;
            tmp   =  2.0f * (y-Y%d[i])/(dY%d*dY%d) ;
        """ % (SPE, SPE, SPE, SPE, SPE)
        # added above 2.0f to tmp from the ()^2 derivation....
        CODE = CODE.replace('@yx', T[ind:].replace('v1','v').replace('v2',''))        
        while (1):
            k = TT.find('HFS')
            if (k<0): break
            TT = TT[(k+6):]
            k  = TT.find(')')
            s  = TT[0:k].split(',')[1:] # list of the arguments 
            x0 = s[0].strip() 
            x1 = s[1].strip()
            x2 = s[2].strip()
            x3 = s[3].strip()
            C = """        
            REAL tau = ZERO ;
            for(int j=0; j<NHF1; j++) tau += GAUSS(v, @03*IHF%d[j], @01+VHF%d[j], @02) ;            
            g@00  +=  tmp  * (ONE-exp(-tau)) ;
            REAL r =  @00  * exp(-tau) ;        
            for(int j=0; j<NHF1; j++) {               
               REAL z = exp(-Q*pown( (@01+VHF%d[j]-v)/@02, 2)) ;
               REAL q = TWO*IHF%d[j]*Q*@03*(v-@01-VHF%d[j])*z/(@02*@02) ; 
               g@01 +=  tmp * r * q ;
               g@02 +=  tmp * r * q * (v-@01-VHF%d[j])  / @02 ;
               g@03 +=  tmp * r * IHF%d[j]*z ;
            }
            """ % (SPE,  SPE, SPE, SPE, SPE, SPE, SPE)
            C = C.replace('@00', x0)
            C = C.replace('@01', x1)
            C = C.replace('@02', x2)
            C = C.replace('@03', x3)
            CODE += C
        CODE += '\n         }\n'
    if (0):
        print(CODE)
    return CODE



def SHOW():
    show(block=True)
