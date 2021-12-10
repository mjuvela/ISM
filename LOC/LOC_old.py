#!/usr/bin/python

import os, sys
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)   
from   LOC_aux import *

"""
x = cl.cltypes.make_float2() 
x['x'], x['y']

x = np.zeros((3,3), cl.cltypes.float2) 
x[0,0]['x']
"""

if (len(sys.argv)<2):
    print("Usage:  LOC.py  <ini-file>")
    sys.exit()
    
INI         =  ReadIni(sys.argv[1])
# print(INI)

MOL         =  ReadMolecule(INI['molecule'])
HFS         =  len(INI['hfsfile'])>0              # HFS with LTE components

NFREQ       =  len(MOL.F)
LEVELS      =  INI['levels']
TRANSITIONS =  MOL.Transitions(LEVELS)            # how many transitions among LEVELS levels
CHANNELS    =  INI['channels']
RHO, TKIN, CLOUD, ABU   =   ReadCloud3D(INI, MOL)

print("================================================================================")
print("CLOUD")
print("    density  %10.3e  %10.3e" % (min(RHO), max(RHO)))
print("    Tkin     %10.3e  %10.3e" % (min(TKIN), max(TKIN)))
print("    Sigma    %10.3e  %10.3e" % (min(ravel(CLOUD[:,:,:]['w'])), max(ravel(CLOUD[:,:,:]['w']))))
print("    vx       %10.3e  %10.3e" % (min(ravel(CLOUD[:,:,:]['x'])), max(ravel(CLOUD[:,:,:]['x']))))
print("    vy       %10.3e  %10.3e" % (min(ravel(CLOUD[:,:,:]['y'])), max(ravel(CLOUD[:,:,:]['y']))))
print("    vz       %10.3e  %10.3e" % (min(ravel(CLOUD[:,:,:]['z'])), max(ravel(CLOUD[:,:,:]['z']))))
print("================================================================================")

Z, Y, X     =  CLOUD.shape  # CLOUD[Z, Y, X].float4,  RHO and TKIN remain 1d vectors
CELLS       =  X*Y*Z
NSIDE       =  int(INI['nside'])
NDIR        =  12*NSIDE*NSIDE
WIDTH       =  INI['bandwidth']/INI['channels']   # channel width [km/s], even for HFS calculations
AREA        =  2.0*(X*Y+Y*Z+Z*X) 
NRAY        =  ((Y+1)//2) * ((Z+1)//2)     #   what if X >> max(Y,Z) ???
VOLUME      =  1.0/CELLS                   #   Vcell / Vcloud
GL          =  INI['angle'] * ARCSEC_TO_RADIAN * INI['distance'] * PARSEC
APL         =  0.0
print("GRID_LENGTH = %12.4e" % GL)
WITH_HALF   =  0
WITH_PL     =  0
LOWMEM      =  INI['lowmem']
COOLING     =  INI['cooling']
if (COOLING & HFS):
    print("*** Cooling not implemented for HFS => cooling will not be calculated!")
    COOLING =  0
if (HFS):
    BAND, MAXCHN, MAXCMP = ReadHFS(INI, MOL)     # CHANNELS becomes the maximum over all transitions
    print("HFS revised =>  CHANNELS %d,  MAXCMP = %d" % (CHANNELS, MAXCMP))
    HF      =  zeros(MAXCMP, cl.cltypes.float2)
    
    
print("TRANSITIONS %d, CELLS %d = %d x %d x %d" % (TRANSITIONS, CELLS, X, Y, Z))
ESC_ARRAY   =  zeros((CELLS, TRANSITIONS), float32)
SIJ_ARRAY   =  zeros((CELLS, TRANSITIONS), float32)

WITH_CRT      =  INI['with_crt']
CRT_TAU     =  []
CRT_EMI     =  []
if (WITH_CRT):
    CRT_TAU =  ReadDustTau('crt.opacity', GL, CELLS, TRANSITIONS)
    CRT_EMI =  ReadDustEmission('crt.emission', CELLS, TRANSITIONS, WIDTH, MOL)
    # conversion from photons / s / channel / H    -->   photons / s / channel / cm3
    RHO.shape = (Z, Y, X)
    for t in range(TRANSITIONS):
        CRT_EMI[:,t] *=  RHO
    RHO       = ravel(RHO)
    
    
GNO         =  55   # number of precalculated Gaussians --- perhaps should be calculated on to fly
# unlike for LOC1D.py which has GAU[TRANSITIONS, CELLS, CHANNELS], LOC.py still has
# GAU[GNO, CHANNELS]
G0, GX, GAU, LIM =  GaussianProfiles(INI['min_sigma'], INI['max_sigma'], GNO, CHANNELS, WIDTH)
MAXCHN           =  INI['channels']

if (INI['GPU']):  LOCAL = 32
else:             LOCAL =  4
GLOBAL        =  IRound(NRAY, 64)
platform, device, context, queue,  mf = InitCL(INI['GPU'], INI['platforms'])

OPT = "-D O2 -D X=%d -D Y=%d -D Z=%d -D NRAY=%d -D CHANNELS=%d -D WIDTH=%.5f \
-D VOLUME=%.5e -D CELLS=%d -D LOCAL=%d -D GLOBAL=%d -D GNO=%d -D SIGMA0=%.5f -D SIGMAX=%.4f \
-D GL=%.4e -D MAXCHN=%d -D WITH_HALF=%d -D WITH_PL=%d -D LOC_LOWMEM=%d \
-D BRUTE_COOLING=%d -D LEVELS=%d -D TRANSITIONS=%d -D WITH_HFS=%d -D WITH_CRT=%d" % \
(X, Y, Z, NRAY, CHANNELS, WIDTH, VOLUME, CELLS, LOCAL, GLOBAL, GNO, G0, GX,
GL, MAXCHN, WITH_HALF, WITH_PL, LOWMEM, (COOLING==2),     # BRUTE COOLING == (COOLING==2)
LEVELS, TRANSITIONS, HFS, WITH_CRT)
if (0):
    OPT  += " -cl-fast-relaxed-math"
print(OPT)

source        =  open(INSTALL_DIR+"/kernel_update_py.c").read()
print("--------------------------------------------------------------------------------")
program       =  cl.Program(context, source).build(OPT)
print("--------------------------------------------------------------------------------")

# Set up kernels
kernel_clear  =  program.Clear
kernel_paths  =  program.Paths
kernel_sim    =  program.Update
kernel_solve  =  program.SolveCL

kernel_sim.set_scalar_arg_dtypes([
# 0     1     2     3           4           5           6           7           8     
# CLOUD GAU   LIM   Aul         A_b         GN          APL         BG          DIRWEI 
None,   None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32,
# 9        10                 11                  12     13     14    
# LEADING  POS0               DIR                 NI     RES    NTRUE 
np.int32,  cl.cltypes.float4, cl.cltypes.float4,  None,  None,  None  ])

   
#                                   RES[CELLS].xy
kernel_clear.set_scalar_arg_dtypes([None, ])

# THIS ASSUMING WITH_PL=0
#                                   TPL   COUNT  LEADING   POS0               DIR                
kernel_paths.set_scalar_arg_dtypes([None, None,  np.int32, cl.cltypes.float4, cl.cltypes.float4])


if (HFS):   
    kernel_hf    =  program.UpdateHF
    kernel_hf.set_scalar_arg_dtypes([
    # 0     1     2     3           4           5           6           7           8           9        
    # CLOUD GAU   LIM   Aul         A_b,        GN          APL         BG          DIRWEI      LEADING  
    None,   None, None, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.int32,
    # 10               11                  12    13    14        15        16     17    
    # POS              DIR                 NI    RES   NCHN      NCOMP     HF     NTRUES
    cl.cltypes.float4, cl.cltypes.float4,  None, None, np.int32, np.int32, None,  None])
else:       
    kernel_hf    =  None


kernel_solve.set_scalar_arg_dtypes(
# 0        1     2     3     4     5         6         7         8         9     10    11   
# BATCH    A     UL    E     G     PARTNERS  NTKIN     NCUL      MOL_TKIN  CUL   C     CABU 
[np.int32, None, None, None, None, np.int32, np.int32, np.int32, None,     None, None, None,
# 12  13    14    15    16    17    18    19    
# RHO TKIN  ABU   NI    SIJ   ESC   RES   WRK   
None, None, None, None, None, None, None, None ])

# Set up input and output arrays
PL_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)
GAU_buf   =  cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=GAU)
LIM_buf   =  cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=LIM)
TPL_buf   =  cl.Buffer(context, mf.WRITE_ONLY,  4*NRAY)
COUNT_buf =  cl.Buffer(context, mf.READ_WRITE,  4*NRAY)
CLOUD_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=CLOUD)   # vx, vy, vz, sigma
NI_buf    =  cl.Buffer(context, mf.READ_ONLY,   8*CELLS)                          # nupper, nb_nb
RES_buf   =  cl.Buffer(context, mf.WRITE_ONLY,  8*CELLS)                          # SIJ, ESC
HF_buf    =  None
if (HFS):
    HF_buf  =  cl.Buffer(context, mf.READ_ONLY,   8*MAXCMP)

if (COOLING==2):
    COOL_buf = cl.Buffer(context,mf.WRITE_ONLY, 4*CELLS)
    
NTRUE_buf =  cl.Buffer(context, mf.READ_WRITE, 4*max([INI['points'][0], NRAY])*MAXCHN)
STAU_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*max([INI['points'][0], NRAY])*MAXCHN)

WRK       =  np.zeros(CELLS, cl.cltypes.float2)  #  NI and RES


# Check explicitly the number of rays and the path travelled
PACKETS   =  0
DIRWEI, TPL, COUNT = [], [], []
if (INI['iterations']>0):
    DIRWEI    =  zeros(NDIR, float32)
    TPL       =  zeros(NRAY, float32)
    COUNT     =  zeros(NRAY, int32)
    for idir in range(NDIR):
        for ioff in range(4):  # concurrent rays two cells apart
            POS, DIR, LEADING = GetHealpixDirection(NSIDE, ioff, idir, X, Y, Z)        
            kernel_paths(queue, [GLOBAL,], [LOCAL], TPL_buf, COUNT_buf, LEADING, POS, DIR)
            queue.finish()
            cl.enqueue_copy(queue, TPL,   TPL_buf)
            cl.enqueue_copy(queue, COUNT, COUNT_buf)
            queue.finish()
            for i in range(NRAY):
                DIRWEI[idir]  += TPL[i]
                PACKETS       += COUNT[i]    
    print("NRAY %d, NDIR %d, NDIR*X*Y %d, PACKETS %d" % (NRAY, NDIR, NDIR*X*Y, PACKETS))
    APL    =  sum(DIRWEI)/CELLS
    print("Average path length %.3e" % APL)
    tmp    = sum(DIRWEI)/NDIR
    DIRWEI = tmp/DIRWEI     # relative weight for each direction ~  <cos_theta>/cos(theta)
    

# Read or generate NI_ARRAY
NI_ARRAY = zeros((CELLS, LEVELS), float32)
ok = False
if (len(INI['load'])>0):  # load saved level populations
    try:
        fp = open(INI['load'], 'rb')
        x, y, z, lev = fromfile(fp, int32, 4)
        if ((x!=X)|(y!=Y)|(z!=Z)|(lev!=LEVELS)):
            print("Reading %s => %d x %d x %d cells, %d levels" % (x, y, z, lev))
            print("but we have now %d x %d x %d cells, %d levels ?? "  % (X, Y, Z, LEVELS))
            sys.exit()
        NI_ARRAY = fromfile(fp, float32).reshape(CELLS, NFREQ)
        fp.close()
        ok = True
        print("Level populations read from: %s" % INI['load'])
    except:
        print("Failed to load level populations from: %s" % INI['load'])
        pass
if (not(ok)): # reset LTE populations
    print("Level populations reset to LTE values !!!")
    J = asarray(arange(LEVELS), int32)
    for icell in range(CELLS):
        NI_ARRAY[icell,:] = RHO[icell] * ABU[icell] * MOL.Partition(J, TKIN[icell])
    # write also directly to disk
    if (len(INI['save'])>0):
        fp = open(INI['save'], 'wb')
        asarray([X, Y, Z, LEVELS], int32).tofile(fp)
        asarray(NI_ARRAY, float32).tofile(fp)
        fp.close()
        print("Level populations (LTE) saved to: %s" % INI['save'])
        if (0):
            for tr in range(TRANSITIONS):
                u, l = MOL.T2L(tr)
                print("  <ni[%02d]*gg-ni[%02d]> = %11.3e" % (l, u, mean(NI_ARRAY[:,l]*MOL.GG[tr]-NI_ARRAY[:,u])))
            Tkin = 10.0
            for tr in range(TRANSITIONS):
                u, l = MOL.T2L(tr)
                print("  %4.1f==%4.1f <ni[%02d]*gg-ni[%02d]> = %11.3e" % (
                MOL.GG[tr],  (2.0*u+1.0)/(2.0*l+1.0),       l, u,
                MOL.GG[tr]*MOL.Partition(l, Tkin) - MOL.Partition(u, Tkin)) )
            Tkin = 20.0
            for tr in range(TRANSITIONS):
                u, l = MOL.T2L(tr)
                print("  %4.1f==%4.1f <ni[%02d]*gg-ni[%02d]> = %11.3e" % (
                MOL.GG[tr],  (2.0*u+1.0)/(2.0*l+1.0),       l, u,
                MOL.GG[tr]*MOL.Partition(l, Tkin) - MOL.Partition(u, Tkin)) )
            sys.exit()

        
#================================================================================
#================================================================================
#================================================================================
#================================================================================


def Simulate():
    global INI, MOL, queue, LOCAL, GLOBAL, WIDTH, VOLUME, GL, COOLING, NSIDE, HFS
    global RES_buf, GAU_buf, CLOUD_buf, NI_buf, LIM_buf
    ncmp    =  1
    tmp_1   =  C_LIGHT*C_LIGHT/(8.0*pi)
    Tbg     =  INI['Tbg']
    SUM_COOL, LEV_COOL, hf = [], [], []
    if (COOLING==2):
        SUM_COOL = zeros(CELLS, float32)
        LEV_COOL = zeros(CELLS, float32)
        hf       = MOL.F*PLANCK/VOLUME        
    sys.stdout.write('      ')
    for tran in range(MOL.TRANSITIONS): # ------>
        upper, lower  =  MOL.T2L(tran)
        Ab            =  MOL.BB[tran]
        Aul           =  MOL.A[tran]
        freq          =  MOL.F[tran]
        gg            =  MOL.GG[tran]
        BGPHOT        =  Planck(freq, Tbg)*pi*AREA/(PLANCK*C_LIGHT)*(1.0e5*WIDTH)*VOLUME/GL
        BG            =  BGPHOT/PACKETS        
        if (HFS):
            nchn = BAND[tran].Channels()
            ncmp = BAND[tran].N
            for i in range(ncmp):
                HF[i]['x']  =  round(BAND[tran].VELOCITY[i]/WIDTH) # offset in channels
                HF[i]['y']  =  BAND[tran].WEIGHT[i]
            HF[0:ncmp]['y']  /= sum(HF[0:ncmp]['y'])
            cl.enqueue_copy(queue, HF_buf, HF)
        ###
        GNORM         = (C_LIGHT/(1.0e5*WIDTH*freq)) * GL  # GRID_LENGTH multiplied to gauss norm
        sys.stdout.write(' %2d' % tran)
        sys.stdout.flush()
        kernel_clear(queue, [GLOBAL,], [LOCAL,], RES_buf)
        # Upload NI[upper] and NB_NB[tran] values
        tmp  =  tmp_1 * Aul * (NI_ARRAY[:,lower]*gg-NI_ARRAY[:,upper]) / (freq*freq)  # [CELLS]
        tmp  =  clip(tmp, -1.0e-3, 1e99)
        tmp[nonzero(abs(tmp)<1.0e-25)] = 2.0e-25
        WRK[:]['x'] = NI_ARRAY[:, upper]
        WRK[:]['y'] = tmp        
        cl.enqueue_copy(queue, NI_buf, WRK)
        # the following loop is 99% of the Simulate() routine run time
        for idir in range(NDIR):
            for ioff in range(4):
                POS, DIR, LEADING  =  GetHealpixDirection(NSIDE, ioff, idir, X, Y, Z) # < 0.001 seconds !
                WEI                =  DIRWEI[idir]                
                if (ncmp==1):
                    #                                       0          1        2        3    4   5      6  
                    kernel_sim(queue, [GLOBAL,], [LOCAL,],  CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab, GNORM, APL,
                    # 7   8    9        10   11   12      13       14        
                    BG,   WEI, LEADING, POS, DIR, NI_buf, RES_buf, NTRUE_buf)
                else:
                    #                                       0          1        2        3    4   5      6   
                    kernel_hf(queue, [GLOBAL,], [LOCAL,],   CLOUD_buf, GAU_buf, LIM_buf, Aul, Ab, GNORM, APL, 
                    # 7  8    9        10   11   12      13       14    15    16      17       
                    BG,  WEI, LEADING, POS, DIR, NI_buf, RES_buf, nchn, ncmp, HF_buf, NTRUE_buf)
                queue.finish()                
        cl.enqueue_copy(queue, WRK, RES_buf)
        # yes - all time in Simulate spent in the above idir loop
        # LOC.cpp simulation take 0.8 seconds, LOC.py 1.5 seconds !!??
        SIJ_ARRAY[:, tran] = WRK[:]['x']
        ESC_ARRAY[:, tran] = WRK[:]['y']                
        if (COOLING==2):
            cl.enqueue_copy(queue, LEV_COOL, COOL_buf)
            SUM_COOL[:]   +=   LEV_COOL[:] * hf[tran]    # per cm3
        if (0):
            print("       tran = %3d  = %2d - %2d  => <SIJ> = %.3e   <ESC> = %.3e" % 
            (tran, upper, lower, mean(WRK[:]['x']), mean(WRK[:]['y'])))
    sys.stdout.write('\n')
    # <--- for tran ---
    if (COOLING==2):
        print("BRUTE COOLING: %10.3e" % (sum(SUM_COOL)/CELLS))
        fpb = open('brute.cooling', 'wb')
        asarray(SUM_COOL, float32).tofile(fpb)
        fpb.close()
        SUM_COOL = []
        LEV_COOL = []
        

        
        
            
def Cooling():
    """
    cell emits             n_u*Aul  photons / cm3
    escaping photons       ESC
    all absorbed in        SIJ
       => enough information to calculate net cooling
    COOL  =  2*ESC - SIJ*NI[lower] - n[upper]*Aul
    """
    global CELLS, TRANSITIONS, MOL, INI, SIJ_ARRAY, ESC_ARRAY, NI_ARRAY
    COOL  =  zeros(CELLS, float32)
    Aul   =  MOL.A
    hf    =  MOL.F*PLANCK
    AVE   =  0.0
    U, L  =  zeros(TRANSITIONS, int32), zeros(TRANSITIONS, int32)
    for tr in range(TRANSITIONS):
        u, l         =  MOL.T2L(tr)
        U[tr], L[tr] =  u, l   
    for icell in range(CELLS):
        # COOL[icell] +=  sum(hf[:] * (  NI_ARRAY[icell,[U[:]]*Aul[:] - SIJ_ARRAY[icell,:]*NI[L[:]] ))
        COOL[icell] +=  sum(hf[:] * (  ESC_ARRAY[icell,:]/VOLUME - SIJ_ARRAY[icell,:]*NI_ARRAY[icell, L[:]] ))
    print("COOLING: AVERAGE %12.4e" % (mean(COOL)))
    fp = open('cooling.bin', 'wb')
    asarray(COOL, float32).tofile(fp)
    fp.close()
    
    
    
def Solve():
    global  MOL, INI, LEVELS, TKIN, RHO, ABU, ESC_ARRAY
    NI_LIMIT  = 1.0e-24
    PARTNERS  = MOL.PARTNERS
    CHECK     = min([INI['uppermost']+1, LEVELS])  # check this many lowest energylevels
    cab       = ones(10, float32)                  # scalar abundances of different collision partners
    for i in range(PARTNERS):
        cab[i] = MOL.CABU[i]                       # default values == 1.0
    # possible abundance file for abundances of all collisional partners
    CABFP = None
    if (len(INI['cabfile'])>0): # we have a file with abundances of each collisional partner
        CABFP = open(INI['cabfile'], 'rb')
        tmp   = fromfile(CABFP, int32, 4)
        if ((tmp[0]!=X)|(tmp[1]!=Y)|(tmp[2]!=Z)|(tmp[3]!=PARTNERS)):
            print("*** ERROR: CABFILE has dimensions %d x %d x %d, for %d partners" % (tmp[0], tmp[1], tmp[2], tmp[3]))
            sys.exit()            
    MATRIX    =  zeros((LEVELS, LEVELS), float32)
    VECTOR    =  zeros(LEVELS, float32)
    COMATRIX  =  []
    ave_max_change, global_max_change = 0.0, 0.0
    
    if (INI['constant_tkin']): # Tkin same for all cells => precalculate collisional part
        print("Tkin assumed to be constant !")
        constant_tkin = True
    else:
        constant_tkin = False
    if (constant_tkin):
        if (CABFP):
            print("Cannot have variable CAB if Tkin is assumed to be constant")
        COMATRIX = zeros((LEVELS, LEVELS), float32)
        tkin     = TKIN[1]
        for iii in range(LEVELS):
            for jjj in range(LEVELS):
                if (iii==jjj):
                    COMATRIX[iii,jjj] = 0.0
                else:
                    if (PARTNERS==1):
                        gamma = MOL.C(iii,jjj,tkin,0)  # rate iii -> jjj
                    else:
                        gamma = 0.0
                        for ip in range(PARTNERS):
                            gamma += cab[ip] * MOL.C(iii, jjj, tkin, ip)
                    COMATRIX[jjj, iii] = gamma
                    
    for icell in range(CELLS):
        if (icell%(CELLS//20)==0):
            print("  solve   %7d / %7d  .... %3.0f%%" % (icell, CELLS, 100.0*icell/float(CELLS)))
        tkin, rho, chi  =  TKIN[icell], RHO[icell], ABU[icell]
        if (rho<0.1):     continue
        if (chi<1.0e-12): continue
        if (constant_tkin):
            MATRIX[:,:] = COMATRIX[:,:] * rho 
        else:
            if (CABFP):
                cab = fromfile(CABFP, float32, PARTNERS)   # abundances for current cell, cab[PARTNERS]
            for iii in range(LEVELS):
                for jjj in range(LEVELS):
                    if (iii==jjj):
                        MATRIX[iii,jjj] = 0.0
                    else:
                        if (PARTNERS==1):
                            gamma = MOL.C(iii,jjj,tkin,0)  # rate iii -> jjj
                        else:
                            gamma = 0.0
                            for ip in range(PARTNERS):
                                gamma += cab[ip] * MOL.C(iii, jjj, tkin, ip)
                        MATRIX[jjj, iii] = gamma*rho
        if (len(ESC_ARRAY)>0):
            for t in range(TRANSITIONS):
                u, l          =  MOL.T2L(t)
                MATRIX[l,u]  +=  ESC_ARRAY[icell, t] / (VOLUME*NI_ARRAY[icell,u])
        else:
            for t in range(TRANSITIONS):
                u,l           =  MOL.T2L(t)
                MATRIX[l,u]  +=  MOL.A[t]
                
        for t in range(TRANSITIONS):
            u, l           =  MOL.T2L(t)   # radiative transitions only !!
            MATRIX[u, l]  +=  SIJ_ARRAY[icell, t]               #  u <-- l
            MATRIX[l, u]  +=  SIJ_ARRAY[icell, t] / MOL.GG[t]   #  l <-- u

        for u in range(LEVELS-1): # diagonal = -sum of the column
            tmp            = sum(MATRIX[:,u])    # MATRIX[i,i] was still == 0
            MATRIX[u,u]    = -tmp
            
        MATRIX[LEVELS-1, :]   =  -MATRIX[0,0]    # replace last equation = last row
        VECTOR[:]             =   0.0
        VECTOR[LEVELS-1]      =  -(rho*chi) * MATRIX[0,0]  # ???
                    
        VECTOR  =  np.linalg.solve(MATRIX, VECTOR)        
        VECTOR  =  clip(VECTOR, NI_LIMIT, 1e99)
        VECTOR *=  rho*chi / sum(VECTOR)        
        
        if (0):
            print("CO_ARRAY")
            for j in range(LEVELS):
                for i in range(LEVELS):
                    sys.stdout.write('%9.2e ' % (MATRIX[j,i]))
                sys.stdout.write('   %9.2e\n' % (VECTOR[j]))
            print('')
            print("SIJ")
            for j in range(TRANSITIONS):
                sys.stdout.write(' %10.2e' % SIJ_ARRAY[icell,j])
            sys.stdout.write('\n')
            print("ESC")
            for j in range(TRANSITIONS):
                sys.stdout.write(' %10.2e' % ESC_ARRAY[icell,j])
            sys.stdout.write('\n')
            print("VECTOR")
            for j in range(LEVELS):
                sys.stdout.write(' %10.2e' % VECTOR[j])
            sys.stdout.write('\n')
            sys.exit()
            
        max_relative_change =  max((NI_ARRAY[icell, 0:CHECK]-VECTOR[0:CHECK])/(NI_ARRAY[icell, 0:CHECK]))        
        NI_ARRAY[icell,:]   =  VECTOR        
        ave_max_change     +=  max_relative_change
        global_max_change   =  max([global_max_change, max_relative_change])    
    # <--- for icell
    ave_max_change /= CELLS
    print("     AVE %10.3e    MAX %10.3e" % (ave_max_change, global_max_change))
    return ave_max_change
    



def SolveCL():
    """
    Solve equilibrium equations on the device. We do this is batches, perhaps 10000 cells
    at a time => could be up to GB of device memory. 
    We could reuse existing buffers:
        NI        ~  PL_buf[CELLS], RW       BATCH*LEVELS      < CELLS
        ESC, SIJ  ~  NI_buf[2*CELLS], RO     BATCH*TRANSITIONS < CELLS
    Cells 128^3, LEVELS~TRANSITIONS~20  => BATCH < 100 000 !!
    """
    global NI_buf, PL_buf, CELLS, queue, kernel_solve, PL_buf, RES_buf, VOLUME, CELLS, LEVELS, TRANSITIONS, MOL
    global RHO, TKIN, ABU
    BATCH  =  CELLS//max([LEVELS, TRANSITIONS]) # now ESC, SIJ fit in NI_buf,  NI fits in PL_buf
    BATCH  =  min([BATCH, 16384])     #  16384*100**2 = 0.6 GB
    BATCH  =  min([BATCH, CELLS])    
    # molecule basic data
    MOL_A_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.A)
    MOL_UL_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.TRANSITION)
    MOL_E_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.E)
    MOL_G_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.G)
    # TKIN
    NTKIN      =  len(MOL.TKIN[0])
    PARTNERS   =  MOL.PARTNERS
    for i in range(1, PARTNERS):
        if (len(MOL.TKIN[i])!=NTKIN):
            print("SolveCL assumes the same number of Tkin for each collisional partner!!"), sys.exit()
    MOL_TKIN = zeros((PARTNERS, NTKIN), float32)
    for i in range(PARTNERS):
        MOL_TKIN[i, :]  =  MOL.TKIN[i][:]
    MOL_TKIN_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL_TKIN)
    # CUL  -- levels are included separately for each partner... must have the same number of rows!
    #         KERNEL ASSUMES IT IS THE SAME TRANSITIONS, IN THE SAME ORDER
    NCUL  =  MOL.CUL[0].shape[0]
    if (MOL.CUL[i].shape[0]!=NCUL):
        print("SolveCL assumes the same number of C rows for each collisional partner!!"),  sys.exit()
    CUL   =  zeros((PARTNERS,NCUL,2), int32)
    for i in range(PARTNERS):
        CUL[i, :, :]  =  MOL.CUL[i][:,:]
        # KERNEL USES ONLY THE CUL ARRAY FOR THE FIRST PARTNER -- CHECK THAT TRANSITIONS ARE IN THE SAME ORDER
        delta =   np.max(ravel(MOL.CUL[i]-MOL.CUL[0]))
        if (delta>0):
            print("*** ERROR: SolveCL assumes all partners have C in the same order of transitions!!"), sys.exit()
    MOL_CUL_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=CUL)
    # C
    C    =  zeros((PARTNERS, NCUL, NTKIN), float32)
    for i in range(PARTNERS):   
        C[i, :, :]  =  MOL.CC[i][:, :]
    MOL_C_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C)
    # abundance of collisional partners
    MOL_CABU_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MOL.CABU)    
    # new buffer for matrices and the right side of the equilibriumm equations
    WRK_buf  = cl.Buffer(context, mf.READ_WRITE, 4*BATCH*LEVELS*(LEVELS+1))
    RHO_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
    TKIN_buf = cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
    ABU_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
    SIJ_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH*TRANSITIONS)
    ESC_buf  = cl.Buffer(context, mf.READ_ONLY,  4*BATCH*TRANSITIONS)    
    # NI        =>  PL_buf          BATCH*LEVELS        < CELLS        PL_buf[BATCH, LEVELS]
    # NI output =>  RES_buf         BATCH*LEVELS        < 2*CELLS      RES_buf[BATCH, LEVELS]
    #   BATCH  <  CELLS//LEVELS   and   BATCH  <  CELLS//TRANSITIONS
    CHECK    = min([INI['uppermost']+1, LEVELS])  # check this many lowest energylevels
    GLOBAL_SOLVE = IRound(BATCH, LOCAL)
    tmp = zeros((BATCH, 2, TRANSITIONS), float32)
    res = zeros((BATCH, LEVELS), float32)
    ave_max_change     = 0.0
    global_max_change  = 0.0
    for ibatch in range(CELLS//BATCH):
        a     = ibatch*BATCH
        b     = min([a+BATCH, CELLS])
        batch = b-a
        #print(" SolveCL, batch %5d,  [%7d, %7d[,  %4d cells, BATCH %4d" % (ibatch, a, b, batch, BATCH))
        # copy RHO, TKIN, ABU
        cl.enqueue_copy(queue, RHO_buf,  RHO[a:b].copy())  # without copy() "ndarray is not contiguous"
        cl.enqueue_copy(queue, TKIN_buf, TKIN[a:b].copy())
        cl.enqueue_copy(queue, ABU_buf,  ABU[a:b].copy())
        cl.enqueue_copy(queue, PL_buf,   NI_ARRAY[a:b,:].copy())   # PL[CELLS] ~ NI[BATCH, LEVELS]
        cl.enqueue_copy(queue, SIJ_buf,  SIJ_ARRAY[a:b,:].copy())
        cl.enqueue_copy(queue, ESC_buf,  ESC_ARRAY[a:b,:].copy())
        # solve
        kernel_solve(queue, [GLOBAL_SOLVE,], [LOCAL,], batch, 
        MOL_A_buf, MOL_UL_buf,  MOL_E_buf, MOL_G_buf,        PARTNERS, NTKIN, NCUL,   
        MOL_TKIN_buf, MOL_CUL_buf,  MOL_C_buf, MOL_CABU_buf,
        RHO_buf, TKIN_buf, ABU_buf,     PL_buf, SIJ_buf, ESC_buf,  RES_buf,  WRK_buf)
        cl.enqueue_copy(queue, res, RES_buf)
        # delta = for each cell, the maximum level populations change amog levels 0:CHECK
        delta             =  np.max((res[:,0:CHECK] - NI_ARRAY[a:b,0:CHECK]) / NI_ARRAY[a:b,0:CHECK], axis=1)
        global_max_change =  max([global_max_change, max(delta)])
        ave_max_change   +=  sum(delta)
        NI_ARRAY[a:b,:] = res
    ave_max_change /= CELLS
    print("      SolveCL    AVE %10.3e    MAX %10.3e" % (ave_max_change, global_max_change))
    return ave_max_change

    
    

def WriteSpectra(INI, u, l):
    global MOL, program, queue, WIDTH, LOCAL, NI_ARRAY, WRK, NI_buf, HFS, CHANNELS, HFS
    global NTRUE_buf, STAU_buf, NI_buf, CLOUD_buf, GAU_buf    
    tmp_1       =  C_LIGHT*C_LIGHT/(8.0*pi)
    tran        =  MOL.L2T(u, l)
    if (tran<0):
        print("*** ERROR:  WriteSpectra  %2d -> %2d not valid transition" % (u, l))
        return
    if (HFS):
        ncmp    =  BAND[tran].N
        nchn    =  BAND[tran].Channels()
        print("     .... WriteSpectra, tran=%d, %d->%d: %d components, %d channels" % (tran, u, l, ncmp, nchn))
    else:
        nchn    =  CHANNELS     #  it is the original INI['channels']
        ncmp    =  1
    Aul         =  MOL.A[tran]
    freq        =  MOL.F[tran]
    gg          =  MOL.G[u]/MOL.G[l]
    GNORM       =  (C_LIGHT/(1.0e5*WIDTH*freq))    #  GRID_LENGTH **NOT** multiplied in    
    int2temp    =  C_LIGHT*C_LIGHT/(2.0*BOLTZMANN*freq*freq)
    BG          =  int2temp * Planck(freq, INI['Tbg'])
    NRA, NDE    =  INI['points']
    DE          =  0.0    
    GLOBAL      =  IRound(NRA, LOCAL)    
    ##NTRUE       =  zeros(NRA*INI['channels'], float32)
    STEP        =  INI['grid'] / INI['angle']
    emissivity  =  (PLANCK/(4.0*pi))*freq*Aul*int2temp    
    direction   =  cl.cltypes.make_float2() 
    direction['x'], direction['y'] = INI['direction']

    if (HFS): # note -- GAU is for CHANNELS channels = maximum over all bands!!
        for i in range(ncmp):
            HF[i]['x']  =  round(BAND[tran].VELOCITY[i]/WIDTH) # offset in channels (from centre of the spectrum)
            HF[i]['y']  =  BAND[tran].WEIGHT[i]
            print("       offset  %5.2f km/s, weight %5.3f" % (HF[i]['x'], HF[i]['y']))
        HF[0:ncmp]['y']  /= sum(HF[0:ncmp]['y'])
        cl.enqueue_copy(queue, HF_buf, HF)
    
    kernel_spe  = program.Spectra        
    #                                 0     1     2     3           4                  5
    kernel_spe.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
    # 6         7         8           9           10          11    12   
    np.float32, np.int32, np.float32, np.float32, np.float32, None, None])
    if (HFS):
        kernel_spe_hf  = program.SpectraHF
        #                                    0     1     2     3           4                  5      
        #                                    CLOUD GAU   LIM   GN          D                  NI     
        kernel_spe_hf.set_scalar_arg_dtypes([None, None, None, np.float32, cl.cltypes.float2, None,
        #  6        7         8           9           10          11      12     
        #  DE       NRA       STEP        BG          emis        NTRUE   SUM_TAU
        np.float32, np.int32, np.float32, np.float32, np.float32, None,   None,
        # 13      14        15   
        # NCHN    NCOMP     HF   
        np.int32, np.int32, None])
        
    wrk         =  (tmp_1 * Aul * (NI_ARRAY[:,l]*gg-NI_ARRAY[:,u])) / (freq*freq)
    wrk         =  clip(wrk, 1.0e-25, 1e10)    
    WRK[:]['x'] =  NI_ARRAY[:, u]
    WRK[:]['y'] =  wrk
    wrk         =  []
    cl.enqueue_copy(queue, NI_buf, WRK)
        
    fp          =  open('%s_%s_%02d-%02d.spe' % (INI['prefix'], MOL.NAME, u, l), 'wb')
    asarray([NRA, NDE, nchn], int32).tofile(fp)
    asarray([-0.5*(nchn-1.0)*WIDTH, WIDTH], float32).tofile(fp)
    fptau       =  open('%s_%s_%02d-%02d.tau' % (INI['prefix'], MOL.NAME, u, l), 'wb')

    NTRUE       =  zeros((NRA, nchn), float32)
    ANGLE       =  INI['angle']
    ave_tau     =  0.0
    tau         =  zeros(NRA, float32)
    for de in range(NDE):
        DE      =  +(de-0.5*(NDE-1.0))*STEP
        if (HFS): # since CHANNELS has been changed, all transitions written using this kernel ???
            print("---------- kernel_spe_hf ----------")
            kernel_spe_hf(queue, [GLOBAL,], [LOCAL,],
            # 0        1        2         3     4          5       6   7    8     9   10          11         12      
            CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity, NTRUE_buf, STAU_buf,
            nchn, ncmp, HF_buf)            
        else:
            print("---------- kernel_spe ----------")
            kernel_spe(queue, [GLOBAL,], [LOCAL,],
            # 0        1        2         3     4          5       6   7    8     9   10          11         12      
            CLOUD_buf, GAU_buf, LIM_buf, GNORM, direction, NI_buf, DE, NRA, STEP, BG, emissivity, NTRUE_buf, STAU_buf)
        # save spectrum
        cl.enqueue_copy(queue, NTRUE, NTRUE_buf)
        for ra in range(NRA):
            asarray([(ra-0.5*(NRA-1.0))*ANGLE, (de-0.5*(NDE-1.0))*ANGLE], float32).tofile(fp) # offsets
            NTRUE[ra,:].tofile(fp)       # spectrum
        # save optical depth
        cl.enqueue_copy(queue, NTRUE, STAU_buf)
        for ra in range(NRA):
            tau[ra]  =  np.max(NTRUE[ra,:])
        ave_tau +=  sum(tau)   # sum of the peak tau values of the individual spectra
        tau.tofile(fptau)      # file containing peak tau for each spectrum
    fp.close()
    fptau.close()
    print("  SPECTRUM %3d  = %2d -> %2d,  <tau_peak> = %.3e" % (tran, u, l, ave_tau/(NRA*NDE)))
    
    

    
    
    
#================================================================================
#================================================================================
#================================================================================
#================================================================================

    
    
        
# Main loop -- simulation and updates to level populations
max_change, Tsin, Tsol, Tsav = 0.0, 0.0, 0.0, 0.0
print("================================================================================")
for ITER in range(INI['iterations']):
    print('   ITERATION %d/%d' % (1+ITER, INI['iterations']))
    t0    =  time.time()
    Simulate()
    Tsim  =  time.time()-t0
    t0    =  time.time()
    if (0):
        ave_max_change = Solve()
    else:
        ave_max_change = SolveCL()
    Tsol  =  time.time()-t0
    t0    =  time.time()
    if (((ITER%4==3)|(ITER==(INI['iterations']-1))) & (len(INI['save'])>0)): # save level populations
        print("      ... save level populations") 
        fp = open(INI['save'], 'wb')
        asarray([X, Y, Z, LEVELS], int32).tofile(fp)
        asarray(NI_ARRAY, float32).tofile(fp)
        fp.close()
    Tsave = time.time()-t0
    print("      SIMULATION %7.2f    SOLVE %7.2f    SAVE %7.2f" % (Tsim, Tsol, Tsav))
    if (ave_max_change<INI['stop']):  break
print("================================================================================")
if ((INI['iterations']>0)&(COOLING)):
    Cooling()        

    
# Save Tex files
ul = INI['Tex']                # upper and lower level for each transition
## print(ul)
for i in range(len(ul)//2):    # loop over transitions
    u, l  =  ul[2*i], ul[2*i+1]
    tr    =  MOL.L2T(u,l)
    if (tr<0):
        print("*** Error:  Tex %2d -> %2d  is not a valid transition" % (u, l))
        continue
    gg    =  MOL.G[u]/MOL.G[l]
    fp    =  open('%s_%s_%02d-%02d.tex' % (INI['prefix'], MOL.NAME, u, l), 'wb')
    asarray([X, Y, Z, LEVELS], int32).tofile(fp)
    tex   =  BOLTZMANN * log(NI_ARRAY[:, l]*gg/NI_ARRAY[:, u])
    m     =  nonzero(abs(tex)>1.0e-35)
    tex   =  PLANCK * MOL.F[tr] / tex
    asarray(tex, float32).tofile(fp)
    fp.close()
    print("  TEX      %3d  = %2d -> %2d,  %.3f K" % (tr, u, l, mean(tex)))
        

# Save spectra
ul = INI['spectra']
for i in range(len(ul)//2):
    WriteSpectra(INI, ul[2*i], ul[2*i+1])

print("================================================================================")


    
    
    
