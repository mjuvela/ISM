import os, sys
import numpy as np
from   matplotlib.pylab import *
import astropy.io.fits as pyfits

FITS = 1

def OT_WriteHierarchyV_LOC(NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ, CHI, filename):
    """
    Usage:
        OT_WriteHierarchyV_LOC(NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ, CHI, filename)
        Write LOC octtree structure to disk.
    """
    LEVELS = len(LCELLS)
    CELLS  = sum(LCELLS)
    fp      =  open(filename, 'wb')
    np.asarray([NX, NY, NZ, LEVELS, CELLS], np.int32).tofile(fp)
    for X in [ H, T, S, VX, VY, VZ, CHI]:
        for i in range(LEVELS):
            np.asarray([LCELLS[i],], np.int32).tofile(fp)
            X[OFF[i]:(OFF[i]+LCELLS[i])].tofile(fp)
    return

def LOC_read_spectra_3D(filename):
    """
    Read spectra written by LOC.py (LOC_OT.py; 3D models)
    Usage:
        V, S = LOC_read_spectra_3D(filename)
    Input:
        filename = name of the spectrum file
    Return:
        V  = vector of velocity values, one per channeö
        S  = spectra as a cube S[NRA, NDE, NCHN] for NRAY lines of sight and
        NRA times NDE points on the sky
    """
    fp              =  open(filename, 'rb')
    NRA, NDE, NCHN  =  fromfile(fp, np.int32, 3)
    V0, DV          =  fromfile(fp, np.float32, 2)
    SPE             =  fromfile(fp, np.float32).reshape(NDE, NRA, 2+NCHN)
    OFF             =  SPE[:,:,0:2].copy()
    SPE             =  SPE[:,:,2:]
    fp.close()
    return V0+arange(NCHN)*DV, SPE


INI = """
octree4        velo.cloud
distance       10.0
angle          1.0
molecule       co.dat
density        1.0e3
temperature    15.0
fraction       1.0e-4
velocity       1.0
sigma          1.0
isotropic      2.73
levels         10
uppermost      3
iterations     0
nside          2
direction      90 90
points         32 30
grid           1.0
spectra        1 0  2 1
transitions    1 0  2 1
bandwidth      7.0
channels       128
prefix         velo
stop          -1.0
lowmem         0
gpu            1
"""

# choose a velocity for each direction
VELO = { 'x' : -1.0, 'y' : 0.5, 'z' : 1.0 }

# make cloud file
NX, NY, NZ  =  26, 28, 30      # c-order == LOC program "X", "Y", and "Z" dimensions
LEVELS  = 1
LCELLS  = np.asarray([NX*NY*NZ, ], np.int32)
OFF     = np.asarray([0,], np.int32)
H       = ones((NZ, NY, NX), np.float32) 
K, J, I = indices((NZ, NY, NX), np.float32)
K      -= 0.5*(NZ-1.0)
J      -= 0.5*(NY-1.0)
I      -= 0.5*(NX-1.0)
H       = exp((-K*K-J*J-I*I)*0.02*0.1)   # some density field...
# indicate directios by adding blobs in the (0,0,1), (0,1,0), and (0,0,1) corners
H[(NZ-3):,  (NY-3):,   :        ]  =  5.0   #   along X axis
H[(NZ-3):,   :     ,  (NX-3):   ]  = 10.0   #   along Y axis
H[:      ,  (NY-3):,  (NX-3):   ]  = 15.0   #   along Z direction, high Y and Z
T       = ones((NZ, NY, NX), np.float32)
S       = ones((NZ, NY, NX), np.float32)
VX      = ones((NZ, NY, NX), np.float32) * VELO['x']  # same velocity for all cells
VY      = ones((NZ, NY, NX), np.float32) * VELO['y']
VZ      = ones((NZ, NY, NX), np.float32) * VELO['z']
CHI     = ones((NZ, NY, NX), np.float32)
OT_WriteHierarchyV_LOC(NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ, CHI, 'velo.cloud')

# run LOC and plot maps of line area and radial velocity for each direction
# == make sure directions are correct and radial velocities are correct
figure(1, figsize=(9, 5.5))
subplots_adjust(left=0.08, right=0.96, bottom=0.09, top=0.95, wspace=0.42, hspace=0.35)
rc('font', size=9)


for DIR in ['x', 'y', 'z']:
    fp = open('velo.ini', 'w')
    fp.write(INI)
    if (DIR=='x'):
        CMD = 'direction 90.0 0.0'
    elif (DIR=='y'):
        CMD = 'direction 90.0 90.0'
    else:
        CMD = 'direction 0.0 0.0'
    fp.write('%s\n' % CMD)    
    fp.write('points  30 34 \n')
    if (FITS): fp.write('FITS 1\n')
    fp.close()    
    os.system('LOC_OT.py velo.ini')
    ##
    if (FITS):
        os.system('cp velo_CO_01-00.fits %s.fits' % DIR)
        F     =  pyfits.open('velo_CO_01-00.fits')
        NX    =  F[0].header['NAXIS1']
        NY    =  F[0].header['NAXIS2']
        nchn  =  F[0].header['NAXIS3']
        V     =  (arange(nchn)-F[0].header['CRPIX3'])*F[0].header['CDELT3']
        S     =  zeros((NY, NX, nchn), float32)
        for i in range(nchn): S[:,:,i] = F[0].data[i,:,:]
    else:
        V, S = LOC_read_spectra_3D('velo_CO_01-00.spe')
    NY, NX, NCHN = S.shape
    W    =  sum(S, axis=2)*(V[1]-V[0])   # line area
    VA   =  zeros((NY, NX), np.float32)  # average LOS velocity from the spectra
    for j in range(NY):
        for i in range(NX):
            VA[j,i] =  sum(S[j,i,:]*V) / sum(S[j,i,:])    
    # upper row = line area
    ax   =  subplot(2, 3, {'x':1, 'y':2, 'z':3}[DIR])
    imshow(W)
    title('W(%s)' % DIR)
    colorbar()
    if (DIR=='x'):
        xlabel(r'$Y$', size=12),  ylabel(r'$Z$', size=12)
    elif (DIR=='y'):
        xlabel(r'$-X$', size=12), ylabel(r'$Z$', size=12)
    else:
        xlabel(r'$Y$', size=12),  ylabel(r'$-X$', size=12)
    # lower row = LOS velocity from the spectra
    ax   =  subplot(2, 3, {'x':4, 'y':5, 'z':6}[DIR])
    imshow(VA, vmin=-1.1, vmax=+1.1)
    text(0.1, 0.9, CMD, size=9, transform=ax.transAxes)
    colorbar()
    title('%s: v(%s) = %.1f' % (DIR, DIR, VELO[DIR]))
    if (DIR=='x'):
        xlabel(r'$Y$', size=12),  ylabel(r'$Z$', size=12)
    elif (DIR=='y'):
        xlabel(r'$-X$', size=12), ylabel(r'$Z$', size=12)
    else:
        xlabel(r'$Y$', size=12),  ylabel(r'$-X$', size=12)
    
# savefig('vel_xyz.png')

show()

    
        
        
        
