import numpy as np
import matplotlib.pylab as plt

# Routines that can be used to read spectrum and excitation temperature 
# files written by LOC1D.py or LOC_OT.py

def LOC_read_spectra_1D(filename):
    """
    Read spectra written by LOC1D.py (spherical models).
    Usage:
        V, S = LOC_read_spectra_1D(filename)
    Input:
        filename = name of the spectrum file
    Return:
        V  = vector of velocity values, one per channeö
        S  = spectra as a cube S[NRAY, NCHN] for NRAY lines of sight and
             NCHN spectral channels
    """
    fp = open(filename, 'rb')
    NRAY, NCHN      =  np.fromfile(fp, np.int32, 2)
    V0, DV          =  np.fromfile(fp, np.float32, 2)
    SPE             =  np.fromfile(fp, np.float32, NRAY*NCHN).reshape(NRAY,NCHN)
    fp.close()
    return V0+np.arange(NCHN)*DV, SPE


def LOC_read_Tex_1D(filename):
    """
    Read excitation temperatures written by LOC1D.py.
    Usage:
        TEX = LOC_read_Tex_1D(filename)
    Input:
        filename = name of the Tex file written by LOC1D.py
    Output:
        TEX = Vector of Tex values [K], one per shell, starting
              with the innermost shell.
    """
    fp    =  open(filename, 'rb')
    CELLS =  np.fromfile(fp, np.int32, 1)[0]
    TEX   =  np.fromfile(fp, np.float32, CELLS)
    fp.close()
    return TEX


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
    NRA, NDE, NCHN  =  np.fromfile(fp, np.int32, 3)
    V0, DV          =  np.fromfile(fp, np.float32, 2)
    SPE             =  np.fromfile(fp, np.float32).reshape(NDE, NRA, 2+NCHN)
    OFF             =  SPE[:,:,0:2].copy()
    SPE             =  SPE[:,:,2:]
    fp.close()
    return V0+np.arange(NCHN)*DV, SPE


def LOC_read_Tex_3D(filename): 
    """
    Read excitation temperatures written by LOC.py (LOC_OT.py).
    Usage:
        TEX = LOC_read_Tex_3D(filename)
    Input:
        filename = name of the Tex file written by LOC1D.py
    Output:
        TEX = Vector of Tex values [K], one per cell.
    Note:
        In case of octree grids, the returned vector must be
        compared to hierarchy information (e.g. from the density file)
        to know the locations of the cells.
        See the routine OT_GetCoordinatesAllV()
    """
    fp    =  open(filename, 'rb')
    NX, NY, NZ, dummy  =  np.fromfile(fp, np.int32, 4)
    TEX                =  np.fromfile(fp, np.float32).reshape(NZ, NY, NX)
    fp.close()
    return TEX
                                        

if (__name__=='__main__'):
    V, SPE = LOC_read_spectra_1D('loc_1d_py_dust.CO_02-01.spe')
    plt.plot(V, SPE[0, :], 'k-', label='Centre spectrum')
    plt.xlabel(r'$V \/ \/ (km/s)$')
    plt.ylabel(r'$T_A \/ \/ (K)$')
    plt.legend()
    plt.show()

    
