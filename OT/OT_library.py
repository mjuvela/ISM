
import pyopencl as cl
import ctypes
import numpy as np

# import pynbody
import yt
import time

OLD_ORIENTATION = True


def I2F(i):
    return ctypes.c_float.from_buffer(ctypes.c_int(i)).value
    
def F2I(x):
    return ctypes.c_int.from_buffer(ctypes.c_float(x)).value


def KernelSource(sname):
    # Try to read kernel source file "sname" from a couple of places
    src = ""
    try:  #  directory specified with the environmental variable KERNEL_DIRECTORY
        src = open("%s/%s" % (os.environ["KERNEL_DIRECTORY"], sname)).read()
    except:
        # print("Environmental variable KERNEL_DIRECTORY not set or directory not containing kernel %s" % sname)
        src = ""
    if (len(src)<1):
        try:  # read from default location "~/starformation/Python/MJ/MJ/Aux/"
            src =  open(HOMEDIR+"/starformation/Python/MJ/MJ/Aux/%s" % sname).read()
        except:
            print("OT_library.py => kernel file %s not found" % sname)
            print("  ... set environmental variable KERNEL_DIRECTORY to point to the directory ")
            print("      containing the kernel source file %s" % sname)
            pass
    return src

            
if (1):
    from MJ.Aux.mjGPU import *
else:
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
    
    
def ReadCloudLOC1D(name):
    fp = open(name, 'rb')
    cells = fromfile(fp, np.int32, 1)[0]
    R     = fromfile(fp, np.float32, cells)
    tmp   = fromfile(fp, np.float32, cells*6).reshape(cells, 5) #  n, T, s, chi, vrad
    fp.close()
    return R, tmp
    

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
    NRAY, NCHN      =  fromfile(fp, np.int32, 2)
    V0, DV          =  fromfile(fp, np.float32, 2)
    SPE             =  fromfile(fp, np.float32, NRAY*NCHN).reshape(NRAY,NCHN)
    fp.close()
    return V0+arange(NCHN)*DV, SPE


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
    CELLS =  fromfile(fp, np.int32, 1)[0]
    TEX   =  fromfile(fp, np.float32, CELLS)
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
    NRA, NDE, NCHN  =  fromfile(fp, np.int32, 3)
    V0, DV          =  fromfile(fp, np.float32, 2)
    SPE             =  fromfile(fp, np.float32).reshape(NDE, NRA, 2+NCHN)
    OFF             =  SPE[:,:,0:2].copy()
    SPE             =  SPE[:,:,2:]
    fp.close()
    return V0+arange(NCHN)*DV, SPE


def LOC_write_spectra_3D(filename, V, SPE): 
    """
    Write spectra to a file in the original LOC format (3D models)
    Usage:
        LOC_write_spectra_3D(filename, V, SPE)
    Input:
        filename = name of the written spectrum file
        V        = vector of channel velocities
        SPE      = spectrum cube with dimensions S[N,M,NCHN]
    """
    NDE, NRA, NCHN  =  SPE.shape
    Y, X            =  indices((NDE, NRA), np.float32)
    V0, DV          =  V[0], V[1]-V[0]
    ###
    fp              =  open(filename, 'wb')
    np.asarray([NRA, NDE, NCHN], np.int32).tofile(fp)
    np.asarray([V0, DV], np.float32).tofile(fp)
    SSPE            =  np.zeros((NDE, NRA, 2+NCHN), np.float32)
    SSPE[:,:,0]     =  Y
    SSPE[:,:,1]     =  X
    SSPE[:,:,2:]    =  SPE
    np.asarray(SSPE, np.float32).tofile(fp)
    fp.close()
    return



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
    NX, NY, NZ, dummy  =  fromfile(fp, np.int32, 4)
    TEX                =  fromfile(fp, np.float32).reshape(NZ, NY, NX)
    fp.close()
    return TEX
                                        


# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV


    
    
def OT_ReadHierarchyV(filename):
    """
    Usage:
        NX, NY, NZ, LCELLS, OFF, H = OT_ReadHierarchyV(filename)
    Read octtree structure from disk, data and links.
    *** H is a single vector ***
    Input:
        name of the input file
    Returns:
        NX, NY, NZ  = dimensions of the root grid
        LCELLS      = vector, number of cells in each level
        OFF         = offsets to the first cell on each hierarchy level
        H           = the hierarchy structure *** single vector ***
    """
    # print('ReadHierarchy ', filename)
    fp = open(filename, 'r')
    nx, ny, nz, levels, cells = fromfile(fp, np.int32, 5)
    # print('    OT_ReadHierarchy: ', nx, ny, nz, levels, cells)
    LCELLS = np.zeros(levels, np.int32)
    OFF    = np.zeros(levels, np.int32)
    H      = np.zeros(cells,  np.float32)
    for i in range(levels):
        tmp       = fromfile(fp, np.int32, 1)[0]
        LCELLS[i] = tmp
        H[OFF[i]:(OFF[i]+LCELLS[i])] =  fromfile(fp, np.float32, LCELLS[i])
        if (i<levels-1):
            OFF[i+1] = OFF[i] + LCELLS[i]
    print("LCELLS ", LCELLS)
    print("LCELLS %d, CELLS %d, LEVELS %d" % (sum(LCELLS), cells, levels))
    return nx, ny, nz, LCELLS, OFF, H


def OT_ReadHierarchyV_LOC(filename):
    """
    Usage:
        NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ, CHI = OT_ReadHierarchyV_LOC(filename)
    Read LOC octtree structure from disk, data and links.
    Input:
        name of the input file
    Returns:
        NX, NY, NZ  = dimensions of the root grid
        LCELLS      = vector, number of cells in each level
        OFF         = offsets to the first cell on each hierarchy level
        H, T, ....  = the hierarchy structure, a single vector for each parameter
    """
    fp      =  open(filename, 'rb')
    NX, NY, NZ, LEVELS, CELLS = fromfile(fp, np.int32, 5)
    LCELLS  =  np.zeros(LEVELS, np.int32)
    OFF     =  np.zeros(LEVELS, np.int32)
    H       =  np.zeros(CELLS,  np.float32)
    T       =  np.zeros(CELLS,  np.float32)
    S       =  np.zeros(CELLS,  np.float32)
    VX      =  np.zeros(CELLS,  np.float32)
    VY      =  np.zeros(CELLS,  np.float32)
    VZ      =  np.zeros(CELLS,  np.float32)
    CHI     =  np.zeros(CELLS,  np.float32)
    for X in [ H, T, S, VX, VY, VZ, CHI]:
        for i in range(LEVELS):
            cells       =  fromfile(fp, np.int32, 1)
            # print('cells = ', cells)
            cells = cells[0]
            if (i>0):  OFF[i] = OFF[i-1] + LCELLS[i-1]
            LCELLS[i]   =  cells
            X[OFF[i]:(OFF[i]+LCELLS[i])] =  fromfile(fp, np.float32, LCELLS[i])
    return NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ, CHI



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
    print("OT_WriteHierarchyV_LOC: %d levels, %d = %.3e cells" % (LEVELS, CELLS, CELLS))
    for X in [ H, T, S, VX, VY, VZ, CHI]:
        for i in range(LEVELS):
            np.asarray([LCELLS[i],], np.int32).tofile(fp)
            X[OFF[i]:(OFF[i]+LCELLS[i])].tofile(fp)
    return



def OT_WriteHierarchyV(NX, NY, NZ, LCELLS, OFF, H, filename):
    """
    Write hierarchy to file
    Usage:
        OT_WriteHierarchyV(NX, NY, NZ, LCELLS, OFF, H, filename)
    Input:
        NX, NY, NZ   =  dimensions of the root grid
        LCELLS       =  number of cells on each hierarchy level
        OFF          =  offsets for the first cell on each hierarchy level
        H            =  the hierarchy, *** a single vector ***
        filename     =  name of the output file
    """
    fp = open(filename, 'wb')
    np.asarray([NX, NY, NZ, len(LCELLS), sum(LCELLS)], np.int32).tofile(fp)
    levels  = len(LCELLS)
    cells   = sum(LCELLS)
    print("OT_WriteHierarchyV_LOC: %d levels, %d = %.3e cells" % (levels, cells, cells))
    for i in range(levels):
        print("    level %2d   lcells %9d = %.3e" % (i, LCELLS[i], LCELLS[i]))
        np.asarray([LCELLS[i],], np.int32).tofile(fp)
        np.asarray( H[  (OFF[i]) : (OFF[i]+LCELLS[i])  ], np.float32).tofile(fp)
    fp.close()

    
    

def OT_GetCoordinatesAllV(NX, NY, NZ, LCELLS, OFF, H, GPU=0, platforms=[0,1,2,3,4], verbose=False):
    """
    Return coordinates for all cells.
    Usage:
        x, y, z      =  OT_GetCoordinatesAllV(NX, NY, NZ, LCELLS, OFF, H, GPU=0, platforms=[0,1,2,3,4])
    Input:
        NX, NY, NZ   =  dimensions of the root grid
        LCELLS       =  number of cells on each hierarchy level
        OFF          =  offsets to the first cell on each hierarchy level
        H            =  the hierarchy *** single vector ***
    Return:
        x, y, z      =  coordinates of all cells in root grid coordinates
    """
    print("OT_GetCoordinatesAllV OFF", OFF, "LCELLS ", LCELLS) 
    # LEVELS     =  len(LCELLS)
    print(".... LCELLS = ", LCELLS)
    LEVELS     =  len(nonzero(LCELLS>0)[0])  # could be [ 999, 0, 0, 0, ...]
    print(".... LEVELS = ", LEVELS)
    CELLS      =  sum(LCELLS)
    N          =  CELLS
    # print('OT_GetCoordinatesAllV() -- LEVELS %d, CELLS %d, LCELLS' % (LEVELS, CELLS), LCELLS)
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=verbose)
    GLOBAL     =  8192
    LOCAL      =  [ 8, 32 ][GPU>0]
    OPT        =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, N, NX, NY, NZ)
    source     =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program    =  cl.Program(context, source).build(OPT)
    # Use kernel Parents to find the parents
    print('Parents...')
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(OFF, np.int32))
    H_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
    PAR_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS)) # lives on device
    PAR        =  np.zeros(CELLS, np.int32)
    cl.enqueue_copy(queue, PAR_buf, PAR)
    Parents    =  program.ParentsV
    Parents.set_scalar_arg_dtypes([None,None,None,None])
    Parents(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf, PAR_buf)
    cl.enqueue_copy(queue, PAR, PAR_buf)
    # Get coordinates for each cell
    print('Coordinates...')
    X_buf      =  cl.Buffer(context, mf.WRITE_ONLY, CELLS*4)
    Y_buf      =  cl.Buffer(context, mf.WRITE_ONLY, CELLS*4)
    Z_buf      =  cl.Buffer(context, mf.WRITE_ONLY, CELLS*4)
    I2C        =  program.Ind2CooAllV
    I2C.set_scalar_arg_dtypes([None,None,None,None,None,None])
    I2C(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, PAR_buf, X_buf, Y_buf, Z_buf)
    X, Y, Z    =  np.zeros(CELLS, np.float32), np.zeros(CELLS, np.float32), np.zeros(CELLS, np.float32),
    cl.enqueue_copy(queue, X, X_buf)
    cl.enqueue_copy(queue, Y, Y_buf)
    cl.enqueue_copy(queue, Z, Z_buf)
    print('OT_GetCoordinatesAllV .... ready --- CELLS WAS %d !' % CELLS)
    return X, Y, Z



def OT_GetCoordinatesAndLevels(NX, NY, NZ, LCELLS, OFF, H, GPU=0, platforms=[0,1,2,3,4]):
    """
    Return coordinates and hierarchy level for all leaves in the hierarchy.
    Usage:
        x, y, z, l   =  OT_GetCoordinatesAndLevels(NX, NY, NZ, LCELLS, OFF, H, GPU=0, platforms=[0,1,2,3,4])
    Input:
        NX, NY, NZ   =  dimensions of the root grid
        LCELLS       =  number of cells on each hierarchy level
        OFF          =  offsets to the first cell on each hierarchy level
        H            =  the hierarchy *** single vector ***
    Return:
        x, y, z      =  coordinate vectors for all leaves in root grid coordinates
        l            =  hierarchy level (>=0) for each cell
    """
    # print("OT_GetCoordinatesAllV OFF", OFF, "LCELLS ", LCELLS) 
    # LEVELS     =  len(LCELLS)
    LEVELS     =  len(nonzero(LCELLS>0)[0])  # could be [ 999, 0, 0, 0, ...]
    CELLS      =  sum(LCELLS)
    N          =  CELLS
    # print('OT_GetCoordinatesAllV() -- LEVELS %d, CELLS %d, LCELLS' % (LEVELS, CELLS), LCELLS)
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    GLOBAL     =  8192
    LOCAL      =  [ 8, 32 ][GPU>0]
    OPT        =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, N, NX, NY, NZ)
    source     =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program    =  cl.Program(context, source).build(OPT)
    # Use kernel Parents to find the parents
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(OFF, np.int32))
    H_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
    PAR_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS)) # lives on device
    PAR        =  np.zeros(CELLS, np.int32)
    cl.enqueue_copy(queue, PAR_buf, PAR)
    Parents    =  program.ParentsV
    Parents.set_scalar_arg_dtypes([None,None,None,None])
    Parents(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf, PAR_buf)
    cl.enqueue_copy(queue, PAR, PAR_buf)
    # Get coordinates for each cell
    X_buf      =  cl.Buffer(context, mf.WRITE_ONLY, CELLS*4)
    Y_buf      =  cl.Buffer(context, mf.WRITE_ONLY, CELLS*4)
    Z_buf      =  cl.Buffer(context, mf.WRITE_ONLY, CELLS*4)
    L_buf      =  cl.Buffer(context, mf.WRITE_ONLY, CELLS*4)
    I2C        =  program.Ind2CooAndLevel
    I2C.set_scalar_arg_dtypes([     None,       None,    None,    None,  None,  None,  None])
    I2C(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, PAR_buf, X_buf, Y_buf, Z_buf, L_buf)
    X, Y, Z    =  np.zeros(CELLS, np.float32), np.zeros(CELLS, np.float32), np.zeros(CELLS, np.float32),
    L          =  np.zeros(CELLS, np.int32)
    cl.enqueue_copy(queue, X, X_buf)
    cl.enqueue_copy(queue, Y, Y_buf)
    cl.enqueue_copy(queue, Z, Z_buf)
    cl.enqueue_copy(queue, L, L_buf)
    # print('OT_GetCoordinatesAllV .... ready --- CELLS WAS %d !' % CELLS)
    return X, Y, Z, L



def OT_GetCoordinatesLevelV(NX, NY, NZ, LCELLS, OFF, H, L, GPU=0, platforms=[0,1,2,3,4]):
    """
    Return coordinates for all cells on the level L.
    Usage:
        x, y, z      =  OT_GetCoordinatesLevelV(NX, NY, NZ, LCELLS, OFF, H, L, GPU=0, platforms=[0,1,2,3,4])
    Input:
        NX, NY, NZ   =  dimensions of the root grid
        LCELLS       =  number of cells on each hierarchy level
        OFF          =  offsets to the first cell on each hierarchy level
        H            =  the hierarchy *** single vector ***
        L            =  index of the hierarchy level (>=0)
    Return:
        x, y, z      =  coordinates of all level L cells in root grid coordinates
    """
    LEVELS     =  len(nonzero(LCELLS>0)[0])  # could be [ 999, 0, 0, 0, ...]
    CELLS      =  sum(LCELLS)
    N          =  CELLS
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms)
    GLOBAL     =  8192
    LOCAL      =  [ 8, 32 ][GPU>0]
    OPT        =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, N, NX, NY, NZ)
    source     =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program    =  cl.Program(context, source).build(OPT)
    # Use kernel Parents to find the parents
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=LCELLS)
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=OFF)
    H_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
    PAR_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS)) # lives on device
    PAR        =  np.zeros(CELLS, np.int32)
    cl.enqueue_copy(queue, PAR_buf, PAR)
    Parents    =  program.ParentsV
    Parents.set_scalar_arg_dtypes([None,None,None,None])
    Parents(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf, PAR_buf)
    cl.enqueue_copy(queue, PAR, PAR_buf)
    # Get coordinates for each cell on level L
    X_buf      =  cl.Buffer(context, mf.WRITE_ONLY, LCELLS[L]*4)
    Y_buf      =  cl.Buffer(context, mf.WRITE_ONLY, LCELLS[L]*4)
    Z_buf      =  cl.Buffer(context, mf.WRITE_ONLY, LCELLS[L]*4)
    I2C        =  program.Ind2CooLevelV
    I2C.set_scalar_arg_dtypes([None,None,None,None,None,None,np.int32])
    I2C(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, PAR_buf, X_buf, Y_buf, Z_buf, L)
    X, Y, Z    =  np.zeros(LCELLS[L], np.float32), np.zeros(LCELLS[L], np.float32), np.zeros(LCELLS[L], np.float32),
    cl.enqueue_copy(queue, X, X_buf)
    cl.enqueue_copy(queue, Y, Y_buf)
    cl.enqueue_copy(queue, Z, Z_buf)
    return X, Y, Z



def OT_GetValueV(x, y, z, NX, NY, NZ, LCELLS, OFF, H):
    """
    Return value for global coordinates (x,y,z) from octtree structure
    Usage:
        val = OT_GetValueV(x, y, z, NX, NY, NZ, LCELLS, OFF, H)
    Input:
        x, y, z    --  global coodinates in the root grid units
        NX, NY, NZ --  dimensions of the root grid
        LCELLS     --  number of cells on each hierarchy level
        OFF        --  offsets to the first cell on each hierarchy level
        H          --  the hierarchy with data and links, *** single vector ***

    """
    LL = []
    # must read negative values of H, change sign, interpret as integers
    levels = len(LCELLS)
    for i in range(levels):
        if (LCELLS[i]<1):
            LL.append([])
            continue
        ll  =  H[OFF[i]:(OFF[i]+LCELLS[i])].copy()
        ll  = -ll           # links positive floats
        np.asarray(ll, np.float32).tofile('/dev/shm/tmp.bin')
        ll  =  fromfile('/dev/shm/tmp.bin', np.int32)  # positive indices
        LL.append(ll)
        m   =  nonzero(H[i]<1e-10)
    # Root cell
    i, j, k = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))      # root grid indices
    if ((i<0)|(i>=NX)|(j<0)|(j>=NY)|(k<0)|(k>=NZ)): return 0.0 # outside the cloud
    # print(k*NX*NY+j*NX+i, len(H[0]))
    if (H[0][k*NX*NY+j*NX+i]>0.0):   # a leaf at root level
        return H[0][k*NX*NY+j*NX+i]
    # Otherwise follow hierarchy to the bottom
    # Move to next level coordinates before the loop
    ind   =  LL[0][k*NX*NY+j*NX+i]
    level =  1
    x     =  2.0*(x-i)  # [0,2] within the next level octet
    y     =  2.0*(y-j)
    z     =  2.0*(z-k)
    while(1):
        # (x,y,z) in which cell of this next level octet?
        i, j, k = 0, 0, 0
        if (x>1.0): i = 1
        if (y>1.0): j = 1
        if (z>1.0): k = 1
        sid   =  4*k +2*j + i
        ind  +=  sid    # index of the current level cell
        if (H[level][ind]>0.0):
            return H[level][ind]  # found a leaf
        # else we follow link to next level
        ind    =      LL[level][ind]            
        level +=  1
        # coordinates in the next level octet
        x      =  2.0*(x-i)
        y      =  2.0*(y-j)
        z      =  2.0*(z-k)
    return 0.0




def OT_GetIndicesV(x, y, z, NX, NY, NZ, LCELLS, OFF, H, GPU=1, max_level=99, global_index=False, platforms=[0,1,2,3,4]):
    """
    Use OpenCL kernel to get indices { level, index } or  {global index} for input coordinates.
    Usage:
        IND = OT_GetIndicesV(x, y, z, NX, NY, NZ, LCELLS, OFF, H, GPU=0, global_index=False, platforms=[0,1,2,3,4])
    Input:
        x, y, z    =  vectors of positions in root grid units
        NX, NY, NZ =  dimensions of the cloud itself
        LCELLS     =  cells on each level of hierarchy
        OFF        =  offset for the first cell on each level of hierarchy
        H          =  vector containing the hierarchy data (a single vector)
        GPU        =  if >0, try to use GPU instead of CPU
        max_level  =  return value from a cell at level <=max_level
                      (if index is used to read values from a file that contains all values, no hierarchy)
        global_index = if True, return asingle index to a global data vector
    Return:
        if (global_index==False):
            levels   =  vector of levels, [len(x)]
            indices  =  vector of cell indices within a level, [len(x)]
        if (global_index==True):
            indices  =  vector of global cell indices [len(x)]
    """
    if ((type(x[0])!=float32)|(type(x[0])!=float32)|(type(x[0])!=float32)|(type(H[0])!=float32)):
        print("OT_GetIndicesV() -- parameters must be vectors of type numpy.float32"), sys.exit()
    if ((type(LCELLS[0])!=np.int32)|(type(OFF[0])!=np.int32)):
        print("OT_GetIndicesV() -- LCELLS and OFF must be numpy.int32 arrays"), sys.exit()
    LEVELS   =  len(LCELLS)
    CELLS    =  sum(LCELLS)
    N        =  size(x)
    ###
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    LOCAL    =  [ 8, 32 ][GPU>0]
    GLOBAL   =  8192
    OPT      =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, N, NX, NY, NZ)
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program  =  cl.Program(context, source).build(OPT)
    #
    if (1):
        print("OT_GetIndicesV ... H has %d = %.2e cells" % (len(H), len(H)))
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(OFF, np.int32))
    H_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
    X_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    Y_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
    Z_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z)
    AIND_buf   =  cl.Buffer(context, mf.WRITE_ONLY, N*4)
    if (global_index==False):
        ALEV_buf   =  cl.Buffer(context, mf.WRITE_ONLY, N*4)  # N = number of (x,y,z) points
        Coo2IndV   = program.Coo2IndV
        Coo2IndV.set_scalar_arg_dtypes([None,None,None,None,None,None,None,None])
        Coo2IndV(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf,
        X_buf, Y_buf, Z_buf,   ALEV_buf, AIND_buf)
        ALEV, AIND = np.zeros(N, np.int32), np.zeros(N, np.int32)
        cl.enqueue_copy(queue, ALEV, ALEV_buf)
        cl.enqueue_copy(queue, AIND, AIND_buf)
        return ALEV, AIND
    else:
        Coo2IndV  =  program.Coo2IndVV
        Coo2IndV.set_scalar_arg_dtypes([     None,       None,    None,  None,  None,  None,  None,     np.int32])
        Coo2IndV(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf, X_buf, Y_buf, Z_buf, AIND_buf, max_level)
        queue.finish()
        IND       =  np.zeros(N, np.int32)
        cl.enqueue_copy(queue, IND, AIND_buf)
        return IND
    

    
    
    
def OT_GetIndicesForLevels(x, y, z, l, NX, NY, NZ, LCELLS, OFF, H, GPU=1, platforms=[0,1,2,3,4]):
    """
    Use OpenCL kernel to get indices {global index} for input coordinates.
    Usage:
        IND = OT_GetIndicesForLevels(x, y, z, l, NX, NY, NZ, LCELLS, OFF, H, GPU=0, platforms=[0,1,2,3,4])
    Input:
        x, y, z    =  vectors of positions in root grid units
        l          =  hierarchy level, return the cell that corresponds to (x, y, z) at level l
        NX, NY, NZ =  dimensions of the cloud itself
        LCELLS     =  cells on each level of hierarchy
        OFF        =  offset for the first cell on each level of hierarchy
        H          =  vector containing the hierarchy data (a single vector)
        GPU        =  if >0, try to use GPU instead of CPU
    Return:
        indices  =  vector of global cell indices [len(x)]
    """
    if ((type(x[0])!=float32)|(type(x[0])!=float32)|(type(x[0])!=float32)|(type(H[0])!=float32)):
        print("OT_GetIndicesV() -- parameters must be vectors of type numpy.float32"), sys.exit()
    if ((type(LCELLS[0])!=np.int32)|(type(OFF[0])!=np.int32)):
        print("OT_GetIndicesV() -- LCELLS and OFF must be numpy.int32 arrays"), sys.exit()
    LEVELS   =  len(LCELLS)
    CELLS    =  sum(LCELLS)
    N        =  size(x)
    ###
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    LOCAL    =  [ 8, 32 ][GPU>0]
    GLOBAL   =  8192
    OPT      =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, N, NX, NY, NZ)
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program  =  cl.Program(context, source).build(OPT)
    #
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(OFF, np.int32))
    H_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
    X_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    Y_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
    Z_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z)
    L_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=l)
    AIND_buf   =  cl.Buffer(context, mf.WRITE_ONLY, N*4)
    Coo2Ind    =  program.Coo2IndLevel
    Coo2Ind.set_scalar_arg_dtypes([      None,       None,    None,  None,  None,  None,  None,  None ])
    Coo2Ind(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf, X_buf, Y_buf, Z_buf, L_buf, AIND_buf)
    queue.finish()
    IND       =  np.zeros(N, np.int32)
    cl.enqueue_copy(queue, IND, AIND_buf)
    return IND
    

    
    

def OT_GetIndicesVLowmem(x, y, z, NX, NY, NZ, LCELLS, OFF, H, GPU=1, global_index=False, platforms=[0,1,2,3,4], max_level=99):
    """
    Use OpenCL kernel to get indices { level, index } or  {global index} for input coordinates.
    Usage:
        IND = OT_GetIndicesV(x, y, z, NX, NY, NZ, LCELLS, OFF, H, GPU=0, global_index=False, platforms=[0,1,2,3,4])
    Input:
        x, y, z    =  vectors of positions in root grid units
        NX, NY, NZ =  dimensions of the cloud itself
        LCELLS     =  cells on each level of hierarchy
        OFF        =  offset for the first cell on each level of hierarchy
        H          =  vector containing the hierarchy data (a single vector)
        GPU        =  if >0, try to use GPU instead of CPU
        global_index = if True, return asingle index to a global data vector
    Return:
        if (global_index==False):
            levels   =  vector of levels, [len(x)]
            indices  =  vector of cell indices within a level, [len(x)]
        if (global_index==True):
            indices  =  vector of global cell indices [len(x)]
    """
    LEVELS   =  len(LCELLS)
    CELLS    =  sum(LCELLS)
    N        =  size(x)
    x        =  np.asarray(x, np.float32)
    y        =  np.asarray(y, np.float32)
    z        =  np.asarray(z, np.float32)
    #
    BATCH      =  np.min([10000000, len(x)])
    ###
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    LOCAL    =  [ 8, 32 ][GPU>0]
    GLOBAL   =  8192
    OPT      =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, BATCH, NX, NY, NZ)
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program  =  cl.Program(context, source).build(OPT)
    #
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(OFF, np.int32))
    H_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
    X_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
    Y_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
    Z_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH)
    AIND_buf   =  cl.Buffer(context, mf.WRITE_ONLY, 4*BATCH)
    if (global_index==False):
        ALEV, AIND =  np.zeros(N, np.int32), np.zeros(N, np.int32)
        ALEV_buf   =  cl.Buffer(context, mf.WRITE_ONLY, 4*BATCH)  # N = number of (x,y,z) points
        Coo2IndV   = program.Coo2IndV
        #                               LCELLS  OFF   H     X     Y     Z     ALEV  AIND
        Coo2IndV.set_scalar_arg_dtypes([None,   None, None, None, None, None, None, None ])
        for ibatch in range(1+N//BATCH):
            a = ibatch*BATCH
            b = np.min([a + BATCH, N])
            cl.enqueue_copy(queue, X_buf, x[a:b])
            cl.enqueue_copy(queue, Y_buf, y[a:b])
            cl.enqueue_copy(queue, Z_buf, z[a:b])
            Coo2IndV(queue, [GLOBAL,], [LOCAL,],   
            # LCELLS    OFF      H          X      Y      Z         ALEV      AIND
            LCELLS_buf, OFF_buf, H_buf,     X_buf, Y_buf, Z_buf,    ALEV_buf, AIND_buf)
        cl.enqueue_copy(queue, ALEV[a:b], ALEV_buf)
        cl.enqueue_copy(queue, AIND[a:b], AIND_buf)
        return ALEV, AIND
    else:
        IND       =  np.zeros(N, np.int32)
        Coo2IndV  =  program.Coo2IndVV
        #                               LCELLS  OFF   H     X     Y     Z     IND   MAX_LEVEL
        Coo2IndV.set_scalar_arg_dtypes([None,   None, None, None, None, None, None, np.int32])
        for ibatch in range(1+N//BATCH): 
            a = ibatch*BATCH
            b = np.min([a + BATCH, N])
            cl.enqueue_copy(queue, X_buf, x[a:b])
            cl.enqueue_copy(queue, Y_buf, y[a:b])
            cl.enqueue_copy(queue, Z_buf, z[a:b])        
            Coo2IndV(queue, [GLOBAL,], [LOCAL,], 
            # LCELLS    OFF      H       X      Y      Z      IND       MAX_LEVEL
            LCELLS_buf, OFF_buf, H_buf,  X_buf, Y_buf, Z_buf, AIND_buf, max_level)
            cl.enqueue_copy(queue, IND[a:b], AIND_buf)
        return IND
    


    
def Reroot(old_ot_filename, new_ot_filename):
    """
    Reroot by adding a new root layer one level up. 
    For example, if old root was 512^3, the new root level will be 256^3
    and the number of levels in the octree hierarchy increases by one.
    Usage:
        Reroot(old_ot_filename, new_ot_filename)
    """
    # reroot using a smaller root grid
    NX, NY, NZ, LCELLS, OFF, H = OT_ReadHierarchyV(old_ot_filename)
    DIM = NX//2    # new root level is one level above old root grid
    # add new root layer
    H0 = np.zeros(DIM*DIM*DIM, np.float32)
    for i in range(DIM*DIM*DIM):
        H0[i] = -I2F(8*i)  # links to old root grid cells, all root cells refined
    # WE MUST STILL REORDER OLD ROOT LEVEL SO THAT CELLS OF AN OCTET (old root level) ARE CONSECUTIVE
    OLD    =  H[0: (NX*NY*NZ)].reshape(NZ, NY, NX)   # old root grid
    NEW    =  np.zeros(NZ*NY*NX,  np.float32)
        
    for k in range(DIM):      # loop over new root level
        print("%3d / %3d" % (1+k, DIM))
        for j in range(DIM):
            for i in range(DIM):
                ind            =  i+DIM*(j+DIM*k)             # index on new root level
                #  2x2x2 cell patch from the original OLD root grid become consecutive 8 cell octet in NEW
                NEW[8*ind+0]   =  OLD[2*k,   2*j,   2*i  ]
                NEW[8*ind+1]   =  OLD[2*k,   2*j,   2*i+1]
                NEW[8*ind+2]   =  OLD[2*k,   2*j+1, 2*i  ]
                NEW[8*ind+3]   =  OLD[2*k,   2*j+1, 2*i+1]
                NEW[8*ind+4]   =  OLD[2*k+1, 2*j  , 2*i  ]
                NEW[8*ind+5]   =  OLD[2*k+1, 2*j  , 2*i+1]
                NEW[8*ind+6]   =  OLD[2*k+1, 2*j+1, 2*i  ]
                NEW[8*ind+7]   =  OLD[2*k+1, 2*j+1, 2*i+1]
    H[0:(NX*NY*NZ)] = ravel(NEW)
    LCELLS = concatenate(( np.asarray([DIM*DIM*DIM], np.int32), LCELLS ))
    OFF    = concatenate(( np.asarray([0,], np.int32), OFF ))
    for i in range(1, len(OFF)):
        OFF[i] += DIM*DIM*DIM
    H = concatenate((H0, H))
    OT_WriteHierarchyV(DIM, DIM, DIM, LCELLS, OFF, H, new_ot_filename)
                        

    
def OT_MakeEmptyHierarchy(DIM, MAXL):
    """
    Create an empty cloud with root grid DIM**3 and all cells refined down to level MAXL
    Usage:
        NX, NY, NZ, LCELLS, OFF, H = OT_MakeFullHierarchy(DIM, MAXL)
    Input:
        DIM   =  dimension of the root grid
        MAXL  =  maximum level (MAXL=LEVELS-1)
    Return:
        NX, NY, NZ  =  root grid dimensions
        LCELLS      =  number of cells on each level, LCELLS[LEVELS]
        OFF         =  offsets (indices) for each level, OFF[LEVELS]
        H           =  the hierarchy
    """
    NX, NY, NZ =  DIM, DIM, DIM
    LEVELS     =  MAXL+1
    LCELLS     =  [ NX*NY*NZ,]
    OFF        =  [ 0,]
    H          =  np.ones(NX*NY*NZ, np.float32)  # root level
    for P in range(MAXL):                                 # loop over parent layers = addition of child layers
        LCELLS.append(LCELLS[P]*8)                        # children on level P+1
        OFF.append(OFF[P]+LCELLS[P])                      # offset for cells on layer P+1
        H  = concatenate((H, np.ones(LCELLS[P+1], np.float32))) # add child cells, filled with ones
        # update links for all level P cells
        for i in range(LCELLS[P]):  H[OFF[P]+i] = -I2F(8*i)
    return  NX, NY, NZ, np.asarray(LCELLS, np.int32), np.asarray(OFF, np.int32), H



def OT_MakeEmptyHierarchyCL(DIM, MAXL, GPU=1, platforms=[0,1,2,3]):
    """
    Create an empty cloud with root grid DIM**3 and all cells refined down to level MAXL
    Usage:
        NX, NY, NZ, LCELLS, OFF, H = OT_MakeFullHierarchy(DIM, MAXL)
    Input:
        DIM   =  dimension of the root grid
        MAXL  =  maximum level (MAXL=LEVELS-1)
    Return:
        NX, NY, NZ  =  root grid dimensions
        LCELLS      =  number of cells on each level, LCELLS[LEVELS]
        OFF         =  offsets (indices) for each level, OFF[LEVELS]
        H           =  the hierarchy
    """
    NX, NY, NZ =  DIM, DIM, DIM
    LEVELS     =  MAXL+1
    LCELLS     =  np.zeros(LEVELS, np.int32)
    OFF        =  np.zeros(LEVELS, np.int32)
    total      =  0
    for i in range(LEVELS):
        LCELLS[i]  =  DIM*DIM*DIM*(8**i)
        total     +=  LCELLS[i]
        if (i>0):  OFF[i] = OFF[i-1]+LCELLS[i-1]
    cells =  sum(LCELLS)
    H     =  np.ones(cells, np.float32)
    ##
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=True)
    LOCAL    =  [ 8, 32 ][GPU>0]
    GLOBAL   =  32768
    OPT      =  ""
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_make_empty_hierarchy.c').read()
    program  =  cl.Program(context, source).build(OPT)
    Index    =  program.Index
    Index.set_scalar_arg_dtypes([np.int32, None])
    C_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(LCELLS[MAXL-1]))
    for j in range(MAXL):   # loop over parent layers, setting the links
        Index(queue, [GLOBAL,], [LOCAL,], LCELLS[j], C_buf)
        tmp = np.zeros(LCELLS[j], np.float32)
        cl.enqueue_copy(queue, tmp, C_buf)
        H[OFF[j]:(OFF[j]+LCELLS[j])] = tmp
    return  NX, NY, NZ, LCELLS, OFF, H




        
def OT_PropagateVelocity(NX, NY, NZ, LCELLS, OFF, H0, VX, VY, VZ, SIGMA, 
      GPU=0, platforms=[0,1,2,3,4], fill_sigma=3.0):
    """
    Propagate velocity information to all cells in (VX, VY, VZ, SIGMA),
    parent cell has the average velocity of children and sigma = k * std(children)
    Usage:
        vx, vy, vz, sigma =  OT_PropagateVelocity(NX, NY, NZ, LCELLS, OFF, H, VX, VY, VZ, SIGMA, GPU=0, platforms=[0,1,2,3,4])
    Input:
        NX, NY, NZ   =  dimensions of the root grid
        LCELLS       =  number of cells on each hierarchy level
        OFF          =  offsets to the first cell on each hierarchy level
        H            =  the hierarchy *** single vector ***
        VX, VY, VZ, SIGMA = similar vectors for the velocity field components
        GPU          =  if >0, use GPU instead of CPU (default GPU=0)
        platforms    =  possible OpenCL platforms (default platforms=[0,1,2,3,4])
        fill_sigma   =  sigma value assigned to cells for which sigma could not be estimated
                        (= root grid cells that do not have any children !)
    Return:
        vx, vy, vz, sigma  =  new vectors with values in all cells, not only leaf cells
    """
    if (len(H0)!=len(VX)):
        print("OT_PropagateVelocity: ERROR IN DIMENSIONS"), sys.exit()
    H          =  H0.copy()    # H vector will be altered !!
    LEVELS     =  len(nonzero(LCELLS>0)[0])  # could be [ 999, 0, 0, 0, ...]
    CELLS      =  sum(LCELLS)
    platform, device, context, queue, mf  =  InitCL(GPU=GPU, platforms=platforms, verbose=False)
    GLOBAL     =  8192
    LOCAL      =  [ 8, 32 ][GPU>0]
    OPT        =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d " % (LEVELS, CELLS, NX, NY, NZ)
    source     =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program    =  cl.Program(context, source).build(OPT)
    # Use kernel Parents to find the parents
    max_cells  =  np.int64(np.max(LCELLS))
    print('max_cells = %.3e' % max_cells)
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(OFF,    np.int32))
    HP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # hierarchy information - parent level
    HC_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # hierarchy information - child level
    PX_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # parameters on parent level
    PY_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # parameters on parent level
    PZ_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # parameters on parent level
    PS_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # parameters on parent level
    CX_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # parameters on child level
    CY_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # parameters on child level
    CZ_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # parameters on child level
    CS_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # parameters on child level
    #
    PROP       =  program.PropagateVelocity
    PROP.set_scalar_arg_dtypes([np.int32, np.int32,  None, None,    None, None, None, None,   None, None, None, None])
    
    # One level at a time, calculate values to parent cells
    for level in range(LEVELS-2, -1, -1):                          # loop over parent levels
        print("========== PARENT LEVEL %d ==========" % level)
        A, B    =  OFF[level  ],  OFF[level  ]+LCELLS[level  ]     # limits for parent cells  [A,B[
        a, b    =  OFF[level+1],  OFF[level+1]+LCELLS[level+1]     # limits for child  cells  [a,b[
        NP, NC  =  np.int32(B-A), np.int32(b-a)        
        cl.enqueue_copy(queue, HP_buf,     H[A:B].copy())          # parent level from the hierarchy
        cl.enqueue_copy(queue, HC_buf,     H[a:b].copy())          # child  level from the hierarchy
        cl.enqueue_copy(queue, CX_buf,     VX[a:b].copy())         # velocity in child cells
        cl.enqueue_copy(queue, CY_buf,     VY[a:b].copy())         # velocity in child cells
        cl.enqueue_copy(queue, CZ_buf,     VZ[a:b].copy())         # velocity in child cells
        cl.enqueue_copy(queue, CS_buf,     SIGMA[a:b].copy())      # sigma    in child cells
        # push also current parent velocities... since some elements may be updated, we need all elements up to date
        cl.enqueue_copy(queue, PX_buf,     VX[A:B])
        cl.enqueue_copy(queue, PY_buf,     VY[A:B])
        cl.enqueue_copy(queue, PZ_buf,     VZ[A:B])
        cl.enqueue_copy(queue, PS_buf,     SIGMA[A:B])        
        PROP(queue,[GLOBAL,],[LOCAL,], NP, NC, HP_buf, HC_buf,  PX_buf, PY_buf, PZ_buf,PS_buf,    CX_buf, CY_buf, CZ_buf, CS_buf)        
        cl.enqueue_copy(queue, VX[A:B],    PX_buf)                 # updated values on the parent level
        cl.enqueue_copy(queue, VY[A:B],    PY_buf)                 # updated values on the parent level
        cl.enqueue_copy(queue, VZ[A:B],    PZ_buf)                 # updated values on the parent level
        cl.enqueue_copy(queue, SIGMA[A:B], PS_buf)                 # updated values on the parent level
        # also sigma may have been set on the child cell = for leaf cells
        cl.enqueue_copy(queue, SIGMA[a:b], CS_buf)                 # updated values in leafs (child level)
        # we also propagate density information up => HP_buf contains all densities, not more links !!
        cl.enqueue_copy(queue, H[A:B],     HP_buf)
        ### if (level==4): break
    ###
    # The above fails when root grid cells have no children => sigma values remain 0.0
    m = nonzero(SIGMA<=0.0)
    SIGMA[m] = fill_sigma
    if (len(m[0])>0):
        if (m[0][-1]>=(NX*NY*NZ)):
            print("OT_PropagateVelocity => Error?  SIGMA=0.0 for some cells above the root grid???")
            print("ROOT GRID %d CELLS,   SIGMA==0.0 FOR %d CELLS" % (NX*NY*NZ, len(m[0])))
    return VX, VY, VZ, SIGMA



def OT_PropagateScalar(NX, NY, NZ, LCELLS, OFF, H, X, GPU=0, platforms=[0,1,2,3,4]):
    """
    Propagate scalar (probably either temperature T or abundance CHI) information from leaves to all nodes,
    with simple averaging over the child nodes.
    Usage:
        X  =  OT_PropagateScalar(NX, NY, NZ, LCELLS, OFF, H, X, GPU=0, platforms=[0,1,2,3,4])
    Input:
        NX, NY, NZ   =  dimensions of the root grid
        LCELLS       =  number of cells on each hierarchy level
        OFF          =  offsets to the first cell on each hierarchy level
        X            =  the parameter vector (usually either temperature T or abundance CHI)
        GPU          =  if >0, use GPU instead of CPU (default GPU=0)
        platforms    =  possible OpenCL platforms (default platforms=[0,1,2,3,4])
    Return:
        X  =  the updated input vector with all nodes having averages over its children
    """
    if (len(H)!=len(X)):
        print("OT_PropagateSclar: ERROR IN DIMENSIONS"), sys.exit()
    LEVELS     =  len(nonzero(LCELLS>0)[0])  # could be [ 999, 0, 0, 0, ...]
    CELLS      =  sum(LCELLS)
    platform, device, context, queue, mf  =  InitCL(GPU=GPU, platforms=platforms, verbose=False)
    GLOBAL     =  8192
    LOCAL      =  [ 1, 32 ][GPU>0]
    OPT        =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d " % (LEVELS, CELLS, NX, NY, NZ)
    source     =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program    =  cl.Program(context, source).build(OPT)
    max_cells  =  np.int64(np.max(LCELLS))
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(OFF,    np.int32))
    HP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # hierarchy information - parent level
    HC_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # hierarchy information - child level
    P_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # parameters on parent level
    C_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)   # parameters on child level
    #
    PROP       =  program.PropagateScalar
    #                           np        HP    HC    XP    XC  
    PROP.set_scalar_arg_dtypes([np.int32, None, None, None, None])    
    # One level at a time, calculate values to parent cells
    for level in range(LEVELS-2, -1, -1):                          # loop over parent levels
        # print("========== PARENT LEVEL %d ==========" % level)
        A, B    =  OFF[level  ],  OFF[level  ]+LCELLS[level  ]     # limits for parent cells  [A,B[
        a, b    =  OFF[level+1],  OFF[level+1]+LCELLS[level+1]     # limits for child  cells  [a,b[
        NP      =  np.int32(B-A)
        cl.enqueue_copy(queue, HP_buf,     H[A:B].copy())          # parent level from the hierarchy
        cl.enqueue_copy(queue, HC_buf,     H[a:b].copy())          # child  level from the hierarchy
        cl.enqueue_copy(queue, P_buf,      X[A:B].copy())          # scalars on parent level
        cl.enqueue_copy(queue, C_buf,      X[a:b].copy())          # scalars on child level
        PROP(queue,[GLOBAL,],[LOCAL,], NP, HP_buf, HC_buf,  P_buf, C_buf)        
        cl.enqueue_copy(queue, X[A:B],     P_buf)                  # updated values on the parent level
    return X
    
    
    

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^






def OT_ReadHierarchy(filename):
    """
    Usage:
        nx, ny, nz, levels, lcells, H = OT_ReadHierarchy(filename)
    Read octtree structure from disk, data and links.
    Input:
        name of the input file
    Returns:
        nx, ny, nz  = dimensions of the root grid
        levels      = number of levels in the hierarchy
        lcells      = vector, number of cells in each level
        H           = the hierarchy file, one vector per hierarchy level
    """
    # print('ReadHierarchy ', filename)
    fp = open(filename, 'r')
    nx, ny, nz, levels, cells = fromfile(fp, np.int32, 5)
    # print('    OT_ReadHierarchy: ', nx, ny, nz, levels, cells)
    lcells = np.zeros(levels, np.int32)
    H = []
    for i in range(levels):
        tmp       = fromfile(fp, np.int32, 1)
        # print(tmp)
        # print('    OT_ReadHierarchy  --- LCELLS[%02d] = %6d' % (i, tmp))
        lcells[i] = tmp
        H.append(fromfile(fp, np.float32, lcells[i]))
    return nx, ny, nz, levels, lcells, H



def OT_WriteHierarchy(nx, ny, nz, levels, lcells, H, filename):
    """
    OT_WriteHierarchy(nx, ny, nz, levels, lcells, H, filename)
    """
    fp = open(filename, 'wb')
    np.asarray([nx, ny, nz, levels, sum(lcells)], np.int32).tofile(fp)
    cells = sum(lcells)
    print("OT_WriteHierarchy, %d levels, %d = %.3e cells" % (levels, cells, cells))
    for i in range(levels):
        np.asarray([lcells[i],], np.int32).tofile(fp)
        np.asarray(H[i], np.float32).tofile(fp)
        print("    level %2d    cells  %.3e = %d" % (i, lcells[i], lcells[i]))
    fp.close()


    
    
def OT_GetValue(x, y, z, NX, NY, NZ, H):
    """
    Return value for global coordinates (x,y,z) from octtree structure
    Usage:
        val = OT_GetValue(x, y, z, nx, ny, nz, H)
    Input:
        x, y, z    --  global coodinates in the root grid units
        nx, ny, nz --  dimensions of the root grid
        H          --  the hierarchy with data and links
                       H[i] is a vector of cell values at level i,
                       including links (indices) of first cell in sub-octet of next level
    """
    LL = []
    # must read negative values of H, change sign, interpret as integers
    levels = len(H)
    for i in range(levels):
        if (len(H[i])<1):
            LL.append([])
            continue
        ll  =  H[i].copy()
        ll  = -ll           # links positive floats
        np.asarray(ll, np.float32).tofile('/dev/shm/tmp.bin')
        ll  =  fromfile('/dev/shm/tmp.bin', np.int32)  # positive indices
        LL.append(ll)
        m   =  nonzero(H[i]<1e-10)
        ## print(ll[m])   # appear to be ok
    ####
    # Root cell
    i, j, k = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))      # root grid indices
    if ((i<0)|(i>=NX)|(j<0)|(j>=NY)|(k<0)|(k>=NZ)): return 0.0 # outside the cloud
    # print(k*NX*NY+j*NX+i, len(H[0]))
    if (H[0][k*NX*NY+j*NX+i]>0.0):   # a leaf at root level
        return H[0][k*NX*NY+j*NX+i]
    # Otherwise follow hierarchy to the bottom
    # Move to next level coordinates before the loop
    ind   =  LL[0][k*NX*NY+j*NX+i]
    level =  1
    x     =  2.0*(x-i)  # [0,2] within the next level octet
    y     =  2.0*(y-j)
    z     =  2.0*(z-k)
    while(1):
        # (x,y,z) in which cell of this next level octet?
        i, j, k = 0, 0, 0
        if (x>1.0): i = 1
        if (y>1.0): j = 1
        if (z>1.0): k = 1
        sid   =  4*k +2*j + i
        ind  +=  sid    # index of the current level cell
        if (H[level][ind]>0.0):
            return H[level][ind]  # found a leaf
        # else we follow link to next level
        ind    =      LL[level][ind]            
        level +=  1
        # coordinates in the next level octet
        x      =  2.0*(x-i)
        y      =  2.0*(y-j)
        z      =  2.0*(z-k)
    return 0.0




def OT_GetIndices(x, y, z, NX, NY, NZ, H, GPU=0, platforms=[0,1,2,3,4]):
    """
    Use OpenCL kernel to get indices { level, index } for input coordinates.
    USES THE NEWLINK FORMAT
    Input:
        x, y, z  =  vectors of positions in root grid units
        H        =  SOCAMO hierarchy
        GPU      =  >0 if using GPU instead of CPU
    Return:
        levels   =  vector of levels, [len(x)]
        indices  =  vector of cell indices within a level, [len(x)]
    """
    LEVELS    =  len(H)
    N         =  len(x)
    if (LEVELS>7):
        print('UPDATE kernel_OT_tools.c -- CURRENT MAXIMUM IS 7 LEVELS, 0-6 !!')
        return  None, None
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    GLOBAL    =  8192
    LOCAL      =  [ 8, 32 ][GPU>0]
    OPT       =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, N, NX, NY, NZ)
    source    =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program   =  cl.Program(context, source).build(OPT)
    #
    DENS_buf  =  []
    LCELLS    =  np.zeros(LEVELS, np.int32)
    for i in range(7):  # maximum of 7 levels
        if (i<LEVELS):
            LCELLS[i] = len(H[i])
            DENS_buf.append( cl.Buffer(context, mf.READ_ONLY, 4*np.int64(np.max([1,LCELLS[i]])))  )
        else:
            DENS_buf.append( cl.Buffer(context, mf.READ_ONLY, 4) )
    # print('LEVELS %d, DENS_buf %d' % (LEVELS, len(DENS_buf)))
    CELLS_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=LCELLS)
    x, y, z    =  np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(z, np.float32)
    X_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    Y_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
    Z_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z)
    CELLS_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=LCELLS)
    ALEV_buf   =  cl.Buffer(context, mf.WRITE_ONLY, N*4)  # N = number of (x,y,z) points
    AIND_buf   =  cl.Buffer(context, mf.WRITE_ONLY, N*4)
    ####
    for i in range(LEVELS):
        if (LCELLS[i]>0):
            # print('Level', i, LCELLS[i])
            h = H[i].copy()
            cl.enqueue_copy(queue, DENS_buf[i], h)
        else:
            pass
            # print('Level', i, 'skip')
    ####
    COO2IND  = program.COO2IND
    COO2IND.set_scalar_arg_dtypes([None, None,None,None,None,None,None,None, None,None,None, None,None])
    ####
    COO2IND(queue, [GLOBAL,], [LOCAL,], CELLS_buf,
    DENS_buf[0], DENS_buf[1], DENS_buf[2], DENS_buf[3], DENS_buf[4], DENS_buf[5], DENS_buf[6],
    X_buf, Y_buf, Z_buf,   ALEV_buf, AIND_buf)
    ####
    ALEV, AIND = np.zeros(N, np.int32), np.zeros(N, np.int32)
    cl.enqueue_copy(queue, ALEV, ALEV_buf)
    cl.enqueue_copy(queue, AIND, AIND_buf)
    return ALEV, AIND        
    


    
    
    
def OT_cut_levels(infile, outfile, maxlevel, HHH_in=[], GPU=0, platforms=[0,1,2,3,4], LOC=False):
    """
    Write a new SOC cloud, cutting levels>maxlevel. maxlevel=0,1,2,...
    Parent cells (old links) are replaced with average density.
    Routine uses OpenCL kernel and the new NEWLINK file format.
    Usage:
        OT_cut_levels(infile, outfile, maxlevel, GPU=0, LOC=False)
    Input:
        infile   = old cloud or magnetic field file
        outfile  = new file
        maxlevel = new maximum level (0,1,2...)
        HHH_in   = optional, separate hierarchy structure
                    ... needed for B files that contain negative values !!!
        GPU      = if >0, use GPU instead of CPU
        LOC      = if >0, assume files are LOC files with seven instead of one field
    Note:
        2021-01-02 tested with a LOC cloud
        2021-08-24 add scaling by 1.4 per hierarchy level to the sigma values
    """
    # Input file
    fpin     =  open(infile, 'rb')
    NX, NY, NZ, LEVELS, CELLS = fromfile(fpin, np.int32, 5)
    maxlevel =  np.min([LEVELS-1, maxlevel])
    LCELLS   =  np.zeros(LEVELS, np.int32)
    print('OT_cut_levels: reading hierarchy with %d levels -> maxlevel %d' % (LEVELS, maxlevel))
    # Output file
    fpout    =  open(outfile, 'wb')
    # Create OpenCL program to average child values into parent node
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    LOCAL    =  [ 8, 32 ][GPU>0]
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    OPT      = '-D NX=0 -D NY=0 -D NZ=0 -D N=0 -D LEVELS=%d' % LEVELS  # all dummy
    program  =  cl.Program(context, source).build(OPT)
    AverageParent = program.AverageParent
    AverageParent.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None])
    GLOBAL   =  0         # becomes known only inside the loop
    P_buf    =  None
    C_buf    =  None
    H_buf    =  None
    #############################
    HHH      =  HHH_in  # if separate hierarchy information given
    for ifield in range([1,7][LOC>0]):  # 1 field for SOC, 7 fields for LOC
        H = []      # current field ---there will not be "V" version, here array of vectors is more convenient
        # Read data for the current field
        for i in range(LEVELS):
            cells      =  fromfile(fpin, np.int32, 1)[0]
            LCELLS[i]  =  cells
            H.append(fromfile(fpin, np.float32, cells))
        if (ifield==0):  # allocate buffers only once LCELLS is known
            GLOBAL   =  np.max(LCELLS)   # at most this many parent cells, at most this many vector elements
            GLOBAL   =  (GLOBAL//64+1)*64
            P_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(GLOBAL))   # parent level cells
            C_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(GLOBAL))   # child level cells 
            H_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(GLOBAL))   # hierarchy for the parent level            
            ### also dimensions of the output file are known only now
            np.asarray([NX, NY, NZ, maxlevel+1, sum(LCELLS[0:(maxlevel+1)])], np.int32).tofile(fpout)            
        # Copy hierarchy information to HHH (in case we have more fields, others not with proper hierarchy data)
        if ((ifield==0)&(len(HHH)<1)):
            HHH = []
            for i in range(LEVELS):
                HHH.append(H[i].copy())
        # Drop levels
        for i in arange(LEVELS-2, maxlevel-1, -1):    # loop over the parent levels, including maxlevel
            # Push H[i] and H[i+1] to device, kernel updates the H[i] level links into leafs
            cl.enqueue_copy(queue, P_buf,   H[i  ])   # parents, current field
            cl.enqueue_copy(queue, C_buf,   H[i+1])   # children, current field
            cl.enqueue_copy(queue, H_buf, HHH[i  ])   # links from HHH...
            AverageParent(queue, [GLOBAL,], [LOCAL,], len(H[i]), len(H[i+1]), P_buf, C_buf, H_buf)
            cl.enqueue_copy(queue, H[i], P_buf)       # H[i] now with averages over children
            if (ifield==2):  # turbulent sigma field
                H[i][nonzero(H[i]>0.0)] *= 1.4 # factor of 2 change in scale ~ factor of 1.5 in turbulent velocity
        # Write data for the current field
        for i in range(maxlevel+1):
            print('   ---> write level %2d' % i)
            np.asarray([LCELLS[i],], np.int32).tofile(fpout)
            np.asarray(H[i], np.float32).tofile(fpout)
    #############################
    fpin.close()
    fpout.close()
    ##
    try:
        if (infile.find('.cloud')>0):  # if converting cloud (rho), update meta
            meta_in  = infile.replace('.cloud', '.meta')
            meta_out = outfile.replace('.cloud', '.meta')
            meta_reset_levels(meta_in, meta_out, maxlevel)
    except:
        print('No meta file found -- none updated')
        pass
    

    
    
def OT_prune_cloud_LOC(infile, outfile, socfile, GPU=0, platforms=[0,1,2,3,4]):
    """
    Write a new LOC cloud file based on a pruned hierarchy.
    Usage:
        OT_prune_cloud_LOC(infile, outfile, socfile, GPU=0, LOC=False)
    Input:
        infile    = old LOC file
        outfile   = new LOC file
        socfile   = SOC file describing the new hierarchy structure
        GPU       = if >0, use GPU instead of CPU (default=0)
        platforms = list of OpenCL platforms (default=[0,1,2,3,4])
    Note:
        we call OT_PropagateVelocity to fill in SIGMA, VX, VY, VZ in all cells
        we call separate kernel on T, CHI to average children to parent nodes
        we use density values from SOCFILE rather than from INFILE, assuming that SOCFILE 
        corresponds to the same INFILE hierarchy after OT_prune_cloud and/or OT_cut_levels
    """
    # start by filling all information (except density=H) to all nodes of the INFILE hierarchy  @@
    NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ, CHI = OT_ReadHierarchyV_LOC(infile)
    VX, VY, VZ, S =  OT_PropagateVelocity(NX, NY, NZ, LCELLS, OFF, H, VX, VY, VZ, S, GPU=GPU, platforms=platforms)    
    T             =  OT_PropagateScalar(NX, NY, NZ, LCELLS, OFF, H,   T, GPU=GPU, platforms=platforms)
    CHI           =  OT_PropagateScalar(NX, NY, NZ, LCELLS, OFF, H, CHI, GPU=GPU, platforms=platforms)
    # now we have  all data except H0=density averaged/calculated to all hierarchy levels, not just leaves
    # => we can read (S, T, VX, VY, VZ, CHI) for the SOCFILE cell coordinates
    nx, ny, nz, lcells, off, h = OT_ReadHierarchyV(socfile)
    # extract from SOC hierarchy the coordinates ***and levels*** of all leaves
    x, y, z, l  =  OT_GetCoordinatesAndLevels(nx, ny, nz, lcells, off, h, GPU=GPU, platforms=platforms)
    # using coordinates ***and levels***, extract cell indices in the original LOC hierarchy
    ind         =  OT_GetIndicesForLevels(x, y, z, l,  NX, NY, NZ, LCELLS, OFF, H, GPU=GPU, platforms=platforms)
    # the new (T, S, VX, VY, VZ, CHI) vectors corresponding to the hierarchy in h
    T           =    T[ind].copy()
    S           =    S[ind].copy()
    VX          =   VX[ind].copy()
    VY          =   VY[ind].copy()
    VZ          =   VZ[ind].copy()
    CHI         =  CHI[ind].copy()
    # save as a new LOC file
    OT_WriteHierarchyV_LOC(nx, ny, nz, lcells, off, h, T, S, VX, VY, VZ, CHI, outfile)

    
def OT_cut_volume_old(infile, outfile, limits, hierarchy=[]):
    """
    Read in a SOCAMO cloud and extract a subvolume to a new file.
    Input:
        infile   =  name of the input SOCAMO octtree file
        outfile  =  name of the new file with the subvolume
        limits   =  3x2 array with the lower and upper limits as root cell indices
                    [ [xmin, xmax], ...], output will include xmin and xmax indices
        hierarchy = if infile is for B, need to provide rho file for hierarchy = FILENAME, NOT ARRAY
    Note:
        if there is a meta file, that is also updated
        rootgrid, dimensions -> new NX, NY, NZ
    Note:
        Does *not* work with LOC, only with SOC clouds.
    """    
    fin  = open(infile,  'rb')
    fout = open(outfile, 'wb')
    faux = None
    NX, NY, NZ, LEVELS, CELLS = fromfile(fin, np.int32, 5)
    print('INFILE:', NX, NY, NZ, LEVELS, CELLS)
    if (len(hierarchy)>0):  
        faux = open(hierarchy, 'rb')
        NX, NY, NZ, LEVELS, CELLS = fromfile(faux, np.int32, 5)
    NX0, NY0, NZ0 = NX, NY, NZ
    TRUE_LEVELS = 0
    limits      = np.asarray(limits)
    rho, RHO, link, LINKS = None, None, None, None  # LINKS = -RHO read as np.int32 == links
    for i in range(LEVELS):
        # READ THE NEXT LEVEL
        cells = fromfile(fin, np.int32, 1)[0]
        rho   = fromfile(fin, np.float32, cells)
        ##########################################################
        links = None
        if (faux==None):              # convert rho into links
            np.asarray(-rho,  np.float32).tofile('/dev/shm/tmp143563.bin')
            links =  fromfile('/dev/shm/tmp143563.bin', np.int32)
        else:                         # read links from separate file
            fromfile(faux, np.int32, 1)  # cells
            tmp = fromfile(faux, np.float32, cells)
            np.asarray(-tmp,  np.float32).tofile('/dev/shm/tmp143563.bin')
            links =  fromfile('/dev/shm/tmp143563.bin', np.int32) # links >= 0
        ##########################################################
        if (i==0):
            # we cut the root grid to a set of root cells
            K, J, I    =  indices((NZ, NY, NX), np.int32)
            K, J, I    =  ravel(K), ravel(J), ravel(I)
            m          =  nonzero((I>=limits[0,0])&(I<=limits[0,1])&(J>=limits[1,0])&(J<=limits[1,1])&(K>=limits[2,0])&(K<=limits[2,1]))
            rho        =  rho[m]           # links and leaf cells, the subvolume only
            links      =  links[m]
            I, J, K    =  I[m], J[m], K[m]
            NX         =  np.max(I)-np.min(I)+1
            NY         =  np.max(J)-np.min(J)+1
            NZ         =  np.max(K)-np.min(K)+1            
            CELLS      =  len(rho)
            np.asarray([NX, NY, NZ, LEVELS, CELLS], np.int32).tofile(fout) # CELLS must be updated later!!
            CELLS      =  0        # SO FAR NO CELL DATA WRITTEN TO FILE
            RHO        =  rho.copy()
            LINKS      =  links.copy()
            continue   # root level written together with level=1
        # RHO is the parent level, still with the old links; actual links in LINKS
        # make new vector rho for the cut child vector and update links in RHO
        # new vector, dropping all children without parents
        rho2    = np.zeros(len(rho), np.float32)
        links2  = np.zeros(len(rho), np.int32)
        ic      = 0                        # new index to child octets
        for iparent in arange(len(RHO)):   # loop over parents
            l  =  LINKS[iparent]
            if (l>=0.0):                   # this is a link!
                r   =  RHO[iparent]
                ind =  LINKS[iparent]      # original first cell in suboctet
                LINKS[iparent] = 8*ic      # new child link, still as positive int
                ############ RHO[iparent]   = -8*ic     # update the link in RHO
                ## print(len(rho2), 8*ic, len(rho), ind)
                rho2[(8*ic):(8*ic+8)]   = rho[ind:(ind+8)]    # copy child values ind -> 8*ic
                links2[(8*ic):(8*ic+8)] = links[ind:(ind+8)]  # copy links for new children
                ic += 1                    # this many child octets so far
        # truncate the child vector, only 8*ic child cells
        rho, links =  None, None
        rho, links =  rho2[0:(8*ic)], links2[0:(8*ic)]
        print('PARENT HAS LINKS TO %d OCTETS, %d CELLS -- THERE ARE %d CHILDREN' % (ic, 8*ic, len(rho)))
        # how many links does this have
        L   =  np.asarray(links[nonzero(links>=0)], np.int32)  # all the links on child level
        LP  =  np.asarray(LINKS[nonzero(LINKS>=0)], np.int32)  # all the links on parent level
        print('LEVEL %d:' % i)
        maxi = -1
        if (len(LP)>0): maxi = np.max(LP)
        print('         parent %5d CELLS, %5d CHILDREN, MAX LINK %5d' %   (len(RHO), 8*len(LP), maxi))
        if (len(L)>0):
            print('         child  %5d CELLS, %5d CHILDREN, MAX LINK %5d' % (len(rho), 8*len(L), np.max(L)))
        else:
            print('         child  %5d CELLS' % (len(rho)))
        # write the data for THE PARENT LEVEL
        CELLS  += len(RHO)   # total number of cells
        np.asarray([len(RHO),], np.int32).tofile(fout)
        # replace all LINK values with the np.int32 read as float, times -1
        np.asarray(LINKS, np.int32).tofile('/dev/shm/tmp143563.bin')
        tmp    =  fromfile('/dev/shm/tmp143563.bin', np.float32)
        #### m      =  nonzero(RHO<=0.0)
        m      =  nonzero(LINKS>=0.0)  # mask for link cells
        RHO[m] = -tmp[m] 
        np.asarray(RHO, np.float32).tofile(fout)
        RHO   = rho.copy()       # the vector of parents for the next iteration
        LINKS = links.copy()
        TRUE_LEVELS += 1         # could be less than LEVELS, if the cut region is not refined
        if (len(rho)<1): break   # the cut region does not contain any deeper levels
        if (len(LP)>0):
            if (np.max(LP)>=len(rho)):
                print('*'*80)
                print('  ERROR:  PARENT HAS A LINK TO CELL %d, THIS LEVEL HAS ONLY %d CELLS' % (np.max(LP), len(rho)))
                print('*'*80)
                sys.exit()
    # ==== end of for levels================================================================================
    
    # write the final level = LEVELS-1 (no links there)
    CELLS  += len(rho)   # total number of cells
    np.asarray([len(rho),], np.int32).tofile(fout)
    ## convert links to proper format int -> float
    np.asarray(links, np.int32).tofile('/dev/shm/tmp143563.bin')
    tmp    =  fromfile('/dev/shm/tmp143563.bin', np.float32)
    m      =  nonzero(links>=0.0)
    rho[m] = -tmp[m]
    ###
    np.asarray(rho, np.float32).tofile(fout)
    TRUE_LEVELS += 1     # could be less than LEVELS, if the cut region is not refined
    fin.close()    
    # update the number of levels and cells
    fout.seek(3*4)
    print('TRUE LEVELS', TRUE_LEVELS)
    np.asarray([TRUE_LEVELS, CELLS,], np.int32).tofile(fout)
    fout.close()
    ####
    ####
    # new meta file ** ONLY IF THIS IS *.cloud !!! not for B files ...
    inmeta  = infile.replace( '.cloud', '.meta')
    outmeta = outfile.replace('.cloud', '.meta')
    if (os.path.exists(inmeta)&(inmeta.find('.meta')>0)&(outmeta.find('.meta')>0)):
        fp = open(outmeta, 'w')
        for line in file(inmeta).readlines():
            s = line.split()
            if (len(s)<1): continue
            if (   s[0]=='xrange'):
                fp.write('xrange %14.8e %14.8e\n'   % (0.0, NX*1.0/NX0))
            elif ( s[0]=='yrange'):
                fp.write('yrange %14.8e %14.8e\n'   % (0.0, NY*1.0/NY0))
            elif ( s[0]=='zrange'):
                fp.write('zrange %14.8e %14.8e\n'   % (0.0, NZ*1.0/NZ0))
            elif ( s[0]=='rootgrid'):
                fp.write('rootgrid %4d %4d %4d\n'   % (NX, NY, NZ))
            elif ( s[0]=='dimensions'):
                fp.write('dimensions %4d %4d %4d\n' % (NX, NY, NZ))
            elif ( s[0]=='direction'):   # set this to the longest LOS direction
                if   (NX == np.max([NX, NY, NZ])):
                    fp.write('direction x\n')
                elif (NY == np.max([NX, NY, NZ])):
                    fp.write('direction y\n')
                else:
                    fp.write('direction z\n')
            else:
                fp.write(line)
        fp.close()
    ###
    os.system('rm /dev/shm/tmp143563.bin')

    
        

def OT_cut_volume_old2(infile, outfile, limits, LOC=False, GPU=0, platforms=[0,1,2,3,4]):
    """
    Usage:
        OT_cut_volume_2(infile, outfile, limitslevels)
    Input:
        infile   -  name of a SOC or LOC hierarchy file
        outfile  -  output file, subvolume cut from the input file
        limits   -  3x2 integer array giving the limits in X, Y, and Z direction,
                    limits are inclusive  [ [ 0 9 ] ...]  means root grid indices 0, 1, ..., 9
        LOC      -  if True, assume 7 fields instead of the one for SOC
    Note:
        We require that all fields contain complete hierarchy information !!
        2021-01-03 ... UNTESTED
    """
    N, T, S, VX, VY, VZ, CHI = [], [], [], [], [], [], []
    if (LOC):
        NX, NY, NZ, LCELLS, OFF, N, T, S, VX, VY, VZ, CHI = OT_ReadHierarchyV_LOC(infile)
    else: 
        NX, NY, NZ, LCELLS, OFF, N = OT_ReadHierarchyV(infile)
    ####
    CELLS    = sum(LCELLS)
    LEVELS   = len(LCELLS)
    ###
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms)
    LOCAL    =   8
    GLOBAL   =  64
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_cut.c').read()
    OPT      = '-D NX=%s -D NY=%d -D NZ=%d -D LEVELS=%d' % (NX, NY, NZ, LEVELS)
    program  =  cl.Program(context, source).build(OPT)
    CV       =  program.CutVolume
    #                           LIM,   OFF,  LCELLS,  H,     A,     B,     lcells
    CV.set_scalar_arg_dtypes([  None,  None, None,    None,  None,  None,  None  ])
    limits     =  np.asarray(limits, np.int32)
    LIM_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=limits)
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=OFF)
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=LCELLS)
    H_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    A_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    B_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    lcells_buf =  cl.Buffer(context, mf.READ_WRITE, 4*LEVELS)
    ####
    cells, lcells, levels   =  0, np.zeros(LEVELS, np.int32), LEVELS
    # limits are inclusive !!
    limits[0,:] =  clip(limits[0,:], 0, NX-1)
    limits[1,:] =  clip(limits[1,:], 0, NY-1)
    limits[2,:] =  clip(limits[2,:], 0, NZ-1)
    nx, ny, nz  =  limits[0,1]-limits[0,0]+1,  limits[1,1]-limits[1,0]+1,  limits[2,1]-limits[2,0]+1
    fp          =  open(outfile, 'wb')
    print()
    for ifield in range([1,7][LOC>0]):
        print("*** ifield = %d ***" % ifield)
        cl.enqueue_copy(queue, H_buf, N)    # copy again for each field (it is modified in the kernel)
        A = [ N, T, S, VX, VY, VZ, CHI ][ifield]
        cl.enqueue_copy(queue, A_buf, A)
        print("*** kernel ***")
        CV(queue, [GLOBAL,], [LOCAL,], LIM_buf, OFF_buf, LCELLS_buf, H_buf, A_buf, B_buf, lcells_buf)
        queue.finish()                
        print("*** kernel DONE ***")
        if (ifield==0):
            cl.enqueue_copy(queue, lcells, lcells_buf)
            cells   = sum(lcells)
            levels  = len(nonzero(lcells>0)[0])  # cut region may not contain all hierarchy levels
            np.asarray([nx, ny, nz, levels, cells], np.int32).tofile(fp)
            print("*** FROM KERNEL %d %d %d  %d %d\n" %(nx, ny, nz, levels, cells))
            print("*** lcells ", lcells)
        # OFF is not changed, only lcells <= LCELLS
        B = np.zeros(CELLS, np.float32)
        cl.enqueue_copy(queue, B, B_buf)
        for ilevel in range(levels):
            print("   ilevel = %2d  ==> %d cells" % (ilevel, lcells[ilevel]))
            np.asarray([lcells[ilevel],], np.int32).tofile(fp)
            B[OFF[ilevel]:(OFF[ilevel]+lcells[ilevel])].tofile(fp) # dropping the cut cells
    fp.close()
            

    
    
    
    
    
def OT_cut_volume(infile, outfile, limits, LOC=False, GPU=0, platforms=[0,1,2,3,4], HIER=''):
    """
    Usage:
        OT_cut_volume(infile, outfile, limitslevels)
    Input:
        infile   -  name of a SOC or LOC hierarchy file
        outfile  -  output file, subvolume cut from the input file
        limits   -  3x2 integer array giving the limits in X, Y, and Z direction,
                    limits are inclusive  [ [ 0 9 ] ...]  means root grid indices 0, 1, ..., 9
        LOC      -  if True, assume 7 fields instead of the one for SOC
        HIER     -  if given, use that exclusively for the hierarchy information
                    (e.g. if other input files contain negative values like velocity or B)
    Note:
        DOES NOT REQUIRE CHILDREN TO BE IN THE ORDER OF PARENTS
    """
    N, T, S, VX, VY, VZ, CHI = [], [], [], [], [], [], []
    if (LOC):
        NX, NY, NZ, LCELLS, OFF, N, T, S, VX, VY, VZ, CHI = OT_ReadHierarchyV_LOC(infile)
    else: 
        NX, NY, NZ, LCELLS, OFF, N = OT_ReadHierarchyV(infile)
    ####
    CELLS    = sum(LCELLS)
    LEVELS   = len(LCELLS)
    ###
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=True)
    LOCAL    =   8
    GLOBAL   =  64
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_cut.c').read()
    OPT      = '-D NX=%s -D NY=%d -D NZ=%d -D LEVELS=%d' % (NX, NY, NZ, LEVELS)
    program  =  cl.Program(context, source).build(OPT)
    CV       =  program.CutVolume
    #                           LIM,   OFF,  LCELLS,  H0,    H1,    X0,   X1    lcells
    CV.set_scalar_arg_dtypes([  None,  None, None,    None,  None,  None, None, None  ])
    limits     =  np.asarray(limits, np.int32)
    LIM_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=limits)
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=OFF)
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=LCELLS)
    H0_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS)) # 4*CELLS could exceed int32 ??
    H1_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    X0_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    X1_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    lcells_buf =  cl.Buffer(context, mf.READ_WRITE, 4*LEVELS)
    ####
    cells, lcells, levels   =  0, np.zeros(LEVELS, np.int32), LEVELS
    # limits are inclusive !!
    limits[0,:] =  clip(limits[0,:], 0, NX-1)
    limits[1,:] =  clip(limits[1,:], 0, NY-1)
    limits[2,:] =  clip(limits[2,:], 0, NZ-1)
    nx, ny, nz  =  limits[0,1]-limits[0,0]+1,  limits[1,1]-limits[1,0]+1,  limits[2,1]-limits[2,0]+1
    fp          =  open(outfile, 'wb')
    print()
    for ifield in range([1,7][LOC>0]):
        print("*** ifield = %d ***" % ifield)
        if (HIER==''):
            print("*** using input vector also for hierarchy") ;
            cl.enqueue_copy(queue, H0_buf, N)    # copy again for each field (it is modified in the kernel)
        else:
            nx, ny, nz, lcells, off, hier = OT_ReadHierarchyV(HIER)
            print("*** using separate hierarchy vector HIER") 
            cl.enqueue_copy(queue, H0_buf, hier)
            del hier
        X = [ N, T, S, VX, VY, VZ, CHI ][ifield].copy()
        cl.enqueue_copy(queue, X0_buf, X)
        print("*** kernel ***")
        CV(queue, [GLOBAL,], [LOCAL,], LIM_buf, OFF_buf, LCELLS_buf, H0_buf, H1_buf, X0_buf, X1_buf, lcells_buf)
        queue.finish()                
        print("*** kernel DONE ***")
        if (ifield==0):
            cl.enqueue_copy(queue, lcells, lcells_buf)  # cells per level in the new hierarchy
            cells   = sum(lcells)
            levels  = len(nonzero(lcells>0)[0])  # cut region may not contain all hierarchy levels
            np.asarray([nx, ny, nz, levels, cells], np.int32).tofile(fp)
            print("*** FROM KERNEL %d %d %d  %d %d\n" %(nx, ny, nz, levels, cells))
            print("*** lcells ", lcells)
        # OFF is not changed, only lcells <= LCELLS
        B = np.zeros(CELLS, np.float32)
        cl.enqueue_copy(queue, X, X1_buf)  # cut hierarchy for selected field
        for ilevel in range(levels):
            print("   ilevel = %2d  ==> %d cells" % (ilevel, lcells[ilevel]))
            np.asarray([lcells[ilevel],], np.int32).tofile(fp)
            X[OFF[ilevel]:(OFF[ilevel]+lcells[ilevel])].tofile(fp) # dropping the cut cells
    fp.close()
            
    
        
        

def OT_prune_hierarchy(infile, outfile, threshold, GPU=0, platforms=[0,1,2,3,4]):
    """
    Write a new SOC cloud, replacing octets<threshold with a single parent cell.
    Parent cells is replaced with the average density of the eight children.
    Usage:
        OT_prune_hierarchy(infile, outfile, threshold, GPU=0, platforms=[0,1,2,3,4])
    Input:
        infile    = old cloud or magnetic field file
        outfile   = new file
        threshold = threshold data value for joining child cells together
                    threshold[LEVELS-1] = threshold for parent layer or scalar
        GPU      = if >0, use GPU instead of CPU
    """    
    NX, NY, NZ, LEVELS, LCELLS, H = OT_ReadHierarchy(infile)    
    print('OT_cut_levels: reading hierarchy with %d levels' % LEVELS)
    LCELLS0 = np.asarray(LCELLS, np.int32).copy()
    # Create OpenCL program to average child values into parent node
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms)
    LOCAL    =  [ 8, 32 ][GPU>0]
    source   = open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    OPT      = '-D NX=0 -D NY=0 -D NZ=0 -D N=0 -D LEVELS=%d' % LEVELS  # all dummy
    program  = cl.Program(context, source).build(OPT)
    ############
    AverageParentThresholded  =  program.AverageParentThresholded
    AverageParentThresholded.set_scalar_arg_dtypes([np.int32, np.int32, None, None, np.float32])
    ############
    UpdateParentLinks         =  program.UpdateParentLinks
    UpdateParentLinks.set_scalar_arg_dtypes([np.int32, None])
    ############
    OrderChildren             =  program.OrderChildren
    OrderChildren.set_scalar_arg_dtypes([np.int32, None, None, None, None])
    ############
    GLOBAL   =  np.max(LCELLS)   # at most this many parent cells, at most this many vector elements
    GLOBAL   =  (GLOBAL//32+1)*32
    P_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(GLOBAL))
    C_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(GLOBAL))
    CO_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(GLOBAL))
    I_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(GLOBAL))
    if (1):
        for i in arange(LEVELS-2, -1, -1):          # loop over the parent levels
            print("PRUNE PARENT-CHILD = %d-%d" % (i, i+1))
            # Push H[i] and H[i+1] to device, kernel updates the H[i] level links into leafs
            P       =  H[i  ].copy()
            C       =  H[i+1].copy()
            NP, NC  =  len(P), len(C)
            cl.enqueue_copy(queue, P_buf, P)   # parents
            cl.enqueue_copy(queue, C_buf, C)   # children
            queue.finish()
            # order children so that they are in the same order as the parents
            print("ORDER CHILDREN")
            m = nonzero(P<=0.0)   # all parents with children
            I = np.asarray(m[0], np.int32)
            cl.enqueue_copy(queue, I_buf, I)
            nop = len(I)          # number of parents with children
            OrderChildren(queue, [GLOBAL,], [LOCAL,], nop, P_buf, C_buf, CO_buf, I_buf)
            queue.finish()
            # update parent links after reordering their children
            print("UPDATE PARENT LINKS")
            UpdateParentLinks(queue, [GLOBAL,], [LOCAL,], NP, P_buf)
            queue.finish()
            if (isscalar(threshold)): thr = threshold
            else:                     thr = threshold[i]
            print('AVERAGE CHILDREN, parent level %d, threshold %.3e' % (i, thr))
            # Average children to parent cell, if the child average below threshold
            AverageParentThresholded(queue, [GLOBAL,], [LOCAL,], NP, NC, P_buf, CO_buf, thr)
            queue.finish()
            # Some children removed - update parent links ... AGAIN !!
            print("UPDATE PARENT LINKS AGAIN!!!") 
            UpdateParentLinks(queue, [GLOBAL,], [LOCAL,], NP, P_buf)
            queue.finish()
            print("LINK UPDATE DONE")
            cl.enqueue_copy(queue, P,  P_buf)
            cl.enqueue_copy(queue, C,  CO_buf)  # removed children have values ~1e31, reordered vector 
            queue.finish()
            # Drop removed child cells from C
            m       =  nonzero(C<1.0e20)        # these child cells will remain
            rem     =  len(C) - len(m[0])       # this many cells removed
            C       =  C[m[0]]
            H[i]    =  P.copy()
            H[i+1]  =  C.copy()
            print(" .... removed %.1e child cells" % (float(rem)))    
    # write the new file
    for i in range(LEVELS): 
        LCELLS[i]  =  len(H[i])
    OT_WriteHierarchy(NX, NY, NZ, LEVELS, LCELLS, H, outfile)
    ####
    print("--------------------------------------------------------------------------------")
    cells, CELLS = sum(LCELLS0), sum(LCELLS)
    print("   %9d -> %9d cells       %.2e -> %.2e cells" % (cells, CELLS, cells, CELLS))
    for i in range(LEVELS):
        print("   level %2d    %.2e  ->  %.2e  =  %.2f %%   cells" % 
        (             i, LCELLS0[i], LCELLS[i], 100.0*LCELLS[i]/LCELLS0[i]))
    print("--------------------------------------------------------------------------------")
    

    
def Turbulence(rhofile, vxfile, vyfile, vzfile, sigmafile):
    """
    Estimate turbulence for every node in the hierarchy.
    Input:
        rhofile  =  OT density file with the hierarchy information
        v*file   =  OT files for the three velocity components
    Output:
        write OT file with velocity dispersions to sigmafile
    """    
    k  =  2.0**1.5    # turbulence ratio between parent and child
    NX, NY, NZ, LEVELS, LCELLS, H  = OT_ReadHierarchy(rhofile)    
    NX, NY, NZ, LEVELS, LCELLS, VX = OT_ReadHierarchy(vxfile)  # originally exists only for leaf nodes
    NX, NY, NZ, LEVELS, LCELLS, VY = OT_ReadHierarchy(vyfile)    
    NX, NY, NZ, LEVELS, LCELLS, VZ = OT_ReadHierarchy(vzfile)        
    CELLS = sum(LCELLS)
    SIGMA = []
    for h in H:    # create SIGMA vectors
        SIGMA.append(zeros(len(h), np.float32))    
    for level in range(LEVELS-2, -1, -1):          # loop over parent levels
        print("LEVEL %d" % level)
        m = nonzero(H[level]<=0)                   # parent cells of "level"
        ind = 0
        for i in m[0]:                             # loop over parent cells
            ind += 1
            if (ind%10000==0): print(" %9d %9d   %.4f" % (ind, len(m[0]), ind/float(len(m[0]))))
            c  = F2I(-H[level][i])                 # index of the first child cell on "level+1"
            sx = std(VX[level+1][c:(c+8)])
            sy = std(VY[level+1][c:(c+8)])
            sz = std(VZ[level+1][c:(c+8)])
            # set sigma value in the parent cell
            sigma           =  (sx+sy+sz)/3.0      # 1d velocity dispersion, assuming it is isotropic !!
            SIGMA[level][i] =  sigma*k
            # set velocity values in the parent cell
            VX[level][i]    =  mean(VX[level+1][c:(c+8)])
            VY[level][i]    =  mean(VY[level+1][c:(c+8)])
            VZ[level][i]    =  mean(VZ[level+1][c:(c+8)])            
            # copy the same sigma value down to the children, if their sigma is still ==0.0
            mc              =  nonzero(SIGMA[level+1][c:(c+8)]<=0.0)
            SIGMA[level+1][c:(c+8)][mc] = sigma     # parent = sigma*k, child is sigma
    # save sigma values to the output file
    fp = open(sigmafile, 'wb')
    np.asarray([NX, NY, NZ, LEVELS, CELLS], np.int32).tofile(fp)
    for level in range(LEVELS):
        np.asarray([LCELLS[level],], np.int32).tofile(fp)
        np.asarray(SIGMA[level], np.float32).tofile(fp)
    fp.close()
    
    

    
def TurbulenceCL(rhofile, vxfile, vyfile, vzfile, sigmafile, GPU=1, PLF=[0,1,2,3,4]):
    """
    Estimate turbulence for every node in the hierarchy.
    Input:
        rhofile  =  OT density file with the hierarchy information
        v*file   =  OT files for the three velocity components
    Output:
        write OT file with velocity dispersions to sigmafile
    Warning:
        this was apparently never completed and/or the kernel no longet matches host code
    """    
    NX, NY, NZ, LEVELS, LCELLS, H  = OT_ReadHierarchy(rhofile)    
    NX, NY, NZ, LEVELS, LCELLS, VX = OT_ReadHierarchy(vxfile)  # originally exists only for leaf nodes
    NX, NY, NZ, LEVELS, LCELLS, VY = OT_ReadHierarchy(vyfile)    
    NX, NY, NZ, LEVELS, LCELLS, VZ = OT_ReadHierarchy(vzfile)        
    CELLS = sum(LCELLS)
    SIGMA = []
    for h in H:    # create SIGMA vectors
        SIGMA.append(zeros(len(h), np.float32))    
    ###
    platform, device, context, queue, mf = InitCL(GPU, platforms=PLF)
    LOCAL       =  [8,32][GPU>0]
    source      =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_turbulence.c').read()
    OPT         =  ''
    program     =  cl.Program(context, source).build(OPT)
    SU          =  program.SigmaUpdate
    SU.set_scalar_arg_dtypes([np.int32, None, None, None, None, None, None, None, None])
    NMAX        =  np.max(LCELLS)
    H_buf       =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(NMAX))
    SIGMAP_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(NMAX))
    SIGMAC_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(NMAX))
    VXP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(NMAX))
    VYP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(NMAX))
    VZP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(NMAX))
    VXC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(NMAX))
    VYC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(NMAX))
    VZC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(NMAX))
    ##
    for level in range(LEVELS-2, -1, -1):          # loop over parent levels
        print("LEVEL %d" % level)        
        cl.enqueue_copy(queue, H_buf,      np.asarray(H[level],       np.float32))        
        cl.enqueue_copy(queue, VXP_buf,    np.asarray(VX[level],      np.float32))
        cl.enqueue_copy(queue, VYP_buf,    np.asarray(VY[level],      np.float32))
        cl.enqueue_copy(queue, VZP_buf,    np.asarray(VZ[level],      np.float32))
        cl.enqueue_copy(queue, SIGMAP_buf, np.asarray(SIGMA[level],   np.float32))
        cl.enqueue_copy(queue, SIGMAC_buf, np.asarray(SIGMA[level+1], np.float32))
        cl.enqueue_copy(queue, VXC_buf,    np.asarray(VX[level+1],    np.float32))
        cl.enqueue_copy(queue, VYC_buf,    np.asarray(VY[level+1],    np.float32))
        cl.enqueue_copy(queue, VZC_buf,    np.asarray(VZ[level+1],    np.float32))
        SU(queue, [GLOBAL,], [LOCAL,], LCELLS[level], H_buf, 
        VXP_buf, VYP_buf, VZP_buf, SIGMAP_buf,        VXC_buf, VYC_buf, VZC_buf, SIGMAC_buf)
        cl.enqueue_copy(queue, SIGMA[level+1], SIGMAC_buf)
        cl.enqueue_copy(queue, SIGMA[level  ], SIGMAP_buf)
        cl.enqueue_copy(queue, VX[level],      VXP_buf)
        cl.enqueue_copy(queue, VY[level],      VYP_buf)
        cl.enqueue_copy(queue, VZ[level],      VZP_buf)
    # save sigma values to the output file
    fp = open(sigmafile, 'wb')
    np.asarray([NX, NY, NZ, LEVELS, CELLS], np.int32).tofile(fp)
    for level in range(LEVELS):
        np.asarray([LCELLS[level],], np.int32).tofile(fp)
        np.asarray(SIGMA[level], np.float32).tofile(fp)
    fp.close()
    

    

def Update_LOC_turbulence(NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ, GPU=1, PLF=[0,1,2,3,4]):
    """
    Update in-cell turbulence estimates for all cells in a LOC model.
    Input:
        NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ  =
                parameter vectors as returned by OT_ReadHierarchyV_LOC()
    Output:
        Updated values in the vector S for microturbulence.
    """    
    CELLS       =  sum(LCELLS)
    LEVELS      =  len(LCELLS)
    platform, device, context, queue, mf = InitCL(GPU, platforms=PLF)
    LOCAL       =  [8,32][GPU>0]
    GLOBAL      =  16384
    source      =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_turbulence.c').read()
    OPT         =  ''
    program     =  cl.Program(context, source).build(OPT)
    SU          =  program.SigmaUpdateLOC
    #                         N         H     VXP   VYP   VZP   SIGMAP VXC   VYC   VZC   SIGMAC
    SU.set_scalar_arg_dtypes([np.int32, None, None, None, None, None,  None, None, None, None])
    NMAX        =  np.max(LCELLS)
    H_buf       =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(NMAX))
    SIGMAP_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(NMAX))
    SIGMAC_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(NMAX))
    VXP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(NMAX))
    VYP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(NMAX))
    VZP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(NMAX))
    VXC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(NMAX))
    VYC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(NMAX))
    VZC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(NMAX))
    S[:]        = -1.0   # mark as missing
    ##
    a, b, c, d  = 0, 0, 0, 0
    for level in range(LEVELS-2, -1, -1):          # loop over parent levels
        print("LEVEL %d" % level)
        a, b  =  OFF[level],    OFF[level]  +LCELLS[level]     # parent level cells
        c, d  =  OFF[level+1],  OFF[level+1]+LCELLS[level+1]   # child level cells
        cl.enqueue_copy(queue, H_buf,      np.asarray(H[a:b],     np.float32))        
        cl.enqueue_copy(queue, VXP_buf,    np.asarray(VX[a:b],    np.float32))
        cl.enqueue_copy(queue, VYP_buf,    np.asarray(VY[a:b],    np.float32))
        cl.enqueue_copy(queue, VZP_buf,    np.asarray(VZ[a:b],    np.float32))
        cl.enqueue_copy(queue, SIGMAP_buf, np.asarray(S[a:b],     np.float32))
        ###
        cl.enqueue_copy(queue, SIGMAC_buf, np.asarray(S[c:d],     np.float32))
        cl.enqueue_copy(queue, VXC_buf,    np.asarray(VX[c:d],    np.float32))
        cl.enqueue_copy(queue, VYC_buf,    np.asarray(VY[c:d],    np.float32))
        cl.enqueue_copy(queue, VZC_buf,    np.asarray(VZ[c:d],    np.float32))
        SU(queue, [GLOBAL,], [LOCAL,], LCELLS[level], H_buf, 
        VXP_buf, VYP_buf, VZP_buf, SIGMAP_buf,        VXC_buf, VYC_buf, VZC_buf, SIGMAC_buf)
        cl.enqueue_copy(queue, S[a:b],     SIGMAP_buf)
        cl.enqueue_copy(queue, S[c:d],     SIGMAC_buf)
        cl.enqueue_copy(queue, VX[a:b],    VXP_buf)
        cl.enqueue_copy(queue, VY[a:b],    VYP_buf)
        cl.enqueue_copy(queue, VZ[a:b],    VZP_buf)
        if (1):
            print("  S[%d] " % level, np.min(S[a:b]), np.max(S[a:b]))
    # now S (and also VX, VY, VZ parent cells) have been updated
    # Only root grid parent cells still miss estimates
    print("LEVEL 0 missing values (%d - %d)" % (a, b))
    SR          =  program.SigmaUpdateLOC_root
    GLOBAL      =  (b//64+1)*64
    #                         NX        NY        NZ        H     VXP   VYP   VZP   SIGMAP
    SR.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None, None, None   ])
    SR(queue, [GLOBAL,], [LOCAL,], NX, NY, NZ, H_buf, VXP_buf, VYP_buf, VZP_buf, SIGMAP_buf)
    cl.enqueue_copy(queue, S[a:b],   SIGMAP_buf)
    cl.enqueue_copy(queue, VX[a:b],  VXP_buf)
    cl.enqueue_copy(queue, VY[a:b],  VYP_buf)
    cl.enqueue_copy(queue, VZ[a:b],  VZP_buf)
    
    


def Reorder(x, y, z, x0, dx, GPU=0, platforms=[0,1,2,3,4]):
    """
    Given cell coordinates (x,y,z) and knowledge that coordinates are
    x0 + i*dx, return an index vector such that all cells in an octet
    are consecutive and in SID order.
    Because some cells in an octet are leafs and some are not, they
    may originally be in any order in the input vectors.
    """    
    t0       =  time.time()
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    print("OT_library -- GPU, platforms", GPU, platforms)
    print("           -- Reorder: platform, device, context ", platform, device, context)
    LOCAL    = [ 8, 64 ][GPU>0]
    N        =  len(x)
    if (N % 8 != 0):
        print('???')
        sys.exit()
    OPT      =  " -D N=%d  -D OFF=%.8ef  -D STEP=%.8ef -D USE_GPU=%d" % (N, x0, dx, GPU)
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_RamsesHierarchy.c').read()
    program  =  cl.Program(context, source).build(OPT)
    SID0     =  np.zeros(len(x), np.int32)
    SID0_buf =  cl.Buffer(context, mf.READ_WRITE, SID0.nbytes)
    x, y, z  =  np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(z, np.float32)
    X_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    Y_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
    Z_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z)
    
    # First kernel -- find sid==0 cells  => SID==1
    Sid      =  program.FindSID0
    Sid.set_scalar_arg_dtypes([None, None, None, None]) # (x, y, z) coordinates, SID vector
    GLOBAL   = ((len(x)//64)+1)*64           # one work item per cell!!
    Sid(queue, [GLOBAL,], [LOCAL,], X_buf, Y_buf, Z_buf, SID0_buf)
    cl.enqueue_copy(queue, SID0, SID0_buf)   # updated SID0 indices, 0 or 1
        
    # Extract the indices of all sid==0 cells
    m             = nonzero(SID0==1)       # indices of sid==0 cells in the original vectors
    SID0[0:(N//8)] = m[0].copy()            # use first N/8 elements of the same SID0 array
    cl.enqueue_copy(queue, SID0_buf, SID0) # indices of sid==0 cells in the original vectors
    print('sid==0 cells %d out of %d, ratio %7.3f' % (len(m[0]), N, N/float(len(m[0]))))
    
    # Second kernel -- given a vector of indices of sid==0 cells, 
    # put all cells to SID in their correct order according to SID (sub-index within each octet)
    SID      =  np.zeros(N, np.int32)
    SID_buf  =  cl.Buffer(context, mf.WRITE_ONLY, SID.nbytes)    
    Sid      =  program.SID_order_sorted
    Sid.set_scalar_arg_dtypes([None, None, None, None, None]) # (x, y, z) coordinates, SID0, SID vector
    GLOBAL   = (((N//8)//64)+1)*64             # one work item per octetd (per SID==0 cell)
    print(GLOBAL, LOCAL)
    Sid(queue, [GLOBAL,], [LOCAL,], X_buf, Y_buf, Z_buf, SID0_buf, SID_buf)
    cl.enqueue_copy(queue, SID, SID_buf)     # updated SID indices

    # return index vector
    return SID
        
    



        
    
def DumpToSOC(INDEX, DUMP, FILENAME, GPU=0, platforms=[0,1,2,3,4], scaling=1.0):
    """
    Dump plain binary files exported from Ramses into SOC hierarchical file.
    Input:
        INDEX     =  file created from Ramses hierarchy by RamsesToIndex()
        DUMP      =  a single variable dump, written by RamsesDump()
        FILENAME  =  name of the output SOC file
        GPU       =  >0 if using GPU instead of CPU
        scaling   =  scale values with this number, default=1.0
    SOC file format:
        DIM1, DIM2, DIM3, LEVELS, sum(CELLS)
        each level:
            LCELLS, { values}
    Routine simply copies np.int32 values (links!!) to a new file where data are np.float32, 
    then adding data read from DUMP.
    Note:  Indices in INDEX are 1,2,3,...  DUMP starts with the number of entries
    Links are changed to    -*(float*)&abs(i) !!!
    """
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    LOCAL    =  [ 8, 32 ][GPU>0]
    ###
    X        = fromfile(DUMP, np.float32)[1:]
    if (scaling!=1.0):  X *= scaling
    fp       = open(INDEX, 'rb')
    fpo      = open(FILENAME, 'wb')
    DIM1, DIM2, DIM3, LEVELS, CELLS = fromfile(fp, np.int32, 5)
    np.asarray([DIM1, DIM2, DIM3, LEVELS, CELLS], np.int32).tofile(fpo)
    LCELLS   = np.zeros(LEVELS, np.int32)
    H        = []
    for i in range(LEVELS):
        LCELLS[i] = fromfile(fp, np.int32, 1)[0]
        H.append(fromfile(fp, np.int32, LCELLS[i]))   # indices int32 !!
        print('... reading level %d,  cells %d' % (i, LCELLS[i]))
    print('Total of %d cells' % sum(LCELLS))
    # Now the hierarchy has been read, the data vector (ramses dump) has been read
    #  NX, NY, NZ are here dummy values !
    OPT      = " -D N=%d -D NX=1 -D NY=1 -D NZ=1 -D LEVELS=1" % len(X)  #  N = length of data vectors (=dumped parameters)
    source   = open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program  = cl.Program(context, source).build(OPT)
    CONVERT  = program.Convert
    CONVERT.set_scalar_arg_dtypes([np.int32, None, None, None])
    GLOBAL   = (np.max(LCELLS)//LOCAL+1)*LOCAL  # maximum number of cells per level
    print("DumpToSOC  LOCAL=%d, GLOBAL=%d ... %.3f" % (LOCAL, GLOBAL, GLOBAL/LOCAL))
    # One level at a time, convert INDEX vector and DUMP array into DATA vector
    #    DATA = np.float32 data values and   *(int *)&(-DATA[i]) links
    #    ==> result will be SOCAMO file in NEWLINK format
    X_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X) # rho, Bx, ... np.float32
    H_buf    = cl.Buffer(context, mf.READ_ONLY,  GLOBAL*4)  # index for one level  int32
    D_buf    = cl.Buffer(context, mf.WRITE_ONLY, GLOBAL*4)  # final data vector    np.float32
    DDD      = np.zeros(GLOBAL, np.float32)
    HHH      = np.zeros(GLOBAL, np.int32)
    for i in range(LEVELS):
        cells  =  LCELLS[i]
        HHH[0:cells] = H[i]
        cl.enqueue_copy(queue, H_buf, HHH)    # index vector for current level
        CONVERT(queue, [GLOBAL,], [LOCAL,], cells, X_buf, H_buf, D_buf)
        cl.enqueue_copy(queue, DDD, D_buf)    # data vector for current level
        # write data for the current level
        np.asarray([cells,], np.int32).tofile(fpo)
        np.asarray(DDD[0:cells], np.float32).tofile(fpo)
    fp.close()
    fpo.close()
        

    
def DumpToLOC_(INDEX, PREFIX, FILENAME, T=10.0, SIGMA=1.0, ABUNDANCE=1.0e-6, GPU=0, platforms=[0,1,2,3,4]):
    """
    Dump plain binary files exported from Ramses into SOCAMO hierarchical file.
    Input:
        INDEX     =  file created from Ramses hierarchy by RamsesToIndex()
        PREFIX    =  prefix for dump files of single variable dump, written by RamsesDump()
                     should have <PREFIX>.rho, <PREFIX>.vx, <PREFIX>.vy ,<PREFIX>.vz
        FILENAME  =  name of the output SOCAMO file
        GPU       =  >0 if using GPU instead of CPU
    SOC file format:
        DIM1, DIM2, DIM3, LEVELS, sum(CELLS)
        each level:
            LCELLS, { densitry_values}
    LOC cloud is the same but, instead of only n, has sections:
        n, T, sigma, vx, vy, vz, abundance
    THIS ROUTINE LEAVES (T, SIGMA, ABUNDANCE) AS CONSTANT VALUES OVER ALL CELLS.
    """
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    LOCAL    =  [ 8, 32 ][GPU>0]
    ###
    X        = fromfile('%s.rho' % PREFIX, np.float32)[1:]
    fp       = open(INDEX, 'rb')
    DIM1, DIM2, DIM3, LEVELS, CELLS = fromfile(fp, np.int32, 5)
    fpo      = open(FILENAME, 'wb')
    np.asarray([DIM1, DIM2, DIM3, LEVELS, CELLS], np.int32).tofile(fpo)
    LCELLS   = np.zeros(LEVELS, np.int32)
    H        = []
    for i in range(LEVELS):
        LCELLS[i] = fromfile(fp, np.int32, 1)[0]
        H.append(fromfile(fp, np.int32, LCELLS[i]))   # indices np.int32 !!
        print('... reading level %d,  cells %d' % (i, LCELLS[i]))
    fp.close()
    print('Total of %d cells' % sum(LCELLS))
    # Now the hierarchy has been read, the data vector (ramses dump) has been read
    #  NX, NY, NZ are here dummy values !
    OPT      = " -D N=%d -D NX=1 -D NY=1 -D NZ=1 -D LEVELS=1" % len(X)  #  N = length of data vectors (=dumped parameters)
    source   = open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_tools.c').read()
    program  = cl.Program(context, source).build(OPT)
    CONVERT  = program.Convert
    CONVERT.set_scalar_arg_dtypes([np.int32, None, None, None])
    GLOBAL   = np.max(LCELLS)  # maximum number of cells per level
    # One level at a time, convert INDEX vector and DUMP array into DATA vector
    #    DATA = np.float32 data values and   *(int *)&(-DATA[i]) links
    #    ==> result will be SOCAMO file in NEWLINK format
    X_buf    = cl.Buffer(context, mf.READ_ONLY,  4*np.int64(len(X))) # rho, Bx, ... np.float32
    H_buf    = cl.Buffer(context, mf.READ_ONLY,  GLOBAL*4)  # index for one level  np.int32
    D_buf    = cl.Buffer(context, mf.WRITE_ONLY, GLOBAL*4)  # final data vector    np.float32
    DDD      = np.zeros(GLOBAL, np.float32)
    HHH      = np.zeros(GLOBAL, np.int32)
    # RHO
    cl.enqueue_copy(queue, X_buf, X)
    for i in range(LEVELS):
        cells        =  LCELLS[i]
        HHH[0:cells] =  H[i]
        cl.enqueue_copy(queue, H_buf, HHH)    # index vector for current level
        CONVERT(queue, [GLOBAL,], [LOCAL,], cells, X_buf, H_buf, D_buf)
        cl.enqueue_copy(queue, DDD, D_buf)    # data vector for current level
        np.asarray([cells,], np.int32).tofile(fpo)
        np.asarray(DDD[0:cells], np.float32).tofile(fpo)
    # 
    for ifield in range(6):
        if   (ifield==0):   X = 0.0*X+T                                    # temperature
        elif (ifield==1):   X = 0.0*X+SIGMA                                # SIGMA
        elif (ifield==2):   X = fromfile('%s.vx' % PREFIX, np.float32)[1:]    # vx
        elif (ifield==3):   X = fromfile('%s.vy' % PREFIX, np.float32)[1:]    # vy
        elif (ifield==4):   X = fromfile('%s.vz' % PREFIX, np.float32)[1:]    # vz
        elif (ifield==2):   X = 0.0*X+ABUNDANCE                            # abundance
        cl.enqueue_copy(queue, X_buf, X)
        for i in range(LEVELS):
            cells        =  LCELLS[i]
            HHH[0:cells] =  H[i]
            cl.enqueue_copy(queue, H_buf, HHH)    # index vector for current level
            CONVERT(queue, [GLOBAL,], [LOCAL,], cells, X_buf, H_buf, D_buf)
            cl.enqueue_copy(queue, DDD, D_buf)    # data vector for current level
            np.asarray([cells,], np.int32).tofile(fpo)
            np.asarray(DDD[0:cells], np.float32).tofile(fpo)
    ###
    fpo.close()
        
    

    
def MakeLOC(PREFIX, GPU=0, TKIN=15.0, rhoscale=1.0):
    """
    Read files dumped from Ramses and make a cloud file for LOC.
    Input:
        PREFIX   =  prefix for input files, e.g. <PREFIX>.vx
        GPU      =  if > 0, use GPU
        TIN      =  constant value for Tkin
        rhoscale =  scaling applied to density values, default = 0.5 (!),
                    assuming that the SOC cloud is being used and that
                    has densities n(H) while LOC will have n(H2)
                    Correct rho scaling is needed also to set correct abundances.
                    === Density and velocities should be scaled already 
                        in calls to DumpToSOC(), here only *= 0.5 for density
    Note:
        * One must already have the SOC cloud as <PREFIX>.soc
          ... in DUMP_RAMSES.py, dump_*.rho, dump_*.vx etc. are already converted to *.soc, *.vx
        * One must already have made the sigma file using Turbulence() --- NO, THAT WILL BE
          CALCULATED IN OT_PropagateVelocity() !!
        * Temperature is set to a constant value 15K
        * Abundances are calculated for 12CO, with a density-dependent function
    """
    nx, ny, nz, lcells, off, h      =  OT_ReadHierarchyV(PREFIX+'.soc')   # needed for index calculations !
    m     =  nonzero(h>0.0)
    if (rhoscale!=1.0):
        print("*** DENSITY SCALED WITH %.3e" % rhoscale)
        h[m] *=  rhoscale
    nx, ny, nz, lcells, off, vx     =  OT_ReadHierarchyV(PREFIX+'.vx')    # RamsesDumpField did zyx -> xyz for velo
    nx, ny, nz, lcells, off, vy     =  OT_ReadHierarchyV(PREFIX+'.vy')
    nx, ny, nz, lcells, off, vz     =  OT_ReadHierarchyV(PREFIX+'.vz')
    sigma  =  -1.0*np.ones(len(vx), np.float32)
    
    # We need (vx, vy, vz, s) where EVERY CELL contains data values, not only the leafs.
    # Fill in this information here, parent cell gets the average velocity over children, 
    # the sigma is based on the velocity dispersion over children
    vx, vy, vz, sigma = OT_PropagateVelocity(nx, ny, nz, lcells, off, h, vx, vy, vz, sigma, GPU=0, fill_sigma=0.3)
    
    if (1):
        levels = len(lcells)
        bad = False
        for l in range(levels):
            m     = nonzero(h[off[l]:(off[l]+lcells[l])]>0.0)
            mbad  = nonzero(~isfinite(sigma[off[l]:(off[l]+lcells[l])][m]))
            print("SIGMA NOT FINITE: LEVEL %2d =>  %d CELLS" % (l, len(mbad[0])))
            if (len(mbad[0])>0): bad = True
        if (bad): sys.exit()
        
    # these are the parameter vectors for the SOURCE HIERARCHY
    # SAVE AS NEW OT FILES ....  "full" meaning that every node contains data (no links)
    OT_WriteHierarchyV(nx, ny, nz, lcells, off, vx,    PREFIX+'_full.vx')
    OT_WriteHierarchyV(nx, ny, nz, lcells, off, vy,    PREFIX+'_full.vy')
    OT_WriteHierarchyV(nx, ny, nz, lcells, off, vz,    PREFIX+'_full.vz')
    OT_WriteHierarchyV(nx, ny, nz, lcells, off, sigma, PREFIX+'_full.sigma')
    # read again the original SOC for hierarchy
    NX, NY, NZ, LCELLS, OFF, H = OT_ReadHierarchyV(PREFIX+'.soc')
    CELLS  = sum(LCELLS)
    LEVELS = len(LCELLS)
    if (rhoscale!=1.0):
        print("*** DENSITY SCALED WITH %.3e" % rhoscale)
        H[m] *=  rhoscale
    # TARGET parameter vectors in upper case
    VX, VY, VZ   =  np.zeros(CELLS, np.float32), np.zeros(CELLS, np.float32), np.zeros(CELLS, np.float32)
    SIGMA        =  np.zeros(CELLS, np.float32)         # value must be zero initially (for bookkeeping), calculated below
    T            =  TKIN * np.ones(CELLS, np.float32)   # eventually from Tdust
    # CHI =  from the Glover & Clark paper, abundances for 12CO !!!
    n   =  logspace(-3.0, 10.0, 2000)
    # V2  =  1e-4* n**2.0  / (5.0e5+n**2.0)
    V3  =  1e-4* n**2.45 / (3.0e8+n**2.45)
    # V4  =  1e-4* 16.0*n**2.0 / (5.0e5 + 16.0*n**2.0)
    # lets use version V3
    t0  =  time.time()
    ip  =  interp1d(n, V3, fill_value=1.0e-12, bounds_error=False)
    CHI =  clip(asarray(ip(H), np.float32), 1.0e-12, 1.0)
    print("Abundance calculated in %.3f seconds" % (time.time()-t0))
    for level in range(LEVELS):  # loop over hierarchy levels (to reduce memory usage)
        a, b        =  OFF[level], OFF[level]+LCELLS[level]
        if (0): # ???????????
            # complex code to copy between similar shaped arrays...
            # and does not work because h no longer has links !!!
            x, y, z     =  OT_GetCoordinatesLevelV(NX, NY, NZ, LCELLS, OFF, H, level, GPU=GPU)
            # based on coordinates in TARGET, get indices to the SOURCE  => not necessarily leaf nodes in the source hierarchy
            ind         =  OT_GetIndicesV(x, y, z, nx, ny, nz, lcells, off, h, max_level=level, GPU=0, global_index=True)
            VX[a:b]     =  vx[ind]
            VY[a:b]     =  vy[ind]
            VZ[a:b]     =  vz[ind]
            SIGMA[a:b]  =  sigma[ind]
        else:
            VX[a:b]     =  vx[a:b]
            VY[a:b]     =  vy[a:b]
            VZ[a:b]     =  vz[a:b]
            SIGMA[a:b]  =  sigma[a:b]
            
    # Write cloud file
    fp = open(PREFIX+'.loc', 'wb')
    np.asarray([NX, NY, NZ, LEVELS, CELLS], np.int32).tofile(fp)
    for F in [ H, T, SIGMA, VX, VY, VZ, CHI ]:
        for level in range(LEVELS):
            a, b    =  OFF[level], OFF[level] + LCELLS[level]
            np.asarray([LCELLS[level],], np.int32).tofile(fp)
            print('%10d     %10d - %10d' % (LCELLS[level], a, b))
            F[a:b].tofile(fp)
    fp.close()
                                                
    

    
    
    
def AverageTemperatureFiles(names, avename):
    """
    AverageTemperatureFiles(names, avename)
    Take a list of SOC dust temperature files and store the average to a new file.
    Input:
        names    =  list of temperature files
        avename  =  name of the new file for the average temperature
    """
    if (os.path.exists(avename)):
        print('AverageTemperatureFiles: target file %s exists' % avename)
        print(' *** ABORT ***')
        return
    # temperature file format is the same as for density
    #   NX, NY, NZ
    #   LEVELS, CELLS
    #   LCELLS, {T}
    fp     =  open(names[0], 'rb')
    dims   =  fromfile(fp, np.int32, 5)
    NX, NY, NZ, LEVELS, CELLS = dims
    LCELLS = np.zeros(LEVELS, np.int32)
    AVE = []
    for i in range(LEVELS):
        lcells     =  fromfile(fp, np.int32, 1)[0]
        LCELLS[i]  =  lcells
        AVE.append(fromfile(fp, np.float32, lcells))
    fp.close()
    count = 1
    # now AVE contains "sum" of count temperature files
    for name in names[1:]: # the rest of the files
        fp = open(name, 'rb')
        nx, ny, nz, levels, cells = fromfile(fp, np.int32, 5)
        if ((nx!=NX)|(ny!=NY)|(nz!=NZ)|(levels!=LEVELS)):
            print('AverageTemperatureFiles: inconsistent dimensions')
            print('%3d %3d %3d %3d for %s' % (NX, NY, NZ, LEVELS, names[0]))
            print('%3d %3d %3d %3d for %s' % (nx, ny, nz, levels, name))
            print(' *** ABORT ***')
            return
        ###
        for i in range(LEVELS):
            lcells = fromfile(fp, np.int32, 1)[0]
            if (lcells!=LCELLS[i]):
                print('AverageTemperatureFiles: error in lcells')
                print('First file LCELLS[%d]=%d -- %s' % (i, LCELLS[i], names[0]))
                print('This file  LCELLS[%d]=%d -- %s' % (i, lcells, name))
                print(' *** ABORT ***')
                return
            x = fromfile(fp, np.float32, lcells)
            AVE[i] += x
        fp.close()
        ###
        count += 1
    ###
    fp = open(avename, 'wb')
    np.asarray(dims, np.int32).tofile(fp)
    for i in range(LEVELS):
        np.asarray([LCELLS[i],], np.int32).tofile(fp)
        np.asarray(AVE[i]/count, np.float32).tofile(fp)
    fp.close()
    return
        
                
        
        
def RollCloud(oldname, newname, dx=0, dy=0, dz=0):
    """
    Roll the cloud moving (dx, dy, dz) root grid cells along each of the coordinate axes.
    Usage:
        RollCloud(oldname, newname, dx=0, dy=0, dz=0, LOC=False)
    Input:
        oldname     =  name of an existing octree cloud file (SOC or LOC !!)
        newname     =  name of the file where the rolled cloud will be saved
        dx, dy, dz  =  amount of roll, in number of root grid cells along x, y, and z axes
        LOC         =  if True, assume it is a LOC file with 7 parameters
    Result:
        file newname will contain the rolled octree cloud
    """
    fp     =  open(oldname, 'rb')
    NX, NY, NZ, LEVELS, CELLS = fromfile(fp, np.int32, 5)    
    fpout  =  open(newname, 'wb')
    np.asarray([NX, NY, NZ, LEVELS, CELLS], np.int32).tofile(fpout)    
    for ipar in range(7):     # one parameter for SOC, seven for LOC = [ H, T, S, VX, VY, VZ, C ]
        for ilevel in range(LEVELS):
            cells = fromfile(fp, np.int32, 1)
            if (len(cells)<1): break   # no more parameters in the file
            data  = fromfile(fp, np.float32, cells[0])
            if (ilevel==0):            # rolling affects only the root level
                data.shape = (NZ, NY, NX)                
                data = np.roll(data.copy(), dz, 0)
                data = np.roll(data.copy(), dy, 1)
                data = np.roll(data.copy(), dx, 2)                
            cells.tofile(fpout)
            data.tofile(fpout)
    fp.close()
    fpout.close()
                
    

    
    
def meta_reset_levels(meta_in, meta_out, maxl):
    """
    Write new meta file, only changing the maximum number of levels.
    """
    print('meta_reset_levels ', meta_in, meta_out, maxl)
    levelmin = 0
    fp = open(meta_out, 'w')
    for line in open(meta_in).readlines():
        s = line.split()
        if (len(s)<2): continue
        if (s[0]=='levelmin'): levelmin = int(s[1])
        if (s[0]=='levelmax'):
            fp.write('levelmax    %d\n' % (levelmin+maxl))
        elif (s[0]=='levels'):
            fp.write('levels      %d\n' % (maxl+1))
        elif (s[0]=='maxrefine'):
            fp.write('maxrefine   %d\n' % (2**maxl))
        else:
            fp.write(line)
    fp.close()
        
    


def I2F_CL(I, GPU=0, PLF=[0,1,2,3,4]):
    """
    Convert integer vector into float vector. As I2F but for vectors and using kernel.
    Input:
       I     =   integer vector (np.int32)
       GPU   =   if GPU>0, use GPU
       PLF   =   list of OpenCL platforms (numbers)
    Return:
       F     =   np.float32 vector with values  F[i] = *(float *)&(I[i])
    """
    N         =  len(I)
    platform, device, context, queue, mf = InitCL(GPU, PLF, verbose=True) # OpenCL environment
    LOCAL     =  [4, 32][GPU>0]     # local work group size
    GLOBAL    =  8192*LOCAL
    OPT       = '-D N=%d' % N
    # source    =  open(HOMEDIR+"/starformation/Python/MJ/MJ/Aux/kernel_I2F_F2I.c").read()
    source    =  KernelSource("kernel_I2F_F2I.c")
    program   =  cl.Program(context, source).build(OPT)
    ####
    I_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(N))
    F_buf     =  cl.Buffer(context, mf.WRITE_ONLY, 4*np.int64(N))
    Fun       =  program.I2F
    cl.enqueue_copy(queue, I_buf, I)
    Fun(queue, [GLOBAL,], [LOCAL,], I_buf, F_buf)
    F         =  np.zeros(N, np.float32)
    cl.enqueue_copy(queue, F, F_buf)
    return F



def F2I_CL(F, GPU=0, PLF=[0,1,2,3,4]):
    """
    Convert float vector into integder vector. As F2I but for vectors and using kernel.
    Input:
       F     =   vector of np.float32 values
       GPU   =   if GPU>0, use GPU
       PLF   =   list of OpenCL platforms (numbers)
    Return:
       I     =   np.int32 vector with values  I[i] = *(int *)&(F[i])
    """
    N         =  len(F)
    platform, device, context, queue, mf = InitCL(GPU, PLF, verbose=True) # OpenCL environment
    LOCAL     =  [4, 32][GPU>0]     # local work group size
    GLOBAL    =  8192*LOCAL
    OPT       = '-D N=%d' % N
    # source    =  open(HOMEDIR+"/starformation/Python/MJ/MJ/Aux/kernel_I2F_F2I.c").read()
    source    =  KernelSource("kernel_I2F_F2I.c")
    program   =  cl.Program(context, source).build(OPT)
    F_buf     =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=F)
    I_buf     =  cl.Buffer(context, mf.WRITE_ONLY,  4*np.int64(N))
    Fun       =  program.F2I
    Fun(queue, [GLOBAL,], [LOCAL,], F_buf, I_buf)
    I         =  np.ones(N, np.int32)
    cl.enqueue_copy(queue, I, I_buf)
    return I

    
    
def OT_points_to_octree(x, y, z, NX, NY, NZ, LEVELS, filename, GPU=0, PLF=[0,1,2,3,4]):
    """
    Make an octree hierarchy based on point data (x, y, z)
    Usage:
        OT_points_to_octree(x, y, z, NX, NY, NZ, LEVELS, filename, GPU=1, PLF=[0,1,2,3,4])
    Input:
        x, y, z    = vectors of point coordinates, in the target octree file root coordinates (0-NX etx.)
        NX, NY, NZ = root grid dimensions for the octree file to be built
        LEVELS     = number of levels in the target octree hierarchy
        filename   = name for the resulting octree file
        GPU        = if GPU>0, use GPU instead of CPU (default GPU=0)
        PLF        = list of possible OpenCL platforms (default PLF=[01,2,3,4])
    Note: 
        If the input points (x,y,z) do not require all LEVELS hierarchy levels, the
        number of levels in the resulting file may be smaller
    """
    print("***** OT_points_to_octree started === %d x %d x %d" % (NX, NY, NZ))
    t000      =  time.time()
    NP        =  np.int64(len(x))
    OFF       =  np.zeros(LEVELS, np.int32)
    LCELLS    =  np.zeros(LEVELS, np.int32)
    LCELLS[0] =  NX*NY*NZ
    # apart from level 0, number of additional parent cells = number of cells / 8
    # LCELLS is not known yet, only that the number of leaf cells is NP
    # each eight leaf cells has one parent cell... but each 8 parent cells may also ave one grandparent
    # => if all cells were on level MAXL, maximum total number of cells would be
    ALLOC     =   np.int64(1.0*NP)
    for l in range(1, LEVELS):
       ALLOC+=   1.0*NP / (8.0**l)
    ALLOC     =  int(ALLOC+128)
    H         =  np.ones(ALLOC, np.float32)   # enough space for all cells
    ####
    platform, device, context, queue, mf = InitCL(GPU, PLF, verbose=False) # OpenCL environment
    LOCAL     =  [4, 32][GPU>0]     # local work group size
    GLOBAL    =  4096*LOCAL
    OPT       = '-D NX=%d -D NY=%d -D NZ=%d -D NP=%d' % (NX, NY, NZ, NP)
    # source    =  open(HOMEDIR+"/starformation/Python/MJ/MJ/Aux/kernel_points_to_octree.c").read()
    source    =  KernelSource("kernel_points_to_octree.c")
    program   =  cl.Program(context, source).build(OPT)
    ####
    print("***** NUMBER OF POINTS  N = %d = %.2e, ALLOC = %d = %.2e  < 1.5*NP = %.2e" % (NP, NP, ALLOC, ALLOC, 1.5*NP))
    OFF_buf   =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS) 
    LCELLS_buf=  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS) 
    CCX_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*ALLOC)     # cell centres
    CCY_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*ALLOC)     # cell centres
    CCZ_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*ALLOC)     # cell centres
    PX_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*NP)
    PY_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*NP)
    PZ_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*NP)
    H_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*ALLOC)       # cannot be more cells than input points
    # Find cells that are to be refined -- marked with H<0
    # z must be in increasing order !!!
    Split =  program.Split_zorder
    #                            L         PX,   PY,   PZ    H     CCX   CCY   CCZ   OFF   LCELLS
    Split.set_scalar_arg_dtypes([np.int32, None, None, None, None, None, None, None, None, None   ])    
    # Add the next level  L -> L+1, update cell centre coordinates for the child
    AddLevel  = program.AddLevel
    #                               L         H     CC    CC    CC    OFF  LCELLS
    AddLevel.set_scalar_arg_dtypes([np.int32, None, None, None, None, None, None  ])    
    ###
    cl.enqueue_copy(queue, PX_buf,     np.asarray(x, np.float32))
    cl.enqueue_copy(queue, PY_buf,     np.asarray(y, np.float32))
    cl.enqueue_copy(queue, PZ_buf,     np.asarray(z, np.float32))
    cl.enqueue_copy(queue, LCELLS_buf, LCELLS)
    cl.enqueue_copy(queue, OFF_buf,    OFF)
    cl.enqueue_copy(queue, H_buf,      H)
    # all H values are set in Split
    # To initialise root grid coordinates in CC
    RootCoord = program.SetRootCoordinates
    #                                CCX   CCY   CCZ    LCELLS
    RootCoord.set_scalar_arg_dtypes([None, None, None, None])
    RootCoord(queue, [GLOBAL,], [LOCAL,], CCX_buf, CCY_buf, CCZ_buf, LCELLS_buf)

    
    for L in range(LEVELS-1):   #  refine from L -> L+1
        
        print("*****. level %d/%d   %9d - %9d   %9d cells" % (L, LEVELS-1, OFF[L], OFF[L]+LCELLS[L], LCELLS[L]))
        t0 = time.time()        
        # *** Split ***
        print("***** Split ...")
        Split(queue, [GLOBAL,], [LOCAL,], L, PX_buf, PY_buf, PZ_buf, H_buf, CCX_buf, CCY_buf, CCZ_buf, OFF_buf, LCELLS_buf)
        # set H of the split cells to the running index over split cells * -1
        cl.enqueue_copy(queue, H, H_buf)
        queue.finish()
        print("***** Split... done")

        tmp         =  H[OFF[L]:(OFF[L]+LCELLS[L])]
        m           =  nonzero(tmp<=0.0)        # these level L PARENT cells are marked to be split
        LCELLS[L+1] =  8*len(m[0])              # this many CHILD cells on the next level L+1
        OFF[L+1]    =  OFF[L] + LCELLS[L]

        # one cannot put directly -index into H, because float has less significant digits
        # =>  encode  these as  *(float *)(&index) => use -I2F
        I      =   np.asarray(8*arange(len(m[0])), np.int32)  # directly first index of child octet
        F      =   I2F_CL(I, GPU, PLF)
        tmp[m] =  -F
        
        H[OFF[L]:(OFF[L]+LCELLS[L])]  =  tmp    # PARENT cells updated to indicate which will be split
        cl.enqueue_copy(queue, H_buf,      H)
        cl.enqueue_copy(queue, LCELLS_buf, LCELLS)
        cl.enqueue_copy(queue, OFF_buf,    OFF)
        queue.finish()        
        # *** AddLevel ***
        print("***** AddLevel....")
        AddLevel(queue, [GLOBAL,], [LOCAL,], L, H_buf, CCX_buf, CCY_buf, CCZ_buf, OFF_buf, LCELLS_buf)
        queue.finish()        
        print("***** AddLevel.... done")
        
    # run Split on the last level, to fill in 1+index_to_yt_vector in the MAXL leaf nodes
    Split(queue, [GLOBAL,], [LOCAL,], LEVELS-1, PX_buf, PY_buf, PZ_buf, H_buf, CCX_buf, CCY_buf, CCZ_buf, OFF_buf, LCELLS_buf)
    cl.enqueue_copy(queue, H, H_buf)

    # Check if LEVELS has decreased
    m      = nonzero(LCELLS>0)
    LCELLS = LCELLS[m]
    OFF    = OFF[m]
    LEVELS = len(LCELLS)
    
    # write the hierarchy
    fp = open(filename, 'wb')
    np.asarray([NX, NY, NZ, len(LCELLS), sum(LCELLS)], np.int32).tofile(fp)
    for i in range(LEVELS):
        np.asarray([LCELLS[i],], np.int32).tofile(fp)
        np.asarray( H[  (OFF[i]) : (OFF[i]+LCELLS[i])  ], np.float32).tofile(fp)
    fp.close()
                                           
    print("   === OT_points_to_octree finished in %.1f seconds ===" % (time.time()-t000))
        

    
    
def OT_energy(filename, tag, GL, ind=[], BX_file='', BY_file='', BZ_file='', GPU=0, PF=[0, 1, 2, 3, 4], 
              kdensity=1.0, kvelocity=1.0, ktemperature=1.0, kB=1.0, v0=[], CLOUD=None):
    """
    Compute energy terms for a LOC cloud.
    Input:
        filename   =  name of a LOC cloud file in octree format
        tag        =  which energy to compute: G = gravity, K = kinetic, P = external pressure, M = magnetic
                      in case of M, arguments BX_file, BY_file, and BZ_file must also be given
        GL         =  length of root grid cell [pc]
        ind        =  optional list of cell indices of leaf cells inside requested volume
        BX_file, BY_file, BZ_file = field of magnetic flux density, needed only for tag='M'
        kdensity   =  scaling to convert file density values to n(H2), default is 1.0
        kvelocity  =  scaling to convert file velocity and microturbulence values to km/s, default is 1.0
        ktemperature = scaling to convert file temperatures to Kelvin degrees
        kB         =  scaling to convert B values to Gauss, default is 1.0
        v0         =  optional array [vx0, vy0, vz0] specifying the rest velocity (for kinetic energy) [km/s]
        GPU        =  if >0, use GPU instead of CPU, default is 0
        PF         =  list of possible OpenCL platforms, default is [0, 1, 2, 3, 4 ]
        CLOUD      =  optional dictionary with NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ, CHI,
                      if given, filename is ignored
    Return:
        energy [erg], mass [MSun], volume [pc^3], inflow [Msun/a]
    Note:
        input file / CLOUD is expected to have densities in n(H2) (if kdensity==1)
        inflow is calculate only for tag='P', for others zero is returned
    """
    if (CLOUD==None):
        NX, NY, NZ, LCELLS, OFF, H0, T, S, VX, VY, VZ, CHI  =  OT_ReadHierarchyV_LOC(filename)
    else:
        NX, NY, NZ, LCELLS, OFF, H0, T, S, VX, VY, VZ, CHI  = \
        CLOUD['NX'], CLOUD['NY'], CLOUD['NZ'], CLOUD['LCELLS'], CLOUD['OFF'], CLOUD['H'], CLOUD['T'], CLOUD['S'], CLOUD['VX'], CLOUD['VY'], CLOUD['VZ'], CLOUD['CHI']
    ## 
    LEVELS, CELLS  =  len(LCELLS), sum(LCELLS)
    del CHI                            # not needed...
    X, Y, Z = [], [], []               # needed only for 'G' and 'P'
    L       =  np.zeros(len(H0), int8)    # helper array to keep track of hierarchy level
    for l in range(LEVELS):
        L[OFF[l]:(OFF[l]+LCELLS[l])] = l
    m = None
    if (len(ind)>0): m = ind               # selected region only
    else:            m = nonzero(H0>0.0)   # all leaf cells
    if (tag in ['G', 'P']):
        X, Y, Z  =  OT_GetCoordinatesAllV(NX, NY, NZ, LCELLS, OFF, H0, GPU=GPU, platforms=PF)
        X, Y, Z  =  X[m], Y[m], Z[m]  # selected cells only
    L          =   L[m]
    H          =  H0[m]    # keep H0 for later coordinate transformations
    T          =   T[m]
    S          =   S[m]
    if (tag in ['P', 'K']):
        VX, VY, VZ = VX[m], VY[m], VZ[m]
        if (kvelocity!=1.0):
            VX *= kvelocity
            VY *= kvelocity
            VZ *= kvelocity
            S  *= kvelocity
    else:  # G, M need not velocity data
        del VX, VY, VZ, S
    cells      =   len(H)  # cells in the selected region
    if (kdensity!=1.0):
        H  *= kdensity
    if (ktemperature!=1.0):
        T  *= ktemperature
    M      =  np.asarray(H*2.8*AMU*((8.0**(-L)*(GL*PARSEC)**3)/MSUN), np.float32)    # MSUN per cell, INPUT IS n(H2) !!!
    MASS   =  sum(M)                    # total mass [Msun] of the selected region
    print("TOTAL MASS OF SELECTED REGION: %.3f MSUN" % MASS) 
    VOLUME =  sum(8.0**(-L))*GL**3.0    # VOLUME [GL^3]
    ENERGY =  0.0
    INFLOW =  0.0
    
    if (tag=='G'):         
        platform, device, context, queue, mf = InitCL(GPU, PF) # OpenCL environment
        LOCAL    =  [1, 32][GPU>0]     # local work group size
        source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_analysis.c').read()
        OPT      =  " -D LEVELS=%d -D CELLS=%d -D NX=%d -D NY=%d -D NZ=%d -D RHO0=%.4ef" % (LEVELS, CELLS, NX, NY, NZ, 0.0)
        program  =  cl.Program(context, source).build(OPT)
        # use kernel to do pairwise summation
        GLOBAL   =  2048
        GLOBAL   =  16384
        Energy_G =  program.Energy_G
        #                               N         x,    y,    z,    M     E  
        Energy_G.set_scalar_arg_dtypes([np.int32, None, None, None, None, None])
        X_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X) # [GL]
        Y_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Y)
        Z_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Z) 
        M_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M) # [MSUN]
        E_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL)   # partial sums, one per work item
        cl.enqueue_copy(queue, X_buf, X)
        cl.enqueue_copy(queue, Y_buf, Y)
        cl.enqueue_copy(queue, Z_buf, Z)
        cl.enqueue_copy(queue, M_buf, M)
        Energy_G(queue, [GLOBAL,], [LOCAL,], cells, X_buf, Y_buf, Z_buf, M_buf, E_buf)
        E        =  np.zeros(GLOBAL, np.float32)
        cl.enqueue_copy(queue, E, E_buf)
        #  kernel  =  sum(M*M/r),   [M]=Msun, [r]=GL
        #  E_grav  =  kernel_result *    GRAV*MSUN**2/(GL*PARSEC)  --  gravitation between cells
        energy1   =  sum(E)                   * GRAV * (MSUN*MSUN/(GL*PARSEC))  # to physical  *=  GRAV*MSUN^2/GL
        #  also add E_grav within the cells = homogeneous cubes have 0.9411 for unit mass and size, ~ M^2/r
        energy2   =  0.9411 * sum(M*M*2.0**L) * GRAV * (MSUN*MSUN/(GL*PARSEC))  # higher L, smaller size, large E_grav
        print("  between cells %.3e, inside cells %.3e   =  %.3e" % (energy1, energy2, energy1+energy2))
        ENERGY = energy1+energy2
    
    elif (tag=='K'):
        # kinetic energy =  macroscopic motion + turbulence within cells + thermal
        #   N * (0.5*m*v^2 + 1.5*k*T)   =   E1 + E2 + E3  =  0.5*M*v^2   +  0.5*M*s^2  +  1.5*k*T*N
        if (len(v0)>1): # kinetic energy wrt velocity in v0
            E1  =  0.5*sum(M*((VX-v0[0])**2.0+(VY-v0[1])**2.0+(VZ-v0[2])**2.0)) * MSUN * (1.0e5)**2
        else:
            E1  =  0.5*sum(M*(VX*VX+VY*VY+VZ*VZ)) * MSUN * (1.0e5)**2   # macroscopic  0.5*M*v^2
        E2  =  1.5*sum(M*S*S)                     * MSUN * (1.0e5)**2   # turbulence   0.5*M*s_3d^2 = 1.5*M*s_1d^2
        E3  =  1.5*BOLTZMANN*sum(T*M)/(2.33*AMU)  * MSUN                # thermal      1.5*N*k*T
        print("  macroscopic %10.3e,  turbulence %10.3e, thermal %10.3e" % (E1, E2, E3))
        print("  mass = %.3e" % sum(M))
        ENERGY = E1+E2+E3        
        print("   VELOCITY DISPERSION: %6.3f" %  ((std(VX)+std(VY)+std(VZ))/3.0))
        print("   SIGMA:               %6.3f" %   mean(S))
        
    elif (tag=='M'):
        # magnetic energy =  B^2 * V / (8*pi)
        nx, ny, nz, lcells, off, b  =  OT_ReadHierarchyV(BX_file)
        B2       =  b[m]**2.0
        nx, ny, nz, lcells, off, b  =  OT_ReadHierarchyV(BY_file)
        B2      +=  b[m]**2.0
        nx, ny, nz, lcells, off, b  =  OT_ReadHierarchyV(BZ_file)
        B2      +=  b[m]**2.0
        # B2 = Bx^2 + By^2 + Bz^2, still without kB
        ENERGY   =  kB*kB  *  sum(B2*8.0**(-L)) * (GL*PARSEC)**3.0  / (8.0*pi)   # B^2 V / (8*pi)
        print("   tag=M ....  kB %.3e, sum(B2*8**-L) %.3e, ENERGY %.3e" % (kB, sum(B2*8.0**(-L)), ENERGY))
        print("   VOLUME = %.3e = %.3e" % (sum(8.0**(-L)), sum(8.0**(-L))*(GL*PARSEC)**3))
                
    elif (tag=='P'):
        # external pressure 3*Pext*V,  Pext = rho*s^2 ....   Pext = n*k*T + n*mu*sigma_1d^2
        #   for each cell, check if neighbour is outside the region
        #   get T and sigma_1d from that cell... calculate surface average <Pext> 
        # test all six sides, based on (X, Y, Z) cell centre coordinates
        # This does not include external turbulent pressure -- but sigma(V) is in the model calculated 
        #   from the delta(V), macroscopic velocity differences
        #   these are at the scale of the current cells =>  normalise these contributions to the GL scale!
        # Calculate at the same time the mass flow across the outer boundary!
        if (len(ind)<1):  
            MASK = np.ones(CELLS, int8)    # all cells selected
        else:            
            MASK      = np.zeros(CELLS, int8)
            MASK[ind] = 1               # MASK==1 for cells (leaf cells) in the selected subvolume
        border = np.zeros(len(X), int8)
        # MASK = for all cells, border = for selected subregion
        W, SUM, INFLOW = 0.0, 0.0, 0.0
        for idir in range(6):
            x, y, z   =   X.copy()+0.001, Y.copy()+0.001, Z.copy()+0.001 #  XYZ for subregion only
            if   (idir==0):   x +=  0.6/2.0**L
            elif (idir==1):   x -=  0.6/2.0**L
            elif (idir==2):   y +=  0.6/2.0**L
            elif (idir==3):   y -=  0.6/2.0**L
            elif (idir==4):   z +=  0.6/2.0**L
            elif (idir==5):   z -=  0.6/2.0**L
            # select cells for which neighbour xyz outside the selected volume but inside the model
            j     =  OT_GetIndicesV(x, y, z, NX, NY, NZ, LCELLS, OFF, H0, GPU=GPU, global_index=True)
            mm    =  nonzero((j>=0)&(MASK[j]==0))  # cells for which neighbour inside cloud, outside subregion
            # rescale turbulence from level L to level 0, corresponding to size scale GL
            # ASSUME SCALING scale^1.5, from scale 0.5^L  to scale 1.0   ==   upscaling  2^(1.5*level)
            sig   =  S[mm] * 2.0**(1.5*L[mm])
            # computing average Pext  --- T, sigma from the cell just *inside* the boundary...
            W    +=  sum(0.25**L[mm])              # total area in units GL^2, cell *inside* the boundary
            SUM  +=  sum((H[mm]*BOLTZMANN*T[mm] + H[mm]*2.33*AMU*sig**2) * (0.5**L[mm]))  # area-weighted, S == sigma_1d
            # SUM uses physical parameters from the neighbour just outside the boundary
            # Calculate also the mass flow across the boudary, positive is inflow
            #   dM*Dt  =  Mcell/V * A * v  * Dt = Mcell/length * v * Dt,   M[] is already [MSUN]
            DT = 365.25*24.0*3600.0  # one year
            if (idir  ==0):  # neighbour above in x, inflow is VX<0
                INFLOW  +=   sum(M[mm] * (2.0**L[mm]) * (-VX[mm]*1.0e5)) *  DT / (GL*PARSEC)
            elif (idir==1):  # inflow = VX>0
                INFLOW  +=   sum(M[mm] * (2.0**L[mm]) * (+VX[mm]*1.0e5)) *  DT / (GL*PARSEC)
            elif (idir==2):  # inflow = VY<0
                INFLOW  +=   sum(M[mm] * (2.0**L[mm]) * (-VY[mm]*1.0e5)) *  DT / (GL*PARSEC)
            elif (idir==3):  # inflow = VY>0
                INFLOW  +=   sum(M[mm] * (2.0**L[mm]) * (+VY[mm]*1.0e5)) *  DT / (GL*PARSEC)
            elif (idir==4):  # inflow   VZ<0
                INFLOW  +=   sum(M[mm] * (2.0**L[mm]) * (-VZ[mm]*1.0e5)) *  DT / (GL*PARSEC)
            elif (idir==5):  # inflow = VZ>0
                INFLOW  +=   sum(M[mm] * (2.0**L[mm]) * (+VZ[mm]*1.0e5)) *  DT / (GL*PARSEC)
                
        Pext     =  SUM/W            # <Pext> on the surface   < n*k*T + n*mu*sigma_1d^2 > 
        V        =  sum(8.0**(-L))   # volume in units GL^3 ... L only contains the subregion cells
        ENERGY   =  3.0*Pext*V * (GL*PARSEC)**3.0

        
    # return energy [erg], mass [MSun], volume [pc^3]
    return ENERGY, MASS, VOLUME, INFLOW
    
        

    
    
def  MaskDensity(filename, xc, yc, zc, rho0, GPU=0, PLF=[0,1,2,3,4], LOCAL0=-1):
    """
    Find indices of cells connected to (xc, yc, zc) and above a density threshold.
    Usage:
        ind = MaskDensity(filename, xc, yc, zc, rho0)
    Input:
        filename    =  name of SOC or LOC cloud
        xc, yc, zc  =  start position in grid units
        rho0        =  density threshold
        GPU         =  if >0, use GPU instead of CPU (default GPU=0)
        PLF         =  possible OpenCL platforms (default PLF=[0, 1, 2, 3, 4])
    Return:
        L           =  integer vector with value 1 for cells in the selected region
                       connected to (xc, yc, zc) and above the density threshold,
                       -1 for cells outside the region
                       2021-08-11 update L also for all the levels above the leaves,
                       setting L=1 if all subcells are inside the region, L=-1 of they are
                       all outside the region, and setting L=0 of only part of the subtree is inside
    """    
    NX, NY, NZ, LCELLS, OFF, H = OT_ReadHierarchyV(filename)
    x, y, z  =  OT_GetCoordinatesAllV(NX, NY, NZ, LCELLS, OFF, H, GPU=0, platforms=[0,1,2,3,4])
    ind0     =  OT_GetIndicesV(asarray([xc,], np.float32), np.asarray([yc,], np.float32), np.asarray([zc,], np.float32), 
                              NX, NY, NZ, LCELLS, OFF, H, GPU=GPU, global_index=True, platforms=PLF)
    rho      =  H[ind0[0]]
    if (rho<rho0):
        print("MaskDensity -- starting cell below density threshold %.3e < %.3e -- nothing to do" % (rho, rho0))
        return None
    LEVELS   =  len(LCELLS)
    CELLS    =  sum(LCELLS)
    if (len(H)!=CELLS): sys.exit()
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=PLF, verbose=False)
    LOCAL    =  [ 1, 32 ][GPU>0]
    if (LOCAL0>0): LOCAL = LOCAL0
    #  ???? POCL FAILS IF LOCAL>1  AND THERE IS NO PRINTF IN THE KERNEL ????
    GLOBAL   =  (CELLS//LOCAL+1)*LOCAL
    OPT      =  " -D LEVELS=%d -D CELLS=%d -D NX=%d -D NY=%d -D NZ=%d -D RHO0=%.4ef" % (LEVELS, CELLS, NX, NY, NZ, rho0)
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_analysis.c').read()
    program  =  cl.Program(context, source).build(OPT)
    #
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
    H_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
    X_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
    Y_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
    Z_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
    IND_buf    =  cl.Buffer(context, mf.READ_WRITE, 2*np.int64(CELLS))
    IND        =  np.zeros(CELLS, int16)
    for i in ind0: IND[i] = 1    # start with one cell flagged
    cl.enqueue_copy(queue, LCELLS_buf, np.asarray(LCELLS, np.int32))
    cl.enqueue_copy(queue, OFF_buf,    np.asarray(OFF,    np.int32))
    cl.enqueue_copy(queue, H_buf,      np.asarray(H,      np.float32))
    cl.enqueue_copy(queue, X_buf,      np.asarray(x,      np.float32))
    cl.enqueue_copy(queue, Y_buf,      np.asarray(y,      np.float32))
    cl.enqueue_copy(queue, Z_buf,      np.asarray(z,      np.float32))
    cl.enqueue_copy(queue, IND_buf,    np.asarray(IND,    int16))
    # one work item per cell, cell coordinates in (x, y, z); one calls ConnectedCells until no more cells get added
    count, count0  =  0, 0
    CC        =   program.ConnectedCells
    CC.set_scalar_arg_dtypes([         None,       None,    None,  None,  None,  None,  None,   ])
    for i in range(NX*2**LEVELS):
        CC(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf, X_buf, Y_buf, Z_buf, IND_buf)
        cl.enqueue_copy(queue, IND, IND_buf)
        count   =  np.sum(IND)      # total number of flagged cells
        if (count==count0): break   # no more cells added
        if (i%20==0):
            print(" ConnectedCells  -- call %6d / %6d -- %6.4f -- added cells: %6d" % (i, NX*2**LEVELS, i/float(NX*2**LEVELS), count-count0))
        count0  =  count
    print("Done!")
    return IND      #  =0, except =1 for cells in the selected region




def  MaskDensity4(filename, xc, yc, zc, rho0, GPU=0, PLF=[0,1,2,3,4], LOCAL0=-1, CLOUD=None):
    """
    Find indices of cells connected to (xc, yc, zc) and above a density threshold.
    Usage:
        ind = MaskDensity(filename, xc, yc, zc, rho0)
    Input:
        filename    =  name of SOC or LOC cloud
        xc, yc, zc  =  start position in grid units
        rho0        =  density threshold
        GPU         =  if >0, use GPU instead of CPU (default GPU=0)
        PLF         =  possible OpenCL platforms (default PLF=[0, 1, 2, 3, 4])
        CLOUD       =  optional dictionary of already read cloud dat:
                       NX, NY, NZ,  LCELLS, OFF, H, x, y, z (all cell coordinates)
    Return:
        L           =  integer vector with value 1 for cells in the selected region
                       connected to (xc, yc, zc) and above the density threshold
    Note:
        As MaskDensity() but using precomputed neighbour lists = one neighbour in each of the six
           sides. This works in MaskDensity() even if each cell may have a large number of neighbours.
        The returned L is now set for all nodes, L=1 for cells/subtrees inside the region, L=-1 for
        outside, and L=0 for subtrees that are only partially in the region.
    """    
    print("MaskDensity4() ---")
    t000       = time.time()
    if (CLOUD==None):
        NX, NY, NZ, LCELLS, OFF, H = OT_ReadHierarchyV(filename)
        x, y, z  =  OT_GetCoordinatesAllV(NX, NY, NZ, LCELLS, OFF, H, GPU=0, platforms=[0,1,2,3,4])
    else:
        NX, NY, NZ   =  CLOUD['NX'], CLOUD['NY'], CLOUD['NZ']
        LCELLS, OFF  =  CLOUD['LCELLS'], CLOUD['OFF']
        x, y, z      =  CLOUD['x'], CLOUD['y'], CLOUD['z']
        H            =  CLOUD['H']
    ###
    ind0     =  OT_GetIndicesV(asarray([xc,], np.float32), np.asarray([yc,], np.float32), np.asarray([zc,], np.float32), 
                               NX, NY, NZ, LCELLS, OFF, H, GPU=GPU, global_index=True, platforms=PLF)
    # print("OT_ReadHierarchyV + OT_GetCoordinatesAllV + OT_GetIndicesV: %.3f seconds" % (time.time()-t0))
    rho      =  H[ind0[0]]
    if (rho<rho0):
        print("MaskDensity -- starting cell below density threshold %.3e < %.3e -- nothing to do" % (rho, rho0))
        return None
    LEVELS   =  len(LCELLS)
    CELLS    =  sum(LCELLS)
    if (len(H)!=CELLS): sys.exit()
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=PLF, verbose=False)
    LOCAL    =  [ 1, 32 ][GPU>0]
    if (LOCAL0>0): LOCAL = LOCAL0
    #  ???? POCL FAILS IF LOCAL>1  AND THERE IS NO PRINTF IN THE KERNEL ????
    GLOBAL   =  (CELLS//LOCAL+1)*LOCAL
    OPT      =  " -D LEVELS=%d -D CELLS=%d -D NX=%d -D NY=%d -D NZ=%d -D RHO0=%.4ef" % (LEVELS, CELLS, NX, NY, NZ, rho0)
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_analysis.c').read()
    program  =  cl.Program(context, source).build(OPT)
    #
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
    H_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
    X_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
    Y_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
    Z_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
    L_buf      =  cl.Buffer(context, mf.READ_WRITE, 2*np.int64(CELLS))
    L          =  -np.ones(CELLS, int16)    # -1 = outside
    for i in ind0: L[i] = 1    # start with one cell flagged, 1 = inside
    cl.enqueue_copy(queue, LCELLS_buf, np.asarray(LCELLS, np.int32))
    cl.enqueue_copy(queue, OFF_buf,    np.asarray(OFF,    np.int32))
    cl.enqueue_copy(queue, H_buf,      np.asarray(H,      np.float32))
    cl.enqueue_copy(queue, X_buf,      np.asarray(x,      np.float32))
    cl.enqueue_copy(queue, Y_buf,      np.asarray(y,      np.float32))
    cl.enqueue_copy(queue, Z_buf,      np.asarray(z,      np.float32))
    cl.enqueue_copy(queue, L_buf,      np.asarray(L,      int16))
    # precalculate neighbours
    t0 = time.time()
    NB_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*6*np.int64(CELLS))  # large array !!
    SN         =  program.SingleNeighbours
    SN.set_scalar_arg_dtypes([     None,       None,    None,   None,   None,   None,  None ])
    SN(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf,  X_buf,  Y_buf,  Z_buf, NB_buf)
    queue.finish()
    print("   Neighbours: %.3f seconds" % (time.time()-t0))
    # print("SingleNeighbours: %.3f seconds" % (time.time()-t0))
    # one work item per cell, cell coordinates in (x, y, z); one calls ConnectedCells until no more cells get added
    count, count0  =  0, 0
    CC         =  program.ConnectedCells4
    CC.set_scalar_arg_dtypes([ None, None, None, None])  # H, L, NB, COUNT
    t0 = time.time()    
    GLOBAL     =  4096
    COUNT_buf  =  cl.Buffer(context, mf.WRITE_ONLY,  2*GLOBAL)
    COUNT      =  np.zeros(GLOBAL, int16)
    t0         =  time.time()
    for i in range(NX*2**LEVELS):
        CC(queue, [GLOBAL,], [LOCAL,], H_buf, L_buf, NB_buf, COUNT_buf)
        cl.enqueue_copy(queue, COUNT, COUNT_buf)
        queue.finish()
        count  =  np.sum(COUNT)    # total number of flagged cells
        if (count<1): break   # no more updates
    # So far we have set L only for leaves.
    # Loop one level at a time (bottom up) and update L in the parent cells, L=1 if all subcells are
    # in the region, L=-1, if they are all outside, and L=0 if some are inside and some outside.
    CU         = program.ConnectedCellsUp
    CU.set_scalar_arg_dtypes([         np.int32, None,       None,    None,   None])
    for level in range(LEVELS-2, -1, -1): # loop over all parent levels
        CU(queue, [GLOBAL,], [LOCAL,], level,    LCELLS_buf, OFF_buf, H_buf,  L_buf)
    cl.enqueue_copy(queue, L, L_buf)
    print("   ConnectedCells: %.3f seconds" % (time.time()-t0))    
    # L=-1 outside, L=1 inside region, L=0 for cells where part of subcells inside, part outside
    # now defined for every cell, not only the leaves
    if (0):  # print out L for all cells
        PL = program.PrintConnectedCells
        PL.set_scalar_arg_dtypes([np.int32, None, None, None, None])
        for level in range(LEVELS):
            print(level)
            PL(queue, [32,], [1,], level, LCELLS_buf, OFF_buf, H_buf, L_buf)
    print("MaskDensity4() -- %.3f seconds" % (time.time()-t000))
    return L      





if (0):
    
    def  MaskDensity2(filename, xc, yc, zc, rho0, GPU=0, PLF=[0,1,2,3,4]):
        """
        Find indices of cells connected to (xc, yc, zc) and above a density threshold.
        Usage:
            ind = MaskDensity(filename, xc, yc, zc, rho0)
        Input:
            filename    =  name of SOC or LOC cloud
            xc, yc, zc  =  start position in grid units
            rho0        =  density threshold
            GPU         =  if >0, use GPU instead of CPU (default GPU=0)
            PLF         =  possible OpenCL platforms (default PLF=[0, 1, 2, 3, 4])
        Return:
            ind         =  integer vector with value 1 for cells in the selected region
                           connected to (xc, yc, zc) and above the density threshold
        """    
        NX, NY, NZ, LCELLS, OFF, H = OT_ReadHierarchyV(filename)
        x, y, z  =  OT_GetCoordinatesAllV(NX, NY, NZ, LCELLS, OFF, H, GPU=0, platforms=[0,1,2,3,4])
        ind0     =  OT_GetIndicesV(asarray([xc,], np.float32), np.asarray([yc,], np.float32), np.asarray([zc,], np.float32), 
                                  NX, NY, NZ, LCELLS, OFF, H, GPU=GPU, global_index=True, platforms=PLF)
        rho      =  H[ind0[0]]
        if (rho<rho0):
            print("MaskDensity -- starting cell below density threshold %.3e < %.3e -- nothing to do" % (rho, rho0))
            return None
        LEVELS   =  len(LCELLS)
        CELLS    =  sum(LCELLS)
        if (len(H)!=CELLS): sys.exit()
        platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=PLF, verbose=False)
        LOCAL    =  [ 1, 32 ][GPU>0]
        #  ???? POCL FAILS IF LOCAL>1  AND THERE IS NO PRINTF IN THE KERNEL ????
        GLOBAL   =  (CELLS//32+1)*32
        OPT      =  " -D LEVELS=%d -D CELLS=%d -D NX=%d -D NY=%d -D NZ=%d -D RHO0=%.4ef" % (LEVELS, CELLS, NX, NY, NZ, rho0)
        source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_analysis.c').read()
        program  =  cl.Program(context, source).build(OPT)
        #
        LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
        OFF_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
        H_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
        X_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
        Y_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
        Z_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
        IND_buf    =  cl.Buffer(context, mf.READ_WRITE, 2*np.int64(CELLS)) #  1 for cells in the masked area)
        ID_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS)) #  cells for which neighbours need to be checked)
        IND        =  np.zeros(CELLS, int16)    # it is just a flag 0/1
        IND[ind0[0]] = 1    # start with one cell flagged
        cl.enqueue_copy(queue, LCELLS_buf, np.asarray(LCELLS, np.int32))
        cl.enqueue_copy(queue, OFF_buf,    np.asarray(OFF,    np.int32))
        cl.enqueue_copy(queue, H_buf,      np.asarray(H,      np.float32))
        cl.enqueue_copy(queue, X_buf,      np.asarray(x,      np.float32))
        cl.enqueue_copy(queue, Y_buf,      np.asarray(y,      np.float32))
        cl.enqueue_copy(queue, Z_buf,      np.asarray(z,      np.float32))
        cl.enqueue_copy(queue, IND_buf,    np.asarray(IND,    int16))    
        count, count0  =  0, 0
        CC        =   program.ConnectedCells2
        CC.set_scalar_arg_dtypes([         np.int32, None,       None,    None,  None,  None,  None,  None,  None ])
        III       =   np.zeros(CELLS, int16)
        ID        =   np.asarray(arange(CELLS), np.int32)   # vector of all cell indices
        for i in range(NX*2**LEVELS):
            m        =   nonzero(IND>0)     # which cells have been flagged
            num      =   len(m[0])          # this many flagged cells
            cl.enqueue_copy(queue, ID_buf, np.asarray(ID[m], np.int32))   #  the cell indices of the num cells to be processed
            GLOBAL   =   (num//32+1)*32     # need num work items
            # kernel run foor cells ID[id], id<num, checking six neighbours and update their IND
            CC(queue, [GLOBAL,], [LOCAL,], num , LCELLS_buf, OFF_buf, H_buf, X_buf, Y_buf, Z_buf, IND_buf, ID_buf)
            queue.finish()
            cl.enqueue_copy(queue, IND, IND_buf)   # perhaps some new cells were flagged
            queue.finish()
            count   =  sum(IND)         # total number of flagged cells
            if (count==count0): break   # no more cells added
            if (i%20==0):
                print(" ConnectedCells  -- call %6d / %6d -- %6.4f -- added cells: %6d" % (i, NX*2**LEVELS, i/float(NX*2**LEVELS), count-count0))
            count0  =  count
            sys.stdout.flush()
        print("Done!")
        return IND      #  =0, except =1 for cells in the selected region
    
                         
    
    
    def  MaskDensity3(filename, xc, yc, zc, rho0, GPU=0, PLF=[0,1,2,3,4]):
        """
        Find indices of cells connected to (xc, yc, zc) and above a density threshold.
        Usage:
            ind = MaskDensity(filename, xc, yc, zc, rho0)
        Input:
            filename    =  name of SOC or LOC cloud
            xc, yc, zc  =  start position in grid units
            rho0        =  density threshold
            GPU         =  if >0, use GPU instead of CPU (default GPU=0)
            PLF         =  possible OpenCL platforms (default PLF=[0, 1, 2, 3, 4])
        Return:
            ind         =  integer vector with value 1 for cells in the selected region
                           connected to (xc, yc, zc) and above the density threshold
        """    
        NX, NY, NZ, LCELLS, OFF, H = OT_ReadHierarchyV(filename)
        x, y, z  =  OT_GetCoordinatesAllV(NX, NY, NZ, LCELLS, OFF, H, GPU=0, platforms=[0,1,2,3,4])
        ind0     =  OT_GetIndicesV(asarray([xc,], np.float32), np.asarray([yc,], np.float32), np.asarray([zc,], np.float32), 
                                  NX, NY, NZ, LCELLS, OFF, H, GPU=GPU, global_index=True, platforms=PLF)
        rho      =  H[ind0[0]]
        if (rho<rho0):
            print("MaskDensity -- starting cell below density threshold %.3e < %.3e -- nothing to do" % (rho, rho0))
            return None
        LEVELS   =  len(LCELLS)
        CELLS    =  sum(LCELLS)
        if (len(H)!=CELLS): sys.exit()
        platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=PLF, verbose=False)
        LOCAL    =  [ 1, 32 ][GPU>0]
        #  ???? POCL FAILS IF LOCAL>1  AND THERE IS NO PRINTF IN THE KERNEL ????
        GLOBAL   =  (CELLS//32+1)*32
        OPT      =  " -D LEVELS=%d -D CELLS=%d -D NX=%d -D NY=%d -D NZ=%d -D RHO0=%.4ef" % (LEVELS, CELLS, NX, NY, NZ, rho0)
        source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_analysis.c').read()
        program  =  cl.Program(context, source).build(OPT)
        #
        LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
        OFF_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
        H_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
        X_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
        Y_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
        Z_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
        IND_buf    =  cl.Buffer(context, mf.READ_WRITE, 2*np.int64(CELLS))  #  1 for cells in the masked area)
        IND        =  np.zeros(CELLS, int16)    # it is just a flag 0/1
        IND[ind0[0]] = 1    # start with one cell flagged
        cl.enqueue_copy(queue, LCELLS_buf, np.asarray(LCELLS, np.int32))
        cl.enqueue_copy(queue, OFF_buf,    np.asarray(OFF,    np.int32))
        cl.enqueue_copy(queue, H_buf,      np.asarray(H,      np.float32))
        cl.enqueue_copy(queue, X_buf,      np.asarray(x,      np.float32))
        cl.enqueue_copy(queue, Y_buf,      np.asarray(y,      np.float32))
        cl.enqueue_copy(queue, Z_buf,      np.asarray(z,      np.float32))
        cl.enqueue_copy(queue, IND_buf,    np.asarray(IND,    int16))    
        count, count0  =  0, 0
        CC        =   program.ConnectedCells3
        CC.set_scalar_arg_dtypes([        None,       None,    None,  None,  None,  None,  None ])
        III       =   np.zeros(CELLS, int16)
        ID        =   np.asarray(arange(CELLS), np.int32)   # vector of all cell indices
        for i in range(NX*2**LEVELS):
            CC(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf, X_buf, Y_buf, Z_buf, IND_buf)
            queue.finish()
            cl.enqueue_copy(queue, IND, IND_buf)   # perhaps some new cells were flagged
            queue.finish()
            count   =  sum(IND)                    # total number of flagged cells ... actually values 0/1/2
            if (count==count0): break              # no more cells added
            if (i%20==0):
                print(" ConnectedCells  -- call %6d / %6d -- %6.4f -- added cells: %6d" % (i, NX*2**LEVELS, i/float(NX*2**LEVELS), count-count0))
            count0  =  count
            sys.stdout.flush()
        print("Done!")
        return IND      #  =0, except =1 for cells in the selected region


    
    
def gravitational_energy(NX, NY, NZ, LCELLS, OFF, H, GL, kdensity=1.0, GPU=0, PLF=[0,1,2,3,4]):
    """
    Given the octree cloud data, calculate for each node (1) the mass contained in the subtree,
    and (2) the centre-of-mass coordinates. Use this information then to calculate the
    gravitational energy (between cells + inside cells)
    Input:
        NX, NY, NZ  =  cloud root grid dimensions
        LCELLS      =  1d array of the number of cells per hierarchy level
        OFF         =  1d array of the cell-index offsets to the start of each hierarchy level subvector
        H           =  the hierarchy with density values
        GL          =  root-grid cell size [pc]
        kdensity    =  scaling to get n(H2) from the H[] values
        GPU         =  if >0, use GPU instead of CPU (default GPU=0)
        PLF         =  list of potential OpenCL platforms to try (default=[0,1,2,3,4])
    Return:
        E           =  gravitational energy in cgs units
    """
    print("gravitational_energy ---")
    t0 = time.time()
    CELLS  = sum(LCELLS)
    LEVELS = len(LCELLS)
    # We loop over all cells, one hierarchy level at a time and starting from the bottom
    #    mass in the node is them sum of the masses of the children,
    # Start by reading the coordinates of all the cells (this is actually needed for leaves only)
    XC, YC, ZC   =  OT_GetCoordinatesAllV(NX, NY, NZ, LCELLS, OFF, H, GPU=GPU, platforms=PLF)
    # use an OpenCL kernel to do the averaging, traversing the hierarchy from bottom up
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=PLF, verbose=False)
    LOCAL    =  [ 1, 32 ][GPU>0]
    GLOBAL   =  16384
    OPT      =  " -D LEVELS=%d -D CELLS=%d -D NX=%d -D NY=%d -D NZ=%d -D KDENS=%.4ef -D GL=%.4ef" % \
                    (LEVELS,      CELLS,      NX,      NY,      NZ,      kdensity,      GL)
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_gravity.c').read()
    program  =  cl.Program(context, source).build(OPT)
    #
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
    H_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
    X_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    Y_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    Z_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    M_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    cl.enqueue_copy(queue, LCELLS_buf, np.asarray(LCELLS, np.int32))
    cl.enqueue_copy(queue, OFF_buf,    np.asarray(OFF,    np.int32))
    cl.enqueue_copy(queue, H_buf,      H)   # H is always np.float32 !
    cl.enqueue_copy(queue, X_buf,      XC)
    cl.enqueue_copy(queue, Y_buf,      YC)
    cl.enqueue_copy(queue, Z_buf,      ZC)
    MC         =  program.mass_centres
    #                                  level         LCELLS      OFF      H      X      Y      Z      M   
    MC.set_scalar_arg_dtypes([         np.int32,     None,       None,    None,  None,  None,  None,  None ])
    t1 = time.time()
    for L in range(LEVELS-1, -1, -1):
        MC(queue, [GLOBAL,], [LOCAL,], np.int32(L),  LCELLS_buf, OFF_buf, H_buf, X_buf, Y_buf, Z_buf, M_buf)
        queue.finish()
    print("   MC --- %.2f seconds" % (time.time()-t1))
    if (1): # check the mass as the sum over root grid entries in M
        M = np.zeros(CELLS, np.float32)
        cl.enqueue_copy(queue, M, M_buf)
        print("   Total mass from root cell entries: %.3e" % (sum(M[0:LCELLS[0]])))
    # Now we have (M, XC, YC, ZC) for each cell in the model, calculate gravitational energy with 
    # direct summation of M*m/r. 
    # Loop over all cells using all work items. Each work item loops over all root-grid cells and
    # all subcells in those trees, not going below level L for which 0.5**L < 0.1*r, where r is the
    # distance between cells.
    E_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL)
    GE         =  program.gravitational_energy
    t1 = time.time()
    GE.set_scalar_arg_dtypes([     None,       None,    None,  None,  None,  None,  None,  None  ])
    GE(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf, X_buf, Y_buf, Z_buf, M_buf, E_buf)
    E          =  np.zeros(GLOBAL, np.float32)
    cl.enqueue_copy(queue, E, E_buf)
    print("   GE --- %.2f seconds" % (time.time()-t1))    
    E1         =  sum(E)                     * GRAV * (MSUN**2.0/(GL*PARSEC))
    # add gravity within the cells
    E2         =  0.0
    for l in range(LEVELS):
        a, b   =  OFF[l], OFF[l]+LCELLS[l]
        rho    =  H[a:b]
        #  mass = density x volume
        mass   =  rho[nonzero(rho>0.0)] * kdensity*2.8*AMU * (GL*PARSEC)**3 / 8.0**l
        #  linear cell size
        d      =  GL*PARSEC / 2.0**l
        E2    +=  sum(mass*mass) /  d
    E2 *= GRAV*0.9411
    print("gravitational_energy        %.4e + %.4e = %.4e  --- %.2f seconds" % (E1, E2, E1+E2, time.time()-t0))
    return E1+E2
    


def gravitational_energy_masked(L, NX, NY, NZ, LCELLS, OFF, H, GL, kdensity=1.0, GPU=0, PLF=[0,1,2,3,4]):
    """
    Given the octree cloud data, calculate for each node (1) the mass contained in the subtree,
    and (2) the centre-of-mass coordinates. Use this information then to calculate the
    gravitational energy (between cells + inside cells).
    Input:
        L           =  index vector (short int) specifying the region over which energy calculated
        NX, NY, NZ  =  cloud root grid dimensions
        LCELLS      =  1d array of the number of cells per hierarchy level
        OFF         =  1d array of the cell-index offsets to the start of each hierarchy level subvector
        H           =  the hierarchy with density values
        GL          =  root-grid cell size [pc]
        kdensity    =  scaling to get n(H2) from the H[] values
        GPU         =  if >0, use GPU instead of CPU (default GPU=0)
        PLF         =  list of potential OpenCL platforms to try (default=[0,1,2,3,4])
    Return:
        E           =  gravitational energy in cgs units
    Note:
        This version takes an additional vector L that specifies the region over which energy is
        calculated. L=1 for cells inside and L=-1 for cells outside the region. L is defined also for
        all parent cells, where it is L=0, if only part of children are inside the region, or
        L=-1 or L=+1, if all children are outside vs. inside.        
    """
    print("gravitational_energy_masked ---")
    t0 = time.time()
    CELLS  = sum(LCELLS)
    LEVELS = len(LCELLS)
    # We loop over all cells, one hierarchy level at a time and starting from the bottom
    #    mass in the node is them sum of the masses of the children,
    # Start by reading the coordinates of all the cells (this is actually needed for leaves only)
    XC, YC, ZC   =  OT_GetCoordinatesAllV(NX, NY, NZ, LCELLS, OFF, H, GPU=GPU, platforms=PLF)
    # use an OpenCL kernel to do the averaging, traversing the hierarchy from bottom up
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=PLF, verbose=False)
    LOCAL    =  [ 1, 32 ][GPU>0]
    GLOBAL   =  16384
    OPT      =  " -D LEVELS=%d -D CELLS=%d -D NX=%d -D NY=%d -D NZ=%d -D KDENS=%.4ef -D GL=%.4ef" % \
                    (LEVELS,      CELLS,      NX,      NY,      NZ,      kdensity,      GL)
    source   =  open(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_OT_gravity.c').read()
    program  =  cl.Program(context, source).build(OPT)
    #
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*LEVELS)
    H_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*np.int64(CELLS))
    X_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    Y_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    Z_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    M_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*np.int64(CELLS))
    cl.enqueue_copy(queue, LCELLS_buf, np.asarray(LCELLS, np.int32))
    cl.enqueue_copy(queue, OFF_buf,    np.asarray(OFF,    np.int32))
    cl.enqueue_copy(queue, H_buf,      H)   # H is always np.float32 !
    cl.enqueue_copy(queue, X_buf,      XC)
    cl.enqueue_copy(queue, Y_buf,      YC)
    cl.enqueue_copy(queue, Z_buf,      ZC)
    MC         =  program.mass_centres  # calculates mass and centre-of-mass coordinates for every cell in the tree
    #                                  level         LCELLS      OFF      H      X      Y      Z      M   
    MC.set_scalar_arg_dtypes([         np.int32,     None,       None,    None,  None,  None,  None,  None ])
    t1 = time.time()
    for level in range(LEVELS-1, -1, -1):
        MC(queue, [GLOBAL,], [LOCAL,], np.int32(level),  LCELLS_buf, OFF_buf, H_buf, X_buf, Y_buf, Z_buf, M_buf)
        queue.finish()
    print("   MC --- %.2f seconds" % (time.time()-t1))
    if (1): # check the mass as the sum over root grid entries in M
        M = np.zeros(CELLS, np.float32)
        cl.enqueue_copy(queue, M, M_buf)
        print("   Total mass from root cell entries: %.3e" % (sum(M[0:LCELLS[0]])))
    # Now we have (M, XC, YC, ZC) for each cell in the model, calculate gravitational energy with 
    # direct summation of M*m/r. 
    # Loop over all cells using all work items. Each work item loops over all root-grid cells and
    # all subcells in those trees, not going below level for which 0.5**level < 0.1*r, where r is the
    # distance between cells.
    # In this version outer loop over all cells with L=1 (leaves),
    # inner loop as before - but of subtree has L=0, must go further down in the hierarchy,
    # until one has L=1 subtrees (possibly all the way down to leaves)
    L_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=L)  # must be short ints
    E_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL)
    GE         =  program.gravitational_energy_masked
    t1 = time.time()
    GE.set_scalar_arg_dtypes([     None,  None,       None,    None,  None,  None,  None,  None,  None  ])
    GE(queue, [GLOBAL,], [LOCAL,], L_buf, LCELLS_buf, OFF_buf, H_buf, X_buf, Y_buf, Z_buf, M_buf, E_buf)
    E          =  np.zeros(GLOBAL, np.float32)
    cl.enqueue_copy(queue, E, E_buf)
    print("   GE-masked ---  %.2f seconds" % (time.time()-t1))
    E1         =  sum(E)                     * GRAV * (MSUN**2.0/(GL*PARSEC))
    # add gravity within the cells
    E2         =  0.0
    for l in range(LEVELS):
        a, b   =  OFF[l], OFF[l]+LCELLS[l]
        rho    =  H[a:b]
        m      =  nonzero((rho>0.0)&(L[a:b]==1))   # only leaves, those with L=1
        #  mass = density x volume
        mass   =  rho[m] * kdensity*2.8*AMU * (GL*PARSEC)**3 / 8.0**l
        #  linear cell size
        d      =  GL*PARSEC / 2.0**l
        E2    +=  sum(mass*mass) /  d
    E2 *= GRAV*0.9411
    print("gravitational_energy_masked %.4e + %.4e = %.4e  --- %.2f seconds" % (E1, E2, E1+E2, time.time()-t0))
    return E1+E2
    



####################################################################################################
####################################################################################################
####################################################################################################


#   # Pymses works only with python2 ... and pyopencl only with python3  !!!!
#   # 2020-12-31 move pymses-dependent routines to OT_pymses and try to work around
#   # the problem of routines requiring BOTH pymses and pyopencl
#   # 2022-07-10: drop pymses and use osyris instead
#   import  numpy as np
#   import  pyopencl as cl
#   from    pymses.filters  import  CellsToPoints
#   import  pymses
#   import  time

        
        
def RamsesDumpOLD(MODEL, OUTPUT, BSCALE=1.0, FROMDIR='', TODIR='./'):
    """
    Dump rho and magnetic field components to plain binary files.
    We must make sure none of the values is zero
    Note:
        2021-05-09:  routine switches Bx and Bz for SOC (from Fortran to C ordering)
    """    
    print("RamsesDump ==> WILL CHANGE BX, BY, BZ --> BZ, BY, BX !!!!")
    if (FROMDIR==''):
        ro           =  pymses.RamsesOutput("/data/mika/PADOAN/RAMSES/%s/" % MODEL,  OUTPUT)
    else:
        ro           =  pymses.RamsesOutput("%s/%s/" % (FROMDIR, MODEL),  OUTPUT)
    ###
    amr          =  ro.amr_source(['rho','Bl', 'Br'])       
    cell_source  =  CellsToPoints(amr)
    if (0):
        cells        =  cell_source.flatten()
        # COO          =  np.asarray(cells.points, np.float32)
        # np.asarray(COO, np.float32).tofile('RAMSES_COO.dump')
        n            =  len(cells.fields['rho'])
        prefix       =  'dump_%s_%06d' % (MODEL, OUTPUT)
        ##
        fp           =  open(prefix+'.rho', 'wb')
        np.asarray([n,], int32).tofile(fp)
        x            =  np.asarray(cells.fields['rho'], np.float32)
        x[nonzero(x<1.0e-10)] = 1.0e-10
        print('RHO ', np.min(x), np.max(x))
        x.tofile(fp)
        fp.close()
        ##
        fp           =  open(prefix+'.Bx', 'wb')
        np.asarray([n,], int32).tofile(fp)
        x            = np.asarray(0.5*ravel(cells.fields['Bl'][:,0]+cells.fields['Br'][:,0]), np.float32)
        x[nonzero(np.abs(x)<1.0e-10)] = 1.0e-10
        x.tofile(fp)
        fp.close()
        ##
        fp           =  open(prefix+'.By', 'wb')
        np.asarray([n,], int32).tofile(fp)
        x            =  np.asarray(0.5*ravel(cells.fields['Bl'][:,1]+cells.fields['Br'][:,1]), np.float32)
        x[nonzero(np.abs(x)<1.0e-10)] = 1.0e-10
        x.tofile(fp)
        fp.close()
        ##
        fp           =  open(prefix+'.Bz', 'wb')
        np.asarray([n,], int32).tofile(fp)
        x            =  np.asarray(0.5*ravel(cells.fields['Bl'][:,2]+cells.fields['Br'][:,2]), np.float32)
        x[nonzero(np.abs(x)<1.0e-10)] = 1.0e-10    
        x.tofile(fp)
        fp.close()
    else:
        # proper ordering, low memory footprint()
        prefix  =  '%s/dump_%s_%06d' % (TODIR, MODEL, OUTPUT)
        fpR     =   open(prefix+'.rho', 'wb')
        fpBx    =   open(prefix+'.Bx',  'wb')
        fpBy    =   open(prefix+'.By',  'wb')
        fpBz    =   open(prefix+'.Bz',  'wb')
        A       =   np.asarray([0,], np.int32)
        A.tofile(fpR)   # total number of data points to be updated later
        A.tofile(fpBx)
        A.tofile(fpBy)
        A.tofile(fpBz)
        n = 0
        for dset in cell_source.iter_dsets():
            n += dset.npoints
            print('%9d %9d' % (dset.npoints, n))
            X = np.asarray(dset.fields['rho'], np.float32)
            X.tofile(fpR)
            ##
            BL = np.asarray(dset.fields['Bl'],  np.float32)
            BR = np.asarray(dset.fields['Br'],  np.float32)
            if (OLD_ORIENTATION==-999):                    
                print("OLD ORIENTATION -- NOT CORRECT")
                sys.exit()
                X  = 0.5*BSCALE*(BL[:,2]+BR[:,2])              #  SWITCH Bx and Bz for SOC !!!
                X.tofile(fpBx)
                X  = 0.5*BSCALE*(BL[:,1]+BR[:,1])
                X.tofile(fpBy)
                X  = 0.5*BSCALE*(BL[:,0]+BR[:,0])
                X.tofile(fpBz)
            else:
                # 2021-06-16 NO SWITCH
                #  Why? We use coordinates (x, y, z) to construct the 3d hierarchy and therefore
                #  it is independent of Fortan/C array ordering differences. RAMSES "x" remains
                #  coordinate "x" also in SOC and Bx should similarly remain Bx...
                print("NEW ORIENTATION")
                X  = 0.5*BSCALE*(BL[:,0]+BR[:,0])              #  Bx and Bz ***NOT*** swicthed 2021-06-16
                X.tofile(fpBx)
                X  = 0.5*BSCALE*(BL[:,1]+BR[:,1])
                X.tofile(fpBy)
                X  = 0.5*BSCALE*(BL[:,2]+BR[:,2])
                X.tofile(fpBz)
        ##
        fpR.seek(0)
        fpBx.seek(0)
        fpBy.seek(0)
        fpBz.seek(0)        
        A    =  np.asarray([n,], np.int32)
        A.tofile(fpR)
        A.tofile(fpBx)
        A.tofile(fpBy)
        A.tofile(fpBz)
        fpR.close()
        fpBx.close()
        fpBy.close()
        fpBz.close()        
        
    
    

def RamsesDumpFieldOLD(MODEL, OUTPUT, TAG='rho', FROMDIR='', TODIR='./'):
    """
    Dump rho and magnetic field components to plain binary files.
    We must make sure none of the values is zero (?)
    Input:
        MODEL    =   e.g. SN50_9_13_sg   (run name)
        OUTPUT   =   e.g. 377 (snapshot)
        TAG      =   e.g. rho (name of the data field)
        FROMDIR  =   directory containing MODEL as subdirectory
        TODIR    =   directory where the output file is written
    Note:
        2021-03-09:
            RAMSES file presumably in Fortran order
            SOC reads the file in C-order=>  original Z becomes X, X becomes Z
            ==> we must also switch velocity fields VX -> VZ, VZ -> VX
    """    
    if (FROMDIR==''):
        ro           =  pymses.RamsesOutput("/data/mika/PADOAN/RAMSES/%s/" % MODEL,  OUTPUT)
    else:
        ro           =  pymses.RamsesOutput("%s/%s/" % (FROMDIR, MODEL),  OUTPUT)
    print(ro.info)
    print(ro.fields_to_read)
    ###
    # amr          =  ro.amr_source(['rho', 'Bl', 'Br'])       
    amr          =  ro.amr_source([TAG,])
    cell_source  =  CellsToPoints(amr)
    # proper ordering, low memory footprint()
    prefix  =  '%s/dump_%s_%06d' % (TODIR, MODEL, OUTPUT)
    if (TAG=='vel'):            
        fpX     =   open(prefix+'.vx', 'wb')   # FORTRAN NAMING 
        fpY     =   open(prefix+'.vy', 'wb')
        fpZ     =   open(prefix+'.vz', 'wb')
        A       =   np.asarray([0,], np.int32)
        A.tofile(fpX)   # total number of data points to be updated later
        A.tofile(fpY)   # total number of data points to be updated later
        A.tofile(fpZ)   # total number of data points to be updated later
        n = 0
        for dset in cell_source.iter_dsets():
            n += dset.npoints
            print('%9d %9d' % (dset.npoints, n))
            X = np.asarray(dset.fields[TAG],  np.float32)
            # *** we do not like zero values !!! ***
            for iii in range(3):
                m              =  nonzero(np.abs(X[:,iii])<1.0e-6)
                X[m[0], iii]  +=  1.0e-4*randn(len(m[0]))
            if (OLD_ORIENTATION==-999):  # B has to be switched => vel must be switched ??
                X[:,2].tofile(fpX)  # fortran Z becomes our X   == ROTATE VECTOR COMPONENTS
                X[:,1].tofile(fpY)
                X[:,0].tofile(fpZ)  # fortran X becomes our Z
                print("VELOCITY CHANGED TO C ORDER:   VX, VY, VZ  ->  VZ, VY, VX !!!!!")
                print("... THIS IS NOT CORRECT")
                sys.exit()
            else:
                # 2021-06-16 NO SWITCH
                #  Probably because cloud is constructed usingdirectly (x,y,z) coordinate values
                #  and not relying in the ordering of the data in the files
                X[:,0].tofile(fpX)
                X[:,1].tofile(fpY)
                X[:,2].tofile(fpZ)
                print("VELOCITY COMPONENTS VX AND VZ ***NOT** SWITCHED !!!!!")
        ##
        fpX.seek(0)
        fpY.seek(0)
        fpZ.seek(0)
        A    =  np.asarray([n,], np.int32)
        A.tofile(fpX)
        A.tofile(fpY)
        A.tofile(fpZ)
        fpX.close()
        fpY.close()
        fpZ.close()
    else:
        fp      =   open(prefix+'.'+TAG, 'wb')
        A       =   np.asarray([0,], np.int32)
        A.tofile(fp)   # total number of data points to be updated later
        n = 0
        for dset in cell_source.iter_dsets():
            n += dset.npoints
            print('%9d %9d' % (dset.npoints, n))
            X = np.asarray(dset.fields[TAG], np.float32)
            if (TAG=='rho'):
                X[nonzero(X<=1.0e-11)] = 1.0e-11
            X.tofile(fp)
        ##
        fp.seek(0)
        A    =  np.asarray([n,], np.int32)
        A.tofile(fp)
        fp.close()

    
        
    
                               

def RamsesToIndexOLD(MODEL, OUTPUT, centre, delta, direction, GPU=False, FROMDIR='', TODIR='./', platforms=[0,1,2,3,4]):
    """
    Write an index file. The format is the same as SOCAMO cloud file but the
    main array is np.int32 instead of float.
    Each cell contains either index to the Ramses vectors == 1,2,3, ... (leaf cells only)
    or values <=0 meaning links.
    """
    
    fp_info = None    
    CMIN    =   np.asarray(centre) - delta
    CMAX    =   np.asarray(centre) + delta
    if (  direction=='x'):  # take the full LOS along the selected direction
        CMIN[0], CMAX[0] = 0.0, 1.0
    elif (direction=='y'):
        CMIN[1], CMAX[1] = 0.0, 1.0
    elif (direction=='z'):
        CMIN[2], CMAX[2] = 0.0, 1.0
    CMIN = clip(CMIN, 0.0, 1.0)
    CMAX = clip(CMAX, 0.0, 1.0)

    if (FROMDIR==''):
        ro  =  pymses.RamsesOutput("/data/mika/PADOAN/RAMSES/%s/" % MODEL,  OUTPUT)
    else:
        ro  =  pymses.RamsesOutput("%s/%s/" % (FROMDIR, MODEL),  OUTPUT)        
    if (0):
        GL           =  ro.info["boxlen"]  /  2.0**ro.info["levelmin"]  # root grid cell size
    else:
        GL           =  1.0                /  2.0**ro.info["levelmin"]  # root grid cell size
    amr          =  ro.amr_source(['rho',])    
    cell_source  =  CellsToPoints(amr)    
    N = 0
    for dset in cell_source.iter_dsets():
        N += dset.npoints
    
    X, Y, Z  =  np.zeros(N, np.float32), np.zeros(N, np.float32), np.zeros(N, np.float32)
    S        =  np.zeros(N, np.float32)
    ## R        =  np.asarray([], np.float32)
    if (0):
        # parallel read ... files for different CPUs read in ***RANDOM*** order
        cells        =  cell_source.flatten()
        COO          =  cells.points          #  [cells, 3]
        X, Y, Z      =  COO[:,0].copy(), COO[:,1].copy(), COO[:,2].copy()
        COO          =  None
        S            =  cells.get_sizes()     #  [cells]
        R            =  cells.fields['rho']
        cells        =  None
    else:
        # make sure everything is read in order          
        # IN FILE "X"   IS OUR X, "Z" IN THE INPUT FILE IS IS OUR X
        # IN FILE "VX"  IS ALONG THEIR "X" = ALONG OUR Z 
        #  => EVERYTHING IS ROTATED BUT OTHERWISE OK ??? 
        # 
        # OR -- ONE COULD SWITCH X AND Z HERE = ROTATE THE DENSITY FIELD
        #       AND LATER ALSO SWITCH THE VX AND VZ CUBES WHEN MAKING LOC CLOUD FILES
        iii = 0
        for dset in cell_source.iter_dsets():            
            if (OLD_ORIENTATION):
                print("OLD ORIENTATION")
                # Ok, is perhaps consistent. Cloud is constructed using the same (x, y, z)
                # axes as in original data files, "x" remains "x" etc. Then also the
                # B and V data will be used without switching axes.
                X[iii:(iii+dset.npoints)] = dset.points[:,0]  # no switch
                Y[iii:(iii+dset.npoints)] = dset.points[:,1]  #    agrees with WMP plot based on get_cube
                Z[iii:(iii+dset.npoints)] = dset.points[:,2]
            else:
                # 2021-06-16  SWITCH HERE NOT FOR B AND V ???
                print("NEW ORIENTATION --- THIS IS NOT CORRECT")
                sys.exit()
                X[iii:(iii+dset.npoints)] = dset.points[:,2]  # SWITCH HERE --- included 2021-06-15
                Y[iii:(iii+dset.npoints)] = dset.points[:,1]  #    agrees with WMP plot based on get_cube
                Z[iii:(iii+dset.npoints)] = dset.points[:,0]
            S[iii:(iii+dset.npoints)] = dset.get_sizes()            
            # 0.00390625    =  1.0/2.0**8  == model refined to level 8 and deeper
            print("S .... ", S[iii:(iii+5)])            
            ## R = concatenate((R, dset.fields['rho']))
            iii += dset.npoints
    print('READ POINTS... ', len(X), len(Y), len(Z), len(S)) ## , len(R)
    
    ### print 'R statistics  %10.3e ... %10.3e    %10.3e +- %10.3e  %d' % (np.min(R), max(R), mean(R), std(R), len(R))
    ### print R[10000], R[20000], R[100000], R[200000], R[1000000]
    ### return
    
    
    # ok, change R into a running, 1-based index to Ramses vectors (leaf cells only)
    # vector is np.int32, values <= are the normal links, 
    # values i>0 =>      cell value  <-  ramses vector value [i-1]
    # ... or read the full ramses dump =  [cells, val0, val1, ...] as float vector, index with [i]
    N            =  len(X)
    R            =  None
    R            =  np.asarray(arange(1, N+1), np.int32)  # 1, 2, ..., number of leaf cells
    
    
    # Find the dimensions of the root grid
    # XMIN, XMAX etc. --- minimum and maximum root grid indices [XMIN, XMAX[
    XMIN, XMAX   =  int(round(CMIN[0]/GL)),  int(round(CMAX[0]/GL))
    YMIN, YMAX   =  int(round(CMIN[1]/GL)),  int(round(CMAX[1]/GL))
    ZMIN, ZMAX   =  int(round(CMIN[2]/GL)),  int(round(CMAX[2]/GL))
    DIM1, DIM2, DIM3 = XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN
    print('X  %4d to %4d, DIM %4d' % (XMIN, XMAX, DIM1))
    print('Y  %4d to %4d, DIM %4d' % (YMIN, YMAX, DIM2))
    print('Z  %4d to %4d, DIM %4d' % (ZMIN, ZMAX, DIM3))
    
    SUBVOLUME = False
    if ((XMIN>0)|(YMIN>0)|(ZMIN>0)): SUBVOLUME = True
    if ((XMAX<(DIM1-1))|(YMAX<(DIM2-1))|(ZMAX<(DIM3-1))): SUBVOLUME = True
    print('SUBVOLUME = ', SUBVOLUME)
    
    # update the coordinate limits == on boundaries of root grid coordinates [XMIN*GL, XMAX*GL]
    print('COORD LIMITS  %8.5f-%8.5f  %8.5f-%8.5f    %8.5f-%8.5f' % (CMIN[0], CMAX[0], CMIN[1], CMAX[1], CMIN[2], CMAX[2]))
    CMIN         =  [   XMIN*GL,   YMIN*GL,   ZMIN*GL  ]
    CMAX         =  [   XMAX*GL,   YMAX*GL,   ZMAX*GL  ]
    print('      ==>     %8.5f-%8.5f  %8.5f-%8.5f    %8.5f-%8.5f' % (CMIN[0], CMAX[0], CMIN[1], CMAX[1], CMIN[2], CMAX[2]))

    print("levelmin ????")
    DIM          =  2**ro.info["levelmin"]   # our root grid, everything refined at least to levelmin
    print("levelmin !!!!")
    print('DIM1 %d, DIM2 %d, DIM3 %d' % (DIM1, DIM2, DIM3))
    print('DIM = %d' % DIM)
    
    # root grid DIM1*DIM2*DIM3 cells
    if ((DIM1>DIM)|(DIM2>DIM)|(DIM3>DIM)):
        print("Everything refined to DIM = %3d" % DIM)
        print('A: ????????????')
        print("OK ???? ---- DIM1, DIM2, DIM3 > DIM !!!!!")
        # sys.exit()


        
    # AD HOC !!!!
    DIM1, DIM2, DIM3 = DIM, DIM, DIM
        
        
        
        
    if (SUBVOLUME):
        sys.exit()
        cloudfile = '%s/%s_%06d_%.4f_%.4f_%.4f_%.4f_%.4f_%.4f.index' % \
        (TODIR, MODEL, OUTPUT, CMIN[0], CMAX[0], CMIN[1], CMAX[1], CMIN[2], CMAX[2])
    else:
        cloudfile = '%s/%s_%06d.index' % (TODIR, MODEL, OUTPUT)
                            
    
    LEVELS = ro.info["levelmax"]-ro.info["levelmin"]+1  # number of levels in the hierarchy
    
    if (0):
        SIZE   = ro.info["boxlen"] / 2.0**(ro.info["levelmin"]+arange(LEVELS))
    else:
        SIZE   = 1.0               / 2.0**(ro.info["levelmin"]+arange(LEVELS))
    
        
    # Save meta file
    # Input data in large arrays { X, Y, Z, R, S }
    fp_info = open(cloudfile.replace('.index', '.meta'), 'w')
    fp_info.write('boxlen      %.8f\n'      % ro.info["boxlen"])  # always 1.0 ???
    fp_info.write('GL          %.8e\n'      % GL)   #  root cell size / box size
    fp_info.write('levelmin    %d\n'        % ro.info["levelmin"])
    fp_info.write('levelmax    %d\n'        % ro.info["levelmax"])
    fp_info.write('levels      %d\n'        % (ro.info["levelmax"]-ro.info["levelmin"]+1))
    fp_info.write('rootgrid    %d %d %d\n'  % (DIM, DIM, DIM))      # full cloud in root cells
    fp_info.write('dimensions  %d %d %d\n'  % (DIM1, DIM2, DIM3))   # extracted cloud in root cells
    fp_info.write('xrange      %.8e %.8e\n' % (CMIN[0], CMAX[0]))
    fp_info.write('yrange      %.8e %.8e\n' % (CMIN[1], CMAX[1]))
    fp_info.write('zrange      %.8e %.8e\n' % (CMIN[2], CMAX[2]))
    # ratio between root grid cell and the most refined cells
    fp_info.write('maxrefine   %d\n'        % (2**(ro.info["levelmax"]-ro.info["levelmin"])))
    fp_info.write('direction   %s\n'        % direction)
    pc = ro.info["unit_length"].val * 100.0 / PARSEC  # box size in parsecs
    fp_info.write('unit_length %.7e\n'      % pc)
    fp_info.close()
    
    
    # parent cells created for cells on the next deeper level cells
    XP, YP, ZP, RP =  [], [], [], []
    CELLS  = np.zeros(LEVELS, np.int32)
    pcells = 0

    
    print("\n\n") 
    
    for i in range(LEVELS-1, 0, -1):  # loop over CHILD levels, starting with the deepest level
           
        
        print("CHILD LEVEL %d,  SIZE[i] = %.5e" % (i, SIZE[i]))
        
        # From ramses hierarchy, extract all the leafs on this level
        m          =  nonzero(np.abs(S/SIZE[i]-1.0)<0.01)   # cells with size SIZE[i]
        XC         =  np.asarray(X[m], np.float64)
        YC         =  np.asarray(Y[m], np.float64)
        ZC         =  np.asarray(Z[m], np.float64)
        RC         =  np.asarray(R[m], np.int32)
        
        if (SUBVOLUME):    # drop all cells outside the subvolume
            m = nonzero((XC>=CMIN[0])&(XC<CMAX[0])&(YC>=CMIN[1])&(YC<CMAX[1])&(ZC>=CMIN[2])&(ZC<CMAX[2]))
            XC, YC, ZC, RC = XC[m], YC[m], ZC[m], RC[m]
        print('LEVEL %2d, LEAFS %6d = %.3f x 8, <R>=%10.3e' % (i, len(XC), len(XC)/8.0, mean(RC)))
        
        # Concatenate with other cells on this level = cells that are parents
        XC = concatenate((XC, XP))
        YC = concatenate((YC, YP))
        ZC = concatenate((ZC, ZP))
        RC = concatenate((RC, RP))
        CELLS[i] = len(XC)
        print('LEVEL %2d, LEAF+PARENT CELLS %6d = %.3f x 8' % (i, len(XC), len(XC)/8.0))
        if (CELLS[i] % 8 != 0):
            print("Error -- number of cells not divisible by 8 !!")
            sys.exit()
    
        # Sort already on Python side - to speed up kernel sorting
        print('**** SORT ****')
        isort = argsort(XC)   # independent of R !
        XC, YC, ZC, RC  =  XC[isort], YC[isort], ZC[isort], RC[isort] 
        print('**** DONE ****')
        
        # Reorder in SID order
        OFF        = GL / 2.0**(i+1)   # coordinate offset of the first cell in octet
        STEP       = GL / 2.0**i       # cell size of these child cells

        # All sid=0 cells should be at coordinate values OFF + i*STEP
        I     =  Reorder(XC, YC, ZC, OFF, STEP, GPU=GPU) # reorder => all 8 subcells always in SID order

        XC, YC, ZC, RC = XC[I].copy(), YC[I].copy(), ZC[I].copy(), RC[I].copy()
        
        # Save data for the current level i - temporary file
        fp = open('%s/level_%02d.bin' % (TODIR, i), 'w')
        np.asarray([len(RC),], np.int32).tofile(fp)
        np.asarray(RC, np.int32).tofile(fp)
        fp.close()
        
        # Create parents for all the cells and make the links
        if (i>1):
            # just create enough parent cells, order does not matter
            pcells = len(RC)/8  # total number of parent cells on the level i-1 (with just links)
            XP = np.zeros(pcells, np.float64)
            YP = np.zeros(pcells, np.float64)
            ZP = np.zeros(pcells, np.float64)
            RP = np.zeros(pcells,   np.int32)
            for j in range(pcells):  # loop over cells parent cells on level i-1
                XP[j] = XC[8*j] + 0.499*STEP  # 8*j is the index of a sid==0 cell on level i
                YP[j] = YC[8*j] + 0.499*STEP  #   these are the true centre coordinates of the
                ZP[j] = ZC[8*j] + 0.499*STEP  #   parent cell
                RP[j] = -8*j                  # the link = index to the first child in the octet on level i
            print('PARENT CELLS %10.8f %10.8f  %10.8f %10.8f  %10.8f %10.8f' % \
            (np.min(XP), np.max(XP), np.min(YP), np.max(YP), np.min(ZP), np.max(ZP)))
        else:
            # i==1  ==> the parent cells are on the root grid
            # use coordinates to match the children with the parent cells
            pcells = DIM1*DIM2*DIM3
            XP     = np.zeros(pcells, np.float64)
            YP     = np.zeros(pcells, np.float64)
            ZP     = np.zeros(pcells, np.float64)
            RP     = np.zeros(pcells,   np.int32)
            if (SUBVOLUME):
                for j in range(0, CELLS[i], 8):  # loop over the sid==0 children on level i
                    x, y, z =  int((XC[j]-CMIN[0])/GL), int((YC[j]-CMIN[1])/GL), int((ZC[j]-CMIN[2])/GL)
                    ind     =  x+DIM1*(y+DIM2*z) # index of the root grid parent cell
                    RP[ind] = -(j)               # link to the first child in the sub-octet
                # Finish the root level == add the leaf cells
                m          = nonzero(np.abs(S/SIZE[0]-1.0)<0.1)   # leaf cells on the root grid
                XC         = np.asarray(X[m], np.float64)
                YC         = np.asarray(Y[m], np.float64)
                ZC         = np.asarray(Z[m], np.float64)
                RC         = np.asarray(R[m], np.int32)
                # Select only the root cells inside the selected volume
                m = nonzero((XC>=CMIN[0])&(XC<CMAX[0])&(YC>=CMIN[1])&(YC<CMAX[1])&(ZC>=CMIN[2])&(ZC<CMAX[2]))
                XC, YC, ZC, RC = XC[m], YC[m], ZC[m], RC[m]
                # Copy those densities into the cartesian grid
                x, y, z =  np.floor((XC-CMIN[0])/GL), np.floor((YC-CMIN[1])/GL), np.floor((ZC-CMIN[2])/GL)
                x, y, z =  np.asarray(x, np.int32), np.asarray(y, np.int32), np.asarray(z, np.int32)
                ind     =  x+DIM1*(y+DIM2*z)  # indices for the root grid leaf cells
                m = nonzero(RP[ind]!=0)
                if (len(m[0])!=0.0): 
                    print('B: ??????????????') # was already a link ???
                RP[ind] =  RC
                    
            else: # take whole volume
                ### i = child level
                print('child level %d' % i)
                print('%10.8f %10.8f   %10.8f %10.8f   %10.8f %10.8f' % \
                (np.min(XC), np.max(XC), np.min(YC), np.max(YC), np.min(ZC), np.max(ZC)))
                ###
                for j in range(0, CELLS[i], 8):  # loop over the sid==0 children on level i
                    x, y, z =  int(XC[j]/GL), int(YC[j]/GL), int(ZC[j]/GL)
                    ind     =  x+DIM*(y+DIM*z)   # index of the root grid cell
                    RP[ind] =  -j                # link to the first child in octet
                # Finish the root level -- the leaf cells are still missing !!
                m          =  nonzero(np.abs(S/SIZE[0]-1.0)<0.1)   # leaf cells on the root grid
                XC         =  np.asarray(X[m], np.float64)
                YC         =  np.asarray(Y[m], np.float64)
                ZC         =  np.asarray(Z[m], np.float64)
                RC         =  np.asarray(R[m],   np.int32)
                x, y, z    =  np.floor(XC/GL), np.floor(YC/GL), np.floor(ZC/GL)
                x, y, z    =  np.asarray(x, np.int32), np.asarray(y, np.int32), np.asarray(z, np.int32)
                if (0):
                    x       =  clip(x, 0, DIM1-1)
                    y       =  clip(y, 0, DIM2-1)
                    z       =  clip(z, 0, DIM3-1)
                ind     =  x+DIM*(y+DIM*z)             # index of the root grid cells
                m       =  nonzero(RP[ind]!=0)
                if (len(m[0])!=0.0):
                    print('C: ??????????????????') # was already a link ???
                RP[ind] =  RC
            ##
            # Save also this first level
            fp = open('%s/level_%02d.bin' % (TODIR, 0), 'w')
            np.asarray([len(RP),], np.int32).tofile(fp)
            np.asarray(RP, np.int32).tofile(fp)
            fp.close()
            CELLS[0]= len(XP)
    
    # Write the SOCAMO cloud like index file
    fp = open(cloudfile, 'w')
    np.asarray([DIM1, DIM2, DIM3, LEVELS, sum(CELLS)], np.int32).tofile(fp)
    for i in range(LEVELS):
        fpi   = open('%s/level_%02d.bin' % (TODIR, i))
        cells = fromfile(fpi, np.int32, 1)
        x     = fromfile(fpi, np.int32, cells)
        fpi.close()
        np.asarray([cells,], np.int32).tofile(fp)
        np.asarray(x, np.int32).tofile(fp)
    fp.close()
        


    
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


    
# def RamsesToIndex(MODEL, OUTPUT, centre, delta, direction, GPU=False, FROMDIR='', TODIR='./', platforms=[0,1,2,3,4]):
#     """
#     Write an index file. The format is the same as SOC cloud but the main array is np.int32 instead of float.
#     Each cell contains either index >=1 to the Ramses vectors (leaf cells only) or values <=0 meaning links.
#     Input:
#         MODEL     =  name of the model directory (without full path), under which are output_????? directories
#         OUPUT     =  number of the snapshot
#         centre    =  centre coordinates for the extracted cube, centre[3] with values in the range [0,1]
#         delta     =  half of the size of the extracted cube
#         direction =  if given, take full extent in the given direction, value 'x', 'y', or 'z'
#         GPU       =  if true, use GPU instead of CPU
#         FROMDIR   =  root directory, under which is the model directory
#         TODIR     =  output directory where the index file is written
#         platforms =  array of possible OpenCL platforms, default is [0,1,2,3,4]
#     Note:
#         This one uses/used pynbody library but no longer works with the latest version pf pynbody.
#     """    
#     fp_info = None    
#     CMIN    =   np.asarray(centre) - delta
#     CMAX    =   np.asarray(centre) + delta
#     if (  direction=='x'):  # take the full LOS along the selected direction
#         CMIN[0], CMAX[0] = 0.0, 1.0
#     elif (direction=='y'):
#         CMIN[1], CMAX[1] = 0.0, 1.0
#     elif (direction=='z'):
#         CMIN[2], CMAX[2] = 0.0, 1.0
#     CMIN = clip(CMIN, 0.0, 1.0)
#     CMAX = clip(CMAX, 0.0, 1.0)

#     tmp     =   FROMDIR+'/'+MODEL+'/output_%05d' % OUTPUT
#     print("Reading data: %s" % tmp)
#     ro      =  pynbody.load(tmp)
#     pos_pc  =  np.asarray(ro['pos'].in_units('pc'), np.float32)

#     # @@@ NO LONGER WORKING 
#     smo_pc  =  np.asarray(ro['smooth'].in_units('pc'))
    
#     # GL           =  1.0                /  2.0**ro.info["levelmin"]  # root grid cell size
#     SIZE_pc =  float(ro.properties['boxsize'].in_units('pc'))
#     min_pc  =  np.min(smo_pc)                # these will be cells on highest refinement level
#     max_pc  =  np.max(smo_pc)                # this will correspond to root grid cells
#     GL      =  float(max_pc/SIZE_pc)         # size of root grid cell [0,1]
#     GL_pc   =  GL*SIZE_pc
#     ROOTDIM =  int(1.0/GL)                   # root grid size in cells
#     print("DIM=%d, SIZE %.3e pc, GL %.3e pc = %.3e, CELLS %.3e ... %.3e" % (ROOTDIM, SIZE_pc, GL_pc, GL, min_pc, max_pc))
    
#     X       =  np.asarray(pos_pc[:,0]/SIZE_pc, np.float32)   # [0,1]
#     Y       =  np.asarray(pos_pc[:,1]/SIZE_pc, np.float32)   # [0,1]
#     Z       =  np.asarray(pos_pc[:,2]/SIZE_pc, np.float32)   # [0,1]
#     N       =  len(X)
    
#     LMAX    =  int(log(max_pc/min_pc)/log(2.0))
#     LEVELS  =  LMAX+1
#     L       =  np.asarray(log(max_pc/smo_pc)/log(2.0), np.int32)  # L=0, ..., LMAX==LEVELS-1
#     ## SIZE    =  GL  / 2.0**arange(LEVELS)
#     print("LEVELS %d, LMAX %d, sizes %.3e pc ... %.3e pc, ratio %.3e == %.3e" % (LEVELS, LMAX, min_pc, max_pc, 2.0**LMAX, max_pc/min_pc))
    
#     # ok, change R into a running, 1-based index to Ramses vectors (leaf cells only)
#     # vector is np.int32, values <= are the normal links, 
#     # values i>0 =>      cell value  <-  ramses vector value [i-1]
#     # ... or read the full ramses dump =  [cells, val0, val1, ...] as float vector, index with [i]
#     R            =  None
#     R            =  np.asarray(arange(1, N+1), np.int32)  # 1, 2, ..., number of leaf cells
    
#     # Find the dimensions of the root grid
#     # XMIN, XMAX etc. --- minimum and maximum root grid indices [XMIN, XMAX[
#     XMIN, XMAX   =  int(round(CMIN[0]/GL)),  int(round(CMAX[0]/GL))
#     YMIN, YMAX   =  int(round(CMIN[1]/GL)),  int(round(CMAX[1]/GL))
#     ZMIN, ZMAX   =  int(round(CMIN[2]/GL)),  int(round(CMAX[2]/GL))
#     DIM1, DIM2, DIM3 = XMAX-XMIN, YMAX-YMIN, ZMAX-ZMIN
#     print('X  %4d to %4d, DIM %4d' % (XMIN, XMAX, DIM1))
#     print('Y  %4d to %4d, DIM %4d' % (YMIN, YMAX, DIM2))
#     print('Z  %4d to %4d, DIM %4d' % (ZMIN, ZMAX, DIM3))
    
#     SUBVOLUME = False
#     if ((XMIN>0)|(YMIN>0)|(ZMIN>0)): SUBVOLUME = True
#     if ((XMAX<(DIM1-1))|(YMAX<(DIM2-1))|(ZMAX<(DIM3-1))): SUBVOLUME = True
#     print('SUBVOLUME = ', SUBVOLUME)
    
#     # update the coordinate limits == on boundaries of root grid coordinates [XMIN*GL, XMAX*GL]
#     print('COORD LIMITS  %8.5f-%8.5f  %8.5f-%8.5f    %8.5f-%8.5f' % (CMIN[0], CMAX[0], CMIN[1], CMAX[1], CMIN[2], CMAX[2]))
#     CMIN         =  [   XMIN*GL,   YMIN*GL,   ZMIN*GL  ]
#     CMAX         =  [   XMAX*GL,   YMAX*GL,   ZMAX*GL  ]
#     print('      ==>     %8.5f-%8.5f  %8.5f-%8.5f    %8.5f-%8.5f' % (CMIN[0], CMAX[0], CMIN[1], CMAX[1], CMIN[2], CMAX[2]))

#     # ASSUMING INPUT IS A CUBE !!
#     DIM1, DIM2, DIM3 = ROOTDIM, ROOTDIM, ROOTDIM
        
        
#     if (SUBVOLUME):
#         sys.exit()
#         cloudfile = '%s/%s_%06d_%.4f_%.4f_%.4f_%.4f_%.4f_%.4f.index' % \
#         (TODIR, MODEL, OUTPUT, CMIN[0], CMAX[0], CMIN[1], CMAX[1], CMIN[2], CMAX[2])
#     else:
#         cloudfile = '%s/%s_%06d.index' % (TODIR, MODEL, OUTPUT)
                                
#     # Save meta file
#     # Input data in large arrays { X, Y, Z, R, S }
#     fp_info = open(cloudfile.replace('.index', '.meta'), 'w')
#     fp_info.write('boxlen      %.8f\n'      % 1.0)                  # always 1.0 ???
#     fp_info.write('GL          %.8e\n'      % GL)                   #  root cell size / box size
#     fp_info.write('levelmin    %d\n'        % 0)
#     fp_info.write('levelmax    %d\n'        % LMAX)
#     fp_info.write('levels      %d\n'        % LEVELS)
#     fp_info.write('rootgrid    %d %d %d\n'  % (DIM1, DIM2, DIM3))   # full cloud in root cells
#     fp_info.write('dimensions  %d %d %d\n'  % (DIM1, DIM2, DIM3))   # extracted cloud in root cells
#     fp_info.write('xrange      %.8e %.8e\n' % (CMIN[0], CMAX[0]))
#     fp_info.write('yrange      %.8e %.8e\n' % (CMIN[1], CMAX[1]))
#     fp_info.write('zrange      %.8e %.8e\n' % (CMIN[2], CMAX[2]))
#     # ratio between root grid cell and the most refined cells
#     fp_info.write('maxrefine   %d\n'        % (2**LMAX))
#     fp_info.write('direction   %s\n'        % direction)
#     fp_info.write('unit_length %.7e\n'      % SIZE_pc)
#     fp_info.close()
    
    
#     # parent cells created for cells on the next deeper level cells
#     XP, YP, ZP, RP =  [], [], [], []
#     CELLS  = np.zeros(LEVELS, np.int32)
#     pcells = 0
#     print("\n\n") 
    
#     for i in range(LEVELS-1, 0, -1):  # loop over CHILD levels, starting with the deepest level
           
#         print("CHILD LEVEL %d" % i)
#         # From ramses hierarchy, extract all the leafs on this level
#         # m          =  nonzero(abs(S/SIZE[i]-1.0)<0.01)   # cells with size SIZE[i]
#         # select cells on the hierarchy level i.... only  leaf cells 
#         m          =  nonzero(L==i)

#         # data on the cells on the child level... in some order
#         XC         =  np.asarray(X[m], np.float64)
#         YC         =  np.asarray(Y[m], np.float64)
#         ZC         =  np.asarray(Z[m], np.float64)
#         RC         =  np.asarray(R[m], np.int32)
        
#         if (SUBVOLUME):    # drop all cells outside the subvolume
#             m = nonzero((XC>=CMIN[0])&(XC<CMAX[0])&(YC>=CMIN[1])&(YC<CMAX[1])&(ZC>=CMIN[2])&(ZC<CMAX[2]))
#             XC, YC, ZC, RC = XC[m], YC[m], ZC[m], RC[m]
#         print('LEVEL %2d, LEAFS %6d = %.3f x 8, <R>=%10.3e' % (i, len(XC), len(XC)/8.0, mean(RC)))
        
#         # Concatenate with other cells on this level = cells that are parents
#         XC = concatenate((XC, XP))
#         YC = concatenate((YC, YP))
#         ZC = concatenate((ZC, ZP))
#         RC = concatenate((RC, RP))
#         CELLS[i] = len(XC)
#         print('LEVEL %2d, LEAF+PARENT CELLS %6d = %.3f x 8' % (i, len(XC), len(XC)/8.0))
#         if (CELLS[i] % 8 != 0):
#             print("Error -- number of cells not divisible by 8 !!")
#             sys.exit()
    
#         # Sort already on Python side - to speed up kernel sorting
#         print('**** SORT ****')
#         isort = argsort(XC)   # independent of R !
#         XC, YC, ZC, RC  =  XC[isort], YC[isort], ZC[isort], RC[isort] 
#         print('**** DONE ****')
        
#         # Reorder in SID order
#         OFF        = GL / 2.0**(i+1)   # coordinate offset of the first cell in octet
#         STEP       = GL / 2.0**i       # cell size of these child cells

#         # All sid=0 cells should be at coordinate values OFF + i*STEP
#         I     =  Reorder(XC, YC, ZC, OFF, STEP, GPU=GPU) # reorder => all 8 subcells always in SID order

#         XC, YC, ZC, RC = XC[I].copy(), YC[I].copy(), ZC[I].copy(), RC[I].copy()
        
#         # Save data for the current level i - temporary file
#         fp = open('%s/level_%02d.bin' % (TODIR, i), 'w')
#         np.asarray([len(RC),], np.int32).tofile(fp)
#         np.asarray(RC, np.int32).tofile(fp)
#         fp.close()
        
#         # Create parents for all the cells and make the links
#         if (i>1):
#             # just create enough parent cells, order does not matter
#             pcells = len(RC)//8               # total number of parent cells on the level i-1 (with just links)
#             XP = np.zeros(pcells, np.float64)
#             YP = np.zeros(pcells, np.float64)
#             ZP = np.zeros(pcells, np.float64)
#             RP = np.zeros(pcells,   np.int32)
#             for j in range(pcells):           # loop over cells parent cells on level i-1
#                 XP[j] = XC[8*j] + 0.499*STEP  # 8*j is the index of a sid==0 cell on level i
#                 YP[j] = YC[8*j] + 0.499*STEP  #   these are the true centre coordinates of the
#                 ZP[j] = ZC[8*j] + 0.499*STEP  #   parent cell
#                 RP[j] = -8*j                  # the link = index to the first child in the octet on level i
#             print('PARENT CELLS %10.8f %10.8f  %10.8f %10.8f  %10.8f %10.8f' % \
#             (np.min(XP), np.max(XP), np.min(YP), np.max(YP), np.min(ZP), np.max(ZP)))
#         else:
#             # i==1  ==> the parent cells are on the root grid
#             # use coordinates to match the children with the parent cells
#             pcells = DIM1*DIM2*DIM3
#             XP     = np.zeros(pcells, np.float64)
#             YP     = np.zeros(pcells, np.float64)
#             ZP     = np.zeros(pcells, np.float64)
#             RP     = np.zeros(pcells,   np.int32)
#             if (SUBVOLUME):
#                 for j in range(0, CELLS[i], 8):  # loop over the sid==0 children on level i
#                     x, y, z =  int((XC[j]-CMIN[0])/GL), int((YC[j]-CMIN[1])/GL), int((ZC[j]-CMIN[2])/GL)
#                     ind     =  x+DIM1*(y+DIM2*z) # index of the root grid parent cell
#                     RP[ind] = -(j)               # link to the first child in the sub-octet
#                 # Finish the root level == add the leaf cells
#                 # m          = nonzero(abs(S/SIZE[0]-1.0)<0.1)   # leaf cells on the root grid
#                 m          = nonzero(L==0)
#                 XC         = np.asarray(X[m], np.float64)
#                 YC         = np.asarray(Y[m], np.float64)
#                 ZC         = np.asarray(Z[m], np.float64)
#                 RC         = np.asarray(R[m], np.int32)
#                 # Select only the root cells inside the selected volume
#                 m = nonzero((XC>=CMIN[0])&(XC<CMAX[0])&(YC>=CMIN[1])&(YC<CMAX[1])&(ZC>=CMIN[2])&(ZC<CMAX[2]))
#                 XC, YC, ZC, RC = XC[m], YC[m], ZC[m], RC[m]
#                 # Copy those densities into the cartesian grid
#                 x, y, z =  np.floor((XC-CMIN[0])/GL), np.floor((YC-CMIN[1])/GL), np.floor((ZC-CMIN[2])/GL)
#                 x, y, z =  np.asarray(x, np.int32), np.asarray(y, np.int32), np.asarray(z, np.int32)
#                 ind     =  x+DIM1*(y+DIM2*z)  # indices for the root grid leaf cells
#                 m = nonzero(RP[ind]!=0)
#                 if (len(m[0])!=0.0): 
#                     print('B: ??????????????') # was already a link ???
#                 RP[ind] =  RC
                    
#             else: # take whole volume
#                 ### i = child level
#                 print('child level %d' % i)
#                 print('%10.8f %10.8f   %10.8f %10.8f   %10.8f %10.8f' % \
#                 (np.min(XC), np.max(XC), np.min(YC), np.max(YC), np.min(ZC), np.max(ZC)))
#                 ###
#                 for j in range(0, CELLS[i], 8):   # loop over the sid==0 children on level i
#                     x, y, z =  int(XC[j]/GL), int(YC[j]/GL), int(ZC[j]/GL)
#                     ind     =  x+DIM1*(y+DIM2*z)  # index of the root grid cell
#                     RP[ind] =  -j                 # link to the first child in octet
#                 # Finish the root level -- the leaf cells are still missing !!
#                 # m          =  nonzero(abs(S/SIZE[0]-1.0)<0.1)   # leaf cells on the root grid
#                 m          =  nonzero(L==0)
#                 XC         =  np.asarray(X[m], np.float64)
#                 YC         =  np.asarray(Y[m], np.float64)
#                 ZC         =  np.asarray(Z[m], np.float64)
#                 RC         =  np.asarray(R[m],   np.int32)
#                 x, y, z    =  np.floor(XC/GL), np.floor(YC/GL), np.floor(ZC/GL)
#                 x, y, z    =  np.asarray(x, np.int32), np.asarray(y, np.int32), np.asarray(z, np.int32)
#                 if (0):
#                     x       =  clip(x, 0, DIM1-1)
#                     y       =  clip(y, 0, DIM2-1)
#                     z       =  clip(z, 0, DIM3-1)
#                 ind     =  x+DIM1*(y+DIM2*z)       # index of the root grid cells
#                 m       =  nonzero(RP[ind]!=0)
#                 if (len(m[0])!=0.0):
#                     print('C: ??????????????????') # was already a link ???
#                 RP[ind] =  RC
#             ##
#             # Save also this first level
#             fp = open('%s/level_%02d.bin' % (TODIR, 0), 'w')
#             np.asarray([len(RP),], np.int32).tofile(fp)
#             np.asarray(RP, np.int32).tofile(fp)
#             fp.close()
#             CELLS[0]= len(XP)
    
#     # Write the SOC-cloud-like index file
#     fp = open(cloudfile, 'w')
#     np.asarray([DIM1, DIM2, DIM3, LEVELS, sum(CELLS)], np.int32).tofile(fp)
#     for i in range(LEVELS):
#         fpi   = open('%s/level_%02d.bin' % (TODIR, i))
#         cells = fromfile(fpi, np.int32, 1)[0]
#         print(cells)
#         x     = fromfile(fpi, np.int32, cells)
#         fpi.close()
#         np.asarray([cells,], np.int32).tofile(fp)
#         np.asarray(x, np.int32).tofile(fp)
#     fp.close()
        


    
# def RamsesDumpField(MODEL, OUTPUT, TAG='rho', FROMDIR='', TODIR='./'):
#     """
#     Dump rho and magnetic field components to plain binary files.
#     We must make sure none of the values is zero (?)
#     Input:
#         MODEL    =   e.g. SN50_9_13_sg   (run name)
#         OUTPUT   =   e.g. 377 (snapshot)
#         TAG      =   data field (rho, vel, B, ...) 
#         FROMDIR  =   directory containing MODEL as subdirectory
#         TODIR    =   directory where the output file is written
#     Note:
#         2021-03-09:
#             RAMSES file presumably in Fortran order
#             SOC reads the file in C-order=>  original Z becomes X, X becomes Z
#             ==> we must also switch velocity fields VX -> VZ, VZ -> VX
#     """    
#     # ro           =  osyris.Dataset(OUTPUT, "%s/%s/" % (FROMDIR, MODEL))
#     ro = pynbody.load(FROMDIR+'/'+MODEL+'/output_%05d' % OUTPUT)    
#     # amr          =  ro.amr_source(['rho', 'Bl', 'Br'])
#     prefix  =  '%s/dump_%s_%06d' % (TODIR, MODEL, OUTPUT)
#     if (TAG=='vel'):            
#         v     =  np.asarray(ro['vel'].in_units('km s**-1'), np.float32)
#         N     =  v.shape[0]
#         m     =  nonzero(np.abs(v)<1.0e-6)
#         v[m] +=  1.0e-4*randn(N)        
#         for i in range(3):
#             tag = ['x', 'y', 'z'][i]
#             fp    =  open(prefix+'.v%s' % tag, 'wb')   # FORTRAN NAMING
#             np.asarray([N,], np.int32).tofile(fp)
#             np.asarray(v[:,i], np.float32).tofile(fp)
#             fp.close()
#     elif (TAG=='B'):  # name ??? ... one can call for individual Bx, By, Bz instead
#         B  =  np.asarray(ro['B'].in_units('Gauss'), np.float32)
#         N  =  B.shape[0]
#         for i in range(3):
#             tag  =  ['x', 'y', 'z'][i]
#             fp   =  open(prefix+'.B%s' % tag, 'wb')    # FORTRAN NAMING
#             np.asarray([N,], np.int32).tofile(fp)
#             np.asarray(B[:,i], np.float32).tofile(fp)
#             fp.close()
#     else:
#         if (TAG=='rho'):
#             X  = np.asarray(ro[TAG].in_units('1.67e-24 g cm**-3'), np.float32)  # as n(H)
#             X[nonzero(X<=1.0e-10)] = 1.0e-10  # density must be non-negative !
#         else:
#             X  = np.asarray(ro[TAG], np.float32)
#         fp = open(prefix+'.'+TAG, 'wb')
#         np.asarray([len(X),], np.int32).tofile(fp)
#         X.tofile(fp)
#         fp.close()

    
    
    
    

        
        
        
        
def YT2SOC(INPUT, SOCFILE, GPU=False, platforms=[0,1,2,3,4], MAXL=999):
    """
    Write an index file. The format is the same as SOC cloud but the main array is np.int32 instead of float.
    Each cell contains either index >=1 to the data vectors (leaf cells only) or values <=0 meaning links.
    Usage:
        YT2SOC(INPUT, SOCFILE, GPU=False, platforms=[0,1,2,3,4], MAXL=999)
    Input:
        INPUT     =  name of the HDF5 file
        SOCFILE   =  name of the output file
        GPU       =  if true, use GPU instead of CPU
        platforms =  array of possible OpenCL platforms, default is [0,1,2,3,4]
        MAXL      =  maximum hierarchy level
    """    
    print("Reading data: %s" % INPUT)
    ds     =  yt.load(INPUT)
    LE_pc  =  ds.index.domain_left_edge.in_units( 'pc').value
    RE_pc  =  ds.index.domain_right_edge.in_units('pc').value    
    LMAX   =  np.min([ds.index.max_level, MAXL])
    LEVELS =  LMAX+1
    # size of the cells on the root grid
    gs     =  ds.index.select_grids(0)
    g      =  gs[0]
    GL_pc  =  (g.RightEdge.in_units('pc').value[0]-g.LeftEdge.in_units('pc').value[0])/g.shape[0]
    DIM    =  int(round((RE_pc[0]-LE_pc[0])/GL_pc))
    # check the unit of density
    unit   =  gs[0]['density'].units
    coeff  =  1.0
    if (unit==(yt.units.g / yt.units.cm**3)):
        coeff =  1.0/(1.4*AMU)  # conversion from file units to n(H)
    else:
        print("Yt read data in units [%s] ???" % unit)
        sys.exit()
    # create hierarchy based on coordinates only, at this point no data is involved
    OFF    =  np.zeros(LEVELS, np.int32)
    LCELLS =  np.zeros(LEVELS, np.int32)
    for ilevel in range(LEVELS):
        print("LEVEL %d" % ilevel)        
        if (ilevel>0): 
            OFF[ilevel] = OFF[ilevel-1] + LCELLS[ilevel-1]
        gs   =   ds.index.select_grids(ilevel)
        n    =   0
        for igrid in range(gs.size): # loop over grids on level ilevel
            n  +=  product(gs[igrid].shape)
        LCELLS[ilevel] = n
    CELLS = sum(LCELLS)
    print("DIM=%d, GL=%.4f pc, LEVELS=%d" % (DIM, GL_pc, LEVELS))
    for ilevel in range(LEVELS):
        print(" %d  OFF %10d   LCELLS  %10d = %.3e cells" % (ilevel, OFF[ilevel], LCELLS[ilevel], LCELLS[ilevel]))
    print("  ---------------------------------------------------")
    print("    %10d = %.3e cells" % (CELLS, CELLS))
    X  =  np.zeros(CELLS, np.float32)
    Y  =  np.zeros(CELLS, np.float32)
    Z  =  np.zeros(CELLS, np.float32)
    H  =  np.zeros(CELLS, np.float32)
    # loop over levels and grid the second time, adding the coordinate and density values
    ii =  0  # running index to the final X, Y, Z, H vectors
    for ilevel in range(LEVELS):
        print("LEVEL %d" % ilevel)
        # we assume child grid has cell size half of the parent AND that the grids 
        # have even dimensions (should have, if subgrid fills integer number of parent cells)
        N = LCELLS[ilevel]
        print('    PARENT LEVEL %2d, LCELLS %6d = %.3f x 8' % (ilevel, N, N/8.0))
        if (ilevel<(LEVELS-1)):
            if (LCELLS[ilevel+1] % 8 != 0):
                print("    Error -- number of cells on child level not divisible by 8 !!")
                sys.exit()    
        # copy the cell coordinates to XC, YC, ZC in grid units [0,DIM]
        gs =  ds.index.select_grids(ilevel)
        for igrid in range(gs.size):   # loop over grids on level ilevel
            C       =  gs[igrid].fcoords.in_units('pc').value.copy()  # coordinates for this one grid
            if (igrid<0):
                print("####  LEVEL %d =>  GRID START  %8.4f %8.4f %8.4f     %8.4f %8.4f %8.4f" %
                (ilevel,  C[0,0], C[0,1], C[0,2],  (C[0,0]-LE_pc[0])/GL_pc, (C[0,1]-LE_pc[1])/GL_pc,
                 (C[0,2]-LE_pc[2])/GL_pc))
            C[:,0]  =  (C[:,0]-LE_pc[0])/GL_pc  # convert to root level coordinates in GL units [0,DIM]
            C[:,1]  =  (C[:,1]-LE_pc[1])/GL_pc  # root grid cells at  0.5, 1.5, 2.5, ...
            C[:,2]  =  (C[:,2]-LE_pc[2])/GL_pc  
            n       =  C.shape[0]    # n cells to add from the current level, current grid
            ###
            X[ii:(ii+n)]  = C[:,0]
            Y[ii:(ii+n)]  = C[:,1]
            Z[ii:(ii+n)]  = C[:,2]
            H[ii:(ii+n)]  = ravel(gs[igrid]['density'].value.copy())
            ii += n    # added n cells from level ilevel grid igrid
        if (ilevel==0):
            x0, x1 = np.min(X[0:LCELLS[0]]), np.max(X[0:LCELLS[0]])
            y0, y1 = np.min(Y[0:LCELLS[0]]), np.max(Y[0:LCELLS[0]])
            z0, z1 = np.min(Z[0:LCELLS[0]]), np.max(Z[0:LCELLS[0]])
            print("ROOT GRID [%.2f,%.2f]  [%.2f,%.2f]  [%.2f,%.2f]" % (x0,x1,y0,y1,z0,z1))        
        # Kernel search assumes z-coordinates are in order => sort always!! (at least wrt z)
        # ... root must be ordered using all z,y,x !!
        assert((OFF[ilevel]+LCELLS[ilevel])==ii)
        ccc = 2.0**(ilevel+1)
        #   coordinates C  =>   C*2^(ilevel+1)  should be ~integers  1, 3, 5, 7, 9,....
        #                       C*2^(ilevel+1) can go up to DIM*2^(ilevel+1)
        CCC = DIM*2.0**(ilevel+1) + 1.0
        q   =       np.asarray(np.round(ccc*X[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])]), int64)   \
        +       CCC*asarray(np.round(ccc*Y[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])]), int64)   \
        +   CCC*CCC*asarray(np.round(ccc*Z[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])]), int64)
        I = argsort(q)
        X[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])] = X[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])].copy()[I]
        Y[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])] = Y[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])].copy()[I]
        Z[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])] = Z[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])].copy()[I]
        H[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])] = H[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])].copy()[I]

        print("  level %2d   %10d - %10d    rho %12.4e - %12.4e" %
        (ilevel, OFF[ilevel], OFF[ilevel]+LCELLS[ilevel],
        np.min(H[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])]), np.max(H[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])])))
    
        
    # AT THIS POINT ALL VALUES IN H > 0  REAL DENSITIES
    H *= coeff                   # to N(H)
    H = clip(H, 1.0e-6, 1.0e20)  # make sure there are no non-positive values
    print("FULL H %.4e ... %.4e" % (np.min(H), np.max(H)))

    
    
    # Now we have all cells in the hierarchy vector H that at this point contains only densities
    # Cell coordinates are in (X, Y, Z)
    #  for each parent level
    #    for each cell on the parent level
    #      if there is a child cell, replace H value with the link
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms)
    LOCAL    =  [ 8, 32 ][GPU>0]
    # source   =  open(HOMEDIR+"/starformation/Python/MJ/MJ/Aux/kernel_OT_tools_yt.c").read()
    source   =  KernelSource("kernel_OT_tools_yt.c")
    OPT      =  ' '
    program  =  cl.Program(context, source).build(OPT)
    # Sort for the root grid cells
    # Sort  = program.Sort
    #                           DIM       N         X     Y     Z     H     HH   
    # Sort.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None, None, None])
    # Link1 identifies parent cells with children
    Link1 = program.Link1
    #                            NP        D           XP    YP    ZP    HP    NC        XC    YC    ZC  
    Link1.set_scalar_arg_dtypes([np.int32, np.float32, None, None, None, None, np.int32, None, None, None])
    # Link2 reorders children to match the links set for the parent cells
    Link2 = program.Link2
    #                            NP*       I     D           XP    YP    ZP
    Link2.set_scalar_arg_dtypes([np.int32, None, np.float32, None, None, None,
    #                            NC                          XC    YC    ZC    HC    xc    yc    zc    hc  
                                 np.int32,                   None, None, None, None, None, None, None, None])
    # 
    max_cells  =  np.int64(np.max(LCELLS))
    GLOBAL     =  (max_cells//LOCAL+1)*LOCAL
    XP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)
    YP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)
    ZP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)
    HP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)
    #
    XC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*max_cells)
    YC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*max_cells)
    ZC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*max_cells)
    HC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*max_cells)  # input for kernel
    xc_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)
    yc_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)
    zc_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)
    hc_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*max_cells)  # output for kernel
    #
    I_buf      =  cl.Buffer(context, mf.READ_ONLY,  4*(max_cells//8))  # cells with children
    TMP        =  np.zeros(max_cells, np.float32)
    print()
    for ilevel in range(LEVELS-1): # loop over parent levels
        print("Linking parent level %d" % ilevel)
        cl.enqueue_copy(queue, XP_buf, X[OFF[ilevel  ]:(OFF[ilevel  ]+LCELLS[ilevel  ])])
        cl.enqueue_copy(queue, YP_buf, Y[OFF[ilevel  ]:(OFF[ilevel  ]+LCELLS[ilevel  ])])
        cl.enqueue_copy(queue, ZP_buf, Z[OFF[ilevel  ]:(OFF[ilevel  ]+LCELLS[ilevel  ])])
        cl.enqueue_copy(queue, HP_buf, H[OFF[ilevel  ]:(OFF[ilevel  ]+LCELLS[ilevel  ])])
        cl.enqueue_copy(queue, XC_buf, X[OFF[ilevel+1]:(OFF[ilevel+1]+LCELLS[ilevel+1])])
        cl.enqueue_copy(queue, YC_buf, Y[OFF[ilevel+1]:(OFF[ilevel+1]+LCELLS[ilevel+1])])
        cl.enqueue_copy(queue, ZC_buf, Z[OFF[ilevel+1]:(OFF[ilevel+1]+LCELLS[ilevel+1])])
        cl.enqueue_copy(queue, HC_buf, H[OFF[ilevel+1]:(OFF[ilevel+1]+LCELLS[ilevel+1])])

        
        # Link1 will check, which cells IND on level ilevel have children on level ilevel+1
        #  and will set HP[IND]=-1.0 for those
        Link1(queue, [GLOBAL,], [LOCAL,], 
        LCELLS[ilevel],  np.float32(1.0/2.0**ilevel),   XP_buf, YP_buf, ZP_buf,    HP_buf,   
                         LCELLS[ilevel+1],              XC_buf, YC_buf, ZC_buf)
        cl.enqueue_copy(queue, TMP, HP_buf)
        print("    ... Link1 done")
        
        # At this point we only know which cells have children, not yet where the children are
        print("    ... host setting links")
        m  =  nonzero(TMP[0:LCELLS[ilevel]]<-1.0)[0]  # parent cells with value -2.0 set
        print("    ... %d parent cells found" % len(m))
        for i in range(len(m)):                       # loop over parent cells, i-th parent has 1st child at 8*i
            H[OFF[ilevel]+m[i]]  =  -I2F(8*i)
            
        # Providing kernel a list of parent cells = m
        cl.enqueue_copy(queue, I_buf, np.asarray(m, np.int32))   # kernel Link2 gets parent indices, does not need HP
        TMP[:] = 1e10
        cl.enqueue_copy(queue, hc_buf, TMP)   # all elements of hc should be set below
        # Link2 to reorder children to match the above linking, HC -> hc
        print("    ... Link2")        
        Link2(queue, [GLOBAL,], [LOCAL,],
        np.int32(len(m)), I_buf, np.float32(1.0/2.0**ilevel),   
        XP_buf, YP_buf, ZP_buf,  
        LCELLS[ilevel+1],   XC_buf, YC_buf, ZC_buf, HC_buf,  xc_buf, yc_buf, zc_buf, hc_buf)
        print("    ... Link2 done")        
        #
        cl.enqueue_copy(queue, TMP, hc_buf)   # child cells in the updated order
        H[OFF[ilevel+1]:(OFF[ilevel+1]+LCELLS[ilevel+1])] = TMP[0:LCELLS[ilevel+1]]  # updated child level
        cl.enqueue_copy(queue, TMP, xc_buf) 
        X[OFF[ilevel+1]:(OFF[ilevel+1]+LCELLS[ilevel+1])] = TMP[0:LCELLS[ilevel+1]]
        cl.enqueue_copy(queue, TMP, yc_buf) 
        Y[OFF[ilevel+1]:(OFF[ilevel+1]+LCELLS[ilevel+1])] = TMP[0:LCELLS[ilevel+1]]
        cl.enqueue_copy(queue, TMP, zc_buf) 
        Z[OFF[ilevel+1]:(OFF[ilevel+1]+LCELLS[ilevel+1])] = TMP[0:LCELLS[ilevel+1]]
        # Second kernel
        
                
    # Write the SOC-cloud
    fp = open(SOCFILE, 'wb')
    np.asarray([DIM, DIM, DIM, LEVELS, sum(LCELLS)], np.int32).tofile(fp)
    for ilevel in range(LEVELS):
        np.asarray([LCELLS[ilevel],], np.int32).tofile(fp)
        np.asarray(H[OFF[ilevel]:(OFF[ilevel]+LCELLS[ilevel])], np.float32).tofile(fp)
    fp.close()
        




def YT2SOC_RAMSES(INPUT, SOCFILE, GPU=False, platforms=[0,1,2,3,4], LOCFILE="", OCTANT=-1):
    """
    Added in 2024, using the YT library: export RAMSES data to SOC and LOC files.
    Input:
        INPUT     =   RAMSES directory (e.g. ./outpt_00176)
        SOCFILE   =   name for the saved SOC cloud file
        GPU       =   if >0, use GPU instead of CPU (default GPU=False)
        platforms =   list of possible OpenCL platforms (defailt [0,1,2,3,4])
        LOCFILE   =   name of output LOC file (optional).
    Return:
        Nothing is returned but SOCFILE and optionally LOCFILE are written to disk
    Note:
        In the LOC file:
            -  Tkin is set to constant 1.0 (K) and must be rescaled using LOC ini file
            -  SIGMA (microturbulence) is set to constant 1.0 (km/s) and must be
               calculated separately, e.g. using Update_LOC_turbulence()
    """
    print("=== YT2SOC_RAMSES create SOC file===")
    print("--- Read RAMSES coordinates")
    t111     =  time.time()
    if (0):
        ds       =  yt.load(INPUT)
    else:
        fields = [ "Density", "x-velocity", "y-velocity", "z-velocity" ]
        ds       =  yt.load(INPUT, fields=fields)
    # ds =  yt.load(INPUT, bbox=BBOX)
    ad       =  ds.all_data()
    DIM      =  ds.domain_dimensions[0]
    NX, NY, NZ = DIM, DIM, DIM
    LMAX     =  ds.max_level
    LEVELS   =  LMAX+1

    if (0):
        x      =  np.asarray(ad["ramses", "x"], np.float32)*DIM   #   [0,DIM] = GL coordinates
        y      =  np.asarray(ad["ramses", "y"], np.float32)*DIM
        z      =  np.asarray(ad["ramses", "z"], np.float32)*DIM
    else:
        l0     =  ds.length_unit  # if not in root-grid units
        x      =  (asarray(ad["ramses", "x"], np.float64)/l0)*DIM
        y      =  (asarray(ad["ramses", "y"], np.float64)/l0)*DIM
        z      =  (asarray(ad["ramses", "z"], np.float64)/l0)*DIM
        # x      =  asarray(x, float32)
        # y      =  asarray(y, float32)
        # z      =  asarray(z, float32)


    cells = len(x)
    print("YT2SOC_RAMSES  OCTANT %2d, %d x %d x %d,  cells %d, %.3e" % (OCTANT, NX, NY, NZ, cells, cells))        
    print(" x  %9.5f %9.5f" % (np.min(x), np.max(x)))
    print(" y  %9.5f %9.5f" % (np.min(y), np.max(y)))
    print(" z  %9.5f %9.5f" % (np.min(z), np.max(z)))

    mo = None
    if (OCTANT>=0):
        NX = NX//2
        NY = NY//2
        NZ = NZ//2
        if   (OCTANT==0):
            mo = nonzero((x<NX)&(y<NY)&(z<NZ))
        elif (OCTANT==1):
            mo = nonzero((x>=NX)&(y<NY)&(z<NZ))
            x -= NX
        elif (OCTANT==2):
            mo = nonzero((x<NX)&(y>=NY)&(z<NZ))
            y -= NY
        elif (OCTANT==3):
            mo = nonzero((x>=NX)&(y>NY)&(z<NZ))
            x -= NX
            y -= NY
        elif (OCTANT==4):
            mo = nonzero((x<NX)&(y<NY)&(z>=NZ))
            z -= NZ
        elif (OCTANT==5):
            mo = nonzero((x>=NX)&(y<NY)&(z>=NZ))
            x -= NX
            z -= NZ
        elif (OCTANT==6):
            mo = nonzero((x<NX)&(y>=NY)&(z>=NZ))
            y -= NY
            z -= NZ
        elif (OCTANT==7):
            mo = nonzero((x>=NX)&(y>NY)&(z>=NZ))
            x -= NX
            y -= NY
            z -= NZ
        #
        x = x[mo]
        y = y[mo]
        z = z[mo]

    x      =  asarray(x, float32)
    y      =  asarray(y, float32)
    z      =  asarray(z, float32)    
    ORD    =  argsort(z)
    x      =  x[ORD]
    y      =  y[ORD]
    z      =  z[ORD]

    cells = len(x)
    print("YT2SOC_RAMSES  OCTANT %2d, %d x %d x %d,  cells %d, %.3e" % (OCTANT, NX, NY, NZ, cells, cells))        
    print(" x  %9.5f %9.5f" % (np.min(x), np.max(x)))
    print(" y  %9.5f %9.5f" % (np.min(y), np.max(y)))
    print(" z  %9.5f %9.5f" % (np.min(z), np.max(z)))

    print("================================================================================")
    print("--- Make SOC hierarchy  =>   OT_points_to_octree...")
    OT_points_to_octree(x, y, z, NX, NY, NZ, LEVELS, SOCFILE, GPU, PLF=platforms)
    print("--- Make SOC hierarchy  =>   OT_points_to_octree ... done")
    print("================================================================================")
        
    del x, y, z
    NX, NY, NZ, LCELLS, OFF, H = OT_ReadHierarchyV(SOCFILE)
    # each leaf H>0.0 is equal to  *(float *)&(1+index_to_yt_vectors)
    m    =  nonzero(H>0.0)     #  int=1  =>  1.401298464324817e-45  !!!
    ind  =  np.zeros(len(m[0]), np.int32)
    for i in range(len(m[0])):           # ind = YT vector indices for leaves.. indices to z-ordered vectors
        ind[i] = F2I( H[m[0][i]] ) - 1   # convert back to YT indices

    if (OCTANT<0):
        rho  =  np.asarray(ad["gas", "density"], np.float32)[ORD]  # reorder to z-order
    else:
        rho  =  np.asarray(ad["gas", "density"], np.float32)[mo][ORD]
                
    H[m] =  rho[ind]
    del rho    
    print("--- Write SOC file")
    t0 = time.time()
    OT_WriteHierarchyV(NX, NY, NZ, LCELLS, OFF, H, SOCFILE)    
    print(" === YT2SOC_RAMSES completed in %.1f seconds ===" % (time.time()-t111))
    
    if (len(LOCFILE)>0):
        print(" === YT2SOC_2024 create LOC file ===")
        t01    =  time.time()
        cells  =  len(H)
        # note the rearrangement of velocity components (z,y,x) -> (x,y,z)
        if (OCTANT<0):
            VX     =  np.zeros(cells, np.float32)
            VX[m]  =  np.asarray(ad["gas", "velocity_z"], np.float32)[ORD][ind] 
            VY     =  np.zeros(cells, np.float32)
            VY[m]  =  np.asarray(ad["gas", "velocity_y"], np.float32)[ORD][ind] 
            VZ     =  np.zeros(cells, np.float32)
            VZ[m]  =  np.asarray(ad["gas", "velocity_x"], np.float32)[ORD][ind] 
            CHI    =  np.ones(cells, np.float32)
        else:
            VX     =  np.zeros(cells, np.float32)
            VX[m]  =  np.asarray(ad["gas", "velocity_z"], np.float32)[mo][ORD][ind] 
            VY     =  np.zeros(cells, np.float32)
            VY[m]  =  np.asarray(ad["gas", "velocity_y"], np.float32)[mo][ORD][ind] 
            VZ     =  np.zeros(cells, np.float32)
            VZ[m]  =  np.asarray(ad["gas", "velocity_x"], np.float32)[mo][ORD][ind] 
            CHI    =  np.ones(cells, np.float32)
        S      =  np.ones(cells, np.float32)
        T      =  np.ones(cells, np.float32)
        OT_WriteHierarchyV_LOC(NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ, CHI, LOCFILE)
        print(" === YT2SOC_RAMSES LOC created in %.1f seconds ===" % (time.time()-t01))
        



        
def YT2SOC_RAMSES_64(INPUT, SOCFILE, GPU=False, platforms=[0,1,2,3,4]):
    """
    Added in 2024, using the YT library: export RAMSES data to SOC.
    SOC files written into 4x4x4 separate files.
    Input:
        INPUT     =   RAMSES directory (e.g. ./outpt_00176)
        SOCFILE   =   name for the saved SOC cloud file
        GPU       =   if >0, use GPU instead of CPU (default GPU=False)
        platforms =   list of possible OpenCL platforms (defailt [0,1,2,3,4])
        LOCFILE   =   name of output LOC file (optional).
    Return:
        Nothing is returned but SOCFILE and optionally LOCFILE are written to disk
    Note:
        In the LOC file:
            -  Tkin is set to constant 1.0 (K) and must be rescaled using LOC ini file
            -  SIGMA (microturbulence) is set to constant 1.0 (km/s) and must be
               calculated separately, e.g. using Update_LOC_turbulence()
    """
    print("=== YT2SOC_RAMSES create SOC file===")
    print("--- Read RAMSES coordinates")
    t111     =  time.time()

    if (1):
        fields = [ "Density", "x-velocity", "y-velocity", "z-velocity" ]
        ds       =  yt.load(INPUT, fields=fields)
        ad       =  ds.all_data()
        DIM      =  ds.domain_dimensions[0]
        NX, NY, NZ = DIM, DIM, DIM
        LMAX     =  ds.max_level
        LEVELS   =  LMAX+1
        l0     =  float(ds.length_unit)  # if not in root-grid units
        print('l0 ', l0, type(l0))
        print("LEVELS %d" % LEVELS)
        # .to_value(),    .ndarray_view()
        x0     =  np.asarray( ad["ramses", "x"].to_value() / l0,  np.float64)*DIM
        y0     =  np.asarray( ad["ramses", "y"].to_value() / l0,  np.float64)*DIM
        z0     =  np.asarray( ad["ramses", "z"].to_value() / l0,  np.float64)*DIM
        rho0   =  np.asarray( ad["gas", "density"].to_value(), np.float32)
        # Switch velocities  (z,y,x) -> (x, y,z)  from Fortran order to order
        vx0    =  np.asarray( ad["gas", "velocity_z"].to_value('km/s'), np.float32)
        vy0    =  np.asarray( ad["gas", "velocity_y"].to_value('km/s'), np.float32)
        vz0    =  np.asarray( ad["gas", "velocity_x"].to_value('km/s'), np.float32)
        #
        del ad
        del ds
        #        
        if (1):
            x0.tofile(  'x0.bin')
            y0.tofile(  'y0.bin')
            z0.tofile(  'z0.bin')
            rho0.tofile('rho0.bin')
            vx0.tofile( 'vx0.bin')
            vy0.tofile( 'vy0.bin')
            vz0.tofile( 'vz0.bin')
            asarray([LMAX, LEVELS, DIM],  int32).tofile('dim.bin')
            sys.exit()
    else:
        x0   = fromfile('x0.bin', np.float64)
        y0   = fromfile('y0.bin', np.float64)
        z0   = fromfile('z0.bin', np.float64)
        rho0 = fromfile('z0.bin', np.float32)
        if (0):
            LMAX, LEVELS, DIM = fromfile('dim.bin', int32)
        else:
            LMAX   = 5
            LEVELS = 6
            NX, NY, NZ = 512, 512, 512
                

            
    cells = len(x0)
    print("================================================================================")
    print("================================================================================")
    print("YT2SOC_RAMSES %d x %d x %d,  cells %d, %.3e" % (NX, NY, NZ, cells, cells))        
    print(" x    %9.5f %9.5f" % (np.min(x0), np.max(x0)))
    print(" y    %9.5f %9.5f" % (np.min(y0), np.max(y0)))
    print(" z    %9.5f %9.5f" % (np.min(z0), np.max(z0)))
    print(" rho  %9.5f %9.5f" % (np.min(rho0), np.max(rho0)))
    print(" TYPES ", type(x0), type(y0), type(z0), type(rho0))
    print("================================================================================")
    print("================================================================================")
    
    mo = None

    # smaller pieces
    NX = NX//4
    NY = NY//4
    NZ = NZ//4
    
    for K in range(4):
        zmin, zmax = K*NZ, (K+1)*NZ
        for J in range(4):
            ymin, ymax = J*NY, (J+1)*NY
            for I in range(4):
                xmin, xmax = I*NX, (I+1)*NX


                print("\n")
                print("--------------------------------------------------------------------------------")
                print("--------------------------------------------------------------------------------")
                print("--------------------------------------------------------------------------------")
                print("\nPIECE ", I, J, K, xmin, xmax, ymin, ymax, zmin, zmax)
                
                t222 = time.time()
                mo   = nonzero((x0>=xmin)&(x0<xmax) & (y0>=ymin)&(y0<ymax) & (z0>=zmin)&(z0<zmax) )
                
                x    =  np.asarray(x0[mo]-xmin, float32)
                y    =  np.asarray(y0[mo]-ymin, float32)
                z    =  np.asarray(z0[mo]-zmin, float32)
                
                ORD  =  argsort(z)
                x    =  x[ORD]
                y    =  y[ORD]
                z    =  z[ORD]

                cells = len(x)
                print("YT2SOC_RAMSES  %d x %d x %d,  cells %d, %.3e" % (NX, NY, NZ, cells, cells))        
                print(" x  %9.5f %9.5f" % (np.min(x), np.max(x)))
                print(" y  %9.5f %9.5f" % (np.min(y), np.max(y)))
                print(" z  %9.5f %9.5f" % (np.min(z), np.max(z)))

                print("--- Make SOC hierarchy  =>   OT_points_to_octree...  -------------------------------------")
                OT_points_to_octree(x, y, z, NX, NY, NZ, LEVELS, SOCFILE, GPU, PLF=platforms)
                print("--- Make SOC hierarchy  =>   OT_points_to_octree ... done --------------------------------")
        
                del x, y, z
                NX, NY, NZ, LCELLS, OFF, H = OT_ReadHierarchyV(SOCFILE)
                # each leaf H>0.0 is equal to  *(float *)&(1+index_to_yt_vectors)
                m    =  nonzero(H>0.0)     #  int=1  =>  1.401298464324817e-45  !!!
                ind  =  np.zeros(len(m[0]), np.int32)
                print("***** update indices ...")
                for i in range(len(m[0])):           # ind = YT vector indices for leaves.. indices to z-ordered vectors
                    ind[i] = F2I( H[m[0][i]] ) - 1   # convert back to YT indices
                print("***** update indices ... done")
                
                # rho  =  np.asarray(ad["gas", "density"], np.float32)[mo][ORD]
                rho  =  rho0[mo][ORD]                
                H[m] =  rho[ind]
                del rho
                
                print("--- Write SOC file")
                t0 = time.time()
                OT_WriteHierarchyV(NX, NY, NZ, LCELLS, OFF, H, SOCFILE+'.%02d' % (16*K+4*J+I))
                
                print(" === YT2SOC_RAMSES_64  PIECE (%d,%d,%d)  completed in %.1f seconds ===" % (I, J, K, time.time()-t111))
                print()
                
                # LOCFILE = ''
                # if (len(LOCFILE)>0):
                #     print(" === YT2SOC_2024 create LOC file ===")
                #     t01    =  time.time()
                #     cells  =  len(H)
                #     VX     =  np.zeros(cells, np.float32)
                #     VX[m]  =  np.asarray(ad["gas", "velocity_z"], np.float32)[mo][ORD][ind] 
                #     VY     =  np.zeros(cells, np.float32)
                #     VY[m]  =  np.asarray(ad["gas", "velocity_y"], np.float32)[mo][ORD][ind] 
                #     VZ     =  np.zeros(cells, np.float32)
                #     VZ[m]  =  np.asarray(ad["gas", "velocity_x"], np.float32)[mo][ORD][ind] 
                #     CHI    =  np.ones(cells, np.float32)
                #     S      =  np.ones(cells, np.float32)
                #     T      =  np.ones(cells, np.float32)
                #     OT_WriteHierarchyV_LOC(NX, NY, NZ, LCELLS, OFF, H, T, S, VX, VY, VZ, CHI, LOCFILE)
                #     print(" === YT2SOC_RAMSES LOC created in %.1f seconds ===" % (time.time()-t01))
        
                print("*** Y2SOC_RAMSES_64 --- single piece in %.2f seconds" % (time.time()-t222))
            
                    
    print("***** YT2SOC_RAMSES_64 --- total time  %.0f seconds" % (time.time()-t111))
