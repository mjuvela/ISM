import pyopencl as cl
import ctypes
import numpy as np
import os, sys

INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)   


OLD_ORIENTATION = True  # important... must be True


def I2F(i):
    return ctypes.c_float.from_buffer(ctypes.c_int(i)).value
    
def F2I(x):
    return ctypes.c_int.from_buffer(ctypes.c_float(x)).value
        

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
    cells = np.fromfile(fp, np.int32, 1)[0]
    R     = np.fromfile(fp, np.float32, cells)
    tmp   = np.fromfile(fp, np.float32, cells*6).reshape(cells, 5) #  n, T, s, chi, vrad
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
    NRAY, NCHN      =  np.fromfile(fp, np.int32, 2)
    V0, DV          =  np.fromfile(fp, np.float32, 2)
    SPE             =  np.fromfile(fp, np.float32, NRAY*NCHN).reshape(NRAY,NCHN)
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
    NX, NY, NZ, dummy  =  np.fromfile(fp, np.int32, 4)
    TEX                =  np.fromfile(fp, np.float32).reshape(NZ, NY, NX)
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
    nx, ny, nz, levels, cells = np.fromfile(fp, np.int32, 5)
    # print('    OT_ReadHierarchy: ', nx, ny, nz, levels, cells)
    LCELLS = np.zeros(levels, np.int32)
    OFF    = np.zeros(levels, np.int32)
    H      = np.zeros(cells,  np.float32)
    for i in range(levels):
        tmp       = np.fromfile(fp, np.int32, 1)[0]
        LCELLS[i] = tmp
        H[OFF[i]:(OFF[i]+LCELLS[i])] =  np.fromfile(fp, np.float32, LCELLS[i])
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
    NX, NY, NZ, LEVELS, CELLS = np.fromfile(fp, np.int32, 5)
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
            cells       =  np.fromfile(fp, np.int32, 1)
            # print('cells = ', cells)
            cells = cells[0]
            if (i>0):  OFF[i] = OFF[i-1] + LCELLS[i-1]
            LCELLS[i]   =  cells
            X[OFF[i]:(OFF[i]+LCELLS[i])] =  np.fromfile(fp, np.float32, LCELLS[i])
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

    
    

def OT_GetCoordinatesAllV(NX, NY, NZ, LCELLS, OFF, H, GPU=0, platforms=[0,1,2,3,4]):
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
    # print("OT_GetCoordinatesAllV OFF", OFF, "LCELLS ", LCELLS) 
    # LEVELS     =  len(LCELLS)
    print(".... LCELLS = ", LCELLS)
    LEVELS     =  len(np.nonzero(LCELLS>0)[0])  # could be [ 999, 0, 0, 0, ...]
    print(".... LEVELS = ", LEVELS)
    CELLS      =  sum(LCELLS)
    N          =  CELLS
    # print('OT_GetCoordinatesAllV() -- LEVELS %d, CELLS %d, LCELLS' % (LEVELS, CELLS), LCELLS)
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    GLOBAL     =  8192
    LOCAL      =  [ 8, 32 ][GPU>0]
    OPT        =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, N, NX, NY, NZ)
    source     =  open(INSTALL_DIR+"/kernel_OT_tools.c").read()
    program    =  cl.Program(context, source).build(OPT)
    # Use kernel Parents to find the parents
    # print('Parents...')
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(OFF, np.int32))
    H_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
    PAR_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS) # lives on device
    PAR        =  np.zeros(CELLS, np.int32)
    cl.enqueue_copy(queue, PAR_buf, PAR)
    Parents    =  program.ParentsV
    Parents.set_scalar_arg_dtypes([None,None,None,None])
    Parents(queue, [GLOBAL,], [LOCAL,], LCELLS_buf, OFF_buf, H_buf, PAR_buf)
    cl.enqueue_copy(queue, PAR, PAR_buf)
    # Get coordinates for each cell
    # print('Coordinates...')
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
    # print('OT_GetCoordinatesAllV .... ready --- CELLS WAS %d !' % CELLS)
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
    LEVELS     =  len(np.nonzero(LCELLS>0)[0])  # could be [ 999, 0, 0, 0, ...]
    CELLS      =  sum(LCELLS)
    N          =  CELLS
    # print('OT_GetCoordinatesAllV() -- LEVELS %d, CELLS %d, LCELLS' % (LEVELS, CELLS), LCELLS)
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    GLOBAL     =  8192
    LOCAL      =  [ 8, 32 ][GPU>0]
    OPT        =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, N, NX, NY, NZ)
    source     =  open(INSTALL_DIR+"/kernel_OT_tools.c").read()
    program    =  cl.Program(context, source).build(OPT)
    # Use kernel Parents to find the parents
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(OFF, np.int32))
    H_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
    PAR_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS) # lives on device
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
    LEVELS     =  len(np.nonzero(LCELLS>0)[0])  # could be [ 999, 0, 0, 0, ...]
    CELLS      =  sum(LCELLS)
    N          =  CELLS
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms)
    GLOBAL     =  8192
    LOCAL      =  [ 8, 32 ][GPU>0]
    OPT        =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, N, NX, NY, NZ)
    source     =  open(INSTALL_DIR+"/kernel_OT_tools.c").read()
    program    =  cl.Program(context, source).build(OPT)
    # Use kernel Parents to find the parents
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=LCELLS)
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=OFF)
    H_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H)
    PAR_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS) # lives on device
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
        ll  =  np.fromfile('/dev/shm/tmp.bin', np.int32)  # positive indices
        LL.append(ll)
        m   =  np.nonzero(H[i]<1e-10)
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
    if ((type(x[0])!=np.float32)|(type(y[0])!=np.float32)|(type(z[0])!=np.float32)|(type(H[0])!=np.float32)):
        print("OT_GetIndicesV() -- coordinates must be 1d numpy.float32 vectors"), sys.exit()
    if ((type(LCELLS[0])!=np.int32)|(type(OFF[0])!=np.int32)):
        print("OT_GetIndicesV() -- LCELLS and OFF must be numpy.int32 arrays"), sys.exit()
    LEVELS   =  len(LCELLS)
    CELLS    =  sum(LCELLS)
    N        =  np.size(x)
    ###
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    LOCAL    =  [ 8, 32 ][GPU>0]
    GLOBAL   =  8192
    OPT      =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, N, NX, NY, NZ)
    source     =  open(INSTALL_DIR+"/kernel_OT_tools.c").read()
    program  =  cl.Program(context, source).build(OPT)
    #
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(OFF, np.int32))
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
    if ((type(x[0])!=np.float32)|(type(y[0])!=np.float32)|(type(z[0])!=np.float32)|(type(H[0])!=np.float32)):
        print("OT_GetIndicesV() -- coordinates must be 1d numpy.float32 vectors"), sys.exit()
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
    source     =  open(INSTALL_DIR+"/kernel_OT_tools.c").read()
    program  =  cl.Program(context, source).build(OPT)
    #
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(OFF, np.int32))
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
    BATCH      =  min([10000000, len(x)])
    ###
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    LOCAL    =  [ 8, 32 ][GPU>0]
    GLOBAL   =  8192
    OPT      =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d" % (LEVELS, BATCH, NX, NY, NZ)
    source     =  open(INSTALL_DIR+"/kernel_OT_tools.c").read()
    program  =  cl.Program(context, source).build(OPT)
    #
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(OFF, np.int32))
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
            b = min([a + BATCH, N])
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
            b = min([a + BATCH, N])
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
    H          =  ones(NX*NY*NZ, np.float32)  # root level
    for P in range(MAXL):                                 # loop over parent layers = addition of child layers
        LCELLS.append(LCELLS[P]*8)                        # children on level P+1
        OFF.append(OFF[P]+LCELLS[P])                      # offset for cells on layer P+1
        H  = concatenate((H, ones(LCELLS[P+1], np.float32))) # add child cells, filled with ones
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
    H     =  ones(cells, np.float32)
    ##
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=True)
    LOCAL    =  [ 8, 32 ][GPU>0]
    GLOBAL   =  32768
    OPT      =  ""
    source   =  open(INSTALL_DIR+'kernel_make_empty_hierarchy.c').read()
    program  =  cl.Program(context, source).build(OPT)
    Index    =  program.Index
    Index.set_scalar_arg_dtypes([np.int32, None])
    C_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*LCELLS[MAXL-1])
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
    LEVELS     =  len(np.nonzero(LCELLS>0)[0])  # could be [ 999, 0, 0, 0, ...]
    CELLS      =  sum(LCELLS)
    platform, device, context, queue, mf  =  InitCL(GPU=GPU, platforms=platforms, verbose=False)
    GLOBAL     =  8192
    LOCAL      =  [ 8, 32 ][GPU>0]
    OPT        =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d " % (LEVELS, CELLS, NX, NY, NZ)
    source     =  open(INSTALL_DIR+'/kernel_OT_tools.c').read()
    program    =  cl.Program(context, source).build(OPT)
    # Use kernel Parents to find the parents
    max_cells  =  max(LCELLS)
    print('max_cells = %.3e' % max_cells)
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(OFF,    np.int32))
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
    m = np.nonzero(SIGMA<=0.0)
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
    LEVELS     =  len(np.nonzero(LCELLS>0)[0])  # could be [ 999, 0, 0, 0, ...]
    CELLS      =  sum(LCELLS)
    platform, device, context, queue, mf  =  InitCL(GPU=GPU, platforms=platforms, verbose=False)
    GLOBAL     =  8192
    LOCAL      =  [ 1, 32 ][GPU>0]
    OPT        =  " -D LEVELS=%d -D N=%d -D NX=%d -D NY=%d -D NZ=%d " % (LEVELS, CELLS, NX, NY, NZ)
    source     =  open(INSTALL_DIR+'/kernel_OT_tools.c').read()
    program    =  cl.Program(context, source).build(OPT)
    max_cells  =  max(LCELLS)
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(LCELLS, np.int32))
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.asarray(OFF,    np.int32))
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
    nx, ny, nz, levels, cells = np.fromfile(fp, np.int32, 5)
    # print('    OT_ReadHierarchy: ', nx, ny, nz, levels, cells)
    lcells = np.zeros(levels, np.int32)
    H = []
    for i in range(levels):
        tmp       = np.fromfile(fp, np.int32, 1)
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
        ll  =  np.fromfile('/dev/shm/tmp.bin', np.int32)  # positive indices
        LL.append(ll)
        m   =  np.nonzero(H[i]<1e-10)
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
    source    =  open(INSTALL_DIR+'/kernel_OT_tools.c').read()
    program   =  cl.Program(context, source).build(OPT)
    #
    DENS_buf  =  []
    LCELLS    =  np.zeros(LEVELS, np.int32)
    for i in range(7):  # maximum of 7 levels
        if (i<LEVELS):
            LCELLS[i] = len(H[i])
            DENS_buf.append( cl.Buffer(context, mf.READ_ONLY, 4*max([1,LCELLS[i]]))  )
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
    NX, NY, NZ, LEVELS, CELLS = np.fromfile(fpin, np.int32, 5)
    maxlevel =  min([LEVELS-1, maxlevel])
    LCELLS   =  np.zeros(LEVELS, np.int32)
    print('OT_cut_levels: reading hierarchy with %d levels -> maxlevel %d' % (LEVELS, maxlevel))
    # Output file
    fpout    =  open(outfile, 'wb')
    # Create OpenCL program to average child values into parent node
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms, verbose=False)
    LOCAL    =  [ 8, 32 ][GPU>0]
    source    =  open(INSTALL_DIR+'/kernel_OT_tools.c').read()
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
            cells      =  np.fromfile(fpin, np.int32, 1)[0]
            LCELLS[i]  =  cells
            H.append(fromfile(fpin, np.float32, cells))
        if (ifield==0):  # allocate buffers only once LCELLS is known
            GLOBAL   =  max(LCELLS)   # at most this many parent cells, at most this many vector elements
            P_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL)   # parent level cells
            C_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*GLOBAL)   # child level cells 
            H_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*GLOBAL)   # hierarchy for the parent level            
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
                H[i][np.nonzero(H[i]>0.0)] *= 1.4 # factor of 2 change in scale ~ factor of 1.5 in turbulent velocity
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
    NX, NY, NZ, LEVELS, CELLS = np.fromfile(fin, np.int32, 5)
    print('INFILE:', NX, NY, NZ, LEVELS, CELLS)
    if (len(hierarchy)>0):  
        faux = open(hierarchy, 'rb')
        NX, NY, NZ, LEVELS, CELLS = np.fromfile(faux, np.int32, 5)
    NX0, NY0, NZ0 = NX, NY, NZ
    TRUE_LEVELS = 0
    limits      = np.asarray(limits)
    rho, RHO, link, LINKS = None, None, None, None  # LINKS = -RHO read as np.int32 == links
    for i in range(LEVELS):
        # READ THE NEXT LEVEL
        cells = np.fromfile(fin, np.int32, 1)[0]
        rho   = np.fromfile(fin, np.float32, cells)
        ##########################################################
        links = None
        if (faux==None):              # convert rho into links
            np.asarray(-rho,  np.float32).tofile('/dev/shm/tmp143563.bin')
            links =  np.fromfile('/dev/shm/tmp143563.bin', np.int32)
        else:                         # read links from separate file
            np.fromfile(faux, np.int32, 1)  # cells
            tmp = np.fromfile(faux, np.float32, cells)
            np.asarray(-tmp,  np.float32).tofile('/dev/shm/tmp143563.bin')
            links =  np.fromfile('/dev/shm/tmp143563.bin', np.int32) # links >= 0
        ##########################################################
        if (i==0):
            # we cut the root grid to a set of root cells
            K, J, I    =  indices((NZ, NY, NX), np.int32)
            K, J, I    =  ravel(K), ravel(J), ravel(I)
            m          =  np.nonzero((I>=limits[0,0])&(I<=limits[0,1])&(J>=limits[1,0])&(J<=limits[1,1])&(K>=limits[2,0])&(K<=limits[2,1]))
            rho        =  rho[m]           # links and leaf cells, the subvolume only
            links      =  links[m]
            I, J, K    =  I[m], J[m], K[m]
            NX         =  max(I)-min(I)+1
            NY         =  max(J)-min(J)+1
            NZ         =  max(K)-min(K)+1            
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
        L   =  np.asarray(links[np.nonzero(links>=0)], np.int32)  # all the links on child level
        LP  =  np.asarray(LINKS[np.nonzero(LINKS>=0)], np.int32)  # all the links on parent level
        print('LEVEL %d:' % i)
        maxi = -1
        if (len(LP)>0): maxi = max(LP)
        print('         parent %5d CELLS, %5d CHILDREN, MAX LINK %5d' %   (len(RHO), 8*len(LP), maxi))
        if (len(L)>0):
            print('         child  %5d CELLS, %5d CHILDREN, MAX LINK %5d' % (len(rho), 8*len(L), max(L)))
        else:
            print('         child  %5d CELLS' % (len(rho)))
        # write the data for THE PARENT LEVEL
        CELLS  += len(RHO)   # total number of cells
        np.asarray([len(RHO),], np.int32).tofile(fout)
        # replace all LINK values with the np.int32 read as float, times -1
        np.asarray(LINKS, np.int32).tofile('/dev/shm/tmp143563.bin')
        tmp    =  np.fromfile('/dev/shm/tmp143563.bin', np.float32)
        #### m      =  np.nonzero(RHO<=0.0)
        m      =  np.nonzero(LINKS>=0.0)  # mask for link cells
        RHO[m] = -tmp[m] 
        np.asarray(RHO, np.float32).tofile(fout)
        RHO   = rho.copy()       # the vector of parents for the next iteration
        LINKS = links.copy()
        TRUE_LEVELS += 1         # could be less than LEVELS, if the cut region is not refined
        if (len(rho)<1): break   # the cut region does not contain any deeper levels
        if (len(LP)>0):
            if (max(LP)>=len(rho)):
                print('*'*80)
                print('  ERROR:  PARENT HAS A LINK TO CELL %d, THIS LEVEL HAS ONLY %d CELLS' % (max(LP), len(rho)))
                print('*'*80)
                sys.exit()
    # ==== end of for levels================================================================================
    
    # write the final level = LEVELS-1 (no links there)
    CELLS  += len(rho)   # total number of cells
    np.asarray([len(rho),], np.int32).tofile(fout)
    ## convert links to proper format int -> float
    np.asarray(links, np.int32).tofile('/dev/shm/tmp143563.bin')
    tmp    =  np.fromfile('/dev/shm/tmp143563.bin', np.float32)
    m      =  np.nonzero(links>=0.0)
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
                if   (NX == max([NX, NY, NZ])):
                    fp.write('direction x\n')
                elif (NY == max([NX, NY, NZ])):
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
    source   =  open(INSTALL_DIR+'/kernel_OT_cut.c').read()
    OPT      = '-D NX=%s -D NY=%d -D NZ=%d -D LEVELS=%d' % (NX, NY, NZ, LEVELS)
    program  =  cl.Program(context, source).build(OPT)
    CV       =  program.CutVolume
    #                           LIM,   OFF,  LCELLS,  H,     A,     B,     lcells
    CV.set_scalar_arg_dtypes([  None,  None, None,    None,  None,  None,  None  ])
    limits     =  np.asarray(limits, np.int32)
    LIM_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=limits)
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=OFF)
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=LCELLS)
    H_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)
    A_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)
    B_buf      =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)
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
            levels  = len(np.nonzero(lcells>0)[0])  # cut region may not contain all hierarchy levels
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
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms)
    LOCAL    =   8
    GLOBAL   =  64
    source   =  open(INSTALL_DIR+'/kernel_OT_cut.c').read()
    OPT      = '-D NX=%s -D NY=%d -D NZ=%d -D LEVELS=%d' % (NX, NY, NZ, LEVELS)
    program  =  cl.Program(context, source).build(OPT)
    CV       =  program.CutVolume
    #                           LIM,   OFF,  LCELLS,  H0,    H1,    X0,   X1    lcells
    CV.set_scalar_arg_dtypes([  None,  None, None,    None,  None,  None, None, None  ])
    limits     =  np.asarray(limits, np.int32)
    LIM_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=limits)
    OFF_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=OFF)
    LCELLS_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,  hostbuf=LCELLS)
    H0_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)
    H1_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)
    X0_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)
    X1_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*CELLS)
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
            levels  = len(np.nonzero(lcells>0)[0])  # cut region may not contain all hierarchy levels
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
    source   = open(INSTALL_DIR+'/kernel_OT_tools.c').read()
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
    GLOBAL   =  max(LCELLS)   # at most this many parent cells, at most this many vector elements
    GLOBAL   =  (GLOBAL//32+1)*32
    P_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL)
    C_buf    =  cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL)
    CO_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*GLOBAL)
    I_buf    =  cl.Buffer(context, mf.READ_ONLY,  4*GLOBAL)
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
            m = np.nonzero(P<=0.0)   # all parents with children
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
            m       =  np.nonzero(C<1.0e20)        # these child cells will remain
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
        SIGMA.append(np.zeros(len(h), np.float32))    
    for level in range(LEVELS-2, -1, -1):          # loop over parent levels
        print("LEVEL %d" % level)
        m = np.nonzero(H[level]<=0)                   # parent cells of "level"
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
            mc              =  np.nonzero(SIGMA[level+1][c:(c+8)]<=0.0)
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
    """    
    NX, NY, NZ, LEVELS, LCELLS, H  = OT_ReadHierarchy(rhofile)    
    NX, NY, NZ, LEVELS, LCELLS, VX = OT_ReadHierarchy(vxfile)  # originally exists only for leaf nodes
    NX, NY, NZ, LEVELS, LCELLS, VY = OT_ReadHierarchy(vyfile)    
    NX, NY, NZ, LEVELS, LCELLS, VZ = OT_ReadHierarchy(vzfile)        
    CELLS = sum(LCELLS)
    SIGMA = []
    for h in H:    # create SIGMA vectors
        SIGMA.append(np.zeros(len(h), np.float32))    
    ###
    platform, device, context, queue, mf = InitCL(GPU, platforms=PLF)
    LOCAL       =  [8,32][GPU>0]
    source      =  open(KERNEL+'/kernel_turbulence.c').read()
    OPT         =  ''
    program     =  cl.Program(context, source).build(OPT)
    SU          =  program.SigmaUpdate
    SU.set_scalar_arg_dtypes([np.int32, None, None, None, None, None, None, None, None])
    NMAX        =  max(LCELLS)
    H_buf       =  cl.Buffer(context, mf.READ_ONLY,  4*NMAX)
    SIGMAP_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*NMAX)
    SIGMAC_buf  =  cl.Buffer(context, mf.READ_WRITE, 4*NMAX)
    VXP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*NMAX)
    VYP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*NMAX)
    VZP_buf     =  cl.Buffer(context, mf.READ_WRITE, 4*NMAX)
    VXC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*NMAX)
    VYC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*NMAX)
    VZC_buf     =  cl.Buffer(context, mf.READ_ONLY,  4*NMAX)
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
    NX, NY, NZ, LEVELS, CELLS = np.fromfile(fp, np.int32, 5)    
    fpout  =  open(newname, 'wb')
    np.asarray([NX, NY, NZ, LEVELS, CELLS], np.int32).tofile(fpout)    
    for ipar in range(7):     # one parameter for SOC, seven for LOC = [ H, T, S, VX, VY, VZ, C ]
        for ilevel in range(LEVELS):
            cells = np.fromfile(fp, np.int32, 1)
            if (len(cells)<1): break   # no more parameters in the file
            data  = np.fromfile(fp, np.float32, cells[0])
            if (ilevel==0):            # rolling affects only the root level
                data.shape = (NZ, NY, NX)                
                data = np.roll(data.copy(), dz, 0)
                data = np.roll(data.copy(), dy, 1)
                data = np.roll(data.copy(), dx, 2)                
            cells.tofile(fpout)
            data.tofile(fpout)
    fp.close()
    fpout.close()
                
    
