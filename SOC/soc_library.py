#!/bin/python
from MJ.mjDefs import *
import mmap
from MJ.Aux.mjGPU import *
INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))


"""
NxNxN bins in the library structure. Inputs are absorption vectors and possibly 
higher SNR vectors of intensity at the reference wavelengths.

Setup of the library:
    - extract limits for first reference wavelength on log scale =>  IMIN0, DMIN0
    - for each 0-bin, extract limits on the second reference wavelenth IMIN1[N], DMIN1[N]
    - for each (I0,I1), extract limits on the third referene wavelength IMIN2[N,N], DMIN2[N,N]
    
    - loop over all cells (absorption vectors):
        - if (i0, i1, i2) corresponds to a so-far empty bin, add this cell to that bin
        
    - for each non-empty bin, copy the full absorption vector to a new file,
      run A2E to get the corresponding emission vectors
    - dump I0, I1, I2 grid parameters and emission vectors to a new file
        
Solve (within SOC), using only the reference intensities (I0, I1, I2) of a cell:
    - calculate index i0 based on I0 and  loop  over i0-1, i0, i0+1:
        - for each i0, calculate I1 bin i1 and loop over i1-1, i1, i1+1:
            - for each i1, calculate I2 bin i2 and loop over i2-1, i2, i2+1:
                - if grid is empty, continue
                - set weight ~  (<I0>^2 + <I1>^2 + <I2>^2)^-p/2, where <I> is distance in 
                  log-intensity between bin and the actual intensity value in the cell, p~2
                - update sum += weight * E, where E is the emission vector of the grid point
        - final estimate for emission = E /sum(weight), write to emitted
        
"""

if (not(len(sys.argv)in[4,5,6])):
    print()
    print("Usage:  soc_library.py  gpu_tag  [ solver  absorbed  |  solver absorbed emitted  [ofreq.dat]  ]")
    print("  soc_library.py   gpu_tag solver_file  absorption_file")
    print("      create te library based on solver file and ASOC-omputed absorptions.")
    print("  soc_library.py   gpu_tag solver_file  absorption_file  emission_file  [ofreq.dat]")
    print("      use library to selve emission")
    print("      the optional file name (here ofreq.dat) lists output frequencies that")
    print("      can be a subset of all the frequencies in the library")
    print("One can use equilibrium dust also but then <solver> must be replaced with the dust file.")
    print("Parameter gpu_tag is 0 for CPU and 1 for GPU. Can be a decimal number including 0.1*platform, ")
    print("which chooses a specific OpenCL platform (0, 1, ...) ")
    print()
    sys.exit()

GPU_TAG        =  float(sys.argv[1])    
FILE_SOLVER    =  sys.argv[2]    # A2E_pre file - or for equilibrium dust directly the dust file
FILE_ABSORBED  =  sys.argv[3]
MAKE_LIBRARY   =  True
EQDUST         =  False          # whether one is calculating equilibrium dust

# check if the argument is equilibrium dust or solver file for stochastically heated grains
EQDUST = False
try:
    tmp = open(FILE_SOLVER, 'r').readline()
    if (tmp.find('eqdust')==0):
        EQDUST = True
except:
    pass


OFREQ          =  []             # OFREQ!=[], emission written only for those frequencies
NOFREQ         =  0
if (len(sys.argv)>4):            # emission file given .. we should be only using the library to solve emission
    FILE_EMITTED = sys.argv[4]
    MAKE_LIBRARY = False         # library should already exist...
    if (len(sys.argv)>5):
        print(len(sys.argv))
        OFREQ  = loadtxt(sys.argv[5])
        if (size(OFREQ)==1): OFREQ = asarray(OFREQ, float32).reshape(1,)
        NOFREQ = len(OFREQ)
if (MAKE_LIBRARY==False):        # ... but check this and make the library if it is still missing
    if (os.path.exists('%s.lib' % FILE_SOLVER)==False):
        print("soc_library.py using library %s.lib that does not yet exist" % FILE_SOLVER)
        print("==> we will now first create it")
        os.system('soc_library.py %s %s' % (FILE_SOLVER, FILE_ABSORBED))
        print("Library has been created - continue to solve all cells....")
    # 2021-01-22  --- if %s.lib exists and MAKE_LIBRARY==False, absorption file may
    #                 contain all frequencies or just the reference frequencies
    
    
FREQ     = loadtxt('freq.dat')
NFREQ    = len(FREQ)
# RUM      = asarray([0.2, 0.5, 1.2], float32)
# FREF     = um2f(RUM)
FREF     = loadtxt('lfreq.dat')
N        = 30
NLFREQ   = 3            # number of reference frequencies

# extract intensities at the reference wavelengths
fp            = open(FILE_ABSORBED, 'rb')
CELLS, NFFREQ = fromfile(fp, int32, 2)
fp.close()
if (NFFREQ!=NFREQ):
    # NFFREQ < NFREQ --- only if MAKE_LIBRARY==False
    # == we solve with library with input of only the reference frequency absorptions
    if (MAKE_LIBRARY==True):
        print("Inconsistent number of frequencies: FREQ has %d, file %d frequencies" % (NFREQ, NFFREQ))
        sys.exit()
    else:  # else = use library
        if (not((NFREQ==NFFREQ)|(NFFREQ==3))):
            print("Using library... but absorption file has %d rather than 3 or %d frequencies" % (NFFREQ, NFREQ))
            sys.exit()
        
print('FILE_ABSORBED %s, CELLS %d, NFREQ %d, NFFREQ %d' % (FILE_ABSORBED, CELLS, NFREQ, NFFREQ))
ABSORBED = np.memmap(FILE_ABSORBED, dtype='float32', mode='r',  shape=(CELLS, NFFREQ), offset=8)
IFREQ    = asarray(arange(3), int32)   # perhaps USELIB and NFFREQ==NLFREQ
if (NFFREQ>NLFREQ):                    # file contains more than just the reference frequencies
    if (NFFREQ!=NFREQ):
        print("*** ERROR in soc_library.py, ABSORBED=%s has %d frequencies, NFREQ = %d" % (FILE_ABSORBED, NFFREQ, NFREQ))
        sys.exit()
    for i in range(3):
        IFREQ[i] = argmin(abs(FREF[i]-FREQ))  # pick reference frequency indices from the full NFREQ
else:
    if (MAKE_LIBRARY):
        print("*** ERROR in soc_library.py, MAKELIB and file %s has only %d frequencies < NFREQ = %d" % (FILE_ABSORBED, NFFREQ, NFREQ))
        sys.exit()
print("*** IFREQ = ", IFREQ)    
        
        
IREF     = zeros((CELLS, NLFREQ), float32)
for i in range(NLFREQ):
    IREF[:, i]  =  ABSORBED[:,IFREQ[i]]
IREF     = clip(IREF, 1.0e-25, 1.0)     # IREF[CELLS, 3]
IREF     = log10(IREF)                  # all interpolations on log scale


if (MAKE_LIBRARY):
    # MAKE_LIBRARY==True => ABSORBED is the full set of frequencies == NFREQ frequencies
    # bins for the first reference frequency
    a, b   =  min(IREF[:,0]), max(IREF[:,0])
    if (1): 
        d = (b-a)/N+0.1
        a, b = a-d, b+d
    dI0    =  clip(1.001*(b-a)/N, 1e-30, 1e30)
    I0     =  a+0.499*dI0       # centre of the first bin
    # centre of the i:th bin  IMIN0+i*dI0, last bin N-1  IMIN0+(N-1)*(b-a)/N = IMIN0+RANGE-dI = ok 
    I1, dI1  =  zeros( N,     float32),  zeros( N,     float32)
    I2, dI2  =  zeros((N, N), float32),  zeros((N, N), float32)
    for i in range(N):          # loop over bins of the first reference frequency
        B0     =  I0 + i*dI0    # B0 = log10(I0) for this bin
        m      =  nonzero(abs(IREF[:,0]-B0)<(0.5*dI0))  # select cells matching B0
        if (len(m[0])<1):
            I1[i], dI1[i] = 100.0, 0.001
        else:
            a, b   =  min(IREF[m[0], 1]), max(IREF[m[0], 1])  # current I0, limits for I1
            if (1): 
                d = (b-a)/N+0.1
                a, b = a-d, b+d
                dI1[i] =  clip(1.001*(b-a)/N, 1.0e-30, 1.0e30)
            I1[i]  =  a+0.499*dI1[i]                          # centre of the first bin
        # the bins for the last reference frequency
        for j in range(N):
            B1       =  I1[i] + j*dI1[i]        # B1 = intensity I1 for the bin j
            m        =  nonzero( (abs(IREF[:,0]-B0)<(0.5*dI0)) & (abs(IREF[:,1]-B1)<(0.5*dI1[i])) )
            if (len(m[0])<2):
                I2[i,j], dI2[i,j] = 100.0, 0.001
                continue
            a, b     =  min(IREF[m[0], 2]), max(IREF[m[0], 2])  # for (i,j), limits for I2
            if (1): 
                d = (b-a)/N+0.1
                a, b = a-d, b+d
            dI2[i,j] =  clip(1.001*(b-a)/N, 1.0e-30, 1.0e30)
            I2[i,j]  =  a+0.499*dI2[i,j]

            
    # We may have many (I0,I1) for which the I2 binning remains undefined 
    # this would prevent one from filling those in on future calls
    # Copy to those (I0,I1) and I2 intensity grid from the previous (I0,I1) bin
    #    with a valid I2 grid ... this will not be perfect but should enable
    #    one to use the library in the future for some I2 values also in that (I0,I1) bin
    if (1):
        a, b = 100.0, 0.001      # ad hoc I2 and dI2
        for ii in range(N):
            for jj in range(N):
                if (I2[ii,jj]<99.0): # some valid binning for the current (ii,jj)
                    a, b = I2[ii,jj], dI2[ii,jj]
                else:                # I2 grid was undefined
                    I2[ii,jj], dI2[ii,jj] = a, b    
        print("0:  %10.3e           %10.3e" % (I0, dI0))
        print("1:  %10.3e %10.3e    %10.3e %10.3e" % (min(I1), max(I1), min(dI1), max(dI1)))
        print("2:  %10.3e %10.3e    %10.3e %10.3e" % (min(ravel(I2)), max(ravel(I2)), min(ravel(dI2)), max(ravel(dI2))))
        
                
    # Try to find a cell for each bin in the (I0, I1, I2) space
    IND = zeros((N,N,N), int32)                  # library bin -> cell index
    DIS = 1e9*ones((N,N,N), float32)             # distance of selected cell wrt bin centre
    XX, YY, ZZ = zeros((N,N,N), float32), zeros((N,N,N), float32), zeros((N,N,N), float32)
    X   = (IREF[:,0]-I0     ) / dI0              #  x ~ coordinate along the first axis -- ALL CELLS
    I   = asarray(np.round(X), int32)            # index along the first axis
    Y   = (IREF[:,1]-I1[I]  ) / dI1[I]
    J   = asarray(np.round(Y), int32)
    Z   = (IREF[:,2]-I2[I,J]) / dI2[I,J]
    K   = asarray(np.round(Z), int32)            # (I,J,K) cells -> index in library grid
    if (1):  # it should be ok to clip... should not do anything because the limits are based on the current data
        I   = clip(I, 0, N-1)
        J   = clip(J, 0, N-1)
        K   = clip(K, 0, N-1)
    dis = abs(X-I)+abs(Y-J)+abs(Z-K)             # ~distance to the closest bin centre
    for icell in range(CELLS):
        i, j, k = I[icell], J[icell], K[icell]   # selected grid indices
        if (dis[icell]<DIS[i,j,k]):              # the closest cell selected to represent the grid point
            DIS[i,j,k] = dis[icell]
            IND[i,j,k] = icell
            XX[i,j,k]  = X[icell]                # XX[bins], X[cells]
            YY[i,j,k]  = Y[icell]
            ZZ[i,j,k]  = Z[icell]
    # Now IND[i,j,k] contains the index of the closest cell (distance DIS[i,j,k])
    IND  = ravel(IND)
    DIS  = ravel(DIS)
    IND[nonzero(DIS>1.5)] = -1    # some bins do not have any matching cells...
    
    # Write the corresponding absorption vectors to a file and calculate emissions
    m     = nonzero(IND>=0)       # bins with valid cell index IND
    ind   = IND[m]                # selected cells, cell indices TO THE FULL CLOUD
    cells = len(ind)              # solve this many cells,  cells <= N*N*N
    print("Out of (N=%d)^3 = %d bins, cells existed for %d bins" % (N, N*N*N, cells))
    
    fp = open('tmp.absorbed', 'wb')
    asarray([cells, NFREQ], int32).tofile(fp)
    for icell in ind:             # i == index to the full cloud
        ABSORBED[icell,:].tofile(fp)
    fp.close()
    
    t000 = time.time()
    # EQ_solver and A2E  will also solver emission for all NFREQ (freq.dat) frequencies
    # and the library will always contain emission for all those frequencies
    print("*** Solve for cells %d, NFREQ %d" % (cells, NFREQ))
    if (EQDUST):     # tmp.emitted [cells, NFREQ]
        os.system('EQ_solver.py  %s tmp.absorbed tmp.emitted 1' % FILE_SOLVER)  
    else:
        os.system('A2E.py        %s tmp.absorbed tmp.emitted 1' % FILE_SOLVER)
    t000 = 1.0e6*(time.time()-t000)/(float(cells))   
    print("*** DIRECT SOLVE %.2f USEC PER CELL ***" % t000)
    
    # Write information on the binning and the emission vectors to a new file
    #  - number of bins
    #  - bin limits
    #  - bin indices do the cases for which we have the emission
    fp = open('%s.lib' % FILE_SOLVER, 'wb')
    asarray([N, NFREQ], int32).tofile(fp)
    if (len(FREQ)!=NFREQ): 
        print("??????")
        sys.exit()
    asarray(FREQ, float32).tofile(fp)
    asarray([I0, dI0], float32).tofile(fp)
    asarray(I1,  float32).tofile(fp)
    asarray(dI1, float32).tofile(fp)
    asarray(I2,  float32).tofile(fp)
    asarray(dI2, float32).tofile(fp)
    # The actual reference intensity values for the bins in the (I0,I1,I2) grid --- AS INDICES TO BINS
    asarray(XX, float32).tofile(fp)
    asarray(YY, float32).tofile(fp)
    asarray(ZZ, float32).tofile(fp)
    # The corresponding emission vectors.... or zeros if there were no cells in the bin
    tmp         =  clip(fromfile('tmp.emitted', float32)[2:].reshape(cells, NFREQ), 1.0e-30, 1.0e30)
    TMP         =  1.0e32*ones((N*N*N, NFREQ), float32)   # space for emission for every bin, 1e32 for missing data
    print("*** TMP[%d,%d] <- tmp[%d,%d]" % (N*N*N, NFREQ, cells, NFREQ))
    TMP[m[0],:] =  tmp                                     # only bins for which corresponding cells were available
    TMP.tofile(fp)
    fp.close()
    # Note --- library lists log10 of intensity and of emission
    #          missing emission vectors are marked with -1.0

    
else:  # Using the library to solve emission
    

    # MAKE_LIBRARY==False => using existing library to solve emission
    # - ABSORBED may be the full NFREQ frequencies or only three frequencies
    
    # Read the library information
    fp       = open('%s.lib' % FILE_SOLVER, 'rb')
    N, NFREQ = fromfile(fp, int32, 2)
    FREQ     = fromfile(fp, float32, NFREQ)
    I0, dI0  = fromfile(fp, float32, 2)
    I1       = fromfile(fp, float32, N)
    dI1      = fromfile(fp, float32, N)
    I2       = fromfile(fp, float32, N*N).reshape(N,N)
    dI2      = fromfile(fp, float32, N*N).reshape(N,N)
    X        = fromfile(fp, float32, N*N*N).reshape(N,N,N)
    Y        = fromfile(fp, float32, N*N*N).reshape(N,N,N)
    Z        = fromfile(fp, float32, N*N*N).reshape(N,N,N)
    EMI      = fromfile(fp, float32, N*N*N*NFREQ).reshape(N,N,N,NFREQ)
    # Upper limit ~   N<100, NFREQ<500  =  1.9 GB !!!
    # Typical     ~   N~50,  NFREQ~200  =  100 MB

    tmp      = ravel(EMI[:,:,:,0])
    m        = nonzero(tmp<1e30)
    print("Library grid: %d out of %d bins with emission vector" % (len(m[0]), N*N*N))

    # If NOFREQ>0, save only the frequencies listed in OFREQ (must be subset of library frequencies)
    # check the indices of the output frequencies
    OIFREQ = []
    nfreq  = NFREQ   # actual number of frequencies in the EMITTED file
    if (NOFREQ>0):
        OIFREQ = zeros(len(OFREQ), int32)
        for ifreq in range(NOFREQ):
            OIFREQ[ifreq] = argmin(abs(OFREQ[ifreq]-FREQ)) # closest frequency...
        nfreq = NOFREQ                                     # actual number of frequencies in the emitted file
    
    print("EMITTED == %s" % FILE_EMITTED)
    fp = open(FILE_EMITTED, 'wb')
    asarray([CELLS, nfreq], int32).tofile(fp)
    fp.close()
    EMITTED = np.memmap(FILE_EMITTED, dtype='float32', mode='r+',  shape=(CELLS, nfreq), offset=8)
    
    # IREF => index to library
    x       =  (IREF[:,0]-I0     )/dI0                # cells =>  bin indices,   x[CELLS]
    i       =  clip(asarray(np.round(x), int32), 0, N-1)            # (i,j,k) indices to library cube
    y       =  (IREF[:,1]-I1[i]  )/dI1[i]             #  y[CELLS]
    j       =  clip(asarray(np.round(y), int32), 0, N-1)
    z       =  (IREF[:,2]-I2[i,j])/dI2[i,j]           #  z[CELLS]
    k       =  clip(asarray(np.round(z), int32), 0, N-1)
        
    MIS     =  zeros(CELLS//2, int32)                 # store cells without library entry... 50% missing = something is wrong
    mis     =  0

    # double check that X[i,j,k] close to x etc.
    bad     =  abs(X[i,j,k]-x) + abs(Y[i,j,k]-y) + abs(Z[i,j,k]-z)
    m       =  nonzero(bad>3.0)
    print('bad %d' % len(m[0]))
    print('bad', bad[m])
    if (0):
        print("x", x[m])
        print("y", y[m])
        print("z", y[m])
    
    print(min(i), max(i))
    print(min(j), max(j))
    print(min(k), max(k))
    
    # (IREF0, IREF1, IREF2)  ->  (x, y, z)   ->  (i, j, k)  == bin indices
    # ==>     (x, y, z)  ~   (X[i,j,k], Y[i,j,k], Z[i,j,k])
    #  x, y, z, X, Y, Z  are all float indices  (I-I0)/dI0 etc.
        
    t000 = time.time()

    METHOD   =  0
    GPU      =  (GPU_TAG>=1.0)
    PLF      =  [ 0, 1, 2, 3 ]
    if (GPU_TAG%1.0>0.0): PLF = [ int(10.0*(GPU_TAG%1.0)), ] # decimal part -> selected platform
    print("GPU %d,  " % GPU, PLF)
    platform, device, context, queue, mf = InitCL(GPU, PLF)
    LOCAL    =  [4, 32][GPU>0]
    BATCH    =  8192
    GLOBAL   =  BATCH
    # note -- -D NFREQ is always NFREQ, the full set of frequencies
    #         kernel gets ABS[] that only has the three reference frequencies
    OPT      = '-D CELLS=%d -D N=%d -D NFREQ=%d -D LOCAL=%d -D METHOD=%d' % \
                  (CELLS,      N,      NFREQ,      LOCAL,      METHOD)
    print(OPT)
    source   =  open(INSTALL_DIR+"/kernel_soc_library.c").read()
    program  =  cl.Program(context, source).build(OPT)
    I1_buf   =  cl.Buffer(context, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=I1)    #  [N]
    dI1_buf  =  cl.Buffer(context, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=dI1)   #  [N]
    I2_buf   =  cl.Buffer(context, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=I2)    #  [N,N]
    dI2_buf  =  cl.Buffer(context, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=dI2)   #  [N,N]
    X_buf    =  cl.Buffer(context, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=X)     #  [N,N,N]
    Y_buf    =  cl.Buffer(context, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=Y)     #  [N,N,N]
    Z_buf    =  cl.Buffer(context, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=Z)     #  [N,N,N]
    E_buf    =  cl.Buffer(context, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=EMI)   #  [N,N,N,NFREQ]
    ABS_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*3*BATCH)                      # ONLY reference frequencies
    EMI_buf  =  cl.Buffer(context, mf.WRITE_ONLY,  4*NFREQ*BATCH)        
    SOL      =  program.LibrarySolve
    SOL.set_scalar_arg_dtypes([np.int32, np.float32, np.float32, \
    None, None, None, None, None, None, None, None, None, None])
    
    a        =  0
    S        =  zeros((BATCH, NFREQ), float32)     # allocated for full set of frequencies
    ABS      =  zeros((BATCH, 3), float32)         # three reference frequencies
    BAD      =  zeros(CELLS, int32)
    while(a<CELLS):
        b    = min([CELLS, a+BATCH])
        no   = b-a
        if (0):
            ABS[0:no, :]  = 1.0*ABSORBED[a:b, IFREQ]            
        else:
            ABS[0:no, 0]  = 1.0*ABSORBED[a:b, IFREQ[0]]
            ABS[0:no, 1]  = 1.0*ABSORBED[a:b, IFREQ[1]]
            ABS[0:no, 2]  = 1.0*ABSORBED[a:b, IFREQ[2]]            
        cl.enqueue_copy(queue, ABS_buf, ABS)
        SOL(queue, [GLOBAL,], [LOCAL,],
        no, I0, dI0,   I1_buf, dI1_buf,   I2_buf, dI2_buf,   X_buf, Y_buf, Z_buf,   E_buf, ABS_buf, EMI_buf)
        cl.enqueue_copy(queue, S, EMI_buf)
        if (NOFREQ==0):   # we save emission for all frequencies
            EMITTED[a:b, :] = S[0:no, :]
        else:             # save NOFREQ frequencies only
            for ifreq in range(NOFREQ):
                EMITTED[a:b, ifreq] = S[0:no, OIFREQ[ifreq]]
                
        # missing emission vectors -- kernel checks the actual distance x-X
        m    = asarray(a+nonzero(S[0:no, 0]>1.0e31)[0], int32)
        bad  = len(m)
        if (bad>0):
            print("MIS[%d]   mis:(mis+bad) = %d:%d" % (len(MIS), mis, mis+bad))
            MIS[mis:(mis+bad)] = m       # icell for cells not matched by the library
            mis += bad
        ##
        a   += BATCH            

        
                
    t000  =  1.0e6*(time.time()-t000)/(float(CELLS))
    print("\n*** LIBRARY SOLVE %.2f USEC PER CELL ***\n" % t000)
    if (0): # testing - recompute everything cell by cell
        mis = CELLS


    # Drop from MIS all cells that fall outside of the limits of the library cube
    MIS  = MIS[0:mis]
    mok  = nonzero(i[MIS]>=0)     #  i<0 for any cell outside the library limits
    MIS  = MIS[mok]
    print("*** Drop from MIS cells outside the library limits: %d -> %d cells to be recomputed" % (mis, len(MIS)))
    mis  = len(MIS)
    
    # Recompute missing cells one by one -- note that some may be outside the limits of the library cube
    if ((mis>0)&(NFFREQ==NFREQ)): # some cells missing... but we can solve them without the library!!
        print("*** Missing cells: %d, %.3f %s of all cells" % (mis, 100.0*mis/float(CELLS), '%'))
        fp = open('missing.dat', 'wb')
        asarray([mis, NFREQ], int32).tofile(fp)
        for s in range(mis):
            asarray(ABSORBED[MIS[s],:], float32).tofile(fp)
        fp.close()
        
        if (EQDUST):
            os.system('EQ_solver.py  %s missing.dat  emitted.add 1' % FILE_SOLVER)
        else:
            os.system('A2E.py        %s missing.dat  emitted.add 1' % FILE_SOLVER)
        
        ADD = fromfile('emitted.add', float32)[2:].reshape(mis, NFREQ)
        if (NOFREQ==0):                       # copy all frequencies
            for icell in range(mis):          # loop over cells
                EMITTED[MIS[icell], :] = ADD[icell]
        else:                                 # copy only the NOFREQ selected output frequencies
            for ifreq in range(NOFREQ):       # loop over selected frequencies
                EMITTED[MIS, ifreq] = ADD[:, OIFREQ[ifreq]]  # ... all of the missing cells
        
        # add the missing entries also back to the library
        for s in range(mis):
            icell = MIS[s]                            # cell that required a new library entry
            ii, jj, kk         =  i[icell], j[icell], k[icell]
            if ((ii>=0)&(EMI[ii,jj,kk,0]<1e30)):      # this should have been be a missing entry in the library
                # Either the point was outside the library or the library entry was missing 
                # What should *not* happen is that (ii,jj,kk) corresponds to an already valid library entry
                # == EMI should be 1e32 (missing value)
                print("!!!!!  (i,j,k) %3d %3d %3d  --- EMI %10.3e" % (ii, jj, kk, EMI[ii,jj,kk,0]))
                if (0):
                    # observed reference intensities and their mapping to library cube indices
                    print("       IREF %5.2f %5.2f %5.2f == %5.2f %5.2f %5.2f =>  %2d %2d %2d" % 
                    ((IREF[icell,0]-I0)/dI0,  (IREF[icell,1]-I1[ii])/dI1[ii],  (IREF[icell,2]-I2[ii,jj])/dI2[ii,jj], x[icell], y[icell], z[icell], ii, jj, kk))
                    # the (x, y, z) values of that closest library grid point
                    print("       LIB  %5.2f %5.2f %5.2f" %   (X[ii,jj,kk], Y[ii,jj,kk], Z[ii,jj,kk]))

            X[ii, jj, kk]      =  x[icell]
            Y[ii, jj, kk]      =  y[icell]
            Z[ii, jj, kk]      =  z[icell]
            EMI[i[icell], j[icell], k[icell], :] =  ADD[s]    # library will always have the emission for all frequencies
            
        # rewrite the library ... but do not overwrite the old one?
        fp = open('%s.lib.new' % FILE_SOLVER, 'wb')
        asarray([N, NFREQ], int32).tofile(fp)
        asarray(FREQ, float32).tofile(fp)
        asarray([I0, dI0], float32).tofile(fp)
        asarray(I1,  float32).tofile(fp)
        asarray(dI1, float32).tofile(fp)
        asarray(I2,  float32).tofile(fp)
        asarray(dI2, float32).tofile(fp)
        # The actual reference intensity values for the bins in the (I0,I1,I2) grid --- AS INDICES TO BINS
        asarray(X, float32).tofile(fp)
        asarray(Y, float32).tofile(fp)
        asarray(Z, float32).tofile(fp)
        # The corresponding emission vectors.... or zeros if there were no cells in the bin
        EMI.tofile(fp)
        fp.close()
    
    else:
        if (mis<1):
            print("*** All cells matched by entries in the library ***")
        else:
            print("*** soc_library.py -- %d cells without answer in the library" % mis)
            print("***                   but file %s does contain only absorptions at %d frequencies" % (FILE_ABSORBED, NFFREQ))
            print("***                   emission of those cells set to ***ZERO***")
            EMITTED[MIS[:], :] =  0.0
                
        
