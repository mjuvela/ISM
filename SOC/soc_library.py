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

if (not(len(sys.argv)in[3,4])):
    print("Usage:  soc_library.py  [ solver  absorbed  |  solver.lib absorbed emitted")
    print("  With two arguments (solver file and absorption file), create te library.")
    print("  With three arguments (solver file, absorption file, emission file) use the")
    print("  library to compute the emission.")
    sys.exit()

FILE_SOLVER    =  sys.argv[1]
FILE_ABSORBED  =  sys.argv[2]
MAKE_LIBRARY   =  True
if (len(sys.argv)>3):            # emission file given .. we should be only using the library to solve emission
    FILE_EMITTED = sys.argv[3]
    MAKE_LIBRARY = False         # library should already exist...
if (MAKE_LIBRARY==False):        # ... but check this and make the library if it is still missing
    if (os.path.exists('%s.lib' % FILE_SOLVER)==False):
        print("soc_library.py using library %s.lib that does not yet exist" % FILE_SOLVER)
        print("==> we will now first create it")
        os.system('soc_library.py %s %s' % (FILE_SOLVER, FILE_ABSORBED))
        print("Library has been created - continue to solve all cells....")
        
FREQ     = loadtxt('freq.dat')
RUM      = asarray([0.2, 0.5, 1.2], float32)
FREF     = um2f(RUM)
N        = 30

# extract intensities at the reference wavelengths
fp       = open(FILE_ABSORBED, 'rb')
CELLS, NFREQ = fromfile(fp, int32, 2)
fp.close()
if (NFREQ!=len(FREQ)):
    print("Inconsistent number of frequencies")
    sys.exit()
    
ABSORBED = np.memmap(FILE_ABSORBED, dtype='float32', mode='r',  shape=(CELLS, NFREQ), offset=8)
IFREQ    = zeros(3, int32)
for i in range(3):
    IFREQ[i] = argmin(abs(FREF[i]-FREQ))
IREF     = zeros((CELLS, 3), float32)
for i in range(3):
    IREF[:, i] = ABSORBED[:,IFREQ[i]]
IREF     = clip(IREF, 1.0e-25, 1.0)     # IREF[CELLS, 3]
IREF     = log10(IREF)                  # all interpolations on log scale


if (MAKE_LIBRARY):
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
    for i in ind:                 # i == index to the full cloud
        ABSORBED[i,:].tofile(fp)
    fp.close()
    t000 = time.time()
    os.system('A2E.py   %s tmp.absorbed tmp.emitted' % FILE_SOLVER)
    t000 = 1.0e6*(time.time()-t000)/(float(cells))
    print("*** DIRECT SOLVE %.2f USEC PER CELL ***" % t000)


            
    
    # Write information on the binning and the emission vectors to a new file
    #  - number of bins
    #  - bin limits
    #  - bin indices do the cases for which we have the emission
    fp = open('%s.lib' % FILE_SOLVER, 'wb')
    asarray([N, NFREQ], int32).tofile(fp)
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
    TMP[m[0],:] =  tmp                                    # only bins for which corresponding cells were available
    TMP.tofile(fp)
    fp.close()
    # Note --- library lists log10 of intensity and of emission
    #          missing emission vectors are marked with -1.0

    
else:  # Using the library to solve emission
    
    
    # Read the library information
    fp       = open('%s.lib' % FILE_SOLVER, 'rb')
    N, NFREQ = fromfile(fp, int32, 2)
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
    print("EMITTED == %s" % FILE_EMITTED)
    fp = open(FILE_EMITTED, 'wb')
    asarray([CELLS, NFREQ], int32).tofile(fp)
    fp.close()
    
    EMITTED = np.memmap(FILE_EMITTED, dtype='float32', mode='r+',  shape=(CELLS, NFREQ), offset=8)
    
    # IREF => index to library
    x       =  (IREF[:,0]-I0     )/dI0                # cells =>  bin indices,   x[CELLS]
    i       =  clip(asarray(np.round(x), int32), 0, N-1)            # (i,j,k) indices to library cube
    y       =  (IREF[:,1]-I1[i]  )/dI1[i]             #  y[CELLS]
    j       =  clip(asarray(np.round(y), int32), 0, N-1)
    z       =  (IREF[:,2]-I2[i,j])/dI2[i,j]           #  z[CELLS]
    k       =  clip(asarray(np.round(z), int32), 0, N-1)
        
    S       =  zeros(NFREQ, float32)
    MIS     =  zeros(CELLS//5, int32)                 # store cells without library entry... 20% missing = something is wrong
    mis     =  0

    # double check that X[i,j,k] close to x etc.
    bad     =  abs(X[i,j,k]-x) + abs(Y[i,j,k]-y) + abs(Z[i,j,k]-z)
    m       =  nonzero(bad>3.0)
    print('bad %d' % len(m[0]))
    print('bad', bad[m])
    print("x", x[m])
    print("y", y[m])
    print("z", y[m])
    
    print(min(i), max(i))
    print(min(j), max(j))
    print(min(k), max(k))
    
    # (IREF0, IREF1, IREF2)  ->  (x, y, z)   ->  (i, j, k)  == bin indices
    # ==>     (x, y, z)  ~   (X[i,j,k], Y[i,j,k], Z[i,j,k])
    #  x, y, z, X, Y, Z  are all float indices  (I-I0)/dI0 etc.
        
    CASE = 4   # 0,1,2,3 = on Python,  4 = on OpenCL
        
    t000 = time.time()
    if (CASE in [0, 1, 2, 3]): # PURE PYTHON
        # must check if  X[ii, jj, kk] close to x[icell] etc. .... e.g. if no emission vectors for any kk for gievn (ii,jj)
        for icell in range(CELLS):
            if (icell%10000==0): 
                print(" CELL %6d / %6d  --- %5.2f %s" % (icell, CELLS, 100.0*icell/float(CELLS), '%'))
            W    = 0
            S[:] = 0.0
            # loop over closest 3x3x3 bins in the library
            # library grid log10(Iref) values are  (X, Y, Z)
            #         cell log10(Iref) values are  (x, y, z)  ---> library grid coordinates (i, j, k)
            CASE = 2
            if (CASE==0): # only direct matches
                ii, jj, kk = i[icell], j[icell], k[icell]   # (ii, jj, kk) in the library cube
                if ((ii>=0)&(EMI[ii, jj, kk, 0]<1e31)):
                    S   =  1.0*EMI[ii, jj, kk, :]
                else:
                    MIS[mis] = icell
                    mis     += 1
            elif (CASE==1):  # only direct matches + intensity correction
                ii, jj, kk = i[icell], j[icell], k[icell]
                if ((ii>=0)&(EMI[ii, jj, kk, 0]<1e31)):
                    # coeff = average step on the log scale....   (x-X) = delta(  (I-I0)/dI  )
                    coeff  =  0.333333 *  (x[icell]-X[ii,jj,kk])*dI0        #  I-I0
                    coeff +=  0.333333 *  (y[icell]-Y[ii,jj,kk])*dI1[ii]    #  I-I1
                    coeff +=  0.333333 *  (z[icell]-Z[ii,jj,kk])*dI2[ii,jj] #  I-I2
                    # now coeff is  ~ delta lg(I)  =>   I *= 10.0**delta
                    coeff  =  10.0**coeff
                    S      =  coeff*EMI[ii, jj, kk, :]
                else:
                    MIS[mis] = icell
                    mis     += 1                
            elif (CASE==2):   # average over up to 3x3x3 bins
                for ii in range(max([0, i[icell]-1]), min([N, i[icell]+2])):
                    for jj in range(max([0, j[icell]-1]), min([N, j[icell]+2])):
                        for kk in range(max([0, k[icell]-1]), min([N, k[icell]+2])):
                            if (EMI[ii, jj, kk, 0]>1e31): continue        # if this bin does not have an emission vector
                            if (abs(x[icell]-X[ii,jj,kk])>1.0): continue  # bin is too far from the cell values
                            if (abs(y[icell]-Y[ii,jj,kk])>1.0): continue
                            if (abs(z[icell]-Z[ii,jj,kk])>1.0): continue
                            w   =  1.0/(0.1+(x[icell]-X[ii,jj,kk])**2.0)  #  ~ 1 /  distance_to_bin_entry^p
                            w  *=  1.0/(0.1+(y[icell]-Y[ii,jj,kk])**2.0)
                            w  *=  1.0/(0.1+(z[icell]-Z[ii,jj,kk])**2.0)
                            W  +=  w
                            S  +=  w*EMI[ii, jj, kk, :]
                ii, jj, kk = i[icell], j[icell], k[icell]
                if (icell%10000==-1):
                    print("   %6d -- %3d %3d %3d -- W=%.3e -- %5.1f %5.1f %5.1f" % \
                    (icell, i[icell], j[icell], k[icell], W, x[icell]-X[ii,jj,kk], y[icell]-Y[ii,jj,kk], z[icell]-Z[ii,jj,kk]))
                if (W>0.0):
                    S        /= W
                else:
                    MIS[mis]  = icell
                    mis      += 1
            elif (CASE==3):   # average over up to 3x3x3 bins + intensity correction
                coeff = 1.0
                for ii in range(max([0, i[icell]-1]), min([N, i[icell]+2])):
                    for jj in range(max([0, j[icell]-1]), min([N, j[icell]+2])):
                        for kk in range(max([0, k[icell]-1]), min([N, k[icell]+2])):
                            if (EMI[ii, jj, kk, 0]<1e31):                        # if this bin does have an emission vector
                                coeff  =  0.333333 *  (x[icell]-X[ii,jj,kk])*dI0        #  I-I0
                                coeff +=  0.333333 *  (y[icell]-Y[ii,jj,kk])*dI1[ii]    #  I-I1
                                coeff +=  0.333333 *  (z[icell]-Z[ii,jj,kk])*dI2[ii,jj] #  I-I2
                                if ((coeff>-1.0)&(coeff<1.0)):
                                    pass
                                else:
                                    print("--------------------------------------------------------------------------------")
                                    print(" coeff  %8.3f" % coeff) ;
                                    print(" %5.2f %5.2f   %5.2f %5.2f   %5.2f %5.2f -- %5.3f %5.3f %5.3f" % (
                                    x[icell], X[ii,jj,kk],  y[icell], Y[ii,jj,kk],  z[icell], Z[ii,jj,kk],
                                    dI0, dI1[ii], dI2[ii,jj]))
                                    print(" %10.3e %10.3e %10.3e..." % (EMI[ii,jj,kk,0], EMI[ii,jj,kk,1], EMI[ii,jj,kk,2]))
                                    print("--------------------------------------------------------------------------------")
                                if (abs(coeff)>1.0): continue # the point X/Y/Z(ii,jj,kk) is far from asked x/y/z(icell) !!
                                coeff  =  10.0**coeff
                                w      =  1.0/(0.1+(x[icell]-X[ii,jj,kk])**2.0)  #  ~ 1 /  distance_to_bin_entry^p
                                w     *=  1.0/(0.1+(y[icell]-Y[ii,jj,kk])**2.0)
                                w     *=  1.0/(0.1+(z[icell]-Z[ii,jj,kk])**2.0)
                                W     +=  w
                                S     +=  w*coeff*EMI[ii, jj, kk, :]                        
                ii, jj, kk = i[icell], j[icell], k[icell]
                if (icell%1000==-1):
                    print("   %6d -- %3d %3d %3d -- W=%.3e -- %5.1f %5.1f %5.1f" % \
                    (icell, i[icell], j[icell], k[icell], W, x[icell]-X[ii,jj,kk], y[icell]-Y[ii,jj,kk], z[icell]-Z[ii,jj,kk]))
                if (W>0.0):
                    S        /= W
                else:
                    MIS[mis]  = icell
                    mis      += 1
                
            EMITTED[icell,:] = S
            
    else:  # OpenCL version

        METHOD   =  0
        GPU      =  1                  
        PLF      =  [ 0, 1, 2, 3 ]      
        platform, device, context, queue, mf = InitCL(GPU, PLF)
        LOCAL    =  [4, 32][GPU>0]
        BATCH    =  8192
        GLOBAL   =  BATCH
        OPT      = '-D CELLS=%d -D N=%d -D NFREQ=%d -D R0=%d -D R1=%d -D R2=%d -D LOCAL=%d -D METHOD=%d' % \
        (CELLS, N, NFREQ, IFREQ[0], IFREQ[1], IFREQ[2], LOCAL, METHOD)
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
        ABS_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NFREQ*BATCH)  
        EMI_buf  =  cl.Buffer(context, mf.WRITE_ONLY,  4*NFREQ*BATCH)        
        SOL      =  program.LibrarySolve
        SOL.set_scalar_arg_dtypes([np.int32, np.float32, np.float32, \
        None, None, None, None, None, None, None, None, None, None])
        
        a        =  0
        S        =  zeros((BATCH, NFREQ), float32)
        ABS      =  zeros((BATCH, NFREQ), float32)
        BAD      =  zeros(CELLS, int32)
        while(a<CELLS):
            b    = min([CELLS, a+BATCH])
            no   = b-a
            ABS[0:no, :]  = 1.0*ABSORBED[a:b, :]
            cl.enqueue_copy(queue, ABS_buf, ABS)
            SOL(queue, [GLOBAL,], [LOCAL,],
            no, I0, dI0,   I1_buf, dI1_buf,   I2_buf, dI2_buf,   X_buf, Y_buf, Z_buf,   E_buf, ABS_buf, EMI_buf)
            cl.enqueue_copy(queue, S, EMI_buf)
            EMITTED[a:b, :] = S[0:no, :]
            # missing emission vectors -- kernel checks the actual distance x-X
            m    = asarray(a+nonzero(S[0:no, 0]>1.0e31)[0], int32)
            bad  = len(m)
            if (bad>0):                
                MIS[mis:(mis+bad)] = m       # icell for cells not matched by the library
                mis += bad
            ##
            a   += BATCH            

                
    t000  =  1.0e6*(time.time()-t000)/(float(CELLS))
    print("*** LIBRARY SOLVE %.2f USEC PER CELL ***" % t000)
    if (0): # testing - recompute everything cell by cell
        mis = CELLS


    # Drop from MIS all cells that fall outside of the limits of the library cube
    MIS  = MIS[0:mis]
    mok  = nonzero(i[MIS]>=0)     #  i<0 for any cell outside the library limits
    MIS  = MIS[mok]
    print("*** Drop from MIS cells outside the library limits: %d -> %d cells to be recomputed" % (mis, len(MIS)))
    mis  = len(MIS)
    
    # Recompute missing cells one by one -- note that some may be outside the limits of the library cube
    if (mis<1):
        print("*** All cells matched by entries in the library ***")
    else:
        print("*** Missing cells: %d, %.3f %s of all cells" % (mis, 100.0*mis/float(CELLS), '%'))
        fp = open('missing.dat', 'wb')
        asarray([mis, NFREQ], int32).tofile(fp)
        for s in range(mis):
            asarray(ABSORBED[MIS[s],:], float32).tofile(fp)
        fp.close()
        os.system('A2E.py %s missing.dat emitted.add 1' % FILE_SOLVER)
        ADD = fromfile('emitted.add', float32)[2:].reshape(mis, NFREQ)
        for s in range(mis):
            EMITTED[MIS[s], :] = ADD[s]
        
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
            EMI[i[icell], j[icell], k[icell], :] =  ADD[s]
            
        # rewrite the library
        fp = open('%s.lib.new' % FILE_SOLVER, 'wb')
        asarray([N, NFREQ], int32).tofile(fp)
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
        
