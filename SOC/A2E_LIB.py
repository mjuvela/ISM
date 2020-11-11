#!/usr/bin/python

if (0):
    from MJ.mjDefs import *
else:
    import os, sys
    import numpy as np
    
import pyopencl as cl

INSTALL_DIR  = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)


if (len(sys.argv)<3):
    #                     1             2          3        4        5             6
    print()
    print("Solve emission for stochastically heated grain using a library")
    print("Usage:")
    print("   A2E_LIB.py  <dust>.solver <dust>.lib freq.dat lfreq.dat abs.data emit.data [makelib] [GPU] [ofreq] [bins]")
    print("Input:")
    print("   <dust>.solver  =  solver file written by A2E_pre.py")
    print("   <dust>.lib     =  name of the library file")
    print("   freq.dat       =  file with the full list of frequencies (frequencies in emitted.dat)")
    print("   lfreq.dat      =  file with the list of reference frequencies ")
    print("   absorbed.data  =  file with the absorptions [CELLS, nfreq] when creating the library,")
    print("                     [CELLS, nfreq] or [CELLS, nlfreq] when using the library")
    print("   emitted.data   =  solved emissions, always [CELLS, nfreq]")
    print("   makelib        =  make the library, A2E.py is first called to solve emission for every cell")
    print("                     => absorbed.data must have all the nfreq frequencies from freq.dat")
    print("                     library is created but emitted.data is the solution from normal A2E.py run")
    print("   GPU            =  if given, use GPU instead of CPU")
    print("   ofreq          =  optional filename listing a subset of frequencies, only these written to emitted")
    print("   number of bins can be defined using a format like  bins-45-25-15")
    print("Notes:")
    print("   When solving, absorbed.data can have either the full set of frequencies or only the")
    print("   reference frequencies. Once library is built, absorptions are needed only for lfreq")
    sys.exit()
  
    
SOLVER   =  sys.argv[1]
LIBNAME  =  sys.argv[2]
FREQ     =  loadtxt(sys.argv[3])
LFREQ    =  loadtxt(sys.argv[4])
NFREQ    =  len(FREQ)
NLFREQ   =  len(LFREQ)
AFILE    =  sys.argv[5]
EFILE    =  sys.argv[6]
F_OFREQ  =  ""
LIB_BINS =  [ 45, 34, 15 ]
SHAREDIR = '/dev/shm'

# file for absorptions
CELLS, nfreq = fromfile(AFILE, int32, 2)
CREATE = False
GPU, MAKELIB = 0, 0
for arg in sys.argv[7:]:
    if (arg.lower()=="gpu"): 
        GPU     = 1
    elif (arg=='makelib'):    
        MAKELIB = 1
    elif (arg.find('bins')>=0):  # binning defined with a string like  bins-45-15-15
        s = arg.split('-')
        LIB_BINS = [ int(s[1]), int(s[2]), int(s[3]) ]
    else:
        F_OFREQ = arg

        
        
if (MAKELIB):
    if (nfreq!=NFREQ):
        print("*** We need %d frequencies when library is created, absorption file has %d" % (nfreq, NFREQ))
        sys.exit()
else:
    if (nfreq==NFREQ):
        print("*** Solve using library, only %d frequencies used of the %d in the absorption file" % (NLFREQ, NFREQ))
    elif (nfreq==NLFREQ):
        print("*** Solve using library, using the %d reference frequencies in the absorption file" % (NLFREQ))
    else:
        print("*** Solve using library: need %d reference frequencies, absorption file has %d" % (NLFREQ, NFREQ))
        sys.exit()
        

    
class Node:
    amin   =  0.0
    bins   =  0.0
    child  =  None
    ARRAY  =  None
        


    
    
def InitCL(GPU=0, platforms=[0,1,2,3,4]):
    """
    platform, device, context, queue, mf = InitCL(GPU=0, platforms=[0,1,2,3,4])
    """
    platform, device, context, queue = None, None, None, None
    for iplatform in platforms:
        try:
            platform = cl.get_platforms()[iplatform]
            if (GPU>0):
                device   = platform.get_devices(cl.device_type.GPU)
            else:
                device   = platform.get_devices(cl.device_type.CPU)
            context   =  cl.Context(device)
            queue     =  cl.CommandQueue(context)
            break
        except:
            pass
    return platform, device, context, queue,  cl.mem_flags


    
def create_library_1(solver, libname, freq, lfreq, file_absorbed, K=1.1, GPU=0):
    """
    Create a library based on provided file of absorptions (by this dust species only!).
    Input:
        solver  =  name of solver file (as read by A2E_pyCL.py)
        libname =  name of the library to be created
        freq    =  full vector of frequencies (as in file_absorbed and file_emitted)
        lfreq   =  input frequencies in the library
        file_absorbed = absorptions written by SOC
        K       =  logarithmic step in the binning of absorptions (default 1.1)
    Result:
        Writes new library file libname that gives the mapping from absorptions at lfreq
        frequencies and the emission at freq frequencies.
    Note:
        Routine uses pieces from A2E_pyCL.py code to calculate the emission for given
        absorption vector
    """
    nlfreq = len(lfreq)
    # read data at reference frequencies from file_absorbed so that we see which
    # cases need to be calculated
    cells, nfreq = fromfile(file_absorbed, int32, 2)
    if (nfreq!=len(freq)):
        print("create_library: frequency file has %d but absorption file %d frequencies" % (len(freq), nfreq))
        sys.exit()
    ABSORBED = np.memmap(file_absorbed, dtype='float32', mode='r', offset=8, shape=(cells, nfreq))
    ABS      = zeros((cells, nlfreq), float32)
    IFREQ    = zeros(3, int32)
    for ilib in range(nlfreq):
        i    = argmin(abs(lfreq[ilib]-freq))
        if (abs(lfreq[ilib]-freq[i])>0.01*lfreq[ilib]):
            print("*** WARNING: library frequency %12.4e does not match absorption frequency %12.4e" % (lfreq[ilib], freq[i]))
        IFREQ[ilib] = i
        print("  ***  IFREQ[%d] = %d" % (ilib, i))
        
    for ilib in range(3):
        ABS[:, ilib] = ABSORBED[:, IFREQ[ilib]]

    AMIN, AMAX, BINS  =  zeros(3, float32),   zeros(3, float32),    zeros(3, int32)    
    #   x = amin*K^i     amax=amin*K^bins  => bins = log(amax/amin) / log(K),    i = log(x/amin)/log(K)
    #                                                                              
    #   ifreq=0             [ [               ] ]                       LINKS[0]   
    #                              /     \                                         
    #                             /       \                                        
    #                            /         \                                       
    #   ifreq=1          [    [   ]       [   ]     [    ]    ]         LINKS[1]   
    #                           /                                                  
    #                          /                                                   
    #                         /                                                    
    #   ifreq=2          [  [   ]   [  ]  ]                             LINKS[2]   
    m                 =  nonzero(ABS[:,0]>0.0)
    if (1):
        AMIN[0], AMAX[0]  =  0.9999*min(ABS[m[0],0]), 1.0001*max(ABS[m[0],0])
    else:
        AMIN[0], AMAX[0]  =  percentile(ABS[m[0],0], (0.01, 99.99))
    BINS[0]           =  1+int(1+log(AMAX[0]/AMIN[0]) / log(K))
    print("A2E_LIB.py -> create_library -> AMIN[0] %10.3e AMAX[0] %10.3e BINS[0] %d" % (AMIN[0], AMAX[0], BINS[0]))
    ii = nonzero(ABS[:,0]== min(ABS[m[0],0]))
    print("   minimum value in cells ", ii)
    if (0):
        clf()
        hist(ABS[m[0],0], 500)
        show()
        sys.exit()
    
    VEC               =  [ [], [], [] ]
    AMI               =  [ [], [], [] ]
    NODES             =  [ 1, 0, 0 ]
    VEC[0].append(arange(BINS[0]))  # VEC[0] contains node numbers of i1 vectors
    AMI[0].append(AMIN[0])

    SARRAY = []     # array of the absorptions
    SIND   =  0     # running index to the absorptions array
    for i0 in range(BINS[0]):        
        amin, amax       =  AMIN[0]*K**i0,  AMIN[0]*K**(i0+1)   # limits of a single i0 bin
        MASK0            =  ones(CELLS, int32)
        MASK0[nonzero((ABS[:,0]<amin)|(ABS[:,0]>amax))] = 0
        MASK0[nonzero((ABS[:,1]<=0.0)|(ABS[:,2]<=0.0))] = 0
        m                =  nonzero(MASK0>0)                    # cells in single i0 bin
                
        if (len(m[0])<1):
            AMIN[1], BINS[1] = -1.0, 0
        else:
            if (0):
                AMIN[1], AMAX[1]  =  0.9999*min(ABS[m[0],1]), 1.0001*max(ABS[m[0],1])
            else:
                AMIN[1], AMAX[1] =  percentile(ABS[:,1][m], (0.01, 99.99))
                AMIN[1], AMAX[1] =  0.9999*AMIN[1], 1.0001*AMAX[1]
            if (AMAX[1]<=0.0):
                AMIN[1], BINS[1] = -1.0, 0
            else:
                if (AMIN[1]<(1.0e-3*AMAX[1])):  AMIN[1] = 1.0e-3*AMAX[1] # all bins must be positive !!
                BINS[1] =  int(2+log(AMAX[1]/AMIN[1]) / log(K))
        #print("0: BIN %2d/%2d  [%10.3e, %10.3e]  %6d   %12.4e" % (i0, BINS[0], amin, amax, len(m[0]), mean(ABSORBED[m[0],IFREQ[0]])))
                
        # level 0 element i0 has vector on level 1, number of elements = i1 discretisation
        # one i0 element has the number of one i1 vector
        # we create the i1 vector and we know that i2 vectors will be created in order => add already numbers of i2 vec
        VEC[1].append( arange(NODES[2], NODES[2]+BINS[1]) )     # one i0 elem. -> one i1 vector -> point to i2 nodes
        AMI[1].append( AMIN[1] )
        NODES[2] += BINS[1]   # one VECTOR added to level 1
        
        for i1 in range(BINS[1]): # we now create i2 vectors in order, all i2 vec for current i1 vec
            amin, amax        =  AMIN[1]*K**i1,  AMIN[1]*K**(i1+1)        # limits for a single i1 bin
            MASK1             =  MASK0.copy()
            MASK1[nonzero((ABS[:,1]<amin)|(ABS[:,1]>amax))] = 0           # cells matching i0 and i1 bins
            m                 =  nonzero(MASK1>0)
            
            #print("  1: BIN %2d/%2d  [%10.3e, %10.3e]  %6d   %12.4e" % (i1, BINS[1], amin, amax, len(m[0]), mean(ABSORBED[m[0],IFREQ[1]])))

            if (len(m[0])<1):
                AMIN[2], BINS[2] = -1.0, 0
            else:
                if (0):
                    AMIN[2], AMAX[2]  =  0.9999*min(ABS[:,2][m]), 1.0001*max(ABS[:,2][m])   # limits for i2 vector
                else:
                    AMIN[2], AMAX[2]  =  percentile(ABS[:,2][m], (0.01, 99.99))
                    AMIN[2], AMAX[2]  =  0.9999*AMIN[2], 1.0001*AMAX[2]                    
                if (AMAX[2]<=0.0):
                    AMIN[2], BINS[2] = -1.0, 0
                else:
                    if (AMIN[2]<(1.0e-3*AMAX[2])): AMIN[2] = 1.0e-3*AMAX[2]  # all bins must be positive !!
                    BINS[2]  =  1+int(1+log(AMAX[2]/AMIN[2]) / log(K))
                #print("    2:  %d cells,  AMIN[2] %12.4e   AMAX[2] %12.4e" % (len(m[0]), AMIN[2], AMAX[2]))
                    
            VEC[2].append( zeros(BINS[2], int32) )                        # one i1 element -> new i2 vector
            AMI[2].append( AMIN[2] )
            
            for i2 in range(BINS[2]):                                     # set values in each i2 bin
                amin, amax        =  AMIN[2]*K**i2,  AMIN[2]*K**(i2+1)    # limits of a single i2 bin
                MASK2             =  MASK1.copy()                         # mask for parent i0, i1 bins
                MASK2[nonzero((ABS[:,2]<amin)|(ABS[:,2]>amax))] = 0       # mask for current i0, i1, i2 bin
                m                 =  nonzero(MASK2>0)
                
                #print("    2: BIN %2d/%2d  [%10.3e, %10.3e]  %6d   %12.4e" % (i2, BINS[2], amin, amax, len(m[0]), mean(ABSORBED[m[0],IFREQ[2]])))

                if (len(m[0])<1):
                    res           = -1.0
                else:
                    res           =  SIND                                 # index to the ABSORPTIONS array
                VEC[2][-1][i2]    =  res                                  # set value in the leaf
                if (res>=0): # we do have some data for this bin
                    if (SIND==0):                        
                        SARRAY =  mean(ABSORBED[m[0],:], axis=0)  # average absorptions for the bin
                    else:
                        SARRAY =  concatenate((SARRAY, mean(ABSORBED[m[0],:], axis=0)))
                    SIND += 1
    SARRAY.shape = (SIND, nfreq)

    if (0):
        clf()
        subplot(221)
        imshow(SARRAY, aspect='auto')
        subplot(222)
        plot(SARRAY[:, IFREQ[0]])
        subplot(223)
        plot(SARRAY[:, IFREQ[1]])
        subplot(224)
        plot(SARRAY[:, IFREQ[2]])
        show()
        sys.exit()
                    
    # Now we have in SARRAY[SIND, nfreq] the absorption vectors for all tree nodes
                
    # Solve emission for all absorption vectors in ABSORBED... use A2E_pyCL
    fp = file(SHAREDIR+'/absorbed.lib', 'wb')
    asarray([SIND, nfreq], int32).tofile(fp)
    asarray(SARRAY, float32).tofile(fp)
    fp.close()
    os.system('A2E_pyCL_2.py %s %s/absorbed.lib %s/emitted.lib %d' % (solver, SHAREDIR, SHAREDIR, GPU))
    # Copy solved emissions back to ARRAY
    fp = file(SHAREDIR+'/emitted.lib', 'rb')
    fromfile(fp, int32, 2)
    SARRAY = fromfile(fp, float32)
    fp.close()
    
    if (0):
        SARRAY.shape = (SIND, nfreq)
        clf()
        subplot(221)
        imshow(SARRAY, aspect='auto')
        subplot(222)
        plot(SARRAY[:, nfreq//4])
        subplot(223)
        plot(SARRAY[:, nfreq//2])
        subplot(224)
        plot(SARRAY[:, (3*nfreq)//4])
        show()
        sys.exit()
    
    # Dump the library to a file
    # number of vectors at each level                          ->  (N0), NV1, NV2
    # single concatenated vector for each level                ->  V0, V1, V2    
    # index of first element of a vector within (V0,V1,V2):    ->  (I0), I1, I2  
    # number of elements in each vector                        ->  N0, N1, N2    
    # AMIN for each vector                                     ->  A0, A1, A2
    VEC0  =  asarray(VEC[0][0], int32)
    AMI0  =  AMI[0][0]
    NE0   =  len(VEC[0][0])
    V0    =  asarray(VEC[0][0], int32)
    A0    =  AMI[0][0]
    NV1   =  len(VEC[1])           #  number of vectors on this level
    I1    =  zeros(NV1, int32)     #  first element of each vector
    N1    =  zeros(NV1, int32)     #  number of bins = elements in each vector
    A1    =  zeros(NV1, float32)   #  amin for each vector in V1
    i     =  0                     #  running index
    for j in range(len(VEC[1])):   #  loop over VEC[1] vectors
        A1[j]  =  AMI[1][j]
        I1[j]  =  i                #  first element of this vector
        N1[j]  =  len(VEC[1][j])   #  number of elements in this vector
        i     +=  N1[j]
    NE1   =  i                     # total number of elements on level1
    V1    =  zeros(NE1, int32)
    for j in range(len(VEC[1])):
        V1[I1[j]:(I1[j]+N1[j])] = VEC[1][j]   
    NV2   =  len(VEC[2])           #  number of vectors on this level
    I2    =  zeros(NV2, int32)     #  first element of each vector
    N2    =  zeros(NV2, int32)     #  number of bins = elements in each vector
    A2    =  zeros(NV2, float32)   #  amin for each vector in V2
    i     =  0                     #  running index
    for j in range(len(VEC[2])):   #  loop over VEC[1] vectors
        A2[j]  =  AMI[2][j]
        I2[j]  =  i                #  first element of this vector
        N2[j]  =  len(VEC[2][j])   #  number of elements in this vector
        i     +=  N2[j]
    NE2   =  i                     # total number of elements on level1
    V2    =  zeros(NE2, int32)     # indices to SARRAY
    for j in range(len(VEC[2])):
        V2[I2[j]:(I2[j]+N2[j])] = VEC[2][j]        
    
    fp = open(libname, 'wb')
    asarray([NE0, NV1, NV2, NE1, NE2, SIND, nfreq, 3], int32).tofile(fp)
    asarray(freq, float32).tofile(fp)       # all frequencies
    asarray(lfreq, float32).tofile(fp)      # reference frequencies
    asarray([K, A0], float32).tofile(fp)
    asarray(V0, int32).tofile(fp)
    asarray(N1, int32).tofile(fp)
    asarray(I1, int32).tofile(fp)
    asarray(A1, float32).tofile(fp)
    asarray(V1, int32).tofile(fp)
    asarray(N2, int32).tofile(fp)
    asarray(I2, int32).tofile(fp)
    asarray(A2, float32).tofile(fp)
    asarray(V2, int32).tofile(fp)    # these are now indices to the SARRAY
    asarray(SARRAY, float32).tofile(fp)
    fp.close()
    
    

    
    
def solve_with_library_1(libname, freq, lfreq, file_absorbed, file_emitted, GPU=0):
    """
    Calculate emissions to file_emitted based on the absorptions in file_absorbed and the
    provided library file libname.
    Input:
        libname       =  name of the library file
        freq          =  full vector of frequencies (as in file_emitted)
        lfreq         =  vector of input frequencies used in file_absorbed
        file_absorbed =  absorption file written by SOC
        file_emitted  =  file of emissions for lfreq frequencies
    """
    nfreq  = len(freq)
    nlfreq = len(lfreq)
    # Load library information from the  file
    fp = open(libname, 'rb')
    NE0, NV1, NV2, NE1, NE2, SIND, nfreq, nlfreq  =  fromfile(fp, int32, 8)
    tfreq  = fromfile(fp, float32, nfreq)
    tlfreq = fromfile(fp, float32, nlfreq)
    
    if (len(tfreq)!=nfreq):
        print("*** solve_with_library: Number of frequencies does not match, %d in tree, %d in parameter" % (len(tfreq), nfreq))
        sys.exit(0)
    if (max(abs(freq-tfreq)/freq)>0.001):
        print("*** solve_with_library: Frequency values do not match")
        print("*** given parameter: ", freq)
        print("***  library file: ", tfreq)
        sys.exit(0)
    if (len(tlfreq)!=nlfreq):
        print("*** solve_with_library: Number of reference frequencies does not match, %d in tree, %d in parameter" % (len(tlfreq), nlfreq))
        sys.exit(0)
    if (max(abs(lfreq-tlfreq)/lfreq)>0.001):
        print("*** solve_with_library: Reference frequency values do not match")
        print("*** given parameter: ", lfreq)
        print("***  library file: ", tlfreq)
        sys.exit(0)
        
    K, A0  =  fromfile(fp, float32, 2)   #   K step value,  amin of level 0
    V0     =  fromfile(fp, int32,    NE0)   #   vector of level 0
    N1     =  fromfile(fp, int32,    NV1)   #   subvectors on level 1
    I1     =  fromfile(fp, int32,    NV1)   #   start indices of level 1 subvectors
    A1     =  fromfile(fp, float32,  NV1)   #   amin for level 1 subvectors
    V1     =  fromfile(fp, int32,    NE1)   #   vector of level 1
    N2     =  fromfile(fp, int32,    NV2)   #   number of elements in each level 2 subvector
    I2     =  fromfile(fp, int32,    NV2)   #   start indices to level 2 subvectors
    A2     =  fromfile(fp, int32,    NV2)   #   amin for level 2 subvectors
    V2     =  fromfile(fp, int32,    NE2)   #   vector of level 2 values = indices to SARRAY
    SARRAY =  fromfile(fp, float32)     #  SARRAY[leafs, nfreq]
    fp.close()
    SARRAY.shape =  (SIND, nfreq)      #  emission vector for each tree leaf

    # Read absorptions
    fp = file(file_absorbed, 'rb')
    cells, b = fromfile(fp, int32, 2)
    if ((b!=nfreq)&(b!=nlfreq)):
        print("*** ERROR: solve_with_library: absorption file has %d, library %d frequencies" % (b, nlfreq)), sys.exit(0)
    if (b==nfreq):
        # --------------------------------------------------------------------------------
        # for testing --- input is full absorbed.data and we just pick the reference bands
        #                 normally we would have absorptions only for the reference frequencies
        tmp = fromfile(fp, float32).reshape(cells, nfreq)
        ABS = zeros((cells, nlfreq), float32)
        for i in range(nlfreq):
            ifreq = argmin(abs(freq-lfreq[i]))
            ABS[:, i] = tmp[:, ifreq]
        # --------------------------------------------------------------------------------
            
    else:
        ABS = fromfile(fp, float32).reshape(cells, nlfreq)
    fp.close()

    # Solve with OpenCL kernel
    platform, device, context, queue,  mf = InitCL(GPU=GPU)
    GLOBAL   =  32768
    LOCAL    =  4
    if (GPU>0): LOCAL = 32
    
    abs_buf =  cl.Buffer(context, mf.READ_ONLY,   4*GLOBAL*nlfreq)  # absorptions at reference frequencies
    res_buf =  cl.Buffer(context, mf.WRITE_ONLY,  4*GLOBAL*nfreq)   # emission after the lookup
    V0_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NE1)
    N1_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NV1)
    I1_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NV1)
    A1_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NV1)
    V1_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NE1)
    N2_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NV2)
    I2_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NV2)
    A2_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NV2)
    V2_buf  =  cl.Buffer(context, mf.READ_ONLY,   4*NE2)
    SA_buf   =  cl.Buffer(context, mf.WRITE_ONLY, 4*SIND*nfreq)     # emission vectors for the tree leafs
    
    cl.enqueue_copy(queue, V0_buf, asarray(V0, int32))
    cl.enqueue_copy(queue, N1_buf, N1)
    cl.enqueue_copy(queue, I1_buf, I1)
    cl.enqueue_copy(queue, A1_buf, A1)
    cl.enqueue_copy(queue, V1_buf, V1)
    cl.enqueue_copy(queue, N2_buf, N2)
    cl.enqueue_copy(queue, I2_buf, I2)
    cl.enqueue_copy(queue, A2_buf, A2)
    cl.enqueue_copy(queue, V2_buf, V2)
    cl.enqueue_copy(queue, SA_buf, SARRAY)
    
    OPT      =  "-D K=%.3ef -D NFREQ=%d -D SIND=%d" % (K, nfreq, SIND)
    source   =  open(INSTALL_DIR+"/kernel_SOC_lookup.c").read()
    program  =  cl.Program(context, source).build(OPT)    
    Lookup   =  program.Lookup
    #                              cells     abs[]   res[]
    Lookup.set_scalar_arg_dtypes([ np.int32, None,   None,
    #  NE0    A0          V0     NV1       N1    I1    A1    V1    NV2       N2    I2    A2    V2    SARRAY
    np.int32, np.float32, None,  np.int32, None, None, None, None, np.int32, None, None, None, None, None])
        

    # Loop over cells
    fp = file(file_emitted, 'wb')
    asarray([cells, nfreq], int32).tofile(fp)
    tmp    =  zeros((GLOBAL, nfreq), float32)
    for ibatch in range(cells//GLOBAL+1):
        a, b   =  ibatch*GLOBAL,  min([(ibatch+1)*GLOBAL, cells])
        if (a>=cells): break
        cl.enqueue_copy(queue, abs_buf, ABS[a:b,:])
        Lookup(queue, [GLOBAL,], [LOCAL,],  CELLS, abs_buf, res_buf,
               NE0, A0, V0_buf, NV1, N1_buf, I1_buf, A1_buf, V1_buf, NV2, N2_buf, I2_buf, A2_buf, V2_buf, SA_buf)
        cl.enqueue_copy(queue, tmp, res_buf)
        asarray(tmp[a:b,:], float32).tofile(fp)
    fp.close()

    
    if (1): # *** TESTING ***
        figure(1, figsize=(8,7))
        subplots_adjust(left=0.08, right=0.96, bottom=0.08, top=0.93, wspace=0.25)
        
        subplot(231)
        x = fromfile(file_emitted, float32)[2:].reshape(cells, nfreq) * 1e3
        title("Library solution")
        imshow(x, aspect='auto')
        colorbar()
    
        subplot(232)
        # Solve again... without the library method !!
        os.system('A2E_pyCL.py aSilx.solver absorbed.test emitted.ref')
        y = fromfile('emitted.ref', float32)[2:].reshape(cells, nfreq) * 1e3
        imshow(y, aspect='auto')
        title("Direct solution")
        colorbar()
        
        subplot(233)
        imshow((x-y)/y, aspect='auto')
        title("Relative error")
        colorbar()
        
        
        subplot(212)
        um = f2um(freq)
        i  = 0
        for icell in range(0, cells, 10):
            m = [ 'o', 's', 'd', '+', 'x' ][i%5]
            loglog(um, x[icell,:], 'r-'+m, ms=5, zorder=3)
            loglog(um, y[icell,:], 'b--'+m, ms=3, zorder=4)
            i += 1
        
        show()
        sys.exit()
        
    return




def create_library_2(solver, libname, FREQ, LFREQ, file_absorbed, BINS=[40,30,10], GPU=0):
    """
    Create a library based on provided file of absorptions (by this dust species only!).
    This version uses a pre-defined constant  number of bins for each of the three levels
    Input:
        solver  =  name of solver file (as read by A2E_pyCL.py)
        libname =  name of the library to be created
        FREQ    =  full vector of frequencies (as in file_absorbed and file_emitted)
        LFREQ   =  input frequencies in the library
        file_absorbed = absorptions written by SOC
        K       =  logarithmic step in the binning of absorptions (default 1.1)
    Result:
        Writes new library file libname that gives the mapping from absorptions at lfreq
        frequencies and the emission at freq frequencies.
    Note:
        Routine uses pieces from A2E_pyCL.py code to calculate the emission for given
        absorption vector
    """
    BINS0, BINS1, BINS2 = BINS
    NFREQ  = len(FREQ)
    NLFREQ = len(LFREQ)
    # read data at reference frequencies from file_absorbed so that we see which
    # cases need to be calculated
    CELLS, nfreq = fromfile(file_absorbed, int32, 2)
    if (nfreq!=len(FREQ)):
        print("create_library: we have %d frequencies, the absorption file has %d" % (NFREQ, nfreq))
        sys.exit()
    ABSORBED = np.memmap(file_absorbed, dtype='float32', mode='r', offset=8, shape=(CELLS, NFREQ))
    ABS      = zeros((CELLS, NLFREQ), float32)
    IFREQ    = zeros(3, int32)
    for ilib in range(NLFREQ):
        i    = argmin(abs(LFREQ[ilib]-FREQ))
        if (abs(LFREQ[ilib]-FREQ[i])>0.01*LFREQ[ilib]):
            print("*** WARNING: library frequency %12.4e does not match absorption frequency %12.4e" % (LFREQ[ilib], FREQ[i]))
        IFREQ[ilib] = i
        print("  ***  IFREQ[%d] = %d" % (ilib, i))        
    for ilib in range(3):
        print("*** A2E_LIB    LIB[%d] ~   FREQ[%3d]" % (ilib, IFREQ[ilib]))
        ABS[:, ilib] = ABSORBED[:, IFREQ[ilib]]

    # While we are creating the tree structure, we can work with the ABS[] array that only has LFREQ values
    # find AMIN and BINS for each bin
    t0  = time.time()
    m   =  nonzero((ABS[:,0]>1.0e-32)&(ABS[:,1]>1.0e-32)&(ABS[:,2]>1.0e-32))
    print("Positive absorptions: %d cells" % len(m[0]))

    if (0):
        AMIN0, AMAX0  =  0.9999*min(ABS[m[0], 0]), 1.0001*max(ABS[m[0], 0])
    else:
        # AMIN0, AMAX0  =  percentile(ABS[m[0], 0], (0.002, 99.998))
        AMIN0, AMAX0  =  percentile(ABS[m[0], 0], (0.001, 99.999))
    K0            =  exp(log(AMAX0/AMIN0)/BINS0)
    K             =  zeros(BINS0*BINS1)  # K values for BINS0*BINS1 nodes on levels 1 and 2
    print("AMIN0 %.3e  K0 %.3e, ABS RANGE %12.4e ... %12.4e" % (AMIN0, K0, AMIN0, AMAX0))
    
    
    if (K0>1.3):        
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!         K0 = %.2f  > 1.3   ---- ARE YOU SURE ???     !!!!!" % K0)
        if (1):
            sys.stdout.write("WE CHANGE AMIN FROM %.3e" % AMIN0)  # we might have some zeros??
            #  1.3**50 ~ 1e6 in dynamical range, 1.2**53 ~ 1e4 
            AMIN0 = AMAX0 / 1.2**BINS0    # ... practically zero ?
            K0    =  exp(log(AMAX0/AMIN0)/BINS0)            
            sys.stdout.write(" TO %.3e, AMAX=%.3e, K0=%.3e\n" % (AMIN0, AMAX0, K0))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        time.sleep(2)
        if (K0>1.4):
            print("*** ABORT !!! ****")
            time.sleep(5)
            sys.exit()
        
    AMIN1, K1     =  zeros(BINS0, float32), zeros(BINS0, float32)
    AMIN2, K2     =  zeros(BINS0*BINS1, float32), zeros(BINS0*BINS1, float32)    
    MASK00        =  zeros(CELLS, int32)
    MASK00[m]     =  1          # positive absorptions in all reference channels
    
    for i0 in range(BINS0):                                                     # loop over level 0 bins
        a, b      =  AMIN0*K0**i0,  AMIN0*K0**(i0+1.0)                          # limits of one level 0 bin
        MASK0     =  MASK00.copy()                                              # valid cells
        m         =  nonzero((ABS[:,0]<a)|(ABS[:,0]>b))                         # not level 0 bin
        MASK0[m]  =  0                                                          # leave cells matching level 0 bin
        m         =  nonzero(MASK0>0)                                           # valid cells in level 0 bin
        #print("i0 %3d/%3d     %.3e - %.3e   %6d cells" % (i0, BINS0, a, b, len(m[0])))
        if (len(m[0])<1):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!! CANNOT HAVE EMPTY LEVEL 0 BIN ..... ABORT !!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            time.sleep(5)
            sys.exit()
            
        # binning on level 1 (bin i0 -> one level 1 vector)
        if (1):
            amin1, amax1  =  min(ABS[m[0], 1]), max(ABS[m[0], 1])
        else:
            amin1, amax1  =  percentile(ABS[m[0], 1], (0.01, 99.99))
        amin1, amax1      =  0.9999*amin1, 1.0001*amax1                         # must be different numbers
        k1                =  exp(log(amax1/amin1)/BINS1)                        #  -> AMIN and K define the vector
        AMIN1[i0]         =  amin1                                              # i0 points to one level 1 vector
        K1[i0]            =  k1
        #print("  AMIN1 %.3e  K0 %.3e" % (amin1, k1))
    
        # if level 1 vector has no data, widen the binning
        MASK1             =  MASK0.copy()
        MASK1[(ABS[:,1]<amin1)|(ABS[:,1]>amax1)] = 0
        m1                =  nonzero(MASK1>0)                                   # cells in i0 -> level 1 vector
        if (len(m1[0])<1):  # expand the range for level 1 vector
            m             =  nonzero(MASK0>0)
            if (1):
                amin1, amax1  =  min(ABS[m[0], 1]), max(ABS[m[0], 1])
            else:
                amin1, amax1  =  percentile(ABS[m[0], 1](0.01, 99.99))
            k1            =  exp(log(amax1/amin1)/BINS1)                        #  -> AMIN and K define the vector
            AMIN1[i0]     =  amin1                                              # i0 points to one level 1 vector
            K1[i0]        =  k1
            #print("  ----------------->  AMIN1 %.3e  K0 %.3e" % (amin1, k1))
            MASK1         =  MASK0.copy()
            MASK1[(ABS[:,1]<amin1)|(ABS[:,1]>amax1)] = 0
            m             =  nonzero(MASK1>0)
            if (len(m[0])<1):  # if we still have no data for level 1 vector corresponding to current i0 
                print("CANNOT HAVE LEVEL1 VECTOR WITH NO DATA !!!")
                sys.exit()
                    
        for i1 in range(BINS1):                                                 # loop over level 1 bins
            a, b          =  amin1*k1**i1, amin1*k1**(i1+1.0)                   # limits of one cell on level 1
            MASK1         =  MASK0.copy()                                       # level 0 bin
            MASK1[(ABS[:,1]<a)|(ABS[:,1]>b)] = 0                                # level 0 and level 1 bin
            m             =  nonzero(MASK1>0)                                   # match level 0 and level 1 bins
            #print("    i1   %.3e - %.3e   %6d cells" % (a, b, len(m[0])))
            i1_bin_empty = False
            if (len(m[0])<1):
                ## print("i1=%d bin is empty --- empty bins allowed only on the lowest level !!!" % i1)
                i1_bin_empty = True
                ## sys.exit()
                
                
            # binning on level 2
            if (i1_bin_empty):
                # tree has a dead branch -- all i2 entries below this i1 bin remain zero !!
                AMIN2[i0*BINS1+i1] =  -1.0
                K2[i0*BINS1+i1]    =  -1.0          
            else:
                if (1):
                    amin2, amax2   =  min(ABS[m[0], 2]), max(ABS[m[0], 2])
                else:
                    amin2, amax2   =  percentile(ABS[m[0], 2], (0.03, 99.97))
                amin2, amax2       =  0.9999*amin2, 1.0001*amax2                     # must be different => k2 > 1.0
                k2                 =  exp(log(amax2/amin2)/BINS2)        
                AMIN2[i0*BINS1+i1] =  amin2
                K2[i0*BINS1+i1]    =  k2
                #print("    AMIN2 %.3e  K0 %.3e" % (amin2, k2))
                MASK2              =  MASK1.copy()
                MASK2[(ABS[:,2]<amin2)|(ABS[:,2]>amax2)] = 0
                m                  =  nonzero(MASK2>0)
                if (len(m[0])<1):
                    print("*** WIDEN THE RANGE FOR LEVEL 2 VECTOR ***")
                    amin2, amax2       =  min(ABS[m[0], 2]), max(ABS[m[0], 2])
                    ### amin2, amax2   =  percentile(ABS[m[0], 2], (0.05, 99.95))
                    amin2, amax2       =  0.9999*amin2, 1.0001*amax2
                    k2                 =  exp(log(amax2/amin2)/BINS2)        
                    AMIN2[i0*BINS1+i1] =  amin2
                    K2[i0*BINS1+i1]    =  k2
                    #print("    -----> AMIN2 %.3e  K0 %.3e" % (amin2, k2))
                    #
                    MASK2             =  MASK1.copy()
                    MASK2[(ABS[:,2]<amin2)|(ABS[:,2]>amax2)] = 0
                    m                 =  nonzero(MASK2>0)    # cells in the final (i0,i1,i2) bin
                    if (len(m[0])<1):
                        print("ONE LEVEL 2 VECTOR COMPLETELY EMPTY")
                        sys.exit()
    ######################################################################
    print("!!! CREATE LIBRARY STRUCTURE: %.2f SECONDS" % (time.time()-t0))
    
    
    # tree is now defined by
    #     AMIN0, K0,    AMIN1, K1,     AMIN2, K2
    #     bin (i0,i1,i2) corresponds to
    #           IFREQ[0]     k=K, a=AMIN0,                          [a*k**i0, a*k**(i0+1)]
    #           IFREQ[1]     k=K[i0*BINS1+ 0], a=AMIN[i0*BINS1+ 0], [a*k**i1, a*k**(i1+1)]
    #           IFREQ[2]     k=K[i0*BINS1+i1], a=AMIN[i0*BINS1+i1], [a*k**i1, a*k**(i1+1)]
    #
    #    the absorption and emission vectors can be found in arrays
    #    [i0*BINS1*BINS2+i1*BINS2+i2]
    
    # Use OpenCL kernel to add input absorption vectors to the right place in the array
    platform, device, context, queue,  mf = InitCL(GPU)
    
    BBB       =  BINS0*BINS1*BINS2
    LOCAL     =  4
    GLOBAL    =  (1+BBB//LOCAL)*LOCAL   # need one work item per leaf !!
    GPU       =  0
    
    AMIN1_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=AMIN1)
    K1_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K1)
    AMIN2_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=AMIN2)
    K2_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K2)
    ADD_buf   =  cl.Buffer(context, mf.READ_ONLY,   4*BBB*NFREQ)
    ALL_buf   =  cl.Buffer(context, mf.READ_ONLY,   4*BBB*NFREQ)
    NUM_buf   =  cl.Buffer(context, mf.READ_WRITE,  4*BBB)
    NUMO_buf  =  cl.Buffer(context, mf.READ_WRITE,  4*BBB)
    
    ALL       =  zeros((BBB, NFREQ), float32)
    NUM       =  zeros(BBB, int32)
    cl.enqueue_copy(queue, ALL_buf,  ALL)
    cl.enqueue_copy(queue, NUM_buf,  NUM)
    
    OPT       =  " -D BINS0=%d -D BINS1=%d -D BINS2=%d -D NFREQ=%d -D IFREQ0=%d -D IFREQ1=%d -D IFREQ2=%d" % \
    (BINS0, BINS1, BINS2, NFREQ, IFREQ[0], IFREQ[1], IFREQ[2])
    source    =  open(INSTALL_DIR+"/kernel_tree3.c").read()
    program   =  cl.Program(context, source).build(OPT)
    Populate  =  program.Populate
    Populate.set_scalar_arg_dtypes([np.int32, np.float32, np.float32, None, None, None, None, None, None, None ])
    

    t0 = time.time()
    for ibatch in range(CELLS//BBB+1):
        a, b = ibatch*BBB, (ibatch+1)*BBB       # ADD is allocated for [GLOBAL, NFREQ]
        b    = min([CELLS, b])   
        if (ibatch%100==0): print("A2E_LIB.py, populate tree,  ibatch %d =>  %d:%d" % (ibatch, a, b))
        cl.enqueue_copy(queue, ADD_buf, ABSORBED[a:b,:])  # the full absorption vector
        Populate(queue, [GLOBAL,], [LOCAL,], 
        b-a, AMIN0, K0, AMIN1_buf, K1_buf, AMIN2_buf, K2_buf, ADD_buf, NUM_buf, ALL_buf)
    print("!!! POPULATE: %.2f SECONDS" % (time.time()-t0))
    t0 = time.time()
    
    print("????? Before interpolation:")
    cl.enqueue_copy(queue, NUM, NUM_buf)
    cl.enqueue_copy(queue, ALL, ALL_buf)
    m = nonzero(NUM<=0) 
    print("?????    NUM HAS %d ZEROS" % (len(m[0])))
    ALL.shape = (BBB, NFREQ)
    m = nonzero(ALL[:,IFREQ[1]]<=0.0)
    print("?????    ALL HAS %d ZEROS" % (len(m[0])))
    
    # interpolate before normalising with the number of input vectors added per bin
    print("=== A2E_LIB INTERPOLATE ===")
    cl.enqueue_copy(queue, NUMO_buf, NUM)    
    Interpolate   = program.Interpolate
    Interpolate.set_scalar_arg_dtypes([np.float32, np.float32, None, None, None, None, None, None, None])
    Interpolate(queue, [GLOBAL,], [LOCAL,], AMIN0, K0, AMIN1_buf, K1_buf, AMIN2_buf, K2_buf, \
    NUMO_buf, NUM_buf, ALL_buf)
    cl.enqueue_copy(queue, NUM, NUM_buf)
    bad =  len(nonzero(NUM==0)[0])
    print("\n!!! INTERPOLATE: %.2f SECONDS, %d ZERO NUM VALUES" % (time.time()-t0, bad))
    t0 = time.time()
    
    # another kernel to just do the division ALL/NUM
    print("=== A2E_LIB DIVIDE ===")
    Divide   = program.Divide
    Divide.set_scalar_arg_dtypes([None, None])
    Divide(queue, [GLOBAL,], [LOCAL,], ALL_buf, NUM_buf)
    print("!!! DIVIDE: %.2f SECONDS" % (time.time()-t0))
    t0 = time.time()

    # Final effort to fill in missing data below some i1 
    print("=== A2E_LIB FILL ===")
    cl.enqueue_copy(queue, NUM,      NUM_buf)    
    cl.enqueue_copy(queue, NUMO_buf, NUM)    
    Fill   = program.Fill
    Fill.set_scalar_arg_dtypes([None, None, None, None])
    Fill(queue, [GLOBAL,], [LOCAL,], ALL_buf, NUMO_buf, NUM_buf, K1_buf)
    cl.enqueue_copy(queue, NUM, NUM_buf)
    bad =  len(nonzero(NUM==0)[0])
    print("!!! FILL: %.2f SECONDS, %d ZERO NUM VALUES" % (time.time()-t0, bad))
    t0 = time.time()

    cl.enqueue_copy(queue, ALL, ALL_buf)
    queue.finish()
    ALL.shape  = (BBB, NFREQ)
    m          = nonzero(sum(ALL, axis=1)<=0.0)
    bad        = len(nonzero(NUM==0)[0])
    print("????? ALL HAS %d ZERO VECTORS, NUM HAS %d ZEROS" % (len(m[0]), bad))
    if (bad>0): sys.exit()
    
    # read the absorption cube back
    cl.enqueue_copy(queue, NUM, NUM_buf)
    cl.enqueue_copy(queue, ALL, ALL_buf)
    
    
    # solve emission
    fp = file('tmp.absorbed', 'wb')
    asarray([BBB, NFREQ], int32).tofile(fp)
    asarray(ALL, float32).tofile(fp)
    fp.close()
    print("==========================================================================================")
    print("  In A2E_LIB.py  ", sys.argv[1:])
    print("  create_library_2 => ")
    print("  A2E.py  %s tmp.absorbed tmp.emitted" % solver)
    print("==========================================================================================")
    os.system("A2E.py  %s tmp.absorbed tmp.emitted" % solver)
    ALL = fromfile('tmp.emitted', float32)[2:].reshape(BBB, NFREQ)
    print("!!! A2E: %.2f SECONDS" % (time.time()-t0))
    t0 = time.time()
    
    # save library to file
    fp = file(libname, 'wb')
    asarray([BINS0, BINS1, BINS2, NFREQ, NLFREQ], int32).tofile(fp)
    asarray(FREQ, float32).tofile(fp)
    asarray(LFREQ, float32).tofile(fp)
    asarray([AMIN0, K0], float32).tofile(fp)
    asarray(AMIN1, float32).tofile(fp)
    asarray(K1, float32).tofile(fp)
    asarray(AMIN2, float32).tofile(fp)
    asarray(K2, float32).tofile(fp)
    asarray(ALL, float32).tofile(fp)   # now the emissions for the tree leaf bins
    fp.close()
    print("=== create_library_2 .... call FINISHED")
        

    
    
        
def solve_with_library_2(libname, freq, lfreq, file_absorbed, file_emitted, GPU=0, f_ofreq=""):
    """
    Calculate emissions to file_emitted based on the absorptions in file_absorbed and the
    provided library file libname. Library created with create_library_2), with fixed number of
    bins for each reference frequency
    Input:
        libname       =  name of the library file
        freq          =  full vector of frequencies (as in file_emitted)
        lfreq         =  vector of input frequencies used in file_absorbed
        file_absorbed =  absorption file written by SOC
        file_emitted  =  file of emissions for lfreq frequencies
        f_ofreq       =  if given, is a file listing subset of freq frequencies and only those
                         will be written to the output file
    """
    # load library from file
    print("=== SOLUTION===  .... solve_with_library_2,  f_ofreq=%s" % f_ofreq)
    t0 = time.time()

    # read the library from file
    fp           =  open(libname, 'rb')
    BINS0, BINS1, BINS2, nfreq, nlfreq = fromfile(fp, int32, 5)
    BBB          =  BINS0*BINS1*BINS2
    FREQ         =  fromfile(fp, float32, nfreq)
    LFREQ        =  fromfile(fp, float32, nlfreq)
    NFREQ        =  len(FREQ)
    NLFREQ       =  len(LFREQ)        
    OFREQ        =  []
    if ((len(freq)!=NFREQ)|(len(lfreq)!=NLFREQ)):
        print("*** solve_with_library_2: inconsistent number of frequencies")
        print("*** arguments nfreq=%d, nlfreq=%d, library file with nfreq=%d, nlfreq=%d" % (len(freq), len(lfreq), NFREQ, NLFREQ))
        sys.exit()
    if (max(abs((freq-FREQ)/FREQ))>0.001):
        print("*** solve_with_library_2: inconsistent full frequency vectors"), sys.exit(0)
    if (max(abs((lfreq-LFREQ)/LFREQ))>0.001):
        print("*** solve_with_library_2: inconsistent reference frequencies"), sys.exit(0)
    AMIN0, K0 = fromfile(fp, float32, 2)
    AMIN1     = fromfile(fp, float32, BINS0)
    K1        = fromfile(fp, float32, BINS0)
    AMIN2     = fromfile(fp, float32, BINS0*BINS1)
    K2        = fromfile(fp, float32, BINS0*BINS1)
    ALL       = fromfile(fp, float32, BBB*NFREQ)
    fp.close()

    # Find the indices for the reference frequencies -- before FREQ is potentially change by ofreq below
    print("=== LIBRARY FREQUENCIES (in solve_with_library_2)") 
    IFREQ        =  zeros(NLFREQ, int32)
    for ilib in range(NLFREQ):
        IFREQ[ilib] = argmin(abs(FREQ-LFREQ[ilib]))
        print("   %8.3f -> %8.3f um, IFREQ = %3d" % (f2um(LFREQ[ilib]), f2um(FREQ[IFREQ[ilib]]), IFREQ[ilib]))
    
    
    if (len(f_ofreq)>1):
        # If we select a subset of output frequencies, that can be done immediately here by modifying the
        # library, its ALL array   ALL[BBB, nfreq]  ->  ALL[BBB, ofreq]
        ALL.shape = (BBB, NFREQ)
        OFREQ = loadtxt(f_ofreq)
        tmp   =  zeros((BBB, len(OFREQ)), float32)
        for i in range(len(OFREQ)):
            ifreq = argmin(abs(freq-OFREQ[i]))
            tmp[:, i] = ALL[:, ifreq]
        ALL    =  None
        ALL    =  ravel(tmp)
        FREQ   =  OFREQ
        nfreq  =  len(OFREQ)
        NFREQ  =  len(OFREQ)
        
        
    CELLS, NAFREQ =  fromfile(file_absorbed, int32, 2)
    # NAFREQ must be either full set of frequencies, NAFREQ == NFREQ  or just reference frequencies, NAFREQ == NLFREQ
    if (NAFREQ!=NFREQ):
        print("*** solve_with_library_2: library built for %d frequencies, absorption file has %d" % (NFREQ, NAFREQ))
        print("    (if using ofreq.dat, nfreq=%d is number of output frequencies, not the full set of original freq.)" % NFREQ)
        if (len(OFREQ)>0):
            print("    .... possibly ok since one is using OFREQ to limit output frequencies")
        else:
            sys.exit()    
    ABS          =  fromfile(file_absorbed, float32)[2:].reshape(CELLS, NAFREQ)
    # NFREQ output frequencies in the loibrary, NAFREQ frequencies in the input absorption file
    # and  NLFREQ reference frequencies in the library
    
    platform, device, context, queue,  mf = InitCL(GPU)
    GLOBAL       =  32768
    LOCAL        =  4
    OPT          =  " -D BINS0=%d -D BINS1=%d -D BINS2=%d -D NFREQ=%d -D IFREQ0=%d -D IFREQ1=%d -D IFREQ2=%d" % \
    (BINS0, BINS1, BINS2, NFREQ, IFREQ[0], IFREQ[1], IFREQ[2])
    source       =  open(INSTALL_DIR+"/kernel_tree3.c").read()
    program      =  cl.Program(context, source).build(OPT)
    
    AMIN1_buf2   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=AMIN1)
    K1_buf2      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K1)
    AMIN2_buf2   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=AMIN2)
    K2_buf2      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K2)
    ALL_buf2     =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ALL)
    
    # Solve with library --- BATCH cells at a time
    BATCH         =  2000
    ABS_buf       =  cl.Buffer(context, mf.READ_ONLY,  4*BATCH*NLFREQ)  # NLFREQ absorption values as input
    EMIT_buf      =  cl.Buffer(context, mf.WRITE_ONLY, 4*BATCH*NFREQ)   # NFREQ emission values as output
    EMIT          =  zeros((CELLS, NFREQ), float32)   # !!! BIG ARRAY !!!
    
    Lookup        =  program.Lookup
    Lookup.set_scalar_arg_dtypes([np.int32, np.float32, np.float32, None, None, None, None, None, None, None])

    
    # Put to ABS only the data of the reference frequencies (file could have the full NFREQ)
    print("=== solve_with_library2....")
    print("    ABS.shape ", ABS.shape, " NFREQ = %d" % NFREQ, "  NLFREQ = %d" % NLFREQ)
    if (ABS.shape[1]>NLFREQ):    # we have absoprtions for more than just the reference frequencies
        print("    INPUT HAS ALL NAFREQ=%d FREQUENCIES" % NAFREQ)
        ABS = asarray(ABS[:, IFREQ], float32).copy()  # IFREQ == index to full original NFREQ
    elif (ABS.shape[1]==NLFREQ):
        print("    INPUT HAS NLFREQ=%d REFERENCE FREQUENCIES" % NLFREQ)
        ABS = asarray(ABS, float32)   # already just the reference frequencies
    else:
        print("??? absorption file has < nlfreq frequencies ???"), sys.exit()
        
        
    # Solve one batch at a time
    print("=== A2E_LIB.py  =>  solve_with_library2.... loop over BATCHES")
    processed = 0
    for i in range(CELLS//BATCH+1):
        a, b      =  i*BATCH, (i+1)*BATCH
        b         =  min([b, CELLS])
        tmp       =  ABS[a:b,:].copy()
        cl.enqueue_copy(queue, ABS_buf, tmp)
        Lookup(queue, [GLOBAL,], [LOCAL,], b-a, AMIN0, K0, AMIN1_buf2, K1_buf2, AMIN2_buf2, K2_buf2, ALL_buf2,
        ABS_buf, EMIT_buf)
        cl.enqueue_copy(queue, EMIT[a:b,:], EMIT_buf)
        processed += (b-a)
    print("USELIB solution: %.2f seconds ... processed %d cells, CELLS=%d" % (time.time()-t0, processed, CELLS))

    # Save emission
    fp = file(file_emitted, 'wb')
    asarray([CELLS, NFREQ], int32).tofile(fp)
    EMIT.tofile(fp)
    fp.close()
    print("######### WROTE EMIT CELLS=%d, NFREQ=%d, EMIT=" % (CELLS, NFREQ), EMIT.shape, " #######")

    if (0):
        # Compare library solution to full reference solution in /dev/shm/emitted.data.ref
        clf()
        figure(1, figsize=(8,6))
        rc('font', size=8)
        subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.93, wspace=0.25, hspace=0.25)
    
        i100  =  argmin(abs(FREQ-um2f(100.0)))
        i500  =  argmin(abs(FREQ-um2f(500.0)))
        print("100um -> ifreq=%d" % i100)
        REF   =  fromfile(SHAREDIR+'/emitted.data.ref', float32)[2:].reshape(CELLS, NFREQ) # full cell-by-cell solution
        m     =  nonzero((REF[:,i100]>0.0)&(REF[:,i500]>0.0))  ###  &(ABS[:,1]>0.0))
        REF   =  REF[m[0],:]
        EMIT  =  EMIT[m[0],:]
        
        subplot(231)
        imshow(EMIT, aspect='auto')
        title("LIB")
        colorbar()
        
        subplot(232)
        imshow(REF, aspect='auto')
        title("REF")
        colorbar()

        subplot(233)
        RERR  =   100.0*(EMIT-REF)/REF
        imshow(RERR, aspect='auto', vmin=-100.0, vmax=+100.0)
        title('RERR')
        colorbar()
        
        
        subplot(223)
        if (1):
            plot(REF[:, i100], EMIT[:,i100]/REF[:, i100], 'bx')
            plot(REF[:, i500], EMIT[:,i500]/REF[:, i500], 'r+')            
        else:
            plot(REF[:, i100], EMIT[:,i100], 'bx')
            plot(REF[:, i500], EMIT[:,i500], 'r+')

            
            
        subplot(224)
        II  =  asarray(linspace(1, REF.shape[0]-1, 6), int32)
        UM  =  f2um(FREQ)
        for i in range(len(II)):
            c  = [ 'k', 'b', 'g', 'r', 'c', 'm',  'y'][i%6]
            loglog(UM,  REF[II[i],:],  'x'+c)
            loglog(UM, EMIT[II[i],:],  '+'+c)
    
        ylim(1e-17, 0.02)
        xlim(10, 3000)
        
        savefig(libname+'.png')
        # show()

    
    
    

if (0): 
    # OLD TREE
    if (MAKELIB):
        # library is created based on the representative absorption vectors gathered in the tree structure
        # => A2E.py is used only to solve those => once the library exists, we still need to call solve_with_library
        create_library(SOLVER, LIBNAME, FREQ, LFREQ, AFILE, K=1.1)
    solve_with_library(LIBNAME, FREQ, LFREQ, sys.argv[5], sys.argv[6])
else:
    # New one with constant number of bins on each level
    ##### LIB_BINS = [ 45, 34, 15 ]
    ## BINS = [ 30, 34, 15 ]
    if (MAKELIB):
        print("=== A2E_LIB.py  =>  create_library_2  ", SOLVER)
        create_library_2(SOLVER, LIBNAME, FREQ, LFREQ, AFILE, BINS=LIB_BINS, GPU=0)
    print("=== A2E_LIB.py  =>  solve_with_library2.... ", SOLVER)
    print("=== A2E_LIB.py ... F_OFREQ ", F_OFREQ, "   SOLVER ", SOLVER)
    solve_with_library_2(LIBNAME, FREQ, LFREQ, AFILE, EFILE, GPU=0, f_ofreq=F_OFREQ)
    print("=== A2E_LIB.py  =>  solve_with_library2.... done ", SOLVER)
    
    
    
