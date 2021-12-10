from MJ.mjDefs import *
from MJ.Aux.MBB_MCMC import InitCL
import pyopencl as cl

CELLS      =  10000
ABS        =  randn(CELLS, 3)
ABS[:,0]  +=  linspace(10.0,  20.0, CELLS)
ABS[:,1]  +=  linspace(50.0,  60.0, CELLS)
ABS[:,2]  +=  linspace(90.0, 100.0, CELLS)

EMIT  =  sum(ABS, axis=1)
K     =  1.03
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
AMIN[0], AMAX[0]  =  0.9999*min(ABS[:,0]), 1.0001*max(ABS[:,0])
BINS[0]           =  1+int(1+log(AMAX[0]/AMIN[0]) / log(K))
VEC               =  [ [], [], [] ]
AMI               =  [ [], [], [] ]
NODES             =  [ 1, 0, 0 ]
VEC[0].append(arange(BINS[0]))  # VEC[0] contains node numbers of i1 vectors
AMI[0].append(AMIN[0])

for i0 in range(BINS[0]):        
    amin, amax       =  AMIN[0]*K**i0,  AMIN[0]*K**(i0+1)   # limits of a single i0 bin
    MASK0            =  ones(CELLS, int32)
    MASK0[nonzero((ABS[:,0]<amin)|(ABS[:,0]>amax))] = 0
    m                =  nonzero(MASK0>0)                    # cells in single i0 bin
    # print("i0=%2d -> %d" % (i0, len(m[0])))
    if (len(m[0])<1):
        AMIN[1], BINS[1] = -1.0, 0
    else:
        AMIN[1], AMAX[1] =  0.9999*min(ABS[:,1][m]), 1.0001*max(ABS[:,1][m])
        BINS[1]          =  int(2+log(AMAX[1]/AMIN[1]) / log(K))
        if (AMIN[1]*K**(BINS[1]-1)<AMAX[1]):
            print("1: data %.3e %.3e    range %.3e %.3e" % (AMIN[1], AMAX[1], AMIN[1], AMIN[1]*K**(BINS[1]-1)))
            sys.exit()
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
        ## print("  i1=%2d ->  %d" % (i1, len(m[0])))        
        if (len(m[0])<1):
            AMIN[2], BINS[2] = -1, 0
        else:
            AMIN[2], AMAX[2]  =  0.9999*min(ABS[:,2][m]), 1.0001*max(ABS[:,2][m])   # limits for i2 vector
            BINS[2]           =  1+int(1+log(AMAX[2]/AMIN[2]) / log(K))
            if (AMIN[2]*K**(BINS[2]-1)<AMAX[2]):
                print("2: data %.3e %.3e    range %.3e %.3e" % (AMIN[2], AMAX[2], AMIN[2], AMIN[2]*K**(BINS[2]-1)))
                sys.exit()
        VEC[2].append( zeros(BINS[2], float32) )                      # one i1 element -> new i2 vector
        AMI[2].append( AMIN[2] )
        
        for i2 in range(BINS[2]):                                     # set values in each i2 bin
            amin, amax        =  AMIN[2]*K**i2,  AMIN[2]*K**(i2+1)    # limits of a single i2 bin
            MASK2             =  MASK1.copy()                         # mask for parent i0, i1 bins
            MASK2[nonzero((ABS[:,2]<amin)|(ABS[:,2]>amax))] = 0       # mask for current i0, i1, i2 bin
            m                 =  nonzero(MASK2>0)
            ## print("    i2 %d" % len(m[0]))
            if (len(m[0])<1):
                res           = -1.0
            else:
                res           =  mean(EMIT[m])
            VEC[2][-1][i2]    =  res                                  # set value in the leaf

            

#print(VEC[0])
#print(VEC[1])
#print(VEC[2])

# print(len(VEC[0]))
# print(len(VEC[1]))
# print(len(VEC[2]))


count = 0
for i0 in range(len(VEC[0][0])):
    amin0, amax0 = AMI[0][0]*K**i0, AMI[0][0]*K**(i0+1)
    v1 = VEC[1][VEC[0][0][i0]]
    a1 = AMI[1][VEC[0][0][i0]]
    for i1 in range(len(v1)):
        amin1, amax1 = a1*K**(i1), a1*K**(i1+1)
        v2 = VEC[2][v1[i1]]
        a2 = AMI[2][v1[i1]]
        for i2 in range(len(v2)):
            amin2, amax2 = a2*K**(i2), a2*K**(i2+1)
            if (0):
                print("%2d [%6.2f,%6.2f]  %2d [%6.2f,%6.2f]  %2d [%6.2f, %6.2f] -> %7.2f" %
                (i0, amin0, amax0,  i1, amin1, amax1,  i2, amin2, amax2, v2[i2]))
            count += 1
print("Number of leafs: %d" % count)            
        

def GetValue(VEC, AMI, x):
    V0   =  VEC[0][0]                          # the only level 0 vec
    a0   =  AMI[0][0]                          # its amin
    i0   =  int(floor(log(x[0]/a0)/log(K)))    # element in i0 vector
    v1   =  V0[i0]                             # number of level 1 vec
    V1   =  VEC[1][v1]                         # level 1 vector itself
    a1   =  AMI[1][v1]                         # its amin    
    i1   =  int(floor(log(x[1]/a1)/log(K)))    # element in i1 vecto
    v2   =  V1[i1]                             # number of level 2 vec
    V2   =  VEC[2][v2]                         # level 2 vector itself
    a2   =  AMI[2][v2]    
    i2   =  int(floor(log(x[2]/a2)/log(K)))    # element in i2 vec
    val  =  V2[i2]                             # final value
    if (0):
        print("--------------------------------------------------------------------------------")
        print("VEC %2d %2d %2d" % (i0, i1, i2))    
        amin, amax = a0*K**i0,  a0*K**(1+i0)
        print("  x[0] = %6.3f   AMI[0][ 0] = %6.3f   [%6.3f, %6.3f]" % (x[0], a0, amin, amax))
        amin, amax = a1*K**i1,  a1*K**(1+i1)
        print("  x[1] = %6.3f   AMI[1][%2d] = %6.3f   [%6.3f, %6.3f]" % (x[1], i1, a1, amin, amax))
        amin, amax = a2*K**i2,  a2*K**(1+i2)
        print("  x[2] = %6.3f   AMI[2][%2d] = %6.3f   [%6.3f, %6.3f]" % (x[2], i2, a2, amin, amax))    
        print("   %12.4e   ->  %12.4e" % (sum(x),  val ))        
    return val
    
    
if (0):
    for i in [0,1000, 1990]:
        x    =  ABS[i,:]
        val  =  GetValue(VEC, AMI, ABS[i,:])
        print("%7.3f  ->  %7.3f" % (sum(x), val))


y = zeros(CELLS, float32)    
for i in range(CELLS):
    y[i] = GetValue(VEC, AMI, ABS[i,:])
    
plot(sum(ABS, axis=1), y, 'bx')


######################################################



ABS  +=  0.4*randn(CELLS, 3)



platform, device, context, queue,  mf = InitCL(GPU=0)

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
V2    =  zeros(NE2, float32)   # EVENTUALLY VECTORS; IN THIS TEST A SCALAR FLOAT PER LEAF
for j in range(len(VEC[2])):
    V2[I2[j]:(I2[j]+N2[j])] = VEC[2][j]
print("NE2=%d" % NE2)


if (0):
    # Dump the library to a file
    fp = open('test.lib', 'wb')
    asarray([NE0, NV1, NV2, NE1, NE2], int32).tofile(fp)
    asarray([A0, ], float32).tofile(fp)
    asarray(V0, int32).tofile(fp)
    asarray(N1, int32).tofile(fp)
    asarray(I1, int32).tofile(fp)
    asarray(A1, float32).tofile(fp)
    asarray(V1, int32).tofile(fp)
    asarray(N2, int32).tofile(fp)
    asarray(I2, int32).tofile(fp)
    asarray(A2, float32).tofile(fp)
    asarray(V2, float32).tofile(fp)
    fp.close()
            
    
    fp = open('test.lib', 'rb')
    NE0, NV1, NV2, NE1, NE2  =  fromfile(fp, int32, 5)
    A0 = fromfile(fp, float32, 1)[0]   #   amin of level 0
    V0 = fromfile(fp, float32,  NE0)   #   vector of level 0
    N1 = fromfile(fp, int32,    NV1)   #   subvectors on level 1
    I1 = fromfile(fp, int32,    NV1)   #   start indices of level 1 subvectors
    A1 = fromfile(fp, float32,  NV1)   #   amin for level 1 subvectors
    V1 = fromfile(fp, int32,    NE1)   #   vector of level 1
    N2 = fromfile(fp, int32,    NV2)   #   number of elements in each level 2 subvector
    I2 = fromfile(fp, int32,    NV2)   #   start indices to level 2 subvectors
    A2 = fromfile(fp, int32,    NV2)   #   amin for level 2 subvectors
    V2 = fromfile(fp, int32,    NE2)   #   vector of level 2 values  ... eventually a matrix
    fp.close()


abs_buf =  cl.Buffer(context, mf.READ_ONLY,   4*3*CELLS)
res_buf =  cl.Buffer(context, mf.WRITE_ONLY,  4*CELLS)
V0_buf  =  cl.Buffer(context, mf.READ_ONLY, 4*NE1)
N1_buf  =  cl.Buffer(context, mf.READ_ONLY, 4*NV1)
I1_buf  =  cl.Buffer(context, mf.READ_ONLY, 4*NV1)
A1_buf  =  cl.Buffer(context, mf.READ_ONLY, 4*NV1)
V1_buf  =  cl.Buffer(context, mf.READ_ONLY, 4*NE1)
N2_buf  =  cl.Buffer(context, mf.READ_ONLY, 4*NV2)
I2_buf  =  cl.Buffer(context, mf.READ_ONLY, 4*NV2)
A2_buf  =  cl.Buffer(context, mf.READ_ONLY, 4*NV2)
V2_buf  =  cl.Buffer(context, mf.READ_ONLY, 4*NE2)

cl.enqueue_copy(queue, abs_buf, asarray(ABS, float32))
cl.enqueue_copy(queue, V0_buf, asarray(V0, int32))
cl.enqueue_copy(queue, N1_buf, N1)
cl.enqueue_copy(queue, I1_buf, I1)
cl.enqueue_copy(queue, A1_buf, A1)
cl.enqueue_copy(queue, V1_buf, V1)
cl.enqueue_copy(queue, N2_buf, N2)
cl.enqueue_copy(queue, I2_buf, I2)
cl.enqueue_copy(queue, A2_buf, A2)
cl.enqueue_copy(queue, V2_buf, V2)

GLOBAL   =  ((ABS.shape[0]//32)+1)*32
LOCAL    =  4

OPT      =  "-D K=%.3ef" % K
source   =  file("kernel_tree_lookup.c").read()
program  =  cl.Program(context, source).build(OPT)    
Lookup   =  program.Lookup
#                              cells     abs[]   res[]
Lookup.set_scalar_arg_dtypes([ np.int32, None,   None,
    #  NE0    A0          V0       NV1       N1      I1      A1      V1      NV2       N2      I2      A2      V2   
    np.int32, np.float32, None,    np.int32, None,   None,   None,   None,   np.int32, None,   None,   None,   None])
    
Lookup(queue, [GLOBAL,], [LOCAL,],  CELLS, abs_buf, res_buf,
    NE0,      A0,         V0_buf,  NV1,      N1_buf, I1_buf, A1_buf, V1_buf, NV2,      N2_buf, I2_buf, A2_buf, V2_buf)
    
RES = zeros(CELLS, float32)
cl.enqueue_copy(queue, RES, res_buf)
    
plot(sum(ABS, axis=1), RES, 'r+')

m = nonzero(RES<=0.0)
print("MISSING: %d\n" % len(m[0]))

SHOW()

    
    
    
    
    
