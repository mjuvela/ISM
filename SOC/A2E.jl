#!/usr/bin/julia

# 2019-04-24  julia translation of A2E.py
using OpenCL, Printf, Mmap #, PyPlot
using DelimitedFiles, Statistics

ICELL   =  0
GPU     =  true
NSTOCH  =  999
NB      =  [ 43, 23, 13]     # bins on each level of the library tree
NB      =  [ 67, 32, 15]

BATCH   =  Int32(prod(NB))   # tree => all in a single call, constraint on BATCH
FACTOR  =  Float32(1.0e20)
DFRAC   =  0.0               # fraction left to be solved directly
USELIB  =  false
INTERP  =  false             # interpolation in the tree
USEHASH =  false             # use hash instead of tree - BACTH can be arbitrary !!
WITH_X  =  false

if (USEHASH)
  BATCH = 4096
end

if (length(ARGS)<3)
  println("")
  println("A2E.jl   dump  absorbed.data emitted.data [gpu nstoch [library]]")   
  println("")
  println("Input:")
  println("   dump            = solver files written by A2E_pre.py or A2E_pre.jl or DE_to_GSET.jl")
  println("                     or even dumped from A2E.cpp")
  println("   absorbed.data   = file of absorptions (photons/Hz/H), as calculated by CRT or SOC")
  println("   emitted.data    = file of emissions (photons/Hz/H) that can be used as input for CRT and SOC")
  println("   library         = if >0, solve emission using a library (created here, we need all frequencies!)")
  println("   gpu             = if >0, try to use a GPU for the OpenCL kernel")
  println("                     if =0, use CPU for the OpenCL kernel")
  println("                     of <0, solve in Julia without OpenCL")
  exit(1)
end
if (length(ARGS)>3)
  GPU = parse(Int32, ARGS[4])
  if (length(ARGS)>4)
    NSTOCH = parse(Int32, ARGS[5])
    if (length(ARGS)>5)
      USELIB = (parse(Int32, ARGS[6])>0)
    end
  end
end


# Read solver dump file
@printf("Reading solver file: %s\n", ARGS[1])
FP      =  open(ARGS[1], "r")
NFREQ   =  read(FP, Int32)                     # NFREQ
FREQ    =  zeros(Float32, NFREQ)
read!(FP,  FREQ)                               # FREQ[NFREQ]
GD      =  read(FP, Float32)                   # GRAIN_DENSITY
NSIZE   =  read(FP, Int32)                     # NSIZE
S_FRAC  =  zeros(Float32, NSIZE)
read!(FP,  S_FRAC)                             # S_FRAC
NE      =  read(FP, Int32)                     # NE
SK_ABS  =  zeros(Float32, NFREQ, NSIZE)        # coordinates switched !!!  FREQ runs faster !!!
read!(FP,  SK_ABS)                             # Python SK_ABS[NSIZE, NFREQ], Julia SK_ABS[NFREQ, NSIZE]
K_ABS   =  sum(SK_ABS, dims=2)                 # sum over sizes

# since 2018-12-29 also SOC absorbed.data starts with [ cells, nfreq ] only
fpA      = open(ARGS[2], "r+")                 # absorbed 
CELLS    = read(fpA, Int32)
nfreq    = read(fpA, Int32)
if (nfreq!=NFREQ)
  @printf("Solver file %s has %d and the absorption file %d frequencies ?\n", ARGS[1], NFREQ, nfreq)
  exit(0)
end
ABSORBED = Mmap.mmap(fpA, Matrix{Float32}, (NFREQ, CELLS))   # FREQ faster
if true   # Necessary to avoid huge emission in rare cells ???
  ABSORBED[NFREQ,:] = clamp.(ABSORBED[NFREQ,:], 0.0, 0.2*ABSORBED[NFREQ-1,:])  # FREQ faster
end  

# Emitted has always 2 int header
fpE      =  open(ARGS[3], "w+")                # emitted
write(fpE, CELLS, NFREQ)
EMITTED  = Mmap.mmap(fpE, Matrix{Float32}, (NFREQ, CELLS))  # FREQ faster
EMITTED[:,:] .= 0.0


function PlanckSafe(f, T)   # Planck function
  # Add clip to get rid of warnings
  H_CC    =  7.372496678e-48    #   PLANCK/C_LIGHT^2
  return 2.0*H_CC*f*f*f / (exp(clamp.(H_K*f/T,-100,+100))-1.0)
end  


NIP         =  30000  # number of interpolation points for the lookup tables (equilibrium dust)



# solve using GPU
platform = cl.platforms()[1]
t0 = time()
if (GPU>0)
  device = cl.devices(platform, cl.CL_DEVICE_TYPE_GPU)[1]
  LOCAL  = Int32(32)
else
  device = cl.devices(platform, cl.CL_DEVICE_TYPE_CPU)[1]
  LOCAL  = Int32(4)
end
context = cl.Context(device)
queue   = cl.CmdQueue(context)

GLOBAL      =  Int32(max(BATCH, 64*LOCAL))
if (GLOBAL%64!=0)
  GLOBAL  = Int32((floor(GLOBAL/64)+1)*64)
end

OPT         =  "-D NE=$NE -D LOCAL=$LOCAL -D NFREQ=$NFREQ -D CELLS=$CELLS -D NIP=$NIP -D WITH_X=0"
OPT        *=  @sprintf(" -D FACTOR=%.4ef", FACTOR)

source      =  open(homedir()*"/starformation/SOC/kernel_A2E.c") do file read(file, String)  end
prog        =  cl.Program(context, source=source)
program     =  cl.build!(prog, options=OPT)

Iw_buf      =  cl.Buffer(Float32, context, :r,  NE*NE*NFREQ)
L1_buf      =  cl.Buffer(Int32,   context, :r,  NE*NE)
L2_buf      =  cl.Buffer(Int32,   context, :r,  NE*NE)
Tdown_buf   =  cl.Buffer(Float32, context, :r,  NE)
EA_buf      =  cl.Buffer(Float32, context, :r,  NE*NFREQ)
Ibeg_buf    =  cl.Buffer(Int32,   context, :r,  NFREQ)
AF_buf      =  cl.Buffer(Float32, context, :r,  NFREQ)
ABS_buf     =  cl.Buffer(Float32, context, :r,  BATCH*NFREQ)
EMIT_buf    =  cl.Buffer(Float32, context, :rw, BATCH*NFREQ)
A_buf       =  cl.Buffer(Float32, context, :rw, BATCH*(Int32(floor(NE*NE-NE)/2))) # lower triangle only

if (false)
  X_buf       =  cl.Buffer(Float32, context, :w,  BATCH*NE)    # no initial values -> write only
end

if (NSTOCH<NSIZE)   # Prepare to solve equilibrium temperature emission for larger grains
  TTT_buf   =  cl.Buffer(Float32, context, :r,   4*NIP)
  T_buf     =  cl.Buffer(Float32, context, :rw,  4*BATCH)
  KABS_buf  =  cl.Buffer(Float32, context, :r,   4*NFREQ)
  FREQ_buf  =  cl.Buffer(Float32, context, :r,   4*NFREQ)
  # we can use EMIT_buf and ABS_buf, which are correct size for processing BATCH cells !!
  kernel_T  =  program.EqTemperature
  #                               icell     kE          oplogkE     Emin       
  ## kernel_T.set_scalar_arg_dtypes([np.int32, np.float32, np.float32, np.float32,
  #  FREQ   KABS    TTT    ABS    T      EMIT
  #None,     None,   None,  None,  None,  None   ])
  #cl.enqueue_copy(queue,   FREQ_buf,  FREQ)
  EMIT      =  zeros(Float32, NFREQ, BATCH) # FREQ faster
end            

DoSolve  =  cl.Kernel(program, "DoSolve")
## DoSolve.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None, None, None, None, None, None, None, None, None ])




X        =  zeros(Float32, BATCH, NE)
# A2E.py strips the option of iterative solvers -- no worries about initial values.
emit     =  zeros(Float32, NFREQ, BATCH)  # freq faster




mutable struct TreeNode
parent::Int
children::Vector{Int}
lo::Float32
hi::Float32
cell::Int32   # index of the cell in the full model
idx::Int32    # index of the cell in ICELLS = cells selected into the library tree
A::Float32    # minimum of the range covered by the children
B::Float32    # maximum of the range covered by the children
level::Int32
end

mutable struct Tree
nodes::Vector{TreeNode}
end

Tree() = Tree([TreeNode(
0, Vector{Int}(), Float32(-1.0), Float32(-1.0), Int32(-1), Int32(-1), Float32(-2.0),  Float32(-3.0), Int32(1)
)])

function addchild(tree::Tree, id::Integer, lo::AbstractFloat, hi::AbstractFloat)
  1 <= id <= length(tree.nodes) || throw(BoundsError(tree, id))
  push!(tree.nodes, TreeNode(Int32(id), Vector{}(), Float32(lo), Float32(hi), Int32(-1), Int32(-1), 
        Float32(-2.0), Float32(-3.0), tree.nodes[id].level+1))
  child = length(tree.nodes)
  push!(tree.nodes[id].children, child)
  child
end

children(tree, id) = tree.nodes[id].children
parent(tree,id) = tree.nodes[id].parent



function prepare_tree(ABSORBED, NB)
  # Analyse absorptions for all cells, some frequency ranges
  # Construct a grid with discretisation in the intensity per frequency range
  freq   =  readdlm("freq.dat")[:,1]
  BANDS  =  3   # how many frequency bands are used in the gridding
  um     =  2.9979e12./freq
  #               UV                 blue                   red
  RANGES =  [ (um.>0.05).&(um.<0.2), (um.>0.3).&(um.<0.55),  (um.>0.7).&(um.<1.5)  ]
  RANGES =  [ (um.>0.05).&(um.<0.1), (um.>0.2).&(um.<0.4),  (um.>0.5).&(um.<1.0)  ]
  RANGES =  [ (um.>0.1).&(um.<0.7),  (um.>0.7).&(um.<1.4),  (um.>1.4).&(um.<3.0)  ]
  RANGES =  [ (um.>0.5).&(um.<1.1),  (um.>1.5).&(um.<2.4),  (um.>2.4).&(um.<7.0)  ]
  # Make  vectors of the sum of absorbed photons in the different bins
  INT1   =  zeros(Float32, CELLS, BANDS)
  for iband =1:3    # INT[CELLS, BANDS]   ABSORBED[NFREQ, CELLS]
    INT1[:,iband] = sum(ABSORBED[RANGES[iband], :], dims=1) # sum of absorptions over band iband
  end  
  @printf("LIMITS  %.3e %.3e   %.3e %.3e   %.3e %.3e\n",
  minimum(INT1[:,1]), maximum(INT1[:,1]), minimum(INT1[:,2]), maximum(INT1[:,2]), minimum(INT1[:,3]), maximum(INT1[:,3]))
  if false
    subplot(221)
    hist(log10.(INT1[:,1]), bins=1000, alpha=0.5, color="b")
    subplot(222)
    hist(log10.(INT1[:,2]), bins=1000, alpha=0.5, color="g")
    subplot(223)
    hist(log10.(INT1[:,3]), bins=1000, alpha=0.5, color="r")
    show()
    exit(0)
  end
  # Create the basic tree structure
  #         root =  node 1
  #      level-1 =  discretize INT1,  children  j=1:50, child c
  #      level-2 =  discretize INT2,  children  k=1:50, child cc
  #      level-3 =  discretize INT3,  children  l=1:50
  T      =  Tree()
  a, b   =  quantile(INT1[:,1], (0.5*DFRAC, 1.0-0.5*DFRAC))
  a, b   =  0.9999*a, 1.0001*b
  T.nodes[1].A, T.nodes[1].B = a, b
  lim1   =  10.0.^range(log10(a), stop=log10(b), length=NB[1]+1)    # gridding of iband=1
  for j=1:NB[1]
    addchild(T, 1, lim1[j], lim1[j+1])      # 1 -> j -> k -> l
  end
  for j=1:NB[1]                                                           # loop over those children
    # select cells for current bin j, based on iband=1
    INT2  =  INT1[ (INT1[:,1].>=lim1[j]) .& (INT1[:,1].<=lim1[j+1]), : ]  # cells for bin j at iband=1
    if (size(INT2)[1]<1)
      continue                              # no cells in this bin
    end
    a, b  =  quantile(INT2[:,2], (0.5*DFRAC, 1.0-0.5*DFRAC))              # bin j of iband=1, intensity limits for iband=2
    a, b  =  0.9999*a, 1.0001*b
    lim2  =  10.0.^range(log10(a), stop=log10(b), length=NB[2]+1)         # bins iband=2, level 2
    c     =   children(T, 1)[j]             # level-1 child
    T.nodes[c].A, T.nodes[c].B = a, b
    for k = 1:NB[2]
      addchild(T, c, lim2[k], lim2[k+1])    # under c -> level-2 children
    end
    for k=1:NB[2]                           # loop over level-2 children
      INT3  =  INT2[(INT2[:,2].>=lim2[k]) .& (INT2[:,2].<lim2[k+1]), :]   # cells in level 2 bin, iband=2
      if (size(INT3)[1]<1)
        continue                            # no cells in the bin
      end
      a, b  =  quantile(INT3[:,3], (0.5*DFRAC, 1.0-0.5*DFRAC))            # limits for iband=3 values
      a, b  =  0.9999*a, 1.0001*b
      lim3  =  10.0.^range(log10(a), stop=log10(b), length=NB[3]+1)       # bins for level 3, iband=3
      cc    =  children(T, c)[k]
      T.nodes[cc].A, T.nodes[cc].B = a, b
      for l = 1:NB[3]
        addchild(T, cc, lim3[l], lim3[l+1]) # under cc -> level-3 nodes
      end # for l
    end # for k
  end # for j
  println("Empty tree created")
  
  # Go again through the hierarchy and select actual cells that fall closest to the
  # centre of the bins, update those values into the nodes as well as the indices of the selected cells
  ICELLS = -1*ones(Int32, prod(NB))
  ind    =  0
  INDEX  =  1:CELLS
  for j in children(T, 1)    #  1 -> j -> k -> l
    if (T.nodes[j].lo<=0.0)  
      continue    
    end   # skip missing nodes
    a      =  (T.nodes[j].lo + T.nodes[j].hi)/2.0
    cond1  =           @.         (INT1[:,1]>=T.nodes[j].lo) & (INT1[:,1]<=T.nodes[j].hi)
    for k in children(T, j)
      if (T.nodes[k].lo<=0.0)
        continue
      end
      b      =  (T.nodes[k].lo + T.nodes[k].hi)/2.0
      cond2  =         @. cond1 & (INT1[:,2]>=T.nodes[k].lo) & (INT1[:,2]<=T.nodes[k].hi)
      for l in children(T, k)
        if (T.nodes[l].lo<=0.0)
          continue
        end
        cond3 =        @. cond2 & (INT1[:,3]>=T.nodes[l].lo) & (INT1[:,3]<=T.nodes[l].hi)
        tmp   =  INT1[cond3, :]
        if (size(tmp)[1]>0)  # we found a cells belonging to this bin = leaves of the library tree
          c               =  0.5*(T.nodes[l].lo + T.nodes[l].hi)
          ## cell         =  argmin( @. (abs(tmp[:,1]-a)/a + abs(tmp[:,2]-b)/b + abs(tmp[:,3]-c)/c) )
          ## cell         =  argmin( @. ((tmp[:,1]-a)/a)^2 + ((tmp[:,2]-b)/b)^2 + (abs(tmp[:,3]-c)/c)^2 )
          cell            =  argmin(  @. (log10(tmp[:,1])-log10(a))^2 + (log10(tmp[:,2])-log10(b))^2 + (log10(tmp[:,3])-log10(c))^2 )
          ind            +=  1      # ICELLS contains only cells selected for the library (nothing for missing leafs)
          cell            =  INDEX[cond3][cell]  # above cell is index within the cond3 selection
          T.nodes[l].cell =  cell   # direct reference to the full cloud
          ICELLS[ind]     =  cell
          T.nodes[l].idx  =  ind    # reference to ICELLS, cells selected for library
        end        
      end
    end
  end
  println("Selected cells added to the tree")
  return T, ICELLS[1:ind], INT1
end  



function IP(parent, icell, INT, T, tmp_2d)
  # For tree node parent, return the interpolated value from its 
  # children. icell is the index to INT giving the absorptions for the cell.
  # For leaf nodes, field idx is index to the emission vectors in tmp_2d.
  level =  T.nodes[parent].level   # root node has level=1 is pointing to level=1 children
  val   =  INT[icell, level]       # the discriminating value at this level
  if (level==3)
    # parent is pointing to leaf nodes, return 1-2 idx values with weights
    # interpolation is done using the actual INT values for the cells that are selected as leafs
    C = children(T, parent)
    for i=1:length(C)
      m = T.nodes[C[i]]
      if (m.lo>val)                   # INT is below the range covered by the children
        return [], []                 # no weights, no idx values
      end
      if (m.hi>=val)                  # found a bin with upper limit above INT
        if (m.idx<1)                  # found a bin... but no cells in the bin
          return [], []               # no weights, no idx values
        end
        # ok, we have m as one node, see if we can interpolated between this C[i] and a neighbour
        
        # return [1.0], [m.idx]
        
        if ((i<NB[3])&&(val>INT[m.cell,level]))   # possible other node on the high side
          o = T.nodes[C[i+1]]         # high-side node
          if (o.cell>0)               # and it is a valid node -- we have interpolation
            if false
              md = abs(log10(val)-log10(INT[m.cell, level]))
              od = abs(log10(val)-log10(INT[o.cell, level]))
            else
              md = val                - INT[m.cell, level]
              od = INT[o.cell, level] - val
            end
            return [  od/(md+od), md/(md+od) ], [ m.idx, o.idx ]
          end
        end
        if ((i>1)&&(val<INT[m.cell,level]))   # possible other node on the low side
          o = T.nodes[C[i-1]]
          if (o.cell>0)
            if false
              md = abs(log10(val)-log10(INT[m.cell, level]))
              od = abs(log10(val)-log10(INT[o.cell, level]))
            else
              md =  INT[m.cell, level] - val
              od =  val                - INT[o.cell, level]
            end
            return [  od/(md+od), md/(md+od) ], [ m.idx, o.idx ]
          end
        end
        # if we come here, we failed to find a neighbour for interpolation
        # -> return just the single one
        return [1.0], [m.idx]
      end
    end # for i
    return [], []
  else
    # we are not yet at the leaf level, call recursively and return combined weight and idx vectors
    # interpolation is done using the nominal bin centres
    C = children(T, parent)
    for i=1:length(C)
      m  = T.nodes[C[i]]
      if (m.lo>val)                   # INT is below the range covered by the children
        return [], []                 # no weights, no idx values
      end
      if (m.hi>=val)                  # found a bin with upper limit above INT
        # ok, we have m as one node with a subtree
        # see if we can interpolated between this and a tree under a neighbouring cell
        wm, im  =   IP(C[i], icell, INT, T, tmp_2d)            # result from the subtree below m
                
        # return wm, im
        
        md      =   abs(log10(val) - log10(0.5*(m.lo+m.hi)))   # using the bin centre to measure distance
        mc      =   0.5*(m.lo+m.hi)                            # bin centre
        if ((i<NB[level])&&(val>mc))                           # possible other node on the high side
          o       =  T.nodes[C[i+1]]                           # the high-side node
          wo, io  =  IP(C[i+1], icell, INT, T, tmp_2d)         # result from the other subtree
          od      =  abs(log10(val) - log10(0.5*(o.lo+o.hi)))  # using the bin centre to measure distance
          wm     *=  (od/(od+md))
          wo     *=  (md/(od+md))
          return vcat(wm, wo),  vcat(im, io)          
        end
        if ((i>1)&&(val<mc))                                   # possible other node on the low side
          o       =  T.nodes[C[i-1]]
          wo, io  =  IP(C[i-1], icell, INT, T, tmp_2d)
          od      =  abs(log10(val) - log10(0.5*(o.lo+o.hi)))  # using the bin centre to measure distance
          wm     *=  (od/(od+md))
          wo     *=  (md/(od+md))
          return vcat(wm, wo),  vcat(im, io)
        end
        # if we come here, we failed to find a neighbour for interpolation  ->  return result for a single subtree
        return wm, im
      end
    end  # for i
    return [], []
  end  # else - no leaf
end




function solve_with_tree(T, ICELLS, INT, isize)
  """
  T            =   precalculated tree structure that defines the library discretisation
  (for all isize), 
  T.node[i].cell is index to full ABSORBED array,
  T.node[i].idx  is index to ICELLS array
  ICELLS is list of cells in the leafs of the library tree
  ABSORBED[:,ICELLS] are first solved into tmp_2d
  then tree is used to find the leaf closest to an arbitrary cell,
  the .idx field of the node gives the index to tmp_2d array                   
  INT          =   INT[CELLS, BANDS] = integrated absorptions per band, the basis of library discretisation
  ICELLS       =   vector of all cell indices included in the tree
  """
  # solve the emission for all cells mentioned in the tree leaves
  global NFREQ, BATCH, GLOBAL, LOCAL
  global SK_ABS, S_FRAC
  global queue, Iw_buf, L1_buf, L2_buf, Tdown_buf, EA_buf, Ibeg_buf, AF_buf, ABS_buf, EMIT_buf, A_buf ## , X_buf
  
  AF    =  Array{Float64}(SK_ABS[:,isize]) ./ Array{Float64}(K_ABS[:])     # => E per grain
  AF  ./=  S_FRAC[isize]*GD      # "invalid value encountered in divide"
  AF    =  Array{Float32}(clamp.(AF, 1.0e-32, 1.0e+100))
  if true
    AF[isfinite.(AF).==false] .= 1.0e-30
  end    
  
  # Initialise OpenCL system to solve emission for the current isize
  # note that we are reading solver data from file, assuming each isize is processed
  # only once and in order
  cl.write!(queue, AF_buf,   AF)      # AF
  noIw  =  read(FP, Int32)
  Iw    =  zeros(Float32, noIw)
  read!(FP, Iw)
  cl.write!(queue, Iw_buf, Iw)        # Iw
  L1    =  zeros(Int32, NE*NE)
  read!(FP, L1)
  cl.write!(queue, L1_buf,  L1)       # L1
  L2    =  zeros(Int32, NE*NE)
  read!(FP, L2)
  cl.write!(queue, L2_buf,  L2)       # L2
  Tdown =  zeros(Float32, NE)
  read!(FP, Tdown)
  cl.write!(queue, Tdown_buf, Tdown)  # Tdown
  EA    =   zeros(Float32, NE*NFREQ)
  read!(FP, EA)
  cl.write!(queue, EA_buf,    EA)     # EA
  Ibeg  =   zeros(Int32, NFREQ)
  read!(FP, Ibeg)
  cl.write!(queue, Ibeg_buf,  Ibeg)   # Ibeg
  cl.finish(queue)

  
  # Solve emission for all cases mentioned in the tree -- cases ABSORBED[:, ICELLS]
  # should be possible in a single kernel call BUT MUST HAVE BATCH>= NB*NB*NB = length(CELLS)
  t0      = time()
  tmp     = zeros(Float32, NFREQ*BATCH)
  tmp_2d  = reshape(tmp,   Int64(NFREQ), Int64(BATCH))  # another view to the same array
  batch   = length(ICELLS)
  if (batch>BATCH)
    println("A2E.jl -- BATCH must be at least as large as the number of TREE leaves !!")
    exit(0)
  end  
  for i = 1:batch  # all batch leaf nodes solved in a single kernel call
    tmp_2d[:, i]  =  ABSORBED[:, ICELLS[i]]
  end
  cl.write!(queue, ABS_buf, tmp)
  cl.finish(queue)
  queue(DoSolve, GLOBAL, LOCAL,  Int32(batch),   Int32(isize-1),  Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
  Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf)  ######    X_buf)
  cl.finish(queue)
  tmp[:]  =  cl.read(queue, EMIT_buf)        # BATCH*NFREQ
  dt = time()-t0 
  @printf("Tree solutions:    %.3f seconds, %.3e seconds per cell\n", dt, dt/prod(NB))
  # at this point tmp_2d[NFREQ, batch] contains the emissions for the current size, all cells in the library tree
  
  
  
  # Loop over the whole cloud, solving (=lookup to the library) for all cells that are in the parameter 
  # space region covered by the tree
  # note the list of cells outside the library that still need to be solved directly
  unsolved = 0
  UNSOLVED = zeros(Int32, CELLS)   # hopefully unsolved remains << CELLS
  fudge    = 1.0
  t0       = time()
  for icell = 1:CELLS    
    # find the leaf from the tree, if exists,   1 -> j -> k -> l
    j0, k0, l0 = 0, 0, 0
    for j in children(T, 1)
      if (T.nodes[j].lo>INT[icell, 1])  # this cell falls below the range covered by the library
        j0 = -1 ; break
      end
      if (T.nodes[j].hi>=INT[icell, 1]) # found a valid bin
        j0    = j
        fudge =  INT[icell,1]/(0.5*(T.nodes[j0].lo+T.nodes[j0].hi))  # zeroth order correction for discretisation
        break    
      end
    end
    if (j0>0)     # j is the correct bin, go through its children
      for k in children(T, j0)
        if (T.nodes[k].lo>INT[icell, 2])
          k0 = -1 ; break
        end
        if (T.nodes[k].hi>=INT[icell, 2])
          k0 = k
          break    # found a valid bin
        end
      end
      if (k0>0)
        for l in children(T, k0)
          if (T.nodes[l].lo>INT[icell, 3])
            l0 = -1 ; break
          end
          if (T.nodes[l].hi>=INT[icell, 3])
            l0 = l
            break    # found a valid bin
          end
        end
        if false
          if (l0<1)
            @printf("-l0    %10.3e  out of %10.3e %10.3e\n", INT[icell,3], T.nodes[k0].A, T.nodes[k0].B)
            @printf("  %6d  %.3e [%.3e,%.3e]   %.3e [%.3e,%.3e]   %.3e [%.3e,%.3e]\n",
            icell, INT[icell, 1], T.nodes[j0].lo, T.nodes[j0].hi,  
            INT[icell, 2],        T.nodes[k0].lo, T.nodes[k0].hi,  
            INT[icell, 3],        T.nodes[k0].A,  T.nodes[k0].B)
          end
        end
      end
    end    
    if ((l0>0)&&(T.nodes[l0].idx>0))  # we found a leaf, that has .idx pointing to the tmp2_d arrays of precalculated emission vectors
      EMITTED[:, icell] +=  fudge .*  tmp_2d[:, T.nodes[l0].idx]  # library solution - no interpolation
    else
      unsolved           += 1
      UNSOLVED[unsolved]  = icell
      continue
    end
  end  # for icell
  dt = time()-t0
  @printf("Solve with library %.3f seconds, %.3e seconds per cell\n", dt, dt/CELLS)
  
  # we are left with some number of unsolved cells ... loop over them in batches <= BATCH
  @printf("  ... %d cells solved with library, remain %d cells = %.2f per cent\n",  
  CELLS-unsolved, unsolved, unsolved*100.0/CELLS)
  
  
  iun = 1  # first unsolved , index to UNSOLVED
  while (iun<=unsolved)
    # solve next <= BATCH cells from UNSOLVED
    batch = min(BATCH, unsolved-iun+1)
    for i = 1:batch
      tmp_2d[:, i]  =  ABSORBED[:, UNSOLVED[iun+i-1]]
    end
    cl.write!(queue, ABS_buf, tmp)
    cl.finish(queue)
    queue(DoSolve, GLOBAL, LOCAL,  Int32(batch),   Int32(isize-1),  Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
    Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf)  ####   X_buf)
    cl.finish(queue)
    tmp[:]  =  cl.read(queue, EMIT_buf)        # BATCH*NFREQ
    # add to EMITTED
    for i = 1:batch
      EMITTED[:, UNSOLVED[iun+i-1]] += tmp_2d[:,i]
    end
    # we processed batch previously unsolved cells
    iun += batch
  end
    
end






function solve_with_hash(isize, ABSORBED, EMITTED)
  """
  Loop through the cells, if solution is not in Dict, add to CL batch, add results to Dict
  """
  # solve the emission for all cells mentioned in the tree leaves
  global NFREQ, BATCH, GLOBAL, LOCAL
  global SK_ABS, S_FRAC
  global queue, Iw_buf, L1_buf, L2_buf, Tdown_buf, EA_buf, Ibeg_buf, AF_buf, ABS_buf, EMIT_buf, A_buf ## , X_buf  


  freq   =  readdlm("freq.dat")[:,1]
  BANDS  =  3   # how many frequency bands are used in the gridding
  um     =  2.9979e12./freq
  RANGES =  [ (um.>0.5).&(um.<1.1),  (um.>1.5).&(um.<2.4),  (um.>2.4).&(um.<7.0)  ]  
  # NB     =  [ 100, 100, 100 ]
  # NB     =  [ 70, 70, 70 ]
  # NB     =  [ 32, 32, 32 ]
  # NB     =  [ 25, 25, 25 ]
  # NB     =  [ 15, 15, 15 ]
  # NOTE -- while tree discretises each band using cells selected by previous bands,
  #         this hash algorihm has the same discretisation for second and third band,
  #         irrespective of the values of the preceding bands => must be larger dimensions!!
  NB     =  [ 176, 176, 176 ]
  
  HASH   =  zeros(Int32, CELLS)  # hash values for each cell, based on absorptions in the three bands
  for iband =1:length(NB)
    if (iband>1)
      HASH .*=  NB[iband]
    end
    tmp   =  sum(ABSORBED[RANGES[iband], :], dims=1) # sum of absorptions over band iband
    a, b  =  minimum(tmp), maximum(tmp)
    for i =1:CELLS
      HASH[i]  +=  Int32(floor(1.0e-6+(tmp[i]-a)/(b-a)*(NB[iband]-1)))
    end
  end  
  HASH[:] .+= 1  #  1 ...   1*(N1-1)*(N2-1)*(N3-1)   ... WRK allocated for N1*N2*N3 !
  @printf("HASH VALUES  %d  %d\n", minimum(HASH), maximum(HASH))
  println("MAXIMUM SHOULD BE < ", prod(NB))
  
  AF    =  Array{Float64}(SK_ABS[:,isize]) ./ Array{Float64}(K_ABS[:])     # => E per grain
  AF  ./=  S_FRAC[isize]*GD      # "invalid value encountered in divide"
  AF    =  Array{Float32}(clamp.(AF, 1.0e-32, 1.0e+100))
  if true
    AF[isfinite.(AF).==false] .= 1.0e-30
  end      
  cl.write!(queue, AF_buf,   AF)      # AF
  noIw  =  read(FP, Int32)
  Iw    =  zeros(Float32, noIw)
  read!(FP, Iw)
  cl.write!(queue, Iw_buf, Iw)        # Iw
  L1    =  zeros(Int32, NE*NE)
  read!(FP, L1)
  cl.write!(queue, L1_buf,  L1)       # L1
  L2    =  zeros(Int32, NE*NE)
  read!(FP, L2)
  cl.write!(queue, L2_buf,  L2)       # L2
  Tdown =  zeros(Float32, NE)
  read!(FP, Tdown)
  cl.write!(queue, Tdown_buf, Tdown)  # Tdown
  EA    =   zeros(Float32, NE*NFREQ)
  read!(FP, EA)
  cl.write!(queue, EA_buf,    EA)     # EA
  Ibeg  =   zeros(Int32, NFREQ)
  read!(FP, Ibeg)
  cl.write!(queue, Ibeg_buf,  Ibeg)   # Ibeg
  cl.finish(queue)

  no       =  1+prod(NB)
  SOLVED   =  zeros(Int32, no)    # 0 = not solved, 1 = added to be computed, 2 = solution available
  WRK      =  zeros(Float32, NFREQ, no)  
  UNSOLVED =  zeros(Int32, BATCH)
  tmp      =  zeros(Float32, NFREQ*BATCH)
  tmp_2d   =  reshape(tmp,   Int64(NFREQ), Int64(BATCH))  # another view to the same array  
  println("WRK allocated:", size(WRK))
  unsolved =  0
  solved   =  0
  hashed   =  0
  
  
  for icell = 1:CELLS        
    if (SOLVED[HASH[icell]]==2)
      EMITTED[:, icell]   +=   WRK[:, HASH[icell]]       # WRK is just this one size
      hashed += 1
    elseif (SOLVED[HASH[icell]]==1)                      # hash is already queued, result not available
      HASH[icell] = -HASH[icell]
    else                                                 # add to be solved
      unsolved            +=   1
      tmp_2d[:,unsolved]   =   ABSORBED[:, icell]
      UNSOLVED[unsolved]   =   icell
      SOLVED[HASH[icell]]  =   1                         # is queued, not yet solved
      if (unsolved==BATCH)                               # buffer is full, solve with a single call to CL kernel
        println("  --- Solve another batch with $BATCH cells")
        cl.write!(queue, ABS_buf, tmp)
        cl.finish(queue)
        queue(DoSolve, GLOBAL, LOCAL,  Int32(BATCH),   Int32(isize-1),  
        Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf,
        Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf)  ##,   X_buf)
        cl.finish(queue)
        solved += BATCH
        # copy results to EMITTED
        tmp[:]  =  cl.read(queue, EMIT_buf)                 # BATCH*NFREQ
        for iun = 1:BATCH
          WRK[:, HASH[UNSOLVED[iun]]]   = tmp_2d[:, iun]    # emission ... for current isize only !!
          EMITTED[:, UNSOLVED[iun]]    += WRK[:, HASH[UNSOLVED[iun]]]  # icell=UNSOLVED[i] is no solved
          SOLVED[HASH[UNSOLVED[iun]]]   = 2                 # result is only now available
        end
        unsolved = 0
      end
    end
  end  # for icell
  dt = time()-t0
  @printf("Solve with hash %.3f seconds, %.3e seconds per cell\n", dt, dt/CELLS)
  @printf("Yes, solved using hash: %d, fraction %.3f of all cells\n", hashed, hashed/CELLS)
  if (unsolved>0) # solve the last partial batch -- last cells for which Dict did not give an answer
    cl.write!(queue, ABS_buf, tmp)
    cl.finish(queue)
    queue(DoSolve, GLOBAL, LOCAL,  Int32(unsolved),   Int32(isize-1),  Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
    Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf)  ##   X_buf)
    cl.finish(queue)
    tmp[:]  =  cl.read(queue, EMIT_buf)        # BATCH*NFREQ
    for iun = 1:unsolved
      WRK[:, HASH[UNSOLVED[iun]]]  =  tmp_2d[:, iun]
      EMITTED[:, UNSOLVED[iun]]   +=  WRK[:, HASH[UNSOLVED[iun]]]
      SOLVED[HASH[UNSOLVED[iun]]]  =  2
    end
    solved += unsolved
  end
  # we are finished... except for cells that were encountered while the same hash was in the queue
  for icell=1:CELLS
    if (HASH[icell]<0)
      EMITTED[:, icell]   +=   WRK[:, -HASH[icell]]
    end
  end
  @printf("Hash method solved %d cells, a fraction %.3f of all\n", solved, solved/CELLS)

end











function solve_with_tree_ip(T, ICELLS, INT, isize)
  """
  T            =   precalculated tree structure that defines the library discretisation
  (for all isize), 
  T.node[i].cell is index to full ABSORBED array,
  T.node[i].idx  is index to ICELLS array
  ICELLS is list of cells in the leafs of the library tree
  ABSORBED[:,ICELLS] are first solved into tmp_2d
  then tree is used to find the leaf closest to an arbitrary cell,
  the .idx field of the node gives the index to tmp_2d array                   
  INT          =   INT[CELLS, BANDS] = integrated absorptions per band, the basis of library discretisation
  ICELLS       =   vector of all cell indices included in the tree
  """
  # solve the emission for all cells mentioned in the tree leaves
  global NFREQ, BATCH, GLOBAL, LOCAL
  global SK_ABS, S_FRAC
  global queue, Iw_buf, L1_buf, L2_buf, Tdown_buf, EA_buf, Ibeg_buf, AF_buf, ABS_buf, EMIT_buf, A_buf ##, X_buf
  
  AF    =  Array{Float64}(SK_ABS[:,isize]) ./ Array{Float64}(K_ABS[:])     # => E per grain
  AF  ./=  S_FRAC[isize]*GD      # "invalid value encountered in divide"
  AF    =  Array{Float32}(clamp.(AF, 1.0e-32, 1.0e+100))
  if true
    AF[isfinite.(AF).==false] .= 1.0e-30
  end    
  
  # Initialise OpenCL system to solve emission for the current isize
  # note that we are reading solver data from file, assuming each isize is processed
  # only once and in order
  cl.write!(queue, AF_buf,   AF)      # AF
  noIw  =  read(FP, Int32)
  Iw    =  zeros(Float32, noIw)
  read!(FP, Iw)
  cl.write!(queue, Iw_buf, Iw)        # Iw
  L1    =  zeros(Int32, NE*NE)
  read!(FP, L1)
  cl.write!(queue, L1_buf,  L1)       # L1
  L2    =  zeros(Int32, NE*NE)
  read!(FP, L2)
  cl.write!(queue, L2_buf,  L2)       # L2
  Tdown =  zeros(Float32, NE)
  read!(FP, Tdown)
  cl.write!(queue, Tdown_buf, Tdown)  # Tdown
  EA    =   zeros(Float32, NE*NFREQ)
  read!(FP, EA)
  cl.write!(queue, EA_buf,    EA)     # EA
  Ibeg  =   zeros(Int32, NFREQ)
  read!(FP, Ibeg)
  cl.write!(queue, Ibeg_buf,  Ibeg)   # Ibeg
  cl.finish(queue)
  
  # Solve emission for all cases mentioned in the tree -- cases ABSORBED[:, ICELLS]
  # should be possible in a single kernel call BUT MUST HAVE BATCH>= NB*NB*NB = length(CELLS)
  tmp     = zeros(Float32, NFREQ*BATCH)
  tmp_2d  = reshape(tmp,   Int64(NFREQ), Int64(BATCH))  # another view to the same array
  batch   = length(ICELLS)
  if (batch>BATCH)
    println("A2E.jl -- BATCH must be at least as large as the number of TREE leaves !!")
    exit(0)
  end  
  for i = 1:batch  # all batch leaf nodes solved in a single kernel call
    tmp_2d[:, i]  =  ABSORBED[:, ICELLS[i]]
  end
  cl.write!(queue, ABS_buf, tmp)
  cl.finish(queue)
  queue(DoSolve, GLOBAL, LOCAL,  Int32(batch),   Int32(isize-1),  Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
  Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf) ##   X_buf)
  cl.finish(queue)
  tmp[:]  =  cl.read(queue, EMIT_buf)        # BATCH*NFREQ
  # at this point tmp_2d[NFREQ, batch] contains the emissions for the current size, all cells in the library tree
   
  
  # Loop over the whole cloud, solving (=lookup to the library) for all cells that are in the parameter 
  # space region covered by the tree
  # note the list of cells outside the library that still need to be solved directly
  unsolved = 0
  UNSOLVED = zeros(Int32, CELLS)   # hopefully unsolved remains << CELLS
  emit     = zeros(Float32, NFREQ)

  
  for icell = 1:CELLS
    weights, indices = IP(1, icell, INT, T, tmp_2d)
    if (length(weights)>0)
      weights         /=  sum(weights)
      emit[:]         .=  Float32(0.0)
      for i in 1:length(weights)
        emit  +=  weights[i] * tmp_2d[:, indices[i]]
      end        
      EMITTED[:, icell]  +=  emit
    else
      unsolved           += 1
      UNSOLVED[unsolved]  = icell
      continue
    end
  end  # for icell
  
  
  # we are left with some number of unsolved cells ... loop over them in batches <= BATCH
  @printf("  ... %d cells solved with library, remain %d cells = %.2f per cent\n",  
  CELLS-unsolved, unsolved, unsolved*100.0/CELLS)
    
  iun = 1  # first unsolved , index to UNSOLVED
  while (iun<=unsolved)
    # solve next <= BATCH cells from UNSOLVED
    batch = min(BATCH, unsolved-iun+1)
    for i = 1:batch
      tmp_2d[:, i]  =  ABSORBED[:, UNSOLVED[iun+i-1]]
    end
    cl.write!(queue, ABS_buf, tmp)
    cl.finish(queue)
    queue(DoSolve, GLOBAL, LOCAL,  Int32(batch),   Int32(isize-1),  Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
    Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf) ##   X_buf)
    cl.finish(queue)
    tmp[:]  =  cl.read(queue, EMIT_buf)        # BATCH*NFREQ
    # add to EMITTED
    for i = 1:batch
      EMITTED[:, UNSOLVED[iun+i-1]] += tmp_2d[:,i]
    end
    # we processed batch previously unsolved cells
    iun += batch
  end
  
  
end








function process_size_equilibrium(isize)  
  AF    =  Array{Float64}(SK_ABS[:,isize]) ./ Array{Float64}(K_ABS[:])     # => E per grain
  AF  ./=  S_FRAC[isize]*GD      # "invalid value encountered in divide"
  AF    =  Array{Float32}(clamp.(AF, 1.0e-32, 1.0e+100))
  if true
    AF[isfinite.(AF).==false] .= 1.0e-30
  end    
  
  if (S_FRAC[isize]<1.0e-30) 
    return   # empty size bin
  end
  KABS   =  SK_ABS[:,isize] / (GD*S_FRAC[isize])
  # Prepare lookup table between energy and temperature
  t1     =  time.time()   # this is fast (<1sec) and can be done by host
  TSTEP  =  1600.0/NIP    # hardcoded upper limit 1600K for the maximum dust temperatures
  TT     =  zeros(NIP, float64)
  Eout   =  zeros(NIP, float64)
  DF     =  FREQ[3:end] - FREQ[1:(end-2)]  #  x[i+1] - x[i-1], lengths of intervals for Trapezoid rule
  for i = 1:NIP
    TT[i]   =  4.0 + TSTEP*i
    TMP     =  FACTOR * KABS * PlanckSafe(asarray(FREQ, float64), TT[i])
    # Trapezoid integration TMP over FREQ frequencies
    res     =  TMP[1]*(FREQ[2]-FREQ[1]) + TMP[end]*(FREQ[end]-FREQ[end-1]) # first and last step
    res    +=  sum(TMP[2:(end-1)]*DF)     # the sum over the rest of TMP*DF
    Eout[i] =  4.0*pi * 0.5 * res         # energy corresponding to TT[i]
  end # -- for i
  # Calculate the inverse mapping    Eout -> TTT
  Emin = Eout[1] ;    Emax = Eout[NIP]*0.9999
  # E ~ T^4  => use logarithmic sampling
  kE     = (Emax/Emin)^(1.0/(NIP-1.0))  # E[i] = Emin*pow(kE, i)
  oplgkE = 1.0/log10(kE)
  #ip     = interp1d(Eout, TT)           # (linear) interpolation from energy to temperature
  ip     = interpolate((Eout,), TT, Gridded(Linear()))
  # @printf("Eout = %10.3e ... %10.3e", Emin, Emax)
  TTT    = Array{Float32}(ip(Emin * kE^(0::(NIP-1))))
  print("Mapping E -> T calculated on host: %.3f seconds" % (time.time()-t1))
  # Calculate temperatures on device
  #   ABSORBED * AF  integrated in the kernel over frequency -> Ein
  #   kernel will get ABSORBED, will do integration and table lookup to get T
  #   Because of the large amount of data, kernel calls for GLOBAL cells at a time ...
  #   the emission will be also calculated already for the BATCH cells
  cl.enqueue_copy(queue, TTT_buf,  TTT)     # NIP elements
  cl.enqueue_copy(queue, KABS_buf, KABS)    # NFREQ elements
  T = zeros(BATCH, np.float32)        
  for icell = 1:BATCH:CELLS                 # T_buf is being updated for GLOBAL cells
    b       =  min(icell+(BATCH-1), CELLS)
    tmp_abs =  ABSORBED[:, icell:b]*AF
    tmp_abs =  Array{Float32}(tmp_abs)
    cl.enqueue_copy(queue, ABS_buf,  tmp_abs)  # BATCH*NFREQ elements
    queue(kernel_T, BATCH, LOCAL, Int32(icell), Float32(kE), Float32(oplgkE), Floats32(Emin),
    FREQ_buf, KABS_buf, TTT_buf, ABS_buf, T_buf, EMIT_buf)
    # Add emission to the final array
    # cl.enqueue_copy_buffer(queue, EMIT_buf, EMIT)   # emission for <= GLOBAL cells
    EMIT  =  cl.read(queue, EMIT_buf)
    for i in icell:b
      EMITTED[:, i]  +=  EMIT[:, i-icell+1] * GD * S_FRAC[isize]
    end
    continue
  end # for icell
end




# @@
function process_size_stochastic(isize::Int64
  , NFREQ::Int32, BATCH::Int32, GLOBAL::Int32, LOCAL::Int32
  , SK_ABS::Array{Float32,2}, K_ABS::Array{Float32, 2}, S_FRAC::Array{Float32,1},
  queue::OpenCL.cl.CmdQueue, Iw_buf::OpenCL.cl.Buffer, L1_buf::OpenCL.cl.Buffer, 
  L2_buf::OpenCL.cl.Buffer, Tdown_buf::OpenCL.cl.Buffer, EA_buf::OpenCL.cl.Buffer, 
  Ibeg_buf::OpenCL.cl.Buffer, AF_buf::OpenCL.cl.Buffer, ABS_buf::OpenCL.cl.Buffer, 
  EMIT_buf::OpenCL.cl.Buffer, A_buf::OpenCL.cl.Buffer,
  ABSORBED::Array{Float32,2}, EMITTED::Array{Float32,2})
  
  println("process_size_stochastic isize = ", isize)
  #  A2E.py does this in 16.8 seconds while this julia routine takes 24.3 seconds.
  #  The loop below (without IO and preparations) is already slower than the whole Python calculation.
  #  Even specifying the types of all input parameters, julia is 30% slower compared to python.
  # t0    =  time()
  AF    =  Array{Float64}(SK_ABS[:,isize]) ./ Array{Float64}(K_ABS[:])     # => E per grain
  AF  ./=  S_FRAC[isize]*GD      # "invalid value encountered in divide"
  AF    =  Array{Float32}(clamp.(AF, 1.0e-32, 1.0e+100))
  if true
    AF[isfinite.(AF).==false] .= 1.0e-30
  end      
  cl.write!(queue, AF_buf,   AF)      # AF
  noIw  =  read(FP, Int32)
  Iw    =  zeros(Float32, noIw)
  read!(FP, Iw)
  cl.write!(queue, Iw_buf, Iw)        # Iw
  L1    =  zeros(Int32, NE*NE)
  read!(FP, L1)
  cl.write!(queue, L1_buf,  L1)       # L1
  L2    =  zeros(Int32, NE*NE)
  read!(FP, L2)
  cl.write!(queue, L2_buf,  L2)       # L2
  Tdown =  zeros(Float32, NE)
  read!(FP, Tdown)
  cl.write!(queue, Tdown_buf, Tdown)  # Tdown
  EA    =   zeros(Float32, NE*NFREQ)
  read!(FP, EA)
  cl.write!(queue, EA_buf,    EA)     # EA
  Ibeg  =   zeros(Int32, NFREQ)
  read!(FP, Ibeg)
  cl.write!(queue, Ibeg_buf,  Ibeg)   # Ibeg
  cl.finish(queue)    
  # Loop over the cells, BATCH cells per kernel call
  # tmp     = zeros(Float32, NFREQ*BATCH)
  # tmp_2d  = reshape(tmp,   Int64(NFREQ), Int64(BATCH))
  emit    = zeros(Float32, NFREQ*BATCH)
  emit_2d = reshape(emit,  Int64(NFREQ), Int64(BATCH))
  # @printf("INITIAL = %.2f SECONDS\n", time()-t0)  # --- time spent in this first part is insignificant
  # t0 = time()
  for icell = 1:BATCH:CELLS
    ## t00   =  time()            
    batch =  min(BATCH, CELLS-icell+1)     # actual number of cells
    ## tmp_2d[:, 1:batch]  =  ABSORBED[:, icell:(icell+batch-1)]
    ## cl.write!(queue, ABS_buf, tmp)
    cl.write!(queue, ABS_buf, ABSORBED[:, icell:(icell+batch-1)])
    queue(DoSolve, GLOBAL, LOCAL,  Int32(batch),   Int32(isize-1),  Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf,
    Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf) ##   X_buf)
    cl.finish(queue)
    emit[:]  =  cl.read(queue, EMIT_buf)           # BATCH*NFREQ
    EMITTED[:, icell:(icell+batch-1)] += emit_2d[:, 1:batch]       # contribution of current dust, current size
    if (icell==0)
      @printf("   SIZE %2d/%2d  icell %6d/%6d  %7.2f s/%d %.3e s/cell/size",
      isize, NSIZE, icell, CELLS, time.time()-t00, batch, (time.time()-t00)/batch)
    end    
  end # -- for icell
  ## @printf("LOOP = %.2f SECONDS\n", time()-t0)
end # process_size



DUMP_ARRAY = zeros(Float32, NE, NSIZE)




function process_size_stochastic_julia(isize, NFREQ, NE, SK_ABS, S_FRAC, ABSORBED, EMITTED)
  """
  Solve emission in pure Julia, single grain size as given by the argument.
  Remember that L1, L2, and Ibeg contain indices in the 0-offset system.
  """
  println("Process_size_stochastic_julia isize = ", isize)
  ## global NFREQ, NE, SK_ABS, S_FRAC
  
  AF    =  Array{Float64}(SK_ABS[:,isize]) ./ Array{Float64}(K_ABS[:])     # => E per grain
  AF  ./=  S_FRAC[isize]*GD      # "invalid value encountered in divide"
  AF    =  Array{Float32}(clamp.(AF, 1.0e-32, 1.0e+100))
  if true
    AF[isfinite.(AF).==false] .= 1.0e-30
  end    
  
  noIw  =   read(FP, Int32)
  Iw    =   zeros(Float32, noIw)
  read!(FP, Iw)
  L1    =   zeros(Int32, NE, NE)
  read!(FP, L1)
  L2    =   zeros(Int32, NE, NE)
  read!(FP, L2)
  Tdown =   zeros(Float32, NE)
  read!(FP, Tdown)
  EA    =   zeros(Float32, NE, NFREQ)
  read!(FP, EA)
  Ibeg  =   zeros(Int32, NFREQ)
  read!(FP, Ibeg)
  emit  =   zeros(Float64, NFREQ)
  X     =   zeros(Float64, NE)
  
  # make indices 1-offset
  L1    .+=  1
  L2    .+=  1
  Ibeg  .+=  1
  
  L     =  zeros(Float64, NE, NE)
  t0    =  time()

  for icell=1:CELLS
    if (icell>101)
      continue   # TO SLOW FOR LARGE MODELS !!
    end    
    emit .= 0.0
    if (icell%100==0)
      @printf("cell %6d / %d   (%.3e seconds per cell and size)\n", icell, CELLS, (time()-t0)/100.0)
      t0 = time()
    end    
    # Heating
    iw_index  =  1
    for l=1:(NE-1)
      for u=(l+1):NE
        I           = 0.0
        for i=L1[u,l]:L2[u,l]
          I        +=   ABSORBED[i, icell] .* Iw[iw_index] .* AF[i]
          iw_index += 1
        end
        L[u,l] = max(I, 0.0)
      end
    end
    # Bottom row is already the original A matrix
    # row NE-1 is also the original... except for the diagonal that we can skip
    for j=(NE-2):-1:2
      u = j+1
      for i=1:(j-1)
        L[j,i] += L[u,i]
      end
    end
    # Solve
    X[1] = 1.0e-20
    for j=2:NE
      X[j] = 0.0
      for i=1:(j-1) 
        X[j] += L[j,i] * X[i] 
      end
      X[j] /= (Tdown[j] + 1.0e-30)
      X[j]  = max(X[j], 0.0)
      if (X[j]>1.0e20)
        for i=1:j 
          X[i] *= 1.0e-20
        end
      end
    end
    # Normalise
    I = 0.0
    for i=1:NE
      I += X[i]
    end
    I = 1.0/I
    for i=1:NE
      X[i] *= I
    end
    # Emission
    for j=1:NFREQ
      I = 0.0
      for i=Ibeg[j]:NE
        I += EA[i,j] * X[i]
      end
      emit[j] = I
    end
    EMITTED[:, icell] += emit  # contribution of current dust, current size
    
    if (icell==10)
      DUMP_ARRAY[:, isize] = X
    end
    
  end # -- for icell
    
end # process_size





function Print(T::Tree)
  for i in children(T, 1)
    if (T.nodes[i].lo<0.0)
      continue
    end
    for j in children(T, i)
      if (T.nodes[j].lo<0.0)
        continue
      end
      for k in children(T, j)
        if (T.nodes[k].lo>0.0)
          c, cc, ccc = T.nodes[i], T.nodes[j], T.nodes[k]
          @printf("%2d %2d %2d   %.3e %.3e   %.3e %.3e   %.3e %.3e    cell %d\n",
          i, j, k, c.lo, c.hi,  cc.lo, cc.hi,  ccc.lo, ccc.hi,  ccc.cell)
        end
      end
    end
  end
end


function Plot(T::Tree)
  # Visualise the range of values in the tree
  i = 0
  for c in children(T, 1)
    i += 1
    semilogy(i, T.nodes[c].lo, "kx")
    plot(i, T.nodes[c].hi, "k+")
    for cc in children(T, c)
      i += 1
      plot(i, T.nodes[cc].lo, "bx")
      plot(i, T.nodes[cc].hi, "b+")
      if (true)
        n = length(children(T, cc))
        x = zeros(Float32, n, 2)
        for i =1:n
          ccc = children(T, cc)[i]
          x[i,1] = T.nodes[ccc].lo
          x[i,2] = T.nodes[ccc].hi
        end
        println(i)
        i += 1
        plot(i:(i+n-1), x[:,1], "rx")
        plot(i:(i+n-1), x[:,2], "r+")
        i += n
      end
    end
  end
  show()
end


if (USEHASH)
  for isize = 1:NSIZE
    solve_with_hash(isize, ABSORBED, EMITTED)
  end
else
  if (USELIB==false)
    # Direct solution for all cells
    for isize = 1:NSIZE
      if (GPU>=0)
        ## @@
        ## process_size_stochastic(isize)
        process_size_stochastic(isize
        ,NFREQ, BATCH, GLOBAL, LOCAL, SK_ABS, K_ABS, S_FRAC,
        queue, Iw_buf, L1_buf, L2_buf, Tdown_buf, EA_buf, Ibeg_buf, AF_buf, ABS_buf, EMIT_buf, A_buf,
        ABSORBED, EMITTED)
      else
        process_size_stochastic_julia(isize, NFREQ, NE, SK_ABS, S_FRAC, ABSORBED, EMITTED)
      end
    end
    fp = open("dump.ARRAY", "w")
    write(fp, DUMP_ARRAY)
    close(fp)
  else   # Using library
    T, ICELLS, INT = prepare_tree(ABSORBED, NB)
    #Print(T)
    #Plot(T)
    for isize=1:NSIZE
      if (INTERP==false)
        solve_with_tree(T, ICELLS, INT, isize)
      else
        solve_with_tree_ip(T, ICELLS, INT, isize)
      end
    end
    if false
      clf()
      hist(INT[:,1], alpha=0.5, bins=200)
      hist(INT[:,2], alpha=0.5, bins=200)
      hist(INT[:,3], alpha=0.5, bins=200)
      show()
    end
  end
end

dt = time()-t0          
@printf("A2E.jl took %.2f seconds\n", dt)

