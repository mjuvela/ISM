#!/usr/bin/julia
# 2019-04-24  julia translation of A2E.py
using OpenCL, Printf, Mmap #, PyPlot
using DelimitedFiles, Statistics

"""
Usage:   MA2E.jl  solver_files.txt   absorbed.data   emitted.data
With solver.data dumped from A2E.cpp or A2E_pre.py, convert absorbed.data to emitted.data.
solver_files.txt is a text file that lists solver files for multiple dust components that are
solved and whose emission is to be added to emitted.data.
If one is using a library the library is built to represent the sum of all dusts (=faster).

2019-04-29 
* script still assumes constant abundances
* does work for single stochastically heated population

A2E_MABU.py emit.ini           soc.absorbed  soc.emitted      15.1 seconds
MA2E.jl     solver_files.txt   soc.absorbed  tmp.emitted      


noIW=0 for aPy ???
- A2E a2e.ini soc.absorbed cpp.emitted               -->  aPy5.solver     IS OK, noIW>0 !!!!
- A2E_pre.py gs_aPy5.dust  freq.dat  aPy5.solver.P   -->  aPy5.solver.P   IS OK !!!


"""

ICELL   =  0
GPU     =  false
NSTOCH  =  999
NB      =  [ 43, 23, 13]     # bins on each level of the library tree
NB      =  [ 67, 32, 15]

BATCH   =  Int32(prod(NB))   # tree => all in a single call, constraint on BATCH
FACTOR  =  Float32(1.0e20)
DFRAC   =  0.0               # fraction left to be solved directly
USELIB  =  true

if (length(ARGS)<3)
  println("\n  A2E.jl  list_solve_files.txt  absorbed.data emitted.data [LIBRARY GPU NSTOCH]") ;  exit(1)
end
if (length(ARGS)>3)
  USELIB = (parse(Int32, ARGS[4])>0)
  if (length(ARGS)>4)
    GPU = parse(Int32, ARGS[5])
    if (length(ARGS)>5)
      NSTOCH = parse(Int32, ARGS[6])
    end
  end
end

# read the solver files
solver_files = []
fp = open(ARGS[1], "r")
while (eof(fp)==false)
  name = strip(readline(fp))
  if ((length(name)>1)&&(name[1:1]!="#"))
    push!(solver_files, strip(name))
  end
end
close(fp)
println(solver_files)

if false
  solver_files = [ "ccm20.solver","ccm20.solver" ]
  # solver_files = [ "ccm20.solver"  ]
end

# since 2018-12-29 also SOC absorbed.data starts with [ cells, nfreq ] only
fpA      = open(ARGS[2], "r+")                 # absorbed 
CELLS    = read(fpA, Int32)
NFREQ    = read(fpA, Int32)

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


t0 = time()
if (GPU>0)
  device = cl.devices(cl.CL_DEVICE_TYPE_GPU)[1]
  LOCAL  = 32
else
  device = cl.devices(cl.CL_DEVICE_TYPE_CPU)[1]
  LOCAL  = 8
end
context = cl.Context(device)
queue   = cl.CmdQueue(context)


GLOBAL      =  max(BATCH, 64*LOCAL)
if (GLOBAL%64!=0)
  GLOBAL  = Int32((floor(GLOBAL/64)+1)*64)
end

NIP         =  30000  # number of interpolation points for the lookup tables (equilibrium dust)
OPT         =  "-D LOCAL=$LOCAL -D NFREQ=$NFREQ -D CELLS=$CELLS -D NIP=$NIP"
OPT        *=  @sprintf(" -D FACTOR=%.4ef", FACTOR)

# Stochastic grain solver
source      =  open(homedir()*"/starformation/SOC/kernel_A2E_julia.c") do file read(file, String)  end
prog        =  cl.Program(context, source=source)
program     =  cl.build!(prog, options=OPT)
DoSolve     =  cl.Kernel(program, "DoSolve")


# Tree structure
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
  global NFREQ, CELLS
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
  println(CELLS, " ", size(ABSORBED), " ", size(INT1), " NFREQ ", NFREQ, " freq ", length(freq))
  for iband=1:3     # INT[CELLS, BANDS]   ABSORBED[NFREQ, CELLS]
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





function solve_stochastic_dust_with_tree(solver_files, T, ICELLS, INT, ABSORBED, EMITTED)
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
  global queue, context
  
  # Solve emission for all cases mentioned in the tree -- cases ABSORBED[:, ICELLS]
  # should be possible in a single kernel call but MUST HAVE BATCH>= NB*NB*NB = length(CELLS)
  t0       =  time()
  tmp      =  zeros(Float32, NFREQ*BATCH)
  tmp_2d   =  reshape(tmp,   Int64(NFREQ), Int64(BATCH))   # another view to the same array
  emit     =  zeros(Float32, NFREQ*BATCH)
  emit_2d  =  reshape(emit,   Int64(NFREQ), Int64(BATCH))  # another view to the same array
  batch    =  length(ICELLS)
  if (batch>BATCH)
    println("A2E.jl -- BATCH must be at least as large as the number of TREE leaves !!")
    exit(0)
  end  

  # we must first find out the total absorption cross sections, sum over all dusts and sizes, 
  # to be able to divide absorptions between dust populations (AF takes care of division inside dust population)
  TOTAL_K_ABS = zeros(Float32, NFREQ)
  for idust = 1:length(solver_files)
    println("----- ", solver_files[idust], " -----")
    FP      =  open(solver_files[idust], "r")
    NFREQ   =  read(FP, Int32)                     # NFREQ
    FREQ    =  zeros(Float32, NFREQ)
    read!(FP,  FREQ)                               # FREQ[NFREQ]
    GD      =  read(FP, Float32)                   # GRAIN_DENSITY
    NSIZE   =  read(FP, Int32)                     # NSIZE
    S_FRAC  =  zeros(Float32, NSIZE)               
    read!(FP,  S_FRAC)                             # S_FRAC[NSIZE]
    NE      =  read(FP, Int32)                     # NE
    SK_ABS  =  zeros(Float32, NFREQ, NSIZE)        # coordinates switched !!!  FREQ runs faster !!!
    read!(FP,  SK_ABS)                             # SK_ABS[NFREQ, NSIZE]
    K_ABS   =  sum(SK_ABS, dims=2)                 # sum over sizes
    TOTAL_K_ABS += K_ABS
    close(FP)
  end
  
  for idust = 1:length(solver_files)
    FP      =  open(solver_files[idust], "r")
    NFREQ   =  read(FP, Int32)                     # NFREQ
    FREQ    =  zeros(Float32, NFREQ)
    read!(FP,  FREQ)                               # FREQ[NFREQ]
    GD      =  read(FP, Float32)                   # GRAIN_DENSITY
    NSIZE   =  read(FP, Int32)                     # NSIZE
    S_FRAC  =  zeros(Float32, NSIZE)
    read!(FP,  S_FRAC)                             # S_FRAC[NSIZE]
    NE      =  read(FP, Int32)                     # NE
    println("NE = ", NE)
    SK_ABS  =  zeros(Float32, NFREQ, NSIZE)        # coordinates switched !!!  FREQ runs faster !!!
    read!(FP,  SK_ABS)                             # Python SK_ABS[NSIZE, NFREQ], Julia SK_ABS[NFREQ, NSIZE]
    K_ABS   =  sum(SK_ABS, dims=2)                 # sum over sizes
    
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
    X_buf       =  cl.Buffer(Float32, context, :w,  BATCH*NE)    # no initial values -> write only
    
    for isize = 1:NSIZE
      @printf("=== DUST[%d] %15s, SIZE %2d/%2d  ", idust, solver_files[idust], isize, NSIZE)
      AF      =  Array{Float64}(SK_ABS[:,isize]) ./ Array{Float64}(K_ABS[:])     # => E per grain
      AF    ./=  S_FRAC[isize]*GD
      AF      =  Array{Float32}(clamp.(AF, 1.0e-32, 1.0e+100))
      if true
        AF[isfinite.(AF).==false] .= 1.0e-30
      end    
      cl.write!(queue, AF_buf,   AF)      # AF
      #
      noIw  =  read(FP, Int32)                              # noIw
      @printf("noIw %5d\n", noIw)
      if (noIw>0)
        Iw    =  zeros(Float32, noIw)
        read!(FP, Iw)                                         # Iw[noIw]
        cl.write!(queue, Iw_buf, Iw)        # Iw
      end
      L1    =  zeros(Int32, NE*NE)                          
      read!(FP, L1)                                         # L1[NE*NE]
      cl.write!(queue, L1_buf,  L1)       # L1
      L2    =  zeros(Int32, NE*NE)                          
      read!(FP, L2)                                         # L2[NE*NE]
      cl.write!(queue, L2_buf,  L2)       # L2
      Tdown =  zeros(Float32, NE)
      read!(FP, Tdown)                                      # Tdown[NE]
      cl.write!(queue, Tdown_buf, Tdown)  # Tdown
      EA    =   zeros(Float32, NE*NFREQ)                   
      read!(FP, EA)                                         # EA[NE*NFREQ]
      cl.write!(queue, EA_buf,    EA)     # EA
      Ibeg  =   zeros(Int32, NFREQ)                         
      read!(FP, Ibeg)                                       # Ibeg[NFREQ]
      cl.write!(queue, Ibeg_buf,  Ibeg)   # Ibeg
      cl.finish(queue)
      #
      for i = 1:batch  # all batch leaf nodes solved in a single kernel call
        tmp_2d[:, i]  =  ABSORBED[:, ICELLS[i]] .* (K_ABS./TOTAL_K_ABS)
      end
      # @printf(" ABS = %10.3e\n", tmp_2d[100,10])
      cl.write!(queue, ABS_buf, tmp)
      cl.finish(queue)
      queue(DoSolve, GLOBAL, LOCAL,  Int32(NE), Int32(batch),   Int32(isize-1),  Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
      Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf,   X_buf)
      cl.finish(queue)
      tmp[:]   =  cl.read(queue, EMIT_buf)    # BATCH*NFREQ
      emit[:] +=  tmp[:]                      # total emission as sum over dust species
      @printf(" += %10.3e\n", tmp_2d[10,10])
      dt = time()-t0
    end # for isize
    close(FP)
    @printf("Tree solutions:    %.3f seconds, %.3e seconds per cell\n", dt, dt/prod(NB))
  end # for idust
  
  # Loop over the whole cloud, = lookup from the library, solved emission in emit_2d
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
      EMITTED[:, icell] +=  fudge .*  emit_2d[:, T.nodes[l0].idx]  # library solution - no interpolation
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
  
  if (unsolved>0)
    for idust=1:length(solver_files)    
      # again initialise CL arrays to solve dust idust
      FP      =  open(solver_files[idust], "r")
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
      X_buf       =  cl.Buffer(Float32, context, :w,  BATCH*NE)    # no initial values -> write only
      
      for isize=1:NSIZE
        AF      =  Array{Float64}(SK_ABS[:,isize]) ./ Array{Float64}(K_ABS[:])     # => E per grain
        AF    ./=  S_FRAC[isize]*GD      # "invalid value encountered in divide"
        AF      =  Array{Float32}(clamp.(AF, 1.0e-32, 1.0e+100))
        if true
          AF[isfinite.(AF).==false] .= 1.0e-30
        end    
        cl.write!(queue, AF_buf,   AF)      # AF
        noIw  =  read(FP, Int32)
        if (noIw>0)
          Iw    =  zeros(Float32, noIw)
          read!(FP, Iw)
          cl.write!(queue, Iw_buf, Iw)        # Iw
        end
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
        ###
        iun = 1  # first unsolved , index to UNSOLVED
        while (iun<=unsolved)
          # solve next <= BATCH cells from UNSOLVED
          batch = min(BATCH, unsolved-iun+1)
          for i = 1:batch
            tmp_2d[:, i]  =  ABSORBED[:, UNSOLVED[iun+i-1]]  .* (K_ABS./TOTAL_K_ABS)
          end
          cl.write!(queue, ABS_buf, tmp)
          cl.finish(queue)
          queue(DoSolve, GLOBAL, LOCAL,  Int32(NE), Int32(batch),   Int32(isize-1),  Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
          Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf,   X_buf)
          cl.finish(queue)
          emit[:]  =  cl.read(queue, EMIT_buf)        # BATCH*NFREQ
          for i = 1:batch
            EMITTED[:, UNSOLVED[iun+i-1]] += emit_2d[:,i]
          end
          iun += batch
        end  # while unsolved
      end  #for isize
      close(FP)
    end  # for idust
  end # if unsolved
  
end





function solve_equilibrium_dust(solver_files, ABSORBED, EMITTED)  

  TOTAL_K_ABS = zeros(Float32, NFREQ)
  for idust = 1:length(solver_files)
    FP      =  open(solver_files[idust], "r")
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
    TOTAL_K_ABS += K_ABS
    close(FP)
  end

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

  
  for  idust=1:length(solver_files)
    FP      =  open(solver_files[idust], "r")
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
    close(FP)
    
    for isize=1:NSIZE
      AF      =  Array{Float64}(SK_ABS[:,isize]) ./ Array{Float64}(K_ABS[:])     # => E per grain
      AF    ./=  S_FRAC[isize]*GD      # "invalid value encountered in divide"
      AF      =  Array{Float32}(clamp.(AF, 1.0e-32, 1.0e+100))
      if true
        AF[isfinite.(AF).==false] .= 1.0e-30
      end        
      if (S_FRAC[isize]<1.0e-30) 
        continue   # empty size bin
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
      cl.enqueue_copy(queue, TTT_buf,  TTT)     # NIP elements
      cl.enqueue_copy(queue, KABS_buf, KABS)    # NFREQ elements
      T = zeros(BATCH, np.float32)        
      for icell = 1:BATCH:CELLS                 # T_buf is being updated for GLOBAL cells
        b       =  min(icell+(BATCH-1), CELLS)
        tmp_abs =  ABSORBED[:, icell:b] * AF * (K_ABS/TOTAL_K_ABS)
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
    end # for isize
  end # for idust
end
    
    



function solve_stochastic_dust(solver_files, ABSORBED, EMITTED)
  global NFREQ, BATCH, GLOBAL, LOCAL
  global SK_ABS, S_FRAC
  global queue, Iw_buf, L1_buf, L2_buf, Tdown_buf, EA_buf, Ibeg_buf, AF_buf, ABS_buf, EMIT_buf, A_buf, X_buf
  TOTAL_K_ABS = zeros(Float32, NFREQ)
  for idust = 1:length(solver_files)
    FP        =  open(solver_files[idust], "r")
    NFREQ     =  read(FP, Int32)                   # NFREQ
    FREQ      =  zeros(Float32, NFREQ)
    read!(FP,  FREQ)                               # FREQ[NFREQ]
    GD        =  read(FP, Float32)                 # GRAIN_DENSITY
    NSIZE     =  read(FP, Int32)                   # NSIZE
    S_FRAC    =  zeros(Float32, NSIZE)
    read!(FP,  S_FRAC)                             # S_FRAC
    NE        =  read(FP, Int32)                   # NE
    SK_ABS    =  zeros(Float32, NFREQ, NSIZE)      # coordinates switched !!!  FREQ runs faster !!!
    read!(FP,  SK_ABS)                             # Python SK_ABS[NSIZE, NFREQ], Julia SK_ABS[NFREQ, NSIZE]
    K_ABS     =  sum(SK_ABS, dims=2)[:,1]          # sum over sizes
    TOTAL_K_ABS += K_ABS
    close(FP)
  end
  for idust=1:length(solver_files)        
    FP      =  open(solver_files[idust], "r")
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
    K_ABS   =  sum(SK_ABS, dims=2)[:,1]            # sum over sizes
    
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
    X_buf       =  cl.Buffer(Float32, context, :w,  BATCH*NE)    # no initial values -> write only
    
    for isize = 1:NSIZE
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
      tmp     =  zeros(Float32, NFREQ*BATCH)
      tmp_2d  =  reshape(tmp,   Int64(NFREQ), Int64(BATCH))
      for icell in 1:BATCH:CELLS
        t00   =  time()            
        batch =  min(BATCH, CELLS-icell+1)     # actual number of cells
        for i=1:batch
          tmp_2d[:, i]  .=  ABSORBED[:, icell+i-1] .* (K_ABS./TOTAL_K_ABS)
        end
        # @printf("Process %d:%d  ... %d cells\n", icell, icell+batch-1, batch)
        cl.write!(queue, ABS_buf, tmp)
        cl.finish(queue)
        queue(DoSolve, GLOBAL, LOCAL,  Int32(NE), Int32(batch),   Int32(isize-1),  Iw_buf,    L1_buf,  L2_buf,   Tdown_buf, EA_buf, 
        Ibeg_buf,  AF_buf,  ABS_buf,  EMIT_buf,  A_buf,   X_buf)
        cl.finish(queue)
        tmp[:]  =  cl.read(queue, EMIT_buf)        # BATCH*NFREQ
        EMITTED[:, icell:(icell+batch-1)] += tmp_2d[:, 1:batch]       # contribution of current dust, current size
        if (icell==0)
          @printf("   SIZE %2d/%2d  icell %6d/%6d  %7.2f s/%d %.3e s/cell/size",
          isize, NSIZE, icell, CELLS, time.time()-t00, batch, (time.time()-t00)/batch)
        end
      end # for icell
    end # for idust
  end # for idust
end



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


if (USELIB==false)    # Direct solution for all cells
  solve_stochastic_dust(solver_files, ABSORBED, EMITTED)
else                  # Using library
  T, ICELLS, INT = prepare_tree(ABSORBED, NB)
  solve_stochastic_dust_with_tree(solver_files, T, ICELLS, INT, ABSORBED, EMITTED)
end

dt = time()-t0          
@printf("MA2E.jl took %.2f seconds\n", dt)

