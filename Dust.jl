#!/usr/bin/julia

using Printf
using DelimitedFiles
using PyPlot


const BOLTZMANN  =  1.3806488e-16
const PLANCK     =  6.62606957e-27
const C_LIGHT    =  29979245800.0
const H_K        =  4.799243348e-11
const AMU        =  1.6605e-24
const mH         =  1.0079*AMU

"""
FACTOR must be the same as in SOC.
SOC does simulation with photons/FACTOR
SOC saves absorption files as photons
A2E.py solves using photons*FACTOR ... same here, Iw /= FACTOR
"""
FACTOR = 1.0e20


function um2f(um)
  return 1.0e4.*C_LIGHT./um
end

function f2um(f)
  return 1.0e+4.*C_LIGHT./f
end


function PlanckIntensity(f, T)
  return  ((2.0*PLANCK).*(f./C_LIGHT).^2.0.*f) ./ (exp.(H_K.*f./T) .- 1.0)
end



"""
integrate_linpol(x, y, A, B)

Integrate function specified by discrete samples (x, y), over the interval [A, B].
We use linear interpolation between grid points.
"""
function integrate_linpol(x, y, A, B)
  # Find the first bin i where the integration interval begins.
  a = A
  if (A<=x[1])
    a = x[1]
    i = 1
  elseif (A>=x[end])
    return 0.0
  else #  A is inside the range of x, find the first bin
    i, j =  1, length(x)
    while((j-i)>4)
      k = Int32(floor(0.5*(j+i)))
      if (x[k]>=A)
        j = k
      else
        i = max(1, k)
      end
    end
    while(x[i]<A)
      i += 1
    end
    i -= 1
  end
  # integration begins somewhere in bin i, one step may extend to x[i+1]
  b   =  min(B, x[i+1])
  ya  =  y[i] + (y[i+1]-y[i])*(a-x[i])/(x[i+1]-x[i])
  yb  =  y[i] + (y[i+1]-y[i])*(b-x[i])/(x[i+1]-x[i])
  I   =  0.5*(b-a)*(ya+yb)
  while (b<min(x[end], B))
    a   =  b
    ya  =  yb
    i  +=  1
    b   =  min(x[i+1], B)
    yb  =  y[i] + (y[i+1]-y[i])*(b-x[i])/(x[i+1]-x[i])
    I  +=  0.5*(b-a)*(ya+yb)
  end
  return I
end



function LinpolWeights(x, x0::AbstractFloat)
  # Return indices of the nearest points and the weights for linear interpolation
  # Implies constant extrapolation outside the range of x
  if (x0<x[1])
    return 1, 1, 0.5, 0.5
  end
  if (x0>x[end])
    return length(x), length(x), 0.5, 0.5
  end
  i = argmin(abs.(x.-x0))
  if (x[i]>x0)
    i -= 1
  end
  #  x[i] < x0 < x[i+1] ... or x0 outside the range covered by x
  i = clamp(i,   1, length(x))
  j = clamp(i+1, 1, length(x))
  wj = 0.5
  if (j!=i)
    wj  =  (x0-x[i]) / (x[j]-x[i])
  end
  return i, j, 1.0-wj, wj
end


function Linpol_old(x, y, x0::AbstractFloat)
  # Linear interpolation with constant-value extrapolation
  if (x0<x[1])
    return y[1]
  end
  if (x0>x[end])
    return y[end]
  end
  i = argmin(abs.(x.-x0))
  if (x[i]>x0)
    i -= 1
  end
  #  x[i] < x0 < x[i+1] ... or x0 outside the range covered by x
  i = clamp(i,   1, length(x))
  j = clamp(i+1, 1, length(x))
  wj = 0.5
  if (j!=i)
    wj  =  (x0-x[i]) / (x[j]-x[i])
  end
  return wj*y[j] + (1.0-wj)*y[i]
end



function Linpol(x, y, x0::AbstractFloat)
  # Another version with binary search
  if (x0<x[1])
    return y[1]
  end
  if (x0>x[end])
    return y[end]
  end
  i, j =  1, length(x)
  while((j-i)>4)
    k = Int32(floor(0.5*(j+i)))
    if (x[k]>=x0)
      j = k
    else
      i = k
    end
    # now x0 is in the interval [x[i], x[j]]
  end
  while(x[i]<x0)
    i += 1
  end
  i -= 1
  i  = clamp(i,   1, length(x))
  j  = clamp(i+1, 1, length(x))
  wj = 0.5
  if (j!=i)
    wj  =  (x0-x[i]) / (x[j]-x[i])
  end
  return wj*y[j] + (1.0-wj)*y[i]
end



function Linpol(x, y, x0::Array{<:Real,1})
  n   = length(x0)
  res = zeros(Float64, n)
  for i=1:n
    res[i] = Linpol(x, y, x0[i])
  end
  return res
end





# Read GSET dust files

mutable struct DustO
  NAME::String                           # prefix
  TYPE::String                           # DustEM size distribution type
  nstoch::Int32                          # nstoch
  file_optical::String                   # file_optical
  file_enthalpies::String                # file_enthalpies
  file_sizes::String                     # file_sizes
  ### GRAIN_DENSITY::Float64                 # GRAIN_DENSITY
  NSIZE::Int32                           # NSIZE
  SIZE_A::Array{Float64,1}               # SIZE_A
  CRT_SFRAC::Array{Float64,1}            # CRT_SFRAC
  SIZE_F::Array{Float64,1}               # SIZE_F
  TMIN::Array{Float64,1}                 # minimum grain temperature per size bin
  TMAX::Array{Float64,1}                 # maximum grain temperature per size bin
  QNSIZE::Int32                          # QNSIZE
  QNFREQ::Int32                          # QNFREQ
  QSIZE::Array{Float64,1}                # QSIZE
  QFREQ::Array{Float64,1}                # QFREQ
  OPT::Array{Float64,3}                  # OPT
  AMIN::Float32                          # AMIN
  AMAX::Float32                          # AMAX
  C_NSIZE::Int32                         # C_NSIZE
  C_SIZE::Array{Float64,1}               # C_SIZE
  C_NTEMP::Int32                         # C_NTEMP
  C_TEMP::Array{Float32,1}               # C_TEMP
  C_E::Array{Float64,2}                  # C_E[C_NSIZE, C_NTEMP]
  C_C::Array{Float64,2}                  # C_C[C_NSIZE, C_NTEMP] --- ony when reading DustEM dust 
end



function DustO()
  DustO(
  "prefix",                              # NAME
  "none",                                # TYPE
  99,                                    # nstoch              
  "",                                    # file_optical        
  "",                                    # file_enthalpies     
  "",                                    # file_sizes          
  0,                                     # NSIZE               
  ### 0.0,                                   # GRAIN_DENSITY
  Array{Float64,1}([]),                  # SIZE_A              
  Array{Float64,1}([]),                  # CRT_SFRAC           
  Array{Float64,1}([]),                  # SIZE_F
  Array{Float64,1}([]),                  # TMIN
  Array{Float64,1}([]),                  # TMAX
  0,                                     # QNSIZE              
  0,                                     # QNFREQ              
  Array{Float64,1}([]),                  # QSIZE               
  Array{Float64,1}([]),                  # QFREQ               
  Array{Float64,3}(undef,1,1,1),         # OPT                 
  0.0,                                   # AMIN                
  0.0,                                   # AMAX                
  0,                                     # C_NSIZE             
  Array{Float64,1}([]),                  # C_SIZE              
  0,                                     # C_NTEMP             
  Array{Float64,1}([]),                  # C_TEMP              
  Array{Float64,2}(undef,1,1),           # C_E[N_SIZE, N_TEMP]
  Array{Float64,2}(undef,1,1)            # C_C[N_SIZE, N_TEMP]
  )
end




function Base.:read(D::DustO, filename)
  # Read the main dust file GSETDustO format
  fp = open(filename, "r")
  for l in eachline(fp)
    # println(l)
    s = split(l)
    if (length(s)<2)
      continue
    end
    if     (s[1]=="prefix")       D.NAME            = s[2]
    elseif (s[1]=="nstoch")       D.nstoch          = parse(Int32, s[2])
    elseif (s[1]=="optical")      D.file_optical    = s[2]
    elseif (s[1]=="enthalpies")   D.file_enthalpies = s[2]
    elseif (s[1]=="sizes")        D.file_sizes      = s[2]
    end
  end
  close(fp)
  
  # Read sizes
  fp = open(D.file_sizes)
  L  = readlines(fp)
  close(fp)
  # GSETDustO file has separate grain_density, sum(SRT_SFRAC)==1
  grain_density   =  parse(Float64, split(L[1])[1])
  d               =  readdlm(D.file_sizes, skipstart=3)
  # columns:  SIZE, S_FRAC, Tmin, Tmax
  D.NSIZE         =  size(d)[1]
  D.SIZE_A        =  d[:,1] * 1.0e-4            # [cm] -- the actual size bins used
  D.CRT_SFRAC     =  d[:,2] * 1.0               # sum(CRT_SFRAC) ==  1.0 in the file
  D.CRT_SFRAC   ./=  sum(D.CRT_SFRAC)           # 
  D.CRT_SFRAC    *=  grain_density              # WITHIN THIS SCRIPT CRT_SFRAC INCLUDES GRAIN_DENSITY !!
  D.CRT_SFRAC     =  clamp(D.CRT_SFRAC, 1.0e-40, 1.0)
  println("CRT_SFRAC: ")
  println(D.CRT_SFRAC)
  D.SIZE_F        =  d[:,1] * 0.0               # NOT USED
  D.TMIN          =  d[:,3] * 1.0
  D.TMAX          =  d[:,4] * 1.0
  # Note: CRT_SFRAC implies logarithmic size grid, values are the actual number of grains per bin
  
  # Read optical data
  fp = open(D.file_optical)
  L  = readlines(fp)
  close(fp)
  D.QNSIZE  =  parse(Int32, split(L[1])[1])
  D.QNFREQ  =  parse(Int32, split(L[1])[2])
  D.QSIZE   =  zeros(Float64, D.QNSIZE)
  D.QFREQ   =  zeros(Float64, D.QNFREQ)
  D.OPT     =  zeros(Float64, (D.QNSIZE, D.QNFREQ, 4))
  row       =  2    # first row, contains the current size
  for isize=1:D.QNSIZE
    D.QSIZE[isize] = parse(Float64, split(L[row])[1]) * 1.0e-4  # grain size [cm]
    A              = pi*D.QSIZE[isize]^2.0               # grain area [cm^2]
    row           += 2                                   # skip the header
    for ifreq=1:D.QNFREQ   # gset file = in increasing order of frequency
      s = split(L[row])
      D.OPT[isize, ifreq, :] = [ parse(Float64,s[1]), parse(Float64,s[2]), parse(Float64,s[3]), parse(Float64,s[4]) ]
      ## @printf("isize=%3d  ifreq=%3d   OPT %10.3e %10.3e %10.3e %10.3e\n", isize, ifreq, 
      ## D.OPT[isize, ifreq,1], D.OPT[isize, ifreq,2], D.OPT[isize, ifreq,3], D.OPT[isize, ifreq,4])
      row += 1
    end  # for ifreq
  end    # for isize
  D.QFREQ =   D.OPT[1,:,1]   # already [Hz], in order of increasing frequency
  D.AMIN  =   D.SIZE_A[1]
  D.AMAX  =   D.SIZE_A[D.NSIZE]
  @printf("--- SIZE DISTRIBUTION SIZE_A  %12.4e - %12.4e\n", D.SIZE_A[1], D.SIZE_A[D.NSIZE])
  @printf("--- OPTICAL DATA      SIZE    %12.4e - %12.4e\n", D.QSIZE[1], D.QSIZE[D.QNSIZE])
  if (D.AMIN<D.QSIZE[1])
    @printf("*** WARNING: AMIN %12.4e < %12.4e = AMIN of optical data\n", D.AMIN, D.QSIZE[1])
    scale  =   (D.AMIN/D.QSIZE[1])^2.0
    D.OPT[1,:,2]  *=  scale        # Kabs, not yet scaled with pi*a^2
    D.OPT[1,:,3]  *=  scale        # Ksca, not yet scaled with pi*a^2
    D.QSIZE[1]     =  D.AMIN       # optical data "extrapolated" tp D.AMIN
  end
  if (D.AMAX>D.QSIZE[end])
    @printf("*** ERROR: AMAX %12.4e > %12.4e = maximum grain size in arrays of optical data", D.AMAX, D.QSIZE[end])
    exit(0)
  end
  
  # read enthalpies as D.CT, D.CE
  fp = open(D.file_enthalpies)
  L  = readlines(fp)
  close(fp)
  # first non-comment line has C_NSIZE, number of grain sizes for enthalpy data
  row = 1
  while (L[row][1:1]=="#")
    row += 1
  end
  D.C_NSIZE = parse(Int32, split(L[row])[1])
  @printf("C_NSIZE %d\n", D.C_NSIZE)
  D.C_SIZE  = zeros(Float64, D.C_NSIZE)
  row += 1
  for j =1:D.C_NSIZE
    D.C_SIZE[j] = parse(Float64, split(L[row])[1])
    row += 1
  end
  D.C_SIZE *= 1.0e-4   # um -> cm
  # number of temperatures in the enthalpy table
  D.C_NTEMP = parse(Int32, split(L[row])[1])
  @printf("C_NTEMP %d\n", D.C_NSIZE)
  D.C_TEMP = zeros(Float64, D.C_NTEMP)
  row += 1
  for j=1:D.C_NTEMP
    D.C_TEMP[j] = parse(Float64, split(L[row])[1])
    row += 1
  end
  # the rest of the file is the array E[C_NSIZE, C_NTEMP]
  fp     =  open(D.file_enthalpies)
  D.C_E  =  readdlm(fp, skipstart=row-1) 
  if (size(D.C_E)[1]!=D.C_NSIZE)
    @printf("File %s: incorrect number of rows in self.C_E: %d != %d\n", D.file_enthalpies, size(D.C_E)[1], D.C_NSIZE)
    exit(0)
  end
  if (size(D.C_E)[2]!=D.C_NTEMP)
    @printf("File %s: incorrect number of columns in self.C_E: %d != %d\n", D.file_enthalpies, size(D.C_E)[2], D.C_NTEMP)
    exit(0)
  end    
end




"""
   read_DE(filename, FREQ)

Read dusts from DustEM file.
"""
function read_DE(filename)
  # Read dusts from a DustEM file
  DUSTS  = Array{DustO,1}([])
  DE_dir = filename[1:(-1+findlast("/",filename)[1])]    # drop filename
  DE_dir = DE_dir[1:(-1+findlast("/",  DE_dir  )[1])]    # drop /data
  fp     = open(filename, "r")
  for l in eachline(fp)
    s = split(l)
    if ((length(s)<5) || (l[1:1]=="#"))
      continue
    end
    push!(DUSTS, make_dustem_dust(DE_dir, s))
  end
  return DUSTS
end



"""
   make_dustem_dust(DE_dir, s, FREQ)

Make dustem dust, given the path to DustEM installation and the
list of keywords defining the size dsitribution.      
"""
function make_dustem_dust(DE_dir, s)
  println("DE_dir $DE_dir")
  D        =  DustO()
  D.NAME   =  s[1]
  D.NSIZE  =  parse(Int32, s[2])
  D.TYPE   =  s[3]
  ##D.GRAIN_DENSITY = 1.0e-7    # NOT USED !!
  ##D.GRAIN_SIZE    = 1.0e-4    # NOT USED !!
  # Make the size distribution
  rmass    =  parse(Float64, s[4])
  rho      =  parse(Float64, s[5])
  D.AMIN   =  parse(Float64, s[6])
  D.AMAX   =  parse(Float64, s[7])
  # types
  #   logn   dn/dloga = exp(-(log(a/a0)/sigma)^2)           :: a0, sigma
  #   plaw   dn/da    = a^alpha                             :: alpha
  #   ed     pl *=  1, for a<at,   exp(-((a-at)/ac)^gamma)  :: alpha, at, ac, gamma 
  #   cv     pl *=  [ 1 + |z| (a/au)^eta ]^(sign(z))        :: alpha, au, z, eta
  #                                                         :: alp...gamma, au, z, eta
  if (D.TYPE=="size")  # read sizes from data/SIZE_
    filename = DE_dir * "/data/SIZE_" * D.NAME * ".DAT"
    sizefile = open(filename)
    lines    = readlines(sizefile)
    i = 1
    while (i<length(lines))
      if ( (length(split(lines[i]))>5) & (lines[i:i]!="#") )
        break
      end
      i += 1
    end
    # read the rest of the file
    d = readdlm(sizefile, skipstart=i-1)
    D.SIZE_A  =  1.0*d[:,1]
    D.SIZE_F  =  1.0*d[:,3]*D.SIZE_A^4.0   #  dn/da [1/cm2/H]
  else 
    # analytical formula for the size distribution
    D.SIZE_A = 10.0.^(range(log10(D.AMIN), stop=log10(D.AMAX), length=D.NSIZE))
    i = 8   # first parameter after amax
    for ss in split(D.TYPE, '-')
      if (ss=="logn")                   # dn/dloga ~  exp(-( log(a/a0)/sigma )^2)
        a0, sigma, i = parse(Float64, s[i]),  parse(Float64, s[i+1]), i+2
        @printf("*** logn ***  a0 %10.3e, sigma %10.3e\n", a0, sigma)
        # factor 0.5 was missing from the documentation ??
        D.SIZE_F   = exp.(- 0.5*(log.(D.SIZE_A/a0)./sigma).^2.0 )
        # convert from dn/dloga to dn/da....  *=  dloga/da = 1/a
        D.SIZE_F ./= D.SIZE_A
      elseif (ss=="plaw")               # dn/da ~ a^alpha
        @printf("*** plaw ***\n")
        alpha, i   =  parse(Float32, s[i]), i+1
        D.SIZE_F   =  D.SIZE_A.^alpha
      elseif (ss=="ed")                 #  *= exp(-( (a-at)/ac )**gamma), for a>at
        a_t, a_c, gamma, i  =  parse(Float64, s[i]), parse(Float64, s[i+1]), parse(Float64, s[i+2]), i+3
        @printf("*** ed   *** at %10.3e ac %10.3e gamma %10.3e\n", a_t, a_c, gamma)
        m             =  D.SIZE_A .> a_t
        D.SIZE_F[m] .*=  exp.(-((D.SIZE_A[m].-a_t)./a_c).^gamma)
      elseif (ss=="cv")                 #  *=  (1+|z|*(a/au)^eta)^sign(z)
        @printf("*** cv   ***\n")
        a_u, z, eta, i =  parse(Float64, s[i]), parse(Float64, s[i+1]), parse(Float64, s[i+2]), i+3
        D.SIZE_F    .*=  (1.0 .+ abs(z) .* (D.SIZE_A./a_u).^eta).^(sign.(z))
      end
    end #for ss
  end # analytical size distribution

  # The files do not specify TMIN, TMAX .... use the same we wrote to GSET dust files -- problem dependent!!
  D.TMIN  =  4.0.*ones(D.NSIZE)
  D.TMAX  =  10.0.^range(log10(2500.0), stop=log10(150.0), length=D.NSIZE)
    
  # CRT-type summation instead of integrals over size distribution
  # must use the logarithmic size grid specified in the DustEM file
  D.CRT_SFRAC    =  D.SIZE_F  .* D.SIZE_A
  vol            =  (4.0*pi/3.0) * sum( D.CRT_SFRAC .* D.SIZE_A.^3.0 )
  @printf("rmass %10.3e, rho %10.3e, vol %10.3e\n", rmass, rho, vol)
  # grain_density  =  mH*ratio / (vol*rho)
  D.CRT_SFRAC  .*=  mH*rmass/(rho*vol)  # grain density included in D.SFRAC
  # make sure none of the CRT_SFRAC values are actually zero
  D.CRT_SFRAC    =  clamp.(D.CRT_SFRAC, 1.0e-32, 1.0)
  vol = sum(D.CRT_SFRAC .* (4.0*pi/3.0) .* D.SIZE_A.^3.0)  
  @printf("Dust mass %10.3e, H mass %10.3e\n", vol*rho, mH)
  println("CRT_SFRAC: ", D.CRT_SFRAC)

  # Apply MIX, only after the initial normalisation
  mix = 0.0
  if (occursin("mix", D.TYPE))
    @printf("*** MIX ***\n")
    # number of grains at each size multiplied with a factor from /data/MIX_tt.dat
    # this is specified for points logspace(amin, amax, nsize)
    mixfile = DE_dir * "/data/MIX_" * D.NAME * ".DAT"
    mix     = readdlm(mixfile)[:]
    println("MIX ", mix)
    # using original dustem grid !!
    D.SIZE_F    .*= mix
    D.CRT_SFRAC .*= mix
  end
  # Parameter "beta" for temperature dependence cannot be used here


  # Read the optical data
  #  dustem directory /oprop/G_<dustname>.DAT
  # (1) read Qabs and Qsca
  #       D.QNSIZE    =  number of sizes for which optical data specified
  #       D.QNFREQ    =  number of frequencies
  #       D.QSIZE     =  the sizes               [um] -> [cm]
  #       D.QFREQ     =  the frequencies IN DECREASING ORDER
  #       D.OPT[QNSIZE, NFREQ, 4] = [ um, Qabs, Qsca, g ]
  # Read the common wavelength grid
  file_lambda = DE_dir * "/oprop/LAMBDA.DAT"
  DUM         = readdlm(file_lambda, skipstart=4)[:]   # [um]
  D.QFREQ     = um2f(DUM)
  D.QNFREQ    = length(D.QFREQ)          # from DustEM LAMBDA file !!
  # Read the Q data
  lines       = readlines(open(DE_dir * "/oprop/Q_" * D.NAME *".DAT"))
  iline       = 1
  while ((lines[iline][1:1]=="#")|(length(lines[iline])<2))
    iline += 1
  end
  D.QNSIZE =  parse(Int32, split(lines[iline])[1])
  D.QSIZE  =  zeros(D.QNSIZE)
  s        =  split(lines[iline+1])
  for i=1:D.QNSIZE
    D.QSIZE[i] = parse(Float64, s[i])
  end
  D.QSIZE *= 1.0e-4
  # Read Qabs and Qsca data --- in increasing order of wavelength -- DECREASING FREQUENCY == LAMBDA.DAT wavelengths
  # 
  filename =  DE_dir * "/oprop/Q_" * D.NAME * ".DAT"
  x        =  readdlm(filename, skipstart=iline+3, comments=true, comment_char='#')
  qabs     =  x[1:D.QNFREQ, :]                     # in order of increaseing wavelength
  qsca     =  x[(D.QNFREQ+1):end, :]               # in order of increaseing wavelength
  # Read the g parameters -- assume the grid is the same as for Q
  filename =  DE_dir * "/oprop/G_" * D.NAME * ".DAT"
  gHG      =  readdlm(filename, skipstart=9)       # [QNFREQ, QNSIZE]
  if (size(gHG)!=size(qabs))
    println("DustLib: mismatch between Q and g files ", size(qabs), " ", size(gHG))
    exit(0)
  end
  
  #println(qabs)
  #println(gHG)
  
  # Put data into D.OPT, converting to the order with INCREASING FREQUENCY
  # OPT[size, freq, 4] = [ freq, Kabs, Ksca, g ]
  # TO BE CONSISTENT WITH CRT AND DUSTEM, Q INTERPOLATED OVER SIZES BEFORE *= pi*a^2
  D.OPT    = zeros(D.QNSIZE, D.QNFREQ, 4)
  for isize=1:D.QNSIZE
    for ium=1:D.QNFREQ
      #                              1         2                 3                 4                
      D.OPT[isize, D.QNFREQ-ium+1, :] = Array([ D.QFREQ[ium], qabs[ium, isize], qsca[ium, isize], gHG[ium, isize] ])
    end
  end        
  reverse!(D.QFREQ)  #  OPT and QFREQ now both in the order of increasing frequency

  @printf("  --- SIZE DISTRIBUTION SIZE_A  %12.4e - %12.4e\n", D.SIZE_A[1], D.SIZE_A[end])
  @printf("  --- OPTICAL DATA      SIZE    %12.4e - %12.4e\n", D.QSIZE[1], D.QSIZE[end])
  if (D.AMIN<D.QSIZE[1])
    @printf("*** WARNING **** AMIN %12.4e < OPTICAL DATA AMIN %12.4e\n", D.AMIN, D.QSIZE[end])
    scale           = (D.AMIN/D.SIZE[1])^2.0
    D.OPT[1,:,2]   *= scale       # Kabs
    D.OPT[0,:,3]   *= scale       # Ksca
    D.SIZE[1]       = D.AMIN      # optical data "extrapolated" to D.AMIN
  end
  if (D.AMAX>D.QSIZE[end])
    @printf("*** ERROR *** AMAX %12.4e > OPTICAL DATA AMAX %12.4e\n", D.AMAX, D.QSIZE[end])
    exit(0)
  end
  @printf("AMIN - AMAX      %10.3e  %10.3e\n", D.AMIN, D.AMAX)
  @printf("D.SIZE_A         %10.3e  %10.3e\n", minimum(D.SIZE_A), maximum(D.SIZE_A))
  @printf("Q defined        %10.3e  %10.3e\n", minimum(D.QSIZE),  maximum(D.QSIZE))

  # Read the file of heat capacities
  filename = DE_dir * "/hcap/C_" * D.NAME * ".DAT"
  lines    = readlines(open(filename))
  i        = 1
  while( (lines[i][1:1]=="#") )
    i += 1
  end
  # this line has the number of sizes
  D.C_NSIZE = parse(Int32, split(lines[i])[1])
  i += 1
  # this line has the sizes... should be the same as D.QSIZE ??
  D.C_SIZE = zeros(D.C_NSIZE)
  s  =  split(lines[i])
  for isize=1:(D.C_NSIZE)
    D.C_SIZE[isize] = parse(Float64, s[isize]) * 1.0e-4   # file [um], CSIZE [cm]
  end
  i += 1
  # this line has the number of T values
  D.C_NTEMP  =  parse(Int32, split(lines[i])[1])
  # the rest, first column = log(T), other columns = C [erg/K/cm3] for each size
  #   D.ClgC[iT, iSize]
  d          =  readdlm(filename, skipstart=i)
  ClgT       =  d[:,1]
  # file has ClgC[NTEMP, NSIZE] --- we should have C_E[NSIZE, NTEMP] => transpose!!
  ClgC       =  transpose(d[:,2:end])   #  ClgC[NSIZE, NTEMP]
  D.C_TEMP   =  10.0.^ClgT   # [K]
  D.C_C      =  10.0.^ClgC   # [erg/K/cm3]    C_C[isize, itemp]
  D.C_E      =  zeros(D.C_NSIZE, D.C_NTEMP)
  if ((size(ClgC)[1]!=D.C_NSIZE)|(size(ClgC)[2]!=D.C_NTEMP))
    println("Error reading enthalpy file !!")
    println("ClgT ", size(ClgT), " ClgC ", size(ClgT), " C_NTEMP ", D.C_NTEMP, " C_NSIZE ", D.C_NSIZE)
    exit(0)
  else
    println("ClgT ", size(ClgT), " ClgC ", size(ClgT), " C_NTEMP ", D.C_NTEMP)
    println(" C_NSIZE ", D.C_NSIZE, "  C_SIZE ", size(D.C_SIZE), " C_C ", size(D.C_C))
  end
  ##
  # calculate D.C_E[C_NSIZE, C_NTEMP] by integrating over C(T)
  for isize=1:D.C_NSIZE
    x  =  1.0.*D.C_TEMP 
    y  =  ((4.0*pi/3.0) * D.C_SIZE[isize]^3.0) .* D.C_C[isize, :]
    pushfirst!(x, 0.0)
    pushfirst!(y, 0.0)
    ### println("C_C ", size(D.C_C), " C_E ", size(D.C_E), " C_TEMP ", size(D.C_TEMP))
    # now (x,y) contain one element more than C_NTEMP
    # Each E(T) is integral from T=0.0 to a value in C_TEMP
    for iT = 1:D.C_NTEMP
      D.C_E[isize, iT]  =  integrate_linpol(x, y,  0.0, D.C_TEMP[iT])  # C_E[isize, itemp]
    end
  end

  return D
  
end  # end of  --- make_dustem_dust

  





function Kabs(D::DustO, freq_in)
  # Return KABS using summation over log-spaced grid, using the array D.CRT_DFRAC
  # Kabs() =  SUM(  pi*a^2  * Qabs * S_FRAC )  
  # Note:  GRAIN_DENSITY is already multiplied into S_FRAC => S_FRAC is the number of particles, sum(S_FRAC)!=1
  nfreq = length(freq_in)
  res   = zeros(Float64, nfreq)
  for ifreq=1:nfreq
    freq = freq_in[ifreq]
    i, j, wi, wj = LinpolWeights(D.QFREQ, freq)  # for interpolation in frequency
    # @printf("%2d %2d %.3e %.3e\n", i, j, wi, wj)
    # we need to interpolate between sizes given in the file == the OPT array and the QSIZE grid
    # interpolation from QSIZE grid onto the SIZE_A grid
    # To be consistent with CRT (and dustem), interpolate Q *before* scaling with a^2
    # we do linear interpolation over these vectors, OPT[isize, ifreq, 4], last index { freq, Kabs, Ksca, g }
    yi  = Linpol(D.QSIZE, D.OPT[:, i, 2], D.SIZE_A)   # interpolation in size
    yj  = Linpol(D.QSIZE, D.OPT[:, j, 2], D.SIZE_A)   # OPT[isize, ifreq, 4]
    y   = wi*yi + wj*yj                               # interpolation in frequency
    # geometrical cross section multiplied in AFTER interpolation of Q factors!
    res[ifreq] = sum(D.CRT_SFRAC .* y .* (pi*D.SIZE_A.^2.0) ) # GRAIN_DENSITY included in CRT_SFRAC
    if (freq<D.QFREQ[1]) # do freq^2 extrapolation if we go out of QFREQ range
      res[ifreq] *= (freq/D.QFREQ[1])^2.0
    end
  end
  return res
end



function Ksca(D::DustO, freq)
  # Return KSCA, sum over the size bins
  nfreq = length(freq)
  res   = zeros(Float64, nfreq)
  for ifreq=1:nfreq
    i, j, wi, wj =  LinpolWeights(D.QFREQ, freq[ifreq])      # interpolation in frequency
    yi           =  Linpol(D.QSIZE, D.OPT[:,i,3], D.SIZE_A)  # interpolation in size
    yj           =  Linpol(D.QSIZE, D.OPT[:,j,3], D.SIZE_A)
    y            =  wi*yi + wj*yj
    res[ifreq]   =  sum(D.CRT_SFRAC.*y.*(pi.*D.SIZE_A.^2.0)) # GRAIN_DENSITY included in CRT_SFRAC
  end
  return res
end


function SKsca_Int(D::DustO, isize::Int, freq)
  # Return pi*a^2*Qsca * S_FRAC, where S_FRAC already includes GRAIN_DENSITY
  # == the scattering opacity caused by grains in a single size bin D.A_SIZE[isize]
  nfreq = length(freq)
  res   = zeros(Float64, nfreq)
  for ifreq=1:nfreq
    i, j, wi, wj  =  LinpolWeights(D.QFREQ, freq[ifreq])               # interpolation in frequency
    yi            =  Linpol(D.QSIZE, D.OPT[:, i, 3], D.SIZE_A[isize])  # interpolation in size -> Qsca
    yj            =  Linpol(D.QSIZE, D.OPT[:, j, 3], D.SIZE_A[isize])
    y             =  wi*yi + wj*yj
    res[ifreq]    =  y * (pi*D.SIZE_A[isize].^2.0)
  end
  res *= D.CRT_SFRAC[isize]  # includes GRAIN_DENSITY
  return res
end    


function SKabs_Int(D::DustO, isize::Int, freq)
  # Return pi*a^2*Qabs * S_FRAC, where S_FRAC already includes GRAIN_DENSITY
  # == the opacity caused by grains in a single size bin D.A_SIZE[isize]
  nfreq = length(freq)
  res   = zeros(Float64, nfreq)
  for ifreq=1:nfreq
    i, j, wi, wj  =  LinpolWeights(D.QFREQ, freq[ifreq])               # interpolation in frequency
    yi            =  Linpol(D.QSIZE, D.OPT[:, i, 2], D.SIZE_A[isize])  # interpolation in size
    yj            =  Linpol(D.QSIZE, D.OPT[:, j, 2], D.SIZE_A[isize])
    y             =  wi*yi + wj*yj
    res[ifreq]    =  y * (pi*D.SIZE_A[isize].^2.0)
    # do freq^2 extrapolation if freq outside the range of QFREQ
    if (freq[ifreq]<D.QFREQ[1])
      res[ifreq] *=  (freq[ifreq]/D.QFREQ[1])^2.0
    end
  end
  res *= D.CRT_SFRAC[isize]
  return res
end    


function SG_Int(D::DustO, isize::Int, freq)
  # Parameter g for given size bin, given frequencies
  nfreq = length(freq)
  res   = zeros(Float64, nfreq)
  for ifreq=1:nfreq
    i, j, wi, wj  =  LinpolWeights(D.QFREQ, freq[ifreq])               # interpolation in frequency
    yi            =  Linpol(D.QSIZE, D.OPT[:, i, 4], D.SIZE_A[isize])  # interpolation in size
    yj            =  Linpol(D.QSIZE, D.OPT[:, j, 4], D.SIZE_A[isize])  #  { freq, Kabs, Ksca, g }
    res[ifreq]    =  wi*yi + wj*yj                                     # g, current frequency, current size
  end
  return res
end    



function SKabs(D, isize::Int, freq)
  # Return pi*a^2*Qabs
  # Note:   SKabs_Int() == SKabs() * S_FRAC,  S_FRAC including GRAIN_DENSITY
  nfreq = length(freq)
  res   = zeros(Float64, nfreq)
  for ifreq=1:nfreq
    i, j, wi, wj  =  LinpolWeights(D.WFREQ, freq[ifreq])
    yi            =  Linpol(D.QSIZE, D.OPT[:,i,2], D.SIZE_A)
    yj            =  Linpol(D.QSIZE, D.OPT[:,j,2], D.SIZE_A)
    res[ifreq]    =  (wi*yi+wj*yj) *  (pi*D.SIZE_A[isize]^2)  # pi*a^2 * Qabs
    # do freq^2 extrapolation if freq outside the range of QFREQ
    if (freq[ifreq]<D.QFREQ[1])
      res[ifreq] *=  (freq[ifreq]/D.QFREQ[1])^2.0
    end
  end
  return res
end



function GetG(D::DustO, freq)
  # Return g as average over grain sizes
  nfreq = length(freq)
  res   = zeros(Float64, nfreq)
  for ifreq=1:nfreq
    W  = 0.0
    for isize=1:D.NSIZE
      w            =  SKsca_Int(D, isize, freq[ifreq])[1]      # already includes SFRAC and a^2
      W           +=  w
      res[ifreq]  +=  w * SG_Int(D, isize, freq[ifreq])[1]     # SG_Int == g for given size bin
    end
    res[ifreq] /= W
  end
  return res  # res[ifreq]
end



function Kabs(DUSTS::Array{DustO,1}, freq)
  # Kabs for a set of dusts
  res = zeros(Float64, length(freq))
  for D in DUSTS
    res .+= Kabs(D, freq)
  end
  return res
end


function Ksca(DUSTS::Array{DustO,1}, freq)
  # Ksca for a set of dusts
  res = zeros(Float64, length(freq))
  for D in DUSTS
    res .+= Ksca(D, freq)
  end
  return res
end


function GetG(DUSTS::Array{DustO,1}, freq)
  # g parameter as a weighted average over a set of dusts
  res = zeros(Float64, length(freq))
  W   = zeros(Float64, length(freq)) .+ 1.0e-60
  for D in DUSTS
    w     =       Ksca(D, freq)
    res .+=  w .* GetG(D, freq)
    W   .+=  w
  end
  res ./=  W
  return res
end




function HG_per_Omega(theta, g)
  # Henyey-Greenstein scattering function, probability per solid angle  d(Omega).
  # Integral over 4pi solid angle of HG_per_omega == 1.0
  return (1.0/(4.0*pi)) * (1.0-g*g) ./ (1.0.+g*g.-2.0.*g.*cos.(theta)).^1.5
end


function HG_per_theta(theta, g)
  # Henyey-Greenstein scattering function,  probability per d(theta)
  # Integral [0,2*pi] of HG_per_theta * dtheta  == 1.0
  return 2.0*pi .* sin.(theta) .* HG_per_Omega(theta, g)
end



function HG_per_mu(mu, g)
  # Henyey-greenstein scattering function, probability per d(cos_theta), cos_theta=mu
  # Integral [-1.0,1.0]  HG_per_cos_theta dmu == 1.0
  return 0.5*(1.0-g*g) ./ (1.0.+g*g.-2.0*g*mu).^1.5
end



function DSF(D::DustO, freq, bins::Integer=2500)
  # Convertion from  mu=cos(theta) to  dp/dOmega
  # DSF is a vector containing probabilities per unit solid angle, 
  #     indices of DSF vector ~ cos(theta) from -1.0 to +1.0
  # DSF is weighted average of different size bins, 
  #     weight is the scattering cross section == CRT_SFRAC*OPT[isize, ifreq, 3]*SIZE_A^2
  #   g == OPT[isize, ifreq, 4]
  # In SOC we have DSF[nfreq, bins] => in julia single dust DSF[bins, nfreq]
  println("DSF(D) ", typeof(D))
  nfreq     =  length(freq)
  res       =  1.0e-60.*ones(Float64, (bins, nfreq))
  W         =  1.0e-60.*ones(Float64, nfreq)             # weight ~ ksca(ifreq), same for all bins
  cos_theta =  Array(range(-1.0, stop=1.0, length=bins)) # grid uniform in cos_theta
  theta     =  acos.(cos_theta)
  ave_g     =  zeros(Float64, nfreq)
  for isize=1:D.NSIZE    # this goes over D.SIZE_A
    # weight Ksca and value of g must be interpolated from OPT, in size and in frequency
    i = 1
    while((D.QSIZE[i+1]<D.SIZE_A[isize])&(i<(D.QNSIZE-1)))
      i += 1     
    end
    # now     QSIZE[i] <  SIZE_A[isize]  < QSIZE[i+1] (or we extrapolate!)
    wp    =  (D.SIZE_A[isize]-D.QSIZE[i]) / (D.QSIZE[i+1]-D.QSIZE[i])  #  weight for size i+1 in OPT[]
    wp    =  clamp(wp, 0.0, 1.0)                                       #  avoids extrapolation!
    Qsca  =  (1.0-wp)*D.OPT[i,:,3] + wp*D.OPT[i+1,:,3]                 #  Qsca interpolated to the current size, QNFREQ frequencies
    gg    =  (1.0-wp)*D.OPT[i,:,4] + wp*D.OPT[i+1,:,4]                 #  g interpolated to the current size, QNFREQ frequencies
    ### gg   .=   0.4
    w     =  D.CRT_SFRAC[isize] .* D.SIZE_A[isize]^2.0  .*  Linpol(D.QFREQ, Qsca, freq) # Ksca for the current size, freq = weight of this size
    W   .+=  w     # summed weight vector [NFREQ]
    for ifreq=1:nfreq
      #  interpolate g from gg for the current frequency freq[ifreq]
      g              =  Linpol(D.QFREQ, gg, freq[ifreq])
      # @printf(" isize=%2d  ifreq=%3d  %10.3e [%10.3e,%10.3e] w=%.3e  g=%.3e\n", isize, ifreq, freq[ifreq], D.QFREQ[1], D.QFREQ[end], w[ifreq], g) 
      dsc            =  HG_per_Omega(theta, g)                               # vector[bins]
      res[:, ifreq] +=  w[ifreq] .* dsc
      ave_g[ifreq]  +=  w[ifreq] * g
    end  # for ifreq
  end    # for isize
  for i=1:bins
    res[i,:] ./= W
  end
  res     = clamp.(res, 1.0e-32, 1e32)   # avoid any zero entries in DSF
  ave_g ./= W
  #println("AVERAGE g")
  #println(ave_g)
  return res
end





function CSF(D::DustO, freq, bins=2500)
  # CSF[bins] contains values of mu = cos_theta
  # indices of CSF correspond to uniform stepping in P, from 0 to +1
  # SOC generates cos_theta values as CSF[bins*rand()]
  #  (1) generate  discretised mapping:                 uniform theta  ->   p(theta)
  #  (2) convert to cumulative probability:             uniform theta  ->   P(theta)
  #  (3) resample mu values at equidistant P values:    uniform P ~ array index -> theta ->  mu
  # Note --- it does not matter if initial axis is increasing or decreasing function of theta
  #          the mu(P) curve may thus be increasing or decreasing between [-1,+1] but both work
  #          equally for the generation of the mu values!! DustLib.py had increasing theta, decreasing mu axis...
  nfreq  =  length(freq)  
  # (1)  generate { mu, p(mu) }, p ~ weighted average of functions HG_per_cos_theta
  BINS   =  10000
  # theta grid with decreasing theta... final vector will be cos(theta)
  if (false)
    theta  =  Array(range(pi, stop=0.0, length=BINS))  #  axis with *decreasing* theta (incresing cos(theta))
  else
    theta  =  Array(range(0.0, stop=pi, length=BINS))  #  axis with *increasing* theta (incresing cos(theta))
  end
  p      =  zeros(Float64, (BINS, nfreq))            #  probability per theta (NOT PER SOLID ANGLE)
  p[end] =  1.0e-60                                  #  avoid zero p when scattering cross section is zero
  W      =  1.0e-60.*ones(Float64, nfreq)            #  W[nfreq], same for all bins
  for isize=1:D.NSIZE
    a  = D.SIZE_A[isize]
    # we need to interpolate from OPT to the current size
    i = 1
    while ((D.QSIZE[i+1]<a)&(i<(D.QNSIZE-1)))
      i += 1
    end
    # now    QSIZE[i] < a < QSIZE[i+1]
    wp   =  (a-D.QSIZE[i]) / (D.QSIZE[i+1]-D.QSIZE[i])
    wp   =  clamp(wp, 0.0, 1.0)                          # eliminate extrapolation
    Qsca =  (1.0-wp)*D.OPT[i,:,3] + wp*D.OPT[i+1,:,3]    # Qsca[QNFREQ] for the current size
    gg   =  (1.0-wp)*D.OPT[i,:,4] + wp*D.OPT[i+1,:,4]    # g[QNFREQ]    for the current size
    w    =  D.CRT_SFRAC[isize]  .* D.SIZE_A[isize]^2.0  .* Linpol(D.QFREQ, Qsca, freq) # weight ~ Ksca[nfreq]
    W  .+=  w     #  total weight [nfreq], same for all bins
    for ifreq=1:nfreq  # Henyey Greensteins for each frequency
      g             =  Linpol(D.QFREQ, gg, freq[ifreq])  # g for the current size and frequency
      pp            =  HG_per_theta(theta, g)            # vector[BINS], with decreasing theta
      p[:, ifreq]  +=  w[ifreq] .* pp
    end
  end
  for i=1:BINS
    p[i,:] ./=  W    #  combined p[bins, nfreq]  .... probability per theta, for uniform grid of BINS theta values
  end
  # convert to cumulative probability ..... P(theta>theta0), later P(mu<mu0)
  for ifreq=1:nfreq
    p[:, ifreq]  =  cumsum(p[:,ifreq])  # P(theta>theta0) == P(mu<mu0), when mu0=cos(theta0)
    p[:, ifreq] /=  p[end, ifreq]       # make sure the last element is P==1
  end
  # println(p[:,nfreq])
  mu    =  cos.(theta)                  #   mapping from  mu -> P, mu in increasing order .... P=P(<mu)
  # println("mu ", mu)
  # println("P  ", p[:,90])
  # with uniform step in cumulative probability, make new array with only bins elements, array contains mu values
  res   =  zeros(Float64, (bins, nfreq))
  P     =  Array(range(0.0, stop=1.0, length=bins))  # the final grid, bins for equidistant P values
  for ifreq=1:nfreq
    res[:, ifreq]  =  Linpol(p[:, ifreq], mu, P)     # mapping CSF[P] -> mu
  end
  return res
end






function CSF(DUSTS::Array{DustO,1}, freq, bins=2500)
  # CSF[bins] contains values of mu = cos_theta
  # indices of CSF correspond to uniform stepping in P, from 0 to +1
  # SOC generates cos_theta values as CSF[bins*rand()]
  #  (1) generate  discretised mapping:                 uniform theta  ->   p(theta)
  #  (2) convert to cumulative probability:             uniform theta  ->   P(theta)
  #  (3) resample mu values at equidistant P values:    uniform P ~ array index -> theta ->  mu
  # Note --- it does not matter if initial axis is increasing or decreasing function of theta
  #          the mu(P) curve may thus be increasing or decreasing between [-1,+1] but both work
  #          equally for the generation of the mu values!! DustLib.py had increasing theta, decreasing mu axis...
  nfreq  =  length(freq)  
  # (1)  generate { mu, p(mu) }, p ~ weighted average of functions HG_per_cos_theta
  BINS   =  10000
  # theta grid with decreasing theta... final vector will be cos(theta)
  if (false)
    theta  =  Array(range(pi, stop=0.0, length=BINS))  #  axis with *decreasing* theta (incresing cos(theta))
  else
    theta  =  Array(range(0.0, stop=pi, length=BINS))  #  axis with *increasing* theta (incresing cos(theta))
  end
  p      =  zeros(Float64, (BINS, nfreq))            #  probability per theta (NOT PER SOLID ANGLE)
  p[end] =  1.0e-60                                  #  avoid zero p when scattering cross section is zero
  W      =  1.0e-60.*ones(Float64, nfreq)            #  W[nfreq], same for all bins
  for D in DUSTS
    for isize=1:D.NSIZE
      a  = D.SIZE_A[isize]
      # we need to interpolate from OPT to the current size
      i = 1
      while ((D.QSIZE[i+1]<a)&(i<(D.QNSIZE-1)))
        i += 1
      end
      # now    QSIZE[i] < a < QSIZE[i+1]
      wp   =  (a-D.QSIZE[i]) / (D.QSIZE[i+1]-D.QSIZE[i])
      wp   =  clamp(wp, 0.0, 1.0)                          # eliminate extrapolation
      Qsca =  (1.0-wp)*D.OPT[i,:,3] + wp*D.OPT[i+1,:,3]    # Qsca[QNFREQ] for the current size
      gg   =  (1.0-wp)*D.OPT[i,:,4] + wp*D.OPT[i+1,:,4]    # g[QNFREQ]    for the current size
      w    =  D.CRT_SFRAC[isize]  .* D.SIZE_A[isize]^2.0  .* Linpol(D.QFREQ, Qsca, freq) # weight ~ Ksca[nfreq]
      W  .+=  w     #  total weight [nfreq], same for all bins
      for ifreq=1:nfreq  # Henyey Greensteins for each frequency
        g             =  Linpol(D.QFREQ, gg, freq[ifreq])  # g for the current size and frequency
        pp            =  HG_per_theta(theta, g)            # vector[BINS], with decreasing theta
        p[:, ifreq]  +=  w[ifreq] .* pp
      end
    end # for isize
  end # for D
  for i=1:BINS
    p[i,:] ./=  W    #  combined p[bins, nfreq]  .... probability per theta, for uniform grid of BINS theta values
  end
  # convert to cumulative probability ..... P(theta>theta0), later P(mu<mu0)
  for ifreq=1:nfreq
    p[:, ifreq]  =  cumsum(p[:,ifreq])  # P(theta>theta0) == P(mu<mu0), when mu0=cos(theta0)
    p[:, ifreq] /=  p[end, ifreq]       # make sure the last element is P==1
  end
  # println(p[:,nfreq])
  mu    =  cos.(theta)                  #   mapping from  mu -> P, mu in increasing order .... P=P(<mu)
  # println("mu ", mu)
  # println("P  ", p[:,90])
  # with uniform step in cumulative probability, make new array with only bins elements, array contains mu values
  res   =  zeros(Float64, (bins, nfreq))
  P     =  Array(range(0.0, stop=1.0, length=bins))  # the final grid, bins for equidistant P values
  for ifreq=1:nfreq
    res[:, ifreq]  =  Linpol(p[:, ifreq], mu, P)     # mapping CSF[P] -> mu
  end
  return res
end







function DSF(DUSTS::Array{DustO,1}, freq, bins::Integer=2500)
  # DSF for a collection of DustO, the average weighted by the KSca of each dust
  # DSF ==  mapping  mu ->  dp/dOmega
  println("DSF(DUSTS) ", typeof(DUSTS))
  nfreq =   length(freq)
  res   =   1.0e-60.*ones(Float64, (bins, nfreq))
  W     =   1.0e-60.*ones(Float64, nfreq)       # weights change with frequency
  for idust=1:length(DUSTS)
    dsf  =  DSF( DUSTS[idust], freq, bins)      # probability per solid angle
    w    =  Ksca(DUSTS[idust], freq)
    # @printf(" --- dust %d, weight[ifreq=3] %.3e,  DSF[10,3] %10.3e\n", idust, w[3], dsc[10,3])
    W  .+=  w
    for ifreq=1:length(freq)
      res[:, ifreq]   +=  w[ifreq] .* dsf[:, ifreq]
    end
  end
  for i=1:bins
    res[i, :] ./= W
  end
  # if per-dust dsc arrays have the correct normalisation, so does the weighted average
  # SOC uses only ratios of DSF towards different directions -> normalisation should not matter anyway
  return res
end



# function CSF(DUSTS::Array{DustO,1}, freq, bins=2500)
#   # Cumulative probability for dust collection = weighted average of the cumulative probabilities
#   # of individual dust components
#   # CSF == mapping P -> mu
#   println("CSF(DUSTS)")
#   nfreq  =  length(freq)  
#   #  generate { mu, p(mu) }, p ~ weighted average of functions HG_per_cos_theta
#   BINS   =  10000
#   theta  =  Array(range(pi, stop=0.0, length=BINS))  #  axis with *decreasing* theta (incresing cos(theta))
#   p      =  zeros(Float64, (BINS, nfreq))            #  probability per theta
#   p[end] =  1.0e-60
#   W      =  1.0e-60.*ones(Float64, nfreq)            #  W[nfreq], same for all bins
#   for D in DUSTS
#     for isize=1:D.NSIZE
#       w    =  D.CRT_SFRAC[isize]  .* D.SIZE_A[isize]^2.0  .* Linpol(D.QFREQ, D.OPT[isize, :, 3], freq) # weight [nfreq]
#       W  .+=  w          #  total weight [nfreq], same for all bins
#       for ifreq=1:nfreq  # Henyey Greenstein for each frequency
#         g             =  Linpol(D.QFREQ, D.OPT[isize, :, 4], freq[ifreq])
#         pp            =  HG_per_theta(theta, g)        # vector[bins], with decreasing theta
#         p[:, ifreq]  +=  w[ifreq] .* pp
#       end
#     end # for isize
#   end # for dust
#   for i=1:BINS
#     p[i,:] ./=  W    #  combined p[bins, nfreq]  .... probability per theta, for uniform grid of BINS theta values
#   end
#   for ifreq=1:nfreq
#     p[:, ifreq]  =  cumsum(p[:,ifreq])  # P(theta>theta0) == P(mu<mu0), when mu0=cos(theta0)
#     p[:, ifreq] /=  p[end, ifreq]       # make sure the last element is P==1
#   end
#   mu    =  cos.(theta)                  #   mapping from  mu -> P, mu in increasing order .... P=P(<mu)
#   # with uniform step in cumulative probability, make new array with only bins elements, array contains mu values
#   res   =  zeros(Float64, (bins, nfreq))
#   P     =  Array(range(0.0, stop=1.0, length=bins))  # the final grid, bins for equidistant P values
#   for ifreq=1:nfreq
#     res[:, ifreq]  =  Linpol(p[:, ifreq], mu, P)     # mapping CSF[P] -> mu
#   end
#   return res  
# end




function write_DSF_CSF(D, filename, freq; bins=2500)
  # Write DSF and CSF to a file, as required by SOC
  # Note:  D::DustO  =>  scattering functions of this single dust
  #        D::Array{DustO,1}  =>  scattering functions for the weighted average of all dusts in D
  println("write_DSF_CSF ", typeof(D))
  dsf  =  DSF(D, freq, bins)   #  Dsc[bins, nfreq]   in Python [nfreq, bins]
  csf  =  CSF(D, freq, bins)   #  Csf[bins, nfreq]   in Python [nfreq, bins]  
  fp   =  open(filename, "w")
  write(fp, Array{Float32}(dsf))
  write(fp, Array{Float32}(csf))
  close(fp)
end





function write_simple_dust(D, filename, freq)
  # Write simple-dust file for SOC   ----- 
  # ??? lowest frequencies have constant Q values ???  ok, input file did not contain lower values
  #     ...  but we could do linear extrapolation lambda^-2 for the longer wavelengths ???
  # Note: D::DustO or D::Array{DustO,1}
  R = 1.0e-7
  a = 1.0e-4
  nfreq = length(freq)
  fp = open(filename, "w")
  @printf(fp, "eqdust\n")
  @printf(fp, "%12.4e\n", R)
  @printf(fp, "%12.4e\n", a)
  @printf(fp, "%d\n", nfreq)
  for ifreq=1:nfreq
    Qabs = Kabs(D, freq[ifreq])[1] / (R*pi*a^2.0)
    Qsca = Ksca(D, freq[ifreq])[1] / (R*pi*a^2.0)
    g    = GetG(D, freq[ifreq])[1]
    @printf(fp, "%12.6e  %8.5f  %12.6e %12.6e\n", freq[ifreq], g, Qabs, Qsca)
  end
end



"""
E2T(DUST, isize, E)

Convert energies to dust temperature.
# Arguments    
- DUST::DustO    :  the dust object
- isize::Int     :  the number of the size bin [1, DUST.NSIZE]
- E              :  scalar or array of enthalpies
"""
function E2T(D::DustO, isize::Int, E)
  a             =  D.SIZE_A[isize]
  # interpolate in size -> Etmp
  i, j, wi, wj  =  LinpolWeights(D.C_SIZE, a)    # interpolation in size
  # C_E[C_NSIZE, C_NTEMP]     C_SIZE[C_NSIZE]
  Etmp          =  wi*(D.C_E[i,:]./D.C_SIZE[i]^3.0) .+  wj.*(D.C_E[j,:]./D.C_SIZE[j]^3.0)  # [C_NTEMP]
  Etmp        .*=  a^3.0     # interpolated to the current size
  if (false)
    clf()
    plot(D.C_TEMP, D.C_E[i,:], "-")
    show()
  end
  # interpolation in energy
  return Linpol(Etmp, D.C_TEMP, E)  # interpolated values from C_TEMP
end



"""
T2E(DUST, isize, E)

Convert temperatures to enthalpies
# Arguments    
- DUST::DustO    : the dust object
- isize::Int     : the index of the size bin
- E              : enthalpies (scalar or array)
"""
function T2E(D::DustO, isize::Int, T)
  # we interpolate from C_SIZE, C_E
  a            =  D.SIZE_A[isize]
  i, j, wi, wj =  LinpolWeights(D.C_SIZE, a)  # interpolation in size
  #  D.C_E[D.C_NSIZE, D.C_NTEMP]
  Etmp         =  wi*(D.C_E[i,:]./D.C_SIZE[i]^3.0) + wj*(D.C_E[j,:]./D.C_SIZE[j]^3.0)
  Etmp        *=  a^3.0
  # interpolation in temperture... not E(T) but T(E)
  return  Linpol(D.C_TEMP, Etmp, T)  
end




# Routines to write solver file


"""
PrepareTdown(queue, [GLOBAL,], [LOCAL,], NFREQ, FREQ_buf, Ef_buf, SKABS_buf, NE, E_buf, T_buf, Tdown_buf)
FREQ_buf    <--  FREQ
Ef_buf      <--  PLANCK*FREQ
SKABS_buf   <--  DUST.SKabs_Int(isize, FREQ) / DUST.CRT_SFRAC[isize]
"""

function PrepareTdown(
  # Prepara cooling rates u -> u-1 into Tdown[NFREQ], a single grain size
  DUST::DustO, 
  E::Array{Float64,1},                # E[NEPO], energy grid for the current size
  T::Array{Float64,1},                # T[NEPO], corresponding grain temperatures
  FREQ::Array{Float64,1},             # FREQ[NFREQ], basic frequency grid
  SKABS::Array{Float64,1}             # SKABS[NFREQ], corresponding pi*a^2*Q, current size only
  # return TDOWN:Array{Float64,2}  TDOWN[NE, NSIZE]
  )  
  Ef        =  PLANCK*FREQ
  @printf("Ef= %.2e  ... %.2e\n", Ef[1], Ef[end])
  NEPO      =  length(E)      # E = bin boundaries, E[NEPO] !!
  NE        =  NEPO-1
  NFREQ     =  length(FREQ)
  Tdown     =  zeros(Float64, NE)
  
  #   PrepareTdown(queue, [GLOBAL,], [LOCAL,], NFREQ, FREQ_buf, Ef_buf, SKABS_buf, NE, E_buf, T_buf, Tdown_buf)
  for u=2:NE    
    # @printf("u=%d <= NE=%d < NEPO = %d\n", u, NE, NEPO)
    Eu   =  0.5*(E[u  ]+E[u+1])           # at most  u+1 = NE+1 = NEPO
    El   =  0.5*(E[u-1]+E[u  ])           # at least u-1 = 1
    #Tu   =  Interpolate(NEPO, E, T, Eu)   # would be better if interpolated on log-log scale ?
    Tu   =  Linpol(E, T, Eu)
    ee0  =  0.0 
    yy0  =  0.0 
    # Integral from 0.0 to Eu of    E^3*C/(exp(E/kT)-1)
    # First the full bins until end of the bin i Ef[i+1] exceeds upper limit Eu
    I    =  0.0 
    i    =  1    # current bin
    while ((i<NFREQ) && (Ef[i+1]<Eu))
      ee0  =  Ef[i] ;                                           # energy at the beginning of the interval (bin i)
      # x    =  Interpolate(NFREQ, FREQ, SKABS, ee0/PLANCK)       # C at the beginning of the interval
      x    =  Linpol(FREQ, SKABS, ee0/PLANCK)
      yy0  =  ee0*ee0*ee0* x /(exp(ee0/(BOLTZMANN*Tu))-1.0)     # integrand at the beginning of the interval
      for j=1:8                                                 # eight subdivisions of the bin i
        ee1  =  Ef[i] + j*(Ef[i+1]-Ef[i])/8.0                   # end of the sub-bin in energy
        # x    =  Interpolate(NFREQ, FREQ, SKABS, ee1/PLANCK)     # C at the end of the interval
        x    =  Linpol(FREQ, SKABS, ee1/PLANCK)
        yy1  =  ee1*ee1*ee1 * x / (exp(ee1/(BOLTZMANN*Tu))-1.0) # integrand at the end of the sub-bin
        I   +=  0.5*(ee1-ee0)*(yy1+yy0)                         # Euler integral of the sub-bin
        ee0  =  ee1                                             # next sub-bin starts at the end of previous 
        yy0  =  yy1
      end
      i += 1   # we have completed the integral till the beginning of the bin i ... now in ee0
    end
    # We have completed the integral up to the beginning of a bin that ends beyond Eu
    # The last partial step from Ef[i] to Eu
    if (Eu<Ef[NFREQ])
      for j=1:8
        ee1  =  Ef[i] + j*(Eu-Ef[i])/8.0
        #x    =  Interpolate(NFREQ, FREQ, SKABS, ee1/PLANCK)
        x    =  Linpol(FREQ, SKABS, ee1/PLANCK)
        yy1  =  ee1*ee1*ee1 * x / (exp(ee1/(BOLTZMANN*Tu))-1.0)
        I   +=  0.5*(ee1-ee0)*(yy1+yy0)
        ee0  =  ee1
        yy0  =  yy1
      end
    end
    # I *= 8.0*PI/((Eu-El)*C_LIGHT*C_LIGHT*PLANCK*PLANCK*PLANCK) ;
    # Warning:   8.0*PI/(C_LIGHT*C_LIGHT*PLANCK*PLANCK*PLANCK)  = 9.612370e+58
    I   *=  9.612370e+58 / (Eu-El) ;
    Tdown[u] = I ;
  end
  return Tdown
end



"""
PrepareIw(queue, [GLOBAL,], [LOCAL,], NFREQ, NE, Ef_buf, E_buf, L1_buf, L2_buf, Iw_buf, wrk_buf, noIw_buf)

Ef_buf   <-   PLANCK*FREQ
E_buf    <-   T2E(isize, T),    T ~  Tmin ... Tmax

"""



function Heating(
  DUST::DustO,
  E::Array{Float64, 1},         # E[NEPO], energy grid for the current size
  FREQ::Array{Float64,1}        # Ef[NFREQ], FREQ*PLANCK
  )  
  # Precalculate integration weights for upwards transitions. Eq. 15, (16) in Draine & Li (2001)
  # With the weight precalculated, the (trapezoidal numerical) integral of eq. 15 is a dot product of the vectors
  # containing the weights and the absorbed photons (=Cabs*u), which can be calculated with (fused) multiply-add.
  # TODO: fix the treatment of the highest bin, although it probably won't make any difference...

  NEPO  =  length(E)
  NE    =  NEPO-1
  NFREQ =  length(FREQ)
  Ef    =  PLANCK*FREQ

  L1    =       zeros(Int32, (NE, NE))
  L2    =  -9999*ones(Int32, (NE, NE))
  Z     =  zeros(Float64, NFREQ, NE, NE)    # [NFREQ, u, l]
  no    =  0                                # number of integration weights
    
  if (false)
    for i=1:NFREQ
      @printf("Ef[%3d] = %12.4e\n", i, Ef[i])
    end
    for i=1:NE
      @printf("E[%3d]  = %12.4e\n", i, E[i])
    end
  end

  
  for l=1:(NE-1)      
    for u=(l+1):NE      
      
      
      El    =  0.5*(E[l]+E[l+1])
      dEl   =  E[l+1] - E[l]    
      Eu    =  0.5*(E[u]+E[u+1])  # E[NEPO]
      dEu   =  E[u+1] - E[u]      
      W1    =  E[u]   - E[l+1] 
      W2    =  min( E[u]-E[l],  E[u+1]-E[l+1] ) 
      W3    =  max( E[u]-E[l],  E[u+1]-E[l+1] ) 
      W4    =  E[u+1] - E[l]

      if (false)
        if (l==1) 
          @printf("u=%3d  W=  %12.4e %12.4e %12.4e %12.4e\n", u, W1, W2, W3, W4)
        end
      end
      
      if ((Ef[1]>W4)||(Ef[end]<W1))
        L1[u,l] = -1000
        L2[u,l] = -9999
        continue
      end
      
      coeff =  1.0 / (Eu-El)  / (PLANCK*FACTOR)
      
      # find bin index i such that Ef[i] < W1 < Ef[i+1]
      i = 1
      while ((i<NFREQ) && (Ef[i]<W1))
        i += 1
      end
      i = max(i-1,1)      
      

      # W1-W2 ==========================================================================================
      a      =  clamp(W1, Ef[i], Ef[i+1])
      b      =  clamp(W2, a,     Ef[i+1])
      alpha  =  (a-Ef[i])/(Ef[i+1]-Ef[i])
      beta   =  (b-Ef[i])/(Ef[i+1]-Ef[i])
      if (false)
        # if we evaluate function at E[i] and do linear interpolation
        G1             =  (Ef[i  ]-W1)/dEl  # these are part of the interpolated function, defined at E[i] only
        G2             =  (Ef[i+1]-W1)/dEl
        Z[i,   u, l]  +=  0.5*(b-a)*(2.0-alpha-beta) * G1 * coeff * Ef[i  ]
        Z[i+1, u, l]  +=  0.5*(b-a)*(    alpha+beta) * G2 * coeff * Ef[i+1]
      else
        # if we evaluate G and E at the end points of the interval and derive weights only for the
        # interpolation of nabs[i] values
        G1             =  (a-W1)/dEl  # directly at the end points of the interval (if and when a, b != E[i])
        G2             =  (b-W1)/dEl
        Z[i,   u, l]  +=  0.5*(b-a)*(G1*a*(1.0-alpha) + G2*b*(1.0-beta)) * coeff 
        Z[i+1, u, l]  +=  0.5*(b-a)*(G1*a*alpha       + G2*b*beta      ) * coeff
      end
      if (b<W2)  # if step was till the end of a bin, we continue with the next bin
        i += 1
      end
      while ((i<NFREQ) && (b<W2))
        a              =  b
        G1             =  G2
        b              =  min(W2, Ef[i+1])
        alpha          =  (a-Ef[i])/(Ef[i+1]-Ef[i])
        beta           =  (b-Ef[i])/(Ef[i+1]-Ef[i])
        if (false)
          G2             =  (Ef[i+1]-W1)/dEl
          Z[i,   u, l]  +=  0.5*(b-a)*(2.0-alpha-beta) * G1 * coeff * Ef[i  ]
          Z[i+1, u, l]  +=  0.5*(b-a)*(    alpha+beta) * G2 * coeff * Ef[i+1]
        else
          G2             =  (b-W1)/dEl
          Z[i,   u, l]  +=  0.5*(b-a)*(G1*a*(1.0-alpha) + G2*b*(1.0-beta))  * coeff
          Z[i+1, u, l]  +=  0.5*(b-a)*(G1*a*alpha       + G2*b*beta      )  * coeff
        end
        if (b<W2)  # continues till the next bin
          i += 1
        end
      end      


      # W2-W3 ==========================================================================================
      while ((i<NFREQ) && (b<W3))
        a              =  b
        G1             =  G2
        b              =  clamp(W3, a, Ef[i+1])
        G2             =  min(dEl, dEu) / dEl
        alpha          =  (a-Ef[i])/(Ef[i+1]-Ef[i])
        beta           =  (b-Ef[i])/(Ef[i+1]-Ef[i])
        if (false)
          Z[i,   u, l]  +=  0.5*(b-a)*(2.0-alpha-beta) * G1 * coeff * Ef[i  ]
          Z[i+1, u, l]  +=  0.5*(b-a)*(    alpha+beta) * G2 * coeff * Ef[i+1]
        else
          Z[i,   u, l]  +=  0.5*(b-a)*(G1*a*(1.0-alpha) + G2*b*(1.0-beta)) * coeff
          Z[i+1, u, l]  +=  0.5*(b-a)*(G1*a*alpha       + G2*b*beta      ) * coeff
        end
        if (b<W3) # integration continues till the next bin
          i += 1
        end
      end      
      
      
      # W3-W4 ==========================================================================================
      while ((i<NFREQ) && (b<W4))
        a              =  b
        G1             =  G2
        b              =  clamp(W4, a, Ef[i+1])
        alpha          =  (a-Ef[i])/(Ef[i+1]-Ef[i])
        beta           =  (b-Ef[i])/(Ef[i+1]-Ef[i])
        if (false)
          G2             =  (W4-Ef[i+1])/dEl
          Z[i,   u, l]  +=  0.5*(b-a)*(2.0-alpha-beta) * G1 * coeff * Ef[i  ]
          Z[i+1, u, l]  +=  0.5*(b-a)*(    alpha+beta) * G2 * coeff * Ef[i+1]
        else
          G2             =  (W4-b)/dEl
          Z[i,   u, l]  +=  0.5*(b-a)*(G1*a*(1.0-alpha) + G2*b*(1.0-beta)) * coeff
          Z[i+1, u, l]  +=  0.5*(b-a)*(G1*a*alpha       + G2*b*beta      ) * coeff
        end
        if (b<W4)  # continue till the next bin
          i += 1
        end
      end


      # Intrabin ==========================================================================================
      if (true)
        # intrabin absorptions  --- not significant ??
        # Integral [0, dEl] of   (c/(eu-El)) (1-E/dEl)  Ef    dE
        if (u==(l+1))
          i     =  1
          b     =  Ef[1]        
          while ((i<NFREQ) && (Ef[i]<dEl))
            a     =   b
            b     =   clamp(dEl, a, Ef[i+1])
            alpha =   (a-Ef[i])/(Ef[i+1]-Ef[i])
            beta  =   (b-Ef[i])/(Ef[i+1]-Ef[i])
            if (false)
              Z[i,   u, l]  +=  0.5*(b-a)*(2.0-alpha-beta) * coeff * (1.0-Ef[i  ]/dEl) * Ef[i]
              Z[+1i, u, l]  +=  0.5*(b-a)*(    alpha+beta) * coeff * (1.0-Ef[i+1]/dEl) * Ef[i]
            else
              Z[i,   u, l]  +=  0.5*(b-a)*((1.0-a/dEl)*a*(1.0-alpha) + (1.0-b/dEl)*b*(1.0-beta))  * coeff
              Z[i+1, u, l]  +=  0.5*(b-a)*((1.0-a/dEl)*a*alpha       + (1.0-b/dEl)*b*beta      )  * coeff
            end
            i   +=  1
          end
        end
      end
      
      # check the first and last nonzero elements => L1[u,l], L2[u,l]
      fi, la = -1000, -9999
      for i=1:NFREQ
        if (Z[i, u, l]>0.0)
          la = i
          if (fi<0)
            fi = i
          end
        end          
      end
      L1[u, l] = fi-1  # ZERO-OFFSET VALUES TO L1 !!!
      L2[u, l] = la-1  # ZERO-OFFSET VALUES TO L2 !!!
      if (fi>0)
        no  +=  la-fi+1
      end


      if (false)        # DEBUGGING
        if (la>0)
          Z[la  , u, l] = 0.0
        end
        if (la>1)
          Z[la-1, u, l] = 0.0
        end
      end

      
      # @printf("%3d -> %3d   number of terms is %3d  =  %3d ... %3d\n", l, u, max(0, L2[u,l] - L1[u,l] + 1), L1[u,l], L2[u,l])
      # @printf("%5d %5d  W= %12.3e %12.3e %12.3e %12.3e   Ef= %12.3e %12.3e\n", L1[u,l], L2[u,l], W1, W2, W3, W4, Ef[1], Ef[end])
      
    end # for  u
  end # for l
  @printf("Heating -> noIw = %d\n", no)
  return no, L1, L2, Z
  
end # Heating()







function HeatingGD(
  DUST::DustO,
  E::Array{Float64, 1},         # E[NEPO], energy grid for the current size
  FREQ::Array{Float64,1}        # Ef[NFREQ], FREQ*PLANCK
  )  
  # Heating from Guhathakurta & Draine (1989)
  NEPO    =  length(E)
  NE      =  NEPO-1
  NFREQ   =  length(FREQ)
  Ef      =  PLANCK*FREQ
  L1      =   -1000*ones(Int32, (NE, NE))
  L2      =  -9999*ones(Int32, (NE, NE))
  Z       =  zeros(Float64, NFREQ, NE, NE)    # [NFREQ, u, l]
  no      =  0                                
  coeff   =  1.0 / (FACTOR * PLANCK)
  # total number of integration weights (current size)
  for l=1:(NE-1)      
    for u=(l+1):NE
      DE    =  E[u]-E[l]            # energy for absorbed photons
      dEu   =  E[u+1]-E[u]          # width of the target bin (in energy)
      
      ##### DE = clamp(DE, 1.00001*Ef[1], 0.99999*Ef[end])
      
      if ((DE>Ef[1])&&(DE<Ef[end])) # we have radiation corresponding to the transition
        i = 2
        while(Ef[i]<DE)
          i += 1
        end
        if (false)  
          # take a single, closest frequency bin
          if (abs(Ef[i-1]-DE)<abs(Ef[i]-DE))
            i -= 1
          end
          L1[u,l]    =  i-1          # 0-offset indices in L1 and L2
          L2[u,l]    =  i-1 
          Z[i,u,l]   =  dEu *coeff   # just the width of the bin in frequency units.... nabs = dn/dfreq
          no        +=  1
        else
          # take two closest frequency bins with a total weight of one
          i -= 1
          # now we have Ef[i] < DE < Ef[i+1]
          wi         =  (DE-Ef[i])/(Ef[i+1]-Ef[i])
          L1[u,l]    =  (i  )-1
          L2[u,l]    =  (i+1)-1
          Z[i,u,l]   =  (wi    )*dEu*coeff
          Z[i+1,u,l] =  (1.0-wi)*dEu*coeff
          no        +=  2
        end
      end      
    end # for  u
  end # for l
  return no, L1, L2, Z
  
end # Heating()





"""
   write_A2E_dustfile(D, NE=128)

Mainly for double checking the DustEM reader. Write a GSETDustO
file for the given dust struct D.      
"""
function write_A2E_dustfile(D::DustO, NE::Integer=128; prefix="gs_")
  # The main file
  fp = open(@sprintf("%s%s.dust", prefix, D.NAME), "w")
  @printf(fp, "gsetdust\n")
  @printf(fp, "prefix     %s\n", D.NAME)
  @printf(fp, "nstoch     99\n")
  @printf(fp, "optical    %s%s.opt\n",  prefix,D.NAME)
  @printf(fp, "enthalpies %s%s.ent\n",  prefix,D.NAME)
  @printf(fp, "sizes      %s%s.size\n", prefix,D.NAME)
  close(fp)

  # Optical properties
  fp = open(@sprintf("%s%s.opt", prefix, D.NAME), "w")
  @printf(fp, "%d %d  # NSIZE, NFREQ\n", D.QNSIZE, D.QNFREQ)
  for isize=1:D.QNSIZE
    @printf(fp, "%12.5e  # SIZE [um]\n", 1.0e4*D.QSIZE[isize])
    @printf(fp, "# FREQ      Qabs     Qsca      g\n")
    a = D.QSIZE[isize]
    for ifreq=1:(D.QNFREQ)
      freq  =      D.OPT[isize, ifreq, 1]  # when reading DustEM files, OPT was changed to order of increasing freq.
      qabs  =      D.OPT[isize, ifreq, 2]
      qsca  =      D.OPT[isize, ifreq, 3]
      g     =      D.OPT[isize, ifreq, 4]
      @printf(fp, "%12.5e %12.5e %12.5e %12.5e\n", freq, qabs, qsca, g)
    end
  end
  close(fp)

  # Grain sizes
  fp = open(@sprintf("%s%s.size", prefix, D.NAME), "w")
  @printf(fp, "%12.5e   # GRAIN_DENSITY\n", sum(D.CRT_SFRAC))
  @printf(fp, "%d %d     # NSIZE  NE\n", D.NSIZE, NE)
  tmp   =  1.0*D.CRT_SFRAC
  tmp  /=  sum(tmp)
  Tmax  =  10.0.^range(log10(2500.0), stop=log10(150.0), length=D.NSIZE)
  @printf(fp, "#   SIZE [um]      S_FRAC      Tmin [K]    Tmax [K]\n")
  for isize=1:(D.NSIZE)
    @printf(fp, "  %12.5e  %12.5e   %10.3e  %10.3e\n", 1.0e4*D.SIZE_A[isize], tmp[isize], 4.0, Tmax[isize])
  end
  close(fp)
  
  # Enthalpies
  fp = open(@sprintf("%s%s.ent", prefix, D.NAME), "w")
  @printf(fp, "# NUMBER OF SIZE     \n")
  @printf(fp, "#    { SIZES [um] }  \n")
  @printf(fp, "# NUMBER OF TEMPERATURES\n")
  @printf(fp, "#    { T }           \n")
  @printf(fp, "# E[SIZES, NT]  #  each row = one size, each column = one temperature !!\n")
  @printf(fp, "#                    \n")
  @printf(fp, "%d   #  NSIZE        \n", D.C_NSIZE)
  for isize=1:(D.C_NSIZE)
    @printf(fp, "   %12.5e\n", 1.0e4*D.C_SIZE[isize])   # write in [um]
  end
  @printf(fp, "%d   #  NTEMP\n", D.C_NTEMP)
  for iT=1:(D.C_NTEMP)
    @printf(fp, "  %12.5e\n", D.C_TEMP[iT])
  end
  # file should contain enthalpy,   C(grain) = (4*pi/3)*a^3*rho*(C/g)
  # DustEM files contain C [erg/K/cm3] => multiply by volume and integrate to T
  #    dust.CC[iT, iSize]  =   erg/K/cm3 
  for isize=1:(D.C_NSIZE)       # one row per size
    # A2E needs E(T), not C(T) !!
    for iT in range(1, stop=D.C_NTEMP)                  # each row = single size, different temperatures
      @printf(fp, " %12.5e", D.C_E[isize, iT])
    end
    @printf(fp, "\n")
  end
  close(fp)
          
end



"""
write_solver_file(dust::DustO, NE::Int, filename)

Write solver file for A2E.cpp, A2E.py, or A2E.jl (same format for all)
# Arguments
- DUST::DustO : the dust in question (defined in Dust.jl)
- NE::Int     : number of enthalpy bins (no default value)
- FREQ        : frequency vector, as used by SOC simulation
- filename    : name of the solver file to be written

"""
function write_solver_file(DUST::DustO, NE::Integer, FREQ, filename)  
  NFREQ  =  length(FREQ)
  NSIZE  =  DUST.NSIZE
  NEPO   =  NE+1
  SKABS  =  zeros(Float64, NFREQ, NSIZE)
  for isize=1:NSIZE
    SKABS[:,isize]  =  SKabs_Int(DUST, isize, FREQ)
  end
  EA     =  zeros(Float32, NE, NFREQ)   # Python EA[NFREQ, NE] .... Julia [NE, NFREQ]
  # Write the solver file ... with in Python SKABS[NSIZE, NFREQ] ..... in Julia SKABS[NFREQ, NSIZE]
  fp     =  open(filename, "w")
  write(fp, Int32(NFREQ))
  write(fp, Array{Float32}(FREQ))
  grain_density = sum(DUST.CRT_SFRAC)
  write(fp, Float32(grain_density))
  write(fp, Int32(NSIZE))
  write(fp, Array{Float32}(DUST.CRT_SFRAC/grain_density))  #  sum(SFRAC/GRAIN_DENSITY)==1, no GRAIN_DENSITY !!
  write(fp, Int32(NE))
  write(fp, Array{Float32}(SKABS))                #  SKAbs_Int()  ~   pi*a^2*Qabs * SFRAC, including GRAIN_DENSITY
  for isize=1:NSIZE
    # Create temperature & energy grid based on temperature limits in DustO
    tmin, tmax    =  DUST.TMIN[isize], DUST.TMAX[isize]
    T             =  tmin.+(tmax-tmin).*(Array(range(0.0, stop=1.0, length=NEPO))).^2.0 # T[NEPO]
    E             =  T2E(DUST, isize, T)      # T[NEPO] ->  E[NEPO]
    @printf("isize %2d  NEPO %3d   T= %.2f %.2f  E= %.2e %.2e\n", isize, NEPO, tmin, tmax, E[1], E[end])
    # Transition rates for upward transitions in energy: integration weights
    if (true)
      n, L1, L2, Z  =  Heating(  DUST, E, FREQ)
    else
      n, L1, L2, Z  =  HeatingGD(DUST, E, FREQ)
    end
    @printf(" write_solver_file isize=%2d,   noIw = %d\n", isize, n)
    write(fp, Int32(n))                       # number of non-zero integration weights
    num = 0
    for l=1:(NE-1)
      for u=(l+1):NE
        if (false)
          @printf("%3d -> %3d   L1 %d, L2 %d, NFREQ %d\n", l, u, L1[u,l], L2[u,l], NFREQ)
          if (L2[u,l]>=0)
            for ibin=L1[u,l]:L2[u,l]
              @printf(" %3d_%.2e", ibin, Z[ibin, u, l])
            end
            @printf("\n")
          end
        end
        if (L2[u,l]>=L1[u,l])          # NOTE: L1 and L2 contain 0-offset indices
          write(fp, Array{Float32, 1}(Z[ (1+L1[u,l]) : (1+L2[u,l]) , u, l]))   # integration weights themselves
          num +=  L2[u,l]-L1[u,l]+1
        end
      end
    end
    @printf("******* n=%d, num=%d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", n, num)
    write(fp, Array{Int32}(L1))     # L1[u,l] == u runs faster in the file
    write(fp, Array{Int32}(L2))     # ZERO-OFFSET INDICES !!
    # Transitions rates down in energy
    Tdown  =  PrepareTdown(DUST, E, T, FREQ, SKABS[:, isize]/DUST.CRT_SFRAC[isize])   # Tdown[NE]
    write(fp, Array{Float32}(Tdown))
    # Prepare EA[iE, ifreq]
    factor = FACTOR*4.0*pi
    for iE=1:NE
      TC       =  E2T(DUST, isize, 0.5*(E[iE]+E[iE+1]))
      EA[iE,:] =  SKABS[:, isize] .* PlanckIntensity(FREQ, TC) ./ (PLANCK*FREQ)  .* factor
    end    
    write(fp, Array{Float32}(EA))
    # Prepare Ibeg array
    Ibeg = zeros(Int32, NFREQ) # first enthalpy bin contributinh to given frequency
    for ifreq=1:NFREQ
      startind = 2
      while((0.5*(E[startind-1]+E[startind])<(PLANCK*FREQ[ifreq])) & (startind<NEPO))
        startind += 1
      end
      Ibeg[ifreq] = startind - 1  #  0-OFFSET IN THE FILE
    end
    write(fp, Array{Int32}(Ibeg))
  end
  close(fp)
end



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



"""
solve_emission(solverfile, file_absorbed, file_emitted)

Solve dust emission using the solver file - pure Julia function.
# Arguments
- solverfile       :  name of the solver file written with write_solver_file    
- file_absorbed    :  name of the file that SOC has save for absortions
- file_emitted     :  result file, emissions that can be read with SOC for map making
"""
function solve_emission(solverfile, file_absorbed, file_emitted, nstoch)  
  # Read solver file
  FP      =  open.(ARGS[1], "r")  
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
  # Open file of absorptions
  fpA     =  open(file_absorbed)
  CELLS   =  read(fpA, Int32)
  nfreq   =  read(fpA, Int32)
  if (nfreq!=NFREQ)
    @printf("*** Error in solve_emission: solver has %d frequencies, file has %d frequencies\n", NFREQ, nfreq)
    exit(0)
  end
  ABSORBED = Mmap.mmap(fpA, Matrix{Float32}, (NFREQ, CELLS))
  # Open file of emissions
  fpE      =  open(file_emitted, "w+")
  write(fpE, CELLS, NFREQ)
  EMITTED  = Mmap.mmap(fpE, Matrix{Float32}, (NFREQ, CELLS))  # FREQ faster
  EMITTED[:,:] .= 0.0
  
  for isize=1:NSIZE
    
    # Read solver parameters for the current size
    noIw  =  read(FP, Int32)      
    Iw    =  zeros(Float32, noIw)
    read!(FP, Iw)
    L1    =  zeros(Int32, NE*NE)
    read!(FP, L1)
    L2    =  zeros(Int32, NE*NE)
    read!(FP, L2)
    Tdown =  zeros(Float32, NE)
    read!(FP, Tdown)
    EA    =   zeros(Float32, NE*NFREQ)
    read!(FP, EA)
    Ibeg  =   zeros(Int32, NFREQ)
    read!(FP, Ibeg)
    
    AF     =  SK_ABS[:,isize] ./ K_ABS
    AF   ./=  S_FRAC[isize]*GD
    KABS   =  SK_ABS[:,isize] / (GD*S_FRAC[isize])
    
    if (isize>NSTOCH)  # solve as equilibrium dust
      # Equilibrium temperature dust
      #   !!! must check that the emission is correct when one has >1 size bins, all with ~same size !!!
      # Prepare lookup table for the temperatures
      TSTEP  =  1600.0./NIP
      TT     =  zeros(Float64, NIP)
      Eout   =  zeros(Float64, NIP)
      DF     =  FREQ[3:end] - FREQ[1:(end-2)]  # x[i+1] - x[i-1], lengths of intervals for Trapezoid rule
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
      kE      =  (Emax/Emin)^(1.0/(NIP-1.0))  # E[i] = Emin*pow(kE, i)
      oplgkE  =  1.0/log10(kE)
      #ip     = interp1d(Eout, TT)           # (linear) interpolation from energy to temperature
      #ip      =  interpolate((Eout,), TT, Gridded(Linear()))
      # @printf("Eout = %10.3e ... %10.3e", Emin, Emax)
      TTT     =  Array{Float32}(Linpol(Eout, TT, Emin * kE^(0::(NIP-1))))
      #  ABSORBED * AF integrated -> Ein, use lookup-table to get T
      T       =  zeros(Float32, CELLS)
      for icell=1:CELLS
        # absorbed by the current size
        sabs  =  ABSORBED[:, icell]*AF
        # integration of Ein
        Ein   =  0.0
        for i=2:NFREQ
          Ein  +=  (sabs[i]*FREQ[i]+sabs[i-1]*FREQ[i-1]) * (FREQ[i]-FREQ[i-1]) * 3.3130348e-27
        end
        # Table lookup
        iE       =  clamp(Int32(floor(oplgkE * log10(Ein/Emin))), 1, NIP-1) 
        wi       = (Emin * kE^(iE+1) - Ein) / (Emin * kE^(iE+1) - kE^iE)
        TP       =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1] 
        T[icell] =  TP
        for ifreq=1:NFREQ
          f = FREQ[ifreq]
          EMITTED[:,icell]  += (2.79639459e-20*FACTOR)*KABS[ifreq]*(f*f/(exp(4.7995074e-11*f/TP)-1.0))
        end
      end
      
    else
      
      # stochastically heated grains   --- like process_size_stochastic from A2E.jl
      sabs = ABSORBED[:, icell]
      for l=1:(NE-1)
        for u=l+1:NE
          I = 0.0
          if (L2[u,l]>0)
            for i= (1+L1[u,l]) : (1+L2[u,l])  # L1 and L2 contain 0-offset indices !!!
              I += sabs[i] * Iq[w_index] * AF[i]
              iw_index += 1
            end
          end
          L[l,u] = max(I, 0.0)    #   L[NE,NE]  l runs faster ???
        end
      end
      # bottom row
      L[NE,:] = 1.0
      # solve
      X[:]    = 0.0
      X[NE]   = 1.0
      X       = L \ X
      # Emission
      for ifreq=1:NFREQ
        I = 0.0
        for iE=1:NE
          I  +=  EA[iE, ifreq] * X[iE]
        end
        EMITTED[ifreq, icell] +=  I
      end
    end # equilibrium or stochastic
  end   # for isize
  # file EMITTED will be closed as we exit the routine
end




function test_1()
  # how does dp/dOmega look when plotted against cos(theta)
  theta = Array(range(0.0, stop=pi, length=100))
  for g in [ 0.0, 0.2, 0.4, 0.7]
    HG    = HG_per_Omega(theta, g)
    plot(cos.(theta), HG, "k-")
  end
  show()
  exit(0)
end



function test_2()
  # is it better to discretise CSF as (theta, p(theta)) or as (mu, P(mu))
  # yes - if we use cos(theta) instead of theta, large g leads to very sharp peak
  # in the forward direction => better to construct CSF using initial
  # discretisation  { theta, p(theta) }
  g       =  0.9
  theta   =  Array(range(0.0, stop=pi, length=100))
  ptheta  =  HG_per_theta(theta, g)
  mu      =  Array(range(-1.0, stop=+1.0, length=100))
  pmu     =  HG_per_mu(mu, g)
  subplot(2,2,1)
  plot(theta, ptheta, "b-")
  subplot(2,2,2)
  plot(theta, cumsum(ptheta), "b-")
  subplot(2,2,3)
  plot(mu, pmu, "r-")
  subplot(2,2,4)
  plot(acos.(mu), cumsum(pmu), "r-")
end



function test_3()
  # check that CSF and DSF routines are consistent
  bins = 100
  freq = 10.0.^Array(range(14, stop=16, length=4))
  Dsc  = DSF(D, freq, bins)
  Csf  = CSF(D, freq, bins)
  
  subplot(221)
  mu    = range(-1.0, stop=+1.0, length=bins)
  ifreq = 4
  plot(mu, Dsc[:, ifreq], "k-")
  title("DSF ==  mu -> dp/dOmega")
  
  subplot(222)
  P    = range(0.0, stop=1.0, length=bins)
  plot(P, Csf[:, ifreq], "r-")
  title("CSF == P -> mu")
  
  subplot(223)
  # plot dp/dtheta = from DSF that is dp/solid angle
  #    ==  DSF * sin(theta),    DSF ==   mu -> dp/dOmega
  mu    = Array(range(-1.0, stop=+1.0, length=bins))
  theta = acos.(mu)
  plot(theta, sin.(theta).*Dsc[:,ifreq], "k-")
  title("DSF converted to dp/dtheta")
  
  subplot(224)
  # same by generating angles from CSF:  CSF == P -> mu
  theta = zeros(Float32, 100000)
  for i=1:100000
    theta[i] =  acos( Csf[1+Int32(floor((bins*rand()))), ifreq] )
  end
  hist(theta, bins=range(0.0, stop=pi, length=50))
  title("CSF samples converted to dp/dtheta")
  
  
  
  # Test dust collection -- with twice the same dust or two different dusts
  D = DustO()
  read(D, "gs_aSilx.dust")
  #read(D, "gs_Gra.dust")
  #read(D, "gs_PAH0_MC10.dust")
  D2 = DustO()
  if (false)
    read(D2, "gs_aSilx.dust")
  else
    read(D2, "gs_amCBEx_copy1.dust")
  end
  DD = [ D2, D ]
  
  bins = 100
  freq = 10.0.^Array(range(14, stop=16, length=4))
  Dsc  = DSF(DD, freq, bins)
  Csf  = CSF(DD, freq, bins)
  
  figure(2)
  subplot(221)
  mu    = range(-1.0, stop=+1.0, length=bins)
  ifreq = 4
  plot(mu, Dsc[:, ifreq], "k-")
  title("DSF ==  mu -> dp/dOmega")
  subplot(222)
  P    = range(0.0, stop=1.0, length=bins)
  plot(P, Csf[:, ifreq], "r-")
  title("CSF == P -> mu")
  subplot(223)
  # plot dp/dtheta = from DSF that is dp/solid angle
  #    ==  DSF * sin(theta),    DSF ==   mu -> dp/dOmega
  mu    = Array(range(-1.0, stop=+1.0, length=bins))
  theta = acos.(mu)
  plot(theta, sin.(theta).*Dsc[:,ifreq], "k-")
  title("DSF converted to dp/dtheta")  
  
  figure(1)
  subplot(223)
  plot(theta, sin.(theta).*Dsc[:,ifreq], "r:")
  figure(2)
  
  subplot(224)
  # same by generating angles from CSF:  CSF == P -> mu
  theta = zeros(Float32, 100000)
  for i=1:100000
    theta[i] =  acos( Csf[1+Int32(floor((bins*rand()))), ifreq] )
  end
  hist(theta, bins=range(0.0, stop=pi, length=50))
  title("CSF samples converted to dp/dtheta")  
end



function test_4()
  function test_40()
    D     =  DustO()
    read(D, "gs_aSilx.dust")
    R     = 1.0e-7
    a     = 1.0e-4
    freq  = 1.0e9
    while (freq<1e10)
      Qabs = Kabs(D, freq)[1] / (R*pi*a^2.0)
      Qsca = Ksca(D, freq)[1] / (R*pi*a^2.0)
      g    = GetG(D, freq)[1]
      @printf("%12.5e   %8.4f  %12.4e %12.4e\n", freq, g, Qabs, Qsca)
      freq *= 1.15
    end
  end
  test40()
end



function compare_solver_files(name1, name2)
  # Read solver file -- Python
  FP      =  open(name1, "r")  
  NFREQ   =  read(FP, Int32)                # NFREQ
  FREQ    =  zeros(Float32, NFREQ)
  read!(FP,  FREQ)                               # FREQ[NFREQ]
  GD      =  read(FP, Float32)                   # GRAIN_DENSITY
  NSIZE   =  read(FP, Int32)                     # NSIZE
  S_FRAC  =  zeros(Float32, NSIZE)
  read!(FP,  S_FRAC)                             # S_FRAC
  NE      =  read(FP, Int32)                     # NE
  SK_ABS  =  zeros(Float32, NFREQ, NSIZE)        # coordinates switched !!!  FREQ runs faster !!!
  read!(FP,  SK_ABS)                             # Python SK_ABS[NSIZE, NFREQ], Julia SK_ABS[NFREQ, NSIZE]
  # Read solver file -- Julia
  fp      =  open(name2, "r")  
  nfreq   =  read(fp, Int32)                     # NFREQ
  freq    =  zeros(Float32, nfreq)
  read!(fp,  freq)                               # FREQ[NFREQ]
  gd      =  read(fp, Float32)                   # GRAIN_DENSITY
  nsize   =  read(fp, Int32)                     # NSIZE
  s_frac  =  zeros(Float32, nsize)
  read!(fp,  s_frac)                             # S_FRAC
  ne      =  read(fp, Int32)                     # NE
  sk_abs  =  zeros(Float32, nfreq, nsize)        # sk_abs[NFREQ, NSIZE]
  read!(fp,  sk_abs)                             # Python SK_ABS[NSIZE, NFREQ], Julia SK_ABS[NFREQ, NSIZE]
  
  Z     =  zeros(NFREQ, NE, NE)
  z     =  zeros(NFREQ, NE, NE)
  
  
  if (false)
    @printf("GD   %12.4e  %12.4e\n", GD, gd)
    @printf("S_FRAC\n")
    println(S_FRAC)
    println(s_frac)
    tmp   =   (SK_ABS .- sk_abs) ./ SK_ABS
    clf()
    imshow(tmp)
    colorbar()
    show()
    exit(0)
  end
  
  for isize=1:NSIZE    
    NOIW  =  read(FP, Int32)      
    IW    =  zeros(Float32, NOIW)
    @printf("NOIW .... %d\n", NOIW)
    read!(FP, IW)
    L1    =  zeros(Int32, NE*NE)
    read!(FP, L1)
    L2    =  zeros(Int32, NE*NE)
    read!(FP, L2)
    TDOWN =  zeros(Float32, NE)
    read!(FP, TDOWN)
    EA    =   zeros(Float32, NE*NFREQ) 
    read!(FP, EA)
    IBEG  =   zeros(Int32, NFREQ)
    read!(FP, IBEG)
    
    noiw  =  read(fp, Int32)
    @printf("noiw .... %d\n", noiw)
    iw    =  zeros(Float32, noiw)
    read!(fp, iw)
    l1    =  zeros(Int32, ne*ne)
    read!(fp, l1)
    l2    =  zeros(Int32, ne*ne)
    read!(fp, l2)
    tdown =  zeros(Float32, ne)
    read!(fp, tdown)
    ea    =   zeros(Float32, ne*nfreq)   # ea[NFREQ, NE]
    read!(fp, ea)
    ibeg  =   zeros(Int32, nfreq)        # ibeg[NFREQ]
    read!(fp, ibeg)
    
    @printf("***  isize=%d  NOIW %4d   noiw %4d\n", isize, NOIW, noiw)

    NE, ne = Int64(NE), Int64(ne)
    L1   = reshape(L1, NE, NE)
    L2   = reshape(L2, NE, NE)
    l1   = reshape(l1, ne, ne)
    l2   = reshape(l2, ne, ne)    
    EA   = reshape(EA, Int64(NE), Int64(NFREQ))
    ea   = reshape(ea, Int64(ne), Int64(nfreq))
    
    if (false) # COMPARE EA
      clf()
      subplot(121)
      imshow(EA)
      title("PY")
      colorbar()
      subplot(122)
      imshow(ea)
      title("JL")
      colorbar()
      show()
      exit(0)
    end


    if (false)
      # COPY IW TO REGULAR ARRAYS --- Z for python, z for julia
      B, b = 0, 0
      for lo=1:(ne-1)
        for up=(lo+1):ne
          A, a =  B+1, b+1   # first element
          B, b =  A+max(0,L2[up,lo]-L1[up,lo]), a+max(0, l2[up,lo]-l1[up,lo]) # last element
          @printf("A=%d, B=%d    L1 %d L2 %d\n", A, B, L1[up,lo], L2[up,lo])
          if (L2[up,lo]>=0)
            Z[ (1+L1[up,lo]):(1+L2[up,lo]), up, lo ] = IW[A:B]
          end
          if (l2[up,lo]>0)
            z[ (1+l1[up,lo]):(1+l2[up,lo]), up, lo ] = iw[a:b] 
          end
        end
      end
    end
    # show as images
    
    if (false)
      lo =  9
      clf()
      subplot(1,3,1)
      imshow(log10.(clamp.(Z[:, :, lo], 1e-30, 1e30)))
      colorbar()
      subplot(1,3,2)
      imshow(log10.(z[:, :, lo]))
      colorbar()
      subplot(1,3,3)
      imshow(clamp.(z[:, :, lo]./Z[:, :, lo], 1e-6, 1e6), vmin=1e-6, vmax=1e6)
      colorbar()
      show()
      exit(0)
    end
    
    if (false)  # COMPARE TDOWN
      # 2019-10-20 -- Tdown identical to A2E_pre.py (.... within << 1%, not exactly)
      if (isize==9)
        plot(TDOWN, tdown, marker="x", linestyle="")
        show()
        exit(0)
      end
    end
    

    if (true)  # COMPARE HEATING RATES
      if (isize==2)
        # Plot the integration weights for a single l -> u transition
        @printf("NOIW %d, IW %d\n", NOIW, length(IW))      
        @printf("noiw %d, iw %d\n", noiw, length(iw))      

        dNO, dno = 0, 0
        IND, ind = 1, 1
        
        for lo=1:(NE-1)
          for up=(lo+1):NE
            
            IND += dNO
            ind += dno
            
            lo0  = 1
            xvec = range(1, stop=NFREQ)
            yvec = zeros(Float32, NFREQ)
                        
            dNO    =   max(0, L2[up,lo]-L1[up,lo]+1)
            dno    =   max(0, l2[up,lo]-l1[up,lo]+1)
            
            if ((lo==lo0)&(up<=lo0+4))
              subplot(2, 2, up-lo0)
              if (L1[up,lo]>=0)
                yvec  .=   0.0
                yvec[(1+L1[up,lo]):(1+L2[up,lo])] = IW[IND:(IND+dNO-1)]
                semilogy(xvec, yvec, color="b")              # NOTE: L1, L2 HAVE 0-OFFSET VALUES
              end
              if (l2[up,lo]>=0)
                yvec  .=   0.0
                yvec[(1+l1[up,lo]):(1+l2[up,lo])] = iw[ind:(ind+dno-1)]
                semilogy(xvec, yvec, color="r")
              end
            end
          end
        end
        show()
        exit(0)
      end
    end
    
  end
end



if (length(ARGS)>0)
  tag = ARGS[1]
  if (tag=="1")
    # write simple dust file 
    D     = DustO()
    read(D, "gs_aSilx.dust")
    # read(D, "gs_Gra.dust")
    freq  = readdlm("freq.dat")[:,1]
    write_simple_dust(D, "jl_simple.dust", freq)
    write_DSF_CSF(D, "jl.dsc", freq, bins=2500)
  elseif (tag=="2")
    # write the solver file 
    D     =  DustO()
    NE    =  128
    read(D, "gs_aSilx.dust")
    # read(D, "gs_Gra.dust")    
    FREQ  =  readdlm("freq.dat")[:,1]  
    write_solver_file(D, NE, FREQ, "jl.solver")
  elseif (tag=="3")  
    compare_solver_files("py.solver", "jl.solver")
  elseif (tag=="4")  
    # read DustEM dust and write gset
    DUSTS  = read_DE("/home/mika/tt/dustem4.0_web/data/GRAIN.DAT")
    write_A2E_dustfile(DUSTS[1], NE=128)
  elseif (tag=="5")
    # write alternative solver file, starting directly with the DustEM files
    # 2019-10-31 --- yes, this is identical to the "Python -> GSETDustO files -> Julia version of solver file"
    freq  = readdlm("freq.dat")[:]
    DUSTS = read_DE("/home/mika/tt/dustem4.0_web/data/GRAIN.DAT")
    for D in DUSTS
      filename = @sprintf("jl_%s.solver", D.NAME)
      write_solver_file(D, 128, freq, filename)  
    end
  end
end


