#!/usr/bin/env julia

using KernelAbstractions
const KA = KernelAbstractions
using Printf
# using Statistics
using Atomix

const N      =  Int32(128)
const EPS    =  1.0f-4
const NRAY   =  Int32(10000)

t00          =  time()

if (length(ARGS)==2)
    const USE_GPU = parse(Int32, ARGS[1])  #   0 = CPU,  1 = GPU
    const ATOMICS = parse(Int32, ARGS[2])  #   0 = no atomics, 1 = with atomic updates
else
    const USE_GPU =  false
    const ATOMICS =  false
end

if USE_GPU>0
    using CUDA
    using CUDA.CUDAKernels
    BACK = CUDABackend()
    CUDA.device!(0)             #  on test system "0" = NVidia 5080 mobile
    println(CUDA.device())
    CUDA.allowscalar(false)
    const GLOBAL  =  Int64(1024)
    const LOCAL   =  Int64(32)
else
    BACK          =  CPU()      #  on test sytstem = POCL OpenCL on CPU
    const GLOBAL  =  Int32(32)
    const LOCAL   =  Int64(4)
end

const int0 = Int32(0)
const int1 = Int32(1)


@inline @inbounds function Index(x::Float32, y::Float32, z::Float32)::Int32
    if ((x<=0.0f0)|(x>=N)|(y<=0.0f0)|(y>=N)|(z<=0.0f0)|(z>=N))
        return  int0
    end
    return  Int32(1.0f0 + floor(floor(x)+N*floor(y)+N*N*floor(z)))
end


@inline @inbounds function Step(x::Float32, y::Float32, z::Float32,  u::Float32, v::Float32, w::Float32)
    ds = min(
        (u>0.0f0)  ?  ((1.0f0+EPS-mod(x,1.0f0))/u) : ((-EPS-mod(x,1.0f0))/u),
        (v>0.0f0)  ?  ((1.0f0+EPS-mod(y,1.0f0))/v) : ((-EPS-mod(y,1.0f0))/v),
        (w>0.0f0)  ?  ((1.0f0+EPS-mod(z,1.0f0))/w) : ((-EPS-mod(z,1.0f0))/w)
    )    
    x    +=  ds*u
    y    +=  ds*v
    z    +=  ds*w
    ind   =  Index(x, y, z)::Int32
    return ds, ind,  x, y, z
end


if (ATOMICS>0)
@kernel inbounds=true unsafe_indices=true function kernel_test5(@Const(DENS), TABS)
    id       =  @index(Global, Linear)              #   3.6 sec
    id      -=  1
    x  =  1.0f0 ;   y = 1.0f0 ;   z = 1.0f0  ;    u = 0.0f0 ;   v = 0.0f0 ;   w = 0.0f0 ;    ds =  0.0f0
    for iray=1:NRAY
        u = x ;     v = 0.1f0 ;      w = 0.5f0 ;
        tmp = sqrt(u*u+v*v+w*w) ;
        u  /= tmp ;   v /= tmp ;    w /= tmp ;
        x  = 39.5f0 ;  y  = 39.5f0 ;  z  = EPS ;
        ind = Index(x, y, z)::Int32        
        while(ind>=int1)
            oind = ind
            ds, ind, x, y, z   =  Step(x, y, z,  u, v, w)
            Atomix.@atomic TABS[oind] += ds            
        end
    end
end
else
@kernel inbounds=true unsafe_indices=true function kernel_test5(@Const(DENS), TABS)
    id       =  @index(Global, Linear)              #   3.6 sec
    id      -=  1
    x  =  1.0f0 ;   y = 1.0f0 ;   z = 1.0f0  ;    u = 0.0f0 ;   v = 0.0f0 ;   w = 0.0f0 ;    ds =  0.0f0
    for iray=1:NRAY
        u = x ;     v = 0.1f0 ;      w = 0.5f0 ;
        tmp = sqrt(u*u+v*v+w*w) ;
        u  /= tmp ;   v /= tmp ;    w /= tmp ;
        x  = 39.5f0 ;  y  = 39.5f0 ;  z  = EPS ;
        ind = Index(x, y, z)::Int32        
        while(ind>=int1)
            oind = ind
            ds, ind, x, y, z   =  Step(x, y, z,  u, v, w)
            TABS[oind] += exp(-ds)
        end
    end
end
end

K_test5!  =  kernel_test5(BACK, LOCAL)

DENS      =            ones(Float32, N*N*N)
TABS      =           zeros(Float32, N*N*N)
D_DENS    =  KA.zeros(BACK, Float32, N*N*N)
D_TABS    =  KA.zeros(BACK, Float32, N*N*N)
KA.copyto!(BACK, D_DENS, DENS)
KA.copyto!(BACK, D_TABS, TABS)

function KA_test()
    t0 =  time()
    for ifreq=1:10
        t1 = time()
        K_test5!(D_DENS, D_TABS, ndrange=(GLOBAL))
        KA.copyto!(BACK, TABS, D_TABS)
        # KA.synchronize(BACK)
        @printf("   %.4f seconds\n", time()-t1)  # run time per kernel call
    end    
end

KA_test()
@printf("%.4f seconds total -- result %.3e\n", time()-t00, sum(TABS))   # total run time




