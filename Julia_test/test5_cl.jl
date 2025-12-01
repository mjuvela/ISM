#!/usr/bin/env julia

using Printf
using Statistics
using OpenCL

const N      =  Int32(128)
const NRAY   =  10000

t00          =  time()
USE_GPU      =  1
ATOMICS      =  1

if (length(ARGS)==2)
    USE_GPU = parse(Int32, ARGS[1])  #  0 = use CPU,  1 = use GPU
    ATOMICS = parse(Int32, ARGS[2])
end

if (USE_GPU>0)
    platform = cl.platforms()[1]        # on test system "1" = NVidia
    device   = cl.devices(platform)[1]  # on test system "1" = NVidia 4080 mobile
    cl.device!(device)
    GLOBAL   =  1024
    LOCAL    =  32
else
    platform = cl.platforms()[3]        # on test system "3" = POCL OpenCL for CPU
    device   = cl.devices(platform)[1]
    cl.device!(device)
    GLOBAL   =  32
    LOCAL    =  4
end

println(device)
fp       =  open("kernel_test5.c")
source   =  read(fp, String)
# P        =  cl.Program(; source) |> cl.build!
P0       =  cl.Program(; source)
P        =  cl.build!(P0; options=@sprintf("-D ATOMICS=%d -D N=%d -D NRAY=%d -D NVIDIA=%d", ATOMICS, N, NRAY, USE_GPU))
K        =  cl.Kernel(P, "test5")

DENS     =  ones(Float32, N*N*N)
TABS     =  zeros(Float32, N*N*N)
D_DENS   =  CLArray(DENS)
D_TABS   =  CLArray(TABS)

function CL_test()
    global DENS, TABS, D_DENS, D_TABS, LOCAL, GLOBAL
    for ifreq=1:10
        t1 = time()
        clcall(K, Tuple{ CLPtr{Float32}, CLPtr{Float32}}, D_DENS, D_TABS; global_size=GLOBAL, local_size=LOCAL)
        TABS = Array(D_TABS)
        @printf(" %.5f seconds\n", time()-t1)  # run time pre kernel call
    end
end

CL_test()
@printf("%.4f seconds total -- result %.3e\n", time()-t00, sum(TABS))    # total run time
