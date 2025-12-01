#!/usr/bin/env python

import pyopencl as cl
from numpy import *
import time
import sys

t00        =  time.time()
N          =  128     # size of "density" grid
NRAY       =  10000   # number of "rays"

USE_GPU    =  1    # 0 = use CPU,     1 = use GPU
ATOMICS    =  0    # 0 = no atomics,  1 = use atomic updates
if (len(sys.argv)>1):
    USE_GPU = int(sys.argv[1])
    ATOMICS = int(sys.argv[2])
    
if (USE_GPU>0):
    GLOBAL =   1024
    LOCAL  =     32
    platform   =  cl.get_platforms()[0]  # test system "0" = POCL OpenCL
    device     =  [platform.get_devices(cl.device_type.GPU)[USE_GPU-1],] # "0" = CPU
else:
    GLOBAL =     32
    LOCAL  =      4
    platform   =  cl.get_platforms()[1]  # test system "1" = NVidia
    device     =  [platform.get_devices(cl.device_type.CPU)[0],] # "0" = NVidia 5080 mobile

print(device)
context    =  cl.Context(device)
queue      =  cl.CommandQueue(context)
mf         =  cl.mem_flags

DENS       =  ones(N*N*N, float32)  # Cartesian grid of N^3 cells, input "density"
DENS_buf   =  cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=DENS)
TABS       =  zeros(N*N*N, float32) # Cartesian grid of N^3 cells, kernel output
TABS_buf   =  cl.Buffer(context, mf.READ_WRITE, 4*N*N*N)

cl.enqueue_copy(queue, TABS_buf, TABS)
cl.enqueue_copy(queue, DENS_buf, DENS)

ARGS          = " -D N=%d -D NRAY=%d -D ATOMICS=%d -D NVIDIA=%d" % (N, NRAY, ATOMICS, USE_GPU>0)
# ARGS         += " -cl-mad-enable -cl-no-signed-zeros -cl-finite-math-only" # additional optimisation

source        =  open("kernel_test5.c").read()
program       =  cl.Program(context, source).build(ARGS)
kernel_test5  =  program.test5

t0            =  time.time()
for ifreq in range(10):
    t1 = time.time()
    kernel_test5(queue, [GLOBAL,], [LOCAL,], DENS_buf, TABS_buf)
    cl.enqueue_copy(queue, TABS, TABS_buf)
    print(" %.4f seconds" % (time.time()-t1))    # timing per kernel call
queue.finish()
print("%.4f seconds total, result %.3e" % (time.time()-t00, sum(ravel(TABS))))  # total time










