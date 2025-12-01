import os, sys
import time
from numpy import *
from matplotlib.pylab import *

if (0):
    # Rerun timing tests

    # Python + OpenCL
    os.system('python test5.py  0  0   >  test5_py.time.G0.A0')  # CPU/GPU   NO/ATOMIC
    time.sleep(20)
    os.system('python test5.py  1  0   >  test5_py.time.G1.A0')
    time.sleep(20)
    os.system('python test5.py  0  1   >  test5_py.time.G0.A1')
    time.sleep(20)
    os.system('python test5.py  1  1   >  test5_py.time.G1.A1')
    time.sleep(20)

    # Julia + OpenCL
    os.system('julia test5_cl.jl  0  0   >  test5_cl.time.G0.A0')  # CPU/GPU   NO/ATOMIC
    time.sleep(20)
    os.system('julia test5_cl.jl  1  0   >  test5_cl.time.G1.A0')
    time.sleep(20)
    os.system('julia test5_cl.jl  0  1   >  test5_cl.time.G0.A1')
    time.sleep(20)
    os.system('julia test5_cl.jl  1  1   >  test5_cl.time.G1.A1')
    time.sleep(20)

    # Julia + KernalAbstractions
    os.system('julia test5.jl  0  0   >  test5_jl.time.G0.A0')  # CPU/GPU   NO/ATOMIC
    time.sleep(20)
    os.system('julia test5.jl  1  0   >  test5_jl.time.G1.A0')
    time.sleep(20)
    os.system('julia test5.jl  0  1   >  test5_jl.time.G0.A1')
    time.sleep(20)
    os.system('julia test5.jl  1  1   >  test5_jl.time.G1.A1')
    time.sleep(20)
    
    sys.exit()
    

    
def Read(tag, idev, iatom):
    #  tag = "_jl", "_cl", "_py"
    name = 'test5%s.time.G%d.A%d' % (tag, idev, iatom)
    print(name)
    res  = []
    for line in open(name).readlines():
        if (line.find('second')>0):
            res.append(float(line.split()[0]))
    return asarray(res, float32)
        

TAGS    = [ '_jl', '_cl', '_py' ]
LABS    = [  r'$\rm KA$', r'$\rm JL-CL$', r'$\rm PY-CL$'   ]


figure(1, figsize=(9,4))

subplots_adjust(left=0.11, right=0.96)
subplot(121)
iatom = 0   #  'no_atomics'
for idev in range(2):
    dev = [ 'CPU', 'GPU'][idev]
    for itag in range(3):
        tag  =  TAGS[itag]
        mrk  =  [ 's', 'o'][idev]
        ls   =  [':', '-', '-'][itag]        
        c    =  [ 'b', 'c', 'r' ][itag]            
        res  =  Read(tag, idev, iatom)
        lab  =  '%s: %s' % (['CPU', 'GPU'][idev], LABS[itag])
        mfc  =  ['none', 'none', c][itag]
        semilogy(arange(1, 11), res[0:10], mrk, color=c, mfc=mfc, ls=ls, label=lab)
        plot(11, res[-1])
ylim(0.05, 20)        
ylabel(r'$\Delta t \/ \/ \rm [s]$')
xlabel(r'$\rm Call$')
title("Without atomics")
legend(loc='upper right', ncols=2)

subplot(122)
iatom = 1   # 'atomics'
for idev in range(2):
    dev = [ 'CPU', 'GPU'][idev]
    ls  = ['-', '--'][idev]
    for itag in range(3):
        tag  =  TAGS[itag]
        mrk  =  [ 's', 'o'][idev]
        ls   =  [':', '-', '-'][itag]        
        c    =  [ 'b', 'c', 'r' ][itag]
        mfc  =  ['none', 'none', c][itag]
        res  =  Read(tag, idev, iatom)
        lab  =  '%s: %s' % (['CPU', 'GPU'][idev], LABS[itag])        
        semilogy(arange(1, 11), res[0:10], mrk, color=c, mfc=mfc, ls=ls, label=lab)
        plot(11, res[-1])
ylim(0.05, 20)                
ylabel(r'$\Delta t \/ \/ \rm [s]$')
xlabel(r'$\rm Call$')
title("With atomics")
legend(loc='upper right', ncols=2)

savefig("jvsp_timings.png")
show(block=True)

        
