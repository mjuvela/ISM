#!/usr/bin/python
from MJ.mjDefs import *
from MJ.Aux.DustLib import *

"""
Usage:
    ASOC_driver.py  soc-ini  [uselib] [makelib]
    
    
If calculations involve stochastically heated grains:
    - creates <dust>.solver files with A2E_pre.py
    - creates <dust>_simple.dust files
    - (1) runs ASOC.py with <dust>_simple.dust files
    - (2) solves dust emission with A2E_MABU.py
    - (3) calculates maps with a second ASOC.py run
    
If calculation involve only eqdust (no stochastically heated ones),
do also the above three runs -- this is needed in case of 
spatially varying dust abundances.   

We assume that ini file contains dusts of the types simple and gs and 
that the corresponding dust files have already been created (e.g. based on DustEM).

These can be done running three routines from DustLib.py:
    write_DUSTEM_files()   -->  dustem_<name>.dust
    write_simple_dust()    -->  <name>_simple.dust, <name>.dsc
    write_A2E_dustfiles()  -->  gs_<name>.dust
(see make_dust.py).


2019-09-22: 
    For the USELIB case, if ini-file contains the keyword ofreq + filename,
    the emitted file will contain only the frequencies listed in the file ---
    ini will be read only in A2E_MABU and passed on as command line argument to A2E_LIB.
    
    Simulation with fewer frequencies is done using <dust>_lib_simple.dust.
    A2E_pre.py gs_PAH1_MC10_lib.dust freq.dat PAH1_MC10_lib.solver is an error !!!
    We will use <dust>.solver and there is no <dust>_lib.solver files !!!    
"""

if (len(sys.argv)<2):
    print("Usage:  ASOC_driver.py  soc.ini  [GPU] [uselib] [makelib] \n")
    print("    makelib - also make the library files for all dusts")
    print("    uselib  - solve emission using existing library files")
    sys.exit()

    
# Read the original ini file
INI   = sys.argv[1]
LINES = open(INI).readlines()

GPU, USELIB, MAKELIB  = 0, 0, 0
for arg in sys.argv[2:]:
    if (arg.lower()=='gpu'): GPU     = 1
    if (arg=='uselib'):      USELIB  = 1
    if (arg=='makelib'):     MAKELIB = 1
    if (arg=='MAKELIB'):     MAKELIB = 2
print(GPU, USELIB, MAKELIB)



# Make a list of the dusts
stochastic, simple = [], []   # list of dust names
fabs, femit = None, None      # names of absorption and emission files
nstoch, ndust = 0, 0          # number of stochastic and all dusts
for line in LINES:
    s = line.split()
    if (len(s)<2): continue
    if (s[0]=='optical'):
        name, ndust = s[1], ndust+1
        if (name[0:3]=='gs_'):
            stochastic.append(name.replace('_lib', ''))  # _lib is just for RT, simulation of lib frequencies only
            simple.append('%s_simple.dust' % (name[3:].split('.')[0]))
            nstoch += 1
        else:
            stochastic.append('')
            simple.append(name)        
    if (s[0]=='absorbed'): fabs  = s[1]
    if (s[0]=='emitted'):  femit = s[1]


    
print("================================================================================")
# using library =>
#   simple      =  PAH0_MC10_lib_simple.dust
#   stochastic  =  gs_PAH0_MC10.dust
print("ASOC_driver")
print("SIMPLE:")
print(simple)
print("STOCHASTIC")
print(stochastic)
# sys.exit()

"""
SIMPLE:
    ['PAH0_MC10_lib_simple.dust', 'PAH1_MC10_lib_simple.dust', 'amCBEx_lib_simple.dust', 'amCBEx_copy1_lib_simple.dust', 'aSilx_lib_simple.dust']
STOCHASTIC
    ['gs_PAH0_MC10.dust', 'gs_PAH1_MC10.dust', 'gs_amCBEx.dust', 'gs_amCBEx_copy1.dust', 'gs_aSilx.dust']    
"""
print("================================================================================")        



# Write solver file for each stochastically heated dust
for dust in stochastic:          # dust == filename for stochastically heated dust
    if (len(dust)<1): continue   # was simple dust only
    solver  =  '%s.solver' % dust[3:].split('.')[0]
    redo    =   True
    if (os.path.exists(solver)): # skip solver creation if that exists and is more recent than dust file
        if (os.stat(dust).st_mtime<os.stat(solver).st_mtime): redo = False
    if (redo):
        print("================================================================================")
        print('A2E_pre.py %s freq.dat %s' % (dust, solver))
        print("================================================================================")
        os.system('A2E_pre.py %s freq.dat %s' % (dust, solver))
    

        
# First SOC run -- gather absorptions
# Write new ini with simple dust only, only reference frequencies
# Note -- the alternative would be to write new dusts that contain only
#         the reference frequencies... but then one should also write
#         new versions of all background and diffuse emission files !
fp    = open('rt_simple.ini', 'w')
idust = 0
for line in LINES:
    s = line.split()
    if (len(s)<1): continue
    if (s[0]=='optical'):
        sto, sim = stochastic[idust], simple[idust]
        if (len(sto)>0):  # stochastic dust to be replaced
            fp.write('optical %s\n' % sim)
        else:
            fp.write(line)
        idust += 1
    else:
        fp.write(line)
fp.write('nomap\n')
fp.write('nosolve\n')
if (USELIB):
    fp.write('fselect  lfreq.dat\n')  # simulation only for the library reference frequencies !!!
fp.close()
# !!!


if (0): # THE ASOC RUNS
    print("================================================================================")
    print('ASOC.py rt_simple.ini')
    print("================================================================================")
    os.system('ASOC.py rt_simple.ini')
    if (MAKELIB==2):
        # In addition to the run covering all frequencies, do another one for the
        # reference frequencies only, with x10 the number of photon packages;
        # finally merge into a single absorption file
        LINES  =  open('rt_simple.ini').readlines()
        fp     =  open('makelib.ini', 'w')
        AFILE  =  ''
        AFILE2 =  '/dev/shm/absorbed.data.lib'
        for line in LINES:
            s = line.split()
            if (len(s)<1): continue
            if (s[0].find('absorb')>=0):             # use a different absorption file
                AFILE = s[1]                         # the normal absorption file
                fp.write('absorbed  %s\n' % AFILE2)  # absorption file for ref. frequencies only
            # all packet numbers multiplied x10
            elif (s[0].find('bgpac')>=0):
                fp.write('%s %d\n' % (s[0], int(s[1])*10))
            elif (s[0].find('cellpac')>=0):
                fp.write('%s %d\n' % (s[0], int(s[1])*10))
            elif (s[0].find('pspac')>=0):
                fp.write('%s %d\n' % (s[0], int(s[1])*10))
            elif (s[0].find('diffpac')>=0):
                fp.write('%s %d\n' % (s[0], int(s[1])*10))
            else:
                fp.write(line)
        # tell ASOC to do this simulation only at reference frequencies
        fp.write('fselect  lfreq.dat \n')
        fp.close()
        if (1): # !!!
            print("================================================================================")
            print('ASOC.py makelib.ini')
            print("================================================================================")
            os.system('ASOC.py makelib.ini')
            # combine absorption files  AFILE and 
            FREQ   =  loadtxt('freq.dat')
            LFREQ  =  loadtxt('lfreq.dat')
            NLFREQ =  len(LFREQ)
            IFREQ  =  zeros(NLFREQ, int32)
            for i in range(NLFREQ):
                IFREQ[i] = argmin(abs(FREQ-LFREQ[i]))        
            print("IFREQ", IFREQ)        
            cells, nfreq = fromfile(AFILE, int32, 2)
            A = fromfile(AFILE,  float32)[2:].reshape(cells, nfreq)   # normal absorptions
            B = fromfile(AFILE2, float32)[2:].reshape(cells, nfreq)   # better for reference frequencies
            for i in IFREQ:
                A[:, i] = B[:,i]
            fp = open(AFILE, 'wb')
            asarray([cells, nfreq], int32).tofile(fp)
            A.tofile(fp)
            fp.close()
                            
    
    
# A2E_MABU.py run -- solve emission for each dust
# MAKELIB means that we have simulated all frequencies, we solve all cells and then make the library
# => A2E_MABU.py takes care of this
args = ""
if (GPU):      args += " GPU"
if (USELIB):   args += " uselib"
if (MAKELIB):  args += " makelib"   # will both calculate emitted and build the library
# calculate emission --- need <dust>.dust
LINES = open(INI).readlines()
fp = open('solve.ini', 'w')
for line in LINES:
    if (USELIB):
        fp.write(line.replace('_lib.dust', '_out_simple.dust').replace('gs_', ''))
    else:
        fp.write(line.replace('_lib.dust', '.dust'))
fp.close()

print("================================================================================")
print('A2E_MABU.py %s %s %s %s' % ('solve.ini', fabs, femit, args))
print("================================================================================")
os.system('A2E_MABU.py %s %s %s %s' % ('solve.ini', fabs, femit, args))
   

# Second SOC run -- compute maps --- fselect does not limit map frequencies
os.system('cat solve.ini | egrep -v nomap > maps.ini') # just renove "nomap" option
os.system('echo "iterations 0" >> maps.ini')
os.system('echo "nosolve"      >> maps.ini')
print("================================================================================")
print("ASOC.py maps.ini")
print("================================================================================")
os.system('ASOC.py maps.ini')

