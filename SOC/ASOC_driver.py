#!/usr/bin/python

if (0):
    from MJ.mjDefs import *
    from MJ.Aux.DustLib import *
else:
    import os,sys
    

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

2019-09-22: 
    For the USELIB case, if ini-file contains the keyword ofreq + filename,
    the emitted file will contain only the frequencies listed in the file ---
    ini will be read only in A2E_MABU and passed on as command line argument to A2E_LIB.
    
    Simulation with fewer frequencies is done using <dust>_lib.dust.
    OR WITH <dust>_simple.dust and keyword fselect !
    We will use <dust>.solver and there is no <dust>_lib.solver files !
    
2019-11-20:
    accept dust files without the initial gs_
    
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


def is_gs_dust(name):
    if (open(name).readline().split()[0]=='gsetdust'): return True
    return False


# Make a list of the dusts ... AND ABUNDANCE FILES, IF GIVEN
stochastic, simple = [], []   # list of dust names, basename without ".dust"
abundance = {}
fabs, femit = None, None      # names of absorption and emission files
nstoch, nsimple = 0, 0        # number of stochastic and equilibrium dusts
for line in LINES:
    s = line.split()
    if (len(s)<2): continue
    if (s[0]=='optical'):
        name = s[1]
        if (is_gs_dust(name)):            
            stochastic.append(name.replace('.dust',''))
            nstoch += 1
        else:
            simple.append(name.replace('.dust',''))
            nsimple += 1
        abu  = ''
        if (len(s)>2):
            if (s[2]!='#'):
                abu = s[2]
        abundance.update({name.replace('.dust', '') : abu})
    if (s[0]=='absorbed'): fabs  = s[1]
    if (s[0]=='emitted'):  femit = s[1]


    
print("================================================================================")
print("ASOC_driver")
print("SIMPLE:")
for name in simple:
    print("%30s -- [%s]" % (name, abundance[name]))
print("STOCHASTIC:")
for name in stochastic:
    print("%30s -- [%s]" % (name, abundance[name]))
# sys.exit()




# Write solver file for each stochastically heated dust
for dust in stochastic:             # dust == filename for stochastically heated dust
    solver  =  '%s.solver' % dust
    redo    =   True
    if (os.path.exists(solver)): # skip solver creation if that exists and is more recent than dust file
        if (os.stat(dust+'.dust').st_mtime<os.stat(solver).st_mtime): redo = False
    if (redo):
        print("================================================================================")
        print('A2E_pre.py %s.dust freq.dat %s' % (dust, solver))
        print("================================================================================")
        os.system('A2E_pre.py %s.dust freq.dat %s' % (dust, solver))
    

        
        
# First SOC run -- gather absorptions, stochastic dusts replaced with simple dusts
# USELIB implies the use of "libabs lfreq.dat" => absorptions only for reference frequencies
fp    = open('rt_simple.ini', 'w')
idust = 0
for line in LINES:
    s = line.split()
    if (len(s)<1): continue
    if (s[0]!='optical'):
        fp.write(line)
for d in simple:
    fp.write('optical %s.dust %s\n' % (d, abundance[d]))
for d in stochastic:
    fp.write('optical %s_simple.dust %s\n' % (d, abundance[d]))
fp.write('nomap\n')
fp.write('nosolve\n')
if (USELIB):
    # simulation only for the library reference frequencies !
    # alternatively we could have used *_lib.dust contaning ref. frequencies only
    fp.write('libabs lfreq.dat\n')
# with makelib, this initial simulation will cover all frequencies
fp.close()



if (1): # THE ASOC RUNS
    print("================================================================================")
    print('ASOC.py rt_simple.ini')
    print("================================================================================")
    if (1):
        os.system('ASOC.py rt_simple.ini')
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!! ASOC.py rt_simple.ini SKIPPED !!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        time.sleep(10)
    if (MAKELIB==2):
        # In addition to the run covering all frequencies, do another one for the
        # reference frequencies only, with x10 the number of photon packages;
        # finally merge into a single absorption file
        LINES  =  open('rt_simple.ini').readlines()
        fp     =  open('makelib.ini', 'w')
        AFILE  =  ''   # probably /dev/shm/makelib_A.lib
        AFILE2 =  '/dev/shm/makelib_A2.lib'
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
        fp.write('libabs lfreq.dat \n')
        fp.close()
        if (1): # !
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
            A = fromfile(AFILE,  float32)[2:].reshape(cells, nfreq)    # normal absorptions
            B = fromfile(AFILE2, float32)[2:].reshape(cells, NLFREQ)   # better for reference frequencies
            for i in range(3):
                A[:, IFREQ[i]] = B[:,i]
            fp = open(AFILE, 'wb')
            asarray([cells, nfreq], int32).tofile(fp)
            A.tofile(fp)
            fp.close()
        print("MAKELIB DONE --- ADDITIONAL SIMULATION OF REFERENCE FREQUENCIES DONE")
    
            

# A2E_MABU.py run -- solve emission for each dust
# MAKELIB means that we have simulated all frequencies, we solve all cells and then make the library
# => A2E_MABU.py takes care of this
args = ""
if (GPU):      args += " GPU"
if (USELIB):   args += " uselib"
if (MAKELIB):  args += " makelib"   # will both calculate emitted and build the library
# calculate emission --- need <dust>.dust


# Give A2E_MABU always the original dusts
lines = open(INI).readlines()
fpo   = open('/dev/shm/a2e.ini', 'w')
for line in lines:
    if (MAKELIB | USELIB):
        fpo.write(line.replace('_simple.dust', '.dust'))
    else:
        fpo.write(line)
fpo.close()    


print("================================================================================")
# possible arguments include makelib and uselib
# makelib ---
#         A2E_MABU.py /dev/shm/makelib.ini /dev/shm/makelib_A.data /dev/shm/makelib_E.data  GPU makelib
print('A2E_MABU.py %s %s %s %s' % (INI, fabs, femit, args))
print("================================================================================")
# sys.exit()
os.system('A2E_MABU.py %s %s %s %s' % (INI, fabs, femit, args))




# Second SOC run -- compute maps 
# remove the "nomap" option and replace stochastic dusts with corresponding simple dusts
# USELIB =>  use full dust but include "libmaps"
lines = open(INI).readlines()
fp3   = open('maps.ini', 'w')
for line in lines:
    s = line.split()
    if (len(s)<1): continue
    if (s[0]=='nomap'): continue
    if (s[0]=='libabs'): continue  # was used when absorptions were computed for the library method
    if ((s[0]=='wavelengths')&(USELIB)): continue  # use only libmaps to specify map frequencies
    if (s[0]=='optical'):
        dust = s[1]
        if (is_gs_dust(dust)):
            line = line.replace(dust, '%s_simple.dust' % (dust.split('.')[0]))        
    fp3.write(line)
fp3.write('iterations 0\n')    
fp3.write('nosolve    0\n') 
if (USELIB):  # library may have been built to provide only these output frequencies
    if (os.path.exists('ofreq.dat')):
        fp3.write('libmaps  ofreq.dat\n')
fp3.close()            

print("================================================================================")
print("ASOC.py maps.ini")
print("================================================================================")
os.system('ASOC.py maps.ini')

