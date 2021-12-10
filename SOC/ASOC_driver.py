#!/usr/bin/python

if (0):
    from MJ.mjDefs import *
    from MJ.Aux.DustLib import *
else:
    import os,sys
    import numpy as np

"""
Usage:
    ASOC_driver.py  soc-ini  [uselib] [makelib] 
        
If calculations involve stochastically heated grains:
    - (1) creates <dust>.solver files with A2E_pre.py
    - (2) creates <dust>_simple.dust files
    - (3) runs ASOC.py with <dust>_simple.dust files
    - (4) solves dust emission with A2E_MABU.py
    - (5) calculates maps with a second ASOC.py run

- makelib: A2E_MABU.py will make the library   
- uselib:  A2E_MABU.py will use library to calculate emission

    
If calculation involve only eqdust (no stochastically heated ones),
do also the above three runs -- this is needed in case of 
spatially varying dust abundances.   

2019-09-22: 
    For the USELIB case, if ini-file contains the keyword ofreq + filename,
    the emitted file will contain only the frequencies listed in the file;
    ini will be read only in A2E_MABU and passed on as command line argument to A2E_LIB.
    
    Simulation with fewer frequencies is done using the normal
    <dust>_simple.dust but with the added keyword fselect.
    
2019-11-20:
    accept non-simple dust files without the initial gs_
    
"""

if (len(sys.argv)<2):
    print("Usage:  ASOC_driver.py  soc.ini  [uselib] [makelib] \n")
    print("    makelib - also make the library files for all dusts")
    print("    MAKELIB - the same with second simulation of reference frequencies")
    print("    uselib  - solve emission using existing library files")
    sys.exit()

    
# Read the original ini file -- one should also make sure "noabsorbed" is dropped, it is now user's responsibility 
INI   = sys.argv[1]
LINES = open(INI).readlines()

USELIB, MAKELIB  =  0, 0
for arg in sys.argv[2:]:
    if (arg=='uselib'):      USELIB  = 1
    if (arg=='makelib'):     MAKELIB = 1  # normal run + make library
    if (arg=='MAKELIB'):     MAKELIB = 2  # ... additional run for reference frequencies
print(USELIB, MAKELIB)


def is_gs_dust(name):
    if (open(name).readline().split()[0]=='gsetdust'): return True
    return False


# Make a list of the dusts ... AND ABUNDANCE FILES, IF GIVEN
DUST, ABUNDANCE, STOCHASTIC = [], [], []
fabs, femit = None, None      # names of absorption and emission files
for line in LINES:
    s = line.split()
    if (len(s)<2): continue
    if (s[0]=='optical'):
        name = s[1]
        DUST.append(name.replace('.dust',''))
        if (is_gs_dust(name)):  # this could be "gs_aSilx"
            STOCHASTIC.append(1)
        else:                   # this coul de "aSilx_simple"
            STOCHASTIC.append(0)
        abu  = ''
        if (len(s)>2):
            if (s[2]!='#'):
                abu = s[2]
        ABUNDANCE.append(abu)
    if (s[0]=='absorbed'): fabs    = s[1]
    if (s[0]=='emitted'):  femit   = s[1]

    
STOCHASTIC = np.asarray(STOCHASTIC, np.int32)
NDUST      = len(DUST)
    
print("================================================================================")
print("ASOC_driver")
for idust in range(NDUST):
    print("%30s, stochastic=%d, [%s]" % (DUST[idust], STOCHASTIC[idust], ABUNDANCE[idust]))




# Write solver file for each stochastically heated dust
for idust in range(NDUST):
    if (STOCHASTIC[idust]==0): continue
    dust    =  DUST[idust]
    solver  =  '%s.solver' % dust
    # A2E_MABU.py will drop the "gs_" prefix from stochastic => do the same here for the solver file
    if (solver[0:3]=='gs_'): solver = solver[3:] # A2E_MABU uses aSilx.solver, not gs_aSilx.solver
    redo    =   True
    if (os.path.exists(solver)): # skip solver creation if that exists and is more recent than dust file
        if (os.stat(dust+'.dust').st_mtime<os.stat(solver).st_mtime): redo = False
    if (redo):
        print("")
        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        print("================================================================================")
        print('A2E_pre.py %s.dust freq.dat %s' % (dust, solver)) # gs_aSilx.dust -> aSilx.solver
        print("================================================================================")
        os.system('A2E_pre.py %s.dust freq.dat %s' % (dust, solver))
    

        
        
# First SOC run -- gather absorptions, stochastic dusts replaced with simple dusts
# USELIB implies the use of "libabs lfreq.dat" => absorptions only for reference frequencies
# It is possible that ini contains several dust components with separate DSC files
# This is ok as long as the "optical" lines are in the original order because
# the i:th dust is associated with the i:th DSC file
fp    = open('rt_simple.ini', 'w')
idust = 0
for line in LINES:
    s = line.split()
    if (len(s)<1): continue
    if (s[0].find('noabsorbed')>=0): continue
    if (s[0].find('libabs')>=0):  continue   # in ASOC_driver.py, simulation and library for all NFREQ frequencies !!!
    if (s[0].find('libmaps')>=0): continue
    if (s[0]!='optical'):
        fp.write(line)
for idust in range(NDUST):
    dust = DUST[idust]
    if (STOCHASTIC[idust]==0):  # simple dust already
        fp.write('optical %s.dust %s\n' % (dust, ABUNDANCE[idust]))
    else:
        # again prefix 'gs_' dropped from the name of simple dusts
        fp.write('optical %s_simple.dust %s\n' % (dust.replace('gs_', ''), ABUNDANCE[idust]))
fp.write('nomap\n')
fp.write('nosolve\n')

if (0):
    # 2021-01-28 -- removed this, use libabs only the ini contains that instruction
    #    when we use the library and the lookup fails, we can still do the full A2E run
    #    as long as the absorption file contains all frequecies == when libabs is not used
    #    If the user is confident that the library works well enough (very few cells result
    #    in failed lookup), one would put libabs into the ini file given to ASOC_driver.py
    if (USELIB):
        # simulation only for the library reference frequencies !
        # alternatively we could have used *_lib.dust containing ref. frequencies only
        fp.write('libabs lfreq.dat\n')
        
        
        
# with makelib, this initial simulation will cover all frequencies
fp.close()



if (1): # THE ASOC RUNS
    print("")
    print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
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
        fp.write('libabs lfreq.dat \n')  # libabs in ASOC_driver.py only here, to improve absorption file
        fp.close()
        if (1): # !
            print("================================================================================")
            print('ASOC.py makelib.ini')
            print("================================================================================")
            os.system('ASOC.py makelib.ini')
            # combine absorption files  from the run with all frequencies and the 
            # better run for the reference frequencies only
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
if (USELIB):   args += " uselib"
if (MAKELIB):  args += " makelib"   # will both calculate emitted and build the library
# calculate emission --- need <dust>.dust


# Give A2E_MABU always the original dusts, e.g. gs_aSilx.dust
# ... no !!!  if the init contains something like aSilx_simple.dust
# the user has chosen that and A2E_MABU.py will solve emission without
# stochastic heating for that dust component
lines   = open(INI).readlines()
fpo     = open('/dev/shm/a2e.ini', 'w')
libmaps = ''
for line in lines:
    if (0):
        if (MAKELIB | USELIB):
            fpo.write(line.replace('_simple.dust', '.dust'))
        else:
            fpo.write(line)
    else:
        fpo.write(line) # A2E_MABU will deal with both stochastic and non-stochastic dust components
    s = line.split()
    if (len(s)>1):
        if (s[0].find('libmap')>=0): libmaps = s[1]
fpo.close()    
if (USELIB):
    args += ' %s' % libmaps   # library solution => emission file with libmaps frequencies only

# If ini contained "libmaps ofreq.dat" that will be used below when writing the maps...
# aln already here the library solution will use the same, limiting emission file to ofreq.dat
print("")
print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
print("================================================================================")
print('A2E_MABU.py %s %s %s %s' % (INI, fabs, femit, args))
print("================================================================================")
# sys.exit()
os.system('cp %s a2e_mabu_backup.ini' % INI)
os.system('A2E_MABU.py %s %s %s %s' % (INI, fabs, femit, args))



# ok --- if we have libmaps in the ini, emission file contains only those frequencies



# Second SOC run -- compute maps 
# remove the "nomap" option and replace stochastic dusts with corresponding simple dusts
# USELIB =>  use full dust but include "libmaps"... if that was in the ini !!!
#            ... in that case the A2E_MABU run above 
lines    = open(INI).readlines()
fp3      = open('maps.ini', 'w')
for line in lines:
    s = line.split()
    if (len(s)<1): continue
    if (s[0]=='nomap'): continue
    if ((s[0]=='wavelengths')&(USELIB)): continue           # use only libmaps to specify map frequencies
    if ((s[0].find('libmap')>=0)&(not(USELIB))):  continue
    if (s[0]=='optical'):
        dust = s[1]
        if (is_gs_dust(dust)): 
            # once again, use dust names without "gs_" in all SOC and A2E calculations !!
            # ... and ASOC will use only simple dust files
            line = line.replace(dust, '%s_simple.dust' % (dust.replace('gs_','').split('.')[0]))
    fp3.write(line)
    # ok, if INI contained libmaps line, the maps will be written for those frequencies only
fp3.write('iterations 0\n')    
fp3.write('nosolve    0\n') 

if (0):
    # no, in ASOC_driver.py, include libmaps only if that was in the provided ini file
    if (USELIB):      # library may have been built to provide only these output frequencies
        if (os.path.exists('ofreq.dat')):
            fp3.write('libmaps  ofreq.dat\n')
fp3.close()            

print("")
print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
print("================================================================================")
print("ASOC.py maps.ini")
print("================================================================================")
os.system('ASOC.py maps.ini')

