import os, sys
sys.path.append('./')
from DustLib import *

"""
Write dust files based on DustEM presciprions. For example:
    aSilx_simple.dust   -- simple format file for individual dusts
    aSilx.dsc           -- scattering functions for individual dusts
    tmp.dust            -- simple format file for the combination
    tmp.dsc             -- combined scattering function
    gs_aSilx.dust       -- GSETDustO format files
    gs_aSilx.size          ...
    gs_aSilx.opt           ...
    gs_aSilx.ent           ...
    
SOC would be use simple format files
* combined simple file if abundances are the default
* individual simple files if abundances are variable
* A2E would use the corresponding GSETDustO files
"""  

# Specify the dust model = file in DustEM directory
DUSTEM_DIR  =  HOMEDIR+'/tt/dustem4.0_web'
DUSTEM_FILE = 'GRAIN_C11.DAT'
FREQ        =  loadtxt('freq.dat')

# write_DUSTEM_files writes dust files for CRT+DustEM calculation, renames 
# dust components so that each size distribution is associated with a unique name
NEWNAME    =  write_DUSTEM_files(DUSTEM_FILE)

# write simple dust = dust files for pure radiative transfer calculation
DEDUST = []
for name in NEWNAME:
    DDUST  = DustemDustO('dustem_%s.dust' % name, force_nsize=200)
    DEDUST.append(DDUST)
    if (SCA):
        write_simple_dust_CRT([DDUST,], FREQ, filename='%s_simple_sca.dust' % name, dscfile='%s_sca.dsc' % name)
    else:
        write_simple_dust_CRT([DDUST,], FREQ, filename='%s_simple.dust' % name, dscfile='%s.dsc' % name)

# and one simple dust for the combination of the dusts
write_simple_dust_CRT(DEDUST, FREQ)      # ==> tmp.dust
write_A2E_dustfiles(NEWNAME, DEDUST, NE_ARRAY=128*ones(len(DEDUST)), prefix='')  # ONLY 128 ENTHALPY BINS

