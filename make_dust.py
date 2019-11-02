#!/usr/bin/python3

import os, sys
HOMEDIR = os.path.expanduser('~/')
sys.path.append(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/')
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

Note:
    * the directory of the dustem installation is set inside DustLib.py!!
    * one must have file freq.dat to list the frequencies used
"""  


DUSTEM_FILE = 'GRAIN_C11.DAT'
FREQ        =  loadtxt('freq.dat')

NEWNAME   =  write_DUSTEM_files(DUSTEM_FILE)  # returns unique dust names

DEDUST = []
METHOD = 'CRT'  # use this !!
for name in NEWNAME:
    DDUST  = DustemDustO('dustem_%s.dust' % name, force_nsize=200)
    DEDUST.append(DDUST)
    write_simple_dust_CRT([DDUST,], FREQ, filename='%s_simple.dust' % name, dscfile='%s.dsc' % name)

# and one simple dustfile for the combination of the dusts
write_simple_dust_CRT(DEDUST, FREQ)       # ==> tmp.dust

if (1):    
    write_A2E_dustfiles(NEWNAME, DEDUST, NE_ARRAY=128*ones(len(DEDUST)))  # only 128 enthalpy bins

