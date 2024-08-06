import os, sys
import numpy as np
from matplotlib.pylab import *
sys.path.append('./')
from DIES_Aux  import *

"""
Full example of DIES run. 
- The cloud file is created here (demo.cloud).
- The file for point source luminosity is created here (demo_ps.dat)
- The dust file is assumed to exist in the current directory (demo.dust)
- The file for isotropic background intensity is assumed to exist
  already (demo_isrf.dat)
"""


#--------------------------------------------------------------------------------
# Start by making the cloud = density as function of radius
# We use here directly radius values in the final [cm] units.
N  =  100
R  =  0.1*linspace(0.0, 1.0, N+1)[1:]**1.0 # shell radius values [pc]
n  =  logspace(3,1,N)  # ad hoc 1e3cm-3 in the centre, 10cm-3 on surface
fp =  open('demo.cloud', 'w')
fp.write('%d\n' % N)
for i in range(N):
    fp.write('%12.4e %12.4e\n' % (R[i], n[i]))
fp.close()

#--------------------------------------------------------------------------------
# One file for point source luminosity. Here T=10000K blackbody, L=L_Sun
Teff    =  10000.0            # Chosen effective temperature of the source
NF      =  400                # using NF frequencies in the frequency grid
FREQ    =  logspace(log10(um2f(10000)), log10(um2f(0.092)), NF) # freq. grid
flux    =  pi*PlanckFunction(FREQ, Teff)
L       =  flux * 4*pi*RSUN**2.0  # at this luminosity with arbitrary scaling
ip      =  interp1d(FREQ, L)      # interpolated values used for integration
k       =  LSUN / quad(ip, FREQ[0], FREQ[-1])[0] # scaling to 1 LSUN luminosity
L      *=  k             # spectrum corresponding to L=L_Sun
L[nonzero(FREQ>um2f(0.092))] = 1.0e-30  # apply hard Lyman limit?
# Write the file of the spectral point source luminosity
fp = open('demo_ps.dat', 'w')
for i in range(NF):
    fp.write('%12.5e %12.5e\n' % (FREQ[i], L[i]))
fp.close()

#--------------------------------------------------------------------------------
# Write the ini file
L = """
cloud            demo.cloud
prefix           demo               # prefix for output files
offsets          128                # spectra for this many impact parameter values
dust             demo.dust          # dust model
background       demo_isrf.dat 1.0  # intensity of isotropic background
bgpackets        9000000            # packages from the isotropic background
pointsource      demo_ps.dat 1.0 4.0e-5  # luminosity file, scaling, nominal radius [pc]
pspackets        1000000            # packages from the central source
seed            -1.0                # random seed
gpu              1                  # use gpu
# platform       0                  # select OpenCL device by number
# sdevice        4070               # select OpenCL device by name
force            1                  # forced first interactions
"""
fp = open('demo.ini', 'w')
fp.write(L)
fp.close()

#--------------------------------------------------------------------------------
# Run the program, assuming DIES.py is in the current directory
os.system('python DIES.py demo.ini')


#--------------------------------------------------------------------------------
# Read and plot the results
figure(1, figsize=(8,8))
rc('font', size=12)
d = np.loadtxt('demo.T')    #  text file with radius and Tdust [K]
subplot(211)
plot(d[:,0], d[:,1], 'rx-')
ylabel(r'$T_{\rm dust} \rm \/ \/ [K]$')
xlabel(r'$\rm Radius \/ \/ [pc]$')

subplot(212)
fp       = open('demo.spe') # binary file with surface brightness spectra
NF, OFFS = fromfile(fp, int32, 2)    # number of frequencies, spatial offsets
F        = fromfile(fp, float32, NF) # frequency values
RES      = fromfile(fp, float32).reshape(OFFS, NF) # RES[OFFS, BF] spectra [cgs]
RES     *= 1.0e23 * 1.0e-6  #from cgs units to units of  MJy/sr
# plot spectrum towards the centre
loglog(f2um(F), RES[0,:], 'r-', label='Centre')
# plot spectrum for half way towards the surface
loglog(f2um(F), RES[OFFS//2,:], 'b-', label='Half radius')
xlabel(r'$\lambda  \rm  \/ \/ [\mu m]$')
ylabel(r'$I_{\nu} \rm \/ \/ [MJy \/ sr^{-1}]$')
xlim(1.0, 4000.0)
ylim(1e-19, 900.0)
legend(loc='lower right')

savefig('DIES_demo.png')
show(block=True)

