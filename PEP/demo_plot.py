import sys
from matplotlib.pylab import *
import numpy as np

if (len(sys.argv)<2):
    print("Usage: python demo_plot <PEP output file>") ; sys.exit(0)
    
fp  = open(sys.argv[1], 'rb')
# number of column densities, densities, transitions, energy levels
NCD, NRHO, NTR, NL = fromfile(fp, int32, 4)
# frequencies of the transitions
F   = fromfile(fp, float32, NTR)
# density grid values
RHO = fromfile(fp, float32, NRHO)
# column density grid values
CD  = fromfile(fp, float32, NCD)
# Tex values
TEX = fromfile(fp, float32, NCD*NRHO*NTR).reshape(NCD, NRHO, NTR)
# Optical depth values
TAU = fromfile(fp, float32, NCD*NRHO*NTR).reshape(NCD, NRHO, NTR)
# Radiation temperatures
TRA = fromfile(fp, float32, NCD*NRHO*NTR).reshape(NCD, NRHO, NTR)
# Level populations
Ni  = fromfile(fp, float32, NCD*NRHO*NL).reshape(NCD, NRHO, NL)
fp.close()


figure(1, figsize=(8.7,9.5))
rc('font', size=11)
subplots_adjust(left=0.1, right=0.94, bottom=0.08, top=0.97, wspace=0.33, hspace=0.3)

# Plots vs. log10 of density and log10 of column density => the limits
E = log10(np.asarray([ min(RHO), max(RHO), min(CD), max(CD) ]))

for irow in range(3):                             # three rows = Tex, tau, T_R
    for itran in [0, 1]:                          # column = first two transitions
        Z   = [TEX, TAU, TRA ][irow][:,:,itran]   # the plotted variable
        ax  = subplot(3,2,1+itran+2*irow)
        ima = imshow(Z, extent=E, cmap=cm.gist_stern)
        levels = [ arange(8.0,25.0, 2), arange(5, 130, 10), arange(2.0, 15.0, 2) ][irow]
        contour(Z, levels=levels, colors='c', extent=E)
        xlabel(r'$\log_{10} n \/ \/ \rm [cm^{-3}]$')
        ylabel(r'$\log_{10} N \/ \/ \rm [cm^{-2}]$')
        LAB = ([ r'$T \rm _{ex}(%d. \/ tran) \/ \/ [K]$', r'$\tau \rm (%d. \/ tran.)$', r'$T \rm _{R}(%d. tran)\/ \/ [K]$' ][irow]) % (1+itran)
        colorbar(ima)
        text(1.32, 0.5, LAB, rotation=90, transform=ax.transAxes, va='center')
        text(0.95, 0.87, ([r'$T \rm _{ex}(%s)$', r'$\tau$(%s)', r'$T \rm _{R}(%s)$'][irow]) % ['1','2'][itran], transform=ax.transAxes, ha='right')

savefig('PEP_demo.png')        
show(block=True)
