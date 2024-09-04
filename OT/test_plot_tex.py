import os, sys
from matplotlib.pylab import *
sys.path.append('/home/mika/GITHUB/ISM/OT/')
from OT_library import *

""" 

This is an example of how to extract values (here excitation temperatures,
Tex, written from LOC_OT.py program) for a set of coordinate positions,
using code from OT_library.py. We uses the routine OT_GetIndicesV(), which
is implemented as an OpenCL kernel and therefore requires input
coordinates as 1d vectors of np.floa32 values.

One could do the same using OT_GetValueV(), which returns directly the
value for given coordinates. However, that routine is implemented in pure
Python and accepts only scalar coordinates. It is therefore a *far slower*
option. 

This file is included here just an example of how reading and plotting of
octree data could be done. The actual files (*.loc, *.tex) that are
referenced in this script are *not* included.

"""

# Check the size of the root grid of your octree hierarchu => RG^3 cells.
RG = 118  

# Extract values for a regular Cartesian grid of N^3 points. 
N  = 128

# We read a cloud file that has been created for the LOC program.
# We can use OT_ReadHierarchyV, instead of OT_ReadHierarchyV_LOC, because we
# need only the first field (i.e. the density) that also encodes the 
# information of the octree hierarchy.
NX, NY, NZ, LCELLS, OFF, H = OT_ReadHierarchyV('000176_IRDC_Tdust.loc')

# Read a file of excitation temperatures (skipping the leading four integers
# in the Tex file). The arrays H and TEX should now have the same dimensions.
TEX = np.fromfile('irdc_13co_Tdust_13CO_01-00.tex', float32)[4:] 

# Create a coordinate grid of N^3 points, each coordinate in the range ]0.0, RG[
L       = linspace(0.0, RG, N+2)[1:-1] # N internal points
x, y, z = meshgrid(L, L, L)
# Coordinates have to be 1d vectors of np.float32 values...
x, y, z = np.array(ravel(x), float32), np.array(ravel(y), float32), np.array(ravel(z), float32)

# Look up cell indices for these coordinates (note the option global_index).
IND     = OT_GetIndicesV(x, y, z, NX, NY, NZ, LCELLS, OFF, H, global_index=True)

# Extract cell values (=Tex values) for these cells and reshape back to N^3 grid.
Tex     = TEX[IND].reshape(N, N, N)

# Plot a 2D cross section of the Tex values.
imshow(Tex[:,:,N//2], vmin=2.7, vmax=20.0)
colorbar()
show(block=True)
