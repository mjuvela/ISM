from MJ.mjDefs import *


"""
Using current absorbed.data and parameters in tmp.dust, run DustEM
for a single cell
- dust is dustem/data/GRAIN.DAT
- radiation field 4*pi*I_nu is dustem/data/ISRF.DAT
"""

FREQ         =  loadtxt('freq.dat')
NFREQ        =  len(FREQ)
CELLS, NFREQ =  fromfile('absorbed.data', int32, 2)
ABS          =  fromfile('absorbed.data', float32)[2:].reshape(CELLS, NFREQ)

imshow(ABS)
SHOW()




