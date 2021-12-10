from MJ.mjDefs import *

x = fromfile('aSilx.lib', float32)[5:] 
x = x[-30*34*15*150:].reshape(30*34*15,150) 
imshow(x, aspect='auto')

SHOW()

