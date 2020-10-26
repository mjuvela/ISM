
# ISM - programs for interstellar medium studies 

Most of the programs require a working OpenCL environment and the
installation of pyOpenCL (see
https://mathema.tician.de/software/pyopencl/).

Apart from SOC, the other programs assume that the source tree exists
under one's own home directory, in ~/GITHUB. If the files are
elsewhere, one can set an environmental variable ISM_DIRECTORY. 
ISM_DIRECTORY should point to the directory that contains ISM as a
subdirectory (which then contains Defs.py and further subdirectories
FITS, TM, etc.). If ISM_DIRECTORY is not set, that is the same as
having ISM_DIRECTORY equal to ~/GITHUB.


## SOC

The directory *SOC* contains the SOC continuum radiative transfer
program implemented with Python and OpenCL. 

The main script for dust emission calculations is ASOC.py. Note that
the map orientations have changed since the previous versions (that
was called SOC.py). The main script for dust scattering calculations
is ASOCS.py. When the program is called, it tries to figure out the
location of the source code and the kernel routines (*.c). These
should always remain in the same directory. ASOC.py and ASOC.py are
development versions (unstable).

make_dust.py is a sample script showing how to convert DustEM files to
the inputs needed by SOC, including the simple combined dust
description (tmp.dust) and the corresponding scattering function file
(tmp.dsc) used by SOC.

To deal with the emission from stochastically heated grains, one uses
"GSET" format dust files written by make_dust.py. In addition,
* A2E_pre.py writes "solver files" for single dust components
* A2E.py solves emission using these "solver files"
* A2E_MABU.py can be used to simplify this when a run includes several 
  dust populations, possibly with spatially varying abundances.
* A2E_LIB.py for making and using the library method (lookup tables
  for faster conversion of absorptions to emission)
* ASOC_driver.py automates the process of (1) calculate absorptions,
  (2) solve dust emission, (3) write emission maps

There are some corresponding julia routines (work in progress). In
particular, DE_to_GSET.jl writes GSET format dust files and the
combined simple ascii file, and the scattering function files needed
by SOC. A2E.jl corresponds to A2E.py and the (still experimental)
script MA2E.jl corresponds to A2E_MABU.py.

Practical examples of the end-to-end calculations will be added here
in the near future. For the moment, one can examine (or even try to
run) the script TEST_LIB.py that includes examples of calculations
with equilibrium temperature dust, with stochastically heated dust,
possibly with spatially varying abundances and possibly sped up by the
use of "library" methods.

For more detailed background and documentation, see 
* http://www.interstellarmedium.org/radiative_transfer/soc/
* Juvela M.: SOC program for dust continuum radiative transfer, 2019,
  A&A 622, A79, https://ui.adsabs.harvard.edu/abs/2019A%26A...622A..79J



## LOC

The directory *LOC* contains the LOC line radiative transfer
program implemented with Python and OpenCL. 

For more details, see 
* http://www.interstellarmedium.org/radiative_transfer/loc/


## TM - template matching analysis of images

The directory *TM* contains scripts for *template matching* (TM) and
Rolling Hough Transform -type (RHT) analysis of images. As an example
of these, the script test_TM.py will run both routines on the provided
FITS image. The expected output is shown in the included test_TM.png.
For further information, see 
* http://www.interstellarmedium.org/template_matching
* Juvela M.: Pattern matching methods for the analysis of interstellar
cloud structure, 2016, A&A 593, A58),
https://ui.adsabs.harvard.edu/abs/2016A%26A...593A..58J
* http://www.interstellarmedium.org/rht-rolling-hough-transform/


## Extinction

The directory *Extinction* includes some routines for the calculation
of extinction maps based on the (near-infrared) reddening of
background stars. The sample script test_Nicer.py should download
input data (photometry for 2Mass stars) from the web, calculate an
extinction map for a region specified in the script, and write the
results to a FITS files. The programs again require a working OpenCL
environment.

For further information, see
* http://www.interstellarmedium.org/numerical_tools/rht
* http://www.interstellarmedium.org/extinction/ 
  and the references mentioned there.



## FITS  - tools for FITS images

The directory *FITS* contains programs related to the handling of FITS
images. At the moment there is a draft program for the resampling of
FITS images using the Drizzle algorithm (for example the optional
shrinking of input-image pixels is not yet implemented). The execution
of 

python test_drizzle.py

should run a series of tests, comparing the run times of
Montage.reproject (assuming that one has installed montage_wrapper)
and the OpenCL routine that is run on CPU and on GPU (assuming one has
those available). Furthermore, the call

ResampleImage.py  g.fits A.fits B.fits

will resample the FITS image g.fits onto the pixels defined by the
header of the file A.fits, writing the result as a new file B.fits. If
one has successfully run test_drizzle.py, g.fits and A.fits should
already exist. A.fits was there produced from g.fits with the Montage
program so that A.fits and the file B.fits, created by the above
ResampleImage call, should be similar (except for the borders, see the
first link below).

For more information,see
* http://www.interstellarmedium.org/numerical_tools/fits_images/, including
a discussion on convolution with OpenCL
* http://www.interstellarmedium.org/numerical_tools/mad/ 
discusses the implementation of the median absolute deviation (MAD)
algorithm with OpenCL
* the original publication on the Drizzle algorithm: Fruchter & Hook,
2002, PASP 112, 144; https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F



## MBB - Modified blackbody fits

The directory contains comparisons between modified blackbody fits
with Scipy leastsq and a simplistic OpenCL routine. There are further
examples of modified blackbody fits with Markov chain Monte Carlo. The
results are discussed at 
* http://www.interstellarmedium.org/numerical_tools/mbb/





