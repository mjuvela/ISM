
# ISM - programs for interstellar medium studies 

Most of the programs require a working OpenCL environment and the
installation of pyOpenCL (see
https://mathema.tician.de/software/pyopencl/).

Apart from SOC, the other programs assume that the source tree exist
under one's own home directory, in ~/GITHUB. If the files are
elsewhere, one can set an environmental variable ISM_DIRECTORY. Thus,
ISM_DIRECTORY should point to a directory that contains ISM as a
subdirectory (which then contains Defs.py and further subdirectories
FITS, TM, etc.). If ISM_DIRECTORY is not set, that is implicitly the
same as having ISM_DIRECTORY equal to ~/GITHUB.


## SOC

The directory *SOC* contains the SOC continuum radiative transfer
program implemented with Python and OpenCL. 

The main script for dust emission calculations is ASOC.py. Note that
the map orientations have changed since the previous versions (called
SOC.py). The main script for dust scattering calculations is ASOCS.py.
When the program is called, it tries to figure out the location of the
source code and the kernel routines (*.c). These should always remain
in the same directory.

make_dust.py is a sample script showing how to convert DustEM files to
the inputs needed by SOC, including the simple combined dust
description (tmp.dust) and the corresponding scattering function file
(tmp.dsc) used by SOC.

To deal with the emission from stochastically heated grains, one uses
the "GSET" format dust files written by make_dust.py. In addition,
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
* http://www.interstellarmedium.org/radiative-transfer/soc/
* Juvela M.: SOC program for dust continuum radiative transfer, 2019,
  A&A 622, A79, https://ui.adsabs.harvard.edu/abs/2019A%26A...622A..79J



## TM - template matching analysis of images

The directory *TM* contains scripts for *template matching* (TM) and
Rolling Hough Transform -type (RHT) analysis of images. As an example
of these, the script test_TM.py will run both routines on the provided
FITS image. The expected output is shown in the included test_TM.png.
For further information, see 
* http://www.interstellarmedium.org/template-matching
* Juvela M.: Pattern matching methods for the analysis of interstellar
cloud structure, 2016, A&A 593, A58),
https://ui.adsabs.harvard.edu/abs/2016A%26A...593A..58J
* http://www.interstellarmedium.org/rht-rolling-hough-transform/


## Extinction

The directory *Extinction* includes some routines for the
calculation of extinction maps based on the (near-infrared)
reddening of background stars. The sample script test_Nicer.py
should download input data (photometry for 2Mass stars) from the
web, calculate an extinction map for a region specified in the
script, and write the results as FITS files. The programs again
require a working OpenCL environment.

For further information, see
* http://www.interstellarmedium.org/nicer-extinction-maps/
* http://www.interstellarmedium.org/extinction/ 
  and the references mentioned there.



## FITS

The directory *FITS* will contain programs related to the handling of
FITS images. At the moment there is a draft program for the resampling
of FITS images using the Drizzle algorithm (for example the optional
shrinking of imput-image pixels is not yet implemented). 

If one executes

python test_drizzle.py

it should run a series of tests, comparing the run times of
Montage.reproject (assuming that one has installed montage_wrapper)
and the OpenCL routine run on CPU and on GPU (assuming one has those
available). Furthermore, the call

ResampleImage.py  g.fits A.fita B.fits

will resample the FITS image g.fits onto the pixels defined in the
header of the file A.fits, writing the result as the new file B.fits.
If one has already run test_drizzle.py, g.fits and A.fits should
already axist. A.fits was produced from g.fits with the Montage
program so that A.fits and the file B.fits, created by the above
ResampleImage call, should be similar (except for the borders, see the
first link below).

For more information,see
* http://www.interstellarmedium.org/tools-for-fits-images/
* the original publication on the Drizzle algorithm: Fruchter & Hook,
2002, PASP 112, 144; https://ui.adsabs.harvard.edu/abs/2002PASP..114..144F
