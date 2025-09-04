
# ISM - programs for interstellar medium studies 

Most of the programs require a working OpenCL environment and the
installation of pyOpenCL (see https://mathema.tician.de/software/pyopencl/).

Programs assume that the source tree exists
under one's own home directory (see the error messages to set suitable symbolic links for the paths).

**Note:** radiative transfer programs SOC and LOC now reside in separate repositories: https://github.com/mjuvela/SOC and https://github.com/mjuvela/LOC .


## TM - template matching analysis of images

The directory *TM* contains scripts for *template matching* (TM) and
Rolling Hough Transform -type (RHT) analysis of images. As an example
of these, the script test_TM.py will run both routines on the provided
FITS image. The expected output is shown in the included test_TM.png.
For further information, see 
* http://www.interstellarmedium.org/numerical_tools/template_matching
* [Juvela M. 2016, A&A 593, A56: Pattern matching methods for the analysis of interstellar
cloud structure](https://ui.adsabs.harvard.edu/abs/2016A%26A...593A..58J)
* [www.interstellarmedium.org/numerical_tools/rht/](http://www.interstellarmedium.org/numerical_tools/rht/)


## Extinction

The directory *Extinction* includes some routines for the calculation
of extinction maps based on the (near-infrared) reddening of
background stars. The sample script test_Nicer.py should download
input data (photometry for 2Mass stars) from the web, calculate an
extinction map for a region specified in the script, and write the
results to a FITS files. The programs again require a working OpenCL
environment.

For further information, see
*[www.interstellarmedium.org/extinction](http://www.interstellarmedium.org/extinction/)'
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
* [www.intertellarmedium.org/numerical_tools/mbb](http://www.interstellarmedium.org/numerical_tools/mbb/)


## Ocfil - OpenCL program for filament extraction

Ocfile directory contains a Python/pyOpenCL program that tries to
identify filaments from a FITS image, trace the filaments, and
produce 2D of them (one dimension running along the filament, the
other being perpendicular). The current GitHub version relies on
scipy.ndimage routine label(), which may not scale well for large
images (>1000x1000 pixels). The (simple) program will be described
in more detail at
* [www.interstellarmedium.org/numerical_tools/filaments](http://www.interstellarmedium.org/numerical_tools/filaments/)

## SPIF - OpenCL program for fitting spectral lines

SPIF can fit individual spectral lines with Gaussian (one or multiple velocity
components) or as hyperfine lines consisting of several components. The input 
and output files are FITS cubes. The program is described in Juvela, M.,
Tharakkal, D., 2024, A&A, "Fast fitting of spectral lines with Gaussian and
hyperfine structure models" [in
arXiv](https://ui.adsabs.harvard.edu/abs/2024arXiv240304352J/abstract) and 
[in ADS](https://ui.adsabs.harvard.edu/abs/2024A%26A...685A.164J/abstract).

## DIES - dust radiative transfer with the immediate reemission method

Program handles currently only spherical models (cells are spherical shells),
computes dust temperatures (equilibrium temperatures only), and returns
spectra for different impact parameters. The program uses OpenCL libraries and
is recommended to be run on GPUs. The paper Juvela M., "DIES: parallel dust
radiative transfer program with the immediate re-emission method" has been
submitted. Further information will be provided at 
* [www.interstellarmedium.org/radiative_transfer](http://www.interstellarmedium.org/radiative_transfer/)


## PEP - line intensity predictions with the escape probability method

Program calculates line-intensity predictions for a given kinetic
temperature and in parallel for a grid of density and column density values. The
calculation is based on the escape probability formalism. A paper
(M. Juvela: PEP: parallel estimation of spectral line parameters
with the escape-probability method, 2024) has been submitted, and
some documentation can be found at
* [www.interstellarmedium.org/radiative_transfer](http://www.interstellarmedium.org/radiative_transfer/)


