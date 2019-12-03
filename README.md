
#ISM 

## SOC

The directory *SOC* containsthe SOC continuum radiative transfer
program implemented with Python and OpenCL. For more detailed
background and documentation, see 
http://www.interstellarmedium.org/radiative-transfer/soc/

The main script for dust emission calculations is ASOC.py. Note that
the map orientations have changed since the previous versions (called
SOC.py). The main script for dust scattering calculations is
ASOCS.py.

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
run) the script TEST_LIB.py



## TM - template matching analysis of images

The directory *TM* contains scripts for *template matching* (TM) and
Rolling Hough Transform -type (RHT) analysis of images. As an example
of these, the script test_TM.py will run both routines on the provided
FITS image. The expected output is shown in the included test_TM.png.
For further information, see 
* http://www.interstellarmedium.org/template-matching
* Juvela M.: Pattern matching methods for the analysis of interstellar cloud structure, 2016, A&A 593, A58)
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
