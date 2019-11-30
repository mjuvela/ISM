# SOC
SOC - continuum radiative transfer with Python and OpenCL

For documentation, see http://www.interstellarmedium.org

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

