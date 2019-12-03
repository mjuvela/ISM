#!/usr/bin/julia

if (length(ARGS)<3)
  println("")
  println("Usage:   A2E_pre.jl dustfile  freqfile  ne")
  println("")
  println("Input:")
  println("     dustfile   = Name of the dust file. It can be in gset format but  if it contains") 
  println("                  the string 'dustem', it is assumed to be the name of a DustEM GRAIN.DAT -type") 
  println("                  file located in the dustem data directory (full path included in 'dustfile')")
  println("     freqfile   = File with list of frequencies (ASCII file with a single column).")
  println("     ne         = Number of enthalpy bins.")
  println("Output:")
  println("     solver files for the A2E programs, named as jl_<dustname>.solver")
  println("")
  exit(0)
end

include("/home/mika/starformation/SOC/Dust.jl")

dustfile   =  ARGS[1]
freqfile   =  ARGS[2]
ne         =  parse(Int32, ARGS[3])      # number of enthalpy bins
freq       =  readdlm(freqfile)[:,1]     # frequencies (in SOC simulation)

dusts      = []
if (occursin("dustem", dustfile))        # DustEM dust (possibly several components)
  dusts = read_DE(dustfile)
else                                     # a single dust of the GSET type
  d     =  DustO()
  read(d, dustfile)
  dusts = [d,]
end

# Write separate solver file for each dust
for d in dusts
  name     =  d.NAME
  filename = "jl_$name.solver"
  write_solver_file(d, ne, freq, filename)
end
