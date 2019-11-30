#!/usr/bin/julia

if (length(ARGS)<3)
  println("")
  println("Usage:   DE_to_GSET.jl dustfile freqfile ne")
  println("   Writes GSET dust files based on the DustEM dust description.")
  println("   Also writes a 'simple-dust' type ascii file for SOC runs.")
  println("Input:")
  println("     dustfile   =  Name of the DustEM GRAIN.DAT -type file with the full path")
  println("     freqfile   =  ascii file with the frequencies (used for the simple-dust type output)")
  println("     ne         =  number of enthalpy bins to use")
  println("Output:")
  println("     - gs_<dustname>.dust files for each dust component and")
  println("       ascii file 'tmp.dust' containing the combined cross sections ")
  println("     - scattering functions of individual dust components in files gs_<dustname>.dsc")
  println("       and the combined scattering function in tmp.dsc")
  println("     - A2E solver files  <dustname>.solver")
  println("")
  exit(0)
end

include("/home/mika/starformation/SOC/Dust.jl")

# Read dusts
dustfile   =  ARGS[1]
freqfile   =  ARGS[2]
ne         =  parse(Int32, ARGS[3])      # number of enthalpy bins
dusts      =  read_DE(dustfile)
freq       =  readdlm(freqfile)[:,1]     # frequencies (in SOC simulation)

# Make sure dust names are unique
ind        =  1
for d in dusts
  global ind
  oldname =  d.NAME
  d.NAME  =  "$(oldname)_$ind"
  println(d.NAME)
  ind    +=  1
end

# Write the files for individual dust components
for d in dusts
  name     =  d.NAME
  write_A2E_dustfile(d, ne, prefix="")  # instead of the normal gs_ prefix !!!
  # filename = "jl_$name.solver"
  filename = "$name.solver"
  write_solver_file(d, ne, freq, filename)
  # filename = "jl_$name.dsc"
  filename = "$name.dsc"
  write_DSF_CSF(d, filename, freq, bins=2500)
  # simple dust for individual dust components
  filename = "$name"*"_simple.dust"
  write_simple_dust([d ], filename, freq)
  
end

# Write the 'simple-dust' files, sum of all components
write_simple_dust(dusts, "tmp.dust", freq)
write_DSF_CSF(dusts, "tmp.dsc", freq, bins=2500)
