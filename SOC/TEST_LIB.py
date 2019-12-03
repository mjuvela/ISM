import os, sys
from matplotlib.pylab import *
rc('font', size=9)


TEST_ABUNDANCE = 0 


""" 

The script includes sections to create a small model cloud, create
frequency grids and dust files, and calculate dust emission assuming
grains that are at equilibrium with the radiation field or are
stochastically heated. For the latter, there are technically different
solutions, including the use of the ASOC_driver.py script and/or the
library method. See the code for details!

The script can plot a figure with emission maps from (I)
equilibrium-temperature dust calculation, (II) from full calculation
with stochastically heated grains, and (III) the emission of
stochastically heated grains calculated via the library method). If
one uses several dust components, the plot for (I) will not
(necessarily) correspond to physical setting behind the solutions (II)
and (III). The same is true if one uses spatially varying abundances
(TEST_ABUNDANCE=1). The results from (II) and (III) should also in
those cases correspond to each other.

"""


#================================================================================
DUST     =  [ 'aSilx', 'PAH0_MC10', 'PAH1_MC10', 'amCBEx', 'amCBEx_copy1' ]
DUST0    =    'tmp.dust'

### DUST     =  [ 'aSilx', 'amCBEx' ]

PLATFORM =    0
GPU      =    1
N        =   64    # cloud dimension
NE       =  250    # enthalpy bins

def um2f(um):
    return 2.99792456e10/(1.0e-4*um)

def f2um(f):
    return 2.99792456e14/f

# gridlength 0.02  =>  Teq = 13.5-14.5K
# gridlength 0.08  =>  Teq = 11.9-14.5K
# gridlength 0.16  =>  Teq = 10.4-14.5K
INI = [ 'gridlength     0.16',
        'cloud          test_lib.cloud',
        'density        1.0e4',
        'seed           -1.0',
        'directions     0.0 0.0',
        'mapping        64 64 1.0',
        'iterations     1',
        'prefix         test',
        'device         %s' % ['c', 'g'][GPU>0],
        'platform       %s' % PLATFORM    
        ]


if (1): # Make a test cloud --- must have enough cells....
    N  = 64
    fp = open('test_lib.cloud', 'wb')
    asarray([N,N,N,1,N**3,N**3], int32).tofile(fp)
    I, J, K = indices((N,N,N), float32)
    c       = 0.5*N
    I, J, K = (I-c)/c, (J-c)/c, (K-c)/c
    r2      = I*I+J*J+K*K
    n       = exp(-4.0*log(2.0)*r2/(0.7**2))
    asarray(n, float32).tofile(fp)
    fp.close()
    if (False):
        imshow(n[N//2,:,:])
        colorbar()
        show()
        sys.exit()
    # write abundance filesfor the dust components
    for i in range(len(DUST)):
        threshold = 0.2+0.7*i/float(len(DUST))
        x         =  0.5+0.45*tanh(10.0*(n-threshold))
        asarray(x, float32).tofile('%s.abu' % DUST[i])
    # sys.exit()

    
if (1): # Make the frequency grid for the test
    F = logspace( log10(1.01e9),  log10(3.15e15), 100 )    
    fp = open('freq.dat', 'w')
    for f in F:
        fp.write('%12.5e\n' % f)
    fp.close()
    # another file for library frequencies (three reference frequencies)
    fp = open('lfreq.dat', 'w')
    # for um in [2.2, 0.55, 0.10]
    for um in [0.5, 0.2, 0.1]:
        ifreq = argmin(abs(um2f(um)-F))
        fp.write('%12.6e\n' % F[ifreq])
    fp.close()
    # third file listing four map frequencies
    fp = open('ofreq.dat', 'w')
    for um in [2500.0, 500.0, 250.0, 100.0, 50.0, 12.0]:
        ifreq = argmin(abs(um2f(um)-F))
        fp.write('%12.6e\n' % F[ifreq])
    fp.close()
    # write bg_intensity based on BISRF.dat
    from scipy.interpolate import interp1d
    d    = loadtxt('BISRF.dat')
    ip   = interp1d(d[:,0], d[:,1])
    freq = loadtxt('freq.dat')
    I    = ip(freq)
    asarray(I, float32).tofile('bg_intensity.bin')
    
    # Make dust files from DustEM
    #   - all dust components combined =>  tmp.dust, tmp.dsc
    #   - single dust component        =>  aSilx_simple.dust, aSilx.dsc
    os.system('python make_dust.py')

    # Crete solver files (to solve emission of stochastically heated grains)
    for dust in DUST:
        os.system('A2E_pre.py %s.dust freq.dat %s.solver %d' % (dust, dust, NE))
    # sys.exit()


    
        
if (1): # FULL CALCULATION WITH EQUILIBRIUM-TEMPERATURE DUST, 64^3 test case 7.5 seconds
    fp = open('test_eq.ini', 'w')
    for line in INI:
        fp.write('%s\n' % line)
    fp.write('emitted    eq.emitted\n')
    fp.write('background bg_intensity.bin\n')
    fp.write('bgpackets  2000000\n')
    fp.write('iterations 1\n')
    fp.write('optical    %s_simple.dust\n' %  DUST0)
    fp.write('dsc        %s.dsc 2500\n' % DUST0)
    fp.write('noabsorbed\n')
    fp.write('CLT\n')
    fp.write('CLE\n')
    fp.write('colden       colden_eq\n')
    fp.write('temperature  eq.T\n')
    fp.close()
    os.system('ASOC.py test_eq.ini')
    os.system('cp map_dir_00.bin  map_eq.bin')
    # PLOT T CROSS SECTIONS FROM THE EQUILIBRIUM-DUST RUN
    NX, NY, NZ, levels, cells, cells1 = fromfile('eq.T', int32, 6)
    T = fromfile('eq.T', float32)[6:].reshape(NX,NY,NZ)
    clf()
    imshow(T[NX//2,:,:])
    colorbar()
    show()
    # sys.exit()

    
    
    
if (0):    
    # FULL CALCULATION WITH STOCHASTICALLY HEATED DUST --- 59 seconds ... or 1:20 ?
    # calculate absorptions
    fp = open('test_sto_abs.ini', 'w')
    for line in INI:
        fp.write('%s\n' % line)
    fp.write('absorbed   sto.absorbed\n')
    fp.write('background bg_intensity.bin\n')
    fp.write('bgpackets  2000000\n')
    fp.write('iterations 1\n')
    fp.write('optical    %s_simple.dust\n' %  DUST0)
    fp.write('dsc        %s.dsc 2500\n' % DUST0)
    fp.write('nosolve\n')
    fp.write('nomaps\n')
    fp.close()
    os.system('ASOC.py test_sto_abs.ini')
    # solve emission
    os.system('A2E.py  %s.solver sto.absorbed sto.emitted %.1f  999' % (DUST0, GPU+0.1*PLATFORM))
    # calculate maps
    fp = open('test_sto_map.ini', 'w')
    for line in INI:
        fp.write('%s\n' % line)
    fp.write('emitted    sto.emitted\n')
    fp.write('iterations 0\n')
    fp.write('optical    %s_simple.dust\n' %  DUST0)
    fp.write('dsc        %s.dsc 2500\n' % DUST0)
    fp.write('nosolve\n')
    fp.write('colden     colden_sto\n')
    fp.close()
    os.system('ASOC.py test_sto_map.ini')
    os.system('cp map_dir_00.bin  map_sto.bin')
    sys.exit()

    
if (1):
    # ALTERNATIVE FULL CALCULATION WITH STOCHASTICALLY HEATED DUST, USING ASOC_driver.py  --- 1:22
    # -- ini must contain keywords "absorbed" and "emitted"
    # -- this includes all dust species from DUST = POSSIBLY MULTIPLE SPECIES
    fp = open('test_sto_drv.ini', 'w')
    for line in INI: fp.write('%s\n' % line)
    fp.write('absorbed    drv.absorbed\n')
    fp.write('emitted     drv.emitted\n')
    fp.write('background  bg_intensity.bin\n')
    fp.write('bgpackets   2000000\n')
    fp.write('iterations  1\n')
    if (TEST_ABUNDANCE):
        for dust in DUST:
            fp.write('optical     %s.dust %s.abu\n'   %  (dust, dust))
    else:
        for dust in DUST:
            fp.write('optical     %s.dust\n'      %  dust)
    fp.write('dsc         tmp.dsc 2500\n')
    fp.close()
    os.system('ASOC_driver.py test_sto_drv.ini GPU')
    os.system('cp map_dir_00.bin map_sto.bin')
    # sys.exit()
    
    
if (0): # FULL LIBRARY CALCULATION  --- for the 64^test case 50 seconds
    # calculate absorptions, all frequencies  ---  9 seconds
    # this one uses only single dust population
    fp = open('test_lib_abs.ini', 'w')
    for line in INI:
        fp.write('%s\n' % line)
    fp.write('absorbed   makelib.absorbed\n')
    fp.write('nosolve\n')
    fp.write('nomap\n')
    fp.write('background bg_intensity.bin\n')
    fp.write('bgpackets  2000000\n')
    fp.write('optical    %s_simple.dust\n' %  DUST0)
    fp.write('dsc        %s.dsc 2500\n' % DUST0)
    fp.close()
    os.system('ASOC.py test_lib_abs.ini')
    # now we have test.absorbed
    # make the library with A2E_LIB.py   ---   39 seconds 
    os.system('A2E_LIB.py %s.solver %s.lib freq.dat lfreq.dat makelib.absorbed makelib.emitted makelib %d' % \
    (DUST0, DUST0, GPU))
    #  now we have (gs_)aSilx.lib
    # LIBRARY METHOD   -- 7 seconds
    # calculate absorptions for the reference frequencies only
    fp = open('test_lib_ref.ini', 'w')
    for line in INI:
        fp.write('%s\n' % line)
    fp.write('absorbed   lib.absorbed\n')
    fp.write('nosolve\n')
    fp.write('nomap\n')
    fp.write('background bg_intensity.bin\n')
    fp.write('bgpackets  2000000\n')
    fp.write('optical    %s_simple.dust\n' %  DUST0)
    fp.write('dsc        %s.dsc 2500\n' % DUST0)
    fp.write('libabs     lfreq.dat\n')
    fp.close()
    os.system('ASOC.py test_lib_ref.ini')
    # now we have in lib.absorbed the absorptions for library frequencies
    if (0):
        fp = open('lib.absorbed', 'rb')
        cells, nfreq = fromfile(fp, int32, 2)
        x = fromfile(fp, float32).reshape(N, N, N, nfreq)
        for i in range(3):
            subplot(2,2,1+i)
            imshow(x[N//2, :, :, i])
            colorbar()
        show()
        sys.exit()

    # use A2E_LIB to solve emission *with the library method*, only ofreq.dat frequencies
    os.system('A2E_LIB.py %s.solver %s.lib freq.dat lfreq.dat lib.absorbed lib.emitted ofreq.dat' % (DUST0, DUST0))
    # now we have emission in lib.emitted
    
    # use ASOC to compute maps for ofreq.dat frequencies   --- 1.5 seconds
    fp = open('test_lib_map.ini', 'w')
    for line in INI:
        fp.write('%s\n' % line)
    fp.write('emitted   lib.emitted\n')
    fp.write('nosolve     \n')
    fp.write('iterations 0\n')
    fp.write('optical    %s_simple.dust\n' %  DUST0)
    fp.write('dsc        %s.dsc 2500\n' % DUST0)
    fp.write('libmaps    ofreq.dat\n')  # emitted and maps all only for these frequencies
    fp.close()
    os.system('ASOC.py test_lib_map.ini')
    os.system('cp map_dir_00.bin  map_lib.bin')
    
    sys.exit()

    

    
if (1): # FULL LIBRARY CALCULATION USING ASOC_driver.py ---
    #  this one includes all dust species from DUST
    fp = open('test_sto_drv.ini', 'w')
    for line in INI: fp.write('%s\n' % line)
    fp.write('absorbed    drv.absorbed\n')
    fp.write('emitted     drv.emitted\n')
    fp.write('background  bg_intensity.bin\n')
    fp.write('bgpackets   2000000\n')
    fp.write('libbins     bins-45-35-15')
    fp.write('iterations  1\n')
    if (TEST_ABUNDANCE):
        for dust in DUST:
            fp.write('optical     %s.dust %s.abu\n'   %  (dust, dust))
    else:
        for dust in DUST:
            fp.write('optical     %s.dust\n'      %  dust)
    fp.write('dsc         tmp.dsc 2500\n')
    fp.close()
    # make the library
    #   first run with all frequencies, creating the library for the mapping 
    #   from the absorptions at lfreq.dat frequencies to the emission at ofreq.dat frequencies
    os.system('ASOC_driver.py test_sto_drv.ini GPU makelib')
    # run simulation again, absorptions only at lfreq.dat frequencies, emission looked up
    #   from the library, maps written for the ofreq.dat frequencies
    os.system('ASOC_driver.py test_sto_drv.ini GPU uselib')
    os.system('cp map_dir_00.bin map_lib.bin')
    #sys.exit()

    
    
if (0): # plot raw data from lib.absorbed and lib.emitted
    close(1)
    figure(1, figsize=(9,3))
    subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.95, wspace=0.25, hspace=0.25)
    fp            = open('lib.absorbed', 'rb')
    cells, nlfreq = fromfile(fp, int32, 2)
    x             = fromfile(fp, float32).reshape(N,N,N,3)  # three reference frequencies
    fp.close()
    lfreq         = loadtxt('lfreq.dat')
    ofreq         = loadtxt('ofreq.dat')
    for i in range(3):
        ax = subplot(2,4,1+i)
        imshow(x[N//2,:,:,i])
        text(0.08, 0.87, r'$%.1f\mu \rm m$' % (f2um(lfreq[i])), transform=ax.transAxes)
        colorbar()
    fp            = open('lib.emitted', 'rb')
    cells, nofreq = fromfile(fp, int32, 2)
    x             = fromfile(fp, float32).reshape(N,N,N,len(ofreq))  # four selected output frequencies
    for i in range(4):   # plot emitted only for four of the ofreq
        ax     =  subplot(2,4,5+i)
        ifreq  =  int(floor(0.32*i*len(ofreq)))
        imshow(x[N//2,:,:,ifreq])
        text(0.08, 0.87, r'$%.1f\mu \rm m$' % (f2um(ofreq[ifreq])), transform=ax.transAxes)
        colorbar()
    show()
    sys.exit()
    
    
        
    
if (0):  # check energy balance for sto.absorbed vs. sto.emitted
    freq   =  loadtxt('freq.dat')
    nfreq  =  len(freq)
    ##
    fp            = open('sto.absorbed', 'rb')
    cells, nlfreq = fromfile(fp, int32, 2)
    A             = fromfile(fp, float32).reshape(N,N,N,nfreq)
    fp.close()
    fp            = open('sto.emitted', 'rb')
    cells, nlfreq = fromfile(fp, int32, 2)
    E             = fromfile(fp, float32).reshape(N,N,N,nfreq)
    fp.close()
    #
    from scipy.interpolate import interp1d
    from scipy.integrate   import quad
    AA   = A[N//2,N//2,N//2,:] * freq    # dE/df   = photons/Hz *  frequency
    ## AA[nonzero(freq>um2f(0.1))] = 0.0
    EE   = E[N//2,N//2,N//2,:] * freq    # dE/df
    ipA  = interp1d(freq, AA)
    ipE  = interp1d(freq, EE)
    IN   = quad(ipA, um2f(1e5), um2f(0.1),  limit=500)[0]  # ABSORBED
    OUT  = quad(ipE, um2f(1e5), um2f(5.0),  limit=500)[0]  # EMITTED
    print("OUT / IN    =     %.3f" % (OUT/IN))  # 0.194 ?????????????????????????????????????????????
    ###
    loglog(f2um(freq), AA, 'b-')
    loglog(f2um(freq), EE, 'r-')
    show()
    sys.exit()

    
    

    
if (1):  # PLOT MAPS -- full stochastic, equilibrium-temperature, library calculations, lib/sto
    # note that in the case of TEST_ABUNDANCE=1, the equilibrium dust solution does not take
    # the changed abundances into account
    close(1)
    figure(1, figsize=(9,8))
    subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.95, wspace=0.28, hspace=0.25)
    rc('font', size=8)
    freq   =  loadtxt('freq.dat')
    ofreq  =  loadtxt('ofreq.dat')
    nfreq  =  len(freq)
    # select frequencies to plot.... in case len(ofreq)>4
    freqp  =  zeros(4, float32)
    for i in range(4):
        freqp[i] =  ofreq[int(floor(0.32*i*len(ofreq)))]
    print(f2um(freqp))
    # plot the maps from the equilibrium-temperature dust calculation
    fp     = open('map_eq.bin', 'rb')
    nx, ny = fromfile(fp, int32, 2)
    E      = fromfile(fp, float32).reshape(nfreq, N, N) * 1.0e-5 # this contains all nfreq frequencies
    fp.close()
    for i in range(4):
        subplot(4,4,1+i)
        ifreq   =   argmin(abs(freq-freqp[i]))
        imshow(E[ifreq,:,:])
        title("EQ  %.0f = %.3e" % (f2um(freq[ifreq]), mean(ravel(E[ifreq,:,:]))), size=8)
        colorbar()
    # plot the maps from the full calculation with stochastic heating
    fp = open('map_sto.bin', 'rb')
    nx, ny = fromfile(fp, int32, 2)
    S      = fromfile(fp, float32).reshape(nfreq, N, N) * 1.0e-5 # this contains all nfreq frequencies
    fp.close()
    for i in range(4):
        subplot(4,4,5+i)
        ifreq   =   argmin(abs(freq-freqp[i]))
        imshow(S[ifreq,:,:])
        title("STO %.0f = %.3e" % (f2um(freq[ifreq]), mean(ravel(S[ifreq,:,:]))), size=8)
        colorbar()
    # plot the maps from the library method
    fp = open('map_lib.bin', 'rb')
    nx, ny = fromfile(fp, int32, 2)
    L  = fromfile(fp, float32).reshape(len(ofreq), N, N) * 1.0e-5
    fp.close()
    for i in range(4):
        subplot(4,4,9+i)
        ifreq   =   argmin(abs(ofreq-freqp[i]))
        imshow(L[ifreq,:,:])
        title("LIB %.0f = %.3e" % (f2um(ofreq[ifreq]), mean(ravel(L[ifreq,:,:]))), size=8)
        colorbar()
    # plot the ratio of the maps from the library method and from the full stochastic calculation
    for i in range(4):
        subplot(4,4,13+i)
        ifreq   =   argmin(abs(freq-freqp[i]))
        ifreqo  =   argmin(abs(ofreq-freqp[i]))
        offset  =        0.0001*max(ravel(S[ifreq,:,:]))
        tmp     =   (offset+L[ifreqo,:,:])/(offset+S[ifreq,:,:])
        imshow(tmp, vmin=0.7, vmax=1.3)
        title("L/S %.0f = %.3e" % (f2um(freq[ifreq]), mean(ravel(tmp[:,:]))), size=8)
        colorbar()
    savefig("TEST_LIB_%s.png" % time.asctime()[0:12].replace(' ','_'))
    show()
    sys.exit()
    
    
    
if (0):
    # compare spectra from the full stochastic and equilibrium dust calculations
    #   sto spectra written using  11.ini 
    #   lib spectra written using   3.ini 
    from scipy.integrate import quad
    from scipy.interpolate import interp1d
    freq   =  loadtxt('freq.dat')
    nfreq  =  len(freq)
    ofreq  =  loadtxt('ofreq.dat')
    ###
    fp     = open('map_sto.bin', 'rb')
    nx, ny = fromfile(fp, int32, 2)
    x      = fromfile(fp, float32).reshape(nfreq, N, N) * 1.0e-5 # this contains all nfreq frequencies
    fp.close()
    STO    = x[:,N//5,N//2]
    ###
    fp     = open('map_eq.bin', 'rb')
    nx, ny = fromfile(fp, int32, 2)
    x      = fromfile(fp, float32).reshape(nfreq, N, N) * 1.0e-5 # this contains all nfreq frequencies
    fp.close()
    EQ     = x[:,N//5,N//2]
    ### add points for the library method prediction
    fp     = open('map_lib.bin', 'rb')
    nx, ny = fromfile(fp, int32, 2)
    x      = fromfile(fp, float32).reshape(len(ofreq), N, N) * 1.0e-5 # ofreq only
    LIB    = x[:, N//5, N//2]
    ## compare the total energies in the two spectra
    ipS = interp1d(freq, STO, bounds_error=False, fill_value=0.0)
    ipE = interp1d(freq,  EQ, bounds_error=False, fill_value=0.0)
    E_S = quad(ipS, um2f(1900.0), um2f(4.0) , limit=200 )[0]
    E_E = quad(ipE, um2f(1900.0), um2f(12.3), limit=200 )[0]
    ##
    ax     = axes()    
    loglog(f2um(freq),  EQ,  'b-')
    loglog(f2um(freq),  STO, 'r-')
    loglog(f2um(ofreq), LIB, 'k+')
    text(0.1, 0.6, 'STO = %.3e' % E_S, transform=ax.transAxes, color='r')
    text(0.1, 0.5, 'EQ  = %.3e' % E_E, transform=ax.transAxes, color='b')
    print("RATIO STO/EQ = %.2f" % (E_E/E_S))  #  E_E = 4.77 * E_S ????
    savefig("TEST_LIB_SPE_%s.png" % time.asctime()[0:12].replace(' ','_'))
    show()
    

    
if (0):
    # compare lib.emitted and sto.emitted
    freq         = loadtxt('freq.dat')
    ofreq        = loadtxt('ofreq.dat')    
    cells, nfreq = fromfile('sto.emitted', int32, 2)
    S            = fromfile('sto.emitted', float32)[2:].reshape(cells, nfreq)
    # extract only the ofreq, the ones library method provides
    nout         = len(ofreq)
    SS           = zeros((cells, nout), float32)
    for i in range(nout):
        ifreq    = argmin(abs(ofreq[i]-freq))
        SS[:,i]  = S[:,ifreq]
    subplot(231)
    imshow(SS, aspect='auto')
    colorbar()

    ofreq        = loadtxt('ofreq.dat')
    cells, nfreq = fromfile('lib.emitted', int32, 2)
    L            = fromfile('lib.emitted', float32)[2:].reshape(cells, nfreq)
    subplot(232)
    imshow(L, aspect='auto')
    colorbar()
    
    subplot(233)
    imshow(L/SS, vmin=0.6, vmax=1.4, aspect='auto')
    colorbar()
    
    # plot cross sections of the model, the predicted emission at a single wavelength
    # these are ok... emission within 10% the same, no large-scale gradients
    ifreq = 2
    N    =   int(round(cells**0.3333333333))
    SC   =   SS[:,ifreq].reshape(N,N,N)
    LC   =    L[:,ifreq].reshape(N,N,N)
    
    subplot(234)
    imshow(SC[:,:,N//2])
    colorbar()

    subplot(235)
    imshow(LC[:,:,N//2])
    colorbar()
    
    subplot(236)
    imshow(LC[:,:,N//2]/SC[:,:,N//2])
    colorbar()
    
    show()
    
