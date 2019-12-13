#!/usr/bin/python
import os, sys, numpy
# one must have already defined ISM_DIRECTORYand added that  to the path !!
from Nicer_aux import *


def read_2mass_file(filename, version=-1, allstars=False, Qflg=False, Eflg=False):
    """
    Read 2MASS table
      COO, PHOT, dPHOT = read_2mass_file(filename, version=-1, allstars=False)
    Input:
        filename = name of the text file containing output from 2Mass archive
        version  = version of file format, with -1 figures it out automatically
        allstars = if True, return magnitudes even when star has not all J, H, and Ks
                   magnitudes OR the uncertainty of a magnitude value is missing,
                   with version 3 only !
        Qflg     = if true, return also array of Qflg values ('AAA' etc.) for each source
        Eflg     = if true, return also array of extended flags ('---' or something else)
    Return:
        COO    = [x, y] the coordinates in radians
        PHOT   = [M1, M2, ...] the magnitudes
        dPHOT  = [dM1, dM2, ...] error estimates for the magnitudes
        Qflg   = optional =>  [ 'AAA', 'ABC', ... ], optional  == ph_qual = 144:147
        Eflg   = optional =>  [ '---', Extnd = ext_key (?) == 357:364
    Note:
        missing magnitudes and uncertainties are set to nan (allstars=True)    
    """
    lines = open(filename).readlines()
    print('################ %d LINES !!!!!!!!!!!!' % len(lines))
    if (version<0):
        # try to figure out the correct format
        if (lines[2].find('CDS')>=0):
            version = 3  # CDS data server
        elif (lines[0].find('HTTP')>=0):
            version = 2  # IRSA URL query result
        elif ((lines[0].find('fixlen')>=0)|(lines[1].find('fixlen')>=0)):
            version = 4
        else:
            version = 1  # because version 0 is obsolete?
        print('read_2mass_file  %s   version %d' % (filename, version))
    COO   = []
    PHOT  = []
    dPHOT = []
    QFLG  = []
    EFLG  = []
    if (version==0):
        for line in lines:
            if (line[0]!=' '): continue # still header lines        
            s = line.split() # extract coordinates and magnitudes from this line
            t = line[55:].split()
            if ((t[4]=='null')|(t[5]=='null')): continue  # skip all stars without some magnitude ???
            if ((t[8]=='null')|(t[9]=='null')): continue
            if ((t[12]=='null')|(t[13]=='null')): continue
            COO.append(   [ float(s[0]), float(s[1]) ] )
            PHOT.append(  [ float(t[4]), float(t[8]), float(t[12]) ] )
            dPHOT.append( [ float(t[5]), float(t[9]), float(t[13]) ] )
    elif (version==1):
        for line in lines:
            if (line[0]!=' '): continue # still header lines        
            s = line.split() # extract coordinates and magnitudes from this line
            if ((s[6]=='null')|(s[7]=='null')): continue  # skip all stars without some magnitude ???
            if ((s[10]=='null')|(s[11]=='null')): continue
            if ((s[14]=='null')|(s[15]=='null')): continue
            COO.append(   [ float(s[0]), float(s[1]) ] )
            PHOT.append(  [ float(s[6]), float(s[10]), float(s[14]) ] )
            dPHOT.append( [ float(s[7]), float(s[11]), float(s[15]) ] )
    elif (version==2):
        # the version from IRSA url enquiry (magnitudes 7, 9, 11)
        for line in lines:
            if (line[0]!=' '): continue # still header lines        
            s = line.split() # extract coordinates and magnitudes from this line
            if ((s[7]=='null')|(s[8]=='null')): continue  # skip all stars without some magnitude ???
            if ((s[9]=='null')|(s[10]=='null')): continue
            if ((s[11]=='null')|(s[12]=='null')): continue
            if (s[7]=='null'):  J   = nan
            else:               J   = float(s[7])
            if (s[8]=='null'):  dJ  = nan
            else:               dJ  = float(s[8])
            if (s[9]=='null'):  H   = nan
            else:               H   = float(s[9])
            if (s[10]=='null'): dH  = nan
            else:               dH  = float(s[10])
            if (s[11]=='null'): K   = nan
            else:               K   = float(s[11])
            if (s[12]=='null'): dK  = nan
            else:               dK  = float(s[12])
            if (allstars | (isfinite(J) & isfinite(dJ) & isfinite(H) & isfinite(dH) & isfinite(K) & isfinite(dK))):
                COO.append(   [ float(s[0]), float(s[1]) ] )
                PHOT.append(  [  J,  H,  K ] )
                dPHOT.append( [ dJ, dH, dK ] )
    elif (version==3): 
        # CDS data server
        for line in lines:
            if (line[0]=='#'): continue
            s = line.replace('|',' ').split() # extract coordinates and magnitudes from this line
            if (s[6]=='---'):  J  = nan
            else:              J  = float(s[6])
            if (s[7]=='---'):  dJ = nan
            else:              dJ = float(s[7])
            if (s[10]=='---'): H  = nan
            else:              H  = float(s[10])
            if (s[11]=='---'): dH = nan
            else:              dH = float(s[11])
            if (s[14]=='---'): K  = nan
            else:              K  = float(s[14])
            if (s[15]=='---'): dK = nan
            else:              dK = float(s[15])
            if (allstars | (isfinite(J) & isfinite(dJ) & isfinite(H) & isfinite(dH) & isfinite(K) & isfinite(dK))):
                COO.append(   [ float(s[0]), float(s[1]) ] )
                PHOT.append(  [  J,  H,  K ] )
                dPHOT.append( [ dJ, dH, dK ] )
                QFLG.append(  line[144:147] )
                EFLG.append(  line[357:364].strip() )
    elif (version==4): 
        # IPAC again different ???
        for line in lines:
            if ((line[0]=='\\')|(line[0]=='|')): continue
            s = line.split() # extract coordinates and magnitudes from this line
            # J, H, K = 8, 12, 16
            if (s[8]=='null'):  J   = nan
            else:               J   = float(s[8])
            if (s[9]=='null'):  dJ  = nan
            else:               dJ  = float(s[9])
            if (s[12]=='null'): H   = nan
            else:               H   = float(s[12])
            if (s[13]=='null'): dH  = nan
            else:               dH  = float(s[13])
            if (s[16]=='null'): K   = nan
            else:               K   = float(s[16])
            if (s[17]=='null'): dK  = nan
            else:               dK  = float(s[17])
            if (allstars | (isfinite(J) & isfinite(dJ) & isfinite(H) & isfinite(dH) & isfinite(K) & isfinite(dK))):
                COO.append(   [ float(s[0]), float(s[1]) ] )
                PHOT.append(  [ float(s[8]), float(s[12]), float(s[16]) ] )
                dPHOT.append( [ float(s[9]), float(s[13]), float(s[17]) ] )        
                #QFLG.append(  line[144:147] )
                #EFLG.append(  line[357:364].strip() )

    if (Qflg):
        if (Eflg):
            return array(COO,float32)*DEGREE_TO_RADIAN, array(PHOT, float32), array(dPHOT, float32), QFLG, EFLG
        else:
            return array(COO,float32)*DEGREE_TO_RADIAN, array(PHOT, float32), array(dPHOT, float32), QFLG
    else:
        if (Eflg):
            return array(COO,float32)*DEGREE_TO_RADIAN, array(PHOT, float32), array(dPHOT, float32), EFLG
        else:
            return array(COO,float32)*DEGREE_TO_RADIAN, array(PHOT, float32), array(dPHOT, float32)
    


def read_2mass_www(ra, dec, box_size, filename='tmp_www_dump.dat', Qflg=False, Eflg=False):
    """
    Download 2MASS data from IRSA.
    Inputs:
           ra, dec  = coordinates for the box centre (J2000, radians)
           box_size = width of the box [radians]
           filename = file for storing the data
           Qflg     = if true, return an array of Qflg values for each source
           Eflg     = if true, return an array of extended flags
        If query is unsuccessful, file is created but will be empty
    Returns:
        COO   =  stellar coordinates [radians], J2000
        PHOT  =  magnitudes for the stars
        dPHOT =  uncertainties of the magnitudes
    """
    print('read_2mass_www', filename)
    # CGI parameters
    #    spatial = Box
    #    size    = size [arcsec]
    #    objstr  = "00+42+44.3+-41+16+08"
    #    catalog = 2MASS ????
    #    outfmt  = 1    (ascii)
    #    http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&objstr=10.68469+41.26904&
    #        spatial=Cone&radius=1&radunits=arcsec&catalog=fp_psc
    # print '\n BOX SIZE %.1f ARCMIN \n\n' % (box_size*RADIAN_TO_ARCMIN)
    if (os.path.exists(filename)):
        COO, PHOT, dPHOT = read_2mass_file(filename, version=-1)
        if (COO.shape[0]>0):
            return read_2mass_file(filename, version=-1, Qflg=Qflg, Eflg=Eflg)

    ok = False
    if (os.path.exists('/usr/local/bin/find2mass')):
        # query CDS archive
        print('going to do find2mass for %s' % filename)
        os.system('find2mass %.3f %.3f -b %.2f -m 99999 > %s' % 
        (ra*RADIAN_TO_DEGREE, dec*RADIAN_TO_DEGREE, box_size*RADIAN_TO_ARCMIN, filename))
        os.sys.stdout.flush()
        no = len(open(filename).readlines())
        print('find2mass done ... %d lines read to %s' % (no, filename))
        if (no<4):
            print('*** WARNING: find2mass failed... trying IRSA instead !!!')
            os.system('rm %s' % filename)
        else:
            print('find2mass done ... ok !!')
            ok = True
    if (not(ok)):
        # try IRSA URL query
        cmd = 'https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?outfmt=1&'
        cmd = cmd + 'objstr=%.4f+%+.4f&spatial=Box&size=%.2f&catalog=fp_psc' % (
              ra*RADIAN_TO_DEGREE, dec*RADIAN_TO_DEGREE, box_size*RADIAN_TO_ARCSEC)
        print('%s' % cmd)
        # os.system('wget "%s" -O %s' % (cmd, filename))    
        os.system('curl -o %s "%s"' % (filename, cmd))
        
    return read_2mass_file(filename, version=-1, Qflg=Qflg, Eflg=Eflg)



def analyze_reference_area(PHOT, dPHOT, weighted=False):
    """
    Given list of stars in the reference field, return vector of average colours and 
    the covariance matrix of the colours.
    """
    N, M = PHOT.shape    # M bands -> (M-1) colours
    REFCOLOURS = zeros(M-1, float32)
    # average colours
    if (weighted):
        # calculate weighted averages
        ok     = ones(PHOT.shape[0], int32)
        for i in range(PHOT.shape[1]): # loop over bands
            m     = nonzero((dPHOT[:,i]>2.0)|(dPHOT[:,i]<=0.0))  # bad stars
            ok[m] = 0
        m      = nonzero(ok==1)      # good stars -- must be exactly the same for all bands
        ave    = zeros(M, float32)   # averages of the bands
        for i in range(M-1):         # loop over colours
            w             = 1.0 /(dPHOT[m[0],i]**2.0 + dPHOT[m[0],i+1]**2.0)
            REFCOLOURS[i] = sum(w*(PHOT[m[0],i]      -  PHOT[m[0],i+1])) / sum(w)         # weighted average
    else:
        # calculate direct averages
        ave    = mean(PHOT, 0)  # column averages
        for i in range(M-1):
            REFCOLOURS[i] = ave[i]-ave[i+1] # should agree with get_nicer_extinctions
    # the covariance matrix
    COL = zeros((N,M-1), float32)
    for i in range(M-1):
        COL[:,i] = PHOT[:,i]-PHOT[:,i+1]
    REFCOVARIANCE = cov(COL, rowvar=False)
    # print N, M, REFCOVARIANCE.shape    
    return REFCOLOURS, REFCOVARIANCE



def get_nicer_extinctions(PHOT, dPHOT, K, REFCOL, REFCOV):
    """
    Calculate extinctions for individual stars given in array PHOT.
    Vector EXT gives ratio between colours and the extinction in the V-band.
    REFCOL contains average colours from the reference field.
    REFCOV is the convariance matric of intrinsic colours (from the reference field).
    This assumes we use colours between consecutive bands 0-1, 1-2, 2-3, ... and 
    REFCOV must corresponds to this convention.
    Return vectors of V-band extinction and calculated errors.
    """
    stars, bands = PHOT.shape
    colours = bands-1
    # construct the matrix
    C = zeros((bands,bands), float32)
    for i in range(colours): # last row and column remain constant
        C[colours, i] = -K[i]
        C[i, colours] = -K[i]
    # the constant right hand side
    b   = zeros((bands,1), float32)
    b[colours] = -1 
    AV  = zeros(stars,float32)
    dAV = zeros(stars,float32)
    # loop over the stars
    for s in range(stars):
        # This is needed if we have more colours...
        # C[colours,:] and C[:,colours] were set outside the loop
        # inside the loop we update only [0:colours]
        C[0:colours, 0:colours] = 0.0 
        # first set to [0:bands,0:bands] part the errors from this star
        # diagonal = var(1)+var(2), var(2)+var(3), var(3)+var(4), ...
        for i in range(colours):
            C[i,i]   =  dPHOT[s,i]**2.0 + dPHOT[s,i+1]**2.0
        # first off-diagonals
        for i in range(colours-1):
            C[i,i+1] = -dPHOT[s,i+1]**2.0
            C[i+1,i] = -dPHOT[s,i+1]**2.0
        # add the covariances from the reference field
        C[0:colours,0:colours] += REFCOV
        # solve the linear equation  C*x=b
        x = solve(C,b)
        # Av estimate is   b0*[(0-1)-(0-1)ref] + ...
        av = 0.0
        for i in range(colours):
            av = av + ((PHOT[s,i]-PHOT[s,i+1])-REFCOL[i])*x[i]
        # the error estimate for the current star
        bb       = x[0:colours].copy()
        bb.shape = (colours)
        dav      = dot(dot(bb, C[0:colours,0:colours]), bb)
        #
        AV[s]  = av
        dAV[s] = sqrt(dav)   # 2016-08-15 --- previously routine returned variances, not std
        if (1):
            if (dAV[s]<=0):
                print('*** [%.3e] %10.3e %10.3e %10.3e   %10.3e %10.3e %10.3e' % \
                (dAV[s], PHOT[s,0],PHOT[s,1],PHOT[s,2],dPHOT[s,0],dPHOT[s,1],dPHOT[s,2]))
        
        if (s%500==0): print('star %5d/%5d   Av %6.3f +- %6.3f' % (s, stars, AV[s], dAV[s]))
    return AV, dAV
        


def get_smoothed_values(COO, AV, dAV, X, Y, FWHM, clip, radec=True):
    """
    Given list of coordinate pairs, data, and error estimates,
    return corresponding extinction and error values that are obtained by averaging 
    over an area that corresponds to the given FWHM value.
    COO   =  [x,y] the coordinates of the stars [radians !!! ... if radec==True ]
    AV    =  Av values of individual stars
    dAV   =  corresponding error estimates
    X,Y   =  vectors of coordinates for which average Av is to be calculated
    FWHM  =  FWHM of the smoothing gaussian [same unit as for coordinates]
    clip  =  sigma-clipping limit in standard deviations
    """
    N       = X.shape[0]
    AA      = zeros(N, float32)
    dAA     = zeros(N, float32)
    K       = 4.0*log(2.0)/(FWHM*FWHM)
    radius2 = (2.0*FWHM)**2.0
    cosy    = 1.0
    for i in range(N):
        x0, y0 = X[i], Y[i]
        # if (i%100==0): print '%5d/%5d %8.4f %8.4f' % (i, N, x0, y0)
        if (radec):
            cosy = cos(y0)
        # calculate distances for all stars
        r2   = ((COO[:,0]-x0)*cosy)**2.0 + (COO[:,1]-y0)**2.0
        # select stars with distance less than 2*FWHM
        mask = numpy.nonzero(r2<radius2)[0]  #  [0] for updated numpy !!!!!!
        # calculate the average weighted with Gaussian and the Av errors
        weight = 0
        A      = 0.0
        for j in range(len(mask)):
            ii      = mask[j]
            d2      = r2[ii]
            w       = exp(-K*d2) / (dAV[ii]*dAV[ii])
            A      += w*AV[ii]
            weight += w
        if (weight>1.0e-10):
            A       = A/weight       # initial estimate
            if (len(mask)>2):
                s   = std(AV[mask])  # ??? using directly the scatter of Av values ???
            else:
                s   = 999.0          # no grounds to exclude any star from the average ???  
            # repeat calculation excluding all stars outside mean()+-clip*std()
            mini    = A-clip*s
            maxi    = A+clip*s
            weight  = 0.0
            A       = 0.0
            s2      = 0.0
            for j in range(len(mask)):
                ii      = mask[j]
                av      = AV[ii]
                if ((av<mini)|(av>maxi)): continue
                d2      = r2[ii]
                w       = exp(-K*d2) / (dAV[ii]*dAV[ii])
                A      += w*av
                s2     += (w*dAV[ii])**2.0
                weight += w
            if (weight>1.0e-10):
                AA[i]   = A/weight        # weighted average
                dAA[i]  = sqrt(s2)/weight # error of weighted average
            else:
                AA[i]  = 0.0
                dAA[i] = 99.0
        else:
            AA[i]  = 0.0
            dAA[i] = 99.0
        if (i%100==10000): 
            print('grid %5d/%5d  -> mask %3d  Av = %6.3f +- %6.3f' % (i, N, len(mask), AA[i], dAA[i]))
    return AA, dAA
    


def extinction_to_fits(F, COO, PHOT, dPHOT, rPHOT, rdPHOT, fwhm, clip=3.0, dF=None, true_error=False,
                       Rv=3.1, excesses=None, return_stars=False, galactic=False):
    """
    Calculate NICER extinction values to a FITS file.
    Assumes J, H, K.
    Inputs:
        F              = pyfits object that defines the pixel positions
        COO            = star coordinates [radians], assume J2000
        PHOT, dPHOT    = photometry and uncertainties for magnitudes in the the on-field
        rPHOT, rdPHOT  = photometry and uncertainties for magnitudes in the the off-field
        fwhm           = resolution [radians]
        clip           = sigma-clipping limit (default 3.0)
        dF             = pyfits object for error map (default None)
        true_error     = use dispersion of Av for individual stars instead of formal errors
                         (default True)
        Rv             = use Cardelli extinction curve with given value of Rv
        excesses       = specify directly the excesses, e.g., E(J-H)/Av, E(H-K)/Av
        return_stars   = if true, Av, dAv for each star
        galactic       = assume FITS is in Galactic coordinate system
    Note: coordinates of COO should be in the same system as coordinates in the FITS.
    """
    #if (excesses==None)
    #    ext          = get_excess_Av_ratios(['J', 'H', 'K'], Rv)
    #else
    ext            = asarray(excesses, float32).copy()
    # extract average colours and covariances from the reference data
    REFCOL, REFCOV = analyze_reference_area(rPHOT, rdPHOT)
    # calculate extinctions for individual stars
    AV, dAV    = get_nicer_extinctions(PHOT, dPHOT, ext, REFCOL, REFCOV)
    # true_error => take Av fluctuation between stars and divide by sqrt(effective number of stars)
    # otherwise  => formal error using the error estimates
    LON, LAT   = PIX2WCS_ALL(F[0].header, radians=True)
    LON, LAT   = ravel(LON), ravel(LAT)
    if (galactic):
        print('*** Converting J2000 stars for a Galactic fits image!!\n')
        ra, de  = pyx_wcscon_rad(LON, LAT, WCS_GALACTIC, WCS_J2000)
        AA, dAA = pyx_get_smoothed_values_2(COO, AV, dAV, ra, de, fwhm, clip, radec=True, true_error=true_error)
    else:
        AA, dAA = pyx_get_smoothed_values_2(COO, AV, dAV, LON, LAT, fwhm, clip, radec=True, true_error=true_error)
    AA.shape   = F[0].data.shape  # our final extinction map
    dAA.shape  = F[0].data.shape  # the error map
    # fix missing extinction values =>  Av=0 => Av=max(Av)
    m     = nonzero(abs(AA)<1e-9)
    AA[m] = 0.0
    print('Missing valus: %d pixels setp to zero' % len(m[0]))
    F[0].data  = AA
    if (dF!=None):
        dF[0].data = dAA
    if (return_stars):
        return AV, dAV
        


def NICER_with_OpenCL(F, COO, PHOT, dPHOT, rPHOT, rdPHOT, EX_AV, FWHM, CLIP=3.0, 
    CLIP_UP=None, CLIP_DOWN=None, dF=None, 
    TRUE_ERROR=False, RETURN_STARS=False, AGRID=[], ASTAR=[], GPU=False):
    """
    Calculate NICER extinction values to a FITS file, data limited to three bands.
    Inputs:
        F              = pyfits object that defines the pixel positions, will contain Av
        COO            = star coordinates array[n, 3] in radians, assumes J2000 coordinates
        PHOT, dPHOT    = photometry and magnitudes erros in the the on-field, arrays[n,3]
        rPHOT, rdPHOT  = photometry and errors for the off-field, arrays[n,3]
        EX_AV          = specify colour excesses, e.g., E(J-H)/Av, E(H-K)/Av        
        FWHM           = resolution [radians]
        CLIP           = sigma clipping limit
        CLIP_UP        = override sigma-clipping limit upwards
        CLIP_DOWN      = optional, different limit downwards
        dF             = pyfits object for error map (default None)
        TRUE_ERROR     = use dispersion of Av for individual stars instead of formal errors
                         (default is False)
        RETURN_STARS   = if true, Av, dAv for each star
        AGRID          = optional, reference tau for output grid points (smoothed)
        ASTAR          = optional, reference tau for each star (high resolution)    
    Note: coordinates of COO should be in the same system as coordinates in the FITS.
    Note: each entry of AGRID<=0.0 and every entry of ASTAR<=0.0 are ignored => no weighting
    """
    # extract average colours and covariances from the reference data
    REFCOL, REFCOV =  analyze_reference_area(rPHOT, rdPHOT)
    # calculate extinctions for individual stars
    if (1):
        AV, dAV    =  get_nicer_extinctions_OpenCL(PHOT, dPHOT, EX_AV, REFCOL, REFCOV, GPU)
    else:
        AV2, dAV2  =  get_nicer_extinctions(PHOT, dPHOT, EX_AV, REFCOL, REFCOV)
    # need to get coordinates [radians] for the pixels = (LON, LAT)
    LON, LAT   =  PIX2WCS_ALL(F[0].header, radians=True)
    LON, LAT   = ravel(LON), ravel(LAT)
    # spatial averaging
    clip_down, clip_up = CLIP, CLIP
    if (CLIP_DOWN!=None):  clip_down = CLIP_DOWN
    if (CLIP_UP  !=None):  clip_up   = CLIP_UP
    AA, dAA = get_smoothed_values_OpenCL(COO, AV, dAV, LON, LAT, FWHM, 
              clip_up, clip_down, TRUE_ERROR=TRUE_ERROR, AGRID=AGRID, ASTAR=ASTAR, GPU=GPU)
    AA.shape       =  F[0].data.shape  # our final extinction map
    dAA.shape      =  F[0].data.shape  # the error map
    # fix missing extinction values =>  Av=0 => Av=max(Av)
    m              =  nonzero(abs(AA)<1.0e-9)
    AA[m]          =  0.0
    print('Missing values: %d pixels setp to zero' % len(m[0]))
    F[0].data      =  AA
    if (dF!=None):
        dF[0].data    = dAA
        dF[0].data[m] = 99.0
    if (RETURN_STARS):
        return AV, dAV

    
    
def get_nicer_extinctions_OpenCL(PHOT, dPHOT, EX_AV, REFCOL, REFCOV, GPU=0, platforms=arange(6), local=-1):
    """
    Use NICER to calculate extinction estimates for stars.
    """
    N          =  PHOT.shape[0] # number of stars
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms)
    LOCAL      =  [8, 32][GPU>0]
    if (local>0): LOCAL = local
    OPT        = " -D STARS=%d -D BANDS=3 -D COLOURS=2 -D CLIP_UP=0 -D CLIP_DOWN=0 -D NPIX=0 -D FWHM=1e-10f" % N
    source     =  open(ISM_DIRECTORY+"/ISM/Extinction/kernel_Nicer.c").read()
    program    =  cl.Program(context, source).build(OPT)
    ##
    k          =  asarray(EX_AV, float32)
    refcol     =  asarray(ravel(REFCOL), float32)
    refcov     =  asarray(ravel(REFCOV), float32)
    mag        =  asarray(ravel(PHOT), float32)
    dmag       =  asarray(ravel(dPHOT), float32)
    Av         =  zeros(N, float32)
    dAv        =  zeros(N, float32)
    ##
    K_buf      =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k)
    REFCOL_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=refcol)
    REFCOV_buf =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=refcov)
    MAG_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mag)
    dMAG_buf   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dmag)
    Av_buf     =  cl.Buffer(context, mf.WRITE_ONLY, Av.nbytes)
    dAv_buf    =  cl.Buffer(context, mf.WRITE_ONLY, dAv.nbytes)
    ##
    nicer    = program.nicer
    nicer.set_scalar_arg_dtypes([None, None, None, None, None, None, None])
    GLOBAL   = (int(N/64)+1)*64
    nicer(queue, [GLOBAL,], [LOCAL,], K_buf, REFCOL_buf, REFCOV_buf, MAG_buf, dMAG_buf, 
                                      Av_buf, dAv_buf)
    cl.enqueue_copy(queue,  Av, Av_buf)
    cl.enqueue_copy(queue, dAv, dAv_buf)
    return Av, dAv

                    
    
def get_smoothed_values_OpenCL(COO, AV, dAV, LON, LAT, FWHM, CLIP_UP, CLIP_DOWN,
              TRUE_ERROR=False, AGRID=[], ASTAR=[], GPU=False, platforms=arange(6), local=-1):
    """
    Calculate averaged Av values for pixels, based on the Av values of individual stars.
    Input:
        COO        =   [n,2] array of coordinates of stars [radians]
        AV         =   extinction values of the stars
        dAV        =   Av uncertainty of individual stars
        LON        =   longitude values of the output grid
        LAT        =   latitude values of the output grid
        FWHM       =   FWHM value [radians] of the output grid
        CLIP_UP    =   sigma clipping threshold for Av > <Av>
        CLIP_DOWN  =   sigma clipping threshold for Av < <Av>
        TRUE_ERROR =   if True, use scatter of individual stars instead of the formal error
        AGRID      =   optional, reference column density at grid points
        ASTAR      =   optional, reference column density at star positions
        platforms  =   list of possible OpenCL platforms, default is arange(6)
        local      =   if >0, set size of local work groups (default local=-1)
    Note:
        All values AGRID<=0.0 and all values ASTAR<=0.0 are ignored (no weighting with reference)
    """
    STARS    =  COO.shape[0] # number of stars
    NPIX     =  len(LON)
    platform, device, context, queue, mf = InitCL(GPU=GPU, platforms=platforms)
    LOCAL    =  [ 8, 32 ][GPU>0]
    if (local>0): LOCAL = local
    OPT      = "-D STARS=%d -D FWHM=%.6ef -D CLIP_UP=%.5ef -D CLIP_DOWN=%.5e -D TRUE_ERROR=%d -D NPIX=%d -D BANDS=3 -D COLOURS=2" % \
                  (STARS,      FWHM,         CLIP_UP,         CLIP_DOWN,        TRUE_ERROR,      NPIX)
    source   =  open(ISM_DIRECTORY+"/ISM/Extinction/kernel_Nicer.c").read()
    program  =  cl.Program(context, source).build(OPT)
    ##
    RA_buf   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(COO[:,0].copy(), float32))
    DE_buf   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(COO[:,1].copy(), float32))
    Av_buf   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(AV, float32))
    dAv_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(dAV, float32))
    LON_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(LON, float32))
    LAT_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(LAT, float32))
    SA, dSA  =  zeros(NPIX, float32),  zeros(NPIX, float32)
    SA_buf   =  cl.Buffer(context, mf.WRITE_ONLY,  SA.nbytes)
    dSA_buf  =  cl.Buffer(context, mf.WRITE_ONLY, dSA.nbytes)
    ##
    GLOBAL   = (int(NPIX/64)+1)*64
    if (len(AGRID)<2): # no reference data used
        print('Normal smoothing\n')
        smooth      =  program.smooth
        smooth.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None])
        smooth(queue, [GLOBAL,], [LOCAL,], RA_buf, DE_buf, Av_buf, dAv_buf, LON_buf, LAT_buf, SA_buf, dSA_buf)
    else:              # including the AGRID and ASTAR arrays
        print('Smoothing with the help of reference data\n')
        AGRID_buf   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(AGRID, float32))
        ASTAR_buf   =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=asarray(ASTAR, float32))
        smooth      =  program.smoothX
        smooth.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, None, None])
        smooth(queue, [GLOBAL,], [LOCAL,], RA_buf, DE_buf, Av_buf, dAv_buf, LON_buf, LAT_buf, SA_buf, dSA_buf, AGRID_buf, ASTAR_buf)
    ###
    cl.enqueue_copy(queue,  SA,  SA_buf)
    cl.enqueue_copy(queue, dSA, dSA_buf)
    ##
    return SA, dSA

    
