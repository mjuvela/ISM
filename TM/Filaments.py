from MJ.mjDefs import *
from MJ.WCS import get_fits_pixel_size
import time
import numpy as np
import pyopencl as cl



def Centipede(F, FWHM_AM, LEGS=3, STUDENT=1, NDIR=17, GPU=0):
    """
    Analyse image by fitting centipede.
    Input:
        F        = pyfits image
        FWHM_AM  = FWHM in arcmin
        LEGS     = number of pairs of legs
        STUDENT  = standardize fitted data
        NDIR     = number of centipede directions
        GPU      = GPU>0 == use GPU
    Return:
        SIG, PA, SS = significant, position angle, filtered image
    """
    PIX       =  get_fits_pixel_size(F)*DEGREE_TO_RADIAN        
    FWHM      =  FWHM_AM*ARCMIN_TO_RADIAN/PIX  # in pixels
    FWHM_HPF  =  2.0*FWHM                      # scale for high pass filter
    N, M      =  F[0].data.shape
    # 
    t0       = time.time()
    platform = cl.get_platforms()[0]
    if (GPU>0):
        device   = platform.get_devices(cl.device_type.GPU)
        LOCAL    = 64
    else:
        device   = platform.get_devices(cl.device_type.CPU)
        LOCAL    =  8
    context  = cl.Context(device)
    queue    = cl.CommandQueue(context)
    mf       = cl.mem_flags
    OPT      = " -D FWHM=%.3f -D FWHM_HPF=%.3f -D N=%d -D M=%d" % (FWHM, FWHM_HPF, N, M)
    OPT     += " -D LEGS=%d -D NDIR=%d -D STUDENT=%d" % (LEGS, NDIR, STUDENT)
    source   =  file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_centipede.c').read()
    program  = cl.Program(context, source).build(OPT)
    print('--- initialisation    %5.2f seconds' % (time.time()-t0))
    #
    S        = asarray(F[0].data, float32)
    SS       = np.empty_like(S)
    S_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf   = cl.Buffer(context, mf.READ_WRITE, SS.nbytes)
    SIG_buf  = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    PA_buf   = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    # Smoothing of the map
    SMOO     = program.Smooth
    SMOO.set_scalar_arg_dtypes([None, None])
    t0       = time.time()
    GLOBAL   = (int((N*M)/64)+1)*64
    SMOO(queue, [GLOBAL,], [LOCAL,], S_buf, SS_buf)
    cl.enqueue_copy(queue, SS, SS_buf)
    print('--- smoothing         %5.2f seconds' % (time.time()-t0))
    # Centipede fit
    t0       = time.time()
    CP       = program.Centipede
    CP.set_scalar_arg_dtypes([None, None, None])
    t0       = time.time()
    CP(queue, [GLOBAL,], [LOCAL,], SS_buf, SIG_buf, PA_buf)
    SIG      = np.empty_like(S)
    PA       = np.empty_like(S)
    cl.enqueue_copy(queue, SIG, SIG_buf)
    cl.enqueue_copy(queue, PA,  PA_buf)
    print('--- centipede         %5.2f seconds' % (time.time()-t0))
    # return maps of significance and position angles
    return SIG, PA, SS




def pyFTrack(S, P, TOP=1.0, BTM=0.5, FWHM=5.0, delta=40.0*DEGREE_TO_RADIAN, REDUCE=0.5):
    """
    Trace filaments
    Input:
        S     =  significance map
        P     =  map of position angles
        TOP   =  minimum peak value of a filament (from S)
        BTM   =  minimum value within a filament (from S)
        FWHM  =  FWHM [pixels] assumed in the analysis
        delta =  limit on changes in direction [radians]
    Returns:
        List of filaments, each filament Nx2 array
    """
    NN, MM  = S.shape
    K       = 4.0*log(2.0)/(FWHM*FWHM)
    STEP    = FWHM * 0.5
    S[nonzero(~isfinite(S))] = 0.0
    M                        = S.copy()  # mark used pixels here
    J, I = indices(M.shape, float32)
    FIL =  1
    FILAMENTS = []
    while (1):  # while we find another filament above TOP
        # select the next best filament = initial pixel
        m       =  nonzero(M>0.0)
        if (len(m[0])<1): 
            break
        y       =  max(ravel(M[m])) ;
        if (y<TOP): break # no more pixels above the upper threshold
        m       =  nonzero(M==y)
        jO, iO  =  m[0][0], m[1][0]
        points  =  0
        POINTS  =  []
        no1     =  0
        print('NEW FILAMENT -- (iO, jO) %4d %4d --- %.3e %.3e' % (iO, jO, S[jO,iO], M[jO,iO]))
        for DIR in [+1, -1]:  # upstream or downstream
            if (DIR==-1): 
                no1  = len(POINTS)  # number of points in +DIR
            i0, j0   = iO, jO       # the original starting point
            M[jO, iO] = 0
            # pold determines the direction, upstream or down
            pold     = P[j0,i0]
            if (DIR==-1):
                pold += pi
            pold =  (pold+4.0*pi) % (2.0*pi)    # angles [0,2*pi]
            while(1):  # step along the filament
                # weighted PA around the current position
                r2    =  (J-j0)**2.0 + (I-i0)**2.0
                m     =  nonzero((r2<(2.0*FWHM**2.0))&(S>BTM))
                pvec  =  P[m]
                svec  =  S[m]
                rvec  =  r2[m]
                # transform all pvec [-0.5*pi,+0.5*pi] to angles closest to pold [0,2.0*pi]
                pvec[nonzero(pvec>(pold+0.5*pi))] -= pi
                pvec[nonzero(pvec>(pold+0.5*pi))] -= pi
                pvec[nonzero(pvec<(pold-0.5*pi))] += pi
                pvec[nonzero(pvec<(pold-0.5*pi))] += pi
                # average only those with PA within delta of the PA[j0,i0]
                ok  =  nonzero(abs(pvec-pold)<delta)
                if (len(ok[0])<1):
                    break 
                w   =  svec[ok] * exp(-K*rvec[ok])
                p0  =  sum(pvec[ok]*w)/sum(w)
                if (points==0):  # initial position... do not step anywhere yet
                    i1, j1 = i0, j0
                else:
                    # One step in the direction of the PA
                    i1   = int(i0 - sin(p0)*STEP)
                    j1   = int(j0 + cos(p0)*STEP)
                    pold = p0
                if ((i1<0)|(j1<0)|(i1>=MM)|(j1>=NN)):
                    break  # next position outside image
                if (1):
                    if ((M[j1,i1]<0.0)&(M[j1,i1]>-FIL)):  # stop if we find processed filament
                        break 
                # take the orthogonal profile, FWHM/2 steps, ~6 points each side
                # only pixels with signal>BTM and angle within delta
                po   = p0+0.5*pi
                pro  = zeros(11, float32)
                for k in range(-5,+6):  # cross profile
                    i        =  i1 - k*sin(po)*0.25*FWHM
                    j        =  j1 + k*cos(po)*0.25*FWHM
                    if ((i<0)|(j<0)|(i>=MM)|(j>=NN)): continue
                    pro[k+5] =  S[int(j),int(i)]
                    dp       =  abs(p0 - P[int(j), int(i)])
                    if ((dp<delta)|((pi-dp)<delta)): # ok - angle is close enough
                        pass
                    else:  # not part of profile of the current filament?
                        pro[k+5] = 0.0
                # mask everything within X*FWHM, if also the angles agree
                r2    =  (J-j1)**2.0 + (I-i1)**2.0
                da    =  abs(P-p0) % pi
                m     =  nonzero((r2<(4.0*FWHM**2.0)) & ((da<delta) | (da>(pi-delta))))
                M[m]  = -FIL
                # ignore also all profile points below BTM
                pro[nonzero(pro<BTM)] = 0.0
                if (sum(pro)<1e-5):
                    break # no valid pixels in the profile
                # centre
                k   =  sum(arange(-5,+6)*pro)/sum(pro)
                # next position along the filament
                i0  =  i1 - k*sin(po)*0.25*FWHM
                j0  =  j1 + k*cos(po)*0.25*FWHM
                if ((i0<0)|(j0<0)|(i0>=MM)|(j0>=NN)):
                    break
                # stop once signal drops below BTM
                if (S[int(j0), int(i0)]<BTM):
                    break # end of current filament
                POINTS.append([i1, j1])
                points += 1
        FIL += 1 
        NP = len(POINTS)    
        # print('ONE FILAMENT READY, %d points. %d in forward direction' % (NP, no1))
        if (NP>2):
            # NP points of which no1 in the DIR=+1 direction
            PT = zeros((NP,2), float32)
            # DIR=+1
            for i in range(no1):
                PT[NP-no1+i,:]  = POINTS[i]
            # then DIR=-1
            for i in range(no1, NP):
                PT[NP-i-1]        = POINTS[i]
            # plot
            # plot(PT[:,0], PT[:,1], 'o-')
            FILAMENTS.append(PT)
            # reduce significance to avoid double filaments
            for i in range(NP):
                r2    =  (J-PT[i,1])**2.0 + (I-PT[i,0])**2.0
                m     =  nonzero(r2<(1.0*FWHM**2.0))
                S[m] *=  1.0 - REDUCE*exp(-K*r2[m])
    ###
    return FILAMENTS




def FTrack(S, P, TOP, BTM, FWHM=5.0, delta=40.0*DEGREE_TO_RADIAN, OVERLAP=2, REPEL=1.0, GPU=0):
    #
    m        =  nonzero(S<BTM)
    S[m]     =  0.0
    P[m]     =  0.0
    S[nonzero(~isfinite(S))] = 0.0
    P[nonzero(~isfinite(P))] = 0.0
    N, M     =  S.shape
    K        =  4.0*log(2.0)/(FWHM*FWHM)
    STEP     =  FWHM * 0.5
    MAXPOS   =  100
    MASK     =  asarray(S.copy(), float32)
    # OpenCL routine to track the filaments
    t0       =  time.time()
    platform =  cl.get_platforms()[0]
    device   =  platform.get_devices(cl.device_type.CPU)
    LOCAL    =  8
    context  =  cl.Context(device)
    queue    =  cl.CommandQueue(context)
    mf       =  cl.mem_flags
    OPT      =  "-D FWHM=%.3ff -D N=%d -D M=%d -D BTM=%.3ff -D MAXPOS=%d -D OVERLAP=%d -D REPEL=%.2f" % \
    (FWHM, N, M, BTM, MAXPOS, OVERLAP, REPEL)
    source   =  file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_ftrack.c').read()
    program  =  cl.Program(context, source).build(OPT)
    print('--- initialisation    %5.2f seconds' % (time.time()-t0))
    #
    S_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    P_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P)
    # Initial positions (X0, Y0)
    # Space to save the positions along filaments
    X, Y     =  zeros(2*MAXPOS, int32), zeros(2*MAXPOS, int32)
    LEN      =  zeros(2, int32)
    X_buf    =  cl.Buffer(context, mf.WRITE_ONLY, X.nbytes)
    Y_buf    =  cl.Buffer(context, mf.WRITE_ONLY, Y.nbytes)
    LEN_buf  =  cl.Buffer(context, mf.WRITE_ONLY, LEN.nbytes)
    MASK_buf =  cl.Buffer(context, mf.READ_WRITE, MASK.nbytes)
    # The program
    PRO      =  program.Follow
    PRO.set_scalar_arg_dtypes([None, None, None, np.int32, np.int32, None, None, None])
    PROM     =  program.Mask
    PROM.set_scalar_arg_dtypes([None, None, None, None, int32, int32])
    cl.enqueue_copy(queue, MASK_buf, MASK)
    # 
    J, I     =  indices(S.shape, float32)
    FIL      =  []
    RADIUS2  =  (FWHM)**2.0   # for masking
    COUNT    =  0
    TDEV, THOST, TMASK, TADD = 0.0, 0.0, 0.0, 0.0
    while (1):  # while we have any pixel above TOP
        t0   =  time.time()
        ii   =  argmax(ravel(MASK))
        X0   =  ii % M
        Y0   =  ii / M
        y    =  MASK[Y0, X0] 
        if (y<TOP): break
        if (COUNT%20==1): print('==== #%03d  Peak %3d %3d   %6.2f > %6.2f ====' % (COUNT, X0, Y0, y, TOP))
        THOST  += time.time()-t0
        t0      = time.time()
        PRO(queue, [8,], [LOCAL,], S_buf, P_buf, MASK_buf, X0, Y0, X_buf, Y_buf, LEN_buf)
        cl.enqueue_copy(queue, X,   X_buf)
        cl.enqueue_copy(queue, Y,   Y_buf)
        cl.enqueue_copy(queue, LEN, LEN_buf)
        queue.finish()
        TDEV  += time.time()-t0
        t0     = time.time()
        if ((LEN[0]+LEN[1])>0): # stitch them together
            t2 = time.time()
            n                   =  LEN[0] + 1 + LEN[1]
            fil                 =  zeros((n, 2), float32)
            fil[0:LEN[1],   0]  = X[(2*MAXPOS-LEN[1]):(2*MAXPOS)]
            fil[LEN[1],     0]  = X0
            fil[LEN[1]+1:,  0]  = X[0:LEN[0]]
            fil[0:LEN[1],   1]  = Y[(2*MAXPOS-LEN[1]):(2*MAXPOS)]
            fil[LEN[1],     1]  = Y0
            fil[LEN[1]+1:,  1]  = Y[0:LEN[0]]
            FIL.append(fil)
            TADD += time.time()-t2
            t2 = time.time()
            if (0):  # HOST 11.3 sec, MASK 8.7 sec
                for i in range(n): # mask the current filament
                    r2 = (J-fil[i,1])**2.0 + (I-fil[i,0])**2.0
                    MASK[nonzero(r2<RADIUS2)] = NaN
            else:    # HOST  5.3 sec, MASK 0.1 sec
                # do this on device --  MASK, X, Y, LEN already on device
                GLOBAL =  ((1+LEN[0]+LEN[1])/LOCAL+1)*LOCAL
                PROM(queue, [GLOBAL,], [LOCAL,], MASK_buf, X_buf, Y_buf, LEN_buf, X0, Y0)
                cl.enqueue_copy(queue, MASK, MASK_buf)
                queue.finish()
            TMASK += time.time()-t2
        else:          # isolated pixels??
            if (0):
                r2 = (J-Y0)**2.0 + (I-X0)**2.0
                MASK[nonzero(r2<RADIUS2)] = NaN
            else:
                PROM(queue, [LOCAL,], [LOCAL,], MASK_buf, X_buf, Y_buf, LEN_buf, X0, Y0)
                cl.enqueue_copy(queue, MASK, MASK_buf)
                queue.finish()
        COUNT += 1
        THOST += time.time()-t0
        
        if (0):
            if (COUNT>100):
                clf()
                imshow(MASK)
                colorbar()
                SHOW()
                sys.exit()
    print('HOST %.1f sec of which MASK %.1f sec, ADD %.1f sec; DEVICE %.1f sec' % \
    (THOST, TMASK, TADD, TDEV))
    
    if (0):
        clf()
        figure(2)
        imshow(MASK)
        colorbar()
        figure(1)
        
    return FIL
                                            
