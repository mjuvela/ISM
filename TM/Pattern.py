from   MJ.mjDefs import *
from   MJ.WCS import get_fits_pixel_size
import time
import numpy as np
import pyopencl as cl


def RHT_ave_angle(si, co):
    T  = fmod(pi+0.5*arctan2(si, co), pi)
    T[nonzero(T>0.5*pi)] -= pi   # make angle east from north
    return T
            

def RollingHoughTransform(F, DK, DW, THRESHOLD, GPU=0, GPU_LOCAL=-1, iplatform=0):
    """
    Calculate Rolling Hough Transform for image F.
    R, SIN, COS, T, SS = RollingHoughTransform(F, DK, DW, THRESHOLD, GPU=0)
    Input:
        F         =  pyfits image to be analysed
        DK        =  diameter of the convolution kernel  (smaller)
        DW        =  diameter of the region = length of the bar
        THRESHOLD =  fraction of SCALE for ON pixels
        GPU_LOCAL =  override default value of 64 (local work group size)
        iplatform =  platform number (default=0)
    """
    t0       = time.time()
    platform = cl.get_platforms()[iplatform]
    if (GPU>0):
        device   = platform.get_devices(cl.device_type.GPU)
        LOCAL    = 64
        if (GPU_LOCAL>0): LOCAL = GPU_LOCAL                
    else:
        device   = platform.get_devices(cl.device_type.CPU)
        LOCAL    =  8
    N, M     = F[0].data.shape
    context  = cl.Context(device)
    queue    = cl.CommandQueue(context)
    mf       = cl.mem_flags
    NDIR     = int(floor((3.14159/sqrt(2.0))*(DW-1.0)+1.0))
    print("NDIR %d" % NDIR)
    OPT      = " -D DK=%d -D DW=%d -D N=%d -D M=%d" % (DK, DW, N, M)
    OPT     += " -D NDIR=%d  -D Z=%.5e " % (NDIR, THRESHOLD)
    source   = file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_RHT.c').read()
    program  = cl.Program(context, source).build(OPT)
    print('--- initialisation  %5.2f seconds:: DK %d DW %d' % (time.time()-t0, DK, DW))
    #
    S        = asarray(F[0].data, float32)
    SS       = np.empty_like(S)
    S_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf   = cl.Buffer(context, mf.READ_WRITE, S.nbytes)
    RR_buf   = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    SIN_buf  = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    COS_buf  = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    # Smoothing of the map
    SMOO     = program.Smooth
    SMOO.set_scalar_arg_dtypes([None, None])
    t0       = time.time()
    GLOBAL   = (int((N*M)/64)+1)*64
    SMOO(queue, [GLOBAL,], [LOCAL,], S_buf, SS_buf)
    cl.enqueue_copy(queue, SS, SS_buf)
    print('--- smoothing       %5.2f seconds' % (time.time()-t0))
    # RHT
    t0       = time.time()
    RHT      = program.R_kernel
    RHT.set_scalar_arg_dtypes([None, None, None, None])
    t0       = time.time()
    RHT(queue, [GLOBAL,], [LOCAL,], SS_buf, RR_buf, SIN_buf, COS_buf)
    R        = np.empty_like(S)
    SIN      = np.empty_like(S)
    COS      = np.empty_like(S)
    cl.enqueue_copy(queue, R,   RR_buf) 
    # R is at this point the sum of NDIR samples ... should be integral over 2:pi
    R       *=  2.0*pi/NDIR   # integration over pi radians?
    cl.enqueue_copy(queue, SIN, SIN_buf)
    cl.enqueue_copy(queue, COS, COS_buf)
    print('--- RHT             %5.2f seconds' % (time.time()-t0))
    T        = RHT_ave_angle(SIN, COS)
    #
    return R, SIN, COS, T, SS



def LIC(T, fwhm, GPU=0):
    """
    Make Line Integration Convolution map for T, which is an image
    of the angles (east from north).
    """
    t0       = time.time()
    platform, device, context, queue = None, None, None, None
    for iplatform in range(3):
        try:
            platform = cl.get_platforms()[iplatform]
            if (GPU>0):
                device   = platform.get_devices(cl.device_type.GPU)
                LOCAL    = 64
            else:
                device   = platform.get_devices(cl.device_type.CPU)
                LOCAL    =  8
            context  = cl.Context(device)
            queue    = cl.CommandQueue(context)
            print('Platform %d has %s:' % (iplatform, ['CPU', 'GPU'][GPU>0]), device)
            break
        except:
            pass
    N, M     = T.shape
    mf       = cl.mem_flags
    OPT      = "-D N=%d -D M=%d -D FWHM=%.3f" % (N, M, fwhm)
    source   = file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_LIC.c').read()
    program  = cl.Program(context, source).build(OPT)
    print('--- initialisation  %5.2f seconds' % (time.time()-t0))
    #
    D        = asarray(randn(N*M).reshape(N,M), float32)
    L        = np.empty_like(D)
    S_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=D)
    T_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=T)
    L_buf    = cl.Buffer(context, mf.WRITE_ONLY, D.nbytes)
    # LIC routine
    FUN      = program.LIC
    FUN.set_scalar_arg_dtypes([None, None, None])
    t0       = time.time()
    GLOBAL   = (int((N*M)/64)+1)*64
    FUN(queue, [GLOBAL,], [LOCAL,], S_buf, T_buf, L_buf)
    cl.enqueue_copy(queue, L, L_buf)
    print('--- LIC             %5.2f seconds' % (time.time()-t0))
    return L
    


def Centipede(F, FWHM_AM, LEGS=3, STUDENT=1, NDIR=17, GPU=0, K_HPF=2.0, GPU_LOCAL=-1):
    """
    Analyse image by fitting centipede.
    Input:
        F         =  pyfits image
        FWHM_AM   =  FWHM in arcmin
        LEGS      =  number of pairs of legs
        STUDENT   =  standardize fitted data
        NDIR      =  number of centipede directions
        GPU       =  GPU>0 == use GPU
        GPU_LOCAL =  override default value of 64 (local work group size)        
    Return:
        SIG, PA, SS = significant, position angle, filtered image
    """
    PIX       =  get_fits_pixel_size(F)*DEGREE_TO_RADIAN        
    FWHM      =  FWHM_AM*ARCMIN_TO_RADIAN/PIX  # in pixels
    FWHM_HPF  =  K_HPF*FWHM                    # scale for high pass filter
    N, M      =  F[0].data.shape
    # 
    t0       = time.time()
    platform = cl.get_platforms()[0]
    if (GPU>0):
        device   = platform.get_devices(cl.device_type.GPU)
        LOCAL    = 64
        if (GPU_LOCAL>0): LOCAL = GPU_LOCAL                
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
        # print 'ONE FILAMENT READY, %d points. %d in forward direction' % (NP, no1)
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




def FTrack(S, P, TOP, BTM, FWHM=5.0, delta=40.0*DEGREE_TO_RADIAN, OVERLAP=2, REPEL=1.0, 
           GPU=0, MIN_LEN=2):
    """
    Trace the filaments using OpenCL kernel
    Input: 
        S        =  input image (2d array)
        P        =  map of position angles (2d array)
        TOP      =  minimum peak value (units of S)
        BTM      =  minimum value within filaments (units of S)
        FWHM     =  scale in pixels
        delta    =  limit on the change of direction (radians)
        OVERLAP  =  maximum overlap between two filaments
        REPEL    =  repel between filaments
        GPU      =  whether to use CPU or GPU
        MIN_LEN  =  minimum length in units of FWHM
    Return:
        FIL      =  [ [x1[], y1[]], [ x2[], y2[] ],  ... ]
    """
    m        =  nonzero(S<BTM)
    S[m]     =  0.0
    P[m]     =  0.0
    S[nonzero(~isfinite(S))] = 0.0
    P[nonzero(~isfinite(P))] = 0.0
    N, M     =  S.shape
    #### K        =  4.0*log(2.0)/(FWHM*FWHM)
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
        # length must be at least MIN_LEN times FWHM == at least MIN_LEN*2 points
        if ((LEN[0]+LEN[1])>=(2*MIN_LEN)): # stitch them together
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
                                            
    





def FTrack2(S, P, TOP, BTM, FWHM=5.0, DTHETA=40.0*DEGREE_TO_RADIAN, OVERLAP=2, REPEL=1.0, 
           GPU=0, MIN_LEN=2, STEP=0.5):
    """
    Trace the filaments
    Input: 
        S        =  input image (2d array)
        P        =  map of position angles (2d array)
        TOP      =  minimum peak value (units of S)
        BTM      =  minimum value within filaments (units of S)
        FWHM     =  scale in pixels
        DTHETA   =  limit on the change of direction (radians)
        OVERLAP  =  maximum overlap between two filaments
        REPEL    =  repel between filaments
        GPU      =  whether to use CPU or GPU
        MIN_LEN  =  minimum length in units of FWHM
        STEP     =  step in units of pixels
    Return:
        FIL      =  [ [x1[], y1[]], [ x2[], y2[] ],  ... ]
    """
    m        =  nonzero(S<BTM)
    S[m]     =  0.0
    P[m]     =  0.0
    S[nonzero(~isfinite(S))] = 0.0
    P[nonzero(~isfinite(P))] = 0.0
    N, M     =  S.shape
    #### K        =  4.0*log(2.0)/(FWHM*FWHM)
    MAXPOS   =  200
    MASK     =  asarray(S.copy(), float32)
    # OpenCL routine to track the filaments
    t0       =  time.time()
    platform =  cl.get_platforms()[0]
    device   =  platform.get_devices(cl.device_type.CPU)
    LOCAL    =  8
    context  =  cl.Context(device)
    queue    =  cl.CommandQueue(context)
    mf       =  cl.mem_flags
    OPT      =  "-D FWHM=%.3ff -D N=%d -D M=%d -D BTM=%.3ff -D MAXPOS=%d \
    -D OVERLAP=%d -D REPEL=%.2f -D STEP=%.3f -D DTHETA=%.5f" % \
    (FWHM, N, M, BTM, MAXPOS, 
    OVERLAP, REPEL, STEP, DTHETA)
    source   =  file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_ftrack2.c').read()
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
    if (0):
        PRO      =  program.Follow
    else:
        PRO      =  program.FollowSimple
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
        # length must be at least MIN_LEN times FWHM == at least MIN_LEN*2 points
        if ((LEN[0]+LEN[1])>=(2*MIN_LEN)): # stitch them together
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
        
    return FIL  #   [  [x, y] ... ]    x = column, y = row
                                            



def PatternMatch(F, FWHM_AM, FILTER, STUDENT, PAT, THRESHOLD, GPU=0, NDIR=21, 
                 K_HPF=2.0, SYMMETRIC=False, STEP=1.0, AAVE=0, FWHM_PIX=-1,
                 GPU_LOCAL=-1, HOST_CONVOLUTION=False):
    """
    Run pattern matching on the image F.
    Usage:
        SIG, PA, SS = PatternMatch(F, FWHM_AM, FILTER, STUDENT, PAT, THRESHOLD, GPU=0, NDIR=21,
                         K_HPF=2.0, SYMMETRIC=False, STEP=1.0, AAVE=0)
    Input:
        F         = pyfits image (in the first HDU, pixels aligned with coordinate axes, east on the left)
        FWHM_AM   = FWHM of the low-pass filter [arcmin]
        FILTER    = if True, include the high-pass and low-pass filtering,
                    otherwise use directly the original image
        STUDENT   = if True, normalise the data below the current template
        PAT       = 2D array giving the template
        THRESHOLD = *** not used ***
        GPU       = 0 for CPU and 1 for GPU calculation
        NDIR      = number of tested position angles
        K_HPF     = ratio FWHM_HPF / FWHM_LPF, default is 2.0
        SYMMETRIC = should be True of template symmetric for 180 degree rotation (PA in [0,180])
        STEP      = step between template elements in units of FWHM_AM,
                    the default value is 1.0
        AAVE      = instead of significance and PA of the best-fitting angle, calculate
                    PA as angular averages (similar to RHT; for AAVE==1), or also
                    calculate angular average of significance (AAVE==2)
        FWHM_PIX  = if >0, overrides FWHM_AM, value in pixels
        GPU_LOCAL = override default value of 64 for the local work group size (GPU only)
        HOST_CONVOLUTION = if True, do FFT convolution on host
    """
    PIX        =  get_fits_pixel_size(F)*DEGREE_TO_RADIAN
    if (FWHM_PIX>0):
        FWHM   =  FWHM_PIX
    else:
        FWHM   =  FWHM_AM*ARCMIN_TO_RADIAN/PIX   # as pixels -- FWHM of LPF == stencil step
    FWHM_HPF   =  K_HPF*FWHM                     # as pixels
    print('PatternMatch FWHM_AM %.3f FWHM_HPF %.3f FILTER %d' % (FWHM_AM, FWHM_AM*K_HPF, FILTER))
    N, M       =  F[0].data.shape
    DIM0, DIM1 =  PAT.shape
    # Kernel assumes the image to be aligned with coordinate axes, east is left!
    t0        =  time.time()
    platform  =  cl.get_platforms()[0]
    if (GPU>0):
        device   = platform.get_devices(cl.device_type.GPU)
        LOCAL    = 64
        if (GPU>1): device = (device[1],)
        if (GPU_LOCAL>0): LOCAL = GPU_LOCAL
    else:
        device   = platform.get_devices(cl.device_type.CPU)
        LOCAL    =  8
    context  =  cl.Context(device)
    queue    =  cl.CommandQueue(context)
    mf       =  cl.mem_flags
    # compiler options (make sure your system does recompile each time this is run)
    OPT      = " -D FWHM=%.3e -D FWHM_HPF=%.3e -D N=%d -D M=%d" % (FWHM, FWHM_HPF, N, M)
    OPT     += " -D DIM0=%d -D DIM1=%d -D NDIR=%d -D SYMMETRIC=%d" % (DIM0, DIM1, NDIR, SYMMETRIC)
    OPT     += " -D STUDENT=%d -D STEP=%.4f -D AAVE=%d" % (STUDENT, STEP, AAVE)    
    source   = file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_PM.c').read()
    program  = cl.Program(context, source).build(OPT)
    print(OPT)
    print('--- initialisation    %5.2f seconds' % (time.time()-t0))
    #
    S        =  asarray(F[0].data.copy(), float32)
    SS       =  zeros((N,M), float32)
    PAT_buf  =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=PAT)
    S_buf    =  cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf   =  cl.Buffer(context, mf.READ_WRITE, SS.nbytes)
    SIG_buf  =  cl.Buffer(context, mf.WRITE_ONLY, 4*N*M)
    PA_buf   =  cl.Buffer(context, mf.WRITE_ONLY, 4*N*M)
    GLOBAL   =  (int((N*M)/64)+1)*64
    if (FILTER>0): # Smoothing of the map
        if (HOST_CONVOLUTION):            # FFT convolution on host
            SS   =  convolve_map_fast(S.copy(), FWHM)
            B    =  convolve_map_fast(S.copy(), FWHM_HPF)
            SS  -=  B
            SS   =  asarray(SS, float32)
            cl.enqueue_copy(queue, SS_buf, SS) 
        else:                             # real space convolution on device
            SMOO     = program.Smooth
            SMOO.set_scalar_arg_dtypes([None, None])
            t0       = time.time()
            SMOO(queue, [GLOBAL,], [LOCAL,], S_buf, SS_buf)
            cl.enqueue_copy(queue, SS, SS_buf)
            print('--- smoothing         %5.2f seconds' % (time.time()-t0))
    else:
        SS = S.copy()
        cl.enqueue_copy(queue, SS_buf, SS)  # put original image to device as SS
    # Pattern matching
    t0       = time.time()
    if (AAVE):
        CP       = program.PatternMatchAA
    else:
        CP       = program.PatternMatch
    CP.set_scalar_arg_dtypes([None, None, None, None])
    t0       = time.time()
    CP(queue, [GLOBAL,], [LOCAL,], PAT_buf, SS_buf, SIG_buf, PA_buf)
    SIG      = np.empty_like(S)
    PA       = np.empty_like(S)
    cl.enqueue_copy(queue, SIG, SIG_buf)
    cl.enqueue_copy(queue, PA,  PA_buf)
    print('--- Pattern matching  %5.2f seconds' % (time.time()-t0))
    print(SIG.shape, PA.shape, SS.shape)
    print('---------------------------------')
    #
    return SIG, PA, SS
    


def CentipedeHealpix(S, FWHM, FWHM_HPF, LEGS=3, NDIR=21, STUDENT=1, GPU=0, GPU_LOCAL=-1):
    """
    Run Centipede on allsky healpix map.
    Input:
        S         =  healpix map
        FWHM      =  FWHM of low pass filter [radians]
        FWHM_HPF  =  FWHM of high pass filter [radians], e.g. 2*FWHM
        LEGS      =  number of pair of legs
        NDIR      =  number of directions (over pi)
        STUDENT   =  if 1, normalise the data
        GPU       =  if 1, run on GPU
        GPU_LOCAL =  override default value of 64 (local work group size)        
    """
    NPIX     = len(S)
    NSIDE    = int(sqrt(NPIX/12))
    print("NPIX %d, NSIZE %d" % (NPIX, NSIDE))
    SS       = np.zeros(NPIX, float32)
    platform = None
    device   = None
    context  = None
    queue    = None
    LOCAL    = 8
    for iplatform in range(3):
        try:
            platform = cl.get_platforms()[iplatform]
            if (GPU>0):
                device   = platform.get_devices(cl.device_type.GPU)
                LOCAL    = 32
                if (GPU_LOCAL>0): LOCAL = GPU_LOCAL        
            else:
                device   = platform.get_devices(cl.device_type.CPU)
                LOCAL    =  8
            context  = cl.Context(device)
            queue    = cl.CommandQueue(context)
            break
        except:
            pass
    #
    mf         =  cl.mem_flags
    OPT        =  " -D NSIDE=%d -D FWHM=%.3e -D FWHM_HPF=%.3e" % (NSIDE, FWHM, FWHM_HPF)
    OPT       +=  " -D LEGS=%d -D NDIR=%d -D STUDENT=%d" % (LEGS, NDIR, STUDENT)
    source     =  file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_centipede_healpix.c').read()
    program    =  cl.Program(context, source).build(OPT)
    S_buf      =  cl.Buffer(context, mf.READ_ONLY, S.nbytes)
    SS_buf     =  cl.Buffer(context, mf.READ_WRITE, SS.nbytes)
    SMOO       =  program.Smooth2
    SMOO.set_scalar_arg_dtypes([None, None])
    t0         =  time.time()
    cl.enqueue_copy(queue, S_buf, S)
    queue.finish()
    SMOO(queue, [NPIX,], [LOCAL,], S_buf, SS_buf)
    queue.finish()
    cl.enqueue_copy(queue, SS, SS_buf)
    print('--- smoothing %5.2f seconds' % (time.time()-t0))
    ##
    SIG_buf    =  cl.Buffer(context, mf.WRITE_ONLY, SS.nbytes)
    PA_buf     =  cl.Buffer(context, mf.WRITE_ONLY, SS.nbytes)
    SIG        =  np.zeros(NPIX, float32)
    PA         =  np.zeros(NPIX, float32)
    CP         =  program.Centipede
    CP.set_scalar_arg_dtypes([None, None, None])
    t0         =  time.time()
    queue.finish()
    CP(queue, [NPIX,], [LOCAL,], SS_buf, SIG_buf, PA_buf)
    queue.finish()
    cl.enqueue_copy(queue, SIG, SIG_buf)
    cl.enqueue_copy(queue, PA,  PA_buf)
    print('--- centipede %5.2f seconds' % (time.time()-t0))
    return SIG, PA, SS
    
    

def RollingHoughTransformHealpix(S, SCALE, K, THRESHOLD, GPU=0, NDIR=25, GPU_LOCAL=-1):
    """
    Rolling Hough Transform in allsky healpix map.
    Input:
        S         =  healpix allsky map
        SCALE     =  scale [radians]
        K         =  high pass filter using K*SCALE
        THRESHOLD =  threshold for ON pixels, fraction of SCALE
        GPU_LOCAL =  override default value of 64 (local work group size)
    Returns:
        RR        =  significance
        PA        =  position angles
        SS        =  thresholded image
    """
    t0       = time.time()
    platform = cl.get_platforms()[0]
    if (GPU>0):
        device   = platform.get_devices(cl.device_type.GPU)
        LOCAL    = 64
        if (GPU_LOCAL>0): LOCAL = GPU_LOCAL        
    else:
        device   = platform.get_devices(cl.device_type.CPU)
        LOCAL    =  8
    #    
    NPIX     = len(S)
    NSIDE    = int(sqrt(NPIX/12))
    SS       = np.empty_like(S)
    
    context  = cl.Context(device)
    queue    = cl.CommandQueue(context)
    mf       = cl.mem_flags
    DK       = float(SCALE)                     # [radians]
    DW       = float(K*SCALE)
    PIX      = sqrt(4.0*pi/(12.0*NSIDE**2.0))   # [radians]
    ## STEP     = 0.6*PIX                          # step used in convolution
    STEP     = 0.46*PIX                          # step used in convolution
    DWPIX    = DW/PIX                           # diameter in terms of pixel size
    NPIX     = len(S)
    OPT      = " -D DK=%.4ef -D DW=%.4ef -D NDIR=%d -D NSIDE=%d -D PIX=%.4ef" % (DK, DW, NDIR, NSIDE,PIX)
    OPT     += " -D Z=%.4ef -D STEP=%.4ef -D NPIX=%d" % (THRESHOLD, STEP, NPIX)
    source   = file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_RHT_healpix.c').read()
    program  = cl.Program(context, source).build(OPT)
    print('--- initialisation  %5.2f seconds' % (time.time()-t0))
    #
    SS       = np.empty_like(S)
    S_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf   = cl.Buffer(context, mf.READ_WRITE, S.nbytes)
    RR_buf   = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    PA_buf   = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    # Smoothing of the map
    SMOO     = program.Smooth
    SMOO.set_scalar_arg_dtypes([None, None])
    t0       = time.time()
    GLOBAL   = NPIX
    if (GLOBAL % LOCAL!=0):
        GLOBAL =  ((GLOBAL/LOCAL)+1)*LOCAL
    SMOO(queue, [GLOBAL,], [LOCAL,], S_buf, SS_buf)
    cl.enqueue_copy(queue, SS, SS_buf)
    print('--- smoothing       %5.2f seconds' % (time.time()-t0))
    
    # RHT
    t0       = time.time()
    RHT      = program.R_kernel
    RHT.set_scalar_arg_dtypes([None, None, None])
    t0       = time.time()
    RHT(queue, [GLOBAL,], [LOCAL,], SS_buf, RR_buf, PA_buf)
    RR       = np.empty_like(S)
    PA       = np.empty_like(S)
    cl.enqueue_copy(queue, RR, RR_buf)
    cl.enqueue_copy(queue, PA, PA_buf)
    print('--- RHT             %5.2f seconds' % (time.time()-t0))
                                                
    return RR, PA, SS
    
    


def PatternMatchHealpix(S, FWHM, FWHM_HPF, PAT,
                        NDIR=21, STUDENT=0, SYMMETRIC=0, DO_SMOOTH=0, GPU=0,
                        GPU_LOCAL=-1):
    """
    Match pattern PAT on healpix map S.
    Input:
        S         =  allsky healpix map (must be in RING order)
        FWHM      =  FWHM [radians]
        FWHM_HPF  =  FWHM of high pass filter [radians]
        PAT       =  the pattern
        NDIR      =  number of directions (over 2pi, or pi, if symmetric)
        SYMMTRIC  =  should be true of left-right symmetric pattern
        DO_SMOOTH =  do the filtering
        GPU       =  if 1, use GPU
        GPU_LOCAL =  override default value of 64 (local work group size)        
    """
    S         =  np.asarray(S, float32)
    SS        =  np.empty_like(S)
    NPIX      =  len(S)
    NSIDE     =  int(sqrt(NPIX/12))    
    DIM       =  PAT.shape[0]
    platform  =  cl.get_platforms()[0]
    if (GPU>0):
        device   = platform.get_devices(cl.device_type.GPU)
        LOCAL    = 64
        if (GPU_LOCAL>0): LOCAL = GPU_LOCAL        
    else:
        device   = platform.get_devices(cl.device_type.CPU)
        LOCAL    =  8
    context  = cl.Context(device)
    queue    = cl.CommandQueue(context)
    mf       = cl.mem_flags
    OPT      = " -D NSIDE=%d -D FWHM=%.3e -D FWHM_HPF=%.3e" % (NSIDE, FWHM, FWHM_HPF)
    OPT     += " -D DIM=%d -D NDIR=%d" % (DIM, NDIR)
    OPT     += " -D STUDENT=%d -D SYMMETRIC=%d" % (STUDENT, SYMMETRIC)
    source   = file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_PM_healpix.c').read()
    program  = cl.Program(context, source).build(OPT)
    PAT_buf  = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=PAT)
    S_buf    = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf   = cl.Buffer(context, mf.READ_WRITE, SS.nbytes)
    if (DO_SMOOTH>0):
        print('Smoothing')
        SMOO       = program.Smooth2
        SMOO.set_scalar_arg_dtypes([None, None])
        t0         = time.time()
        SMOO(queue, [NPIX,], [LOCAL,], S_buf, SS_buf)
        cl.enqueue_copy(queue, SS, SS_buf)
        print('--- smoothing    %5.2f seconds' % (time.time()-t0))
    SIG_buf    = cl.Buffer(context, mf.WRITE_ONLY, SS.nbytes)
    PA_buf     = cl.Buffer(context, mf.WRITE_ONLY, SS.nbytes)
    SIG        = np.empty_like(S)
    PA         = np.empty_like(S)
    PM         = program.PatternMatch
    PM.set_scalar_arg_dtypes([None, None, None, None])
    t0         = time.time()
    if (DO_SMOOTH>0):
        PM(queue, [NPIX,], [LOCAL,], PAT_buf, SS_buf, SIG_buf, PA_buf)
    else:
        PM(queue, [NPIX,], [LOCAL,], PAT_buf, S_buf, SIG_buf, PA_buf)
    cl.enqueue_copy(queue, SIG, SIG_buf)
    cl.enqueue_copy(queue, PA,  PA_buf)
    print('--- PatternMatch %5.2f seconds' % (time.time()-t0))
    return SIG, PA, SS
    




#=================================================================================================



"""
These are more recent than code in MJ.Pattern !
2016-10-27
"""


def RollingHoughTransformBasic(F, DK, DW, FRAC, GPU=0, GPU_LOCAL=-1):
    """
    Calculate Rolling Hough Transform for image F.
    R, THETA, dTHETA = RollingHoughTransform(F, DK, DW, FRAC, GPU=0)
    Input:
        F         =  pyfits image to be analysed
        DK        =  size of convolution kernel
        DW        =  diameter of the region [pixels]
        FRAC      =  fraction of ON pixels used as the threshold
        GPU       =  if >0, use GPU instead of CPU
        GPU_LOCAL =  override default value of 64 (local work group size)
    """
    t0       = time.time()
    platform = cl.get_platforms()[0]
    if (GPU>0):
        device   = platform.get_devices(cl.device_type.GPU)
        LOCAL    = 64        
        if (GPU_LOCAL>0): LOCAL = GPU_LOCAL
        if (GPU>1): device = (device[1],)        
    else:
        device   = platform.get_devices(cl.device_type.CPU)
        LOCAL    =  8
    N, M      = F[0].data.shape
    context   = cl.Context(device)
    queue     = cl.CommandQueue(context)
    mf        = cl.mem_flags
    THRESHOLD = int(FRAC*DW) ;
    NDIR      = int(np.floor((3.14159/np.sqrt(2.0))*(DW-1.0)+1.0))
    OPT       = " -D DK=%d -D DW=%d -D N=%d -D M=%d" % (DK, DW, N, M)
    OPT      += " -D NDIR=%d  -D THRESHOLD=%d" % (NDIR, THRESHOLD)
    # dummy parameters
    OPT      += " -D BORDER=0 -D STEP=0 -D NL=0 -D NW=0 -D NPIX=0"
    ## source    = file(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_mjRHT.c').read()
    source   = file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_RHT2.c').read()    
    program   = cl.Program(context, source).build(OPT)
    print('--- initialisation  %5.2f seconds:: DK %d DW %d' % (time.time()-t0, DK, DW))
    #
    S         = np.asarray(np.ravel(F[0].data), np.float32)
    SS        = np.empty_like(S)
    S_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf    = cl.Buffer(context, mf.READ_WRITE, S.nbytes)
    INT_buf   = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    THETA_buf = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    DTHETA_buf= cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    # Smoothing of the map
    SMOO     = program.Smooth
    SMOO.set_scalar_arg_dtypes([None, None])
    t0       = time.time()
    GLOBAL   = (int((N*M)/64)+1)*64
    SMOO(queue, [GLOBAL,], [LOCAL,], S_buf, SS_buf)
    cl.enqueue_copy(queue, SS, SS_buf)
    print('--- smoothing       %5.2f seconds' % (time.time()-t0))
    if (0):
        clf(),subplot(221),imshow(S.reshape(N,M)),colorbar()
        subplot(222),imshow(SS.reshape(N,M)),colorbar(),show()
        sys.exit()
    # RHT
    t0       = time.time()
    RHT      = program.R_kernel_oneline
    RHT.set_scalar_arg_dtypes([None, None, None, None])
    t0       = time.time()
    RHT(queue, [GLOBAL,], [LOCAL,], SS_buf, INT_buf, THETA_buf, DTHETA_buf)
    INT      = np.empty_like(S)
    THETA    = np.empty_like(S)
    DTHETA   = np.empty_like(S)
    cl.enqueue_copy(queue, INT,     INT_buf)
    cl.enqueue_copy(queue, THETA,   THETA_buf)
    cl.enqueue_copy(queue, DTHETA,  DTHETA_buf)
    print('--- RHT             %5.2f seconds' % (time.time()-t0))
    #
    return  INT.reshape(N,M), THETA.reshape(N,M), DTHETA.reshape(N,M)




def RollingHoughTransformOversampled(F, DK, DW, NW=1, STEP=1.0, FRAC=0.8, GPU=0, GPU_LOCAL=-1):
    """
    Calculate Rolling Hough Transform using oversampling.
    R, THETA, dTHETA = RollingHoughTransformS(F, DK, NL, NW, STEP, FRAC, GPU=0)
    Input:
        F         =  pyfits image to be analysed
        DK        =  size of convolution kernel (smaller than (2*NL+1)*STEP; should be odd ?)
        DW        =  length of the bar [pixels]
        NW        =  steps in each perpendicular direction (total is 2*NW+1)
        STEP      =  step in pixels, values 0<STEP<1 mean oversampling
        FRAC      =  fraction of positive samples used as the threshold
        GPU       =  if >0, use GPU instead of CPU
        GPU_LOCAL =  override default value of 64 (local work group size)        
    Return:
        R, T, dT  = significance, position angle, position angle error
    """
    t0       = time.time()
    platform = cl.get_platforms()[0]
    if (GPU>0):
        device   = platform.get_devices(cl.device_type.GPU)
        LOCAL    = 64
        if (GPU_LOCAL>0): LOCAL = GPU_LOCAL        
        if (GPU>1): device = (device[1],)        
    else:
        device   = platform.get_devices(cl.device_type.CPU)
        LOCAL    =  8
    N, M      = F[0].data.shape
    context   = cl.Context(device)
    queue     = cl.CommandQueue(context)
    mf        = cl.mem_flags
    ### DW        = (2.0*NL+1)*STEP
    NL        = int( (DW/STEP-1.0)/2.0+0.5 )
    NL        = max([NL, 1])
    THRESHOLD = int(FRAC*(2*NL+1)*(2*NW+1)) ;
    BORDER    = int(1.1+np.sqrt(NL**2+NW**2.0)*STEP)  # border avoidance
    ##
    NDIR      = int(np.floor((3.14159/np.sqrt(2.0))*(DW-1.0)+1.0))    
    OPT       = " -D DK=%d -D N=%d -D M=%d" % (DK, N, M)
    OPT      += " -D NDIR=%d  -D THRESHOLD=%d -D BORDER=%d" % (NDIR, THRESHOLD, BORDER)
    OPT      += " -D NL=%d -D NW=%d -D STEP=%.4f" % (NL, NW, STEP)
    # summy arguments
    OPT      += " -D DW=0 -D NPIX=0"
    print(OPT)
    ## source    = file(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_mjRHT.c').read()
    source   = file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_RHT2.c').read()        
    program   = cl.Program(context, source).build(OPT)
    print('--- initialisation  %5.2f seconds:: DK %d DW %d' % (time.time()-t0, DK, DW))
    #
    S         = np.asarray(np.ravel(F[0].data), np.float32)
    SS        = np.empty_like(S)
    S_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf    = cl.Buffer(context, mf.READ_WRITE, S.nbytes)
    INT_buf   = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    THETA_buf = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    DTHETA_buf= cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    # Smoothing of the map
    SMOO     = program.Smooth
    SMOO.set_scalar_arg_dtypes([None, None])
    t0       = time.time()
    GLOBAL   = (int((N*M)/64)+1)*64
    SMOO(queue, [GLOBAL,], [LOCAL,], S_buf, SS_buf)
    cl.enqueue_copy(queue, SS, SS_buf)
    print('--- smoothing       %5.2f seconds' % (time.time()-t0))
    if (0):
        clf(),subplot(221),imshow(S.reshape(N,M)),colorbar()
        subplot(222),imshow(SS.reshape(N,M)),colorbar(),SHOW()
        sys.exit()
    # RHT
    t0       = time.time()
    RHT      = program.R_kernel_substep
    RHT.set_scalar_arg_dtypes([None, None, None, None])
    t0       = time.time()
    RHT(queue, [GLOBAL,], [LOCAL,], SS_buf, INT_buf, THETA_buf, DTHETA_buf)
    INT      = np.empty_like(S)
    THETA    = np.empty_like(S)
    DTHETA   = np.empty_like(S)
    cl.enqueue_copy(queue, INT,     INT_buf)
    cl.enqueue_copy(queue, THETA,   THETA_buf)
    cl.enqueue_copy(queue, DTHETA,  DTHETA_buf)
    print('--- RHT             %5.2f seconds' % (time.time()-t0))
    #
    return  INT.reshape(N,M), THETA.reshape(N,M), DTHETA.reshape(N,M)



def RollingHoughTransformExternalKernel(F, FT, DK, FRAC=0.8, GPU=0, GPU_LOCAL=-1):
    """
    Calculate (Rolling Hough) Transform using external, precalculated kernels.
    R, THETA, dTHETA = RollingHoughTransformS(F, DK, NL, NW, STEP, FRAC, GPU=0)
    Input:
        F         =  pyfits image to be analysed
        FT        =  template kernel, image dimensions [npix, npix, ntheta]
        DK        =  size of convolution kernel (smaller than (2*NL+1)*STEP; should be odd ?)
        FRAC      =  fraction of positive pixels used as the threshold
        GPU       =  if >0, use GPU instead of CPU
        GPU_LOCAL =  override default value of 64 (local work group size)                
    """
    npix, npix, ndir = FT[0].data.shape  # assume square images
    ###
    t0       = time.time()
    platform = cl.get_platforms()[0]
    if (GPU>0):
        device   = platform.get_devices(cl.device_type.GPU)
        LOCAL    = 64
        if (GPU_LOCAL>0): LOCAL = GPU_LOCAL                        
        if (GPU>1): device = (device[1],)        
    else:
        device   = platform.get_devices(cl.device_type.CPU)
        LOCAL    =  8
    N, M      = F[0].data.shape
    context   = cl.Context(device)
    queue     = cl.CommandQueue(context)
    mf        = cl.mem_flags
    THRESHOLD = FRAC * sum(np.ravel(FT[0].data[:,:,0]))      # fraction of potential maximum
    BORDER    = 1+int(np.sqrt(2.0)*0.5*npix)
    OPT       = " -D DK=%d -D N=%d -D M=%d" % (DK, N, M)
    OPT      += " -D NDIR=%d -D THRESHOLD=%.3ff -D BORDER=%d" % (ndir, THRESHOLD, BORDER)
    OPT      += " -D NPIX=%d" % npix
    # dummy arguments
    OPT      += " -D NL=0 -D NW=0 -D STEP=0 -D DW=0"
    print(OPT)
    # source    = file(HOMEDIR+'/starformation/Python/MJ/MJ/Aux/kernel_mjRHT.c').read()
    source   = file(HOMEDIR+'/starformation/Python/MJ/MJ/Pattern/kernel_RHT2.c').read()        
    program   = cl.Program(context, source).build(OPT)
    print('--- initialisation  %5.2f seconds:: DK %d DW %d' % (time.time()-t0, DK, DW))
    #
    S         = np.asarray(np.ravel(F[0].data), np.float32)
    KER       = np.asarray(np.ravel(FT[0].data), np.float32)  # theta runs fastest
    SS        = np.empty_like(S)
    S_buf     = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
    SS_buf    = cl.Buffer(context, mf.READ_WRITE, S.nbytes)
    INT_buf   = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    THETA_buf = cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    DTHETA_buf= cl.Buffer(context, mf.WRITE_ONLY, S.nbytes)
    KER_buf   = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=KER)
    # Smoothing of the map
    SMOO     = program.Smooth
    SMOO.set_scalar_arg_dtypes([None, None])
    t0       = time.time()
    GLOBAL   = (int((N*M)/64)+1)*64
    SMOO(queue, [GLOBAL,], [LOCAL,], S_buf, SS_buf)
    cl.enqueue_copy(queue, SS, SS_buf)
    print('--- smoothing       %5.2f seconds' % (time.time()-t0))
    if (0):
        clf(),subplot(221),imshow(S.reshape(N,M)),colorbar()
        subplot(222),imshow(SS.reshape(N,M)),colorbar(),SHOW()
        sys.exit()
    # RHT
    t0       = time.time()
    RHT      = program.R_kernel_template
    RHT.set_scalar_arg_dtypes([None, None, None, None, None])
    t0       = time.time()
    RHT(queue, [GLOBAL,], [LOCAL,], KER_buf, SS_buf, INT_buf, THETA_buf, DTHETA_buf)
    INT      = np.empty_like(S)
    THETA    = np.empty_like(S)
    DTHETA   = np.empty_like(S)
    cl.enqueue_copy(queue, INT,     INT_buf)
    cl.enqueue_copy(queue, THETA,   THETA_buf)
    cl.enqueue_copy(queue, DTHETA,  DTHETA_buf)
    print('--- RHT             %5.2f seconds' % (time.time()-t0))
    #
    return  INT.reshape(N,M), THETA.reshape(N,M), DTHETA.reshape(N,M)



