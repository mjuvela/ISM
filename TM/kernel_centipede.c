
//  NDIR     = number of directions
//  FWHM     = FWHM of the analysis, low pass filter
//  FWHM_HPF = large FWHM for low-pass filtering of input image
//  LEGS     = number of pairs of legs
//  Z    = threshold value
//  N    = rows
//  M    = columns, this runs faster

// indexing   [i,j]          = [i+M*j]
//            [itheta, i, j] = [ itheta + NDIR*(i+N*j) ]

#define PI      3.1415926536f
#define TWOPI   6.2831853072f
#define PIHALF  1.5707963268f
#define PYOPENCL_NO_CACHE 1



__kernel void Smooth(
                     __global float *S,      // input data [N]
                     __global float *SS      // output data
                    )  {
   // A = image smoothed with FWHM_HPF
   // B = image smoothed with FWHM
   // return B-A
   int  id  = get_global_id(0) ;  // one output pixel per work item
   if (id>=(N*M)) return ;
   // S[i,j], dimensions [N,M]  ->  [i,j] = [i+j*N]
   int i0   = id % M ;    // column   0 <= i < M
   int j0   = id / M ;    // row      0 <= j < N
   int RF   = (int)(2*FWHM_HPF) ;  // filter radius
   int RF2  = RF*RF ;
   float K0 = 4.0f/log(2.0f)/(FWHM*FWHM) ;          // low pass filter
   float K1 = 4.0f/log(2.0f)/(FWHM_HPF*FWHM_HPF) ;  // high pass filter
   float sum0=0.0f, sum1=0.0f, wei0=0.0f, wei1=0.0, w, r2, y ;
   for(int i=max(0, i0-RF); i<min(i0+RF+1, M); i++) {          // row
      for(int j=max(0, j0-RF); j<min(j0+RF+1, N); j++) {       // column
         r2 = (i-i0)*(i-i0)+(j-j0)*(j-j0) ;
         if (r2<(4.0f*RF2)) { // close enough to affect at least HPF
            y      =  S[i+M*j] ;
            w      =  exp(-K0*r2) ;    // FWHM
            if (y==0.0f) w = 0.0f ;
            wei0  +=  w ;
            sum0  +=  w*y ;
            w      =  exp(-K1*r2) ;    // FWHM_HPF
            if (y==0.0f) w = 0.0f ;
            wei1  +=  w ;
            sum1  +=  w*y ;
         }
      }
   }
   if (fabs(sum0)>1.0e-30f) {
      SS[id] = sum0/wei0 - sum1/wei1 ;
   } else {
      SS[id] = 0.0f ;
   }
   if (S[id]==0.0f) SS[id] = 0.0f ;
}




__kernel void Centipede(
                        __global float *SS,    // filtered input image
                        __global float *SIG,   // signal (significance)
                        __global float *PHI    // best estimate of feature angle
                       )
{
   // This kernel does not return RR = R for each direction separately
   // If one wants to test agreement to given direction (B field), one
   // could add that as an argument and use the loop below to calculate
   // the probability of the coincidence
   int   id = get_global_id(0) ;
   if (id>=(N*M)) return ;
   int   i0 = id % M ;     // column  --   0 <= i < M
   int   j0 = id / M ;     // row
   int   ii, jj ;
   float border = 1.0f + sqrt(pow((float)(0.5f*LEGS*FWHM),2.0f)+(float)(FWHM*FWHM)) ;
   if ((i0<border)||(j0<border)||(i0>=(M-1.0f-border))||(j0>=(N-1.0f-border))) {  // too close to image borders
      SIG[id] = PHI[id] = 0.0f ;
      return ;
   }
   float val0   =  -1.0e10f ;
   float phi0   =   0.0f ;
   float s, s2, y, phi, si, co, centre ;
   float D[3*LEGS] ;
   int   SKIP=0 ;
   for(int iphi=0; iphi<NDIR; iphi++) {
      
      phi = -PIHALF+(iphi+0.5f)*PI/NDIR ;
      si  =  sin(phi) ;
      co  =  cos(phi) ;
      s   =  0.0f ;    s2 = 0.0f ;
      for(int i=0; i<3; i++) {           // left-centre-right
         for(int j=0; j<LEGS; j++) {     // along the length of the centipede
            y        =  -si*(j-0.5f*(LEGS-1.0f)) - co*(i-1.0f) ;
            ii       =  (int)(i0+y*FWHM) ;     // legs (and left/centre/right) are FWHM apart
            y        =  +co*(j-0.5f*(LEGS-1.0f)) - si*(i-1.0f) ;
            jj       =  (int)(j0+y*FWHM) ;
            y        =  SS[ii+M*jj] ;
#if 0
            if ((ii<0)||(jj<0)||(ii>=M)||(jj>=N)) printf("?????  %3d %3d\n", ii, jj) ;
#endif
            D[i+3*j] =  y ;
            s       +=  y ;
            s2      +=  y*y ;
            if (y==0) SKIP=1 ; // every data point must be defined
         }
      }
      // normalise data
      y    = s/(3.0f*LEGS) ;               // mean
#if (STUDENT==1)
      s2   = sqrt((s2-y*y)/(3.0f*LEGS)) ;  // standard deviation
#endif
      // sum of D*S
      s    =  0.0f ;
      for(int j=0; j<LEGS; j++) {     // along the length of the centipede
         s  +=  -0.333333f * (D[0+3*j]-y) ;  // left
         s  +=  +0.666666f * (D[1+3*j]-y) ;  // middle
         s  +=  -0.333333f * (D[2+3*j]-y) ;  // right
      }      
#if (STUDENT==1)
      s  /= (LEGS*s2) ;
#endif
      if (s>val0) {
         val0 = s ;   phi0 = phi ;
      }
   }  // for itheta
   SIG[id] = val0 ;
   PHI[id] = phi0 ;
   if (SKIP) {
      SIG[id] = PHI[id] = 0.0f ;
   }
}

