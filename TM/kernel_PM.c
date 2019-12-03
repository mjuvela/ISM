
//  NDIR       = number of directions
//  FWHM       = FWHM of the low pass filter [pixels]
//  FWHM_HPF   = larger FWHM for low-pass filtering of the input image [pixels]
//  DIM0, DIM1 = pattern dimensions [DIM0,DIM1]
//  N          = image dimension, rows
//  M          = image dimension, columns (this index runs faster)
//  STEP       = step between template elements, in units of FWHM,
//               the default is 1.0
// indexing   [i,j]          = [i+M*j] 

#define PI      3.1415926536f
#define TWOPI   6.2831853072f
#define PIHALF  1.5707963268f
#define PYOPENCL_NO_CACHE 1



__kernel void Smooth(
                     __global float *S,      // input data [N]
                     __global float *SS      // output data
                    )  {
   //  Calculate the difference of images convolved with FWHM and FWHM_HPF.
   int  id  = get_global_id(0) ;  // one output pixel per work item
   if (id>=(N*M)) return ;
   // S[i,j], dimensions [N,M]  ->  [i,j] = [i+j*N]
   int i0   = id % M ;    // column   0 <= i < M
   int j0   = id / M ;    // row      0 <= j < N
   int RF   = (int)(2*FWHM_HPF+1) ;  // radius for filter calculation
   float K0 = 4.0f/log(2.0f)/(FWHM*FWHM) ;          // low pass filter
   float K1 = 4.0f/log(2.0f)/(FWHM_HPF*FWHM_HPF) ;  // high pass filter
   
   float sum0=0.0f, sum1=0.0f, wei0=0.0f, wei1=0.0, w, r2, y ;
   for(int i=max(0, i0-RF); i<min(i0+RF+1, M); i++) {          // row
      for(int j=max(0, j0-RF); j<min(j0+RF+1, N); j++) {       // column
         r2     =  (i-i0)*(i-i0) + (j-j0)*(j-j0) ;
         y      =   S[i+M*j] ;
         w      =   exp(-K0*r2) ;    // FWHM
         if (y==0.0f) w = 0.0f ;
         wei0  +=   w ;
         sum0  +=   w*y ;
         w      =   exp(-K1*r2) ;    // FWHM_HPF
         if (y==0.0f) w = 0.0f ;
         wei1  +=   w ;
         sum1  +=   w*y ;
      }
   }
   // Do not expand the image
   if ((S[id]!=0.0f)&&(fabs(sum0)>1.0e-30f)) {
      SS[id] = sum0/wei0 - sum1/wei1 ;
   } else {
      SS[id] = 0.0f ;
   }
}




__kernel void PatternMatch(
                           constant float *PAT,   // the pattern
                           __global float *SS,    // filtered input image
                           __global float *SIG,   // signal (significance)
                           __global float *PA     // best estimate of feature angle
                          )
{
   //  FWHM = FWHM [pixels] of the low-pass filter
   //  STEP = step between template elements in units of FWHM (default STEP=1.0)
   //  2016-10-28 -- avoid edges where template can step outside the image
   int    id = get_global_id(0) ;
   if    (id>=(N*M)) return ;
   int    i0 = id % M ;     // column  --   0 <= i < M
   int    j0 = id / M ;     // row
   int    ii, jj ;
   float  val0   =  -1.0e10f ;
   float  pa0    =   0.0f ;
   float  s, s2, y, pa, si, co ;
   float  D[DIM0*DIM1] ;    // data for the stencil [DIM0, DIM1]
   int    SKIP=0 ;          // skip this pixel if any of the data is missing
   float  centre0 = 0.5f*(DIM0-1.0f) ;  // vertical
   float  centre1 = 0.5f*(DIM1-1.0f) ;  // horizontal
   float  step    = STEP * FWHM ;       // actual step [pixels] between template elements
   int    bcount  = 0 ;                 // consecutive angles with best significance
   float  delta ;
   delta  = (sqrt( pow((DIM0+1.0f)/2.0f, 2.0f) + pow((DIM1+1.0f)/2.0f, 2.0f) ) + 1.0f) * step ; // pixels
   if (((i0-delta)<0)||((j0-delta)<0)||((i0+delta)>=M)||((j0+delta)>=N)) {
      SIG[id] = 0.0f ;   PA[id] = 0.0f ;  return ;
   }
     
#if SYMMETRIC
   delta =    PI/NDIR ;
#else
   delta = TWOPI/NDIR ;
#endif
   for(int ipa=0; ipa<NDIR; ipa++) {
#if (SYMMETRIC>0)
      pa  =  -PIHALF+(ipa+0.5f)*delta  ;  // position angle of the pattern -- [-90,+90] degrees
#else
      pa  =  (ipa+0.5f)*delta  ;  // position angle of the pattern -- [0, 360] degrees
#endif
      si  =  sin(pa) ;
      co  =  cos(pa) ;
      // up     =    j*co + i*si
      // right  =    i*co - j*si
      // template [DIM0, DIM1]  =>  indexing [i+j*DIM1]
      s   =  0.0f ;    s2 = 0.0f ;
      for(int i=0; i<DIM1; i++) {               // one dimension of the pattern
         for(int j=0; j<DIM0; j++) {            // the second dimension
            y           =  -co*(j-centre0)  + si*(i-centre1) ;  // up
            jj          =  (int)(j0+y*step) ;
            y           =  +si*(j-centre0)  + co*(i-centre1) ;  // right
            ii          =  (int)(i0+y*step) ;
            y           =  SS[ii+M*jj] ;
            D[i+DIM1*j] =  y   ;
            s          +=  y   ;
            s2         +=  y*y ;
            if (y==0.0f) SKIP=1 ; // every data point must be defined
         }
      }
      // normalise data
      y    =  s/(DIM0*DIM1) ;       // mean
#if (STUDENT>0)
      s2   =  sqrt(s2-y*y)/sqrt((float)(DIM0*DIM1)) ;  // standard deviation
#endif
      // sum of data minus pattern
      s    =  0.0f ;
      for(int i=0; i<DIM0*DIM1; i++) s  +=  PAT[i] * (D[i]-y) ;
#if (STUDENT>0)
# if (STUDENT==1)
      s  /= (DIM0*DIM1*s2) ;
# else
      s  /= DIM0*DIM1*sqrt(s2) ;
# endif
#endif
      
      if (s>val0) {  // best direction so far
         val0 = s ;    pa0 = pa ;    bcount = 1 ;
      } else {
         // we may have a dense angular grid so that consecutive angles could mean
         // indentical pixel values... could bias in the angles!
         if (s==val0) bcount += 1 ;   // count of best angle positions
      }
      
   }  // for itheta
   SIG[id] = val0 ;
   PA[id]  = pa0 + 0.5f*(bcount-1)*delta ;
   if (SKIP) {
      SIG[id] = PA[id] = 0.0f ;
   }
}




__kernel void PatternMatchAA(
                             constant float *PAT,   // the pattern
                             __global float *SS,    // filtered input image
                             __global float *SIG,   // signal (significance)
                             __global float *PA     // best estimate of feature angle
                            )
{
   // A variation where PA is weighted average of all angles (not just the best-fittinf angle).
   // if AAVE<=1, significance is the maximum significance value
   // if AAVE>=2, significance is also an angular-averaged quantity
   //   FWHM = FWHM [pixels] of the low-pass filter
   //   STEP = step between template elements in units of FWHM (default STEP=1.0)
   // Is AAVE=1 the best option?
   //  ... angle distributions not sensitive to number of histogram bins 
   //      (default PatternMatch does not work well with too many histogram bins)
   //  ... best-fitting-angle significance figure looks nicer than angle-averaged significance
   //      And why should significance value of a less well fitting angle corroborate
   //      the fit with a different position angle?
   int   id = get_global_id(0) ;
   if   (id>=(N*M)) return ;
   int   i0 = id % M ;     // column  --   0 <= i < M
   int   j0 = id / M ;     // row
   int   ii, jj ;
   float val0   =   0.0f ;
   float s, s2, y, pa, si, co ;
   float D[DIM0*DIM1] ;    // data for the stencil [DIM0, DIM1]
   int   SKIP=0 ;          // skip this pixel if any of the data is missing
   float centre0 = 0.5f*(DIM0-1.0f) ;  // vertical
   float centre1 = 0.5f*(DIM1-1.0f) ;  // horizontal
   float step    = STEP * FWHM ;   // actual step [pixels] between template elements
   float ssi=0.0, sco=0.0 ;
   for(int ipa=0; ipa<NDIR; ipa++) {
#if (SYMMETRIC>0)
      pa  =  -PIHALF+(ipa+0.5f)*PI/NDIR  ;  // position angle of the pattern -- [-90,+90] degrees
#else
      pa  =  (ipa+0.5f)*TWOPI/NDIR  ;  // position angle of the pattern -- [0, 360] degrees
#endif
      si  =  sin(pa) ;
      co  =  cos(pa) ;
      // up     =    j*co + i*si
      // right  =    i*co - j*si
      // template [DIM0, DIM1]  =>  indexing [i+j*DIM1]
      s   =  0.0f ;    s2 = 0.0f ;
      for(int i=0; i<DIM1; i++) {               // one dimension of the pattern
         for(int j=0; j<DIM0; j++) {            // the second dimension
            y           =  -co*(j-centre0)  + si*(i-centre1) ;  // up
            jj          =  (int)(j0+y*step) ;
            y           =  +si*(j-centre0)  + co*(i-centre1) ;  // right
            ii          =  (int)(i0+y*step) ;
            y           =  SS[clamp(ii+M*jj, 0, N*M-1)] ;
            D[i+DIM1*j] =  y   ;
            s          +=  y   ;
            s2         +=  y*y ;
            if (y==0.0f) SKIP=1 ; // every data point must be defined
         }
      }
      // normalise data
      y    =  s/(DIM0*DIM1) ;       // mean
#if (STUDENT>0)
      s2   =  sqrt(s2-y*y)/sqrt((float)(DIM0*DIM1)) ;  // standard deviation
#endif
      // sum of data minus pattern
      s    =  0.0f ;
      for(int i=0; i<DIM0*DIM1; i++) s  +=  PAT[i] * (D[i]-y) ;
#if (STUDENT>0)
# if (STUDENT==1)
      s  /= (DIM0*DIM1*s2) ;
# else
      s  /= DIM0*DIM1*sqrt(s2) ;
# endif
#endif
      
      if (AAVE<=1) {
         // As before, take s for the best-fitting single direction
         if (s>val0)  val0 = s ;
      } else {
         // Instead of the one best direction,  we calculate here the sum(s) and weighted direction
         if (s>0.0f)  val0  += s ;
      }
      ssi   += s * sin(2.0*pa) ;
      sco   += s * cos(2.0*pa) ;      
   }  // for itheta - loop over position angles
   if (AAVE<=1)  SIG[id]  =  val0 ;
   else          SIG[id]  =  val0/NDIR ;
   PA[id]   =  fmod(PI+0.5f*atan2(ssi,sco), PI);
   if (PA[id]>(0.5f*PI)) PA[id] = -PI+PA[id] ;
   if (SKIP) {
      SIG[id] = PA[id] = 0.0f ;
   }
}

