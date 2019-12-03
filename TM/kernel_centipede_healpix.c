
#define PI        3.1415926536f
#define TWOPI     6.2831853072f
#define TWOTHIRD  0.6666666667f
#define PIHALF    1.5707963268f



__kernel void Threshold(const float threshold, __global float *S, __global float *SS) {
   int id = get_global_id(0) ;
   if (id>=(12*NSIDE*NSIDE)) return ;
   if (S[id]>threshold) SS[id] = 1.0 ;
   else                 SS[id] = 0.0 ;   
}



int Angles2PixelRing(float phi, float theta) {
   // Convert angles to pixel index in ring order
   int     nl2, nl4, ncap, npix, jp, jm, ipix1 ;
   float   z, za, tt, tp, tmp ;
   int     ir, ip, kshift ;
   if ((theta<0.0)||(theta>PI)) return -1 ;
   z   = cos(theta) ;
   za  = fabs(z) ;
   if (phi>=TWOPI)  phi -= TWOPI ;
   if (phi<0.0f)    phi += TWOPI ;
   tt  = phi / PIHALF ;    // in [0,4)
   nl2 = 2*NSIDE ;
   nl4 = 4*NSIDE ;
   ncap  = nl2*(NSIDE-1) ;   //  number of pixels in the north polar cap
   npix  = 12*NSIDE*NSIDE ;
   if (za<=TWOTHIRD) {  //  Equatorial region ------------------
      jp = (int)(NSIDE*(0.5f+tt-z*0.75f)) ;   // ! index of  ascending edge line 
      jm = (int)(NSIDE*(0.5f+tt+z*0.75f)) ;   // ! index of descending edge line
      ir = NSIDE + 1 + jp - jm ;   // ! in {1,2n+1} (ring number counted from z=2/3)
      kshift = 0 ;
      if (ir%2==0) kshift = 1 ;    // ! kshift=1 if ir even, 0 otherwise
      ip = (int)( ( jp+jm - NSIDE + kshift + 1 ) / 2 ) + 1 ;   // ! in {1,4n}
      if (ip>nl4) ip -=  nl4 ;
      ipix1 = ncap + nl4*(ir-1) + ip ;
   } else {  // ! North & South polar caps -----------------------------
      tp  = tt - (int)(tt)  ;    // !MOD(tt,1.d0)
      tmp = sqrt( 3.0f*(1.0f - za) ) ;
      jp = (int)(NSIDE*tp         *tmp ) ;   // ! increasing edge line index
      jm = (int)(NSIDE*(1.0f - tp)*tmp ) ;   // ! decreasing edge line index
      ir = jp + jm + 1  ;          // ! ring number counted from the closest pole
      ip = (int)( tt * ir ) + 1 ;    // ! in {1,4*ir}
      if (ip>(4*ir)) ip -= 4*ir ;
      ipix1 = 2*ir*(ir-1) + ip ;
      if (z<=0.0f) {
         ipix1 = npix - 2*ir*(ir+1) + ip ;
      }
   }
   return  ( ipix1 - 1 ) ;    // ! in {0, npix-1}
}



bool Pixel2AnglesRing(const int ipix, float *phi, float *theta) {
   int    nl2, nl4, npix, ncap, iring, iphi, ip, ipix1 ;
   float  fact1, fact2, fodd, hip, fihip ;
   npix = 12*NSIDE*NSIDE;      // ! total number of points
   // if ((ipix<0)||(ipix>(npix-1))) return false ;
   ipix1 = ipix + 1 ;    // ! in {1, npix}
   nl2   = 2*NSIDE ;
   nl4   = 4*NSIDE ;
   ncap  = 2*NSIDE*(NSIDE-1) ;  // ! points in each polar cap, =0 for NSIDE =1
   fact1 = 1.5f*NSIDE ;
   fact2 = 3.0f*NSIDE*NSIDE  ;
   if (ipix1<=ncap) {   // ! North Polar cap -------------
      hip   = ipix1/2.0f ;
      fihip = (int)( hip ) ;
      iring = (int)( sqrt( hip - sqrt(fihip) ) ) + 1 ;  // ! counted from North pole
      iphi  = ipix1 - 2*iring*(iring - 1) ;
      *theta= acos( 1.0f - iring*iring / fact2 ) ;
      *phi  = (iphi - 0.5f) * PI/(2.0f*iring) ;
   } else {
      if (ipix1<=nl2*(5*NSIDE+1)) { // ! Equatorial region ------
         ip    = ipix1 - ncap - 1 ;
         iring = (int)( ip / nl4 ) + NSIDE ;   // ! counted from North pole
         iphi  = (ip%nl4) + 1 ;
         fodd  = 0.5E0 * (1 + (iring+NSIDE)%2) ;  // ! 1 if iring+NSIDE is odd, 1/2 otherwise
         *theta= acos( (nl2 - iring) / fact1 ) ;
         *phi  = (iphi - fodd) * PI /(2.0f*NSIDE) ;
      } else {   // ! South Polar cap -----------------------------------
         ip    = npix - ipix1 + 1 ;
         hip   = ip/2.0f ;
         fihip = (int) ( hip ) ;
         iring = (int)( sqrt( hip - sqrt(fihip) ) ) + 1 ;   // ! counted from South pole
         iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1)) ;
         *theta= acos( -1.0f + iring*iring / fact2 ) ;
         *phi  = (iphi - 0.5f) * PI/(2.0f*iring) ;
      }
   }
   return true ;
}




__kernel void Smooth(
                     const    int    NPIX,   // to be calculated
                     __global int   *IPIX,   // calculate for these pixels
                     const    int    NLOC,   // number of local area pixels used in convolution
                     __global int   *ILOC,   // list of local pixels
                     __global float *S,      // input data, allsky with NSIDE
                     __global float *SS      // output data, allsky with NSIDE
                    )  {
   // Assume that ILOC is short enough so that sums can be always calculated over all
   // the ILOC pixels (without too much slow-down)
   //    A = image smoothed with FWHM_HPF
   //    B = image smoothed with FWHM
   //    return B-A
   // One work item per sky pixel
   //   :: call routine for each NSIDE=32 area = 4096 pixels + nearby pixels = ILOC
   //                                            12288 calls
   int  id    = get_global_id(0) ;  // one output pixel per work item
   if (id>=NPIX) return ;
   int  ipix  = IPIX[id] ;                            // current pixel id
   float K0   = 4.0f/log(2.0f)/(FWHM*FWHM) ;          // low pass filter
   float K1   = 4.0f/log(2.0f)/(FWHM_HPF*FWHM_HPF) ;  // high pass filter
   // current pixel coordinates
   float y, theta0, phi0 ;
   Pixel2AnglesRing(ipix, &phi0, &theta0) ;
   float sum0=0.0f, sum1=0.0f, wei0=0.0f, wei1=0.0, w, r2, theta, phi ;
   for(int i=0; i<NLOC; i++)  { // loop over ***ALL*** local pixels
      Pixel2AnglesRing(ILOC[i], &phi, &theta) ;  // coordinates of that nearby pixel
      r2    =  pow(theta0-theta,2.0f) + pow(sin(theta0)*(phi-phi0),2.0f) ; // radians
      y     =  S[ILOC[i]] ;
      w     =  exp(-K0*r2) ;
      wei0 +=  w ;
      sum0 +=  w*y ;
      w     =  exp(-K1*r2) ;
      wei1 +=  w ;
      sum1 +=  w*y ;
      // if (id==0) printf("%6d %8.5f %8.5f  %4d %6d  %8.5f %8.5f\n", ipix, phi0, theta0, i, ILOC[i], phi, theta) ;
   }
   // if (id==0) printf("%.3e %.3e   %.3e %.3e\n", sum0, wei0, sum1, wei1) ;
   if (fabs(sum0)>1.0e-30f) {
      SS[ipix] = sum0/wei0 - sum1/wei1 ;
      // SS[ipix] = sum0/wei0  ;
   } else {
      SS[ipix] = 0.0f ;
   }
   // printf("   %8.3f\n", SS[ipix]) ;
}






__kernel void Smooth2(
                      __global float *S,      // input data, allsky with NSIDE
                      __global float *SS      // output data, allsky with NSIDE
                     )  {
   int  id    = get_global_id(0) ;                    // one output pixel per work item
   float K0   = 4.0f/log(2.0f)/(FWHM*FWHM) ;          // low pass filter
   float K1   = 4.0f/log(2.0f)/(FWHM_HPF*FWHM_HPF) ;  // high pass filter
   float y, theta0, phi0 ;
   Pixel2AnglesRing(id, &phi0, &theta0) ;
   float siny = sin(theta0) ;
   float sum0 = 0.0f, sum1=0.0f, wei0=0.0f, wei1=0.0, w, r2, theta, phi ;
   int   NX   = 21             ;      // this many samples over FWHM_HPF in each direction
   float STEP = 2.0*FWHM_HPF/NX ;     // radians
   int   index ;
   for(int i=-NX; i<=+NX; i++)  {     // loop over phi
      phi = fmod(phi0 + i*STEP/siny+TWOPI,TWOPI) ;
      for(int j=-NX; j<=+NX; j++) {   // loop over theta
         theta    =  clamp(theta0+j*STEP, 0.0f, PI) ;
         r2       =  (i*i+j*j)*STEP*STEP ;  // ***approximation***
         y        =  S[Angles2PixelRing(phi, theta)] ;
#if 0
         index = Angles2PixelRing(phi, theta) ;
         if ((index<0)|| (index>=(12*NSIDE*NSIDE))) {
            printf("phi %8.5f  theta %8.5f   INDEX %9d >= %d\n", phi, theta, index, NSIDE) ;
         }
         y        =  S[clamp(Angles2PixelRing(phi, theta),0,12*NSIDE*NSIDE-1)] ;
#endif
         if (y>-99.0f) {
            w     =  exp(-K0*r2) ;
            wei0 +=  w ;
            sum0 +=  w*y ;
            w     =  exp(-K1*r2) ;
            wei1 +=  w ;
            sum1 +=  w*y ;
         }
      }
   }
   if (fabs(sum0)>1.0e-30f) {
      SS[id] = sum0/wei0 - sum1/wei1 ;
   } else {
      SS[id] = 0.0f ;
   }
}





__kernel void Centipede(
                        __global float *SS,    // filtered input image
                        __global float *SIG,   // signal (significance)
                        __global float *PA     // best estimate of feature angle
                       )
{
   int id = get_global_id(0) ;
   int   SKIP=0 ;
   float val0   =  -1.0e10f ;
   float pa0    =   0.0f ;
   float s, s2, y, si, co, pa ;
   float D[3*LEGS] ;
   float theta0, phi0, theta, phi, siny ;
   
   Pixel2AnglesRing(id, &phi0, &theta0) ;
   siny = sin(theta0) ;   // for coordinate scaling
   
   for(int iPA=0; iPA<NDIR; iPA++) {
      
      pa  = -1.570796326f + iPA*3.141592654f/NDIR ;
      si  =  sin(pa) ;
      co  =  cos(pa) ;
      s   =  0.0f ;    s2 = 0.0f ;
      for(int i=0; i<3; i++) {           // left-centre-right
         for(int j=0; j<LEGS; j++) {     // along the length of the centipede
            y        =  -si*(j-0.5f*(LEGS-1.0f)) - co*(i-1.0f)  ;
            phi      =   phi0 + y*FWHM/siny ;
            phi      =   fmod(phi+TWOPI,TWOPI) ;
            y        =  +co*(j-0.5f*(LEGS-1.0f)) - si*(i-1.0f)  ;
            theta    =   theta0 + y*FWHM ;
            theta    =   clamp(theta, 0.0f, PI) ;
            y        =   SS[Angles2PixelRing(phi, theta)] ;
            D[i+3*j] =   y  ;
            s       +=   y ;
            s2      +=   y*y ;
            if (y==0) SKIP=1 ; // every data point must be defined
         }
      }
      // normalise data
      y    =  s/(3.0f*LEGS) ;                 // mean
#ifdef STUDENT
      s2   = sqrt((s2-y*y)/(3.0f*LEGS)) ;     // standard deviation
#endif
      // sum of D*S
      s    =  0.0f ;
      for(int j=0; j<LEGS; j++) {             // along the length of the centipede
         s  +=  -0.3333f * (D[0+3*j]-y) ;     // left
         s  +=  +0.6666f * (D[1+3*j]-y) ;     // middle
         s  +=  -0.3333f * (D[2+3*j]-y) ;     // right
      }
#ifdef STUDENT
      s /= s2 ;
#endif
      if (s>val0) {
         val0 = s ;   pa0 = pa ;
      }
   }  // for itheta
   
   SIG[id] = val0 ;
   PA[id]  = pa0 ;
   if (SKIP) {
      SIG[id] = PA[id] = 0.0f ;
   }
   
}



