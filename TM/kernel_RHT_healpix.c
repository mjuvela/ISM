//  NDIR = number of directions
//  DK   = diameter of smoothing kernel [radians]
//  DW   = diameter of circular regions [radians]
//  Z    = threshold value
//  STEP = step for sampling the beam

#define PI        3.1415926536f
#define TWOPI     6.2831853072f
#define TWOTHIRD  0.6666666667f
#define PIHALF    1.5707963268f


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



__kernel void Smooth(__global float *S,       // input data [N]
                     __global float *SS) {    // output data
   // Naive convolution, ok for small beams only
   int  id  =  get_global_id(0) ;     // one output pixel per work item
   if (id>=NPIX) return ;
   int R    =  (int)(0.5f*DK/STEP) ;  // radius in steps
   int R2   =  R*R ;
   float theta0, phi0, theta, phi, siny ;
   Pixel2AnglesRing(id, &phi0, &theta0) ;
   siny     =  sin(theta0) ;
   float sum=0.0f, wsum=0.0f, y ;
   // sample the beam at intervals of STEP
   for(int i=-R; i<=R; i++) {
      theta = clamp(theta0+i*STEP, 0.0f, PI) ;
      for(int j=-R; j<=R; j++) {
         if ((i*i+j*j)<=R2) {
            phi   =  phi0+j*STEP/siny ;
            y     =  S[Angles2PixelRing(phi, theta)] ;
            if (y>-99.0f) {
               wsum += 1.0f ;
               sum  += y ;
            }
         }
      }
   }
   if (fabs(sum)>1.0e-30f) {
      if ((S[id]-sum/wsum)>0.0) SS[id] = 1.0f ;
      else                      SS[id] = 0.0f ;
   } else {
      SS[id] = 0.0f ;
   }
}




__kernel void R_kernel(__global float *SS,  // thresholded image
                       __global float *RR,  // significance
                       __global float *PA)  // position angles
{
   int id    =  get_global_id(0) ;
   if (id>=NPIX) return ;
   int   RW  =  (int)(0.5f*DW/STEP) ;  // step is STEP and not PIX ...
   float si=0.0f, co=0.0f ;
   int   sum=0, val ;
   int   npix=(int)(Z*2.0f*RW) ;       // how many ON pixels there should be in given direction
   float theta0, phi0, theta, phi, siny, y, pa, dx, dy ;
   Pixel2AnglesRing(id, &phi0, &theta0) ;
   siny      =  sin(theta0) ;
   for(int k=0; k<NDIR; k++) {
      pa    =  k*6.2831853f/NDIR ;
      dx    = -sin(pa)*STEP/siny ;    // east from north
      dy    = +cos(pa)*STEP ;
      val   =  0 ;
      for(int l=-RW; l<=+RW; l++) {         // loop along the line
         theta =  clamp(theta0 + l*dy, 0.0f, PI) ;
         phi   =        phi0   + l*dx ;
         y     =  SS[Angles2PixelRing(phi, theta)] ;
         if (y>0.0f) val++ ;
      }
      if (val<npix) val = 0.0f ;            // thresholding
      // calculate SIN and COS averages
      si  += val * sin(2.0f*pa) ;
      co  += val * cos(2.0f*pa) ;
      sum += val ;
   }
   RR[id] = sum ;  // significance
   y = fmod(PI+0.5f*atan2(si,co), PI) ;
   if (y>0.5f*PI) y -= PI ;
   PA[id] = y ;
}


