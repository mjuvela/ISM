
#define PI      3.1415926536f
#define TWOPI   6.2831853072f
#define PIHALF  1.5707963268f
#define PYOPENCL_NO_CACHE 1


#if 0
__kernel void Saddle(__global float *S,   // significance
                     __global float *P,   // position angle
                     __global int   *X,   // x-coordinate of tested pixels
                     __global int   *Y,   // y-coordinate = row of tested pixels
                     __global int   *SAD) // 1 for saddle points
{
   // For each candidate position (X,Y), return SAD==1 if it is a saddle point
   // more precisely, the centre of the perpendicular profile
   int  id = get_global_id(0) ;
   if  (id>=NSP) return ;
   int   i0=X[id], j0=Y[id], ind, k ;
   float p0=P[i0+M*j0] ;
   // calculate signal-weighted average in a direction perpendicular to p0
   float weight=0.0f, w, sum=0.0f, dp ;
   for(k=-FWHM; k<=FWHM; k++) {  // k runs over orthogobal direction
      ind     =  (int)(i0-k*cos(p0)) + M*(int)(j0-k*sin(p0)) ;
      w       =  clamp(S[ind]-BTM, 0.0f, 999.0f) ;
      // dp      =  fabs(P[ind]-p0) ;
      // if ((dp>0.6)&&(dp<(PI-0.6f))) w = 0.0f ;
      weight +=    w ;
      sum    +=  k*w ;
   }
   w = sum/weight ;
   SAD[id] = 0 ;
   if (fabs(w)<0.5f) SAD[id] = 1 ;
}
#endif




__kernel void Follow(__global float *S,    // significance
                     __global float *P,    // position angle
                     __global float *MASK, // mask of other filaments
                     const    int    X0,   // initial position
                     const    int    Y0,   // initial position
                     __global int   *X,    // positions along the filament
                     __global int   *Y,
                     __global int   *LEN) {
   // Start positions X0, Y0
   // global id  [2*NPOS], odd and even go upstream and downstream, respectively
   // storage is X[id*MAXPOS]
   int id  = get_global_id(0) ;
   if (id>1) return ;   // ONE FILAMENT AT A TIME ??
   int OFF = MAXPOS*id ;
   float t0, t ;
   int   i0, j0, i, j, ipos ;
   float w, dt = 0.2f ;               // [-5,+5]*dt = [-57, +57] degrees
   i0   = X0 ;    j0 = Y0 ;           // initial coordinates
   t0   = P[i0+M*j0] ;                // initial direction
   if (id%2==0) t0 = fmod(t0+3.0f*PI, TWOPI) ; // must be [0,2*pi]
   float step =  0.5f*FWHM ;          // in pixels
   int bad = 0 ;
   for(ipos=0 ; ipos<MAXPOS-1; ipos++) {
      // scan a sector [-delta,+delta] radians, distance FWHM
      float weight=0.0f, sum=0.0f, peak=0.0f ;
      for(int it=0; it<11; it++) {
         i       =  clamp((int)(i0 - step*sin(t0+(it-5)*dt)), 0, M) ;
         j       =  clamp((int)(j0 + step*cos(t0+(it-5)*dt)), 0, N) ;
         w       =  S[i+M*j] ;
         t       =  P[i+M*j] ;                 // this is known to be within [-PIHALF,+PIHALF]         
         t      +=  round((t0-t)/PI)*PI ;      // convert t to angle within 90 degrees of t0
         if (fabs(t-t0)>1.0f) w = 0.0f ;       // angle difference is too large
         weight +=  w ;
         sum    +=  w*it ;
         peak    =  max(peak, w) ;
      }
      if (weight<1.0e-5f) break ;  // no more valid positions
      if (peak<BTM) break ;        // values dropped below BTM
      w    =  sum/weight ;         // average angle index
      // register the new position
      if (MASK[i0+M*j0]<BTM) {
         bad +=1 ; 
         if (bad>OVERLAP) break ;
         // try extrapolating --- follow old direction???
         i0  -=  step*sin(t0) ;
         j0  +=  step*cos(t0) ;
      } else {
         bad = 0 ;
         i0  -=  step*sin(t0+(w-5)*dt) ;
         j0  +=  step*cos(t0+(w-5)*dt) ;         
      }
      if (id%2==0) {
         X[OFF+ipos]          = i0 ;  Y[OFF+ipos]          = j0 ;
      } else {
         X[OFF+MAXPOS-1-ipos] = i0 ;  Y[OFF+MAXPOS-1-ipos] = j0 ;
      }
      t   =  P[i0+M*j0] ;           // must be rotated within 90 degrees of t0
      t  +=  round((t0-t)/PI)*PI ;
      t0  = fmod(t,TWOPI) ;         // start again with value in the range [0,2*pi]
   } // loop along the filament
   LEN[id] = ipos ;
}



__kernel void Mask(__global float *MASK,
                   __global int   *X,
                   __global int   *Y,
                   __global int   *LEN,
                   const    int   X0,
                   const    int   Y0) {
   // two work items have calculated positions into (X, Y)
   int  id = get_global_id(0) ;
   int  i, j, i0, j0 ;
   if (id==0) {
      i0 = X0 ; j0 = Y0 ;
   } else {
      i  = id-1 ;
      if (i%2==0) {
         i = i/2 ;                  // index of the position in X[0:LEN[0]]
         if (i>LEN[0])  return ;
      } else {
         i = (i-1)/2 ;              // index to X[MAXPOS:2*MAXPOS]
         if (i>LEN[1]) return ;
         i = 2*MAXPOS-1-i ;
      }
      i0 = X[i] ;  j0 = Y[i] ;
   }
   float R = REPEL*FWHM ;
   for(i=i0-R; i<i0+R+1; i++) {
      if ((i<0)||(i>=M)) continue ;
      for(j=j0-R; j<j0+R+1; j++) {
         if ((j<0)||(j>=N)) continue ;   
         float r2 = (i-i0)*(i-i0) + (j-j0)*(j-j0) ;
         if (r2<(R*R)) {
            MASK[i+M*j] = -1.0f ;
         }
      }
   }
}

