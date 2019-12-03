
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
   int   i0, j0, i, j, ipos, ipeak ;
   float w, dt, tstep;                // [-5,+5]*dt = [-57, +57] degrees
   i0   = X0 ;    j0 = Y0 ;           // initial coordinates
   t0   = P[i0+M*j0] ;                // initial direction
   if (id%2==0) t0 = fmod(t0+3.0f*PI, TWOPI) ; // must be [0,2*pi]
   float step =  STEP ;               // in pixels
   int bad = 0 ;
#define NTHETA 20
   dt = DTHETA/(float)NTHETA ;      // -DTHETA... +DTHETA, 2*NTHETA+1 steps
   for(ipos=0 ; ipos<MAXPOS-1; ipos++) {
      float weight=0.0f, sum=0.0f, peak=-999.0f ;
      for(int it=-NTHETA; it<=+NTHETA; it++) {  // test angles within a sector
         i       =  clamp((int)(i0 - step*sin(t0+it*dt)), 0, M) ;
         j       =  clamp((int)(j0 + step*cos(t0+it*dt)), 0, N) ;
         w       =  S[i+M*j]-BTM ;          // weight of the current data point
         t       =  P[i+M*j] ;              // this is known to be within [-PIHALF,+PIHALF]         
#if 0
         t      +=  round((t0-t)/PI)*PI ;   // convert t to angle within 90 degrees of t0
         if (fabs(t-t0)>DTHETA) w = 0.0f ;  // angle difference is too large, t vs t0
#endif
         if (w>0.0f) {
            weight +=       w*w ;
            sum    +=  it * w*w ;
            if (w>peak) {
               peak    =  w ;
               ipeak   = it ;
            }
         }
      }
      if (peak<=0.0f) {
         if ((fabs(i-85.0f)<10)&&(fabs(j-295.0f)<10)) printf("@@ BTM %d %d  %.3f\n", i, j, peak) ;
         break ;        // values dropped below BTM
      }
#if 0
      w    =  sum/weight ;     // average angle index
#else
      w    =  ipeak ;          // simply the index of the peak direction
#endif
      // register the new position
      if (MASK[i0+M*j0]<BTM) {
         bad +=1 ; 
         if (bad>OVERLAP) {
            if ((fabs(i-85.0f)<10)&&(fabs(j-295.0f)<10)) printf("@@ BAD %d %d\n", i, j) ;
            break ;
         }
         // try extrapolating --- follow old direction???
         tstep = t0 ;
      } else {
         bad    = 0 ;
         tstep  = t0+w*dt ;  // actual direction stepped towards
      }
      i0    -= step*sin(tstep) ;
      j0    += step*cos(tstep) ;
      if (id%2==0) {
         X[OFF+ipos]          = i0 ;  Y[OFF+ipos]          = j0 ;
      } else {
         X[OFF+MAXPOS-1-ipos] = i0 ;  Y[OFF+MAXPOS-1-ipos] = j0 ;
      }
      // check the PA of the new pixel
      t   =  P[i0+M*j0] ;            // must be rotated within 90 degrees of t0
      t  +=  round((t0-t)/PI)*PI ;
      // carry memory of the previous point
      t   =  0.25f*t0 + 0.75f*t ;
      if (fabs(t-t0)>0.5*PI) printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n") ;
      t0  =  fmod(t+4.0f*PI,TWOPI) ;         // start again with value in the range [0,2*pi]
#if 1
      // or use the direction of the previous step !!
      t0 +=  round((tstep-t0)/PI)*PI ;
      t0  =  fmod(t0, TWOPI) ;
#endif
   } // loop along the filament
   LEN[id] = ipos ;
}








__kernel void FollowSimple(__global float *S,    // significance
                           __global float *P,    // position angle
                           __global float *MASK, // mask of other filaments
                           const    int    X0,   // initial position -- column
                           const    int    Y0,   // initial position -- row
                           __global int   *X,    // positions along the filament
                           __global int   *Y,
                           __global int   *LEN) {
   // SIMPLE VERSION, USING SIG ONLY
   int id  = get_global_id(0) ;
   if (id>1) return ;   // ONE FILAMENT AT A TIME ??
   int OFF = MAXPOS*id ;
   float t0, t ;
   int   i0, j0, i, j, ipos, ipeak ;
   float w, dt, tstep;                // [-5,+5]*dt = [-57, +57] degrees
   //  X ~ i = column coordinate
   //  Y ~ j = row coordinate
   i0   = X0 ;    j0 = Y0 ;           // initial coordinates
   t0   = P[i0+M*j0] ;                // initial direction ACTUAL POSITION ANGLE
   if (id%2==0) t0 = fmod(t0+3.0f*PI, TWOPI) ; // must be [0,2*pi]
   float step =  STEP ;               // in pixels
   int bad = 0 ;
#define NTHETA 20
   dt = DTHETA/(float)NTHETA ;      // -DTHETA... +DTHETA, 2*NTHETA+1 steps
   for(ipos=0 ; ipos<MAXPOS-1; ipos++) {
      float weight=0.0f, sum=0.0f, peak=-999.0f ;
      for(int it=-NTHETA; it<=+NTHETA; it++) {  // test angles within a sector
         i       =  clamp((int)(i0 - step*sin(t0+it*dt)), 0, M) ;
         j       =  clamp((int)(j0 + step*cos(t0+it*dt)), 0, N) ;
         w       =  S[i+M*j]-BTM ;          // weight of the current data point
         if (w>0.0f) {
            weight +=       w*w ;
            sum    +=  it * w*w ;
            if (w>peak) {
               peak    =  w ;
               ipeak   = it ;
            }
         }
      }
#if 0
      w    =  sum/weight ;     // average angle index
#else
      w    =  ipeak ;          // simply the index of the peak direction
#endif
      if (peak<=0.0) break ;
      // register the new position
      if (MASK[i0+M*j0]<BTM) {
         bad   += 1 ; 
         tstep  = t0 ;
      } else {
         bad    = 0 ;
         tstep  = t0+w*dt ;  // actual direction stepped towards
      }
      i0    -= step*sin(tstep) ;
      j0    += step*cos(tstep) ;
      if ((i0<1)||(j0<1)||(i0>=M)||(j0>=N)) break ;
      if (id%2==0) {
         X[OFF+ipos]          = i0 ;  Y[OFF+ipos]          = j0 ;
      } else {
         X[OFF+MAXPOS-1-ipos] = i0 ;  Y[OFF+MAXPOS-1-ipos] = j0 ;
      }
      t0 = tstep ;
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

