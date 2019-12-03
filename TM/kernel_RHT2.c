


__kernel void Smooth(
                     __global float *S,      // input data [N]
                     __global float *SS      // output data
                    )  {
   
   int  id  = get_global_id(0) ;  // one output pixel per work item
   if (id>=(N*M)) return ;
   // S[i,j], dimensions [N,M]  ->  [i,j] = [i+j*N]
   int i0 = id % M ;    // column   0 <= i < M
   int j0 = id / M ;    // row      0 <= j < N
   int RK = (DK-1)/2 ;  // DK should be an odd number !
   int R2 = RK*RK ;
   float K = 4.0f/log(2.0f)/(DK*DK) ;
   float sum=0.0f, wsum=0.0f, r2 ;
   for(int i=max(0, i0-RK); i<min(i0+RK+1, M); i++) {          // row
      for(int j=max(0, j0-RK); j<min(j0+RK+1, N); j++) {       // column
         r2 = (i-i0)*(i-i0)+(j-j0)*(j-j0) ;
#ifdef GAUSS  // Gaussian convolution
         if (r2<(4.0f*R2)) {
            float w  = exp(-K*r2) ;
            if (S[i+M*j]==0.0f) w = 0.0f ;
            wsum += w ;
            sum  += w*S[i+M*j] ;
         }
#else // top hat (the original way?)
         if (r2<=R2) {
            if (S[i+M*j]>0.0) {
               wsum += 1.0f ;
               sum  += S[i+M*j] ;
            }
         }
#endif
      }
   }
   if (fabs(sum)>1.0e-30f) {
      if ((S[id]-sum/wsum)>0.0) SS[id] = 1.0f ;
      else                      SS[id] = 0.0f ;
   } else {
      SS[id] = 0.0f ;
   }
}





__kernel void R_kernel_oneline(
                               __global float *SS,        // thresholded image
                               __global float *INT,       // output R[i,j,i_theta]
                               __global float *THETA,
                               __global float *DTHETA
                              )
{
   // Basic version -- one steps along a single line, in steps of 1.0 pixels == no oversampling
   //   This kernel does not return R for each direction separately
   //   If one wants to test agreement to given direction (B field), one
   //   could add that as an argument and use the loop below to calculate
   //   the probability of the coincidence
   int id = get_global_id(0) ;
   if (id>=(N*M)) return ;
   int i0 = id % M ;      // column  --   0 <= i < M
   int j0 = id / M ;      // row
   int RW  = (DW-1)/2 ;   // DW should be an odd number
   if ((i0<(1+RW))||(i0>=(M-1-RW))||(j0<(1+RW))||(j0>(N-1-RW))) {
      INT[id] = 0 ; THETA[id] = 0.0f  ;  DTHETA[id] = 0.0f ;  return ;
   }
   // printf("RW %3d   (i,j) = (%3d,%3d)  NxM = %3d x %3d\n", RW, i0, j0, N, M) ;
   int   val, sum=0 ;
   float Q=0.0f, U=0.0f, Qsq=0.0f, Usq=0.0f, si, co, theta ;
   for(int k=0; k<NDIR; k++) {
      theta =  k*6.2831853f/NDIR ;
      si = sin(theta) ;  co = cos(theta) ;   // first use of si and co
      val = 0 ;
      for(int l=-RW; l<=+RW; l++) {  // loop along the line
         int i = (int)floor(i0+0.5f-l*si) ;  // reference is the centre of the pixel!
         int j = (int)floor(j0+0.5f+l*co) ;
         if (SS[i+M*j]>0.0f) val++ ;
      }
      // HTHETS[180*id+k] = val ; --- if one needs R for each direction separately
      val      = (val<THRESHOLD) ? 0 : (val-THRESHOLD) ;
      // QRNHT = sum over angles of val2 * cos(2*theta)
      co    = cos(2.0f*theta) ;  // re-use si and co
      si    = sin(2.0f*theta) ; 
      Q    += val*co ;
      U    += val*si ;
      Qsq  += val*co*co ;
      Usq  += val*si*si ;
      sum  += val ;
   }
   // if (id==0) printf("sum %d\n", sum) ;
   INT[id]      = sum ;                    // straight sum over angles
   THETA[id]    = 0.5f*atan2(U, Q) ;       // [radians]
   si   =  Q*Q ;           // re-use si and co
   co   =  U*U ;
   si   =  sqrt((si*Usq+co*Qsq) / (4.0f*(si+co)*(si+co))) ; // [radians]
   if (si>0.01f)  DTHETA[id] = clamp(si, 0.01f, 3.14159f)   ;
   else           DTHETA[id] = 0.0f ;
}




__kernel void R_kernel_substep(
                               __global float *SS,        // thresholded image
                               __global float *INT,       // output R[i,j]
                               __global float *THETA,     // position angle [i,j]
                               __global float *DTHETA     // position angle error [i,j]
                              )
{
   // Version with more than one line of points, option for substepping
   //      STEP    =  step in units of a pixel
   //      NL, NW  =  number of steps along the bar and in the perpendicular direction
   //                 **in each direction**
   //      BORDER  =  border avoidance [pixels]
   // Note: THRESHOLD should be a correspondingly larger number
   int id = get_global_id(0) ;
   if (id>=(N*M)) return ;
   int i0 = id % M ;      // column  --   0 <= i < M
   int j0 = id / M ;      // row
   if ((i0<BORDER)||(i0>=(M-BORDER))||(j0<BORDER)||(j0>=(N-BORDER))) {
      INT[id] = 0 ; THETA[id] = 0.0f  ;  DTHETA[id] = 0.0f ;  return ;
   }
   int   val, sum=0 ;
   float Q=0.0f, U=0.0f, Qsq=0.0f, Usq=0.0f, si, co, theta ;
   for(int k=0; k<NDIR; k++) {
      theta =  k*6.2831853f/NDIR ;
      si = STEP*sin(theta) ;  co = STEP*cos(theta) ;   // first use of si and co
      val = 0 ;
      for(int l=-NL; l<=+NL; l++) {          // along the line
         for(int w=-NW; w<=NW; w++) {        // steps in the perpendicular direction
            int i = (int)floor((i0+0.5f)+w*co-l*si) ; // reference is the centre of the pixel!
            int j = (int)floor((j0+0.5f)+w*si+l*co) ;
            if (SS[i+M*j]>0.0f) val++ ;
         }
      }
      val      = (val<THRESHOLD) ? 0 : (val-THRESHOLD) ;
      co    = cos(2.0f*theta) ;  // re-use si and co
      si    = sin(2.0f*theta) ; 
      Q    += val*co ;
      U    += val*si ;
      Qsq  += val*co*co ;
      Usq  += val*si*si ;
      sum  += val ;
   }
   // if (id==0) printf("sum %d\n", sum) ;
   INT[id]      = sum ;                    // straight sum over angles
   THETA[id]    = 0.5f*atan2(U, Q) ;       // [radians]
   si   =  Q*Q ;           // re-use si and co
   co   =  U*U ;
   theta  =  sqrt((si*Usq+co*Qsq) / (4.0f*(si+co)*(si+co))) ; // [radians]
   if (theta>0.01f)  DTHETA[id] = clamp(theta, 0.01f, 3.14159f)   ;
   else              DTHETA[id] = 0.0f ;
#if 0
   if (theta>90.0) {
      printf("theta %10.3e +- %10.3e  si %10.3e co %10.3e Q %10.3e U %10.3e\n", 
             THETA[id],       theta,  si,      co,       Q,       U) ;
   }
#endif
   
}




__kernel void R_kernel_template(
                                __global float *KER,       // kernels
                                __global float *SS,        // thresholded image
                                __global float *INT,       // output R[i,j]
                                __global float *THETA,     // position angles [i,j]
                                __global float *DTHETA     // angle errors [i,j]
                               )
{
   // This version gets a "kernel" array, NPIX*NPIX*NDIR array with the weights
   // The kernel images are assumed to be rotated in total over 180 degrees
   int id = get_global_id(0) ;
   if (id>=(N*M)) return ;
   int i0 = id % M ;      // column  --   0 <= i < M
   int j0 = id / M ;      // row
   int DP = NPIX/2 ;      // NPIX should be odd !!!   kernel loop -dx, ..., +dx pixels
   if ((i0<DP)||(i0>(M-1-DP))||(j0<DP)||(j0>(N-1-DP))) {
      INT[id] = 0 ; THETA[id] = 0.0f  ;  DTHETA[id] = 0.0f ;  return ;
   }
   float val, sum=0 ;
   float Q=0.0f, U=0.0f, Qsq=0.0f, Usq=0.0f, si, co, theta ;
   for(int k=0; k<NDIR; k++) {
      val = 0.0f ;                            // this time val is a float!
      for(int i=0; i<NPIX; i++) {             // loop over the npix*npix kernel pixels
         for(int j=0; j<NPIX; j++) {
            val += SS[M*(j0-DP+j)+(i0-DP+i)] * KER[k+NDIR*(i+NPIX*j)] ;
         }
      }
      val    =  (val<THRESHOLD) ? 0.0f : (val-THRESHOLD) ;
      theta  =  k*3.14159f/NDIR + 1.57080f ;  // only from 0 to 180 degrees !!
      co     =  cos(2.0f*theta) ;  // re-use si and co
      si     =  sin(2.0f*theta) ; 
      Q     +=  val*co ;
      U     +=  val*si ;
      Qsq   +=  val*co*co ;
      Usq   +=  val*si*si ;
      sum   +=  val ;
   }
   INT[id]   =  sum ;                    // straight sum over angles
   THETA[id] =  0.5f*atan2(U, Q) ;       // [radians]
   si        =  Q*Q ;           // re-use si and co
   co        =  U*U ;
   si        =  sqrt((si*Usq+co*Qsq) / (4.0f*(si+co)*(si+co))) ; // [radians]
   if (si>0.01f)  DTHETA[id] = clamp(si, 0.01f, 3.14159f)   ;
   else           DTHETA[id] = 0.0f ;
}


