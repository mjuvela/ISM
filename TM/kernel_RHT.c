
//  NDIR = number of directions
//  DK   = diameter of smoothing kernel (pixels)
//  DW   = diameter of circular regions (pixels)
//  Z    = threshold value
//  N    = rows
//  M    = columns, this runs faster

// indexing   [i,j]          = [i+M*j]
//            [itheta, i, j] = [ itheta + NDIR*(i+N*j) ]



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
   float sum=0.0f, wsum=0.0f, w, r2 ;
   for(int i=max(0, i0-RK); i<min(i0+RK+1, M); i++) {          // row
      for(int j=max(0, j0-RK); j<min(j0+RK+1, N); j++) {       // column
         r2 = (i-i0)*(i-i0) + (j-j0)*(j-j0) ;
#ifdef GAUSS  // Gaussian convolution
         if (r2<(4.0f*R2)) {            
            w     = exp(-K*r2) ;
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
#if 0
      SS[id] = sum / wsum ;
#else
      if ((S[id]-sum/wsum)>0.0) SS[id] = 1.0f ;
      else                      SS[id] = 0.0f ;
#endif
   } else {
      SS[id] = 0.0f ;
   }
}



__kernel void R_kernel(
                       __global float *SS,  // thresholded image
                       __global float *RR,  // output R[i,j,i_theta]
                       __global float *SIN,
                       __global float *COS
                      )
{
   // This kernel does not return RR = R for each direction separately
   // If one wants to test agreement to given direction (B field), one
   // could add that as an argument and use the loop below to calculate
   // the probability of the coincidence
   int id = get_global_id(0) ;
   if (id>=(N*M)) return ;
   int i0 = id % M ;      // column  --   0 <= i < M
   int j0 = id / M ;      // row
   int val, i, j ;
   int RW  = (DW-1)/2 ;   // DW should be an odd number
   float si=0.0f, co=0.0f ;
   int   sum=0 ;
   for(int k=0; k<NDIR; k++) {
      float theta =  k*6.2831853f/NDIR ;  // only over pi ???
      float dx    = -sin(theta) ;    // east from north
      float dy    = +cos(theta) ;
      val = 0 ;
      for(int l=-RW; l<=+RW; l++) {  // loop along the line
         i = (int)floor(i0+l*dx) ;
         j = (int)floor(j0+l*dy) ;
         if ((i>=0)&&(j>=0)&&(i<M)&&(j<N)) {
            if (SS[i+M*j]>0.0f) val++ ;
         }
      }
      if (val<(Z*DW)) val = 0.0f ;
      // calculate SIN and COS averages
      si  += val * sin(2.0f*theta) ;
      co  += val * cos(2.0f*theta) ;
      sum += val ;
   }
   RR[id]  = sum ;  // straight sum of ON pixels 
   SIN[id] = si ;   // sin(2*theta) weighted average
   COS[id] = co ;   // cos(2*theta) weighted average
}




