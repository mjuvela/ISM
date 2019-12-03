
//  N    = rows
//  M    = columns, this runs faster
//  FWHM = FWHM in pixels

// indexing   [i,j]          = [i+M*j]

__kernel void LIC(
                  __global float *S,      // input data [N]
                  __global float *T,      // vector angle, east of north [radians]
                  __global float *L       // output LIC image
                 )  {
   
   int  id  = get_global_id(0) ;  // one output pixel per work item
   if (id>=(N*M)) return ;
   int i0 = id % M ;    // column   0 <= i < M
   int j0 = id / M ;    // row      0 <= j < N
   
   // assume that the data is well sampled => do convolution at one pixel steps
   float K = -4.0f/log(2.0f)/(FWHM*FWHM) ;
   int   D = (int)(2.0f*FWHM) ;  // maximum distance each way
   
   float y  =  T[id] ;
   float dx = -sin(y) ;
   float dy = +cos(y) ;

   float sum=0.0f, wei=1.0e-10f, w ;
   int i, j ;
   for(int l=-D; l<=+D; l++) {  // steps along the vector direction
      i    =  (int)floor(i0+l*dx) ;
      j    =  (int)floor(j0+l*dy) ;
      w    =   exp(K*l*l) ;
      if ((i>=0)&&(j>=0)&&(i<M)&&(j<N)) {
         y    =   S[i+M*j] ;
         if (y>=0.0f) {
            sum +=  w*y ;
            wei +=  w ;
         }
      }
   }
   if (wei>0.0f) L[id] = sum/wei ;
   else          L[id] = 0.0f ;
}

