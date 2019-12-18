

// Image dimensions (N, M)
// Weight array P  2*WDIM+1 times 2*WDIM+1
//  - P is kept in global memory, large P might not fit in constant memory


__kernel void Con(
                  __global   float* X,      // input image [N*M]
                  __global   float* W,      // pixel weights [N*M]
                  __global   float* P,      // convolving beam [(2*WDIM+1)*(2*WDIM+1)]
                  __global   float* Z       // result, convolved image
                 )
{
   int  id  = get_global_id(0) ;   // one pixel
   if (id>=NPIX) return ;
   int  i0 = id / M ;              // column runs faster, i ~ row
   int  j0 = id % M ;              // j ~ column
   int  i, j ;
   float w, wsum, y, sum ;
   sum   = 0.0f ;
   wsum  = 0.0f ;
#if (BORDER==0)   // pixels outside the map are simply ignored
   for(i=max(0, i0-WDIM); i<min(i0+WDIM+1, N); i++) {                   // row
      for(j=max(0, j0-WDIM); j<min(j0+WDIM+1, M); j++) {                // column
         //  beam element (i+WDIM-i0, j+WDIM-j0), map pixel (i,j)
         w     =   P[(WDIM+i-i0)*(2*WDIM+1)+(WDIM+j-j0)] * W[i*M+j] ;   // weight
         y     =                                           X[i*M+j] ;   // pixel value
         if (y!=MASKED_VALUE) {    // zero == masked value !!
            wsum +=   w     ;
            sum  +=   w * y ;
         }
      }
   }
#endif
#if (BORDER==1)   // use nearest map pixel when pixel is outside the map
   int ii, jj ;
   for(i=-WDIM; i<=WDIM; i++) {                       // row
      for(j=-WDIM; j<=WDIM; j++) {                    // column
         ii    =   clamp(i0+i, 0, N-1) ;
         jj    =   clamp(j0+j, 0, M-1) ;
         // map pixel (ii,jj), beam elemement (i,j)
         w     =   P[(i+WDIM)*(2*WDIM+1)+j+WDIM] * W[ii*M+jj] ;   // weight
         y     =                                   X[ii*M+jj] ;   // pixel value
         if (y!=MASKED_VALUE) {    // zero == masked value !!
            wsum +=   w     ;
            sum  +=   w * y ;
         }
      }
   }
#endif
   if (fabs(wsum)>1.0e-30f) {
      Z[id] = sum / wsum ;
   } else {
      Z[id] = 0.0f ;
   }
}


__kernel void ConC(
                   __global   float* X,      // input image [N*M]
                   __global   float* W,      // pixel weights [N*M]
                   __constant float* P,      // convolving beam [(2*WDIM+1)*(2*WDIM+1)]
                   __global   float* Z       // result, convolved image
                  )
{
   // Version with the beam in constant memory
   int  id  = get_global_id(0) ;   // one pixel
   if (id>=NPIX) return ;
   int  i0 = id / M ;              // column runs faster, i ~ row
   int  j0 = id % M ;              // j ~ column
   int  i, j ;
   float w, wsum, y, sum ;
   sum   = 0.0f ;
   wsum  = 0.0f ;
   for(i=max(0, i0-WDIM); i<min(i0+WDIM+1, N); i++) {                   // row
      for(j=max(0, j0-WDIM); j<min(j0+WDIM+1, M); j++) {                // column
         w     =   P[(WDIM+i-i0)*(2*WDIM+1)+(WDIM+j-j0)] * W[i*M+j] ;   // weight
         y     =                                           X[i*M+j] ;   // pixel value
         if (y!=MASKED_VALUE) {    // zero == masked value !!
            wsum +=   w     ;
            sum  +=   w * y ;
         }
      }
   }
   if (fabs(wsum)>1.0e-30f) {
      Z[id] = sum / wsum ;
   } else {
      Z[id] = 0.0f ;
   }
}


__kernel void ConCU(
                    __global   float* X,      // input image [N*M]
                    __constant float* P,      // convolving beam [(2*WDIM+1)*(2*WDIM+1)]
                    __global   float* Z       // result, convolved image
                   )
{
   // Version with P in constant memory and without individual pixel weighting
   int  id  = get_global_id(0) ;   // one pixel
   if  (id>=NPIX) return ;
   int  i0 = id / M ;              // column runs faster, i ~ row
   int  j0 = id % M ;              // j ~ column
   int  i, j ;
   float w, y, wsum=0.0f, sum=0.0f ;
   for(i=max(0, i0-WDIM); i<min(i0+WDIM+1, N); i++) {                   // row
      for(j=max(0, j0-WDIM); j<min(j0+WDIM+1, M); j++) {                // column
         w   =   P[(WDIM+i-i0)*(2*WDIM+1)+(WDIM+j-j0)] ;
         y   =   X[i*M+j] ;
         if (y!=MASKED_VALUE) {
            wsum += w ;    sum += w * y ;
         }
      }
   }
   if (wsum>0.5f) {
      Z[id] = sum / wsum ;
   } else {
      Z[id] = 0.0f ;
   }
}



__kernel void ConCUZ(
                     __global   float* X,      // input image [N*M]
                     __constant float* P,      // convolving beam [(2*WDIM+1)*(2*WDIM+1)]
                     __global   float* Z       // result, convolved image
                    )
{
   // Version with P in constant memory and without individual pixel weighting, zero values not ignored
   int  id  = get_global_id(0) ;   // one pixel
   if  (id>=NPIX) return ;
   int  i0 = id / M ;              // column runs faster, i ~ row
   int  j0 = id % M ;              // j ~ column
   int  i, j ;
   float w, y, wsum=0.0f, sum=0.0f ;
   for(i=max(0, i0-WDIM); i<min(i0+WDIM+1, N); i++) {                   // row
      for(j=max(0, j0-WDIM); j<min(j0+WDIM+1, M); j++) {                // column
         w     =  P[(WDIM+i-i0)*(2*WDIM+1)+(WDIM+j-j0)] ;
         y     =  X[i*M+j] ;
         wsum +=  w ;    
         sum  +=  w * y ;
      }
   }
   if (wsum>0.5f) {
      Z[id] = sum / wsum ;
   } else {
      Z[id] = 0.0f ;
   }
}




