
#define DEBUG 0




__kernel void Median(__global float *X,   // input  array [N,M]
                     __global int   *D,  
                     __global float *Y    // output array [N,M] of (pseudo-)median values
                    ) {
   // Another version of "exact" median calculation.
   //    make a histogram of the data values and zoom into the bin containing the median
   //    sort the elements only in the selected bin
   int id = get_global_id(0) ; // one work item per pixel of the input map
   if  (id>=(N*M)) return ;   
   int lid = get_local_id(0) ;
   int NO=0, below=0, igap, d, i, j, k ;
   int i0 = id % M ;
   int j0 = id / M ;
   float r2, R22=R2*R2, R12=R1*R1, xmin, xmax, delta ; 
   
   // Local work space up to 100 bins
#define BINS 50
   __local float  BB[LOCAL*BINS] ;
   __local float *B = &(BB[lid*BINS]) ;
   
   // Calculate mean and stdev ... approximation would be enough
   float s=0.0f, s2=0.0f, x, a, b ;
   xmin = 1.0e32f ; xmax =-1.0e32f ;
   for(j=max(0,j0-R2); j<=min(j0+R2,N-1); j++) {
      for(i=max(0,i0-R2); i<=min(i0+R2,M-1); i++) {
#if (CIRCLE>0)
         r2  = (j-j0)*(j-j0) + (i-i0)*(i-i0) ;
         if ((r2<R12)||(r2>R22)) continue ;
#endif
         x   = X[j*M+i] ;
         s  += x ;
         s2 += x*x ;
         NO += 1 ;    // actual number of elements within the footprint
         if (x<xmin) xmin = x ;
         if (x>xmax) xmax = x ;
      }
   }
   s2 =  sqrt(s2/NO - s*s/(NO*NO)) ;
   s  =   s/NO ;
   
   // As the initial range take [s-2*s2, s+2*s2] where,  s = mean, s2 = stdev
   a     =  max(xmin, s-2.0f*s2) ;
   b     =  min(xmax, s+2.0f*s2) ;
   delta =  (b-a)/(BINS-2.0f) ;
   
   // Put elements to histogram and iterate until the number of elements in the 
   // median-containing bin is less than BINS
#ifdef DEBUG
   int counter = 0 ;
#endif
   
   while(1) {
      
      for(k=0; k<BINS; k++) B[k] = 0.0f ;
      
      for(j=max(0,j0-R2); j<=min(j0+R2,N-1); j++) {
         for(i=max(0,i0-R2); i<=min(i0+R2,M-1); i++) {
#if (CIRCLE>0)
            r2  = (j-j0)*(j-j0) + (i-i0)*(i-i0) ;
            if ((r2<R12)||(r2>R22)) continue ;
#endif
            x   =  X[j*M+i] ;
            // Put the element in a bin (all elements go somewhere)
            k   =  floor(1.0f+(x-a)/delta) ; // bin [lower,upper[
            if (k<1) {
               B[0] += 1.0f ;
            } else {
               if (k>=(BINS-1)) {
                  B[BINS-1] += 1.0f ;
               } else {
                  B[k] += 1.0f ;
               }
            }
         }  
      }            
      // Find the bin that contains the median... and refine
      i = 0 ;
      for(k=0; k<BINS; k++) {
         i += B[k] ;
         if (i>(NO/2)) break ;  // median is in bin k... or between bin k and k-1
      }      
      
#ifdef DEBUG
      counter += 1 ;
      if (counter>10) {
         printf("A: id=%d, Counter=%d, B[%d]=%.1f points, [%.4e,%.4e] \n", id, counter, B[k], a, b) ;
      }
#endif
      
      
      if (B[k]<BINS) break ;    // time to sort elements in that bin, now with limits [a,b]
      // refine to that single bin
      if ((k>0)&&(k<(BINS-1))) {
         // "normal" case, median was in the range [a,b] and we just refine to that bin
         // After the first iteration, median remains in the range [a,b] => no edge bins
         b     =  a + delta*k     ;  // upper border of bin k
         a     =  a + delta*(k-1) ;  // lower border of bin k
      } else {
         // Median was in an edge bin.... initial iteration, because of very large outliers?
         // If there were outliers, the initial stdev was large, a is close to xmin and b close to xmax
         // and median cannot be far outside [a,b]
         if (k==0) {   
            b = a ;  a -= 10.0f*delta ;
         } else {  // last bin
            a = b ;  b += 10.0f*delta ;
         }
      }
      delta = (b-a)/(BINS-2.0f) ;
   } // while (1)
   
   
   // We are left with <BINS elements that are in the range [a,b]
   // perhaps just copy data to the local array and sort?
   xmin = -1.0e30f ;  // we may need the largest element below the [a,b] bin
   lid  =  0 ;
   for(j=max(0,j0-R2); j<=min(j0+R2,N-1); j++) {
      for(i=max(0,i0-R2); i<=min(i0+R2,M-1); i++) {
#if (CIRCLE>0)
         r2  = (j-j0)*(j-j0) + (i-i0)*(i-i0) ;
         if ((r2<R12)||(r2>R22)) continue ;
#endif
         x   =  X[j*M+i] ;
         // because of rounding errors this has to be identical to the above loop!!
         d   =  floor(1.0f+(x-a)/delta) ; // bin [lower,upper[
         if (d<k) {
            below += 1 ;   xmin = max(xmin,x) ;
         } 
         if (d==k) {   // bin [a,b[
            B[lid] = x ;   lid += 1 ;
         }
      }
   }        
   
   
   // Now we have lid elements in B -> sort the array B[]
   igap=0 ;
   while(1) {
      d  =  D[igap] ; 
      for(i=d; i<lid; i++) {
         x = B[i] ;
         for (j=i; (j>=d)&&(B[j-d]>x); j-= d) {
            B[j] = B[j-d] ;
         }
         B[j] = x ;
      }
      if (d<=1) break ;
      igap += 1 ;
   }            
   
   // The median value
   if (NO%2==0) {   // the median is the average of two values
      // ... either two elements in B[] or the average of xmin and B[0]
      i = (NO/2) - below ;
      if (i==0) {
         Y[id] = 0.5f*(B[0]+xmin) ;     
      } else {
         Y[id] = 0.5f*(B[i]+B[i-1]) ;
      }
      // if (Y[id]>0.7f) printf("(1) bin %3d  k=%3d points  B[%3d]=%7.4f, %d/%d\n", k, B[k], i, k) ;
   } else {         // median == single element
      Y[id] = B[(NO-1)/2-below] ;
      // if (Y[id]>0.7f) printf("(2) bin %3d  k=%3d points  B[%3d]=%7.4f, %d/%d\n", k, B[k], (NO-1)/2-below, k) ;
   }
   
}







__kernel void MAD(__global float *X,   // input  array [N,M]
                  __global int   *D,   // 
                  __global float *Y    // output array [N,M] of (pseudo-)median values
                 ) {
   // Calculate MAD = median absolute deviation estimate for every pixel in the image
   // Note: the routine assumes that one can find a value range around the median
   //       so that the range contains > BINS values
   //       ==> if there are BINS equal values (~median) this WILL FAIL!
   int id = get_global_id(0) ; // one work item per pixel of the input map
   if  (id>=(N*M)) return ;   
   int lid = get_local_id(0) ;
   int NO, below, igap, d, i, j, k ;
   int i0 = id % M ;
   int j0 = id / M ;
   float r2, R22=R2*R2, R12=R1*R1, xmin, xmax, delta ; 
   float s, s2, x, a, b ;
   float MED=0.0f ;
   
#ifdef DEBUG
   int counter = 0 ;
#endif
   
   
   // Local work space up to 100 bins
#define BINS 50
   __local float  BB[LOCAL*BINS] ;
   __local float *B = &(BB[lid*BINS]) ;
   
   
   
   for(int ITER=0; ITER<2; ITER++) {   
      
      
#ifdef DEBUG
      counter = 0 ;
#endif
      
      // calculate median( x-MED )
      //   ITER=0  ->  MED=0.0
      //   ITER=1  ->  MED= median(x)      
      NO = 0 ;  s = 0.0f ;  s2 = 0.0f ;
      xmin = 1.0e32f ; xmax =-1.0e32f ;
      for(j=max(0,j0-R2); j<=min(j0+R2,N-1); j++) {
         for(i=max(0,i0-R2); i<=min(i0+R2,M-1); i++) {
#if (CIRCLE>0)
            r2  = (j-j0)*(j-j0) + (i-i0)*(i-i0) ;
            if ((r2<R12)||(r2>R22)) continue ;
#endif
            if (ITER==0)   x  =      X[j*M+i] ;
            else           x  = fabs(X[j*M+i] - MED) ;
            s  += x ;
            s2 += x*x ;
            NO += 1 ;    // actual number of elements within the footprint
            if (x<xmin) xmin = x ;
            if (x>xmax) xmax = x ;
         }
      }
      s2 =  sqrt(s2/NO - s*s/(NO*NO)) ;
      s  =   s/NO ;  // ITER=0, normal mean; ITER=1, <x-median(x)> 
      
      // As the initial range take [s-2*s2, s+2*s2] where,  s = mean, s2 = stdev
      a     =  max(xmin, s-2.0f*s2) ;
      b     =  min(xmax, s+2.0f*s2) ;
      delta =  (b-a)/(BINS-2.0f) ;
#ifdef DEBUG
      if (delta<1.0e-6) printf("?????? %12.4e %12.4e %12.4e,  s=%10.3e, s2=%10.3e, x = %10.3e, %10.3e, no=%D\n",
                              a, b, delta, s, s2, xmin, xmax, NO) ;
#endif
      
      // Put elements to histogram and iterate until the number of elements in the 
      // median-containing bin is less than BINS
      while(1) {
         for(k=0; k<BINS; k++) B[k] = 0.0f ;         
         for(j=max(0,j0-R2); j<=min(j0+R2,N-1); j++) {
            for(i=max(0,i0-R2); i<=min(i0+R2,M-1); i++) {
#if (CIRCLE>0)
               r2  = (j-j0)*(j-j0) + (i-i0)*(i-i0) ;
               if ((r2<R12)||(r2>R22)) continue ;
#endif
               if (ITER==0)   x  =      X[j*M+i] ;
               else           x  = fabs(X[j*M+i] - MED) ;
               // Put the element in a bin (all elements go somewhere)
               //    x<a  =>  k = 0
               //    x>b  =>  k = BINS-1 .....      x>=b =>  k=1+(>=b-a)/(b-a)*(BINS-2) = BINS-1
               // Bin borders x ~ [ a+(k-1)*delta,  a+k*delta ]
               k   =  floor(1.0f+(x-a)/delta) ;    // bin [lower,upper[
               if (k<1) {
                  B[0] += 1.0f ;
               } else {
                  if (k>=(BINS-1)) {
                     B[BINS-1] += 1.0f ;
                  } else {
                     B[k] += 1.0f ;
                  }
               }
            }  
         }            
         // Find the bin that contains the median... and refine
         i = 0 ;
         for(k=0; k<BINS; k++) {
            i += B[k] ;
            if (i>(NO/2)) break ;  // median is in bin k... or between bin k and k-1
         }      
         
         
#ifdef DEBUG
         counter += 1 ;
         if (counter>10) {
            printf("MAD: ITER %d, id=%d, Count=%d, B[%d]=%.1f points, NO=%d, [%.4e,%.4e] %.3e\n", ITER, id, counter, k, B[k], NO, a, b, delta) ;
         }
#endif
         
         
         if (B[k]<BINS) break ; // time to sort elements in that bin, now with limits [a,b]
         
         // refine to that single bin
         if ((k>0)&&(k<(BINS-1))) {
            // "normal" case, median was in the range [a,b] and we just refine to that bin
            // After the first iteration, median remains in the range [a,b] => no edge bins
            b     =  a + delta*(k+1.1f) ;   // upper border of bin k
            a     =  a + delta*(k-2.1f) ;   // lower border of bin k
         } else {
            // Median was in an edge bin.... initial iteration, because of very large outliers?
            // If there were outliers, the initial stdev was large, a is close to xmin and b close to xmax
            // and median cannot be far outside [a,b]
            if (k==0) {   
               // printf("id %d, count %3d::  DOWN  %10.3e %10.3e %10.3e -->  ", id, counter, a, b, delta) ;
               b = a ;  a -= BINS*delta ;
               // printf("%10.3e %10.3e %10.3e\n", a, b, (b-a)/(BINS-2.0f)) ;
            } else {  // last bin
               a = b-delta ;  b += BINS*delta ;
               // printf("id %d UP\n", id) ;
            }
         }
         delta = (b-a)/(BINS-2.0f) ;
      } // while (1)
      
      
      // We are left with <BINS elements that are in the range [a,b]
      // perhaps just copy data to the local array and sort?
      xmin  = -1.0e30f ;  // we may need the largest element below the [a,b] bin
      below =  0 ;
      lid   =  0 ;
      for(j=max(0,j0-R2); j<=min(j0+R2,N-1); j++) {
         for(i=max(0,i0-R2); i<=min(i0+R2,M-1); i++) {
#if (CIRCLE>0)
            r2  = (j-j0)*(j-j0) + (i-i0)*(i-i0) ;
            if ((r2<R12)||(r2>R22)) continue ;
#endif
            if (ITER==0)   x  =      X[j*M+i] ;
            else           x  = fabs(X[j*M+i] - MED) ;
            // because of rounding error this has to be identical to the above loop!!
            d   =  floor(1.0f+(x-a)/delta) ; // bin [lower,upper[
            if (d<k) {
               below += 1 ;   xmin = max(xmin,x) ;
            } 
            if (d==k) {   // bin [a,b[
               B[lid] = x ;   lid += 1 ;
            }
         }
      }        
      
      
#ifdef DEBUG
      counter = 0 ;
#endif
      
      
      // Now we have lid elements in B -> sort the array B[]
      igap=0 ;
      while(1) {
         d  =  D[igap] ; 
         for(i=d; i<lid; i++) {
            x = B[i] ;
            for (j=i; (j>=d)&&(B[j-d]>x); j-= d) {
               B[j] = B[j-d] ;
            }
            B[j] = x ;
         }
         if (d<=1) break ;
         igap += 1 ;
         
#ifdef DEBUG
         if (counter>15) {
            printf("MAD-B: ??? if %6d, counter %5d, lid %d\n", id, counter, lid) ;
         }
#endif
      }            
      
      // The median value
      if (NO%2==0) {   // the median is the average of two values
         i = (NO/2) - below ;
         if (i==0) {
            MED = 0.5f*(B[0]+xmin) ;     
         } else {
            MED = 0.5f*(B[i]+B[i-1]) ;
         }
      } else {         // median == single element
         MED = B[(NO-1)/2-below] ;
      }
      
   } // for ITER
   
   Y[id] = MED ;  //    median[   | x - median(x) |  ]
   
}







