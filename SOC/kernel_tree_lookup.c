


void __kernel Lookup(const int        CELLS,   //  0  number of cells = rows in ABS
                     __global float  *ABS,     //  1  absorptions in reference bands [CELLS, 3]
                     __global float  *RES,     //  2  results [CELLS, ...]
                     const int        N0,      //  3  number of elements in level 0 vector
                     const float      A0,      //  4  amin at level 0
                     __global int    *V0,      //  5  vec on level 0 (indices of level 1 vector)
                     const int        NV1,     //  6  number of vectors on level 1
                     __global int    *N1,      //  7  number of elements in each (sub)vector on level 1
                     __global int    *I1,      //  8  indices of first element of each (sub)vector...  in V1A
                     __global float  *A1,      //  9  amin for each (sub)vector
                     __global int    *V1A,     // 10  concatenated vector on level 1
                     const int        NV2,     // 11  number of (sub)vectors on level 2
                     __global int    *N2,      // 12  number of elements in each (sub)vector on level 2
                     __global int    *I2,      // 13  indices of first elements of each (sub)vector... in V2A
                     __global float  *A2,      // 14  amin for each (sub)vector on level 2
                     __global float  *V2A      // 15  concatenated vector on level 2 (eventually a matrix)
                    )  {
   int id = get_global_id(0) ;
   if (id>=CELLS) return ;
   __global int    *V1 ;
   __global float  *V2 ;
   int   i0, v1, i1, v2, i2 ;
   float a1, a2, val=0.0f ;
   i0  =  clamp((int)floor(log(ABS[3*id+0]/A0)/log(K)), 0, N0-1) ;     // element in the level 0 vector
   v1  =  V0[i0] ;                                                     // number of level 1 subvector

   if (N1[v1]<=0) {                   // The level 1 vector has no elements => try the neighbouring vectors
      if (N1[max(0,v1-1)]>0) {
         v1 = max(0, v1-1) ;          // previous level 1 vector
      } else {
         v1 = min(v1+1, N1[v1]-1) ;   // next level 1 vector
      }
      if (N1[v1]>0) printf("1a switch\n") ;
   }
   
   if (N1[v1]<=0) {                   // try 2 steps either way
      if (N1[max(0,v1-2)]>0) {
         v1 = max(0, v1-2) ;          // previous level 1 vector
      } else {
         v1 = min(v1+2, N1[v1]-1) ;   // next level 1 vector
      }
      if (N1[v1]>0) printf("1b switch\n") ;
   }
   
   if (N1[v1]<=0) {                   // try 3 steps either way
      if (N1[max(0,v1-3)]>0) {
         v1 = max(0, v1-3) ;          // previous level 1 vector
      } else {
         v1 = min(v1+3, N1[v1]-1) ;   // next level 1 vector
      }
      if (N1[v1]>0) printf("1c switch\n") ;
      if (N1[v1]<1) {
         RES[id] = -1.0f ; return ;
      }
   }

   
   V1  =  &(V1A[I1[v1]]) ;            // pointer to the selected level 1 subvector
   a1  =  A1[v1] ;                    
   i1  =  clamp((int)floor(log(ABS[3*id+1]/a1)/log(K)), 0, N1[v1]-1) ; // index of level 1 subvector
   v2  =  V1[i1] ;                    // number of level 2 subvector
   
   if (N2[v2]<=0) {                   // The level 2 vector has no elements => try the neighbouring vectors
      if (N2[max(0,v2-1)]>0) {
         v2 = max(0, v2-1) ;          // previous level 2 vector
      } else {
         v2 = min(v2+1, N2[v2]-1) ;   // next level 2 vector
      }
      if (N2[v2]>0) printf("2a switch -> %d\n", v2) ;
   }
   
   if (N2[v2]<=0) {                   // The level 2 vector has no elements => try the neighbouring vectors
      if (N2[max(0,v2-2)]>0) {
         v2 = max(0, v2-2) ;          // previous level 2 vector
      } else {
         v2 = min(v2+2, N2[v2]-1) ;   // next level 2 vector
      }
      if (N2[v2]>0) printf("2b switch\n") ;
   }
   
   if (N2[v2]<=0) {                   // The level 2 vector has no elements => try the neighbouring vectors
      if (N2[max(0,v2-3)]>0) {
         v2 = max(0, v2-3) ;          // previous level 2 vector
      } else {
         v2 = min(v2+3, N2[v2]-1) ;   // next level 2 vector
      }
      if (N2[v2]>0) printf("2c switch\n") ;
      if (N2[v2]<1) {
         RES[id] = -2.0f ; return ;
      }
   }
   
   V2  =  &(V2A[I2[v2]]) ;
   a2  =  A2[v2] ;    
   i2  =  clamp((int)floor(log(ABS[3*id+2]/a2)/log(K)), 0, N2[v2]-1) ;
   val =  V2[i2] ;
   if (val<0.0f) { // try a neighbour
      val = V2[max(0,i2-1)] ;
      if (val<0.0f) val = V2[min(i2+1, N2[v2]-1)] ;         
   }
   
   RES[id] =  val ;
}



void __kernel LookupIP(const int        CELLS,   //  0  number of cells = rows in ABS
                       __global float  *ABS,     //  1  absorptions in reference bands [CELLS, 3]
                       __global float  *RES,     //  2  results [CELLS, ...]
                       const int        N0,      //  3  number of elements in level 0 vector
                       const float      A0,      //  4  amin at level 0
                       __global int    *V0,      //  5  vec on level 0 (indices of level 1 vector)
                       const int        NV1,     //  6  number of vectors on level 1
                       __global int    *N1,      //  7  number of elements in each (sub)vector on level 1
                       __global int    *I1,      //  8  indices of first element of each (sub)vector...  in V1A
                       __global float  *A1,      //  9  amin for each (sub)vector
                       __global int    *V1A,     // 10  concatenated vector on level 1
                       const int        NV2,     // 11  number of (sub)vectors on level 2
                       __global int    *N2,      // 12  number of elements in each (sub)vector on level 2
                       __global int    *I2,      // 13  indices of first elements of each (sub)vector... in V2A
                       __global float  *A2,      // 14  amin for each (sub)vector on level 2
                       
                       __global float  *V2A      // 15  concatenated vector on level 2 (eventually a matrix)
                      )  {
   // Interpolation on each three levels => a maximum of eight leaf nodes, if they are all defined
   return ;  // does not yet work better than the simpler Lookup() routine above
   int id = get_global_id(0) ;
   if (id>=CELLS) return ;
   __global int    *V1 ;
   __global float  *V2 ;
   int   i0x, i0y, v1, i1x, i1y, v2, i2x, i2y ;
   float a1, a2, res, resx, resy, x, y, z, wx ;
   
   z   =  ABS[3*id+0] ;
   i0x =  clamp((int)(floor(log(z/A0)/log(K))), 0, N0-1) ;
   x   =  A0*pow(K, i0x+0.5f) ;   // centre of the i0x bin
   i0y =  (z<x) ?  max(0, i0x-1) : min(N0-1, i0x+1) ;
   // printf(" 0 <=  %2d %2d  < %2d\n", i0x, i0y, N0) ; 
   y   =  A0*pow(K, i0y) ;
   wx  =  (i0x==i0y) ?  1.0f : fabs( (z-y)/(x-y+1.0e-25f) ) ;
   // result is   wx*resx + (1-wx)*resy
   i2x = i2y = -1 ;
   
   // calculate resx
   v1    =  V0[i0x] ;                                 // index of level 1 subvector -- V0[i0x] == i0x !! 
   V1    =  &(V1A[I1[v1]]) ;                          // level 1 subvector
   a1    =  A1[v1] ;
   i1x   =  floor(log(ABS[3*id+1]/a1)/log(K)) ;       // element of level 1 subvector
   if ((i1x>=0)&&(i1x<N1[v1])) {
      v2    =  V1[i1x] ;                              // number of the selected level 2 subvector
      V2    =  &(V2A[I2[v2]]) ;                       // selected level 2 subvector
      a2    =  A2[v2] ;                               
      i2x   =  floor(log(ABS[3*id+2]/a2)/log(K)) ;    // element of level 2 subvector
      if (i2x>=N2[v2]) i2x = -1 ;
   }
   
   // calculate resy
   v1    =  V0[i0y] ;
   V1    =  &(V1A[I1[v1]]) ;
   a1    =  A1[v1] ;
   i1y   =  floor(log(ABS[3*id+1]/a1)/log(K)) ;
   if ((i1y>=0)&&(i1y<N1[v1])) {
      v2    =  V1[i1y] ;
      V2    =  &(V2A[I2[v2]]) ;
      a2    =  A2[v2] ;    
      i2y   =  floor(log(ABS[3*id+2]/a2)/log(K)) ;
      if (i2y>=N2[v2]) i2y = -1 ;
   }
   
   if (i2x>=0) if (V2[i2x]<0.0f) i2x = -1 ;
   if (i2y>=0) if (V2[i2y]<0.0f) i2y = -1 ;
   res = 0.0f ;
   
   if ((i2x>=0)&&(i2y>=0)) {
      res = wx*V2[i2x] + (1.0f-wx)*V2[i2y] ;
      printf("  %.3e x %7.3f   +   %.3e x %7.3f   =  %7.3f  ?  %7.3f\n", 
             wx, V2[i2x], 1.0f-wx, V2[i2y], res, ABS[3*id]+ABS[3*id+1]+ABS[3*id+2]) ;
   } else {
      if (i2x>=0) res = V2[i2x] ;
      if (i2y>=0) res = V2[i2y] ;
   }
   RES[id] = res ;
   
}


