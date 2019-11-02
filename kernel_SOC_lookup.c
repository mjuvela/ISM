


void __kernel Lookup(const int        CELLS,   //  0  number of cells = batch
                     __global float  *ABS,     //  1  absorptions in reference bands [batch, nlfreq]
                     __global float  *RES,     //  2  results [batch, nfreq]
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
                     __global int    *V2A,     // 15  indices to SARRAY
                     __global float  *SARRAY   // 16  SARRAY[SIND, nfreq] emission vectors from the library
                    )  {
   int id = get_global_id(0) ;
   if (id>=CELLS) return ;
   __global int    *V1, *V2 ;
   int   i0, v1, i1, v2, i2 ;
   float a1, a2  ;
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
   
   V2  =  &(V2A[I2[v2]]) ;            // level 2 vector with indices of emission vectors
   a2  =  A2[v2] ;    
   i2  =  clamp((int)floor(log(ABS[3*id+2]/a2)/log(K)), 0, N2[v2]-1) ; // element of the level 2 vector
   i0  =  V2[i2] ;                    // index of the emission vector (reuse i0)
   if (i0<0) {
      i0  =  V2[max(0,i2-1)] ;
      if (i0<0.0f) i0 = V2[min(i2+1, N2[v2]-1)] ;         
   }
   if (i0>=0) {
      // copy to res[id,:] the emission vector SARRAY[i0,:]
      for(int i=0; i<NFREQ; i++)   RES[id*NFREQ+i] = SARRAY[i0*NFREQ+i] ;
   } else {
      for(int i=0; i<NFREQ; i++)   RES[id*NFREQ+i] = 0.0f ;
   }
}


