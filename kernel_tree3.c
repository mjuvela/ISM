

void __kernel Populate(const    int    BATCH,    // number of vectors in ADD  <= GLOBAL
                       const    float  amin0,    // AMIN on level 0
                       const    float  k0,       // step K on level 0
                       __global float *AMIN1,    // AMIN1[i0]  AMIN for every level 1 vector
                       __global float *K1,       // step K1[i0] for every level 1 ector
                       __global float *AMIN2,    // AMIN2[i0*BINS+i1]  AMIN for every level 2 vector
                       __global float *K2,       // step K2[i0*BINS+i1] for every level 2 vector
                       __global float *ADD,      // absorption vectors to be added, ADD[BATCH, NFREQ]
                       __global int   *NUM,      // number of added absorption vectors per bin
                       __global float *ALL       // all absorption vectors in the tree ADD[BINS^3, NFREQ]
                      ) 
{
   const int id = get_global_id(0) ;   
   if (id>=(BINS0*BINS1*BINS2))  return ;    // one work item per leaf
   int   i0, i1, i2, j0, j1, j2 ;
   float amin1, k1, amin2, k2 ;
   // the bin for this work item
   i0   =  id / (BINS1*BINS2) ;
   i1   =  (id/BINS2) % BINS1 ;   
   i2   =  id % BINS2 ;   
   amin1 = AMIN1[i0] ;             k1  =  K1[i0] ;
   amin2 = AMIN2[i0*BINS1+i1] ;    k2  =  K2[i0*BINS1+i1] ;   
   if ((k1<0.0f)||(k2<0.0f)) { // dead branch
      return ;
   }
   // loop over all input vectors
   for(int i=0; i<BATCH; i++) {
      j0 = log(ADD[i*NFREQ+IFREQ0]/amin0) / log(k0) ;
      if (i0==j0) {
         j1 = log(ADD[i*NFREQ+IFREQ1]/amin1) / log(k1) ;
         if (i1==j1) {
            j2 = log(ADD[i*NFREQ+IFREQ2]/amin2) / log(k2) ;
            if (i2==j2) {  // should go into our bin
               for(int ifreq=0; ifreq<NFREQ; ifreq++) {
                  ALL[id*NFREQ+ifreq] += ADD[i*NFREQ+ifreq] ;
               }
               NUM[id] += 1 ;
            }
         }
      }
   }   
}



void __kernel Interpolate(const    float  amin0,    // AMIN on level 0
                          const    float  k0,       // step K on level 0
                          __global float *AMIN1,    // AMIN1[i0]  AMIN for every level 1 vector
                          __global float *K1,       // step K1[i0] for every level 1 ector
                          __global float *AMIN2,    // AMIN2[i0*BINS1+i1]  AMIN for every level 2 vector
                          __global float *K2,       // step K2[i0*BINS1+i1] for every level 2 vector
                          __global int   *NUM,      // number of added absorption vectors per bin
                          __global float *ALL       // all absorption vectors in the tree
                         ) {
   // Go through all i2 bins. When NUM[id]==0, find the closest absorption vector within the
   // same (i0, i1) bin (guaranteed to exist) and interpolate/scale it to the missing vector
   // Call this before Divide so we can include the weighting with the number of input vectors,
   // when using one bin from either side of the current one
   const int id = get_global_id(0) ; 
   if (id>=(BINS0*BINS1*BINS2)) return ;  // one work item per leaf
   if (NUM[id]>0) return ;                // the bin is already ok
   int i0, i1, i2, i2x ;
   float x ;
   i0   =  id / (BINS1*BINS2) ;
   i1   =  (id/BINS2) % BINS1 ;      
   i2   =  id % BINS2 ; 
   // find the first populated vector below i2, if exists... still within the same parent bin (i0, i1)
   i2x = -1 ;
   for(int i=i2-1; i>=0; i--) {
      if (NUM[id+(i-i2)]>0) {
         i2x = i ;
         break ;
      }           
   }   
   if (i2x>=0) {    // extrapolate vector from bin i2x to bin i2
      x  =  pow(K2[i0*BINS1+i1], i2-i2x) ;  // i2x<i2 => k>1
      for(int ifreq=0; ifreq<NFREQ; ifreq++) {
         ALL[id*NFREQ+ifreq] += x*ALL[(id+i2x-i2)*NFREQ+ifreq] ;
      }
      NUM[id] +=  NUM[id+i2x-i2] ;
      // printf("FILL IN %3d:    i2=%3d, %3d     <==   i2x=%3d, %3d\n", id, i2, NUM[id], i2x, NUM[id+i2x-i2]) ;
   }
   // find the first populated vector above i2, if exists
   i2x = -1 ;
   for(int i=i2+1; i<BINS2; i++) {
      if (NUM[id+(i-i2)]>0) {
         i2x = i ;
         break ;
      }           
   }   
   if (i2x>=0) {    // extrapolate vector from bin i2x to bin i2
      x  =  pow(K2[i0*BINS1+i1], i2-i2x) ;  // i2x>i2 => k<1
      for(int ifreq=0; ifreq<NFREQ; ifreq++) {
         ALL[id*NFREQ+ifreq] += x*ALL[(id+i2x-i2)*NFREQ+ifreq] ;
      }
      NUM[id] +=  NUM[id+i2x-i2] ;
      // printf("FILL IN %3d:    i2=%3d, %3d     <==   i2x=%3d, %3d\n", id, i2, NUM[id], i2x, NUM[id+i2x-i2]) ;
   }
   if (ALL[id*NFREQ+IFREQ1]<=0.0f) printf("*** INTERPOLATION FAILURE: %3d %3d %3d ***\n", i0, i1, i2) ;
}



void __kernel Divide(__global float *ALL,  __global int *NUM) {
   // division ALL/NUM
   const int id = get_global_id(0) ;
   if (id>=(BINS0*BINS1*BINS2)) return ;    // one work item per leaf
   if (NUM[id]>0) {
      for(int ifreq=0; ifreq<NFREQ; ifreq++) ALL[id*NFREQ+ifreq] /= NUM[id] ;
   } 
}



void __kernel Fill(__global float *ALL, __global int *NUM, __global float *K1) {
   const int id = get_global_id(0) ;
   if (id>=(BINS0*BINS1*BINS2)) return ;   // ALL[id*NFREQ+ifreq]
   int i0, i1, i2, count, i2x, found ;
   float coeff ;
   i0   =  id/(BINS1*BINS2) ;
   i1  =  (id/BINS2) % BINS1 ;             // level 1 index of the missing bin
   i2  =   id % BINS2 ;                    // index to i2=0 vector  ..... absorptions [i2*NFREQ+ifreq]
   // next absorption vector below
   found = 0 ;
   for(i2x=id; i2x>=0; i2x--) {    // i2x index to leaf NFREQ vectors
      if (NUM[i2x]>0) {
         found = 1 ; break ;
      }
   }
   if (found) {
      count = 1 ;
      for(int ifreq=0; ifreq<NFREQ; ifreq++) {
         // id-i2x steps on level 2  =>   (id-i2x)/BINS steps in level 1 index
         coeff                   =  pow(K1[i0], (id-i2x)/BINS1) ;      // hopefully still same i0
         ALL[id*NFREQ+ifreq]     =  ALL[i2x*NFREQ+ifreq] * coeff ;     // fill i2=0 vector
      }
   }
   // next absorption vector above
   found = 0 ;
   for(i2x=id+1; i2x<BINS0*BINS1*BINS2; i2x++) {
      if (NUM[i2x]>0) {
         found = 1 ; break ;
      }
   }
   if (found>0) {
      count += 1 ;
      for(int ifreq=0; ifreq<NFREQ; ifreq++) {
         coeff                 =  pow(K1[i0], (id-i2x)/BINS1) ;
         ALL[id*NFREQ+ifreq]  +=  ALL[i2x*NFREQ+ifreq] * coeff ;     // fill i2=0 vector
      }
   }
   if (count==2){
      for(int ifreq=0; ifreq<NFREQ; ifreq++)   ALL[id*NFREQ+ifreq] *= 0.5f ;
   } 
   if (count==0) {
      printf("**** ALL VECTORS BELOW i0=%d, i1=%d REMAIN UNDEFINED ***\n", i0, i1) ;
   }
}



void __kernel Lookup(const int        batch,
                     const float      AMIN0,
                     const float      K0,
                     __global float  *AMIN1,    // [BINS0]
                     __global float  *K1,       // [BINS0]
                     __global float  *AMIN2,    // [BINS0*BINS1]
                     __global float  *K2,       // [BINS0*BINS1]
                     __global float  *ALL,      // [BBB,   NFREQ] 
                     __global float  *ABS,      // [BATCH, NLFREQ]
                     __global float  *EMIT      // [BATCH, NFREQ ]
                    ) {
   const int id = get_global_id(0) ;
   const int gs = get_global_size(0) ;
   int i0, i1, i2, bin, i1x, ind, low, hi ;
   
   for(int ind=id; ind<batch; ind+=gs) {   // loop over input vectors    ABS -> EMIT
      i0     =  clamp((int)(log(ABS[3*ind+0]/AMIN0)              / log(K0)),              0, BINS0-1) ;
      i1     =  clamp((int)(log(ABS[3*ind+1]/AMIN1[i0])          / log(K1[i0])),          0, BINS1-1) ;
      i2     =  clamp((int)(log(ABS[3*ind+2]/AMIN2[i0*BINS1+i1]) / log(K2[i0*BINS1+i1])), 0, BINS2-1) ;
      // printf("%2d %2d %2d\n", i0, i1, i2) ;
      bin    =  i0*BINS1*BINS2 + i1*BINS2 + i2 ;  // the leaf
      for(int ifreq=0; ifreq<NFREQ; ifreq++)  EMIT[ind*NFREQ+ifreq]  =  ALL[bin*NFREQ+ifreq] ;
      
#if 0
      low = hi = 0 ;
      if (EMIT[ind*NFREQ]<=0.0f) {  // search first valid i1 bin ***BELOW***
         for(i1x=i1-1; i1x>=0; i1x--) {
            bin  = i0*BINS1*BINS2 + i1x*BINS2 + (BINS2-1) ;
            if (ALL[bin*NFREQ]>0.0f) {
               for(int ifreq=0; ifreq<NFREQ; ifreq++)  EMIT[ind*NFREQ+ifreq]  =  ALL[bin*NFREQ+ifreq] ;
               low = 1 ; break ;
            }
         }
      }     
      if (EMIT[ind*NFREQ]<=0.0f) {  // search first valid i1 bin ***ABOVE***
         for(i1x=i1+1; i1x<BINS1; i1x++) {
            bin = i0*BINS1*BINS2 + i1x*BINS2 + 0 ;
            if (ALL[bin*NFREQ]>0.0f) {
               for(int ifreq=0; ifreq<NFREQ; ifreq++)  EMIT[ind*NFREQ+ifreq] +=  ALL[bin*NFREQ+ifreq] ;
               hi = 1 ;  break ;
            }
         }
         if ((low==1)&&(hi==1)) {     // we added two vectors, one from above, one from below
            for(int ifreq=0; ifreq<NFREQ; ifreq++)  EMIT[ind*NFREQ+ifreq] *= 0.5f ;
         }
      }
#endif
   }
}

