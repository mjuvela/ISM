

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
      j0 = clamp((int)floor(log(ADD[i*NFREQ+IFREQ0]/amin0) / log(k0)), 0, BINS0-1) ;
      if (i0==j0) {
         j1 = clamp((int)floor(log(ADD[i*NFREQ+IFREQ1]/amin1) / log(k1)), 0, BINS1-1) ;
         if (i1==j1) {
            j2 = clamp((int)floor(log(ADD[i*NFREQ+IFREQ2]/amin2) / log(k2)), 0, BINS2-1) ;
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



void __kernel InterpolateX(const    float  amin0,    // AMIN on level 0
                          const    float  k0,       // step K on level 0
                          __global float *AMIN1,    // AMIN1[i0]  AMIN for every level 1 vector
                          __global float *K1,       // step K1[i0] for every level 1 ector
                          __global float *AMIN2,    // AMIN2[i0*BINS1+i1]  AMIN for every level 2 vector
                          __global float *K2,       // step K2[i0*BINS1+i1] for every level 2 vector
                          __global int   *NUM_OLD,  // number of added absorption vectors per bin
                          __global int   *NUM,
                          __global float *ALL       // all absorption vectors in the tree
                         ) {
   // Go through all i2 bins. When NUM[id]==0, find the closest absorption vector within the
   // same (i0, i1) bin (guaranteed to exist?) and interpolate/scale it to the missing vector
   // Call this before Divide so we can include the weighting with the number of input vectors,
   // when using one bin from either side of the current one
   // 2019-11-28:  there is *no* guarantee that there is any valid vectors for (i0,i1) !!
   // 2019-11-30:  since all work items were updating ALL[], there was potential fore race conditions
   //              => added separate NUM_OLD output array
   const int id = get_global_id(0) ; 
   if (id>=(BINS0*BINS1*BINS2)) return ;  // one work item per leaf
   NUM[id] = NUM_OLD[id] ;
   if (NUM[id]>0) return ;                // this bin is already ok
   int i0, i1, i2, i2x ;
   float x ;
   // index ~ BINS0*BINS1*BINS2
   i0   =  id / (BINS1*BINS2) ;
   i1   =  (id/BINS2) % BINS1 ;      
   i2   =  id % BINS2 ; 
   // find the first populated vector below i2, if exists... still within the same parent bin (i0, i1)
   i2x = -1 ;
   for(int i=i2-1; i>=0; i--) {
      if (NUM_OLD[id+(i-i2)]>0) {
         i2x = i ;
         break ;
      }           
   }   
   if (i2x>=0) {    // extrapolate vector from bin i2x to bin i2
      for(int ifreq=0; ifreq<NFREQ; ifreq++) {
         ALL[id*NFREQ+ifreq]  +=  ALL[(id+i2x-i2)*NFREQ+ifreq] ;
      }
      NUM[id] +=  NUM_OLD[id+i2x-i2] ;
   }
   // find the first populated vector above i2, if exists still within the same (i0, i1) parent bin
   i2x = -1 ;
   for(int i=i2+1; i<BINS2; i++) {
      if (NUM_OLD[id+(i-i2)]>0) {
         i2x = i ;
         break ;
      }           
   }   
   if (i2x>=0) {    // extrapolate vector from bin i2x to bin i2
      for(int ifreq=0; ifreq<NFREQ; ifreq++) {
         ALL[id*NFREQ+ifreq]  +=  ALL[(id+i2x-i2)*NFREQ+ifreq] ;
      }
      NUM[id] +=  NUM_OLD[id+i2x-i2] ;
   }
#if 0
   if (ALL[id*NFREQ+IFREQ1]<=0.0f) {
      // printf("*** INTERPOLATION FAILURE: %3d %3d %3d ***\n", i0, i1, i2) ;
      printf("[%d,%d,%d]", i0, i1, i2) ;
   }
#endif
}





void __kernel Interpolate(const    float  amin0,    // AMIN on level 0
                           const    float  k0,       // step K on level 0
                           __global float *AMIN1,    // AMIN1[i0]  AMIN for every level 1 vector
                           __global float *K1,       // step K1[i0] for every level 1 ector
                           __global float *AMIN2,    // AMIN2[i0*BINS1+i1]  AMIN for every level 2 vector
                           __global float *K2,       // step K2[i0*BINS1+i1] for every level 2 vector
                           __global int   *NUM_OLD,  // number of added absorption vectors per bin
                           __global int   *NUM,
                           __global float *ALL       // all absorption vectors in the tree
                          ) {
   // Go through all i2 bins. When NUM[id]==0, find the closest absorption vector within the
   // same (i0, i1) bin (guaranteed to exist?) and interpolate/scale it to the missing vector
   // Call this before Divide so we can include the weighting with the number of input vectors,
   // when using one bin from either side of the current one
   // This version does weighted average of vectors on the lower and higher side, if both exist
   // *** worse than the plain Interpolate() routine above ??? ***
   const int id = get_global_id(0) ; 
   if (id>=(BINS0*BINS1*BINS2)) return ;  // one work item per leaf
   NUM[id] = NUM_OLD[id] ;
   if (NUM[id]>0) return ;                // this bin is already ok
   int i0, i1, i2, iL, iH ;
   float k2, wL, wH ;
   k2   =  K2[i0*BINS1+i1] ;   
   // index ~ BINS0*BINS1*BINS2
   i0   =  id / (BINS1*BINS2) ;
   i1   =  (id/BINS2) % BINS1 ;      
   i2   =  id % BINS2 ; 
   // find the first populated vector below i2, if exists... still within the same parent bin (i0, i1)
   iL = -1 ;
   for(int i=i2-1; i>=0; i--) {
      if (NUM_OLD[id+(i-i2)]>0) {
         iL = i ;
         break ;
      }           
   }   
   // find the first populated vector above i2, if exists still within the same (i0, i1) parent bin
   iH = -1 ;
   for(int i=i2+1; i<BINS2; i++) {
      if (NUM_OLD[id+(i-i2)]>0) {
         iH = i ;
         break ;
      }           
   }   
   if ((iL>=0)&&(iH>=0)) {  // try linear interpolation, bin values are ~ k2^i
#if 0
      wH  =  (pow(k2,i2)-pow(k2,iL)) / (pow(k2,iH)-pow(k2,iL)) ;   // [0,1]
#else
      wH  =  (i2-iL) / (1.0f*iH-1.0f*iL) ;   // [0,1]
#endif
      wL  =  (1.0f-wH) / NUM_OLD[id+iL-i2] ;
      wH /=              NUM_OLD[id+iH-i2] ;
      for(int ifreq=0; ifreq<NFREQ; ifreq++) {
         ALL[id*NFREQ+ifreq] =  wL * ALL[(id+iL-i2)*NFREQ+ifreq]  +  wH * ALL[(id+iH-i2)*NFREQ+ifreq] ;
      }
      NUM[id] = 1 ;
   } else {   
      iL  =  max(iL, iH) ;   // at most one of these is  >=0
      if (iL>=0) {
         for(int ifreq=0; ifreq<NFREQ; ifreq++)  ALL[id*NFREQ+ifreq]  =  ALL[(id+iL-i2)*NFREQ+ifreq] ;
         NUM[id] +=  NUM_OLD[id+iL-i2] ;
      }
   }
#if 0
   if (ALL[id*NFREQ+IFREQ1]<=0.0f) {
      // printf("*** INTERPOLATION FAILURE: %3d %3d %3d ***\n", i0, i1, i2) ;
      printf("[%d,%d,%d]", i0, i1, i2) ;
   }
#endif
}





void __kernel Divide(__global float *ALL,  __global int *NUM) {
   // division ALL/NUM
   const int id = get_global_id(0) ;
   if (id>=(BINS0*BINS1*BINS2)) return ;    // one work item per leaf
   if (NUM[id]>0) {
      for(int ifreq=0; ifreq<NFREQ; ifreq++) ALL[id*NFREQ+ifreq] /= NUM[id] ;
   } 
}



void __kernel FillX(__global float *ALL, __global int *NUM_OLD, __global int *NUM, __global float *K1) {
   // Fill any remaining missing emission vectors (NUM==0) based on nearby existing vectors
   // This is called after Divide() => NUM is either 1 or 0
   const int id = get_global_id(0) ;
   if (id>=(BINS0*BINS1*BINS2)) return ;   // ALL[id*NFREQ+ifreq]
   int i0, i1, i2, count, i2x, found ;
   float coeff ;
   NUM[id] = NUM_OLD[id] ;
   if (NUM[id]>0) return ;                 // is already ok
   //  BINS0*BINS1*BINS2
   i0   =  id/(BINS1*BINS2) ;
   i1  =  (id/BINS2) % BINS1 ;             // level 1 index of the missing bin
   i2  =   id % BINS2 ;                    // index to i2=0 vector  ..... absorptions [i2*NFREQ+ifreq]
   // next absorption vector below
   found = 0 ;
   for(i2x=id-1; i2x>=0; i2x--) {          // i2x index to leaf NFREQ vectors  .... any leaf below !!
      if (NUM_OLD[i2x]>0) {
         found = 1 ; break ;
      }
   }
   if (found) {
      count = 1 ;
      for(int ifreq=0; ifreq<NFREQ; ifreq++)   ALL[id*NFREQ+ifreq]  =  ALL[i2x*NFREQ+ifreq]  ;
   }   
   found = 0 ;
   for(i2x=id+1; i2x<BINS0*BINS1*BINS2; i2x++) {  // next absorption vector above .... any leaf above !!
      if (NUM_OLD[i2x]>0) {
         found = 1 ; break ;
      }
   }
   if (found>0) {
      count += 1 ;
      for(int ifreq=0; ifreq<NFREQ; ifreq++)   ALL[id*NFREQ+ifreq]  +=  ALL[i2x*NFREQ+ifreq] ;
   }
   if (count==2){
      for(int ifreq=0; ifreq<NFREQ; ifreq++)   ALL[id*NFREQ+ifreq] *= 0.5f ;
   } 
   if (count==0) {
      printf("**** ALL VECTORS BELOW i0=%d, i1=%d REMAINS UNDEFINED ***\n", i0, i1) ;
   } else {
      NUM[id] = 1 ;
   }
}




void __kernel Fill(__global float *ALL, __global int *NUM_OLD, __global int *NUM, __global float *K1) {
   // Fill any remaining missing emission vectors (NUM==0) based on nearby existing vectors
   // This is called after Divide() => NUM is either 1 or 0
   const int id = get_global_id(0) ;
   if (id>=(BINS0*BINS1*BINS2)) return ;   // ALL[id*NFREQ+ifreq]
   int i0, i1, i2, count, iL=-1, iH=-1, found ;
   float coeff ;
   NUM[id] = NUM_OLD[id] ;
   if (NUM[id]>0) return ;                 // is already ok
   //  BINS0*BINS1*BINS2
   i0  =   id/(BINS1*BINS2) ;
   i1  =  (id/BINS2) % BINS1 ;             // level 1 index of the missing bin
   i2  =   id % BINS2 ;                    // index to i2=0 vector  ..... absorptions [i2*NFREQ+ifreq]
   // next absorption vector below
   for(int i=id-1; i>=0; i--) {            // i2x index to leaf NFREQ vectors  .... any leaf below !!
      if (NUM_OLD[i]>0) {   iL = i ; break ;    }
   }
   // next absorption vector above
   for(int i=id+1; i<BINS0*BINS1*BINS2; i++) {  // next absorption vector above .... any leaf above !!
      if (NUM_OLD[i]>0) {   iH = i ; break ;    }
   }
   if ((iL>=0)&&(iH>=0)) {
      for(int ifreq=0; ifreq<NFREQ; ifreq++) {
         ALL[id*NFREQ+ifreq]  =  0.5f*(ALL[iL*NFREQ+ifreq]+ALL[iH*NFREQ+ifreq]) ;
      }
      NUM[id] = 1 ;
   } else {
      iL = max(iL, iH) ;
      if (iL>=0) {
         for(int ifreq=0; ifreq<NFREQ; ifreq++)  ALL[id*NFREQ+ifreq] = ALL[iL*NFREQ+ifreq] ;
      } else {
         NUM[id] = 0 ;          
      }
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

