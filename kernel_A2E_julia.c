
__kernel void DoSolve(const      int     NE,
                      const      int     batch,   //  0  cells per call
                      const      int     isize,   //  1  size
                      __global   float   *Iw,     //  2  integration weights      [NE*NE*NFREQ]
                      __global   int     *L1,     //  3  integration first bin    [NE*NE]
                      __global   int     *L2,     //  4  integration last bin     [NE*NE]
                      __constant float   *Tdown,  //  5  cooling rates u->u-1     [NE]
                      __global   float   *EA,     //  6  EMIT_ARRAY               [NE*NFREQ]
                      __constant int     *Ibeg,   //  7  first bin emission cal.  [NFREQ]
                      __constant float   *AF,     //  8  absorption fraction      [NFREQ]
                      __global   float   *AABS,   //  9  absorptions              [batch*NFREQ]
                      __global   float   *AEMIT,  // 10  emissions                [batch*NFREQ]
                      __global   float   *LL,     // 11  A matrices               [NE*NE*batch] work space
                      __global   float   *XX      // 12  solution vectors         [NE*batch]
                     ) {
   // This is the stable version!
   int id  = get_global_id(0) ;
   int lid = get_local_id(0) ;
   int wg  = get_group_id(0) ;
   if (id>=batch) return ;
   
   // return ;
   
   __global float  *L    = &(LL[wg*LOCAL*(NE*NE-NE)/2]) ; // allocation is multiple of LOCAL
   __global float  *ABS  = &(AABS[id*NFREQ]) ;
   __global float  *EMIT = &(AEMIT[id*NFREQ]) ;   
#if 0
   float XL[NE]  ;
#else
   __global float *XL = &(XX[id*NE]) ;
#endif
   float I ;
   long int  iw_index = 0 ;
   int j, u, l ;
#define IND(a,b) (((a*a-a)/2+b)*LOCAL+lid)
   // Heating
#if 0 // --- local AA[] not used
   for(l=0; l<NE-1; l++) {
      for(u=l+1; u<NE; u++) {
         I = 0.0f ;
         for(int i=L1[l*NE+u]; i<=L2[l*NE+u]; i++) {
            I += ABS[i] * Iw[iw_index]  * AF[i] ;
            iw_index++ ;
         }
         L[IND(u,l)] = max(I, 0.0f) ;
      }
   }
#else // --- else using local AA[] array
   // slightly slower on CPU, slightly faster on GPU (like 10.9 -> 9.3 seconds)
   __local float AA[LOCAL*NFREQ] ;
   for(int i=0; i<NFREQ; i++)     AA[lid*NFREQ+i] = ABS[i]*AF[i] ;
   for(l=0; l<NE-1; l++) {
      for(u=l+1; u<NE; u++) {
         I = 0.0f ;
         for(int i=L1[l*NE+u]; i<=L2[l*NE+u]; i++) {
            I += Iw[iw_index]  * AA[lid*NFREQ+i] ;
            iw_index++ ;
         }
         L[IND(u,l)] = max(I, 0.0f) ;
      }
   }
#endif
   // bottom row --- is already the original A matrix !
   // row NE-2 is still also the original... except for the diagonal that we can skip
   for(j=NE-3; j>0; j--) {
      u = j+1 ;   
      for(int i=0; i<j; i++) {
         L[IND(j,i)] += L[IND(u,i)] ;
      }
   }
   // Solve
   XL[0] = 1.0e-20f ;
   for(j=1; j<NE; j++) {
      XL[j] = 0.0f ;
      for(int i=0; i<=j-1; i++)  XL[j] += L[IND(j,i)] * XL[i] ;
      XL[j] /= (Tdown[j] + 1.0e-30f) ;
      XL[j]  = max(XL[j], 0.0f) ;
      if (XL[j]>1.0e20f) {
         for(int i=0; i<=j; i++) XL[i] *= 1.0e-20f ;
      }
   }
   // Normalise
   I = 0.0 ;
   for(int i=0; i<NE; i++) I += XL[i] ;
   I = 1.0f / I ;
   for(int i=0; i<NE; i++) XL[i] = XL[i]*I ;
   // Emission
   for(j=0; j<NFREQ; j++) {
      I = 0.0f ;
      for(int i=Ibeg[j]; i<NE; i++)  I +=  EA[j*NE+i] * XL[i] ;
      EMIT[j] = I ; 
   }   
   for(int i=0; i<NE; i++)  XX[id*NE+i] = clamp(XL[i], 1.0e-35f, 10.0f) ;
}
