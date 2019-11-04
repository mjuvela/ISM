
__kernel void DoSolve(const      int     batch,   //  0  cells per call
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
                      __global   float   *LL      // 11  A matrices               [NE*NE*batch] work space
#if (WITH_X>0)
                      ,__global  float   *XX      // 12  solution vectors         [NE*batch]
#endif
                     ) {
   // This is the stable version!
   int id  = get_global_id(0) ;
   int lid = get_local_id(0) ;
   int wg  = get_group_id(0) ;
   if (id>=batch) return ;
   
   // call without calculations          =   2.0 seconds
   // call + only L matrix calculation   =  13.5 seconds
   // call + L matrix + solution         =  18.0 seconds

#if 1  // This makes threads to address consecutive array elements
   __global float  *L    = &(LL[wg*LOCAL*(NE*NE-NE)/2]) ; // allocation is multiple of LOCAL
#define IND(a,b) (((a*a-a)/2+b)*LOCAL+lid)
#else  // ... and this version is a factor of x10 slower on GPU !!
   __global float  *L    = &(LL[id*(NE*NE-NE)/2]) ; // allocation is multiple of LOCAL
#define IND(a,b) ((a*a-a)/2+b)
#endif
   
   __global float  *ABS  = &(AABS[id*NFREQ]) ;
   __global float  *EMIT = &(AEMIT[id*NFREQ]) ;   
   float XL[NE]  ;   
   float I ;
   int  iw_index = 0 ;
   int j, u, l ;
   // Heating
   
   // 2019-11-04 --- in test problem this was now faster on GPU than the the use of local AA, no effect on CPU
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
   
   //   // using local AA[]
   //   __local float AA[LOCAL*NFREQ] ;
   //   for(int i=0; i<NFREQ; i++)     AA[lid*NFREQ+i] = ABS[i]*AF[i] ;
   //   for(l=0; l<NE-1; l++) {
   //      for(u=l+1; u<NE; u++) {
   //         I = 0.0f ;
   //         for(int i=L1[l*NE+u]; i<=L2[l*NE+u]; i++) {
   //            I += Iw[iw_index]  * AA[lid*NFREQ+i] ;
   //            iw_index++ ;
   //         }
   //         L[IND(u,l)] = max(I, 0.0f) ;
   //      }
   //   }
   
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
   I = 0.0 ;        for(int i=0; i<NE; i++)  I     +=  XL[i] ;
   I = 1.0f / I ;   for(int i=0; i<NE; i++)  XL[i]  =  XL[i]*I ;
   // Emission
   // __attribute__((opencl_unroll_hint(4)))
#pragma unroll 4
   for(j=0; j<NFREQ; j++) {
      I = 0.0f ;
      for(int i=Ibeg[j]; i<NE; i++)  I +=  EA[j*NE+i] * XL[i] ;
      EMIT[j] = I ; 
   }   
#if (WITH_X>0)
   for(int i=0; i<NE; i++)  XX[id*NE+i] = clamp(XL[i], 1.0e-35f, 10.0f) ;
#endif
}
