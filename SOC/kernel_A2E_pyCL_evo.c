
inline void atomicAdd_g_f(volatile __global float *addr, float val) {
#if 1
   union{
      unsigned int u32;
      float        f32;
   } next, expected, current;
   current.f32    = *addr;
   do {
      expected.f32 = current.f32;
      next.f32     = expected.f32 + val;
      current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                                     expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
#else
   *addr += val ;
#endif
}




__kernel void DoSolveEvo(const      int     batch,      //  0  cells per call
                         __global   float   *Iw_all,    //  1  integration weights      [NSIZE, NE*NE*NFREQ]
                         __global   int     *L1_all,    //  2  integration first bin    [NSIZE, NE*NE]
                         __global   int     *L2_all,    //  3  integration last bin     [NSIZE, NE*NE]
                         __global   float   *Tdown_all, //  4  cooling rates u->u-1     [NSIZE, NE]
                         __global   float   *EA_all,    //  5  EMIT_ARRAY               [NSIZE, NE*NFREQ]
                         __global   int     *Ibeg_all,  //  6  first bin emission cal.  [NSIZE, NFREQ]
                         __constant float   *AF,        //  7  absorption fraction      [NSIZE, NFREQ]
                         __global   float   *AABS,      //  8  absorptions              [batch*NFREQ]
                         __global   float   *AEMIT,     //  9  emissions                [batch*NFREQ]
                         __global   float   *L_all      // 10  A matrices (lower trid.) [GLOBAL*(NE*NE-NE)/2]
                        ) {
   // Work group does a single cell, local work items loop over sizes
   int id   = get_global_id(0) ;
   int lid  = get_local_id(0) ;
   int gid  = get_group_id(0) ;
   if (gid>=batch) return ;  // one work group per cell, process all sizes within the group
   if (lid>=NSIZE) return ;
#if 1
   __global float  *L     = &(L_all[gid*(int)((NE*NE-NE)/2)]) ;  // (NE*NE-NE)/2 per work item
#else
   // too much shared data
   __local float L[(NE*NE-NE)/2] ;
   for(int i=lid; i<(NE*NE-NE)/2; i+=LOCAL)  L[i]  =  L_all[gid*(int)((NE*NE-NE)/2)+i] ;
#endif
   __global float  *GEMIT = &(AEMIT[gid*NFREQ]) ;        // [NFREQ] global emission vector
   __global int    *L1, *L2, *Ibeg ;
   __global float  *Iw, *Tdown, *EA ;
   
#if 1
   float X[NE]  ;
#else
   __local float XL[LOCAL*NE] ;
   __local float *X = &(XL[lid*NE]) ;
#endif
   float I ;
   long int  iw_index ;
   int j, u, l ;
#define IND(a,b) (((a*a-a)/2+b)*LOCAL+lid)  // L_all[GLOBAL*(NE*NE-NE)/2]
   
   
#if 1
   __global float  *ABS   = &(AABS[gid*NFREQ]) ;         // work group = one cell
#else
   __local float ABS[NFREQ] ;
   for(int ifreq=lid; ifreq<NFREQ; ifreq+=LOCAL) ABS[ifreq] = AABS[gid*NFREQ+ifreq] ;
#endif
   
   for(int isize=lid; isize<NSIZE; isize+=LOCAL) { // each work item = one size
      
      Iw    =  &(Iw_all[isize*NE*NE*NFREQ]) ;
      L1    =  &(L1_all[isize*NE*NE]) ;
      L2    =  &(L2_all[isize*NE*NE]) ;
      Tdown =  &(Tdown_all[isize*NE]) ;
      EA    =  &(EA_all[isize*NE*NFREQ]) ;
      Ibeg  =  &(Ibeg_all[isize*NFREQ]) ;
      
      // Heating   
      iw_index = 0 ;
      for(l=0; l<NE-1; l++) {
         for(u=l+1; u<NE; u++) {
            I = 0.0f ;
            for(int i=L1[l*NE+u]; i<=L2[l*NE+u]; i++) {
               I += ABS[i] * Iw[iw_index]  * AF[isize*NFREQ+i] ;
               iw_index++ ;
            }
            L[IND(u,l)] = max(I, 0.0f) ;
         }
      }
      
      // bottom row --- is already the original A matrix !
      // row NE-2 is still also the original... except for diagonal that we can skip
      for(j=NE-3; j>0; j--) {
         u = j+1 ;   
         for(int i=0; i<j; i++) {
            L[IND(j,i)] += L[IND(u,i)] ;
         }
      }
      
      // Solve
      X[0] = 1.0e-20f ;  // private variable
      for(j=1; j<NE; j++) {
         X[j] = 0.0f ;
         for(int i=0; i<=j-1; i++)  X[j] += L[IND(j,i)] * X[i] ;
         X[j] /= (Tdown[j] + 1.0e-30f) ;
         X[j]  = max(X[j], 0.0f) ;
         if (X[j]>1.0e20f) {
            for(int i=0; i<=j; i++) X[i] *= 1.0e-20f ;
         }
      }
      
      // Normalise
      I = 0.0 ;
      for(int i=0; i<NE; i++) I += X[i] ;
      I = 1.0f / I ;
      for(int i=0; i<NE; i++) X[i] = X[i]*I ;  
      
      // Emission
      for(j=0; j<NFREQ; j++) {
         I = 0.0f ;
         for(int i=Ibeg[j]; i<NE; i++)  I +=  EA[j*NE+i] * X[i] ;         
         atomicAdd_g_f(&(GEMIT[j]), I) ;
      }   
      
   } // for isize
   
}







__kernel void DoSolveEvo2(const      int     batch,      //  0  cells per call
                          __global   float   *Iw_all,    //  1  integration weights      [NSIZE, NE*NE*NFREQ]
                          __global   int     *L1_all,    //  2  integration first bin    [NSIZE, NE*NE]
                          __global   int     *L2_all,    //  3  integration last bin     [NSIZE, NE*NE]
                          __global   float   *Tdown_all, //  4  cooling rates u->u-1     [NSIZE, NE]
                          __global   float   *EA_all,    //  5  EMIT_ARRAY               [NSIZE, NE*NFREQ]
                          __global   int     *Ibeg_all,  //  6  first bin emission cal.  [NSIZE, NFREQ]
                          __constant float   *AF,        //  7  absorption fraction      [NSIZE, NFREQ]
                          __global   float   *AABS,      //  8  absorptions              [batch*NFREQ]
                          __global   float   *AEMIT,     //  9  emissions                [batch, NFREQ, NSIZE]
                          __global   float   *L_all      // 10  A matrices (lower trid.) [GLOBAL*(NE*NE-NE)/2]
                         ) {
   // Work group does a single size, work items loop over cells
   // Host must sum the emission from different grain sizes !!
   int id    = get_global_id(0) ;
   int lid   = get_local_id(0) ;
   int gid   = get_group_id(0) ;
   __global float  *L   = &(L_all[id*(int)((NE*NE-NE)/2)]) ;  // (NE*NE-NE)/2 per work item
   __global float  *ABS    ;        // vector for current cell, absorbed photons [NFREQ]
   __global int    *L1, *L2, *Ibeg ;
   __global float  *Iw, *EA ;

   // each work group does a single size, we may have n times NSIZE work groups =>
   const int  isize = gid % NSIZE ;      
   const int  STEP  = ((GLOBAL/LOCAL)/NSIZE)*LOCAL ;  // number of work items doing this SIZE
   const int  wi    = ((gid/NSIZE))*LOCAL+lid ;       // running index among all work items doing this SIZE
      
#if 1
   float X[NE]  ;
#else
   __local float XL[LOCAL*NE] ;
   __local float *X = &(XL[lid*NE]) ;
#endif
   float I ;
   long int  iw_index ;
   int j, u, l ;
#define INDEX(a,b) (((a*a-a)/2+b))  // L[(NE*NE-NE)/2], work space for individual work item
   
   // These the same for all work items in the group, for all cells
   Iw    =  &(Iw_all[isize*NE*NE*NFREQ]) ;
   L1    =  &(L1_all[isize*NE*NE]) ;
   L2    =  &(L2_all[isize*NE*NE]) ;
#if 0
   __global float *Tdown =  &(Tdown_all[isize*NE]) ;
#else
   __local Tdown[NE] ;
   for(int i=lid; i<NE; i+=LOCAL) Tdown[i] = Tdown_all[isize*NE+i] ;
#endif
   EA    =  &(EA_all[isize*NE*NFREQ]) ;
   Ibeg  =  &(Ibeg_all[isize*NFREQ]) ;
   
   
   for(int icell=wi; icell<batch; icell+=STEP) {   // one work item per cell

      ABS   = &(AABS[icell*NFREQ]) ;
      
      // Heating   
      iw_index = 0 ;
      for(l=0; l<NE-1; l++) {
         for(u=l+1; u<NE; u++) {
            I = 0.0f ;
            for(int i=L1[l*NE+u]; i<=L2[l*NE+u]; i++) {
               I += ABS[i] * Iw[iw_index]  * AF[isize*NFREQ+i] ;
               iw_index++ ;
            }
            L[INDEX(u,l)] = max(I, 0.0f) ;
         }
      }
      
      // bottom row --- is already the original A matrix !
      // row NE-2 is still also the original... except for diagonal that we can skip
      for(j=NE-3; j>0; j--) {
         u = j+1 ;   
         for(int i=0; i<j; i++) {
            L[INDEX(j,i)] += L[INDEX(u,i)] ;
         }
      }
      
      // Solve
      X[0] = 1.0e-20f ;  // private variable
      for(j=1; j<NE; j++) {
         X[j] = 0.0f ;
         for(int i=0; i<=j-1; i++)  X[j] += L[INDEX(j,i)] * X[i] ;
         X[j] /= (Tdown[j] + 1.0e-30f) ;
         X[j]  = max(X[j], 0.0f) ;
         if (X[j]>1.0e20f) {
            for(int i=0; i<=j; i++) X[i] *= 1.0e-20f ;
         }
      }
      
      // Normalise
      I = 0.0 ;
      for(int i=0; i<NE; i++) I += X[i] ;
      I = 1.0f / I ;
      for(int i=0; i<NE; i++) X[i] = X[i]*I ;  
      
      // Emission
      for(int ifreq=0; ifreq<NFREQ; ifreq++) {
         I = 0.0f ;
         for(int i=Ibeg[j]; i<NE; i++)  I +=  EA[ifreq*NE+i] * X[i] ;
         AEMIT[icell*NFREQ*NSIZE+ifreq*NSIZE+isize] = I ;
      }   
      
   } // for isize
   
   
   
}
