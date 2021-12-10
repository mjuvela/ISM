
__kernel void EqTemperature(const int       icell,
                            const float     kE,
                            const float     oplgkE,
                            const float     Emin,
                            __global float *FREQ,   // [NFREQ]
                            __global float *KABS,   // [NFREQ]
                            __global float *TTT,    // [NIP]
                            __global float *ABS,    // [BATCH*NFREQ]
                            __global float *T,      // [BATCH]
                            __global float *EMIT    // [BATCH*NFREQ]
                           ) {
   int id   =  get_global_id(0), iE ;  // id < BATCH
   int ind  = icell+id ;
   if (ind>=CELLS) return ;
   float Ein, wi, TP, f ;
   __global float *A = &(ABS[id*NFREQ]) ;
   // Trapezoid integration of  Ein
   // ABS = number of absorbed photons per dust grain
   //  KABS == SK_ABS[isize] / (GRAIN_DENSITY*S_FRAC) == Q*pi*a^2  == single grain
   Ein = 0.0f ;
   for(int i=1; i<NFREQ; i++) {
      // 0.5*PLANCK == 3.3130348e-27f
      Ein += (A[i]*FREQ[i]+A[i-1]*FREQ[i-1]) * ((FREQ[i]-FREQ[i-1])*3.3130348e-27f) ;
   }
   // Table lookup --- table uses Eout = E * FACTOR because ABS read from file is n_true x FACTOR
   iE    =  clamp((int)floor(oplgkE * log10(Ein/Emin)), 0, NIP-2) ;
   wi    = (Emin*pown(kE,iE+1)-Ein) / (Emin*pown(kE, iE+1)-pown(kE, iE)) ;
   TP    =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1] ;
   T[id] =  TP ;
   
   // printf("Ein %12.4e  Eout [%.2e,%.2e]  iE %d   T %10.3e\n", Ein, Emin, Emin*pown(kE, NIP), iE, TP) ;
   
   // Compute emission for the current cell -- all frequencies, EMIT[icell, NFREQ]
   //  H/K = 4.7995074e-11
   //  EMIT = number of photons, scaled by 1e20  .... NOW WITH FACTOR !!
   //  1e20 * 4*pi * (2*PLANCK/C_LIGHT**2) / PLANCK  =  2.7963945936914554
   for(int ifreq=0; ifreq<NFREQ; ifreq++) {
      f = FREQ[ifreq] ;
      // EMIT[id*NFREQ+ifreq] = 2.79639459f*KABS[ifreq]*(f*f/(exp(4.7995074e-11f*f/TP)-1.0f)) ;
      // H_K = 4.799243348e-11
      // 4*pi*2/C_LIGHT**2 == 2.796394593691455e-20
      EMIT[id*NFREQ+ifreq] = (2.79639459e-20f*FACTOR)*KABS[ifreq]*(f*f/(exp(4.7995074e-11f*f/TP)-1.0f)) ;
      // emission must be still scaled with GRAIN_DENSITY*S_FRAC = actual number of grains per H
   }
}



__kernel void DoSolve4(const      int     batch,   //  0  cells per call
                       const      int     isize,   //  1  size
                       __global   float   *Iw,     //  2  integration weights      [NE*NE*NFREQ]
                       __global   int     *L1,     //  3  integration first bin    [NE*NE]
                       __global   int     *L2,     //  4  integration last bin     [NE*NE]
                       __constant float   *Tdown,  //  5  cooling rates u->u-1     [NE]
                       __global   float   *EA,     //  6  EMIT_ARRAY               [NE*NFREQ] -- was __constant
                       __constant int     *Ibeg,   //  7  first bin emission cal.  [NFREQ]  -- WAS CONSTANT
                       __constant float   *AF,     //  8  absorption fraction      [NFREQ]  -- WAS CONSTANT
                       __global   float   *AABS,   //  9  absorptions              [batch*NFREQ]
                       __global   float   *AEMIT,  // 10  emissions                [batch*NFREQ]
                       __global   float   *LL      // 11  A matrices               [NE*NE*batch] work space
                      ) {
   // This is the stable version!
   // Single size, one cell per work item
   int id  = get_global_id(0) ;
   int lid = get_local_id(0) ;
   int wg  = get_group_id(0) ;
   if (id>=batch) return ;
   __global float  *L    = &(LL[wg*LOCAL*(NE*NE-NE)/2]) ; // allocation is multiple of LOCAL
   __global float  *ABS  = &(AABS[id*NFREQ]) ;
   __global float  *EMIT = &(AEMIT[id*NFREQ]) ;   
   float X[NE]  ;   
   float I ;
   long int  iw_index = 0 ;
   int j, u, l ;
   // DoSolveX is finally more stable... no high-E tail in pdf:s
#define IND(a,b) (((a*a-a)/2+b)*LOCAL+lid)

   // Heating   
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
      
   // bottom row --- is already the original A matrix!
   // row NE-2 is still also the original... except for diagonal that we can skip
   for(j=NE-3; j>0; j--) {
      u = j+1 ;   
      for(int i=0; i<j; i++) L[IND(j,i)] += L[IND(u,i)] ;
   }
   
   // Solve
   X[0] = 1.0e-20f ;
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
      EMIT[j] = I ; 
   }   
   
}


