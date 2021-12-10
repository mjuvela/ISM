// Simplified version, only plain Gauss-Seidel, better storage usage


# define ERR_LIMIT  1.0e-3f
# define DERR_LIMIT 1.0e-3f


#if 0

__kernel void DoSolveGS(const int batch,        //  0  cells per call
                        const int isize,        //  1  size
                        __global float *Iw,     //  2  integration weights      [NE*NE*NFREQ]
                        __global int   *L1,     //  3  integration first bin    [NE*NE]
                        __global int   *L2,     //  4  integration last bin     [NE*NE]
                        __global float *Tdown,  //  5  cooling rates u->u-1     [NE]
                        __global float *EA,     //  6  EMIT_ARRAY               [NE*NFREQ] -- was __constant
                        __constant int   *Ibeg, //  7  first bin emission cal.  [NFREQ]
                        __constant float *AF,   //  8  absorption fraction      [NFREQ]
                        __global float *AABS,   //  9  absorptions              [batch*NFREQ]
                        __global float *AEMIT,  // 10  emissions                [batch*NFREQ]
                        __global float *LL,     // 11  A matrices               [NE*NE*batch] work space
                        __global float *XX      // 12  solution vectors         [NE*batch]
                       ) {
   int id   = get_global_id(0) ;
   int lid  = get_local_id(0) ;
   int wg   = get_group_id(0) ;
   if (id>=batch) return ;
   __global float *L    = &(   LL[wg*LOCAL * (NE*NE-NE)/2]) ;
   __global float *X    = &(   XX[id*NE]) ;

   
   // There may be something wrong here because the emissivity in diffuse regions
   // is always lower than with A2E or the explicit solver... while dense regions are ok!
   // In test case at 10um one is 10 orders of magnitude below the SED peak and the error is ~10%
   // ... this is fine but even at 10um the level difference is ~5%
   
   
   
   // We will process "batch" cells using GLOBAL workers.
   __global float *ABS  = &( AABS[id*NFREQ]) ;
   __global float *EMIT = &(AEMIT[id*NFREQ]) ;
   
   // if (ABS[0]<0.0f) return ;  // was a link, not a real cell
   
   
#define DIAG   
   // DIAG = store DIAG values and do not normalise diagonal values to one
   //      = no Jacobi preconditioning
#ifdef DIAG
   double D[NE] ;
#endif
   
   
   // int NITER = 200 ;
   int ITER, NITER = 120 ;
   float  s, z, I, err, derr ;
   
   // We use lower diagonal matrix and the elements above the diagonal,
   //   diagonal values will be normalised to 1.0 and need not be stored (unless DIAG).
   // Because each work iterm will read its own L[j,i] at the same time,
   //   store those in adjacent locations: W0L00, W1L00, ..., W0L01, W1, L02, ...
   // L has a size of (N*N-N)/2 elements.
   // Indexing is   A[j,i] = L[(j*j-j)/2+i].
   // U contains values from above the diagonal, *before* rescaling of A, 
   //   these are the Tdown values,  U[j] = Tdown[j+1], so it is indexed using the row index
   //   These should fit in local memory.
   
#if (NE>NFREQ)
   float TMP[NE+NE] ;
#else
   float TMP[NFREQ+NE] ;
#endif
   float *XL = TMP ;            // NE elements used in XL[]
   float *U  = &(TMP[NE]) ;     // NE elements (actually NE-1 used)
   
   
   for(int i=0; i<NFREQ; i++) {
      TMP[i] = AF[i]*ABS[i] ;
#if 0
      if (TMP[i]>1.0e-32) ;  else  TMP[i] = 1.0e-32 ;
#endif
   }
   
   
   
   // TMP[NFREQ-1] = clamp(TMP[NFREQ-1], 0.0f, TMP[NFREQ-2]) ;  -- not enough to avoid hot pixel
   TMP[NFREQ-1] = clamp(TMP[NFREQ-1], 0.0f, 0.5f*TMP[NFREQ-2]) ; // THIS FIXED IT
   
   
   
   
   int iw_index = 0 ;   // Note: Iw must be a vector containing only the [L1,L2] elements
   for(int l=0; l<NE-1; l++) {
      for(int u=l+1; u<NE; u++) {
         I = 0.0f ;
         for(int i=L1[l*NE+u]; i<=L2[l*NE+u]; i++) {
            // I += AF[i] * ABS[i] * Iw[iw_index] ;
            I +=         TMP[i] * Iw[iw_index] ;
            iw_index++ ;
         } 
         // without max() the results are garbage
         // test case:  L>=1.0e-27  X>1e-19,   L>=1.0e-30  X>1e-22
         L[((u*u-u)/2+l)*LOCAL+lid] = max(1.0e-29f, I) ;
      }
   }
   // --- after this TMP not used directly --- XL and U use the same allocation
      
   
   // Cooling   [ i, i+1 ], above diagonal
   for(int j=1; j<NE; j++)  U[j-1] = Tdown[j] ;
   
   

   // Diagonal entries ... each equation will be scaled so that the diagonal comes 1.0
   // in the mean time, store diagonal values to XL
   for(int i=0; i<NE; i++) {         // column
      s  =  0.0f ;
      for(int j=i+1; j<NE; j++) {    // lower triangle
         s +=  L[((j*j-j)/2+i)*LOCAL+lid] ;
      }
      s += (i>0) ? U[i-1] : 0.0f ;
#ifdef DIAG  // we store diagonal values to D without normalisation!!!
      D[i]  = min(-s, 1.0e-29f) ;    // store diagonal values, no normalisation
#else
      XL[i] = min(-s, -1.0e-29f) ;   // diagonal value
#endif
   }

#ifndef DIAG   // D[] not stored, assumed 1.0
   // Renormalise equations so that the diagonal == 1.0
   double d ;
   for(int j=0; j<NE-1; j++) {                             // loop over equations, j = row
      d = 1.0f/XL[j] ;
      for(int i=0; i<j; i++)  L[((j*j-j)/2+i)*LOCAL+lid] *= d ;        // lower triangle part of the row
      U[j] *= d ;                                          // U[j] is part of equation on the row j
   }
   // Diagonal is now implicitly 1.0
#endif
   
   
   //barrier(CLK_LOCAL_MEM_FENCE) ;
   
   

#if 1   // If host provided init array = vector for each work item separately
   for(int i=0; i<NE; i++) XL[i] = X[i] ;
#else   // one init vector for all work items
   for(int i=0; i<NE; i++) XL[i] = X[i] = XX[i] ;   
#endif
   

   
   
   
   // some Gauss-Seidel iterations
   for(ITER=0; ITER<NITER; ITER++) {
      z = 0.0f ;
      for(int j=0; j<NE-1; j++) {             // equation j .... last one skipped!
         s      =  0.0f ;
         for(int i=0; i<j; i++)  {            // row from the lower triangle
            s  += L[((j*j-j)/2+i)*LOCAL+lid] * XL[i] ;
         }
         // element right of the diagonal, still same row j, column i=j+1
         s += (j<(NE-1)) ? (U[j]* XL[j+1]) : 0.0f ;
#if 0    // unncessary!
         b      =  (j<(NE-1)) ? 0.0f : 1.0f ;
         b      =  clamp(b-s,   1.0e-27f, 1.0f) ;  // (b-s)/A[j,j] but A[j,j]==1.0 !!
         XL[j]  =  b ;
#else
# ifdef DIAG  // original diagonal entries in D[]
         XL[j]  =  clamp(-s/D[j], 1.0e-27, 1.0) ;
# else
         XL[j]  =  clamp(-s, 1.0e-27f, 1.0f) ;
# endif
         z     +=  XL[j] ;
#endif
      } // for j=0, <NE-1
      // Renormalisation
      z += XL[NE-1] ;   // not in the loop above ???
      z  = 1.0f/z ;  
      for(int i=0; i<NE; i++)  XL[i] *= z ;
      
      if (ITER%9==8) {
         derr = 0.0f ;       // change in X values
         for(int i=0; i<NE; i++) {
            derr = max(derr, fabs(XL[i]-X[i])) ;   
            X[i] = XL[i] ;             // update global array -- store the old solution for the next round
         }
         err  = 0.0f ;       // change in residual
#if 1
         for(int j=0; j<NE-1; j++) {   // ignore last equation???? that replaced by explicit normalisation??
            // diagonal term
# ifdef DIAG
            s   =  XL[j] * D[j] ;
#else
            s   =  XL[j] ;   // A[j,j] == 1.0
#endif
            // lower triangle
            for(int i=0; i<j; i++)  s += L[((j*j-j)/2+i)*LOCAL+lid] * XL[i] ;
            // term right of the diagonal ... A[j,j+1] == U[j]
            s   += (j<(NE-1)) ? (U[j] * XL[j]) : 0.0f ;
            err += fabs(s) ;
         }
#endif
         if ((err<ERR_LIMIT)||(derr<DERR_LIMIT)) {
            break ;
         }
      } // ITER % something
      
   } // for iter
   
   // if (id%3789==0) printf("[%6d]    NITER  %3d\n", id, ITER) ;
   
#if 1
   // Clip low X values and remormalise
   z = 0.0f ;
   for(int i=0; i<NE; i++) {
      XL[i] =  (XL[i]>1.0e-20f) ? (XL[i]) : 1.0e-25f ;
      z += XL[i] ;
   }
   for(int i=0; i<NE; i++) XL[i] /= z ;
#endif
   
   // Calculate emission -- cumulative over different sizes
   for(int ifreq=0; ifreq<NFREQ; ifreq++) {
      s = 0.0f ;
      for(int i=Ibeg[ifreq]; i<NE; i++)  s +=  EA[ifreq*NE+i] * XL[i] ;
      EMIT[ifreq] = s ; 
   }
   for(int i=0; i<NE; i++) X[i] = XL[i] ;
   
   //barrier(CLK_LOCAL_MEM_FENCE) ;     // <----- WHY IS THIS NECESSARY ???
}







__kernel void DoSolve4(const int batch,        //  0  cells per call
                       const int isize,        //  1  size
                       __global float *Iw,     //  2  integration weights      [NE*NE*NFREQ]
                       __global int   *L1,     //  3  integration first bin    [NE*NE]
                       __global int   *L2,     //  4  integration last bin     [NE*NE]
                       __global float *Tdown,  //  5  cooling rates u->u-1     [NE]
                       __global float *EA,     //  6  EMIT_ARRAY               [NE*NFREQ] -- was __constant
                       __constant int   *Ibeg, //  7  first bin emission cal.  [NFREQ]
                       __constant float *AF,   //  8  absorption fraction      [NFREQ]
                       __global float *AABS,   //  9  absorptions              [batch*NFREQ]
                       __global float *AEMIT,  // 10  emissions                [batch*NFREQ]
                       __global float *LL,     // 11  A matrices               [NE*NE*batch] work space
                       __global float *XX      // 12  solution vectors         [NE*batch]
                      ) {
   int id   = get_global_id(0) ;
   int lid  = get_local_id(0) ;
   int wg   = get_group_id(0) ;
   if (id>=batch) return ;

   // change of L to double -> more memory used, no effect on the results
   __global float *L    = &(   LL[wg*LOCAL * (NE*NE-NE)/2] ) ;
   __global float *X    = &(   XX[id*NE] ) ;
   
   // We will process batch cells using GLOBAL workers.
   __global float *ABS  = &( AABS[id*NFREQ]) ;
   __global float *EMIT = &(AEMIT[id*NFREQ]) ;
   
   // We use lower diagonal matrix and the elements above the diagonal,
   //   diagonal values will be normalised to 1.0 and need not be stored.
   // Because each work iterm will read its own L[j,i] at the same time,
   //   store those in adjacent locations: W0L00, W1L00, ..., W0L01, W1, L02, ...
   // L has a size of (N*N-N)/2 elements, indexing   A[j,i] = L[(j*j-j)/2+i].
   // U contains values from above the diagonal, *before* rescaling of A, 
   //   these are the Tdown values,  U[j] = Tdown[j+1], so it is indexed using the row index
   //   These should fit in local memory.
   
# if (NFREQ>(2*NE))
   double TMP[NE+max(NE, NFREQ)NFREQ] ;
#else
   double TMP[2*NE] ;
#endif
   double s, m = 1.0 ;          // s and m need to be double
   double D[NE] ;               // needs to be double
   double *XL = (double*)TMP ;  // XL needs to be double -- re-use TMP which is large enough
   
   float  I ;
   // float  U[NE] ;              // U can remain float
   double U[NE] ;
      
   // U can be float
   // I can be float
   // D needs to be double !!!
   // XL needs to be double !!!

   
   for(int i=0; i<NFREQ; i++) {
      TMP[i] = AF[i]*ABS[i] ;
   }


   
#if 1
   // This is necessary, even for Gaussian elimination
   // it looks like there might be something wrong / unstable with the integration of the
   // last frequency bin... if the spectrum is increasing from the next to last to the 
   // last frequency
   // but this will also affect the spectra so that result is different from A2E ... add same
   // safeguard to A2E ???
   TMP[NFREQ-1] = min(TMP[NFREQ-1], 0.2*TMP[NFREQ-2]) ;  // was 0.1
   TMP[0]       = min(TMP[0],       0.2*TMP[1]) ;
#endif
   
         
   
   int iw_index = 0 ;   // Note: Iw must be a vector containing only the [L1,L2] elements
   for(int l=0; l<NE-1; l++) {
      for(int u=l+1; u<NE; u++) {
         I = 0.0f ;
         for(int i=L1[l*NE+u]; i<=L2[l*NE+u]; i++) {
            I +=  TMP[i] * Iw[iw_index] ;
            iw_index++ ;
         } 
         // this is necessary: max() --- clip even at -1e30 and result is garbage
         L[((u*u-u)/2+l)*LOCAL+lid] = max(0.0f, I) ;
      }
   }
   // TMP[NFREQ] not used any more -- we can reuse that for XL and U !!!
   
   
   // Cooling   [ i, i+1 ], above diagonal,  U[NE-1] not used
   // U[NE-1] = 0.0f ; // not used
   for(int j=1; j<NE; j++)  U[j-1] = Tdown[j] ;

   // Compute diagonal value = sum of other values on the column
   for(int i=0; i<NE; i++) {          // column
      s = (i>0) ? U[i-1] : 0.0 ;      // upper diagonal entry, if exists
      for(int j=i+1; j<NE; j++) {     // lower triangle
         s +=  L[((j*j-j)/2+i)*LOCAL+lid] ;
      }
      XL[i] = -s ;                    // diagonal value
   }
   
   
#if 0
   // Renormalise equations so that the diagonal == 1.0 
   //  any scaling != 1.0 results in bad results ?????????????????????????????????????????????????????
   //  .... and double the run time ??????????????????????????????????????????????????????????????????
   double z = 0.0 ;
   for(int j=0; j<NE; j++) {          // loop over equations, j = row
      z = -1.0/(XL[j]-1.0e-20) ;
      for(int i=0; i<j; i++) {
         L[((j*j-j)/2+i)*LOCAL+lid] *= z ;        // lower triangle part of the row
      }
      U[j] *=  z ;                    // U[j] is part of equation on the row j
      D[j]  =  XL[j]*z ;              // diagonal entry
   }
   // Diagonal is now implicitly one
#else
   // ok ... plateau  2e-18, emission knee 2e-18
   for(int i=0; i<NE; i++) D[i] = XL[i] ;
#endif
   
   // Elimination --- row j solved for X[j+1]
   m     = 0.0 ;
   XL[0] = 1.0 ; 
   for(int j=0; j<NE-1; j++) {
      s  = D[j]*XL[j] ;             // start with the diagonal term
      for(int i=0; i<j; i++) {      // sum over lower triangle
         s += L[((j*j-j)/2+i)*LOCAL+lid] * XL[i] ;
      }
      XL[j+1]  =  -s / U[j]  ;
      m        =   max(m, XL[j+1]) ;
#if 0
      // this even not needed ??
      if (m>1.0e10) {
         for(int i=0; i<=(j+1); i++) {
            XL[i]  /=  m ;
         }
         m = 1.0 ;
      }
#endif
#if 0
      if (id==0) {
         if ((j>70)&&(XL[j+1]>1.0e-19)) {
            printf("%12.4e -> %12.4e  -%12.4e/%12.4e\n", XL[j], XL[j+1], s, U[j]) ;
         }
      }
#endif
   }
   
   
   
   // Normalisation, sum(XL)==1.0
   s = 0.0 ;
   for(int i=0; i<NE; i++)   s     +=  XL[i] ;
#if 0
   for(int i=0; i<NE; i++)   XL[i]  =  clamp(XL[i]/s, 1.0e-32, 1.0e32) ;
#else
   for(int i=0; i<NE; i++)  {
      m     =  XL[i]/s ;
      XL[i] = (m>1.0e-19) ? m : 1.0e-30 ;
   }
#endif

   
   
   // Calculate emission -- cumulative over different sizes
   for(int ifreq=0; ifreq<NFREQ; ifreq++) {
      s = 0.0 ;
      for(int i=Ibeg[ifreq]; i<NE; i++)  s +=  EA[ifreq*NE+i] * XL[i] ;
      EMIT[ifreq] = s ; 
   }
   for(int i=0; i<NE; i++) X[i] = clamp(XL[i], 1.0e-32, 1.0e32) ;
   
}



#endif // unused routines above




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
      Ein += (A[i]*FREQ[i]+A[i-1]*FREQ[i-1]) * ((FREQ[i]-FREQ[i-1])*3.3130348e-27f) ;
   }
   // Table lookup
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
      EMIT[id*NFREQ+ifreq] = (2.79639459e-20f*FACTOR)*KABS[ifreq]*(f*f/(exp(4.7995074e-11f*f/TP)-1.0f)) ;
      // emission must be still scaled with GRAIN_DENSITY*S_FRAC = actual number of grains per H
   }
}





__kernel void DoSolve4X(const      int     batch,   //  0  cells per call
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
                        __global   float   *LL,     // 11  A matrices               [NE*NE*batch] work space
                        __global   float   *XX      // 12  solution vectors         [NE*batch]
                       ) {
   // This is the stable version!
   int id  = get_global_id(0) ;
   int lid = get_local_id(0) ;
   int wg  = get_group_id(0) ;
   if (id>=batch) return ;
   __global float  *L    = &(LL[wg*LOCAL*(NE*NE-NE)/2]) ; // allocation is multiple of LOCAL
   __global float  *ABS  = &(AABS[id*NFREQ]) ;
   __global float  *EMIT = &(AEMIT[id*NFREQ]) ;   
   float XL[NE]  ;   
   float I ;
   long int  iw_index = 0 ;
   int j, u, l ;
   // DoSolveX is finally more stable... no high-E tail in pdf:s
   //   run time CPU 25s, GPU 20s .... compared to DoSolve4()
#define IND(a,b) (((a*a-a)/2+b)*LOCAL+lid)
   // Initialise
   // Heating   
#if 0 // local AA[] not used
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
# if 1
   for(int i=0; i<NFREQ; i++)     AA[lid*NFREQ+i] = ABS[i]*AF[i] ;
# else
   // *** THIS WAS NECESSARY TO AVOID SMALL NUMBER OF CELLS WITH HUGE EMISSION ***
   // *** NOW DONE IN A2E_pyCL.py ?? ***
   for(int i=0; i<NFREQ-1; i++)   AA[lid*NFREQ+i] = ABS[i]*AF[i] ;
   AA[lid*NFREQ+NFREQ-1]   =  AF[NFREQ-1]*min(ABS[NFREQ-2]*0.2f, ABS[NFREQ-1]) ;
# endif
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
   // row NE-2 is still also the original... except for diagonal that we can skip
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

   
   for(int i=0; i<NE; i++)  XX[id*NE+i] = clamp(XL[i], 1.0e-35f, 10.0f) ;
   
   // Emission
   for(j=0; j<NFREQ; j++) {
      I = 0.0f ;
      for(int i=Ibeg[j]; i<NE; i++)  I +=  EA[j*NE+i] * XL[i] ;
      EMIT[j] = I ; 
   }   

#if 0
   printf("NFREQ %d X = %10.3e ... %10.3e   EMIT = %10.3e ... %10.3e\n", 
          NFREQ, I, XL[3], XL[10], EMIT[3], EMIT[10], EA[10*NE+3], EA[10*NE+10] ) ;
#endif
   
}


