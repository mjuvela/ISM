

void solve_cramer3(float *A, float *B, float *DET) {
   // Given A(3,3) and B(3), return X(3) such that A*X=B.
   // If DET==0 A is singular, and X is meaningless.
   // REAL DET,A(3,3),AINV(3,3),B(3),X(3)
   float a, b, c ;
   float AINV[9] ;
   AINV[0+3*0] = A[1+3*1]*A[2+3*2]-A[1+3*2]*A[2+3*1] ;
   AINV[0+3*1] = A[2+3*1]*A[0+3*2]-A[2+3*2]*A[0+3*1] ;
   AINV[0+3*2] = A[0+3*1]*A[1+3*2]-A[0+3*2]*A[1+3*1] ;
   AINV[1+3*0] = A[1+3*2]*A[2+3*0]-A[1+3*0]*A[2+3*2] ;
   AINV[1+3*1] = A[2+3*2]*A[0+3*0]-A[2+3*0]*A[0+3*2] ;
   AINV[1+3*2] = A[0+3*2]*A[1+3*0]-A[0+3*0]*A[1+3*2] ;
   AINV[2+3*0] = A[1+3*0]*A[2+3*1]-A[1+3*1]*A[2+3*0] ;
   AINV[2+3*1] = A[2+3*0]*A[0+3*1]-A[2+3*1]*A[0+3*0] ;
   AINV[2+3*2] = A[0+3*0]*A[1+3*1]-A[0+3*1]*A[1+3*0] ;
   *DET        = A[0+3*0]*AINV[0+3*0]+A[0+3*1]*AINV[1+3*0]+A[0+3*2]*AINV[2+3*0] ;
   if ((*DET)==0.0) return ;
   *DET = 1.0/(*DET) ;
   a = (*DET)*(AINV[0+3*0]*B[0]+AINV[0+3*1]*B[1]+AINV[0+3*2]*B[2]) ;
   b = (*DET)*(AINV[1+3*0]*B[0]+AINV[1+3*1]*B[1]+AINV[1+3*2]*B[2]) ;
   c = (*DET)*(AINV[2+3*0]*B[0]+AINV[2+3*1]*B[1]+AINV[2+3*2]*B[2]) ;
   B[0] = a ; B[1] = b ; B[2] = c ;
}




__kernel void nicer(
                    __global float   *K,        // COLOURS  (J/A, H/A, ...)
                    __global float   *REFCOL,   // COLOURS  (J-
                    __global float   *REFCOV,   // 
                    __global float   *MAG,
                    __global float   *dMAG,
                    __global float   *AV,
                    __global float   *dAV
                   )
{
   
   int  i, j, s  = get_global_id(0) ;   // one pixel
   if (s>=STARS) return ;
   float C[BANDS*BANDS] ;
   float av, b[10] ;
   for(i=0; i<COLOURS; i++) {
      C[COLOURS*BANDS+i] = -K[i] ;
      C[i*BANDS+COLOURS] = -K[i] ;
   }
   // right hand side
   for(i=0; i<BANDS; i++) b[i] = 0.0f ;
   b[COLOURS] = -1.0f ;   
   for(i=0; i<COLOURS; i++) {
      for(j=0; j<COLOURS; j++)  {
         C[i*BANDS+j] = 0.0f ;
      }
   }
   // diagonal
   for(i=0; i<COLOURS; i++) {
      C[i*BANDS+i] =  dMAG[s*BANDS+i]*dMAG[s*BANDS+i] + dMAG[s*BANDS+i+1]*dMAG[s*BANDS+i+1] ;
   }
   C[COLOURS+BANDS*COLOURS] = 0.0 ; //  ??????????????
   // first off-diagonals
   for(i=0; i<COLOURS-1; i++) {
      C[i*BANDS + i+1] = -dMAG[s*BANDS+i+1]*dMAG[s*BANDS+i+1] ;
      C[(i+1)*BANDS+i] = -dMAG[s*BANDS+i+1]*dMAG[s*BANDS+i+1] ;
   }
   // covariance from the reference field
   for(i=0; i<COLOURS; i++) {
      for(j=0;j<COLOURS; j++) {
         C[i*BANDS+j] += REFCOV[i*COLOURS+j*j] ;
      }
   }
   // C01 C02 C03       b0 b1  *   C01 C02  *  b0  == b0 b1   C0 C1   b0
   // C10 C11 C12  =>              C10 C11     b1             C1 C4   b1
   // C20 C21 C22
   float C0=C[0], C1=C[1], C4=C[4] ;  // matrix C gets overwritten
   float det ;
   solve_cramer3(C, b, &det) ;
   av    = 0.0 ;
   for(i=0; i<COLOURS; i++) av   += ((MAG[s*BANDS+i]-MAG[s*BANDS+(i+1)])-REFCOL[i]) * b[i] ;
   AV[s] = av ;
   // dAV =  bb*C[0:colours,0:colours]*nn
   dAV[s] = sqrt(b[0]*b[0]*C0 + b[1]*b[1]*C4 + 2.0f*b[0]*b[1]*C1) ;
}



__kernel void smooth(
                     __global float  *RA,         // coordinates of the stars
                     __global float  *DE,
                     __global float  *A,
                     __global float  *dA,
                     __global float  *SRA,        // coordinates of the pixels
                     __global float  *SDE,
                     __global float  *SA,
                     __global float  *dSA
                    )
{
   int  j, id  = get_global_id(0) ;   // index of smoothed value, single pixel
   if (id>=NPIX) return ;
   // calculate weighted average with sigma-clipping
   float ra=SRA[id], de=SDE[id] ;     // centre of the beam
   float cosy = cos(de) ;             // we can use the plane approximation for distances
   float w, dx, dy, weight, sum, s2, uw, d2, ave, std, count ;
   float K = 4.0f*log(2.0f)/(FWHM*FWHM) ;        // radian^-2
   const float LIMIT2 = 9.0f*FWHM*FWHM ;  // ignore stars further than sqrt(LIMIT2)
   //  RA should be on the same round... not 0.1 and 359.9 degrees, for example!!
   weight = sum = s2 = uw = count = 0.0f ;
   for(j=0; j<STARS; j++) {
      dx      =  cosy* (RA[j]-ra) ;
      dy      =        (DE[j]-de) ;
      d2      =  dx*dx + dy*dy ;
      if (d2<LIMIT2) {
         w       =  exp(-K*d2) / (dA[j]*dA[j]) ;
         weight +=  w ;
         sum    +=  w*A[j] ;       // weighted sum
         uw     +=    A[j] ;       // unweighted sum
         s2     +=    A[j]*A[j] ;  // for unweighted standard deviation
         count  +=  1.0f ;
      }
   }
   ave    = sum/weight ;  // weighted average
   std    = sqrt(s2/count - (uw/count)*(uw/count)) ; // scatter between stars, unweighted std
   // Again, this time with sigma clipping
   weight = sum = s2 = uw = count = 0.0f ;
   for(j=0; j<STARS; j++) {
      dx      =  cosy* (RA[j]-ra) ;
      dy      =         DE[j]-de  ;
      d2      =  dx*dx+dy*dy ;
      if ((d2<LIMIT2) && (A[j]>(ave-CLIP_DOWN*std)) && (A[j]<(ave+CLIP_UP*std)) ) {
         w       =  exp(-K*d2) /  (dA[j]*dA[j]) ;
         weight +=  w ;
         sum    +=  w*A[j] ;
         uw     +=    A[j] ;
         count  +=  1.0f ;
#if (TRUE_ERROR>0) // -----   error estimates based on the scatter between stars
# ifdef UNWEIGHTED                    // error estimates (std) without weighting
         s2     +=    A[j]*A[j] ; 
# else                                // weighted, based on selection below !!!!!!!
         s2     +=  w*A[j]*A[j] ;  
# endif
#else              // ------  not true but formal error of the mean
# ifdef UNWEIGHTED
         s2     +=  dA[j]*dA[j] ;      // unweighted
# else
         s2     +=  w*w*dA[j]*dA[j] ;  // weighted
# endif
#endif // TRUE_ERROR or not
      }
   }
   if (count>1.0) {
      SA[id]   = sum/weight ;      // weighted average -- the value itself always weighted average
#if (TRUE_ERROR>0)
# ifdef UNWEIGHTED                 // unweighted true (?) error estimate
      dSA[id]  = sqrt(s2/count  - (uw/count)  *(uw/count  ))  / sqrt(count)  ;
# else                             // weighted true error estimate
      dSA[id]  = sqrt(s2/weight - (sum/weight)*(sum/weight))  / sqrt(weight) ;
# endif
#else                              // not true but formal error ==================================
# ifdef UNWEIGHTED                 // unweighted
      dSA[id] = sqrt(s2)/count ;
# else
      dSA[id] = sqrt(s2)/weight ;    // weighted
# endif
#endif  // if-else-endif TRUE_ERROR
   } else {                        // count <= 1 
      if (weight>0.0) {
         SA[id] = sum/weight ;   dSA[id] = 999.0 ;
      } else {
         SA[id] = 0.0 ;          dSA[id] = 999.0 ;
      }
   }
}





__kernel void smoothX(
                      __global float  *RA,         // coordinates of the stars
                      __global float  *DE,         //   in radians
                      __global float  *A,          // Av estimates of the stars
                      __global float  *dA,         // Av uncertainty
                      __global float  *SRA,        // coordinates of the pixels
                      __global float  *SDE,        //   in radians
                      __global float  *SA,         // result smoothed Av values
                      __global float  *dSA,        // error estimates of smoothed values
                      __global float  *AGRID,      // tau averaged for grid position
                      __global float  *ASTAR       // tau at the star location (hi-res)
                     )
{
   int  j, id  = get_global_id(0) ;   // index of smoothed value, single pixel
   if (id>=NPIX) return ;
   // calculate weighted average with sigma-clipping
   float ra=SRA[id], de=SDE[id] ;     // centre of the beam
   float cosy = cos(de) ;             // we can use the plane approximation for distances
   float w, dx, dy, weight, sum, s2, uw, d2, ave, std, count ;
   float K = 4.0f*log(2.0f)/(FWHM*FWHM) ;        // radian^-2
   float AA, dAA, A0 = AGRID[id] ;
   const float LIMIT2 = 9.0f*FWHM*FWHM ;  // ignore stars further than sqrt(LIMIT2)
   //  RA should be on the same round... not 0.1 and 359.9 degrees, for example!!
   weight = sum = s2 = uw = count = 0.0f ;
   for(j=0; j<STARS; j++) {
      dx      =  cosy* (RA[j]-ra) ;
      dy      =        (DE[j]-de) ;
      d2      =  dx*dx + dy*dy ;
      if (d2<LIMIT2) {
         w       =  exp(-K*d2) / (dA[j]*dA[j]) ;
         weight +=  w ;
         if ((A0>0.0f)&&(ASTAR[j]>0.0f)) {
            AA      =    A[j] * A0 / ASTAR[j] ;   // <A> estimate based on a single star
         } else {
            AA      =    A[j] ;   // <A> estimate based on a single star
         }
         sum    +=  w*AA ;       // weighted sum
         uw     +=    AA ;       // unweighted sum
         s2     +=    AA*AA ;    // for unweighted standard deviation
         count  +=    1.0f ;
      }
   }
   ave    = sum/weight ;  // weighted average
   std    = sqrt(s2/count - (uw/count)*(uw/count)) ; // scatter between stars, unweighted std
   // Again, this time with sigma clipping
   weight = sum = s2 = uw = count = 0.0f ;
   for(j=0; j<STARS; j++) {
      dx      =  cosy* (RA[j]-ra) ;
      dy      =         DE[j]-de  ;
      d2      =  dx*dx+dy*dy ;
      if ((A0>0.0f)&&(ASTAR[j]>0.0f)) {
         AA      =  A[j] * A0 / ASTAR[j] ;
      } else {
         AA      =  A[j]  ;
      }
      if ((d2<LIMIT2) && (AA>(ave-CLIP_DOWN*std)) && (AA<(ave+CLIP_UP*std)) ) {
         if ((A0>0.0f)&&(ASTAR[j]>0.0f)) {
            dAA     =  dA[j] * A0 / ASTAR[j] ;
         } else {
            dAA     =  dA[j] ;
         }
         w       =  exp(-K*d2) /  (dAA*dAA) ;
         weight +=  w ;
         sum    +=  w*AA ;
         uw     +=    AA ;
         count  +=  1.0f ;
#if (TRUE_ERROR>0) // -----   error estimates based on the scatter between stars
# ifdef UNWEIGHTED                    // error estimates (std) without weighting
         s2     +=    AA*AA ; 
# else                                // weighted, based on selection below !!!!!!!
         s2     +=  w*AA*AA ;  
# endif
#else              // ------  not true but formal error of the mean
# ifdef UNWEIGHTED
         s2     +=  dAA*dAA ;      // unweighted
# else
         s2     +=  w*w*dAA*dAA ;  // weighted
# endif
#endif // TRUE_ERROR or not
      }
   }
   if (count>1.0) {
      SA[id]   = sum/weight ;      // weighted average -- the value itself always weighted average
#if (TRUE_ERROR>0)
# ifdef UNWEIGHTED                 // unweighted true (?) error estimate
      dSA[id]  = sqrt(s2/count  - (uw/count)  *(uw/count  ))  / sqrt(count)  ;
# else                             // weighted true error estimate
      dSA[id]  = sqrt(s2/weight - (sum/weight)*(sum/weight))  / sqrt(weight) ;
# endif
#else                              // not true but formal error ==================================
# ifdef UNWEIGHTED                 // unweighted
      dSA[id] = sqrt(s2)/count ;
# else
      dSA[id] = sqrt(s2)/weight ;    // weighted
# endif
#endif  // if-else-endif TRUE_ERROR
   } else {                        // count <= 1 
      if (weight>0.0) {
         SA[id] = sum/weight ;   dSA[id] = 999.0 ;
      } else {
         SA[id] = 0.0 ;          dSA[id] = 999.0 ;
      }
   }
}

