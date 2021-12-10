
#define real float


// solver assumes row order
#define INDEX(a,b) (a*LEVELS+b)


int Doolittle_LU_Decomposition_with_Pivoting(__global real *A, __local ushort *pivot, int n) {
   // pivot = n elements per work item, can be >> 32kB per work group
   // Difficult to defactor and make efficient for a work group =>
   // entirely separate problems for each work item !!!
   int i, j, k ;
   __global real *p_k, *p_row, *p_col ;
   real maxi;
   // For each row and column, k = 0, ..., n-1,
   for (k=0, p_k=A; k<n; p_k+=n, k++) {
      // find the pivot row
      pivot[k] = k;
      maxi     = fabs( p_k[k] );
      for (j=k+1, p_row = p_k+n; j<n; j++, p_row += n) {
         if (maxi < fabs(p_row[k])) {
            maxi      =  fabs(p_row[k]) ;
            pivot[k]  =  j;
            p_col     =  p_row;
         }          
      }             
      // and if the pivot row differs from the current row, then interchange the two rows.
      if (pivot[k] != k) {
         for (j = 0; j < n; j++) {
            maxi       = *(p_k+j);
            *(p_k+j)   = *(p_col+j);
            *(p_col+j) = maxi ;
         }
      }
      // and if the matrix is singular, return error
      if (p_k[k]==0.0) return -1 ;
      // otherwise find the lower triangular matrix elements for column k
      for (i=k+1, p_row=p_k+n; i<n; p_row+=n, i++) {
         p_row[k]  /=  p_k[k] ;
      }
      // update remaining matrix
      for (i=k+1, p_row=p_k+n; i<n; p_row+=n, i++) {
         for (j=k+1; j<n; j++) {
            p_row[j] -= p_row[k] * p_k[j] ;
         }
      }
   }
   return 0 ;
}     



int Doolittle_LU_with_Pivoting_Solve(__global real *A,       __global real *B,
                                     __local  ushort *pivot, __global real *x, int n) {
   int i, k;
   __global real *p_k;
   real dum;
   // Solve Lx = B for x, where L is a lower triangular matrix with an implied 1 along the diagonal.
   for (k=0, p_k = A; k<n; p_k+=n, k++) {
      if (pivot[k]!=k) {
         dum = B[k];    B[k] = B[pivot[k]];    B[pivot[k]] = dum;
      }
      x[k] = B[k];
      for (i = 0; i < k; i++) x[k] -= x[i] * p_k[i] ;
   }
   // Solve the linear equation Ux = y, where 
   // y is the solution obtained above of Lx = B and U is an upper triangular matrix.
   for(k=n-1, p_k=A+n*(n-1); k>=0; k--, p_k-=n) {
      if (pivot[k]!=k) {
         dum = B[k];    B[k] = B[pivot[k]];    B[pivot[k]] = dum;
      }
      for (i=k+1; i<n; i++) x[k] -= x[i] * p_k[i] ;
      if (p_k[k]==0.0) return -1;
      x[k]  /=  p_k[k] ;
   }
   return 0;
}


float get_C(const float tkin, const int NTKIN, __global float *TKIN, __global float *C) {
   // Interpolate C() for the correct temperature
   for(i=1; i<NTKIN-1; i++) {
      if (TKIN[i]>tkin) break ;
   }
   // linear interpolation between elements i-1 and i
   float w = (tkin-TKIN[i-1])/(TKIN[i]-TKIN[i-1]) ;
   return   w*C[i-1] + (1.0f-w)*C[i] ;
}


__kernel SolveCL(const int         BATCH,         //  0 number of cells per kernel call
                 const float       VOLUME,        //  1 cell volume
                 __global float3  *MOL_AUL,       //  2 MOL_AUL[TRANSITIONS] =  A, U, L  (radiative!)
                 __global float   *MOL_EG,        //  3 MOL_EG[LEVELS]       =  E, G
                 const int         PARTNERS,      //  4 number of collisional partners
                 const int         NTKIN,         //  5 number of Tkin for collisions -- same for all partners !!!???
                 const int         NCUL,          //  6 number of rows in C arrays
                 __global float   *MOL_TKIN,      //  7 MOL_TKIN[PARTNERS, NTKIN]
                 __global int     *MOL_CUL,       //  8 MOL_CUL[PARTNERS, NCUL, 2]
                 __global float   *MOL_C,         //  9 MOL_C[PARTNERS, NCUL, NTKIN]
                 __global float   *MOL_CABU,      // 10 MOL_CAB[PARTNERS]  --- no spatial variation yet
                 __global float   *RHO,           // 11 RHO[BATCH]
                 __global float   *TKIN,          // 12 TKIN[BATCH]
                 __global float   *ABU,           // 13 ABU[BATCH]
                 __global float   *NI,            // 14 NI[BATCH,  LEVELS]   ---- PL_buf on host !!! READ-WRITE !!
                 __global float   *SIJ,           // 15 SIJ[BATCH,TRANSITIONS]
                 __global float   *ESC,           // 16 ESC[BATCH,TRANSITIONS]
                 __global float   *RES,           // 17 RES[BATCH, LEVELS]
                 __global float   *WRK            // 18 WRK[BATCH*LEVELS*(LEVELS+1)]
                ) {
   const int id = get_global_id(0) ;
   if (id>=BATCH) return ;   
   __global float *MATRIX = &WRK[id*LEVELS*(LEVELS+1)] ;
   __global float *VECTOR = &WRK[id*LEVELS*(LEVELS+1)+LEVELS*LEVELS] ;
   __global float *B = &RES[id*LEVELS] ;
   __local  float  X[LOCAL*LEVELS] ;  //  < 64*100  = 25 kB, probably <= 32*50 = 6 kB
   __local  float  P[LOCAL*LEVELS] ;
   __global float *X = &NI[id*LEVELS] ;   
   __local  float *pivot = &P[id*LEVELS] ;   
   
   float tmp ;
   int row ;
   MATRIX[INDEX(0,0)] = 0.0f ;
   // Note -- we use CUL only for the first collisional partner, others must have (i,j) rows in same order!
   for(int i=1; i<LEVELS; i++) {  // loop explicitly only over downward transitions  M[j,i] ~ i -> j, i>j
      MATRIX[INDEX(i,i)] = 0.0f ; 
      for(int j=0; j<i; j++) {         
         for(row=0; row<NCUL; row++) {  // find the row in collisional coefficient table  (i,j)
            if ((MOL_CUL[2*row]==i)&(MOL_CUL[2*row+1]==j)) break ;
         }
         tmp = 0.0f ;
         for(int p=0; p<PARTNERS; p++) { // get_C has the correct row from C, NTKIN element vector DOWNWARDS !!
            tmp += CABU[p]*get_C(i, j, TKIN[id], &MOL_TKIN[p*NTKIN], &MOL_C[(p*NCUL*NTKIN+row*NTKIN]) ;
                                }
            MATRIX[INDEX(j,i)] = tmp*RHO[id] ;    //  INDEX(j,i) = transition j <-- i
            // the corresponding element for UPWARD transition  j -> i
            tmp  *=  (MOL_G[j]/MOL_G[i]) *  exp(-H_K*(MOL_E[j]-MOL_E[i])/TKIN[id]) ;
            MATRIX[INDEX(i,j)] = tmp*RHO[id] ;    //  INDEX(j,i) = transition j <-- i
         }
      }   
   }
#if (WITH_ALI>0)
   for(int t=0; t<TRANSITIONS; t++)  // ....NI_ARRAY[icell, upper]
     MATRIX[INDEX(l,u)] += ESC[id*TRANSITIONS+t] / (VOLUME*NI[id*LEVELS+UL[2*t]]) ;
#else
   for(int t=0; t<TRANSITIONS; t++)
     MATRIX[INDEX[UL[2*t+1], UL[2*t]]] += MOL_A[t] ;  // MATRIX[l,u]
#endif   
   // NI is no longer used so we can reuse it to store vector X below
   for(int t=0; t<TRANSITIONS; t++) {
      u = MOL_UL[2*t] ;  l = MOL_UL[2*t+1] ;
      MATRIX[INDEX(u,l)]  +=  SIJ[id*TRANSITIONS+t] ;
      MATRIX[INDEX(l,u)]  +=  SIJ[id*TRANSITIONS+t] / MOL_GG[t] ;
   }   
   for(int i=0; i<LEVELS-1; i++) {  // loop over columns
      tmp = 0.0f ;
      for(int j=0; j<LEVELS; j++)  tmp +=  MATRIX[INDEX(j,i)] ;  // sum over column
      MATRIX[INDEX(u,u)] = -tmp ;
   }
   for(int i=0; i<LEVELS; i++)  MATRIX[INDEX(LEVELS-1,i)] = -MATRIX[0] ;
   for(int i=0; i<LEVELS; i++)  VECTOR[i] = 0.0f ;
   VECTOR[LEVELS-1] = -MATRIX[0]*RHO[id]*ABU[id] ;         
   ok  = Doolittle_LU_Decomposition_with_Pivoting(MATRIX, pivot, LEVELS) ;
   ok *= Doolittle_LU_with_Pivoting_Solve(MATRIX, VECTOR, pivot, X, LEVELS) ;   
   tmp = 0.0f ;
   for(int i=0; i<LEVELS; i++)  tmp  +=  X[i] ;
   for(int i=0; i<LEVELS; i++)  B[i]  =  X[i]*RHO[id]*ABU[id] / tmp ;  // B ro, X rw
}

















