// #define EPS 6.0e-5f  BAD !!!
#define EPS 4.0e-4f
#define UNROLL 1
#define PRECALCULATED_GAUSS 1
#define real float
#define H_K  4.799243348e-11

#define USE_ATOMICS 0



inline void AADD(volatile __global float *addr, float val)
{
   union{
      unsigned int u32;
      float        f32;
   } next, expected, current;
   current.f32    = *addr;
   do{
      expected.f32 = current.f32;
      next.f32     = expected.f32 + val;
      current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr,
                                     expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
}


// solver assumes row order
#define IDX(a,b)  ((a)*LEVELS+(b))


int Doolittle_LU_Decomposition_with_Pivoting(__global real *A, __local ushort *pivot, int n) {
   // pivot = n elements per work item, can be >> 32kB per work group
   // Difficult to defactor and make efficient for a work group => separate problems for each work item !!!
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
      if (p_k[k]==0.0f) return -1 ;
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
                                     __local  ushort *pivot, __local real *x,  int n) {
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
      if (p_k[k]==0.0f) return -1;
      x[k]  /=  p_k[k] ;
   }
   return 0;
}



float get_C(const float tkin, const int NTKIN, __global float *TKIN, __global float *C) {
   // Interpolate C() for the correct temperature,   { TKIN[NTKIN],  C[NTKIN] } (tkin)
   int i ;
   for(i=1; i<NTKIN-1; i++) {    if (TKIN[i]>tkin) break ;   }
   // linear interpolation between elements i-1 and i
   float w = (tkin-TKIN[i-1]) / (TKIN[i]-TKIN[i-1]) ;  // distance from i-1 == weight for i
   return   w*C[i] + (1.0f-w)*C[i-1] ;
}




__kernel void Clear_X(__global  float *SIJ,    //  0  [NRAY, CELLS, TRANSITIONS]
                      __global  float *ESC,    //  1  [NRAY, CELLS, TRANSITIONS]
                      __global  float *NI,     //  2  [CELLS, LEVELS]
                      __global  float *NBNB,   //  3  [CELLS, TRANSITIONS]
                      __global  float *Aul,    //  4  [TRANSITIONS]
                      __global  float *GG,     //  5  [TRANSITIONS]
                      constant  int   *UPPER,  //  6  [TRANSITIONS]
                      constant  int   *LOWER,  //  7  [TRANSITIONS]
                      __global  float *FREQ    //  8  [TRANSITIONS]
                     ) {
   // Clear SIJ and ESC arrays [NRAY,CELLS,TRANSITIONS],  calculate NBNB values
   int id = get_global_id(0), u, l ;
   float tmp ;
#if (USE_ATOMICS==0)   
   for(int i=id; i<NRAY*CELLS*TRANSITIONS; i+=GLOBAL)  SIJ[i] = 0.0f ;
   for(int i=id; i<NRAY*CELLS*TRANSITIONS; i+=GLOBAL)  ESC[i] = 0.0f ;
#else
   for(int i=id; i<CELLS*TRANSITIONS; i+=GLOBAL)  SIJ[i] = 0.0f ;
   for(int i=id; i<CELLS*TRANSITIONS; i+=GLOBAL)  ESC[i] = 0.0f ;
#endif
   for(int t=0; t<TRANSITIONS; t++) {
      tmp  =  (Aul[t]*3.57603323e+19f)/(FREQ[t]*FREQ[t]) ;    u = UPPER[t] ;  l =  LOWER[t] ;
      for(int i=id; i<CELLS; i+=GLOBAL) {      
         NBNB[i*TRANSITIONS+t] =  tmp * (NI[i*LEVELS+l]*GG[t]-NI[i*LEVELS+u]) ;
      }
   }
}



__kernel void Update_X(
                       __global float4 *CLOUD,       //  0 [CELLS]: Vrad, Rc, dummy, sigma
                       constant int    *UPPER,       //  1 [TRANSITIONS]
                       constant float  *GAU,         //  2 gaussian profiles [GNO,CHANNELS]
                       __global int2   *LIM,         //  3 [CELLS] 
                       constant float  *Aul,         //  4 Einstein A  [TRANSITIONS]
                       constant float  *Ab,          //  5 (g_u/g_l)*B(upper->lower) [TRANSITIONS]
                       constant float  *GG,          //  6 [TRANSITIONS]
                       constant float  *GN,          //  7 Gauss norm. == C_LIGHT/(1e5*DV*freq) * GL  [TRANSITIONS]
                       __global float  *DIRWEI,      //  8 individual weights for the rays   [NRAY]
                       __global float  *VOLUME,      //  9 cell volume / cloud volume   [CELLS]
                       __global float  *STEP,        // 10 average path length [NRAY,CELLS] (GL)
                       __global float  *APL,         // 11 average path length [CELLS] (GL)
                       constant float  *BG,          // 12 background value (photons per ray)  [TRANSITIONS]
                       __global float  *IP,          // 13 impact parameter [NRAY] [GL]
                       __global float  *NI,          // 14 [CELLS, LEVELS]
                       __global float  *NBNB,        // 15 [CELLS, TRANSITIONS]
                       __global float  *NTRUE_ALL,   // 16 [NRAY, TRANSITIONS, CHANNELS]
                       __global float  *GSIJ,        // 17 [NRAY, CELLS, TRANSITIONS]
                       __global float  *GESC         // 18 [NRAY, CELLS, TRANSITIONS]
                      )  {
   // Follow one ray, update SIJ and ESC counters
   // One work item per ray => each work item updates separate entries in SIJ and ESC arrays
   int   id  = get_global_id(0) ;  // id = ray index
   if   (id>=NRAY) return ;   
   int   lid = get_local_id(0) ;   // index within the local work group
   float weight, dx, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;
   float sum_delta_true, all_escaped, nu ;
   int   row, shift, INDEX, c1, c2 ;
   float ip = IP[id] ;             // impact parameter [GL]
   __global float *ntrue, *NTRUE = &NTRUE_ALL[id*TRANSITIONS*CHANNELS] ;  // [NRAY, TRANSITIONS, CHANNELS]
   constant float *profile ;
   for(int t=0; t<TRANSITIONS; t++) {
      for(int i=0; i<CHANNELS; i++)   NTRUE[t*CHANNELS+i]  =   BG[t] * DIRWEI[id]  ;
   }
#if (USE_ATOMICS==0)
   __global float* SIJ = &(GSIJ[id*CELLS*TRANSITIONS]) ; // SIJ[CELLS,TRANSITIONS]
   __global float* ESC = &(GESC[id*CELLS*TRANSITIONS]) ;
#else
   __global float* SIJ = GSIJ ; // SIJ[CELLS,TRANSITIONS]
   __global float* ESC = GESC ;
#endif
   INDEX = CELLS-1 ;    // always starts with the outermost ray
   int dstep = -1 ;     // we first go inwards (dstep<0), then outwards until ray exits
   while(1) {
      dx        =  STEP[id*CELLS+INDEX] ;  // [GL]
      weight    =  DIRWEI[id]*(dx/APL[INDEX])*VOLUME[INDEX] ;  // VOLUME == fraction of cloud volume
      doppler   =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(ip/CLOUD[INDEX].y, 2.0f)) ) ;
      shift     =  round(doppler/WIDTH) ;
      profile   =  &GAU[INDEX*CHANNELS] ;
#if 0
      c1        =  clamp(shift, 0, CHANNELS) ;
      c2        =  clamp(CHANNELS+shift, 0, CHANNELS) ;
#else
      c1        =  clamp(LIM[INDEX].x+shift, 0, CHANNELS) ;
      c2        =  clamp(LIM[INDEX].y+shift, 0, CHANNELS) ;
#endif
      for(int tran=0; tran<TRANSITIONS; tran++) {
         ntrue     =  &(NTRUE[tran*CHANNELS]) ;      // vector for the current transition
         nu        =     NI[INDEX*LEVELS+UPPER[tran]] ;
         nb_nb     =   NBNB[INDEX*TRANSITIONS+tran] ;
         if (fabs(nb_nb)<1.0e-37f) nb_nb=1.0e-37f ; 
         tmp_tau        =  dx*nb_nb*GN[tran] ;         
         tmp_emit       =  weight * nu*Aul[tran] / tmp_tau ;         
         sum_delta_true =  all_escaped = 0.0f ;         
         for(int ii=c1; ii<c2; ii++)  {
#if 0
            factor  =  1.0f-exp(-tmp_tau*profile[ii-shift]) ;
#else
            factor  =  tmp_tau*profile[ii-shift] ;
            factor  =  (fabs(factor)>0.001f) ?  (1.0f-exp(-factor)) : (factor-0.5f*factor*factor) ;
#endif
            escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
            absorbed         =  ntrue[ii]*factor ;   // incoming photons that are absorbed
            ntrue[ii]       +=  escape-absorbed ;
            sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
            all_escaped     +=  escape ;             // sum of escaping photons over the profile
         }  // loop over channels
         // factor  = VOLUME[INDEX] ; // division by volume now already here ????
#if (USE_ATOMICS==0)
         SIJ[INDEX*TRANSITIONS+tran] += Ab[tran] * sum_delta_true / nb_nb  ;
         ESC[INDEX*TRANSITIONS+tran] += all_escaped  ;
#else
         AADD((__global float*)(SIJ+INDEX*TRANSITIONS+tran),  Ab[tran]*sum_delta_true/nb_nb) ;
         AADD((__global float*)(ESC+INDEX*TRANSITIONS+tran),  all_escaped)  ;
#endif
      } // for tran
      INDEX += dstep ;
      if (INDEX>=CELLS) break ;
      if (INDEX<0) {             // we went through the centre cell, turn outwards
         INDEX = 1 ;  dstep = 1 ;
      } else {
         if (STEP[id*CELLS+INDEX]<=0.0f) { // ray does not go that far in, turn outwards
            INDEX += 2 ;  dstep  = 1 ;
         }
      }
      if (INDEX>=CELLS) break ;
   } // while(1)
}



void __kernel Solve_X(constant float   *A,             //  0 MOL_A[TRANSITIONS]
                      constant int     *UPPER,         //  1 MOL_UL[TRANSITIONS,2]
                      constant int     *LOWER,         //  2 
                      constant float   *E,             //  3 MOL_E[LEVELS]
                      constant float   *G,             //  4 G[LEVELS]  ... not GG !!!
                      const int         PARTNERS,      //  5 number of collisional partners
                      const int         NTKIN,         //  6 number of Tkin for collisions -- same for all partners !!!???
                      const int         NCUL,          //  7 number of rows in C arrays
                      __global float   *MOL_TKIN,      //  8 MOL_TKIN[PARTNERS, NTKIN]
                      __global int     *CUL,           //  9 CUL[PARTNERS, NCUL, 2]
                      __global float   *C,             // 10 C[PARTNERS, NCUL, NTKIN]
                      __global float   *CABU,          // 11 CAB[PARTNERS]  --- no spatial variation yet
                      __global float   *RHO,           // 12 RHO[CELLS]
                      __global float   *TKIN,          // 13 TKIN[CELLS]
                      __global float   *ABU,           // 14 ABU[CELLS]
                      __global float   *NI,            // 15 NI[CELLS, LEVELS]
                      __global float   *GSIJ,          // 16 SIJ[NRAY, CELLS, TRANSITIONS]
                      __global float   *GESC,          // 17 ESC[NRAY, CELLS, TRANSITIONS]
                      __global float   *WRK,           // 18 WRK[GLOBAL*LEVELS*(LEVELS+1)]
                      __global float   *PC,            // 19 PATH_CORRECTION
                      __global float   *ERR,           // 20
                      __global float   *VOLUME         // 21
                     ) {
   const int id  = get_global_id(0) ;
   if (id>=CELLS) return ;  //  id ~ cell
   const int lid = get_local_id(0) ;
   __global float  *MATRIX = &WRK[id*LEVELS*(LEVELS+1)] ;
   __global float  *VECTOR = &WRK[id*LEVELS*(LEVELS+1)+LEVELS*LEVELS] ;
   __local  ushort  P[LOCAL*LEVELS] ;       // 64 workers x 100 levels x 2B  = 12.5 kB
   __local  ushort *pivot = &P[lid*LEVELS] ;
   float  err=0.0f, x ;
   double tmp ;
   int  u, l ;   
   
   __local float LV[LOCAL*LEVELS] ;   
   __local float *V   = &(  LV[lid*LEVELS]) ;
   
   // first sum SIJ and ESC from the global arrays
#if (USE_ATOMICS==0)
   __local float  LSIJ[LOCAL*TRANSITIONS], LESC[LOCAL*TRANSITIONS] ;
   __local float *SIJ = &(LSIJ[lid*TRANSITIONS]) ;
   __local float *ESC = &(LESC[lid*TRANSITIONS]) ;
   for(int t=0; t<TRANSITIONS; t++) {
      SIJ[t] = 0.0f ;  ESC[t] = 0.0f ;
      for(int iray=0; iray<NRAY; iray++) {
         SIJ[t] += GSIJ[iray*CELLS*TRANSITIONS+id*TRANSITIONS+t] ; // [TRANSITIONS]
         ESC[t] += GESC[iray*CELLS*TRANSITIONS+id*TRANSITIONS+t] ;
      }
      // it does not matter whether volume-division is in the simulation kernel or here?
      SIJ[t] /= VOLUME[id] ;   ESC[t] /= VOLUME[id] ;
   }   
#else
   __global float *SIJ =  &(GSIJ[id*TRANSITIONS]) ;
   __global float *ESC =  &(GESC[id*TRANSITIONS]) ;
   for(int t=0; t<TRANSITIONS; t++) {
      SIJ[t] /= VOLUME[id] ;   ESC[t] /= VOLUME[id] ;
   }   
#endif
   
   MATRIX[IDX(0,0)] = 0.0f ;   
   // Note -- we use CUL only for the first collisional partner, others must have (i,j) rows in same order!
   for(int i=1; i<LEVELS; i++) {  // loop explicitly only over downward transitions  M[j,i] ~   j <- i,   i>j
      MATRIX[IDX(i,i)] = 0.0f ; 
      for(int j=0; j<i; j++) {    // j<i == downward transitions
         for(u=0; u<NCUL; u++) {  // find the row  u  in collisional coefficient table  (i,j)
            if ((CUL[2*u]==i)&(CUL[2*u+1]==j)) break ;  // <-- CUL OF THE FIRST PARTNER ONLY !!!
         }
         tmp = 0.0f ;
         for(int p=0; p<PARTNERS; p++) { // get_C has the correct row from C, NTKIN element vector DOWNWARDS !!
            tmp += CABU[p]*get_C(TKIN[id], NTKIN, &MOL_TKIN[p*NTKIN], &C[p*NCUL*NTKIN + u*NTKIN]) ;
         }
         MATRIX[IDX(j,i)]  =  tmp * RHO[id] ;    //  IDX(j,i) = transition j <-- i  == downwards
         MATRIX[IDX(i,j)]  =  tmp * RHO[id] * (G[i]/G[j]) * exp(-H_K*(E[i]-E[j])/TKIN[id]) ; // upwards
      }
   }   
   
   for(int t=0; t<TRANSITIONS; t++) {  // modified Einstein A
      u = UPPER[t] ;  l = LOWER[t] ;
      MATRIX[IDX(l,u)]  +=  ESC[t] / (NI[id*LEVELS+u]) ;  // 1D has no division by VOLUME !!
   }   
   for(int t=0; t<TRANSITIONS; t++) {
      u = UPPER[t] ;   l = LOWER[t] ;
      MATRIX[IDX(u,l)]  +=  SIJ[t] * PC[id]  ;          // with PATH_CORRECTION
      MATRIX[IDX(l,u)]  +=  SIJ[t] * PC[id] * G[l]/G[u] ;
   }   
   for(int i=0; i<LEVELS-1; i++) {  // loop over columns
      tmp = 0.0f ;
      for(int j=0; j<LEVELS; j++)  tmp +=  MATRIX[IDX(j,i)] ;  // sum over column
      MATRIX[IDX(i,i)] = -tmp ;
   }
   for(int i=0; i<LEVELS; i++)  MATRIX[IDX(LEVELS-1, i)]  = -MATRIX[IDX(0,0)] ;
   for(int i=0; i<LEVELS; i++)  VECTOR[i] = 0.0f ;
   VECTOR[LEVELS-1]  =  -MATRIX[0]*RHO[id]*ABU[id] ;
   
   
#if 0
   if (id==0) {
      for(int j=0; j<LEVELS; j++) {
         for(int i=0; i<LEVELS; i++) printf(" %10.3e", MATRIX[IDX(j,i)]) ;
         printf("\n") ;
      }
      printf("\n") ;
      for(int i=0; i<LEVELS; i++) printf(" %10.3e", VECTOR[i]) ;
      printf("\n") ;
   }
#endif
   
   
   u  = Doolittle_LU_Decomposition_with_Pivoting(MATRIX, pivot,    LEVELS) ;
   u *= Doolittle_LU_with_Pivoting_Solve(MATRIX, VECTOR, pivot, V, LEVELS) ;
   
#if 0
   if (id==0) {
      for(int i=0; i<LEVELS; i++) printf(" %10.3e", V[i]) ;
      printf("\n") ;  
   }
#endif
   
   tmp = 0.0f ;   for(int i=0; i<LEVELS; i++)  tmp  +=  V[i] ;   
   for(int i=0; i<LEVELS; i++) {
      V[i]            =  V[i]*RHO[id]*ABU[id] / tmp ;  // B ro, X rw
      x               =  NI[id*LEVELS+i] ;
      err             =  max(err, fabs((V[i]-x)/x)) ;
      NI[id*LEVELS+i] =  V[i] ;
   }
   ERR[id] = err ;
   
}





__kernel void Spectra(
                      __global float4 *CLOUD,         //  0 [CELLS]: Vrad, Rc, ABUNDANCE, sigma
                      __global float  *GAU,           //  1 precalculated gaussian profiles
                      const float      GN,            //  2 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                      __global float2 *NI,            //  3 [CELLS] -> nu, nbnb
                      const float      BG,            //  4 background = B(Tbg)*I2T
                      const float      emis0,         //  5 h/(4pi)*freq*Aul*int2temp
                      __global float  *IP,            //  6 impact parameter [GL]
                      __global float  *STEP,          //  7 STEP[iray*CELLS+iray]  [GL]
                      __global float  *NTRUE_ARRAY,   //  8 NRAY*CHANNELS
                      __global float  *SUM_TAU_ARRAY, //  9 NRAY*CHANNELS
                      __global float  *RHO,           // 10 [CELLS] ... for column density calculation
                      __global float4 *COLDEN         // 11 [NRAY_SPE] = N, N_mol, tau, dummy
#if (WITH_CRT>0)
                      ,
                      const int       TRAN,           //  index of the current transition
                      __global float *CRT_TAU,        //  dust optical depth / GL
                      __global float *CRT_EMI         //  dust emission photons/c/channel/H
#endif
                     )
{
   // one work item per ray; the same rays as used in the actual calculation!
   int id = get_global_id(0) ;   // id == ray index
   if (id>=NRAY_SPE) return ;    // no more rays
   __global float *NTRUE   = &(NTRUE_ARRAY[id*CHANNELS]) ;
   __global float *SUM_TAU = &(SUM_TAU_ARRAY[id*CHANNELS]) ;
   int i ;
   for(int i=0; i<CHANNELS; i++) {
      NTRUE[i]   = 0.0f ;  SUM_TAU[i] = 0.0f ;
   }
   float tau, dtau, emissivity, doppler, nu, dx ;
   int row, shift ;
   __global float* profile ;
   float ip = IP[id] ; // impact parameter [GL] ;
   int INDEX=CELLS-1, dstep = -1 ;
   float colden=0.0f, coldenmol=0.0f, maxtau=0.0f ;
   
#if (WITH_CRT>0)
   float Ctau, Cemit, pro, distance=0.0f ;
#endif
   while (1) {
      dx         =  STEP[id*CELLS+INDEX] ;          // [GL]
      colden    +=  dx*RHO[INDEX] ;                 // needs to be scaled later by GL
      coldenmol +=  dx*RHO[INDEX]*CLOUD[INDEX].z ;
      nu         =  NI[INDEX].x ;     
      // tau        =  (fabs(NI[INDEX].y)<1.0e-29f) ? (1.0e-29f) :  (NI[INDEX].y) ;
      tau        =  NI[INDEX].y ;
      tau       *=  GN*GL*dx ;
      tau        =  clamp(tau, -2.0f, 1.0e10f) ;      
      // CLOUD[].y = effective radius,  CLOUD[].w = sigma, CLOUD[].x = Vrad
      doppler    =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(ip/CLOUD[INDEX].y,2.0f)) ) ;
#if 0
      row        =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      if (id==0) printf("doppler  %7.2f  --- tau %12.4e  --- row %3d\n", doppler, tau, row) ;
      profile    =  &GAU[row*CHANNELS] ;
#else
      profile    =  &GAU[INDEX*CHANNELS] ;  // GAU[CELLS, CHANNELS]
#endif
      shift      =  round(doppler/WIDTH) ;
      // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      emissivity =  emis0 * nu * dx * GL ;      
#if (WITH_CRT>0)
      distance      +=   dx*GL ;
      Ctau           =   CRT_TAU[INDEX*TRANSITIONS+TRAN]      * dx ;  // tau            * s
      Cemit          =   CRT_EMI[INDEX*TRANSITIONS+TRAN] * GL * dx ;  // phot/s/chn/cm3 * s
      // loop over all channels... to get continuum correct also at band edges
      for(i=0; i<CHANNELS; i++) {
         pro         =   profile[clamp(i-shift, 0, CHANNELS-1)] ;
         dtau        =   tau*pro + Ctau ;
         NTRUE[i]   +=  (emissivity*pro*GN + Cemit) * exp(-SUM_TAU[i]) * ((1.0f-exp(-dtau))/dtau) ;
         SUM_TAU[i] +=   dtau ;
      }     
#else
      for(i=max(0, shift); i<min(CHANNELS, CHANNELS+shift); i++) {         
         dtau        =   tau*profile[i-shift] ;
         dx          =   emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i]) ;
         dx         *=  (fabs(dtau)>1.0e-7f) ? ((1.0f-exp(-dtau)) / dtau) : (1.0f-0.5f*tau) ;
         NTRUE[i]   +=   dx ;
         SUM_TAU[i] +=   dtau  ;
      }
#endif
      // next cell
      INDEX += dstep ;
      if (INDEX>=CELLS) break ;
      if (INDEX<0) {             // we went through the centre cell, turn outwards
         INDEX = 1 ;  dstep = 1 ;
      } else {
         if (STEP[id*CELLS+INDEX]<=0.0f) { // ray does not go further in, turn outwards
            INDEX += 2 ; // we went through i but not i-1 => continue with i+1
            dstep  = 1 ;
         }
      }
      if (INDEX>=CELLS) break ;
   } // while(1)
   for (i=0; i<CHANNELS; i++) {
      NTRUE[i] +=  BG*(exp(-SUM_TAU[i])-1.0f) ;
      maxtau    =  max(maxtau, SUM_TAU[i]) ;
   }
#if 0
# if (WITH_CRT>0)
   if (id==0) {
      printf("TRAN %2d,  TAU_CON %12.5e,  TAU_TOT %12.5e,  TA %6.4f,   d = %12.5e pc\n", 
             TRAN, SUM_TAU[1], SUM_TAU[CHANNELS/2], NTRUE[CHANNELS/2], distance/3.0857e+18f) ;
   }
# endif
#endif
   COLDEN[id].x  = colden ;
   COLDEN[id].y  = coldenmol ;
   COLDEN[id].z  = maxtau ;
}







