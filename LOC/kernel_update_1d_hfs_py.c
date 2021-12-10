#define PRECALCULATED_GAUSS 1

#define EPS 4.0e-4f
#define UNROLL 1
#define real float
#define H_K  4.799243348e-11



__kernel void Clear(__global float2 *ARES) {  // [CELLS*NRAY] SIJ, ESC
   // Clear the ARES array before simulation of the current transition
   int id = get_global_id(0) ;  // id = ray index
   if (id>=NRAY) return ;
   for(int i=0; i<CELLS; i++) {
      ARES[id*CELLS+i].x = 0.0f ;      ARES[id*CELLS+i].y = 0.0f ;
   }
}


__kernel void Sum(__global float2 *ARES, __global float2 *RES, __global float *VOLUME) {
   // Each work item has updated entries in ARES[id*CELLS+INDEX], sum these to RES[CELLS]
   int  id = get_global_id(0) ;  // id = cell index
   if (id>=CELLS) return ;
   float sij=0.0f, esc=0.0f ;
   for(int i=0; i<NRAY; i++) {
      sij += ARES[i*CELLS+id].x ;    
      esc += ARES[i*CELLS+id].y ;
   }
   RES[id].x = sij  / VOLUME[id] ;    
   RES[id].y = esc  / VOLUME[id] ;
}



__kernel void Update(
                     __global float4 *CLOUD,      //  0 [CELLS]: Vrad, Rc, dummy, sigma
                     __global float  *GAU,        //  1 precalculated gaussian OBprofiles [TRANSITIONS*CELLS*CHANNELS]
                     const float      Aul,        //  2 Einstein A(upper->lower)
                     const float      A_b,        //  3 (g_u/g_l)*B(upper->lower)
                     const float      GN,         //  4 Gauss norm. == C_LIGHT/(1e5*DV*freq) * GL
                     __global float  *DIRWEI,     //  5 individual weights for the rays
                     __global float  *VOLUME,     //  6 cell volume / cloud volume
                     __global float  *STEP,       //  7 average path length [NRAY*CELLS] (GL)
                     constant float  *APL,        //  8 average path length [CELLS] (GL)
                     const float      BG,         //  9 background value (photons per ray)
                     __global float  *IP,         // 10 impact parameter [NRAY] [GL]
                     __global float2 *NI,         // 11 [CELLS]:  NI[upper] + NB_NB
                     __global float2 *ARES,       // 12 [CELLS*NRAY]:  SIJ, ESC
                     const int        TRAN,       // 13 transition
                     const int        NCHN,       // 14 channels used
                     __global float *NTRUE_ARRAY, // 15 NTRUE_ARRAY[NRAY*CHANNELS]
                     __global int2  *LIM          // 16 first and last significant channels
#if (WITH_CRT>0)
                     ,
                     constant float *CRT_TAU,     // 17 dust optical depth / GL
                     constant float *CRT_EMI      // 18 dust emission photons/c/channel/H
#endif
                    )  {
   // Follow one ray, update SIJ and ESC counters
   // One work item per ray => each work item must update a separate element of ARES
   // NCHN <= CHANNELS
   float weight, dx, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;
   float sum_delta_true, all_escaped, nu ;
   __global float* profile ;
   int   shift, INDEX ;
   int   id  = get_global_id(0) ;  // ray index
   int   lid = get_local_id(0) ;   // index within the local work group
   if  (id>=NRAY) return ;   
   float ip = IP[id] ;             // impact parameter [GL]
   
#if (LOWMEM>0)
   __global float *NTRUE = &NTRUE_ARRAY[id*CHANNELS] ;  // spectrum in the current ray
#else // insufficient resources...
   __local float  NTRUES[LOCAL*CHANNELS] ;
   __local float *NTRUE = &NTRUES[lid*CHANNELS] ;
#endif
   for(int i=0; i<NCHN; i++) NTRUE[i] = BG * DIRWEI[id]  ;  // limited number of channels   
   
#if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, sij, pro ;
#endif
   
   INDEX = CELLS-1 ;    // always starts with the outermost ray
   int dstep = -1 ;     // we first go inwards (dstep<0), then outwards until ray exits
   while(1) {
      dx        =  STEP[id*CELLS+INDEX] ;  // [GL]
      nu        =  NI[INDEX].x ;
      nb_nb     =  NI[INDEX].y ;
      // emitted photons divided between all passing rays ~ path length
      weight    =  DIRWEI[id]*(dx/APL[INDEX])*VOLUME[INDEX] ;  // VOLUME == fraction of cloud volume
      // CLOUD.x == radial velocity, CLOUD.y == effective radius
      doppler   =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(ip/CLOUD[INDEX].y, 2.0f)) ) ;
      if (fabs(nb_nb)<1.0e-37f) nb_nb=1.0e-37f ; 
      tmp_tau   =  dx*nb_nb*GN ;  // GN includes GL [cm]
      shift     =  round(doppler/WIDTH) ;
      profile   =  &GAU[TRAN*CELLS*CHANNELS+INDEX*CHANNELS] ; // use at most NCHN first channels
      sum_delta_true = all_escaped = 0.0f ;
      // loop only over the significant channels
      int i1 = LIM[TRAN*CELLS+INDEX].x ;
      int i2 = LIM[TRAN*CELLS+INDEX].y ;
      // printf("  0   %3d %3d  %3d\n", i1, i2, NCHN-1) ;
      //    channels [i1,i2] from profile  < CHANNELS
      //    into NTRUE[0, NCHN-1]
#if (WITH_CRT>0)
      sij = 0.0f ;
      // Dust optical depth and emission
      Ctau      =  dx     * CRT_TAU[INDEX*TRANSITIONS+TRAN] ;
      Cemit     =  weight * CRT_EMI[INDEX*TRANSITIONS+TRAN] ;      
      for(int ii=max(i1, shift); ii<min(NCHN, i2+shift); ii++)  {       // max NCHN channels
         // factor           =  1.0f-exp(-tmp_tau*profile[ii-shift]) ;         
         // escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
         // absorbed         =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         // NTRUE[ii]       +=  escape-absorbed ;
         // sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
         // all_escaped     +=  escape ;             // sum of escaping photons over the profile
         pro    =  profile[ii-shift] ;
         Ltau   =  tmp_tau*pro ;
         Ttau   =  Ctau + Ltau ;
         tt     =  (1.0f-exp(-Ttau)) / Ttau ;
         // tt     =  (fabs(Ttau)>1.0e-4) ? ((1.0f-exp(-Ttau))/Ttau) : (1.0f-0.5f*Ttau) ;
         ttt    =  (1.0f-tt)/Ttau  ;
         // ttt    =  (fabs(Ttau)>1.0e-4) ? ((1.0f-tt)/Ttau) : (0.5f-Ttau/6.0f) ;
         // Line emission leaving the cell, GL in profile
         Lleave =  weight*nu*Aul*pro * tt ;
         // Dust emission leaving the cell 
         Dleave =  Cemit *                     tt ;
         // SIJ -- incoming photons
         sij   += A_b * pro*GN*dx * NTRUE[ii] * tt ;
         // SIJ -- dust emission
         sij   +=  A_b * pro*GN*dx* Cemit     * ttt ; // GN includes GL!
         // "Escaping" line photons = absorbed by dust or leave the cell
         // escape      +=  Lleave  +  weight*nu*Aul*profile[ii] * (Ctau/Ttau) * (1.0f-tt) ;
         all_escaped  +=  Lleave  +  weight*nu*Aul*pro * Ctau * ttt ;
         // Total change of photons in the package
         NTRUE[ii]     =  NTRUE[ii]*exp(-Ttau) + (Dleave + Lleave) ;
      }  // loop over channels
      // Update SIJ and ESC (note - there may be two updates, one incoming, one outgoing ray)
      ARES[id*CELLS+INDEX].x  += sij ;            // to be divided by VOLUME
      // Emission ~ path length dx but also weighted according to direction, works because <WEI>==1.0
      ARES[id*CELLS+INDEX].y  += all_escaped ;    // to be divided by VOLUME
#else
      tmp_emit  =  weight * nu*Aul / tmp_tau ;
      // Normal calculation without dust absorption and emission
      for(int ii=max(i1, shift); ii<min(NCHN, i2+shift); ii++)  { // max NCHN channels
         factor           =  1.0f-exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed         =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped     +=  escape ;             // sum of escaping photons over the profile
      }  // loop over channels
      // Update SIJ and ESC (note - there may be two updates, one incoming, one outgoing ray)
      ARES[id*CELLS+INDEX].x  += A_b * sum_delta_true / nb_nb  ; // to be divided by VOLUME
      // Emission ~ path length dx but also weighted according to direction, works because <WEI>==1.0
      ARES[id*CELLS+INDEX].y  += all_escaped ;                   // to be divided by VOLUME
#endif
      
      // next cell
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



__kernel void Spectra(
                      __global float4 *CLOUD,         //  0 [CELLS]: Vrad, Rc, dummy, sigma
                      __global float  *GAU,           //  1 precalculated gaussian profiles
                      const float      GN,            //  2 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                      __global float2 *NI,            //  3 [CELLS]:  NI[upper] + NB_NB
                      const float      BG,            //  4 background intensity
                      const float      emis0,         //  5 h/(4pi)*freq*Aul*int2temp
                      __global float  *IP,            //  6 impact parameter [GL]
                      __global float  *STEP,          //  7 STEP[iray*CELLS+iray]  [GL]
                      __global float  *NTRUE_ARRAY,   //  8 NRAY*CHANNELS
                      __global float  *SUM_TAU_ARRAY, //  9 NRAY*CHANNELS
                      const int        TRAN,          // 10 transition
                      const int        NCHN,          // 11 number of actual channels
#if (WITH_CRT>0)
                      constant float *CRT_TAU,        // 12 CRT_TAU[CELLS, TRANSITIONS] in python storage order
                      constant float *CRT_EMI,        // 13 CRT_EMI[CELLS, TRANSITIONS]
#endif
                      const int       savetau
                     )
{
   // 2017-01-14  -- now with equidistant rays
   //  array STEP[iray*CELLS+INDEX] has been updated for NRAY_SPE impact parameters in IP[NRAY_SPE]
   //  ip = i*1.0/(NRAY_SPE-1.0)
   
   // one work item per ray; the rays are NOT the same as during the simulation
   // probably NRAY_SPE < NRAY, the change means that STEP[] has been recalculated
   int id = get_global_id(0) ;   // id == ray index
   if (id>=NRAY_SPE) return ;    // no more rays
   __global float *NTRUE   = &(NTRUE_ARRAY[id*NCHN]) ;    // was CHANNELS
   __global float *SUM_TAU = &(SUM_TAU_ARRAY[id*NCHN]) ;  // was CHANNELS
   int i ;
   for(int i=0; i<NCHN; i++) {
      NTRUE[i]   = 0.0f ;  SUM_TAU[i] = 0.0f ;
   }
   float tau, dtau, emissivity, doppler, nu, dx ;
#if (WITH_CRT>0)
   float pro, Ctau, Cemit ;
#endif
   int shift ;
   __global float* profile ;
   float ip = IP[id] ; // impact parameter [GL] ;
   int INDEX=CELLS-1, dstep = -1 ;
   while (1) {
      dx         =  STEP[id*CELLS+INDEX] ;          // [GL]
      nu         =  NI[INDEX].x ;     
      tau        =  (fabs(NI[INDEX].y)<1.0e-26f) ? (1.0e-26f) :  (NI[INDEX].y) ;
      tau       *=  GN*GL*dx ;
      tau        =  clamp(tau, -2.0f, 1.0e10f) ;      
      // CLOUD[].y = effective radius,  CLOUD[].w = sigma, CLOUD[].x = Vrad
      doppler    =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(ip/CLOUD[INDEX].y,2.0f)) ) ;
      profile    =  &GAU[TRAN*CELLS*CHANNELS+INDEX*CHANNELS] ; // use NCHN first elements
      // if NCHN<CHANNELS, move spectrum left
      shift      =  round(doppler/WIDTH) ;
      // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      emissivity =  emis0 * nu * dx * GL ;      
#if (WITH_CRT>0) // ---------------------------------------------------------
      Ctau        =  CRT_TAU[INDEX*TRANSITIONS+TRAN] * dx ;
      Cemit       =  CRT_EMI[INDEX*TRANSITIONS+TRAN] * GL * dx ;
# if (KILL_EMISSION<CELLS)
      if (INDEX>=KILL_EMISSION) Cemit = 0.0f ;
# endif
      for(i=0; i<NCHN; i++) {
         pro         =  profile[clamp(i-shift, 0, NCHN-1)] ;
         dtau        =  tau*pro + Ctau ;
# if (KILL_EMISSION<CELLS)
         dx          =  (INDEX>=KILL_EMISSION) ? (0.0f)  :  ((emissivity*pro*GN + Cemit) * exp(-SUM_TAU[i])) ;
# else
         dx          =  (emissivity*pro*GN + Cemit) * exp(-SUM_TAU[i]) ;
# endif
         // dx      *=   (fabs(dtau)>1.0e-4f) ? ((1.0f-exp(-dtau))/dtau) : (1.0f-0.5f*dtau) ;
         dx         *=   (1.0f-exp(-dtau)) / dtau ;
         NTRUE[i]   +=  dx ;
         SUM_TAU[i] +=  dtau  ;
      }
#else    // NOT CRT ---------------------------------------------------------
      for(i=max(0, shift); i<min(NCHN, NCHN+shift); i++) {         
         dtau       =  tau*profile[i-shift] ;
# if (KILL_EMISSION<CELLS)
         if (INDEX>=KILL_EMISSION) {
            dx = 0.0f ;
         } else {
            dx         =  emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i]) ;         
            dx        *=   (fabs(dtau)>1.0e-7f) ? ((1.0f-exp(-dtau)) / dtau) : (1.0f-0.5f*tau) ;
         }
# else
         dx         =  emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i]) ;         
         dx        *=   (fabs(dtau)>1.0e-7f) ? ((1.0f-exp(-dtau)) / dtau) : (1.0f-0.5f*tau) ;
# endif
         NTRUE[i]   +=  dx ;
         SUM_TAU[i] +=  dtau  ;
      }
#endif // -------------------------------------------------------------------
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
   if (savetau>0) {   
      for (i=0; i<NCHN; i++)  NTRUE[i]  =  SUM_TAU[i] ;
   } else {
      for (i=0; i<NCHN; i++)  NTRUE[i] -=  BG*(1.0f-exp(-SUM_TAU[i])) ;
   }
   
#if 0
   if (id==0)   printf("           id %3d  =>  TRAN = %d   => CONTINUUM OPTICAL DEPTH %.3e\n", id, TRAN, SUM_TAU[2]) ;
#endif
}






// solver assumes row order
#define IDX(a,b)  ((a)*LEVELS+(b))
// #define IDX(a,b) (a+LEVELS*b)


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
                                     __local  ushort *pivot, __global real *x,  int n) {
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
   //  { TKIN[NTKIN],  C[NTKIN] } (tkin)
   int i ;
   for(i=1; i<NTKIN-1; i++) {    if (TKIN[i]>tkin) break ;   }
   // linear interpolation between elements i-1 and i
   float w = (tkin-TKIN[i-1]) / (TKIN[i]-TKIN[i-1]) ;  // distance from i-1 == weight for i
   return   w*C[i] + (1.0f-w)*C[i-1] ;
}



void __kernel SolveCL(const int         BATCH,         //  0 number of cells per kernel call
                      __global float   *A,             //  1 MOL_A[TRANSITIONS]
                      __global int     *UL,            //  2 MOL_UL[TRANSITIONS,2]
                      __global float   *E,             //  3 MOL_E[LEVELS]
                      __global float   *G,             //  4 MOL_G[LEVELS]
                      const int         PARTNERS,      //  5 number of collisional partners
                      const int         NTKIN,         //  6 number of Tkin for collisions -- same for all partners !!!???
                      const int         NCUL,          //  7 number of rows in C arrays
                      __global float   *MOL_TKIN,      //  8 MOL_TKIN[PARTNERS, NTKIN]
                      __global int     *CUL,           //  9 CUL[PARTNERS, NCUL, 2]
                      __global float   *C,             // 10 C[PARTNERS, NCUL, NTKIN]
                      __global float   *CABU,          // 11 CAB[PARTNERS]  --- no spatial variation yet
                      __global float   *RHO,           // 12 RHO[BATCH]
                      __global float   *TKIN,          // 13 TKIN[BATCH]
                      __global float   *ABU,           // 14 ABU[BATCH]
                      __global float   *NI,            // 15 NI[BATCH,  LEVELS]   ---- PL_buf on host !!! READ-WRITE !!
                      __global float   *SIJ,           // 16 SIJ[BATCH,TRANSITIONS]
                      __global float   *ESC,           // 17 ESC[BATCH,TRANSITIONS]
                      __global float   *RES,           // 18 RES[BATCH, LEVELS]
                      __global float   *WRK,           // 19 WRK[BATCH*LEVELS*(LEVELS+1)]
                      __global float   *PC             // 20 PATH_CORRECTION
                     ) {
   const int id  = get_global_id(0) ;
   const int lid = get_local_id(0) ;
   if (id>=BATCH) return ;   
   __global float  *MATRIX = &WRK[id*LEVELS*(LEVELS+1)] ;
   __global float  *VECTOR = &WRK[id*LEVELS*(LEVELS+1)+LEVELS*LEVELS] ;
   __global float  *B = &RES[id*LEVELS] ;   // output vector
   __global float  *V =  &NI[id*LEVELS] ;   // temporary storage for solved NI vector
   __local  ushort  P[LOCAL*LEVELS] ;       // 64 workers x 100 levels x 2B  = 12.5 kB
   __local  ushort *pivot = &P[lid*LEVELS] ;
   
   float tmp ;
   int  u, l ;
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
         MATRIX[IDX(j,i)] = tmp*RHO[id] ;    //  IDX(j,i) = transition j <-- i  == downwards
         // the corresponding element for UPWARD transition  j -> i
         tmp  *=  (G[i]/G[j]) *  exp(-H_K*(E[i]-E[j])/TKIN[id]) ;
         MATRIX[IDX(i,j)] = tmp*RHO[id] ;    //  IDX(j,i) = transition j <-- i
      }
   }
   
// #define USE_ESC 1
#if (WITH_ALI>0)
   for(int t=0; t<TRANSITIONS; t++) {  // modified Einstein A
      u = UL[2*t] ;  l = UL[2*t+1] ;
      // MATRIX[IDX(l,u)]  +=  ESC[id*TRANSITIONS+t] / (VOLUME*NI[id*LEVELS+u]) ;
# if 0
      MATRIX[IDX(l,u)]  +=  ESC[id*TRANSITIONS+t] / (NI[id*LEVELS+u]) ;  // 1D has no division by VOLUME !!
# else
      tmp = NI[id*LEVELS+u] ;
      if (tmp>1.0e-30f) MATRIX[IDX(l,u)]  +=  ESC[id*TRANSITIONS+t] / tmp ;  // 1D has no division by VOLUME !!
# endif
   }
#else
   for(int t=0; t<TRANSITIONS; t++) 
     MATRIX[IDX(UL[2*t+1], UL[2*t])] += A[t] ;  // MATRIX[l,u]
#endif   
   // --- NI is no longer used so we can reuse it to store vector X below ---
   
   for(int t=0; t<TRANSITIONS; t++) {
      u = UL[2*t] ;   l = UL[2*t+1] ;
      MATRIX[IDX(u,l)]  +=  SIJ[id*TRANSITIONS+t] * PC[id]  ;          // with PATH_CORRECTION
      MATRIX[IDX(l,u)]  +=  SIJ[id*TRANSITIONS+t] * PC[id] * G[l]/G[u] ;
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
      for(int j=0; j<LEVELS; j++) {      // row
         for(int i=0; i<LEVELS; i++) {   // column
            printf(" %10.3e", MATRIX[IDX(j,i)]) ;
         }
         printf("\n") ;
      }
   }
#endif
   
   u  = Doolittle_LU_Decomposition_with_Pivoting(MATRIX, pivot,    LEVELS) ;
   u *= Doolittle_LU_with_Pivoting_Solve(MATRIX, VECTOR, pivot, V, LEVELS) ;   
   tmp = 0.0f ;   for(int i=0; i<LEVELS; i++)  tmp  +=  V[i] ;
   for(int i=0; i<LEVELS; i++)  B[i]  =  V[i]*RHO[id]*ABU[id] / tmp ;  // B ro, X rw
   
#if 0
   if (id==0) {    
      printf("    =>    %12.4e  %12.4e  %12.4e  %12.4e\n", B[0], B[1], B[2], B[3]) ;      
   }
#endif
}



