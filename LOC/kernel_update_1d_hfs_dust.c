#define PRECALCULATED_GAUSS 1

// HFS with LTE rations, including dust emission and absorption

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
                     __global float4 *CLOUD,   //  0 [CELLS]: Vrad, Rc, dummy, sigma
                     __global float  *GAU,     //  1 precalculated gaussian profiles [TRANSITIONS*CELLS*CHANNELS]
                     const float      Aul,     //  2 Einstein A(upper->lower)
                     const float      A_b,     //  3 (g_u/g_l)*B(upper->lower)
                     const float      GN,      //  4 Gauss norm. == C_LIGHT/(1e5*DV*freq) * GL
                     __global float  *DIRWEI,  //  5 individual weights for the rays
                     __global float  *VOLUME,  //  6 cell volume / cloud volume
                     __global float  *STEP,    //  7 average path length [NRAY*CELLS] (GL)
                     constant float  *APL,     //  8 average path length [CELLS] (GL)
                     const float      BG,      //  9 background value (photons per ray)
                     __global float  *IP,      // 10 impact parameter [NRAY] [GL]
                     __global float2 *NI,      // 11 [CELLS]:  NI[upper] + NB_NB
                     __global float2 *ARES,    // 12 [CELLS*NRAY]:  SIJ, ESC
                     const int TRAN,           // 13 transition
                     const int NCHN,           // 14 channels used
                     __global float *NTRUE_ARRAY, // 15 NTRUE_ARRAY[NRAY*CHANNELS]
                     __global int2  *LIM,      // 16 first and last significant channels
                     constant float *CRT_TAU,  // 17 dust optical depth / GL
                     constant float *CRT_EMI   // 18 dust emission photons/s/channel/H2
                    )  {
   // Follow one ray, update SIJ and ESC counters
   // One work item per ray => each work item must update a separate element of ARES
   // NCHN <= CHANNELS
   float weight, dx, doppler, tmp_tau, nb_nb, escape, tt, ttt ;
   float nu, Ctau, Cemit, sij, Ltau, Ttau, Lleave, Dleave, pro ;
   __global float* profile ;
   int   shift, INDEX ;
   int   id  = get_global_id(0) ;  // ray index
   int   lid = get_local_id(0) ;   // index within the local work group
   if  (id>=NRAY) return ;   
   float ip = IP[id] ;             // impact parameter [GL]
   
   __global float *NTRUE = &NTRUE_ARRAY[id*CHANNELS] ;  // spectrum in the current ray
   for(int i=0; i<NCHN; i++) NTRUE[i] = BG * DIRWEI[id]  ;  // limited number of channels
   
   
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
      sij       =  0.0f ;
      // Dust optical depth and emission
      Ctau      =  dx     * CRT_TAU[INDEX*TRANSITIONS+TRAN] ;
      Cemit     =  weight * CRT_EMI[INDEX*TRANSITIONS+TRAN] ;
      escape    = 0.0f ;
      sij       = 0.0f ;
      for(int ii=max(0, shift); ii<min(NCHN, NCHN+shift); ii++)  { // max NCHN channels
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
         escape       +=  Lleave  +  weight*nu*Aul*pro * Ctau * ttt ;
         // Total change of photons in the package
         NTRUE[ii]     =  NTRUE[ii]*exp(-Ttau) + (Dleave + Lleave) ;
      }  // loop over channels
      // Update SIJ and ESC (note - there may be two updates, one incoming, one outgoing ray)
      ARES[id*CELLS+INDEX].x  += sij ;           // to be divided by VOLUME
      // Emission ~ path length dx but also weighted according to direction, works because <WEI>==1.0
      ARES[id*CELLS+INDEX].y  += escape ;        // to be divided by VOLUME
      
      // barrier(CLK_LOCAL_MEM_FENCE) ;
      
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



__kernel void SpectraOld(
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
                         constant float  *CRT_TAU,       // 12 optical depth / GL
                         constant float  *CRT_EMI        // 13 dust emissivity
                        )
{
   // one work item per ray; the same rays as used in the actual calculation!
   int id = get_global_id(0) ;   // id == ray index
   if (id>=NRAY) return ; // no more rays
   __global float *NTRUE   = &(NTRUE_ARRAY[id*CHANNELS]) ;
   __global float *SUM_TAU = &(SUM_TAU_ARRAY[id*CHANNELS]) ;
   int i ;
   for(int i=0; i<NCHN; i++) {
      NTRUE[i]   = 0.0f ;  SUM_TAU[i] = 0.0f ;
   }
   float tau, dtau, emissivity, doppler, nu, dx, Ctau, Cemit ;
   int shift ;
   __global float* profile ;
   float ip = IP[id] ; // impact parameter [GL] ;
   int INDEX=CELLS-1, dstep = -1 ;
   
#if 0
   if (id==0) {
      printf(" TRAN %2d  CRT_TAU %10.3e   CRT_EMIT %10.3e\n", 
             TRAN, CRT_TAU[0*TRANSITIONS+TRAN], CRT_EMI[0*TRANSITIONS+TRAN]) ;
   }
#endif
   
   while (1) {
      dx          =  STEP[id*CELLS+INDEX] ;  // [GL]
      nu          =  NI[INDEX].x ;
      // Line optical depth and emissivity
      tau         =  (fabs(NI[INDEX].y)<1.0e-24f) ? (1.0e-24f) :  (NI[INDEX].y) ;
      tau        *=  GN*GL*dx ;
      tau         =  clamp(tau, -2.0f, 1.0e10f) ;
      doppler     =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(ip/CLOUD[INDEX].y,2.0f)) ) ;
      profile     =  &GAU[TRAN*CELLS*CHANNELS+INDEX*CHANNELS] ; // use NCHN first elements
      shift       =  round(doppler/WIDTH) ;      
      emissivity  =  emis0 * nu * dx * GL ; // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      // Dust optical depth and dust emissivity (density already in CRT_EMI)
      Ctau        =  CRT_TAU[INDEX*TRANSITIONS+TRAN] * dx ;
      Cemit       =  CRT_EMI[INDEX*TRANSITIONS+TRAN] * GL * dx ;
      for(i=max(0, shift); i<min(NCHN, NCHN+shift); i++) {         
         dtau        =  tau*profile[i-shift] + Ctau ;
         dx          =  (emissivity*profile[i-shift]*GN + Cemit) * exp(-SUM_TAU[i]) ;
         dx         *=   (fabs(dtau)>1.0e-4f) ? ((1.0f-exp(-dtau))/dtau) : (1.0f-0.5f*dtau) ;
         NTRUE[i]   +=  dx ;
         SUM_TAU[i] +=  dtau  ;
#if 0
         if (NTRUE[i]>=0.0f) {
            ;
         } else {
            printf("INDEX %2 dx %10.3e dtau %10.3e emissivity %10.3e\n", INDEX, dx, dtau, emissivity) ;
         }
#endif
      }      
      // next cell
      INDEX += dstep ;
      if (INDEX>=CELLS) break ;  // going outwards and outermost shell was processed
      if (INDEX<0) {             // we went through the centre cell, turn outwards
         INDEX = 1 ;  dstep = 1 ;
      } else {
         if (STEP[id*CELLS+INDEX]<=0.0f) { // ray does not go further in, turn outwards
            INDEX += 2 ; // we went through i but not i-1 => continue with i+1
            dstep  = 1 ;            
         }
      }
      if (INDEX>=CELLS) break ;  // went tangentially through the outermost shell
   } // while(1)
#if 0
   printf(" CONTINUUM %10.3e / %10.3e  TAU %10.3e BG %8.2f\n", 
          NTRUE[1], NTRUE[1]-BG*(1.0f-exp(-SUM_TAU[1])), SUM_TAU[1], BG) ;
#endif
   if (BGSUB>0) {
      for (i=0; i<NCHN; i++) {
         NTRUE[i] -= BG *
           ( (fabs(SUM_TAU[i])>1.0e-4) ? (1.0f-exp(-SUM_TAU[i])) : (SUM_TAU[i]) ) ;
#if 0
         if (NTRUE[i]>=0.0f) {
            ;
         } else {
            printf("??? NTRUE %10.3e SUM_TAU %10.3e FACTOR %10.3e\n",
                   NTRUE[i], SUM_TAU[i],
                   ( (fabs(SUM_TAU[i])>1.0e-4) ? (1.0f-exp(-SUM_TAU[i])) : (SUM_TAU[i]) )) ;
         }
#endif         
      }
   }
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
                      constant float  *CRT_TAU,       // 12 optical depth / GL
                      constant float  *CRT_EMI        // 13 dust emissivity
                     )
{
   // 2017-01-15  a version using equidistant rays (STEP, IP have been updated after simulation)
   // one work item per ray; the same rays as used in the actual calculation!
   int id = get_global_id(0) ;   // id == ray index
   if (id>=NRAY_SPE) return ; // no more rays
   __global float *NTRUE   = &(NTRUE_ARRAY[id*CHANNELS]) ;
   __global float *SUM_TAU = &(SUM_TAU_ARRAY[id*CHANNELS]) ;
   int i ;
   for(int i=0; i<NCHN; i++) {
      NTRUE[i]   = 0.0f ;  SUM_TAU[i] = 0.0f ;
   }
   float tau, dtau, emissivity, doppler, nu, dx, Ctau, Cemit, pro ;
   int shift ;
   __global float* profile ;
   float ip = IP[id] ; // impact parameter [GL] ;
   int INDEX=CELLS-1, dstep = -1 ;
   
#if 0
   if (id==0) {
      printf(" TRAN %2d  CRT_TAU %10.3e   CRT_EMIT %10.3e\n", 
             TRAN, CRT_TAU[0*TRANSITIONS+TRAN], CRT_EMI[0*TRANSITIONS+TRAN]) ;
   }
#endif
   
   while (1) {
      dx          =  STEP[id*CELLS+INDEX] ;  // [GL]
      nu          =  NI[INDEX].x ;
      // Line optical depth and emissivity
      tau         =  (fabs(NI[INDEX].y)<1.0e-24f) ? (1.0e-24f) :  (NI[INDEX].y) ;
      tau        *=  GN*GL*dx ;
      tau         =  clamp(tau, -2.0f, 1.0e10f) ;
      doppler     =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(ip/CLOUD[INDEX].y,2.0f)) ) ;
      profile     =  &GAU[TRAN*CELLS*CHANNELS+INDEX*CHANNELS] ; // use NCHN first elements
      shift       =  round(doppler/WIDTH) ;      
      emissivity  =  emis0 * nu * dx * GL ; // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      // Dust optical depth and dust emissivity (density already in CRT_EMI)
      Ctau        =  CRT_TAU[INDEX*TRANSITIONS+TRAN] * dx ;
      Cemit       =  CRT_EMI[INDEX*TRANSITIONS+TRAN] * GL * dx ;
      for(i=0; i<NCHN; i++) {
         pro         =  profile[(clamp(i-shift, 0, NCHN-1))] ;
         dtau        =  tau*pro + Ctau ;
         dx          =  (emissivity*pro*GN + Cemit) * exp(-SUM_TAU[i]) ;
         dx         *=   (fabs(dtau)>1.0e-4f) ? ((1.0f-exp(-dtau))/dtau) : (1.0f-0.5f*dtau) ;
         NTRUE[i]   +=  dx ;
         SUM_TAU[i] +=  dtau  ;
#if 0
         if (NTRUE[i]>=0.0f) {
            ;
         } else {
            printf("INDEX %2 dx %10.3e dtau %10.3e emissivity %10.3e\n", INDEX, dx, dtau, emissivity) ;
         }
#endif
      }      
      // next cell
      INDEX += dstep ;
      if (INDEX>=CELLS) break ;  // going outwards and outermost shell was processed
      if (INDEX<0) {             // we went through the centre cell, turn outwards
         INDEX = 1 ;  dstep = 1 ;
      } else {
         if (STEP[id*CELLS+INDEX]<=0.0f) { // ray does not go further in, turn outwards
            INDEX += 2 ; // we went through i but not i-1 => continue with i+1
            dstep  = 1 ;            
         }
      }
      if (INDEX>=CELLS) break ;  // went tangentially through the outermost shell
   } // while(1)
#if 0
   printf(" CONTINUUM %10.3e / %10.3e  TAU %10.3e BG %8.2f\n", 
          NTRUE[1], NTRUE[1]-BG*(1.0f-exp(-SUM_TAU[1])), SUM_TAU[1], BG) ;
#endif
   if (BGSUB>0) {
      for (i=0; i<NCHN; i++) {
         NTRUE[i] -= BG *
           ( (fabs(SUM_TAU[i])>1.0e-4) ? (1.0f-exp(-SUM_TAU[i])) : (SUM_TAU[i]) ) ;
#if 0
         if (NTRUE[i]>=0.0f) {
            ;
         } else {
            printf("??? NTRUE %10.3e SUM_TAU %10.3e FACTOR %10.3e\n",
                   NTRUE[i], SUM_TAU[i],
                   ( (fabs(SUM_TAU[i])>1.0e-4) ? (1.0f-exp(-SUM_TAU[i])) : (SUM_TAU[i]) )) ;
         }
#endif         
      }
   }
}

