#define PRECALCULATED_GAUSS 1




inline void atomicAdd_g_f(volatile __global float *addr, float val)
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




__kernel void Update(
                     __global float4 *CLOUD,   //  0 [CELLS]: Vrad, Rc, dummy, sigma
                     constant float  *GAU,     //  1 precalculated gaussian profiles [GNO*CHANNELS]
                     const int        ITRAN,   //  2 index of the transition
                     const float      Aul,     //  3 Einstein A(upper->lower)
                     const float      A_b,     //  4 (g_u/g_l)*B(upper->lower)
                     const float      GN,      //  5 Gauss norm. == C_LIGHT/(1e5*DV*freq) * GL
                     __global float  *DIRWEI,  //  6 individual weights for the rays
                     __global float  *VOLUME,  //  7 cell volume / cloud volume
                     __global float  *STEP,    //  8 average path length [NRAY*CELLS] (GL)
                     constant float  *APL,     //  9 average path length [CELLS] (GL)
                     const float      BG,      // 10 background value (photons per ray)
                     __global float  *IP,      // 11 impact parameter [NRAY] [GL]
                     __global float2 *NI,      // 12 [CELLS]:  NI[upper] + NB_NB
                     __global float  *SIJ_ARRAY,     // 13 [CELLS]:  SIJ
                     __global float  *ESC_ARRAY      // 14 [CELLS]:  ESC
#if (COOLING>0)
                     ,__global float *COOL     // 15 [CELLS] = cooling 
#endif
                    )  {
   // Follow one ray, update SIJ and ESC counters
   // One work item per ray
   float weight, dx, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;
   float sum_delta_true, all_escaped, nu ;
   constant float* profile ;
   int   row, shift, INDEX ;
   int   id  = get_global_id(0) ;  // ray index
   int   lid = get_local_id(0) ;   // index within the local work group
   if  (id>=NRAY) return ;   
   float ip = IP[id] ;             // impact parameter [GL]
   
   __local float  NTRUES[LOCAL*CHANNELS] ;         // 32 * 200 channels = 25kB
   __local float *NTRUE = &NTRUES[lid*CHANNELS] ;  // spectrum in the current ray
   for(int i=0; i<CHANNELS; i++) NTRUE[i] = BG * DIRWEI[id]  ;
   
   __global float *SIJ = &SIJ_ARRAY[ITRAN*CELLS] ;
   __global float *ESC = &ESC_ARRAY[ITRAN*CELLS] ;
   
   INDEX = CELLS-1 ;    // always starts with the outermost ray
   
   
#if (COOLING>0)
   float cool = BG*DIRWEI[id]*CHANNELS ;
   atomicAdd_g_f(&(COOL[INDEX]), -cool) ; // heating of the entered cell
#endif
   
   
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
      tmp_emit  =  weight * nu*Aul / tmp_tau ;
      shift     =  round(doppler/WIDTH) ;
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      profile   =  &GAU[row*CHANNELS] ;
      sum_delta_true = all_escaped = 0.0f ;
      for(int ii=max(0, shift); ii<min(CHANNELS, CHANNELS+shift); ii++)  {
#if 1
         factor           =  1.0f-exp(-tmp_tau*profile[ii-shift]) ;
#else
         factor = (fabs(tmp_tau*profile[ii-shift])>0.001f) ?  (1.0f-exp(-tmp_tau*profile[ii-shift])) : (tmp_tau*profile[ii-shift]) ;
#endif
         escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed         =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped     +=  escape ;             // sum of escaping photons over the profile
      }  // loop over channels
      
      
#if 0
      // Update SIJ and ESC (note - there may be two updates, one incoming, one outgoing ray)
      ARES[id*CELLS+INDEX].x  += A_b * sum_delta_true / nb_nb  ; // to be divided by VOLUME
      // Emission ~ path length dx but also weighted according to direction, works because <WEI>==1.0
      ARES[id*CELLS+INDEX].y  += all_escaped ;                   // to be divided by VOLUME
#else
      atomicAdd_g_f(&SIJ[INDEX], A_b * sum_delta_true / nb_nb) ;
      atomicAdd_g_f(&ESC[INDEX], all_escaped) ;
#endif

      
      
#if (COOLING>0)
      // total number of photons in the package as it exits the cell
      float cool = 0.0f ;
      for(int ii=0; ii<CHANNELS; ii++) cool += NTRUE[ii] ;
      atomicAdd_g_f(&(COOL[INDEX]), cool) ; // cooling of cell INDEX
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
      
#if (COOLING>0)
      atomicAdd_g_f(&(COOL[INDEX]), -cool) ; // heating of the next cell
#endif
      
      
   } // while(1)
}




__kernel void Spectra(
                      __global float4 *CLOUD,        //  0 [CELLS]: Vrad, Rc, dummy, sigma
                      constant float  *GAU,          //  1 precalculated gaussian profiles
                      const float      GN,           //  3 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                      __global float2 *NI,           //  5 [CELLS]:  NI[upper] + NB_NB
                      const float      BG,           //  9 background intensity
                      const float      emis0,        // 10 h/(4pi)*freq*Aul*int2temp
                      __global float  *IP,           //    impact parameter [GL]
                      __global float  *STEP,         //    STEP[iray*CELLS+iray]  [GL]
                      __global float  *NTRUE_ARRAY,  // 11 NRAY*CHANNELS
                      __global float  *SUM_TAU_ARRAY // 12 NRAY*CHANNELS
                     )
{
   // one work item per ray; the same rays as used in the actual calculation!
   int id = get_global_id(0) ;   // id == ray index
   if (id>=NRAY) return ; // no more rays
   __global float *NTRUE   = &(NTRUE_ARRAY[id*CHANNELS]) ;
   __global float *SUM_TAU = &(SUM_TAU_ARRAY[id*CHANNELS]) ;
   int i ;
   for(int i=0; i<CHANNELS; i++) {
      NTRUE[i]   = 0.0f ;  SUM_TAU[i] = 0.0f ;
   }
   float tau, dtau, emissivity, doppler, nu, dx ;
   int row, shift ;
   constant float* profile ;
   float ip = IP[id] ; // impact parameter [GL] ;
   int INDEX=CELLS-1, dstep = -1 ;
   while (1) {
      dx    =  STEP[id*CELLS+INDEX] ;  // [GL]
      nu    =  NI[INDEX].x ;     
      tau   =  (fabs(NI[INDEX].y)<1.0e-26f) ? (1.0e-26f) :  (NI[INDEX].y) ;
      tau  *=  GN*GL*dx ;
      tau   =  clamp(tau, -2.0f, 1.0e10f) ;      
      doppler    =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(ip/CLOUD[INDEX].y,2.0f)) ) ;
      row        =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      profile    =  &GAU[row*CHANNELS] ;
      shift      =  round(doppler/WIDTH) ;
      // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      emissivity =  emis0 * nu * dx * GL ;      
      for(i=max(0, shift); i<min(CHANNELS, CHANNELS+shift); i++) {         
         dtau       =  tau*profile[i-shift] ;
         dx         =  emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i]) ;
         dx        *=   (fabs(dtau)>1.0e-7f) ? ((1.0f-exp(-dtau)) / dtau) : (1.0f-0.5f*tau) ;
         NTRUE[i]   +=  dx ;
         SUM_TAU[i] +=  dtau  ;
      }
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
   for (i=0; i<CHANNELS; i++) NTRUE[i] -=  BG*(1.0f-exp(-SUM_TAU[i])) ;
}

