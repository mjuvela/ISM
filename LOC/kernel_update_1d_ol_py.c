#define real float
#define H_K  4.799243348e-11

__kernel void Clear(__global float2 *ARES) {
   // Clear the work array ARES[NWG*TRANSITIONS*CELLS]
   int id = get_global_id(0) ;
   for(int i=id; i<NWG*TRANSITIONS*CELLS; i+=GLOBAL) {
      ARES[i] = (float2)(0.0f, 0.0f) ;
   }
}


__kernel void Sum(__global float2 *ARES, __global float2 *RES, __global float *VOLUME) {
   // Sum over results of individual work groups
   //    ARES[NWG * TRANSITIONS * CELLS]  ->  RES[CELLS * TRANSITIONS]
   int id = get_global_id(0) ;  // CELLS
   if (id>=CELLS) return ;      // id = cell
   for(int itran=0; itran<TRANSITIONS; itran++) {
      float sij=0.0f, esc=0.0f ;
      for(int i=0; i<NWG; i++) {
         sij += ARES[i*TRANSITIONS*CELLS+itran*CELLS+id].x ;
         esc += ARES[i*TRANSITIONS*CELLS+itran*CELLS+id].y ;
      }
      RES[id*TRANSITIONS+itran].x = sij / VOLUME[id] ;
      RES[id*TRANSITIONS+itran].y = esc / VOLUME[id] ;
   }
#if 0
   if (id==0) {
      for(int i=0; i<NWG; i++) {
         // for(int t=0; t<TRANSITIONS; t++) printf(" %12.4e ", ARES[i*TRANSITIONS*CELLS+t*CELLS+id].x) ;
         for(int t=0; t<TRANSITIONS; t++) printf(" %12.4e ", ARES[i*TRANSITIONS*CELLS+t*CELLS+id].y) ;
         printf("\n") ;
      }
      printf("\n") ;
   }
#endif
}



__kernel void Update(
                     __global float4 *CLOUD,   //  0 [CELLS]: Vrad, Rc, dummy, sigma
                     __global float  *VOLUME,  //  1 [CELLS] cell volume / cloud volume
                     __global float  *Aul,     //  2 [TRANSITIONS], Einstein A
                     __global float  *Blu,     //  3 [TRANSITIONS],  Blu*(h*f)/(4*pi)
                     __global float  *GAU,     //  4 [CELLS*CHANNELS], precalculated line profiles
                     __global int2   *LIM,     //  5 [CELLS], first and last significant channels
                     __global float  *GN,      //  6 [TRANSITIONS]
                     __global float  *DIRWEI,  //  7 [NRAY]  individual weights for the rays
                     __global float  *STEP,    //  8 [NRAY,CELLS] step lengths (GL)
                     __global float  *APL,     //  9 [CELLS], average path length (GL)
                     __global float  *IP,      // 10 [NRAY], impact parameter [GL]
                     __global float  *BG,      // 11 [TRANSITIONS], background values (photons per ray)
                     __global float  *NU,      // 12 [CELLS*TRANSITIONS], n(uppper)
                     __global float  *NBNB,    // 13 [CELLS*TRANSITIONS], nb_nb
                     __global float2 *ARES,    // 14 [NWG,TRANSITIONS,CELLS]:  SIJ, ESC
                     __global int    *SINGLE,  // 15 list of single transitions
                     __global float  *NTRUES   // 16 global work space, [maxi]
#if (WITH_CRT>0)
                     ,
                     __global float *CRT_TAU,
                     __global float *CRT_EMI
#endif
                    )  {
   // Update normal single transitions, we have NWG work groups, NRAY is multiple of NWG,
   // each work group loops over NRAY//NWG rays
   //      loop over transitions
   //         loop over channels, updating local soc[], loc[]
   //         add soc[] and loc[] and finally update ARES[NWG*TRANSITIONS*CELLS]
   float weight, dx, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;
   float sum_delta_true, all_escaped, nu ;
   int   shift, INDEX, dstep ;
   int   id  = get_global_id(0) ;
   int   lid = get_local_id(0) ;   // index within the local work group
   int   gid = get_group_id(0) ;
   if    (gid>=NWG) return ;  // NWG work groups
   
   // NTRUE must be at least NWG*TRANSITIONS*CHANNELS
#if (LOWMEM==0)
   __local  float NTRUE[TRANSITIONS*CHANNELS] ;
   __local  float *ntrue ;
#else
   __global float *NTRUE = &NTRUES[gid*TRANSITIONS*CHANNELS] ;
   __global float *ntrue ;
#endif   
   __global float *profile ;
   
#if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, pro, sij ;
#endif
   
   
   for(int iray=gid; iray<NRAY; iray+=NWG) {
      INDEX = CELLS-1 ;   // always starts with the outermost ray
      dstep = -1 ;        // we first go inwards (dstep<0), then outwards until ray exits
      for(int itran=0; itran<TRANSITIONS; itran++) {
         for(int i=lid; i<CHANNELS; i+=LOCAL) NTRUE[itran*CHANNELS+i] = DIRWEI[iray] * BG[itran] ;
      }
      barrier(CLK_LOCAL_MEM_FENCE) ;
      while(1) { // follow one ray through the cloud
         dx        =  STEP[iray*CELLS+INDEX] ;  // [GL]
         weight    =  DIRWEI[iray]*(dx/APL[INDEX])*VOLUME[INDEX] ;  // VOLUME == fraction of cloud volume
         doppler   =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(IP[iray]/CLOUD[INDEX].y, 2.0f)) ) ;
         // a single profile vector used by all transitions, the whole work group
         profile = &GAU[INDEX*CHANNELS] ;
         barrier(CLK_LOCAL_MEM_FENCE) ;
         shift   =  round(doppler/WIDTH) ;
         int i1  =  max(shift,          LIM[INDEX].x) ;
         int i2  =  min(CHANNELS, shift+LIM[INDEX].y) ;
         
         // loop over transitions WORK ITEM PER LEVEL
         for(int itran=lid; itran<TRANSITIONS; itran+=LOCAL) {
            if (SINGLE[itran]==0) continue ;
            ntrue     =  &NTRUE[itran*CHANNELS] ;
            nu        =   NU[INDEX*TRANSITIONS  +itran] ;    // [CELLS*TRANSITIONS]
            nb_nb     =   NBNB[INDEX*TRANSITIONS+itran] ;
            if (fabs(nb_nb)<1.0e-37f) nb_nb=1.0e-37f ;
            tmp_tau   =  dx*GL*nb_nb*GN[itran] ;           // GN DOES NOT include GL [cm]
            tmp_emit  =  weight * nu*Aul[itran] / tmp_tau ;
            sum_delta_true = all_escaped = 0.0f ;
#if (WITH_CRT>0)
            sij        =  0.0f ;
            Ctau       =  dx     * CRT_TAU[INDEX*TRANSITIONS+itran] ;
            Cemit      =  weight * CRT_EMI[INDEX*TRANSITIONS+itran] ;
            for(int ii=i1; ii<i2; ii++)  {
               pro     =  profile[ii-shift] ;
               Ltau    =  tmp_tau*pro ;
               Ttau    =  Ctau + Ltau ;
               tt      =  (1.0f-exp(-Ttau))/Ttau ;
               ttt     =  (1.0f-tt)/Ttau ;
               Lleave  =  weight*nu*Aul[itran]*pro* tt ;
               Dleave  =  Cemit * tt ;
               sij    +=  Blu[itran] * pro*GN[itran]*dx * NTRUE[ii] * tt ;
               sij    +=  Blu[itran] * pro*GN[itran]*dx * Cemit     * ttt ; // GN includes GL!
               all_escaped  +=   Lleave  +  weight*nu*Aul[itran]*pro * Ctau * ttt ;
               ntrue[ii]     =   ntrue[ii]*exp(-Ttau) + (Dleave + Lleave) ;               
            }
            ARES[gid*TRANSITIONS*CELLS+itran*CELLS+INDEX].x +=  sij ;
            ARES[gid*TRANSITIONS*CELLS+itran*CELLS+INDEX].y +=  all_escaped ;
#else
            // each work item processes one channel at a time
            for(int ii=i1; ii<i2; ii++)  {
               // factor           =  1.0f-exp(-tmp_tau*profile[ii-shift]) ;
               nu               =  tmp_tau*profile[ii-shift] ;
               factor           =  (fabs(nu)>1.0e-4) ? (1.0f-exp(-nu)) : (nu) ;
               escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
               absorbed         =  ntrue[ii]*factor ;   // incoming photons that are absorbed
               ntrue[ii]       +=  escape-absorbed ;
               sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
               all_escaped     +=  escape ;             // sum of escaping photons over the profile
            } 
            // update SIJ and ESC in local array
            ARES[gid*TRANSITIONS*CELLS+itran*CELLS+INDEX].x +=  Blu[itran] * sum_delta_true / nb_nb  ;
            ARES[gid*TRANSITIONS*CELLS+itran*CELLS+INDEX].y +=  all_escaped ;
#endif
         } // loop over transitions
         
         barrier(CLK_LOCAL_MEM_FENCE) ;
         // next cell
         INDEX += dstep ;
         if (INDEX>=CELLS) break ;  // going outwards and outermost shell was processed
         if (INDEX<0) {             // we went through the centre cell, turn outwards
            INDEX = 1 ;  dstep = 1 ;
         } else {
            if (STEP[iray*CELLS+INDEX]<=0.0f) { // ray does not go that far in, turn outwards
               INDEX += 2 ;  dstep  = 1 ;               
            }
         }
         if (INDEX>=CELLS) break ;
      } // while(1) ... loop over ray
   } // for iray
   
}





__kernel void Spectra(
                      __global float4 *CLOUD,         //  0 [CELLS]: Vrad, Rc, ABUNDANCE, sigma
                      __global float  *GAU,           //  1 precalculated gaussian profiles
                      const float      GN,            //  2 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                      __global float2 *NI,            //  3 [CELLS]:  NI[upper] + NB_NB
                      const float      BG,            //  4 background = B(Tbg)*I2T
                      const float      emis0,         //  5 h/(4pi)*freq*Aul*int2temp
                      __global float  *IP,            //  6 impact parameter [GL]
                      __global float  *STEP,          //  7 STEP[iray*CELLS+iray]  [GL]
                      __global float  *NTRUE_ARRAY,   //  8 NRAY*CHANNELS
                      __global float  *SUM_TAU_ARRAY, //  9 NRAY*CHANNELS
                      __global float  *RHO,           // 10 [CELLS] ... for column density calculation
                      __global float4 *COLDEN,        // 11 [NRAY_SPE] = N, N_mol, tau, dummy
#if (WITH_CRT>0)
                      const int       TRAN,           //  index of the current transition
                      constant float *CRT_TAU,        //  dust optical depth / GL
                      constant float *CRT_EMI,        //  dust emission photons/c/channel/H
#endif
                      const int savetau
                     )
{
   // one work item per ray; the same rays as used in the actual calculation!
   // IDENTICAL TO THE Spectra() ROUTINE WITHOUT OVERLAPPING
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
      row        =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      profile    =  &GAU[row*CHANNELS] ;
      shift      =  round(doppler/WIDTH) ;
      // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      emissivity =  emis0 * nu * dx * GL ;      
#if (KILL_EMISSION<CELLS)
      if (INDEX>=KILL_EMISSION)  emissivity = 0.0f ;  // kill emission from outer shells
#endif      
#if (WITH_CRT>0)
      distance      +=   dx*GL ;
      Ctau           =   CRT_TAU[INDEX*TRANSITIONS+TRAN]      * dx ;  // tau            * s
      Cemit          =   CRT_EMI[INDEX*TRANSITIONS+TRAN] * GL * dx ;  // phot/s/chn/cm3 * s
# if (KILL_EMISSION<CELLS)
      if (INDEX>=KILL_EMISSION)  Cemit = 0.0f ;  // kill emission from outer shells
# endif      
      // loop over all channels... to get continuum correct also at band edges
      for(i=0; i<CHANNELS; i++) {
         pro         =   profile[clamp(i-shift, 0, CHANNELS-1)] ;
         dtau        =   tau*pro + Ctau ;
         dx          =  (emissivity*pro*GN + Cemit) * exp(-SUM_TAU[i]) ;
         //dx         *=  (fabs(dtau)>1.0e-4f) ? ((1.0f-exp(-dtau))/dtau) : (1.0f-0.5f*dtau) ;
         dx         *=  ((1.0f-exp(-dtau))/dtau) ;
         NTRUE[i]   +=   dx   ;
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
   if (savetau>0) {
      for (i=0; i<CHANNELS; i++) {
         NTRUE[i]  =  SUM_TAU[i] ;
         maxtau    =  max(maxtau, SUM_TAU[i]) ;
      }
   } else {
      for (i=0; i<CHANNELS; i++) {
         NTRUE[i] +=  BG*(exp(-SUM_TAU[i])-1.0f) ;
         maxtau    =  max(maxtau, SUM_TAU[i]) ;
      }
   }
#if (WITH_CRT>0)
   if (id==0) {
      printf("TRAN %2d,  TAU_CON %12.5e,  TAU_TOT %12.5e,  TA %6.4f,   d = %12.5e pc\n", 
             TRAN, SUM_TAU[1], SUM_TAU[CHANNELS/2], NTRUE[CHANNELS/2], distance/3.0857e+18f) ;
   }
#endif
   COLDEN[id].x  = colden ;
   COLDEN[id].y  = coldenmol ;
   COLDEN[id].z  = maxtau ;
   
}




// ===========================================================================================
   


__kernel void Overlap(
                      __global   float4 *CLOUD,   //  0 [CELLS]: Vrad, Rc, dummy, sigma
                      __global   float  *VOLUME,  //  1 cell volume / cloud volume
                      __constant float  *Aul,     //  2 Aul [TRANSITIONS]
                      __constant float  *BLU,     //  3 A_b [TRANSITIONS] = B_lu * h*f / (4*pi)
                      __global   float  *GAU,     //  4 precalculated gaussian profiles [CELLS*CHANNELS]
                      __constant int2   *LIM,     //  5 first and last significant channels
                      __constant float  *GN,      //  6 gauss normalisation [TRANSITIONS] -- GL **NOT** inlcuded
                      __global   float  *DIRWEI,  //  7 individual weights for the rays
                      __global   float  *STEP,    //  8 average path length [NRAY*CELLS] (GL)
                      __global   float  *APL,     //  9 average path length [CELLS] (GL)
                      __global   float  *IP,      // 10 impact parameter [NRAY] [GL]
                      __global   float  *BG,      // 11 [TRANSITIONS]  (photons per ray)
                      __global   float  *NU,      // 12 n_u   [TRANSITIONS*CELLS]
                      __global   float  *NBNB,    // 13 nb_nb [TRANSITIONS*CELLS]
                      __global   float2 *ARES,    // 14 [NWG*TRANSITIONS*CELLS]
                      const int          NCMP,    // 15 number of lines in this band
                      const int          NCHN,    // 16 channels in combined spectrum
                      __constant int    *TRAN,    // 17 list of transitions
                      __constant float  *OFF,     // 18 offsets in channels
                      __global float    *NTRUES,  // 19 [4*NRAY*CHANNELS] !!!!!!
                      __global float    *WRK      // 20 LOWMEM>1 only, work space [NWG*MAXCHN*3], only MAXCHN used
#if (WITH_CRT>0)
                      ,
                      constant float *CRT_TAU,  //  dust optical depth / GL
                      constant float *CRT_EMI   //  dust emission photons/c/channel/H
#endif
                     )  {
   // Each work group loops over rays.
   // Each work item of a work group is working on the same ray at the same time. 
   float weight, dx, doppler, tmp_tau, tmp, tmp_emit, emi, own, phi ;
   float s ;
   int   shift, INDEX, dstep, itran ;
   int   id  = get_global_id(0) ;  // *** not used ***
   int   lid = get_local_id(0) ;   // index within the local work group
   int   gid = get_group_id(0) ;   // work group index
   if (gid>=NWG) return ;          // NWG work groups used, may be less than allocated wgs
   
   // one kernel call = loop over rays but ONLY A SINGLE BAND
   // each loop needs TAU[MAXCHN], STAU[MAXCHN] that do fit in local
   __global float *NTRUE = &NTRUES[           gid*NCHN] ;
   __global float *EMIT  = &NTRUES[1*NWG*NCHN+gid*NCHN] ;
   __global float *TAU   = &NTRUES[2*NWG*NCHN+gid*NCHN] ;
#if (LOWMEM>1)
   __global float *TT    = &NTRUES[3*NWG*NCHN+gid*NCHN] ;
#else
   __local float TT[MAXCHN] ;
#endif
   
   
#if (LOWMEM>1)
   __global float *profile  =  &(WRK[gid*MAXCHN]) ;
#else
   __local  float profile[MAXCHN] ;
#endif
   
   __local  float sij[MAXCMP*LOCAL] ; // 32*20 < 3kB
   __local  float esc[MAXCMP*LOCAL] ; // only NCMP*LOCAL used
   
   float bg = BG[TRAN[0]] ;
   
   
   for(int iray=gid; iray<NRAY; iray+=NWG) { // work group loops over rays
      
      INDEX  = CELLS-1 ;   // always starts with the outermost ray
      dstep  = -1 ;        // we first go inwards (dstep<0), then outwards until ray exits
      for(int i=lid; i<NCHN; i+=LOCAL)  NTRUE[i] = bg*DIRWEI[iray]  ;
      
      // NTRUE[] is updated only by this work group
      barrier(CLK_GLOBAL_MEM_FENCE) ;
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
      while(1) { // follow one ray through the cloud
         // barrier(CLK_GLOBAL_MEM_FENCE) ;
         barrier(CLK_LOCAL_MEM_FENCE) ;
         dx        =  STEP[iray*CELLS+INDEX] ;  // [GL]
         s         =  dx*GL ;                   // [cm]
         weight    =  DIRWEI[iray]*(dx/APL[INDEX])*VOLUME[INDEX] ;  // VOLUME == fraction of cloud volume
         doppler   =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(IP[iray]/CLOUD[INDEX].y, 2.0f)) ) ;
         // a single profile[CHANNELS] used by all transitions, the whole work group
         for(int i=lid; i<CHANNELS; i+=LOCAL)   profile[i]  =  GAU[INDEX*CHANNELS+i] ;
         for(int i=lid; i<NCMP*LOCAL; i+=LOCAL) sij[i] = esc[i] =  0.0f ;
#if (WITH_CRT>0)
         for(int i=lid; i<NCHN; i+= LOCAL){
            TAU[i]    =  dx     * CRT_TAU[INDEX*TRANSITIONS+TRAN[0]] ;
            EMIT[i]   =  weight * CRT_EMI[INDEX*TRANSITIONS+TRAN[0]] ;
         }
#else
         for(int i=lid; i<NCHN; i+= LOCAL) {
            TAU[i]    =  0.0f ;
            EMIT[i]     =  0.0f ;
         }
#endif
         
         barrier(CLK_LOCAL_MEM_FENCE) ;
#if (LOWMEM>1)
         barrier(CLK_GLOBAL_MEM_FENCE) ;
#endif
         // TAU[], EMIT[], sij[], esc[] updated only by this work group
         //    optical depth:   tau == nb_nb*s*GN*GL
         // phi  =   profile[]*GN = actual profile function
         // tau  =   nb_nb*s*GN*profile[]
         // nu*Aul*phi*s => weight*nu*Aul == true photons from cell / package / cloud volume
         
         // sum of component optical depths => TAU
         for(int itran0=0; itran0<NCMP; itran0++) {            // loop over NCMP components
            itran    =  TRAN[itran0] ;
            tmp_tau  =  s * NBNB[INDEX*TRANSITIONS+itran] * GN[itran] ;
            tmp      =  weight * NU[INDEX*TRANSITIONS+itran] * Aul[itran] ;
            shift    =  round(doppler/WIDTH) + OFF[itran0] ;   // offset, current component
            int i1   =  max(    -shift,  LIM[INDEX].x) ;
            int i2   =  min(NCHN-shift,  LIM[INDEX].y) ;
            for(int i=(i1+lid); i<i2; i+=LOCAL) {              // over profile
               TAU[i+shift]  +=  tmp_tau*profile[i] ;          // sum of all lines... and possibly dust
               EMIT[i+shift] +=  tmp * profile[i] ;            // photons within one channel
            }
            // TAU[], EMIT[] updated only by this work group
            barrier(CLK_LOCAL_MEM_FENCE) ;  // next loop, element updated by different work item?
            // barrier(CLK_GLOBAL_MEM_FENCE) ; // next loop, element updated by different work item?
         }
         
         // calculate (1-exp(-TAU))/TAU for all channels
         for(int i=lid; i<NCHN; i+=LOCAL) {
            TT[i] =  (fabs(TAU[i])>1.0e-5f) ? ((1.0f-exp(-TAU[i]))/TAU[i]) : (1.0f-0.5f*TAU[i]) ;
         }
         barrier(CLK_LOCAL_MEM_FENCE) ;
         
         // count absorptions of incoming photons:  BLU*ntrue*phi*(1-exp(-tau))/tau
         for(int itran0=0; itran0<NCMP; itran0++) { // loop over NCMP components
            itran   =  TRAN[itran0] ;
            shift   =  round(doppler/WIDTH) + OFF[itran0] ;  // offset, current component
            int i1  =  max(    -shift,  LIM[INDEX].x) ;
            int i2  =  min(NCHN-shift,  LIM[INDEX].y) ;
            for(int i=(i1+lid); i<i2; i+=LOCAL) {     // over the profile of this transition
               phi                    = GN[itran]*profile[i] ;  // PHI (absorption)
               // this is correct also with dust continuum
               sij[itran0*LOCAL+lid] += NTRUE[i+shift]*BLU[itran]*phi*s*TT[i+shift] ;
            }
         }
         barrier(CLK_LOCAL_MEM_FENCE) ;
         
         // emission-absorption -- each element updated by the same work item as above
         for(int itran0=0; itran0<NCMP; itran0++) {
            itran     =  TRAN[itran0] ;
            shift     =  round(doppler/WIDTH) + OFF[itran0] ;  // offset, current component
            int i1    =  max(    -shift,  LIM[INDEX].x) ;
            int i2    =  min(NCHN-shift,  LIM[INDEX].y) ;
            for(int i=(i1+lid); i<i2; i+=LOCAL) {   // over the profile of transition itran
               emi      =  weight*NU[INDEX*TRANSITIONS+itran]*Aul[itran]*profile[i] ; // emitted in one channel
               phi      =  profile[i] * GN[itran] ; // profile function in current channel = phi(freq)
               tmp_tau  =  s*NBNB[INDEX*TRANSITIONS+itran]*phi ;   // tau for individual line
               // sij <= all absorptions, except the photons emitted by the current line componentn
               tmp      =  (EMIT[i+shift]-emi) * BLU[itran]*phi*s*(1.0f-TT[i+shift]) / TAU[i+shift] ;
               sij[itran0*LOCAL+lid] +=  tmp ;               
               // Escape = W*nu*Aul*profile * ((1-exp(-TAU))/TAU) * (tau/TAU))
               //        = emi              *  TT        *  tmp_tau/TAU
               // if there were no other lines:
               //    esc[itran0*LOCAL+lid] +=  emi * TT[i+shift] * tmp_tau / TAU[i+shift] ;
               // taking into account overlapping:
               // esc[itran0*LOCAL+lid] +=  emi * (1.0f-(1.0f-TT[i+shift])*tmp_tau/TAU[i+shift]) ;
               esc[itran0*LOCAL+lid] +=  emi * (1.0f-(1.0f-TT[i+shift])*tmp_tau/TAU[i+shift]) ;
            }
         } // for itran0
         
         barrier(CLK_LOCAL_MEM_FENCE) ;
         // sij[], esc[] updated only by this work group
         // barrier(CLK_GLOBAL_MEM_FENCE) ;
         
         // update NTRUE
         for(int i=lid; i<NCHN; i+=LOCAL) {
            NTRUE[i]  *=  exp(-TAU[i]) ;
            NTRUE[i]  +=  EMIT[i] * TT[i] ;
         }
         
         barrier(CLK_LOCAL_MEM_FENCE) ;
         // NTRUE[] updated only by this work group
         // barrier(CLK_GLOBAL_MEM_FENCE) ;
         
         // sij[LOCAL*itran0+lid] -> ARES[...itran*CELLS+INDEX]
         for(int itran0=lid; itran0<NCMP; itran0+=LOCAL) {
            itran = TRAN[itran0] ;
            tmp   = 0.0f ;
            for(int i=0; i<LOCAL; i++) tmp += sij[itran0*LOCAL+i] ;
            ARES[gid*TRANSITIONS*CELLS+itran*CELLS+INDEX].x  += tmp ; // only this WG updates this element
            tmp   = 0.0f ;
            for(int i=0; i<LOCAL; i++) tmp += esc[itran0*LOCAL+i] ;
            ARES[gid*TRANSITIONS*CELLS+itran*CELLS+INDEX].y  += tmp ; // only this WG updates this element
         }
         
         barrier(CLK_LOCAL_MEM_FENCE) ;
         // each element of ARES updated only by a single work group
         // barrier(CLK_GLOBAL_MEM_FENCE) ;
         
         // next cell
         INDEX += dstep ;
         if (INDEX>=CELLS) break ;  // going outwards and outermost shell was processed
         if (INDEX<0) {             // we went through the centre cell, turn outwards
            INDEX = 1 ;  dstep = +1 ;
         } else {
            if (STEP[iray*CELLS+INDEX]<=0.0f) { // ray does not go that far in, turn outwards
               INDEX += 2 ;  dstep  = +1 ;
               if (INDEX>=CELLS) break ;       // went tangentially through the outermost shell
            }
         }
      } // while(1) --- loop through the cloud
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
      barrier(CLK_GLOBAL_MEM_FENCE) ;
      
   } // for iray
   
}




__kernel void Spectra_OL(
                         const float      FREQ,    //  0 
                         __global float4 *CLOUD,   //  1 [CELLS]: Vrad, Rc, dummy, sigma
                         __global float  *GAU,     //  2 precalculated gaussian profiles [CELLS*CHANNELS]
                         __global int2   *LIM,     //  3 first and last significant channels
                         __global float  *GN,      //  4 [TRANSITIONS]
                         __global float  *NU,      //  5 Nu  [CELLS*TRANSITIONS]
                         __global float  *Aul,     //  6 Aul [TRANSITIONS]
                         __global float  *NBNB,    //  7 nb_nb [TRANSITIONS*CELLS]
                         __global float  *STEP,    //  8 step lengths (GL)
                         __global float  *IP,      //  9 impact parameter [NRAY] [GL]
                         __global float  *BG,      // 10 background value (photons per ray)
                         __global float  *NTRUES,  // 11 [NRAY*MAXCHN] ... without overlap
                         __global float  *STAUS,   // 12 [NRAY*CHANNELS]
                         const int        NCMP,    // 13 number of lines in this band
                         const int        NCHN,    // 14 channels in combined spectrum
                         __global int    *TRAN,    // 15 list of transitions
                         __global float  *OFF,     // 16 offsets in channels
                         __global float  *WRK,     // 17 for LOWMEM>1 only, work space [NWG*MAXCHN*3]
#if (WITH_CRT>0)
                         __global  float *CRT_TAU, // 18  [CELLS, TRANSITIONS]
                         __global  float *CRT_EMI, // 19  [CELLS, TRANSITIONS]
#endif
                         const int        savetau
                        )  {
   // Write spectra in case of overlapping lines; one work group per ray
   int id  = get_global_id(0) ;
   int lid = get_local_id(0) ;
   int gid = get_group_id(0) ;
   if (gid>=NWG) return ;          // one work group per ray
   float doppler, dx, tmp, ip ;
   int shift, itran, INDEX, dstep ;
   
   __global float *profile ;
   __global float *NTRUE ;
   
   
#if (LOWMEM>1)  // very low memory or very large number of channels
   __global  float *EMIT = &(WRK[gid*3*MAXCHN         ]) ;
   __global  float *STAU = &(WRK[gid*3*MAXCHN+  MAXCHN]) ;
   __global  float *TAU  = &(WRK[gid*3*MAXCHN+2*MAXCHN]) ;
#else
   __local  float  EMIT[MAXCHN] ;
   __local  float  STAU[MAXCHN] ;
   __local  float  TAU[MAXCHN] ;
#endif
   
   // n +=  [ h*f/(4*pi)  * Aul * I2T * nu*dx*GL*pro*GN + Cemit] * ((1-exp(-tau))/tau)*exp(-tau)
   //    =  [gamma * f*GN*nu*Aul*dx*pro + Cemit] * ((1-exp(-tau))/tau)*exp(-tau)
   // gamma =  PLANCK*C_LIGHT**2*GL / (8*pi*BOTZMANN*freq)
   // float gamma =  (GL/FREQ) * PLANCK*C_LIGHT*C_LIGHT / (8.0*PI*BOLTZMANN) ;
   const float gamma = (GL/FREQ) * 1.7162254e+09f ;
   //  emissivity =  gamma* nu*Aul*dx*pro*GN   
   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE) ;            
   
   
   // one work group per ray, loop over rays
   // below always lid ~ NTRUE[i]
   // for EMIT, STAU, TAU,  lid ~ [i], in the middle lid ~ [i-i1]
   // if   LOWMEM<=1, it is therefore enough to have    barrier(CLK_LOCAL_MEM_FENCE),
   // with LOWMEM==2  we need barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE) ??
   
   
   for(int iray=gid; iray<NRAY; iray+=NWG) {
      
      NTRUE  =  &NTRUES[iray*NCHN] ;      
      for(int i=lid; i<NCHN; i+=LOCAL) STAU[i]  = 0.0f ; // shared by this WG, one work item per element !!
      for(int i=lid; i<NCHN; i+=LOCAL) NTRUE[i] = 0.0f ; // shared by this WG
      
      INDEX =  CELLS-1 ; 
      dstep = -1 ; 
      ip    =  IP[iray] ;
      
      while (1) {
         
         barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE) ;         
         
         dx       =  STEP[iray*CELLS+INDEX] ;  // [GL]
         doppler  =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(ip/CLOUD[INDEX].y,2.0f)) ) ;
         profile  =  &GAU[INDEX*CHANNELS] ;
         // sum of component optical depths => TAU

#if (KILL_EMISSION<CELLS)
# if (WITH_CRT>0)
         for(int i=lid; i<NCHN; i+=LOCAL) {
            TAU[i]  =  CRT_TAU[INDEX*TRANSITIONS+TRAN[0]] * dx ;
            EMIT[i] =  (INDEX>=KILL_EMISSION) ?  (0.0f) :  (CRT_EMI[INDEX*TRANSITIONS+TRAN[0]] * dx * GL) ;
         }
# else
         for(int i=lid; i<NCHN; i+=LOCAL) {
            TAU[i]  = 0.0f ;       // optical depth of current cell
            EMIT[i] = 0.0f ;       // emission of current cell
         }
# endif
#else      // ni KILL_EMISSION
# if (WITH_CRT>0)
         for(int i=lid; i<NCHN; i+=LOCAL) {
            TAU[i]  =  CRT_TAU[INDEX*TRANSITIONS+TRAN[0]] * dx ;
            EMIT[i] =  CRT_EMI[INDEX*TRANSITIONS+TRAN[0]] * dx * GL;
         }
# else
         for(int i=lid; i<NCHN; i+=LOCAL) {
            TAU[i]  = 0.0f ;       // optical depth of current cell
            EMIT[i] = 0.0f ;       // emission of current cell
         }
# endif
#endif
         
         // above  lid ~ [i]  <-->  table element in STAU, EMIT == each element updated by the same work item
         barrier(CLK_LOCAL_MEM_FENCE) ;         
         barrier(CLK_GLOBAL_MEM_FENCE) ;
         // below we have shifts <-->   elements updated by another work item, lid ~ [i-i1]
         
         for(int itran0=0; itran0<NCMP; itran0++) {             // loop over NCMP components
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE) ;
            itran   = TRAN[itran0] ;                            // index of the current transition
            tmp     = dx*GL*GN[itran]*NBNB[INDEX*TRANSITIONS+itran] ; // optical depth; GN **NOT* including GL
            shift   = round(doppler/WIDTH) + OFF[itran0] ;      // offset of the first channel
            int i1  = max(    -shift,  LIM[INDEX].x) ;          // LIM = first and last significant channel
            int i2  = min(NCHN-shift,  LIM[INDEX].y) ;
            for(int i=(i1+lid); i<i2; i+=LOCAL) {               // loop over profile of this comp.
               TAU[i+shift]  +=  tmp*profile[i] ;
            }
            tmp     = gamma*NU[INDEX*TRANSITIONS+itran]*Aul[itran]*dx*GN[itran] ;
#if (KILL_EMISSION<CELLS)
            if (INDEX>=KILL_EMISSION) tmp = 0.0f ;
#endif
            for(int i=(i1+lid); i<i2; i+=LOCAL) {               // over present component profile only
               EMIT[i+shift]  += tmp*profile[i] ;               // photons in one channel / cm3
            }
         }
         barrier(CLK_LOCAL_MEM_FENCE) ;
         barrier(CLK_GLOBAL_MEM_FENCE) ;
         // below again  lid ~ [i]
         for(int i=lid; i<NCHN; i+=LOCAL) {           // over the full band
            tmp       =  TAU[i] ;                     // local optical depth - lines + continuum
            tmp       =  (fabs(tmp)>1.0e-5f) ? ((1.0f-exp(-tmp))/tmp) : (1.0f-0.5f*tmp) ;
            NTRUE[i] +=  EMIT[i]*tmp*exp(-STAU[i]) ;  // observed photons         --- always same work item
            STAU[i]  +=  TAU[i] ;                     // cumulative optical depth --- always same work item
         }         
         // next cell
         INDEX += dstep ;
         if (INDEX>=CELLS) break ;  // going outwards and outermost shell was processed
         if (INDEX<0) {             // we went through the centre cell, turn outwards
            INDEX = 1 ;  dstep = 1 ;
         } else {
            if (STEP[iray*CELLS+INDEX]<=0.0f) { // ray does not go further in, turn outwards
               INDEX += 2 ; // we went through i but not into i-1 => continue with shell i+1
               dstep  = 1 ;
               if (INDEX>=CELLS) break ;
            }
         }
      } // while(1)
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
      barrier(CLK_GLOBAL_MEM_FENCE) ;
      
      if (savetau>0) {
         for (int i=lid; i<NCHN; i+=LOCAL)  NTRUE[i]  = STAU[i] ;
      } else {
         for (int i=lid; i<NCHN; i+=LOCAL)  NTRUE[i] -= BG[TRAN[0]]*(1.0f-exp(-STAU[i])) ;
      }
      
      barrier(CLK_GLOBAL_MEM_FENCE) ;
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
      // if (iray==0) printf("CENTRE TAU %10.3e   BG[%d] %10.3e\n", STAU[NCHN/2], TRAN[0], BG[TRAN[0]]) ;
      
   } // for iray
   
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
                      __global float   *WRK,           // 19 WRK[BATCH,LEVELS*(LEVELS+1)]
                      __global float   *PC             // 20 PC[BATCH], PATH_CORRECTION
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
   
#if 0
   if (id==0) {
      printf("--------------------------------------------------------------------------------\n") ;
      for(int i=0; i<TRANSITIONS; i++) printf(" %12.4e", SIJ[i]) ;  printf("\n") ;
      for(int i=0; i<TRANSITIONS; i++) printf(" %12.4e", ESC[i]) ;  printf("\n") ;
      printf("--------------------------------------------------------------------------------\n") ;
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
         // if (id==0) printf("%d / %d   ... PARTNERS %d\n", u, NCUL, PARTNERS) ;
         tmp = 0.0f ;
         if (u>=NCUL) {  // not all transitions have collisional coefficients ???
            MATRIX[IDX(j,i)] = 0.0f  ; 
            MATRIX[IDX(i,j)] = 0.0f ;
         } else {
            for(int p=0; p<PARTNERS; p++) { // get_C has the correct row from C, NTKIN element vector DOWNWARDS !!
               tmp += CABU[p]*get_C(TKIN[id], NTKIN, &MOL_TKIN[p*NTKIN], &C[p*NCUL*NTKIN + u*NTKIN]) ;
            }
            MATRIX[IDX(j,i)] = tmp*RHO[id] ;    //  IDX(j,i) = transition j <-- i  == downwards
            // the corresponding element for UPWARD transition  j -> i
            tmp  *=  (G[i]/G[j]) *  exp(-H_K*(E[i]-E[j])/TKIN[id]) ;
            MATRIX[IDX(i,j)] = tmp*RHO[id] ;    //  IDX(j,i) = transition j <-- i
         }
      }
   }
   
   
#if 0
   if (id==0) {
      printf("0xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n") ;
      for(int j=0; j<LEVELS; j++) {      // row
         for(int i=0; i<LEVELS; i++) {   // column
            printf(" %10.3e", MATRIX[IDX(j,i)]) ;
         }
         printf("\n") ;
      }
      printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n") ;
   }
#endif
   
   // #define USE_ESC 1
#if (WITH_ALI>0)
   for(int t=0; t<TRANSITIONS; t++) {  // modified Einstein A
      u = UL[2*t] ;  l = UL[2*t+1] ;
      // MATRIX[IDX(l,u)]  +=  ESC[id*TRANSITIONS+t] / (VOLUME*NI[id*LEVELS+u]) ;
# if 0
      MATRIX[IDX(l,u)]  +=  ESC[id*TRANSITIONS+t] / (NI[id*LEVELS+u]) ;  // 1D has no division by VOLUME !!
# else
      tmp = NI[id*LEVELS+u] ;
      if (tmp>1.0e-30)  MATRIX[IDX(l,u)]  +=  ESC[id*TRANSITIONS+t] / tmp ;  // 1D has no division by VOLUME !!
# endif
   }
#else
   for(int t=0; t<TRANSITIONS; t++) 
     MATRIX[IDX(UL[2*t+1], UL[2*t])] += A[t] ;  // MATRIX[l,u]
#endif   
   // --- NI is no longer used so we can reuse it to store vector X below ---
   
   
#if 0
   if (id==0) {
      printf("1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n") ;
      for(int j=0; j<LEVELS; j++) {      // row
         for(int i=0; i<LEVELS; i++) {   // column
            printf(" %10.3e", MATRIX[IDX(j,i)]) ;
         }
         printf("\n") ;
      }
      printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n") ;
   }
#endif
   
   
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
      printf("3xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n") ;
      for(int j=0; j<LEVELS; j++) {      // row
         for(int i=0; i<LEVELS; i++) {   // column
            printf(" %10.3e", MATRIX[IDX(j,i)]) ;
         }
         printf("\n") ;
      }
      printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n") ;
   }
#endif
   
   u  = Doolittle_LU_Decomposition_with_Pivoting(MATRIX, pivot,    LEVELS) ;
   u *= Doolittle_LU_with_Pivoting_Solve(MATRIX, VECTOR, pivot, V, LEVELS) ;   
   tmp = 0.0f ;   for(int i=0; i<LEVELS; i++)  tmp  +=  V[i] ;
   for(int i=0; i<LEVELS; i++)  B[i]  =  V[i]*RHO[id]*ABU[id] / tmp ;  // B ro, X rw
   
   
#if 0
   if (id==0) { // id=0, first batch == cell 0
      printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n") ;
      for(int i=0; i<LEVELS; i++) printf(" %10.3e", B[i]) ;
      printf("\n") ;
      printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n") ;
   }
#endif
   
}



