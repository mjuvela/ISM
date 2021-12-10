
__kernel void Clear(__global float2 *ARES) {
   // Clear the work array ARES[NWG*NTRAN*CELLS]
   int id = get_global_id(0) ;
   for(int i=id; i<NWG*NTRAN*CELLS; i+=GLOBAL) {
      ARES[i] = (float2)(0.0f, 0.0f) ;
   }
}


__kernel void Sum(__global float2 *ARES, __global float2 *RES, __global float *VOLUME) {
   // Sum over results of individual work groups
   //    ARES[NWG * NTRAN * CELLS]  ->  RES[CELLS * NTRAN]
   int id = get_global_id(0) ;  // CELLS
   if (id>=CELLS) return ;
   for(int itran=0; itran<NTRAN; itran++) {
      float sij=0.0f, esc=0.0f ;
      for(int i=0; i<NWG; i++) {
         sij += ARES[i*NTRAN*CELLS+itran*CELLS+id].x ;
         esc += ARES[i*NTRAN*CELLS+itran*CELLS+id].y ;
      }
      RES[id*NTRAN+itran].x = sij / VOLUME[id] ;
      RES[id*NTRAN+itran].y = esc / VOLUME[id] ;
   }
}



__kernel void Update(
                     __global float4 *CLOUD,   //  0 [CELLS]: Vrad, Rc, dummy, sigma
                     __global float  *VOLUME,  //  1 cell volume / cloud volume
                     __global float  *Aul,     //  2  A_ul[NTRAN]
                     __global float  *Blu,     //  3  old A_b [NTRAN] = Blu*(h*f)/(4*pi)
                     __global float  *GAU,     //  4 precalculated gaussian profiles [CELLS*CHANNELS]
                     __global int2   *LIM,     //  5 first and last significant channels
                     __global float  *GN,      //  6 [NTRAN]
                     __global float  *DIRWEI,  //  7 individual weights for the rays
                     __global float  *STEP,    //  8 average path length [NRAY*CELLS] (GL)
                     __global float  *APL,     //  9 average path length [CELLS] (GL)
                     __global float  *IP,      // 10 impact parameter [NRAY] [GL]
                     __global float  *BG,      // 11 background value (photons per ray)
                     __global float  *NU,      // 12 Nu    [CELLS*NTRAN]
                     __global float  *NBNB,    // 13 nb_nb [CELLS*NTRAN]
                     __global float2 *ARES,    // 14 [NWG*NTRAN*CELLS]:  SIJ, ESC
                     __global int    *SINGLE,  // 15 list of single transitions
                     __global float  *NTRUES   // 16 global work space
                    )  {
   // Update normal single transitions
   // WG loops over NRAY/NWG rays
   //      loop over transitions
   //         loop over channels, updating local soc[], loc[]
   //         add soc[] and loc[] and finally update ARES[NWG*NTRAN*CELLS]
   float weight, dx, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;
   float sum_delta_true, all_escaped, nu ;
   int   shift, INDEX, dstep ;
   int   id  = get_global_id(0) ;
   int   lid = get_local_id(0) ;   // index within the local work group
   int   gid = get_group_id(0) ;
   if (gid>=NWG) return ;  // NWG work groups
   
#define BYTRAN 1 // divide by transition rather than by channel
#if (BYTRAN==0)
   broken ???
   __local  float sij[NTRAN*LOCAL] ;    // used by 1 WG = 1 ray
   __local  float esc[NTRAN*LOCAL] ;
#endif
   
   // NTRUE must be at least NWG*NTRAN*CHANNELS
#define LOCAL_NTRUE 1 // no effect on GPU run time
#if (LOCAL_NTRUE==1)
   __local  float NTRUE[NTRAN*CHANNELS] ;
   __local  float *ntrue ;
#else
   __global float *NTRUE = &NTRUES[gid*NTRAN*CHANNELS] ;
   __global float *ntrue ;
#endif
   
#define LOCAL_PROFILE 1  // 1 = faster
#if (LOCAL_PROFILE==1)
   __local  float profile[CHANNELS] ;   // current cell, used by all work items, ~1kB
#else
   __global float *profile ;
#endif
   
   
   for(int iray=gid; iray<NRAY; iray+=NWG) {
      INDEX = CELLS-1 ;   // always starts with the outermost ray
      dstep = -1 ;        // we first go inwards (dstep<0), then outwards until ray exits
      for(int itran=0; itran<NTRAN; itran++) {
         for(int i=lid; i<CHANNELS; i+=LOCAL) NTRUE[itran*CHANNELS+i] = DIRWEI[iray] * BG[itran] ;
      }
      barrier(CLK_LOCAL_MEM_FENCE) ;
      while(1) { // follow one ray through the cloud
         dx        =  STEP[iray*CELLS+INDEX] ;  // [GL]
         weight    =  DIRWEI[iray]*(dx/APL[INDEX])*VOLUME[INDEX] ;  // VOLUME == fraction of cloud volume
         doppler   =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(IP[iray]/CLOUD[INDEX].y, 2.0f)) ) ;
         // a single profile vector used by all transitions, the whole work group
#if(LOCAL_PROFILE==1)
         for(int i=lid; i<CHANNELS; i+=LOCAL) profile[i] = GAU[INDEX*CHANNELS+i] ;
#else
         profile = &GAU[INDEX*CHANNELS] ;
#endif
         barrier(CLK_LOCAL_MEM_FENCE) ;
         shift   =  round(doppler/WIDTH) ;
         int i1  =  max(shift,          LIM[INDEX].x) ;
         int i2  =  min(CHANNELS, shift+LIM[INDEX].y) ;
         
#if (BYTRAN==0)
         for(int itran=0; itran<NTRAN; itran++) {
            if (SINGLE[itran]==0) continue ;
            ntrue          =  &NTRUE[itran*CHANNELS] ;
            nu             =    NU[INDEX*NTRAN+itran] ;  // [CELLS*NTRAN]
            nb_nb          =  NBNB[INDEX*NTRAN+itran] ;
            if (fabs(nb_nb)<1.0e-37f) nb_nb=1.0e-37f ;
            tmp_tau        =  dx*GL*nb_nb*GN[itran] ;    // GN DOES NOT include GL [cm]
            tmp_emit       =  weight * nu*Aul[itran] / tmp_tau ;
            sum_delta_true =  all_escaped = 0.0f ;
            for(int ii=(i1+lid); ii<i2; ii+=LOCAL)  {
# if 0
               factor           =  1.0f-exp(-tmp_tau*profile[ii-shift]) ;
#else
               nu               =  tmp_tau*profile[ii-shift] ;
               factor           = (fabs(nu)>1.0e-4f) ? (1.0f-exp(-nu)) : (nu) ;
#endif
               escape           =  tmp_emit*factor ;     // emitted photons that escape current cell
               absorbed         =  ntrue[ii]*factor ;    // incoming photons that are absorbed
               ntrue[ii]       +=  escape-absorbed ;
               sum_delta_true  +=  absorbed  ;           // ignore photons absorbed in emitting cell
               all_escaped     +=  escape ;              // sum of escaping photons over the profile
            }
            sij[itran*LOCAL+lid] =  Blu[itran] * sum_delta_true / nb_nb  ;
            esc[itran*LOCAL+lid] =  all_escaped ;
         } // NTRAN
         barrier(CLK_LOCAL_MEM_FENCE) ;
         // sum over sij and esc -> ARES; one work item = one transition
         for(int itran=lid; itran<NTRAN; itran+=LOCAL) {
            dx = 0.0f ; factor = 0.0f ;
            for(int i=0; i<LOCAL; i++) {
               dx      += sij[itran*LOCAL+i] ;
               factor  += esc[itran*LOCAL+i] ;
            }
            ARES[gid*NTRAN*CELLS+itran*CELLS+INDEX] += (float2)(dx, factor) ;
         }
#else
         // loop over transitions WORK ITEM PER LEVEL
         for(int itran=lid; itran<NTRAN; itran+=LOCAL) {
            if (SINGLE[itran]==0) continue ;
            ntrue     =  &NTRUE[itran*CHANNELS] ;
            nu        =   NU[INDEX*NTRAN+itran] ;    // [CELLS*NTRAN]
            nb_nb     =   NBNB[INDEX*NTRAN+itran] ;
            if (fabs(nb_nb)<1.0e-37f) nb_nb=1.0e-37f ;
            tmp_tau   =  dx*GL*nb_nb*GN[itran] ;      // GN DOES NOT include GL [cm]
            tmp_emit  =  weight * nu*Aul[itran] / tmp_tau ;
            sum_delta_true = all_escaped = 0.0f ;
            // each work item processes one channel at a time
            for(int ii=i1; ii<i2; ii++)  {
# if 0         // problems at small optical depths
               factor           =  1.0f-exp(-tmp_tau*profile[ii-shift]) ;
# else
               nu               =  tmp_tau*profile[ii-shift] ;
               factor           =  (fabs(nu)>1.0e-4) ? (1.0f-exp(-nu)) : (nu) ;
# endif
               escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
               absorbed         =  ntrue[ii]*factor ;   // incoming photons that are absorbed
               ntrue[ii]       +=  escape-absorbed ;
               sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
               all_escaped     +=  escape ;             // sum of escaping photons over the profile
            } 
            // update SIJ and ESC in local array
            ARES[gid*NTRAN*CELLS+itran*CELLS+INDEX].x +=  Blu[itran] * sum_delta_true / nb_nb  ;
            ARES[gid*NTRAN*CELLS+itran*CELLS+INDEX].y +=  all_escaped ;
         } // loop over transitions
#endif  // BYTRAN
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
                      __global float4 *CLOUD,   //  0 [CELLS]: Vrad, Rc, dummy, sigma
                      __global float  *GAU,     //  1 precalculated gaussian profiles [CELLS*CHANNELS]
                      __global int2   *LIM,     //  2 first and last significant channels
                      __global float  *NU,      //  3 Nu [NTRAN*CELLS]
                      __global float  *NBNB,    //  4 nb_nb [NTRAN*CELLS]
                      __global float  *STEP,    //  5 average path length [NRAY*CELLS] (GL)
                      __global float  *IP,      //  6 impact parameter [NRAY] [GL]
                      const    float   BG,      //  7 background value (photons per ray)
                      const    int     itran,   //  8 the transition
                      const    float   EMI,     //  9 h/(4*pi)*freq*Aul*int2temp
                      const    float   GN,      // 10 GL **NOT** included
                      __global float  *NTRUES,  // 11 [NRAY*CHANNELS] ... without overlap
                      __global float  *STAUS    // 12 [NRAY*CHANNELS]emi
                     )
{
   // Normal spectra without overlapping lines
   // One work item per ray; the same rays as used in the actual calculation!
   // NTRUES[] must be at least  NRAY*CHANNELS
   int  id = get_global_id(0) ;
   int lid = get_local_id(0) ;
   if (id>=NRAY) return ; // no more rays
#if 1
   __global float *NTRUE = &(NTRUES[id*CHANNELS]) ; // NRAY*CHANNELS
#else
   __local float LNTRUES[LOCAL*CHANNELS] ;
   __local float *NTRUE = &LNTRUES[lid*CHANNELS] ;
#endif
#if 1
   __global float *STAU  = &( STAUS[id*CHANNELS]) ;
#else
   __local float LSTAUS[LOCAL*CHANNELS] ;
   __local float *STAU = &LSTAUS[lid*CHANNELS] ;
#endif
   int i ;
   for(int i=0; i<CHANNELS; i++) {
      NTRUE[i]   = 0.0f ;  STAU[i] = 0.0f ;
   }
   float tau, dtau, emissivity, doppler, nu, dx ;
   int shift ;
   __global float* profile ;
   float ip = IP[id] ; // impact parameter [GL] ;
   int INDEX=CELLS-1, dstep = -1 ;
   // emissivity = (PLANCK/(4.0*PI))*freq*A_Aul*C_LIGHT*C_LIGHT/(2.0*BOLZMANN*freq*freq) ;
   while (1) {
      dx       =  STEP[id*CELLS+INDEX] ;  // [GL]
      nu       =  NU[INDEX*NTRAN+itran] ;     
      tau      =  (fabs(NBNB[INDEX*NTRAN+itran])<1.0e-28f) ? (1.0e-28f) :  (NBNB[INDEX*NTRAN+itran]) ;
      tau     *=  GN*GL*dx ;   // 1e5*1e18 = 1e23?
      tau      =  clamp(tau, -2.0f, 1.0e10f) ;      
      doppler  =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(ip/CLOUD[INDEX].y,2.0f)) ) ;
      profile  =  &GAU[INDEX*CHANNELS] ;
      shift    =  round(doppler/WIDTH) ;
      // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      emissivity =  EMI * nu * dx * GL ;
      //    i in [0, CHANNELS[
      //    i-shift in [0,CHANNELS[  => i in [+shift, CHANNELS+shift[
      for(i=max(0, shift); i<min(CHANNELS, CHANNELS+shift); i++) {         
         dtau       =  tau*profile[i-shift] ;
         dx         =  emissivity*profile[i-shift]*GN*exp(-STAU[i]) ;
         dx        *=  (fabs(dtau)>1.0e-5f) ? ((1.0f-exp(-dtau))/dtau) : (1.0f-0.5f*dtau) ;
         NTRUE[i]  +=  dx ;
         STAU[i]   +=  dtau ;
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
      if (INDEX>=CELLS) break ;
   } // while(1)
   for (i=0; i<CHANNELS; i++) NTRUE[i] -=  BG*(1.0f-exp(-STAU[i])) ;

   // if (id==0) printf("CENTRE TAU %10.3e\n", STAU[CHANNELS/2]) ;
}




// ===========================================================================================
   


__kernel void Overlap(
                      __global   float4 *CLOUD,   //  0 [CELLS]: Vrad, Rc, dummy, sigma
                      __global   float  *VOLUME,  //  1 cell volume / cloud volume
                      __constant float  *Aul,     //  2 Aul [NTRAN]
                      __constant float  *BLU,     //  3 A_b [NTRAN] = B_lu * h*f / (4*pi)
                      __global   float  *GAU,     //  4 precalculated gaussian profiles [CELLS*CHANNELS]
                      __constant int2   *LIM,     //  5 first and last significant channels
                      __constant float  *GN,      //  6 gauss normalisation [NTRAN] -- GL **NOT** inlcuded
                      __global   float  *DIRWEI,  //  7 individual weights for the rays
                      __global   float  *STEP,    //  8 average path length [NRAY*CELLS] (GL)
                      __global   float  *APL,     //  9 average path length [CELLS] (GL)
                      __global   float  *IP,      // 10 impact parameter [NRAY] [GL]
                      __global   float  *BG,      // 11 [NTRAN]  (photons per ray)
                      __global   float  *NU,      // 12 n_u   [NTRAN*CELLS]
                      __global   float  *NBNB,    // 13 nb_nb [NTRAN*CELLS]
                      __global   float2 *ARES,    // 14 [NWG*NTRAN*CELLS]
                      const int        NCMP,      // 15 number of lines in this band
                      const int        NCHN,      // 16 channels in combined spectrum
                      __constant int    *TRAN,    // 17 list of transitions
                      __constant float  *OFF      // 18 offsets in channels
#if (USE_LOCALS==0)
                      ,__global float  *NTRUES     // 19 [4*NRAY*CHANNELS] !!!!!!
#endif                       
                     )  {
   // Each work group loops over rays.
   // Each work item of a work group is working on the same ray at the same time.
   
   // ????????????????????????????????????????????????????????????????????????
   //  GPU sometimes works, sometimes gives wrong (different) answers for Tex
   //  was caused by LOCAL being changed on the host side?
   // ????????????????????????????????????????????????????????????????????????
   
   float weight, dx, doppler, tmp_tau, tmp, tmp_emit, emi, own, phi ;
   float s ;
   int   shift, INDEX, dstep, itran ;
   int   id  = get_global_id(0) ;  // *** not used ***
   int   lid = get_local_id(0) ;   // index within the local work group
   int   gid = get_group_id(0) ;   // work group index
   if (gid>=NWG) return ;          // NWG work groups used, may be less than allocated wgs
   
   // one kernel call = loop over rays but ONLY A SINGLE BAND
   // each loop needs TAU[MAXCHN], STAU[MAXCHN] that do fit in local

   
#if (USE_LOCALS==0)
   __global float *NTRUE = &NTRUES[           gid*NCHN] ;
   __global float *EMIT  = &NTRUES[2*NWG*NCHN+gid*NCHN] ;
   __global float *TAU   = &NTRUES[  NWG*NCHN+gid*NCHN] ;
   __global float *TT    = &NTRUES[3*NWG*NCHN+gid*NCHN] ;
#else
   // not significant effect... from 18 to 15 seconds
   __local float  EMIT[MAXCHN] ;
   __local float  TAU[MAXCHN] ;
   __local float  NTRUE[MAXCHN] ;
   __local float  TT[MAXCHN] ;
   if (NCHN>MAXCHN) {
      // printf("NOT ENOUGH CHANNELS ALLOCATED IN KERNEL OVERLAP !!!\n") ;
      ARES[gid*NTRAN*CELLS+itran*CELLS+INDEX].x = +1.0e30f ;
      ARES[gid*NTRAN*CELLS+itran*CELLS+INDEX].y = -1.0e30f ;
      return ;
   }
#endif
   
   __local  float sij[MAXCMP*LOCAL] ; // 32*20 < 3kB
   __local  float esc[MAXCMP*LOCAL] ; // only NCMP*LOCAL used
   __local  float profile[CHANNELS] ;
   

   float bg = BG[TRAN[0]] ;
   
   
   for(int iray=gid; iray<NRAY; iray+=NWG) { // work group loops over rays
      
      INDEX  = CELLS-1 ;   // always starts with the outermost ray
      dstep  = -1 ;        // we first go inwards (dstep<0), then outwards until ray exits
      // for(int i=lid; i<NCHN; i+=LOCAL)  NTRUE[i] = BG[TRAN[0]]*DIRWEI[iray]  ;
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
         for(int i=lid; i<CHANNELS; i+=LOCAL) profile[i] = GAU[INDEX*CHANNELS+i] ;
         for(int i=lid; i<NCHN; i+= LOCAL)      TAU[i]  = 0.0f ;
         for(int i=lid; i<NCHN; i+= LOCAL)      EMIT[i] = 0.0f ;
         for(int i=lid; i<NCMP*LOCAL; i+=LOCAL) sij[i]  = 0.0f ;
         for(int i=lid; i<NCMP*LOCAL; i+=LOCAL) esc[i]  = 0.0f ;
         barrier(CLK_LOCAL_MEM_FENCE) ;
         // TAU[], EMIT[], sij[], esc[] updated only by this work group
         // barrier(CLK_GLOBAL_MEM_FENCE) ;
         
         //    optical depth:   tau == nb_nb*s*GN*GL
         // phi  =   profile[]*GN = actual profile function
         // tau  =   nb_nb*s*GN*profile[]
         
         // sum of component optical depths => TAU
         for(int itran0=0; itran0<NCMP; itran0++) { // loop over NCMP components
            itran    =  TRAN[itran0] ;
            tmp_tau  =  s * NBNB[INDEX*NTRAN+itran] * GN[itran] ;
            shift    =  round(doppler/WIDTH) + OFF[itran0] ;  // offset, current component
            int i1   =  max(    -shift,  LIM[INDEX].x) ;
            int i2   =  min(NCHN-shift,  LIM[INDEX].y) ;
            for(int i=(i1+lid); i<i2; i+=LOCAL) {  // over profile
               TAU[i+shift] += tmp_tau*profile[i] ;
            }
            // calculate the vector of total emission: SUM(nu*Aul*phi)
            // nu*Aul*phi*s => weight*nu*Aul == true photons from cell / package / cloud volume
            tmp  = weight * NU[INDEX*NTRAN+itran] * Aul[itran] ;
            for(int i=(i1+lid); i<i2; i+=LOCAL) {     //  phi * dnu  =  profile*GN * (1/GN) == profile
               EMIT[i+shift] +=   tmp * profile[i] ;  // photons within one channel
            }
            // TAU[], EMIT[] updated only by this work group
            barrier(CLK_LOCAL_MEM_FENCE) ;  // next loop, element updated by different work item?
            // barrier(CLK_GLOBAL_MEM_FENCE) ; // next loop, element updated by different work item?
         }
         
         // calculate (1-exp(-TAU))/TAU for all channels
         for(int i=lid; i<NCHN; i+=LOCAL) {
            TT[i] =  (fabs(TAU[i])>1.0e-5f) ? ((1.0f-exp(-TAU[i]))/TAU[i]) : (1.0f-0.5f*TAU[i]) ;
         }

         // TT[] updated only by this work group
         // barrier(CLK_GLOBAL_MEM_FENCE) ;  // make sure all have the final TT
         barrier(CLK_LOCAL_MEM_FENCE) ;
         
         // count absorptions of incoming photons:  BLU*ntrue*phi*(1-exp(-tau))/tau
         for(int itran0=0; itran0<NCMP; itran0++) { // loop over NCMP components
            itran   =  TRAN[itran0] ;
            shift   =  round(doppler/WIDTH) + OFF[itran0] ;  // offset, current component
            int i1  =  max(    -shift,  LIM[INDEX].x) ;
            int i2  =  min(NCHN-shift,  LIM[INDEX].y) ;
            for(int i=(i1+lid); i<i2; i+=LOCAL) {     // over the profile of this transition
               phi                    = GN[itran]*profile[i] ;  // PHI (absorption)
               // TT  = (1-exp(-tau))/tau
               // tau = NBNB * s * GN * profile
               sij[itran0*LOCAL+lid] += NTRUE[i+shift]*BLU[itran]*phi*s*TT[i+shift] ;
            }
         }

         // sij[] updated only by this work group
         // barrier(CLK_GLOBAL_MEM_FENCE) ;
         barrier(CLK_LOCAL_MEM_FENCE) ;
         
         // emission-absorption -- each element updated by the same work item as above
         for(int itran0=0; itran0<NCMP; itran0++) {
            itran     =  TRAN[itran0] ;
            shift     =  round(doppler/WIDTH) + OFF[itran0] ;  // offset, current component
            int i1    =  max(    -shift,  LIM[INDEX].x) ;
            int i2    =  min(NCHN-shift,  LIM[INDEX].y) ;
            for(int i=(i1+lid); i<i2; i+=LOCAL) {   // over the profile of transition itran
               emi      =  weight*NU[INDEX*NTRAN+itran]*Aul[itran]*profile[i] ; // emitted in one channel
               phi      =  profile[i] * GN[itran] ; // profile function in current channel = phi(freq)
               tmp_tau  =  s*NBNB[INDEX*NTRAN+itran]*phi ;
               // SIJ  -- count absorptions ignoring photons from current line
#if 0
               tmp      =  (EMIT[i+shift]-emi) * BLU[itran]*phi*s*(1.0f-TT[i+shift])/TAU[i+shift] ;
# else
               tmp      =  (EMIT[i+shift]-emi) * BLU[itran]*phi*s ;
               tmp     *=  (fabs(TAU[i+shift])>1.0e-3f)
                 ?    ((1.0f-TT[i+shift])/TAU[i+shift])   :    ( 0.5f-TAU[i+shift]/6.0f) ;
# endif
               sij[itran0*LOCAL+lid] +=  tmp ;
               
               // Escape = W*nu*Aul*profile * ((1-exp(-TAU))/TAU) * (tau/TAU))
               //        = emi              *  TT         *  tmp_tau/TAU
               // if there were no other lines:
               //    esc[itran0*LOCAL+lid] +=  emi * TT[i+shift] * tmp_tau / TAU[i+shift] ;
               // taking into account overlapping:
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
            ARES[gid*NTRAN*CELLS+itran*CELLS+INDEX].x  += tmp ; // only this WG updates this element
            tmp   = 0.0f ;
            for(int i=0; i<LOCAL; i++) tmp += esc[itran0*LOCAL+i] ;
            ARES[gid*NTRAN*CELLS+itran*CELLS+INDEX].y  += tmp ; // only this WG updates this element
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
                         __global float4 *CLOUD,   //  0 [CELLS]: Vrad, Rc, dummy, sigma
                         __global float  *GAU,     //  1 precalculated gaussian profiles [CELLS*CHANNELS]
                         __global int2   *LIM,     //  2 first and last significant channels
                         __global float  *GN,      //  3 precalculated gaussian profiles [CELLS*CHANNELS]
                         __global float  *NU,      //  4 Nu  [CELLS*NTRAN]
                         __global float  *Aul,     //  5 Aul [NTRAN]
                         __global float  *NBNB,    //  6 nb_nb [NTRAN*CELLS]
                         __global float  *STEP,    //  7 step lengths (GL)
                         __global float  *IP,      //  8 impact parameter [NRAY] [GL]
                         __global float  *BG,      //  9 background value (photons per ray)
                         __global float  *NTRUES,  // 10 [NRAY*CHANNELS] ... without overlap
                         __global float  *STAUS,   // 11 [NRAY*CHANNELS]emi
                         const int        NCMP,    // 12 number of lines in this band
                         const int        NCHN,    // 13 channels in combined spectrum
                         __global int    *TRAN,    // 14 list of transitions
                         __global float  *OFF      // 15 offsets in channels
                        )
{
   // Write spectra in case of overlapping lines; one work group per ray
   int id  = get_global_id(0) ;
   int lid = get_local_id(0) ;
   int gid = get_group_id(0) ;
   if (gid>=NWG) return ;
   float doppler, dx, tmp, ip ;
   int shift, itran, INDEX, dstep ;
   
   __global float *profile ;
   __global float *NTRUE ;
#if 0
   __global float *EMIT  = &NTRUES[(2*gid+1)*NCHN] ;
   __global float *STAU  = &(STAUS[gid*NCHN]) ;
#else
   __local float EMIT[MAXCHN] ;
   __local float STAU[MAXCHN] ;
#endif
   __local  float TAU[MAXCHN] ;
   
   barrier(CLK_GLOBAL_MEM_FENCE || CLK_LOCAL_MEM_FENCE) ;         
   
   
   // one work group per ray, loop over rays
   for(int iray=gid; iray<NRAY; iray+=NWG) {
      
      NTRUE  =  &NTRUES[iray*NCHN] ;      
      for(int i=lid; i<NCHN; i+=LOCAL) STAU[i]  = 0.0f ; // shared by this WG
      for(int i=lid; i<NCHN; i+=LOCAL) NTRUE[i] = 0.0f ; // shared by this WG
      
      INDEX =  CELLS-1 ; 
      dstep = -1 ; 
      ip    =  IP[iray] ;
      
      while (1) {
         barrier(CLK_GLOBAL_MEM_FENCE || CLK_LOCAL_MEM_FENCE) ;         
         
         dx       =  STEP[iray*CELLS+INDEX] ;  // [GL]
         doppler  =  dstep*CLOUD[INDEX].x * sqrt( max(0.0f, 1.0f-pow(ip/CLOUD[INDEX].y,2.0f)) ) ;
         // doppler  =  0 ;   // this was a leftover from some debuggin !!! 2018-07-27
         profile  =  &GAU[INDEX*CHANNELS] ;
         
         // sum of component optical depths => TAU
         for(int i=lid; i<NCHN; i+=LOCAL) {
            TAU[i]  = 0.0f ;       // optical depth of current cell
            EMIT[i] = 0.0f ;       // emission of current cell
         }
         barrier(CLK_LOCAL_MEM_FENCE) ;         
         barrier(CLK_GLOBAL_MEM_FENCE) ;
         
         
         for(int itran0=0; itran0<NCMP; itran0++) {             // loop over NCMP components
            barrier(CLK_LOCAL_MEM_FENCE || CLK_GLOBAL_MEM_FENCE) ;
            itran   = TRAN[itran0] ;                            // index of the current transition
            tmp     = dx*GL*GN[itran]*NBNB[INDEX*NTRAN+itran] ; // optical depth; GN **NOT* including GL
            shift   = round(doppler/WIDTH) + OFF[itran0] ;      // offset of the first channel
            int i1  = max(    -shift,  LIM[INDEX].x) ;          // LIM = first and last significant channel
            int i2  = min(NCHN-shift,  LIM[INDEX].y) ;
            for(int i=(i1+lid); i<i2; i+=LOCAL) {            // loop over profile of this comp.
               TAU[i+shift]  +=  tmp*profile[i] ;
            }
            tmp     = NU[INDEX*NTRAN+itran]*Aul[itran] ;     // emitted photons / cm3 / channel
            for(int i=(i1+lid); i<i2; i+=LOCAL) {            // over present component profile only
               EMIT[i+shift]  += tmp*profile[i] ;            // photons in one channel / cm3
            }
         }
         
         
         barrier(CLK_LOCAL_MEM_FENCE) ;
         barrier(CLK_GLOBAL_MEM_FENCE) ;
         
         for(int i=lid; i<NCHN; i+=LOCAL) {           // over the full band
            tmp       =  TAU[i] ;                     // local optical depth
            tmp       =  (fabs(tmp)>1.0e-5f) ? ((1.0f-exp(-tmp))/tmp) : (1.0f-0.5f*tmp) ;
            NTRUE[i] +=  EMIT[i]*dx*GL*exp(-STAU[i])*tmp ;   // observed photons
            STAU[i]  +=  TAU[i] ;                     // cumulative optical depth
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

      for (int i=lid; i<NCHN; i+=LOCAL)  NTRUE[i] -= BG[TRAN[0]]*(1.0f-exp(-STAU[i])) ;
      
      barrier(CLK_GLOBAL_MEM_FENCE) ;
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
      // if (iray==0) printf("CENTRE TAU %10.3e   BG[%d] %10.3e\n", STAU[NCHN/2], TRAN[0], BG[TRAN[0]]) ;
      
   } // for iray

}

