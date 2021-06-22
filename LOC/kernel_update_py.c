// #define EPS 6.0e-5f  BAD !!!
// #define EPS 4.0e-4f  ---- ??? this is now defined in kernel_LOC_aux.c !!!
#define UNROLL 1
#define real float
// #define H_K  4.799243348e-11

#include "kernel_LOC_aux.c"

// - set LOC_LOWMEM=0, speedup 7.4 -> 5.1 seconds ~ 25% improvement
//   need to have local storage LOCAL*CHANNELS
//   ...  32 x 256 = 8 kB   ... probably

// Can we have GAU[GNO,CHANNELS] in constant memory ?
//        55 x 256 = 14 kB  ... possibly
#define GAUSTORE  __constant

// - GPU: first run 5.07 seconds, second run 4.46 seconds for loc_3d_co.ini / Tux
//   .... after a while back to 7.5 seconds ???
//   .... simulation from 1.70 to 2.01 seconds ???



__kernel void Clear(__global float *RES) {  // [2*CELLS] SIJ, ESC
   // Clear the arrays before simulation of the current transition
   int id = get_global_id(0) ;   
   for(int i=id; i<2*CELLS; i+=GLOBAL) RES[i] = 0.0f ;
}



int Index(const float3 pos) {  // for regular cartesian grid case
   if ((pos.x<=0.0f)||(pos.y<=0.0f)||(pos.z<=0.0f))  return -1 ;
   if ((pos.x>=  NX)||(pos.y>=  NY)||(pos.z>=  NZ))  return -1 ;
   // without (int), this FAILS for cloud sizes >256^3 !!
   return  (int)trunc(pos.x) + NX*((int)trunc(pos.y)+NY*(int)trunc(pos.z)) ;
}







__kernel void Paths(
#if (WITH_PL==1)
                    __global float   *PL,      // [CELLS]
#endif
                    __global float   *TPL,     // [NRAY]
                    __global int     *COUNT,   // [NRAY] total number of rays entering the cloud
                    const    int      LEADING, // leading edge
                    const    float3   POS0,    // initial position of ray 0
                    const    float3   DIR      // direction of the rays
                   ) {   
   float tpl, dx, dy, dz ;
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge, if ray exits through a side, a new one is created 
   // on the opposite side. Ray ends when the downstream edge is reached
   int  id  =  get_global_id(0), INDEX ;
   int  ny  =  (NY+1)/2 ;
   int  nz  =  (NZ+1)/2 ;   
   barrier(CLK_LOCAL_MEM_FENCE) ;  // barrier ????
   float3 POS = POS0 ;
#if (FIX_PATHS>0)
   float3 POS_LE ;
#endif
   float  s ;
   // NRAY is the upper limit for the rays actually needed
   if (id>=NRAY) return ;  // NRAY = ((X+1)/2) * ((Y+1)/2)
   // each ray shifted by two grid units
   
   switch (LEADING) {
    case 0: POS.x =    PEPS ;  POS.y += 2.0f*(id/nz) ;  POS.z += 2.0f*(id%nz) ;  break ;
    case 1: POS.x = NX-PEPS ;  POS.y += 2.0f*(id/nz) ;  POS.z += 2.0f*(id%nz) ;  break ;
    case 2: POS.y =    PEPS ;  POS.x += 2.0f*(id/nz) ;  POS.z += 2.0f*(id%nz) ;  break ;
    case 3: POS.y = NY-PEPS ;  POS.x += 2.0f*(id/nz) ;  POS.z += 2.0f*(id%nz) ;  break ;
    case 4: POS.z =    PEPS ;  POS.x += 2.0f*(id/ny) ;  POS.y += 2.0f*(id%ny) ;  break ;
    case 5: POS.z = NZ-PEPS ;  POS.x += 2.0f*(id/ny) ;  POS.y += 2.0f*(id%ny) ;  break ;
   }
#if (FIX_PATHS>0)
   POS_LE = POS ; // initial position of this ray
#endif
   
   COUNT[id] = 0 ;
   tpl       = 0 ;
   
   INDEX = Index(POS) ;
   if (INDEX>=0)   COUNT[id] += 1 ;
   while(INDEX>=0) {
      
      if (DIR.x<0.0f)   dx = -      fmod(POS.x, 1.0f)  / DIR.x ;  else  dx =  (1.0f-fmod(POS.x, 1.0f)) / DIR.x ;
      if (DIR.y<0.0f)   dy = -      fmod(POS.y, 1.0f)  / DIR.y ;  else  dy =  (1.0f-fmod(POS.y, 1.0f)) / DIR.y ;
      if (DIR.z<0.0f)   dz = -      fmod(POS.z, 1.0f)  / DIR.z ;  else  dz =  (1.0f-fmod(POS.z, 1.0f)) / DIR.z ;
      dx         = min(dx, min(dy, dz)) + PEPS ;
      
      
#if (WITH_PL==1)
      // PL[INDEX] += dx ;           // path length, cumulative over idir and ioff
      AADD(&(PL[INDEX]), dx) ;
#endif
      tpl       += dx ;           // just the total value for current idir, ioff
      POS       += dx*DIR ;
      
#if (FIX_PATHS>0)
      // try to precise the position
      // printf("FIX\n") ;
      if (LEADING<2) {
         if ((int)(POS.x)%10==0) {
            s      =  (LEADING==0) ?  (POS.x/DIR.x) : ((POS.x-NX)/DIR.x) ;
            POS.y  =  POS_LE.y + s*DIR.y ;
            POS.z  =  POS_LE.z + s*DIR.z ;
            if (POS.y<0.0f) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;
            if (POS.z<0.0f) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
         }
      } else {
         if (LEADING<4) {
            s      =  (LEADING==2) ?  (POS.y/DIR.y) : ((POS.y-NY)/DIR.y) ;
            POS.x  =  POS_LE.x + s*DIR.x ;
            POS.z  =  POS_LE.z + s*DIR.z ;
            if (POS.x<0.0f) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
            if (POS.z<0.0f) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
         } else {
            s      =  (LEADING==4) ?  (POS.z/DIR.z) : ((POS.z-NY)/DIR.z) ;
            POS.x  =  POS_LE.x + s*DIR.x ;
            POS.y  =  POS_LE.y + s*DIR.y ;
            if (POS.x<0.0f) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
            if (POS.y<0.0f) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;            
         }
      }
#endif
      
      INDEX      = Index(POS) ;
      
      
      
      if (INDEX<0) {  // exits the cloud... but on which side?
         if (POS.x<0.0f) {
            if (LEADING!=1)  POS.x =  NX-PEPS ;   // create new ray
         }
         if (POS.x>NX) {
            if (LEADING!=0)  POS.x =     PEPS ;   // create new ray
         }
         if (POS.y<0.0f) {
            if (LEADING!=3)  POS.y =  NY-PEPS ;   // create new ray
         } 
         if (POS.y>NY) {
            if (LEADING!=2)  POS.y =     PEPS ;   // create new ray
         }
         if (POS.z<0.0f) {
            if (LEADING!=5)  POS.z =  NZ-PEPS ;   // create new ray
         } 
         if (POS.z>NZ) {
            if (LEADING!=4)  POS.z =     PEPS ;   // create new ray
         }
         INDEX = Index(POS) ;
         if (INDEX>=0)  COUNT[id] += 1 ;         // ray re-enters
      }
   } // while INDEX>=0
   TPL[id] = tpl ;
}




__kernel void Update(
#if (WITH_HALF==1)
                     __global short4 *CLOUD,   //  0 [CELLS]: vx, vy, vz, sigma
#else
                     __global float4 *CLOUD,   //  0 [CELLS]: vx, vy, vz, sigma
#endif
                     GAUSTORE  float *GAU,     //  1 precalculated gaussian profiles [GNO,CHANNELS]
                     constant int2   *LIM,     //  2 limits of ~zero profile function [GNO]
                     const float      Aul,     //  3 Einstein A(upper->lower)
                     const float      A_b,     //  4 (g_u/g_l)*B(upper->lower)
                     const float      GN,      //  5 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                     const float      APL,     //  6 average path length [GL]
                     const float      BG,      //  7 background value (photons)
                     const float      DIRWEI,  //  8 weight factor (based on current direction)
                     const int        LEADING, //  9 leading edge
                     const float3     POS0,    // 10 initial position of id=0 ray
                     const float3     DIR,     // 11 ray direction
                     __global float  *NI,      // 12 [2*CELLS]:  NI[upper] + NB_NB
                     __global float  *RES,     // 13 [2*CELLS]:  SIJ, ESC
                     __global float  *NTRUES   // 14 [GLOBAL*MAXCHN]
#if (WITH_CRT>0)
                     ,
                     constant float *CRT_TAU,  //  dust optical depth / GL
                     constant float *CRT_EMI   //  dust emission photons/c/channel/H
#endif                     
#if (BRUTE_COOLING>0)
                     ,__global float *COOL     // 14,15 [CELLS] = cooling 
#endif
                    )  {
   float weight ;        
   float dx, doppler ;
   float tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;
   float sum_delta_true, all_escaped, nu ;
   int row, shift, INDEX, c1, c2 ;
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge. If ray exists through a side, a new one is created 
   // on the opposite side and the ray ends when the downstream edge is reached.
   //  ==> all work items have exactly the same load
   int  id = get_global_id(0) ;
   int lid = get_local_id(0) ;
   if (id>=NRAY) return ;
   // TODO: replace with actual number of rays along the two dimensions
   int nx = (NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ; // dimensions of the current ray grid
   GAUSTORE float *profile ;
#if (LOC_LOWMEM==1)
   __global float *NTRUE = &NTRUES[id*CHANNELS] ;
#else    // this is ok for CPU
   __local float  NTRUESSS[LOCAL*CHANNELS] ;
   __local float *NTRUE = &NTRUESSS[lid*CHANNELS] ;
#endif
   
#if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, pro, sij ;
#endif
   
   
   // Initial position of each ray shifted by two grid units
   //  == host has a loop over 4 offset positions
   float3 POS = POS0 ;
   float3 POS_LE ;
   switch (LEADING) {
    case 0:  POS.x =    PEPS ;  POS.y += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 1:  POS.x = NX-PEPS ;  POS.y += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 2:  POS.y =    PEPS ;  POS.x += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 3:  POS.y = NY-PEPS ;  POS.x += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 4:  POS.z =    PEPS ;  POS.x += 2.0f*(id/ny) ;   POS.y += 2.0f*(id%ny) ;   break ;
    case 5:  POS.z = NZ-PEPS ;  POS.x += 2.0f*(id/ny) ;   POS.y += 2.0f*(id%ny) ;   break ;
   }
#if (FIX_PATHS>0)
   POS_LE =  POS ;
#endif
   INDEX = Index(POS) ;
   for(int i=0; i<CHANNELS; i++) NTRUE[i] = BG * DIRWEI ;
   
   
#if (BRUTE_COOLING>0)
   float cool = BG*DIRWEI*CHANNELS ;
   if (INDEX>=0) {
      atomicAdd_g_f(&(COOL[INDEX]), -cool) ; // heating of the entered cell
   }
#endif
   
   
   
   while(INDEX>=0) {
      
      float dy, dz ;
      if (DIR.x<0.0f)   dx = -      fmod(POS.x,1.0f)  / DIR.x ;  else   dx =  (1.0f-fmod(POS.x,1.0f)) / DIR.x ;
      if (DIR.y<0.0f)   dy = -      fmod(POS.y,1.0f)  / DIR.y ;  else   dy =  (1.0f-fmod(POS.y,1.0f)) / DIR.y ;
      if (DIR.z<0.0f)   dz = -      fmod(POS.z,1.0f)  / DIR.z ;  else   dz =  (1.0f-fmod(POS.z,1.0f)) / DIR.z ;
      dx      =  min(dx, min(dy, dz)) + PEPS ;
      nu        =  NI[2*INDEX  ] ;
      nb_nb     =  NI[2*INDEX+1] ;
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
      weight    =  (dx/APL)*VOLUME ;  // VOLUME == 1.0/CELLS, fraction of cloud volume
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
#if (WITH_HALF==1)
      doppler  *=  0.002f ;  // half integer times 0.002f km/s
#endif                 
      shift     =  round(doppler/WIDTH) ;      
#if (WITH_HALF==1)
      //               sigma = 0.002f * w
      // lookup table: sigma = SIGMA0 * SIGMAX^row
      row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      // if (id==0) printf("  sigma %6.2f  --- row %d/%d\n", CLOUD[INDEX].w*0.002f, row, GNO) ;
#else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
#endif
      profile   =  &GAU[row*CHANNELS] ;
      // avoid profile function outside profile channels LIM.x, LIM.y
      c1        =  max(LIM[row].x+shift, max(0, shift)) ;
      c2        =  min(LIM[row].y+shift, min(CHANNELS-1, CHANNELS-1+shift)) ;
      
      tmp_tau   =   dx*nb_nb*GN ;
      if (fabs(tmp_tau)<1.0e-32f) tmp_tau = 1.0e-32f ;   // was e-32
      tmp_emit  =  weight * nu *(Aul/tmp_tau) ;  // GN include grid length [cm] --- DROPPED DIRWEI == BG ONLY
      
      sum_delta_true = 0.0f ;
      all_escaped    = 0.0f ;
#if (WITH_CRT>0)
      sij = 0.0f ;
      // Dust optical depth and emission
      //   here escape = line photon exiting the cell + line photons absorbed by dust
      Ctau      =  dx     * CRT_TAU[INDEX] ;
      Cemit     =  weight * CRT_EMI[INDEX] ;      
      for(int ii=c1; ii<=c2; ii++)  {
         pro    =  profile[ii-shift] ;
         Ltau   =  tmp_tau*pro ;
         Ttau   =  Ctau + Ltau ;
         // tt     =  (1.0f-exp(-Ttau)) / Ttau ;
         tt     =  (1.0f-exp(-Ttau))/Ttau ;
         // ttt    = (1.0f-tt)/Ttau
         ttt    =  (1.0f-tt)/Ttau ;
         // Line emission leaving the cell   --- GL in profile
         Lleave =  weight*nu*Aul*pro * tt ;
         // Dust emission leaving the cell 
         Dleave =  Cemit *                     tt ;
         // SIJ updates, first incoming photons then absorbed dust emission
         sij   +=  A_b * pro*GN*dx * NTRUE[ii]*tt ;
         // sij         += A_b * profile[ii]*GN*Cemit*dx*(1.0f-tt)/Ttau ; // GN includes GL!
         sij   +=  A_b * pro*GN*dx * Cemit*ttt ;    // GN includes GL!
         // "Escaping" line photons = absorbed by dust or leave the cell
         all_escaped +=  Lleave  +  weight*nu*Aul*pro * Ctau * ttt ;
         // Total change of photons in the package
         NTRUE[ii]    =  NTRUE[ii]*exp(-Ttau) + Dleave + Lleave ;
      }  // loop over channels
      // Update SIJ and ESC (note - there may be two updates, one incoming, one outgoing ray)
      RES[2*INDEX]   += sij ;            //  2020-06-02  divided by VOLUME only in solver 
      // Emission ~ path length dx but also weighted according to direction, works because <WEI>==1.0
      RES[2*INDEX+1] += all_escaped ;    // divided by VOLUME only oin Solve() !!!
#else
      for(int ii=c1; ii<=c2; ii++)  {
         weight           =  tmp_tau*profile[ii-shift] ;
         factor           =  (fabs(weight)>0.01f) ? (1.0f-exp(-weight)) : (weight*(1.0f-weight*(0.5f-0.166666667f*weight))) ;
         escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed         =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped     +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      // Update SIJ and ESC
      
      if ((A_b*(sum_delta_true/nb_nb))>0.1f) {
         printf("SIJ UPDATE %6d  A_b %10.3e, sum_delta_true %10.3e  nb_nb %.3e = %.3e\n",
                INDEX, A_b, sum_delta_true, nb_nb, A_b*(sum_delta_true/nb_nb)) ;
      }
# if 1
      RES[2*INDEX  ]  +=  A_b * (sum_delta_true/nb_nb) ; // 2020-06-02, divide by VOLUME only in solver
      RES[2*INDEX+1]  +=  all_escaped ;
# else
      AADD((__global float*)(RES+2*INDEX),   A_b * (sum_delta_true/nb_nb)) ; // parentheses!
      AADD((__global float*)(RES+2*INDEX+1), all_escaped) ;      
# endif
#endif
      POS      += dx*DIR ;  // end of the step

      
#if (FIX_PATHS>0)
      // try to precise the position
      float s ;
      if (LEADING<2) {
         if ((int)(POS.x)%10==0) {
            s      =  (LEADING==0) ?  (POS.x/DIR.x) : ((POS.x-NX)/DIR.x) ;
            POS.y  =  POS_LE.y + s*DIR.y ;
            POS.z  =  POS_LE.z + s*DIR.z ;
            if (POS.y<0.0f) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;
            if (POS.z<0.0f) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
         }
      } else {
         if (LEADING<4) {
            s      =  (LEADING==2) ?  (POS.y/DIR.y) : ((POS.y-NY)/DIR.y) ;
            POS.x  =  POS_LE.x + s*DIR.x ;
            POS.z  =  POS_LE.z + s*DIR.z ;
            if (POS.x<0.0f) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
            if (POS.z<0.0f) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
         } else {
            s      =  (LEADING==4) ?  (POS.z/DIR.z) : ((POS.z-NY)/DIR.z) ;
            POS.x  =  POS_LE.x + s*DIR.x ;
            POS.y  =  POS_LE.y + s*DIR.y ;
            if (POS.x<0.0f) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
            if (POS.y<0.0f) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;            
         }
      }
#endif
      
      
      
#if (BRUTE_COOLING>0)
      // total number of photons in the package as it exits the cell
      float cool = 0.0f ;
      for(int ii=0; ii<CHANNELS; ii++) cool += NTRUE[ii] ;
      atomicAdd_g_f(&(COOL[INDEX]), cool) ; // cooling of cell INDEX
#endif
      
      INDEX      = Index(POS) ;
      
#if (BRUTE_COOLING>0)
      if (INDEX>=0) {
         atomicAdd_g_f(&(COOL[INDEX]), -cool) ; // heating of the next cell
      }
#endif
      
      
      if (INDEX<0) {  // exits the cloud... but on which side?
         if (POS.x>=NX  ) {   if (LEADING!=0)   POS.x =    PEPS ;    }
         if (POS.x<=0.0f) {   if (LEADING!=1)   POS.x = NX-PEPS ;    } 
         if (POS.y>=NY  ) {   if (LEADING!=2)   POS.y =    PEPS ;    }
         if (POS.y<=0.0f) {   if (LEADING!=3)   POS.y = NY-PEPS ;    } 
         if (POS.z>=NZ  ) {   if (LEADING!=4)   POS.z =    PEPS ;    }
         if (POS.z<=0.0f) {   if (LEADING!=5)   POS.z = NZ-PEPS ;    } 
         INDEX = Index(POS) ;
         if (INDEX>=0) {   // new ray started on the opposite side (same work item)
            for(int ii=0; ii<CHANNELS; ii++) NTRUE[ii] = BG * DIRWEI ;            
#if (BRUTE_COOLING>0)
            float cool = BG*DIRWEI*CHANNELS ;
            atomicAdd_g_f(&(COOL[INDEX]), -cool) ; // heating of the entered cell
#endif            
         }
      } // if INDEX<0
      
   } // while INDEX>=0
   
}












__kernel void Spectra(
#if (WITH_HALF==1)
                      __global short4  *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
#else
                      __global float4  *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
#endif
                      GAUSTORE   float *GAU,          //  1 precalculated gaussian profiles
                      constant int2    *LIM,          //  2 limits of ~zero profile function
                      const float       GN,           //  3 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                      const float2      D,            //  4 ray direction == theta, phi
                      __global float   *NI,           //  5 [2*CELLS]:  NI[upper] + NB_NB
                      const float       DE,           //  6 grid units, offset in RA direction
                      const int         NRA,          //  7 number of DEC points = work items
                      const float       STEP,         //  8 step between spectra (grid units)
                      const float       BG,           //  9 background intensity
                      const float       emis0,        // 10 h/(4pi)*freq*Aul*int2temp
                      __global float   *NTRUE_ARRAY,  // 11 NRA*CHANNELS
                      __global float   *SUM_TAU_ARRAY // 12 NRA*CHANNELS
#if (WITH_CRT>0)
                      ,
                      constant float *CRT_TAU,        //  dust optical depth / GL
                      constant float *CRT_EMI         //  dust emission photons/c/channel/H
#endif
                      ,
                      const float2 CENTRE             //  map centre in units of the pixel size
                     )
{
   // each work item calculates one spectrum for 
   //   ra  =  RA (grid units, from cloud centre)
   //   de  =  DE(id)
   int id = get_global_id(0) ;
   if (id>=NRA) return ; // no more rays
   __global float *NTRUE   = &(NTRUE_ARRAY[  id*CHANNELS]) ;
   __global float *SUM_TAU = &(SUM_TAU_ARRAY[id*CHANNELS]) ;   
   int i ;
   float RA ;  // grid units, offset of current ray
   // DE  =   (id-0.5f*(NDEC-1.0f))*STEP ;
   // RA  =   -(id-0.5f*(NRA-1.0f))*STEP ;   // spectra from left to right = towards -RA
   // RA  =   -id*STEP ;   // spectra from left to right = towards -RA
   RA     =   id ;       // spectra from left to right = towards -RA

   // calculate the initial position of the ray
   float3 POS, DIR ;
   float dx, dy, dz ;
   POS.x   =  0.500001f*NX ;  POS.y = 0.500001f*NY ;  POS.z = 0.500001f*NZ ;
#if 1
   DIR.x   =   sin(D.x)*cos(D.y) ;
   DIR.y   =   sin(D.x)*sin(D.y) ;
   DIR.z   =   cos(D.x)            ;
   float3 RV, DV ;
   // Definition:  DE follows +Z, RA is now right
   if (DIR.z>0.9999f) {
      RV.x= 0.0001f ;  RV.y=+0.9999f ; RV.z=0.0001f ;    // RA = Y
      DV.x=-0.9999f ;  DV.y= 0.0001f ; DV.z=0.0001f ;    // DE = -X
   } else {
      if (DIR.z<-0.9999f) {                              // view from -Z =>  (Y,X)
         RV.x= 0.0001f ;  RV.y=+0.9999f ; RV.z=0.0001f ;
         DV.x=+0.9999f ;  DV.y= 0.0001f ; DV.z=0.0001f ; 
      } else {
         // RA orthogonal to DIR and to +Z,   DIR=(1,0,0) => RV=(0,+1,0)
         //                                   DIR=(0,1,0) => RV=(-1,0,0)
         RV.x = -DIR.y ;   RV.y = +DIR.x ;  RV.z = ZERO ;  RV = normalize(RV) ;
         // DV  =   RV x DIR
         DV.x = -RV.y*DIR.z+RV.z*DIR.y ;
         DV.y = -RV.z*DIR.x+RV.x*DIR.z ;
         DV.z = -RV.x*DIR.y+RV.y*DIR.x ;
      }
   }
   // printf("OBS = %.4f %.4f %.4f    RA = %.4f %.4f %.4f    DE = %.4f %.4f %.4f \n", DIR.x, DIR.y, DIR.z,  RV.x, RV.y, RV.z,  DV.x, DV.y, DV.z) ;
   // Offsets in RA and DE directions
   POS.x  +=  (RA-CENTRE.x)*STEP*RV.x + (DE-CENTRE.y)*STEP*DV.x ;
   POS.y  +=  (RA-CENTRE.x)*STEP*RV.y + (DE-CENTRE.y)*STEP*DV.y ;
   POS.z  +=  (RA-CENTRE.x)*STEP*RV.z + (DE-CENTRE.y)*STEP*DV.z ;
   // Change DIR to direction away from the observer
   DIR *= -1.0f ;
#else
   // RA offset
   POS.x  +=  +RA * sin(D.y)  ;          // RA in grid units
   POS.y  +=  -RA * cos(D.y)  ;
   POS.z  +=   0.0f ;
   // DE offset
   POS.x  +=  -DE* cos(D.x) * cos(D.y) ; // DE in grid units
   POS.y  +=  -DE* cos(D.x) * sin(D.y) ;
   POS.z  +=   DE* sin(D.x) ;   
   // direction AWAY from the observer
   DIR.x   = -sin(D.x)*cos(D.y) ;
   DIR.y   = -sin(D.x)*sin(D.y) ;
   DIR.z   = -cos(D.x);
#endif
   if (fabs(DIR.x)<1.0e-10f) DIR.x = 1.0e-10f ;
   if (fabs(DIR.y)<1.0e-10f) DIR.y = 1.0e-10f ;
   if (fabs(DIR.z)<1.0e-10f) DIR.z = 1.0e-10f ;
   
   // printf("ORI   POS %8.4f %8.4f %8.4f  DIR %8.4f %8.4f %8.4f\n", POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z) ;
   
   // go to front surface
   POS -=  1000.0f*DIR ;   // make sure it will be a step forward
   if (DIR.x>0.0f)  dx = (0.0f-POS.x)/DIR.x ;
   else             dx = (NX-POS.x)/DIR.x ;
   if (DIR.y>0.0f)  dy = (0.0f-POS.y)/DIR.y ;
   else             dy = (NY-POS.y)/DIR.y ;
   if (DIR.z>0.0f)  dz = (0.0f-POS.z)/DIR.z ;
   else             dz = (NZ-POS.z)/DIR.z ;
   dx          =  max(dx, max(dy, dz)) + 1.0e-4f ;  // max because we are outside
   POS        +=  dx*DIR ;
   int INDEX   =  Index(POS) ;
   
   for(int i=0; i<CHANNELS; i++) {
      NTRUE[i]   = 0.0f ;
      SUM_TAU[i] = 0.0f ;
   }
   
   float tau, dtau, emissivity, doppler, nu ;
   int row, shift, c1, c2  ;
   GAUSTORE float* profile ;
#if (WITH_CRT>0)
   float Ctau, Cemit, pro, distance=0.0f ;
#endif
   
   
   // printf("emis0 %12.4e, nu %12.4e, nbnb %12.4e, GL %12.4e\n", emis0, NI[INDEX].x, NI[INDEX].y, GL) ;
   
   while (INDEX>=0) {
      if (DIR.x<0.0f)   dx = -      fmod(POS.x,1.0f)  / DIR.x - EPS/DIR.x;
      else              dx =  (1.0f-fmod(POS.x,1.0f)) / DIR.x + EPS/DIR.x;
      if (DIR.y<0.0f)   dy = -      fmod(POS.y,1.0f)  / DIR.y - EPS/DIR.y;
      else              dy =  (1.0f-fmod(POS.y,1.0f)) / DIR.y + EPS/DIR.y;
      if (DIR.z<0.0f)   dz = -      fmod(POS.z,1.0f)  / DIR.z - EPS/DIR.z;
      else              dz =  (1.0f-fmod(POS.z,1.0f)) / DIR.z + EPS/DIR.z;
      dx         =  min(dx, min(dy, dz)) + EPS ;  // actual step
      nu         =  NI[2*INDEX] ;     
      
#if 0
      tau        =  dx * NI[2*INDEX+1] * GN * GL ; // need to separate GN and GL ?
      tau        =  clamp(tau, -0.05f, 1.0e5f) ;    // NO MASERS !!
#else
      if (fabs(NI[2*INDEX+1])<1.0e-24f) {
         tau     =  dx * 1.0e-24f * GN * GL ;
      } else {
         tau     =  dx * NI[2*INDEX+1] * GN * GL ;         
         // Comment this out to make results consistent with Cppsimu!!
         // tau = clamp(tau, -0.05f, 1.0e10f) ;  // CLIP MASERS ???
         // >-0.5f clear effect in test case [10,2000]K, [10,1e5]cm-3
         //        ... although only in extreme parameter combinations
         // >-1.0f already ~identical to Cppsimu
         // The limit will depend on cell size and density... is >-2.0f a safe option??
         tau = clamp(tau, -2.0f, 1.0e10f) ;
         // if (fabs(tau)<1.0e-5) tau=1.0e-5 ;
      }
#endif 
      doppler    =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
#if (WITH_HALF==1)
      doppler   *=  0.002f ;
      if (fabs(doppler)>(0.5f*CHANNELS*WIDTH)) {
         POS  += dx*DIR ;
         INDEX = Index(POS) ;
         continue ;
      }
      row        =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
#else
      row        =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
#endif
      profile    =  &GAU[row*CHANNELS] ;
      shift      =  round(doppler/WIDTH) ;
      // printf("SHIFT %2d\n", shift) ;
      c1         =  LIM[row].x+shift ;
      c2         =  LIM[row].y+shift ;
      c1         =  max(c1,   max(shift,      0               )) ;
      c2         =  min(c2,   min(CHANNELS-1, CHANNELS-1+shift)) ;
      // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      emissivity =  emis0 * nu * dx * GL ;
      
#if (WITH_CRT>0)
      Ctau       =  CRT_TAU[INDEX] * dx      ;
      Cemit      =  CRT_EMI[INDEX] * dx * GL ;
      for(i=0; i<CHANNELS; i++) {
         pro         =  profile[clamp(i-shift, 0, CHANNELS-1)] ;
         dtau        =  tau*pro + Ctau ;
         NTRUE[i]   += (emissivity*pro*GN + Cemit) * exp(-SUM_TAU[i]) * ((1.0f-exp(-dtau))/dtau) ;
         SUM_TAU[i] +=  dtau  ;
      }     
#else
      for(i=c1; i<=c2; i++) {         
         dtau = tau*profile[i-shift] ;
# if 0
         if (fabs(dtau)>1.0e-5f)  {            
            NTRUE[i] +=  emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i]) * (1.0f-exp(-dtau)) / dtau ;
         } else {
            NTRUE[i] +=  emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i]) ;
         }
# else
         NTRUE[i] += emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i])
           * (   (fabs(dtau)>1.0e-5f)   ?   ((1.0f-exp(-dtau))/dtau)   :   1.0f ) ;
# endif
         SUM_TAU[i] += dtau  ;
      }
#endif
      
      POS  += dx*DIR ;
      INDEX = Index(POS) ;
   } // while INDEX
   
   for (i=0; i<CHANNELS; i++) NTRUE[i] -=  BG*(1.0f-exp(-SUM_TAU[i])) ;
   
}




// ==============================================================================
   





#if (WITH_HFS>0)

__kernel void UpdateHF(
                       __global float4 *CLOUD,   //  0 [CELLS]: vx, vy, vz, sigma
                       constant float  *GAU,     //  1 precalculated gaussian profiles [GNO*CHANNELS]
                       constant int2   *LIM,     //  2 limits of ~zero profile function [GNO]
                       const float      Aul,     //  3 Einstein A(upper->lower)
                       const float      A_b,     //  4 (g_u/g_l)*B(upper->lower)
                       const float      GN,      //  5 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                       const float      APL,     //  6 average path length [GL]
                       const float      BG,      //  7 background value (photons)
                       const float      DIRWEI,  //  8 weight factor (based on current direction)
                       const int        LEADING, //  9 leading edge
                       const float3     POS0,    // 10 initial position of id=0 ray
                       const float3     DIR,     // 11 ray direction
                       __global float  *NI,      // 12 [2*CELLS]:  NI[upper] + NB_NB
                       __global float  *RES,     // 13 [2*CELLS]:  SIJ, ESC
                       const int        NCHN,    // 14 number of channels (>= CHANNELS)
                       const int        NCOMP,   // 15 number of HF components
                       __global float2 *HF,      // 16 [MAXCMP].x = offset, [MAXCMP].y = weight
                       __global float  *NTRUES   // 17 [GLOBAL*MAXCHN]
                      )  {
   // this one used for HF in LTE -- for LOC_HF.cpp
   float weight, dx ; // doppler -- reusing dx
   float tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;   
   float sum_delta_true, all_escaped ; // nu -- using directly NI[].x
   int row, shift, INDEX, c1, c2 ;
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge 
   // if ray exists through a side, a new one is created on the opposite side
   // Ray ends when the downstream edge is reached
   int  id = get_global_id(0) ;
   int lid = get_local_id(0) ;   
   if (id>=NRAY) return ;
   int ny=(NY+1)/2, nz=(NZ+1)/2 ; // dimensions of the current ray grid
   
   constant float *pro ;             // pointer to GAU vector
   
# if (LOC_LOWMEM==1)
   __global float *NTRUE   = &NTRUES[id*MAXCHN] ;
# else   // this leads to "insufficient resources" in case of GPU
   __local float  NTRUESSS[LOCAL*MAXCHN] ;
   __local float *NTRUE = &NTRUESSS[lid*MAXCHN] ;
# endif
   
   // It is ok up to ~150 channels to have these on GPU local memory
   __local float  profile_array[LOCAL*MAXCHN] ;  // MAXCHN = max of NCHN
   __local float *profile = &profile_array[lid*MAXCHN] ;   
   
   // Initial position of each ray shifted by two grid units
   //  == host has a loop over 4 offset positions
   float3 POS = POS0 ;
   float3 POS_LE ;
   switch (LEADING) {
    case 0:  POS.x = EPS ;     POS.y += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 1:  POS.x = NX-EPS ;  POS.y += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 2:  POS.y = EPS ;     POS.x += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 3:  POS.y = NY-EPS ;  POS.x += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 4:  POS.z = EPS ;     POS.x += 2.0f*(id/ny) ;   POS.y += 2.0f*(id%ny) ;   break ;
    case 5:  POS.z = NZ-EPS ;  POS.x += 2.0f*(id/ny) ;   POS.y += 2.0f*(id%ny) ;   break ;
   }
# if (FIX_PATHS>0)
   POS_LE = POS ;
# endif
   INDEX = Index(POS) ;
   for(int i=0; i<NCHN; i++) NTRUE[i] = BG * DIRWEI ;
   
   while(INDEX>=0) {
      
# if 0
      int dy, dz ;
      if (DIR.x<0.0f)   dx = -      fmod(POS.x,1.0f)  / DIR.x - EPS/DIR.x;
      else              dx =  (1.0f-fmod(POS.x,1.0f)) / DIR.x + EPS/DIR.x;
      if (DIR.y<0.0f)   dy = -      fmod(POS.y,1.0f)  / DIR.y - EPS/DIR.y;
      else              dy =  (1.0f-fmod(POS.y,1.0f)) / DIR.y + EPS/DIR.y;
      if (DIR.z<0.0f)   dz = -      fmod(POS.z,1.0f)  / DIR.z - EPS/DIR.z;
      else              dz =  (1.0f-fmod(POS.z,1.0f)) / DIR.z + EPS/DIR.z;
      dx      =  min(dx, min(dy, dz)) ;  // actual step
# else
      dx =     ((DIR.x<0.0f) ? (-fmod(POS.x,1.0f)/DIR.x-EPS/DIR.x) : ((1.0f-fmod(POS.x,1.0f))/DIR.x+EPS/DIR.x)) ;
      dx=min(dx,(DIR.y<0.0f) ? (-fmod(POS.y,1.0f)/DIR.y-EPS/DIR.y) : ((1.0f-fmod(POS.y,1.0f))/DIR.y+EPS/DIR.y)) ;
      dx=min(dx,(DIR.z<0.0f) ? (-fmod(POS.z,1.0f)/DIR.z-EPS/DIR.z) : ((1.0f-fmod(POS.z,1.0f))/DIR.z+EPS/DIR.z)) ;      
# endif      
      // nu        =  NI[INDEX].x ;
      nb_nb     =  NI[2*INDEX+1] ;
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
      weight    =  (dx/APL)*VOLUME ;  // VOLUME == 1.0/CELLS, fraction of cloud volume
      POS       += dx*DIR ;
      
      
# if (FIX_PATHS>99)
      // try to precise the position
      if (LEADING<2) {
         if ((int)(POS.x)%10==0) {
            s      =  (LEADING==0) ?  (POS.x/DIR.x) : ((POS.x-NX)/DIR.x) ;
            POS.y  =  POS_LE.y + s*DIR.y ;
            POS.z  =  POS_LE.z + s*DIR.z ;
            if (POS.y<0.0f) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;
            if (POS.z<0.0f) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
         }
      } else {
         if (LEADING<4) {
            s      =  (LEADING==2) ?  (POS.y/DIR.y) : ((POS.y-NY)/DIR.y) ;
            POS.x  =  POS_LE.x + s*DIR.x ;
            POS.z  =  POS_LE.z + s*DIR.z ;
            if (POS.x<0.0f) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
            if (POS.z<0.0f) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
         } else {
            s      =  (LEADING==4) ?  (POS.z/DIR.z) : ((POS.z-NY)/DIR.z) ;
            POS.x  =  POS_LE.x + s*DIR.x ;
            POS.y  =  POS_LE.y + s*DIR.y ;
            if (POS.x<0.0f) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
            if (POS.y<0.0f) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;            
         }
      }
# endif
      
      
      
      
# if 0 // no difference between these two! = small tau tmp_emit not important
      tmp_tau   =  dx*nb_nb*GN  ;
      tmp_emit  =  DIRWEI * weight * NI[2*INDEX] * Aul / tmp_tau ;  // GN include grid length [cm]
      if (fabs(tmp_tau)<1.0e-10f) {
         tmp_tau  = 1.0e-10f ;
         tmp_emit = 0.0f ;
      }
# else
      if (fabs(nb_nb)<1.0e-37f) nb_nb=1.0e-37f ;  // was e-32
      tmp_tau   =  dx*nb_nb*GN ;
      tmp_emit  =  DIRWEI * weight * NI[2*INDEX] * Aul / tmp_tau ;  // GN include grid length [cm]
# endif
      
      // reusing dx for doppler !
      dx  =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# if (WITH_HALF==1)
      dx *=  0.002f ;  // half integer times 0.002f km/s
# endif
      
      
      
# if (WITH_HALF==1)
      //               sigma = 0.002f * w
      // lookup table: sigma = SIGMA0 * SIGMAX^row
      row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      // if (id==0) printf("  sigma %6.2f  --- row %d/%d\n", CLOUD[INDEX].w*0.002f, row, GNO) ;
# else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
# endif
      
      // Calculate profile as weighted sum of precalculated Gaussian profiles
      for(int i=0; i<NCHN; i++) profile[i] = 0.0f ;
      pro       =  &GAU[row*CHANNELS] ;
      for(int icomp=0; icomp<NCOMP; icomp++) {
         shift  =  round(dx/WIDTH) + HF[icomp].x  ;  // shift in channels dx == doppler !
         // skip profile function outside channels [LIM.x, LIM.y]
         // LIM.x in pro[] is mapped to channel c1 in profile[NCHN]
         c1 = 0.5f*(NCHN-1.0f) + shift - 0.5f*(CHANNELS-1.0f) + LIM[row].x ; // could be negative
         c2 = c1+LIM[row].y-LIM[row].x ;  // update [c1,c2[ in profile
         //   c1 could be negative => require c1+ii >= 0
         //   c1+LIM.y-LIM.x could be >= NCHN ==> require   c1+ii < NCHN ==>  ii< NCHN-c1
         for(int ii=max(0,-c1); ii<min(LIM[row].y-LIM[row].x, NCHN-c1); ii++) {
            profile[c1+ii] +=  HF[icomp].y * pro[LIM[row].x+ii] ;
         }  // this assumes that sum(HFI[].y) == 1.0 !!
      }
      
      sum_delta_true = all_escaped = 0.0f ;
      
# if (UNROLL==0)  // no manual unroll
      for(int ii=0; ii<NCHN; ii++)  {
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;
         escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed         =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped     +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
# else
#  define UR 8
      int no = (NCHN-1)/UR, ii ;
      for(int i=0; i<no; i++) {
         // 0
         ii               =  UR*i+0 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 1
         ii               =  UR*i+1 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 2
         ii               =  UR*i+2 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 3
         ii               =  UR*i+3 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 4
         ii               =  UR*i+4 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 5
         ii               =  UR*i+5 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 6
         ii               =  UR*i+6 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 7
         ii               =  UR*i+7 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
      }
      for(ii++; ii<NCHN; ii++)  {
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;
         escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed         =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped     +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
# endif
      
      
      
      
      // if nb_nb was already checked
      RES[2*INDEX  ]  +=   A_b * sum_delta_true / nb_nb ;  // 2020-06-02 divide by VOLUME in solver
      // Emission ~ path length dx but also weighted according to direction, 
      //            works because <WEI>==1.0
      RES[2*INDEX+1]  +=  all_escaped ;
      
      // POS       += dx*DIR ;
      INDEX      = Index(POS) ;
      
      if (INDEX<0) {  // exits the cloud... but on which side?
         if (POS.x<=0.0f) {
            if (LEADING!=1)  POS.x = NX-EPS ;
         } 
         if (POS.x>=NX) {
            if (LEADING!=0)  POS.x = EPS ;  
         }
         if (POS.y<=0.0f) {
            if (LEADING!=3)  POS.y = NY-EPS ;
         } 
         if (POS.y>=NY) {
            if (LEADING!=2)  POS.y = EPS ;  
         }
         if (POS.z<=0.0f) {
            if (LEADING!=5)  POS.z = NZ-EPS ;
         } 
         if (POS.z>=NZ) {
            if (LEADING!=4)  POS.z = EPS ;  
         }
         INDEX = Index(POS) ;
         if (INDEX>=0) {   // new ray started on the opposite side (same work item)
            for(int ii=0; ii<NCHN; ii++) NTRUE[ii] = BG * DIRWEI ;
         }
      } // if INDEX<0
      
   } // while INDEX>=0
   
}







__kernel void SpectraHF(
# if (WITH_HALF==1)
                        __global short4 *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
# else
                        __global float4 *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
# endif
                        constant float  *GAU,          //  1 precalculated gaussian profiles
                        constant int2   *LIM,          //  2 limits of ~zero profile function
                        const float      GN,           //  3 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                        const float2     D,            //  4 ray direction == theta, phi
                        __global float  *NI,           //  5 [2*CELLS]:  NI[upper] + NB_NB
                        const float      DE,           //  6 grid units, offset in DE direction
                        const int        NRA,          //  7 number of RA points = work items
                        const float      STEP,         //  8 step between spectra (grid units)
                        const float      BG,           //  9 background intensity
                        const float      emis0,        // 10 h/(4pi)*freq*Aul*int2temp
                        __global float  *NTRUE_ARRAY,  // 11 NRA*MAXNCHN
                        __global float  *SUM_TAU_ARRAY,// 12 NRA*MAXNCHN
                        const int NCHN,                // 13 channels (in case of HF spectrum)
                        const int NCOMP,               // 14 number of components
                        __global float2 *HF,           // 15 channel offsets, weights
                        const float2 CENTRE            // 16 map centre in pixel units (offset)
                       )
{
   // printf("SpectraHF\n") ;
   // each work item calculates one spectrum for 
   //   ra  =  RA (grid units, from cloud centre)
   //   de  =  DE(id)
   int  id = get_global_id(0) ;
   if (id>=NRA) return ; // no more rays
   int lid = get_local_id(0) ;
   
   __global float *NTRUE   = &(NTRUE_ARRAY[id*MAXCHN]) ;
   __global float *SUM_TAU = &(SUM_TAU_ARRAY[id*MAXCHN]) ;
   int i ;
   float RA ; // grid units, offset of current ray
   // RA  =   -(id-0.5f*(NRA-1.0f))*STEP ;
   // RA  =   -id*STEP ;
   RA  =   id ; 
   
   
   // calculate the initial position of the ray
   float3 POS, DIR ;
   float dx, dy, dz ;
   POS.x   =  0.500001f*NX ;  POS.y = 0.500001f*NY ;  POS.z = 0.500001f*NZ ;
# if 1
   DIR.x   =   sin(D.x)*cos(D.y) ;
   DIR.y   =   sin(D.x)*sin(D.y) ;
   DIR.z   =   cos(D.x)            ;
   float3 RV, DV ;
   // Definition:  DE follows +Z, RA is now right
   if (DIR.z>0.9999f) {
      RV.x= 0.0001f ;  RV.y=+0.9999f ; RV.z=0.0001f ;    // RA = Y
      DV.x=-0.9999f ;  DV.y= 0.0001f ; DV.z=0.0001f ;    // DE = -X
   } else {
      if (DIR.z<-0.9999f) {                              // view from -Z =>  (Y,X)
         RV.x= 0.0001f ;  RV.y=+0.9999f ; RV.z=0.0001f ;
         DV.x=+0.9999f ;  DV.y= 0.0001f ; DV.z=0.0001f ; 
      } else {
         // RA orthogonal to DIR and to +Z,   DIR=(1,0,0) => RV=(0,+1,0)
         //                                   DIR=(0,1,0) => RV=(-1,0,0)
         RV.x = -DIR.y ;   RV.y = +DIR.x ;  RV.z = ZERO ;  RV = normalize(RV) ;
         // DV  =   RV x DIR
         DV.x = -RV.y*DIR.z+RV.z*DIR.y ;
         DV.y = -RV.z*DIR.x+RV.x*DIR.z ;
         DV.z = -RV.x*DIR.y+RV.y*DIR.x ;
      }
   }
   // printf("OBS = %.4f %.4f %.4f    RA = %.4f %.4f %.4f    DE = %.4f %.4f %.4f \n",          DIR.x, DIR.y, DIR.z,  RV.x, RV.y, RV.z,  DV.x, DV.y, DV.z) ;
   // Offsets in RA and DE directions
   POS.x  +=  (RA-CENTRE.x)*STEP*RV.x + (DE-CENTRE.y)*STEP*DV.x ;
   POS.y  +=  (RA-CENTRE.x)*STEP*RV.y + (DE-CENTRE.y)*STEP*DV.y ;
   POS.z  +=  (RA-CENTRE.x)*STEP*RV.z + (DE-CENTRE.y)*STEP*DV.z ;
   // Change DIR to direction away from the observer
   DIR *= -1.0f ;
# else
   // RA offset
   POS.x  +=  +RA * sin(D.y)  ;  // RA in grid units
   POS.y  +=  -RA * cos(D.y)  ;
   POS.z  +=   0.0f ;
   // DE offset
   POS.x  +=  -DE* cos(D.x) * cos(D.y) ; // DE in grid units
   POS.y  +=  -DE* cos(D.x) * sin(D.y) ;
   POS.z  +=   DE* sin(D.x) ;   
   // direction AWAY from the observer
   DIR.x   = -sin(D.x)*cos(D.y) ;
   DIR.y   = -sin(D.x)*sin(D.y) ;
   DIR.z   = -cos(D.x);
# endif
   if (fabs(DIR.x)<1.0e-10f) DIR.x = 1.0e-10f ;
   if (fabs(DIR.y)<1.0e-10f) DIR.y = 1.0e-10f ;
   if (fabs(DIR.z)<1.0e-10f) DIR.z = 1.0e-10f ;
   
   // printf("ORI   POS %8.4f %8.4f %8.4f  DIR %8.4f %8.4f %8.4f\n", POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z) ;
   
   // go to front surface
   POS -=  1000.0f*DIR ;   // make sure it will be a step forward
   if (DIR.x>0.0f)  dx = (0.0f-POS.x)/DIR.x ;
   else             dx = (NX-POS.x)/DIR.x ;
   if (DIR.y>0.0f)  dy = (0.0f-POS.y)/DIR.y ;
   else             dy = (NY-POS.y)/DIR.y ;
   if (DIR.z>0.0f)  dz = (0.0f-POS.z)/DIR.z ;
   else             dz = (NZ-POS.z)/DIR.z ;
   dx          =  max(dx, max(dy, dz)) + 1.0e-4f ;  // max because we are outside
   POS        +=  dx*DIR ;
   int INDEX   =  Index(POS) ;
   
   for(int i=0; i<NCHN; i++) {
      NTRUE[i]   = 0.0f ;    SUM_TAU[i] = 0.0f ;
   }
   
   float tau, dtau, emissivity, doppler, nu ;
   int row, shift, c1, c2  ;
   
   constant float* pro ;
   __local float  profile_array[LOCAL*MAXCHN] ;
   __local float *profile = &profile_array[lid*MAXCHN] ;
   
   // printf("emis0 %12.4e, nu %12.4e, nbnb %12.4e, GL %12.4e\n", emis0, NI[INDEX].x, NI[INDEX].y, GL) ;
   
   while (INDEX>=0) {
      if (DIR.x<0.0f)   dx = -      fmod(POS.x,1.0f)  / DIR.x - EPS/DIR.x;
      else              dx =  (1.0f-fmod(POS.x,1.0f)) / DIR.x + EPS/DIR.x;
      if (DIR.y<0.0f)   dy = -      fmod(POS.y,1.0f)  / DIR.y - EPS/DIR.y;
      else              dy =  (1.0f-fmod(POS.y,1.0f)) / DIR.y + EPS/DIR.y;
      if (DIR.z<0.0f)   dz = -      fmod(POS.z,1.0f)  / DIR.z - EPS/DIR.z;
      else              dz =  (1.0f-fmod(POS.z,1.0f)) / DIR.z + EPS/DIR.z;
      dx         =  min(dx, min(dy, dz)) + EPS ;  // actual step
      nu         =  NI[2*INDEX] ;     
      
      if (fabs(NI[2*INDEX+1])<1.0e-24f) {
         tau     =  dx * 1.0e-24f * GN * GL ;
      } else {
         tau     =  dx * NI[2*INDEX+1] * GN * GL ;
         
         // Comment this out to make results consistent with Cppsimu!!
         // tau = clamp(tau, -0.05f, 1.0e10f) ;  // CLIP MASERS ???
         // >-0.5f clear effect in test case [10,2000]K, [10,1e5]cm-3
         //        ... although only in extreme parameter combinations
         // >-1.0f already ~identical to Cppsimu
         // The limit will depend on cell size and density... is >-2.0f a safe option??
         tau = clamp(tau, -2.0f, 1.0e10f) ;
         // if (fabs(tau)<1.0e-5) tau=1.0e-5 ;
         
      }
      
      doppler    =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# if (WITH_HALF==1)
      doppler   *=  0.002f ;
      if (fabs(doppler)>(0.5f*NCHN*WIDTH)) {
         POS  += dx*DIR ;
         INDEX = Index(POS) ;
         continue ;
      }
      row        =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
# else
      row        =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
# endif
      
      
      // Calculate profile as weighted sum of precalculated Gaussian profiles
      for(int i=0; i<NCHN; i++) profile[i] = 0.0f ;      
      pro       =  &GAU[row*CHANNELS] ;   // GAU has only CHANNELS channels
      for(int icomp=0; icomp<NCOMP; icomp++) {
         shift  =  round(doppler/WIDTH) + HF[icomp].x  ;  // shift in channels
         // skip profile function outside channels [LIM.x, LIM.y]
         // LIM.x in pro[] is mapped to channel c1 in profile[NCHN]
         
         c1 = 0.5f*(NCHN-1.0f) + shift - 0.5f*(CHANNELS-1.0f) + LIM[row].x ; // could be negative
         c2 = c1+LIM[row].y-LIM[row].x ;  // update [c1,c2[ in profile
         
         //   c1 could be negative => require c1+ii >= 0
         //   c1+LIM.y-LIM.x could be >= NCHN
         //     ==> require   c1+ii < NCHN ==>  ii< NCHN-c1
# if 0
         if (id==0) printf("icomp %d   %3d %3d  inside %3d %3d  weight %5.3f\n",
                           icomp, max(0,-c1), min(LIM[row].y-LIM[row].x, NCHN-c1), 0, NCHN,
                           HF[icomp].y) ;
# endif
         for(int ii=max(0,-c1); ii<min(LIM[row].y-LIM[row].x, NCHN-c1); ii++) {
            profile[c1+ii] +=  HF[icomp].y * pro[LIM[row].x+ii] ;
         }
         // this assumes that sum(HFI[].y) == 1.0
      }
      
      // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      emissivity =  emis0 * nu * dx * GL ;
      
      for(i=0; i<NCHN; i++) {         
         dtau = tau*profile[i] ;         
         if (fabs(dtau)>1.0e-5f)  {            
            NTRUE[i] +=  emissivity*profile[i]*GN*exp(-SUM_TAU[i]) * (1.0f-exp(-dtau)) / dtau ;
         } else {
            NTRUE[i] += (emissivity*profile[i]*GN*exp(-SUM_TAU[i])) ;
         }
         SUM_TAU[i] += dtau  ;
      }
      POS  += dx*DIR ;
      INDEX = Index(POS) ;
   } // while INDEX
   for (i=0; i<NCHN; i++) NTRUE[i] -=  BG*(1.0f-exp(-SUM_TAU[i])) ;
   
}


#endif // WITH_HFS





