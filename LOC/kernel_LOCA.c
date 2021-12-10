// #define EPS 6.0e-5f  BAD !!!
#define EPS 4.0e-4f
#define UNROLL 1
#define PRECALCULATED_GAUSS 1
// without precalculated GAU total runtime increases x4 !!




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




__kernel void Clear(__global float *SIJ.
                    __global float *ESC) {  // [CELLS] SIJ, ESC
   // Clear the arrays before simulation of the current transition
   int id = get_global_id(0) ;   
   for(int i=id; i<CELLS; i+=GLOBAL) {
      SIJ[i] = 0.0f ;
      ESC[i] = 0.0f ;
   }
}




int Index(const float4 pos) {
   if ((pos.x<=0.0f)||(pos.y<=0.0f)||(pos.z<=0.0f))  return -1 ;
   if ((pos.x>=   X)||(pos.y>=   Y)||(pos.z>=   Z))  return -1 ;
   // without (int), this FAILS for cloud sizes >256^3 !!
   return  (int)trunc(pos.x) + X*((int)trunc(pos.y)+Y*(int)trunc(pos.z)) ;
}




__kernel void Paths(
#if (WITH_PL==1)
                    __global float   *PL,      // [CELLS]
#endif
                    __global float   *TPL,     // [NRAY]
                    __global int     *COUNT,   // [NRAY] total number of rays entering the cloud
                    const    int      LEADING, // leading edge
                    const    float4   POS0,    // initial position of ray 0
                    const    float4   DIR      // direction of the rays
                   ) {   
   float tpl, dx, dy, dz ;
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge, if ray exists through a side, a new one is created 
   // on the opposite side. Ray ends when the downstream edge is reached
   int  id  =  get_global_id(0), INDEX ;
   int  ny  =  (Y+1)/2 ;
   int  nz  =  (Z+1)/2 ;
   
   barrier(CLK_LOCAL_MEM_FENCE) ;
   
   if (id>=NRAY) return ;  // NRAY = ((X+1)/2) * ((Y+1)/2)
   COUNT[id] = 0 ;
   tpl       = 0 ;
   
   // each ray shifted by two grid units
   float4 POS = POS0 ;
   switch (LEADING) {
    case 0: POS.x = EPS ;   POS.y += 2.0f*(id/nz) ; POS.z += 2.0f*(id%nz) ;  break ;
    case 1: POS.x = X-EPS ; POS.y += 2.0f*(id/nz) ; POS.z += 2.0f*(id%nz) ;  break ;
    case 2: POS.y = EPS ;   POS.x += 2.0f*(id/nz) ; POS.z += 2.0f*(id%nz) ;  break ;
    case 3: POS.y = Y-EPS ; POS.x += 2.0f*(id/nz) ; POS.z += 2.0f*(id%nz) ;  break ;
    case 4: POS.z = EPS ;   POS.x += 2.0f*(id/ny) ; POS.y += 2.0f*(id%ny) ;  break ;
    case 5: POS.z = Z-EPS ; POS.x += 2.0f*(id/ny) ; POS.y += 2.0f*(id%ny) ;  break ;
   }
   INDEX = Index(POS) ;
   if (INDEX>=0)   COUNT[id] += 1 ;
   while(INDEX>=0) {
      if (DIR.x<0.0f)   dx = -      fmod(POS.x,1.0f)  / DIR.x ;
      else              dx =  (1.0f-fmod(POS.x,1.0f)) / DIR.x ;
      if (DIR.y<0.0f)   dy = -      fmod(POS.y,1.0f)  / DIR.y ;
      else              dy =  (1.0f-fmod(POS.y,1.0f)) / DIR.y ;
      if (DIR.z<0.0f)   dz = -      fmod(POS.z,1.0f)  / DIR.z ;
      else              dz =  (1.0f-fmod(POS.z,1.0f)) / DIR.z ;
      dx         = min(dx, min(dy, dz)) + EPS ;
#if (WITH_PL==1)
      PL[INDEX] += dx ;           // path length, cumulative over idir and ioff
#endif
      tpl       += dx ;           // just the total value for current idir, ioff
      POS       += dx*DIR ;
      INDEX      = Index(POS) ;
      if (INDEX<0) {  // exits the cloud... but on which side?
         if (POS.x<0.0f) {
            if (LEADING!=1)  POS.x = X-EPS ;    // create new ray
         } 
         if (POS.x>X) {
            if (LEADING!=0)  POS.x = EPS ;      // create new ray
         }
         if (POS.y<0.0f) {
            if (LEADING!=3)  POS.y = Y-EPS ;    // create new ray
         } 
         if (POS.y>Y) {
            if (LEADING!=2)  POS.y = EPS ;      // create new ray
         }
         if (POS.z<0.0f) {
            if (LEADING!=5)  POS.z = Z-EPS ;    // create new ray
         } 
         if (POS.z>Z) {
            if (LEADING!=4)  POS.z = EPS ;      // create new ray
         }
         INDEX = Index(POS) ;
         if (INDEX>=0)  COUNT[id] += 1 ;
      }
   } // while INDEX>=0
   TPL[id] = tpl ;
}




__kernel void Update(
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
                     const float4     POS0,    // 10 initial position of id=0 ray
                     const float4     DIR,     // 11 ray direction
                     __global float2 *NI,      // 12 [CELLS]:  NI[upper] + NB_NB
                     __global float  *SIJ,     // 13 [CELLS]:  SIJ
                     __global float  *ESC      // 14 [CELLS]:  ESC
#if (LOC_LOWMEM==1)
                     ,__global float *NTRUES   // 15 [GLOBAL*MAXCHN]
#endif
                    )  {
   float weight ;        
   float dx, doppler ;
   float tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;
   float sum_delta_true, all_escaped, nu ;
   int row, shift, INDEX, c1, c2 ;
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation!!
   // Rays start on the leading edge 
   // if ray exists through a side, a new one is created on the opposite side
   // Ray ends when the downstream edge is reached
   int  id = get_global_id(0) ;
   int lid = get_local_id(0) ;
   if (id>=NRAY) return ;
   int ny=(Y+1)/2, nz=(Z+1)/2 ; // dimensions of the current ray grid
   
#if (PRECALCULATED_GAUSS==1)
   constant float* profile ;
#else
   __local float profile_array[LOCAL*CHANNELS] ; // ???????????????????????????????????????????????????????
   __local float *profile = &profile_array[lid*CHANNELS] ;
#endif
   
   
#if (LOC_LOWMEM==1)
   __global float *NTRUE = &NTRUES[id*CHANNELS] ;
#else
   // this is ok for CPU
   __local float  NTRUESSS[LOCAL*CHANNELS] ;
   __local float *NTRUE = &NTRUESSS[lid*CHANNELS] ;
#endif
   
   
   // Initial position of each ray shifted by two grid units
   //  == host has a loop over 4 offset positions
   float4 POS = POS0 ;
   switch (LEADING) {
    case 0:  POS.x = EPS ;    POS.y += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 1:  POS.x = X-EPS ;  POS.y += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 2:  POS.y = EPS ;    POS.x += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 3:  POS.y = Y-EPS ;  POS.x += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 4:  POS.z = EPS ;    POS.x += 2.0f*(id/ny) ;   POS.y += 2.0f*(id%ny) ;   break ;
    case 5:  POS.z = Z-EPS ;  POS.x += 2.0f*(id/ny) ;   POS.y += 2.0f*(id%ny) ;   break ;
   }
   INDEX = Index(POS) ;
   for(int i=0; i<CHANNELS; i++) NTRUE[i] = BG * DIRWEI ;
   
   while(INDEX>=0) {
      
      
      
      dx =      (DIR.x<0.0f) ? (-fmod(POS.x,1.0f)/DIR.x-EPS/DIR.x) : ((1.0f-fmod(POS.x,1.0f))/DIR.x+EPS/DIR.x) ;
      dx=min(dx,(DIR.y<0.0f) ? (-fmod(POS.y,1.0f)/DIR.y-EPS/DIR.y) : ((1.0f-fmod(POS.y,1.0f))/DIR.y+EPS/DIR.y)) ;
      dx=min(dx,(DIR.z<0.0f) ? (-fmod(POS.z,1.0f)/DIR.z-EPS/DIR.z) : ((1.0f-fmod(POS.z,1.0f))/DIR.z+EPS/DIR.z)) ;      
      nu        =  NI[INDEX].x ;
      nb_nb     =  NI[INDEX].y ;
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
      weight    =  (dx/APL)*VOLUME ;  // VOLUME == 1.0/CELLS, fraction of cloud volume
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
      
      
#if 0 // no difference between these two! = small tau tmp_emit not important
      tmp_tau   =  dx*nb_nb*GN  ;
      tmp_emit  =  DIRWEI * weight * nu*Aul / tmp_tau ;  // GN include grid length [cm]
      if (fabs(tmp_tau)<1.0e-10f) {
         tmp_tau  = 1.0e-10f ;
         tmp_emit = 0.0f ;
      }
#else
      if (fabs(nb_nb)<1.0e-37f) nb_nb=1.0e-37f ;  // was e-32
      tmp_tau   =  dx*nb_nb*GN ;
      tmp_emit  =  DIRWEI * weight * nu*Aul / tmp_tau ;  // GN include grid length [cm]
#endif
      
      
      
      shift     =  round(doppler/WIDTH) ;
      
      
#if (PRECALCULATED_GAUSS==1)
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
      profile   =  &GAU[row*CHANNELS] ;
      // avoid profile function outside profile channels LIM.x, LIM.y
      c1        =  LIM[row].x+shift ;
      c2        =  LIM[row].y+shift ;
      c1        =  max(c1,   max(shift,      0               )) ;
      c2        =  min(c2,   min(CHANNELS-1, CHANNELS-1+shift)) ;
#else // ***NOT*** PRECALCULATED_GAUSS
      // calculate profile function from the start
      float tmp = 0.0f, dv ;
      float s   = 0.002f*CLOUD[INDEX].w ;
      c1=-1 ; c2=-1 ;
      for(int col=0; col<CHANNELS; col++)  {
         dv            = (-0.5f*CHANNELS-0.5f+col)*WIDTH / s ;
         dv            = exp(-dv*dv) ;
         profile[col]  = dv ;
         tmp          += dv ;
         if (c1<0) {                       // not yet set
            if (dv>1.0e-6f) c1 = col;      // when profile first rises above threshold
         } else {
            if (c2<0) {                    // if not yet set
               if (dv<1.0e-6f) c2 = col ;  // when profile falls below threshold
            }
         }
      }
      if (c2<0) c2 = CHANNELS-1 ;    // in case of very wide profile
      for(int col=0; col<CHANNELS; col++)  profile[col] /= tmp ;
      // avoid profile function outside profile channels c1 c2
      c1        =  c1+shift ;
      c2        =  c2+shift ;
      c1        =  max(c1,   max(shift,      0               )) ;
      c2        =  min(c2,   min(CHANNELS-1, CHANNELS-1+shift)) ;      
#endif
      
      // ONLY CORE SATURATION VERSION !!
      
      sum_delta_true = all_escaped = 0.0f ;
#if (UNROLL==0)
      for(int ii=c1; ii<=c2; ii++)  {
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed         =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped     +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
#else
      // try manual unroll
# define UR 8
      int ii, no = (c2-c1)/UR ;
      ii = c2+1 ;
      for(int i=0; i<no; i++) {
         // 0
         ii               =  c1+UR*i+0 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 1
         ii               =  c1+UR*i+1 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 2
         ii               =  c1+UR*i+2 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 3
         ii               =  c1+UR*i+3 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 4
         ii               =  c1+UR*i+4 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 5
         ii               =  c1+UR*i+5 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 6
         ii               =  c1+UR*i+6 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
         // 7
         ii               =  c1+UR*i+7 ;
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ; 
         absorbed         =  NTRUE[ii]*factor ;
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;       
         all_escaped     +=  escape ;          
      }
      for(ii++; ii<=c2; ii++)  {
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii-shift]) ;         
         escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed         =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped     +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
#endif
      
      
      
      // Update SIJ and ESC
#if 1
      SIJ[INDEX]  +=  (A_b/VOLUME) * sum_delta_true /  nb_nb ;
      ESC[INDEX]  +=  all_escaped ;
#else
      atomicAdd_g_f(&(SIJ[INDEX]), (A_b/VOLUME) * sum_delta_true / nb_nb) ;
      atomicAdd_g_f(&(ESC[INDEX]), all_escaped) ;      
#endif
      
      
      POS       += dx*DIR ;
      INDEX      = Index(POS) ;
      
      if (INDEX<0) {  // exits the cloud... but on which side?
         if (POS.x<=0.0f) {
            if (LEADING!=1)  POS.x = X-EPS ;
         } 
         if (POS.x>=X) {
            if (LEADING!=0)  POS.x = EPS ;  
         }
         if (POS.y<=0.0f) {
            if (LEADING!=3)  POS.y = Y-EPS ;
         } 
         if (POS.y>=Y) {
            if (LEADING!=2)  POS.y = EPS ;  
         }
         if (POS.z<=0.0f) {
            if (LEADING!=5)  POS.z = Z-EPS ;
         } 
         if (POS.z>=Z) {
            if (LEADING!=4)  POS.z = EPS ;  
         }
         INDEX = Index(POS) ;
         if (INDEX>=0) {   // new ray started on the opposite side (same work item)
            for(int ii=0; ii<CHANNELS; ii++) NTRUE[ii] = BG * DIRWEI ;
         }
      } // if INDEX<0
      
   } // while INDEX>=0
   
}












__kernel void Spectra(
                      __global float4 *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
                      constant float  *GAU,          //  1 precalculated gaussian profiles
                      constant int2   *LIM,          //  2 limits of ~zero profile function
                      const float      GN,           //  3 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                      const float2     D,            //  4 ray direction == theta, phi
                      __global float2 *NI,           //  5 [CELLS]:  NI[upper] + NB_NB
                      const float      RA,           //  6 grid units, offset in RA direction
                      const int        NDEC,         //  7 number of DEC points = work items
                      const float      STEP,         //  8 step between spectra (grid units)
                      const float      BG,           //  9 background intensity
                      const float      emis0,        // 10 h/(4pi)*freq*Aul*int2temp
                      __global float  *NTRUE_ARRAY,  // 11 NDEC*CHANNELS
                      __global float  *SUM_TAU_ARRAY // 12 NDEC*CHANNELS
                     )
{
   // each work item calculates one spectrum for 
   //   ra  =  RA (grid units, from cloud centre)
   //   de  =  DE(id)
   int id = get_global_id(0) ;
   if (id>=NDEC) return ; // no more rays
   
   __global float *NTRUE   = &(NTRUE_ARRAY[id*CHANNELS]) ;
   __global float *SUM_TAU = &(SUM_TAU_ARRAY[id*CHANNELS]) ;
   
   int i ;
   float DE ; // grid units, offset of current ray
   DE  =   (id-0.5f*(NDEC-1.0f))*STEP ;
   
   
   
   // calculate the initial position of the ray
   float4 POS, DIR ;
   float dx, dy, dz ;
   POS.x   =  0.500001f*X ;  POS.y = 0.500001f*Y ;  POS.z = 0.500001f*Z ;
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
   if (fabs(DIR.x)<1.0e-10f) DIR.x = 1.0e-10f ;
   if (fabs(DIR.y)<1.0e-10f) DIR.y = 1.0e-10f ;
   if (fabs(DIR.z)<1.0e-10f) DIR.z = 1.0e-10f ;
   
   // printf("ORI   POS %8.4f %8.4f %8.4f  DIR %8.4f %8.4f %8.4f\n", POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z) ;
   
   // go to front surface
   POS -=  1000.0f*DIR ;   // make sure it will be a step forward
   if (DIR.x>0.0f)  dx = (0.0f-POS.x)/DIR.x ;
   else             dx = (X-POS.x)/DIR.x ;
   if (DIR.y>0.0f)  dy = (0.0f-POS.y)/DIR.y ;
   else             dy = (Y-POS.y)/DIR.y ;
   if (DIR.z>0.0f)  dz = (0.0f-POS.z)/DIR.z ;
   else             dz = (Z-POS.z)/DIR.z ;
   dx          =  max(dx, max(dy, dz)) + 1.0e-4f ;  // max because we are outside
   POS        +=  dx*DIR ;
   int INDEX   =  Index(POS) ;
   
   for(int i=0; i<CHANNELS; i++) {
      NTRUE[i]   = 0.0f ;
      SUM_TAU[i] = 0.0f ;
   }
   
   float tau, dtau, emissivity, doppler, nu ;
   int row, shift, c1, c2  ;
   constant float* profile ;
   
   
   // printf("emis0 %12.4e, nu %12.4e, nbnb %12.4e, GL %12.4e\n", emis0, NI[INDEX].x, NI[INDEX].y, GL) ;
   
   while (INDEX>=0) {
      if (DIR.x<0.0f)   dx = -      fmod(POS.x,1.0f)  / DIR.x - EPS/DIR.x;
      else              dx =  (1.0f-fmod(POS.x,1.0f)) / DIR.x + EPS/DIR.x;
      if (DIR.y<0.0f)   dy = -      fmod(POS.y,1.0f)  / DIR.y - EPS/DIR.y;
      else              dy =  (1.0f-fmod(POS.y,1.0f)) / DIR.y + EPS/DIR.y;
      if (DIR.z<0.0f)   dz = -      fmod(POS.z,1.0f)  / DIR.z - EPS/DIR.z;
      else              dz =  (1.0f-fmod(POS.z,1.0f)) / DIR.z + EPS/DIR.z;
      dx         =  min(dx, min(dy, dz)) + EPS ;  // actual step
      nu         =       NI[INDEX].x ;     
      
#if 0
      tau        =  dx * NI[INDEX].y * GN * GL ; // need to separate GN and GL ?
      tau        =  clamp(tau, -0.05f, 1.0e5f) ;    // NO MASERS !!
#else
      if (fabs(NI[INDEX].y)<1.0e-24f) {
         tau     =  dx * 1.0e-24f * GN * GL ;
      } else {
         tau     =  dx * NI[INDEX].y * GN * GL ;
         
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
      row        =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      profile    =  &GAU[row*CHANNELS] ;
      shift      =  round(doppler/WIDTH) ;
      // printf("SHIFT %2d\n", shift) ;
      c1         =  LIM[row].x+shift ;
      c2         =  LIM[row].y+shift ;
      c1         =  max(c1,   max(shift,      0               )) ;
      c2         =  min(c2,   min(CHANNELS-1, CHANNELS-1+shift)) ;
      
      // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      emissivity =  emis0 * nu * dx * GL ;
      
      for(i=c1; i<=c2; i++) {
         
         dtau = tau*profile[i-shift] ;
         
         if (fabs(dtau)>1.0e-5f)  {            
            NTRUE[i] += 
              emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i]) *
              (1.0f-exp(-dtau)) / dtau ;
         } else {
            NTRUE[i] += (emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i])) ;
         }
         
         SUM_TAU[i] += dtau  ;
      }
      
      POS  += dx*DIR ;
      INDEX = Index(POS) ;
   } // while INDEX
   
   for (i=0; i<CHANNELS; i++) NTRUE[i] -=  BG*(1.0f-exp(-SUM_TAU[i])) ;
   
}




// ==============================================================================





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
                       const float4     POS0,    // 10 initial position of id=0 ray
                       const float4     DIR,     // 11 ray direction
                       __global float2 *NI,      // 12 [CELLS]:  NI[upper] + NB_NB
                       __global float  *SIJ,     // 13 [CELLS]:  SIJ, ESC
                       __global float  *ESC,
                       const int        NCHN,    // 14 number of channels (>= CHANNELS)
                       const int        NCOMP,   // 15 number of HF components
                       __global float2 *HF       // 16 [].x = offset, [].y = weight
#if (LOC_LOWMEM==1)
                       ,
                       __global float *NTRUES,   // 17 [GLOBAL*MAXCHN]
                       __global float *PROFS     // 18 [GLOBAL*MAXCHN]
#endif
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
   int ny=(Y+1)/2, nz=(Z+1)/2 ; // dimensions of the current ray grid
   
   // *** MUST HAVE PRECALCULATED_GAUSSIAN ***
   constant float *pro ;             // pointer to GAU vector
   
#if (LOC_LOWMEM==1)
   __global float *NTRUE   = &NTRUES[id*MAXCHN] ;
   // __global float *profile =  &PROFS[id*MAXCHN] ;
#else   // this leads to "insufficient resources" in case of GPU
   __local float  NTRUESSS[LOCAL*MAXCHN] ;
   __local float *NTRUE = &NTRUESSS[lid*MAXCHN] ;
#endif
   
   // It is ok up to ~150 channels to have these on GPU local memory
   __local float  profile_array[LOCAL*MAXCHN] ;  // MAXCHN = max of NCHN
   __local float *profile = &profile_array[lid*MAXCHN] ;   
   
   // Initial position of each ray shifted by two grid units
   //  == host has a loop over 4 offset positions
   float4 POS = POS0 ;
   switch (LEADING) {
    case 0:  POS.x = EPS ;    POS.y += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 1:  POS.x = X-EPS ;  POS.y += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 2:  POS.y = EPS ;    POS.x += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 3:  POS.y = Y-EPS ;  POS.x += 2.0f*(id/nz) ;   POS.z += 2.0f*(id%nz) ;   break ;
    case 4:  POS.z = EPS ;    POS.x += 2.0f*(id/ny) ;   POS.y += 2.0f*(id%ny) ;   break ;
    case 5:  POS.z = Z-EPS ;  POS.x += 2.0f*(id/ny) ;   POS.y += 2.0f*(id%ny) ;   break ;
   }
   INDEX = Index(POS) ;
   for(int i=0; i<NCHN; i++) NTRUE[i] = BG * DIRWEI ;
   
   while(INDEX>=0) {
      
      dx =     ((DIR.x<0.0f) ? (-fmod(POS.x,1.0f)/DIR.x-EPS/DIR.x) : ((1.0f-fmod(POS.x,1.0f))/DIR.x+EPS/DIR.x)) ;
      dx=min(dx,(DIR.y<0.0f) ? (-fmod(POS.y,1.0f)/DIR.y-EPS/DIR.y) : ((1.0f-fmod(POS.y,1.0f))/DIR.y+EPS/DIR.y)) ;
      dx=min(dx,(DIR.z<0.0f) ? (-fmod(POS.z,1.0f)/DIR.z-EPS/DIR.z) : ((1.0f-fmod(POS.z,1.0f))/DIR.z+EPS/DIR.z)) ;      
      // nu        =  NI[INDEX].x ;
      nb_nb     =  NI[INDEX].y ;
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
      weight    =  (dx/APL)*VOLUME ;  // VOLUME == 1.0/CELLS, fraction of cloud volume
      

      POS       += dx*DIR ;
      
      if (fabs(nb_nb)<1.0e-37f) nb_nb=1.0e-37f ;  // was e-32
      tmp_tau   =  dx*nb_nb*GN ;
      tmp_emit  =  DIRWEI * weight * NI[INDEX].x * Aul / tmp_tau ;  // GN include grid length [cm]

      // reusing dx for doppler !
      dx  =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
      
      row =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
      
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
      
#if (UNROLL==0)  // no manual unroll
      for(int ii=0; ii<NCHN; ii++)  {
         factor           =  1.0f-native_exp(-tmp_tau*profile[ii]) ;
         escape           =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed         =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]       +=  escape-absorbed ;
         sum_delta_true  +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped     +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
#else
# define UR 8
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
#endif
      
      
      
#if 1
      SIJ[INDEX]  +=  (A_b/VOLUME) * sum_delta_true / nb_nb ;      
      ESC[INDEX]  +=  all_escaped ;
#else
      atomicAdd_g_f(&(SIJ[INDEX]), (A_b/VOLUME) * sum_delta_true / nb_nb) ;
      atomicAdd_g_f(&(ESC[INDEX]), all_escaped) ;
#endif
      
      // POS       += dx*DIR ;
      INDEX      = Index(POS) ;
      
      if (INDEX<0) {  // exits the cloud... but on which side?
         if (POS.x<=0.0f) {
            if (LEADING!=1)  POS.x = X-EPS ;
         } 
         if (POS.x>=X) {
            if (LEADING!=0)  POS.x = EPS ;  
         }
         if (POS.y<=0.0f) {
            if (LEADING!=3)  POS.y = Y-EPS ;
         } 
         if (POS.y>=Y) {
            if (LEADING!=2)  POS.y = EPS ;  
         }
         if (POS.z<=0.0f) {
            if (LEADING!=5)  POS.z = Z-EPS ;
         } 
         if (POS.z>=Z) {
            if (LEADING!=4)  POS.z = EPS ;  
         }
         INDEX = Index(POS) ;
         if (INDEX>=0) {   // new ray started on the opposite side (same work item)
            for(int ii=0; ii<NCHN; ii++) NTRUE[ii] = BG * DIRWEI ;
         }
      } // if INDEX<0
      
   } // while INDEX>=0
   
}







#ifdef WITH_HFS


__kernel void SpectraHF(
                        __global float4 *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
                        constant float  *GAU,          //  1 precalculated gaussian profiles
                        constant int2   *LIM,          //  2 limits of ~zero profile function
                        const float      GN,           //  3 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                        const float2     D,            //  4 ray direction == theta, phi
                        __global float2 *NI,           //  5 [CELLS]:  NI[upper] + NB_NB
                        const float      RA,           //  6 grid units, offset in RA direction
                        const int        NDEC,         //  7 number of DEC points = work items
                        const float      STEP,         //  8 step between spectra (grid units)
                        const float      BG,           //  9 background intensity
                        const float      emis0,        // 10 h/(4pi)*freq*Aul*int2temp
                        __global float  *NTRUE_ARRAY,  // 11 NDEC*MAXNCHN
                        __global float  *SUM_TAU_ARRAY,// 12 NDEC*MAXNCHN
                        const int NCHN,                // 13 channels (in case of HF spectrum)
                        const int NCOMP,               // 14 number of components
                        __global float2 *HF            // 15 channel offsets, weights
                       )
{
   // printf("SpectraHF\n") ;
   // each work item calculates one spectrum for 
   //   ra  =  RA (grid units, from cloud centre)
   //   de  =  DE(id)
   int  id = get_global_id(0) ;
   if (id>=NDEC) return ; // no more rays
   int lid = get_local_id(0) ;
   
   __global float *NTRUE   = &(NTRUE_ARRAY[id*MAXCHN]) ;
   __global float *SUM_TAU = &(SUM_TAU_ARRAY[id*MAXCHN]) ;
   int i ;
   float DE ; // grid units, offset of current ray
   DE  =   (id-0.5f*(NDEC-1.0f))*STEP ;
   
   
   // calculate the initial position of the ray
   float4 POS, DIR ;
   float dx, dy, dz ;
   POS.x   =  0.500001f*X ;  POS.y = 0.500001f*Y ;  POS.z = 0.500001f*Z ;
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
   if (fabs(DIR.x)<1.0e-10f) DIR.x = 1.0e-10f ;
   if (fabs(DIR.y)<1.0e-10f) DIR.y = 1.0e-10f ;
   if (fabs(DIR.z)<1.0e-10f) DIR.z = 1.0e-10f ;
   
   // printf("ORI   POS %8.4f %8.4f %8.4f  DIR %8.4f %8.4f %8.4f\n", POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z) ;
   
   // go to front surface
   POS -=  1000.0f*DIR ;   // make sure it will be a step forward
   if (DIR.x>0.0f)  dx = (0.0f-POS.x)/DIR.x ;
   else             dx = (X-POS.x)/DIR.x ;
   if (DIR.y>0.0f)  dy = (0.0f-POS.y)/DIR.y ;
   else             dy = (Y-POS.y)/DIR.y ;
   if (DIR.z>0.0f)  dz = (0.0f-POS.z)/DIR.z ;
   else             dz = (Z-POS.z)/DIR.z ;
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
      nu         =       NI[INDEX].x ;     
      
      if (fabs(NI[INDEX].y)<1.0e-24f) {
         tau     =  dx * 1.0e-24f * GN * GL ;
      } else {
         tau     =  dx * NI[INDEX].y * GN * GL ;
         
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
      row        =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      
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

