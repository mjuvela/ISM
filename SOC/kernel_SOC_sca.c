#include "kernel_SOC_aux.c"


#define MAX_SCATTERINGS   30      // 20  ... 100  ... 30
#define RUSSIAN_ROULETTE   1

// Intel OpenCL has a bug (?) that prevents the use of for(BLOOP=...) loop ?
#define BLOOP_FUDGE 1




__kernel void zero_out(const int       NDIR,
                       const int2      NPIX,
                       __global float  *OUT)   {
   // Clear the OUT array (used to gather outcoming photons) -- space for one frequency
   const int id  = get_global_id(0) ;
   const int gs  = get_global_size(0) ;
   for(int i=id; i<NDIR*NPIX.x*NPIX.y; i+=gs)  OUT[i] = 0.0f ;
}


// KERNEL FOR CONSTANT SOURCES




__kernel void SimRAM_HP(const      int      PACKETS,   //  0 - number of packets
                        const      int      BATCH,     //  1 - for SOURCE==2, packages per cell
                        const      float    SEED,      //  2 
                        __global   float*   ABS,       //  3 
                        __global   float*   SCA,       //  4 
                        constant   int     *LCELLS,    //  5 - number of cells on each level
                        constant   int     *OFF,       //  6 - index of first cell on each level
                        __global   int     *PAR,       //  7 - index of parent cell        [CELLS]
                        __global   float   *DENS,      //  8 - density and hierarchy       [CELLS]
                        constant   float   *DSC,       //  9 - BINS entries [BINS] or [NDUST,BINS]
                        constant   float   *CSC,       // 10 - cumulative scattering function
                        // scattering
                        const      int      NDIR,      // 11 - number of scattering maps = directions
                        __global   float3  *ODIRS,     // 12 - observer directions
                        const      int2     NPIX,      // 13 - map has NPIX.x * NPIX.y pixels
                        const      float    MAP_DX,    // 14 - map pixel size in root grid units
                        const      float3   CENTRE,    // 15 - map centre 
                        __global   float3  *ORA,       // 16 - unit vector for RA coordinate axis
                        __global   float3  *ODE,       // 17 - unit vector for DE coordinate axis
                        __global   float   *OUT,       // 18 -- scattering arrays, NDIR*NPIX*NPIX
                        __global   float   *ABU,       // 19 - for -DWITH_MSF, abundances of each dust species
                        __global   OTYPE   *OPT,       // 20
                        __global   float   *BG,        // 21 - Healpix map of background sky
                        __global   float   *HPBGP      // 22 - Cumulative probability for selecting pixels
                       ) 
{
   // Background emission using a Healpix map
   const int id     = get_global_id(0) ;
   const int GLOBAL = get_global_size(0) ;
   int    oind=0, level=0, scatterings, SIDE, i, j ;
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, theta ;
   float3 DIR=0.0f, POS, POS0, ODIR ; 
   float  PHOTONS, x, y, z, v1, v2, W ;
   mwc64x_state_t rng;
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   // MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod((double)(SEED*(id+PI)),1.0)*4294967296L), samplesPerStream);
   
   int ind   = -1 ;
   int ind0=-1, level0, idust=0 ;
   float Rout =  0.5f*sqrt(1.0f*NX*NX+NY*NY+NZ*NZ) ;
   
   
   for(int III=0; III<BATCH; III++) {
      
#if (HPBG_WEIGHTED<1)
      // Unweighted: select *random* Healpix pixel for the emission
      ind     =   clamp((int)(floor(Rand(&rng)*49152)), 0, 49151) ;  // NSIDE==64
      PHOTONS =   BG[ind] ;
#else
      // Select Healpix pixel according to the cumulative probability in HPBGP
      //   NSIDE==64 =>  NPIX = 49152 ==> need at most 16 comparisons... not fast but fast enough (?)
      x       =   Rand(&rng) ;
      ind0    =   0 ;
      level0  =   49151 ;
      for(int i=0; i<10; i++) {       // 12 divisions -> 12 cells remain in the interval
         ind  =   (ind0+level0)/2 ;
         if (HPBGP[ind]>x) level0 = ind ;
         else              ind0   = ind ;
      }
      for(ind=ind0; ind<=level0; ind++) {  // direct loop < 12 cells
         if (HPBGP[ind]>=x) break ;  // remove equality and you will read beyond the array!!
      }
      // ind = clamp(ind, 0, 49151) ; // not needed if the above >= works... also safeguarded on the host side
      // Weighting for the relative probability between pixels is already included in BG !!
      PHOTONS =   BG[ind] ;
#endif
      
      // Calculate vector for the direction *from* the Healpix pixel "ind"
      Pixel2AnglesRing(64, ind, &phi, &theta) ;      
      DIR.x   =  +sin(theta)*cos(phi) ;
      DIR.y   =  +sin(theta)*sin(phi) ;   
      DIR.z   =  -cos(theta) ;      
      if (fabs(DIR.x)<DEPS) DIR.x = DEPS ;
      if (fabs(DIR.y)<DEPS) DIR.y = DEPS ;
      if (fabs(DIR.z)<DEPS) DIR.z = DEPS ; 
      DIR         =  normalize(DIR) ;
      
#if 0
      // *** THIS IGNORES THE FACT THAT PROJECTED CLOUD AREA DEPENDS ON THE DIRECTION ***
      // source is at infinite distance => always three sides of the model illuminated
      // calculate the relative projected surface areas and select one of the three
      //   area ~ cos(LOS vs. surface normal)  =  component of the unit vector
      x   =   fabs(DIR.x) ;   // +/- X plane
      y   =   fabs(DIR.y) ;
      z   =   fabs(DIR.z) ;
      ds  =   x+y+z ;    x /= ds ;    y /= ds ;   z /= ds ;
      ds  =   Rand(&rng) ;     v1 = Rand(&rng) ;    v2 =  Rand(&rng) ;
      if (ds<x) {           // hit an element at surface x=0 or x=NX
         POS.y = v1*NY ;   POS.z = v2*NZ ;
         POS.x = (DIR.x>0.0f) ? ( PEPS ) : ( NX-PEPS) ;
      }  else {
         if (ds<(x+y)) {    // hit an element at surface y=0 or y=NY
            POS.x = v1*NX ;   POS.z = v2*NZ ;
            POS.y = (DIR.y>0.0f) ? ( PEPS ) : ( NY-PEPS) ;
         } else {           // hit an element at surface z=0 or z=NZ
            POS.x = v1*NX ;   POS.y = v2*NY ;
            POS.z = (DIR.z>0.0f) ? ( PEPS ) : ( NZ-PEPS) ;
         }
      }     
#else
      // Package directed towards the cloud, onto a surface of a sphere with radius Rout
      ds     =  2.0*PI*Rand(&rng) ;   
      dx     =  sqrt(Rand(&rng)) ;
      POS.x  =  dx*cos(ds) ;   POS.y = dx*sin(ds) ;   POS.z = sqrt(1.001f-dx*dx) ;
      // This would be a random position on the surface if healpix pixel were towards +Z
      // However, the healpix pixel is now in the direction (theta,phi)
      // (1) rotation theta around +Y
      POS0.x =  POS.x*cos(theta) + POS.z*sin(theta) ;
      POS0.y =  POS.y ;
      POS0.z = -POS.x*sin(theta) + POS.z*cos(theta) ;
      // (2) rotation phi around new +Z
      //     phi is now angle from -X, not from +X !! because healpix (l,b)=(0,0) is towards [-1,0,0]
      //    => rotation is by angle pi-phi
      POS.x  =  POS0.x*cos(PI-phi) + POS0.y*sin(PI-phi) ;
      POS.y  = -POS0.x*sin(PI-phi) + POS0.y*cos(PI-phi) ;
      POS.z  =  POS0.z ;
      //
      POS.x  =  0.5f*NX + Rout*POS.x ;
      POS.y  =  0.5f*NY + Rout*POS.y ;
      POS.z  =  0.5f*NZ + Rout*POS.z ;
      // Position should now be on the sphere, on the same side of the cloud as the Healpix pixel
      // -> try to step onto the actual cloud surface -- part of rays will be rejected (miss the cloud)
      Surface(&POS, &DIR) ;   // input POS should be outside the cloud, on the upstream side
#endif
      
      IndexG(&POS, &level, &ind, DENS, OFF) ;      
      if (ind<0) continue ; // we are inside the loop: for II in BATCH 
      
      
      
      
#if (FFS>0)  // Forced first scattering ============================================================
      POS0       =  POS ;     // do not touch the original parameters [POS, ind, level]
      ind0       =  ind ;
      level0     =  level ;
      tau        =  0.0f ;
      while(ind0>=0) {
         oind    =  OFF[level0]+ind0 ;
         ds      =  GetStep(&POS0, &DIR, &level0, &ind0, DENS, OFF, PAR) ; // POS, level, ind updated !!
# if (WITH_ABU>0)
         tau    +=  ds*DENS[oind]*GOPT(2*oind+1) ;
# else
         tau    +=  ds*DENS[oind]*(*SCA) ;
# endif
      }
      if (tau<1.0e-22f) {   // we are inside for II ~ BATCH loop
         ind = -1 ;         // nothing along the LOS
         continue ;
      }
      W          =  1.0f-exp(-tau) ;
      free_path  = -log(1.0-W*Rand(&rng)) ;
      PHOTONS   *=  W ;
#else // No forced first scattering ============================================================
      free_path  = -log(Rand(&rng)) ;
#endif // ========================================================================================
      
      
      
      
      
      
      // Ray defined by [POS, DIR, ind, level]
      scatterings =  0 ;
      
      while(ind>=0) {  // loop until this ray is really finished
         
         
         
         tau = 0.0f ;         
         while(ind>=0) {    // loop until scattering
            ind0      =  ind ;               // indices at the beginning of the step
            level0    =  level ;
            POS0      =  POS ;               // because GetStep does coordinate transformations...
            oind      =  OFF[level0]+ind0 ;  // global index at the beginning of the step
            ds        =  GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // POS, level, ind updated !!
            // step ds from cell [ind0,level0,oind] to cell [ind, level]
#if (WITH_ABU>0)
            dtau      =  ds*DENS[oind]*GOPT(2*oind+1) ;
#else
            dtau      =  ds*DENS[oind]*(*SCA) ;
#endif       
            if (free_path<(tau+dtau)) {  // tau = optical depth since last scattering
               ind = ind0 ;              // this ray is not yet done - only scattered
               break ;                   // scatter before doing the step ds
            }
            tau      +=  dtau ;
         }
         if (ind<0) break ;  // ray is out of the cloud -- break the loop: while(ind>=0)
         
         // we scatter - after a partial step from old position POS
         //  == we are still inside the scattering cell {POS0, level0, ind0, oind}
         scatterings++ ;
         // dtau               =  free_path-tau ;
#if (WITH_ABU>0)
         ds                 =  (free_path-tau)/(GOPT(2*oind+1)*DENS[oind]) ; 
#else
         ds                 =  (free_path-tau)/((*SCA)*DENS[oind]) ;    // actual step forward in GLOBAL coordinates
#endif
         ds                 =  ldexp(ds, level) ;         // in LOCAL coordinates
         POS0               =  POS0 + ds*DIR ;            // POS0 becomes the position of the scattering
         // remove absorptions since last scattering
#if (WITH_ABU>0)
         PHOTONS           *=  exp(-free_path*GOPT(2*oind)/GOPT(2*oind+1)) ;
#else
         PHOTONS           *=  exp(-free_path*(*ABS)/(*SCA)) ;
#endif         
         
         
         // IndexG(&POS0, &level0, &ind0, DENS, OFF) ; // --- no effect on worms
         
         
         // Do peel-off  -- starting at the location of the scattering
         // == start in the scattering cell == { POS0, level0, ind0 } -- oind will be overwritten
         for(int idir=0; idir<NDIR; idir++) {
            POS       =  POS0 ;                                              
            ind       =  ind0 ;    // coordinates of the scattering location
            level     =  level0 ;
            ODIR      =  ODIRS[idir] ;         
            tau       =  0.0f ;            
            while(ind>=0) {                 // loop to surface, towards observer
               oind   =  OFF[level]+ind ;   // cell index at the start of the step
               ds     =  GetStep(&POS, &ODIR, &level, &ind, DENS, OFF, PAR) ; // POS updated !!
#if (WITH_ABU>0)
               tau   +=  ds*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
#else               
               tau   +=  ds*DENS[oind]*((*ABS)+(*SCA))   ;
#endif
            }
            // time to add something to the scattering array
            cos_theta =  clamp(DIR.x*ODIR.x+DIR.y*ODIR.y+DIR.z*ODIR.z, -0.9999f, +0.9999f) ;
#if (WITH_MSF>0)
            // Using DSC of a randomly selected dust component -> set idust
            oind      =  OFF[level0]+ind0 ;                   // back to the scattering cell
            dx        =  GOPT(2*oind+1) ;                     // sum(ABU*SCA) for the current cell
            ds        =  0.99999f*Rand(&rng) ;
            for(idust=0; idust<NDUST; idust++) {
               ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
               if (ds<=0.0) break ;
            }                        
#endif
#ifdef HG_TEST
            // fraction = probability of scattering per solid angle
            float G = 0.65f, fraction ;
            fraction  =  (1.0f/(4.0f*PI)) * (1.0f-G*G) / pow(1.0f+G*G-2.0f*G*cos_theta, 1.5f) ;
            // delta     =  PHOTONS *  exp(-tau) *  fraction ;
            delta     =  PHOTONS *  fraction * ((tau>TAULIM) ?  (1.0f-exp(-tau)) : (tau*(1.0f-0.5f*tau))) ;
#else
            delta     =  PHOTONS *  exp(-tau) *  DSC[idust*BINS+clamp((int)(BINS*(1.0f+cos_theta)*0.5f), 0, BINS-1)] ;
#endif
            // coordinates  = projections on (ORA, ODE) vectors
            POS      -=   CENTRE ;   // exit position relative the model centre
            i         =   (0.5f*NPIX.x-0.00005f) - dot(POS, ORA[idir]) * MAP_DX ;
            j         =   (0.5f*NPIX.y-0.00005f) + dot(POS, ODE[idir]) * MAP_DX ;
            if ((i>=0)&&(j>=0)&&(i<NPIX.x)&&(j<NPIX.y)) {
               i     +=  idir*NPIX.x*NPIX.y   +    j*NPIX.x ;
               atomicAdd_g_f(&(OUT[i]), delta) ;
            }
         } // for NDIR -- end of peel-off
         
         
         // IndexG(&POS0, &level0, &ind0, DENS, OFF) ; // -- no effect on worms
         
         
         // return to original indices, at the location of the scattering
         // these are still { POS0, level0, ind0 } == position and indices of the scattering cell
         POS            =  POS0 ;
         ind            =  ind0 ;                // cell has not changed !?
         level          =  level0 ;              // (ind, level) at the beginning of the step = at the end
         oind           =  OFF[level0]+ind0 ;    // global index
         
#if (WITH_MSF==0)
         Scatter(&DIR, CSC, &rng) ;              // new direction, basic case with a single scattering function
#else
         dx        =  GOPT(2*oind+1) ;           // sum(ABU*SCA) for the current cell
         ds        =  0.99999f*Rand(&rng) ;
         for(idust=0; idust<NDUST; idust++) {
            ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
            if (ds<=0.0) break ;
         }
         Scatter(&DIR, &CSC[idust*BINS], &rng) ; // use the scattering function of the ind0:th dust species
#endif
         free_path      = -log(Rand(&rng)) ;         
         
#if (RUSSIAN_ROULETTE==0)
         if (scatterings==MAX_SCATTERINGS) {
            ind = -1 ;  break  ; // go and get next ray
         }
#else
         //  Russian roulette to remove packages
         if (scatterings==MAX_SCATTERINGS) {
            if (Rand(&rng)<0.25f) {
               ind = -1 ; break ;
            } else {
               PHOTONS     *=  1.33333333f ;
               scatterings  =  0 ;
            }
         }
#endif
         
      } // while (ind>0) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      
      
   } // for III
   
   
   
}














__kernel void SimRAM_PB(const      int      SOURCE,    //  0 - PSPAC/BGPAC/CLPAC = 0/1/2
                        const      int      PACKETS,   //  1 - number of packets
                        const      int      BATCH,     //  2 - for SOURCE==2, packages per cell
                        const      float    SEED,      //  3 
                        __global   float   *ABS,       //  4 
                        __global   float   *SCA,       //  5 
                        const      float    BG,        //  6 - background intensity
                        __global   float3  *PSPOS,     //  7 - positions of point sources
                        __global   float   *PS,        //  8 - point source luminosities
                        constant   int     *LCELLS,    //  9 - number of cells on each level
                        constant   int     *OFF,       // 10 - index of first cell on each level
                        __global   int     *PAR,       // 11 - index of parent cell        [CELLS]
                        __global   float   *DENS,      // 12 - density and hierarchy       [CELLS]
                        constant   float   *DSC,       // 13 - BINS entries
                        constant   float   *CSC,       // 14 - cumulative scattering function
                        // scattering
                        const      int      NDIR,      // 15 - number of scattering maps = directions
                        __global   float3  *ODIRS,     // 16 - observer directions
                        const      int2     NPIX,      // 17 - map has NPIX.x * NPIX.y pixels
                        const      float    MAP_DX,    // 18 - map pixel size in root grid units
                        const      float3   CENTRE,    // 19 - map centre 
                        __global   float3  *ORA,       // 20 - unit vector for RA coordinate axis
                        __global   float3  *ODE,       // 21 - unit vector for DE coordinate axis
                        __global   float   *OUT,       // 22 -- scattering arrays, NDIR*NPIX*NPIX
                        __global   float   *ABU,       // 23
                        __global   OTYPE   *OPT,       // 24
                        __global   int     *XPS_NSIDE, // 25 -- sides visible        [  NO_PS]
                        __global   int     *XPS_SIDE,  // 26 -- visible side indices [3*NO_PS]
                        __global   float   *XPS_AREA   // 27 -- solid angles         [3*NO_PS]
                       ) 
{
   // This routine is simulating the radiation field in order to determine the
   // absorbed energy in each cell.
   const int id     = get_global_id(0) ;
   const int GLOBAL = get_global_size(0) ;
   
   int    oind=0, level=0, scatterings, SIDE, i, j ;
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, sin_theta ;
   float3 DIR=0.0f, POS, POS0, ODIR ; 
   float  PHOTONS, X0, Y0, Z0, DX, DY, DZ, v1, v2, W ;
   
   // ulong rng = 123*id + 151313*SEED + id*SEED  ;
   // Rand(&rng) ; Rand(&rng) ;  Rand(&rng) ;
   mwc64x_state_t rng;
   // Assume that 2^38 = 2.7e11 random numbers per worker is sufficient
   // For each NITER, host will also give a new seed [0,1] that is multiplied by 2^32 = 4.3e9
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod((double)(SEED*(id+PI)),1.0)*4294967296L), samplesPerStream);
   
   // if (id==0) printf("*** %.5f %.5f %d ***\n", SEED, Rand(&rng), (int)(fmod((double)(SEED*(id+PI)),1.0)*4294967296L)) ;
   
   int ind   = -1 ;
   int ind0=-1, level0, idust=0 ;
   
   
   // Each work item simulates BATCH packages
   //  SOURCE==0   --  PSPAC   -- all from the same cell
   //  SOURCE==1   --  BGPAC   -- all from the same surface element
   //  #### SOURCE==2   --  CLPAC   -- loop over cells, BATCH from each
   if ((SOURCE==1)&&(id>=(8*AREA))) return ;  // BGPAC=BATCH*(8*AREA), GLOBAL>=8*AREA (2017-12-24)
   //  ####if ((SOURCE==2)&&(id>=CELLS)) return ;  (( 
   
   // 2017-12-24  --  BGPAC = BATCH*8*AREA < BATCH*GLOBAL
  
   if (SOURCE==1) { // BGPAC -- find the surface element and precalculate (X0,Y0,Z0,DX,DY,DZ)
      ind = id  % AREA;        // id == number of the surface element
      DX  = 1.0f ; DY = 1.0f ; DZ = 1.0f ;
      if (ind<(NY*NZ)) {                  // lower X
         SIDE = 0 ;               X0 = PEPS ;     Y0 = ind % NY ;   Z0 = ind/NY ;   DX = 0.0f ;
      } else {
         ind -= NY*NZ ;
         if (ind<(NY*NZ)) {               // upper X
            SIDE = 1 ;            X0 = NX-PEPS ;  Y0 = ind % NY ;   Z0 = ind/NY ;   DX = 0.0f ;
         } else {
            ind -= NY*NZ ;
            if (ind<(NX*NZ)) {            // lower Y
               SIDE = 2 ;         Y0 = PEPS ;     X0 = ind % NX ;   Z0 = ind/NX ;   DY = 0.0f ;
            } else {
               ind -= NX*NZ ;
               if (ind<(NX*NZ)) {         // upper Y
                  SIDE = 3 ;      Y0 = NY-PEPS ;  X0 = ind % NX ;   Z0 = ind/NX ;   DY = 0.0f ;
               } else {
                  ind -= NX*NZ ;
                  if (ind<(NX*NY)) {      // lower Z
                     SIDE = 4 ;   Z0 = PEPS ;     X0 = ind % NX ;   Y0 = ind/NX ;   DZ = 0.0f ;
                  } else {                // upper Z
                     ind -= NX*NY ;
                     SIDE = 5;    Z0 = NZ-PEPS ;  X0 = ind % NX ;   Y0 = ind/NX ;   DZ = 0.0f ;
                  }
               }
            }
         }
      }
   }
   
#if (BLOOP_FUDGE>0)
   int BLOOP = 0 ;
   while(1) 
#else
     for(int BLOOP=0; BLOOP<BATCH; BLOOP++) 
#endif
       {
          
          // =============== PSPAC ===================  BATCH packages per work item
          // 2017-12-25 -- BATCH is multiple of the number of point sources,
          //               each work item loops over the full set of sources
          //    GLOBAL*BATCH is the total number of photon packages over all sources
          //    each work item simulates BATCH packages for each source
          //    II%NO_PS used to loop over all point sources
          if (SOURCE==0) {
             phi       =  TWOPI*Rand(&rng) ;
             cos_theta =  0.999997f-1.999995f*Rand(&rng) ;
             sin_theta =  sqrt(1.0f-cos_theta*cos_theta) ;
             DIR.x     =  sin_theta*cos(phi) ;   
             DIR.y     =  sin_theta*sin(phi) ;
             DIR.z     =  cos_theta ;
             level0    =  BLOOP % NO_PS ;              // index of the point sources
             PHOTONS   =  1.0f*PS[level0] ;          // WPS ~ 1/PSPAC, PSPAC ~ GLOBAL*PATCH
             POS.x     =  PSPOS[level0].x ;   
             POS.y     =  PSPOS[level0].y ;   
             POS.z     =  PSPOS[level0].z ;
             IndexG(&POS, &level, &ind, DENS, OFF) ;  // THIS CHANGES POS !!
         
             if ((ind<0)||(ind>=CELLS)) {   // if pointsource is outside the model volume ================================================================================
                POS.x  =  PSPOS[level0].x ;    POS.y     =  PSPOS[level0].y ;      POS.z     =  PSPOS[level0].z ;
#if (PS_METHOD==0)
                // Basic (inefficient) isotropic emission
                Surface(&POS, &DIR) ;                     
                IndexG(&POS, &level, &ind, DENS, OFF) ;
#endif
#if (PS_METHOD==1)
                // pointsource outside the volume, send all packages to 2pi solid angle
                // => only sqrt(2) decrease in the noise, many packages will miss the cloud altogether
                if (POS.z>NZ) {  // source is in the direction of +Z
                   if (DIR.z>0.0f) DIR.z = -DIR.z ;
                } else {
                   if (POS.z<0.0f) {
                      if (DIR.z<0.0f) DIR.z = -DIR.z ;            
                   } else {
                      if (POS.x>NX) {
                         if (DIR.x>0.0f) DIR.x = -DIR.x ;            
                      } else {
                         if (POS.x<0.0f) {
                            if (DIR.x<0.0f) DIR.x = -DIR.x ;            
                         } else {
                            if (POS.y>NY) {
                               if (DIR.y>0.0f) DIR.y = -DIR.y ;            
                            } else {
                               if (POS.y<0.0f) {
                                  if (DIR.y<0.0f) DIR.y = -DIR.y ;            
                               }
                            }
                         }
                      }
                   }
                }
                Surface(&POS, &DIR) ;
                IndexG(&POS, &level, &ind, DENS, OFF) ;            
                PHOTONS *= 0.5f ;  // photons emitted to half-space
#endif
#if (PS_METHOD==2)
                // METHOD B
                //   This is ok method.... but gives equal number of rays for surface elements far from the 
                //   source as for the surface elements close to the source... a concern if the point source
                //   is very close to the surface (and should mostly illuminate a small number of surface elements)
                // - select random X, Y, or Z direction
                // - select random cell on the surface ....
                // - calculate the direction towards that cell element
                // In normal case a surface element would be hit a number of times that is
                //        (GL^2*cos(theta)) * PACKETS / (4*pi*r^2)
                //       theta = angle for LOS vs. surface normal
                //       r     = distance from point source to the surface
                // New scheme:  a surface element is hit this many times:
                //        (1/3)*(1/(XY))*PACKETS
                //       where "XY" stands for the surface area of that side of the cloud
                // ==> 
                //    PHOTONS  *=   3*cos(theta)*XY / (4*pi*r^2),  [r] = GL
                // What if only one or two sides are visible from the point source
                //    above (1/3) -> (1/visible_sides)
                //     => host prepares for each point source "visible_nside", "visible_side[6]"
                // This version generates only packages that will hit the cloud.
                // TODO: select surface element more systematically to reduce Monte Carlo noise
                //       and to choose sides according to their projected area rather than 
                //       selecting any visible side
                // select random side, XY, YZ, or XZ
# if 1          // equal probability for all visible sides (irrespective of their projected visible area)
                ind      =  floor(Rand(&rng)*XPS_NSIDE[level0]*0.999999f) ;  // side is the one in XPS_SIDE[ind], ind=0,1,2
                PHOTONS /=  XPS_AREA[3*level0+ind] ;  // while ind = 0,1,2
#  if 0
                if (isfinite(PHOTONS)) { 
                   ;
                } else {
                   // sometimes ind=1 although only one side was illuminated ... 
                   printf("ind %d, area %.3e, PHOTONS %.3e\n", ind, XPS_AREA[3*level0+ind], PHOTONS) ; 
                   PHOTONS = 0.0f ;
                }
#  endif     
                ind      =  XPS_SIDE[3*level0+ind] ;  // side 0-5,  +X,-X,+Y,-Y,+Z,-Z ~ 0-5
# else
                // weight selection according to the projected area == XPS_AREA[]
                // *** XPS_AREA are not yet actually calculated, see host ***
                v1       =  Rand(&rng) ;
                for(ind=0; v1>0.0f; ind++) v1 -= XPS_AREA[3*level0+ind] ; // [0,3[
                PHOTONS *=  ??? ;
                ind      =  XPS_SIDE[3*level0+ind] ;                      // [0,6[, +X,-X,+Y,-Y,+Z,-Z ~ 0-5
# endif
                // select random point on the surface
                float a=Rand(&rng), b = Rand(&rng) ;
                if (ind==0) { // +X side
                   POS.x = NX-PEPS;   POS.y = a*NY ; POS.z = b*NZ ;  b = NY*NZ ;
                }
                if (ind==1) { // -X side
                   POS.x = PEPS;      POS.y = a*NY ; POS.z = b*NZ ;  b = NY*NZ ;
                }
                if (ind==2) { // +Y side
                   POS.y = NY-PEPS;   POS.x = a*NX ; POS.z = b*NZ ;  b = NX*NZ ;
                }
                if (ind==3) { // -Y side
                   POS.y = PEPS;      POS.x = a*NX ; POS.z = b*NZ ;  b = NX*NZ ;
                }
                if (ind==4) { // +Z side
                   POS.z = NZ-PEPS;   POS.x = a*NX ; POS.y = b*NY ;  b = NX*NY ;
                }
                if (ind==5) { // -Z side
                   POS.z = PEPS;      POS.x = a*NX ; POS.y = b*NY ;  b = NX*NY ;
                }
                // distance and cos(theta)
                v1  = distance(POS, PSPOS[level0]) ;   // distance from source to surface [GL]
                DIR = POS-PSPOS[level0] ;
                DIR = normalize(DIR) ;
                // cos(theta)
                v2  = (ind<2) ? (fabs(DIR.x)) :   ((ind<4) ? (fabs(DIR.y)) : (fabs(DIR.z))) ;
                // re-weight photon numbers
                PHOTONS  *=   v2*b / (4.0f*PI*v1*v1) ; // division by XPS_AREA done above when ind was still [0,3[
                IndexG(&POS, &level, &ind, DENS, OFF) ;
#endif
#if (PS_METHOD==3)
                // We use a Healpix map to select and to weight the direction of
                // generated photon packages
                //    XPS_NSIDE = NSIDE parameter of the Healpix map and the number of prob. bins
                //    XPS_AREA  = relative weight given for each pixel
                //    XPS_SIDE  = corresponding vector of pixel indices for equidistant grid of P
                //  ==>
                //    u ~(0, XPS_NSIDE[1]) gives a bin in XPS_SIDE == ipix == healpix pixel index
                //    generate direction towards that pixel
                //    PACKETS /= XPS_AREA[ipix], probability of selecting a pixel is proportional to XPS_AREA
                ind =  floor(Rnd(&rng)*XPS_NSIDE[1]) ;  // selected probability bin
                ind =  XPS_SIDE[ind] ;                  // corresponding Healpix pixel index
                PHOTONS /= XPS_AREA[ind] ;              // relative weighting of package
                // Generate direction towards that pixel
                Pixel2AnglesRing(ind, float *phi, float *theta) ;
                DIR.x = sin(theta)*cos(phi) ;
                DIR.y = sin(theta)*sin(phi) ;
                DIR.z = cos(theta) ;
                // a random direction near the pixel centre direction
                phi   = TWOPI*Rnd(&rng) ;
                // NSIDE = 128 => pixel size 0.5 degrees ... 0.5 degrees in cos_theta ~ 3.8e-5
                // NSIDE =  64 => pixel size 1   degree
                cos_theta = 1.0f-Rand(&rng)*4.0e-5f ;
                Deflect(&DIR, cos_theta, phi) ;
                // this was NOT EXACT but probably good enough (unless NSIDE is small and 
                // emission is heavily weighted towards a low number of Healpix pixels)
                // => one could loop until the deflected direction is checked to be inside the
                //    intended Healpix pixel... 
                // }  // loop until DIR is inside pixel ind
                Surface(&POS, &DIR) ;
                IndexG(&POS, &level, &ind, DENS, OFF) ;            
#endif
#if (PS_METHOD==4)
                // We require that the point source is in Z above the cloud
                // and the source (X,Y) coordinates correspond to the centre of the cloud XY plane.
                // In this case we can calculate the cone in which the cloud is seen
                // and packages are only sent within this cone.
                // The solid angle is defined by the distance to the surface and by radius
                // sqrt[   (0.5*NX)^2  + (0.5*NY)^2  ]
                v1         =  PSPOS[level0].z - NZ ;          // height above the cloud surface
                cos_theta  =  v1 / sqrt(v1*v1 + 0.25f*NX*NX + 0.25f*NY*NY) ;
                PHOTONS   *=  0.5f*(1.0f-cos_theta) ;         // because packages go to a smaller solid angle
                // Generate random direction within the cone (some rays will be rejected)
                cos_theta  =  1.0f-Rand(&rng)*(1.0f-cos_theta) ;   // now a uniform cos_theta within the cone
                v1         =  TWOPI*Rand(&rng) ;
                DIR.x      =  sqrt(1.0f-cos_theta*cos_theta)*cos(v1) ;
                DIR.y      =  sqrt(1.0f-cos_theta*cos_theta)*sin(v1) ;
                DIR.z      =  -cos_theta ;
                Surface(&POS, &DIR) ;            
                IndexG(&POS, &level, &ind, DENS, OFF) ;            
#endif
#if (PS_METHOD==5)
                // Generilisation of PS_METHOD==4
                // Host has calculated the main illuminated side, values 0-5 of 
                //   XPS_SIDE[3*level0] correspond to illuminated sides +X, -X, +Y, -Y, +Z, -Z.
                //   XPS_AREA[3*level0] is the cosine of the cone opening angle.
                //   Note that level0 is here still the index of the point source!!
                cos_theta  =  XPS_AREA[3*level0] ;
                PHOTONS   *=  0.5f*(1.0f-cos_theta) ;
                cos_theta  =  1.0f-Rand(&rng)*(1.0f-cos_theta) ;   // uniform cos_theta within the cone
                v1         =  TWOPI*Rand(&rng) ;                   // random phi angle
                oind       =  XPS_SIDE[3*level0+0];                // the illuminated side --- ONLY ONE !!
                if (oind<2) {        // +X or -X illuminated
                   DIR.y      =   sqrt(1.0f-cos_theta*cos_theta)*cos(v1) ;
                   DIR.z      =   sqrt(1.0f-cos_theta*cos_theta)*sin(v1) ;
                   if (oind==0)   DIR.x  =  -cos_theta ; // +X illuminated, package in -X direction
                   else           DIR.x  =  +cos_theta ;
                } else {
                   if (oind<4) {     // +X or -X illuminated
                      DIR.x      =   sqrt(1.0f-cos_theta*cos_theta)*cos(v1) ;
                      DIR.z      =   sqrt(1.0f-cos_theta*cos_theta)*sin(v1) ;
                      if (oind==2)   DIR.y = -cos_theta ;
                      else           DIR.y = +cos_theta ;
                   } else {          // +X or -X illuminated
                      DIR.x      =   sqrt(1.0f-cos_theta*cos_theta)*cos(v1) ;
                      DIR.y      =   sqrt(1.0f-cos_theta*cos_theta)*sin(v1) ;
                      if (oind==4)   DIR.z  =  -cos_theta ;
                      else           DIR.z  =  +cos_theta ;
                   }
                }
                Surface(&POS, &DIR) ;            
                IndexG(&POS, &level, &ind, DENS, OFF) ;
#endif                        
                
             }  // external point source ==================================================================================================================================
          } // SOURCE==0
          
          
          
          
          // =============== BGPAC =================== BATCH packages per element
          if (SOURCE==1) {   // background emission -- find out the surface element
             // generate new position and direction -- using SIDE, X0, Y0, Z0, DX, DY, DZ
             POS.x      =  clamp(X0 + DX*Rand(&rng), PEPS, NX-PEPS) ;
             POS.y      =  clamp(Y0 + DY*Rand(&rng), PEPS, NY-PEPS) ;
             POS.z      =  clamp(Z0 + DZ*Rand(&rng), PEPS, NZ-PEPS) ;
             cos_theta  =  sqrt(Rand(&rng)) ;
             phi        =  TWOPI*Rand(&rng) ;
             sin_theta  =  sqrt(1.0f-cos_theta*cos_theta) ;
             v1 =sin_theta*cos(phi) ;   v2 = sin_theta*sin(phi) ;
             switch (SIDE)  {  
              case 0:  // lower X
                DIR.x =  cos_theta ; DIR.y = v1 ;  DIR.z = v2 ;    break ;
              case 1:  // upper X
                DIR.x = -cos_theta ; DIR.y = v1 ;  DIR.z = v2 ;    break ;
              case 2:  // lower Y
                DIR.y =  cos_theta ; DIR.x = v1 ;  DIR.z = v2 ;    break ;
              case 3:  // upper Y
                DIR.y = -cos_theta ; DIR.x = v1 ;  DIR.z = v2 ;    break ;
              case 4:  // lower Z
                DIR.z =  cos_theta ; DIR.x = v1 ;  DIR.y = v2 ;    break ;
              case 5:  // upper Z
                DIR.z = -cos_theta ; DIR.x = v1 ;  DIR.y = v2 ;    break ;
             }
             PHOTONS    =  BG ;        // every packet has equal weight
             IndexG(&POS, &level, &ind, DENS, OFF) ;
          } // end of -- SOURCE==1 -- BGPAC
          
          
          
#if (BLOOP_FUDGE>0)
          if (ind<0) {
             BLOOP += 1 ;   if (BLOOP>=BATCH) break ;
             continue ;  // if PS is outside the volume, some rays may miss the cloud -> next BLOOP
          }
#else
          if (ind<0) continue ;  // continue with next BLOOP loop 
#endif
          
#if 0
          if (fabs(DIR.x)<DEPS) DIR.x = DEPS ;
          if (fabs(DIR.y)<DEPS) DIR.y = DEPS ;
          if (fabs(DIR.z)<DEPS) DIR.z = DEPS ;
          DIR         =  normalize(DIR) ;
#endif
          scatterings =  0 ;
          tau         =  0.0f ; 
          
          
#if 0
          if (DENS[OFF[level0]+ind]<=0.0f)
            printf("A: %10.4f %10.4f %10.4f  %6.3f %6.3f %6.3f   level %2d  ind %6d  PHOT %.3e FLIP %.0f\n", POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, level, ind, PHOTONS, phi) ;
          // return ;
#endif 
          
          
          
          
          
          
#if (FFS>0)
          // Forced first scattering
          POS0       =  POS ;     // do not touch the parameters of the real ray {POS, ind, level}
          ind0       =  ind ;
          level0     =  level ;
          tau        =  0.0f ;
          while(ind0>=0) {
             oind    =  OFF[level0]+ind0 ;
             ds      =  GetStep(&POS0, &DIR, &level0, &ind0, DENS, OFF, PAR) ; // POS, level, ind updated !!
# if (WITH_ABU>0)
             tau    +=  ds*DENS[oind]*GOPT(2*oind+1) ;
# else
             tau    +=  ds*DENS[oind]*(*SCA) ;
# endif
          }
          if (tau<1.0e-22f) ind = -1 ;      // nothing along the LOS
          W          =  1.0f-exp(-tau) ;
          free_path  = -log(1.0-W*Rand(&rng)) ;
          // free_path  = clamp(free_path, 1.0e-7f, 10.0f) ;          
          PHOTONS   *=  W ;
#else
          // no forced first scattering
          free_path  = -log(Rand(&rng)) ;
#endif
          
          
          
          
          // Ray defined by POS, DIR, ind, level
          scatterings =  0 ;
          
          
          
          while(ind>=0) {  // loop until this ray is really finished
             
             
#if 1
             free_path = clamp(free_path, 1.0e-6f, 99.0f) ; // 2019-07-16
#endif
             
             tau = 0.0f ;         
             
             while(ind>=0) {    // loop until scattering
                ind0      =  ind ;               // indices at the beginning of the step
                level0    =  level ;
                POS0      =  POS ;               // because GetStep does coordinate transformations...
                oind      =  OFF[level0]+ind0 ;  // global index at the beginning of the step
                
                ds        =  GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // POS, level, ind updated !!
#if (WITH_ABU>0)
                dtau      =  ds*DENS[oind]*GOPT(2*oind+1) ;
#else
                dtau      =  ds*DENS[oind]*(*SCA) ;
#endif
                if (free_path<(tau+dtau)) {  // tau = optical depth since last scattering
                   ind = ind0 ;              // !!!
                   break ;                   // scatter before ds
                }
                tau      +=  dtau ;          // total optical depth for scattering
             }
             if (ind<0) break ;  // ray is out of the cloud
             
             
             // we do scatter - after partial step from old position POS0, staying in the same cell
             scatterings++ ;
             dtau               =  free_path-tau ;
#if (WITH_ABU>0)
             dx                 =  dtau/(GOPT(2*oind+1)*DENS[oind]) ;
#else
             dx                 =  dtau/((*SCA)*DENS[oind]) ;    // actual step forward in GLOBAL coordinates
#endif
             dx                 =  ldexp(dx, level) ;            // in LOCAL coordinates
             dx                 =  max(0.0001f, dx-2.0f*PEPS) ;  // must remain in cell (level0, ind0)
             POS0               =  POS0 + dx*DIR ;               // POS0 becomes the position of the scattering
             // remove absorptions since last scattering
             // ??? THIS IGNORES POSSIBLE RELATIVE ABUNDANCE VARIATIONS IN CELLS BETWEEN SCATTERING EVENTS !!!
#if (WITH_ABU>0)  // OPT contains per-cell values
             PHOTONS           *=  exp(-free_path*GOPT(2*oind)/GOPT(2*oind+1)) ;
#else
             PHOTONS           *=  exp(-free_path*(*ABS)/(*SCA)) ;
#endif
             
             
#if 1 // PEEL-OFF
             // Do peel-off  -- starting at the location of the scattering = POS0
             for(int idir=0; idir<NDIR; idir++) {
                POS       =  POS0 ;
                ind       =  ind0 ;    // coordinates of the scattering location
                level     =  level0 ;
                tau       =  0.0f ;            
                ODIR      =  ODIRS[idir] ;         
                while(ind>=0) {       // loop to surface, towards observer
                   oind   =  OFF[level]+ind ;   // need to store index of the cell at the step beginning
                   ds     =  GetStep(&POS, &ODIR, &level, &ind, DENS, OFF, PAR) ;
# if (WITH_ABU>0)
                   tau   +=  ds*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
# else
                   tau   +=  ds*DENS[oind]*((*ABS)+(*SCA)) ;
# endif
                }
                // time to add something to the scattering array
                cos_theta =  clamp(DIR.x*ODIR.x+DIR.y*ODIR.y+DIR.z*ODIR.z, -0.999f, +0.999f) ;
# if (WITH_MSF>0)
                // Using DSC of a randomly selected dust component
                oind      =  OFF[level0]+ind0 ;                   // back to the scattering cell
                dx        =  GOPT(2*oind+1) ;                     // sum(ABU*SCA) for the current cell
                ds        =  0.99999f*Rand(&rng) ;
                for(idust=0; idust<NDUST; idust++) {             // ind0 ~ dust index
                   ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ; // RE-USING ind0 and free_path
                   if (ds<=0.0) break ;
                }                
# endif
                delta     =  PHOTONS* exp(-tau) *  DSC[idust*BINS+clamp((int)(BINS*(1.0f+cos_theta)*0.5f), 0, BINS-1)] ;
                // coordinates  = projections on (ORA, ODE) vectors
                POS      -=  CENTRE ;
                i         =  (0.5f*NPIX.x-0.00005f) - dot(POS, ORA[idir]) * MAP_DX ;
                j         =  (0.5f*NPIX.y-0.00005f) + dot(POS, ODE[idir]) * MAP_DX ;
                if ((i>=0)&&(j>=0)&&(i<NPIX.x)&&(j<NPIX.y)) {   // ind  =  i+j*NPIX.x ;
                   i     +=  idir*NPIX.x*NPIX.y   +    j*NPIX.x ;
                   atomicAdd_g_f(&(OUT[i]), delta) ;
                }
             } // for idir in NDIR
#endif // PEEL-OFF
             
             
             // return to original indices, at the location of the scattering
             POS            =  POS0 ;
             ind            =  ind0 ;           // cell has not changed !?
             level          =  level0 ;         // (ind, level) at the beginning of the step = at the end
             oind           =  OFF[level]+ind ;
             
#if (WITH_MSF==0)
             Scatter(&DIR, CSC, &rng) ;             // new direction, basic case with a single scattering function
#else
             dx        =  GOPT(2*oind+1) ;           // sum(ABU*SCA) for the current cell
             ds        =  0.99999f*Rand(&rng) ;
             for(idust=0; idust<NDUST; idust++) {             
                ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
                if (ds<=0.0) break ;
             }
             Scatter(&DIR, &CSC[idust*BINS], &rng) ; // use the scattering function of the ind0:th dust species
#endif
             
             free_path      = -log(Rand(&rng)) ;
             
#if (RUSSIAN_ROULETTE==0)
             if (scatterings>=MAX_SCATTERINGS) {
                ind = -1 ;  continue  ; // go and get next ray
             }
#else
             //  Russian roulette to remove packages
             if (scatterings>MAX_SCATTERINGS) {
                if (Rand(&rng)<0.25f) {   // one in four terminated
                   ind = -1 ;  continue ;
                } else {                  // ther rest *= 4/3
                   PHOTONS     *= 1.3333333f ;
                   // scatterings  = 0 ;
                }
             }
#endif         
          } // while (ind>0) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
          
          
#if (BLOOP_FUDGE>0)
          // for(III=...) DOES NOT WORK !!!!!!
          // if (III==(BATCH-1)) break ; // WTF !!!!!!!!!!! 
          BLOOP += 1 ;
          if (BLOOP>=BATCH) break ;
#endif
          
          
       } // for BLOOP  or while(1) !!!!!!!!
   
}




















__kernel void SimRAM_P(const      int      PACKETS,   //  0 - number of packets
                       const      int      BATCH,     //  1 - for SOURCE==2, packages per cell
                       const      float    SEED,      //  2 
                       __global   float   *ABS,       //  3 
                       __global   float   *SCA,       //  4 
                       __global   float3  *PSPOS,     //  5 - positions of point sources
                       __global   float   *PS,        //  6 - point source luminosities
                       constant   int     *LCELLS,    //  7 - number of cells on each level
                       constant   int     *OFF,       //  8 - index of first cell on each level
                       __global   int     *PAR,       //  9 - index of parent cell        [CELLS]
                       __global   float   *DENS,      // 10 - density and hierarchy       [CELLS]
                       constant   float   *DSC,       // 11 - BINS entries
                       constant   float   *CSC,       // 12 - cumulative scattering function
                       const      int      NDIR,      // 13 - number of scattering maps = directions
                       __global   float3  *ODIRS,     // 14 - observer directions
                       const      int2     NPIX,      // 15 - map has NPIX.x * NPIX.y pixels
                       const      float    MAP_DX,    // 16 - map pixel size in root grid units
                       const      float3   CENTRE,    // 17 - map centre 
                       __global   float3  *ORA,       // 18 - unit vector for RA coordinate axis
                       __global   float3  *ODE,       // 19 - unit vector for DE coordinate axis
                       __global   float   *OUT,       // 20 -- scattering arrays, NDIR*NPIX*NPIX
                       __global   float   *ABU,       // 21
                       __global   OTYPE   *OPT,       // 22
                       __global   int     *XPS_NSIDE, // 23 -- sides visible        [  NO_PS]
                       __global   int     *XPS_SIDE,  // 24 -- visible side indices [3*NO_PS]
                       __global   float   *XPS_AREA   // 25 -- solid angles         [3*NO_PS]
                      ) 
{
   // Extracted from above SimRAM_PB, for point sources only
   const int id     = get_global_id(0) ;
   int    oind=0, level=0, scatterings, i, j ;
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, sin_theta ;
   float3 DIR=0.0f, POS, POS0, ODIR ; 
   float  PHOTONS, v1, v2, W ;
   
   mwc64x_state_t rng;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod((double)(SEED*(id+PI)),1.0)*4294967296L), samplesPerStream);
   
   int ind =-1 ;
   int ind0=-1, level0, idust=0 ;
   
   
#if (BLOOP_FUDGE>0)
   int BLOOP = 0 ;
   while(1) 
#else
     for(int BLOOP=0; BLOOP<BATCH; BLOOP++) 
#endif
       {
          
          // 2017-12-25 -- BATCH is multiple of the number of point sources,
          //               each work item loops over the full set of sources
          //    GLOBAL*BATCH is the total number of photon packages over all sources
          //    each work item simulates BATCH packages for each source
          //    II%NO_PS used to loop over all point sources
          phi       =  TWOPI*Rand(&rng) ;
          cos_theta =  0.999995f-1.999995f*Rand(&rng) ;
          sin_theta =  sqrt(1.0f-cos_theta*cos_theta) ;
          DIR.x     =  sin_theta*cos(phi) ;   
          DIR.y     =  sin_theta*sin(phi) ;
          DIR.z     =  cos_theta ;
          level0    =  BLOOP % NO_PS ;             // index of the current point sources
          PHOTONS   =  1.0f*PS[level0] ;           // WPS ~ 1/PSPAC, PSPAC ~ GLOBAL*PATCH
          POS.x     =  PSPOS[level0].x ;   
          POS.y     =  PSPOS[level0].y ;   
          POS.z     =  PSPOS[level0].z ;
          IndexG(&POS, &level, &ind, DENS, OFF) ;  // THIS CHANGES POS !!
         
          if ((ind<0)||(ind>=CELLS)) {   // if pointsource is outside the model volume ================================================================================
#if (PS_METHOD==0)
             // Basic (inefficient) isotropic emission
             POS.x  =  PSPOS[level0].x ;    POS.y  =  PSPOS[level0].y ;      POS.z  =  PSPOS[level0].z ;
             Surface(&POS, &DIR) ;                     
             IndexG(&POS, &level, &ind, DENS, OFF) ;
#endif
#if (PS_METHOD==1)
             // pointsource outside the volume, send all packages to 2pi solid angle
             // => only sqrt(2) decrease in the noise, many packages will miss the cloud altogether
             POS.x  =  PSPOS[level0].x ;    POS.y  =  PSPOS[level0].y ;      POS.z  =  PSPOS[level0].z ;
             if (POS.z>NZ) {  // source is in the direction of +Z
                if (DIR.z>0.0f) DIR.z = -DIR.z ;
             } else {
                if (POS.z<0.0f) {
                   if (DIR.z<0.0f) DIR.z = -DIR.z ;            
                } else {
                   if (POS.x>NX) {
                      if (DIR.x>0.0f) DIR.x = -DIR.x ;            
                   } else {
                      if (POS.x<0.0f) {
                         if (DIR.x<0.0f) DIR.x = -DIR.x ;            
                      } else {
                         if (POS.y>NY) {
                            if (DIR.y>0.0f) DIR.y = -DIR.y ;            
                         } else {
                            if (POS.y<0.0f) {
                               if (DIR.y<0.0f) DIR.y = -DIR.y ;            
                            }
                         }
                      }
                   }
                }
             }
             Surface(&POS, &DIR) ;
             IndexG(&POS, &level, &ind, DENS, OFF) ;            
             PHOTONS *= 0.5f ;  // photons emitted to half-space
#endif
#if (PS_METHOD==2)
             //   This is ok -- but gives equal number of rays for surface elements far from the 
             //   source as for the surface elements close to the source... inefficient if the point source
             //   is very close to the surface (and should mostly illuminate a small number of surface elements)
             // - select random X, Y, or Z direction
             // - select random cell on the surface ....
             // - calculate the direction towards that cell element
             // In normal case a surface element would be hit a number of times that is
             //        (GL^2*cos(theta)) * PACKETS / (4*pi*r^2)
             //       theta = angle for LOS vs. surface normal
             //       r     = distance from point source to the surface
             // Here a surface element is hit this many times:
             //        (1/3)*(1/(XY))*PACKETS
             //       where "XY" stands for the surface area of that side of the cloud
             // ==> 
             //    PHOTONS  *=   3*cos(theta)*XY / (4*pi*r^2),  [r] = GL
             // What if only one or two sides are visible from the point source
             //    above (1/3) -> (1/visible_sides)
             //     => host prepares for each point source "visible_nside", "visible_side[6]"
             // This version generates only packages that will hit the cloud.
             // TODO: select surface element more systematically to reduce Monte Carlo noise
             //       and to choose sides according to their projected area rather than 
             //       selecting any visible side
             // select random side, XY, YZ, or XZ

             // equal probability for all visible sides (irrespective of their projected visible area) !!!
             ind      =  floor(Rand(&rng)*XPS_NSIDE[level0]*0.999999f) ;  // side is the one in XPS_SIDE[ind], ind=0,1,2
             PHOTONS /=  XPS_AREA[3*level0+ind] ;  // while ind = 0,1,2
             ind      =  XPS_SIDE[3*level0+ind] ;  // side 0-5  ==  +X,-X,+Y,-Y,+Z,-Z
             // select random point on the surface
             float a=Rand(&rng), b = Rand(&rng) ;
             if (ind==0) { // +X side
                POS.x = NX-PEPS;   POS.y = a*NY ; POS.z = b*NZ ;  b = NY*NZ ;
             }
             if (ind==1) { // -X side
                POS.x = PEPS;      POS.y = a*NY ; POS.z = b*NZ ;  b = NY*NZ ;
             }
             if (ind==2) { // +Y side
                POS.y = NY-PEPS;   POS.x = a*NX ; POS.z = b*NZ ;  b = NX*NZ ;
             }
             if (ind==3) { // -Y side
                POS.y = PEPS;      POS.x = a*NX ; POS.z = b*NZ ;  b = NX*NZ ;
             }
             if (ind==4) { // +Z side
                POS.z = NZ-PEPS;   POS.x = a*NX ; POS.y = b*NY ;  b = NX*NY ;
             }
             if (ind==5) { // -Z side
                POS.z = PEPS;      POS.x = a*NX ; POS.y = b*NY ;  b = NX*NY ;
             }
             // distance and cos(theta)
             v1   =  distance(POS, PSPOS[level0]) ;   // distance from source to surface [GL]
             DIR  =  POS-PSPOS[level0] ;
             DIR  =  normalize(DIR) ;
             // cos(theta)
             v2   =  (ind<2) ? (fabs(DIR.x)) :   ( (ind<4) ? (fabs(DIR.y)) : (fabs(DIR.z)) ) ;
             // re-weight photon numbers
             PHOTONS  *=   v2*b / (4.0f*PI*v1*v1) ; // division by XPS_AREA done above when ind was still [0,3[
             IndexG(&POS, &level, &ind, DENS, OFF) ;
#endif
#if (PS_METHOD==3)
             // We use a Healpix map to select and to weight the direction of
             // generated photon packages
             //    XPS_NSIDE = NSIDE parameter of the Healpix map and the number of prob. bins
             //    XPS_AREA  = relative weight given for each pixel
             //    XPS_SIDE  = corresponding vector of pixel indices for equidistant grid of P
             //  ==>
             //    u ~(0, XPS_NSIDE[1]) gives a bin in XPS_SIDE == ipix == healpix pixel index
             //    generate direction towards that pixel
             //    PACKETS /= XPS_AREA[ipix], probability of selecting a pixel is proportional to XPS_AREA
             POS.x  =  PSPOS[level0].x ;    POS.y  =  PSPOS[level0].y ;      POS.z  =  PSPOS[level0].z ;
             ind =  floor(Rnd(&rng)*XPS_NSIDE[1]) ;  // selected probability bin
             ind =  XPS_SIDE[ind] ;                  // corresponding Healpix pixel index
             PHOTONS /= XPS_AREA[ind] ;              // relative weighting of package
             // Generate direction towards that pixel
             Pixel2AnglesRing(ind, float *phi, float *theta) ;
             DIR.x = sin(theta)*cos(phi) ;
             DIR.y = sin(theta)*sin(phi) ;
             DIR.z = cos(theta) ;
             // a random direction near the pixel centre direction
             phi   = TWOPI*Rnd(&rng) ;
             // NSIDE = 128 => pixel size 0.5 degrees ... 0.5 degrees in cos_theta ~ 3.8e-5
             // NSIDE =  64 => pixel size 1   degree
             cos_theta = 1.0f-Rand(&rng)*4.0e-5f ;
             Deflect(&DIR, cos_theta, phi) ;
             // this was NOT EXACT but probably good enough (unless NSIDE is small and 
             // emission is heavily weighted towards a low number of Healpix pixels)
             // => one could loop until the deflected direction is checked to be inside the
             //    intended Healpix pixel... 
             // }  // loop until DIR is inside pixel ind
             Surface(&POS, &DIR) ;
             IndexG(&POS, &level, &ind, DENS, OFF) ;            
#endif
#if (PS_METHOD==4)
             // We require that the point source is in Z above the cloud
             // and the source (X,Y) coordinates correspond to the centre of the cloud XY plane.
             // In this case we can calculate the cone in which the cloud is seen
             // and packages are only sent within this cone.
             // The solid angle is defined by the distance to the surface and by radius
             // sqrt[   (0.5*NX)^2  + (0.5*NY)^2  ]
             POS.x      =  PSPOS[level0].x ;    POS.y  =  PSPOS[level0].y ;      POS.z  =  PSPOS[level0].z ;
             v1         =  PSPOS[level0].z - NZ ;          // height above the cloud surface
             cos_theta  =  v1 / sqrt(v1*v1 + 0.25f*NX*NX + 0.25f*NY*NY) ;
             PHOTONS   *=  0.5f*(1.0f-cos_theta) ;         // because packages go to a smaller solid angle
             // Generate random direction within the cone (some rays will be rejected)
             cos_theta  =  1.0f-Rand(&rng)*(1.0f-cos_theta) ;   // now a uniform cos_theta within the cone
             v1         =  TWOPI*Rand(&rng) ;
             DIR.x      =  sqrt(1.0f-cos_theta*cos_theta)*cos(v1) ;
             DIR.y      =  sqrt(1.0f-cos_theta*cos_theta)*sin(v1) ;
             DIR.z      =  -cos_theta ;
             Surface(&POS, &DIR) ;            
             IndexG(&POS, &level, &ind, DENS, OFF) ;            
#endif
#if (PS_METHOD==5)
             // Generilisation of PS_METHOD==4
             // Host has calculated the main illuminated side, values 0-5 of 
             //   XPS_SIDE[3*level0] correspond to illuminated sides +X, -X, +Y, -Y, +Z, -Z.
             //   XPS_AREA[3*level0] is the cosine of the cone opening angle.
             //   Note that level0 is here still the index of the point source!!
             POS.x      =  PSPOS[level0].x ;    POS.y  =  PSPOS[level0].y ;      POS.z  =  PSPOS[level0].z ;
             cos_theta  =  XPS_AREA[3*level0] ;
             PHOTONS   *=  0.5f*(1.0f-cos_theta) ;
             cos_theta  =  1.0f-Rand(&rng)*(1.0f-cos_theta) ;   // uniform cos_theta within the cone
             v1         =  TWOPI*Rand(&rng) ;                   // random phi angle
             oind       =  XPS_SIDE[3*level0+0];                // the illuminated side --- ONLY ONE !!
             if (oind<2) {        // +X or -X illuminated
                DIR.y      =   sqrt(1.0f-cos_theta*cos_theta)*cos(v1) ;
                DIR.z      =   sqrt(1.0f-cos_theta*cos_theta)*sin(v1) ;
                if (oind==0)   DIR.x  =  -cos_theta ; // +X illuminated, package in -X direction
                else           DIR.x  =  +cos_theta ;
             } else {
                if (oind<4) {     // +X or -X illuminated
                   DIR.x      =   sqrt(1.0f-cos_theta*cos_theta)*cos(v1) ;
                   DIR.z      =   sqrt(1.0f-cos_theta*cos_theta)*sin(v1) ;
                   if (oind==2)   DIR.y = -cos_theta ;
                   else           DIR.y = +cos_theta ;
                } else {          // +X or -X illuminated
                   DIR.x      =   sqrt(1.0f-cos_theta*cos_theta)*cos(v1) ;
                   DIR.y      =   sqrt(1.0f-cos_theta*cos_theta)*sin(v1) ;
                   if (oind==4)   DIR.z  =  -cos_theta ;
                   else           DIR.z  =  +cos_theta ;
                }
             }
             Surface(&POS, &DIR) ;            
             IndexG(&POS, &level, &ind, DENS, OFF) ;
#endif                        
             
          }  // external point source ==================================================================================================================================
          

          
#if (BLOOP_FUDGE>0)
          if (ind<0) {
             BLOOP += 1 ;   if (BLOOP>=BATCH) break ;
             continue ;  // if PS is outside the volume, some rays may miss the cloud -> next BLOOP
          }
#else
          if (ind<0) continue ;  // continue with next BLOOP loop 

#endif
          
#if 0
          if (fabs(DIR.x)<DEPS) DIR.x = DEPS ;
          if (fabs(DIR.y)<DEPS) DIR.y = DEPS ;
          if (fabs(DIR.z)<DEPS) DIR.z = DEPS ;
          DIR         =  normalize(DIR) ;
#endif
          
          
          
          
          
#if (FFS>0)
          // Forced first scattering
          POS0       =  POS ;     // do not touch the parameters of the real ray {POS, ind, level}
          ind0       =  ind ;
          level0     =  level ;
          tau        =  0.0f ;
          while(ind0>=0) {
             oind    =  OFF[level0]+ind0 ;
             ds      =  GetStep(&POS0, &DIR, &level0, &ind0, DENS, OFF, PAR) ; // POS, level, ind updated !!
# if (WITH_ABU>0)
             tau    +=  ds*DENS[oind]*GOPT(2*oind+1) ;
# else
             tau    +=  ds*DENS[oind]*(*SCA) ;
# endif
          }
          if (tau<1.0e-22f) ind = -1 ;      // nothing along the LOS
          W          =  1.0f-exp(-tau) ;
          free_path  = -log(1.0-W*Rand(&rng)) ;
          // free_path  = clamp(free_path, 1.0e-7f, 10.0f) ;          
          PHOTONS   *=  W ;
#else
          // no forced first scattering
          free_path  = -log(Rand(&rng)) ;
#endif
          
          
                    
          // Ray defined by POS, DIR, ind, level
          scatterings =  0 ;
                    
          
          while(ind>=0) {  // loop until this ray is really finished
                          
#if 0
             free_path = clamp(free_path, 1.0e-6f, 99.0f) ; // 2019-07-16
#endif
             
             tau = 0.0f ;         
             
             while(ind>=0) {    // loop until scattering
                ind0      =  ind ;               // indices at the beginning of the step
                level0    =  level ;
                POS0      =  POS ;               // because GetStep does coordinate transformations...
                oind      =  OFF[level0]+ind0 ;  // global index at the beginning of the step
                ds        =  GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // POS, level, ind updated !!
#if (WITH_ABU>0)
                dtau      =  ds*DENS[oind]*GOPT(2*oind+1) ;
#else
                dtau      =  ds*DENS[oind]*(*SCA) ;
#endif
                if (free_path<(tau+dtau)) {  // tau = optical depth since last scattering
                   ind = ind0 ;              // !!!
                   break ;                   // scatter before ds
                }
                tau      +=  dtau ;          // total optical depth for scattering
             }
             if (ind<0) break ;  // ray is out of the cloud
             
             // we do scatter - after partial step from old position POS0, staying in the same cell
             scatterings++ ;
             dtau               =  free_path-tau ;
#if (WITH_ABU>0)
             dx                 =  dtau/(GOPT(2*oind+1)*DENS[oind]) ;
#else
             dx                 =  dtau/((*SCA)*DENS[oind]) ;    // actual step forward in GLOBAL coordinates
#endif
             dx                 =  ldexp(dx, level) ;            // in LOCAL coordinates
             dx                 =  max(0.0001f, dx-2.0f*PEPS) ;  // must remain in cell (level0, ind0)
             POS0               =  POS0 + dx*DIR ;               // POS0 becomes the position of the scattering
             // remove absorptions since last scattering
             // ??? THIS IGNORES POSSIBLE RELATIVE ABUNDANCE VARIATIONS IN CELLS BETWEEN SCATTERING EVENTS !!!
#if (WITH_ABU>0)  // OPT contains per-cell values
             PHOTONS           *=  exp(-free_path*GOPT(2*oind)/GOPT(2*oind+1)) ;
#else
             PHOTONS           *=  exp(-free_path*(*ABS)/(*SCA)) ;
#endif
             
             
#if 1 // PEEL-OFF
             // Do peel-off  -- starting at the location of the scattering = POS0
             for(int idir=0; idir<NDIR; idir++) {
                POS       =  POS0 ;
                ind       =  ind0 ;    // coordinates of the scattering location
                level     =  level0 ;
                tau       =  0.0f ;            
                ODIR      =  ODIRS[idir] ;         
                while(ind>=0) {       // loop to surface, towards observer
                   oind   =  OFF[level]+ind ;   // need to store index of the cell at the step beginning
                   ds     =  GetStep(&POS, &ODIR, &level, &ind, DENS, OFF, PAR) ;
# if (WITH_ABU>0)
                   tau   +=  ds*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
# else
                   tau   +=  ds*DENS[oind]*((*ABS)+(*SCA)) ;
# endif
                }
                // time to add something to the scattering array
                cos_theta =  clamp(DIR.x*ODIR.x+DIR.y*ODIR.y+DIR.z*ODIR.z, -0.999f, +0.999f) ;
# if (WITH_MSF>0)
                // Using DSC of a randomly selected dust component
                oind      =  OFF[level0]+ind0 ;                   // back to the scattering cell
                dx        =  GOPT(2*oind+1) ;                     // sum(ABU*SCA) for the current cell
                ds        =  0.99999f*Rand(&rng) ;
                for(idust=0; idust<NDUST; idust++) {             // ind0 ~ dust index
                   ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ; // RE-USING ind0 and free_path
                   if (ds<=0.0) break ;
                }                
# endif
                delta     =  PHOTONS* exp(-tau) *  DSC[idust*BINS+clamp((int)(BINS*(1.0f+cos_theta)*0.5f), 0, BINS-1)] ;
                // coordinates  = projections on (ORA, ODE) vectors
                POS      -=  CENTRE ;
                i         =  (0.5f*NPIX.x-0.00005f) - dot(POS, ORA[idir]) * MAP_DX ;
                j         =  (0.5f*NPIX.y-0.00005f) + dot(POS, ODE[idir]) * MAP_DX ;
                if ((i>=0)&&(j>=0)&&(i<NPIX.x)&&(j<NPIX.y)) {   // ind  =  i+j*NPIX.x ;
                   i     +=  idir*NPIX.x*NPIX.y   +    j*NPIX.x ;
                   atomicAdd_g_f(&(OUT[i]), delta) ;
                }
             } // for idir in NDIR
#endif // PEEL-OFF
             
             
             // return to original indices, at the location of the scattering
             POS            =  POS0 ;
             ind            =  ind0 ;           // cell has not changed !?
             level          =  level0 ;         // (ind, level) at the beginning of the step = at the end
             oind           =  OFF[level]+ind ;
             
#if (WITH_MSF==0)
             Scatter(&DIR, CSC, &rng) ;             // new direction, basic case with a single scattering function
#else
             dx        =  GOPT(2*oind+1) ;           // sum(ABU*SCA) for the current cell
             ds        =  0.9999f*Rand(&rng) ;
             for(idust=0; idust<NDUST; idust++) {             
                ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
                if (ds<=0.0) break ;
             }
             Scatter(&DIR, &CSC[idust*BINS], &rng) ; // use the scattering function of the ind0:th dust species
#endif
             
             free_path      = -log(Rand(&rng)) ;
             
#if (RUSSIAN_ROULETTE==0)
             if (scatterings>=MAX_SCATTERINGS) {
                ind = -1 ;  continue  ; // go and get next ray
             }
#else
             //  Russian roulette to remove packages
             if (scatterings>MAX_SCATTERINGS) {
                if (Rand(&rng)<0.25f) {   // one in four terminated
                   ind = -1 ;  continue ;
                } else {                  // ther rest *= 4/3
                   PHOTONS     *= 1.3333333f ;
                   // scatterings  = 0 ;
                }
             }
#endif         
          } // while (ind>0) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
          
          
#if (BLOOP_FUDGE>0)
          // for(III=...) DOES NOT WORK !!!!!!
          // if (III==(BATCH-1)) break ; // WTF !!!!!!!!!!! 
          BLOOP += 1 ;
          if (BLOOP>=BATCH) break ;
#endif
          
          
       } // for BLOOP  or while(1) !!!!!!!!
   
}


























#if (USE_EMWEIGHT<2)


__kernel void SimRAM_CL(const      int      SOURCE,  //  0 - PSPAC/BGPAC/CLPAC = 0/1/2 --- here dummy argument
                        const      int      PACKETS, //  1 - number of packets
                        const      int      BATCH,   //  2 - for SOURCE==2, packages per cell
                        const      float    SEED,    //  3 
                        __global   float*   ABS,     //  4 
                        __global   float*   SCA,     //  5 
                        constant   int     *LCELLS,  //  6 - number of cells on each level
                        constant   int     *OFF,     //  7 - index of first cell on each level
                        __global   int     *PAR,     //  8 - index of parent cell        [CELLS]
                        __global   float   *DENS,    //  9 - density and hierarchy       [CELLS]
                        __global   float   *EMIT,    // 10 - emission from cells         [CELLS]
                        constant   float   *DSC,     // 11 - BINS entries
                        constant   float   *CSC,     // 12 - cumulative scattering function
                        // Scattering
                        const      int      NDIR,    // 13 - number of scattering maps = directions
                        __global   float3  *ODIRS,   // 14 - directions towards observers
                        const      int2     NPIX,    // 15 - map has NPIX.x * NPIX.y pixels
                        const      float    MAP_DX,  // 16 - map step in root grid units
                        const      float3   CENTRE,  // 17 - map centre 
                        __global   float3  *ORA,     // 18 - unit vector for RA coordinate axis
                        __global   float3  *ODE,     // 19 - unit vector for DE coordinate axis
                        __global   float   *OUT,     // 20 - scattering arrays, NDIR * NPIX*NPIX
                        __global   OTYPE   *OPT,     // 21 - OPT[CELLS+CELLS], ABS and SCA iff -DWITH_ABU
                        __global   float   *ABU,     // 22 - ABU[CELLS*NDUST], iff -DWITH_MSF
                        __global   float   *EMWEI,
                        __global   float   *EMINDEX
                       ) 
{
   const int id     = get_global_id(0) ;
   if (id>=CELLS) return ;
   const int GLOBAL = get_global_size(0) ;
   
   int    oind=0, level=0, scatterings, batch, i, j ;
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, sin_theta ;
   float3 DIR=0.0f, POS, POS0, ODIR ; 
   float  PHOTONS, X0, Y0, Z0, PWEI=1.0f, W ;
   
   mwc64x_state_t rng;
   // Assume that 2^38 = 2.7e11 random numbers per worker is sufficient
   // For each NITER, host will also give a new seed [0,1] that is multiplied by 2^32 = 4.3e9
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   // MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod((double)(SEED*(id+PI)),1.0)*4294967296L), samplesPerStream);
   
   
   int ICELL =  id-GLOBAL ;  // current cell for the current work item
   int IRAY  =  0 ;          // counter for packages done for current cell
   int ind   = -1 ;
   int ind0, level0, idust ;
   batch = -1 ;
   
   
   while(1) {
      
      // generate new ray >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      
      // One work item simulates batch packages from a cell, then moves to
      // nest cell with index += GLOBAL, BATCH = CLPAC/CELLS
      
      // ICELL = id+()*GLOBAL < CELLS
      // IRAY  = 0...         < batch = [BATCH | EMWEI]
      if (IRAY>=batch) {    // move to next cell (on first call 0>=-1)
         IRAY  = 0 ;
         PWEI  = 1.0f ;         // additional weighting to photon numbers
         while(1) {
            ICELL += GLOBAL ;   // initially ICELL==id-GLOBAL, move to next cell for this work item
            if (ICELL>=CELLS) {
               return ; // no more cells for this work item
            }
# if (USE_EMWEIGHT>0)
            PWEI  = EMWEI[ICELL] ;
            if ((PWEI<1e-10f)||(DENS[ICELL]<=0.0f)) {
               continue ;  // no rays from this cell
            }
            batch = (int)floor(PWEI) ;
            //  EMWEI>=1 => batch=EMWEI, weight is 1.0/batch, 
            //  EMWEI<1  => batch=1,     weight 1/EMWEI
            if (batch<1) {  // Russian roulette if EMWEI rounds to zero
               batch = 1 ;   PWEI = 1.0/(PWEI+1.0e-30f) ;
            } else {
               PWEI = 1.0/(batch+1.0e-9f) ;
            }            
# else
            batch = BATCH ;
            PWEI  = 1.0f/(batch+1.0e-9f) ;
# endif
            break ;
         } // while(1)
      } // IRAY>=batch 
      ind     =  ICELL ;  // we start in the cell ICELL
      IRAY   +=  1 ;      // this many rays created for the current cell
      
      
      for(level=0; level<LEVELS-1; level++) { // find the level of this cell
         ind -= LCELLS[level] ;
         if (ind<0) {              // ok, current cell was on level "level"
            ind += LCELLS[level] ; // index within "level"
            break ;
         }
      }
      // level=0 => coordinates are global coordinates, otherwise of the current octet
      if (level==0) {
         X0  = (ind % NX) ;
         Y0  = ((ind/NX) % NY) ;
         Z0  = (ind/(NX*NY)) ;
      } else {
         int sid = ind % 8 ;               // which cell in the octet
         X0  = (sid%2) ;                   // lower border in local coordinates
         Y0  = ((sid%4)>1) ? 1.0f : 0.0f ; // fixed (?) 2016-06-29
         Z0  = (sid/4) ;
      }
      PHOTONS    =  EMIT[OFF[level]+ind] * PWEI ;
      POS.x      =  X0 + Rand(&rng) ;  POS.y = Y0 + Rand(&rng) ;  POS.z = Z0 + Rand(&rng) ;
      // Direction
      phi        =  TWOPI*Rand(&rng) ;
      cos_theta  =  0.999997f-1.999995f*Rand(&rng) ;
      sin_theta  =  sqrt(1.0f-cos_theta*cos_theta) ;
      DIR.x      =  sin_theta*cos(phi) ;   
      DIR.y      =  sin_theta*sin(phi) ;
      DIR.z      =  cos_theta ;
      
      if (fabs(DIR.x)<DEPS) DIR.x = DEPS ;
      if (fabs(DIR.y)<DEPS) DIR.y = DEPS ;
      if (fabs(DIR.z)<DEPS) DIR.z = DEPS ; 
      DIR         =  normalize(DIR) ;
      
      
      
      
      
# if (FFS>0)  // Forced first scattering ============================================================
      POS0       =  POS ;     // do not touch the original parameters [POS, ind, level]
      ind0       =  ind ;
      level0     =  level ;
      tau        =  0.0f ;
      while(ind0>=0) {
         oind    =  OFF[level0]+ind0 ;
         ds      =  GetStep(&POS0, &DIR, &level0, &ind0, DENS, OFF, PAR) ; // POS, level, ind updated !!
#  if (WITH_ABU>0)
         tau    +=  ds*DENS[oind]*GOPT(2*oind+1) ;
#  else
         tau    +=  ds*DENS[oind]*(*SCA) ;
#  endif
      }
      if (tau<1.0e-22f) {   // we are inside for II ~ BATCH loop
         ind = -1 ;         // nothing along the LOS  -> select next ray
         continue ;
      }
      W          =  1.0f-exp(-tau) ;
      free_path  = -log(1.0-W*Rand(&rng)) ;
      PHOTONS   *=  W ;
# else // No forced first scattering ============================================================
      free_path  = -log(Rand(&rng)) ;
# endif // ========================================================================================
      scatterings =  0 ;
      // Ray defined by [POS, DIR, ind, level]
      
      
      
      
      while(ind>=0) {  // loop until this ray is really finished
         
         tau = 0.0f ;         
         while(ind>=0) {    // loop until scattering
            ind0      =  ind ;               // indices at the beginning of the step
            level0    =  level ;
            POS0      =  POS ;               // because GetStep does coordinate transformations...
            oind      =  OFF[level0]+ind0 ;  // global index at the beginning of the step
            ds        =  GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // POS, level, ind updated !!
            // step ds from cell [ind0,level0,oind] to cell [ind, level]
# if (WITH_ABU>0)
            dtau      =  ds*DENS[oind]*GOPT(2*oind+1) ;
# else
            dtau      =  ds*DENS[oind]*(*SCA) ;
# endif       
            if (free_path<(tau+dtau)) {  // tau = optical depth since last scattering
               ind = ind0 ;              // this ray is not yet done - only scattered
               break ;                   // scatter before doing the step ds
            }
            tau      +=  dtau ;
         }
         if (ind<0) break ;  // ray is out of the cloud -- break the loop: while(ind>=0)
         
         
         // we scatter - after a partial step from old position POS
         //  == we are still inside the scattering cell {POS0, level0, ind0, oind}
         scatterings++ ;
         // dtau               =  free_path-tau ;
# if (WITH_ABU>0)
         ds                 =  (free_path-tau)/(GOPT(2*oind+1)*DENS[oind]) ; 
# else
         ds                 =  (free_path-tau)/((*SCA)*DENS[oind]) ;    // actual step forward in GLOBAL coordinates
# endif
         ds                 =  ldexp(ds, level) ;         // in LOCAL coordinates
         POS0               =  POS0 + ds*DIR ;            // POS0 becomes the position of the scattering
         // remove absorptions since last scattering
# if (WITH_ABU>0)
         PHOTONS           *=  exp(-free_path*GOPT(2*oind)/GOPT(2*oind+1)) ;
# else
         PHOTONS           *=  exp(-free_path*(*ABS)/(*SCA)) ;
# endif         
         
         
         // Do peel-off  -- starting at the location of the scattering
         // == start in the scattering cell == { POS0, level0, ind0 } -- oind will be overwritten
         for(int idir=0; idir<NDIR; idir++) {
            POS       =  POS0 ;                                              
            ind       =  ind0 ;    // coordinates of the scattering location
            level     =  level0 ;
            ODIR      =  ODIRS[idir] ;         
            tau       =  0.0f ;            
            while(ind>=0) {                 // loop to surface, towards observer
               oind   =  OFF[level]+ind ;   // cell index at the start of the step
               ds     =  GetStep(&POS, &ODIR, &level, &ind, DENS, OFF, PAR) ; // POS updated !!
# if (WITH_ABU>0)
               tau   +=  ds*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
# else               
               tau   +=  ds*DENS[oind]*((*ABS)+(*SCA))   ;
# endif
            }
            // time to add something to the scattering array
            cos_theta =  clamp(DIR.x*ODIR.x+DIR.y*ODIR.y+DIR.z*ODIR.z, -0.9999f, +0.9999f) ;
# if (WITH_MSF>0)
            // Using DSC of a randomly selected dust component -> set idust
            oind      =  OFF[level0]+ind0 ;                   // back to the scattering cell
            dx        =  GOPT(2*oind+1) ;                     // sum(ABU*SCA) for the current cell
            ds        =  0.99999f*Rand(&rng) ;
            for(idust=0; idust<NDUST; idust++) {
               ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
               if (ds<=0.0) break ;
            }                        
# endif
# ifdef HG_TEST
            // fraction = probability of scattering per solid angle
            float G = 0.65f, fraction ;
            fraction  =  (1.0f/(4.0f*PI)) * (1.0f-G*G) / pow(1.0f+G*G-2.0f*G*cos_theta, 1.5f) ;
            // delta     =  PHOTONS *  exp(-tau) *  fraction ;
            delta     =  PHOTONS *  fraction * ((tau>TAULIM) ?  (1.0f-exp(-tau)) : (tau*(1.0f-0.5f*tau))) ;
# else
            delta     =  PHOTONS *  exp(-tau) *  DSC[idust*BINS+clamp((int)(BINS*(1.0f+cos_theta)*0.5f), 0, BINS-1)] ;
# endif
            // coordinates  = projections on (ORA, ODE) vectors
            POS      -=   CENTRE ;   // exit position relative the model centre
            i         =   (0.5f*NPIX.x-0.00005f) - dot(POS, ORA[idir]) * MAP_DX ;
            j         =   (0.5f*NPIX.y-0.00005f) + dot(POS, ODE[idir]) * MAP_DX ;
            if ((i>=0)&&(j>=0)&&(i<NPIX.x)&&(j<NPIX.y)) {
               i     +=  idir*NPIX.x*NPIX.y   +    j*NPIX.x ;
               atomicAdd_g_f(&(OUT[i]), delta) ;
            }
         } // for NDIR -- end of peel-off
         
         
         // return to original indices, at the location of the scattering
         // these are still { POS0, level0, ind0 } == position and indices of the scattering cell
         POS            =  POS0 ;
         ind            =  ind0 ;                // cell has not changed !?
         level          =  level0 ;              // (ind, level) at the beginning of the step = at the end
         oind           =  OFF[level0]+ind0 ;    // global index
         
# if (WITH_MSF==0)
         Scatter(&DIR, CSC, &rng) ;              // new direction, basic case with a single scattering function
# else
         dx        =  GOPT(2*oind+1) ;           // sum(ABU*SCA) for the current cell
         ds        =  0.99999f*Rand(&rng) ;
         for(idust=0; idust<NDUST; idust++) {
            ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
            if (ds<=0.0) break ;
         }
         Scatter(&DIR, &CSC[idust*BINS], &rng) ; // use the scattering function of the ind0:th dust species
# endif
         free_path      = -log(Rand(&rng)) ;         
         
# if (RUSSIAN_ROULETTE==0)
         if (scatterings==MAX_SCATTERINGS) {
            ind = -1 ;  continue ;  // break  ; // go and get next ray continue==break
         }
# else
         //  Russian roulette to remove packages 1 == 0.25*0 + 3/4 * 4/3
         if (scatterings==MAX_SCATTERINGS) {
            if (Rand(&rng)<0.25f) {
               ind = -1 ; break ;
            } else {
               PHOTONS     *=  1.33333333f ;
               scatterings  =  0 ;
            }
         }
# endif         
      } // while (ind>0) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      
      
   } // for III
   
   
   
}






#else // USE_EMWEIGHT==2





__kernel void SimRAM_CL(const      int      SOURCE,  //  0 - PSPAC/BGPAC/CLPAC = 0/1/2
                        const      int      PACKETS, //  1 - number of packets
                        const      int      BATCH,   //  2 - for SOURCE==2, packages per cell
                        const      float    SEED,    //  3 
                        __global   float*   ABS,     //  4 
                        __global   float*   SCA,     //  5 
                        constant   int     *LCELLS,  //  6 - number of cells on each level
                        constant   int     *OFF,     //  7 - index of first cell on each level
                        __global   int     *PAR,     //  8 - index of parent cell        [CELLS]
                        __global   float   *DENS,    //  9 - density and hierarchy       [CELLS]
                        __global   float   *EMIT,    // 10 - emission from cells         [CELLS]
                        constant   float   *DSC,     // 11 - BINS entries
                        constant   float   *CSC,     // 12 - cumulative scattering function
                        // Scattering
                        const      int      NDIR,    // 13 - number of scattering maps = directions
                        __global   float3  *ODIRS,   // 14 - directions towards observers
                        const      int2     NPIX,    // 15 - map has NPIX.x * NPIX.y pixels
                        const      float    MAP_DX,  // 16 - map step in root grid units
                        const      float3   CENTRE,  // 17 - map centre 
                        __global   float3  *ORA,     // 18 - unit vector for RA coordinate axis
                        __global   float3  *ODE,     // 19 - unit vector for DE coordinate axis
                        __global   float   *OUT,     // 20 - scattering arrays, NDIR * NPIX*NPIX
                        __global   OTYPE   *OPT,     // 21 - OPT[CELLS+CELLS], ABS and SCA iff -DWITH_ABU
                        __global   float   *ABU,     // 22 - ABU[CELLS*NDUST], iff -DWITH_MSF
                        __global   float   *EMWEI,
                        __global   int     *EMINDEX
                       ) 
{
   const int id     = get_global_id(0) ;
   if (id>=CELLS) return ;
   const int GLOBAL = get_global_size(0) ;
   
   int    oind=0, level=0, scatterings ;
   int    level0, ind0, i, j, idust ;
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, sin_theta ;
   float3 DIR=0.0f, POS, POS0, ODIR ; 
   float  PHOTONS, X0, Y0, Z0, PWEI=1.0f, W ;
   
   mwc64x_state_t rng;
   // Assume that 2^38 = 2.7e11 random numbers per worker is sufficient
   // For each NITER, host will also give a new seed [0,1] that is multiplied by 2^32 = 4.3e9
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   // MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod((double)(SEED*(id+PI)),1.0)*4294967296L), samplesPerStream);
   
   
   int IND = id-GLOBAL, ind ;
   int ICELL = -1 ;
   
   
   while(1) {
      
      // find next cell that is emitting something
      while (1) {
         IND += GLOBAL ;  // try next cell
         if (IND>=CELLS) return ;      // no more valid cells
         if (EMINDEX[IND]<0) return ;  // no more valid cells
         if (EMINDEX[IND]>0) {
            ICELL   = EMINDEX[IND] ;   // found an emitting cell
            PWEI    = EMWEI[ICELL] ;   // ... and its package weight
            break ;
         }
      }    
      
      // !!! iter over EMWEI2_STEP photon packages !!!
      for(int iter=0; iter<100; iter++) { // three packages from this cell
         ind = ICELL ;
         // generate new ray >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
         for(level=0; level<LEVELS-1; level++) { // find the level of this cell
            ind -= LCELLS[level] ;
            if (ind<0) {              // ok, current cell was on level "level"
               ind += LCELLS[level] ; // index within "level"
               break ;
            }
         }
         // level=0 => coordinates are global coordinates, otherwise of the current octet
         if (level==0) {
            X0  = (ind % NX) ;
            Y0  = ((ind/NX) % NY) ;
            Z0  = (ind/(NX*NY)) ;
         } else {
            int sid = ind % 8 ;               // which cell in the octet
            X0  = (sid%2) ;                   // lower border in local coordinates
            Y0  = ((sid%4)>1) ? 1.0f : 0.0f ; // fixed (?) 2016-06-29
            Z0  = (sid/4) ;
         }
         PHOTONS    =  EMIT[OFF[level]+ind] * PWEI ;
         POS.x      =  X0 + Rand(&rng) ;  POS.y = Y0 + Rand(&rng) ;  POS.z = Z0 + Rand(&rng) ;
         // Direction
         phi        =  TWOPI*Rand(&rng) ;
         cos_theta  =  0.999997f-1.999995f*Rand(&rng) ;
         sin_theta  =  sqrt(1.0f-cos_theta*cos_theta) ;
         DIR.x      =  sin_theta*cos(phi) ;
         DIR.y      =  sin_theta*sin(phi) ;
         DIR.z      =  cos_theta ;
         if (fabs(DIR.x)<DEPS) DIR.x = DEPS ;
         if (fabs(DIR.y)<DEPS) DIR.y = DEPS ;
         if (fabs(DIR.z)<DEPS) DIR.z = DEPS ;
         DIR         =  normalize(DIR) ;
         scatterings =  0 ;
         tau         =  0.0f ;

# if (FFS>0)  // Forced first scattering ============================================================
         POS0       =  POS ;     // do not touch the original parameters [POS, ind, level]
         ind0       =  ind ;
         level0     =  level ;
         tau        =  0.0f ;
         while(ind0>=0) {
            oind    =  OFF[level0]+ind0 ;
            ds      =  GetStep(&POS0, &DIR, &level0, &ind0, DENS, OFF, PAR) ; // POS, level, ind updated !!
#  if (WITH_ABU>0)
            tau    +=  ds*DENS[oind]*GOPT(2*oind+1) ;
#  else
            tau    +=  ds*DENS[oind]*(*SCA) ;
#  endif
         }
         if (tau<1.0e-22f) {   // we are inside for II ~ BATCH loop @
            ind = -1 ;         // nothing along the LOS
            continue ;
         }
         W          =  1.0f-exp(-tau) ;
         free_path  = -log(1.0-W*Rand(&rng)) ;
         PHOTONS   *=  W ;
# else // No forced first scattering ============================================================
         free_path  = -log(Rand(&rng)) ;
# endif // ========================================================================================
         scatterings =  0 ;
         // Ray defined by [POS, DIR, ind, level]
      
         
         
         
         while(ind>=0) {  // loop until this ray is really finished
            
            tau = 0.0f ;         
            while(ind>=0) {    // loop until scattering
               ind0      =  ind ;               // indices at the beginning of the step
               level0    =  level ;
               POS0      =  POS ;               // because GetStep does coordinate transformations...
               oind      =  OFF[level0]+ind0 ;  // global index at the beginning of the step
               ds        =  GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // POS, level, ind updated !!
               // step ds from cell [ind0,level0,oind] to cell [ind, level]
# if (WITH_ABU>0)
               dtau      =  ds*DENS[oind]*GOPT(2*oind+1) ;
# else
               dtau      =  ds*DENS[oind]*(*SCA) ;
# endif       
               if (free_path<(tau+dtau)) {  // tau = optical depth since last scattering
                  ind = ind0 ;              // this ray is not yet done - only scattered
                  break ;                   // scatter before doing the step ds
               }
               tau      +=  dtau ;
            }
            if (ind<0) break ;  // ray is out of the cloud -- break the loop: while(ind>=0)
            
            
            // we scatter - after a partial step from old position POS
            //  == we are still inside the scattering cell {POS0, level0, ind0, oind}
            scatterings++ ;
            // dtau               =  free_path-tau ;
# if (WITH_ABU>0)
            ds                 =  (free_path-tau)/(GOPT(2*oind+1)*DENS[oind]) ; 
# else
            ds                 =  (free_path-tau)/((*SCA)*DENS[oind]) ;    // actual step forward in GLOBAL coordinates
# endif
            ds                 =  ldexp(ds, level) ;         // in LOCAL coordinates
            POS0               =  POS0 + ds*DIR ;            // POS0 becomes the position of the scattering
            // remove absorptions since last scattering
# if (WITH_ABU>0)
            PHOTONS           *=  exp(-free_path*GOPT(2*oind)/GOPT(2*oind+1)) ;
# else
            PHOTONS           *=  exp(-free_path*(*ABS)/(*SCA)) ;
# endif         
            
            
            // Do peel-off  -- starting at the location of the scattering
            // == start in the scattering cell == { POS0, level0, ind0 } -- oind will be overwritten
            for(int idir=0; idir<NDIR; idir++) {
               POS       =  POS0 ;                                              
               ind       =  ind0 ;    // coordinates of the scattering location
               level     =  level0 ;
               ODIR      =  ODIRS[idir] ;         
               tau       =  0.0f ;            
               while(ind>=0) {                 // loop to surface, towards observer
                  oind   =  OFF[level]+ind ;   // cell index at the start of the step
                  ds     =  GetStep(&POS, &ODIR, &level, &ind, DENS, OFF, PAR) ; // POS updated !!
# if (WITH_ABU>0)
                  tau   +=  ds*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
# else               
                  tau   +=  ds*DENS[oind]*((*ABS)+(*SCA))   ;
# endif
               }
               // time to add something to the scattering array
               cos_theta =  clamp(DIR.x*ODIR.x+DIR.y*ODIR.y+DIR.z*ODIR.z, -0.9999f, +0.9999f) ;
# if (WITH_MSF>0)
               // Using DSC of a randomly selected dust component -> set idust
               oind      =  OFF[level0]+ind0 ;                   // back to the scattering cell
               dx        =  GOPT(2*oind+1) ;                     // sum(ABU*SCA) for the current cell
               ds        =  0.99999f*Rand(&rng) ;
               for(idust=0; idust<NDUST; idust++) {
                  ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
                  if (ds<=0.0) break ;
               }                        
# endif
# ifdef HG_TEST
               // fraction = probability of scattering per solid angle
               float G = 0.65f, fraction ;
               fraction  =  (1.0f/(4.0f*PI)) * (1.0f-G*G) / pow(1.0f+G*G-2.0f*G*cos_theta, 1.5f) ;
               // delta     =  PHOTONS *  exp(-tau) *  fraction ;
               delta     =  PHOTONS *  fraction * ((tau>TAULIM) ?  (1.0f-exp(-tau)) : (tau*(1.0f-0.5f*tau))) ;
# else
               delta     =  PHOTONS *  exp(-tau) *  DSC[idust*BINS+clamp((int)(BINS*(1.0f+cos_theta)*0.5f), 0, BINS-1)] ;
# endif
               // coordinates  = projections on (ORA, ODE) vectors
               POS      -=   CENTRE ;   // exit position relative the model centre
               i         =   (0.5f*NPIX.x-0.00005f) - dot(POS, ORA[idir]) * MAP_DX ;
               j         =   (0.5f*NPIX.y-0.00005f) + dot(POS, ODE[idir]) * MAP_DX ;
               if ((i>=0)&&(j>=0)&&(i<NPIX.x)&&(j<NPIX.y)) {
                  i     +=  idir*NPIX.x*NPIX.y   +    j*NPIX.x ;
                  atomicAdd_g_f(&(OUT[i]), delta) ;
               }
            } // for NDIR -- end of peel-off
            
            
            // IndexG(&POS0, &level0, &ind0, DENS, OFF) ; // -- no effect on worms
         
            
            // return to original indices, at the location of the scattering
            // these are still { POS0, level0, ind0 } == position and indices of the scattering cell
            POS            =  POS0 ;
            ind            =  ind0 ;                // cell has not changed !?
            level          =  level0 ;              // (ind, level) at the beginning of the step = at the end
            oind           =  OFF[level0]+ind0 ;    // global index
            
# if (WITH_MSF==0)
            Scatter(&DIR, CSC, &rng) ;              // new direction, basic case with a single scattering function
# else
            dx        =  GOPT(2*oind+1) ;           // sum(ABU*SCA) for the current cell
            ds        =  0.99999f*Rand(&rng) ;
            for(idust=0; idust<NDUST; idust++) {
               ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
               if (ds<=0.0) break ;
            }
            Scatter(&DIR, &CSC[idust*BINS], &rng) ; // use the scattering function of the ind0:th dust species
# endif
            free_path      = -log(Rand(&rng)) ;         
            
# if (RUSSIAN_ROULETTE==0)
            if (scatterings==MAX_SCATTERINGS) {
               ind = -1 ;  break  ; // go and get next ray
            }
# else
            //  Russian roulette to remove packages
            if (scatterings==MAX_SCATTERINGS) {
               if (Rand(&rng)<0.25f) {
                  ind = -1 ; break ;
               } else {
                  PHOTONS     *=  1.33333333f ;
                  scatterings  =  0 ;
               }
            }
# endif
            
         } // while (ind>0) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         
         
      } // for iter
   } // while(1)
   
   
}





#endif

