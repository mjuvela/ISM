#include "kernel_pySOC_aux.c"



__kernel void zero_out(const int       NDIR,
                       const int2      NPIX,
                       __global float  *OUT)   {
   // Clear the OUT array (used to gather outcoming photons) -- space for one frequency
   const int id  = get_global_id(0) ;
   const int gs  = get_global_size(0) ;
   for(int i=id; i<NDIR*NPIX.x*NPIX.y; i+=gs)  OUT[i] = 0.0f ;
}




// KERNEL FOR CONSTANT SOURCES

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
                        __global   float   *OPT,       // 24
                        __global   float   *XPS_NSIDE, // 25
                        __global   float   *XPS_SIDE,  // 26
                        __global   float   *XPS_AREA   // 27
                       ) 
{
   // This routine is simulating the radiation field in order to determine the
   // absorbed energy in each cell.
   const int id     = get_global_id(0) ;
   const int GLOBAL = get_global_size(0) ;
   
   int    oind=0, level=0, scatterings, steps, SIDE, i, j ;
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, sin_theta ;
   float3 DIR=0.0f, POS, POS0, ODIR ; 
   float  PHOTONS, X0, Y0, Z0, DX, DY, DZ, v1, v2, W ;
   
#if (USE_HD>0)
   ulong rng = 123*id + 151313*SEED + id*SEED  ;
   Rand(&rng) ; Rand(&rng) ;  Rand(&rng) ;
#else
   mwc64x_state_t rng;
   // Assume that 2^38 = 2.7e11 random numbers per worker is sufficient
   // For each NITER, host will also give a new seed [0,1] that is multiplied by 2^32 = 4.3e9
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
#endif
   
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
   
   
   
   
   for(int III=0; III<BATCH; III++) { // for III
      
      
      // =============== PSPAC ===================  BATCH packages per work item
      // 2017-12-25 -- BATCH is multiple of the number of point sources,
      //               each work item loops over the full set of sources
      if (SOURCE==0) {
         phi       =  TWOPI*Rand(&rng) ;
         cos_theta =  0.999997f-1.999995f*Rand(&rng) ;
         sin_theta =  sqrt(1.0f-cos_theta*cos_theta) ;
         DIR.x     =  sin_theta*cos(phi) ;   
         DIR.y     =  sin_theta*sin(phi) ;
         DIR.z     =  cos_theta ;
         ind       =  III % NO_PS ;              // ind here the index of the point sources
         POS.x     =  PSPOS[ind].x ;   POS.y = PSPOS[ind].y ;   POS.z = PSPOS[ind].z ;
         PHOTONS   =  PS[ind] ;
         IndexG(&POS, &level, &ind, DENS, OFF) ; // ind again cell index
#if 0
         if ((ind<0)||(ind>=CELLS)) {  // ok, source outside the cloud, **must be** higher in Z
            if (DIR.z>0.0f) DIR.z = -DIR.z ;            
            ds     =  (NZ-PSPOS.z) / DIR.z  ; // try to step to the boundary z=NZ
            POS   +=  ds*DIR ;
            POS.z  =  NZ-PEPS ;
            PHOTONS *= 0.5f ;
            IndexG(&POS, &level, &ind, DENS, OFF) ;
         }
#endif
      }
      
      
      
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
      
      
      if (fabs(DIR.x)<DEPS) DIR.x = DEPS ;
      if (fabs(DIR.y)<DEPS) DIR.y = DEPS ;
      if (fabs(DIR.z)<DEPS) DIR.z = DEPS ; 
      DIR         =  normalize(DIR) ;
      
      
#if 1
      // Forced first scattering
      POS0       =  POS ;     // do not touch the parameters of the real ray
      ind0       =  ind ;
      level0     =  level ;
      tau        =  0.0f ;
      while(ind0>=0) {
         oind    =  OFF[level0]+ind0 ;
         ds      =  GetStep(&POS0, &DIR, &level0, &ind0, DENS, OFF, PAR) ; // POS, level, ind updated !!
# if (WITH_ABU>0)
         tau    +=  ds*DENS[oind]*OPT[2*oind+1] ;
# else
         tau    +=  ds*DENS[oind]*(*SCA) ;
# endif
      }
      if (tau<1.0e-22f) ind = -1 ;      // nothing along the LOS
      W          =  1.0f-exp(-tau) ;
      free_path  = -log(1.0-W*Rand(&rng)) ;
      PHOTONS   *=  W ;
#else
      // no peeloff ... we have for now removed all weighting schemes from this version
      free_path  = -log(Rand(&rng)) ;
#endif
      
      // Ray defined by POS, DIR, ind, level
      
      
      scatterings =  0 ;
      steps = 0 ;
      
      while(ind>=0) {  // loop until this ray is really finished
         
         tau = 0.0f ;         
         while(ind>=0) {    // loop until scattering
            oind      =  OFF[level]+ind ;  // global index at the beginning of the step
            ind0      =  ind ;             // indices at the beginning of the step
            level0    =  level ;
            POS0      =  POS ;             // because GetStep does coordinate transformations...
            ds        =  GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // POS, level, ind updated !!
#if (WITH_ABU>0)
            dtau      =  ds*DENS[oind]*OPT[2*oind+1] ;
#else
            dtau      =  ds*DENS[oind]*(*SCA) ;
#endif
            if (free_path<(tau+dtau)) {  // tau = optical depth since last scattering
               break ;                   // scatter before ds
            }
            tau      +=  dtau ;
         }
         if (ind<0) break ;  // ray is out of the cloud
         
         // we do scatter - after partial step from old position POS0, staying in the same cell
         scatterings++ ;
         dtau               =  free_path-tau ;
#if (WITH_ABU>0)
         dx                 =  dtau/(OPT[2*oind+1]*DENS[oind]) ;
#else
         dx                 =  dtau/((*SCA)*DENS[oind]) ;    // actual step forward in GLOBAL coordinates
#endif
         dx                 =  ldexp(dx, level) ;         // in LOCAL coordinates
         POS0               =  POS0 + dx*DIR ;            // POS0 becomes the position of the scattering
         // remove absorptions since last scattering
#if (WITH_ABU>0)  // OPT contains per-cell values
         PHOTONS           *=  exp(-free_path*OPT[2*oind]/OPT[2*oind+1]) ;
#else
         PHOTONS           *=  exp(-free_path*(*ABS)/(*SCA)) ;
#endif
         
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
#if (WITH_ABU>0)
               tau   +=  ds*DENS[oind]*(OPT[2*oind]+OPT[2*oind+1]) ;
#else
               tau   +=  ds*DENS[oind]*((*ABS)+(*SCA)) ;
#endif
            }
            // time to add something to the scattering array
            cos_theta =  clamp(DIR.x*ODIR.x+DIR.y*ODIR.y+DIR.z*ODIR.z, -0.999f, +0.999f) ;
#if (WITH_MSF>0)
            // Using DSC of a randomly selected dust component
            dx        =  OPT[2*oind+1] ;                     // sum(ABU*SCA) for the current cell
            ds        =  Rand(&rng) ;
            for(idust=0; idust<NDUST; idust++) {             // ind0 ~ dust index
               ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ; // RE-USING ind0 and free_path
               if (ds<=0.0) break ;
            }   
#endif
            delta     =  PHOTONS* exp(-tau) *  DSC[idust*BINS+clamp((int)(BINS*(1.0f+cos_theta)*0.5f), 0, BINS-1)] ;
            // coordinates  = projections on (ORA, ODE) vectors
            POS      -=  CENTRE ;
            i         =  (0.5f*NPIX.x-0.00005f) + dot(POS, ORA[idir]) * MAP_DX ;
            j         =  (0.5f*NPIX.y-0.00005f) + dot(POS, ODE[idir]) * MAP_DX ;
            if ((i>=0)&&(j>=0)&&(i<NPIX.x)&&(j<NPIX.y)) {   // ind  =  i+j*NPIX.x ;
               i     +=  idir*NPIX.x*NPIX.y   +    j*NPIX.x ;
               atomicAdd_g_f(&(OUT[i]), delta) ;
            }
         } // for idir in NDIR
         
         // return to original indices, at the location of the scattering
         POS            =  POS0 ;
         ind            =  ind0 ;           // cell has not changed !?
         level          =  level0 ;         // (ind, level) at the beginning of the step = at the end
         oind           =  OFF[level]+ind ;
                  
#if (WITH_MSF==0)
         Scatter(&DIR, CSC, &rng) ;             // new direction, basic case with a single scattering function
#else
         dx        =  OPT[2*oind+1] ;           // sum(ABU*SCA) for the current cell
         ds        =  Rand(&rng) ;
         for(idust=0; idust<NDUST; idust++) {             
            ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
            if (ds<=0.0) break ;
         }
         Scatter(&DIR, &CSC[idust*BINS], &rng) ; // use the scattering function of the ind0:th dust species
#endif
         
         free_path      = -log(Rand(&rng)) ;
         
#if 1
         if (scatterings>20) {
            ind = -1 ;  continue  ; // go and get next ray
         }
#else
         //  Russian roulette to remove packages
         if (scatterings>15) {
            if (Rand(&rng)<0.25f) {   // one in four terminated
               ind = -1 ; continue ;
            } else {                  // ther rest *= 4/3
               PHOTONS *= 1.3333333f ;
            }
         }
#endif         
      } // while (ind>0) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      
   } // for III
   
}




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
                        __global   float   *OPT,       // 20
                        __global   float   *BG,        // 21 - Healpix map of background sky
                        __global   float   *HPBGP      // 22 - Cumulative probability for selecting pixels
                       ) 
{
   // Background emission using a Healpix map
   const int id     = get_global_id(0) ;
   const int GLOBAL = get_global_size(0) ;
   int    oind=0, level=0, scatterings, steps, SIDE, i, j ;
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, theta ;
   float3 DIR=0.0f, POS, POS0, ODIR ; 
   float  PHOTONS, x, y, z, v1, v2, W ;
   mwc64x_state_t rng;
   // Assume that 2^38 = 2.7e11 random numbers per worker is sufficient
   // For each NITER, host will also give a new seed [0,1] that is multiplied by 2^32 = 4.3e9
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
   
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
      // if (id==0) printf("WEIGHTED\n") ;
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
      Pixel2AnglesRing(64, ind, &phi, &theta) ;
      
      DIR.x   =  -sin(theta)*cos(phi) ;
      DIR.y   =  -sin(theta)*sin(phi) ;
      DIR.z   =  -cos(theta) ;
      
      if (fabs(DIR.x)<DEPS) DIR.x = DEPS ;
      if (fabs(DIR.y)<DEPS) DIR.y = DEPS ;
      if (fabs(DIR.z)<DEPS) DIR.z = DEPS ; 
      DIR         =  normalize(DIR) ;
      
#if 0
      // *** THIS IGNORE THE FACT THAT CLOUD AREA DEPENDS ON THE DIRECTION ***
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
      ds     = 2.0*M_PI*Rand(&rng) ;  dx = sqrt(Rand(&rng)) ; 
      POS0.x = dx*cos(ds) ;   POS0.y = dx*sin(ds) ;   POS0.z = sqrt(1.0f-dx*dx) ;
      // This would be a random position on the surface if healpix pixel were towards +Z
      //  => need to rotate using (theta,phi) of the direction towards the healpix pixel
      // rotation around +Z is not needed because phi rotation around Z was already arbitrary
      // ==> e.g. POS.x remains unchanged, only rotation theta around +X axis
      POS.y =   cos(theta)*POS0.y + sin(theta)*POS0.z ;
      POS.z =  -sin(theta)*POS0.y + cos(theta)*POS0.z ;
      // Position should now be on the sphere, on the same side of the cloud as the Healpix pixel
      // -> try to step onto the actual cloud surface (part of rays will be rejected = miss the cloud)
      Surface(&POS, &DIR) ;
#endif
      
      
      IndexG(&POS, &level, &ind, DENS, OFF) ;
      
      
      
      
#if 1 // Forced first scattering
      POS0       =  POS ;     // do not touch the parameters of the real ray
      ind0       =  ind ;
      level0     =  level ;
      tau        =  0.0f ;
      while(ind0>=0) {
         oind    =  OFF[level0]+ind0 ;
         ds      =  GetStep(&POS0, &DIR, &level0, &ind0, DENS, OFF, PAR) ; // POS, level, ind updated !!
# if (WITH_ABU>0)
         tau    +=  ds*DENS[ind0]*OPT[2*oind+1] ;
# else
         tau    +=  ds*DENS[ind0]*(*SCA) ;
# endif
      }
      if (tau<1.0e-22f) ind = -1 ;      // nothing along the LOS
      W          =  1.0f-exp(-tau) ;
      free_path  = -log(1.0-W*Rand(&rng)) ;
      PHOTONS   *=  W ;
#else // No forced first scattering
      free_path  = -log(Rand(&rng)) ;
#endif
      // Ray defined by POS, DIR, ind, level
      
      
      scatterings =  0 ;
      steps = 0 ;
      
      while(ind>=0) {  // loop until this ray is really finished
         
         tau = 0.0f ;         
         while(ind>=0) {    // loop until scattering
            oind      =  OFF[level]+ind ;  // global index at the beginning of the step
            ind0      =  ind ;             // indices at the beginning of the step
            level0    =  level ;
            POS0      =  POS ;             // because GetStep does coordinate transformations...
            ds        =  GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // POS, level, ind updated !!
#if (WITH_ABU>0)
            dtau      =  ds*DENS[oind]*OPT[2*oind+1] ;
#else
            dtau      =  ds*DENS[oind]*(*SCA) ;
#endif       
            if (free_path<(tau+dtau)) {  // tau = optical depth since last scattering
               break ;                   // scatter before ds
            }
            tau      +=  dtau ;
         }
         if (ind<0) break ;  // ray is out of the cloud
         
         // we do scatter - after partial step from old position POS0, staying in the same cell
         scatterings++ ;
         dtau               =  free_path-tau ;
#if (WITH_ABU>0)
         dx                 =  dtau/(OPT[2*oind+1]*DENS[oind]) ; 
#else
         dx                 =  dtau/((*SCA)*DENS[oind]) ;    // actual step forward in GLOBAL coordinates
#endif
         dx                 =  ldexp(dx, level) ;         // in LOCAL coordinates
         POS0               =  POS0 + dx*DIR ;            // POS0 becomes the position of the scattering
         // remove absorptions since last scattering
#if (WITH_ABU>0)
         PHOTONS           *=  exp(-free_path*OPT[2*oind]/OPT[2*oind+1]) ;
#else
         PHOTONS           *=  exp(-free_path*(*ABS)/(*SCA)) ;
#endif
         
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
#if (WITH_ABU>0)
               tau   +=  ds*DENS[oind]*(OPT[2*oind]+OPT[2*oind+1]) ;
#else               
               tau   +=  ds*DENS[oind]*((*ABS)+(*SCA)) ;
#endif
            }
            // time to add something to the scattering array
            cos_theta =  clamp(DIR.x*ODIR.x+DIR.y*ODIR.y+DIR.z*ODIR.z, -0.999f, +0.999f) ;
#if (WITH_MSF>0)
            // Using DSC of a randomly selected dust component
            dx        =  OPT[2*oind+1] ;                     // sum(ABU*SCA) for the current cell
            ds        =  Rand(&rng) ;
            for(idust=0; idust<NDUST; idust++) {             // ind0 ~ dust index
               ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ; // RE-USING ind0 and free_path
               if (ds<=0.0) break ;
            }                        
#endif
            delta     =  PHOTONS *  exp(-tau) *  DSC[idust*BINS+clamp((int)(BINS*(1.0f+cos_theta)*0.5f), 0, BINS-1)] ;
            // coordinates  = projections on (ORA, ODE) vectors
            POS      -=  CENTRE ;
            i         =  (0.5f*NPIX.x-0.00005f) + dot(POS, ORA[idir]) * MAP_DX ;
            j         =  (0.5f*NPIX.y-0.00005f) + dot(POS, ODE[idir]) * MAP_DX ;
            if ((i>=0)&&(j>=0)&&(i<NPIX.x)&&(j<NPIX.y)) {
               // ind  =  i+j*NPIX.x ;
               i     +=  idir*NPIX.x*NPIX.y   +    j*NPIX.x ;
               atomicAdd_g_f(&(OUT[i]), delta) ;
            }
         } // for NDIR
         
         // return to original indices, at the location of the scattering
         POS            =  POS0 ;
         ind            =  ind0 ;              // cell has not changed !?
         level          =  level0 ;            // (ind, level) at the beginning of the step = at the end
         oind           =  OFF[level0]+ind0 ;  // global index
                  
#if (WITH_MSF==0)
         Scatter(&DIR, CSC, &rng) ;             // new direction, basic case with a single scattering function
#else
         dx        =  OPT[2*oind+1] ;           // sum(ABU*SCA) for the current cell
         ds        =  Rand(&rng) ;
         for(idust=0; idust<NDUST; idust++) {
            ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
            if (ds<=0.0) break ;
         }
         Scatter(&DIR, &CSC[idust*BINS], &rng) ; // use the scattering function of the ind0:th dust species
#endif
         
         free_path      = -log(Rand(&rng)) ;         
         
#if 1
         if (scatterings>20) {
            ind = -1 ;  continue  ; // go and get next ray
         }
#else
         //  Russian roulette to remove packages
         if (scatterings>15) {
            if (Rand(&rng)<0.25f) {   // one in four terminated
               ind = -1 ; continue ;
            } else {                  // ther rest *= 4/3
               PHOTONS *= 1.3333333f ;
            }
         }
#endif         
      } // while (ind>0) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      
   } // for III
   
}





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
                        __global   float   *OPT,     // 21 - OPT[CELLS+CELLS], ABS and SCA iff -DWITH_ABU
                        __global   float   *ABU      // 22 - ABU[CELLS*NDUST], iff -DWITH_MSF
                       ) 
{
   const int id     = get_global_id(0) ;
   if (id>=CELLS) return ;
   const int GLOBAL = get_global_size(0) ;
   
   int    oind=0, level=0, scatterings, batch, i, j ;
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, sin_theta ;
   float3 DIR=0.0f, POS, POS0, ODIR ; 
   float  PHOTONS, X0, Y0, Z0, PWEI=1.0f, W ;
   
#if (USE_HD>0)
   ulong rng = 123*id + 151313*SEED + id*SEED  ;
   Rand(&rng) ; Rand(&rng) ;  Rand(&rng) ;
#else
   mwc64x_state_t rng;
   // Assume that 2^38 = 2.7e11 random numbers per worker is sufficient
   // For each NITER, host will also give a new seed [0,1] that is multiplied by 2^32 = 4.3e9
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
#endif
   
   int ICELL = id-GLOBAL ;
   int IRAY  =  0 ;
   int ind   = -1 ;
   int ind0, level0, idust ;
   batch = -1 ;
   
   
   
   while(1) {
      
      // generate new ray >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
         
      
      // ICELL = id+()*GLOBAL < CELLS
      // IRAY  = 0...         < batch = [BATCH | EMWEI]
      if (IRAY>=batch) {    // move to next cell (on first call 0>=-1)
         IRAY  = 0 ;
         PWEI  = 1.0f ;     // additional weighting to photon numbers
         while(1) {
            ICELL += GLOBAL ;          // initially ICELL==id-GLOBAL
            if (ICELL>=CELLS) {
               return ; // no more cells for this work item
            }
            batch = BATCH ;
            PWEI  = 1.0f/(batch+1.0e-9f) ;
            break ;
         } // while(1)
      } // IRAY>=batch 
      ind     =  ICELL ;
      IRAY   +=  1 ;     // this many rays created for the current cell
      
      
      
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
      
      
#if 1 
      // Forced first scattering
      POS0       =  POS ;     // do not touch the parameters of the real ray
      ind0       =  ind ;
      level0     =  level ;
      tau        =  0.0f ;
      while(ind0>=0) {
         oind    =  OFF[level0]+ind0 ;
         ds      =  GetStep(&POS0, &DIR, &level0, &ind0, DENS, OFF, PAR) ; // POS, level, ind updated !!
# if (WITH_ABU>0)
         tau    +=  ds*DENS[ind0]*OPT[2*oind+1] ;
# else
         tau    +=  ds*DENS[ind0]*(*SCA) ;
# endif
      }
      if (tau<1.0e-22f) ind = -1 ;  // nothing along the LOS
      W          =  1.0f-exp(-tau) ;
      free_path  = -log(1.0-W*Rand(&rng)) ;
      PHOTONS   *=  W ;
#else
      // normal, unweighted case, no FFS
      free_path  = -log(Rand(&rng)) ;
#endif
      
      
      
      
      
      while(ind>=0) {  // loop until this ray is really finished
         
         tau = 0.0f ;
         while(ind>=0) {    // loop until scattering
            oind      =  OFF[level]+ind ;  // global index at the beginning of the step
            ind0      =  ind ;             // indices at the beginning of the step
            level0    =  level ;
            POS0      =  POS ;   // because GetStep does coordinate transformations...
            ds        =  GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // POS, level, ind updated !!
#if (WITH_ABU>0)
            dtau      =  ds*DENS[oind]*OPT[2*oind+1] ;
#else
            dtau      =  ds*DENS[oind]*(*SCA) ;
#endif
            if (free_path<(tau+dtau)) {
               ind    =  ind0 ;       // what if we scatter on step ending outside the cloud?
               level  =  level0 ;
               break ;            // scatter before ds
            }
            tau      +=  dtau ;
         } ;
         if (ind<0) break ;  // ray out of the cloud
         
         
         // package has scattered -- cell == ind0, level0
         scatterings++ ;
         dtau         =  free_path-tau ;            // remining step in optical depth
#if (WITH_ABU>0)
         dx           =  dtau/(OPT[2*oind+1]*DENS[oind]) ;
#else
         dx           =  dtau/((*SCA)*DENS[oind]) ;    // actual step forward in GLOBAL coordinates
#endif
         dx           =  ldexp(dx, level) ;         // in LOCAL coordinates
         POS0         =  POS0 + dx*DIR ;            // POS0 becomes the position of the scattering (old cell)
         // remove absorptions since last scattering         
#if (WITH_ABU>0)
         PHOTONS     *=  exp(-free_path*OPT[2*oind]/OPT[2*oind+1]) ;
#else
         PHOTONS     *=  exp(-free_path*(*ABS)/(*SCA)) ;  
#endif
         
         for(int idir=0; idir<NDIR; idir++) {
            POS       =  POS0 ;
            ind       =  ind0 ;    // coordinates of the scattering location
            level     =  level0 ;
            tau       =  0.0f ;
            ODIR      =  ODIRS[idir] ;   // one direction towards which peeloff calculated
            while(ind>=0) {       // loop to surface, towards observer
               oind   =  OFF[level]+ind ; 
               ds     =  GetStep(&POS, &ODIR, &level, &ind, DENS, OFF, PAR) ;
#if (WITH_ABU>0)
               tau   +=  ds*DENS[oind]*(OPT[2*oind]+OPT[2*oind+1]) ;
#else
               tau   +=  ds*DENS[oind]*((*ABS)+(*SCA)) ;
#endif
            }
            // time to add something to the scattering array
            cos_theta =  clamp(DIR.x*ODIR.x+DIR.y*ODIR.y+DIR.z*ODIR.z, -0.999f, +0.999f) ;
#if (WITH_MSF>0)
            // Using DSC of a randomly selected dust component
            dx        =  OPT[2*oind+1] ;                     // sum(ABU*SCA) for the current cell
            ds        =  Rand(&rng) ;
            for(idust=0; idust<NDUST; idust++) {             // ind0 ~ dust index
               ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ; // RE-USING ind0 and free_path
               if (ds<=0.0) break ;
            }                        
#endif            
            delta     =  PHOTONS * exp(-tau) * DSC[clamp((int)(BINS*(1.0f+cos_theta)*0.5f), 0, BINS-1)] ;
            // coordinates  = projections on (ORA, ODE) vectors
            // LOOKS LIKE POS *IS* IN ROOT GRID COORDINATES WHENEVER RAY EXITS CLOUD ??
            // if (level>0) printf("????? NOT ROOTGRID\n") ;
            POS      -=  CENTRE ;
            i         =  (0.5f*NPIX.x-0.00005f) + dot(POS, ORA[idir]) * MAP_DX ;
            j         =  (0.5f*NPIX.y-0.00005f) + dot(POS, ODE[idir]) * MAP_DX ;
            if ((i>=0)&&(j>=0)&&(i<NPIX.x)&&(j<NPIX.y)) {
               // ind  =   idir*NPIX.x*NPIX.y + j*NPIX.x + i ;  // can reuse ind...
               i  +=   idir*NPIX.x*NPIX.y + j*NPIX.x ;  // can reuse ind...
               atomicAdd_g_f(&(OUT[i]), delta) ;
            }            
         }
         
#if 1 
         if (scatterings>20) {
            ind = -1 ;  continue  ; // go and get next ray
         }
#else
         // Russian roulette to remove packages
         if (scatterings>15) {
            if (Rand(&rng)<0.25f) {   // one in four terminated
               ind = -1 ; continue ;
            } else {                  // ther rest *= 4/3
               PHOTONS *= 1.3333333f ;
            }
         }
#endif
         // normal, unweighted case
         free_path  = -log(Rand(&rng)) ;
         // return to original indices
         POS            =  POS0 ;
         ind            =  ind0 ;           // cell has not changed
         level          =  level0 ;         // (ind, level) at the beginning of the step = at the end
         oind           =  OFF[level]+ind ;

#if (WITH_MSF==0)
         Scatter(&DIR, CSC, &rng) ;             // new direction, basic case with a single scattering function
#else
         dx        =  OPT[2*oind+1] ;           // sum(ABU*SCA) for the current cell
         ds        =  Rand(&rng) ;
         for(idust=0; idust<NDUST; idust++) {
            ds -= ABU[idust+NDUST*oind]*SCA[idust] / dx ;
            if (ds<=0.0) break ;
         }
         Scatter(&DIR, &CSC[idust*BINS], &rng) ; // use the scattering function of the ind0:th dust species
#endif
         
         free_path      = -log(Rand(&rng)) ;         
         tau = 0.0f ;
         
      } // while (ind>0) -- loop until the end of ray before checking if new are required
      
      
   } // while (1) -- loop until all packages simulated
   
}






