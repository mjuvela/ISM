#include "kernel_pySOC_aux.c"

/*
This version works for a single dust population.
It also works for multiple dust species with varying abundances (-DWITH_ABU) 
   in which case the host provides the effective cross sections in OPT ==
   ABS and SCA for every cell, already weighted by the relative abundances
THIS DOES NOT INCLUDE ABUNDANCE VARIATIONS IN THE SCATTERING FUNCTION
ONLY THE FIRST
*/


__kernel void SimRAM_PB(const      int      SOURCE,    //  0 - PSPAC/BGPAC/CLPAC = 0/1/2
                        const      int      PACKETS,   //  1 - number of packets
                        const      int      BATCH,     //  2 - for SOURCE==2, packages per cell
                        const      float    SEED,      //  3
                        __global   float   *ABS,       //  4
                        __global   float   *SCA,       //  5
                        const      float    BG,        //  6 - background intensity
                        __global   float3  *PSPOS,     //  7 - positions of point sources
                        __global   float   *PS,        //  8 - point source luminosities
                        const      float    TW,        //  9 - weight of current frequency in integral
                        constant   int     *LCELLS,    // 10 - number of cells on each level
                        constant   int     *OFF,       // 11 - index of first cell on each level
                        __global   int     *PAR,       // 12 - index of parent cell        [CELLS]
                        __global   float   *DENS,      // 13 - density and hierarchy       [CELLS]
                        __global   float   *EMIT,      // 14 - emission from cells         [CELLS]
                        __global   float   *TABS,      // 15 - buffer for absorptions      [CELLS]
                        constant   float   *DSC,       // 16 - BINS entries [BINS] or [NDUST,BINS]
                        constant   float   *CSC,       // 17 - cumulative scattering function
                        __global   float   *XAB,       // 18 - reabsorbed energy per cell  [CELLS]
                        __global   float   *EMWEI,     // 19 - number of packages per cell
                        __global   float   *INT,       // 20 - net intensity
                        __global   float   *INTX,      // 21 - net intensity vector
                        __global   float   *INTY,      // 22 - net intensity vector
                        __global   float   *INTZ,      // 23 - net intensity vector
                        __global   float   *OPT,       // 24 - WITH_ABU>0 =>  OPT[CELLS+CELLS] ABS and SCA
                        __global   float   *ABU,       // 25 - WITH_MSF>0 =>  ABU[CELLS, NDUST]
                        __global  int      *XPS_NSIDE, // 26 - number of visible sides from each external pointsource
                        __global   int     *XPS_SIDE,  // 27 - [NO_PS*3] = indices of the visible cloud sides
                        __global   float   *XPS_AREA   // 28 - external point sources, visible area [NO_PS*3]
                       ) 
{
   // This routine is simulating the radiation field in order to determine the
   // absorbed energy in each cell.
   const int id     = get_global_id(0) ;
   const int GLOBAL = get_global_size(0) ;
   // SOURCE==0      GLOBAL arbitrary, BATCH>=1
   // SOURCE==1      GLOBAL>=AREA,     BATCH>=1
   // => each work item does precisely BATCH packages
   
   int    oind=0, level=0, scatterings, steps, SIDE ;
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, sin_theta ;
   float3 DIR=0.0f, POS, POS0 ; 
   float  PHOTONS, X0, Y0, Z0, DX, DY, DZ, v1, v2 ;
   
   mwc64x_state_t rng;
   // Assume that 2^38 = 2.7e11 random numbers per worker is sufficient
   // For each NITER, host will also give a new seed [0,1] that is multiplied by 2^32 = 4.3e9
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
   
   
   int ind   = -1 ;
   int ind0=-1, level0 ;
   
   // Each work item simulates BATCH packages
   //  SOURCE==0   --  PSPAC   -- all from the same cell
   //  SOURCE==1   --  BGPAC   -- all from the same surface element
   if ((SOURCE==1)&&(id>=(8*AREA))) return ;  // BGPAC=BATCH*(8*AREA), GLOBAL>=8*AREA (2017-12-24)
   
   
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
   
   
   
   // 2018-12-22
   //   methods A, B, D give identical results --- when the respective requirements are fullfilled
   //     PS_METHOD_A works only if PS sees a single cloud side
   //     PS_METHOD_D works only if PS above +Z surface, PSPOS.x and PSPOS.y are at the centre of XY
   //                 plane and NX==NY
   //  method C has not been implemented on the host side
   
   
   
   for(int III=0; III<BATCH; III++) {
      
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
         level0    =  III % NO_PS ;              // level0 here the index of the point sources !!
         PHOTONS   =  PS[level0] ;
         POS.x     =  PSPOS[level0].x ;   POS.y = PSPOS[level0].y ;   POS.z = PSPOS[level0].z ;
         // Host should already flag the external point sources so that this could be skipped?
         IndexG(&POS, &level, &ind, DENS, OFF) ;
         
         // By default external point sources are not simulated...
         //  not unless one explicitly chooses one of the PS_METHODs below !!
         if ((ind<0)||(ind>=CELLS)) {  // if pointsource is outside the model volume
            
            
#if (PS_METHOD==0)
            // Basic (inefficient) isotropic emission
            Surface(&POS, &DIR) ;
            IndexG(&POS, &level, &ind, DENS, OFF) ;            
#endif
            
#if (PS_METHOD==1)
            // METHOD A
            if (id==0) printf("PS_METHOD_A\n") ;
            // pointsource outside the volume, send all packages to 2pi solid angle
            // => only sqrt(2) decrease in the noise, many packages will miss the cloud altogether
            // *** THIS WORKS ONLY IFF SOURCE SEES ONLY ONE CLOUD SURFACE !!!!! ***
            POS.x     =  PSPOS[level0].x ;   POS.y = PSPOS[level0].y ;   POS.z = PSPOS[level0].z ;
            if (POS.z>NZ) {
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
            PHOTONS *= 0.5f ;  // photons emitted to half-space
            IndexG(&POS, &level, &ind, DENS, OFF) ;            
#endif
            
#if (PS_METHOD==2)
            if (id==0) printf("PS_METHOD_B\n") ;
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
            //       and to choose sides according to their projected area rather than any visible side
            // pointsource outside the volume, send all packages to 2pi solid angle
            // => only sqrt(2) decrease in the noise, many packages will miss the cloud altogether
            POS.x    =  PSPOS[level0].x ;   POS.y = PSPOS[level0].y ;   POS.z = PSPOS[level0].z ;
            // select random side, XY, YZ, or XZ
# if 1      // equal probability for all visible sides (irrespective of their projected visible area)
            ind      =  floor(Rand(&rng)*XPS_NSIDE[level0]) ;  // side is the one in XPS_SIDE[ind], ind=0,1,2
            PHOTONS /=  XPS_AREA[3*level0+ind] ;  // while ind = 0,1,2
            ind      =  XPS_SIDE[3*level0+ind] ;  // side 0-5,  +X,-X,+Y,-Y,+Z,-Z ~ 0-5
# else
            // weight selection according to the projected area == XPS_AREA[]
            // *** XPS_AREA are not yet actually calculated, see host ***
            v1       =  Rand(&rng) ;
            for(ind=0; v1>0.0f; ind++) v1 -= XPS_AREA[3*level0+ind] ; // [0,3[
            PHOTONS *=  ??? ;
            ind      =  XPS_SIDE[3*level0+ind] ;                      // [0,6[, +X,-X,+Y,-Y,+Z,-Z ~ 0-5
# endif
            // printf("SIDE %d\n", ind) ;
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
               POS.z = NZ-PEPS;   POS.x = a*NX ; POS.y = b*NY ;  b = NX*NZ ;
            }
            if (ind==5) { // -Z side
               POS.z = PEPS;      POS.x = a*NX ; POS.y = b*NY ;  b = NX*NZ ;
            }
            // distance and cos(theta)
            v1  = distance(POS, PSPOS[level0]) ;   // distance from source to surface [GL]
            DIR = POS-PSPOS[level0] ;
            DIR = normalize(DIR) ;
            // cos(theta)
            v2  = (ind<2) ? (fabs(DIR.x)) :   ((ind<4) ? (fabs(DIR.y)) : (fabs(DIR.z))) ;
            // re-weight photon numbers
            PHOTONS  *=   v2*b / (4.0f*M_PI*v1*v1) ; // division by XPS_AREA done above when ind was still [0,3[
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
            if (id==0) printf("PS_METHOD_C\n") ;
            ind =  floor(Rnd(&rng)*XPS_NSIDE[1]) ;  // selected probability bin
            ind =  XPS_SIDE[ind] ;                  // corresponding Healpix pixel index
            PHOTONS /= XPS_AREA[ind] ;              // relative weighting of package
            // Generate direction towards that pixel
            // for(int i=0; i<10; i++) {
            Pixel2AnglesRing(ind, float *phi, float *theta) ;
            DIR.x = sin(theta)*cos(phi) ;
            DIR.y = sin(theta)*sin(phi) ;
            DIR.z = cos(theta) ;
            // a random direction near the pixel centre direction
            phi   = TWOPI*Rnd(&rng) ;
            // NSIDE = 128 => pixel size 0.5 degrees ... 0.5 degrees in cos_theta ~ 3.8e-5
            // NSIDE =  64 => pixel size 1   degree
            cos_theta = 1.0-Rand(&rng)*4.0e-5f ;
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
            // We require that the point source is in Z above the cloud, NX==NY
            // and the source (X,Y) coordinates correspond to the centre of the cloud XY plane.
            // In this case we can calculate the cone in which the cloud is seen
            // and packages are only sent within this cone.
            // The solid angle is defined by the distance to the surface and by NX==NY
            if (id==0) printf("PS_METHOD_D\n") ;
            v1         =  PSPOS[level0].z - NZ ;       // height above the cloud surface
            cos_theta  =  v1 / sqrt(v1*v1+(NX*NX)/4.0) ;
            PHOTONS   *=  0.5*(1.0-cos_theta) ;        // because packages go to a smaller solid angle
            // Generate random direction within the cone (some rays will be rejected)
            cos_theta  =  1.0-Rand(&rng)*(1.0f-cos_theta) ;   // now a uniform cos_theta within the cone
            v1         =  TWOPI*Rand(&rng) ;
            DIR.x      =  sqrt(1-cos_theta*cos_theta)*cos(v1) ;
            DIR.y      =  sqrt(1-cos_theta*cos_theta)*sin(v1) ;
            DIR.z      =  -cos_theta ;
            Surface(&POS, &DIR) ;            
            IndexG(&POS, &level, &ind, DENS, OFF) ;            
#endif
            
         } // if pointsource outside the model volume
         
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
      
      
      // printf("%8.4f %8.4f %8.4f    %8.4f %8.4f %8.4f   %10.3e\n", POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, PHOTONS) ;
      
      
      if (fabs(DIR.x)<DEPS) DIR.x = DEPS ;
      if (fabs(DIR.y)<DEPS) DIR.y = DEPS ;
      if (fabs(DIR.z)<DEPS) DIR.z = DEPS ; 
      DIR         =  normalize(DIR) ;
      scatterings =  0 ;
      tau         =  0.0f ;         
      
      
#if (STEP_WEIGHT<=0)
      // normal, unweighted case
      free_path  = -log(Rand(&rng)) ;
#endif
#if (STEP_WEIGHT==1)   
      // Single exponential, free path  *=  1/SW_A,   use 0<SW_A<1
      //   p   =  A*exp(-A*tau)
      //   tau = -log(u/A)/A
      //   w   =  exp(A*tau-tau)/A
      //  SW_A = multiplier for the argument of the exponential function
      free_path   = -log(Rand(&rng)) / SW_A ;
      PHOTONS    *=  exp(SW_A*free_path-free_path) / SW_A ;
#endif
#if (STEP_WEIGHT==2)
      // Two exponentials, alpha and 2*alpha, e.g., alpha=0.5 == exp(-tau) & exp(-0.5*tau)
      //   p    =  0.5*a*exp(-a*t) + a*exp(-2*a*t)
      //   tau  =  -log(-0.5+sqrt(0.25+2*u)) / a
      //   w    =  1 / (  0.5*a*exp((1-a)*t) + a*exp((1-2*a)*t) )
      // SW_A = Weight of modified exponential term
      // SW_B = Multiplier for the argument of the exponential function
      free_path =      -log( 
                            (-SW_B+sqrt(SW_B*SW_B+4.0f*Rand(&rng)*(1.0f-SW_B))) / 
                            (2.0f-2.0f*SW_B) 
                           ) / SW_A ;
      PHOTONS *= 1.0f/(SW_A*SW_B*exp((1.0f-SW_A)*free_path)+2.0f*SW_A*(1.0f-SW_B)*exp((1.0f-2.0f*SW_A)*free_path)) ;
#endif
      
      
      steps = 0 ;
      
      // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
            if (free_path<(tau+dtau)) {  // tau = optical depth since last scattering
               ind = ind0 ;       // what if we scatter on step ending outside the cloud?
               break ;            // scatter before ds
            }
#if (WITH_ABU>0)
            tauA      =  ds*DENS[oind]*OPT[2*oind] ;  // OPT = total cross section, sum over dust species
#else
            tauA      =  ds*DENS[oind]*(*ABS) ;       // ABS is basically a scalar
#endif
            if (tauA>TAULIM) {
               delta  =  PHOTONS*(1.0f-exp(-tauA)) ;
            } else {
               delta  =  PHOTONS*tauA*(1.0f-0.5f*tauA) ;
            }
            // if (DENS[oind]>1.0e-9f) {
            atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
            
            // printf(" DELTA %10.3e\n", delta*TW*ADHOC) ;
            
#if (SAVE_INTENSITY==1)  // Cannot use TABS because that is cumulative over frequency...
            atomicAdd_g_f(&(INT[oind]),  delta) ;
#endif
#if (SAVE_INTENSITY==2)  // Save vector components of the net intensity
            // Cannot use TABS because that is cumulative over frequency...
            atomicAdd_g_f(&(INT[ oind]), delta      ) ;
            atomicAdd_g_f(&(INTX[oind]), delta*DIR.x) ;
            atomicAdd_g_f(&(INTY[oind]), delta*DIR.y) ;
            atomicAdd_g_f(&(INTZ[oind]), delta*DIR.z) ;
#endif
            // }
            PHOTONS   *=  exp(-tauA) ;
            tau       +=  dtau ;
            
            
            // Should have moved to another cell
            if ((level==level0)&&(ind==ind0)) {  // FAILED STEP !!
               // something wrong ???  ---- level 6, stays in the same cell because of POS.x rounding
               // printf("??? 10000 STEPS: fp=%12.4e tau=%12.4e dtau=%12.4e\n", free_path, tau, dtau) ;
#if 1
               POS +=  PEPS * DIR ;
#else
               if (steps>2) {
                  printf("  [%5d]  FROM  %d %d    To %d %d   --- %d\n", id, level0, ind0, level, ind, steps)  ;
                  printf("  POS0  %10.7f %10.7f %10.7f   %10.7f %10.7f %10.7f\n", POS0.x, POS0.y, POS0.z, DIR.x, DIR.y, DIR.z) ;
                  printf("  POS   %10.7f %10.7f %10.7f\n", POS.x, POS.y, POS.z) ;
                  POS +=  PEPS *  DIR ;   // ds is in root grid units
                  printf("  POS   %10.7f %10.7f %10.7f\n", POS.x, POS.y, POS.z) ;
               } else {               
                  POS +=  PEPS * DIR ;
               }
#endif
               steps += 1 ;
               // printf("  POS   %10.7f %10.7f %10.7f\n", POS.x, POS.y, POS.z) ;
               // printf("  POS0  %10.7f %10.7f %10.7f\n", POS0.x, POS0.y, POS0.z) ;
               // printf("  POS   %10.7f %10.7f %10.7f\n", POS.x, POS.y, POS.z) ;
               // printf("[%d] %2d  %7d   DIR = %9.6f %9.6f %9.6f    ds = %10.3e\n", id, level, ind, DIR.x, DIR.y, DIR.z, ds) ;
               // printf("    POS00 %9.6f %9.6f %9.6f\n", POS00.x, POS00.y, POS00.z) ;
               // ind = -1 ; steps = 0 ;  break ;
            }
         } ;
         
         
         // package has scattered or exited the volume
         // if (ind<0) break ;
         if (ind<0) break ;
         
         // OTHERWISE SCATTER
         
         scatterings++ ;
         dtau               =  free_path-tau ;
#if (WITH_ABU>0)
         dx                 =  dtau/(OPT[2*oind+1]*DENS[oind]) ;
         tauA               =  dx*DENS[oind]*OPT[2*oind] ;
#else
         dx                 =  dtau/((*SCA)*DENS[oind]) ;  // actual step forward in GLOBAL coordinates
         tauA               =  dx*DENS[oind]*(*ABS) ;
#endif
#if 0
         if (tauA>TAULIM) {
            delta           =  PHOTONS*(1.0f-exp(-tauA)) ;
         } else {
            delta           =  PHOTONS*tauA*(1.0f-0.5f*tauA) ;
         }
#else
         delta = (tauA>TAULIM) ?  (PHOTONS*(1.0f-exp(-tauA))) : (PHOTONS*tauA*(1.0f-0.5f*tauA)) ;
#endif
         atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
#if (SAVE_INTENSITY==1)  // Cannot use TABS because that is cumulative over frequency...
         atomicAdd_g_f(&(INT[oind]),  delta) ;
#endif
#if (SAVE_INTENSITY==2)  // Save vector components of the net intensity
         // Cannot use TABS because that is cumulative over frequency...
         atomicAdd_g_f(&(INT[oind]),  delta      ) ;
         atomicAdd_g_f(&(INTX[oind]), delta*DIR.x) ;
         atomicAdd_g_f(&(INTY[oind]), delta*DIR.y) ;
         atomicAdd_g_f(&(INTZ[oind]), delta*DIR.z) ;
#endif
         dx             =  ldexp(dx, level0) ;
         dx             =  max(0.0f, dx-2.0f*PEPS) ; // must remain in cell (level0, ind0)
         POS            =  POS0 + dx*DIR ;  // location of scattering -- coordinates of current level
         PHOTONS       *=  exp(-tauA) ;
         
         
#if (STEP_WEIGHT<=0)
         // normal, unweighted case
         free_path  = -log(Rand(&rng)) ;
#endif
#if (STEP_WEIGHT==1)   
         // Single exponential, free path  *=  1/SW_A,   use 0<SW_A<1
         //   p   =  A*exp(-A*tau)
         //   tau = -log(u/A)/A
         //   w   =  exp(A*tau-tau)/A
         //  SW_A = multiplier for the argument of the exponential function
         free_path   = -log(Rand(&rng)) / SW_A ;
         PHOTONS    *=  exp(SW_A*free_path-free_path) / SW_A ;
#endif
#if (STEP_WEIGHT==2)
         // Two exponentials, B and 2*B, e.g., B=0.5 == exp(-tau) & exp(-0.5*tau)
         //   p    =  0.5*a*exp(-a*t) + a*exp(-2*a*t)
         //   tau  =  -log(-0.5+sqrt(0.25+2*u)) / a
         //   w    =  1 / (  0.5*a*exp((1-a)*t) + a*exp((1-2*a)*t) )
         // SW_A = Weight of modified exponential term
         // SW_B = Multiplier for the argument of the exponential function
         free_path =      -log( 
                               (-SW_B+sqrt(SW_B*SW_B+4.0f*Rand(&rng)*(1.0f-SW_B))) / 
                               (2.0f-2.0f*SW_B) 
                              ) / SW_A ;
         PHOTONS  *= 1.0f /     ( SW_A*SW_B*exp((1.0f-SW_A)*free_path) +  2.0f*SW_A*(1.0f-SW_B)*exp((1.0f-2.0f*SW_A)*free_path) ) ;
#endif
         // return to original indices
         ind            =  ind0 ;           // cell has not changed !?
         level          =  level0 ;         // (ind, level) at the beginning of the step = at the end
         

         
#if (DIR_WEIGHT>0)  // cannot be used with -D WITH_MSF
         POS0 = DIR ;  // old direction
         WScatter(&DIR, CSC, &rng, &pweight) ;
         // Yet another weighting = ratio of scattering functions
         // ... for the scattering angle of the photon package (not peeloff direction!!)
         tau           =  DIR.x*POS0.x+DIR.y*POS0.y+DIR.z*POS0.z ;  // ct=="tau" reused         
         pind          =  clamp((int)(BINS*(1.0f+tau)*0.5f), 0, BINS-1) ;
         PHOTONS      *=  DSC[pind] / pweight ;
#else
# if (WITH_MSF==0)
         // Basic situation, only single scattering function in use
         Scatter(&DIR, CSC, &rng) ;   // new direction
# else
         // We must select the scatterer -- using the properties of the cell with global index oind
         // and the relative values of ABU[oind*NDUST+idust]*SCA[idust] / OPT[2*oind+1]
         //  *** re-using ds, free_path, ind0 ***
         dx     =  OPT[2*oind+1] ;        // sum(ABU*SCA) for the current cell
         ds     =  Rand(&rng) ;
         for(ind0=0; ind0<NDUST; ind0++) {   // ind0 ~ dust index
            ds -= ABU[ind0+oind*NDUST]*SCA[ind0] / dx ;
            if (ds<=0.0) break ;
         }
         if (ind0>=NDUST) printf("?????\n") ;
         Scatter(&DIR, &CSC[ind0*BINS], &rng) ; // use the scattering function of the ind0:th dust species
# endif
#endif
         

         
         
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
         
         
         // tau = 0.0f ;
         // end of -- scattering
         
      } // while (ind>0) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      
   } // for III
   
}






__kernel void SimRAM_HP(const      int      PACKETS,  //  0 - number of packets
                        const      int      BATCH,    //  1 - for SOURCE==2, packages per cell
                        const      float    SEED,     //  2
                        __global   float   *ABS,      //  3
                        __global   float   *SCA,      //  4
                        const      float    TW,       //  6 - weight of current frequency in integral
                        constant   int     *LCELLS,   //  7 - number of cells on each level
                        constant   int     *OFF,      //  8 - index of first cell on each level
                        __global   int     *PAR,      //  9 - index of parent cell        [CELLS]
                        __global   float   *DENS,     // 10 - density and hierarchy       [CELLS]
                        __global   float   *EMIT,     // 11 - emission from cells         [CELLS]
                        __global   float   *TABS,     // 12 - buffer for absorptions      [CELLS]
                        constant   float   *DSC,      // 13 - BINS entries
                        constant   float   *CSC,      // 14 - cumulative scattering function
                        __global   float   *XAB,      // 15 - reabsorbed energy per cell  [CELLS]
                        __global   float   *INT,      // 16 - net intensity
                        __global   float   *INTX,     // 17 - net intensity vector
                        __global   float   *INTY,     // 18 - net intensity vector
                        __global   float   *INTZ,     // 19 - net intensity vector
                        __global   float   *OPT,      // 20 - ABS+SCA for abundance runs
                        __global   float   *BG,       // 21 - Healpix map of background intensity
                        __global   float   *HPBGP,    // 22 - cumulative probability for Healpix pixels
                        __global   float   *ABU       // 23 - abundances, for -D MSF only
                       ) 
{
   // This routine is simulating the radiation field in order to determine the
   // absorbed energy in each cell -- background radiation from a Healpix map.
   const int id     = get_global_id(0) ;
   const int GLOBAL = get_global_size(0) ;
   int    oind=0, level=0, scatterings, steps, SIDE ;
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, theta, cos_theta, sin_theta ;
   float3 DIR=0.0f, POS, POS0 ; 
   float  PHOTONS, x, y, z, DX, DY, DZ, v1, v2 ;  
   mwc64x_state_t rng;
   // Assume that 2^38 = 2.7e11 random numbers per worker is sufficient
   // For each NITER, host will also give a new seed [0,1] that is multiplied by 2^32 = 4.3e9
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*(1+id),1.0f)*4294967296L), samplesPerStream);
   
   int ind   = -1 ;
   int ind0=-1, level0 ;
   
   // Each work item simulates BATCH packages
   // although we do not go systematically through cloud surface elements, BGPAC
   // is still defined so that   GLOBAL > 8*AREA   and   BGPAC == 8*USER.AREA*BATCH
   // => ignore work items for which  id > 8*USER.AREA
   if (id>=(8*AREA)) return ;
   
   
   
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
      
      
      // if (id==0) printf("%8.4f %8.4f %8.4f\n", DIR.x, DIR.y, DIR.z) ;
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
      
      // source is at infinite distance => always three sides of the model illuminated
      // calculate the relative projected surface areas to select one of the three
      // area ~ cos(LOS vs. surface normal)  =  component of the unit vector
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
      
      IndexG(&POS, &level, &ind, DENS, OFF) ;
      
      // if (id==0) printf("@ %8.4f %8.4f %8.4f   %8.4f %8.4f %8.4f   %12.4e  %7d\n", POS.x, POS.y, POS.z, DIR.x,DIR.y, DIR.z, PHOTONS, ind) ;
      
      scatterings =  0 ;
      tau         =  0.0f ;         
      
      
#if (STEP_WEIGHT<=0)
      // normal, unweighted case
      free_path  = -log(Rand(&rng)) ;
#endif
#if (STEP_WEIGHT==1)   
      // Single exponential, free path  *=  1/SW_A,   use 0<SW_A<1
      //   p   =  A*exp(-A*tau)
      //   tau = -log(u/A)/A
      //   w   =  exp(A*tau-tau)/A
      //  SW_A = multiplier for the argument of the exponential function
      free_path   = -log(Rand(&rng)) / SW_A ;
      PHOTONS    *=  exp(SW_A*free_path-free_path) / SW_A ;
#endif
#if (STEP_WEIGHT==2)
      // Two exponentials, alpha and 2*alpha, e.g., alpha=0.5 == exp(-tau) & exp(-0.5*tau)
      //   p    =  0.5*a*exp(-a*t) + a*exp(-2*a*t)
      //   tau  =  -log(-0.5+sqrt(0.25+2*u)) / a
      //   w    =  1 / (  0.5*a*exp((1-a)*t) + a*exp((1-2*a)*t) )
      // SW_A = Weight of modified exponential term
      // SW_B = Multiplier for the argument of the exponential function
      free_path =      -log( 
                            (-SW_B+sqrt(SW_B*SW_B+4.0f*Rand(&rng)*(1.0f-SW_B))) / 
                            (2.0f-2.0f*SW_B) 
                           ) / SW_A ;
      PHOTONS *= 1.0f/(SW_A*SW_B*exp((1.0f-SW_A)*free_path)+2.0f*SW_A*(1.0f-SW_B)*exp((1.0f-2.0f*SW_A)*free_path)) ;
#endif
      
      
      steps = 0 ;
      
      // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
            if (free_path<(tau+dtau)) {  // tau = optical depth since last scattering
               ind = ind0 ;       // what if we scatter on step ending outside the cloud?
               break ;            // scatter before ds
            }
#if (WITH_ABU>0)
            tauA      =  ds*DENS[oind]*OPT[2*oind] ;
#else
            tauA      =  ds*DENS[oind]*(*ABS) ;
#endif
            if (tauA>TAULIM) {
               delta  =  PHOTONS*(1.0f-exp(-tauA)) ;
            } else {
               delta  =  PHOTONS*tauA*(1.0f-0.5f*tauA) ;
            }
            // if (DENS[oind]>1.0e-9f) {
            atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
#if (SAVE_INTENSITY==1)  // Cannot use TABS because that is cumulative over frequency...
            atomicAdd_g_f(&(INT[oind]),  delta) ;
#endif
#if (SAVE_INTENSITY==2)  // Save vector components of the net intensity
            // Cannot use TABS because that is cumulative over frequency...
            atomicAdd_g_f(&(INT[ oind]), delta      ) ;
            atomicAdd_g_f(&(INTX[oind]), delta*DIR.x) ;
            atomicAdd_g_f(&(INTY[oind]), delta*DIR.y) ;
            atomicAdd_g_f(&(INTZ[oind]), delta*DIR.z) ;
#endif
            // }
            PHOTONS   *=  exp(-tauA) ;
            tau       +=  dtau ;
            
            
            // Should have moved to another cell
            if ((level==level0)&&(ind==ind0)) {  // FAILED STEP !!
               // something wrong ???  ---- level 6, stays in the same cell because of POS.x rounding
               // printf("??? 10000 STEPS: fp=%12.4e tau=%12.4e dtau=%12.4e\n", free_path, tau, dtau) ;
#if 1
               POS +=  PEPS * DIR ;
#else
               if (steps>2) {
                  printf("  [%5d]  FROM  %d %d    To %d %d   --- %d\n", id, level0, ind0, level, ind, steps)  ;
                  printf("  POS0  %10.7f %10.7f %10.7f   %10.7f %10.7f %10.7f\n", POS0.x, POS0.y, POS0.z, DIR.x, DIR.y, DIR.z) ;
                  printf("  POS   %10.7f %10.7f %10.7f\n", POS.x, POS.y, POS.z) ;
                  POS +=  PEPS *  DIR ;   // ds is in root grid units
                  printf("  POS   %10.7f %10.7f %10.7f\n", POS.x, POS.y, POS.z) ;
               } else {               
                  POS +=  PEPS * DIR ;
               }
#endif
               steps += 1 ;
               // printf("  POS   %10.7f %10.7f %10.7f\n", POS.x, POS.y, POS.z) ;
               // printf("  POS0  %10.7f %10.7f %10.7f\n", POS0.x, POS0.y, POS0.z) ;
               // printf("  POS   %10.7f %10.7f %10.7f\n", POS.x, POS.y, POS.z) ;
               // printf("[%d] %2d  %7d   DIR = %9.6f %9.6f %9.6f    ds = %10.3e\n", id, level, ind, DIR.x, DIR.y, DIR.z, ds) ;
               // printf("    POS00 %9.6f %9.6f %9.6f\n", POS00.x, POS00.y, POS00.z) ;
               // ind = -1 ; steps = 0 ;  break ;
            }
         } ;
         
         
         // package has scattered or exited the volume
         // if (ind<0) break ;
         if (ind<0) break ;
         
         // OTHERWISE SCATTER
         
         scatterings++ ;
         dtau               =  free_path-tau ;
#if (WITH_ABU>0)
         dx                 =  dtau/(OPT[2*oind+1]*DENS[oind]) ;
         tauA               =  dx*DENS[oind]*OPT[2*oind] ;
#else
         dx                 =  dtau/((*SCA)*DENS[oind]) ;  // actual step forward in GLOBAL coordinates
         tauA               =  dx*DENS[oind]*(*ABS) ;
#endif
#if 0
         if (tauA>TAULIM) {
            delta           =  PHOTONS*(1.0f-exp(-tauA)) ;
         } else {
            delta           =  PHOTONS*tauA*(1.0f-0.5f*tauA) ;
         }
#else
         delta = (tauA>TAULIM) ?  (PHOTONS*(1.0f-exp(-tauA))) : (PHOTONS*tauA*(1.0f-0.5f*tauA)) ;
#endif
         atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
#if (SAVE_INTENSITY==1)  // Cannot use TABS because that is cumulative over frequency...
         atomicAdd_g_f(&(INT[oind]),  delta) ;
#endif
#if (SAVE_INTENSITY==2)  // Save vector components of the net intensity
         // Cannot use TABS because that is cumulative over frequency...
         atomicAdd_g_f(&(INT[oind]),  delta      ) ;
         atomicAdd_g_f(&(INTX[oind]), delta*DIR.x) ;
         atomicAdd_g_f(&(INTY[oind]), delta*DIR.y) ;
         atomicAdd_g_f(&(INTZ[oind]), delta*DIR.z) ;
#endif
         dx             =  ldexp(dx, level0) ;
         dx             =  max(0.0f, dx-2.0f*PEPS) ; // must remain in cell (level0, ind0)
         POS            =  POS0 + dx*DIR ;  // location of scattering -- coordinates of current level
         PHOTONS       *=  exp(-tauA) ;
         
#if (STEP_WEIGHT<=0)
         // normal, unweighted case
         free_path  = -log(Rand(&rng)) ;
#endif
#if (STEP_WEIGHT==1)   
         // Single exponential, free path  *=  1/SW_A,   use 0<SW_A<1
         //   p   =  A*exp(-A*tau)
         //   tau = -log(u/A)/A
         //   w   =  exp(A*tau-tau)/A
         //  SW_A = multiplier for the argument of the exponential function
         free_path   = -log(Rand(&rng)) / SW_A ;
         PHOTONS    *=  exp(SW_A*free_path-free_path) / SW_A ;
#endif
         
         
#if (STEP_WEIGHT==2)
         // Two exponentials, B and 2*B, e.g., B=0.5 == exp(-tau) & exp(-0.5*tau)
         //   p    =  0.5*a*exp(-a*t) + a*exp(-2*a*t)
         //   tau  =  -log(-0.5+sqrt(0.25+2*u)) / a
         //   w    =  1 / (  0.5*a*exp((1-a)*t) + a*exp((1-2*a)*t) )
         // SW_A = Weight of modified exponential term
         // SW_B = Multiplier for the argument of the exponential function
         free_path =      -log( 
                               (-SW_B+sqrt(SW_B*SW_B+4.0f*Rand(&rng)*(1.0f-SW_B))) / 
                               (2.0f-2.0f*SW_B) 
                              ) / SW_A ;
         PHOTONS  *= 1.0f /     ( SW_A*SW_B*exp((1.0f-SW_A)*free_path) +  2.0f*SW_A*(1.0f-SW_B)*exp((1.0f-2.0f*SW_A)*free_path) ) ;
#endif
         // return to original indices
         ind            =  ind0 ;           // cell has not changed !?
         level          =  level0 ;         // (ind, level) at the beginning of the step = at the end
         
         
         
#if (DIR_WEIGHT>0)  // =============================================================================================
         POS0 = DIR ;  // old direction
         WScatter(&DIR, CSC, &rng, &pweight) ;
         // Yet another weighting = ratio of scattering functions
         // ... for the scattering angle of the photon package (not peeloff direction!!)
         tau           =  DIR.x*POS0.x+DIR.y*POS0.y+DIR.z*POS0.z ;  // ct=="tau" reused         
         pind          =  clamp((int)(BINS*(1.0f+tau)*0.5f), 0, BINS-1) ;
         PHOTONS      *=  DSC[pind] / pweight ;
#else // ===========================================================================================================
         
         
# if (WITH_MSF==0) // ----------------------------------------------------------------------------------------------
         // Basic situation, only single scattering function in use
         Scatter(&DIR, CSC, &rng) ;   // new direction
# else // ----------------------------------------------------------------------------------------------------------
         // We must select the scatterer -- using the properties of the cell with global index oind
         // and the relative values of ABU[idust+NDUST*oind]*SCA[idust] / OPT[2*oind+1]
         //  *** re-using ds, free_path, ind0 ***
         dx     =  OPT[2*oind+1] ;        // sum(ABU*SCA) for the current cell
         ds     =  Rand(&rng) ;
         for(ind0=0; ind0<NDUST; ind0++) {   // ind0 ~ dust index
            ds -= ABU[ind0+NDUST*oind]*SCA[ind0] / dx  ;
            if (ds<=0.0) break ;
         }
         if (ind0>=NDUST) printf("?????\n") ;
         Scatter(&DIR, &CSC[ind0*BINS], &rng) ; // use the scattering function of the ind0:th dust species
# endif // --------------------------------------------------------------------------------------------------------
         
         
#endif // ==========================================================================================================
         
         
         
         
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
         
         
         // tau = 0.0f ;
         // end of -- scattering
         
      } // while (ind>0) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      
   } // for III
   
}









// KERNEL FOR CLPAC ONLY
         

#if (USE_EMWEIGHT<2)

__kernel void SimRAM_CL(const      int      SOURCE,  //  0 - PSPAC/BGPAC/CLPAC = 0/1/2
                        const      int      PACKETS, //  1 - number of packets
                        const      int      BATCH,   //  2 - for SOURCE==2, packages per cell
                        const      float    SEED,    //  3
                        __global   float   *ABS,     //  4
                        __global   float   *SCA,     //  5
                        // const      float    BG,      //  7 - background intensity
                        // const      float    PS,      //  8 - point source luminosity
                        // __global   float3  *PSPOS,   //  9 - position of point source
                        const      float    TW,      // 10 - weight of current frequency in integral
                        constant   int     *LCELLS,  // 11 - number of cells on each level
                        constant   int     *OFF,     // 12 - index of first cell on each level
                        __global   int     *PAR,     // 13 - index of parent cell        [CELLS]
                        __global   float   *DENS,    // 14 - density and hierarchy       [CELLS]
                        __global   float   *EMIT,    // 15 - emission from cells         [CELLS]
                        __global   float   *TABS,    // 16 - buffer for absorptions      [CELLS]
                        constant   float   *DSC,     // 17 - BINS entries
                        constant   float   *CSC,     // 18 - cumulative scattering function
                        __global   float   *XAB,     // 19 - reabsorbed energy per cell  [CELLS]
                        __global   float   *EMWEI,   // 20 - number of packages per cell
                        __global   float   *INT,     // 21 - net intensity
                        __global   float   *INTX,    // 22 - net intensity vector
                        __global   float   *INTY,    // 23 - net intensity vector
                        __global   float   *INTZ,    // 24 - net intensity vector
                        __global   int     *EMINDEX, // 25 - dummy
                        __global   float   *OPT,     // 26 - ABS + SCA vectors
                        __global   float   *ABU      // 27 - abundances, for -D MSF only
                       ) 
{
   // This routine is simulating the radiation field in order to determine the
   // absorbed energy in each cell.
   const int id     = get_global_id(0) ;
   if (id>=CELLS) return ;
   const int GLOBAL = get_global_size(0) ;
   
   int    oind=0, level=0, scatterings, batch ;
# if (WITH_ALI>0)
   int    e_index ; // for photon packages sent from a cell, the index of emitting cell
# endif
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, sin_theta ;
   float3 DIR=0.0f, POS, POS0 ; 
   float  PHOTONS, X0, Y0, Z0, PWEI=1.0f ;
   
   mwc64x_state_t rng;
   // Assume that 2^38 = 2.7e11 random numbers per worker is sufficient
   // For each NITER, host will also give a new seed [0,1] that is multiplied by 2^32 = 4.3e9
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*(id+1),1.0f)*4294967296L), samplesPerStream);
   
# if (USE_EMWEIGHT==2)
   // Host provides EMINDEX[] = cell indices, one per emitted ray.
   // Length of EMINDEX is >= number-of-rays + GLOBAL, last values being -1.
   // Host also provides EMWEI[CELLS] = 1.0/no_of_rays.
   int IENTRY = id-GLOBAL ;
# else
   int ICELL = id-GLOBAL ;
   int IRAY  =  0 ;
# endif
   int ind   = -1 ;
   int ind0, level0 ;
   batch = -1 ;
   
   
   
   while(1) {
      
      // generate new ray >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
         
# if (WITH_ALI>0)
      e_index = -1 ;
# endif
      
      // =============== CLPAC ===================
      
      
# if (USE_EMWEIGHT==2)
      // Figure out cell index == ind and its weight == PWEI
      // These come from the host, in EMINDEX and EMWEI
      // In this case IENTRY is index to EMINDEX, not directly a cell index
      IENTRY += GLOBAL ;          // next ray
      ind     = EMINDEX[IENTRY] ; // host makes sure EMINDEX >= rays+GLOBAL, last values -1
      if (ind<0) return ;         // no more rays
      PWEI    = EMWEI[ind] ;      // host sets EMWEI == 1.0/ number-of-rays-from-cell
      
# else // =============================================================================
      
      // ICELL = id+()*GLOBAL < CELLS
      // IRAY  = 0...         < batch = [BATCH | EMWEI]
      if (IRAY>=batch) {    // move to next cell (on first call 0>=-1)
         // if (id==0) printf("from %6d ", ICELL) ;
         IRAY  = 0 ;
         PWEI  = 1.0f ;     // additional weighting to photon numbers
         while(1) {
            ICELL += GLOBAL ;          // initially ICELL==id-GLOBAL
            if (ICELL>=CELLS) {
               return ; // no more cells for this work item
            }
#  if (USE_EMWEIGHT>0)
            PWEI  = EMWEI[ICELL] ;     // float ~ number of packages
            if ((PWEI<1e-10f)||(DENS[ICELL]<1.0e-20f)) {
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
#  else
            batch = BATCH ;
            // COMMENT OUT THE FOLLOWING LINES AND GPU IS WORKING ?????????
            //               if ((batch<=0)||(DENS[ICELL]<1.0e-20f)) {
            //                  continue ;  // no rays from this cell
            //               }
            PWEI  = 1.0f/(batch+1.0e-9f) ;
#  endif
            break ;
         } // while(1)
         // if (id==0) printf("to %6d [%d]\n", ICELL, IRAY) ;
      } // IRAY>=batch 
      ind     =  ICELL ;
      // if ((ind<0)||(ind>=CELLS)) return ; // ???
      IRAY   +=  1 ;     // this many rays created for the current cell
# endif // ===========================================================================
      
      
      
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
# if (WITH_ALI>0)
      e_index    =  OFF[level]+ind ;  // index of the emitting cell
# endif
      
      
      if (fabs(DIR.x)<DEPS) DIR.x = DEPS ;
      if (fabs(DIR.y)<DEPS) DIR.y = DEPS ;
      if (fabs(DIR.z)<DEPS) DIR.z = DEPS ; 
      DIR         =  normalize(DIR) ;
      scatterings =  0 ;
      tau         =  0.0f ;         
      
      
# if (STEP_WEIGHT<=0)
      // normal, unweighted case
      free_path  = -log(Rand(&rng)) ;
# endif
# if (STEP_WEIGHT==1)   
      // Single exponential, free path  *=  1/SW_A,   use 0<SW_A<1
      //   p   =  A*exp(-A*tau)
      //   tau = -log(u/A)/A
      //   w   =  exp(A*tau-tau)/A
      //  SW_A = multiplier for the argument of the exponential function
      free_path   = -log(Rand(&rng)) / SW_A ;
      PHOTONS    *=  exp(SW_A*free_path-free_path) / SW_A ;
# endif
# if (STEP_WEIGHT==2)
      // Two exponentials, alpha and 2*alpha, e.g., alpha=0.5 == exp(-tau) & exp(-0.5*tau)
      //   p    =  0.5*a*exp(-a*t) + a*exp(-2*a*t)
      //   tau  =  -log(-0.5+sqrt(0.25+2*u)) / a
      //   w    =  1 / (  0.5*a*exp((1-a)*t) + a*exp((1-2*a)*t) )
      // SW_A = Weight of modified exponential term
      // SW_B = Multiplier for the argument of the exponential function
      free_path =      -log( 
                            (-SW_B+sqrt(SW_B*SW_B+4.0f*Rand(&rng)*(1.0f-SW_B))) / 
                            (2.0f-2.0f*SW_B) 
                           ) / SW_A ;
      PHOTONS *= 1.0f/(SW_A*SW_B*exp((1.0f-SW_A)*free_path)+2.0f*SW_A*(1.0f-SW_B)*exp((1.0f-2.0f*SW_A)*free_path)) ;
# endif
      
      // ind<0 -- generate new package <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      
      
      
      // if (id==0) printf("DFPAC %12.4e\n", PHOTONS) ;
      
      
      
      while(ind>=0) {  // loop until this ray is really finished
         
         
         while(ind>=0) {    // loop until scattering
            oind      =  OFF[level]+ind ;  // global index at the beginning of the step
            ind0      =  ind ;             // indices at the beginning of the step
            level0    =  level ;
            POS0      =  POS ;   // because GetStep does coordinate transformations...
            ds        =  GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // POS, level, ind updated !!
# if (WITH_ABU>0)
            dtau      =  ds*DENS[oind]*OPT[2*oind+1] ;
# else
            dtau      =  ds*DENS[oind]*(*SCA) ;
# endif
            if (free_path<(tau+dtau)) {
               ind = ind0 ;       // what if we scatter on step ending outside the cloud?
               break ;            // scatter before ds
            }
# if (WITH_ABU>0)
            tauA      =  ds*DENS[oind]*OPT[2*oind] ;
# else
            tauA      =  ds*DENS[oind]*(*ABS) ;
# endif
            if (tauA>TAULIM) {
               delta  =  PHOTONS*(1.0f-exp(-tauA)) ;
            } else {
               delta  =  PHOTONS*tauA*(1.0f-0.5f*tauA) ;
            }
# if (WITH_ALI==1)  // with ALI
            if (oind==e_index) {  // absorptions in emitting cell
               atomicAdd_g_f(&(XAB[oind]), delta*TW) ;
            } else {              // absorptions of photons from elsewhere
               atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
            }
# else              // no ALI
            atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
            
            
            // if (oind==1201) printf("%12.4e to %6d si %12.4e\n", delta*TW*ADHOC, oind, TABS[oind]) ;
            
# endif
# if (SAVE_INTENSITY==1)  // Cannot use TABS because that is cumulative over frequency...
            atomicAdd_g_f(&(INT[oind]),  delta) ;
# endif
# if (SAVE_INTENSITY==2)  // Save vector components of the net intensity
            // Cannot use TABS because that is cumulative over frequency...
            atomicAdd_g_f(&(INT[ oind]), delta      ) ;
            atomicAdd_g_f(&(INTX[oind]), delta*DIR.x) ;
            atomicAdd_g_f(&(INTY[oind]), delta*DIR.y) ;
            atomicAdd_g_f(&(INTZ[oind]), delta*DIR.z) ;
# endif
            PHOTONS   *=  exp(-tauA) ;
            tau       +=  dtau ;
         } ;
         
         // package has scattered or exited the volume
      
         
         
         
         if (ind>=0) {  // it was a scattering
            scatterings++ ;
# if 1 
            if (scatterings>20) {
               ind = -1 ;  continue  ; // go and get next ray
            }
# else
            // Russian roulette to remove packages
            // perhaps we skip this in for CLPAC -- emission is always at long wavelengths
            if (scatterings>15) {
               if (Rand(&rng)<0.25f) {   // one in four terminated
                  ind = -1 ; continue ;
               } else {                  // ther rest *= 4/3
                  PHOTONS *= 1.3333333f ;
               }
            }
# endif
            dtau               =  free_path-tau ;
# if (WITH_ABU>0)
            dx                 =  dtau/(OPT[2*oind+1]*DENS[oind]) ;
            tauA               =  dx*DENS[oind]*OPT[2*oind] ;
# else
            dx                 =  dtau/((*SCA)*DENS[oind]) ;  // actual step forward in GLOBAL coordinates
            tauA               =  dx*DENS[oind]*(*ABS) ;
# endif
            if (tauA>TAULIM) {
               delta           =  PHOTONS*(1.0f-exp(-tauA)) ;
            } else {
               delta           =  PHOTONS*tauA*(1.0f-0.5f*tauA) ;
            }
# if (WITH_ALI==1)   // with ALI
            if (oind==e_index) { // absorptions in emitting cell
               atomicAdd_g_f(&(XAB[oind]), delta*TW) ;
            } else {             // absorptions of photons from elsewhere
               atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
            }
# else               // no ALI
            atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
# endif
# if (SAVE_INTENSITY==1)  // Cannot use TABS because that is cumulative over frequency...
            atomicAdd_g_f(&(INT[oind]),  delta) ;
# endif
# if (SAVE_INTENSITY==2)  // Save vector components of the net intensity
            // Cannot use TABS because that is cumulative over frequency...
            atomicAdd_g_f(&(INT[oind]),  delta      ) ;
            atomicAdd_g_f(&(INTX[oind]), delta*DIR.x) ;
            atomicAdd_g_f(&(INTY[oind]), delta*DIR.y) ;
            atomicAdd_g_f(&(INTZ[oind]), delta*DIR.z) ;
# endif
            dx             =  ldexp(dx, level0) ;
            dx             =  max(0.0f, dx-2.0f*PEPS) ; // must remain in cell (level0, ind0)
            POS            =  POS0 + dx*DIR ;  // location of scattering -- coordinates of current level
            PHOTONS       *=  exp(-tauA) ;
            
# if (STEP_WEIGHT<=0)
            // normal, unweighted case
            free_path  = -log(Rand(&rng)) ;
# endif
# if (STEP_WEIGHT==1)   
            // Single exponential, free path  *=  1/SW_A,   use 0<SW_A<1
            //   p   =  A*exp(-A*tau)
            //   tau = -log(u/A)/A
            //   w   =  exp(A*tau-tau)/A
            //  SW_A = multiplier for the argument of the exponential function
            free_path   = -log(Rand(&rng)) / SW_A ;
            PHOTONS    *=  exp(SW_A*free_path-free_path) / SW_A ;
# endif
# if (STEP_WEIGHT==2)
            // Two exponentials, B and 2*B, e.g., B=0.5 == exp(-tau) & exp(-0.5*tau)
            //   p    =  0.5*a*exp(-a*t) + a*exp(-2*a*t)
            //   tau  =  -log(-0.5+sqrt(0.25+2*u)) / a
            //   w    =  1 / (  0.5*a*exp((1-a)*t) + a*exp((1-2*a)*t) )
            // SW_A = Weight of modified exponential term
            // SW_B = Multiplier for the argument of the exponential function
            free_path =      -log( 
                                  (-SW_B+sqrt(SW_B*SW_B+4.0f*Rand(&rng)*(1.0f-SW_B))) / 
                                  (2.0f-2.0f*SW_B) 
                                 ) / SW_A ;
            PHOTONS  *= 1.0f /     ( SW_A*SW_B*exp((1.0f-SW_A)*free_path) +  2.0f*SW_A*(1.0f-SW_B)*exp((1.0f-2.0f*SW_A)*free_path) ) ;
# endif
            // return to original indices
            ind            =  ind0 ;           // cell has not changed !?
            level          =  level0 ;         // (ind, level) at the beginning of the step = at the end
            
            
            
# if (DIR_WEIGHT>0) // =============================================================================================
            POS0 = DIR ;  // old direction
            WScatter(&DIR, CSC, &rng, &pweight) ;
            // Yet another weighting = ratio of scattering functions
            // ... for the scattering angle of the photon package (not peeloff direction!!)
            tau           =  DIR.x*POS0.x+DIR.y*POS0.y+DIR.z*POS0.z ;  // ct=="tau" reused         
            pind          =  clamp((int)(BINS*(1.0f+tau)*0.5f), 0, BINS-1) ;
            PHOTONS      *=  DSC[pind] / pweight ;
# else // ==========================================================================================================
            
#  if (WITH_MSF==0) // ----------------------------------------------------------------------------------------------
            // Basic situation, only single scattering function in use
            Scatter(&DIR, CSC, &rng) ;   // new direction
#  else // ----------------------------------------------------------------------------------------------------------
            // We must select the scatterer -- using the properties of the cell with global index oind
            // and the relative values of ABU[idust+NDUST*oind]*SCA[idust] / OPT[2*oind+1]
            //  *** re-using ds, free_path, ind0 ***
            free_path =  OPT[2*oind+1] ;        // sum(ABU*SCA) for the current cell
            ds        =  Rand(&rng) ;
            for(ind0=0; ind0<NDUST; ind0++) {   // ind0 ~ dust index
               ds -= ABU[ind0+NDUST*oind]*SCA[ind0] / OPT[2*oind+1] ;
               if (ds<=0.0) break ;
            }
            if (ind0>=NDUST) printf("?????\n") ;
            Scatter(&DIR, &CSC[ind0*BINS], &rng) ; // use the scattering function of the ind0:th dust species
#  endif // --------------------------------------------------------------------------------------------------------
            
# endif // ========================================================================================================
            
         } // end of -- scattering
         
         
      } // while (ind>0) -- loop until the end of ray before checking if new are required
      
      
   } // while (1) -- loop until all packages simulated
   
}






#else  // USE_EMWEIGHT == 2

// In this version host provides package weights in EMWEI
// indices of simulated cells in EMINDEX
// kernel simulates *3* packages for each of these cells
// Also, GLOBAL<<CELLS
//   => each thread loops over many cells
         

__kernel void SimRAM_CL(const      int      SOURCE,  //  0 - PSPAC/BGPAC/CLPAC = 0/1/2
                        const      int      PACKETS, //  1 - number of packets
                        const      int      BATCH,   //  2 - for SOURCE==2, packages per cell
                        const      float    SEED,    //  3
                        __global   float   *ABS,     //  4
                        __global   float   *SCA,     //  5
                        // const      float    BG,      //  7 - background intensity
                        // const      float    PS,      //  8 - point source luminosity
                        // __global   float3  *PSPOS,   //  9 - position of point source
                        const      float    TW,      // 10 - weight of current frequency in integral
                        constant   int     *LCELLS,  // 11 - number of cells on each level
                        constant   int     *OFF,     // 12 - index of first cell on each level
                        __global   int     *PAR,     // 13 - index of parent cell        [CELLS]
                        __global   float   *DENS,    // 14 - density and hierarchy       [CELLS]
                        __global   float   *EMIT,    // 15 - emission from cells         [CELLS]
                        __global   float   *TABS,    // 16 - buffer for absorptions      [CELLS]
                        constant   float   *DSC,     // 17 - BINS entries
                        constant   float   *CSC,     // 18 - cumulative scattering function
                        __global   float   *XAB,     // 19 - reabsorbed energy per cell  [CELLS]
                        __global   float   *EMWEI,   // 20 - number of packages per cell
                        __global   float   *INT,     // 21 - net intensity
                        __global   float   *INTX,    // 22 - net intensity vector
                        __global   float   *INTY,    // 23 - net intensity vector
                        __global   float   *INTZ,    // 24 - net intensity vector
                        __global   int     *EMINDEX, // 25 - dummy !!
                        __global   float   *OPT      // 26 - ABS + SCA vectors when abundances are not constant
                       ) 
{
   // This routine is simulating the radiation field in order to determine the
   // absorbed energy in each cell.
   const int id     = get_global_id(0) ;
   if (id>=CELLS) return ;
   const int GLOBAL = get_global_size(0) ;
   
   int    oind=0, level=0, scatterings  ;
   int    level0, ind0 ;
# if (WITH_ALI>0)
   int    e_index ; // for photon packages sent from a cell, the index of emitting cell
# endif
   float  ds, free_path, tau, dtau, delta, tauA, dx, phi, cos_theta, sin_theta ;
   float3 DIR=0.0f, POS, POS0 ; 
   float  PHOTONS, X0, Y0, Z0, PWEI=1.0f ;
   
   mwc64x_state_t rng;
   // Assume that 2^38 = 2.7e11 random numbers per worker is sufficient
   // For each NITER, host will also give a new seed [0,1] that is multiplied by 2^32 = 4.3e9
   // The generator has a period of 2^63=9e18, which is 33e6 times 2^38... 33e6 >> workers
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*(id+1),1.0f)*4294967296L), samplesPerStream);
   
   
   int IND = id-GLOBAL, ind ;   // index to EMINDEX
   int ICELL = -1 ; // current cell
   
   // Loop over cells
   while (1) {
      
      // if (id==4100)  printf("IND=%d   < CELLS=%d\n", IND, CELLS) ;
      
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
      
      // if (ICELL==3789) printf(" <<<<<<  3789  >>>>>> %d\n", id) ;
      
      
      // !!! iter over EMWEI2_STEP photon packages !!!
      for(int iter=0; iter<100; iter++) { // three packages from this cell
         
         // if (id==4100) printf("   ----  %5d  %8d  %d/3\n", id, IND, 1+iter) ;
                    
         ind = ICELL ;
         
         // generate new ray >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
         
# if (WITH_ALI>0)
         e_index = -1 ;
# endif
         
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
# if (WITH_ALI>0)
         e_index    =  OFF[level]+ind ;  // index of the emitting cell
# endif
         
         
         if (fabs(DIR.x)<DEPS) DIR.x = DEPS ;
         if (fabs(DIR.y)<DEPS) DIR.y = DEPS ;
         if (fabs(DIR.z)<DEPS) DIR.z = DEPS ; 
         DIR         =  normalize(DIR) ;
         scatterings =  0 ;
         tau         =  0.0f ;         
         
         
# if (STEP_WEIGHT<=0)
         // normal, unweighted case
         free_path  = -log(Rand(&rng)) ;
# endif
# if (STEP_WEIGHT==1)   
         // Single exponential, free path  *=  1/SW_A,   use 0<SW_A<1
         //   p   =  A*exp(-A*tau)
         //   tau = -log(u/A)/A
         //   w   =  exp(A*tau-tau)/A
         //  SW_A = multiplier for the argument of the exponential function
         free_path   = -log(Rand(&rng)) / SW_A ;
         PHOTONS    *=  exp(SW_A*free_path-free_path) / SW_A ;
# endif
# if (STEP_WEIGHT==2)
         // Two exponentials, alpha and 2*alpha, e.g., alpha=0.5 == exp(-tau) & exp(-0.5*tau)
         //   p    =  0.5*a*exp(-a*t) + a*exp(-2*a*t)
         //   tau  =  -log(-0.5+sqrt(0.25+2*u)) / a
         //   w    =  1 / (  0.5*a*exp((1-a)*t) + a*exp((1-2*a)*t) )
         // SW_A = Weight of modified exponential term
         // SW_B = Multiplier for the argument of the exponential function
         free_path =      -log( 
                               (-SW_B+sqrt(SW_B*SW_B+4.0f*Rand(&rng)*(1.0f-SW_B))) / 
                               (2.0f-2.0f*SW_B) 
                              ) / SW_A ;
         PHOTONS *= 1.0f/(SW_A*SW_B*exp((1.0f-SW_A)*free_path)+2.0f*SW_A*(1.0f-SW_B)*exp((1.0f-2.0f*SW_A)*free_path)) ;
# endif
         
         // ind<0 -- generate new package <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      
         
         
         // if (id==0) printf("DFPAC %12.4e\n", PHOTONS) ;
      
         
         
         while(ind>=0) {  // loop until this ray is really finished
            
            
            while(ind>=0) {    // loop until scattering
               oind      =  OFF[level]+ind ;  // global index at the beginning of the step
               ind0      =  ind ;             // indices at the beginning of the step
               level0    =  level ;
               POS0      =  POS ;   // because GetStep does coordinate transformations...
               ds        =  GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // POS, level, ind updated !!
# if (WITH_ABU>0)
               dtau      =  ds*DENS[oind]*OPT[2*oind+1] ;
# else
               dtau      =  ds*DENS[oind]*(*SCA) ;
# endif
               if (free_path<(tau+dtau)) {
                  ind = ind0 ;       // what if we scatter on step ending outside the cloud?
                  break ;            // scatter before ds
               }
# if (WITH_ABU>0)
               tauA      =  ds*DENS[oind]*OPT[2*oind] ;
# else
               tauA      =  ds*DENS[oind]*(*ABS) ;  // single dust or constant abundance = single float
# endif
               if (tauA>TAULIM) {
                  delta  =  PHOTONS*(1.0f-exp(-tauA)) ;
               } else {
                  delta  =  PHOTONS*tauA*(1.0f-0.5f*tauA) ;
               }
# if (WITH_ALI==1)  // with ALI
               if (oind==e_index) {  // absorptions in emitting cell
                  atomicAdd_g_f(&(XAB[oind]), delta*TW) ;
               } else {              // absorptions of photons from elsewhere
                  atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
               }
# else              // no ALI
               atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
               // if (oind==1201) printf("%12.4e to %6d si %12.4e\n", delta*TW*ADHOC, oind, TABS[oind]) ;
# endif
# if (SAVE_INTENSITY==1)  // Cannot use TABS because that is cumulative over frequency...
               atomicAdd_g_f(&(INT[oind]),  delta) ;
# endif
# if (SAVE_INTENSITY==2)  // Save vector components of the net intensity
               // Cannot use TABS because that is cumulative over frequency...
               atomicAdd_g_f(&(INT[ oind]), delta      ) ;
               atomicAdd_g_f(&(INTX[oind]), delta*DIR.x) ;
               atomicAdd_g_f(&(INTY[oind]), delta*DIR.y) ;
               atomicAdd_g_f(&(INTZ[oind]), delta*DIR.z) ;
# endif
               PHOTONS   *=  exp(-tauA) ;
               tau       +=  dtau ;
            } ;
            
            // package has scattered or exited the volume
      
            
            
            
            if (ind>=0) {  // it was a scattering
               scatterings++ ;
# if 1 
               if (scatterings>20) {
                  ind = -1 ;  continue  ; // go and get next ray
               }
# else
               // Russian roulette to remove packages
               // perhaps we skip this in for CLPAC -- emission is always at long wavelengths
               if (scatterings>15) {
                  if (Rand(&rng)<0.25f) {   // one in four terminated
                     ind = -1 ; continue ;
                  } else {                  // ther rest *= 4/3
                     PHOTONS *= 1.3333333f ;
                  }
               }
# endif
               dtau               =  free_path-tau ;
# if (WITH_ABU>0)
               dx                 =  dtau/(OPT[2*oind+1]*DENS[oind]) ;  // actual step forward in GLOBAL coordinates
               tauA               =  dx*DENS[oind]*OPT[2*oind] ;
# else
               dx                 =  dtau/((*SCA)*DENS[oind]) ;  // actual step forward in GLOBAL coordinates
               tauA               =  dx*DENS[oind]*(*ABS) ;
# endif
               if (tauA>TAULIM) {
                  delta           =  PHOTONS*(1.0f-exp(-tauA)) ;
               } else {
                  delta           =  PHOTONS*tauA*(1.0f-0.5f*tauA) ;
               }
# if (WITH_ALI==1)   // with ALI
               if (oind==e_index) { // absorptions in emitting cell
                  atomicAdd_g_f(&(XAB[oind]), delta*TW) ;
               } else {             // absorptions of photons from elsewhere
                  atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
               }
# else               // no ALI
               atomicAdd_g_f(&(TABS[oind]), delta*TW*ADHOC) ;
# endif
# if (SAVE_INTENSITY==1)  // Cannot use TABS because that is cumulative over frequency...
               atomicAdd_g_f(&(INT[oind]),  delta) ;
# endif
# if (SAVE_INTENSITY==2)  // Save vector components of the net intensity
               // Cannot use TABS because that is cumulative over frequency...
               atomicAdd_g_f(&(INT[oind]),  delta      ) ;
               atomicAdd_g_f(&(INTX[oind]), delta*DIR.x) ;
               atomicAdd_g_f(&(INTY[oind]), delta*DIR.y) ;
               atomicAdd_g_f(&(INTZ[oind]), delta*DIR.z) ;
# endif
               dx             =  ldexp(dx, level0) ;
               dx             =  max(0.0f, dx-2.0f*PEPS) ; // must remain in cell (level0, ind0)
               POS            =  POS0 + dx*DIR ;  // location of scattering -- coordinates of current level
               PHOTONS       *=  exp(-tauA) ;
               
# if (STEP_WEIGHT<=0)
               // normal, unweighted case
               free_path  = -log(Rand(&rng)) ;
# endif
# if (STEP_WEIGHT==1)   
               // Single exponential, free path  *=  1/SW_A,   use 0<SW_A<1
               //   p   =  A*exp(-A*tau)
               //   tau = -log(u/A)/A
               //   w   =  exp(A*tau-tau)/A
               //  SW_A = multiplier for the argument of the exponential function
               free_path   = -log(Rand(&rng)) / SW_A ;
               PHOTONS    *=  exp(SW_A*free_path-free_path) / SW_A ;
# endif
# if (STEP_WEIGHT==2)
               // Two exponentials, B and 2*B, e.g., B=0.5 == exp(-tau) & exp(-0.5*tau)
               //   p    =  0.5*a*exp(-a*t) + a*exp(-2*a*t)
               //   tau  =  -log(-0.5+sqrt(0.25+2*u)) / a
               //   w    =  1 / (  0.5*a*exp((1-a)*t) + a*exp((1-2*a)*t) )
               // SW_A = Weight of modified exponential term
               // SW_B = Multiplier for the argument of the exponential function
               free_path =      -log( 
                                     (-SW_B+sqrt(SW_B*SW_B+4.0f*Rand(&rng)*(1.0f-SW_B))) / 
                                     (2.0f-2.0f*SW_B) 
                                    ) / SW_A ;
               PHOTONS  *= 1.0f /     ( SW_A*SW_B*exp((1.0f-SW_A)*free_path) +  2.0f*SW_A*(1.0f-SW_B)*exp((1.0f-2.0f*SW_A)*free_path) ) ;
# endif
               // return to original indices
               ind            =  ind0 ;           // cell has not changed !?
               level          =  level0 ;         // (ind, level) at the beginning of the step = at the end
               
# if (DIR_WEIGHT>0)
               POS0 = DIR ;  // old direction
               WScatter(&DIR, CSC, &rng, &pweight) ;
               // Yet another weighting = ratio of scattering functions
               // ... for the scattering angle of the photon package (not peeloff direction!!)
               tau           =  DIR.x*POS0.x+DIR.y*POS0.y+DIR.z*POS0.z ;  // ct=="tau" reused         
               pind          =  clamp((int)(BINS*(1.0f+tau)*0.5f), 0, BINS-1) ;
               PHOTONS      *=  DSC[pind] / pweight ;
# else
#  if (WITH_MSF==0) // ----------------------------------------------------------------------------------------------
               // Basic situation, only single scattering function in use
               Scatter(&DIR, CSC, &rng) ;   // new direction
#  else // ----------------------------------------------------------------------------------------------------------
               // We must select the scatterer -- using the properties of the cell with global index oind
               // and the relative values of ABU[idust+NDUST*oind]*SCA[idust] / OPT[2*oind+1]
               //  *** re-using ds, free_path, ind0 ***
               free_path =  OPT[2*oind+1] ;        // sum(ABU*SCA) for the current cell
               ds        =  Rand(&rng) ;
               for(ind0=0; ind0<NDUST; ind0++) {   // ind0 ~ dust index
                  ds -= ABU[ind0+NDUST*oind]*SCA[ind0] / OPT[2*oind+1] ;
                  if (ds<=0.0) break ;
               }
               if (ind0>=NDUST) printf("?????\n") ;
               Scatter(&DIR, &CSC[ind0*BINS], &rng) ; // use the scattering function of the ind0:th dust species
#  endif // --------------------------------------------------------------------------------------------------------
# endif
               tau = 0.0f ;
            } // end of -- scattering
            
            
         } // while (ind>0) -- loop until the end of ray before checking if new are required
         
         
      } // for --- loop over three packages from current cell
      
   } // while() --- loop to next emitting cell
   
}


#endif
