#include "kernel_LOC_aux.c"
// #define GAUSTORE  __global

// this options would write "spectra" with just sum(dx*rho*profile),
// mass-weighted sum of LOS profile functions
#define MASS_WEIGHTED_VELOCITY_PROFILE 0

#define GID   24
#define SPLIT_UPSTREAM_ONLY 0
#define SAFE  0           // SAFE>0 => add some warning messages

// One might think that "NO_ATOMICS 1" is faster (allowed by ONESHOT=0)
// but in practice it was slower... (?)
#define NO_ATOMICS 0


// single precision, instead of  1.0f-exp(-t)  use
//     |t|<0.005   =>    t*(1.0f-0.5f*t)   .... or
//     |t|<0.02    =>    t*(1.0f-t*(0.5f-0.1666666667f*t))

// single precision, instead of (1-exp(-t))/t  use
//     |t|<0.005   =>       1.0f-0.5f*t
//     |t|<0.01    =>       1.0f-t*(0.5f-0.1666666667f*t)



__kernel void Clear(__global float *RES) {   // [CELLS,1-2] for  SIJ or both SIJ and ESC
   // Clear the arrays before simulation of the current transition
   int id = get_global_id(0) ;   
#if (WITH_ALI>0)
   for(int i=id; i<2*CELLS; i+=GLOBAL)  RES[i] = 0.0f ;
#else
   for(int i=id; i<  CELLS; i+=GLOBAL)  RES[i] = 0.0f ;
#endif
}



int Index(const REAL3 pos) {
   if ((pos.x<=ZERO)||(pos.y<=ZERO)||(pos.z<=ZERO))  return -1 ;
   if ((pos.x>=  NX)||(pos.y>=  NY)||(pos.z>=  NZ))  return -1 ;
   // without (int), this FAILS for cloud sizes >256^3 !!
   return  (int)trunc(pos.x) + NX*((int)trunc(pos.y)+NY*(int)trunc(pos.z)) ;
}





__kernel void Spectra(
#if (WITH_HALF==1)
                      __global half     *CLOUD,        //  0 [CELLS,4] ==  vx, vy, vz, sigma
#else
                      __global float4   *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
#endif
                      GAUSTORE   float  *GAU,          //  1 precalculated gaussian profiles
                      constant int2     *LIM,          //  2 limits of ~zero profile function
                      const float        GN,           //  3 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                      const float2       D,            //  4 ray direction == theta, phi
                      __global float    *NI,           //  5 [CELLS]:  NI[upper] + NB_NB
                      const float        DE,           //  6 grid units, single offset
                      const int          NRA,          //  7 number of RA points = work items
                      const float        STEP,         //  8 step between spectra (grid units)
                      const float        BG,           //  9 background intensity
                      const float        emis0,        // 10 h/(4pi)*freq*Aul*int2temp
                      __global float    *NTRUE_ARRAY,  // 11 NRA*CHANNELS -- spectrum
                      __global float    *SUM_TAU_ARRAY,// 12 NRA*CHANNELS -- optical depths
#if (WITH_CRT>0)
                      constant float    *CRT_TAU,      // 13 dust optical depth / GL
                      constant  float   *CRT_EMI,      // 14 dust emission photons/c/channel/H
#endif
#if (WITH_OCTREE>0)
                      __global  int     *LCELLS,       // 13, 15
                      __constant int    *OFF,          // 14, 16
                      __global   int    *PAR,          // 15, 17
                      __global   float  *RHO,          // 16, 18
#endif
                      const int FOLLOW,                // 17, 19
                      const float3    CENTRE           // 18, 20  map centre in current
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
   float RA, maxi ;       // grid units, offset of current ray
   // DE  =   (id-0.5f*(NDEC-1.0f))*STEP ;
   // RA  =   -(id-0.5f*(NRA-1.0f))*STEP ;   // spectra from left to right = towards -RA
   RA  =   id  ;
   // calculate the initial position of the ray
   REAL3   POS, dr, RPOS ;
   float3  DIR ;
   REAL    dx, dy, dz ;
   DIR.x   =   sin(D.x)*cos(D.y) ;     // D.x = theta,   D.y = phi
   DIR.y   =   sin(D.x)*sin(D.y) ;
   DIR.z   =   cos(D.x)            ;
   REAL3 RV, DV ; 
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
   
   // Offsets in RA and DE directions, (RA, DE) are just indices [0, NRA[, [0,NDE[
   // CENTRE are the indices for the map centre using the current pixel size
   // POS is already at the map centre
   POS.x  =  CENTRE.x + (RA-0.5*(NRA-1.0f))*STEP*RV.x + DE*STEP*DV.x ;
   POS.y  =  CENTRE.y + (RA-0.5*(NRA-1.0f))*STEP*RV.y + DE*STEP*DV.y ;
   POS.z  =  CENTRE.z + (RA-0.5*(NRA-1.0f))*STEP*RV.z + DE*STEP*DV.z ;
   
   
   // int ID = ((fabs(POS.y-1.5)<0.02)&&(fabs(POS.z-0.7)<0.02)) ? id : -1 ;
   int ID = ((fabs(POS.y-2.0)<0.05)&&(fabs(POS.z-1.5)<0.02)) ? id : -1 ;
   
   
   
   // printf("%6.2f %6.2f  %8.4f %8.4f %8.4f    %8.4f %8.4f %8.4f\n", RA, DE, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z) ;
   // if (id==ID) printf("%6.2f %6.2f   %8.4f %8.4f %8.4f\n", RA, DE, POS.x, POS.y, POS.z) ;
   
   // Change DIR to direction away from the observer
   DIR *= -1.0f ;
   
   if (fabs(DIR.x)<1.0e-10f) DIR.x = 1.0e-10f ;
   if (fabs(DIR.y)<1.0e-10f) DIR.y = 1.0e-10f ;
   if (fabs(DIR.z)<1.0e-10f) DIR.z = 1.0e-10f ;
   
   // go to front surface, first far enough upstream (towards observer), then step forward to cloud (if ray hits)
   POS.x -= 1000.0*DIR.x ;  POS.y -= 1000.0*DIR.y ;  POS.z -= 1000.0*DIR.z ;
   if (DIR.x>ZERO)  dx = (ZERO-POS.x)/DIR.x ;
   else             dx = (NX  -POS.x)/DIR.x ;
   if (DIR.y>ZERO)  dy = (ZERO-POS.y)/DIR.y ;
   else             dy = (NY  -POS.y)/DIR.y ;
   if (DIR.z>ZERO)  dz = (ZERO-POS.z)/DIR.z ;
   else             dz = (NZ  -POS.z)/DIR.z ;
   dx      =  max(dx, max(dy, dz)) + 1.0e-4f ;  // max because we are outside
   POS.x  +=  dx*DIR.x ;   POS.y  +=  dx*DIR.y ;   POS.z  +=  dx*DIR.z ;   // even for OT, still in root grid units
   
   
   
   
   int level0 ;
#if (MAP_INTERPOLATION>0)
   int   slevel, sind, ind0, ind ;
   float A1, A2,  B1, B2,  a, b, K, dopA, dopB ;
   float3 ADIR, BDIR ;
   REAL3  MPOS, POS0 ;
   if (fabs(DIR.x)>fabs(DIR.y)) {
      if (fabs(DIR.z)>fabs(DIR.x)) {
         ADIR.x=0.0005f ; ADIR.y=1.0f ; ADIR.z=-DIR.y/DIR.z ;
      } else {
         ADIR.x=-DIR.z/DIR.x ; ADIR.y=0.0005f ; ADIR.z=1.0f ;
      }
   } else {
      if (fabs(DIR.z)>fabs(DIR.y)) {
         ADIR.x=0.0005f ; ADIR.y=1.0f ; ADIR.z=-DIR.y/DIR.z ;
      } else {
         ADIR.x=1.0f ; ADIR.y=-DIR.x/DIR.y ; ADIR.z=0.0005f ;
      }
   }
   ADIR   = normalize(ADIR) ;
   BDIR.x = DIR.y*ADIR.z-DIR.z*ADIR.y ;
   BDIR.y = DIR.z*ADIR.x-DIR.x*ADIR.z ;
   BDIR.z = DIR.x*ADIR.y-DIR.y*ADIR.x ;
   BDIR   = normalize(BDIR) ;
# if (MAP_INTERPOLATION==2)
   float LA, LB ;  // cell sizes relative to root grid cells
# endif
# if (MAP_INTERPOLATION==3)
   float C1, C2, LA, LB, LC, dopC, w0, wA, wB, wC, c ;
   // float3  XDIR ;   XDIR = ADIR ; ADIR = BDIR ; BDIR = XDIR ;
# endif
# if 0
   printf("DIR:   %8.4f %8.4f %8.4f\n", DIR.x,  DIR.y,  DIR.z) ; 
   printf("ADIR:  %8.4f %8.4f %8.4f\n", ADIR.x, ADIR.y, ADIR.z) ;
   printf("BDIR:  %8.4f %8.4f %8.4f\n", BDIR.x, BDIR.y, BDIR.z) ;
# endif
#endif
   
   
   
   
#if (WITH_OCTREE>0)
   int OTL, OTI, INDEX ;
   // Note: for OCTREE=5,  input is [0,NX], output coordinates are [0,1] for root-grid cells
   IndexG(&POS, &OTL, &OTI, RHO, OFF) ;
   INDEX =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;
#else
   int INDEX   =  Index(POS) ;
#endif
   
   for(int i=0; i<CHANNELS; i++) {
      NTRUE[i]   = 0.0f ;
      SUM_TAU[i] = 0.0f ;
   }
   
   float tau, dtau, emissivity, doppler, nu, sst ;
   int row, shift, c1, c2  ;
   GAUSTORE float* profile ;
#if (WITH_CRT>0)
   float Ctau, Cemit, pro, distance=0.0f ;
#endif
   
   float distance = 0.0f ;
   
   
   
   
   while (INDEX>=0) {
      
#if (MAP_INTERPOLATION>0)
      POS0 = POS ;    level0 = OTL ;   ind0 = OTI ;     K = ldexp(1.0f, -level0) ;
#endif
      
      
#if (WITH_OCTREE>0)   // INDEX  =  OFF[OTL] + OTI ;  --- update INDEX at the end of the step
      // OCTREE==5 uses level=0 coordinates POS=[0,1], not [0,NX]
      dx        =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, 99, NULL, -1) ; // updates POS, OTL, OTI
      // RPOS = POS ; RootPos(&RPOS, OTL, OTI, OFF, PAR) ;
#else
      if (DIR.x<0.0f)   dx = -     fmod(POS.x,ONE)  / DIR.x - EPS/DIR.x;
      else              dx =  (ONE-fmod(POS.x,ONE)) / DIR.x + EPS/DIR.x;
      if (DIR.y<0.0f)   dy = -     fmod(POS.y,ONE)  / DIR.y - EPS/DIR.y;
      else              dy =  (ONE-fmod(POS.y,ONE)) / DIR.y + EPS/DIR.y;
      if (DIR.z<0.0f)   dz = -     fmod(POS.z,ONE)  / DIR.z - EPS/DIR.z;
      else              dz =  (ONE-fmod(POS.z,ONE)) / DIR.z + EPS/DIR.z;
      dx         =  min(dx, min(dy, dz)) + EPS ;      // actual step
#endif
      
      
      
      
      
      
      
#if (MAP_INTERPOLATION>0)  // @@ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      
# if (MAP_INTERPOLATION==1) 
      
#  if 1  // little to no effect from substepping ???
      if (dx>(0.13*K)) {       // limit step size  ---- K = current cell size in GL units, K <= 1
         dx    =  0.13*K ;     // step in GL units
         OTL   =  level0 ;   OTI = ind0 ;   
         POS.x =  POS0.x + 0.13f*DIR.x ;  // step in current POS coordinates, in old cell
         POS.y =  POS0.y + 0.13f*DIR.y ; 
         POS.z =  POS0.z + 0.13f*DIR.z ; 
      }
#  endif
      doppler =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
      
      slevel  =  level0 ;   sind = ind0 ;  
      MPOS.x  =  POS0.x+(0.49f*dx/K)*DIR.x ;  MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ;  MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
      a       =  GetStepOT(&MPOS, &ADIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1) / K ;
      if ((a>0.0f)&&(a<=0.5f)&&(sind>=0)) {
         ind   =  OFF[slevel] + sind ;         A1 = NI[2*ind] ;    A2 = NI[2*ind+1] ;
         dopA  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;         
      } else {
         slevel =  level0 ;    sind = ind0 ;    ADIR *= -1.0f ;   
         MPOS.x =  POS0.x+(0.49f*dx/K)*DIR.x ;   MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ;   MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
         a      =  GetStepOT(&MPOS, &ADIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1) / K ;
         if ((a>0.0f)&&(a<=0.5f)&&(sind>=0)) {
            ind   =  OFF[slevel] + sind ;      A1 =  NI[2*ind] ;   A2 = NI[2*ind+1] ;
            dopA  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
         } else {
            a = 0.5f ;   A1 = 0.0f ;   A2 = 0.0f ;  dopA = 0.0f ;
         }
      }

      slevel = level0 ;   sind = ind0 ;
      MPOS.x = POS0.x+(0.49f*dx/K)*DIR.x ; MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ;  MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
      b      =  GetStepOT(&MPOS, &BDIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1) / K ;      
      if ((b>0.0f)&&(b<=0.51f)&&(sind>=0)) {
         ind   =  OFF[slevel] + sind ;         B1 =  NI[2*ind] ;   B2 = NI[2*ind+1] ;
         dopB  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;     
      } else {
         slevel =  level0 ;    sind = ind0 ;    BDIR *= -1.0f ;   
         MPOS.x = POS0.x+(0.49f*dx/K)*DIR.x ; MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ; MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
         b      =  GetStepOT(&MPOS, &BDIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1) / K ;
         if ((b>0.0f)&&(b<=0.51f)&&(sind>=0)) {
            ind   =  OFF[slevel] + sind ;      B1 =  NI[2*ind] ;   B2 = NI[2*ind+1] ;
            dopB  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
         } else {
            b = 0.5f ;   B1 = 0.0f ;   B2 = 0.0f ;  dopB = 0.0f ;
         }
      }
            
      // if (slevel<level0) b = 0.5f*(b+0.5f) ;  // if b was in larger cell, it should get smaller weight?
      a   =  0.5f-a ;      b = 0.5f-b ;
#  if 0
      a   =  0.6f*a ;      b *= 0.6f ;
#  endif
      doppler =  (1.0f-a-b)*doppler       + a*dopA + b*dopB ;
      // if (id==ID) printf("   %6.3f %6.3f   %6.3f\n", a, b, 1.0f-a-b) ; 
      nu      =  (1.0f-a-b)*NI[2*INDEX  ]  +  a*A1   +  b*B1 ;   // INDEX = still index of the step start
      A2      =  (1.0f-a-b)*NI[2*INDEX+1]  +  a*A2   +  b*B2 ;
      tau     =  (fabs(A2)<1.0e-30f) ? (dx*1.0e-30f*GN*GL) : clamp((float)(dx*A2*GN*GL), -2.0f, 1.0e10f) ;
      
      
# endif  // MAP_INTERPOLATION==1
      
      
      
      
      
      
      
# if (MAP_INTERPOLATION==2) // 22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
      //  (X1, Y1)  == (0.0, 0.0) ==  centre of current cell
      //  (X2, Y2)  =  (0.0, 0.5*(K+LA))   centre of the next cell in direction ADIR
      //  (X3, Y3)  =  (0.5*(K+LB), 0.0)   centre of the next cell in direction BDIR
      //
      //     W1 =  (2*b+LB)/(K+LB) + (2*a-L0)/(K+LA)
      //     W2 =  (K-2*a)/(K+LA)
      //     W3 =  1-W1-W2
      // L0==K, LA, LB  are the cell sizes in root grid length units
      // a, b are the step lengths to cell boundary, also in root grid length units
      doppler  =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
      slevel   =  level0 ;   sind = ind0 ;  
      MPOS.x   =  POS0.x+(0.49f*dx/K)*DIR.x ;  MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ;  MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
      // dx    =  GetStepOT( &POS,  &DIR, &OTL,    &OTI,  RHO, OFF, PAR, 99, NULL, -1) ;
      a        =  GetStepOT(&MPOS, &ADIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1) ;  // [GL]
      if ((a<=(0.5f*K))&&(sind>=0)) {
         ind   =  OFF[slevel] + sind ;          A1 = NI[2*ind] ;    A2 = NI[2*ind+1] ;
         dopA  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
         LA    =  ldexp(1.0f, -slevel) ;
      } else {
         slevel   =  level0 ;    sind = ind0 ;     ADIR *= -1.0f ;   
         MPOS.x   =  POS0.x+(0.49f*dx/K)*DIR.x ;   MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ;   MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
         a        =  GetStepOT(&MPOS, &ADIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1) ;
         if ((a<=(0.5f*K))&&(sind>=0)) {
            ind   =  OFF[slevel] + sind ;       A1 =  NI[2*ind] ;   A2 = NI[2*ind+1] ;
            dopA  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
            LA    =  ldexp(1.0f, -slevel) ;
         } else {
            a = 0.5f*K ;   A1 = NI[2*INDEX] ;   A2 = NI[2*INDEX+1] ;  dopA = doppler ;
         }
      }
      slevel   = level0 ;    sind = ind0 ;
      MPOS.x   = POS0.x+(0.49f*dx/K)*DIR.x ; MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ;  MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
      b        =  GetStepOT(&MPOS, &BDIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1)  ; // [GL]
      if ((b<=(0.5f*K))&&(sind>=0)) {
         ind   =  OFF[slevel] + sind ;         B1 =  NI[2*ind] ;   B2 = NI[2*ind+1] ;
         dopB  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
         LB    =  ldexp(1.0f, -slevel) ;
         // COMMENT OUT THE FOLLOWING PRINTF EVERYTHING WILL BE NAN ?????????? ONLY ON GPU ....
         // if (id==ID) printf("    B: %02d %6d ", slevel, sind) ;
      } else {
         slevel =  level0 ;    sind = ind0 ;    BDIR *= -1.0f ;   
         MPOS.x = POS0.x+(0.49f*dx/K)*DIR.x ; MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ; MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
         b      =  GetStepOT(&MPOS, &BDIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1) ;
         if ((b<=(0.5f*K))&&(sind>=0)) {
            ind   =  OFF[slevel] + sind ;      B1 =  NI[2*ind] ;   B2 = NI[2*ind+1] ;
            dopB  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
            LB    =  ldexp(1.0f, -slevel) ;
            // if (id==ID) printf("    B: %02d %6d ", slevel, sind) ;                     
         } else {
            b = 0.5f*K ;   B1 = NI[2*INDEX] ;   B2 = NI[2*INDEX+1] ;  dopB = doppler ;
            // if (id==ID) printf("    B: %02d %6d ", -1, -1) ;
         }
      }
      b       =   (2.0f*b+LB)/(K+LB) + (2.0f*a-K)/(K+LA) ;   // b = W1 ~ Centre

      a       =   (K-2.0f*a)/(K+LA) ;                        // a = W2 ~ A,   (1-a-b) ~ B
      doppler =   b*doppler      + a*dopA + (1.0f-a-b)*dopB ;
      nu      =   b*NI[2*INDEX  ]  +  a*A1   +  (1.0f-a-b)*B1 ;   // INDEX = still index of the step start
      A2      =   b*NI[2*INDEX+1]  +  a*A2   +  (1.0f-a-b)*B2 ;
      
      tau     =  (fabs(A2)<1.0e-30f) ? (dx*1.0e-30f*GN*GL) : clamp((float)(dx*A2*GN*GL), -2.0f, 1.0e10f) ;
      // if (id==ID) printf("   W: %6.3f %6.3f %6.3f   nu %10.3e  tau %10.3e\n", b, a, 1.0f-a-b, nu, tau) ; // centre, ADIR, BDIR  = this, A, B      

      
      
# endif // MAP_INTERPOLATION==2 22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
      

      
      

      
# if (MAP_INTERPOLATION==3) // four point interpolation
#  if 0
      if (dx>(0.14*K)) {       
         dx    =  0.14*K ;                // step in GL units
         OTL   =  level0 ;   OTI = ind0 ;   
         POS.x =  POS0.x + 0.14f*DIR.x ;  // step in current POS coordinates, in old cell
         POS.y =  POS0.y + 0.14f*DIR.y ; 
         POS.z =  POS0.z + 0.14f*DIR.z ; 
      }
#  endif
      dopC = C1 = C2 = 0.0f ;
      LC   = -1.0f ;
      //  original cell + ADIR + BDIR + (NDIR+ADIR)
      doppler  =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
      // ADIR
      slevel   =  level0 ;   sind = ind0 ;  
      MPOS.x   =  POS0.x+(0.49f*dx/K)*DIR.x ;  MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ;  MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
      a        =  GetStepOT(&MPOS, &ADIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1) ;
      if ((a<=(0.5f*K))&&(sind>=0)) {
         ind   =  OFF[slevel] + sind ;          A1 = NI[2*ind] ;    A2 = NI[2*ind+1] ;
         dopA  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
         LA    =  ldexp(1.0f, -slevel) ;
      } else {
         slevel   =  level0 ;    sind = ind0 ;     ADIR *= -1.0f ;   
         MPOS.x   =  POS0.x+(0.49f*dx/K)*DIR.x ;   MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ;   MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
         a        =  GetStepOT(&MPOS, &ADIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1) ;
         if ((a<=(0.5f*K))&&(sind>=0)) {
            ind   =  OFF[slevel] + sind ;       A1 =  NI[2*ind] ;   A2 = NI[2*ind+1] ;
            dopA  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
            LA    =  ldexp(1.0f, -slevel) ;
         } else {
            a = 0.5f*K ;   A1 = NI[2*INDEX] ;   A2 = NI[2*INDEX+1] ;  dopA = doppler ;
         }
      }
      // BDIR
      slevel   = level0 ;    sind = ind0 ;
      MPOS.x   = POS0.x+(0.49f*dx/K)*DIR.x ; MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ;  MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
      b        =  GetStepOT(&MPOS, &BDIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1)  ;
      if ((b<=(0.5f*K))&&(sind>=0)) {
         ind   =  OFF[slevel] + sind ;          B1 =  NI[2*ind] ;   B2 = NI[2*ind+1] ;
         dopB  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
         LB    =  ldexp(1.0f, -slevel) ;
         // BDIR+ADIR
         MPOS.x += 0.01f*BDIR.x ;  MPOS.y += 0.01f*BDIR.y ;  MPOS.z += 0.01f*BDIR.z ; // further from the border
         c     =  GetStepOT(&MPOS, &ADIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1)  ;
         if (sind>=0) {
            ind   =  OFF[slevel] + sind ;       C1 =  NI[2*ind] ;   C2 = NI[2*ind+1] ;
            dopC  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
            LC    =  ldexp(1.0f, -slevel) ;
         }
      } else {
         slevel =  level0 ;    sind = ind0 ;    BDIR *= -1.0f ;   
         MPOS.x = POS0.x+(0.49f*dx/K)*DIR.x ; MPOS.y = POS0.y+(0.49f*dx/K)*DIR.y ; MPOS.z = POS0.z+(0.49f*dx/K)*DIR.z ;
         b      =  GetStepOT(&MPOS, &BDIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1) ;
         if ((b<=(0.5f*K))&&(sind>=0)) {
            ind   =  OFF[slevel] + sind ;       B1 =  NI[2*ind] ;   B2 = NI[2*ind+1] ;
            dopB  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
            LB    =  ldexp(1.0f, -slevel) ;
            // BDIR+ADIR
            MPOS.x += 0.01f*BDIR.x ;  MPOS.y += 0.01f*BDIR.y ;  MPOS.z += 0.01f*BDIR.z ; // further from the border
            c     =  GetStepOT(&MPOS, &ADIR, &slevel, &sind, RHO, OFF, PAR, 99, NULL, -1)  ;
            if (sind>=0) {
               ind   =  OFF[slevel] + sind ;    C1 =  NI[2*ind] ;   C2 = NI[2*ind+1] ;
               dopC  =  CLOUD[ind].x*DIR.x + CLOUD[ind].y*DIR.y + CLOUD[ind].z*DIR.z ;
               LC    =  ldexp(1.0f, -slevel) ;
            }
         } else {
            b = 10.0f*K ;   B1 = NI[2*INDEX] ;   B2 = NI[2*INDEX+1] ;  dopB = doppler ;
         }
      }

            
      
#  define FUN(x) (sqrt(x))
      
      // distance (0.5*K-a ,  0.5*K-b) for original cell = (doppler, NI[2*INDEX...]
      // distance (a+0.5*LA,      0.0) for cell (dopA, A1, A2)
      // distance (     0.0, b+0.5*LB) for cell (dopB, B1, B2)
      // distance (b+0.5*L4, c+0.5*L4) for cell (dopC, C1, C2)
      w0      =   1.0f / FUN(   (0.02f*K)*(0.02f*K) + (0.5f*K-a)*(0.5f*K-a) + (0.5f*K-b)*(0.5f*K-b)  ) ;
      
      if (b>0.3f*K)   b *= 1.0f + (b-0.3f)*3.0f ;
      
      wA      =   1.0f / FUN(   (a+0.5*LA)*(a+0.5*LA) + (0.5f*LA-b)*(0.5f*LA-b)   ) ;
      wB      =   1.0f / FUN(   (b+0.5*LB)*(b+0.5*LB) + (0.5f*LB-a)*(0.5f*LB-a)   ) ;
      wC      =   (LC>0.0f)  ?  FUN(1.0f/((b+0.5*LC)*(b+0.5*LC) + (c+0.5*LC)*(c+0.5*LC))) : 0.0f ;
      
      // wA= wC = 0.0f ;
      
      c       =   1.0f/(w0+wA+wB+wC) ;
      
      if (id==100)  printf("%10.3e %10.3e %10.3e   %.2f %.2f %.2f %.2f   %10.3e %10.3e %10.3e %10.3e\n", a, b, c,   K, LA, LB, LC,   w0, wA, wB, wC) ;
      
      doppler =   c*(w0*doppler         +  wA*dopA + wB*dopB  + wC*dopC) ;
      nu      =   c*(w0*NI[2*INDEX  ]   +  wA*A1   + wB*B1    + wC*C1  ) ;
      A2      =   c*(w0*NI[2*INDEX+1]   +  wA*A2   + wB*B2    + wC*C2  ) ;
      
      tau     =  (fabs(A2)<1.0e-30f) ? (dx*1.0e-30f*GN*GL) : clamp((float)(dx*A2*GN*GL), -2.0f, 1.0e10f) ;

           
# endif // MAP_INTERPOLATION==3
      
      
      
      
      
#else //  -- no interpolation --- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
# if (WITH_HALF==0)
      doppler    =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# else
      doppler    =  
        vload_half(0,&(CLOUD[4*INDEX]))*DIR.x + 
        vload_half(1,&(CLOUD[4*INDEX]))*DIR.y + 
        vload_half(2,&(CLOUD[4*INDEX]))*DIR.z ;
# endif
      nu         =       NI[2*INDEX] ;           
# if 0
      tau        =  dx * NI[2*INDEX+1]*GN*GL ;        // need to separate GN and GL ?
# else
      // if (fabs(NI[INDEX].y)<1.0e-24f) {
      //    tau     =  dx * 1.0e-24f * GN * GL ;
      // } else {
      //    tau     =  dx * NI[INDEX].y * GN * GL ;         
      //    // Comment this out to make results consistent with Cppsimu!!
      //    // tau = clamp(tau, -0.05f, 1.0e10f) ;  // CLIP MASERS ???
      //    // >-0.5f clear effect in test case [10,2000]K, [10,1e5]cm-3
      //    //        ... although only in extreme parameter combinations
      //    // >-1.0f already ~identical to Cppsimu
      //    // The limit will depend on cell size and density... is >-2.0f a safe option??
      //    tau = clamp(tau, -2.0f, 1.0e10f) ;
      // }
      tau =  (fabs(NI[2*INDEX+1])<1.0e-30f) ? (dx*1.0e-30f*GN*GL) : clamp((float)(dx*NI[2*INDEX+1]*GN*GL), -2.0f, 1.0e10f) ;
# endif       
      
#endif // --- no interpolation --- %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      
      
      distance  += dx ;                  
      
#if 1      
      tau        =  clamp(tau, 1.0e-30f, 1.0e10f) ;  // $$$  KILL ALL MASERS
#endif
      
      
      
#if (WITH_HALF==0)
      row        =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
#else
      row        =  clamp((int)round(log(vload_half(3,&(CLOUD[4*INDEX]))/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
#endif
      
      profile    =  &GAU[row*CHANNELS] ;
      shift      =  round(doppler/WIDTH) ;
      c1         =  LIM[row].x+shift ;
      c2         =  LIM[row].y+shift ;
      c1         =  max(c1,   max(shift,      0               )) ;
      c2         =  min(c2,   min(CHANNELS-1, CHANNELS-1+shift)) ;      
      emissivity =  emis0 * nu * dx * GL ;    // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      
      // printf("Doppler %.3e,  sigma %.3e, row %d/%d\n", doppler, CLOUD[INDEX].w, row, GNO) ;
      
      
#if 1
      if (level0<MINMAPLEVEL) {
         tau = 0.0f ; emissivity = 0.0f ;   // ignore cells below MINMAPLEVEL
      }
#endif
      
      
#if (MASS_WEIGHTED_VELOCITY_PROFILE>0)
      // Instead of spectra, compute direct mass-weighted LOS velocity profile
      for(i=0; i<CHANNELS; i++) {
         NTRUE[i] += dx*RHO[INDEX]*profile[clamp(i-shift, 0, CHANNELS-1)] ;
      }
#else
#if (WITH_CRT>0)
      Ctau       =  CRT_TAU[INDEX] * dx      ;
      Cemit      =  CRT_EMI[INDEX] * dx * GL ;
      for(i=0; i<CHANNELS; i++) {
         pro         =  profile[clamp(i-shift, 0, CHANNELS-1)] ;
         dtau        =  tau*pro + Ctau ;
         NTRUE[i]   += (emissivity*pro*GN + Cemit)*exp(-SUM_TAU[i])* 
           (  (fabs(dtau)>0.01f)  ? ((1.0f-exp(-dtau))/dtau)  :  (1.0f-dtau*(0.5f-0.166666667f*dtau))  ) ;
         SUM_TAU[i] +=  dtau  ;
      }     
#else
      for(i=c1; i<=c2; i++) {         
         dtau        = tau*profile[i-shift] ;
         NTRUE[i]   += emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i])* 
           (  (fabs(dtau)>0.01f)  ?  ((1.0f-exp(-dtau))/dtau)  :  (1.0f-dtau*(0.5f-0.166666667f*dtau))) ;
         SUM_TAU[i] += dtau  ;
      }
#endif
#endif 
      
      
#if (WITH_OCTREE>0)
      INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;
#else
      POS.x  += dx*DIR.x ;  POS.y  += dx*DIR.y ;  POS.z  += dx*DIR.z ;
      INDEX = Index(POS) ;         
#endif
   } // while INDEX>=0
   
   
#if (MASS_WEIGHTED_VELOCITY_PROFILE<1)
#if 1
   for (i=0; i<CHANNELS; i++) {
      // NTRUE[i] -=  BG*(1.0f-exp(-SUM_TAU[i])) ;
      // optical depths 1e-10 or below => final NTRUE may be negative???
      tau    =   SUM_TAU[i] ;
      dtau   =   NTRUE[i] ;
      //  tau*(1-0.5*tau*(1-0.333333*tau)) better for |tau|<0.02
      NTRUE[i] -=  BG * ((fabs(tau)>0.01f ) ?  (1.0f-exp(-tau))  :  (tau*(1.0f-tau*(0.5f-0.166666667f*tau)))) ;
#if 0
      if (NTRUE[i]<-1.0e-6) {
         printf("%12.4e -> %12.4e,   BG=%12.4e  x  %12.4e, SUM_TAU=%.3e\n", dtau, NTRUE[i], BG,
                ((fabs(tau)>0.005f) ? (1.0f-exp(-tau)) : (tau*(1.0f-tau*(0.5f-0.166666667f*tau)))), tau) ;
      }
#endif
   } // for i over CHANNELS
#else  // WITHOUT BACKGROUND -- ONLY FOR TESTING
   if (id==0) printf("x?") ;
#endif
#endif
   
   
}






#if (OCTREE>0) // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Cartesian grid not implemented - would need to pass density as another parameter

__kernel void Spectra_vs_LOS(                             
#if (WITH_HALF==1)
                             __global half     *CLOUD,        //  0 [CELLS,4] ==  vx, vy, vz, sigma
#else
                             __global float4   *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
#endif
                             GAUSTORE   float  *GAU,          //  1 precalculated gaussian profiles
                             constant int2     *LIM,          //  2 limits of ~zero profile function
                             const float        GN,           //  3 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                             const float2       D,            //  4 ray direction == theta, phi
                             __global float    *NI,           //  5 [CELLS]:  NI[upper] + NB_NB
                             const float        DE,           //  6 grid units, single DE offset
                             const int          NRA,          //  7 grid units, single RA offset
                             const float        STEP,         //  8 step between spectra (grid units)
                             const float        BG,           //  9 background intensity
                             const float        emis0,        // 10 h/(4pi)*freq*Aul*int2temp
                             __global float    *NTRUE_ARRAY,  // 11 nra*CHANNELS -- spectrum
                             __global float    *SUM_TAU_ARRAY,// 12 nde*CHANNELS -- optical depths
#if (WITH_CRT>0)
                             constant float    *CRT_TAU,      // 13 dust optical depth / GL
                             constant  float   *CRT_EMI,      // 14 dust emission photons/c/channel/H
#endif
#if (WITH_OCTREE>0)
                             __global  int     *LCELLS,       // 13, 15
                             __constant int    *OFF,          // 14, 16
                             __global   int    *PAR,          // 15, 17
                             __global   float  *RHO,          // 16, 18
#endif
                             const int FOLLOW,                // 17, 19
                             const float3       CENTRE,       // 18, 20  map centre in current
                             __global float    *TKIN,         // TKIN[CELLS]
                             __global float    *ABU,          // ABU[CELLS]
                             __global float    *LOS_EMIT,     // 19, 21 LOS_EMIT[maxsteps, CHANNELS]
                             __global float    *LENGTH,       // 20, 22 LENGTH[maxsteps], step of each individual step
                             __global float    *LOS_RHO,
                             __global float    *LOS_TKIN,
                             __global float    *LOS_ABU
                            )
{
   // single work item, single LOS
   //   ra  =  RA (grid units, from cloud centre)
   //   de  =  DE(id)
   int id = get_global_id(0) ;
   if (id>0) return ;    // a single work item !!
   __global float *NTRUE   = &(NTRUE_ARRAY[  id*CHANNELS]) ;
   __global float *SUM_TAU = &(SUM_TAU_ARRAY[id*CHANNELS]) ;   
   int i, step=0 ;
   float maxi ;       // grid units, offset of current ray
   // calculate the initial position of the ray
   REAL3   POS, dr, RPOS ;
   float3  DIR ;
   REAL    dx, dy, dz ;
   DIR.x   =   sin(D.x)*cos(D.y) ;     // D.x = theta,   D.y = phi
   DIR.y   =   sin(D.x)*sin(D.y) ;
   DIR.z   =   cos(D.x)            ;
   REAL3 RV, DV ; 
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
   
   
   // Offsets in RA and DE directions, (RA, DE) are just indices [0, NRA[, [0,NDE[
   // CENTRE are the indices for the map centre using the current pixel size
   // POS is already at the map centre
#if 0
   POS.x  =  CENTRE.x + (RA-0.5*(NRA-1.0f))*STEP*RV.x + DE*STEP*DV.x ;
   POS.y  =  CENTRE.y + (RA-0.5*(NRA-1.0f))*STEP*RV.y + DE*STEP*DV.y ;
   POS.z  =  CENTRE.z + (RA-0.5*(NRA-1.0f))*STEP*RV.z + DE*STEP*DV.z ;
#else  // a single LOS, towards the map centre
   POS.x  =  CENTRE.x ; 
   POS.y  =  CENTRE.y ;
   POS.z  =  CENTRE.z ;
#endif

   
   // int ID = ((fabs(POS.y-1.5)<0.02)&&(fabs(POS.z-0.7)<0.02)) ? id : -1 ;
   int ID = ((fabs(POS.y-2.0)<0.05)&&(fabs(POS.z-1.5)<0.02)) ? id : -1 ;
   // Change DIR to direction away from the observer
   DIR *= -1.0f ;   
   if (fabs(DIR.x)<1.0e-10f) DIR.x = 1.0e-10f ;
   if (fabs(DIR.y)<1.0e-10f) DIR.y = 1.0e-10f ;
   if (fabs(DIR.z)<1.0e-10f) DIR.z = 1.0e-10f ;   
   // go to front surface, first far enough upstream (towards observer), then step forward to cloud (if ray hits)
   POS.x -= 1000.0*DIR.x ;  POS.y -= 1000.0*DIR.y ;  POS.z -= 1000.0*DIR.z ;
   if (DIR.x>ZERO)  dx = (ZERO-POS.x)/DIR.x ;
   else             dx = (NX  -POS.x)/DIR.x ;
   if (DIR.y>ZERO)  dy = (ZERO-POS.y)/DIR.y ;
   else             dy = (NY  -POS.y)/DIR.y ;
   if (DIR.z>ZERO)  dz = (ZERO-POS.z)/DIR.z ;
   else             dz = (NZ  -POS.z)/DIR.z ;
   dx      =  max(dx, max(dy, dz)) + 1.0e-4f ;  // max because we are outside
   POS.x  +=  dx*DIR.x ;   POS.y  +=  dx*DIR.y ;   POS.z  +=  dx*DIR.z ;   // even for OT, still in root grid units
   
   int level0 ;
   
#if (WITH_OCTREE>0)
   int OTL, OTI, INDEX ;
   // Note: for OCTREE=5,  input is [0,NX], output coordinates are [0,1] for root-grid cells
   IndexG(&POS, &OTL, &OTI, RHO, OFF) ;
   INDEX =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;
#else
   int INDEX   =  Index(POS) ;
#endif
   for(int i=0; i<CHANNELS; i++) {
      NTRUE[i]   = 0.0f ;
      SUM_TAU[i] = 0.0f ;
   }   
   float tau, dtau, emissivity, doppler, nu, sst ;
   int row, shift, c1, c2  ;
   GAUSTORE float* profile ;
#if (WITH_CRT>0)
   float Ctau, Cemit, pro, distance=0.0f ;
#endif   
   float distance = 0.0f ;
   
   
   
   
   while (INDEX>=0) {
      
#if (WITH_OCTREE>0)   // INDEX  =  OFF[OTL] + OTI ;  --- update INDEX at the end of the step
      // OCTREE==5 uses level=0 coordinates POS=[0,1], not [0,NX]
      dx        =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, 99, NULL, -1) ; // updates POS, OTL, OTI
      // RPOS = POS ; RootPos(&RPOS, OTL, OTI, OFF, PAR) ;
#else
      if (DIR.x<0.0f)   dx = -     fmod(POS.x,ONE)  / DIR.x - EPS/DIR.x;
      else              dx =  (ONE-fmod(POS.x,ONE)) / DIR.x + EPS/DIR.x;
      if (DIR.y<0.0f)   dy = -     fmod(POS.y,ONE)  / DIR.y - EPS/DIR.y;
      else              dy =  (ONE-fmod(POS.y,ONE)) / DIR.y + EPS/DIR.y;
      if (DIR.z<0.0f)   dz = -     fmod(POS.z,ONE)  / DIR.z - EPS/DIR.z;
      else              dz =  (ONE-fmod(POS.z,ONE)) / DIR.z + EPS/DIR.z;
      dx         =  min(dx, min(dy, dz)) + EPS ;      // actual step
#endif
      
      
      
# if (WITH_HALF==0)
      doppler    =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# else
      doppler    =  
        vload_half(0,&(CLOUD[4*INDEX]))*DIR.x + 
        vload_half(1,&(CLOUD[4*INDEX]))*DIR.y + 
        vload_half(2,&(CLOUD[4*INDEX]))*DIR.z ;
# endif
      nu         =       NI[2*INDEX] ;           
      tau =  (fabs(NI[2*INDEX+1])<1.0e-30f) ? (dx*1.0e-30f*GN*GL) : clamp((float)(dx*NI[2*INDEX+1]*GN*GL), -2.0f, 1.0e10f) ;
      distance  += dx ;                  
      tau        =  clamp(tau, 1.0e-30f, 1.0e10f) ;  // $$$  KILL ALL MASERS
      
#if (WITH_HALF==0)
      row        =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
#else
      row        =  clamp((int)round(log(vload_half(3,&(CLOUD[4*INDEX]))/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
#endif
      
      profile    =  &GAU[row*CHANNELS] ;
      shift      =  round(doppler/WIDTH) ;
      c1         =  LIM[row].x+shift ;
      c2         =  LIM[row].y+shift ;
      c1         =  max(c1,   max(shift,      0               )) ;
      c2         =  min(c2,   min(CHANNELS-1, CHANNELS-1+shift)) ;      
      emissivity =  emis0 * nu * dx * GL ;    // emissivity =  H_PIx4 * freq * nu * Aul *dx  * I2T ;
      
      // printf("Doppler %.3e,  sigma %.3e, row %d/%d\n", doppler, CLOUD[INDEX].w, row, GNO) ;
      
      
#if 1
      if (level0<MINMAPLEVEL) {
         tau = 0.0f ; emissivity = 0.0f ;   // ignore cells below MINMAPLEVEL
      }
#endif
      
      
      
#if (WITH_CRT>0)
      Ctau       =  CRT_TAU[INDEX] * dx      ;
      Cemit      =  CRT_EMI[INDEX] * dx * GL ;
      for(i=0; i<CHANNELS; i++) {
         pro         =  profile[clamp(i-shift, 0, CHANNELS-1)] ;
         dtau        =  tau*pro + Ctau ;
         NTRUE[i]   += (emissivity*pro*GN + Cemit)*exp(-SUM_TAU[i])* 
           (  (fabs(dtau)>0.01f)  ? ((1.0f-exp(-dtau))/dtau)  :  (1.0f-dtau*(0.5f-0.166666667f*dtau))  ) ;
         // the contribution of the current step to the final spectrum, each channel separately
         LOS_EMIT[step*CHANNELS+i]  = (emissivity*pro*GN + Cemit)*exp(-SUM_TAU[i])* 
           (  (fabs(dtau)>0.01f)  ? ((1.0f-exp(-dtau))/dtau)  :  (1.0f-dtau*(0.5f-0.166666667f*dtau))  ) ;
         SUM_TAU[i] +=  dtau  ;
      }     
#else
      for(i=c1; i<=c2; i++) {         
         dtau        = tau*profile[i-shift] ;
         NTRUE[i]   += emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i])* 
           (  (fabs(dtau)>0.01f)  ?  ((1.0f-exp(-dtau))/dtau)  :  (1.0f-dtau*(0.5f-0.166666667f*dtau))) ;
         // the contribution of the current step to the final spectrum, each channel separately
         LOS_EMIT[step*CHANNELS+i] = emissivity*profile[i-shift]*GN*exp(-SUM_TAU[i])* 
           (  (fabs(dtau)>0.01f)  ?  ((1.0f-exp(-dtau))/dtau)  :  (1.0f-dtau*(0.5f-0.166666667f*dtau))) ;
         SUM_TAU[i] += dtau  ;
      }
#endif
      LENGTH[step]   = dx ;
      LOS_RHO[step]  =  RHO[INDEX] ;
      LOS_TKIN[step] = TKIN[INDEX] ;
      LOS_ABU[step]  =  ABU[INDEX] ;      
      step += 1 ;
      
      
      
#if (WITH_OCTREE>0)
      INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;
#else
      POS.x  += dx*DIR.x ;  POS.y  += dx*DIR.y ;  POS.z  += dx*DIR.z ;
      INDEX = Index(POS) ;         
#endif
   } // while INDEX>=0
   
   

#if 1  // background
   for (i=0; i<CHANNELS; i++) {
      tau    =   SUM_TAU[i] ;
      dtau   =   NTRUE[i] ;
      NTRUE[i] -=  BG * ((fabs(tau)>0.01f ) ?  (1.0f-exp(-tau))  :  (tau*(1.0f-tau*(0.5f-0.166666667f*tau)))) ;
      LOS_EMIT[step*CHANNELS+i] = 
        - BG * ((fabs(tau)>0.01f ) ?  (1.0f-exp(-tau))  :  (tau*(1.0f-tau*(0.5f-0.166666667f*tau)))) ;
   } // for i over CHANNELS
#else  // WITHOUT BACKGROUND -- ONLY FOR TESTING
   if (id==0) printf("x?") ;
#endif
   LENGTH[step] = -1.1e10 ;   // background entry has a corresponding large negative entry in the LENGTH array
   
}  // Spectra_vs_LOS() 
#endif // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



// ==============================================================================















__kernel void Columndensity(
                            const float2       D,            //  0  ray direction == theta, phi
                            const float3       CENTRE,       //  1  map centre
                            const float        DE,           //  2  single dec. offset
                            const int          NRA,          //  3   number of RA points = work items
                            const float        STEP,         //  4 step between spectra (grid units)
#if (WITH_OCTREE>0)
                            __global  int     *LCELLS,       //  5
                            __constant int    *OFF,          //  6
                            __global   int    *PAR,          //  7
#endif
                            __global   float  *RHO,          //  5, 8
                            __global   float  *ABU,          //  6, 9
                            __global   float  *TKIN,         //  7,10
                            __global float    *COLDEN,       //  8,11 -- NRA -- total column density
                            __global float    *MCOLDEN,      //  9,12 -- NRA -- column density of molecule
                            __global float    *WTKIN,        // 10,13 -- NRA -- column density of molecule
                            __global float    *WV,           // mass-weighted <LOS velocity>
#if (WITH_HALF==1)
                            __global half     *CLOUD         //  0 [CELLS,4] ==  vx, vy, vz, sigma
#else
                            __global float4  *CLOUD          //  0 [CELLS]: vx, vy, vz, sigma
#endif
                           )
{
   // each work item calculates results for one pixel
   //   ra  =  RA (grid units, from cloud centre)
   //   de  =  DE(id)
   int id = get_global_id(0) ;
   if (id>=NRA) return ; // no more rays
   int i ;
   float RA ;              // grid units, offset of current ray
   float colden  = 0.0f ;  // 2021-08-04, total column density on the LOS
   float mcolden = 0.0f ;  // 2021-08-22, LOS column density of this species
   float STkin   = 0.0f ;  // for calculation of density-weighted average Tkin -- total weight == mcolden
   float SV      = 0.0f ;  //                                             LOS velocity
   float tmp ;
   RA  =   id  ;
   REAL3   POS, dr, RPOS ;
   float3  DIR ;
   REAL    dx, dy, dz ;
   DIR.x   =   sin(D.x)*cos(D.y) ;     // D.x = theta,   D.y = phi
   DIR.y   =   sin(D.x)*sin(D.y) ;
   DIR.z   =   cos(D.x)            ;
   REAL3 RV, DV ; 
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
   // Offsets in RA and DE directions, (RA, DE) are just indices [0, NRA[, [0,NDE[
   // CENTRE contains indices for the map centre using the current pixel size
   POS.x  =  CENTRE.x + (RA-0.5*(NRA-1.0f))*STEP*RV.x + DE*STEP*DV.x ;
   POS.y  =  CENTRE.y + (RA-0.5*(NRA-1.0f))*STEP*RV.y + DE*STEP*DV.y ;
   POS.z  =  CENTRE.z + (RA-0.5*(NRA-1.0f))*STEP*RV.z + DE*STEP*DV.z ;
   int ID =  ((fabs(POS.y-2.0)<0.05)&&(fabs(POS.z-1.5)<0.02)) ? id : -1 ;
   // Change DIR to direction away from the observer
   DIR *= -1.0f ;   // <----------------------------------------------------<<<
   if (fabs(DIR.x)<1.0e-10f) DIR.x = 1.0e-10f ;
   if (fabs(DIR.y)<1.0e-10f) DIR.y = 1.0e-10f ;
   if (fabs(DIR.z)<1.0e-10f) DIR.z = 1.0e-10f ;   
   // go to front surface, first far enough upstream (towards observer), then step forward to cloud (if ray hits)
   POS.x -= 1000.0*DIR.x ;  POS.y -= 1000.0*DIR.y ;  POS.z -= 1000.0*DIR.z ;
   if (DIR.x>ZERO)  dx = (ZERO-POS.x)/DIR.x ;     // DIR away from the observer
   else             dx = (NX  -POS.x)/DIR.x ;
   if (DIR.y>ZERO)  dy = (ZERO-POS.y)/DIR.y ;
   else             dy = (NY  -POS.y)/DIR.y ;
   if (DIR.z>ZERO)  dz = (ZERO-POS.z)/DIR.z ;
   else             dz = (NZ  -POS.z)/DIR.z ;
   dx      =  max(dx, max(dy, dz)) + 1.0e-4f ;  // max because we are outside
   POS.x  +=  dx*DIR.x ;   POS.y  +=  dx*DIR.y ;   POS.z  +=  dx*DIR.z ;   // even for OT, still in root grid units
   
   int level0 ;   
#if (WITH_OCTREE>0)
   int OTL, OTI, INDEX ;
   IndexG(&POS, &OTL, &OTI, RHO, OFF) ;
   INDEX =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;
#else
   int INDEX   =  Index(POS) ;
#endif
   float tau, dtau  ;
   float distance = 0.0f ;
   
   while (INDEX>=0) {
#if (WITH_OCTREE>0)   // INDEX  =  OFF[OTL] + OTI ;  --- update INDEX at the end of the step
      // OCTREE==5 uses level=0 coordinates POS=[0,1], not [0,NX]
      dx        =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, 99, NULL, -1) ; // updates POS, OTL, OTI
#else
      if (DIR.x<0.0f)   dx = -     fmod(POS.x,ONE)  / DIR.x - EPS/DIR.x;
      else              dx =  (ONE-fmod(POS.x,ONE)) / DIR.x + EPS/DIR.x;
      if (DIR.y<0.0f)   dy = -     fmod(POS.y,ONE)  / DIR.y - EPS/DIR.y;
      else              dy =  (ONE-fmod(POS.y,ONE)) / DIR.y + EPS/DIR.y;
      if (DIR.z<0.0f)   dz = -     fmod(POS.z,ONE)  / DIR.z - EPS/DIR.z;
      else              dz =  (ONE-fmod(POS.z,ONE)) / DIR.z + EPS/DIR.z;
      dx         =  min(dx, min(dy, dz)) + EPS ;      // actual step
#endif
      distance  +=  dx ;                  
      colden    +=  dx*RHO[INDEX] ;
#if 1  // default
      tmp        =  dx*RHO[INDEX]*ABU[INDEX] ;
#else  // test n^2 weighting to see the effect on <v>
      tmp        =  dx*RHO[INDEX]*RHO[INDEX]*ABU[INDEX] ;
#endif      
      mcolden   +=  tmp ;
      STkin     +=  tmp*TKIN[INDEX] ;
      SV        +=  tmp*(CLOUD[INDEX].x*DIR.x+CLOUD[INDEX].y*DIR.y+CLOUD[INDEX].z*DIR.z) ;
#if (WITH_OCTREE>0)
      INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;
#else
      POS.x  += dx*DIR.x ;  POS.y  += dx*DIR.y ;  POS.z  += dx*DIR.z ;
      INDEX = Index(POS) ;         
#endif
   } // while INDEX>=0
   
   COLDEN[id]  = colden*GL ;     // total LOS column density (H2)
   MCOLDEN[id] = mcolden*GL ;    // LOS column density of this species
   WTKIN[id]   = (mcolden>0.0f) ? (STkin/mcolden) : 0.0f ;  // molecular mass weighted average kinetic temperature
   WV[id]      = (mcolden>0.0f) ? (SV   /mcolden) : 0.0f ;  // mass weighted average LOS velocity
}


































#if (WITH_HFS>0)
// HFS case --- simulation kernels UpdateHF() for cartesian grid and UpdateHF4() for OCTREE==4 runs
//              SpectraHF() for both Cartesian and OCTREE==4 cases, to write the spectra

# if (WITH_OCTREE==0)




__kernel void UpdateHF( // @h   non-octree version !!!!!!!!
                        __global float4 *CLOUD,   //  0 [CELLS]: vx, vy, vz, sigma
                        GAUSTORE float  *GAU,     //  1 precalculated gaussian profiles [GNO*CHANNELS]
                        constant int2   *LIM,     //  2 limits of ~zero profile function [GNO]
                        const float      Aul,     //  3 Einstein A(upper->lower)
                        const float      A_b,     //  4 (g_u/g_l)*B(upper->lower)
                        const float      GN,      //  5 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                        const float      APL,     //  6 average path length [GL]
                        const float      BG,      //  7 background value (photons)
                        const float      DIRWEI,  //  8 weight factor cos(theta) / <cos(theta)>
                        const float      EWEI,    //  9 weight factor 1/<1/cos(theta)>/NDIR
                        const int        LEADING, // 10 leading edge
                        const REAL3      POS0,    // 11 initial position of id=0 ray
                        const float3     DIR,     // 12 ray direction
                        __global float  *NI,      // 13 [CELLS]:  NI[upper] + NB_NB
                        __global float  *RES,     // 14 [CELLS]:  SIJ, ESC
                        const int        NCHN,    // 15 number of channels (>= CHANNELS)
                        const int        NCOMP,   // 16 number of HF components
                        __global float2 *HF,      // 17 [MAXCMP].x = offset, [MAXCMP].y = weight
                        __global float  *NTRUES,  // 18 [GLOBAL*MAXCHN]
                        __global float  *PROFILE  // 19 PROFILE[GLOBAL*MAXCHN]
                      )  {
   // this one used for HF in LTE -- for LOC_HF.cpp
   float weight, dx, w, doppler ;
   float tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;
   float sum_delta_true, all_escaped ; // nu -- using directly NI[].x
   int row, shift, INDEX, c1, c2 ;
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge 
   // if ray exits through a side, a new one is created on the opposite side
   // Ray ends when the downstream edge is reached
   int  id = get_global_id(0) ;
   int  lid = get_local_id(0) ;   
   if  (id>=NRAY) return ;
   int  nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ; // dimensions of the current ray grid
   
   GAUSTORE float *pro ;             // pointer to GAU vector
   
#  if (LOC_LOWMEM>0)  // kernel has only distinction LOC_LOWMEM<1 or not (host has 0, 1, >1 !!)
   __global float *NTRUE   = &NTRUES[id*MAXCHN] ;
#  else   // this leads to "insufficient resources" in case of GPU
   __local float  NTRUESSS[LOCAL*MAXCHN] ;
   __local float *NTRUE = &NTRUESSS[lid*MAXCHN] ;
#  endif
   
   
   // It is ok up to ~150 channels to have these on GPU local memory --  LOCAL=32 and 150 channels =14kB ...
#  if (LOC_LOWMEM>0)
   __global float *profile = &PROFILE[id*MAXCHN] ;
#  else
   __local float  profile_array[LOCAL*MAXCHN] ;  // MAXCHN = max of NCHN
   __local float *profile = &profile_array[lid*MAXCHN] ;
#  endif
   
   
   // Initial position of each ray shifted by two grid units
   //  == host has a loop over 4 offset positions
   REAL3 POS = POS0 ;
#  if (FIX_PATHS>0)
   REAL3 POS_LE ;
#  endif
   switch (LEADING) {
    case 0: POS.x = EPS ;    POS.y += TWO*(id%ny) ; POS.z += TWO*(int)(id/ny) ;  break ;
    case 1: POS.x = NX-EPS ; POS.y += TWO*(id%ny) ; POS.z += TWO*(int)(id/ny) ;  break ;
    case 2: POS.y = EPS ;    POS.x += TWO*(id%nx) ; POS.z += TWO*(int)(id/nx) ;  break ;
    case 3: POS.y = NY-EPS ; POS.x += TWO*(id%nx) ; POS.z += TWO*(int)(id/nx) ;  break ;
    case 4: POS.z = EPS ;    POS.x += TWO*(id%nx) ; POS.y += TWO*(int)(id/nx) ;  break ;
    case 5: POS.z = NZ-EPS ; POS.x += TWO*(id%nx) ; POS.y += TWO*(int)(id/nx) ;  break ;
   }
#  if (FIX_PATHS>0)
   POS_LE = POS ;  // initial position of the ray when it is first created
#  endif
   INDEX = Index(POS) ;
   for(int i=0; i<NCHN; i++) NTRUE[i] = BG * DIRWEI ;
   
   while(INDEX>=0) {
      
#  if (NX>DIMLIM) // ====================================================================================================
      double dx, dy, dz ;
      dx = (DIR.x>0.0f)  ?  ((1.0+EPS-fmod(POS.x,ONE))/DIR.x)  :  ((-EPS-fmod(POS.x,ONE))/DIR.x) ;
      dy = (DIR.y>0.0f)  ?  ((1.0+EPS-fmod(POS.y,ONE))/DIR.y)  :  ((-EPS-fmod(POS.y,ONE))/DIR.y) ;
      dz = (DIR.z>0.0f)  ?  ((1.0+EPS-fmod(POS.z,ONE))/DIR.z)  :  ((-EPS-fmod(POS.z,ONE))/DIR.z) ;
      dx =  min(dx, min(dy, dz)) ;
#  else
      dx=        (DIR.x<0.0f) ? (-fmod(POS.x,ONE)/DIR.x-EPS/DIR.x) : ((ONE-fmod(POS.x,ONE))/DIR.x+EPS/DIR.x) ;
      dx= min(dx,(DIR.y<0.0f) ? (-fmod(POS.y,ONE)/DIR.y-EPS/DIR.y) : ((ONE-fmod(POS.y,ONE))/DIR.y+EPS/DIR.y)) ;
      dx= min(dx,(DIR.z<0.0f) ? (-fmod(POS.z,ONE)/DIR.z-EPS/DIR.z) : ((ONE-fmod(POS.z,ONE))/DIR.z+EPS/DIR.z)) ;
#  endif
      // nu        =  NI[INDEX].x ;
      nb_nb     =  NI[2*INDEX+1] ;
      
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
      weight    =  (dx/APL)*VOLUME ;  // VOLUME == 1.0/CELLS, fraction of cloud volume
      POS.x    +=  dx*DIR.x ;   POS.y    += dx*DIR.y ;   POS.z    += dx*DIR.z ;
      tmp_tau   =  dx*nb_nb*GN ;
      if (fabs(tmp_tau)<1.0e-32f) tmp_tau = 1.0e-32f ;     // was e-32
      tmp_emit  =  weight * NI[2*INDEX] * (Aul/tmp_tau) ;  // GN include grid length [cm]
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
#  if (WITH_HALF==1)
      dx *=  0.002f ;  // half integer times 0.002f km/s
#  endif
#  if (WITH_HALF==1)
      //               sigma = 0.002f * w
      // lookup table: sigma = SIGMA0 * SIGMAX^row
      row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      // if (id==0) printf("  sigma %6.2f  --- row %d/%d\n", CLOUD[INDEX].w*0.002f, row, GNO) ;
#  else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
#  endif
      
      // Calculate profile as weighted sum of precalculated Gaussian profiles
      for(int i=0; i<NCHN; i++) profile[i] = 0.0f ;
      pro       =  &GAU[row*CHANNELS] ;
      for(int icomp=0; icomp<NCOMP; icomp++) {
         shift  =  round(doppler/WIDTH) + HF[icomp].x  ;  // shift in channels doppler !
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
      
      for(int ii=0; ii<NCHN; ii++)  {
         w               =  tmp_tau*profile[ii] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         // factor          =  clamp(factor,  1.0e-30f, 1.0f) ;  // KILL MASERS $$$
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]      +=  escape-absorbed ;
         sum_delta_true +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      
      
#  if (NO_ATOMICS>0)
      // if nb_nb was already checked
      // RES[2*INDEX]   +=  (A_b/VOLUME) * (sum_delta_true/nb_nb) ;
      RES[2*INDEX]   +=  A_b * (sum_delta_true/nb_nb) ;
      // Emission ~ path length dx but also weighted according to direction, 
      //            works because <WEI>==1.0
      RES[2*INDEX+1] +=  all_escaped ;
#  else  
      w  =  A_b * (sum_delta_true/nb_nb) ;    // parentheses!
      AADD((__global float*)RES+2*INDEX,   w) ;
      AADD((__global float*)RES+2*INDEX+1, all_escaped) ;      
#  endif
      
      // POS       += dx*DIR ;
      INDEX      = Index(POS) ;
      
      if (INDEX<0) {  // exits the cloud... but on which side?
         if (POS.x>=NX  ) {  if (LEADING!=0)  POS.x =    EPS ;   }
         if (POS.x<=ZERO) {  if (LEADING!=1)  POS.x = NX-EPS ;   } 
         if (POS.y>=NY  ) {  if (LEADING!=2)  POS.y =    EPS ;   }
         if (POS.y<=ZERO) {  if (LEADING!=3)  POS.y = NY-EPS ;   } 
         if (POS.z>=NZ  ) {  if (LEADING!=4)  POS.z =    EPS ;   }
         if (POS.z<=ZERO) {  if (LEADING!=5)  POS.z = NZ-EPS ;   } 
         INDEX = Index(POS) ;
         if (INDEX>=0) {   // new ray started on the opposite side (same work item)
            for(int ii=0; ii<NCHN; ii++) NTRUE[ii] = BG * DIRWEI ;
         }
      } // if INDEX<0
      
   } // while INDEX>=0
   
}


# endif  // OCTREE==0





__kernel void SpectraHF(  // @h
# if (WITH_HALF==1)
                          __global short4 *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
# else
                          __global float4 *CLOUD,        //  0 [CELLS]: vx, vy, vz, sigma
# endif
                          GAUSTORE float  *GAU,          //  1 precalculated gaussian profiles
                          constant int2   *LIM,          //  2 limits of ~zero profile function
                          const float      GN,           //  3 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                          const float2     D,            //  4 ray direction == theta, phi
                          __global float  *NI,           //  5 [CELLS]:  NI[upper] + NB_NB
                          const float      DE,           //  6 grid units, offset in DE direction
                          const int        NRA,          //  7 number of RA points = work items
                          const float      STEP,         //  8 step between spectra (grid units)
                          const float      BG,           //  9 background intensity
                          const float      emis0,        // 10 h/(4pi)*freq*Aul*int2temp
                          __global float  *NTRUE_ARRAY,  // 11 NRA*MAXNCHN
                          __global float  *SUM_TAU_ARRAY,// 12 NRA*MAXNCHN
# if (WITH_OCTREE>0)
                          __global int    *LCELLS,       // 13
                          __constant int  *OFF,          // 14
                          __global int    *PAR,          // 15
                          __global float  *RHO,          // 16
# endif
                          const int NCHN,                // 13/17 channels (in case of HF spectrum)
                          const int NCOMP,               // 14/18 number of components
                          __global float2 *HF,           // 15/19 channel offsets, weights
                          __global float *PROFILE,       // 16/20 PROFILE[GLOBAL*MAXCHN]
                          const float3  CENTRE           // 17/21 map centre, offset in units of the pixel size
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
   REAL3  POS ;
   float3 DIR ;
   // float dx, dy, dz ;
   double dx, dy, dz ;
# if 1
   DIR.x   =   sin(D.x)*cos(D.y) ;
   DIR.y   =   sin(D.x)*sin(D.y) ;
   DIR.z   =   cos(D.x)            ;
   REAL3 RV, DV ;
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
   POS.x  =  CENTRE.x + (RA-0.5*(NRA-1.0f))*STEP*RV.x + DE*STEP*DV.x ;
   POS.y  =  CENTRE.y + (RA-0.5*(NRA-1.0f))*STEP*RV.y + DE*STEP*DV.y ;
   POS.z  =  CENTRE.z + (RA-0.5*(NRA-1.0f))*STEP*RV.z + DE*STEP*DV.z ;
   // Change DIR to direction away from the observer
   DIR *= -1.0f ;
# else
   // RA offset
   POS.x  +=  +RA * sin(D.y)  ;  // RA in grid units
   POS.y  +=  -RA * cos(D.y)  ;
   POS.z  +=   ZERO ;
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
   
   // go to front surface, step 1000 makes sure we are outside the cloud and on the observer side
   // ... as long as the root grid is no larger than ~1000^3
   // the step by 1000 also means position needs to be calculated with double precision (float would need corrections)
   POS.x -= 1000.0f*DIR.x ;  POS.y -= 1000.0f*DIR.y ;  POS.z -= 1000.0f*DIR.z ;
   if (DIR.x>0.0f)  dx = (ZERO-POS.x)/DIR.x ;
   else             dx = (NX  -POS.x)/DIR.x ;
   if (DIR.y>0.0f)  dy = (ZERO-POS.y)/DIR.y ;
   else             dy = (NY  -POS.y)/DIR.y ;
   if (DIR.z>0.0f)  dz = (ZERO-POS.z)/DIR.z ;
   else             dz = (NZ  -POS.z)/DIR.z ;
   dx    =  max(dx, max(dy, dz)) + EPS ;  // max because we are outside
# if 1
   POS.x += dx*DIR.x ;   POS.y += dx*DIR.y ;   POS.z += dx*DIR.z ;
# else
   POS        +=  dx*DIR ;
# endif
   
# if (WITH_OCTREE>0)
   int OTL, OTI, INDEX ;
   // Note: for OCTREE=5,  input is [0,NX], output coordinates are [0,1] for root-grid cells
   IndexG(&POS, &OTL, &OTI, RHO, OFF) ;
   INDEX =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;
   // printf("INDEX  %8d   %9.6f %9.6f %9.6f\n", INDEX, POS.x, POS.y, POS.z) ;
# else
   int INDEX   =  Index(POS) ;
# endif
   
   for(int i=0; i<NCHN; i++) {
      NTRUE[i]   = 0.0f ;    SUM_TAU[i] = 0.0f ;
   }
   
   float tau, dtau, emissivity, doppler, nu ;
   int row, shift, c1, c2  ;
   
   GAUSTORE float* pro ;
   
# if (LOC_LOWMEM<1)   
   __local float  profile_array[LOCAL*MAXCHN] ;
   __local float *profile = &profile_array[lid*MAXCHN] ;
# else
   __global float *profile = &(PROFILE[id*MAXCHN]) ;
# endif
   
   // printf("emis0 %12.4e, nu %12.4e, nbnb %12.4e, GL %12.4e\n", emis0, NI[INDEX].x, NI[INDEX].y, GL) ;
   
   
   
   while (INDEX>=0) {
      
# if (WITH_OCTREE>0)
      dx     =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, 99, NULL, -1) ; // updates POS, OTL, OTI
      // distance += dx ;      
# else
      if (DIR.x<0.0f)   dx = -     fmod(POS.x,ONE)  / DIR.x - EPS/DIR.x;
      else              dx =  (ONE-fmod(POS.x,ONE)) / DIR.x + EPS/DIR.x;
      if (DIR.y<0.0f)   dy = -     fmod(POS.y,ONE)  / DIR.y - EPS/DIR.y;
      else              dy =  (ONE-fmod(POS.y,ONE)) / DIR.y + EPS/DIR.y;
      if (DIR.z<0.0f)   dz = -     fmod(POS.z,ONE)  / DIR.z - EPS/DIR.z;
      else              dz =  (ONE-fmod(POS.z,ONE)) / DIR.z + EPS/DIR.z;
      dx         =  min(dx, min(dy, dz)) + EPS ;  // actual step
# endif
      
      nu         =       NI[2*INDEX] ;     
      
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
         // POS  += dx*DIR ;
         POS.x += dx*DIR.x ;  POS.y += dx*DIR.y ;  POS.z += dx*DIR.z ;
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
# if 0
         if (fabs(dtau)>1.0e-5f)  {            
            NTRUE[i] +=  emissivity*profile[i]*GN*exp(-SUM_TAU[i]) * (1.0f-exp(-dtau)) / dtau ;
         } else {
            NTRUE[i] += (emissivity*profile[i]*GN*exp(-SUM_TAU[i])) ;
         }
# else
         NTRUE[i] +=  emissivity*profile[i]*GN*exp(-SUM_TAU[i]) * 
           (  (fabs(dtau)>0.01f) ? ((1.0f-exp(-dtau))/dtau) : (1.0f-dtau*(0.5f-0.166666667f*dtau))  ) ;
# endif
         SUM_TAU[i] += dtau  ;
      } // for ... over NCHN
      
      
# if (WITH_OCTREE>0)
      INDEX    =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;
# else
      POS.x += dx*DIR.x ;  POS.y += dx*DIR.y ;  POS.z += dx*DIR.z ;      
      INDEX = Index(POS) ;
# endif
   } // while INDEX
   

   
# if 0
   for (i=0; i<NCHN; i++) NTRUE[i] -=  BG*(1.0f-exp(-SUM_TAU[i])) ;
# else
   for (i=0; i<NCHN; i++) {
      dtau      =  SUM_TAU[i] ;
      NTRUE[i] -=  BG *  (  (fabs(dtau)>0.02f) ? (1.0f-exp(-dtau)) : (dtau*(1.0f-dtau*(0.5f-0.166666667f*dtau)))  ) ;
   }
# endif
   
}








# if (WITH_OCTREE==4)



__kernel void UpdateHF4(  // @h
                          const int        gid0,    //  0 first gid in the index running over NRAY>=NWG
                          __global float  *PL,      //  1
#  if (WITH_HALF==1)
                          __global short4 *CLOUD,   //  2 [CELLS]: vx, vy, vz, sigma
#  else
                          __global float4 *CLOUD,   //  2 [CELLS]: vx, vy, vz, sigma
#  endif
                          GAUSTORE  float *GAU,     //  3 precalculated gaussian profiles [GNO,CHANNELS]
                          constant int2   *LIM,     //  4 limits of ~zero profile function [GNO]
                          const float      Aul,     //  5 Einstein A(upper->lower)
                          const float      A_b,     //  6 (g_u/g_l)*B(upper->lower)
                          const float      GN,      //  7 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                          const float      APL,     //  8 average path length [GL]
                          const float      BG,      //  9 background value (photons)
                          const float      DIRWEI,  // 10 weight factor (based on current direction)
                          const float      EWEI,    // 11 weight 1/<1/cosT>/NDIR
                          const int        LEADING, // 12 leading edge
                          const REAL3      POS0,    // 13 initial position of id=0 ray
                          const float3     DIR,     // 14 ray direction
                          __global float  *NI,      // 15 [CELLS]:  NI[upper] + NB_NB
                          const int        NCHN,    // 16 number of channels (!=CHANNELS)
                          const int        NCOMP,   // 17 number of HF components
                          __global float2 *HF,      // 18 [MAXCMP].x = offset, [MAXCMP].y = weight
                          __global float  *RES,     // 19 [CELLS]: SIJ, ESC ---- or just [CELLS] for SIJ
                          __global float  *NTRUES,  // 20 [NWG*MAXCHN]   --- NWG>=simultaneous level 0 rays
#  if (WITH_CRT>0)
                          constant float *CRT_TAU,  //  dust optical depth / GL
                          constant float *CRT_EMI,  //  dust emission photons/c/channel/H
#  endif                     
#  if (BRUTE_COOLING>0)
                          __global float   *COOL,   // [CELLS] = cooling 
#  endif
                          __global   int   *LCELLS, //  21
                          __constant int   *OFF,    //  22
                          __global   int   *PAR,    //  23
                          __global   float *RHO,    //  24  -- needed only to describe the hierarchy
                          __global   float *BUFFER_ALL  //  25 -- buffer to store split rays
                       )  {   
   // Each ***WORK GROUP*** processes one ray. 
   // Unlike in Update4, profile is here NCHN wide and BUFFER requires storage for NCHN instead of CHANNELS channels
   int id  = get_global_id(0), lid = get_local_id(0), gid = get_group_id(0), ls  = get_local_size(0) ;
   __global float *BUFFER = &BUFFER_ALL[gid*(26+NCHN)*MAX_NBUF] ;  // here gid <= NRAY
   gid += gid0 ;                  // becomes running index over NRAY ....           here gid ~ NRAY
   if (gid>=NRAY) return ;        // one work group per ray .... NWG==NRAY   
   __local  float  NTRUE[MAXCHN] ;
   // MAXCHN=1024, LOCAL=32  =>  local memory (32-64kB) already exhausted !!!
   // ==> profile may have to be moved to global memory !!
#  if 1
   __local  float  profile[MAXCHN] ;
#  else
   // IF PROFILE IS MADE GLOBAL,MUST ALSO CHECK BARRIERS => CLK_GLOBAL_MEM_FENCE_
   __global float *profile = &(BUFFER[gid*(26+NCHN)*(MAX_NBUF-1)]) ; // NEVER USING MAX_NBUF ???
#  endif
   __local  float  SDT[LOCAL] ;    // per-workitem sum_delta_true
   __local  float  AE[LOCAL] ;     // per-workitem all_escaped
   __local  int2   LINT ;          // SID, NBUF
   float weight, w, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed, sum_delta_true, all_escaped, nu ;
   int row, shift, INDEX, c1, c2, OTL, OTI, OTLO, XL, RL, SID, NBUF=0, sr, sid, level, b1, b2, I, i, ind, otl, oti ;
   int level1, ind1 ;
   REAL3  POS, pos0, pos1, pos, RDIR ;
   REAL   dx, dy, dz, s ;
   float dr, flo ;
   GAUSTORE float *pro ;
#  if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, sij ;
#  endif  
#  if (ONESHOT<1)
   int nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ;
#  endif
   RDIR.x = DIR.x ; RDIR.y = DIR.y ; RDIR.z = DIR.z ;
   
   // if (id==0) printf("UpdateHF4\n") ;
   
   int *SUBS ;
   POS.x = POS0.x ;   POS.y = POS0.y ;   POS.z = POS0.z ;
   
   // when split done only on the leading edge -- HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
#  if (ONESHOT<1)
   if (LEADING<3) {
      if (LEADING==0) { 
         HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;
         POS.x =    EPS ;  POS.y += TWO*(gid%ny) ;  POS.z += TWO*(int)(gid/ny) ;  // gid is index running over all NRAY
      }
      if (LEADING==1) { 
         HEAD[0] = 1 ;   HEAD[1] = 3 ;   HEAD[2] = 5 ;  HEAD[3] = 7 ;
         POS.x = NX-EPS ;  POS.y += TWO*(gid%ny) ;  POS.z += TWO*(int)(gid/ny) ;
      }
      if (LEADING==2) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;
         POS.y =    EPS ;  POS.x += TWO*(gid%nx) ;  POS.z += TWO*(int)(gid/nx) ;
      }
   } else {
      if (LEADING==3) { 
         HEAD[0] = 2 ;   HEAD[1] = 3 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.y = NY-EPS ;  POS.x += TWO*(gid%nx) ;  POS.z += TWO*(int)(gid/nx) ;
      }
      if (LEADING==4) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;
         POS.z =    EPS ;  POS.x += TWO*(gid%nx) ;  POS.y += TWO*(int)(gid/nx) ;
      }
      if (LEADING==5) { 
         HEAD[0] = 4 ;   HEAD[1] = 5 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.z = NZ-EPS ;  POS.x += TWO*(gid%nx) ;  POS.y += TWO*(int)(gid/nx) ;
      }
   }
#  else
   if (LEADING<3) {
      if (LEADING==0) { 
         HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;
         POS.x =    EPS ;  POS.y += gid%NY ;  POS.z += gid/NY ;  // gid is index running over all NRAY
      }
      if (LEADING==1) { 
         HEAD[0] = 1 ;   HEAD[1] = 3 ;   HEAD[2] = 5 ;  HEAD[3] = 7 ;
         POS.x = NX-EPS ;  POS.y += gid%NY ;  POS.z += gid/NY ;
      }
      if (LEADING==2) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;
         POS.y =    EPS ;  POS.x += gid%NX ;  POS.z += gid/NX ;
      }
   } else {
      if (LEADING==3) { 
         HEAD[0] = 2 ;   HEAD[1] = 3 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.y = NY-EPS ;  POS.x += gid%NX ;  POS.z += gid/NX ;
      }
      if (LEADING==4) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;
         POS.z =    EPS ;  POS.x += gid%NX ;  POS.y += gid/NX ;
      }
      if (LEADING==5) { 
         HEAD[0] = 4 ;   HEAD[1] = 5 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.z = NZ-EPS ;  POS.x += gid%NX ;  POS.y += gid/NX ;
      }
   }
#  endif
   
   
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;     // remain at root level, not yet going to leaf
   if (OTI<0) return ;
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
   OTLO      =  OTL ;
   RL        =  0 ;
   for(int i=lid; i<NCHN; i+=ls)  NTRUE[i] = BG * DIRWEI ;
   
   
   
   while(INDEX>=0) {  // INDEX may refer to a cell that is not a leaf
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
      // If we are not in a leaf, we have gone to some higher level. 
      // Go one level higher, add >=three rays, pick one of them as the current,
      // and return to the beginning of the loop.
      if (RHO[INDEX]<=0.0f) {                    // go to the sub-cell and add sibling rays
         
         sr  = 0 ;
         c1  = NBUF*(26+NCHN) ;
         SID = -1 ;
         
         POS.x  =  TWO*fmod(POS.x, ONE) ;      // coordinate inside parent cell [0,1]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         flo    =  -RHO[INDEX] ;                 // OTL, OTI of the parent cell
         OTL   +=  1  ;                          // step to next level = refined level
         OTI    =  *(int *)&flo ;                // OTI for the first child in octet
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // original subcell
         OTI   +=  SID;                          // cell in octet, original ray
         c2     =  0 ;   // set c2=1 if ray is to be split (is this the leading cell edge?)
         if (LEADING<3) {
            if ((LEADING==0)&&(POS.x<  EPS2)) c2 = 1 ; 
            if ((LEADING==1)&&(POS.x>TMEPS2)) c2 = 1 ; 
            if ((LEADING==2)&&(POS.y<  EPS2)) c2 = 1 ;
         } else {
            if ((LEADING==3)&&(POS.y>TMEPS2)) c2 = 1 ; 
            if ((LEADING==4)&&(POS.z<  EPS2)) c2 = 1 ;
            if ((LEADING==5)&&(POS.z>TMEPS2)) c2 = 1 ;
         }
         // @@  rescale always when the resolution changes
         for(int i=lid; i<NCHN; i+=ls) {   // SCALE ON EVERY REFINEMENT EVEN WHEN NOT SPLIT
            NTRUE[i] *= 0.25f ; 
         }
         barrier(CLK_LOCAL_MEM_FENCE) ;
         
         if (c2>0) {  // split the ray and choose one of the new ones to be followed first
            for(int i=lid; i<NCHN; i+=ls) {   // ray effectively split to four
               BUFFER[c1+26+i] = NTRUE[i] ;
            }
            barrier(CLK_LOCAL_MEM_FENCE) ;            
            if (lid==0) {
               BUFFER[c1+0]  =  OTL ;                     // level where the split is done, OTL>this =>
               BUFFER[c1+1]  =  I2F(OTI) ;                // buffer contains the OTI of the original ray 
               BUFFER[c1+2]  =  POS.x ;                   // Store the original MAIN RAY to buffer as the first one
               BUFFER[c1+3]  =  POS.y ;
               BUFFER[c1+4]  =  POS.z ;
               BUFFER[c1+5]  =  RL  ;                     // main ray exists at levels >= RL
            }            
            // Two new subrays added to buffer, third one becomes the current ray
            // Add first two subrays to buffer
            if (                  HEAD[0]==SID) {  // 0=original => (1,2) to buffer, 3 as current
               if (lid==0) {
                  sid              = HEAD[1] ;
                  BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                  BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                  BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                  sid              = HEAD[2] ;
                  BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                  BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                  BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
               }
               SID              = HEAD[3] ;  // SID of the subray to be followed first
            } else {
               if (                  HEAD[1]==SID) {
                  if (lid==0) {
                     sid              = HEAD[2] ;
                     BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                     BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                     BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                     sid              = HEAD[3] ;
                     BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                     BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                     BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                  }
                  SID              = HEAD[0] ;
               } else {
                  if (                  HEAD[2]==SID) {
                     if (lid==0) {
                        sid              = HEAD[3] ;
                        BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                        BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                        BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                        sid              = HEAD[0] ;
                        BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                        BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                        BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                     }
                     SID              = HEAD[1] ;
                  } else {
                     if (                  HEAD[3]==SID) {
                        if (lid==0) {
                           sid              = HEAD[0] ;
                           BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                           BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                           BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                           sid              = HEAD[1] ;
                           BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                           BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                           BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                        }
                        SID              = HEAD[2] ;
                     } else {
                        ; // printf("???\n") ;
                     }
                  }
               }
            } 
            // for the two subrays just added, update RL, SPLIT
            sr = 3 ;   // so far the original main ray and two split subrays
            if (lid==0) {
               BUFFER[c1+5+4*1]  =  OTL ;    BUFFER[c1+5+4*2]  =  OTL ;  // RL = current OTL
               for(int i=sr; i<6; i++)  BUFFER[c1+2+4*i] = -99.0f ;      // mark the rest as unused
            } // lid==0
            // We added leading edge rays, old main ray is in buffer, SID refers to one of the subrays
            // update OTI and POS to correspond to that subray
            OTI       =  (*(int *)&flo)  + SID ;    // the new ray to be followed, OTI = index of first subcell + SID
            POS.x     =  fmod(POS.x,ONE) + (int)( SID%2)    ;  // dx and SID known to all work items
            POS.y     =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
            POS.z     =  fmod(POS.z,ONE) + (int)( SID/4)    ;
            RL        =  OTL ;  // when we reach OTL<RL, this ray will be terminated
            NBUF++ ;            // >= 1 subrays were added, NBUF increases just by one
            
         } // c2>0  == we split rays on the leading edge
         
      } // RHO<0
      
      
      INDEX  = OFF[OTL] + OTI ;
      if (RHO[INDEX]<=0.0f) continue ;  // not a leaf, go back to the beginning of the main loop
      
      
      
      
      
#  if 1  // adding siderays
      
      // It is essential that we are already in a leaf node when siderays are added:
      // once ray has produced siderays at the current location, it will immediately take a 
      // real step forward.
      
      // we use root coordinates to determine whether RL ray is also RL-1 ray
      // the required accuracy is only 0.5*0.5**MAXL, float32 works at least to MAXL=15
      pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;   // current position in root coordinates
      
      
      // Current ray is still (OTL, OTI, POS)
      // presumably current NTRUE is correct for ray at level OTL
      
      // Siderays are based on rays hitting the leading edge. We have stepped down to leaf level.
      // Therefore, incoming ray has already been split to four, and NTRUE has been scaled by 0.25 to
      // correspond to the step from OTL-1 to OTL.
      
      
      for(XL=OTL; XL>RL; XL--) {   // try to add subrays at levels   RL<XL<=OTL
         
         c1     =  NBUF*(26+NCHN) ;          // 26+NCHN elements per buffer entry
         sr     =  0  ;
         
         if (((LEADING==0)&&(POS.x<EPS2))||((LEADING==1)&&(POS.x>TMEPS2))) {  // +/-X leading edge, at OTL level
            dr   = pown(0.5f, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.x/dr)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test smaller XL (larger dr) values
            // even number of dr,is therefore a border between level XL octets
            // calculate (pos, level, ind) for the position at level XL  (in the octet that we are in)
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // (pos, level, ind) now define the position at level XL
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL ===== offsets                              Y and Z
            // check XL-scale neighbour
            pos1.x =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       Y * 
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level,NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               // Current ray position at level==XL is defined by (level, ind, pos).
               // We loop over XL positions on the leading-edge plane, ignore those not common with
               // XL-1 rays, and choose those that actually hit the -Y side of the current octet.
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.y += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =       Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;    // still inside the current octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     // pos1 = initial coordinates at the leading-edge level, step forward to the Y sidewall
                     if (DIR.y>0.0f)   pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ;  // step based on    Y ****
                     else              pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;  // to 2.0  (level==RL+1)
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.y  =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in   octet        Y *****
#   endif
                     // Add the ray from the upstream direction to buffer as a new subray
                     //    we leave the buffer entry at level XL, if leafs are >XL, we drop to next 
                     //    refinement level only at the beginning of the main loop.
                     // Each sideray will actually continue in location refined to some level >=XL
                     // We will store NTRUE correct for XL. If sideray is in refined region, beginning of the
                     // main loop will drop the ray to the higher level and rescale NTRUE accordingly.
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                        BUFFER[c1+3+4*sr] =  pos1.y ;
                        BUFFER[c1+4+4*sr] =  pos1.z ;
                        BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level==XL
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL ===== offsets                              Z and Y
            // current ray at level level==RL+1 defined by (level, ind, pos)
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {     // neighbour not refined to level==XL, will not provide XL siderays
               for(int a=-1; a<=3; a++) {       // offset in +/- Z direction from the level RL ray, in XL steps
                  for(int b=-1; b<=3; b++) {    // ray offset in +/- Y direction, candidate relative to current ray
                     if ((a%2==0)&&(b%2==0)) continue ; // skip LR rays
                     // if we come this far, we will add the ray if it just hits the current octet with RL+1 cells
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =      Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // in current octet, will not hit sidewall
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // second offset           Y ***
                     // pos1 = initial coordinates at the leading edge plane, step to the Z sidewall
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in octet,        Z *****
#   endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                        BUFFER[c1+3+4*sr] = pos1.y ;
                        BUFFER[c1+4+4*sr] = pos1.z ;
                        BUFFER[c1+5+4*sr] = XL     ;      // ray created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (((LEADING==2)&&(POS.y<EPS2))||((LEADING==3)&&(POS.y>TMEPS2))) {  // +/-Y leading edge
            dr   = pown(0.5f, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.y/dr)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // ***A*** even number of dr,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL) IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITTING THE X SIDEWALL ===== offsets                             X and Z
            pos1.y =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.x =  (DIR.x>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       X *
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos inthe octet,      X *****
#   endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                        BUFFER[c1+3+4*sr] =  pos1.y ;
                        BUFFER[c1+4+4*sr] =  pos1.z ;
                        BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL =====  offsets                             Z and X
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =     Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =     X ***
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in the octet,    Z *****
#   endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                        BUFFER[c1+3+4*sr] = pos1.y ;
                        BUFFER[c1+4+4*sr] = pos1.z ;
                        BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         
         if (((LEADING==4)&&(POS.z<EPS2))||((LEADING==5)&&(POS.z>TMEPS2))) {  // +/-Z leading edge
            dr   = pown(0.5f, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.z/dr)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // even number of dr,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITING THE X SIDEWALL ===== offsets                              X and Y
            pos1.y =  0.1f ;     pos1.z = 0.1f ;
            pos1.x =  (DIR.x>0.0f) ? (-0.1f) : (2.1f) ;    // upstream neighbour, main offset       X *
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.y += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Y ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos in the octet,     X *****
#   endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                        BUFFER[c1+3+4*sr] =  pos1.y ;
                        BUFFER[c1+4+4*sr] =  pos1.z ;
                        BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     }
                     sr += 1 ;
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL =====  offsets                            Y and X
            pos1.x =  0.1f ;    pos1.z = 0.1f ;
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset       Y *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL 
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =    Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =    X ***
                     if (DIR.y>0.0f) pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ; // step based on      Y ****
                     else            pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.y   =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in the octet,   Y *****
#   endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                        BUFFER[c1+3+4*sr] = pos1.y ;
                        BUFFER[c1+4+4*sr] = pos1.z ;
                        BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (sr>0) {  // added some siderays
            // While the current ray was at level OTL, there are no guarantees that the same
            // refinement exists for all added siderays. Therefore, we add all siderays
            // using the level XL coordinates. This also means that the siderays and the
            // leading-edge subrays (below) must be stored as separate BUFFER entries (different NBUF).
            if (lid==0){
               BUFFER[c1+0] = XL ;           // at level XL all rays stored as level XL rays
               BUFFER[c1+1] = I2F(ind) ;     // index of *original ray*, at level XL
               // We leave the original ray as the current one == (OTL, OTI, POS) remain unchanged.
               for(int i=sr; i<6; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
            }
            // NTRUE is correct for OTL but we store data for level XL
            // Therefore, BUFFER will contain  NTRUE * 0.25**(XL-OTL)  == NTRUE * 4**(OTL-XL)
            // ==>  the BUFFER values are   NTRUE(correct for RL) * 4.0**(OTL-XL), OTL and not RL !!!!
            // When sideray is taken from buffer and if it is located in a region with OTL>XL, the
            // beginning of the main loop will again rescale with  4.0*(XL-OTL)
            dr  =  pown(4.0f, OTL-XL) ;          // was RL-XL but surely must be OTL-XL !!!!!
            for(int i=lid; i<NCHN; i+=ls) {
               BUFFER[c1+26+i] = NTRUE[i]*dr ;   // NTRUE scaled from RL to XL 
            }            
            NBUF += 1 ;            
         }
         
      } // for XL -- adding possible siderays
      
      
      // Subrays are added only when we are at leaf level. Also, when subrays are added, 
      // we continue with the original RL ray and immediately make a step forward. Therefore
      // there is no risk that we would add siderays multiple times.
#  endif // adding siderays
      
      
      
      
      // global index -- must be now >=0 since we started with while(INDEX>=0) and just possibly went down a level
      INDEX = OFF[OTL]+OTI ;   
      
      // if not yet a leaf, jump back to start of the loop => another step down
      if (RHO[INDEX]<=0.0f) {
         continue ;
      }
      
      // we are now in a leaf node, ready to make the step
      OTLO   =  OTL ;
      
      
      
      dr     =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL   , LEADING) ;  // POS is now the end of this step
      
#  if 0   // @@ double check PL calculation ... PL[:] should be reduced to zero
      if (lid==0)  AADD(&(PL[INDEX]), -dr) ;
#  endif
      
      
      // if (RHO[INDEX]>CLIP) { // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // one workgroup per ray => can have barriers inside the condition
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
#  if 0
      for(int i=lid; i<LOCAL; i+=LOCAL) {
         printf("[%3d]  %2d %6d   %8.4f %8.4f %8.4f   %2d\n", lid, OTL, OTI, POS.x, POS.y, POS.z, NBUF) ;
      }
#  endif
      
      
      // with path length already being the same in all cells !   V /=8,  rays*4==length/2 /=2 =>  2/8=1/4
      weight    =  (dr/APL) *  VOLUME  *  pown(0.25f, OTLO) ;  // OTL=1 => dr/APL=0.5
      nu        =  NI[2*INDEX] ;
#  if 0
      nb_nb     =  NI[2*INDEX+1] ;
#  else
      nb_nb     =  max(1.0e-30f, NI[2*INDEX+1]) ; // $$$ KILL MASERS 
#  endif
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
#  if (WITH_HALF==1)
      doppler  *=  0.002f ;  // half integer times 0.002f km/s
#  endif
      tmp_tau   =  dr*nb_nb*GN ;
      tmp_emit  =  weight * nu*Aul / tmp_tau ;  // GN include grid length [cm]
      shift     =  round(doppler/WIDTH) ;
#  if (WITH_HALF==1)      //               sigma = 0.002f * w,   lookup table: sigma = SIGMA0 * SIGMAX^row
      row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
#  else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
#  endif      
      
      // Instead of   "profile   =  &GAU[row*CHANNELS]" ....
      // calculate profile as weighted sum of Gaussian profiles
      for(int i=lid; i<NCHN; i+=ls) profile[i] = 0.0f ;
      barrier(CLK_LOCAL_MEM_FENCE) ;      // *** as long as profile is __local
      pro       =  &GAU[row*CHANNELS] ;   // GAU has vectors for CHANNELS, not NCHN
      for(int icomp=0; icomp<NCOMP; icomp++) {
         shift  =  round(doppler/WIDTH) + HF[icomp].x ;
         // skip profile function outside channels [LIM.x, LIM.y]
         // LIM.x in pro[] is mapped to channel c1 in profile[NCHN]
         c1 = 0.5f*(NCHN-1.0f) + shift - 0.5f*(CHANNELS-1.0f) + LIM[row].x ; // could be negative
         c2 = c1+LIM[row].y-LIM[row].x ;  // update [c1,c2[ in profile
         //   c1 could be negative => require c1+ii >= 0
         //   c1+LIM.y-LIM.x could be >= NCHN ==> require   c1+ii < NCHN ==>  ii< NCHN-c1
         for(int ii=max(0,-c1)+lid; ii<min(LIM[row].y-LIM[row].x, NCHN-c1); ii+=ls) {
            profile[c1+ii] +=  HF[icomp].y * pro[LIM[row].x+ii] ;
         }  // this assumes that sum(HFI[].y) == 1.0 !!
         barrier(CLK_LOCAL_MEM_FENCE) ;
      }
      sum_delta_true = 0.0f ;
      all_escaped    = 0.0f ;
      
      
#  if (WITH_CRT>0) // WITH_CRT
      sij = 0.0f ;
      // Dust optical depth and emission
      //   here escape = line photon exiting the cell + line photons absorbed by dust
      Ctau      =  dr     * CRT_TAU[INDEX] ;
      Cemit     =  weight * CRT_EMI[INDEX] ;      
      for(int i=lid; i<NCHN; i+=ls)  {
         flo    =  profile[i] ;
         Ltau   =  tmp_tau*flo ;
         Ttau   =  Ctau + Ltau ;
         // tt     =  (1.0f-exp(-Ttau)) / Ttau ;
         tt     =  (fabs(Ttau)>0.01f) ?  ((1.0f-exp(-Ttau))/Ttau) : (1.0f-Ttau*(0.5f-0.166666667f*Ttau)) ;
         // ttt    = (1.0f-tt)/Ttau
         ttt    =  (1.0f-tt)/Ttau ;
         // Line emission leaving the cell   --- GL in profile
         Lleave =  weight*nu*Aul*flo * tt ;
         // Dust emission leaving the cell 
         Dleave =  Cemit *                     tt ;
         // SIJ updates, first incoming photons then absorbed dust emission
         sij   +=  A_b * flo*GN*dr * NTRUE[i]*tt ;
         // sij         += A_b * profile[i]*GN*Cemit*dr*(1.0f-tt)/Ttau ; // GN includes GL!
         sij   +=  A_b * flo*GN*dr * Cemit*ttt ;    // GN includes GL!
         // "Escaping" line photons = absorbed by dust or leave the cell
         all_escaped +=  Lleave  +  weight*nu*Aul*flo * Ctau * ttt ;
         // Total change of photons in the package
         NTRUE[i]     =  NTRUE[i]*exp(-Ttau) + Dleave + Lleave ;
      }  // loop over channels
      // RES[2*INDEX]    += sij ;            // division by VOLUME done in the solver (kernel)
      // RES[2*INDEX+1]  += all_escaped ;    // divided by VOLUME only oin Solve() !!!
#   if (NO_ATOMICS>0)
      RES[2*INDEX  ]  +=  sij ;
      RES[2*INDEX+1]  +=  all_escaped) ;
#   else
      AADD(&(RES[2*INDEX]), sij) ;
      AADD(&(RES[2*INDEX+1]), all_escaped) ;
#   endif
      
#  else   // not  WITH_CRT ***************************************************************************************
      
      
      // because of c1, the same NTRUE elements may be updated each time by different work items...
      barrier(CLK_LOCAL_MEM_FENCE) ;    // local NTRUE elements possibly updated by different threads
      
      
#   if (WITH_ALI>0) // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      for(int i=lid; i<NCHN; i+=ls)  {
         w               =  tmp_tau*profile[i] ; // profile already covers same NCHN channels as NTRUE
#    if 0
         if (w<1.0e-6f) continue ;
#   endif
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         // factor          =  clamp(factor,  1.0e-30f, 1.0f) ;  // KILL MASERS $$$
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;    // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      SDT[lid] = sum_delta_true ;  AE[lid] = all_escaped ; // all work items save their own results
      barrier(CLK_LOCAL_MEM_FENCE) ;             // all agree on NTRUE, all know SDT and AE
      if (lid==0) {                              // lid=0 sums up and saves absorptions and escaped photons
         for(int i=1; i<LOCAL; i++) {  
            sum_delta_true += SDT[i] ;      all_escaped    +=  AE[i] ;     
         }
#    if 0
         all_escaped     =  clamp(all_escaped, 0.0001f*weight*nu*Aul, 0.9999f*weight*nu*Aul) ; // must be [0,1]
#    endif
         // RES[2*INDEX]   +=  A_b * (sum_delta_true / nb_nb) ;
         // RES[2*INDEX+1] +=  all_escaped ;
         w  =  A_b * (sum_delta_true/nb_nb) ;
#    if (NO_ATOMICS>0)
         RES[2*INDEX  ] +=  w ;
         RES[2*INDEX+1] +=  all_escaped ;
#    else
         AADD(&(RES[2*INDEX]),    w) ;
         AADD(&(RES[2*INDEX+1]),  all_escaped) ;
#    endif
      } // lid==0
      // }      
#   else // else no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      for(int i=lid; i<NCHN; i+=ls)  {
         w               =  tmp_tau*profile[i] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         factor          =  clamp(factor,  1.0e-30f, 1.0f) ;  // KILL MASERS $$$
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;    // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed - escape ;  // later sum_delta_true +=  W*nu*Aul
      }   // over channels
      SDT[lid] = sum_delta_true ;
      barrier(CLK_LOCAL_MEM_FENCE) ;
      if (lid==0) {
         for(int i=1; i<LOCAL; i++)  sum_delta_true += SDT[i] ;    
         w  =   A_b  * ((weight*nu*Aul + sum_delta_true) / nb_nb)  ;
#    if (NO_ATOMICS>0)
         RES[INDEX] +=  w ;
#    else
         AADD((__global float*)(RES+INDEX), w) ;
#    endif
      } 
#   endif // no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
#  endif  // WITH OR WITHOUT CRT
      
      
      // } // RHO>CLIP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
#  if (BRUTE_COOLING>0)
      // total number of photons in the package as it exits the cell
      if (lid==0) {
         float cool = 0.0f ;
         for(int i=0; i<NCHN; i++) cool += NTRUE[i] ;
         COOL[INDEX] += cool ; // cooling of cell INDEX --- each work group distinct rays => no need for atomics
      }
#  endif
      
      
      
      
      
      // Updates at the end of the step, POS has been already updated, OTL and OTI point to the new cell
      INDEX   =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;
      
#  if (BRUTE_COOLING>0)  // heating of the next cell, once INDEX has been updated
      if (INDEX>=0) {
         if (lid==0)  COOL[INDEX] -= cool ; // heating of the next cell
      }
#  endif
      
      if (INDEX>=0) {
         if (RHO[INDEX]<=0.0f) { // we stepped to a refined cell (GetStep goes only to a cell OTL<=OTLO)
            continue ;           // step down one level at the beginning of the main loop
         }
      }
      
      
      if (OTL<RL) {        // we are up to a level where this ray no longer exists
         INDEX=-1 ;        
      } else {      
         if (INDEX<0) {    // ray exits the cloud... possibly continues on the other side
            if (POS.x>=NX  ) {   if (LEADING!=0)  POS.x =    EPS ;   }
            if (POS.x<=ZERO) {   if (LEADING!=1)  POS.x = NX-EPS ;   } 
            if (POS.y>=NY  ) {   if (LEADING!=2)  POS.y =    EPS ;   }
            if (POS.y<=ZERO) {   if (LEADING!=3)  POS.y = NY-EPS ;   } 
            if (POS.z>=NZ  ) {   if (LEADING!=4)  POS.z =    EPS ;   }
            if (POS.z<=ZERO) {   if (LEADING!=5)  POS.z = NZ-EPS ;   } 
            IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;  // we remain in a root-grid cell => OTL==0 !
            INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
            if (INDEX>=0) {   // new level-0 ray started on the opposite side (may be a parent cell)
               RL = 0 ;  OTLO = 0 ; 
               // we had already barrier after the previous NTRUE update
               for(int i=lid; i<NCHN; i+=ls)  NTRUE[i] = BG * DIRWEI ;
               barrier(CLK_LOCAL_MEM_FENCE) ;
#  if (BRUTE_COOLING>0)
               if (lid==0) {
                  dr = BG*DIRWEI*NCHN ;  COOL[INDEX] -= dr ; // heating of the entered cell
               }
#  endif
               continue ;
            }
         } // if INDEX<0
      }
      
      
      // rescale on every change of resolution
      if ((INDEX>=0)&&(OTL<OTLO)) {   // @s ray continues at a lower hierarchy level => NTRUE may have to be scaled
         dr = pown(4.0f, OTLO-OTL) ;  // scale on every change of resolution
         for(int i=lid; i<NCHN; i+=ls)  NTRUE[i] *= dr ;     
         continue ;  // we went to lower level => this cell is a leaf
      }
      
      
      // if INDEX still negative, try to take a new ray from the buffer
      // 0   1     2  3  4  6     ...       NTRUE[NCHN] 
      // OTL OTI   x  y  z  RL    x y z RL                  
      if ((INDEX<0)&&(NBUF>0)) {            // NBUF>0 => at least one ray exists in the buffer
         barrier(CLK_GLOBAL_MEM_FENCE) ;    // all work items access BUFFER
         c1    =  (NBUF-1)*(26+NCHN) ;      // CHANNELS->NCHN elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // OTL ...
         OTLO  =  OTL ;                     // ???
         OTI   =  F2I(BUFFER[c1+1]) ;       // and OTI of the ray that was split
         for(sr=5; sr>=0; sr--) {         
            dr    =  BUFFER[c1+2+4*sr] ;    // read dr
            if (dr>-0.1f) break ;           // found a ray
         }
         POS.x   =  dr ;
         POS.y   =  BUFFER[c1+3+4*sr] ;  
         POS.z   =  BUFFER[c1+4+4*sr] ;
         RL      =  BUFFER[c1+5+4*sr] ;
         SID     =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
         OTI     =  8*(int)(OTI/8) + SID ;        // OTI in the buffer was for the original ray that was split
         INDEX   =  OFF[OTL]+OTI ;                // global index -- must be >=0 !!!
         // copy NTRUE --- values in BUFFER have already been divided by four
         barrier(CLK_GLOBAL_MEM_FENCE) ;          // all have read BUFFER
         if (lid==0)  BUFFER[c1+2+4*sr] = -1.0f ; // mark as used
         if (sr==0)   NBUF -= 1 ;
         c1     +=  26 ;
         for(int i=lid; i<NCHN; i+=ls)  NTRUE[i] = BUFFER[c1+i] ;  // NTRUE correct for level OTL
         barrier(CLK_LOCAL_MEM_FENCE) ;         
         // note - this ray be inside a parent cell => handled at the beginnign of the main loop
      } // (INDEX<=0)&(NBUF>0)
      
      
   } // while INDEX>=0
   
   
}  // end of UpdateHF4()


# endif // OCTREE==4

#endif // WITH_HFS














#if (WITH_OCTREE==0)



__kernel void Paths(  // 
                      const int  id0,
# if (PLWEIGHT>0)
                      __global float   *PL,      // [CELLS]
# endif
                      __global float   *TPL,     // [NRAY]
                      __global int     *COUNT,   // [NRAY] total number of rays entering the cloud
                      const    int      LEADING, // leading edge
                      const    REAL3    POS0,    // initial position of ray 0
                      const    float3   DIR      // direction of the rays
                   ) {   
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge, if ray exits through a side, a new one is created 
   // on the opposite side. Ray ends when the downstream edge is reached.
   // Update() does not use PL ==> PL should be identical for all the cells !!
   int   id  =  get_global_id(0) ;   
   id += id0 ;
   if    (id>=NRAY) return ;  
   int   INDEX,  nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ;   
   float tpl=0.0f ;
# if (NX>DIMLIM)
   double dx, dy, dz ;
# else
   float dx, dy, dz ;
# endif
   REAL3 POS = POS0 ;
# if (FIX_PATHS>0)
   REAL3 POS_LE ;
   int count = 0 ;
# endif
   
# if (ONESHOT<1)
   switch (LEADING) {
    case 0: POS.x =    EPS ;  POS.y += TWO*(id%ny) ;  POS.z += TWO*(int)(id/ny) ;  break ;
    case 1: POS.x = NX-EPS ;  POS.y += TWO*(id%ny) ;  POS.z += TWO*(int)(id/ny) ;  break ;
    case 2: POS.y =    EPS ;  POS.x += TWO*(id%nx) ;  POS.z += TWO*(int)(id/nx) ;  break ;
    case 3: POS.y = NY-EPS ;  POS.x += TWO*(id%nx) ;  POS.z += TWO*(int)(id/nx) ;  break ;
    case 4: POS.z =    EPS ;  POS.x += TWO*(id%nx) ;  POS.y += TWO*(int)(id/nx) ;  break ;
    case 5: POS.z = NZ-EPS ;  POS.x += TWO*(id%nx) ;  POS.y += TWO*(int)(id/nx) ;  break ;
   }
# else   // all rays, not only every other
   switch (LEADING) {
    case 0: POS.x =    EPS ;  POS.y += (id%NY) ;  POS.z += (int)(id/NY) ;  break ;
    case 1: POS.x = NX-EPS ;  POS.y += (id%NY) ;  POS.z += (int)(id/NY) ;  break ;
    case 2: POS.y =    EPS ;  POS.x += (id%NX) ;  POS.z += (int)(id/NX) ;  break ;
    case 3: POS.y = NY-EPS ;  POS.x += (id%NX) ;  POS.z += (int)(id/NX) ;  break ;
    case 4: POS.z =    EPS ;  POS.x += (id%NX) ;  POS.y += (int)(id/NX) ;  break ;
    case 5: POS.z = NZ-EPS ;  POS.x += (id%NX) ;  POS.y += (int)(id/NX) ;  break ;
   }
# endif
   
   
# if (FIX_PATHS>0)
   POS_LE = POS ;
# endif
   INDEX     = Index(POS) ;
   COUNT[id] = (INDEX>=0) ? 1 : 0 ;
   float distance = 0.0f ;
   
   while(INDEX>=0) {
      
      
# if (NX>DIMLIM) // ====================================================================================================
      double dx, dy, dz ;
      dx = (DIR.x>0.0f)  ?  ((1.0+DEPS-fmod(POS.x,ONE))/DIR.x)  :  ((-DEPS-fmod(POS.x,ONE))/DIR.x) ;
      dy = (DIR.y>0.0f)  ?  ((1.0+DEPS-fmod(POS.y,ONE))/DIR.y)  :  ((-DEPS-fmod(POS.y,ONE))/DIR.y) ;
      dz = (DIR.z>0.0f)  ?  ((1.0+DEPS-fmod(POS.z,ONE))/DIR.z)  :  ((-DEPS-fmod(POS.z,ONE))/DIR.z) ;
      dx =  min(dx, min(dy, dz)) ;
# else
      dx=        (DIR.x<0.0f) ? (-fmod(POS.x,ONE)/DIR.x-EPS/DIR.x) : ((ONE-fmod(POS.x,ONE))/DIR.x+EPS/DIR.x) ;
      dx= min(dx,(DIR.y<0.0f) ? (-fmod(POS.y,ONE)/DIR.y-EPS/DIR.y) : ((ONE-fmod(POS.y,ONE))/DIR.y+EPS/DIR.y)) ;
      dx= min(dx,(DIR.z<0.0f) ? (-fmod(POS.z,ONE)/DIR.z-EPS/DIR.z) : ((ONE-fmod(POS.z,ONE))/DIR.z+EPS/DIR.z)) ;
# endif
      
      
      distance += dx ;
# if (PLWEIGHT>0)
#  if (NO_ATOMICS>0)
      PL[INDEX] += dx ;           // path length, cumulative over idir and ioff
#  else
      AADD(&(PL[INDEX]), dx) ;
#  endif
# endif      
      tpl       += dx ;           // just the total value for current idir, ioff
      POS.x     += dx*DIR.x ;  POS.y += dx*DIR.y ;  POS.z += dx*DIR.z ;
      
      
# if (FIX_PATHS>0)
      // try to precise the position
      count += 1 ;
      if (count%7==2) {
         if (LEADING<2) {
            float s =  (LEADING==0) ?  (POS.x/DIR.x) : ((POS.x-NX)/DIR.x) ;
            POS.y   =  POS_LE.y + s*DIR.y ;
            POS.z   =  POS_LE.z + s*DIR.z ;
            if (POS.y<ZERO) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;
            if (POS.z<ZERO) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
         } else {
            if (LEADING<4) {
               float s =  (LEADING==2) ?  (POS.y/DIR.y) : ((POS.y-NY)/DIR.y) ;
               POS.x   =  POS_LE.x + s*DIR.x ;
               POS.z   =  POS_LE.z + s*DIR.z ;
               if (POS.x<ZERO) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
               if (POS.z<ZERO) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
            } else {
               float s =  (LEADING==4) ?  (POS.z/DIR.z) : ((POS.z-NY)/DIR.z) ;
               POS.x   =  POS_LE.x + s*DIR.x ;
               POS.y   =  POS_LE.y + s*DIR.y ;
               if (POS.x<ZERO) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
               if (POS.y<ZERO) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;            
            }
         }
      }
# endif
      
      
      
      
      INDEX      = Index(POS) ;
      if (INDEX<0) {  // exits the cloud... but on which side?
         if (POS.x>=NX)   {  if (LEADING!=0)  POS.x =    EPS ; }   // create new ray
         if (POS.x<=ZERO) {  if (LEADING!=1)  POS.x = NX-EPS ; }   // create new ray
         if (POS.y>=NY)   {  if (LEADING!=2)  POS.y =    EPS ; }   // create new ray
         if (POS.y<=ZERO) {  if (LEADING!=3)  POS.y = NY-EPS ; }   // create new ray
         if (POS.z>=NZ)   {  if (LEADING!=4)  POS.z =    EPS ; }   // create new ray
         if (POS.z<=ZERO) {  if (LEADING!=5)  POS.z = NZ-EPS ; }   // create new ray
         INDEX = Index(POS) ;
         if (INDEX>=0)  COUNT[id] += 1 ;         // ray re-enters
      }
   } // while INDEX>=0
   TPL[id] = tpl ;   
   // Each ray travel exactly the same distance... but some PL entries are still smaller???
   // .... because rays still  were interfering with each other (rounding errors?)
   //      ==> generate rays at steps of three cells and let host loop over 3x3 offset positions??
   // NO !!! even when rays start in every third cell, problem persists --- need to use atomic updates ????
   // Less problems on CPU, random from run to the next !!!
   // Atomic works on GPU  --  CPU build does not even complete !!!
# if 0
   if ((fabs(distance)>0.001f)&&(fabs(distance-37.5271f)>0.001)) {
      printf("distance  %8.4f\n", distance) ;
   }
# endif
   
   // barrier(CLK_LOCAL_MEM_FENCE) ;
   // barrier(CLK_GLOBAL_MEM_FENCE) ;
   
}




__kernel void Update(   //  @u  Cartesian grid, PL not used, only APL
                        const int id0,             //  0 
# if (WITH_HALF==1)
                        __global short4 *CLOUD,    //  1 [CELLS]: vx, vy, vz, sigma
# else
                        __global float4 *CLOUD,    //  1 [CELLS]: vx, vy, vz, sigma
# endif
                        GAUSTORE  float *GAU,      //  2 precalculated gaussian profiles [GNO,CHANNELS]
                        constant int2   *LIM,      //  3 limits of ~zero profile function [GNO]
                        const float      Aul,      //  4 Einstein A(upper->lower)
                        const float      A_b,      //  5 (g_u/g_l)*B(upper->lower)
                        const float      GN,       //  6 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                        __global float  *PL,       //  7 just for testing
                        const float      APL,      //  8 average path length [GL]
                        const float      BG,       //  9 background value (photons)
                        const float      DIRWEI,   // 10  <cos(theta)> for rays entering leading edge
                        const float      EWEI,     // 11  1/<1/cosT>/NDIR
                        const int        LEADING,  // 12 leading edge
                        const REAL3      POS0,     // 13 initial position of id=0 ray
                        const float3     DIR,      // 14 ray direction
                        __global float  *NI,       // 15 [CELLS]:  NI[upper] + NB_NB
                        __global float  *RES,      // 16 [CELLS]:  SIJ, ESC
                        __global float  *NTRUES    // 17 [GLOBAL*MAXCHN]
# if (WITH_CRT>0)
                        ,constant float *CRT_TAU,  // 18 dust optical depth / GL
                        constant float *CRT_EMI    // 19 dust emission photons/c/channel/H
# endif                     
# if (BRUTE_COOLING>0)
                        ,__global float *COOL      // 14,15 [CELLS] = cooling 
# endif
                    )  {
   float weight ;        
   float dx, doppler, w ;
# if 0 // DOUBLE
   double  wd, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;
# else
   float  tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed ;
# endif
   
   float sum_delta_true, all_escaped, nu ;
   int row, shift, INDEX, c1, c2 ;
   // Each work item processes one ray, rays are two cells apart (ONESHOT==0) to avoid need for synchronisation
   // Rays start on the leading edge. If ray exits through a side, a new one is created 
   // on the opposite side and the ray ends when the downstream border is reached.
   int  id = get_global_id(0) ;
   int lid = get_local_id(0) ;
   
# if (LOC_LOWMEM>0)
   __global float *NTRUE = &NTRUES[id*CHANNELS] ;
# else    // this is ok for CPU
   __local float  NTRUESSS[LOCAL*CHANNELS] ;
   __local float *NTRUE = &NTRUESSS[lid*CHANNELS] ;
# endif
   
   id += id0 ;
   if (id>=NRAY) return ;
   
# if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, pro, sij ;
# endif
# if (ONESHOT<1)
   int nx = (NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ; // dimensions of the current ray grid
# endif
   GAUSTORE float *profile ;
   
   
   // Initial position of each ray shifted by two grid units == host has a loop over 4 offset positions
   REAL3 POS = POS0 ;
# if (FIX_PATHS>0)
   REAL3 POS_LE ;
   int count = 0 ;
# endif
   
   
   
# if (ONESHOT<1)
   switch (LEADING) {
    case 0:   POS.x =    EPS ;  POS.y += TWO*(id%ny) ;   POS.z += TWO*(int)(id/ny) ;      break ;
    case 1:   POS.x = NX-EPS ;  POS.y += TWO*(id%ny) ;   POS.z += TWO*(int)(id/ny) ;      break ;
    case 2:   POS.y =    EPS ;  POS.x += TWO*(id%nx) ;   POS.z += TWO*(int)(id/nx) ;      break ;
    case 3:   POS.y = NY-EPS ;  POS.x += TWO*(id%nx) ;   POS.z += TWO*(int)(id/nx) ;      break ;
    case 4:   POS.z =    EPS ;  POS.x += TWO*(id%nx) ;   POS.y += TWO*(int)(id/nx) ;      break ;
    case 5:   POS.z = NZ-EPS ;  POS.x += TWO*(id%nx) ;   POS.y += TWO*(int)(id/nx) ;      break ;
   }
# else
   switch (LEADING) {
    case 0:   POS.x =    EPS ;  POS.y += (id%NY) ;   POS.z += (int)(id/NY) ;      break ;
    case 1:   POS.x = NX-EPS ;  POS.y += (id%NY) ;   POS.z += (int)(id/NY) ;      break ;
    case 2:   POS.y =    EPS ;  POS.x += (id%NX) ;   POS.z += (int)(id/NX) ;      break ;
    case 3:   POS.y = NY-EPS ;  POS.x += (id%NX) ;   POS.z += (int)(id/NX) ;      break ;
    case 4:   POS.z =    EPS ;  POS.x += (id%NX) ;   POS.y += (int)(id/NX) ;      break ;
    case 5:   POS.z = NZ-EPS ;  POS.x += (id%NX) ;   POS.y += (int)(id/NX) ;      break ;
   }
# endif
   
   
# if (FIX_PATHS>0)
   POS_LE = POS ;
# endif
   
   INDEX = Index(POS) ;
   // BG       =  average number of photons per ray
   // DIRWEI   =  cos(theta) / <cos(theta)>,   weight for current direction relative to average
   for(int i=0; i<CHANNELS; i++) NTRUE[i] = BG * DIRWEI ; // DIRWEI ~ cos(theta) / <cos(theta)>
   
# if (BRUTE_COOLING>0)
   float cool = BG*DIRWEI*CHANNELS ;
   if (INDEX>=0) {
#  if (NO_ATOMICS>0)
      COOL[INDEX] -= cool ;
#  else
      AADD(&(COOL[INDEX]), -cool) ; // heating of the entered cell
#  endif
   }
# endif
   
   
   // printf("%d: %8.4f %8.4f %8.4f    %9.5f %9.5f %9.5f   %7d\n", LEADING, DIR.x, DIR.y, DIR.z, POS.x, POS.y, POS.z, INDEX) ;
   
   
   while(INDEX>=0) {
      
      
# if (NX>DIMLIM) // ====================================================================================================
      double dx, dy, dz ;
      dx = (DIR.x>0.0f)  ?  ((1.0+DEPS-fmod(POS.x,ONE))/DIR.x)  :  ((-DEPS-fmod(POS.x,ONE))/DIR.x) ;
      dy = (DIR.y>0.0f)  ?  ((1.0+DEPS-fmod(POS.y,ONE))/DIR.y)  :  ((-DEPS-fmod(POS.y,ONE))/DIR.y) ;
      dz = (DIR.z>0.0f)  ?  ((1.0+DEPS-fmod(POS.z,ONE))/DIR.z)  :  ((-DEPS-fmod(POS.z,ONE))/DIR.z) ;
      dx =  min(dx, min(dy, dz)) ;
# else
      dx=        (DIR.x<0.0f) ? (-fmod(POS.x,ONE)/DIR.x-EPS/DIR.x) : ((ONE-fmod(POS.x,ONE))/DIR.x+EPS/DIR.x) ;
      dx= min(dx,(DIR.y<0.0f) ? (-fmod(POS.y,ONE)/DIR.y-EPS/DIR.y) : ((ONE-fmod(POS.y,ONE))/DIR.y+EPS/DIR.y)) ;
      dx= min(dx,(DIR.z<0.0f) ? (-fmod(POS.z,ONE)/DIR.z-EPS/DIR.z) : ((ONE-fmod(POS.z,ONE))/DIR.z+EPS/DIR.z)) ;
# endif
      
      nu        =  NI[2*INDEX]  ;
      nb_nb     =  NI[2*INDEX+1] ;
      
      // As long as the raytracing is working, expected length per root grid cell is
      //    offs*offs * sum_over_NDIR (1.0/cos(theta))  ....  this is APL
      //    APL =  offs*oggs   *    sum_over_NDIR( 1.0/DIR.leading )
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# if (WITH_HALF==1)
      doppler  *=  0.002f ;  // half integer times 0.002f km/s
# endif
      shift     =  round(doppler/WIDTH) ;
# if (WITH_HALF==1)
      //               sigma = 0.002f * w
      // lookup table: sigma = SIGMA0 * SIGMAX^row
      row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      // if (id==0) printf("  sigma %6.2f  --- row %d/%d\n", CLOUD[INDEX].w*0.002f, row, GNO) ;
# else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
# endif
      profile   =  &GAU[row*CHANNELS] ;
      // avoid profile function outside profile channels LIM.x, LIM.y
      c1        =  max(LIM[row].x+shift, max(0, shift)) ;
      c2        =  min(LIM[row].y+shift, min(CHANNELS-1, CHANNELS-1+shift)) ;
      
      weight    =  (dx/APL)*VOLUME ;                    // correct !!   .. NDIR=0, weight==1.0/6.0
      tmp_tau   =   dx*nb_nb*GN ;
      if (fabs(tmp_tau)<1.0e-32f) tmp_tau = 1.0e-32f ;  // was e-32
      tmp_emit  =  weight*nu*(Aul/tmp_tau) ;            // GN include grid length [cm]
      
      sum_delta_true =  0.0f ;
      all_escaped    =  0.0f ;
      
# if (WITH_CRT>0)
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
         tt     =  (fabs(Ttau)>0.01f) ?  ((1.0f-exp(-Ttau))/Ttau) : (1.0f-Ttau*(0.5f-0.166666667f*Ttau)) ;
         // ttt    = (1.0f-tt)/Ttau
#  if 1
         ttt    =  (1.0f-tt)/Ttau ;
#  else
         ttt    =  (fabs(Ttau)>0.01f) ?  ((1.0f-tt)/Ttau)  : (0.5f-0.166666667f*Ttau) ;
#  endif
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
      
      
      
# else  // not CRT 
      
      
      
#  if  (WITH_ALI>0)      
      for(int ii=c1; ii<=c2; ii++)  {
         w               =  tmp_tau*profile[ii-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell .... tmp_emit  =  weight*nu*(Aul/tmp_tau)
         absorbed        =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]      +=  escape-absorbed ;         
         sum_delta_true +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      // Update SIJ and ESC
      // without atomics, large dTex in individual cells
#   if (NO_ATOMICS>0)
      RES[2*INDEX  ]  +=  A_b*(sum_delta_true/nb_nb) ; // parentheses!
      RES[2*INDEX+1]  +=  all_escaped ;
#   else
      AADD(&(RES[2*INDEX  ]),  A_b*(sum_delta_true/nb_nb)) ; // parentheses!
      AADD(&(RES[2*INDEX+1]),  all_escaped) ;
#   endif
#  else  // ELSE NO ALI
      
#   if 0 // DOUBLE  ... absolutely no effect on optical thin Tex !!
      for(int ii=c1; ii<=c2; ii++)  {
         wd              =  tmp_tau*profile[ii-shift] ;
         factor          =  1.0f-exp(-wd) ;
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell .... tmp_emit=weight*nu*(Aul/tmp_tau)
         absorbed        =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]      +=  escape-absorbed ;
         sum_delta_true +=  absorbed - escape ;  // later absorbed ~ W*nu*Aul - escape
         // all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }  // over channels
      w    =   A_b * (  (weight*nu*Aul  + sum_delta_true) / nb_nb )  ;
#   else
      for(int ii=c1; ii<=c2; ii++)  {
         w               =  tmp_tau*profile[ii-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell .... tmp_emit=weight*nu*(Aul/tmp_tau)
         absorbed        =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]      +=  escape-absorbed ;
         sum_delta_true +=  absorbed - escape ;  // later absorbed ~ W*nu*Aul - escape
         // all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }  // over channels
      w    =   A_b * (  (weight*nu*Aul  + sum_delta_true) / nb_nb )  ;
#   endif
      
      // printf("    dx = %7.4f  ... EPS = %.3e\n", dx, EPS) ;
#   if (NO_ATOMICS>0)
      RES[INDEX] += w ;
#   else
      AADD((__global float*)(RES+INDEX),   w) ;   // Sij counter update
#   endif
      
      // printf("dx = %8.4f  w = %12.4e  w/dx %12.4e  W = %12.4e  BG = %12.4e\n", dx, w, w/dx, RES[INDEX], BG) ;
      
#  endif
      
      
# endif  // not CRT
      
      
      POS.x += dx*DIR.x ;  POS.y += dx*DIR.y ;  POS.z += dx*DIR.z ;
      
      
# if 0  // testing, is the path length the same as in Paths
      AADD(&(PL[INDEX]), -dx) ;
# endif
      
      
# if (FIX_PATHS>0)
      not used
        // try to precise the position
        count += 1 ;
      if (count%7==2) {
         if (LEADING<2) {
            float s =  (LEADING==0) ?  (POS.x/DIR.x) : ((POS.x-NX)/DIR.x) ;
            POS.y   =  POS_LE.y + s*DIR.y ;
            POS.z   =  POS_LE.z + s*DIR.z ;
            if (POS.y<ZERO) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;
            if (POS.z<ZERO) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
         } else { 
            if (LEADING<4) {
               float s =  (LEADING==2) ?  (POS.y/DIR.y) : ((POS.y-NY)/DIR.y) ;
               POS.x   =  POS_LE.x + s*DIR.x ;
               POS.z   =  POS_LE.z + s*DIR.z ;
               if (POS.x<ZERO) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
               if (POS.z<ZERO) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
            } else {
               float s =  (LEADING==4) ?  (POS.z/DIR.z) : ((POS.z-NY)/DIR.z) ;
               POS.x   =  POS_LE.x + s*DIR.x ;
               POS.y   =  POS_LE.y + s*DIR.y ;
               if (POS.x<ZERO) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
               if (POS.y<ZERO) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;            
            }
         }
      }
# endif
      
      
      
# if (BRUTE_COOLING>0)
      // total number of photons in the package as it exits the cell
      float cool = 0.0f ;
      for(int ii=0; ii<CHANNELS; ii++) cool += NTRUE[ii] ;
#  if (NO_ATOMICS>0)
      COOL[INDEX] += cool ;
#  else
      AADD(&(COOL[INDEX]), cool) ; // cooling of cell INDEX
#  endif
# endif
      
      INDEX      = Index(POS) ;
      
# if (BRUTE_COOLING>0)
      if (INDEX>=0) {
#  if (NO_ATOMICS>0)
         COOL[INDEX] -= cool ;
#  else
         AADD(&(COOL[INDEX]), -cool) ; // heating of the next cell
#  endif
      }
# endif
      
      
      if (INDEX<0) {  
         // exits the cloud... but on which side?
         // even when the ray is now entering via non-leading edge, DIRWEI is still the
         // same (the rays will hit the side walls at correspondingly larger intervals)
         if (POS.x>=NX)   {   if (LEADING!=0)    POS.x =    EPS ;    } ;
         if (POS.x<=ZERO) {   if (LEADING!=1)    POS.x = NX-EPS ;    } ;
         if (POS.y>=NY)   {   if (LEADING!=2)    POS.y =    EPS ;    } ;
         if (POS.y<=ZERO) {   if (LEADING!=3)    POS.y = NY-EPS ;    } ;
         if (POS.z>=NZ)   {   if (LEADING!=4)    POS.z =    EPS ;    } ;
         if (POS.z<=ZERO) {   if (LEADING!=5)    POS.z = NZ-EPS ;    } ;
         INDEX = Index(POS) ;
         if (INDEX>=0) {   // new ray started on the opposite side (same work item)
            // printf("SIDERAY !!!!!!!\n") ;
            for(int ii=0; ii<CHANNELS; ii++) NTRUE[ii] = BG * DIRWEI ;
# if (BRUTE_COOLING>0)
            float cool = BG*DIRWEI*CHANNELS ;
#  if (NO_ATOMICS>0)
            COOL[INDEX] -= cool ;
#  else
            AADD(&(COOL[INDEX]), -cool) ; // heating of the entered cell
#  endif
# endif
            
         }
      } // if INDEX<0
      
   } // while INDEX>=0
   
}


#endif  // WITH_OCTREE==0













#if (WITH_OCTREE==1)


__kernel void PathsOT1(__global   float   *PL,      // 
                       __global   float   *TPL,     // [NRAY]
                       __global   int     *COUNT,   // [NRAY] total number of rays entering the cloud
                       const      int      LEADING, // leading edge
                       const      REAL3    POS0,    // initial position of ray 0
                       const      float3   DIR,     // direction of the rays
                       __global   int     *LCELLS,   
                       __constant int     *OFF,
                       __global   int     *PAR,
                       __global   float   *RHO
                      ) {   
   // OT1 = no ray splitting
   int  id = get_global_id(0) ;
   if   (id>=NRAY) return ;  // one work item per ray,  NRAY <= GLOBAL
   int   count=0,  INDEX, nx=(NX+1)/2, ny=(NY+1)/2,  nz=(NZ+1)/2, OTL, OTI ;
   float tpl=0.0f, dx ;
   REAL3 POS = POS0 ;
# if (FIX_PATHS>0)
   REAL3 POS_LE ;
# endif
   
   switch (LEADING) {
    case 0:  POS.x =    EPS ;  POS.y += TWO*(id%ny) ;   POS.z += TWO*(int)(id/ny) ;   break ;
    case 1:  POS.x = NX-EPS ;  POS.y += TWO*(id%ny) ;   POS.z += TWO*(int)(id/ny) ;   break ;
    case 2:  POS.y =    EPS ;  POS.x += TWO*(id%nx) ;   POS.z += TWO*(int)(id/nx) ;   break ;
    case 3:  POS.y = NY-EPS ;  POS.x += TWO*(id%nx) ;   POS.z += TWO*(int)(id/nx) ;   break ;
    case 4:  POS.z =    EPS ;  POS.x += TWO*(id%nx) ;   POS.y += TWO*(int)(id/nx) ;   break ;
    case 5:  POS.z = NZ-EPS ;  POS.x += TWO*(id%nx) ;   POS.y += TWO*(int)(id/nx) ;   break ;
   }   
# if (FIX_PATHS>0)
   POS_LE = POS ;
# endif
   IndexG(&POS, &OTL, &OTI, RHO, OFF) ;  // go directly to a leaf node
   INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
   if (INDEX>=0) count++ ;
   while(INDEX>=0) {
# if (FIX_PATHS>0)
      dx      =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, 99, &POS_LE, LEADING) ; // step [GL] == root grid units !!
# else
      dx      =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, 99, NULL   , LEADING) ; // step [GL] == root grid units !!
# endif
      tpl    +=  dx ;
# if (NO_ATOMICS>0)
      PL[INDEX] += dx ;
# else
      AADD(&(PL[INDEX]), dx) ;
# endif
      INDEX   =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;  // OTL, OTI, POS already updated = end of the current step
      if (INDEX<0) {    // ray exits the cloud on root grid, possibly create a new OTL=0 ray on the other side
         if (POS.x>=NX  ) {   if (LEADING!=0)  POS.x =    EPS ;   }
         if (POS.x<=ZERO) {   if (LEADING!=1)  POS.x = NX-EPS ;   }
         if (POS.y>=NY  ) {   if (LEADING!=2)  POS.y =    EPS ;   }
         if (POS.y<=ZERO) {   if (LEADING!=3)  POS.y = NY-EPS ;   } 
         if (POS.z>=NZ  ) {   if (LEADING!=4)  POS.z =    EPS ;   }
         if (POS.z<=ZERO) {   if (LEADING!=5)  POS.z = NZ-EPS ;   } 
         IndexG(&POS, &OTL, &OTI, RHO, OFF) ; // go directly to the leaf node
         INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
         if (INDEX>=0) {   // new level-0 ray started on the opposite side
            count += 1 ;
            continue ;
         }
      }
   } // while INDEX>=0  --- stops when buffer is empty and the main level-0 ray has exited
   TPL[id]   = tpl ;       // TPL_buf[NRAY]  
   COUNT[id] = count ;     // COUNT_buf[NRAY]
   
   
}





__kernel void UpdateOT1( // 
# if (PLWEIGHT>0)
                         __global float *PL,
# endif
# if (WITH_HALF==1)
                         __global short4 *CLOUD,   //  0 [CELLS]: vx, vy, vz, sigma
# else
                         __global float4 *CLOUD,   //  0 [CELLS]: vx, vy, vz, sigma
# endif
                         GAUSTORE  float *GAU,     //  1 precalculated gaussian profiles [GNO,CHANNELS]
                         constant int2   *LIM,     //  2 limits of ~zero profile function [GNO]
                         const float      Aul,     //  3 Einstein A(upper->lower)
                         const float      A_b,     //  4 (g_u/g_l)*B(upper->lower)
                         const float      GN,      //  5 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                         const float      APL,     //  6 average path length [GL]
                         const float      BG,      //  7 background value (photons)
                         const float      DIRWEI,  //  8 weight factor cos(theta)/<cos(theta)>
                         const float      EWEI,    //  9 weight 1/<1/cosT>/NDIR
                         const int        LEADING, // 10 leading edge
                         const REAL3      POS0,    // 11 initial position of id=0 ray
                         const float3     DIR,     // 12 ray direction
                         __global float  *NI,      // 13 [CELLS]:  NI[upper] + NB_NB
                         __global float  *RES,     // 14 [CELLS]:  SIJ, ESC
                         __global float  *NTRUES   // 15 [GLOBAL*MAXCHN]
# if (WITH_CRT>0)
                         ,constant float *CRT_TAU, //  dust optical depth / GL
                         constant float *CRT_EMI   //  dust emission photons/c/channel/H
# endif                     
# if (BRUTE_COOLING>0)
                         ,__global float *COOL     // 14,15 [CELLS] = cooling 
# endif
                         ,__global   int   *LCELLS   //  16
                         ,__constant int   *OFF      //  17
                         ,__global   int   *PAR      //  18
                         ,__global   float *RHO      //  19  -- needed only to describe the hierarchy
                       )  {
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge. If ray exits through a side, a new one is created 
   // on the opposite side and the ray ends when the downstream edge is reached.
   int   id  =  get_global_id(0) ;
   if   (id>=NRAY) return ;                      // unlike UpdateOT2, work item per ray
   int   lid =  get_local_id(0) ;
   float weight, w, dx, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed, sum_delta_true, all_escaped, nu ;
   int   row, shift, c1, c2, INDEX, nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2, OTL, OTI, OTLO ;
   
   GAUSTORE float *profile ;   
# if (LOC_LOWMEM>0)
   __global float *NTRUE = &NTRUES[id*CHANNELS] ;
# else    // this is ok for CPU
   __local float   NTRUESSS[LOCAL*CHANNELS] ;
   __local float  *NTRUE = &NTRUESSS[lid*CHANNELS] ;
# endif
   
# if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, pro, sij ;
# endif
   
   // Initial position of each ray shifted by two grid units
   //  == host has a loop over 4 offset positions
   REAL3 POS = POS0 ;
# if (FIX_PATHS>0)
   REAL3 POS_LE ;
# endif
   switch (LEADING) {
    case 0:  POS.x =    EPS ;  POS.y += TWO*(id%ny) ;   POS.z += TWO*(int)(id/ny) ;   break ;
    case 1:  POS.x = NX-EPS ;  POS.y += TWO*(id%ny) ;   POS.z += TWO*(int)(id/ny) ;   break ;
    case 2:  POS.y =    EPS ;  POS.x += TWO*(id%nx) ;   POS.z += TWO*(int)(id/nx) ;   break ;
    case 3:  POS.y = NY-EPS ;  POS.x += TWO*(id%nx) ;   POS.z += TWO*(int)(id/nx) ;   break ;
    case 4:  POS.z =    EPS ;  POS.x += TWO*(id%nx) ;   POS.y += TWO*(int)(id/nx) ;   break ;
    case 5:  POS.z = NZ-EPS ;  POS.x += TWO*(id%nx) ;   POS.y += TWO*(int)(id/nx) ;   break ;
   }
# if (FIX_PATHS>0)
   POS_LE = POS ;
# endif
   // BG     =  average number of photons per ray
   // DIRWEI =  cos(theta)/<cos(theta)>, used forto scale the total number of photons entering per ray = BG
   for(int i=0; i<CHANNELS; i++)  NTRUE[i] =  BG * DIRWEI ;
   
   IndexG(&POS, &OTL, &OTI, RHO, OFF) ;          // packet indices for the start position
   INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  // global index to data vectors
   
# if (BRUTE_COOLING>0)
   float cool = BG*DIRWEI*CHANNELS ;
   if (INDEX>=0) {
#  if (NO_ATOMICS>0)
      COOL[INDEX] -= cool ;
#  else
      AADD(&(COOL[INDEX]), -cool) ; // heating of the entered cell
#  endif
   }
# endif
   
   
   while(INDEX>=0) {
      
      OTLO      =  OTL ;
# if (FIX_PATHS>0)
      dx        =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, 99, &POS_LE, LEADING) ;
# else
      dx        =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, 99, NULL   , LEADING) ;
# endif
      // POS is now the end of this step
      // INDEX is still the index for the cell where the step starts
      nu        =  NI[2*INDEX] ;
      nb_nb     =  NI[2*INDEX+1] ;
      
      
# if 0  //  double check PL calculation ... PL[:] should be reduced to zero (plot_pl.py)
      AADD(&(PL[INDEX]), -dx) ;
# endif
      
      
      
      
      
      
      
      
      
      // if (RHO[INDEX]>CLIP) { // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         
      
      // host post scaling is good... Tex ~ 22.485 +- 0.000 for offs = 2
      //                              Tex ~ 22.489 +- 0.001 for offs = 1 !!!
      //   sum(dx) should be  APL/8**l,  path length is simply proportional to the volume
      //   weight =  dx/(APL/8**l)  * (VOLUME/8**l)  = dx/APL * VOLUME
      weight    =  (dx/APL)*VOLUME ;   // dx already proportion to the cell size
      tmp_tau   =  dx*nb_nb*GN ;
      if (fabs(tmp_tau)<1.0e-32f) tmp_tau = 1.0e-32f ;  // allow masering ??
      tmp_emit  =  weight * nu * (Aul / tmp_tau) ;        // GN include grid length [cm]
      
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# if (WITH_HALF==1)
      doppler  *=  0.002f ;  // half integer times 0.002f km/s
# endif                 
      shift     =  round(doppler/WIDTH) ;
# if (WITH_HALF==1)      //               sigma = 0.002f * w,   lookup table: sigma = SIGMA0 * SIGMAX^row
      row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
# else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
# endif
      profile   =  &GAU[row*CHANNELS] ;
      c1        =  max(LIM[row].x+shift, max(0, shift)) ;
      c2        =  min(LIM[row].y+shift, min(CHANNELS-1, CHANNELS-1+shift)) ;
      sum_delta_true = 0.0f ;
      all_escaped    = 0.0f ;
      
# if (WITH_CRT>0)
      sij = 0.0f ;
      // Dust optical depth and emission
      //   here escape = line photon exiting the cell + line photons absorbed by dust
      Ctau      =  dx     * CRT_TAU[INDEX] ;
      Cemit     =  weight * CRT_EMI[INDEX] ;      
      for(int i=c1; i<=c2; i++)  {
         pro    =  profile[i-shift] ;
         Ltau   =  tmp_tau*pro ;
         Ttau   =  Ctau + Ltau ;
         // tt     =  (1.0f-exp(-Ttau)) / Ttau ;
         tt     =  (fabs(Ttau)>0.01f) ?  ((1.0f-exp(-Ttau))/Ttau) : (1.0f-Ttau*(0.5f-0.166666667f*Ttau)) ;
         // ttt    = (1.0f-tt)/Ttau
         ttt    =  (1.0f-tt)/Ttau ;
         // Line emission leaving the cell   --- GL in profile
         Lleave =  weight*nu*Aul*pro * tt ;
         // Dust emission leaving the cell 
         Dleave =  Cemit *                     tt ;
         // SIJ updates, first incoming photons then absorbed dust emission
         sij   +=  A_b * pro*GN*dx * NTRUE[i]*tt ;
         // sij         += A_b * profile[i]*GN*Cemit*dx*(1.0f-tt)/Ttau ; // GN includes GL!
         sij   +=  A_b * pro*GN*dx * Cemit*ttt ;    // GN includes GL!
         // "Escaping" line photons = absorbed by dust or leave the cell
         all_escaped +=  Lleave  +  weight*nu*Aul*pro * Ctau * ttt ;
         // Total change of photons in the package
         NTRUE[i]     =  NTRUE[i]*exp(-Ttau) + Dleave + Lleave ;
      }  // loop over channels
      // Update SIJ and ESC (note - there may be two updates, one incoming, one outgoing ray)
      // RES[INDEX].x   += sij/VOLUME * pow(8.0f,OTL) ;  // octree 8^OTL done in the solver (host)
      RES[2*INDEX]    += sij ;   // division by VOLUME done in the solver (kernel)
      // Emission ~ path length dx but also weighted according to direction, works because <WEI>==1.0
      RES[2*INDEX+1]  += all_escaped ;    // divided by VOLUME only oin Solve() !!!
# else  // else -- not CRT
#  if  (WITH_ALI>0)
      for(int i=c1; i<=c2; i++)  {
         // factor           =  1.0f-native_exp(-(double)tmp_tau*profile[ii-shift]) ;
         w               =  tmp_tau*profile[i-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;    // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      // RES[2*INDEX]   +=   A_b* sum_delta_true / nb_nb ;  // division by VOLUME now in the solver
      // RES[2*INDEX+1] +=   all_escaped ;
      // parentheses are important, otherwise A_b*sum_delta_true/nb_nb may underflow !!!
      w  =  A_b * (sum_delta_true/nb_nb) ;
      AADD(&RES[2*INDEX  ],  w) ;
      AADD(&RES[2*INDEX+1],  all_escaped) ;
#  else // else -- not ALI
      for(int i=c1; i<=c2; i++)  {
         w               =  tmp_tau*profile[i-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         escape          =  tmp_emit*factor ;     // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;     // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed - escape ;   // later absorbed ~ W*nu*Aul - escape
         all_escaped    +=  escape ;              // sum of escaping photons over the profile
      }   // over channels
      w    =   A_b * ((weight*nu*Aul + sum_delta_true) / nb_nb)  ;
      AADD((__global float*)(RES+INDEX),   w) ;   // Sij counter update      
#  endif // ALI or not
# endif // CRT or not
      
      
      
      
      // } // CLIP  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      
      
      
      
      
      
# if (BRUTE_COOLING>0)
      // total number of photons in the package as it exits the cell
      float cool = 0.0f ;
      for(int i=0; i<CHANNELS; i++) cool += NTRUE[i] ;
      AADD(&(COOL[INDEX]), cool) ; // cooling of cell INDEX
# endif      
      // POS has been already updated, OTL and OTI point to the new cell, INDEX can be updated
      INDEX   =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;      
# if (BRUTE_COOLING>0)  // heating of the next cell, once INDEX has been updated
      if (INDEX>=0) {
         AADD(&(COOL[INDEX]), -cool) ; // heating of the next cell
      }
# endif
      
      // Note -- GetStepOT() must return accurate root grid coordinates in POS
      //         even when that position is outside the cloud !!
      if (INDEX<0) {  // exits the cloud... but on which side?
         if (POS.x>=NX  ) {     if (LEADING!=0)  POS.x =    EPS ;        }  // was EPS ...
         if (POS.x<=ZERO) {     if (LEADING!=1)  POS.x = NX-EPS ;        } 
         if (POS.y>=NY  ) {     if (LEADING!=2)  POS.y =    EPS ;        }
         if (POS.y<=ZERO) {     if (LEADING!=3)  POS.y = NY-EPS ;        } 
         if (POS.z>=NZ  ) {     if (LEADING!=4)  POS.z =    EPS ;        }
         if (POS.z<=ZERO) {     if (LEADING!=5)  POS.z = NZ-EPS ;        } 
         IndexG(&POS, &OTL, &OTI, RHO, OFF) ;
         INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;           
         if (INDEX>=0) {   // new ray started on the opposite side (same work item)
            for(int i=0; i<CHANNELS; i++)  NTRUE[i] = BG * DIRWEI ;
# if (BRUTE_COOLING>0)
            float cool = BG*DIRWEI*CHANNELS ;
            AADD(&(COOL[INDEX]), -cool) ; // heating of the entered cell
# endif
         }
      } // if INDEX<0
   } // while INDEX>=0
   
   
}


#endif   // WITH_OCTREE==1   ---- CHANNELS -> NCHN











#if (WITH_OCTREE==2)


__kernel void PathsOT2(  // 
                         __global   float  *PL,      // [CELLS]
                         __global   float  *TPL,     // [NRAY]
                         __global   int    *COUNT,   // [NRAY] total number of rays entering the cloud
                         const      int     LEADING, // leading edge
                         const      REAL3   POS0,    // initial position of ray 0
                         const      float3  DIR,     // direction of the rays
                         __global   int    *LCELLS,   
                         __constant int    *OFF,
                         __global   int    *PAR,
                         __global   float  *RHO,
                         __global   float  *BUFFER_ALL
                      ) {   
   // OT2 = split rays at the leading edge only
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge, if ray exits through a side, a new one is created 
   // on the opposite side. Ray ends when the downstream edge is reached
   int  lid =  get_local_id(0),  ls = get_local_size(0),  gid = get_group_id(0) ;
   if  (gid>=NRAY) return ;     // NRAY = ((X+1)/2) * ((Y+1)/2)
   if  (lid!=0)    return ;     // a single work item from each work group !!
   // printf("%3d / %3d\n", gid, GLOBAL/LOCAL) ;
   int  INDEX, SID, NBUF=0, nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ;
   float tpl=0.0f, dx, dy, dz ;
   __global float *BUFFER = &BUFFER_ALL[gid*(14+CHANNELS)*512] ;
   COUNT[gid] = 0 ;         // COUNT[NRAY] .... now ~ COUNT[NWG]
# if (SPLIT_UPSTREAM_ONLY==1)
   // when split done only on the leading edge -- HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
   if (LEADING==0) { HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;   }
   if (LEADING==1) { HEAD[0] = 1 ;   HEAD[1] = 3 ;   HEAD[2] = 5 ;  HEAD[3] = 7 ;   }
   if (LEADING==2) { HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;   }
   if (LEADING==3) { HEAD[0] = 2 ;   HEAD[1] = 3 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;   }
   if (LEADING==4) { HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;   }
   if (LEADING==5) { HEAD[0] = 4 ;   HEAD[1] = 5 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;   }
# else // split on each resolution change, HEAD and TAIL = subcells at lower and higher coordinates along main axis
   int HEAD[4], TAIL[4] ;  
   if ((LEADING==0)||(LEADING==1)) { 
      HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;  // lower along X
      TAIL[0] = 1 ;   TAIL[1] = 3 ;   TAIL[2] = 5 ;  TAIL[3] = 7 ;  // higher along X
   }
   if ((LEADING==2)||(LEADING==3)) { 
      HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;   
      TAIL[0] = 2 ;   TAIL[1] = 3 ;   TAIL[2] = 6 ;  TAIL[3] = 7 ;   
   }
   if ((LEADING==4)||(LEADING==5)) { 
      HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;   
      TAIL[0] = 4 ;   TAIL[1] = 5 ;   TAIL[2] = 6 ;  TAIL[3] = 7 ;   
   }
# endif
   int *SUBS ;
   // each ray shifted by two grid units ... gid over  (NX+1)/2 * (NY+1)/2 or similar number of rays
   REAL3 POS = POS0 ;
# if (FIX_PATHS>0)
   REAL3 POS_LE ;
# endif
   switch (LEADING) {
    case 0: POS.x =    EPS ; POS.y += TWO*(gid%ny) ; POS.z += TWO*(int)(gid/ny) ;  break ;
    case 1: POS.x = NX-EPS ; POS.y += TWO*(gid%ny) ; POS.z += TWO*(int)(gid/ny) ;  break ;
    case 2: POS.y =    EPS ; POS.x += TWO*(gid%nx) ; POS.z += TWO*(int)(gid/nx) ;  break ;
    case 3: POS.y = NY-EPS ; POS.x += TWO*(gid%nx) ; POS.z += TWO*(int)(gid/nx) ;  break ;
    case 4: POS.z =    EPS ; POS.x += TWO*(gid%nx) ; POS.y += TWO*(int)(gid/nx) ;  break ;
    case 5: POS.z = NZ-EPS ; POS.x += TWO*(gid%nx) ; POS.y += TWO*(int)(gid/nx) ;  break ;
   }
# if (FIX_PATHS>0)
   POS_LE = POS ;
# endif
   int OTL, OTI, OTLO, OTL_RAY, OTL_SPLIT, c1, c2, i ;
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;     // remain at root level, not yet going to leaf
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
   OTL_RAY   =  0 ;
   OTL_SPLIT =  0 ;
   
# if (DEBUG>2)
   if ((lid==0)&&(gid==GID)) printf("START: %d %5d   %8.4f %8.4f %8.4f  %8.4f %8.4f %8.4f : %d\n",
                                    OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, INDEX) ;
# endif
   
   if (INDEX>=0)  COUNT[gid] += 1 ;  // number of incoming rays
   
   
   while(INDEX>=0) {  // INDEX may refer to a cell that is not a leaf
      
# if (DEBUG>2)
      if ((lid==0)&&(gid==GID)) printf("START: OTL %d, OTI %d, INDEX %d, RHO %.3e\n", OTL, OTI, INDEX, RHO[INDEX]) ;
# endif
      
      // If we are not in a leaf, we have gone to some higher level. 
      // Go one level higher, add three rays, pick one of them as the current,
      // and return to the beginning of the loop.
      // However, if we do not enter refined region via upstream cell boundary,
      // go down but do not split the ray... an continue directly to the rest of the loop.
      if (RHO[INDEX]<=0.0f) {                    // go to the sub-cell and add sibling rays
         POS.x  =  TWO*fmod(POS.x, ONE) ;      // coordinate inside parent cell [0,1]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         dx     =  -RHO[INDEX] ;                 // OTL, OTI of the parent cell
         OTL   +=  1  ;                          // step to next level
         OTI    =  *(int *)&dx ;                 // OTI for the first child in octet
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // original subcell
         OTI   +=  SID;                          // cell in octet, original ray
# if (SPLIT_UPSTREAM_ONLY==1)
         // Now we are in the refined cell, the next level up.  Current ray is defined by (OTL, OTI, POS).
         // If and only if this is the leading edge, add the three siblings (for neighbouring cells at 
         // the current level) to buffer, pick one of the four rays, and go to beginnig of the loop.
         // If would be wasteful to split also rays if it enters refined regions via other cell sides.
         c2     =  0 ;  // set c2=1 if ray is to be split (is this the leading cell edge?)
         if ((LEADING==0)&&(POS.x<0.01f)) c2 = 1 ;  // approximately only rays that hit the upstream side
         if ((LEADING==1)&&(POS.x>1.99f)) c2 = 1 ;
         if ((LEADING==2)&&(POS.y<0.01f)) c2 = 1 ;
         if ((LEADING==3)&&(POS.y>1.99f)) c2 = 1 ;
         if ((LEADING==4)&&(POS.z<0.01f)) c2 = 1 ;
         if ((LEADING==5)&&(POS.z>1.99f)) c2 = 1 ;
         c1    =  NBUF*(14+CHANNELS) ;               // 14+CHANNELS elements per buffer entry
         if (c2>0) {  // split the ray and choose one of the new ones to be followed first
            BUFFER[c1+0]  =  OTL ;                   // level where the split is done, OTL>this =>
            BUFFER[c1+1]  =  I2F(OTI) ;              // buffer contains the OTI of the original ray
            BUFFER[c1+2]  =  POS.x ;    BUFFER[c1+3] = POS.y ;    BUFFER[c1+4] = POS.z ;
            // buffer should have three SID, the last of which is the original ray; we continue first with the 4th
            if (HEAD[0]==SID) {                     // 0=original => (1,2) to buffer, 3 as current
               BUFFER[c1+5] = HEAD[1] ;       BUFFER[c1+8] = HEAD[2] ;  SID = HEAD[3] ; // SID selects new ray
            } else {
               if (HEAD[1]==SID) {
                  BUFFER[c1+5] = HEAD[0] ;    BUFFER[c1+8] = HEAD[2] ;  SID = HEAD[3] ;
               } else {
                  if (HEAD[2]==SID) {
                     BUFFER[c1+5] = HEAD[0] ; BUFFER[c1+8] = HEAD[1] ;  SID = HEAD[3] ;
                  } else {
                     BUFFER[c1+5] = HEAD[0] ; BUFFER[c1+8] = HEAD[1] ;  SID = HEAD[2] ;
                  }
               }
            } 
            // Store the current main ray to buffer
            BUFFER[c1+11] = OTI % 8 ;               // SID of the old main ray
            BUFFER[c1+12] = OTL_RAY ;               // main ray exists at levels >= OTL_RAY
            OTL_SPLIT     = OTL_SPLIT | (1<<OTL) ;  // this ray has siblings now also on level OTL
            BUFFER[c1+13] = *(float *)&OTL_SPLIT ;  // record of levels where the main ray has been split
            // Two new rays go to buffer: they have OTL_RAY equal to current OTL
            BUFFER[c1+6]  =  OTL ;          BUFFER[c1+9 ] = OTL ;     // new rays exist only at levels >= OTL
            i = 0 ;
            BUFFER[c1+7]  = *(float *)&i  ; BUFFER[c1+10] = *(float *)&i ; // new rays not yet split
            NBUF++ ;     // three rays added, NBUF increases just by one
            // current ray defined by OTL, SID... update OTI and POS; OTI is still the cell for the main ray
            OTI   = (*(int *)&dx) + SID ;    // the new ray to be followed, OTI = index of first subcell + SID
            POS.x = fmod(POS.x,ONE) + (int)( SID%2)    ;
            POS.y = fmod(POS.y,ONE) + (int)((SID/2)%2) ;
            POS.z = fmod(POS.z,ONE) + (int)( SID/4)    ;
            OTL_SPLIT = 0 ;    // the newly created ray has never been split so far
            OTL_RAY   = OTL ;  // when we reach OTL<OTL_RAY, this ray will be terminated
         } // --- split ray
# else  
         // split the ray EVERY TIME the discretisation changed, all rays still in the same octet
         // but not necessarily the leading-edge cells
         if ((LEADING==0)||(LEADING==1))  SUBS = (POS.x<ONE) ? HEAD : TAIL ;  // HEAD, TAIL along the main 
         if ((LEADING==2)||(LEADING==3))  SUBS = (POS.y<ONE) ? HEAD : TAIL ;
         if ((LEADING==4)||(LEADING==5))  SUBS = (POS.z<ONE) ? HEAD : TAIL ;
         c1 =  NBUF*(14+CHANNELS) ;
         // split always, using the four HEAD (c2==1) or TAIL (c2==2) subcells
         BUFFER[c1+0] = OTL ;
         BUFFER[c1+1] = I2F(OTI) ;
         BUFFER[c1+2] = POS.x ;    BUFFER[c1+3] = POS.y ;    BUFFER[c1+4] = POS.z ;
         if (SUBS[0]==SID) {  // 0=original ray, that and rays (1,2) go to buffer, continue with the last new ray
            BUFFER[c1+5] = SUBS[1] ;       BUFFER[c1+8] = SUBS[2] ;  SID = SUBS[3] ; // SID selects new ray
         } else {
            if (SUBS[1]==SID) {
               BUFFER[c1+5] = SUBS[0] ;    BUFFER[c1+8] = SUBS[2] ;  SID = SUBS[3] ;
            } else {
               if (SUBS[2]==SID) {
                  BUFFER[c1+5] = SUBS[0] ; BUFFER[c1+8] = SUBS[1] ;  SID = SUBS[3] ;
               } else {
                  BUFFER[c1+5] = SUBS[0] ; BUFFER[c1+8] = SUBS[1] ;  SID = SUBS[2] ;
               }
            }
         }
         // Store the current main ray to buffer
         BUFFER[c1+11] = OTI % 8 ;               // SID of the old main ray
         BUFFER[c1+12] = OTL_RAY ;               // main ray exists at levels >= OTL_RAY
         OTL_SPLIT     = OTL_SPLIT | (1<<OTL) ;  // this ray has siblings now also on level OTL
         BUFFER[c1+13] = *(float *)&OTL_SPLIT ;  // record of levels where the main ray has been split
         // Two new rays go to buffer: they have OTL_RAY equal to current OTL
         BUFFER[c1+6]  = OTL ;    BUFFER[c1+9 ] = OTL ;   // new rays exist only at levels >= OTL
         i = 0 ;
         BUFFER[c1+7]  = *(float *)&i  ; BUFFER[c1+10] = *(float *)&i ; // new rays not yet split
         NBUF++ ;     // three rays added, NBUF increases just by one
         // current ray defined by OTL, SID... update OTI and POS; OTI is still the cell for the main ray
         OTI   = (*(int *)&dx) + SID ;    // the new ray to be followed, OTI = index of first subcell + SID
         POS.x = fmod(POS.x,ONE) + (int)( SID%2)    ;
         POS.y = fmod(POS.y,ONE) + (int)((SID/2)%2) ;
         POS.z = fmod(POS.z,ONE) + (int)( SID/4)    ;
         OTL_SPLIT = 0 ;    // the newly created ray has never been split so far
         OTL_RAY   = OTL ;  // when we reach OTL<OTL_RAY, this ray will be terminated
# endif
         
         
         INDEX = OFF[OTL]+OTI ;   // global index -- must be now >=0 since we went down a level
         
         // if not yet a leaf, jump back to start of the loop => another step down
         if (RHO[INDEX]<=0.0f) {
# if (DEBUG>2)
            if ((lid==0)&&(gid==GID)) printf("RHO<=0  ==>  CONTINUE ... TO STEP DOWN AT THE START\n") ;
# endif
            continue ;
         }
         
      } // RHO<0, was not leaf cell
      
      
      
# if (DEBUG>2)
      if ((lid==0)&&(gid==GID)) printf("A: %d %5d %6d   %8.4f %8.4f %8.4f   %8.4f %8.4f %8.4f\n", 
                                       OTL, OTI, INDEX, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z) ;
# endif
      
      // we come here only if we have a ray that is inside a leaf node => can do the step forward
      // if the step is to refined cell, (OTL, OTI) will refer to a cell that is not a leaf node
      // ==> the beginning of the present loop will have to deal with the step to higher refinement
      // level and the creation of three additional rays
      OTLO    =  OTL ;
      // get step but do not yet go to level > OTLO
# if (FIX_PATHS>0)
      dx      =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, &POS_LE, LEADING) ; // step [GL] == root grid units !!
# else
      dx      =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL   , LEADING) ; // step [GL] == root grid units !!
# endif
      // note -- if we go to refined region, at this point still OTL==OTLO
      //         if we go to lower hierarchy level,              OTL< OTLO
                         
# if 0
      PL[INDEX] += dx ;           // path length, cumulative over idir and ioff --- to old cell
# else  
      // again... without atomic much larger noise in path lengths ....
      // but why??? concurrent rays should never touch the same cells during the same kernel call
      AADD(&(PL[INDEX]), dx) ;
# endif
      tpl       += dx ;               // just the total value for current idir, ioff
      
      // The new index at the end of the step  --- OTL >= OTL  => INDEX may not refer to a leaf cell
      INDEX    =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;  // OTL, OTI, POS already updated = end of the current step
      
      
      
# if (DEBUG==2) // print only coordinates when entering real calculations
      if ((lid==0)&&(gid==GID)) {
         if (OTL==0) {
            printf(" %d %5d %6d   %8.4f %8.4f %8.4f   %8.4f %8.4f %8.4f\n", 
                   OTL, OTI, INDEX, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z) ;
         } else {
            // convert coordinates to root grid coordinates
            REAL3 RPOS = POS ;
            RootPos(&RPOS, OTL, OTI, OFF, PAR) ;
            printf(" %d %5d %6d   %8.4f %8.4f %8.4f   %8.4f %8.4f %8.4f\n", 
                   OTL, OTI, INDEX, RPOS.x, RPOS.y, RPOS.z, DIR.x, DIR.y, DIR.z) ;
         }
      }
# endif
      
      
      
      if (INDEX>=0) {
         if (RHO[INDEX]<=0.0f) {  // we ended up in a parent cell
# if (DEBUG>2)
            if ((lid==0)&&(gid==GID)) printf("E: INDEX %d but RHO<0 => refine at the loop start\n", INDEX) ;
# endif
            continue ; // we moved to refined region, handled at the beginning of the main loop
         }
      }
      
      
      if (OTL<OTL_RAY) {   // up in hierarchy and this ray is terminated -- cannot be on root grid
# if (DEBUG>2)
         if ((lid==0)&&(gid==GID)) printf("E: RAY TERMINATED\n") ;
# endif
         INDEX=-1 ;        // triggers the generation of a new ray below
      } else {    
         if (INDEX<0) {    // ray exits the cloud on root grid, possibly create a new OTL=0 ray on the other side
# if (DEBUG>2)
            if ((lid==0)&&(gid==GID)) printf("E: RAY EXITS\n") ;
# endif
            if (POS.x>=NX  ) {   if (LEADING!=0)   POS.x =    EPS ;   }
            if (POS.x<=ZERO) {   if (LEADING!=1)   POS.x = NX-EPS ;   }
            if (POS.y>=NY  ) {   if (LEADING!=2)   POS.y =    EPS ;   }
            if (POS.y<=ZERO) {   if (LEADING!=3)   POS.y = NY-EPS ;   } 
            if (POS.z>=NZ  ) {   if (LEADING!=4)   POS.z =    EPS ;   }
            if (POS.z<=ZERO) {   if (LEADING!=5)   POS.z = NZ-EPS ;   } 
            IndexGR(&POS, &OTL, &OTI, RHO, OFF) ; // not necessarily a leaf!
            INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
            if (INDEX>=0) {    // new level-0 ray started on the opposite side
               OTL_RAY = 0 ;    OTLO    = 0 ;   COUNT[gid] += 1 ;
               continue ;
            }
         } // if INDEX<0
      }
      
      
      // [C] if INDEX still negative, current ray truly ended => take new ray from the buffer, if such exists
      if ((INDEX<0)&&(NBUF>0)) {            // NBUF>0 => at least one ray exists in the buffer
# if (DEBUG==1)
         if ((lid==0)&&(gid==GID))  printf("E: TAKE RAY: %d left\n", NBUF) ;
# endif
         c1    =  (NBUF-1)*(14+CHANNELS) ;  // 11+CHANNELS elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // level where sibling rays were created
         OTI   =  F2I(BUFFER[c1+1]) ;       // OTI for the original ray that was split
         POS.x =  BUFFER[c1+2] ;   POS.y = BUFFER[c1+3] ;   POS.z = BUFFER[c1+4] ;
         // three SID numbers... of which at least one is still non-negative
         SID =  (int)BUFFER[c1+5] ;
         if (SID>=0) {   // first saved subray not yet simulated
            BUFFER[c1+5] = -1.0f ;        OTL_RAY = BUFFER[c1+ 6] ;  dx = BUFFER[c1+ 7] ;  OTL_SPLIT = *(int*)&dx ;
         } else {
            SID = (int)BUFFER[c1+8] ;
            if (SID>=0) {
               BUFFER[c1+8] = -1.0f ;     OTL_RAY = BUFFER[c1+ 9] ;  dx = BUFFER[c1+10] ;  OTL_SPLIT = *(int*)&dx ;
            } else { // must be the third one -- the original main ray among the four rays at OTL
               SID = (int)BUFFER[c1+11] ; OTL_RAY = BUFFER[c1+12] ;  dx = BUFFER[c1+13] ;  OTL_SPLIT = *(int*)&dx ;
               NBUF -= 1 ;  // was last of the three subrays in this buffer entry
            }
         }
         // we must have found one ray  -- figure out its OTI (index within level OTL)
         OTI   =  8*(int)(OTI/8)   + SID ;               // OTI in the buffer was for the original ray that was split
         // update the position from POS for "some subray" to POS of the current subray == SID
         POS.x =  fmod(POS.x,ONE) + (int)( SID%2)    ;  // [0,2]
         POS.y =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
         POS.z =  fmod(POS.z,ONE) + (int)( SID/4)    ;
         // this stored ray is at level OTL but may be in a cell that is itself still further
         // refined => this will be handled at the start of the main loop => possible further
         // refinement before on the rays is stepped forward
         INDEX = OFF[OTL]+OTI ;  // global index -- must be >=0 !!!
# if (DEBUG>2)
         if ((lid==0)&&(gid==GID)) printf("%d %5d %6d   %8.4f %8.4f %8.4f   %8.4f %8.4f %8.4f -- from buffer\n",
                                          OTL, OTI, INDEX, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z) ;
# endif         
      }  // (INDEX<0)&&(NBUF>0)
      
   } // while INDEX>=0  --- stops when buffer is empty and the main level-0 ray has exited
   
   TPL[gid] = tpl ;
}







__kernel void UpdateOT2(  // 
                          __global float  *PL,      //  0
# if (WITH_HALF==1)
                          __global short4 *CLOUD,   //  1 [CELLS]: vx, vy, vz, sigma
# else
                          __global float4 *CLOUD,   //  1 [CELLS]: vx, vy, vz, sigma
# endif
                          GAUSTORE  float *GAU,     //  2 precalculated gaussian profiles [GNO,CHANNELS]
                          constant int2   *LIM,     //  3 limits of ~zero profile function [GNO]
                          const float      Aul,     //  4 Einstein A(upper->lower)
                          const float      A_b,     //  5 (g_u/g_l)*B(upper->lower)
                          const float      GN,      //  6 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                          const float      APL,     //  7 average path length [GL]
                          const float      BG,      //  8 background value (photons)
                          const float      DIRWEI,  //  9 weight factor (based on current direction)
                          const float      EWEI,    // 10 weight 1/<1/cosT>/NDIR
                          const int        LEADING, //  0 leading edge
                          const REAL3      POS0,    // 11 initial position of id=0 ray
                          const float3     DIR,     // 12 ray direction
                          __global float  *NI,      // 13 [CELLS]:  NI[upper] + NB_NB
                          __global float  *RES,     // 14 [CELLS]:  SIJ, ESC
                          __global float  *NTRUES,  // 15 [NWG*MAXCHN]   --- NWG>=simultaneous level 0 rays
# if (WITH_CRT>0)
                          constant float *CRT_TAU,  //  dust optical depth / GL
                          constant float *CRT_EMI,  //  dust emission photons/c/channel/H
# endif                     
# if (BRUTE_COOLING>0)
                          __global float   *COOL,   // 14,15 [CELLS] = cooling 
# endif
                          __global   int   *LCELLS, //  16
                          __constant int   *OFF,    //  17
                          __global   int   *PAR,    //  18
                          __global   float *RHO,    //  19  -- needed only to describe the hierarchy
                          __global   float *BUFFER_ALL  //  20 -- buffer to store split rays
                       )  {   
   // Each ***WORK GROUP*** processes one ray. The rays are two cells apart to avoid
   // synchronisation problems (???). Rays start on the leading edge. If ray exits through a
   // side (wrt axis closest to direction of propagation), a new one is created on the
   // opposite side and the ray ends when the downstream edge is reached.
   // 
   // As one goes to a higher hierarchy level (into a refined cell), one pushes the
   // original and two new rays to buffer. The fourth ray (one of the new ones) is
   // continued. When ray goes up in hierarchy (to a larger grain) ray is ended if that
   // was created at higher level. When ray ends, the next one is taken from the buffer
   // (stack).  As before, if level=0 ray exits from sides continue
   // If ray goes up in hierarchy, above a level where it had siblings, NTRUE is scaled *= 4
   // or, if one goes up several levels, *=4^n, where n is the number of levels with siblings
   int id  = get_global_id(0), lid = get_local_id(0), gid = get_group_id(0)   ;
   int ls  = get_local_size(0) ;
   if (gid>=NRAY) return ;        // one work group per ray .... NWG==NRAY
   
   int nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ;
   GAUSTORE float *profile ;
   __global float *BUFFER = &BUFFER_ALL[gid*(14+CHANNELS)*512] ;
   __local  float  NTRUE[CHANNELS] ;
   __local  int3  LINT ;  // SID, NBUF, OTL_SPLIT
   __local  float SDT[LOCAL] ;  // per-workitem sum_delta_true
   __local  float AE[LOCAL] ;   // per-workitem all_escaped
   float weight, w, dx, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed, sum_delta_true, all_escaped, nu ;
   int row, shift, INDEX, c1, c2, OTL, OTI, OTLO, OTL_RAY, SID, OTL_SPLIT, NBUF=0 ;
# if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, pro, sij ;
# endif
   
   
# if (SPLIT_UPSTREAM_ONLY==1)
   // when split done only on the leading edge -- HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
   if (LEADING==0) { HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;   }
   if (LEADING==1) { HEAD[0] = 1 ;   HEAD[1] = 3 ;   HEAD[2] = 5 ;  HEAD[3] = 7 ;   }
   if (LEADING==2) { HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;   }
   if (LEADING==3) { HEAD[0] = 2 ;   HEAD[1] = 3 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;   }
   if (LEADING==4) { HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;   }
   if (LEADING==5) { HEAD[0] = 4 ;   HEAD[1] = 5 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;   }
# else // split on each resolution change, HEAD and TAIL = subcells at lower and higher coordinates along main axis
   int HEAD[4], TAIL[4] ;  
   if ((LEADING==0)||(LEADING==1)) { 
      HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;  // lower along X
      TAIL[0] = 1 ;   TAIL[1] = 3 ;   TAIL[2] = 5 ;  TAIL[3] = 7 ;  // higher along X
   }
   if ((LEADING==2)||(LEADING==3)) { 
      HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;   
      TAIL[0] = 2 ;   TAIL[1] = 3 ;   TAIL[2] = 6 ;  TAIL[3] = 7 ;   
   }
   if ((LEADING==4)||(LEADING==5)) { 
      HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;   
      TAIL[0] = 4 ;   TAIL[1] = 5 ;   TAIL[2] = 6 ;  TAIL[3] = 7 ;   
   }
# endif
   
   int *SUBS ;
   REAL3 POS = POS0 ;
# if (FIX_PATHS>0)
   REAL3 POS_LE ;
# endif
   switch (LEADING) {
    case 0: POS.x =    EPS ;  POS.y += TWO*(gid%ny) ;  POS.z += TWO*(int)(gid/ny) ;  break ;
    case 1: POS.x = NX-EPS ;  POS.y += TWO*(gid%ny) ;  POS.z += TWO*(int)(gid/ny) ;  break ;
    case 2: POS.y =    EPS ;  POS.x += TWO*(gid%nx) ;  POS.z += TWO*(int)(gid/nx) ;  break ;
    case 3: POS.y = NY-EPS ;  POS.x += TWO*(gid%nx) ;  POS.z += TWO*(int)(gid/nx) ;  break ;
    case 4: POS.z =    EPS ;  POS.x += TWO*(gid%nx) ;  POS.y += TWO*(int)(gid/nx) ;  break ;
    case 5: POS.z = NZ-EPS ;  POS.x += TWO*(gid%nx) ;  POS.y += TWO*(int)(gid/nx) ;  break ;
   }
# if (FIX_PATHS>0)
   POS_LE = POS ;
# endif
   for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BG * DIRWEI ;
   
   // unlike in UpdateOT1, initial ray is generated at level 0 because that will be split
   // at the beginning of the main loop
   
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;             // packet indices for the start position
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  // global index to data vectors
   OTL_RAY   =  0 ;  // level at which ray was created
   OTL_SPLIT =  0 ;  // levels (encoded in individual bits) at which this ray was split to four
   
# if (BRUTE_COOLING>0)
   float cool = BG*DIRWEI*CHANNELS ;
   if ((INDEX>=0)&&(lid==0)) {
      COOL[INDEX] -= cool ; // heating of the entered cell --- no need for atomics
   }
# endif
   
   
   
   
   while(INDEX>=0) {
      
      barrier(CLK_GLOBAL_MEM_FENCE) ;
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
      if (RHO[INDEX]<=0.0f) {                    // we start in a cell with children
# if (DEBUG>0)
         if ((lid==0)&&(gid==GID)) printf("START IN PARENT CELL\n") ;
# endif
         POS.x  =  TWO*fmod(POS.x, ONE) ;      // coordinate inside parent cell, in octet [0,2]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         dx     =  -RHO[INDEX] ;                 // OTL, OTI of the parent cell
         OTL   +=  1  ;                          // step to next level
         OTI    =  *(int *)&dx ;                 // OTI for the first child in octet
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // subcell for the original ray
         OTI   +=  SID ;
         
# if (SPLIT_UPSTREAM_ONLY==1) // <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
         // Now we are in the refined cell, the next level up.  Current ray is defined by (OTL, OTI, POS).
         // If and only if this is the leading edge, add the three siblings (for neighbouring cells at 
         // the current level) to buffer, pick one of the four rays, and go to beginnig of the loop.
         // If would be wasteful to split also rays if it enters refined regions via other cell sides.
         c2     =  0 ;  // set c2=1 if ray is to be split (is this the leading cell edge?)
         if ((LEADING==0)&&(POS.x<0.01f)) c2 = 1 ;  // approximately only rays that hit the upstream side
         if ((LEADING==1)&&(POS.x>1.99f)) c2 = 1 ;
         if ((LEADING==2)&&(POS.y<0.01f)) c2 = 1 ;
         if ((LEADING==3)&&(POS.y>1.99f)) c2 = 1 ;
         if ((LEADING==4)&&(POS.z<0.01f)) c2 = 1 ;
         if ((LEADING==5)&&(POS.z>1.99f)) c2 = 1 ;
         c1    =  NBUF*(14+CHANNELS) ;        // 14+CHANNELS elements per buffer entry
         // -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o
         if ((lid==0)&(c2>0)) {               // split the ray and choose one of the new ones to be followed first
            BUFFER[c1+0]  =  OTL ;            // level where the split is done, OTL>this =>
            BUFFER[c1+1]  =  I2F(OTI) ;       // buffer contains the OTI of the original ray
            BUFFER[c1+2]  =  POS.x ;    BUFFER[c1+3] = POS.y ;    BUFFER[c1+4] = POS.z ;
            // buffer should have three SID, the last of which is the original ray; we now continue with the 4th
            if (HEAD[0]==SID) {               // 0=original => (1,2) to buffer, 3 as current
               BUFFER[c1+5] = HEAD[1] ;       BUFFER[c1+8] = HEAD[2] ;  SID = HEAD[3] ; // SID selects new ray
            } else {
               if (HEAD[1]==SID) {
                  BUFFER[c1+5] = HEAD[0] ;    BUFFER[c1+8] = HEAD[2] ;  SID = HEAD[3] ;
               } else {
                  if (HEAD[2]==SID) {
                     BUFFER[c1+5] = HEAD[0] ; BUFFER[c1+8] = HEAD[1] ;  SID = HEAD[3] ;
                  } else {
                     BUFFER[c1+5] = HEAD[0] ; BUFFER[c1+8] = HEAD[1] ;  SID = HEAD[2] ;
                  }
               }
            } 
            // now SID defines the new ray that is to be followed first
            // Store the current main ray to buffer
            BUFFER[c1+11] = OTI % 8 ;               // SID of the old main ray
            BUFFER[c1+12] = OTL_RAY ;               // main ray exists at levels >= OTL_RAY
            OTL_SPLIT     = OTL_SPLIT | (1<<OTL) ;  // this ray has siblings now also on level OTL
            BUFFER[c1+13] = *(float *)&OTL_SPLIT ;  // record of levels where the main ray has been split
            // Two new rays go to buffer: they have OTL_RAY equal to current OTL
            BUFFER[c1+6]  =  OTL ;          BUFFER[c1+9 ] = OTL ;     // new rays exist only at levels >= OTL
            int i = 0 ;
            BUFFER[c1+7]  = *(float *)&i  ; BUFFER[c1+10] = *(float *)&i ; // new rays not yet split
            NBUF++ ;     // three rays added, NBUF increases just by one
            LINT.x = SID ; LINT.y = NBUF ; LINT.z = OTL_SPLIT ;
         }
         // -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o
         barrier(CLK_LOCAL_MEM_FENCE) ;
         SID = LINT.x ;  NBUF = LINT.y ; OTL_SPLIT = LINT.z ;
         if (c2>0) {   // was split
            // current ray defined by OTL, SID... update OTI and POS; OTI is still the cell for the main ray
            OTI   = (*(int *)&dx) + SID ;    // the new ray to be followed, OTI = index of first subcell + SID
            POS.x = fmod(POS.x,ONE) + (int)( SID%2)    ;
            POS.y = fmod(POS.y,ONE) + (int)((SID/2)%2) ;
            POS.z = fmod(POS.z,ONE) + (int)( SID/4)    ;
            OTL_SPLIT = 0 ;    // the newly created ray has never been split so far
            OTL_RAY   = OTL ;  // when we reach OTL<OTL_RAY, this ray will be terminated
            for(int i=lid; i<CHANNELS; i+=ls) {
               NTRUE[i] /= 4.0f ;       BUFFER[c1+14+i] =  NTRUE[i] ;
            }
         }
# else   // not SPLIT_UPSTREAM_ONLY <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
         // split the ray EVERY TIME the discretisation level increases, all subrays still in the same octet
         // but not necessarily on the leading edge
         if ((LEADING==0)||(LEADING==1))  SUBS = (POS.x<ONE) ? HEAD : TAIL ;  // are we in HEAD or TAIL subcells
         if ((LEADING==2)||(LEADING==3))  SUBS = (POS.y<ONE) ? HEAD : TAIL ;
         if ((LEADING==4)||(LEADING==5))  SUBS = (POS.z<ONE) ? HEAD : TAIL ;
         c1 =  NBUF*(14+CHANNELS) ;
         // -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o
         if (lid==0) {  
            BUFFER[c1+0] = OTL ;
            BUFFER[c1+1] = I2F(OTI) ;
            BUFFER[c1+2] = POS.x ;    BUFFER[c1+3] = POS.y ;    BUFFER[c1+4] = POS.z ;
            if (SUBS[0]==SID) {  // 0=original ray, that and rays (1,2) go to buffer, continue with the last new ray
               BUFFER[c1+5] = SUBS[1] ;       BUFFER[c1+8] = SUBS[2] ;  SID = SUBS[3] ; // SID selects new ray
            } else {
               if (SUBS[1]==SID) {
                  BUFFER[c1+5] = SUBS[0] ;    BUFFER[c1+8] = SUBS[2] ;  SID = SUBS[3] ;
               } else {
                  if (SUBS[2]==SID) {
                     BUFFER[c1+5] = SUBS[0] ; BUFFER[c1+8] = SUBS[1] ;  SID = SUBS[3] ;
                  } else {
                     BUFFER[c1+5] = SUBS[0] ; BUFFER[c1+8] = SUBS[1] ;  SID = SUBS[2] ;
                  }
               }
            }
            // 0   1    2  3  4    5    6   7         8    9    10       11   12  13       14   
            // OTL OTI  x  y  z    SID  RAY SPLIT     SID  RAY  SPLIT    SID  RAY SPLIT    NTRUE
            // Store the current main ray to buffer as the third one
            BUFFER[c1+11] = OTI % 8 ;               // SID of the old main ray
            BUFFER[c1+12] = OTL_RAY ;               // main ray exists at levels >= OTL_RAY
            OTL_SPLIT     = OTL_SPLIT | (1<<OTL) ;  // this ray has siblings now also on level OTL
            BUFFER[c1+13] = *(float *)&OTL_SPLIT ;  // record of levels where the main ray has been split
            // Two new rays go to buffer: they have OTL_RAY equal to current OTL
            BUFFER[c1+6]  = OTL ;    BUFFER[c1+9 ] = OTL ;   // new rays exist only at levels >= OTL
            c2 = 0 ;
            BUFFER[c1+7]  = *(float *)&c2  ; BUFFER[c1+10] = *(float *)&c2 ; // new rays not yet split
            NBUF++ ;     // three rays added, NBUF increases just by one
            LINT.x = SID ; LINT.y = NBUF ; LINT.z = OTL_SPLIT ;
         } 
         // -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o
         barrier(CLK_LOCAL_MEM_FENCE) ;
         SID = LINT.x ;   NBUF = LINT.y ;  OTL_SPLIT = LINT.z ;
         // current ray defined by OTL, SID... update OTI and POS; OTI is still the cell for the main ray
         OTI   = (*(int *)&dx) + SID ;    // the new ray to be followed, OTI = index of first subcell + SID
         POS.x = fmod(POS.x,ONE) + (int)( SID%2)    ;
         POS.y = fmod(POS.y,ONE) + (int)((SID/2)%2) ;
         POS.z = fmod(POS.z,ONE) + (int)( SID/4)    ;
         OTL_SPLIT = 0 ;    // the newly created ray has never been split so far
         OTL_RAY   = OTL ;  // when we reach OTL<OTL_RAY, this ray will be terminated
         // ray was split => divide NTRUE by four and copy to buffer
         for(int i=lid; i<CHANNELS; i+=ls) {
            NTRUE[i] /= 4.0f ;       BUFFER[c1+14+i] =  NTRUE[i] ;
         }
# endif // SPLIT_UPSTREAM_ONLY or not <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
         
         INDEX = OFF[OTL]+OTI ;            // global index -- must be >=0 since we went down a level
# if (DEBUG>0)
         if ((lid==0)&&(gid==GID)) printf("RAY SPLIT, ONE SELECTED, RHO %.3e\n", RHO[INDEX]) ;
# endif
         if (RHO[INDEX]<=0.0f) continue ;  // not yet leaf => go to the start of the main loop again
      } // RHO<=0, cell was not a leaf
      
      
      
      // we are now in a leaf node, ready to make the step
      OTLO   =  OTL ;
# if (FIX_PATHS>0)
      dx     =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, &POS_LE, LEADING) ;  // POS is now the end of this step
# else
      dx     =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL   , LEADING) ;  // POS is now the end of this step
# endif
      
# if 0 // debugging - check that PL will be creased to zero !!
      if (lid==1) AADD( &(PL[INDEX]), -dx ) ;  
# endif
      
      
      
      // default is/was  (A) + weighting (APL/PL) on the host side
      // alternative (?): (B) and no (APL/PL) weighting on the host side
      
# if 1  // (A)  EITHER THIS WITH POST WEIGHTING   
      // APL = expected path length for a root grid cell
      // With OCTREE2, we should have 4 subrays for an octet =>  one ray for each subcell but 0.5 length
      //   (dx/APL) => dx/(APL/2^L)  while  VOLUME => VOLUME/8^L   =>   () /4^L
      weight    =  (dx/ APL     ) *  VOLUME  /  pown(4.0f, OTLO) ;  
# else  // (B)  OR THIS WITHOUT POST ---  BUT POST WEIGHTING WOULD APPLY ALSO TO SIJ FROM INCOMING PHOTONS!!!
      // PL = actual distance travelled in a cell   <>  APL
      // dx/PL[INDEX]  ~ what fraction of emission to be attributed to the current ray => emitted photons exact
      weight    =  (dx/PL[INDEX]) *  VOLUME /   pown(8.0f, OTLO) ;
# endif
      
      // INDEX is still the index for the cell where the step starts
      nu        =  NI[2*INDEX] ;
      nb_nb     =  NI[2*INDEX+1] ;
      
      
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# if (WITH_HALF==1)
      doppler  *=  0.002f ;  // half integer times 0.002f km/s
# endif                 
      // tmp_tau   =  max(1.0e-35f, dx*nb_nb*GN) ;
      tmp_tau   =  dx*nb_nb*GN ;
      if (fabs(tmp_tau)<1.0e-28f) tmp_tau = 1.0e-28f ;  // was 1e-32
      tmp_emit  =  weight * nu*Aul / tmp_tau ;  // GN include grid length [cm]
      shift     =  round(doppler/WIDTH) ;
# if (WITH_HALF==1)      //               sigma = 0.002f * w,   lookup table: sigma = SIGMA0 * SIGMAX^row
      row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
# else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
# endif
      profile   =  &GAU[row*CHANNELS] ;
      // avoid profile function outside profile channels LIM.x, LIM.y
      c1        =  max(LIM[row].x+shift, max(0, shift)) ;
      c2        =  min(LIM[row].y+shift, min(CHANNELS-1, CHANNELS-1+shift)) ;
      sum_delta_true = 0.0f ;
      all_escaped    = 0.0f ;
      
      
# if (WITH_CRT>0) // WITH_CRT
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
         tt     =  (fabs(Ttau)>0.01f) ?  ((1.0f-exp(-Ttau))/Ttau) : (1.0f-Ttau*(0.5f-0.166666667f*Ttau)) ;
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
      // RES[INDEX].x   += sij/VOLUME * pow(8.0f,OTL) ;  // octree 8^OTL done in the solver (host)
      RES[2*INDEX]     += sij ;            // division by VOLUME done in the solver (kernel)
      // Emission ~ path length dx but also weighted according to direction, works because <WEI>==1.0
      RES[2*INDEX+1]   += all_escaped ;    // divided by VOLUME only oin Solve() !!!
      
      
# else   // no WITH_CRT
      // because of c1, the same NTRUE elements may be updated each time by different work items...
      
      barrier(CLK_LOCAL_MEM_FENCE) ;    // local NTRUE elements possibly updated by different threads
      
#  if (WITH_ALI>0) // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      for(int i=c1+lid; i<=c2; i+=ls)  {
         w               =  tmp_tau*profile[i-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         //factor        =  (fabs(w)>0.005f)?(1.0f-exp(-w)):(w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;    // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      SDT[lid] = sum_delta_true ;  AE[lid] = all_escaped ;
      barrier(CLK_LOCAL_MEM_FENCE) ;
      if (lid==0) {
         for(int i=1; i<LOCAL; i++) {  
            sum_delta_true += SDT[i] ;    
            all_escaped    +=  AE[i] ;     
         }
#   if 0
         RES[2*INDEX]    +=   A_b * (sum_delta_true/nb_nb) ;  // division by VOLUME now in the solver
         RES[2*INDEX+1]  +=   all_escaped ;
#   else
         AADD((__global float*)(RES+2*INDEX  ),  A_b * (sum_delta_true/nb_nb)) ;
         AADD((__global float*)(RES+2*INDEX+1),  all_escaped) ;
#   endif
      } // lid==0
      
#  else // else no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
      for(int ii=c1+lid; ii<=c2; ii+=ls)  {
         w               =  tmp_tau*profile[ii-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[ii]*factor ;   // incoming photons that are absorbed
         NTRUE[ii]      +=  escape-absorbed ;
         sum_delta_true +=  absorbed - escape ;  // later absorbed ~ W*nu*Aul - escape
         all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      SDT[lid] = sum_delta_true ;
      barrier(CLK_LOCAL_MEM_FENCE) ;
      if (lid==0) {
         for(int i=1; i<LOCAL; i++) sum_delta_true += SDT[i] ;    
         w  =   A_b * (  (weight*nu*Aul  + sum_delta_true) / nb_nb )  ;
         AADD((__global float*)(RES+INDEX),   w) ;   // Sij counter update
      } 
      
#  endif // no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
      
      
# endif  // WITH OR WITHOUT CRT
      
      
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
# if (BRUTE_COOLING>0)
      // total number of photons in the package as it exits the cell
      if (lid==0) {
         float cool = 0.0f ;
         for(int ii=0; ii<CHANNELS; ii++) cool += NTRUE[ii] ;
         COOL[INDEX] += cool ; // cooling of cell INDEX --- each work group distinct rays => no need for atomics
      }
# endif      
      // Updates at the end of the step, POS has been already updated, OTL and OTI point to the new cell
      INDEX   =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;            
# if (BRUTE_COOLING>0)  // heating of the next cell, once INDEX has been updated
      if (INDEX>=0) {
         if (lid==0)  COOL[INDEX] -= cool ; // heating of the next cell
      }
# endif
      
      if (INDEX>=0) {
         if (RHO[INDEX]<=0.0f) { // we stepped to a refined cell (GetStep goes only to a cell OTL<=OTLO)
# if (DEBUG>0)
            if ((lid==0)&&(gid==GID)) printf("ENTERED REFINED CELL, OTL=%d\n", OTL) ;
# endif
            continue ;       // step down one level at the beginning of the main loop
         }
      }
      
      
      
      if (OTL<OTL_RAY) {   // we are up to a level where this ray no longer exists
         INDEX=-1 ;        
      } else {      
         if (INDEX<0) {    // ray exits the cloud... possibly continues on the other side
            if (POS.x>=NX  ) {   if (LEADING!=0)  POS.x =    EPS ;   }
            if (POS.x<=ZERO) {   if (LEADING!=1)  POS.x = NX-EPS ;   } 
            if (POS.y>=NY  ) {   if (LEADING!=2)  POS.y =    EPS ;   }
            if (POS.y<=ZERO) {   if (LEADING!=3)  POS.y = NY-EPS ;   } 
            if (POS.z>=NZ  ) {   if (LEADING!=4)  POS.z =    EPS ;   }
            if (POS.z<=ZERO) {   if (LEADING!=5)  POS.z = NZ-EPS ;   } 
            IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;  // we remain in a root-grid cell => OTL==0 !
            INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
            if (INDEX>=0) {   // new level-0 ray started on the opposite side (may be a parent cell)
               OTL_RAY = 0 ;  OTLO = 0 ;  OTL_SPLIT = 0 ;
               for(int ii=lid; ii<CHANNELS; ii+=ls)  NTRUE[ii] = BG * DIRWEI ;
               barrier(CLK_LOCAL_MEM_FENCE) ;
# if (BRUTE_COOLING>0)
               if (lid==0) {
                  dx = BG*DIRWEI*CHANNELS ;  COOL[INDEX] -= dx ; // heating of the entered cell
               }
# endif
               continue ;
            }
         }
      }
      
      
      if ((INDEX>=0)&&(OTL<OTLO)) {    // ray continues at a lower hierarchy level => NTRUE may have to be scaled
         // must check any splits between levels [OTL+1, OTLO]
         dx = 1.0f ;
         for(int i=OTL+1; i<=OTLO; i++) {
            if (OTL_SPLIT & (1<<i)) {  dx *= 4.0f ;  OTL_SPLIT = OTL_SPLIT ^ (1<<i) ;  }
         }
         if (dx>1.0f) {   
            for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] *= dx ;     
         }
         continue ;  // we went to lower level => this cell is a leaf
      }
      
      
      // if INDEX still negative, try to take a new ray from the buffer
      // 0   1    2  3  4    5    6   7         8    9    10       11   12  13       14   
      // OTL OTI  x  y  z    SID  RAY SPLIT     SID  RAY  SPLIT    SID  RAY SPLIT    NTRUE
      if ((INDEX<0)&&(NBUF>0)) {          // NBUF>0 => at least one ray exists in the buffer
# if (DEBUG>0)
         if ((lid==0)&&(gid==GID)) printf("RAY FROM BUFFER\n") ;
# endif
         c1    =  (NBUF-1)*(14+CHANNELS) ;  // 8+CHANNELS elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // OTL ...
         OTI   =  F2I(BUFFER[c1+1]) ;       // and OTI of the ray that was split
         POS.x =  BUFFER[c1+2] ;   POS.y = BUFFER[c1+3] ;   POS.z = BUFFER[c1+4] ;
         // three SID numbers... of which at least one is still non-negative
         // -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o
         if (lid==0) { 
            SID   =  (int)BUFFER[c1+5] ;
            if (SID>=0) {
               BUFFER[c1+5] = -1.0f ;         c2 = 0 ;
            } else {
               SID = (int)BUFFER[c1+8] ;
               if (SID>=0) {
                  BUFFER[c1+8] = -1.0f ;      c2 = 1 ;
               } else { // must be the third one
                  SID = (int)BUFFER[c1+11] ;  c2 = 2 ; NBUF -= 1 ;  // last of the three subrays in this buffer entry
               }
            }
            LINT.x = SID ; LINT.y = NBUF ; LINT.z = c2 ;
         }  // if lid==0
         // -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o
         barrier(CLK_LOCAL_MEM_FENCE) ;
         SID   =  LINT.x ;    NBUF  =  LINT.y ;   c2 = LINT.z ;
         if (c2==0) {   // was the first saved subray
            OTL_RAY    = (int)BUFFER[c1+ 6] ;   dx = BUFFER[c1+ 7] ;  OTL_SPLIT = *(int*)&dx ;
         } else {
            if (c2==1) {
               OTL_RAY = (int)BUFFER[c1+ 9] ;   dx = BUFFER[c1+10] ;  OTL_SPLIT = *(int*)&dx ;
            } else {
               OTL_RAY = (int)BUFFER[c1+12] ;   dx = BUFFER[c1+13] ;  OTL_SPLIT = *(int*)&dx ;
            }
         }
# if (DEBUG>0)
         if ((lid==0)&&(gid==GID)&&(c2==2)) printf("..... RAY FROM BUFFER - WAS LAT ONE = ORIGINAL MAIN\n") ;
# endif
         // we must have found one ray  -- figure out its OTI
         OTI   =  8*(int)(OTI/8)  + SID ;     // OTI in buffer was for the original ray that was split
         INDEX =  OFF[OTL]+OTI  ;             // must be positive here !
         // update the position from POS for "some subray" to POS of this subray
         POS.x =  fmod(POS.x,ONE) + (int)( SID%2)     ;
         POS.y =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
         POS.z =  fmod(POS.z,ONE) + (int)( SID/4) ;
         // copy NTRUE --- values in BUFFER have already been divided by four
         c1 += 14 ;
         for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BUFFER[c1+i] ;
         // note - this ray be inside a parent cell => handled at the beginnign of the main loop
      } // (INDEX<=0)&(NBUF>0)
      
      
   } // while INDEX>=0
   
}


#endif  // WITH_OCTREE==2

















#if (WITH_OCTREE==3)


# if (DEBUG>0)
void report(const int OTL, const int OTI, const int OTL_RAY, REAL3 *POS, __constant int *OFF, __global int *PAR) {
   REAL3 pos = *POS ;
   RootPos(&pos, OTL, OTI, OFF, PAR) ;   
   printf(" (%d,%d) %4d    L: %4.2f %4.2f %4.2f  R: %4.2f %4.2f %4.2f  OTL=%d OTL_RAY=%d\n",
          OTL, OTI, OFF[OTL]+OTI, POS->x, POS->y, POS->z, pos.x, pos.y, pos.z, OTL, OTL_RAY) ;   
}
# endif


__kernel void PathsOT3(  // 
                         __global   float   *PL,      // [CELLS]
                         __global   float   *TPL,     // [NRAY]
                         __global   int     *COUNT,   // [NRAY] total number of rays entering the cloud
                         const      int      LEADING, // leading edge
                         const      REAL3    POS0,    // initial position of ray 0
                         const      float3   DIR,     // direction of the rays
                         __global   int     *LCELLS,   
                         __constant int     *OFF,
                         __global   int     *PAR,
                         __global   float   *RHO,
                         __global   float   *BUFFER_ALL
                      ) {   
   // OT3 = split rays at the leading edge and add ***part of*** siderays (hitting sides wrt main direction)
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge, if ray exits through a side, a new one is created 
   // on the opposite side. Ray ends when the downstream edge is reached
   
   // This version tries to make sure we make all the rays so that there is no variation in the
   // path lengths between the cells (avoids the need for PL weighting!!)
   // When entering a refined cell with level0, split the ray to four on the leading edge, like in PathsOT2
   // Regarding additional rays hitting the sides of the refined cell, check the two upstream neighbours.
   // If the neighbour is:
   //   - at level >= level0, do nothing -- refined rays will come from the neighbour
   //   - otherwise, test 4 refined rays that might hit the current refined cell from that side
   //     skip the rays that will come otherwise at some point from the neighbour
   int  ls = get_local_size(0),  gid = get_group_id(0) ;
   if  (gid>=NRAY) return ;
   if  (get_local_id(0)!=0)    return ;     // a single work item from each work group !!
   
   int  INDEX, SID, NBUF=0, nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2, sr, sid, level, b1, b2, ind ;
   int  OTL, OTI, OTLO, OTL_RAY, c1, c2, i, SIDERAYS_ADDED=0, oti, otl, otl_ray ;
   float tpl=0.0f, dx, dr, s ;
   __global float *BUFFER = &(BUFFER_ALL[gid*(50+CHANNELS)*512]) ;
   COUNT[gid] = 0 ;         // COUNT[NRAY] .... now ~ COUNT[NWG]
   REAL3 RDIR ;    RDIR.x = DIR.x ; RDIR.y = DIR.y ;  RDIR.z = DIR.z ;
   
   
   // when split done only on the leading edge -- HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
   if (LEADING==0) { HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;   }
   if (LEADING==1) { HEAD[0] = 1 ;   HEAD[1] = 3 ;   HEAD[2] = 5 ;  HEAD[3] = 7 ;   }
   if (LEADING==2) { HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;   }
   if (LEADING==3) { HEAD[0] = 2 ;   HEAD[1] = 3 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;   }
   if (LEADING==4) { HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;   }
   if (LEADING==5) { HEAD[0] = 4 ;   HEAD[1] = 5 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;   }
   
   int *SUBS ;
   // each root ray shifted by two grid units ... gid over  (NX+1)/2 * (NY+1)/2 or similar number of rays
   REAL3 POS = POS0, pos, pos0, pos1 ;
# if (FIX_PATHS>0)
   float POS_LE ;
# endif
   switch (LEADING) {
    case 0: POS.x =    EPS ; POS.y += TWO*(gid%ny) ; POS.z += TWO*(int)(gid/ny) ;  break ;
    case 1: POS.x = NX-EPS ; POS.y += TWO*(gid%ny) ; POS.z += TWO*(int)(gid/ny) ;  break ;
    case 2: POS.y =    EPS ; POS.x += TWO*(gid%nx) ; POS.z += TWO*(int)(gid/nx) ;  break ;
    case 3: POS.y = NY-EPS ; POS.x += TWO*(gid%nx) ; POS.z += TWO*(int)(gid/nx) ;  break ;
    case 4: POS.z =    EPS ; POS.x += TWO*(gid%nx) ; POS.y += TWO*(int)(gid/nx) ;  break ;
    case 5: POS.z = NZ-EPS ; POS.x += TWO*(gid%nx) ; POS.y += TWO*(int)(gid/nx) ;  break ;
   }
# if (FIX_PATHS>0)
   POS_LE = POS ;
# endif
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;     // remain at root level, not yet going to leaf
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;
   OTLO      =  OTL ;
   OTL_RAY   =  0 ;
   
   if (INDEX>=0)  COUNT[gid] += 1 ;  // number of incoming rays
   
# if (DEBUG>0)
   printf("CREATE    ") ;
   report(OTL, OTI, OTL_RAY, &POS, OFF, PAR) ; 
# endif   
   
   while(INDEX>=0) {  // INDEX may refer to a cell that is not a leaf
      
      
      sr   =  0 ;                     // no subrays added so far
      c1   =  NBUF*(50+CHANNELS) ;    // 50+CHANNELS elements per buffer entry
      SID  =  -1 ;
      
      
      
      // If we are not in a leaf, we have gone to go one level up,
      // add >=three rays, pick one of them as the current, return to the beginning of the loop.
      if (RHO[INDEX]<=0.0f) {                    // go to the sub-cell and add sibling rays
         
         POS.x  =  TWO*fmod(POS.x, ONE) ;      // coordinate inside parent cell [0,1]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         dx     =  -RHO[INDEX] ;                 // OTI of the first subcell in octet
         OTL   +=  1  ;                          // step to next level = refined level
         OTI    =  *(int *)&dx ;                 // OTI for the first child in octet, not necessarily a leaf
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // original subcell
         OTI   +=  SID;                          // cell in octet, cell of the main ray
         
         c2     =  0 ;                           // set c2=1 if ray is to be split (is this the leading cell edge?)
         if ((LEADING==0)&&(POS.x<  EPS2)) {     // approximately only rays that hit the upstream side
            c2 = 1 ;   //     POS.x =      EPS ;
         }
         if ((LEADING==1)&&(POS.x>TMEPS2)) {      // TTEPS = 2-TEPS, here we are always inside an octet
            c2 = 1 ;   //     POS.x = TWO-EPS ;
         }
         if ((LEADING==2)&&(POS.y<  EPS2)) {
            c2 = 1 ;   //     POS.y =      EPS ;
         }
         if ((LEADING==3)&&(POS.y>TMEPS2)) {
            c2 = 1 ;   //     POS.y = TWO-EPS ;
         }
         if ((LEADING==4)&&(POS.z<  EPS2)) {
            c2 = 1 ;   //     POS.z =      EPS ;
         }
         if ((LEADING==5)&&(POS.z>TMEPS2)) {
            c2 = 1 ;   //     POS.z = TWO-EPS ;
         }
         
         
         if (c2>0) {  // split the ray and choose one of the new ones to be followed first
            BUFFER[c1+0]  =  OTL ;                   // level where the split is done, OTL>this =>
            BUFFER[c1+1]  =  I2F(OTI) ;              // buffer contains the OTI of the original ray
            // Store the original MAIN RAY to buffer as the first one, local coordinates
            BUFFER[c1+2]  =  POS.x ;
            BUFFER[c1+3]  =  POS.y ;
            BUFFER[c1+4]  =  POS.z ;
            BUFFER[c1+5]  =  OTL_RAY ;                    // main ray exists at levels >= OTL_RAY
            
            // Add two subrays to buffer, take third one as the current ray
            if (                  HEAD[0]==SID) {  // 0=original => (1,2) to buffer, 3 as current
               sid              = HEAD[1] ;
               BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (int)( sid % 2) ;
               BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
               BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (int)( sid/4) ;
               sid              = HEAD[2] ;
               BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (int)( sid % 2) ;
               BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + (int)((sid/2)%2)  ;
               BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (int)( sid/4) ;
               SID              = HEAD[3] ;  // SID of the subray to be followed first
            } else {
               if (                  HEAD[1]==SID) {
                  sid              = HEAD[2] ;
                  BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (int)( sid   %2) ;
                  BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                  BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (int)( sid/4) ;
                  sid              = HEAD[3] ;
                  BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (int)( sid   %2) ;
                  BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                  BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (int)( sid/4) ;
                  SID              = HEAD[0] ;
               } else {
                  if (                  HEAD[2]==SID) {
                     sid              = HEAD[3] ;
                     BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (int)( sid % 2) ;
                     BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                     BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (int)( sid/4) ;
                     sid              = HEAD[0] ;
                     BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (int)( sid % 2) ;
                     BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                     BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (int)( sid/4) ;
                     SID              = HEAD[1] ;
                  } else {
                     if (                  HEAD[3]==SID) {
                        sid              = HEAD[0] ;
                        BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (int)( sid % 2) ;
                        BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                        BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (int)( sid/4) ;
                        sid              = HEAD[1] ;
                        BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (int)( sid % 2) ;
                        BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                        BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (int)( sid/4) ;
                        SID              = HEAD[2] ;
                     } else {
                        ;  // printf("???\n") ;
                     }
                  }
               }
            }
            // for the two subrays just added, update their OTL_RAY
            BUFFER[c1+5+4*1]  =  OTL ;       BUFFER[c1+5+4*2]  =  OTL ;
            sr = 3 ;   // so far the original main ray and two split subrays added
            
         }  //  c2>0  == other leading-edge subrays added
      } // RHO < 0.0 == we entered refined region
      
      
      
      if (sr>0) { // some leading-edge rays were added to buffer --- SID indicates the current subray
         for(i=sr; i<12; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
         // SID>=0 => We added leading edge rays, old main ray is in buffer, SID refers to one of the subrays
         // update OTI and POS; OTI is still the cell for the main ray
         OTI       =  (*(int *)&dx) + SID ;    // the new ray to be followed, OTI = index of first subcell + SID
         POS.x     =  fmod(POS.x,ONE) + (int)( SID%2)    ;
         POS.y     =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
         POS.z     =  fmod(POS.z,ONE) + (int)( SID/4)    ;
         OTL_RAY   =  OTL ;  // when we reach OTL<OTL_RAY, this ray will be terminated
         NBUF++ ;            // >= 1 subrays were added, NBUF increases just by one
         
# if (DEBUG>0)
         for(int i=0; i<sr; i++) {
            printf("ADD.LEAD  ") ;
            otl   =  BUFFER[c1+0] ;
            oti   =  F2I(BUFFER[c1+1]) ;
            pos.x =  BUFFER[c1+2+4*i] ; pos.y =  BUFFER[c1+3+4*i] ; pos.z =  BUFFER[c1+4+4*i] ;
            sid   =  4*(int)floor(pos.z) + 2*(int)floor(pos.y) + (int)floor(pos.x) ; 
            oti   =  8*(int)(OTI/8) + SID ;
            otl_ray   =  BUFFER[c1+5+4*i] ;            
            report(otl, oti, otl_ray, &pos, OFF, PAR) ; 
         }
         printf("CONTINUE  ") ;
         report(OTL, OTI, OTL_RAY, &POS, OFF, PAR) ; 
# endif         
      }
      
      
      
      
      // Check on EVERY STEP if some refined rays enter via the side walls
      //  neighbour is refined to level >= OTL  =>   do not add any rays
      //  neighbour is refined to level  < OTL  =>   add side rays, unless common to neighbout at lower level
      
      // SIDERAYS_ADDED is true this is a main ray directly from the buffer and its siderays were already added
      // If we added above leading-edge rays, the current ray is one of the subrays and we will not
      // add siderays until later (when the main ray is taken from the buffer)
      
      if ((OTL_RAY<OTL)&&(SIDERAYS_ADDED==0)) {  
         
         BUFFER[c1+0] = OTL ;
         BUFFER[c1+1] = I2F(OTI) ;
         
         c2     =  0 ; 
         if ((LEADING==0)&&(POS.x<  EPS2))  c2 = 1 ;
         if ((LEADING==1)&&(POS.x>TMEPS2))  c2 = 1 ;
         if ((LEADING==2)&&(POS.y<  EPS2))  c2 = 1 ;
         if ((LEADING==3)&&(POS.y>TMEPS2))  c2 = 1 ;
         if ((LEADING==4)&&(POS.z<  EPS2))  c2 = 1 ;
         if ((LEADING==5)&&(POS.z>TMEPS2))  c2 = 1 ; 
         
         if (c2==1) {  // we have the main ray hitting the octet leading edge - add siderays
            
            switch (LEADING) {               
             case 0:  // main direction X
             case 1:
               // We are at refined level => POS the same irrespective of PLMETHOD
               // RootPos returns pos0 in global coordinates [0,NX] but that is used only to check ray positions.
               pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;          // current position in root grid coordinates
               if (LEADING==0)  pos1 = POS0 + RDIR * ((pos0.x-ZERO)/RDIR.x) ; // root ray at the current plane
               else             pos1 = POS0 + RDIR * ((pos0.x-NX  )/RDIR.x) ; // ... in root grid coordinates
               pos0 = (pos0-pos1)*pown(TWO, OTL) ;     // current - rootray, in OTL steps
               // First offset in Y, second offset in Z
               pos.x  = 1.013579f ;  pos.y = 1.013579f ;  pos.z = 1.013579f ; // ANY pos in current octet, local coord.
               pos.y += (DIR.y>ZERO) ? -1.02f : +1.02f ;        // ANY pos in the upstream neighbour
               level  = OTL ;   ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1)   ;   // extract level of the neighbour cell
               if (level<OTL) {  // neighbour not refined to same level => check subrays from that side
                  dr  = pown(TWO, OTL-level) ;    // neighbour ray step, as OTL ray steps
                  dr  = TWO ; // if  level < OTL-1 < OTL, OTL-1 siderays must have been added already ????
                  for(int a=-1; a<=3; a++) {      // ray offset to upstream direction  (1,2) or (2,3) ... test all
                     for(int b=-1; b<=3; b++) {   // orthogonal direction, (-1,0) or (0,+1)... test all
                        pos     =  POS ;                                      // our main ray position
                        pos.y  +=  (DIR.y>0.0f) ?  (-ONE*a) : (+ONE*a) ;    // jump to upstream ray
                        if ((pos.y>=ZERO)&&(pos.y<=TWO)) continue ;          // it is in current cell - skip here
                        pos.z  +=  (DIR.z>0.0f) ?  (-ONE*b) : (+ONE*b) ;
                        pos1    =  pos0 ;         // offset from rootray, as level OTL steps
                        pos1.y +=  (DIR.y>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.z +=  (DIR.z>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.y - dr*round(pos1.y/dr))<0.0004f) &&
                             (fabs(pos1.z - dr*round(pos1.z/dr))<0.0004f)) continue ; // ray exists in neighbour
                        // step the candidate ray to upstream cell side border
                        if (DIR.y>0.0f) s =  (ZERO-pos.y)/DIR.y ;
                        else            s =  (TWO-pos.y)/DIR.y ;
                        pos    += (s+EPS)*DIR ;         // now possibly on the border of the current cell
                        if ((pos.x<=ZERO)||(pos.x>=TWO)||(pos.z<=ZERO)||(pos.z>=TWO)) continue ; // no hit
                        // ok, add this ray from the upstream direction to buffer as new subray
                        pos.y   =  clamp(pos.y, EPS, TMEPS) ; // make sure pos is inside the octet
                        BUFFER[c1+2+4*sr] = pos.x ;
                        BUFFER[c1+3+4*sr] = pos.y ;
                        BUFFER[c1+4+4*sr] = pos.z ;
                        BUFFER[c1+5+4*sr] = OTL ;    // ray created at the current level
                        sr += 1 ;
                     }
                  }
               }
               // First offset Z, second offset Y
               pos.x  =  1.013579f ; pos.y = 1.013579f ; pos.z = 1.013579f ;    // centre of current octet
               pos.z +=  (DIR.z>0.0f) ? -1.02f : +1.02f ;         // tpos in the upstream neighbour
               level  =  OTL ; ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1) ;  // extract level of the neighbour cell
               if (level<OTL) {  // neighbour not refined to the same level => check subrays entering from that side
                  dr  =  pown(TWO, OTL-level) ;   // step between rays at neighbour discretisation level
                  dr  =  TWO ;
                  for(int a=-1; a<=3; a++) {      // ray offset to upstream direction  (1,2) or (2,3) ... test all
                     for(int b=-1; b<=3; b++) {   // orthogonal direction, (-1,0) or (0,+1)... test all
                        pos     =  POS ;                                      // our main ray position
                        pos.z  += (DIR.z>0.0f) ?  (-ONE*a) : (+ONE*a) ;     // jump to upstream ray
                        if ((pos.z>=ZERO)&&(pos.z<=TWO)) continue ;         // still in current cell - skip here
                        pos.y  += (DIR.y>0.0f) ?  (-ONE*b) : (+ONE*b) ;     // step in cross direction
                        // pos1 = distance ro rootray, in OTL steps -- divisible with dr ??
                        pos1    =  pos0 ;    // current - rootray
                        pos1.z += (DIR.z>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.y += (DIR.y>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.z - dr*round(pos1.z/dr))<0.0005f) &&
                             (fabs(pos1.y - dr*round(pos1.y/dr))<0.0005f)) continue ;
                        // step the candidate ray to upstream side border of the current octet -- here along z
                        if (DIR.z>0.0f) s =  (ZERO-pos.z)/RDIR.z ;
                        else            s =  (TWO -pos.z)/RDIR.z ;
                        pos    += (s+EPS)*RDIR ;  // now possibly on the border of the current cell
                        if ((pos.x<=ZERO)||(pos.x>=TWO)||(pos.y<=ZERO)||(pos.y>=TWO)) continue ; // no hit
                        // ok, the ray from upstream direction hits this cell => add to buffer as new subray
                        pos.z   =  clamp(pos.z, EPS, TWO-TMEPS) ; // make sure pos is inside the octet
                        BUFFER[c1+2+4*sr] = pos.x ;
                        BUFFER[c1+3+4*sr] = pos.y ;
                        BUFFER[c1+4+4*sr] = pos.z ;
                        BUFFER[c1+5+4*sr] = OTL ;    // ray created at the current level
                        sr += 1 ;
                     }
                  }
               }
               break ;
             case 2:  // Y = main direction
             case 3:  
               // Main offset X, secondary offset Z
               // Start by finding the offset relative to POS0... so that we can check if some subrays
               // already are coming from the neighbout
               pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;         // current root grid position
               if (LEADING==2)  pos1 = POS0 + DIR * ((pos0.y-ZERO)/DIR.y) ;  // pos0 root grid ray at the current plane
               else             pos1 = POS0 + DIR * ((pos0.y-NY  )/DIR.y) ;
               pos0   = (pos0-pos1)*pown(TWO, OTL) ;  //  current - rootray, in OTL steps, current plane, OTL units
               pos.x  = 1.013579f ; pos.y = 1.013579f ; pos.z = 1.013579f ; // centre of current octet
               pos.x += (DIR.x>0.0f) ? -1.02f : +1.02f ;          // pos is now in the upstream neighbour
               level  = OTL ;   ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1) ;   // extract level of the neighbour cell
               if (level<OTL) {  // neighbour not refined to same level => check subrays from that side
                  dr  = pown(TWO, OTL-level) ;       // step between rays at neighbour discretisation level
                  dr  = TWO ;
                  for(int a=-1; a<=3; a++) {         // main offset along      X
                     for(int b=-1; b<=3; b++) {      // secondary offset along Z
                        pos     =  POS ;                                      // our main ray position
                        pos.x  += (DIR.x>0.0f) ?  (-ONE*a) : (+ONE*a) ;     // jump to upstream ray
                        if ((pos.x>=ZERO)&&(pos.x<=TWO)) continue ;          // still in current cell - skip here
                        pos.z  += (DIR.z>0.0f) ?  (-ONE*b) : (+ONE*b) ;     // secondary offset
                        //
                        pos1    =  pos0 ;
                        pos1.x += (DIR.x>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.z += (DIR.z>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.x - dr*round(pos1.x/dr))<0.0005f) &&
                             (fabs(pos1.z - dr*round(pos1.z/dr))<0.0005f))  continue ;                        
                        if (DIR.x>0.0f) s =  (ZERO-pos.x)/DIR.x ;
                        else            s =  (TWO-pos.x)/DIR.x ;
                        pos   += (s+EPS)*RDIR ;  // now possibly on the border of the current cell
                        if ((pos.y<=ZERO)||(pos.y>=TWO)||(pos.z<=ZERO)||(pos.z>=TWO)) continue ; // no hit
                        // ok, the ray from upstream direction hits this cell => add to buffer as new subray
                        pos.x =  clamp(pos.x, EPS, TMEPS) ; // make sure pos is inside the octet
                        BUFFER[c1+2+4*sr] = pos.x ;
                        BUFFER[c1+3+4*sr] = pos.y ;
                        BUFFER[c1+4+4*sr] = pos.z ;
                        BUFFER[c1+5+4*sr] = OTL ;   // ray created at the current level
                        sr += 1 ;
                     }
                  }
               }
               // Main offset Z, secondary X
               pos.x  = 1.013579f ; pos.y = 1.013579f ; pos.z = 1.013579f ;    // centre of current octet
               pos.z += (DIR.z>0.0f) ? -1.02f : +1.02f ;             // pos is now in the upstream neighbour
               level  = OTL ;  ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1) ;      // extract level of the neighbour cell
               if (level<OTL) {  // neighbour not refined to the same level => check subrays entering from that side
                  dr  = pown(TWO, OTL-level) ;   // step between rays at neighbour discretisation level, in root grid units
                  dr  = TWO ;
                  for(int a=-1; a<=3; a++) {     // upstream direction   =  Z
                     for(int b=-1; b<=3; b++) {  // orthogonal direction =  X
                        pos     =  POS ;                                      // our main ray position
                        pos.z  += (DIR.z>0.0f) ?  (-ONE*a) : (+ONE*a) ;     // jump to upstream ray -- in Z
                        if ((pos.z>=ZERO)&&(pos.z<=TWO)) continue ;         // still in current cell - skip here
                        pos.x  += (DIR.x>0.0f) ?  (-ONE*b) : (+ONE*b) ; 
                        // check if the ray would come from the neighbour in any case and can be skipped here
                        pos1    =  pos0 ;   // current - rootray
                        pos1.z += (DIR.z>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.x += (DIR.x>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.z - dr*round(pos1.z/dr))<0.0004f) &&
                             (fabs(pos1.x - dr*round(pos1.x/dr))<0.0004f)) continue ;
                        // step the candidate ray back to upstream cell side border -- along Z
                        if (DIR.z>0.0f) s =  (ZERO-pos.z)/RDIR.z ;
                        else            s =  (TWO -pos.z)/RDIR.z ;
                        pos   += (s+EPS)*RDIR ;  // now possibly on the border of the current cell
                        if ((pos.x<=ZERO)||(pos.x>=TWO)||(pos.y<=ZERO)||(pos.y>=TWO)) continue ; // no hit
                        // ok, the ray from upstream direction hits this cell => add to buffer as new subray
                        pos.z =  clamp(pos.z, EPS, TMEPS) ; // make sure pos is inside the octet
                        BUFFER[c1+2+4*sr] = pos.x ;
                        BUFFER[c1+3+4*sr] = pos.y ;
                        BUFFER[c1+4+4*sr] = pos.z ;
                        BUFFER[c1+5+4*sr] = OTL ;   // ray created at the current level
                        sr += 1 ;
                     }
                  }
               }
               break ;
               
             case 4:  // Main direction Z
             case 5:  // Main offset X, secondary offset Y
               pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;         // current main ray, root grid units
               if (LEADING==4)  pos1 = POS0 + DIR * ((pos0.z-ZERO)/DIR.z) ;  // root ray, current plane
               else             pos1 = POS0 + DIR * ((pos0.z-NZ  )/DIR.z) ;
               pos0   = (pos0-pos1)*pown(TWO, OTL) ;  //  current - rootray, in OTL steps, current plane, OTL units
               // First offset X, second offset Y
               pos.x  = 1.013579f ; pos.y = 1.013579f ; pos.z = 1.013579f ;  // centre of current octet
               pos.x += (DIR.x>0.0f) ? (-1.02f) : (+1.02f) ;          // pos is now in the upstream neighbour
               level  = OTL ;   ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1) ;     // extract level of the neighbour cell
               if (level<OTL) {                   // neighbour not refined to same level => check subrays from that side
                  dr  = pown(TWO, OTL-level) ;    // step between neighbour rays, in units of OTL step
                  dr  = TWO ;
                  for(int a=-1; a<=3; a++) {      // main offset along      X
                     for(int b=-1; b<=3; b++) {   // secondary offset along Y
                        pos     =  POS ;                                      // our main ray position
                        pos.x  += (DIR.x>0.0f) ?  (-ONE*a) : (+ONE*a) ;     // jump to upstream ray
                        if ((pos.x>=ZERO)&&(pos.x<=TWO)) continue ;          // still in current cell - skip here
                        pos.y  += (DIR.y>0.0f) ?  (-ONE*b) : (+ONE*b) ;     // secondary offset
                        //
                        pos1    =  pos0 ;         // current - rootray, in OTL steps
                        pos1.x += (DIR.x>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.y += (DIR.y>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.x - dr*round(pos1.x/dr))<0.0005f) &&
                             (fabs(pos1.y - dr*round(pos1.y/dr))<0.0005f)) continue ;
                        // step to upstream border along X direction
                        if (DIR.x>0.0f) s =  (ZERO-pos.x)/RDIR.x ;
                        else            s =  (TWO -pos.x)/RDIR.x ;
                        pos   += (s+EPS)*RDIR ;  // now possibly on the border of the current cell
                        if ((pos.y<=ZERO)||(pos.y>=TWO)||(pos.z<=ZERO)||(pos.z>=TWO)) continue ; // no hit
                        // ok, the ray from upstream direction hits this cell => add to buffer as new subray
                        pos.x =  clamp(pos.x, EPS, TMEPS) ; // make sure pos is inside the octet
                        BUFFER[c1+2+4*sr] = pos.x ;
                        BUFFER[c1+3+4*sr] = pos.y ;
                        BUFFER[c1+4+4*sr] = pos.z ;
                        BUFFER[c1+5+4*sr] = OTL ;    // ray created at the current level
                        sr += 1 ;
                     }
                  }
               }
               // Main offset Y, secondary X
               pos.x  = 1.013579f ; pos.y = 1.013579f ; pos.z = 1.013579f ;  // centre of current octet
               pos.y += (DIR.y>0.0f) ? (-1.02f) : (+1.02f) ;                 // pos is now in the upstream neighbour
               level  = OTL ;   ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1) ;   // extract level of the neighbour cell
               if (level<OTL) {  // neighbour not refined to the same level => check subrays entering from that side
                  dr  = pown(TWO, OTL-level) ;    // step between neighbour rays, in units of OTL step
                  dr  = TWO ;
                  for(int a=-1; a<=3; a++) {      // upstream direction   =  Z
                     for(int b=-1; b<=3; b++) {   // orthogonal direction =  X
                        pos     =  POS ;                                      // our main ray position
                        pos.y  += (DIR.y>0.0f) ?  (-ONE*a) : (+ONE*a) ;     // jump to upstream ray -- in Z
                        if ((pos.y>=ZERO)&&(pos.y<=TWO)) continue ;          // still in current cell - skip here
                        pos.x  += (DIR.x>0.0f) ?  (-ONE*b) : (+ONE*b) ;     // secondary offset
                        // pos1 = distance ro rootray, in OTL steps -- divisible with dr ??
                        pos1    =  pos0 ;    // current - rootray
                        pos1.y += (DIR.y>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.x += (DIR.x>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.y - dr*round(pos1.y/dr))<0.0005f) &&
                             (fabs(pos1.x - dr*round(pos1.x/dr))<0.0005f)) continue ;
                        // step the candidate ray back to upstream cell side border -- along Z
                        if (DIR.y>0.0f) s =  (ZERO-pos.y)/RDIR.y ;
                        else            s =  (TWO -pos.y)/RDIR.y ;
                        pos    +=  (s+EPS)*RDIR ;  // now possibly on the border of the current cell
                        if ((pos.x<=ZERO)||(pos.x>=TWO)||(pos.z<=ZERO)||(pos.z>=TWO)) continue ; // no hit
                        // ok, the ray from upstream direction hits this cell => add to buffer as new subray
                        pos.y   =  clamp(pos.y, EPS, TMEPS) ; // make sure pos is inside the octet
                        BUFFER[c1+2+4*sr] = pos.x ;
                        BUFFER[c1+3+4*sr] = pos.y ;
                        BUFFER[c1+4+4*sr] = pos.z ;
                        BUFFER[c1+5+4*sr] = OTL ;   // ray created at the current level
                        sr += 1 ;
                     }
                  }
               }
               break ;
               
            }  // switch  --- adding possible sidewall rays
            
            
            
            if (sr>0) {
               for(i=sr; i<12; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
               // We want to do the siderays before the current main ray
               //  => switch the main ray with the sr=0 entry in the buffer
               // AND INDICATE IN BUFFER THAT THE RAY HAS ALREADY BEEN SPLIT SO THAT WE WHEN GET IT BACK
               // FROM THE BUFFER, WE DO NOT ADD SIDERAYS AGAIN
               i     = OTI ;
               // get data for one of the siderays
               pos.x = BUFFER[c1+2] ; pos.y = BUFFER[c1+3] ; pos.z = BUFFER[c1+4] ;  c2 = BUFFER[c1+5] ;
               // WE SUBTRACT 100 FROM OTL_RAY TO INDICATE THAT SIDERAYS WERE ALREADY ADDED =>
               // WHEN THIS MAIN RAY IS POPED FROM BUFFER, WE DO NOT ADD SIDERAYS AGAIN
               BUFFER[c1+2] = POS.x ; BUFFER[c1+3] = POS.y ; BUFFER[c1+4] = POS.z ;  BUFFER[c1+5] = OTL_RAY-100 ;
               POS      =  pos ;    
               OTL_RAY  =  c2 ;
               SID      =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
               OTI      =  8*(int)(OTI/8) + SID ;  // OTI in the buffer was for the main ray
               NBUF += 1 ;
               
               
# if (DEBUG>0)
               for(int i=0; i<sr; i++) {
                  printf("ADD.SIDE  ") ;
                  otl   =  BUFFER[c1+0] ;
                  oti   =  F2I(BUFFER[c1+1]) ;
                  pos.x =  BUFFER[c1+2+4*i] ; pos.y =  BUFFER[c1+3+4*i] ; pos.z =  BUFFER[c1+4+4*i] ;
                  sid   =  4*(int)floor(pos.z) + 2*(int)floor(pos.y) + (int)floor(pos.x) ; 
                  oti   =  8*(int)(OTI/8) + SID ;
                  otl_ray   =  BUFFER[c1+5+4*i] ;            
                  report(otl, oti, otl_ray, &pos, OFF, PAR) ; 
               }
               printf("CONTINUE  ") ;
               report(OTL, OTI, OTL_RAY, &POS, OFF, PAR) ;                               
# endif
            } // if sr>0
            
         }  // c2==1  --- we are on the leading edge
      }  // OTL_RAY < OTL
      
      
      
      
      INDEX = OFF[OTL]+OTI ;   // global index -- must be now >=0 since we went down a level
      
      
      // if not yet a leaf, jump back to start of the loop => another step down
      if (RHO[INDEX]<=0.0f) {
         continue ;
      }
      
      
      // if (FOLLOW) printf("*** PRESTEP  [%d,%d]  %8.4f %8.4f %8.4f \n", OTL, OTI, POS.x, POS.y, POS.z, dr) ;
      
      
      // we come here only if we have a ray that is inside a leaf node => can do the step forward
      // if the step is to refined cell, (OTL, OTI) will refer to a cell that is not a leaf node
      // ==> the beginning of the present loop will have to deal with the step to higher refinement
      // level and the creation of three additional rays
      OTLO    =  OTL ;  c2 = OTI ;
      // get step but do not yet go to level > OTLO
      
# if (DEBUG>0)
      pos1 = POS ;
      RootPos(&pos1, OTL, OTI, OFF, PAR) ;      
      c2 = OFF[OTL]+OTI ;
      printf("@ %9.6f %9.6f %9.6f   %d  ", pos1.x, pos1.y, pos1.z, OTLO) ;      
# endif
      
# if (FIX_PATHS>0)
      dr      =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, &POS_LE, LEADING) ; // step [GL] == root grid units !!
# else
      dr      =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL   , LEADING) ; // step [GL] == root grid units !!
# endif
      
# if (DEBUG>0)
      pos1 = POS ;
      RootPos(&pos1, OTL, OTI, OFF, PAR) ;
      printf("%9.6f %9.6f %9.6f  %d   --- %3d => %3d\n", pos1.x, pos1.y, pos1.z, OTL, c2, OFF[OTL]+OTI) ;
# endif
      
      
      SIDERAYS_ADDED = 0 ;  // whenever we have made one step, it is again safe to test if main ray has siderays
      
      
      
# if 0
      PL[INDEX] += dr ;
# else
      AADD(&(PL[INDEX]), dr) ;
# endif
      
      
      
      tpl       += dr ;               // just the total value for current idir, ioff
      
      // The new index at the end of the step  --- OTL >= OTL  => INDEX may not refer to a leaf cell
      INDEX    =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;  // OTL, OTI, POS already updated = end of the current step
      
      
      if (INDEX>=0) {
         if (RHO[INDEX]<=0.0f) {  // we ended up in a parent cell
            continue ; // we moved to refined region, handled at the beginning of the main loop
         }
      }
      
      
      if (OTL<OTL_RAY) {   // up in hierarchy and this ray is terminated -- cannot be on root grid
         INDEX=-1 ;        // triggers the generation of a new ray below
      } else {    
         if (INDEX<0) {    // ray exits the cloud on root grid, possibly create a new OTL=0 ray on the other side
            if (POS.x>=NX  ) {   if (LEADING!=0)   POS.x =    EPS ;   }
            if (POS.x<=ZERO) {   if (LEADING!=1)   POS.x = NX-EPS ;   }
            if (POS.y>=NY  ) {   if (LEADING!=2)   POS.y =    EPS ;   }
            if (POS.y<=ZERO) {   if (LEADING!=3)   POS.y = NY-EPS ;   } 
            if (POS.z>=NZ  ) {   if (LEADING!=4)   POS.z =    EPS ;   }
            if (POS.z<=ZERO) {   if (LEADING!=5)   POS.z = NZ-EPS ;   } 
            IndexGR(&POS, &OTL, &OTI, RHO, OFF) ; // not necessarily a leaf!
            INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
            if (INDEX>=0) {    // new level-0 ray started on the opposite side
               OTL_RAY = 0 ;    OTLO    = 0 ;   COUNT[gid] += 1 ;
# if (DEBUG>0)
               printf("MIRROR    ") ;
               report(OTL, OTI, OTL_RAY, &POS, OFF, PAR) ;
# endif
               continue ;
            }
         } // if INDEX<0
      }
      
      
      // [C] if INDEX still negative, current ray truly ended => take new ray from the buffer, if such exist
      if ((INDEX<0)&&(NBUF>0)) {            // NBUF>0 => at least one ray exists in the buffer
         c1    =  (NBUF-1)*(50+CHANNELS) ;  // 50+CHANNELS elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // level where sibling rays were created
         OTLO  =  OTL ;
         OTI   =  F2I(BUFFER[c1+1]) ;       // OTI for the original ray that was split
         // a maximum of 12 rays per buffer entry, the entry sr=0 is the original main ray => choose last
         for(sr=11; sr>=0; sr--) {         
            dr    =  BUFFER[c1+2+4*sr] ;
            if (dr>-0.1f) break ;      // found a ray
         }
         POS.x     =  BUFFER[c1+2+4*sr] ;         
         POS.y     =  BUFFER[c1+3+4*sr] ;
         POS.z     =  BUFFER[c1+4+4*sr] ;
         OTL_RAY   =  BUFFER[c1+5+4*sr] ;    
         SIDERAYS_ADDED = 0 ;
         if (OTL_RAY<0) { // <0 values meant that main ray had already added siderays
            OTL_RAY = OTL_RAY+100 ;   SIDERAYS_ADDED = 1 ;
         }
         // we must have found a ray  -- figure out its OTI (index within level OTL)
         SID     =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
         OTI     =  8*(int)(OTI/8) + SID ;    // OTI in the buffer was for the original ray that was split
         INDEX   =  OFF[OTL]+OTI ;    // global index -- must be >=0 !!!
         
# if (DEBUG>0)
         printf("FROMBUF%d  ", SIDERAYS_ADDED) ;
         report(OTL, OTI, OTL_RAY, &POS, OFF, PAR) ; 
# endif         
         // this stored ray is at level OTL but may be in a cell that is itself still further
         // refined => this will be handled at the start of the main loop => possible further
         // refinement before on the rays is stepped forward
         BUFFER[c1+2+4*sr] = -1.0f ;              // mark as used
         if (sr==0) NBUF -= 1 ;       // was last of the <=12 subrays in this buffer entry
      }  // (INDEX<0)&&(NBUF>0)
      
      
      // if (INDEX<0) { if (NBUF<=0)   FOLLOW=0 ;  }
      
   } // while INDEX>=0  --- stops when buffer is empty and the main level-0 ray has exited
   
   TPL[gid] = tpl ;
}  // END OF PathsOT3      no OTL_SPLIT,   5*  changed to 4*






__kernel void UpdateOT3(  // 
                          __global float  *PL,      //  0
# if (WITH_HALF==1)
                          __global short4 *CLOUD,   //  1 [CELLS]: vx, vy, vz, sigma
# else
                          __global float4 *CLOUD,   //  1 [CELLS]: vx, vy, vz, sigma
# endif
                          GAUSTORE  float *GAU,     //  2 precalculated gaussian profiles [GNO,CHANNELS]
                          constant int2   *LIM,     //  3 limits of ~zero profile function [GNO]
                          const float      Aul,     //  4 Einstein A(upper->lower)
                          const float      A_b,     //  5 (g_u/g_l)*B(upper->lower)
                          const float      GN,      //  6 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                          const float      APL,     //  7 average path length [GL]
                          const float      BG,      //  8 background value (photons)
                          const float      DIRWEI,  //  9 weight factor (based on current direction)
                          const float      EWEI,    // 10 weight 1/<1/cosT>/NDIR
                          const int        LEADING, //  0 leading edge
                          const REAL3      POS0,    // 11 initial position of id=0 ray
                          const float3     DIR,     // 12 ray direction
                          __global float  *NI,      // 13 [CELLS]:  NI[upper] + NB_NB
                          __global float  *RES,     // 14 [CELLS]: SIJ, ESC ---- or just [CELLS] for SIJ
                          __global float  *NTRUES,  // 15 [NWG*MAXCHN]   --- NWG>=simultaneous level 0 rays
# if (WITH_CRT>0)
                          constant float *CRT_TAU,  //  dust optical depth / GL
                          constant float *CRT_EMI,  //  dust emission photons/c/channel/H
# endif                     
# if (BRUTE_COOLING>0)
                          __global float   *COOL,   // 14,15 [CELLS] = cooling 
# endif
                          __global   int   *LCELLS, //  16
                          __constant int   *OFF,    //  17
                          __global   int   *PAR,    //  18
                          __global   float *RHO,    //  19  -- needed only to describe the hierarchy
                          __global   float *BUFFER_ALL  //  20 -- buffer to store split rays
                       )  {   
   // Each ***WORK GROUP*** processes one ray. The rays are two cells apart to avoid
   // synchronisation problems (???). Rays start on the leading edge. If ray exits through a
   // side (wrt axis closest to direction of propagation), a new one is created on the
   // opposite side and the ray ends when the downstream edge is reached.
   // 
   // As one goes to a higher hierarchy level (into a refined cell), one pushes the
   // original and two new rays to buffer. The fourth ray (one of the new ones) is
   // continued. When ray goes up in hierarchy (to a larger grain) ray is ended if that
   // was created at higher level. When ray ends, the next one is taken from the buffer
   // (stack).  As before, if level=0 ray exits from sides continue
   // If ray goes up in hierarchy, above a level where it had siblings, NTRUE is scaled *= 4
   // or, if one goes up several levels, *=4^n, where n is the number of levels with siblings
   int id  = get_global_id(0), lid = get_local_id(0), gid = get_group_id(0), ls  = get_local_size(0) ;
   if (gid>=NRAY) return ;        // one work group per ray .... NWG==NRAY
   int nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ;
   GAUSTORE float *profile ;
   __global float *BUFFER = &BUFFER_ALL[gid*(50+CHANNELS)*512] ;
   __local  float  NTRUE[CHANNELS] ;
   __local  int2   LINT ;          // SID, NBUF
   __local  float  SDT[LOCAL] ;    // per-workitem sum_delta_true
   __local  float  AE[LOCAL] ;     // per-workitem all_escaped
   float weight, w, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed, sum_delta_true, all_escaped, nu ;
   int row, shift, INDEX, c1, c2, OTL, OTI, OTLO, OTL_RAY, SID, NBUF=0, sr, sid, level, b1, b2, I, i, ind ;
   int SIDERAYS_ADDED = 0 ;
   REAL3  pos0, pos1, RDIR ;
   REAL dx, dy, dz, dr, s ;
# if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, pro, sij ;
# endif  
   RDIR.x = DIR.x ; RDIR.y = DIR.y ; RDIR.z = DIR.z ;
   
   int *SUBS ;
   REAL3 POS = POS0, pos ;
# if (FIX_PATHS>0)
   REAL3 POS_LE ;
# endif
   
   // when split done only on the leading edge -- HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
   if (LEADING<3) {
      if (LEADING==0) { 
         HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;
         POS.x =    EPS ;  POS.y += TWO*(gid%ny) ;  POS.z += TWO*(int)(gid/ny) ;
      }
      if (LEADING==1) { 
         HEAD[0] = 1 ;   HEAD[1] = 3 ;   HEAD[2] = 5 ;  HEAD[3] = 7 ;
         POS.x = NX-EPS ;  POS.y += TWO*(gid%ny) ;  POS.z += TWO*(int)(gid/ny) ;
      }
      if (LEADING==2) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;
         POS.y =    EPS ;  POS.x += TWO*(gid%nx) ;  POS.z += TWO*(int)(gid/nx) ;
      }
   } else {
      if (LEADING==3) { 
         HEAD[0] = 2 ;   HEAD[1] = 3 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.y = NY-EPS ;  POS.x += TWO*(gid%nx) ;  POS.z += TWO*(int)(gid/nx) ;
      }
      if (LEADING==4) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;
         POS.z =    EPS ;  POS.x += TWO*(gid%nx) ;  POS.y += TWO*(int)(gid/nx) ;
      }
      if (LEADING==5) { 
         HEAD[0] = 4 ;   HEAD[1] = 5 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.z = NZ-EPS ;  POS.x += TWO*(gid%nx) ;  POS.y += TWO*(int)(gid/nx) ;
      }
   }
# if (FIX_PATHS>0)
   POS_LE = POS ;
# endif
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;     // remain at root level, not yet going to leaf
   if (OTI<0) return ;
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
   OTLO      =  OTL ;
   OTL_RAY   =  0 ;
   for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BG * DIRWEI ;
   
   
   while(INDEX>=0) {  // INDEX may refer to a cell that is not a leaf
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
      sr  = 0 ;
      c1  = NBUF*(50+CHANNELS) ;
      SID = -1 ;
      
      // If we are not in a leaf, we have gone to some higher level. 
      // Go one level higher, add >=three rays, pick one of them as the current,
      // and return to the beginning of the loop.
      if (RHO[INDEX]<=0.0f) {                    // go to the sub-cell and add sibling rays
         POS.x  =  TWO*fmod(POS.x, ONE) ;      // coordinate inside parent cell [0,1]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         dx     =  -RHO[INDEX] ;                 // OTL, OTI of the parent cell
         OTL   +=  1  ;                          // step to next level = refined level
         OTI    =  *(int *)&dx ;                 // OTI for the first child in octet
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // original subcell
         OTI   +=  SID;                          // cell in octet, original ray
         c2     =  0 ;   // set c2=1 if ray is to be split (is this the leading cell edge?)
         if (LEADING<3) {
            if (LEADING==0) {  
               if (POS.x<EPS2)      c2 = 1 ; 
            } else {
               if (LEADING==1) {  
                  if (POS.x>TMEPS2) c2 = 1 ; 
               } else {
                  if (POS.y<EPS2)   c2 = 1 ; 
               }
            }
         } else {
            if (LEADING==3) { 
               if (POS.y>TMEPS2)    c2 = 1 ; 
            } else {
               if (LEADING==4) {  
                  if (POS.z<EPS2)   c2 = 1 ; 
               } else {
                  if (POS.z>TMEPS2) c2 = 1 ; 
               }
            }
         }
         // @@  rescale always when the resolution changes
         for(int i=lid; i<CHANNELS; i+=ls) {   // SCALE ON EVERY REFINEMENT EVEN WHEN NOT SPLIT
            NTRUE[i] *= 0.25f ; 
         }
         barrier(CLK_LOCAL_MEM_FENCE) ;
         
         if (c2>0) {  // split the ray and choose one of the new ones to be followed first
            for(int i=lid; i<CHANNELS; i+=ls) {   // ray effectively split to four
               BUFFER[c1+50+i] = NTRUE[i] ;
            }
            barrier(CLK_LOCAL_MEM_FENCE) ;            
            if (lid==0) {
               BUFFER[c1+0]  =  OTL ;                     // level where the split is done, OTL>this =>
               BUFFER[c1+1]  =  I2F(OTI) ;                // buffer contains the OTI of the original ray 
               BUFFER[c1+2]  =  POS.x ;                   // Store the original MAIN RAY to buffer as the first one
               BUFFER[c1+3]  =  POS.y ;
               BUFFER[c1+4]  =  POS.z ;
               BUFFER[c1+5]  =  OTL_RAY ;                 // main ray exists at levels >= OTL_RAY
            }            
            // Two new subrays added to buffer, third one becomes the current ray
            // Add first two subrays to buffer
            if (                  HEAD[0]==SID) {  // 0=original => (1,2) to buffer, 3 as current
               if (lid==0) {
                  sid              = HEAD[1] ;
                  BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (int)(sid % 2) ;
                  BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                  BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (int)(sid/4) ;
                  sid              = HEAD[2] ;
                  BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (int)(sid % 2) ;
                  BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                  BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (int)(sid/4) ;
               }
               SID              = HEAD[3] ;  // SID of the subray to be followed first
            } else {
               if (                  HEAD[1]==SID) {
                  if (lid==0) {
                     sid              = HEAD[2] ;
                     BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (int)(sid % 2) ;
                     BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                     BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (int)(sid/4) ;
                     sid              = HEAD[3] ;
                     BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (int)(sid % 2) ;
                     BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                     BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (int)(sid/4) ;
                  }
                  SID              = HEAD[0] ;
               } else {
                  if (                  HEAD[2]==SID) {
                     if (lid==0) {
                        sid              = HEAD[3] ;
                        BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (int)(sid % 2) ;
                        BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                        BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (int)(sid/4) ;
                        sid              = HEAD[0] ;
                        BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (int)(sid % 2) ;
                        BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                        BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (int)(sid/4) ;
                     }
                     SID              = HEAD[1] ;
                  } else {
                     if (                  HEAD[3]==SID) {
                        if (lid==0) {
                           sid              = HEAD[0] ;
                           BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (int)(sid % 2) ;
                           BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                           BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (int)(sid/4) ;
                           sid              = HEAD[1] ;
                           BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (int)(sid % 2) ;
                           BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                           BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (int)(sid/4) ;
                        }
                        SID              = HEAD[2] ;
                     } else {
                        ; // printf("???\n") ;
                     }
                  }
               }
            } 
            // for the two subrays just added, update OTLRAY, SPLIT
            sr = 3 ;   // so far the original main ray and two split subrays
            if (lid==0) {
               BUFFER[c1+5+4*1]  =  OTL ;    BUFFER[c1+5+4*2]  =  OTL ;               
               for(int i=sr; i<12; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
            } // lid==0
         } // c2>0  == we split rays on the leading edge
      } // RHO<0
      
      if (sr>0) {  // if c2>0 above...
         // We added leading edge rays, old main ray is in buffer, SID refers to one of the subrays
         // update OTI and POS to correspond to that subray
         OTI       =  (*(int *)&dx) + SID ;    // the new ray to be followed, OTI = index of first subcell + SID
         POS.x     =  fmod(POS.x,ONE) + (int)( SID%2)    ;  // dx and SID known to all work items
         POS.y     =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
         POS.z     =  fmod(POS.z,ONE) + (int)( SID/4)    ;
         OTL_RAY   =  OTL ;  // when we reach OTL<OTL_RAY, this ray will be terminated
         NBUF++ ;            // >= 1 subrays were added, NBUF increases just by one
      } // SID>=0
      
      
      
      
# if 1
      if ((OTL_RAY<OTL)&&(SIDERAYS_ADDED==0)) {  
         // Check for siderays to be added 
         // If we entered via the leading edge, the main ray was put to buffer
         // When that is again taken from the buffer, we may add siderays... and the main ray goes back to buffer
         // FIX ME: one should add siderays starting with sr=1 because the main ray is already in sr=0
         //         and now we explicitly switch main ray back to sr=0, once the siderays are first put to buffer
         if (lid==0) {
            BUFFER[c1+0] = OTL ;          BUFFER[c1+1] = I2F(OTI) ;
         }         
         c2     =  0 ; 
         if (LEADING<3) {
            if (LEADING==0) {
               if (POS.x<EPS2)      c2 = 1 ;
            } else {
               if (LEADING==1) {
                  if (POS.x>TMEPS2) c2 = 1 ;
               } else {
                  if (POS.y<EPS2)   c2 = 1 ;
               }
            }
         } else {
            if (LEADING==3) {
               if (POS.y>TMEPS2)    c2 = 1 ;
            } else {
               if (LEADING==4) {
                  if (POS.z<EPS2)   c2 = 1 ;
               } else {
                  if (POS.z>TMEPS2) c2 = 1 ;
               }
            }
         }
         if (c2==1) {  // we have the main ray hitting the octet leading edge - add siderays
            // c1 is the current buffer entry to be filled
            for(int i=lid; i<CHANNELS; i+=ls) {
               BUFFER[c1+50+i] = NTRUE[i] ;  // NTRUE is already scaled for OTL => direct copy
            }
            switch (LEADING) {               
             case 0:  // main direction X
             case 1:
               pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;         // current position in root grid coordinates
               if (LEADING==0)  pos1 = POS0 + RDIR * ((pos0.x-ZERO)/DIR.x) ; // root ray at the current plane
               else             pos1 = POS0 + RDIR * ((pos0.x-NX  )/DIR.x) ; // ... in root grid coordinates
               pos0 = (pos0-pos1)*pown(TWO, OTL) ;     // current - rootray
               // First offset in Y, second offset in Z
               pos.x  = 1.013579f ;  pos.y = 1.013579f ;  pos.z = 1.013579f ; // centre of current octet, local coordinates
               pos.y += (DIR.y>0.0f) ? -1.02f : +1.02f ;            // pos is now in the upstream neighbour
               level  = OTL ;   ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1)   ;     // extract level of the neighbour cell
               if (level<OTL) {  // neighbour not refined to same level => check subrays from that side
                  dr  = pown(TWO, OTL-level) ;    // neighbour ray step, as OTL ray steps
                  for(int a=-1; a<=3; a++) {      // ray offset to upstream direction  (1,2) or (2,3) ... test all
                     for(int b=-1; b<=3; b++) {   // orthogonal direction, (-1,0) or (0,+1)... test all
                        pos     =  POS ;                                      // our main ray position
                        pos.y  +=  (DIR.y>0.0f) ?  (-ONE*a) : (+ONE*a) ;    // jump to upstream ray
                        if ((pos.y>=ZERO)&&(pos.y<=TWO)) continue ;          // it is in current cell - skip here
                        pos.z  +=  (DIR.z>0.0f) ?  (-ONE*b) : (+ONE*b) ;
                        pos1    =  pos0 ;         // offset from rootray, as level OTL steps
                        pos1.y +=  (DIR.y>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.z +=  (DIR.z>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.y - dr*round(pos1.y/dr))<0.0004f) &&
                             (fabs(pos1.z - dr*round(pos1.z/dr))<0.0004f)) continue ; // ray exists in neighbour
                        // step the candidate ray to upstream cell side border
                        if (DIR.y>0.0f) s =  (ZERO-pos.y)/RDIR.y ;
                        else            s =  (TWO -pos.y)/RDIR.y ;
                        pos    += (s+EPS)*RDIR ;         // now possibly on the border of the current cell
                        if ((pos.x<=ZERO)||(pos.x>=TWO)||(pos.z<=ZERO)||(pos.z>=TWO)) continue ; // no hit
                        // ok, add this ray from the upstream direction to buffer as new subray
                        pos.y   =  clamp(pos.y, EPS, TMEPS) ; // make sure pos is inside the octet
                        if (lid==0) {
                           BUFFER[c1+2+4*sr] = pos.x ;
                           BUFFER[c1+3+4*sr] = pos.y ;
                           BUFFER[c1+4+4*sr] = pos.z ;
                           BUFFER[c1+5+4*sr] = OTL ;    // ray created at the current level
                        }
                        sr += 1 ;
                     }
                  }
               }
               // First offset Z, second offset Y
               pos.x  =  1.013579f ; pos.y = 1.013579f ; pos.z = 1.013579f ;    // centre of current octet
               pos.z +=  (DIR.z>0.0f) ? -1.02f : +1.02f ;         // tpos in the upstream neighbour
               level  =  OTL ; ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1) ;   // extract level of the neighbour cell
               if (level<OTL) {  // neighbour not refined to the same level => check subrays entering from that side
                  dr  =  pown(TWO, OTL-level) ;    // step between rays at neighbour discretisation level
                  for(int a=-1; a<=3; a++) {       // ray offset to upstream direction  (1,2) or (2,3) ... test all
                     for(int b=-1; b<=3; b++) {    // orthogonal direction, (-1,0) or (0,+1)... test all
                        pos     =  POS ;                                      // our main ray position
                        pos.z  += (DIR.z>0.0f) ?  (-ONE*a) : (+ONE*a) ;     // jump to upstream ray
                        if ((pos.z>=ZERO)&&(pos.z<=TWO)) continue ;         // still in current cell - skip here
                        pos.y  += (DIR.y>0.0f) ?  (-ONE*b) : (+ONE*b) ;     // step in cross direction
                        // pos1 = distance ro rootray, in OTL steps -- divisible with dr ??
                        pos1    =  pos0 ;    // current - rootray
                        pos1.z += (DIR.z>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.y += (DIR.y>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.z - dr*round(pos1.z/dr))<0.0005f) &&
                             (fabs(pos1.y - dr*round(pos1.y/dr))<0.0005f)) continue ;
                        // step the candidate ray to upstream side border of the current octet -- here along z
                        if (DIR.z>0.0f) s =  (ZERO-pos.z)/RDIR.z ;
                        else            s =  (TWO -pos.z)/RDIR.z ;
                        pos    += (s+EPS)*RDIR ;  // now possibly on the border of the current cell
                        if ((pos.x<=ZERO)||(pos.x>=TWO)||(pos.y<=ZERO)||(pos.y>=TWO)) continue ; // no hit
                        // ok, the ray from upstream direction hits this cell => add to buffer as new subray
                        pos.z   =  clamp(pos.z, EPS, TMEPS) ; // make sure pos is inside the octet
                        if (lid==0) {
                           BUFFER[c1+2+4*sr] = pos.x ;
                           BUFFER[c1+3+4*sr] = pos.y ;
                           BUFFER[c1+4+4*sr] = pos.z ;
                           BUFFER[c1+5+4*sr] = OTL ;    // ray created at the current level
                        }
                        sr += 1 ;
                     }
                  }
               }
               break ;
             case 2:  // Y = main direction
             case 3:  
               // Main offset X, secondary offset Z
               // Start by finding the offset relative to POS0... so that we can check if some subrays
               // already are coming from the neighbout
               pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;         // current root grid position
               if (LEADING==2)  pos1 = POS0 + DIR * ((pos0.y-ZERO)/DIR.y) ;  // pos0 root grid ray at the current plane
               else             pos1 = POS0 + DIR * ((pos0.y-NY  )/DIR.y) ;
               pos0   = (pos0-pos1)*pown(TWO, OTL) ;  //  current - rootray, in OTL steps, current plane, OTL units
               pos.x  = 1.013579f ; pos.y = 1.013579f ; pos.z = 1.013579f ; // centre of current octet
               pos.x += (DIR.x>0.0f) ? -1.02f : +1.02f ;          // pos is now in the upstream neighbour
               level  = OTL ;   ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1) ;   // extract level of the neighbour cell
               if (level<OTL) {  // neighbour not refined to same level => check subrays from that side
                  dr     = pown(TWO, OTL-level) ;    // step between rays at neighbour discretisation level
                  for(int a=-1; a<=3; a++) {         // main offset along      X
                     for(int b=-1; b<=3; b++) {      // secondary offset along Z
                        pos     =  POS ;                                      // our main ray position
                        pos.x  += (DIR.x>0.0f) ?  (-ONE*a) : (+ONE*a) ;     // jump to upstream ray
                        if ((pos.x>=ZERO)&&(pos.x<=TWO)) continue ;          // still in current cell - skip here
                        pos.z  += (DIR.z>0.0f) ?  (-ONE*b) : (+ONE*b) ;     // secondary offset
                        //
                        pos1    =  pos0 ;
                        pos1.x += (DIR.x>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.z += (DIR.z>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.x - dr*round(pos1.x/dr))<0.0005f) &&
                             (fabs(pos1.z - dr*round(pos1.z/dr))<0.0005f))  continue ;                        
                        if (DIR.x>0.0f) s =  (ZERO-pos.x)/RDIR.x ;
                        else            s =  (TWO -pos.x)/RDIR.x ;
                        pos   += (s+EPS)*RDIR ;  // now possibly on the border of the current cell
                        if ((pos.y<=ZERO)||(pos.y>=TWO)||(pos.z<=ZERO)||(pos.z>=TWO)) continue ; // no hit
                        // ok, the ray from upstream direction hits this cell => add to buffer as new subray
                        pos.x =  clamp(pos.x, EPS, TMEPS) ; // make sure pos is inside the octet
                        if (lid==0) {
                           BUFFER[c1+2+4*sr] = pos.x ;
                           BUFFER[c1+3+4*sr] = pos.y ;
                           BUFFER[c1+4+4*sr] = pos.z ;
                           BUFFER[c1+5+4*sr] = OTL ;   // ray created at the current level
                        }
                        sr += 1 ;
                     }
                  }
               }
               // Main offset Z, secondary X
               pos.x  = 1.013579f ; pos.y = 1.013579f ; pos.z = 1.013579f ;    // centre of current octet
               pos.z += (DIR.z>0.0f) ? -1.02f : +1.02f ;             // pos is now in the upstream neighbour
               level  = OTL ;  ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1) ;      // extract level of the neighbour cell
               if (level<OTL) {  // neighbour not refined to the same level => check subrays entering from that side
                  dr  = pown(TWO, OTL-level) ;   // step between rays at neighbour discretisation level, in root grid units
                  for(int a=-1; a<=3; a++) {     // upstream direction   =  Z
                     for(int b=-1; b<=3; b++) {  // orthogonal direction =  X
                        pos     =  POS ;                                      // our main ray position
                        pos.z  += (DIR.z>0.0f) ?  (-ONE*a) : (+ONE*a) ;     // jump to upstream ray -- in Z
                        if ((pos.z>=ZERO)&&(pos.z<=TWO)) continue ;         // still in current cell - skip here
                        pos.x  += (DIR.x>0.0f) ?  (-ONE*b) : (+ONE*b) ; 
                        // check if the ray would come from the neighbour in any case and can be skipped here
                        pos1    =  pos0 ;   // current - rootray
                        pos1.z += (DIR.z>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.x += (DIR.x>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.z - dr*round(pos1.z/dr))<0.0004f) &&
                             (fabs(pos1.x - dr*round(pos1.x/dr))<0.0004f)) continue ;
                        // step the candidate ray back to upstream cell side border -- along Z
                        if (DIR.z>0.0f) s =  (ZERO-pos.z)/RDIR.z ;
                        else            s =  (TWO -pos.z)/RDIR.z ;
                        pos   += (s+EPS)*RDIR ;  // now possibly on the border of the current cell
                        if ((pos.x<=ZERO)||(pos.x>=TWO)||(pos.y<=ZERO)||(pos.y>=TWO)) continue ; // no hit
                        // ok, the ray from upstream direction hits this cell => add to buffer as new subray
                        pos.z =  clamp(pos.z, EPS, TMEPS) ; // make sure pos is inside the octet
                        if (lid==0) {
                           BUFFER[c1+2+4*sr] = pos.x ;
                           BUFFER[c1+3+4*sr] = pos.y ;
                           BUFFER[c1+4+4*sr] = pos.z ;
                           BUFFER[c1+5+4*sr] = OTL ;   // ray created at the current level
                        }
                        sr += 1 ;
                     }
                  }
               }
               break ;               
             case 4:  // Main direction Z
             case 5:  // Main offset X, secondary offset Y
               pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;         // current main ray, root grid units
               if (LEADING==4)  pos1 = POS0 + DIR * ((pos0.z-ZERO)/DIR.z) ;  // root ray, current plane
               else             pos1 = POS0 + DIR * ((pos0.z-NZ  )/DIR.z) ;
               pos0   = (pos0-pos1)*pown(TWO, OTL) ;  //  current - rootray, in OTL steps, current plane, OTL units
               // First offset X, second offset Y
               pos.x  = 1.013579f ; pos.y = 1.013579f ; pos.z = 1.013579f ;  // centre of current octet
               pos.x += (DIR.x>0.0f) ? (-1.02f) : (+1.02f) ;          // pos is now in the upstream neighbour
               level  = OTL ;   ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1) ;    // extract level of the neighbour cell
               if (level<OTL) {                   // neighbour not refined to same level => check subrays from that side
                  dr  = pown(TWO, OTL-level) ;    // step between neighbour rays, in units of OTL step
                  for(int a=-1; a<=3; a++) {      // main offset along      X
                     for(int b=-1; b<=3; b++) {   // secondary offset along Y
                        pos     =  POS ;                                      // our main ray position
                        pos.x  += (DIR.x>0.0f) ?  (-ONE*a) : (+ONE*a) ;     // jump to upstream ray
                        if ((pos.x>=ZERO)&&(pos.x<=TWO)) continue ;          // still in current cell - skip here
                        pos.y  += (DIR.y>0.0f) ?  (-ONE*b) : (+ONE*b) ;     // secondary offset
                        //
                        pos1    =  pos0 ;         // current - rootray, in OTL steps
                        pos1.x += (DIR.x>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.y += (DIR.y>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.x - dr*round(pos1.x/dr))<0.0005f) &&
                             (fabs(pos1.y - dr*round(pos1.y/dr))<0.0005f)) continue ;
                        // step to upstream border along X direction
                        if (DIR.x>0.0f) s =  (ZERO-pos.x)/RDIR.x ;
                        else            s =  (TWO -pos.x)/RDIR.x ;
                        pos   += (s+EPS)*RDIR ;  // now possibly on the border of the current cell
                        if ((pos.y<=ZERO)||(pos.y>=TWO)||(pos.z<=ZERO)||(pos.z>=TWO)) continue ; // no hit
                        // ok, the ray from upstream direction hits this cell => add to buffer as new subray
                        pos.x =  clamp(pos.x, EPS, TMEPS) ; // make sure pos is inside the octet
                        if (lid==0) {
                           BUFFER[c1+2+4*sr] = pos.x ;
                           BUFFER[c1+3+4*sr] = pos.y ;
                           BUFFER[c1+4+4*sr] = pos.z ;
                           BUFFER[c1+5+4*sr] = OTL ;    // ray created at the current level
                        }
                        sr += 1 ;
                     }
                  }
               }
               // Main offset Y, secondary X
               pos.x  = 1.013579f ; pos.y = 1.013579f ; pos.z = 1.013579f ;  // centre of current octet
               pos.y += (DIR.y>0.0f) ? (-1.02f) : (+1.02f) ;                 // pos is now in the upstream neighbour
               level  = OTL ;   ind = OTI ;
               IndexOT(&pos, &level, &ind, RHO, OFF, PAR, 99, NULL, NULL, -1) ;   // extract level of the neighbour cell
               if (level<OTL) {  // neighbour not refined to the same level => check subrays entering from that side
                  dr  = pown(TWO, OTL-level) ;      // step between neighbour rays, in units of OTL step
                  for(int a=-1; a<=3; a++) {        // upstream direction   =  Z
                     for(int b=-1; b<=3; b++) {     // orthogonal direction =  X
                        pos     =  POS ;                                      // our main ray position
                        pos.y  += (DIR.y>0.0f) ?  (-ONE*a) : (+ONE*a) ;     // jump to upstream ray -- in Z
                        if ((pos.y>=ZERO)&&(pos.y<=TWO)) continue ;          // still in current cell - skip here
                        pos.x  += (DIR.x>0.0f) ?  (-ONE*b) : (+ONE*b) ;     // secondary offset
                        // pos1 = distance ro rootray, in OTL steps -- divisible with dr ??
                        pos1    =  pos0 ;    // current - rootray
                        pos1.y += (DIR.y>0.0f) ?  (-ONE*a) : (+ONE*a) ; // candidate - rootray
                        pos1.x += (DIR.x>0.0f) ?  (-ONE*b) : (+ONE*b) ; // is multiple of 2^(level-OTL) ??
                        if ( (fabs(pos1.y - dr*round(pos1.y/dr))<0.0005f) &&
                             (fabs(pos1.x - dr*round(pos1.x/dr))<0.0005f)) continue ;
                        // step the candidate ray back to upstream cell side border -- along Z
                        if (DIR.y>0.0f) s =  (ZERO-pos.y)/RDIR.y ;
                        else            s =  (TWO -pos.y)/RDIR.y ;
                        pos    +=  (s+EPS)*RDIR ;  // now possibly on the border of the current cell
                        if ((pos.x<=ZERO)||(pos.x>=TWO)||(pos.z<=ZERO)||(pos.z>=TWO)) continue ; // no hit
                        // ok, the ray from upstream direction hits this cell => add to buffer as new subray
                        pos.y   =  clamp(pos.y, EPS, TMEPS) ; // make sure pos is inside the octet
                        if (lid==0) {
                           BUFFER[c1+2+4*sr] = pos.x ;
                           BUFFER[c1+3+4*sr] = pos.y ;
                           BUFFER[c1+4+4*sr] = pos.z ;
                           BUFFER[c1+5+4*sr] = OTL ;   // ray created at the current level
                        }
                        sr += 1 ;
                     }
                  }
               }
               break ;               
            }  // switch  --- adding possible sidewall rays
            
            
            if (sr>0) {  // we added >=1 siderays to buffer
               if (lid==0) {
                  for(i=sr; i<12; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
               }
               // However, we want to do the siderays before the main ray
               //  => switch the main ray with the sr=0 entry in the buffer
               // AND INDICATE IN BUFFER THAT THE RAY HAS ALREADY BEEN SPLIT SO THAT WE WHEN GET IT BACK
               // FROM THE BUFFER, WE DO NOT ADD SIDERAYS AGAIN
               // NTRUE is the same for main ray and the sideray and needs no copying here
               i     = OTI ;
               // get data for one of the siderays
               barrier(CLK_GLOBAL_MEM_FENCE) ;
               pos.x = BUFFER[c1+2] ; pos.y = BUFFER[c1+3] ; pos.z = BUFFER[c1+4] ;  c2 = BUFFER[c1+5] ;
               barrier(CLK_GLOBAL_MEM_FENCE) ;
               // WE SUBTRACT 100 FROM OTL_RAY TO INDICATE THAT SIDERAYS WERE ALREADY ADDED =>
               // WHEN THIS MAIN RAY IS POPPED FROM THE BUFFER, WE DO NOT ADD SIDERAYS AGAIN
               if (lid==0) {
                  BUFFER[c1+2] = POS.x ; BUFFER[c1+3] = POS.y ; BUFFER[c1+4] = POS.z ;  BUFFER[c1+5] = OTL_RAY-100 ;
               }
               POS      =  pos ;    
               OTL_RAY  =  c2 ;
               SID      =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
               OTI      =  8*(int)(OTI/8) + SID ;  // OTI in the buffer was for the main ray
               NBUF += 1 ;
            } // if (sr>0)
            
            
         }  // c2==1  --- we are on the leading edge
      }  // OTL_RAY < OTL
# endif // adding siderays
      
      
      // === still in the while(INDEX) loop ===
      
      
      // global index -- must be now >=0 since we started with while(INDEX>=0) and just possibly went down a level
      INDEX = OFF[OTL]+OTI ;   
      
      // if not yet a leaf, jump back to start of the loop => another step down
      if (RHO[INDEX]<=0.0f) {
         continue ;
      }
      
      
      
      // we are now in a leaf node, ready to make the step
      OTLO   =  OTL ;
# if (FIX_PATHS>0)
      dx     =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, &POS_LE, LEADING) ;  // POS is now the end of this step
# else
      dx     =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL   , LEADING) ;  // POS is now the end of this step
# endif
      SIDERAYS_ADDED = 0 ;  // after the step, it is safe to check for siderays again (avoid infinite loop)
      
      
# if 0  // @@ double check PL calculation ... PL[:] should be reduced to zero
      if (lid==0)  AADD(&(PL[INDEX]), -dx) ;
# endif
      
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
# if 0
      barrier(CLK_LOCAL_MEM_FENCE) ;
      barrier(CLK_GLOBAL_MEM_FENCE) ;
      for(int i=lid; i<LOCAL; i+=LOCAL) {
         printf("[%3d]  %2d %6d   %8.4f %8.4f %8.4f   %2d\n", lid, OTL, OTI, POS.x, POS.y, POS.z, NBUF) ;
      }
# endif
      
      // with path length already being the same in all cells !   V/8,  rays*4, length/2 = 1/4
      weight    =  (dx/APL) *  VOLUME  /  pown(4.0f, OTLO) ;  // OTL=1 => dx/APL=0.5
      
      // INDEX is still the index for the cell where the step starts
      nu        =  NI[2*INDEX] ;
      nb_nb     =  NI[2*INDEX+1] ;
      
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# if (WITH_HALF==1)
      doppler  *=  0.002f ;  // half integer times 0.002f km/s
# endif                 
      // tmp_tau   =  max(1.0e-35f, dx*nb_nb*GN) ;
      tmp_tau   =  dx*nb_nb*GN ;
      if (fabs(tmp_tau)<1.0e-32f) tmp_tau = 1.0e-32f ;  // was 1e-32
      tmp_emit  =  weight * nu*Aul / tmp_tau ;  // GN include grid length [cm]
      shift     =  round(doppler/WIDTH) ;
# if (WITH_HALF==1)      //               sigma = 0.002f * w,   lookup table: sigma = SIGMA0 * SIGMAX^row
      row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
# else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
# endif
      profile   =  &GAU[row*CHANNELS] ;
      // avoid profile function outside profile channels LIM.x, LIM.y
      c1        =  max(LIM[row].x+shift, max(0, shift)) ;
      c2        =  min(LIM[row].y+shift, min(CHANNELS-1, CHANNELS-1+shift)) ;
      sum_delta_true = 0.0f ;
      all_escaped    = 0.0f ;
      
      
# if (WITH_CRT>0) // WITH_CRT
      sij = 0.0f ;
      // Dust optical depth and emission
      //   here escape = line photon exiting the cell + line photons absorbed by dust
      Ctau      =  dx     * CRT_TAU[INDEX] ;
      Cemit     =  weight * CRT_EMI[INDEX] ;      
      for(int i=c1; i<=c2; i++)  {
         pro    =  profile[i-shift] ;
         Ltau   =  tmp_tau*pro ;
         Ttau   =  Ctau + Ltau ;
         // tt     =  (1.0f-exp(-Ttau)) / Ttau ;
         tt     =  (fabs(Ttau)>0.01f) ?  ((1.0f-exp(-Ttau))/Ttau) : (1.0f-Ttau*(0.5f-0.166666667f*Ttau)) ;
         // ttt    = (1.0f-tt)/Ttau
         ttt    =  (1.0f-tt)/Ttau ;
         // Line emission leaving the cell   --- GL in profile
         Lleave =  weight*nu*Aul*pro * tt ;
         // Dust emission leaving the cell 
         Dleave =  Cemit *                     tt ;
         // SIJ updates, first incoming photons then absorbed dust emission
         sij   +=  A_b * pro*GN*dx * NTRUE[i]*tt ;
         // sij         += A_b * profile[i]*GN*Cemit*dx*(1.0f-tt)/Ttau ; // GN includes GL!
         sij   +=  A_b * pro*GN*dx * Cemit*ttt ;    // GN includes GL!
         // "Escaping" line photons = absorbed by dust or leave the cell
         all_escaped +=  Lleave  +  weight*nu*Aul*pro * Ctau * ttt ;
         // Total change of photons in the package
         NTRUE[i]     =  NTRUE[i]*exp(-Ttau) + Dleave + Lleave ;
      }  // loop over channels
      // Update SIJ and ESC (note - there may be two updates, one incoming, one outgoing ray)
      // RES[INDEX].x   += sij/VOLUME * pow(8.0f,OTL) ;  // octree 8^OTL done in the solver (host)
#  if 1
      RES[2*INDEX]    += sij ;            // division by VOLUME done in the solver (kernel)
      // Emission ~ path length dx but also weighted according to direction, works because <WEI>==1.0
      RES[2*INDEX+1]  += all_escaped ;    // divided by VOLUME only oin Solve() !!!
#  else
      AADD(&(RES[2*INDEX]), sij) ;
      AADD(&(RES[2*INDEX+1]), all_escaped) ;
#  endif
      
      
# else   // not  WITH_CRT ***************************************************************************************
      
      
      // because of c1, the same NTRUE elements may be updated each time by different work items...
      barrier(CLK_LOCAL_MEM_FENCE) ;    // local NTRUE elements possibly updated by different threads
      
      
#  if (WITH_ALI>0) // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      for(int i=c1+lid; i<=c2; i+=ls)  {
         w               =  tmp_tau*profile[i-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;   // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      SDT[lid] = sum_delta_true ;  AE[lid] = all_escaped ; // all work items save their own results
      barrier(CLK_LOCAL_MEM_FENCE) ;             // all agree on NTRUE, all know SDT and AE
      if (lid==0) {                              // lid=0 sums up and saves absorptions and escaped photons
         for(int i=1; i<LOCAL; i++) {  
            sum_delta_true += SDT[i] ;      all_escaped    +=  AE[i] ;     
         }
         RES[2*INDEX]   +=  A_b * (sum_delta_true / nb_nb) ;
         RES[2*INDEX+1] +=  all_escaped ;         
         // AADD(&(RES[2*INDEX]),    A_b * sum_delta_true / nb_nb) ;
         // AADD(&(RES[2*INDEX+1]),  all_escaped) ; 
         // AADD((__global float*)(RES+2*INDEX  ),  A_b * sum_delta_true / nb_nb) ;
         // AADD((__global float*)(RES+2*INDEX+1),  all_escaped) ;  // THIS DOES NOT WORK ???
         // if ((gid==0)&&(INDEX%100==33)) printf("all_escaped %12.4e\n", all_escaped) ;
      } // lid==0
#  else // else no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      for(int i=c1+lid; i<=c2; i+=ls)  {
         w               =  tmp_tau*profile[i-shift] ;
         // factor          =  1.0f-exp(-1.0*w) ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         // factor          =  (fabs(w)>0.005f) ? (1.0f-exp(-w)) : (w*(1.0f-w*0.5f)) ;
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;    // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed - escape ;  // later absorbed ~ W*nu*Aul - escape
      }   // over channels
      SDT[lid] = sum_delta_true ;
      barrier(CLK_LOCAL_MEM_FENCE) ;
      if (lid==0) {
         for(int i=1; i<LOCAL; i++) sum_delta_true += SDT[i] ;    
         w  =   A_b  * ((weight*nu*Aul  + sum_delta_true) / nb_nb)  ;
         RES[INDEX] += w ;
         // atomic has no significant overhead
         // AADD(&(RES[INDEX]),   w) ;   // Sij counter update
      } 
#  endif // no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
      
      
# endif  // WITH OR WITHOUT CRT
      
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
# if (BRUTE_COOLING>0)
      // total number of photons in the package as it exits the cell
      if (lid==0) {
         float cool = 0.0f ;
         for(int i=0; i<CHANNELS; i++) cool += NTRUE[i] ;
         COOL[INDEX] += cool ; // cooling of cell INDEX --- each work group distinct rays => no need for atomics
      }
# endif
      
      
      
      
      // Updates at the end of the step, POS has been already updated, OTL and OTI point to the new cell
      INDEX   =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;
      
# if (BRUTE_COOLING>0)  // heating of the next cell, once INDEX has been updated
      if (INDEX>=0) {
         if (lid==0)  COOL[INDEX] -= cool ; // heating of the next cell
      }
# endif
      
      if (INDEX>=0) {
         if (RHO[INDEX]<=0.0f) { // we stepped to a refined cell (GetStep goes only to a cell OTL<=OTLO)
            continue ;           // step down one level at the beginning of the main loop
         }
      }
      
      
      if (OTL<OTL_RAY) {   // we are up to a level where this ray no longer exists
         INDEX=-1 ;        
      } else {      
         if (INDEX<0) {    // ray exits the cloud... possibly continues on the other side
            if (POS.x>=NX  ) {   if (LEADING!=0)  POS.x =    EPS ;   }
            if (POS.x<=ZERO) {   if (LEADING!=1)  POS.x = NX-EPS ;   } 
            if (POS.y>=NY  ) {   if (LEADING!=2)  POS.y =    EPS ;   }
            if (POS.y<=ZERO) {   if (LEADING!=3)  POS.y = NY-EPS ;   } 
            if (POS.z>=NZ  ) {   if (LEADING!=4)  POS.z =    EPS ;   }
            if (POS.z<=ZERO) {   if (LEADING!=5)  POS.z = NZ-EPS ;   } 
            IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;  // we remain in a root-grid cell => OTL==0 !
            INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
            if (INDEX>=0) {   // new level-0 ray started on the opposite side (may be a parent cell)
               OTL_RAY = 0 ;  OTLO = 0 ; 
               // we had already barrier after the previous NTRUE update
               for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BG * DIRWEI ;
               barrier(CLK_LOCAL_MEM_FENCE) ;
# if (BRUTE_COOLING>0)
               if (lid==0) {
                  dx = BG*DIRWEI*CHANNELS ;  COOL[INDEX] -= dx ; // heating of the entered cell
               }
# endif
               continue ;
            }
         } // if INDEX<0
      }
      
      
      // @@  rescale on every change of resolution
      if ((INDEX>=0)&&(OTL<OTLO)) {  // @s ray continues at a lower hierarchy level => NTRUE may have to be scaled
         dx = pown(4.0f, OTLO-OTL) ;  // scale on every change of resolution
         for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] *= dx ;     
         continue ;  // we went to lower level => this cell is a leaf
      }
      
      
      // if INDEX still negative, try to take a new ray from the buffer
      // 0   1    2  3  4    5    6   7         8    9    10       11   12  13       14   
      // OTL OTI  x  y  z    SID  RAY SPLIT     SID  RAY  SPLIT    SID  RAY SPLIT    NTRUE
      if ((INDEX<0)&&(NBUF>0)) {            // NBUF>0 => at least one ray exists in the buffer
         barrier(CLK_GLOBAL_MEM_FENCE) ;    // all work items access BUFFER
         c1    =  (NBUF-1)*(50+CHANNELS) ;  // 8+CHANNELS elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // OTL ...
         OTLO  =  OTL ;                     // ???
         OTI   =  F2I(BUFFER[c1+1]) ;       // and OTI of the ray that was split
         for(sr=11; sr>=0; sr--) {         
            dr    =  BUFFER[c1+2+4*sr] ;
            if (dr>-0.1f) break ;           // found a ray
         }
         POS.x   =  BUFFER[c1+2+4*sr] ;  
         POS.y   =  BUFFER[c1+3+4*sr] ;  
         POS.z   =  BUFFER[c1+4+4*sr] ;
         OTL_RAY =  BUFFER[c1+5+4*sr] ;
         SIDERAYS_ADDED = 0 ;
         if (OTL_RAY<0) { // <0 values meant that main ray had already added siderays when it was put to buffer
            OTL_RAY = OTL_RAY+100 ;   SIDERAYS_ADDED = 1 ;
         }         
         if (sr==0)   NBUF -= 1 ;
         SID     =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
         OTI     =  8*(int)(OTI/8) + SID ;        // OTI in the buffer was for the original ray that was split
         INDEX   =  OFF[OTL]+OTI ;                // global index -- must be >=0 !!!
         // copy NTRUE --- values in BUFFER have already been divided by four
         barrier(CLK_GLOBAL_MEM_FENCE) ;          // all have read BUFFER
         if (lid==0)  BUFFER[c1+2+4*sr] = -1.0f ; // mark as used
         c1     +=  50 ;
         for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BUFFER[c1+i] ;
         barrier(CLK_LOCAL_MEM_FENCE) ;         
         // note - this ray be inside a parent cell => handled at the beginnign of the main loop
      } // (INDEX<=0)&(NBUF>0)
      
      
   } // while INDEX>=0
   
}  // end of UpdateOT3()


#endif // OCTREE==3















#if (WITH_OCTREE==40)   // #@

// 40 == one work item per ray


# if (DEBUG>0)
void report(const int       OTL, 
            const int       OTI,
            const int       OTL_RAY, 
            REAL3          *POS,
            __constant int *OFF, 
            __global int   *PAR) {
   REAL3  pos = *POS ;
   RootPos(&pos, OTL, OTI, OFF, PAR) ;   
   printf(" (%2d,%2d) %4d    L: %6.2f %6.2f %6.2f      R: %6.2f %6.2f %6.2f      OTL=%d OTL_RAY=%d\n",
          OTL, OTI, OFF[OTL]+OTI, POS->x, POS->y, POS->z, pos.x, pos.y, pos.z, OTL, OTL_RAY) ;  
}
# endif




__kernel void PathsOT40(  // @p
                          const    int      id0,     // do in batches, now starting with id==id0
                          __global float   *PL,      // [CELLS]
                          __global float   *TPL,     // TPL[NRAY]
                          __global int     *COUNT,   // COUNT[NRAY] total number of rays entering the cloud
                          const    int      LEADING, // leading edge
                          const    REAL3    POS0,    // initial position of ray 0
                          const    float3   DIR,     // direction of the rays
                          __global   int   *LCELLS,   
                          __constant int   *OFF,
                          __global   int   *PAR,
                          __global float   *RHO,
                          __global float   *BUFFER_ALL
                       ) {   
   // OT40 is the same as OT4 but each ray is processed with a separate work item, not a work group
   // With ONESHOT, kernel will be called only with a single offset POS0 and all the rays 
   // will be covered in a single kernel call
   
   // This version tries to make sure we make all the rays so that there is no variation in the
   // path lengths between the cells (avoids the need for PL weighting!!)
   // When entering a refined cell with level0, split the ray to four on the leading edge, like in PathsOT2
   // Regarding additional rays hitting the sides of the refined cell, check the two upstream neighbours.
   // If the neighbour is:
   //   - at level >= level0, do nothing -- refined rays will come from the neighbour
   //   - otherwise, test 4 refined rays that might hit the current refined cell from that side
   //     skip the rays that will come otherwise at some point from the neighbour
   int  id = get_global_id(0) ;  // actually id of a single work item
   // things indexed with the current real group id
   __global float *BUFFER = &(BUFFER_ALL[id*(26+CHANNELS)*MAX_NBUF]) ; // allocation for each work item
   id += id0 ;                // running of the ray, id=0, ..., NRAY-1,  TPL[NRAY]
   if  (id>=NRAY) return ;    // no more rays
   __global int *count  = &COUNT[id] ;     // COUNT[NRAY]
   *count = 0 ;
   int  INDEX, SID, NBUF=0, sr, sid, level, b1, XL, ind ;
   int  OTL, OTI, OTLO, RL, c1, c2, i, SIDERAYS_ADDED=0, oti, otl, otl_ray, ae, be, level1, ind1 ;
   float  dr, flo, tpl=0.0f ;
   REAL  dx ;
   REAL3 POS, pos, pos0, pos1, RDIR ;
   POS.x  = POS0.x ; POS.y  = POS0.y ; POS.z  = POS0.z ;
   RDIR.x = DIR.x ;  RDIR.y = DIR.y ;  RDIR.z = DIR.z ;
# if (ONESHOT<1)
   int nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ;
# endif
   bool drop ;
   
   // HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
# if (ONESHOT<1)
   if (LEADING<2){ 
      if (LEADING==0) {
         HEAD[0] = 0 ;          HEAD[1] = 2 ;            HEAD[2] = 4 ;       HEAD[3] = 6 ;
         POS.x   = EPS ;        POS.y  += TWO*(id%ny) ;  POS.z  += TWO*(int)(id/ny) ;
      } else {
         HEAD[0] = 1 ;          HEAD[1] = 3 ;            HEAD[2] = 5 ;       HEAD[3] = 7 ;
         POS.x   = NX-EPS ;     POS.y  += TWO*(id%ny) ;  POS.z  += TWO*(int)(id/ny) ;
      }
   } else {
      if (LEADING<4) {
         if (LEADING==2) { 
            HEAD[0] = 0 ;       HEAD[1] = 1 ;            HEAD[2] = 4 ;       HEAD[3] = 5 ;
            POS.y   = EPS ;     POS.x  += TWO*(id%nx) ;  POS.z  += TWO*(int)(id/nx) ; 
         } else {
            HEAD[0] = 2 ;       HEAD[1] = 3 ;            HEAD[2] = 6 ;       HEAD[3] = 7 ;
            POS.y   = NY-EPS ;  POS.x  += TWO*(id%nx) ;  POS.z  += TWO*(int)(id/nx) ;
         }
      } else {      
         if (LEADING==4) { 
            HEAD[0] = 0 ;       HEAD[1] = 1 ;            HEAD[2] = 2 ;        HEAD[3] = 3 ;
            POS.z   = EPS ;     POS.x  += TWO*(id%nx) ;  POS.y  += TWO*(int)(id/nx) ; 
         } else {
            HEAD[0] = 4 ;       HEAD[1] = 5 ;            HEAD[2] = 6 ;        HEAD[3] = 7 ;
            POS.z   = NZ-EPS ;  POS.x += TWO*(id%nx) ;   POS.y  += TWO*(int)(id/nx) ;
         }
      }
   }
# else
   if (LEADING<2){ 
      if (LEADING==0) {
         HEAD[0]    = 0 ;        HEAD[1] = 2 ;            HEAD[2] = 4 ;       HEAD[3] = 6 ;
         POS.x      = EPS ;      POS.y  += id%NY ;        POS.z  += id/NY ;
      } else {
         HEAD[0]    = 1 ;        HEAD[1] = 3 ;            HEAD[2] = 5 ;       HEAD[3] = 7 ;
         POS.x      = NX-EPS ;   POS.y  += id%NY ;        POS.z  += id/NY ;
      }
   } else {
      if (LEADING<4) {
         if (LEADING==2) { 
            HEAD[0] = 0 ;        HEAD[1] = 1 ;            HEAD[2] = 4 ;       HEAD[3] = 5 ;
            POS.y   = EPS ;      POS.x  += id%NX ;        POS.z  += id/NX ;
         } else { 
            HEAD[0] = 2 ;        HEAD[1] = 3 ;            HEAD[2] = 6 ;       HEAD[3] = 7 ;
            POS.y   = NY-EPS ;   POS.x  += id%NX ;        POS.z  += id/NX ;
         }
      } else {      
         if (LEADING==4) { 
            HEAD[0] = 0 ;        HEAD[1] = 1 ;            HEAD[2] = 2 ;        HEAD[3] = 3 ;
            POS.z   = EPS ;      POS.x  += id%NX ;        POS.y  += id/NX ;
         } else {
            HEAD[0] = 4 ;        HEAD[1] = 5 ;            HEAD[2] = 6 ;        HEAD[3] = 7 ;
            POS.z   = NZ-EPS ;   POS.x  += id%NX ;         POS.y += id/NX ;
         }
      }
   }
# endif
   
   
   int *SUBS ;
   
   
   // IndexGR takes [0,NX] coordinates and just returns the root-grid index
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;   // remain at root level, not yet going to leaf -- this just calculates index
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;
   OTLO      =  OTL ;
   RL        =  0 ;    // level of the ray (0 for the root-grid rays)
   if (INDEX>=0)  *count += 1 ;  // number of incoming rays
   
# if (DEBUG>0)
   printf("CREATE    ") ;
   report(OTL, OTI, RL, &POS, OFF, PAR) ; 
# endif   
   
   
   int steps = 0 ;
   
   
   while(INDEX>=0) {  // INDEX may refer to a cell that is not a leaf
      
      // ===== ADD LEADING EDGE RAYS WHEN ENTERING REFINED REGION THROUGH UPSTREAM CELL BORDER =====
      // This guarantees that the leading edge always has one ray per cell (4 per octet).
      // These are added whenever we step into a refined cell. The other needed rays are added as
      // "siderays" above. Also that relies on the fact that for each octet at level OTL, among the
      // rays that hit its leading edge, there is exactly one ray with RL < OTL (as well as three 
      // with RL==OTL).
      // If we step over several levels, we must again worry about the precision of POS
      // After each normal step we are EPS from a cell boundary.... but we might step
      // here several layers steeper!!
      if (RHO[INDEX]<=0.0f) {                    // go to the sub-cell and add sibling rays
         SID    =  -1 ;
         NBUF   =  min(NBUF, MAX_NBUF-1) ;       // IF BUFFER IS ALREADY FULL !!
         c1     =  NBUF*(26+CHANNELS) ;          // 62+CHANNELS elements per buffer entry
         sr     =  0  ;
         // If we are not in a leaf, we have to go one level up,
         // add >=three rays, pick one of them as the current, return to the beginning of the loop.
         POS.x  =  TWO*fmod(POS.x, ONE) ;      // coordinates inside the single child cell [0,1]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         flo    =  -RHO[INDEX] ;                 // OTI for the first subcell in octet
         OTL   +=  1  ;                          // step to the next refinement level
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // SID for subcell with the ray
         OTI    =  *(int *)&flo + SID ;          // cell of the incoming ray
         
         c2     =  0 ;                           // set c2=1 if ray is to be split: is this the leading edge?
         if (LEADING<3) {
            if ((LEADING==0)&&(POS.x<  EPS2)) c2 = 1 ; 
            if ((LEADING==1)&&(POS.x>TMEPS2)) c2 = 1 ; 
            if ((LEADING==2)&&(POS.y<  EPS2)) c2 = 1 ;
         } else {
            if ((LEADING==3)&&(POS.y>TMEPS2)) c2 = 1 ; 
            if ((LEADING==4)&&(POS.z<  EPS2)) c2 = 1 ;
            if ((LEADING==5)&&(POS.z>TMEPS2)) c2 = 1 ;
         }
         if (c2>0) {  // split the ray and choose one of the new ones to be followed first
            BUFFER[c1+0]  =  OTL ;               // level where the split is done
            BUFFER[c1+1]  =  I2F(OTI) ;          // buffer contains OTI of the original ray
            // Put the main ray as the first, sr=0 entry in buffer
            BUFFER[c1+2]  =  POS.x ; 
            BUFFER[c1+3]  =  POS.y ;
            BUFFER[c1+4]  =  POS.z ;
            BUFFER[c1+5]  =  RL    ;
            sr            =  1 ;
            // Add two more subrays to buffer, the third one is taken as the current ray
            // HEAD is the four cells on the upstream border
            if (                     HEAD[0] ==  SID) {  // 0=original => (1,2) to buffer, 3 as current
               sid                            =  HEAD[1] ;
               BUFFER[c1+2+4*sr]              =  fmod(POS.x,ONE) + (int)( sid % 2) ;
               BUFFER[c1+3+4*sr]              =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
               BUFFER[c1+4+4*sr]              =  fmod(POS.z,ONE) + (int)( sid/4) ;
               sid                            =  HEAD[2] ;
               BUFFER[c1+2+4*(sr+1)]          =  fmod(POS.x,ONE) + (int)( sid % 2) ;
               BUFFER[c1+3+4*(sr+1)]          =  fmod(POS.y,ONE) + (int)((sid/2)%2)  ;
               BUFFER[c1+4+4*(sr+1)]          =  fmod(POS.z,ONE) + (int)( sid/4) ;
               SID                            =  HEAD[3] ;  // SID = the subray that becomes the current ray
            } else {
               if (                  HEAD[1] ==  SID) {
                  sid                         =  HEAD[2] ;
                  BUFFER[c1+2+4*sr]           =  fmod(POS.x,ONE) + (int)( sid   %2) ;
                  BUFFER[c1+3+4*sr]           =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                  BUFFER[c1+4+4*sr]           =  fmod(POS.z,ONE) + (int)( sid/4) ;
                  sid                         =  HEAD[3] ;
                  BUFFER[c1+2+4*(sr+1)]       =  fmod(POS.x,ONE) + (int)( sid   %2) ;
                  BUFFER[c1+3+4*(sr+1)]       =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                  BUFFER[c1+4+4*(sr+1)]       =  fmod(POS.z,ONE) + (int)( sid/4) ;
                  SID                         =  HEAD[0] ;
               } else {
                  if (               HEAD[2] ==  SID) {
                     sid                      =  HEAD[3] ;
                     BUFFER[c1+2+4*sr]        =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                     BUFFER[c1+3+4*sr]        =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                     BUFFER[c1+4+4*sr]        =  fmod(POS.z,ONE) + (int)( sid/4) ;
                     sid                      =  HEAD[0] ;
                     BUFFER[c1+2+4*(sr+1)]    =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                     BUFFER[c1+3+4*(sr+1)]    =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                     BUFFER[c1+4+4*(sr+1)]    =  fmod(POS.z,ONE) + (int)( sid/4) ;
                     SID                      =  HEAD[1] ;
                  } else {
                     if (            HEAD[3] ==  SID) {
                        sid                   =  HEAD[0] ;
                        BUFFER[c1+2+4*sr]     =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                        BUFFER[c1+3+4*sr]     =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                        BUFFER[c1+4+4*sr]     =  fmod(POS.z,ONE) + (int)( sid/4) ;
                        sid                   =  HEAD[1] ;
                        BUFFER[c1+2+4*(sr+1)] =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                        BUFFER[c1+3+4*(sr+1)] =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                        BUFFER[c1+4+4*(sr+1)] =  fmod(POS.z,ONE) + (int)( sid/4) ;
                        SID                   =  HEAD[2] ;
                     } else {
                        ; // printf("??????????????? LEADING %d, SID %d\n", LEADING, SID) ;
                     }
                  }
               }
            }
            // two subrays added to buffer, update their RL
            // remember that when refinement jumps to higher value, we come here for every step in refinement
            // before stepping forward
            BUFFER[c1+5+4*sr]  =  OTL ;       BUFFER[c1+5+4*(sr+1)]  =  OTL ;   // RL == current OTL
            sr += 2 ;   // main ray and  two subrays added, the third subray==SID will become the current one
            for(int i=sr; i<6; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused, max 6 subrays
            NBUF += 1 ;
            
            // if (NBUF>=1000) printf("!!!!!!!!!!  NBUF = %d\n", NBUF) ;
            
            
# if (DEBUG>0)
            printf("------------------------------------------------------------------------------------\n") ;
            for(int i=0; i<sr; i++) {
               printf("ADD.LEAD%d ", i) ;
               otl       =  BUFFER[c1+0] ;
               oti       =  F2I(BUFFER[c1+1]) ;
               pos.x     =  BUFFER[c1+2+4*i] ; pos.y =  BUFFER[c1+3+4*i] ; pos.z =  BUFFER[c1+4+4*i] ;
               sid       =  4*(int)floor(pos.z) + 2*(int)floor(pos.y) + (int)floor(pos.x) ; 
               oti       =  8*(int)(OTI/8) + SID ;
               otl_ray   =  BUFFER[c1+5+4*i] ;            
               report(otl, oti, otl_ray, &pos, OFF, PAR) ; 
            }
            printf("------------------------------------------------------------------------------------\n") ;
# endif         
            
            // SID = is the third new leading-edge subray =>  switch to that
            // OTL was already increase by one = refinement
            OTI       =  (*(int *)&flo) + SID ;    // the new ray to be followed, OTI = index of first subcell
            POS.x     =  fmod(POS.x,ONE) + (int)( SID%2)    ;
            POS.y     =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
            POS.z     =  fmod(POS.z,ONE) + (int)( SID/4)    ;
            RL        =  OTL ;      // leading edge rays created at level OTL
            
            
         }  //  c2>0  == other leading-edge subrays added
      } // RHO < 0.0 == we entered refined region
      
      
      
      
      INDEX = OFF[OTL]+OTI ;   // global index -- must be now >=0 since we went down a level
      
      
      // if not yet a leaf, jump back to start of the loop, for another step down
      if (RHO[INDEX]<=0.0f) {
         // printf("RHO<0\n") ;
         steps += 1  ;   
         // if (steps>100000) printf("RHO[%d]<=0.0  --- OTL %d, OTI %d\n", INDEX, OTL, OTI) ;
         continue ;
      }
      
      
      
# if 1
      // RL  = ray level, the level at which the ray was created, root rays are RL=0
      // OTL = cell level, the current discretisation level, root grid is OTL=0
      // =>  Check at each step if there are potential siderays to be added.
      // RL rays provide siderays at levels XL,    RL <  XL <= OTL
      //    (A) it is boundary between level XL octets
      //    (B) upstream-side neighbour has OTL<XL -- so that it will not provide the XL rays directly
      //    (C) the XL sideray is not also XL-1 ray --  those XL-1 siderays are provided by XL-2 rays
      //        since current ray is XL ray, skip even offsets where the step=1.0 in level XL coordinates
      //    (D) the XL sideray actually hits the side of the current octet with level XL cells
      
      pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;   // current position in root coordinates
      
      
      // Current ray is still (OTL, OTI, POS)
      for(XL=OTL; XL>RL; XL--) {   // try to add subrays at level XL -- never on level=0
         
#  if 1
         if (NBUF>=MAX_NBUF) {
            // printf("*** BUFFER FULL -- CANNOT CHECK SIDERAYS\n") ; 
            break ;
         }
#  endif
         
         c1     =  NBUF*(26+CHANNELS) ;          // 26+CHANNELS elements per buffer entry
         sr     =  0  ;
         
         // printf("!!!  RL=%d  <  XL=%d  <= OTL=%d\n", RL, XL, OTL) ;
         
         
         if (((LEADING==0)&&(POS.x<EPS2))||((LEADING==1)&&(POS.x>TMEPS2))) {  // +/-X leading edge, at OTL level
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.x/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test smaller XL (larger dx) values
            // even number of dx,is therefore a border between level XL octets
            // calculate (pos, level, ind) for the position at level XL  (in the octet that we are in)
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            // Note IndexUP used only at levels>0 => independent of the handling of the root-grid coordinates
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // (pos, level, ind) now define the position at level XL
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL ===== offsets                              Y and Z
            // check XL-scale neighbour
            pos1.x =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       Y * 
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            // It is called only at levels level>0, not affected by the handling of root-grid coordinates
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               // Current ray position at level==XL is defined by (level, ind, pos).
               // We loop over XL positions on the leading-edge plane, ignore those not common with
               // XL-1 rays, and choose those that actually hit the -Y side of the current octet.
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.y += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =       Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;    // still inside the current octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     // pos1 = initial coordinates at the leading-edge level, step forward to the Y sidewall
                     if (DIR.y>0.0f)   pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ;  // step based on    Y ****
                     else              pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;   // to 2.0  (level==RL+1)
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.y  =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in   octet        Y *****
#  endif
                     // Add the ray from the upstream direction to buffer as a new subray
                     //    we leave the buffer entry at level XL, if leafs are >XL, we drop to next 
                     //    refinement level only at the beginning of the main loop.
                     // Each sideray will actually continue in location refined to some level >=XL
                     // We will store NTRUE correct for XL. If sideray is in refined region, beginning of the
                     // main loop will drop the ray to the higher level and rescale NTRUE accordingly.
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level==XL
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL ===== offsets                              Z and Y
            // current ray at level level==RL+1 defined by (level, ind, pos)
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {     // neighbour not refined to level==XL, will not provide XL siderays
               for(int a=-1; a<=3; a++) {       // offset in +/- Z direction from the level RL ray, in XL steps
                  for(int b=-1; b<=3; b++) {    // ray offset in +/- Y direction, candidate relative to current ray
                     if ((a%2==0)&&(b%2==0)) continue ; // skip LR rays
                     // if we come this far, we will add the ray if it just hits the current octet with RL+1 cells
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =      Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // in current octet, will not hit sidewall
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // second offset           Y ***
                     // pos1 = initial coordinates at the leading edge plane, step to the Z sidewall
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in octet,        Z *****
#  endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL     ;      // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (((LEADING==2)&&(POS.y<EPS2))||((LEADING==3)&&(POS.y>TMEPS2))) {  // +/-Y leading edge
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.y/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // ***A*** even number of dx,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL) IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITTING THE X SIDEWALL ===== offsets                             X and Z
            pos1.y =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.x =  (DIR.x>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       X *
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos inthe octet,      X *****
#  endif
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL =====  offsets                             Z and X
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =     Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =     X ***
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in the octet,    Z *****
#  endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         
         if (((LEADING==4)&&(POS.z<EPS2))||((LEADING==5)&&(POS.z>TMEPS2))) {  // +/-Z leading edge
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.z/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // even number of dx,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITING THE X SIDEWALL ===== offsets                              X and Y
            pos1.y =  0.1f ;     pos1.z = 0.1f ;
            pos1.x =  (DIR.x>0.0f) ? (-0.1f) : (2.1f) ;    // upstream neighbour, main offset       X *
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.y += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Y ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos in the octet,     X *****
#  endif
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     sr += 1 ;                     
                     
#  if (DEBUG>0)
                     printf("!!!A  ") ;
                     report(level, ind, XL, &pos, OFF, PAR) ;
#  endif      
                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL =====  offsets                            Y and X
            pos1.x =  0.1f ;    pos1.z = 0.1f ;
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset       Y *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL 
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =    Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =    X ***
                     if (DIR.y>0.0f) pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ; // step based on      Y ****
                     else            pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.y   =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in the octet,   Y *****
#  endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         
         
         
         if (sr>0) {  // added some siderays
            // While the current ray was at level OTL, there are no guarantees that the same
            // refinement exists for all added siderays. Therefore, we add all siderays
            // using the level XL coordinates. This also means that the siderays and the
            // leading-edge subrays (below) must be stored as separate BUFFER entries (different NBUF).
            BUFFER[c1+0] = XL ;          // at level XL all rays stored as level XL rays
            BUFFER[c1+1] = I2F(ind) ;    // index of *original ray*, at level XL
            // We leave the original ray as the current one == (OTL, OTI, POS), these unchanged.
            
#  if (DEBUG>0)
            printf("!!! id=%d ADDED =========== (%d, %d) %d\n", id, XL, ind, OFF[XL]+ind) ;
            printf("================================================================================\n") ;
            for(int i=0; i<sr; i++) {   // skip sr=0 that is so far an empty slot
               otl       =  BUFFER[c1+0] ;
               oti       =  F2I(BUFFER[c1+1]) ;   // cell index for  the icoming ray
               pos.x     =  BUFFER[c1+2+4*i] ; pos.y =  BUFFER[c1+3+4*i] ; pos.z =  BUFFER[c1+4+4*i] ;
               sid       =  4*(int)floor(pos.z) + 2*(int)floor(pos.y) + (int)floor(pos.x) ; 
               oti       =  8*(int)(OTI/8) + sid ;   // cell index for the added ray
               otl_ray   =  BUFFER[c1+5+4*i] ;            
               printf("ADD.SIDE%d   main ray (%d,%d) %d   L: %8.4f %8.4f %8.4f --- NBUF =%d\n",
                      i, otl, oti, OFF[otl]+oti, pos.x, pos.y, pos.z, NBUF) ;
               // splitpos  =  BUFFER[c1+6+4*i] ;
               printf("ADD.SIDE%d ", i) ;
               report(otl, oti, otl_ray, &pos, OFF, PAR) ;
               
            }
            printf("================================================================================\n") ;
#  endif
            
            for(int i=sr; i<6; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
            NBUF += 1 ;
            // if (NBUF>=1000) printf("!!!!!!!!!!  NBUF = %d\n", NBUF) ;
            
            // When NTRUE is saved to buffer, the values must be stored rescaled to the level XL.
            // Thus, the stored values are   NTRUE(for_OTL) * 4.0**(OTL-XL)
            // When sideray is taken from buffer and is located in a region with OTL>XL, the
            // beginning of the main loop will again rescale with  4.0*(XL-OTL)
         }
         
      } // for XL -- adding possible siderays
      
      
      // Subrays are added only when we are at leaf level. Also, when subrays are added, 
      // we continue with the original RL ray and immediately make a step forward. Therefore
      // there is no risk that we would add siderays multiple times.
# endif
      
      
      
      
# if (DEBUG>1)      
      printf("CONTINUE  ") ;
      report(OTL, OTI, RL, &POS, OFF, PAR) ; 
# endif      
      
      
      if (NBUF>=MAX_NBUF) {
         TPL[id] = -1.0e20f ; return ;
      }
      
      // we come here only if we have a ray that is inside a leaf node => can do the step forward
      // if the step is to refined cell, (OTL, OTI) will refer to a cell that is not a leaf node
      // ==> the beginning of the present loop will have to deal with the step to higher refinement
      // level and the creation of three additional rays
      OTLO    =  OTL ;  c2 = OTI ;
      // get step but do not yet go to level > OTLO
          
# if (DEBUG>0)
      pos1 = POS ;
      RootPos(&pos1, OTL, OTI, OFF, PAR) ;      
      c2 = OFF[OTL]+OTI ;
      printf("[%d] @ STEP    %9.6f %9.6f %9.6f   %d  ", id, pos1.x, pos1.y, pos1.z, OTLO) ;      
# endif
      
      
      
      dr      =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL   , LEADING) ; // step [GL] == root grid units !!
      
      
      SIDERAYS_ADDED = 0 ;  // whenever we have made one step, it is again safe to test if main ray has siderays
      
      
# if (NO_ATOMICS>0)
      PL[INDEX] += dr ;
# else
      AADD(&(PL[INDEX]), dr) ;
#endif      
      
      
      tpl       += dr ;               // just the total value for current idir, ioff
      
      // The new index at the end of the step  --- OTL >= OTL  => INDEX may not refer to a leaf cell
      INDEX    =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;  // OTL, OTI, POS already updated = end of the current step
      
      
      if (INDEX>=0) {
         if (RHO[INDEX]<=ZERO) {  // we ended up in a parent cell
            steps += 1 ;
# if (SAFE>0)
            if (steps>100000) printf("LINE 4528 !!!!!!!!!!!!\n") ;
# endif
            continue ; // we moved to refined region, handled at the beginning of the main loop
         }
      }
      
      
      if (OTL<RL) {           // up in hierarchy and this ray is terminated -- cannot be on root grid
         INDEX=-1 ;           // triggers the generation of a new ray below
      } else {     
         if (INDEX<0) {       // ray exits the root grid, possibly create a new OTL=0 ray on the other side
            if (POS.x>=NX  ) {   if (LEADING!=0)   POS.x =    EPS ;   }
            if (POS.x<=ZERO) {   if (LEADING!=1)   POS.x = NX-EPS ;   }
            if (POS.y>=NY  ) {   if (LEADING!=2)   POS.y =    EPS ;   }
            if (POS.y<=ZERO) {   if (LEADING!=3)   POS.y = NY-EPS ;   } 
            if (POS.z>=NZ  ) {   if (LEADING!=4)   POS.z =    EPS ;   }            
            if (POS.z<=ZERO) {   if (LEADING!=5)   POS.z = NZ-EPS ;   }
            IndexGR(&POS, &OTL, &OTI, RHO, OFF) ; // not necessarily a leaf!
            INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
            if (INDEX>=0) {    // new level-0 ray started on the opposite side
               RL = 0 ;    OTLO    = 0 ;   *count += 1 ;
# if (DEBUG>0)
               printf("MIRROR    ") ;
               report(OTL, OTI, RL, &POS, OFF, PAR) ;
# endif
               continue ;
            }
         } // if INDEX<0
      }
      
      
      
      // [C] if INDEX still negative, current ray truly ended => take new ray from the buffer, if such exist
      //  NBUF>0 is no guarantee that rays exist in the buffer because we reserved sr=0 for the main ray
      //  and if only siderays were added, this NBUF entry has nothing for sr=0
      if ((INDEX<0)&&(NBUF>0))   {                // find a ray from BUFFER
         c1    =  (NBUF-1)*(26+CHANNELS) ;  // 26+CHANNELS elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // level where sibling rays were created
         OTLO  =  OTL ;
         OTI   =  F2I(BUFFER[c1+1]) ;       // OTI for the original ray that was split
         // a maximum of 6 rays per buffer entry, the entry sr=0 was reserved for the main ray
         for(sr=5; sr>=0; sr--) {         
            dx    =  BUFFER[c1+2+4*sr] ;
            if (dx>-0.1f) break ;      // found a ray
         }
# if (SAFE>0)
         if (dx<ZERO) printf("??? BUFFER WITH NEGATIVE POS.x ???\n") ;
# endif
         POS.x     =  dx ;
         POS.y     =  BUFFER[c1+3+4*sr] ;
         POS.z     =  BUFFER[c1+4+4*sr] ;
         RL        =  BUFFER[c1+5+4*sr] ;    
         // we must have found a ray  -- figure out its OTI (index within level OTL)
         SID       =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
         OTI       =  8*(int)(OTI/8) + SID ;    // OTI in the buffer was for the original ray that was split
         INDEX     =  OFF[OTL]+OTI ;    // global index -- must be >=0 !!!
# if (DEBUG>0)
         printf("FROMBUF   ") ;
         report(OTL, OTI, RL, &POS, OFF, PAR) ; 
# endif         
         // this stored ray is at level OTL but may be in a cell that is itself still further
         // refined => this will be handled at the start of the main loop => possible further
         // refinement before on the rays is stepped forward
         // However, if we are still at SPLITPOS, new siderays will not be possible before >=1 steps
         BUFFER[c1+2+4*sr] = -1.0f ;   // mark as used
         if (sr==0) NBUF -= 1 ;        // was last of the <=6 subrays in this buffer entry
      }  // (INDEX<0)
      
      
      // if (INDEX<0) { if (NBUF<=0)   FOLLOW=0 ;  }
          
   } // while INDEX>=0  --- stops when buffer is empty and the main level-0 ray has exited
   
   TPL[id] = tpl ;    // TPL[NRAY]  (could be just TPL[GLOBAL], because GLOBAL might be smaller)
}  // END OF PathsOT4















__kernel void UpdateOT40(  // @u
                           const int        id0,     //  first id in the index running over NRAY>=NWG
                           __global float  *PL,      //  0
# if (WITH_HALF==1)
                           __global short4 *CLOUD,   //  1 [CELLS]: vx, vy, vz, sigma
# else
                           __global float4 *CLOUD,   //  1 [CELLS]: vx, vy, vz, sigma
# endif
                           GAUSTORE  float *GAU,     //  2 precalculated gaussian profiles [GNO,CHANNELS]
                           constant int2   *LIM,     //  3 limits of ~zero profile function [GNO]
                           const float      Aul,     //  4 Einstein A(upper->lower)
                           const float      A_b,     //  5 (g_u/g_l)*B(upper->lower)
                           const float      GN,      //  6 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                           const float      APL,     //  7 average path length [GL]
                           const float      BG,      //  8 background value (photons)
                           const float      DIRWEI,  //  9 weight factor (based on current direction)
                           const float      EWEI,    // 10 weight 1/<1/cosT>/NDIR
                           const int        LEADING, //  0 leading edge
                           const REAL3      POS0,    // 11 initial position of id=0 ray
                           const float3     DIR,     // 12 ray direction
                           __global float  *NI,      // 13 [CELLS]:  NI[upper] + NB_NB
                           __global float  *RES,     // 14 [CELLS]: SIJ, ESC ---- or just [CELLS] for SIJ
                           __global float  *NTRUES,  // 15 [NWG*MAXCHN]   --- NWG>=simultaneous level 0 rays
# if (WITH_CRT>0)
                           constant float *CRT_TAU,  //  dust optical depth / GL
                           constant float *CRT_EMI,  //  dust emission photons/c/channel/H
# endif                     
# if (BRUTE_COOLING>0)
                           __global float   *COOL,   // 14,15 [CELLS] = cooling 
# endif
                           __global   int   *LCELLS, //  16
                           __constant int   *OFF,    //  17
                           __global   int   *PAR,    //  18
                           __global   float *RHO,    //  19  -- needed only to describe the hierarchy
                           __global   float *BUFFER_ALL //  20 -- buffer to store split rays
                        )  {   
   // Each ***WORK ITEM*** processes one ray -- no barriers. 
   // The rays are two cells apart to avoid synchronisation problems (?). 
   // Rays start on the leading edge. If ray exits through a side (wrt axis closest to
   // direction of propagation), a new one is created on the opposite side and the ray ends when the
   // downstream edge is reached.
   // 
   // As one goes to a higher hierarchy level (into a refined cell), one pushes the original and two
   // new rays to buffer. The fourth ray (one of the new ones) is followed first. When ray goes up in
   // hierarchy (to a larger cell) ray is terminated if ray was created at a higher level. When one
   // ray is terminated, the next one is taken from the buffer, if that is not empty. When ray goes up
   // in hierarchy, NTRUE is scaled *= 4 or, if one goes up several levels, *=4^n, where n is the
   // decrease in the hierarchy level.
   int id  = get_global_id(0), gs = get_global_size(0), ls=get_local_size(0) ;
   __global float *BUFFER = &BUFFER_ALL[id*(26+CHANNELS)*MAX_NBUF] ;   // here id ~ GLOBAL (allocation for GLOBAL)
   id += id0 ;                     // becomes a running index over NRAY ....  id ~ NRAY
   if (id>=NRAY) return ;          // one work item per ray .... GLOBAL >= NRAY   
   GAUSTORE float *profile ;
   // cannot use __local NTRUE, that could be ~32kB ---  __local  float  NTRUE[CHANNELS*LOCAL] ;
   __global float *NTRUE  =  &(BUFFER_ALL[gs*(26+CHANNELS)*MAX_NBUF + id]) ; // one work item steps with steps of ls
   float weight, w, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed, sum_delta_true, all_escaped, nu ;
   int row, shift, INDEX, c1, c2, OTL, OTI, OTLO, XL, RL, SID, NBUF=0, sr, sid, level, b1, b2, I, i, ind, otl, oti ;
   int level1, ind1 ;
   REAL3  POS, pos0, pos1, pos, RDIR ;
   REAL   dx, dy, dz, s ;
   float dr ;
   float flo ;
# if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, pro, sij ;
# endif  
   RDIR.x = DIR.x ; RDIR.y = DIR.y ; RDIR.z = DIR.z ;
# if (ONESHOT<1)
   int nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ;
# endif
   int *SUBS ;
   POS.x = POS0.x ;   POS.y = POS0.y ;   POS.z = POS0.z ;
   
   // when split done only on the leading edge -- HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
   
# if (ONESHOT<1)
   if (LEADING<3) {
      if (LEADING==0) { 
         HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;
         POS.x =    EPS ;  POS.y += TWO*(id%ny) ;  POS.z += TWO*(int)(id/ny) ;  // id is index running over all NRAY
      }
      if (LEADING==1) { 
         HEAD[0] = 1 ;   HEAD[1] = 3 ;   HEAD[2] = 5 ;  HEAD[3] = 7 ;
         POS.x = NX-EPS ;  POS.y += TWO*(id%ny) ;  POS.z += TWO*(int)(id/ny) ;
      }
      if (LEADING==2) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;
         POS.y =    EPS ;  POS.x += TWO*(id%nx) ;  POS.z += TWO*(int)(id/nx) ;
      }
   } else {
      if (LEADING==3) { 
         HEAD[0] = 2 ;   HEAD[1] = 3 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.y = NY-EPS ;  POS.x += TWO*(id%nx) ;  POS.z += TWO*(int)(id/nx) ;
      }
      if (LEADING==4) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;
         POS.z =    EPS ;  POS.x += TWO*(id%nx) ;  POS.y += TWO*(int)(id/nx) ;
      }
      if (LEADING==5) { 
         HEAD[0] = 4 ;   HEAD[1] = 5 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.z = NZ-EPS ;  POS.x += TWO*(id%nx) ;  POS.y += TWO*(int)(id/nx) ;
      }
   }
# else
   if (LEADING<3) {
      if (LEADING==0) { 
         HEAD[0] = 0 ;       HEAD[1] = 2 ;       HEAD[2] = 4 ;     HEAD[3] = 6 ;
         POS.x =    EPS ;    POS.y += id%NY ;    POS.z += id/NY ;  // id is index running over all NRAY
      }
      if (LEADING==1) { 
         HEAD[0] = 1 ;       HEAD[1] = 3 ;       HEAD[2] = 5 ;     HEAD[3] = 7 ;
         POS.x = NX-EPS ;    POS.y += id%NY ;    POS.z += id/NY ;
      }
      if (LEADING==2) { 
         HEAD[0] = 0 ;       HEAD[1] = 1 ;       HEAD[2] = 4 ;     HEAD[3] = 5 ;
         POS.y =    EPS ;    POS.x += id%NX ;    POS.z += id/NX ;
      }
   } else {
      if (LEADING==3) { 
         HEAD[0] = 2 ;       HEAD[1] = 3 ;       HEAD[2] = 6 ;     HEAD[3] = 7 ;
         POS.y = NY-EPS ;    POS.x += id%NX ;    POS.z += id/NX ;
      }
      if (LEADING==4) { 
         HEAD[0] = 0 ;       HEAD[1] = 1 ;       HEAD[2] = 2 ;     HEAD[3] = 3 ;
         POS.z =    EPS ;    POS.x += id%NX ;    POS.y += id/NX ;
      }
      if (LEADING==5) { 
         HEAD[0] = 4 ;       HEAD[1] = 5 ;       HEAD[2] = 6 ;     HEAD[3] = 7 ;
         POS.z = NZ-EPS ;    POS.x += id%NX ;    POS.y += id/NX ;
      }
   }
# endif
   
   
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;     // remain at root level, not yet going to leaf
   if (OTI<0) return ;
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
   OTLO      =  OTL ;
   RL        =  0 ;
   for(int i=0; i<CHANNELS; i++)  NTRUE[i*gs] = BG * DIRWEI ;  // just a single work item
   
   
   
   while(INDEX>=0) {  // INDEX may refer to a cell that is not a leaf
      
      // barrier(CLK_LOCAL_MEM_FENCE) ;
      
      // If we are not in a leaf, we have gone to some higher level. 
      // Go one level higher, add >=three rays, pick one of them as the current,
      // and return to the beginning of the loop.
      if (RHO[INDEX]<=0.0f) {                    // go to the sub-cell and add sibling rays
         
         sr     =  0 ;
         NBUF   =  min(NBUF, MAX_NBUF-1) ;       // IF BUFFER IS ALREADY FULL !!
         c1     =  NBUF*(26+CHANNELS) ;
         SID    =  -1 ;
         
         POS.x  =  TWO*fmod(POS.x, ONE) ;        // coordinate inside parent cell [0,1]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         flo    =  -RHO[INDEX] ;                 // OTL, OTI of the parent cell
         OTL   +=  1  ;                          // step to next level = refined level
         OTI    =  *(int *)&flo ;                // OTI for the first child in octet
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // original subcell
         OTI   +=  SID;                          // cell in octet, original ray
         c2     =  0 ;   // set c2=1 if ray is to be split (is this the leading cell edge?)
         if (LEADING<3) {
            if ((LEADING==0)&&(POS.x<  EPS2)) c2 = 1 ; 
            if ((LEADING==1)&&(POS.x>TMEPS2)) c2 = 1 ; 
            if ((LEADING==2)&&(POS.y<  EPS2)) c2 = 1 ;
         } else {
            if ((LEADING==3)&&(POS.y>TMEPS2)) c2 = 1 ; 
            if ((LEADING==4)&&(POS.z<  EPS2)) c2 = 1 ;
            if ((LEADING==5)&&(POS.z>TMEPS2)) c2 = 1 ;
         }
         //  rescale always when the resolution changes
         for(int i=0; i<CHANNELS; i++) {   // SCALE ON EVERY REFINEMENT EVEN WHEN NOT SPLIT
            NTRUE[i*gs] *= 0.25f ; 
         }
         // barrier(CLK_LOCAL_MEM_FENCE) ;
         
         if (c2>0) {  // split the ray and choose one of the new ones to be followed first
            for(int i=0; i<CHANNELS; i++) {   // ray effectively split to four
               BUFFER[c1+26+i] = NTRUE[i*gs] ;
            }
            BUFFER[c1+0]  =  OTL ;                     // level where the split is done, OTL>this =>
            BUFFER[c1+1]  =  I2F(OTI) ;                // buffer contains the OTI of the original ray 
            BUFFER[c1+2]  =  POS.x ;                   // Store the original MAIN RAY to buffer as the first one
            BUFFER[c1+3]  =  POS.y ;
            BUFFER[c1+4]  =  POS.z ;
            BUFFER[c1+5]  =  RL  ;                     // main ray exists at levels >= RL
            // Two new subrays added to buffer, third one becomes the current ray
            // Add first two subrays to buffer
            if (                  HEAD[0]==SID) {  // 0=original => (1,2) to buffer, 3 as current
               sid              = HEAD[1] ;
               BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
               BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
               BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
               sid              = HEAD[2] ;
               BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
               BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
               BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
               SID              = HEAD[3] ;  // SID of the subray to be followed first
            } else {
               if (                  HEAD[1]==SID) {
                  sid              = HEAD[2] ;
                  BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                  BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                  BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                  sid              = HEAD[3] ;
                  BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                  BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                  BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                  SID              = HEAD[0] ;
               } else {
                  if (                  HEAD[2]==SID) {
                     sid              = HEAD[3] ;
                     BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                     BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                     BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                     sid              = HEAD[0] ;
                     BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                     BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                     BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                     SID              = HEAD[1] ;
                  } else {
                     if (                  HEAD[3]==SID) {
                        sid              = HEAD[0] ;
                        BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                        BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                        BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                        sid              = HEAD[1] ;
                        BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                        BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                        BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                        SID              = HEAD[2] ;
                     } else {
                        ; // printf("???\n") ;
                     }
                  }
               }
            } 
            // for the two subrays just added, update RL, SPLIT
            sr = 3 ;   // so far the original main ray and two split subrays
            BUFFER[c1+5+4*1]  =  OTL ;    BUFFER[c1+5+4*2]  =  OTL ;  // RL = current OTL
            for(int i=sr; i<6; i++)  BUFFER[c1+2+4*i] = -99.0f ;      // mark the rest as unused
            // We added leading edge rays, old main ray is in buffer, SID refers to one of the subrays
            // update OTI and POS to correspond to that subray
            OTI       =  (*(int *)&flo)  + SID ;    // the new ray to be followed, OTI = index of first subcell + SID
            POS.x     =  fmod(POS.x,ONE) + (int)( SID%2)    ;  // dx and SID known to all work items
            POS.y     =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
            POS.z     =  fmod(POS.z,ONE) + (int)( SID/4)    ;
            RL        =  OTL ;  // when we reach OTL<RL, this ray will be terminated
            NBUF++ ;            // >= 1 subrays were added, NBUF increases just by one
            
         } // c2>0  == we split rays on the leading edge
         
      } // RHO<0
      
      
      INDEX  = OFF[OTL] + OTI ;
      if (RHO[INDEX]<=0.0f) continue ;  // not a leaf, go back to the beginning of the main loop
      
      
      
      
      
# if 1  // adding siderays
      
      // It is essential that we are already in a leaf node when siderays are added:
      // once ray has produced siderays at the current location, it will immediately take a 
      // real step forward.
      
      
      // Current ray is still (OTL, OTI, POS)
      // presumably current NTRUE is correct for ray at level OTL
      
      // Siderays are based on rays hitting the leading edge. We have stepped down to leaf level.
      // Therefore, incoming ray has already been split to four, and NTRUE has been scaled by 0.25 to
      // correspond to the step from OTL-1 to OTL.
      
      
      for(XL=OTL; XL>RL; XL--) {   // try to add subrays at levels   RL<XL<=OTL
         
#  if 1
         if (NBUF>=MAX_NBUF) {
#   if (SAFE>0)
            printf("*** BUFFER FULL -- CANNOT CHECK SIDERAYS\n") ;
#   endif
            break ;
         }
#  endif
         
         if (XL==OTL) {
            // we use root coordinates to determine whether RL ray is also RL-1 ray
            // the required accuracy is only 0.5*0.5**MAXL, float32 works at least to MAXL=15
            pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;   // current position in root coordinates
         }
         
         c1     =  NBUF*(26+CHANNELS) ;          // 26+CHANNELS elements per buffer entry
         sr     =  0  ;
         
         if (((LEADING==0)&&(POS.x<EPS2))||((LEADING==1)&&(POS.x>TMEPS2))) {  // +/-X leading edge, at OTL level
            dr   = pown(0.5f, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.x/dr)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test smaller XL (larger dr) values
            // even number of dr,is therefore a border between level XL octets
            // calculate (pos, level, ind) for the position at level XL  (in the octet that we are in)
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // (pos, level, ind) now define the position at level XL
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL ===== offsets                              Y and Z
            // check XL-scale neighbour
            pos1.x =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       Y * 
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level,NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               // Current ray position at level==XL is defined by (level, ind, pos).
               // We loop over XL positions on the leading-edge plane, ignore those not common with
               // XL-1 rays, and choose those that actually hit the -Y side of the current octet.
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.y += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =       Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;     // still inside the current octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     // pos1 = initial coordinates at the leading-edge level, step forward to the Y sidewall
                     if (DIR.y>0.0f)   pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ;  // step based on    Y ****
                     else              pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;  // to 2.0  (level==RL+1)
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.y  =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in   octet        Y *****
#  endif
                     // Add the ray from the upstream direction to buffer as a new subray
                     //    we leave the buffer entry at level XL, if leafs are >XL, we drop to next 
                     //    refinement level only at the beginning of the main loop.
                     // Each sideray will actually continue in location refined to some level >=XL
                     // We will store NTRUE correct for XL. If sideray is in refined region, beginning of the
                     // main loop will drop the ray to the higher level and rescale NTRUE accordingly.
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level==XL
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL ===== offsets                              Z and Y
            // current ray at level level==RL+1 defined by (level, ind, pos)
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {     // neighbour not refined to level==XL, will not provide XL siderays
               for(int a=-1; a<=3; a++) {       // offset in +/- Z direction from the level RL ray, in XL steps
                  for(int b=-1; b<=3; b++) {    // ray offset in +/- Y direction, candidate relative to current ray
                     if ((a%2==0)&&(b%2==0)) continue ; // skip LR rays
                     // if we come this far, we will add the ray if it just hits the current octet with RL+1 cells
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =      Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // in current octet, will not hit sidewall
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // second offset           Y ***
                     // pos1 = initial coordinates at the leading edge plane, step to the Z sidewall
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in octet,        Z *****
#  endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL     ;      // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (((LEADING==2)&&(POS.y<EPS2))||((LEADING==3)&&(POS.y>TMEPS2))) {  // +/-Y leading edge
            dr   = pown(0.5f, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.y/dr)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // ***A*** even number of dr,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL) IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITTING THE X SIDEWALL ===== offsets                             X and Z
            pos1.y =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.x =  (DIR.x>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       X *
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos inthe octet,      X *****
#  endif
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL =====  offsets                             Z and X
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =     Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =     X ***
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in the octet,    Z *****
#  endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         
         if (((LEADING==4)&&(POS.z<EPS2))||((LEADING==5)&&(POS.z>TMEPS2))) {  // +/-Z leading edge
            dr   = pown(0.5f, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.z/dr)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // even number of dr,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITING THE X SIDEWALL ===== offsets                              X and Y
            pos1.y =  0.1f ;     pos1.z = 0.1f ;
            pos1.x =  (DIR.x>0.0f) ? (-0.1f) : (2.1f) ;    // upstream neighbour, main offset       X *
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.y += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Y ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos in the octet,     X *****
#  endif
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     sr += 1 ;
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL =====  offsets                            Y and X
            pos1.x =  0.1f ;    pos1.z = 0.1f ;
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset       Y *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL 
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =    Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =    X ***
                     if (DIR.y>0.0f) pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ; // step based on      Y ****
                     else            pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.y   =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in the octet,   Y *****
#  endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (sr>0) {  // added some siderays
            // While the current ray was at level OTL, there are no guarantees that the same
            // refinement exists for all added siderays. Therefore, we add all siderays
            // using the level XL coordinates. This also means that the siderays and the
            // leading-edge subrays (below) must be stored as separate BUFFER entries (different NBUF).
            BUFFER[c1+0] = XL ;        // at level XL all rays stored as level XL rays
            BUFFER[c1+1] = I2F(ind) ;  // index of *original ray*, at level XL
            // We leave the original ray as the current one == (OTL, OTI, POS) remain unchanged.
            for(int i=sr; i<6; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
            // NTRUE is correct for OTL but we store data for level XL
            // Therefore, BUFFER will contain  NTRUE * 0.25**(XL-OTL)  == NTRUE * 4**(OTL-XL)
            // ==>  the BUFFER values are   NTRUE(correct for RL) * 4.0**(OTL-XL), OTL and not RL !!!!
            // When sideray is taken from buffer and if it is located in a region with OTL>XL, the
            // beginning of the main loop will again rescale with  4.0*(XL-OTL)
            dr  =  pown(4.0f, OTL-XL) ;             // was RL-XL but surely must be OTL-XL !!!!!
            for(int i=0; i<CHANNELS; i++) {
               BUFFER[c1+26+i] = NTRUE[i*gs]*dr ;   // NTRUE scaled from RL to XL 
            }            
            NBUF += 1 ;            
         }
         
      } // for XL -- adding possible siderays
      
      
      // Subrays are added only when we are at leaf level. Also, when subrays are added, 
      // we continue with the original RL ray and immediately make a step forward. Therefore
      // there is no risk that we would add siderays multiple times.
# endif // adding siderays
      
      
      
      
      // global index -- must be now >=0 since we started with while(INDEX>=0) and just possibly went down a level
      INDEX = OFF[OTL]+OTI ;   
      
      // if not yet a leaf, jump back to start of the loop => another step down
      if (RHO[INDEX]<=0.0f) {
         continue ;
      }
      
      // we are now in a leaf node, ready to make the step
      OTLO   =  OTL ;
      
      
      
      dr     =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL   , LEADING) ;  // POS is now the end of this step
      
      
      
# if 0   // @@ double check PL calculation ... PL[:] should be reduced to zero
      AADD(&(PL[INDEX]), -dr) ;
# endif
      
      
      
      // if (RHO[INDEX]>CLIP) { // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                  
      
# if 0
      for(int i=0; i<LOCAL; i++) {
         printf("[%3d]  %2d %6d   %8.4f %8.4f %8.4f   %2d\n", id, OTL, OTI, POS.x, POS.y, POS.z, NBUF) ;
      }
# endif
      
      
      // with path length already being the same in all cells !   V /=8,  rays*4==length/2 /=2 =>  2/8=1/4
      weight    =  (dr/APL) *  VOLUME  *  pown(0.25f, OTLO) ;  // OTL=1 => dr/APL=0.5
      
      // INDEX is still the index for the cell where the step starts
      nu        =  NI[2*INDEX] ;
# if 0
      nb_nb     =  NI[2*INDEX+1] ;
# else
      nb_nb     =  max(1.0e-30f, NI[2*INDEX+1]) ; // $$$ KILL MASERS 
# endif
      
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# if (WITH_HALF==1)
      doppler  *=  0.002f ;  // half integer times 0.002f km/s
# endif
      
# if 0 // did not solve the oscillating-cell problem
      // ... one must avoid division by zero but lets move that to host side
      if (fabs(nb_nb)<1.0e-25f)  nb_nb   = 1.0e-25f ;
# endif        
      
      
      // tmp_tau   =  max(1.0e-35f, dr*nb_nb*GN) ;
      tmp_tau   =  dr*nb_nb*GN ;
      
      
      tmp_emit  =  weight * nu*Aul / tmp_tau ;  // GN include grid length [cm]
      shift     =  round(doppler/WIDTH) ;
# if (WITH_HALF==1)      //               sigma = 0.002f * w,   lookup table: sigma = SIGMA0 * SIGMAX^row
      row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
# else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
# endif
      profile   =  &GAU[row*CHANNELS] ;
      // avoid profile function outside profile channels LIM.x, LIM.y
      c1        =  max(LIM[row].x+shift, max(0, shift)) ;
      c2        =  min(LIM[row].y+shift, min(CHANNELS-1, CHANNELS-1+shift)) ;
      sum_delta_true = 0.0f ;
      all_escaped    = 0.0f ;
      
      
# if (WITH_CRT>0) // WITH_CRT
      sij = 0.0f ;
      // Dust optical depth and emission
      //   here escape = line photon exiting the cell + line photons absorbed by dust
      Ctau      =  dr     * CRT_TAU[INDEX] ;
      Cemit     =  weight * CRT_EMI[INDEX] ;      
      for(int i=c1; i<=c2; i++)  {
         pro    =  profile[i-shift] ;
         Ltau   =  tmp_tau*pro ;
         Ttau   =  Ctau + Ltau ;
         // tt     =  (1.0f-exp(-Ttau)) / Ttau ;
         tt     =  (fabs(Ttau)>0.01f) ?  ((1.0f-exp(-Ttau))/Ttau) : (1.0f-Ttau*(0.5f-0.166666667f*Ttau)) ;
         // ttt    = (1.0f-tt)/Ttau
         ttt    =  (1.0f-tt)/Ttau ;
         // Line emission leaving the cell   --- GL in profile
         Lleave =  weight*nu*Aul*pro * tt ;
         // Dust emission leaving the cell 
         Dleave =  Cemit *                     tt ;
         // SIJ updates, first incoming photons then absorbed dust emission
         sij   +=  A_b * pro*GN*dr * NTRUE[i*gs]*tt ;
         // sij         += A_b * profile[i]*GN*Cemit*dr*(1.0f-tt)/Ttau ; // GN includes GL!
         sij   +=  A_b * pro*GN*dr * Cemit*ttt ;    // GN includes GL!
         // "Escaping" line photons = absorbed by dust or leave the cell
         all_escaped +=  Lleave  +  weight*nu*Aul*pro * Ctau * ttt ;
         // Total change of photons in the package
         NTRUE[i*gs]  =  NTRUE[i*gs]*exp(-Ttau) + Dleave + Lleave ;
      }  // loop over channels
      // RES[2*INDEX]    += sij ;            // division by VOLUME done in the solver (kernel)
      // RES[2*INDEX+1]  += all_escaped ;    // divided by VOLUME only oin Solve() !!!
#  if (NO_ATOMICS>0)
      RES[2*INDEX]    +=  sij ;
      RES[2*INDEX+1]  +=  all_escaped ;
#  else
      AADD(&(RES[2*INDEX]), sij) ;
      AADD(&(RES[2*INDEX+1]), all_escaped) ;
#  endif
      
# else   // not  WITH_CRT ***************************************************************************************
      
      
      
#  if (WITH_ALI>0) // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
      // Taylor approximations remain accurate as argument goes to zero !!!
      //              no change if using -- fabs(nb_nb)>1.0e-25f
      // if (fabs(nb_nb)>1.0e-30f) {  // $$$
      for(int i=c1; i<=c2; i++)  {
         w               =  tmp_tau*profile[i-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         // absence affects strongly low-density regions
#   if 0
         factor          =  clamp(factor, -1.0e-12f, 1.0f) ;  // $$$
#   else
         factor          =  clamp(factor,  1.0e-30f, 1.0f) ;  // KILL MASERS $$$
#   endif
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i*gs]*factor ; // incoming photons that are absorbed
         NTRUE[i*gs]    +=  escape-absorbed ;
         sum_delta_true +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      // SDT[lid] = sum_delta_true ;  AE[lid] = all_escaped ; // all work items save their own results
      // barrier(CLK_LOCAL_MEM_FENCE) ;             // all agree on NTRUE, all know SDT and AE
      all_escaped     =  clamp(all_escaped, 0.0001f*weight*nu*Aul, 0.9999f*weight*nu*Aul) ; // must be [0,1]
      // RES[2*INDEX]   +=  A_b * (sum_delta_true / nb_nb) ;
      // RES[2*INDEX+1] +=  all_escaped ;
      w  =  A_b * (sum_delta_true/nb_nb) ;
#   if (NO_ATOMICS>0)
      RES[2*INDEX]    +=   w ;
      RES[2*INDEX+1]  +=   all_escaped ;
#   else
      AADD(&(RES[2*INDEX]),    w) ;
      AADD(&(RES[2*INDEX+1]),  all_escaped) ;
#   endif
      
#  else // else no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
      
      // if (isfinite(tmp_tau)==0) printf("tmp_tau not finite !!!\n") ;
      
      for(int i=c1; i<=c2; i++)  {
         w               =  tmp_tau*profile[i-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
#   if 0
         factor          =  clamp(factor, -1.0e-12f, 1.0f) ;  // $$$
#   else
         factor          =  clamp(factor,  1.0e-30f, 1.0f) ;  // KILL MASERS $$$
#   endif
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i*gs]*factor ; // incoming photons that are absorbed
         NTRUE[i*gs]    +=  escape-absorbed ;
         sum_delta_true +=  absorbed - escape ;  // later sum_delta_true +=  W*nu*Aul
      }   // over channels
      // SDT[lid] = sum_delta_true ;
      // barrier(CLK_LOCAL_MEM_FENCE) ;
      // for(int i=1; i<LOCAL; i++)  sum_delta_true += SDT[i] ;    
      w  =   A_b  * ((weight*nu*Aul + sum_delta_true) / nb_nb)  ;
      // RES[INDEX] += w ;
      // AADD(&(RES[INDEX]), w) ;
#   if (NO_ATOMICS>0)
      RES[INDEX] += w ;
#  else
      AADD((__global float*)(RES+INDEX), w) ;
#  endif
#  endif // no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
# endif  // WITH OR WITHOUT CRT
      
      
      // } // RHO>CLIP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      
# if (BRUTE_COOLING>0)
      // total number of photons in the package as it exits the cell
      float cool = 0.0f ;
      for(int i=0; i<CHANNELS; i++) cool += NTRUE[i*gs] ;
      COOL[INDEX] += cool ; // cooling of cell INDEX --- each work group distinct rays => no need for atomics
# endif
      
      
      
      
      // Updates at the end of the step, POS has been already updated, OTL and OTI point to the new cell
      INDEX   =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;
      
# if (BRUTE_COOLING>0)  // heating of the next cell, once INDEX has been updated
      if (INDEX>=0) {
         COOL[INDEX] -= cool ; // heating of the next cell
      }
# endif
      
      if (INDEX>=0) {
         if (RHO[INDEX]<=0.0f) { // we stepped to a refined cell (GetStep goes only to a cell OTL<=OTLO)
            continue ;           // step down one level at the beginning of the main loop
         }
      }
      
      
      if (OTL<RL) {        // we are up to a level where this ray no longer exists
         INDEX=-1 ;        
      } else {      
         if (INDEX<0) {    // ray exits the cloud... possibly continues on the other side
            if (POS.x>=NX  ) {   if (LEADING!=0)  POS.x =    EPS ;   }
            if (POS.x<=ZERO) {   if (LEADING!=1)  POS.x = NX-EPS ;   } 
            if (POS.y>=NY  ) {   if (LEADING!=2)  POS.y =    EPS ;   }
            if (POS.y<=ZERO) {   if (LEADING!=3)  POS.y = NY-EPS ;   } 
            if (POS.z>=NZ  ) {   if (LEADING!=4)  POS.z =    EPS ;   }
            if (POS.z<=ZERO) {   if (LEADING!=5)  POS.z = NZ-EPS ;   } 
            IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;  // we remain in a root-grid cell => OTL==0 !
            INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
            if (INDEX>=0) {   // new level-0 ray started on the opposite side (may be a parent cell)
               RL = 0 ;  OTLO = 0 ; 
               // we had already barrier after the previous NTRUE update
               for(int i=0; i<CHANNELS; i++)  NTRUE[i*gs] = BG * DIRWEI ;
               // barrier(CLK_LOCAL_MEM_FENCE) ;
# if (BRUTE_COOLING>0)
               dr = BG*DIRWEI*CHANNELS ;  COOL[INDEX] -= dr ; // heating of the entered cell
# endif
               continue ;
            }
         } // if INDEX<0
      }
      
      
      // rescale on every change of resolution
      if ((INDEX>=0)&&(OTL<OTLO)) {   // @s ray continues at a lower hierarchy level => NTRUE may have to be scaled
         dr = pown(4.0f, OTLO-OTL) ;  // scale on every change of resolution
         for(int i=0; i<CHANNELS; i++)  NTRUE[i*gs] *= dr ;     
         continue ;  // we went to lower level => this cell is a leaf
      }
      
      
      // if INDEX still negative, try to take a ray from the buffer
      // 0   1     2  3  4  6     ...       NTRUE[CHANNELS] 
      // OTL OTI   x  y  z  RL    x y z RL                  
      if ((INDEX<0)&&(NBUF>0)) {            // NBUF>0 => at least one ray exists in the buffer
         c1    =  (NBUF-1)*(26+CHANNELS) ;  // 8+CHANNELS elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // OTL ...
         OTLO  =  OTL ;                     // ???
         OTI   =  F2I(BUFFER[c1+1]) ;       // and OTI of the ray that was split
         for(sr=5; sr>=0; sr--) {         
            dr    =  BUFFER[c1+2+4*sr] ;    // read dr
            if (dr>-0.1f) break ;           // found a ray
         }
         POS.x   =  dr ;
         POS.y   =  BUFFER[c1+3+4*sr] ;  
         POS.z   =  BUFFER[c1+4+4*sr] ;
         RL      =  BUFFER[c1+5+4*sr] ;
         SID     =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
         OTI     =  8*(int)(OTI/8) + SID ;        // OTI in the buffer was for the original ray that was split
         INDEX   =  OFF[OTL]+OTI ;                // global index -- must be >=0 !!!
         // copy NTRUE --- values in BUFFER have already been divided by four
         // barrier(CLK_GLOBAL_MEM_FENCE) ;          // all have read BUFFER
         BUFFER[c1+2+4*sr] = -1.0f ; // mark as used
         if (sr==0)   NBUF -= 1 ;
         c1     +=  26 ;
         for(int i=0; i<CHANNELS; i++)  NTRUE[i*gs] = BUFFER[c1+i] ;  // NTRUE correct for level OTL
         // note - this ray be inside a parent cell => handled at the beginning of the main loop
      } // (INDEX<=0)&(NBUF>0)
      
      
   } // while INDEX>=0
   
}  // end of UpdateOT4()




#endif // WITH_OCTREE==40     no lid, id is actually id0+id, no lid used










#if (WITH_OCTREE==4)



# if (DEBUG>0)
void report(const int       OTL, 
            const int       OTI,
            const int       OTL_RAY, 
            REAL3          *POS,
            __constant int *OFF, 
            __global int   *PAR,
            const int        NBUF) {
   REAL3  pos = *POS ;
   RootPos(&pos, OTL, OTI, OFF, PAR) ;   
   printf(" (%2d,%2d) %4d   L: %6.2f %6.2f %6.2f   R: %6.2f %6.2f %6.2f  OTL=%d OTL_RAY=%d NBUF=%d\n",
          OTL, OTI, OFF[OTL]+OTI, POS->x, POS->y, POS->z, pos.x, pos.y, pos.z, OTL, OTL_RAY, NBUF) ;   
}
# endif








# if (PLWEIGHT>0)   // do not even compile the routine if it is not going to be used

__kernel void PathsOT4(  // @p
                         const    int      gid0,    // do in batches, now starting with gid==gid0
                         __global float   *PL,      // [CELLS]
                         __global float   *TPL,     // TPL[NRAY]
                         __global int     *COUNT,   // COUNT[NRAY] total number of rays entering the cloud
                         const    int      LEADING, // leading edge
                         const    REAL3    POS0,    // initial position of ray 0
                         const    float3   DIR,     // direction of the rays
                         __global   int   *LCELLS,   
                         __constant int   *OFF,
                         __global   int   *PAR,
                         __global float   *RHO,
                         __global float   *BUFFER_ALL
                      ) {   
   // OT4 = split rays at the leading edge and sides, should give identical path length for every cell
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge, if ray exits through a side, a new one is created 
   // on the opposite side. Ray ends when the downstream edge is reached
     
   // This version tries to make sure we make all the rays so that there is no variation in the
   // path lengths between the cells (avoids the need for PL weighting!!)
   // When entering a refined cell with level0, split the ray to four on the leading edge, like in PathsOT2
   // Regarding additional rays hitting the sides of the refined cell, check the two upstream neighbours.
   // If the neighbour is:
   //   - at level >= level0, do nothing -- refined rays will come from the neighbour
   //   - otherwise, test 4 refined rays that might hit the current refined cell from that side
   //     skip the rays that will come otherwise at some point from the neighbour
   int  ls = get_local_size(0),  gid = get_group_id(0), lid=get_local_id(0) ;
   if  (get_local_id(0)!=0)    return ;     // a single work item from each work group !!
   
   if  (gid>=NWG) return ;
   __global float *BUFFER = &(BUFFER_ALL[gid*(26+CHANNELS)*MAX_NBUF]) ;
   
   // ONLY NOW MAKE gid ~ RUNNING INDEX OVER NRAY
   gid += gid0 ;        // index running over all NRAY:  TPL[NRAY]
   if  (gid>=NRAY) return ;  // test again...
   
   __global int *count  = &COUNT[gid] ;     // COUNT[NRAY]
   *count = 0 ;
   int  INDEX, SID, NBUF=0, sr, sid, level, b1, XL, ind ;
   int  OTL, OTI, OTLO, RL, c1, c2, i, oti, otl, otl_ray, ae, be, level1, ind1 ;
   float  dr ;
   float flo, tpl=0.0f ;
#  if (ONESHOT<1)
   int nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ;
#  endif
   REAL  dx ;
   REAL3 POS, pos, pos0, pos1, RDIR ;
   POS.x  = POS0.x ; POS.y  = POS0.y ; POS.z  = POS0.z ;
   RDIR.x = DIR.x ;  RDIR.y = DIR.y ;  RDIR.z = DIR.z ;
   
   // HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
#  if (ONESHOT<1)
   if (LEADING<2){ 
      if (LEADING==0) {
         HEAD[0] = 0 ;          HEAD[1] = 2 ;             HEAD[2] = 4 ;       HEAD[3] = 6 ;
         POS.x   = EPS ;        POS.y += TWO*(gid%ny) ;   POS.z  += TWO*(int)(gid/ny) ;
      } else {
         HEAD[0] = 1 ;          HEAD[1] = 3 ;             HEAD[2] = 5 ;       HEAD[3] = 7 ;
         POS.x   = NX-EPS ;     POS.y  += TWO*(gid%ny) ;  POS.z  += TWO*(int)(gid/ny) ;
      }
   } else {
      if (LEADING<4) {
         if (LEADING==2) { 
            HEAD[0] = 0 ;       HEAD[1] = 1 ;             HEAD[2] = 4 ;       HEAD[3] = 5 ;
            POS.y   = EPS ;     POS.x  += TWO*(gid%nx) ;  POS.z  += TWO*(int)(gid/nx) ; 
         } else {
            HEAD[0] = 2 ;       HEAD[1] = 3 ;             HEAD[2] = 6 ;       HEAD[3] = 7 ;
            POS.y   = NY-EPS ;  POS.x  += TWO*(gid%nx) ;  POS.z  += TWO*(int)(gid/nx) ;
         }
      } else {      
         if (LEADING==4) { 
            HEAD[0] = 0 ;       HEAD[1] = 1 ;             HEAD[2] = 2 ;        HEAD[3] = 3 ;
            POS.z   = EPS ;     POS.x  += TWO*(gid%nx) ;  POS.y  += TWO*(int)(gid/nx) ; 
         } else {
            HEAD[0] = 4 ;       HEAD[1] = 5 ;             HEAD[2] = 6 ;        HEAD[3] = 7 ;
            POS.z   = NZ-EPS ;  POS.x += TWO*(gid%nx) ;   POS.y  += TWO*(int)(gid/nx) ;
         }
      }
   }
#  else
   if (LEADING<2){ 
      if (LEADING==0) {
         HEAD[0]    = 0 ;       HEAD[1] = 2 ;             HEAD[2] = 4 ;       HEAD[3] = 6 ;
         POS.x      = EPS ;     POS.y  += gid%NY ;        POS.z  += gid/NY ;
      } else {
         HEAD[0]    = 1 ;       HEAD[1] = 3 ;             HEAD[2] = 5 ;       HEAD[3] = 7 ;
         POS.x      = NX-EPS ;  POS.y  += gid%NY ;        POS.z  += gid/NY ;
      }
   } else {
      if (LEADING<4) {
         if (LEADING==2) { 
            HEAD[0] = 0 ;       HEAD[1] = 1 ;             HEAD[2] = 4 ;       HEAD[3] = 5 ;
            POS.y   = EPS ;     POS.x  += gid%NX ;        POS.z  += gid/NX ; 
         } else {
            HEAD[0] = 2 ;       HEAD[1] = 3 ;             HEAD[2] = 6 ;       HEAD[3] = 7 ;
            POS.y   = NY-EPS ;  POS.x  += gid%NX ;        POS.z  += gid/NX ;
         }
      } else {      
         if (LEADING==4) { 
            HEAD[0] = 0 ;       HEAD[1] = 1 ;             HEAD[2] = 2 ;       HEAD[3] = 3 ;
            POS.z   = EPS ;     POS.x  += gid%NX ;        POS.y  += gid/NX ; 
         } else {
            HEAD[0] = 4 ;       HEAD[1] = 5 ;             HEAD[2] = 6 ;       HEAD[3] = 7 ;
            POS.z   = NZ-EPS ;  POS.x  += gid%NX ;        POS.y  += gid/NX ;
         }
      }
   }
#  endif
   
   int *SUBS ;
   
   
   // IndexGR takes [0,NX] coordinates and just returns the root-grid index
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;   // remain at root level, not yet going to leaf -- this just calculates index
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;
   OTLO      =  OTL ;
   RL        =  0 ;    // level of the ray (0 for the root-grid rays)
   if (INDEX>=0)  *count += 1 ;  // number of incoming rays
   
#  if (DEBUG>0)
   printf("%6d -  CREATE   %12.7f %12.7f %12.7f \n", gid, POS.x, POS.y, POS.z) ;
   report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
#  endif   
   
   
   
   
   while(INDEX>=0) {  // INDEX may refer to a cell that is not a leaf
      
#  if 0
      if (OTI>=24141184) {
         printf("--- 0: %3d --------------------------------------------------------------------------\n", steps) ;
         printf(" %9d  %2d %9d  P %9.7f %9.7f %9.7f   D %7.4f %7.4f %7.4f  RHO %.3e\n", 
                OFF[OTL]+OTI, OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, RHO[OFF[OTL]+OTI]) ;
         report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
         printf("-------------------------------------------------------------------------------------\n") ;
         printf("0:") ; report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
      }
#  endif
      
      // ===== ADD LEADING EDGE RAYS WHEN ENTERING REFINED REGION THROUGH UPSTREAM CELL BORDER =====
      // This guarantees that the leading edge always has one ray per cell (4 per octet).
      // These are added whenever we step into a refined cell. The other needed rays are added as
      // "siderays" above. Also that relies on the fact that for each octet at level OTL, among the
      // rays that hit its leading edge, there is exactly one ray with RL < OTL (as well as three 
      // with RL==OTL).
      // If we step over several levels, we must again worry about the precision of POS
      // After each normal step we are EPS from a cell boundary.... but we might step
      // here several layers steeper!!
      if (RHO[INDEX]<=0.0f) {                    // go to the sub-cell and add sibling rays
         SID    =  -1 ;
         
#  if (SAFE>0)
         if (NBUF>=(MAX_NBUF-1)) {
            printf("FULL !!!   gid %d, NBUF %d\n", gid, NBUF) ;  
            return ;
         }
#  endif
         
         NBUF   =  min(NBUF, MAX_NBUF-1) ;       // IF BUFFER IS ALREADY FULL !!         
         c1     =  NBUF*(26+CHANNELS) ;          // 62+CHANNELS elements per buffer entry
         sr     =  0  ;
         // If we are not in a leaf, we have gone to go one level down,
         // add >=three rays, pick one of them as the current, return to the beginning of the loop.
         POS.x  =  TWO*fmod(POS.x, ONE) ;      // coordinates inside the single child cell [0,1]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         flo    =  -RHO[INDEX] ;                 // OTI for the first subcell in octet
         OTL   +=  1  ;                          // step to the next refinement level
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // SID for subcell with the ray
         OTI    =  *(int *)&flo + SID ;          // cell of the incoming ray
         
#  if 0
         if (OTI>=24141184) {
            printf("--- LEADING EDGE A --------------------------------------------------------------\n") ;
            printf(" %9d  %2d %9d  P %9.7f %9.7f %9.7f   D %7.4f %7.4f %7.4f  RHO %.3e\n", 
                   OFF[OTL]+OTI, OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, RHO[OFF[OTL]+OTI]) ;
            report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
            printf("----------------------------------------------------------------------------------\n") ;
            printf("0:") ; report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
         }
#  endif
         
         c2     =  0 ;                           // set c2=1 if ray is to be split: is this the leading edge?
         if (LEADING<3) {
            if ((LEADING==0)&&(POS.x<  EPS2)) c2 = 1 ; 
            if ((LEADING==1)&&(POS.x>TMEPS2)) c2 = 1 ; 
            if ((LEADING==2)&&(POS.y<  EPS2)) c2 = 1 ;
         } else {
            if ((LEADING==3)&&(POS.y>TMEPS2)) c2 = 1 ; 
            if ((LEADING==4)&&(POS.z<  EPS2)) c2 = 1 ;
            if ((LEADING==5)&&(POS.z>TMEPS2)) c2 = 1 ;
         }
         if (c2>0) {  // split the ray and choose one of the new ones to be followed first
            BUFFER[c1+0]  =  OTL ;               // level where the split is done
            BUFFER[c1+1]  =  I2F(OTI) ;          // buffer contains OTI of the original ray
            
#  if 0
            if (OTI>=24141183) {
               printf("--- LEADING EDGE B ---------------------------------------------------------------\n") ;
               printf(" %9d  %2d %9d  P %9.7f %9.7f %9.7f   D %7.4f %7.4f %7.4f  RHO %.3e\n", 
                      OFF[OTL]+OTI, OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, RHO[OFF[OTL]+OTI]) ;
               report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
               c2  =  F2I(BUFFER[c1+1]) ;
               printf(" BUFFER %.3f, OTI %d, c2 %d\n", BUFFER[c1+1], OTI, c2) ;
               //  BUFFER 24141184.000, OTI 24141183, c2 24141184
               // CONVERSION  INT -> FLOAT IS IN CORRECT !!!!
               c2 = 1 ;
               printf("----------------------------------------------------------------------------------\n") ;
               printf("0:") ; report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
            }
#  endif     
            
            
            // Put the main ray as the first, sr=0 entry in buffer
            BUFFER[c1+2]  =  POS.x ; 
            BUFFER[c1+3]  =  POS.y ;
            BUFFER[c1+4]  =  POS.z ;
            BUFFER[c1+5]  =  RL    ;
            sr            =  1 ;
            // Add two more subrays to buffer, the third one is taken as the current ray
            // HEAD is the four cells on the upstream border
            if (                     HEAD[0] ==  SID) {  // 0=original => (1,2) to buffer, 3 as current
               sid                            =  HEAD[1] ;
               BUFFER[c1+2+4*sr]              =  fmod(POS.x,ONE) + (int)( sid % 2) ;
               BUFFER[c1+3+4*sr]              =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
               BUFFER[c1+4+4*sr]              =  fmod(POS.z,ONE) + (int)( sid/4) ;
               sid                            =  HEAD[2] ;
               BUFFER[c1+2+4*(sr+1)]          =  fmod(POS.x,ONE) + (int)( sid % 2) ;
               BUFFER[c1+3+4*(sr+1)]          =  fmod(POS.y,ONE) + (int)((sid/2)%2)  ;
               BUFFER[c1+4+4*(sr+1)]          =  fmod(POS.z,ONE) + (int)( sid/4) ;
               SID                            =  HEAD[3] ;  // SID = the subray that becomes the current ray
            } else {
               if (                  HEAD[1] ==  SID) {
                  sid                         =  HEAD[2] ;
                  BUFFER[c1+2+4*sr]           =  fmod(POS.x,ONE) + (int)( sid   %2) ;
                  BUFFER[c1+3+4*sr]           =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                  BUFFER[c1+4+4*sr]           =  fmod(POS.z,ONE) + (int)( sid/4) ;
                  sid                         =  HEAD[3] ;
                  BUFFER[c1+2+4*(sr+1)]       =  fmod(POS.x,ONE) + (int)( sid   %2) ;
                  BUFFER[c1+3+4*(sr+1)]       =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                  BUFFER[c1+4+4*(sr+1)]       =  fmod(POS.z,ONE) + (int)( sid/4) ;
                  SID                         =  HEAD[0] ;
               } else {
                  if (               HEAD[2] ==  SID) {
                     sid                      =  HEAD[3] ;
                     BUFFER[c1+2+4*sr]        =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                     BUFFER[c1+3+4*sr]        =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                     BUFFER[c1+4+4*sr]        =  fmod(POS.z,ONE) + (int)( sid/4) ;
                     sid                      =  HEAD[0] ;
                     BUFFER[c1+2+4*(sr+1)]    =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                     BUFFER[c1+3+4*(sr+1)]    =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                     BUFFER[c1+4+4*(sr+1)]    =  fmod(POS.z,ONE) + (int)( sid/4) ;
                     SID                      =  HEAD[1] ;
                  } else {
                     if (            HEAD[3] ==  SID) {
                        sid                   =  HEAD[0] ;
                        BUFFER[c1+2+4*sr]     =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                        BUFFER[c1+3+4*sr]     =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                        BUFFER[c1+4+4*sr]     =  fmod(POS.z,ONE) + (int)( sid/4) ;
                        sid                   =  HEAD[1] ;
                        BUFFER[c1+2+4*(sr+1)] =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                        BUFFER[c1+3+4*(sr+1)] =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                        BUFFER[c1+4+4*(sr+1)] =  fmod(POS.z,ONE) + (int)( sid/4) ;
                        SID                   =  HEAD[2] ;
                     } else {
                        ; // printf("??????????????? LEADING %d, SID %d\n", LEADING, SID) ;
                     }
                  }
               }
            }
            // two subrays added to buffer, update their RL
            // remember that when refinement jumps to higher value, we come here for every step in refinement
            // before stepping forward
            BUFFER[c1+5+4*sr]  =  OTL ;       BUFFER[c1+5+4*(sr+1)]  =  OTL ;   // RL == current OTL
            sr += 2 ;   // main ray and  two subrays added, the third subray==SID will become the current one
            for(int i=sr; i<6; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused, max 6 subrays
            NBUF += 1 ;
            
#  if (SAFE>0)
            if (NBUF>=MAX_NBUF) {
               printf("ADD UPSTREAM  LEVEL %d -> %d    NBUF = %2d\n", OTL-1, OTL, NBUF) ;
               printf("BUFFER FULL !!!\n") ;
            }
#  endif
            
            
#  if (DEBUG>0)
            printf("------------------------------------------------------------------------------------\n") ;
            for(int i=0; i<sr; i++) {
               printf("ADD.LEAD%d ", i) ;
               otl       =  BUFFER[c1+0] ;
               oti       =  F2I(BUFFER[c1+1]) ;
               pos.x     =  BUFFER[c1+2+4*i] ; pos.y =  BUFFER[c1+3+4*i] ; pos.z =  BUFFER[c1+4+4*i] ;
               sid       =  4*(int)floor(pos.z) + 2*(int)floor(pos.y) + (int)floor(pos.x) ; 
               oti       =  8*(int)(OTI/8) + SID ;
               otl_ray   =  BUFFER[c1+5+4*i] ;            
               report(otl, oti, otl_ray, &pos, OFF, PAR, NBUF) ;
            }
            printf("------------------------------------------------------------------------------------\n") ;
#  endif         
            
            // SID = is the third new leading-edge subray =>  switch to that
            // OTL was already increase by one = refinement
            OTI       =  (*(int *)&flo) + SID ;    // the new ray to be followed, OTI = index of first subcell
            POS.x     =  fmod(POS.x,ONE) + (int)( SID%2)    ;
            POS.y     =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
            POS.z     =  fmod(POS.z,ONE) + (int)( SID/4)    ;
            RL        =  OTL ;      // leading edge rays created at level OTL
            
#  if 0
            if (OTI>=24141183) {
               printf("--- LEADING EDGE C ---------------------------------------------------------------\n") ;
               printf(" %9d  %2d %9d  P %9.7f %9.7f %9.7f   D %7.4f %7.4f %7.4f  RHO %.3e\n", 
                      OFF[OTL]+OTI, OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, RHO[OFF[OTL]+OTI]) ;
               report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
               printf("----------------------------------------------------------------------------------\n") ;
               printf("0:") ; report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
            }
#  endif
            
            
         }  //  c2>0  == other leading-edge subrays added
      } // RHO < 0.0 == we entered refined region
      
      
      
      
      INDEX = OFF[OTL]+OTI ;   // global index -- must be now >=0 since we went down a level
      
      
      
#  if 0      
      if (OTI>=24141184) {
         printf("--- LEADING EDGE D ---------------------------------------------------------------\n") ;
         printf(" %9d  %2d %9d  P %9.7f %9.7f %9.7f   D %7.4f %7.4f %7.4f  RHO %.3e\n", 
                OFF[OTL]+OTI, OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, RHO[OFF[OTL]+OTI]) ;
         report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
         printf("----------------------------------------------------------------------------------\n") ;
         printf("0:") ; report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
      }
#  endif
      
      
      // if not yet a leaf, jump back to start of the loop, for another step down
      if (RHO[INDEX]<=0.0f) {
         // printf("RHO<0\n") ;
#  if 0
         steps += 1  ;   
         if (steps>10000) printf("STEPS>10000 --- RHO[%d]<=0.0  --- OTL %d, OTI %d\n", INDEX, OTL, OTI) ;
#  endif
         continue ;
      }
      
      
#  if 0
      if ((FOLLOW)||(OTI>=24141184)) {
         if (steps==19) {
            printf("--- 2: %3d -------------------------------------------------------------------------\n", steps) ;
            printf(" %9d  %2d %9d  P %9.7f %9.7f %9.7f   D %7.4f %7.4f %7.4f  RHO %.3e\n", 
                   OFF[OTL]+OTI, OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, RHO[OFF[OTL]+OTI]) ;
            report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
            printf("------------------------------------------------------------------------------------\n") ;
            printf("0:") ; report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
            // return ;
         }
      }
#  endif
      
      
      
#  if 1  // ******************** SIDERAYS ********************************
      
      // RL  = ray level, the level at which the ray was created, root rays are RL=0
      // OTL = cell level, the current discretisation level, root grid is OTL=0
      // =>  Check at each step if there are potential siderays to be added.
      // RL rays provide siderays at levels XL,    RL <  XL <= OTL
      //    (A) it is boundary between level XL octets
      //    (B) upstream-side neighbour has OTL<XL -- so that it will not provide the XL rays directly
      //    (C) the XL sideray is not also XL-1 ray --  those XL-1 siderays are provided by XL-2 rays
      //        since current ray is XL ray, skip even offsets where the step=1.0 in level XL coordinates
      //    (D) the XL sideray actually hits the side of the current octet with level XL cells
      
      
      // Current ray is still (OTL, OTI, POS)
      for(XL=OTL; XL>RL; XL--) {   // try to add subrays at level XL -- never on level=0
         
         
         if (XL==OTL) {
#   if (SAFE>0)
            if (NBUF>=MAX_NBUF) {
               printf("*** BUFFER FULL -- CANNOT CHECK SIDERAYS\n") ; 
               break ;
            }
#   endif
            pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;   // current position in root coordinates
         }
         
         c1     =  NBUF*(26+CHANNELS) ;          // 26+CHANNELS elements per buffer entry
         sr     =  0  ;
         
         // printf("!!!  RL=%d  <  XL=%d  <= OTL=%d\n", RL, XL, OTL) ;
         
         
         if (((LEADING==0)&&(POS.x<EPS2))||((LEADING==1)&&(POS.x>TMEPS2))) {  // +/-X leading edge, at OTL level
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.x/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test smaller XL (larger dx) values
            // even number of dx,is therefore a border between level XL octets
            // calculate (pos, level, ind) for the position at level XL  (in the octet that we are in)
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            // Note IndexUP used only at levels>0 => independent of the handling of the root-grid coordinates
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // (pos, level, ind) now define the position at level XL
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL ===== offsets                              Y and Z
            // check XL-scale neighbour
            pos1.x =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       Y * 
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            // It is called only at levels level>0, not affected by the handling of root-grid coordinates
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               // Current ray position at level==XL is defined by (level, ind, pos).
               // We loop over XL positions on the leading-edge plane, ignore those not common with
               // XL-1 rays, and choose those that actually hit the -Y side of the current octet.
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.y += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =       Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;    // still inside the current octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     // pos1 = initial coordinates at the leading-edge level, step forward to the Y sidewall
                     if (DIR.y>0.0f)   pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ;   // step based on    Y ****
                     else              pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;   // to 2.0  (level==RL+1)
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.y  =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in   octet        Y *****
#   endif
                     // Add the ray from the upstream direction to buffer as a new subray
                     //    we leave the buffer entry at level XL, if leafs are >XL, we drop to next 
                     //    refinement level only at the beginning of the main loop.
                     // Each sideray will actually continue in location refined to some level >=XL
                     // We will store NTRUE correct for XL. If sideray is in refined region, beginning of the
                     // main loop will drop the ray to the higher level and rescale NTRUE accordingly.
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level==XL
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL ===== offsets                              Z and Y
            // current ray at level level==RL+1 defined by (level, ind, pos)
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {     // neighbour not refined to level==XL, will not provide XL siderays
               for(int a=-1; a<=3; a++) {       // offset in +/- Z direction from the level RL ray, in XL steps
                  for(int b=-1; b<=3; b++) {    // ray offset in +/- Y direction, candidate relative to current ray
                     if ((a%2==0)&&(b%2==0)) continue ; // skip LR rays
                     // if we come this far, we will add the ray if it just hits the current octet with RL+1 cells
                     pos1     =  pos ;                                 // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =      Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;     // in current octet, will not hit sidewall
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // second offset           Y ***
                     // pos1 = initial coordinates at the leading edge plane, step to the Z sidewall
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in octet,        Z *****
#   endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL     ;      // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (((LEADING==2)&&(POS.y<EPS2))||((LEADING==3)&&(POS.y>TMEPS2))) {  // +/-Y leading edge
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.y/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // ***A*** even number of dx,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL) IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITTING THE X SIDEWALL ===== offsets                             X and Z
            pos1.y =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.x =  (DIR.x>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       X *
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos inthe octet,      X *****
#   endif
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL =====  offsets                             Z and X
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                 // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =     Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;     // still inside the current octet, ignore
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =     X ***
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in the octet,    Z *****
#   endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         
         if (((LEADING==4)&&(POS.z<EPS2))||((LEADING==5)&&(POS.z>TMEPS2))) {  // +/-Z leading edge
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.z/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // even number of dx,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITING THE X SIDEWALL ===== offsets                              X and Y
            pos1.y =  0.1f ;     pos1.z = 0.1f ;
            pos1.x =  (DIR.x>0.0f) ? (-0.1f) : (2.1f) ;    // upstream neighbour, main offset       X *
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;    // still inside the current   octet, ignore
                     pos1.y += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Y ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos in the octet,     X *****
#   endif
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     sr += 1 ;                     
                     
#   if (DEBUG>0)
                     printf("!!!A  ") ;
                     report(level, ind, XL, &pos, OFF, PAR, NBUF) ;
#   endif      
                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL =====  offsets                            Y and X
            pos1.x =  0.1f ;    pos1.z = 0.1f ;
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset       Y *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL 
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =    Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =    X ***
                     if (DIR.y>0.0f) pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ; // step based on      Y ****
                     else            pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#   if (DOUBLE_POS==0)
                     pos1.y   =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in the octet,   Y *****
#   endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         
         // should have space for 6 rays => sr cannot be >=6 !!
         // if (sr>=6) printf("sr = %d\n", sr) ;
         // sr = 0 ;
         
         
         
         
         
         if (sr>0) {  // added some siderays
            // While the current ray was at level OTL, there are no guarantees that the same
            // refinement exists for all added siderays. Therefore, we add all siderays
            // using the level XL coordinates. This also means that the siderays and the
            // leading-edge subrays (below) must be stored as separate BUFFER entries (different NBUF).
            BUFFER[c1+0] = XL ;        // at level XL all rays stored as level XL rays
            BUFFER[c1+1] = I2F(ind) ;  // index of *original ray*, at level XL
            
            // We leave the original ray as the current one == (OTL, OTI, POS), these unchanged.
#   if 0            
            if (ind>=24141184) {
               printf("--- SIDERAYS --- ind = %d  -------------------------------------------------------\n", ind) ;
               printf(" %9d  %2d %9d  P %9.7f %9.7f %9.7f   D %7.4f %7.4f %7.4f  RHO %.3e\n", 
                      OFF[OTL]+OTI, OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, RHO[OFF[OTL]+OTI]) ;
               report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
               printf("-----------------------------------------------------------------------------------\n") ;
               printf("0:") ; report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
            }
#   endif       
            
            
#   if (DEBUG>0)
            printf("!!! gid=%d ADDED =========== (%d, %d) %d\n", gid, XL, ind, OFF[XL]+ind) ;
            printf("================================================================================\n") ;
            for(int i=0; i<sr; i++) {   // skip sr=0 that is so far an empty slot
               otl       =  BUFFER[c1+0] ;
               oti       =  F2I(BUFFER[c1+1]) ;   // cell index for  the icoming ray
               pos.x     =  BUFFER[c1+2+4*i] ; pos.y =  BUFFER[c1+3+4*i] ; pos.z =  BUFFER[c1+4+4*i] ;
               sid       =  4*(int)floor(pos.z) + 2*(int)floor(pos.y) + (int)floor(pos.x) ; 
               oti       =  8*(int)(OTI/8) + sid ;   // cell index for the added ray
               otl_ray   =  BUFFER[c1+5+4*i] ;            
               printf("ADD.SIDE%d   main ray (%d,%d) %d   L: %8.4f %8.4f %8.4f --- NBUF =%d\n",
                      i, otl, oti, OFF[otl]+oti, pos.x, pos.y, pos.z, NBUF) ;
               // splitpos  =  BUFFER[c1+6+4*i] ;
               printf("ADD.SIDE%d ", i) ;
               report(otl, oti, otl_ray, &pos, OFF, PAR, NBUF) ;
               
            }
            printf("================================================================================\n") ;
#   endif
            
            // each NBUF slot = [ level, ind, { x, y, z, rl } ] = 2 + 6*4 = 26 floats   =>   sr = [0,5]
            for(int i=sr; i<6; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
            NBUF += 1 ;
            
            // if (NBUF>30) printf("ADD SIDERAYS  LEVEL %d -> %d     NBUF = %2d\n", OTL, XL, NBUF) ;
            
#   if (SAFE>0)
            if (NBUF>=MAX_NBUF) {
               printf("gid %7d ------  NBUF = %d !!!!!!!!!!!!!!!!!!!!!!\n", gid, NBUF) ;
               // NBUF -= 1 ;
            }
#   endif
            
            // When NTRUE is saved to buffer, the values must be stored rescaled to the level XL.
            // Thus, the stored values are   NTRUE(for_OTL) * 4.0**(OTL-XL)
            // When sideray is taken from buffer and is located in a region with OTL>XL, the
            // beginning of the main loop will again rescale with  4.0*(XL-OTL)
         }  // sr>0
         
      } // for XL -- adding possible siderays
      
      
      // Subrays are added only when we are at leaf level. Also, when subrays are added, 
      // we continue with the original RL ray and immediately make a step forward. Therefore
      // there is no risk that we would add siderays multiple times.
#  endif
      
      
      
#  if (SAFE>0)
      if (NBUF>=MAX_NBUF) {
         printf("ABORT --- NBUF>=MAX_NBUF !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n") ;
         TPL[gid] = -1.0e20 ; return ;    // one should probably abort, unless plweight is used
      }
#  endif
      
      
      
#  if (DEBUG>1)      
      printf("CONTINUE  ") ;
      report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
#  endif      
      
      
#  if 0      
      if (OTI>=24141183) {
         printf("--- CONTINUE ---------------------------------------------------------------------\n") ;
         printf(" %9d  %2d %9d  P %9.7f %9.7f %9.7f   D %7.4f %7.4f %7.4f  RHO %.3e\n", 
                OFF[OTL]+OTI, OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, RHO[OFF[OTL]+OTI]) ;
         report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
         printf("----------------------------------------------------------------------------------\n") ;
         printf("0:") ; report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
      }
#  endif
      
      // we come here only if we have a ray that is inside a leaf node => can do the step forward
      // if the step is to refined cell, (OTL, OTI) will refer to a cell that is not a leaf node
      // ==> the beginning of the present loop will have to deal with the step to higher refinement
      // level and the creation of three additional rays
      OTLO    =  OTL ;  c2 = OTI ;
      // get step but do not yet go to level > OTLO
          
#  if (DEBUG>0)
      pos1 = POS ;
      RootPos(&pos1, OTL, OTI, OFF, PAR) ;      
      c2 = OFF[OTL]+OTI ;
      printf("[%d] @ STEP    %9.6f %9.6f %9.6f   %d  ", gid, pos1.x, pos1.y, pos1.z, OTLO) ;      
#  endif
      
      
      dr      =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL, LEADING) ; // step [GL] == root grid units !!

#  if (NO_ATOMICS>0)
      PL[INDEX] += dr ;
#  else
      AADD(&(PL[INDEX]), dr) ;
#  endif
      tpl       += dr ;               // just the total value for current idir, ioff
      
      
      // The new index at the end of the step  --- OTL >= OTL  => INDEX may not refer to a leaf cell
      INDEX    =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;  // OTL, OTI, POS already updated = end of the current step
      if (INDEX>=0) {
         if (RHO[INDEX]<=0.0f) {  // we ended up in a parent cell
            continue ; // we moved to refined region, handled at the beginning of the main loop
         }
      }
      
      if (OTL<RL) {           // up in hierarchy and this ray is terminated -- cannot be on root grid
         // printf("TERMINATE\n") ;
         INDEX=-1 ;           // triggers the generation of a new ray below
      } else {     
         if (INDEX<0) {       // ray exits the root grid, possibly create a new OTL=0 ray on the other side
            // if (FOLLOW)   printf("*** LEADING %d   MIRROR FROM %8.4f %8.4f %8.4f ***\n", LEADING, POS.x, POS.y, POS.z) ;
            if (POS.x>=NX  ) {   if (LEADING!=0)   POS.x =    EPS ;   }
            if (POS.x<=ZERO) {   if (LEADING!=1)   POS.x = NX-EPS ;   }
            if (POS.y>=NY  ) {   if (LEADING!=2)   POS.y =    EPS ;   }
            if (POS.y<=ZERO) {   if (LEADING!=3)   POS.y = NY-EPS ;   } 
            if (POS.z>=NZ  ) {   if (LEADING!=4)   POS.z =    EPS ;   }            
            if (POS.z<=ZERO) {   if (LEADING!=5)   POS.z = NZ-EPS ;   }            
            IndexGR(&POS, &OTL, &OTI, RHO, OFF) ; // not necessarily a leaf!
            INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
            if (INDEX>=0) {    // new level-0 ray started on the opposite side
               RL = 0 ;    OTLO    = 0 ;   *count += 1 ;
#  if (DEBUG>0)
               printf("MIRROR    ") ;
               report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
#  endif
               continue ;
            }
         } // if INDEX<0
      }
      
      
      
      // [C] if INDEX still negative, current ray truly ended => take new ray from the buffer, if such exist
      //  NBUF>0 is no guarantee that rays exist in the buffer because we reserved sr=0 for the main ray
      //  and if only siderays were added, this NBUF entry has nothing for sr=0
      if ((INDEX<0)&&(NBUF>0))   {                // find a ray from BUFFER
         // printf("FROM BUFFER\n") ;
         c1    =  (NBUF-1)*(26+CHANNELS) ;  // 26+CHANNELS elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // level where sibling rays were created
         OTLO  =  OTL ;       
         OTI   =  F2I(BUFFER[c1+1]) ;       // OTI for the original ray that was split
         
         // if (OTI>=24141184) printf("ERROR IN OTI READ FROM BUFFER: %d  %.3f\n", OTI, BUFFER[c1+1]) ;
         
         // a maximum of 6 rays per buffer entry, the entry sr=0 was reserved for the main ray
         for(sr=5; sr>=0; sr--) {         
            dx    =  BUFFER[c1+2+4*sr] ;
            if (dx>-0.1f) break ;      // found a ray
         }
         // if (dx<ZERO) printf("??? BUFFER WITH NEGATIVE POS.x ???\n") ;
         POS.x     =  dx ;
         POS.y     =  BUFFER[c1+3+4*sr] ;
         POS.z     =  BUFFER[c1+4+4*sr] ;
         RL        =  BUFFER[c1+5+4*sr] ;    
         // we must have found a ray  -- figure out its OTI (index within level OTL)
         SID       =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
         OTI       =  8*(int)(OTI/8) + SID ;    // OTI in the buffer was for the original ray that was split
         INDEX     =  OFF[OTL]+OTI ;    // global index -- must be >=0 !!!
         
#  if 0
         if ((INDEX<0)||(INDEX>CELLS)) {
            printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n") ; 
            printf("WTF --- FROM BUFFER: INDEX %d [0,%d] SID %d   sr=%d ????\n",  INDEX, CELLS, SID, sr) ;
            printf(" %9d  %2d %9d  P %9.7f %9.7f %9.7f   D %7.4f %7.4f %7.4f  RHO %.3e\n", 
                   OFF[OTL]+OTI, OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, RHO[OFF[OTL]+OTI]) ;
            report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
            printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n") ; 
         }
#  endif
         
         // this stored ray is at level OTL but may be in a cell that is itself still further
         // refined => this will be handled at the start of the main loop => possible further
         // refinement before on the rays is stepped forward
         // However, if we are still at SPLITPOS, new siderays will not be possible before >=1 steps
         BUFFER[c1+2+4*sr] = -1.0f ;   // mark as used
         if (sr==0) {
            NBUF -= 1 ;        // was last of the <=6 subrays in this buffer entry
         } else {
            ; // printf("TAKE => NBUF SAME %2d\n", NBUF) ;
         }
         
         
#  if 0
         printf("---  %3d -------------------------------------------------------------------------------------------\n", steps) ;
         printf(" %9d  %2d %9d  P %9.7f %9.7f %9.7f   D %7.4f %7.4f %7.4f  RHO %.3e\n", 
                OFF[OTL]+OTI, OTL, OTI, POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, RHO[OFF[OTL]+OTI]) ;
         report(OTL, OTI, RL, &POS, OFF, PAR, NBUF) ;
         printf("----------------------------------------------------------------------------------------------------\n") ;
         FOLLOW = true ;           // start follow when first ray taken from the buffer
         if (FOLLOW) steps += 1 ;
         // if (steps>20) return ;
         if (steps>19) return ;
#  endif         
         
      }  // (INDEX<0)
      
      
   } // while INDEX>=0  --- stops when buffer is empty and the main level-0 ray has exited
   
   TPL[gid] += tpl ;   // TPL[NRAY]
}  // END OF PathsOT4


# endif  // PLWEIGHT>0









__kernel void UpdateOT4(  // @u
                          const int        gid0,    //  first gid in the index running over NRAY>=NWG
# if (PLWEIGHT>0)
                          __global float  *PL,      //  0
# endif
# if (WITH_HALF==1)
                          __global half  *CLOUD,    //  1 [CELLS,4] ==  vx, vy, vz, sigma
# else
                          __global float4 *CLOUD,   //  1 [CELLS]: vx, vy, vz, sigma
# endif
                          GAUSTORE  float *GAU,     //  2 precalculated gaussian profiles [GNO,CHANNELS]
                          constant int2   *LIM,     //  3 limits of ~zero profile function [GNO]
                          const float      Aul,     //  4 Einstein A(upper->lower)
                          const float      A_b,     //  5 (g_u/g_l)*B(upper->lower)
                          const float      GN,      //  6 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                          const float      APL,     //  7 average path length [GL]
                          const float      BG,      //  8 background value (photons)
                          const float      DIRWEI,  //  9 weight factor (based on current direction)
                          const float      EWEI,    // 10 weight 1/<1/cosT>/NDIR
                          const int        LEADING, //  0 leading edge
                          const REAL3      POS0,    // 11 initial position of id=0 ray
                          const float3     DIR,     // 12 ray direction
                          __global float  *NI,      // 13 [CELLS]:  NI[upper] + NB_NB
                          __global float  *RES,     // 14 [CELLS]: SIJ, ESC ---- or just [CELLS] for SIJ
                          __global float  *NTRUES,  // 15 [NWG*MAXCHN]   --- NWG>=simultaneous level 0 rays
# if (WITH_CRT>0)
                          constant float *CRT_TAU,  //  dust optical depth / GL
                          constant float *CRT_EMI,  //  dust emission photons/c/channel/H
# endif                     
# if (BRUTE_COOLING>0)
                          __global float   *COOL,   // 14,15 [CELLS] = cooling 
# endif
                          __global   int   *LCELLS, //  16
                          __constant int   *OFF,    //  17
                          __global   int   *PAR,    //  18
                          __global   float *RHO,    //  19  -- needed only to describe the hierarchy
                          __global   float *BUFFER_ALL  //  20 -- buffer to store split rays
                       )  {   
   // Each ***WORK GROUP*** processes one ray. The rays are two cells apart to avoid synchronisation
   // problems. Rays start on the leading edge. If ray exits through a side (wrt axis closest to
   // direction of propagation), a new one is created on the opposite side and the ray ends when the
   // downstream edge is reached.
   // 
   // As one goes to a higher hierarchy level (into a refined cell), one pushes the original and two
   // new rays to buffer. The fourth ray (one of the new ones) is followed first. When ray goes up in
   // hierarchy (to a larger cell) ray is terminated if ray was created at a higher level. When one
   // ray is terminated, the next one is taken from the buffer, if that is not empty. When ray goes up
   // in hierarchy, NTRUE is scaled *= 4 or, if one goes up several levels, *=4^n, where n is the
   // decrease in the hierarchy level.
   int id  = get_global_id(0), lid = get_local_id(0), gid = get_group_id(0), ls  = get_local_size(0) ;
   __global float *BUFFER = &BUFFER_ALL[gid*(26+CHANNELS)*MAX_NBUF] ;  // here gid ~ NWG
   
   
   if (gid>=NWG)  return ;
   gid += gid0 ;  // becomes running index over NRAY ....           here gid ~ NRAY
   if (gid>=NRAY) return ;        // one work group per ray .... NWG==NRAY
   
   GAUSTORE float *profile ;
   __local  float  NTRUE[CHANNELS] ;
   __local  int2   LINT ;          // SID, NBUF
   __local  float  SDT[LOCAL] ;    // per-workitem sum_delta_true
   __local  float  AE[LOCAL] ;     // per-workitem all_escaped
   float weight, w, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed, sum_delta_true, all_escaped, nu ;
   int row, shift, INDEX, c1, c2, OTL, OTI, OTLO, XL, RL, SID, NBUF=0, sr, sid, level, b1, b2, I, i, ind, otl, oti ;
   int level1, ind1 ;
   REAL3  POS, pos0, pos1, pos, RDIR ;
   REAL   dx, dy, dz, s ;
   float dr ;
   float flo ;
   
   
# if 0
   printf("GID %9d\n", gid) ;
   if (gid0!=21) return ;
# endif
   
   
   
# if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, pro, sij ;
# endif  
# if (ONESHOT<1)
   int nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ;
# endif
   RDIR.x = DIR.x ; RDIR.y = DIR.y ; RDIR.z = DIR.z ;
   
   int *SUBS ;
   POS.x = POS0.x ;   POS.y = POS0.y ;   POS.z = POS0.z ;
   
   // when split done only on the leading edge -- HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
# if (ONESHOT<1)
   if (LEADING<3) {
      if (LEADING==0) { 
         HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;
         POS.x =    EPS ;  POS.y += TWO*(gid%ny) ;  POS.z += TWO*(int)(gid/ny) ;  // gid is index running over all NRAY
      }
      if (LEADING==1) { 
         HEAD[0] = 1 ;   HEAD[1] = 3 ;   HEAD[2] = 5 ;  HEAD[3] = 7 ;
         POS.x = NX-EPS ;  POS.y += TWO*(gid%ny) ;  POS.z += TWO*(int)(gid/ny) ;
      }
      if (LEADING==2) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;
         POS.y =    EPS ;  POS.x += TWO*(gid%nx) ;  POS.z += TWO*(int)(gid/nx) ;
      }
   } else {
      if (LEADING==3) { 
         HEAD[0] = 2 ;   HEAD[1] = 3 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.y = NY-EPS ;  POS.x += TWO*(gid%nx) ;  POS.z += TWO*(int)(gid/nx) ;
      }
      if (LEADING==4) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;
         POS.z =    EPS ;  POS.x += TWO*(gid%nx) ;  POS.y += TWO*(int)(gid/nx) ;
      }
      if (LEADING==5) { 
         HEAD[0] = 4 ;   HEAD[1] = 5 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.z = NZ-EPS ;  POS.x += TWO*(gid%nx) ;  POS.y += TWO*(int)(gid/nx) ;
      }
   }
# else
   if (LEADING<3) {
      if (LEADING==0) { 
         HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;
         POS.x =    EPS ;  POS.y += gid%NY ;  POS.z += gid/NY ;  // gid is index running over all NRAY
      }
      if (LEADING==1) { 
         HEAD[0] = 1 ;   HEAD[1] = 3 ;   HEAD[2] = 5 ;  HEAD[3] = 7 ;
         POS.x = NX-EPS ;  POS.y += gid%NY ;  POS.z += gid/NY ;
      }
      if (LEADING==2) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;
         POS.y =    EPS ;  POS.x += gid%NX ;  POS.z += gid/NX ;
      }
   } else {
      if (LEADING==3) { 
         HEAD[0] = 2 ;   HEAD[1] = 3 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.y = NY-EPS ;  POS.x += gid%NX ;  POS.z += gid/NX ;
      }
      if (LEADING==4) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;
         POS.z =    EPS ;  POS.x += gid%NX ;  POS.y += gid/NX ;
      }
      if (LEADING==5) { 
         HEAD[0] = 4 ;   HEAD[1] = 5 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.z = NZ-EPS ;  POS.x += gid%NX ;  POS.y += gid/NX ;
      }
   }
# endif
   
   
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;     // remain at root level, not yet going to leaf
   if (OTI<0) return ;
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
   OTLO      =  OTL ;
   RL        =  0 ;
   for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BG * DIRWEI ;
   
   
   while(INDEX>=0) {  // INDEX may refer to a cell that is not a leaf
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
      // If we are not in a leaf, we have gone to some higher level. 
      // Go one level higher, add >=three rays, pick one of them as the current,
      // and return to the beginning of the loop.
      if (RHO[INDEX]<=0.0f) {                    // go to the sub-cell and add sibling rays
         
         sr     =  0 ;         
         NBUF   =  min(NBUF, MAX_NBUF-1) ;  // BUFFER IS ALREADY FULL ...??? replace last one ???
         c1     =  NBUF*(26+CHANNELS) ;
         SID    =  -1 ;
         
         POS.x  =  TWO*fmod(POS.x, ONE) ;      // coordinate inside parent cell [0,1]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         flo    =  -RHO[INDEX] ;                 // OTL, OTI of the parent cell
         OTL   +=  1  ;                          // step to next level = refined level
         OTI    =  *(int *)&flo ;                // OTI for the first child in octet
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // original subcell
         OTI   +=  SID;                          // cell in octet, original ray
         c2     =  0 ;   // set c2=1 if ray is to be split (is this the leading cell edge?)
         if (LEADING<3) {
            if ((LEADING==0)&&(POS.x<  EPS2)) c2 = 1 ; 
            if ((LEADING==1)&&(POS.x>TMEPS2)) c2 = 1 ; 
            if ((LEADING==2)&&(POS.y<  EPS2)) c2 = 1 ;
         } else {
            if ((LEADING==3)&&(POS.y>TMEPS2)) c2 = 1 ; 
            if ((LEADING==4)&&(POS.z<  EPS2)) c2 = 1 ;
            if ((LEADING==5)&&(POS.z>TMEPS2)) c2 = 1 ;
         }
         // @@  rescale always when the resolution changes
         for(int i=lid; i<CHANNELS; i+=ls) {   // SCALE ON EVERY REFINEMENT EVEN WHEN NOT SPLIT
            NTRUE[i] *= 0.25f ; 
         }
         barrier(CLK_LOCAL_MEM_FENCE) ;
         
         if (c2>0) {  // split the ray and choose one of the new ones to be followed first
            for(int i=lid; i<CHANNELS; i+=ls) {   // ray effectively split to four
               BUFFER[c1+26+i] = NTRUE[i] ;
            }
            barrier(CLK_LOCAL_MEM_FENCE) ;            
            if (lid==0) {
               BUFFER[c1+0]  =  OTL ;                     // level where the split is done, OTL>this =>
               BUFFER[c1+1]  =  I2F(OTI) ;                // buffer contains the OTI of the original ray 
               BUFFER[c1+2]  =  POS.x ;                   // Store the original MAIN RAY to buffer as the first one
               BUFFER[c1+3]  =  POS.y ;
               BUFFER[c1+4]  =  POS.z ;
               BUFFER[c1+5]  =  RL  ;                     // main ray exists at levels >= RL
            }            
            // Two new subrays added to buffer, third one becomes the current ray
            // Add first two subrays to buffer
            if (                  HEAD[0]==SID) {  // 0=original => (1,2) to buffer, 3 as current
               if (lid==0) {
                  sid              = HEAD[1] ;
                  BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                  BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                  BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                  sid              = HEAD[2] ;
                  BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                  BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                  BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
               }
               SID              = HEAD[3] ;  // SID of the subray to be followed first
            } else {
               if (                  HEAD[1]==SID) {
                  if (lid==0) {
                     sid              = HEAD[2] ;
                     BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                     BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                     BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                     sid              = HEAD[3] ;
                     BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                     BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                     BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                  }
                  SID              = HEAD[0] ;
               } else {
                  if (                  HEAD[2]==SID) {
                     if (lid==0) {
                        sid              = HEAD[3] ;
                        BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                        BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                        BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                        sid              = HEAD[0] ;
                        BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                        BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                        BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                     }
                     SID              = HEAD[1] ;
                  } else {
                     if (                  HEAD[3]==SID) {
                        if (lid==0) {
                           sid              = HEAD[0] ;
                           BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                           BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                           BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                           sid              = HEAD[1] ;
                           BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                           BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                           BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                        }
                        SID              = HEAD[2] ;
                     } else {
                        ; // printf("???\n") ;
                     }
                  }
               }
            } 
            // for the two subrays just added, update RL, SPLIT
            sr = 3 ;   // so far the original main ray and two split subrays
            if (lid==0) {
               BUFFER[c1+5+4*1]  =  OTL ;    BUFFER[c1+5+4*2]  =  OTL ;  // RL = current OTL
               for(int i=sr; i<6; i++)  BUFFER[c1+2+4*i] = -99.0f ;      // mark the rest as unused
            } // lid==0
            // We added leading edge rays, old main ray is in buffer, SID refers to one of the subrays
            // update OTI and POS to correspond to that subray
            OTI       =  (*(int *)&flo)  + SID ;    // the new ray to be followed, OTI = index of first subcell + SID
            POS.x     =  fmod(POS.x,ONE) + (int)( SID%2)    ;  // dx and SID known to all work items
            POS.y     =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
            POS.z     =  fmod(POS.z,ONE) + (int)( SID/4)    ;
            RL        =  OTL ;  // when we reach OTL<RL, this ray will be terminated
            NBUF++ ;            // >= 1 subrays were added, NBUF increases just by one
            
         } // c2>0  == we split rays on the leading edge
         
      } // RHO<0
      
      
      INDEX  = OFF[OTL] + OTI ;
      if (RHO[INDEX]<=0.0f) continue ;  // not a leaf, go back to the beginning of the main loop
      
      
      
      
# if 1  // adding siderays
      
      // It is essential that we are already in a leaf node when siderays are added:
      // once ray has produced siderays at the current location, it will immediately take a 
      // real step forward.
      
      
      
      // Current ray is still (OTL, OTI, POS)
      // presumably current NTRUE is correct for ray at level OTL
      
      // Siderays are based on rays hitting the leading edge. We have stepped down to leaf level.
      // Therefore, incoming ray has already been split to four, and NTRUE has been scaled by 0.25 to
      // correspond to the step from OTL-1 to OTL.
      
      
      for(XL=OTL; XL>RL; XL--) {   // try to add subrays at levels   RL<XL<=OTL
         
         
         if (XL==OTL) {  // calculate only if we actually come to the loop
#  if (SAFE>0)
            if (NBUF>=MAX_NBUF) {
               // printf("*** BUFFER FULL -- CANNOT CHECK SIDERAYS\n") ; 
               break ;
            }
#  endif
            // we use root coordinates to determine whether RL ray is also RL-1 ray
            // the required accuracy is only 0.5*0.5**MAXL, float32 works at least to MAXL=15
            pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;   // current position in root coordinates
         }
         
         c1     =  NBUF*(26+CHANNELS) ;          // 26+CHANNELS elements per buffer entry
         sr     =  0  ;
         
         if (((LEADING==0)&&(POS.x<EPS2))||((LEADING==1)&&(POS.x>TMEPS2))) {  // +/-X leading edge, at OTL level
            dr   = pown(0.5f, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.x/dr)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test smaller XL (larger dr) values
            // even number of dr,is therefore a border between level XL octets
            // calculate (pos, level, ind) for the position at level XL  (in the octet that we are in)
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // (pos, level, ind) now define the position at level XL
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL ===== offsets                              Y and Z
            // check XL-scale neighbour
            pos1.x =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       Y * 
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level,NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               // Current ray position at level==XL is defined by (level, ind, pos).
               // We loop over XL positions on the leading-edge plane, ignore those not common with
               // XL-1 rays, and choose those that actually hit the -Y side of the current octet.
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.y += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =       Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;     // still inside the current octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     // pos1 = initial coordinates at the leading-edge level, step forward to the Y sidewall
                     if (DIR.y>0.0f)   pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ;  // step based on    Y ****
                     else              pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;  // to 2.0  (level==RL+1)
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.y  =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in   octet        Y *****
#  endif
                     // Add the ray from the upstream direction to buffer as a new subray
                     //    we leave the buffer entry at level XL, if leafs are >XL, we drop to next 
                     //    refinement level only at the beginning of the main loop.
                     // Each sideray will actually continue in location refined to some level >=XL
                     // We will store NTRUE correct for XL. If sideray is in refined region, beginning of the
                     // main loop will drop the ray to the higher level and rescale NTRUE accordingly.
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                        BUFFER[c1+3+4*sr] =  pos1.y ;
                        BUFFER[c1+4+4*sr] =  pos1.z ;
                        BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level==XL
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL ===== offsets                              Z and Y
            // current ray at level level==RL+1 defined by (level, ind, pos)
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {     // neighbour not refined to level==XL, will not provide XL siderays
               for(int a=-1; a<=3; a++) {       // offset in +/- Z direction from the level RL ray, in XL steps
                  for(int b=-1; b<=3; b++) {    // ray offset in +/- Y direction, candidate relative to current ray
                     if ((a%2==0)&&(b%2==0)) continue ; // skip LR rays
                     // if we come this far, we will add the ray if it just hits the current octet with RL+1 cells
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =      Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // in current octet, will not hit sidewall
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // second offset           Y ***
                     // pos1 = initial coordinates at the leading edge plane, step to the Z sidewall
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in octet,        Z *****
#  endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                        BUFFER[c1+3+4*sr] = pos1.y ;
                        BUFFER[c1+4+4*sr] = pos1.z ;
                        BUFFER[c1+5+4*sr] = XL     ;      // ray created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (((LEADING==2)&&(POS.y<EPS2))||((LEADING==3)&&(POS.y>TMEPS2))) {  // +/-Y leading edge
            dr   = pown(0.5f, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.y/dr)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // ***A*** even number of dr,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL) IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITTING THE X SIDEWALL ===== offsets                             X and Z
            pos1.y =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.x =  (DIR.x>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       X *
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos inthe octet,      X *****
#  endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                        BUFFER[c1+3+4*sr] =  pos1.y ;
                        BUFFER[c1+4+4*sr] =  pos1.z ;
                        BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL =====  offsets                             Z and X
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =     Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =     X ***
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in the octet,    Z *****
#  endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                        BUFFER[c1+3+4*sr] = pos1.y ;
                        BUFFER[c1+4+4*sr] = pos1.z ;
                        BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         
         if (((LEADING==4)&&(POS.z<EPS2))||((LEADING==5)&&(POS.z>TMEPS2))) {  // +/-Z leading edge
            dr   = pown(0.5f, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.z/dr)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // even number of dr,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITING THE X SIDEWALL ===== offsets                              X and Y
            pos1.y =  0.1f ;     pos1.z = 0.1f ;
            pos1.x =  (DIR.x>0.0f) ? (-0.1f) : (2.1f) ;    // upstream neighbour, main offset       X *
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.y += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Y ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos in the octet,     X *****
#  endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                        BUFFER[c1+3+4*sr] =  pos1.y ;
                        BUFFER[c1+4+4*sr] =  pos1.z ;
                        BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     }
                     sr += 1 ;
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL =====  offsets                            Y and X
            pos1.x =  0.1f ;    pos1.z = 0.1f ;
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset       Y *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL 
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =    Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =    X ***
                     if (DIR.y>0.0f) pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ; // step based on      Y ****
                     else            pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.y   =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in the octet,   Y *****
#  endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                        BUFFER[c1+3+4*sr] = pos1.y ;
                        BUFFER[c1+4+4*sr] = pos1.z ;
                        BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (sr>0) {  // added some siderays
            // While the current ray was at level OTL, there are no guarantees that the same
            // refinement exists for all added siderays. Therefore, we add all siderays
            // using the level XL coordinates. This also means that the siderays and the
            // leading-edge subrays (below) must be stored as separate BUFFER entries (different NBUF).
            if (lid==0){
               BUFFER[c1+0] = XL ;          // at level XL all rays stored as level XL rays
               BUFFER[c1+1] = I2F(ind) ;    // index of *original ray*, at level XL
               // We leave the original ray as the current one == (OTL, OTI, POS) remain unchanged.
               for(int i=sr; i<6; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
            }
            // NTRUE is correct for OTL but we store data for level XL
            // Therefore, BUFFER will contain  NTRUE * 0.25**(XL-OTL)  == NTRUE * 4**(OTL-XL)
            // ==>  the BUFFER values are   NTRUE(correct for RL) * 4.0**(OTL-XL), OTL and not RL !!!!
            // When sideray is taken from buffer and if it is located in a region with OTL>XL, the
            // beginning of the main loop will again rescale with  4.0*(XL-OTL)
            dr  =  pown(4.0f, OTL-XL) ;          // was RL-XL but surely must be OTL-XL !!!!!
            for(int i=lid; i<CHANNELS; i+=ls) {
               BUFFER[c1+26+i] = NTRUE[i]*dr ;   // NTRUE scaled from RL to XL 
            }            
            NBUF += 1 ;            
#  if 0
            if (NBUF>=MAX_NBUF) {
               printf("update => BUFFER FULL !!!\n") ;
            }
#  endif
         }
         
      } // for XL -- adding possible siderays
      
      
      // Subrays are added only when we are at leaf level. Also, when subrays are added, 
      // we continue with the original RL ray and immediately make a step forward. Therefore
      // there is no risk that we would add siderays multiple times.
# endif // adding siderays
      
      
      
      
      // global index -- must be now >=0 since we started with while(INDEX>=0) and just possibly went down a level
      INDEX = OFF[OTL]+OTI ;   
      
      // if not yet a leaf, jump back to start of the loop => another step down
      if (RHO[INDEX]<=0.0f) {
         continue ;
      }
      
      // we are now in a leaf node, ready to make the step
      OTLO   =  OTL ;
      
      
      
      dr     =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL   , LEADING) ;  // POS is now the end of this step
      
      
      
# if 0   // @@ double check PL calculation ... PL[:] should be reduced to zero
      if (lid==0)  AADD(&(PL[INDEX]), -dr) ;   // must not have SKIP_PL !!
# endif
      
      
      
      
      
      
      
      
      // if (RHO[INDEX]>CLIP) { // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // one workgroup per ray => can have barriers inside the condition
                  
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
# if 0
      barrier(CLK_LOCAL_MEM_FENCE) ;
      barrier(CLK_GLOBAL_MEM_FENCE) ;
      for(int i=lid; i<LOCAL; i+=LOCAL) {
         printf("[%3d]  %2d %6d   %8.4f %8.4f %8.4f   %2d\n", lid, OTL, OTI, POS.x, POS.y, POS.z, NBUF) ;
      }
# endif
      
      
      // with path length already being the same in all cells !   V /=8,  rays*4==length/2 /=2 =>  2/8=1/4
      weight    =  (dr/APL) *  VOLUME  *  pown(0.25f, OTLO) ;  // OTL=1 => dr/APL=0.5
      
      // INDEX is still the index for the cell where the step starts
      nu        =  NI[2*INDEX] ;
# if 0
      nb_nb     =  NI[2*INDEX+1] ;
# else
      nb_nb     =  max(1.0e-30f, NI[2*INDEX+1]) ; // $$$ KILL MASERS 
# endif
      
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
# if (WITH_HALF==0)
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# else
      doppler   =  
        vload_half(0, &(CLOUD[4*INDEX+0]))*DIR.x +
        vload_half(0, &(CLOUD[4*INDEX+1]))*DIR.y + 
        vload_half(0, &(CLOUD[4*INDEX+2]))*DIR.z ;
# endif
      
# if 0 // did not solve the oscillating-cell problem
      // ... one must avoid division by zero but lets move that to host side
      if (fabs(nb_nb)<1.0e-25f)  nb_nb   = 1.0e-25f ;
# endif        
      
      
      // tmp_tau   =  max(1.0e-35f, dr*nb_nb*GN) ;
      tmp_tau   =  dr*nb_nb*GN ;
      tmp_emit  =  weight * nu*Aul / tmp_tau ;  // GN include grid length [cm]
      shift     =  round(doppler/WIDTH) ;
# if (WITH_HALF==1)      //               sigma = 0.002f * w,   lookup table: sigma = SIGMA0 * SIGMAX^row
      // row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
      row       =  clamp((int)round(log(vload_half(0,&(CLOUD[4*INDEX+3]))/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
# else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
# endif
      profile   =  &GAU[row*CHANNELS] ;
      // avoid profile function outside profile channels LIM.x, LIM.y
      c1        =  max(LIM[row].x+shift, max(0, shift)) ;
      c2        =  min(LIM[row].y+shift, min(CHANNELS-1, CHANNELS-1+shift)) ;
      sum_delta_true = 0.0f ;
      all_escaped    = 0.0f ;
      
      
# if (WITH_CRT>0) // WITH_CRT
      sij = 0.0f ;
      // Dust optical depth and emission
      //   here escape = line photon exiting the cell + line photons absorbed by dust
      Ctau      =  dr     * CRT_TAU[INDEX] ;
      Cemit     =  weight * CRT_EMI[INDEX] ;      
      for(int i=c1; i<=c2; i++)  {
         pro    =  profile[i-shift] ;
         Ltau   =  tmp_tau*pro ;
         Ttau   =  Ctau + Ltau ;
         // tt     =  (1.0f-exp(-Ttau)) / Ttau ;
         tt     =  (fabs(Ttau)>0.01f) ?  ((1.0f-exp(-Ttau))/Ttau) : (1.0f-Ttau*(0.5f-0.166666667f*Ttau)) ;
         // ttt    = (1.0f-tt)/Ttau
         ttt    =  (1.0f-tt)/Ttau ;
         // Line emission leaving the cell   --- GL in profile
         Lleave =  weight*nu*Aul*pro * tt ;
         // Dust emission leaving the cell 
         Dleave =  Cemit *                     tt ;
         // SIJ updates, first incoming photons then absorbed dust emission
         sij   +=  A_b * pro*GN*dr * NTRUE[i]*tt ;
         // sij         += A_b * profile[i]*GN*Cemit*dr*(1.0f-tt)/Ttau ; // GN includes GL!
         sij   +=  A_b * pro*GN*dr * Cemit*ttt ;    // GN includes GL!
         // "Escaping" line photons = absorbed by dust or leave the cell
         all_escaped +=  Lleave  +  weight*nu*Aul*pro * Ctau * ttt ;
         // Total change of photons in the package
         NTRUE[i]     =  NTRUE[i]*exp(-Ttau) + Dleave + Lleave ;
      }  // loop over channels
      
#  if (NO_ATOMICS>0)
      RES[2*INDEX]    += sij ;            // division by VOLUME done in the solver (kernel)
      RES[2*INDEX+1]  += all_escaped ;    // divided by VOLUME only oin Solve() !!!
#  else
      AADD(&(RES[2*INDEX]), sij) ;
      AADD(&(RES[2*INDEX+1]), all_escaped) ;
#  endif
      
      
# else   // not  WITH_CRT ***************************************************************************************
      
      
      // because of c1, the same NTRUE elements may be updated each time by different work items...
      barrier(CLK_LOCAL_MEM_FENCE) ;    // local NTRUE elements possibly updated by different threads
      
      
      
      
#  if (WITH_ALI>0) // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      // if (fabs(nb_nb)>1.0e-30f) {  // $$$
      for(int i=c1+lid; i<=c2; i+=ls)  {
         w               =  tmp_tau*profile[i-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         // absence affects strongly low-density regions
#   if 0
         factor          =  clamp(factor, -1.0e-12f, 1.0f) ;  // $$$
#   else
         factor          =  clamp(factor,  1.0e-30f, 1.0f) ;  // KILL MASERS $$$
#   endif
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;    // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      SDT[lid] = sum_delta_true ;  AE[lid] = all_escaped ; // all work items save their own results
      barrier(CLK_LOCAL_MEM_FENCE) ;             // all agree on NTRUE, all know SDT and AE
      if (lid==0) {                              // lid=0 sums up and saves absorptions and escaped photons
         for(int i=1; i<LOCAL; i++) {  
            sum_delta_true += SDT[i] ;      all_escaped    +=  AE[i] ;     
         }
         all_escaped     =  clamp(all_escaped, 0.0001f*weight*nu*Aul, 0.9999f*weight*nu*Aul) ; // must be [0,1]
         // RES[2*INDEX]   +=  A_b * (sum_delta_true / nb_nb) ;
         // RES[2*INDEX+1] +=  all_escaped ;
         w  =  A_b * (sum_delta_true/nb_nb) ;
#   if (NO_ATOMICS>0)
         RES[2*INDEX]    +=  w ;
         RES[2*INDEX+1]  +=  all_escaped ;
#   else
         AADD(&(RES[2*INDEX]),    w) ;
         AADD(&(RES[2*INDEX+1]),  all_escaped) ;
#   endif
      } // lid==0
      // }
      
#  else // else no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
      
      // if (isfinite(tmp_tau)==0) printf("tmp_tau not finite !!!\n") ;
      
      for(int i=c1+lid; i<=c2; i+=ls)  {
         w               =  tmp_tau*profile[i-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
#   if 0
         factor          =  clamp(factor, -1.0e-12f, 1.0f) ;  // $$$
#   else
         factor          =  clamp(factor,  1.0e-30f, 1.0f) ;  // KILL MASERS $$$
#   endif
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;    // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed - escape ;  // later sum_delta_true +=  W*nu*Aul
      }   // over channels
      SDT[lid] = sum_delta_true ;
      barrier(CLK_LOCAL_MEM_FENCE) ;
      if (lid==0) {
         for(int i=1; i<LOCAL; i++)  sum_delta_true += SDT[i] ;    
         w  =   A_b  * ((weight*nu*Aul + sum_delta_true) / nb_nb)  ;
#   if (NO_ATOMICS>0)         
         RES[INDEX] += w ;
#   else
         // AADD(&(RES[INDEX]), w) ;
         AADD((__global float*)(RES+INDEX), w) ;
#   endif
      } 
#  endif // no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
# endif  // WITH OR WITHOUT CRT
      
      
      // } // RHO>CLIP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      
      
      
      
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
# if (BRUTE_COOLING>0)
      // total number of photons in the package as it exits the cell
      if (lid==0) {
         float cool = 0.0f ;
         for(int i=0; i<CHANNELS; i++) cool += NTRUE[i] ;
         COOL[INDEX] += cool ; // cooling of cell INDEX --- each work group distinct rays => no need for atomics
      }
# endif
      
      
      
      
      
      
      
      
      
      // Updates at the end of the step, POS has been already updated, OTL and OTI point to the new cell
      INDEX   =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;
      
# if (BRUTE_COOLING>0)  // heating of the next cell, once INDEX has been updated
      if (INDEX>=0) {
         if (lid==0)  COOL[INDEX] -= cool ; // heating of the next cell
      }
# endif
      
      if (INDEX>=0) {
         if (RHO[INDEX]<=0.0f) { // we stepped to a refined cell (GetStep goes only to a cell OTL<=OTLO)
            continue ;           // step down one level at the beginning of the main loop
         }
      }
      
      
      if (OTL<RL) {        // we are up to a level where this ray no longer exists
         INDEX=-1 ;        
      } else {      
         if (INDEX<0) {    // ray exits the cloud... possibly continues on the other side
            if (POS.x>=NX  ) {   if (LEADING!=0)  POS.x =    EPS ;   }
            if (POS.x<=ZERO) {   if (LEADING!=1)  POS.x = NX-EPS ;   } 
            if (POS.y>=NY  ) {   if (LEADING!=2)  POS.y =    EPS ;   }
            if (POS.y<=ZERO) {   if (LEADING!=3)  POS.y = NY-EPS ;   } 
            if (POS.z>=NZ  ) {   if (LEADING!=4)  POS.z =    EPS ;   }
            if (POS.z<=ZERO) {   if (LEADING!=5)  POS.z = NZ-EPS ;   } 
            IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;  // we remain in a root-grid cell => OTL==0 !
            INDEX =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
            if (INDEX>=0) {   // new level-0 ray started on the opposite side (may be a parent cell)
               RL = 0 ;  OTLO = 0 ; 
               // we had already barrier after the previous NTRUE update
               for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BG * DIRWEI ;
               barrier(CLK_LOCAL_MEM_FENCE) ;
# if (BRUTE_COOLING>0)
               if (lid==0) {
                  dr = BG*DIRWEI*CHANNELS ;  COOL[INDEX] -= dr ; // heating of the entered cell
               }
# endif
               continue ;
            }
         } // if INDEX<0
      }
      
      
      // rescale on every change of resolution
      if ((INDEX>=0)&&(OTL<OTLO)) {   // @s ray continues at a lower hierarchy level => NTRUE may have to be scaled
         dr = pown(4.0f, OTLO-OTL) ;  // scale on every change of resolution
         for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] *= dr ;     
         continue ;  // we went to lower level => this cell is a leaf
      }
      
      
      // if INDEX still negative, try to take a new ray from the buffer
      // 0   1     2  3  4  6     ...       NTRUE[CHANNELS] 
      // OTL OTI   x  y  z  RL    x y z RL                  
      if ((INDEX<0)&&(NBUF>0)) {            // NBUF>0 => at least one ray exists in the buffer
         barrier(CLK_GLOBAL_MEM_FENCE) ;    // all work items access BUFFER
         c1    =  (NBUF-1)*(26+CHANNELS) ;  // 8+CHANNELS elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // OTL ...
         OTLO  =  OTL ;                     // ???
         OTI   =  F2I(BUFFER[c1+1]) ;       // and OTI of the ray that was split
         for(sr=5; sr>=0; sr--) {         
            dr    =  BUFFER[c1+2+4*sr] ;    // read dr
            if (dr>-0.1f) break ;           // found a ray
         }
         POS.x   =  dr ;
         POS.y   =  BUFFER[c1+3+4*sr] ;  
         POS.z   =  BUFFER[c1+4+4*sr] ;
         RL      =  BUFFER[c1+5+4*sr] ;
         SID     =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
         OTI     =  8*(int)(OTI/8) + SID ;        // OTI in the buffer was for the original ray that was split
         INDEX   =  OFF[OTL]+OTI ;                // global index -- must be >=0 !!!
         // copy NTRUE --- values in BUFFER have already been divided by four
         barrier(CLK_GLOBAL_MEM_FENCE) ;          // all have read BUFFER
         if (lid==0)  BUFFER[c1+2+4*sr] = -1.0f ; // mark as used
         if (sr==0)   NBUF -= 1 ;
         c1     +=  26 ;
         for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BUFFER[c1+i] ;  // NTRUE correct for level OTL
         barrier(CLK_LOCAL_MEM_FENCE) ;         
         // note - this ray be inside a parent cell => handled at the beginnign of the main loop
      } // (INDEX<=0)&(NBUF>0)
      
      
   } // while INDEX>=0
   
   
   
# if 0
   printf("GID %9d DONE\n", gid) ;
# endif
   
   
   
}  // end of UpdateOT4()





#endif // WITH_OCTREE==4

















#if (WITH_OCTREE==5)



"""
This was an attempt to rewrite ray tracing without double precision.
Did not work yet but was getting complex and did not seem to be much
faster than OCTREE4, the one using double precision.
One should probably concentrate on optimising OCTREE4 instead.
"""







# if (DEBUG>0)
void report(const int       OTL, 
            const int       OTI,
            const int       OTL_RAY, 
            REAL3          *POS,
            __constant int *OFF, 
            __global int   *PAR) {
   REAL3  pos = *POS ;
   RootPos(&pos, OTL, OTI, OFF, PAR) ;
   printf(" (%2d,%2d) %4d    L: %6.2f %6.2f %6.2f      R: %6.2f %6.2f %6.2f      OTL=%d OTL_RAY=%d\n",
          OTL, OTI, OFF[OTL]+OTI, POS->x, POS->y, POS->z, pos.x, pos.y, pos.z, OTL, OTL_RAY) ;
# endif
   
}






__kernel void PathsOT5(  // 
                         const    int      gid0,    // do in bacthes, now starting with gid==gid0
                         __global float   *PL,      // [CELLS]
                         __global float   *TPL,     // [NRAY]
                         __global int     *COUNT,   // [NRAY] total number of rays entering the cloud
                         const    int      LEADING, // leading edge
                         const    REAL3    POS0,    // initial position of ray 0
                         const    float3   DIR,     // direction of the rays
                         __global   int   *LCELLS,   
                         __constant int   *OFF,
                         __global   int   *PAR,
                         __global float   *RHO,
                         __global float   *BUFFER_ALL
                      ) {   
   // OT4 = split rays at the leading edge and sides, should give identical path length for every cell
   // Each work item processes one ray, rays are two cells apart to avoid synchronisation
   // Rays start on the leading edge, if ray exits through a side, a new one is created 
   // on the opposite side. Ray ends when the downstream edge is reached
   
   // This version tries to make sure we make all the rays so that there is no variation in the
   // path lengths between the cells (avoids the need for PL weighting!!)
   // When entering a refined cell with level0, split the ray to four on the leading edge, like in PathsOT2
   // Regarding additional rays hitting the sides of the refined cell, check the two upstream neighbours.
   // If the neighbour is:
   //   - at level >= level0, do nothing -- refined rays will come from the neighbour
   //   - otherwise, test 4 refined rays that might hit the current refined cell from that side
   //     skip the rays that will come otherwise at some point from the neighbour
   int  ls = get_local_size(0),  gid = get_group_id(0) ;
   // things indexed with the current real group id
   __global float *BUFFER = &(BUFFER_ALL[gid*(26+CHANNELS)*MAX_NBUF]) ;  // gid ~ NWG 
   gid += gid0 ;        // index running over all NRAY   ......       gid ~ NRAY
   if  (gid>=NRAY) return ;
   if  (get_local_id(0)!=0)    return ;     // a single work item from each work group !!
   __global int *count  = &COUNT[gid] ;     // COUNT[NRAY]
   *count = 0 ;        
   
   int  INDEX, SID, NBUF=0, nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2, sr, sid, level, b1, XL, ind ;
   int  OTL, OTI, OTLO, RL, c1, c2, i, oti, otl, otl_ray, ae, be, level1, ind1 ;
   float  dr ;
   float flo, tpl=0.0f ;
   
   REAL  dx ;
   REAL3 POS, pos, pos0, pos1, RDIR ;
   POS.x  = POS0.x ; POS.y  = POS0.y ; POS.z  = POS0.z ;
   RDIR.x = DIR.x ;  RDIR.y = DIR.y ;  RDIR.z = DIR.z ;
   
   bool drop ;
   
   // HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
   if (LEADING<2){ 
      if (LEADING==0) {
         HEAD[0] = 0 ;          HEAD[1] = 2 ;             HEAD[2] = 4 ;       HEAD[3] = 6 ;
         POS.x   = EPS ;        POS.y += TWO*(gid%ny) ;   POS.z  += TWO*(int)(gid/ny) ;
      } else {
         HEAD[0] = 1 ;          HEAD[1] = 3 ;             HEAD[2] = 5 ;       HEAD[3] = 7 ;
         POS.x   = NX-EPS ;     POS.y  += TWO*(gid%ny) ;  POS.z  += TWO*(int)(gid/ny) ;
      }
   } else {
      if (LEADING<4) {
         if (LEADING==2) { 
            HEAD[0] = 0 ;       HEAD[1] = 1 ;             HEAD[2] = 4 ;       HEAD[3] = 5 ;
            POS.y   = EPS ;     POS.x  += TWO*(gid%nx) ;  POS.z  += TWO*(int)(gid/nx) ; 
         } else {
            HEAD[0] = 2 ;       HEAD[1] = 3 ;             HEAD[2] = 6 ;       HEAD[3] = 7 ;
            POS.y   = NY-EPS ;  POS.x  += TWO*(gid%nx) ;  POS.z  += TWO*(int)(gid/nx) ;
         }
      } else {      
         if (LEADING==4) { 
            HEAD[0] = 0 ;       HEAD[1] = 1 ;             HEAD[2] = 2 ;        HEAD[3] = 3 ;
            POS.z   = EPS ;     POS.x  += TWO*(gid%nx) ;  POS.y  += TWO*(int)(gid/nx) ; 
         } else {
            HEAD[0] = 4 ;       HEAD[1] = 5 ;             HEAD[2] = 6 ;        HEAD[3] = 7 ;
            POS.z   = NZ-EPS ;  POS.x += TWO*(gid%nx) ;   POS.y  += TWO*(int)(gid/nx) ;
         }
      }
   }
   int *SUBS ;
   
   
   // At this point POS [0,NX]
   // IndexGR takes [0,NX] coordinates and just returns the root-grid index
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;   // remain at root level, not yet going to leaf -- this just calculates index
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;
   OTLO      =  OTL ;
   RL        =  0 ;    // level of the ray (0 for the root-grid rays)
   if (INDEX>=0)  *count += 1 ;  // number of incoming rays
   
# if (DEBUG>0)
   printf("CREATE    ") ;
   report(OTL, OTI, RL, &POS, OFF, PAR) ; 
# endif   
   
   
   int steps = 0 ;
   
   
   while(INDEX>=0) {  // INDEX may refer to a cell that is not a leaf
      
      
      
      // ===== ADD LEADING EDGE RAYS WHEN ENTERING REFINED REGION THROUGH UPSTREAM CELL BORDER =====
      // This guarantees that the leading edge always has one ray per cell (4 per octet).
      // These are added whenever we step into a refined cell. The other needed rays are added as
      // "siderays" above. Also that relies on the fact that for each octet at level OTL, among the
      // rays that hit its leading edge, there is exactly one ray with RL < OTL (as well as three 
      // with RL==OTL).
      // If we step over several levels, we must again worry about the precision of POS
      if (RHO[INDEX]<=0.0f) {                    // go to the sub-cell and add sibling rays
         SID    =  -1 ;
         c1     =  NBUF*(26+CHANNELS) ;          // 62+CHANNELS elements per buffer entry
         sr     =  0  ;
         // If we are not in a leaf, we have gone to go one level up,
         // add >=three rays, pick one of them as the current, return to the beginning of the loop.
         // Ok also for OCTREE5, same step into refined cells with fmod(pos,1)
         POS.x  =  TWO*fmod(POS.x, ONE) ;      // coordinates inside the single child cell [0,1]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         flo    =  -RHO[INDEX] ;                 // OTI for the first subcell in octet
         OTL   +=  1  ;                          // step to the next refinement level
# if 1
         // Safeguards??
         POS.x  =  clamp(POS.x, EPS, TMEPS) ;
         POS.y  =  clamp(POS.y, EPS, TMEPS) ;
         POS.z  =  clamp(POS.z, EPS, TMEPS) ;
# endif     
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // SID for subcell with the ray
         OTI    =  *(int *)&flo + SID ;          // cell of the incoming ray
         
         c2     =  0 ;                           // set c2=1 if ray is to be split: is this the leading edge?
         if (LEADING<3) {
            if ((LEADING==0)&&(POS.x<  EPS2)) c2 = 1 ; 
            if ((LEADING==1)&&(POS.x>TMEPS2)) c2 = 1 ; 
            if ((LEADING==2)&&(POS.y<  EPS2)) c2 = 1 ;
         } else {
            if ((LEADING==3)&&(POS.y>TMEPS2)) c2 = 1 ; 
            if ((LEADING==4)&&(POS.z<  EPS2)) c2 = 1 ;
            if ((LEADING==5)&&(POS.z>TMEPS2)) c2 = 1 ;
         }
         if (c2>0) {  // split the ray and choose one of the new ones to be followed first
            BUFFER[c1+0]  =  OTL ;               // level where the split is done
            BUFFER[c1+1]  =  I2F(OTI) ;          // buffer contains OTI of the original ray
            // Put the main ray as the first, sr=0 entry in buffer
            BUFFER[c1+2]  =  POS.x ;  // for OCTREE5 these [0,1]
            BUFFER[c1+3]  =  POS.y ;
            BUFFER[c1+4]  =  POS.z ;
            BUFFER[c1+5]  =  RL    ;
            sr            =  1 ;
            // Add two more subrays to buffer, the third one is taken as the current ray
            // HEAD is the four cells on the upstream border
            if (                     HEAD[0] ==  SID) {  // 0=original => (1,2) to buffer, 3 as current
               sid                            =  HEAD[1] ;
               BUFFER[c1+2+4*sr]              =  fmod(POS.x,ONE) + (int)( sid % 2) ;
               BUFFER[c1+3+4*sr]              =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
               BUFFER[c1+4+4*sr]              =  fmod(POS.z,ONE) + (int)( sid/4) ;
               sid                            =  HEAD[2] ;
               BUFFER[c1+2+4*(sr+1)]          =  fmod(POS.x,ONE) + (int)( sid % 2) ;
               BUFFER[c1+3+4*(sr+1)]          =  fmod(POS.y,ONE) + (int)((sid/2)%2)  ;
               BUFFER[c1+4+4*(sr+1)]          =  fmod(POS.z,ONE) + (int)( sid/4) ;
               SID                            =  HEAD[3] ;  // SID = the subray that becomes the current ray
            } else {
               if (                  HEAD[1] ==  SID) {
                  sid                         =  HEAD[2] ;
                  BUFFER[c1+2+4*sr]           =  fmod(POS.x,ONE) + (int)( sid   %2) ;
                  BUFFER[c1+3+4*sr]           =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                  BUFFER[c1+4+4*sr]           =  fmod(POS.z,ONE) + (int)( sid/4) ;
                  sid                         =  HEAD[3] ;
                  BUFFER[c1+2+4*(sr+1)]       =  fmod(POS.x,ONE) + (int)( sid   %2) ;
                  BUFFER[c1+3+4*(sr+1)]       =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                  BUFFER[c1+4+4*(sr+1)]       =  fmod(POS.z,ONE) + (int)( sid/4) ;
                  SID                         =  HEAD[0] ;
               } else {
                  if (               HEAD[2] ==  SID) {
                     sid                      =  HEAD[3] ;
                     BUFFER[c1+2+4*sr]        =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                     BUFFER[c1+3+4*sr]        =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                     BUFFER[c1+4+4*sr]        =  fmod(POS.z,ONE) + (int)( sid/4) ;
                     sid                      =  HEAD[0] ;
                     BUFFER[c1+2+4*(sr+1)]    =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                     BUFFER[c1+3+4*(sr+1)]    =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                     BUFFER[c1+4+4*(sr+1)]    =  fmod(POS.z,ONE) + (int)( sid/4) ;
                     SID                      =  HEAD[1] ;
                  } else {
                     if (            HEAD[3] ==  SID) {
                        sid                   =  HEAD[0] ;
                        BUFFER[c1+2+4*sr]     =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                        BUFFER[c1+3+4*sr]     =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                        BUFFER[c1+4+4*sr]     =  fmod(POS.z,ONE) + (int)( sid/4) ;
                        sid                   =  HEAD[1] ;
                        BUFFER[c1+2+4*(sr+1)] =  fmod(POS.x,ONE) + (int)( sid % 2) ;
                        BUFFER[c1+3+4*(sr+1)] =  fmod(POS.y,ONE) + (int)((sid/2)%2 ) ;
                        BUFFER[c1+4+4*(sr+1)] =  fmod(POS.z,ONE) + (int)( sid/4) ;
                        SID                   =  HEAD[2] ;
                     } else {
                        ; // printf("??????????????? LEADING %d, SID %d\n", LEADING, SID) ;
                     }
                  }
               }
            }
            // two subrays added to buffer, update their RL
            // remember that when refinement jumps to higher value, we come here for every step in refinement
            // before stepping forward
            BUFFER[c1+5+4*sr]  =  OTL ;       BUFFER[c1+5+4*(sr+1)]  =  OTL ;   // RL == current OTL
            sr += 2 ;   // main ray and  two subrays added, the third subray==SID will become the current one
            for(int i=sr; i<6; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused, max 6 subrays
            NBUF += 1 ;
            
            // if (NBUF>=1000) printf("!!!!!!!!!!  NBUF = %d\n", NBUF) ;
            
            
# if (DEBUG>0)
            printf("------------------------------------------------------------------------------------\n") ;
            for(int i=0; i<sr; i++) {
               printf("ADD.LEAD%d ", i) ;
               otl       =  BUFFER[c1+0] ;
               oti       =  F2I(BUFFER[c1+1]) ;
               pos.x     =  BUFFER[c1+2+4*i] ; pos.y =  BUFFER[c1+3+4*i] ; pos.z =  BUFFER[c1+4+4*i] ;
               sid       =  4*(int)floor(pos.z) + 2*(int)floor(pos.y) + (int)floor(pos.x) ; 
               oti       =  8*(int)(OTI/8) + SID ;
               otl_ray   =  BUFFER[c1+5+4*i] ;            
               report(otl, oti, otl_ray, &pos, OFF, PAR) ; 
            }
            printf("------------------------------------------------------------------------------------\n") ;
# endif         
            
            // SID = is the third new leading-edge subray =>  switch to that
            // OTL was already increase by one = refinement
            OTI       =  (*(int *)&flo) + SID ;    // the new ray to be followed, OTI = index of first subcell
            POS.x     =  fmod(POS.x,ONE) + (int)( SID%2)    ;
            POS.y     =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
            POS.z     =  fmod(POS.z,ONE) + (int)( SID/4)    ;
            RL        =  OTL ;      // leading edge rays created at level OTL
            
            
         }  //  c2>0  == other leading-edge subrays added
      } // RHO < 0.0 == we entered refined region
      
      
      
      
      INDEX = OFF[OTL]+OTI ;   // global index -- must be now >=0 since we went down a level
      
      
      // if not yet a leaf, jump back to start of the loop, for another step down
      if (RHO[INDEX]<=0.0f) {
         // printf("RHO<0\n") ;
         steps += 1  ;   
         // if (steps>100000) printf("RHO[%d]<=0.0  --- OTL %d, OTI %d\n", INDEX, OTL, OTI) ;
         continue ;
      }
      
      
      
      
      
# if 1
      // RL  = ray level, the level at which the ray was created, root rays are RL=0
      // OTL = cell level, the current discretisation level, root grid is OTL=0
      // =>  Check at each step if there are potential siderays to be added.
      // RL rays provide siderays at levels XL,    RL <  XL <= OTL
      //    (A) it is boundary between level XL octets
      //    (B) upstream-side neighbour has OTL<XL -- so that it will not provide the XL rays directly
      //    (C) the XL sideray is not also XL-1 ray --  those XL-1 siderays are provided by XL-2 rays
      //        since current ray is XL ray, skip even offsets where the step=1.0 in level XL coordinates
      //    (D) the XL sideray actually hits the side of the current octet with level XL cells
      
      // This ok also for OCTREE5, RootPos returns always [0,NX] coordinates
      pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;   // current position in root coordinates
      
      
      // Current ray is still (OTL, OTI, POS)
      for(XL=OTL; XL>RL; XL--) {   // try to add subrays at level XL -- never on level=0
         // In this loop we do not have to deal with any root-grid coordinates
         
         c1     =  NBUF*(26+CHANNELS) ;          // 26+CHANNELS elements per buffer entry
         sr     =  0  ;
         
         if (((LEADING==0)&&(POS.x<EPS2))||((LEADING==1)&&(POS.x>TMEPS2))) {  // +/-X leading edge, at OTL level
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.x/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test smaller XL (larger dx) values
            // even number of dx,is therefore a border between level XL octets
            // calculate (pos, level, ind) for the position at level XL  (in the octet that we are in)
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            // Note IndexUP used only at levels>0 => independent of the handling of the root-grid coordinates
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // (pos, level, ind) now define the position at level XL
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL ===== offsets                              Y and Z
            // check XL-scale neighbour
            pos1.x =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       Y * 
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            // It is called only at levels level>0, not affected by the handling of root-grid coordinates
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               // Current ray position at level==XL is defined by (level, ind, pos).
               // We loop over XL positions on the leading-edge plane, ignore those not common with
               // XL-1 rays, and choose those that actually hit the -Y side of the current octet.
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.y += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =       Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;    // still inside the current octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     // pos1 = initial coordinates at the leading-edge level, step forward to the Y sidewall
                     if (DIR.y>0.0f)   pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ;  // step based on    Y ****
                     else              pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;   // to 2.0  (level==RL+1)
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
                     pos1.y  =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in   octet        Y *****
                     // Add the ray from the upstream direction to buffer as a new subray
                     //    we leave the buffer entry at level XL, if leafs are >XL, we drop to next 
                     //    refinement level only at the beginning of the main loop.
                     // Each sideray will actually continue in location refined to some level >=XL
                     // We will store NTRUE correct for XL. If sideray is in refined region, beginning of the
                     // main loop will drop the ray to the higher level and rescale NTRUE accordingly.
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level==XL
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL ===== offsets                              Z and Y
            // current ray at level level==RL+1 defined by (level, ind, pos)
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {     // neighbour not refined to level==XL, will not provide XL siderays
               for(int a=-1; a<=3; a++) {       // offset in +/- Z direction from the level RL ray, in XL steps
                  for(int b=-1; b<=3; b++) {    // ray offset in +/- Y direction, candidate relative to current ray
                     if ((a%2==0)&&(b%2==0)) continue ; // skip LR rays
                     // if we come this far, we will add the ray if it just hits the current octet with RL+1 cells
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =      Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // in current octet, will not hit sidewall
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // second offset           Y ***
                     // pos1 = initial coordinates at the leading edge plane, step to the Z sidewall
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in octet,        Z *****
#  endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL     ;      // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (((LEADING==2)&&(POS.y<EPS2))||((LEADING==3)&&(POS.y>TMEPS2))) {  // +/-Y leading edge
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.y/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // ***A*** even number of dx,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL) IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITTING THE X SIDEWALL ===== offsets                             X and Z
            pos1.y =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.x =  (DIR.x>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       X *
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos inthe octet,      X *****
#  endif
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL =====  offsets                             Z and X
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =     Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =     X ***
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in the octet,    Z *****
#  endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         
         if (((LEADING==4)&&(POS.z<EPS2))||((LEADING==5)&&(POS.z>TMEPS2))) {  // +/-Z leading edge
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.z/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // even number of dx,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITING THE X SIDEWALL ===== offsets                              X and Y
            pos1.y =  0.1f ;     pos1.z = 0.1f ;
            pos1.x =  (DIR.x>0.0f) ? (-0.1f) : (2.1f) ;    // upstream neighbour, main offset       X *
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.y += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Y ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos in the octet,     X *****
#  endif
                     BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                     BUFFER[c1+3+4*sr] =  pos1.y ;
                     BUFFER[c1+4+4*sr] =  pos1.z ;
                     BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     sr += 1 ;                     
                     
#  if (DEBUG>0)
                     printf("!!!A  ") ;
                     report(level, ind, XL, &pos, OFF, PAR) ;
#  endif      
                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL =====  offsets                            Y and X
            pos1.x =  0.1f ;    pos1.z = 0.1f ;
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset       Y *
            level1 = level ;    ind1 = ind  ;  
            IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL 
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =    Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =    X ***
                     if (DIR.y>0.0f) pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ; // step based on      Y ****
                     else            pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.y   =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in the octet,   Y *****
#  endif
                     BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                     BUFFER[c1+3+4*sr] = pos1.y ;
                     BUFFER[c1+4+4*sr] = pos1.z ;
                     BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         
         
         
         if (sr>0) {  // added some siderays
            // While the current ray was at level OTL, there are no guarantees that the same
            // refinement exists for all added siderays. Therefore, we add all siderays
            // using the level XL coordinates. This also means that the siderays and the
            // leading-edge subrays (below) must be stored as separate BUFFER entries (different NBUF).
            BUFFER[c1+0] = XL ;         // at level XL all rays stored as level XL rays
            BUFFER[c1+1] = I2F(ind) ;   // index of *original ray*, at level XL
            // We leave the original ray as the current one == (OTL, OTI, POS), these unchanged.
            
#  if (DEBUG>0)
            printf("!!! gid=%d ADDED =========== (%d, %d) %d\n", gid, XL, ind, OFF[XL]+ind) ;
            printf("================================================================================\n") ;
            for(int i=0; i<sr; i++) {   // skip sr=0 that is so far an empty slot
               otl       =  BUFFER[c1+0] ;
               oti       =  F2I(BUFFER[c1+1]) ;   // cell index for  the icoming ray
               pos.x     =  BUFFER[c1+2+4*i] ; pos.y =  BUFFER[c1+3+4*i] ; pos.z =  BUFFER[c1+4+4*i] ;
               sid       =  4*(int)floor(pos.z) + 2*(int)floor(pos.y) + (int)floor(pos.x) ; 
               oti       =  8*(int)(OTI/8) + sid ;   // cell index for the added ray
               otl_ray   =  BUFFER[c1+5+4*i] ;            
               printf("ADD.SIDE%d   main ray (%d,%d) %d   L: %8.4f %8.4f %8.4f --- NBUF =%d\n",
                      i, otl, oti, OFF[otl]+oti, pos.x, pos.y, pos.z, NBUF) ;
               // splitpos  =  BUFFER[c1+6+4*i] ;
               printf("ADD.SIDE%d ", i) ;
               report(otl, oti, otl_ray, &pos, OFF, PAR) ;
               
            }
            printf("================================================================================\n") ;
#  endif
            
            for(int i=sr; i<6; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
            NBUF += 1 ;
            // if (NBUF>=1000) printf("!!!!!!!!!!  NBUF = %d\n", NBUF) ;
            
            // When NTRUE is saved to buffer, the values must be stored rescaled to the level XL.
            // Thus, the stored values are   NTRUE(for_OTL) * 4.0**(OTL-XL)
            // When sideray is taken from buffer and is located in a region with OTL>XL, the
            // beginning of the main loop will again rescale with  4.0*(XL-OTL)
         }
         
      } // for XL -- adding possible siderays
      
      
      // Subrays are added only when we are at leaf level. Also, when subrays are added, 
      // we continue with the original RL ray and immediately make a step forward. Therefore
      // there is no risk that we would add siderays multiple times.
# endif
      
      
      
      
# if (DEBUG>1)      
      printf("CONTINUE  ") ;
      report(OTL, OTI, RL, &POS, OFF, PAR) ; 
# endif      
      
      
      // we come here only if we have a ray that is inside a leaf node => can do the step forward
      // if the step is to refined cell, (OTL, OTI) will refer to a cell that is not a leaf node
      // ==> the beginning of the present loop will have to deal with the step to higher refinement
      // level and the creation of three additional rays
      OTLO    =  OTL ;  c2 = OTI ;
      // get step but do not yet go to level > OTLO
          
# if (DEBUG>0)
      pos1 = POS ;
      RootPos(&pos1, OTL, OTI, OFF, PAR) ;      
      c2 = OFF[OTL]+OTI ;
      printf("[%d] @ STEP    %9.6f %9.6f %9.6f   %d  ", gid, pos1.x, pos1.y, pos1.z, OTLO) ;      
# endif
      
      
      
      dr      =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL   , LEADING) ; // step [GL] == root grid units !!
      
      
      AADD(&(PL[INDEX]), dr) ;
      
      
      
      tpl       += dr ;               // just the total value for current idir, ioff
      
      // The new index at the end of the step  --- OTL >= OTL  => INDEX may not refer to a leaf cell
      INDEX    =  (OTI>=0) ?  (OFF[OTL]+OTI) : (-1) ;  // OTL, OTI, POS already updated = end of the current step
      
      
      if (INDEX>=0) {
         if (RHO[INDEX]<=ZERO) {  // we ended up in a parent cell
            steps += 1 ;    // if (steps>100000) printf("LINE 4528 !!!!!!!!!!!!\n") ;
            continue ; // we moved to refined region, handled at the beginning of the main loop
         }
      }
      
      
      if (OTL<RL) {           // up in hierarchy and this ray is terminated -- cannot be on root grid
         INDEX=-1 ;           // triggers the generation of a new ray below
      } else {     
         if (INDEX<0) {       // ray exits the root grid, possibly create a new OTL=0 ray on the other side
            OTL    =  0 ;
            OTI    = -OTI-1 ;  // this was the cell where we exited, step ending at current POS
            INDEX  = -1 ;
            if (POS.x>1.0f) {
               if (LEADING!=0) {   POS.x = EPS ;    INDEX -= NX-1 ;   }
            }
            if (POS.x<0.0f) {
               if (LEADING!=1) {   POS.x = OMEPS ;  INDEX += NX-1 ;   }
            }
            if (POS.y>1.0f) {
               if (LEADING!=2) {   POS.y = EPS ;    INDEX -= NX*(NY-1) ;   }
            }
            if (POS.y<0.0f) {
               if (LEADING!=3) {   POS.y = OMEPS ;  INDEX += NX*(NY-1) ;   }
            }
            if (POS.z>1.0f) {
               if (LEADING!=4) {   POS.z = EPS ;    INDEX -= NX*NY*(NZ-1) ;   }
            }
            if (POS.z<0.0f) {
               if (LEADING!=5) {   POS.x = OMEPS ;  INDEX += NX*NY*(NZ-1) ;   }
            }
            if (INDEX>=0) {    // new level-0 ray started on the opposite side
               RL = 0 ;    OTLO    = 0 ;   *count += 1 ;
# if (DEBUG>0)
               printf("MIRROR    ") ;
               report(OTL, OTI, RL, &POS, OFF, PAR) ;
# endif
               continue ;
            }
         } // if INDEX<0
      }
      
      
      
      // [C] if INDEX still negative, current ray truly ended => take new ray from the buffer, if such exist
      //  NBUF>0 is no guarantee that rays exist in the buffer because we reserved sr=0 for the main ray
      //  and if only siderays were added, this NBUF entry has nothing for sr=0
      if ((INDEX<0)&&(NBUF>0))   {                // find a ray from BUFFER
         c1    =  (NBUF-1)*(26+CHANNELS) ;  // 26+CHANNELS elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // level where sibling rays were created
         OTLO  =  OTL ;
         OTI   =  F2I(BUFFER[c1+1]) ;       // OTI for the original ray that was split
         // a maximum of 6 rays per buffer entry, the entry sr=0 was reserved for the main ray
         for(sr=5; sr>=0; sr--) {         
            dx    =  BUFFER[c1+2+4*sr] ;
            if (dx>-0.1f) break ;      // found a ray
         }
         // if (dx<ZERO) printf("??? BUFFER WITH NEGATIVE POS.x ???\n") ;
         POS.x     =  dx ;
         POS.y     =  BUFFER[c1+3+4*sr] ;
         POS.z     =  BUFFER[c1+4+4*sr] ;
         RL        =  BUFFER[c1+5+4*sr] ;    
         // we must have found a ray  -- figure out its OTI (index within level OTL)
         SID       =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
         OTI       =  8*(int)(OTI/8) + SID ;    // OTI in the buffer was for the original ray that was split
         INDEX     =  OFF[OTL]+OTI ;    // global index -- must be >=0 !!!
# if (DEBUG>0)
         printf("FROMBUF   ") ;
         report(OTL, OTI, RL, &POS, OFF, PAR) ; 
# endif         
         // this stored ray is at level OTL but may be in a cell that is itself still further
         // refined => this will be handled at the start of the main loop => possible further
         // refinement before on the rays is stepped forward
         // However, if we are still at SPLITPOS, new siderays will not be possible before >=1 steps
         BUFFER[c1+2+4*sr] = -1.0f ;   // mark as used
         if (sr==0) NBUF -= 1 ;        // was last of the <=6 subrays in this buffer entry
      }  // (INDEX<0)
      
      
      // if (INDEX<0) { if (NBUF<=0)   FOLLOW=0 ;  }
          
   } // while INDEX>=0  --- stops when buffer is empty and the main level-0 ray has exited
   
   TPL[gid-gid0] = tpl ;
}  // END OF PathsOT5











__kernel void UpdateOT5(  // 
                          const int        gid0,    //  first gid in the index running over NRAY>=NWG
                          __global float  *PL,      //  0
# if (WITH_HALF==1)
                          __global short4 *CLOUD,   //  1 [CELLS]: vx, vy, vz, sigma
# else
                          __global float4 *CLOUD,   //  1 [CELLS]: vx, vy, vz, sigma
# endif
                          GAUSTORE  float *GAU,     //  2 precalculated gaussian profiles [GNO,CHANNELS]
                          constant int2   *LIM,     //  3 limits of ~zero profile function [GNO]
                          const float      Aul,     //  4 Einstein A(upper->lower)
                          const float      A_b,     //  5 (g_u/g_l)*B(upper->lower)
                          const float      GN,      //  6 Gauss normalisation == C_LIGHT/(1e5*DV*freq)
                          const float      APL,     //  7 average path length [GL]
                          const float      BG,      //  8 background value (photons)
                          const float      DIRWEI,  //  9 weight factor (based on current direction)
                          const float      EWEI,    // 10 weight 1/<1/cosT>/NDIR
                          const int        LEADING, //  0 leading edge
                          const REAL3      POS0,    // 11 initial position of id=0 ray
                          const float3     DIR,     // 12 ray direction
                          __global float  *NI,      // 13 [CELLS]:  NI[upper] + NB_NB
                          __global float  *RES,     // 14 [CELLS]: SIJ, ESC ---- or just [CELLS] for SIJ
                          __global float  *NTRUES,  // 15 [NWG*MAXCHN]   --- NWG>=simultaneous level 0 rays
# if (WITH_CRT>0)
                          constant float *CRT_TAU,  //  dust optical depth / GL
                          constant float *CRT_EMI,  //  dust emission photons/c/channel/H
# endif                     
# if (BRUTE_COOLING>0)
                          __global float   *COOL,   // 14,15 [CELLS] = cooling 
# endif
                          __global   int   *LCELLS, //  16
                          __constant int   *OFF,    //  17
                          __global   int   *PAR,    //  18
                          __global   float *RHO,    //  19  -- needed only to describe the hierarchy
                          __global   float *BUFFER_ALL  //  20 -- buffer to store split rays
                       )  {   
   // Each ***WORK GROUP*** processes one ray. The rays are two cells apart to avoid synchronisation
   // problems. Rays start on the leading edge. If ray exits through a side (wrt axis closest to
   // direction of propagation), a new one is created on the opposite side and the ray ends when the
   // downstream edge is reached.
   // 
   // As one goes to a higher hierarchy level (into a refined cell), one pushes the original and two
   // new rays to buffer. The fourth ray (one of the new ones) is followed first. When ray goes up in
   // hierarchy (to a larger cell) ray is terminated if ray was created at a higher level. When one
   // ray is terminated, the next one is taken from the buffer, if that is not empty. When ray goes up
   // in hierarchy, NTRUE is scaled *= 4 or, if one goes up several levels, *=4^n, where n is the
   // decrease in the hierarchy level.
   int id  = get_global_id(0), lid = get_local_id(0), gid = get_group_id(0), ls  = get_local_size(0) ;
   __global float *BUFFER = &BUFFER_ALL[gid*(26+CHANNELS)*MAX_NBUF] ;
   gid += gid0 ;  // becomes running index over NRAY, instead of NWG
   if (gid>=NRAY) return ;        // one work group per ray .... NWG==NRAY   
   int nx=(NX+1)/2, ny=(NY+1)/2, nz=(NZ+1)/2 ;
   GAUSTORE float *profile ;
   __local  float  NTRUE[CHANNELS] ;
   __local  int2   LINT ;          // SID, NBUF
   __local  float  SDT[LOCAL] ;    // per-workitem sum_delta_true
   __local  float  AE[LOCAL] ;     // per-workitem all_escaped
   float weight, w, doppler, tmp_tau, tmp_emit, nb_nb, factor, escape, absorbed, sum_delta_true, all_escaped, nu ;
   int row, shift, INDEX, c1, c2, OTL, OTI, OTLO, XL, RL, SID, NBUF=0, sr, sid, level, b1, b2, I, i, ind, otl, oti ;
   int level1, ind1 ;
   REAL3  POS, pos0, pos1, pos, RDIR ;
   REAL   dx, dy, dz, dr, s ;
   float flo ;
# if (WITH_CRT>0)
   float Ctau, Cemit, Ltau, Ttau, tt, ttt, Lleave, Dleave, pro, sij ;
# endif  
   RDIR.x = DIR.x ; RDIR.y = DIR.y ; RDIR.z = DIR.z ;
   
   int *SUBS ;
   POS.x = POS0.x ;   POS.y = POS0.y ;   POS.z = POS0.z ;
   
   // when split done only on the leading edge -- HEAD are the four subscells on the leading edge
   int HEAD[4] ;   // the sub-indices of the four leading-edge subcells
   if (LEADING<3) {
      if (LEADING==0) { 
         HEAD[0] = 0 ;   HEAD[1] = 2 ;   HEAD[2] = 4 ;  HEAD[3] = 6 ;
         POS.x =    EPS ;  POS.y += TWO*(gid%ny) ;  POS.z += TWO*(int)(gid/ny) ;  // gid is index running over all NRAY
      }
      if (LEADING==1) { 
         HEAD[0] = 1 ;   HEAD[1] = 3 ;   HEAD[2] = 5 ;  HEAD[3] = 7 ;
         POS.x = NX-EPS ;  POS.y += TWO*(gid%ny) ;  POS.z += TWO*(int)(gid/ny) ;
      }
      if (LEADING==2) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 4 ;  HEAD[3] = 5 ;
         POS.y =    EPS ;  POS.x += TWO*(gid%nx) ;  POS.z += TWO*(int)(gid/nx) ;
      }
   } else {
      if (LEADING==3) { 
         HEAD[0] = 2 ;   HEAD[1] = 3 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.y = NY-EPS ;  POS.x += TWO*(gid%nx) ;  POS.z += TWO*(int)(gid/nx) ;
      }
      if (LEADING==4) { 
         HEAD[0] = 0 ;   HEAD[1] = 1 ;   HEAD[2] = 2 ;  HEAD[3] = 3 ;
         POS.z =    EPS ;  POS.x += TWO*(gid%nx) ;  POS.y += TWO*(int)(gid/nx) ;
      }
      if (LEADING==5) { 
         HEAD[0] = 4 ;   HEAD[1] = 5 ;   HEAD[2] = 6 ;  HEAD[3] = 7 ;
         POS.z = NZ-EPS ;  POS.x += TWO*(gid%nx) ;  POS.y += TWO*(int)(gid/nx) ;
      }
   }
   
   IndexGR(&POS, &OTL, &OTI, RHO, OFF) ;     // remain at root level, not yet going to leaf
   if (OTI<0) return ;
   INDEX     =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1)  ;  
   OTLO      =  OTL ;
   RL        =  0 ;
   for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BG * DIRWEI ;
   
   
   
   while(INDEX>=0) {  // INDEX may refer to a cell that is not a leaf
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
      sr  = 0 ;
      c1  = NBUF*(26+CHANNELS) ;
      SID = -1 ;
      
      // If we are not in a leaf, we have gone to some higher level. 
      // Go one level higher, add >=three rays, pick one of them as the current,
      // and return to the beginning of the loop.
      if (RHO[INDEX]<=0.0f) {                    // go to the sub-cell and add sibling rays
         POS.x  =  TWO*fmod(POS.x, ONE) ;      // coordinate inside parent cell [0,1]
         POS.y  =  TWO*fmod(POS.y, ONE) ;
         POS.z  =  TWO*fmod(POS.z, ONE) ;
         flo    =  -RHO[INDEX] ;                 // OTL, OTI of the parent cell
         OTL   +=  1  ;                          // step to next level = refined level
         OTI    =  *(int *)&flo ;                // OTI for the first child in octet
         SID    =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // original subcell
         OTI   +=  SID;                          // cell in octet, original ray
         c2     =  0 ;   // set c2=1 if ray is to be split (is this the leading cell edge?)
         if (LEADING<3) {
            if (LEADING==0) {  
               if (POS.x<EPS2)      c2 = 1 ; 
            } else {
               if (LEADING==1) {  
                  if (POS.x>TMEPS2) c2 = 1 ; 
               } else {
                  if (POS.y<EPS2)   c2 = 1 ; 
               }
            }
         } else {
            if (LEADING==3) { 
               if (POS.y>TMEPS2)    c2 = 1 ; 
            } else {
               if (LEADING==4) {  
                  if (POS.z<EPS2)   c2 = 1 ; 
               } else {
                  if (POS.z>TMEPS2) c2 = 1 ; 
               }
            }
         }
         // @@  rescale always when the resolution changes
         for(int i=lid; i<CHANNELS; i+=ls) {   // SCALE ON EVERY REFINEMENT EVEN WHEN NOT SPLIT
            NTRUE[i] *= 0.25f ; 
         }
         barrier(CLK_LOCAL_MEM_FENCE) ;
         
         if (c2>0) {  // split the ray and choose one of the new ones to be followed first
            for(int i=lid; i<CHANNELS; i+=ls) {   // ray effectively split to four
               BUFFER[c1+26+i] = NTRUE[i] ;
            }
            barrier(CLK_LOCAL_MEM_FENCE) ;            
            if (lid==0) {
               BUFFER[c1+0]  =  OTL ;                     // level where the split is done, OTL>this =>
               BUFFER[c1+1]  =  I2F(OTI) ;                // buffer contains the OTI of the original ray 
               BUFFER[c1+2]  =  POS.x ;                   // Store the original MAIN RAY to buffer as the first one
               BUFFER[c1+3]  =  POS.y ;
               BUFFER[c1+4]  =  POS.z ;
               BUFFER[c1+5]  =  RL  ;                     // main ray exists at levels >= RL
            }            
            // Two new subrays added to buffer, third one becomes the current ray
            // Add first two subrays to buffer
            if (                  HEAD[0]==SID) {  // 0=original => (1,2) to buffer, 3 as current
               if (lid==0) {
                  sid              = HEAD[1] ;
                  BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                  BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                  BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                  sid              = HEAD[2] ;
                  BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                  BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                  BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
               }
               SID              = HEAD[3] ;  // SID of the subray to be followed first
            } else {
               if (                  HEAD[1]==SID) {
                  if (lid==0) {
                     sid              = HEAD[2] ;
                     BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                     BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                     BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                     sid              = HEAD[3] ;
                     BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                     BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                     BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                  }
                  SID              = HEAD[0] ;
               } else {
                  if (                  HEAD[2]==SID) {
                     if (lid==0) {
                        sid              = HEAD[3] ;
                        BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                        BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                        BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                        sid              = HEAD[0] ;
                        BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                        BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                        BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                     }
                     SID              = HEAD[1] ;
                  } else {
                     if (                  HEAD[3]==SID) {
                        if (lid==0) {
                           sid              = HEAD[0] ;
                           BUFFER[c1+2+4*1] = fmod(POS.x,ONE) + (sid % 2) ;
                           BUFFER[c1+3+4*1] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                           BUFFER[c1+4+4*1] = fmod(POS.z,ONE) + (sid/4) ;
                           sid              = HEAD[1] ;
                           BUFFER[c1+2+4*2] = fmod(POS.x,ONE) + (sid % 2) ;
                           BUFFER[c1+3+4*2] = fmod(POS.y,ONE) + ((sid/2)%2 ) ;
                           BUFFER[c1+4+4*2] = fmod(POS.z,ONE) + (sid/4) ;
                        }
                        SID              = HEAD[2] ;
                     } else {
                        ; // printf("???\n") ;
                     }
                  }
               }
            } 
            // for the two subrays just added, update RL, SPLIT
            sr = 3 ;   // so far the original main ray and two split subrays
            if (lid==0) {
               BUFFER[c1+5+4*1]  =  OTL ;    BUFFER[c1+5+4*2]  =  OTL ;  // RL = current OTL
               for(int i=sr; i<6; i++)  BUFFER[c1+2+4*i] = -99.0f ;      // mark the rest as unused
            } // lid==0
            // We added leading edge rays, old main ray is in buffer, SID refers to one of the subrays
            // update OTI and POS to correspond to that subray
            OTI       =  (*(int *)&flo)  + SID ;    // the new ray to be followed, OTI = index of first subcell + SID
            POS.x     =  fmod(POS.x,ONE) + (int)( SID%2)    ;  // dx and SID known to all work items
            POS.y     =  fmod(POS.y,ONE) + (int)((SID/2)%2) ;
            POS.z     =  fmod(POS.z,ONE) + (int)( SID/4)    ;
            RL        =  OTL ;  // when we reach OTL<RL, this ray will be terminated
            NBUF++ ;            // >= 1 subrays were added, NBUF increases just by one
            
         } // c2>0  == we split rays on the leading edge
         
      } // RHO<0
      
      
      INDEX  = OFF[OTL] + OTI ;
      if (RHO[INDEX]<=0.0f) continue ;  // not a leaf, go back to the beginning of the main loop
      
      
      
      
      
# if 1  // adding siderays
      
      // It is essential that we are already in a leaf node when siderays are added:
      // once ray has produced siderays at the current location, it will immediately take a 
      // real step forward.
      
      // we use root coordinates to determine whether RL ray is also RL-1 ray
      // the required accuracy is only 0.5*0.5**MAXL, float32 works at least to MAXL=15
      pos0 = POS ;     RootPos(&pos0, OTL, OTI, OFF, PAR) ;   // current position in root coordinates
      
      
      // Current ray is still (OTL, OTI, POS)
      // presumably current NTRUE is correct for ray at level OTL
      
      // Siderays are based on rays hitting the leading edge. We have stepped down to leaf level.
      // Therefore, incoming ray has already been split to four, and NTRUE has been scaled by 0.25 to
      // correspond to the step from OTL-1 to OTL.
      
      
      for(XL=OTL; XL>RL; XL--) {   // try to add subrays at levels   RL<XL<=OTL
         
         c1     =  NBUF*(26+CHANNELS) ;          // 26+CHANNELS elements per buffer entry
         sr     =  0  ;
         
         if (((LEADING==0)&&(POS.x<EPS2))||((LEADING==1)&&(POS.x>TMEPS2))) {  // +/-X leading edge, at OTL level
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.x/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test smaller XL (larger dx) values
            // even number of dx,is therefore a border between level XL octets
            // calculate (pos, level, ind) for the position at level XL  (in the octet that we are in)
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // (pos, level, ind) now define the position at level XL
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL ===== offsets                              Y and Z
            // check XL-scale neighbour
            pos1.x =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       Y * 
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level,NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               // Current ray position at level==XL is defined by (level, ind, pos).
               // We loop over XL positions on the leading-edge plane, ignore those not common with
               // XL-1 rays, and choose those that actually hit the -Y side of the current octet.
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.y += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =       Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;     // still inside the current octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     // pos1 = initial coordinates at the leading-edge level, step forward to the Y sidewall
                     if (DIR.y>0.0f)   pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ;  // step based on    Y ****
                     else              pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;  // to 2.0  (level==RL+1)
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.y  =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in   octet        Y *****
#  endif
                     // Add the ray from the upstream direction to buffer as a new subray
                     //    we leave the buffer entry at level XL, if leafs are >XL, we drop to next 
                     //    refinement level only at the beginning of the main loop.
                     // Each sideray will actually continue in location refined to some level >=XL
                     // We will store NTRUE correct for XL. If sideray is in refined region, beginning of the
                     // main loop will drop the ray to the higher level and rescale NTRUE accordingly.
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                        BUFFER[c1+3+4*sr] =  pos1.y ;
                        BUFFER[c1+4+4*sr] =  pos1.z ;
                        BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level==XL
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL ===== offsets                              Z and Y
            // current ray at level level==RL+1 defined by (level, ind, pos)
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {     // neighbour not refined to level==XL, will not provide XL siderays
               for(int a=-1; a<=3; a++) {       // offset in +/- Z direction from the level RL ray, in XL steps
                  for(int b=-1; b<=3; b++) {    // ray offset in +/- Y direction, candidate relative to current ray
                     if ((a%2==0)&&(b%2==0)) continue ; // skip LR rays
                     // if we come this far, we will add the ray if it just hits the current octet with RL+1 cells
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset =      Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // in current octet, will not hit sidewall
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // second offset           Y ***
                     // pos1 = initial coordinates at the leading edge plane, step to the Z sidewall
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in octet,        Z *****
#  endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                        BUFFER[c1+3+4*sr] = pos1.y ;
                        BUFFER[c1+4+4*sr] = pos1.z ;
                        BUFFER[c1+5+4*sr] = XL     ;      // ray created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (((LEADING==2)&&(POS.y<EPS2))||((LEADING==3)&&(POS.y>TMEPS2))) {  // +/-Y leading edge
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.y/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // ***A*** even number of dx,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL) IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITTING THE X SIDEWALL ===== offsets                             X and Z
            pos1.y =  0.1f ;     pos1.z = 0.1f ;           // some position in the current octet (level==XL)
            pos1.x =  (DIR.x>0.0f) ? (-0.1f)  :  (2.1f) ;  // upstream neighbour, main offset       X *
            // IndexOT will find the cell with the pos1 coordinates, stops downward search at level==RL+1
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // ***C*** neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     // if we come this far, we will add the ray if it just hits the current octet
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.z += (DIR.z>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Z ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos inthe octet,      X *****
#  endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                        BUFFER[c1+3+4*sr] =  pos1.y ;
                        BUFFER[c1+4+4*sr] =  pos1.z ;
                        BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Z SIDEWALL =====  offsets                             Z and X
            pos1.x =  0.1f ;    pos1.y = 0.1f ;
            pos1.z =  (DIR.z>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset        Z *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL
                     pos1.z  += (DIR.z>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =     Z **
                     if ((pos1.z>=ZERO)&&(pos1.z<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =     X ***
                     if (DIR.z>0.0f) pos1  +=  ((EPS  -pos1.z)/RDIR.z) * RDIR ; // step based on       Z ****
                     else            pos1  +=  ((TMEPS-pos1.z)/RDIR.z) * RDIR ;
                     // main offset Z matches, check X and Y
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.y<=ZERO)||(pos1.y>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.z   =  clamp(pos1.z, EPS2, TMEPS2) ; // make sure pos in the octet,    Z *****
#  endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                        BUFFER[c1+3+4*sr] = pos1.y ;
                        BUFFER[c1+4+4*sr] = pos1.z ;
                        BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         
         if (((LEADING==4)&&(POS.z<EPS2))||((LEADING==5)&&(POS.z>TMEPS2))) {  // +/-Z leading edge
            dx   = pown(HALF, XL) ;            // XL level cell size in root grid units
            b1   = (int)(round(pos0.z/dx)) ;   // current position is steps of XL cell size, main direction
            if (b1%2!=0)  break ;              // not XL boundary, no need to test lower XL values either
            // even number of dx,is therefore a border between level XL octets
            pos = POS ;  level = OTL ;  ind = OTI ;   // original ray at some refinement level OTL>=XL
            if (XL<OTL)  IndexUP(&pos, &level, &ind, RHO, OFF, PAR, XL) ; // (level, ind, pos) at level XL
            // if (level!=XL) printf("PROBLEM !!!!!!!!!!!!!!\n") ;
            // ===== TEST SIDERAYS HITING THE X SIDEWALL ===== offsets                              X and Y
            pos1.y =  0.1f ;     pos1.z = 0.1f ;
            pos1.x =  (DIR.x>0.0f) ? (-0.1f) : (2.1f) ;    // upstream neighbour, main offset       X *
            level1 = level ;  ind1 = ind ;    IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {    // neighbour not refined to XL==level, provides no level XL rays
               for(int a=-1; a<=3; a++) {               // level==XL offsets in +/- Y direction from the current ray
                  for(int b=-1; b<=3; b++) {            // ray offset in +/- Z direction
                     if ((a%2==0)&&(b%2==0)) continue ; // ***D*** even offsets... is also level XL-1 ray
                     pos1    = pos ;  // again the incoming ray, at the leading edge, in level==XL coordinates
                     pos1.x += (DIR.x>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =      X **
                     if ((pos1.x>=ZERO)&&(pos1.x<=TWO)) continue ;     // still inside the current   octet, ignore
                     pos1.y += (DIR.y>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =      Y ***
                     if (DIR.x>0.0f)   pos1  +=  ((EPS  -pos1.x)/RDIR.x) * RDIR ;  // step based on    X ****
                     else              pos1  +=  ((TMEPS-pos1.x)/RDIR.x) * RDIR ;
                     // main offset X matches, check Y and Z
                     if ((pos1.y<=ZERO)||(pos1.y>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.x  =  clamp(pos1.x, EPS2, TMEPS2) ; // make sure pos in the octet,     X *****
#  endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] =  pos1.x ;   // level XL coordinates
                        BUFFER[c1+3+4*sr] =  pos1.y ;
                        BUFFER[c1+4+4*sr] =  pos1.z ;
                        BUFFER[c1+5+4*sr] =  XL  ;      // ray is created at level RL+1
                     }
                     sr += 1 ;
                  } // for b
               } // for a
            } // level1<OTL_NRAY
            // ===== TEST SIDERAYS HITING THE Y SIDEWALL =====  offsets                            Y and X
            pos1.x =  0.1f ;    pos1.z = 0.1f ;
            pos1.y =  (DIR.y>0.0f) ? (-0.1f)  : (2.1f) ;  // upstream neighbour, main offset       Y *
            level1 = level ;    ind1 = ind  ;  IndexOT(&pos1, &level1, &ind1, RHO, OFF, PAR, level, NULL, NULL, -1) ;
            if (level1<level) {   // ***C*** neighbour not refined to level==RL+1, will not provide RL+1 siderays
               for(int a=-1; a<=3; a++) {       // first offset direction
                  for(int b=-1; b<=3; b++) {    // second offset direction
                     if ((a%2==0)&&(b%2==0)) continue ; // skip XL+1 rays
                     pos1     =  pos ;                                   // original ray at level XL 
                     pos1.y  += (DIR.y>0.0f) ? (-ONE*a) : (+ONE*a) ;   // add first offset  =    Y **
                     if ((pos1.y>=ZERO)&&(pos1.y<=TWO)) continue ;      // still inside the current octet, ignoe
                     pos1.x  += (DIR.x>0.0f) ? (-ONE*b) : (+ONE*b) ;   // add second offset =    X ***
                     if (DIR.y>0.0f) pos1  +=  ((EPS  -pos1.y)/RDIR.y) * RDIR ; // step based on      Y ****
                     else            pos1  +=  ((TMEPS-pos1.y)/RDIR.y) * RDIR ;
                     // main offset Y matches, check X and Z
                     if ((pos1.x<=ZERO)||(pos1.x>=TWO)||(pos1.z<=ZERO)||(pos1.z>=TWO)) continue ; // no hit
#  if (DOUBLE_POS==0)
                     pos1.y   =  clamp(pos1.y, EPS2, TMEPS2) ; // make sure pos in the octet,   Y *****
#  endif
                     if (lid==0) {
                        BUFFER[c1+2+4*sr] = pos1.x ;     // level XL coordinates
                        BUFFER[c1+3+4*sr] = pos1.y ;
                        BUFFER[c1+4+4*sr] = pos1.z ;
                        BUFFER[c1+5+4*sr] = XL  ;        // ray created at level RL+1
                     }
                     sr += 1 ;                     
                  } // for b
               } // for a
            } // level1<OTL_NRAY
         } // LEADING=0
         
         
         if (sr>0) {  // added some siderays
            // While the current ray was at level OTL, there are no guarantees that the same
            // refinement exists for all added siderays. Therefore, we add all siderays
            // using the level XL coordinates. This also means that the siderays and the
            // leading-edge subrays (below) must be stored as separate BUFFER entries (different NBUF).
            if (lid==0){
               BUFFER[c1+0] = XL ;         // at level XL all rays stored as level XL rays
               BUFFER[c1+1] = I2F(ind) ;   // index of *original ray*, at level XL
               // We leave the original ray as the current one == (OTL, OTI, POS) remain unchanged.
               for(int i=sr; i<6; i++) BUFFER[c1+2+4*i] = -99.0f ; // mark the rest as unused
            }
            // NTRUE is correct for OTL but we store data for level XL
            // Therefore, BUFFER will contain  NTRUE * 0.25**(XL-OTL)  == NTRUE * 4**(OTL-XL)
            // ==>  the BUFFER values are   NTRUE(correct for RL) * 4.0**(OTL-XL), OTL and not RL !!!!
            // When sideray is taken from buffer and if it is located in a region with OTL>XL, the
            // beginning of the main loop will again rescale with  4.0*(XL-OTL)
            dx  =  pown(4.0, OTL-XL) ;           // was RL-XL but surely must be OTL-XL !!!!!
            for(int i=lid; i<CHANNELS; i+=ls) {
               BUFFER[c1+26+i] = NTRUE[i]*dx ;   // NTRUE scaled from RL to XL 
            }            
            NBUF += 1 ;            
         }
         
      } // for XL -- adding possible siderays
      
      
      // Subrays are added only when we are at leaf level. Also, when subrays are added, 
      // we continue with the original RL ray and immediately make a step forward. Therefore
      // there is no risk that we would add siderays multiple times.
# endif // adding siderays
      
      
      
      
      // global index -- must be now >=0 since we started with while(INDEX>=0) and just possibly went down a level
      INDEX = OFF[OTL]+OTI ;   
      
      // if not yet a leaf, jump back to start of the loop => another step down
      if (RHO[INDEX]<=0.0f) {
         continue ;
      }
      
      // we are now in a leaf node, ready to make the step
      OTLO   =  OTL ;
      
      
      
      dx     =  GetStepOT(&POS, &DIR, &OTL, &OTI, RHO, OFF, PAR, OTLO, NULL   , LEADING) ;  // POS is now the end of this step
      
      
      
      
# if 0   // @@ double check PL calculation ... PL[:] should be reduced to zero
      flo = -dx ;
      if (lid==0)  AADD(&(PL[INDEX]), flo) ;
# endif
      
      
      
      
      
      
      
      
      // if (RHO[INDEX]>CLIP) { // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // one workgroup per ray => can have barriers inside the condition
                  
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
# if 0
      barrier(CLK_LOCAL_MEM_FENCE) ;
      barrier(CLK_GLOBAL_MEM_FENCE) ;
      for(int i=lid; i<LOCAL; i+=LOCAL) {
         printf("[%3d]  %2d %6d   %8.4f %8.4f %8.4f   %2d\n", lid, OTL, OTI, POS.x, POS.y, POS.z, NBUF) ;
      }
# endif
      
      
      // with path length already being the same in all cells !   V /=8,  rays*4==length/2 /=2 =>  2/8=1/4
      weight    =  (dx/APL) *  VOLUME  *  pown(0.25f, OTLO) ;  // OTL=1 => dx/APL=0.5
      
      // INDEX is still the index for the cell where the step starts
      nu        =  NI[2*INDEX] ;
# if 0
      nb_nb     =  NI[2*INDEX+1] ;
# else
      nb_nb     =  max(1.0e-30f, NI[2*INDEX+1]) ; // $$$ KILL MASERS 
# endif
      
      // emitted photons divided between passing packages as before
      // this is accurate -- simulation sends exactly nu*Aul photons
      doppler   =  CLOUD[INDEX].x*DIR.x + CLOUD[INDEX].y*DIR.y + CLOUD[INDEX].z*DIR.z ;
# if (WITH_HALF==1)
      doppler  *=  0.002f ;  // half integer times 0.002f km/s
# endif
      
# if 0 // did not solve the oscillating-cell problem
      // ... one must avoid division by zero but lets move that to host side
      if (fabs(nb_nb)<1.0e-25f)  nb_nb   = 1.0e-25f ;
# endif        
      
      
      // tmp_tau   =  max(1.0e-35f, dx*nb_nb*GN) ;
      tmp_tau   =  dx*nb_nb*GN ;
      
      
      tmp_emit  =  weight * nu*Aul / tmp_tau ;  // GN include grid length [cm]
      shift     =  round(doppler/WIDTH) ;
# if (WITH_HALF==1)      //               sigma = 0.002f * w,   lookup table: sigma = SIGMA0 * SIGMAX^row
      row       =  clamp((int)round(log(CLOUD[INDEX].w*0.002f/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;
# else
      row       =  clamp((int)round(log(CLOUD[INDEX].w/SIGMA0)/log(SIGMAX)), 0, GNO-1) ;      
# endif
      profile   =  &GAU[row*CHANNELS] ;
      // avoid profile function outside profile channels LIM.x, LIM.y
      c1        =  max(LIM[row].x+shift, max(0, shift)) ;
      c2        =  min(LIM[row].y+shift, min(CHANNELS-1, CHANNELS-1+shift)) ;
      sum_delta_true = 0.0f ;
      all_escaped    = 0.0f ;
      
      
# if (WITH_CRT>0) // WITH_CRT
      sij = 0.0f ;
      // Dust optical depth and emission
      //   here escape = line photon exiting the cell + line photons absorbed by dust
      Ctau      =  dx     * CRT_TAU[INDEX] ;
      Cemit     =  weight * CRT_EMI[INDEX] ;      
      for(int i=c1; i<=c2; i++)  {
         pro    =  profile[i-shift] ;
         Ltau   =  tmp_tau*pro ;
         Ttau   =  Ctau + Ltau ;
         // tt     =  (1.0f-exp(-Ttau)) / Ttau ;
         tt     =  (fabs(Ttau)>0.01f) ?  ((1.0f-exp(-Ttau))/Ttau) : (1.0f-Ttau*(0.5f-0.166666667f*Ttau)) ;
         // ttt    = (1.0f-tt)/Ttau
         ttt    =  (1.0f-tt)/Ttau ;
         // Line emission leaving the cell   --- GL in profile
         Lleave =  weight*nu*Aul*pro * tt ;
         // Dust emission leaving the cell 
         Dleave =  Cemit *                     tt ;
         // SIJ updates, first incoming photons then absorbed dust emission
         sij   +=  A_b * pro*GN*dx * NTRUE[i]*tt ;
         // sij         += A_b * profile[i]*GN*Cemit*dx*(1.0f-tt)/Ttau ; // GN includes GL!
         sij   +=  A_b * pro*GN*dx * Cemit*ttt ;    // GN includes GL!
         // "Escaping" line photons = absorbed by dust or leave the cell
         all_escaped +=  Lleave  +  weight*nu*Aul*pro * Ctau * ttt ;
         // Total change of photons in the package
         NTRUE[i]     =  NTRUE[i]*exp(-Ttau) + Dleave + Lleave ;
      }  // loop over channels
      // RES[2*INDEX]    += sij ;            // division by VOLUME done in the solver (kernel)
      // RES[2*INDEX+1]  += all_escaped ;    // divided by VOLUME only oin Solve() !!!
      AADD(&(RES[2*INDEX]), sij) ;
      AADD(&(RES[2*INDEX+1]), all_escaped) ;
      
      
# else   // not  WITH_CRT ***************************************************************************************
      
      
      // because of c1, the same NTRUE elements may be updated each time by different work items...
      barrier(CLK_LOCAL_MEM_FENCE) ;    // local NTRUE elements possibly updated by different threads
      
      
      
      
#  if (WITH_ALI>0) // *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
      // Taylor approximations remain accurate as argument goes to zero !!!
      //              no change if using -- fabs(nb_nb)>1.0e-25f
      // if (fabs(nb_nb)>1.0e-30f) {  // $$$
      for(int i=c1+lid; i<=c2; i+=ls)  {
         w               =  tmp_tau*profile[i-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
         // absence affects strongly low-density regions
#   if 0
         factor          =  clamp(factor, -1.0e-12f, 1.0f) ;  // $$$
#   else
         factor          =  clamp(factor,  1.0e-30f, 1.0f) ;  // KILL MASERS $$$
#   endif
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;    // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed  ;          // ignore photons absorbed in emitting cell
         all_escaped    +=  escape ;             // sum of escaping photons over the profile
      }   // over channels
      SDT[lid] = sum_delta_true ;  AE[lid] = all_escaped ; // all work items save their own results
      barrier(CLK_LOCAL_MEM_FENCE) ;             // all agree on NTRUE, all know SDT and AE
      if (lid==0) {                              // lid=0 sums up and saves absorptions and escaped photons
         for(int i=1; i<LOCAL; i++) {  
            sum_delta_true += SDT[i] ;      all_escaped    +=  AE[i] ;     
         }
         all_escaped     =  clamp(all_escaped, 0.0001f*weight*nu*Aul, 0.9999f*weight*nu*Aul) ; // must be [0,1]
         // RES[2*INDEX]   +=  A_b * (sum_delta_true / nb_nb) ;
         // RES[2*INDEX+1] +=  all_escaped ;
         w  =  A_b * (sum_delta_true/nb_nb) ;
         AADD(&(RES[2*INDEX]),    w) ;
         AADD(&(RES[2*INDEX+1]),  all_escaped) ; 
      } // lid==0
      // }
      
#  else // else no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
      
      // if (isfinite(tmp_tau)==0) printf("tmp_tau not finite !!!\n") ;
      
      for(int i=c1+lid; i<=c2; i+=ls)  {
         w               =  tmp_tau*profile[i-shift] ;
         factor          =  (fabs(w)>0.01f) ? (1.0f-exp(-w)) : (w*(1.0f-w*(0.5f-0.166666667f*w))) ;
#   if 0
         factor          =  clamp(factor, -1.0e-12f, 1.0f) ;  // $$$
#   else
         factor          =  clamp(factor,  1.0e-30f, 1.0f) ;  // KILL MASERS $$$
#   endif
         escape          =  tmp_emit*factor ;    // emitted photons that escape current cell
         absorbed        =  NTRUE[i]*factor ;    // incoming photons that are absorbed
         NTRUE[i]       +=  escape-absorbed ;
         sum_delta_true +=  absorbed - escape ;  // later sum_delta_true +=  W*nu*Aul
      }   // over channels
      SDT[lid] = sum_delta_true ;
      barrier(CLK_LOCAL_MEM_FENCE) ;
      if (lid==0) {
         for(int i=1; i<LOCAL; i++)  sum_delta_true += SDT[i] ;    
         w  =   A_b  * ((weight*nu*Aul + sum_delta_true) / nb_nb)  ;
         // RES[INDEX] += w ;
         // AADD(&(RES[INDEX]), w) ;
         AADD((__global float*)(RES+INDEX), w) ;
      } 
#  endif // no ALI *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
      
# endif  // WITH OR WITHOUT CRT
      
      
      // } // RHO>CLIP !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      
      
      
      
      
      barrier(CLK_LOCAL_MEM_FENCE) ;
# if (BRUTE_COOLING>0)
      // total number of photons in the package as it exits the cell
      if (lid==0) {
         float cool = 0.0f ;
         for(int i=0; i<CHANNELS; i++) cool += NTRUE[i] ;
         COOL[INDEX] += cool ; // cooling of cell INDEX --- each work group distinct rays => no need for atomics
      }
# endif
      
      
      
      
      
      
      
      
      
      // Updates at the end of the step, POS has been already updated, OTL and OTI point to the new cell
      INDEX   =  (OTI>=0) ? (OFF[OTL]+OTI) : (-1) ;
      
# if (BRUTE_COOLING>0)  // heating of the next cell, once INDEX has been updated
      if (INDEX>=0) {
         if (lid==0)  COOL[INDEX] -= cool ; // heating of the next cell
      }
# endif
      
      if (INDEX>=0) {
         if (RHO[INDEX]<=0.0f) { // we stepped to a refined cell (GetStep goes only to a cell OTL<=OTLO)
            continue ;           // step down one level at the beginning of the main loop
         }
      }
      
      
      if (OTL<RL) {        // we are up to a level where this ray no longer exists
         INDEX=-1 ;        
      } else {      
         if (INDEX<0) {    // ray exits the cloud... possibly continues on the other side
            OTL    =  0 ;
            OTI    = -OTI-1 ;  // this was the cell where we exited, step ending at current POS
            INDEX  = -1 ;
            if (POS.x>1.0f) {
               if (LEADING!=0) {   POS.x = EPS ;    INDEX -= NX-1 ;   }
            }
            if (POS.x<0.0f) {
               if (LEADING!=1) {   POS.x = OMEPS ;  INDEX += NX-1 ;   }
            }
            if (POS.y>1.0f) {
               if (LEADING!=2) {   POS.y = EPS ;    INDEX -= NX*(NY-1) ;   }
            }
            if (POS.y<0.0f) {
               if (LEADING!=3) {   POS.y = OMEPS ;  INDEX += NX*(NY-1) ;   }
            }
            if (POS.z>1.0f) {
               if (LEADING!=4) {   POS.z = EPS ;    INDEX -= NX*NY*(NZ-1) ;   }
            }
            if (POS.z<0.0f) {
               if (LEADING!=5) {   POS.x = OMEPS ;  INDEX += NX*NY*(NZ-1) ;   }
            }
            if (INDEX>=0) {   // new level-0 ray started on the opposite side (may be a parent cell)
               RL = 0 ;  OTLO = 0 ; 
               // we had already barrier after the previous NTRUE update
               for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BG * DIRWEI ;
               barrier(CLK_LOCAL_MEM_FENCE) ;
# if (BRUTE_COOLING>0)
               if (lid==0) {
                  dx = BG*DIRWEI*CHANNELS ;  COOL[INDEX] -= dx ; // heating of the entered cell
               }
# endif
               continue ;
            }
         } // if INDEX<0
      }
      
      
      // rescale on every change of resolution
      if ((INDEX>=0)&&(OTL<OTLO)) {   // @s ray continues at a lower hierarchy level => NTRUE may have to be scaled
         dx = pown(4.0f, OTLO-OTL) ;  // scale on every change of resolution
         for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] *= dx ;     
         continue ;  // we went to lower level => this cell is a leaf
      }
      
      
      // if INDEX still negative, try to take a new ray from the buffer
      // 0   1     2  3  4  6     ...       NTRUE[CHANNELS] 
      // OTL OTI   x  y  z  RL    x y z RL                  
      if ((INDEX<0)&&(NBUF>0)) {            // NBUF>0 => at least one ray exists in the buffer
         barrier(CLK_GLOBAL_MEM_FENCE) ;    // all work items access BUFFER
         c1    =  (NBUF-1)*(26+CHANNELS) ;  // 8+CHANNELS elements per buffer entry
         OTL   =  (int)BUFFER[c1+0] ;       // OTL ...
         OTLO  =  OTL ;                     // ???
         OTI   =  F2I(BUFFER[c1+1]) ;       // and OTI of the ray that was split
         for(sr=5; sr>=0; sr--) {         
            dr    =  BUFFER[c1+2+4*sr] ;    // READL dr
            if (dr>-0.1f) break ;           // found a ray
         }
         POS.x   =  dr ;
         POS.y   =  BUFFER[c1+3+4*sr] ;  
         POS.z   =  BUFFER[c1+4+4*sr] ;
         RL      =  BUFFER[c1+5+4*sr] ;
         SID     =  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;   // cell in octet
         OTI     =  8*(int)(OTI/8) + SID ;        // OTI in the buffer was for the original ray that was split
         INDEX   =  OFF[OTL]+OTI ;                // global index -- must be >=0 !!!
         // copy NTRUE --- values in BUFFER have already been divided by four
         barrier(CLK_GLOBAL_MEM_FENCE) ;          // all have read BUFFER
         if (lid==0)  BUFFER[c1+2+4*sr] = -1.0f ; // mark as used
         if (sr==0)   NBUF -= 1 ;
         c1     +=  26 ;
         for(int i=lid; i<CHANNELS; i+=ls)  NTRUE[i] = BUFFER[c1+i] ;  // NTRUE correct for level OTL
         barrier(CLK_LOCAL_MEM_FENCE) ;         
         // note - this ray be inside a parent cell => handled at the beginnign of the main loop
      } // (INDEX<=0)&(NBUF>0)
      
      
   } // while INDEX>=0
   
}  // end of UpdateOT5()


#endif // WITH_OCTREE==5



