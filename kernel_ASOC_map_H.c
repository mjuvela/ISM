
// One is using either scalar ABS, SCA (without -D WITH_ABU == with single dust species)
// or pre-calculated sums over dust populations (with -D WITH_ABU, weighted by abundances)
// -D WITH_MSF has no effect on map making and thus vector ABS, SCA are never needed here

// #define EPS 3.0e-5f   --- slows down because of EPS steps (some cases)
// #define EPS 6.0e-5f   // enough to avoid slow-down??? --- 2016-06-28
// #define EPS 1.0e-4f   // enough to avoid slow-down??? --- 2016-06-28

#define EPS   2.5e-4f   // needed to reduce errors in maps -- 2016-08-14
#define PEPS  5.0e-4f


#if (NX>100)
// double3  POS ;
# define ONE   1.0
# define HALF  0.5
# define TWO   2.0
#else
// float3   POS ;
# define ONE   1.0f
# define HALF  0.5f
# define TWO   2.0f
#endif


// #define WITH_COLDEN -1

// NOTE: in kernel_SOCAMO_1.c EPS = 4e-4 == much larger!!

#define PI        3.1415926536f
#define TWOPI     6.2831853072f
#define TWOTHIRD  0.6666666667f
#define PIHALF    1.5707963268f

#define DEBUG 0

#if (OPT_IS_HALF>0)
# define GOPT(i) (vload_half(i,OPT))
#define OTYPE half
#else
#define GOPT(i) (OPT[i])
#define OTYPE float
#endif

#pragma OPENCL EXTENSION cl_khr_fp16: enable




int Angles2PixelRing(float phi, float theta) {
   // Convert angles to Healpix pixel index, theta=0.5*PI-latitude, phi=longitude.
   // The Healpix map must be in RING order.
   int     nl2, nl4, ncap, npix, jp, jm, ipix1 ;
   float   z, za, tt, tp, tmp ;
   int     ir, ip, kshift ;
   if ((theta<0.0f)||(theta>PI)) return -1 ;
   z   = cos(theta) ;
   za  = fabs(z) ;
   if (phi>=TWOPI)  phi -= TWOPI ;
   if (phi<0.0f)    phi += TWOPI ;
   tt  = phi / PIHALF ;    // in [0,4)
   nl2 = 2*NSIDE ;
   nl4 = 4*NSIDE ;
   ncap  = nl2*(NSIDE-1) ;   //  number of pixels in the north polar cap
   npix  = 12*NSIDE*NSIDE ;
   if (za<=TWOTHIRD) {  //  Equatorial region ------------------
      jp = (int)(NSIDE*(0.5f+tt-z*0.75f)) ;   // ! index of  ascending edge line
      jm = (int)(NSIDE*(0.5f+tt+z*0.75f)) ;   // ! index of descending edge line
      ir = NSIDE + 1 + jp - jm ;   // ! in {1,2n+1} (ring number counted from z=2/3)
      kshift = 0 ;
      if (ir%2==0) kshift = 1 ;    // ! kshift=1 if ir even, 0 otherwise
      ip = (int)( ( jp+jm - NSIDE + kshift + 1 ) / 2 ) + 1 ;   // ! in {1,4n}
      if (ip>nl4) ip -=  nl4 ;
      ipix1 = ncap + nl4*(ir-1) + ip ;
   } else {  // ! North & South polar caps -----------------------------
      tp  = tt - (int)(tt)  ;    // !MOD(tt,1.d0)
      tmp = sqrt( 3.0f*(1.0f - za) ) ;
      jp = (int)(NSIDE*tp         *tmp ) ;   // ! increasing edge line index
      jm = (int)(NSIDE*(1.0f - tp)*tmp ) ;   // ! decreasing edge line index
      ir = jp + jm + 1  ;          // ! ring number counted from the closest pole
      ip = (int)( tt * ir ) + 1 ;    // ! in {1,4*ir}
      if (ip>(4*ir)) ip -= 4*ir ;
      ipix1 = 2*ir*(ir-1) + ip ;
      if (z<=0.0f) {
         ipix1 = npix - 2*ir*(ir+1) + ip ;
      }
   }
   return  ( ipix1 - 1 ) ;    // ! in {0, npix-1} -- return the pixel index
}


bool Pixel2AnglesRing(const int ipix, float *phi, float *theta) {
   // Convert Healpix pixel index to angles (phi, theta), theta=0.5*pi-lat, phi=lon
   // Map must be in RING order.
   int    nl2, nl4, npix, ncap, iring, iphi, ip, ipix1 ;
   float  fact1, fact2, fodd, hip, fihip ;
   npix = 12*NSIDE*NSIDE;      // ! total number of points
   // if ((ipix<0)||(ipix>(npix-1))) return false ;
   ipix1 = ipix + 1 ;    // ! in {1, npix}
   nl2   = 2*NSIDE ;
   nl4   = 4*NSIDE ;
   ncap  = 2*NSIDE*(NSIDE-1) ;  // ! points in each polar cap, =0 for NSIDE =1
   fact1 = 1.5f*NSIDE ;
   fact2 = 3.0f*NSIDE*NSIDE  ;
   if (ipix1<=ncap) {   // ! North Polar cap -------------
      hip   = ipix1/2.0f ;
      fihip = (int)( hip ) ;
      iring = (int)( sqrt( hip - sqrt(fihip) ) ) + 1 ;  // ! counted from North pole
      iphi  = ipix1 - 2*iring*(iring - 1) ;
      *theta= acos( 1.0f - iring*iring / fact2 ) ;
      *phi  = (iphi - 0.5f) * PI/(2.0f*iring) ;
   } else {
      if (ipix1<=nl2*(5*NSIDE+1)) { // ! Equatorial region ------
         ip    = ipix1 - ncap - 1 ;
         iring = (int)( ip / nl4 ) + NSIDE ;   // ! counted from North pole
         iphi  = (ip%nl4) + 1 ;
         fodd  = 0.5f * (1 + (iring+NSIDE)%2) ;  // ! 1 if iring+NSIDE is odd, 1/2 otherwise
         *theta= acos( (nl2 - iring) / fact1 ) ;
         *phi  = (iphi - fodd) * PI /(2.0f*NSIDE) ;
      } else {   // ! South Polar cap -----------------------------------
         ip    = npix - ipix1 + 1 ;
         hip   = ip/2.0f ;
         fihip = (int) ( hip ) ;
         iring = (int)( sqrt( hip - sqrt(fihip) ) ) + 1 ;   // ! counted from South pole
         iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1)) ;
         *theta= acos( -1.0f + iring*iring / fact2 ) ;
         *phi  = (iphi - 0.5f) * PI/(2.0f*iring) ;
      }
   }
   return true ;
}


#if (DEBUG>0)
int CheckPos(float3 *pos, float3 *dir, 
             int *level, int *ind, __global float *DENS, __constant int *OFF,
             __global int *PAR, int tag) {
   // the position should be inside the cell!!
   if ((*level)==0) {
      int i, j, k ;
      i = (*ind) % NX ;   j = ((*ind)/NX)%NY ;   k = (*ind)/(NX*NY) ;
      if ((*ind)<0) { // left the model
         if (((*pos).x<=0.0f)||((*pos).y<=0.0f)||((*pos).z<=0.0f)) return 1; // ok
         if (((*pos).x>=NX)||((*pos).y>=NY)||((*pos).z>=NZ)) return 1 ; // ok
      }
      if (((*pos).x<i)||((*pos).x>(i+1.0f))||
          ((*pos).y<j)||((*pos).y>(j+1.0f))||
          ((*pos).z<k)||((*pos).z>(k+1.0f))) {
         printf("[%d] WRONG ROOT CELL %3d %3d %3d %8.5f %8.5f %8.5f  %5.2f %5.2f %5.2f **************\n", 
                tag, i, j, k, (*pos).x, (*pos).y, (*pos).z, (*dir).x, (*dir).y, (*dir).z) ;
         return 0 ;
      }
   } else {
      int sid = (*ind) % 8 ;
      if (((sid<4)&&((*pos).z>1.0f))||((sid>3)&&((*pos).z<1.0f))) {
         printf("[%d] WRONG OCTET CELL %d %7.5f %7.5f %7.5f **********************\n", 
                tag, sid, (*pos).x, (*pos).y, (*pos).z) ;
         return 0 ;
      }
      if (((sid%2==0)&&((*pos).x>1.0f))||((sid%2==1)&&((*pos).x<1.0f))) {
         printf("[%d] WRONG OCTET CELL %d %7.5f %7.5f %7.5f **********************\n",
                tag, sid, (*pos).x, (*pos).y, (*pos).z) ;
         return 0 ;
      }
      if (((sid%4<2)&&((*pos).y>1.0f))||((sid%4>1)&&((*pos).y<1.0f))) {
         printf("[%d] WRONG OCTET CELL %d %7.5f %7.5f %7.5f **********************\n",
                tag, sid, (*pos).x, (*pos).y, (*pos).z) ;
         return 0;
      }
   }
   // printf("OK  %2d %5d  -- %8.5f %8.5f %8.5f\n", (*level), (*ind), (*pos).x, (*pos).y, (*pos).z) ;
   return 1 ;
}
#endif



void IndexG(float3 *pos, int *level, int *ind, __global float *DENS, __constant int *OFF) {
   // Return cell level and index for given global position
   //    pos    =  on input global coordinates, returns either
   //              global coordinates (root grid) or relative within current octet
   //    level  =  hierarchy level on which the cell resides, 0 = root grid
   //    ind    =  index to global array of densities for cells at this level
   // Note: we need (not here) arrays that give for each cell its parent cell
   // first the root level
   *ind = -1 ;
   if (((*pos).x<=0.0f)||((*pos).y<=0.0f)||((*pos).z<=0.0f)) return  ;
   if (((*pos).x>=NX)  ||((*pos).y>=NY)  ||((*pos).z>=NZ)) return  ;
   *level = 0 ;
   *ind   = (int)floor((*pos).z)*NX*NY + (int)floor((*pos).y)*NX + (int)floor((*pos).x) ;
   // if ((*ind)>=(NX*NY*NZ)) printf("????? INVALID INDEX\n") ;
   // if density>0 this is a leaf, otherwise go through the levels
   if (DENS[*ind]>0.0f) return ;
   // before loop, update coordinates from root grid to sub-octet coordinates
   (*pos).x = 2.0f*fmod((*pos).x, 1.0f) ;
   (*pos).y = 2.0f*fmod((*pos).y, 1.0f) ;
   (*pos).z = 2.0f*fmod((*pos).z, 1.0f) ;
   float link ;
   while(1) {
      link   =       -DENS[OFF[*level]+(*ind)] ;
      *ind   = *(int*)(&link) ;      
      (*level)++ ;   
      *ind  += 4*(int)floor((*pos).z) + 2*(int)floor((*pos).y) + (int)floor((*pos).x) ; // cell in octet
      if (DENS[OFF[*level]+(*ind)]>0.0f) return ; // found a leaf
      // convert coordinates to those of sub-octet
      (*pos).x -= floor((*pos).x) ;
      (*pos).y -= floor((*pos).y) ;
      (*pos).z -= floor((*pos).z) ;
      *pos     *= 2.0f ;
   }
}



void Index(float3 *pos, int *level, int *ind, 
           __global float *DENS, __constant int *OFF, __global int *PAR) {
   // Return the level and index of a neighbouring cell based on pos
   //   (level, ind) are the current cell coordinates, pos the local coordinates
   //   on return these will be the new coordinates or ind=-1 == one has exited
   int sid ;
#if (NX>100)
   double3  POS ;
#else
   float3   POS ;
#endif
   
   POS.x = (*pos).x ;   POS.y = (*pos).y ;   POS.z = (*pos).z ;
   if (*level==0) {   // on root grid
      if ((POS.x<=0.0f)||(POS.x>=NX)||(POS.y<=0.0f)||(POS.y>=NY)||(POS.z<=0.0f)||(POS.z>=NZ)) {
         *ind = -1 ; return ;
      }
      *ind = (int)floor(POS.z)*NX*NY + (int)floor(POS.y)*NX + (int)floor(POS.x) ;
      if (DENS[*ind]>0.0f) return ;  // level 0 cell was a leaf -- we are done
      // ind is root grid cell, pos are root grid coordinates
      // convert to sub-octet coordinates
   } else {     // go UP until the position is inside the octet
      while ((*level)>0) {
         *ind = PAR[OFF[*level]+(*ind)-NX*NY*NZ] ;  *level -= 1 ;  // parent cell index
         if ((*level)==0) {       // arrived at root grid
            POS   *=  HALF ;   // convertion of sub-octet to global coordinates
            POS.x +=  (*ind)      % NX  ; 
            POS.y +=  ((*ind)/NX) % NY ; 
            POS.z +=  (*ind) / (NX*NY) ;
            if ((POS.x<=0.0f)||(POS.x>=NX)||(POS.y<=0.0f)||(POS.y>=NY)||
                (POS.z<=0.0f)||(POS.z>=NZ)) {
               *ind = -1 ; return ;  // we left the model volume
            }
            // the position is not necessarily in this cell, could be a neighbour!
            *ind = (int)floor(POS.z)*NX*NY + (int)floor(POS.y)*NX + (int)floor(POS.x) ;
            // printf("Arrived at root: %8.5f %8.5f %8.5f\n", POS.x, POS.y, POS.z) ;
            if (DENS[*ind]>0.0f) return ;  // found the cell on level 0
            // ind is root grid index, pos is root grid coordinates
            break ;
         } else {
            // this was a step from a sub-octet to a parent octet, parent cell == ind
            sid       = (*ind) % 8 ; // parent cell = current cell in current octet
            POS   *= HALF ;
            POS.x += sid % 2 ;  POS.y += (sid/2)%2  ;  POS.z += sid/4 ;
            // octet covering the position if coordinates are in [0,2]
            if ((POS.x>=0.0f)&&(POS.x<=2.0f)&&(POS.y>=0.0f)&&(POS.y<=2.0f)&&
                (POS.z>=0.0f)&&(POS.z<=0.0f)) break ;
            // level, ind in some octet, pos are coordinates within that octet
         }
      } // while 
   } // else
   
   
   float link ;
   while(DENS[OFF[*level]+(*ind)]<=0.0f) {  // loop until (level,ind) points to a leaf
      // convert to sub-octet coordinates -- same for transition from root or parent octet
      POS.x =  TWO*fmod(POS.x,ONE) ;
      POS.y =  TWO*fmod(POS.y,ONE) ;
      POS.z =  TWO*fmod(POS.z,ONE) ;
      link     =        -DENS[OFF[*level]+(*ind)] ;
      *ind     = *(int*)&link ;     
      *level  += 1 ;      
#if 0
      if ((*level)>6) {
         printf("LEVEL %d  %8.4f %8.4f %8.4f  ind %9d\n", *level, POS.x, POS.y, POS.z, *ind) ;
      }
#endif      
      // printf("    ... first in octet %d with sid %d  %8.6f %8.6f %8.6f\n", (*ind), (*ind) % 8, POS.x, POS.y, POS.z) ;
      *ind   += 4*(int)floor(POS.z)+2*(int)floor(POS.y)+(int)floor(POS.x) ; // subcell (suboctet)
      // printf("    ... actual cell %d with sid %d\n", (*ind), (*ind) % 8) ;
   }
   (*pos).x = POS.x ;  (*pos).y = POS.y ;  (*pos).z = POS.z ; 
   return ;
}




#if 1
// 2016-10-11 this copied from kernel_SOCAMO_1.c
//  the alternative routine failed for deep hierarchies (??)
float GetStep(float3 *POS, const float3 *DIR, int *level, int *ind, 
              __global float *DENS, __constant int *OFF, __global int*PAR) {
   // Calculate step to next cell, update level and ind for the next cell
   // Returns the step length in GL units (units of the root grid)
   float dx, dy, dz ;
   dx = ((((*DIR).x)>0.0f) ? ((1.0f+PEPS-fmod((*POS).x,1.0f))/((*DIR).x)) 
         : ((-PEPS-fmod((*POS).x,1.0f))/((*DIR).x))) ;
   dy = ((((*DIR).y)>0.0f) ? ((1.0f+PEPS-fmod((*POS).y,1.0f))/((*DIR).y)) 
         : ((-PEPS-fmod((*POS).y,1.0f))/((*DIR).y))) ;
   dz = ((((*DIR).z)>0.0f) ? ((1.0f+PEPS-fmod((*POS).z,1.0f))/((*DIR).z)) 
         : ((-PEPS-fmod((*POS).z,1.0f))/((*DIR).z))) ;
   // float dx0 = dx ;
   dx    =  min(dx, min(dy, dz)) ;
   
   *POS +=  dx*(*DIR) ;                    // update LOCAL coordinates - overstep by PEPS
   // step returned in units [GL] = root grid units
   dx     =  ldexp(dx, -(*level)) ;
   Index(POS, level, ind, DENS, OFF, PAR) ; // update (level, ind)
# if (DEBUG>1)
   if (get_local_id(0)==2) {
      if ((dx<1.0e-4)||(dx>1.733f)) {
         printf("  Step  dx %10.3e  dy %10.3e  dz %10.3e\n", dx, dy, dz) ;
         printf("  LEV %2d, IND %6d\n", *level, *ind) ;
         printf("  POS  %9.6f %9.6f %9.6f\n", (*POS).x, (*POS).y, (*POS).z) ;
         printf("  DIR  %9.6f %9.6f %9.6f\n", (*DIR).x, (*DIR).y, (*DIR).z) ;
      }
   }
# endif   
   return dx ;                 // step length [GL]
}
#else
float GetStep(float3 *POS, const float3 *DIR, int *level, int *ind, 
              __global float *DENS, __constant int *OFF, __global int*PAR) {
   // Calculate step to next cell, update level and ind for the next cell
   // Returns the step length in GL units (units of the root grid)
   // printf("STEP FROM %2d %5d %d  %7.5f %7.5f %7.5f\n", (*level), (*ind), (*ind)%8, (*POS).x, (*POS).y, (*POS).z) ;
   float dx, dy, dz ;
   // POS is in local coordinates - one cell is always 1.0 units
   dx = ((((*DIR).x)>0.0f) ? ((1.0f-fmod((*POS).x,1.0f))/((*DIR).x))
         : (-fmod((*POS).x,1.0f)/((*DIR).x))) ;
   dy = ((((*DIR).y)>0.0f) ? ((1.0f-fmod((*POS).y,1.0f))/((*DIR).y)) 
         : (-fmod((*POS).y,1.0f)/((*DIR).y))) ;
   dz = ((((*DIR).z)>0.0f) ? ((1.0f-fmod((*POS).z,1.0f))/((*DIR).z)) 
         : (-fmod((*POS).z,1.0f)/(*DIR).z)) ;
# if 0
   if (fabs((*DIR).x)<5.0e-6f) dx = 1e10f ;
   if (fabs((*DIR).y)<5.0e-6f) dy = 1e10f ;
   if (fabs((*DIR).z)<5.0e-6f) dz = 1e10f ;
# else
   // direction nearly along main axis => dx, dy, or dz zero => step becomes EPS
   // Added test on dx, dy, dz --- NEEDED FOR Healpix map???? to avoid 1e-5 steps
   // one of dx, dy, or dz could become zero, resulting in EPS steps
   // IS THIS NOT NEEDED IN kernel_SOCAMO_1.c ALSO ???
   if ((dx<1.0e-7f)||(fabs((*DIR).x)<2.0e-6f)) dx = 1e10f ;
   if ((dy<1.0e-7f)||(fabs((*DIR).y)<2.0e-6f)) dy = 1e10f ;
   if ((dz<1.0e-7f)||(fabs((*DIR).z)<2.0e-6f)) dz = 1e10f ;
# endif
   float step  =  min(dx, min(dy, dz)) + EPS ;
   
   // if (step>3.0f) printf("HUGE STEP %8.3f  level %d ind %9d\n", step, level, ind) ;
   
# if (DEBUG>1)
   if (get_local_id(0)==2) {
      if ((step<1.0e-4)||(step>1.733f)) {
         printf("  Step %10.3e    dx %10.3e  dy %10.3e  dz %10.3e\n", step, dx, dy, dz) ;
         printf("  LEV %2d, IND %6d\n", *level, *ind) ;
         printf("  POS  %9.6f %9.6f %9.6f\n", (*POS).x, (*POS).y, (*POS).z) ;
         printf("  DIR  %9.6f %9.6f %9.6f\n", (*DIR).x, (*DIR).y, (*DIR).z) ;
      }
   }
# endif
   *POS       +=  step*(*DIR) ;          // update coordinates in units of current level
   step       *=  pow(0.5f, *level) ;    // step returned in units [GL]
   Index(POS, level, ind, DENS, OFF, PAR) ;   
   return step ;                 // step length in units [GL]
}
#endif






__kernel void Mapping(
                      const      float    DX,     //  0 - pixel size in grid units
                      const      int2     NPIX,   //  1 - map dimensions (square map)
                      __global   float   *MAP,    //  2 - 4 * NPIX * NPIX * LEVELS !!!!!!!!!!!!!!
                      __global   float   *EMIT,   //  3 - emitted
                      const      float3   DIR,    //  4 - direction of observer
                      const      float3   RA,     //  5
                      const      float3   DE,     //  6
                      constant   int     *LCELLS, //  7 - number of cells on each level
                      constant   int     *OFF,    //  8 - index of first cell on each level
                      __global   int     *PAR,    //  9 - index of parent cell
                      __global   float   *DENS,   // 10 - density and hierarchy
                      const      float    ABS,    // 11
                      const      float    SCA,    // 12
                      const      float3   CENTRE, // 13
                      const      float3   INTOBS, // 14 position of internal observer
                      __global   OTYPE   *OPT,    // 15 OPT = ABS, SCA array float float2
                      __global   float   *COLDEN  // 16 column density image
                     )
{
   const int id   = get_global_id(0) ;  // one work item per map pixel
   if (id>=(NPIX.x*NPIX.y)) return ;
   float DTAU, TAU=0.0f, colden = 0.0f ;
   
   float PHOTONS[LEVELS] ; // at maximum from 8 different levels
   for(int ilev=0; ilev<LEVELS; ilev++) PHOTONS[ilev] = 0.0f ;
   float3 POS, TMP ;
   float  sx, sy, sz ;
   int    ind, level, oind, olevel ;
   int    i = id % NPIX.x ;   // longitude runs faster
   int    j = id / NPIX.x ;
   
   if (INTOBS.x>-1e10) {  // perspective image from position inside the model
      float phi   = TWOPI*i/(float)(NPIX.x) ;  // NPIX.x = longitude points
      // Move longitude zero to the middle of the figure
      phi        += PI ;
      float pix   = TWOPI/NPIX.x ;             // pixel size in radians
      float theta = pix*(j-(NPIX.y-1)/2) ;     // latitude
      POS   =  INTOBS ;      // location of the observer inside the model (root coordinates)
      // TMP is away from the observer, towards GC == along +X
      //   lon=0,  lat=90    ==  +Z
      //   lon=0,  lat=0     ==  +X
      //   lon=90, lat=0     ==  +Y
      // ....
      //  2019-04-12:  GC is towards -X
      // 
      TMP.x = -cos(theta)*sin(phi) ; 
      TMP.y = -cos(theta)*cos(phi) ;
      TMP.z = +sin(theta) ;
      if (fabs(TMP.x)<1.0e-5f)      TMP.x  = 1.0e-5f ;
      if (fabs(TMP.y)<1.0e-5f)      TMP.y  = 1.0e-5f ;
      if (fabs(TMP.z)<1.0e-5f)      TMP.z  = 1.0e-5f ;
      if (fmod(POS.x,1.0f)<1.0e-5f) POS.x += 2.0e-5f ;
      if (fmod(POS.y,1.0f)<1.0e-5f) POS.y += 2.0e-5f ;
      if (fmod(POS.z,1.0f)<1.0e-5f) POS.z += 2.0e-5f ;
   } else {               // normal external map
      // 2016-10-02  changed direction of X axis to be consistent with polarisation maps
      // 2019-04-12  RA axis is pointing right...
      POS.x = CENTRE.x + (i-0.5f*(NPIX.x-1))*DX*RA.x + (j-0.5f*(NPIX.y-1))*DX*DE.x ;
      POS.y = CENTRE.y + (i-0.5f*(NPIX.x-1))*DX*RA.y + (j-0.5f*(NPIX.y-1))*DX*DE.y ;
      POS.z = CENTRE.z + (i-0.5f*(NPIX.x-1))*DX*RA.z + (j-0.5f*(NPIX.y-1))*DX*DE.z ;   
      // find position on the front surface (if exists)
      POS -=  (NX+NY+NZ)*DIR ;   // move behind, step to front surface is positive
      // find last crossing -- DIR pointing towards observer
      if (DIR.x>=0.0f)    sx = (NX      - POS.x) / (DIR.x+1.0e-10f) - EPS ;
      else                sx = (0.0f    - POS.x) /  DIR.x - EPS ;
      if (DIR.y>=0.0f)    sy = (NY      - POS.y) / (DIR.y+1.0e-10f) - EPS ; 
      else                sy = (0.0f    - POS.y) /  DIR.y - EPS ;
      if (DIR.z>=0.0f)    sz = (NZ      - POS.z) / (DIR.z+1.0e-10f) - EPS ;
      else                sz = (0.0f    - POS.z) /  DIR.z - EPS ;
      // select largest value that still leaves POS inside the cloud (cloud may be missed)
      TMP = POS + sx*DIR ;
      if ((TMP.x<=0.0f)||(TMP.x>=NX)||(TMP.y<=0.0f)||(TMP.y>=NY)||(TMP.z<=0.0f)||(TMP.z>=NZ)) sx = -1e10f ;
      TMP = POS + sy*DIR ;
      if ((TMP.x<=0.0f)||(TMP.x>=NX)||(TMP.y<=0.0f)||(TMP.y>=NY)||(TMP.z<=0.0f)||(TMP.z>=NZ)) sy = -1e10f ;
      TMP = POS + sz*DIR ;
      if ((TMP.x<=0.0f)||(TMP.x>=NX)||(TMP.y<=0.0f)||(TMP.y>=NY)||(TMP.z<=0.0f)||(TMP.z>=NZ)) sz = -1e10f ;
      //
      sx    =  max(sx, max(sy, sz)) ;
      POS   =  POS + sx*DIR ;
      TMP   = -DIR ; // DIR towards observer, TMP away from the observer
   }
   
   // Index
   IndexG(&POS, &level, &ind, DENS, OFF) ;
   // printf("POS %8.4f %8.4f %8.4f  ind %d\n", POS.x, POS.y, POS.z, ind) ;
   
#if (DEBUG>0)
   CheckPos(&POS, &DIR, &level, &ind, DENS, OFF, PAR, 1) ;
#endif
   while (ind>=0) {
      oind    = OFF[level]+ind ;    // original cell, before step out of it
      olevel  = level ;
      sx      = GetStep(&POS, &TMP, &level, &ind, DENS, OFF, PAR) ;
#ifdef USE_ABU
      DTAU    = sx*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;  // sx in global units
#else
      DTAU    = sx*DENS[oind]*(SCA+ABS) ;  // sx in global units
#endif      
      // count emission from individual levels of the hierarchy
      if (DTAU<1.0e-3f) {
         PHOTONS[olevel] += exp(-TAU) *  (1.0f-0.5f*DTAU)        * sx * EMIT[oind]*DENS[oind] ;
      } else {
         PHOTONS[olevel] += exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * sx * EMIT[oind]*DENS[oind] ;
      }      
      TAU    += DTAU ;
      colden += sx*DENS[oind] ;
      // ind  = IndexG(POS, &level, &ind, DENS, OFF) ; --- should not be necessary!
   }   
   // i = longitude, j = latitude = runs faster
   for(int ilev=0; ilev<LEVELS; ilev++) {
      MAP[ilev*NPIX.x*NPIX.y+id] = PHOTONS[ilev] ;
   }
   // printf("colden %10.3e\n", colden) ;
#if (WITH_COLDEN>0)
   COLDEN[id] = colden * LENGTH ;
#endif
}









__kernel void HealpixMapping(
                             const      float    DX,     //  0 - pixel size in grid units
                             const      int2     NPIX,   //  1 - map dimensions (square map)
                             __global   float   *MAP,    //  2 - 4 * NPIX * NPIX
                             __global   float   *EMIT,   //  3 - emitted
                             const      float3   DIR,    //  4 - direction of observer
                             const      float3   RA,     //  5
                             const      float3   DE,     //  6
                             constant   int     *LCELLS, //  7 - number of cells on each level
                             constant   int     *OFF,    //  8 - index of first cell on each level
                             __global   int     *PAR,    //  9 - index of parent cell
                             __global   float   *DENS,   // 10 - density and hierarchy
                             const      float    ABS,    // 11
                             const      float    SCA,    // 12
                             const      float3   CENTRE, // 13
                             const      float3   INTOBS, // 14 position of internal observer
                             __global float *COLDEN      // 15 colden
                            )
{
   const int id   = get_global_id(0) ;   // one work item per map pixel
   if (id>=(12*NSIDE*NSIDE)) return ;    // NPIX.x == NSIDE
   float  DTAU, TAU=0.0f, PHOTONS=0.0f, colden = 0.0f ;   
   float3 POS, TMP ;
   float  dx ;
   int    ind, level, oind ;
   float  theta, phi ;
   Pixel2AnglesRing(id, &phi, &theta) ;  // phi=longitude, theta=0.5*pi-latitude
   //  2019-04-12 GC is towards -X !!!
   //  2019-05-08 Note that TMP is a step from the observer (map pixel) towards the model !!
   //     the map centre is GC, (lon,lat)=(0,0) where TMP=[-1,0,0]
   // (lon,lat)=(0,0) =>   (phi,theta)=(0,0.5*pi)
   TMP.x = -sin(theta)*cos(phi) ;  //  (0,0.5*pi) =>  -1
   TMP.y = -sin(theta)*sin(phi) ;  //             =>   0
   TMP.z = +cos(theta) ;           //             =>   0
   if (fabs(TMP.x)<1.0e-5f)      TMP.x  = -1.0e-5f ;
   if (fabs(TMP.y)<1.0e-5f)      TMP.y  = -1.0e-5f ;
   if (fabs(TMP.z)<1.0e-5f)      TMP.z  = -1.0e-5f ;
   POS   =  INTOBS ;      // location of the observer inside the model (root coordinates)
   if ((fmod(POS.x,1.0f)<1.0e-5f)||(fmod(POS.x,1.0f)<0.99999f)) POS.x += 2.0e-5f ;
   if ((fmod(POS.y,1.0f)<1.0e-5f)||(fmod(POS.y,1.0f)<0.99999f)) POS.y += 2.0e-5f ;
   if ((fmod(POS.z,1.0f)<1.0e-5f)||(fmod(POS.z,1.0f)<0.99999f)) POS.z += 2.0e-5f ;
   IndexG(&POS, &level, &ind, DENS, OFF) ;
   while (ind>=0) {
      oind    = OFF[level]+ind ;           // original cell, before step out of it
      dx      = GetStep(&POS, &TMP, &level, &ind, DENS, OFF, PAR) ;
#ifdef USE_ABU
      DTAU    = dx*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;  // dx in global units
#else
      DTAU    = dx*DENS[oind]*(SCA+ABS) ;  // dx in global units
#endif
      if (DTAU<1.0e-3f) {
         PHOTONS += exp(-TAU) *  (1.0f-0.5f*DTAU)        * dx * EMIT[oind]*DENS[oind] ;
      } else {
         PHOTONS += exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * dx * EMIT[oind]*DENS[oind] ;
      }
      TAU    += DTAU ;
      colden += dx*DENS[oind] ;
   }   
   MAP[id] = PHOTONS ;   // printf("PHOTONS %10.3e\n", PHOTONS) ;
   COLDEN[id] = colden*LENGTH ;
}






#if (POLSTAT==0)

__kernel void PolHealpixMapping( // HEALPIX POLARISATION MAP
                                 const      float    DX,      //  0 - pixel size in grid units
                                 const      int2     NPIX,    //  1 - map dimensions (square map)
                                 __global   float   *MAP,     //  2 - I[], Q[], U[], column density
                                 __global   float   *EMIT,    //  3 - emitted
                                 const      float3   DIR,     //  4 - direction of observer
                                 const      float3   RA,      //  5
                                 const      float3   DE,      //  6
                                 constant   int     *LCELLS,  //  7 - number of cells on each level
                                 constant   int     *OFF,     //  8 - index of first cell on each level
                                 __global   int     *PAR,     //  9 - index of parent cell
                                 __global   float   *DENS,    // 10 - density and hierarchy
                                 const      float    ABS,     // 11
                                 const      float    SCA,     // 12
                                 const      float3   CENTRE,  // 13
                                 const      float3   INTOBS,  // 14 position of internal observer
                                 __global   float   *Bx,      // 15 magnetic field vectors
                                 __global   float   *By,      // 16
                                 __global   float   *Bz,      // 17
                                 __global   OTYPE   *OPT,     // 18 ABS + SCA vectors
                                 const      float    Y_SHEAR  // 18 xy periodicity, shear in y direction
                               )
{
   const int id   = get_global_id(0) ;  // one work item per map pixel
   int npix = 12*NSIDE*NSIDE ;
   if (id>=npix) return ;
   float DTAU, TAU=0.0f, colden = 0.0f, dens ;
   float3 PHOTONS ;
   float3 POS, HDIR ;
   float  sx, sz ;
   int    ind, level, oind, olevel ;
   float  phi, theta, los=0.0f ;
   float3 BN  ;
   PHOTONS.x = 0.0f ;  PHOTONS.y = 0.0f ;  PHOTONS.z = 0.0f ;
# if (INTERPOLATE>0)
   float3 MPOS, mpos ; 
   int i0, j0, k0, mlevel, mind ;
   float sum, w, weight, delta ;
# endif   
   Pixel2AnglesRing(id, &phi, &theta) ;
   HDIR.x = +sin(theta)*cos(phi) ;
   HDIR.y = +sin(theta)*sin(phi) ;
   HDIR.z = -cos(theta) ;
   if (fabs(HDIR.x)<1.0e-5f)      HDIR.x  = 1.0e-5f ;
   if (fabs(HDIR.y)<1.0e-5f)      HDIR.y  = 1.0e-5f ;
   if (fabs(HDIR.z)<1.0e-5f)      HDIR.z  = 1.0e-5f ;
   POS   =  INTOBS ;      // location of the observer inside the model (root coordinates)
   if ((fmod(POS.x,1.0f)<1.0e-5f)||(fmod(POS.x,1.0f)<0.99999f)) POS.x += 2.0e-5f ;
   if ((fmod(POS.y,1.0f)<1.0e-5f)||(fmod(POS.y,1.0f)<0.99999f)) POS.y += 2.0e-5f ;
   if ((fmod(POS.z,1.0f)<1.0e-5f)||(fmod(POS.z,1.0f)<0.99999f)) POS.z += 2.0e-5f ;
   IndexG(&POS, &level, &ind, DENS, OFF) ;
   
   float3 HRA, HDE ;  // HRA points left, HDE points to north
   HRA.x  =  -sin(phi) ;
   HRA.y  =  +cos(phi) ;
   HRA.z  =   0.0f ;
   HDE.x  =  -cos(theta)*cos(phi) ;
   HDE.y  =  -cos(theta)*sin(phi) ;
   HDE.z  =  +sin(theta) ;
   
   
# if (DEBUG>0)
   CheckPos(&POS, &HDIR, &level, &ind, DENS, OFF, PAR, 1) ;
# endif
   
   float p = p00 ;  // constant p0 or encoded in the length of the B vector
   
   while (ind>=0) {
      oind    = OFF[level]+ind ;    // original cell, before step out of it
      olevel  = level ;
# if (INTERPOLATE>0)
      MPOS    = POS ;   // position before taking the step
# endif
      sx      = GetStep(&POS, &HDIR, &level, &ind, DENS, OFF, PAR) ;
      dens    = DENS[oind] ;
      

      
# if (INTERPOLATE==1)
      // Only for regular Cartesian grid: use interpolated density at the centre
      // position of the step. This uses only 4 points.
      MPOS     =  MPOS + 0.5f*sx*HDIR ;   // sx in [GL], root grid units
      i0       =  clamp((int)floor(MPOS.x), 0, NX-1) ;
      j0       =  clamp((int)floor(MPOS.y), 0, NY-1) ;
      k0       =  clamp((int)floor(MPOS.z), 0, NZ-1) ;
      MPOS.x   =  fmod(MPOS.x, 1.0f) - 0.5f ;    // put (i0,j0,k0) to origin
      MPOS.y   =  fmod(MPOS.y, 1.0f) - 0.5f ; 
      MPOS.z   =  fmod(MPOS.z, 1.0f) - 0.5f ;
      // interpolation using only 4 points
      sum      =  (3.0f-fabs(MPOS.x)-fabs(MPOS.y)-fabs(MPOS.z)) * dens ;
      if (MPOS.x>0.0f) { // in x, take one point from below
         sum +=   MPOS.x  *  DENS[k0*NX*NY+j0*NX+max(0, i0-1)] ;
      } else {           // MPOS below (i0,j0,k0) => other point at i0+1
         sum +=  -MPOS.x  *  DENS[k0*NX*NY+j0*NX+min(i0+1,NX-1)] ;
      }
      if (MPOS.y>0.0f) {
         sum +=  +MPOS.y  *  DENS[k0*NX*NY+max(j0-1, 0)*NX+i0] ;
      } else {
         sum +=  -MPOS.y  *  DENS[k0*NX*NY+min(j0+1,NY-1)*NX+i0] ;
      }
      if (MPOS.z>0.0f) {
         sum +=  +MPOS.z  *  DENS[max(k0-1,0)*NX*NY+j0*NX+i0] ;
      } else {
         sum +=  -MPOS.z  *  DENS[min(k0+1,NZ-1)*NX*NY+j0*NX+i0] ;
      }
      dens = 0.333333f*sum ;
# endif
     
      
      
#if (INTERPOLATE==2)
      // Only for regular Cartesian grid. This uses 3x3x3 points, works ok?
      MPOS     =  MPOS + 0.5f*sx*HDIR ;
      i0       =  (int)floor(MPOS.x) ;
      j0       =  (int)floor(MPOS.y) ;
      k0       =  (int)floor(MPOS.z) ;
      sum      =  0.0f ;
      weight   =  0.0f ;
      for(int k=max(0, k0-1); k<min(k0+2, NZ); k++) {
         for(int j=max(0, j0-1); j<min(j0+2, NY); j++) {
            for(int i=max(0, i0-1); i<min(i0+2, NX); i++) {
               mpos.x   =  MPOS.x - (i+0.5f) ;    // distance to that other cell
               mpos.y   =  MPOS.y - (j+0.5f) ;
               mpos.z   =  MPOS.z - (k+0.5f) ;
               w       =  1.0f / (0.1f+length(mpos)) ;
               weight +=  w ;
               sum    +=  w * DENS[k*NY*NX+j*NX+i] ; 
            }
         }
      }
      dens = sum/weight ;
#endif
      

      
#if (INTERPOLATE==3)
      // This might still ~work also for hierarchical grids. Uses 3x3x3 points.
      MPOS     =  MPOS + 0.5f*sx*HDIR ;  // centre point for the current step
      sum      =  0.0f ;
      weight   =  0.0f ;
      delta    =  pow(0.5f, olevel) ;    // current cell size -> +-delta is a neighbour?
      for(int k=-1; k<2; k++) {
         for(int j=-1; j<2; j++) {
            for(int i=-1; i<2; i++) {
               mpos.x   =  MPOS.x +  i*delta ;    // some position in global coordinates
               mpos.y   =  MPOS.y +  j*delta ;
               mpos.z   =  MPOS.z +  k*delta ;
               IndexG(&mpos, &mlevel, &mind, DENS, OFF) ;
               if (mind>=0) {
                  w       =  1.0f/sqrt(0.2f+i*i+j*j+k*k) ;
                  weight +=  w ;
                  sum    +=  w * DENS[OFF[mlevel]+mind] ;
               }
            }
         }
      }
      dens = sum/weight ;
#endif
      
      
#ifdef USE_ABU
      DTAU    = sx*dens*(GOPT(2*oind)+GOPT(2*oind+1)) ;  // sx in global units
#else
      DTAU    = sx*dens*(SCA+ABS) ;  // sx in global units
#endif
      los    += sx ;
#if 1
      if (los>MAXLOS) {
         ind = -1 ;  POS.z = -1.0f ;
         sx  =  MAXLOS-(los-sx) ;
      }
#endif
      // the angles
      // for LOS along x,  Psi = atan2(By, Bz), gamma  = atan2(Bx, sqrt(By*By+Bz*Bz)) ;
      //   Psi    = angle east of north, in the plane of the sky --- full 2*pi !!!
      //   Psi is in IAU convention   tan(2*Psi) = U/Q, Psi = 0.5*atan2(U,Q)
      //   gamma  = angle away from the POS = 90 - angle wrt. HDIR
      //    cos(gamma) = sin(complement)
# if 0 // ---
      BN  =  normalize(B[oind]) ;
#  if (POLRED>0)   // using polarisation reduction encoded in B vector length
      p   =  length(B[oind]) ;
#  endif      
# else  // ---
      BN.x = Bx[oind] ;  BN.y = By[oind] ;  BN.z = Bz[oind] ;
#  if (POLRED>0)
      p = length(BN) ;
#  endif
      BN = normalize(BN) ;
# endif // ---
      // Psi angle from North to East -- assumes that RA points to left
      // add 0.5*PI so that this becomes angle of polarised emission, not of B
      float Psi  =  0.5*PI+atan2(dot(BN, HRA), dot(BN, HDE)) ; // ANGLE FOR POLARISED EMISSION, NOT B
      float cc   =  0.99999f - 0.99998f * dot(BN, HDIR)*dot(BN, HDIR) ; // cos(gamma)^2
      if (DTAU<1.0e-3f) {
         sz = exp(-TAU) *  (1.0f-0.5f*DTAU)        * sx * EMIT[oind]*dens ;
      } else {
         sz = exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * sx * EMIT[oind]*dens ;
      }
      
      if (los<MINLOS) continue ;  // do not register yet
      
# if (LEVEL_THRESHOLD>0) // ---------------------------------------------------------------------------------
      // if (id==9999) printf("level %2d\n", olevel) ;
      if (olevel>=LEVEL_THRESHOLD) {
         PHOTONS.x  +=      sz * (1.0f-p*(cc-0.6666667f)) ;                // I
         PHOTONS.y  +=  p * sz * cos(2.0f*Psi)*cc ; // Q
         PHOTONS.z  +=  p * sz * sin(2.0f*Psi)*cc ; // U   -- IAU convention, Psi East of North
      }  else {
         ;  // if (id==9999) printf("SKIP\n") ;
      }
# else // no threshold  --------------------------------------------------------------------------------------
      PHOTONS.x  +=      sz * (1.0f-p*(cc-0.6666667f)) ;                     // I
      PHOTONS.y  +=  p * sz * cos(2.0f*Psi)*cc ;       // Q
      PHOTONS.z  +=  p * sz * sin(2.0f*Psi)*cc ;       // U   -- IAU convention, Psi East of North
# endif // LEVEL_THRESHOLD------------------------------------------------------------------------------------
      
      TAU     += DTAU ;
      colden  += sx*dens ;
      // colden  += sx ;
      
      // ind  = IndexG(POS, &level, &ind, DENS, OFF) ; --- should not be necessary!
      
      
      if (Y_SHEAR!=0.0f) {
         // We are dealing with a shearing box simulation where we assume periodicity in the
         // x and y directions, having additionally shear Y_SHEAR [root grid cells] between
         // the high and the low x edges
         //  ==>  ray continues from x=0 to x=NX-1 but y-coordinate is shifted by -Y_SHEAR
         //       ray continues from x=NX-1 to x=0 but y-coordinate is shifted by +Y_SHEAR
         //  left = -Y_SHEAR, right = +Y_SHEAR
         if ((ind<0)&&(los<MAXLOS)) {  // if we reach MAXLOS, do not try to continue
            // printf("EXIT   %8.4f %8.4f %8.4f\n", POS.x, POS.y, POS.z) ;
            if ((POS.z>0.0f)&&(POS.z<NZ)) {  // ray exited but not on the z boundaries
               if (POS.y<0.0f) POS.y = NY-PEPS ;
               if (POS.y>NY)   POS.y = +PEPS ;
               if (POS.x<0.0f) {
                  POS.x = NX-PEPS ;    POS.y = fmod(POS.y+NY-Y_SHEAR, (float)NY) ;
                  // ok -- density looks continuous along the path when yshift included
                  // IndexG(&POS, &level, &ind, DENS, OFF) ;            
                  // printf("LEFT    %10.3e -> %10.3e\n", DENS[oind], DENS[ind]) ;
               }
               if (POS.x>NX)   {
                  POS.x = +PEPS   ;    POS.y = fmod(POS.y   +Y_SHEAR, (float)NY) ;
                  // IndexG(&POS, &level, &ind, DENS, OFF) ;            
                  // printf("RIGHT   %10.3e -> %10.3e\n", DENS[oind], DENS[ind]) ;
               }
               IndexG(&POS, &level, &ind, DENS, OFF) ;            
            }         
         }
      }
            
   } // while ind>=0
   
   
   // i = longitude, j = latitude = runs faster
   MAP[0*npix+id] = PHOTONS.x ;  // I
   MAP[1*npix+id] = PHOTONS.y ;  // Q
   MAP[2*npix+id] = PHOTONS.z ;  // U
   MAP[3*npix+id] = colden * LENGTH ; // LENGTH == GL [cm]
   //  Q = cos(2*Psi), U = sin(2*Psi)
   //  tan(2*Chi) = U/Q
   // we rotated Psi by 0.5*pi to make this polarisation angle, not angle of B
   
   // printf("colden %10.3e --- %10.3e %10.3e %10.3e\n", colden, PHOTONS.x, PHOTONS.y, PHOTONS.z) ;
}






#else




// Calculate not I, Q, U, N  but  rT, rI, jT, jI !!
//  *T =  sqrt[   sum( w * (psi-<psi>)^2 ) / sum(w)  ] ,    weight w = rho or emission
//            <psi> is the density weighted mean along the line of sight
//            <psi> = Sum(n*psi) / sum(n)
//      -- this is dispersion of Psi angles along the line of sight
//  *I =  arccos[  sqrt(  sum(w*cos^2 gamma) / sum(w) ) ]
// See Eqs. (11)-(14), Chen et al. 2016, arXiv-1605.00648
// 
// Vaisala, Planck
//      S(r, delta) = sqrt[  (1/N) * sum( (psi(r)-psi(r+delta))^2 ) ]
//      sum over [0.5*delta, 1.5*delta] annulus
//      this is from the final map, nothing to do with LOS integration == post-processing of (Q,U)


__kernel void PolHealpixMapping( // HEALPIX POLSTAT MAP
                                 const      float    DX,     //  0 - pixel size in grid units
                                 const      int2     NPIX,   //  1 - map dimensions
                                 __global   float   *MAP,    //  2 - rT, rI, jT, jI
                                 __global   float   *EMIT,   //  3 - emitted
                                 const      float3   DIR,    //  4 - direction of observer
                                 const      float3   RA,     //  5
                                 const      float3   DE,     //  6
                                 constant   int     *LCELLS, //  7 - number of cells on each level
                                 constant   int     *OFF,    //  8 - index of first cell on each level
                                 __global   int     *PAR,    //  9 - index of parent cell
                                 __global   float   *DENS,   // 10 - density and hierarchy
                                 const      float    ABS,    // 11
                                 const      float    SCA,    // 12
                                 const      float3   CENTRE, // 13
                                 const      float3   INTOBS, // 14 position of internal observer
                                 __global   float   *Bx,     // 15 magnetic field vectors
                                 __global   float   *By,     // 16 magnetic field vectors
                                 __global   float   *Bz      // 17 magnetic field vectors
                               )
{
   const int id   = get_global_id(0) ;  // one work item per map pixel
   int npix = 12*NSIDE*NSIDE ;
   if (id>=npix) return ;
   float DTAU, TAU=0.0f ;
   // sum(rho), sum(rho*Psi), sum(rho*cos^2 gamma)
   // sum(j),   sum(j*Psi),   sum(j  *cos^2 gamma)
   float3 POS, POS0, HDIR, BB ;
   float  sx, sz ;
   int    ind, level, oind, olevel ;
   float  phi, theta, los=0.0f ;
   
   Pixel2AnglesRing(id, &phi, &theta) ;
   HDIR.x = +sin(theta)*cos(phi) ;
   HDIR.y = +sin(theta)*sin(phi) ;
   HDIR.z =  cos(theta) ;
   if (fabs(HDIR.x)<1.0e-5f)      HDIR.x  = 1.0e-5f ;
   if (fabs(HDIR.y)<1.0e-5f)      HDIR.y  = 1.0e-5f ;
   if (fabs(HDIR.z)<1.0e-5f)      HDIR.z  = 1.0e-5f ;
   POS   =  INTOBS ;      // location of the observer inside the model (root coordinates)
   if ((fmod(POS.x,1.0f)<1.0e-5f)||(fmod(POS.x,1.0f)<0.99999f)) POS.x += 2.0e-5f ;
   if ((fmod(POS.y,1.0f)<1.0e-5f)||(fmod(POS.y,1.0f)<0.99999f)) POS.y += 2.0e-5f ;
   if ((fmod(POS.z,1.0f)<1.0e-5f)||(fmod(POS.z,1.0f)<0.99999f)) POS.z += 2.0e-5f ;
   IndexG(&POS, &level, &ind, DENS, OFF) ;   
# if (DEBUG>0)
   CheckPos(&POS, &HDIR, &level, &ind, DENS, OFF, PAR, 1) ;
# endif
   POS0 = POS ;
   
   // first pass through the cloud, calculate  sum(w*cos^2 gamma), 
   // and Q and U that will define the values of  <Psi>
   float sR=0.0f, sJ=0.0f, sRG=0.0f, sJG=0.0f, RQ=0.0f, RU=0.0f, JQ=0.0f, JU=0.0f, sRP=0.0f, sJP=0.0f, d=0.0f, PR=0.0f ;
   TAU = 0.0f ;
   
   while (ind>=0) {
      oind     =  OFF[level]+ind ;    // original cell, before step out of it
      olevel   =  level ;
      sx       =  GetStep(&POS, &HDIR, &level, &ind, DENS, OFF, PAR) ;
      los     +=  sx ;
      if (los>MAXLOS) {
         ind = -1 ;  POS.z = -1.0f ;
         sx  = MAXLOS - (los-sx)
      }
# ifdef USE_ABU
      DTAU     =  sx*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;  // sx in global units
# else
      DTAU     =  sx*DENS[oind]*(SCA+ABS) ;  // sx in global units
# endif
# if 0
      BB  =  B[oind] ;
# else
      BB.x = Bx[oind] ;  BB.y = By[oind] ;  BB.z = Bz[oind] ;
# endif
# if (POLRED>0)
      PR    =  length(BB) ;
# endif
      BB       =  normalize(BB) ;
      // Psi is defined in IAU convention, East of North, angle between B and north direction !!!
      //   tan Psi =  (B*RA) / (B*DE)
      // ***NOTE*** Psi is angle for B, not for polarised emission... does not matter
      //            since we are calculating just the dispersion
      float Psi  =  atan2(dot(BB, RA), dot(BB, DE)) ;  // angle over full 2*pi !!!
      // gamma is wrt plane of the sky, cos(complement) = dot(BN, HDIR)
      float cc   =  0.99999f - 0.99998f * dot(BB, HDIR)*dot(BB, HDIR) ; // cos(gamma)^2
      float rho  =  DENS[oind] ;
      if (DTAU<1.0e-3f) {
         sz = exp(-TAU) *  (1.0f-0.5f*DTAU)        * sx * EMIT[oind]*rho ;
      } else {
         sz = exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * sx * EMIT[oind]*rho ;
      }
# if (LEVEL_THRESHOLD>0) // -------------------------------------------------------------------------
      if (olevel<LEVEL_THRESHOLD) {  // ignore low resolution regions
         sz = 0.0f ;   rho = 0.0f ;
      } 
# endif // ------------------------------------------------------------------------------------------
      // default case
# if (POLRED>0)                           
      // polarisation reduction encoded in |B|
      // PR    =  length(B[oind]) ;
      // density-weighted quantities
      sR   +=  rho*sx * PR  ;
      sRG  +=  rho*sx * PR * cc  ;                // sum(rho*cos^2 gamma)
      RQ   +=  rho*sx * PR * cos(2.0f*Psi)*cc ;   
      RU   +=  rho*sx * PR * sin(2.0f*Psi)*cc ;   // IAU convention
      // emission-weighted quantities
      sJ   +=  sz     * PR ;
      sJG  +=  sz     * PR * cc  ;                // sum(j  *cos^2 gamma)
      JQ   +=  sz     * PR * cos(2.0f*Psi)*cc  ;
      JU   +=  sz     * PR * sin(2.0f*Psi)*cc  ;
# else 
      // R==1, no polarisation reduction
      // NOTE -- if constant, p0 has no effect on these statistics
      sR   +=  rho*sx  ;                          // WEIGHT INCLUDES CELL SIZE
      sRG  +=  rho*sx * cc  ;                     // sum( rho * cos^2 gamma )
      RQ   +=  rho*sx * cos(2.0f*Psi)*cc ;        // Q
      RU   +=  rho*sx * sin(2.0f*Psi)*cc ;        // U
      sJ   +=  sz      ;
      sJG  +=  sz     * cc  ;
      JQ   +=  sz     * cos(2.0f*Psi)*cc ;
      JU   +=  sz     * sin(2.0f*Psi)*cc ;
# endif      
      TAU  += DTAU ;
      
      
      if (Y_SHEAR!=0.0f) {
         // We are dealing with a shearing box simulation where we assume periodicity in the
         // x and y directions, having additionally shear Y_SHEAR [root grid cells] between
         // the high and the low x edges
         //  ==>  ray continues from x=0 to x=NX-1 but y-coordinate is shifted by -Y_SHEAR
         //       ray continues from x=NX-1 to x=0 but y-coordinate is shifted by +Y_SHEAR
         //  left = -Y_SHEAR, right = +Y_SHEAR
         if (ind<0) {
            if ((POS.z>0.0f)&&(POS.z<NZ)) {  // ray exited but not on the z boundaries
               if (POS.y<0.0f) POS.y = NY-PEPS ;
               if (POS.y>NY)   POS.y = +PEPS ;
               if (POS.x<0.0f) {
                  POS.x = NX-PEPS ;    POS.y = fmod(POS.y+NY-Y_SHEAR, (float)NY) ;
               }
               if (POS.x>NX)   {
                  POS.x = +PEPS   ;    POS.y = fmod(POS.y   +Y_SHEAR, (float)NY) ;
               }
               IndexG(&POS, &level, &ind, DENS, OFF) ;            
            }         
         }
      }
            
      
   } // while ind>0
   
   
   // Now we can rI and jI = results of gamma^2 averaging
   MAP[1*npix+id] =  acos(sqrt(sRG/sR)) ;   //  rI = <gamma> = acos(sqrt(<cos^2 gamma>))
   MAP[3*npix+id] =  acos(sqrt(sJG/sJ)) ;   //  jI, the same with j-weighting
   
   
   // Calculate polarisation angle <Psi>= Chi
   //   Psi defined in IAU convention, East of North
   float RPsi =  0.5*atan2(RU, RQ) ;  // density-weighted  Psi
   float JPsi =  0.5*atan2(JU, JQ) ;  // emission-weighted Psi -- IAU convention
   
   
   
   
   // another loop through the cloud to calculate sum(w*(Psi-<Psi>)^2)
   POS = POS0 ;  
   IndexG(&POS, &level, &ind, DENS, OFF) ;
   
   TAU = 0.0f ;
   sRP=0.0f ;  sJP=0.0f ;  sR=0.0f ; sJ=0.0f ; // sR and sJ should not change, recalculate anyway
   los = 0.0f ;
   while (ind>=0) {
# if 0
      BB  =  B[oind] ;
# else
      BB.x = Bx[oind] ;  BB.y = By[oind] ;  BB.z = Bz[oind] ;  
# endif
      oind       =  OFF[level]+ind ;    // original cell, before step out of it
      olevel     =  level ;
      sx         =  GetStep(&POS, &HDIR, &level, &ind, DENS, OFF, PAR) ;
      los       +=  sx ;
      if (los>MAXLOS) {
         ind = -1 ;   POS.z = -1.0f ;
         sx  = MAXLOS - (los-sx) ;
      }
# ifdef USE_ABU
      DTAU       =  sx*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;  // sx in global units
# else
      DTAU       =  sx*DENS[oind]*(SCA+ABS) ;  // sx in global units
# endif
# if (POLRED>0)
      // PR         =  length(B[oind]) ;
      PR         =  length(BB) ;
# endif
      BB         =  normalize(BB) ;
      // Psi defined in IAU convention, East of North
      float Psi  =  atan2(dot(BB, RA), dot(BB, DE)) ; // same for density and emission weighting !!!
      float rho  =  DENS[oind] ;
      if (DTAU<1.0e-3f) {
         sz = exp(-TAU) *  (1.0f-0.5f*DTAU)        * sx * EMIT[oind]*rho ;
      } else {
         sz = exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * sx * EMIT[oind]*rho ;
      }
      // Calculate only the weighted sum of (Psi-<Psi>)^2, making sure the 
      // absolute value if the difference is below 0.5*pi
      
      
# if (LEVEL_THRESHOLD>0) // ------------------------------------------------------------------------------
      if (olevel<LEVEL_THRESHOLD) {
         sz = 0.0f ;   rho = 0.0f ; // effectively ignore low resolution volume
      }
# endif // -----------------------------------------------------------------------------------------------
      
      
# if (POLRED>0)
      // PR    =  length(B[oind]) ;  <--- is not BB that is itself normalised above
      sR   +=  rho*PR*sx ;                           // MUST INCLUDE WEIGHTING BY CELL SIZE
      d     =  fmod(fabs(TWOPI + RPsi - Psi), PI) ;  // [ 0, pi ]
      if (d>PIHALF) d = PI-d ;                       // [ 0, 0.5*pi ]
      sRP  +=  rho*PR*sx  * d*d  ;                   // sum( w*(Psi-<Psi>)^2 )
      sJ   +=  sz*PR ;
      d     =  fmod(fabs(TWOPI + JPsi - Psi), PI) ;  // [ 0, pi ]
      if (d>PIHALF) d = PI-d ;   // [ 0, 0.5*pi ]
      sJP  +=  sz*PR    *d*d  ;
# else
      sR   +=  rho*sx ;
      d     =  fmod(fabs(TWOPI + RPsi - Psi), PI) ;  // [ 0, pi ]
      if (d>PIHALF) d = PI-d ;   // [ 0, 0.5*pi ]... actually should be -PI+d but it will be squared
      sRP  +=  rho*sx  * d*d ;
      sJ   +=  sz ;
      d     =  fmod(fabs(TWOPI + JPsi - Psi), PI) ;  // [ 0, pi ]
      if (d>PIHALF) d = PI-d ;   // [ 0, 0.5*pi ]
      sJP  +=  sz      * d*d ;
      // Better way? Use directly Q and U values at centre and at offset position
      //    iff one started with (Q, U) instead of Psi angle...
      // d  =  Psi - Psi' = 0.5*arctan(Q'U-Q*U', Q'Q+U'U)
      // d  =  fmod(fabs(TWOPI + RPsi - Psi), PI) ;  // [ 0, pi ]    --> 
      // d  =  0.5f * arctan(Q*RU-RQ*U, Q*RQ+U*RU) ;
# endif
      
      TAU     += DTAU ;
      
      
      if (Y_SHEAR!=0.0f) {
         // We are dealing with a shearing box simulation where we assume periodicity in the
         // x and y directions, having additionally shear Y_SHEAR [root grid cells] between
         // the high and the low x edges
         //  ==>  ray continues from x=0 to x=NX-1 but y-coordinate is shifted by -Y_SHEAR
         //       ray continues from x=NX-1 to x=0 but y-coordinate is shifted by +Y_SHEAR
         //  left = -Y_SHEAR, right = +Y_SHEAR
         if (ind<0) {
            if ((POS.z>0.0f)&&(POS.z<NZ)) {  // ray exited but not on the z boundaries
               if (POS.y<0.0f) POS.y = NY-PEPS ;
               if (POS.y>NY)   POS.y = +PEPS ;
               if (POS.x<0.0f) {
                  POS.x = NX-PEPS ;    POS.y = fmod(POS.y+NY-Y_SHEAR, (float)NY) ;
               }
               if (POS.x>NX)   {
                  POS.x = +PEPS   ;    POS.y = fmod(POS.y   +Y_SHEAR, (float)NY) ;
               }
               IndexG(&POS, &level, &ind, DENS, OFF) ;            
            }         
         }
      }
            
      
      
   } // while ind
   
   // The values of sum(w*(Psi-<Psi>)^2) / sum(w)
   MAP[0*npix+id] =  sqrt(sRP/sR) ;    //  rT = sum(w*(Psi-<Psi>)^2)/sum(w)
   MAP[2*npix+id] =  sqrt(sJP/sJ) ;    //  jT
   
   
}


#endif






