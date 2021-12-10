
// One is using either scalar ABS, SCA (without -D WITH_ABU)
// or the pre-calculated sums over dust populations (with -D WITH_ABU)
// -D WITH_MSF has no effect on map making and thus vector ABS, SCA are not needed here


// #define EPS 3.0e-5f   --- slows down because of EPS steps (some cases)
// #define EPS 6.0e-5f   // enough to avoid slow-down? --- 2016-06-28
// #define EPS 1.0e-4f   // enough to avoid slow-down? --- 2016-06-28
#define EPS   2.5e-4f   // needed to reduce errors in maps -- 2016-08-14
#define PEPS  5.0e-4f

// NOTE: in kernel_SOCAMO_1.c EPS = 4e-4 == much larger!!

#define PI        3.1415926536f
#define TWOPI     6.2831853072f
#define TWOTHIRD  0.6666666667f
#define PIHALF    1.5707963268f

#define DEBUG 0

#if (OPT_IS_HALF>0)
# define GOPT(i) (vload_half(i,OPT))
# define OTYPE half
#else
# define GOPT(i) (OPT[i])
# define OTYPE float
#endif

// #pragma OPENCL EXTENSION cl_khr_fp16: enable


bool InRoi(const int level, const int ind, constant int *ROI, __global int *PAR, __constant int *OFF) {
   // Return true if cell (level, ind) is inside ROI=[x0,x1,y0,y1,z0,z1]
   int i, j, k ;
   // Go up to the root grid
   k = level ;   i = ind ;
   while(k>0) {
      i = PAR[OFF[k]+i-NX*NY*NZ] ;  k-- ;
   }
   // Calculate i,j,k indices of the root grid cell
   k   =  i / (NX*NY) ;
   j   =  (i/NX) % NY ;
   i   =  i % NX ;
   // printf("ROI %2d %2d  %2d %2d  %2d %2d\n", ROI[0], ROI[1], ROI[2], ROI[3], ROI[4], ROI[5]) ;
   if ((i>=ROI[0])&&(i<=ROI[1])&&(j>=ROI[2])&&(j<=ROI[3])&&(k>=ROI[4])&&(k<=ROI[5])) {
      return true ;
   } else {
      // printf("   %2d %2d %2d  OUT\n", i, j, k) ;
      return false ;
   }
}


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
   // if ((*ind)>=(NX*NY*NZ)) printf("? INVALID INDEX\n") ;
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


#if 0 // ==================================================================================================

void Index(float3 *pos, int *level, int *ind, 
           __global float *DENS, __constant int *OFF, __global int *PAR) {
   // Return the level and index of a neighbouring cell based on pos
   //   (level, ind) are the current cell coordinates, pos the local coordinates
   //   on return these will be the new coordinates or ind=-1 == one has exited
   int sid ;
   if (*level==0) {   // on root grid
      if (((*pos).x<=0.0f)||((*pos).x>=NX)||((*pos).y<=0.0f)||((*pos).y>=NY)||((*pos).z<=0.0f)||((*pos).z>=NZ)) {
         *ind = -1 ; return ;
      }
      *ind = (int)floor((*pos).z)*NX*NY + (int)floor((*pos).y)*NX + (int)floor((*pos).x) ;
      if (DENS[*ind]>0.0f) return ;  // level 0 cell was a leaf -- we are done
      // ind is root grid cell, pos are root grid coordinates
      // convert to sub-octet coordinates
   } else {     // go UP until the position is inside the octet
      while ((*level)>0) {
         *ind = PAR[OFF[*level]+(*ind)-NX*NY*NZ] ;  *level -= 1 ;  // parent cell index
         if ((*level)==0) {       // arrived at root grid
            (*pos)   *=  0.5f ;   // convertion of sub-octet to global coordinates
            (*pos).x +=  (*ind)      % NX  ; 
            (*pos).y +=  ((*ind)/NX) % NY ; 
            (*pos).z +=  (*ind) / (NX*NY) ;
            if (((*pos).x<=0.0f)||((*pos).x>=NX)||((*pos).y<=0.0f)||((*pos).y>=NY)||
                ((*pos).z<=0.0f)||((*pos).z>=NZ)) {
               *ind = -1 ; return ;  // we left the model volume
            }
            // the position is not necessarily in this cell, could be a neighbour!
            *ind = (int)floor((*pos).z)*NX*NY + (int)floor((*pos).y)*NX + (int)floor((*pos).x) ;
            // printf("Arrived at root: %8.5f %8.5f %8.5f\n", (*pos).x, (*pos).y, (*pos).z) ;
            if (DENS[*ind]>0.0f) return ;  // found the cell on level 0
            // ind is root grid index, pos is root grid coordinates
            break ;
         } else {
            // this was a step from a sub-octet to a parent octet, parent cell == ind
            sid       = (*ind) % 8 ; // parent cell = current cell in current octet
            (*pos)   *= 0.5f ;
            (*pos).x += sid % 2 ;  (*pos).y += (sid/2)%2  ;  (*pos).z += sid/4 ;
            // octet covering the position if coordinates are in [0,2]
            if (((*pos).x>=0.0f)&&((*pos).x<=2.0f)&&((*pos).y>=0.0f)&&((*pos).y<=2.0f)&&
                ((*pos).z>=0.0f)&&((*pos).z<=0.0f)) break ;
            // level, ind in some octet, pos are coordinates within that octet
         }
      } // while 
   } // else      
   float link ;
   while(DENS[OFF[*level]+(*ind)]<=0.0f) {  // loop until (level,ind) points to a leaf
      // convert to sub-octet coordinates -- same for transition from root or parent octet
      (*pos).x =  2.0f*fmod((*pos).x,1.0f) ;
      (*pos).y =  2.0f*fmod((*pos).y,1.0f) ;
      (*pos).z =  2.0f*fmod((*pos).z,1.0f) ;
      link     =        -DENS[OFF[*level]+(*ind)] ;
      *ind     = *(int*)&link ;     
      *level  += 1 ;      
# if 0
      if ((*level)>6) {
         printf("LEVEL %d  %8.4f %8.4f %8.4f  ind %9d\n", *level, (*pos).x, (*pos).y, (*pos).z, *ind) ;
      }
# endif      
      // printf("    ... first in octet %d with sid %d  %8.6f %8.6f %8.6f\n", (*ind), (*ind) % 8, (*pos).x, (*pos).y, (*pos).z) ;
      *ind   += 4*(int)floor((*pos).z)+2*(int)floor((*pos).y)+(int)floor((*pos).x) ; // subcell (suboctet)
      // printf("    ... actual cell %d with sid %d\n", (*ind), (*ind) % 8) ;
   }
   return ;
}

#else  // =================================================================================================



void Index(float3 *pos, int *level, int *ind, 
           __global float *DENS, __constant int *OFF, __global int *PAR) {
   // Return the level and index of a neighbouring cell based on pos
   //   (level, ind) are the current cell coordinates, pos the local coordinates
   //   on return these will be the new coordinates or ind=-1 == one has exited
   // Assume that one has already checked that the neighbour is not just another cell in the
   //   current octet
   int sid ;
# if (NX>100)
   double3  POS ;
# else
   float3   POS ;
# endif
   
   POS.x = (*pos).x ;   POS.y = (*pos).y ;   POS.z = (*pos).z ;
   
   if (*level==0) {   // on root grid
      if (((*pos).x<=0.0f)||((*pos).x>=NX)||((*pos).y<=0.0f)||((*pos).y>=NY)||((*pos).z<=0.0f)||((*pos).z>=NZ)) {
         *ind = -1 ; return ;
      }
      *ind = (int)floor((*pos).z)*NX*NY + (int)floor((*pos).y)*NX + (int)floor((*pos).x) ;
      if (DENS[*ind]>0.0f) return ;  // level 0 cell was a leaf -- we are done
   } else {     // go UP until the position is inside the octet
      while ((*level)>0) {
         *ind = PAR[OFF[*level]+(*ind)-NX*NY*NZ] ;  (*level)-- ;  // parent cell index
         if ((*level)==0) {       // arrived at root grid
# if (NX>100)
            POS   *=  0.5 ;   // convertion from sub-octet to global coordinates
# else
            POS   *=  0.5f ;
# endif
            POS.x +=  (*ind)      % NX  ; 
            POS.y +=  ((*ind)/NX) % NY ; 
            POS.z +=  (*ind) / (NX*NY) ;
            if ((POS.x<=0.0)||(POS.x>=NX)||(POS.y<=0.0)||(POS.y>=NY)||(POS.z<=0.0)||(POS.z>=NZ)) {
               *ind = -1 ; 
               (*pos).x = POS.x ;  (*pos).y = POS.y ;  (*pos).z = POS.z ; 
               return ;  // we left the model volume
            }
            // the position is not necessarily in cell 'ind', could be a neighbour!
            *ind = (int)floor(POS.z)*NX*NY + (int)floor(POS.y)*NX + (int)floor(POS.x) ;
            if (DENS[*ind]>0.0f) {
               (*pos).x = POS.x ;  (*pos).y = POS.y ;  (*pos).z = POS.z ; 
               return ;  // found the cell on level 0               
            }
            break ;
         } else {
            // this was a step from a sub-octet to a parent octet, parent cell == ind
            sid    = (*ind) % 8 ; // parent cell = current cell in current octet
# if (NX>100)
            POS   *=  0.5 ;
# else
            POS   *=  0.5f ;
# endif
            POS.x += sid % 2 ;  POS.y += (sid/2)%2  ;  POS.z += sid/4 ;
            // is the position inside this octet?
            if ((POS.x>=0.0)&&(POS.x<=2.0)&&(POS.y>=0.0)&&(POS.y<=2.0)&&(POS.z>=0.0)&&(POS.z<=0.0)) break ;
         }
      } // while not root grid
   } // else -- going up
   // Go down - position *is* inside the current octet
   while(DENS[OFF[*level]+(*ind)]<=0.0f) {  // loop until (level,ind) points to a leaf
      // convert to sub-octet coordinates -- same for transition from root and from parent octet
# if 0
      POS.x =  2.0*clamp(fmod(POS.x,1.0), DEPS, 1.0-DEPS) ;
      POS.y =  2.0*clamp(fmod(POS.y,1.0), DEPS, 1.0-DEPS) ;
      POS.z =  2.0*clamp(fmod(POS.z,1.0), DEPS, 1.0-DEPS) ;
# else
#  if (NX>100)
      POS.x =  2.0 *fmod(POS.x, 1.0) ;
      POS.y =  2.0 *fmod(POS.y, 1.0) ;
      POS.z =  2.0 *fmod(POS.z, 1.0) ;
#  else
      POS.x =  2.0f*fmod(POS.x, 1.0f) ;
      POS.y =  2.0f*fmod(POS.y, 1.0f) ;
      POS.z =  2.0f*fmod(POS.z, 1.0f) ;
#  endif
# endif
      float link = -DENS[OFF[*level]+(*ind)] ;
      *ind       = *(int *)&link ;      // first cell in the sub-octet
      (*level)++ ;
      *ind    += 4*(int)floor(POS.z)+2*(int)floor(POS.y)+(int)floor(POS.x) ; // subcell (suboctet)
   }
   (*pos).x = POS.x ;  (*pos).y = POS.y ;  (*pos).z = POS.z ; 
   return ;
}

#endif  // ================================================================================================



#if 1

float GetStep(float3 *POS, const float3 *DIR, int *level, int *ind, 
              __global float *DENS, __constant int *OFF, __global int*PAR) {
   // Calculate step to next cell, update level and ind for the next cell
   // Returns the step length in GL units (units of the root grid)
   // #if (NX>DIMLIM)
   // 2020-07-07 ... the double version was not here before, now still commented out with NX>9999
# if (NX>9999)
   // use of DPEPS << PEPS reduces cases where step jumps between non-neighbour cells
   //   == jump over a single intervening cell ... but this still had no effect on scatter tests !!
   double dx, dy, dz ;
   dx = ((((*DIR).x)>0.0) 
         ? ((1.0+PEPS-fmod((*POS).x,1.0f))/((*DIR).x)) 
         : ((   -PEPS-fmod((*POS).x,1.0f))/((*DIR).x))) ;
   dy = ((((*DIR).y)>0.0) 
         ? ((1.0+PEPS-fmod((*POS).y,1.0f))/((*DIR).y)) 
         : ((   -PEPS-fmod((*POS).y,1.0f))/((*DIR).y))) ;
   dz = ((((*DIR).z)>0.0) 
         ? ((1.0+PEPS-fmod((*POS).z,1.0f))/((*DIR).z)) 
         : ((   -PEPS-fmod((*POS).z,1.0f))/((*DIR).z))) ;
   dx    =  min(dx, min(dy, dz)) ;
   *POS +=  ((float)dx)*(*DIR) ;                    // update LOCAL coordinates - overstep by PEPS
   // step returned in units [GL] = root grid units
   dx    =  ldexp(dx, -(*level)) ;
   Index(POS, level, ind, DENS, OFF, PAR) ;         // update (level, ind)   
   return (float)dx ;                 // step length [GL]
# else
   float dx, dy, dz ;
   dx = ((((*DIR).x)>0.0f) 
         ? ((1.0f+PEPS-fmod((*POS).x,1.0f))/((*DIR).x)) 
         : ((    -PEPS-fmod((*POS).x,1.0f))/((*DIR).x))) ;
   dy = ((((*DIR).y)>0.0f) 
         ? ((1.0f+PEPS-fmod((*POS).y,1.0f))/((*DIR).y)) 
         : ((    -PEPS-fmod((*POS).y,1.0f))/((*DIR).y))) ;
   dz = ((((*DIR).z)>0.0f) 
         ? ((1.0f+PEPS-fmod((*POS).z,1.0f))/((*DIR).z)) 
         : ((    -PEPS-fmod((*POS).z,1.0f))/((*DIR).z))) ;
   dx    =  min(dx, min(dy, dz)) ;   
   *POS +=  dx*(*DIR) ;                    // update LOCAL coordinates - overstep by PEPS
   // step returned in units [GL] = root grid units
   dx    =  ldexp(dx, -(*level)) ;
   Index(POS, level, ind, DENS, OFF, PAR) ; // update (level, ind)
   return dx ;
# endif
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
   // Added test on dx, dy, dz --- NEEDED FOR Healpix map? to avoid 1e-5 steps
   // one of dx, dy, or dz could become zero, resulting in EPS steps
   // IS THIS NOT NEEDED IN kernel_SOCAMO_1.c ALSO ?
   if ((dx<1.0e-7f)||(fabs((*DIR).x)<2.0e-6f)) dx = 1e10f ;
   if ((dy<1.0e-7f)||(fabs((*DIR).y)<2.0e-6f)) dy = 1e10f ;
   if ((dz<1.0e-7f)||(fabs((*DIR).z)<2.0e-6f)) dz = 1e10f ;
# endif
   float step  =  min(dx, min(dy, dz)) + EPS ;
   
   // if (step>3.0f) printf("HUGE STEP %8.3f  level %d ind %9d\n", step, level, ind) ;
   
# if (DEBUG>1)
   if (get_local_id(0)==2) {
      if ((step<1.0e-4f)||(step>1.733f)) {
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
                      const      float    MAP_DX, //  0 - pixel size in grid units
                      const      int2     NPIX,   //  1 - map dimensions (square map)
                      __global   float   *MAP,    //  2 - 4 * NPIX * NPIX
                      __global   float   *EMIT,   //  3 - emitted  [CELLS] -- already includes ABU effects
                      const      float3   DIR,    //  4 - direction of observer
                      const      float3   RA,     //  5 - 2019-04-12 "RA" is axis pointing right !!
                      const      float3   DE,     //  6
                      constant   int     *LCELLS, //  7 - number of cells on each level
                      constant   int     *OFF,    //  8 - index of first cell on each level
                      __global   int     *PAR,    //  9 - index of parent cell
                      __global   float   *DENS,   // 10 - density and hierarchy
                      const      float    ABS,    // 11  scalar even with WITH_MSF !!!!
                      const      float    SCA,    // 12
                      const      float3   CENTRE, // 13
                      const      float3   INTOBS, // 14 position of internal observer
                      __global   OTYPE   *OPT,    // 15 ABS + SCA, taking into account abundances
                      __global   float   *COLDEN  // 16 column density image
#if (ROI_MAP>0)
                      ,constant   int     *ROI     // 17 [x0, x1,y0, y1, z0, z1] limits of ROI
#endif
                     )
{
   const int id   = get_global_id(0) ;  // one work item per map pixel
   if (id>=(NPIX.x*NPIX.y)) return ;
   float DTAU, TAU=0.0f, PHOTONS=0.0f, colden = 0.0f ;   
   float3 POS, TMP ;
   float  sx, sy, sz, dens, emit ;
   int    ind, level, oind, olevel ;
#if (ROI_MAP>0)
   int roi = 0 ;
#endif
   int    i = id % NPIX.x ;   // longitude runs faster
   int    j = id / NPIX.x ;
   
// #define BUGID 18063
   
   if (INTOBS.x>-1e10f) {  // perspective image from position inside the model
      float phi   = TWOPI*i/(float)(NPIX.x) ;  // NPIX.x = longitude points
      // Move longitude zero to the middle of the figure
      phi += PI ;
      float pix   = TWOPI/NPIX.x ;             // pixel size in radians
      float theta = pix*(j-(NPIX.y-1)/2) ;     // latitude
      POS   =  INTOBS ;      // location of the observer inside the model (root coordinates)
      // TMP is away from the observer, towards GC == along +X
      //   lon=0,  lat=90    ==  +Z
      //   lon=0,  lat=0     ==  +X
      //   lon=90, lat=0     ==  +Y
      TMP.x = cos(theta)*cos(phi) ; 
      TMP.y = cos(theta)*sin(phi) ;
      TMP.z = sin(theta) ;
      if (fabs(TMP.x)<1.0e-5f)      TMP.x  = 1.0e-5f ;
      if (fabs(TMP.y)<1.0e-5f)      TMP.y  = 1.0e-5f ;
      if (fabs(TMP.z)<1.0e-5f)      TMP.z  = 1.0e-5f ;
      if (fmod(POS.x,1.0f)<1.0e-5f) POS.x += 2.0e-5f ;
      if (fmod(POS.y,1.0f)<1.0e-5f) POS.y += 2.0e-5f ;
      if (fmod(POS.z,1.0f)<1.0e-5f) POS.z += 2.0e-5f ;
   } else {               // normal external map
      // 2016-10-02  changed direction of X axis to be consistent with polarisation maps
      // 2019-04-12  DE is up (j) and RA is **right** (i)
      POS.x = CENTRE.x + (i-0.5f*(NPIX.x-1))*MAP_DX*RA.x + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.x ;
      POS.y = CENTRE.y + (i-0.5f*(NPIX.x-1))*MAP_DX*RA.y + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.y ;
      POS.z = CENTRE.z + (i-0.5f*(NPIX.x-1))*MAP_DX*RA.z + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.z ;   
      
      // if (id==BUGID) printf("A:  %12.5f %12.5f %12.5f   %12.5f %12.5f %12.5f\n", POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z) ;
      
      // Step towards the observer, somewhere in front of the cloud
      POS +=  (NX+NY+NZ)*DIR ;
      
      
      // if (id==BUGID) printf("B:  %12.5f %12.5f %12.5f  --- in front\n", POS.x, POS.y, POS.z) ;
      // if (id==BUGID) printf("***    (0 - POS.x)/(-DIR.x) = (0-%10.3e)/(%10.3e) = %10.3e = sx \n", POS.x, DIR.x, (0.0f-POS.x)/(-DIR.x)) ;
      
      // go then in the direction -DIR (away from the observer) till we are inside the cloud
#if (NX<300)
      if (DIR.x>=0.0f)    sx = (NX      - POS.x) /  (-DIR.x)  +  EPS ;
      else                sx = (0.0f    - POS.x) /  (-DIR.x)  +  EPS ;
      if (DIR.y>=0.0f)    sy = (NY      - POS.y) /  (-DIR.y)  +  EPS ; 
      else                sy = (0.0f    - POS.y) /  (-DIR.y)  +  EPS ;
      if (DIR.z>=0.0f)    sz = (NZ      - POS.z) /  (-DIR.z)  +  EPS ;
      else                sz = (0.0f    - POS.z) /  (-DIR.z)  +  EPS ;
      // select smallest value that leaves POS inside the cloud (...cloud may be missed)
      TMP   = POS - sx*DIR ;   // -DIR is away from the observer
      // if (id==BUGID) printf("*** sx %12.5f  ==>   %12.5f %12.5f %12.5f\n", sx, TMP.x, TMP.y, TMP.z) ;
      if ((TMP.x<0.0f)||(TMP.x>NX)||(TMP.y<0.0f)||(TMP.y>NY)||(TMP.z<0.0f)||(TMP.z>NZ)) sx = 1e10f ;
      TMP   = POS - sy*DIR ;
      // if (id==BUGID) printf("*** sy %12.5f  ==>   %12.5f %12.5f %12.5f\n", sy, TMP.x, TMP.y, TMP.z) ;
      if ((TMP.x<0.0f)||(TMP.x>NX)||(TMP.y<0.0f)||(TMP.y>NY)||(TMP.z<0.0f)||(TMP.z>NZ)) sy = 1e10f ;
      TMP   = POS - sz*DIR ;
      // if (id==BUGID) printf("*** sz %12.5f  ==>   %12.5f %12.5f %12.5f\n", sz, TMP.x, TMP.y, TMP.z) ;
      if ((TMP.x<0.0f)||(TMP.x>NX)||(TMP.y<0.0f)||(TMP.y>NY)||(TMP.z<0.0f)||(TMP.z>NZ)) sz = 1e10f ;
      sx    =  min(sx, min(sy, sz)) ;
      POS   =  POS - sx*DIR ;
#else
      // In very rare cases, step to surface left a coordinate exactly on the cloud x, y, or z border.
      // Below, the change from "(TMP.x<=0.0f)" to "(TMP.x<0.0f)" already (almost) fixed that. 
      // We added 3*EPS to the step... still a few cases where that was not enough.
      // --- root grid ~1e3 and a float with 7 digits => minimum resolved step ~ 1e-3 ---
      // Now adding EPS to all coordinates as a separate step.
      if (DIR.x>=0.0f)    sx = (NX      - POS.x) /  (-DIR.x)  ;
      else                sx = (0.0f    - POS.x) /  (-DIR.x)  ;
      if (DIR.y>=0.0f)    sy = (NY      - POS.y) /  (-DIR.y)  ;
      else                sy = (0.0f    - POS.y) /  (-DIR.y)  ;
      if (DIR.z>=0.0f)    sz = (NZ      - POS.z) /  (-DIR.z)  ;
      else                sz = (0.0f    - POS.z) /  (-DIR.z)  ;
      // select smallest value that leaves POS inside the cloud (...cloud may be missed)
      TMP   = POS - sx*DIR ;   // -DIR is away from the observer
      TMP.x += (DIR.x>0.0f) ? (-EPS) : (+EPS) ;
      TMP.y += (DIR.y>0.0f) ? (-EPS) : (+EPS) ;
      TMP.z += (DIR.z>0.0f) ? (-EPS) : (+EPS) ;
      // if (id==BUGID) printf("*** sx %12.5f  ==>   %12.5f %12.5f %12.5f\n", sx, TMP.x, TMP.y, TMP.z) ;
      if ((TMP.x<0.0f)||(TMP.x>NX)||(TMP.y<0.0f)||(TMP.y>NY)||(TMP.z<0.0f)||(TMP.z>NZ)) sx = 1e10f ;
      TMP   = POS - sy*DIR ;
      TMP.x += (DIR.x>0.0f) ? (-EPS) : (+EPS) ;
      TMP.y += (DIR.y>0.0f) ? (-EPS) : (+EPS) ;
      TMP.z += (DIR.z>0.0f) ? (-EPS) : (+EPS) ;
      // if (id==BUGID) printf("*** sy %12.5f  ==>   %12.5f %12.5f %12.5f\n", sy, TMP.x, TMP.y, TMP.z) ;
      if ((TMP.x<0.0f)||(TMP.x>NX)||(TMP.y<0.0f)||(TMP.y>NY)||(TMP.z<0.0f)||(TMP.z>NZ)) sy = 1e10f ;
      TMP   = POS - sz*DIR ;
      TMP.x += (DIR.x>0.0f) ? (-EPS) : (+EPS) ;
      TMP.y += (DIR.y>0.0f) ? (-EPS) : (+EPS) ;
      TMP.z += (DIR.z>0.0f) ? (-EPS) : (+EPS) ;
      // if (id==BUGID) printf("*** sz %12.5f  ==>   %12.5f %12.5f %12.5f\n", sz, TMP.x, TMP.y, TMP.z) ;
      if ((TMP.x<0.0f)||(TMP.x>NX)||(TMP.y<0.0f)||(TMP.y>NY)||(TMP.z<0.0f)||(TMP.z>NZ)) sz = 1e10f ;
      sx    =  min(sx, min(sy, sz)) ;
      POS   =  POS - sx*DIR ;
      POS.x += (DIR.x>0.0f) ? (-EPS) : (+EPS) ;
      POS.y += (DIR.y>0.0f) ? (-EPS) : (+EPS) ;
      POS.z += (DIR.z>0.0f) ? (-EPS) : (+EPS) ;      
#endif
      
      // if (id==BUGID) printf("*** STEPS ---  %12.5f %12.5f %12.5f\n", sx, sy, sz) ;
      
#if 0
      POS   =  POS  - EPS*DIR ;
#endif
      
      // if (id==BUGID) printf("C:  %12.5f %12.5f %12.5f  --- front surface\n", POS.x, POS.y, POS.z) ;
      
      TMP   = -DIR ;              // DIR towards observer, TMP away from the observer = our stepping
      if (fabs(TMP.x)<1.0e-5f)      TMP.x  = 1.0e-5f ;
      if (fabs(TMP.y)<1.0e-5f)      TMP.y  = 1.0e-5f ;
      if (fabs(TMP.z)<1.0e-5f)      TMP.z  = 1.0e-5f ;
   }
   
   
   // float3  POS0 = POS ;
   
   
   // Index
   IndexG(&POS, &level, &ind, DENS, OFF) ;
   // printf("POS %8.4f %8.4f %8.4f  ind %d\n", POS.x, POS.y, POS.z, ind) ;
   
   
#if (DEBUG>0)
   CheckPos(&POS, &DIR, &level, &ind, DENS, OFF, PAR, 1) ;
#endif
   
   
#if (MAP_INTERPOLATION>0)
   // Prepare to interpolate quantities in directions perpendicular to the ray direction
   int  slevel, sind, level0, ind0 ;
   float a, b, Adens, Bdens, Aemit, Bemit, K ;
# if (MAP_INTERPOLATION==2)
   float Cemit, Cdens, c, wa, wb, wc, wd ;
# endif
   float3 ADIR, BDIR, MPOS, POS0 ;
   if (fabs(TMP.x)>fabs(TMP.y)) {
      if (fabs(TMP.z)>fabs(TMP.x)) {
         ADIR.x=0.0005f ; ADIR.y=1.0f ; ADIR.z=-TMP.y/TMP.z ;
      } else {
         ADIR.x=-TMP.z/TMP.x ; ADIR.y=0.0005f ; ADIR.z=1.0f ;
      }
   } else {
      if (fabs(TMP.z)>fabs(TMP.y)) {
         ADIR.x=0.0005f ; ADIR.y=1.0f ; ADIR.z=-TMP.y/TMP.z ;
      } else {
         ADIR.x=1.0f ; ADIR.y=-TMP.x/TMP.y ; ADIR.z=0.0005f ;
      }
   }
   ADIR   = normalize(ADIR) ;
   BDIR.x = TMP.y*ADIR.z-TMP.z*ADIR.y ;
   BDIR.y = TMP.z*ADIR.x-TMP.x*ADIR.z ;
   BDIR.z = TMP.x*ADIR.y-TMP.y*ADIR.x ;
   BDIR   = normalize(BDIR) ;        
   // if (id==123) printf("%8.4f %8.4f %8.4f    %8.4f %8.4f %8.4f    %8.4f %8.4f %8.4f \n", TMP.x, TMP.y, TMP.z,  ADIR.x, ADIR.y, ADIR.z,  BDIR.x, BDIR.y, BDIR.z) ;
#endif
   
   
   
   
   while (ind>=0) {
      
      oind    = OFF[level]+ind ;           // original cell, before step out of it
      olevel  = level ;
      
#if (MAP_INTERPOLATION>0)
      // store the initial position and coordinates at the start of the step
      POS0 = POS ;  ind0 = ind ;   level0 = level ;
      K       =  ldexp(1.0f, -level0) ;    // local to global length,   K <= 1.0
#endif
      
      //  TMP = step AWAY from the observer
      sx      = GetStep(&POS, &TMP, &level, &ind, DENS, OFF, PAR) ;  // sx == root grid coordinates !!
      
      dens    = DENS[oind] ;
      emit    = EMIT[oind] ;
      
      
      
#if (MAP_INTERPOLATION==2)
# if 1
      // we also restrict the length of the step to half of the cell length
      a  = 0.22f*K ;                     // maximum allowed step length in root grid coordinates
      if (sx>a) {
         sx = a ;                        // sx is in root grid coordinates
         POS = POS0 + 0.22f*TMP ;    ind = ind0 ;    level = level0 ;
         Index(&POS, &level, &ind, DENS, OFF, PAR) ; // update (level, ind)
      }
# endif      
      // find the neighbour in the ADIR direction
      slevel =  level0 ;                 // original level
      sind   =  ind0 ;                   // original index
      MPOS   =  POS0+(0.5f*sx/K)*TMP ;   // midway through the step
      a      =  GetStep(&MPOS, &ADIR, &slevel, &sind, DENS, OFF, PAR) ;
      a     /=  K  ;
      if ((a<=0.52f)&&(sind>=0)) {
         Adens = DENS[OFF[slevel]+sind] ;      Aemit = EMIT[OFF[slevel]+sind] ;
      } else {  // try the other direction
         slevel = level0 ; sind = ind0 ;    ADIR *= -1.0f ;   MPOS = POS0+(0.5f*sx/K)*TMP ;     
         a      = GetStep(&MPOS, &ADIR, &slevel, &sind, DENS, OFF, PAR) ;
         a     /= K ;
         if ((a<=0.52f)&&(sind>=0)) {
            Adens = DENS[OFF[slevel]+sind] ;   Aemit = EMIT[OFF[slevel]+sind] ;
         } else {
            a = 0.5f ;  Adens = 0.0f ; Aemit = 0.0f ;
         }
      }      // now ADIR is the direction towards the neighbour, distance a
      // find neighbour in the BDIR direction
      slevel  =  level0 ; 
      sind    =  ind0 ;             
      MPOS    =  POS0+(0.5f*sx/K)*TMP ;  
      b       =  GetStep(&MPOS, &BDIR, &slevel, &sind, DENS, OFF, PAR) ;
      b      /=  K  ;
      if ((b<=0.52f)&&(sind>=0)) {
         Bdens = DENS[OFF[slevel]+sind] ;      Bemit = EMIT[OFF[slevel]+sind] ;
      } else {
         slevel = level0 ; sind = ind0 ;    BDIR *= -1.0f ;   MPOS = POS0+(0.5f*sx/K)*TMP ;  
         b      = GetStep(&MPOS, &BDIR, &slevel, &sind, DENS, OFF, PAR) ;
         if ((b<=0.52f)&&(sind>=0)) {
            Bdens = DENS[OFF[slevel]+sind] ;   Bemit = EMIT[OFF[slevel]+sind] ;
         } else {
            b = 0.5f ;   Bdens = 0.0f ; Bemit = 0.0f ;
         }
      }     // now BDIR is the direction towards the neighbour, distance b
      a  = clamp(a, 0.0f, 0.51f) ;  //   this would smooth in horizontal direction
      b  = clamp(b, 0.0f, 0.51f) ;  //   
      // if (id%13==0) printf("  %8.4f  %8.4f     %10.3e %10.3e     %10.3e %10.3e\n", a, b, Adens, Bdens, Aemit, Bemit) ;
      // triangle including cell centre
      dens  =   (0.5f-a)*Adens + (0.5f-b)*Bdens + (a+b)*dens ;
      emit  =   (0.5f-a)*Aemit + (0.5f-b)*Bemit + (a+b)*emit ;
#endif
      
      
#if (MAP_INTERPOLATION==1)
      // interpolate density and emission in the direction perpendicular to the LOS
      //     - go to  POS-0.5*sx*TMP
      //     - measure distance to cell centre dc
      //     - go step  0.5 along ADIR, read value there  =>  ((1-dc)*centre+dc*off)/dc
      //     - go step  0.5 along BDIR, read value there  =>  ((1-dc)*centre+dc*off)/dc
      // how far is it to next cell in the direction ADIR
      slevel =  level0 ;                  // original level
      sind   =  ind0 ;                    // original index
      MPOS   =  POS0+(0.5f*sx/K)*TMP ;    // midway through the step, local coordinates
      a      =  GetStep(&MPOS, &ADIR, &slevel, &sind, DENS, OFF, PAR) ;
      a     /=  K ;                       // a is now in local coordinates
      // if (id==123) printf("\n %8.4f %8.4f %8.4f    %8.4f %8.4f %8.4f    %8.4f\n", MPOS.x, MPOS.y, MPOS.z, ADIR.x, ADIR.y, ADIR.z, a) ;
      if ((a<=0.502f)&&(sind>=0)) {        // distance less than 0.5 in local coordinates
         Adens = DENS[OFF[slevel]+sind] ;      Aemit = EMIT[OFF[slevel]+sind] ;
      } else {  // try the other direction
         // if (id==123) printf("FAILURE1  ... a=%8.4f ADENS=%10.3e   moved %2d %6d -> %2d %6d\n", a, Adens, level0, ind0, slevel, sind) ;
         slevel =  level0 ;    sind = ind0 ;    ADIR *= -1.0f ;   MPOS = POS0+(0.5f*sx/K)*TMP ;
         a      =  GetStep(&MPOS, &ADIR, &slevel, &sind, DENS, OFF, PAR) ;
         a     /=  K ;
         // if (id==123) printf(" %8.4f %8.4f %8.4f    %8.4f %8.4f %8.4f    %8.4f\n", MPOS.x, MPOS.y, MPOS.z, ADIR.x, ADIR.y, ADIR.z, a) ;
         if ((a<=0.502f)&&(sind>=0)) {
            Adens = DENS[OFF[slevel]+sind] ;   Aemit = EMIT[OFF[slevel]+sind] ;
         } else {
            // if (id==123) printf("FAILURE2  ... a=%8.4f ADENS=%10.3e   moved %2d %6d -> %2d %6d\n", a, Adens, level0, ind0, slevel, sind) ;
            a = 0.5f ;   Adens = 0.0f ;   Aemit = 0.0f ;
         }
      }
      // the orthogonal second direction --- K == local-to-global conversion
      slevel  =  level0 ;    sind = ind0 ;                      MPOS = POS0+(0.5f*sx/K)*TMP ;  
      b       =  GetStep(&MPOS, &BDIR, &slevel, &sind, DENS, OFF, PAR) ;
      b      /=  K ;
      if ((b<=0.502f)&&(sind>=0)) {
         Bdens = DENS[OFF[slevel]+sind] ;                     Bemit = EMIT[OFF[slevel]+sind] ;
      } else {
         slevel = level0 ; sind = ind0 ;    BDIR *= -1.0f ;   MPOS = POS0+(0.5f*sx/K)*TMP ;  
         b     = GetStep(&MPOS, &BDIR, &slevel, &sind, DENS, OFF, PAR) ;
         b    /= K ;
         if ((b<=0.502f)&&(sind>=0)) {
            Bdens = DENS[OFF[slevel]+sind] ;                  Bemit = EMIT[OFF[slevel]+sind] ;
         } else {
            b = 0.5f ;   Bdens = 0.0f ;     Bemit = 0.0f ;
         }
      }
      // if (id%13==0) printf(" %8.4f %8.4f    %10.3e %10.3e    %10.3e %10.3e\n", a, b, Adens, Bdens, Aemit, Bemit) ;
      // b = 0.5f ;  // vertical interpolation => eliminated by b=0.5
      // dens ~ (0,0),   A~(1,0),    B~(0,1),   (x,y) = (0.5-a, 0.5-b)
      a    = 0.5f-a ;         b = 0.5f-b ;      //  a=x, b=y
      dens = (1.0f-a-b)*dens + a*Adens + b*Bdens ;
      emit = (1.0f-a-b)*emit + a*Aemit + b*Bemit ;
#endif
      
      
#if (WITH_ABU>0)
      DTAU    = sx*dens * (GOPT(2*oind)+GOPT(2*oind+1)) ;
#else
      DTAU    = sx*dens * (SCA+ABS) ;  // sx in global units
#endif
      
      
#if (ROI_MAP>0)
      if (InRoi(olevel, oind-OFF[olevel], ROI, PAR, OFF)) { // if oind was inside ROI, add its emission
#endif        
         
#if (LEVEL_THRESHOLD>0) // -------------------------------------------------------------------------------
         // exclude emission from levels < LEVEL_THRESHOLD (== ignore low resolution regions)
         if (olevel>=LEVEL_THRESHOLD) {
            if (DTAU<1.0e-3f) {
               PHOTONS += exp(-TAU) *  (1.0f-0.5f*DTAU)        * sx * emit*dens ;
            } else {
               PHOTONS += exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * sx * emit*dens ;
            }
         }
#else // -------------------------------------------------------------------------------------------------
         // count emission from all levels of the hierarchy
         // we step AWAY from the observer so S(1-exp(-dtau)) multiplied by exp(-tau)
         // olevel = olevel ;
         if (DTAU<1.0e-3f) {
            PHOTONS += exp(-TAU) *  (1.0f-0.5f*DTAU)        * sx * emit*dens ;
         } else {
            PHOTONS += exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * sx * emit*dens ;
         }
#endif // ------------------------------------------------------------------------------------------------
#if (ROI_MAP>0)
      } 
#endif
      
      TAU    += DTAU ;
      // if (SAVE_TAU<=0.0f) 
      colden += sx*dens ;
      // ind  = IndexG(POS, &level, &ind, DENS, OFF) ; --- should not be necessary!
   }  // while ind>=0
   
   // i = longitude, j = latitude = runs faster
   MAP[id] = PHOTONS ;
   // printf("colden %10.3e x %10.3e\n", colden, LENGTH) ;
   if (SAVE_TAU>0.0f)  COLDEN[id] = TAU ;                 // saving optical depth instead of column density
   if (SAVE_TAU<=0.0f) COLDEN[id] = colden * LENGTH ;
   
   // if (id==BUGID) {
   //if (MAP[id]==0.0f) {
   // printf(" id %8d   MAP %12.4e  TAU %12.4e  COLDEN %12.4e\n", id, MAP[id], TAU, colden) ;
   // }
   // }
   
}














__kernel void HealpixMapping(
                             const      float    MAP_DX, //  0 - pixel size in grid units
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
                             __global   OTYPE   *OPT,    // 15 ABS + SCA (if abundance variations)
                             __global float    *COLDEN   // 16
#if (ROI_MAP>0)
                             ,constant   int    *ROI     // 17 [x0,x1,y0,y1,z0,z1] includive limits of ROI
#endif
                            )
{
   const int id   = get_global_id(0) ;   // one work item per map pixel
   if (id>=(12*NSIDE*NSIDE)) return ;    // NPIX.x == NSIDE
   float  DTAU, TAU=0.0f, PHOTONS=0.0f, colden = 0.0f ;   
   float3 POS, TMP ;
   float  dx ;
   int    ind, level, oind, olevel ;
   float  theta, phi ;
   Pixel2AnglesRing(id, &phi, &theta) ;
   //  2019-05-08 Note that TMP is a step from the observer (map pixel) towards the model !!
   //     the map centre is GC, (lon,lat)=(0,0) where TMP=[-1,0,0]
   // (lon,lat)=(0,0) =>   (phi,theta)=(0,0.5*pi)
   TMP.x = -sin(theta)*cos(phi) ;  //  (0,0.5*pi) =>  -1
   TMP.y = -sin(theta)*sin(phi) ;  //             =>   0
   TMP.z = +cos(theta) ;           //             =>   0
   if (fabs(TMP.x)<1.0e-5f)      TMP.x  = 1.0e-5f ;
   if (fabs(TMP.y)<1.0e-5f)      TMP.y  = 1.0e-5f ;
   if (fabs(TMP.z)<1.0e-5f)      TMP.z  = 1.0e-5f ;
   POS   =  INTOBS ;      // location of the observer inside the model (root coordinates)
   if ((fmod(POS.x,1.0f)<1.0e-5f)||(fmod(POS.x,1.0f)<0.99999f)) POS.x += 2.0e-5f ;
   if ((fmod(POS.y,1.0f)<1.0e-5f)||(fmod(POS.y,1.0f)<0.99999f)) POS.y += 2.0e-5f ;
   if ((fmod(POS.z,1.0f)<1.0e-5f)||(fmod(POS.z,1.0f)<0.99999f)) POS.z += 2.0e-5f ;
   IndexG(&POS, &level, &ind, DENS, OFF) ;
   while (ind>=0) {
      oind    = OFF[level]+ind ;           // original cell, before step out of it
      olevel  = level ;                    // for ROI calculation
      dx      = GetStep(&POS, &TMP, &level, &ind, DENS, OFF, PAR) ;
#if (WITH_ABU>0)
      DTAU    = dx*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
#else
      DTAU    = dx*DENS[oind]*(SCA+ABS) ;  // dx in global units
#endif
      
      
#if (ROI_MAP>0) 
      if (InRoi(olevel, oind-OFF[olevel], ROI, PAR, OFF)) {
#endif
         if (DTAU<1.0e-3f) {
            PHOTONS += exp(-TAU) *  (1.0f-0.5f*DTAU)        * dx * EMIT[oind]*DENS[oind] ;
         } else {
            PHOTONS += exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * dx * EMIT[oind]*DENS[oind] ;
         }
#if (ROI_MAP>0)
      } 
#endif
      
      
      TAU    += DTAU ;
      colden += dx*DENS[oind] ;
   }   
   MAP[id] = PHOTONS ;   // printf("PHOTONS %10.3e\n", PHOTONS) ;
   if (SAVE_TAU<=0.0f)  COLDEN[id] = colden*LENGTH ;
   if (SAVE_TAU>0.0f)   COLDEN[id] = TAU ;
}





#if (POLSTAT==0)

__kernel void PolMapping(
                         const      float    MAP_DX,     //  0 - pixel size in grid units
                         const      int2     NPIX,   //  1 - map dimensions (square map)
                         __global   float   *MAP,    //  2 - I[], Q[], U[], column density
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
                         __global   float   *By,     // 16
                         __global   float   *Bz,     // 17
                         __global   OTYPE   *OPT     // 18 
                        )
{
   const int id   = get_global_id(0) ;  // one work item per map pixel
   if (id>=(NPIX.x*NPIX.y)) return ;
   float DTAU, TAU=0.0f, colden = 0.0f ;
   float3 PHOTONS ;
   float3 POS, TMP, BN ;
   float  sx, sy, sz ;
   int    ind, level, oind, olevel ;
   int    i = id % NPIX.x ;   // longitude runs faster
   int    j = id / NPIX.x ;
   
   PHOTONS.x = 0.0f ;  PHOTONS.y = 0.0f ;  PHOTONS.z = 0.0f ;
   
   // normal external map
   //    RA increases **right** !!!,  DE increases UP
   // note the direction of X axis !!
   POS.x = CENTRE.x + (i-0.5f*(NPIX.x-1))*MAP_DX*RA.x + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.x ;
   POS.y = CENTRE.y + (i-0.5f*(NPIX.x-1))*MAP_DX*RA.y + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.y ;
   POS.z = CENTRE.z + (i-0.5f*(NPIX.x-1))*MAP_DX*RA.z + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.z ;   
   // find position on the front surface (if exists)
   POS -=  (NX+NY+NZ)*DIR ;   // move behind the cloud
   // find last crossing -- DIR pointing towards observer -- go to front surface
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
   
   // Index
   IndexG(&POS, &level, &ind, DENS, OFF) ;
   // printf("POS %8.4f %8.4f %8.4f  ind %d\n", POS.x, POS.y, POS.z, ind) ;
   
# if (DEBUG>0)
   CheckPos(&POS, &DIR, &level, &ind, DENS, OFF, PAR, 1) ;
# endif
   
   
   // See Planck XX
   // 2017-01-13  -- update the formula for total intensity, including p0 as a parameter
   //        sz = exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * sx * EMIT[oind]*DENS[oind]
   //
   //  OLD   I +=       sz ;
   //  NEW   I +=       sz  *   ( 1 + p0*(cc-0.6666666667)) ;
   //
   //  OLD   Q +=       sz  *   cos(2.0f*Psi)*cc  *  length(B[oind])
   //  NEW   Q +=       sz  *   cos(2.0f*Psi)*cc  *  length(B[oind])  ... OR
   //  NEW   Q +=  p0 * sz  *   cos(2.0f*Psi)*cc  ;
   //
   //  OLD   Q +=       sz  *   sin(2.0f*Psi)*cc  *  length(B[oind])
   //  NEW   Q +=       sz  *   sin(2.0f*Psi)*cc  *  length(B[oind])  ...  OR
   //  NEW   Q +=  p0 * sz  *   sin(2.0f*Psi)*cc  ;
   
   // polarisation reduction factor
   float p = p00 ;  // constant p0 or encoded in the length of the B vector
   
   
   while (ind>=0) {
      oind    = OFF[level]+ind ;    // original cell, before step out of it
      olevel  = level ;
      sx      = GetStep(&POS, &TMP, &level, &ind, DENS, OFF, PAR) ; // step away from the observer
# if (WITH_ABU>0)
      DTAU    = sx*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
# else
      DTAU    = sx*DENS[oind]*(SCA+ABS) ;  // sx in global units
# endif
      // the angles
      // for LOS along x,  Psi = atan2(By, Bz), gamma  = atan2(Bx, sqrt(By*By+Bz*Bz)) ;
      //    Psi    = angle east of north, in the plane of the sky --- full 2*pi !!!
      //    Psi is in IAU convention   tan(2*Psi) = U/Q, Psi = 0.5*atan2(U,Q)
      //    gamma  = angle away from the POS = 90 - angle wrt. DIR
      //    cos(gamma) = sin(complement)
      BN.x  =  Bx[oind] ;   BN.y = By[oind] ;   BN.z = Bz[oind] ;
# if (POLRED>0)   // using polarisation reduction encoded in B vector length
      p     =  length(BN) ;
# endif
      BN    =  normalize(BN) ;
      // Psi = polarisation angle = 0.5*pi + PA of the magnetic field,  tan(PA) =  -(B*RA) / (B*DE)
      // Minus because our "RA" is an axis pointing to the right!!
      float Psi  =  0.5f*PI + atan2(dot(BN, -RA), dot(BN, DE)) ;  // polarisation angle
      
      // gamma between B and POS,  zeta between B and LOS == B vs DIR
      // cos(gamma) = cos(90-zeta)  =  sin(zeta) 
      // cos^2(gamma) =  sin^2 zeta =  1 - cos^2 zeta = 1 - dot(B, DIR)^2
      float cc   =  0.99999f - 0.99998f * dot(BN, DIR)*dot(BN, DIR) ; // cos(gamma)^2
      
# if (POL_RHO_WEIGHT==1)  // polarisation maps with density, not emission weighted
      sz =  sx * DENS[oind] ;
# else
      if (DTAU<1.0e-3f) {
         sz = exp(-TAU) *  (1.0f-0.5f*DTAU)        * sx * EMIT[oind]*DENS[oind] ;
      } else {
         sz = exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * sx * EMIT[oind]*DENS[oind] ;
      }
# endif
      
# if (LEVEL_THRESHOLD>0) // ---------------------------------------------------------------------------------
      // if (id==9999) printf("level %2d\n", olevel) ;
      if (olevel>=LEVEL_THRESHOLD) {
         PHOTONS.x  +=      sz * (1.0f-p*(cc-0.6666667f)) ;                // I
         PHOTONS.y  +=  p * sz * cos(2.0f*Psi)*cc ; // Q   -- ok ... PSI = POSITION ANGLE OF B PROJECTION [0,2*pi] !!
         PHOTONS.z  +=  p * sz * sin(2.0f*Psi)*cc ; // U   -- IAU convention, Psi East of North
      }  else {
         ;  // if (id==9999) printf("SKIP\n") ;
      }
# else // no threshold  --------------------------------------------------------------------------------------
      olevel = olevel ;
      PHOTONS.x  +=      sz * (1.0f-p*(cc-0.6666667f)) ;                     // I
      PHOTONS.y  +=  p * sz * cos(2.0f*Psi)*cc ;       // Q --- Psi is polarisation angle (not PS of B-field)
      PHOTONS.z  +=  p * sz * sin(2.0f*Psi)*cc ;       // U   -- IAU convention, Psi East of North
# endif // LEVEL_THRESHOLD------------------------------------------------------------------------------------
      
      TAU     += DTAU ;
      colden  += sx*DENS[oind] ;
      // ind  = IndexG(POS, &level, &ind, DENS, OFF) ; --- should not be necessary!
   }   
   // i = longitude, j = latitude = runs faster
   MAP[0*NPIX.x*NPIX.y+id] = PHOTONS.x ;  // I
   MAP[1*NPIX.x*NPIX.y+id] = PHOTONS.y ;  // Q
   MAP[2*NPIX.x*NPIX.y+id] = PHOTONS.z ;  // U
   MAP[3*NPIX.x*NPIX.y+id] = colden * LENGTH ; // LENGTH == GL [cm]
   //  Q = cos(2*Psi), U = sin(2*Psi)
   //  tan(2*Chi) = U/Q
   // what does this mean ?
   //    "we rotated Psi by 0.5*pi to make this polarisation angle, not angle of B" 
   // ?
   
   // printf("colden %10.3e --- %10.3e %10.3e %10.3e\n", colden, PHOTONS.x, PHOTONS.y, PHOTONS.z) ;
}


#endif  // POLSTAT == 0









#if (POLSTAT==1)


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
   

__kernel void PolMapping(
                         const      float    MAP_DX,     //  0 - pixel size in grid units
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
                         __global   float   *Bz,     // 17 magnetic field vectors
                         __global   OTYPE   *OPT     // 18 OPT == ABS + SCA for variable abundances
                        )
{
   const int id   = get_global_id(0) ;  // one work item per map pixel
   if (id>=(NPIX.x*NPIX.y)) return ;
   float DTAU, TAU=0.0f ;
   // sum(rho), sum(rho*Psi), sum(rho*cos^2 gamma)
   // sum(j),   sum(j*Psi),   sum(j  *cos^2 gamma)
   float3 POS, TMP, BN ;
   float  sx, sy, sz ;
   int    ind, level, oind, olevel ;
   int    i = id % NPIX.x ;   // longitude runs faster
   int    j = id / NPIX.x ;
   
   // normal external map
   //    RA increases left !!!,  DE increases UP
   //  2019-04-12 RA increased **right** !!
   POS.x = CENTRE.x - (i-0.5f*(NPIX.x-1))*MAP_DX*RA.x + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.x ;
   POS.y = CENTRE.y - (i-0.5f*(NPIX.x-1))*MAP_DX*RA.y + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.y ;
   POS.z = CENTRE.z - (i-0.5f*(NPIX.x-1))*MAP_DX*RA.z + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.z ;   
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
   
   // Index
   float3 POS0 = POS ;
   IndexG(&POS, &level, &ind, DENS, OFF) ;
   
   // first pass through the cloud, calculate  sum(w*cos^2 gamma), 
   // and Q and U that will define the values of  <Psi>
   float sR=0.0f, sJ=0.0f, sRG=0.0f, sJG=0.0f, RQ=0.0f, RU=0.0f, JQ=0.0f, JU=0.0f, sRP, sJP, d, PR ;
   TAU = 0.0f ;
   
   
   while (ind>=0) {
      // float3 BB  =  B[oind] ;
      oind       =  OFF[level]+ind ;    // original cell, before step out of it
      olevel     =  level ;
      sx         =  GetStep(&POS, &TMP, &level, &ind, DENS, OFF, PAR) ;
# if (WITH_ABU>0)
      DTAU       =  sx*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
# else
      DTAU       =  sx*DENS[oind]*(SCA+ABS) ;  // sx in global units
# endif
      BN.x = Bx[oind] ;  BN.y = By[oind] ;  BN.z = Bz[oind] ;
      PR         =  length(BN) ;
      BN         =  normalize(BN) ;
      // *Polarisation angle* Psi is defined in IAU convention, East of North
      // ==  0.5*pi + angle between B and north direction
      // In this dispersion calculation the constant 0.5*PI could be omitted...
      float Psi  =  0.5f*PI + atan2(dot(BN, -RA), dot(BN, DE)) ;  // polarisation angle
      // gamma is wrt plane of the sky, cos(complement) = dot(BN, DIR)
      //      dot(BN,DIR) = cos(90-gamma) = -sin(gamma)
      // =>   cos(gamma)  = 1 - [dot(BN,DIR)]^2
      float cc   =  0.99999f - 0.99998f * dot(BN, DIR)*dot(BN, DIR) ; // cos(gamma)^2
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
      RU   +=  rho*sx * PR * sin(2.0f*Psi)*cc ;   // IAU convention, angle from north, to east
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
   } // while ind>0
   
   
   // Now we can rI and jI = results of gamma^2 averaging
   MAP[1*NPIX.x*NPIX.y+id] =  acos(sqrt(sRG/sR)) ;   //  rI = <gamma> = acos(sqrt(<cos^2 gamma>))
   MAP[3*NPIX.x*NPIX.y+id] =  acos(sqrt(sJG/sJ)) ;   //  jI, the same with j-weighting
   
   
   // Calculate polarisation angle Chi
   //   Chi defined in IAU convention, East of North
   float RChi =  0.5*atan2(RU, RQ) ;  // density-weighted  CHi
   float JChi =  0.5*atan2(JU, JQ) ;  // emission-weighted Chi -- IAU convention
   
   
   
   
   // another loop through the cloud to calculate sum(w*(Chi-<Chi>)^2)
   POS = POS0 ;  
   IndexG(&POS, &level, &ind, DENS, OFF) ;
   
   TAU = 0.0f ;
   sRP=0.0f ;  sJP=0.0f ;  sR=0.0f ; sJ=0.0f ; // sR and sJ should not change, recalculate anyway
   while (ind>=0) {
      oind       =  OFF[level]+ind ;    // original cell, before step out of it
      olevel     =  level ;
      sx         =  GetStep(&POS, &TMP, &level, &ind, DENS, OFF, PAR) ;
# if (WITH_ABU>0)
      DTAU       =  sx*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
# else
      DTAU       =  sx*DENS[oind]*(SCA+ABS) ;  // sx in global units
# endif
      BN.x = Bx[oind] ;   BN.y = By[oind] ;   BN.z = Bz[oind] ;
      PR         =  length(BN) ;
      BN         =  normalize(BN) ;
      // Chi defined in IAU convention, East of North
      float Chi  =  0.5f*PI + atan2(dot(BN, -RA), dot(BN, DE)) ; // same for density and emission weighting !!!
      float rho  =  DENS[oind] ;
      if (DTAU<1.0e-3f) {
         sz = exp(-TAU) *  (1.0f-0.5f*DTAU)        * sx * EMIT[oind]*rho ;
      } else {
         sz = exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * sx * EMIT[oind]*rho ;
      }
      // Calculate only the weighted sum of (Chi-<Chi>)^2, making sure the 
      // absolute value of the difference is below 0.5*pi
      
      
# if (LEVEL_THRESHOLD>0) // ------------------------------------------------------------------------------
      if (olevel<LEVEL_THRESHOLD) {
         sz = 0.0f ;   rho = 0.0f ; // effectively ignore low resolution volume
      }
# endif // -----------------------------------------------------------------------------------------------
      
      
# if (POLRED>0)
      // PR    =  length(B[oind]) ;
      // BN.x  =  Bx[oind] ;    BN.y = By[oind] ;   BN.z = Bz[oind] ;
      // PR    =  length(BN) ;
      sR   +=  rho*PR*sx ;                           // MUST INCLUDE WEIGHTING BY CELL SIZE
      d     =  fmod(fabs(TWOPI + RChi - Chi), PI) ;  // [ 0, pi ]
      if (d>PIHALF) d = PI-d ;                       // [ 0, 0.5*pi ]
      sRP  +=  rho*PR*sx  * d*d  ;                   // sum( w*(Chi-<Chi>)^2 )
      sJ   +=  sz*PR ;
      d     =  fmod(fabs(TWOPI + JChi - Chi), PI) ;  // [ 0, pi ]
      if (d>PIHALF) d = PI-d ;   // [ 0, 0.5*pi ]
      sJP  +=  sz*PR    *d*d  ;
# else
      sR   +=  rho*sx ;
      d     =  fmod(fabs(TWOPI + RChi - Chi), PI) ;  // [ 0, pi ]
      if (d>PIHALF) d = PI-d ;   // [ 0, 0.5*pi ]... actually should be -PI+d but it will be squared
      sRP  +=  rho*sx  * d*d ;
      sJ   +=  sz ;
      d     =  fmod(fabs(TWOPI + JChi - Chi), PI) ;  // [ 0, pi ]
      if (d>PIHALF) d = PI-d ;   // [ 0, 0.5*pi ]
      sJP  +=  sz      * d*d ;
      // Better way? Use directly Q and U values at centre and at offset position
      //    iff one started with (Q, U) instead of Chi angle...
      // d  =  Chi - Chi' = 0.5*arctan(Q'U-Q*U', Q'Q+U'U)
      // d  =  fmod(fabs(TWOPI + RChi - Chi), PI) ;  // [ 0, pi ]    --> 
      // d  =  0.5f * arctan(Q*RU-RQ*U, Q*RQ+U*RU) ;
# endif
      
      TAU     += DTAU ;
   }   
   
   // The values of sum(w*(Chi-<Chi>)^2) / sum(w)
   MAP[0*NPIX.x*NPIX.y+id] =  sqrt(sRP/sR) ;    //  rT = sum(w*(Chi-<Chi>)^2)/sum(w)
   MAP[2*NPIX.x*NPIX.y+id] =  sqrt(sJP/sJ) ;    //  jT
   
   
}


#endif // POLSTAT==1












#if (POLSTAT==2)


// Calculate I, Q, U, N
//   observer is outside the volume at NX*GL from cloud centre
//   use perspective view so that each pixel corresponds to
//   a pixel in the plane through the model centre
//   each LOS extends up to MAXLOS grid units, re-entering on the opposite side
//   if needed
   

__kernel void PolMapping(
                         const      float    MAP_DX,     //  0 - pixel size in grid units
                         const      int2     NPIX,   //  1 - map dimensions (square map)
                         __global   float   *MAP,    //  2 - I[], Q[], U[], column density
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
   if (id>=(NPIX.x*NPIX.y)) return ;
   float DTAU, TAU=0.0f, colden=0.0f, distance=0.0f ;
   float3 PHOTONS ;
   float3 POS, TMP, VDIR, BN ;
   float  sx, sy, sz ;
   int    ind, level, oind, olevel ;
   int    i = id % NPIX.x ;   // longitude runs faster
   int    j = id / NPIX.x ;
   
   PHOTONS.x = 0.0f ;  PHOTONS.y = 0.0f ;  PHOTONS.z = 0.0f ;
   
   // RA increases **right** !!!,  DE increases UP
   POS.x = CENTRE.x - (i-0.5f*(NPIX.x-1))*MAP_DX*RA.x + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.x ;
   POS.y = CENTRE.y - (i-0.5f*(NPIX.x-1))*MAP_DX*RA.y + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.y ;
   POS.z = CENTRE.z - (i-0.5f*(NPIX.x-1))*MAP_DX*RA.z + (j-0.5f*(NPIX.y-1))*MAP_DX*DE.z ;
   // Here DIR is different for each pixel
   // observer is located at distance NX from the cloud centre towards the input DIR
   float3 OPOS ;         
   // how far outside the box is the observer?
   //   if observer is close, strong perspective effect but nearest voxels are big
   //   on the sky...
   //   distance 1.0*NX: 250pc box, observer 125pc from the cloud front
   //   distance 1.5*NX  250pc from the front of the cloud
   OPOS = CENTRE + 1.0*NX*DIR ;      // position of the observer -- DIR towards observer
   VDIR = POS-OPOS ;                 // vector into the cloud
   VDIR = normalize(VDIR) ;          // unit vector from observer into the cloud
   POS  = OPOS - (NX+NY+NZ)*VDIR ;   // position somewhere upstream of the cloud
   // find FIRST crossing -- VDIR pointing downstream
   if (VDIR.x>=0.0f)    sx = (0.0f    - POS.x) /  VDIR.x + EPS ;
   else                 sx = (NX      - POS.x) / (VDIR.x+1.0e-10f) + EPS ;
   if (VDIR.y>=0.0f)    sy = (0.0f    - POS.y) /  VDIR.y - EPS ;
   else                 sy = (NY      - POS.y) / (VDIR.y+1.0e-10f) + EPS ; 
   if (VDIR.z>=0.0f)    sz = (0.0f    - POS.z) /  VDIR.z - EPS ;
   else                 sz = (NZ      - POS.z) / (VDIR.z+1.0e-10f) + EPS ;
   // select largest step that puts POS inside the cloud (cloud may also be missed)
   TMP = POS + sx*VDIR ;
   if ((TMP.x<=0.0f)||(TMP.x>=NX)||(TMP.y<=0.0f)||(TMP.y>=NY)||(TMP.z<=0.0f)||(TMP.z>=NZ)) sx = -1e10f ;
   TMP = POS + sy*VDIR ;
   if ((TMP.x<=0.0f)||(TMP.x>=NX)||(TMP.y<=0.0f)||(TMP.y>=NY)||(TMP.z<=0.0f)||(TMP.z>=NZ)) sy = -1e10f ;
   TMP = POS + sz*VDIR ;
   if ((TMP.x<=0.0f)||(TMP.x>=NX)||(TMP.y<=0.0f)||(TMP.y>=NY)||(TMP.z<=0.0f)||(TMP.z>=NZ)) sz = -1e10f ;
   //
   sx    =  max(sx, max(sy, sz)) ;
   POS   =  POS + sx*VDIR ;
   // VDIR away from the observer
   // Index
   IndexG(&POS, &level, &ind, DENS, OFF) ;
   
# if (DEBUG>0)
   CheckPos(&POS, &DIR, &level, &ind, DENS, OFF, PAR, 1) ;
# endif
   
   // polarisation reduction factor
   float p = p00 ;  // constant p0 or encoded in the length of the B vector
   
   
   
   while (ind>=0) {   // keep on stepping in direction VDIR
      oind       = OFF[level]+ind ;    // original cell, before step out of it
      olevel     = level ;
      sx         = GetStep(&POS, &VDIR, &level, &ind, DENS, OFF, PAR) ; // step away from observer
# if (WITH_ABU>0)
      DTAU       = sx*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
# else
      DTAU       = sx*DENS[oind]*(SCA+ABS) ;  // sx in global units
# endif
      // float3 BN  = normalize(B[oind]) ;
      BN.x = Bx[oind] ;    BN.y = By[oind] ;    BN.z = Bz[oind] ;
# if (POLRED>0)   // using polarisation reduction encoded in B vector length
      p   =  length(BN) ;
# endif
      BN  = normalize(BN) ;
      // note: 0.5*PI is needed for correct Psi
      // atan2() is the position angle of the B-field, polarisation angle differs by 0.5*pi
      float Psi  = 0.5f*PI + atan2(dot(BN, -RA), dot(BN, DE)) ;
      float cc   = 0.99999f - 0.99998f * dot(BN, DIR)*dot(BN, DIR) ; // cos(gamma)^2
      if (DTAU<1.0e-3f) {
         sz = exp(-TAU) *  (1.0f-0.5f*DTAU)        * sx * EMIT[oind]*DENS[oind] ;
      } else {
         sz = exp(-TAU) * ((1.0f-exp(-DTAU))/DTAU) * sx * EMIT[oind]*DENS[oind] ;
      }
      PHOTONS.x  +=      sz * (1.0f-p*(cc-0.6666667f)) ;  // I 
      PHOTONS.y  +=  p * sz * cos(2.0f*Psi)*cc ;          // Q 
      PHOTONS.z  +=  p * sz * sin(2.0f*Psi)*cc ;          // U 
      TAU        +=  DTAU ;
      colden     +=  sx*DENS[oind] ;
      distance   +=  sx ;   // total distance travellend in [GL] units
      if (distance>MAXLOS) break ; // stop the ray
      // ind  = IndexG(POS, &level, &ind, DENS, OFF) ; --- should not be necessary!
      if (ind<0){    // went out of the cloud => put the ray back into cloud
         if (POS.x<=0.0f) POS.x = NX-EPS ;
         if (POS.x>=NX)   POS.x =   +EPS ;
         if (POS.y<=0.0f) POS.y = NY-EPS ;
         if (POS.y>=NY)   POS.y =   +EPS ;
         if (POS.z<=0.0f) POS.z = NZ-EPS ;
         if (POS.z>=NZ)   POS.z =   +EPS ;
         IndexG(&POS, &level, &ind, DENS, OFF) ;
      }
   } // while(ind>=0) --- follow one ray till the end
   
   // i = longitude, j = latitude = runs faster
   MAP[0*NPIX.x*NPIX.y+id] = PHOTONS.x ;  // I
   MAP[1*NPIX.x*NPIX.y+id] = PHOTONS.y ;  // Q
   MAP[2*NPIX.x*NPIX.y+id] = PHOTONS.z ;  // U
   MAP[3*NPIX.x*NPIX.y+id] = colden * LENGTH ; // LENGTH == GL [cm]
}


#endif  // POLSTAT == 2  -- perspective images with given MAXLOS LOS distance






__kernel void PSTau(
                    const      int      no,       //  0 - number of point sources
                    __global   float3  *PSPOS,    //  1 -- point source positions
                    const      float3   DIR,      //  2 - direction of observer
                    const      float3   RA,       //  3 - 2019-04-12 "RA" is axis pointing right !!
                    const      float3   DE,       //  4
                    constant   int     *LCELLS,   //  5 - number of cells on each level
                    constant   int     *OFF,      //  6 - index of first cell on each level
                    __global   int     *PAR,      //  7 - index of parent cell
                    __global   float   *DENS,     //  8 - density and hierarchy
                    const      float    ABS,      //  9  scalar even with WITH_MSF !!!!
                    const      float    SCA,      // 10
                    __global   OTYPE   *OPT,      // 11 ABS + SCA, taking into account abundances
                    __global   float   *pscolden, // 12 column density image
                    __global   float   *pstau     // 13 
                   )
{
   const int id   = get_global_id(0) ;  // one work item per map pixel
   if (id>=no) return ;                 // one work item per source
   float DTAU, TAU=0.0f, colden = 0.0f ;   
   float3 POS ;
   float  sx, sy, sz ;
   int    ind, level, oind, olevel ;
   POS.x  = PSPOS[id].x ;   POS.y  = PSPOS[id].y ;   POS.z  = PSPOS[id].z ;
   IndexG(&POS, &level, &ind, DENS, OFF) ;  
   while (ind>=0) {  // step TOWARDS the observer (maps were integrated stepping AWAY from the observer)
      oind    = OFF[level]+ind ;    // original cell, before step out of it
      olevel  = level ;
      sx      = GetStep(&POS, &DIR, &level, &ind, DENS, OFF, PAR) ; // step TOWARDS the observer !!!
#if (WITH_ABU>0)
      DTAU    = sx*DENS[oind]*(GOPT(2*oind)+GOPT(2*oind+1)) ;
#else
      DTAU    = sx*DENS[oind]*(SCA+ABS) ;  // sx in global units
#endif     
      TAU    += DTAU ;
      colden += sx*DENS[oind] ;
   }   
   pscolden[id] = colden * LENGTH ;                 // saving optical depth instead of column density
   pstau[id]    = TAU ;
}




