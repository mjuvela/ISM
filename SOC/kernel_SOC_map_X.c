
// #define EPS 3.0e-5f   --- slows down because of EPS steps (some cases)
// #define EPS 6.0e-5f   // enough to avoid slow-down??? --- 2016-06-28
#define EPS 1.0e-4f   // enough to avoid slow-down??? --- 2016-06-28
// NOTE: in kernel_SOCAMO_1.c EPS = 4e-4 == much larger!!

#define PI        3.1415926536f
#define TWOPI     6.2831853072f
#define TWOTHIRD  0.6666666667f
#define PIHALF    1.5707963268f

#define DEBUG 0


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
   while(1) {
      float link = -DENS[OFF[*level]+(*ind)] ;
      *ind       = *(int *)&link ;      
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
   // Assume that one has already checked that the neighbour is not just another cell in the
   //   current octet
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
   
   
   while(DENS[OFF[*level]+(*ind)]<=0.0f) {  // loop until (level,ind) points to a leaf
      // convert to sub-octet coordinates -- same for transition from root or parent octet
      (*pos).x =  2.0f*fmod((*pos).x,1.0f) ;
      (*pos).y =  2.0f*fmod((*pos).y,1.0f) ;
      (*pos).z =  2.0f*fmod((*pos).z,1.0f) ;
      float link = -DENS[OFF[*level]+(*ind)] ;
      *ind       = *(int *)&link ;      
      *level  += 1 ;
      // printf("    ... first in octet %d with sid %d  %8.6f %8.6f %8.6f\n", (*ind), (*ind) % 8, (*pos).x, (*pos).y, (*pos).z) ;
      *ind   += 4*(int)floor((*pos).z)+2*(int)floor((*pos).y)+(int)floor((*pos).x) ; // subcell (suboctet)
      // printf("    ... actual cell %d with sid %d\n", (*ind), (*ind) % 8) ;
   }
   return ;
}



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
#if 0
   if (fabs((*DIR).x)<5.0e-6f) dx = 1e10f ;
   if (fabs((*DIR).y)<5.0e-6f) dy = 1e10f ;
   if (fabs((*DIR).z)<5.0e-6f) dz = 1e10f ;
#else
   // direction nearly along main axis => dx, dy, or dz zero => step becomes EPS
   // Added test on dx, dy, dz --- NEEDED FOR Healpix map???? to avoid 1e-5 steps
   // one of dx, dy, or dz could become zero, resulting in EPS steps
   // IS THIS NOT NEEDED IN kernel_SOCAMO_1.c ALSO ???
   if ((dx<1.0e-7f)||(fabs((*DIR).x)<2.0e-6f)) dx = 1e10f ;
   if ((dy<1.0e-7f)||(fabs((*DIR).y)<2.0e-6f)) dy = 1e10f ;
   if ((dz<1.0e-7f)||(fabs((*DIR).z)<2.0e-6f)) dz = 1e10f ;
#endif
   float step  =  min(dx, min(dy, dz)) + EPS ;
#if (DEBUG>0)
   if (step>3.0f) printf("HUGE STEP %.3e????\n", step) ;
#endif
#if (DEBUG>1)
   if (get_local_id(0)==2) {
      if ((step<1.0e-4)||(step>1.733f)) {
         printf("  Step %10.3e    dx %10.3e  dy %10.3e  dz %10.3e\n", step, dx, dy, dz) ;
         printf("  LEV %2d, IND %6d\n", *level, *ind) ;
         printf("  POS  %9.6f %8.4f %8.4f\n", (*POS).x, (*POS).y, (*POS).z) ;
         printf("  DIR  %9.6f %8.4f %8.4f\n", (*DIR).x, (*DIR).y, (*DIR).z) ;
      }
   }
#endif
   *POS       +=  step*(*DIR) ;          // update coordinates in units of current level
   step       *=  pow(0.5f, *level) ;    // step returned in units [GL]
   Index(POS, level, ind, DENS, OFF, PAR) ;
   return step ;                 // step length in units [GL]
}




__kernel void MappingX(
                       const      float    DX,     //  0 - pixel size in grid units
                       const      int2     NPIX,   //  1 - map dimensions (square map)
                       __global   float   *MAPX,   //  2 - 4 * NPIX * NPIX * NF
                       __global   float   *EMITX,  //  3 - EMIT[CELLS, NF]
                       const      float3   DIR,    //  4 - direction of observer
                       const      float3   RA,     //  5
                       const      float3   DE,     //  6
                       constant   int     *LCELLS, //  7 - number of cells on each level
                       constant   int     *OFF,    //  8 - index of first cell on each level
                       __global   int     *PAR,    //  9 - index of parent cell
                       __global   float   *DENS,   // 10 - density and hierarchy
                       __global   float   *ABS,    // 11
                       __global   float   *SCA,    // 12
                       const      float3   CENTRE, // 13
                       const      float3   INTOBS  // 14 position of internal observer
#if (WITH_COLDEN>0)
                      ,__global   float   *COLDEN  // 15 column density image
#endif
                      )
{
   // MappingX = MAP_FAST ... only when there are no abundance variations !
   
   // Calculate map for NF frequencies 
   // With   ABS[NF], SCA[NF], EMITX[CELLS, NF], MAPX[npix, NF]
   const int id   = get_global_id(0) ; // one work item per map pixel
   if (id>=(NPIX.x*NPIX.y)) return ;
   // printf("     id %6d / %6d\n", id, NPIX.x*NPIX.y) ;
   float  colden = 0.0f ;
   // float8 DTAU, TAU=(float8)(0.0f), PHOTONS=(float8)(0.0f) ;
   __local float DTAU[NF], TAU[NF], PHOTONS[NF] ;
   float3 POS, TMP ;
   float  sx, sy, sz ;
   int    ind, level, oind, ii ;
   int    i = id % NPIX.x ;   // longitude runs faster
   int    j = id / NPIX.x ;
   
   for(ii=0; ii<NF; ii++) {
      TAU[ii] = 0.0f ;   DTAU[ii] = 0.0f ; PHOTONS[ii] = 0.0f ;
   }
   
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
      TMP.x = cos(theta)*cos(phi) ;
      TMP.y = cos(theta)*sin(phi) ;
      TMP.z = sin(theta) ;
      if (fabs(TMP.x)<2.0e-5f)      TMP.x  = 2.0e-5f ;
      if (fabs(TMP.y)<2.0e-5f)      TMP.y  = 2.0e-5f ;
      if (fabs(TMP.z)<2.0e-5f)      TMP.z  = 2.0e-5f ;
      if (fmod(POS.x,1.0f)<2.0e-5f) POS.x += 4.0e-5f ;
      if (fmod(POS.y,1.0f)<2.0e-5f) POS.y += 4.0e-5f ;
      if (fmod(POS.z,1.0f)<2.0e-5f) POS.z += 4.0e-5f ;
   } else {               // normal external map
      // 2016-10-02  changed direction of X axis      
      POS.x = CENTRE.x - (i-0.5f*(NPIX.x-1))*DX*RA.x + (j-0.5f*(NPIX.y-1))*DX*DE.x ;
      POS.y = CENTRE.y - (i-0.5f*(NPIX.x-1))*DX*RA.y + (j-0.5f*(NPIX.y-1))*DX*DE.y ;
      POS.z = CENTRE.z - (i-0.5f*(NPIX.x-1))*DX*RA.z + (j-0.5f*(NPIX.y-1))*DX*DE.z ;
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
      sx      = GetStep(&POS, &TMP, &level, &ind, DENS, OFF, PAR) ;
      for(ii=0; ii<NF; ii++) DTAU[ii] = sx*DENS[oind]*(SCA[ii]+ABS[ii]) ;
      if (DTAU[NF-1]<1.0e-3f) {
         for(ii=0; ii<NF; ii++)
           PHOTONS[ii] += exp(-TAU[ii]) *  (1.0f-0.5f*DTAU[ii])  * sx * EMITX[oind*NF+ii]*DENS[oind] ;            
      } else {
         for(ii=0; ii<NF; ii++)
           PHOTONS[ii] += exp(-TAU[ii]) * ((1.0f-exp(-DTAU[ii]))/DTAU[ii]) * sx * EMITX[oind*NF+ii]*DENS[oind] ;
      }
      for(ii=0; ii<NF; ii++)   TAU[ii] += DTAU[ii] ;
      colden += sx*DENS[oind] ;
      // ind  = IndexG(POS, &level, &ind, DENS, OFF) ; --- should not be necessary!
   }
   for(ii=0; ii<NF; ii++)  MAPX[id*NF+ii] = PHOTONS[ii] ; 
   // on host side MAP[CELLS, NF]
#if (WITH_COLDEN>0)
   COLDEN[id] = colden * LENGTH ;
#endif
}







__kernel void HealpixMappingX(
                              const      float    DX,     //  0 - pixel size in grid units
                              const      int2     NPIX,   //  1 - total number of pixels
                              __global   float   *MAP,    //  2 - MAP[NPIX, NF]
                              __global   float   *EMIT,   //  3 - EMIT[CELLS, NF]
                              const      float3   DIR,    //  4 - direction of observer
                              const      float3   RA,     //  5
                              const      float3   DE,     //  6
                              constant   int     *LCELLS, //  7 - number of cells on each level
                              constant   int     *OFF,    //  8 - index of first cell on each level
                              __global   int     *PAR,    //  9 - index of parent cell
                              __global   float   *DENS,   // 10 - density and hierarchy
                              __global   float   *ABS,    // 11 - ABS[NF]
                              __global   float   *SCA,    // 12 - SCA[NF]
                              const      float3   CENTRE, // 13
                              const      float3   INTOBS  // 14 position of internal observer
                             )
{
   // Variable abundances not implemented here !
   const int id   = get_global_id(0) ;   // one work item per map pixel
   if (id>=(NPIX.x*NPIX.y)) return ;     // NPIX == 12*NSIDE*NSIDE
   float  colden = 0.0f ;
   __local float DTAU[NF], TAU[NF], PHOTONS[NF] ;
   float3 POS, TMP ;
   float  dx ;
   int    ind, level, oind, ii ;
   float  theta, phi ;
   for(ii=0; ii<NF; ii++) {
      DTAU[ii] = 0.0f ;  TAU[ii] = 0.0f ;  PHOTONS[ii] = 0.0f ;
   }
   
   Pixel2AnglesRing(id, &phi, &theta) ;
   // X-axis points towards the Galactic centre,
   // Y-axis towards the left
   // Z-axis up
   //    theta=0              ==  +Z
   //    theta=90, phi = 0    ==  +X
   //    theta=90, phi = 90   ==  +Y in Galactic ***and*** in grid coordinates
   TMP.x = +sin(theta)*cos(phi) ;
   TMP.y = +sin(theta)*sin(phi) ;
   TMP.z =  cos(theta) ;
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
      dx      = GetStep(&POS, &TMP, &level, &ind, DENS, OFF, PAR) ;
      // if (dx<1.0e-4f) printf("%6d  dx %10.3e\n", id, dx) ;
      for(ii=0; ii<NF; ii++) DTAU[ii] = dx*DENS[oind]*(SCA[ii]+ABS[ii]) ;  // dx in global units
      if (DTAU[NF-1]<1.0e-3f) {
         for(ii=0; ii<NF; ii++)
           PHOTONS[ii] += exp(-TAU[ii])*(1.0f-0.5f*DTAU[ii])*dx*EMIT[oind*NF+ii]*DENS[oind] ;
      } else {
         for(ii=0; ii<NF; ii++)
           PHOTONS[ii] += exp(-TAU[ii])*((1.0f-exp(-DTAU[ii]))/DTAU[ii])*dx*EMIT[oind*NF+ii]*DENS[oind] ;
      }
      for(ii=0; ii<NF; ii++)  TAU[ii] += DTAU[ii] ;
      colden += dx*DENS[oind] ;
   }
   for(ii=0; ii<NF; ii++) MAP[id*NF+ii] = PHOTONS[ii] ;
}

