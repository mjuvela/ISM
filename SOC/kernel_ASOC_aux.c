#define HG_TEST 0

#define TWOPI     6.28318531f
#define TAULIM    5.0e-4f
#define PIHALF    1.5707963268f
#define TWOTHIRD  0.6666666667f
#define PI        3.1415926535897f
//  #define ADHOC   (1.0e-10f)

#if (OPT_IS_HALF>0)
# define GOPT(i) (vload_half(i,OPT))
# define OTYPE half
#else
# define GOPT(i) (OPT[i])
# define OTYPE float
#endif


#define F2I(x)    (*(__global int*)&x)   //  __global BUFFER -> OTI
#define I2F(x)    (*(float *)&x)         //           OTI    -> __global BUFFER

#if 0
#define DIMLIN 300
#else
#define DIMLIM 100 // if base grid NX is larger, use double precision for position
#endif

#if (NX>DIMLIM)  // from Index()
# define ZERO 0.0
# define HALF 0.5
# define ONE  1.0
# define TWO  2.0
#else
# define ZERO 0.0f
# define HALF 0.5f
# define ONE  1.0f
# define TWO  2.0f
#endif


inline void atomicAdd_g_f(volatile __global float *addr, float val) {
#if 1
   union{
      unsigned int u32;
      float        f32;
   } next, expected, current;
   current.f32    = *addr;
   do {
      expected.f32 = current.f32;
      next.f32     = expected.f32 + val;
      current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                                     expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
#else
   *addr += val ;
#endif
}


constant float PHOTON_LIMIT = 1.0e-30f ;
// constant double DPEPS       =  5.0e-6  ;
constant double DPEPS       =  2.0e-5  ;  // 2020-07-18

#if 1

// 2020-07-17  these were the previous values
// constant float  EPS         = 12.0e-4f  ;  // used only in SimBgSplit()
// constant float  PEPS        =  5.0e-4f  ;

// 2020-07-17 try these??
// constant float  EPS         =  3.5e-4f  ; => "FAILED TO FIND THE BORDER !!!" in SimBgSplit, for large MAXL models!!
constant float  EPS         =  5.0e-4f  ; // same => -''- ; add DIMLIM to GetStep
constant float  PEPS        =  1.0e-4f  ;
constant float  PEPS2       =  2.0e-4f  ;  // used in "new" IndeOT() only ... was 2e-4
constant float  DEPS        =  5.0e-5f  ;

#else

constant float  PEPS        =  1.0e-4f  ;
constant float  DEPS        =  1.0e-5f  ;

#endif

#define DEBUG   0


#include "mwc64x_rng.cl"
#define Rand(x)  (MWC64X_NextUint(x)/4294967295.0f)



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
   if (DENS[*ind]>0.0f) return ;  // found a leaf
   while(1) {
      // coordinates in sub-octet
      (*pos).x = 2.0f*fmod((*pos).x, 1.0f) ;  // coordinate inside root grid cell [0,1]
      (*pos).y = 2.0f*fmod((*pos).y, 1.0f) ;
      (*pos).z = 2.0f*fmod((*pos).z, 1.0f) ;
      // new indices
      float link = -DENS[OFF[*level]+(*ind)] ;
      // printf("   PARENT[%d] = %d\n", *ind, *(int*)(&link)) ;
      *ind       = *(int *)&link ;  // index **WITHIN THE LEVEL*** (not global index)
      (*level)++ ;   
      *ind  += 4*(int)floor((*pos).z) + 2*(int)floor((*pos).y) + (int)floor((*pos).x) ; // cell in octet
      if (DENS[OFF[*level]+(*ind)]>0.0f) {
         return ; // found a leaf
      }
   }
}


void RootPos(float3 *POS, const int ilevel, const int iind, __constant int *OFF, __global int *PAR) {
   // Given current coordinates (ilevel, iind) and current position POS, return the 
   // position in the root grid coordinates.
   int level=ilevel, ind=iind, sid ;   
   if (level==0) return ;
   // Otherwise go UP
   while (level>0) {
      ind  =  PAR[OFF[level]+ind-NX*NY*NZ] ;  level-- ;  // parent cell index
      if (level==0) {              // arrived at root grid
         *POS     *=  0.5f ;       // sub-octet coordinates [0,2] into global coordinates [0,1]
         (*POS).x +=  ind      % NX  ; 
         (*POS).y +=  (ind/NX) % NY ; 
         (*POS).z +=  ind  / (NX*NY) ;
         return ;
      } else {  // step from a sub-octet to a parent octet, parent cell == ind
         sid    =  ind % 8 ;  // parent cell = current cell in current octet
         *POS  *=  0.5f ;
         (*POS).x +=  sid % 2 ;   (*POS).y += (sid/2)%2  ;   (*POS).z += sid/4 ;
      }
   } // while not root grid
}



#if 1  // @i pre 2020-07-18  --- use this one, the alternative (new) is worse!


void Index(float3 *pos, int *level, int *ind, 
           __global float *DENS, __constant int *OFF, __global int *PAR) {
   // Return the level and index of a neighbour cell based on pos
   //   (level, ind) are the current cell coordinates, pos the local coordinates
   //   on return these will be the new coordinates or ind=-1 == one has exited
   // Assume that one has already checked that the neighbour is not just another cell in the
   //   current octet
   int sid ;
# if (NX>DIMLIM)  // MUST AGREE WITH DEFS AT THE BEGINNING OF THIS FILE
   double3  POS ;
# else
   float3   POS ;
# endif
   POS.x = pos->x ;   POS.y = pos->y ;   POS.z = pos->z ;
   
   // printf("Index:  %9.5f %9.5f %9.5f\n", POS.x, POS.y, POS.z) ;
   
   if (*level==0) {   // on root grid
      if (((*pos).x<=ZERO)||((*pos).x>=NX)||((*pos).y<=ZERO)||((*pos).y>=NY)||((*pos).z<=ZERO)||((*pos).z>=NZ)) {
         *ind = -1 ; return ;
      }
      *ind = (int)floor((*pos).z)*NX*NY + (int)floor((*pos).y)*NX + (int)floor((*pos).x) ;
      if (DENS[*ind]>0.0f) return ;  // level 0 cell was a leaf -- we are done
   } else {     // go UP until the position is inside the octet
      while ((*level)>0) {
         *ind = PAR[OFF[*level]+(*ind)-NX*NY*NZ] ;  (*level)-- ;  // parent cell index
         if ((*level)==0) {       // arrived at root grid
            POS   *=  HALF ;       // convert from sub-octet to parent coordinates
            POS.x +=  (*ind)      % NX  ; 
            POS.y +=  ((*ind)/NX) % NY ; 
            POS.z +=  (*ind) / (NX*NY) ;            
            if ((POS.x<=ZERO)||(POS.x>=NX)||(POS.y<=ZERO)||(POS.y>=NY)||(POS.z<=ZERO)||(POS.z>=NZ)) {
               *ind = -1 ; 
               pos->x = POS.x ;  pos->y = POS.y ;  pos->z = POS.z ; 
               return ;  // we left the model volume
            }
            // the position is not necessarily in cell 'ind', could be a neighbour!
            *ind = (int)floor(POS.z)*NX*NY + (int)floor(POS.y)*NX + (int)floor(POS.x) ;
            if (DENS[*ind]>0.0f) {
               pos->x = POS.x ;  pos->y = POS.y ;  pos->z = POS.z ; 
               return ;  // found the cell on level 0               
            }
            break ;
         } else {
            // this was a step from a sub-octet to a parent octet, parent cell (level, ind), level>0
            sid    = (*ind) % 8 ; // current = parent cell as part of the current octet
            POS   *=  HALF ;
            // parent cell cell is at level>0 and thus part of an octet with [0,2] local coordinates
            POS.x += sid % 2 ;     POS.y += (sid/2)%2  ;    POS.z += (sid/4) ;
            
            // printf("Up:  %9.5f %9.5f %9.5f\n", POS.x, POS.y, POS.z) ;
            
            // is the position inside this octet
            // if ((POS.x>=0.0)&&(POS.x<=2.0)&&(POS.y>=0.0)&&(POS.y<=2.0)&&(POS.z>=0.0)&&(POS.z<=0.0)) break ;
            // BUGGY LINE NEVER TRUE !!!!!
            if ((POS.x>=ZERO)&&(POS.x<=TWO)&&(POS.y>=ZERO)&&(POS.y<=TWO)&&(POS.z>=ZERO)&&(POS.z<=TWO)) {
               // it is in current octet but not in the current cell => update ind, local coordinates unchanged
               *ind  += -sid +  4*(int)floor(POS.z)+2*(int)floor(POS.y)+(int)floor(POS.x) ;
               break ;
            }
         }
      } // while not root grid
   } // else -- going up
   // Go down - position *is* inside the current ***CELL***
   // printf("GO DOWN:  %8.4f %8.4f %8.4f  %d %d\n", POS.x, POS.y, POS.z, *level, *ind) ;
   while(DENS[OFF[*level]+(*ind)]<=0.0f) {  // loop until (level,ind) points to a leaf
      // printf("DENS[%d] = %.5e   ", OFF[*level]+(*ind), DENS[OFF[*level]+(*ind)]) ;
      // convert to sub-octet coordinates -- same for transition from root and from parent octet
      POS.x =  TWO*fmod(POS.x, ONE) ;      POS.y =  TWO*fmod(POS.y, ONE) ;      POS.z =  TWO*fmod(POS.z, ONE) ;
      float link = -DENS[OFF[*level]+(*ind)] ;
      *ind       = *(int *)&link ;      // first cell in the sub-octet
      // printf(" => ind %d !!!!!!!!!!!!!!!!!!\n", *ind) ;
      (*level)++ ;
      // printf("SUBOCTET %9.5f %9.5f %9.5f  %d %d  link %.4e", POS.x, POS.y, POS.z, *level, *ind, link) ;
      *ind      += 4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ; // subcell (suboctet)
      // printf("  -->  %d\n", *ind) ;
   }
   pos->x = POS.x ;  pos->y = POS.y ;  pos->z = POS.z ; 
   return ;
}



float GetStep(float3 *POS, const float3 *DIR, int *level, int *ind, 
              __global float *DENS, __constant int *OFF, __global int*PAR) {
   // Calculate step to next cell, update level and ind for the next cell
   // Returns the step length in GL units (units of the root grid)
   // #if (NX>DIMLIM)
   // NX>9999 was there 2020-07-07 .... perhaps should be NX>DIMLIM ??? unless no problems seen
   // HOWEVER -- seems that NX>DIMLIM would have a significant effect on run time...
# if (NX>9999)  // was NX>9999, then NX>DIMLIM  .... but this is very slow????
   // use of DPEPS << PEPS reduces cases where step jumps between non-neighbour cells
   //   == jump over a single intervening cell ... but this still had no effect on scatter tests
   double dx, dy, dz ;
   dx = (DIR->x>0.0f) ? ((1.0+DPEPS-fmod((*POS).x,1.0f))/((*DIR).x)) : ((-DPEPS-fmod((*POS).x,1.0f))/((*DIR).x)) ;
   dy = (DIR->y>0.0f) ? ((1.0+DPEPS-fmod((*POS).y,1.0f))/((*DIR).y)) : ((-DPEPS-fmod((*POS).y,1.0f))/((*DIR).y)) ;
   dz = (DIR->z>0.0f) ? ((1.0+DPEPS-fmod((*POS).z,1.0f))/((*DIR).z)) : ((-DPEPS-fmod((*POS).z,1.0f))/((*DIR).z)) ;
   dx    =  min(dx, min(dy, dz)) ;
   *POS +=  ((float)dx)*(*DIR) ;                    // update LOCAL coordinates - overstep by PEPS
   // step returned in units [GL] = root grid units
   dx     =  ldexp(dx, -(*level)) ;
   Index(POS, level, ind, DENS, OFF, PAR) ;         // update (level, ind)   
   return (float)dx ;                 // step length [GL]
# else
   float dx, dy, dz ;
   dx = (DIR->x>0.0f) ? ((1.0f+PEPS-fmod((*POS).x,1.0f))/((*DIR).x)) : ((-PEPS-fmod((*POS).x,1.0f))/((*DIR).x)) ;
   dy = (DIR->y>0.0f) ? ((1.0f+PEPS-fmod((*POS).y,1.0f))/((*DIR).y)) : ((-PEPS-fmod((*POS).y,1.0f))/((*DIR).y)) ;
   dz = (DIR->z>0.0f) ? ((1.0f+PEPS-fmod((*POS).z,1.0f))/((*DIR).z)) : ((-PEPS-fmod((*POS).z,1.0f))/((*DIR).z)) ;
   dx     =  min(dx, min(dy, dz))  ;
   *POS  +=  dx*(*DIR) ;                    // update LOCAL coordinates - overstep by PEPS
   // printf("MOVE %.3e   %8.4f %8.4f %8.4f   ", dx, POS->x, POS->y, POS->z) ;
   dx     =  ldexp(dx, -(*level)) ;         // step returned in units [GL] = root grid units
   Index(POS, level, ind, DENS, OFF, PAR) ; // update (level, ind)
   // printf(" ==>  %8.4f %8.4f %8.4f   %d %9d\n", POS->x, POS->y, POS->z, *level, *ind) ;
   return dx ;                 // step length [GL]
# endif
}



#else  // @i


not to be used before the question of more zero TABS is resolved

  // 2020-07-18 --- "New" version of Index() and GetStep()
  //   idea was to use only floats and add safeguards (xs, ys, zs below) to ensure
  //   that ray positions close to the borders are not rounded to even numbers
  //   It was as fast as old version ... but results were suspicious for larger, root>512
  //   models where this produced TABS=0.0 for more cells than the "old" method.
  //   When one included also here doubles for NX>DIMLIM, run time on Radeon-VII
  //   increased from 127 to 142 seconds while "old" was at 136 seconds...
  //   and "new" has still more zero TABS values...
  // == new has no advantage in run time, not even on GPU, but the zero-TABS question
  //    should be resolved before it is used... there should be no reason why for the 
  //    same number of photon packages one would have updates ina fewer number of cells
  //  As of 2020-07-19, keep on using the old version!!!
void Index(float3 *pos, const float3 *dir, int *level, int *ind, __global float *DENS, 
           __constant int *OFF, __global int *PAR) {
   float   dx, xs=0.0f, ys=0.0f, zs=0.0f ;
   // int FOLLOW = 139106114 == (*ind) ;
# if (NX>9999) // # if (NX>DIMLIM)
   double3 POS ;   POS.x = pos->x ;  POS.y = pos->y ;  POS.z = pos->z ;
#  define HALF 0.5
#  define ONE  1.0
#  define TWO  2.0
# else
   float3  POS = *pos ;
#  define HALF 0.5f 
#  define ONE  1.0f
#  define TWO  2.0f
# endif
   int     sid ; 
   if ((*level)==0) {   // on root grid
      if ((POS.x<=0.0f)||(POS.x>=NX)||(POS.y<=0.0f)||(POS.y>=NY)||(POS.z<=0.0f)||(POS.z>=NZ)) {
         *ind = -1 ; return ;  // outside the root grid, outside the model
      }
      *ind = (int)floor(POS.z)*NX*NY + (int)floor(POS.y)*NX + (int)floor(POS.x) ; // pos is in cell *ind
      if (DENS[*ind]>0.0f) return ;  // level 0 cell was a leaf -- we are done, otherwise go down below
   } else {
# if 0
      // safeguards to ensure that when coordinate is close to border, it remain inside/outside
      // even when we sometimes add to POS root coordinates that would make EPS round to zero
      dx = POS.x-round(POS.x) ;   if (fabs(dx)<PEPS2)  xs = (dx<0.0f) ? (-PEPS) : (+PEPS) ;
      dx = POS.y-round(POS.y) ;   if (fabs(dx)<PEPS2)  ys = (dx<0.0f) ? (-PEPS) : (+PEPS) ;
      dx = POS.z-round(POS.z) ;   if (fabs(dx)<PEPS2)  zs = (dx<0.0f) ? (-PEPS) : (+PEPS) ;
      // if (FOLLOW) printf("safeguards %.3e %.3e %.3e\n", xs, ys, zs) ;
# endif
# if 1  // --- this does not improve the problem with more zero TABS for levels>0 ---
      // safeguards to ensure that when coordinate is close to border, it remain inside/outside
      // even when we sometimes add to POS root coordinates that would make EPS round to zero
      // changed PEPS2 -> PEPS... and the run will not complete !!
      dx = POS.x-round(POS.x) ;   if (fabs(dx)<PEPS2)  xs = dx ;
      dx = POS.y-round(POS.y) ;   if (fabs(dx)<PEPS2)  ys = dx ;
      dx = POS.z-round(POS.z) ;   if (fabs(dx)<PEPS2)  zs = dx ;
      // if (FOLLOW) printf("safeguards %.3e %.3e %.3e\n", xs, ys, zs) ;
# endif
      while ((*level)>0) {  // step up in the hierarchy until POS is inside the current octet
         *ind = PAR[OFF[*level]+(*ind)-NX*NY*NZ] ;   (*level)-- ;  // parent cell index
         if ((*level)==0) {        // arrived at root grid
            POS    *=  HALF ;      // in local coordinates pos may be rounded from PEPS->0, 1-PEPS->1 !!
            POS.x  +=  (*ind)      % NX ;  // global coordinates
            POS.y  +=  ((*ind)/NX) % NY ; 
            POS.z  +=  (*ind)  / (NX*NY) ;
            if ((POS.x<=0.0f)||(POS.x>=NX)||(POS.y<=0.0f)||(POS.y>=NY)||(POS.z<=0.0f)||(POS.z>=NZ)) {
               *ind = -1 ; 
               return ;  // we left the model volume
            }
            // the position is not necessarily in cell 'ind', could be a neighbour!
            *ind = (int)floor(POS.z)*NX*NY + (int)floor(POS.y)*NX + (int)floor(POS.x) ;
            if (DENS[*ind]>0.0f) {
               pos->x = POS.x ; pos->y = POS.y ; pos->z = POS.z ;
               //*pos = POS ;
               return ;  // found the cell on level 0
            }
            break ; // position is in the current cell (on level 0) but it is still not a leaf
         } else {
            // a step from a sub-octet to a parent octet, parent cell == (*level, *ind)
            //  in octet coordinates [0,2]... any boundary is still multiple of 1.0f
            sid     = (*ind) % 8 ;  // parent cell is part of an octet
            POS    *=  HALF ;
            POS.x  += sid % 2 ;   POS.y += (sid/2)%2  ;   POS.z += sid/4 ;
            // if (FOLLOW) printf("BEFORE   %9.6f %9.6f %9.6f\n", POS.x, POS.y, POS.z) ;
# if 1
            // ****** MAKE SURE POS IS NOT ROUNDED TO INTEGER, check only the coordinate indicated by iside
            // even if we go to another octet, because this was a step between neighbouring cells,
            // the position must be on the border of the new octet as well, in the direction indicated by iside
            if (xs!=0.0f)  POS.x = round(POS.x)+xs ;
            if (ys!=0.0f)  POS.y = round(POS.y)+ys ;
            if (zs!=0.0f)  POS.z = round(POS.z)+zs ;            
# endif
            // if (FOLLOW) printf("AFTER    %9.6f %9.6f %9.6f\n", POS.x, POS.y, POS.z) ;
            // is the position now inside the current octet?
            if ((POS.x>=0.0f)&&(POS.x<=2.0f)&&(POS.y>=0.0f)&&(POS.y<=2.0f)&&(POS.z>=0.0f)&&(POS.z<=2.0f)) {
               // ok, it is in the same octet with the current cell => change the cell index
               *ind  += -sid +  4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;
               break ;  // not necessarily a leaf, we go down the hierarchy below
            }
         }
      } // while not root grid
   } // else -- going up
   // Go down - position *is* inside the current ***CELL**
   // Note -- pos is all the time at the border of the current octet, 
   // could be internal boundary between cells in an octet (but that handled already above!)
   
   // if (FOLLOW) printf("MID      %9.6f %9.6f %9.6f\n", POS.x, POS.y, POS.z) ;
   
   while(DENS[OFF[*level]+(*ind)]<=0.0f) {  // loop until (level,ind) points to a leaf
      // convert to sub-octet coordinates -- same for transition from root and from parent octet
      POS.x =  TWO*fmod(POS.x, ONE) ;  // this would tend to move ray away from the border in local coordinates
      POS.y =  TWO*fmod(POS.y, ONE) ;   
      POS.z =  TWO*fmod(POS.z, ONE) ;
      dx     = -DENS[OFF[*level]+(*ind)] ;  
      *ind   = *(int *)&dx ;           // first cell in the sub-octet
      (*level)++ ;
      // if (FOLLOW) printf("DOWN     %9.6f %9.6f %9.6f\n", POS.x, POS.y, POS.z) ;     
# if 1
      // **** make sure position remains on the correct side of the border
      if (xs!=0.0f)  POS.x = round(POS.x)+xs ;
      if (ys!=0.0f)  POS.y = round(POS.y)+ys ;
      if (zs!=0.0f)  POS.z = round(POS.z)+zs ;            
# endif      
      // if (FOLLOW) printf("FINAL    %9.6f %9.6f %9.6f\n", POS.x, POS.y, POS.z) ;      
      *ind  +=   4*(int)floor(POS.z) + 2*(int)floor(POS.y) + (int)floor(POS.x) ;
   }
   pos->x = POS.x ; pos->y = POS.y ; pos->z = POS.z ;     // *pos = POS ;
   // *pos = POS ;
   return ;
}




float GetStep(float3 *POS, const float3 *DIR, int *level, int *ind, 
              __global float *DENS, __constant int *OFF, __global int*PAR) {
   float dx, dy, dz ;
   dx    =  (DIR->x>0.0f) ? ((1.0f+PEPS-fmod(POS->x,1.0f))/(DIR->x)) : ((-PEPS-fmod(POS->x,1.0f))/(DIR->x)) ;
   dy    =  (DIR->y>0.0f) ? ((1.0f+PEPS-fmod(POS->y,1.0f))/(DIR->y)) : ((-PEPS-fmod(POS->y,1.0f))/(DIR->y)) ;
   dz    =  (DIR->z>0.0f) ? ((1.0f+PEPS-fmod(POS->z,1.0f))/(DIR->z)) : ((-PEPS-fmod(POS->z,1.0f))/(DIR->z)) ;
   dx    =  min(dx, min(dy, dz)) ;
   *POS +=  dx*(*DIR) ;                     // update LOCAL coordinates - POS ~ at cell boundary
   // POS does not have to be very accurate... as long as it tells which coordinate is near the border
   // Index() will explicitly make sure POS remains on the correct side of the boundary, even if we
   // go many levels up and down in the hierarchy
   // 2020-07-19 -- there is a small change that ray goes through a corner and our safeguards in Index()
   //               will bounce the ray back to the original cell
   dx    =  ldexp(dx, -(*level)) ;         // step returned in units [GL] = root grid units   
   
# if 0 // this makes it a little slower... or no effect
   // if step was inside the octet, do not even call Index() ??
   if (*level>0) {
      if ((POS->x>0.0f)&&(POS->x<2.0f)&&(POS->y>0.0f)&&(POS->y<2.0f)&&(POS->z>0.0f)&&(POS->z<2.0f)) {
         *ind += -(*ind)%8 +  4*(int)floor(POS->z)+2*(int)floor(POS->y)+(int)floor(POS->x) ; // d(subcell index)
         if (DENS[*ind]>0.0f) return dx ;
      }
   }
# endif   
   
# if 0
   float3 RPOS = *POS ;
   RootPos(&RPOS, *level, *ind, OFF, PAR)  ;     
   printf("A %d %9d  dx=%.6f   %9.5f %9.5f %9.5f   %9.5f %9.5f %9.5f   %8.4f %8.4f %8.4f\n", *level, *ind,
          dx, POS->x, POS->y, POS->z, DIR->x, DIR->y, DIR->z, RPOS.x, RPOS.y, RPOS.z) ;
# endif
   Index(POS, DIR, level, ind, DENS, OFF, PAR) ; // update (level, ind)
# if 0
   RPOS = *POS ;
   RootPos(&RPOS, *level, *ind, OFF, PAR)  ;  
   printf("  %d %9d  dx=%.6f   %9.5f %9.5f %9.5f   %9.5f %9.5f %9.5f   %8.4f %8.4f %8.4f\n\n", *level, *ind,
          dx, POS->x, POS->y, POS->z, DIR->x, DIR->y, DIR->z, RPOS.x, RPOS.y, RPOS.z) ;
# endif
   return dx ;                 // step length [GL]
}

#endif





void Deflect(float3 *DIR, const float COS_THETA, const float phi) {
   float cx, cy, cz, ox, oy, oz, theta0, phi0, cos_theta, sin_theta, sin_phi, cos_phi ;
   cx = (*DIR).x ;  cy = (*DIR).y ;  cz = (*DIR).z ;
   // Deflection from Z-axis: a vector angle theta from z-axis, phi rotation around z
   sin_theta =  sqrt(1.0f-COS_THETA*COS_THETA) ;
#if 0
   ox        =  sin_theta*cos(phi) ;
   oy        =  sin_theta*sin(phi) ;
   oz        =                  COS_THETA ;
#else
   sin_phi   =  sincos(phi, &cos_phi) ;
   ox        =  sin_theta*cos_phi ;
   oy        =  sin_theta*sin_phi ;
   oz        =  COS_THETA ;
#endif
   // compute direction of the old vector - rotate in the opposite direction: theta0, phi0
   theta0    =  acos(cz/sqrt(cx*cx+cy*cy+cz*cz+DEPS)) ;
   phi0      =  acos(cx/sqrt(cx*cx+cy*cy+DEPS)) ;
   if (((*DIR).y)<0.0f) phi0 = (TWOPI-phi0) ;
   theta0    = -theta0 ;  
   phi0      = -phi0 ;
   // rotate (ox,oy,oz) with angles theta0 and phi0
   // 1. rotate around z angle phi0
   // 2. rotate around x (or y?) angle theta0
#if 0
   sin_theta =  sin(theta0) ;    cos_theta    = cos(theta0) ;
   sin_phi   =  sin(phi0) ;      cos_phi   = cos(phi0) ;
#else
   sin_theta =  sincos(theta0, &cos_theta) ;
   sin_phi   =  sincos(phi0,   &cos_phi) ;
#endif
   (*DIR).x  = +ox*cos_theta*cos_phi   + oy*sin_phi   -  oz*sin_theta*cos_phi ;
   (*DIR).y  = -ox*cos_theta*sin_phi   + oy*cos_phi   +  oz*sin_theta*sin_phi ;
   (*DIR).z  = +ox*sin_theta                          +  oz*cos_theta ;   
}




// Normal case, a single scatter function
// In case of multiple scatter functions, the caller must make the selection
void Scatter(float3 *DIR, constant float *CSC, mwc64x_state_t *rng) 
{
   // generate theta angle CSC[720+rand()*180] -- reference is the third wavelength
   // float tmp ;
   // version using input cos_theta array --- CSC[BINS]
   float cos_theta ;
#if (HG_TEST>0)
   // testing ---- Henyey-Greenstein with a fixed g parameter value
   //  .... no effect on scatter tests with tau>>1, Qabs=0
   float tmp, G=0.65f ;
   tmp        =  (1.0f-G*G)/(1.0f-G+2.0f*G*Rand(rng)) ;
   cos_theta  =  ((1.0f+G*G) - tmp*tmp)/(2.0f*G) ;
#else
   cos_theta  = CSC[clamp((int)floor(Rand(rng)*BINS), 0, BINS-1) ] ;
#endif
   // printf("cos_theta %8.4f\n", cos_theta) ;
   Deflect(DIR, cos_theta, TWOPI*Rand(rng)) ;
   if (fabs((*DIR).x)<DEPS)  (*DIR).x   = DEPS ;     
   if (fabs((*DIR).y)<DEPS)  (*DIR).y   = DEPS ;  
   if (fabs((*DIR).z)<DEPS)  (*DIR).z   = DEPS ;
   *DIR = normalize(*DIR) ;
}



#if (DIR_WEIGHT>0)
# if (WITH_MSF==0)
void WScatter(float3 *DIR, constant float *DSC, mwc64x_state_t *rng, float *pweight)
{
   // Generate new direction wrt OLD DIRECTION OF PROPAGATION, using Henyey-Greenstein
   //    with asymmetry parameter g = DW_A -- DW_A<0 means more 
   //    scatter towards -Z direction
   //    return pweight = p(DSC)/p(HG)   --- probabilities p(theta) without sin(theta) weighting !!
   // 2016-10-24  
   //  in test case DW=0.9  drops <S> from 22.459 to 22.060 !!!!!?????
   // NOTE: does not work with multiple scatter functions (-D WITH_MSF) !!!
   float tmp, phi, cos_theta ;
   phi        =  TWOPI * Rand(rng) ;
#  if 0   // generate new direction relative to -Z, use HG
   tmp        =  (1.0f - DW_A*DW_A) / (1.0f-DW_A+2.0f*DW_A*Rand(rng)) ;
   cos_theta  =  (1.0f + DW_A*DW_A - tmp*tmp) / (2.0f*DW_A+0.000001f) ;  // g>0 => mo$
   (*DIR).z   =  cos_theta ;        // try to direct towards
   tmp        =  sqrt(1.0f-cos_theta*cos_theta) ; // sin_theta
   (*DIR).x   =  tmp*cos(phi) ;
   (*DIR).y   =  tmp*sin(phi) ;
#  else   // generate new direction relative to current DIR, use HG
   tmp        =  (1.0f - DW_A*DW_A) / (1.0f-DW_A+2.0f*DW_A*Rand(rng)) ;
   cos_theta  =  (1.0f + DW_A*DW_A - tmp*tmp) / (2.0f*DW_A+0.000001f) ;  // g>0 => forward
   Deflect(DIR, cos_theta, phi) ;
#  endif
   if (fabs((*DIR).x)<DEPS)  (*DIR).x   = DEPS ;     
   if (fabs((*DIR).y)<DEPS)  (*DIR).y   = DEPS ;  
   if (fabs((*DIR).z)<DEPS)  (*DIR).z   = DEPS ;
   *DIR = normalize(*DIR) ;
   *pweight   = 
     max((1.0f/(4.0f*PI))*(1.0f-DW_A*DW_A)/pow((float)(1.0f+DW_A*DW_A-2.0f*DW_A*cos_theta),1.5f),
         0.000001f);
#  if 0
   // what is the probability according to HG?
   // should correspond to  DSC = dp / dtheta
   //                       DSC = combined_scattering_function2_simple
   //                       DSF2_simple
   // run_SOC.py calls combined_scattering_function2_simple with SIN_WEIGHT=False
   //      -->  DSF2_simple(SIN_WEIGHT=False)
   //      -->  HenyeyGreenstein()   !!!!!!  sin(theta) ***NOT*** included
   *pweight *=  max(tmp*TWOPI, 0.000001f) ;  // result sensitive to normalisation
#  endif
}
# endif
#endif



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
   return 1 ;
}
#endif




__kernel void ZeroAMC(int tag, __global float *TABS, __global float *XAB,
                      __global float *INT,  __global float *INTX, 
                      __global float *INTY, __global float *INTZ) {
   // tag==0  --  between iterations, clear TABS and XAB
   // tag==1  --  between frequencies, clear INT buffers
   int id     = get_global_id(0) ;
   if (id>=CELLS) return ;
   int GLOBAL = get_global_size(0) ;
   if (tag==0) {  // called between iterations
      // if (id==0) printf("CLEAR!\n") ;
      for(int i=id; i<CELLS; i+=GLOBAL) TABS[i] = 0.0f ;
#if (WITH_ALI>0)
      for(int i=id; i<CELLS; i+=GLOBAL) XAB[i]  = 0.0f ;
#endif
   } 
#if ((SAVE_INTENSITY==1)||(SAVE_INTENSITY==2)||(NOABSORBED==0))
   // note:  NOABSORBED==0 =>  INT is used for the per-frequency absorptions
   if (tag==1)  {       // called between frequencies
      for(int i=id; i<CELLS; i+=GLOBAL) INT[i]   = 0.0f ;
# if (SAVE_INTENSITY==2)
      for(int i=id; i<CELLS; i+=GLOBAL) INTX[i]  = 0.0f ;   
      for(int i=id; i<CELLS; i+=GLOBAL) INTY[i]  = 0.0f ;   
      for(int i=id; i<CELLS; i+=GLOBAL) INTZ[i]  = 0.0f ;
# endif
   }
#endif
}




__kernel void Parents(__global   float  *DENS,
                      __constant   int  *LCELLS,  //  8 - number of cells on each level
                      __constant   int  *OFF,
                      __global     int  *PAR
                     ) {
   // Go through the density data in DENS and put to PAR information about the cell parents
   // 2019-02-19 -- PAR no longer allocated to include dummy entries for the root grid cells
   int id     = get_global_id(0) ;
   int GLOBAL = get_global_size(0) ;
   int ind ;
   float link ;
   for(int level=0; level<(LEVELS-1); level++) {     // loop over parent level = level
      for(int ipar=id; ipar<LCELLS[level]; ipar+=GLOBAL) {  // all cells on parent level
         link = DENS[OFF[level]+ipar] ;
         if (link<1.0e-10f) {                         // is a parent to sub-octet
            link = -link ;                            // positive float
            ind  = *(int *)(&link) ;                  // index of first child on level level+1
            for(int i=0; i<8; i++) {                  // loop over cells in the sub-octet
               // OFF[1] == NX*NY*NZ
               PAR[OFF[level+1]-NX*NY*NZ+ind+i] = ipar ;       // link to parent cell
            }                                         // eight cells in sub-octet
         } // if there was a sub-octet
      } // parent cells
   } // parent levels
#if 0
   if (id%101==1) {
      printf("PARENTS => 886247  = %.3e  1886248  = %.3e  2886249  = %.3e\n", 
             DENS[886247], DENS[1886248], DENS[2886249]) ;      
   }
#endif
}




__kernel void AverageDensity(const int lower,
                             __global float *DENS,
                             __constant int *LCELLS,
                             __constant int *OFF,
                             __global int   *PAR) {
   // For each child octet of level lower, replace the value of the parent
   // cell with the average of the eight child cells.
   // printf("*** AverageDensity ***\n") ;
   int id = get_global_id(0) ;
   int gs = get_global_size(0) ;
   float ave ;
   for(int i=8*id; i<LCELLS[lower]; i+=(8*gs)) { // first cell of child octet
      ave = 0.0f ;
      for(int j=0; j<8; j++) ave += DENS[OFF[lower]+i+j] ;
      int pind = PAR[OFF[lower]+i-NX*NY*NZ] ;
      DENS[pind] = ave/8.0f ;   // overwrites the link to the first child cell
   }       
}




__kernel void EqTemperature(const int       level,
                            const float     adhoc,
                            const float     kE,
                            const float     Emin,
                            const int       NE,              // TTT[NE]
                            __global int   *OFF,
                            __global int   *LCELLS,
                            __global float *TTT,
                            __global float *DENS,
                            __global float *EMIT,
                            __global float *TNEW) {
   int  id = get_global_id(0) ;
   int  gs = get_global_size(0) ;
   if (id>=CELLS) return ;
   float scale   = (6.62607e-27f*FACTOR)/LENGTH ;
   float oplgkE  = 1.0f/log10(kE) ;
   float wi, beta=1.0f, Ein ;
   int   iE ;
   int   ind ;    // int32  +2,147,483,647   uint32 4,294,967,295
   // for(int level=0; level<LEVELS; level++) {
   for(int i=id; i<LCELLS[level]; i+=gs) {   // loop over cells on a given level
      ind       =  OFF[level] + i ;
      Ein       =  (scale/adhoc) * EMIT[ind] * pown(8.0f, level) / DENS[ind] ; // 1e10 == ADHOC on host !!!!
      iE        =  clamp((int)floor(oplgkE * log10((Ein/beta)/Emin)), 0, NE-2) ;
      wi        =  (Emin*pown(kE,iE+1)-(Ein/beta)) / (Emin*pown(kE, iE)*(kE-1.0f)) ;
      // printf("wi %8.4f\n", wi) ;
      if ((ind<0)||(ind>=CELLS)) printf("????\n") ;
      TNEW[ind] =  (DENS[ind]>1.0e-7f) ?  clamp(wi*TTT[iE] + (1.0f-wi)*TTT[iE+1], 3.0f, 1600.0f) : (10.0f) ;
      if ((TNEW[ind]>1.0f)&&(TNEW[ind]<1000.0f)) {
         ;
      } else {
         printf("???    Ein %12.4e  [%12.4e, %12.4e]         T = %.3e,  NE=%d, kE=%.6f\n",
                             Ein,   Emin, Emin*pown(kE, NE), TNEW[ind], NE,   kE) ;
      }
   }
   // }
}



__kernel void Emission(const float     FREQ,
                       const float     FABS,
                       __global float *DENS,
                       __global float *T,
                       __global float *EMIT) {
   // Calculate emission based on temperature (non-equilibrium grains)
   int id = get_global_id(0) ;
   int gs = get_global_size(0) ;
   if (id>CELLS) return ;
   for(int i=id; i<CELLS; i+=gs) {
      // 1.0e20*4.0*PI/(PLANCK*FFREQ[a:b])) * FABS[a:b]*Planck(FFREQ[a:b], TNEW[icell])/(USER.GL*PARSEC)
      // 1e20 * 4 *pi / (h*f) = 1.8965044e+47
      EMIT[i] = (2.79639459e-20f*FACTOR) * FABS * (FREQ*FREQ/(exp(4.7995074e-11f*FREQ/T[i])-1.0f)) / LENGTH ;
   }
}



#if 0
__kernel void Emission2(const float     FREQ,
                        const float     FABS,
                        __global float *DENS,
                        __global float *T,
                        __global float *EMIT) {
   // This version
   int id = get_global_id(0) ;
   int gs = get_global_size(0) ;
   if (id>CELLS) return ;
   for(int i=id; i<CELLS; i+=gs) {
      // 1.0e20*4.0*PI/(PLANCK*FFREQ[a:b])) * FABS[a:b]*Planck(FFREQ[a:b], TNEW[icell])/(USER.GL*PARSEC)
      // 1e20 * 4 *pi / (h*f) = 1.8965044e+47
      EMIT[i] =  (2.79639459e-20f*FACTOR) * FABS * (FREQ*FREQ/(exp(4.7995074e-11f*FREQ/T[i])-1.0f)) / LENGTH ;
   }
}
#endif




void Surface(float3 *POS, float3 *DIR) {
   // Assuming the initial position is outside the cloud, step to nearest boundary and
   // return that new position. If ray misses the cloud.... position will remain outside the cloud.
   float dx, dy, dz ;
   if (DIR->x>0.0f) {
      if (POS->x<0.0f)  dx = (PEPS - POS->x)/DIR->x ;
      else              dx = -1.0e10f ;
   } else {
      if (POS->x>NX)    dx = (NX-PEPS - POS->x)/DIR->x ;
      else              dx = -1.0e10f ;
   }
   if (DIR->y>0.0f) {
      if (POS->y<0.0f)  dy = (PEPS - POS->y)/DIR->y ;
      else              dy = -1.0e10f ;
   } else {
      if (POS->y>NY)    dy = (NY-PEPS - POS->y)/DIR->y ;
      else              dy = -1.0e10f ;
   }
   if (DIR->z>0.0f) {
      if (POS->z<0.0f)  dz = (PEPS - POS->z)/DIR->z ;
      else              dz = -1.0e10f ;
   } else {
      if (POS->z>NZ)    dz = (NZ-PEPS - POS->z)/DIR->z ;
      else              dz = -1.0e10f ;
   }
   // If ray hits the cloud, the longest step should bring one to border
   dx = max(dx, max(dy, dz)) ;
   *POS += dx*(*DIR) ;
}




int Angles2PixelRing(const int nside, float phi, float theta) {
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
   nl2 = 2*nside ;
   nl4 = 4*nside ;
   ncap  = nl2*(nside-1) ;   //  number of pixels in the north polar cap
   npix  = 12*nside*nside ;
   if (za<=TWOTHIRD) {  //  Equatorial region ------------------
      jp = (int)(nside*(0.5f+tt-z*0.75f)) ;   // ! index of  ascending edge line
      jm = (int)(nside*(0.5f+tt+z*0.75f)) ;   // ! index of descending edge line
      ir = nside + 1 + jp - jm ;   // ! in {1,2n+1} (ring number counted from z=2/3)
      kshift = 0 ;
      if (ir%2==0) kshift = 1 ;    // ! kshift=1 if ir even, 0 otherwise
      ip = (int)( ( jp+jm - nside + kshift + 1 ) / 2 ) + 1 ;   // ! in {1,4n}
      if (ip>nl4) ip -=  nl4 ;
      ipix1 = ncap + nl4*(ir-1) + ip ;
   } else {  // ! North & South polar caps -----------------------------
      tp  = tt - (int)(tt)  ;    // !MOD(tt,1.d0)
      tmp = sqrt( 3.0f*(1.0f - za) ) ;
      jp = (int)(nside*tp         *tmp ) ;   // ! increasing edge line index
      jm = (int)(nside*(1.0f - tp)*tmp ) ;   // ! decreasing edge line index
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


bool Pixel2AnglesRing(const int nside, const int ipix, float *phi, float *theta) {
   // Convert Healpix pixel index to angles (phi, theta), theta=0.5*pi-lat, phi=lon
   // Map must be in RING order.
   int    nl2, nl4, npix, ncap, iring, iphi, ip, ipix1 ;
   float  fact1, fact2, fodd, hip, fihip ;
   npix = 12*nside*nside;      // ! total number of points
   // if ((ipix<0)||(ipix>(npix-1))) return false ;
   ipix1 = ipix + 1 ;    // ! in {1, npix}
   nl2   = 2*nside ;
   nl4   = 4*nside ;
   ncap  = 2*nside*(nside-1) ;  // ! points in each polar cap, =0 for nside =1
   fact1 = 1.5f*nside ;
   fact2 = 3.0f*nside*nside  ;
   if (ipix1<=ncap) {   // ! North Polar cap -------------
      hip   = ipix1/2.0f ;
      fihip = (int)( hip ) ;
      iring = (int)( sqrt( hip - sqrt(fihip) ) ) + 1 ;  // ! counted from North pole
      iphi  = ipix1 - 2*iring*(iring - 1) ;
      *theta= acos( 1.0f - iring*iring / fact2 ) ;
      *phi  = (iphi - 0.5f) * PI/(2.0f*iring) ;
   } else {
      if (ipix1<=nl2*(5*nside+1)) { // ! Equatorial region ------
         ip    = ipix1 - ncap - 1 ;
         iring = (int)( ip / nl4 ) + nside ;   // ! counted from North pole
         iphi  = (ip%nl4) + 1 ;
         fodd  = 0.5f * (1 + (iring+nside)%2) ;  // ! 1 if iring+nside is odd, 1/2 otherwise
         *theta= acos( (nl2 - iring) / fact1 ) ;
         *phi  = (iphi - fodd) * PI /(2.0f*nside) ;
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




int InRoi(const int level, const int ind, constant int *ROI, __global int *PAR, __constant int *OFF) {
   // Return true if cell (level, ind) is inside ROI=[x0,x1,y0,y1,z0,z1]
   int i, j, k ;
   // Go up to the root grid
   k = level ;  i = ind ;
   while(k>0) {
      i = PAR[OFF[k]+i-NX*NY*NZ] ;  k-- ;
   }
   // Calculate i,j,k indices of the root grid cell
   k   =  i / (NX*NY) ;
   j   =  (i/NX) % NY ;
   // i   =  i % NX ;
   if (((i%NX)>=ROI[0])&&((i%NX)<=ROI[1])&&(j>=ROI[2])&&(j<=ROI[3])&&(k>=ROI[4])&&(k<=ROI[5])) {
      return i ;  // root grid index
   } else {
      return -1 ;
   }
}





