#define DEBUG 0
#define NULL 0
#define DIMLIM 29
#define real  float   // Eq.Eq.!!

#define F2I(x)    (*(__global int*)&x)   //  __global BUFFER -> OTI
#define I2F(x)    (*(float *)&x)         //           OTI    -> __global BUFFER

#if (DOUBLE_POS<1)
# define  ZERO   0.0f
# define  HALF   0.5f
# define  ONE    1.0f
# define  TWO    2.0f
# define  REAL   float
# define  REAL3  float3
#else  // ELSE DOUBLE_POS
# define  ZERO   0.0
# define  HALF   0.5
# define  ONE    1.0
# define  TWO    2.0
# define  REAL   double
# define  REAL3  double3
#endif



#if (WITH_OCTREE>0) // ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// TEPS, TTEPS for ray splitting:
//    TEPS   =  check which side ray entered the cell (high precision not needed)
//    TTEPS  =  2-TEPS

# if (DOUBLE_POS<0) 

constant float  EPS    =  0.000013f ;  //  1.0e-5 !!
constant float  OMEPS  =  0.999987f ;
constant float  OPEPS  =  1.000013f ;
constant float  TMEPS  =  1.999987f ;
constant float  TPEPS  =  2.000013f ;
constant float  EPS2   =  0.000030f ;  // PEPS2 = PEPS*2
constant float  TMEPS2 =  1.999970f ;



# else    // DOUBLE_POS currently for WITH_OCTREE=4 only


#  if 0
// 2020-09-04
constant double EPS    =  0.000009 ;  
constant double OMEPS  =  0.999991 ;
constant double TMEPS  =  1.999991 ;
constant double EPS2   =  0.000030 ;
constant double TMEPS2 =  1.999970 ;  
#  endif
#  if 1
// better
constant double EPS    =  0.0000002 ;  
constant double OMEPS  =  0.9999998 ;
constant double TMEPS  =  1.9999998 ;  // 2-PEPS
constant double EPS2   =  0.0000100 ;  // PEPS2 = PEPS*2
constant double TMEPS2 =  1.9999900 ;
#  endif
#  if 0
// similar to the above
constant double EPS    =  0.0000001 ;  
constant double OMEPS  =  0.9999999 ;
constant double TMEPS  =  1.9999998 ;  // 2-PEPS
constant double EPS2   =  0.0000030 ;  // PEPS2 = PEPS*2
constant double TMEPS2 =  1.9999970 ;
#  endif


# endif


#else  // else not octree  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#if (NX<DIMLIM)

# define PEPS              0.0002f
# define EPS               0.0002f
# define TMEPS             1.9998f
# define EPS2              0.0008f
# define TMEPS2            1.9992f
# define DEPS              0.00002f 

#else

# define PEPS              0.0002f
# define EPS               0.0002f
# define OMEPS             0.9998f
# define TMEPS             1.9997f
# define EPS2              0.0008f
# define TMEPS2            1.9992f
# define DEPS              0.00004f 

#endif


#endif  // ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------







#define UNROLL   1
#define H_K      4.799243348e-11f
#define TWOPI    6.28318531f
#define PIHALF   1.5707963268f
#define TWOTHIRD 0.6666666667f
#define PI       3.1415926535897f

constant float  PHOTON_LIMIT = 1.0e-30f ;


// matrix indices in solver, assumes row order ... LEVELS = number of levels in the molecule !!
#define IDX(a,b)  ((a)*LEVELS+(b)) 



inline void AADD(volatile __global float *addr, float val)
{
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
}





void IndexG(REAL3 *pos, int *level, int *ind, __global float *DENS, __constant int *OFF) {
   // Return cell level and index for given global position
   //    pos    =  on input global coordinates, returns either
   //              global coordinates (root grid) or relative within current octet
   //    level  =  hierarchy level on which the cell resides, 0 = root grid
   //    ind    =  index to global array of densities for cells at this level
   //    DENS, OFF needed for the OT hierarchy
   // Note: this must return accurate coordinates even when position is outside the volume,
   //       because that is used to mirror and restart level-0 rays
   // first the root level
   float link ;
   *ind = -1 ;
   *level = 0 ;
   if ((pos->x<=ZERO)||(pos->y<=ZERO)||(pos->z<=ZERO)) return  ;
   if ((pos->x>=NX)  ||(pos->y>=NY)  ||(pos->z>=NZ))   return  ;
   *ind   = (int)floor((*pos).z)*NX*NY + (int)floor((*pos).y)*NX + (int)floor((*pos).x) ;
   if (DENS[*ind]>0.0f) {
      return ;  // found a leaf
   }
   while(1) {
      // go down, coordinates in sub-octet
      pos->x = TWO*fmod(pos->x, ONE) ;  // coordinate inside root grid cell [0,1]
      pos->y = TWO*fmod(pos->y, ONE) ;
      pos->z = TWO*fmod(pos->z, ONE) ;
      // new indices
      link   =  -DENS[OFF[*level]+(*ind)] ;
      *ind   =  *(int *)(&link );  // index **WITHIN THE LEVEL*** (not global index); first subcell
      (*level)++ ;
      *ind  +=  4*(int)floor((*pos).z) + 2*(int)floor((*pos).y) + (int)floor((*pos).x) ; // actual cell
      if (DENS[OFF[*level]+(*ind)]>0.0f) {
         return ; // found a leaf
      }
   }
}


void IndexGR(REAL3 *pos, int *level, int *ind, __global float *DENS, __constant int *OFF) {
   // Return cell level and index for given global position.
   // This version return (OTL, OTI, POS) for a ROOT GRID CELL, even if that is refined.
   //    pos    =  on input global coordinates, returns either
   //              global coordinates (root grid) or relative within current octet
   //    level  =  hierarchy level on which the cell resides, 0 = root grid
   //    ind    =  index to global array of densities for cells at this level
   //    DENS, OFF needed for the OT hierarchy
   // Note: this must return accurate coordinates even when position is outside the volume,
   //       because that is used to mirror and restart level-0 rays
   // first the root level
   *ind   = -1 ;
   *level =  0 ;
   if ((pos->x<=ZERO)||(pos->y<=ZERO)||(pos->z<=ZERO)) return  ;
   if ((pos->x>=NX)  ||(pos->y>=NY)  ||(pos->z>=NZ))   return  ;
   *ind   = ((int)floor(pos->z))*NX*NY + ((int)floor(pos->y))*NX + ((int)floor(pos->x)) ;
   return ;  // stay on the root grid
}



void RootPos(REAL3 *POS, const int ilevel, const int iind, __constant int *OFF, __global int *PAR) {
   // Given current indices (ilevel, iind) and current position POS (current level), return the
   // position in root grid coordinates.
   int level=ilevel, ind=iind, sid ;
   if (level==0) return ;  // no change, POS is already position in root grid coordinates
   // Otherwise go UP in the grid hierarchy
   while (level>0) {
      ind  =  PAR[OFF[level]+ind-NX*NY*NZ] ;  level-- ;  // parent cell index
      if (level==0) {              // arrived at root grid
         *POS     *=  HALF ;       // sub-octet coordinates [0,2] into global coordinates [0,1]
         (*POS).x +=  ind      % NX  ;
         (*POS).y +=  (ind/NX) % NY ;
         (*POS).z +=  ind  / (NX*NY) ;
         return ;
      } else {  // step from a sub-octet to a parent octet, parent cell == ind
         sid       =  ind % 8 ;    // index of the sub-cell, child of current cell ind
         *POS     *=  HALF ;       // from sub-octet to parent cell, positions *= 0.5
         (*POS).x +=  sid % 2 ;   (*POS).y += (sid/2)%2  ;   (*POS).z += sid/4 ; // add offset based on subcell
      } 
   } // while not root grid
}



void IndexUP(REAL3 *pos, int *level, int *ind,
             __global float *DENS, __constant int *OFF, __global int *PAR, const int parent_otl) {
   // For coordinates inside a cell (level, ind, pos), return the same for a parent cell
   // at level parent_otl. We require that the initial (level, ind, pos) correspond to a
   // position in the model and parent_otl >= 1. We will not ever rise to root grid level!!
   int sid ;
   while ((*level)>parent_otl) {   // go UP until the position is at the requested level
      *ind   = PAR[OFF[*level]+(*ind)-NX*NY*NZ] ;  (*level)-- ;  // parent cell index
      // this was a step from a sub-octet to a parent octet, parent cell == ind
      sid     = (*ind) % 8 ;  // current sub-cell 0-7 in the current octet
      (*pos) *=  HALF ;
      pos->x += sid % 2 ;  pos->y += (sid/2)%2  ;  pos->z += sid/4 ;
#if 0
      if ((*level)%2==0) {  
         pos->x = clamp(pos->x, EPS, TMEPS) ;
         pos->y = clamp(pos->y, EPS, TMEPS) ;
         pos->z = clamp(pos->z, EPS, TMEPS) ;
      }
#endif      
   }
}










#if (DOUBLE_POS==0)


// (DOUBLE_POS==0)
void IndexOT(float3 *pos, int *level, int *ind,
             __global float *DENS, __constant int *OFF, __global int *PAR, const int max_otl, 
             const REAL3 *POS_LE,  const float3 *DIR, const int LEADING) {
   // Return the level and index of a neighbour cell based on pos
   //   (level, ind) are the current cell coordinates, pos the local coordinates
   //   on return these will be the new coordinates or ind=-1 if one is outside the model volume.
   // Assume that one has already checked that the neighbour is not just another cell in the current octet
   // Note: if the position is outside the model volume, routine must return
   //       level==0, ind<0, and pos containing accurate root grid coordinates
   // One will not go below max_otl level, even if the cell is further subdivided.
   int sid ;

   // Need a better Index() that works for root grids ~1000 and hierarchies with ~10 levels !
   REAL3  POS ;  // we have EPS~1e-5 ... but we may loose ~3 digits when we step up to root level => need double
   POS.x = pos->x ;   POS.y = pos->y ;   POS.z = pos->z ;
   
   if (*level==0) {   // on root grid
      if ((pos->x<=0.0f)||(pos->x>=NX)||(pos->y<=0.0f)||(pos->y>=NY)||(pos->z<=0.0f)||(pos->z>=NZ)) {
         *ind = -1 ; return ;  // ok -- pos contains accurate root grid coordinates, even outside the volume
      }
      *ind = (int)floor((*pos).z)*NX*NY + (int)floor((*pos).y)*NX + (int)floor((*pos).x) ;
      if (DENS[*ind]>0.0f) return ;  // level 0 cell was a leaf -- we are done
   } else {     
      while ((*level)>0) {           // go UP until the position is inside the octet
         *ind = PAR[OFF[*level]+(*ind)-NX*NY*NZ] ;  (*level)-- ;  // parent cell index
         if ((*level)==0) {                // arrived at root grid
            POS   *=  HALF ;               // convertion from sub-octet to global coordinates
            POS.x +=  (*ind)      % NX  ;  // real root grid coordinates
            POS.y +=  ((*ind)/NX) % NY ;
            POS.z +=  (*ind) / (NX*NY) ;
            if ((POS.x<=0.0f)||(POS.x>=NX)||(POS.y<=0.0f)||(POS.y>=NY)||(POS.z<=0.0f)||(POS.z>=NZ)) {
               *ind = -1 ;   // we came up to root grid cell ind, but POS was outside model volume
               pos->x = POS.x ;  pos->y = POS.y ;  pos->z = POS.z ;
               return ;  // we left the model volume -- pos still containing proper coordinates = ok !!!
            }
            // the position is not necessarily in cell 'ind', could be a neighbour!
            *ind = (int)floor(POS.z)*NX*NY + (int)floor(POS.y)*NX + (int)floor(POS.x) ;
            if (DENS[*ind]>0.0f) {
               pos->x = POS.x ;  pos->y = POS.y ;  pos->z = POS.z ;
               return ;  // found the cell on level 0               
            }
            break ;
         } else {
            // this was a step from a sub-octet to a parent octet, parent cell == ind
            sid    = (*ind) % 8 ; // parent cell = current cell in current octet
            POS   *=  HALF ;
            POS.x += sid % 2 ;  POS.y += (sid/2)%2  ;  POS.z += sid/4 ;
            // is the position inside this octet?
            if ((POS.x>0.0f)&&(POS.x<2.0f)&&(POS.y>0.0f)&&(POS.y<2.0f)&&(POS.z>0.0f)&&(POS.z<2.0f)) {
               // BugError POS.z<=0.0f fixed 2020-07-18
               // changed <= to < because floor(POS.*) must never be 2
               // we must update the index to that of the cell that has the ray !!
               *ind += -sid + 4*(int)(floor(POS.z)) + 2*(int)(floor(POS.y)) + (int)(floor(POS.x)) ;
               break ;
            }
         }
      } // while not root grid
   } // else -- going up
   
   
   
   
#  if (FIX_PATHS>0)
   // try to reduce noise in the positions
   // ONE SHOULD ALSO SET THE LEADING COORDINATE TO  round(POS)+/-EPS ??????????????????????
   // IndexOT() IS CALLED ONLY FROM GetStepOT, WHERE RAY IS ALWAYS ASSUMED TO BE ON CELL BOUNDARY,
   // OR FROM OT4 just TO CHECK THE LEVEL OF NEIGHBOURING CELLS
   if ((*level==0)&&(POS_LE!=NULL)) {
      float s, dx, dy, dz ;
      float3 POS00 = POS ;
      dx = fabs(round(POS.x)-POS.x) ;  dy = fabs(round(POS.y)-POS.y) ;  dz = fabs(round(POS.z)-POS.z) ;
      // try to precise the position... the coordinate closest to border is kept fixed !!!
      if ((dx<dy)&&(dx<dz)) {   // we are at x-boundary
         if (LEADING<2) {       // correct only if we are on x boundary when x is the leading direction
            if (LEADING==0) {
               // POS.x =  round(POS.x)+EPS ;           
               s     =  POS.x/DIR->x ;
            } else {
               // POS.x =  round(POS.x)-EPS ;            
               s     =  (POS.x-NX)/DIR->x ;
            }
            POS.y =  POS_LE->y + s*DIR->y ;    POS.z =  POS_LE->z + s*DIR->z ;
            if (POS.y<0.0f) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;
            if (POS.z<0.0f) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
         }
      } else {
         if (dy<dz) {
            if ((LEADING==2)||(LEADING==3)) {
               if (DIR->y>0.0f) {
                  // POS.y =  round(POS.y)+EPS ;         
                  s     =  POS.y/DIR->y ;
               } else {
                  // POS.y =  round(POS.y)-EPS ;            
                  s     =  (POS.y-NY)/DIR->y ;
               }
               POS.x =  POS_LE->x + s*DIR->x ;    POS.z =  POS_LE->z + s*DIR->z ;
               if (POS.x<0.0f) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
               if (POS.z<0.0f) POS.z+= NZ ;  else  if (POS.z>NZ)   POS.z-= NZ ;
            }
         } else {
            if (LEADING>3) {
               if (DIR->z>0.0f) {
                  // POS.z =  round(POS.z)+EPS ;  
                  s     =  POS.z/DIR->z ;
               } else {
                  // POS.z =  round(POS.z)-EPS ;           
                  s     =  (POS.z-NZ)/DIR->z ;
               }
               POS.x =  POS_LE->x + s*DIR->x ;    POS.y =  POS_LE->y + s*DIR->y ;
               if (POS.x<0.0f) POS.x+= NX ;  else  if (POS.x>NX)   POS.x-= NX ;
               if (POS.y<0.0f) POS.y+= NY ;  else  if (POS.y>NY)   POS.y-= NY ;
            }
         }
      }
#   if (DEBUG>0)
      printf("%8.5f %8.5f %8.5f   %8.5f %8.5f %8.5f      %9.6f %9.6f %9.6f\n", 
             POS.x, POS.y, POS.z, POS00.x, POS00.y, POS00.z,
             POS.x-POS00.x, POS.y-POS00.y, POS.z-POS00.z) ;
#   endif
   }
#  endif  // if FIX_PATHS
   

   
   
#  if (DEBUG>0)
   printf("ZZZZ  %d %9d --- %9.6f %9.6f %9.6f     <------ at root level \n", *level, *ind, POS.x, POS.y, POS.z) ;
#  endif
   
   
   
   // Go down - position *is* inside the current octet
   while((*level<max_otl)&&(DENS[OFF[*level]+(*ind)]<=0.0f)) {  // loop until (level,ind) points to a leaf
      // convert to sub-octet coordinates -- same for transition from root and from parent octet
#  if (NX>DIMLIM)
      POS.x =  TWO*fmod(POS.x, ONE) ;
      POS.y =  TWO*fmod(POS.y, ONE) ;
      POS.z =  TWO*fmod(POS.z, ONE) ;
#  else
      POS.x =  clamp(TWO*fmod(POS.x, ONE), EPS, TMEPS) ;
      POS.y =  clamp(TWO*fmod(POS.y, ONE), EPS, TMEPS) ;
      POS.z =  clamp(TWO*fmod(POS.z, ONE), EPS, TMEPS) ;
#  endif
      
      // if (((*ind)==8478)&&((*level)==1)) printf("*** 1, 8478 \n") ;
      float link = -DENS[OFF[*level]+(*ind)] ;
      *ind       = *(int *)&link ;      // first cell in the sub-octet
      (*level)++ ;
      *ind    += 4*(int)floor(POS.z)+2*(int)floor(POS.y)+(int)floor(POS.x) ; // subcell (suboctet)
   }        
      
   
   
#  if 0
   // enforce initial closeness to border to result in similar closeness (PEPS) in the end result
   // printf("XXXX  %d %9d --- %9.6f %9.6f %9.6f \n", *ind, *level, POS.x, POS.y, POS.z) ;
   if (fabs(xbo)<EPS2)  POS.x = round(POS.x) + sign(xbo)*EPS ;
   if (fabs(ybo)<EPS2)  POS.y = round(POS.y) + sign(ybo)*EPS ;
   if (fabs(zbo)<EPS2)  POS.z = round(POS.z) + sign(zbo)*EPS ;
#  endif
   
   
   
#  if (DEBUG>0)
   printf("YYYY  %d %9d --- %9.6f %9.6f %9.6f     <------ final leaf level\n", *level, *ind, POS.x, POS.y, POS.z) ;
#  endif
   
   
   (*pos).x = POS.x ;  (*pos).y = POS.y ;  (*pos).z = POS.z ;
   
   return ;
}




// (DOUBLE_POS==0)
float GetStepOT(float3 *POS, const float3 *DIR, int *level, int *ind,
                __global float *DENS, __constant int *OFF, __global int*PAR, const int max_otl,
                const REAL3 *POS_LE, const int LEADING) {
   // Calculate step to next cell, update level and ind for the next cell.
   // Returns the step length in GL units (units of the root grid). POS is updated.
   // Note --- even when one steps outside the model volume, POS should be accurate
   //          because that may be used to start a new ray on the opposite side !!!
   // max_otl => do not go to level > max_otl, even if the cell is not yet a leaf

#  if (DOUBLE_POS>0)
   double dx, dy, dz ;
   // try to step over the border, with distance EPS to the border
   dx = ((DIR->x)>0.0)  ?  ((1.0+EPS-fmod(POS->x,1.0f))/(double)(DIR->x))  :  ((-EPS-fmod(POS->x,1.0f))/(double)(DIR->x)) ;
   dy = ((DIR->y)>0.0)  ?  ((1.0+EPS-fmod(POS->y,1.0f))/(double)(DIR->y))  :  ((-EPS-fmod(POS->y,1.0f))/(double)(DIR->y)) ;
   dz = ((DIR->z)>0.0)  ?  ((1.0+EPS-fmod(POS->z,1.0f))/(double)(DIR->z))  :  ((-EPS-fmod(POS->z,1.0f))/(double)(DIR->z)) ;
   dx =  min(dx, min(dy, dz)) ;
#  if 0
   int follow=0 ;
   if (dx<1.0e-4) follow=1 ;
   if (follow>0) printf("\n\nA ----- %2d %7d ---  POS %9.6f %9.6f %9.6f   DIR %8.4f %8.4f %8.4f   dx=%.4e\n",
                        *level, *ind, POS->x, POS->y, POS->z, DIR->x, DIR->y, DIR->z, dx) ;
#  endif
   POS->x +=  dx*DIR->x ;   POS->y +=  dx*DIR->y ;   POS->z +=  dx*DIR->z ;
   dx =  ldexp(dx, -(*level)) ;   // step returned in units [GL] = root grid units
#  if 0
   if (follow>0) printf("B ----- %2d %7d ---  POS %9.6f %9.6f %9.6f   DIR %8.4f %8.4f %8.4f   DX=%.4e\n", 
                        *level, *ind, POS->x, POS->y, POS->z, DIR->x, DIR->y, DIR->z, dx) ;
#  endif
   IndexOT(POS, level, ind, DENS, OFF, PAR, max_otl, POS_LE, DIR, LEADING) ;       // update (level, ind)   
#  if 0
   if (follow>0) printf("C ----- %2d %7d ---  POS %9.6f %9.6f %9.6f   DIR %8.4f %8.4f %8.4f   DX=%.4e\n", 
                        *level, *ind, POS->x, POS->y, POS->z, DIR->x, DIR->y, DIR->z, dx) ;
#  endif
   return (float)dx ;                 // step length [GL]
# else

   
   float dx ;
   dx =                (DIR->x<ZERO) ? (-fmod(POS->x,ONE)/DIR->x-EPS/DIR->x) : ((ONE-fmod(POS->x,ONE))/DIR->x+EPS/DIR->x) ;
   dx = min(dx,(float)((DIR->y<ZERO) ? (-fmod(POS->y,ONE)/DIR->y-EPS/DIR->y) : ((ONE-fmod(POS->y,ONE))/DIR->y+EPS/DIR->y))) ;
   dx = min(dx,(float)((DIR->z<ZERO) ? (-fmod(POS->z,ONE)/DIR->z-EPS/DIR->z) : ((ONE-fmod(POS->z,ONE))/DIR->z+EPS/DIR->z))) ;
   dx = clamp(dx, (float)EPS, 2.0f) ;
   
#  if 0
   int follow=0 ;
   if (dx<1.0e-3) follow=1 ;
   if (follow>0) printf("\n\nA ----- %2d %7d ---  POS %9.6f %9.6f %9.6f   DIR %8.4f %8.4f %8.4f   dx=%.4e\n",
                        *level, *ind, POS->x, POS->y, POS->z, DIR->x, DIR->y, DIR->z, dx) ;
#  endif
   *POS  +=  dx*(*DIR) ;                               // update LOCAL coordinates - overstep by EPS
   dx     =  ldexp(dx, -(*level)) ;                    // step returned in units [GL] = root grid units
   
#  if 0
   if (follow>0) printf("B ----- %2d %7d ---  POS %9.6f %9.6f %9.6f   DIR %8.4f %8.4f %8.4f   DX=%.4e\n", 
                        *level, *ind, POS->x, POS->y, POS->z, DIR->x, DIR->y, DIR->z, dx) ;
#  endif
   
   IndexOT(POS, level, ind, DENS, OFF, PAR, max_otl, POS_LE, DIR, LEADING) ; // update (level, ind)
   
#  if 0
   if (follow>0) printf("C ----- %2d %7d ---  POS %9.6f %9.6f %9.6f   DIR %8.4f %8.4f %8.4f   DX=%.4e\n", 
                        *level, *ind, POS->x, POS->y, POS->z, DIR->x, DIR->y, DIR->z, dx) ;
#  endif
   
   return dx ;                 // step length [GL]
   // ==============================================================================================================================
# endif // =====================================================================================================================
}










#else // if (DOUBLE_POS>0)   @d   WITH_OCTREE=4  uses DOUBLE_POS==1





// (DOUBLE_POS>0)
void IndexOT(REAL3 *pos, int *level, int *ind,
             __global float *DENS, __constant int *OFF, __global int *PAR, const int max_otl, 
             const REAL3 *POS_LE,  const float3 *DIR, const int LEADING) {
   // Return the level and index of a neighbour cell based on pos
   //   (level, ind) are the current cell coordinates, pos the local coordinates
   //   on return these will be the new coordinates or ind=-1 if one is outside the model volume.
   // Assume that one has already checked that the neighbour is not just another cell in the current octet
   // Note: if the position is outside the model volume, routine must return
   //       level==0, ind<0, and pos containing accurate root grid coordinates
   // One will not go below max_otl level, even if the cell is further subdivided.
   int sid ;
   double3  POS ;  // we have EPS~1e-5 ... but we may loose ~3 digits when we step up to root level => need double
   POS.x = pos->x ;   POS.y = pos->y ;   POS.z = pos->z ;   
   if (*level==0) {   // on root grid
      if ((pos->x<=ZERO)||(pos->x>=NX)||(pos->y<=ZERO)||(pos->y>=NY)||(pos->z<=ZERO)||(pos->z>=NZ)) {
         *ind = -1 ; return ;  // ok -- pos contains accurate root grid coordinates, even outside the volume
      }
      *ind = (int)floor(pos->z)*NX*NY + (int)floor(pos->y)*NX + (int)floor(pos->x) ;
      if (DENS[*ind]>0.0f) return ;  // level 0 cell was a leaf -- we are done
   } else {     
      while ((*level)>0) {           // go UP until the position is inside the octet
         *ind = PAR[OFF[*level]+(*ind)-NX*NY*NZ] ;  (*level)-- ;  // parent cell index
         if ((*level)==0) {                // arrived at root grid
            POS   *=  HALF ;               // convertion from sub-octet to parent coordinates = root coordinates
            POS.x +=  (*ind)      % NX  ;  // real root grid coordinates
            POS.y +=  ((*ind)/NX) % NY ;
            POS.z +=  (*ind) / (NX*NY) ;
            if ((POS.x<=ZERO)||(POS.x>=NX)||(POS.y<=ZERO)||(POS.y>=NY)||(POS.z<=ZERO)||(POS.z>=NZ)) {
               *ind = -1 ;   // we came up to root grid cell ind, but POS was outside model volume
               pos->x = POS.x ;
               pos->y = POS.y ;
               pos->z = POS.z ;
               return ;  // we left the model volume -- pos still containing proper coordinates = ok !!!
            }
            // the position is not necessarily in cell 'ind', could be a neighbour
            *ind = (int)floor(POS.z)*NX*NY + (int)floor(POS.y)*NX + (int)floor(POS.x) ;
            if (DENS[*ind]>0.0f) {
# if 1
               pos->x = clamp(POS.x, floor(POS.x)+EPS, floor(POS.x)+OMEPS) ; // make sure it is clearly inside the cell
               pos->y = clamp(POS.y, floor(POS.y)+EPS, floor(POS.y)+OMEPS) ;
               pos->z = clamp(POS.z, floor(POS.z)+EPS, floor(POS.z)+OMEPS) ;
# else
               pos->x = POS.x ;    pos->y = POS.y ;    pos->z = POS.z ;
# endif
               return ;  // found the cell on level 0               
            }
            break ;  // position is inside the current root level cell
         } else {
            // this was a step from a sub-octet to a parent octet, parent cell == ind
            sid    = (*ind) % 8 ; // parent cell = current cell in current octet
            POS   *=  HALF ;
            POS.x += sid % 2 ;  POS.y += (sid/2)%2  ;  POS.z += sid/4 ;  // double position in POS
            // is the position inside this octet?
            if ((POS.x>ZERO)&&(POS.x<TWO)&&(POS.y>ZERO)&&(POS.y<TWO)&&(POS.z>ZERO)&&(POS.z<TWO)) {
               // we must update the index to that of the cell that has the ray !!
               // we continue with double POS....
               *ind += -sid + 4*(int)(floor(POS.z)) + 2*(int)(floor(POS.y)) + (int)(floor(POS.x)) ;
               break ;
            }
         }
      } // while not root grid
   } // else -- going up
   
   
   // Go down - position *is* inside the current octet
   while((*level<max_otl)&&(DENS[OFF[*level]+(*ind)]<=0.0f)) {  // loop until (level,ind) points to a leaf
      // convert to sub-octet coordinates -- same for transition from root and from parent octet
      POS.x =  TWO*fmod(POS.x, ONE) ;
      POS.y =  TWO*fmod(POS.y, ONE) ;
      POS.z =  TWO*fmod(POS.z, ONE) ;
      float link = -DENS[OFF[*level]+(*ind)] ;
      *ind       = *(int *)&link ;      // first cell in the sub-octet
      (*level)++ ;
      *ind    += 4*(int)floor(POS.z)+2*(int)floor(POS.y)+(int)floor(POS.x) ; // subcell (suboctet)
   }        
   
# if 0
   // We come here only if the ray is at level>0 => force position to be inside the cell
   (*pos).x = clamp(POS.x, floor(POS.x)+EPS, floor(POS.x)+TMEPS) ;
   (*pos).y = clamp(POS.y, floor(POS.y)+EPS, floor(POS.y)+TMEPS) ;
   (*pos).z = clamp(POS.z, floor(POS.z)+EPS, floor(POS.z)+TMEPS) ;
# else
   pos->x = POS.x ;  pos->y = POS.y ;  pos->z = POS.z ;
# endif
   
   return ;
}







// (DOUBLE_POS>0)
float GetStepOT(REAL3 *POS, const float3 *DIR, int *level, int *ind,
                __global float *DENS, __constant int *OFF, __global int*PAR, const int max_otl,
                const REAL3 *POS_LE, const int LEADING) {
   // Calculate step to next cell, update level and ind for the next cell.
   // Returns the step length in GL units (units of the root grid). POS is updated.
   // max_otl => do not go to level > max_otl, even if the cell is not yet a leaf
   double dx, dy, dz ;
   dx = ((DIR->x)>0.0f)  ?  ((ONE+EPS-fmod(POS->x,ONE))/(DIR->x))  :  ((-EPS-fmod(POS->x,ONE))/(DIR->x)) ;
   dy = ((DIR->y)>0.0f)  ?  ((ONE+EPS-fmod(POS->y,ONE))/(DIR->y))  :  ((-EPS-fmod(POS->y,ONE))/(DIR->y)) ;
   dz = ((DIR->z)>0.0f)  ?  ((ONE+EPS-fmod(POS->z,ONE))/(DIR->z))  :  ((-EPS-fmod(POS->z,ONE))/(DIR->z)) ;
   dx =  clamp(min(dx, min(dy, dz)), EPS, 3*ONE) ;
   
   int follow=0 ;
   
   POS->x +=  dx*DIR->x ;   POS->y +=  dx*DIR->y ;   POS->z +=  dx*DIR->z ;   
   dx =  ldexp(dx, -(*level)) ;   // step returned in units [GL] = root grid units

# if 0
   // root grid only
   *level = 0 ; 
   if ((POS->x<=ZERO)||(POS->x>=NX)||(POS->y<=ZERO)||(POS->y>=NY)||(POS->z<=ZERO)||(POS->z>=NZ)) {
      *ind = -1 ; 
   } else {
      *ind = ((int)floor(POS->z))*NX*NY + ((int)floor(POS->y))*NX + (int)floor(POS->x) ;
   }
#else
   IndexOT(POS, level, ind, DENS, OFF, PAR, max_otl, POS_LE, DIR, LEADING) ;       // update (level, ind)
#endif
   
   
   return (float)dx ;                 // step length [GL]
}




#endif // DOUBLE_POS>0





















__kernel void Parents(__global   float  *DENS,
                      __constant   int  *LCELLS,  //  8 - number of cells on each level
                      __constant   int  *OFF,
                      __global     int  *PAR
                     ) {
   // Go through the density data in DENS and put to PAR information about the cell parents
   // 2019-02-19 -- PAR no longer allocated to include dummy entries for the root grid cells
   int id   = get_global_id(0) ;
   int GLOB = get_global_size(0) ;
   int ind ;
   float link ;
   for(int level=0; level<(OTLEVELS-1); level++) {     // loop over parent level = level
      for(int ipar=id; ipar<LCELLS[level]; ipar+=GLOB) {  // all cells on parent level
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
}





// o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=oo=o=o=o=o=o
   



// Solver routines
            
int Doolittle_LU_Decomposition_with_Pivoting(__global real *A, __local ushort *pivot, int n) {
   // pivot = n elements per work item, can be >> 32kB per work group
   // Difficult to defactor and make efficient for a work group =>
   // entirely separate problems for each work item !!!
   int i, j, k ;
   __global real *p_k, *p_row, *p_col ;
   real maxi;
   // For each row and column, k = 0, ..., n-1,
   for (k=0, p_k=A; k<n; p_k+=n, k++) {
      // find the pivot row
      pivot[k] = k;
      maxi     = fabs( p_k[k] );
      for (j=k+1, p_row = p_k+n; j<n; j++, p_row += n) {
         if (maxi < fabs(p_row[k])) {
            maxi      =  fabs(p_row[k]) ;
            pivot[k]  =  j;
            p_col     =  p_row;
         }          
      }             
      // and if the pivot row differs from the current row, then interchange the two rows.
      if (pivot[k] != k) {
         for (j = 0; j < n; j++) {
            maxi       = *(p_k+j);
            *(p_k+j)   = *(p_col+j);
            *(p_col+j) = maxi ;
         }
      }
      // and if the matrix is singular, return error
      if (p_k[k]==0.0f) return -1 ;
      // otherwise find the lower triangular matrix elements for column k
      for (i=k+1, p_row=p_k+n; i<n; p_row+=n, i++) {
         p_row[k]  /=  p_k[k] ;
      }
      // update remaining matrix
      for (i=k+1, p_row=p_k+n; i<n; p_row+=n, i++) {
         for (j=k+1; j<n; j++) {
            p_row[j] -= p_row[k] * p_k[j] ;
         }
      }
   }
   return 0 ;
}     



int Doolittle_LU_with_Pivoting_Solve(__global real *A,       __global real *B,
                                     __local  ushort *pivot, __global real *x,  int n) {
   int i, k;
   __global real *p_k;
   real dum;
   // Solve Lx = B for x, where L is a lower triangular matrix with an implied 1 along the diagonal.
   for (k=0, p_k = A; k<n; p_k+=n, k++) {
      if (pivot[k]!=k) {
         dum = B[k];    B[k] = B[pivot[k]];    B[pivot[k]] = dum;
      }
      x[k] = B[k];
      for (i = 0; i < k; i++) x[k] -= x[i] * p_k[i] ;
   }
   // Solve the linear equation Ux = y, where 
   // y is the solution obtained above of Lx = B and U is an upper triangular matrix.
   for(k=n-1, p_k=A+n*(n-1); k>=0; k--, p_k-=n) {
      if (pivot[k]!=k) {
         dum = B[k];    B[k] = B[pivot[k]];    B[pivot[k]] = dum;
      }
      for (i=k+1; i<n; i++) x[k] -= x[i] * p_k[i] ;
      if (p_k[k]==0.0f) return -1;
      x[k]  /=  p_k[k] ;
   }
   return 0;
}



float get_C(const float tkin, const int NTKIN, __global float *TKIN, __global float *C) {
   // Interpolate C() for the correct temperature
   //  { TKIN[NTKIN],  C[NTKIN] } (tkin)
   int i ;
   for(i=1; i<NTKIN-1; i++) {    if (TKIN[i]>tkin) break ;   }
   // linear interpolation between elements i-1 and i
   float w = (tkin-TKIN[i-1]) / (TKIN[i]-TKIN[i-1]) ;  // distance from i-1 == weight for i
   return   w*C[i] + (1.0f-w)*C[i-1] ;
}



void __kernel SolveCL(
#if (WITH_OCTREE>0)
                      const int         OTL,           //  0 hierarchy level (root level is 0)
#endif
                      const int         BATCH,         //  1 number of cells per kernel call
                      __global float   *A,             //  2 MOL_A[TRANSITIONS]
                      __global int     *UL,            //  3 MOL_UL[TRANSITIONS,2]
                      __global float   *E,             //  4 MOL_E[LEVELS]
                      __global float   *G,             //  5 MOL_G[LEVELS]
                      const int         PARTNERS,      //  6 number of collisional partners
                      const int         NTKIN,         //  7 number of Tkin for collisions -- same for all partners !!!???
                      const int         NCUL,          //  8 number of rows in C arrays
                      __global float   *MOL_TKIN,      //  9 MOL_TKIN[PARTNERS, NTKIN]
                      __global int     *CUL,           // 10 CUL[PARTNERS, NCUL, 2]
                      __global float   *C,             // 11 C[PARTNERS, NCUL, NTKIN]
                      __global float   *CABU,          // 12 CAB[PARTNERS]  --- no spatial variation yet
                      __global float   *RHO,           // 13 RHO[BATCH]
                      __global float   *TKIN,          // 14 TKIN[BATCH]
                      __global float   *ABU,           // 15 ABU[BATCH]
                      __global float   *NI,            // 16 NI[BATCH,  LEVELS]   ---- PL_buf on host !!! READ-WRITE !!
                      __global float   *SIJ,           // 17 SIJ[BATCH,TRANSITIONS]
                      __global float   *ESC,           // 18 ESC[BATCH,TRANSITIONS]
                      __global float   *RES,           // 19 RES[BATCH, LEVELS]
                      __global float   *WRK,           // 20 WRK[BATCH*LEVELS*(LEVELS+1)]
                      const int follow_i
                     ) {
   // Solve equilibriumm equations for BATCH cells, based on upward transitions counted in SIJ and
   // number of emitted photons escaping a cell (ESC). New level populations are returned in RES.
   // 2020-06-02 --- SIJ /= VOLUME moved from simulation to here, in the solver
   const int id  = get_global_id(0) ;
   const int lid = get_local_id(0) ;
   if ((id>=BATCH)||(RHO[id]<=0.0f))  return ;
   __global float  *MATRIX = &WRK[id*LEVELS*(LEVELS+1)] ;
   __global float  *VECTOR = &WRK[id*LEVELS*(LEVELS+1)+LEVELS*LEVELS] ;
   __global float  *B = &RES[id*LEVELS] ;   // output vector
   __global float  *V =  &NI[id*LEVELS] ;   // temporary storage for solved NI vector
   __local  ushort  P[LOCAL*LEVELS] ;       // 64 workers x 100 levels x 2B  = 12.5 kB
   __local  ushort *pivot = &P[lid*LEVELS] ;
   
#if 0
   if (RHO[id]<=CLIP)  {  // skip calculation for cells RHO<CLIP
      for(int i=0; i<LEVELS; i++)  B[i] = V[i] ;
      return ;  // INI['clip']
   }
#endif
   
#if 0
   if (id==follow_i) {
      printf("SIJ ") ;
      for(int t=0; t<TRANSITIONS; t++) printf(" %10.3e", SIJ[id*TRANSITIONS+t]) ;
      printf("\n") ;
# if (WITH_ALI>0)
      printf("ESC ") ;
      for(int t=0; t<TRANSITIONS; t++) printf(" %10.3e", ESC[id*TRANSITIONS+t]) ;
      printf("\n") ;
# endif
      printf("NI< ") ;
      for(int t=0; t<LEVELS; t++) printf(" %10.3e", V[t]) ;
      printf("\n") ;
   }
#endif
   
   float tmp ;
#if (WITH_OCTREE>0)
   float volume = VOLUME / pow(8.0f, (float)OTL) ;
#else   
   float volume = VOLUME ;
#endif
   int  u, l ;
   MATRIX[IDX(0,0)] = 0.0f ;   
   // Note -- we use CUL only for the first collisional partner, others must have (i,j) rows in same order!
   for(int i=1; i<LEVELS; i++) {  // loop explicitly only over downward transitions  M[j,i] ~   j <- i,   i>j
      MATRIX[IDX(i,i)] = 0.0f ; 
      for(int j=0; j<i; j++) {    // j<i == downward transitions
         for(u=0; u<NCUL; u++) {  // find the row  u  in collisional coefficient table  (i,j)
            if ((CUL[2*u]==i)&(CUL[2*u+1]==j)) break ;  // <-- CUL OF THE FIRST PARTNER ONLY !!!
         }
         tmp = 0.0f ;
         for(int p=0; p<PARTNERS; p++) { // get_C has the correct row from C, NTKIN element vector DOWNWARDS !!
            tmp += CABU[p]*get_C(TKIN[id], NTKIN, &MOL_TKIN[p*NTKIN], &C[p*NCUL*NTKIN + u*NTKIN]) ;
         }
         MATRIX[IDX(j,i)] = tmp*RHO[id] ;    //  IDX(j,i) = transition j <-- i  == downwards
         // the corresponding UPWARD transition  j -> i
         tmp  *=  (G[i]/G[j]) *  exp(-H_K*(E[i]-E[j])/TKIN[id]) ;
         MATRIX[IDX(i,j)] = tmp*RHO[id] ;    //  IDX(j,i) = transition j <-- i
      }
   }
   
   
#if (WITH_ALI>0)
   for(int t=0; t<TRANSITIONS; t++) {  // modified Einstein A
      u = UL[2*t] ;  l = UL[2*t+1] ;
# if 0
      MATRIX[IDX(l,u)]  +=  ESC[id*TRANSITIONS+t] / (volume*NI[id*LEVELS+u]) ;
# else
      tmp                =  NI[id*LEVELS+u]  ;
      // if nu==0.0, we should have  beta=1  ==>  element == A[t] ??
      // if (tmp>1.0e-30f)  MATRIX[IDX(l,u)]  +=  ESC[id*TRANSITIONS+t]/(volume*tmp) ;
      // else               MATRIX[IDX(l,u)]  +=  A[t]  ;
      MATRIX[IDX(l,u)] += (tmp>1.0e-20f) ? (ESC[id*TRANSITIONS+t]/(volume*tmp)) : (A[t]) ;
# endif
   }
#else // not ALI
   for(int t=0; t<TRANSITIONS; t++)  MATRIX[IDX(UL[2*t+1], UL[2*t])]  +=  A[t] ;
#endif
   
   
   // --- NI is no longer used so we can reuse it to store vector X below ---
   for(int t=0; t<TRANSITIONS; t++) {
      u = UL[2*t] ;   l = UL[2*t+1] ;
      MATRIX[IDX(u,l)]  +=  SIJ[id*TRANSITIONS+t]               / volume ;
      MATRIX[IDX(l,u)]  +=  SIJ[id*TRANSITIONS+t] * (G[l]/G[u]) / volume ;   //  /= GG  == G[l]/G[u]
   }   
   for(int i=0; i<LEVELS-1; i++) {  // loop over columns
      tmp = 0.0f ;
      for(int j=0; j<LEVELS; j++)  tmp +=  MATRIX[IDX(j,i)] ;  // sum over column
      MATRIX[IDX(i,i)] = -tmp ;
   }
   for(int i=0; i<LEVELS; i++)  MATRIX[IDX(LEVELS-1, i)]  = -MATRIX[IDX(0,0)] ;
   for(int i=0; i<LEVELS; i++)  VECTOR[i] = 0.0f ;
   
   // one cannot drop ABU from here or the solution fails for test_3d_ot.py
   // ... but it will also fail if RHO*ABU scaling is kept and ABU is extremely small!!
   // In test_3d_ot.py, eq.eq. is ok for scaling - replacing ABU-  with  1e-3 ... 1e-22
   // VECTOR[LEVELS-1]  =  -MATRIX[0]*RHO[id]*ABU[id] ;         
   // ... or replacing RHO*ABU with any constant  0.1 ... 1e-21 is ok for test_3d_ot.py
   // VECTOR[LEVELS-1]  =  -MATRIX[0]*1e-11f  ;
   //  1.0e-11f used for tests test_3d_ot.py
   VECTOR[LEVELS-1]  =  -MATRIX[0]*1.0e-10f  ;
   // VECTOR[LEVELS-1]  =  -MATRIX[0]*1.0e-5f  ;
   
   
#if 0 
   // renormalise the whole equation set --- this does nothing for the condition number, which
   // is in any case small (~197 for the adhoc-problem problem cells?)
   tmp = -MATRIX[0] ;
   for(int i=0; i<LEVELS*LEVELS; i++) MATRIX[i] /= tmp ;
   VECTOR[LEVELS-1] = 1.0f ;
#endif
   
   
#if 0
   if (id==follow_i) {
      for(int j=0; j<LEVELS; j++) {      // row
         for(int i=0; i<LEVELS; i++) {   // column
            printf(" %10.3e", MATRIX[IDX(j,i)]) ;
         }
         printf("\n") ;
      }
      printf("\nright side  0,0, ..., %.4e, RHO=%.3e, ABU=%.3e\n", VECTOR[LEVELS-1], RHO[id], ABU[id]) ;
   }
#endif
   u  = Doolittle_LU_Decomposition_with_Pivoting(MATRIX, pivot,    LEVELS) ;
   u *= Doolittle_LU_with_Pivoting_Solve(MATRIX, VECTOR, pivot, V, LEVELS) ;   
   tmp = 0.0f ;   
   for(int i=0; i<LEVELS; i++)  {
      V[i]  =  clamp(V[i], 1.0e-30f, 1.0f) ; // was 1.0e-25f
      tmp  +=  V[i] ;
   }
   for(int i=0; i<LEVELS; i++)  B[i]  =  V[i]*RHO[id]*ABU[id] / tmp ;  // B ro, X rw
#if 0
   if (id==follow_i) {
      printf("NI> ") ;
      for(int t=0; t<LEVELS; t++) printf(" %10.3e", B[t]) ;
      printf("\n") ;
   }
#endif
}




__kernel void LTE(const int         BATCH,         //  1 number of cells per kernel call
                  __global float   *E,             //  MOL_E[LEVELS]
                  __global float   *G,             //  MOL_G[LEVELS]
                  __global float   *TKIN,          //  TKIN[BATCH]
                  __global float   *RHO,           //  RHO[BATCH]
                  __global float   *NI)            //  NI[BATCH,  LEVELS]   ---- PL_buf on host !!! READ-WRITE !!
{
   int id = get_global_id(0)   ;
   float sum, T, R ;
   for(int i=id; i<BATCH; i+=GLOBAL) {
      T   = clamp(TKIN[i], 3.0f, 9999.0f)  ; // also takes care of link cells => no NaNs to NI_ARRAY
#if 0
      T   =  15.0f  ;  // *** TEST ISOTHERMAL CASE ***
#endif
      R   = clamp(RHO[i], 1.0e-10f, 1.0e20f) ;
      sum = 0.0f ;
      for(int j=0; j<LEVELS; j++)   sum            +=   G[j]*exp(-H_K*E[j]/T) ;
      for(int j=0; j<LEVELS; j++)   NI[i*LEVELS+j]  =   G[j]*exp(-H_K*E[j]/T)   * (R/sum) ;
#if 0
      if (isfinite(NI[i*LEVELS+LEVELS-1])==0) {
         printf("LTE ---- NOT FINITE:  T=%.3e, n=%.3e\n", T, R) ;
      }
#endif
   }
}








// o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=o=oo=o=o=o=o=o
   





#if (WITH_OCTREE==5)

// Root grid POS=[0,1], ind used to specify the actual cell


void IndexG(REAL3 *pos, int *level, int *ind, __global float *DENS, __constant int *OFF) {
   // Return cell level and index for given global position
   //    pos    =  on input global coordinates, returns either
   //              global coordinates (root grid) or relative within current octet
   //    level  =  hierarchy level on which the cell resides, 0 = root grid
   //    ind    =  index to global array of densities for cells at this level
   //    DENS, OFF needed for the OT hierarchy
   // Note: this must return accurate coordinates even when position is outside the volume,
   //       because that is used to mirror and restart level-0 rays
   // PLMETHOD==1 version
   //    input pos is still global coordinate [0,NX], on return pos is [0,1], if on level=0 !!
   float link ;
   *ind = -1 ;
   *level = 0 ;
   if ((pos->x<=ZERO)||(pos->y<=ZERO)||(pos->z<=ZERO)) return  ;
   if ((pos->x>=NX)  ||(pos->y>=NY)  ||(pos->z>=NZ))   return  ;
   *ind   = (int)floor((*pos).z)*NX*NY + (int)floor((*pos).y)*NX + (int)floor((*pos).x) ;
   if (DENS[*ind]>0.0f) {
      pos->x = fmod(pos->x, ONE) ;   pos->y = fmod(pos->y, ONE) ;   pos->z = fmod(pos->z, ONE) ;
      return ;  // found a leaf
   }
   while(1) {
      // go down, coordinates in sub-octet
      pos->x = TWO*fmod(pos->x, ONE) ;  // coordinate inside root grid cell [0,1]
      pos->y = TWO*fmod(pos->y, ONE) ;
      pos->z = TWO*fmod(pos->z, ONE) ;
      // new indices
      link   =  -DENS[OFF[*level]+(*ind)] ;
      *ind   =  *(int *)(&link );  // index **WITHIN THE LEVEL*** (not global index)
      (*level)++ ;
      *ind  +=  4*(int)floor((*pos).z) + 2*(int)floor((*pos).y) + (int)floor((*pos).x) ; // cell in octet
      if (DENS[OFF[*level]+(*ind)]>0.0f) {
         return ; // found a leaf
      }
   }
}



void IndexGR(REAL3 *pos, int *level, int *ind, __global float *DENS, __constant int *OFF) {
   // Find indices for a given root-grid position [0,NX].
   // PLMETHOD==1 version: no difference because pos is not returned.
   *ind   = -1 ;
   *level = 0 ;
   if ((pos->x<=ZERO)||(pos->y<=ZERO)||(pos->z<=ZERO)) return  ;
   if ((pos->x>=NX)  ||(pos->y>=NY)  ||(pos->z>=NZ))   return  ;
   *ind   = (int)floor(pos->z)*NX*NY + (int)floor(pos->y)*NX + (int)floor(pos->x) ;
   return ;  // stay on the root grid
}




void RootPos(REAL3 *POS, const int ilevel, const int iind, __constant int *OFF, __global int *PAR) {
   // Given current indices (ilevel, iind) and current position POS (current level), return the
   // position in root grid coordinates [0,NX].
   // Even with PLMETHOD==1, results are global coordinates [0,NX] because this is used
   // only to identify when a given position matches rays at a given level (not raytracing).
   int level=ilevel, ind=iind, sid ;
   if (level==0) return ;  // no change, POS is already position in root grid coordinates
   // Otherwise go UP in the grid hierarchy
   while (level>0) {
      ind  =  PAR[OFF[level]+ind-NX*NY*NZ] ;  level-- ;  // parent cell index
      if (level==0) {              // arrived at root grid
         *POS     *=  HALF ;       // sub-octet coordinates [0,2] into global coordinates [0,1]
         (*POS).x +=  ind      % NX  ;  // returns coordinates [0,NX] even for PLMETHOD
         (*POS).y +=  (ind/NX) % NY ;
         (*POS).z +=  ind  / (NX*NY) ;
         return ;
      } else {  // step from a sub-octet to a parent octet, parent cell == ind
         sid       =  ind % 8 ;    // index of the sub-cell, child of current cell ind
         *POS     *=  HALF ;       // from sub-octet to parent cell, positions *= 0.5
         (*POS).x +=  sid % 2 ;   (*POS).y += (sid/2)%2  ;   (*POS).z += sid/4 ; // add offset based on subcell
      } 
   } // while not root grid
}





void IndexOT(float3 *pos, int *level, int *ind,
             __global float *DENS, __constant int *OFF, __global int *PAR, const int max_otl, 
             const REAL3 *POS_LE,  const float3 *DIR, const int LEADING) {
   // LEADING = the main direction, used for FIX_PATHS
   // NOTE: PLMETHOD==1 assumes that for level=0 POS is just the position in coordinates [0,1]
   //       or just outside when one has stepped to another cell, and ind is needed to tell
   //       the actual root-grid position
   // With PLMETHOD==1, this is used **ONLY** to find the level of the neighbour cell
   //    ... with coordinates that are already safely inside that neighbour cell
   int sid ;
   REAL3  POS ;  // we have EPS~1e-5 ... but we may loose ~3 digits when we step up to root level => need double
   POS.x = pos->x ;   POS.y = pos->y ;   POS.z = pos->z ;
   
   if (*level==0) {   // on root grid
      if ((pos->x<=0.0f)||(pos->x>=NX)||(pos->y<=0.0f)||(pos->y>=NY)||(pos->z<=0.0f)||(pos->z>=NZ)) {
         *ind = -1 ; return ;  // ok -- pos contains accurate root grid coordinates, even outside the volume
      }
      *ind = (int)floor((*pos).z)*NX*NY + (int)floor((*pos).y)*NX + (int)floor((*pos).x) ;
      if (DENS[*ind]>0.0f) return ;  // level 0 cell was a leaf -- we are done
   } else {     
      while ((*level)>0) {           // go UP until the position is inside the octet
         *ind = PAR[OFF[*level]+(*ind)-NX*NY*NZ] ;  (*level)-- ;  // parent cell index
         if ((*level)==0) {                // arrived at root grid
            POS   *=  HALF ;               // convertion from sub-octet to global coordinates
            POS.x +=  (*ind)      % NX  ;  // real root grid coordinates
            POS.y +=  ((*ind)/NX) % NY ;
            POS.z +=  (*ind) / (NX*NY) ;
            if ((POS.x<=0.0f)||(POS.x>=NX)||(POS.y<=0.0f)||(POS.y>=NY)||(POS.z<=0.0f)||(POS.z>=NZ)) {
               *ind = -1 ;   // we came up to root grid cell ind, but POS was outside model volume
               pos->x = POS.x ;  pos->y = POS.y ;  pos->z = POS.z ;
               return ;  // we left the model volume -- pos still containing proper coordinates = ok !!!
            }
            // the position is not necessarily in cell 'ind', could be a neighbour!
            *ind = (int)floor(POS.z)*NX*NY + (int)floor(POS.y)*NX + (int)floor(POS.x) ;
            if (DENS[*ind]>0.0f) {
               pos->x = POS.x ;  pos->y = POS.y ;  pos->z = POS.z ;
               return ;  // found the cell on level 0               
            }
            break ;
         } else {
            // this was a step from a sub-octet to a parent octet, parent cell == ind
            sid    = (*ind) % 8 ; // parent cell = current cell in current octet
            POS   *=  HALF ;
            POS.x += sid % 2 ;  POS.y += (sid/2)%2  ;  POS.z += sid/4 ;
            // is the position inside this octet?
            if ((POS.x>0.0f)&&(POS.x<2.0f)&&(POS.y>0.0f)&&(POS.y<2.0f)&&(POS.z>0.0f)&&(POS.z<2.0f)) {
               *ind += -sid + 4*(int)(floor(POS.z)) + 2*(int)(floor(POS.y)) + (int)(floor(POS.x)) ;
               break ;
            }
         }
      } // while not root grid
   } // else -- going up
   
   // Go down - position *is* inside the current octet
   while((*level<max_otl)&&(DENS[OFF[*level]+(*ind)]<=0.0f)) {  // loop until (level,ind) points to a leaf
#  if (1)
      POS.x =  clamp(TWO*fmod(POS.x, ONE), EPS, TMEPS) ;
      POS.y =  clamp(TWO*fmod(POS.y, ONE), EPS, TMEPS) ;
      POS.z =  clamp(TWO*fmod(POS.z, ONE), EPS, TMEPS) ;
#  else
      POS.x =  TWO*fmod(POS.x, ONE) ;
      POS.y =  TWO*fmod(POS.y, ONE) ;
      POS.z =  TWO*fmod(POS.z, ONE) ;
#  endif
      float link = -DENS[OFF[*level]+(*ind)] ;
      *ind       = *(int *)&link ;      // first cell in the sub-octet
      (*level)++ ;
      *ind    += 4*(int)floor(POS.z)+2*(int)floor(POS.y)+(int)floor(POS.x) ; // subcell (suboctet)
   }        
   (*pos).x = POS.x ;  (*pos).y = POS.y ;  (*pos).z = POS.z ; // this not even needed !!
   return ;
}





float GetStepOT(float3 *POS, const float3 *DIR, int *level, int *ind,
                __global float *DENS, __constant int *OFF, __global int*PAR, const int max_otl,
                const REAL3 *POS_LE, const int LEADING) {
   // PLMETHOD==1
   // Calculate step to next cell, update level and ind for the next cell.
   // Returns the step length in GL units (units of the root grid). POS is updated.
   // Note --- even when one steps outside the model volume, POS should be accurate
   //          because that may be used to start a new ray on the opposite side !!!
   // max_otl => do not go to level > max_otl, even if the cell is not yet a leaf
   // PLMETHOD==1  ==> on root grid POS=[0,1] and ind is needed to point the cell
   // We must be at some border, one coordinate within ~EPS of the boundary, others EPS2 from border !!!
   float3 SPOS, DPOS ;
   float dx, dy, dz, k ;
   int I, J, K, ibo, up=0, level0, sid ;
   // overstep by EPS over the border -- using local coordinates
   dx = ((DIR->x)>0.0f)  ?  ((1.0f+EPS-fmod(POS->x,1.0f))/(double)(DIR->x))  :  ((-EPS-fmod(POS->x,1.0f))/(double)(DIR->x)) ;
   dy = ((DIR->y)>0.0f)  ?  ((1.0f+EPS-fmod(POS->y,1.0f))/(double)(DIR->y))  :  ((-EPS-fmod(POS->y,1.0f))/(double)(DIR->y)) ;
   dz = ((DIR->z)>0.0f)  ?  ((1.0f+EPS-fmod(POS->z,1.0f))/(double)(DIR->z))  :  ((-EPS-fmod(POS->z,1.0f))/(double)(DIR->z)) ;
   // choose the shortest step and set ibo to tell which border one is supposed to have stepped over
   if (dx<dy) {
      if (dx<dz) {
         ibo = (DIR->x>0.0f) ? 0 : 1 ;  // ibo indicates the direction, 0=+X
      } else  {
         ibo = (DIR->z>0.0f) ? 4 : 5 ;
         dx  =  dz ;
      }
   } else {
      if (dy<dz) {
         ibo = (DIR->y>0.0f) ? 2 : 3 ;
         dx  = dy ;
      } else  {
         ibo = (DIR->z>0.0f) ? 4 : 5 ;
         dx  = dz ;
      }
   }
   POS->x +=  dx*DIR->x ;   POS->y +=  dx*DIR->y ;   POS->z +=  dx*DIR->z ; // POS outside current cell
   dx =  ldexp(dx, -(*level)) ;   // dx in root-grid units

   
   // Handle immediately steps between root grid cells and between cells within the same octet
   if ((*level)==0) {
      I = (*ind)%NX ;   J = ((*ind)/NX)%NY ;  K = (*ind)/(NX*NY) ;  // root grid indices for the original cell
      if (ibo<2) { 
         if (ibo==0)    { POS->x = EPS   ;  I += 1 ; } // POS relative to the new cell, (I,J,K) its root indices
         else           { POS->x = OMEPS ;  I -= 1 ; }
      } else {
         if (ibo<4) {
            if (ibo==2) { POS->y = EPS ;    J += 1 ; }
            else        { POS->y = OMEPS ;  J -= 1 ; }
         } else {
            if (ibo==4) { POS->z = EPS ;    K += 1 ; }
            else        { POS->z = OMEPS ;  K -= 1 ; }
         }
      }  
      if ((I<0)||(I>=NX)||(J<0)||(J>=NY)||(K<0)||(K>=NZ)) {
         // we left the cloud, set *ind to (-*ind-1), to tell where we exited
         *ind = -*ind-1 ;   return dx ;  // ind is still the last root cell visited, POS outside the volume
      }      
      *ind =  I+NX*(J+NY*K) ;            // inside the model; update the root-grid index
      if (DENS[*ind]>0.0f) return dx ;   // if target cell was a leaf, we are done
   } else {
      // We started in some octet, check if it was simply an internal steps within that octet
      if ((POS->x>0.0f)&&(POS->x<2.0f)&&(POS->y>0.0f)&&(POS->y<2.0f)&&(POS->z>0.0f)&&(POS->z<2.0f)) {
         *ind  += -(*ind%8) + 4*(int)floor(POS->z) + 2*(int)floor(POS->y) + (int)floor(POS->x) ;
         if (DENS[*ind]>0.0f) return dx ;  // found the cell (a leaf) directly within the same octet
         // otherwise, up==0,  POS and *ind point to a parent cell, some levels above the final target cell
      }
      up = 1 ;  // POS was outside the original octet, need to go up to a common parent cell first
      // otherwise we need to go up to find a common parent... or down to find a leaf
   }
   
   
   // We will be travelling up/down in the hierarchy; set up SPOS and DPOS for safe transformations
   if ((*level)==0) {
      // Already at root level, we have stepped to the new cell that is a parent of the target cell
      // A step between root-grid cells => POS cannot be at the internal border of any octet.
      SPOS.x  =  clamp(POS->x, 0.001f, 0.999f) ;  // some position away from the border, close to POS
      SPOS.y  =  clamp(POS->y, 0.001f, 0.999f) ;   
      SPOS.z  =  clamp(POS->z, 0.001f, 0.999f) ;
   } else {
      // We are in original octet after an internal step to a non-leaf cell  (up==0)
      // or POS is still outside the octet and we must find a common parent octet (up==1; original -> target)
      if (up==0) {
         // We are already in the parent octet, *ind and POS are correct, just need to go down to find a leaf.
         // POS is [0,2] but can also be near an internal border of the octet.
         // SPOS should be something away from any integer values, close to the actual border.
         SPOS.x  =  floor(POS->x) + clamp(POS->x-floor(POS->x), 0.001f, 1.999f) ;
         SPOS.y  =  floor(POS->y) + clamp(POS->y-floor(POS->y), 0.001f, 1.999f) ;
         SPOS.z  =  floor(POS->z) + clamp(POS->z-floor(POS->z), 0.001f, 1.999f) ;
         DPOS    =  (*POS)-SPOS ;
      } else {
         // up==1, POS is outside the octet but near the outer border of current octet, the border indicated by ibo
         SPOS.x = clamp(POS->x, 0.001f, 1.999f) ;  // two out of three coordinates are inside...
         SPOS.y = clamp(POS->y, 0.001f, 1.999f) ;
         SPOS.z = clamp(POS->z, 0.001f, 1.999f) ;
         if (ibo<2) {
            SPOS.x    =  (ibo==0) ? 2.001f : -0.001f ;
         } else {
            if (ibo<4) {
               SPOS.y =  (ibo==2) ? 2.001f : -0.001f ;
            } else {
               SPOS.z =  (ibo==4) ? 2.001f : -0.001f ;
            }
         }
      }      
   }
   DPOS   =  (*POS)-SPOS ; 
   level0 = *level ;
   // Now SPOS is a position safely inside the target cell, DPOS is the distance to the actual position
   
   
   if (up==1) {  // first go up to the common parent, before going down to find the leaf
      while ((*level)>0) {           // go up until the position is inside the octet
         *ind  = PAR[OFF[*level]+(*ind)-NX*NY*NZ] ;  (*level)-- ;  // parent cell index
         SPOS *=  HALF ;             // this should now be safe, SPOS is now [0,1] or just outside
         if ((*level)==0) {          // arrived at root grid
            I = (*ind)%NX ;   J = ((*ind)/NX)%NY ;  K = (*ind)/(NX*NY) ;  // root grid indices
            if (ibo<2) {             // root grid => do the step to the correct neighbour
               if (ibo==0)    { SPOS.x = EPS   ;  I += 1 ; }   // root grid, POS ~ [0,1]
               else           { SPOS.x = OMEPS ;  I -= 1 ; }
            } else {
               if (ibo<4) {
                  if (ibo==2) { SPOS.y = EPS   ;  J += 1 ; }
                  else        { SPOS.y = OMEPS ;  J -= 1 ; }
               } else {
                  if (ibo==4) { SPOS.z = EPS   ;  K += 1 ; }
                  else        { SPOS.z = OMEPS ;  K -= 1 ; }
               }
            }
            if ((I<0)||(I>=NX)||(J<0)||(J>=NY)||(K<0)||(K>=NZ)) { // we left the model volume
               *ind  = -1-*ind ;             // information about the last visited cell, for mirroring the ray
               k     = pown(0.5f, level0) ;  // scale difference between level0 and this level=0
               SPOS += k*DPOS ;              // final accurate position; just outside the model
               // safeguards, ensure that only one coordinate is close to border... necessary?
               if (ibo<2) {
                  POS->x     =  (ibo==0)    ?  OPEPS :  -EPS  ;   // this one just outside cell *ind
                  POS->y     =  clamp(SPOS.y,    EPS,  OMEPS) ;   // root grid => POS ~ [0,1]
                  POS->z     =  clamp(SPOS.z,    EPS,  OMEPS) ;
               } else {
                  if (ibo<4) {
                     POS->y  =  (ibo==2)    ?  OPEPS :  -EPS  ;   // this one just outside cell *ind
                     POS->x  =  clamp(SPOS.x,    EPS,  OMEPS) ;  
                     POS->z  =  clamp(SPOS.z,    EPS,  OMEPS) ;
                  } else {
                     POS->z  =  (ibo==4)   ?   OPEPS :  -EPS  ;   // this one just outside cell *ind
                     POS->x  =  clamp(SPOS.x,    EPS,  OMEPS) ;  
                     POS->z  =  clamp(SPOS.z,    EPS,  OMEPS) ;
                  }
               }
               return dx ;
            }
            // stepped to a new root-grid cell, SPOS and (I,J,K) already for the target root cell
            *ind = I+NX*(J+NY*K) ;            // the new root-grid cell
            if (DENS[*ind]>0.0f) {            // if that is a leaf, take care of that already here
               k     = pown(0.5f, level0) ;   // DPOS was in smaller units
               SPOS += SPOS + k*DPOS ;        // final position, inside this level=0, *ind cell
               // safeguards -- position inside cell *ind, on the root grid
               if (ibo<2) {
                  POS->x     =  (ibo==0) ?      EPS : OMEPS ;   // inside cell *ind, on the level 0 = [0,1]
                  POS->y     =  clamp(SPOS.y,   EPS,  OMEPS) ;  
                  POS->z     =  clamp(SPOS.z,   EPS,  OMEPS) ;
               } else {
                  if (ibo<4) {
                     POS->y  =  (ibo==2) ?      EPS : OMEPS ;   
                     POS->x  =  clamp(SPOS.x,   EPS,  OMEPS) ;  
                     POS->z  =  clamp(SPOS.z,   EPS,  OMEPS) ;
                  } else {
                     POS->z  =  (ibo==4) ?      EPS : OMEPS ;   
                     POS->x  =  clamp(SPOS.x,   EPS,  OMEPS) ;  
                     POS->z  =  clamp(SPOS.z,   EPS,  OMEPS) ;
                  }
               }               
               return dx ;  // found the cell on level 0
            }
            // found a new root-grid cell (*ind, SPOS for that cell), but the cell is still a parent
            break ;   
         } else {  // else *level > 0
            // this was a step to a parent *octet*
            // *ind is the parent cell and SPOS is inside or still just outside its octet
            sid     = (*ind) % 8 ;    // *ind of the cell, SPOS is not inside this one
            SPOS.x += sid % 2 ;   SPOS.y += (sid/2)%2  ;   SPOS.z += sid/4 ; // actual coordinates in the octet
            // is SPOS now inside the current octet?
            if ((SPOS.x>0.0f)&&(SPOS.x<2.0f)&&(SPOS.y>0.0f)&&(SPOS.y<2.0f)&&(SPOS.z>0.0f)&&(SPOS.z<2.0f)) {
               // yes, we have found the parent octet, update *ind to point to the correct sibling
               *ind += -sid + 4*(int)(floor(SPOS.z)) + 2*(int)(floor(SPOS.y)) + (int)(floor(SPOS.x)) ;
               break ;  // break:  *ind is the correct cell but may still be a parent
            }
         }
      } // while not root grid
   } // if (up==1)
   
   
   // At this point we are in a root-grid cell or some octet that contains the target position.
   // If level==0, the cell is guaranteed to have children, 
   // if level>0, it may already be the final leaf cell (*ind is pointing to that, SPOS is ok for that octet)
   // In both cases, we may have to go down towards the leaf.
   
   
   // Go down - position *is* inside the current root-grid cell or the current octet
   while((*level<max_otl)&&(DENS[OFF[*level]+(*ind)]<=0.0f)) {  // loop until (level, ind) points to a leaf
      // convert to sub-octet coordinates -- same for transition from root and from parent octet
      SPOS.x =  TWO*fmod(SPOS.x, ONE) ;  // coordinates in the sub-octet
      SPOS.y =  TWO*fmod(SPOS.y, ONE) ;  // multiplication by 2.0 is safe
      SPOS.z =  TWO*fmod(SPOS.z, ONE) ;      
      float link = -DENS[OFF[*level]+(*ind)] ;
      *ind       = *(int *)&link ;       // *first* cell in the sub-octet
      (*level)++ ;
      *ind      += 4*(int)floor(SPOS.z) + 2*(int)floor(SPOS.y) + (int)floor(SPOS.x) ; // subcell index
   }           
   
   // The final cell is *ind and SPOS is inside it.
   // Update *POS with the final coordinates.
   k      =  pown(0.5f, level0-(*level)) ;   
   SPOS  +=  k*DPOS ;   // rounding errors could make this equal to border coordinates!
   // Safeguards. The cell is at some level>0, all level=0 cases were taken care of above.
   // Direct steps between octet members were also handled above. This could be outer boundary of the
   // original octet, or it could be internal step to a cell with children -- but in that refined region
   // the position would again be at the outer boundary of that higher-level octet.
   // Therefore, we know that POS should be in [0,2] and in the ibo direction it is either ~0 or ~2. 
   // For the other coordinates one could end up by chance on an internal border of the octet but
   // we do not check for that (not a problem?).
   if (ibo<2) {
      POS->x  =  (ibo==0) ?        EPS : TMEPS  ;   // this one just inside cell *ind
      POS->y  =  clamp(SPOS.y,     EPS,  TMEPS) ;  
      POS->z  =  clamp(SPOS.z,     EPS,  TMEPS) ;
   } else {
      if (ibo<4) {
         POS->y  =  (ibo==2) ?     EPS : TMEPS  ;
         POS->x  =  clamp(SPOS.x,  EPS,  TMEPS) ;  
         POS->z  =  clamp(SPOS.z,  EPS,  TMEPS) ;
      } else {
         POS->z  =  (ibo==4) ?     EPS : TMEPS  ;
         POS->x  =  clamp(SPOS.x,  EPS,  TMEPS) ;  
         POS->z  =  clamp(SPOS.z,  EPS,  TMEPS) ;
      }
   }
   return dx ;                 // step length [GL]
}



#endif  // WITH_OCTREE==5


