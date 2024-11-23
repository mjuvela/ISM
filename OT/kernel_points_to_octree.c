

void __kernel SetRootCoordinates(__global float3 *CC,
                                 __global int    *LCELLS) 
{
   // Set CC coordinates for the root grid
   const int id = get_global_id(0)  ;
   const int gs = get_global_size(0) ;
   int i, j, k ;
   for(int ind=id; ind<LCELLS[0]; ind+=gs) {
      i  =   ind % NX ;     
      j  =  (ind/NX) % NY ;  
      k  =   ind / (NX*NY) ;
      CC[ind].x = (i+0.5f) ;      // coordinates in normal SOC coordinates  [0,NX] etc. 
      CC[ind].y = (j+0.5f) ;
      CC[ind].z = (k+0.5f) ;      // centre of the cell
   }      
}



void __kernel Split(const    int      L,          // parent level, L=0, 1, 2, ..., LEVELS-
                    __global float   *PX,         // point coordinates
                    __global float   *PY,         // point coordinates
                    __global float   *PZ,         // point coordinates
                    __global float   *HH,         // hierarchy as a single vector
                    __global float3  *CC,         // CC[].x/y/z centre coordinates of each cell
                    __global int     *OFF,        // OFF[LEVELS]
                    __global int     *LCELLS)     // LCELLS[LEVELS]
{  
   // Loop over the cells on level L, set   H=1+index_to_yt_vectors   if the cell will not be split, 
   // set H=-1.0 if the cell is to be split -- the links are not yet created
   const int id = get_global_id(0)  ;
   const int gs = get_global_size(0) ;
   int i, j, k, ii ;
   float x, y, z, d, dx, dy, dz ;
   __global float *H = &(HH[OFF[L]]) ;  // vector of parent cells
   d  =  0.495f*pow(0.5f, L) ;          // half of the level L cell size -- must be <0.5 not to match parent cell
   for(i=id; i<LCELLS[L]; i+=gs) {      // loop over level L cells
     x    = CC[OFF[L]+i].x ;            // PARENT cell centre in root grid coordinates
     y    = CC[OFF[L]+i].y ;  
     z    = CC[OFF[L]+i].z ; 
     H[i] = 1.0f ;                      // H>0  == not split
     for(int p=0; p<NP; p++) {          // loop over the provided point coordinates
       dx  =  fabs(PX[p]-x) ;   
       dy  =  fabs(PY[p]-y) ;   
       dz  =  fabs(PZ[p]-z) ;
       if ((dx<d)&&(dy<d)&&(dz<d)) {  // point is inside this cell
	 // as long as that point is not == the centre of this level L cell, split the cell
	 if ((dx>0.01f)||(dy>0.01f)||(dz>0.01f)) {  // point was not the centre of this level L cell
	   H[i] = -1.0f ;             // point is inside the cell, mark the cell for splitting
	 } else {                     // point is at the centre of the cell => store index
	   ii = 1+p ;
	   H[i] = *(float *)(&ii) ;
	 }
       }
     }
   }
}



void __kernel Split_zorder(const    int      L,          // parent level, L=0, 1, 2, ..., LEVELS-
			   __global float   *PX,         // point coordinates
			   __global float   *PY,         // point coordinates
			   __global float   *PZ,         // point coordinates
			   __global float   *HH,         // hierarchy as a single vector
			   __global float3  *CC,         // CC[].x/y/z centre coordinates of each cell
			   __global int     *OFF,        // OFF[LEVELS]
			   __global int     *LCELLS)     // LCELLS[LEVELS]
{  
  // Loop over the cells on level L, set H=1.0+index_to_yt_vector if the cell will not be split, 
  // set H=-1.0 if the cell is to be split -- the links are not yet created
  // Assume that PZ coordinates are in increasing order.
  const int id = get_global_id(0)  ;
  const int gs = get_global_size(0) ;
  int a, b, ii ;
  float x, y, z, d, dx, dy, dz ;
  
  d  =  0.49f*pow(0.5f, L) ;           // half of the level L cell size -- must be <0.5 not to match parent cell
  for(int i=id; i<LCELLS[L]; i+=gs) {  // loop over level L cells
    x    =  CC[OFF[L]+i].x ;           // level L cell centre in root grid coordinates
    y    =  CC[OFF[L]+i].y ;  
    z    =  CC[OFF[L]+i].z ; 
    HH[OFF[L]+i] =  1.0f ;             // H>0  == not split
    // find last index a with PZ[a] < z-d and the first index with  z+d < PZ[b]
    a=0  ;   b=NP ;
    for(int k=0; k<NP; k+=16384)  {
      if (PZ[k]<(z-d)) a=k ;
      if (PZ[k]>(z+d)) {
	b = k ; break ;
      }
    }
    for(int k=a; k<NP; k+=64) {
      if (PZ[k]<(z-d)) a=k ;
      else break ;
    }
    for(int k=b; k>0; k-=64) {
      if (PZ[k]>(z+d)) 	b = k ;
      else break ;
    }
    for(int p=a; p<b; p++) {         // loop over the provided point coordinates
      dx  =  fabs(PX[p]-x) ;   
      dy  =  fabs(PY[p]-y) ;   
      dz  =  fabs(PZ[p]-z) ;
      if ((dx<d)&&(dy<d)&&(dz<d)) {  // point is inside this cell
	if ((dx>0.001f)||(dy>0.001f)||(dz>0.001f)) {  // ok, point was not the centre of this level L cell
	  HH[OFF[L]+i] = -1.0f ;     // mark the cell for splitting
	} else {
	  ii = 1+p ;
	  HH[OFF[L]+i] =  *(float *)(&ii) ;     // H=1+index_to_yt_vector as float ==> is a positive value
	}
      }
    }
  }
}




void __kernel AddLevel(const    int     L,        // level
                       __global float  *H,        // hierarchy vector
                       __global float3 *CC,       // cell centre coordinates
                       __global int    *OFF,      // index offset for each hierarchy level
                       __global int    *LCELLS)   // number of cells on each hierarchy level
{
  // Go over the cells on level L and for each H<=0.0 cell create eight subcells on the next level.
  // Negative values are the child level running index times -1.
  const int id  =  get_global_id(0) ;
  const int gs  =  get_global_size(0) ;
  __global float *HP = &(H[OFF[L  ]]) ;    // vector of parent cells
  __global float *HC = &(H[OFF[L+1]]) ;    // vector of child cells
  float d, x, y, z, f ;
  int ind ;
  // add 1e-6 and it stops working ???????????????
  d  =  0.25f*pow(0.5f, L) ;            // one quarter of the parent cell size [root grid units]
  for(int i=id; i<LCELLS[L]; i+=gs) {   // i = level L cell index
    if (HP[i]<=0.0f) {                  // split the parent cell i
#if 0
      // split parent =  -1 times running index of split parent cells
      ind   = 8*(int)(-HP[i]) ;         // index of the first subcell on the next level L+1
      HP[i] = -*(float *)&ind ;         // the link
#else
      // the above fails due to loss of precision in float
      // now   HP is not -index   but    H = 8 *  [-I2F(index)] == directly 1st child index
      f    =  -HP[i] ;
      ind  =  *(int *)(&f) ;      //     index = 8 *  -F2I(H)
      if ((ind>LCELLS[L+1])||(ind<0)) {
	printf("ERROR IN SPLIT INDEX:  %d\n", ind) ;
      }
#endif
      
      // add the child cells... set value to 1.0 and calculate centre coordinates to CC
      for(int sid=0; sid<8; sid++) {
	HC[ind+sid] = 1.0f ;            // child not split (yet)
	// centre of the parent cell
	x  =  CC[OFF[L]+i].x ;     
	y  =  CC[OFF[L]+i].y ;  
	z  =  CC[OFF[L]+i].z ;
	// each child cell has centre offset by d
	x +=  (sid%2==0)   ?  (-d) :  (+d) ;
	y +=  (sid%4<=1)   ?  (-d) :  (+d) ;
	z +=  (sid<=3)     ?  (-d) :  (+d) ;
	CC[OFF[L+1]+ind+sid].x = x ;
	CC[OFF[L+1]+ind+sid].y = y ;
	CC[OFF[L+1]+ind+sid].z = z ;
      }
    }
  }       
}


