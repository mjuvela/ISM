
// Conversion from global coordinates to cell index
// Conversion from cell index to global coordinates



__kernel void Convert(
                      const  int       cells,       // cells on the current level
                      __global float  *X,           // float[CELLS], the Ramses data vector
                      __global int    *H,           // int[cells], indices for current level
                      __global float  *D            // float[cells], data + encoded indices
                     ) {
   // The encoding and decoding of links
   //        x   = -*(float *)&i  ;
   //        y   = -x ;    i = *(int *)&y ;   
   int id = get_global_id(0) ;
   if (id>=cells) return ;   // one work item per cell of current level
   int   i ;
   i = H[id] ;
   if (i>0) {     
      // i is 1-based index to data in X
      D[id]   =   X[i-1] ;
      if ((i-1)>=N) printf("?????\n") ;
   } else {       
      // this is -link => encode to float
      i       =  -i ;              // positive link  [int32]
      D[id]   =  -*(float *)&i ;   // must be negative as float32
   }
}




__kernel void COO2IND(__global int   *CELLS,   // number of cells on each level
                      __global float *DENS0,   // level 0 densities
                      __global float *DENS1,   // level 1 densities
                      __global float *DENS2,   // level 2 densities
                      __global float *DENS3,   // level 3 densities
                      __global float *DENS4,   // level 4 densities
                      __global float *DENS5,   // level 5 densities                      
                      __global float *DENS6,   // level 6 densities                                            
                      __global float *X,       // X coordinate in the root system
                      __global float *Y,       // Y coordinate in the root system
                      __global float *Z,       // Z coordinate in the root system
                      __global int   *ALEV,    // array of level indices
                      __global int   *AIND     // array if cell indices (within a level)
                     ) {
   // Convert cell indices (level, index) to global coordinates (cell centre)
   //   this for the NEWLINK file format
   int  id = get_global_id(0) ;
   int  gs = get_global_size(0) ;
   int  ind, level, sid, i, j, k ;
   float x, y, z ;
   __global float *DENS[7] ;
   DENS[0] = DENS0 ;  DENS[1] = DENS1 ;  DENS[2] = DENS2 ;  DENS[3] = DENS3 ;  DENS[4] = DENS4 ;
   DENS[5] = DENS5 ;  DENS[6] = DENS6 ;
   for(int ii=id; ii<N; ii+=gs) {           // loop over input coordinates
      x     =  X[ii] ; 
      y     =  Y[ii] ; 
      z     =  Z[ii] ;  // initial root grid coordinates
      i     = (int)floor(x) ;
      j     = (int)floor(y) ;
      k     = (int)floor(z) ;
      if ((i<0)||(i>=NX)||(j<0)||(j>=NY)||(k<0)||(k>=NZ)) {  // outside the cloud
         ALEV[ii] = -999 ;    AIND[ii] = -999 ;   continue ;
      }
      ind   = i + NX*( j + NY*k ) ;    // x runs fastest => python array CLOUD[z, y, x]
      level = 0 ;
      while(DENS[level][ind]<=0.0f) {  // while this is not a leaf node
         // change coordinates (we go to a sub-octet)
         x  =  2.0f*(x-i) ;   y  =  2.0f*(y-j) ;   z  =  2.0f*(z-k) ;
         // (x,y,z), in which cell of the next level octet
         i  =  (x>1.0f) ?  1 : 0 ;
         j  =  (y>1.0f) ?  1 : 0 ;
         k  =  (z>1.0f) ?  1 : 0 ;
         sid = 4*k+2*j+i ;
#if 0         
         ind = (int)(-DENS[level][ind]) + sid ; // sid = index offset within the octet
#else
         float tmp = -DENS[level][ind] ;         
         ind = *(int*)(&tmp) + sid ; // sid = index offset within the octet
#endif    
         level++ ;
      }
      // we found a leaf
      ALEV[ii] = level ;
      AIND[ii] = ind ;
   }
}




__kernel void AverageParent(const int  NP,        // number of parent cells
                            const int  NC,        // number of child nodes
                            __global float *P,    // parent cells
                            __global float *C,    // child cells
                            __global float *H     // links from this, not from P !!
                           ) {
   int id = get_global_id(0) ;
   if (id>=NP) return ;        // only N parent cells
   if (H[id]>0.0f) return ;    // parent cell is already a leaf
   // Otherwise, calculate the average of the eight childred
   //    CHILDREN GUARANTEED TO BE ALL LEAFS !
   float f = -H[id] ;
   int   j = *(int*)(&f) ;   // index of the first child in an octet
   if ((j<0)||(j>=NC)) printf("??? H=%.3e P%.3e %d > %d  delta %d\n", 
                              H[id], P[id], j, NC, j-NC) ;
   f = 0.0f ;
   for(int i=j; i<(j+8); i++) {
      f  +=  C[i] ;
   }
   P[id] = f/8.0f ;  // parent becomes a new leaf cell
}





__kernel void OrderChildren_old(const int NP,
                                __global float *P,
                                __global float *C,
                                __global float *CO) {
   // Go through parent cells with children, copy children from C to CO in the parent order,
   // update the link in the parent cell
   const int id = get_global_id(0) ;  // id == index of parent
   if (id>NP) return ;                // work item updates P[id] and its children
   if (P[id]>0.0f) return ;           // no children
   // first find out how many of the previous cells are parents
   float f ;
   int o, n=0 ;
   for(int i=0; i<id; i++) {      // loop over parent cells, those before id
      if (P[i]<=0.0f) n += 1 ;    // found parent with children => their number is n
   }   
   // there are n parent cells with children before this one => copy id:s children to slot n
   f = -P[id] ;     o   =  *(int*)(&f) ;          // index of the first child in the octet -- in C
   for(int i=0; i<8; i++)  CO[8*n+i] = C[o+i] ;   // copy children from C to CO
   // update the link in the parent =  children are in order =>  8*id !!!!
   // n  = 8*id ;    PO[id] = -*(float*)(&n) ;  
   // CANNO UPDATE P[] WHILE OTHER WORK ITEMS STILL RELY ON THE OLD VALUES
}




__kernel void OrderChildren(const int np,        // parents with children
                            __global float *P,
                            __global float *C,
                            __global float *CO,
                            __global int   *I) {
   // Go through parent cells with children, copy children from C to CO in the parent order,
   // update the link in the parent cell
   const int id = get_global_id(0) ;  // each id = one parent with children
   if (id>=np) return ;               // work item updates P[id] and its children
   // I[] contains NP parents that have children
   int  i = I[id] ;   //   P[i] has some children
   //  i = parent index, parent is P[i] .... this is the id:th parent with children
   float f = -P[i] ;      
   int   o = *(int*)(&f) ;                         // index of the first child in old vector C
   for(int j=0; j<8; j++)  CO[8*id+j] = C[o+j] ;   // copy children from C to CO,  id:th parent with children
   // parent indices must be updated separately - cannot change P[] while it is being used
}

__kernel void AverageParentThresholded(const int  NP,        // number of parent cells
                                       const int  NC,        // number of child nodes
                                       __global float *P,    // parent cells
                                       __global float *C,    // child cells
                                       const float TH        // theshold value for joining
                                      ) {
   // Work items calculates average of eight children and puts the mean value to parent,
   // if all children are below the threshold value
   // Children get values 1e30 to indicate that they have been removed
   const int id = get_global_id(0) ;
   if (id>=NP) return ;        // only NP parent cells
   if (P[id]>0.0f) return ;    // parent cell is already a leaf, do nothing
   float f =  -P[id] ;
   int   o =  *(int*)(&f) ;    // index of the first child in an octet
   f       =  0.0f ;
   for(int i=o; i<(o+8); i++) {
      f  +=  (C[i]<=0.0f) ? (1.0e30f) : (C[i]) ;   // child is a link => cannot average
   }
   f    /= 8.0f ;       // the average value
   if (f<1.0e-7f) printf("??????\n") ;
   if (f>TH) return ;   // average density remains above the threshold (or a child was also a parent)
   P[id] = max(f, 1.0e-7f) ;                   // parent becomes a new leaf cell
   for(int i=o; i<(o+8); i++) C[i] = 1.0e31f ; // mark the child cells as being removed
}



__kernel void UpdateParentLinks(const int NP, __global float *P) {
   const int id = get_global_id(0) ;
   if (id>0) return ;            // single work item !!
   int oind ;
   float f ;
   int ind = 0 ;                 // number of children passed on the next level
   for(int i=0; i<NP; i++) {     // loop over parents, counting number that are not leafs
      if (P[i]<=0.0f) {          // this parent is still a link   D[id]   =  -*(float *)&i ;
         P[i]  =  -(*(float*)(&ind)) ;
         ind  +=  8 ;
      }
   }   
}




__kernel void IND2COO(__global int   *PAR1,    // parents of level 1 cells
                      __global int   *PAR2,    // parents of level 2 cells
                      __global int   *PAR3,    // parents of level 3 cells
                      const    int    NUM,     // number of input cell indices
                      __global int   *ALEV,    // array of level indices
                      __global int   *AIND,    // array if cell indices (within a level)
                      __global float *X,       // array of X coordinates (root coordinates)
                      __global float *Y,       // Y coordinates
                      __global float *Z        // Z coordinates
                     ) {
   // Convert cell indices (level, index-within-level) into global coordinates (cell centre)
   int id = get_global_id(0) ;
   int gs = get_global_size(0) ;
   float x, y, z ;
   // DENS==1e-8 stands for missing density value
   // DENS<=0 means a link to a higher hierarchy level
   int  ind, level ;
   __global int   *PAR[4] ;
   PAR[1]  = PAR1 ;   PAR[2]  = PAR2 ;   PAR[3]  = PAR3 ;
   for(int ii=id; ii<NUM; ii+=gs) {   // loop over input coordinates
      level =  ALEV[ii] ;    
      ind   =  AIND[ii] ;
      if (ind<0) {
         X[ii] = -999.0f ;  Y[ii] = -999.0f ; Z[ii] = -999.0f ;  continue ;
      }
      if (level==0) {   // on the root grid, calculate the coordinates directly
         X[ii] =  (ind % NX)    + 0.5f ;
         Y[ii] =  ((ind/NX)%NY) + 0.5f ;
         Z[ii] =  (ind/(NX*NY)) + 0.5f ;
         continue ;
      } 
      // in a CELL, coordinates [0,2]
      x = 0.5f ;   y = 0.5f ;   z = 0.5f ;
      // step upwards in the hierarchy until we reach the root grid
      while(level>0) {
         if ( ind%2==1)   x += 1.0f ;
         if ((ind%8)>3)   z += 1.0f ;
         if ((ind%4)>1)   y += 1.0f ;           // in OCTET --- 2019-05-18 was += 0.5f ??
         x *= 0.5f ;  y *= 0.5f ;  z *= 0.5f ;  // in parent CELL
         ind  =  PAR[level][ind] ;              // index of parent CELL
         level-- ;                              // level of parent CELL
      }
      // on the root grid 
      X[ii] = (ind % NX)     + x ;
      Y[ii] = ((ind/NX)%NY)  + y ;
      Z[ii] = (ind/(NX*NY))  + z ;
   }
}







// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV






__kernel void ParentsV(
                       __constant   int  *LCELLS,  //  8 - number of cells on each level
                       __constant   int  *OFF,
                       __global   float  *H,
                       __global     int  *PAR
                      ) {
   // *** H and PAR single vectors ***
   // Go through the hierarchy in H and put to PAR links to parent cells
   int id     = get_global_id(0) ;
   int GLOBAL = get_global_size(0) ;
   int ind ;
   float link ;
   for(int level=0; level<(LEVELS-1); level++) {     // loop over parent level = level
      for(int ipar=id; ipar<LCELLS[level]; ipar+=GLOBAL) {  // all cells on parent level
         link = H[OFF[level]+ipar] ;
         if (link<=0.0f) {                            // is a parent to a sub-octet
            link = -link ;                            // positive float
            ind  = *(int *)(&link) ;                  // index of first child on level level+1
            for(int i=0; i<8; i++) {                  // loop over cells in the sub-octet
               PAR[OFF[level+1]+ind+i] = ipar ;       // link to parent cell
               // printf("PARENT %2d %4d -> %2d %4d\n", level+1, ind+i, level, ipar) ; 
            }                                         // eight cells in sub-octet
         } // if there was a sub-octet
      } // parent cells
   } // parent levels
}





__kernel void Ind2CooAllV(
                          __global int   *LCELLS,
                          __global int   *OFF,
                          __global int   *PAR,     // parents all cells *** single vector ***
                          __global float *X,       // array of X coordinates (root coordinates)
                          __global float *Y,       // Y coordinates
                          __global float *Z        // Z coordinates
                         ) {
   // *** PAR is a single vector ***
   // Return coordinates of all cells, in units of the root grid
   int id = get_global_id(0) ;
   int gs = get_global_size(0) ;
   float x, y, z ;
   int  ind, ind0, level0, level ;
   for(int level0=LEVELS-1; level0>=0; level0--) {
      // if (id==0) printf("LEVEL %d --- has %d CELLS\n", level0, LCELLS[level0]) ;
      for(int ind0=id; ind0<LCELLS[level0]; ind0+=gs) {   // loop over input coordinates... here just all cells
         if (level0==0) {   // on the root grid, calculate the coordinates directly
            X[ind0] =  (ind0 % NX)    + 0.5f ;   // X-coordinate runs fastest
            Y[ind0] =  ((ind0/NX)%NY) + 0.5f ;
            Z[ind0] =  (ind0/(NX*NY)) + 0.5f ;
            // if (id%10==0) printf(">>>> %8.2f %8.2f %8.2f\n", X[ind0], Y[ind0],Z[ind0]) ;
            continue ;
         } 
         // in a CELL, coordinates [0,2]
         x = 0.5f ;   y = 0.5f ;   z = 0.5f ;
         ind = ind0 ; level = level0 ;
         // step upwards in the hierarchy until we reach the root grid
         while(level>0) {
            if ( ind%2==1)   x += 1.0f ;           // coordinate 1.5f, not 0.5f
            if ((ind%8)>3)   z += 1.0f ;
            if ((ind%4)>1)   y += 1.0f ;           // in OCTET   --- 2019-05-18  was += 0.5f ???
            x *= 0.5f ;  y *= 0.5f ;  z *= 0.5f ;  // in parent CELL --- [0.0, 1.0]
            ind  =  PAR[OFF[level]+ind] ;          // index of parent CELL
            // printf("Parent %d\n", ind) ;
            level-- ;                              // level of parent CELL
         }
         // on the root grid 
         X[OFF[level0]+ind0] = (ind % NX)     + x ;  // X-coordinate runs fastest
         Y[OFF[level0]+ind0] = ((ind/NX)%NY)  + y ;
         Z[OFF[level0]+ind0] = (ind/(NX*NY))  + z ;
         // printf(">>>> %5d = %10.3e\n", OFF[level0]+ind0, X[OFF[level0]+ind0]) ;
      } // for ii -- cells on current hieararchy level
   } // for level
}




__kernel void Ind2CooAndLevel(
                              __global int   *LCELLS,
                              __global int   *OFF,
                              __global int   *PAR,     // parents all cells *** single vector ***
                              __global float *X,       // array of X coordinates (root coordinates)
                              __global float *Y,       // Y coordinates
                              __global float *Z,       // Z coordinates
                              __global int   *L        // hierarchy level of each cell
                             ) {
   // Return coordinates of all cells, in units of the root grid, also hierarchy level is returned
   int id = get_global_id(0) ;
   int gs = get_global_size(0) ;
   float x, y, z ;
   int  ind, ind0, level0, level ;
   for(int level0=LEVELS-1; level0>=0; level0--) {
      for(int ind0=id; ind0<LCELLS[level0]; ind0+=gs) {   // loop over input coordinates... here just all cells
         if (level0==0) {   // on the root grid, calculate the coordinates directly
            X[ind0] =  (ind0 % NX)    + 0.5f ;   // X-coordinate runs fastest
            Y[ind0] =  ((ind0/NX)%NY) + 0.5f ;
            Z[ind0] =  (ind0/(NX*NY)) + 0.5f ;
            L[ind0] =  0 ;
            // if (id%10==0) printf(">>>> %8.2f %8.2f %8.2f\n", X[ind0], Y[ind0],Z[ind0]) ;
            continue ;
         } 
         // in a CELL, coordinates [0,2]
         x = 0.5f ;   y = 0.5f ;   z = 0.5f ;
         ind = ind0 ; level = level0 ;
         // step upwards in the hierarchy until we reach the root grid
         while(level>0) {
            if ( ind%2==1)   x += 1.0f ;           // coordinate 1.5f, not 0.5f
            if ((ind%8)>3)   z += 1.0f ;
            if ((ind%4)>1)   y += 1.0f ;           // in OCTET   --- 2019-05-18  was += 0.5f ???
            x *= 0.5f ;  y *= 0.5f ;  z *= 0.5f ;  // in parent CELL --- [0.0, 1.0]
            ind  =  PAR[OFF[level]+ind] ;          // index of parent CELL
            // printf("Parent %d\n", ind) ;
            level-- ;                              // level of parent CELL
         }
         // on the root grid 
         X[OFF[level0]+ind0] = (ind % NX)     + x ;  // X-coordinate runs fastest
         Y[OFF[level0]+ind0] = ((ind/NX)%NY)  + y ;
         Z[OFF[level0]+ind0] = (ind/(NX*NY))  + z ;
         L[OFF[level0]+ind0] = level0 ;
      } // for ii -- cells on current hieararchy level
   } // for level
}



__kernel void Ind2CooLevelV(
                            __global int   *LCELLS,
                            __global int   *OFF,
                            __global int   *PAR,     // parents all cells *** single vector ***
                            __global float *X,       // array of X coordinates (root coordinates)
                            __global float *Y,       // Y coordinates
                            __global float *Z,       // Z coordinates
                            const int level0
                           ) {
   // *** PAR is a single vector ***
   // Return coordinates of all cells on level=level0, in units of the root grid
   int id = get_global_id(0) ;
   int gs = get_global_size(0) ;
   float x, y, z ;
   int  ind, ind0, level ;
   for(int ind0=id; ind0<LCELLS[level0]; ind0+=gs) {   // loop over input coordinates... here just all cells
      if (level0==0) {   // on the root grid, calculate the coordinates directly
         X[ind0] =  (ind0 % NX)    + 0.5f ;   // X-coordinate runs fastest
         Y[ind0] =  ((ind0/NX)%NY) + 0.5f ;
         Z[ind0] =  (ind0/(NX*NY)) + 0.5f ;
         // if (id%10==0) printf(">>>> %8.2f %8.2f %8.2f\n", X[ind0], Y[ind0],Z[ind0]) ;
         continue ;
      } 
      // in a CELL, coordinates [0,2]
      x = 0.5f ;   y = 0.5f ;   z = 0.5f ;
      ind = ind0 ; level = level0 ;
      // step upwards in the hierarchy until we reach the root grid
      while(level>0) {
         if ( ind%2==1)   x += 1.0f ;           // coordinate 1.5f, not 0.5f
         if ((ind%8)>3)   z += 1.0f ;
         if ((ind%4)>1)   y += 1.0f ;           // in OCTET   --- 2019-05-18  was += 0.5f ???
         x *= 0.5f ;  y *= 0.5f ;  z *= 0.5f ;  // in parent CELL --- [0.0, 1.0]
         ind  =  PAR[OFF[level]+ind] ;          // index of parent CELL
         level-- ;                              // level of parent CELL
      }
      // on the root grid 
      X[ind0] = (ind % NX)     + x ;  // X-coordinate runs fastest
      Y[ind0] = ((ind/NX)%NY)  + y ;
      Z[ind0] = (ind/(NX*NY))  + z ;
   } // for ii -- cells on current hieararchy level
}




__kernel void Coo2IndV(__global int   *LCELLS,  // number of cells on each level
                       __global int   *OFF,
                       __global float *H,       // hierarchy
                       __global float *X,       // X coordinate in the root system
                       __global float *Y,       // Y coordinate in the root system
                       __global float *Z,       // Z coordinate in the root system
                       __global int   *ALEV,    // array of level indices
                       __global int   *AIND     // array if cell indices (within a level)
                      ) {   
   // Convert coordinates (x, y, z) to  cell indices
   int  id = get_global_id(0) ;
   int  gs = get_global_size(0) ;
   int  ind, level, sid, i, j, k ;
   float x, y, z, tmp ;
   for(int ii=id; ii<N; ii+=gs) {           // loop over input coordinates
      x     =  X[ii] ; 
      y     =  Y[ii] ; 
      z     =  Z[ii] ;  // initial root grid coordinates
      i     = (int)floor(x) ;
      j     = (int)floor(y) ;
      k     = (int)floor(z) ;
      if ((i<0)||(i>=NX)||(j<0)||(j>=NY)||(k<0)||(k>=NZ)) {  // outside the cloud
         ALEV[ii] = -999 ;    AIND[ii] = -999 ;   continue ;
      }
      ind   = i + NX*( j + NY*k ) ;   // in C, x runs fastest
      level = 0 ;
      while(H[OFF[level]+ind]<=0.0f) {  // while this is not a leaf node
         // change coordinates (we go to a sub-octet)
         x    =  2.0f*(x-i) ;   y  =  2.0f*(y-j) ;   z  =  2.0f*(z-k) ;
         // (x,y,z), in which cell of the next level octet
         i    =  (x>1.0f) ?  1 : 0 ;
         j    =  (y>1.0f) ?  1 : 0 ;
         k    =  (z>1.0f) ?  1 : 0 ;
         sid  =  4*k+2*j+i ;
         tmp  = -H[OFF[level]+ind] ;  // to be converted as index to next level
         ind  = *(int*)(&tmp) + sid ; // sid = index offset within the octet
         level++ ;
      }
      // we found a leaf
      ALEV[ii] = level ;
      AIND[ii] = ind ;
   }
}



__kernel void Coo2IndVV(__global int   *LCELLS,  // number of cells on each level
                        __global int   *OFF,
                        __global float *H,       // hierarchy
                        __global float *X,       // X coordinate in the root system
                        __global float *Y,       // Y coordinate in the root system
                        __global float *Z,       // Z coordinate in the root system
                        __global int   *IND,     // array if cell indices (to global vector)
                        const int MAX_LEVEL
                       ) {   
   // Convert coordinates (x, y, z) to  cell indices
   // As Coo2IndV but returning a single index to the global array
   // Calculations use H that contains the grid hierarchy. However, if MAX_LEVEL>=0, we can stop
   // at level MAX_LEVEL and return the index of that cell, which is not necessarily a leaf in the hierarchy.
   // This can be used to read values from data files that do not contain the hierarchy but instead have
   // physical values in all cells, not only in the leaf cells.
   int  id = get_global_id(0) ;
   int  gs = get_global_size(0) ;
   int  ind, level, sid, i, j, k ;
   float x, y, z, tmp ;
   for(int ii=id; ii<N; ii+=gs) {           // loop over input coordinates
      x     =  X[ii] ; 
      y     =  Y[ii] ; 
      z     =  Z[ii] ;  // initial root grid coordinates
      i     = (int)floor(x) ;
      j     = (int)floor(y) ;
      k     = (int)floor(z) ;
      if ((i<0)||(i>=NX)||(j<0)||(j>=NY)||(k<0)||(k>=NZ)) {  // outside the cloud
         IND[ii] = -999 ;   continue ;
      }
      ind   = i + NX*(j + NY*k) ;
      level = 0 ;
      while(H[OFF[level]+ind]<=0.0f) {  // while this is not a leaf node -> go down
         if (level>=MAX_LEVEL) break ;  // do not go any further to the next level
         // change coordinates (we go to a sub-octet)
         x    =  2.0f*(x-i) ;   y  =  2.0f*(y-j) ;   z  =  2.0f*(z-k) ;
         // (x,y,z),  which cell of the next level octet
         i    =  (x>1.0f) ?  1 : 0 ;
         j    =  (y>1.0f) ?  1 : 0 ;
         k    =  (z>1.0f) ?  1 : 0 ;
         sid  =  4*k+2*j+i ;
         tmp  = -H[OFF[level]+ind] ;  // to be converted as index to next level
         ind  = *(int*)(&tmp) + sid ; // sid = index offset within the octet
         level++ ;
#if 0
         if ((ind<0)||(ind>=LCELLS[level])) {
            printf("!!!  %2d %6d < %6d --- sid=%d  %8.4f %8.4f %8.4f  XYZ %8.4f %8.4f %8.4f\n", 
                   level, ind, LCELLS[level], sid, x, y, z, X[ii], Y[ii], Z[ii]) ;
            return ;
         }
#endif
      }
      // we found a leaf
#if 0
      if ((ind<0)||(ind>=LCELLS[level])) {
         printf("***  IND[%6d<%6d] = %d\n", ii, N, OFF[level]+ind) ;
      }
#endif
      IND[ii] = OFF[level] + ind ;
   }
}



__kernel void Coo2IndLevel(__global int   *LCELLS,  // number of cells on each level
                           __global int   *OFF,
                           __global float *H,       // hierarchy
                           __global float *X,       // X coordinate in the root system
                           __global float *Y,       // Y coordinate in the root system
                           __global float *Z,       // Z coordinate in the root system
                           __global int   *L,       // the level at which the cell index is returned
                           __global int   *IND      // array if cell indices (to global vector)
                          ) {   
   // Convert coordinates (x, y, z) to  cell indices
   // As Coo2IndV but returning a single index to the global array
   // Calculations use H that contains the grid hierarchy. However, if MAX_LEVEL>=0, we can stop
   // at level MAX_LEVEL and return the index of that cell, which is not necessarily a leaf in the hierarchy.
   // This can be used to read values from data files that do not contain the hierarchy but instead have
   // physical values in all cells, not only in the leaf cells.
   int  id = get_global_id(0) ;
   int  gs = get_global_size(0) ;
   int  ind, level, sid, i, j, k ;
   float x, y, z, tmp ;
   for(int ii=id; ii<N; ii+=gs) {           // loop over input coordinates
      x     =  X[ii] ; 
      y     =  Y[ii] ; 
      z     =  Z[ii] ;  // initial root grid coordinates
      i     = (int)floor(x) ;
      j     = (int)floor(y) ;
      k     = (int)floor(z) ;
      if ((i<0)||(i>=NX)||(j<0)||(j>=NY)||(k<0)||(k>=NZ)) {  // outside the cloud
         IND[ii] = -999 ;   continue ;
      }
      ind   = i + NX*(j + NY*k) ;
      level = 0 ;
      while(H[OFF[level]+ind]<=0.0f) {  // while this is not a leaf node -> go down
         if (level>=L[ii]) break ;      // do not go any further to the next level
         // change coordinates (we go to a sub-octet)
         x    =  2.0f*(x-i) ;   y  =  2.0f*(y-j) ;   z  =  2.0f*(z-k) ;
         // (x,y,z),  which cell of the next level octet
         i    =  (x>1.0f) ?  1 : 0 ;
         j    =  (y>1.0f) ?  1 : 0 ;
         k    =  (z>1.0f) ?  1 : 0 ;
         sid  =  4*k+2*j+i ;
         tmp  = -H[OFF[level]+ind] ;  // to be converted as index to next level
         ind  = *(int*)(&tmp) + sid ; // sid = index offset within the octet
         level++ ;
      }
      // we found a leaf
      IND[ii] = OFF[level] + ind ;
   }
}




__kernel void PropagateVelocity(const int  np,            // number of cells on the parent level
                                const int  nc,            // number of cells on the child level
                                __global float *HP,       // parent level from hierarchy file
                                __global float *HC,       // child level from hierarchy file
                                __global float *PX,       // parent level in parameter = velocity
                                __global float *PY,       // parent level in parameter
                                __global float *PZ,       // parent level in parameter
                                __global float *PS,       // parent level in parameter = velocity dispersion
                                __global float *CX,       // child level in parameter
                                __global float *CY,       // child level in parameter
                                __global float *CZ,       // child level in parameter
                                __global float *CS) {     // child level in parameter
   // H tells which are the child cells of a given cell on the parent level
   // Given parameter vectors P for the parent level and C for the child level,
   // propagate velocity information C -> P, parent has the average of the child velocities
   // Also propagate velocity dispersion back to child, P->C, if C is a leaf node
   // Assume the radio of turbulence is   2.0**1.5 = 2.83 between hierarchy levels
   const int id = get_global_id(0) ;
   const int gs = get_global_size(0) ;
   double sx, sy, sz, s2x, s2y, s2z, t, w, W ;
   float tmp ;
   int ind ;
   
   for(int i=id; i<np; i+=gs) {                // loop over cells on the parent level
      if (HP[i]>0.0f) {                        // skip if "parent" is actually a leaf node
         if (PX[i]==0.0f) printf("????? -- LEAF WITH VX %12.4e\n", PX[i]) ;
         continue ;
      }
      tmp = -HP[i] ;   ind = *(int *)&tmp ;    // ind = index of the first child cell
      sx = sy = sz = s2x = s2y = s2z = 0.0 ;
      
      
#if 1  // WITHOUT DENSITY WEIGHTING
      for(int j=0; j<8; j++) {
         t = CX[ind+j] ;    sx += t ;  s2x += t*t ;
         t = CY[ind+j] ;    sy += t ;  s2y += t*t ;
         t = CZ[ind+j] ;    sz += t ;  s2z += t*t ;
         // if (id==1) printf("  n %12.4e    CV    %12.4e %12.4e %12.4e\n", HC[ind+j], CX[ind+j], CY[ind+j], CZ[ind+j]) ;
         if (HC[ind+j]<=0.0f) printf(" %d ERROR IN DENSITY\n", id) ;
      }
      // parent has the density-weighted average of the child velocities...
      // except that H does not contain densities for all cells !!  =>  direct geometrical average
      PX[i] =  sx/8.0 ;   PY[i] = sy/8.0 ;   PZ[i] = sz/8.0 ;
      // calculate velocity dispersions in the three directions, std =  sqrt(s2/8.0 - s*s/64.0)
      sx    =  sqrt(clamp(s2x/8.0-sx*sx/64.0, 0.0005, 2500.0)) ; // clamp sigma to [0.022, 50.0]
      sy    =  sqrt(clamp(s2y/8.0-sy*sy/64.0, 0.0005, 2500.0)) ;
      sz    =  sqrt(clamp(s2z/8.0-sz*sz/64.0, 0.0005, 2500.0)) ;      
#else   //  USING DENSITY WEIGHTING
      W = 0.0 ;
      for(int j=0; j<8; j++) {
         w = HC[ind+j] ;
         t = CX[ind+j] ;    sx += w*t ;  s2x += w*t*t ;
         t = CY[ind+j] ;    sy += w*t ;  s2y += w*t*t ;
         t = CZ[ind+j] ;    sz += w*t ;  s2z += w*t*t ;
         if (w<=0.0) printf(" %d ERROR IN DENSITY\n", id) ;
         W += w ;         
# if 0
         if (id%9999==0) {
            printf("vx %10.3e vy %10.3e vz %10.3e  n %10.3e\n", CX[ind+j], CY[ind+j], CZ[ind+j], HC[ind+j]) ;
         }
# endif
      }
      // parent has the density-weighted average of the child velocities...
      // except that H does not contain densities for all cells !!  =>  direct geometrical average
      PX[i] =  sx/W ;   PY[i] = sy/W ;   PZ[i] = sz/W ;                     
      // calculate velocity dispersions in the three directions, std =  sqrt(s2/8.0 - s*s/64.0)
      sx    =  sqrt( clamp(s2x/W-sx*sx/(W*W), 0.0005, 2500.0) ) ;
      sy    =  sqrt( clamp(s2y/W-sy*sy/(W*W), 0.0005, 2500.0) ) ;
      sz    =  sqrt( clamp(s2z/W-sz*sz/(W*W), 0.0005, 2500.0) ) ;           
#endif      
      
      tmp   =  (sx+sy+sz)/3.0 ;  // isotropic, 1d velocity dispersion
      if (isfinite(tmp)) {
         ; // if (id%9999==0) printf(" s = %10.3e\n", tmp) ;
      } else {
         printf("sigma not finite in kernel !!!!\n") ;
      }
#if 0
      PS[i] =  2.83f * tmp ;      // turbulence in parent cell, including scaling ?????
#else
      //  looking at std vs. separation, it looks like standard deviation of velocity values
      //  scales as  L^0.6  =>  ratio between parent and child (L=2) is ~ 1.52 ... ***NOT*** 2.83
      //  (perhaps that scaling was for variance and not std ???
      //  2021-06-09 --- use now scaling  1.6 !!!
      PS[i] =  1.60f * tmp ;
#endif
      
      // set velocity dispersion also to leafs that so far had only (vx, vy, vz)
      for(int j=0; j<8; j++) {
         if (CS[ind+j]<=0.0f)  {  // child was a leaf node => does not have sigma yet
            CS[ind+j] = tmp ;
            if (CS[ind+j]<=0.0f)  printf("???? -- SIGMA IN CHILD STILL %12.4e\n", CS[ind+j]) ;
         }
      }      
      // if (id==-1) printf(" V = %8.4f %8.4f %8.4f   %8.4f    ind=%9d\n", PX[i], PY[i], PZ[i], PS[i], ind) ;
#if (1)
      // make HP on the fly equal to the average density over children => we can then use hierarchy file
      // for the density weighting in the above calculations for all cells
      tmp = 0.0f ;     
      for(int j=0; j<8; j++)  {
         tmp += HC[ind+j] ;
         if (HC[ind+j]<=0.0) printf("ERROR IN DENSITY PROPAGATION !!!\n") ;
      }
      HP[i] = tmp/8.0 ;
      if (isfinite(HP[i])) {
         ; 
      } else {
         printf("Computed HP not finite !!!!\n") ;
      }          
#endif
   }
}



__kernel void PropagateScalar(const    int    np,       // number of cells on the parent level
                              __global float *HP,       // parent level from hierarchy file
                              __global float *HC,       // child level from hierarchy file
                              __global float *P,        // parent level in parameter
                              __global float *C         // child level in parameter
                             ) {
   // Update parent cell parameters P with the average over parameters C on the next level
   const int id = get_global_id(0) ;
   const int gs = get_global_size(0) ;
   float tmp, s ;
   int   ind ;   
   for(int i=id; i<np; i+=gs) {                 // loop over cells on the parent level
      if (HP[i]>0.0f) {
         if (P[i]<0.0f) printf("WTF !\n") ;
         continue ;                             // skip if "parent" is actually a leaf node
      }
      tmp  = -HP[i] ;   ind = *(int *)&tmp ;    // ind = index of the first child cell
      s    = 0.0f ;            
      for(int j=0; j<8; j++) {
         s += C[ind+j] ;
         if (C[id+j]<0.0f) printf("????\n") ;
      }
      P[i] =  s/8.0f ;
      if (id==0) printf(" p[%7d] = %.3e\n", i, P[i]) ;
   }
}

