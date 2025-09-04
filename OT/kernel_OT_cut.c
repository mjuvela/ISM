

__kernel void CutVolumeOld(__global int   *LIMITS,
                        __global int   *OFF,
                        __global int   *LCELLS,
                        __global float *H,              // hierarchy, used only for the links
                        __global float *A,              // cut from A ...
                        __global float *B,              //            ... to B
                        __global int   *lcells) {   
   const int id = get_global_id(0) ;
   if (id>0) return ;    // single work item !!
   int Ia=LIMITS[0], Ib=LIMITS[1], Ja=LIMITS[2], Jb=LIMITS[3], Ka=LIMITS[4], Kb=LIMITS[5];
   int op, oc, count, i, j, k, ind ;
   float p ;
   // root grid
   count = 0 ;  // running index in the new cloud
   for(int ind=0; ind<(NX*NY*NZ); ind++) {        //  loop over root level, old cloud, x-axis runs fastest
      i  =  ind % NX ;
      j  =  (ind/NX) % NY ;
      k  =  ind/(NX*NY) ;
      if ((i<Ia)||(i>Ib)||(j<Ja)||(j>Jb)||(k<Ka)||(k>Kb)) continue ;   // outside cut volume
      B[count] = A[ind] ;      // if there are links, they still point to the original hierarchy
      H[count] = H[ind] ;      // modify H in place... only the cut volume cells, H used for the links only
      count   += 1 ;
   }
   // after this point we cannot be sure than for parents p1 < p2 child indices are c1 < c2 ???
   lcells[0] = count ;
   for(int ilevel=1; ilevel<LEVELS; ilevel++) {    // loop over child level
      op = OFF[ilevel-1] ;  oc = OFF[ilevel] ;     // offsets on parent and child level
      count = 0 ;                                  // indices start with zero for each level
      for(int iparent=0; iparent<lcells[ilevel-1]; iparent++) {      // loop over all **new** parents at ilevel-1
         // all remaining parents have been moved to the beginning of the H vector
         p = H[op+iparent] ;                       // parent cell --- in the new hierarchy, is that a link?
         if (p>0.0f) continue ;                    // if ilevel-1 cell was a leaf... nothing to do
         p = -p ;   ind =  *(int*)(&p) ;           // otherwise it is a link = index to full vector A[ilevel]
         B[op+iparent]  =  -(*(float*)(&count)) ;  // B parent link at level ilevel-1 is updated --> count
         // do not bother to update H[op+iparent]... that is not needed in the kernel
         for(int i=0; i<8; i++) {                  // copy the eight child cells to proper place
            B[oc+count+i] = A[oc+ind+i] ;
            H[oc+count+i] = H[oc+ind+i] ;          // update child level also in H ... to have the correct links
            if (count>ind) printf("???? Trouble in CutVolume !!!!!\n") ;      // we assume that child cells are in the order of their parent cells...
         }
         count += 8 ;
      }
      lcells[ilevel] = count ;
      // now the first count elements after B[oc] is the level ilevel vector in the cut hierarchy
   }
   // when writing the file, the host must only drop the extra cells i>=lcells[ilevel] from each level
}
   
   




__kernel void CutVolume(__global int   *LIMITS,
                        __global int   *OFF,
                        __global int   *LCELLS,
                        __global float *H0,             // hierarchy, used only for the links
                        __global float *H1,             // hierarchy, all children reordered to parent order
                        __global float *X0,             // cut from A ...
                        __global float *X1,             //            ... to B
                        __global int   *lcells) {   
   const int id = get_global_id(0) ;
   if (id>0) return ;    // single work item !!
   int Ia=LIMITS[0], Ib=LIMITS[1], Ja=LIMITS[2], Jb=LIMITS[3], Ka=LIMITS[4], Kb=LIMITS[5];
#if 0
   int op, oc, count, i, j, k, ind ;
#else
   long int op, oc ;
   int  count, i, j, k, ind ;
#endif
   float p ;
   // root grid
   count = 0 ;  // running index in the new cloud
   printf("K: root grid\n") ;
   for(int ind=0; ind<(NX*NY*NZ); ind++) {        //  loop over root level, old cloud, x-axis runs fastest
      i  =   ind % NX ;
      j  =  (ind/NX) % NY ;
      k  =   ind/(NX*NY) ;
      if ((i<Ia)||(i>Ib)||(j<Ja)||(j>Jb)||(k<Ka)||(k>Kb)) continue ; // out of selected volume
      X1[count] = X0[ind] ;    // COPY DATA      --- if there are links, they still point to the original hierarchy
      H1[count] = H0[ind] ;    // COPY HIERARCHY --- only the cut volume cells, H used for the links only
      count   += 1 ;
   }
   // level 0 has been fixed, selected cells at the beginning of X1 and H1
   lcells[0] = count ;   // parent cells ...
   for(int ilevel=0; ilevel<LEVELS-1; ilevel++) {   // loop over parent level cells
      printf("ilevel = %d\n", ilevel) ;
      // parent cells are at the beginning of X1 and H1, pointing to original level+1 cells in X0 and H0
      // loop over lcells[ilevel] parent cells
      //   if parent is lead, do nothing
      //   of parent is link
      //       copy children in order X0->X1, H0->H1
      //       update link in the parent
      op = OFF[ilevel] ;  oc = OFF[ilevel+1] ;      // offsets on parent and child level in X0 and H0
      count = 0 ;                                   // count children -- indices start with zero for each level
      for(int ip=0; ip<lcells[ilevel]; ip++) {      // loop over INCLUDED PARENTS
         // printf("parent %d / %d\n", ip, lcells[ilevel]) ;
         p = H1[op+ip] ;                            // parent cell, in case of link, points to original hierarchy
         if (p>0.0f) continue ;                     // parent was leaf, do nothing
         p = -p ;   ind =  *(int*)(&p) ;            // otherwise it is a link = index to X0[ilevel+1]
         // copy children to the beginning of H1, X1
         for(int i=0; i<8; i++) {                   // copy the eight child cells to proper place in X1 and H1
            X1[oc+count+i] = X0[oc+ind+i] ;         // ind   = pointer to old H0 and X0
            H1[oc+count+i] = H0[oc+ind+i] ;         // count = runs in order in X1 and H1
         }
         // update the link in the parent
         X1[op+ip]  =  -(*(float*)(&count)) ;  // link at level ilevel-1 is updated --> count, X1 is filled in order
         H1[op+ip]  =  -(*(float*)(&count)) ;  // update target hierarchy -- in H1 active cells in order 
         count     += 8 ;                      // direct counter for the number of added child cells
      }
      lcells[ilevel+1] = count ;
      printf("next levels %d has %d cells\n", ilevel+1, count) ;
      // now the first count elements after B[oc] is the level ilevel vector in the cut hierarchy
   }
   // when writing the file, the host must only drop the extra cells i>=lcells[ilevel] from each level
}
   
   
