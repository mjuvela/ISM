
// 2017-12-08


__kernel void AverageParent(const int  NP,        // number of parent cells
                            const int  NC,        // number of child nodes
                            __global float *P,    // parent cells
                            __global float *C     // child cells
                           ) {
   int id = get_global_id(0) ;
   if (id>=NP) return ;          // only NP parent cells
   if (P[id]>1.0e-9f) return ;   // parent cell is already a leaf
   // Otherwise, calculate the average of the eight childred
   // children are at this point guaranteed to be all leafs cells
   float f = -P[id] ;        // P[id] was a link to suboctet
   int   j = *(int*)(&f) ;   // index of the first child in an octet
   if ((j<0)||(j>=NC)) printf("??? P %.3e -> %d  not in [0, %d]\n", P[id], j, NC-1) ;
   // Average the values of the eight subcells
   f = 0.0f ;
   for(int i=j; i<(j+8); i++)  f += C[i] ;
   P[id] = f/8.0f ;  // parent becomes a new leaf cell
}



