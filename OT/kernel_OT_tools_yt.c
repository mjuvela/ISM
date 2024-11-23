


__kernel void Link1(const    int     NP,  //  number of parent cells
                    const float      D,   //  size of parent cell
                    __global float  *XP,  //  parent coordinates [0,DIM]
                    __global float  *YP,  //  
                    __global float  *ZP,  //  
                    __global float  *HP,  //  parent density/link vector
                    const    int     NC,  //  number of child cells
                    __global float  *XC,  //  child coordinates
                    __global float  *YC,  //  
                    __global float  *ZC   //  
                   ) {
   const int id = get_global_id(0) ;    // id = single parent cell
   if (id>=NP) return ;
   // Find children with coordinates offset by 0.25*D from (XP,YP,ZP), the centre of the parent cell.
   // If the first child is found, set parent cell H=-1.
   // Because the cells in the child octet are not consecutive, cannot make links yet.
   //  level L, cell coordinates  (0.5)^(L+1) + i *0.5^L
   //                L = 0        0.5   +  i*1.0  
   //                L = 1        0.25  +  i*0.5  
   //                L = 2        0.125 +  i*0.25
   // Cell size D, children at (XP,YP,ZP) +- 0.25*D
   const float x   = XP[id]-0.25f*D ;   // coordinates of the first cell in the child octet
   const float y   = YP[id]-0.25f*D ;
   const float z   = ZP[id]-0.25f*D ;
   const float EPS = 3.0e-5f ;          // float enough for DIM=512, up to NLEV=7 ???
   // Find an interval that contains ZC==z, if such exists
   int a=0, b=NC-1, c ;
   float v ;
   for(int k=0; k<11; k++) {
      c = (a+b)/2 ;
      if (ZC[c]>(z+EPS)) {
         b = c ;
      } else {
         if (ZC[c]<(z-EPS)) {
            a = c ;
         }
      }
   }
   for(int k=a; k<=b; k++) {  // test all child cells [a,b]
      if ((fabs(XC[k]-x)<EPS)&&(fabs(YC[k]-y)<EPS)&&(fabs(ZC[k]-z)<EPS)) {
         HP[id] = -2.0f ;     // just mark as a cell having children
      }
   }
}




__kernel void Link2(const    int     NP,  //  number of parent cells in I to process
                    __global int    *I,   //  indices of the parent cells I[NP]
                    const float      D,   //  size of parent cell
                    __global float  *XP,  //  parent coordinates [0,DIM]
                    __global float  *YP,  //  
                    __global float  *ZP,  //  
                    const    int     NC,  //  number of child cells
                    __global float  *XC,  //  child coordinates
                    __global float  *YC,  //  
                    __global float  *ZC,  //  
                    __global float  *HC,  //  child vector, input
                    __global float  *xc,  //  output child coordinates
                    __global float  *yc,  //  
                    __global float  *zc,  //  
                    __global float  *hc   //  child vector, output
                   ) {
   const int id = get_global_id(0) ;  // id finds children of the parent I[id]
   // parent cell index I[id], should have the first child at index 8*id
   if (id>=NP) return ;
   int a, b, c, i, ip = I[id] ;    // parent cell ip
   // Coordinates of the parent cell, children at x/y/z +- 0.25*D
   const float x =  XP[ip] ;
   const float y =  YP[ip] ;
   const float z =  ZP[ip] ;
   // Octet will start in hc at index 8*id
   // Find the eight child cells based on the coordinates (XC, YC, ZC)
   // These are in an ascending order of z coordinate => start by bracketing all cells that
   // are in the z-interval  [z-0.3*D, z+0.3*D]
   a = 0 ;  b = NC-1 ;
   for(int k=0; k<12; k++) {
      c = (a+b)/2 ;
      if (ZC[c]>(z+0.3f*D)) {
         b = c ;
      } else {
         if (ZC[c]<(z-0.3*D)) {
            a = c ;
         }
      }
   }
   // Parent coordinates (x,y,z)
   // Now all child cells of current parent are within the [a,b] interval of child cells.
   // The child cells are at coordinates  x+-0.25*D, y+-0.25*D, z+-0.25*D
   c = 0 ;  // now the count of child cells found
   for(int k=a; k<=b; k++) {
      if ((fabs(x-XC[k])<(0.3*D))&&(fabs(y-YC[k])<(0.3*D))&&(fabs(z-ZC[k])<(0.3*D))) {
         // HC[k] should be one of the eight children
         if (ZC[k]<z) {    // lower z plane
            if (YC[k]<y) {
               i =  (XC[k]<x) ? 0 : 1 ;
            } else {
               i =  (XC[k]<x) ? 2 : 3 ;
            }
         } else { // upper z plane
            if (YC[k]<y) {
               i =  (XC[k]<x) ? 4 : 5 ;
            } else {
               i =  (XC[k]<x) ? 6 : 7 ;
            }
         }
         // copy the child k from HC to element 8*id+i in hc
         xc[8*id+i] = XC[k] ;
         yc[8*id+i] = YC[k] ;
         zc[8*id+i] = ZC[k] ;
         hc[8*id+i] = HC[k] ;
         c += 1 ;
      }
   }
   if (c!=8) { // something wrong, all children not found or too many found
      printf("parent ip = %10d  %8.4f %8.4f %8.4f  =>  children:  %d ???\n", 
             ip, x, y, z, c) ;
   }
}
   
   
   
