

__kernel void SigmaUpdate(const int  N,
                          __global float *H,
                          __global float *VXP, 
                          __global float *VYP, 
                          __global float *VZP, 
                          __global float *SIGMAP,
                          __global float *VXC,
                          __global float *VYC, 
                          __global float *VZC,
                          __global float *SIGMAC) {
   const int id = get_global_id(0) ;
   const int gs = get_global_size(0) ;
   if  (id>=N) return ;   // id = parent index
   float f, sx, sy, sz, s2x, s2y, s2z ;
   for(int i=id; i<N; i+=gs) {   // loop over parent cells
      if (H[i]>0.0f) continue ;  // not a parent cell, nothing to do
      f   = -H[i] ;
      c   =  *(int*)(&f)  ;      // index of the child cell
      // compute std in three directions, loop over the eight child cells
      for(int j=c; j<c+8; j++) {
         f = VXC[c] ;   sx += f ;    s2x += f*f ;
         f = VYC[c] ;   sy += f ;    s2y += f*f ;
         f = VZC[c] ;   sz += f ;    s2z += f*f ;
      }
      // set sigma value in the parent cell
      s2x  =  sqrt((s2x-sx*sx/8.0f)/8.0f) ;
      s2y  =  sqrt((s2y-sy*sy/8.0f)/8.0f) ;
      s2z  =  sqrt((s2z-sz*sz/8.0f)/8.0f) ;
      s2z  =  2.83f*(s2x+s2y+s2z)/3.0f ;     // 2.83f = 2.0^1.5, scaling with scale
      SIGMA[level][i] = s2z ;
      //  set velocity values in the parent cell  = average over the children
      VXP[i]   =   sx/8.0f ;     VYP[i]   =   sy/8.0f ;     VZP[i]   =   sz/8.0f ;
      // copy the same sigma value down to the children, if their sigma is still == 0.0
      for(int j=c; j<c+8; j++) {
         if (SIGMAC[j]<=0.0f) SIGMAC[j] = s2z ; 
      }
   }
}
