

__kernel void SigmaUpdateLOC(const    int    N,       // cells on the parent level
							 __global float *H,       // parent cells (including child links)
							 __global float *VXP,     // velocities at parent level
							 __global float *VYP, 
							 __global float *VZP, 
							 __global float *SIGMAP,  // sigma values for parent level cells
							 __global float *VXC,     // velocities at child level
							 __global float *VYC, 
							 __global float *VZC,
							 __global float *SIGMAC   // sigma values for child level cells
							 ) {
	// Turbulence estimated as std over octet = over the 8 child cells, assuming [km/s] values.
	// Parent level turbulence is equal to    std * k1
	// Child level turbulence  is equal to    std * k2
	// Now using:
	//               k1 = 1.0        i.e. std between children == parent level microturbulence
	//               k2 = 1.0/2.83   i.e. x2 smaller scale corresponding to 1/2.83 turbulence
	const int id = get_global_id(0) ;
	const int gs = get_global_size(0) ;
	float f, sx, sy, sz, s2x, s2y, s2z ;
	int   c ;
	for(int i=id; i<N; i+=gs) {      // loop over parent cells
		if (H[i]>0.0f) {
			SIGMAP[i] = -2.0f ;    // mark the turbulence of leaf cells as so far undefined
			continue ;             // no children, nothing to do
		}
		f   =  -H[i] ;
		c   =  *(int*)(&f)  ;     // index of the child cell (first in octet)
		// compute std in three directions, loop over the eight child cells
		// printf(" V = %10.3e  %10.3e  %10.3e\n", VXC[c], VYC[c], VZC[c]) ;
		sx  = 0.0f ; sy  = 0.0f ; sz  = 0.0f ;
		s2x = 0.0f ; s2y = 0.0f ; s2z = 0.0f ;
		for(int j=c; j<c+8; j++) {   // j = index of child cells
			f = VXC[j] ;   sx += f ;    s2x += f*f ;
			f = VYC[j] ;   sy += f ;    s2y += f*f ;
			f = VZC[j] ;   sz += f ;    s2z += f*f ;
		}
		// set sigma value in the parent cell
		//    var  =    (s2 - s^2/N) / N
		s2x  =  sqrt((s2x-sx*sx/8.0f+1.0e-5f)/8.0f) ;
		s2y  =  sqrt((s2y-sy*sy/8.0f+1.0e-5f)/8.0f) ;
		s2z  =  sqrt((s2z-sz*sz/8.0f+1.0e-5f)/8.0f) ;
		s2z  =  (s2x+s2y+s2z)/3.0f ;        // sigma(parent) == std(children)
		// printf("   s2z   %10.3e\n", s2z) ;
#if 1
      // If child cells already have turbulence values, perhaps that should be used to bias
      // turbulence estimates in parent towards 1.5 times mean turbulence of children ?
      float child_turbulence = 0.0f ;
      int   no = 0 ;
		for(int j=c; j<c+8; j++) {  // j = child level index
			if (SIGMAC[j]>0.0f) {
            no += 1 ;
            child_turbulence += SIGMAC[j] ;
         }
		}
      if (no==8) {
         s2z =  0.7f*s2z + 0.3f*1.5f*child_turbulence ; //  < velocity difference,  1.5 * average child turbulence times >
      }
#endif
#if 1
		// include clipping to avoid extreme values ???
		s2z  =  clamp(s2z, 0.01f, 90.0f) ;  // relevant for dense ISM (minus large shocks)
#endif
		SIGMAP[i] = s2z ;                   // microturbulence into the parent cell
		// printf("     %9d  s= %8.4f      %8.4f %8.4f %8.4f \n", id, s2z, sx/8.0f, sy/8.0f, sz/8.0f) ;
		//  set velocity values in the parent cell  = average over the children
		//  kenel is called from bottom up, and one needs velocity also for parent cells
		//  .... why? just to have (unused) estimates also above the leafs ?
		//  ....  no, because parent may have leaf siblings, whose turbulence estimate
		//  ....  also requires the velocity of this parent cell
		VXP[i]   =   sx/8.0f ;     VYP[i]   =   sy/8.0f ;     VZP[i]   =   sz/8.0f ;
		// set sigma in children, if their sigma estimate is still missing
		for(int j=c; j<c+8; j++) {  // j = child level index
			if (SIGMAC[j]<=0.0f) SIGMAC[j] = s2z/1.50f ;  // sigma(children) = std(children)/2.83 ??
		}
	}
}





__kernel void SigmaUpdateLOC_root(const    int    NX,      // cells on the parent level
								  const    int    NY,
								  const    int    NZ,
								  __global float *H, 
								  __global float *VX,
								  __global float *VY, 
								  __global float *VZ, 
								  __global float *S
								  ) {
	// Fill in missing microturbulence at the root level.
	// Calculate standard deviation over 3x3x3 cells and scale std down by ... as the
	// estimate of the in-cell microturbulence.
	const int id = get_global_id(0) ;
	const int gs = get_global_size(0) ;
	if (id>=(NX*NY*NZ)) return ;
	if (S[id]>0.0f) return ;
	const int i0  =  id % NX ;
	const int j0  =  (id/NX) % NY ;
	const int k0  =  id / (NX*NY) ;
	int   n = 0 ;
	float sx =0.0f, sy =0.0f, sz =0.0f, v ;
	float s2x=0.0f, s2y=0.0f, s2z=0.0f ;
	for(int k=max(0,k0-1); k<min(NZ, k0+2); k++) {
		for(int j=max(0,j0-1); j<min(NY,j0+2); j++) {
			for(int i=max(0,i0-1); i<min(NZ,i0+2); i++) {
				v = VX[i+NX*(j+NY*k)] ;   sx += v ;   s2x += v*v ;
				v = VY[i+NX*(j+NY*k)] ;   sy += v ;   s2y += v*v ;
				v = VZ[i+NX*(j+NY*k)] ;   sz += v ;   s2z += v*v ;
				// printf("    v = %10.3e\n", v) ;
				n += 1 ;
			}
		}
	}
	s2x   =  sqrt((s2x-sx*sx/n+1.0e-5f)/n) ;  // std in x direction
	s2y   =  sqrt((s2y-sy*sy/n+1.0e-5f)/n) ;
	s2z   =  sqrt((s2z-sz*sz/n+1.0e-5f)/n) ;
	s2z   =  (s2x+s2y+s2z)/3.0f ;
	S[id] =  s2x/2.0f ;              //     2.83f  ???
#if 0
	if (isfinite(s2x)) {
		;
	} else {
		printf("   n=%d  last v = %10.3e   ==> s = %10.3e\n", n, v, s2x) ;
	}
	printf(" id = %7d   s = %10.3e\n", id, s2x/2.0f) ;
#endif
}

