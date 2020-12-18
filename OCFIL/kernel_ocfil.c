#define PI      3.1415926536f
#define PIHALF  1.5707963268f
#define PYOPENCL_NO_CACHE 1

__kernel void Sigma(__global float *S,     // input image
                    __global float *P) {   // partial sums for variance calculation
   // Std calculated by comparing data pairs S(x) and S(x+STEP)
   const int   id = get_global_id(0) ;
   const int   GS = get_global_size(0) ;
   float s=0.0f, s2=0.0f, y0, y ;
   int   n=0, i, j, ind, d = (int)STEP ;
   for(int k=id; k<(N*M); k+=GS) {
      i = k % M ;    j = k/M ; 
      if ((i<STEP)|(i>=(M-STEP))|(j<STEP)|(j>=(N-STEP))) continue ;
      y0 =  S[k] ;
      if (y0==0.0f) continue ;  // missing pixels should have value 0.0
      y  =  S[k-d] ;  // for simplicity, we take STEP only along coordinate axes (should be enough for statistics)
      if (y!=0.0f) {    s +=  y-y0 ;   s2 += (y-y0)*(y-y0) ;  n += 1 ;      }
      y  =  S[k+d] ;
      if (y!=0.0f) {    s +=  y-y0 ;   s2 += (y-y0)*(y-y0) ;  n += 1 ;      }
      y  =  S[k-d*M] ;
      if (y!=0.0f) {    s +=  y-y0 ;   s2 += (y-y0)*(y-y0) ;  n += 1 ;      }
      y  =  S[k+d*M] ;
      if (y!=0.0f) {    s +=  y-y0 ;   s2 += (y-y0)*(y-y0) ;  n += 1 ;      }         
   }
   P[id] =  s2/n - (s/n)*(s/n) ;   // variance estimate from one work item
}


__kernel void Probability(const float SIGMA,     // std of variations at the STEP scale
                          __global float *S,     // input image
                          __global float *P,     // signal significance / probability
                          __global float *T) {   // position angle
   // Probability of a pixel being on a filament
   int   id = get_global_id(0) ;
   if (id>=(N*M)) return ;
   int   i0 = id % M ;     // column  --   0 <= i < M
   int   j0 = id / M ;     // row
   int   ii, jj ;
   float border = 1.0f + sqrt(pow((float)(0.5f*LEGS*STEP),2.0f)+(float)(STEP*STEP)) ;
   if ((i0<border)||(j0<border)||(i0>=(M-1.0f-border))||(j0>=(N-1.0f-border))) {  // too close to image borders
      P[id] = T[id] = -999.0f ;      return ;
   }
   float a, b, c, y, phi, si, co ;
   float lnP, lnP0 = -1.0e30f, phi0=-999.0f  ;   
   for(int iphi=0; iphi<NDIR; iphi++) {     // test different position angles for the filament
      phi = -PIHALF+(iphi+0.5f)*PI/NDIR ;   // only over pi radians
      si  =  sin(phi) ;
      co  =  cos(phi) ;
      lnP =  0.0f ;                         // log-probability for current position angle
      for(int j=0; j<LEGS; j++) {           // along the length of the centipede
         // each position along the potential filament direction, calculate the probability
         // that this is part of a filament = centre value larger than either of the flank values left
         y        =  -si*(j-0.5f*(LEGS-1.0f)) - co ;
         ii       =  (int)(i0+y*STEP) ;
         y        =  +co*(j-0.5f*(LEGS-1.0f)) - si  ;
         jj       =  (int)(j0+y*STEP) ;
         a        =  S[ii+M*jj] ;
         // centre
         y        =  -si*(j-0.5f*(LEGS-1.0f))  ;
         ii       =  (int)(i0+y*STEP) ;
         y        =  +co*(j-0.5f*(LEGS-1.0f))   ;
         jj       =  (int)(j0+y*STEP) ;
         b        =  S[ii+M*jj] ;
         // right
         y        =  -si*(j-0.5f*(LEGS-1.0f)) + co  ;
         ii       =  (int)(i0+y*STEP) ;
         y        =  +co*(j-0.5f*(LEGS-1.0f)) + si  ;
         jj       =  (int)(j0+y*STEP) ;
         c        =  S[ii+M*jj] ;
         // probability p(b>a) and p(b>c) = p(b>a)*p(b>c) ~ ridgeness,  prob(b>a) =  erf( (b-a)/SIGMA )
         lnP     +=  log(erf((b-a)/SIGMA)) + log(erf((b-c)/SIGMA)) ;
      }
      if (lnP>lnP0) {   // most probable direction so far
         lnP0 = lnP ;  phi0 = phi ;
      }
   }  // for iphi (loop over position angles)
   P[id]  = lnP0 ;     T[id]  = phi0 ;
}


#define I(x,y) ((int)(floor((float)x)+floor((float)y)*M))  // indexing of 2d images
#define SS    7   // number of sidesteps, keeping on the track of maximum p


__kernel void Trace(const    int   NO,     // number of filaments
                    const    float PLIM,   // stay within P>PLIM
                    __global float *XY,    // pixel coordinates as start positions XY[NO, 2]
                    __global float *S,     // input image
                    __global float *P,     // probability, 2D image
                    __global float *T,     // position angle
                    __global int   *L,     // labels (int32 image)
                    __global float *RW) {  // route waypoints R[NO, 1000, 2]
   // Based on P image, follow the spines of the filements => (x, y, p) along the ridge
   int   id = get_global_id(0) ;
   if   (id>=NO) return ;
   int   no=0 ;
   float x=XY[2*id], y = XY[2*id+1], pa=T[I(x,y)], vx, vy, ux, uy, xx, yy, x0, y0, x1, y1 ;
   float DX=STEP*0.25f ;     // step along the filament
   __global float *R = &(RW[id*3000]) ;    // (x,y) pixel coordinates for this filament to be saved
   // Initial step down in the map
   vx = DX*sin(-pa) ;    vy = DX*cos(pa) ;
   // go always down so the final trace will be in increasing order of declination
   if (vy>0.0f)  {  vx *= -1.0f ; vy *= -1.0f ; }
   while(1) {                                     // loop in one direction until P<PLIM
      x0 = x ; y0 = y ;
      x  += vx ;   y  += vy ;
      pa = T[I(x,y)] ;                            // PA at the new position
      ux = DX*sin(-pa) ; 
      uy = DX*cos( pa) ;                          // vector for direction forward from the new position
      if ((vx*ux+vy*uy)<0.0f) {                   // if v and u at more than 90 degrees apart, flip u
         ux *= -1.0f ;  uy *= -1.0f ;
      }
      vx = ux ;   vy = uy ;                       // step completed, new position (x,y) with direction (vx, vy)
      // Move sideways to the centre of the ridge
      x1 = x ; y1 = y ;
      for(int k=-SS; k<=+SS; k++) {               // at most SS sidesteps either way
         xx = x1-0.2f*k*vy ;  yy = y1+0.2f*k*vx ; // along perpendicular direction (-vy, +vx)
         if (P[I(xx,yy)]>P[I(x,y)]) {             // replace (x,y) with (xx, yy)
            x = xx ;  y = yy ; 
         }
      }
      if (L[I(x,y)]!=id) {
         x  = x0 ;   y = y0 ;   break ;           // we are at one end of the filament, drop the last step
      }      
      pa = T[I(x,y)] ;   vx = DX*sin(-pa) ;     vy = DX*cos( pa) ;
      if ((vx*ux+vy*uy)<0.0f) {  vx *= -1.0f ;  vy *= -1.0f ;  }  // u || v
      no++ ; if (no>999) break ;
   }      
   // now we are at one end of the filament, start in the opposite direction, hopefully increased dec order
   ux = -1.0f*vx ; uy = -1.0f*vy ;       // position (x,y), start in directin (vx, vy)
   pa =  T[I(x,y)] ;
   vx =  DX*sin(-pa) ;   vy =  DX*cos( pa) ;
   if ((vx*ux+vy*uy)<0.0f) {   vx *= -1.0f ;  vy *= -1.0f ;  }
   R[0] = x ; R[1] = y ; R[2] = P[I(x,y)] ; no = 1 ;        // add the first position in the filament route
   while(1) {                            // step to the other end of the filament
      x += vx ;   y += vy ;
      pa = T[I(x,y)] ;                   // PA at the new position
      ux = DX*sin(-pa) ;   uy = DX*cos( pa) ;
      if ((vx*ux+vy*uy)<0.0f) {  ux *= -1.0f ;  uy *= -1.0f ;   }   // u || v
      vx = ux ;   vy = uy ;
      // Move sideways to the centre of the ridge
      x1 = x ;  y1 = y ;
      for(int k=-SS; k<=+SS; k++) {
         xx = x1-0.2f*k*vy ; yy = y1+0.2f*k*vx ;  // filament (vx, vy), perpendicular direction (-vy, vx)
         if (P[I(xx,yy)]>P[I(x,y)]) {             // replace (x,y) with (xx, yy)
            x = xx ;  y = yy ;
         }
      }
      if (L[I(x,y)]!=id) break ;                  // we stepped out of the filament area
      R[3*no] = x ;    R[3*no+1] = y ;   R[3*no+2] = P[I(x,y)] ;  no += 1 ;
      if (no>999) break ;  // there could be a loop !!
   }
   R[3*no] = -1.0f ;  R[3*no+1] = -1.0f ;  R[3*no+2] = -1.0f ;
}


__kernel void Filament2D(const int       no,    // index of the filament
                         const int       np,    // number of samples along the filament
                         const int       nr,    // radial points, samples perpendicular to the filament
                         __global float *RW,    // RW[no, 1000, 3] of the waypoints
                         __global float *S,     // image
                         __global float *T,     // position angles of the local filament direction
                         __global float *F) {   // image of the 2d filament  [no, 2*nr+1]
   const int id = get_global_id(0) ;
   if (id>=np) return ;
   // one work item per one image row
   float x0, y0, x, y, vx, vy, pa, smax=-999.0f ;
   int kmax=-999 ;
   __global float *f = &(F[id*(2*nr+1)]) ;      // single row in the target image
   x0 =  RW[no*3000+id*3+0] ;  y0 =  RW[no*3000+id*3+1] ;  pa =  T[I(x0,y0)] ;
   vx =  0.25f*STEP*sin(-pa) ;  vy =  0.25f*STEP*cos( pa) ;  // perpendicular direction
   for(int k=-nr; k<=+nr; k++) {
      x = x0 + k*(-vy) ;     y = y0 + k*vx ;
      if ((x<0)|(x>=M)|(y<0)|(y>=N)) {
         f[k+nr] = 0.0f ;
      } else {
         y       =  S[I(x, y)] ;
         f[k+nr] = y ;
         if (y>smax) {
            smax = y ;   kmax = k ;
         }
      }
   }   
#if 0  // shift the intensity peak explicitly to the centre
   if (abs(kmax)<STEP) {  // a maximum shift by ~one beam
      if (kmax>0) {       // shift kmax steps downwards
         for(int i=0; i<(2*nr+1-kmax); i++)  f[i] = f[i+kmax] ;
         for(int i=(2*nr+1-kmax); i<(2*nr+1); i++) f[i] = 0.0f ;
      } else {            // shift upwards
         for(int i=(2*nr+1-1); i>kmax; i--) f[i] = f[i-kmax] ;
         for(int i=0; i<kmax; i++) f[i] = 0.0f ;
      }
   }
#endif
}
