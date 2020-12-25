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


#if 0
#define I(x,y) ((int)(floor((float)x)+floor((float)y)*M))  // indexing of 2d images
#else
# define I(x,y) (clamp((int)(floor((float)(x))),0,M-1)+clamp((int)floor((float)(y)),0,N-1)*M)  // indexing of 2d images
#endif

#define SS    9   // number of sidesteps, keeping on the track of maximum p



__kernel void Trace(const    int   NO,     // number of filaments
                    const    float PLIM,   // stay within P>PLIM
                    __global float *XY,    // pixel coordinates as start positions XY[NO, 2]
                    __global float *S,     // input image
                    __global float *P,     // probability, 2D image
                    __global float *T,     // position angle
                    __global int   *L,     // labels (int32 image)
                    __global float *RW,    // route waypoints R[NO, 1000, 2]
                    __global int   *NP) {  // number of points per filament
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
   NP[id] = no ;
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





__kernel void TraceWithDirection(const    int   NO,      // number of filaments (input)
                                 __global float *XY,     // pixel coordinates as start positions XY[NO, 2]
                                 __global float *P,      // probability, 2D image (input)
                                 __global float *T,      // position angles, 2D image (input)
                                 __global int   *L,      // labels, 2D image (input)
                                 __global float *RW,     // route waypoints R[NO, 1000, 3]  (output)
                                 __global int   *NP,     // number of points along the filament (output)
                                 __global float *POS) {  // buffer POS[NO, 100, 2] workspace 
   // Based on P image, trace the spine of a filement with the help of already computed
   // position anfle (image P) and return (x, y, p) along the ridge.
   // Mask the part of the filament already traversed (to avoid loops). 
   // If there are side filaments with the same label, as they are passed, push their 
   // start position to POS and trace them after the main filament.
   int   id = get_global_id(0) ;
   if   (id>=NO) return ;   // one work item per filament
   int   no=0, A=0, B=0, nsf=0 ;
   bool  ok ;
   float x=XY[2*id], y = XY[2*id+1], pa=T[I(x,y)], vx, vy, ux, uy, xx, yy, x0, y0, x1, y1, x2, y2 ;
   float DX  = STEP*0.2f ;                   // step along the filament [pixels]
   int   DXI = 1+(int)floor(DX) ;
   __global float *R   = &(RW[id*3000]) ;    // (x,y,p) for points along the filament
   __global float *pos = &(POS[id*200]) ;    // pos[100,2]
   // Initial step down in the map
   vx = DX*sin(-pa) ;    vy = DX*cos(pa) ;
   // go always down so the final trace will be in increasing order of declination
   if (vy>0.0f)  {  vx *= -1.0f ; vy *= -1.0f ; }
   // if (id==ID) printf("START  %8.3f %8.3f  dir %8.4f %8.4f\n", x, y, vx, vy) ;
   no = 0 ;
   while(1) {                                     // step to one end of the filament
      no += 1 ;
      x0 = x ; y0 = y ;
      x  += vx ;   y  += vy ;
      // if (id==ID) printf("DOW N  %8.3f %8.3f  dir %8.4f %8.4f\n", x, y, vx, vy) ;
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
   }      
   printf("ID %d  ... end found in %d steps\n", id, no) ;
   NP[id] = 0 ;  R[0] = -1.0f ; R[1] = -1.0f ; R[2] = -1.0f ;
   // return ;

   
   // now we are at one end of the filament, start in the opposite direction, hopefully increased dec order
   ux = -1.0f*vx ; uy = -1.0f*vy ;       // position (x,y), start in directin (vx, vy)
   pa =  T[I(x,y)] ;
   vx =  DX*sin(-pa) ;   vy =  DX*cos( pa) ;
   if ((vx*ux+vy*uy)<0.0f) {   vx *= -1.0f ;  vy *= -1.0f ;  }
   R[0] = x ; R[1] = y ; R[2] = P[I(x,y)] ; no = 1 ;        // add the first position in the filament route
   // if (id==ID) printf("SWITCH  %8.3f %8.3f    dir %8.4f %8.4f\n", x, y, vx, vy) ;   

   
   while(1) {                            // step to the other end of the filament
      x0 = x ;    y0 = y ;
      x += vx ;   y += vy ;
      pa = T[I(x,y)] ;                   // PA at the new position
      ux = DX*sin(-pa) ;   uy = DX*cos( pa) ;
      if ((vx*ux+vy*uy)<0.0f) {  ux = -ux ;  uy = -uy ;   }   // u || v
      vx = ux ;   vy = uy ;
      // Move sideways to the centre of the ridge = position of highest p
      x1 = x ;  y1 = y ;
      for(int k=-SS; k<=+SS; k++) {
         xx = x1-0.2f*k*vy ; yy = y1+0.2f*k*vx ;  // filament (vx, vy), perpendicular direction (-vy, vx)
         if (P[I(xx,yy)]>P[I(x,y)]) {             // replace (x,y) with (xx, yy)
            x = xx ;  y = yy ;
         }
      }
      
      if (L[I(x,y)]!=id) {  // (x,y) is already outside the filament region -- stop this trace
         ok = false ;
         while (nsf>0) {    // can take a new filament from buffer
            // we have stored start positions for some side filaments, pick the first one
            x  = pos[2*A] ;   y = pos[2*A+1] ;   nsf -= 1 ;   A = (A+1)%100 ;
            // we need to update (vx, vy)
#if 0
            pa = T[I(x,y)] ;                   // PA at the new position
            vx = DX*sin(-pa) ;   vy = DX*cos( pa) ;
            x1 = x+vx ;          y1 = y+vy ;
            ok = false ;
            if (L[I(x1,y1)]!=id) {          // perhaps (ux,uy) is step back to already masked old LI==id area
               vx = -vx ;  vy = -vy ;       // try a step in the opposite direction
               x1 = x+vx ;          y1 = y+vy ;            
            }       
            x = x1 ; y = y1 ;
            if (L[I(x,y)]==id) ok = true ;  // ok, found a valid start point and valid direction to follow
            printf("Took [%2d] %3.0f %3.0f => %d, L=%d\n", id, x, y, (int)ok, L[I(x,y)]) ;
#else
            // Find the direction with the largest probability and L still equal to id
            ux = 0.0f ;   // p
            uy = 0.0f ;   // phi
            ok = false ;
            for(float phi=0.0f; phi<(2.0f*PI); phi+=0.2f) {
               x1 = x + STEP*cos(phi) ;   y1 = y + STEP*sin(phi) ;
               if (P[I(x1,y1)]>ux) {
                  if (L[I(x1,y1)]==id) {
                     ux = P[I(x,y)] ; uy = phi ;  ok = true ;
                  }
               }
            }
            vx  = DX*cos(uy) ;    vy  = DX*sin(uy) ;
            vx *= DX ;            vy *= DX ;
            ux  = vx ;            uy  = vy ;
            x  += vx ;            y  += vy ;
#endif
            if (ok) break ;
         } // loop until no side filaments left or we have one that can be followed
         if (ok) continue ;   // continue tracinf this side filament
         break ;  // nothing more to do, break whil(1)
      }           
      R[3*no] = x ;    R[3*no+1] = y ;   R[3*no+2] = P[I(x,y)] ;  no += 1 ;
      
#if 1
      // We made the step (x0, y0) -> (x, y). Remove labels L==ID over a rectangle covering this step.
      // The rectangle extends distance STEP from the (x0, y0)->(x, y) line so normally will mask all of the 
      // filament. If L==ID still at that distance, push this coordinate to POS as potential crossing filament.
      ux  =  (x-x0)/DX ;   uy  =  (y-y0)/DX ;
      for(int j=-5; j<DXI-2; j++) {                     // along the step direction
         x1 = x0+j*ux ;    y1 = y0+j*uy ;               // along the step
         for(int i=-(int)STEP; i<(int)STEP; i++)  {     // the cross direction, DXI-1 steps in either direction
            x2 = x1-i*uy ;   y2 = y1+i*ux ;             // final pixel coordinates
            if (L[I(x2,y2)]==id) L[I(x2,y2)] = -2-id ;  // masked pixels come -2-ID
         }
      }
#endif
      
#if 1
      // Check the centre of the step, both left and right side at distance DXI+1 (to make sure this
      // pixel is really outside the masked rectangle). If that pixel has label ID, add that position 
      // to POS. DX is smaller than STEP so this will result in multiple entries in POS...
      x1 = x0+(DXI/2)*ux ;   y1 = y0+(DXI/2)*uy ;       // along the step
      x2 = x1-(-DXI-2)*uy ;  y2 = y1+(-DXI-2)*ux ;      // pixel just outside the rectangle
      if ((L[I(x2,y2)]==id)&&(nsf<99)) {                // add side filament to buffer
         printf("  [%3d]  push  %4.0f %4.0f\n", id, x2, y2) ;
         pos[nsf*2] = x2 ;    pos[nsf*2+1] = y2 ;  nsf += 1 ;  B = (B+1)%100 ;
      }
      x2 = x1-(DXI+2)*uy ;   y2 = y1+(DXI+2)*ux ;       // pixel just outside the rectangle
      if ((L[I(x2,y2)]==id)&&(nsf<99)) {
         printf("  [%3d]  push  %4.0f %4.0f\n", id, x2, y2) ;
         pos[nsf*2] = x2 ;    pos[nsf*2+1] = y2 ;  nsf += 1 ;  B = (B+1)%100 ;
      }
#endif
      if (no>999) break ;  // there could be a loop !!
      
   } // while (1)
   
   R[3*no] = -1.0f ;  R[3*no+1] = -1.0f ;  R[3*no+2] = -1.0f ;
   NP[id] = no ;   // number of points along the filament
   printf("TraceWithDirection:  filament %4d has %4d points\n", id, no) ;
}








__kernel void TraceWithProbability(const    int    NO,     // number of filaments, IDS[NO] not necessarily in order
                                   __global float *XY,     // pixel coordinates as start positions XY[NO, 2]
                                   __global float *P,      // probability, 2D image (input)
                                   __global float *T,      // position angles, 2D image (input)
                                   __global int   *L,      // labels (int32 image)
                                   __global float *RW,     // route waypoints R[NO, 1000, 3]
                                   __global int   *NP,     // number of points along the filament
                                   __global float *POS) {  // buffer POS[NO, 100, 3] to store start positions if sidef.
   // Instead of following the filament based on the local estimated position angle, 
   // do a full scan in angle for filaments or side filaments, follow the one closest to the original direction and
   // put the others to buffer. One does not use position angles (P) at all (except that it is still used for the
   // initial stepping to one end of the main filament). Only P and L are used for the actual trace.
   int   id = get_global_id(0) ;
   if   (id>=NO) return ;
   int   no=0, A=0, B=0, nsf=0, imax ;
   bool  ok ;
   float pro[NDIR] ;
   float x=XY[2*id], y = XY[2*id+1], pa=T[I(x,y)], vx, vy, ux, uy, xx, yy, x0, y0, x1, y1, x2, y2, phi, pmax ;
   __global float *R   = &(RW[id*3000]) ;         // (x,y) pixel coordinates for this filament to be saved
   __global float *pos = &(POS[id*300]) ;         // pos[100,3]
   // Initial step down in the map
   vx = sin(-pa) ;    vy = cos(pa) ;
   if (vy>0.0f)  {  vx *= -1.0f ; vy *= -1.0f ; }
   while(1) {                                     // loop in one direction till the end of the filament
      x0 = x ; y0 = y ;
      x  += 0.5f*STEP*vx ;   y  += 0.5f*STEP*vy ;
      // if (id==ID) printf("DOW N  %8.3f %8.3f  dir %8.4f %8.4f\n", x, y, vx, vy) ;
      pa = T[I(x,y)] ;                            // PA at the new position
      ux = sin(-pa) ; 
      uy = cos( pa) ;                             // vector for direction forward from the new position
      if ((vx*ux+vy*uy)<0.0f) {                   // if v and u at more than 90 degrees apart, flip u
         ux = -ux ;  uy = -uy ;
      }
      vx = ux ;   vy = uy ;                       // step completed, new position (x,y) with direction (vx, vy)
      // Move sideways to the centre of the ridge
      x1 = x ; y1 = y ;
      for(int k=-SS; k<=+SS; k++) {               // at most SS sidesteps either way
         xx = x1-k*uy ;  yy = y1+k*ux ;           // along perpendicular direction (-vy, +vx)
         if (P[I(xx,yy)]>P[I(x,y)]) {             // replace (x,y) with (xx, yy)
            x = xx ;  y = yy ; 
         }
      }
      if (L[I(x,y)]!=id) {
         x  = x0 ;   y = y0 ;   break ;           // we are at one end of the filament, drop the last step
      }      
      pa = T[I(x,y)] ;   vx = sin(-pa) ;     vy = cos( pa) ;
      if ((vx*ux+vy*uy)<0.0f) {  vx *= -1.0f ;  vy *= -1.0f ;  }  // u || v
   }      
   // now we are at one end of the filament (x, y), starting to move back to the direction (vx, vy)
   vx = -ux ;  vy = -uy ;      // turn back
   const float dphi = 2.0f*PI/NDIR ;
   
   while(1) {                            // step to the other end of the filament
      ux = vx ;  uy = vy ;   x0 = x ;  y0 = y ;
      // Test NDIR directions for the step
      for(int i=0; i<NDIR; i++) {
         vx = sin(i*dphi) ;    vy = cos(i*dphi) ;         // i*dphi != pa
         x1 = x + STEP*vx ;    y1 = y + STEP*vy ;         // test this step in direction i
         if (L[I(x1, y1)]==id) {
            pro[i] = P[I(x1,y1)] + 1.0f + (vx*ux+vy*uy) ; // probability, with preference for forward direction
            if ((vx*ux+vy*uy)<-0.7f) pro[i] = 0.0f ;      // do not turn back to the direction that we are coming from
         } else {
            pro[i] = 0.0f ;
         }
      }
      // locate local maxima in different directions, pro = probability
      pmax = -1.0f ;  imax = -1 ;
      for(int i=0; i<NDIR; i++) {
         if ((pro[i]<pro[(i+1)%NDIR])||(pro[i]<pro[(i-1+NDIR)%NDIR])) {
            pro[i] = 0.0f ;    // not local maximum => ignore
         }
         if (pro[i]>pmax) {  pmax = pro[i] ;  imax = i ; }
      }
      if (pmax<=0.0f) {      // no valid steps anymore
         if (nsf<1) break ;  // all done, unless one can restart with a position from the buffer
         x  = pos[3*A] ;   y  = pos[3*A+1] ;   pa = pos[3*A+2] ;   A = (A+1)%100 ;   nsf -= 1 ;
         vx = sin(pa) ;    vy = cos(pa) ;
         x0 = x ;          y0 = y ;
      } else {
         // check all maxima in pro
         for(int i=0; i<NDIR; i++) {
            pa = i*dphi ;
            if (i==imax) {                      // maximum probability =  will be followed next
               vx = sin(pa) ;   vy = cos(pa) ; 
            } else {             
               if ((pro[i]>0.0f)&&(nsf<99)) {   // other maxima will be pushed to buffer
                  x1 = x + STEP*sin(pa) ;    y1 = y + STEP*cos(pa) ;
                  if (id==2) printf("  side  %5.1f %5.1f   %5.1f %5.1f\n", x, y, x1, y1) ;
                  pos[3*B] = x1 ;  pos[3*B+1] = y1 ;  pos[3*B+2] = pa ;   B = (B+1)%100 ;  nsf += 1 ;
               }
            }
         }
         x += STEP*vx ;   y += STEP*vy ;            // the next position, already known to have L==id
      }
      // we have updated x, y, vx, and vy
      R[3*no] = x ;     R[3*no+1] = y ;     R[3*no+2] = P[I(x,y)] ;   no += 1 ; 
      // We made the step (x0, y0) -> (x, y). Remove labels L==ID over a rectangle covering this step.
      // The rectangle extends distance STEP from the (x0, y0)->(x, y) line so normally will mask all of the 
      // filament. If L==ID still at that distance, push this coordinate to POS as potential crossing filament.
      // Perhaps it is enough to mask a narrrow region that will still prevent loops
      ux  =  0.8f*(x-x0)/STEP ;   uy  =  0.8f*(y-y0)/STEP ;
      for(int j=(int)(-1-0.4f*STEP); j<(int)(1+1.2f*STEP); j++) {             // along the step direction
         x1 = x0+j*ux ;    y1 = y0+j*uy ;                // along the step
         for(int i=-(int)(STEP); i<(int)(STEP); i++)  {  // the cross direction
            x2 = x1-i*uy ;   y2 = y1+i*ux ;              // final pixel coordinates
            if (L[I(x2,y2)]==id) L[I(x2,y2)] = -2-id ;   // masked pixels come -2-ID
         }
      }
      
   } // while(1)
   
   printf("TraceWithProbability:  filament %4d has %4d points\n", id, no) ;
   R[3*no] = -1.0f ;  R[3*no+1] = -1.0f ;  R[3*no+2] = -1.0f ;
   NP[id] = no ;   // number of points along the filament
   
}




