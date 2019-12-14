
// #define ID (152*256+8)


float PixelCrossSection(float *Xin, float *Yin) {
   // Return the cross section of the target pixel x=0.0-1.0, y=0.0-1.0
   // and the input pixel that has corners at (Xin, Yin)
   const int id = get_global_id(0) ;
   float a, b, c, d, xmin=1e10f, xmax=-1e10f, alpha, beta, area, x, y, xa, xb, ya, yb, dx, dy ;
   float XX[8], YY[8] ;  // we believe we need at most eight x-axis points
   int   inside=0, crossings=0, j, NP=0 ;
   // first check the x-axis limits of the integration, limits are [0.0, 1.0] or min/max of the input pixel corners
   for (int i=0; i<4; i++) {
      xmin = min(xmin, Xin[i]) ;
      xmax = max(xmax, Xin[i]) ;
   }
   //  corners of pix II and all crossings between pix II and pix I
   if ((xmin>1.0f)||(xmax<0.0f))  return 0.0f ; // no intersection possible
   xmin  = max(0.0f, xmin) ;    xmax  = min(1.0f, xmax) ;
   XX[0] = xmin ;   XX[1] = xmax ;   NP = 2 ;
   // for the moment, use area to store the area of the input pixel II ... could be approximately constant?
   a   =  Xin[1]-Xin[0] ;          //  dx for size 0
   b   =  Yin[1]-Yin[0] ;          //  dy for side 0
   c   =  Xin[3]-Xin[0] ;          //  dx for side 3
   d   =  Yin[3]-Yin[0] ;          //  dy for side 3
   xa  =  sqrt(a*a+b*b) ;          //  length of the first side
   xb  =  sqrt(c*c+d*d) ;          //  length of the second side
   x   =  (a*c+b*d)/(xa*xb) ;      //  cosine of the angle between the two sides
   area  =  xa*xb*sqrt(1.0f-x*x) ; //  pixel area   
   inside    = 0 ;                 //  number of input pixel corners that are inside the target pixel
   crossings = 0 ;                 //  number of crossings with the top and bottom of the target pixel
   for(int i=0; i<4; i++) {        //  loop over the sides of the input pixel
      x  =  Xin[i] ;   y = Yin[i] ;
      // check if the corner i is inside the target pixel
      if ((x>=0.0f)&&(x<=1.0f)&&(y>=0.0f)&&(y<=1.0f))  inside += 1 ;
      // check if the side i -> i+1 crosses an edge of the target pixel
      a     =  Xin[(i+1)%4]-x ;       
      c     =  Yin[(i+1)%4]-y ;              // (a,c) is the vector along the edge i of the input pixel
      if (fabs(c)>1.0e-6f) {                 // y is not constant => check for crossing with the top and bottom borders
         alpha =  (0.0f-y)/c ;
         if ((alpha>=0.0f)&&(alpha<=1.0f)) { // the side does cross correct y=0 lines
            b = x+alpha*a  ;                 // x-coordinate of the crossing
            if ((b>=0.0f)&&(b<=1.0f)) {      // crossing inside the target pixel
               XX[NP]  =  b ;    NP++ ;  crossings += 1 ;
               // if (id==ID) printf("(1) CROSS AT %7.2f\n", x+alpha*a) ;
            } 
         }
         alpha =   (1.0f-y)/c ;
         if ((alpha>=0.0f)&&(alpha<=1.0f)) { // does cross with the top of the target pixel
            b = x+alpha*a ;
            if ((b>=0.0f)&&(b<=1.0f)) {
               XX[NP]  =  b  ;  NP++ ;     crossings += 1 ;
               // if (id==ID) printf("(2) CROSS AT %7.2f\n", x+alpha*a) ;
            }
         } 
      }
      // check also the sides, just to count the total number of crossings, not added to XX[]
      if (fabs(a)>1e-6f) {
         alpha =  (0.0f-x)/a ;
         if ((alpha>0.0)&&(alpha<1.0))  { // side crosses the x=0.0 lines
            b =  y+alpha*c ;
            if ((b>=0.0f)&&(b<=1.0f))  crossings++ ;  // that crossing inside the target pixel
         }
         alpha =  (1.0f-x)/a ;
         if ((alpha>=0.0)&&(alpha<=1.0))  { // side crosses the x=1.0 line
            b = y+alpha*c ;
            if ((b>=0.0f)&&(b<=1.0f))  crossings++ ;
         }
      }
   } // for i --- loop over II sides
   if (inside==4) return area ;    //  input pixel is completely inside the target pixel
   if ((inside==0)&&(crossings==0)) {  //  no common area... or target pixel is completely inside the input pixel
      x      =  0.0f    ;          //  pick one corner of the target pixel
      y      =  0.0f    ;          
      a      =  Xin[1]-Xin[0] ;    //  dx for side 0   (...basis vectors definining the input pixel)
      b      =  Xin[3]-Xin[0] ;    //  dx for side 3
      c      =  Yin[1]-Yin[0] ;    //  dy for side 0
      d      =  Yin[3]-Yin[0] ;    //  dy for side 2
      //  (x,y) = (Xin[0],Yin[0]) + alpha*(dx0,dy0) + beta*(dx3, dy3)  
      //  if both alpha and beta in [0,1], corner (x,y) was inside the input pixel => all corners inside input pixel
      beta   =  ( x*c - y*a - Xin[0]*c + Yin[0]*a )   /  ( b*c-a*d ) ;
      alpha  =  ( x - Xin[0] - beta*b ) / a ;
      if ((alpha>0.0f)&&(alpha<1.0f)&&(beta>0.0f)&&(beta<1.0f)) {
         return 1.0f ;    // target pixel completely inside the input pixel => return the full area of the target pixel
      } else {
         return 0.0f ;    // no crossings between the edges and all corners outside input pixel => no overlap
      }
   }   
   // We end up integrating the area based on a set of (x, y) points. We have in XX already:
   //     -  0.0 and 1.0, or the min and max of the input pixel x-values
   //     -  the crossings of pix II with the top and bottom of the target pixel I
   // Add still the corners of the input pixel if they are inside the target pixel, in ]xmin, xmax[
   for(int i=0; i<4; i++) {
      if ((Xin[i]>xmin)&&(Xin[i]<xmax)&&(Yin[i]>0.0f)&&(Yin[i]<1.0f)) {
         XX[NP] = Xin[i] ;    NP++ ;
         // if (id==ID) printf("(3) add corner %7.2f\n", Xin[i]) ;
      }
   }   
   // All vertex points are now in XX but they need to be still sorted
   for(int i=1; i<NP; i++) {
      a  = XX[i] ;
      j  = i-1 ;
      while((j>=0)&&(XX[j]>a)) {
         XX[j+1] = XX[j] ;
         j-- ;
      }
      XX[j+1] = a ;
   }      

#if 0   
   if (id==ID) {
      printf("A===========================================================================================\n") ;
      for(int i=0; i<4; i++) printf("#1   %7.4f %7.4f\n",  Xin[i], Yin[i]) ;    
      for(int i=0; i<NP; i++) printf(" %7.4f ", XX[i]) ;
      printf("\n") ;
   }
#endif
   
   // Calculate the y-values for each XX point
   for(int i=0; i<NP; i++) {          // for each XX[i], find the y-values at input pixel boundaries
      a = 1.0e10f ; b = -1.0e10f ;    // reused for minimum and maximum y values
      for(int II=0; II<4; II++) {     // loop over the sides of the input pixel
         xa  =  Xin[II] ;
         ya  =  Yin[II] ;
         dx  =  Xin[(II+1)%4]-xa ;
         dy  =  Yin[(II+1)%4]-ya ;
         if (fabs(dx)<5.0e-6f) continue ;  // skip vertical sides, there will be two other horizontal lines?
         alpha =  (XX[i]-xa)/dx ;          // relative distance from the beginning of the side
         // if (id==ID) printf(" *** alpha *** %.3f\n", alpha) ;
         if ((alpha>=-0.00001f)&&(alpha<=1.00001f)) { // this side extends over XX[i]
            c =  ya + alpha*dy ;             // y value at the crossing
            b =  max(b, c) ;                 // maximum y value for the input pixel at position XX[i]
            a =  min(a, c) ;                 // minimum y value
#if 0
            if (id==ID) {
               printf(" === X = %7.4f - %7.4f,    Y = %7.4f - %7.4f\n", xa, xa+dx, ya, ya+dy) ;
               printf(" === a = %7.4f  <=  c = %7.4f = %7.4f + %7.4f x %7.4f\n", a, c, ya, alpha, dy) ;
            }
#endif
         }
      }
      b = min(1.0f, b) ;    a = max(0.0f, a) ; // min of maxs and max of mins
      YY[i] = clamp(b-a, 0.0f, 1.0e10f) ;
   }

#if 0
   if (id==ID) {
      for(int i=0; i<NP; i++) {
         printf("                 XX = %.3f       YY = %.3f\n", XX[i], YY[i]) ;
      }
      printf("O===========================================================================================\n") ;
   }
#endif
   
   // Integrate the area, cross section between the pixels
   area = 0.0f ;
   for(int i=1; i<NP; i++) {
      area += (XX[i]-XX[i-1])*0.5f*(YY[i]+YY[i-1]) ;
      // if (id==ID)  printf("#2  %6d   %7.4f %7.4f   %7.4f %7.4f    area + %7.4f = %7.4f\n\n", id, XX[i-1], YY[i-1], XX[i], YY[i], (XX[i]-XX[i-1])*0.5f*(YY[i]+YY[i-1]), area) ;
   }
   return area ;
}




__kernel void Sampler(__global float  *Xin,   //  Xin[ N, M]  X-coordinates of input image pixels
                      __global float  *Yin,   //  Yin[ N, M]  y-coordinates of input image pixels
                      __global float  *S,     //    S[ N, M]  input image
                      __global float  *SS) {  //   SS[NN,MM]  output image
   // Xin, Yin are the coordinates for the centre of each input image pixel
   const int id = get_global_id(0);
   if (id>=(NN*MM)) return ;    // we use one work item for each target image pixel
   int I = id % MM ;            // target image dimensions (NN, MM)
   int J = id / MM ;            // => work item will compute the value of SS[J,I]
   // find position (i0,j0) in the input image that corresponds to the output image pixel (J, I)
   int   i0=M/2, j0=N/2, k, l ;
   float x, y, ix, iy, jx, jy ;
   __private float X[4], Y[4] ;  // corners of an input pixel, in pixel coordinates of the output image
   // (I,J) correspond to the centre of the target pixel => we need (i0,j0) that map to that pixel centre
   for(int ii=0; ii<1; ii++) {  // some number of iterations (in case of nonlinear pixel coordinates)
      // at the current location (i0,j0) of the input image, compute derivatives ix, iy, jx, jy --  ix = dX/di etc.
      i0   =   clamp(i0, 0, M-2) ;        j0   =   clamp(j0, 0, N-2) ;
      // for any valid pixel (i0,j0) we are guaranteed to have coordinate points i0/STEP and i0/STEP+1 !
      i0   =   i0/STEP ;                     // a nearby coordinate grid position
      j0   =   j0/STEP ;
      k    =   i0 + j0*Mc  ;                 // index to coordinate grid point
      x    =   Xin[k] ;                      // coordinate grid position, X-coordinate
      y    =   Yin[k] ;                      //                           Y-coordinate
      ix   =   (Xin[k+1 ]-x)/STEP ;          // dX/di  ... distance between coordinate grid positions = STEP pixels
      jx   =   (Xin[k+Mc]-x)/STEP ;          // dX/dj
      iy   =   (Yin[k+1 ]-y)/STEP ;          // dY/di
      jy   =   (Yin[k+Mc]-y)/STEP ;          // dY/dj
      // add correction based on the difference between (I,J) and the coordinate grid position
      j0   =   j0*STEP + ((I-x)*iy  - (J-y)*ix) / (jx*iy-ix*jy) ;
      i0   =   i0*STEP + ((I-x)*jy  - (J-y)*jx) / (ix*jy-jx*iy) ;
   }   
   if ((i0<0)||(i0>=M)||(j0<0)||(j0>=N)) { 
      SS[id] = 0.0f ; // no input data
      return ;
   }
   // figure out the size of the input pixel  -- TODO: we assume square input pixels
   k  =   (i0/STEP) + (j0/STEP)*Mc ;
   x  =   Yin[k+1]  -  Yin[k] ;
   y  =   Xin[k+1]  -  Xin[k] ;
   x  =   sqrt(x*x+y*y)/STEP ; 
   int delta =  (int)(2.1f+0.5f/x) ;   // number of steps in the input image ... x is still the pixel size
   // Calculate area-weighted average of the input pixels
   float W=0.0f, SUM=0.0f ;   
   for(int j=max(0, j0-delta); j<=min(N-1, j0+delta); j++) {  // loop over pixels of the input image
      for(int i=max(0, i0-delta); i<=min(M-1, i0+delta); i++) {
         // centre of the input pixel (i, j)
         x     =  Xin[(i/STEP)+Mc*(j/STEP)] + (i%STEP)*ix + (j%STEP)*jx  ;
         y     =  Yin[(i/STEP)+Mc*(j/STEP)] + (i%STEP)*iy + (j%STEP)*jy  ;
         // corners of the input pixel (i, j)  ---- relative to the lower left corner of the target pixel
         // TODO: target pixel is assumed to be square
#if 0
         // we can probably use the derivatives estimated at target pixel centre for all the source
         // pixels that overlap with that pixel... unless the target pixel is huge!!
         k     =  clamp(i, 0, M-2) ;
         ix    =  Xin[k+1 +  j   *M] - Xin[k + j*M] ;  //  dx/di
         iy    =  Yin[k+1 +  j   *M] - Yin[k + j*M] ;  //  dy/di
         k     =  clamp(j, 0, N-2) ;
         jx    =  Xin[i   + (k+1)*M] - Xin[i + k*M] ;  //  dx/dj
         jy    =  Yin[i   + (k+1)*M] - Yin[i + k*M] ;  //  dy/dj
#endif
         

#define QQ (0.5f*Q)  
         X[0]  =  x - QQ*ix - QQ*jx - (I-0.5f) ;    // lower left  corner, relative to target pixel corner
         X[1]  =  x + QQ*ix - QQ*jx - (I-0.5f) ;    // lower right corner
         X[2]  =  x + QQ*ix + QQ*jx - (I-0.5f) ;    // upper right corner
         X[3]  =  x - QQ*ix + QQ*jx - (I-0.5f) ;    // upper left  corner
         Y[0]  =  y - QQ*iy - QQ*jy - (J-0.5f) ;
         Y[1]  =  y + QQ*iy - QQ*jy - (J-0.5f) ;
         Y[2]  =  y + QQ*iy + QQ*jy - (J-0.5f) ;
         Y[3]  =  y - QQ*iy + QQ*jy - (J-0.5f) ;
         x     =  PixelCrossSection(X, Y) ;
         W    +=  x ;
         SUM  +=  x*S[i+j*M] ;
      }
   }
   SS[id] = (W>0.0f) ? (SUM/W) : 0.0f ;   
   return ;
}

