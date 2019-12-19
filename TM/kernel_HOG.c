

__kernel void Gradient(__global float  *S,
                       __global float  *Gx,
                       __global float  *Gy) {
   int id = get_global_id(0) ;
   if (id>=(N*M)) return ;
   int   d = (int)(5.0f*SIGMA) ;           // convolution extends to >2*FHWHM from beam centre
   int   ind, i0=id % M,  j0 = id / M ;
   float dx, dy ;
   // float a = 1.0f/(2.0f*pi*SIGMA*SIGMA*SIGMA*SIGMA) ;  // normalisation --- now through weight sum
   const float b = 1.0f/(2.0f*SIGMA*SIGMA) ;  // scale in exponent
   float sumX=0.0f, sumY=0.0f, wei=0.0f, w ;
   // Gaussian is           exp(-r^2*b) / [ 2*pi*SIGMA^2 ]
   // Derivative is     x * exp(-r^2*b) / [ 2*pi*SIGMA^4 ]
   // We calculate sums of  exp(-r^2*b)        *=  1 / [2*pi*SIGMA^2]
   //             vs.   x * exp(-r^2*b)        *=  1 / [2*pi*SIGMA^4]
   // => normalise derivative sum with the sum of Gaussian weights / SIGMA^2
   for(int j=max(0, j0-d); j<min(N, j0+d+1); j++) {
      for(int i=max(0, i0-d); i<min(M, i0+d+1); i++) {
         ind    =  j*M+i ; 
         dx     =  i-i0 ;
         dy     =  j-j0 ;
         w      =  exp(-b*(dx*dx + dy*dy)) ;
         wei   +=  w ;
         sumX  +=  S[ind] * ( - dx * w ) ;
         sumY  +=  S[ind] * ( - dy * w ) ;         
      }
   }
   sumX    /=  (wei*SIGMA*SIGMA) ;
   sumY    /=  (wei*SIGMA*SIGMA) ;
   Gx[id]   =  sumX ;   // return gradient as a vector
   Gy[id]   =  sumY ;
}




__kernel void Sums(__global float *Gx,      // first plane, d/dx         [N*M]
                   __global float *Gy,      // first plane, d/dy         [N*M]
                   __global float *Hx,      // second plane, d/dx        [N*M]
                   __global float *Hy,      // second plane, d/dy        [N*M]
                   __global float *Csum,    // sum of W*cos(2*phi)       [global]
                   __global float *Ssum,    // sum of W*sin(2*phi)       [global]
                   __global float *Wsum,    // sum of W                  [global]
                   __global float *W2sum,   // sum of W^2                [global]
                   __global float *weights  
                  ) {
   // Calculate only sums Csun, Ssum, Wsum, W2sum -- host uses these to compute Vlm and rlm
   int id = get_global_id(0) ;
   int gs = get_global_size(0) ;
   float nom, den, phi ;
   int   ind, i0=id % M,  j0 = id / M ;
   float Cs=0.0f, Ss=0.0f, Ws=0.0f, W2s=0.0f ;
   // weight W is constant ?  ---   ~  (pix/SIGMA)^2
   float W =  1.0f/(SIGMA*SIGMA) ;
   for(int ind=id; ind<(N*M); ind+=gs) {
      nom   =  Gx[ind]*Hy[ind] - Gy[ind]*Hx[ind] ;
      den   =  Gx[ind]*Hx[ind] + Gy[ind]*Hy[ind] ;
      phi   =  atan2(nom, den) ;
      Cs   +=  W*cos(2.0f*phi) ;
      Ss   +=  W*sin(2.0f*phi) ;
      Ws   +=  W   ;
      W2s  +=  W*W ;
   }
   Csum[id]   =  Cs ;  
   Ssum[id]   =  Ss ;
   Wsum[id]   =  Ws ;
   W2sum[id]  =  W2s ;
}






__kernel void HOG_gradients(__global float *I1,    // First image [NY, NX]
                            __global float *I2,    // Second image [NY, NX]
                            __global float *phi,   // angle for the gradient direction
                            const float gthresh1,  // gradthresh1
                            const float gthresh2   // gradthresh2
                           ) {
   // Calculate gradients for the images I1 and I2, return array for the relative angle phi.
   // Elements where the gradient is below given gradient thresholds are set to NaN.
   int   id  =  get_global_id(0) ;
   int   gs  =  get_global_size(0) ;   
   if   (id>=(N*M)) return ;                 // image dimensions [N,M]
   int   d   = (int)(5.0f*SIGMA) ;           // maximum distance to which convolution extends
   int   ind, i0=id % M,  j0 = id / M ;      // indices of the current pixel
   float dx, dy ;
   float b   =  1.0f/(2.0f*SIGMA*SIGMA) ;    // scale in exponent
   float GX1 =  0.0f, GY1=0.0f, wei=0.0f, w ;
   float GX2 =  0.0f, GY2=0.0f ;
   const float NaN = nan((uint)0) ;
   //  f      =    exp(-0.5*(x^2+y^2)/s^2) / (2*pi*s^2)
   //  df/dx  =    -0.5*2*x/s^2 * exp(-0.5*(x^2+y^2)/s^2) /  (2*pi*s^2)
   //         =    -x   (1/s^2)  *  f
   //  wei    ~    (s*pi*s^2)  --> normalisation is division by (wei*s^2)
   for(int j=j0-d; j<=j0+d; j++) {
      for(int i=i0-d; i<=i0+d; i++) {
         ind     =  clamp(j, 0, N-1)*M+clamp(i, 0, M-1) ; // implements "nearest"
         dx      =  i-i0 ;
         dy      =  j-j0 ;
         w       =  exp(-b*(dx*dx + dy*dy)) ;
         wei    +=  w ;
         GX1    +=  I1[ind] * ( - dx * w ) ;   // local gradient in the first image
         GY1    +=  I1[ind] * ( - dy * w ) ;         
         GX2    +=  I2[ind] * ( - dx * w ) ;   // local gradient in the second image
         GY2    +=  I2[ind] * ( - dy * w ) ;         
      }
   }
   dx       =   wei*SIGMA*SIGMA ;
   GX1     /=   dx ;
   GY1     /=   dx ;
   GX2     /=   dx ;
   GY2     /=   dx ;
   w        =  atan2(GX1*GY2-GY1*GX2, GX1*GX2+GY1*GY2) ;  // phi !!
   w        =  atan(tan(w)) ;
   dx       =  sqrt(GX1*GX1+GY1*GY1) ;
   dy       =  sqrt(GX2*GX2+GY2*GY2) ;
   phi[id] =  ((dx<=gthresh1)||(dy<=gthresh2)) ?  NaN :  w ;
}




__kernel void HOG_sums(__global float *phi, 
                       __global float *scos,
                       __global float *ssin,
                       __global float *scos2,
                       __global float *ssin2,
                       __global float *weights,
                       __global float *wscos,
                       __global float *wssin,
                       __global float *wscos2,
                       __global float *wssin2
                      ) {
   int id = get_global_id(0) ;
   int gs = get_global_size(0) ;
   // Compute in kernel partial sums of cos, sin, cos^2, sin^2. Host calculates the final sums.
   float sc=0.0f, ss=0.0f, sc2=0.0f, ss2=0.0f, wsc=0.0f, wss=0.0f, wsc2=0.0f, wss2=0.0f, p, w, s, c ;
   for(int i=id; i<(N*M); i+=gs) {
      p = phi[i] ;  w = weights[i] ; s = sin(p) ; c = cos(p) ;
      if (isfinite(p)) {
         // NaN pixels contribute nothing to the sums
         // ... alternatively one could include all masks in the weight vector
         sc    +=  c ;
         ss    +=  s ;
         sc2   +=  c*c ;
         ss2   +=  s*s ;
         wsc   +=  w * c ;
         wss   +=  w * s ; 
         wsc2  +=  w * c*c ;
         wss2  +=  w * s*s ;
      }
   }
   scos[id]    =  sc ;     // kernel computes these partial sums
   ssin[id]    =  ss ;     // ... host calculates the final total sums
   scos2[id]   =  sc2 ;
   ssin2[id]   =  ss2 ;
   wscos[id]   =  wsc ;
   wssin[id]   =  wss ;
   wscos2[id]  =  wsc2 ;
   wssin2[id]  =  wss2 ;
}





