
// library grid is [N,N,N] points, as defined by I0, dI0, I1, dI1, D2, dI2
// reference frequencies are channels R1, R2, and R3 in ABS


void __kernel  LibrarySolve(const int       no,      // number of cells to solve
                            const float     I0,      // I0, first bin log-intensity at first reference frequency
                            const float     dI0,     // dI0, step in lg-intensity for the first reference frequency
                            __global float *I1,      // I1, first-bin lg-intensity for the second reference frequency
                            __global float *dI1,     // step in I1
                            __global float *I2,      // I2[i,j], first-bin lg-intensity for the third reference frequency
                            __global float *dI2,     // dI2[i,j], step in lg-intensity for the third reference frequency
                            __global float *X,       // library grid, lg-intensity for first reference frequency [N,N,N]
                            __global float *Y,       // .... second
                            __global float *Z,       // ...  third frequency
                            __global float *E,       // library emission vectors [N,N,N,NFREQ]
                            __global float *ABS,     // input, absorption values [no, nfreq]
                            __global float *EMI) {   // output,  emission values [no, NFREQ],   no <= BATCH
   const int id  = get_global_id(0) ;      // index to current BATCH
   const int lid = get_local_id(0) ;
   if (id>=no) return ;
   float x, y, z, w, W ;
   int   i, j, k ;
   // NFREQ = number of frequencies in EMI == always the full set of frequencies (as in freq.dat)
   // ABS[] is always only three frequencies = the reference frequencies
   
#if (METHOD==0)  // no interpolation, only the closest grid position
   // indexing of the library cube is based on log10() of absorptions, first bin I0, bin width dI0 etc.
   x =  (log10(clamp(ABS[id*3+0], 1.0e-29f, 1.0e10f)) - I0        ) / dI0  ;    // index based on the reference intensity 
   i =  clamp((int)round(x), 0, N-1) ;
   y =  (log10(clamp(ABS[id*3+1], 1.0e-29f, 1.0e10f)) - I1[i]     ) / dI1[i]  ;
   j =  clamp((int)round(y), 0, N-1) ;
   z =  (log10(clamp(ABS[id*3+2], 1.0e-29f, 1.0e10f)) - I2[i*N+j] ) / dI2[i*N+j]  ;  // I2[i,j] = I2[N*i+j] !!
   k =  clamp((int)round(z), 0, N-1) ;   
   // It is possible that X[i,j,k] is far from x, Y[i,j,k] far from y, or Z[i,j,k] far from z,
   // especially because we clamp the indices (i, j, k)   ==>   flag these as "missing library data"
   if ((fabs(x-X[k+N*(j+N*i)])>1.1f)||(fabs(y-Y[k+N*(j+N*i)])>1.1f)||(fabs(z-Z[k+N*(j+N*i)])>1.1f)) {
      EMI[id*NFREQ] = 1.0e32f ;   // this stands for missing data
# if 1
      if (id==0) {
         printf("KERNEL MISSING %10.3e %10.3e %10.3e <> %10.3e %10.3e %10.3e == %2d %2d %2d\n", x, y, z, 
                X[k+N*(j+N*i)], Y[k+N*(j+N*i)], Z[k+N*(j+N*i)], i, j, k) ;
         printf("               %10.3e %10.3e  %10.3e %10.3e  %10.3e %10.3e\n", I0, dI0, I1[i], dI1[i], I2[i*N+j], dI2[i*N+j]) ;
      }
# endif
      return ;
   }   
   // Otherwise we can just pick the emission vector EMI[i,j,k,:]
   for(int s=0; s<NFREQ; s++) {
      EMI[id*NFREQ+s] = E[NFREQ*(k+N*(j+N*i))+s] ;
   }   
#else    // average value over possibly several nearby grid values
   __local float BUF[LOCAL*NFREQ] ;        // for averages of emission vectors
   __local float *buf = &(BUF[lid*NFREQ]) ;
   x =  (log10(clamp(ABS[id*3+0], 1.0e-25f, 1.0f))-I0       )/dI0  ;    // index based on the reference intensity 
   i =  clamp((int)round(x), 0, N-1) ;   
   y =  (log10(clamp(ABS[id*3+1], 1.0e-25f, 1.0f))-I1[i]    )/dI1[i]  ;
   j =  clamp((int)round(y), 0, N-1) ;   
   z =  (log10(clamp(ABS[id*3+2], 1.0e-25f, 1.0f))-I2[i*N+j])/dI2[i*N+j]  ;  // I2[i,j] = I2[N*i+j] !!
   k =  clamp((int)round(z), 0, N-1) ;      
   W = 0.0f ;  for(int s=0; s<NFREQ; s++) buf[s] = 0.0f ;
   for(int ii=max(0, i-1); ii<(N, i+2); ii++) {
      for(int jj=max(0, j-1); jj<(N, j+2); jj++) {
         for(int kk=max(0, k-1); kk<(N, k+2); kk++) {
            if ((fabs(x-X[kk+N*(jj+N*ii)])>1.0f)||(fabs(y-Y[kk+N*(jj+N*ii)])>1.0f)||(fabs(z-Z[kk+N*(jj+N*ii)])>1.0f)) {
               continue ;  // the grid position (ii,jj,kk) is far from the intensity (x,y,z)
            }
            if (E[NFREQ*(kk+N*(jj+N*ii))]>1.0e31f) continue ;  // library is missing this emission vector
            w      =  1.0f/(0.1f+pow(x-X[kk+N*(jj+N*ii)], 2.0f)) ;
            w     *=  1.0f/(0.1f+pow(y-Y[kk+N*(jj+N*ii)], 2.0f)) ;
            w     *=  1.0f/(0.1f+pow(z-Z[kk+N*(jj+N*ii)], 2.0f)) ;
            for(int s=0; s<NFREQ; s++)  buf[s]  +=  w*E[NFREQ*(kk+N*(jj+N*ii))+s] ;
            W     +=  w ;
         }
      }
   }
   if (W>0.0f) {
      for(int s=0; s<NFREQ; s++) {
         EMI[id*NFREQ+s] = buf[s] / W ;
      }
   } else {
      EMI[id*NFREQ] = 1.0e32f ;
   }   
#endif
}







