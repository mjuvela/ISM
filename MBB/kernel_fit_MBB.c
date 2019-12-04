
// Note --- reference wavelength is 250 um   =>  f0 =  1.1992e+12 Hz
// h/k       =  4.7995074e-11       -->     4.799243348e-11
// h/k * f0  =  57.55424482540757   -->     57.551078393482776

float MBB(float f, float t, float b) {
   // printf("*** %12.4e *** %12.4e ***\n", t, b) ;
   return pow(f/1.1991698e+12f, 3.0f+b) * (exp(57.551078f/t)-1.0f) / (exp(4.799243348e-11f*f/t)-1.0f) ;
}





#if (DO_CC<1)  // no colour correction


#if 1
float Chi2(float I, float T, float B, __private float *S, __private float *dS, __constant float *F) {
   float chi2=0.0f, dy ;
   for(int i=0; i<NF; i++) {      
      dy     =  (S[i] - I * MBB(F[i], T, B)) / dS[i] ;
      chi2  +=  dy*dy ;      
   }
   return chi2 ;
}
#else
float Chi2(float I, float T, float B, __private float *S, __private float *dS, __constant float *F) {
   double chi2=0.0f, dy ;
   for(int i=0; i<NF; i++) {      
      dy     =  (S[i] - I * MBB(F[i], T, B)) / dS[i] ;
      chi2  +=  dy*dy ;      
   }
   return (float)chi2 ;
}
#endif



__kernel void FitMBB(
                     __constant float *F,    //  frequencies
                     __global   float *CC,   //  DUMMY --- here no colour corrections
                     __global   float *SS,   //  S[N, NF] brightness values
                     __global   float *dSS,  // dS[N, NF] error estimates
                     __global   float *II,   //  I[N] fitted 250um intensities & initial values
                     __global   float *TT,   //  T[N] fitted temperatures & initial values
                     __global   float *BB    //  B[N] fitted spectral indices & initial values
) {
   // One work item per pixel
   int   id = get_global_id(0) ;
   if (id>=N) return;
   const float NaN = nan((uint)0) ;
   int P = (id==0) ;   
   __private float  S[NF]  ;
   __private float dS[NF] ;
   for(int i=0; i<NF; i++)   S[i] =  SS[id*NF+i] ;
   for(int i=0; i<NF; i++)  dS[i] = dSS[id*NF+i] ;
   // Initial values and initial step sizes
   float  I = II[id], T = TT[id], B = BB[id], x, y ;
   if (isfinite(I)==0) {
      TT[id] = NaN ;  BB[id] = NaN ;   return ;
   }
   float chi2, chi2_in=1.0e30f, chi2_out=1e32f, K=0.90f ;  // initial step 10% of values
   int count = 0 ;
   
   // When no improvement for give step size, decrease step size by a factor K.
   // Stop when the step size "small enough".
   chi2_in =  chi2_out   =   Chi2(I, T, B, S, dS, F) ;
   while((1.0f-K)>KMIN) {
      // if (P) printf("K=%.3e\n", K) ;
      while(1) {  // while steps with this K bring improvement
         // Downward
         chi2       =    Chi2(I*K, T, B, S, dS, F) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            I      *=    K ;
         }
         x          =    clamp(T*K, TMIN, TMAX) ;
         chi2       =    Chi2(I,   x, B, S, dS, F) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            T       =    x ;
         }
#if (FIXED_BETA<0)
         x          =    clamp(B*K, BMIN, BMAX) ;
         chi2       =    Chi2(I,   T, x, S, dS, F) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            B       =    x ;
         }
#endif
         // Upward
         chi2       =    Chi2(I/K, T, B, S, dS, F) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            I      /=    K ;
         }
         x          =    clamp(T/K, TMIN, TMAX) ;
         chi2       =    Chi2(I,   x, B, S, dS, F) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            T       =    x ;
         }
#if (FIXED_BETA<0)
         x          =    clamp(B/K, BMIN, BMAX) ;
         chi2       =    Chi2(I,   T, x, S, dS, F) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            B       =    x ;
         }
         // Joint T+B
         x          =    clamp(T*K, TMIN, TMAX) ;
         y          =    clamp(B/K, BMIN, BMAX) ;
         chi2       =    Chi2(I,   x, y, S, dS, F) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            T       =    x ;
            B       =    y ;
         }
         x          =    clamp(T/K, TMIN, TMAX) ;
         y          =    clamp(B*K, BMIN, BMAX) ;
         chi2       =    Chi2(I,   x, y, S, dS, F) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            T       =    x ;
            B       =    y ;
         }
#endif
         // 
         // if (P) printf("    %8.5f %7.5f %7.5f    chi2 %12.4e  old %12.4e  K=%.4f\n", I, T, B, chi2_in, chi2_out, K) ;
         count += 1 ;
         if ((chi2_in==chi2_out)||(count>1999)) {
            break ;       // no more improvement with the current K -- break from the inner loop
         } 
         chi2_out =   chi2_in ;
      } // while (1)
      K  = 1.0f - (1.0f-K)*KK ;   // decrease step  --- K=0.9 -> 1.0
      // if (P) printf("STEP DECREASED -> K = %.4f\n", K) ;
   } // while K>KMIN
   II[id] = I ;
   TT[id] = T ;
   BB[id] = B ;
#if 0
   printf("MBB_fit_CL: %6d  count = %5d\n", id, count) ;
#endif
}





#else  // CC > 0   --- include colour corrections



float Interpolate(__global float *X, float T, float B) {
   //  X[NCC, NCC] = X[T, B] ... B runs faster
   //  array elements correspond to TMIN...TMAX, BMIN...BMAX values
   //  because of optimisation, this should be a continuous function
   int i, j ;
   float w, a, b ;
   a   =   (NCC-1)*(T-TMIN)/(TMAX-TMIN) ;     // [TMIN, TMAX] -> [0, NCC-1] --- T index as float
   b   =   (NCC-1)*(B-BMIN)/(BMAX-BMIN) ;     //                            --- B index as float
   // interpolate in beta for two constant T rows = i and i+1
   i   =   clamp((int)floor(a), 0, NCC-2) ;   // lower T index
   j   =   clamp((int)floor(b), 0, NCC-2) ;   // lower beta index
   w   =   fmod(b, 1.0f) ;                    // weight for upper beta bin
   a   =   (1.0f-w)*X[ i   *NCC+j] + w*X[ i   *NCC+j+1] ;
   // similar linear interpolation at T index i+1
   b   =   (1.0f-w)*X[(i+1)*NCC+j] + w*X[(i+1)*NCC+j+1] ;
   // final interpolation between the two T rows, between a and b values
   w   =   fmod(a, 1.0f) ;                    // weight for the upper T bins
   return  (1.0f-w)*a + w*b ;
}


float Chi2(float I, float T, float B,
           __private float *S, __private float *dS, __constant float *F, __global float *CC) {
   float chi2=0.0f, dy, cc ;
   for(int i=0; i<NF; i++) {
      cc     =  Interpolate(&(CC[i*NCC*NCC]), T, B) ;  // multiply with cc to go from monochromatic to in-band value
      dy     =  (S[i] - cc * I * MBB(F[i], T, B)) / dS[i] ;
      chi2  +=  dy*dy ;      
   }
   return chi2 ;
}




__kernel void FitMBB(
                     __constant float *F,    //  frequencies
                     __global   float *CC,   //  colour correction factors NCC x NCC
                     __global   float *SS,   //  S[N, NF] brightness values
                     __global   float *dSS,  // dS[N, NF] error estimates
                     __global   float *II,   //  I[N] fitted 250um intensities & initial values
                     __global   float *TT,   //  T[N] fitted temperatures & initial values
                     __global   float *BB    //  B[N] fitted spectral indices & initial values
) {
   // One work item per pixel
   int   id = get_global_id(0) ;
   if (id>=N) return;   
   int P = (id==0) ;   
   __private float  S[NF]  ;
   __private float dS[NF] ;
   for(int i=0; i<NF; i++)   S[i] =  SS[id*NF+i] ;
   for(int i=0; i<NF; i++)  dS[i] = dSS[id*NF+i] ;
   // Initial values and initial step sizes
   float  I = II[id], T = TT[id], B = BB[id], x, y ;
   float chi2, chi2_in=1.0e30f, chi2_out=1e32f, K=0.90f ;  // initial step 10% of values
   
   // When no improvement for give step size, decrease step size by a factor K.
   // Stop when the step size "small enough".
   chi2_in =  chi2_out   =   Chi2(I, T, B, S, dS, F, CC) ;
   while((1.0f-K)>KMIN) {
      // if (P) printf("K=%.3e\n", K) ;
      while(1) {  // while steps with this K bring improvement
         // Downward
         chi2       =    Chi2(I*K, T, B, S, dS, F, CC) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            I      *=    K ;
         }
         x          =    clamp(T*K, TMIN, TMAX) ;
         chi2       =    Chi2(I,   x, B, S, dS, F, CC) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            T       =    x ;
         }
#if (FIXED_BETA<0)
         x          =    clamp(B*K, BMIN, BMAX) ;
         chi2       =    Chi2(I,   T, x, S, dS, F, CC) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            B       =    x ;
         }
#endif
         // Upward
         chi2       =    Chi2(I/K, T, B, S, dS, F, CC) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            I      /=    K ;
         }
         x          =    clamp(T/K, TMIN, TMAX) ;
         chi2       =    Chi2(I,   x, B, S, dS, F, CC) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            T       =    x ;
         }
#if (FIXED_BETA<0)
         x          =    clamp(B/K, BMIN, BMAX) ;
         chi2       =    Chi2(I,   T, x, S, dS, F, CC) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            B       =    x ;
         }
         // Joint T+B
         x          =    clamp(T*K, TMIN, TMAX) ;
         y          =    clamp(B/K, BMIN, BMAX) ;
         chi2       =    Chi2(I,   x, y, S, dS, F, CC) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            T       =    x ;
            B       =    y ;
         }
         x          =    clamp(T/K, TMIN, TMAX) ;
         y          =    clamp(B*K, BMIN, BMAX) ;
         chi2       =    Chi2(I,   x, y, S, dS, F, CC) ;
         if (chi2<chi2_in) {
            chi2_in =    chi2 ;
            T       =    x ;
            B       =    y ;
         }
#endif
         // 
         // if (P) printf("    %8.5f %7.5f %7.5f    chi2 %12.4e  old %12.4e  K=%.4f\n", I, T, B, chi2_in, chi2_out, K) ;
         if (chi2_in==chi2_out) {
            break ;       // no more improvement with the current K -- break from the inner loop
         } 
         chi2_out =   chi2_in ;
      } // while (1)
      K  = 1.0f - (1.0f-K)*KK ;   // decrease step  --- K=0.9 -> 1.0
      // if (P) printf("STEP DECREASED -> K = %.4f\n", K) ;
   } // while K>KMIN
   II[id] = I ;
   TT[id] = T ;
   BB[id] = B ;
}




#endif
