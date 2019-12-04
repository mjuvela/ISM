#define TWOPI  6.2831853072f
#define F0     1199169832000.0f  // reference frequency ~ 250um
// #define H_K    4.7995074e-11f
#define H_K    4.799243348e-11f


#define exp native_exp


// #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#if (USE_HD>0)
# define A_RND    742938285
# define UV_AHI  (A_RND >> 15)
# define UV_ALO  (A_RND & 0x7FFF)
float  Rand(ulong *xxx) {
   // HD generator
   private ulong xhi, xlo, mid ;
   xhi   = (*xxx) >> 16;
   xlo   = (*xxx) & 0xFFFF;
   mid   = UV_AHI * xlo + (UV_ALO << 1) * xhi ;
   *xxx  = UV_AHI*xhi + (mid >> 16) + UV_ALO*xlo ;
   if ((*xxx) & 0x80000000)  (*xxx) -= 0x7FFFFFFF ;
   *xxx  += ((mid & 0xFFFF) << 15) ;
   if ((*xxx) & 0x80000000)   *xxx   -= 0x7FFFFFFF ;
   return ((*xxx) / 2147483647.0f ) ;
}
#else
# include "mwc64x_rng.cl"
# define Rand(x)  (MWC64X_NextUint(x)/4294967295.0f)
#endif

#define DEBUG       0
#define CHECK  (100*THIN)   // how often to update C, step




__kernel void MCMC(const      float  SEED,    // random seed for current call
                   __constant float* F,       // frequencies of the bands
                   __global   float* GS,      // fluxes [isource*NF+ifreq]
                   __global   float* dGS,     // flux uncertainties
                   __global   float* INI,     // initial values
                   __global   float* RES,     // {I, T, beta} = RES[3*SAMPLES*i+3*j+{0-2}]
                   __global   float* COVI)    // inv(cov) => dGS not used
{                                    
   // By default RES will contain MCMC samples, {I, T, beta} = RES[3*SAMPLES*i+3*j+{0-2}]
   // If SUMMARY>0, RES will be RES[6*N], faster index 0-5 running over 
   //      mean(I), std(I), mean(T), std(T), mean(beta), std(beta)
   
   
   // METHOD=0  independent normal distributed step propositions
   // METHOD=1  propositions using a parameter covariance matrix
   int  id  = get_global_id(0) ;              // id = one source = one work item
   if (id>=N) return ;
   
   float n0, t0, b0, n, t, b, lnPO=-1e30, lnP ;
   n = n0 = INI[3*id+0] ; 
   t = t0 = INI[3*id+1] ;
   b = b0 = INI[3*id+2] ;
   
#if (METHOD==0)
   const float step_n = 0.02f*n0    ;
   const float step_t = 0.03f*15.0f ;
   const float step_b = 0.02f*2.0f  ;
#endif
   
#if (METHOD==1)
   const float W = (WIN-1.0f)/WIN ;
   int cov_updates = 0 ;
#endif
   
   // Update covariance matrix
   __private float3 s  ;
   __private float3 s2 ;
   __private float3 sc ;
   __private float8 C, L  ;
   
#if (USE_COV>0)
   float sum ;
   __global float *IC = 0L ;
   IC = &(COVI[id*NF*NF]) ;  //  [NF,NF] matrix
   __private float y[NF] ;
#else
   float y ;
#endif
   
   
#if 1
   __private float S[NF], dS[NF] ;
   for(int i=0; i<NF; i++) {
      S[i]  =  GS[id*NF+i] ;
      dS[i] = dGS[id*NF+i] ;
   }
#else
   __global float  *S =  &GS[id*NF] ;
   __global float *dS = &dGS[id*NF] ;
#endif
   
   
   // Pointer to results of this work item
#if (SUMMARY==0)
   __global float *res = &(RES[3*SAMPLES*id]) ;      // 3 = { n, t, b}
#else
   __global float *res = &(RES[8*id]) ;              // 6 = { I, dI, T, dT, B, dB }
   float sn=0.0f, sn2=0.0f, st=0.0f, st2=0.0f, sb=0.0f, sb2=0.0f, so=0.0f, so2=0.0f ;
   int   count = 0 ;
#endif
   
#if (METHOD>0)
   // Start with some plausible covariance matrix
   //   note: for y = y0 + dy*N(0.1), sum(y) = N*y, sum(y^2) = N*y^2 + N*dy^2
   //         if further dy = alpha*y,  sum(y^2) ~ N*y^2 * (1+alpha^2)
   //         dy ~ 0.2*y  =>  1+alpha^2 ~ 1.04
   s   =  (float3)(WIN*n0,          WIN*t0,          WIN*b0         ) ; // sn, st, sb
   s2  =  (float3)(WIN*n0*n0*1.05f, WIN*t0*t0*1.05f, WIN*b0*b0*1.05f) ; // sn^2, st^2, sb^2
   sc  =  (float3)(WIN*n0*t0,       WIN*n0*b0,       WIN*t0*b0) ;       // snt, snb, stb   
   // Covariance matrix C based on the above sums
   //                     s0  s3  s4       C11  C21  C31
   //                     s3  s1  s5       C21  C22  C32
   //                     s4  s5  s2       C31  C32  C33
   // Variances
   C.s0  =  (s2.x-s.x*s.x/WIN) / (WIN-1.0f) ;  // s11  =  var(n)
   C.s1  =  (s2.y-s.y*s.y/WIN) / (WIN-1.0f) ;  // s22  =  var(t)
   C.s2  =  (s2.z-s.z*s.z/WIN) / (WIN-1.0f) ;  // s22  =  var(b)
   // Covariances
# if 0
   C.s3  =   (sc.x-s.x*s.y/WIN) / (WIN-1.0f) ;  // s12  =  cov(n,t)
   C.s4  =   (sc.y-s.x*s.z/WIN) / (WIN-1.0f) ;  // s13  =  cov(n,b)
   C.s5  =   (sc.z-s.y*s.z/WIN) / (WIN-1.0f) ;  // s23  =  cov(t,b)
# else
   C.s3  = 0.0f ;  C.s4 = 0.0f ; C.s5 = 0.0f ;
# endif   
   // Cholesky decomposition of C = matrix L
   //                      s0   0    0        L11  0    0  
   //                      s1   s3   0        L21  L22  0  
   //                      s2   s4   s5       L31  L32  L33   
   // #Cholesky decomposition
   //  L.s0 =   L11  =  sqrt(C11)
   //  L.s1 =   L21  =  C21/L11
   //  L.s3 =   L22  =  sqrt(C22 - L21**2.0)
   //  L.s2 =   L31  =  C31/L11
   //  L.s4 =   L32  =  (C32-L31*L21)/L22
   //  L.s5 =   L33  =  sqrt(C33 - L31**2.0 - L32**2.0)       
   L.s0  =  sqrt(C.s0) ;
   L.s1  =  C.s3/L.s0 ;
   L.s3  =  sqrt(max(C.s1 - L.s1*L.s1, 1.0e-6f)) ;
   L.s2  =  C.s4/L.s0 ;
   L.s4  =  (C.s5-L.s2*L.s1)/L.s3 ;
   L.s5  =  sqrt(max(C.s2 - L.s2*L.s2 - L.s4*L.s4, 1.0e-6f)) ;
#endif
   
   
#if (USE_HD>0)
   ulong rng = 123317 + 1231*id + SEED*37125*id ;
#else
   mwc64x_state_t rng ;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
#endif
   
   
   int accepted=0 ;
   float step = 0.1f ;
   float U1, U2, r1, r2, r3, r4 ;
   
   
   for(int c=0; c<BURNIN+SAMPLES*THIN; c++) {
      
      // generate step based on covariance matrix (covariance -> matrix L)
#if (METHOD==0) // ------------------------------------------------------------------------
      
      // Use approximate normal-distributed random variables
      U1 = Rand(&rng) ; U2 = Rand(&rng) ; 
      r1 = sqrt(-2.0f*log(U1)) ; r2 = r1*cos(TWOPI*U2) ; r1 *= sin(TWOPI*U2) ;
      U1 = Rand(&rng) ; U2 = Rand(&rng) ; 
      r3 = sqrt(-2.0f*log(U1)) ; r4 = r3*cos(TWOPI*U2) ; r3 *= sin(TWOPI*U2) ;
      //
      n  = n0 + r1*step_n ;
      t  = t0 + r2*step_t ;
# if (FIXED_BETA==0)
      b  = b0 + r3*step_b ;
# endif
      
#else  // METHOD == 1 ---------------------------------------------------------------------
      
# if 1
      // Use approximate normal-distributed random variables
      U1 = Rand(&rng) ; U2 = Rand(&rng) ; 
      r1 = sqrt(-2.0f*log(U1)) ; r2 = r1*cos(TWOPI*U2) ; r1 *= sin(TWOPI*U2) ;
      U1 = Rand(&rng) ; U2 = Rand(&rng) ; 
      r3 = sqrt(-2.0f*log(U1)) ; r4 = r3*cos(TWOPI*U2) ; r3 *= sin(TWOPI*U2) ;
      //
      n  = L.s0 * r1 ;
      t  = L.s1 * r1  +  L.s3 * r2 ;
#  if (FIXED_BETA==0)
      b  = L.s2 * r1  +  L.s4 * r2  + L.s5 * r3 ;
#  endif
# else 
      // Use uniform random numbers
      r1 = Rand(&rng)-0.5f ;  r2 =  Rand(&rng)-0.5f ;  r3 =  Rand(&rng)-0.5f ;
      n  = L.s0 * r1 ;
      t  = L.s1 * r1  +  L.s3 * r2 ;
#  if (FIXED_BETA==0)
      b  = L.s2 * r1  +  L.s4 * r2  + L.s5 * r3 ;
#  endif
# endif
      n *= step ;  t *= step ;  b *= step ;      
# if 0
      // Bactrian kernel
      lnP = n*n/C.s0  + t*t/C.s1  + b*b/C.s2 ;
      lnP = 1.0 + 1.0/(0.3+lnP) ;
      n *= lnP ; t *= lnP ; 
#  if (FIXED_BETA==0)
      b *= lnP ;
#  endif
# endif
      n += n0 ;  t += t0 ;  
# if (FIXED_BETA==0)
      b += b0 ;
# endif      
#endif // ---------------------------------------------------------------------------------
      
      

      
      // "prior" -- hard limits on temperature and spectral index
      t = clamp(t,  TMIN, TMAX) ;
#if (FIXED_BETA==0)
      b = clamp(b,  BMIN, BMAX) ;
#endif      
      
      lnP  =  0.0f ;
      
      
#if (USE_COV>0) // using full covariance matrix    d' * IC * d
      for(int i=0; i<NF; i++) {
         y[i] =  n*pow(F[i]/F0, 3.0f+b) * (exp(H_K*F0/t)-1.0f) / (exp(H_K*F[i]/t)-1.0f) ;
      }
      for(int i=0; i<NF; i++) {   // over IC columns
         sum = 0.0f ;
         for(int j=0; j<NF; j++)  sum += (y[j]-S[j]) * IC[NF*j+i] ;  // IC[j,i] = IC[NF*j+i]
         lnP  -=  0.5f*sum*(y[i]-S[i]) ;
      }      
#else  
      for(int i=0; i<NF; i++) {
         y    =  n*pow(F[i]/F0, 3.0f+b) * (exp(H_K*F0/t)-1.0f) / (exp(H_K*F[i]/t)-1.0f) ;
         lnP -=  0.5*(y-S[i])*(y-S[i])/(dS[i]*dS[i]) ;
      }
      
#endif
      
      if ((lnP-lnPO)>log(Rand(&rng))) {
         n0 = n ;  t0 = t ;  
#  if (FIXED_BETA==0)
         b0 = b ;  
#  endif
         lnPO = lnP ; accepted++ ;
      }      
      
      
      
      if (c%THIN==0) {      // Update output arrays
         
         if (c>=BURNIN) {               // Start saving only after BURNIN steps
            int j = (c-BURNIN) / THIN ;
#if (SUMMARY==0)
            res[3*j+0] = n0 ;   res[3*j+1] = t0 ;   res[3*j+2] = b0 ;
#else
            sn += n0 ;  sn2 += n0*n0 ;
            st += t0 ;  st2 += t0*t0 ;
            sb += b0 ;  sb2 += b0*b0 ;
            // sample of 250um optical depth -- assuming [n] = MJy/sr
            //    I*1e-17 * (exp()-1) *   (c^2/(2*h*f^3)) =  3.9329039126517397e-07
            y  =   n0 * (exp(H_K*F0/t0)-1.0f)  *  3.932903912651e-07f ;
            so += y  ;  so2 += y*y ;
            count += 1 ;
#endif
         }
         
         
#if (METHOD==1)
         // Update counters -- running sums over "WIN" samples
         // old sums have weight (WIN-1.0)/WIN, new element has weight one
         s.x  = mad(W,  s.x, n0) ;     s.y  = mad(W,  s.y, t0) ;    s.z  = mad(W,  s.z, b0) ;
         s2.x = mad(W, s2.x, n0*n0) ;  s2.y = mad(W, s2.y, t0*t0) ; s2.z = mad(W, s2.z, b0*b0) ;
         sc.x = mad(W, sc.x, n0*t0) ;  sc.y = mad(W, sc.y, n0*b0) ; sc.z = mad(W, sc.z, t0*b0) ;
#endif
         // From time to time, also update the covariance matrix
         if ((c%CHECK)==(CHECK-1)) {    // CHECK=100*THIN ~ hundreds of steps
#if (METHOD==1)
            if (accepted>(CHECK/20)) {     // do not update if the chain was stuck
               // Covariance matrix C based on the sums s, s2, and sc
               C.s0  =  (s2.x-s.x*s.x/WIN) / (WIN-1.0) ;  // s11  =  var(n)
               C.s1  =  (s2.y-s.y*s.y/WIN) / (WIN-1.0) ;  // s22  =  var(t)
               C.s2  =  (s2.z-s.z*s.z/WIN) / (WIN-1.0) ;  // s22  =  var(b)
               //
               C.s3  =  (sc.x-s.x*s.y/WIN) / (WIN-1.0) ;  // s12  =  cov(n,t)
               C.s4  =  (sc.y-s.x*s.z/WIN) / (WIN-1.0) ;  // s13  =  cov(n,b)
               C.s5  =  (sc.z-s.y*s.z/WIN) / (WIN-1.0) ;  // s23  =  cov(t,b)
               
               
               // Cholesky decomposition => matrix L based on the covariance matrix
               //       C.s0                           L.s0
               //       C.s3  C.s1             -->     L.s1   L.s3
               //       C.s4  C.s5  C.s2               L.s2   L.s4  L.s5
               L.s0  =  sqrt(C.s0) ;
               L.s1  =  C.s3/L.s0 ;
               L.s3  =  sqrt(max(C.s1 - L.s1*L.s1, 1.0e-6f)) ;
               L.s2  =  C.s4/L.s0 ;
               L.s4  =  (C.s5-L.s2*L.s1)/L.s3 ;
               L.s5  =  sqrt(max(C.s2 - L.s2*L.s2 - L.s4*L.s4, 1.0e-6f)) ;               
# if 0
               if (id==0) {
                  printf(" %5d C  %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n", c-BURNIN, C.s0,C.s1,C.s2,C.s3,C.s4,C.s5) ;
                  printf("       L  %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n", L.s0,L.s1,L.s2,L.s3,L.s4,L.s5) ;
               }
# endif                              
               if (cov_updates<2) {  // to make sure chain has stabilised before steps are stored
                  cov_updates++ ; 
                  if (c>=BURNIN) c = BURNIN ;
               }
            }
#endif  // end of -- METHOD==1
            
            
#if 1  // step size adjustment
            if (accepted>(CHECK/2)) {       // over 50% acceptance rate => longer steps
               step = clamp(   step*1.20f, 0.003f, 10.0f) ;
            } else {
               if (accepted<(CHECK/40)) {   // less than 2.5% acceptance rate => shorter steps
                  step = clamp(step*0.87f, 0.003f, 10.0f) ;
               }
            }
#endif
            accepted = 0 ;
         } // CHECK-1
         
         
         
      } // c%THIN==0
   } // for c

   
#if (SUMMARY>0)
   // I, T, beta, tau --- mean values and std values
   res[0]   =   sn / count ;
   res[1]   =   sqrt( (sn2 - sn*sn/count) / count ) ;
   res[2]   =   st / count ;
   res[3]   =   sqrt( (st2 - st*st/count) / count ) ;
   res[4]   =   sb / count ;   
#  if (FIXED_BETA==0)   
   res[5]   =   sqrt( (sb2 - sb*sb/count) / count ) ;
#  else
   res[5]   =   0.0f ;
#  endif
   res[6]   =   so / count ;   
   res[7]   =   sqrt( (so2 - so*so/count) / count ) ;
#endif
   
}











__kernel void HMCMC(const      float  SEED,    // random seed for current call
                    __constant float* F,       // frequencies of the bands
                    __global   float* GS,      // fluxes [isource*NF+ifreq]
                    __global   float* dGS,     // flux uncertainties
                    __global   float* INI,     // initial values
                    __global   float* RES,     // {I, T, beta} = RES[3*SAMPLES*i+3*j+{0-2}]
                    __global   float* COVI)    // inv(cov) => dGS not used
{                                                                 
   int  id  = get_global_id(0) ;               // id = one source = one work item
   if (id>=N) return ;
   
   float n0, t0, b0, n, t, b, lnPO=-1e30, lnP ;
   float pn=0.0f, pt=0.0f, pb=0.0f ;
   n = n0 = INI[3*id+0] ; 
   t = t0 = INI[3*id+1] ;
   b = b0 = INI[3*id+2] ;
   
#if (USE_COV>0)
   float sum ;
   __global float *IC = 0L ;
   IC = &(COVI[id*NF*NF]) ;  //  [NF,NF] matrix
   __private float y[NF] ;
#else
   float y ;
#endif
   
#if 1
   __private float S[NF], dS[NF] ;
   for(int i=0; i<NF; i++) {
      S[i]  =  GS[id*NF+i] ;
      dS[i] = dGS[id*NF+i] ;
   }
#else
   __global float  *S =  &GS[id*NF] ;
   __global float *dS = &dGS[id*NF] ;
#endif
   
   
   // Pointer to results of this work item
#if (SUMMARY==0)
   __global float *res = &(RES[3*SAMPLES*id]) ;      // 3 = { n, t, b}
#else
   __global float *res = &(RES[8*id]) ;              // 3 = { n, t, b}
   float sn=0.0f, sn2=0.0f, st=0.0f, st2=0.0f, sb=0.0f, sb2=0.0f, so=0.0f, so2=0.0f ;
   int   count = 0 ;
#endif
   

   
#if (USE_HD>0)
   ulong rng = 123317 + 1231*id + SEED*37125*id ;
#else
   mwc64x_state_t rng ;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
#endif
   
   
   int   accepted=0, cov_updates=0 ;
   float U1, U2, r1, r2, r3, r4 ;
   float mn=1.0f, mt=1.0f, mb=2.0f ; // masses
   
   
#define nstep 20   
   float     step  = 0.01f ;
   
   
   
   for(int c=0; c<BURNIN+SAMPLES*THIN; c++) {
      
      
      // Random momentum variables (pn,pt,pb)
      U1 = Rand(&rng) ; U2 = Rand(&rng) ; 
      r1 = sqrt(-2.0f*log(U1)) ; r2 = r1*cos(TWOPI*U2) ; r1 *= sin(TWOPI*U2) ;
      pn = 2.5f*r1 ;
      pt = 2.5f*r2 ;
      U1 = Rand(&rng) ; U2 = Rand(&rng) ; 
      r1 = sqrt(-2.0f*log(U1)) ; r2 = r1*cos(TWOPI*U2) ; r1 *= sin(TWOPI*U2) ;
      pb = 2.5f*r1 ;
      // Leapfrog
      //        p_i(+0.5) = p_i(0)    -  (e/2) dU/dq_i ( q(0) )
      //        q_i(+1)   = q_i(0)    +   e   p_i(+0.5) / m_i
      //        p_i(+1)   = p_i(+0.5) -  (e/2) dU/dq_i ( q(+1) )
      //
      // U = -ln(p) = 0.5*(Y-y)^2,  Y = a*f^(3+b) / (exp(H_K*f/t)-1)
      n = n0 ; t = t0 ; b = b0  ;          // old values, start of the step
      
      // This loop on GPU: nstep = 2 -> 20,   run time ~1sec -> >10 sec
      //    -cl-fast-relaxed-math     -> 7.4 seconds
      for(int istep=0; istep<nstep; istep++) {
         // Gradients at the initial position (n0, t0, b0)
         r1 = 0.0f ; r2 = 0.0f ; r3 = 0.0f ;  // dU/dn,  dU/dt,  dU/db
         for(int i=0; i<NF; i++) {
            y    =   pow(F[i]/F0, 3.0f+b) * (exp(H_K*F0/t)-1.0f) / (exp(H_K*F[i]/t)-1.0f) ;
            r4   =   y * (n*y-S[i]) / (dS[i]*dS[i]) ;
            r1  +=   r4  ;
            r2  +=   r4 * n * H_K*F[i]/(t*t) / (1.0 - exp(-H_K*F[i]/t)) ;
            r3  +=   r4 * n * log(F[i]/F0) ;
         }        
         // Update momentum, first half step
         pn  -=  (0.5*step) * r1 ;
         pt  -=  (0.5*step) * r2 ;
         pb  -=  (0.5*step) * r3 ;
         // Update parameters
         n   +=   step*pn/mn ;
         t   +=   step*pt/mt ;
         b   +=   step*pb/mb ;
         // Gradients at the end point (with the new n, t, b)
         r1 = 0.0f ; r2 = 0.0f ; r3 = 0.0f ;  // dU/dn,  dU/dt,  dU/db
         for(int i=0; i<NF; i++) {
            y    =   pow(F[i]/F0, 3.0f+b) * (exp(H_K*F0/t)-1.0f) / (exp(H_K*F[i]/t)-1.0f) ;
            r4   =   y * (n*y-S[i]) / (dS[i]*dS[i]) ;
            r1  +=   r4  ;
            r2  +=   r4 * n * H_K*F[i]/(t*t) / (1.0 - exp(-H_K*F[i]/t)) ;
            r3  +=   r4 * n * log(F[i]/F0) ;
         }        
         // Update momentum, second half step
         pn  -=  (0.5f*step)*r1 ;
         pt  -=  (0.5f*step)*r2 ;
         pb  -=  (0.5f*step)*r3 ;
      }
      
            
      // "prior" -- hard limits on temperature and spectral index
      n = clamp(n,  1e-10f, 1e10f) ;
      t = clamp(t,  TMIN, TMAX) ;
#  if (FIXED_BETA==0)
      b = clamp(b,  BMIN, BMAX) ;
#  else
      b = b0 ;
#  endif
      // Final probability
      lnP  =  0.0f ;
#if (USE_COV>0) // using full covariance matrix    d' * IC * d
      for(int i=0; i<NF; i++) {
         y[i] =  n*pow(F[i]/F0, 3.0f+b) * (exp(H_K*F0/t)-1.0f) /  (exp(H_K*F[i]/t)-1.0f) ;
      }
      for(int i=0; i<NF; i++) {   // over IC columns
         sum = 0.0f ;
         for(int j=0; j<NF; j++)  sum += (y[j]-S[j]) * IC[NF*j+i] ;  // IC[j,i] = IC[NF*j+i]
         lnP  -=  0.5f*sum*(y[i]-S[i]) ;
      }      
#else  
      for(int i=0; i<NF; i++) {
         y    =  n * pow(F[i]/F0, 3.0f+b) * (exp(H_K*F0/t)-1.0f) / (exp(H_K*F[i]/t)-1.0f) ;
         lnP -=  0.5*(y-S[i])*(y-S[i])/(dS[i]*dS[i]) ;
      }
      
#endif         
      // Test acceptance of the step
      if ((lnP-lnPO)>log(Rand(&rng))) {
         n0 = n ;  t0 = t ;  b0 = b ;  lnPO = lnP ; accepted++ ;
      }      
      
      if ((c%THIN)==0) {                // Update output arrays
         if (c>=BURNIN) {               // Start saving only after BURNIN steps
            int j = (c-BURNIN) / THIN ;
#if (SUMMARY==0)
            res[3*j+0] = n0 ;   res[3*j+1] = t0 ;   res[3*j+2] = b0 ;
#else
            sn += n0 ;  sn2 += n0*n0 ;
            st += t0 ;  st2 += t0*t0 ;
            sb += b0 ;  sb2 += b0*b0 ;
            // sample of 250um optical depth -- assuming [n] = MJy/sr
            y  =   n0 * (exp(H_K*F0/t0)-1.0f)  *  3.932903912651e-07f ;
            so += y  ;  so2 += y*y ;
            count += 1 ;
#endif
         }
      }
      
      if ((c%CHECK)==0) {                // step adjustment
         if (accepted>(0.8*CHECK)) {     // accepted too often
            step = clamp(   step*1.20f, 0.001f, 1.0f) ;               
         } else {
            if (accepted<(0.3*CHECK)) {   // accepted too rarely
               step = clamp(step*0.87f, 0.001f, 1.0f) ;
            }
         }
         accepted = 0 ;
      }
      
   } // for c

#if (SUMMARY>0)
   res[0]   =   sn / count ;
   res[1]   =   sqrt( (sn2 - sn*sn/count) / count ) ;
   res[2]   =   st / count ;
   res[3]   =   sqrt( (st2 - st*st/count) / count ) ;
   res[4]   =   sb / count ;   
#  if (FIXED_BETA==0)
   res[5]   =   sqrt( (sb2 - sb*sb/count) / count ) ;
#  else
   res[5]   =   0.0f ;
#  endif
   res[6]   =   so / count ;   
   res[7]   =   sqrt( (so2 - so*so/count) / count ) ;
#endif
   
}






__kernel void RAM(const      float  SEED,    // random seed for current call
                  __constant float* F,       // frequencies of the bands
                  __global   float* GI,      // fluxes [isource*NF+ifreq]
                  __global   float* dGI,     // flux uncertainties
                  __global   float* INI,     // initial values
                  __global   float* RES,     // {I, T, beta} = RES[3*SAMPLES*i+3*j+{0-2}]
                  __global   float* COVI)    // inv(cov) => dGS not used
{                                                                 
   int  id  = get_global_id(0) ;              // id = one source = one work item
   if (id>=N) return ;
   
   float n0, t0, b0, n, t, b, lnPO=-1e30, lnP, lnA, k ;
   n = n0 = INI[3*id+0] ; 
   t = t0 = INI[3*id+1] ;
   b = b0 = INI[3*id+2] ;
   
#if (USE_COV>0)
   float sum ;
   __global float *IC = 0L ;
   IC = &(COVI[id*NF*NF]) ;  //  [NF,NF] matrix
   __private float y[NF] ;
#else
   float y ;
#endif
   
   float8 S, C ;
   float3 U ;
   
   __private float I[NF], dI[NF] ;
   for(int i=0; i<NF; i++) {
      I[i]  =  GI[id*NF+i] ;
      dI[i] = dGI[id*NF+i] ;
   }
   
   // Pointer to results of this work item
#if (SUMMARY==0)
   __global float *res = &(RES[3*SAMPLES*id]) ;      // 3 = { n, t, b}
#else
   __global float *res = &(RES[8*id]) ;              // 6 = { I, dI, T, dT, B, dB }
   float sn=0.0f, sn2=0.0f, st=0.0f, st2=0.0f, sb=0.0f, sb2=0.0f, so=0.0f, so2=0.0f ;
   int   count = 0 ;
#endif
   
   
   // Cholesky decomposition of C = matrix L
   //                      s0   0    0        L11  0    0  
   //                      s1   s3   0        L21  L22  0  
   //                      s2   s4   s5       L31  L32  L33   
   // #Cholesky decomposition
   //  L.s0 =   L11  =  sqrt(C11)
   //  L.s1 =   L21  =  C21/L11
   //  L.s3 =   L22  =  sqrt(C22 - L21**2.0)
   //  L.s2 =   L31  =  C31/L11
   //  L.s4 =   L32  =  (C32-L31*L21)/L22
   //  L.s5 =   L33  =  sqrt(C33 - L31**2.0 - L32**2.0)       
   
   
#if (USE_HD>0)
   ulong rng = 123317 + 1231*id + SEED*37125*id ;
#else
   mwc64x_state_t rng ;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
#endif
   
   // Initial S matrix, lower diagonal, stored as
   //                       s0  0   0 
   //                       s1  s2  0 
   //                       s3  s4  s5
   S.s0 = 0.1f ;
   S.s1 = 0.0f ;    S.s2 = 0.1f ;
   S.s3 = 0.0f ;    S.s4 = 0.0f ;    S.s5 = 0.03f ;
   
   
   for(int c=0; c<BURNIN+SAMPLES*THIN; c++) {
      
      // Proposition  Y = X + S*u
      // S is updated lower diagonal matrix, u random vector (spherically symmetric)
      U.x = Rand(&rng)-0.5f ;   U.y = Rand(&rng)-0.5f ;   U.z = Rand(&rng)-0.5f ;      
      n   = n0 + S.s0*U.x ;
      t   = t0 + S.s1*U.x + S.s2*U.y ;
#if (FIXED_BETA==0)
      b   = b0 + S.s3*U.x + S.s4*U.y + S.s5*U.z ;
#endif

      
      // "prior" -- hard limits on temperature and spectral index
      t = clamp(t,  TMIN, TMAX) ;
      b = clamp(b,  BMIN, BMAX) ;
      
      // Calculate likelihood
      lnP  =  0.0f ;      
#if (USE_COV>0) // using full covariance matrix    d' * IC * d
      for(int i=0; i<NF; i++) {
         y[i] =  n*pow(F[i]/F0, 3.0f+b) * (exp(H_K*F0/t)-1.0f) / (exp(H_K*F[i]/t)-1.0f) ;
      }
      for(int i=0; i<NF; i++) {   // over IC columns
         sum = 0.0f ;
         for(int j=0; j<NF; j++)  sum += (y[j]-I[j]) * IC[NF*j+i] ;  // IC[j,i] = IC[NF*j+i]
         lnP  -=  0.5f*sum*(y[i]-I[i]) ;
      }      
#else  
      for(int i=0; i<NF; i++) {
         y    =  n*pow(F[i]/F0, 3.0f+b) * (exp(H_K*F0/t)-1.0f) / (exp(H_K*F[i]/t)-1.0f) ;
         lnP -=  0.5*(y-I[i])*(y-I[i])/(dI[i]*dI[i]) ;
      }
      
#endif
      
      // Acceptance probability
      lnA = lnP-lnPO ;      
      if (lnA>log(Rand(&rng))) {
         n0 = n ;  t0 = t ;  b0 = b ;  lnPO = lnP ;
      }      
      
#if 1
      // Update matrix S
      //   S * [   I + eeta * (alpha - alpha*) * U*U^T / ||U||   ] * S^T
      //   eeta   = min{ 1, d*n^(-2/3) },    alpha* = 0.235
      k  =  min(1.0f, 3.0f*pow(c+1.0f,-0.66667f)) * (min(1.0f,exp(lnA))-0.235f) ;
      U  =  normalize(U) ;
      // Now  C  =  S*(I+k*U*U')*S'
      //           C0  C1  C3      s0  s1  s3
      //           C1  C2  C4      s1  s2  s4
      //           C3  C4  C5      s3  s4  s5
      //       S0^2   *(Ux^2   *k+1)
      C.s0 = S.s0*S.s0*(U.x*U.x*k+1) ;
      //       S0  *S1*( Ux^2  *k+1)  +S0 * S2* Ux *Uy*k
      C.s1 = S.s0*S.s1*(U.x*U.x*k+1)+S.s0*S.s2*U.x*U.y*k ;
      //     S2*  (  S2*( Uy^2  *k+1)+  S1* Ux *Uy*k)+  S1* ( S1*( Ux^2  *k+1) + S2* Ux* Uy*k)
      C.s2 = S.s2*(S.s2*(U.y*U.y*k+1)+S.s1*U.x*U.y*k)+S.s1*(S.s1*(U.x*U.x*k+1)+S.s2*U.x*U.y*k) ;
      //       S0 * S3*( Ux^2  *k+1)+  S0  *S5* Ux *Uz*k+  S0*  S4* Ux *Uy*k
      C.s3 = S.s0*S.s3*(U.x*U.x*k+1)+S.s0*S.s5*U.x*U.z*k+S.s0*S.s4*U.x*U.y*k ;
      //       S4*(  S2*(Uy^2   *k+1)+  S1 *Ux* Uy*k)+  S3*(  S1*( Ux^2  *k+1) + S2* Ux *Uy*k)+  S5*(  S2* Uy* Uz*k+  S1* Ux *Uz*k)
      C.s4 = S.s4*(S.s2*(U.y*U.y*k+1)+S.s1*U.x*U.y*k)+S.s3*(S.s1*(U.x*U.x*k+1)+S.s2*U.x*U.y*k)+S.s5*(S.s2*U.y*U.z*k+S.s1*U.x*U.z*k) ;
      //       S5*  (S5*( Uz^2  *k+1)+  S4* Uy* Uz*k+  S3* Ux* Uz*k)+  S4*(  S4*( Uy^2  *k+1)+  S5* Uy* Uz*k+  S3* Ux* Uy*k)+  S3*(  S3*( Ux^2*  k+1)+  S5* Ux* Uz*k+  S4* Ux* Uy*k)
      C.s5 = S.s5*(S.s5*(U.z*U.z*k+1)+S.s4*U.y*U.z*k+S.s3*U.x*U.z*k)+S.s4*(S.s4*(U.y*U.y*k+1)+S.s5*U.y*U.z*k+S.s3*U.x*U.y*k)+S.s3*(S.s3*(U.x*U.x*k+1)+S.s5*U.x*U.z*k+S.s4*U.x*U.y*k) ;
      // printf("  C:\n") ;
      // printf("    %10.3e\n", C.s0) ;
      // printf("    %10.3e  %10.3e\n", C.s1, C.s2) ;
      // printf("    %10.3e  %10.3e  %10.3e\n", C.s3, C.s4, C.s5) ;
      // New S from the Cholesky decomposition of C
      //                      S.s0   0     0                  C.s0             
      //                      S.s1   S.s2  0          <--     C.s1   C.s2      
      //                      S.s3   S.s4  S.s5               C.s3   C.s4  C.s5
      // #Cholesky decomposition
      //  S.s0 =   S11  =  sqrt(C11)                        =  sqrt(C.s0)
      //  S.s1 =   S21  =  C21/S11                          =  C.s1/S.s0
      //  S.s2 =   S22  =  sqrt(C22 - S21**2.0)             =  sqrt(C.s2-S.s1**2)
      //  S.s3 =   S31  =  C31/S11                          =  C.s3/S.s0
      //  S.s4 =   S32  =  (C32-S31*S21)/S22                =  (C.s4-S.s3*S.s1)/S.s2
      //  S.s5 =   S33  =  sqrt(C33 - S31**2.0 - S32**2.0)  =  sqrt(C.s5-S.s3**2-S.s4**2)
      S.s0 =    sqrt(C.s0) ;
      S.s1 =    C.s1/S.s0 ;
      S.s2 =    sqrt(max(C.s2-S.s1*S.s1,1.0e-6f)) ;
      S.s3 =    C.s3/S.s0 ;
      S.s4 =    (C.s4-S.s3*S.s1)/S.s2 ;
      S.s5 =    sqrt(max(C.s5-S.s3*S.s3-S.s4*S.s4,1.0e-6f)) ;
      // printf("  S:\n") ;
      // printf("  %10.3e\n", S.s0) ;
      // printf("  %10.3e  %10.3e\n", S.s1, S.s2) ;
      // printf("  %10.3e  %10.3e  %10.3e\n", S.s3, S.s4, S.s5) ;
#endif
      
      if (c%THIN==0) {      // Update output arrays
         if (c>=BURNIN) {               // Start saving only after BURNIN steps
            int j = (c-BURNIN) / THIN ;
#if (SUMMARY==0)
            res[3*j+0] = n0 ;   res[3*j+1] = t0 ;   res[3*j+2] = b0 ;
#else
            sn += n0 ;  sn2 += n0*n0 ;
            st += t0 ;  st2 += t0*t0 ;
            sb += b0 ;  sb2 += b0*b0 ;
            // sample of 250um optical depth -- assuming [n] = MJy/sr
            y   =   n0 * (exp(H_K*F0/t0)-1.0f)  *  3.932903912651e-07f ;
            so += y  ;  so2 += y*y ;
            count += 1 ;
#endif
         }
      } // c%THIN==0
      
   } // for c -- loop over samples
 
   
#if (SUMMARY>0)
   res[0]   =   sn / count ;
   res[1]   =   sqrt( (sn2 - sn*sn/count) / count ) ;
   res[2]   =   st / count ;
   res[3]   =   sqrt( (st2 - st*st/count) / count ) ;
   res[4]   =   sb / count ;   
#  if (FIXED_BETA==0)
   res[5]   =   sqrt( (sb2 - sb*sb/count) / count ) ;
#  else
   res[5]   =   0.0f ;
#  endif
   res[6]   =   so / count ;   
   res[7]   =   sqrt( (so2 - so*so/count) / count ) ;
#endif
   
}





