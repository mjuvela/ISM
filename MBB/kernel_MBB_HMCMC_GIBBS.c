#define USE_PRIORS 1

#if (DOUBLE>0)  // -------------------------------------------------------------------------------------------------------
# define  gstep    0.003
# define  real     double
# define  PI       3.141592653589793
# define  RMAX     0.99
# define  ZERO     0.0
# define  EPS      1.0e-5
# define  HALF     0.5
# define  ONE      1.0
# define  ONEHALF  1.5 
# define  TWO      2.0
# define  THREE    3.0
# define  LARGE    1e10
# include "mwc64x_rng.cl"
# define RandU(x)  (MWC64X_NextUint(x)*2.32830643653869628906e-010)
# if 0
#  define RandC(x) (RandU(x)-0.5)
# else    // Normal random variables
#  define RandC(x)  (sqrt(-2.0f*log( clamp(MWC64X_NextUint(x)*2.32830643653869628906e-010f, 1.0e-9f, 0.999999f) ))*cos(2.0f*PI*(MWC64X_NextUint(x)*2.32830643653869628906e-010f)))
# endif
float MBB_Function(float freq, float T, float B) {
   // Modified blackbody law relative to 250um --- um2f(250.0) = 1.19917e+12
   // h/k = 4.799243348e-11,    (h/k)*um2f(250.0) = 57.55107840, um2f(250) = 1.199170e+12
   return  pow(freq/1199169832000.0, 3.0+B)*(exp(57.551078393482776/T)-1.0) / (exp(4.799243348e-11*freq/T)-1.0) ;
}

#else  //  else real is float ------------------------------------------------------------------------------------------

# define  gstep   0.001f
# define  real    float
# define  PI      3.141592653589f
// # define  RMAX    0.999f   // 0.990 -> 0.999
# define  RMAX    0.99f
# define  ZERO    0.0f
# define  EPS     1.0e-5f
# define  HALF    0.5f
# define  ONE     1.0f
# define  ONEHALF 1.5f
# define  TWO     2.0f
# define  THREE   3.0f
# define  LARGE   1e10f
# include "mwc64x_rng.cl"
// # define RandU(x)  ((float)(MWC64X_NextUint(x)/4294967295.0f))
# define RandU(x)  (MWC64X_NextUint(x)*2.32830643653869628906e-010f)
# if 0
#  define RandC(x) (RandU(x)-0.5f)
# else  // Normal random variables
#  define RandC(x)  (sqrt(-2.0f*log(clamp((MWC64X_NextUint(x)*2.32830643653869628906e-010f), 1.0e-9f, 0.999999f) ))*cos(2.0f*PI*(MWC64X_NextUint(x)*2.32830643653869628906e-010f)))
# endif
float MBB_Function(float freq, float T, float B) {
   // Modified blackbody law relative to 250um --- um2f(250.0) = 1.19917e+12
   // h/k = 4.799243348e-11,    (h/k)*um2f(250.0) = 57.55107840, um2f(250) = 1.199170e+12
   return  pow(freq/1199169832000.0f, 3.0f+B)*(exp(57.551078393482776f/T)-1.0f) / (exp(4.799243348e-11f*freq/T)-1.0f) ;
}
#endif // --------------------------------------------------------------------------------------------------------------







__kernel void GPROP(const int tag,            // <0 == initialise rng, 0 == use G, 1 == use GS as starting point
                    __global real *G,         // old global state     G[NG]
                    __global real *GP,        // global proposition   GP[NG+8],  SI == GP[NG:(NG+8)]
                    __global uint *rng0,      // rng state            rng0[2*GLOBAL] ulong
                    __global float *step
                   ) {
   const int id  = get_global_id(0) ;
   if (id>=N) return ;
   real  det ;
   mwc64x_state_t rng ;      
   __local real S[10] ;
   // tag<0   =>    seed rng, do not touch GP but calculate SI based on GP --- CALLER HAS INITIAL VALUES IN GP
   if (tag<0) {  // initialise random number generator
      ulong samplesPerStream = 274877906944L ;  // 2^38
      MWC64X_SeedStreams(&rng, (unsigned long)(fmod(PI*(PI+id+SEED),ONE)*4294967296L), samplesPerStream) ;
   } else {          // read previous state of rng
      rng.c  =   rng0[2*id  ] ;
      rng.x  =   rng0[2*id+1] ;
   }   
   __global real *SI = &(GP[NG]) ;
   
   if (id==0) {
#if (HIER>0)            
      if (tag>=0) { // Calculation new propositions to GP --- on all except the initial call with tag=-1
# if (HIER==1)
         GP[0]  =         G[0] + step[0]*RandC(&rng) ;                 // T   was 0.030
         GP[1]  =         G[1] + step[1]*RandC(&rng) ;                 // B   was 0.020
         GP[2]  =   clamp(G[2] + step[2]*RandC(&rng), EPS,   LARGE) ;  // dT 
         GP[3]  =   clamp(G[3] + step[3]*RandC(&rng), EPS,   LARGE) ;  // dB 
         GP[4]  =   clamp(G[4] + step[4]*RandC(&rng), -RMAX, +RMAX) ;  // rTB
         // GP[0] = G[0] ; GP[1] = G[1] ; GP[2] = G[2] ; GP[3] = G[3] ; GP[4] = G[4] ;
# endif
# if (HIER==20) // trust on priors for correlations
         GP[0]  =         G[0] + step[0]*RandC(&rng)  ;   // I     k
         GP[1]  =         G[1] * step[1]*RandC(&rng)  ;   // T     b
         GP[2]  =         G[2] * step[2]*RandC(&rng)  ;   // B     g *
         GP[3]  =   clamp(G[3] * step[3]*RandC(&rng),  EPS,   LARGE)   ;   // dI    r *   was larger  0.0003 ok !!
         GP[4]  =   clamp(G[4] * step[4]*RandC(&rng),  EPS,   LARGE)   ;   // dT    c
         GP[5]  =   clamp(G[5] * step[5]*RandC(&rng),  EPS,   LARGE)   ;   // dB    m
         GP[6]  =   clamp(G[6] + step[6]*RandC(&rng), -RMAX, +RMAX)  ;     // rIT   k
         GP[7]  =   clamp(G[7] + step[7]*RandC(&rng), -RMAX, +RMAX)  ;     // rIB   b
         GP[8]  =   clamp(G[8] + step[8]*RandC(&rng), -RMAX, +RMAX)  ;     // rTB   g --- 0.004 -> 0.008 @@@
# endif
# if (HIER==2) // trust on priors for correlations
         GP[0]  =   G[0] * (1.0f+gstep*RandC(&rng))  ;  // I     k
         GP[1]  =   G[1] * (1.0f+gstep*RandC(&rng))  ;  // T     b
         GP[2]  =   G[2] * (1.0f+gstep*RandC(&rng))  ;  // B     g *
         GP[3]  =   G[3] * (1.0f+gstep*RandC(&rng))  ;  // dI    r *   was larger  0.0003 ok !!
         GP[4]  =   G[4] * (1.0f+gstep*RandC(&rng))  ;  // dT    c
         GP[5]  =   G[5] * (1.0f+gstep*RandC(&rng))  ;  // dB    m
         GP[6]  =   G[6]        +gstep*RandC(&rng)  ;   // rIT   k
         GP[7]  =   G[7]        +gstep*RandC(&rng)  ;   // rIB   b
         GP[8]  =   G[8]        +gstep*RandC(&rng)  ;   // rTB   g --- 0.004 -> 0.008 @@@
         GP[6]  =   clamp(GP[6], -RMAX, +RMAX) ;
         GP[7]  =   clamp(GP[7], -RMAX, +RMAX) ;
         GP[8]  =   clamp(GP[8], -RMAX, +RMAX) ;         
# endif
      } // tag>0
      
      // Calculate det and SI based on current GP[],  SI[] is part of GP vector
# if (HIER==2)
      // Probability for (I, T, B) using 9 hyperparameters
      //            0   1   2   3   4   5   6    7    8    
      //  GP[9] = [ I0, T0, B0, sI, sT, sB, rIT, rIB, rTB ]
      //  The covariance matrix S[] and its inverse SI[] are stored in column order!
      //          II                  0   1   2
      //    S =   IT  TT              1   3   4
      //          IB  TB   BB         2   4   5
      S[0]   =  GP[3]*GP[3] ;               // var(I)
      S[3]   =  GP[4]*GP[4] ;               // var(T)
      S[5]   =  GP[5]*GP[5] ;               // var(beta)
      S[1]   =  GP[3]*GP[4]* GP[6] ;        // cov(I, T)
      S[2]   =  GP[3]*GP[5]* GP[7] ;        // cov(I, beta)
      S[4]   =  GP[4]*GP[5]* GP[8] ;        // cov(T, beta)
      det    =  S[0]*(S[3]*S[5]-S[4]*S[4]) + S[1]*(S[2]*S[4]-S[1]*S[5]) + S[2]*(S[1]*S[4]-S[2]*S[3]) ;
      // SI * det =       
      // [S3*S5-S4^2,    S2*S4-S1*S5,    S1*S4-S2*S3],
      // [S2*S4-S1*S5,   S0*S5-S2^2,     S1*S2-S0*S4],
      // [S1*S4-S2*S3,   S1*S2-S0*S4,    S0*S3-S1^2 ] 
      SI[0]  =  (S[3]*S[5]-S[4]*S[4]) / det ;
      SI[1]  =  (S[2]*S[4]-S[1]*S[5]) / det ;
      SI[2]  =  (S[1]*S[4]-S[2]*S[3]) / det ;
      SI[3]  =  (S[0]*S[5]-S[2]*S[2]) / det ;
      SI[4]  =  (S[1]*S[2]-S[0]*S[4]) / det ;
      SI[5]  =  (S[0]*S[3]-S[1]*S[1]) / det ;
      SI[6]  =  det ;
      // printf("det %12.4e ->  SI[6]=%12.4e  ... %12.4e %12.4e\n", det, SI[6], G[NG+6], GP[NG+6]) ;
# endif // end of -- HIER==2
#endif  // end of HIER>0
            
      SI[7] = 0.0f ;  // if no priors  ....  additional term to lnP
#if (USE_PRIORS>0)    // Compute probabilitities associated to hyperparameter priors, given the new GP  =>  SI[7]
# if (HIER==1)
      // Each variance as Cauchy(0,sigma) distribution with sigma=2.5 --- (T, beta) ONLY !!!!
      //   p = (1/(pi*sigma)) * [  1+(s/sigma)^2 ]^(-1),  using sigma=2.5
      SI[7]   =   -log((1.0f+(GP[2]/2.5f)*(GP[2]/2.5f)))  -  log((1.0f+(GP[3]/2.5f)*(GP[3]/2.5f))) ;
      // Covariance matrix  R =>   det==1-rho^2   ==>   p = det(R)^(nu-1) = det(R) for nu=2
      // rho close to +/- 1.0  ==> probability goes to zero
      SI[7]  +=   (   (fabs(GP[4])<1.0f) ?  log(1.0f-GP[4]*GP[4]) : -1.0e32f   ) ;
# endif      
# if (HIER==2)
      // Each variance as Cauchy(0,sigma) distribution with sigma=2.5 --- (T, beta) ONLY !!!!
      //   p = (1/(pi*sigma)) * [  1+(s/sigma)^2 ]^(-1),  using sigma=2.5
      SI[7]   =   -log(PI*2.5f*(1.0f+(GP[4]/2.5f)*(GP[4]/2.5f)))  -  log(PI*2.5f*(1.0f+(GP[5]/2.5f)*(GP[5]/2.5f))) ;
      // Covariance matrix  R =>   det==1-rho^2   ==>   p = det(R)^(nu-1) = det(R) for nu=2
      // .... only T-beta correlation
      // SI[7]  +=   (   (fabs(GP[8])<1.0f) ?  log(1.0f-GP[8]*GP[8]) : -1.0e32f   ) ;
      SI[7]  +=   (   (fabs(det)<1.0f) ?  log(1.0f-det*det) : -1.0e32f   ) ; // using directly det
# endif      
#endif   // USE_PRIORS
   }  // if id ==0

   rng0[2*id  ]  = rng.c ;   
   rng0[2*id+1]  = rng.x ;
}






__kernel void GSTEP(__constant  float  *F,         //   F[NF]
                    __global    float  *SS,        //   S[N, NF]
                    __global    float  *dSS,       //  dS[N, NF]
                    __global    real   *G,         //   G[NG+8]
                    __global    real   *L_all,     //   L[N, 3] = current state
                    __global    real   *lnP_all)   //  lnP_all[N]
{
   // Calculate lnP for the current proposal in G[] -- contribution from a single source -> lnP_all
   const int id        =    get_global_id(0) ;
   if (id>=N) return ;
   __global real  *L   =    &(L_all[id*3]) ;        // current source (I, T, beta) = L[3]
   __global float *S   =    &(SS[id*NF]) ;          // current source, S[NF]
   __global float *dS  =    &(dSS[id*NF]) ;
   __global real  *SI  =    &(G[NG]) ;
   real lnP, t, b, dT, dB, r ;   
   
   lnP = ZERO ;


   
#if 1
#if (HIER>0)        
# if (HIER==1)
   // HIER==1 => calculate probability directly based on the hyperparameters (no SI[])
   t     =   L[1] - G[0] ;  // delta T  for the proposal,  using GP[NG]
   b     =   L[2] - G[1] ;  // delta beta for the proposal
   dT    =          G[2] ;   
   dB    =          G[3] ;
   r     =          G[4] ;
   lnP   =  (-HALF/(ONE-r*r)) * ( t*t/(dT*dT) + b*b/(dB*dB) - TWO*r*t*b/(dT*dB) ) - log(TWO*PI*dT*dB*sqrt(ONE-r*r)) ;
# endif // end of --- hier==1
# if (HIER==2)
#  if (STUDENT==0)  
   lnP   = -HALF * (
                    (L[2]-G[2]) * ( SI[5]*(L[2]-G[2]) + SI[4]*(L[1]-G[1]) + SI[2]*(L[0]-G[0]) ) + 
                    (L[1]-G[1]) * ( SI[4]*(L[2]-G[2]) + SI[3]*(L[1]-G[1]) + SI[1]*(L[0]-G[0]) ) + 
                    (L[0]-G[0]) * ( SI[2]*(L[2]-G[2]) + SI[1]*(L[1]-G[1]) + SI[0]*(L[0]-G[0]) )
                   ) ;
   lnP  -=  HALF*log(SI[6]) ; // + ONEHALF*log(TWO*PI) ;  // SI[6] == det
#  else // else student
   // assuming 3D Student t -distribution
   r    = 8*ONE ;
   lnP  = log(pow(ONE + (
                         (L[2]-G[2]) * ( SI[5]*(L[2]-G[2]) + SI[4]*(L[1]-G[1]) + SI[2]*(L[0]-G[0]) ) +
                         (L[1]-G[1]) * ( SI[4]*(L[2]-G[2]) + SI[3]*(L[1]-G[1]) + SI[1]*(L[0]-G[0]) ) +
                         (L[0]-G[0]) * ( SI[2]*(L[2]-G[2]) + SI[1]*(L[1]-G[1]) + SI[0]*(L[0]-G[0]) )
                        )/r,   -(r+THREE)/TWO)    /   sqrt(SI[6]) ) ;
#  endif  // --- end of normal or student
# endif   // --- end of HIER==2   
#endif    // --- end of HIER>0 
#endif
   

   

#if 1
   // Likelihood for flux measurements
   t = ZERO ;
   for(int iband=0; iband<NF; iband++) {
      r     =   L[0] * MBB_Function(F[iband], L[1], L[2]) ;
      r     =  (r-S[iband]) / dS[iband] ;
      t    +=   -HALF*r*r  -  log(dS[iband]*sqrt(TWO*PI)) ;
   } // for iband
   lnP += t ;
#endif
   

#if 1 // enforce parameter limits
   // not needed because only global parameters changed, local parameters remain valid
   // ... or could it be that some source had initially an invalid T or beta ... no !
      if ((L[1]<TMIN)||(L[1]>TMAX)||(L[2]<BMIN)||(L[2]>BMAX)) lnP = -1.0e32f ;  // reject!
#endif 
   
   
#if HIER==2
   if ((SI[6]>DETMIN)&&(isfinite(SI[6]))&&(isfinite(lnP))) {
      lnP_all[id]  = lnP ;
   } else {
      lnP_all[id]  = -1.0e30f ;  // det not ok...
   }
#else
   lnP_all[id] = lnP ;
   // printf("lnO[%3d] = %10.3e\n", id, lnP_all[id]) ;
#endif

   // The iteration on global parameters includes the prior probability of the hyperparameters
   // stored in SI[7], now copied to lnP_ALL[N]
   if (id==0) {
      lnP_all[N] = ZERO ;
#if ((USE_PRIORS>0)&&(HIER>0))
      lnP_all[N] = SI[7] ;   // prior on hyperparameters
#endif
   }
   
}





__kernel void LSTEP(const       int    sample,
                    __global    float  *F,         //  F[NF]
                    __global    float  *SS,        //  S[N, NF]
                    __global    float  *dSS,       //  dS[N, NF]
                    __global    real   *G,         //  G[[5|9]] = mT, mB, dT, dB, rho
                    __global    real   *L_all,     //  L[NPIX, 3] = current state
                    __global    int    *rng0,      //  rng0[2*GLOBAL]
                    __global    real   *lnP0_all,  //  current lnP per source, lnP0[GLOBAL]
                    __global    real   *GS,
                    __global    real   *LS,
                    __global    float  *step)
{
   // Make proposals and test acceptance of the per-source parameters
   // lnP[id] is the likelihood plus the probability of the per-source parameters,
   //         given a fixed hyperdistribution
   // The prior probability of hyperparameters is not visible here, only in the 
   // steps on hyperparameters
   const int id        =   get_global_id(0) ;
   if (id>=N) return ;  // each work item calculates lnP for one source
   __global float *S   =   &( SS[id*NF]) ;       // current source, S[NF]
   __global float *dS  =   &(dSS[id*NF]) ;
   __global real  *SI  =   &(G[NG]) ;            //  G[NG+8]
   real t, b, dT, dB, r, lnP, lnP0 ;
   mwc64x_state_t rng ;
   real L0[3] ;
   real LP[3] ;
   
   // Copy initial L to L0 -- parameters of a single source, tested and accepted or rejected in the following loop
   L0[0]  =  L_all[3*id+0] ;   // I
   L0[1]  =  L_all[3*id+1] ;   // T
   L0[2]  =  L_all[3*id+2] ;   // B
   lnP0   =  lnP0_all[id] ;
   rng.c  =  rng0[2*id  ] ;  
   rng.x  =  rng0[2*id+1] ;

   
   for(int iii=0; iii<LITER; iii++) {

      // a number of iterations based only on the lnP of this particular source
      lnP    = ZERO ;      
      // Proposition for this source L0 -> LP
      LP[0]  =  L0[0] + step[NG+3*id+0]*RandC(&rng) ;    // I
      LP[1]  =  L0[1] + step[NG+3*id+1]*RandC(&rng) ;    // T
      LP[2]  =  L0[2] - step[NG+3*id+2]*RandC(&rng) ;    // B

#if 1
#if (HIER>0) // ......................................................................................................
# if (HIER==1)
      // HIER==1 => calculate probability directly based on the hyperparameters (no SI[])
      t     =   LP[1] - G[0] ;  // delta T  for the proposal,  using G[NG]
      b     =   LP[2] - G[1] ;  // delta beta for the proposal
      dT    =           G[2] ;
      dB    =           G[3] ;
      r     =           G[4] ;
      // probability of the source (T,beta) in the distribution defined by hyperparameters
      lnP   =  (-HALF/(ONE-r*r)) * ( t*t/(dT*dT) + b*b/(dB*dB) - TWO*r*t*b/(dT*dB) ) - log(TWO*PI*dT*dB*sqrt(ONE-r*r)) ;     
# endif            
# if (HIER==2)
#  if (STUDENT==0)  
      lnP   = -HALF * (
                       (LP[2]-G[2]) * ( SI[5]*(LP[2]-G[2]) + SI[4]*(LP[1]-G[1]) + SI[2]*(LP[0]-G[0]) ) + 
                       (LP[1]-G[1]) * ( SI[4]*(LP[2]-G[2]) + SI[3]*(LP[1]-G[1]) + SI[1]*(LP[0]-G[0]) ) + 
                       (LP[0]-G[0]) * ( SI[2]*(LP[2]-G[2]) + SI[1]*(LP[1]-G[1]) + SI[0]*(LP[0]-G[0]) )
                      ) ;
      lnP   -=  HALF*log(SI[6]) ; // + ONEHALF*log(TWO*PI) ;  // SI[6] == det
#  else
      // assuming 3D Student t -distribution
      r    = 8*ONE ;
      lnP  = log(pow(ONE + 
                     (ONE/r)*(
                              (LP[2]-G[2]) * ( SI[5]*(LP[2]-G[2]) + SI[4]*(LP[1]-G[1]) + SI[2]*(LP[0]-G[0]) ) +
                              (LP[1]-G[1]) * ( SI[4]*(LP[2]-G[2]) + SI[3]*(LP[1]-G[1]) + SI[1]*(LP[0]-G[0]) ) +
                              (LP[0]-G[0]) * ( SI[2]*(LP[2]-G[2]) + SI[1]*(LP[1]-G[1]) + SI[0]*(LP[0]-G[0]) )
                             ), -(r+THREE)/TWO)    /   sqrt(SI[6])) ;
#  endif // student or not    
# endif  // end of --- HIER==2
#endif   // end of --- HIER>0 ........................................................................................
#endif
      

#if 1
      // Per-source likelihood, sum over bands
      t = ZERO ;  
      for(int iband=0; iband<NF; iband++) {
         r     =    LP[0] * MBB_Function(F[iband], LP[1], LP[2]) ;
         r     =    (r-S[iband]) / dS[iband] ;
         t    +=   -HALF*r*r  -  log(dS[iband]*sqrt(TWO*PI)) ;
      } // for iband
      lnP += t ;   // lnP must be calculated exactly the same as in GSTEP !!!
#endif
      
      
            
#if 1 // enforce parameter limits
      if ((LP[1]<TMIN)||(LP[1]>TMAX)||(LP[2]<BMIN)||(LP[2]>BMAX)) lnP = -1.0e32f ;  // reject!
#endif 
      
#if (HIER==2) 
      if ((SI[6]>DETMIN)&&(isfinite(SI[6]))&&(isfinite(lnP))&&((lnP-lnP0)>log(RandU(&rng)))) { // accept step for this source
         L0[0] = LP[0] ;   L0[1] = LP[1] ;   L0[2] = LP[2] ;      lnP0 = lnP ;
      }            
#else
      if ((lnP-lnP0)>log(RandU(&rng))) {
         L0[0] = LP[0] ;   
         L0[1] = LP[1] ;   
         L0[2] = LP[2] ;     
         lnP0  = lnP ;
      }            
#endif            
   } // for iii
   

   // Global priors are and remain in lnP_all[N]
   
   
   // Copy final accepted values to persistent arrays:  L0 -> L_all
   L_all[3*id+0] = L0[0] ;  
   L_all[3*id+1] = L0[1] ;  
   L_all[3*id+2] = L0[2] ;    // parameters of this source
   lnP0_all[id]  = lnP0 ;     // last accepted lnP of this source
   rng0[2*id  ]  = rng.c ;  
   rng0[2*id+1]  = rng.x ;

   
   // also add a sample
   if (sample>=0) {
      if (id<NG)  GS[id*SAMPLES+sample] = G[id] ;
      LS[id*3*SAMPLES+0*SAMPLES+sample] = L0[0] ;
      LS[id*3*SAMPLES+1*SAMPLES+sample] = L0[1] ;
      LS[id*3*SAMPLES+2*SAMPLES+sample] = L0[2] ;
   }
}

