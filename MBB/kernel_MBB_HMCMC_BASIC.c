# define PI    3.1415926536f
# include "mwc64x_rng.cl"
# define USE_PRIORS 1

# define RandU(x)  (MWC64X_NextUint(x)*2.32830643653869628906e-010f)
# define RandC(x)  (sqrt(-2.0f*log(clamp((MWC64X_NextUint(x)*2.32830643653869628906e-010f), 1.0e-9f, 0.999999f) ))*cos(2.0f*PI*(MWC64X_NextUint(x)*2.32830643653869628906e-010f)))

#if (DOUBLE>0)
#define real double
#else
#define real float
#endif


float MBB_Function(float freq, float T, float B) {  // Modified blackbody law relative to 250um
   return  pow(freq/1199169832000.0f, 3.0f+B)*(exp(57.551078393482776f/T)-1.0f) / (exp(4.799243348e-11f*freq/T)-1.0f) ;
}


__kernel void Seed(const float seed, __global uint *rng_state) {
   const int id = get_global_id(0) ;
   // Initialise random number generator
   mwc64x_state_t rng ;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(seed+id*PI,0.9999999f)*4294967296L), samplesPerStream) ;
   rng_state[2*id  ] = rng.x ;   // rng_state[2*GLOBAL]
   rng_state[2*id+1] = rng.c ;
}


__kernel void MakeProposal(__global uint  *rng_state,
                           __global float *P1,
                           __global float *P2,
                           __global float *DX
                          ) {
   // GLOBAL>=N, the number of sources
   const int id = get_global_id(0) ;
   mwc64x_state_t rng ;
   rng.x  =  rng_state[2*id  ] ;  
   rng.c  =  rng_state[2*id+1] ;
# if 0
   for(int i=id; i<NP; i+=GLOBAL) {                    // NP probably >> GLOBAL
         P2[i]  =  P1[i] ;  // TEST WITH CONSTANT GLOBAL PARAMETERS
   }
# else
   for(int i=id; i<NP; i+=GLOBAL) {                    // NP probably >> GLOBAL
      // P2[i]  =  P1[i]*(1.0f+STEP_SCALE*0.001f*RandC(&rng)) ;      // 
      P2[i]  =  P1[i] + STEP_SCALE*DX[i]*RandC(&rng) ; // OK ?
      // P2[i]  =  P1[i] + STEP_SCALE*DX[i]*(-0.5f+RandU(&rng)) ;  // OK ?
   }
# endif
   rng_state[2*id  ]  =  rng.x ;  
   rng_state[2*id+1]  =  rng.c ;
# if (HIER==1)
   P2[4] = clamp(P2[4], -0.99f, +0.99f) ;  // correlation
#endif
}


__kernel void LNP(__global float  *F,
                  __global float  *S_all,
                  __global float  *dS_all,
                  __global float  *G,           // all parameters, the proposal (P1 or P2)
                  __global float  *lnP_all) {
   const int id = get_global_id(0) ;
   if (id>=N) return ;                          // one work item per source, GLOBAL>=N
   float t, b, dT, dB, r, y ;
   real  lnP ;
   __global float *S   =  &(S_all[id*NF]) ;     // flux measurements of this source
   __global float *dS  =  &(dS_all[id*NF]) ;
   __global float *P   =  &(G[NG+3*id]) ;       // 3 parameters for this source
   lnP = 0.0f ;

#if 1
   // The probability --- current source parameters vs. the hyperdistribution
# if (HIER==1)    // NG=5 --- only hyperparameters for the joint (T, B) probability distribution
#  if (STUDENT==0)
   // Using 2D Gaussian probability distribution for (T, B)
   //  one source has X = [ I, T, beta], hyperparameters are  GP = [ T, B, dT, dB, rho(T,B) ]
   t   =   P[1] -  G[0] ;  // difference of proposed T and the hyperparameter T
   b   =   P[2] -  G[1] ;  // difference in beta
   dT  =           G[2] ;
   dB  =           G[3] ;
   r   =           G[4] ;
   lnP +=  (-0.5f/(1.0f-r*r)) * ( (t*t/(dT*dT)) + (b*b/(dB*dB)) - 2.0f*r*t*b/(dT*dB) )  -  log(2.0f*PI*dT*dB*sqrt(1.0f-r*r)) ;
#  else
   // assuming 2D Student distribution
   d    =          4.0f ;   //  d=3-1=2 ?? .... or 5-1=4 ??,   m=2 for (T, beta), d=Inf == normal distribution
   t    =   P[1] - G[0] ;   // difference of proposed T and the hyperparameter T
   b    =   P[2] - G[1] ;   // difference in beta
   dT   =          G[2] ;
   dB   =          G[3] ;
   r    =          G[4] ;
   // exponent is  -(d+m)/2 where  d=dof and  m=dimension of the vector = 2 for 2D distribution
   lnP += log(pow(1.0f+(t*t/(dT*dT) + b*b/(dB*dB) - 2.0f*r*t*b/(dT*dB))/(d*(1.0f-r*r)), -(d+2.0f)/2.0f)  /  ( 2.0f*PI*dT*dB*sqrt(1.0f-r*r) )  ) ;
#  endif
# endif
#endif

   
# if 1
   // The likelihood part, individual flux measurements
   r = 0.0f ;
   for(int iband=0; iband<NF; iband++) {
      y     =  P[0] * MBB_Function(F[iband], P[1], P[2]) ;
      y     =  (y-S[iband]) / dS[iband] ;
      r    +=  -0.5f*y*y - log(dS[iband]*sqrt(2.0f*PI)) ;
   } // for iband
   lnP += r ;
# endif
   
   
   // boundaries
   if ((P[0]<=0.0f)||(P[1]<TMIN)||(P[1]>TMAX)||(P[2]<BMIN)||(P[2]>BMAX))  lnP -= 1.001e30f ;

# if (USE_PRIORS>0)
# if (HIER==1)
   if (id==0) {
      // Probability from the prior distribution of hyperparameters
      //     0  1  2   3   4  
      //     T, B, dT, dB, rho
      lnP +=   -log(1.0f+(G[2]/2.5f)*(G[2]/2.5f))   -log(1.0f+(G[3]/2.5f)*(G[3]/2.5f)) ;
      lnP +=   ( (fabs(G[4])<1.0f) ?  log(1.0f-G[4]*G[4]) : -1.0e32f  ) ;
      if ((G[2]<0.0f)||(G[3]<0.0f))                             lnP -=  1.010e30f ;   // keep std positive
      if ((G[0]<TMIN)||(G[0]>TMAX)||(G[1]<BMIN)||(G[1]>BMAX))   lnP -=  1.100e30f ;
   }
# endif
#endif
      
   if (isfinite(lnP)) lnP_all[id] =  lnP ;
   else               lnP_all[id] = -1.0e30f ;

}

