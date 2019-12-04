
# define USE_PRIORS 1
# define DEBUG 1
# define PI    3.1415926536f

# include "mwc64x_rng.cl"
// Uniform random variates [0,1]
# define RandU(x)  (MWC64X_NextUint(x)/4294967295.0f)

# if 0
#  define RandC(x)  ((MWC64X_NextUint(x)/4294967295.0f)-0.5f)
# else
// Normal random variables  --- scaling to smaller absolute values important ???
//                              0.01f works... 1.0f does not even complete??
#  define RandC(x)  (sqrt(-2.0f*log(clamp(MWC64X_NextUint(x)/4294967295.0f, 1.0e-6f, 0.99999f)))*cos(2.0f*PI*(MWC64X_NextUint(x)/4294967295.0f)))
# endif

# define RMAX  0.999f       // maximum allowed correlation


// Robust Adaptive Metropolis (RAM)
//    HIER=0     no hierarchical part at all
//    HIER=1     hierarchical part  (T, B, dT, dB, rho(T,B))
//    HIER=2     hierarchical part  (I, T, B, dI, dT, dB, rIT, rIB, rTB)
//
//    STUDENT=0  using Gaussian probabiity distributions
//    STUDENT=1  using Student-t distributions
//
//    HRAM=0     basic MCMC for the hyperparameters
//    HRAM=1     use RAM also for hyperparameters but in groups of 2-3 parameters each
//    HRAM=2     use RAM for hyperparameters, all as a single group


#  define MBB_Function(freq,T,B) (pow(freq/1.199170e+12f,3.0f+B)*(exp(57.551078f/T)-1.0f)/(exp(4.7992433e-11f*freq/T)-1.0f))





__kernel void MCMC_RAM(__global    float  *F,         //   F[NF]
                       __global    float  *SO,        //   S[N, NF]
                       __global    float  *dSO,       //  dS[N, NF]
                       __global    float  *G_all,     //   G[NWG, [5|9]] -- current/initial/final state of h.p.
                       __global    float  *L_all,     //   L[NWG, NPIX, 3] -- current state
                       __global    float  *GS,        //  GS[NWG, 5, SAMPLES]
                       __global    float  *LS,        //   L[NWG, NPIX, 3, SAMPLES]
                       __global    float  *WRK,       //   WRK[NWG*3*N]
                       __global    float  *R_all)     //   R[NWG*N*6]
{
   // RAM -- Robust Adaptive Metropolis -- Vihola 2011
   const int id   =  get_global_id(0) ;
   const int lid  =  get_local_id(0) ;
   const int gid  =  get_group_id(0) ;   
   
   __local  float det ;
   __local  float GP[NG] ;                   // common to one work group, proposal
   __global float *L  = &(L_all[gid*N*3]) ;  // for this work group, NPIX*3 parameters = NPIX*(I,T,beta)
   __global float *G  = &(G_all[gid*NG]) ;   // hyperparameter vector of the current work group
   __global float *LP = &(WRK[gid*3*N]) ;    // room for (I,T,beta) propositions for this work group
   __global float *X, *Y, *dY ;
   __global float *R = &(R_all[gid*N*6]) ;   // cov matrix elements for each source, (I, T, beta) R[N,6]
   __local  float lnP_A[LOCAL] ;             // per-work item probability (lnP) values
   int   sample=0, ind ;
   float lnP, lnP0=-1.0e30f, lnA ;
   float r, t, s, b, tmp, y, dT, dB, k ;
   float   U[NG] ;          // random number for per-source updates - private
   __local float UU[NG] ;   // random number for hyperparameter updates - local
# if (STUDENT>0)
   float d ;
# endif
   int reject = 0 ;
   
   if (lid==0) det = 1.0f ;
   
   //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# if (HIER>0) // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  if (HIER==1)
   __local float S[3] ;
#  endif
#  if (HIER==2)
   __local float S[6], SI[6] ;
#  endif
#  if (HIER==1) && (HRAM==1)  // @@@   5 h.p. divided into two groups
   // (T, B) one group, (dT, dB, rho) second group
   __local float A[3], B[6], XU[6] ; // can reuse XU for both parts
   if (lid==0) {
      for(int i=0; i<3; i++) A[i] = 0.0f ;
      for(int i=0; i<6; i++) B[i] = 0.0f ;
      A[0] = 0.02f ;   A[2] = 0.02f ;   // diagonal entries
      B[0] = 0.02f ;   B[2] = 0.02f ;   B[5] = 0.02f ;
   }
#  endif   
#  if (HIER==1) && (HRAM==2) 
   // single group of 5 h.p.
   // covariance matrix of hyperparameters is 5x5, saved using A[15]
   __local float  A[15], XU[6] ;   // for NG=5  hyperparameters G == [ T, B, dT, dB, rho(T,B) ] 
   if (lid==0) {
      for(int i=0; i<15; i++) A[i] = 0.0f ;
      A[ 0] = 0.05f ;       // T
      A[ 2] = 0.05f ;       // B
      A[ 5] = 0.02f ;       // dT
      A[ 9] = 0.01f ;       // dB
      A[14] = 0.01f ;       // rho(T,B)
   }
#  endif
#  if (HIER==2) && (HRAM==1)                   // NG=9, RAM for three groups of three hyperparameters
   __local float A[6], B[6], C[6], XU[NG+1] ; // A for (I,T,B), B for (dI, dT, dB), C for rhos
   if (lid==0) {
      for(int i=0; i<6; i++)  { A[i] = 0.0f ;  B[i] = 0.0f ;  C[i] = 0.0f ; }
      A[0] = 0.1f ;  A[2] = 0.1f ;  A[5] = 0.1f ;
      B[0] = 0.1f ;  B[2] = 0.1f ;  B[5] = 0.1f ;
      C[0] = 0.1f ;  C[2] = 0.1f ;  C[5] = 0.1f ;
   }
#  endif
#  if (HIER==2) && (HRAM==2)           // single group of 9 h.p.
   __local float  A[45], XU[NG+1] ;   // for NG=9
   if (lid==0) {
      for(int i=0; i<45; i++)  A[i] = 0.0f ;
      // A is Cholesky factor of the covariance matrix of  G == [ I, T, B, dI, dT, dB, sIT, sIB, sTB ]
      // ... initialise diagonal with expected standard deviations of those parameters
      // A is stored in *row* order, skipping the empty upper diagonal part
      A[ 0]  =    0.1f ;     // I
      A[ 2]  =    0.1f ;     // T
      A[ 5]  =    0.1f ;     // B
      A[ 9]  =    0.1f ;     // dI
      A[14]  =    0.1f ;     // dT
      A[20]  =    0.1f ;     // dB
      A[27]  =    0.1f ;     // rho(I,T)
      A[35]  =    0.1f ;     // rho(I,B)
      A[44]  =    0.1f ;     // rho(T,B)   !!! ~std of the parameter that happens to be T-beta correlation
   }
#  endif
#  if (HIER==1) && (HRAM==0)  // 5 h.p. without RAM ---   (T, B, dT, dB, rho)
   float STEP[NG] = { 0.001f, 0.001f, 0.0005f, 0.0001f, 0.0001f } ;
#  endif
#  if (HIER==2) && (HRAM==0)  //  9 h.p. without RAM
   //                 I      T      B      dI     dT     dB     rIT    rIB    rTB   
   // float STEP[NG] = { 0.03f, 0.03f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f } ;
   // float STEP[NG] = { 0.10f, 0.10f, 0.05f, 0.10f, 0.10f, 0.10f, 0.01f, 0.01f, 0.01f } ;
   // float STEP[NG] = { 0.1f, 0.1f, 0.05f,    0.2f, 0.2f, 0.2f,    0.01f, 0.01f, 0.01f } ;
   // float STEP[NG] = { 0.05f, 0.05f, 0.02f,    0.2f, 0.2f, 0.2f,    0.01f, 0.01f, 0.01f } ;
   float STEP[NG] = { 0.005f, 0.005f, 0.002f,    0.02f, 0.02f, 0.02f,    0.003f, 0.003f, 0.003f } ;
#  endif
# endif  // initialisations for HIER >0 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   
   
   float8 CC ;   
   // Set initial matrix elements to global R[N,6] -- for (I, T, beta) of N individual sources
   for(int i=lid; i<N; i+=LOCAL) {
      R[6*i+0] = 0.05f ;   
      R[6*i+1] = 0.20f ;   R[6*i+2] = 0.50f ;
      R[6*i+3] = 0.02f ;   R[6*i+4] = 0.02f ;   R[6*i+5] = 0.04f ;      
   }      
   // Initialise random number generator
   mwc64x_state_t rng, rng0 ;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod((1.0+1.735197*id)*M_PI,0.9999999)*4294967296L), samplesPerStream) ;
   
   
   
   barrier(CLK_LOCAL_MEM_FENCE) ;
   barrier(CLK_GLOBAL_MEM_FENCE) ;
   
   
   
   
   
   
   
   for(int isample=0; isample<(BURNIN+SAMPLES*THIN); isample++) { // MCMC steps
      
      
      lnP = 0.0f ;
      
      
      // We need to be able to reset Rand() to the state before the RAM array updates
      rng0.x = rng.x ;  rng0.c = rng.c ;                            
      // we may have updated the global arrays G[] and L[]
      barrier(CLK_LOCAL_MEM_FENCE) ;      
      barrier(CLK_GLOBAL_MEM_FENCE) ;      
      
      
      // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# if (HIER>0)  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      // always collaborative random numbers if using hierarchical part --- U[] is private, UU local
      for(int i=lid; i<NG; i+=LOCAL) UU[i] = RandC(&rng) ;   // UU __local
      barrier(CLK_LOCAL_MEM_FENCE) ;
      
      
#  if (HIER==1) && (HRAM==0)  // NG=5, no RAM for hyperparameters
      if (lid==0) {
         // A is stored in *row* order  ==>  GP = G + A*U    ---- GP is local
         GP[0]  =   clamp(G[0]  * (1.0f + STEP[0]*UU[0]),     TMIN,    TMAX) ;  //   T 
         GP[1]  =   clamp(G[1]  * (1.0f + STEP[1]*UU[1]),     BMIN,    BMAX) ;  //   B 
         GP[2]  =   clamp(G[2]  * (1.0f + STEP[2]*UU[2]),    0.001f,  20.0f) ;  //  dT 
         GP[3]  =   clamp(G[3]  * (1.0f + STEP[3]*UU[3]),    0.001f,   3.0f) ;  //  dB 
         GP[4]  =   clamp(G[4]  +   STEP[4]*UU[4],    -RMAX,    RMAX) ;         // rTB 
      }      
#  endif
#  if (HIER==1) && (HRAM==1)  // NG=5, RAM for hyperparameter groups (T, B) and (dT, dB, rho)
      if (lid==0) {
         // (T, B) =  G + A*U(0,1)
         GP[0] = clamp( G[0] + A[0]*UU[0],                             TMIN,    TMAX  ) ;   // T
         GP[1] = clamp( G[1] + A[1]*UU[0] + A[2]*UU[1],                BMIN,    BMAX  ) ;   // B
         // (dT, dB, rho) = G + B*U(2,3,4)
         GP[2] = clamp( G[2] + B[0]*UU[2],                             0.001f,   20.0f) ;   // dT
         GP[3] = clamp( G[3] + B[1]*UU[2] + B[2]*UU[3],                0.001f,    3.0f) ;   // dB
         GP[4] = clamp( G[4] + B[3]*UU[2] + B[4]*UU[3] + B[5]*UU[4],  -RMAX,    RMAX  ) ;   // rho
      }
#  endif
#  if (HIER==1) && (HRAM==2) // NG=5, RAM applied to all h.p. as a single group,  [ T, B, dT, dB, rho ]
      // NG=5 hyperparameters, A is 5x5 matrix, storing the 15 lower diagonal elements
      if (lid==0) {  // A is stored in *row* order  ==>  GP = G + A*U
         GP[0] = G[0] + A[ 0]*UU[0] ;
         GP[1] = G[1] + A[ 1]*UU[0] + A[ 2]*UU[1]  ;
         GP[2] = G[2] + A[ 3]*UU[0] + A[ 4]*UU[1] + A[ 5]*UU[2]  ;
         GP[3] = G[3] + A[ 6]*UU[0] + A[ 7]*UU[1] + A[ 8]*UU[2] + A[ 9]*UU[3]  ;
         GP[4] = G[4] + A[10]*UU[0] + A[11]*UU[1] + A[12]*UU[2] + A[13]*UU[3] + A[14]*UU[4] ; 
      }
#  endif
      
      
      
#  if (HIER==2) && (HRAM==0)   // NG=9, no RAM for hyperparameters
      if (lid==0) {
         for(int i=0; i<NG; i++)   GP[i]  =  G[i] + STEP[i]*UU[i] ;
         GP[0] = clamp(GP[0],  0.001f,  1e10f ) ;
         GP[1] = clamp(GP[1],    TMIN,   TMAX ) ;
         GP[2] = clamp(GP[2],    BMIN,   BMAX ) ;
         GP[3] = clamp(GP[3],  0.001f, 999.0f ) ;    // dI -- std must be positive
         GP[4] = clamp(GP[4],  0.001f,  20.0f ) ;    // dT  
         GP[5] = clamp(GP[5],  0.001f,   3.0f ) ;    // dB
         GP[6] = clamp(GP[6], -RMAX,     RMAX ) ;    // correlations must be [-1,+1]
         GP[7] = clamp(GP[7], -RMAX,     RMAX ) ;
         GP[8] = clamp(GP[8], -RMAX,     RMAX ) ;
      }
#  endif
#  if (HIER==2) && (HRAM==1) // NG=9, RAM applied to three groups of three h.p. --> need UU[NG] random numbers
      if (lid==0) {   //    0  1  2    3   4   5     6    7    8   
         //                (I, T, B,   dI, dT, dB,   rIT, rIB, rTB)
         GP[0]  =   clamp(G[0]  +   A[0]*UU[0],                            1.0e-10f,+1e10f) ;  // I
         GP[1]  =   clamp(G[1]  +   A[1]*UU[0] + A[2]*UU[1],               TMIN,      TMAX) ;  // T
         GP[2]  =   clamp(G[2]  +   A[3]*UU[0] + A[4]*UU[1] + A[5]*UU[2],  BMIN,      BMAX) ;  // B
         GP[3]  =   clamp(G[3]  +   B[0]*UU[3],                            0.001f, 1.0e10f) ;  // dI
         GP[4]  =   clamp(G[4]  +   B[1]*UU[3] + B[2]*UU[4],               0.001f,   20.0f) ;  // dT
         GP[5]  =   clamp(G[5]  +   B[3]*UU[3] + B[4]*UU[4] + B[5]*UU[5],  0.001f,    3.0f) ;  // dB
         GP[6]  =   clamp(G[6]  +   C[0]*UU[6],                           -RMAX,      RMAX) ;  // rIT
         GP[7]  =   clamp(G[7]  +   C[1]*UU[6] + C[2]*UU[7],              -RMAX,      RMAX) ;  // rIB
         GP[8]  =   clamp(G[8]  +   C[3]*UU[6] + C[4]*UU[7] + C[5]*UU[8], -RMAX,      RMAX) ;  // rTB
      }
#  endif
#  if (HIER==2) && (HRAM==2) // NG=9, RAM applied to all h.p. as a single group
      // NG=9, A is 9 x 9 row-order matrix, only lower diagonal 45 elements stored
      if (lid==0) {
         ind = 0 ;
         for(int i=0; i<NG; i++) {  // loop over rows
            r = 0.0f ;   for(int j=0; j<=i; j++)  {  r += A[ind]*UU[j] ;  ind += 1 ; }
            GP[i] = G[i] + r ;
         }
         GP[0]  =  clamp(GP[0],  1.0e-10f,  1.0e10f )  ;  // I
         GP[1]  =  clamp(GP[1],  TMIN,         TMAX )  ;  // T
         GP[2]  =  clamp(GP[2],  BMIN,         BMAX )  ;  // B
         GP[3]  =  clamp(GP[3],  0.001f,    1.0e10f )  ;  // sI
         GP[4]  =  clamp(GP[4],  0.001f,      20.0f )  ;  // sT
         GP[5]  =  clamp(GP[5],  0.001f,       3.0f )  ;  // sB
         GP[6]  =  clamp(GP[6], -RMAX,         RMAX )  ;  // rho(I,T)
         GP[7]  =  clamp(GP[7], -RMAX,         RMAX )  ;  // rho(I,B)
         GP[8]  =  clamp(GP[8], -RMAX,         RMAX )  ;  // rho(T,B)
      }
#  endif
      
      
      
      barrier(CLK_LOCAL_MEM_FENCE) ;      
      barrier(CLK_GLOBAL_MEM_FENCE) ;      
      // rest of the hierarchical part precomputations
      
      
      
#  if (HIER==1)
      // Probability for (T, B) distribution using 5 hyperparameters
      //              0    1    2     3    4     
      //   GP[5] = [  T,   B,   dT,   dB   rTB ] 
      //
      //   Covariance matrix for (T, B) only,      S =     TT   TB        0   1
      //   (S stored in *column* order...)                 TB   BB        1   2
      if (lid==0) {
         S[0]   =   GP[2]*GP[2] ;        // var(T)
         S[1]   =   GP[4]*GP[2]*GP[3] ;  // cov(T,B)
         S[2]   =   GP[3]*GP[3] ;        // var(B)
      }
      barrier(CLK_LOCAL_MEM_FENCE) ;  // all work items need the updated SI
#  endif
      
      
#  if (HIER==2)
      // Probability for (I, T, B) using 9 hyperparameters
      //            0   1   2   3   4   5   6    7    8    
      //  GP[9] = [ I0, T0, B0, sI, sT, sB, rIT, rIB, rTB ]
      //  The covariance matrix S[] and its inverse SI[] are stored in column order!
      //
      //          II                  0   1   2
      //    S =   IT  TT              1   3   4
      //          IB  TB   BB         2   4   5
      if (lid==0) {
         // S[] and SI[] are local arrays  .... use lid==0 to update
         S[0]   =  GP[3]*GP[3] ;               // var(I)
         S[3]   =  GP[4]*GP[4] ;               // var(T)
         S[5]   =  GP[5]*GP[5] ;               // var(beta)
         S[1]   =  GP[3]*GP[4]* GP[6] ;        // cov(I, T)
         S[2]   =  GP[3]*GP[5]* GP[7] ;        // cov(I, beta)
         S[4]   =  GP[4]*GP[5]* GP[8] ;        // cov(T, beta)
         // determinant of the covariance matrix  -- local variable !!
         // S0*(S3*S5-S4^2)-S1*(S1*S5-S2*S4)+S2*(S1*S4-S2*S3)
         det    =  S[0]*(S[3]*S[5]-S[4]*S[4]) + S[1]*(S[2]*S[4]-S[1]*S[5]) + S[2]*(S[1]*S[4]-S[2]*S[3]) ;
#   if 0   // <<<<<<<<<<<<<< THIS NEEDS TO BE ON ?
         while (det<1.0e-20f) {   // #1 
#    if 1
            GP[6] *= 0.95 ;    GP[7] *= 0.95 ;      GP[8] *= 0.97 ;
#    else  // this is worse, at least for HIER=2 both HRAM=1 and HRAM=2
            GP[6] *= 0.98f ;   GP[7] *= 0.98f ;     GP[8] *= 0.990f ;
            GP[3] *= 1.01f ;   GP[4] *= 1.01f ;     GP[5] *= 1.005f ;
#    endif
            S[0]   =  GP[3]*GP[3] ;               // var(I)
            S[3]   =  GP[4]*GP[4] ;               // var(T)
            S[5]   =  GP[5]*GP[5] ;               // var(beta)
            S[1]   =  GP[3]*GP[4]* GP[6] ;        // sigma(I, T)
            S[2]   =  GP[3]*GP[5]* GP[7] ;        // sigma(I, beta)
            S[4]   =  GP[4]*GP[5]* GP[8] ;        // sigma(T, beta)
            det    =  S[0]*(S[3]*S[5]-S[4]*S[4]) + S[1]*(S[2]*S[4]-S[1]*S[5]) + S[2]*(S[1]*S[4]-S[2]*S[3]) ;
         }
#   else
         if (det<1.0e-20f) {
            det = 1.0e-10f ; lnP -= 1.01e30f ;
         }
#   endif
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
      }
#  endif // HIER==2
      barrier(CLK_LOCAL_MEM_FENCE) ;      
      barrier(CLK_GLOBAL_MEM_FENCE) ;      
# endif // end of HIER>0 calculations of hyperparameter propositions ++++++++++++++++++++++++++++++++++++++++++++

      
      
      
      
      
      
      
      
      for(int ipix=lid; ipix<N; ipix+=LOCAL) {         
         
         //  each pixel always handled by the same work item
         X   =   &LP[ipix*3]  ;   // propositions on (I, T, beta) for the current pixel
         Y   =   &SO[ipix*NF]  ;  // fluxes for current pixel
         dY  =  &dSO[ipix*NF]  ;         
         // Propositions for the individual (I, T, b)
         U[0] =  RandC(&rng) ;   U[1] = RandC(&rng) ;   U[2] = RandC(&rng) ;  // U[] is _private
         X[0] =  clamp(L[3*ipix+0]   +   U[0]*R[6*ipix+0],                                         1e-10f, 1e10f) ;
         X[1] =  clamp(L[3*ipix+1]   +   U[0]*R[6*ipix+1]  + U[1]*R[6*ipix+2],                     TMIN,   TMAX) ;
         X[2] =  clamp(L[3*ipix+2]   +   U[0]*R[6*ipix+3]  + U[1]*R[6*ipix+4] + U[2]*R[6*ipix+5],  BMIN,   BMAX) ;
         // X is private pointer to __global arrays, L is private pointer to __global array
         
         
         
         // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# if (HIER>0)      // probabilities related to hyperparameter ++++++++++++++++++++++++++++++++++++++++++++++++++++
         
         
#  if (HIER==1)    // NG=5 --- only hyperparameters for the joint (T, B) probability distribution
#   if (STUDENT==0)
         // Using 2D Gaussian probability distribution for (T, B)
         //  one source has X = [ I, T, beta], hyperparameters are  GP = [ T, B, dT, dB, rho(T,B) ]
         t   =   X[1] - GP[0] ;  // difference of proposed T and the hyperparameter T
         b   =   X[2] - GP[1] ;  // difference in beta
         dT  =          GP[2] ;
         dB  =          GP[3] ;
         r   =          GP[4] ;
         lnP +=  (-0.5f/(1.0f-r*r)) * ( (t*t/(dT*dT)) + (b*b/(dB*dB)) - 2.0f*r*t*b/(dT*dB) )
           -     log(2.0f*PI*dT*dB*sqrt(1.0f-r*r)) ;
#   else
         // assuming 2D Student distribution
         d    =           4.0f ;  //  d=3-1=2 ?? .... or 5-1=4 ??,   m=2 for (T, beta), d=Inf == normal distribution
         t    =   X[1] - GP[0] ;  // difference of proposed T and the hyperparameter T
         b    =   X[2] - GP[1] ;  // difference in beta
         dT   =          GP[2] ;
         dB   =          GP[3] ;
         r    =          GP[4] ;
         // exponent is  -(d+m)/2 where  d=dof and  m=dimension of the vector = 2 for 2D distribution
         lnP += log(pow(1.0f+(t*t/(dT*dT) + b*b/(dB*dB) - 2.0f*r*t*b/(dT*dB))/(d*(1.0f-r*r)), -(d+2.0f)/2.0f)
                    /  ( 2.0f*PI*dT*dB*sqrt(1.0f-r*r) )  ) ;
#   endif
#  endif
         
         
#  if (HIER==2)
#   if (STUDENT==0)  
         // assuming 3D multinormal distribution - here using GP = [ I, T, Beta, ...]
         // note that SI does no longer contain uppers diagonal elements => only SI[0]-SI[5]
         // X'.SI.X =  x2 * (SI5*x2+SI4*x1+SI2*x0) +  x1 * (SI4*x2+SI3*x1+SI1*x0)  +   x0 * (SI2*x2+SI1*x1+SI0*x0)
         // denominator  sqrt(det) *(2*pi)^1.5 =>   -0.5*log(det) - 1.5*log(2*pi)
         // det = as calculated above
         tmp        = -0.5f * ((X[2]-GP[2]) * ( SI[5]*(X[2]-GP[2]) + SI[4]*(X[1]-GP[1]) + SI[2]*(X[0]-GP[0]) ) + 
                               (X[1]-GP[1]) * ( SI[4]*(X[2]-GP[2]) + SI[3]*(X[1]-GP[1]) + SI[1]*(X[0]-GP[0]) ) + 
                               (X[0]-GP[0]) * ( SI[2]*(X[2]-GP[2]) + SI[1]*(X[1]-GP[1]) + SI[0]*(X[0]-GP[0]) )) ;
         lnP  +=  tmp - 0.5f*log(det) - 1.5f*log(2.0f*PI) ;
#   else
         // assuming 3D Student t -distribution
         d    =  8.0f ;
         tmp  =  log(pow(1.0f + 
                         (1.0f/d)* ((X[2]-GP[2]) * ( SI[5]*(X[2]-GP[2]) + SI[4]*(X[1]-GP[1]) + SI[2]*(X[0]-GP[0]) ) + 
                                    (X[1]-GP[1]) * ( SI[4]*(X[2]-GP[2]) + SI[3]*(X[1]-GP[1]) + SI[1]*(X[0]-GP[0]) ) + 
                                    (X[0]-GP[0]) * ( SI[2]*(X[2]-GP[2]) + SI[1]*(X[1]-GP[1]) + SI[0]*(X[0]-GP[0]) )
                                   ), -(d+3.0f)/2.0f)    /   sqrt(det)) ;
         lnP += tmp ;         
#   endif
#  endif
         
         
#  if (DEBUG>0)
         if (isfinite(tmp)) { ; } else {
            printf("GLOBAL  lid=%d, tmp=%10.3e, det=%10.3e\n", lid, tmp, det) ;
            printf("  X:    %10.3e %10.3e %10.3e\n", X[0], X[1], X[2]) ;
            printf("  GP:  ") ;  for(int i=0; i<NG; i++) printf(" %10.3e", GP[i]) ;   printf("\n") ;
#   if (HIER==2)
            printf("  SI:  ") ;  for(int i=0; i<6; i++)  printf(" %10.3e", SI[i])  ;  printf("\n") ;
#   endif
         }
#  endif         
# endif  // HIER>0+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

         
                               
         // The likelihood part, individual flux measurements
         tmp = 0.0f ;
         for(int iband=0; iband<NF; iband++) {
            y     =  X[0] * MBB_Function(F[iband], X[1], X[2]) ;
            y     =  (y-Y[iband]) / dY[iband] ;
            tmp  +=  -0.5f*y*y - log(dY[iband]*sqrt(2.0f*PI)) ;
         } // for iband
         lnP += tmp ;
         
         
         
#if 1
         if ((X[0]<=0.0f)||(X[1]<=TMIN)||(X[1]>=TMAX)||(X[2]<=BMIN)||(X[2]>=BMAX)) {
            if (gid==0) {
               if (reject>1e6) {
                  printf("   %4d  T=%5.2f  B=%5.2f  %10.3e %10.3e %10.3e %10.3e %10.3e\n", 
                         ipix, X[1], X[2], R[6*ipix+0], R[6*ipix+1], R[6*ipix+2], R[6*ipix+3], R[6*ipix+4], R[6*ipix+5]) ;
               }
            }
            lnP -= 1.001e30f ;
         }
#endif
         
      } // for ipix
      
      
      
      
      
      
      // now lnP is the contribution computed by the current work item
      lnP_A[lid] = lnP ;                   // lnP_A is local array
      barrier(CLK_LOCAL_MEM_FENCE) ;

           
      
      if (lid==0) {                        // work item lid==0 sums up all contributions
         lnP = 0.0f ;
         for(int i=0; i<LOCAL; i++)  lnP += lnP_A[i] ;
# if (USE_PRIORS>0)
#  if (HIER==1)
         // Probability from the prior distribution of hyperparameters
         lnP +=  -log(1.0f+(GP[2]/2.5f)*(GP[2]/2.5f)) - log(1.0f+(GP[3]/2.5f)*(GP[3]/2.5f)) ;
         lnP +=   (   (fabs(GP[4])<1.0f) ?  log(1.0f-GP[4]*GP[4]) : -1.0e32f   ) ;           
         if ((GP[2]<0.0f)||(GP[3]<0.0f))                                  lnP -=  1.010e30f ;   // keep std positive
         if ((GP[0]<=TMIN)||(GP[0]>=TMAX)||(GP[1]<=BMIN)||(GP[1]>=BMAX))  lnP -=  1.100e30f ;
#  endif
# endif
         lnP_A[0] = lnP ;                  // save the total lnP
         lnP_A[1] = RandU(&rng) ;          // save u for the acceptance test
      }
      barrier(CLK_LOCAL_MEM_FENCE) ;       // all work items now have upddated lnP_A
      lnP = lnP_A[0] ;                     // final lnP to all work items
      barrier(CLK_GLOBAL_MEM_FENCE) ;      // ???
      lnA = lnP-lnP0 ;      
                  
         

      if ((det>1.0e-30f)&&(isfinite(lnP))) {
         ;
      } else {  // make sure step is not accepted
         det = 1.0e-30f ;   lnA = -1.0e32f ;    lnP -= 1.0e32f ;
      }

                  
      
      if (lnA>log(lnP_A[1])) {  // MCMC step accepted
         if (lid==0) {
#  if (HIER>0)
            for(int i=0; i<NG;  i++)   G[i] = GP[i] ;  // update to global array G_all
#  endif
            for(int i=0; i<3*N; i++)   L[i] = LP[i] ;
         } // accept or not
         lnP0    =  lnP ;  // everyone needs lnP0 ... to compute lnA that is used to compute k for  RAM
         reject  = 0 ;
      } else {
         reject += 1 ;
      }
      
      
      barrier(CLK_LOCAL_MEM_FENCE) ;    
      barrier(CLK_GLOBAL_MEM_FENCE) ;    // all work items aware of the updated L and G arrays

      if (gid==0) {
         if (reject>1e3) {
            printf("R: %10.3e %10.3e  %10.3e %10.3e  %10.3e  d %10.3e rej %6d lnP0 %10.3e lnP %10.3e\n",
                   GP[0], GP[1], GP[2], GP[3], GP[4], det, reject, lnP0, lnP) ;
         }
      }
      
      
      // return to previous state to get the same random numbers U as used above
      //  for lid==0 it is before GP generation, for others it is before local (I, T, beta) generation
      rng.x =  rng0.x ;  rng.c = rng0.c  ;
      
      

      
      
# if (HIER>0) // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      // Update matrix R --- this is used for all HIER and all HRAM values (part of local RAM too!)
      //   S * [   I + eeta * (alpha - alpha*) * U*U^T / ||U||   ] * S^T
      //   eeta   = min{ 1, d*n^(-2/3) },    alpha* = 0.234,   d = dimensionality == 3 here
      k   =  min(1.0f, NG*pow(isample+1.0f,-0.66667f))   *   (min(1.0f,exp(lnA))-0.234f) ;
      // always collaborative random numbers for hyperparameters
      for(int i=lid; i<NG; i+=LOCAL)  UU[i] = RandC(&rng) ;
      barrier(CLK_LOCAL_MEM_FENCE) ;      
      

      
#  if (HIER==1) && (HRAM==1) // NG=5, RAM applied to h.p. groups (T,B) and (dT, dB,rho)   @@@
      // in this case we calculate Cholesky factors directly, not via rank-one updates
      if (lid==0) {
         // (1) calculate normalised vector UU
         r = 0.0f ;   for(int i=0; i<NG; i++) r += UU[i]*UU[i] ;   r = 1.0f/sqrt(r) ;
         for(int i=0; i<NG; i++) UU[i] *= r ;          // normalised U
         // (2) Update 2-element Cholesky factor A -- using random numbers UU[0] and UU[1]
         //     start by calculating the matrix elements of  the new A*A'
         XU[0] = A[0]* A[0]*(UU[0]*UU[0]*k+1.0f) ;
         XU[1] = A[0]* A[1]*(UU[0]*UU[0]*k+1.0f) + A[0]*A[2]*UU[0]*UU[1]*k ;
         XU[2] = A[2]*(A[2]*(UU[1]*UU[1]*k+1.0f) + A[1]*UU[0]*UU[1]*k)+A[1]*(A[1]*(UU[0]*UU[0]*k+1)+A[2]*UU[0]*UU[1]*k) ;
         // then the new elements of A
         A[0]  = sqrt(XU[0]) ;
         A[1]  = XU[1]/A[0] ;
         A[2]  = sqrt(XU[2]-A[1]*A[1]) ;
         // (3) update 3-element Cholesky factor B  --- using random numbers UU[2], UU[3], UU[4]
         //     elements of the new B*B' matrix  =  B*(I+k*U*U')*B'
         XU[0] = B[0]* B[0]*(UU[2]*UU[2]*k+1.0f) ;
         XU[1] = B[0]* B[1]*(UU[2]*UU[2]*k+1.0f)+B[0]* B[2]*UU[2]*UU[3]*k ;
         XU[2] = B[2]*(B[2]*(UU[3]*UU[3]*k+1.0f)+B[1]*UU[2]*UU[3]*k)+B[1]*(B[1]*(UU[2]*UU[2]*k+1)+B[2]*UU[2]*UU[3]*k) ;
         XU[3] = B[0]* B[3]*(UU[2]*UU[2]*k+1.0f)+B[0]* B[5]*UU[2]*UU[4]*k+B[0]*B[4]*UU[2]*UU[3]*k ;
         XU[4] = B[4]*(B[2]*(UU[3]*UU[3]*k+1.0f)+B[1]*UU[2]*UU[3]*k)+B[3]*(B[1]*(UU[2]*UU[2]*k+1)+B[2]*UU[2]*UU[3]*k)+B[5]*(B[2]*UU[3]*UU[4]*k+B[1]*UU[2]*UU[4]*k) ;
         XU[5] = B[5]*(B[5]*(UU[4]*UU[4]*k+1.0f)+B[4]*UU[3]*UU[4]*k+B[3]*UU[2]*UU[4]*k)+B[4]*(B[4]*(UU[3]*UU[3]*k+1.0f)+B[5]*UU[3]*UU[4]*k+B[3]*UU[2]*UU[3]*k)+B[3]*(B[3]*(UU[2]*UU[2]*k+1.0f)+B[5]*UU[2]*UU[4]*k+B[4]*UU[2]*UU[3]*k) ;
         // updated elements of B
         B[0]  = sqrt(XU[0]) ;
         B[1]  = XU[1] / B[0] ;
         B[2]  = sqrt(max(XU[2]-B[1]*B[1], 1.0e-10f)) ;
         B[3]  = XU[3]/B[0] ;
         B[4]  = (XU[4]-B[3]*B[1])/B[2] ;
         B[5]  = sqrt(max(XU[5]-B[3]*B[3]-B[4]*B[4], 1.0e-10f)) ;
      }
#  endif

      
#  if (HIER==2) && (HRAM==1) // NG=9, RAM applied in groups of three
      if (lid==0) {  // update __local arrays A[], B[], C[]
         // calculate normalised vector UU
         r = 0.0f ; for(int i=0; i<NG; i++)  r += UU[i]*UU[i] ;  r = 1.0/sqrt(r) ;
         for(int i=0; i<NG; i++) UU[i] *= r ;          // normalised U
         // update Cholesky factors with direct calculation
         XU[0] = A[0]* A[0]*(UU[0]*UU[0]*k+1.0f) ;
         XU[1] = A[0]* A[1]*(UU[0]*UU[0]*k+1.0f)+A[0]*A[2]*UU[0]*UU[1]*k ;
         XU[2] = A[2]*(A[2]*(UU[1]*UU[1]*k+1.0f)+A[1]*UU[0]*UU[1]*k)+A[1]*(A[1]*(UU[0]*UU[0]*k+1)+A[2]*UU[0]*UU[1]*k) ;
         XU[3] = A[0]* A[3]*(UU[0]*UU[0]*k+1.0f)+A[0]*A[5]*UU[0]*UU[2]*k+A[0]*A[4]*UU[0]*UU[1]*k ;
         XU[4] = A[4]*(A[2]*(UU[1]*UU[1]*k+1.0f)+A[1]*UU[0]*UU[1]*k)+A[3]*(A[1]*(UU[0]*UU[0]*k+1)+A[2]*UU[0]*UU[1]*k)+A[5]*(A[2]*UU[1]*UU[2]*k+A[1]*UU[0]*UU[2]*k) ;
         XU[5] = A[5]*(A[5]*(UU[2]*UU[2]*k+1.0f)+A[4]*UU[1]*UU[2]*k+A[3]*UU[0]*UU[2]*k)+A[4]*(A[4]*(UU[1]*UU[1]*k+1.0f)+A[5]*UU[1]*UU[2]*k+A[3]*UU[0]*UU[1]*k)+A[3]*(A[3]*(UU[0]*UU[0]*k+1.0f)+A[5]*UU[0]*UU[2]*k+A[4]*UU[0]*UU[1]*k) ;
         A[0]  = sqrt(XU[0]) ;
         A[1]  = XU[1] / A[0] ;
         A[2]  = sqrt(max(XU[2]-A[1]*A[1], 1.0e-12f)) ;
         A[3]  = XU[3]/A[0] ;
         A[4]  = (XU[4]-A[3]*A[1])/A[2] ;
         A[5]  = sqrt(max(XU[5]-A[3]*A[3]-A[4]*A[4], 1.0e-12f)) ;
         //
         XU[0] = B[0]* B[0]*(UU[3]*UU[3]*k+1.0f) ;
         XU[1] = B[0]* B[1]*(UU[3]*UU[3]*k+1.0f)+B[0]*B[2]*UU[3]*UU[4]*k ;
         XU[2] = B[2]*(B[2]*(UU[4]*UU[4]*k+1.0f)+B[1]*UU[3]*UU[4]*k)+B[1]*(B[1]*(UU[3]*UU[3]*k+1)+B[2]*UU[3]*UU[4]*k) ;
         XU[3] = B[0]* B[3]*(UU[3]*UU[3]*k+1.0f)+B[0]*B[5]*UU[3]*UU[5]*k+B[0]*B[4]*UU[3]*UU[4]*k ;
         XU[4] = B[4]*(B[2]*(UU[4]*UU[4]*k+1.0f)+B[1]*UU[3]*UU[4]*k)+B[3]*(B[1]*(UU[3]*UU[3]*k+1)+B[2]*UU[3]*UU[4]*k)+B[5]*(B[2]*UU[4]*UU[5]*k+B[1]*UU[3]*UU[5]*k) ;
         XU[5] = B[5]*(B[5]*(UU[5]*UU[5]*k+1.0f)+B[4]*UU[4]*UU[5]*k+B[3]*UU[3]*UU[5]*k)+B[4]*(B[4]*(UU[4]*UU[4]*k+1.0f)+B[5]*UU[4]*UU[5]*k+B[3]*UU[3]*UU[4]*k)+B[3]*(B[3]*(UU[3]*UU[3]*k+1.0f)+B[5]*UU[3]*UU[5]*k+B[4]*UU[3]*UU[4]*k) ;
         B[0]  = sqrt(XU[0]) ;
         B[1]  = XU[1] / B[0] ;
         B[2]  = sqrt(max(XU[2]-B[1]*B[1], 1.0e-12f)) ;
         B[3]  = XU[3]/B[0] ;
         B[4]  = (XU[4]-B[3]*B[1])/B[2] ;
         B[5]  = sqrt(max(XU[5]-B[3]*B[3]-B[4]*B[4], 1.0e-12f)) ;
         //
         XU[0] = C[0]* C[0]*(UU[6]*UU[6]*k+1.0f) ;
         XU[1] = C[0]* C[1]*(UU[6]*UU[6]*k+1.0f)+C[0]*C[2]*UU[6]*UU[7]*k ;
         XU[2] = C[2]*(C[2]*(UU[7]*UU[7]*k+1.0f)+C[1]*UU[6]*UU[7]*k)+C[1]*(C[1]*(UU[6]*UU[6]*k+1)+C[2]*UU[6]*UU[7]*k) ;
         XU[3] = C[0]* C[3]*(UU[6]*UU[6]*k+1.0f)+C[0]*C[5]*UU[6]*UU[8]*k+C[0]*C[4]*UU[6]*UU[7]*k ;
         XU[4] = C[4]*(C[2]*(UU[7]*UU[7]*k+1.0f)+C[1]*UU[6]*UU[7]*k)+C[3]*(C[1]*(UU[6]*UU[6]*k+1)+C[2]*UU[6]*UU[7]*k)+C[5]*(C[2]*UU[7]*UU[8]*k+C[1]*UU[6]*UU[8]*k) ;
         XU[5] = C[5]*(C[5]*(UU[8]*UU[8]*k+1.0f)+C[4]*UU[7]*UU[8]*k+C[3]*UU[6]*UU[8]*k)+C[4]*(C[4]*(UU[7]*UU[7]*k+1.0f)+C[5]*UU[7]*UU[8]*k+C[3]*UU[6]*UU[7]*k)+C[3]*(C[3]*(UU[6]*UU[6]*k+1.0f)+C[5]*UU[6]*UU[8]*k+C[4]*UU[6]*UU[7]*k) ;
         C[0]  = sqrt(XU[0]) ;
         C[1]  = XU[1] / C[0] ;
         C[2]  = sqrt(max(XU[2]-C[1]*C[1], 1.0e-12f)) ;
         C[3]  = XU[3]/C[0] ;
         C[4]  = (XU[4]-C[3]*C[1])/C[2] ;
         C[5]  = sqrt(max(XU[5]-C[3]*C[3]-C[4]*C[4], 1.0e-12f)) ;
      }
#  endif
      
      
#  if (HIER>0) && (HRAM==2)     // RAM with 5x5 or 9x9 hierarchical part == all h.p. together
      if (lid==0) {
         r = 1.0e-32f ; for(int i=0; i<NG; i++) r += UU[i]*UU[i] ;  r = 1.0/sqrt(r) ;
         for(int i=0; i<NG; i++) UU[i] *= r ;          // normalised U
         // We use rank-one update on the existing Cholesky factor A
         // based on the fact that  A*A' =  A*A' + x*x'. With already normalised U,
         // the update is  A*A' =  A *  (I + k*U*U') * A'  = A*A'  + A*k*U*U'*A'
         //                     =  A*A'  +  (sqrt(k)*A*U) * (sqrt(k)*A*U)'
         //                 =>  x == sqrt(|k|)*A*U  .... if k<0, use CholeskyDown(A, x)
         // (1) calculate the vector XU == sqrt(k)*A*U, upper triangle of A is zeros
         r = sqrt(fabs(k)) ;  
         ind = 0 ;
         for(int i=0; i<NG; i++) {
            XU[i]  =  0.0f ;  
            for(int j=0; j<=i; j++) { XU[i] += A[ind]*UU[j] ;  ind += 1 ;}
            XU[i] *=  r ;            
         }
         XU[NG] = k ;  // just to let everyone know the sign of k ???
      }
      barrier(CLK_LOCAL_MEM_FENCE) ;      
      if (lid==0) {   
         // A stored in row order =>  A[j,i] = A[((j+1)*j)/2 + i]
         if (k>0.0f) {   // Cholesky up
            for(int i=0; i<NG; i++) {
               s                 =   A[((i+1)*i)/2+i] ;   // A[i,i] diagonal entry
               r                 =   sqrt(s*s + XU[i]*XU[i]) ;
               t                 =   r / s ;
               s                 =   XU[i] / s ;
               A[((i+1)*i)/2+i]  =   r ;
               for(int j=i+1; j<NG; j++) {
                  A[((j+1)*j)/2 + i]  =  ( A[((j+1)*j)/2 + i] + s *  XU[j]) / t ;
                  XU[j]               =    t * XU[j]          - s *  A[((j+1)*j)/2 + i] ;
               }
            }
         } else {       // Cholesky down
            for(int i=0; i<NG; i++) {
               s                 =   A[((i+1)*i)/2+i] ;   // A[i,i] diagonal entry
               r                 =   sqrt(s*s - XU[i]*XU[i]) ;
               t                 =   r / s ;
               s                 =   XU[i] / s ;
               A[((i+1)*i)/2+i]  =   r ;
               for(int j=i+1; j<NG; j++) {
                  A[((j+1)*j)/2 + i]  =  ( A[((j+1)*j)/2 + i] - s *  XU[j]) / t ;
                  XU[j]               =    t * XU[j]          - s *  A[((j+1)*j)/2 + i] ;
               }
            }
         }
      }
#  endif
      
# endif  // end of HIER>0 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      
      
      
      // Different k for the local updates --- NG -> 3.0  ??
      // k   =  min(1.0f, 3.0f*pow(isample+1.0f,-0.66667f))   *   (min(1.0f,exp(lnA))-0.234f) ;
      k   =  min(1.0f, 3.0f*pow(isample+1.0f,-0.66667f))   *   (min(1.0f,exp(lnA))-0.234f) ;

      
      // RAM updates for the (I, T, beta) of individual sources  --- CC is private, U is private
      for(int ipix=lid; ipix<N; ipix+=LOCAL) {
         // per-source random numbers (each work item separately)
         U[0]    =  RandC(&rng) ;     U[1] = RandC(&rng) ;     U[2] = RandC(&rng) ;    
         r       =  sqrt(U[0]*U[0]+U[1]*U[1]+U[2]*U[2]) ;
         U[0]   /=  r ;   U[1]   /=  r ;   U[2]   /=  r ;
         // Now  C  =  S*(I+k*U*U')*S'   =  S*S',  C is symmetric and positive definite
         //           C0  C1  C3        s0  s1  s3 
         //           C1  C2  C4        s1  s2  s4 
         //           C3  C4  C5        s3  s4  s5 
         //       s0^2*(k*u0^2+1)
         CC.s0 = R[6*ipix+0]*R[6*ipix+0]*(U[0]*U[0]*k+1.0f) ;
         //       k*s0*s2*u0*u1 + s0*s1*(k*u0^2+1)
         CC.s1 = R[6*ipix+0]*R[6*ipix+1]*(U[0]*U[0]*k+1.0f)  + R[6*ipix+0]*R[6*ipix+2]*U[0]*U[1]*k ;
         //      s2*(s2*(k*u1^2+1)+k*s1*u0*u1)                                        + s1*(k*s2*u0*u1+s1*(k*u0^2+1))
         CC.s2 = R[6*ipix+2]*(R[6*ipix+2]*(U[1]*U[1]*k+1.0f) + R[6*ipix+1]*U[0]*U[1]*k) + R[6*ipix+1]*(R[6*ipix+1]*(U[0]*U[0]*k+1.0f)+R[6*ipix+2]*U[0]*U[1]*k) ;
         //      s0*s3*(k*u0^2+1)                            + k*s0*s5*u0*u2                        +  k*s0*s4*u0*u1 
         CC.s3 = R[6*ipix+0]*R[6*ipix+3]*(U[0]*U[0]*k+1.0f)  + R[6*ipix+0]*R[6*ipix+5]*U[0]*U[2]*k  +  R[6*ipix+0]*R[6*ipix+4]*U[0]*U[1]*k ;
         //      s4*(s2*(k*u1^2+1)+k*s1*u0*u1)                                         +  s3*(k*s2*u0*u1+s1*(k*u0^2+1))                                         +  s5*(k*s2*u1*u2+k*s1*u0*u2)
         CC.s4 = R[6*ipix+4]*(R[6*ipix+2]*(U[1]*U[1]*k+1.0f)+R[6*ipix+1]*U[0]*U[1]*k)  +  R[6*ipix+3]*(R[6*ipix+1]*(U[0]*U[0]*k+1.0f)+R[6*ipix+2]*U[0]*U[1]*k)  +  R[6*ipix+5]*(R[6*ipix+2]*U[1]*U[2]*k+R[6*ipix+1]*U[0]*U[2]*k) ;
         //      s5*(s5*(k*u2^2+1)+k*s4*u1*u2+k*s3*u0*u2)                                                     +  s4*(k*s5*u1*u2+s4*(k*u1^2+1)+k*s3*u0*u1)                                                    + s3*(k*s5*u0*u2+k*s4*u0*u1+s3*(k*u0^2+1))  
         CC.s5 = R[6*ipix+5]*(R[6*ipix+5]*(U[2]*U[2]*k+1.0f)+R[6*ipix+4]*U[1]*U[2]*k+R[6*ipix+3]*U[0]*U[2]*k) + R[6*ipix+4]*(R[6*ipix+4]*(U[1]*U[1]*k+1.0f)+R[6*ipix+5]*U[1]*U[2]*k+R[6*ipix+3]*U[0]*U[1]*k) + R[6*ipix+3]*(R[6*ipix+3]*(U[0]*U[0]*k+1.0f)+R[6*ipix+5]*U[0]*U[2]*k+R[6*ipix+4]*U[0]*U[1]*k) ;
         
         

         // New S from the Cholesky decomposition of C, which is stored as
         //                      R[6*ipix+0]   0            0                      C.s0             
         //                      R[6*ipix+1]   R[6*ipix+2]  0              <--     C.s1   C.s2      
         //                      R[6*ipix+3]   R[6*ipix+4]  R[6*ipix+5]            C.s3   C.s4  C.s5
         // Cholesky decomposition
         //  R[6*ipix+0] =   S11  =  sqrt(C11)                        =  sqrt(C.s0)
         //  R[6*ipix+1] =   S21  =  C21/S11                          =  C.s1/R[6*ipix+0]
         //  R[6*ipix+2] =   S22  =  sqrt(C22 - S21**2.0)             =  sqrt(C.s2-R[6*ipix+1]**2)
         //  R[6*ipix+3] =   S31  =  C31/S11                          =  C.s3/R[6*ipix+0]
         //  R[6*ipix+4] =   S32  =  (C32-S31*S21)/S22                =  (C.s4-R[6*ipix+3]*R[6*ipix+1])/R[6*ipix+2]
         //  R[6*ipix+5] =   S33  =  sqrt(C33 - S31**2.0 - S32**2.0)  =  sqrt(C.s5-R[6*ipix+3]**2-R[6*ipix+4]**2)
# if 0
         R[6*ipix+0] =    sqrt(max(CC.s0, 1.0e-32f)) ;
         R[6*ipix+1] =    CC.s1 / R[6*ipix+0] ;
         R[6*ipix+2] =    sqrt(max(CC.s2-R[6*ipix+1]*R[6*ipix+1], 1.0e-32f)) ;
         R[6*ipix+3] =    CC.s3 / R[6*ipix+0] ;
         R[6*ipix+4] =    (CC.s4-R[6*ipix+3]*R[6*ipix+1]) / R[6*ipix+2] ;
         R[6*ipix+5] =    sqrt(max(CC.s5-R[6*ipix+3]*R[6*ipix+3]-R[6*ipix+4]*R[6*ipix+4], 1.0e-32f)) ;
# else
         // ??? CPU working ok, GPU failing with some sources developing NaN R[3], R[4], R[5] values ???
         // Actually they go to inf and CC.s5 goes to minimum allowed, 1e-16.
         // Clamping tried here, does not solve  the GPU problem !!!
         // Reason for the different behaviour of CPU and GPU unknown!!!
         R[6*ipix+0] =    sqrt(max(CC.s0, 1.0e-32f)) ;
         R[6*ipix+1] =    CC.s1 / R[6*ipix+0] ;
         R[6*ipix+2] =    sqrt(max(CC.s2-R[6*ipix+1]*R[6*ipix+1], 1.0e-32f)) ;
         R[6*ipix+3] =    clamp(  CC.s3 / R[6*ipix+0], -9.0f, +9.0f) ;
         R[6*ipix+4] =    clamp( (CC.s4-R[6*ipix+3]*R[6*ipix+1]) / R[6*ipix+2], -9.0f, +9.0f) ;
         R[6*ipix+5] =    max(sqrt(max(CC.s5-R[6*ipix+3]*R[6*ipix+3]-R[6*ipix+4]*R[6*ipix+4], 1.0e-32f)), 1.0e-10f) ;
# endif
         // R[N, 6] is __global array
         
      } // for ipix
      
      
      
      // If it is time, add current parameters to GS and LS 
      //   =  MCMC samples of hierarchical and per-source parameter values
      if (isample>=BURNIN) {
         if ((isample-BURNIN)%THIN==0) {
            // GS[NWG, NG, SAMPLES]
            for(int i=lid; i<NG; i+=LOCAL) GS[gid*NG*SAMPLES+i*SAMPLES+sample] = G[i] ;
            // LS[NWG, N, 3, SAMPLES]
            for(int i=lid; i<N; i+=LOCAL) {
               LS[gid*N*3*SAMPLES+i*3*SAMPLES+0*SAMPLES+sample] = L[3*i+0] ; // I
               LS[gid*N*3*SAMPLES+i*3*SAMPLES+1*SAMPLES+sample] = L[3*i+1] ; // T
               LS[gid*N*3*SAMPLES+i*3*SAMPLES+2*SAMPLES+sample] = L[3*i+2] ; // B
            }
            sample += 1 ;
         } //  %THIN == 0
      }
      
   } // for isample
   
   
# if (DEBUG>1)
   if (id==10) {
      for(int ipix=0; ipix<10; ipix++) {
         for(int i=0; i<6; i++) printf("%12.4e ", R[6*ipix+i]) ;
         printf("\n") ;
      }
   }
# endif
   
}




