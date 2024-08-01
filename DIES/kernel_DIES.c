
#define DEBUG      0
#if 0
# define EPS      1.0e-5f
# define DEPS     2.0e-5f
#else
# define EPS      1.0e-6f
# define DEPS     1.0e-6f
#endif

#define C_LIGHT  29979245800.0f

#include "mwc64x_rng.cl"
#define Rand(x)  (MWC64X_NextUint(x)/4294967295.0f)
#define PI 3.14159265f

//    K         =   ALFA - BETA * (log10(RHO[IND])-log10(RHO[NC-1])) / (log10(RHO[0])-log10(RHO[NC-1])) ;
//  minimum density => ALFA         K>1 for shorter step  ====== TEST VALUES LARGER THAN ONE =====
//  maximum density => ALFA-BETA  ~ 1
#define ALFA  3.0f
#define BETA  2.0f


#if (NVIDIA>0)  //  NVIDIA SPECIFIC ..........................................................................
// ... otherwise the default atomicAdd below is very slow
float atomicAdd(__global float* p, float val)
{
   float prev;
   asm volatile(
		"atom.global.add.f32 %0, [%1], %2;"
		: "=f"(prev)
		: "l"(p) , "f"(val)
		: "memory"
	       );
   return prev;
}

#else           // GENERIC / AMD VERSION .....................................................................

inline void atomicAdd(volatile __global float *addr, float val) {
   union{
      unsigned int u32;
      float        f32;
   } next, expected, current;
   current.f32    = *addr;
   do {
      expected.f32 = current.f32;
      next.f32     = expected.f32 + val;
      current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
}

#endif // ........................................................................................................


float f2um(const float f) {
   return 1.0e4f*C_LIGHT/f ;
}


float HenyeyGreenstein(const float g,  mwc64x_state_t *rng)  { //  return random cos(theta) for scattering angle
   return (1.0f+g*g-pown((1.0f-g*g)/(1.0f-g+2.0f*g*Rand(rng)),2)) / (2.0f*g) ;
}


float pHG(const float g, const float ct) {  // return probability for ct=cos(theta)
   return 0.5f * (1.0f-g*g) / pow(1+g*g-2.0f*g*ct, 1.5f) ;
}


void Deflect(float3 *DIR, const float ct, mwc64x_state_t *rng) {
   float sin_theta, cos_theta, ox, oy, oz, theta0, phi0, sin_phi, cos_phi ;
   phi0      =  2.0f*PI*Rand(rng) ;
   sin_theta =  sqrt(clamp(1.0f-ct*ct, 0.0f, 1.0f)) ;
   ox        =  sin_theta*cos(phi0) ;
   oy        =  sin_theta*sin(phi0) ;
   oz        =  ct ;
   // compute direction of the old vector - rotate in the opposite direction: theta0, phi0
   theta0    =  acos(DIR->z/(length(*DIR)+EPS)) ;
   phi0      =  acos(DIR->x/( sqrt(DIR->x*DIR->x+DIR->y*DIR->y+EPS) ))  ;
   if (DIR->y<0.0f)  phi0 = (2.0f*PI-phi0) ;
   theta0    = -theta0 ;
   phi0      = -phi0 ;
   // rotate (ox,oy,oz) with angles theta0 and phi0
   // #1. rotate around z angle phi0,   2. rotate around x (or y?) angle theta0
   sin_theta =  sin(theta0) ;
   cos_theta =  cos(theta0) ;
   sin_phi   =  sin(phi0) ;
   cos_phi   =  cos(phi0) ;
   DIR->x    = +ox*cos_theta*cos_phi   + oy*sin_phi   -  oz*sin_theta*cos_phi ;
   DIR->y    = -ox*cos_theta*sin_phi   + oy*cos_phi   +  oz*sin_theta*sin_phi ;
   DIR->z    = +ox*sin_theta                          +  oz*cos_theta ;          
}


int Goto(const float3 *POS, __global float *RC) {
   // Return cell index based on the position vector
   const float r = length(*POS) ;
   if (r>=RC[NC-1]) return -1000 ;
   for(int i=0; i<NC; i++) if (RC[i]>r) return i ;
   return -1000 ;
}


float Step(const float3 *POS, const float3 *DIR, const int IND, const __global float *RC) {
   float r0, alfa, beta, r, b, c, det1, det2, distance, R ;
   R    = length(*POS) ;
   alfa =  1.0e10f ;
   r    =  RC[IND]+EPS ;
   b    =  2.0f*dot(*POS, *DIR) ;
   c    =  R*R-r*r ;
   det1 =  b*b-4.0f*c ;
   if (det1>=0.0f) {
      det1  =  sqrt(det1) ;
      alfa  =   0.5f*(-b-det1) ;        // choose smaller of the two solutions
      if (alfa<EPS)  {
	 alfa = 0.5f*(-b+det1) ;
	 if (alfa<EPS) alfa = 1.0e10 ;  // do not accept zero step
	 // if (alfa<EPS) alfa = EPS ;  // do not accept zero step
      }
   }
   // try inner radius
   beta = 1.0e10f ;
   if (IND>0) {
      r    =  RC[IND-1]-EPS ;          // radius for the next border inwards
      c    =  R*R - r*r ;
      det2 =  b*b - 4.0f*c ;
      if (det2>=0.0f) {                // solutions exist of determinant non-negative
	 det2 = sqrt(det2) ;
	 beta = 0.5f*(-b-det2) ;
	 if (beta<EPS) {
	    beta = 0.5f*(-b+det2) ;
	    if (beta<EPS) beta = 1.0e10f ;
	    // // // if (beta<EPS) beta = EPS ;
	 }
      }
   }   
   if (alfa<beta) {
      distance = alfa ;
   } else {
      if (beta<1.0e9f) {
	 distance = beta ;
      } else {
#if (DEBUG>0)
	 if (IND>0)  b = RC[IND-1] ;
	 else        b = -1.0f ;
	 if (IND<NC) c = RC[IND] ;
	 else        c = -1.0f ;
	 printf("**** det1 %.3e det2 %.3e alfa %.3e beta %.3e   %9.7f <= %9.7f <= %9.7f ???\n",
		det1, det2, alfa, beta, b, R, c) ;
	 printf("     POS  %8.4f %8.4f %8.4f     DIR  %8.4f %8.4f %8.4f\n", POS->x, POS->y, POS->z, DIR->x, DIR->y, DIR->z) ;
#endif
	 distance = EPS ;
      }
   }
   return distance  ; // [pc]
}



void __kernel SimulateBG(const    float SEED,    // [ 0]
			 __global float *RC,     // [ 1]  R[NC]    [pc]
			 __global float *VC,     // [ 2]  VC[NC]   [pc^3]
			 __global float *RHO,    // [ 3]  RHO[NC]  [cm^-3]
			 __global float *ipFS,   // [ 4]  ipFS[NF]    = P(freq) for freq = F0*KF**ifreq, source emission
			 __global float *ipFR,   // [ 5]  ipFR[NT, NF]
			 __global float *ipET,   // [ 6]  ipET[NT]
			 __global float *ipKABS, // [ 7]  ipKABS[NF]
			 __global float *ipKSCA, // [ 8]  ipSCA[NF]
			 __global float *ipG,    // [ 9]  ipG[NF]
			 __global float *ABS,    // [10]  ABS[NC]
			 __global float *DABS    // [11]  DABS[NC]
			) {
   // constants NT, NF
   const int id  = get_global_id(0) ;
   
   float3  POS, DIR, DIR0 ;
   float   cos_theta, sin_theta, phi, FPA, FPS, F, G, KABS, KSCA, dx, dxa, dxs, Eabs, T, alfa, W, DE_BG0, K ;
   int     IND, iE, ifreq, iT, j ;
   
   mwc64x_state_t rng;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED,1.0f)*4294967296L), samplesPerStream);
   
   // generate photon package using spherical symmetry
   POS.x     =  0.0f ;      POS.y     =  0.0f ;      POS.z     =  -RC[NC-1]+DEPS ;
   W         =  DE_BG ;
   
#if (WEIGHT_BG>0)           //  weighting the background packages
   if (K_WEIGHT_BG>10.0) {
      //  alternative background weighting, normal p = 2*cos(th)*sin(th), new   p = 1 / (pi/2) = 2/pi,  0<th<0.5*pi
      cos_theta  =  cos(0.5f*PI*Rand(&rng)) ;         // theta unifor [0,pi/2]
      sin_theta  =  sqrt(clamp(1.0f-cos_theta*cos_theta, 0.0f, 1.0f)) ;
      W         *=  cos_theta*sin_theta * PI ;
   } else {
      // with 0<K_WEIGHT_BG<0.5, direct more photon packages towards the model centre
      //  original probability    p = 2*cos(th)*sin(th)
      //  modified probability    p = sin(th) * cos(th)^(1/K-1) / K,     K=0.5 is the normal unweighted case
      //                          P = 1 - cos(th)^(1/K)  =>  u = cos(th)^(1/K),  u^K = cos(th)
      //                          W = 2*cos(th)*sin(th) / [ sin(th)*cos(th)^(1/K-1) ] * K
      //                            = 2*cos(th)^[2-1/K] * K
      cos_theta  =  pow(Rand(&rng), K_WEIGHT_BG)  ;
      W         *=  2.0f*K_WEIGHT_BG*pow(cos_theta, 2.0f-1.0f/K_WEIGHT_BG) ;
   }
#else
   // W         =  DE_BG ;
   cos_theta =  sqrt(Rand(&rng))  ;
#endif
   
   sin_theta =  sqrt(clamp(1.0f-cos_theta*cos_theta, 0.0f, 1.0f)) ;   
   phi       =  2.0f*PI*Rand(&rng) ;
   DIR.x     =  sin_theta*cos(phi) ;
   DIR.y     =  sin_theta*sin(phi) ;
   DIR.z     =  cos_theta ;
   IND       =  NC-1 ;
   
   // random frequency for the source,  lookup table   P[NF] -> ipFS[NF] == frequency
   j         =  clamp((int)(round(NF*Rand(&rng))), 0, NF-1) ;  // index to FF frequencies
   F         =  ipFS[j] ;                   // ipFS[NF], could include linear interpolation
   
   // update dust parameters, lookup table FF -> KABS,  F = F0*KF**ifreq
   ifreq     =  round(log(F/F0)/log(KF)) ;
#if (DEBUG>0)
   if ((ifreq<0)||(ifreq>=NF)) printf("?!?!?!?!?!?!   0<=%d<%d, F=%.3e, j=%d, ipFS %.3e %.3e %.3e\n", ifreq, NF, F, j, ipFS[0], ipFS[1], ipFS[2]) ;
#endif
   KABS      =  ipKABS[ifreq] ;
   KSCA      =  ipKSCA[ifreq] ;
   G         =  ipG[ifreq] ;         
   
   // free path for absorption
#if (WEIGHT_STEP_ABS>0)
# if (WEIGHT_STEP_ABS>1)
   // adaptive step lengthening --- 
   //       a = RHO[0], b = RHO[NC-1],   K =  3 - 2.7*(log10(RHO)-log10(b))/(log10(a)-log10(b))
   //       minimum density => K = 3.0 (shorter step)
   //       maximum density => K = 0.3 (longer step)
   K         =   ALFA - BETA * (log10(RHO[IND])-log10(RHO[NC-1])) / (log10(RHO[0])-log10(RHO[NC-1])) ;
   FPA       =  -log(Rand(&rng)) / K ;
   W        *=   exp(FPA*(K-1.0f)) / K ;
# else
   // fixed step lengthening,   new probability   exp(-k*tau)  =>  k<1 means longer steps
   FPA       =  -log(Rand(&rng)) / K_WEIGHT_STEP_ABS ;
   W        *=   exp(FPA*(K_WEIGHT_STEP_ABS-1.0f)) / K_WEIGHT_STEP_ABS ;
# endif
#else
   FPA       =  -log(Rand(&rng)) ;           // free path for absorption (in optical depth units)
#endif
   
#if (WEIGHT_STEP_SCA>0)
   // free path for scattering increased by factor K_WEIGHT_STEP < 1
   FPS       =  -log(Rand(&rng)) / K_WEIGHT_STEP_SCA ;
   W        *=   exp(FPS*(K_WEIGHT_STEP_SCA-1.0f)) / K_WEIGHT_STEP_SCA ;
#else
   FPS       = -log(Rand(&rng)) ;           // ... and scattering
#endif
   
#if (FORCED>0)  // FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
   // with forced first interaction
   dxa = 0.0f ;    dxs = 0.0f ;     DIR0 = POS ;
   while (IND>=0) {
      dx    =  Step(&POS, &DIR, IND, RC)  ;     // POS and IND not yet updated, [dx] = 1 pc
      dxa  +=  dx*KABS*RHO[IND] ;
      dxs  +=  dx*KSCA*RHO[IND] ;
      // updates
      POS  += (dx+DEPS)*DIR ;      
      alfa  =  length(POS) ;
      if ((IND>0)&&(RC[IND-1]>alfa)) {
	 IND -= 1 ;
      } else {
	 if (alfa>RC[IND]) IND += 1 ;	    
	 if (IND>=NC) IND = -1000 ;       //  outof the model
      }      
   }
   // return to start
   IND       =   NC-1 ;
   POS       =   DIR0 ;
   // total optical depth along the ray => TAU
   dx        =   dxa+dxs ;
   FPA       =  -log(1.0f-Rand(&rng)*(1.0-exp(-dx))) ;
   FPS       =  -log(1.0f-Rand(&rng)*(1.0-exp(-dx))) ;
   //// W         =   DE_BG  * (1.0f-exp(-dx)) ;
   W        *=   1.0f-exp(-dx) ;
#endif // FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
   
   DE_BG0    =  W  ;                        // weighting of the original package
   
   while(IND>=0) {   // stepping
      
      // safeguard against a weighted photon package getting stuck in the model centre
      if (W<(1.0e-12f*DE_BG)) break ;
      
#if (DEBUG>0)
      printf("[%7d] %3d\n", id, IND) ;
      if ((IND>=0)&&(IND<NC)) { ; } else { printf("???\n") ; }
#endif
      dx  = Step(&POS, &DIR, IND, RC)  ;     // POS and IND not yet updated, [dx] = 1 pc
#if (DEBUG>0)
      if (isfinite(dx)) {
	 ;
      } else {
	 printf("????? [%d]\n", id) ;
	 return ;
      }
#endif
      dxs = FPS/(RHO[IND]*KSCA) ;            // distance to next scattering, tau = dx*RHO*K
      dxa = FPA/(RHO[IND]*KABS) ;            // distance to next absorption
      
      if (dx<min(dxa, dxs)) {                // full step without interactions
	 FPA  -=  dx*KABS*RHO[IND] ;
	 FPS  -=  dx*KSCA*RHO[IND] ;         // update free paths
	 POS  += (dx+DEPS)*DIR ;	 
      } else {
	 // scattering or absorption before the full step
	 
	 if (dxs<dxa) {                      // SCATTERING =====================================================
	    dx     =  dxs ;                  // actual length of the step
	    POS   +=  (dx+DEPS)*DIR ;        // ... before DIR is changed, go to location of scattering
	    FPA   -=  dx*KABS*RHO[IND] ;     // update free path for absorptions
#if (WEIGHT_SCA>0)
	    // weighted case
	    //   direction generated wrt centre direction, using HG
	    //   weight is the ratio of HG functions, with parameters g and g'=K_WEIGHT_SCA
	    //   first HG for deflection from original direction, second HG for deflection wrt centre direction
	    cos_theta = HenyeyGreenstein(K_WEIGHT_SCA, &rng) ;   // random ct wrt centre direction
	    DIR0 = DIR ;                                         // original direction
	    // new direction = centre direction + deflection by cos_theta (HG with K_WEIGHT_SCA)
	    DIR.x =  -POS.x ;  DIR.y =  -POS.y ;  DIR.z =  -POS.z ;  DIR = normalize(DIR) ;
	    Deflect(&DIR, cos_theta, &rng) ;	    
	    T  =  dot(DIR0, DIR) ;   // cos(scattering angle).... actual change from DIR0 to DIR
	    W *=  pHG(G, T)  /  pHG(K_WEIGHT_SCA, cos_theta) ; // p(normal) / p(current modified)
#else
	    // normal case, scattering angle = deflection wrt original direction
	    cos_theta  =  HenyeyGreenstein(G, &rng) ; 
	    Deflect(&DIR, cos_theta, &rng) ; // new dir
#endif
	    
#if (WEIGHT_STEP_SCA>0)
	    // free path for scattering increased by factor K_WEIGHT_STEP < 1
	    FPS    =  -log(Rand(&rng)) / K_WEIGHT_STEP_SCA ;     // longer free path for K_WEIGHT_SCA<1.0
	    W     *=   exp(FPS*(K_WEIGHT_STEP_SCA-1.0f)) / K_WEIGHT_STEP_SCA ;	 // modified weight for the package
#else
	    FPS    = -log(Rand(&rng)) ;      // new free path for scattering
#endif	  	    
	 } else {                            // ABSORPTION
	    
	    // ABS = true energy absorbed within a cell / pc^2 ..... because DE_BG = true energy / pc^2
	    // atomicAdd_g_f(&(ABS[IND]), DE_BG) ; // still in the same cell IND
#if (BATCH<=1)
	    atomicAdd(&( ABS[IND]), W) ;  // still in the same cell IND
#else
	    atomicAdd(&(DABS[IND]), W) ; // still in the same cell IND
#endif
	    dx        =  dxa ;               // actual step size
	    POS      +=  (dx+EPS)*DIR ;      // truncated step to location of absorption
	    FPS       = -log(Rand(&rng)) ;   // it is a new photon package at a new frequency =>  new free path
	    
#if (WEIGHT_STEP_ABS>0)
# if (WEIGHT_STEP_ABS>1)
	    K         =   ALFA - BETA * (log10(RHO[IND])-log10(RHO[NC-1])) / (log10(RHO[0])-log10(RHO[NC-1])) ;
	    FPA       =  -log(Rand(&rng)) / K ;
	    W        *=   exp(FPA*(K-1.0f)) / K ;
# else
	    FPA       =  -log(Rand(&rng)) / K_WEIGHT_STEP_ABS ;
	    W        *=   exp(FPA*(K_WEIGHT_STEP_ABS-1.0f)) / K_WEIGHT_STEP_ABS ;
# endif
#else	    
	    FPA       = -log(Rand(&rng)) ;   // new free path for absorptions, after the current absorption
#endif
	    // ipET = temperatures for energies E = E0*KE**iE,    iE =  log(E/E0) / log(KE)
	    //  energies are  E / H * pc
	    //  DE = E / pc^2   =>   ABS = true energy / pc^2
	    //  E / H * pc  =  ABS*pc^2 / (RHO * VC*pc^3) * pc =  ABS / RHO / VC
#if (BATCH<=1)
	    Eabs      =  ABS[IND]            / VC[IND] / RHO[IND] ; //  E / H * pc  --- ASSUMES ABS IS GLOBALLY CURRENT
#else
	    Eabs      = (ABS[IND]+DABS[IND]) / VC[IND] / RHO[IND] ; //  E / H * pc  --- ASSUMES ABS IS GLOBALLY CURRENT
#endif
	    iE        =  clamp((int)(round(log(Eabs/E0)/log(KE))), 0, NT-1) ;
	    T         =  ipET[iE] ;          // convert energy to temperature
	    // At current temperature, generate random frequency for re-emission
	    // first the current temperature index, T = T0*kT**iT  =>  iT = log(T/T0) / log(kT)
	    iT        =  clamp((int)(round(log(T/T0)/log(KT))), 0, NT-1) ; 
	    //  ipFR[NT, NF]  ->  frequency
	    F         =  ipFR[(int)round(iT*NF+NF*Rand(&rng))] ; // random frequency from the lookup table
	    //  frequency -> frequency index,  F =  F0*KF**iF  =>   iF = log(F/F0) / log(KF)
	    ifreq     =  clamp((int)round(log(F/F0)/log(KF)), 0, NF-1) ;	    
	    // update dust parameters -- for the new frequency
	    KABS      =  ipKABS[ifreq] ;
	    KSCA      =  ipKSCA[ifreq] ;
	    G         =  ipG[ifreq] ;
	    
#if (WEIGHT_EMIT>0)
	    // emission directed towards the cloud centre
	    //   normal probability     p  = 1/(4*pi)  * dOmega
	    //   modified probability   p' = C*(1+cos(theta))^k,   k>0 more directed towards the centre
	    // Use HG as the weight function
	    cos_theta =  HenyeyGreenstein(K_WEIGHT_EMIT, &rng) ;   // wrt to centre direction
	    W        *=  0.5f / pHG(K_WEIGHT_EMIT, cos_theta) ;
	    //  deflect ~cos_theta relative to the centre direction
	    DIR.x  =  -POS.x ;   DIR.y  =  -POS.y ;   DIR.z  =  -POS.z ;
	    DIR    =   normalize(DIR) ;       // this unit vector now pointing towards the cloud centre
	    Deflect(&DIR, cos_theta, &rng) ;  // deflection cos_theta from the centre direction
#else
	    // isotropic emission --- no weighting
	    cos_theta =  -1.0f+2.0f*Rand(&rng) ;
	    sin_theta =  sqrt(clamp(1.0f-cos_theta*cos_theta, 0.0f, 1.0f)) ;   
	    phi       =  2.0f*PI*Rand(&rng) ;
	    DIR.x     =  sin_theta*cos(phi) ;
	    DIR.y     =  sin_theta*sin(phi) ;
	    DIR.z     =  cos_theta ;
	    // W         =   DE_BG ;
#endif	    
	    
	 }
      }      
      // take the step and update cell index
      alfa    =  length(POS) ;
      if ((IND>0)&&(RC[IND-1]>alfa)) {
	 IND -= 1 ;
      } else {
	 if (alfa>RC[IND]) IND += 1 ;	    
	 if (IND>=NC) IND = -1000 ;       //  outof the model
      }
      
   } // while IND
   
}  // Simulate



void __kernel SimulatePS(const    float  SEED,   // [ 0]
			 __global float *RC,     // [ 1]  R[NC]    [pc]
			 __global float *VC,     // [ 2]  VC[NC]   [pc^3]
			 __global float *RHO,    // [ 3]  RHO[NC]  [cm^-3]
			 __global float *ipFS,   // [ 4]  ipFS[NF]    = P(freq) for freq = F0*KF**ifreq, source emission
			 __global float *ipFR,   // [ 5]  ipFR[NT, NF]
			 __global float *ipET,   // [ 6]  ipET[NT]
			 __global float *ipKABS, // [ 7]  ipKABS[NF]
			 __global float *ipKSCA, // [ 8]  ipSCA[NF]
			 __global float *ipG,    // [ 9]  ipG[NF]
			 __global float *ABS,    // [10]  ABS[NC]
			 __global float *DABS    // [10]  ABS[NC]
			) {
   // constants NT, NF
   const int id = get_global_id(0) ;   
   float3  POS, DIR, DIR0 ;
   float   cos_theta, sin_theta, phi, FPA, FPS, F, G, KABS, KSCA, dx, dxa, dxs, Eabs, T, alfa, W, K ;
   int     IND, iE, ifreq, iT, j ;
   
   mwc64x_state_t rng;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED,1.0f)*4294967296L), samplesPerStream);
   
   // generate photon package using spherical symmetry, source radius RPS
   POS.x     =  0.0f ;      POS.y     =  0.0f ;      POS.z     =   RPS ;   
   W         =  DE_PS ;
   cos_theta =  sqrt(Rand(&rng))  ;
   
   sin_theta =  sqrt(clamp(1.0f-cos_theta*cos_theta, 0.0f, 1.0f)) ;   
   phi       =  2.0f*PI*Rand(&rng) ;
   DIR.x     =  sin_theta*cos(phi) ;
   DIR.y     =  sin_theta*sin(phi) ;
   DIR.z     =  cos_theta ;
   IND       =  0 ;         // source assumed to be inside centre cell
   
   // random frequency for the source,  lookup table   P[NF] -> ipFS[NF] == frequency
   j         =  clamp((int)(round(NF*Rand(&rng))), 0, NF-1) ;  // index to FF frequencies
   F         =  ipFS[j] ;                   // ipFS[NF], could include linear interpolation
   
   // free path for absorption
#if (WEIGHT_STEP_ABS>0)
# if (WEIGHT_STEP_ABS>1)
   K         =   ALFA - BETA * (log10(RHO[IND])-log10(RHO[NC-1])) / (log10(RHO[0])-log10(RHO[NC-1])) ;
   FPA       =  -log(Rand(&rng)) / K ;
   W        *=   exp(FPA*(K-1.0f)) / K ;
# else
   FPA       =  -log(Rand(&rng)) / K_WEIGHT_STEP_ABS ;
   W        *=   exp(FPA*(K_WEIGHT_STEP_ABS-1.0f)) / K_WEIGHT_STEP_ABS ;
# endif
#else
   FPA       =  -log(Rand(&rng)) ;           // free path for absorption (in optical depth units)
#endif
   
#if (WEIGHT_STEP_SCA>0)
   // free path for scattering increased by factor K_WEIGHT_STEP < 1
   FPS       =  -log(Rand(&rng)) / K_WEIGHT_STEP_SCA ;
   W        *=   exp(FPS*(K_WEIGHT_STEP_SCA-1.0f)) / K_WEIGHT_STEP_SCA ;
#else
   FPS       = -log(Rand(&rng)) ;           // ... and scattering
#endif
   
   // update dust parameters, lookup table FF -> KABS,  F = F0*KF**ifreq
   ifreq     =  round(log(F/F0)/log(KF)) ;
   KABS      =  ipKABS[ifreq] ;
   KSCA      =  ipKSCA[ifreq] ;
   G         =  ipG[ifreq] ;         
   
   while(IND>=0) {                           // step until photon package exits the model volume
      
      if (W<(1.0e-10f*DE_PS)) break ;        // terminate if the weight becomes insignificant
      
#if (DEBUG>0)
      if ((IND>=0)&&(IND<NC)) { ; } else { printf("???\n") ; }
#endif
      dx  = Step(&POS, &DIR, IND, RC)  ;     // POS and IND not yet updated, [dx] = 1 pc
#if (DEBUG>0)
      if (isfinite(dx)) {
	 ;
      } else {
	 printf("????? [%d]\n", id) ;
	 return ;
      }
#endif
      dxs = FPS/(RHO[IND]*KSCA) ;            // distance to next scattering, tau = dx*RHO*K
      dxa = FPA/(RHO[IND]*KABS) ;            // distance to next absorption
      
      if (dx<min(dxa, dxs)) {                // full step without interactions
	 FPA  -=  dx*KABS*RHO[IND] ;
	 FPS  -=  dx*KSCA*RHO[IND] ;         // update free paths
	 POS  += (dx+DEPS)*DIR ;
      } else {
	 // scattering or absorption before the full step	      
	 if (dxs<dxa) {                      // SCATTERING ---------------------------------------------------
	    dx     =  dxs ;
	    POS   +=  (dx+DEPS)*DIR ;        // ... before DIR is changed, go to location of scattering
	    FPA   -=  dx*KABS*RHO[IND] ;     // update free path for absorptions
	    FPS    = -log(Rand(&rng)) ;      // new free path for scattering
#if (WEIGHT_SCA_PS>0)
	    // weighted case
	    //   direction generated wrt outward  direction, using HG
	    //   weight is the ratio of HG functions, with parameters g and g'=K_WEIGHT_SCA_PS
	    //   first HG for deflection from original direction, second HG for deflection wrt outward direction
	    cos_theta = HenyeyGreenstein(K_WEIGHT_SCA_PS, &rng) ;  // relative to outwards direction
	    DIR0 = DIR ;              // original direction
	    // new direction = out direction + deflection by cos_theta
	    DIR.x =  POS.x ;  DIR.y =  POS.y ;  DIR.z =  POS.z ;  DIR = normalize(DIR) ;  // outwards !!
	    Deflect(&DIR, cos_theta, &rng) ; // DIR is the new direction
	    T   =  dot(DIR0, DIR) ;          // cos(scattering angle)
	    W  *=  pHG(G, T)  /  pHG(K_WEIGHT_SCA_PS, cos_theta) ;
#else
	    // normal case, scattering angle = deflection wrt original direction
	    cos_theta  =  HenyeyGreenstein(G, &rng) ; 
	    Deflect(&DIR, cos_theta, &rng) ; // new dir
#endif	    	    
#if (WEIGHT_STEP_SCA>0)
	    // free path for scattering increased by factor K_WEIGHT_STEP < 1
	    FPS    =  -log(Rand(&rng)) / K_WEIGHT_STEP_SCA ;     // longer free path for K_WEIGHT_SCA<1.0
	    W     *=   exp(FPS*(K_WEIGHT_STEP_SCA-1.0f)) / K_WEIGHT_STEP_SCA ;	 // modified weight for the package
#else
	    FPS    = -log(Rand(&rng)) ;      // new free path for scattering
#endif
	 } else {                            // ABSORPTION ---------------------------------------------------
#if (BATCH<=1)
	    atomicAdd(&( ABS[IND]), W) ;     // still in the same cell IND
#else
	    atomicAdd(&(DABS[IND]), W) ;     // still in the same cell IND
#endif
	    dx        =  dxa ;
	    POS      +=  (dx+EPS)*DIR ;      // truncated step to location of absorption
	    FPS       = -log(Rand(&rng)) ;   // new photon packahe at  new frequency => regenerate also FPS
#if (WEIGHT_STEP_ABS>0)
# if (WEIGHT_STEP_ABS>1)
	    K         =   ALFA - BETA * (log10(RHO[IND])-log10(RHO[NC-1])) / (log10(RHO[0])-log10(RHO[NC-1])) ;
	    FPA       =  -log(Rand(&rng)) / K ;
	    W        *=   exp(FPA*(K-1.0f)) / K ;
# else	    
	    FPA       =  -log(Rand(&rng)) / K_WEIGHT_STEP_ABS ;
	    W        *=   exp(FPA*(K_WEIGHT_STEP_ABS-1.0f)) / K_WEIGHT_STEP_ABS ;
# endif
#else	    
	    FPA       = -log(Rand(&rng)) ;   // free path for sabsorptions, after the current absorption
#endif
	    // ipET = temperatures for energies E = E0*KE**iE,    iE =  log(E/E0) / log(KE)
	    //  energies are  E / H * pc
	    //  DE = E / pc^2   =>   ABS = true energy / pc^2
	    //  E / H * pc  =  ABS*pc^2 / (RHO * VC*pc^3) * pc =  ABS / RHO / VC
#if (BATCH<=1)
	    Eabs      =   ABS[IND]            / VC[IND] / RHO[IND] ;  //  E / H * pc
#else
	    Eabs      =  (ABS[IND]+DABS[IND]) / VC[IND] / RHO[IND] ;  //  E / H * pc
#endif
	    iE        =  clamp((int)(round(log(Eabs/E0)/log(KE))), 0, NT-1) ;
	    T         =  ipET[iE] ;          // convert energy to temperature
	    // at current temperature, generate random frequency for re-emission
	    // first the current temperature index, T = T0*kT**iT  =>  iT = log(T/T0) / log(kT)
	    iT        =  clamp((int)(round(log(T/T0)/log(KT))), 0, NT-1) ; 
	    //  ipFR[NT, NF]  ->  frequency
	    F         =  ipFR[(int)round(iT*NF+NF*Rand(&rng))] ; // random frequency from the lookup table
	    //  frequency -> frequency index,  F =  F0*KF**iF  =>   iF = log(F/F0) / log(KF)
	    ifreq     =  clamp((int)round(log(F/F0)/log(KF)), 0, NF-1) ;	    
	    // update dust parameters -- for the new frequency
	    KABS      =  ipKABS[ifreq] ;
	    KSCA      =  ipKSCA[ifreq] ;
	    G         =  ipG[ifreq] ;	    
#if (WEIGHT_EMIT_PS>0)
	    // emission directed towards cloud surface
	    //   normal probability     p  = 1/(4*pi)  * dOmega
	    //   modified probability   p' = C*(1+cos(theta))^k,   k>0 more directed towards the centre
	    
	    // generate ct relative to the direction outwards, towards the model surface
	    cos_theta =  HenyeyGreenstein(K_WEIGHT_EMIT_PS, &rng) ;
	    W        *=    0.5f / pHG(K_WEIGHT_EMIT_PS, cos_theta) ;  // uniform -> HG angular distribution
	    //  deflect ~cos_theta relative to the outward direction
	    DIR.x     =  POS.x ;   DIR.y  =  POS.y ;   DIR.z  =  POS.z ;
	    DIR       =  normalize(DIR) ;     // unit vector pointing outwards
	    Deflect(&DIR, cos_theta, &rng) ;  // deflection cos_theta from the out direction
#else
	    // isotropic emission
	    cos_theta =  -1.0f+2.0f*Rand(&rng) ;
	    sin_theta =  sqrt(clamp(1.0f-cos_theta*cos_theta, 0.0f, 1.0f)) ;   
	    phi       =  2.0f*PI*Rand(&rng) ;
	    DIR.x     =  sin_theta*cos(phi) ;
	    DIR.y     =  sin_theta*sin(phi) ;
	    DIR.z     =  cos_theta ;
#endif
	 }  // -----------------------------------------------------------------------------------------------
      }      
      // take the step and update cell index
      alfa    =  length(POS) ;
      if ((IND>0)&&(RC[IND-1]>alfa)) {
	 IND -= 1 ;
      } else {
	 if (alfa>RC[IND]) IND += 1 ;	    
	 if (IND>=NC) IND = -1000 ;       //  outof the model
      }      
   } // while IND
   
}  // Simulate



float Planck(const float f, const float T) {
   return 1.47449933569e-17f*pown(f*1.0e-10f, 3) / ( exp(4.799243348e-11f*f/T) - 1.0f ) ;
}



void __kernel Map(__global float *RC,      //  RC[NC]
		  __global float *RHO,     //  RHO[NC]
		  __global float *FF,      //  FF[NF]
		  __global float *ipKABS,  //  ipKABS[NF]
		  __global float *ipKSCA,  //  ipKSCA[NF]
		  __global float *T,       //  T[NC]
		  __global float *RES      //  RES[OFFS, NF]
		 ) {
   // Calculate spectrum for one impact parameter (offset OFFS)
   const int id = get_global_id(0) ;
   if (id>=(OFFS*NF)) return ;             // one work item, OFFS impact parameters, NF frequencies
   int  ioff  =  id / NF ;                 // impact parameter
   int  ifreq =  id % NF ;                 // frequency
   int  IND ;
   float x, z, res=0.0f, TAU=0.0f, tau, dx, R ;
   float3 POS, DIR ;
   x  =  (ioff/(OFFS-1.0f)) * RC[NC-1] * 0.9995f ;
   z  =  sqrt(RC[NC-1]*RC[NC-1] -  x*x) ;
   // printf("ioff %3d/%3d  x %12.4e,   z %12.4e\n", ioff, OFFS, x, z) ;
   DIR.x = 0.0f ;     DIR.y = 0.0f ;   DIR.z = 1.0f ;
   POS.x = x ;        POS.y = 0.0f ;   POS.z = -0.9995*z ;
   float  KABS =  ipKABS[ifreq] ;   //  tau/H/pc
   float  KSCA =  ipKSCA[ifreq] ;   
   IND = Goto(&POS, RC) ;
   while (IND<NC) {                 // *****IGNORING SCATTERING*****
      // printf("[%2d] ioff=%3d, ifreq=%3d\n", id, ioff, ifreq) ;
      dx   =  Step(&POS, &DIR, IND, RC)  ; 
      tau  =  dx*KABS*RHO[IND] ;    // optical depth for absorption
      res +=  Planck(FF[ifreq], T[IND]) * exp(-TAU) * ( (tau<0.01f) ? (tau * (1.0f-0.5f*tau)) : (1.0f-exp(-tau)) ) ;
      TAU +=  tau ;                 // total optical depth so far
      POS +=  (dx+DEPS)*DIR ;      
      // printf(" %8.5f %8.5f %8.5f  %7.3f %7.3f %7.3f dx = %.3e\n", POS.x, POS.y, POS.z, DIR.x, DIR.y, DIR.z, dx) ;
      R    =  length(POS) ;
      if (R>RC[IND]) {
	 IND += 1 ;
      } else {
	 if ((IND>0)&&(R<RC[IND-1])) IND -= 1 ;
      }
   }
   // printf("ioff %3d   TAU %.3e\n", ioff, TAU) ;
   // RES[ioff*NF+ifreq]  =  TAU/FF[ifreq] ;  // + BG[ifreq]*exp(-TAU) ;
   RES[ioff*NF+ifreq]  =  res ;  // + BG[ifreq]*exp(-TAU) ;
}



__kernel void  UpdateABS(__global float *ABS, __global float *DABS) {
   // ABS += DABS
   const int id = get_global_id(0) ;
   if (id>=NC) return ;
   ABS[id]  += DABS[id] ;
   DABS[id]  = 0.0f ;
}

