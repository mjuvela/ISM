
# include "mwc64x_rng.cl"
# define  Rand(x)  (MWC64X_NextUint(x)/4294967295.0)
# define  Q    2.77258872f       // 4.0*log(2.0)
# define  PI2  6.283185307f


// 2023-12-14 -- all float, simplex results ~ok (almost no change with double precision)
# define BYTES 4

# if (BYTES==4)
#  define REAL float
#  define ZERO 0.0f
#  define TWO  2.0f
#  define ONE  1.0f
#  define HALF 0.5f
# else
#  define REAL double
#  define ZERO 0.0
#  define TWO  2.0
#  define ONE  1.0
#  define HALF 0.5
# endif

# define ASSERT(x,s) ( x ? 1 : printf("Asserttion error: %s\n", s) )

// Gaussian function, velocity and two parameters, avoiding division by zero...
# define GAUSS(v,I,V,FWHM) (I*native_exp(-Q*pown((V-v)/(FWHM+1.1234567e-6f),2)))


//==========================================================================================
// Pre-defined penalty functions
// POS(x, a) = value of x should be positive, adding penalty 1.0 for each step a to the negative side
# define POS( x, a)  ((x>=a) ? (0.0f) : (-x/a))
# define POS2(x, a)  ((x>=a) ? (0.0f) : (x*x/(a*a)))
// NEG(x, a) = value of x should be negative
# define NEG( x, a)  ((x<=a) ? (0.0f) : (-x/a))
# define NEG2(x, a)  ((x<=a) ? (0.0f) : (x*x/(a*a)))
// SQUARE(x, a) =  penalty (x/a)^2
# define SQUARE(x, a)  (x*x/(a*a))
//==========================================================================================


//==========================================================================================
// Pre-defined prior functions
// NORMAL(x, s)  ==   x ~ N(0,s)
// # define NORMAL(x, s)  (exp(-0.5f*x*x/(s*s)) / (2.50662827463f*s))
// === or ===
// since the constant normalisation factor does not affect the optimisation and
// because kernel returns a "chi2" value, add also priors only as similar "chi2" value
// => "chi2" = -2*lnP but dropping the normalisation factor from P
// ... one probably wants to know the fractional change in chi2 due to the priors
# define NORMAL(x, s)  (exp(-0.5f*(x)*(x)/((s)*(s))))
// BOX(x, a, b)  ==  1/(b-a) for a<x<b, 0 elsewhere
# define BOX(x, a, b)  (((x<a)||(x>b)) ? (0.0f) : (1.0f/(b-a)))
//==========================================================================================



# define Jfun(T,T0) (T0/(native_exp(T0/T)-ONE))



// Hyperfine structure   HFS(v1, Tex, v, fwhm, tau), keeping Tex as the parameter
// does not matter if HFS1 returns float (for simplex, BF)
float HFS1(const float v1, const float TEX, const float V, const float FWHM, const float TAU,
	   __constant float *VHF1, __constant float *IHF1) {
   // For accurate HFS parameter distributions, t must be defined double??
   float t = ZERO ; // BF, simplex: does not matter if this is float, only faster (6.3e3 vs. 1.4e3 s/s)
   for(int j=0; j<NHF1; j++)  t +=  GAUSS(v1, TAU*IHF1[j], V+VHF1[j], FWHM) ;
# if 1
   return (Jfun(TEX,HFK1)-Jfun(TBG,HFK1)) * (ONE-exp(-t)) ;
# else
   return (Jfun(TEX,HFK1)-Jfun(TBG,HFK1)) * ((fabs(t)>2e-3f) ? (1.0f-exp(-t)) : (t*(1.0f-HALF*t))) ;
# endif
}


float HFS2(const float v2, const float TEX, const float V, const float FWHM, const float TAU,
	   __constant float *VHF2, __constant float *IHF2) {
   float t = ZERO ;
   for(int j=0; j<NHF2; j++)  t +=  GAUSS(v2, TAU*IHF2[j], V+VHF2[j], FWHM) ;
   return (Jfun(TEX,HFK2)-Jfun(TBG,HFK2)) * (ONE-exp(-t)) ;
}



// Another version of HFS with z = [J(Tex,HFS1)-J(Tbg,HFS1)]  directly as the optimisede parameter
float HFSX1(const float v1, const float JT, const float V, const float FWHM, const float TAU,
	    __constant float *VHF1, __constant float *IHF1) {
   float t = 0.0f ; // BF, simplex: does not matter if this is float, only faster (6.3e3 vs. 1.4e3 s/s)
   for(int j=0; j<NHF1; j++)  t +=  GAUSS(v1, TAU*IHF1[j], V+VHF1[j], FWHM) ;
   return JT * (ONE-exp(-t)) ;
}

float HFSX2(const float v2, const float JT, const float V, const float FWHM, const float TAU,
	    __constant float *VHF2, __constant float *IHF2) {
   float t = 0.0f ; // BF, simplex: does not matter if this is float, only faster (6.3e3 vs. 1.4e3 s/s)
   for(int j=0; j<NHF2; j++)  t +=  GAUSS(v2, TAU*IHF2[j], V+VHF2[j], FWHM) ;
   return JT * (ONE-exp(-t)) ;
}






float  Chi2_1(const          float  *x, 
	      const __global float  *V1,
	      const __global float  *Y1, 
	      const          float   dY1, 
	      __constant float      *VHF1,
	      __constant float      *IHF1,
	      __global  float       *aux    // AUX[k,NY,NX],  k:th auxiliary image
	     ) {
   // Return chi2 (unnormalised) for the current model, spectrum y1.
   // The effect of possible penalties and priors is added directly into the returnc chi2 value.
   float  y1, v1, chi2=0.0f ;
   for(int i=0; i<M1; i++) {   // M1 velocity channels in the spectrum Y1
      v1 = V1[i] ;
      // @y1 is replaced with the model expression from the INI-file, 
      // such as a sum of Gaussians  "y1 = GAUSS(v1, x[0],x[1],x[2]) + ..."
      //  or HFS spectra:            "y1 = HFS(v1, x[0], x[1], x[2], x[3], VHFS1, IHFS1)"
      @y1 ;  // model prediction
      chi2 += (y1-Y1[i])*(y1-Y1[i]) ;
   }
   chi2  /=  (dY1*dY1) ;
# if (PRIORS>0)  // prior is probability  =>  chi2 -= 2*log(prior)
   chi2  -=  2.0f * log(@prior) ;
# endif
# if (PENALTIES>0) // penalty is added directly to the chi2 value
   chi2  +=  @pen ;
# endif
   return chi2 ;
}



float Chi2_2(const          float  *x, 
             const __global float  *V2,
             const __global float  *Y2, 
             const          float   dY2, 
             __constant float      *VHF2,
             __constant float      *IHF2,
             __global float        *aux
            ) {
   // Chi2 values (with penalties and priors) for the second spectrum y2.
   float chi2=0.0f, y2, v2 ;   
   for(int i=0; i<M2; i++) {  // M2 velocity channels in the spectrum Y2
      v2 = V2[i] ;
      @y2 ;
      chi2 += (y2-Y2[i])*(y2-Y2[i]) ;
   }
   chi2  /=  (dY2*dY2) ;
# if (PRIORS>0)  // prior is probability
   chi2  -=  2.0f*log(@prior) ; 
# endif
# if (PENALTIES>0)
   chi2  +=  @pen ; 
# endif
   return chi2 ;
}





void __kernel BF(__global float    *V1,   // X1[M1]      =  velocity [km/s] for first spectra
		 __global float    *YY1,  // YY1[N, M1]  =  Ta [K] for first spectra
		 __global float    *dYY1, // dYY1[N]     =  Ta [K] for first spectra                 
                 __global float    *V2,   // X2[M2]      =  velocity [km/s] for second spectra
                 __global float    *YY2,  // YY2[N, M2]  =  Ta [K] for second spectra
                 __global float    *dYY2, // dYY2[N]     =  Ta [K] for second spectra
		 __constant float  *VHF1, // V1[NHF1]    =  relative velocities of HFS components
		 __constant float  *IHF1, // I1[NHF1]    =  relative intensities of HFS components
		 __constant float  *VHF2, // V2[NHF2]    =  relative velocities of HFS components
		 __constant float  *IHF2, // I2[NHF2]    =  relative intensities of HFS components
		 __global float    *P,    // P[N, NP]    =  parameter vector
		 __global float    *C2,   // C[N]        =  final chi2 values C[N]
                 __global float    *AUX
		) {
   // Brute force optimisation ... robust and still "fast enough"?
   const int id = get_global_id(0) ;             //  one work item = one spectrum
   if (id>=N) return ;                           //  only N spectra
   float x[NP] ;                                  //  NP = number of parameters
   for(int i=0; i<NP; i++) x[i] = P[id*NP+i] ;   //  copy parameters for one spectrum to private memory
   __global float  *Y1  =  &( YY1[id*M1]) ;     
   const float     dY1  =  dYY1[id] ;    
# if (TWIN>0) // if there is a second observed spectrum to be fitted simultaneously
   __global float  *Y2  =  &( YY2[id*M2]) ;
   const float     dY2  =  dYY2[id] ;
# endif
   
   float a, b  ;
   float step=0.233f  ;
   // float step=0.323f  ;  0.233 -> 0.323  .... 3-dispersion clearly stronger tail
   // float step=0.07145 ;  // smaller better for 3-dispersion ??
   
   REAL f0, f1 ;
   int iter=0, accept=0 ;
   __global float *aux  ;
# if (N_AUX>0)
   aux = &(AUX[id*N_AUX]) ;  // pointer to all AUX for current spectrum
# endif
   
   f0  =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
# if (TWIN>0)
   f0 +=  Chi2_2(x, V2, Y2, dY2, VHF2, IHF2, aux) ;
# endif
   
# if (NHF1>0)
#  if (BYTES==4)
   const int    niter   = 300 ;      // 100 enough even for N2H+ (with appropriate initial values)
   const float  minstep = 1.0e-7f ;  // 1e-7 -> 1e-6, no change
#  else
   const int    niter   = 400 ;
   const float  minstep = 1.0e-8f ;
#  endif
# else
   const int    niter   = 500 ;
   const float  minstep = 1.0e-4f ;
# endif
   
   
   while((step>minstep)&&(iter<niter)) {
      
      iter++ ;
      accept = 0 ;
      
      
# if 1
#  if (NHF1>0) // ================================================================================
      
#   if 1
      if (iter%2==0) {
	 x[0] += 0.4f*step ;           x[3] -= 0.1f*step ;
	 f1     =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
#    if (TWIN>0)
	 f1    +=  Chi2_2(x, V2, Y2, dY2, VHF2, IHF2, aux) ;
#    endif      
	 if (f1>f0) {        // reject step
	    x[0] -= 0.4f*step;          x[3] += 0.1f*step ;
	 } else {
	    f0 = f1 ;        // accept, f0 = best chi2 so far
	    accept++ ;
	 }
      } else{
	 x[0] -= 0.5f*step ;        x[3] += 0.1f*step ;
	 f1     =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
#    if (TWIN>0)
	 f1    +=  Chi2_2(x, V2, Y2, dY2, VHF2, IHF2, aux) ;
#    endif      
	 if (f1>f0) {        // reject step
	    x[0] += 0.5f*step ;     x[3] -= 0.1f*step ;
	 } else {
	    f0 = f1 ;        // accept, f0 = best chi2 so far
	    accept++ ;
	 }
      }
#   endif
      
      
#   if 0 // ~good
      if (iter%1==0) {
	 for(int iii=0; iii<10; iii++) {
	    a = 0.3f*cos(0.01f*iter+iii*0.5712f) ;  b = 0.3f*sin(0.01f*iter+iii*0.5712f) ;
	    x[0] += a ;   x[3] += b ;
	    f1     =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
#    if (TWIN>0)
	    f1    +=  Chi2_2(x, V2, Y2, dY2, VHF2, IHF2, aux) ;
#    endif
	    if (f1>f0) {        // reject step
	       x[0] -= a ;     x[3] -= b ;
	    } else {
	       f0 = f1 ;       accept++ ;
	    }	    
	 } // for iii
      }
#   endif
      
      
#   if 1 //  ~ ok... for test problem > 1e5 s/s
      if (iter%5==0) {
	 for(int iii=0; iii<10; iii++) {
	    a = 0.6f*cos(0.01f*iter+iii*0.5712f) ;  b = 0.3f*sin(0.01f*iter+iii*0.5712f) ;
	    x[0] += a ;   x[3] += b ;
	    f1     =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
#    if (TWIN>0)
	    f1    +=  Chi2_2(x, V2, Y2, dY2, VHF2, IHF2, aux) ;
#    endif
	    if (f1>f0) {        // reject step
	       x[0] -= a ;     x[3] -= b ;
	    } else {
	       f0 = f1 ;       accept++ ;
	    }	    
	 } // for iii
      }
#   endif
      
      
      
#  endif // ====================================================================================
# endif
      
      
      for(int ip=0; ip<NP; ip++) {
	 x[ip] +=  step ;
         f1     =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
# if (TWIN>0)
         f1    +=  Chi2_2(x, V2, Y2, dY2, VHF2, IHF2, aux) ;
# endif      	 
	 if (f1>f0) { // reject step ... heuristically small step in opposite direction
	    x[ip] -= step*1.0f ;
	 } else {     // accept, f0 = best chi2 so far, keep current x	    
	    f0 = f1 ;      
	    accept++ ;
	 }
      } // for ip
      
      for(int ip=0; ip<NP; ip++) {
	 x[ip] -=  step ;                           
         f1     =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
# if (TWIN>0)
         f1    +=  Chi2_2(x, V2, Y2, dY2, VHF2, IHF2, aux) ;
# endif      
	 if (f1>f0) {
	    x[ip] += step*1.0f ; 
	 } else {
	    f0 = f1 ;
	    accept++ ;
	 }
      } // for ip
      
      if (accept<1)  step *= 0.90f ;   // no +/- step accepted for any of the parameters
      
# if 0
      if (id==10) printf("[%4d]  %8.4f %8.4f %8.4f   step %.3e  accept %d - %10.3e %10.3e\n",
			 iter, x[0], x[1], x[2], step, accept, f0, f1) ;
# endif
      
   }      
   
# if (0)
   // save in order of increasing velocity ... except that we do not know here what x[] are...
   if (x[1]<x[4]) {
      for(int i=0; i<NP; i++) P[id*NP+i] = x[i] ;
   } else {
      for(int i=0; i<3; i++) {
	 P[id*NP+i]   = x[i+3] ;
	 P[id*NP+i+3] = x[i] ;
      }
   }
# else
   for(int i=0; i<NP; i++) P[id*NP+i] = x[i] ;
# endif
   C2[id] =  f0 ;  // UN-normalised = standard chi2
}






void __kernel CG(__global float    *V1,   // X1[M1]      =  velocity [km/s] for first spectra
		 __global float    *YY1,  // YY1[N, M1]  =  Ta [K] for first spectra
		 __global float    *dYY1, // dYY1[N]     =  Ta [K] for first spectra                 
                 __global float    *V2,   // X2[M2]      =  velocity [km/s] for second spectra
                 __global float    *YY2,  // YY2[N, M2]  =  Ta [K] for second spectra
                 __global float    *dYY2, // dYY2[N]     =  Ta [K] for second spectra
		 __constant float  *VHF1, // V1[NHF1]    =  relative velocities of HFS components
		 __constant float  *IHF1, // I1[NHF1]    =  relative intensities of HFS components
		 __constant float  *VHF2, // V2[NHF2]    =  relative velocities of HFS components
		 __constant float  *IHF2, // I2[NHF2]    =  relative intensities of HFS components
		 __global float    *P,    // P[N, NP]    =  parameter vector
		 __global float    *C2,   // C[N]        =  final chi2 values C[N]
                 __global float    *AUX
		) {
   // Conjugate gradient... with finite-difference derivatives
   const int id = get_global_id(0) ;             //  one work item = one spectrum
   if (id>=N) return ;                           //  only N spectra
   float x[NP], xx[NP] ;                         //  NP = number of parameters
   for(int i=0; i<NP; i++) x[i] = P[id*NP+i] ;   //  copy parameters for one spectrum to private memory
   __global float  *Y1  =  &( YY1[id*M1]) ;     
   const float     dY1  =  dYY1[id] ;    
# if (TWIN>0) // if there is a second observed spectrum to be fitted simultaneously
   __global float  *Y2  =  &( YY2[id*M2]) ;
   const float     dY2  =  dYY2[id] ;
# endif
   int iter=0, accept=0, BAD=0, n ;
   
   // BRENT NOT WORKING !!!
# define BRENT 0
   
# if (BRENT>0)
#  define tol   1.0e-6f
#  define CGOLD 0.3819660f
#  define ZEPS  1.0e-10f
#  define SIGN(a,b)  (b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a))
#  define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d) ;      
   float a,b,d,etemp,fu,fv,fw,fz,p,q,r,tol1,tol2,u,w,z,xm, xmin, e=0.0f, ax, bx, cx ;
# else
   float lA, h, lB, lC, lD  ;
   float fA, fB, fC, fD   ;
# endif
   float NO=ONE, DE, beta, tmp, v, y, f, f0, f1 ;
   
   __global float *aux ;   
# if (N_AUX>0)
   aux  =  &(AUX[id*N_AUX]) ;
# endif
   
# if (BYTES==4)
   const float EPS_STEP = 1.0e-7f ;   //  1.0e-4 not small enough
   const float EPS_GRAD = 1.0e-4f ;   //  0.1 no different from 1e-5
# else
   const float EPS_STEP = 1.0e-6 ;
   const float EPS_GRAD = 1.0e-5 ;
# endif
   
   const int   MAX_ITER = 100 ;
   
   const float invphi  = 0.6180339887498949f ;
   const float invphi2 = 0.3819660112501051f ;
   float *gx, *gx0, gg[NP], gg0[NP], s[NP], *A, *B, *ptr ;
   gx  = &(gg[0]) ;    // WriteGrad uses the name gx
   gx0 = &(gg0[0]) ;
   
# define ID -1
   
   for(int iter=0; iter<MAX_ITER; iter++) {
      
      
      // if (id==11) printf("iter %d\n", iter) ;
      
      for(int i=0; i<NP; i++) gx[i] = 0.0f ;
      
      // The following line is replaced with inlined code to calculate the gradient
      //   parameter vector must be called "x", needs variables v, y, tmp
      //@GRAD
      
# if (0)
      // check gradient against finite differences
      // GRAD return df/dp,   dchi2/dp = (2/dy^2) Sum{   (f-Y1) * gx  }
      if (id==ID) {
	 for(int i=0; i<NP; i++) printf("  --  %8.4f", x[i]) ;  printf("\n") ;
	 const float dd = 2.0e-3f ;
	 for(int i=0; i<NP; i++) {
	    x[i] += dd ;
	    fB    =   Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
	    x[i] -= 2*dd ;
	    fA    =   Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
	    x[i] += dd ;
	    if (ID==id) {
	       printf(" gx[%d]  %12.4e  vs. difference %12.4e .... % 12.4e %12.4e\n",
		      i, gx[i], (fB-fA)/(2*dd), fA, fB) ;
	    }
	 }
      }
      // return ;
# endif
      
      if (iter==0) {
	 beta = ZERO ;  for(int i=0; i<NP; i++) s[i] = ZERO ;
      } else {
	 NO = ZERO ;   DE = ZERO ;
# if (POLAK>0) // Polak-Ribiere
	 for(int i=0; i<NP; i++) {
	    NO += (gx[i]-gx0[i])*gx[i] ;    DE += gx0[i]*gx0[i] ;
	 }
# else  // Fletcher-Reeves
	 for(int i=0; i<NP; i++) {
	    NO += gx[i]*gx[i] ;             DE += gx0[i]*gx0[i] ;
	 }
# endif
	 // clamp needed for -polak 0 ??
	 // beta = max(0.0f, clamp(NO/DE, 0.0f, 0.5f)) ;
	 beta = max(0.0f, NO/DE) ;
      }
      
      // beta = 0.0f ;
      
      if (BAD||(iter%NP==0))  beta = ZERO ;
      NO = sqrt(NO) ;
      if (NO<EPS_GRAD) {
	 if (id==ID) printf("EXIT WITH GRAD %10.3e\n", NO) ;
	 break ;  // |grad| small enough
      }
      for(int i=0; i<NP; i++)  s[i] = -gx[i] + beta*s[i] ;
      
      
      
# if (BRENT==0)  // GOLDEN RATIO
      
      // line search with golden ratio
      A   =  &(x[0]) ;
      B   =  &(xx[0]) ;      
      // find an interval containing the minimum... minimum should be in direction s
      lA  =   0.0f ;  for(int i=0; i<NP; i++)  B[i] = A[i] + lA*s[i] ;
      fA  =   Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;
      lB  =   1.0e-5f ;   for(int i=0; i<NP; i++)  B[i] = A[i] + lB*s[i] ;
      fB  =   Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;
      for(int j=0; j<5; j++) {
	 if (fB>fA) break ;  // far enough for the function to again increase
	 lB  *=  4.0f;   for(int i=0; i<NP; i++)  B[i] = A[i] + lB*s[i] ;
	 fB  =   Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;
      }
      // if (id==ID) printf(" %10.3e  %10.3e    l = %10.3e %10.3e\n", fA, fB, lA, lB) ;
      h   =  lB - lA ;
      ASSERT(h>0.0f,"h") ;
      if (h<EPS_STEP) {
	 if (id==ID) printf("  step!\n") ;
	 for(int i=0; i<NP; i++) x[i] = A[i] + 0.5f*(B[i]-A[i]) ;
	 break ; // final interval [a,b]  -- break MAX_ITER loop
      }
      
      n   =  floor((log(EPS_STEP/h)) / log(invphi))  ;
      lC  = lA + invphi2 * h ;
      lD  = lA + invphi  * h ;
      for(int i=0; i<NP; i++)  B[i] = A[i] + lC*s[i] ;
      fC  =   Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;
      for(int i=0; i<NP; i++)  B[i] = A[i] + lD*s[i] ;
      fD  =   Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;      
      for(int k=0; k<n; k++) {
	 if (fC<fD) {  //  yc > yd to find the maximum
	    lB  =  lD ;  lD  =  lC ;   fD  =  fC ;
	    h   =  invphi * h ;        lC  =  lA + invphi2 * h ;
	    for(int i=0; i<NP; i++)    B[i] = A[i] + lC*s[i] ;
	    fC  =   Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;	    	       
	 } else {
	    lA  =  lC ;  lC  =  lD ;   fC  =  fD ;
	    h   =  invphi * h ;        lD  =  lA + invphi * h ;
	    for(int i=0; i<NP; i++)    B[i] = A[i] + lD*s[i] ;
	    fD  =   Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;	    
	 }
#  if (ID>=0)
	 if (id==ID) {
	    printf(" %11.4e %11.4e %11.4e %11.4e   %10.7e %10.7e %10.7e %10.7e\n",
		   lA, lB, lC, lD,  fA, fB, fC, fD) ;
	    printf("      %12.6f   %12.6f   %12.6f   %12.6f\n", x[0], x[1], x[2], x[3]) ;
	 }
#  endif
	 
      }      
      if (fC<fD) {
	 for(int i=0; i<NP; i++) x[i] = A[i] + HALF*(lA+lD)*s[i] ;
      } else {
	 for(int i=0; i<NP; i++) x[i] = A[i] + HALF*(lC+lB)*s[i] ;
      }
#  if (ID>=0)
      if (id==ID) {
	 printf(" %11.4e %11.4e %11.4e %11.4e   %10.7e %10.7e %10.7e %10.7e\n",
		lA, lB, lC, lD,  fA, fB, fC, fD) ;
	 printf("      %12.6f   %12.6f   %12.6f   %12.6f\n", x[0], x[1], x[2], x[3]) ;
	 printf("    g = %12.4e %12.4e %12.4e %12.4e\n", gx[0], gx[1], gx[2], gx[3]) ;
      }
#  endif
      
      
      
# else // Brent's method ================================================================================
      
      
      A   =  &(x[0]) ;
      B   =  &(xx[0]) ;            
      // find an interval containing the minimum... minimum should be in direction s
      ax  =  -1.0e-6f ;   for(int i=0; i<NP; i++)  B[i] = A[i] + ax*s[i] ;
      fz  =   Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;
      cx  =   1.0e-6f ;   for(int i=0; i<NP; i++)  B[i] = A[i] + cx*s[i] ;
      fu  =   Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;
      for(int j=0; j<9; j++) {
	 if (fu>fz) break ;  // far enough for the function to again increase
	 cx  *=  4.0f;   for(int i=0; i<NP; i++)  B[i] = A[i] + cx*s[i] ;
	 fu  =   Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;
      }
      z   =  fu ; 
      bx  =  1.0e-7f  ;
      fu  =  Chi2_1(A, V1, Y1, dY1, VHF1, IHF1, aux) ;
      
      if (id==1) printf(" %10.3e < %10.3e < %10.3e    %12.4e > %12.4e < %12.4e\n",
			ax, bx, cx,   fz, fu, z) ;
      ASSERT((fu<fz)&&(fu<z),"f") ;
      
      // must have  ax < bx < cx   and   f(bx)<f(ax) and f(bx)<f(cx)
      // a = (ax < cx ? ax : cx) ;
      // b = (ax > cx ? ax : cx) ;
      z  = w = v = bx ;
      // fw = fv = fz = (*f)(z) ;
      for(int i=0; i<NP; i++)  B[i] = A[i] + z*s[i] ;
      fz  =  Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;      
      fw = fz ;  fv = fz ;
      
      for (int iter=0; iter<MAX_ITER; iter++) {
	 if (id==ID) printf("iter %d\n", iter) ;
	 xm   = 0.5f*(a+b);
	 tol1 = tol*fabs(z)+ZEPS ;
	 tol2 = 2.0f*tol1 ;
	 if (fabs(z-xm) <= (tol2-0.5f*(b-a))) {
	    if (id==ID) printf("EXIT TOLERANCE\n") ;
	    // xmin = z ;
	    // return fz;
	    if (iter>2) break ;
	 }
	 if (fabs(e) > tol1) {
	    r = (z-w)*(fz-fv);   q = (z-v)*(fz-fw);    p = (z-v)*q-(z-w)*r;    q = 2.0f*(q-r);
	    if (q > 0.0f) p = -p;
	    q = fabs(q);      etemp = e;      e = d;
	    if (fabs(p)>=fabs(0.5f*q*etemp) || p<=q*(a-z) || p>=q*(b-z)) {
	       d = CGOLD*(e=(z>=xm ? a-z : b-z)) ;
	    } else {
	       d = p/q ;  u = z+d ;
	       if ((u-a<tol2)||(b-u<tol2))  d=SIGN(tol1,xm-z);
	    }
	    if (id==ID) printf(":: PARABOLOID\n") ;
	 } else {
	    d = CGOLD*(e=(z>=xm ? a-z : b-z)) ;
	    if (id==ID) printf(":: GOLDEN RATIO\n") ;
	 }
	 u  = (fabs(d)>=tol1 ? z+d : z+SIGN(tol1,d)) ;
	 // fu = (*f)(u);
	 for(int i=0; i<NP; i++)  B[i] = A[i] + u*s[i] ;
	 fu  =  Chi2_1(B, V1, Y1, dY1, VHF1, IHF1, aux) ;      
	 //
	 if (fu <= fz) {
	    if (u>=z)  a=z;   else  b=z ;
	    SHFT(v,w,z,u) ;
	    SHFT(fv,fw,fz,fu) ;
	 } else {
	    if (u<z)   a=u ;  else  b=u ;
	    if (fu<=fw || w==z) {
	       v=w;  w=u;  fv=fw;  fw=fu;
	    } else {
	       if (fu<=fv || v==z || v==w) {
		  v  = u;
		  fv = fu;
	       }
	    }
	 }
      } // for iter
      
      if (id==1) printf("  z=%12.4e ... for s = %12.4e %12.4e %12.4e %12.4e\n",
			z, s[0], s[1], s[2], s[3]) ;
      
      // result (z, fz)
      for(int i=0; i<NP; i++) x[i] = A[i]+z*s[i] ;
      
# endif //  GOLDEN RATIO  or  BRENT's METHOD
      
      
# if 0
      ptr = gx ;  gx = gx0 ; gx0 = ptr ;    // set  g -> g0 for the next iteration
# else
      for(int i=0; i<NP; i++) {
	 tmp = gx[i] ; gx[i] = gx0[i] ;  gx0[i] = tmp ;
      }
# endif
   } // for iter in MAX_ITER
   
   
# if (NP==6)
   // save in order of increasing velocity
   if (x[1]<x[4]) {
      for(int i=0; i<NP; i++) P[id*NP+i] = x[i] ;
   } else {
      for(int i=0; i<3; i++) {
	 P[id*NP+i]   = x[i+3] ;
	 P[id*NP+i+3] = x[i] ;
      }
   }
# else
   for(int i=0; i<NP; i++) P[id*NP+i] = x[i] ;
# endif
   f0      =   Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;	       
   C2[id]  =   f0 ;
}






void __kernel Simplex(__global float    *V1,   // X1[M1]      =  velocity [km/s] for first spectra
		      __global float    *YY1,  // YY1[N, M1]  =  Ta [K] for first spectra
		      __global float    *dYY1, // dYY1[N]     =  Ta [K] for first spectra                 
		      __global float    *V2,   // X2[M2]      =  velocity [km/s] for second spectra
		      __global float    *YY2,  // YY2[N, M2]  =  Ta [K] for second spectra
		      __global float    *dYY2, // dYY2[N]     =  Ta [K] for second spectra
		      __constant float  *VHF1, // V1[NHF1]    =  relative velocities of HFS components
		      __constant float  *IHF1, // I1[NHF1]    =  relative intensities of HFS components
		      __constant float  *VHF2, // V2[NHF2]    =  relative velocities of HFS components
		      __constant float  *IHF2, // I2[NHF2]    =  relative intensities of HFS components
		      __global float    *P,    // P[N, NP]    =  parameter vector
		      __global float    *C2,   // C[N]        =  final chi2 values C[N]
		      __global float    *AUX
		     ) {
   // Nelder-Mead Simplex optimisation algorithm
   const int id = get_global_id(0) ;             //  one work item = one spectrum
   if (id>=N) return ;                           //  only N spectra
   __global float  *Y1  =  &( YY1[id*M1]) ;     
   const float     dY1  =  dYY1[id] ;    
# if (TWIN>0) // if there is a second observed spectrum to be fitted simultaneously
   __global float  *Y2  =  &( YY2[id*M2]) ;
   const float     dY2  =  dYY2[id] ;
# endif
   float f0, f1  ;
   float f[NP+1] ;
   __global float *aux ;
# if (N_AUX>0)
   aux  =  &(AUX[id*N_AUX]) ;
# endif
   
   const float alpha=1.0f, gamma=2.0f, rho=0.5f, sigma=0.5f ;
   int n ;
   float x0[NP], xr[NP], xe[NP], xc[NP] ;
   // double fr, fe, fc  ; // not needed as double for accurate distributions but for faster convergence
   // These REAL not needed....
   float fr, fe, fc ;
   float y, s, s2 ;
   
   // Initial NP+1 vertices
   float  x[(NP+1)*NP] ;                        //  x[NP+1, NP]
   for(int i=0; i<NP; i++) x[i] = P[id*NP+i] ;  // "x[0]"
   for(int j=1; j<=NP; j++) {                   // "x[1]" ... "xlast"
      // larger initial Simplex => N2H+ high SN fits result in large chi2 values...
# if 0
      // for(int i=0; i<NP; i++) x[j*NP+i] = x[i] + ((i==j) ? 0.01f : -0.005f) ;
      for(int i=0; i<NP; i++) x[j*NP+i] = x[i] + ((i==j) ? 0.04f : -0.03f) ;
# else
      // Simplex *IS* sensitive to the initial Simplex
      // If it works, it works... if it fails, fails everywhere ??
      for(int i=0; i<NP; i++) x[j*NP+i] = x[i] + ((i==j) ? (0.05f+x[i]*0.01f) : (-x[i]*0.01f-0.04f)) ;
# endif
   }
   
   // Initial chi2 values for vertices
   for(int k=0; k<=NP; k++) {
      f0  =  Chi2_1(&(x[k*NP]), V1, Y1, dY1, VHF1, IHF1, aux) ;
# if (TWIN>0)
      f0 +=  Chi2_2(&(x[k*NP]), V2, Y2, dY2, VHF2, IHF2, aux) ;
# endif
      f[k] = f0 ;
   }
   
   
   //==========================================================================================
   // Nelder-Mead loop
   int iter = 0 ;
   
   while (1) {
      
      iter +=1  ;
      if (iter>2000) break ;
      
      f0  =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
      
      // sort x[NP+1,NP] and f[NP+1] in increasing order of function values
      for(int k=0; k<NP; k++) {
	 s = 1.0e30f ; n=-1 ;
	 for(int j=k+1; j<=NP; j++) {  // find minimum among [k+1, NP+1]
	    if (f[j]<s) { s = f[j] ;   n = j ; }
	 }
	 if (s<f[k]) {   // if smaller found, swap k and n
	    for(int i=0; i<NP; i++) {	       
	       s = x[k*NP+i] ;  x[k*NP+i] = x[n*NP+i] ;  x[n*NP+i] = s ;  // x[NP+1, NP]
	    }
	    s = f[k] ;      f[k] = f[n] ;    f[n] = s ;   // f[NK+1]
	 }
      }
      
      // calculate centroid of[0:NP]
      for(int i=0; i<NP; i++) {
	 s = 0.0f ;   for(int j=0; j<NP; j++)  s += x[j*NP+i] ;   x0[i] = s/NP ;
      }			    
      
      // reflected point based on x0 and last x,   xr = x0 + alpha*(x0-xlast)
      for(int i=0; i<NP; i++)   xr[i]  =  x0[i] + alpha*(x0[i]-x[NP*NP+i]) ;
      fr  =  Chi2_1(xr, V1, Y1, dY1, VHF1, IHF1, aux) ;
# if (TWIN>0)
      fr +=  Chi2_2(xr, V2, Y2, dY2, VHF2, IHF2, aux) ;
# endif      
      // not best but better than second worst => replace last with xr and start from beginning
      if ((fr>f[0])&&(fr<f[NP-1])) {
	 for(int i=0; i<NP; i++)  x[NP*NP+i] = xr[i] ;     f[NP] = fr ;
	 continue ;  // back to [1]
      }
      
      // expansion
      if (fr<f[0]) {
	 // calculate expanded point  xe = x0 + gamma*(xr-x0)
	 for(int i=0; i<NP; i++)  xe[i] = x0[i] + gamma*(xr[i]-x0[i]) ; 
	 fe  =  Chi2_1(xe, V1, Y1, dY1, VHF1, IHF1, aux) ;
# if (TWIN>0)
	 fe +=  Chi2_2(xe, V2, Y2, dY2, VHF2, IHF2, aux) ;
# endif
	 if (fe<fr) {   // replace last with xe
	    for(int i=0; i<NP; i++) x[NP*NP+i] = xe[i] ;    f[NP] = fe ;
	    continue ;
	 } else {       // replace last with xr
	    for(int i=0; i<NP; i++) x[NP*NP+i] = xr[i] ;    f[NP] = fr ;
	    continue ;
	 }
      }
      
      // contraction
      if (fr<f[NP]) {  //  xc = x0  + rho*(xr-x0)
	 for(int i=0; i<NP; i++)  xc[i] = x0[i] + rho*(xr[i]-x0[i]) ;
	 fc  =  Chi2_1(xc, V1, Y1, dY1, VHF1, IHF1, aux) ;
# if (TWIN>0)
	 fc +=  Chi2_2(xc, V2, Y2, dY2, VHF2, IHF2, aux) ;
# endif
	 if (fc<fr) { // replace last with xc
	    for(int i=0; i<NP; i++)  x[NP*NP+i] = xc[i] ;    f[NP] = fc ;
	    continue ;
	 } // else continue to shrink
      } else { // fr>=flast   ...   xc = x0 + rho*(xlast-x0)
	 for(int i=0; i<NP; i++)  xc[i] = x0[i]  +  rho*(x[NP*NP+i]-x0[i]) ;
	 fc  =  Chi2_1(xc, V1, Y1, dY1, VHF1, IHF1, aux) ;
# if (TWIN>0)
	 fc +=  Chi2_2(xc, V2, Y2, dY2, VHF2, IHF2, aux) ;
# endif
	 if (fc<f[NP]) {
	    for(int i=0; i<NP; i++)  x[NP*NP+i] = xc[i] ;  f[NP] = fc ;
	    continue ;
	 }
      }
      
      // shrink,   all except first => x[j] = x[0] + sigma*(x[j]-x[0])
      for(int j=1; j<=NP; j++) {
	 for(int i=0; i<NP; i++)  x[j*NP+i] = x[i] + sigma*(x[j*NP+i]-x[i]) ;
	 f0  =  Chi2_1(&(x[j*NP]), V1, Y1, dY1, VHF1, IHF1, aux) ;
# if (TWIN>0)
	 f0 +=  Chi2_2(&(x[j*NP]), V2, Y2, dY2, VHF2, IHF2, aux) ;
# endif
	 f[j] = f0 ;
      }
      
      f0 = 1.0f ;
      for(int j=0; j<NP+1; j++) {  
	 s = 0.0f ;  s2 = 0.0f ;
	 for(int i=0; i<NP; i++)  {
	    y = x[j*NP+i] ;   s += y ;  s2 += y*y ;	    
	 }
	 if (((s2-s*s/NP)/NP)>1.0e-6f)  f0 = -1.0f ;
      }
      if ((f0>0.0f)&&(iter>10)) break ;      
      
   } // while(1)
   
   
   //==========================================================================================
   
   
# if (NP==6)
   // save in order of increasing velocity
   if (x[1]<x[4]) {
      for(int i=0; i<NP; i++) P[id*NP+i] = x[i] ;
   } else {
      for(int i=0; i<3; i++) {
	 P[id*NP+i]   = x[i+3] ;
	 P[id*NP+i+3] = x[i] ;
      }
   }
# else
   for(int i=0; i<NP; i++) P[id*NP+i] = x[i] ;
# endif
   
   
   C2[id] =  f[0] ;  // UN-normalised chi2
}







//================================================================================

//================================================================================

//================================================================================



# if (USE_MCMC==1) // basic Metropolis

void __kernel MCMC(const float        SEED,
		   __global float      *V1,    //      X1[M1] velocities of first spectra
		   __global float     *YY1,    //   YY1[N,M1] intensities of first spectra
		   __global float    *dYY1,    //  dYY1[N]    error estimates
		   __global float      *V2,    //      X2[M2] velocities of second spectra
		   __global float     *YY2,    //   YY2[N,M2]
		   __global float    *dYY2,    //  dYY2[N]
		   __constant float  *VHF1,    //  VHF1[NHF1] velocities of HFS components, first spectra
		   __constant float  *IHF1,    //  IHF1[NHF1] velocities of HFS compoennts, first spectra
		   __constant float  *VHF2,    //  VHF2[NHF2]
		   __constant float  *IHF2,    //  IHF2[NHF2]
		   __global float    *P,       //  P[N, NP]
		   __global float    *ARES,    //  ARES[N, SAMPLES, NP+1] MCMC samples
		   __global float    *AUX,     //  AUX[N]
		   __global float    *KSTEP    //  KSTEP[NP]
		  ) {
   // Markov chain Monte Carlo, basic Metropolis algorithm
   const int id = get_global_id(0) ;               // one work item = one spectrum
   if (id>=N) return ;                             // only N spectra
   float x00[NP], x11[NP] ;                        // NP = number of parameters
   float *x0 = &(x00[0]), *x=&(x11[0]), *ptr ;
   float r1, r2, U1, U2 ;
   __global float  *Y1  =  &( YY1[id*M1]) ;        // pointer to the single spectrum (observations)
   const float     dY1  =  dYY1[id] ;              // single error estimate for entire spectrum
#  if (TWIN>0)
   __global float  *Y2  =  &( YY2[id*M2]) ;        // pointer to the single spectrum (observations)
   const float     dY2  =  dYY2[id] ; 
#  endif
   __global float *RES  =  &(ARES[id*SAMPLES*(NP+1)]) ; // RES[SAMPLES, NP+1]
   float lnP0, lnP1 ;
   float STEP=0.005f, ratio   ;
   int   iter=0, accepted=0 ;
   const int batch=100 ;
   float step[NP] ;
   __global float *aux ;
#  if (N_AUX>0)
   aux  =  &(AUX[id*N_AUX]) ;
#  endif
   
   mwc64x_state_t rng ;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
   
   for(int i=0; i<NP; i++) x0[i]=x[i]=P[id*NP+i] ;  // copy parameters for one spectrum to private memory
   for(int i=0; i<NP; i++) step[i] = clamp(fabs(x0[i]), 0.001f, 1.0f) ;
   // lnP0 =  -0.5f*F(p0, X, Y, VHF, IHF) ;
   lnP0 = -1e20f ;

   // set relative steps based ini "step" ini the INI file
   for(int i=0; i<NP; i++)  step[i] *= KSTEP[i] ;

#  if 0
#   if (NP==4)
   /* step[0]  =   5.3f ;    // Tex    black */
   /* step[1]  =   1.5f ;    // V      blue */
   /* step[2]  =   0.5f ;    // FWHM   green */
   /* step[3] *=  60.0f ;    // tau    red */
   step[0]  =   0.3f ;    // Tex    black
   step[1]  =   1.5f ;    // V      blue
   step[2]  =   0.5f ;    // FWHM   green
   step[3] *=  60.0f ;    // tau    red   
#   endif
#  endif

#  if 0
#  if (ADAPT==1)  // optimise step[]
   double s[NP], s2[NP], var ;
   for(int i=0; i<NP; i++) { s[i]=0.0 ; s2[i]=0.0 ; }
#  endif
# endif
   
   for(int ITER=0; ITER<BURNIN+SAMPLES*THIN; ITER++) {
      
      // Create proposal
      if (ITER==0) {
	 for(int i=0; i<NP; i++) x[i] = x0[i] ;
      } else {
	 for(int i=0; i<NP/2; i++) {
	    U1 = Rand(&rng) ; U2 = Rand(&rng) ;
	    r1 = sqrt(-2.0f*log(U1)) ; r2 = r1*cos(PI2*U2) ; r1 *= sin(PI2*U2) ;
	    x[2*i+0] = x0[2*i+0] + STEP*r1*step[i] ;
	    x[2*i+1] = x0[2*i+1] + STEP*r2*step[i] ;
	 }
#  if (NP%2==1) // last odd element
	 U1 = Rand(&rng) ; U2 = Rand(&rng) ;    r1 = sin(PI2*U2)*sqrt(-2.0f*log(U1)) ;
	 x[NP-1]     = x0[NP-1 ] + STEP*r1*step[NP-1] ;
#  endif
      }
      
      lnP1  =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ; // actually chi2, not yet lnP
#  if (TWIN>0)
      lnP1 +=  Chi2_2(x, V2, Y2, dY2, VHF2, IHF2, aux) ;
#  endif
      // chi2 =  sum(((y-yp)/dy)^2) - 2*log(bias) + penalty
      lnP1 = -0.5f*lnP1 ;   // lnP = -0.5*chi2            
      

#  if 0
      if (id==11) printf("%10.3e %10.3e %10.3e  %10.3e %10.3e %10.3e   %9.2f\n",
			x[0], x[1], x[2],  x[3], x[4], x[5],  lnP1) ;
#  endif
      
      if ((lnP1-lnP0)>log(Rand(&rng))) { // accept step
#  if 0
	 // the following line does not work on AMD ????
	 // ptr = x ;  x = x0 ;  x0 = ptr ;  // make current p the p0
	 ptr = &(x[0]) ;  x = &(x0[0]) ;  x0 = &(ptr[0]) ;
#  else
	 for(int i=0; i<NP; i++) x0[i] = x[i] ;
#  endif
	 lnP0  =  lnP1 ;
	 accepted += 1 ;	 
      }
      

      // why is 1st stored chi2 abnormal? ~18-19, next ones <0.1 ????
      
      if (ITER<BURNIN) { // ##############################################################
	 
	 // "normal" adjustment of the scalar STEP
	 if (ITER % batch == 0) {
	    ratio = accepted*100.0f/batch ; // acceptance ratio [%]
	    if (ratio<20.0f) {
	       STEP *= 0.6f ;
	    } else {
	       if (ratio>40.0f) {
		  STEP *= 1.5f ;
	       }
	    }
	    accepted = 0 ;
	 } // ITER % batch ==0

#if 0	 
#  if (ADAPT==1)
	 // updating step[] based on the estimated stndr deviations
	 for(int i=0; i<NP; i++) {
	    var    =  (double)x0[i] ;      s[i]  +=  var ;         s2[i] +=  var*var ;
	 }        
	 if ((ITER>0)&&(ITER%500==0)) {
	    for(int i=0; i<NP; i++) {
	       if (id==932) printf("%6d - %d-- s2 = %10.3e, s = %10.3e, s2-s*s/N = %10.3e -- STEP %.3e\n",
				   ITER, i, s2[i], s[i],  s2[i]-s[i]*s[i]/(5.0*batch), STEP) ;
	       var      =  (s2[i] - s[i]*s[i]/500.0) / 500.0 ;
	       step[i]  =  sqrt(clamp(var, 1.0e-10, 1e1)) ;  // new step size
	       s2[i]    =  0.0 ; 
	       s[i]     =  0.0 ;
	    }
	    r1 = 0.0f ;  for(int i=0; i<NP; i++)  r1 += step[i]*step[i] ;
	    r1 = 1.0f/sqrt(r1) ;
	    for(int i=0; i<NP; i++) step[i] = r1*step[i] ;  // normalise step since STEP is adjusted
	 }
#  endif
# endif
	 
      } else { // else ITER>=BURNIN #########################################################
	 
	 if ((ITER-BURNIN)%THIN==0) {
	    // register sample ITER/THIN
	    int j = (ITER-BURNIN)/THIN ;                       // sample
	    // ASSERT(j<SAMPLES, "samples") ;
	    for(int i=0; i<NP; i++) RES[j*(NP+1)+i] = x0[i] ;  // RES[SAMPLES, NP+1]
	    RES[j*(NP+1)+NP] = -2.0f*lnP0 ;                    // chi2 value
	 }
	 
      } // ################################################################################
   }
   
}

# endif





# if (USE_MCMC==2)  // use conditional compilation... MCMC_RAM may require too much memory

void __kernel MCMC_RAM(const float        SEED,
		       __global   float  *V1,      //      V1[M1]  velocities of first spectra
		       __global   float  *YY1,     //   YY1[N,M1]  intensities of first spectra
		       __global   float  *dYY1,    //   dYY1[N]     error estimates
		       __global   float  *V2,      //    V2[M2]  velocities of second spectra
		       __global   float  *YY2,     //  YY2[N,M2]
		       __global   float *dYY2,     //   dYY2[N]
		       __constant float  *VHF1,    //  VHF1[NHF1]  velocities of HFS components, first spectra
		       __constant float  *IHF1,    //  IHF1[NHF1]  velocities of HFS compoennts, first spectra
		       __constant float  *VHF2,    //  VHF2[NHF2]
		       __constant float  *IHF2,    //  IHF2[NHF2]
		       __global   float     *P,    //  P[N, NP]
		       __global   float  *ARES,    // ARES[N, SAMPLES, NP+1] MCMC samples
		       __global   float  *AUX,
		       __global   float  *dummy
		      ) {
   // Markov chain Monte Carlo, Robust Adaptive MCMC (Haario et al. 2001)
   const int id  = get_global_id(0) ;              // one work item = one spectrum
   const int lid = get_local_id(0) ;
   if (id>=N) return ;                            // only N spectra
   float x00[NP], x11[NP] ;                        // NP = number of parameters
   float *x0 = x00, *x=x11, *ptr ;
   float r1, r2, U1, U2, eta ;
   __global float  *Y1  =  &( YY1[id*M1]) ;        // pointer to the single spectrum (observations)
   const float     dY1  =  dYY1[id] ;              // single error estimate for entire spectrum
#  if (TWIN>0)
   __global float  *Y2  =  &( YY2[id*M2]) ;        // pointer to the single spectrum (observations)
   const float     dY2  =  dYY2[id] ; 
#  endif
   __global float *RES =  &(ARES[id*SAMPLES*(NP+1)]) ; // RES[SAMPLES, NP+1]
   float lnP0, lnP1, lnA, k  ;
   float rate, z ;
   int   iter=0, accepted=0 ;
   const int BATCH=1 ;
   __global float *aux ;
#  if (N_AUX>0)
   aux  = &(AUX[id*N_AUX]) ;
#  endif
   mwc64x_state_t rng ;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
   
   for(int i=0; i<NP; i++) x0[i]   = P[id*NP+i] ;  // copy parameters for one spectrum to private memory
   
#  define WITH_SCALED_STEPS 0
#  if (WITH_SCALED_STEPS>0)
   float step[NP] ;
   for(int i=0; i<NP; i++) step[i] = clamp(0.1f*x0[i], 0.0001f, 1.0f) ;
#  endif
   
   // lnP0 =  -0.5f*F(p0, X, Y, VHF, IHF) ;
   lnP0 = -1e20f ;
   
   // const float rate0 = 0.234 ;
   const float rate0 = 0.4 ;
   
   __local float        L_S[LOCAL*NP*NP] ;
   __local float        L_Z[LOCAL*NP*NP] ;
   __local float        L_C[LOCAL*NP*NP] ;
   __local float        L_U[LOCAL*NP] ;
   __local float *S = &(L_S[lid*NP*NP]) ;
   __local float *Z = &(L_Z[lid*NP*NP]) ;
   __local float *C = &(L_C[lid*NP*NP]) ;
   __local float *U = &(L_U[lid*NP]) ;
   // initial matrix S
   for(int j=0; j<NP; j++) {
      for(int i=0; i<NP; i++) {
	 S[j*NP+i] = (i==j) ? (0.01f) : (0.0f) ;
      }
   }
   
   
   for(int ITER=0; ITER<BURNIN+SAMPLES*THIN; ITER++) {
      
#  if 0  // uniform random numbers
      for(int i=0; i<NP; i++) U[i] = Rand(&rng)-0.5f ;
#  else  // normal-distributed random numbers
      for(int i=0; i<NP/2; i++) {
	 U1 = Rand(&rng) ;  U2 = Rand(&rng) ;
	 r1 = sqrt(-2.0f*log(U1)) ;   r2 = r1*cos(PI2*U2) ;   r1 *= sin(PI2*U2) ;
	 U[2*i+0] = r1 ;    U[2*i+1] = r2 ;
      }
#   if (NP%2==1) // last odd element
      U1 = Rand(&rng) ; U2 = Rand(&rng) ;  
      U[NP-1]  = sin(PI2*U2)*sqrt(-2.0f*log(U1)) ;
#   endif
#  endif
      
      // x = x0 + S*U
      for(int j=0; j<NP; j++) {
	 x[j] = x0[j] ;
#  if (WITH_SCALED_STEPS>0)
	 for(int i=0; i<=j; i++)  x[j] += step[j]*S[j*NP+i]*U[i] ;  // x[j] = x0[j] + S[j,i]*U[i]
#  else
	 for(int i=0; i<=j; i++)  x[j] += S[j*NP+i]*U[i] ;  // x[j] = x0[j] + S[j,i]*U[i]
#  endif
      }
      
      
      lnP1  =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
#  if (TWIN>0)
      lnP1 +=  Chi2_2(x, V2, Y2, dY2, VHF2, IHF2, aux) ;
#  endif
      lnP1 = -0.5f*lnP1 ;   // lnP = -0.5*chi2
      lnA  = lnP1-lnP0 ;
      if (lnA>log(Rand(&rng))) { // accept step
	 // ptr = x ;  x = x0 ;  x0 = ptr ;  // make current p the p0 --- does not work on AMDGPU ???
	 for(int i=0; i<NP; i++) x0[i] = x[i] ;
	 lnP0      =  lnP1 ;
	 accepted +=  1 ;	 
      }
      
      
      if (ITER>=BURNIN) {
	 if ((ITER-BURNIN)%THIN==0) {
	    // register sample ITER/THIN
	    int j = (ITER-BURNIN)/THIN ;                       // sample	    
	    for(int i=0; i<NP; i++) RES[j*(NP+1)+i] = x0[i] ;  // RES[SAMPLES, NP]
	    RES[j*(NP+1)+NP] = -2.0f*lnP0 ;                    // chi2 value
	 }
      }      
      
      
      rate = accepted/ITER ;
      
      // Calculate  C = S *  (I + eta*(alpha-alpha*)(U*U')/||U||^2)  * S'
      //   - eta
      //   - alpha  = current acceptance rate
      //   - alpha* = target acceptance rate ~ 0.234
      //   - U ~ random vector
      if ((ITER<3)||(eta<1.0e-9f)) continue ;
      eta =  pow((float)ITER, -0.6f) ;
      
      // I + eta*(rate-rate0) * (U*U') / ||U||^2
      z   =  0.0f ;  for(int i=0; i<NP; i++)  z += U[i]*U[i] ;   z = 1.0/z ;
      z  *=  eta*(rate-rate0) ;
      //   C =  I + z * U*U',  NPxNP matrix,     z = eta*(rate-rate0)/||U||^2
      for(int j=0; j<NP; j++) {
	 for(int i=0; i<NP; i++) {
	    C[j*NP+i]  =  z*U[j]*U[i] + ((i==j) ? 1.0f : 0.0f) ; // I + norm*U*U'
	 }
      }      
      //  Z = S*C, where C = (I+eta*(rate-rate0))*(U*U')/||U||^2
      for(int j=0; j<NP; j++) {
	 for(int i=0; i<NP; i++) {
	    z = 0.0f ;
	    for(int k=0; k<NP; k++) {
	       z += S[j*NP+k]*C[k*NP+i] ;  // Z[j,i] = (S*C) [j,i]
	    }
	    Z[j*NP+i] = z ;  // S*C
	 }
      }           
      // C = Z*S'
      for(int j=0; j<NP; j++) {
	 for(int i=0; i<NP; i++) {
	    z = 0.0f ;
	    for(int k=0; k<NP; k++) {
	       z += Z[j*NP+k]*S[i*NP+k] ;  // (S*C) * S',   where S'[k,i] = S[i,k]
	    }
	    C[j*NP+i] = z ;  //  C = (S*C)*S'
	 }
      }            
      // Calculate new S through Cholesky decomposition of S*S'=C
      for(int j=0; j<NP; j++) {
	 float sum = 0.0f ;
	 for (int k=0; k<j; k++) {
	    sum += S[j*NP+k] * S[j*NP+k] ;
	 }
	 S[j*NP+j] = sqrt(C[j*NP+j] - sum) ;	   
	 for(int i=j+1; i<NP; i++) {
	    sum = 0.0f ;
	    for(int k=0; k<j; k++) {
	       sum += S[i*NP+k] * S[j*NP+k];
	    }
	    S[i*NP+j] = (1.0/S[j*NP+j] * (C[i*NP+j] - sum)) ;
	 }
      }
      
   } // for ITER
   
} // RAM

# endif




# if (USE_MCMC==3)

void __kernel HMCMC(const float        SEED,
		    __global float      *V1,    //      X1[M1] velocities of first spectra
		    __global float     *YY1,    //   YY1[N,M1] intensities of first spectra
		    __global float    *dYY1,    //  dYY1[N]    error estimates
		    __global float      *X2,    //      X2[M2] velocities of second spectra
		    __global float     *YY2,    //   YY2[N,M2]
		    __global float    *dYY2,    //  dYY2[N]
		    __constant float  *VHF1,    //  VHF1[NHF1] velocities of HFS components, first spectra
		    __constant float  *IHF1,    //  IHF1[NHF1] velocities of HFS compoennts, first spectra
		    __constant float  *VHF2,    //  VHF2[NHF2]
		    __constant float  *IHF2,    //  IHF2[NHF2]
		    __global   float     *P,    // P[N, NP]
		    __global   float  *ARES,    // ARES[N, SAMPLES, NP+1] MCMC samples
		    __global   float  *AUX
		   ) {
   // Hamiltonian Markov chain Monte Carlo
   // Expensive... and fails systematically for weak spectra (some bugs perhaps...)
   const int id = get_global_id(0) ;              // one work item = one spectrum
   if (id>=N) return ;                            // only N spectra
   float x00[NP], x11[NP], p00[NP], p01[NP] ;     // NP = number of parameters
   float *x0 = x00, *x=x11, *p0=p00, *p=p01, *ptr ;
   float r1, r2, U1, U2 ;
   __global float  *Y1  =  &( YY1[id*M1]) ;        // pointer to the single spectrum (observations)
   const float     dY1  =  dYY1[id] ;              // single error estimate for entire spectrum
#  if (TWIN>0)
   __global float  *Y2  =  &( YY2[id*M2]) ;        // pointer to the single spectrum (observations)
   const float     dY2  =  dYY2[id] ; 
#  endif
   __global float *RES =  &(ARES[id*SAMPLES*(NP+1)]) ; // RES[SAMPLES, NP+1]
   __global float *aux ;
#  if (N_AUX>0)
   aux =  &(AUX[id*N_AUX]) ;
#  endif
   float gx[NP], M[NP], H, H0=1e-30f, chi2   ;
   float y, v, tmp ;
   
   const float step = 0.005f ;
   const int nstep  = 10 ;
   
   mwc64x_state_t rng ;
   ulong samplesPerStream = 274877906944L ;  // 2^38
   MWC64X_SeedStreams(&rng, (unsigned long)(fmod(SEED*7*id,1.0f)*4294967296L), samplesPerStream);
   
   for(int i=0; i<NP; i++) {
      x0[i] = P[id*NP+i] ;  // copy parameters for one spectrum to private memory
      M[i]  = 1.0f ;
   }
   
   
   
   for(int ITER=0; ITER<BURNIN+SAMPLES*THIN; ITER++) {
      
      // Random momentum variables
      for(int i=0; i<NP/2; i++) {
	 U1 = Rand(&rng) ; U2 = Rand(&rng) ;
	 r1 = sqrt(-2.0f*log(U1)) ; r2 = r1*cos(PI2*U2) ; r1 *= sin(PI2*U2) ;
	 p[2*i] = 2.5f*r1 ;   p[2*i+1] = 2.5f*r2 ;
      }
      if (NP%2==1) { // last odd variable
	 U1 = Rand(&rng) ; U2 = Rand(&rng) ;
	 r1 = sqrt(-2.0f*log(U1)) * sin(PI2*U2) ;
	 p[NP-1] = 2.5f*r1 ;
      }
      
      // x0 = old values
      for(int i=0; i<NP; i++) x[i] = x0[i] ;
      
      /* gradient
       for(int i=0; i<M1; i++) {
       v     =  V1[i] ;
       y     =  GAUSS(v, x[0], x[1], x[2]) ;
       tmp   =  (y-Y1[i]) / (dY1*dY1) ;
       * 
       gx[0] +=  tmp * (y/x[0]) ;
       gx[1] +=  tmp * 2.0f*Q*(v-x[1])/(x[2]*x[2]) * y ;
       gx[2] +=  tmp * 2.0f*Q*(v-x[1])*(v-x[1])*y/pown(x[2],3) ;
       }   
       */
      
      // leapfrog
      for(int istep=0; istep<nstep; istep++) {
	 // gradients at initial position       grad(U)
	 for(int i=0; i<NP; i++) gx[i] = 0.0f ;
	 //@GRAD
	 // update momentum for half step       p -= (dt/2) * grad(U)
	 for(int i=0; i<NP; i++) p[i] -= 0.5f*step*gx[i] ;
	 // update parameters                   x += dt * M^-1 * p
	 for(int i=0; i<NP; i++) x[i] += step*p[i]/M[i] ;
	 // gradients with new x                grad(U)
	 for(int i=0; i<NP; i++) gx[i] = 0.0f ;	
	 //@GRAD
	 // update momentum second half step    p -= (dt/2) * grad(U)
	 for(int i=0; i<NP; i++) p[i] -= 0.5f*step*gx[i] ;
      }
      
      // accept, if    rand() <  exp(-H_new) / exp(-H_old)
      // H = U + 0.5*p'*M*p
      H = 0.0f ;
      for(int i=0; i<NP; i++)  H += 0.5f*p[i]*p[i]/M[i] ;
      
      chi2  =  Chi2_1(x, V1, Y1, dY1, VHF1, IHF1, aux) ;
#  if (TWIN>0)
      chi2 +=  Chi2_2(x, V2, Y2, dY2, VHF2, IHF2, aux) ;
#  endif
      // H += U,  where  U = -ln f = +0.5*chi2
      H += 0.5f*chi2 ;  // 
      
      if (Rand(&rng)<(H/H0)) { // accept step
	 H0 = H ;
	 ptr = p0 ; p0 = p ; p = ptr ;
	 ptr = x0 ; x0 = x ; x = ptr ;
      }
      
      // Register
      if ((ITER>=BURNIN)&&((ITER-BURNIN)%THIN==0)) {
	 // register sample ITER/THIN
	 int j = (ITER-BURNIN)/THIN ;                       // sample	    
	 for(int i=0; i<NP; i++) RES[j*(NP+1)+i] = x0[i] ;  // RES[SAMPLES, NP]
	 RES[j*(NP+1)+NP] = -2.0f*chi2 ;                    // chi2 value
      }
      
      
   } // for ITER
   
   
}

# endif




