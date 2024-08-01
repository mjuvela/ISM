
#if (DOUBLE>0)
# define real double
#else
# define real float
#endif

#define h0  6.62606957e-27f 
#define kb  1.3806488e-16f
#define c0  29979245800.0f
#define hk  4.799243348e-11f
#define PI  3.1415926536f

float Planck(const float f, const float T) {
   return (2.0f*(h0*f)*pown(f/c0,2)) / (exp((hk*f)/T)-1.0f) ;
}


int Doolittle_LU_Decomposition_with_Pivoting(__global real *A, __global ushort *pivot, int n) {
   // pivot = n elements per work item, can be >> 32kB per work group
   // Difficult to defactor and make efficient for a work group =>
   // entirely separate problems for each work item !!!
   int i, j, k ;
   __global real *p_k, *p_row, *p_col ;
   real maxi;
   // For each row and column, k = 0, ..., n-1,
   for (k=0, p_k=A; k<n; p_k+=n, k++) {
      // find the pivot row
      pivot[k] = k;
      maxi     = fabs( p_k[k] );
      for (j=k+1, p_row = p_k+n; j<n; j++, p_row += n) {
         if (maxi < fabs(p_row[k])) {
            maxi      =  fabs(p_row[k]) ;
            pivot[k]  =  j;
            p_col     =  p_row;
         }          
      }             
      // and if the pivot row differs from the current row, then interchange the two rows.
      if (pivot[k] != k) {
         for (j = 0; j < n; j++) {
            maxi       = *(p_k+j);
            *(p_k+j)   = *(p_col+j);
            *(p_col+j) = maxi ;
         }
      }
      // and if the matrix is singular, return error
      if (p_k[k]==0.0) return -1 ;
      // otherwise find the lower triangular matrix elements for column k
      for (i=k+1, p_row=p_k+n; i<n; p_row+=n, i++) {
         p_row[k]  /=  p_k[k] ;
      }
      // update remaining matrix
      for (i=k+1, p_row=p_k+n; i<n; p_row+=n, i++) {
         for (j=k+1; j<n; j++) {
            p_row[j] -= p_row[k] * p_k[j] ;
         }
      }
   }
   return 0 ;
}     


int Doolittle_LU_with_Pivoting_Solve(__global real *A,       __global real *B,
                                     __global ushort *pivot, __global real *x,  int n) {
   int i, k;
   __global real *p_k;
   real dum;
   // Solve Lx = B for x, where L is a lower triangular matrix with an implied 1 along the diagonal.
   for (k=0, p_k = A; k<n; p_k+=n, k++) {
      if (pivot[k]!=k) {
         dum = B[k];    B[k] = B[pivot[k]];    B[pivot[k]] = dum;
      }
      x[k] = B[k];
      for (i = 0; i < k; i++) x[k] -= x[i] * p_k[i] ;
   }
   // Solve the linear equation Ux = y, where 
   // y is the solution obtained above of Lx = B and U is an upper triangular matrix.
   for(k=n-1, p_k=A+n*(n-1); k>=0; k--, p_k-=n) {
      if (pivot[k]!=k) {
         dum = B[k];    B[k] = B[pivot[k]];    B[pivot[k]] = dum;
      }
      for (i=k+1; i<n; i++) x[k] -= x[i] * p_k[i] ;
      if (p_k[k]==0.0) return -1;
      x[k]  /=  p_k[k] ;
   }
   return 0;
}


__kernel void Step(__constant real   *E,        // E[NL] level energies 
		   __constant real   *G,        // G[NL] level statistical weights
		   __constant real   *F,        // F[NT] transition frequencies 
		   __constant real   *A,        // A[NT] transition Einstein A coefficients
		   __global real     *IBG,      // A[NT] background intensity
		   __global real     *C,        // C[NL,NL] collisional coefficients (current Tkin) (constant -> slower ?)
		   __global int      *UL,       // UL[NT,2] {u,l} for given transition
		   __global int      *TR,       // TR[NL,NL] mapping {u,l}->t
		   __global real     *RHO,      // RHO[NRHO]
		   __global real     *CD,       // CD[NCD]  .... CD == N/DV
		   __global real     *XX,       // XX[NCD, NRHO, NL, NL] rate matrices
		   __global real     *bb,       // bb[NCD, NRHO, NL] right hand sides
		   __global real     *nn,       // nn[NCD, NRHO, NL] level population vectors
		   __global real     *TEX,      // TEX[NCD, NRHO, NT] excitation temperatures
		   __global real     *TAU,      // TAU[NCD, NRHO, NT] line optical depths
		   __global real     *TRAD,     // TRAD[NCD, NRHO, NT] radiation temperatures
		   __global real     *VV,       // VV[NCD, NRHO, 2*NL] work space
		   __global ushort   *PP,       // PP[NCD, NRHO, NL]
		   __global real     *HFSC
		  ) {     
   const int id  = get_global_id(0) ;
   const int lid = get_local_id(0) ;
   if (id>=(NRHO*NCD)) return ;
   int u, l, t, ITER=0 ;
   real tau, Bul, Blu, tex, trad, beta ;
   const real rho    = RHO[id % NRHO] ;   // ~  [NCD, NRHO]
   const real colden = CD[id/NRHO]  ;     // N(mol)/FWHM * C * sqrt(4*log(2)/pi)
   __global real    *X    =  &(XX[id*NL*NL]) ;
   __global real    *b    =  &(bb[id*NL]) ;
   __global real    *n    =  &(nn[id*NL]) ;
   __global real    *V    =  &(VV[              id*NL]) ;
   __global real    *BETA =  &(VV[NCD*NRHO*NL + id*NL]) ;
   __global ushort  *P    =  &(PP[id*NL]) ;   
   for(int i=0; i<NL; i++) BETA[i] = 0.5f ;   
   for(ITER=0; ITER<MAXITER; ITER++) {      
      for(int u=1; u<NL; u++) {
	 for(int l=0; l<u; l++) {	    
	    // collisional coefficients in both directions
	    X[l*NL+u]  =  C[l*NL+u] * rho ;   // X[l,u] = C[l,u] = C(u->l)
	    X[u*NL+l]  =  C[u*NL+l] * rho ;   // X[u,l] = C[u,l] = C(l->u)
	    t = TR[u*NL+l] ;    // i->j is a downward radiative transition?  TR[u,l] = tr(u->l)
	    if (t>=0) {         // downward transition exists ---   X[l,u]  ~  u->l
	       // recompute escape probability for transition u->l == t
	       Bul =  A[t] / (2.0f*h0*pown(F[t]/c0,2)*F[t]) ;
	       Blu =  Bul * G[u] / G[l] ;
	       tau =  (h0*colden/(4.0f*PI)) * ( n[l]*Blu - n[u]*Bul )  ;

	       if ((id==0)&&(ITER==0)) {
		  printf(" tau[%d] = %.3e   nu %.3e  nl %.3e Aul %.3e  Blu %.3e  co %.3e\n", t, tau, n[u], n[l], A[t], Blu, colden) ;
	       }

	       
#if (ESC==-1)   // LVG - tested
	       if (tau<14.0f) {  // the limit is important
		  beta = (fabs(tau)<0.001f) ?  (1.0f+0.585f*tau) : ((1.0f-exp(-1.17f*tau))/(1.17f*tau)) ;
	       } else {
		  beta = 1.0f/(tau*sqrt(log(0.5f*tau/sqrt(PI)))) ;
	       }
#endif
#if (ESC==0)   // LVG - tested
	       beta = (tau<14.0f) ?
		 ((fabs(tau)<0.001f) ?  (1.0f+0.585f*tau) : ((1.0f-exp(-1.17f*tau))/(1.17f*tau)))  :
		 (1.0f/(tau*sqrt(log(0.5f*tau/sqrt(PI))))) ;
#endif
#if (ESC==1)   //   ~ ad hoc simple    (1-exp(-tau)) / tau
	       beta = (fabs(tau)>0.001f) ? ((1.0f-exp(-tau))/tau) : (1.0f-0.5f*tau) ;
#endif
#if (ESC==2)   // Slab  (1-exp(-3*tau)) / (3*tau)
	       beta = (
		       (fabs(tau)<0.01f) ?
		       (1.0f-1.5f*(tau+tau*tau))  :
		       ((1.0f-exp(-3.0f*tau)) / (3.0f*tau)) 
		      ) ;
#endif
#if (ESC==3)   // Uniform sphere -- tested
	       if (fabs(tau)<0.1f) {
		  beta = 1.0f + tau * (-0.375f + tau * ( 0.1f + tau * ( -1.333f + 0.0035714f * tau))) ;
	       } else {
		  if (fabs(tau)>100) {
		     beta = 1.5f/tau ;
		  } else {
		     beta = 1.5f/tau * (1.0f-2.0f/(tau*tau)+(2.0f/tau+2.0f/(tau*tau))*exp(-tau)) ;
		  }
	       }
#endif
#if (HFSTR>=0)
	       if (t==HFSTR) {
		  beta *=  HFSC[clamp((int)(log(beta/HFS0) / log(HFSK)), 0, HFSN-1)] ;  // corrected escape probability
	       }
#endif
#if 1          // dampened beta
	       beta    =  clamp(beta, (real)0.0f, (real)1.0) ;
	       beta    =  (1.0f-ALFA)*BETA[t] + ALFA*beta ;
	       BETA[t] =  beta ;    // old beta
#endif	       
#if 0
	       //  printf("beta = %.3e   nl %10.3e   nu %10.3e   Bul %10.3e   Blu %10.3e  CD %12.4e\n", beta, n[l], n[u], Bul, Blu, colden) ;
	       beta = 0.5f ;
#endif	       
	       // update rate matrix X[l,u] = transition downwards u -> l
	       X[l*NL+u] += beta * (A[t] + Bul * IBG[t]) ;	       
	       // update rate matrix X[u,l] = transition upwards l-> u
	       X[u*NL+l] += beta *         Blu * IBG[t] ;	       
	    } // if t>0  == radiative transitions
	 } // for j
      } // for i

#if 0
      if ((id==0)&&(ITER==0)) {
	 printf("ALFA = %.3e\n", ALFA) ;
	 for(int i=0; i<NT; i++) printf("BETA[%d] = %.3e\n", i, BETA[i]) ;
      }
#endif
      // Set diagonal elements based on column sums
      for(int i=0; i<NL; i++) {
	 tau = 0.0f ;
	 for(int j=0; j<NL; j++) {
	    if (j==i) continue ;
	    tau += X[j*NL+i] ;
	 }
	 X[i*NL+i] = -tau ;
      }
      
      // Coefficients for the last equation and the right hand side
      for(int i=0; i<NL; i++)  {
	 X[(NL-1)*NL+i] = 1.0f ;     b[i] = 0.0f ;
      }
      b[NL-1] = 1.0f ;
      
#if 1
      if ((ITER==0)&&(id==0)) {
	 printf("--------------------------------------------------------------------------------\n") ; 
	 for(int j=0; j<NL; j++) {
	    for(int i=0; i<NL; i++) {
	       printf(" %10.3e", X[j*NL+i]) ;
	    }
	    printf("\n") ;
	 }
      }
#endif
# if 0
      for(t=0; t<NT; t++) {
	 u   = UL[2*t] ;     l  =  UL[2*t+1] ;
	 Bul = A[t] / (2.0f*(h0*F[t])*pown(F[t]/c0,2)) ;
	 Blu = Bul * G[u] / G[l] ;	 
	 printf("t %2d  u %d  l %d  A %12.4e  Bul %12.4e  Blu %12.4e  F %12.4e\n",
		t, u, l, A[t], Bul, Blu, F[t]) ;
      }
      printf("--------------------------------------------------------------------------------\n") ;
#endif
      
      // Solve
      u  = Doolittle_LU_Decomposition_with_Pivoting(X, P,  NL) ;
      u *= Doolittle_LU_with_Pivoting_Solve(X, b, P, V, NL) ;
      
      tau = 0.0f ;
      // Test convergence
      for(int i=0; i<NL; i++) {
	 tau = max(tau,  min(fabs(V[i]-n[i])/ATOL , fabs(((V[i]-n[i])/V[i])/RTOL))) ;
	 n[i] = V[i] ;
      }
      if (tau<1.0f) {  // minimum of absolute and relative tolerances ok
	 break ;
      }
   } // for ITER
   
#if 1
   if (id==0) {
      for(int i=0; i<NL; i++) {
	 printf("  n[%d] = %10.3e \n", i, n[i]) ;
      }
      printf("\n") ;
   }
#endif
   
   // results:  TEX, TAU, TRAD
   for(t=0; t<NT; t++) {
      u = UL[2*t] ;  l = UL[2*t+1] ;
      // * optical depth
      Bul = A[t] / (2.0f*h0*pown(F[t]/c0,2)*F[t]) ;
      Blu = Bul * G[u] / G[l] ;
      tau = (h0*colden/(4.0f*PI)) * ( n[l]*Blu - n[u]*Bul )  ;
      // in case of HFS, peak optical depth lower by factor HFSMAX
      if (t==HFSTR) {
	 tau *= HFSMAX ;
      }
      TAU[id*NT+t] = tau ;    //  TAU[NW, NT]
      // * excitation temperature
      tex = - ((E[u]-E[l])/kb)  /  log( (n[u]/n[l]) * (G[l]/G[u]) ) ;
      TEX[id*NT+t] = tex ;
      // * radiation temperature
      trad = (Planck(F[t], tex)-IBG[t])*( (fabs(tau)>0.01f) ? (1.0f-exp(-tau)) : (tau-0.5f*tau*tau));
      TRAD[id*NT+t] = trad * (pown(c0/F[t],2)/(2.0f*kb)) ;

#if 1
      if (id==0) printf(" TEX[%d] = %7.3f\n", t, tex) ;
#endif
   }
}

