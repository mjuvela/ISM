
#define BOLTZMANN (1.38065e-16f)
#define PLANCK    (6.62607e-27f)
#define C_LIGHT   (2.99792e+10f)
// #define SCALE     (1.0e20f)  --- replaced by FACTOR 


float Interpolate(const int n, const __global float *x, const __global float *y, const float x0) {
   // Return linear-interpolated value based on (x,y) vectors.
   if (x0<=x[0  ])  return y[0] ;    // no extrapolation !
   if (x0>=x[n-1])  return y[n-1] ;
   int a=0, c=n-1, b ;
   while((c-a)>4) {
      b = (a+c)/2 ;
      if (x[b]>x0) c = b ; else a = b ;
   }
   for(b=a; b<=c; b++) {     // used b=a+1 ... leads to seg.fault (if a==b ???)
      if (x[b]>=x0) break ;  // until x[b] first above x0
   }
   float  w = (x[b]-x0) / (x[b]-x[b-1]) ;
   return w*y[b-1] + (1.0f-w)*y[b] ;
}



__kernel void PrepareTdown(const    int     NFREQ,       // number of frequencies in the grid
                           __global float  *FREQ,        // FREQ[NFREQ], basic frequency grid
                           __global float  *Ef,          // Ef[NFREQ]
                           __global float  *SKABS,       // SKABS[NFREQ], corresponding pi*a^2*Q  (current size)
                           const    int     NE,          // number of energy bins
                           __global float  *E,           // E[NEPO], energy grid for the current size
                           __global float  *T,           // T[NEPO], corresponding grain temperatures
                           __global float  *Tdown        // Tdown[NE] current size
                          )
{
   //  Prepare vector TDOWN = transition matrix elements for transitions u -> u-1, D&L01, Eq. 41
   //  Needs SKabs_Int(f) =  SKabs(SIZE[size], freq) * GRAIN_DENSITY * S_FRAC[size]
   //                     =  [ PI*a^2 * Q ]  *  [ GD*S_FRAC ],  with interpolated Q
   //  We replaced SKabs_Int() with SKabs() so final division with GD*S_FRAC is here removed!
   //  One work item per upper level u... probably efficient only on CPU
   const int u = 1 + get_global_id(0) ;  //  u = 1,..., NE-1
   if (u>=NE) return ;   
   if (u==1) Tdown[0]  = 0.0 ;
   double Tu, I, ee0, ee1, yy0, yy1, Eu, El, x ;
   int i ;         
   Eu   =  0.5*(E[u  ]+E[u+1]) ;          // at most  u+1 = NE-1+1 = NE < NEPO
   El   =  0.5*(E[u-1]+E[u  ]) ;          // at least u-1 = 1-1 = 0
   Tu   =  Interpolate(NE+1, E, T, Eu) ;  // would be better if interpolated on log-log scale ?
   ee0  =  0.0 ;
   yy0  =  0.0 ;         
   i    =  0 ;            
   I    =  0.0 ;
   while (Ef[i+1]<Eu && (i<(NFREQ-1))) {
      ee0  =  Ef[i] ;
      // In A2ELIB, SKabs_Int() uses Interpolate0 for frequency interp... which is linear interpolation
      x    =  Interpolate(NFREQ, FREQ, SKABS, ee0/PLANCK) ; // this apparently fine on linear scale...
      yy0  =  ee0*ee0*ee0* x /(exp(ee0/(BOLTZMANN*Tu))-1.0) ;
      for(int j=0; j<8; j++) {
         ee1  =  Ef[i] + (j+1)*(Ef[i+1]-Ef[i])/8.0 ;
         x    =  Interpolate(NFREQ, FREQ, SKABS, ee1/PLANCK) ; // SKabs_Int(size, ee1/Planck)
         yy1  =  ee1*ee1*ee1 * x / (exp(ee1/(BOLTZMANN*Tu))-1.0) ;
         I   +=  0.5*(ee1-ee0)*(yy1+yy0) ;
         ee0  =  ee1 ;
         yy0  =  yy1 ;
      }
      i++ ;
      // if (i>(NFREQ-2)) break ;  // [NFREQ-2] +1 = NFREQ-1 on next loop ... still ok
   }
   if (Eu<Ef[NFREQ-1]) {
      for (int j=0; j<8; j++) {
         ee1  =  Ef[i]+(j+1)*(Eu-Ef[i])/8.0  ;
         // yy1  =  ee1*ee1*ee1*SKABS(size,ee1/PLANCK)/(exp(ee1/(BOLZMANN*Tu))-1.0) ;
         x    =  Interpolate(NFREQ, FREQ, SKABS, ee1/PLANCK) ;  // SKabs_Int(size, ee1/PLANCK)
         yy1  =  ee1*ee1*ee1 * x / (exp(ee1/(BOLTZMANN*Tu))-1.0) ;
         I   +=  0.5*(ee1-ee0)*(yy1+yy0) ;
         ee0  =  ee1 ;
         yy0  =  yy1 ;         
      }
   } 
   // I *= 8.0*PI/((Eu-El)*C_LIGHT*C_LIGHT*PLANCK*PLANCK*PLANCK) ;
   // Warning:   8.0*PI/(C_LIGHT*C_LIGHT*PLANCK*PLANCK*PLANCK)  = 9.612370e+58
   I   *=  9.612370e+58 / (Eu-El) ;
   // printf("%12.4e\n", I) ;
   Tdown[u] = I ;
}





__kernel void PrepareIntegrationWeights(const int NFREQ,                                        
                                        const int NE,
                                        __global float *Ef,         // Ef[NFREQ], FREQ*PLANCK
                                        __global float *E,          // E[NEPO], energy grid for the current size
                                        __global int   *L1,         // L1[NE*NE]
                                        __global int   *L2,         // L2[NE*NE]
                                        __global float *IW ,        // Iw[ << 0.5*NE*NE*NFREQ]
                                        __global float *wrk,        // NE*NFREQ + NE*(NFREQ+4)
                                        __global int   *noIw        // noIw for each l = lower bin
                                       ) 
{
   // Precalculate integration weights for upwards transitions. Eq. 15, (16) in Draine & Li (2001)
   // With the weight precalculated, the (trapezoidal numerical) integral of eq. 15 is a dot product of the vectors
   // containing the weights and the absorbed photons (=Cabs*u), which can be calculated with (fused) multiply-add.
   // TODO: fix the treatment of the highest bin, although it probably won't make any difference...
   // Single size => Iw is filled from the beginning, not all elements are needed, only weights_for_size first ones.
   // Each grain size is a separate kernel call.
   // ****************************************************************************************************
   // ***WARNING*** this has problems with larger grains, because of something weird happening with the
   //               absorptions in the two highest-frequency bins
   const int l = get_global_id(0) ; // Each work item works on different "l", initial enthalpy bin.
   // ****************************************************************************************************
   if (l>=(NE-1)) return ;
   __global float *temp_Iw  =  &(wrk[ l*NFREQ]) ;               // NFREQ
   __global float *freq_e2  =  &(wrk[NE*NFREQ+l*(NFREQ+4)]) ;   // NFREQ+4
   __global float *Iw       =  &(IW[l*NE*NFREQ]) ;              // at most NE*NFREQ cooeffs per lower bin l
   int index = 0 ;  
   double Eu, El, dEu, dEl, Gul, Gul2, I, W[4] ;
   int i, j, k;
   
   
#if 0
   if (l==0) {
      for(int i=0; i<NFREQ; i++) printf("Ef[%3d] = %12.4e\n", i, Ef[i]) ;
      for(int i=0; i<NE; i++)    printf("E[%3d]  = %12.4e\n", i, E[i]) ;
   }
#endif
   
   
   dEl =  E[l+1]-E[l] ;
   El  =  0.5*(E[l]+E[l+1]) ;
   
   for(int  u=l+1; u<NE; u++) {      
      Eu    =  0.5*(E[u]+E[u+1]) ;
      dEu   =  E[u+1]-E[u] ;
      W[0]  =  E[u]-E[l+1] ;
      W[1]  =  min( E[u]-E[l],  E[u+1]-E[l+1] ) ;
      W[2]  =  max( E[u]-E[l],  E[u+1]-E[l+1] ) ;
      W[3]  =  E[u+1] - E[l] ;         
      
#if 0
      if (l==0) {
         printf("u=%3d  W %12.4e %12.4e %12.4e %12.4e\n", u+1, W[0], W[1], W[2], W[3]) ;
      }
#endif
      
      
      if ((Ef[0]>W[3]) || (Ef[NFREQ-1]<W[0])) {
         L1[l*NE+u] = -1 ;
         L2[l*NE+u] = -2 ;
         continue ;
      }         
      i=0;  j=0;  k=0;
      while (i<NFREQ+4) {  // after this freq_e2 contains all the breakpoints of the integrand: NFREQ from freq grids + 4 from bin edges
         if (j<NFREQ && k<4) {
            freq_e2[i] = min(Ef[j], (float)W[k]) ;
            if (Ef[j]<W[k]) j++ ;
            else k++ ;
         } else {
            if (j>=NFREQ && k<4) freq_e2[i]=W[k];
            else if (k>=4 && j<NFREQ) freq_e2[i]=Ef[j] ;
            else {
               ;
               // assert(1==0) ;
            }
         }
         i++ ;
      }         
      for (i=0; i<NFREQ; i++) temp_Iw[i] = 0.0f ;      
      // done: integrate Gul*nabs*E over [freq_e2[i-1], freq_e2[i]]
      // Gul, NABS and E are all linear functions on the interval,
      // so the integrand is (at most) third degree polynomial
      for (i=1; i<NFREQ+4; i++) {         
         if (freq_e2[i-1]>Ef[NFREQ-1] || freq_e2[i-1]>=W[3]) break ;	// Ef[i] = photon energy at freq[i]
         if (freq_e2[i]<W[0] || freq_e2[i-1]<=Ef[0]) continue ;            
         j=0;
         while (Ef[j]<freq_e2[i-1] && j<NFREQ-2) j++ ;            
         // now Ef[j-1] < freq_e2[i-1] <= Ef[j],   freq_e2[i]<=Ef[j+1]
                        
         // 1
         if (W[1]<=freq_e2[i-1] && freq_e2[i]<=W[2]) {	 // on the flat part of Gul
            Gul = min(dEu, dEl)/(dEu*dEl) ;
            
            if (freq_e2[i]<=Ef[j]) { // between Ef[j-1] and Ef[j]
               //                  (                                                                                                                                        )
               temp_Iw[j  ] += Gul*(   freq_e2[i]*freq_e2[i]*(  1.0/3.0*freq_e2[i]-1.0/2.0*Ef[j-1] )  -  freq_e2[i-1]*freq_e2[i-1]*( 1.0/3.0*freq_e2[i-1]-1.0/2.0*Ef[j-1])   )   /  (Ef[j]-Ef[j-1]+1e-120) ;
               temp_Iw[j-1] += Gul*(   freq_e2[i]*freq_e2[i]*( -1.0/3.0*freq_e2[i]+1.0/2.0*Ef[j  ] )  -  freq_e2[i-1]*freq_e2[i-1]*(-1.0/3.0*freq_e2[i-1]+1.0/2.0*Ef[j  ])   )   /  (Ef[j]-Ef[j-1]+1e-120) ;
            } else {		        // between Ef[j] and Ef[j+1]
               temp_Iw[j+1] += Gul*(freq_e2[i]*freq_e2[i]*(1.0/3.0*freq_e2[i]-1.0/2.0*Ef[j]) - freq_e2[i-1]*freq_e2[i-1]*(1.0/3.0*freq_e2[i-1]-1.0/2.0*Ef[j])) / (Ef[j+1]-Ef[j]+1e-120) ;
               temp_Iw[j] += Gul*(freq_e2[i]*freq_e2[i]*(-1.0/3.0*freq_e2[i]+1.0/2.0*Ef[j+1]) -
                                  freq_e2[i-1]*freq_e2[i-1]*(-1.0/3.0*freq_e2[i-1]+1.0/2.0*Ef[j+1])) / (Ef[j+1]-Ef[j]+1e-120) ;
            }
         }
         
         // 2
         if (W[0]<=freq_e2[i-1] && freq_e2[i]<=W[1]) {		// on the upslope
            // Gul = (freq_e2[i-1]-W[0])/(dEu*dEl) ;    Gul2 = (freq_e2[i]-W[0])/(dEu*dEl) ;
            if (freq_e2[i]<=Ef[j]) { // between Ef[j-1] and Ef[j]
               temp_Iw[j] += (  freq_e2[i]*freq_e2[i]    *(1.0/4.0*freq_e2[i]*freq_e2[i]     - 1.0/3.0*(Ef[j-1]+W[0])*freq_e2[i] + 0.5*W[0]*Ef[j-1]) -
                                freq_e2[i-1]*freq_e2[i-1]*(1.0/4.0*freq_e2[i-1]*freq_e2[i-1] - 1.0/3.0*(Ef[j-1]+W[0])*freq_e2[i-1] + 0.5*W[0]*Ef[j-1]))     /  (dEl*dEu*(Ef[j]-Ef[j-1]+1e-120)) ;                  
               temp_Iw[j-1] += (freq_e2[i]*freq_e2[i]*(-1.0/4.0*freq_e2[i]*freq_e2[i] +
                                                       1.0/3.0*(Ef[j]+W[0])*freq_e2[i] - 0.5*W[0]*Ef[j]) -
                                freq_e2[i-1]*freq_e2[i-1]*(-1.0/4.0*freq_e2[i-1]*freq_e2[i-1] +
                                                           1.0/3.0*(Ef[j]+W[0])*freq_e2[i-1] - 0.5*W[0]*Ef[j])) / (dEl*dEu*(Ef[j]-Ef[j-1]+1e-120)) ;
            } else {		  // between Ef[j] and Ef[j+1]
               temp_Iw[j+1] += (freq_e2[i]*freq_e2[i]*(1.0/4.0*freq_e2[i]*freq_e2[i] -
                                                       1.0/3.0*(Ef[j]+W[0])*freq_e2[i] + 1.0/2.0*W[0]*Ef[j]) -
                                freq_e2[i-1]*freq_e2[i-1]*(1.0/4.0*freq_e2[i-1]*freq_e2[i-1] -
                                                           1.0/3.0*(Ef[j]+W[0])*freq_e2[i-1] + 1.0/2.0*W[0]*Ef[j])) / (dEl*dEu*(Ef[j+1]-Ef[j]+1e-120)) ;
               temp_Iw[j] += (freq_e2[i]*freq_e2[i]*(-1.0/4.0*freq_e2[i]*freq_e2[i] +
                                                     1.0/3.0*(Ef[j+1]+W[0])*freq_e2[i] - 1.0/2.0*W[0]*Ef[j+1]) -
                              freq_e2[i-1]*freq_e2[i-1]*(-1.0/4.0*freq_e2[i-1]*freq_e2[i-1] +
                                                         1.0/3.0*(Ef[j+1]+W[0])*freq_e2[i-1] - 1.0/2.0*W[0]*Ef[j+1])) / (dEl*dEu*(Ef[j+1]-Ef[j]+1e-120)) ;
            }
         }
         
         // 3
         if (W[2]<=freq_e2[i-1] && freq_e2[i]<=W[3]) {	// on the downslope
            // Gul = (W[3]-freq_e2[i-1])/(dEu*dEl) ;   Gul2 = (W[3]-freq_e2[i])/(dEu*dEl) ;
            if (freq_e2[i]<=Ef[j]) { // between Ef[j-1] and Ef[j]
               temp_Iw[j] += (freq_e2[i]*freq_e2[i]*(-1.0/4.0*freq_e2[i]*freq_e2[i] +
                                                     1.0/3.0*(Ef[j-1]+W[3])*freq_e2[i] - 1.0/2.0*W[3]*Ef[j-1]) -
                              freq_e2[i-1]*freq_e2[i-1]*(-1.0/4.0*freq_e2[i-1]*freq_e2[i-1] +
                                                         1.0/3.0*(Ef[j-1]+W[3])*freq_e2[i-1] - 1.0/2.0*W[3]*Ef[j-1])) / (dEl*dEu*(Ef[j]-Ef[j-1]+1e-120)) ;
               temp_Iw[j-1] += (freq_e2[i]*freq_e2[i]*(1.0/4.0*freq_e2[i]*freq_e2[i] -
                                                       1.0/3.0*(Ef[j]+W[3])*freq_e2[i] + 1.0/2.0*W[3]*Ef[j]) -
                                freq_e2[i-1]*freq_e2[i-1]*(1.0/4.0*freq_e2[i-1]*freq_e2[i-1] -
                                                           1.0/3.0*(Ef[j]+W[3])*freq_e2[i-1] + 1.0/2.0*W[3]*Ef[j])) / (dEl*dEu*(Ef[j]-Ef[j-1]+1e-120)) ;
            } else {		  // between Ef[j] and Ef[j+1]
               temp_Iw[j+1] += (freq_e2[i]*freq_e2[i]*(-1.0/4.0*freq_e2[i]*freq_e2[i] +
                                                       1.0/3.0*(Ef[j]+W[3])*freq_e2[i] - 1.0/2.0*W[3]*Ef[j]) -
                                freq_e2[i-1]*freq_e2[i-1]*(-1.0/4.0*freq_e2[i-1]*freq_e2[i-1] +
                                                           1.0/3.0*(Ef[j]+W[3])*freq_e2[i-1] - 1.0/2.0*W[3]*Ef[j])) / (dEl*dEu*(Ef[j+1]-Ef[j]+1e-120)) ;
               temp_Iw[j] += (freq_e2[i]*freq_e2[i]*(1.0/4.0*freq_e2[i]*freq_e2[i] -
                                                     1.0/3.0*(Ef[j+1]+W[3])*freq_e2[i] + 1.0/2.0*W[3]*Ef[j+1]) -
                              freq_e2[i-1]*freq_e2[i-1]*(1.0/4.0*freq_e2[i-1]*freq_e2[i-1] -
                                                         1.0/3.0*(Ef[j+1]+W[3])*freq_e2[i-1] + 1.0/2.0*W[3]*Ef[j+1])) / (dEl*dEu*(Ef[j+1]-Ef[j]+1e-120)) ;
            }
         }
         
      } // for frequency
      
      // intrabin absorptions: second term of Eq. 28
      // assume NABS piecewise linear and integrate the product of 2 or 3 linear functions...
      
#if 1
      if (u==l+1) {
         j = 1 ;            
         while (Ef[j]<dEl && j<NFREQ) {
            temp_Iw[j] += (Ef[j]*Ef[j]*(1.0/3.0*Ef[j]-1.0/2.0*Ef[j-1]-1.0/4.0*Ef[j]*Ef[j]/dEl+1.0/3.0*Ef[j-1]*Ef[j]/dEl) -
                           Ef[j-1]*Ef[j-1]*(1.0/3.0*Ef[j-1]-1.0/2.0*Ef[j-1]-1.0/4.0*Ef[j-1]*Ef[j-1]/dEl+1.0/3.0*Ef[j-1]*Ef[j-1]/dEl)) /
              (dEu*(Ef[j]-Ef[j-1]+1e-120)) ;
            
            temp_Iw[j-1] += (Ef[j]*Ef[j]*(-1.0/3.0*Ef[j]+1.0/2.0*Ef[j]+1.0/4.0*Ef[j]*Ef[j]/dEl-1.0/3.0*Ef[j]*Ef[j]/dEl) -
                             Ef[j-1]*Ef[j-1]*(-1.0/3.0*Ef[j-1]+1.0/2.0*Ef[j]+1.0/4.0*Ef[j-1]*Ef[j-1]/dEl-1.0/3.0*Ef[j]*Ef[j-1]/dEl)) /
              (dEu*(Ef[j]-Ef[j-1]+1e-120)) ;               
            j++ ;
         }            
         if (j<NFREQ) {
            temp_Iw[j] += (dEl*dEl*(1.0/3.0*dEl-1.0/2.0*Ef[j-1]-1.0/4.0*dEl+1.0/3.0*Ef[j-1]) -
                           Ef[j-1]*Ef[j-1]*(1.0/3.0*Ef[j-1]-1.0/2.0*Ef[j-1]-1.0/4.0*Ef[j-1]*Ef[j-1]/dEl+1.0/3.0*Ef[j-1]*Ef[j-1]/dEl))/
              (dEu*(Ef[j]-Ef[j-1]+1e-120)) ;
            
            temp_Iw[j-1] += (dEl*dEl*(1.0/2.0*Ef[j]-1.0/3.0*dEl-1.0/3.0*Ef[j]+1.0/4.0*dEl) -
                             Ef[j-1]*Ef[j-1]*(1.0/2.0*Ef[j]-1.0/3.0*Ef[j]-1.0/3.0*Ef[j]*Ef[j]/dEl+1.0/4.0*Ef[j]*Ef[j]/dEl))/
              (dEu*(Ef[j]-Ef[j-1]+1e-120)) ;               
         }                        
      }
#endif
      
      int first_non_zero = -1, last_non_zero=-2 ;
      for (i=0; i<NFREQ; i++) {
         if (temp_Iw[i]>0.0 && first_non_zero<0) first_non_zero = i  ;
         if (temp_Iw[i]>0.0) last_non_zero = i ;
      }
      L1[l*NE+u] = first_non_zero ;
      L2[l*NE+u] = last_non_zero ;
      for (i=L1[l*NE+u]; i<=L2[l*NE+u]; i++) {
         if (i<NFREQ)  Iw[index] = temp_Iw[i] * dEu/((Eu-El)* (FACTOR*PLANCK)) ;
         else          Iw[index] = 0.0 ;            
         index++ ;
      }
      // printf("L1 %7d,  L2 %7d\n", first_non_zero, last_non_zero) ;
      
   } // for u
   
   noIw[l] = index ;   
}





__kernel void PrepareIntegrationWeightsGD(const int NFREQ,                                        
                                          const int NE,
                                          __global float *Ef,         // Ef[NFREQ], FREQ*PLANCK
                                          __global float *E,          // E[NEPO], energy grid for the current size
                                          __global int   *L1,         // L1[NE*NE]
                                          __global int   *L2,         // L2[NE*NE]
                                          __global float *IW ,        // Iw[ << 0.5*NE*NE*NFREQ]
                                          __global float *wrk,        // NE*NFREQ + NE*(NFREQ+4)
                                          __global int   *noIw        // noIw for each l = lower bin
                                         ) 
{
   // Integration weights using the Guhathagurta & Draine formulas
   const int l = get_global_id(0) ; // Each work item works on different "l", initial enthalpy bin.
   if (l>=(NE-1)) return ;
   int index = 0 ;  
   double DE, dEu, Eu, El, I, wi ;
   int i, j, k;
   __global float *Iw       =  &(IW[l*NE*NFREQ]) ;       // at most NE*NFREQ coeffs per lower bin l
   
   double  coeff =  1.0/(FACTOR*PLANCK) ;
   
   for(int  u=l+1; u<NE; u++) {
#if 0
      DE    =  E[u]   - E[l] ;
#else
      DE    =  0.5*((E[u]+E[u+1])-(E[l]+E[l+1])) ;
#endif
      dEu   =  E[u+1] - E[u] ;
      if ((DE<=Ef[0]) || (DE>=Ef[NFREQ-1])) {
         L1[l*NE+u] = -1   ;
         L2[l*NE+u] = -999 ;
         continue ;
      }        
      i = 1 ;
      while(Ef[i]<DE) {
         i += 1 ;
      }       
      if (0) { // take the closest bin only
         if (fabs(Ef[i-1]-DE)<fabs(Ef[i]-DE)) {  
            i -= 1 ;
         }
         L1[l*NE+u]   =   i   ;
         L2[l*NE+u]   =   i   ;
         Iw[index]    =   dEu * coeff ;   
         index       +=   1 ;
      } else {  // weighted sum of two bins
         i -= 1 ;
         // now we have   Ef[i] < DE < Ef[i+1]
         wi           =   (DE-Ef[i]) / (Ef[i+1]-Ef[i]) ; // weight for [i+1]
         L1[l*NE+u]   =   i   ;
         L2[l*NE+u]   =   i+1 ;
         Iw[index]    =   (1.0-wi)*dEu*coeff ; 
         Iw[index+1]  =   (    wi)*dEu*coeff ;
         index += 2 ;
      }
   } // for u
   noIw[l] = index ;
}





__kernel void PrepareIntegrationWeightsEuler(const int NFREQ,                                        
                                             const int NE,
                                             __global float *Ef,         // Ef[NFREQ], FREQ*PLANCK
                                             __global float *E,          // E[NEPO], energy grid for the current size
                                             __global int   *L1,         // L1[NE*NE]
                                             __global int   *L2,         // L2[NE*NE]
                                             __global float *IW ,        // Iw[ << 0.5*NE*NE*NFREQ]
                                             __global float *wrk,        // NE*NFREQ + NE*(NFREQ+4)
                                             __global int   *noIw        // noIw for each l = lower bin
                                            ) 
{
   // Integration weights using the Guhathagurta & Draine formulas
   const int l = get_global_id(0) ; // Each work item works on different "l", initial enthalpy bin.
   if (l>=(NE-1)) return ;
   int index = 0 ;  
   double El, Eu, dEl, dEu, I, wi, coeff, alpha, beta, G1, G2 ;
   float W1, W2, W3, W4, a, b ;
   int i, j ;
   __global float *temp_Iw  =  &(wrk[ l*NFREQ]) ;        // NFREQ
   __global float *Iw       =  &(IW[l*NE*NFREQ]) ;       // at most NE*NFREQ coeffs per lower bin l
      

   El    =  0.5*(E[l]+E[l+1]) ;
   dEl   =  E[l+1] - E[l] ;
   
   for(int  u=l+1; u<NE; u++) {
      Eu    =  0.5*(E[u]+E[u+1]) ;  // E[NEPO] ... maximum index NE (ok, array has NEPO elements)
      dEu   =  E[u+1] - E[u] ;
      W1    =  E[u]   - E[l+1] ;
      W2    =  min( E[u]-E[l],  E[u+1]-E[l+1] ) ;
      W3    =  max( E[u]-E[l],  E[u+1]-E[l+1] ) ;
      W4    =  E[u+1] - E[l] ;
      
      if ((Ef[0]>W4) || (Ef[NFREQ-1]<W1)) { // integration interval deos not overlap with simulated frequencies
         L1[l*NE+u] = -1   ;
         L2[l*NE+u] = -2 ;
         continue ;
      }  
      

      for (i=0; i<NFREQ; i++) temp_Iw[i] = 0.0f ;
      coeff = 1.0 / (Eu-El) / (FACTOR*PLANCK) ;
      
      // find bin index i such that Ef[i] < W1 < Ef[i+1]
      i = 1 ;
      while ((i<(NFREQ-1)) && (Ef[i]<W1)) {
         i += 1 ;
      }
      i = max(i-1, 0) ;
      
      
      // W1-W2 ==========================================================================================
      a      =  clamp(W1, Ef[i], Ef[i+1]) ;
      b      =  clamp(W2, a,     Ef[i+1]) ;
      alpha  =  (a-Ef[i])/(Ef[i+1]-Ef[i]) ;
      beta   =  (b-Ef[i])/(Ef[i+1]-Ef[i]) ;
      // if we evaluate G and E at the end points of the interval and derive weights only for the
      // interpolation of nabs[i] values
      G1             =  (a-W1)/dEl ;  // directly at the end points of the interval (if and when a, b != E[i])
      G2             =  (b-W1)/dEl ;
      temp_Iw[i]    +=  0.5*(b-a)*(G1*a*(1.0-alpha) + G2*b*(1.0-beta)) * coeff ;
      temp_Iw[i+1]  +=  0.5*(b-a)*(G1*a*alpha       + G2*b*beta      ) * coeff ;
      if (b<W2) {  // if step was till the end of a bin, we continue with the next bin
         i += 1 ;
      }
      while ((i<(NFREQ-1)) && (b<W2)) {
         a              =  b ;
         G1             =  G2 ;
         b              =  min(W2, Ef[i+1]) ;
         alpha          =  (a-Ef[i])/(Ef[i+1]-Ef[i]) ;
         beta           =  (b-Ef[i])/(Ef[i+1]-Ef[i]) ;
         G2             =  (b-W1)/dEl ;
         temp_Iw[i]    +=  0.5*(b-a)*(G1*a*(1.0-alpha) + G2*b*(1.0-beta))  * coeff ;
         temp_Iw[i+1]  +=  0.5*(b-a)*(G1*a*alpha       + G2*b*beta      )  * coeff ;
         if (b<W2)  {  // continues till the next bin
            i += 1 ;
         }
      }

      
      // W2-W3 ==========================================================================================
      while ((i<(NFREQ-1)) && (b<W3)) {
         a             =  b ;
         G1            =  G2 ;
         b             =  clamp(W3, a, Ef[i+1]) ;
         G2            =  min(dEl, dEu) / dEl ;
         alpha         =  (a-Ef[i])/(Ef[i+1]-Ef[i]) ;
         beta          =  (b-Ef[i])/(Ef[i+1]-Ef[i]) ;
         temp_Iw[i]   +=  0.5*(b-a)*(G1*a*(1.0-alpha) + G2*b*(1.0-beta)) * coeff ;
         temp_Iw[i+1] +=  0.5*(b-a)*(G1*a*alpha       + G2*b*beta      ) * coeff ;
         if (b<W3) { //  integration continues till the next bin
            i += 1 ;
         }
      }
      
      
      // W3-W4 ==========================================================================================
      while ((i<(NFREQ-1)) && (b<W4)) {
         a               =  b ;
         G1              =  G2 ;
         b               =  clamp(W4, a, Ef[i+1]) ;
         alpha           =  (a-Ef[i])/(Ef[i+1]-Ef[i]) ;
         beta            =  (b-Ef[i])/(Ef[i+1]-Ef[i]) ;
         G2              =  (W4-b)/dEl ;
         temp_Iw[i]     +=  0.5*(b-a)*(G1*a*(1.0-alpha) + G2*b*(1.0-beta)) * coeff ;
         temp_Iw[i+1]   +=  0.5*(b-a)*(G1*a*alpha       + G2*b*beta      ) * coeff ;
         if (b<W4)  {  //  continue till the next bin
            i += 1 ;
         }
      }
      
      
      // Intrabin ==========================================================================================
      if (1) {
         // intrabin absorptions  --- not significant ??
         // Integral [0, dEl] of   (c/(eu-El)) (1-E/dEl)  Ef    dE
         if (u==(l+1)) {
            i     =  0 ;
            b     =  Ef[0] ;
            while ((i<(NFREQ-1)) && (Ef[i]<dEl)) {
               a             =  b ;
               b             =  clamp((float)dEl, a, Ef[i+1]) ;
               alpha         =  (a-Ef[i])/(Ef[i+1]-Ef[i]) ;
               beta          =  (b-Ef[i])/(Ef[i+1]-Ef[i]) ;
               temp_Iw[i]   +=  0.5*(b-a)*((1.0-a/dEl)*a*(1.0-alpha) + (1.0-b/dEl)*b*(1.0-beta))  * coeff ;
               temp_Iw[i+1] +=  0.5*(b-a)*((1.0-a/dEl)*a*alpha       + (1.0-b/dEl)*b*beta      )  * coeff ;
               i            +=  1 ;
            }
         } // if l->l+1
      } // if - including intrabin

      
      int first_non_zero = -1, last_non_zero=-2 ;
      for (i=0; i<NFREQ; i++) {
         if (temp_Iw[i]>0.0 && first_non_zero<0) first_non_zero = i  ;
         if (temp_Iw[i]>0.0) last_non_zero = i ;
      }
      L1[l*NE+u] = first_non_zero ;
      L2[l*NE+u] = last_non_zero ;
      for (i=L1[l*NE+u]; i<=L2[l*NE+u]; i++) {
         if (i<NFREQ)  Iw[index] = temp_Iw[i] ;  // * dEu/((Eu-El)* (FACTOR*PLANCK)) ;
         else          Iw[index] = 0.0 ;            
         index++ ;
      }
      
   } // for u
   noIw[l] = index ;
   
}

