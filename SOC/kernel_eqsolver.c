


__kernel void EqTemperature(const int       icell,    // first cell of current batchs
                            const float     kE,       // parameters of the E discretisation
                            const float     oplgkE,   //  ...
                            const float     Emin,     //  ...
                            const int       NE,       // number of energy bins
                            __global float *FREQ,     // vector of frequencies
                            __global float *TTT,      // temperatures for discrete energy values
                            __global float *ABS,      // vector of absorbed photons [GLOBAL*NFREQ]
                            __global float *T) {      // resulting temperatures => T[CELLS] !!!
   // Calculate equilibrium temperature based on number of absorbed photons and the E<->T mapping
   // ABS = number of absorbed photons at each frequency, per unit density, in a band of 1 Hz
   // Called for GLOBAL consecutive cells, starting at icell
   int  id = get_global_id(0) ;         // id  = [0, GLOBAL[
   int ind = icell+id, iE ;             // ind = [0, CELLS [
   if (ind>=CELLS) return ;
   // float scale = 6.62607e-27f ;         // cgs absorbed energy (per unit density)
   float scale = 6.62607e-27f ; // this is just Planck constant, no FACTOR here !!
   float wi, beta=1.0f, Ein ;
   __global float *A = &ABS[id*NFREQ] ; // vector of absorbed photons for the current cell
   if (A[0]<-0.5f) {  // If first element is -1.0, this is a link cell and temperature cannot be calculated
      T[ind] = -1.0f ;
   }
   // Trapezoid integration frequency, integrand =  nphotons * h*f  => absorbed energy
   Ein   =  A[0      ]*FREQ[0      ]* scale * (FREQ[1      ]-FREQ[0      ]) ; // first step
   Ein  +=  A[NFREQ-1]*FREQ[NFREQ-1]* scale * (FREQ[NFREQ-1]-FREQ[NFREQ-2]) ; // last step
   //  the sum over the rest of TMP*DF
   for(int i=1; i<(NFREQ-1); i++) {    // bin  [i, i+1]
      Ein  +=  A[i]*FREQ[i] * scale * (FREQ[i+1]-FREQ[i-1]) ;
   }
#if 0
   if (id==0) {
      printf("Ein %12.4e      Egrid %12.4e %12.4e\n", Ein, Emin, Emin*pow(kE, NE-1)) ;
   }
#endif
   // Ein = 1e20 * absorbed energy [cgs] per unit density -- since ABS is per unit density
   // T<->E mapping is also for energy emitted per unit density * 1e20
   iE      =  clamp((int)floor(oplgkE * log10((0.5f*Ein/beta)/Emin)), 0, NE-2) ;
   wi      = (Emin*pown(kE,iE+1)-(Ein/beta)) / (Emin*pown(kE, iE+1)-pown(kE, iE)) ;
   T[ind]  =  wi*TTT[iE] + (1.0-wi)*TTT[iE+1] ;
   if (Ein<=0.0f) T[ind] = 2.7f ;  // must have been a link, not real cell
   if (isfinite(T[ind])) {
      ;
   } else {
      // printf("\n*********************************************************") ;
      printf("[%8d]  Ein %10.3e  Emin %10.3e log10 %10.3e iE=%d NE=%d TTT[iE]=%.3f TTT[iE+1]=%.3f\n", ind, Ein, Emin, log10(Ein/Emin), iE, NE, TTT[iE], TTT[iE+1]) ;
      // printf("*********************************************************\n") ;
      T[ind] = 2.7f ;
   }
}



__kernel void Emission(const float     FREQ,
                       const float     KABS,
                       __global float *T,
                       __global float *EMIT) {
   // Calculate emission based on temperature (non-equilibrium grains)
   int id = get_global_id(0) ;
   if (id>=CELLS) return ;
   // 1.0e20*4.0*PI/(PLANCK*FFREQ[a:b])) * FABS[a:b]*Planck(FFREQ[a:b], TNEW[icell])/(USER.GL*PARSEC)
   // 1e20 * 4 *pi / (h*f) = 1.8965044e+47
   // float res = 2.79639459f * KABS * (FREQ*FREQ/(exp(4.7995074e-11f*FREQ/T[id])-1.0f)) ;
   float res = (2.79639459e-20f*FACTOR) * KABS * (FREQ*FREQ/(exp(4.7995074e-11f*FREQ/T[id])-1.0f)) ;
   EMIT[id] =  isfinite(res) ? res : 0.0f ;
}


