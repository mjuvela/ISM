

__kernel void Convolve(const int NSPE, 
                       const int NCHN, 
                       const int DI,        // number of steps in each directions, to add samples
                       const float fwhm,    // beam fwhm in units of the step between NSPE spectra
                       __global float *SPE, __global float *CON) {
   // SPE contains SPE spectra, each NCHN channels, taken equidistantly at offsets [0.0, NSPE-1.0] in our units
   // Convolve spectra with Gaussian beam, fwhm in the same units, and put result to CON
   const int id = get_global_id(0) ;
   if (id>=NSPE) return ;
   // Switch to coordinates where the radial distances in SPE are  [0, NSPE-1.0]
   // This work item is at distance    id/(NSPE-1.0)  in the coordinates [0,1]
   // we loop over[-3*FWHM,+3*FWHM].... which is not ok if fwhm >> model size (extreme beam dilution)!
   float W=0.0f, w, wb, r, d=3.0f*fwhm/DI ;    // spectra available at steps of 1.0, fwhm in the same units
   float K= -4.0f*log(2.0f)*d*d/(fwhm*fwhm) ;  // Gaussian scaling, effectively FWHM = fwhm/d, for distance [d]
   int   a, b, k ;                             //                                                 |  
   for(int c=0; c<NCHN; c++) CON[id*NCHN+c] = 0.0f ;  // clear the target array                   |  
   for(int i=-DI; i<=+DI; i++) {         // stepping over area [-3*FWHM, +3*FWHM]                 |  
      for(int j=-DI; j<=+DI; j++) {      // one unit in (i,j) corresponds to distance d           V  
         r  =  i*i+j*j ;                 // squared distance from the beam centre,         in units of d
         w  =  exp(K*r) ;                // Gaussian weight
         W +=  w ;
         // r = distance from the cloud centre, (i,j) = offset from that tot the actual pointing for one LOS
         r  =  sqrt((float)(j*j*d*d+(i*d+id)*(i*d+id))) ;  // pointing centre (id,0), offsets over the (beam) ~ (i,j)
         // id  = offset in units == step between originally computed spectra = RADIUS/(NSPE-1)
         // i,j = local offsets in units [d] = small step used to sample the beam area
#if 1
         // no interpolation, use the closest matching annulus
         k  =  round(r) ;
         if (k<NSPE) {
            for(int c=0; c<NCHN; c++)     CON[id*NCHN+c]  +=   w*SPE[k*NCHN+c] ;
         }
#else
         // linear interpolation between two annuli
         k = floor(r) ;        // the annulus on the inner side
         if (k<NSPE) {
            if (k<(NSPE-1)) {  // can interpolate  between annuli  k and k+1
               wb  =   r-k ;   // weight [0.0, 1.0] for the outer annulus
               for(int c=0; c<NCHN; c++)  CON[id*NCHN+c]  +=  w*((1.0f-wb)*SPE[k*NCHN+c] + wb*SPE[(k+1)*NCHN+c]) ;
            } else {           // use the single annulus = k
               wb  =  1.0f - (r-k) ;  //  we give weight (r-k) for the offset outside the cloud (values zero)
               for(int c=0; c<NCHN; c++)  CON[id*NCHN+c]  +=  w*SPE[k*NCHN+c] ;
            }
         }
#endif
      }
   }
   if (W>0.0f) {
      for(int c=0; c<NCHN; c++) CON[id*NCHN+c] /= W ;
   }
}
   
   
   
   
