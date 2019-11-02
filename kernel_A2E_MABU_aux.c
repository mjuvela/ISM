

void __kernel  split_absorbed(const    int    IDUST,
                              const    int    N,
                              __global float *RABS,   // [ifreq, idust]
                              __global float *ABU,    // [icell, idust]
                              __global float *IN,     // [N, NFREQ]
                              __global float *OUT     // [N, NFREQ]
                              ) {
   const int icell = get_global_id(0) ;
   if (icell>=N) return ;   
   float den = 0.0 ;
   for(int ifreq=0; ifreq<NFREQ; ifreq++) {
      den = 0.0f ;
      for(int idust=0; idust<NDUST; idust++)   den += ABU[icell*NDUST+idust]*RABS[ifreq*NDUST+idust] ;
      OUT[icell*NFREQ+ifreq] = IN[icell*NFREQ+ifreq] * RABS[ifreq*NDUST+IDUST] / den ;
   }
}
