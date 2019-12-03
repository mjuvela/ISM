
//  NPIX    = number of pixels
//  R1, R2  = limits of the angular separation
#define PI      3.1415926536f


__kernel void ADF(
                  __global float *X,    // longitude
                  __global float *Y,    // latitude
                  __global float *P,    // position angles [radians]
                  __global float *S     // map of angle dispersion function
                 )  {
   
   int  id  = get_global_id(0) ;  // one output pixel per work item
   if (id>=NPIX) return ;
   float a = P[id], lon1 = X[id],  lat1 = Y[id] ; // centre pixel
   if (a<-99.0f) {  // missing data
      S[id] = 0.0f ; return ;
   }
   float sum=0.0f, b, dlon, nom, den, d, lon2, lat2 ;
   int count=0 ;
   // if (id%100==0) printf("%7d / %7d\n", id, NPIX) ;
   for(int j=0; j<NPIX; j++) {
      if (j<=id) continue ;   //  each pixel pair only once
      b = P[j] ;  lon2 = X[j] ; lat2 = Y[j] ; // the other pixel
      // calculate true angular distance between the points
#if (FAST_DISTANCE>0)
      nom = (lon2-lon1)*cos(lat1) ;
      den = lat2-lat1 ;
      d   = sqrt(nom*nom+den*den) ;
#else
      dlon = lon1-lon2 ;
      nom  = (pow(cos(lat1)*sin(dlon), 2.0f) + 
              pow(cos(lat2)*sin(lat1)-sin(lat2)*cos(lat1)*cos(dlon), 2.0f)) ;
      den  = sin(lat2)*sin(lat1) + cos(lat2)*cos(lat1)*cos(dlon) ;
      d    = atan2(sqrt(nom), den) ;
#endif
      if ((d>R1)&&(d<R2)&&(b>-99.0f)) {  // distance ok, data ok
         d      = min( fabs(a-b), PI-fabs(a-b) ) ;
         sum   += d*d ;
         count += 1 ;
      }
   }
   S[id] =  (count>0) ? sqrt(sum/count) : 0.0f ;   // radians
}

