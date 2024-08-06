


void __kernel Index(const int n, __global float *C) {
   const int id = get_global_id(0) ;
   const int gs = get_global_size(0) ;
   int   ind ;
   for(int i=id; i<n; i+=gs) {
      ind  =  8*i ;
      C[i] = -(*(float*)&ind) ;
   }
}
