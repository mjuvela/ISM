

void __kernel F2I(__global float *F, __global int *I) {
  const int id = get_global_id(0) ;
  const int gs = get_global_size(0) ;
  float tmp  ;
  for(int i=id; i<N; i+=gs) {
    tmp  =   F[i] ;
    I[i] =  *(int *)(&tmp) ;
  }
}



void __kernel I2F(__global int *I, __global float *F) {
  const int id = get_global_id(0) ;
  const int gs = get_global_size(0) ;
  int tmp ;
  for(int i=id; i<N; i+=gs)  {
    tmp  =   I[i] ;
    F[i] =  *(float *)(&tmp) ;
  }
}

