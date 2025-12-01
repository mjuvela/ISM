#define EPS 1.0e-4f

// #define ATOMICS 1

#if ATOMICS>0
#if (NVIDIA>0)
float atomicAdd_g_f(__global float* p, float val) {
   float prev;
   asm volatile(
		"atom.global.add.f32 %0, [%1], %2;"
		: "=f"(prev)
		: "l"(p) , "f"(val)
		: "memory"
	       );
   return prev;
}
#else
inline void atomicAdd_g_f(volatile __global float *addr, float val) {
   union{
      unsigned int u32;
      float        f32;
   } next, expected, current;
   current.f32    = *addr;
   do {
      expected.f32 = current.f32;
      next.f32     = expected.f32 + val;
      current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                                     expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
}
#endif
#endif


int Index(const float x, const float y, const float z) {
   if ((x<=0.0f)||(y<=0.0f)||(z<=0.0f)||(x>=N)||(y>=N)||(z>=N)) {
      return -1 ;
   }
   return   (int)(floor(x)+N*floor(y)+N*N*floor(z)) ;
}


void Step(float *x, float *y, float *z,  const float u, const float v, const float w, float *ds, int *ind) {
   float s ;
   s      =         (u>0.0f)  ?  ((1.0f+EPS-fmod(*x, 1.0f))/u) : ((-EPS-fmod(*x, 1.0f))/u)  ;
   s      =  min(s, (v>0.0f)  ?  ((1.0f+EPS-fmod(*y, 1.0f))/v) : ((-EPS-fmod(*y, 1.0f))/v)) ;
   *ds    =  min(s, (w>0.0f)  ?  ((1.0f+EPS-fmod(*z, 1.0f))/w) : ((-EPS-fmod(*z, 1.0f))/w)) ;
   *x    +=  (*ds)*u ;
   *y    +=  (*ds)*v ;
   *z    +=  (*ds)*w ;
   *ind   =  Index(*x, *y, *z) ;
}

   
__kernel void test5(__global float *DENS, __global float *TABS)  {
   const int id = get_global_id(0) ;
   float x = 1.0f, y = 1.0f, z=1.0f, u, v, w, tmp, ds ;  
   int ind, oind ;
   for(int iray=0; iray<NRAY ; iray++) {
      u = x ;    v = 0.1f ;    w = 0.5f ;
      tmp = sqrt(u*u+v*v+w*w) ;
      u  /= tmp ;      v /= tmp ;      w /= tmp ;      
      x   = 39.5f  ;    y  = 39.5f   ;   z  = EPS ;
      ind = Index(x, y, z) ;      
      while(ind>=0) {
         oind = ind ;
         Step(&x, &y, &z,  u, v, w,  &ds, &ind) ;
#if ATOMICS
         atomicAdd_g_f(&(TABS[oind]), ds) ;
#else
         TABS[oind] += exp(-ds) ;
#endif
      }
   }
}
