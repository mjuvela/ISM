
// drafting kernels for RT with full Stokes


// C = pi*a^2 * Q
//   dI  =   -CMAT * IQUV  * n*ds  +  B(T) * CVEC * n*ds

//            C_ext    C_pol     0       0    
// CMAT  =    C_pol    C_ext     0       0    
//            0        0         C_ext   C_crc
//            0        0        -C_crc   C_ext

//                C_abs
// CVEC  =       DC_abs
//               DC_abd
//                0

// Reissl 2014, source terms
//         dI =    C_abs*B(T)             * n_d*s
//         dQ =  +DC_abs*B(T)*cos(2*phi)  * n_d*s
//         dU =  -DC_abs*B(T)*cos(2*phi)  * n_d*s    --- note the sign !! 
//
// phi    =  B projected to Stokes vector system, angle wrt -Q
// theta  =  angle between ray direction and the plane perpendicular to B
//
//  C  = 0.5*(  C_per + C_par + (C_per-C_par)*cos(theta)*cos(theta)
//  DC = 0.5*(C_per-C_par)*sin(theta)*sin(theta)
//
// grain and incident light not in the same frame of reference
// rotate Stokes vector to the system of the grain
//
//            1?    0              0            0
//  ROT =     0     cos(2*phi)    -sin(2*phi)   0
//            0    +sin(2*phi)     cos(2*phi)   0
//            0     0              0            1?


// Above all C terms averages over the size distribution


//   AC_ext  =  (2*C_ext_par+C_ext_per)/3.0

//   C_ext_x =  AC_ext  + (1.0/3.0)*R*(C_ext_par-C_ext_per)
//   C_ext_y =  AC_ext  + (1.0/3.0)*R*(C_ext_par-C_ext_per)*(1.0-3.0*sin(theta)*sin(theta))
//   theta   =  angle between light and magnetic field directions

// Oblate dust grains:
//    C_ext =  0.5*(C_ext_x+C_ext_y)
//    C_pol =  0.5*(C_ext_x-C_ext_y)

// R = <  G(cos(beta)*cos(beta)) * G(cos(zeta)*cos(zeta)) >
//  beta = alignment cone angle between J and B
//  zeta = internal alignment angle I_par and J
//  G(x) = 1.5*x-0.5

// RAT
//  fraction f_hi in the high-J attractor point (not known, free parameter?)
//  R =  0, for a<a_alg
//  R =  f_hi + (1.0-f_hi)  <  G( cos(zeta)*cos(zeta) ) >,   for <>a_alg

// a_alg from ratio = (omega_rad/omega_gas)^2
//
// ratio  =   (a*rho_d/(delta*mH)) * A*A,   delta a shape-dependent parameter
// A      =   INT / (n_gas*kB*T_gas) / (1.0+t_gas/t_rad)
// INT    =   integral{
//                  Q_Gamma(a,lambda,eps) * lambda * gamma * u(lambda) * dlambda
//            }
// eps    =   angle between B and main direction of the radiation
// gamma  =   local anisotropy (current wavelength)



// Reissl 2017
// I  =  (I+Q)*exp(-n_d*s*(Ce+DCe))  + B(T)*(Ca+DCa*cos(2*phi))*n_d*s
// Q  =  (I-Q)*exp(-n_d*s*(Ce-DCe))  - B(T)*(Ca+DCa*cos(2*phi))*n_d*s
// U  =         exp(-n_d*s*Ce)*(U*cos(n_d*s*DCc)-V*sin(n_d*s*DCc))
//           +  n_d*s*DCa*B(T)*sin(2*phi)
// phi = angle between B and "direction of light polarisation"
//
//   Cx = Sum( frac_i  * Integral[ pi*a^2*(Qxpar+Qxper)*n(a)*R(a)*da ]
//  DCx = Sum( frac_i  * Integral[ pi*a^2*(Qxpar-Qxper)*sin(theta)*sin(theta)*n(a)*R(a)*da ]
// theta = angle between ray and B directions


// Reissl 2014
// 
//            Cext   DCext     0        0    
// CMAT  =   DCext    Cext     0        0    
//            0       0        Cext    +DCc  
//            0       0       -DCc      Cext 
// 
// C_crc had different sign compared to Reissl 2016 --- corrected above
// ... signs are also as in Whitney & Wolff (2002)
//
// equations decouple (Whitney & Wolff 2002); new quantities
// on the left, old on the right:
//
//  I+Q  =   (I+Q)*exp(-n*s*(Cext+DCext))          ???? (2017) this is directly I
//            Whitney had Cext-DCext ?? different sign convention?
//            so that "-" would agree with the signs as written in CMAT above ??
//  I-Q  =   (I-Q)*exp(-n*s*(Cext-DCext))          ???? (2017) this is directly Q
//  U    =   [ U*cos(n*s*DCc) - V*sin(n*s*DCc) ] * exp(-n*s*Cext)
//  V    =   [ U*sin(n*s*DCc) + V*cos(n*s*DCc) ] * exp(-n*s*Cext)  --- Reissl (2014) had  "-V*cos..."
//
//  C    =   0.5*[ Cper + Cpar + (Cper-Cpar)* cos(theta)*cos(theta)
// DC    =   0.5*( Cper - Cpar )*sin(theta)*sin(theta)
// theta =   ray direction vs. plane perpendicular to B
//
// Note: above gives
//  I    =   





