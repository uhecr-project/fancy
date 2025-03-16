/**
 * Functions used with Stan Fits in joint model.
 *
 * @author Keito Watanabe
 * @date March 2024
 */

/**
 * Arrival rigidity spectrum for a single source, interpolated from look-up tables.
 * .
 * @param R : rigidity in RV
 * @param alpha : source spectral index
 * @param log10_arr_spectrum : log_e(arrival rigidity spectrum(alpha, R))
 * @param alpha_grid : grid of alphas for interpolation grid
 * @param log10_Rgrid : grid of log10(rigidities) for interpolation grid
 */
real arrival_spectrum_lpdf(real R, real alpha, array[ , ] real log_arr_spectrum, array [] real alpha_grid, array [] real log10_Rgrid) 
 {
  real log10_R = log10(R);
  /* 2-D interpolatoin of arrival spectrum  */
  return interp2d(log10_R, alpha, log10_Rgrid, alpha_grid, log_arr_spectrum);
}

/**
 * Arrival rigidity spectrum for isotropic background sources.
 * .
 * @param R : rigidity in RV
 * @param alpha : background spectral index
 * @param Rmin : minimum rigidity in sample, must match min(R_grid)
 * @param Rmax : maximum rigidity in sample, must match max(R_grid)
 */
real background_spectrum_lpdf(real R, real alpha, real Rmin, real Rmax)
{
  real N;
  real p;
  if(alpha != 1.0)
  {
    N = ((1.0-alpha)/((Rmax^(1.0-alpha))-(Rmin^(1.0-alpha))));
  }
  else
  {
    N = (1.0/(log(Rmax)-log(Rmin)));
  }

  p = (N * pow(R, (-alpha)));
  return log(p);
}

/**
 * Calculate the deflection parameter of the vMF distribution.
 * Derived from Eq. 4 and Eq. 9 in Cape; & Mortlock, 2018.
 * @param R rigidity in EV
 * @param B rms magnetic field strength in nG
 * @param D distance in Mpc / 10
 */
real get_kappa(real R, real B, real D) {
  
  return 7552.0 * inv_square(2.3 * inv(R / 50.0) * B * sqrt(D));
}

/**
 * Calculate the spatial deflection angular scale of the vMF distribution.
 * @param R rigidity in EV
 * @param B rms magnetic field strength in nG
 * @param D distance in Mpc / 10
 */
real get_theta(real R, real B, real D) {
  
  return 2.3 * inv(R / 50.0) * B * sqrt(D);
}

/**
 * Define the fik PDF.
 * NB: Cannot be vectorised.
 * Uses sinh(kappa) ~ exp(kappa)/2 
 * approximation for kappa > 100.
 */
real fik_lpdf(vector v, vector mu, real kappa, real kappa_d) {
  
  real lprob;
  real inner = abs_val((kappa_d * v) + (kappa * mu));
  
  if (kappa > 100 || kappa_d > 100) {
    lprob = log(kappa * kappa_d) - log(4 * pi() * inner) + inner - (kappa + kappa_d) + log(2);
  }
  else {   
    lprob = log(kappa * kappa_d) - log(4 * pi() * sinh(kappa) * sinh(kappa_d)) + log(sinh(inner)) - log(inner);
  }
  
  return lprob;   
}

/**
 * Calculate the vector of kappa values for different sources.
 * @param R rigidity in EV
 * @param B rms magnetic field strength in nG
 * @param D distance in Mpc / 10
 */
vector get_kappa_ex(vector R, real B, vector D) {
  
  int Ns = num_elements(R);
  vector[Ns] kappa_ex = 7552 * inv_square(2.3 * inv(R / 50) * B .* sqrt(D));
  return kappa_ex;
}
