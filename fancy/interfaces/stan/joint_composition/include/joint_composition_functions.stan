/**
 * Functions used with Stan Fits in joint model.
 *
 * @author Keito Watanabe
 * @date March 2024
 */

/**
 * Arrival rigidity spectrum for a single source, interpolated from look-up tables.
 * .
 * @param Eearth : energy at earth in EeV
 * @param alpha : source spectral index
 * @param log_Eearth_spectrum : log_e(arrival energy spectrum(alpha, R))
 * @param alpha_grid : grid of alphas for interpolation grid
 * @param log10_Eearth : grid of log10(rigidities) for interpolation grid
 */
real arrival_spectrum_lpdf(real Eearth, real alpha, array[ , ] real log_Eearth_spectrum, array [] real alpha_grid, array [] real log10_Eearth) 
 {
  real log10_Eearth = log10(Eearth);
  /* 2-D interpolatoin of arrival spectrum  */
  return interp2d(log10_Eearth, alpha, log10_Eearth, alpha_grid, log_Eearth_spectrum);
}

/**
 * Arrival rigidity spectrum for isotropic background sources.
 * .
 * @param Eearth : energy at earth in EeV
 * @param alpha : background spectral index
 * @param Emin : minimum rigidity in sample, must match min(Eearth_grid)
 * @param Emax : maximum rigidity in sample, must match max(Eearth_grid)
 */
real background_spectrum_lpdf(real Eearth, real alpha, real Emin, real Emax)
{
  real N;
  real p;
  if(alpha != 1.0)
  {
    N = ((1.0-alpha)/((Emax^(1.0-alpha))-(Emin^(1.0-alpha))));
  }
  else
  {
    N = (1.0/(log(Emax)-log(Emin)));
  }

  p = (N * pow(Eearth, (-alpha)));
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
