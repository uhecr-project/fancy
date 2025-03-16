/**
 * Functions used with Stan simulations in joint model.
 *
 * @author Francesca Capel
 * @date October 2018
 */



/**
 * Calculate weights from source distances.
 */
vector get_source_weights(array[] real Q, array[] real D) {
  
  int N = num_elements(D);
  vector[N] weights;
  
  real normalisation = 0;
  
  for (k in 1:N) {
    normalisation += (Q[k] / pow(D[k], 2));
  }
  for (k in 1:N) {
    weights[k] = (Q[k] / pow(D[k], 2)) / normalisation;
  }
  
  return weights;
}

/**
 * Calculate weights for each source accounting for exposure 
 * and propagation effects.
 */
vector get_exposure_weights(vector F, vector eps, real alpha_T, vector Eth_src, real Eth, real alpha) {
  
  int N = num_elements(F);
  vector[N] weights;
  
  real normalisation = 0;
  
  for (k in 1:N-1) {
    normalisation += F[k] * eps[k] * pow(Eth_src[k] / Eth, 1 - alpha);
  }
  normalisation += F[N] * (alpha_T / (4 * pi()));
  
  for (k in 1:N-1) {
    weights[k] = (F[k] * eps[k] * pow(Eth_src[k] / Eth, 1 - alpha)) / normalisation;
  }
  weights[N] = (F[N] * (alpha_T / (4 * pi()))) / normalisation;
  
  return weights;
}

/**
 * Calculate the expected value of N for the generative model.
 */
real get_Nex_sim(vector F, vector eps, real alpha_T, vector Eth_src, real Eth, real alpha) {
  
  int N = num_elements(F);
  real Nex = 0;
  
  for (k in 1:N-1) {
    Nex += F[k] * eps[k] * pow(Eth_src[k] / Eth, 1 - alpha);
  }
  Nex += F[N] * (alpha_T / (4 * pi()));

  return Nex;
}

/**
 * Calculate the deflection parameter of the vMF distribution.
 * Derived from Eq. 4 and Eq. 9 in Cape; & Mortlock, 2018.
 * @param E energy in EeV
 * @param B rms magnetic field strength in nG
 * @param D distance in Mpc / 10
 */
real get_kappa(real E, real B, real D) {
  
  return 2.3 * inv_square( 0.0401 * inv(E / 50) * B * sqrt(D) );
}

/**
 * Shape of the energy spectrum: a power law.
 */
real dNdE_pl(real E, real alpha) {
  
  real spec = pow(E, -alpha);
  return spec; 
  
}

/**
 * Sample an energy from a power law spectrum defined by alpha.
 * Uses rejection sampling.
 * Sampled energy is in units of EeV.
 * @param alpha the spectral index
 * @param Emin the minimum energy
 */  
real spectrum_rng(real alpha, real Emin) {
  
  real E;
  real d;
  real d_upp_lim = dNdE_pl(Emin, alpha);

  int accept = 0;
  
  while(accept != 1) {
    
    E = uniform_rng(Emin, 1e4);
    d = uniform_rng(0, d_upp_lim);
    
    if (d < dNdE_pl(E, alpha)) {
      accept = 1;
    }
  }

  return E;
}