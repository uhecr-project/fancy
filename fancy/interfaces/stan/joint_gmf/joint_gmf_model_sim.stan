/**
 * Joint model for UHECR energies and arrival directions.
 * Generative model (i.e. forward simulation)
 *
 * @author Francesca Capel
 * @date October 2018
 */

functions {

#include /include/uhecr_propagation.stan
#include /include/observatory_exposure_rng.stan
#include /include/vMF_rng.stan
#include /include/joint_model_sim_functions.stan
  
}

data {

  /* source positions */
  int<lower=0> Ns;
  array[Ns] unit_vector[3] varpi;
  array[Ns] real D;

  /* source spectrum */
  real alpha;
  real<lower=0> Eth;
  real<lower=0> Eerr;
  
  /* flux */
  array[Ns] real<lower=0> Q;
  real<lower=0> F0;
  
  /* deflection */
  real<lower=0> B;
  int Z; 

  /* observatory parameters */
  real A;
  real a0;
  real<lower=0> theta_m;
  real<lower=0> alpha_T;
  vector[Ns] eps;
  
}

transformed data {
  
  /* definitions */
  real<lower=0> Fs = get_Fs(Q , D);
  real<lower=0> F_T = F0 + Fs;
  real<lower=0, upper=1> f = Fs / F_T;
  simplex[Ns] w = get_source_weights(Q , D);  
  vector[Ns+1] F;

  array[1] real x_r;
  array[0] int x_i;
  vector[Ns] Eth_src;
  array[Ns,1] real D_in;
  vector[Ns] D_kappa;
  
  simplex[Ns+1] w_exposure;
  real<lower=0> Nex;
  int<lower=0> N;

  /* flux and distance */
  for (k in 1:Ns) {
    F[k] = w[k] * Fs;
    D_in[k, 1] = (D[k] / 3.086) * 100 - 0.02; // Mpc
  }
  F[Ns+1] = F0;

  /* 
  D in Mpc / 10 for kappa calculation 
  Subtract by 20kpc to account for galactic boundary
  */
  for (k in 1:Ns) {
    D_kappa[k] = ((D[k] / 3.086) * 10) - 0.2; // Mpc / 10
  }
 
  /* Eth_src */
  x_r[1] = 1.0e4; // approx inf
  Eth_src = get_Eth_src_sim(Eth, D_in, x_r, x_i);
  
  /* N */
  w_exposure = get_exposure_weights(F, eps, alpha_T, Eth_src, Eth, alpha);

  Nex = get_Nex_sim(F, eps, alpha_T, Eth_src, Eth, alpha);
  
  N = poisson_rng(Nex);

}

generated quantities {

  array[N] int lambda;
  // unit_vector[3] omega;
  // unit_vector[3] arrival_direction[N];
  real Nex_sim = Nex;
  
  array[N] real E;
  array[N] real kappa;
  array[N] real Earr;
  array[N] real Edet;
  
  for (i in 1:N) {

    /* label */
    lambda[i] = categorical_rng(w_exposure);

    /* source */
    if (lambda[i] < Ns+1) {

      E[i] = spectrum_rng(alpha, Eth_src[lambda[i]]);
      kappa[i] = get_kappa(E[i], B, D_kappa[lambda[i]], Z);
      Earr[i] = get_arrival_energy_sim(E[i], D_in[lambda[i]], x_r, x_i);

    }
  
    /* background */
    if (lambda[i] == Ns+1) {

      E[i] = spectrum_rng(alpha, Eth);
      kappa[i] = not_a_number();  // kappa not defined for background propagation
      Earr[i] = E[i];

    }
    
    /* detection */ 	  
    Edet[i] = normal_rng(Earr[i], Eerr * Earr[i]);

  }
  
}

