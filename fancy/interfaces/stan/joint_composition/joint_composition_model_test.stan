/**
 * Joint model with energy, directions, GMF, and compositions.
 *
 * @author Keito Watanabe
 * @date March 2024
 */

functions {

#include /include/joint_composition_functions.stan
#include /include/utils.stan

}

data {

  /* sources */
  int<lower=1> Ns;
  array[Ns] unit_vector[3] varpi; 
  vector[Ns] D;
   
  /* observatory */ 
  real<lower=0> alpha_T;
  real<lower=0> Rth;
  real<lower=0> Rth_max;
  real<lower=0> Rerr;

  /* uhecr */
  int<lower=0> N; 
  array[N] unit_vector[3] arrival_direction; 
  array[N] real<lower=Rth> Rdet;
  vector[N] zenith_angle;
  vector[N] A;
  vector[N] kappa_ds;
  vector[N] exp_factors;

  /* composition */
  int<lower=0> Ndmaxs;       /* number of distance grid points, in Mpc */
  int<lower=0> NRs;    /* number of rigidity grid points, in EV */
  int<lower=0> Nalphas;   /* number of alpha grid points */
  vector[Ndmaxs] dmax_grid;
  vector[NRs] log10_Rgrid;
  vector[Nalphas] alpha_grid;
  array [Ndmaxs, NRs, Nalphas] real log_arr_spectrum_grid;  /* in log_e!! */
  
  /* Nex */
  array [Ndmaxs, Nalphas] real log10_QLratios_grid;
  int<lower=0> NBigmfs;  /* number of grid points for B */
  vector[NBigmfs] log10_Bigmf_grid;
  array [Ns+1, Nalphas, NBigmfs] real log10_Neff_per_flux_grid;  /* Ns+1 since it includes BG */

}

transformed data {

  /* unit conversion factors */
  real km_per_Mpc = 3.08567758e19;       /* km / Mpc, for distance*/

  /* transform units for distance to make units consistent */
  vector[Ns] D_kappa;
  vector[Ns] D_flux;

  /* to store indices from distance grid in prince */
  array[Ns] int D_indices;
  
  for (k in 1:Ns) {
    D_kappa[k] = D[k] / 10;  /* D in Mpc / 10 for kappa calculation */
    D_flux[k] = D[k] * km_per_Mpc;  /* D in km for flux calculation */
    D_indices[k] = binary_search(D[k], to_array_1d(dmax_grid));  /* index of distance for rigidity loss tables */
  }

  real Rmin = min(pow(10.0, log10_Rgrid));
  real Rmax = max(pow(10.0, log10_Rgrid));

}


parameters { 

  /* source luminosity, in units of log10(EeV/yr) */
  real<lower=37.0, upper=50.0> log10_L;

  // /* IGMF strength, in nG */
  real<lower=1e-6, upper=20> Bigmf;

  /* background flux, in km^-2 yr^-1 */
  real log10_F0;
  
  /* energy spectrum */
  real<lower=-3.0, upper=3.0> alpha_s;  /* source spectral index */
  real<lower=-3.0, upper=10.0> alpha_b;  /* background spectral index */

  /* rigidity, in EV */
  vector<lower=Rmin, upper=Rmax>[N] R;

}


transformed parameters {
    
  /* flux properties, flux in units of km^-2 yr^-1 */
  real log10_QLratio;    /* Conversion factor from Q <-> L, in units of 1/EeV */
  vector<lower=0.0>[Ns+1] F;
  real<lower=0.0> Fs;
  real<lower=0.0> FT;
  real<lower=0.0, upper=1> f1;  /* source fraction before detection */
      
  /* association probability parameters */
  array[N] vector[Ns+1] lp;  
  vector[Ns+1] log_F;

  /* spatial likelihood parameters */
  vector<lower=0.0>[N] kappas;
  
  /* Nex parameters */
  real log10_Neff_per_flux;  
  vector[Ns+1] Nex_arr;
  real<lower=0> Nex;   /* expected number of events  */
  real<lower=0> Nsrc;  /* expected number of events from sources */
  real<lower=0> Nbg;   /* expected number of events from background */
  real<lower=0.0, upper=1> f;  /* source fraction after detection */



  /* compute source flux */
  for (k in 1:Ns) {

    /* get conversion from Q to L */
    log10_QLratio = interpolate(alpha_grid, to_vector(log10_QLratios_grid[D_indices[k],:]), alpha_s);  /* in 1/EeV */
    F[k] = ( pow(10.0, log10_L + log10_QLratio) / (4.0 * pi() * pow(D_flux[k], 2.0))) ;
  }
  
  Fs = sum(F[1:Ns]);  /* source flux */
  F[Ns+1] = pow(10.0, log10_F0);  /* background flux */

  log_F = log(F);

  FT = sum(F);  /* total flux */
  f1 = Fs / FT;  /* association fraction from flux definition (before detection) */

  /* likelihood calculation */
  /* rate factor */
  for (i in 1:N) {

    lp[i] = log_F;

    for (k in 1:Ns+1) {

      /* sources */
      if ((k < Ns+1) ) {

        /* rigidity spectrum */

        lp[i,k] += arrival_spectrum_lpdf(R[i] | alpha_s, log_arr_spectrum_grid[D_indices[k],:,:], to_array_1d(alpha_grid), to_array_1d(log10_Rgrid));
        // lp[i,k] += background_spectrum_lpdf(R[i] | alpha_s, Rmin, Rmax);

        /* spatial likelihood as vMF */
        kappas[i] = get_kappa(R[i], Bigmf, D_kappa[k]);
        lp[i, k] += fik_lpdf(arrival_direction[i] | varpi[k], kappas[i], kappa_ds[i]);
      
      }

      else {
        /* bounded rigidity spectrum */
        lp[i,k] += background_spectrum_lpdf(R[i] | alpha_b, Rmin, Rmax);
        /* spatial likelihood is all directions == 4pi */
        lp[i,k] += log(1.0 / ( 4.0 * pi() ));

      }

      /* detection probability for rigidity information */
      lp[i,k] += normal_lpdf(Rdet[i] | R[i], Rerr * R[i]);
      if (Rdet[i] < Rth || Rdet[i] > Rth_max) {
        lp[i,k] += negative_infinity();
      }
      else {
        lp[i,k] += -log_diff_exp(normal_lcdf(Rth_max | R[i], Rerr * R[i]), 
                                normal_lcdf(Rth | R[i], Rerr * R[i]));
      }

      /* add exposure factor */
      lp[i,k] += log(exp_factors[i]);

    }
  }

  /* Nex */
  for (k in 1:Ns+1) {

    /* number of expected events per flux (source & BG) */
    if ((k < Ns+1) ) {
      log10_Neff_per_flux = interp2d(alpha_s, log10(Bigmf), to_array_1d(alpha_grid), to_array_1d(log10_Bigmf_grid),  log10_Neff_per_flux_grid[k,:,:]);
    }
    else {
      log10_Neff_per_flux = interp2d(alpha_b, log10(Bigmf), to_array_1d(alpha_grid), to_array_1d(log10_Bigmf_grid),  log10_Neff_per_flux_grid[k,:,:]);
    }

    Nex_arr[k] = F[k] * pow(10.0,log10_Neff_per_flux);  
  }

  Nex = sum(Nex_arr);
  Nsrc = sum(Nex_arr[1:Ns]);
  Nbg = Nex_arr[Ns+1];  /* not needed, but good to keep for diagnosis */

  /* source fraction as fraction of **detected** events from source vs BG */
  f = Nsrc / Nex;
  
}


model {

  /* rate factor */
  for (i in 1:N) {
    target += log_sum_exp(lp[i]);
  }
  
  /* normalise */
  target += -Nex; 

  /* priors */
  alpha_s ~ normal(0.0, 3.0);
  alpha_b ~ normal(5.0, 5.0);
  Bigmf ~ normal(0.0, 7.0);
  log10_L ~ normal(42.0, 4.0);
  log10_F0 ~ normal(0.0, 10.0);
  // F0 ~ normal(0.0, 100.0);

}

generated quantities {

  array[N] int lambda;

  /* used in calculating the source-UHECR association probabilities */
  for (i in 1:N) {

    lambda[i] = categorical_logit_rng(lp[i]);

  }

}


