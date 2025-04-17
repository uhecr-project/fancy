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
  real<lower=0> Eth;
  real<lower=0> Eerr;
  array[2] real mean_lnA_params;
  array[2] real sigma_lnA_params;
  real<lower=0> lnA_min;
  real<lower=0> lnA_max;

  /* uhecr */
  int<lower=0> N; 
  array[N] unit_vector[3] arrival_direction; 
  array[N] real<lower=Eth> Edet;
  vector[N] kappa_ds; /* also encodes GMF information if GMF is enabled */
  vector[N] exp_factors;

  /* composition */
  int<lower=0> Nds;       /* number of distance grid points, in Mpc */
  int<lower=0> NEearths;    /* number of energy grid points, in EeV */
  int<lower=0> Nalphas;   /* number of alpha grid points */
  vector[Nds] distances_grid;
  vector[NEearths] log10_Eearth_grid;
  vector[Nalphas] alpha_grid;
  array [Nds, NEearths, Nalphas] real log_Eearth_spectrum;  /* in log_e!! */
  
  /* Nex */
  array [Nds, Nalphas] real log10_Esrcs_grid;
  int<lower=0> NBigmfs;  /* number of grid points for B */
  vector[NBigmfs] log10_Bigmf_grid;
  array [Ns, Nalphas, NBigmfs] real log10_source_exposure_grid;
  array [Nalphas] real log10_backgrond_exposure_grid;

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
    D_indices[k] = binary_search(D[k], to_array_1d(distances_grid));  /* index of distance for rigidity loss tables */
  }

  real Emin = min(pow(10.0, log10_Eearth_grid));
  real Emax = max(pow(10.0, log10_Eearth_grid));

}


parameters { 

  /* source luminosity, in units of log10(EeV/yr) */
  real<lower=35.0, upper=60.0> log10_L;

  // /* IGMF strength, in nG Mpc^1/2 */
  real<lower=-3, upper=1> log10_Bigmf;

  /* background flux, in km^-2 yr^-1 */
  real<lower=-5, upper=5> log10_F0;
  
  /* energy spectrum */
  real<lower=-3.0, upper=3.0> alpha_s;  /* source spectral index */
  real<lower=2.0, upper=10.0> alpha_b;  /* background spectral index */

  /* energy in EV */
  vector<lower=Emin, upper=Emax>[N] Eearth;

  /* earth composition */
  vector<lower=lnA_min, upper=lnA_max>[N] lnAearth;

}


transformed parameters {
    
  /* flux properties, flux in units of km^-2 yr^-1 */
  real log10_Esrc;    /* Conversion factor from Q <-> L, in units of 1/EeV */
  vector<lower=0.0>[Ns+1] F;
  real<lower=0.0> Fs;
  real<lower=0.0> FT;
  real<lower=0.0, upper=1> f1;  /* source fraction before detection */
      
  /* association probability parameters */
  array[N] vector[Ns+1] lp;  
  vector[Ns+1] log_F;

  /* spatial likelihood parameters */
  // vector<lower=1e-3, upper=1e6>[N] kappas;
  vector[N] kappas;
  vector[N] rigidities;
  
  /* Nex parameters */
  real log10_exposure;   /* log10(weighted exposure) */
  vector[Ns+1] log10_exposure_arr;
  vector[Ns+1] Nex_arr;
  real<lower=0> Nex;   /* expected number of events  */
  real<lower=0> Nsrc;  /* expected number of events from sources */
  real<lower=0> Nbg;   /* expected number of events from background */
  real<lower=0.0, upper=1> f;  /* source fraction after detection */



  /* compute source flux */
  for (k in 1:Ns) {

    /* get inverse of mean energy at source */
    log10_Esrc = interpolate(alpha_grid, to_vector(log10_Esrcs_grid[D_indices[k],:]), alpha_s);  /* in 1/EeV */
    F[k] = ( pow(10.0, log10_L - log10_Esrc) / (4.0 * pi() * pow(D_flux[k], 2.0))) ;
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

        // /* energy spectrum */
        lp[i,k] += arrival_spectrum_lpdf(Eearth[i] | alpha_s, log_Eearth_spectrum[D_indices[k],:,:], to_array_1d(alpha_grid), to_array_1d(log10_Eearth_grid));
        /* bounded rigidity spectrum */
        // lp[i,k] += background_spectrum_lpdf(R[i] | alpha_s, Rmin, Rmax);

        /* earth composition information as truncated Gaussian */
        mean_lnA = mean_lnA_params[1] * log10(Eearth[i]) + mean_lnA_params[2];
        sigma_lnA = sigma_lnA_params[1] * log10(Eearth[i]) + sigma_lnA_params[2];

        lp[i,k] += normal_lpdf(lnAearth[i] | mean_lnA, sigma_lnA) - normal_lcdf(lnA_max | mean_lnA, sigma_lnA) - normal_lcdf(lnA_min | mean_lnA, sigma_lnA)

        /* compute rigidities directly */
        rigidities[i] = Eearth[i] / (0.5 * exp(lnAearth[i]));

        /* spatial likelihood as vMF */
        kappas[i] = get_kappa(rigidities[i], pow(10.0, log10_Bigmf), D_kappa[k]);
        if (kappas[i] > 1e6) {
          kappas[i] = 1e6;
        }
        // kappas[i] = pow(10.0, interpolate(thetas_grid, log10_kappas_grid , get_theta(R[i], Bigmf, D_kappa[k])));
        lp[i, k] += fik_lpdf(arrival_direction[i] | varpi[k], kappas[i], kappa_ds[i]);

        /* spatial likelihood is all directions == 4pi */
        lp[i,k] += log(1.0 / ( 4.0 * pi() ));
      
      }

      else {
        /* bounded energy spectrum */
        lp[i,k] += background_spectrum_lpdf(Eearth[i] | alpha_b, Emin, Emax);

        /* earth composition information as truncated Gaussian */
        mean_lnA = mean_lnA_params[1] * log10(Eearth[i]) + mean_lnA_params[2];
        sigma_lnA = sigma_lnA_params[1] * log10(Eearth[i]) + sigma_lnA_params[2];

        lp[i,k] += normal_lpdf(lnAearth[i] | mean_lnA, sigma_lnA) - normal_lcdf(lnA_max | mean_lnA, sigma_lnA) - normal_lcdf(lnA_min | mean_lnA, sigma_lnA)

        /* spatial likelihood is all directions == 4pi */
        lp[i,k] += log(1.0 / ( 4.0 * pi() ));

      }

      /* detection probability for energy information */
      lp[i,k] += normal_lpdf(Edet[i] | Eearth[i], Eerr * Eearth[i]);
      if (Edet[i] < Eth) {
        lp[i, k] += negative_infinity();
      }
      else {
        lp[i, k] += -normal_lccdf(Eth | Eearth[i], Eerr * Eearth[i]);
      }

      /* add exposure factor */
      lp[i,k] += log(exp_factors[i]);

    }
  }

  /* Nex */
  for (k in 1:Ns+1) {

    /* number of expected events per flux (source & BG) */
    if ((k < Ns+1) ) {
      log10_exposure = interp2d(alpha_s, log10_Bigmf, to_array_1d(alpha_grid), to_array_1d(log10_Bigmf_grid),  log10_source_exposure_grid[k,:,:]);
    }
    else {
      log10_exposure = interpolate(alpha_grid, to_vector(log10_background_exposure_grid), alpha_b);
    }

    log10_exposure_arr[k] = log10_exposure;
    Nex_arr[k] = F[k] * pow(10.0,log10_exposure);
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
  log10_Bigmf ~ normal(0.0, 3.0);
  log10_L ~ normal(42.0, 4.0);
  log10_F0 ~ normal(-0.5, 3.0);

}

generated quantities {

  array[N] int lambda;

  /* used in calculating the source-UHECR association probabilities */
  for (i in 1:N) {

    lambda[i] = categorical_logit_rng(lp[i]);

  }

}


