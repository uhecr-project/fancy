/**
 * Joint model with energy, directions, and GMF for protons only. Model following Capel & Mortlock 2019
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
  real<lower=0> Rerr;

  /* uhecr */
  int<lower=0> N; 
  array[N] unit_vector[3] arrival_direction; 
  array[N] real<lower=Rth> Rdet;
  vector[N] zenith_angle;
  vector[N] A;
  vector[N] kappa_ds; /* also encodes GMF information if GMF is enabled */
  vector[N] exp_factors;

  /* composition */
  int<lower=0> Nds;       /* number of distance grid points, in Mpc */
  int<lower=0> NRs;    /* number of rigidity grid points, in EV */
  int<lower=0> Nalphas;   /* number of alpha grid points */
  vector[Nds] distances_grid;
  vector[Nalphas] alpha_grid;
  vector[NRs] log10_Rgrid;
  array[Nds] vector[NRs] Rarr_grid;
  
  
  /* Nex */
  array [Nds, Nalphas] real log10_Eexs_grid;
  int<lower=0> NBigmfs;  /* number of grid points for B */
  vector[NBigmfs] log10_Bigmf_grid;
  array [Ns, Nalphas, NBigmfs] real log10_wexp_src_grid;
  array [Nalphas] real log10_wexp_bg_grid;

  /* conversion of kappa <-> theta */
  int<lower=0> Nkappas;
  int<lower=0> Nthetas;
  vector[Nkappas] log10_kappas_grid;
  vector[Nthetas] thetas_grid;

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

  real Rmin = min(pow(10.0, log10_Rgrid));
  real Rmax = max(pow(10.0, log10_Rgrid));

}


parameters { 

  /* source luminosity, in units of log10(EeV/yr) */
  real<lower=37.0, upper=50.0> log10_L;

  // /* IGMF strength, in nG */
  real<lower=1e-3, upper=5> Bigmf;

  /* background flux, in km^-2 yr^-1 */
  real log10_F0;
  
  /* energy spectrum */
//   real<lower=-3.0, upper=3.0> alpha_s;  /* source spectral index */
//   real<lower=-3.0, upper=10.0> alpha_b;  /* background spectral index */
  real<lower=1.0, upper=10.0> alpha_s;  /* spectral index, both source & background */

  /* rigidity, in EV */
  vector<lower=Rmin, upper=Rmax>[N] R;

}


transformed parameters {
    
  /* flux properties, flux in units of km^-2 yr^-1 */
  real log10_Eex;    /* Conversion factor from Q <-> L, in units of 1/EeV */
  vector<lower=0.0>[Ns+1] F;
  real<lower=0.0> Fs;
  real<lower=0.0> FT;
  real<lower=0.0, upper=1> f1;  /* source fraction before detection */
      
  /* association probability parameters */
  array[N] real Rarr;
  array[N] vector[Ns+1] lp;  
  vector[Ns+1] log_F;

  /* spatial likelihood parameters */
  vector<lower=1e-3, upper=1e8>[N] kappas;
  
  /* Nex parameters */
  real log10_wexp;   /* log10(weighted exposure) */
  vector[Ns+1] Nex_arr;
  real<lower=0> Nex;   /* expected number of events  */
  real<lower=0> Nsrc;  /* expected number of events from sources */
  real<lower=0> Nbg;   /* expected number of events from background */
  real<lower=0.0, upper=1> f;  /* source fraction after detection */



  /* compute source flux */
  for (k in 1:Ns) {

    /* get inverse of mean energy at source */
    log10_Eex = interpolate(alpha_grid, to_vector(log10_Eexs_grid[D_indices[k],:]), alpha_s);  /* in 1/EeV */
    F[k] = ( pow(10.0, log10_L - log10_Eex) / (4.0 * pi() * pow(D_flux[k], 2.0))) ;
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

        /* energy spectrum */

        // lp[i,k] += background_spectrum_lpdf(E[i] | alpha_s, Eth, Emax);
        lp[i,k] += pareto_lpdf(R[i] | Rth, alpha_s - 1);
        Rarr[i] = interpolate(pow(10.0, log10_Rgrid), Rarr_grid[D_indices[k]], R[i]); // interpolation

        /* spatial likelihood as vMF */
        // kappas[i] = get_kappa(R[i], Bigmf, D_kappa[k]); /* E = R for protons */
        kappas[i] = pow(10.0, interpolate(thetas_grid, log10_kappas_grid , get_theta(R[i], Bigmf, D_kappa[k])));
        lp[i, k] += fik_lpdf(arrival_direction[i] | varpi[k], kappas[i], kappa_ds[i]);
      
      }

      else {
        /* same spectrum */
        // lp[i,k] += background_spectrum_lpdf(E[i] | alpha_b, Eth, Emax);
        lp[i,k] += pareto_lpdf(R[i] | Rth, alpha_s - 1);
        Rarr[i] = R[i];  
        /* spatial likelihood is all directions == 4pi */
        lp[i,k] += log(1.0 / ( 4.0 * pi() ));

      }

      /* detection probability for rigidity information */
      lp[i,k] += normal_lpdf(Rdet[i] | Rarr[i], Rerr * Rarr[i]);
      if (Rdet[i] < Rth) {
        lp[i, k] += negative_infinity();
      }
      else {
        lp[i, k] += -normal_lccdf(Rth | Rarr[i], Rerr * Rarr[i]);
      }

      /* add exposure factor */
      lp[i,k] += log(exp_factors[i]);

    }
  }

  /* Nex */
  for (k in 1:Ns+1) {

    /* number of expected events per flux (source & BG) */
    if ((k < Ns+1) ) {
      log10_wexp = interp2d(alpha_s, log10(Bigmf), to_array_1d(alpha_grid), to_array_1d(log10_Bigmf_grid),  log10_wexp_src_grid[k,:,:]);
    }
    else {
      log10_wexp = interpolate(alpha_grid, to_vector(log10_wexp_bg_grid), alpha_s);
    }

    Nex_arr[k] = F[k] * pow(10.0,log10_wexp);  
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
  alpha_s ~ normal(3.0, 3.0);
  Bigmf ~ lognormal(0.0, 3.0);
  log10_L ~ normal(42.0, 4.0);
  log10_F0 ~ normal(0.0, 10.0);

}

generated quantities {

  array[N] int lambda;

  /* used in calculating the source-UHECR association probabilities */
  for (i in 1:N) {

    lambda[i] = categorical_logit_rng(lp[i]);

  }

}


