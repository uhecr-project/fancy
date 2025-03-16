/**
 * Joint model with energy, directions, GMF, and compositions.
 * Background-only model that only assumed BG events, with 
 * isotropic associations with power-law rigidity spectrum
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
  real<lower=0> kappa_d;

  /* uhecr */
  int<lower=0> N; 
  array[N] unit_vector[3] arrival_direction; 
  array[N] real<lower=Rth, upper=Rth_max> Rdet;
  vector[N] zenith_angle;
  vector[N] A;

  /* composition */
  int<lower=0> NDs;       /* number of distance grid points, in Mpc */
  int<lower=0> Nalphas;   /* number of alpha grid points */
  int<lower=0> NRs;    /* number of rigidity grid points, in GV */
  vector[Nalphas] alpha_grid;
  vector[NRs] log10_Rgrid;
  vector[NDs] log10_Dgrid;
  array [NDs, NRs, Nalphas] real log10_arr_spectrum_grid;
  array [NDs, Nalphas] real log10_QLratio_grid;
  
  /* Nex */
  // array [NDs, Nalphas] real log10_surv_ratio_grid;

}

transformed data {

  /* unit conversion factors */
  real km_per_Mpc = 3.08567758e19;       /* km / Mpc, for distance*/
  real EeVyr_per_ergs = 19.6967046;    /* (EeV/yr) / (erg/s) , for luminosity*/

  /* transform units for distance to make units consistent */
  vector[Ns] D_kappa;
  vector[Ns] D_flux;

  /* to store indices from distance grid in prince */
  array[Ns] int D_indices;
  
  for (k in 1:Ns) {
    D_kappa[k] = D[k] / 10;  /* D in Mpc / 10 for kappa calculation */
    D_flux[k] = D[k] * km_per_Mpc;  /* D in km for flux calculation */
    D_indices[k] = binary_search(log10(D[k]), to_array_1d(log10_Dgrid));  /* index of distance for rigidity loss tables */
  }
  
  /* min & maximum rigidity to perform simulation from */
  /* constrained by the size of the grid */
  real Rmin = min(pow(10.0, log10_Rgrid));
  real Rmax = max(pow(10.0, log10_Rgrid));

}


parameters { 

  /* log10 of source luminosity */
  real<lower=1.0, upper=60.0> log10_L;

  // /* EGMF, in nG */
  real<lower=0.0, upper=10> B;

  /* background flux, in km^-2 yr^-1 */
  real<lower=0.0, upper=10.0> F0;
  
  /* energy spectrum */
  real<lower=-3, upper=10.0> alpha_s;  /* source spectral index */
  real<lower=-3, upper=10.0> alpha_b;  /* background spectral index */

  /* rigidity, in EV */
  vector<lower=Rmin, upper=Rmax>[N] R;

}


transformed parameters {

  /* EGMF, in nG */
  // real<lower=0.0, upper=10> B=3;
  // real<lower=-3, upper=10.0> alpha_s=-1;  /* source spectral index */
  // real<lower=-3, upper=10.0> alpha_b=3;  /* background spectral index */
    
  /* particle flux, in units of km^-2 yr^-1 */
  real<lower=0.0> Fs;
  vector<lower=0.0>[Ns+1] F;
  real<lower=0.0> FT;

  /* Conversion factor from Q <-> L, in units of 1/EeV */
  real log10_QLratio;  
  
  /* associated fraction */
  real<lower=0.0, upper=1> f;  /* after detection */
  real<lower=0.0,upper=1> f1;  /* before detection */
    
  /* association probability */
  array[N] vector[Ns+1] lp;
  vector[Ns+1] log_F;

  /* spatial likelihood */
  vector<lower=0.0>[N] kappas;
  
  /* Nex */
  real<lower=0.0> surv_ratio;  /* fraction of survived particles  */

  vector[Ns+1] Nex_arr;
  real<lower=0> Nex;   /* expected number of events  */
  real<lower=0> Nsrc;  /* expected number of events from sources */
  real<lower=0> Nbg;   /* expected number of events from background */

  /* define transformed paramaters */

  /* compute flux, in units of km^-2 yr^-1 */
  for (k in 1:Ns) {

    /* get conversion from Q to L */
    log10_QLratio = interpolate(alpha_grid, to_vector(log10_QLratio_grid[D_indices[k],:]), alpha_s);  /* in log10(1/EeV) */

    /* to make units consistent, need to add some conversion factors */
    /* D_flux is in km, Q/L is in 1/EeV, L is in ( erg / s)  */
    /* so to make units of flux to km^-2 yr^-1, need factor of */
    /* ( (EeV / yr) / (erg / s)) */
    F[k] = (pow(10.0, log10_L + log10_QLratio) * EeVyr_per_ergs / (4.0 * pi() * pow(D_flux[k], 2.0)) ) ;
  }
  
  Fs = sum(F[1:Ns]);  /* source flux */
  F[Ns+1] = F0;  /* background flux */

  log_F = log(F);

  FT = F0 + Fs;  /* total flux */
  f1 = Fs / FT;  /* association fraction from flux definition (before detection) */

  /* likelihood calculation */
  /* rate factor */
  for (i in 1:N) {

    lp[i] = log_F;

    for (k in 1:Ns+1) {

      /* sources */
      if ((k < Ns+1) ) {

        /* rigidity spectrum */
        lp[i,k] += arrival_spectrum_lpdf(R[i] | alpha_s, log10_arr_spectrum_grid[D_indices[k],:,:], to_array_1d(alpha_grid), to_array_1d(log10_Rgrid));
        // lp[i, k] += background_spectrum_lpdf(R[i] | alpha_s, Rmin, Rmax);

        /* spatial likelihood as vMF */
        kappas[i] = get_kappa(R[i], B, D_kappa[k]);
        lp[i, k] += fik_lpdf(arrival_direction[i] | varpi[k], kappas[i], kappa_d);
      
      }

      else {

        /* bounded rigidity spectrum */
        lp[i, k] += background_spectrum_lpdf(R[i] | alpha_b, Rmin, Rmax);
        /* spatial likelihood is all directions == 4pi */
        lp[i,k] += log(1.0 / ( 4.0 * pi() ));

      }

      /* truncated gaussian */
      lp[i,k] += normal_lpdf(Rdet[i] | R[i], Rerr * R[i]);
      if (Rdet[i] < Rth)
      {
        lp[i,k] += negative_infinity();
      }
      else
      {
        lp[i,k] += -normal_lccdf(Rth | R[i], Rerr * R[i]);
      }

    }
            
  }

  /* Nex */
  for (k in 1:Ns) {

    /* loss factor */
    // surv_ratio = pow(10.0, interpolate(alpha_grid, to_vector(log10_surv_ratio_grid[D_indices[k],:]), alpha_s));
    surv_ratio = 0.5;

    /* Nex calculation */
    Nex_arr[k] = F[k] * surv_ratio * (alpha_T / (4.0 * pi()));
  }

  /* evalluate for BG model */
  Nex_arr[Ns+1] = F0 * (alpha_T / (4.0 * pi()));

  Nex = sum(Nex_arr);
  Nsrc = sum(Nex_arr[1:Ns]);
  Nbg = Nex_arr[Ns+1];  /* not needed, but good to keep for diagnosis */

  /* evaluate source fraction as fraction of **detected** events from source vs BG */
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
  alpha_s ~ normal(2.0, 3.0);
  alpha_b ~ normal(5.0, 3.0);
  B ~ normal(1.0, 3.0);
  log10_L ~ normal(42.0, 6.0);
  F0 ~ normal(0.0, 1.0);  /* remove this */

}

generated quantities {

  array[N] int lambda;

  /* used in calculating the source-UHECR association probabilities */
  for (i in 1:N) {

    lambda[i] = categorical_logit_rng(lp[i]);

  }

}


