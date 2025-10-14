/**
 * Energy + lnA model
 *
 * @author Keito Watanabe
 * @date March 2024
 */

functions {
    #include /utils.stan

     /**
    * sample from the energy spectrum at Earth
    * @param E energy in EeV
    * @param alpha spectral index
    * @param log_en_grid : grid of log(E) values for interpolation
    * @param alph_grid : grid of alpha values for interpolation
    * @param espect_mfs : mass fraction weighted energy spectrum at Earth. Shape in (Nalphas, NEs)
    * @return log probability density of the energy spectrum at Earth
    */
    real energy_spectrum_lpdf(real logE, real alpha, array [] real log_en_grid, array [] real alph_grid, matrix log_espect_mfs) {
        return interp2d(alpha, logE, alph_grid, log_en_grid, to_array_2d(log_espect_mfs));
    }

    /**
    * Calculate the deflection parameter of the vMF distribution.
    * Derived from Eq. 4 and Eq. 9 in Capel & Mortlock, 2018.
    * @param R rigidity in EV
    * @param mag_beta magnetic spread in nG Mpc^(1/2)
    * @param D distance in Mpc / 10
    * @return deflection parameter kappa
    */
    real get_kappa(real R, real mag_beta, real D) {
    
    return 7552.0 * inv_square(2.3 * inv(R / 50.0) * mag_beta * sqrt(D));
    }

    /**
    * Define the fik PDF.
    * NB: Cannot be vectorised.
    * Uses sinh(kappa) ~ exp(kappa)/2 
    * approximation for kappa > 100.
    */
    real fik_lpdf(vector v, vector mu, real kappa, real kappa_d) {
    
    real lprob;
    real inner = sqrt(dot_self((kappa_d * v) + (kappa * mu)));
    
    if (kappa > 100 || kappa_d > 100) {
        lprob = log(kappa * kappa_d) - log(4 * pi() * inner) + inner - (kappa + kappa_d) + log(2);
    }
    else {   
        lprob = log(kappa * kappa_d) - log(4 * pi() * sinh(kappa) * sinh(kappa_d)) + log(sinh(inner)) - log(inner);
    }
    
    return lprob;   
    }

// --- per-event likelihood chunk for reduce_sum ---
  real event_likelihood_chunk(
      array [] int slice_i,         // indices of events in this chunk
      int start, int end,           // range of indices in this chunk
      real alpha_bg,                // alphas per source
      real F0,                     // fluxes per source
      array [] real alpha_grid,     // grid of alpha for energy interpolation
      array [] real logE_grid,    // grid of log10(E) for energy interpolation
      matrix espect_mfs,          // energy spectra at Earth per source
      vector logE_true,                 // latent true energies
      vector Edet,                  // detected energies
      real logE_stat_unc,           // statistical energy uncertainty (log-normal)
      real logE_sys_unc,            // systematic energy uncertainty, global systematic shift
      real Emin, real Emax,          // energy range for truncated lognormal likelihood
      vector mean_lnA_true,         // mean lnA per energy bin per source
      vector var_lnA_true,          // variance of lnA per energy bin per source
      array[] real lnA_logE_grid,   // grid of energy bins for finding the mean and variance of lnA per energy
      vector nu_lnAs,              // latent variable for sampling Zsrcs
      array [] vector omega_det,    // detected directions
      real beta_egmf         // EGMF spread parameter
  ) {
      real lp_chunk = 0.0;  // log likelihood for this chunk
      int len = size(slice_i); // number of events in this chunk
        // loop over events in this chunk
      for (n in 1:len) {
          // the index of the event
          int i = slice_i[n];

          // log likelihood per source + isotropic background
          real lp_i = log(F0);

          // pre-computation for the rigidity
          // computing the mean and variance of lnA through a binned search
          real mean_lnA = mean_lnA_true[binary_search(logE_true[i], lnA_logE_grid)];
          real var_lnA = var_lnA_true[binary_search(logE_true[i], lnA_logE_grid)];

          // // calculating the true charge and rigidity
          real Zsrc = 0.5 * exp(mean_lnA + sqrt(var_lnA) * nu_lnAs[i]);
          real Rtrue = exp(logE_true[i]) / Zsrc;
          
          // energy sampling (spectrum at Earth + truncated lognormal)
          lp_i += energy_spectrum_lpdf(logE_true[i] | alpha_bg, logE_grid, alpha_grid, log(espect_mfs));
          lp_i += truncated_lognormal_lpdf(Edet[i] | logE_true[i] + logE_sys_unc, logE_stat_unc, Emin, Emax);
          /* isotropic background */
          lp_i += -log(4*pi());

          lp_chunk += lp_i;
        }      

      return lp_chunk;
  }


}

data {

    // /* sources */
    int<lower=1> Nsrcs;
    vector[Nsrcs] D;
    array[Nsrcs] unit_vector[3] omega_src;  /* source directions */

    /* uhecr */
    int<lower=0> N;
    int<lower=0> NEbins;  /* number of energy bins for lnA measurements */
    vector[N] Edet;
    array[N] unit_vector[3] omega_det; /* arrival directions */
    vector[N] kappa_ds; /* deflection parameters, including GMF deflections + arrival direction uncertainty */
    array[NEbins] real mean_lnA_det;
    array[NEbins] real var_lnA_det;

    /* model */
    int<lower=0> Nalphas;   /* number of alpha grid points */
    int<lower=0> NEs;    /* number of energy grid points in EeV */
    int<lower=0> NAsrcs;  /* number of source masses */
    array[Nalphas] real alpha_grid;
    array[NEs] real logE_grid;
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEs] earth_spectrum_grid;
    /* for lnA, binned via detected energies */
    array[NEbins] real lnA_logE_grid; /* grid of energy bins for lnA model */
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEbins] mean_lnA_grid;
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEbins] var_lnA_grid;


    /* detector */
    real<lower=0> Emin;
    real Emax;

    /* statitistical uncertainty */
    real<lower=0> logE_stat_unc;
    vector[NEbins] mean_lnA_stat_unc;
    vector[NEbins] var_lnA_stat_unc;

    /* systematic uncertainties, as global shifts */
    real logE_sys_unc;
    real mean_lnA_sys_unc;
    real var_lnA_sys_unc;

    /* Nex */
    int <lower=0> Nbeta_egmfs;
    array [Nbeta_egmfs] real log10_beta_egmf_grid; /* grid of EGMF spread parameters */
    /* grid of weighted exposures at earth and source */
    array[Nsrcs+1, NAsrcs] matrix[Nalphas, Nbeta_egmfs] log_wexp_earth_grid;
    array[Nsrcs, NAsrcs] matrix[Nalphas, Nbeta_egmfs] log_wexp_src_grid;
    array[Nsrcs, NAsrcs] vector[Nalphas] esrc_ratio_grid;

    /* computation parameters */
    int<lower=1> grain_size; /* for reduce_sum, generatlly N / (4 * ncores) is a good estimate */
}

transformed data {
  // --- constants ---
  real logEmin = log(Emin);
  real logEmax = log(Emax);

  real alpha_min = min(alpha_grid);
  real alpha_max = max(alpha_grid);

  real beta_egmf_min = pow(10.0, min(log10_beta_egmf_grid));
  real beta_egmf_max = pow(10.0, max(log10_beta_egmf_grid));

  // --- distance conversion once ---
  vector[Nsrcs] D_flux = D * 3.08567758e19;

  // --- precompute 1D arrays for interpolation ---
  vector[Nalphas] alpha_grid_vec = to_vector(alpha_grid);

  // number of events for reduced sum
    array[N] int event_ids;
    for (n in 1:N) {event_ids[n] = n;}
}

parameters {

    /* spectral information */
    real <lower=alpha_min, upper=alpha_max> alpha_bg;

    /* mass fractions (in future, 2D structure with sources) */
    simplex[NAsrcs] mass_fracs_bg;   

    /* total flux AT EARTH */
    real log10_F0;

    /* EGMF spread parameter, in nG Mpc^1/2 */
    real<lower=beta_egmf_min, upper=beta_egmf_max> beta_egmf;

    /* latent parameters for energy */
    vector <lower=logEmin, upper=logEmax>[N] logE_true;

    vector[N] nu_lnAs; /* latent variable for sampling lnA (Zsrcs) */

}

transformed parameters {
  // --- only quantities needed downstream ---
  real F0 = pow(10.0, log10_F0);
  matrix [Nalphas, NEs] espect_mfs;
  matrix [Nalphas, NEbins] mulnA_mfs;
  matrix [Nalphas, NEbins] varlnA_mfs;
  real wexp_earth;

  // initialise
  espect_mfs = rep_matrix(0.0, Nalphas, NEs);
  mulnA_mfs = rep_matrix(0.0, Nalphas, NEbins);
  varlnA_mfs = rep_matrix(0.0, Nalphas, NEbins);
  wexp_earth = 0.0;

  // accumulate spectra and weights
  for (j in 1:NAsrcs) {
    espect_mfs += mass_fracs_bg[j] * earth_spectrum_grid[Nsrcs+1,j];
    mulnA_mfs += mass_fracs_bg[j] * mean_lnA_grid[Nsrcs+1,j];
    varlnA_mfs += mass_fracs_bg[j] * var_lnA_grid[Nsrcs+1,j];

    wexp_earth += mass_fracs_bg[j] * exp(interp2d(
      alpha_bg, log10(beta_egmf),
      alpha_grid, log10_beta_egmf_grid,
      to_array_2d(log_wexp_earth_grid[Nsrcs+1,j])
    ));
  }

  real Nex = F0 * wexp_earth;

  // mean and sigma lnA values
  vector[NEbins] mean_lnA_true = rep_vector(0.0, NEbins);
  vector[NEbins] var_lnA_true = rep_vector(0.0, NEbins);

   // --- binned lnA likelihood ---
  for (l in 1:NEbins) {

      /* calculate the mean and variance of lnA for each energy bin */
      mean_lnA_true[l] += interpolate(alpha_grid_vec, to_vector(mulnA_mfs[,l]), alpha_bg);
      var_lnA_true[l] += interpolate(alpha_grid_vec, to_vector(varlnA_mfs[,l]), alpha_bg);

  }
}

model {
  // --- priors ---
  // spectral indices : normal distribution
  alpha_bg ~ normal(0.0, 2.0);

  // mass fractions : Dirichlet distribution per source
  mass_fracs_bg ~ dirichlet([2.0, 3.0, 2.0]);

  // total flux : normal distribution in log10
  log10_F0 ~ normal(-1.0, 3.0);

  // magnetic spread: normal in log10
  // log10_beta_egmf ~ normal(log10(0.5), 0.1);
  beta_egmf ~ normal(0.0, 1.0);

  // latent variables for lnA : normal distribution
  nu_lnAs ~ normal(0.0, 1.0);

   // --- binned lnA likelihood ---
  for (l in 1:NEbins) {
    target += left_truncated_normal_lpdf(mean_lnA_det[l] |
              mean_lnA_true[l] + mean_lnA_sys_unc,
              mean_lnA_stat_unc[l], 0.0);
    target += left_truncated_normal_lpdf(var_lnA_det[l] |
              var_lnA_true[l] + var_lnA_sys_unc,
              var_lnA_stat_unc[l], -2.0);
  }

  // --- parallelized unbinned likelihood ---
  target += reduce_sum(
    event_likelihood_chunk,         // the per-chunk unbinned likelihood function
    event_ids,                      // the data to be sliced and passed to each chunk
    grain_size,                     // tuning parameter for the chunk size
    alpha_bg,                         // spectral indices
    F0,                              // fluxes per source
    alpha_grid,                     // alpha grid for interpolation        
    logE_grid,                      // log10(E) grid for interpolation  
    espect_mfs,                     // energy spectra at Earth per source
    logE_true,                      // latent true energies
    Edet,                           // detected energies
    logE_stat_unc,                  // statistical energy uncertainty (log-normal)
    logE_sys_unc,                   // systematic energy uncertainty, global systematic shift
    Emin, Emax,     // energy range for truncated lognormal likelihood
    mean_lnA_true,                  // mean lnA per energy bin per source
    var_lnA_true,                   // variance of lnA per energy bin per source
    lnA_logE_grid,                      // grid of energy bins in lnA
    nu_lnAs,                       // latent variable for sampling lnA
    omega_det,                      // detected directions
    beta_egmf                // EGMF spread parameter
  );

  // --- Poisson normalization ---
  target += -Nex;
}

generated quantities {

    real log10_beta_egmf = log10(beta_egmf);

    // generate the log-likelihood of the event to get the
    // association probability as well
    array[N] real loglik_event;
    array[N] real loglik_event_energy;
    array[N] real loglik_event_spatial;

    for (i in 1:N) {
      loglik_event[i] = log(F0);
      int Ebin_idx = binary_search(logE_true[i], lnA_logE_grid);
      real mean_lnA = mean_lnA_true[Ebin_idx];
      real var_lnA = var_lnA_true[Ebin_idx];
      real Zsrc = 0.5 * exp(mean_lnA + sqrt(var_lnA) * nu_lnAs[i]);
      real Rtrue = exp(logE_true[i]) / Zsrc;

      loglik_event[i] += energy_spectrum_lpdf(logE_true[i] | alpha_bg, logE_grid, alpha_grid, log(espect_mfs));
      loglik_event[i] += truncated_lognormal_lpdf(Edet[i] | logE_true[i] + logE_sys_unc, logE_stat_unc, Emin, Emax);
      loglik_event_energy[i] = loglik_event[i];


      loglik_event[i] += -log(4*pi());
      loglik_event_spatial[i] = -log(4*pi());
    }
        
}