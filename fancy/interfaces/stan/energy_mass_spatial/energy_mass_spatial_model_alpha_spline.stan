/**
 * Energy + lnA model -- "knots method" variant.
 *
 * Identical to energy_mass_spatial_model.stan EXCEPT that every
 * interpolation over alpha_grid (log_espect_at_alpha, mulnA_mfs/varlnA_mfs,
 * esrc_ratio_grid) uses a natural cubic spline instead of linear
 * interpolation, evaluated via a precomputed second-derivative matrix
 * (alpha_spline_matrix, built once in Python from alpha_grid alone -- see
 * fancy.utils.helpers.natural_cubic_spline_matrix). This targets the
 * grid-funneling pathology where alpha posteriors snapped to alpha_grid
 * nodes under linear interpolation (see project_alpha_grid_funneling memory).
 *
 * Interpolation over logE_grid (energy_spectrum_lpdf) and lnA_logE_grid
 * (mean/var lnA per event) is UNCHANGED (still linear via interpolate()) --
 * only the alpha_grid axis is splined here, since that's where the
 * funneling was diagnosed.
 *
 * Deliberately kept as a separate file from energy_mass_spatial_model.stan
 * so the default linear-interpolation model stays untouched and
 * reproducible, and the two can be run side by side on the same simulated
 * dataset for direct comparison.
 *
 * @author Keito Watanabe
 * @date September 2026
 */

functions {
    #include /utils_alpha_spline.stan

     /**
    * sample from the energy spectrum at Earth
    * @param E energy in EeV
    * @param log_en_grid : grid of log(E) values for interpolation
    * @param espect_mfs : mass fraction weighted energy spectrum at Earth. Shape in (Nalphas, NEs)
    * @return log probability density of the energy spectrum at Earth
    *
    * NB: still linear interpolation over logE_grid -- only the alpha_grid
    * axis is splined in this model variant.
    */
    real energy_spectrum_lpdf(real logE,
                          vector log_en_grid,
                          vector log_espect_at_alpha) {
        return interpolate(log_en_grid, log_espect_at_alpha, logE);
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
    * approximation for kappa > 100. -> 
    * Now taken care in utils/log_sinh()
    */

    real fik_lpdf(vector v, vector mu, real kappa, real kappa_d) {
      real inner = sqrt(dot_self((kappa_d * v) + (kappa * mu)));
      return log(kappa) + log(kappa_d) - log(4 * pi())
            - log_sinh(kappa) - log_sinh(kappa_d)
            + log_sinh(inner) - log(inner);
    }

// --- per-event likelihood chunk for reduce_sum ---
  real event_likelihood_chunk(
      array [] int slice_i,         // indices of events in this chunk
      int start, int end,           // range of indices in this chunk
      vector alphas,                // alphas per source
      vector F,                     // fluxes per source
      array [] real alpha_grid,     // grid of alpha for energy interpolation
      vector logE_grid,    // grid of log10(E) for energy interpolation
      array [] vector log_espect_at_alpha,   // energy spectra at Earth per source
      vector logE_true,                 // latent true energies
      vector Edet,                  // detected energies
      real logE_stat_unc,           // statistical energy uncertainty (log-normal)
      real logE_sys_unc,            // systematic energy uncertainty, global systematic shift
      real Emin, real Emax,          // energy range for truncated lognormal likelihood
      vector mean_lnA_true,         // mean lnA per energy bin per source
      vector var_lnA_true,          // variance of lnA per energy bin per source
      vector lnA_logE_grid_vec,   // grid of energy bins for finding the mean and variance of lnA per energy
      vector nu_lnAs,              // latent variable for sampling Zsrcs
      array [] vector omega_det,    // detected directions
      vector kappa_ds,              // deflection parameters including GMF and arrival direction uncertainty
      array [] vector omega_src,    // source directions
      real beta_egmf,         // EGMF spread parameter
      vector D,                     // source distances
      int Nsrcs,                     // number of sources
      vector exp_factors          // exposure correction factors per event
  ) {
      real lp_chunk = 0.0;  // log likelihood for this chunk
      int len = size(slice_i); // number of events in this chunk
        // loop over events in this chunk
      for (n in 1:len) {
          // the index of the event
          int i = slice_i[n];

          // log likelihood per source + isotropic background
          vector[Nsrcs+1] lp_i = log(F);

          // pre-computation for the rigidity
          // computing the mean and variance of lnA through interpolation
          real mean_lnA = interpolate(lnA_logE_grid_vec, mean_lnA_true, logE_true[i]);
          real var_lnA  = interpolate(lnA_logE_grid_vec, var_lnA_true,  logE_true[i]);

          // // calculating the true charge and rigidity
          real Zsrc = 0.5 * exp(mean_lnA + sqrt(var_lnA) * nu_lnAs[i]);
          real Rtrue = exp(logE_true[i]) / Zsrc;
          
          // iterate over sources + isotropic background
          for (k in 1:Nsrcs+1) {
              // energy sampling (spectrum at Earth + truncated lognormal -> now after the k-loop)
              lp_i[k] += energy_spectrum_lpdf(logE_true[i] | logE_grid, log_espect_at_alpha[k]);

              // spatial likelihood (EGMF and GMF deflections for source, isotropic for background)

              if (k <= Nsrcs) {
                /* GMF and EGMF deflections */
                real kappa_egmf = get_kappa(Rtrue, beta_egmf, D[k]/10.0);
                lp_i[k] += fik_lpdf(
                  omega_det[i] | omega_src[k],
                  kappa_egmf,
                  kappa_ds[i]
                );
              }
              
              else {
                /* isotropic background */
                lp_i[k] += -log(4*pi());
                
              }
                  
          }

          // apply exposure correction
          lp_i += log(exp_factors[i]);

          // log-sum-exp over sources + isotropic background
          lp_chunk += log_sum_exp(lp_i);

          // we add the detector uncertainties of the energies here. 
          // since it doesnt depend on the distance k
          lp_chunk += truncated_lognormal_lpdf(Edet[i] | logE_true[i] + logE_sys_unc, logE_stat_unc, Emin, Emax);
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
    vector[N] exposure_factor; /* exposure correction factors per event */
    vector[N] kappa_ds; /* deflection parameters, including GMF deflections + arrival direction uncertainty */
    array[NEbins] real lnA_logE_grid_det;
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
    int<lower=0> NEbins_lnAgrid;
    array[NEbins_lnAgrid] real lnA_logE_grid; /* grid of energy bins for lnA model */
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEbins_lnAgrid] mean_lnA_grid;
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEbins_lnAgrid] var_lnA_grid;


    /* detector */
    real<lower=0> Emin;
    real Emax;

    /* statitistical uncertainty */
    real<lower=0> logE_stat_unc;
    vector[NEbins] mean_lnA_stat_unc;
    vector[NEbins] var_lnA_stat_unc;

    // // /* systematic uncertainties, as global shifts */
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
    int<lower=1> grain_size; /* for reduce_sum, generally N / (4 * ncores) is a good estimate */

    /* "knots method": precomputed natural cubic spline second-derivative
       matrix for alpha_grid (see fancy.utils.helpers.natural_cubic_spline_matrix).
       Depends only on alpha_grid's knot locations, computed once in Python;
       y2 = alpha_spline_matrix * y gives the spline's second derivatives at
       the knots for any y sampled on alpha_grid this iteration. */
    matrix[Nalphas, Nalphas] alpha_spline_matrix;
}

transformed data {
  // --- constants ---
  real logEmin = log(Emin);
  real logEmax = log(Emax);

  real alpha_min = min(alpha_grid);
  real alpha_max = max(alpha_grid);
  // real alpha_max = 3.0;

  real beta_egmf_min = pow(10.0, min(log10_beta_egmf_grid));
  real beta_egmf_max = pow(10.0, max(log10_beta_egmf_grid));

  // --- distance conversion once ---
  vector[Nsrcs] D_flux = D * 3.08567758e19;

  // --- precompute 1D arrays for interpolation ---
  vector[Nalphas] alpha_grid_vec = to_vector(alpha_grid);
  vector[NEs] logE_grid_vec = to_vector(logE_grid);
  vector[NEbins_lnAgrid] lnA_logE_grid_vec = to_vector(lnA_logE_grid);

  // number of events for reduced sum
    array[N] int event_ids;
    for (n in 1:N) {event_ids[n] = n;}
}

parameters {

    /* spectral information */
    vector <lower=alpha_min, upper=alpha_max>  [Nsrcs+1] alphas;

    /* mass fractions (in future, 2D structure with sources) */
    array[Nsrcs+1] simplex[NAsrcs] mass_fracs;

    /* flux fraction per source (+ BG) */
    simplex[Nsrcs+1] flux_frac;        

    /* total flux AT EARTH */
    real log10_Ftot;

    /* EGMF spread parameter, in nG Mpc^1/2 */
    real<lower=beta_egmf_min, upper=beta_egmf_max> beta_egmf;

    /* latent parameters for energy */
    vector <lower=logEmin, upper=logEmax>[N] logE_true;

    vector[N] nu_lnAs; /* latent variable for sampling lnA (Zsrcs) */
    /* global systematic uncertainties (shift) for lnA */
    // real mean_lnA_sys_unc;
    // real var_lnA_sys_unc;

}

transformed parameters {
  // --- only quantities needed downstream ---
  vector[Nsrcs+1] F;
  array[Nsrcs+1] matrix [Nalphas, NEs] espect_mfs;
  array[Nsrcs+1] matrix [Nalphas, NEbins_lnAgrid] mulnA_mfs;
  array[Nsrcs+1] matrix [Nalphas, NEbins_lnAgrid] varlnA_mfs;
  vector[Nsrcs+1] wexp_earths;
  array[Nsrcs+1] vector[NEs] log_espect_at_alpha;

  // initialise
  F = rep_vector(0.0, Nsrcs+1);
  espect_mfs = rep_array(rep_matrix(0.0, Nalphas, NEs), Nsrcs+1);
  mulnA_mfs = rep_array(rep_matrix(0.0, Nalphas, NEbins_lnAgrid), Nsrcs+1);
  varlnA_mfs = rep_array(rep_matrix(0.0, Nalphas, NEbins_lnAgrid), Nsrcs+1);
  wexp_earths = rep_vector(0.0, Nsrcs+1);

  // real logE_sys_unc = 0.0;
  // vector[NEbins] mean_lnA_sys_unc = rep_vector(0.0, NEbins);
  // vector[NEbins] var_lnA_sys_unc = rep_vector(0.0, NEbins);

  for (k in 1:Nsrcs+1) {
    // flux at Earth
    F[k] = pow(10.0, log10_Ftot) * flux_frac[k];

    // accumulate spectra and weights
    for (j in 1:NAsrcs) {
      espect_mfs[k] += mass_fracs[k][j] * earth_spectrum_grid[k,j];
      mulnA_mfs[k][,] += mass_fracs[k][j] * mean_lnA_grid[k,j];
      varlnA_mfs[k][,] += mass_fracs[k][j] * var_lnA_grid[k,j];

      wexp_earths[k] += mass_fracs[k][j] * exp(interp2d_alpha_spline(
        alphas[k], log10(beta_egmf),
        alpha_grid_vec, alpha_grid, log10_beta_egmf_grid,
        to_array_2d(log_wexp_earth_grid[k,j]),
        alpha_spline_matrix
      ));

    }

    // since the alpha is only source dependent, and not dependent
    // per event, we can already just interpolate it here and store the
    // spectrum per "distance".
    // this is fine since we are already in the transformed_parameters block.
    //
    // "knots method": natural cubic spline instead of linear interpolation
    // over alpha_grid. log(espect_mfs[k]) is (Nalphas, NEs); batch the
    // spline coefficient solve once across all NEs columns rather than
    // recomputing alpha_spline_matrix * y inside this loop.
    {
      matrix[Nalphas, NEs] log_espect_mfs_k = log(espect_mfs[k]);
      log_espect_at_alpha[k] = interpolate_spline_batch(
        alpha_grid_vec, log_espect_mfs_k, alpha_spline_matrix, alphas[k]
      );
    }
  }

  vector[Nsrcs+1] Nex_arr = F .* wexp_earths;
  real Nex = sum(Nex_arr);

  // mean and sigma lnA values
  vector[NEbins_lnAgrid] mean_lnA_true = rep_vector(0.0, NEbins_lnAgrid);
  vector[NEbins_lnAgrid] var_lnA_true = rep_vector(0.0, NEbins_lnAgrid);

   // --- binned lnA likelihood ---
  // "knots method": batch the spline coefficient solve once per source k
  // across all NEbins columns, instead of recomputing it per (l, k) pair.
  {
    array[Nsrcs+1] vector[NEbins_lnAgrid] mulnA_spline_k;
    array[Nsrcs+1] vector[NEbins_lnAgrid] varlnA_spline_k;
    for (k in 1:Nsrcs+1) {
      mulnA_spline_k[k] = interpolate_spline_batch(
        alpha_grid_vec, mulnA_mfs[k], alpha_spline_matrix, alphas[k]
      );
      varlnA_spline_k[k] = interpolate_spline_batch(
        alpha_grid_vec, varlnA_mfs[k], alpha_spline_matrix, alphas[k]
      );
    }
    for (l in 1:NEbins_lnAgrid) {
      for (k in 1:Nsrcs+1) {
          /* calculate the mean and variance of lnA for each energy bin */
          mean_lnA_true[l] += Nex_arr[k] * mulnA_spline_k[k][l] / Nex;
          var_lnA_true[l] += Nex_arr[k] * varlnA_spline_k[k][l] / Nex;
      }
    }
  }
}

model {
  // --- priors ---
  // spectral indices : normal distribution
  alphas ~ normal(0.0, 2.0);

  // mass fractions : Dirichlet distribution per source
  for (k in 1:Nsrcs+1) {
    mass_fracs[k] ~ dirichlet(rep_vector(2.0, NAsrcs));
  }

  // flux fraction weights: Dirichlet-like prior
  flux_frac ~ dirichlet(rep_vector(2.0, Nsrcs+1));

  // total flux : normal distribution in log10
  log10_Ftot ~ normal(-1.0, 3.0);

  // magnetic spread: normal distribution
  // prior uncertainty is 10 since we do not have enough
  // handle on it.
  beta_egmf ~ normal(0.0, 10.0);

  // latent variables for lnA : normal distribution
  nu_lnAs ~ normal(0.0, 1.0);

  // global systematic uncertainties (shift) for lnA : normal distribution
  // mean_lnA_sys_unc ~ normal(0.0, 1.0);
  // var_lnA_sys_unc ~ normal(0.0, 1.0);

   // --- binned lnA likelihood ---
  for (l in 1:NEbins) {
    real mean_lnA_data = interpolate(lnA_logE_grid_vec, mean_lnA_true, lnA_logE_grid_det[l]);
    real var_lnA_data = interpolate(lnA_logE_grid_vec, var_lnA_true, lnA_logE_grid_det[l]);
    target += left_truncated_normal_lpdf(mean_lnA_det[l] |
              mean_lnA_data + mean_lnA_sys_unc,
              mean_lnA_stat_unc[l], 0.0);
    target += left_truncated_normal_lpdf(var_lnA_det[l] |
              var_lnA_data + var_lnA_sys_unc,
              var_lnA_stat_unc[l], -2.0);
  }

  // --- parallelized unbinned likelihood ---
  target += reduce_sum(
    event_likelihood_chunk,         // the per-chunk unbinned likelihood function
    event_ids,                      // the data to be sliced and passed to each chunk
    grain_size,                     // tuning parameter for the chunk size
    alphas,                         // spectral indices
    F,                              // fluxes per source
    alpha_grid,                     // alpha grid for interpolation        
    logE_grid_vec,                      // log10(E) grid for interpolation  
    log_espect_at_alpha,                 // log(energy spectra) at Earth per source (also per alpha)
    logE_true,                      // latent true energies
    Edet,                           // detected energies
    logE_stat_unc,                  // statistical energy uncertainty (log-normal)
    logE_sys_unc,                   // systematic energy uncertainty, global systematic shift
    Emin, Emax,     // energy range for truncated lognormal likelihood
    mean_lnA_true,                  // mean lnA per energy bin per source
    var_lnA_true,                   // variance of lnA per energy bin per source
    lnA_logE_grid_vec,                      // grid of energy bins in lnA
    nu_lnAs,                       // latent variable for sampling lnA
    omega_det,                      // detected directions
    kappa_ds,                       // deflection parameters including GMF and arrival direction uncertainty
    omega_src,                      // source directions
    beta_egmf,                // EGMF spread parameter
    D,                              // source distances
    Nsrcs,                           // number of sources
    exposure_factor               // exposure correction factors per event
  );

  // --- Poisson normalization ---
  target += -Nex;
}

generated quantities {
    real Nex_src = sum(Nex_arr[1:Nsrcs]);
    real Nex_bg = Nex_arr[Nsrcs+1];

    real src_frac = Nex_src / Nex;

    real log10_beta_egmf = log10(beta_egmf);

    vector[Nsrcs] Lsrcs;

    for (k in 1:Nsrcs) {
      real esrc_ratios = 0.0;
      real wexp_src = 0.0;

      for (j in 1:NAsrcs) {
        vector[Nalphas] esrc_ratio_y2 = alpha_spline_matrix * esrc_ratio_grid[k,j];
        esrc_ratios += mass_fracs[k][j] * interpolate_spline(
          alpha_grid_vec, esrc_ratio_grid[k,j], esrc_ratio_y2, alphas[k]
        );
        wexp_src += mass_fracs[k][j] * exp(interp2d_alpha_spline(
          alphas[k], log10_beta_egmf,
          alpha_grid_vec, alpha_grid, log10_beta_egmf_grid,
          to_array_2d(log_wexp_src_grid[k,j]),
          alpha_spline_matrix
        ));
      }

      Lsrcs[k] = F[k] * wexp_src / wexp_earths[k] * 4*pi() * square(D_flux[k]) * esrc_ratios;
    
    }

    vector[Nsrcs] log10_Lsrcs = log10(Lsrcs);

    

    // generate the log-likelihood of the event to get the
    // association probability as well
    array[N] vector[Nsrcs+1] loglik_event;
    array[N] vector[Nsrcs+1] loglik_event_energy;
    array[N] vector[Nsrcs+1] loglik_event_spatial;
    array[NEbins] vector[Nsrcs+1] loglik_event_mass;

    vector[N] kappa_egmfs;
    vector[N] rigidities;
    for (i in 1:N) {
      loglik_event[i] = log(F);

      real mean_lnA = interpolate(lnA_logE_grid_vec, mean_lnA_true, logE_true[i]);
      real var_lnA = interpolate(lnA_logE_grid_vec, var_lnA_true, logE_true[i]);
      real Zsrc = 0.5 * exp(mean_lnA + sqrt(var_lnA) * nu_lnAs[i]);
      real Rtrue = exp(logE_true[i]) / Zsrc;

      rigidities[i] = Rtrue;

      for (k in 1:(Nsrcs+1)) {
        loglik_event[i,k] += energy_spectrum_lpdf(logE_true[i] | logE_grid_vec, log_espect_at_alpha[k]);
        loglik_event[i,k] += truncated_lognormal_lpdf(Edet[i] | logE_true[i] + logE_sys_unc, logE_stat_unc, Emin, Emax);
        loglik_event_energy[i,k] = loglik_event[i,k];
        if (k <= Nsrcs) {
          real kappa_egmf = get_kappa(Rtrue, beta_egmf, D[k]/10.0);
          loglik_event[i,k] += fik_lpdf(omega_det[i]|omega_src[k], kappa_egmf, kappa_ds[i]);
          loglik_event_spatial[i,k] = fik_lpdf(omega_det[i]|omega_src[k], kappa_egmf, kappa_ds[i]);

          kappa_egmfs[i] = kappa_egmf;
        } else {
          loglik_event[i,k] += -log(4*pi());
          loglik_event_spatial[i,k] = -log(4*pi());
        }
      }
    }

    for (l in 1:NEbins) {

        real mean_lnA_data = interpolate(lnA_logE_grid_vec, mean_lnA_true, lnA_logE_grid_det[l]);
        real var_lnA_data = interpolate(lnA_logE_grid_vec, var_lnA_true, lnA_logE_grid_det[l]);

        // binned (per detector lnA bin) term: identical for every source
        // column, since the binned likelihood is not split by source
        loglik_event_mass[l] = rep_vector(0.0, Nsrcs+1);
        loglik_event_mass[l] += left_truncated_normal_lpdf(mean_lnA_det[l] |
            mean_lnA_data + mean_lnA_sys_unc,
            mean_lnA_stat_unc[l], 0.0);
        loglik_event_mass[l] += left_truncated_normal_lpdf(var_lnA_det[l] |
            var_lnA_data + var_lnA_sys_unc,
            var_lnA_stat_unc[l], -2.0);
    }
        
}