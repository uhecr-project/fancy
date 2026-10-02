/**
 * Energy-only model -- "knots method" variant.
 *
 * Identical to energy_model.stan EXCEPT that every interpolation over
 * alpha_grid (log_espect_at_alpha, log_wexp_earth_grid, log_wexp_src_grid,
 * esrc_ratio_grid) uses a natural cubic spline instead of linear
 * interpolation, evaluated via a precomputed second-derivative matrix
 * (alpha_spline_matrix, built once in Python from alpha_grid alone -- see
 * fancy.utils.helpers.natural_cubic_spline_matrix). This targets the
 * grid-funneling pathology where alpha posteriors snapped to alpha_grid
 * nodes under linear interpolation.
 *
 * Interpolation over logE_grid (energy_spectrum_lpdf) is UNCHANGED (still
 * linear via interpolate()) -- only the alpha_grid axis is splined here.
 *
 * Ported from energy_mass_spatial_model_alpha_spline.stan. Deliberately kept
 * as a separate file from energy_model.stan so the default
 * linear-interpolation model stays untouched and reproducible.
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
      real Emin, real Emax,          // energy range for truncated lognormal likelihoo
      int Nsrcs
  ) {
      real lp_chunk = 0.0;  // log likelihood for this chunk
      int len = size(slice_i); // number of events in this chunk
        // loop over events in this chunk
      for (n in 1:len) {
          // the index of the event
          int i = slice_i[n];

          // log likelihood per source + isotropic background
          vector[Nsrcs+1] lp_i = log(F);
          
          // iterate over sources + isotropic background
          for (k in 1:Nsrcs+1) {
              // energy sampling (spectrum at Earth + truncated lognormal -> now after the k-loop)
              lp_i[k] += energy_spectrum_lpdf(logE_true[i] | logE_grid, log_espect_at_alpha[k]);                  
          }

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

    /* uhecr */
    int<lower=0> N;
    int<lower=0> NEbins;  /* number of energy bins for lnA measurements */
    vector[N] Edet;

    /* model */
    int<lower=0> Nalphas;   /* number of alpha grid points */
    int<lower=0> NEs;    /* number of energy grid points in EeV */
    int<lower=0> NAsrcs;  /* number of source masses */
    array[Nalphas] real alpha_grid;
    array[NEs] real logE_grid;
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEs] earth_spectrum_grid;


    /* detector */
    real<lower=0> Emin;
    real Emax;

    /* statitistical uncertainty */
    real<lower=0> logE_stat_unc;

    /* systematic uncertainties, as global shifts */
    real logE_sys_unc;

    /* Nex */
    /* grid of weighted exposures at earth and source */
    array[Nsrcs+1, NAsrcs] vector[Nalphas] log_wexp_earth_grid;
    array[Nsrcs, NAsrcs] vector[Nalphas] log_wexp_src_grid;
    array[Nsrcs, NAsrcs] vector[Nalphas] esrc_ratio_grid;

    /* "knots method": precomputed natural cubic spline second-derivative
       matrix for alpha_grid (see fancy.utils.helpers.natural_cubic_spline_matrix).
       Depends only on alpha_grid's knot locations, computed once in Python;
       y2 = alpha_spline_matrix * y gives the spline's second derivatives at
       the knots for any y sampled on alpha_grid this iteration. */
    matrix[Nalphas, Nalphas] alpha_spline_matrix;

    /* computation parameters */
    int<lower=1> grain_size; /* for reduce_sum, generally N / (4 * ncores) is a good estimate */
}

transformed data {
    // --- constants ---
  real logEmin = log(Emin);
  real logEmax = log(Emax);

  real alpha_min = min(alpha_grid);
  real alpha_max = max(alpha_grid);
  // real alpha_max = 3.0;

  // --- distance conversion once ---
  vector[Nsrcs] D_flux = D * 3.08567758e19;

  // --- precompute 1D arrays for interpolation ---
  vector[Nalphas] alpha_grid_vec = to_vector(alpha_grid);
  vector[NEs] logE_grid_vec = to_vector(logE_grid);

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

    /* latent parameters for energy */
    vector <lower=logEmin, upper=logEmax>[N] logE_true;

}

transformed parameters {

    // --- only quantities needed downstream ---
    vector[Nsrcs+1] F;
    array[Nsrcs+1] matrix [Nalphas, NEs] espect_mfs;
    vector[Nsrcs+1] wexp_earths;
    array[Nsrcs+1] vector[NEs] log_espect_at_alpha;

    // initialise
    F = rep_vector(0.0, Nsrcs+1);
    espect_mfs = rep_array(rep_matrix(0.0, Nalphas, NEs), Nsrcs+1);
    wexp_earths = rep_vector(0.0, Nsrcs+1);

    for (k in 1:Nsrcs+1) {
        // flux at Earth
        F[k] = pow(10.0, log10_Ftot) * flux_frac[k];

        // accumulate spectra and weights
        for (j in 1:NAsrcs) {
        espect_mfs[k] += mass_fracs[k][j] * earth_spectrum_grid[k,j];

        // "knots method": 1D natural cubic spline over alpha_grid
        // (energy_only has no beta_egmf axis, so no interp2d here).
        vector[Nalphas] log_wexp_earth_y2 = alpha_spline_matrix * log_wexp_earth_grid[k,j];
        wexp_earths[k] += mass_fracs[k][j] * exp(interpolate_spline(
            alpha_grid_vec,
            log_wexp_earth_grid[k,j],
            log_wexp_earth_y2,
            alphas[k]
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

}

model {

  // --- priors ---
  // spectral indices : normal distribution
  alphas ~ normal(0.0, 2.0);

  // mass fractions : Dirichlet distribution per source
  for (k in 1:Nsrcs+1) {
    mass_fracs[k] ~ dirichlet([2.0, 2.0, 2.0, 2.0]);
  }

  // flux fraction weights: Dirichlet-like prior
  flux_frac ~ dirichlet(rep_vector(2.0, Nsrcs+1));

  // total flux : normal distribution in log10
  log10_Ftot ~ normal(-1.0, 3.0);

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
    Nsrcs
  );

  // --- Poisson normalization ---
  target += -Nex;

}


generated quantities {
    real Nex_src = sum(Nex_arr[1:Nsrcs]);
    real Nex_bg = Nex_arr[Nsrcs+1];

    real src_frac = Nex_src / Nex;

    vector[Nsrcs] Lsrcs;

    for (k in 1:Nsrcs) {
      real esrc_ratios = 0.0;
      real wexp_src = 0.0;

      for (j in 1:NAsrcs) {
        vector[Nalphas] esrc_ratio_y2 = alpha_spline_matrix * esrc_ratio_grid[k,j];
        esrc_ratios += mass_fracs[k][j] * interpolate_spline(
          alpha_grid_vec, esrc_ratio_grid[k,j], esrc_ratio_y2, alphas[k]
        );
        vector[Nalphas] log_wexp_src_y2 = alpha_spline_matrix * log_wexp_src_grid[k,j];
        wexp_src += mass_fracs[k][j] * exp(interpolate_spline(
          alpha_grid_vec,
          log_wexp_src_grid[k,j],
          log_wexp_src_y2,
          alphas[k]
        ));
      }

      Lsrcs[k] = F[k] * wexp_src / wexp_earths[k] * 4*pi() * square(D_flux[k]) * esrc_ratios;
    
    }

    vector[Nsrcs] log10_Lsrcs = log10(Lsrcs);

    

    // generate the log-likelihood of the event to get the
    // association probability as well
    array[N] vector[Nsrcs+1] loglik_event;
    array[N] vector[Nsrcs+1] loglik_event_energy;

    for (i in 1:N) {
      loglik_event[i] = log(F);

      for (k in 1:(Nsrcs+1)) {
        loglik_event[i,k] += energy_spectrum_lpdf(logE_true[i] | logE_grid_vec, log_espect_at_alpha[k]);
        loglik_event[i,k] += truncated_lognormal_lpdf(Edet[i] | logE_true[i] + logE_sys_unc, logE_stat_unc, Emin, Emax);
        loglik_event_energy[i,k] = loglik_event[i,k];
      }
    }
        
}