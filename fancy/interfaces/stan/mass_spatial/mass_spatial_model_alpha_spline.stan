/**
 * Mass + spatial model -- "knots method" variant.
 *
 * Identical to mass_spatial_model.stan EXCEPT that every interpolation over
 * alpha_grid (mulnA_mfs/varlnA_mfs, esrc_ratio_grid, and the alpha axis of
 * log_wexp_earth_grid / log_wexp_src_grid via interp2d_alpha_spline, plus
 * the alpha axis of the (unused) energy_spectrum_lpdf helper) uses a natural
 * cubic spline instead of linear interpolation, evaluated via a precomputed
 * second-derivative matrix (alpha_spline_matrix, built once in Python from
 * alpha_grid alone -- see fancy.utils.helpers.natural_cubic_spline_matrix).
 * This targets the grid-funneling pathology where alpha posteriors snapped
 * to alpha_grid nodes under linear interpolation.
 *
 * Interpolation over log10_beta_egmf_grid (y-axis of the 2D exposure
 * interpolation) is UNCHANGED (still linear) -- only the alpha_grid axis is
 * splined here.
 *
 * Ported from energy_mass_spatial_model_alpha_spline.stan. Deliberately kept
 * as a separate file from mass_spatial_model.stan so the default
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
    * @param alpha spectral index
    * @param log_en_grid : grid of log(E) values for interpolation
    * @param alph_grid_vec : grid of alpha values for interpolation (as vector)
    * @param alph_grid : grid of alpha values for interpolation
    * @param espect_mfs : mass fraction weighted energy spectrum at Earth. Shape in (Nalphas, NEs)
    * @param alpha_spline_matrix : precomputed natural cubic spline matrix for alph_grid
    * @return log probability density of the energy spectrum at Earth
    *
    * NB: natural cubic spline along the alpha axis, still linear along
    * log_en_grid (interp2d_alpha_spline). Not called anywhere in this model
    * (kept only to mirror mass_spatial_model.stan).
    */
    real energy_spectrum_lpdf(real logE, real alpha, array [] real log_en_grid, vector alph_grid_vec, array [] real alph_grid, matrix log_espect_mfs, matrix alpha_spline_matrix) {
        return interp2d_alpha_spline(alpha, logE, alph_grid_vec, alph_grid, log_en_grid, to_array_2d(log_espect_mfs), alpha_spline_matrix);
    }

    /**
    * Calculate the deflection parameter of the vMF distribution.
    * Derived from Eq. 4 and Eq. 9 in Capel & Mortlock, 2018.
    * @param R rigidity in EV
    * @param mag_beta magnetic spread in nG Mpc^(1/2)
    * @param D distance in Mpc / 10
    * @return deflection parameter kappa, capped at 1e5
    *
    * The cap addresses a saturation in fik_lpdf: once kappa >> kappa_d,
    * log_sinh(kappa) and log_sinh(inner) both grow without bound and
    * nearly cancel, leaving a residual that no longer depends on kappa
    * except through a slowly-growing log(kappa) term -- i.e. the
    * likelihood keeps preferring smaller mag_beta (larger kappa) with no
    * remaining directional constraint once kappa exceeds ~1e5 (confirmed
    * empirically: the fik_lpdf directional term is numerically identical
    * across kappa in [1e5, 1e11] for realistic kappa_d ~ 1-1e3). Without
    * this cap this produced a persistent SBC miscalibration (beta_egmf
    * posteriors landing 5-200x above the true value across trials).
    */
    real get_kappa(real R, real mag_beta, real D) {

    return fmin(7552.0 * inv_square(2.3 * inv(R / 50.0) * mag_beta * sqrt(D)), 1e5);
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
      vector F,                     // fluxes per source
      array [] vector omega_det,    // detected directions
      vector kappa_ds,              // deflection parameters including GMF and arrival direction uncertainty
      array [] vector omega_src,    // source directions
      real kappa_egmf,         // EGMF spread parameter
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

          // iterate over sources + isotropic background
          for (k in 1:Nsrcs+1) {

              // spatial likelihood (EGMF and GMF deflections for source, isotropic for background)
              if (k <= Nsrcs) {
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
    vector[N] kappa_ds; /* deflection parameters, including GMF deflections + arrival direction uncertainty (unused, kept for compatibility) */
    array[NEbins] real mean_lnA_det;
    array[NEbins] real var_lnA_det;

    /* model */
    int<lower=0> Nalphas;   /* number of alpha grid points */
    int<lower=0> NAsrcs;  /* number of source masses */
    array[Nalphas] real alpha_grid;
    /* for lnA, binned via detected energies */
    array[NEbins] real lnA_logE_grid; /* grid of energy bins for lnA model */
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEbins] mean_lnA_grid;
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEbins] var_lnA_grid;

    /* statitistical uncertainty */
    vector[NEbins] mean_lnA_stat_unc;
    vector[NEbins] var_lnA_stat_unc;

    /* systematic uncertainties, as global shifts */
    real mean_lnA_sys_unc;
    real var_lnA_sys_unc;

    /* grid of effective exposures at earth vs kappa (natural log); last row is the background */
    int Nkappas;
    vector[Nkappas] log_kappa_grid;
    array[Nsrcs+1] vector[Nkappas] log_wexp_earth_grid;

    /* "knots method": precomputed natural cubic spline second-derivative
       matrix for alpha_grid (see fancy.utils.helpers.natural_cubic_spline_matrix).
       Depends only on alpha_grid's knot locations, computed once in Python;
       y2 = alpha_spline_matrix * y gives the spline's second derivatives at
       the knots for any y sampled on alpha_grid this iteration. */
    matrix[Nalphas, Nalphas] alpha_spline_matrix;

    /* computation parameters */
    int<lower=1> grain_size; /* for reduce_sum, generatlly N / (4 * ncores) is a good estimate */
}

transformed data {

  real alpha_min = min(alpha_grid);
  real alpha_max = max(alpha_grid);
  // real alpha_max = 3.0;

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
    vector <lower=alpha_min, upper=alpha_max>  [Nsrcs+1] alphas;

    /* mass fractions (in future, 2D structure with sources) */
    array[Nsrcs+1] simplex[NAsrcs] mass_fracs;

    /* flux fraction per source (+ BG) */
    simplex[Nsrcs+1] flux_frac;

    /* total flux AT EARTH */
    real log10_Ftot;

    real kappa_egmf;

}

transformed parameters {

  // --- only quantities needed downstream ---
  vector[Nsrcs+1] F;
  array[Nsrcs+1] matrix [Nalphas, NEbins] mulnA_mfs;
  array[Nsrcs+1] matrix [Nalphas, NEbins] varlnA_mfs;
  vector[Nsrcs+1] wexp_earths;

  // initialise
  F = rep_vector(0.0, Nsrcs+1);
  mulnA_mfs = rep_array(rep_matrix(0.0, Nalphas, NEbins), Nsrcs+1);
  varlnA_mfs = rep_array(rep_matrix(0.0, Nalphas, NEbins), Nsrcs+1);
  wexp_earths = rep_vector(0.0, Nsrcs+1);

  for (k in 1:Nsrcs+1) {
    // flux at Earth
    F[k] = pow(10.0, log10_Ftot) * flux_frac[k];

    // accumulate spectra and weights
    for (j in 1:NAsrcs) {
      mulnA_mfs[k][,] += mass_fracs[k][j] * mean_lnA_grid[k,j];
      varlnA_mfs[k][,] += mass_fracs[k][j] * var_lnA_grid[k,j];
    }

    // effective exposure at Earth, interpolated (linearly in log-log) at the shared kappa_egmf
    wexp_earths[k] = exp(interpolate(
      log_kappa_grid, log_wexp_earth_grid[k], log(kappa_egmf)
    ));
  }

  vector[Nsrcs+1] Nex_arr = F .* wexp_earths;
  real Nex = sum(Nex_arr);

  // mean and sigma lnA values
  vector[NEbins] mean_lnA_true = rep_vector(0.0, NEbins);
  vector[NEbins] var_lnA_true = rep_vector(0.0, NEbins);

   // --- binned lnA likelihood ---
  // "knots method": batch the spline coefficient solve once per source k
  // across all NEbins columns, instead of recomputing it per (l, k) pair.
  {
    array[Nsrcs+1] vector[NEbins] mulnA_spline_k;
    array[Nsrcs+1] vector[NEbins] varlnA_spline_k;
    for (k in 1:Nsrcs+1) {
      mulnA_spline_k[k] = interpolate_spline_batch(
        alpha_grid_vec, mulnA_mfs[k], alpha_spline_matrix, alphas[k]
      );
      varlnA_spline_k[k] = interpolate_spline_batch(
        alpha_grid_vec, varlnA_mfs[k], alpha_spline_matrix, alphas[k]
      );
    }
    for (l in 1:NEbins) {
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

  // mean deflection parameter from EGMF
  kappa_egmf ~ lognormal(log(100.), 1.);

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
    F,                              // fluxes per source
    omega_det,                      // detected directions
    kappa_ds,                       // deflection parameters including GMF and arrival direction uncertainty
    omega_src,                      // source directions
    kappa_egmf,
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

    // generate the log-likelihood of the event to get the
    // association probability as well
    array[N] vector[Nsrcs+1] loglik_event;
    array[N] vector[Nsrcs+1] loglik_event_spatial;
    for (i in 1:N) {
      loglik_event[i] = log(F);

      for (k in 1:(Nsrcs+1)) {
        if (k <= Nsrcs) {
          loglik_event[i,k] += fik_lpdf(omega_det[i]|omega_src[k], kappa_egmf, kappa_ds[i]);
          loglik_event_spatial[i,k] = fik_lpdf(omega_det[i]|omega_src[k], kappa_egmf, kappa_ds[i]);

        } else {
          loglik_event[i,k] += -log(4*pi());
          loglik_event_spatial[i,k] = -log(4*pi());
        }
      }
    }

    array[NEbins] vector[Nsrcs+1] loglik_event_mass;

    for (l in 1:NEbins) {

        loglik_event_mass[l] += left_truncated_normal_lpdf(mean_lnA_det[l] |
            mean_lnA_true[l] + mean_lnA_sys_unc,
            mean_lnA_stat_unc[l], 0.0);
        loglik_event_mass[l] += left_truncated_normal_lpdf(var_lnA_det[l] |
            var_lnA_true[l] + var_lnA_sys_unc,
            var_lnA_stat_unc[l], -2.0);
    }

}
