/**
 * Spatial-only model -- "knots method" variant.
 *
 * Identical to spatial_model.stan EXCEPT that every interpolation over
 * alpha_grid (log_wexp_earth_grid / log_wexp_src_grid via
 * interp2d_alpha_spline, mulnA_mfs/varlnA_mfs, esrc_ratio_grid, and the
 * alpha axis of energy_spectrum_lpdf's 2D interpolation) uses a natural
 * cubic spline instead of linear interpolation, evaluated via a precomputed
 * second-derivative matrix (alpha_spline_matrix, built once in Python from
 * alpha_grid alone -- see fancy.utils.helpers.natural_cubic_spline_matrix).
 *
 * Interpolation over logE_grid and log10_beta_egmf_grid (the y-axes of the
 * 2D interpolations) is UNCHANGED (still linear) -- only the alpha_grid axis
 * is splined here.
 *
 * NB: in this analysis_type alphas are fixed data, not parameters, so the
 * spline only changes how Nex / Lsrcs / lnA moments are evaluated at the
 * fixed alphas (for consistency with the other *_alpha_spline models).
 *
 * Ported from energy_mass_spatial_model_alpha_spline.stan. Deliberately kept
 * as a separate file from spatial_model.stan so the default
 * linear-interpolation model stays untouched and reproducible.
 *
 * @author Keito Watanabe
 * @date September 2026
 */

functions {
    #include /utils_alpha_spline.stan

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
    array[N] unit_vector[3] omega_det; /* arrival directions */
    vector[N] exposure_factor; /* exposure correction factors per event */
    vector[N] kappa_ds; /* deflection parameters, including GMF deflections + arrival direction uncertainty */

    /* grid of effective exposures at earth vs kappa (natural log); last row is the background */
    int Nkappas;
    vector[Nkappas] log_kappa_grid;
    array[Nsrcs+1] vector[Nkappas] log_wexp_earth_grid;

    /* computation parameters */
    int<lower=1> grain_size; /* for reduce_sum, generatlly N / (4 * ncores) is a good estimate */
}

transformed data {

  // --- distance conversion once ---
  vector[Nsrcs] D_flux = D * 3.08567758e19;

  // number of events for reduced sum
    array[N] int event_ids;
    for (n in 1:N) {event_ids[n] = n;}
}

parameters {

    /* flux fraction per source (+ BG) */
    simplex[Nsrcs+1] flux_frac;        

    /* total flux AT EARTH */
    real log10_Ftot;

    /* mean deflection parameter per UHECR */
    real<lower=0> kappa_egmf;

}

transformed parameters {

  // --- only quantities needed downstream ---
  vector[Nsrcs+1] F;
  vector[Nsrcs+1] wexp_earths;

  // initialise
  F = rep_vector(0.0, Nsrcs+1);
  wexp_earths = rep_vector(0.0, Nsrcs+1);

  for (k in 1:Nsrcs+1) {
    // flux at Earth
    F[k] = pow(10.0, log10_Ftot) * flux_frac[k];

    // effective exposure at Earth, interpolated (linearly in log-log) at the shared kappa_egmf
    wexp_earths[k] = exp(interpolate(
      log_kappa_grid, log_wexp_earth_grid[k], log(kappa_egmf)
    ));
  }

  vector[Nsrcs+1] Nex_arr = F .* wexp_earths;
  real Nex = sum(Nex_arr);
}

model {
  // --- priors ---

  // flux fraction weights: Dirichlet-like prior
  flux_frac ~ dirichlet(rep_vector(2.0, Nsrcs+1));

  // total flux : normal distribution in log10
  log10_Ftot ~ normal(-1.0, 3.0);

  // mean deflection parameter from EGMF
  kappa_egmf ~ lognormal(log(100.), 1.);

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
        
}