/**
 * Energy + lnA model
 *
 * @author Keito Watanabe
 * @date March 2024
 */

functions {
    #include /utils.stan
}

data {

    // /* sources */
    int<lower=1> Nsrcs;
    vector[Nsrcs] D;

    /* uhecr */
    int<lower=0> NEbins;  /* number of energy bins for lnA measurements */
    array[NEbins] real mean_lnA_det;
    array[NEbins] real var_lnA_det;

    /* model */
    int<lower=0> Nalphas;   /* number of alpha grid points */
    int<lower=0> NEs;    /* number of energy grid points in EeV */
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

    /* Nex */
    array[Nsrcs+1, NAsrcs] vector[Nalphas] log_wexp_earth_grid;
    array[Nsrcs, NAsrcs] vector[Nalphas] log_wexp_src_grid;
    array[Nsrcs, NAsrcs] vector[Nalphas] esrc_ratio_grid;

    /* computation parameters */
    int<lower=1> grain_size; /* for reduce_sum, generally N / (4 * ncores) is a good estimate */
}

transformed data {

    real alpha_min = min(alpha_grid);
    real alpha_max = max(alpha_grid);

    /* transform units for distance to make units consistent */
    vector[Nsrcs] D_flux = D * 3.08567758e19;

    // --- precompute 1D arrays for interpolation ---
    vector[Nalphas] alpha_grid_vec = to_vector(alpha_grid);
    vector[NEbins] lnA_logE_grid_vec = to_vector(lnA_logE_grid);
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

}

transformed parameters {

    // --- only quantities needed downstream ---
  vector[Nsrcs+1] F;
  array[Nsrcs+1] matrix [Nalphas, NEbins] mulnA_mfs;
  array[Nsrcs+1] matrix [Nalphas, NEbins] varlnA_mfs;
  vector[Nsrcs+1] wexp_earths;
  array[Nsrcs+1] vector[NEs] log_espect_at_alpha;

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

      wexp_earths[k] += mass_fracs[k][j] * exp(interpolate(
        alpha_grid_vec,
        log_wexp_earth_grid[k,j],
        alphas[k]
    ));

    }
  }

  vector[Nsrcs+1] Nex_arr = F .* wexp_earths;
  real Nex = sum(Nex_arr);

  // mean and sigma lnA values
  vector[NEbins] mean_lnA_true = rep_vector(0.0, NEbins);
  vector[NEbins] var_lnA_true = rep_vector(0.0, NEbins);

   // --- binned lnA likelihood ---
  for (l in 1:NEbins) {
    for (k in 1:Nsrcs+1) {

        /* calculate the mean and variance of lnA for each energy bin */
        mean_lnA_true[l] += Nex_arr[k] * interpolate(alpha_grid_vec, to_vector(mulnA_mfs[k][,l]), alphas[k]) / Nex;
        var_lnA_true[l] += Nex_arr[k] * interpolate(alpha_grid_vec, to_vector(varlnA_mfs[k][,l]), alphas[k]) / Nex;

    }
  }
   

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
  flux_frac ~ dirichlet([2.0, 2.0]);

  // total flux : normal distribution in log10
  log10_Ftot ~ normal(-1.0, 3.0);

  // --- binned lnA likelihood ---
  for (l in 1:NEbins) {
    target += left_truncated_normal_lpdf(mean_lnA_det[l] |
              mean_lnA_true[l] + mean_lnA_sys_unc,
              mean_lnA_stat_unc[l], 0.0);
    target += left_truncated_normal_lpdf(var_lnA_det[l] |
              var_lnA_true[l] + var_lnA_sys_unc,
              var_lnA_stat_unc[l], -2.0);
  }

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
        esrc_ratios += mass_fracs[k][j] * interpolate(alpha_grid_vec, esrc_ratio_grid[k,j], alphas[k]);
        wexp_src += mass_fracs[k][j] * exp(interpolate(
          alpha_grid_vec,
          log_wexp_src_grid[k,j],
          alphas[k]
        ));
      }

      Lsrcs[k] = F[k] * wexp_src / wexp_earths[k] * 4*pi() * square(D_flux[k]) * esrc_ratios;
    
    }

    vector[Nsrcs] log10_Lsrcs = log10(Lsrcs);

    

    // generate the log-likelihood of the event to get the
    // association probability as well
    array[NEbins] vector[Nsrcs+1] loglik_event;

    for (l in 1:NEbins) {

        loglik_event[l] += left_truncated_normal_lpdf(mean_lnA_det[l] |
            mean_lnA_true[l] + mean_lnA_sys_unc,
            mean_lnA_stat_unc[l], 0.0);
        loglik_event[l] += left_truncated_normal_lpdf(var_lnA_det[l] |
            var_lnA_true[l] + var_lnA_sys_unc,
            var_lnA_stat_unc[l], -2.0);
    }
        
}