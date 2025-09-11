/**
 * Energy + lnA model
 *
 * @author Keito Watanabe
 * @date March 2024
 */

functions {
    #include /utils.stan

    real energy_spectrum_lpdf(real E, real alpha_s, vector log10_en_grid, vector alph_grid, matrix espect_mfs) {
        return interp2d(alpha_s, log10(E), to_array_1d(alph_grid), to_array_1d(log10_en_grid), to_array_2d(log(espect_mfs)));
    }
}

data {

    // /* sources */
    int<lower=1> Nsrcs;
    vector[Nsrcs] D;

    /* uhecr */
    int<lower=0> N;
    array[N] real Edet;

    /* model */
    int<lower=0> Nalphas;   /* number of alpha grid points */
    int<lower=0> NEs;    /* number of energy grid points in EeV */
    int<lower=0> NAsrcs;  /* number of source masses */
    vector[Nalphas] alpha_grid;
    vector[NEs] log10_Egrid;
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEs] earth_spectrum_grid;


    /* detector */
    real alpha_T;
    real<lower=0> Eth;
    real<lower=0> logE_stat_unc;

    /* systematic uncertainties, as global shifts */
    real logE_sys_unc;

    /* Nex */
    array[Nsrcs+1, NAsrcs] vector[Nalphas] det_rate_grid;
    array[Nsrcs, NAsrcs] vector[Nalphas] esrc_ratio_grid;
}

transformed data {
    real Emin = pow(10.0, min(log10_Egrid));
    real Emax = pow(10.0, max(log10_Egrid));

    real alpha_min = min(alpha_grid);
    real alpha_max = max(alpha_grid);

    /* transform units for distance to make units consistent */
    vector[Nsrcs] D_flux = D * 3.08567758e19;
}

parameters {

    /* spectral information */
    vector <lower=alpha_min, upper=alpha_max>  [Nsrcs+1] alphas;

    /* mass fractions (in future, 2D structure with sources) */
    array[Nsrcs+1] simplex[NAsrcs] mass_fracs;

    /* association fraction per source (+ BG) */
    simplex [Nsrcs+1] f_s;

    /* total flux AT EARTH */
    real<upper=0> log10_Ftot;

    /* latent parameters */
    vector <lower=Eth, upper=Emax>[N] Etrue;   

}

transformed parameters {

    /* likelihood calculation */   

    /* flux */
    vector[Nsrcs+1] F;
    vector[Nsrcs+1] log_F;

    /* mass fraction weighted spectrum, mean lnA & var lnA */
    /* should be distance dependent too  */
    array[Nsrcs+1] matrix[Nalphas, NEs] espect_mfs;
    vector[Nsrcs] esrc_ratios = rep_vector(0.0, Nsrcs); /* source ratios for each source */
    /* calculate the fluxes & mfrac weighted grids */
    for (k in 1:Nsrcs+1) {

        /* initialise the matrix to zero first */
        espect_mfs[k] = rep_matrix(0.0, Nalphas, NEs);

        /* now incrementally add the weights with values */
        for (j in 1:NAsrcs) {
            espect_mfs[k] += mass_fracs[k][j] * earth_spectrum_grid[k,j];
            if (k < Nsrcs+1) {
                esrc_ratios[k] += mass_fracs[k][j] * interpolate(alpha_grid, esrc_ratio_grid[k,j], alphas[k]);
            }
        }

        /* calculate the flux at Earth! */
        F[k] = pow(10, log10(f_s[k]) + log10_Ftot);
    }


    /* log likelihood for energy */
    array[N] vector[Nsrcs+1] lp;
    log_F = log(F);

    // real alpha;

    /* rate factor */
    for (i in 1:N) {

        lp[i] = log_F;

        for (k in 1:Nsrcs+1) {
        
            /* calculate energy spectrum */
            lp[i,k] += energy_spectrum_lpdf(Etrue[i] | alphas[k], log10_Egrid, alpha_grid, espect_mfs[k]);

            /* detector response for energy */
            lp[i,k] += truncated_lognormal_lpdf(Edet[i] | log(Etrue[i]) + logE_sys_unc, logE_stat_unc, Eth, Emax);
        }
    }


    /* Nex */
    array[Nsrcs+1] vector[NAsrcs] det_rates;
    real<lower=0> Nex;   /* expected number of events  */
    real<lower=0, upper=1> src_frac; /* source fraction */
    vector[Nsrcs+1] Nex_arr = rep_vector(0.0, Nsrcs+1);

    for (k in 1:Nsrcs+1) {

        for (j in 1:NAsrcs) {
            det_rates[k][j] = interpolate(alpha_grid, det_rate_grid[k,j], alphas[k]);
        }

        Nex_arr[k] = F[k] * alpha_T * dot_product(det_rates[k], mass_fracs[k]);
    }

    Nex = sum(Nex_arr);
    real Nex_src = sum(Nex_arr[1:Nsrcs]);
    real Nex_bg = Nex_arr[Nsrcs+1];  /* not needed, but good to keep for diagnosis */
    src_frac = Nex_src / Nex; /* fraction of detected events from sources */

     /* Here we calculate the source luminosity by transforming earth flux -> source flux using integrated source spectrum */
    array[Nsrcs] real Lsrcs; /* source luminosity */
    vector[Nsrcs] src_det_ratio = rep_vector(0.0, Nsrcs); /* source detection ratio */
    for (k in 1:Nsrcs) {
        for (j in 1:NAsrcs) {
            src_det_ratio[k] += mass_fracs[k][j] * interpolate(alpha_grid, det_rate_grid[k,j], alphas[k]);
        }
        real earth_det_ratio = dot_product(det_rates[k], mass_fracs[k]);
        Lsrcs[k] = F[k] * src_det_ratio[k] / earth_det_ratio * 4 * pi() * pow(D_flux[k], 2.0) * esrc_ratios[k];
    }
   

}

model {

  /* rate factor */
  for (i in 1:N) {
    target += log_sum_exp(lp[i]);
  }
  
  /* normalise */
  target += -Nex; 

  /* priors */
  for (k in 1:Nsrcs+1) {
    alphas[k] ~ normal(-1.0, 3.0);
    mass_fracs[k] ~ dirichlet(rep_vector(1.0, NAsrcs));
  }
  
  /* priors defined for each source */
  f_s ~ dirichlet(rep_vector(1.0, Nsrcs+1));

  /* log10_Ftot prior */
  log10_Ftot ~ normal(-1.0, 3.0);

}