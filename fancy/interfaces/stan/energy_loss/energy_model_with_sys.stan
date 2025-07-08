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
    int<lower=0> NEbins;  /* number of energy bins for lnA measurements */
    array[N] real Edet;
    array[NEbins] real mean_lnA_det;
    array[NEbins] real var_lnA_det;

    /* model */
    int<lower=0> Nalphas;   /* number of alpha grid points */
    int<lower=0> NEs;    /* number of energy grid points in EeV */
    int<lower=0> NAsrcs;  /* number of source masses */
    vector[Nalphas] alpha_grid;
    vector[NEs] log10_Egrid;
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEs] earth_spectrum_grid;
    /* for lnA, binned via detected energies */
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEbins] mean_lnA_grid;
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEbins] var_lnA_grid;


    /* detector */
    real alpha_T;
    real<lower=0> Eth;
    real<lower=0> E_unc;
    /* below in principle can be a function of NEbins */
    vector[NEbins] mean_lnA_unc;
    vector[NEbins] var_lnA_unc;

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

    /* nuisance parameters */
    real <lower=-2, upper=2>delta_mulnA_sys; /* mean lnA systematic uncertainty */
    real <lower=-3, upper=3>delta_varlnA_sys;  /* var lnA systematic uncertainty */
    real <lower=-0.2, upper=0.2> delta_logE_sys; /* energy scale uncertainty */

}

transformed parameters {

    /* likelihood calculation */   

    /* flux */
    vector[Nsrcs+1] F;
    vector[Nsrcs+1] log_F;

    /* mass fraction weighted spectrum, mean lnA & var lnA */
    /* should be distance dependent too  */
    array[Nsrcs+1] matrix[Nalphas, NEs] espect_mfs;
    array[Nsrcs+1] matrix[Nalphas, NEbins] mulnA_mfs;
    array[Nsrcs+1] matrix[Nalphas, NEbins] varlnA_mfs;
    vector[Nsrcs] esrc_ratios = rep_vector(0.0, Nsrcs); /* source ratios for each source */
    /* calculate the fluxes & mfrac weighted grids */
    for (k in 1:Nsrcs+1) {

        /* initialise the matrix to zero first */
        espect_mfs[k] = rep_matrix(0.0, Nalphas, NEs);
        mulnA_mfs[k] = rep_matrix(0.0, Nalphas, NEbins);
        varlnA_mfs[k] = rep_matrix(0.0, Nalphas, NEbins);

        /* now incrementally add the weights with values */
        for (j in 1:NAsrcs) {
            espect_mfs[k] += mass_fracs[k][j] * earth_spectrum_grid[k,j];
            mulnA_mfs[k] += mass_fracs[k][j] * mean_lnA_grid[k,j];
            varlnA_mfs[k] += mass_fracs[k][j] * var_lnA_grid[k,j];
            if (k < Nsrcs+1) {
                esrc_ratios[k] += mass_fracs[k][j] * interpolate(alpha_grid, esrc_ratio_grid[k,j], alphas[k]);
            }
        }

        /* calculate the flux at Earth! */
        F[k] = pow(10, log10(f_s[k]) + log10_Ftot);
    }


    /* log likelihood for energy + spatial */
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
            lp[i,k] += truncated_lognormal_lpdf(Edet[i] | log(Etrue[i]) + delta_logE_sys, E_unc, Eth, Emax);
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


    /* likelihood factor for lnA */
    array[NEbins] vector[2] lp_lnA;
    vector [NEbins] mean_lnA_true = rep_vector(0.0, NEbins);
    vector [NEbins] var_lnA_true = rep_vector(0.0, NEbins);

    for (l in 1:NEbins) {

        for (k in 1:Nsrcs+1) {

            /* calculate the mean and variance of lnA for each energy bin */
            mean_lnA_true[l] += Nex_arr[k] * interpolate(alpha_grid, to_vector(mulnA_mfs[k][,l]), alphas[k]) / Nex;
            var_lnA_true[l] += Nex_arr[k] * interpolate(alpha_grid, to_vector(varlnA_mfs[k][,l]), alphas[k]) / Nex;

        }

        /* detector response for mean & var lnA */
        lp_lnA[l][1] = left_truncated_normal_lpdf(mean_lnA_det[l] | mean_lnA_true[l] + delta_mulnA_sys, mean_lnA_unc[l], 0.0);
        lp_lnA[l][2] = left_truncated_normal_lpdf(var_lnA_det[l] | var_lnA_true[l] + delta_varlnA_sys, var_lnA_unc[l], -1.0);
    }

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
    /* lnA factor */
  for (l in 1:NEbins) {
    target += lp_lnA[l][1];
    target += lp_lnA[l][2];
  }
  
  /* normalise */
  target += -Nex; 

//   alpha_s ~ normal(-1.0, 0.5);

  /* priors */
  for (k in 1:Nsrcs+1) {
    alphas[k] ~ normal(-1.0, 3.0);
    mass_fracs[k] ~ dirichlet(rep_vector(1.0, NAsrcs));
  }
  
  /* priors defined for each source */
  f_s ~ dirichlet(rep_vector(1.0, Nsrcs+1));

  /* log10_Ftot prior */
  log10_Ftot ~ normal(-1.0, 3.0);

  /* nuisance parameters */
    delta_mulnA_sys ~ normal(0.0, 1.0);
    delta_varlnA_sys ~ normal(0.0, 1.0);

}