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
    * @param log10_en_grid : grid of log10(E) values for interpolation
    * @param alph_grid : grid of alpha values for interpolation
    * @param espect_mfs : mass fraction weighted energy spectrum at Earth. Shape in (Nalphas, NEs)
    * @return log probability density of the energy spectrum at Earth
    */
    real energy_spectrum_lpdf(real E, real alpha, vector log10_en_grid, vector alph_grid, matrix espect_mfs) {
        return interp2d(alpha, log10(E), to_array_1d(alph_grid), to_array_1d(log10_en_grid), to_array_2d(log(espect_mfs)));
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
    real inner = abs_val((kappa_d * v) + (kappa * mu));
    
    if (kappa > 100 || kappa_d > 100) {
        lprob = log(kappa * kappa_d) - log(4 * pi() * inner) + inner - (kappa + kappa_d) + log(2);
    }
    else {   
        lprob = log(kappa * kappa_d) - log(4 * pi() * sinh(kappa) * sinh(kappa_d)) + log(sinh(inner)) - log(inner);
    }
    
    return lprob;   
    }

}

data {

    // /* sources */
    int<lower=1> Nsrcs;
    vector[Nsrcs] D;
    array[Nsrcs+1] unit_vector[3] omega_src;  /* source directions */

    /* uhecr */
    int<lower=0> N;
    int<lower=0> NEbins;  /* number of energy bins for lnA measurements */
    array[N] real Edet;
    array[N] unit_vector[3] omega_det; /* arrival directions */
    vector[N] kappa_ds; /* deflection parameters, including GMF deflections + arrival direction uncertainty */
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
    vector[NEbins] lnA_Egrid; /* grid of energy bins for lnA model */
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEbins] mean_lnA_grid;
    array[Nsrcs+1, NAsrcs] matrix [Nalphas, NEbins] var_lnA_grid;


    /* detector */
    real<lower=0> Eth;

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
    vector [Nbeta_egmfs] log10_beta_egmf_grid; /* grid of EGMF spread parameters */
    /* grid of weighted exposures at earth and source */
    array[Nsrcs+1, NAsrcs] matrix[Nalphas, Nbeta_egmfs] wexp_earth_grid;
    array[Nsrcs, NAsrcs] vector[Nalphas] wexp_src_grid;
    array[Nsrcs, NAsrcs] vector[Nalphas] esrc_ratio_grid;
}

transformed data {
    real Emin = pow(10.0, min(log10_Egrid));
    real Emax = pow(10.0, max(log10_Egrid));

    real alpha_min = min(alpha_grid);
    real alpha_max = max(alpha_grid);

    real beta_egmf_min = pow(10.0, min(log10_beta_egmf_grid));
    real beta_egmf_max = pow(10.0, max(log10_beta_egmf_grid));

    /* transform units for distance to make units consistent */
    vector[Nsrcs] D_flux = D * 3.08567758e19;
}

parameters {

    /* spectral information */
    vector <lower=alpha_min, upper=alpha_max>  [Nsrcs+1] alphas;

    /* mass fractions (in future, 2D structure with sources) */
    array[Nsrcs+1] simplex[NAsrcs] mass_fracs;

    /* flux fraction per source (+ BG) */
    simplex [Nsrcs+1] flux_frac;

    /* total flux AT EARTH */
    real log10_Ftot;

    /* EGMF spread parameter, in nG Mpc^1/2 */
    real<lower=beta_egmf_min, upper=beta_egmf_max> beta_egmf;

    /* latent parameters */
    vector <lower=Eth, upper=Emax>[N] Etrue;

    // real nu_lnA; /* latent variable for sampling lnA */

}

transformed parameters {

    /* likelihood calculation */ 

    /* log likelihood parameter */
    array[N] vector[Nsrcs+1] lp;  /* energy & spatial likelihood */
    array[NEbins] vector[2] lp_lnA;  /* lnA likelihood */

    /* flux */
    vector[Nsrcs+1] F; /* flux at EARTH for each contribution */
    array[Nsrcs] real Lsrcs; /* source luminosity */

    /* spatial deflection parameters */
    vector [N] kappas;  /* EGMF deflection parameter for each UHECR */
    vector[N] Rtrue; /* true rigidity of each UHECR */
    vector[N] Zsrc_true; /* true charge number of each UHECR at the source */

    /* mass parameters */
    vector [NEbins] mean_lnA_true = rep_vector(0.0, NEbins);
    vector [NEbins] var_lnA_true = rep_vector(0.0, NEbins);

    /* Nex */
    real<lower=0> Nex;   /* expected number of events  */
    real<lower=0> Nex_src;  /* expected number of events from sources */
    real<lower=0> Nex_bg;   /* expected number of events from background */
    real<lower=0, upper=1> src_frac; /* source fraction */
    vector[Nsrcs+1] Nex_arr = rep_vector(0.0, Nsrcs+1);

    /* mass fraction weighted spectrum, mean lnA & var lnA */
    /* should be distance dependent too  */
    array[Nsrcs+1] matrix[Nalphas, NEs] espect_mfs;
    array[Nsrcs+1] matrix[Nalphas, NEbins] mulnA_mfs;
    array[Nsrcs+1] matrix[Nalphas, NEbins] varlnA_mfs;

    /* mean source energy */
    vector[Nsrcs] esrc_ratios = rep_vector(0.0, Nsrcs); /* source ratios for each source */
    /* source detection factor */
    vector[Nsrcs] wexp_src = rep_vector(0.0, Nsrcs);
    /* earth detection factor */
    vector[Nsrcs+1] wexp_earth = rep_vector(0.0, Nsrcs+1);


    /* calculate the fluxes & mfrac weighted grids */
    /* here all the mass fraction weighted grids are calculated */
    /* also all the Nex stuff too */
    for (k in 1:Nsrcs+1) {

        /* calculate the flux at earth coming from each source + BG */
        F[k] = pow(10.0, log10(flux_frac[k]) + log10_Ftot);

        /* initialise the matrix to zero first */
        espect_mfs[k] = rep_matrix(0.0, Nalphas, NEs);
        mulnA_mfs[k] = rep_matrix(0.0, Nalphas, NEbins);
        varlnA_mfs[k] = rep_matrix(0.0, Nalphas, NEbins);

        /* now incrementally add the weights with values */
        for (j in 1:NAsrcs) {
            espect_mfs[k] += mass_fracs[k][j] * earth_spectrum_grid[k,j];
            mulnA_mfs[k] += mass_fracs[k][j] * mean_lnA_grid[k,j];
            varlnA_mfs[k] += mass_fracs[k][j] * var_lnA_grid[k,j];

            wexp_earth[k] += mass_fracs[k][j] * pow(10.0, interp2d(
                alphas[k], log10(beta_egmf), to_array_1d(alpha_grid), to_array_1d(log10_beta_egmf_grid), to_array_2d(log10(wexp_earth_grid[k,j]))
            ));

            if (k < Nsrcs+1) {
                esrc_ratios[k] += mass_fracs[k][j] * interpolate(alpha_grid, esrc_ratio_grid[k,j], alphas[k]);
                wexp_src[k] += mass_fracs[k][j] * interpolate(alpha_grid, wexp_src_grid[k,j], alphas[k]);
            }
        }

        /* Here we calculate the source luminosity by transforming earth flux -> source flux using integrated source spectrum */
        if (k < Nsrcs+1) {
            Lsrcs[k] = F[k] * wexp_src[k] / wexp_earth[k] * 4 * pi() * pow(D_flux[k], 2.0) * esrc_ratios[k];
        }

        Nex_arr[k] = F[k] * wexp_earth[k];

    }

    Nex = sum(Nex_arr);
    Nex_src = sum(Nex_arr[1:Nsrcs]);
    Nex_bg = Nex_arr[Nsrcs+1]; 
    src_frac = Nex_src / Nex; 

    /* unbinned likelihood energy */
    for (i in 1:N) {

        lp[i] = log(F);

        for (k in 1:Nsrcs+1) {
        
            /* calculate energy spectrum */
            lp[i,k] += energy_spectrum_lpdf(Etrue[i] | alphas[k], log10_Egrid, alpha_grid, espect_mfs[k]);

            /* detector response for energy */
            lp[i,k] += truncated_lognormal_lpdf(Edet[i] | log(Etrue[i]) + logE_sys_unc, logE_stat_unc, Eth, Emax);
        }
    }

    /* binned likelihood calculation for lnA, separate from energy */
    for (l in 1:NEbins) {

        for (k in 1:Nsrcs+1) {

            /* calculate the mean and variance of lnA for each energy bin */
            mean_lnA_true[l] += Nex_arr[k] * interpolate(alpha_grid, to_vector(mulnA_mfs[k][,l]), alphas[k]) / Nex;
            var_lnA_true[l] += Nex_arr[k] * interpolate(alpha_grid, to_vector(varlnA_mfs[k][,l]), alphas[k]) / Nex;

        }

        /* detector response for mean & var lnA */
        lp_lnA[l][1] = left_truncated_normal_lpdf(mean_lnA_det[l] | mean_lnA_true[l] + mean_lnA_sys_unc, mean_lnA_stat_unc[l], 0.0);
        lp_lnA[l][2] = left_truncated_normal_lpdf(var_lnA_det[l] | var_lnA_true[l] + var_lnA_sys_unc, var_lnA_stat_unc[l], -1.0);
    }

    /* unbinned spatial likelihood */
    for (i in 1:N) {
        
        /* calculate the true rigidity for each UHECR, using the mean and variance of lnA */
        real mean_lnA = mean_lnA_true[binary_search(Etrue[i], to_array_1d(lnA_Egrid))];
        // real var_lnA = var_lnA_true[binary_search(Etrue[i], to_array_1d(lnA_Egrid))];

        // real lnA_sample = mean_lnA + sqrt(var_lnA) * nu_lnA;
        // Zsrc_true[i] = 0.5 * exp(lnA_sample);
        Zsrc_true[i] = 0.5 * exp(mean_lnA); /* use mean lnA for rigidity calculation */

        Rtrue[i] = Etrue[i] / Zsrc_true[i];

        for (k in 1:Nsrcs+1) {
            /* now we can calculate the deflection parameter for each source */
            if (k <= Nsrcs) {
                /* we use the EGMF deflection parameter, which is a function of Rtrue, beta_egmf and D */
                /* note that D is in Mpc, so we need to convert it to Mpc/10 */
                /* beta_egmf is to nG Mpc^1/2 */
                kappas[i] = get_kappa(Rtrue[i], beta_egmf, D[k] / 10.0);
                /* add the combined spatial likelihood from EGMF + GMF deflections + arrival direction uncertainty */
                lp[i, k] += fik_lpdf(omega_det[i] | omega_src[k], kappas[i], kappa_ds[i]);
            }
            else {
                /* for the background, we assume isotropic deflection */
                kappas[i] = 0.0;
                lp[i,k] += -log(4 * pi());
            }

            
        }   
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

  /* priors */
  for (k in 1:Nsrcs+1) {
    alphas[k] ~ normal(0.0, 3.0);
    mass_fracs[k] ~ dirichlet(rep_vector(1.0, NAsrcs));
  }
  
  /* priors defined for each source */
  flux_frac ~ dirichlet(rep_vector(1.0, Nsrcs+1));

  /* log10_Ftot prior */
  log10_Ftot ~ normal(-1.0, 3.0);

  /* magnetic spread prior */
  /* NB: lognormal -> mean and std are in log space */
  beta_egmf ~ lognormal(log(0.5), 1.0);

  /* nuisance parameters */
//   nu_lnA ~ normal(0.0, 1.0);

}