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
  
  /* uhecr */
  int<lower=0> N;
   
  /* observatory */ 
  real<lower=0> alpha_T;

  /* rigidity */
  real<lower=0> Rth;
  real<lower=0> Rerr;
  array[N] real<lower=Rth> Rdet;

}

transformed data {

    real Rmin = Rth;  /* EV */
    real Rmax = 250; /* EV */

}


parameters { 

  /* background flux */
  real<lower=0, upper=10> F0;
  
  /* energy spectrum */
  real<lower=-3, upper=10> alpha_b; 
  vector<lower=Rmin, upper=Rmax>[N] R;

}


transformed parameters {
    
  /* association probability */
  array[N] vector[1] lp;
  vector[N] kappa;
  real log_F;
  
  /* Nex */
  real<lower=0> Nex;

  log_F = log(F0);

  /* likelihood calculation */
  /* rate factor */
  for (i in 1:N) {

    lp[i,1] = log_F;

    /* spatial */
    lp[i,1] += log(1 / ( 4.0 * pi() ));

    /* rigidity */
    lp[i,1] += background_spectrum_lpdf(R[i] | alpha_b, Rmin, Rmax);

    /* truncated gaussian */
    lp[i,1] += normal_lpdf(Rdet[i] | R[i], Rerr * R[i]);
    
    if (Rdet[i] < Rth)
    {
      lp[i,1] += negative_infinity();
    }
    else
    {
       lp[i,1] += -normal_lccdf(Rth | R[i], Rerr * R[i]);
    }
    

  }

  Nex = F0 * (alpha_T / (4.0 * pi()));
  
}


model {

  /* rate factor */
  for (i in 1:N) {
    target += log_sum_exp(lp[i]);
  }
  
  /* normalise */
  target += -Nex; 

  /* priors */
  alpha_b ~ normal(2.0, 3.0);
  // F0 ~ normal(0.0, 1.0);

}

generated quantities {

  array[N] int lambda;

  /* used in calculating the source-UHECR association probabilities */
  for (i in 1:N) {

    lambda[i] = categorical_logit_rng(lp[i]);

  }

}


