/**
 * Joint model with energy, directions, GMF, and compositions.
 *
 * @author Keito Watanabe
 * @date March 2024
 */

functions {

/**
 * Arrival rigidity spectrum for isotropic background sources.
 * .
 * @param R : rigidity in RV
 * @param alpha : background spectral index
 * @param Rmin : minimum rigidity in sample, must match min(R_grid)
 * @param Rmax : maximum rigidity in sample, must match max(R_grid)
 */
real background_spectrum_lpdf(real R, real alpha, real Rmin, real Rmax)
{
  real n;
  if(alpha != 1.0)
  {
    n = ((1.0-alpha)/((Rmax^(1.0-alpha))-(Rmin^(1.0-alpha))));
  }
  else
  {
    n = (1.0/(log(Rmax)-log(Rmin)));
  }
  if (R > Rmax || R < Rmin) {
    return negative_infinity();
  }
  else {
    return log((n * pow(R, (-alpha))));
  }
}

}

data {
   
  /* observatory */ 
  real<lower=0> Rth;
  real<lower=0> Rth_max;
  real<lower=0> Rerr;

  /* uhecr */
  int<lower=0> N; 
  array[N] real Rdet;

  real Rmin;
  real Rmax;

}


parameters { 

  real alpha_b;  /* background spectral index */

  /* rigidity, in EV */
  vector<lower=Rmin, upper=Rmax>[N] R;

}


transformed parameters {
      
  /* association probability parameters */
  array[N] real lp;
  

  /* likelihood calculation */
  /* rate factor */
  for (i in 1:N) {

    lp[i] = 1.0;  

    /* bounded rigidity spectrum */
    lp[i] += background_spectrum_lpdf(R[i] | alpha_b, Rmin, Rmax);

    /* detection probability for rigidity information */
    lp[i] += normal_lpdf(Rdet[i] | R[i], Rerr * R[i]);
    if (Rdet[i] < Rth || Rdet[i] > Rth_max) {
      lp[i] += negative_infinity();
    }
    else {
      lp[i] += -log_diff_exp(normal_lcdf(Rth_max | R[i], Rerr * R[i]), normal_lcdf(Rth | R[i], Rerr * R[i]));
    }

  }
  
}


model {

  /* rate factor */
  for (i in 1:N) {
    target += lp[i];
  }

  /* priors */
  // alpha_b ~ normal(2.0, 5.0);

}


