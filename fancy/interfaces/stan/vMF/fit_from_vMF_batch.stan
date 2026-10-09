/**
/*
/* Estimate the concentration parameters of M independent vMF distributions from samples,
/* in one program. Same likelihood and (flat) prior as fit_from_vMF.stan for each kappa,
/* written in terms of the sufficient statistics of each fit:
/*   sum_i vMF_lpdf(cos_theta_i | kappa) = kappa * sum_cos + N * (log normalisation)
/* with sum_cos = sum_i dot_product(n_i, mu) over the N samples of that fit.
/*
*/

data {
    int<lower=0> M;  /* number of independent fits */
    array[M] int<lower=1> N;  /* number of samples per fit */
    vector[M] sum_cos;  /* sum of dot products between the samples and their mean direction */
}

parameters {
    vector<lower=0>[M] kappa;  /* concentration parameters */
}

model {
    for (m in 1:M) {
        // same large-kappa branch as vMF_lpdf in fit_from_vMF.stan
        if (kappa[m] > 100) {
            target += kappa[m] * sum_cos[m] + N[m] * (log(kappa[m]) - log(4 * pi()) - kappa[m] + log(2));
        }
        else {
            target += kappa[m] * sum_cos[m] + N[m] * (log(kappa[m]) - log(4 * pi() * sinh(kappa[m])));
        }
    }
}
