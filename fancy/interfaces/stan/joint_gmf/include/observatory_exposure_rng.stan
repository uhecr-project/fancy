/**
 * Functions for modelling the exposure of a UHECR observatory.
 *
 * @author Francesca Capel
 * @date May 2018
 */


/**
 * Calculate xi part of exposure.
 * @param theta from 0 to pi.
 * @param p observatory dependent parameters.
 */
real xi_exp(real theta, array[] real p) { 
  return (p[3] - (p[2] * cos(theta))) / (p[1] * sin(theta));
}

/**
 * Calculate alpha_m part of exposure.
 * @param theta from 0 to pi.
 * @param p observatory dependent parameters.
 */
real alpha_m(real theta, array[] real p) {
  
  real am;
  
  real xi_val = xi_exp(theta, p);
  if (xi_val > 1) {
    am = 0;
  }
  else if (xi_val < -1) {
    am = pi();
  }
  else {
    am = acos(xi_val);
  }
  
  return am;
}

/**
 * Calculate the exposure factor for a given position on the sky. 
 * @param theta from 0 to pi.
 * @param p observatory dependent parameters.
 *
 * KW: theta = pi/2 - declination, so dec = -90 deg
 * corresponds to theta = pi, dec = +90 deg 
 * is theta = 0.
 */
real m(real theta, array[] real p) {
  return (p[1] * sin(theta) * sin(alpha_m(theta, p)) 
	  + alpha_m(theta, p) * p[2] * cos(theta));
}

/**
 * Convert from unit vector omega to theta of spherical coordinate system.
 * @param omega a 3D unit vector.
 */
real omega_to_theta(vector omega) {
  
  real theta;
  
  int N = num_elements(omega);
  
  if (N != 3) {
    print("Error: input vector omega must be of 3 dimensions");
  }
  
  theta = acos(omega[3]);
  
  return theta;
}

/**
 * Sample from the vMF centred on varpi with spread kappa, 
 * accounting for detector exposure effects.
 * Uses rejection sampling.
 */
vector exposure_limited_vMF_rng(vector varpi, real kappa, real a0, real theta_m) {
  
  array[3] real params;
  real m_max;
  real accept;
  int count;
  vector[3] omega;
  real theta;
  real pdet;
  real max_dec;
  vector[2] p;
  
  /* exposure */
  params[1] = cos(a0);
  params[2] = sin(a0);
  params[3] = cos(theta_m);

  /* declination corresponding to maximum
   * exposure factor changes between PAO and TA.
   * We set this based on latitude sign 
   * (PAO: a0 < 0, TA: a0 > 0).
  */
  if (a0 < 0) {
    max_dec = pi();
  }
  else if (a0 > 0) {
    max_dec = 0.;
  }

  m_max = m(max_dec, params);
  
  accept = 0;
  count = 0;

  while (accept != 1) {	  

    omega = vMF_rng(varpi, kappa);
    theta = omega_to_theta(omega);
    pdet = m(theta, params) / m_max;
    p[1] = pdet;
    p[2] = 1 - pdet;
    accept = categorical_rng(p);
    count += 1;
    if (count > 1.0e7) {
      
      print("Was stuck in exposure_limited_rng");
      accept = 1;

    }
    
  } 
  
    return omega;
  }

/**
 * Sample uniformly from the unit sphere, 
 * accounting for detector exposure effects.
 * Uses rejection sampling.
 */
vector exposure_limited_sphere_rng(real a0, real theta_m) {
  
  array[3] real params;
  real m_max;
  real accept;
  vector[3] omega;
  real theta;
  real pdet;
  real max_dec;
  vector[2] p;
  
  params[1] = cos(a0);
  params[2] = sin(a0);
  params[3] = cos(theta_m);
  
  if (a0 < 0) {
    max_dec = pi();
  }
  else if (a0 > 0) {
    max_dec = 0.;
  }

  m_max = m(max_dec, params);
  
  accept = 0;
  while (accept != 1) {	  

    omega = sphere_rng(1);
    theta = omega_to_theta(omega);
    pdet = m(theta, params) / m_max;
    p[1] = pdet;
    p[2] = 1 - pdet;
    accept = categorical_rng(p);

  } 

    return omega;
  }
