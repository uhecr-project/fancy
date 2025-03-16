/**
 * Functions used with Stan Fits in joint model.
 *
 * @author Francesca Capel
 * @date October 2018
 */

/**
 * Calculate the deflection parameter of the vMF distribution.
 * Derived from Eq. 4 and Eq. 9 in Cape; & Mortlock, 2018.
 * @param E energy in EeV
 * @param B rms magnetic field strength in nG
 * @param D distance in Mpc / 10
 */
real get_kappa(real E, real B, real D, int Z) {
  
  return 7552 * inv_square(2.3 * Z * inv(E / 50) * B * sqrt(D));
}

/**
 * compute the absolute value of a vector
 */
real abs_val(vector input_vector) {
  real av;
  int n = num_elements(input_vector);
  
  real sum_squares = 0;
  for (i in 1:n) {
    sum_squares += (input_vector[i] * input_vector[i]);
  }
  av = sqrt(sum_squares);

  return av;
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

/**
 * Interpolate x from a given set of x and y values.
 */
real interpolate(vector x_values, vector y_values, real x) {
  real x_left;
  real y_left;
  real x_right;
  real y_right;
  real dydx;
  
  int Nx = num_elements(x_values);
  real xmin = x_values[1];
  real xmax = x_values[Nx];
  int i = 1;
  
  if (x > xmax || x < xmin) {
    
    if(x > xmax) {
      return y_values[Nx];
    }
    else if (x < xmin) {
      return y_values[1];
    }
  }
  
  if( x >= x_values[Nx - 1] ) {
    i = Nx - 1;
  }
  else {
    while( x > x_values[i + 1] ) { i += 1; }
  }
  
  x_left = x_values[i];
  y_left = y_values[i];
  x_right = x_values[i + 1];
  y_right = y_values[i + 1];
  
  dydx = (y_right - y_left) / (x_right - x_left);
  
  return y_left + dydx * (x - x_left);
}

/**
 * Calculate the vector of expected E values for different sources.
 */
vector get_Eex(real alpha, vector Eth_src) {
  
  int N = num_elements(Eth_src);
  vector[N] Eex = pow(2, 1 / (alpha - 1)) * Eth_src[1:N];
  
  return Eex;
}

/**
 * Calculate the vector of kappa values for different sources.
 * @param E energy in EeV
 * @param B rms magnetic field strength in nG
 * @param D distance in Mpc / 10
 */
vector get_kappa_ex(vector E, real B, vector D, int Z) {
  
  int Ns = num_elements(E);
  vector[Ns] kappa_ex = 7552 * inv_square(2.3 * Z * inv(E / 50) * B .* sqrt(D));
  return kappa_ex;
}

/**
 * Calculate the Nex for a given kappa by
 * interpolating over a vector of eps values
 * for each source.
 */
real get_Nex(vector F, array[] vector eps, vector kappa_grid, vector kappa, real alpha_T, vector Eth_src, real Eth, real alpha) {
  
  int Ns = num_elements(F);
  vector[Ns] N;
  real eps_from_kappa;
  
  for (k in 1:Ns-1) {
    eps_from_kappa = interpolate(kappa_grid, eps[k], kappa[k]);
    N[k] = F[k] * eps_from_kappa * pow(Eth_src[k] / Eth, 1 - alpha); 
  }
  N[Ns] = F[Ns] * (alpha_T / (4 * pi()));
  
  return sum(N);
}