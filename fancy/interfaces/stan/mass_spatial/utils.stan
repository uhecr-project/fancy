/**
 * Utility functions / tools for fitting.
 *
 * @author Keito Watanabe
 * @date March 2024
 */

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


/* 
Perform a binary search from given bin edges (array) and specific value
to get the index corresponding to the given value
*/
int binary_search(real value, array [] real binedges)
{
    int L = 1;
    int R = size(binedges);
    int a;
    if (value < binedges[1])
        return 1;
    else if(value > binedges[R])
        // return R+1;
        return R;
    else{
        while (L < R-1)
        {
            a = (L + R) %/% 2;
            if (binedges[a] < value)
                L = a;
            else if (binedges[a] > value)
                R = a;
            else
                return a;
        }
    }
    return L;
}

/* 
* 2-D bilinear interpolation via repeated linear interpolation
*/
real interp2d_bilinear(real x, real y, vector xdata, vector ydata, array [ , ] real fdata) {

  /* for interpolation evaluation */
  real x_left;
  real y_left;
  real x_right;
  real y_right;
  matrix[2,2] D;
  row_vector[2] X;
  vector[2] Y;
  
  /* for index searching */
  int Nx = num_elements(xdata);
  int Ny = num_elements(ydata);
  real xmin = xdata[1];
  real xmax = xdata[Nx];
  real ymin = ydata[1];
  real ymax = ydata[Ny];
  int x_idx = 1;
  int y_idx = 1;

  /* 
  get the index of (xdata, ydata) corresponding to the right edge
  this is why if x <= xmin / y <=ymin we take the index to be 2 instead of 1

  Otherwise we can just raise an error somehow (if this doesnt work)
  */
    
  /* 1. if y > ymax */
  if (y >= ymax) {
    /* x >= xmax */
    if (x >= xmax) {
      x_idx = Nx;
    }
    /* x <= xmin */
    else if (x <= xmin) {
      x_idx = 2;
    }
    /* x is within range */
    else {
      x_idx = binary_search(x, to_array_1d(xdata));
    }
    y_idx = Ny;
  }
  /* 2. if y < ymin */
  else if (y <= ymin) {
    /* x >= xmax */
    if (x >= xmax) {
      x_idx = Nx;
    }
    /* x <= xmin */
    else if (x <= xmin) {
      x_idx = 2;
    }
    /* x is within range */
    else {
      x_idx = binary_search(x, to_array_1d(xdata));
    }
    y_idx = 2;
  }
  /* 3: if y is within the range*/
  else {
    /* x >= xmax */
    if (x >= xmax) {
      x_idx = Nx;
    }
    /* x <= xmin */
    else if (x <= xmin) {
      x_idx = 2;
    }
    /* x is within range */
    else {
      x_idx = binary_search(x, to_array_1d(xdata));
    }
    y_idx = binary_search(y, to_array_1d(ydata));
  }

  /* deal with if binary_search yields 1 */
  if (x_idx == 1) {
    x_idx = 2;
  }
  if (y_idx == 1) {
    y_idx = 2;
  }
  
  /* get the left & right values of the coordinates near (x, y) */
  x_left = xdata[x_idx-1];
  x_right = xdata[x_idx];
  y_left = ydata[y_idx-1];
  y_right = ydata[y_idx];

  /* construct data matrix based on obtained indices */
  D = [
    [fdata[x_idx-1, y_idx-1], fdata[x_idx-1, y_idx]],
    [fdata[x_idx, y_idx-1], fdata[x_idx, y_idx]]
  ];

  /* construct differnce vectors */
  X = [x_right - x, x - x_left];
  Y = [y_right - y, y - y_left]';
  
  /* evaluate */
  return (X * D * Y) / ((x_right - x_left) * (y_right - y_left));
  
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

real interp2d(real x, real y, array[] real xp, array[] real yp, array[,] real fp) {
  /*
  Interpolation on a 2d grid.
  xp and yp should be the points at which fp is evaluated.
  If some point (x, y) is outside the domain, the values along the
  respective boarder are returned.
  */
  int idx_y = binary_search(y, yp);
  int idx_yp1 = idx_y + 1;
  //safeguard against y values outside the defined range
  // interpolate will take care of the same issue in x direction
  if (idx_y == 0) {
    // return result from lowest slice
    return interpolate(to_vector(xp), to_vector(fp[:, 1]), x);
  }
  else if (idx_y >= size(yp)) {
    return interpolate(to_vector(xp), to_vector(fp[:, size(yp)]), x);
  }
  real y_vals_low = interpolate(to_vector(xp), to_vector(fp[:, idx_y]), x);
  real y_vals_high = interpolate(to_vector(xp), to_vector(fp[:, idx_yp1]), x);
  real val = interpolate(to_vector(yp[idx_y:idx_yp1]), [y_vals_low, y_vals_high]', y);
  return val;
}

real truncated_normal_lpdf(real x, real mu, real sigma, real xmin, real xmax) {
    real log_pdf = normal_lpdf(x | mu, sigma); // Log PDF of normal
    real log_cdf_diff = log(Phi((xmax - mu) / sigma) - Phi((xmin - mu) / sigma)); // Log of normalization constant
    return log_pdf - log_cdf_diff; // Adjusted log PDF for truncation
}

real truncated_lognormal_lpdf(real x, real mu, real sigma, real xmin, real xmax) {
    real log_pdf = lognormal_lpdf(x | mu, sigma); // Log PDF of lognormal
    real log_cdf_diff = log(lognormal_cdf(xmax | mu, sigma) - lognormal_cdf(xmin | mu, sigma)); // Log of normalization constant
    return log_pdf - log_cdf_diff; // Adjusted log PDF for truncation
}

real left_truncated_normal_lpdf(real x, real mu, real sigma, real xmin) {
    if (x < xmin) {
        return negative_infinity(); // x is outside the truncation range
    }
    real log_pdf = normal_lpdf(x | mu, sigma); // Log PDF of normal
    real log_cdf_diff = log(1 - Phi((xmin-mu) / sigma)); // Log of normalization constant
    return log_pdf - log_cdf_diff; // Adjusted log PDF for truncation
}

/**
  * Calculate the softmax weights for a vector x with temperature tau.
  * This is used to calculate the weights for the source charge number.
  * @param x vector of values to calculate softmax weights for
  * @param tau dampening parameter for softmax, lower means more peaked distribution
  * @return vector of softmax weights
  */
  vector soft_argmax_weights(vector x, real tau) {
      return softmax(x / tau);
  }