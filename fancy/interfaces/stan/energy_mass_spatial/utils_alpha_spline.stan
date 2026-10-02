/**
 * Natural cubic spline ("knots method") interpolation over alpha_grid.
 *
 * Separate from utils.stan on purpose: energy_mass_spatial_model.stan (the
 * default, linear-interpolation model) stays completely untouched and
 * reproducible. This file is included only by
 * energy_mass_spatial_model_alpha_spline.stan.
 *
 * @author Keito Watanabe
 * @date September 2026
 */

#include /utils.stan

/**
 * Evaluate a natural cubic spline at a single query point x, given:
 *   - x_values : the fixed knot locations (alpha_grid)
 *   - y_values : the y-values sampled at those knots (recomputed every HMC
 *                iteration, e.g. a mass_fracs-weighted combination of
 *                data-block grids)
 *   - y2_values : the knots' second derivatives, y2 = S * y_values, where S
 *                 is precomputed ONCE in Python from x_values alone (see
 *                 fancy.utils.helpers.natural_cubic_spline_matrix) and passed
 *                 in as data (alpha_spline_matrix). Computing y2 = S * y is
 *                 the caller's job (a single matrix-vector or matrix-matrix
 *                 product, batched across whatever axis is being interpolated
 *                 -- see interpolate_spline_batch below), so this function
 *                 only does the O(1) per-segment cubic evaluation.
 *
 * Outside [x_values[1], x_values[Nx]], falls back to flat extrapolation
 * (returns the boundary y-value), matching interpolate()'s convention in
 * utils.stan so the two models are comparable everywhere except the
 * within-grid interpolation order.
 */
real interpolate_spline(vector x_values, vector y_values, vector y2_values, real x) {
  int Nx = num_elements(x_values);
  real xmin = x_values[1];
  real xmax = x_values[Nx];

  if (x <= xmin) return y_values[1];
  if (x >= xmax) return y_values[Nx];

  int i = binary_search(x, to_array_1d(x_values));
  // binary_search returns an index in [1, Nx-1] for x strictly inside the
  // range (see its docstring in utils.stan); guard the edges defensively
  // since x==xmin/xmax are already handled above.
  if (i < 1) i = 1;
  if (i > Nx - 1) i = Nx - 1;

  real x_lo = x_values[i];
  real x_hi = x_values[i + 1];
  real h = x_hi - x_lo;

  real a = (x_hi - x) / h;
  real b = (x - x_lo) / h;

  real y_lo = y_values[i];
  real y_hi = y_values[i + 1];
  real y2_lo = y2_values[i];
  real y2_hi = y2_values[i + 1];

  return a * y_lo + b * y_hi
       + ((a^3 - a) * y2_lo + (b^3 - b) * y2_hi) * square(h) / 6.0;
}

/**
 * Batched version: evaluate the natural cubic spline at a single query point
 * x for every column of Y at once, given the precomputed second-derivative
 * matrix S (alpha_spline_matrix, Nalphas x Nalphas, fixed data) and the
 * knot y-values Y (Nalphas x M, recomputed this iteration).
 *
 * Y2 = S * Y is a single matrix-matrix product done ONCE per call to this
 * function -- callers should call this once per source/mass-fraction
 * combination (looping over ee/l as needed only for the O(1) segment
 * evaluation), not call interpolate_spline() in a loop with S * y_values
 * recomputed for every column. See energy_mass_spatial_model_alpha_spline.stan
 * for the intended call pattern (mirrors the log_espect_at_alpha /
 * mulnA_mfs / varlnA_mfs loops in the default linear model).
 */
vector interpolate_spline_batch(vector x_values, matrix Y, matrix S, real x) {
  int M = cols(Y);
  matrix[rows(Y), M] Y2 = S * Y;
  vector[M] out;
  for (m in 1:M) {
    out[m] = interpolate_spline(x_values, Y[, m], Y2[, m], x);
  }
  return out;
}

/**
 * 2D interpolation on a (alpha, log10_beta_egmf) grid, splining the alpha
 * (x) axis with a natural cubic spline and keeping the log10_beta_egmf (y)
 * axis linear -- i.e. the same structure as interp2d() in utils.stan, but
 * with interpolate() replaced by interpolate_spline() only in the x
 * direction. Only the alpha axis showed grid-funneling, so only it is
 * splined here; interp2d_alpha_spline_matrix additionally takes S
 * (alpha_spline_matrix) to evaluate the two x-direction splines.
 *
 * xp, yp: the fixed grid points (alpha_grid, log10_beta_egmf_grid).
 * fp: values on that grid, shape (Nalphas, Nbeta_egmfs).
 * S: precomputed alpha_spline_matrix (Nalphas x Nalphas).
 */
real interp2d_alpha_spline(real x, real y, vector xp_vec, array[] real xp,
                            array[] real yp, array[,] real fp, matrix S) {
  int idx_y = binary_search(y, yp);
  int idx_yp1 = idx_y + 1;

  matrix[size(xp), size(yp)] fp_mat = to_matrix(fp);
  matrix[size(xp), size(yp)] fp_y2 = S * fp_mat;  // spline coeffs for every beta column, once

  if (idx_y == 0) {
    return interpolate_spline(xp_vec, fp_mat[, 1], fp_y2[, 1], x);
  }
  else if (idx_y >= size(yp)) {
    return interpolate_spline(xp_vec, fp_mat[, size(yp)], fp_y2[, size(yp)], x);
  }
  real y_vals_low = interpolate_spline(xp_vec, fp_mat[, idx_y], fp_y2[, idx_y], x);
  real y_vals_high = interpolate_spline(xp_vec, fp_mat[, idx_yp1], fp_y2[, idx_yp1], x);
  real val = interpolate(to_vector(yp[idx_y:idx_yp1]), [y_vals_low, y_vals_high]', y);
  return val;
}
