"""Helper functions for the code."""

import numpy as np
import astropy.units as u
from scipy.stats import lognorm

km_per_Mpc = 3.08567758e19 


def theta_igmf(R: float, beta_egmf: float, D: float, lc_mpc: float = 1) -> float:
    """
    Deflection angle for IGMF in degrees.

    Parameters
    ----------
    R: float
        rigidity in EV
    beta_egmf: float
        EGMF magnetic field spread in nG Mpc^1/2
    D: float
        distance of the source in Mpc
    lc_mpc: float
        coherence length normalized to 1 Mpc
    """
    return (
        2.3
        * (50 * u.EV / R)
        * (beta_egmf / (1 * u.nG * u.Mpc**(1/2)))
        * np.sqrt(D / (10 * u.Mpc))
        * np.sqrt(lc_mpc)
    ) * u.deg

def theta_igmfs(Rs : np.ndarray, beta_egmf : float, D : float, lc_mpc : float=1) -> np.ndarray:
    """
    Deflection angle for IGMF in degrees.

    Parameters
    ----------
    Rs: np.ndarray
        grid of rigidities in EV
    beta_egmf: float
        EGMF magnetic field spread in nG Mpc^1/2
    D: float
        distance of the source in Mpc
    lc_mpc: float
        coherence length normalized to 1 Mpc
    """
    return np.array([theta_igmf(R, beta_egmf, D, lc_mpc).to_value(u.deg) for R in Rs])


def bounded_power_law(
    x: np.ndarray, alpha: float, xmin: float, xmax: float
) -> np.ndarray:
    """
    Bounded power law in both directions.

    Parameters
    ----------
    x: np.ndarray
        array of rigidities in EV
    alpha: float
        spectral index
    xmin: float
        minimum rigidity in EV
    xmax: float
        maximum rigidity in EV
    """
    if alpha != 1.0:
        norm = (1.0 - alpha) / (xmax ** (1.0 - alpha) - xmin ** (1.0 - alpha))
    else:
        norm = 1.0 / (np.log(xmax) - np.log(xmin))

    return norm * x ** (-alpha)

def vMF(x: np.array, mu: np.array, kappa: float) -> np.ndarray:
    """
    Return a vMF distribution.

    NB: shape of x must be (N, 3)

    Parameters
    ----------
    x: np.array
        array of cartesian coordinates
    mu: np.array
        array of cartesian coordinates for the mean direction
    kappa: float
        deflection parameter
    """
    if kappa > 100:
        return np.exp(
            kappa * np.dot(x.T, mu) + np.log(kappa) - np.log(4 * np.pi / 2) - kappa
        )
    elif kappa < 1e-5:  # L'Hopital's rule
        return (
            (1 + kappa * np.dot(x.T, mu))
            / (4 * np.pi * np.cosh(kappa))
            * np.exp(kappa * np.dot(x.T, mu))
        )
    else:
        return (
            kappa / (4 * np.pi * np.sinh(kappa)) * np.exp(kappa * np.dot(x.T, mu))
        )
    

def truncated_lognormal_sample(mu : float, sigma : float, a : float, b : float) -> np.ndarray:
    """
    Sample from a truncated lognormal distribution.

    Parameters
    ----------
    mu : float
        Mean of the underlying normal distribution.
    sigma : float
        Standard deviation of the underlying normal distribution.
    a, b : float
        Truncation bounds [a, b].
    """
    # Convert to scipy's lognorm parameters
    s = sigma                  # shape parameter (std dev in log space)
    scale = np.exp(mu)        # scale = exp(mu)

    # Get CDF values for truncation bounds
    lower_cdf = lognorm.cdf(a, s=s, scale=scale)
    upper_cdf = lognorm.cdf(b, s=s, scale=scale)

    # Sample uniformly in truncated CDF range
    u = np.random.uniform(lower_cdf, upper_cdf, size=1)

    # Invert CDF to get samples
    samples = lognorm.ppf(u, s=s, scale=scale)

    return samples

def truncated_lognorm_ccdf(x : np.ndarray, mu : float, sigma : float, a : float, b : float) -> float:
    """
    CCDF of a truncated lognormal distribution.

    Parameters
    ----------
    x : float or array
        Point(s) at which to evaluate the CCDF.
    s : float
        Shape parameter (sigma) of the underlying lognormal.
    scale : float, optional
        Scale parameter exp(mu). Default = 1.
    a, b : float, optional
        Truncation bounds [a, b]. Default is [0, inf).
    """
    # Convert to scipy's lognorm parameters
    s = sigma                  # shape parameter (std dev in log space)
    scale = np.exp(mu)        # scale = exp(mu)

    # Original lognormal distribution
    dist = lognorm(s=s, scale=scale)

    # Survival function of untruncated lognormal
    Sa = dist.sf(a)
    Sb = dist.sf(b)
    Sx = dist.sf(x)

    # Normalize to account for truncation
    return (Sx - Sb) / (Sa - Sb)

def source_spectrum(energy : np.ndarray, alpha : float, charge : float, Rmax : float = 1.7) -> np.ndarray:
    """
    Return the source spectrum.

    This returns the Auger source spectrum, given by:

    dN/dE ∝ E^(-alpha) * exp(1 - E / (charge * Rmax))
    
    Parameter
    ----------
    energy : np.ndarray
        Array of energies in EeV.
    alpha : float
        Spectral index.
    charge : float
        Charge of the nucleus.
    Rmax : float, optional
        Maximum rigidity in EV. Default is 1.7 EV.
    """
    Emax = charge * Rmax
    exp_cutoff = np.where(energy > Emax, np.exp(1 - energy / Emax), 1.0)
    return energy**-alpha * exp_cutoff