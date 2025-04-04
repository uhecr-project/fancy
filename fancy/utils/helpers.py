"""Helper functions for the code."""

import numpy as np
import astropy.units as u


def theta_igmf(R: float, Bigmf: float, D: float, lc: float = 1) -> float:
    """
    Deflection angle for IGMF in degrees.

    Parameters
    ----------
    R: float
        rigidity in EV
    Bigmf: float
        IGMF magnetic field strength in nG
    D: float
        distance of the source in Mpc
    lc: float
         coherence length in Mpc (default 1 Mpc)
    """
    return (
        2.3
        * (50 * u.EV / R)
        * (Bigmf / (1 * u.nG))
        * np.sqrt(D / (10 * u.Mpc))
        * np.sqrt(lc)
    ) * u.deg

def theta_igmfs(Rs : np.ndarray, Bigmf : float, D : float, lc : float=1) -> np.ndarray:
    return np.array([theta_igmf(R, Bigmf, D, lc).to_value(u.deg) for R in Rs])


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