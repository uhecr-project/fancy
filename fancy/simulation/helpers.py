"""Helper functions for simulation generation."""

import pickle as pickle
import datetime
from astropy.time import Time
from astropy.coordinates import AltAz
from typing_extensions import Union, Tuple

from astropy.coordinates import SkyCoord
import numpy as np
from scipy.stats import truncnorm
from fancy.utils.helpers import truncated_lognormal_sample
from fancy.detector.exposure import m_dec

from vMF import sample_vMF

def get_Edet(log_Etrue : np.ndarray, en_unc : float, Eth : float, Emax : float) -> np.ndarray:
    """Get the detected energies given true energy and energy uncertainty."""
    # assuming Gaussian uncertainty
    Edet = truncated_lognormal_sample(log_Etrue, sigma=en_unc, a=Eth, b=Emax)
    return Edet


def get_mean_lnA_det(mean_lnA_true : np.ndarray, mean_lnA_unc : Union[np.ndarray, float]=2) -> np.ndarray:
    """Get the detected mean lnA given true mean lnA and uncertainty."""
    # assuming Gaussian uncertainty
    a = -(mean_lnA_true) / mean_lnA_unc
    b = np.inf
    mean_lnA_det = truncnorm.rvs(a, b, loc=mean_lnA_true, scale=mean_lnA_unc, size=1)
    return mean_lnA_det


def get_var_lnA_det(var_lnA_true : np.ndarray, var_lnA_unc: Union[np.ndarray, float]=0.5) -> np.ndarray:
    """Get the detected var lnA given true var lnA and uncertainty."""
    # assuming Gaussian uncertainty
    a = (-2 - (var_lnA_true)) / var_lnA_unc
    b = np.inf
    var_lnA_det = truncnorm.rvs(a, b, loc=var_lnA_true, scale=var_lnA_unc, size=1)
    return var_lnA_det

def get_direction_acceptance(coord_earth : np.ndarray, kappa_det: float, detector_params : tuple, max_exposure : float) -> Tuple[int, SkyCoord, float]:
    """
    Return if the direction is accepted based on the detector's exposure.

    Parameters
    ----------
    coord_earth : np.ndarray
        The coordinates of the UHECR in the Earth frame. 
        Must be in cartersian coordinates (x, y, z).
    kappa_det : float
        The concentration parameter for the von Mises-Fisher distribution.
        This quantifies the angular uncertainty of the detector.
    
    Returns
    -------
    accept : int
        1 if the direction is accepted, 0 otherwise.
    reconstr_uv : SkyCoord
        The reconstructed coordinates of the UHECR in the Galactic frame.
    m_omega : float
        The exposure function evaluated at the reconstructed declination.
        Used to calculate the exposure factor per UHECR.
    """
    rng_det = np.random.default_rng()

    reconstr_uv = sample_vMF(
        coord_earth, kappa_det, num_samples=1
    ).T.squeeze()
    reconstr_uv /= np.linalg.norm(reconstr_uv)
    reconst_coord = SkyCoord(
        *reconstr_uv, representation_type="cartesian", frame="galactic"
    )
    reconst_coord.transform_to("icrs")

    # evaluate exposure function at that declination -> construct pdet
    m_omega = m_dec(
        reconst_coord.icrs.dec.rad, detector_params
    )
    pdet = m_omega / max_exposure
    accept = rng_det.choice(
        [1, 0], p=[pdet, 1 - pdet]
    )  # use binomial distribution to sample that UHECR

    reconst_coord.transform_to("galactic")
    reconst_coord.representation_type = "unitspherical"

    return accept, reconst_coord, m_omega

charge_massid_map = {
    101 : 1,
    402 : 2,
    1407 : 7,
    2814 : 14,
    5626 : 26
}

def source_spectrum(energy, alpha, charge, Rmax = 1.7):
    """Return the source spectrum."""
    Emax = charge * Rmax
    exp_cutoff = np.where(energy > Emax, np.exp(1 - energy / Emax), 1.0)
    return energy**-alpha * exp_cutoff

# convert starting period to decimal year
def get_fractional_year_from_date(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year + 1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

def simulate_zenith_angles( c_icrs, zen_thresh = None, period_start = None, location = None):
    """Simulate zenith angles, using ICRS SKyCoord"""
    years = []
    days = []
    times = []
    zenith_angles = []
    stuck = []

    k = 0
    first = True
    for d in c_icrs:
        za = 99
        i = 0
        while za > zen_thresh:
            dt = np.random.exponential(1.0 / len(c_icrs))
            if first:
                t = get_fractional_year_from_date(period_start) + dt
            else:
                t = times[-1] + dt
            tdy = Time(t, format="decimalyear")
            c_altaz = d.transform_to(
                AltAz(obstime=tdy, location=location)
            )
            za = np.pi / 2 - c_altaz.alt.rad

            i += 1
            if i > 100:
                za = zen_thresh
                stuck.append(1)

        # convert decimal years to year & days
        year, year_frac = divmod(t, 1)
        day = np.round(year_frac * 365.2425)  # round to nearest even value
        years.append(int(year))
        days.append(int(day))

        # append time also for the algorithm
        times.append(t)
        first = False
        zenith_angles.append(za)
        k += 1
        # print(j , za)

    if len(stuck) > 1:
        print(
            "Warning: % of zenith angles stuck is",
            len(stuck) / len(zenith_angles) * 100,
        )

    return np.array(zenith_angles), np.array(years), np.array(days)