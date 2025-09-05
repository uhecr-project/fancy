"""Calculate the weighted exposure, integrating the rigidity-dependent exposure with a given spectrum."""

from typing import Union

import astropy.units as u
import h5py
import healpy
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import truncnorm
from astropy.coordinates import SkyCoord
from typing_extensions import Self, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from fancy import Data
from fancy.detector.exposure import m_dec
from fancy.physics.gmf import GMFLensing
from fancy.physics.effective_exposure import EffectiveExposure
from fancy.physics.energy_loss import EnergyLossModel
from fancy.utils.package_data import (
    get_path_to_exposure_tables,
)
from fancy.utils.helpers import theta_igmf, vMF, truncated_lognorm_ccdf


class WeightedExposure:
    """Calculates the exposure weights by integrating the rigidity-dependent exposure with a given spectrum."""

    def __init__(
        self: Self,
        data: Data,
        energy_loss_model: EnergyLossModel,
        eff_exp: EffectiveExposure,
    ) -> None:
        """
        Initialise the WeightedExposure class.

        Parameters
        ----------
        data : Data
            The Data object containing the detector information.
        energy_loss_model : EnergyLossModel
            The EnergyLossModel object for calculating energy losses.
            From this, we can access the spectrum and mean/var lnA.
        """
        self.data = data
        self.eff_exp = eff_exp
        self.energy_loss_model = energy_loss_model

        # labels
        self.source_type = self.data.source.label
        self.detector_type = self.data.detector.label
        self.gmf_model = self.eff_exp.gmf_model

        self.Nsrcs = self.data.source.N

        # set the objects from the external modules, assuming computation has already been performed
        if not energy_loss_model.solvers_loaded:
            raise ValueError(
                "EnergyLossModel solvers have not been loaded. Please run load_injection_solvers() first."
            )

        if eff_exp.effective_exposure == None:
            raise ValueError(
                "EffectiveExposure has not been computed. Please run compute_effective_exposure() first."
            )

        # energy loss parameters
        self.energy_grid: np.ndarray = None
        self.energy_grid_widths: np.ndarray = None
        self.lnA_energy_grid: np.ndarray = None
        self.spectrum_grid: np.ndarray = None
        self.mean_lnA_grid: np.ndarray = None
        self.var_lnA_grid: np.ndarray = None

        # indexing parameters
        self.Nmass_fracs = len(self.energy_loss_model.massids)
        self.Nalphas = len(self.energy_loss_model.alphas)
        self.Nbeta_egmfs = len(self.eff_exp.beta_egmf_grid)
        self.Nsrcs = self.data.source.N

        self.weighted_exposure: np.ndarray = None

    def initialise_grids(
        self: Self,
        energy_gridparams: tuple = (32, 500, 50),
        lnA_energy_gridparams: tuple = (3, 100, 50),
        var_lnA_min: float = 1e-6,
    ) -> None:
        """
        Initialise the grids, i.e. the energy, spectrum, and lnA grid.

        Parameters
        ----------
        energy_gridparams : tuple, optional
            Parameters for the energy grid as (min, max, num_points), by default (32, 500, 50)
        lnA_energy_gridparams : tuple, optional
            Parameters for the lnA grid as (min, max, num_points), by default (3, 100, 50)
        var_lnA_min : float, optional
            Minimum value for the variance of lnA to avoid numerical issues, by default 1e-6
        """
        energy_grid_binedges = np.logspace(
            np.log10(energy_gridparams[0]),
            np.log10(energy_gridparams[1]),
            energy_gridparams[2] + 1,
        )
        self.energy_grid = 10 ** np.sqrt(
            np.log10(energy_grid_binedges[:-1]) * np.log10(energy_grid_binedges[1:])
        )
        self.energy_grid_widths = np.diff(energy_grid_binedges)
        self.lnA_energy_grid = np.logspace(
            np.log10(lnA_energy_gridparams[0]),
            np.log10(lnA_energy_gridparams[1]),
            lnA_energy_gridparams[2],
        )

        spectra, lnAs = self.energy_loss_model.compute_spectrum_and_lnA(
            egrid=self.energy_grid,
            egrid_lnA=self.lnA_energy_grid,
            egrid_widths=self.energy_grid_widths,
        )
        self.spectrum_grid = spectra

        # mean and var lnA grid are evaluated at the energies of the energy grid
        self.mean_lnA_grid = CubicSpline(x=self.lnA_energy_grid, y=lnAs[0, ...], axis=0, extrapolate=False, bc_type='clamped')(
            self.energy_grid
        )
        self.var_lnA_grid = CubicSpline(x=self.lnA_energy_grid, y=lnAs[1, ...], axis=0, extrapolate=False, bc_type='clamped')(
            self.energy_grid
        )

        # clean up the grid for energy values outside the interpolation range
        # self.mean_lnA_grid[np.isnan(self.mean_lnA_grid)] = lnAs[0,-1,...]  # deal with Nans
        # self.var_lnA_grid[np.isnan(self.var_lnA_grid)] = lnAs[1,-1,...]  # deal with Nans

        self.var_lnA_grid[self.var_lnA_grid < 0] = var_lnA_min  # numerical issues can lead to negative variance
        self.var_lnA_grid[np.isclose(self.var_lnA_grid, 0.0)] = var_lnA_min  # avoid zero variance

    def calculate_threshold_prob(
        self: Self,
        Eth: Union[float, None] = None,
        logE_stat: Union[None, float] = None,
        logE_sys: float = 0,
    ) -> np.ndarray:
        """
        Calculate the CCDF of the detector response function CCDF(Edet >= Eth | Erec + logE_sys, sigma_E).

        This function is a truncated log-normal distribution, with the truncation at the minimum and maximum energies of the energy grid.

        Parameters
        ----------
        Eths : float
            Energy threshold in EeV, by default None (i.e. use the min of energy grid)
        logE_stat : Union[None, float], optional
            Log10 of the statistical energy resolution, by default None and reads the value from the Data object.
        logE_sys : Union[None, float], optional
            Log10 of the systematic energy resolution, by default 0 (i.e. no systematic shift)

        Returns
        -------
        np.ndarray
            The detector response function evaluated on the energy grid.
            Shape of (NEs,)
        """
        if self.energy_grid is None:
            raise ValueError(
                "Energy grid has not been initialised. Please run initialise_grids() first."
            )

        if Eth is None:
            Eth = np.min(self.energy_grid)

        if logE_stat is None:
            logE_stat = self.data.detector.energy_uncertainty

        P_Eth = np.array(
            [
                truncated_lognorm_ccdf(
                    x=Eth,
                    mu=np.log(E) + logE_sys,
                    sigma=logE_stat,
                    a=np.min(self.energy_grid),
                    b=np.max(self.energy_grid),
                )
                for E in self.energy_grid
            ]
        )  # shape (NEs,)

        return P_Eth

    def compute_mean_rigidities(self: Self) -> np.ndarray:
        """
        Compute the mean rigidities from the energies and lnA grids.

        We can also use the mean and variance to sample for lnA, but this is not necessary for the mean rigidity as its just an estimate. As such, we just use the mean lnA.

        Parameters
        ----------
        Nsamples : int, optional
            Number of samples to draw from the lnA distribution, by default 100

        Returns
        -------
        np.ndarray, np.ndarray
            Array of rigidities in EV
            Shape of (NEs, Nalphas, Nmass_fracs, Nsrcs + 1)
        """
        NEs = len(self.energy_grid)
        mean_rigidities = np.zeros(
            (NEs, self.Nalphas, self.Nmass_fracs, self.Nsrcs + 1)
        )

        for k in range(self.Nsrcs + 1):
            for i, ia, j in np.ndindex(NEs, self.Nalphas, self.Nmass_fracs):
                mean_rigidities[i, ia, j, k] = self.energy_grid[i] / (0.5 * np.exp(self.mean_lnA_grid[i, ia, j, k]))

        return mean_rigidities

    def calculate_weighted_exposure(self: Self) -> np.ndarray:
        """
        One-shot approach to calculate the spectrum-weighted effective exposure from each source + background.

        Parameters
        ----------
        Nsamples : int, optional
            Number of samples to draw from the lnA distribution, by default 100
        """
        # get the mean rigidity from the energy & mean lnA grids
        mean_rigidities = self.compute_mean_rigidities()

        # detector threshold efficiency calculation
        P_Eths = self.calculate_threshold_prob()

        self.wexp_earth_grid = np.zeros(
            (self.Nsrcs + 1, self.Nalphas, self.Nbeta_egmfs, self.Nmass_fracs)
        )

        for k in range(self.Nsrcs + 1):
            # interpolate the effective exposure in rigidity
            f_effexp_rig = CubicSpline(
                x=self.eff_exp.rigidity_grid,
                y=self.eff_exp.effective_exposure[k, :, :],
                axis=0,
            )
            for j in range(self.Nmass_fracs):
                rig = mean_rigidities[:, :, j, k]

                # here the weights are:
                # wexp(alpha, beta, massfrac) = int dE (dN/dE)(E, alpha, massfrac) * A_eff(E/Z, beta) * P_det(E)
                self.wexp_earth_grid[k, :, :, j] = np.trapz(
                        self.spectrum_grid[:, :, None, j, k]
                        * f_effexp_rig(rig)
                        * P_Eths[:, None, None],
                        x=self.energy_grid,
                        axis=0,
                    )

        return self.wexp_earth_grid
    
    def save(
        self : Self,
        outfile: str = "weighted_exposures.h5") -> None:
        """
        Save tabulated results to h5py File.

        Parameter:
        ----------
        outfile : str
            the path to the output file. must be in .h5 format.
        """
        assert outfile.find(".h5") > 0, (
            f"Output file {outfile} needs to have a .h5 extension."
        )
        with h5py.File(str(get_path_to_exposure_tables(outfile)), "a") as f:
            config_label = f"{self.source_type}_{self.detector_type}_{self.gmf_model}"
            if config_label in f.keys():
                del f[config_label]
            config_gr = f.create_group(config_label)

            config_gr.create_dataset("source_distances", data=self.data.source.distance)
            config_gr.create_dataset(
                "log10_beta_egmf_grid",
                data=np.log10(self.eff_exp.beta_egmf_grid.to_value(u.nG * u.Mpc**1 / 2))
            )
            config_gr.create_dataset(
                "alphas", data=self.energy_loss_model.alphas
            )
            config_gr.create_dataset(
                "mass_ids", data=self.energy_loss_model.massids
            )
            config_gr.create_dataset(
                "log10_wexp_earth_grid",
                data=np.log10(self.wexp_earth_grid.to_value(u.km**2 * u.yr)),
            )

    def plot(
        self : Self,
        beta_val : float,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Small shortcut to plot the weighted exposure for a given beta value.

        For more rigorous plotting, please extract the arrays and plot them yourself.

        Parameters
        ----------
        beta_val : float
            The beta value to plot.
            Should just be a float, not with units. Assumed to be in nG Mpc^(1/2).
        """
        beta = beta_val * u.nG * u.Mpc**(1/2)
        fig, axs = plt.subplots(self.Nsrcs + 1, 1, figsize=(8, 5 * (self.Nsrcs + 1)))
        for k in range(self.Nsrcs + 1):
            for j in range(self.Nmass_fracs):
                wexp = self.wexp_earth_grid[k, :, :, j]
                wexp = wexp[:,np.digitize(beta, self.eff_exp.beta_egmf_grid, right=True)]
                axs[k].plot(self.energy_loss_model.alphas, wexp, label=f"Z={self.energy_loss_model.Zs[j]}")

            # axs[k].set_yscale("log")
            axs[k].set_xlabel(r"$\alpha$")
            axs[k].set_ylabel("Weighted exposure / km^2 yr sr")
            axs[k].legend()
            src_label = "background" if k == self.Nsrcs else f"source at {self.data.source.distance[k]:.1f} Mpc"
            axs[k].set_title(f"{self.detector_type}, {src_label}, {self.gmf_model}, beta={beta}, {k}")

        plt.tight_layout()
        return fig, axs