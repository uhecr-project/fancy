"""Calculate the weighted exposure, integrating the rigidity-dependent exposure with a given spectrum."""

from typing import Union

import astropy.units as u
import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from typing_extensions import Self, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from fancy import Data
from fancy.physics.effective_exposure import EffectiveExposure
from fancy.physics.energy_loss import EnergyLossModel
from fancy.utils.package_data import (
    get_path_to_exposure_tables,
)
from fancy.utils.helpers import truncated_lognorm_ccdf, source_spectrum


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
        self.src_spectrum_grid : np.ndarray = None
        self.mean_lnA_grid: np.ndarray = None
        self.var_lnA_grid: np.ndarray = None

        # indexing parameters
        self.Nmass_fracs = len(self.energy_loss_model.massids)
        self.Nalphas = len(self.energy_loss_model.alphas)
        self.Nbeta_egmfs = len(self.eff_exp.beta_egmf_grid)
        self.Nsrcs = self.data.source.N

        self.weighted_exposure: np.ndarray = None
        self.src_weighted_exposure: np.ndarray = None

    def initialise_grids(
        self: Self,
        energy_grid: np.ndarray,
        lnA_energy_grid: np.ndarray,
        energy_grid_widths: np.ndarray,
        var_lnA_min: float = 1e-12,
    ) -> None:
        """
        Initialise the grids, i.e. the energy, spectrum, and lnA grid.

        Parameters
        ----------
        energy_grid : np.ndarray
            Energy grid in EeV, shape (NEs,)
        lnA_energy_grid : np.ndarray
            Energy grid for the lnA calculation in EeV, shape (NElnAs,)
        energy_grid_widths : np.ndarray
            Widths of the energy grid in EeV, shape (NEs,)
        var_lnA_min : float, optional
            Minimum value for the variance of lnA to avoid numerical issues, by default 1e-6
        """
        self.energy_grid = energy_grid
        self.energy_grid_widths = energy_grid_widths
        self.lnA_energy_grid = lnA_energy_grid

        spectra, lnAs, src_spectra = self.energy_loss_model.compute_spectrum_and_lnA(
            egrid=self.energy_grid,
            egrid_lnA=self.lnA_energy_grid,
            egrid_widths=self.energy_grid_widths,
            compute_src=True
        )
        self.spectrum_grid = spectra
        self.src_spectrum_grid = src_spectra

        # mean and var lnA grid are evaluated at the energies of the energy grid
        self.mean_lnA_grid = lnAs[
            0, np.digitize(self.energy_grid, self.lnA_energy_grid, right=True) - 1, ...
        ]
        self.var_lnA_grid = lnAs[
            1, np.digitize(self.energy_grid, self.lnA_energy_grid, right=True) - 1, ...
        ]

        self.var_lnA_grid[self.var_lnA_grid < 0] = (
            var_lnA_min  # numerical issues can lead to negative variance
        )
        self.var_lnA_grid[np.isclose(self.var_lnA_grid, 0.0)] = (
            var_lnA_min  # avoid zero variance
        )

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

    def compute_rigidities(self: Self, Nsamples: int = 100) -> np.ndarray:
        """
        Compute the mean rigidities from the energies and lnA grids.

        Parameters
        ----------
        Nsamples : int, optional
            Number of samples to draw from the lnA distribution, by default 100

        Returns
        -------
        np.ndarray, np.ndarray
            Array of rigidities in EV
            Shape of (Nsamples, NEs, Nalphas, Nmass_fracs, Nsrcs + 1)
        """
        NEs = len(self.energy_grid)
        rigidity_samples = np.zeros(
            (Nsamples, NEs, self.Nalphas, self.Nmass_fracs, self.Nsrcs + 1)
        )

        rng = np.random.default_rng()
        for k in range(self.Nsrcs + 1):
            for i, ia, j in np.ndindex(NEs, self.Nalphas, self.Nmass_fracs):
                lnA_samples = rng.normal(
                    loc=self.mean_lnA_grid[i, ia, j, k],
                    scale=np.sqrt(self.var_lnA_grid[i, ia, j, k]),
                    size=Nsamples,
                )
                rigidity_samples[:, i, ia, j, k] = self.energy_grid[i] / (
                    0.5 * np.exp(lnA_samples)
                )

        return rigidity_samples

    def calculate_weighted_exposure(
        self: Self,
        Nsamples: Union[int, None] = 100,
        wexp_lim: Union[float, None] = 1e-40,
    ) -> np.ndarray:
        """
        One-shot approach to calculate the spectrum-weighted effective exposure from each source + background.

        Parameters
        ----------
        Nsamples : int, optional
            Number of samples to draw from the lnA distribution, by default 100
        wexp_lim : float, optional
            Minimum value for the weighted exposure to avoid numerical issues, by default 1e-10 km^2 yr
        """
        # get the mean rigidity from the energy & mean lnA grids
        rigidity_samples = self.compute_rigidities(
            Nsamples=Nsamples
        )  # shape (Nsamples, NEs, Nalphas, Nmass_fracs, Nsrcs + 1)

        # detector threshold efficiency calculation
        # P_Eths = self.calculate_threshold_prob()

        self.weighted_exposure = np.zeros(
            (self.Nsrcs + 1, self.Nalphas, self.Nbeta_egmfs, self.Nmass_fracs)
        )

        for k in range(self.Nsrcs + 1):
            # interpolate the effective exposure in rigidity
            f_effexp_logrig = CubicSpline(
                x=np.log(self.eff_exp.rigidity_grid.to_value(u.EV)),
                y=self.eff_exp.effective_exposure[k, :, :].to_value(u.km**2 * u.yr),
                axis=0,
            )
            for j in range(self.Nmass_fracs):
                log_rig = np.log(rigidity_samples[:, :, :, j, k])

                # here the weights are:
                # wexp(alpha, beta, massfrac) = int dE (dN/dE)(E, alpha, massfrac) * A_eff(E/Z, beta) * P_det(E)
                self.weighted_exposure[k, :, :, j] = np.mean(
                    np.trapz(
                        self.spectrum_grid[None, :, :, None, j, k]
                        * f_effexp_logrig(log_rig),
                        # * P_Eths[None, :, None, None],
                        x=self.energy_grid,
                        axis=1,
                    ),
                    axis=0,
                )


        # add limiters
        self.weighted_exposure[self.weighted_exposure < wexp_lim] = wexp_lim

        # add units now
        self.weighted_exposure *= u.km**2 * u.yr

        return self.weighted_exposure
    
    def calculate_src_weighted_exposure(
        self: Self,
        Nsamples: Union[int, None] = 100,
        wexp_lim: Union[float, None] = 1e-10,
    ) -> np.ndarray:
        """
        Calculate the weighted exposure from each source, considering the source spectrum instead of the observed spectrum.
        
        This is used to calculate the relative contributions of each source to the total flux, 
        as:
        Q_earth_k = Q_src_k * (wexp_src_k / wexp_earth_k) for each source k.

        Parameters
        ----------
        Nsamples : int, optional
            Number of samples to draw from the lnA distribution, by default 100
        wexp_lim : float, optional
            Minimum value for the weighted exposure to avoid numerical issues, by default 1e-10 km^2 yr
        """
        # get the mean rigidity from the energy & mean lnA grids
        rigidity_samples = self.compute_rigidities(
            Nsamples=Nsamples
        )  # shape (Nsamples, NEs, Nalphas, Nmass_fracs, Nsrcs + 1)

        # detector threshold efficiency calculation
        P_Eths = self.calculate_threshold_prob()

        self.src_weighted_exposure = np.zeros(
            (self.Nsrcs, self.Nalphas, self.Nbeta_egmfs, self.Nmass_fracs)
        )

        for k in range(self.Nsrcs): # only over actual sources, not background
            # interpolate the effective exposure in log(rigidity)
            f_effexp_logrig = CubicSpline(
                x=np.log(self.eff_exp.rigidity_grid.to_value(u.EV)),
                y=self.eff_exp.effective_exposure[k, :, :].to_value(u.km**2 * u.yr),
                axis=0,
            )
            for j in range(self.Nmass_fracs):
                log_rig = np.log(rigidity_samples[:, :, :, j, k])

                # here the weights are:
                # wexp(alpha, beta, massfrac) = int dE (dN/dE)(E, alpha, massfrac) * A_eff(E/Z, beta) * P_det(E)
                self.src_weighted_exposure[k, :, :, j] = np.mean(
                    np.trapz(
                        self.src_spectrum_grid[None, :, :, None, j, k]
                        * f_effexp_logrig(log_rig),
                        # * P_Eths[None, :, None, None],
                        x=self.energy_grid,
                        axis=1,
                    ),  # shape (Nsamples, Nalphas, Nbeta_egmfs)
                    axis=0,
                ) # shape (Nalphas, Nbeta_egmfs)


        # add limiters
        self.src_weighted_exposure[self.src_weighted_exposure < wexp_lim] = wexp_lim

        # add units now
        self.src_weighted_exposure *= u.km**2 * u.yr

        return self.src_weighted_exposure

    def save(self: Self, outfile: str = "weighted_exposures.h5") -> None:
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
                data=np.log10(
                    self.eff_exp.beta_egmf_grid.to_value(u.nG * u.Mpc**1 / 2)
                ),
            )
            config_gr.create_dataset("alphas", data=self.energy_loss_model.alphas)
            config_gr.create_dataset("mass_ids", data=self.energy_loss_model.massids)
            config_gr.create_dataset(
                "log_wexp_earth_grid",
                data=np.log(self.weighted_exposure.to_value(u.km**2 * u.yr)),
            )
            if self.src_weighted_exposure is not None:
                config_gr.create_dataset(
                    "log_wexp_src_grid",
                    data=np.log(self.src_weighted_exposure.to_value(u.km**2 * u.yr)),
                )

    def plot(
        self: Self,
        beta_val: float,
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
        beta = beta_val * u.nG * u.Mpc ** (1 / 2)
        fig, axs = plt.subplots(self.Nsrcs + 1, 1, figsize=(8, 5 * (self.Nsrcs + 1)))
        for k in range(self.Nsrcs + 1):
            for j in range(self.Nmass_fracs):
                wexp = self.weighted_exposure[k, :, :, j]
                wexp = wexp[
                    :, np.digitize(beta, self.eff_exp.beta_egmf_grid, right=True)
                ]
                axs[k].plot(
                    self.energy_loss_model.alphas,
                    wexp,
                    label=f"Z={self.energy_loss_model.Zs[j]}",
                )

            # axs[k].set_yscale("log")
            axs[k].set_xlabel(r"$\alpha$")
            axs[k].set_ylabel("Weighted exposure / km^2 yr sr")
            axs[k].legend()
            src_label = (
                "background"
                if k == self.Nsrcs
                else f"source at {self.data.source.distance[k]:.1f} Mpc"
            )
            axs[k].set_title(
                f"{self.detector_type}, {src_label}, {self.gmf_model}, beta={beta}, {k}"
            )

        plt.tight_layout()
        return fig, axs
