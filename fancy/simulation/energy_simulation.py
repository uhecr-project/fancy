"""Class to manage simulation of energy + mass model."""

import os
import pickle as pickle
import tempfile

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import truncnorm
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from typing_extensions import ClassVar, List, Self, Tuple, Union

from fancy import Data
from fancy.physics import EnergyLossModel
from fancy.utils.helpers import km_per_Mpc, truncated_lognormal_sample
from fancy.simulation.helpers import (
    get_Edet,
    get_mean_lnA_det,
    get_var_lnA_det,
    source_spectrum,
)

charge_massid_map = {101: 1, 402: 2, 1407: 7, 2814: 14, 5626: 26}


class EnergySimulation:
    """Handles the generation of simulation samples."""

    def __init__(
        self,
        data: Data,
    ) -> None:
        """
        Handle the generation of simulation samples.

        data: fancy.interfaces.Data
            Data object from fancy
        energy_loss_table_file: str
            file for energy tables
        exposure_table_file: str
            file where exposure tables are contained
        gmf_model : str, default="None"
            the GMF model to use when generating the simulations.
            Default to None, i.e. we dont perform GMF lensing
        """
        self.detector_type = data.detector.label
        self.mass_model = data.detector.mass_model
        self.source_type = data.source.label

        # source parameters
        self.Nsrcs = data.source.N

        # data object that encompasses detector & source information
        self.data = data

        # other objects we store for later
        self.energy_grid = None
        self.lnA_energy_grid = None
        self.alpha_grid = None
        self.mass_ids_grid = None
        self.truths = {}
        self.config = {}

        # injection solver results
        self.spectrum_grid = None
        self.mean_lnA_grid = None
        self.var_lnA_grid = None
        self.detection_rates_grid = None

        # shape parameters
        self.NEs = 0
        self.NElnAs = 0
        self.Nalphas = 0
        self.Nmass_fracs = 0

    def initialise_grids(
        self: Self,
        energy_grid: np.ndarray,
        energy_grid_widths: np.ndarray,
        lnA_energy_grid: np.ndarray,
        src_inj_kwargs: dict = {
            "dinits": [4],
            "Rmax": 1.7,
        },
        bg_inj_kwargs: dict = {
            "z_max": 3.0,
            "source_evo": "SFR",
            "Rmax": 1.7,
        },
        energy_loss_model_kwargs: dict = {},
    ) -> None:
        """
        Initialise the energy and lnA grids from the injection solver results from Prince.

        The energies here are default to be in EV. The src and bg_inj_kwargs are used to read off
        the source and background injection solver results from the Prince run. Modify them as you see fit.

        Parameters
        ----------
        energy_grid : np.ndarray
            the energy grid to use for the simulation
        lnA_energy_grid : np.ndarray
            the energy grid for lnA to use for the simulation
        src_inj_kwargs : dict, optional
            keyword arguments for the source injection solver, by default {"dinits": [4], "Rmax": 1.7}
            TODO: add some option where the source label is used automatically / has a choice to do so.
        bg_inj_kwargs : dict, optional
            keyword arguments for the background injection solver, by default {"z_max": 3.0, "source_evo": "SFR", "Rmax": 1.7}
        energy_loss_model_kwargs : dict, optional
            keyword arguments for the energy loss model, by default {}.
            What you can enter here is like the cross section model, alpha grid, mass IDs used
        """
        self.energy_grid = energy_grid
        self.energy_grid_widths = energy_grid_widths
        self.lnA_energy_grid = lnA_energy_grid

        # assert that the dinit is set correctly to the data.source model
        if "dinits" not in src_inj_kwargs:
            src_inj_kwargs["dinits"] = self.data.source.label
            print(f"Using dinit={src_inj_kwargs['dinits']} from data.source.label")

        # TODO: some way to check the dinit
        # else:
        #     if src_inj_kwargs["dinit"] != self.data.source.label:
        #         raise ValueError(f"dinit={src_inj_kwargs['dinit']} does not match data.source.label={self.data.source.label}")

        # initalise the energy loss model
        energy_loss_model = EnergyLossModel(**energy_loss_model_kwargs)
        energy_loss_model.load_injection_solvers(
            src_inj_config=src_inj_kwargs, bg_inj_config=bg_inj_kwargs
        )

        # store the grids for later use
        spectra, lnAs = energy_loss_model.compute_spectrum_and_lnA(
            egrid=self.energy_grid,
            egrid_lnA=self.lnA_energy_grid,
            egrid_widths=self.energy_grid_widths,
        )

        # store the injection solver results
        # NB: shapes are in (Ngrid, Nalphas, Nmass_fracs, Nsrcs)
        self.spectrum_grid = spectra
        self.mean_lnA_grid = lnAs[0, ...]
        self.var_lnA_grid = lnAs[1, ...]
        self.alpha_grid = energy_loss_model.alphas
        self.mass_ids_grid = energy_loss_model.massids
        charges = np.array([charge_massid_map[massid] for massid in self.mass_ids_grid])

        # store the shape parameters
        self.NEs = self.spectrum_grid.shape[0]
        self.Nalphas = self.spectrum_grid.shape[1]
        self.Nmass_fracs = self.spectrum_grid.shape[2]
        self.NElnAs = self.mean_lnA_grid.shape[0]

        # also compute the detection rate grid here
        self.src_spectrum_grid = source_spectrum(
            self.energy_grid[:, np.newaxis, np.newaxis, np.newaxis],
            self.alpha_grid[np.newaxis, :, np.newaxis, np.newaxis],
            charges[np.newaxis, np.newaxis, :, np.newaxis],
            Rmax=src_inj_kwargs["Rmax"],
        )
        # this gives the fraction of detected events for a given source
        # at given alpha for each mass ID
        self.detection_rates_grid = np.trapz(
            self.spectrum_grid, x=self.energy_grid, axis=0
        )
        self.src_detection_rates_grid = np.trapz(
            self.src_spectrum_grid, x=self.energy_grid, axis=0
        )

        self.esrc_ratio_grid = np.trapz(
            y=self.energy_grid[:, None, None, None] * self.src_spectrum_grid,
            x=self.energy_grid,
            axis=0,
        ) / np.trapz(y=self.src_spectrum_grid, x=self.energy_grid, axis=0)

        # store all these in the config dictionary
        self.config["energy_grid"] = self.energy_grid
        self.config["energy_grid_widths"] = self.energy_grid_widths
        self.config["lnA_energy_grid"] = self.lnA_energy_grid
        self.config["spectrum_grid"] = self.spectrum_grid
        self.config["mean_lnA_grid"] = self.mean_lnA_grid
        self.config["var_lnA_grid"] = self.var_lnA_grid
        self.config["detection_rates_grid"] = self.detection_rates_grid
        self.config["src_spectrum_grid"] = self.src_spectrum_grid
        self.config["esrc_ratio_grid"] = self.esrc_ratio_grid
        self.config["alpha_grid"] = self.alpha_grid
        self.config["mass_ids_grid"] = self.mass_ids_grid
        self.config["Nsrcs"] = self.Nsrcs
        self.config["NEs"] = self.NEs
        self.config["NElnAs"] = self.NElnAs
        self.config["Nalphas"] = self.Nalphas
        self.config["Nmass_fracs"] = self.Nmass_fracs

    def set_truths(
        self: Self,
        mass_fracs: np.ndarray,
        alphas: np.ndarray,
        source_fraction: float = 0.5,
        Lsrcs: Union[np.ndarray, None] = None,
        Nex: Union[int, None] = None,
    ) -> dict:
        """
        Set the truth values for the simulation.

        Parameters
        ----------
        mass_fracs : np.ndarray
            the mass fractions for the sources
        alphas : np.ndarray
            the spectral indices for the sources
        source_fraction : float, optional
            fraction of sources to use, by default 0.5
        Lsrc: np.ndarray, optional, default = None
            luminosity of the sources, by default None
            If None, then Nex must be provided.
        Nex : int, optional
            total number of expected events in the simulation, by default None
            If None, then Lsrcs must be provided.

        Returns
        -------
        dict
            dictionary with truth values for the simulation.
        """
        # ensure that we have the correct shape for the mass fractions and alphas
        if mass_fracs.shape[0] != self.Nmass_fracs:
            raise ValueError(f"mass_fracs must have shape {self.Nmass_fracs}")

        if (mass_fracs.shape[1] != self.Nsrcs + 1) or (
            alphas.shape[0] != self.Nsrcs + 1
        ):
            raise ValueError(
                f"mass_fracs and alphas must have shape {self.Nsrcs + 1} in second axis"
            )

        if (Lsrcs is None) and (Nex is None):
            raise ValueError(
                "Either Lsrcs or Nex must be provided. Both cannot be None."
            )
        if (Lsrcs is not None) and (Nex is not None):
            raise ValueError(
                "Either Lsrcs or Nex must be provided. Both cannot be provided."
            )

        # set the truth values based on the input parameters
        fit_truths = {
            "alphas": alphas,
            "mass_fracs": mass_fracs,
            "Lsrcs": Lsrcs,
            "Nex": Nex,
            "src_frac": source_fraction,
        }

        fit_truths = self.__compute_flux_truths(fit_truths)

        # set it as an object
        self.truths = fit_truths

        return fit_truths

    def __compute_flux_truths(self: Self, truths: dict) -> dict:
        """
        Compute the truths related to the fluxes and store this in the truths dictionary.

        Parameters
        ----------
        truths : dict
            dictionary with truth values for the simulation.
        """
        alpha_T = self.data.detector.alpha_T

        # extending the mass fractions to include alpha axis
        # summing over mass fraction axis here.
        earth_flux_det_rate = np.sum(
            self.detection_rates_grid
            * truths["mass_fracs"][np.newaxis, :, :]
            * alpha_T,
            axis=1,
        )
        f_earth_flux_det = CubicSpline(self.alpha_grid, earth_flux_det_rate, axis=0)
        earth_flux_det_rate = np.array(
            [f_earth_flux_det(truths["alphas"][k])[k] for k in range(self.Nsrcs)]
        )

        src_flux_det_rate = np.sum(
            self.src_detection_rates_grid
            * truths["mass_fracs"][np.newaxis, :, :]
            * alpha_T,
            axis=1,
        )
        f_src_flux_det = CubicSpline(self.alpha_grid, src_flux_det_rate, axis=0)
        src_flux_det_rate = np.array(
            [f_src_flux_det(truths["alphas"][k])[k] for k in range(self.Nsrcs)]
        )

        # calculate conversion from L -> Q
        f_en_ratio = CubicSpline(
            self.alpha_grid,
            np.sum(
                self.esrc_ratio_grid * truths["mass_fracs"][np.newaxis, :, :], axis=1
            ),
            axis=0,
        )

        # now calculate truths based on if we have Nex or Lsrcs or not
        if truths["Nex"] is not None:
            Nex = truths["Nex"]
            Nex_src = int(truths["Nex"] * truths["src_frac"])
            Nex_bg = truths["Nex"] - Nex_src

            # here: calculate the total flux from the source at Earth
            Fearth_tot = Nex_src / alpha_T

            # then calcualte the particle rate by multiplying by distance factor
            Qearths_truths = Fearth_tot * (
                4 * np.pi * (self.data.source.distance * km_per_Mpc) ** 2
            )

            # convert to source particle rate
            Qsrcs_truths = Qearths_truths * src_flux_det_rate / earth_flux_det_rate
            Fsrcs_truths = Qsrcs_truths / (
                4 * np.pi * (self.data.source.distance * km_per_Mpc) ** 2
            )

            # now we can calculate the relative contribution of each source to the flux at Earth
            Fearths_truths = Qearths_truths / (
                4 * np.pi * (self.data.source.distance * km_per_Mpc) ** 2
            )

            # luminosity simply calculated via multiplying with Eex
            Lsrcs = Qsrcs_truths * np.array(
                [f_en_ratio(truths["alphas"][k])[k] for k in range(self.Nsrcs)]
            )

            truths["Lsrcs"] = Lsrcs
            truths["log10_Lsrcs"] = np.log10(Lsrcs)
            truths["Nex_src"] = Nex_src
            truths["Nex_bg"] = Nex_bg

        elif truths["Lsrcs"] is not None:
            Qsrcs_truths = truths["Lsrcs"] / np.array(
                [f_en_ratio(truths["alphas"][k])[k] for k in range(self.Nsrcs)]
            )
            Qearths_truths = Qsrcs_truths * earth_flux_det_rate / src_flux_det_rate

            Fsrcs_truths = np.zeros(self.Nsrcs)  # excluding the background source
            Fearths_truths = np.zeros(self.Nsrcs)  # including the background source
            Nex_src = 0.0
            for k in range(self.Nsrcs):
                Fsrcs_truths[k] = Qsrcs_truths[k] / (
                    4 * np.pi * (self.data.source.distance[k] * km_per_Mpc) ** 2
                )
                Fearths_truths[k] = Qearths_truths[k] / (
                    4 * np.pi * (self.data.source.distance[k] * km_per_Mpc) ** 2
                )
                # Nex_src += Fearths_truths[k] * Nex_per_flux[k]
                Nex_src += Fearths_truths[k] * alpha_T

            Nex_src = int(Nex_src)
            Nex = int(Nex_src / truths["src_frac"])
            Nex_bg = Nex - Nex_src

            truths["Nex"] = Nex
            truths["Nex_src"] = Nex_src
            truths["Nex_bg"] = Nex_bg
            truths["log10_Lsrcs"] = np.log10(truths["Lsrcs"])
        else:
            raise ValueError(
                "Either Nex or Lsrcs must be provided in the truths dictionary."
            )

        print(f"Total number of expected events: {Nex} (src: {Nex_src}, bg: {Nex_bg})")

        truths["Qsrcs"] = Qsrcs_truths
        truths["Qearths"] = Qearths_truths
        truths["Fsrcs"] = Fsrcs_truths
        truths["log10_Fsrcs"] = np.log10(Fsrcs_truths)

        truths["Fearths"] = Fearths_truths
        truths["log10_Fearths"] = np.log10(Fearths_truths)

        truths["F0"] = Nex_bg / alpha_T
        truths["log10_F0"] = np.log10(truths["F0"])

        truths["Ftot"] = np.sum(Fearths_truths) + truths["F0"]
        truths["log10_Ftot"] = np.log10(truths["Ftot"])

        truths["Nex_per_src"] = (
            np.concatenate([Fearths_truths, [truths["F0"]]]).T * alpha_T
        ).astype(int)

        # double check that the calculation makes sense
        # Nex_expected = np.sum((Fearths_truths + truths["F0"]) * alpha_T)

        return truths

    def generate_samples(
        self: Self, seed: Union[int, None] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate samples (energy truths, lnA truths) from the simulation.

        Parameters
        ----------
        seed : int or None, optional
            random seed for reproducibility, by default None
        """
        rng = np.random.default_rng(seed=seed)

        Etruths = np.zeros(self.truths["Nex"])
        mean_lnA_truths = np.zeros(self.NElnAs)
        var_lnA_truths = np.zeros(self.NElnAs)

        # calculatig mass fraction weighted values
        energy_spect_mf = np.sum(
            self.spectrum_grid
            * self.truths["mass_fracs"][np.newaxis, np.newaxis, :, :],
            axis=2,
        )
        mean_lnA_mfs = np.sum(
            self.mean_lnA_grid
            * self.truths["mass_fracs"][np.newaxis, np.newaxis, :, :],
            axis=2,
        )
        var_lnA_mfs = np.sum(
            self.var_lnA_grid * self.truths["mass_fracs"][np.newaxis, np.newaxis, :, :],
            axis=2,
        )

        # create an 2-D interpolation grid
        f_log_espect = RegularGridInterpolator(
            (np.log10(self.energy_grid), self.alpha_grid), np.log(energy_spect_mf)
        )

        f_mulnA = CubicSpline(y=mean_lnA_mfs, x=self.alpha_grid, axis=1)
        f_varlnA = CubicSpline(y=var_lnA_mfs, x=self.alpha_grid, axis=1)

        N_prev_idx = 0

        for k in range(self.Nsrcs + 1):  # +1 for the background source
            alpha_truth = self.truths["alphas"][k]
            en_spect = np.exp(
                f_log_espect((np.log10(self.energy_grid), alpha_truth))[:, k]
            )
            en_prob = (en_spect * self.energy_grid_widths) / np.sum(
                en_spect * self.energy_grid_widths
            )

            Nex_per_src = self.truths["Nex_per_src"][k]
            N_next_idx = Nex_per_src + N_prev_idx
            en_samples_src = rng.choice(self.energy_grid, size=Nex_per_src, p=en_prob)
            Etruths[N_prev_idx:N_next_idx] = en_samples_src

            N_prev_idx = Nex_per_src

            # ideally this would be calcualted via number of expected events per flux * flux
            # which would be evaluated per distance
            # but since we just have a two-source model, we just use the source fraction
            # for sake of conveninencec
            # fac = self.truths['src_frac'] if k < len(distances) else (1 - self.truths['src_frac'])
            mean_lnA_truths += (
                Nex_per_src * f_mulnA(alpha_truth)[:, k] / self.truths["Nex"]
            )
            var_lnA_truths += (
                Nex_per_src * f_varlnA(alpha_truth)[:, k] / self.truths["Nex"]
            )

        # now we have the energy truths, mean lnA truths and var lnA truths
        self.truths["Etruths"] = Etruths
        self.truths["mean_lnA_truths"] = mean_lnA_truths
        self.truths["var_lnA_truths"] = var_lnA_truths

        return Etruths, mean_lnA_truths, var_lnA_truths

    def apply_detector_response(
        self: Self,
        mean_lnA_stat: Union[float, np.ndarray],
        var_lnA_stat: Union[float, np.ndarray],
        logE_stat: Union[float, None] = None,
        mean_lnA_sys: float = 0.0,
        var_lnA_sys: float = 0.0,
        logE_sys: float = 0.0,
    ) -> None:
        """
        Apply the detector response to the simulation truths.

        Parameters
        ----------
        mean_lnA_stat : Union[np.ndarray, float]
            uncertainty in the mean lnA values. If float, assumes a universal value for
            all bins, otherwise takes in a different value for each bin.
            Shape should be (NlnA_bins,)
        var_lnA_stat : Union[np.ndarray, float]
            uncertainty in the variance of lnA values. If float, assumes a universal value for
            all bins, otherwise takes in a different value for each bin.
            Shape should be (NlnA_bins,)
        energy_unc : float
            uncertainty in the logarithm of energy values, in percentage of the truth.
            If None, then uses the energy uncertainty reported in data.detector.f_E.
        mean_lnA_sys : float, optional
            global systematic uncertainty in the mean lnA values, by default 0.0
        var_lnA_sys : float, optional
            global systematic uncertainty in the variance of lnA values, by default 0.0
        logE_sys : float, optional
            global systematic uncertainty in the logarithm of energy values, by default 0.0
        """
        if isinstance(mean_lnA_stat, float):
            mean_lnA_stat = np.full(self.NElnAs, mean_lnA_stat)

        if isinstance(var_lnA_stat, float):
            var_lnA_stat = np.full(self.NElnAs, var_lnA_stat)

        # if None then use the energy uncertainty reported in
        # data.detector
        if logE_stat is None:
            logE_stat = self.data.detector.f_E

        # apply the uncertainties to the truths
        Edets = np.array(
            [
                get_Edet(
                    np.log(en) + logE_sys,
                    en_unc=logE_stat,
                    Eth=np.min(self.energy_grid),
                    Emax=np.max(self.energy_grid),
                )
                for en in self.truths["Etruths"]
            ]
        ).flatten()
        mean_lnA_dets = np.array(
            [
                get_mean_lnA_det(
                    mean_lnA + mean_lnA_sys,
                    mean_lnA_unc=mean_lnA_stat[ibin],
                )
                for ibin, mean_lnA in enumerate(self.truths["mean_lnA_truths"])
            ]
        ).flatten()
        var_lnA_dets = np.array(
            [
                get_var_lnA_det(
                    var_lnA + var_lnA_sys,
                    var_lnA_unc=var_lnA_stat[ibin],
                )
                for ibin, var_lnA in enumerate(self.truths["var_lnA_truths"])
            ]
        ).flatten()

        # store the detected values in the truths dictionary
        self.truths["Edets"] = Edets
        self.truths["mean_lnA_dets"] = mean_lnA_dets
        self.truths["var_lnA_dets"] = var_lnA_dets

        # also set the uncertainties here
        self.config["mean_lnA_stat"] = mean_lnA_stat
        self.config["var_lnA_stat"] = var_lnA_stat
        self.config["logE_stat"] = logE_stat
        self.config["mean_lnA_sys"] = mean_lnA_sys
        self.config["var_lnA_sys"] = var_lnA_sys
        self.config["logE_sys"] = logE_sys

        return Edets, mean_lnA_dets, var_lnA_dets

    def save_truths(self: Self, filename: str = None) -> None:
        """
        Save the truths to a file.

        Parameters
        ----------
        filename : str, optional
            filename to save the truths to, by default None.
            If None, a temporary file will be created.
        """
        if filename is None:
            # create a temporary file
            fd, filename = tempfile.mkstemp(suffix=".pkl")
            os.close(fd)

        # save the truths to the file
        with open(filename, "wb") as f:
            pickle.dump(self.truths, f)

        print(f"Saved truths to {filename}")

    def plot_samples(self: Self) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """
        Plot the energy spectrum and lnA distributions of the samples.

        This will plot the energy spectrum and lnA distributions of the samples.
        It will also plot the total energy spectrum and lnA distributions.

        This function is meant to be a quick way to check if the code is doing the
        right thing. If you want something more advanced then use the outputs directly.
        """
        fig, axs = plt.subplots(
            3, 1, gridspec_kw={"height_ratios": [3, 1, 1]}, figsize=(10, 10)
        )

        dis_labels = ["src", "bg"]
        dis_lss = ["--", ":"]

        tot_espect = np.zeros_like(self.energy_grid)

        for k in range(self.Nsrcs + 1):
            alpha_idx = np.digitize(self.truths["alphas"][k], self.alpha_grid) - 1
            for ims, massid in enumerate(self.mass_ids_grid):
                # get the energy grid
                axs[0].loglog(
                    self.energy_grid,
                    self.spectrum_grid[:, alpha_idx, ims, k]
                    * self.truths["mass_fracs"][ims, k],
                    color=f"C{ims}",
                    ls=dis_lss[k],
                    label=f"{massid}, {dis_labels[k]}",
                )

                # get the lnA grid
                axs[1].semilogx(
                    self.lnA_energy_grid,
                    self.mean_lnA_grid[:, alpha_idx, ims, k]
                    * self.truths["mass_fracs"][ims, k],
                    color=f"C{ims}",
                    ls=dis_lss[k],
                    label=f"{massid}, {dis_labels[k]}",
                )
                axs[2].semilogx(
                    self.lnA_energy_grid,
                    self.var_lnA_grid[:, alpha_idx, ims, k]
                    * self.truths["mass_fracs"][ims, k],
                    color=f"C{ims}",
                    ls=dis_lss[k],
                    label=f"{massid}, {dis_labels[k]}",
                )

            axs[1].semilogx(
                self.lnA_energy_grid,
                np.sum(
                    self.mean_lnA_grid[:, alpha_idx, :, k]
                    * self.truths["mass_fracs"][None, :, k],
                    axis=1,
                ),
                color="k",
                ls=dis_lss[k],
                lw=2,
                label=f"total, {dis_labels[k]}",
            )
            axs[2].semilogx(
                self.lnA_energy_grid,
                np.sum(
                    self.var_lnA_grid[:, alpha_idx, :, k]
                    * self.truths["mass_fracs"][None, :, k],
                    axis=1,
                ),
                color="k",
                ls=dis_lss[k],
                lw=2,
                label=f"total, {dis_labels[k]}",
            )

            tot_espect_per_d = np.sum(
                self.spectrum_grid[:, alpha_idx, :, k]
                * self.truths["mass_fracs"][np.newaxis, :, k],
                axis=-1,
            )

            axs[0].loglog(
                self.energy_grid,
                tot_espect_per_d,
                color="k",
                ls=dis_lss[k],
                lw=2,
                label=f"total, {dis_labels[k]}",
            )
            axs[1].semilogx(
                self.lnA_energy_grid,
                self.truths["mean_lnA_truths"],
                color="k",
                ls="-",
                lw=3,
                label="total",
            )
            axs[2].semilogx(
                self.lnA_energy_grid,
                self.truths["var_lnA_truths"],
                color="k",
                ls="-",
                lw=3,
                label="total",
            )

            tot_espect += (
                self.truths["Nex_per_src"][k] * tot_espect_per_d / self.truths["Nex"]
            )

        # plot total spectrum
        axs[0].loglog(
            self.energy_grid, tot_espect, color="k", ls="-", lw=3, label="total"
        )

        # histogram the samples
        hist_vals, ebinedges = np.histogram(
            self.truths["Etruths"], bins=20, density=False
        )
        yvals = hist_vals / np.sum(hist_vals) / np.diff(ebinedges)
        yerr = np.sqrt(hist_vals) / np.sum(hist_vals) / np.diff(ebinedges)  # Error bars
        axs[0].errorbar(
            np.sqrt(ebinedges[:-1] * ebinedges[1:]),
            yvals,
            yerr=yerr,
            fmt="o",
            label="sampled",
            color="gray",
        )

        axs[2].set_xlabel(r"$E$ [EeV]")
        axs[0].set_ylabel("energy spectrum")
        axs[1].set_ylabel("mean lnA")
        axs[2].set_ylabel("var lnA")
        axs[0].legend()
        axs[0].set_ylim(ymin=1e-7, ymax=1)

        return fig, axs
