"""Simulation class for energy + mass + spatial"""

import os
import pickle
import tempfile
import h5py

from astropy.coordinates import SkyCoord
from astropy.coordinates import concatenate as concatenate_skycoords
import astropy.units as u
import numpy as np
import matplotlib as mpl
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from typing_extensions import List, Self, Tuple, Union

from fancy import Data
from fancy.physics import EnergyLossModel, EffectiveExposure, LossLengthModel
from fancy.physics.gmf import GMFLensing, GMFBackPropagation
from fancy.utils.helpers import km_per_Mpc, theta_igmfs
from fancy.simulation.helpers import (
    get_Edet,
    get_mean_lnA_det,
    get_var_lnA_det,
    get_direction_acceptance,
    source_spectrum,
    simulate_zenith_angles,
)
from fancy.plotting import AllSkyMapCartopy as AllSkyMap

from fancy.simulation.plotters import *

from vMF import sample_vMF

charge_massid_map = {101: 1, 402: 2, 1407: 7, 2814: 14, 5626: 26}


class Simulation:
    """Class for spatial + energy simulation."""

    def __init__(
        self, data: Data, gmf_model: str = "UF23base", n_jobs: Union[int, None] = None
    ) -> None:
        """
        Initialise the full energy + mass + spatial simulation class.

        Parameters
        ----------
        data: Data
            Data object containing the simulation information, such as the
            source position and source distance.
        gmf_model : str, default "UF23base"
            The GMF model to use for the simulation.
        n_jobs : int, optional
            The number of jobs to use for parallel processing of the
            effective exposure & GMF backtracking calculation.
            If None, then by default 3/4 of the available CPUs are used.
        """
        self.detector_type = data.detector.label
        self.mass_model = data.detector.mass_model
        self.source_type = data.source.label
        self.gmf_model = gmf_model
        self.n_jobs = n_jobs if n_jobs is not None else int(0.75 * os.cpu_count())

        # source parameters
        self.Nsrcs = data.source.N

        # stack the source coordinates and add a unit vector for the zenith
        # for the background model, since it doesnt matter what vector it is
        self.source_uvs = np.vstack((
            data.source.coord.cartesian.xyz.value.T,
            np.array([0,0,1])
        ))

        # data object that encompasses detector & source information
        self.data = data
        self.eff_exp = None

        # other objects we store for later
        self.truths = {}
        self.config = {}

        # grid related parameters
        self.energy_grid = None
        self.lnA_energy_grid = None
        self.alpha_grid = None
        self.mass_ids_grid = None
        self.beta_egmf_grid = None
        self.rigidity_grid = None

        self.spectrum_grid = None
        self.mean_lnA_grid = None
        self.var_lnA_grid = None
        self.src_spectrum_grid = None
        self.esrc_ratio_grid = None
        self.eff_exp_grid = None

        # shape parameters
        self.NEs = 0
        self.NElnAs = 0
        self.Nalphas = 0
        self.Nmass_fracs = 0
        self.Nbeta_egmfs = 0
        self.Nrigidities = 0

    def initialise_grids(
        self: Self,
        beta_egmf_gridparams: tuple = (1e-3, 1, 10),
        rigidity_gridparams: tuple = (1, 500, 25),
        energy_gridparams: tuple = (32, 500, 50),
        lnA_energy_gridparams: tuple = (3, 100, 50),
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
        Initialise the grid of the simulation.

        This means (in practice) to generate the grid of effective exposure
        through grid parameters of beta_egmf and rigidity.

        Parameters
        ----------
        beta_egmf_gridparams : tuple
            The grid parameters for the beta EGMF values.
            given as (beta_egmf_min, beta_egmf_max, Nbins)
        rigidity_gridparams : tuple
            The grid parameters for the rigidity values.
            given as (R_min, R_max, Nbins)
        energy_gridparams : tuple, optional
            The grid parameters for the energy values.
            given as (E_min, E_max, Nbins), by default (32, 500, 50).
            The grid will be logarithmically spaced in energy.
        lnA_energy_gridparams : tuple, optional
            The grid parameters for the lnA energy values.
            given as (lnA_min, lnA_max, Nbins), by default (0, 10, 50).
            The grid will be linearly spaced in lnA.
        """
        eff_exposure = EffectiveExposure(data=self.data, gmf_model=self.gmf_model)
        eff_exposure.initialise_grids(
            beta_egmf_gridparams=beta_egmf_gridparams, R_gridparams=rigidity_gridparams
        )
        eff_exposure.compute_effective_exposure(n_jobs=self.n_jobs)

        # store the effective exposure grid
        self.eff_exp_grid = (
            eff_exposure.effective_exposure
        )  # in shape (Nsrcs+1, NRs, Nbeta_egmfs)
        self.beta_egmf_grid = eff_exposure.beta_egmf_grid
        self.rigidity_grid = eff_exposure.rigidity_grid
        self.Nbeta_egmfs = len(self.beta_egmf_grid)
        self.Nrigidities = len(self.rigidity_grid)

        # store the results into the config object
        self.config["eff_exp_grid"] = eff_exposure.effective_exposure
        self.config["beta_egmf_grid"] = eff_exposure.beta_egmf_grid
        self.config["rigidity_grid"] = eff_exposure.rigidity_grid
        self.config["Nbeta_egmfs"] = len(eff_exposure.beta_egmf_grid)
        self.config["Nrigidities"] = len(eff_exposure.rigidity_grid)

        # store the effective exposure object for later use
        self.eff_exp = eff_exposure

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

        #  initalise the energy loss model
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
        self.charges_grid = np.array(
            [charge_massid_map[massid] for massid in self.mass_ids_grid]
        )

        # store the shape parameters
        self.NEs = self.spectrum_grid.shape[0]
        self.Nalphas = self.spectrum_grid.shape[1]
        self.Nmass_fracs = self.spectrum_grid.shape[2]
        self.NElnAs = self.mean_lnA_grid.shape[0]

        # also compute the src spectrum grid here
        self.src_spectrum_grid = source_spectrum(
            self.energy_grid[:, np.newaxis, np.newaxis, np.newaxis],
            self.alpha_grid[np.newaxis, :, np.newaxis, np.newaxis],
            self.charges_grid[np.newaxis, np.newaxis, :, np.newaxis],
            Rmax=src_inj_kwargs["Rmax"],
        )

        self.esrc_ratio_grid = np.trapz(
            y=self.energy_grid[:, None, None, None] * self.src_spectrum_grid,
            x=self.energy_grid,
            axis=0,
        ) / np.trapz(y=self.src_spectrum_grid, x=self.energy_grid, axis=0)

        # now calculate grid for flux weights at Earth and source
        self.wexp_earth_grid = np.zeros(
            (self.Nsrcs + 1, self.Nbeta_egmfs, self.Nalphas, self.Nmass_fracs)
        )
        self.wexp_src_grid = np.zeros(
            (self.Nsrcs + 1, self.Nbeta_egmfs, self.Nalphas, self.Nmass_fracs)
        )

        for k in range(self.Nsrcs + 1):
            f_effexp_rig = CubicSpline(
                x=self.rigidity_grid, y=self.eff_exp_grid[k, :, :], axis=0
            )
            for j in range(self.Nmass_fracs):
                rig = self.energy_grid / self.charges_grid[j]

                # TODO: update here 
                self.wexp_earth_grid[k, :, :, j] = np.trapz(
                    self.spectrum_grid[:, None, :, j, k]
                    * f_effexp_rig(rig)[:, :, None],
                    x=self.energy_grid,
                    axis=0,
                )

                # here there is a zero indexed since we keep the source spectrum
                # as the same shape as self.spectrum_grid.
                # but the source spectrum is identical for all sources + background
                self.wexp_src_grid[k, :, :, j] = np.trapz(
                    self.src_spectrum_grid[:, None, :, j, 0]
                    * f_effexp_rig(rig)[:, :, None],
                    x=self.energy_grid,
                    axis=0,
                )

        # store all these in the config dictionary
        self.config["energy_grid"] = self.energy_grid
        self.config["energy_grid_widths"] = self.energy_grid_widths
        self.config["lnA_energy_grid"] = self.lnA_energy_grid
        self.config["spectrum_grid"] = self.spectrum_grid
        self.config["mean_lnA_grid"] = self.mean_lnA_grid
        self.config["var_lnA_grid"] = self.var_lnA_grid
        self.config["src_spectrum_grid"] = self.src_spectrum_grid
        self.config["esrc_ratio_grid"] = self.esrc_ratio_grid
        self.config["wexp_earth_grid"] = self.wexp_earth_grid
        self.config["wexp_src_grid"] = self.wexp_src_grid
        self.config["alpha_grid"] = self.alpha_grid
        self.config["mass_ids_grid"] = self.mass_ids_grid
        self.config["charges_grid"] = self.charges_grid
        self.config["Nsrcs"] = self.Nsrcs
        self.config["NEs"] = self.NEs
        self.config["NElnAs"] = self.NElnAs
        self.config["Nalphas"] = self.Nalphas
        self.config["Nmass_fracs"] = self.Nmass_fracs

        # finally calculate loss lengths for 
        # proton case
        loss_length_model = LossLengthModel()
        loss_length_model.compute_source_energies(
            self.energy_grid,
            dinits=src_inj_kwargs["dinits"],
            save = False
        )
        self.config["proton_Esrc_grid"] = loss_length_model.Esrc_grid

    def set_truths(
        self: Self,
        mass_fracs: np.ndarray,
        alphas: np.ndarray,
        source_fraction: float = 0.5,
        beta_egmf: float = 1,
        Lsrcs: Union[np.ndarray, None] = None,
        Nex: Union[int, None] = None,
    ) -> dict:
        """
        Set the truths for the simulation.

        Parameters
        ----------
        mass_fracs : np.ndarray
            the mass fractions for the sources.
            Shape should be (Nmass_fracs, Nsrcs+1)
        alphas : np.ndarray
            the spectral indices for the sources
            Shape should be (Nsrcs+1)
        source_fraction : float, optional
            fraction of sources to use, by default 0.5
        beta_egmf : float, optional
            the magnetic spread parameter to use.
            By default set to 1 nG Mpc^1/2
        Lsrcs: np.ndarray, optional, default = None
            luminosity of the sources, by default None
            If None, then Nex must be provided.
            Shape must be (Nsrcs,)
        Nex : int, optional
            total number of expected events in the simulation, by default None
            If None, then Lsrcs must be provided.

        Returns
        -------
        truths : dict
            The updated truths dictionary.
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
            "beta_egmf": beta_egmf,
            "Lsrcs": Lsrcs,
            "Nex": Nex,
            "src_frac": source_fraction,
        }

        # calculate the mean charge at the source per source
        # NB: sum is enough since sum(mass_fracs) = 1.0
        fit_truths["Zsrc_mean"] = np.sum(
            mass_fracs * self.charges_grid[:, np.newaxis],
            axis=0,
        )

        fit_truths = self.__calculate_flux_truths(fit_truths)

        self.truths = fit_truths

        return fit_truths

    def __calculate_flux_weights(
        self: Self, fit_truths: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the weights related to the flux for the simulation.

        Parameters
        ----------
        fit_truths : dict
            The truths dictionary containing the simulation parameters.

        Returns
        -------
        w_exp_earth : np.ndarray
            The weights for the effective flux at Earth.
        w_exp_src : np.ndarray
            The weights for the effective flux at the source.
        """
        w_exp_earth = np.zeros(self.Nsrcs + 1)
        w_exp_src = np.zeros(self.Nsrcs + 1)

        for k in range(self.Nsrcs + 1):
            for j in range(self.Nmass_fracs):
                f_A = fit_truths["mass_fracs"][j, k]

                f_wexp_earth = RegularGridInterpolator(
                    (self.beta_egmf_grid, self.alpha_grid),
                    self.wexp_earth_grid[k, :, :, j],
                    bounds_error=False,
                    # fill_value=0.0,
                )

                w_exp_earth[k] += (
                    f_wexp_earth((fit_truths["beta_egmf"], fit_truths["alphas"][k]))
                    * f_A
                )

                # # do the same thing with the source spectrum
                f_wexp_src = RegularGridInterpolator(
                    (self.beta_egmf_grid, self.alpha_grid),
                    self.wexp_src_grid[k, :, :, j],
                    bounds_error=False,
                    # fill_value=0.0,
                )

                w_exp_src[k] += (
                    f_wexp_src((fit_truths["beta_egmf"], fit_truths["alphas"][k])) * f_A
                )

        return w_exp_earth, w_exp_src

    def __calculate_flux_truths(self: Self, fit_truths: dict) -> dict:
        """
        Calculate the truths related to the flux for the simulation.

        Parameters
        ----------
        fit_truths : dict
            The truths dictionary containing the simulation parameters.

        Returns
        -------
        truths_dict : dict
            A dictionary containing the calculated flux truths.
        """
        # calculate the weighted effective exposure at Earth and source
        w_exp_earth, w_exp_src = self.__calculate_flux_weights(fit_truths)

        # calculate conversion from L -> Q
        f_en_ratio = CubicSpline(
            self.alpha_grid,
            np.sum(
                self.esrc_ratio_grid * fit_truths["mass_fracs"][np.newaxis, :, :],
                axis=1,
            ),
            axis=0,
        )
        # now calculate truths based on if we have Nex or Lsrcs or not
        if fit_truths["Nex"] is not None:
            Nex = fit_truths["Nex"]
            Nex_src = int(fit_truths["Nex"] * fit_truths["src_frac"])
            Nex_bg = fit_truths["Nex"] - Nex_src

            # here: calculate the total flux from the source at Earth
            Fearth_tot = Nex_src / w_exp_earth

            # then calcualte the particle rate by multiplying by distance factor
            Qearths_truths = Fearth_tot * (
                4 * np.pi * (self.data.source.distance * km_per_Mpc) ** 2
            )

            # convert to source particle rate
            Qsrcs_truths = Qearths_truths * w_exp_src / w_exp_earth
            Fsrcs_truths = Qsrcs_truths / (
                4 * np.pi * (self.data.source.distance * km_per_Mpc) ** 2
            )

            # now we can calculate the relative contribution of each source to the flux at Earth
            Fearths_truths = Qearths_truths / (
                4 * np.pi * (self.data.source.distance * km_per_Mpc) ** 2
            )

            # luminosity simply calculated via multiplying with Eex
            Lsrcs = Qsrcs_truths * np.array(
                [f_en_ratio(fit_truths["alphas"][k])[k] for k in range(self.Nsrcs)]
            )

            fit_truths["Lsrcs"] = Lsrcs
            fit_truths["log10_Lsrcs"] = np.log10(Lsrcs)
            fit_truths["Nex_src"] = Nex_src
            fit_truths["Nex_bg"] = Nex_bg

        elif fit_truths["Lsrcs"] is not None:
            Qsrcs_truths = fit_truths["Lsrcs"] / np.array(
                [f_en_ratio(fit_truths["alphas"][k])[k] for k in range(self.Nsrcs)]
            )
            Qearths_truths = Qsrcs_truths * w_exp_earth / w_exp_src

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
                Nex_src += Fearths_truths[k] * w_exp_earth[k]

            Nex_src = int(Nex_src)
            Nex = int(Nex_src / fit_truths["src_frac"])
            Nex_bg = Nex - Nex_src

            fit_truths["Nex"] = Nex
            fit_truths["Nex_src"] = Nex_src
            fit_truths["Nex_bg"] = Nex_bg
            fit_truths["log10_Lsrcs"] = np.log10(fit_truths["Lsrcs"])
        else:
            raise ValueError(
                "Either Nex or Lsrcs must be provided in the truths dictionary."
            )

        print(f"Total number of expected events: {Nex} (src: {Nex_src}, bg: {Nex_bg})")

        fit_truths["Qsrcs"] = Qsrcs_truths
        fit_truths["Qearths"] = Qearths_truths
        fit_truths["Fsrcs"] = Fsrcs_truths
        fit_truths["log10_Fsrcs"] = np.log10(Fsrcs_truths)

        fit_truths["Fearths"] = Fearths_truths
        fit_truths["log10_Fearths"] = np.log10(Fearths_truths)

        fit_truths["F0"] = Nex_bg / w_exp_earth[-1]
        fit_truths["log10_F0"] = np.log10(fit_truths["F0"])

        fit_truths["Ftot"] = np.sum(Fearths_truths) + fit_truths["F0"]
        fit_truths["log10_Ftot"] = np.log10(fit_truths["Ftot"])

        fit_truths["Nex_per_src"] = (
            np.concatenate([Fearths_truths, [fit_truths["F0"]]]).T * w_exp_earth
        ).astype(int)

        return fit_truths

    def generate_samples(
        self: Self, seed: Union[int, None] = None, sampling_factor: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate samples for the simulation.

        In particular, these generate the following:
        - true energy samples at Earth (in EeV)
        - true samples of mean and var lnA at Earth, binned in energy
        - true arrival directions at the Galactic Boundary

        Parameters
        ----------
        seed : int, optional
            The random seed to use for the simulation.
            If None, then no seed is set.
        sampling_factor : int, optional
            The factor by which to increase the number of samples.
            This is useful for ensuring that we have enough samples
            for the simulation for the exposure calculation later.
            By default set to 10.
        """
        print(
            f"Generating samples for the simulation with sampling factor of {sampling_factor}"
        )
        rng = np.random.default_rng(seed=seed)

        mean_lnA_truths = np.zeros(self.NElnAs)
        var_lnA_truths = np.zeros(self.NElnAs)

        skycoord_gb_truths = []
        Etruths = np.zeros(self.truths["Nex"] * sampling_factor)
        rigidity_truths = np.zeros(self.truths["Nex"] * sampling_factor)
        kappa_egmf_truths = np.zeros(self.truths["Nex"] * sampling_factor)

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
            Nex_per_src = self.truths["Nex_per_src"][k]
            N_next_idx = Nex_per_src * sampling_factor + N_prev_idx

            alpha_truth = self.truths["alphas"][k]
            en_spect = np.exp(
                f_log_espect((np.log10(self.energy_grid), alpha_truth))[:, k]
            )
            en_prob = (en_spect * self.energy_grid_widths) / np.sum(
                en_spect * self.energy_grid_widths
            )

            en_samples_src = rng.choice(
                self.energy_grid, size=Nex_per_src * sampling_factor, p=en_prob
            )
            Etruths[N_prev_idx:N_next_idx] = en_samples_src

            # the rigidities at the source can also be easily calculated
            # since we have the mean charge at the source
            rigidity_truths[N_prev_idx:N_next_idx] = (
                en_samples_src / self.truths["Zsrc_mean"][k]
            )

            # now calcualte the kappa_EGMF from the source
            if k < self.Nsrcs:
                kappa_egmfs = (
                    7552
                    * (
                        theta_igmfs(
                            (en_samples_src / self.truths["Zsrc_mean"][k]) * u.EV,
                            self.truths["beta_egmf"] * (u.nG * u.Mpc ** (1 / 2)),
                            self.data.source.distance[k] * u.Mpc,
                        )
                        / (1 * u.deg)
                    ).value
                    ** -2
                )
            else:
                kappa_egmfs = np.zeros(Nex_per_src * sampling_factor)

            kappa_egmf_truths[N_prev_idx:N_next_idx] = kappa_egmfs

            # use this with vMF distribution to get the arrival directions at the GB
            for i in range(N_prev_idx, N_next_idx):
                arrdir_gb_truth = sample_vMF(
                    self.source_uvs[k,:], kappa_egmfs[i - N_prev_idx], num_samples=1
                ).T
                skycoord_gb = SkyCoord(
                    arrdir_gb_truth,
                    frame="galactic",
                    representation_type="cartesian",
                )
                skycoord_gb.representation_type = "unitspherical"
                skycoord_gb_truths.append(skycoord_gb)

            N_prev_idx = Nex_per_src * sampling_factor

            # now calculate the mean and var lnA at Earth
            mean_lnA_truths += (
                Nex_per_src * f_mulnA(alpha_truth)[:, k] / self.truths["Nex"]
            )
            var_lnA_truths += (
                Nex_per_src * f_varlnA(alpha_truth)[:, k] / self.truths["Nex"]
            )

        skycoord_gb_truths = concatenate_skycoords(skycoord_gb_truths)

        # now we have the energy truths, mean lnA truths and var lnA truths
        self.truths["Etruths"] = Etruths
        self.truths["mean_lnA_truths"] = mean_lnA_truths
        self.truths["var_lnA_truths"] = var_lnA_truths
        self.truths["skycoord_gb_truths"] = skycoord_gb_truths
        self.truths["rigidity_truths"] = rigidity_truths
        self.truths["kappa_egmf_truths"] = kappa_egmf_truths

        return Etruths, skycoord_gb_truths, mean_lnA_truths, var_lnA_truths

    def get_skycoords_earth(self: Self, skycoords_gb: SkyCoord) -> SkyCoord:
        """
        Apply GMF lens by sampling & re-sampling of particles.

        Parameters
        ----------
        rigidities: list[np.ndarray]
            rigidities at the Galactic boundary
        skycoords_gb: list[astropy.coordinate.SkyCoord]
            sampled arrival diretions at the Galactic boundary

        Returns
        -------
        skycoords_earth: list[astropy.coordinate.SkyCoord]
            sampled arrival directions at the Earth after lensing.
        """
        if self.gmf_model == "None":
            print(
                "GMF is disabled. Will not run this code and set the skycoords_earth to the skycoords_gb."
            )
            self.truths["skycoord_earth_truths"] = skycoords_gb
            return skycoords_gb

        # initialise gmf lens object
        gmflens = GMFLensing(self.gmf_model)

        skycoords_earth = gmflens.apply_lens_with_particles(
            self.truths["rigidity_truths"],
            skycoords_gb,
        )

        self.truths["skycoord_earth_truths"] = skycoords_earth

        return skycoords_earth

    def apply_mass_response(
        self: Self,
        mean_lnA_stat: Union[float, np.ndarray],
        var_lnA_stat: Union[float, np.ndarray],
        mean_lnA_sys: float = 0.0,
        var_lnA_sys: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the detector response to the true values for the mean and variance of lnA.

        Parameter:
        ----------
        mean_lnA_stat : float or np.ndarray
            The statistical uncertainty on the mean lnA.
            If a single value, then it is applied to all energy bins.
            If an array, then it must have shape (NElnAs).
        var_lnA_stat : float or np.ndarray
            The statistical uncertainty on the variance of lnA.
            If a single value, then it is applied to all energy bins.
            If an array, then it must have shape (NElnAs).
        mean_lnA_sys : float, optional
            The systematic uncertainty on the mean lnA.
            Default is 0.0.
        var_lnA_sys : float, optional
            The systematic uncertainty on the variance of lnA.
            Default is 0.0.
        """
        if isinstance(mean_lnA_stat, float):
            mean_lnA_stat = np.full(self.NElnAs, mean_lnA_stat)

        if isinstance(var_lnA_stat, float):
            var_lnA_stat = np.full(self.NElnAs, var_lnA_stat)

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
        self.truths["mean_lnA_dets"] = mean_lnA_dets
        self.truths["var_lnA_dets"] = var_lnA_dets

        # also set the uncertainties here
        self.config["mean_lnA_stat"] = mean_lnA_stat
        self.config["var_lnA_stat"] = var_lnA_stat
        self.config["mean_lnA_sys"] = mean_lnA_sys
        self.config["var_lnA_sys"] = var_lnA_sys

        return mean_lnA_dets, var_lnA_dets

    def apply_energy_directional_response(
        self: Self,
        logE_stat: Union[float, None] = None,
        kappa_det: Union[float, None] = None,
        logE_sys: float = 0.0,
    ) -> Tuple[np.ndarray, List[SkyCoord]]:
        """
        Apply the detector response to the true values for the energy and direction.

        i.e., we apply response to the unbinned quantities.

        This is treated separately as we do a accept-reject implementation
        for the direction, which is not the case for the lnA values.

        Parameter:
        ----------
        logE_stat : float or None, optional
            The statistical uncertainty on the log energy.
            If None, then the uncertainty from data.detector is used.
        logE_sys : float, optional
            The systematic uncertainty on the log energy.
            Default is 0.0.
        """
        # if None then use the energy uncertainty reported in
        # data.detector
        if logE_stat is None:
            logE_stat = self.data.detector.energy_uncertainty

        if kappa_det is None:
            kappa_det = self.data.detector.kappa_d

        skycoord_earth_dets = []
        exposure_factor = np.zeros(self.truths["Nex"])
        Edets = np.zeros(self.truths["Nex"])

        uhecr_idx = 0

        for j, skycoord_earth in enumerate(self.truths["skycoord_earth_truths"]):
            # sample reconstruction uncertainty using vMF
            # and calculate if the direction is within the
            # exposure boundary or not.

            # other two arguments returned are the reconstructed direction
            # and exposure function at that direction (declination)
            accept, skycoord_earth_det, m_exp = get_direction_acceptance(
                skycoord_earth.cartesian.xyz.value,
                kappa_det,
                detector_params=self.data.detector.params,
                max_exposure=self.data.detector.exposure_max,
            )

            if accept != 0:
                # if the acceptance is not zero, then we have a valid
                # direction and can store it

                # store the glon and glat
                # as well as the exposure factor
                skycoord_earth_dets.append(skycoord_earth_det)
                exposure_factor[uhecr_idx] = (
                    m_exp * self.data.detector.alpha_T / self.data.detector.M
                )
                Edets[uhecr_idx] = get_Edet(
                    np.log(self.truths["Etruths"][uhecr_idx]) + logE_sys,
                    en_unc=logE_stat,
                    Eth=np.min(self.energy_grid),
                    Emax=np.max(self.energy_grid),
                )

                uhecr_idx += 1

            if uhecr_idx >= self.truths["Nex"]:
                # if we have reached the number of expected events,
                # then we can stop
                break

        skycoord_earth_dets = concatenate_skycoords(skycoord_earth_dets)

        # store the detected values in the truths dictionary
        self.truths["Edets"] = Edets
        self.truths["skycoord_earth_dets"] = skycoord_earth_dets
        self.truths["exposure_factor"] = exposure_factor

        # also set the uncertainties here
        self.config["logE_stat"] = logE_stat
        self.config["logE_sys"] = logE_sys
        self.config["kappa_det"] = kappa_det

        return Edets, skycoord_earth_dets

    # add some function to backtrack samples to get the kappa_GMF per mass model
    def backpropagate_events(self: Self, n_samples: int = 100, n_jobs: int = 4) -> None:
        """
        Backpropagate the sampled & exposure-applied events at Earth back to the GB.

        Parameters
        ----------
        n_samples: int
            the number of samples for backpropagation simulation
        n_jobs: int
            the number of jobs to use for parallelisation
        """
        if self.gmf_model == "None":
            print(
                "GMF is disabled. Will not run this code and set the deflection parameters to None."
            )
            self.truths["skycoord_gb_truths_bp"] = None
            self.truths["rigidity_bp"] = None

            # for the kappa gmfs and theta gmfs, we just store the deflection parameters from
            # the angular reconstruction uncertainty set in the simulation.
            self.truths["kappa_gmfs"] = None  # no GMF deflection
            self.truths["theta_gmfs"] = None  # no GMF deflection
            return
        # first write data to temporary file such that Data can read it
        outfile = (
            tempfile.mkstemp()[1] + "sim.h5"
        )  # add keyword "sim" so that the data UHECR reader knows that the full path should be used instead
        with h5py.File(outfile, "w") as f:
            data_gr = f.create_group(self.detector_type)
            data_gr.create_dataset("energy", data=self.truths["Edets"])
            data_gr.create_dataset(
                "glon", data=self.truths["skycoord_earth_dets"].galactic.l.deg
            )
            data_gr.create_dataset(
                "glat", data=self.truths["skycoord_earth_dets"].galactic.b.deg
            )

            # following datasets are stubs, as they are not used in the backpropagation
            data_gr.create_dataset(
                "exposure_factors", data=np.full(self.truths["Nex"], 1.0)
            )  # stub
            data_gr.create_dataset("theta", data=np.full(self.truths["Nex"], 0))  # stub
            data_gr.create_dataset(
                "year",
                data=np.full(
                    self.truths["Nex"], self.data.detector.start_year, dtype=int
                ),
            )  # stub
            data_gr.create_dataset(
                "day", data=np.ones(self.truths["Nex"], dtype=int)
            )  # stub

        # add this to a newly generated data object
        data_for_bp = Data()
        data_for_bp.add_detector(label=self.detector_type, mass_model=self.mass_model)
        data_for_bp.add_uhecr(
            label=self.detector_type,
            mass_model=self.mass_model,
            gmf_model="None",
            filename=outfile,
        )

        # now perform GMF back propagation
        gmfbackprop = GMFBackPropagation(data_for_bp, self.gmf_model)

        # setup the "detector response" through the mean and sigma lnA observed at Earth
        # from our model
        # NB: we pass in the true mean and variance of lnA at Earth
        # as opposed to the detected one, since they will have
        # systematic uncertainties, which can lead to
        # weird values for mean and variance of lnA
        gmfbackprop.setup_detector_response(
            mean_lnA_grid=self.truths["mean_lnA_truths"],
            var_lnA_grid=self.truths["var_lnA_truths"],
            E_lnA_grid=self.lnA_energy_grid,
            logE_stat=self.config["logE_stat"],
            kappa_det=self.config["kappa_det"],
            Eth=np.min(self.config["energy_grid"]),
            Eth_max=np.max(self.config["energy_grid"]),
        )
        gmfbackprop.run_backpropagation(n_samples, njobs=n_jobs)
        gmfbackprop.compute_kappa_gmf()

        # set properties
        if self.gmf_model != "None":
            self.truths["rigidity_bp"] = gmfbackprop.mean_rigidity
            self.truths["kappa_gmfs"] = gmfbackprop.kappa_gmfs
            self.truths["theta_gmfs"] = np.rad2deg(gmfbackprop.thetaPs)
            self.truths["skycoord_gb_truths_bp"] = gmfbackprop.uhecr_coords_gb

    def save(self: Self, outfile: str) -> None:
        """
        Save the simulation file as a new UHECR file.

        Parameters
        ----------
        outfile: str
            the output file as an UHECR file
        """
        # get ra, dec, glon, glat
        glons_det, glats_det = (
            self.truths["skycoord_earth_dets"].galactic.l.deg,
            self.truths["skycoord_earth_dets"].galactic.b.deg,
        )
        # simulate for zenith angles for compatibility
        c_icrs = self.truths["skycoord_earth_dets"].transform_to("icrs")
        zeniths_sim, years_sim, days_sim = simulate_zenith_angles(
            c_icrs,
            zen_thresh=self.data.detector.threshold_zenith_angle.value,
            period_start=self.data.detector.period_start,
            location=self.data.detector.location,
        )

        ras_det, decs_det = c_icrs.ra.deg, c_icrs.dec.deg

        with h5py.File(outfile, "a") as file:
            if self.detector_type in list(file.keys()):
                del file[self.detector_type]
            simulated_data = file.create_group(f"{self.detector_type}")
            simulated_data.create_dataset("day", data=days_sim)
            simulated_data.create_dataset("year", data=years_sim)
            simulated_data.create_dataset("theta", data=zeniths_sim)
            simulated_data.create_dataset("rigidity", data=self.truths["rigidity_bp"])
            simulated_data.create_dataset("energy", data=self.truths["Edets"])
            simulated_data.create_dataset("ra", data=ras_det)
            simulated_data.create_dataset("dec", data=decs_det)
            simulated_data.create_dataset("glat", data=glats_det)
            simulated_data.create_dataset("glon", data=glons_det)
            simulated_data.create_dataset(
                "exposure", data=self.truths["exposure_factor"]
            )

            if self.gmf_model != "None":
                gmfdefl_datas_grp = simulated_data.create_group("gmf")
                config_key = f"{self.gmf_model}_{self.mass_model}"
                if config_key in gmfdefl_datas_grp.keys():
                    del gmfdefl_datas_grp[config_key]
                gmfdefl_datas_config_grp = gmfdefl_datas_grp.create_group(config_key)
                gmfdefl_datas_config_grp.create_dataset(
                    "kappa_gmf", data=self.truths["kappa_gmfs"]
                )
                gmfdefl_datas_config_grp.create_dataset(
                    "thetaP", data=self.truths["theta_gmfs"]
                )
                gmfdefl_datas_config_grp.create_dataset(
                    "glons_gb", data=self.truths["skycoord_gb_truths_bp"].galactic.l.deg
                )
                gmfdefl_datas_config_grp.create_dataset(
                    "glats_gb",
                    data=self.truths["skycoord_gb_truths_bp"].galactic.b.deg,
                )

    def plot_samples(
        self: Self, plotting_mode: str = "all"
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """
        Plot the samples of the simulation.

        This function will plot the following things:
        1. A skymap of the arrival directions at the Galactic boundary, color coded by its rigidity. The EGMF deflections is encoded as a circle in the plots. The source direction is also shown here.
        2. A skymap of the arrival directions at the Earth, color coded by its energy. The source direction is also shown.
        3. The energy spectrum, showing the relative contribution of source & background sources, as well as the mass fraction weighted spectrum for each source.
        4. The mean and variance of lnA at Earth, color coded by the source index.
        5. The deflection directions from Earth -> GB, along with the deflection angles per UHECR.

        Note: this is a simple plotting function and does not include any advanced features. Generate your own plots if you want to do more advanced things.

        Parameters
        ----------
        plotting_mode : str, optional
            The plotting mode to use. Can be "all", "skymap", "energy", "mass", or "backprop".
            By default set to "all".
        """
        if plotting_mode not in ["all", "skymap", "energy", "mass", "backprop"]:
            raise ValueError(
                f"Invalid plotting mode {plotting_mode}. Choose from 'all', 'skymap', 'energy_mass' or 'backprop'."
            )

        if plotting_mode == "all" or plotting_mode == "skymap":
            _ = plot_skymap_gb(self.data, self.truths)
            _ = plot_skymap_earth(self.data, self.truths, self.gmf_model)
        if plotting_mode == "all" or plotting_mode == "energy":
            _ = plot_energy(self.data, self.truths, self.config)
        if plotting_mode == "all" or plotting_mode == "mass":
            _ = plot_mean_sigma_lnA(self.data, self.truths, self.config)
        if plotting_mode == "all" or plotting_mode == "backprop":
            _ = plot_backprop_skymap(self.data, self.truths, self.gmf_model)
            _ = plot_backprop_rigidities(self.data, self.truths, self.gmf_model)
