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
from fancy.physics.effective_exposure import WeightedExposure
from fancy.interfaces.grid_generator import GridGenerator
from fancy.utils.helpers import km_per_Mpc, theta_igmfs, truncated_lognorm_ccdf
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
        self.source_uvs = np.vstack(
            (data.source.coord.cartesian.xyz.value.T, np.array([0, 0, 1]))
        )

        # data object that encompasses detector & source information
        self.data = data
        self.eff_exp = None

        # other objects we store for later
        self.truths = {}
        self.config = {}

        # grid related parameters
        self.energy_grid = None
        self.lnA_energy_grid = None
        self.energy_grid_widths = None
        self.alpha_grid = None
        self.mass_ids_grid = None
        self.beta_egmf_grid = None
        self.rigidity_grid = None
        self.Emin = None
        self.Emax = None

        self.spectrum_grid = None
        self.mean_lnA_grid = None
        self.var_lnA_grid = None
        self.src_spectrum_grid = None
        self.esrc_ratio_grid = None
        self.eff_exp_grid = None
        self.wexp_earth_grid = None
        self.log_wexp_earth_grid = None
        self.wexp_src_grid = None
        self.log_wexp_src_grid = None

        # shape parameters
        self.NEs = 0
        self.NElnAs = 0
        self.Nalphas = 0
        self.Nmass_fracs = 0
        self.Nbeta_egmfs = 0
        self.Nrigidities = 0

    def initialise_grids(
        self: Self,
        energy_gridparams: tuple = (32, 500, 50),
        lnA_energy_gridparams: tuple = (1, 500, 50),
        effexp_model_kwargs: dict = {
            "beta_egmf_gridparams": (1e-3, 100, 30),
            "R_gridparams": (1, 300, 50),
        },
        src_inj_kwargs: dict = {
            "dinits": [4],
            "Rmax": 1.7,
        },
        bg_inj_kwargs: dict = {
            "z_max": 3.0,
            "source_evo": "SFR",
            "Rmax": 1.7,
        },
        energy_loss_model_kwargs: dict = {
            "massids" : [402, 1407, 2814]
        },
    ) -> None:
        """
        Initialise the grid of the simulation.

        This means (in practice) to generate the grid of effective exposure
        through grid parameters of beta_egmf and rigidity.

        Parameters
        ----------
        energy_gridparams : tuple, optional
            The grid parameters for the energy values.
            given as (E_min, E_max, Nbins), by default (32, 500, 50).
            The grid will be logarithmically spaced in energy.
        lnA_energy_gridparams : tuple, optional
            The grid parameters for the lnA energy values.
            given as (lnA_min, lnA_max, Nbins), by default (3, 100, 50).
            The grid will be linearly spaced in lnA.
        effexp_model_kwargs : dict, optional
            The keyword arguments for the effective exposure model.
            By default set to:
            {
                "beta_egmf_gridparams" : (1e-3, 1, 30),
                "R_gridparams" : (1, 500, 50),
            }
            where beta_egmf_gridparams are the grid parameters for the
            beta_egmf values (in nG Mpc^1/2) and R_gridparams are the grid
            parameters for the rigidity values (in EV).
        src_inj_kwargs : dict, optional
            The keyword arguments for the source injection model.
            By default set to:
            {
                "dinits": [4],
                "Rmax": 1.7,
            }
            where dinits are the distances of the sources (in Mpc)
            and Rmax is the maximum rigidity (in EV).
        bg_inj_kwargs : dict, optional
            The keyword arguments for the background injection model.
            By default set to:
            {
                "z_max": 3.0,
                "source_evo": "SFR",
                "Rmax": 1.7,
            }
            where z_max is the maximum redshift of the background sources,
            source_evo is the source evolution model (only "SFR" is implemented),
            and Rmax is the maximum rigidity (in EV).
        """
        grid_generator = GridGenerator(data=self.data, gmf_model=self.gmf_model)

        grid_generator.get_effective_exposure_grid(effexp_model_kwargs, self.n_jobs)

        grid_generator.get_energy_mass_grid(
            energy_gridparams,
            lnA_energy_gridparams,
            src_inj_kwargs,
            bg_inj_kwargs,
            energy_loss_model_kwargs,
        )

        grid_generator.get_weighted_exposures()

        self.config = grid_generator.store_grids_to_dict()

        self.energy_grid = self.config["energy_grid"]
        self.lnA_energy_grid = self.config["lnA_energy_grid"]
        self.energy_grid_widths = self.config["energy_grid_widths"]
        self.alpha_grid = self.config["alpha_grid"]
        self.mass_ids_grid = self.config["mass_ids_grid"]
        self.beta_egmf_grid = self.config["beta_egmf_grid"]
        self.rigidity_grid = self.config["rigidity_grid"]
        self.charges_grid = self.config["charges_grid"]
        self.mass_ids_grid = self.config["mass_ids_grid"]
        self.NEs = self.config["NEs"]
        self.NElnAs = self.config["NElnAs"]
        self.Nalphas = self.config["Nalphas"]
        self.Nmass_fracs = self.config["Nmass_fracs"]
        self.Nbeta_egmfs = self.config["Nbeta_egmfs"]

        self.spectrum_grid = self.config["spectrum_grid"]
        self.mean_lnA_grid = self.config["mean_lnA_grid"]
        self.var_lnA_grid = self.config["var_lnA_grid"]
        self.src_spectrum_grid = self.config["src_spectrum_grid"]
        self.esrc_ratio_grid = self.config["esrc_ratio_grid"]
        self.wexp_earth_grid = self.config["wexp_earth_grid"]
        self.log_wexp_earth_grid = self.config["log_wexp_earth_grid"]
        self.wexp_src_grid = self.config["wexp_src_grid"]
        self.log_wexp_src_grid = self.config["log_wexp_src_grid"]
        self.Emin = self.config["Emin"]
        self.Emax = self.config["Emax"]

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
            "log10_beta_egmf": np.log10(beta_egmf),
            "Lsrcs": Lsrcs,
            "Nex": Nex,
            "src_frac": source_fraction,
        }

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
        w_exp_src = np.zeros(self.Nsrcs)
        esrc_ratios = np.zeros(self.Nsrcs)

        for k in range(self.Nsrcs + 1):
            f_log10_wexp_earth = RegularGridInterpolator(
                (self.alpha_grid, np.log10(self.beta_egmf_grid.value)),
                np.log10(self.wexp_earth_grid[k, ...].value),
                bounds_error=False,
                # fill_value=0.0,
            )

            w_exp_earth[k] = np.sum(
                fit_truths["mass_fracs"][:, k]
                * 10.0
                ** f_log10_wexp_earth(
                    (fit_truths["alphas"][k], fit_truths["log10_beta_egmf"])
                )
            )

            if k < self.Nsrcs:
                esrc_ratios_mf = np.sum(
                    self.esrc_ratio_grid[:, :, k]
                    * fit_truths["mass_fracs"][:, k][np.newaxis, :],
                    axis=1,
                )
                f_log10_wexp_src = RegularGridInterpolator(
                    (self.alpha_grid, np.log10(self.beta_egmf_grid.value)),
                    np.log10(self.wexp_src_grid[k, :, :, :].value),
                    bounds_error=False,
                    # fill_value=0.0,
                )
                f_esrc_ratio = CubicSpline(self.alpha_grid, esrc_ratios_mf, axis=0)
                w_exp_src[k] = np.sum(
                    fit_truths["mass_fracs"][:, k]
                    * 10.0
                    ** f_log10_wexp_src(
                        (fit_truths["alphas"][k], fit_truths["log10_beta_egmf"])
                    )
                )
                esrc_ratios[k] = f_esrc_ratio(fit_truths["alphas"][k])

        return w_exp_earth, w_exp_src, esrc_ratios

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
        w_exp_earth, w_exp_src, esrc_ratios = self.__calculate_flux_weights(fit_truths)
        fit_truths["wexp_earth"] = w_exp_earth
        fit_truths["wexp_src"] = w_exp_src
        fit_truths["esrc_ratios"] = esrc_ratios

        # now calculate truths based on if we have Nex or Lsrcs or not
        if fit_truths["Nex"] is not None:
            Nex = fit_truths["Nex"]
            Nex_src = np.ceil(fit_truths["Nex"] * fit_truths["src_frac"]).astype(int)
            Nex_bg = fit_truths["Nex"] - Nex_src

            # here: calculate the total flux from the source at Earth
            # here we have -1 to exclude the background source
            Fearth_tot = Nex_src / w_exp_earth[:-1]

            # then calcualte the particle rate by multiplying by distance factor

            Qearths_truths = Fearth_tot * (
                4 * np.pi * (self.data.source.distance * km_per_Mpc) ** 2
            )

            # convert to source particle rate
            Qsrcs_truths = Qearths_truths * w_exp_src / w_exp_earth[:-1]
            Fsrcs_truths = Qsrcs_truths / (
                4 * np.pi * (self.data.source.distance * km_per_Mpc) ** 2
            )

            # now we can calculate the relative contribution of each source to the flux at Earth
            Fearths_truths = Qearths_truths / (
                4 * np.pi * (self.data.source.distance * km_per_Mpc) ** 2
            )

            # luminosity simply calculated via multiplying with Eex
            Lsrcs = Qsrcs_truths * esrc_ratios

            fit_truths["Lsrcs"] = Lsrcs
            fit_truths["log10_Lsrcs"] = np.log10(Lsrcs)
            fit_truths["Nex_src"] = Nex_src
            fit_truths["Nex_bg"] = Nex_bg

            # fit_truths["Nex_per_src"] = np.ceil(
            #     np.concatenate([Fearths_truths, [Nex_bg / w_exp_earth[-1]]]).T * w_exp_earth
            # ).astype(int)

        elif fit_truths["Lsrcs"] is not None:
            Qsrcs_truths = fit_truths["Lsrcs"] / esrc_ratios
            Qearths_truths = Qsrcs_truths * w_exp_earth[:-1] / w_exp_src

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
            
            Nex_src = np.ceil(Nex_src).astype(int)
            Nex = np.ceil(Nex_src / fit_truths["src_frac"]).astype(int)
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

        # fit_truths["Nex_per_src"] = np.ceil(
        #     np.concatenate([Fearths_truths, [fit_truths["F0"]]]).T * w_exp_earth
        # ).astype(int)
        # HACK!!! since we only have one source, we set the Nex_per_src manually
        fit_truths["Nex_per_src"] = np.array([
            fit_truths["Nex_src"], Nex_bg
        ])

        if np.sum(fit_truths["Nex_per_src"]) != Nex:
            print(fit_truths["Nex_per_src"], np.sum(fit_truths["Nex_per_src"]), Nex)
            raise ValueError(
                "Nex_per_src does not sum to Nex. Something went wrong."
            )

        fit_truths["flux_frac"] = (
            np.concatenate([Fearths_truths, [fit_truths["F0"]]]).T / fit_truths["Ftot"]
        )

        return fit_truths

    def generate_samples(
        self: Self, seed: Union[int, None] = None, sampling_factor: int = 10
    ) -> Tuple[np.ndarray, SkyCoord, np.ndarray, np.ndarray]:
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
        lnA_truths = np.zeros(self.truths["Nex"] * sampling_factor)
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
            (np.log(self.energy_grid), self.alpha_grid), np.log(energy_spect_mf)
        )

        f_mulnA = CubicSpline(y=mean_lnA_mfs, x=self.alpha_grid, axis=1)
        f_varlnA = CubicSpline(y=var_lnA_mfs, x=self.alpha_grid, axis=1)

        # binned likelihood (lnA) sampling
        for k in range(self.Nsrcs + 1):
            Nex_per_src = self.truths["Nex_per_src"][k]
            alpha_truth = self.truths["alphas"][k]
            # now calculate the mean and var lnA at Earth
            mean_lnA_truths += (
                Nex_per_src * f_mulnA(alpha_truth)[:, k] / self.truths["Nex"]
            )
            var_lnA_truths += (
                Nex_per_src * f_varlnA(alpha_truth)[:, k] / self.truths["Nex"]
            )

        N_prev_idx = 0

        # store the number of samples per source
        # which is needed to apply the detector response later
        Nsamples_per_src = []

        for k in range(self.Nsrcs + 1):  # +1 for the background source
            Nex_per_src = self.truths["Nex_per_src"][k]
            N_next_idx = Nex_per_src * sampling_factor + N_prev_idx

            alpha_truth = self.truths["alphas"][k]
            en_spect = np.exp(
                f_log_espect((np.log(self.energy_grid), alpha_truth))[:, k]
            )
            en_prob = (en_spect * self.energy_grid_widths) / np.sum(
                en_spect * self.energy_grid_widths
            )

            en_samples_src = rng.choice(
                self.energy_grid, size=Nex_per_src * sampling_factor, p=en_prob
            )
            Etruths[N_prev_idx:N_next_idx] = en_samples_src

            # the rigidities at the source can also be easily calculated
            # since we have the mean and sigma lnA
            # and simply assume rigidity conservation

            # we then just sample normally to get lnA
            # TODO: should investigate whether we should extend the binning to incorporate
            # all energies valid within the energy uncertainty
            mean_lnAs = mean_lnA_truths[
                np.digitize(en_samples_src, self.lnA_energy_grid) - 1
            ]
            var_lnAs = var_lnA_truths[
                np.digitize(en_samples_src, self.lnA_energy_grid) - 1
            ]
            lnA_samples = rng.normal(
                loc=mean_lnAs,
                scale=np.sqrt(var_lnAs),
                size=Nex_per_src * sampling_factor,
            )

            lnA_truths[N_prev_idx:N_next_idx] = lnA_samples

            # now calculate the rigidity at the source
            # by calculating Z = 0.5 * exp(lnA)
            rigidity_samples = en_samples_src / (0.5 * np.exp(lnA_samples))
            rigidity_truths[N_prev_idx:N_next_idx] = rigidity_samples

            # now calcualte the kappa_EGMF from the source
            if k < self.Nsrcs:
                kappa_egmfs = (
                    7552
                    * (
                        theta_igmfs(
                            rigidity_samples * u.EV,
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
                    self.source_uvs[k, :], kappa_egmfs[i - N_prev_idx], num_samples=1
                ).T
                skycoord_gb = SkyCoord(
                    arrdir_gb_truth,
                    frame="galactic",
                    representation_type="cartesian",
                )
                skycoord_gb.representation_type = "unitspherical"
                skycoord_gb_truths.append(skycoord_gb)

            N_prev_idx = Nex_per_src * sampling_factor
            Nsamples_per_src.append(N_prev_idx)

        skycoord_gb_truths = concatenate_skycoords(skycoord_gb_truths)

        # now we have the energy truths, mean lnA truths and var lnA truths
        self.truths["Etruths_samples"] = Etruths
        self.truths["mean_lnA_truths"] = mean_lnA_truths
        self.truths["var_lnA_truths"] = var_lnA_truths
        self.truths["skycoord_gb_truths"] = skycoord_gb_truths
        self.truths["lnA_truths"] = lnA_truths
        self.truths["rigidity_truths_samples"] = rigidity_truths
        self.truths["kappa_egmf_truth_samples"] = kappa_egmf_truths

        self.config["Nsamples_per_src"] = Nsamples_per_src
        self.config["sampling_factor"] = sampling_factor

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

        skycoords_earth = []
        # need to apply lens per source to ensure correct ratio
        N_prev_idx = 0
        for k in range(self.Nsrcs + 1):  # +1 for the background source

            N_next_idx = self.config["Nsamples_per_src"][k] + N_prev_idx
            defl_skycoord = gmflens.apply_lens_with_particles(
                self.truths["rigidity_truths_samples"][N_prev_idx:N_next_idx],
                skycoords_gb[N_prev_idx:N_next_idx],
            )

            skycoords_earth.append(defl_skycoord)
            N_prev_idx = N_next_idx

        skycoords_earth = concatenate_skycoords(skycoords_earth)

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
        logE_stat : float, optional
            The statistical uncertainty on the log energy.
        kappa_det : float, optional
            The concentration parameter for the vMF distribution
            that quantifies the uncertainty of the directional reconstruction.
            If None, then the value from the data.detector is used.
        logE_sys : float, optional
            The systematic uncertainty on the log energy.
            Default is 0.0.
            TODO: move this to when initialising the grid for the weighted exposure calculation.
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
        Etruths_mean = np.zeros(self.truths["Nex"])
        kappa_egmf_truths_mean = np.zeros(self.truths["Nex"])
        rigidities_mean = np.zeros(self.truths["Nex"])

        uhecr_idx = 0
        N_starting_idx = 0  # the starting index for each source
        rng_det = np.random.default_rng()

        # here iterate over each source, which has the number of samples.
        # this is required to keep the ratio constant
        for k, Nsample_per_src in enumerate(self.config["Nsamples_per_src"]):
            # enumerate only up to the number of samples per source
            Nex_per_src_idx = 0  # keep track of the number of events per source

            # get all the skycoords & energies related to this particular source
            skycoord_earths_per_src = self.truths["skycoord_earth_truths"][
                N_starting_idx:Nsample_per_src
            ]
            energies_per_src = self.truths["Etruths_samples"][
                N_starting_idx:Nsample_per_src
            ]
            kappa_egmfs_per_src = self.truths["kappa_egmf_truth_samples"][
                N_starting_idx:Nsample_per_src
            ]
            rigidities_per_src = self.truths["rigidity_truths_samples"][
                N_starting_idx:Nsample_per_src
            ]

            # here we randomise the order of the samples to ensure
            # that we do not introduce any bias in the accept-reject
            # algorithm
            sample_idces = np.arange(len(skycoord_earths_per_src))
            rng_det.shuffle(sample_idces)

            for i in sample_idces:
                # sample reconstruction uncertainty using vMF
                # and calculate if the direction is within the
                # exposure boundary or not.
                skycoord_earth = skycoord_earths_per_src[i]
                Etrue = energies_per_src[i]
                kappa_egmf = kappa_egmfs_per_src[i]
                rig = rigidities_per_src[i]

                # other two arguments returned are the reconstructed direction
                # and exposure function at that direction (declination)
                accept, skycoord_earth_det, m_exp = get_direction_acceptance(
                    skycoord_earth.cartesian.xyz.value,
                    kappa_det,
                    detector_params=self.data.detector.params,
                    max_exposure=self.data.detector.exposure_max,
                )

                Edet = get_Edet(
                    np.log(Etrue) + logE_sys,
                    en_unc=logE_stat,
                    Eth=self.Emin,
                    Emax=self.Emax,
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
                    # we also sample for the detected energy here
                    # truncated lognormal with global shift
                    Edets[uhecr_idx] = Edet

                    # also append the mean true energy & mean kappa EGMF
                    Etruths_mean[uhecr_idx] = Etrue
                    kappa_egmf_truths_mean[uhecr_idx] = kappa_egmf
                    rigidities_mean[uhecr_idx] = rig

                    uhecr_idx += 1
                    Nex_per_src_idx += 1

                if Nex_per_src_idx >= self.truths["Nex_per_src"][k]:
                    # if we have reached the number of events for this source,
                    # then we can stop
                    N_starting_idx += Nsample_per_src
                    print(f"Source {k + 1}: detected {Nex_per_src_idx} events.")
                    break

            if uhecr_idx >= self.truths["Nex"]:
                # if we have reached the number of expected events,
                # then we can stop
                print(f"Reached the expected number of events: {uhecr_idx}")
                break

        if uhecr_idx < self.truths["Nex"]:
            print(
                f"Error: only {uhecr_idx} events were detected out of the expected {self.truths['Nex']} events.",
                "Try increasing the sampling_factor.",
            )
            raise ValueError("Not enough events detected.")

        skycoord_earth_dets = concatenate_skycoords(skycoord_earth_dets)

        # store the detected values in the truths dictionary
        self.truths["Edets"] = Edets
        self.truths["skycoord_earth_dets"] = skycoord_earth_dets
        self.truths["Etruths"] = Etruths_mean
        self.truths["kappa_egmf_truths"] = kappa_egmf_truths_mean
        self.truths["rigidity_truths"] = rigidities_mean
        self.truths["exposure_factor"] = exposure_factor
        self.truths["kappa_ds"] = np.full(self.truths["Nex"], fill_value=kappa_det)

        # also set the uncertainties here
        self.config["logE_stat"] = logE_stat
        self.config["logE_sys"] = logE_sys
        self.config["kappa_det"] = kappa_det

        return Edets, skycoord_earth_dets

    # add some function to backtrack samples to get the kappa_GMF per mass model
    def backpropagate_events(
        self: Self, n_samples: int = 500, n_jobs: int = 4
    ) -> Tuple[SkyCoord, np.ndarray]:
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
            self.truths["kappa_gmfs"] = np.full(
                self.truths["Nex"], fill_value=self.config["kappa_det"]
            )  # no GMF deflection
            self.truths["theta_gmfs"] = np.full(
                self.truths["Nex"], fill_value=np.sqrt(7552 / self.config["kappa_det"])
            )  # no GMF deflection
            return self.truths["skycoord_earth_dets"], self.truths["kappa_gmfs"]
        # first write data to temporary file such that Data can read it
        outfile = (
            tempfile.mkstemp()[1] + "sim.h5"
        )  # add keyword "sim" so that the data UHECR reader knows that the full path should be used instead
        with h5py.File(outfile, "w") as f:
            data_gr = f.create_group(self.detector_type)
            data_gr.create_dataset("energy", data=self.truths["Etruths"])
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
            data_gr.create_dataset(
                "kappa_ds", data=np.full(self.truths["Nex"], 1.0)
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
            Emin=self.Emin,
            Emax=self.Emax,
        )
        gmfbackprop.run_backpropagation(n_samples, njobs=n_jobs, parallel=False)
        gmfbackprop.compute_kappa_gmf(n_jobs=n_jobs)

        # set properties
        if self.gmf_model != "None":
            self.truths["rigidity_bp"] = gmfbackprop.rigidities
            self.truths["kappa_gmfs"] = gmfbackprop.kappa_gmfs
            self.truths["theta_gmfs"] = np.rad2deg(gmfbackprop.thetaPs)
            self.truths["skycoord_gb_truths_bp"] = gmfbackprop.uhecr_coords_gb
            self.truths["kappa_ds"] = gmfbackprop.kappa_gmfs

        return gmfbackprop.uhecr_coords_gb, gmfbackprop.kappa_gmfs

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
            if self.gmf_model != "None":
                simulated_data.create_dataset(
                    "rigidity", data=self.truths["rigidity_bp"]
                )
            simulated_data.create_dataset("energy", data=self.truths["Edets"])
            simulated_data.create_dataset("ra", data=ras_det)
            simulated_data.create_dataset("dec", data=decs_det)
            simulated_data.create_dataset("glat", data=glats_det)
            simulated_data.create_dataset("glon", data=glons_det)
            simulated_data.create_dataset(
                "exposure", data=self.truths["exposure_factor"]
            )
            simulated_data.create_dataset(
                "kappa_ds", data=self.config["kappa_det"] * np.ones(self.truths["Nex"])
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
            The plotting mode to use. Can be "all", "skymap", "energy", "mass", "detected", or "backprop".
            By default set to "all".
        """
        if plotting_mode not in [
            "all",
            "skymap",
            "energy",
            "mass",
            "detected",
            "backprop",
        ]:
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
        if plotting_mode == "all" or plotting_mode == "detected":
            _, _, _ = plot_detected_events(
                self.data, self.truths, self.gmf_model, self.config
            )
        if plotting_mode == "all" or plotting_mode == "backprop":
            if self.gmf_model == "None":
                print("GMF is disabled. Will not produce backpropagation plots.")
                return
            _ = plot_backprop_skymap(self.data, self.truths, self.gmf_model)
            _ = plot_kappas(self.data, self.truths, self.gmf_model)
            _ = plot_thetas(self.data, self.truths, self.gmf_model)
            _ = plot_backprop_rigidities(self.data, self.truths, self.gmf_model)
