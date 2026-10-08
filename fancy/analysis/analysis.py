"""Container to manage the inputs and outputs of the fits."""
import os
import pickle
from typing import Union

import cmdstanpy
import h5py
import astropy.units as u
import numpy as np
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from cmdstanpy import CmdStanModel
from typing_extensions import Self  # change to typing for py>3.11
import arviz as az

from fancy.interfaces.data import Data
from fancy.interfaces.grid_generator import GridGenerator
from fancy.physics.effective_exposure import EffectiveExposureKappaOnly
from fancy.simulation import Simulation
from fancy.utils.egmf_priors import get_log10_beta_egmf_prior
from fancy.utils.helpers import create_dataset_compressed, pick_grain_size
from fancy.utils.package_data import (
    get_path_to_stan_file,
    get_path_to_stan_includes,
)


class FitResult:
    """Read-only, h5-backed stand-in for a ``cmdstanpy.CmdStanMCMC``.

    Exposes just the subset of the cmdstanpy interface that the plotting code
    and ``PPC`` actually use (``stan_variable``, ``stan_variables``,
    ``method_variables``, ``divergences``, ``max_treedepths``, ``step_size``,
    ``diagnose``), backed directly by the ``fit/samples`` and
    ``fit/diagnostics`` groups written by :meth:`Analysis.save`. This avoids
    ever needing to re-run HMC or unpickle a raw ``CmdStanMCMC`` just to
    regenerate plots from a previously-saved fit.
    """

    def __init__(self: Self, samples: dict, diagnostics: dict) -> None:
        self._samples = samples
        self._diagnostics = diagnostics

    def stan_variable(self: Self, var: str) -> np.ndarray:
        return self._samples[var]

    def stan_variables(self: Self) -> dict:
        return self._samples

    def method_variables(self: Self) -> dict:
        method_vars = {"lp__": self._diagnostics["log_post"]}
        for key in ("divergent__", "treedepth__", "energy__", "n_leapfrog__"):
            if key in self._diagnostics:
                method_vars[key] = self._diagnostics[key]
        return method_vars

    @property
    def divergences(self: Self) -> np.ndarray:
        return self._diagnostics["divergences"]

    @property
    def max_treedepths(self: Self) -> np.ndarray:
        return self._diagnostics["max_treedepths"]

    @property
    def step_size(self: Self) -> np.ndarray:
        return self._diagnostics["step_size"]

    def diagnose(self: Self) -> str:
        report = self._diagnostics.get("diagnose_report", "")
        return report if isinstance(report, str) else report.decode()


class Analysis:
    """Container to manage the inputs and outputs of the fits."""

    # pre-defined analysis types
    energy_type = "energy_only"
    mass_type = "mass_only"
    spatial_type = "spatial_only" # NOTE: this case is only possible as debug
    mass_spatial_type = "mass_spatial"
    energy_mass_type = "energy_mass"
    energy_spatial_type = "energy_spatial"
    energy_mass_spatial_type = "energy_mass_spatial"

    fit_input_keys = [  # noqa: RUF012
        "Nsrcs",
        "D",
        "omega_src",
        "N",
        "NEbins",
        "Edet",
        "omega_det",
        "kappa_ds",
        "mean_lnA_det",
        "var_lnA_det",
        "Nalphas",
        "alpha_grid",
        "NEs",
        "logE_grid",
        "NAsrcs",
        "earth_spectrum_grid",
        "lnA_logE_grid",
        "mean_lnA_grid",
        "var_lnA_grid",
        "Eth",
        "logE_stat_unc",
        "logE_sys_unc",
        "mean_lnA_stat_unc",
        "var_lnA_stat_unc",
        "mean_lnA_sys_unc",
        "var_lnA_sys_unc",
        "Nbeta_egmfs",
        "log10_beta_egmf_grid",
        "log_wexp_earth_grid",
        "log_wexp_src_grid",
        "esrc_ratio_grid",
        "grain_size",
        "exposure_factor",
    ] 

    # data-block inputs of the kappa-only spatial model
    # (spatial_only/spatial_model_alpha_spline.stan)
    kappa_only_fit_input_keys = (  # noqa: RUF012
        "Nsrcs",
        "D",
        "omega_src",
        "N",
        "omega_det",
        "kappa_ds",
        "exposure_factor",
        "Nkappas",
        "log_kappa_grid",
        "log_wexp_earth_grid",
        "grain_size",
    )

    def __init__(
        self: Self,
        data: Data,
        gmf_model : str = "None",
        analysis_type: str = energy_mass_spatial_type,
        background_only : bool = False,
        use_rigidity_grid : bool = False,
        use_with_systematics : bool = False,
        use_alpha_spline : bool = False,
        fit_logE_sys : bool = False,
        beta_egmf_max : Union[float, None] = None,
        use_beta_spline : bool = True,
    ) -> None:
        """
        Container to manage the inputs and outputs of the fits.

        Parameters
        ----------
        data: fancy.interfaces.data.Data
            Container that handles the source, uhecr, and detector information.
            All such information should already be initialised (see relevant class for
            more information.)
        gmf_model: str, default="None"
            The GMF model to consider.
        analysis_type: str, default=energy_mass_spatial
            The analysis type to consider.
        background_only : bool, default=False
            Whether to consider only background sources in the analysis.
        use_rigidity_grid : bool, default=False
            If True (only supported for analysis_type=energy_mass_spatial or
            analysis_type=spatial_only), compiles the rigidity-resolved
            kappa_GMF(R) model variant, which requires log10_gmf_Rgrid /
            log_kappa_gmf_grid to be present in the loaded data / simulation
            (see RigidityResolvedGMFBackPropagation).
        use_with_systematics : bool, default=False
            If True (only supported for analysis_type=energy_mass_spatial),
            compiles the "_with_systematics" model variant, which samples
            additional latent nuisance parameters (nu_logE_sys,
            nu_kappa_gmf_sys, nu_mean_lnA_sys, nu_var_lnA_sys) to marginalise
            over the systematic uncertainties instead of ignoring them.
        beta_egmf_max : float or None, default=None
            Only for the split lnA grid model: upper bound on beta_egmf
            (nG Mpc^1/2), applied in Stan as min(beta grid maximum,
            beta_egmf_max). None -> the beta grid's maximum.
        use_beta_spline : bool, default=True
            Only for the split lnA grid model: interpolate the log weighted
            exposure grids with a natural cubic spline in log10(beta_egmf) as
            well as alpha. If False, linear in log10(beta_egmf), which leaves
            kinks at the beta grid nodes that a sharp likelihood pins
            beta_egmf to.
        fit_logE_sys : bool, default=False
            Only for the split lnA grid model (see `split_lnA_grid`). If True,
            the energy-scale systematic is fitted as a global shift
            nu_logE_sys * f_E_sys (nu_logE_sys ~ N(0, 1)). If False, the
            detector's f_E_sys is applied as a fixed shift (logE_sys_scale = 0,
            so nu_logE_sys decouples from the data).
        use_alpha_spline : bool, default=False
            If True, compiles the "_alpha_spline" model variant (the "knots
            method"), which replaces the default model's linear
            interpolation over alpha_grid with a natural cubic spline (see
            fancy.utils.helpers.natural_cubic_spline_matrix), evaluated using
            a precomputed second-derivative matrix passed in as
            alpha_spline_matrix. Kept as a fully separate Stan model file per
            analysis_type so each default linear-interpolation model stays
            unchanged and reproducible; intended for direct A/B comparison
            against it on the same simulated dataset. Supported for every
            analysis_type except the background-only variant (bg_only=True).

            NB: for analysis_type=spatial_only this selects the kappa-only
            model (spatial_model_alpha_spline.stan), which has no alpha_grid /
            alpha_spline_matrix at all: the effective exposure is tabulated vs
            a single shared kappa_egmf instead (see `kappa_only`), loaded from
            the tables written by EffectiveExposureKappaOnly.

            NB: for analysis_type=energy_mass_spatial this model also decouples
            the lnA model grid (lnA_logE_grid, NEbins_lnAgrid bins, set by
            lnA_energy_gridparams) from the detector's lnA bins
            (lnA_logE_grid_det, NEbins bins), see `split_lnA_grid`.
        """
        self.data = data
        # the rest of the codebase compares against the string "None"
        self.gmf_model = "None" if gmf_model is None else gmf_model
        self.analysis_type = analysis_type
        self.bg_only = background_only
        self.use_rigidity_grid = use_rigidity_grid
        self.use_with_systematics = use_with_systematics
        self.use_alpha_spline = use_alpha_spline
        self.fit_logE_sys = fit_logE_sys
        self.beta_egmf_max = beta_egmf_max
        self.use_beta_spline = use_beta_spline

        self.stan_model = None
        self.grid_config = None
        self.nthreads_per_chain = None
        # instance-level copy: appending optional keys (e.g. the rigidity-resolved
        # kappa_GMF grid) must not leak into the shared class-level list
        self.fit_input_keys = list(self.fit_input_keys)
        self.fit_inputs = {key: None for key in self.fit_input_keys}
        self.fit = None
        self.inits_dict = None

        # the kappa-only model has no energy / mass / alpha / beta_egmf inputs:
        # drop them so they are neither required nor passed to Stan
        if self.kappa_only:
            self.fit_input_keys = list(self.kappa_only_fit_input_keys)
            self.fit_inputs = {key: None for key in self.fit_input_keys}

        if self.split_lnA_grid:
            for key in ("NEbins_lnAgrid", "lnA_logE_grid_det", "mean_lnA_sys_scale", "var_lnA_sys_scale", "logE_sys_scale", "beta_egmf_ub", "use_beta_spline"):
                self.fit_input_keys.append(key)
                self.fit_inputs[key] = None

    @property
    def split_lnA_grid(self: Self) -> bool:
        """
        True if the Stan model takes a separate lnA model grid
        (lnA_logE_grid, NEbins_lnAgrid) from the detector's lnA bins
        (lnA_logE_grid_det, NEbins), interpolating between them.

        Only energy_mass_spatial_model_alpha_spline.stan does so. That model
        also treats the lnA systematics as one global shift per moment
        (nu * mean/var_lnA_sys_scale, nu ~ N(0, 1)) with statistical-only
        mean/var_lnA_stat_unc, and the energy-scale systematic as a global
        shift nu_logE_sys * logE_sys_scale (nu ~ N(0, 1)).
        """
        return (
            self.analysis_type == self.energy_mass_spatial_type
            and self.use_alpha_spline
            and not self.use_rigidity_grid
            and not self.use_with_systematics
            and not self.bg_only
        )

    def _set_lnA_grid_inputs(
        self: Self,
        lnA_energy_grid: np.ndarray,
        lnA_energy_grid_det: np.ndarray,
    ) -> None:
        """
        Set the lnA energy-grid sizes / grids in fit_inputs.

        Parameters
        ----------
        lnA_energy_grid : np.ndarray
            energies (EeV) on which mean_lnA_grid / var_lnA_grid are tabulated
            (the model grid).
        lnA_energy_grid_det : np.ndarray
            energies (EeV) of the detector's lnA bins, i.e. where
            mean_lnA_det / var_lnA_det are given.
        """
        logE_grid = np.log(np.asarray(lnA_energy_grid, dtype=float))
        logE_grid_det = np.log(np.asarray(lnA_energy_grid_det, dtype=float))

        self.fit_inputs["NEbins"] = len(logE_grid_det)
        self.fit_inputs["lnA_logE_grid"] = logE_grid

        if self.split_lnA_grid:
            # the Stan model interpolates the model grid onto the detector
            # bins, clamping outside its range, so the model grid must cover them
            tol = 1e-6
            if logE_grid_det.min() < logE_grid.min() - tol or logE_grid_det.max() > logE_grid.max() + tol:
                raise ValueError(
                    "The lnA model energy grid "
                    f"[{np.exp(logE_grid.min()):.3g}, {np.exp(logE_grid.max()):.3g}] EeV "
                    "does not cover the detector lnA bins "
                    f"[{np.exp(logE_grid_det.min()):.3g}, {np.exp(logE_grid_det.max()):.3g}] EeV."
                )
            self.fit_inputs["NEbins_lnAgrid"] = len(logE_grid)
            self.fit_inputs["lnA_logE_grid_det"] = logE_grid_det
        elif len(logE_grid) != len(logE_grid_det) or not np.allclose(logE_grid, logE_grid_det):
            raise ValueError(
                "This model requires the lnA model grid to be the detector's lnA "
                "bins (use lnA_energy_gridparams=None). Only the energy_mass_spatial "
                "alpha_spline model supports a separate lnA model grid."
            )

    @property
    def kappa_only(self: Self) -> bool:
        """True if the spatial-only model with effective exposure vs kappa is used."""
        return (
            self.analysis_type == self.spatial_type
            and self.use_alpha_spline
            and not self.use_rigidity_grid
            and not self.bg_only
        )

    def _set_kappa_exposure_grid(
        self: Self,
        data: Data,
        infile: str = "effective_exposures_kappaonly.h5",
    ) -> None:
        """
        Load the kappa-only effective exposure table into fit_inputs.

        The table must have been precomputed (see
        new_uhecr_model/1_precompute_tables/precompute_exposure_tables.py
        with --parameterisation kappa) for this source / detector / GMF model.
        Stan receives natural logs of both kappa and the exposure.
        """
        eff_exp = EffectiveExposureKappaOnly(
            data, gmf_model=self.gmf_model, lazy_gmf_lens=True
        )
        try:
            eff_exp.load_from_tables(infile)
        except (OSError, AssertionError) as e:
            raise RuntimeError(
                f"Could not load kappa-only effective exposure tables from {infile} "
                f"for {data.source.label}, {data.detector.label}, {self.gmf_model}. "
                "Precompute them with precompute_exposure_tables.py "
                f"--parameterisation kappa. ({e})"
            ) from e

        self.fit_inputs["Nkappas"] = eff_exp.Nkappas
        self.fit_inputs["log_kappa_grid"] = np.log(eff_exp.kappa_grid)
        self.fit_inputs["log_wexp_earth_grid"] = np.log(
            eff_exp.effective_exposure.to_value(u.km**2 * u.yr)
        )

    def initialise_grid(
        self : Self,
        energy_gridparams: tuple = (32, 250, 50),
        lnA_energy_gridparams = None,
        effexp_model_kwargs : dict = {
            "beta_egmf_gridparams" : (1e-3, 50, 30),
            "R_gridparams" : (1, 500, 30),
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
            "massids":[201, 402, 1407, 2814, 5626]
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
            given as (E_min, E_max, Nbins), by default None.

            If None, the grid parameters will be set to match the energy grid of the detector's lnA mass model.

            Only the energy_mass_spatial alpha_spline model (`split_lnA_grid`)
            accepts a grid differing from the detector's lnA bins; it must
            then cover the detector bins' energy range.
        effexp_model_kwargs : dict, optional
            The keyword arguments for the effective exposure model.
            By default set to:
            {
                "beta_egmf_gridparams" : (1e-3, 1, 10),
                "R_gridparams" : (1, 500, 25),
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
        if self.kappa_only:
            # only the effective exposure vs kappa is needed: skip the energy /
            # mass grids (PriNCe solvers etc.) entirely
            self._set_kappa_exposure_grid(self.data)
            return

        grid_generator = GridGenerator(data=self.data, gmf_model=self.gmf_model)

        grid_generator.get_effective_exposure_grid(
            effexp_model_kwargs
        )

        # lnA_energy_gridparams=None is resolved by GridGenerator to the
        # detector's own lnA energy bins.
        grid_generator.get_energy_mass_grid(
            energy_gridparams,
            lnA_energy_gridparams,
            src_inj_kwargs,
            bg_inj_kwargs,
            energy_loss_model_kwargs
        )

        grid_generator.get_weighted_exposures()

        self.grid_config = grid_generator.store_grids_to_dict()

        if self.data.detector.lnA_logE_grid is None:
            raise ValueError("The detector's lnA data must be loaded.")
        self._set_lnA_grid_inputs(
            grid_generator.lnA_energy_grid,
            np.exp(self.data.detector.lnA_logE_grid),
        )
        self.fit_inputs["Nalphas"] = grid_generator.Nalphas
        self.fit_inputs["NEs"] = grid_generator.NEs
        self.fit_inputs["NAsrcs"] = grid_generator.Nmass_fracs
        self.fit_inputs["alpha_grid"] = grid_generator.alpha_grid
        self.fit_inputs["logE_grid"] = np.log(grid_generator.energy_grid)
        self.fit_inputs["Emin"] = np.min(grid_generator.energy_grid)
        self.fit_inputs["Emax"] = np.max(grid_generator.energy_grid)
        self.fit_inputs["earth_spectrum_grid"] = grid_generator.spectrum_grid.T
        self.fit_inputs["mean_lnA_grid"] = grid_generator.mean_lnA_grid.T
        self.fit_inputs["var_lnA_grid"] = grid_generator.var_lnA_grid.T
        self.fit_inputs["Nbeta_egmfs"] = len(grid_generator.beta_egmf_grid)
        self.fit_inputs["log10_beta_egmf_grid"] = np.log10(grid_generator.beta_egmf_grid.value)
        self.fit_inputs["log_wexp_earth_grid"] = np.moveaxis(grid_generator.log_wexp_earth_grid, (0,1,2,3), (0,2,3,1))
        self.fit_inputs["log_wexp_src_grid"] = np.moveaxis(grid_generator.log_wexp_src_grid, (0,1,2,3), (0,2,3,1))
        self.fit_inputs["esrc_ratio_grid"] = grid_generator.esrc_ratio_grid.T

        if self.use_alpha_spline:
            self.fit_inputs["alpha_spline_matrix"] = grid_generator.alpha_spline_matrix
            if "alpha_spline_matrix" not in self.fit_input_keys:
                self.fit_input_keys.append("alpha_spline_matrix")

    def _compute_no_gmf_log_wexp_grids(self, simulation : Simulation) -> tuple:
        """
        Recompute log_wexp_earth_grid / log_wexp_src_grid without GMF lensing,
        on the same grids the simulation was initialised with.

        Returns
        -------
        (log_wexp_earth_grid, log_wexp_src_grid), same shapes as the simulation's.
        """
        kwargs = simulation._grid_init_kwargs
        if kwargs is None:
            raise ValueError("Simulation has no grid init kwargs; cannot recompute no-GMF wexp grids.")
        grid_generator = GridGenerator(data=self.data, gmf_model="None")
        grid_generator.get_effective_exposure_grid(kwargs["effexp_model_kwargs"])
        grid_generator.get_energy_mass_grid(
            kwargs["energy_gridparams"],
            kwargs["lnA_energy_gridparams"],
            kwargs["src_inj_kwargs"],
            kwargs["bg_inj_kwargs"],
            kwargs["energy_loss_model_kwargs"],
        )
        grid_generator.get_weighted_exposures()
        for name in ("energy_grid", "alpha_grid", "beta_egmf_grid"):
            a = np.asarray(getattr(grid_generator, name).value if hasattr(getattr(grid_generator, name), "value") else getattr(grid_generator, name))
            b = np.asarray(getattr(simulation, name).value if hasattr(getattr(simulation, name), "value") else getattr(simulation, name))
            if a.shape != b.shape or not np.allclose(a, b):
                raise ValueError(f"No-GMF {name} does not match the simulation's grid.")
        return grid_generator.log_wexp_earth_grid, grid_generator.log_wexp_src_grid

    def load_from_simulation(
        self : Self,
        simulation : Simulation
    ) -> None:
        """
        Load everything directly from the simulation object.

        Parameters
        ----------
        simulation : fancy.simulation.Simulation
            The simulation object that contains the data, model and tables.
        """
        # make sure that the simulation and data are consistent
        if simulation.detector_type != self.data.detector.label:
            raise ValueError("Detector type in simulation and analysis do not match.")
        if simulation.source_type != self.data.source.label:
            raise ValueError("Source type in simulation and analysis do not match.")
        if simulation.mass_model != self.data.detector.mass_model:
            raise ValueError("Mass model in simulation and analysis do not match.")

        if self.kappa_only:
            self._load_kappa_only_from_simulation(simulation)
            return

        self.fit_inputs["Nsrcs"] = simulation.Nsrcs
        self.fit_inputs["D"] = simulation.data.source.distance
        self.fit_inputs["omega_src"] = simulation.source_uvs[:-1,:] # exclude background source
        self.fit_inputs["N"] = simulation.truths['Nex']
        self.fit_inputs["Edet"] = simulation.truths['Edets']
        self.fit_inputs["kappa_ds"] = simulation.truths["kappa_ds"]
        self.fit_inputs["mean_lnA_det"] = simulation.truths['mean_lnA_dets']
        self.fit_inputs["var_lnA_det"] = simulation.truths['var_lnA_dets']
        self.fit_inputs['exposure_factor'] = simulation.truths['exposure_factor']

        self.fit_inputs["Eth"] = np.min(simulation.energy_grid)
        self.fit_inputs["logE_stat_unc"] = simulation.config["logE_stat"]
        self.fit_inputs["mean_lnA_stat_unc"] = simulation.config["mean_lnA_stat"]
        self.fit_inputs["var_lnA_stat_unc"] = simulation.config["var_lnA_stat"]
        self.fit_inputs["logE_sys_unc"] = simulation.config["logE_sys"]
        self.fit_inputs["mean_lnA_sys_unc"] = simulation.config["mean_lnA_sys"]
        self.fit_inputs["var_lnA_sys_unc"] = simulation.config["var_lnA_sys"]
        if self.split_lnA_grid:
            no_shift = np.zeros_like(np.asarray(simulation.config["mean_lnA_stat"], dtype=float))
            self.fit_inputs["mean_lnA_sys_scale"] = simulation.config.get("mean_lnA_sys_scale", no_shift)
            self.fit_inputs["var_lnA_sys_scale"] = simulation.config.get("var_lnA_sys_scale", no_shift)
            self.fit_inputs["logE_sys_scale"] = self.data.detector.logE_sys_scale if self.fit_logE_sys else 0.0
            # no cap -> a value above any beta grid; Stan takes min(grid max, this)
            self.fit_inputs["beta_egmf_ub"] = 1e6 if self.beta_egmf_max is None else float(self.beta_egmf_max)
            self.fit_inputs["use_beta_spline"] = int(self.use_beta_spline)

        self._set_lnA_grid_inputs(
            simulation.lnA_energy_grid, simulation.lnA_energy_grid_det
        )
        self.fit_inputs["Nalphas"] = simulation.Nalphas
        self.fit_inputs["NEs"] = simulation.NEs
        self.fit_inputs["NAsrcs"] = simulation.Nmass_fracs
        self.fit_inputs["alpha_grid"] = simulation.alpha_grid
        self.fit_inputs["logE_grid"] = np.log(simulation.energy_grid)
        self.fit_inputs["Emin"] = np.min(simulation.energy_grid)
        self.fit_inputs["Emax"] = np.max(simulation.energy_grid)
        self.fit_inputs["earth_spectrum_grid"] = simulation.spectrum_grid.T
        self.fit_inputs["mean_lnA_grid"] = simulation.mean_lnA_grid.T
        self.fit_inputs["var_lnA_grid"] = simulation.var_lnA_grid.T
        self.fit_inputs["esrc_ratio_grid"] = simulation.esrc_ratio_grid.T

        # the simulation's wexp grids include GMF lensing of the exposure; if
        # we are fitting without a GMF, recompute them with an un-lensed
        # effective exposure so that Nex is consistent with omega_det (Earth).
        if self.gmf_model == "None" and simulation.gmf_model != "None":
            log_wexp_earth_grid_sim, log_wexp_src_grid_sim = self._compute_no_gmf_log_wexp_grids(simulation)
        else:
            log_wexp_earth_grid_sim = simulation.log_wexp_earth_grid
            log_wexp_src_grid_sim = simulation.log_wexp_src_grid

        # energy_only has no beta_egmf axis in its Stan declaration: it never
        # models EGMF deflection, so log_wexp_earth_grid/log_wexp_src_grid
        # must be collapsed from (Nsrcs[+1], Nalphas, Nbeta_egmfs, NAsrcs) down
        # to (Nsrcs[+1], Nalphas, NAsrcs) before the usual moveaxis to
        # (Nsrcs[+1], NAsrcs, Nalphas). We slice at this simulation's truth
        # beta_egmf (not beta_egmf->0) so Nex matches the exposure the data
        # was actually drawn from, keeping ablated and full fits comparable.
        # beta_egmf is per-source (shape (Nsrcs,), no background entry), so
        # each source's row (k < Nsrcs) is sliced at its own truth value;
        # the background row (k == Nsrcs) uses the grid's first beta value,
        # matching the convention used elsewhere (its wexp_earth_grid row is
        # beta-independent by construction -- see
        # EffectiveExposure.compute_effective_exposure, D=3000 Mpc).
        if self.analysis_type == self.energy_type or self.analysis_type == self.mass_type or self.analysis_type == self.energy_mass_type:
            log10_beta_egmf_truth = np.atleast_1d(simulation.truths["log10_beta_egmf"])
            log10_beta_egmf_grid = np.log10(simulation.beta_egmf_grid.value)

            def _slice_at_truth_beta_egmf(log_wexp_grid: np.ndarray) -> np.ndarray:
                # log_wexp_grid shape: (Nk, Nalphas, Nbeta_egmfs, NAsrcs)
                n_k = log_wexp_grid.shape[0]
                sliced = np.empty(
                    (n_k,) + log_wexp_grid.shape[1:2] + log_wexp_grid.shape[3:]
                )
                for k in range(n_k):
                    log10_beta_k = (
                        log10_beta_egmf_truth[k] if k < len(log10_beta_egmf_truth)
                        else log10_beta_egmf_grid[0]
                    )
                    interpolator_k = interp1d(log10_beta_egmf_grid, log_wexp_grid[k], axis=1)
                    sliced[k] = interpolator_k(log10_beta_k)
                return sliced  # (Nk, Nalphas, NAsrcs)

            self.fit_inputs["log_wexp_earth_grid"] = np.moveaxis(
                _slice_at_truth_beta_egmf(log_wexp_earth_grid_sim), (0, 1, 2), (0, 2, 1)
            )
            self.fit_inputs["log_wexp_src_grid"] = np.moveaxis(
                _slice_at_truth_beta_egmf(log_wexp_src_grid_sim), (0, 1, 2), (0, 2, 1)
            )
            # energy_model.stan has no beta_egmf axis at all: drop these keys
            # rather than leaving them None, or the missing-keys check below
            # would reject a perfectly valid energy_only fit_inputs dict.
            for key in ("Nbeta_egmfs", "log10_beta_egmf_grid"):
                self.fit_inputs.pop(key, None)
                if key in self.fit_input_keys:
                    self.fit_input_keys.remove(key)
        else:
            self.fit_inputs["Nbeta_egmfs"] = len(simulation.beta_egmf_grid)
            self.fit_inputs["log10_beta_egmf_grid"] = np.log10(simulation.beta_egmf_grid.value)
            self.fit_inputs["log_wexp_earth_grid"] = np.moveaxis(log_wexp_earth_grid_sim, (0,1,2,3), (0,2,3,1))
            self.fit_inputs["log_wexp_src_grid"] = np.moveaxis(log_wexp_src_grid_sim, (0,1,2,3), (0,2,3,1))

            # rigidity-grid model variants still declare
            # beta_egmf_prior_mean_log10/sd_log10 unconditionally in their
            # data block (per-source, log10 space); the non-rigidity-grid
            # model now samples a single scalar beta_egmf with its prior
            # fixed in the .stan file, so this is a no-op for that case.
            self._apply_beta_egmf_priors()

        # needed by every *_alpha_spline.stan variant, including the ones
        # without a beta_egmf axis (energy_only, mass_only, energy_mass)
        if self.use_alpha_spline:
            self.fit_inputs["alpha_spline_matrix"] = simulation.alpha_spline_matrix
            if "alpha_spline_matrix" not in self.fit_input_keys:
                self.fit_input_keys.append("alpha_spline_matrix")

        # for omega_det, deal with this depending on gmf model
        if self.gmf_model == "None":
            omega_det = simulation.truths['skycoord_earth_dets']
        else:
            omega_det = simulation.truths['skycoord_gb_truths_bp']
        omega_det.representation_type = "cartesian"
        self.fit_inputs["omega_det"] = omega_det.cartesian.xyz.value.T

        # rigidity-resolved kappa_GMF(R) table, only if it was computed
        # (leaves kappa_ds / omega_det above untouched either way)
        if "log10_gmf_Rgrid" in simulation.truths and "log_kappa_gmf_grid" in simulation.truths:
            self.fit_inputs["Nr_gmf"] = len(simulation.truths["log10_gmf_Rgrid"])
            self.fit_inputs["log10_gmf_Rgrid"] = simulation.truths["log10_gmf_Rgrid"]

            self.fit_inputs["log_kappa_gmf_grid"] = simulation.truths["log_kappa_gmf_grid"]
            # if gmf model is set to None, then we set the log_kappa_gmf_grid to be a constant value of kappa_det, which is the kappa value for the detector
            # in this way we do not break / alter the stan model.
            if self.gmf_model == "None":
                self.fit_inputs["log_kappa_gmf_grid"] = np.full_like(simulation.truths["log_kappa_gmf_grid"], simulation.config['kappa_det'])

            for key in ("log10_gmf_Rgrid", "log_kappa_gmf_grid"):
                if key not in self.fit_input_keys:
                    self.fit_input_keys.append(key)

        self._calculate_grain_size()

        if self.analysis_type == self.spatial_type:
            # then we explicitly give the alphas, mass fracs, and logE_true to the fit, rather than letting them be free parameters
            self.fit_inputs["alphas"] = simulation.truths["alphas"]
            self.fit_inputs["mass_fracs"] = simulation.truths["mass_fracs"].T #?
            self.fit_inputs["logE_true"] = np.log(simulation.truths["Etruths"])

        if self.analysis_type == self.mass_spatial_type:
            # we give the DETECTED energies, since we do not know them apriori
            self.fit_inputs["logE_det"] = np.log(simulation.truths["Edets"])

        if self.analysis_type == self.energy_spatial_type:
            # the variance can be negative in the measurement. So we explicitly set
            # the variance to be zero if it is negative. This is a hack to avoid the Stan model from crashing.
            self.fit_inputs["var_lnA_det"] = np.where(
                simulation.truths['var_lnA_dets'] > 0,
                simulation.truths['var_lnA_dets'],
                0.0
            )

        # warn if anything is None
        missing_keys = [k for k, v in self.fit_inputs.items() if v is None]
        if len(missing_keys) > 0:
            raise ValueError(f"Missing fit inputs: {missing_keys}")



    def _load_kappa_only_from_simulation(self: Self, simulation: Simulation) -> None:
        """Populate the (reduced) fit inputs of the kappa-only model from a simulation."""
        self.fit_inputs["Nsrcs"] = simulation.Nsrcs
        self.fit_inputs["D"] = simulation.data.source.distance
        self.fit_inputs["omega_src"] = simulation.source_uvs[:-1, :]  # exclude background source
        self.fit_inputs["N"] = simulation.truths["Nex"]
        self.fit_inputs["kappa_ds"] = simulation.truths["kappa_ds"]
        self.fit_inputs["exposure_factor"] = simulation.truths["exposure_factor"]

        if self.gmf_model == "None":
            omega_det = simulation.truths["skycoord_earth_dets"]
        else:
            omega_det = simulation.truths["skycoord_gb_truths_bp"]
        omega_det.representation_type = "cartesian"
        self.fit_inputs["omega_det"] = omega_det.cartesian.xyz.value.T

        self._set_kappa_exposure_grid(simulation.data)
        self._calculate_grain_size()

        missing_keys = [k for k, v in self.fit_inputs.items() if v is None]
        if len(missing_keys) > 0:
            raise ValueError(f"Missing fit inputs: {missing_keys}")

    def _calculate_grain_size(self: Self) -> None:
        """Calculate the grain size for the fit."""
        self.fit_inputs["grain_size"] = pick_grain_size(
            self.fit_inputs["N"], self.nthreads_per_chain
        )
        print(f"Using grain size of {self.fit_inputs['grain_size']} for {self.fit_inputs['N']} events.")
        

    def compile_stan_model(self: Self, stan_threads : int = 4) -> None:
        """
        Compile the Stan model for the analysis.
        
        Parameters
        ----------
        stan_threads : int, default=4
            number of stan threads per chain to run
        """
        # get path to the stan file
        if self.use_rigidity_grid:
            if self.analysis_type not in (
                self.energy_mass_spatial_type,
                self.spatial_type,
                self.mass_spatial_type,
                self.energy_spatial_type
            ):
                raise ValueError(
                    "use_rigidity_grid is only supported for "
                    f"analysis_type={self.energy_mass_spatial_type} or "
                    f"{self.spatial_type} or {self.mass_spatial_type} or {self.energy_spatial_type}."
                )
            if self.bg_only:
                raise ValueError(
                    "use_rigidity_grid has no background-only variant."
                )
            stan_ext = "_rigidity_grid.stan"
        elif self.use_with_systematics:
            if self.analysis_type != self.energy_mass_spatial_type:
                raise ValueError(
                    "use_with_systematics is only supported for "
                    f"analysis_type={self.energy_mass_spatial_type}."
                )
            if self.bg_only:
                raise ValueError(
                    "use_with_systematics has no background-only variant."
                )
            stan_ext = "_with_systematics.stan"
        elif self.use_alpha_spline:
            if self.bg_only:
                raise ValueError(
                    "use_alpha_spline has no background-only variant."
                )
            stan_ext = "_alpha_spline.stan"
        else:
            stan_ext = "_background.stan" if self.bg_only else ".stan"

        if self.analysis_type == self.energy_type:
            stan_path = get_path_to_stan_includes(self.energy_type)
            path_to_stan_file = get_path_to_stan_file(
                self.energy_type, f"energy_model{stan_ext}"
            )
        elif self.analysis_type == self.mass_type:
            stan_path = get_path_to_stan_includes(self.mass_type)
            path_to_stan_file = get_path_to_stan_file(
                self.mass_type, f"mass_model{stan_ext}"
            )
        elif self.analysis_type == self.spatial_type:
            stan_path = get_path_to_stan_includes(self.spatial_type)
            path_to_stan_file = get_path_to_stan_file(
                self.spatial_type, f"spatial_model{stan_ext}"
            )
        elif self.analysis_type == self.energy_mass_type:
            stan_path = get_path_to_stan_includes(self.energy_mass_type)
            path_to_stan_file = get_path_to_stan_file(
                self.energy_mass_type, f"energy_mass_model{stan_ext}"
            )
        elif self.analysis_type == self.mass_spatial_type:
            stan_path = get_path_to_stan_includes(self.mass_spatial_type)
            path_to_stan_file = get_path_to_stan_file(
                self.mass_spatial_type, f"mass_spatial_model{stan_ext}"
            )
        elif self.analysis_type == self.energy_spatial_type:
            stan_path = get_path_to_stan_includes(self.energy_spatial_type)
            path_to_stan_file = get_path_to_stan_file(
                self.energy_spatial_type, f"energy_spatial_model{stan_ext}"
            )
        elif self.analysis_type == self.energy_mass_spatial_type:
            stan_path = get_path_to_stan_includes(self.energy_mass_spatial_type)
            path_to_stan_file = get_path_to_stan_file(
                self.energy_mass_spatial_type, f"energy_mass_spatial_model{stan_ext}"
            )
        else:
            raise ValueError(f"Analysis type {self.analysis_type} not recognised.")

        self.nthreads_per_chain = stan_threads
        os.environ["STAN_NUM_THREADS"] = str(self.nthreads_per_chain)

        stanc_options = {"include-paths": str(stan_path)}
        cpp_options = {"STAN_THREADS": True}

        # TODO: compiling stan like this is deprecated, should fix this at some point
        self.stan_model = CmdStanModel(
            stan_file=str(path_to_stan_file), stanc_options=stanc_options, cpp_options=cpp_options, force_compile=True
        )

    def prepare_fit_inputs(self: Self) -> None:
        """Gather inputs from Model, Data and IntegrationTables."""
        # prepare fit inputs

        if self.kappa_only:
            self.fit_inputs["Nsrcs"] = self.data.source.N
            self.fit_inputs["D"] = self.data.source.distance
            self.fit_inputs["omega_src"] = self.data.source.coord.cartesian.xyz.value.T
            self.fit_inputs["N"] = self.data.uhecr.N
            self.fit_inputs["kappa_ds"] = self.data.uhecr.kappa_ds
            self.fit_inputs["exposure_factor"] = self.data.uhecr.exposure
            if self.gmf_model == "None":
                self.fit_inputs["omega_det"] = self.data.uhecr.coord.cartesian.xyz.value.T
            else:
                self.fit_inputs["omega_det"] = self.data.uhecr.coords_gb.cartesian.xyz.value.T
            # exposure table is loaded by initialise_grid
            self._calculate_grain_size()
            missing_keys = [k for k, v in self.fit_inputs.items() if v is None]
            if len(missing_keys) > 0:
                raise ValueError(f"Missing fit inputs: {missing_keys}")
            return

        self.fit_inputs["Nsrcs"] = self.data.source.N
        self.fit_inputs["D"] = self.data.source.distance
        self.fit_inputs["omega_src"] = self.data.source.coord.cartesian.xyz.value.T
        self.fit_inputs["N"] = self.data.uhecr.N
        self.fit_inputs["Edet"] = self.data.uhecr.energy
        self.fit_inputs["kappa_ds"] = self.data.uhecr.kappa_ds
        self.fit_inputs["mean_lnA_det"] = self.data.detector.mean_lnA
        self.fit_inputs["var_lnA_det"] = self.data.detector.var_lnA
        self.fit_inputs['exposure_factor'] = self.data.uhecr.exposure

        self.fit_inputs["Eth"] = self.data.detector.Eth
        self.fit_inputs["logE_stat_unc"] = self.data.detector.logE_stat
        self.fit_inputs["logE_sys_unc"] = self.data.detector.logE_sys
        self.fit_inputs["mean_lnA_stat_unc"] = self.data.detector.mean_lnA_stat
        self.fit_inputs["var_lnA_stat_unc"] = self.data.detector.var_lnA_stat
        self.fit_inputs["mean_lnA_sys_unc"] = self.data.detector.mean_lnA_sys
        self.fit_inputs["var_lnA_sys_unc"] = self.data.detector.var_lnA_sys
        if self.split_lnA_grid:
            # statistical-only errors; the systematics enter as a global shift
            self.fit_inputs["mean_lnA_stat_unc"] = self.data.detector.mean_lnA_stat_only
            self.fit_inputs["var_lnA_stat_unc"] = self.data.detector.var_lnA_stat_only
            self.fit_inputs["mean_lnA_sys_scale"] = self.data.detector.mean_lnA_sys_scale
            self.fit_inputs["var_lnA_sys_scale"] = self.data.detector.var_lnA_sys_scale
            # energy-scale systematic: either fitted (nu_logE_sys) or the
            # detector's f_E_sys applied as a fixed shift
            if self.fit_logE_sys:
                self.fit_inputs["logE_sys_unc"] = 0.0
                self.fit_inputs["logE_sys_scale"] = self.data.detector.logE_sys_scale
            else:
                self.fit_inputs["logE_sys_scale"] = 0.0
            self.fit_inputs["beta_egmf_ub"] = 1e6 if self.beta_egmf_max is None else float(self.beta_egmf_max)
            self.fit_inputs["use_beta_spline"] = int(self.use_beta_spline)

        # for omega_det, deal with this depending on gmf model
        if self.gmf_model == "None":
            self.fit_inputs["omega_det"] = self.data.uhecr.coord.cartesian.xyz.value.T
        else:
            self.fit_inputs["omega_det"] = self.data.uhecr.coords_gb.cartesian.xyz.value.T

        # rigidity-resolved kappa_GMF(R) table, only if it was loaded
        # (leaves kappa_ds / omega_det above untouched either way)
        if self.data.uhecr.log10_gmf_Rgrid is not None and self.data.uhecr.log_kappa_gmf_grid is not None:
            self.fit_inputs["Nr_gmf"] = len(self.data.uhecr.log10_gmf_Rgrid)
            self.fit_inputs["log10_gmf_Rgrid"] = self.data.uhecr.log10_gmf_Rgrid
            self.fit_inputs["log_kappa_gmf_grid"] = self.data.uhecr.log_kappa_gmf_grid
            for key in ("Nr_gmf", "log10_gmf_Rgrid", "log_kappa_gmf_grid"):
                if key not in self.fit_input_keys:
                    self.fit_input_keys.append(key)

        # per-source beta_egmf prior, resolved from each source's
        # egmf_structure category.
        self._apply_beta_egmf_priors()

        # calculate the grain size
        self._calculate_grain_size()

        # warn if anything is None
        missing_keys = [k for k, v in self.fit_inputs.items() if v is None]
        if len(missing_keys) > 0:
            raise ValueError(f"Missing fit inputs: {missing_keys}")

    def _apply_beta_egmf_priors(self: Self) -> None:
        """
        Resolve the per-source beta_egmf prior and write it into
        self.fit_inputs/self.fit_input_keys. Shared by prepare_fit_inputs and
        load_from_simulation, since the Stan data block requires this field
        for every rigidity-grid model variant (and its non-rigidity-grid
        linear-space equivalent) regardless of which method built the rest
        of fit_inputs.

        The rigidity-grid model samples in log10 space; the non-rigidity-grid
        model samples beta_egmf directly in linear (nG Mpc^1/2) space, so the
        two variants need differently scaled prior parameters.
        """
        mean_log10, sd_log10, has_category = self.get_beta_egmf_priors()
        if self.use_rigidity_grid:
            self.fit_inputs["beta_egmf_prior_mean_log10"] = mean_log10
            self.fit_inputs["beta_egmf_prior_sd_log10"] = sd_log10
            for key in ("beta_egmf_prior_mean_log10", "beta_egmf_prior_sd_log10"):
                if key not in self.fit_input_keys:
                    self.fit_input_keys.append(key)
        else:
            # Reverted to the single shared beta_egmf model (2026-09-28):
            # the non-rigidity-grid Stan model now samples one scalar
            # beta_egmf ~ normal(0, 10), written directly in the .stan
            # file's model block -- no per-source prior fields to pass in.
            pass

    def get_beta_egmf_priors(
        self: Self,
        default_mean: float = 0.0,
        default_sd: float = 2.0,
    ) -> tuple:
        """
        Resolve per-source priors on log10(beta_egmf) from each source's
        egmf_structure category (see fancy.utils.egmf_priors).

        Used by prepare_fit_inputs to populate beta_egmf_prior_mean_log10 /
        beta_egmf_prior_sd_log10 for the rigidity-grid model, which samples
        log10_beta_egmf directly (per source). The non-rigidity-grid model
        samples a single scalar beta_egmf shared across all sources, with a
        fixed normal(0, 10) prior written directly in the .stan file, so
        this method's output is unused for that model variant.

        Parameters
        ----------
        default_mean, default_sd: float
            prior mean/sd (in log10 space) used for sources with no
            egmf_structure set, matching the previous global prior
            (log10_beta_egmf ~ normal(0, 2)).

        Returns
        -------
        (mean_log10, sd_log10, has_category): tuple of np.ndarray, each
        shape (Nsrcs,). has_category is a bool mask, True where a source has
        an assigned egmf_structure (as opposed to falling back to
        default_mean/default_sd).
        """
        egmf_structure = self.data.source.egmf_structure
        n_src = self.data.source.N

        if egmf_structure is None:
            means = np.full(n_src, default_mean)
            sds = np.full(n_src, default_sd)
            has_category = np.zeros(n_src, dtype=bool)
            return means, sds, has_category

        means = np.empty(n_src)
        sds = np.empty(n_src)
        has_category = np.empty(n_src, dtype=bool)
        for i, structure in enumerate(egmf_structure):
            if structure in (None, "", "none"):
                means[i], sds[i] = default_mean, default_sd
                has_category[i] = False
            else:
                means[i], sds[i] = get_log10_beta_egmf_prior(structure)
                has_category[i] = True
        return means, sds, has_category

    def _prior_inits(self: Self, chains: int, seed: Union[int, None] = None) -> list:
        """
        One initial-value dict per chain, drawn from the priors of
        energy_mass_spatial_model_alpha_spline.stan (truncated to the
        parameter bounds). The per-event true energies all start at the median
        of the detected log energies; the per-event nu_lnAs are drawn from their
        N(0, 1) prior.
        """
        rng = np.random.default_rng(seed)
        nk = self.fit_inputs["Nsrcs"] + 1
        na = self.fit_inputs["NAsrcs"]
        alpha_min = float(np.min(self.fit_inputs["alpha_grid"]))
        alpha_max = float(np.max(self.fit_inputs["alpha_grid"]))
        beta_min = float(10 ** np.min(self.fit_inputs["log10_beta_egmf_grid"]))
        beta_max = float(10 ** np.max(self.fit_inputs["log10_beta_egmf_grid"]))
        if self.split_lnA_grid:
            beta_max = min(beta_max, float(self.fit_inputs["beta_egmf_ub"]))

        def truncated_normal(mu, sigma, lo, hi, size=None):
            a, b = (lo - mu) / sigma, (hi - mu) / sigma
            return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size, random_state=rng)

        logE_init = float(np.median(np.log(self.fit_inputs["Edet"])))
        inits = []
        for _ in range(chains):
            init = {
                "alphas": truncated_normal(0.0, 2.0, alpha_min, alpha_max, size=nk),
                "mass_fracs": rng.dirichlet(np.full(na, 2.0), size=nk),
                "flux_frac": rng.dirichlet(np.full(nk, 2.0)),
                "log10_Ftot": float(rng.normal(-1.0, 3.0)),
                "beta_egmf": float(truncated_normal(0.0, 10.0, beta_min, beta_max)),
                "logE_true": np.full(self.fit_inputs["N"], logE_init),
                "nu_lnAs": rng.normal(0.0, 1.0, size=self.fit_inputs["N"]),
            }
            if self.split_lnA_grid:
                init.update({k: float(rng.normal()) for k in ("nu_mean_lnA_sys", "nu_var_lnA_sys", "nu_logE_sys")})
            inits.append(init)
        return inits

    def fit_model(
        self: Self,
        iterations: int = 1000,
        chains: int = 4,
        seed: Union[int, None] = None,
        warmup: Union[int, None] = None,
        inits : Union[dict, None] = None,
        init_model : Union[str, None] = None,
        **kwargs: dict,
    ):
        """
        Fit a model.

        Parameters
        ----------
        iterations: int, default=1000
            number of iterations
        chains: int, default=4
            number of chains
        seed: int, default=None
            seed for RNG
        output_dir: str, default=None
            output directory for raw stan outputs
        warmup : int, default=None
            number of iterations used for warmup
        inits : Union[dict, None], default=None
            initial values for the parameters.
            If None, the default initial values are used.
        init_model : Union[str, None], default=None
            whether to use the variational inference (VI) output as initial values for the parameters.
            Default is None. If "pathfinder", the PathFinder output will be used as initial values for the parameters.
            If "stacking", each chain gets its own initial values drawn from the
            model's priors (see `_prior_inits`), so that the chains start
            dispersed and can find different posterior modes; the chains are
            then meant to be combined by chain stacking (see
            new_uhecr_model/3_simulate_and_fit/stack_chains.py) rather than
            pooled.
        kwargs : dict
            additional arguments to pass to the fit method

        Returns
        -------
        fit : cmdstanpy.stanfit.mcmc.CmdStanMCMC
            The fit output from stan that contains the samples
            as well as other diagnostic information.

            See https://cmdstanpy.readthedocs.io/en/v1.2.0/api.html#cmdstanpy.CmdStanMCMC
            for more information on what can be accessed

        Additional arguments that match the keyword arguments
        in cmdstanpy.model.model.sample can also be passed.
        See https://cmdstanpy.readthedocs.io/en/v1.2.0/api.html#cmdstanpy.CmdStanModel.sample
        for more details.
        """
        # warn if anything is None
        missing_keys = [k for k, v in self.fit_inputs.items() if v is None]
        if len(missing_keys) > 0:
            raise ValueError(f"Missing fit inputs: {missing_keys}")
        
        # compile the stan model
        if self.stan_model is None:
            raise ValueError("Run `compile_stan_model` first!")

        if warmup is None:
            print("Setting warmup to 1000.")
            warmup = 1000

        if init_model is None and self.kappa_only:
            inits_dict = self._kappa_only_inits()
        elif init_model is None:
            print("Not using variational inference (VI) output as initial values for the parameters.")
            inits_dict={
                "alphas" : np.zeros((self.fit_inputs["Nsrcs"]+1)),
                "mass_fracs" : np.full((self.fit_inputs["Nsrcs"]+1, self.fit_inputs["NAsrcs"]), 1 / self.fit_inputs["NAsrcs"]),
                # "mass_fracs" : np.tile(np.array([0.1, 0.3, 0.3, 0.3])[:, None], (1, self.fit_inputs["Nsrcs"] + 1)).T,
                "logE_true" : [np.median(np.log(self.fit_inputs["Edet"]))] * self.fit_inputs['N'],
                "flux_frac" : np.full((self.fit_inputs["Nsrcs"]+1), 1 / (self.fit_inputs["Nsrcs"]+1)),
                "log10_Ftot" : -2,
                **({"log10_beta_egmf": np.zeros(self.fit_inputs["Nsrcs"])} if self.use_rigidity_grid else {"beta_egmf": 1.0}),
                "nu_lnAs": np.full(self.fit_inputs['N'], 0.5),
                **({"nu_mean_lnA_sys": 0.0, "nu_var_lnA_sys": 0.0, "nu_logE_sys": 0.0} if self.split_lnA_grid else {}),
            }
        elif init_model == "stacking":
            print("Drawing per-chain initial values from the priors.")
            inits_dict = self._prior_inits(chains, seed)
        elif init_model == "pathfinder":
            print("Using PathFinder variational inference (VI) output as initial values for the parameters.")
            pathfinder = self.stan_model.pathfinder(
                data=self.fit_inputs,
            )
            # PathFinder's own approximation can occasionally place a draw
            # somewhere pathological (e.g. a simplex boundary, or a
            # beta_egmf/log10_Ftot combination that makes a downstream
            # truncated-density evaluate to -inf/nan) -- blindly using
            # create_inits()[0] with no validation meant a single bad draw
            # would seed every chain identically and make sample() fail
            # near-instantly on ALL chains (observed as an opaque cmdstanpy
            # "Operation not permitted" error, since show_console=True is
            # only set on this pathfinder() call, not on the later
            # sample() call where the actual crash happens). Confirmed
            # non-deterministic in practice: an unmodified re-run of the
            # exact same script sometimes fails and sometimes succeeds,
            # since create_inits() draws a fresh (unseeded) sample from the
            # PathFinder approximation every call.
            #
            # Fix: request one init candidate per requested chain, and
            # validate each with the model's own log_prob() (documented by
            # cmdstanpy as diagnostics-only, which is exactly this use) --
            # a candidate is accepted only if log_prob() doesn't raise (a
            # constraint violation, e.g. an invalid simplex) AND every
            # returned lp__/gradient value is finite. Falls back to the
            # validated fixed interior init dict (the init_model=None
            # branch above -- already known to work reliably for this
            # model, see e.g. map_laplace_truths.py's docstring) if none of
            # the candidates validate, rather than handing sample() a point
            # known to be bad.
            #
            # Each chain must get its OWN PathFinder draw (identical inits
            # across chains defeats the point of running multiple chains --
            # e.g. R-hat can no longer detect between-chain disagreement).
            # cmdstanpy's create_inits(chains=N) draws N candidates, but a
            # candidate can fail validation, so requesting exactly `chains`
            # candidates isn't enough to guarantee `chains` valid ones.
            # Oversample instead: pull chains*4 candidates from the same
            # PathFinder approximation and keep validating (drawing further
            # batches if needed) until we have `chains` accepted, unique
            # per-chain init dicts, or we give up and fall back.
            oversample_factor = 4
            max_attempts = 5  # hard cap so a pathological run can't loop forever
            validated_inits = []
            attempt = 0
            n_requested_total = 0
            while len(validated_inits) < chains and attempt < max_attempts:
                n_candidates = chains * oversample_factor
                candidates = pathfinder.create_inits(chains=n_candidates)
                if isinstance(candidates, dict):
                    candidates = [candidates]
                n_requested_total += len(candidates)

                for candidate in candidates:
                    if len(validated_inits) >= chains:
                        break
                    try:
                        log_prob_df = self.stan_model.log_prob(candidate, data=self.fit_inputs)
                    except Exception as e:
                        print(f"  PathFinder candidate rejected (log_prob raised: {e}).")
                        continue

                    if np.all(np.isfinite(log_prob_df.to_numpy())):
                        print(f"  PathFinder candidate accepted for chain {len(validated_inits)} "
                              f"(lp__={log_prob_df['lp__'].iloc[0]:.3f}).")
                        validated_inits.append(candidate)
                    else:
                        print("  PathFinder candidate rejected (non-finite lp__/gradient).")

                attempt += 1

            if len(validated_inits) < chains:
                print(f"  Only {len(validated_inits)}/{chains} valid PathFinder candidates found "
                      f"after {n_requested_total} draws over {attempt} attempt(s) -- "
                      "falling back to the fixed interior init dict (broadcast to all "
                      "chains) instead of an init point known to be bad.")
                inits_dict = self._kappa_only_inits() if self.kappa_only else {
                    "alphas": np.full(self.fit_inputs["Nsrcs"] + 1, 0.0),
                    "mass_fracs": np.full((self.fit_inputs["Nsrcs"] + 1, self.fit_inputs["NAsrcs"]), 1 / self.fit_inputs["NAsrcs"]),
                    "logE_true": [np.median(np.log(self.fit_inputs["Edet"]))] * self.fit_inputs['N'],
                    "flux_frac": np.full(self.fit_inputs["Nsrcs"] + 1, 1 / (self.fit_inputs["Nsrcs"] + 1)),
                    "log10_Ftot": -2,
                    **({"log10_beta_egmf": np.zeros(self.fit_inputs["Nsrcs"])} if self.use_rigidity_grid else {"beta_egmf": 0.5}),
                    "nu_lnAs": np.full(self.fit_inputs['N'], 0.5),
                    **({"nu_mean_lnA_sys": 0.0, "nu_var_lnA_sys": 0.0, "nu_logE_sys": 0.0} if self.split_lnA_grid else {}),
                }
            else:
                # one distinct, validated init dict per chain
                inits_dict = validated_inits

        elif init_model == "vi":
            print("Using variational inference (VI) output as initial values for the parameters.")
            vi = self.stan_model.variational(
                data=self.fit_inputs,
                required_converged=False, # we don't require convergence for the VI output, just want to use it as initial values
                seed=seed,
                **kwargs,
            )
            inits_dict = {var: samples.mean(axis=0) for var, samples in vi.stan_variables(mean=False).items()}

        # raise Exception("The following code is not yet implemented for cmdstanpy backend. Please use pystan backend for now.")

        # different parameter names and configurations for background only fits
        # if self.spatial_type:
        #     inits_dict.pop("alphas")
        #     inits_dict.pop("mass_fracs")
        #     inits_dict.pop("logE_true")
        # elif self.energy_type:
        #     inits_dict.pop("beta_egmf")
        #     inits_dict.pop("nu_lnAs")
        if self.bg_only:
            # inits_dict.pop("flux_frac")
            # inits_dict.pop("log10_Ftot")
            # inits_dict.pop("alphas")
            # inits_dict.pop("mass_fracs")
            # inits_dict may now be either a single dict (broadcast to all
            # chains) or a list of one dict per chain (pathfinder init_model,
            # when per-chain validated candidates were found) -- add the
            # bg-only keys to every dict in either case.
            bg_only_extra = {
                "alpha_bg": 1,
                "mass_fracs_bg": np.full((self.fit_inputs["NAsrcs"]), 1 / self.fit_inputs["NAsrcs"]),
            }
            if isinstance(inits_dict, list):
                for d in inits_dict:
                    d.update(bg_only_extra)
            else:
                inits_dict.update(bg_only_extra)
        if inits is not None:
            print("Using user-provided initial values for the parameters.")
            inits_dict = inits

        # print("Initial values for the parameters:")
        # for key, value in inits_dict.items():
        #     print(f"  {key}: {value}")
            
        
        # keep the resolved inits (fixed / pathfinder / VI-derived / user-provided)
        # so they can be recovered later via `save`, without needing to re-run
        # pathfinder/VI or re-derive the fixed dict.
        self.inits_dict = inits_dict

        # fit
        print("Performing fitting...")
        self.fit = self.stan_model.sample(
            data=self.fit_inputs,
            iter_sampling=iterations,
            chains=chains,
            seed=seed,
            iter_warmup=warmup,
            inits=inits_dict,
            parallel_chains=chains,
            threads_per_chain=self.nthreads_per_chain,
            **kwargs,
        )

        # # Diagnositics
        # print("Checking all diagnostics...")
        # print(self.fit.diagnose())

        self.chain = self.fit.stan_variables()
        print("Done!")
        return self.fit

    def _kappa_only_inits(self: Self) -> dict:
        """Fixed interior init for the kappa-only model's parameters."""
        return {
            "flux_frac": np.full(self.fit_inputs["Nsrcs"] + 1, 1 / (self.fit_inputs["Nsrcs"] + 1)),
            "log10_Ftot": -2,
            "kappa_egmf": 100.0,
        }

    @classmethod
    def load(
        cls,
        infile: str,
        gmf_model: str = "None",
        lnA_moments_filename: str = "lnA_moments_data.h5",
    ) -> Self:
        """
        Reconstruct an ``Analysis`` instance from a file written by :meth:`save`.

        Rebuilds ``data`` (source/uhecr/detector), ``fit_inputs``, ``inits_dict``,
        and a read-only :class:`FitResult` shim (assigned to ``self.fit``,
        exposing ``stan_variable``/``stan_variables``/``method_variables``/
        ``diagnose`` etc.) directly from the saved HDF5 groups. This never
        re-runs HMC and never unpickles a raw ``CmdStanMCMC`` -- everything
        needed for downstream plotting (spectra, skymaps, posteriors, PPCs) is
        already present in the file. ``stan_model`` is left as ``None`` since
        the Stan model is not needed (and not compiled) for plotting-only use.

        Parameters
        ----------
        infile : str
            path to a ``.h5`` file previously written by :meth:`save`.
        gmf_model : str, default="None"
            the GMF model this fit used. Not stored in the saved file (it's a
            load-time argument to ``Uhecr``/``Data``, not a persisted
            attribute), so callers who need it downstream (e.g. PPC
            generation) must pass it back in explicitly.
        lnA_moments_filename : str, default="lnA_moments_data.h5"
            lnA moments file for the reconstructed ``Detector`` (also not
            persisted in the Analysis output file). Pass the simulation's
            "...sim..." lnA h5 path if this fit used simulated data, matching
            the ``lnA_moments_filename`` used at fit time.

        Returns
        -------
        analysis : Analysis
        """
        data = Data()
        data.load_from_analysis_file(infile, lnA_moments_filename=lnA_moments_filename)

        with h5py.File(infile, "r") as f:
            fit_inputs = {key: value[()] for key, value in f["fit"]["input"].items()}
            samples = {key: value[()] for key, value in f["fit"]["samples"].items()}
            diagnostics = {}
            for key, value in f["fit"]["diagnostics"].items():
                diagnostics[key] = value[()]
            if "diagnose_report" in f["fit"]["diagnostics"].attrs:
                diagnostics["diagnose_report"] = f["fit"]["diagnostics"].attrs["diagnose_report"]
            inits_dict = None
            if "inits" in f["fit"]:
                inits_handle = f["fit"]["inits"]
                # per-chain layout (list of dicts saved as chain_0, chain_1, ...)
                # vs. the flat single-dict layout -- see `save`.
                if all(k.startswith("chain_") for k in inits_handle):
                    n_chains = len(inits_handle.keys())
                    inits_dict = [
                        {key: value[()] for key, value in inits_handle[f"chain_{i}"].items()}
                        for i in range(n_chains)
                    ]
                else:
                    inits_dict = {key: value[()] for key, value in inits_handle.items()}

        log_post = samples.pop("log_post")
        diagnostics["log_post"] = log_post

        use_rigidity_grid = "log10_gmf_Rgrid" in fit_inputs
        use_alpha_spline = "alpha_spline_matrix" in fit_inputs
        # the kappa-only spatial model has no alpha_spline_matrix, but is
        # identified by its kappa grid
        kappa_only = "log_kappa_grid" in fit_inputs

        analysis = cls(
            data,
            gmf_model=gmf_model,
            analysis_type=cls.spatial_type if kappa_only else cls.energy_mass_spatial_type,
            use_rigidity_grid=use_rigidity_grid,
            use_alpha_spline=use_alpha_spline or kappa_only,
        )
        analysis.fit_inputs = fit_inputs
        analysis.fit_input_keys = list(fit_inputs.keys())
        analysis.inits_dict = inits_dict
        analysis.chain = samples
        analysis.fit = FitResult(samples, diagnostics)

        return analysis

    def save(self: Self, outfile: str) -> None:
        """
        Write the analysis output to an output file.

        Parameter:
        ----------
        outfile : str
            the path to the output file where the
            analysis outputs are to be stored.
            Must be a .h5 format.
        """
        # ensure that the file is a h5 format
        assert outfile.find(".h5"), f"Output file {outfile} must have a .h5 extension!"

        with h5py.File(outfile, "w") as f:
            source_handle = f.create_group("source")
            if self.data.source:
                self.data.source.save(source_handle)

            uhecr_handle = f.create_group("uhecr")
            if self.data.uhecr:
                self.data.uhecr.save(uhecr_handle, self.analysis_type)

            detector_handle = f.create_group("detector")
            if self.data.detector:
                self.data.detector.save(detector_handle)

            if self.fit is None:
                raise ValueError("Run `fit_model` first!")
            fit_handle = f.create_group("fit")
            # fit inputs
            fit_input_handle = fit_handle.create_group("input")
            for key, value in self.fit_inputs.items():
                create_dataset_compressed(fit_input_handle, key, value)

            # initial values used to start the chains (fixed dict / pathfinder
            # / VI-derived / user-provided -- see `fit_model`), so the exact
            # inits can be recovered without re-running pathfinder/VI.
            # `inits_dict` is either a single dict (broadcast to all chains)
            # or a list of one dict per chain (pathfinder init_model, when
            # per-chain validated candidates were found) -- store the list
            # case as one subgroup per chain so each chain's distinct inits
            # survive a round trip through `load`.
            if self.inits_dict is not None:
                inits_handle = fit_handle.create_group("inits")
                if isinstance(self.inits_dict, list):
                    for chain_idx, chain_inits in enumerate(self.inits_dict):
                        chain_handle = inits_handle.create_group(f"chain_{chain_idx}")
                        for key, value in chain_inits.items():
                            create_dataset_compressed(chain_handle, key, value)
                else:
                    for key, value in self.inits_dict.items():
                        create_dataset_compressed(inits_handle, key, value)

            # samples (includes loglik_event, since it is a generated quantity
            # returned by stan_variables())
            samples = fit_handle.create_group("samples")
            for key, value in self.chain.items():
                create_dataset_compressed(samples, key, value)

            # log posterior
            create_dataset_compressed(
                samples, "log_post", self.fit.method_variables()["lp__"]
            )

            # sampler diagnostics (small per-chain/per-draw arrays, plus the
            # cmdstan `diagnose` report) -- kept so the raw CmdStanMCMC / its
            # pickle is no longer needed to check divergences, treedepth,
            # E-BFMI, step size, etc.
            diagnostics = fit_handle.create_group("diagnostics")
            diagnostics.attrs["diagnose_report"] = self.fit.diagnose()
            diagnostics.create_dataset("divergences", data=self.fit.divergences)
            diagnostics.create_dataset("max_treedepths", data=self.fit.max_treedepths)
            diagnostics.create_dataset("step_size", data=self.fit.step_size)
            method_vars = self.fit.method_variables()
            for key in ("divergent__", "treedepth__", "energy__", "n_leapfrog__"):
                if key in method_vars:
                    create_dataset_compressed(diagnostics, key, method_vars[key])