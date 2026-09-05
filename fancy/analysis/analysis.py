"""Container to manage the inputs and outputs of the fits."""
import os
import pickle
from typing import Union

import cmdstanpy
import h5py
import numpy as np
from scipy.interpolate import interp1d
from cmdstanpy import CmdStanModel
from typing_extensions import Self  # change to typing for py>3.11
import arviz as az

from fancy.interfaces.data import Data
from fancy.interfaces.grid_generator import GridGenerator
from fancy.simulation import Simulation
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

    def __init__(
        self: Self,
        data: Data,
        gmf_model : str = "None",
        analysis_type: str = energy_mass_spatial_type,
        background_only : bool = False,
        use_rigidity_grid : bool = False,
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
        """
        self.data = data
        self.gmf_model = gmf_model
        self.analysis_type = analysis_type
        self.bg_only = background_only
        self.use_rigidity_grid = use_rigidity_grid

        self.stan_model = None
        self.grid_config = None
        self.nthreads_per_chain = None
        # instance-level copy: appending optional keys (e.g. the rigidity-resolved
        # kappa_GMF grid) must not leak into the shared class-level list
        self.fit_input_keys = list(self.fit_input_keys)
        self.fit_inputs = {key: None for key in self.fit_input_keys}
        self.fit = None
        self.inits_dict = None

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
        grid_generator = GridGenerator(data=self.data, gmf_model=self.gmf_model)

        grid_generator.get_effective_exposure_grid(
            effexp_model_kwargs
        )

        # for the lnA grid parameters, use the detector's mass model
        if lnA_energy_gridparams is not None:
            lnA_energy_gridparams = lnA_energy_gridparams
        else:
            lnA_energy_gridparams = (
                np.min(np.exp(self.data.detector.lnA_logE_grid)),
                np.max(np.exp(self.data.detector.lnA_logE_grid)),
                len(self.data.detector.lnA_logE_grid),
            )

        grid_generator.get_energy_mass_grid(
            energy_gridparams,
            lnA_energy_gridparams,
            src_inj_kwargs,
            bg_inj_kwargs,
            energy_loss_model_kwargs
        )

        grid_generator.get_weighted_exposures()

        self.grid_config = grid_generator.store_grids_to_dict()

        self.fit_inputs["NEbins"] = len(grid_generator.lnA_energy_grid)
        self.fit_inputs["Nalphas"] = grid_generator.Nalphas
        self.fit_inputs["NEs"] = grid_generator.NEs
        self.fit_inputs["NAsrcs"] = grid_generator.Nmass_fracs
        self.fit_inputs["alpha_grid"] = grid_generator.alpha_grid
        self.fit_inputs["logE_grid"] = np.log(grid_generator.energy_grid)
        self.fit_inputs["Emin"] = np.min(grid_generator.energy_grid)
        self.fit_inputs["Emax"] = np.max(grid_generator.energy_grid)
        self.fit_inputs["earth_spectrum_grid"] = grid_generator.spectrum_grid.T
        self.fit_inputs["lnA_logE_grid"] = np.log(grid_generator.lnA_energy_grid)
        self.fit_inputs["mean_lnA_grid"] = grid_generator.mean_lnA_grid.T
        self.fit_inputs["var_lnA_grid"] = grid_generator.var_lnA_grid.T
        self.fit_inputs["Nbeta_egmfs"] = len(grid_generator.beta_egmf_grid)
        self.fit_inputs["log10_beta_egmf_grid"] = np.log10(grid_generator.beta_egmf_grid.value)
        self.fit_inputs["log_wexp_earth_grid"] = np.moveaxis(grid_generator.log_wexp_earth_grid, (0,1,2,3), (0,2,3,1))
        self.fit_inputs["log_wexp_src_grid"] = np.moveaxis(grid_generator.log_wexp_src_grid, (0,1,2,3), (0,2,3,1))
        self.fit_inputs["esrc_ratio_grid"] = grid_generator.esrc_ratio_grid.T

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

        self.fit_inputs["NEbins"] = len(simulation.lnA_energy_grid)
        self.fit_inputs["Nalphas"] = simulation.Nalphas
        self.fit_inputs["NEs"] = simulation.NEs
        self.fit_inputs["NAsrcs"] = simulation.Nmass_fracs
        self.fit_inputs["alpha_grid"] = simulation.alpha_grid
        self.fit_inputs["logE_grid"] = np.log(simulation.energy_grid)
        self.fit_inputs["Emin"] = np.min(simulation.energy_grid)
        self.fit_inputs["Emax"] = np.max(simulation.energy_grid)
        self.fit_inputs["earth_spectrum_grid"] = simulation.spectrum_grid.T
        self.fit_inputs["lnA_logE_grid"] = np.log(simulation.lnA_energy_grid)
        self.fit_inputs["mean_lnA_grid"] = simulation.mean_lnA_grid.T
        self.fit_inputs["var_lnA_grid"] = simulation.var_lnA_grid.T
        self.fit_inputs["esrc_ratio_grid"] = simulation.esrc_ratio_grid.T

        # energy_only has no beta_egmf axis in its Stan declaration: it never
        # models EGMF deflection, so log_wexp_earth_grid/log_wexp_src_grid
        # must be collapsed from (Nsrcs[+1], Nalphas, Nbeta_egmfs, NAsrcs) down
        # to (Nsrcs[+1], Nalphas, NAsrcs) before the usual moveaxis to
        # (Nsrcs[+1], NAsrcs, Nalphas). We slice at this simulation's truth
        # beta_egmf (not beta_egmf->0) so Nex matches the exposure the data
        # was actually drawn from, keeping ablated and full fits comparable.
        if self.analysis_type == self.energy_type or self.analysis_type == self.mass_type or self.analysis_type == self.energy_mass_type:
            log10_beta_egmf_truth = simulation.truths["log10_beta_egmf"]
            log10_beta_egmf_grid = np.log10(simulation.beta_egmf_grid.value)

            def _slice_at_truth_beta_egmf(log_wexp_grid: np.ndarray) -> np.ndarray:
                # log_wexp_grid shape: (Nk, Nalphas, Nbeta_egmfs, NAsrcs)
                interpolator = interp1d(
                    log10_beta_egmf_grid, log_wexp_grid, axis=2
                )
                return interpolator(log10_beta_egmf_truth)  # (Nk, Nalphas, NAsrcs)

            self.fit_inputs["log_wexp_earth_grid"] = np.moveaxis(
                _slice_at_truth_beta_egmf(simulation.log_wexp_earth_grid), (0, 1, 2), (0, 2, 1)
            )
            self.fit_inputs["log_wexp_src_grid"] = np.moveaxis(
                _slice_at_truth_beta_egmf(simulation.log_wexp_src_grid), (0, 1, 2), (0, 2, 1)
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
            self.fit_inputs["log_wexp_earth_grid"] = np.moveaxis(simulation.log_wexp_earth_grid, (0,1,2,3), (0,2,3,1))
            self.fit_inputs["log_wexp_src_grid"] = np.moveaxis(simulation.log_wexp_src_grid, (0,1,2,3), (0,2,3,1))

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

        # calculate the grain size
        self._calculate_grain_size()
        
        # warn if anything is None
        missing_keys = [k for k, v in self.fit_inputs.items() if v is None]
        if len(missing_keys) > 0:
            raise ValueError(f"Missing fit inputs: {missing_keys}")

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

        if init_model is None:
            print("Not using variational inference (VI) output as initial values for the parameters.")
            inits_dict={
                "alphas" : [-1, 1],
                "mass_fracs" : np.full((self.fit_inputs["Nsrcs"]+1, self.fit_inputs["NAsrcs"]), 1 / self.fit_inputs["NAsrcs"]),
                "logE_true" : [np.median(np.log(self.fit_inputs["Edet"]))] * self.fit_inputs['N'],
                "flux_frac" : [0.1, 0.9],
                "log10_Ftot" : -2,
                "beta_egmf" : 0.5,
                "nu_lnAs": np.full(self.fit_inputs['N'], 0.5),
            }
        elif init_model == "pathfinder":
            print("Using PathFinder variational inference (VI) output as initial values for the parameters.")
            pathfinder = self.stan_model.pathfinder(
                data=self.fit_inputs
            )
            inits_dict = pathfinder.create_inits()[0]

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
            inits_dict["alpha_bg"] = 1
            inits_dict["mass_fracs_bg"] = np.full((self.fit_inputs["NAsrcs"]), 1 / self.fit_inputs["NAsrcs"])
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
                inits_dict = {key: value[()] for key, value in f["fit"]["inits"].items()}

        log_post = samples.pop("log_post")
        diagnostics["log_post"] = log_post

        use_rigidity_grid = "log10_gmf_Rgrid" in fit_inputs

        analysis = cls(
            data,
            gmf_model=gmf_model,
            use_rigidity_grid=use_rigidity_grid,
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
            if self.inits_dict is not None:
                inits_handle = fit_handle.create_group("inits")
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