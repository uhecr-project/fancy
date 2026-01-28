"""Container to manage the inputs and outputs of the fits."""
import os
import pickle
from typing import Union

import cmdstanpy
import h5py
import numpy as np
from cmdstanpy import CmdStanModel
from typing_extensions import Self  # change to typing for py>3.11
import arviz as az

from fancy.interfaces.data import Data
from fancy.interfaces.grid_generator import GridGenerator
from fancy.simulation import Simulation
from fancy.utils.helpers import pick_grain_size
from fancy.utils.package_data import (
    get_path_to_stan_file,
    get_path_to_stan_includes,
)


class Analysis:
    """Container to manage the inputs and outputs of the fits."""

    # pre-defined analysis types
    energy_type = "energy_only"
    mass_type = "mass_only"
    spatial_type = "spatial_only"
    energy_mass_type = "energy_mass"
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
        background_only : bool = False
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
        """
        self.data = data
        self.gmf_model = gmf_model
        self.analysis_type = analysis_type
        self.bg_only = background_only

        self.stan_model = None
        self.grid_config = None
        self.nthreads_per_chain = None
        self.fit_inputs = {key: None for key in self.fit_input_keys}
        self.fit = None

    def initialise_grid(
        self : Self,
        energy_gridparams: tuple = (32, 250, 50),
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
            "massids":[402, 1407, 2814]
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
        self.fit_inputs["omega_src"] = simulation.source_uvs
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
        self.fit_inputs["earth_spectrum_grid"] = simulation.spectrum_grid.T
        self.fit_inputs["lnA_logE_grid"] = np.log(simulation.lnA_energy_grid)
        self.fit_inputs["mean_lnA_grid"] = simulation.mean_lnA_grid.T
        self.fit_inputs["var_lnA_grid"] = simulation.var_lnA_grid.T
        self.fit_inputs["Nbeta_egmfs"] = len(simulation.beta_egmf_grid)
        self.fit_inputs["log10_beta_egmf_grid"] = np.log10(simulation.beta_egmf_grid.value)
        self.fit_inputs["log_wexp_earth_grid"] = np.moveaxis(simulation.log_wexp_earth_grid, (0,1,2,3), (0,2,3,1))
        self.fit_inputs["log_wexp_src_grid"] = np.moveaxis(simulation.log_wexp_src_grid, (0,1,2,3), (0,2,3,1))
        self.fit_inputs["esrc_ratio_grid"] = simulation.esrc_ratio_grid.T

        # for omega_det, deal with this depending on gmf model
        if simulation.gmf_model == "None":
            omega_det = simulation.truths['skycoord_earth_dets']
        else:
            omega_det = simulation.truths['skycoord_gb_truths_bp']
        omega_det.representation_type = "cartesian"
        self.fit_inputs["omega_det"] = omega_det.cartesian.xyz.value.T


        # warn if anything is None
        missing_keys = [k for k, v in self.fit_inputs.items() if v is None]
        if len(missing_keys) > 0:
            raise ValueError(f"Missing fit inputs: {missing_keys}")
        

    def compile_stan_model(self: Self, stan_threads : int = 4) -> None:
        """
        Compile the Stan model for the analysis.
        
        Parameters
        ----------
        stan_threads : int, default=4
            number of stan threads per chain to run
        """
        # get path to the stan file
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
            stan_file=str(path_to_stan_file), stanc_options=stanc_options, cpp_options=cpp_options
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

        # calculate the grain size
        self.fit_inputs["grain_size"] = pick_grain_size(
            self.fit_inputs["N"], self.nthreads_per_chain
        )
        print(f"Using grain size of {self.fit_inputs['grain_size']} for {self.fit_inputs['N']} events.")
        
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

        inits_dict={
            "alphas" : [-1, 1],
            "mass_fracs" : np.full((self.fit_inputs["Nsrcs"]+1, self.fit_inputs["NAsrcs"]), 1 / self.fit_inputs["NAsrcs"]),
            "logE_true" : [np.median(np.log(self.fit_inputs["Edet"]))] * self.fit_inputs['N'],
            "flux_frac" : [0.1, 0.9],
            "log10_Ftot" : -2,
            "beta_egmf" : 0.5,
            "nu_lnAs": np.full(self.fit_inputs['N'], 0.5),
            "mean_lnA_sys_unc" : 0.0,
            "var_lnA_sys_unc" : 0.0,
        }
        # different parameter names and configurations for background only fits
        if self.bg_only:
            inits_dict.pop("flux_frac")
            inits_dict.pop("log10_Ftot")
            inits_dict.pop("alphas")
            inits_dict.pop("mass_fracs")
            inits_dict["alpha_bg"] = 1
            inits_dict["mass_fracs_bg"] = np.full((self.fit_inputs["NAsrcs"]), 1 / self.fit_inputs["NAsrcs"])
        if inits is not None:
            print("Using user-provided initial values for the parameters.")
            inits_dict = inits
            
        
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

        # Diagnositics
        print("Checking all diagnostics...")
        print(self.fit.diagnose())

        self.chain = self.fit.stan_variables()
        print("Done!")
        return self.fit
    
    def run_diagnostics(self: Self) -> None:
        """
        Run a more sophisticated diagnostics on the fit output.

        Returns
        -------
        df : pd.DataFrame
            Dataframe containing the summary statistics for the parameters.
        flags : dict
            Dictionary containing flags for the diagnostics.
        summary_txt : str
            Summary text of the diagnostics.
        """
        if self.fit is None:
            raise ValueError("Run `fit_model` first!")
        
        raise NotImplementedError("Diagnostics not yet implemented for cmdstanpy backend.")

        # idata = az.from_cmdstanpy(
        #     posterior=self.fit,           # your cmdstanpy fit object
        #     observed_data=self.fit_inputs,  # your simulated data
        # )

        # var_base_names = [key for key in self.chain.keys() if key != "lp__"]
        # df, flags, summary_txt = run_single_diagnostics(idata, var_base_names=var_base_names)

        # print(summary_txt)
        # if any(flags[k] for k in ["any_bad_rhat", "any_bad_ess_bulk", 
        #                       "any_bad_ess_tail", "any_bad_ebfmi", "has_divergences"]):
        #     print("⚠️ Some diagnostics failed:")
        #     print(flags)

        # return df, flags, summary_txt
    
    def plot_diagnostics(self : Self, truths : dict = {}) -> None:
        pass

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
                fit_input_handle.create_dataset(key, data=value)

            # samples
            samples = fit_handle.create_group("samples")
            for key, value in self.chain.items():
                samples.create_dataset(key, data=value)

            # log posterior
            samples.create_dataset("log_post", data=self.fit.method_variables()["lp__"])