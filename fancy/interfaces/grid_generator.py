"""
Container to generate grids used for the stan analysis.

This can be used to generate the energy, mass, and effective exposure grids used both in
simulations and for the analysis (stan model).
"""

import numpy as np
import os
from typing_extensions import Self, Union

from fancy import Data
from fancy.physics import (
    EffectiveExposure,
    EnergyLossModel,
    WeightedExposure,
    LossLengthModel,
)
from fancy.utils.helpers import source_spectrum

charge_massid_map = {101: 1, 402: 2, 1407: 7, 2814: 14, 5626: 26}


class GridGenerator:
    """Container to generate the grids used for the analysis."""
    
    def __init__(self, data: Data, gmf_model: str = "None") -> None:
        """
        Initialise the GridGenerator class.

        Parameters
        ----------
        data : Data
            The Data object containing the sources and detectors.
        gmf_model : str, optional
            The Galactic magnetic field model to use. Default is "None".
        """
        self.data = data
        self.detector_type = data.detector.label
        self.mass_model = data.detector.mass_model
        self.source_type = data.source.label
        self.gmf_model = gmf_model

        # objects
        self.loss_length_model = LossLengthModel()
        self.energy_loss_model = None
        self.eff_exp_model = None

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
        self.wexp_src_grid = None
        self.eff_exp_grid = None
        self.wexp_earth_grid = None
        self.proton_esrc_grid = None
        self.energy_grid_widths = None
        self.charges_grid = None


        # shape parameters
        self.NEs = 0
        self.NElnAs = 0
        self.Nalphas = 0
        self.Nmass_fracs = 0
        self.Nbeta_egmfs = 0
        self.Nrigidities = 0

    def get_effective_exposure_grid(
        self: Self,
        effexp_model_kwargs: dict = {
            "beta_egmf_gridparams": (1e-3, 1, 10),
            "R_gridparams": (1, 500, 25),
        },
        n_jobs: int = 4,
    ) -> None:
        """
        Generate the effective exposure grid.

        Parameters
        ----------
        effexp_model_kwargs : dict, optional
            Keyword arguments for the EffectiveExposure model. Default is {
                "beta_egmf_gridparams" : (1e-3, 1, 10),
                "R_gridparams" : (1, 500, 25),
            }.
        """
        self.eff_exp_model = EffectiveExposure(data=self.data, gmf_model=self.gmf_model)
        self.eff_exp_model.initialise_grids(**effexp_model_kwargs)
        self.eff_exp_model.compute_effective_exposure(n_jobs=n_jobs)

        # store the effective exposure grid
        self.eff_exp_grid = (
            self.eff_exp_model.effective_exposure
        )  # in shape (Nsrcs+1, NRs, Nbeta_egmfs)
        self.beta_egmf_grid = self.eff_exp_model.beta_egmf_grid
        self.rigidity_grid = self.eff_exp_model.rigidity_grid
        self.Nbeta_egmfs = len(self.beta_egmf_grid)
        self.Nrigidities = len(self.rigidity_grid)

    def get_energy_mass_grid(
        self: Self,
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
        compute_source: bool = True,
    ) -> None:
        """
        Get the grid of energies and masses.

        Parameters
        ----------
        energy_gridparams : tuple, optional
            Parameters for the energy grid. Default is (32, 500, 50).
            This corresponds to (Emin, Emax, Nbins).
        lnA_energy_gridparams : tuple, optional
            Parameters for the lnA grid. Default is (3, 100, 50).
            This corresponds to (lnAmin, lnAmax, Nbins).
        src_inj_kwargs : dict, optional
            Keyword arguments for the source injection model. Default is {
                "dinits": [4],
                "Rmax": 1.7,
            }.
        bg_inj_kwargs : dict, optional
            Keyword arguments for the background injection model. Default is {
                "z_max": 3.0,
                "source_evo": "SFR",
                "Rmax": 1.7,
            }.
        energy_loss_model_kwargs : dict, optional
            Keyword arguments for the energy loss model. Default is {}.
        compute_source : bool, optional
            Whether to compute the source spectrum grid. Default is True.
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

        #  initalise the energy loss model
        self.energy_loss_model = EnergyLossModel(**energy_loss_model_kwargs)
        self.energy_loss_model.load_injection_solvers(
            src_inj_config=src_inj_kwargs, bg_inj_config=bg_inj_kwargs
        )

        # store the grids for later use
        spectra, lnAs = self.energy_loss_model.compute_spectrum_and_lnA(
            egrid=self.energy_grid,
            egrid_lnA=self.lnA_energy_grid,
            egrid_widths=self.energy_grid_widths,
        )

        # store the injection solver results
        # NB: shapes are in (Ngrid, Nalphas, Nmass_fracs, Nsrcs)
        self.spectrum_grid = spectra
        self.mean_lnA_grid = lnAs[0, ...]
        self.var_lnA_grid = lnAs[1, ...]
        self.alpha_grid = self.energy_loss_model.alphas
        self.mass_ids_grid = self.energy_loss_model.massids
        self.charges_grid = np.array(
            [charge_massid_map[massid] for massid in self.mass_ids_grid]
        )

        # store the shape parameters
        self.NEs = self.spectrum_grid.shape[0]
        self.Nalphas = self.spectrum_grid.shape[1]
        self.Nmass_fracs = self.spectrum_grid.shape[2]
        self.NElnAs = self.mean_lnA_grid.shape[0]

        if compute_source:
            self.get_source_spectrum_grid(
                energy_grid=self.energy_grid,
                alpha_grid=self.alpha_grid,
                charges_grid=self.charges_grid,
                Rmax=src_inj_kwargs["Rmax"],
            )

    def get_source_spectrum_grid(
        self: Self,
        energy_grid: np.ndarray = None,
        alpha_grid: np.ndarray = None,
        charges_grid: np.ndarray = None,
        Rmax: float = 1.7,
    ) -> None:
        """
        Get the source spectrum grid.

        It also computes the weighted exposure and the esrc_ratio_grid.

        Parameters
        ----------
        energy_grid : np.ndarray, optional
            The energy grid to use. If None, use the stored energy grid. Default is None.
        alpha_grid : np.ndarray, optional
            The alpha grid to use. If None, use the stored alpha grid. Default is None.
        charges_grid : np.ndarray, optional
            The charges grid to use. If None, use the stored charges grid. Default is None.
        Rmax : float, optional
            The maximum rigidity to use. Default is 1.7 EV.
        """
        if energy_grid is None:
            energy_grid = self.energy_grid
        if alpha_grid is None:
            alpha_grid = self.alpha_grid
        if charges_grid is None:
            charges_grid = self.charges_grid
        self.src_spectrum_grid = source_spectrum(
            energy_grid[:, np.newaxis, np.newaxis, np.newaxis],
            alpha_grid[np.newaxis, :, np.newaxis, np.newaxis],
            charges_grid[np.newaxis, np.newaxis, :, np.newaxis],
            Rmax=Rmax,
        )

        self.wexp_src_grid = np.trapz(
            y=self.src_spectrum_grid, x=self.energy_grid, axis=0
        )

        self.esrc_ratio_grid = np.trapz(
            y=self.energy_grid[:, None, None, None] * self.src_spectrum_grid,
            x=self.energy_grid,
            axis=0,
        ) / np.trapz(y=self.src_spectrum_grid, x=self.energy_grid, axis=0)

    def get_weighted_exposure(
        self: Self,
    ) -> None:
        """
        Get the weighted exposure grid.

        Parameters
        ----------
        detector_model_kwargs : dict, optional
            Keyword arguments for the WeightedExposure model. Default is {}.
        n_jobs : int, optional
            Number of parallel jobs to use. Default is 4.
        """
        if self.energy_grid is None:
            raise ValueError(
                "Energy grid is not defined. Please run get_energy_mass_grid() first."
            )
        if self.eff_exp_grid is None:
            raise ValueError(
                "Effective exposure grid is not defined. Please run get_effective_exposure_grid() first."
            )

        weighted_exposure = WeightedExposure(
            data=self.data, eff_exp=self.eff_exp_model, energy_loss_model=self.energy_loss_model
        )
        weighted_exposure.initialise_grids(
            energy_grid=self.energy_grid,
            lnA_energy_grid=self.lnA_energy_grid,
            energy_grid_widths=self.energy_grid_widths,
        )
        self.wexp_earth_grid = weighted_exposure.calculate_weighted_exposure()

    def get_loss_length_grid(
        self : Self,
        dinits: list = [4],
    ) -> None:
        """
        Get the loss length grid.

        Parameters
        ----------
        dinits : list, optional
            List of initial distances to compute the loss length for. Default is [4] Mpc.
        """
        loss_length_model = LossLengthModel()
        loss_length_model.load_loss_length_tables(dinits=dinits)
        self.proton_esrc_grid = loss_length_model.Esrc_grid

    def store_grids_to_dict(
        self: Self,
    ) -> dict:
        """
        Store the grids to a dictionary.

        Returns
        -------
        grids_dict : dict
            Dictionary containing the grids.
        """
        grids_dict = {
            "energy_grid": self.energy_grid,
            "lnA_energy_grid": self.lnA_energy_grid,
            "energy_grid_widths": self.energy_grid_widths,
            "alpha_grid": self.alpha_grid,
            "mass_ids_grid": self.mass_ids_grid,
            "charges_grid": self.charges_grid,
            "beta_egmf_grid": self.beta_egmf_grid,
            "rigidity_grid": self.rigidity_grid,
            "spectrum_grid": self.spectrum_grid,
            "mean_lnA_grid": self.mean_lnA_grid,
            "var_lnA_grid": self.var_lnA_grid,
            "src_spectrum_grid": self.src_spectrum_grid,
            "esrc_ratio_grid": self.esrc_ratio_grid,
            "wexp_src_grid": self.wexp_src_grid,
            "eff_exp_grid": self.eff_exp_grid,
            "wexp_earth_grid": self.wexp_earth_grid,
            "proton_esrc_grid": self.proton_esrc_grid,
            "NEs": self.NEs,
            "NElnAs": self.NElnAs,
            "Nalphas": self.Nalphas,
            "Nmass_fracs": self.Nmass_fracs,
            "Nbeta_egmfs": self.Nbeta_egmfs,
            "Nrigidities": self.Nrigidities,
        }
        return grids_dict
    
    def save(
        self: Self,
        outdir: str,
        filename: str = "grids.npz",
    ) -> None:
        """
        Save the grids to a .npz file.

        Parameters
        ----------
        outdir : str
            Directory to save the file to.
        filename : str, optional
            Filename to save the file as. Default is "grids.npz".
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        grids_dict = self.store_grids_to_dict()
        np.savez_compressed(os.path.join(outdir, filename), **grids_dict
    )
