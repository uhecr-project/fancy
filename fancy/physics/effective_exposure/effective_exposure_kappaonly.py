"""Class that calculates effective exposure as a function of kappa only."""

import astropy.units as u
import h5py
import healpy
import numpy as np
from astropy.coordinates import SkyCoord
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing_extensions import Self, Tuple, Optional

from fancy.physics.effective_exposure.effective_exposure import EffectiveExposure
from fancy.utils.package_data import get_path_to_exposure_tables


class EffectiveExposureKappaOnly(EffectiveExposure):
    """
    Effective exposure tabulated as a function of the vMF concentration kappa only.

    Intended for spatial-only models, where there is no energy / rigidity / beta_egmf
    dependence. The table has shape (Nsrcs + 1, Nkappa), where the last row is the
    background (kappa = 0, i.e. isotropic, hence constant in kappa).

    The GMF treatment is set by `gmf_model`:
      - "None": no GMF, effective exposure = exposure . vMF(kappa).
      - anything else: the vMF map is lensed by the GMF at each rigidity of a grid
        and the resulting effective exposures are averaged over rigidity
        (weights set in `initialise_grids`).
    """

    def initialise_grids(
        self: Self,
        kappa_gridparams: Tuple[float, float, int] = (1e-1, 1e6, 40),
        R_gridparams: Tuple[float, float, int] = (1, 500, 25),
        R_weights: Optional[np.ndarray] = None,
        Npixels: int = 49152,
    ) -> None:
        """
        Initialise grids used for effective exposure calculation.

        Parameter:
        ----------
        kappa_gridparams : tuple, default=(1e-1, 1e6, 40)
            (min, max, N) of the logarithmically spaced kappa grid.
            The same grid is used for every source.
        R_gridparams : tuple, default=(1, 500, 25)
            (min, max, N) of the log-spaced rigidity grid in EV.
            Only used if a GMF model is set.
        R_weights : np.ndarray, optional
            weights over the rigidity grid used for the average (normalised internally).
            Default is uniform in log R. Only used if a GMF model is set.
        Npixels : int, default=49152
            the number of healpy pixels.
            DO NOT CHANGE UNLESS CRPROPA DOES SO!
        """
        self.kappa_grid = np.logspace(
            np.log10(kappa_gridparams[0]),
            np.log10(kappa_gridparams[1]),
            kappa_gridparams[2],
        )
        self.Nkappas = len(self.kappa_grid)

        # no beta_egmf dependence in this model
        self.beta_egmf_grid = None
        self.Nbeta_egmfs = None

        if self.gmf_model != "None":
            self.rigidity_grid = (
                np.logspace(
                    np.log10(R_gridparams[0]),
                    np.log10(R_gridparams[1]),
                    R_gridparams[2],
                )
                * u.EV
            )
            self.NRs = len(self.rigidity_grid)
            if R_weights is None:
                R_weights = np.ones(self.NRs)
            R_weights = np.asarray(R_weights, dtype=float)
            assert R_weights.shape == (self.NRs,), (
                f"R_weights must have shape ({self.NRs},), got {R_weights.shape}."
            )
            self.R_weights = R_weights / R_weights.sum()
        else:
            self.rigidity_grid = None
            self.NRs = None
            self.R_weights = None

        # initialise healpy grid for exposure calculation
        Nside = healpy.npix2nside(Npixels)
        self.delta_ang = (4 * np.pi) / Npixels  # uniform grid spacing
        if self.verbose:
            print(f"Nside: {Nside}, angular spacing: {self.delta_ang:.3e} sr")

        pix_arr = np.arange(0, Npixels, 1, dtype=int)
        uvs_healpy = np.array(healpy.pix2vec(Nside, pix_arr)).T
        self.coords_healpy = SkyCoord(
            uvs_healpy, frame="galactic", representation_type="cartesian"
        )

        # name-mangled in the parent class
        self.exposures = self._EffectiveExposure__compute_exposure()

    def compute_effective_exposure(
        self: Self,
        exposure_min: float = 1e-30,
    ) -> u.Quantity:
        """
        Compute the effective exposure as a function of kappa.

        Parameter:
        ----------
        exposure_min : float, default=1e-30
            minimum threshold value for exposure in km^2 yr

        Returns
        -------
        effective exposure with shape (Nsrcs + 1, Nkappa); last row is the background.
        """
        if self._EffectiveExposure__computed_effective_exposure:
            print(
                "Effective exposure has already been computed / loaded in. "
                "Returning existing results."
            )
            return self.effective_exposure

        eff_exp = np.zeros((self.Nsrcs + 1, self.Nkappas)) * u.km**2 * u.yr

        for isrc in tqdm(range(self.Nsrcs), desc="Iterating over sources: "):
            eff_exp[isrc, :] = self.compute_single_effective_exposure(
                (self.data.source.unit_vector[isrc], exposure_min)
            )

        # background is isotropic (kappa = 0) so it does not depend on the kappa grid
        eff_exp_bg = self._effective_exposure_at_kappa(
            np.array([0, 0, 1]), 0.0, exposure_min
        )
        eff_exp[self.Nsrcs, :] = eff_exp_bg

        self.effective_exposure = eff_exp
        self._EffectiveExposure__computed_effective_exposure = True

        return self.effective_exposure

    def compute_single_effective_exposure(self: Self, args: tuple) -> u.Quantity:
        """
        Compute the effective exposure over the kappa grid for a single source.

        Parameters
        ----------
        args : tuple
            (src_uv, exposure_min)

        Returns
        -------
        effective exposure over the kappa grid, shape (Nkappa,)
        """
        src_uv, exposure_min = args

        eff_exps = np.zeros(self.Nkappas) * u.km**2 * u.yr
        for ik in range(self.Nkappas):
            eff_exps[ik] = self._effective_exposure_at_kappa(
                src_uv, self.kappa_grid[ik], exposure_min
            )
        return eff_exps

    def _effective_exposure_at_kappa(
        self: Self, src_uv: np.ndarray, kappa: float, exposure_min: float
    ) -> u.Quantity:
        """Effective exposure for one source and kappa, R-averaged if GMF is on."""
        if self.gmf_model == "None":
            _, lensed_map = self.calculate_lensed_map(
                src_uv=src_uv, kappa_igmf=kappa, R=None
            )
            eff_exp = np.dot(self.exposures, lensed_map)
        else:
            eff_exp = 0.0 * u.km**2 * u.yr
            for ir in range(self.NRs):
                _, lensed_map = self.calculate_lensed_map(
                    src_uv=src_uv, kappa_igmf=kappa, R=self.rigidity_grid[ir]
                )
                eff_exp = eff_exp + self.R_weights[ir] * np.dot(
                    self.exposures, lensed_map
                )
        return max(eff_exp, exposure_min * u.km**2 * u.yr)

    def save(self: Self, outfile: str = "effective_exposures_kappaonly.h5") -> None:
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
            config_gr.create_dataset("source_uvs", data=self.data.source.unit_vector)
            config_gr.create_dataset("log10_kappa_grid", data=np.log10(self.kappa_grid))
            config_gr.create_dataset(
                "log10_effective_exposure",
                data=np.log10(self.effective_exposure.to_value(u.km**2 * u.yr)),
            )
            if self.rigidity_grid is not None:
                config_gr.create_dataset(
                    "rigidity_grid", data=self.rigidity_grid.to_value(u.EV)
                )
                config_gr.create_dataset("rigidity_weights", data=self.R_weights)

    def load_from_tables(
        self: Self, infile: str = "effective_exposures_kappaonly.h5"
    ) -> None:
        """
        Load tabulated results from h5py File.

        Parameter:
        ----------
        infile : str
            the path to the input file. must be in .h5 format.
        """
        assert infile.find(".h5") > 0, (
            f"Input file {infile} needs to have a .h5 extension."
        )
        with h5py.File(str(get_path_to_exposure_tables(infile)), "r") as f:
            config_label = f"{self.source_type}_{self.detector_type}_{self.gmf_model}"
            assert config_label in f.keys(), (
                f"Configuration {config_label} not found in {infile}."
            )
            config_gr = f[config_label]

            source_distances = config_gr["source_distances"][:]
            source_uvs = config_gr["source_uvs"][:]
            assert np.allclose(source_distances, self.data.source.distance), (
                "Source distances do not match."
            )
            assert np.allclose(source_uvs, self.data.source.unit_vector), (
                "Source unit vectors do not match."
            )

            self.kappa_grid = 10 ** config_gr["log10_kappa_grid"][:]
            self.Nkappas = len(self.kappa_grid)
            self.effective_exposure = (
                10 ** config_gr["log10_effective_exposure"][:] * (u.km**2 * u.yr)
            )

            if "rigidity_grid" in config_gr:
                self.rigidity_grid = config_gr["rigidity_grid"][:] * u.EV
                self.NRs = len(self.rigidity_grid)
                self.R_weights = config_gr["rigidity_weights"][:]

        self.beta_egmf_grid = None
        self.Nbeta_egmfs = None
        self._EffectiveExposure__computed_effective_exposure = True

    def plot_heatmap(self: Self, source: str = "all") -> plt.Figure:
        """
        Plot the effective exposure as a function of kappa (line plot, not a heatmap).

        Name kept for interface compatibility with the parent class.

        Parameter:
        ----------
        source : str, default="all"
            "all", "background", or the name of an individual source.
        """
        if not self._EffectiveExposure__computed_effective_exposure:
            raise ValueError("Effective exposure has not been computed yet.")

        if source == "all":
            indices = range(self.Nsrcs + 1)
        elif source == "background":
            indices = [self.Nsrcs]
        elif source in self.data.source.names:
            indices = [list(self.data.source.names).index(source)]
        else:
            raise ValueError(f"Source {source} not found in catalogue.")

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        for isrc in indices:
            label = (
                "Background" if isrc == self.Nsrcs else self.data.source.names[isrc]
            )
            ax.plot(
                self.kappa_grid,
                self.effective_exposure[isrc].to_value(u.km**2 * u.yr),
                label=label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$\epsilon_\mathrm{eff} \: / \: \mathrm{km}^2 \,\mathrm{yr}$")
        ax.legend()

        return fig
