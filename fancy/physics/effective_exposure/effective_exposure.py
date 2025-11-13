"""Class that calculates effective exposure."""

from typing import Union

import astropy.units as u
import h5py
import healpy
import numpy as np
from astropy.coordinates import SkyCoord
from typing_extensions import Self, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from fancy import Data
from fancy.detector.exposure import m_dec
from fancy.physics.gmf import GMFLensing
from fancy.utils.package_data import (
    get_path_to_exposure_tables,
)
from fancy.utils.helpers import theta_igmf, vMF


class EffectiveExposure:
    """Class to manage calculation of the effective exposure from given source(s), and constructs tables that will be passed to stan for interpolation."""

    def __init__(
        self: Self, data: Data, gmf_model: str = "None", verbose: bool = False
    ) -> None:
        """
        Class to manage calculation of the effective exposure from given source(s).

        Also constructs tables that will be passed to stan for interpolation.

        data : fancy.interfaces.Data
            Container object that tracks the source, UHECR, and detector information
        gmf_model : str, default="None"
            the GMF model to consider. Default is None, which ignores GMF effects
        verbose : bool, default=False
            to print out additional statements for debugging or not.
        """
        self.data = data
        self.verbose = verbose

        # label properties
        self.source_type = data.source.label
        self.detector_type = data.detector.label
        self.mass_model = data.detector.mass_model
        self.gmf_model = gmf_model
        self.gmf_lens = GMFLensing(gmf_model=self.gmf_model)

        # parameters otherwised used here
        self.beta_egmf_grid = None
        self.rigidity_grid = None
        self.alpha_grid = None
        self.Nbeta_egmfs = None
        self.NRs = None
        self.Nsrcs = len(self.data.source.distance)

        self.delta_ang = None
        self.coords_healpy = None

        # exposure parameters
        self.effective_exposure = None

        if verbose:
            print(
                f"Configuration: {self.source_type}, {self.detector_type}, {self.mass_model}, {self.gmf_model}"
            )
            print(f"Sources: {data.source.name}")
            print(f"Distances: {data.source.distance * u.Mpc}")
            print(f"Coordinates: {data.source.coord}")

    def initialise_grids(
        self: Self,
        beta_egmf_gridparams: Tuple[float, float, int] = (1e-3, 1, 10),
        R_gridparams: Tuple[float, float, int] = (1, 500, 25),
        Npixels: int = 49152,
    ) -> None:
        """
        Initialise grids used for effective exposure calculation.

        Parameter:
        ----------
        Bigmf_min : float, default=0.001
            the minimum value of the IGMF strength used when generating the grid (in log space)
        Bigmf_max : float, default=1 nG
            same as Bigmf_min, but maximum value instead
        NBigmfs : int, default=50
            the number of points in the Bigmf grid
        Npixels : int, default=49152
            the number of pixels used to describe the healpy grid.
            Default is 49152, which is the default value used when generating the
            lens in CRPropa.
            DO NOT CHANGE UNLESS CRPROPA DOES SO!
        """
        # generate logarithmically spaced magnetic field spread grid
        # in nG * Mpc**1/2
        self.beta_egmf_grid = np.logspace(
            np.log10(beta_egmf_gridparams[0]),
            np.log10(beta_egmf_gridparams[1]),
            beta_egmf_gridparams[2],
        ) * (u.nG * u.Mpc**(1 / 2))
        self.Nbeta_egmfs = len(self.beta_egmf_grid)

        # similarly generate a rigidity grid in EV
        self.rigidity_grid = (
            np.logspace(
                np.log10(R_gridparams[0]), np.log10(R_gridparams[1]), R_gridparams[2]
            )
            * u.EV
        )
        self.NRs = len(self.rigidity_grid)

        # initialise healpy grid for exposure calculation
        Nside = healpy.npix2nside(Npixels)
        self.delta_ang = (4 * np.pi) / Npixels  # uniform grid spacing
        if self.verbose:
            print(f"Nside: {Nside}, angular spacing: {self.delta_ang:.3e} sr")

        # converting from pixels -> skycoord
        pix_arr = np.arange(0, Npixels, 1, dtype=int)
        uvs_healpy = np.array(healpy.pix2vec(Nside, pix_arr)).T
        self.coords_healpy = SkyCoord(
            uvs_healpy, frame="galactic", representation_type="cartesian"
        )

        # compute the exposure here
        self.exposures = self.__compute_exposure()

    def __compute_exposure(self: Self) -> np.ndarray:
        """Compute the exposure as a function of declination in healpy."""
        # first transform coordianates to declination
        self.coords_healpy.representation_type = "unitspherical"
        self.coords_healpy.transform_to("icrs")
        decs_healpy_grid = self.coords_healpy.icrs.dec.rad

        # compute exposure, which is function of declination only
        p = self.data.detector.params  # exposure parameters
        m_grid = m_dec(decs_healpy_grid, p)
        if self.data.detector.label == "all_sky":
            m_grid = np.ones_like(m_grid)

        exposures = p[3] / p[4] * m_grid * (u.km**2 * u.yr)

        # transform the coordinates back to galactic
        self.coords_healpy.transform_to("galactic")

        return exposures

    def compute_effective_exposure(
        self: Self,
        kappa_max: float = 1e6,
        exposure_min: float = 1e-30,
        dsrc_max : float = 1000,
        n_jobs: int = 4,
    ) -> None:
        """
        Compute the effective exposure.

        TODO: reformat this function such that we can easily access the intermediate functions
        for plotting purposes?

        Parameter:
        ----------
        kappa_max : float, default=1e6
            maximum threshold value for kappa computation
        exposure_min: float, default=1e-30
            minimum threshold value for exposure in km^2 yr
        dsrc_max : float, default=500
            maximum distance of the source in Mpc.
            Beyond this distance, we set the kappa_igmf to be 0.
        Nsamples : int, default=1000
            the number of samples used for sampling lnA
        n_jobs : int, default=4
            the number of jobs to parallelise over for each source.
            ignored if only one source.
        """
        # prepare arguments
        exp_args = [
            (
                dis_idx,
                self.data.source.distance[dis_idx],
                self.data.source.unit_vector[dis_idx],
                kappa_max,
                exposure_min,
                dsrc_max
            )
            for dis_idx in range(self.Nsrcs)
        ]

        # append the background model, which is effectively the same as the source model but with kappa == 0, i.e. can set distance = 1000 -> kappa \propto inv(theta(D=inf)) -> 0
        exp_args.append(
            (self.Nsrcs, 3000, np.array([0, 0, 1]), kappa_max, exposure_min, dsrc_max)
        )

        # only run paralllelisation if more than one source
        # opting to not parallelise over distances due to pickling issue with GMF lensing.
        eff_exp_results = [
            self.compute_single_effective_exposure(exp_args[iarg])
            for iarg in tqdm(
                range(len(exp_args)),
                desc="Iterating over all distances: ",
                total=self.Nsrcs + 1,
            )
        ]

        self.effective_exposure = (
            np.zeros((self.Nsrcs + 1, self.NRs, self.Nbeta_egmfs)) * u.km**2 * u.yr
        )

        for dis_idx, eff_exp in eff_exp_results:
            self.effective_exposure[dis_idx, ...] = eff_exp

        return self.effective_exposure

    def compute_single_effective_exposure(self: Self, args: tuple) -> np.ndarray:
        """
        Compute the exposure for a single source.

        Wrapper function for parallelising over distance.

        Paramters
        ---------
        args : tuple

        """
        dis_idx, dsrc, src_uv, kappa_max, exposure_min, dsrc_max = args

        eff_exps = np.zeros((self.NRs, self.Nbeta_egmfs)) * u.km**2 * u.yr

        for ib in range(self.Nbeta_egmfs):
            beta_egmf = self.beta_egmf_grid[ib]

            eff_exps_per_R = np.zeros(self.NRs) * u.km**2 * u.yr

            for ir in range(self.NRs):
                kigmf = (
                    7552
                    * (
                        theta_igmf(
                            self.rigidity_grid[ir],
                            beta_egmf,
                            dsrc * u.Mpc,
                        )
                        / (1 * u.deg)
                    ).value
                    ** -2
                )
                kigmf = min(kigmf, kappa_max)
                # if the source is too far away, we set kappa_igmf to 0
                if dsrc > dsrc_max:
                    kigmf = 0.0

                # map lensed map
                _, lensed_map = self.calculate_lensed_map(
                    src_uv=src_uv,
                    R=self.rigidity_grid[ir],
                    kappa_igmf=kigmf,
                )

                # compute effective exposure
                eff_exps_per_R[ir] = max(
                    np.dot(self.exposures, lensed_map),
                    exposure_min * u.km**2 * u.yr,
                )

            eff_exps[:, ib] = eff_exps_per_R

        return (dis_idx, eff_exps)

    def calculate_lensed_map(
        self: Self, src_uv: np.ndarray, kappa_igmf: float, R: float
    ) -> np.ndarray:
        """
        Calculate the lensed map per source & rigidity.

        Parameters
        ----------
        src_uv : np.ndarray
            unit vector for source coordinate
        kappa_igmf : float
            the deflection parameter for vMF
        R : float
            the rigidity in EV

        Returns
        -------
        the map weighted with a vMF and the map lensed via the GMF (or not if gmf_model = None)
        """
        self.coords_healpy.representation_type = "cartesian"
        weighted_map = (
            vMF(self.coords_healpy.cartesian.xyz.value, src_uv, kappa_igmf)
            * self.delta_ang
        )
        weighted_map /= np.sum(
            weighted_map
        )  # some numerical error in normalisation, so we force normalisation here

        # lens the map only if we want to include GMF
        if self.gmf_model != "None":
            lensed_map = self.gmf_lens.apply_lens_to_map(weighted_map, R.to_value(u.EV))
        else:
            lensed_map = weighted_map

        return weighted_map, lensed_map

    def save(self: Self, outfile: str = "effective_exposure_tables.h5") -> None:
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
            config_label = f"{self.source_type}_{self.gmf_model}"
            if config_label in f.keys():
                del f[config_label]
            config_gr = f.create_group(config_label)

            config_gr.create_dataset("source_distances", data=self.data.source.distance)
            config_gr.create_dataset("source_uvs", data=self.data.source.unit_vector)
            config_gr.create_dataset(
                "log10_beta_egmf_grid",
                data=np.log10(self.beta_egmf_grid.to_value(u.nG * u.Mpc**1 / 2)),
            )
            config_gr.create_dataset(
                "log10_effective_exposure",
                data=np.log10(self.effective_exposure.to_value(u.km**2 * u.yr)),
            )

    def plot_heatmap(self : Self, source : str = "all") -> plt.Figure:
        """
        Plot heatmap of the effective exposure as a function of rigidity and beta_egmf.

        Parameter:
        ----------
        source : str, default="all"
            which source to plot. Default is "all", which plots all sources.
            Other options include:
            - "background": plots the background model
            - name of the individual source in the catalogue
        """
        if self.effective_exposure is None:
            raise ValueError("Effective exposure has not been computed yet.")
        if source == "all":
            nsources = self.Nsrcs + 1
            fig, axes = plt.subplots(
                nrows=int(np.ceil((nsources) / 2)),
                ncols=2,
                figsize=(12, 4 * int(np.ceil((nsources) / 2))),
                constrained_layout=True,
            )
            axes = axes.flatten()
            for isrc in range(nsources):
                ax = axes[isrc]
                im = ax.imshow(
                    np.log10(self.effective_exposure[isrc].to_value(u.km**2 * u.yr)),
                    aspect="auto",
                    origin="lower",
                    extent=[
                        np.log10(self.beta_egmf_grid.to_value(u.nG * u.Mpc**(1 / 2))).min(),
                        np.log10(self.beta_egmf_grid.to_value(u.nG * u.Mpc**(1 / 2))).max(),
                        np.log10(self.rigidity_grid.to_value(u.EV)).min(),
                        np.log10(self.rigidity_grid.to_value(u.EV)).max(),
                    ],
                    cmap="viridis",
                    vmin=np.log10(self.effective_exposure.to_value(u.km**2 * u.yr).min()),
                    vmax=np.log10(self.effective_exposure.to_value(u.km**2 * u.yr).max()),
                )
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(r"$\log_{10}(\epsilon_\mathrm{eff} \: / \: \mathrm{km}^2 \,\mathrm{yr})$")
                if isrc < self.Nsrcs:
                    ax.set_title(f"Source {isrc}: {self.data.source.names[isrc]}")
                else:
                    ax.set_title("Background")
                ax.set_xlabel(r"$\log_{10}(\beta_{\mathrm{EGMF}} / \mathrm{nG\,Mpc}^{1/2})$")
                ax.set_ylabel(r"$\log_{10}(R / \mathrm{EV})$")
            
            return fig

        else:
            if source == "background":
                src_idx = self.Nsrcs
                title = "Background"
            elif source in self.data.source.names:
                src_idx = self.data.source.names.index(source)
                title = f"Source: {source}"
            else:
                raise ValueError(f"Source {source} not found in catalogue.")
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            im = ax.imshow(
                np.log10(self.effective_exposure[src_idx].to_value(u.km**2 * u.yr)),
                aspect="auto",
                origin="lower",
                extent=[
                    np.log10(self.beta_egmf_grid.to_value(u.nG * u.Mpc**(1 / 2))).min(),
                    np.log10(self.beta_egmf_grid.to_value(u.nG * u.Mpc**(1 / 2))).max(),
                    np.log10(self.rigidity_grid.to_value(u.EV)).min(),
                    np.log10(self.rigidity_grid.to_value(u.EV)).max(),
                ],
                cmap="viridis",
                vmin=np.log10(self.effective_exposure.to_value(u.km**2 * u.yr).min()),
                vmax=np.log10(self.effective_exposure.to_value(u.km**2 * u.yr).max()),
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"$\log_{10}(\epsilon_\mathrm{eff} \: / \: \mathrm{km}^2 \,\mathrm{yr})$")
            ax.set_title(title)
            ax.set_xlabel(r"$\log_{10}(\beta_{\mathrm{EGMF}} / \mathrm{nG\,Mpc}^{1/2})$")
            ax.set_ylabel(r"$\log_{10}(R / \mathrm{EV})$")
            
            return fig


