"""Class that calculates effective exposure"""

import os
import numpy as np
import healpy
import h5py
import pickle as pickle
from scipy.interpolate import CubicSpline
from scipy.stats import norm

from astropy.coordinates import SkyCoord
import astropy.units as u

from fancy import Data
from tqdm import tqdm
import h5py

from fancy.detector.exposure import m_dec
from fancy.physics.gmf import GMFLensing


class EffectiveExposure:
    """
    Class to manage calculation of the effective exposure from given source(s).
    Also constructs tables that will be passed to stan for interpolation.
    """

    def __init__(self, data: Data, gmf_model: str = "JF12", verbose=False):
        """
        Class to manage calculation of the effective exposure from given source(s).
        Also constructs tables that will be passed to stan for interpolation.

        :param data: Data object from fancy
        :param verbose: enable verbosity
        """
        self.data = data
        self.gmf_model = gmf_model
        self.verbose = verbose

        # detector properties
        self.mass_group = data.detector.mass_group
        self.detector_type = data.detector.label

        self.source_type = data.source.label
        self.Dsrcs = data.source.distance * u.Mpc
        self.coords_src = data.source.coord  # this returns galactic coordinates
        src_names = data.source.name
        self.Nsrcs = len(self.Dsrcs)

        print(
            f"Configuration: {self.source_type}, {self.detector_type}, {self.mass_group}"
        )

        if verbose:
            print(f"Sources: {src_names}")
            print(f"Distances: {self.Dsrcs}")
            print(f"Coordinates: {self.coords_src}")

    def initialise_grids(
        self,
        energy_loss_table_file: str,
        Bigmf_min: float = 0.001,
        Bigmf_max: float = 10,
        NBigmfs: int = 50,
        Npixels: int = 49152,
    ):
        """
        Initialise grids used for effective exposure calculation.

        :param energy_loss_table_file: table for energy losses
        :param Bigmf_min, Bigmf_max, NBigmfs: min / max and density of log-spaced magnetic field grid in nG
        :param Npixels: number of pixels used for healpy grid. DO NOT CHANGE UNLESS CRPROPA DOES SO
        """
        # generate logarithmically spaced magnetic field grid
        self.Bigmf_grid = (
            np.logspace(np.log10(Bigmf_min), np.log10(Bigmf_max), NBigmfs) * u.nG
        )
        self.NBigmfs = NBigmfs

        # initialise healpy grid for exposure calculation
        Npixels = 49152  # set from CRPropa, pixelisation of order 6
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

        # create grid of rigidities to compute effective exposure
        self.rigidities_grid = (
            np.logspace(
                np.log10(self.data.detector.Rth),
                np.log10(self.data.detector.Rth_max),
                50,
            )
            * u.EV
        )
        self.NRs = len(self.rigidities_grid)

        # read out relevant parameters from energy loss tables
        if not os.path.exists(energy_loss_table_file):
            raise FileNotFoundError(
                f"Energy loss tables have not been generated. Construct them first!"
            )

        with h5py.File(energy_loss_table_file, "r") as f:
            # find the relevant group
            config_label = f"{self.detector_type}_mg{self.mass_group}"
            self.alpha_grid = f[config_label]["alpha_grid"][()]
            self.distances_grid = f[config_label]["distances_grid"][()] * u.Mpc

            self.Nalphas = len(self.alpha_grid)
            self.Ndistances = len(self.distances_grid)

            if self.mass_group != 1:
                log10_arrspect_grid = f[config_label]["log10_arrspect_grid"][()]
                log10_Rgrid = f[config_label]["log10_rigidities"][()]

                # get the arrival spectrum values for the values of the rigidity
                # grid defined using look-up tables
                # interpolation doesnt work for faraway sources due to sharp gradients
                Rs_idces = [
                    np.digitize(r.value, 10**log10_Rgrid, right=True)
                    for r in self.rigidities_grid
                ]

                self.arrspects_grid = np.zeros(
                    (self.Ndistances, self.NRs, self.Nalphas)
                ) * (1 / u.EV)
                for ir, r_idx in enumerate(Rs_idces):
                    self.arrspects_grid[:, ir, :] = (
                        10 ** log10_arrspect_grid[:, r_idx, :]
                    ) * (1 / u.EV)
            else:
                # parametrize expected energies as rigidities
                self.Eexs_grid = 10 ** f[config_label]["log10_Eexs_grid"][()] * u.EV
                self.Rth_srcs = f[config_label]["Rth_src_grid"][()] * u.EV

    def _compute_exposure(self):
        """Compute the exposure as a function of declination in healpy"""
        # first transform coordianates to declination
        self.coords_healpy.representation_type = "unitspherical"
        self.coords_healpy.transform_to("icrs")
        decs_healpy_grid = self.coords_healpy.icrs.dec.rad

        # compute exposure, which is function of declination only
        p = self.data.detector.params
        self.exposures = p[3] / p[4] * m_dec(decs_healpy_grid, p) * (u.km**2 * u.yr)

        # transform the coordinates back to galactic
        self.coords_healpy.transform_to("galactic")

    def compute_effective_exposure(
        self, gmflens: GMFLensing, kappa_max=1e6, exposure_min=1e-30 * u.km**2 * u.yr
    ):
        """
        Computation of the effective exposure.

        :param gmflens: GMFLensing object that will map the lens back to Earth
        :param disable_gmf: force disabling of GMF
        :param kappa_max: maximum threshold value for kappa computation
        :param exposure_min: minimum threshold value for exposure
        """
        # calculate effective exposure
        self.eff_exposure_grid = np.zeros((self.Nsrcs + 1, self.NRs, self.NBigmfs)) * (
            u.km**2 * u.yr
        )

        # first compute exposure
        self._compute_exposure()

        for id in range(self.Nsrcs + 1):
            # print if we are running source case or background case
            if id < self.Nsrcs:
                Dsrc = self.Dsrcs[id]
                src_uv = self.coords_src[id].cartesian.xyz.value
            else:
                src_uv = np.array([1, 0, 0])  # some random unit vector

            for ir in tqdm(
                range(self.NRs),
                desc="Computing effective exposure grid over rigidities: ",
                total=self.NRs,
            ):
                R = self.rigidities_grid[ir]

                # if source model, then iterate for each magnetic field and compute individual kappas
                if id < self.Nsrcs:
                    for ib, Bigmf in enumerate(self.Bigmf_grid):
                        kigmf = kappa_igmf(R, Bigmf, Dsrc, kappa_max=kappa_max)

                        self.coords_healpy.representation_type = "cartesian"
                        weighted_map = (
                            self.vMF(
                                self.coords_healpy.cartesian.xyz.value, src_uv, kigmf
                            )
                            * self.delta_ang
                        )
                        weighted_map /= np.sum(
                            weighted_map
                        )  # some numerical error in normalisation, so we force normalisation here

                        # lens the map
                        if self.gmf_model != "None":
                            lensed_map = gmflens.apply_lens_to_map(
                                weighted_map, R.to_value(u.EV)
                            )
                            # compute effective exposure
                            eff_exp = np.dot(self.exposures, lensed_map)
                        else:
                            eff_exp = np.dot(self.exposures, weighted_map)

                        # set some limit incase the effective exposure is so small
                        eff_exp = exposure_min if eff_exp < exposure_min else eff_exp
                        self.eff_exposure_grid[id, ir, ib] = eff_exp

                else:
                    self.coords_healpy.representation_type = "cartesian"
                    weighted_map = (
                        self.vMF(self.coords_healpy.cartesian.xyz.value, src_uv, 0.0)
                        * self.delta_ang
                    )
                    weighted_map /= np.sum(
                        weighted_map
                    )  # some numerical error in normalisation, so we force normalisation here

                    # lens the map
                    if self.gmf_model != "None":
                        lensed_map = gmflens.apply_lens_to_map(
                            weighted_map, R.to_value(u.EV)
                        )
                        # compute effective exposure
                        eff_exp = np.dot(self.exposures, lensed_map)
                    else:
                        eff_exp = np.dot(self.exposures, weighted_map)

                    # compute effective exposure
                    self.eff_exposure_grid[id, ir, :] = eff_exp

    def get_weighted_exposure(self):
        """
        Compute the weighted exposure
        """
        self.wexp_src_grid = np.zeros((self.Nsrcs, self.Nalphas, self.NBigmfs)) * (
            u.km**2 * u.yr
        )
        self.wexp_bg_grid = np.zeros(self.Nalphas) * (u.km**2 * u.yr)

        if self.mass_group != 1:
            # compute the detection threshold CCDF to take into account downscattering of events
            p_rdet = 1 - np.array(
                [
                    norm.cdf(
                        self.data.detector.Rth,
                        loc=R.value,
                        scale=self.data.detector.rigidity_uncertainty * R.value,
                    )
                    for R in self.rigidities_grid
                ]
            )

        for ia in range(self.Nalphas):
            alpha = self.alpha_grid[ia]

            # source case
            for id in range(self.Nsrcs):
                d_idx = np.digitize(
                    self.Dsrcs[id], self.distances_grid, right=True
                )  # get index from distance grid in prince calculation

                for ib in range(self.NBigmfs):
                    if self.mass_group != 1:
                        # integrate over all rigidities including detection effects
                        self.wexp_src_grid[id, ia, ib] = np.trapz(
                            y=self.arrspects_grid[d_idx, :, ia]
                            * self.eff_exposure_grid[id, :, ib]
                            * p_rdet,
                            x=self.rigidities_grid,
                        )
                    else:
                        # compute expected energy and find index in rigidity grid corresponding to it (we parametrize energy as rigidity for MG1)
                        Rex = self.Eexs_grid[d_idx, ia]
                        Rex_idx = np.digitize(
                            Rex.value, self.rigidities_grid.value, right=True
                        )

                        # weighting factor calculated by analytical integral of source & arrival distribution, see CM19 for details
                        # TODO: update this for bounded energy spectrum
                        w_factor = (
                            self.Rth_srcs[d_idx].value / self.data.detector.Rth
                        ) ** (1.0 - alpha)

                        self.wexp_src_grid[id, ia, ib] = (
                            self.eff_exposure_grid[id, Rex_idx, ib] * w_factor
                        )

            # background case
            if self.mass_group != 1:
                # integrate over background spectrum w/ detection effects, which is jsut a power law
                bg_spectrum = bounded_power_law(
                    self.rigidities_grid.value,
                    alpha,
                    self.data.detector.Rth,
                    self.data.detector.Rth_max,
                ) * (1 / u.EV)
                self.wexp_bg_grid[ia] = np.trapz(
                    y=bg_spectrum * self.eff_exposure_grid[-1, :, 0] * p_rdet,
                    x=self.rigidities_grid,
                    axis=0,
                )

            else:  # for MG1 we just use the default exposure since rigidity / energy doesnt play a role here
                self.wexp_bg_grid[ia] = (
                    self.data.detector.alpha_T / (4 * np.pi) * (u.km**2 * u.yr)
                )

    def save(self, outfile):
        """Save tabulated results to h5py File"""
        with h5py.File(outfile, "a") as f:
            config_label = f"{self.source_type}_{self.detector_type}_mg{self.mass_group}_{self.gmf_model}"
            if config_label in f.keys():
                del f[config_label]
            config_gr = f.create_group(config_label)

            config_gr.create_dataset("Dsrcs", data=self.Dsrcs)
            config_gr.create_dataset("alpha_grid", data=self.alpha_grid)
            config_gr.create_dataset("rigidities_grid", data=self.rigidities_grid)
            config_gr.create_dataset(
                "log10_Bigmf_grid", data=np.log10(self.Bigmf_grid.to_value(u.nG))
            )
            config_gr.create_dataset("distances_grid", data=self.distances_grid)
            config_gr.create_dataset("effective_exposure", data=self.eff_exposure_grid)
            config_gr.create_dataset(
                "log10_wexp_src_grid",
                data=np.log10(self.wexp_src_grid.to_value(u.km**2 * u.yr)),
            )
            config_gr.create_dataset(
                "log10_wexp_bg_grid",
                data=np.log10(self.wexp_bg_grid.to_value(u.km**2 * u.yr)),
            )

    def vMF(self, x, mu, kappa):
        """
        vMF distribution given mean direction mu, spread parameter kappa
        NB: shape of x must be (N, 3)
        """
        if kappa > 100:
            return np.exp(
                kappa * np.dot(x.T, mu) + np.log(kappa) - np.log(4 * np.pi / 2) - kappa
            )
        elif kappa < 1e-5:  # L'Hopital's rule
            return (
                (1 + kappa * np.dot(x.T, mu))
                / (4 * np.pi * np.cosh(kappa))
                * np.exp(kappa * np.dot(x.T, mu))
            )
        else:
            return (
                kappa / (4 * np.pi * np.sinh(kappa)) * np.exp(kappa * np.dot(x.T, mu))
            )


def theta_igmf(R, Bigmf, D, lc=1):
    """
    Deflection angle for IGMF in degrees

    :param R: rigidity in EV
    :param Bigmf: IGMF magnetic field strength in nG
    :param D: distance of the source in Mpc
    :param lc: coherence length in Mpc (default 1 Mpc)
    """
    return (
        2.3
        * (50 * u.EV / R)
        * (Bigmf / (1 * u.nG))
        * np.sqrt(D / (10 * u.Mpc))
        * np.sqrt(lc)
    ) * u.deg


def kappa_igmf(R, Bigmf, D, lc=1, kappa_max=1e6):
    """
    Deflection parameter for IGMF. Calculated using the approximate formula (for k >> 1)

    :param R: rigidity in EV
    :param Bigmf: IGMF magnetic field strength in nG
    :param D: distance of the source in Mpc
    :param lc: coherence length in Mpc (default 1 Mpc)
    :param kappa_max: some maximum threshold value for high kappa (== super small angles)
    """
    k = (7552 * (theta_igmf(R, Bigmf, D, lc) / (1 * u.deg)) ** -2).value
    return k if k < kappa_max else kappa_max


def bounded_power_law(R, alpha_b, Rmin, Rmax):
    """Background spectrum"""
    if alpha_b != 1.0:
        norm = (1.0 - alpha_b) / (Rmax ** (1.0 - alpha_b) - Rmin ** (1.0 - alpha_b))
    else:
        norm = 1.0 / (np.log(Rmax) - np.log(Rmin))

    return norm * R ** (-alpha_b)
