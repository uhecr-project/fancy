"""Class that calculates effective exposure."""

from typing import Union

import astropy.units as u
import h5py
import healpy
import numpy as np
from astropy.coordinates import SkyCoord
from joblib import Parallel, delayed
from typing_extensions import Self
from tqdm import tqdm

from fancy import Data
from fancy.detector.exposure import m_dec
from fancy.physics.gmf import GMFLensing
from fancy.utils.package_data import (
    get_path_to_energy_loss_tables,
    get_path_to_exposure_tables
)
from fancy.utils.helpers import (
    theta_igmf, vMF, bounded_power_law
)


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
        self.Bigmf_grid = None
        self.rigidity_grid = None
        self.alpha_grid = None
        self.NBigmfs = None
        self.NRs = None
        self.Nsrcs = len(self.data.source.distance)

        self.delta_ang = None
        self.coords_healpy = None

        # exposure parameters
        self.source_exposure = None
        self.background_exposure = None
        # integrated source / BG exposure
        self.int_source_exposure = None
        self.int_background_exposure = None

        # self.Dsrcs = data.source.distance * u.Mpc
        # self.coords_src = data.source.coord  # this returns galactic coordinates
        # src_names = data.source.name

        print(
            f"Configuration: {self.source_type}, {self.detector_type}, {self.mass_model}, {self.gmf_model}"
        )

        if verbose:
            print(f"Sources: {data.source.name}")
            print(f"Distances: {data.source.distance * u.Mpc}")
            print(f"Coordinates: {data.source.coord}")

    def initialise_grids(
        self: Self,
        energy_loss_table: str = "energy_tables.h5",
        Bigmf_min: float = 0.001,
        Bigmf_max: float = 1,
        NBigmfs: int = 50,
        R_min: Union[float, None] = None,
        R_max: float = 1e3,
        NRs: int = 50,
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
        # generate logarithmically spaced magnetic field grid
        self.Bigmf_grid = (
            np.logspace(np.log10(Bigmf_min), np.log10(Bigmf_max), NBigmfs) * u.nG
        )
        self.NBigmfs = NBigmfs

        # similarly generate a rigidity grid
        R_min = self.data.detector.Rth if R_min is None else R_min
        self.rigidity_grid = np.logspace(np.log10(R_min), np.log10(R_max), NRs) * u.EV
        self.NRs = NRs

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

        # now load the rest of the grid properties via the energy loss table
        self.__load_energy_loss_table(energy_loss_table)

        # finally compute the background energy spectrum from this
        self.Eearth_background_spectrum = np.zeros((self.Nalphas, self.NEearths)) * (
            1 / u.EeV
        )
        for ialpha, alpha in enumerate(self.alpha_grid):
            self.Eearth_background_spectrum[ialpha, :] = bounded_power_law(
                self.Eearth_grid,
                alpha,
                np.min(self.Eearth_grid),
                np.max(self.Eearth_grid),
            )

    def __compute_exposure(self: Self) -> np.ndarray:
        """Compute the exposure as a function of declination in healpy."""
        # first transform coordianates to declination
        self.coords_healpy.representation_type = "unitspherical"
        self.coords_healpy.transform_to("icrs")
        decs_healpy_grid = self.coords_healpy.icrs.dec.rad

        # compute exposure, which is function of declination only
        p = self.data.detector.params
        exposures = p[3] / p[4] * m_dec(decs_healpy_grid, p) * (u.km**2 * u.yr)

        # transform the coordinates back to galactic
        self.coords_healpy.transform_to("galactic")

        return exposures

    def __load_energy_loss_table(
        self: Self, energy_loss_table: str = "energy_tables.h5"
    ) -> None:
        """
        Load the energy loss table that is stored in h5 format.

        Parameters
        ----------
        energy_loss_table : str, default=energy_tables.h5
            the name of the file in which the energy tables are stored.
            Full path is taken from fancy.utils.get_path_to_energy_loss()
        """
        with h5py.File(
            str(get_path_to_energy_loss_tables(energy_loss_table)), "a"
        ) as f:
            config_label = f"{self.detector_type}_{self.mass_model}"

            self.alpha_grid = f[config_label]["alpha_grid"][()]
            self.Eearth_grid = 10 ** f[config_label]["log10_Eearth_grid"][()] * u.EeV
            Eearth_spectrum = 10 ** f[config_label]["log10_Eearth_spectrum"][()] * (
                1 / u.EeV
            )
            self.As = f[config_label]["As"][()]

            # filter out spectrum values that only match within the number of sources
            # in consideration
            dis_indices = np.digitize(
                self.data.source.distance, f[config_label]["distances_grid"][()]
            )
            self.Eearth_spectrum = np.take(Eearth_spectrum, indices=dis_indices, axis=0)

        self.NEearths = len(self.Eearth_grid)
        self.Nalphas = len(self.alpha_grid)

    def compute_source_exposure(
        self: Self,
        kappa_max: float = 1e6,
        exposure_min: float = 1e-30,
        Nsamples: int = 1000,
        n_jobs: int = 4,
    ) -> None:
        """
        Compute the effective exposure from the source.

        TODO: reformat this function such that we can easily access the intermediate functions
        for plotting purposes?

        Parameter:
        ----------
        kappa_max : float, default=1e6
            maximum threshold value for kappa computation
        exposure_min: float, default=1e-30
            minimum threshold value for exposure in km^2 yr
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
                self.Eearth_spectrum[dis_idx, ...],
                kappa_max,
                exposure_min,
                Nsamples,
            )
            for dis_idx in range(self.Nsrcs)
        ]

        # only run paralllelisation if more than one source
        src_exp_results = []
        if self.Nsrcs == 1:
            src_exp_results = [self.compute_single_source_exposure(exp_args[0])]
        else:
            # opting to not parallelise over distances due to pickling issue with GMF lensing.
            for iarg in tqdm(range(len(exp_args)), desc="Iterating over all distances: ", total=self.Nsrcs):
                src_exp_results.append(self.compute_single_source_exposure(exp_args[iarg]))

        self.source_exposure = (
            np.zeros((self.Nsrcs, self.NEearths, self.NBigmfs)) * u.km**2 * u.yr
        )
        self.int_source_exposure = (
            np.zeros((self.Nsrcs, self.Nalphas, self.NBigmfs)) * u.km**2 * u.yr
        )

        for dis_idx, src_exp, int_src_exp in src_exp_results:
            self.source_exposure[dis_idx, ...] = src_exp
            self.int_source_exposure[dis_idx, ...] = int_src_exp

    def compute_single_source_exposure(self: Self, args: tuple) -> np.ndarray:
        """
        Compute the exposure for a single source.

        Wrapper function for parallelising over distance.

        Paramters
        ---------
        args : tuple

        """
        dis_idx, dsrc, src_uv, earth_spect, kappa_max, exposure_min, Nsamples = args

        src_exposure = np.zeros((self.NEearths, self.NBigmfs)) * u.km**2 * u.yr
        int_src_exposure = np.zeros((self.Nalphas, self.NBigmfs)) * u.km**2 * u.yr

        for ib in range(self.NBigmfs):
            Bigmf = self.Bigmf_grid[ib]

            eff_exps_per_R = np.zeros(self.NRs) * u.km**2 * u.yr

            for ir in range(self.NRs):
                kigmf = (
                    7552
                    * (
                        theta_igmf(
                            self.rigidity_grid[ir],
                            Bigmf,
                            dsrc,
                        )
                        / (1 * u.deg)
                    ).value
                    ** -2
                )
                kigmf = min(kigmf, kappa_max)

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

            # now convert effective exposure to energy
            eff_exps_per_Ee = self._convert_eff_exp_to_energy(
                eff_exps_per_R, Nsamples=Nsamples
            )

            src_exposure[:, ib] = eff_exps_per_Ee

            # now integrate over each alphya
            # factoring into account the earth spectrum
            for ialpha in range(self.Nalphas):
                int_src_exposure[ialpha, ib] = np.trapz(
                    y=earth_spect[:, ialpha]
                    * eff_exps_per_Ee
                    * self.data.detector.get_p_Edet(self.Eearth_grid.value),
                    x=self.Eearth_grid,
                )

        return (dis_idx, src_exposure, int_src_exposure)

    def compute_background_exposure(self: Self, Nsamples: int = 1000) -> None:
        """
        Compute the effective exposure from the background.

        Parameter:
        ----------
        Nsamples : int, default=1000
            the number of samples used for sampling lnA
        """
        eff_exps_per_R = np.zeros(self.NRs) * u.km**2 * u.yr
        for ir in range(self.NRs):
            # map lensed map
            _, lensed_map = self.calculate_lensed_map(
                src_uv=np.array([0, 0, 1]), R=self.rigidity_grid[ir], kappa_igmf=0.0, 
            )

            # compute effective exposure
            eff_exps_per_R[ir] = np.dot(self.exposures, lensed_map)

        # now convert effective exposure to energy
        self.background_exposure = self._convert_eff_exp_to_energy(
            eff_exps_per_R, Nsamples=Nsamples
        )

        self.int_background_exposure = np.zeros(self.Nalphas) * u.km**2 * u.yr
        # now integrate over each alphya
        # factoring into account the earth spectrum
        for ialpha in range(self.Nalphas):
            self.int_background_exposure[ialpha] = np.trapz(
                y=self.Eearth_background_spectrum[ialpha, :]
                * self.background_exposure
                * self.data.detector.get_p_Edet(self.Eearth_grid.value),
                x=self.Eearth_grid,
            )

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

    def _convert_eff_exp_to_energy(
        self: Self, eff_exp_rigidity: np.ndarray, Nsamples: int = 1000
    ) -> np.ndarray:
        """
        Convert the rigidity-based effective exposure to function of energy.

        This is done by adding earth mass information by lnA sampling.

        Parameters
        ----------
        eff_exp_rigidity : np.ndarray
            the effective exposure in km^2 yr as a function of rigidity

        Returns
        -------
        same but as a function of energy after convolving with lnA information
        """
        eff_exp_energy = np.zeros(self.NEearths) * u.km**2 * u.yr

        for iEe, Eearth in enumerate(self.Eearth_grid):
            lnA_samples = self.data.detector.sample_lnAs(
                energy=Eearth.value,
                Nsamples=Nsamples,
                lnA_min=1,
                lnA_max=np.log(self.As).max(),
            )

            # now iterate over each sample
            # and histogram the contribution
            for lnA in lnA_samples:
                # compute the rigidity for each energy + sampled composition
                R = (Eearth.value / (0.5 * np.exp(lnA))) * u.EV
                # find the corresponding bin
                Rbin_idx = np.digitize(R, self.rigidity_grid, right=True)

                eff_exp_energy[iEe] += eff_exp_rigidity[Rbin_idx]

            eff_exp_energy[iEe] /= Nsamples

        return eff_exp_energy

    def save(self: Self, outfile: str):
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
            config_label = f"{self.source_type}_{self.detector_type}_{self.mass_model}_{self.gmf_model}"
            if config_label in f.keys():
                del f[config_label]
            config_gr = f.create_group(config_label)

            config_gr.create_dataset("source_distances", data=self.data.source.distance)
            config_gr.create_dataset("source_uvs", data=self.data.source.unit_vector)
            config_gr.create_dataset("alpha_grid", data=self.alpha_grid)
            config_gr.create_dataset(
                "log10_Eearth_grid", data=np.log10(self.Eearth_grid.value)
            )
            config_gr.create_dataset(
                "log10_Bigmf_grid", data=np.log10(self.Bigmf_grid.to_value(u.nG))
            )
            config_gr.create_dataset(
                "log10_source_exposure",
                data=np.log10(self.source_exposure.to_value(u.km**2 * u.yr)),
            )
            config_gr.create_dataset(
                "log10_background_exposure",
                data=np.log10(self.background_exposure.to_value(u.km**2 * u.yr)),
            )
            config_gr.create_dataset(
                "log10_integrated_source_exposure",
                data=np.log10(self.int_source_exposure.to_value(u.km**2 * u.yr)),
            )
            config_gr.create_dataset(
                "log10_integrated_background_exposure",
                data=np.log10(self.int_background_exposure.to_value(u.km**2 * u.yr)),
            )
