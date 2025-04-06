"""Class to manage simulations for UHECR propagation & detector effects"""

import datetime
import os
import pickle as pickle
import tempfile

import astropy.units as u
import h5py
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from scipy.stats import truncnorm
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from typing_extensions import ClassVar, List, Self, Tuple, Union
from vMF import sample_vMF

from fancy import Data
from fancy.detector.exposure import m_dec
from fancy.physics.gmf import GMFBackPropagation, GMFLensing
from fancy.utils.helpers import bounded_power_law, theta_igmfs
from fancy.utils.package_data import (
    get_path_to_energy_loss_tables,
    get_path_to_exposure_tables,
)


class Simulation:
    """Handles the generation of simulation samples."""

    __truth_input_keys : ClassVar[list] = ["f", "alpha_s", "alpha_b", "log10_L", "Bigmf", "Nex", "F0"]

    def __init__(
        self,
        data: Data,
        energy_loss_table_file: str = "energy_tables.h5",
        exposure_table_file: str = "exposure_tables.h5",
        gmf_model: str = "None",
        verbose : bool = False
    ):
        """
        Handle the generation of simulation samples.

        data: fancy.interfaces.Data
            Data object from fancy
        energy_loss_table_file: str
            file for energy tables
        exposure_table_file: str
            file where exposure tables are contained
        gmf_model : str, default="None"
            the GMF model to use when generating the simulations. 
            Default to None, i.e. we dont perform GMF lensing
        """
        self.detector_type = data.detector.label
        self.mass_model = data.detector.mass_model
        self.source_type = data.source.label
        self.gmf_model = gmf_model

        # source parameters
        self.Nsrcs = data.source.N

        self.data = data
        self.verbose = verbose

        # initialise the grids
        self._initialise_grids(energy_loss_table_file, exposure_table_file)

    def _initialise_grids(
        self, energy_loss_table_file: str = "energy_tables.h5", exposure_table_file: str = "exposure_tables.h5"
    ) -> None:
        """
        Initialise grids used for simulation.

        energy_loss_table_file : str
            file for energy tables
        exposure_table_file : str
            path to file where exposure tables are contained
        """
        with h5py.File(str(get_path_to_energy_loss_tables(energy_loss_table_file)), "r") as f:
            # find the relevant group
            config_label = f"{self.detector_type}_{self.mass_model}"
            self.alpha_grid = f[config_label]["alpha_grid"][()]
            self.Eearth_grid = 10**f[config_label]["log10_Eearth_grid"][()] * u.EeV
            self.dEearth_grid = f[config_label]["dEearth_grid"][()] * u.EeV
            log10_Eearth_spectrum = f[config_label]["log10_Eearth_spectrum"][()] * (1 / u.EeV)

            # also get the expected energies
            log10_Esrc = f[config_label]["log10_Esrc"][()] * u.EeV

            # filter out spectrum values that only match within the number of sources
            # in consideration
            dis_indices = np.digitize(
                self.data.source.distance, f[config_label]["distances_grid"][()], right=True
            )
            self.log10_Eearth_spectrum = np.take(log10_Eearth_spectrum, indices=dis_indices, axis=0)
            self.log10_Esrc = np.take(log10_Esrc, indices=dis_indices, axis=0)

        self.Nalphas = len(self.alpha_grid)
        self.NEearths = len(self.Eearth_grid)

        # now read the exposure table file
        with h5py.File(str(get_path_to_exposure_tables(exposure_table_file)), "r") as f:
            # find the relevant group
            config_label = f"{self.source_type}_{self.detector_type}_{self.mass_model}_{self.gmf_model}"
            self.log10_Bigmf_grid = f[config_label]["log10_Bigmf_grid"][()] * u.nG
            self.log10_integrated_source_exposure = (
                f[config_label]["log10_integrated_source_exposure"][()]
            )
            self.log10_integrated_background_exposure = (
                f[config_label]["log10_integrated_background_exposure"][()]
            )

        self.NBigmfs = len(self.log10_Bigmf_grid)

    def set_parameters_from_inputs(self, input_dict : dict) -> None:
        """
        Set parameters from fit inputs from posteriors.

        Parameter:
        ----------
        input_dict : dict
            dictionary of input parameters that contain all values
        truth_outfile : str
            output file for truths
        """
        # create dictionaryu of truths
        self.truth_dict = {
            "alpha_s": input_dict["alpha_s"],
            "alpha_b" : input_dict["alpha_b"],
            "f": input_dict["f"],
            "L": 10**input_dict["log10_L"],
            "log10_L": input_dict["log10_L"],
            "Bigmf": input_dict["Bigmf"],
            "F0": 10**input_dict["log10_F0"],
            "log10_F0": 10**input_dict["log10_F0"],
            "Nex": input_dict["Nex"],
            "Nsrc": input_dict["Nex"] * input_dict["f"],
            "Nbg": input_dict["Nex"] * (1 - input_dict["f"]),
        }

         # convert to integer using np.round (TODO: strictly should be Poisson, edit later)
        # store as object since we need to use this for sampling
        self.Nuhecrs_arr = np.zeros(self.Nsrcs + 1, dtype=int)  # type: ignore
        self.Nuhecrs_arr[: self.Nsrcs] = int(np.round(input_dict["Nex"] * input_dict["f"]))
        self.Nuhecrs_arr[self.Nsrcs] = int(np.round(input_dict["Nex"] * (1 - input_dict["f"])))

        self.Nuhecrs = np.sum(self.Nuhecrs_arr)

    def set_truths(self : Self, input_dict: dict) -> None:
        """
        Set simulation truths based on truths. 

        Parameter:
        ----------
        input_dict : dict
            dictionary of input parameters that contain all values
        """
        # first assert that the inputs keys match the ones the class definition
        # assert np.all(
        #     [k_input in self.truth_input_keys for k_input in input_dict.keys()]
        # ), "Truth inputs do not match."

        alpha_s = input_dict["alpha_s"]
        Bigmf = input_dict["Bigmf"] * u.nG
        Nex = input_dict["Nex"]
        f = input_dict["f"]
        alpha_b = input_dict["alpha_b"]

        # calculate expected events from background using source fraction
        Nex_src = Nex * f
        Nex_bg = Nex * (1 - f)

        # calculate the number of expected events from all sources using weighted exposure * total flux
        wexps_src = np.zeros(self.Nsrcs) * (u.km**2 * u.yr)
        Fs_per_Ls = np.zeros(self.Nsrcs) * (u.km**-2 * u.EeV**-1)

        for id, Dsrc in enumerate(self.Dsrcs):
            dmax_idx = np.digitize(Dsrc, self.distances_grid, right=True)

            # TODO: investigate whether doing interpolation is truly alright here
            f_log10_wexp_src = RegularGridInterpolator(
                (self.alpha_grid, self.log10_Bigmf_grid),
                self.log10_integrated_source_exposure[id, ...],
            )
            log10_wexp_src = f_log10_wexp_src((alpha_s, np.log10(Bigmf.value)))
            wexps_src[id] = 10.0**log10_wexp_src * (u.km**2 * u.yr)

            # TODO: same here
            f_log10_Esrc = CubicSpline(
                x=self.alpha_grid, y=self.log10_Esrc[dmax_idx, :]
            )
            Eex = 10.0 ** f_log10_Esrc(alpha_s) * u.EeV
            Fs_per_Ls[id] = 1 / (4 * np.pi * Dsrc.to(u.km) ** 2) / Eex

        L = Nex_src / (np.sum(wexps_src * Fs_per_Ls)) / self.Nsrcs

        # convert to integer using np.round (TODO: strictly should be Poisson, edit later)
        # store as object since we need to use this for sampling
        self.Nuhecrs_arr = np.zeros(self.Nsrcs + 1, dtype=int)  # type: ignore
        self.Nuhecrs_arr[: self.Nsrcs] = np.round(L * wexps_src * Fs_per_Ls * self.Nsrcs).astype(int)
        self.Nuhecrs_arr[self.Nsrcs] = np.round(Nex_bg).astype(int)
        self.Nuhecrs = np.sum(self.Nuhecrs_arr)

        # now calculate the flux
        Fs = np.sum(Fs_per_Ls) * L
        # for background flux, interpolate weighted exposure and calculate using Nex_bg
        # TODO: same here as well
        f_log10_wexp_bg = CubicSpline(
            x=self.alpha_grid, y=self.log10_integrated_background_exposure
        )  # NB: take any index for Bigmf since no dependence on it
        F0 = Nex_bg / (10.0 ** f_log10_wexp_bg(alpha_s) * (u.km**2 * u.yr))
        FT = Fs + F0  # total flux

        if self.verbose:
            print("Computed parameters from inputs: ")
            print(f"Luminosity per source: {L:.4e}")
            print(f"FT: {FT:.3e}, Fs: {Fs:.3e}, F0: {F0:.3e}")
            print(f"f = {f}")
            print(f"Nex: {Nex:.3f}, Nex_src: {Nex_src:.3f}, Nex_bg: {Nex_bg:.3f}")
            print(
                f"Nuhecrs: {np.sum(self.Nuhecrs_arr):d}, Nuhecrs_src: {np.sum(self.Nuhecrs_arr[:-1]):d}, Nuhecrs_bg: {self.Nuhecrs_arr[-1]:d}"
            )

        # create dictionaryu of truths
        self.truth_dict = {
            "alpha_s": alpha_s,
            "f": f,
            "L": L.value,
            "log10_L": np.log10(L.value),
            "Bigmf": Bigmf.value,
            "F0": F0.value,
            "log10_F0": np.log10(F0.value),
            "Fs": Fs.value,
            "FT": FT.value,
            "Nex": Nex,
            "Nsrc": Nex_src,
            "Nbg": Nex_bg,
            "alpha_b" : alpha_b
        }

    def sample_events(self : Self, sampling_factor: int = 10) -> Tuple[List, List, List]:
        """
        Sample energies, earth mass composition, and arrival directions at the Galactic boundary.

        - The energies are sampled via the arrival energy spectrum pre-computed via the 
        energy loss module. 
        - The arrival directions are sampled using the vMF model with deflections from the EGMF model.
        - The mass composition is sampled via the mass models used in the energy loss module.

        NB: we combine energy & mass to get rigidities since this is what we use for lensing.

        Parameters
        ----------
        sampling_factor: int, default=10
             factor to multiply with number of UHECRs to simulate for sampling

        Returns
        -------
        sampled_energies: list
            sampled energies at the Galactic boundary
        sampled_masses: list
            sampled masses at the Galactic boundary
        sampled_coords_gb: list
            sampled arrival directions at the Galactic boundary
        """
        Nsamples_arr = (
            np.full(self.Nsrcs + 1, sampling_factor, dtype=int) * self.Nuhecrs_arr
        )
        # sample using rng.choice
        rng = np.random.default_rng()

        """Sampling energies & masses at GB"""
        sampled_energies = []
        sampled_masses = []

        for id, Nsamples in enumerate(Nsamples_arr):

            if id < self.Nsrcs:
                # sample energies from the arrival spectrum
                # here we interpolate over all spectral indices and 
                # evaluate at the truth
                src_spectrum_grid = 10**CubicSpline(
                    y=self.log10_Eearth_spectrum[id, :, :],
                    x=self.alpha_grid,
                    axis=1,
                )(self.truth_dict["alpha_s"]) 

                # the probabilities we sample from are normalised by multiplying with
                # the energy widths
                probs_energy = (
                    src_spectrum_grid
                    * self.dEearth_grid.value
                    / np.sum(src_spectrum_grid * self.dEearth_grid.value)
                )

            else:
                 # precompute background spectrum & its probability
                bg_spectrum_grid = bounded_power_law(
                    self.Eearth_grid.value,
                    self.truth_dict["alpha_b"],
                    np.min(self.Eearth_grid.value),
                    np.max(self.Eearth_grid.value),
                )
                probs_energy = (
                    bg_spectrum_grid
                    * self.dEearth_grid.value
                    / np.sum(bg_spectrum_grid * self.dEearth_grid.value)
                )

            # then sample energies via rng.choice
            Ee_samples = rng.choice(self.Ees_grid, size=Nsamples, p=probs_energy)
            sampled_energies.append(Ee_samples)

            # now sample for lnA
            lnA_samples = np.array([self.data.detector.sample_lnAs(energy=Ee * u.EeV, Nsamples=1) for Ee in Ee_samples])
            sampled_masses.append(np.exp(lnA_samples))

        """Sampling for arrival directions"""
        sampled_coords_gb = []

        for id, Nsamples in enumerate(Nsamples_arr):

            if id < self.Nsrcs:
                # compute kappa_igmf
                # kappa_igmfs = 10 ** self.f_log10_kappa(
                #     theta_igmf_vec(
                #     sampled_rigidities[id] * u.EV,
                #     self.truth_dict["Bigmf"] * u.nG,
                #     self.Dsrcs[id],
                # ))

                kappa_igmfs = 7552 * (theta_igmfs(
                    (sampled_energies[id] / (0.5 * sampled_masses[id])) * u.EV,
                    self.truth_dict["Bigmf"] * u.nG,
                    self.Dsrcs[id],
                    ) / (1 * u.deg)).value**-2

                sampled_vectors_src = np.zeros((Nsamples, 3))
                for i in range(Nsamples):
                    sampled_vectors_src[i, :] = sample_vMF(
                        self.data.source.coord[id].cartesian.xyz.value,
                        kappa_igmfs[i],
                        num_samples=1,
                    )

                sampled_coords_gb.append(
                    SkyCoord(
                        sampled_vectors_src,
                        frame="galactic",
                        representation_type="cartesian",
                    )
                )

            else:
                sampled_vectors_bg = sample_vMF(
                    np.array([1, 0, 0]), 0.0, num_samples=Nsamples
                )  # uniform sampling
                sampled_coords_gb.append(
                    SkyCoord(
                        sampled_vectors_bg,
                        frame="galactic",
                        representation_type="cartesian",
                    )
                )

        return sampled_energies, sampled_masses, sampled_coords_gb

    def apply_lens(self : Self, sampled_rigidities : List[np.ndarray], sampled_coords : List[SkyCoord]) -> List[SkyCoord]:
        """
        Apply GMF lens by sampling & re-sampling of particles.

        Parameters
        ----------
        sampled_rigidites: list[np.ndarray]
            sampled rigidities at the Galactic boundary
        sampled_coords: list[astropy.coordinate.SkyCoord]
            sampled arrival diretions at the Galactic boundary
        
        Returns
        -------
        sampled_coords_earth: list[astropy.coordinate.SkyCoord]
            sampled arrival directions at the Earth after lensing.
        """
        if self.gmf_model == "None":
            print(
                "GMF is disabled. Will not run this code and return the initial samples"
            )
            return sampled_coords

        # initialise gmf lens object
        gmflens = GMFLensing(self.gmf_model)

        # apply lensing to the coordinates at GB for each rigidity
        sampled_coords_earth = []
        for id, sampled_coord in enumerate(sampled_coords):
            sampled_coords_earth.append(
                gmflens.apply_lens_with_particles(sampled_rigidities[id], sampled_coord)
            )

        return sampled_coords_earth

    def apply_detector_cuts(
        self : Self, sampled_energies : List[np.ndarray], sampled_masses : List[np.ndarray], sampled_coords : List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, SkyCoord]:
        """
        Apply detector cuts to sampled events at Earth.
        
        Parameters
        ----------
        sampled_energies : List[np.ndarray]
            energies after sampling (list of np arrays)
        sampled_masses : List[np.ndarray]
            masses after sampling (list of np arrays)
        sampled_coords: List[np.ndarray]
            coordinates after sampling (list of SkyCoord)
        """
        self.energies_det = np.zeros(self.Nuhecrs)
        self.lnAs_det = np.zeros(self.Nuhecrs)
        glons_det = np.zeros(self.Nuhecrs)
        glats_det = np.zeros(self.Nuhecrs)
        self.exposure_factors = np.zeros(self.Nuhecrs)

        rng_det = np.random.default_rng()
        uhecr_idx = 0

        for id, Nuhecrs_per_src in enumerate(self.Nuhecrs_arr):
            if id != len(self.Nuhecrs_arr) - 1:
                print(f"Current Source: {self.data.source.name[id]}")
            else:
                print("Background Case")

            count_per_src = 0
            iters = 0

            while count_per_src < Nuhecrs_per_src:
                # shuffle indices from samples
                sample_idces = np.arange(len(sampled_coords[id]))
                rng_det.shuffle(sample_idces)

                for i in sample_idces:

                    # angular reconstruction uncertainty
                    coord_earth = sampled_coords[id][i]

                    # sample reconstruction uncertainty using vMF
                    reconstr_uv = sample_vMF(
                        coord_earth.cartesian.xyz.value, self.data.detector.kappa_d, num_samples=1
                    )[0]
                    reconstr_uv /= np.linalg.norm(reconstr_uv)
                    reconst_coord = SkyCoord(
                        *reconstr_uv, representation_type="cartesian", frame="galactic"
                    )
                    reconst_coord.transform_to("icrs")

                    # evaluate exposure function at that declination -> construct pdet
                    m_omega = m_dec(
                        reconst_coord.icrs.dec.rad, self.data.detector.params
                    )
                    pdet = m_omega / self.data.detector.exposure_max
                    accept = rng_det.choice(
                        [0, 1], p=[pdet, 1 - pdet]
                    )  # use binomial distribution to sample that UHECR

                    # energy reconstruction
                    # done via truncated normal distribution 
                    en = sampled_energies[id][i]
                    reconst_en = self.data.detector.sample_energies(en, 1)

                    # mass reconstruction
                    # done via truncated normal distribution as well
                    mass = sampled_masses[id][i]
                    # reconst_mass = self.data.detector.sam(mass, 1)

                    # if particle is within exposure & within cuts then append
                    if (
                        (accept == 0)
                        and (reconst_en >= self.data.detector.Eth)
                        # and (reconst_rig <= self.data.detector.Rth_max)
                    ):
                        self.energies_det[uhecr_idx] = reconst_en
                        self.lnAs_det[uhecr_idx] = np.log(mass)
                        reconst_coord.transform_to("galactic")
                        reconst_coord.representation_type = "unitspherical"
                        glons_det[uhecr_idx] = reconst_coord.galactic.l.deg
                        glats_det[uhecr_idx] = reconst_coord.galactic.b.deg
                        self.exposure_factors[uhecr_idx] = (
                            m_omega * self.data.detector.alpha_T / self.data.detector.M
                        )

                        uhecr_idx += 1
                        count_per_src += 1

                    # break when we have the UHECRs contributing for this particular source
                    if count_per_src >= Nuhecrs_per_src:
                        print(count_per_src)
                        break

                    iters += 1

                if iters % 100 == 0:
                    print(f"Counts / src = {count_per_src}")

            if uhecr_idx > self.Nuhecrs:
                raise ValueError(
                    f"something wrong with indexing: {uhecr_idx}, {self.Nuhecrs}"
                )

        self.coords_det = SkyCoord(
            glons_det * u.deg, glats_det * u.deg, frame="galactic"
        )

        if np.any(self.energies_det == 0):
            raise ValueError(
                f"zero value detected in rigidity computation: {np.where(self.energies_det == 0)}"
            )
        
        return self.energies_det, self.lnAs_det, self.coords_det

    def backpropagate_events(self : Self, n_samples: int = 100, n_jobs : int=4):
        """
        Backpropagate the sampled & exposure-applied events at Earth back to the GB.

        Parameters
        ----------
        n_samples: int
            the number of samples for backpropagation simulation
        n_jobs: int
            the number of jobs to use for parallelisation
        """
        # first write data to temporary file such that Data can read it
        outfile = tempfile.mkstemp()[1]
        with h5py.File(outfile, "w") as f:
            data_gr = f.create_group(self.detector_type)
            data_gr.create_dataset(
                "energy", data=self.energies_det * self.data.detector.meanZ
            )
            data_gr.create_dataset("rigidity", data=self.energies_det / (0.5 * np.exp(self.lnAs_det)))
            data_gr.create_dataset("exposure", data=self.exposure_factors)
            data_gr.create_dataset("glon", data=self.coords_det.galactic.l.deg)
            data_gr.create_dataset("glat", data=self.coords_det.galactic.b.deg)
            data_gr.create_dataset("theta", data=np.full(self.Nuhecrs, 70))  # stub
            data_gr.create_dataset(
                "year",
                data=np.full(self.Nuhecrs, self.data.detector.start_year, dtype=int),
            )  # stub
            data_gr.create_dataset("day", data=np.ones(self.Nuhecrs, dtype=int))  # stub

        # add this to the data object
        self.data.add_uhecr(outfile, label=self.detector_type, hadr_model=self.mass_model, gmf_model="None")

        # now perform GMF back propagation
        gmfbackprop = GMFBackPropagation(self.data, self.gmf_model)
        gmfbackprop.run_backpropagation(n_samples, njobs=n_jobs)
        gmfbackprop.compute_kappa_gmf()

        # set properties
        if self.gmf_model != "None":
            self.kappa_gmfs = gmfbackprop.kappa_gmfs
            self.thetaPs = gmfbackprop.thetaPs
            self.glons_gb = gmfbackprop.uhecr_coords_gb.galactic.l.deg
            self.glats_gb = gmfbackprop.uhecr_coords_gb.galactic.b.deg

    def save(self : Self, outfile: str) -> None:
        """
        Save the simulation file as a new UHECR file.

        Parameters
        ----------
        outfile: str
            the output file as an UHECR file
        """
        # simulate for zenith angles for compatibility
        zeniths_sim, years_sim, days_sim = self._simulate_zenith_angles(
            self.coords_det.transform_to("icrs")
        )

        # get ra, dec, glon, glat
        glons_det, glats_det = (
            self.coords_det.galactic.l.deg,
            self.coords_det.galactic.b.deg,
        )
        c_icrs = self.coords_det.transform_to("icrs")
        ras_det, decs_det = c_icrs.ra.deg, c_icrs.dec.deg

        with h5py.File(outfile, "a") as file:
            if self.detector_type in list(file.keys()):
                del file[self.detector_type]
            simulated_data = file.create_group(f"{self.detector_type}")
            simulated_data.create_dataset("day", data=days_sim)
            simulated_data.create_dataset("year", data=years_sim)
            simulated_data.create_dataset("theta", data=zeniths_sim)
            simulated_data.create_dataset("rigidity", data=self.rigidities_det)
            simulated_data.create_dataset("energy", data=self.energies_det)
            simulated_data.create_dataset("mass", data=np.exp(self.lnAs_det))
            simulated_data.create_dataset("ra", data=ras_det)
            simulated_data.create_dataset("dec", data=decs_det)
            simulated_data.create_dataset("glat", data=glats_det)
            simulated_data.create_dataset("glon", data=glons_det)
            simulated_data.create_dataset("exposure", data=self.exposure_factors)

            if self.gmf_model != "None":
                gmfdefl_datas_grp = simulated_data.create_group("gmf")
                config_key = f"{self.gmf_model}_{self.mass_model}"
                if config_key in gmfdefl_datas_grp.keys():
                    del gmfdefl_datas_grp[config_key]
                gmfdefl_datas_config_grp = gmfdefl_datas_grp.create_group(config_key)
                gmfdefl_datas_config_grp.create_dataset(
                    "kappa_gmf", data=self.kappa_gmfs
                )
                gmfdefl_datas_config_grp.create_dataset("thetaP", data=self.thetaPs)
                gmfdefl_datas_config_grp.create_dataset("glons_gb", data=self.glons_gb)
                gmfdefl_datas_config_grp.create_dataset("glats_gb", data=self.glats_gb)

    # convert starting period to decimal year
    def _year_fraction(self, date):
        start = datetime.date(date.year, 1, 1).toordinal()
        year_length = datetime.date(date.year + 1, 1, 1).toordinal() - start
        return date.year + float(date.toordinal() - start) / year_length

    def _simulate_zenith_angles(self, c_icrs):
        """Simulate zenith angles, using ICRS SKyCoord"""
        years = []
        days = []
        times = []
        zenith_angles = []
        stuck = []

        k = 0
        first = True
        for d in c_icrs:
            za = 99
            i = 0
            while za > self.data.detector.threshold_zenith_angle.rad:
                dt = np.random.exponential(1.0 / self.Nuhecrs)
                if first:
                    t = self._year_fraction(self.data.detector.period_start) + dt
                else:
                    t = times[-1] + dt
                tdy = Time(t, format="decimalyear")
                c_altaz = d.transform_to(
                    AltAz(obstime=tdy, location=self.data.detector.location)
                )
                za = np.pi / 2 - c_altaz.alt.rad

                i += 1
                if i > 100:
                    za = self.data.detector.threshold_zenith_angle.rad
                    stuck.append(1)

            # convert decimal years to year & days
            year, year_frac = divmod(t, 1)
            day = np.round(year_frac * 365.2425)  # round to nearest even value
            years.append(int(year))
            days.append(int(day))

            # append time also for the algorithm
            times.append(t)
            first = False
            zenith_angles.append(za)
            k += 1
            # print(j , za)

        if len(stuck) > 1:
            print(
                "Warning: % of zenith angles stuck is",
                len(stuck) / len(zenith_angles) * 100,
            )

        return np.array(zenith_angles), np.array(years), np.array(days)
