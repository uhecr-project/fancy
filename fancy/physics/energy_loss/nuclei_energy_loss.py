"""Class that generates rigidity loss tables from the generated composition weights"""

import os
import numpy as np
import h5py
import pickle as pickle
from scipy.optimize import minimize, Bounds
from scipy.interpolate import UnivariateSpline
from scipy import stats
import astropy.units as u
from typing_extensions import Self, Tuple, Union

from joblib import Parallel, delayed

from fancy import Data
from fancy.physics.energy_loss.energy_loss import EnergyLoss

from fancy.utils.package_data import get_path_to_energy_loss_tables


class NucleiEnergyLoss(EnergyLoss):
    """Class to determine the arrival spectrum of nuclei at Earth using a rigidity conservation approximation."""

    def __init__(self, data: Data, verbose: bool = False) -> None:
        """
        Class to determine the arrival spectrum of nuclei at Earth using a rigidity conservation approximation.

        Parameters
        ----------
        data : Data
            the Data object as defined from fancy.Data. must contain detector information.
        verbose : bool, default=False
            flag to allow verbosity or not
        """
        super().__init__(data, verbose)

        # initialise other objects to be defined
        self.Esrcs = None  # mean energy at the soruce
        self.Eearth_spectrum = None  # the energy spectrum at earth

        # grid-related variables
        self.rigidity_grid = None  # rigidity grid read from the file
        self.NRs = None

    def initialise_grid(
        self,
        matrix_dir: str = "",
        alpha_min: float = -3,
        alpha_max: float = 10,
        Nalphas: int = 25,
        Eemin: Union[float, None] = None,
        Eemax: float = 1000,
        NEes: int = 50,
    ) -> None:
        """
        Initalise our grid using composition weights.

        Here we also add the composition information into the propagation
        matrices.

        Note that the energy range provided is directly used to interpolate
        the lnA information from Auger.

        Parameters
        ----------
        matrix_dir: str
            directory to composition weights
        alpha_min : float, default=-3
            the minimum source spectral index for the grid
        alpha_max : float, default=10
            the maximum source spectral index for the grid
        Nalphas : int, default=50
            number of elements for the source spectral index grid
        Eemin : float, default=1
            the minimum energy at Earth for the grid
        Eemax : float, default=10
            the maximum energy at Earth for the grid
        NEes : int, default=50
            number of elements for the energy at Earth grid
        """
        super().initialise_grid(
            matrix_dir, alpha_min, alpha_max, Nalphas, Eemin, Eemax, NEes
        )

        # now read in the source mass PDFs and the rigidity grid that we use
        with h5py.File(matrix_dir, "r") as f:
            rigidities_grid_file = (f["rigidities"][()] * u.GV).to(u.EV)
            self.propagation_matrix = f["propa_matrix"][()]

        # get the mean rigidity that directly maps with the energy grid
        self.rigidity_grid = np.zeros(len(self.Eearth_grid)) * u.EV
        for iE, Ee in enumerate(self.Eearth_grid):
            self.rigidity_grid[iE] = (
                Ee.value
                / (
                    0.5
                    * np.mean(
                        np.exp(self.data.detector.sample_lnAs(Ee.value, Nsamples=10000))
                    )
                )
            ) * u.EV

        self.NRs = len(self.rigidity_grid)

        # now we need to "interpolate" the propagation matrix
        self.propagation_matrix = self.propagation_matrix[
            :, :, :, np.digitize(self.rigidity_grid, rigidities_grid_file, right=True)
        ]

        # set initial values for the other variables
        self.Astars = None  # representative source masses
        self.source_mass_pdf = None  # source mass PDF

        # update number of As at earth 
        # subtract 1 since we dont consider protons
        self.NAearths -= 1

    def compute_repre_source_masses(
        self: Self,
        Astar_init: np.ndarray = np.array([8, 14, 20, 28, 50]),
        sigma_Astar: float = 3,
        excl_threshold: float = 2,
        n_jobs: int = 4,
        reset: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the representative source masses from the propagation matrix.

        Based on optimising the cost function that tries to minimise the contributions
        outside the observed composition and maximise the contributions within the
        observed composition.

        The representative source masses are then used to compute the source spectrum.

        Here, we assume that the source masses are distributed according to a Gaussian
        with the mean as the representative source mass, and a fixed standard deviation.

        Parameter
        --------
        sigma_Astar : float, default=2
            the standard deviation of the Gaussian distribution used to sample the source masses.
            This is used to sample the representative source masses.
        excl_threshold : float, default=2
            threshold value (in sigma) to exclude the source masses that are outside the observed composition.
        reset : bool, default=False
            whether to reset the computiation or not
        """
        if not reset:
            # if not reset, then read from the energy tables
            if os.path.exists(get_path_to_energy_loss_tables("energy_tables.h5")):
                with h5py.File(
                    str(get_path_to_energy_loss_tables("energy_tables.h5")), "r"
                ) as f:
                    config_label = f"{self.detector_type}_{self.mass_model}"
                    if config_label in f.keys():
                        self.Astars = f[config_label]["Astars"][()]
                        self.source_mass_pdfs = f[config_label]["source_mass_pdfs"][()]
                        return
                    else:
                        raise ValueError(
                            f"Configuration {config_label} not found in the file."
                        )
            else:
                raise ValueError(
                    "Energy tables not found. Please run the computation first."
                )

        # first prepare prefactors for the optimisation
        opt_prefactors = self.__prepare_opt_prefactors(excl_threshold=excl_threshold)

        # prepare fit inputs
        n_Astars = len(Astar_init)

        src_mass_args = [
            (
                dis_idx,
                Ee_idx,
                Astar_init,
                sigma_Astar,
                self.As,
                opt_prefactors[dis_idx, Ee_idx, :, :],
            )
            for dis_idx, Ee_idx in np.ndindex(self.Ndistances, self.NEearths)
        ]

        # now run the optimisation
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_single_optimisation)(arg) for arg in src_mass_args
        )

        self.Astars = np.zeros((self.Ndistances, self.NEearths, n_Astars))
        self.source_mass_pdfs = np.zeros(
            (self.Ndistances, self.NEearths, self.NAsrcs, n_Astars)
        )

        for dis_idx, Ee_idx, Astars in results:
            self.Astars[dis_idx, Ee_idx, :] = Astars

            for iastar, Astar in enumerate(Astars):
                # also compute the source mass PDF here
                a_src, b_src = (2 - Astar) / sigma_Astar, (56 - Astar) / sigma_Astar

                self.source_mass_pdfs[dis_idx, Ee_idx, :, iastar] = stats.truncnorm.pdf(
                    self.As, a=a_src, b=b_src, loc=Astar, scale=sigma_Astar
                )

        return self.Astars, self.source_mass_pdfs

    def __prepare_opt_prefactors(self: Self, excl_threshold: float = 2) -> np.ndarray:
        """
        Prepare the prefactors for optimisation.

        Parameters
        ----------
        excl_threshold : float, default=2
            threshold value (in sigma) to exclude the source masses that are outside the observed composition.

        Returns
        -------
        opt_prefactors : np.ndarray
            the prefactors for the optimisation.
            shape (Ndistances, NEearths, NAs, 2)
            where 2 is for lnA inclusion and exclusion
        """
        opt_rng = np.random.default_rng()

        n_samples = 1000  # number of samples to store for lnA inclusion / exclusion
        opt_prefactors = np.zeros((self.Ndistances, self.NEearths, self.NAsrcs, 2))
        for iEe, Ee in enumerate(self.Eearth_grid):
            lnA_total_samples = self.data.detector.sample_lnAs(
                energy=Ee.value,
                Nsamples=20000,
                lnA_min=1,
                lnA_max=np.log(self.As).max(),
            )

            mu_lnA, sigma_lnA = self.data.detector.get_mu_sigma_lnA(Ee.value)

            lower_bound = (
                mu_lnA - excl_threshold * sigma_lnA
            )
            upper_bound = (
                mu_lnA + excl_threshold * sigma_lnA
            )

            lnA_samples_incl = opt_rng.choice(
                a=lnA_total_samples[
                    (lnA_total_samples > lower_bound)
                    & (lnA_total_samples < upper_bound)
                ],
                size=n_samples,
            )
            lnA_samples_excl = opt_rng.choice(
                a=lnA_total_samples[
                    (lnA_total_samples < lower_bound)
                    | (lnA_total_samples > upper_bound)
                ],
                size=n_samples,
            )

            for imode, lnA_samples in enumerate([lnA_samples_incl, lnA_samples_excl]):
                # compute the PDF using a truncated normal distribution
                lnA_pdfs = self.data.detector.get_lnA_pdf(
                    lnA_samples,
                    energy=Ee.value,
                    lnA_min=1,
                    lnA_max=np.log(self.As).max(),
                )
                R_sample_idces = np.digitize(
                    (Ee.value / (0.5 * np.exp(lnA_samples))) * u.EV,
                    self.rigidity_grid,
                    right=True,
                ) - 1
                Ae_sample_idces = np.digitize(np.exp(lnA_samples), self.As, right=True)
                dR_dE = 1 / (0.5 * np.exp(lnA_samples))

                opt_prefactors[:, iEe, :, imode] = np.mean(
                    Ee
                    * self.propagation_matrix[:, :, Ae_sample_idces, R_sample_idces]
                    * lnA_pdfs[None, None, :]
                    * dR_dE[None, None, :]
                    * self.dEearth_grid[iEe].value,
                    axis=-1,
                )

        return opt_prefactors

    def _run_single_optimisation(
        self: Self, args: tuple
    ) -> Tuple[int, int, np.ndarray]:
        """
        Optimise for representative source weights for a single source.

        Parameters
        ----------
        As_reps : np.ndarray
            The representative source masses.
        args : tuple
        """
        print(f"Current distance index: {args[0]}, energy index: {args[1]}")
        dis_idx, Ee_idx, As_reps_init, sigma_As_reps, A_grid, prefactors = args

        As_rep_per_src = minimize(
            cost_function,
            x0=As_reps_init,
            args=(sigma_As_reps, A_grid, prefactors),
            bounds=Bounds(2, 56),
            method="L-BFGS-B",
        ).x
        return dis_idx, Ee_idx, As_rep_per_src

    def compute_rigidity_spectrum(
        self: Self,
        R_cutoff: float = 5 * u.EV,
        n_jobs: int = 4,
        return_src_spectrum: bool = False,
    ) -> None:
        """
        Compute the source and earth rigidity spectrum for all arrival masses.

        We parallelise over distances to speed up the computation.

        Parameters
        ----------
        R_cutoff : float, default=5 EV
            The rigidity cutoff used in the exponential component
            of the function. Defaults to 5 EV, which is what
            current Auger fits show.
        n_jobs : int, default=4
            the number of jobs to use for parallelisation over distances.
        return_src_spectrum : bool, default=False
            whether to return the source spectrum or not.
            Useful for plotting.
        """
        spect_args = [
            (
                idis,
                self.source_mass_pdfs[idis, ...],
                self.propagation_matrix[idis, ...],
                R_cutoff,
            )
            for idis in range(self.Ndistances)
        ]

        spect_results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_single_spectrum_calculation)(arg) for arg in spect_args
        )

        Rearth_spectrum = np.zeros(
            (self.Ndistances, self.NAearths, self.NRs, self.Nalphas)
        ) * (1 / u.EV)
        self.Esrcs = np.zeros((self.Ndistances, self.Nalphas)) * u.EeV

        for dis_idx, _, earth_spect, Esrc in spect_results:
            Rearth_spectrum[dis_idx, ...] = earth_spect
            self.Esrcs[dis_idx, ...] = Esrc

        if return_src_spectrum:
            Rsrc_spectrum = np.zeros(
                (self.Ndistances, self.NAsrcs, self.NRs, self.Nalphas)
            ) * (1 / u.EV)
            for dis_idx, src_spect, _, _ in spect_results:
                Rsrc_spectrum[dis_idx, ...] = src_spect

            return Rsrc_spectrum, Rearth_spectrum
        else:
            return Rearth_spectrum

    def _run_single_spectrum_calculation(self: Self, spect_args: tuple) -> np.ndarray:
        """
        Calculate the source and arrival rigidity spectrum for a single distance.

        Parameters
        ----------
        spect_args : tuple
            dis_idx : int
                index for distance
            source_mass_pdf_per_d : np.ndarray
                mass PDF per distance
            propa_mat_per_d : np.ndarray
                propagation matrix at this distance
            R_cutoff : float
                rigidity cutoff value
        """
        dis_idx, source_mass_pdf_per_d, propa_mat_per_d, R_cutoff = spect_args
        print(f"Current distance index: {dis_idx}")

        src_spects = np.zeros((self.NAsrcs, self.NRs, self.Nalphas)) * (1 / u.EV)

        Esrcs = np.zeros(self.Nalphas) * u.EeV

        for ia in range(self.Nalphas):
            # temporary arrays that store the normaliseation per alpha per distance
            # alsostore the unnormalised source spectrum here
            src_norm = 0.0  # [EV^(1-alpha)]
            src_Enorm = 0.0  # [EeV^(2-alpha)]
            src_spect_unnormed = np.zeros((self.NAsrcs, self.NRs))  # [ e EeV^-alpha ]

            for ims in range(self.NAsrcs):
                # summing over representative source masses (source mass PDF)
                src_spect_unnormed[ims, :] = np.sum(
                    source_mass_pdf_per_d[:, ims, :]
                    * self.rigidity_grid.to_value(u.EV)[:, None]
                    ** (-self.alpha_grid[ia])
                    * np.exp(-(self.rigidity_grid[:, None] / R_cutoff).value),
                    axis=-1,
                )

                # zeroths moment, in (EeV)^(1-alpha)
                src_norm += np.trapz(
                    y=src_spect_unnormed[ims, :], x=self.rigidity_grid.to_value(u.EV)
                )

                # first moment
                src_Enorm += np.trapz(
                    y=self.Zs[ims]
                    * self.rigidity_grid.to_value(u.EV)
                    * src_spect_unnormed[ims, :],
                    x=self.rigidity_grid.to_value(u.EV),
                )

            # normalise the source spectrum appropriately
            src_spects[:, :, ia] = (src_spect_unnormed / src_norm) * (
                1 / u.EV
            )  # [1 / EV]

            # same here, but this time we get the first moment so
            # independent of rigidity & source mass
            Esrcs[ia] = (src_Enorm / src_norm) * u.EeV

        # apply some absolute minimum
        Esrcs[Esrcs < 1e-10 * u.EeV] = 1e-10 * u.EeV

        # sum over source masses here
        # NB: new axis for alpha & for arrival mass
        # Shape is (NAsrcs, NAEarths, NRs, Nalphas)
        earth_spects = np.sum(
            propa_mat_per_d[:, 1:, :, np.newaxis] * src_spects[:, np.newaxis, :, :],
            axis=0,
        )

        return dis_idx, src_spects, earth_spects, Esrcs

    def compute_energy_spectrum(
        self: Self, Rearth_spectrum: np.ndarray, Nsamples: int = 1000
    ) -> None:
        """
        Compute the energy spectrum using lnA information.

        Parameters
        ----------
        Rearth_spectrum : np.ndarray
            the rigidity spectrum at earth.
            Must have shape (Ndistances, NAearths, NRs, Nalphas)
        Nsamples : int, default=1000
            the number of samples used for sampling lnA
        """
        self.Eearth_spectrum = np.zeros(
            (self.Ndistances, self.NEearths, self.Nalphas)
        ) * (1 / u.EeV)
        # start from 1 since we dont consider protons
        dR_dEe = 1 / (0.5 * self.As[1:]) * (u.EV / u.EeV)
        # we sum over all earth masses here
        # note all the new axis to get the right shape
        Eearth_spectrum = np.sum(Rearth_spectrum * dR_dEe[np.newaxis, :, np.newaxis, np.newaxis], axis=1)

        # perform smoothening
        for idis, ialpha in np.ndindex(self.Ndistances, self.Nalphas):
            f_smoothened_arr_dist = UnivariateSpline(self.Eearth_grid, np.log(Eearth_spectrum[idis, :, ialpha].value), s=0.01, k=1)
            self.Eearth_spectrum[idis, :, ialpha] = np.exp(
                f_smoothened_arr_dist(self.Eearth_grid)
            ) * (1 / u.EeV)

        return self.Eearth_spectrum

    def compute_earth_mass_pdf(self: Self, Rearth_spectrum: np.ndarray) -> None:
        """Compute mass PDF at earth for each rigidity."""
        earth_mass_pdf = np.zeros(
            (self.Ndistances, self.NAearths, self.NRs, self.Nalphas)
        )

        # multiply with the first moment
        # +1 since we dont consider protons
        for ime in range(self.NAearths):
            earth_mass_pdf[:, ime, :, :] = self.As[ime+1] * Rearth_spectrum[:, ime, :, :]

        # then normalise over all compositions
        earth_mass_pdf /= np.sum(earth_mass_pdf, axis=1, keepdims=True)

        return earth_mass_pdf

    def save(self: Self, outfile: str = "energy_tables.h5") -> None:
        """
        Save outputs to h5py file.

        In particular this function only saves the relevant information
        that is used for the fit, otherwise it will be too memory
        intensive.

        Parameters
        ----------
        outfile : str
            the filepath to the output file.
        """
        # maybe we can store this more efficiently, but I am also too lazy to
        # bother with this.
        with h5py.File(str(get_path_to_energy_loss_tables(outfile)), "a") as f:
            config_label = f"{self.detector_type}_{self.mass_model}"
            if config_label in f.keys():
                del f[config_label]
            config_gr = f.create_group(config_label)

            config_gr.create_dataset("alpha_grid", data=self.alpha_grid)
            config_gr.create_dataset("distances_grid", data=self.distances)
            config_gr.create_dataset(
                "log10_Eearth_grid", data=np.log10(self.Eearth_grid.to_value(u.EeV))
            )  # in log10(EV)
            config_gr.create_dataset(
                "log10_rigidity_grid", data=np.log10(self.rigidity_grid.to_value(u.EV))
            )  # in log10(EV)
            config_gr.create_dataset(
                "dEearth_grid", data=self.dEearth_grid.to_value(u.EeV)
            )  # in log10(EV)
            config_gr.create_dataset(
                "log10_Esrcs", data=np.log10(self.Esrcs.to_value(u.EeV))
            )
            config_gr.create_dataset(
                "log10_Eearth_spectrum",
                data=np.log10(self.Eearth_spectrum.to_value(1 / u.EeV)),
            )

            # stored for plotting sake
            config_gr.create_dataset("As", data=self.As)
            config_gr.create_dataset("Zs", data=self.Zs)

            # also save the representative source masses
            config_gr.create_dataset("Astars", data=self.Astars)
            config_gr.create_dataset("source_mass_pdfs", data=self.source_mass_pdfs)


def cost_function(
    As_reps: np.ndarray,
    sigma_As_rep: float,
    A_grid: np.ndarray,
    prefactors: np.ndarray,
) -> float:
    """
    Minimise for the representative source masses.

    This is done by maximizing the observed luminosity within the observed
    lnA values and minimizing those thatare outside of this, all as a function of the arrival energy.

    Parameters
    ----------
    As_reps : array-like
        The representative source masses.
    args : tuple
        sigma_As_rep : float
            the spread of the source masses around the representative values
    """
    # now define the luminosity at earth within threshold and outside threshold
    earth_lum_incl = np.zeros(len(As_reps))
    earth_lum_excl = np.zeros(len(As_reps))

    for iArep, As_rep in enumerate(As_reps):
        # normally distribute the source mass around representative values
        # with some spread to neighbouring masses
        a_src, b_src = (2 - As_rep) / sigma_As_rep, (56 - As_rep) / sigma_As_rep
        source_mass_pdf = stats.truncnorm.pdf(
            A_grid, a_src, b_src, loc=As_rep, scale=sigma_As_rep
        )

        earth_lum_incl[iArep] = np.sum(prefactors[:, 0] * source_mass_pdf)
        earth_lum_excl[iArep] = np.sum(prefactors[:, 1] * source_mass_pdf)

    # finally define the function to minimize
    return np.mean((earth_lum_excl) / earth_lum_incl)
