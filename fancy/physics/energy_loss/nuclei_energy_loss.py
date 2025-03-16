"""Class that generates rigidity loss tables from the generated composition weights"""

import os
import numpy as np
import h5py
import pickle as pickle
from scipy.interpolate import RegularGridInterpolator
from scipy import stats
import astropy.units as u
from typing import Union
from typing_extensions import Self

from joblib import Parallel, delayed

from fancy import Data
from fancy.physics.energy_loss.energy_loss import EnergyLoss


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
        NEes: int = 25,
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
            self.source_mass_pdf = f["source_mass_pdf"][()]

        # truncate the rigidity grid such that it starts from the rigidity threshold
        # Rth_idx = np.digitize(self.data.detector.Rth * u.EV, rigidities_grid_file, right=True)
        Rth_idx = 0
        self.rigidity_grid = rigidities_grid_file[Rth_idx:]
        # truncate the propagation matrix and source mass PDF too
        self.propagation_matrix = self.propagation_matrix[:, :, :, Rth_idx:]
        self.source_mass_pdf = self.source_mass_pdf[:, :, :, Rth_idx:]
        self.NRs = len(self.rigidity_grid)

        if self.verbose:
            # initial size of wieghts
            print(
                f"Shape of propagation matrix from file: {self.propagation_matrix.shape}"
            )  # (Dsrc, Asrc, Aearths, R)
            print(
                f"Shape of source mass PDF from file: {self.source_mass_pdfs.shape}"
            )  # (Dsrc, Asrc, Aearths, R)
            # check if this makes sense
            print(f"Length of rigidity array: {len(self.rigidity_grid)}")
            print(f"Minimum rigidity in new grid: {np.min(self.rigidity_grid):.2f}")
            print(f"Maximum rigidity in new grid: {np.max(self.rigidity_grid):.2f}")

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
                self.source_mass_pdf[idis, ...],
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
        self.Esrcs = np.zeros((self.Ndistances, self.NAearths, self.Nalphas)) * u.EeV

        for dis_idx, _, earth_spect, Esrc in spect_results:
            Rearth_spectrum[dis_idx, ...] = earth_spect
            self.Esrcs[dis_idx, ...] = Esrc

        if return_src_spectrum:
            Rsrc_spectrum = np.zeros(
                (self.Ndistances, self.NAsrcs, self.NAearths, self.NRs, self.Nalphas)
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
                mass PDF at this distance
            propa_mat_per_d : np.ndarray
                propagation matrix at this distance
            R_cutoff : float
                rigidity cutoff value
        """
        dis_idx, source_mass_pdf_per_d, propa_mat_per_d, R_cutoff = spect_args
        print(f"Current distance index: {dis_idx}")

        src_spects = np.zeros((self.NAsrcs, self.NAearths, self.NRs, self.Nalphas)) * (
            1 / u.EV
        )

        Esrcs = np.zeros((self.NAearths, self.Nalphas)) * u.EeV

        for ime, ia in np.ndindex(self.NAearths, self.Nalphas):
            # temporary arrays that store the normaliseation per alpha per distance
            # alsostore the unnormalised source spectrum here
            src_norm = 0.0  # [EV^(1-alpha)]
            src_Enorm = 0.0  # [EeV^(2-alpha)]
            src_spect_unnormed = np.zeros((self.NAsrcs, self.NRs))  # [ e EeV^-alpha ]

            for ims in range(self.NAsrcs):
                src_spect_unnormed[ims, :] = (
                    source_mass_pdf_per_d[ims, ime, :]
                    * self.Zs[ims] ** (1.0 - self.alpha_grid[ia])
                    * self.rigidity_grid.to_value(u.EV) ** (-self.alpha_grid[ia])
                    * np.exp(-(self.rigidity_grid / R_cutoff).value)
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
            src_spects[:, ime, :, ia] = (src_spect_unnormed / src_norm) * (
                1 / u.EV
            )  # [1 / EV]

            # same here, but this time we get the first moment so
            # independent of rigidity & source mass
            Esrcs[ime, ia] = (src_Enorm / src_norm) * u.EeV

        # apply some absolute minimum
        Esrcs[Esrcs < 1e-10 * u.EeV] = 1e-10 * u.EeV

        # sum over source masses here
        # NB: new axis for alpha
        earth_spects = np.sum(propa_mat_per_d[..., np.newaxis] * src_spects, axis=0)

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
            (self.Ndistances, self.Nalphas, self.NEearths)
        ) * (1 / u.EeV)

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
                Aearth_idx = np.digitize(np.exp(lnA), self.As, right=True)

                # also need the binning factor from dR -> dEe
                dR_dEe = (1 / (0.5 * np.exp(lnA))) * (u.EV / u.EeV)

                # sum up the contributions
                self.Eearth_spectrum[:, :, iEe] += (
                    Rearth_spectrum[:, Aearth_idx, Rbin_idx, :] * dR_dEe
                )

            # then normalise over number of samples
            self.Eearth_spectrum[:, :, iEe] /= Nsamples

        return self.Eearth_spectrum

    def compute_earth_mass_pdf(self) -> None:
        """Compute mass PDF at earth for each rigidity."""
        earth_mass_pdf = np.zeros(
            (self.Ndistances, self.NAearths, self.NRs, self.Nalphas)
        )

        # multiply with the first moment
        for ime in range(self.NAearths):
            earth_mass_pdf[:, ime, :, :] = (
                self.As[ime] * self.Rearth_spectrum[:, ime, :, :]
            )

        # then normalise over all compositions
        earth_mass_pdf /= np.sum(earth_mass_pdf, axis=1, keepdims=True)

        return earth_mass_pdf

    def save(self: Self, outfile: str) -> None:
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
        with h5py.File(outfile, "a") as f:
            config_label = f"{self.detector_type}_{self.mass_model}"
            if config_label in f.keys():
                del f[config_label]
            config_gr = f.create_group(config_label)

            config_gr.create_dataset("alpha_grid", data=self.alpha_grid)
            config_gr.create_dataset("distances_grid", data=self.distances)
            # config_gr.create_dataset(
            #     "log10_rigidity_grid", data=np.log10(self.rigidity_grid.to_value(u.EV))
            # )  # in log10(EV)
            config_gr.create_dataset(
                "log10_Eearth_grid", data=np.log10(self.Eearth_grid.to_value(u.EeV))
            )  # in log10(EV)
            config_gr.create_dataset(
                "dEearth_grid", data=self.dEearth_grid.to_value(u.EeV)
            )  # in log10(EV)
            # config_gr.create_dataset("lnA_grid", data=self.lnA_grid)
            # config_gr.create_dataset(
            #     "log10_Rsrcs_spectrum",
            #     data=np.log10(self.Rsrc_spectrum.to_value(1 / u.EV)),
            # )
            # config_gr.create_dataset(
            #     "log10_Rearth_spectrum",
            #     data=np.log10(self.Rearth_spectrum.to_value(1 / u.EV)),
            # )
            config_gr.create_dataset(
                "log10_Esrcs", data=np.log10(self.Esrcs.to_value(u.EeV))
            )
            config_gr.create_dataset(
                "log10_Eearth_spectrum",
                data=np.log10(self.Eearth_spectrum.to_value(1 / u.EeV)),
            )

            # stored for plotting sake
            # config_gr.create_dataset("earth_mass_pdf", data=self.earth_mass_pdf)
            config_gr.create_dataset("As", data=self.As)
            config_gr.create_dataset("Zs", data=self.Zs)
