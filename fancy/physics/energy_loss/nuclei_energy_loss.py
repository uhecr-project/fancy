"""Class that generates rigidity loss tables from the generated composition weights"""

import os
import numpy as np
import h5py
import pickle as pickle
from scipy.interpolate import CubicSpline
from scipy import stats
import astropy.units as u

from fancy import Data
from fancy.physics.energy_loss.energy_loss import EnergyLoss


class NucleiEnergyLoss(EnergyLoss):
    """Class to determine the arrival spectrum of nuclei at Earth using a rigidity conservation approximation."""

    def __init__(self, data: Data, verbose: bool = False) -> None:
        """
        Class to determine the arrival spectrum of nuclei at Earth using a rigidity conservation approximation.

        Parameter
        ---------
        data : Data
            the Data object as defined from fancy.Data. must contain detector information.
        verbose : bool, default=False
            flag to allow verbosity or not
        """
        super().__init__(data, verbose)

    def initialise_grid(
        self,
        matrix_dir: str = "",
        alpha_min: float = -3,
        alpha_max: float = 10,
        Nalphas: int = 25,
        Eemin: float = 1,
        Eemax: float = 300,
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

        with h5py.File(matrix_dir, "r") as f:
            rigidities_grid_file = (f["rigidities"][()] * u.GV).to(u.EV)
            propagation_matrix_file = f["propa_matrix"][()]

        # perform bin-based "integration" for each lnA value in 
        # the energy range that we provide.
        self.propagation_matrix = np.zeros((self.Ndistances, self.NAsrcs, self.NRs))
        for iae, lnA in enumerate(self.lnA_grid):
            # find the corresponding bin
            Rbin_idx = np.digitize(
                self.rigidities_grid[iae], rigidities_grid_file, right=True
            )
            AE_index = np.digitize(np.exp(lnA), self.As, right=True)

            # sum up the contributions
            self.propagation_matrix[:, :, iae] += propagation_matrix_file[
                :, :, AE_index, Rbin_idx
            ]

        if self.verbose:
            # initial size of wieghts
            print(
                f"Shape of propagation matrix from file: {self.propagation_matrix.shape}"
            )  # (Dsrc, Asrc, Aearths, R)
            # check if this makes sense
            print(
                f"Length of rigidity array for MG{self.mass_group}: {len(self.rigidities_grid)}"
            )
            print(f"Minimum rigidity in new grid: {np.min(self.rigidities_grid):.2f}")
            print(f"Maximum rigidity in new grid: {np.max(self.rigidities_grid):.2f}")

    def _compute_source_PDF(self) -> None:
        """Compute source composition PDF for injection efficiency."""
        self.Asrc_pdfs = self.propagation_matrix / np.sum(
            self.propagation_matrix, axis=1, keepdims=True
        )

        # remove unnecessary NaNs in PDF where
        self.Asrc_pdfs[np.isnan(self.Asrc_pdfs)] = 0.0

    def compute_source_spectrum(self, R_cutoff=20 * u.EV) -> None:
        """Compute the source spectrum."""
        self.src_spects_full = np.zeros(
            (self.Ndistances, self.NAsrcs, self.NRs, self.Nalphas)
        ) * (1 / u.EV)  # since its for all source compositions

        self._compute_source_PDF()  # compute the source composition PDF

        # finally get the source spectrrum, normalised over constatn marginalisaed over all source compositions
        for id, ia in np.ndindex(self.Ndistances, self.Nalphas):
            # temporary arrays that store the normaliseation per alpha per distance
            # alsostore the unnormalised source spectrum here
            src_norm = 0.0  # [EV^(1-alpha)]
            src_spect_unnormed = np.zeros((self.NAsrcs, self.NRs))  # [ e EeV^-alpha ]

            for im in range(self.NAsrcs):
                src_spect_unnormed[im, :] = (
                    self.Asrc_pdfs[id, im, :]
                    * self.Zs[im] ** (1.0 - self.alpha_grid[ia])
                    * self.rigidities_grid.to_value(u.EV) ** (-self.alpha_grid[ia])
                    * np.exp(-(self.rigidities_grid / R_cutoff).value)
                )

                # zeroths moment, in (EeV)^(1-alpha)
                src_norm += np.trapz(
                    y=src_spect_unnormed[im, :], x=self.rigidities_grid.to_value(u.EV)
                )

            # normalisation is carried by contribution over all masses at the source
            self.src_spects_full[id, ..., ia] = (src_spect_unnormed / src_norm) * (
                1 / u.EV
            )  # [1 / EV]

        # summed over all soruce compostiions that contribute in mass group
        self.src_spects = np.sum(
            self.src_spects_full, axis=1
        )  # distances x rigidities x alphas

    def compute_arrival_spectrum(self) -> None:
        """Compute the arrival spectrum by multiplying it with the propagation matrix."""
        # arrival spectrum is propagation matrix with lnA information wiuth
        # source spectrum, summed over all sources > min(Amg)
        self.arr_spects = np.sum(
            self.propagation_matrix[..., np.newaxis]  # last axis contains alpha
            * self.src_spects_full,
            axis=1,
        )

        # # apply some absolute minimum
        self.arr_spects[self.arr_spects < 1e-200 * (1 / u.EV)] = 1e-200 * (
            1 / u.EV
        )  # distances x rigidities x alphas

    def compute_Eexs(self) -> None:
        """Compute the mean source energy using the first moment of the source spectrum."""
        for id, ia in np.ndindex(self.Ndistances, self.Nalphas):
            # temporary arrays that store the normaliseation per alpha per distance
            # alsostore the unnormalised source spectrum here
            src_norm = 0.0  # [EeV^(1-alpha)]
            src_Enorm = 0.0  # [EeV^(2-alpha)]
            src_spect_unnormed = np.zeros((self.NAsrcs, self.NRs))  # [ e EeV^-alpha ]

            for im in range(self.NAsrcs):
                src_spect_unnormed[im, :] = (
                    self.Asrc_pdfs[id, im, :]
                    * self.Zs[im] ** (1.0 - self.alpha_grid[ia])
                    * self.rigidities_grid.to_value(u.EV) ** (-self.alpha_grid[ia])
                )

                # zeroths and first moment
                src_norm += np.trapz(
                    y=src_spect_unnormed[im, :], x=self.rigidities_grid.to_value(u.EV)
                )
                src_Enorm += np.trapz(
                    y=self.Zs[im]
                    * self.rigidities_grid.to_value(u.EV)
                    * src_spect_unnormed[im, :],
                    x=self.rigidities_grid.to_value(u.EV),
                )

            # important to sum first, then divide to avoid Nan issues
            self.Eexs[id, ia] = (src_Enorm / src_norm) * u.EeV

        # apply some absolute minimum
        self.Eexs[self.Eexs < 1e-10 * u.EeV] = 1e-10 * u.EeV

    def compute_arrival_PDF(self) -> None:
        """Compute arrival composition PDF."""
        pass
        # self.Aearth_pdfs = np.zeros_like(self.arr_spects_full)

        # for ime in range(self.NAearths):
        #     self.Aearth_pdfs[:, ime, ...] = (
        #         self.Zs[ime] * self.arr_spects_full[:, ime, ...]
        #     )

        # self.Aearth_pdfs /= np.sum(self.Aearth_pdfs, axis=1)[:, None, ...]

    def p_gt_Rth(self, delta=None) -> None:
        """
        Probability that rigidity is anove threshold. For MG1, this is the arrival energy.

        Parameters
        ----------
        delta: float | None, default None
            Uncertainty in energy reconstruction (%)
        """
        delta = self.delta if delta is None else delta
        return 1 - np.array(
            [stats.norm.cdf(self.Rth, R, delta * R) for R in self.rigidities_grid.value]
        )

    def save(self, outfile) -> None:
        """
        Save outputs to h5py file.

        Parameters
        ----------
        outfile : str
            the filepath to the output file.
        """
        with h5py.File(outfile, "a") as f:
            config_label = f"{self.detector_type}_{self.hadr_model}"
            if config_label in f.keys():
                del f[config_label]
            config_gr = f.create_group(config_label)

            config_gr.create_dataset("alpha_grid", data=self.alpha_grid)
            config_gr.create_dataset("distances_grid", data=self.distances)
            config_gr.create_dataset(
                "log10_rigidities", data=np.log10(self.rigidities_grid.to_value(u.EV))
            )  # in log10(EV)
            config_gr.create_dataset(
                "log10_Ees_grid", data=np.log10(self.Ee_grid.to_value(u.EeV))
            )  # in log10(EV)
            config_gr.create_dataset("lnA_grid", data=self.lnA_grid)

            config_gr.create_dataset(
                "log10_arrspect_grid", data=np.log10(self.arr_spects.to_value(1 / u.EV))
            )
            config_gr.create_dataset(
                "log10_Eexs_grid", data=np.log10(self.Eexs.to_value(u.EeV))
            )

            # stored for plotting sake
            config_gr.create_dataset("src_spects", data=self.src_spects.value)
            config_gr.create_dataset("Asrc_pdf", data=self.Asrc_pdfs)
            # config_gr.create_dataset("Aearth_pdf", data=self.Aearth_pdfs)
            config_gr.create_dataset("As", data=self.As)
            config_gr.create_dataset("Zs", data=self.Zs)
