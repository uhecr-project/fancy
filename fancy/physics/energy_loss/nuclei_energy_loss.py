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
    """
    Class to determine the arrival spectrum of nuclei at Earth using a rigidity conservation approximation.
    """

    def __init__(self, data: Data, verbose=False):
        """
        Class to determine the arrival spectrum of nuclei at Earth using a rigidity conservation approximation.

        :param data: Data object from fancy

        """
        super().__init__(data, verbose)

        # raise exception if mass group = 1, since we should use proton energy loss for this
        if self.mass_group == 1:
            raise ValueError(
                f"Mass Group {self.mass_group} not valid with this approach. Use Proton Energy Loss Model isntead."
            )

    def initialise_grid(
        self,
        weights_dir: str = "./resources/composition_weights_PSB.h5",
        alpha_min=-3,
        alpha_max=10,
        Nalphas=50,
    ):
        """
        Initalise our grid using composition weights

        :param weights_dir: directory to composition weights
        :param alpha_min, alpha_max, Nalphas: the min / max and density of spectral index grid
        :param Emax, NEs: the maximum energy (in EeV) and density of the log-spaced source energy grid
        """
        super().initialise_grid(weights_dir, alpha_min, alpha_max, Nalphas)

        with h5py.File(weights_dir, "r") as f:
            rigidities = f["rigidities"][()] * u.GV
            massids = f["massids"][()]
            self.As = f["As"][()]
            self.Zs = f["Zs"][()]

            weights_full = f["full"]["weights"][()]
            weights_full_mg = f["mass_groups"]["weights_mg"][()]
            mass_group_idxlims = f["mass_groups"]["mass_group_idxlims"][()]

        # set up dimensions
        self.NAsrcs = len(massids)
        self.NAearths = len(massids)

        # limits for each mass group, required to filter out the required mass group range for As and Zs
        self.mg_lidx, self.mg_uidx = mass_group_idxlims[self.mass_group_idx]
        print(
            f"Range of masses (A) for MG{self.mass_group}: [{self.As[self.mg_lidx]}, {self.As[self.mg_uidx]}]"
        )

        # truncate the weights also
        self.weights = CubicSpline(
            x=rigidities.to_value(u.EV), y=weights_full, axis=-1
        )(self.rigidities_grid)
        self.weights_mg = CubicSpline(
            x=rigidities.to_value(u.EV),
            y=weights_full_mg[..., self.mass_group_idx, :],
            axis=-1,
        )(self.rigidities_grid)

        if self.verbose:
            # initial size of wieghts
            print(
                f"Shape of weights from file: {weights_full.shape}"
            )  # (Dsrc, Asrc, Aearths, R)
            print(
                f"Shape of mass-group integrated weights from file: {weights_full_mg.shape}"
            )  # (Dsrc, Asrc, MG, R)

            # check if this makes sense
            print(
                f"Length of rigidity array for MG{self.mass_group}: {len(self.rigidities_grid)}"
            )
            print(f"Minimum rigidity in new grid: {np.min(self.rigidities_grid):.2f}")
            print(f"Maximum rigidity in new grid: {np.max(self.rigidities_grid):.2f}")

            # new shape of wieghts
            print("\nWeights after truncating to appropriate range of rigidities:")
            print(
                f"Shape of weights for MG{self.mass_group}: {self.weights.shape}"
            )  # (Dsrc, Asrc, Aearths, R_mg)
            print(
                f"Shape of weights for MG{self.mass_group}: {self.weights_mg.shape}"
            )  # (Dsrc, Asrc, R_mg)

            # new shapes for everything
            print(
                f"\ndimensions relevant for MG{self.mass_group}: Ndmax = {self.Ndistances}, Nrigidites_mg = {self.NRs}, NAsrcs = {self.NAsrcs}, NAearths = {self.NAearths}, Nalphas={self.Nalphas}"
            )

    def _compute_source_PDF(self):
        """Compute source composition PDF"""
        self.Asrc_pdfs = np.zeros(
            (self.Ndistances, self.NAsrcs, self.NRs)
        )  # shape of (distances, source masses, rigidities)

        # iterate over each distance & rigidity
        for id, ir in np.ndindex((self.Ndistances, self.NRs)):
            self.Asrc_pdfs[id, :, ir] = self.weights_mg[id, :, ir] / np.sum(
                self.weights_mg[id, :, ir]
            )

    def compute_source_spectrum(self):
        """Computation of the source & arrival spectrum & mean energy"""
        self.src_spects_full = np.zeros(
            (self.Ndistances, self.NAsrcs, self.NRs, self.Nalphas)
        ) * (
            1 / u.EV
        )  # since its for all source compositions

        self._compute_source_PDF()  # compute the source composition PDF

        # finally get the source spectrrum, normalised over constatn marginalisaed over all source compositions
        for id, ia in np.ndindex(self.Ndistances, self.Nalphas):

            # temporary arrays that store the normaliseation per alpha per distance
            # alsostore the unnormalised source spectrum here
            src_norm = 0.0  # [EeV^(1-alpha)]
            src_spect_unnormed = np.zeros((self.NAsrcs, self.NRs))  # [ e EeV^-alpha ]

            for im in range(self.NAsrcs):

                src_spect_unnormed = (
                    self.Asrc_pdfs[id, im, :]
                    * self.Zs[im] ** (1 - self.alpha_grid[ia])
                    * self.rigidities_grid.to_value(u.EV) ** (-self.alpha_grid[ia])
                )

                # zeroths moment
                src_norm += np.trapz(
                    y=src_spect_unnormed, x=self.rigidities_grid.to_value(u.EV)
                )

            # normalisation is carried by contribution over all masses at the source
            self.src_spects_full[id, ..., ia] = (src_spect_unnormed / src_norm) * (
                1 / u.EV
            )

        # summed over all soruce compostiions that contribute in mass group
        self.src_spects = np.sum(self.src_spects_full, axis=1)

    def compute_arrival_spectrum(self):
        """Computation of the arrival spectrum"""
        self.arr_spects_full = np.zeros(
            (self.Ndistances, self.NAearths, self.NRs, self.Nalphas)
        ) * (
            1 / u.EV
        )  # here NAearths is for the *arrival composition*

        for ime, ir, ia in np.ndindex(self.NAearths, self.NRs, self.Nalphas):

            self.arr_spects_full[:, ime, ir, ia] = np.sum(
                self.weights[:, :, ime, ir] * self.src_spects_full[:, :, ir, ia], axis=1
            )

        self.arr_spects = np.sum(
            self.arr_spects_full[:, self.mg_lidx : self.mg_uidx + 1, ...], axis=1
        )

        # apply some absolute minimum
        self.arr_spects[self.arr_spects < 1e-200 * (1 / u.EV)] = 1e-200 * (1 / u.EV)

    def compute_Eexs(self):
        for id, ia in np.ndindex(self.Ndistances, self.Nalphas):

            # temporary arrays that store the normaliseation per alpha per distance
            # alsostore the unnormalised source spectrum here
            src_norm = 0.0  # [EeV^(1-alpha)]
            src_Enorm = 0.0  # [EeV^(2-alpha)]
            src_spect_unnormed = np.zeros((self.NAsrcs, self.NRs))  # [ e EeV^-alpha ]

            for im in range(self.NAsrcs):

                src_spect_unnormed = (
                    self.Asrc_pdfs[id, im, :]
                    * self.Zs[im] ** (1.0 - self.alpha_grid[ia])
                    * self.rigidities_grid.to_value(u.EV) ** (-self.alpha_grid[ia])
                )

                # zeroths and first moment
                src_norm += np.trapz(
                    y=src_spect_unnormed, x=self.rigidities_grid.to_value(u.EV)
                )
                src_Enorm += np.trapz(
                    y=self.Zs[im]
                    * self.rigidities_grid.to_value(u.EV)
                    * src_spect_unnormed,
                    x=self.rigidities_grid.to_value(u.EV),
                )

            # important to sum first, then divide to avoid Nan issues
            self.Eexs[id, ia] = (src_Enorm / src_norm) * u.EeV

        # apply some absolute minimum
        self.Eexs[self.Eexs < 1e-10 * u.EeV] = 1e-10 * u.EeV

    def compute_arrival_PDF(self):
        """Compute arrival composition PDF"""
        arr_spect_intR = np.trapz(
            y=self.arr_spects_full, x=self.rigidities_grid, axis=2
        )
        self.Aearth_pdfs = arr_spect_intR / np.sum(arr_spect_intR, axis=1)[:, None, :]

    def p_gt_Rth(self, delta=None):
        """
        Probability that rigidity is anove threshold. For MG1, this is the arrival energy

        :param delta: Uncertainty in energy reconstruction (%)
        """
        delta = self.delta if delta == None else delta
        return 1 - np.array(
            [stats.norm.cdf(self.Rth, R, delta * R) for R in self.rigidities_grid.value]
        )

    def save(self, outfile):
        """Save outputs to h5py file"""
        with h5py.File(outfile, "a") as f:
            config_label = f"{self.detector_type}_mg{self.mass_group}"
            if config_label in f.keys():
                del f[config_label]
            config_gr = f.create_group(config_label)

            config_gr.create_dataset("alpha_grid", data=self.alpha_grid)
            config_gr.create_dataset("distances_grid", data=self.distances)
            config_gr.create_dataset(
                "log10_rigidities", data=np.log10(self.rigidities_grid.to_value(u.EV))
            )  # in log10(EV)
            config_gr.create_dataset("dRs_grid", data=self.dRs_grid.to_value(u.EV))

            config_gr.create_dataset(
                "log10_arrspect_grid", data=np.log10(self.arr_spects.to_value(1 / u.EV))
            )
            config_gr.create_dataset(
                "log10_Eexs_grid", data=np.log10(self.Eexs.to_value(u.EeV))
            )

            # stored for plotting sake
            config_gr.create_dataset("src_spects_full", data=self.src_spects_full.value)
            config_gr.create_dataset("arr_spects_full", data=self.arr_spects_full.value)
            config_gr.create_dataset("Asrc_pdf", data=self.Asrc_pdfs)
            config_gr.create_dataset("Aearth_pdf", data=self.Aearth_pdfs)
            config_gr.create_dataset("As", data=self.As)
            config_gr.create_dataset("Zs", data=self.Zs)
            config_gr.create_dataset(
                "As_mg", data=self.As[self.mg_lidx : self.mg_uidx + 1]
            )
