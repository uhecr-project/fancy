from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy import stats, optimize

import os, h5py
import numpy as np
import astropy.units as u


from fancy import Data


class EnergyLoss(ABC):
    """
    Abstract base class for energy loss calculations.
    """

    def __init__(self, data: Data, verbose=False):
        """
        Abstract base class for energy loss calculations.
        """
        self.lnA_params = data.detector.lnA_params
        self.detector_type = data.detector.label
        self.hadr_model = data.detector.hadr_model
        self.Eth = data.detector.Eth
        self.verbose = verbose

    @abstractmethod
    def initialise_grid(
        self,
        matrix_dir: str = "./resources/composition_weights_PSB.h5",
        alpha_min : float=-3,
        alpha_max : float=10,
        Nalphas : int=50,
        Eemin : float = 1,
        Eemax : float = 400,
        NEes : int = 200
    ) -> None:
        """
        Initalise our grid using composition weights.
        
        Parameter:
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
        # set grid for spectral index
        self.alpha_grid = np.linspace(alpha_min, alpha_max, Nalphas)
        self.Nalphas = Nalphas
        print(
            f"Shape of alpha grid: [{np.min(self.alpha_grid):.1f} : {np.max(self.alpha_grid):.1f} : {self.Nalphas}]"
        )

        if not os.path.exists(matrix_dir):
            raise FileNotFoundError(
                f"Composition weights file is missing. Re-run composition weight calculation."
            )

        with h5py.File(matrix_dir, "r") as f:
            massids = f["massids"][()]
            self.distances = f["distances"][()] * u.Mpc
            self.As = f["As"][()]
            self.Zs = f["Zs"][()]

        # set up dimensions
        self.NAsrcs = len(massids)
        self.NAearths = len(massids)

        # generate energy grid for Earth
        self.Ee_grid = np.logspace(
            np.log10(Eemin),
            np.log10(Eemax),
            NEes
        ) * u.EeV

        # the mean and sigma of the lnA read from the data.detector.lnA object
        mu_sigma_lnAs = self.lnA_params[:,0,np.newaxis] * np.log10(self.Ee_grid.value)[np.newaxis,:] + self.lnA_params[:,1,np.newaxis]

        # for each energy bin, we take some number of samples and take the mean value as the lnA
        self.lnA_grid = np.zeros_like(self.Ee_grid.value)
        Nsamples = 1000

        for ie in range(len(self.Ee_grid)):
            mu_lnA, sigma_lnA = mu_sigma_lnAs[:,ie]
            lnA_samples = stats.norm.rvs(loc=mu_lnA, scale=sigma_lnA, size=Nsamples)  
            self.lnA_grid[ie] = np.mean(lnA_samples)

        self.rigidities_grid = (self.Ee_grid.value  / (0.5 * np.exp(self.lnA_grid))) * u.EV # in EV
        self.Rmax = np.min(self.rigidities_grid)
        self.Rmin = np.max(self.rigidities_grid)

        self.Ndistances = len(self.distances)
        self.NRs = len(self.rigidities_grid)

        self.Eexs = np.zeros((self.Ndistances, self.Nalphas)) * u.EeV

    @abstractmethod
    def compute_Eexs(self):
        """Compute expected energies for all distance and energies"""
        pass

    @abstractmethod
    def p_gt_Rth(self, delta):
        """
        Probability that rigidity is anove threshold. For MG1, this is the arrival energy

        :param delta: Uncertainty in energy reconstruction (%)
        """
        pass
