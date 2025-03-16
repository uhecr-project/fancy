from abc import ABC, abstractmethod
from typing import List, Tuple, Union
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
        self.data = data
        self.detector_type = data.detector.label
        self.mass_model = data.detector.mass_model
        self.verbose = verbose

    @abstractmethod
    def initialise_grid(
        self,
        matrix_dir: str = "./resources/composition_weights_PSB.h5",
        alpha_min : float=-3,
        alpha_max : float=10,
        Nalphas : int=50,
        Eearth_min : Union[float, None] = None,
        Eearth_max : float = 400,
        NEearths : int = 200
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
        Eearth_min : float, default=1
            the minimum energy at Earth for the grid
        Eearth_max : float, default=10
            the maximum energy at Earth for the grid
        NEearths : int, default=50
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
        # lower limit should be at energy threshold 
        # this is to preserve the normalisaation 
        # of event samples that we use
        # Eearth_min = self.data.detector.Eth if Eearth_min is None else Eearth_min
        Eearth_min = 1
        Eearth_grid = np.logspace(
            np.log10(Eearth_min),
            np.log10(Eearth_max),
            NEearths + 1
        ) * u.EeV
        self.Eearth_grid = np.sqrt(Eearth_grid[1:] * Eearth_grid[:-1]) # bin centers
        self.dEearth_grid = np.diff(Eearth_grid)  # bin widths

        self.Ndistances = len(self.distances)
        self.NEearths = len(self.Eearth_grid)

    # @abstractmethod
    # def compute_Eexs(self):
    #     """Compute expected energies for all distance and energies"""
    #     pass

    # @abstractmethod
    # def p_gt_Rth(self, delta):
    #     """
    #     Probability that rigidity is anove threshold. For MG1, this is the arrival energy

    #     :param delta: Uncertainty in energy reconstruction (%)
    #     """
    #     pass
