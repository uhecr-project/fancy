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

    def __init__(self, data : Data, verbose=False):
        """
        Abstract base class for energy loss calculations.
        """
        self.mass_group = data.detector.mass_group
        self.mass_group_idx = int(self.mass_group)-1
        self.detector_type = data.detector.label
        self.Eth = data.detector.Eth
        self.Rth = data.detector.Rth
        self.Rth_max = data.detector.Rth_max
        self.verbose = verbose

    @abstractmethod
    def initialise_grid(
            self, 
            weights_dir : str = "./resources/composition_weights_PSB.h5", 
            alpha_min=-3, 
            alpha_max=10, 
            Nalphas=50,
        ):
        '''
        Initalise our grid using composition weights
        
        :param weights_dir: directory to composition weights
        :param alpha_min, alpha_max, Nalphas: the min / max and density of spectral index grid
        '''
        # set grid for spectral index
        self.alpha_grid = np.linspace(alpha_min, alpha_max, Nalphas)
        self.Nalphas = Nalphas
        print(f"Shape of alpha grid: [{np.min(self.alpha_grid):.1f} : {np.max(self.alpha_grid):.1f} : {self.Nalphas}]")

        if not os.path.exists(weights_dir):
            raise FileNotFoundError(f"Composition weights file is missing. Re-run composition weight calculation.")

        with h5py.File(weights_dir, "r") as f:
            self.distances = f["distances"][()] * u.Mpc

        self.Ndistances = len(self.distances)

        self.Eexs = np.zeros((self.Ndistances, self.Nalphas)) * u.EeV
    
    @abstractmethod
    def compute_Eexs(self):
        '''Compute expected energies for all distance and energies'''
        pass

    @abstractmethod
    def p_gt_thresh(self, delta):
        """
        Probability that rigidity is anove threshold. For MG1, this is the arrival energy

        :param delta: Uncertainty in energy reconstruction (%)
        """
        pass
