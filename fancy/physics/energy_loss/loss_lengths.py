import numpy as np
import os
from scipy import integrate
from matplotlib import pyplot as plt
from astropy import units as u
from typing_extensions import Self, Union, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm
import h5py

from fancy.interfaces.source import Source
from fancy.utils.package_data import (
    get_path_to_loss_length_tables,
)

from fancy.physics.energy_loss.loss_length_helpers import _dEdr_rev, Ltot
from fancy.physics.energy_loss.cosmology import DH 

class LossLengthModel:
    """
    Container for semi-analytic approach to modelling proton energy losses.

    This class generates the interpolation grid that is used to convert between arrival and source energies of protons, using the loss-length formula.

    This is needed to use the rigidity-dependent deflections in our model.
    """

    def __init__(
        self: Self,
    ) -> None:
        """
        Container for semi-analytic approach to modelling proton energy losses.

        This class generates the interpolation grid that is used to convert between arrival and source energies of protons, using the loss-length formula.

        This is needed to use the rigidity-dependent deflections in our model.

        Parameter
        ----------
        nthreads : int, default 8
            The number of threads to use for parallelisation
        """
        self.Earr_grid = None
        self.Esrc_grid = None
        self.distances = None

    def load_loss_length_tables(
        self : Self,
        dinits: Union[str, list] = [4],
    ) -> None:
        """
        Load the loss length tables from a file.

        Parameters
        ----------
        dinits : Union[str, list], default [4]
            The initial distance in Mpc for the computation. Can be a string to load from a file or a list of distances.
            If a string, it should be the name of the source as given in sourcedata.h5
            If a list, it should contain the distances in Mpc.
            Default is [4], which is the default distance used in the model.
        """
        if type(dinits) is str:
            source = Source()
            source.load_from_data_file(label=dinits)
            self.distances = source.distance
            loss_length_tables = get_path_to_loss_length_tables(
                f"loss_length_{dinits}.h5"
            )
        
        elif type(dinits) is list:
            self.distances = dinits
            loss_length_tables = get_path_to_loss_length_tables(
                f"loss_length_D{min(dinits)}_{max(dinits)}.h5"
            )
        else:
            raise ValueError("dinits must be either a list of floats or a string.")
        
        if not os.path.exists(loss_length_tables):
            raise FileNotFoundError(
                f"Loss length tables not found at {loss_length_tables}. Please compute them first."
            )
        
        with h5py.File(loss_length_tables, "r") as f:
            self.Esrc_grid = f["Esrc_grid"][:]
            self.Earr_grid = f["Earr_grid"][:]
            self.distances = f["distances"][:]

        return self.Esrc_grid, self.Earr_grid, self.distances

    def compute_source_energies(
        self : Self,
        Earr_grid: np.ndarray,
        dinits: Union[str, list] = [4],
        parallel: bool = True,
        njobs: int = 4,
        save: bool = False,
    ):
        """
        Compute the arrival energies for a grid of source energies.

        Parameters
        ----------
        Earr_grid : np.ndarray
            The grid of arrival energies to compute the source energies for.
            Units must be in EeV.
        dinits : Union[str, list], default [4]
            The initial distance in Mpc for the computation. Can be a string to load from a file or a list of distances.
            If a string, it should be the name of the source as given in sourcedata.h5
            If a list, it should contain the distances in Mpc.
            Default is [4], which is the default distance used in the model.
        parallel : bool, default True
            Whether to use parallelisation to speed up the computation.
        njobs : int, default 4
            The number of jobs to use for parallelisation.
        save : bool, default False
            Whether to save the computed source energies to a file.

        Returns
        -------
        np.ndarray
            The grid of arrival energies.
        """
        if type(dinits) is str:
            source = Source()
            source.load_from_data_file(label=dinits)
            self.distances = source.distance
            loss_length_tables = get_path_to_loss_length_tables(
                f"loss_length_{dinits}.h5"
            )
        
        elif type(dinits) is list:
            self.distances = dinits
            loss_length_tables = get_path_to_loss_length_tables(
                f"loss_length_D{min(dinits)}_{max(dinits)}.h5"
            )
        else:
            raise ValueError("dinits must be either a list of floats or a string.")
        
        # append the background distance here,
        # for now set to some large distance
        d_bg = 100  # in Mpc, GZK horizon for protons ~ 50 Mpc, so set to 100 Mpc just in case
        self.distances = np.append(self.distances, d_bg)
        
        self.Earr_grid = Earr_grid
        self.Esrc_grid = np.zeros(
            (len(Earr_grid), len(self.distances))
        )

        if parallel:

            args_list = [
                (i, Earr_grid, d)
                for i, d in enumerate(self.distances)
            ]

            # parallelize for each source distance
            results = Parallel(n_jobs=njobs)(
                delayed(self._get_source_energy_vec)(arg) for arg in args_list
            )

            for didx, Esrc_vec in results:
                self.Esrc_grid[:, didx] = Esrc_vec

        else:
            for i in tqdm(
                range(len(self.distances)), desc="Precomputing energy grids"
            ):
                d = self.distances[i]
                for j, Earr in enumerate(self.Earr_grid):
                    self.Esrc_grid[j, i] = self._get_source_energy(Earr, d)

        # save to file
        if save:
            with h5py.File(loss_length_tables, "w") as f:
                f.create_dataset("Esrc_grid", data=self.Esrc_grid)
                f.create_dataset("Earr_grid", data=Earr_grid)
                f.create_dataset("distances", data=self.distances)

    def _get_source_energy_vec(self, args: Tuple[int, np.ndarray, float]) -> Tuple[int, np.ndarray]:
        """
        Helper function to compute the source energies for a given distance in parallel.
        
        Parameters
        ----------
        args : Tuple[int, np.ndarray, float]
            The index of the distance, the grid of arrival energies, and the distance in Mpc.
        
        Returns
        -------
        Tuple[int, np.ndarray]
            The index of the distance and the computed arrival energies.
        """
        idx, Earr_vec, d = args
        Esrc_vec = np.zeros(len(Earr_vec))
        
        for j, Earr in enumerate(Earr_vec):
            Esrc_vec[j] = self._get_source_energy(Earr, d)
        
        return idx, Esrc_vec
    
    def _get_source_energy(self, Earr: float, d: float) -> float:
        """
        Compute the source energy for a given arrival energy and distance.

        Parameters
        ----------
        Earr : float
            The arrival energy in EeV.
        d : float
            The distance in Mpc.

        Returns
        -------
        float
            The computed arrival energy in EeV.
        """
        Earr = Earr * 1.0e18
        integrator = integrate.ode(_dEdr_rev).set_integrator("lsoda", method="bdf")
        integrator.set_initial_value(Earr, 0).set_f_params(d)
        r1 = d
        dr = min(d / 10, 10)

        while integrator.successful() and integrator.t < r1:
            integrator.integrate(integrator.t + dr)

        Esrc = integrator.y / 1.0e18
        return Esrc

    def plot_loss_length(self : Self):
        """
        PLot the loss lengths for different distances.
        """
        import matplotlib.pyplot as plt

        if self.Earr_grid is None or self.Esrc_grid is None or self.distances is None:
            raise ValueError("Loss length tables not loaded. Call load_loss_length_tables() first.")

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, d in enumerate(self.distances):
            ax.plot(
                self.Earr_grid,
                [Ltot(d / DH.to_value(u.Mpc), e * 1.0e18) for e in self.Earr_grid],
                label=f"D = {d:.2f} Mpc",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arrival Energy (EeV)")
        ax.set_ylabel("Loss Length (Mpc)")
        ax.legend()

    def plot_source_energies(self : Self):
        """
        Plot source energies as a function of distance and arrival energies.
        """
        if self.Earr_grid is None or self.Esrc_grid is None or self.distances is None:
            raise ValueError("Loss length tables not loaded. Call load_loss_length_tables() first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, d in enumerate(self.distances):
            ax.plot(
                self.Earr_grid,
                self.Esrc_grid[:, i],
                label=f"D = {d:.2f} Mpc",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arrival Energy (EeV)")
        ax.set_ylabel("Source Energy (EeV)")
        ax.legend()
