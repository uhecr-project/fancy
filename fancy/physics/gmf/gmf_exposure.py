"""
Class that calculates the exposure at the Galactic boundary, backtracked from the detector exposure through a given GMF model.
"""
import os
import typing
from typing_extensions import Self
import numpy as np
import healpy
import h5py
import pickle
from scipy.interpolate import make_interp_spline
from astropy.coordinates import SkyCoord
import astropy.units as u
from joblib import Parallel, delayed
from tqdm_joblib import ParallelPbar

from fancy import Data
from fancy.detector.exposure import m_dec
from fancy.utils.package_data import get_path_to_gmf_tables

try:
    import crpropa as cr
except ImportError:
    cr = None

class GMFExposure:
    """
    Class that calculates the exposure at the Galactic boundary, backtracked from the detector exposure through a given GMF model.
    """
    __gmf_models: typing.ClassVar[list] = [
        "JF12",
        "UF23all",
        "UF23base",
        "UF23allTurb",
        "UF23baseTurb",
        "PT11",
        "TF17",
        "KST24",  # to include in the future
    ]  # type of GMF models
    __Nmodels_UF23: int = 8  # number of models in UF23
    def __init__(self : Self, data : Data, gmf_model : str = "UF23base", nside : int = 64):
        """
        Initialize the GMFExposure class.

        Parameters:
        ---–-------
        data : Data
            the data container that contains the detector information.
        gmf_model : str, default = "UF23base"
            The GMF model to use. Options include:
                "JF12", "UF23base", "UF23baseTurb", "UF23all", "UF23allTurb", "PT11", "TF17", "KST24".
        nside : int
            The HEALPix nside parameter for the exposure map resolution. Must be a power of 2.
        """
        self.data = data
        self.detector = data.detector.label
        self.gmf_model = gmf_model
        self.nside = nside
        self.npix = healpy.nside2npix(nside)

        assert gmf_model in self.__gmf_models, (
            f"GMF model {gmf_model} is not an available GMF model."
        )

        # raise exception if CRPropa is not installed, since it requires CRPropa
        if cr is None:
            raise ImportError(
                "CRPropa is not installed and is required for using this module."
            )

        # calculate the coordinates in SkyCoord format
        # converting from pixels -> skycoord
        pix_arr = np.arange(0, self.npix, 1, dtype=int)
        uvs_healpy = np.array(healpy.pix2vec(self.nside, pix_arr)).T
        self.coords_healpy = SkyCoord(
            uvs_healpy, frame="galactic", representation_type="cartesian"
        )

        # calculate the exposure map already
        self.exposure_map = self.calculate_exposure_map()

        # containers
        self.deflected_exposure_maps = None  # to be filled later
        self.defl_exp_interpolators = None
        self.config = {
            "gmf_model": gmf_model,
            "nside": nside,
            "npix" : self.npix,
            "detector" : data.detector.label,
        }

        # checks
        self.__computed_defl_maps = False
        self.__set_interpolator = False

    def calculate_exposure_map(self : Self) -> np.ndarray:
        """Compute the exposure as a function of declination in healpy."""
        # first transform coordianates to declination
        self.coords_healpy.representation_type = "unitspherical"
        self.coords_healpy.transform_to("icrs")
        decs_healpy_grid = self.coords_healpy.icrs.dec.rad

        # compute exposure, which is function of declination only
        p = self.data.detector.params  # exposure parameters
        m_grid = m_dec(decs_healpy_grid, p)
        if self.data.detector.label == "all_sky":
            m_grid = np.ones_like(m_grid)

        exposures = p[3] / p[4] * m_grid
        exposures /= np.max(exposures)  # normalize total exposure

        # transform the coordinates back to galactic
        self.coords_healpy.transform_to("galactic")

        return exposures

    def plot_exposure_map(self):
        """
        Plot the calculated normalized exposure map.

        Just used to test that the exposure map is calculated correctly.

        Returns:
        None
        """
        healpy.mollview(self.exposure_map, title=f"Exposure Map for {self.detector}", unit="Normalized Exposure")
    
    def run_single_backpropagation(self : Self, rigidity : float, n_samples : int, seed : int = 0) -> np.ndarray:
        """
        Run a single backpropagation for a set of UHECR arrival directions.

        This means for a single simulation instance, i.e. for a single GMF model instance.

        Parameters:
        ----------
        rigidity : float
            the rigidity in EV
        n_samples : int
            the number of samples to backpropagate
        seed : int, default = 0
            the random seed for the magnetic field turbulence. If 0, a random seed is chosen.

        Returns:
        -------
        defl_uvs : np.ndarray
            the deflected UHECR directions at the Galactic boundary in galactic cartesian unit vectors
        """
        # first sample arrival directions according to the exposure map
        cumsum_exp = np.cumsum(self.exposure_map)
        sampled_pix = np.searchsorted(cumsum_exp, np.random.rand(n_samples) * cumsum_exp[-1])
        uhecr_uvs = np.array(healpy.pix2vec(self.nside, sampled_pix)).T

        defl_uvs = np.zeros_like(uhecr_uvs)

        # Position of the Earth in galactic coordinates
        pos_earth = cr.Vector3d(-8.5, 0, 0) * cr.kpc

        # PID set to protons, take charge into account via rigidity
        # negative since we backtrack
        pid = -cr.nucleusId(1, 1)

        # observer at galactic boundary (20 kpc)
        obs = cr.Observer()
        obs.add(cr.ObserverSurface(cr.Sphere(cr.Vector3d(0), 20 * cr.kpc)))

        rng = np.random.default_rng()
        seed = int(rng.integers(low=1, high=1000000)) if seed == 0 else seed
        sim = self.__setup_simulation(obs, seed)

        for isamp, uhecr_uv in enumerate(uhecr_uvs):
            uhecr_vector3d = cr.Vector3d(*uhecr_uv)

            c = cr.Candidate(
                cr.ParticleState(pid, rigidity * cr.EeV, pos_earth, uhecr_vector3d)
            )

            sim.run(c)
            uhecr_defl_v3d = c.current.getDirection()

            # simply ignore any vector that is nan so that the mean
            # would not include any nans
            if np.any(np.isnan(uhecr_defl_v3d)):
                continue

            defl_uvs[isamp, :] = np.array(
                [uhecr_defl_v3d.x, uhecr_defl_v3d.y, uhecr_defl_v3d.z]
            )

        return defl_uvs
    
    def calculate_single_deflected_exposure_map(self : Self, rigidity : float, n_samples : int = 50000, seed : int = 0) -> np.ndarray:
        """
        Calculate the deflected exposure map at the Galactic boundary for a single rigidity.

        Parameters:
        ----------
        rigidity : float
            the rigidity in EV
        n_samples : int, default = 50000
            the number of samples to use for the backpropagation.
        seed : int, default = 0
            the seed to use for the turbulent field. 
            
            If provided, then the same seed will be used. Otherwise (for seed = 0), a randomly sampled seed is used.

            This argument is meant to be used more for debugging purposes.

        Returns
        -------
        deflected_exposure_map : np.ndarray
            the deflected exposure map at the Galactic boundary for the given rigidity.
        """
        defl_uvs = self.run_single_backpropagation(rigidity, n_samples, seed)

        # now calculate the exposure map from the deflected directions
        defl_pix = np.array(healpy.vec2pix(self.nside, defl_uvs[:, 0], defl_uvs[:, 1], defl_uvs[:, 2]))

        defl_exp_map = np.bincount(defl_pix, minlength=self.npix).astype(float)
        defl_exp_map = defl_exp_map / len(defl_exp_map) # normalize 

        # remove any < 0 values
        defl_exp_map[defl_exp_map < 0] = 0

        return defl_exp_map
    
    def calculate_deflected_exposure_map(self : Self, rigidities : np.ndarray, n_samples : int = 50000, n_rand : int = 10, n_jobs : int = 1, seed : int = 0) -> np.ndarray:
        """
        Calculate the deflected exposure map at the Galactic boundary for a set of rigidities.

        Parameters:
        ----------
        rigidities : np.ndarray
            the array of rigidities in EV
        n_samples : int, default = 50000
            the number of samples to use for the backpropagation.

            If you have turbulent fields, then this number is the number of samples per turbulence realization.
        n_rand : int, default = 10
            the number of random realizations to average over for turbulent fields.

            If no turbulent fields are used, then this number is ignored.
        n_jobs : int, default = 1
            The number of jobs to run. If n_jobs > 1, then joblib is used to parallelize the backpropagation over multiple cores, parallelizing over different rigidities (and different seeds with turbulence). Otherwise, the backpropagation is run in a single thread.
        seed : int, default = 0
            the seed to use for the turbulent field. 
            
            If provided, then the same seed will be used for all realizations. Otherwise (for seed = 0), each realization will use a randomly sampled seed. The default value is 0, which means random seeds.

            This argument is meant to be used more for debugging purposes.

        Returns
        -------
        deflected_exposure_maps : np.ndarray
            the deflected exposure maps at the Galactic boundary for each rigidity. If turbulent fields are used, then the exposure maps are averaged over n_rand realizations.
            Shape is (n_rigidities, npix)
        """
        defl_exp_maps_all = np.zeros((len(rigidities), self.npix, n_rand))

        # if no turbulence, we only need one realization
        if self.gmf_model.find("Turb") < 0:
            print(f"No turbulent fields detected in GMF model {self.gmf_model}. Setting n_rand = 1.")
            n_rand = 1

        if n_jobs == 1:
            for irig, rig in enumerate(rigidities):
                for irand in range(n_rand):
                    defl_exp_maps_all[irig, :, irand] = self.calculate_single_deflected_exposure_map(rig, n_samples, seed)

        elif n_jobs > 1:
            bt_args = [
                (rig, n_samples, seed) for rig in rigidities for _ in range(n_rand)
            ]
            bt_idces = [
                (irig, irand) for irig in range(len(rigidities)) for irand in range(n_rand)
            ]
            results = ParallelPbar("Calculating deflected exposure map: ")(n_jobs=n_jobs)(
                delayed(self.calculate_single_deflected_exposure_map)(*args) for args in bt_args
            )
            for i, result in enumerate(results):
                irig, irand = bt_idces[i]
                defl_exp_maps_all[irig, :, irand] = result

        # average over random realizations
        defl_exp_maps = np.mean(defl_exp_maps_all, axis=2)

        # normalize with all maps
        defl_exp_maps /= np.max(defl_exp_maps)

        # assign deflection exposure maps
        self.deflected_exposure_maps = defl_exp_maps
        self.config["rigidities"] = rigidities
        self.config["n_rand"] = n_rand
        self.config["n_samples"] = n_samples

        self.__computed_defl_maps = True

        return defl_exp_maps

    def set_interpolated_deflected_exposure_map(self : Self, order : int = 0, **interp_args : dict) -> None:
        """
        Set the interpolated deflected exposure map at the Galactic boundary.

        Parameters:
        ----------
        order : int, default = 0
            the interpolation order. Default is 0 (nearest neighbor). Other options are 1 (linear), 2 (quadratic), and 3 (cubic).
        interp_args : dict
            additional arguments to pass to the scipy.interpolate.make_interp_spline function.
        """
        if not self.__computed_defl_maps:
            raise RuntimeError(
                "Deflected exposure maps have not been computed yet. Please run calculate_deflected_exposure_map() first."
            )

        rigidities = self.config["rigidities"]

        self.defl_exp_interpolators = make_interp_spline(
            rigidities, self.deflected_exposure_maps, axis=0, k=order, **interp_args
        )

        self.__set_interpolator = True

    def get_deflected_exposure_map(self : Self, rigidity : float) -> np.ndarray:
        """
        Get the deflected exposure map at the Galactic boundary for a given rigidity.

        Parameters:
        ----------
        rigidity : float
            the rigidity in EV

        Returns:
        -------
        deflected_exposure_map : np.ndarray
            the deflected exposure map at the Galactic boundary for the given rigidity.
        """
        if not self.__set_interpolator:
            raise RuntimeError(
                "Deflected exposure interpolator has not been set yet. Please run get_interpolated_deflected_exposure_map() first."
            )
        if rigidity < self.config["rigidities"][0] or rigidity > self.config["rigidities"][-1]:
            raise ValueError(
                f"Rigidity {rigidity} EV is out of bounds for the interpolator. Valid range is [{self.config['rigidities'][0]}, {self.config['rigidities'][-1]}] EV."
            )

        defl_exp_map = self.defl_exp_interpolators(rigidity)

        return defl_exp_map
    
    def plot_deflected_exposure_map(self : Self, rigidity : float):
        """
        Plot the calculated normalized deflection exposure map.

        Returns:
        None
        """
        if not self.__set_interpolator:
            raise RuntimeError(
                "Deflected exposure interpolator has not been set yet. Please run get_interpolated_deflected_exposure_map() first."
            )
        healpy.mollview(self.get_deflected_exposure_map(rigidity), title=f"GMF Exposure Map for {self.detector} using {self.gmf_model}", unit="Normalized Exposure")
    
    def save(self : Self, outfile : str = "gmf_exposure.h5") -> None:
        """
        Save the deflected exposure information into a hdf5 file.

        A pickle file is required since we save the interpolators directly. Maybe in a future version this need not be saved.

        Parameters:
        -----------
        filename : str
            the filename to save the data to.
        """
        assert outfile.find(".h5") > 0, (
            f"Output file {outfile} needs to have a .h5 extension."
        )
        if not self.__computed_defl_maps:
            raise RuntimeError(
                "Deflected exposure maps have not been computed yet. Please run calculate_deflected_exposure_map() first."
            )
        
        with h5py.File(str(get_path_to_gmf_tables(outfile)), "a") as f:
            config_label = f"{self.detector}_{self.gmf_model}"
            if config_label in f.keys():
                del f[config_label]

            config_gr = f.create_group(config_label)
            # save config
            config_gr.attrs["nside"] = self.nside
            config_gr.attrs["gmf_model"] = self.gmf_model
            config_gr.attrs["detector"] = self.detector
            config_gr.attrs["npix"] = self.npix

            config_gr.create_dataset(
                "rigidities", data=self.config["rigidities"]
            )
            config_gr.create_dataset(
                "deflected_exposure_maps", data=self.deflected_exposure_maps, compression='gzip'
            )
            config_gr.create_dataset(
                "exposure_map", data=self.exposure_map, compression='gzip'
            )

    def load_from_tables(self : Self, infile : str = "gmf_exposure.h5") -> None:
        """
        Load tabulated results from h5py File.

        Parameter:
        ----------
        infile : str
            the path to the input file. must be in .h5 format.
        """
        assert infile.find(".h5") > 0, (
            f"Input file {infile} needs to have a .h5 extension."
        )
        with h5py.File(str(get_path_to_gmf_tables(infile)), "r") as f:
            config_label = f"{self.detector}_{self.gmf_model}"
            assert config_label in f.keys(), (
                f"Configuration {config_label} not found in {infile}."
            )
            config_gr = f[config_label]

            self.config['nside'] = config_gr.attrs['nside']
            self.config['gmf_model'] = config_gr.attrs['gmf_model']
            self.config['detector'] = config_gr.attrs['detector']
            self.config['npix'] = config_gr.attrs['npix']
            self.config['rigidities'] = config_gr['rigidities'][:]

            self.deflected_exposure_maps = config_gr['deflected_exposure_maps'][:]
            self.exposure_map = config_gr['exposure_map'][:]

        self.nside = self.config['nside']
        self.npix = self.config['npix']
        self.detector = self.config['detector']
        self.gmf_model = self.config['gmf_model']

        self.__computed_defl_maps = True
    
    def __setup_simulation(self, obs, mt_num = 0, seed : int = 0):
        """
        Prepare the crpropa backtracking simulation.

        Parameters
        ----------
        obs : cr.Observer
            CRPropa observer object
        mt_num : int
            the montel number for the UF23 model

        Returns
        -------
        cr.ModuleList
            the simulation object containing the observer and propagation model
        """
        sim = cr.ModuleList()

        # setup magnetic field
        if self.gmf_model == "JF12":
            gmf_cr = cr.JF12Field()
            gmf_cr.randomStriated(seed)
            gmf_cr.randomTurbulent(seed)

        elif self.gmf_model == "UF23all":
            gmf_cr = cr.UF23Field(mt_num)

        elif self.gmf_model == "UF23base":
            gmf_cr = cr.UF23Field(0)

        elif self.gmf_model == "UF23allTurb":

            gmf_cr = cr.UF23Field(mt_num)
            gmf_cr.randomStriated(seed)
            gmf_cr.randomTurbulent(seed)

        elif self.gmf_model == "UF23baseTurb":

            gmf_cr = cr.UF23Field(0)
            gmf_cr.randomStriated(seed)
            gmf_cr.randomTurbulent(seed)

        elif self.gmf_model == "PT11":
            gmf_cr = cr.PT11Field()
        elif self.gmf_model == "TF17":
            gmf_cr = cr.TF17Field()
        else:
            raise NotImplementedError(
                f"GMF model {self.gmf_model} is not implemented yet."
            )

        # Propagation model, parameters: (B-field model, target error, min step, max step)
        sim.add(cr.PropagationCK(gmf_cr, 1e-3, 0.1 * cr.parsec, 100 * cr.parsec))

        sim.add(obs)  # add observer at galactic boundary
        return sim
