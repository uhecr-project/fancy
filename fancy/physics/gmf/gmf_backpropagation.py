"""Class to determine backpropagated events from a given distribution of UHECRs with a particular detector."""

import os
import pickle
import typing

import numpy as np
from astropy.coordinates import SkyCoord
from cmdstanpy import CmdStanModel
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import ParallelPbar
from scipy.stats import norm, truncnorm
from typing_extensions import Self, Union
from vMF import sample_vMF

from fancy.utils.package_data import get_path_to_stan_includes, get_path_to_stan_file

from fancy import Data
from fancy.utils.helpers import truncated_lognormal_sample

# os.environ["OPENMP_NUM_THREADS"] = f"{int(os.cpu_count() * 0.8)}"  # to avoid openmp conflicts
os.environ["OPENMP_NUM_THREADS"] = "1"  # to avoid openmp conflicts

try:
    import crpropa as cr
except ImportError:
    cr = None


class GMFBackPropagation:
    """Class to simulate back propagation of UHECRs within a given dataset (simulated or real data) and obtain the deflected events and their individual kappa values."""

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

    def __init__(self: Self, data: Data, gmf_model: str = "JF12") -> None:
        """
        Class to simulate back propagation of UHECRs within a given dataset (simulated or real data).

        Parameters
        ----------
        data : Data
            object generated from fancy.interfaces.data
        gmf_model : str
            the GMF model considered for backpropagation.
        """
        self.gmf_model = gmf_model

        # settings for the detector
        self.mean_lnA_grid = None
        self.var_lnA_grid = None
        self.E_lnA_grid = None
        self.logE_stat = data.detector.logE_stat
        self.logE_sys = data.detector.logE_sys
        self.kappa_det = data.detector.kappa_d  # default value for the detector kappa
        self.Eth = data.detector.Eth  # default value for the threshold energy in EeV
        self.Eth_max = data.detector.Eth_max  # default value for the maximum energy in EeV

        self.mean_lnA_grid = data.detector.mean_lnA
        self.var_lnA_grid = data.detector.var_lnA
        self.lnA_logE_grid = data.detector.lnA_logE_grid
        self.mean_lnA_stat = data.detector.mean_lnA_stat
        self.mean_lnA_sys = data.detector.mean_lnA_sys
        self.var_lnA_stat = data.detector.var_lnA_stat
        self.var_lnA_sys = data.detector.var_lnA_sys


        assert gmf_model in self.__gmf_models, (
            f"GMF model {gmf_model} is not an available GMF model."
        )

        # raise exception if CRPropa is not installed, since it requires CRPropa
        if cr is None:
            raise ImportError(
                "CRPropa is not installed and is required for using this module."
            )

        # uhecr direction properties
        self.uhecr_uv = (data.uhecr.coord.cartesian.xyz.value).T
        self.uhecr_coords_earth = SkyCoord(
            self.uhecr_uv, frame="galactic", representation_type="cartesian"
        )
        self.uhecr_coords_earth.representation_type = "unitspherical"

        # uhecr energy properties
        self.uhecr_energy = data.uhecr.energy
        self.Nuhecrs = len(self.uhecr_energy)

        # store the rigidities generated from the backpropagation model
        self.rigidities = None

        # compile vMF model
        self.__compile_vMFmodel()

    def __compile_vMFmodel(self: Self) -> None:
        """Compile the vMF fitting function used in stan."""
        # model to fit vMF with
        stanc_options = {"include-paths": str(get_path_to_stan_includes("vMF"))}

        self.vMF_model = CmdStanModel(
            stan_file=str(get_path_to_stan_file("vMF", "fit_from_vMF.stan")),
            model_name="vMF",
            stanc_options=stanc_options,
        )

    # parallelize for each UHECR
    def run_backpropagation(
        self: Self, Nsamples: int = 500, njobs: int = 4, parallel: bool = True
    ) -> None:
        """
        Run backpropagation for all UHECRs.

        Parameters
        ----------
        Nsamples : int
            number of samples to generate for each UHECR.
        njobs : int, default=4
            number of jobs to run in parallel. not used if parallel=False
        parallel : bool
            flag whether to run in parallel or not.
        """
        # if UF23, make sure that number of samples are divisible by
        # number of models in UF23 (8 models)
        # such that we have uniform number of samples per model
        # we just multiply the number of samples by 8
        if self.gmf_model.find("UF23all") != -1:
            Nsamples = int(Nsamples * self.__Nmodels_UF23)

        self.arr_sampled_uvs = np.zeros((self.Nuhecrs, Nsamples, 3))
        self.defl_sampled_uvs = np.zeros((self.Nuhecrs, Nsamples, 3))
        self.defl_mean_uvs = np.zeros((self.Nuhecrs, 3))
        self.time_delays = np.zeros((self.Nuhecrs, Nsamples))

        # generate backtrakcing arguments for all uhecrs
        bt_args = self._generate_backtracking_arguments(Nsamples)

        # use joblib to run parallel jobs otherwise use serial
        if parallel:
            results = ParallelPbar("Running Backpropagation: ")(n_jobs=njobs)(
                delayed(self.run_single_backpropagation)(arg) for arg in bt_args
            )
        else:
            results = []
            for iarg, arg in tqdm(enumerate(bt_args), desc="Running Backpropagation: ", total=len(bt_args)):
                results.append(self.run_single_backpropagation(arg))

        # append the results
        for uhecr_idx, ars, dls, dlm, td in results:
            self.arr_sampled_uvs[uhecr_idx, ...] = ars
            self.defl_sampled_uvs[uhecr_idx, ...] = dls
            self.defl_mean_uvs[uhecr_idx, :] = dlm
            self.time_delays[uhecr_idx, :] = td

        # take care of nans
        for uhecr_idx in range(self.Nuhecrs):
            defl_sample_uv = self.defl_sampled_uvs[uhecr_idx, ...]
            # find a sample vector that is not nan so that we can assign it to a nan vector
            # since the array is collapsed, we just take the first three elements, which would be one
            # vector that doesnt have nans
            a_sample_with_nonans = defl_sample_uv[~np.isnan(defl_sample_uv)][0:3]
            # assign the deflected unit vectors such that if it is nan, we set it to
            # a non-NaN vector. Since these are anyways rare and are sampled over for kappa_GMF
            # setting one to another should not make too much difference
            self.defl_sampled_uvs[uhecr_idx, ...] = np.where(
                np.isnan(defl_sample_uv), a_sample_with_nonans, defl_sample_uv
            )

        self.uhecr_coords_gb = SkyCoord(
            self.defl_mean_uvs, frame="galactic", representation_type="cartesian"
        )
        self.uhecr_coords_gb.representation_type = "unitspherical"

    def compute_kappa_gmf(self: Self, njobs : int = 4) -> None:
        """Compute kappa gmf & theta by fitting to vMF distribution pre-computed via stan."""
        self.kappa_gmfs = ParallelPbar("Calculating kappa_GMF: ")(n_jobs=njobs)(delayed(self._get_kappa_gmf)(uhecr_idx) for uhecr_idx in range(self.Nuhecrs))
        self.thetaPs = self.f_theta(self.kappa_gmfs)  # for plotting purposes
    

    def run_single_backpropagation(self: Self, bt_arg: tuple) -> tuple:
        """
        Run a single back-propagation simulation for a given UHECR.

        Parameters
        ----------
        bt_arg : tuple
            tuple containing the UHECR index, sampled arrival directions and sampled rigidities.

        Returns
        -------
        a tuple containing the UHECR index, sampled arrival directions, deflected directions, mean deflected direction and time delays.
        """
        uhecr_idx, uhecr_uvs, uhecr_Rs = bt_arg
        uhecr_defl_uvs = np.zeros_like(uhecr_uvs)
        uhecr_time_delays = np.zeros(uhecr_uvs.shape[0])

        # Position of the Earth in galactic coordinates
        pos_earth = cr.Vector3d(-8.5, 0, 0) * cr.kpc

        # PID set to protons, take charge into account via rigidity
        # negative since we backtrack
        pid = -cr.nucleusId(1, 1)

        # observer at galactic boundary (20 kpc)
        obs = cr.Observer()
        obs.add(cr.ObserverSurface(cr.Sphere(cr.Vector3d(0), 20 * cr.kpc)))

        # store the mean deflected direction at the GB
        # mean in vMF distribution == arithmetic mean
        uhecr_mean_defl_v3d = cr.Vector3d(0.0, 0.0, 0.0)

        for k, uhecr_uv in enumerate(uhecr_uvs):
            # setup simulations once every 50 samples
            if k % 50 == 0:
                # map the model number given the number of samples we dealt so far
                # so that we cover all models
                mt_num = int(np.floor(k / (len(uhecr_uvs) // self.__Nmodels_UF23)))
                sim = self.__setup_simulation(obs, mt_num)

            # get crropa Vector3D version of sampled arrival directions
            uhecr_vector3d = cr.Vector3d(*uhecr_uv)

            c = cr.Candidate(
                cr.ParticleState(pid, uhecr_Rs[k] * cr.EeV, pos_earth, uhecr_vector3d)
            )
            sim.run(c)

            # compute time delay and direction
            uhecr_time_delays[k] = self.__get_time_delay(c, pos_earth)
            uhecr_defl_v3d = c.current.getDirection()

            # store sampled deflected directions
            uhecr_defl_uvs[k, :] = np.array(
                [uhecr_defl_v3d.x, uhecr_defl_v3d.y, uhecr_defl_v3d.z]
            )

            # simply ignore any vector that is nan so that the mean
            # would not include any nans
            if np.any(np.isnan(uhecr_defl_v3d)):
                continue

            # for calculation of arithmetic mean
            uhecr_mean_defl_v3d += uhecr_defl_v3d

        # divide by number of samples to get arithmetic mean
        uhecr_mean_defl_v3d /= len(uhecr_Rs)
        uhecr_mean_defl_uv = np.array(
            [uhecr_mean_defl_v3d.x, uhecr_mean_defl_v3d.y, uhecr_mean_defl_v3d.z]
        )
        uhecr_mean_defl_uv /= np.linalg.norm(uhecr_mean_defl_uv)  # normalise incase

        return (
            uhecr_idx,
            uhecr_uvs,
            uhecr_defl_uvs,
            uhecr_mean_defl_uv,
            uhecr_time_delays,
        )

    def __get_time_delay(self: Self, c, pos_earth) -> np.ndarray:
        """
        Return delay between entering the galactic disc and arrival at Earth through magnetic field.

        Parameters
        ----------
        c : cr.Candidate
            CRPropa candidate object
        pos_earth : cr.Vector3d
            position of the Earth in galactic coordinates

        Returns
        -------
        time delay in years
        """
        return (
            (c.getTrajectoryLength() - c.current.getPosition().getDistanceTo(pos_earth))
            / cr.c_light
            / (60 * 60 * 24 * 365)
        )

    def __setup_simulation(self: Self, obs, mt_num: int):
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
        rng = np.random.default_rng()

        # setup magnetic field
        if self.gmf_model == "JF12":
            seed = int(rng.integers(low=0, high=10000000))
            gmf_cr = cr.JF12Field()
            gmf_cr.randomStriated(seed)
            gmf_cr.randomTurbulent(seed)

        elif self.gmf_model == "UF23all":
            gmf_cr = cr.UF23Field(mt_num)

        elif self.gmf_model == "UF23base":
            gmf_cr = cr.UF23Field(0)

        elif self.gmf_model == "UF23allTurb":
            seed = int(rng.integers(low=0, high=10000000))

            gmf_cr = cr.UF23Field(mt_num)
            gmf_cr.randomStriated(seed)
            gmf_cr.randomTurbulent(seed)

        elif self.gmf_model == "UF23baseTurb":
            seed = int(rng.integers(low=0, high=10000000))

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
        sim.add(cr.PropagationCK(gmf_cr, 1e-4, 0.1 * cr.parsec, 100 * cr.parsec))

        sim.add(obs)  # add observer at galactic boundary
        return sim

    def _generate_backtracking_arguments(self: Self, Nsamples: int = 500) -> list:
        """
        Generate arguments used for backtracking.

        Here we sample the arrival directions and rigidities for each UHECR.

        The arrival directions is sampled via a vMF distribution using the angular
        reconstruction uncertainty.
        The energy is sampled via a normal distribution using the energy
        uncertainty, which is then used to compute the mean lnA & sigma lnA.
        The composition is sampled for each energy sample, which is then
        combined to get rigidity samples.

        Parameters
        ----------
        Nsamples : int
            number of samples to generate for each UHECR.

        Returns
        -------
        a list containing a tuple of sampled directions & rigidities for each UHECR.
        """
        # generate arguments
        bt_args = []
        self.rigidities = []
        for i in range(self.Nuhecrs):
            # sample arrival directions via vMF
            uhecr_sampled_uvs = sample_vMF(
                self.uhecr_uv[i], self.kappa_det, num_samples=Nsamples
            )

            # to do this, we sample over all energies first with truncated gaussian
            E_samples = np.zeros(Nsamples)
            lnA_samples = np.zeros(Nsamples)

            for j in range(Nsamples):
                # sample energy from truncated lognormal distribution
                E_samples[j] = truncated_lognormal_sample(
                    mu=np.log(self.uhecr_energy[i]) + self.logE_sys,
                    sigma=self.logE_stat,
                    a=self.Eth,  # minimum energy in EeV
                    b=self.Eth_max,  # maximum energy in EeV
                )

                # # now compute mean and variance of lnA, as a function of log10(E / EeV)
                logE_idx = np.digitize(np.log(E_samples[j]), self.lnA_logE_grid, right=True)-1
                mean_lnA = truncnorm.rvs(
                    loc=self.mean_lnA_grid[logE_idx] + self.mean_lnA_sys,
                    scale=self.mean_lnA_stat[logE_idx],
                    a=0,
                    b=np.inf,
                    size=1
                )
                var_lnA = truncnorm.rvs(
                    loc=self.var_lnA_grid[logE_idx] + self.var_lnA_sys,
                    scale=self.var_lnA_stat[logE_idx],
                    a=-2,
                    b=np.inf,
                    size=1
                )
                # mean_lnA = self.mean_lnA_grid[logE_idx]
                # var_lnA = self.var_lnA_grid[logE_idx]

                # force non-negative variance
                var_lnA = max(var_lnA, 1e-12)

                # generate a single sampled lnA value from the normal distribution
                lnA_samples[j] = norm.rvs(
                    loc=mean_lnA, scale=np.sqrt(var_lnA)
                )

            # now compute the rigidities using R = (E / Z) * (Z /A) * (A / (exp(lnA)))
            uhecr_sampled_Rs = E_samples / (0.5 * np.exp(lnA_samples))  # in EV

            # calculate the mean rigidity for each UHECR for later use
            self.rigidities.append(uhecr_sampled_Rs)

            bt_args.append((i, uhecr_sampled_uvs, uhecr_sampled_Rs))

        self.rigidities = np.array(self.rigidities)
        return bt_args

    def _get_kappa_gmf(self: Self, uhecr_idx: int) -> float:
        """
        Get kappa_GMF for a given UHECR index.

        Parameters
        ----------
        uhecr_idx : int
            the UHECR index
        """
        # nested function for conviencience
        rng_kgmf = np.random.default_rng()

        fit = self.vMF_model.sample(
            data={
                "n": self.defl_sampled_uvs[uhecr_idx, :, :],  # deflected unit vectors
                "N": self.defl_sampled_uvs.shape[
                    1
                ],  # shape of the deflected unit vectors
                "mu": self.defl_mean_uvs[
                    uhecr_idx, :
                ],  # mean direction of deflected vectors
            },
            iter_warmup=1000,
            iter_sampling=2000,
            chains=2,  # fix number of chains since we dont need that many anyways
            seed=int(rng_kgmf.integers(low=1, high=10000)),
            show_progress=False,
        )

        return np.mean(fit.stan_variable("kappa"))

    def __f_theta_scalar(self: Self, kappa: float, P: float = 0.683) -> float:
        """
        Compute the Pth containment angle for a given kappa value.

        Parameters
        ----------
        kappa : float
            the kappa value
        P : float, default=0.683
            the containment probability. Default to 1 sigma.
        """
        if kappa <= 1e5 and kappa > 1e-3:
            return np.arccos(1 + np.log((1 - P * (1 - np.exp(-2 * kappa)))) / kappa)
        elif kappa > 1e5:
            return np.arccos(1 + np.log(1 - P) / kappa)
        elif kappa <= 1e-3:
            return np.arccos(1 + np.log(1 - 2 * P * kappa) / kappa)

    def f_theta(self: Self, kappa: np.ndarray, P: float = 0.683) -> np.ndarray:
        """
        Compute the Pth containment angle for a range of kappa values.

        Parameters
        ----------
        kappa : np.ndarray
            array of kappa values
        P : float, default=0.683
            the containment probability. Default to 1 sigma.
        """
        return np.vectorize(self.__f_theta_scalar)(kappa, P)

    def save(self: Self, outfile: str) -> None:
        """
        Save the result as a pickle file.

        Parameters
        ----------
        outfile : str
            the output file to save the results.
        """
        print(f"Saving results to {outfile}")
        pickle.dump(
            (
                self.kappa_gmfs,
                self.thetaPs,
                self.uhecr_coords_earth.galactic.l.deg,
                self.uhecr_coords_earth.galactic.b.deg,
                self.uhecr_coords_gb.galactic.l.deg,
                self.uhecr_coords_gb.galactic.b.deg,
                self.arr_sampled_uvs,
                self.defl_sampled_uvs,
                self.defl_mean_uvs,
                self.time_delays,
                np.sqrt(7552 / self.kappa_det),  # sigma_det in deg
                self.kappa_det,
                self.rigidities,
            ),
            open(outfile, "wb"),
            protocol=-1,
        )
