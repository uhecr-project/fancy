"""Class to determine backpropagated events from a given distribution of UHECRs with a particular detector"""

import os
import numpy as np
from astropy.coordinates import SkyCoord
from fancy import Data

from cmdstanpy import CmdStanModel
from joblib import Parallel, delayed
from vMF import sample_vMF
import pickle

try:
    import crpropa as cr
except ImportError:
    cr = None


class GMFBackPropagation:
    """
    Class to simulate back propagation of UHECRs within a given dataset (simulated or real data) and obtain the deflected events and their individual kappa values.

    :param data : Data object generated from fancy.interfaces.data
    :param gmf_model : the GMF model considered for backpropagation.

    """

    def __init__(self, data: Data, gmf_model="JF12"):
        """
        Class to simulate back propagation of UHECRs within a given dataset (simulated or real data) and obtain the deflected events and their individual kappa values.

        :param data : Data object generated from fancy.interfaces.data
        :param gmf_model : the GMF model considered for backpropagation.

        """
        self.gmf_model = gmf_model

        # raise exception if CRPropa is not installed, since it requires CRPropa
        if cr == None:
            raise ImportError(
                "CRPropa is not installed and is required for using this module."
            )

        # uhecr direction properties
        self.uhecr_uv = (data.uhecr.coord.cartesian.xyz.value).T
        self.uhecr_coords_earth = SkyCoord(
            self.uhecr_uv, frame="galactic", representation_type="cartesian"
        )
        self.uhecr_coords_earth.representation_type = "unitspherical"

        # uhecr rigidity properties
        self.uhecr_rigidity = (
            data.uhecr.rigidity
            if len(data.uhecr.rigidity) > 0
            else data.uhecr.energy / data.detector.meanZ
        )
        self.Nuhecrs = len(self.uhecr_rigidity)

        # detector properties
        self.deltaR = data.detector.rigidity_uncertainty
        self.deltaAng = data.detector.coord_uncertainty
        self.kappa_d = data.detector.kappa_d

        # compile vMF model
        self._compile_vMFmodel()

    def _compile_vMFmodel(self):
        # model to fit vMF with
        stan_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "stan")
        fit_filename = os.path.join(stan_path, "fit_from_vMF.stan")
        stanc_options = {"include-paths": stan_path}

        self.vMF_model = CmdStanModel(
            stan_file=fit_filename,
            model_name="vMF",
            stanc_options=stanc_options,
        )

    def _get_time_delay(self, c, pos_earth):
        """Returns delay between entering the galactic disc
        and arrival at Earth through magnetic field."""
        return (
            (c.getTrajectoryLength() - c.current.getPosition().getDistanceTo(pos_earth))
            / cr.c_light
            / (60 * 60 * 24 * 365)
        )

    def _setup_simulation(self, obs):
        """Setup crpropa backtracking simulation"""
        sim = cr.ModuleList()

        # setup magnetic field
        if self.gmf_model == "JF12":
            seed = np.random.randint(10000000)
            gmf_cr = cr.JF12Field()
            gmf_cr.randomStriated(seed)
            gmf_cr.randomTurbulent(seed)
        elif self.gmf_model == "PT11":
            gmf_cr = cr.PT11Field()
        elif self.gmf_model == "TF17":
            gmf_cr = cr.TF17Field()

        # Propagation model, parameters: (B-field model, target error, min step, max step)
        sim.add(cr.PropagationCK(gmf_cr, 1e-4, 0.1 * cr.parsec, 100 * cr.parsec))

        sim.add(obs)  # add observer at galactic boundary
        return sim

    def run_single_backpropagation(self, bt_arg):
        """Wrapper function for parallelising backtracking simulation"""

        uhecr_idx, uhecr_uvs, uhecr_Rs = bt_arg
        uhecr_defl_uvs = np.zeros_like(uhecr_uvs)
        uhecr_time_delays = np.zeros(uhecr_uvs.shape[0])

        print(f"Current UHECR index: {uhecr_idx}")

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
                sim = self._setup_simulation(obs)

            # get crropa Vector3D version of sampled arrival directions
            uhecr_vector3d = cr.Vector3d(*uhecr_uv)

            c = cr.Candidate(
                cr.ParticleState(pid, uhecr_Rs[k], pos_earth, uhecr_vector3d)
            )
            sim.run(c)

            uhecr_time_delays[k] = self._get_time_delay(c, pos_earth)
            uhecr_defl_v3d = c.current.getDirection()

            # store sampled deflected directions
            uhecr_defl_uvs[k, :] = np.array(
                [uhecr_defl_v3d.x, uhecr_defl_v3d.y, uhecr_defl_v3d.z]
            )

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

    def _generate_backtracking_arguments(self, Nsamples=500):
        """Generate arguments used for backtracking"""
        # generate arguments
        bt_args = []
        for i in range(self.Nuhecrs):

            # sample arrival directions via vMF
            uhecr_sampled_uvs = sample_vMF(
                self.uhecr_uv[i], self.kappa_d, num_samples=Nsamples
            )

            # sample rigidity via normal distribution
            # rigidity uncertainty == energy uncertainty (TODO: add composition uncertainty too)
            uhecr_sampled_Rs = (
                np.random.normal(
                    loc=self.uhecr_rigidity[i],
                    scale=self.deltaR * self.uhecr_rigidity[i],
                    size=Nsamples,
                )
                * cr.EeV
            )

            bt_args.append((i, uhecr_sampled_uvs, uhecr_sampled_Rs))
        return bt_args

    # parallelize for each UHECR
    def run_backpropagation(self, Nsamples=500, njobs=4, parallel=True):
        """Run backpropagation for all UHECRs"""

        self.arr_sampled_uvs = np.zeros((self.Nuhecrs, Nsamples, 3))
        self.defl_sampled_uvs = np.zeros((self.Nuhecrs, Nsamples, 3))
        self.defl_mean_uvs = np.zeros((self.Nuhecrs, 3))
        self.time_delays = np.zeros((self.Nuhecrs, Nsamples))

        # generate backtrakcing arguments for all uhecrs
        bt_args = self._generate_backtracking_arguments(Nsamples)

        # use joblib to run parallel jobs otherwise use serial
        if parallel:
            results = Parallel(n_jobs=njobs)(
                delayed(self.run_single_backpropagation)(arg) for arg in bt_args
            )
        else:
            results = []
            for arg in bt_args:
                results.append(self.run_single_backpropagation(arg))

        # append the results
        for uhecr_idx, ars, dls, dlm, td in results:
            self.arr_sampled_uvs[uhecr_idx, ...] = ars
            self.defl_sampled_uvs[uhecr_idx, ...] = dls
            self.defl_mean_uvs[uhecr_idx, :] = dlm
            self.time_delays[uhecr_idx, :] = td

        self.uhecr_coords_gb = SkyCoord(
            self.defl_mean_uvs, frame="galactic", representation_type="cartesian"
        )
        self.uhecr_coords_gb.representation_type = "unitspherical"

    def compute_kappa_gmf(self):
        """Compute kappa gmf by fitting to vMF distribution"""
        self.kappa_gmfs = Parallel(n_jobs=2)(
            delayed(self._get_kappa_gmf)(uhecr_idx) for uhecr_idx in range(self.Nuhecrs)
        )
        self.thetaPs = self.f_theta(self.kappa_gmfs)  # for plotting purposes

    def _get_kappa_gmf(self, uhecr_idx):
        # nested function for conviencience
        rng_kgmf = np.random.default_rng()

        fit = self.vMF_model.sample(
            data={
                "n": self.defl_sampled_uvs[uhecr_idx, :, :],
                "N": self.defl_sampled_uvs.shape[1],
                "mu": self.defl_mean_uvs[uhecr_idx, :],
            },
            iter_warmup=1000,
            iter_sampling=2000,
            chains=2,  # fix number of chains since we dont need that many anyways
            seed=int(rng_kgmf.integers(low=1, high=10000)),
            show_progress=False,
        )

        return np.mean(fit.stan_variable("kappa"))

    def _f_theta_scalar(self, kappa, P=0.683):
        """Returns costheta"""
        if kappa <= 1e5 and kappa > 1e-3:
            return np.arccos(1 + np.log((1 - P * (1 - np.exp(-2 * kappa)))) / kappa)
        elif kappa > 1e5:
            return np.arccos(1 + np.log(1 - P) / kappa)
        elif kappa <= 1e-3:
            return np.arccos(1 + np.log(1 - 2 * P * kappa) / kappa)

    def f_theta(self, kappa, P=0.683):
        """vectorized version"""
        return np.vectorize(self._f_theta_scalar)(kappa, P)

    def save(self, outfile):
        """Save the result"""
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
                self.deltaAng,
                self.kappa_d,
                self.deltaR,
            ),
            open(outfile, "wb"),
            protocol=-1,
        )
