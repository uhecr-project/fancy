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

# directory with the pre-generated turbulent field realisations used by the batched
# backpropagation (see generate_turb_realisation_library). ~205 MB per realisation.
DEFAULT_TURB_LIBRARY_DIR = os.environ.get(
    "FANCY_GMF_TURB_LIBRARY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables", "turb_realisations"),
)

# random grids of CRPropa's JF12Field::randomStriated / randomTurbulent (inherited by
# UF23Field). Both are generated with unit strength and scaled on evaluation, so the
# same grids are valid for JF12 and all UF23*Turb models: (N per axis, spacing in pc)
_STRIATED_GRID = (100, 100.0)
_TURBULENT_GRID = (256, 4.0)


def _is_turbulent(gmf_model: str) -> bool:
    """GMF models whose field setup draws random striated / turbulent grids."""
    return gmf_model == "JF12" or gmf_model.find("Turb") != -1


def _set_omp_num_threads(nthreads: int) -> list:
    """
    Set the number of OpenMP threads for CRPropa's ModuleList.run(CandidateVector).

    numpy (MKL) loads Intel's libiomp5 next to libgomp, and the CRPropa omp_* calls
    resolve to whichever came first, so set it on every loaded OpenMP runtime.
    Returns the previous values, to pass to _restore_omp_num_threads.
    """
    import ctypes
    import re

    with open("/proc/self/maps") as f:
        paths = sorted({l.split()[-1] for l in f if re.search(r"lib(gomp|iomp5)[^/]*\.so", l)})
    libs = [ctypes.CDLL(p) for p in paths]
    prev = [(lib, lib.omp_get_max_threads()) for lib in libs]
    for lib in libs:
        lib.omp_set_num_threads(int(nthreads))
    return prev


def _restore_omp_num_threads(prev: list) -> None:
    for lib, n in prev:
        lib.omp_set_num_threads(n)


def _turb_library_index_file(gmf_model: str, library_dir: str = None) -> str:
    library_dir = DEFAULT_TURB_LIBRARY_DIR if library_dir is None else library_dir
    return os.path.join(library_dir, f"gmf_turb_realisations_{gmf_model}.pkl")


def generate_turb_realisation_library(
    gmf_model: str,
    n_realisations: int = 50,
    library_dir: str = None,
    seed: int = None,
    overwrite: bool = False,
) -> str:
    """
    Pre-generate `n_realisations` striated + turbulent field realisations, exactly as
    `GMFBackPropagation.__setup_simulation` draws them (randomStriated(s) and
    randomTurbulent(s) with the same seed s), and dump the grids with CRPropa's
    dumpGrid (raw float32) so that they can be loaded back with cr.loadGrid instead
    of regenerated.

    Writes `<library_dir>/gmf_turb_realisations_{gmf_model}/{striated,turbulent}_XXX.raw`
    and the index `<library_dir>/gmf_turb_realisations_{gmf_model}.pkl` (written last,
    so an interrupted run never leaves a usable but incomplete library).

    Returns the path of the index file.
    """
    assert _is_turbulent(gmf_model), f"{gmf_model} has no random field components."
    library_dir = DEFAULT_TURB_LIBRARY_DIR if library_dir is None else library_dir
    index_file = _turb_library_index_file(gmf_model, library_dir)
    if os.path.exists(index_file) and not overwrite:
        print(f"Turbulent realisation library already exists: {index_file}")
        return index_file

    grid_dir = os.path.join(library_dir, f"gmf_turb_realisations_{gmf_model}")
    os.makedirs(grid_dir, exist_ok=True)
    # keep the ~GB grids out of the fancy git repository
    with open(os.path.join(library_dir, ".gitignore"), "w") as f:
        f.write("*\n")

    rng = np.random.default_rng(seed)
    # seed 0 means "unseeded" in CRPropa, so draw from [1, 1e7)
    seeds = [int(s) for s in rng.choice(np.arange(1, 10_000_000), size=n_realisations, replace=False)]
    files = []
    for r, s in enumerate(tqdm(seeds, desc=f"Generating {n_realisations} turbulent realisations")):
        field = cr.JF12Field()  # UF23Field inherits these generators unchanged
        field.randomStriated(s)
        field.randomTurbulent(s)
        striated_file = f"striated_{r:03d}.raw"
        turbulent_file = f"turbulent_{r:03d}.raw"
        cr.dumpGrid(field.getStriatedGrid(), os.path.join(grid_dir, striated_file))
        cr.dumpGrid(field.getTurbulentGrid(), os.path.join(grid_dir, turbulent_file))
        files.append((striated_file, turbulent_file))
        del field

    index = {
        "gmf_model": gmf_model,
        "n_realisations": n_realisations,
        "seeds": seeds,
        "files": files,
        "grid_dir": os.path.basename(grid_dir),
        "striated_grid": _STRIATED_GRID,
        "turbulent_grid": _TURBULENT_GRID,
        "crpropa_version": getattr(cr, "__version__", None),
    }
    with open(index_file, "wb") as f:
        pickle.dump(index, f, protocol=-1)
    print(f"Wrote turbulent realisation library: {index_file}")
    return index_file


def load_turb_realisation_library(gmf_model: str, library_dir: str = None) -> dict:
    """Load the index of a turbulent realisation library (the grids are loaded lazily)."""
    index_file = _turb_library_index_file(gmf_model, library_dir)
    with open(index_file, "rb") as f:
        index = pickle.load(f)
    index["grid_dir"] = os.path.join(os.path.dirname(index_file), index["grid_dir"])
    return index


def load_turb_realisation(index: dict, r: int) -> tuple:
    """Load realisation `r` of a library as (striated Grid1f, turbulent Grid3f)."""
    Ns, ds = index["striated_grid"]
    Nt, dt = index["turbulent_grid"]
    striated_file, turbulent_file = (os.path.join(index["grid_dir"], f) for f in index["files"][r])
    # cr.loadGrid does not check the file length
    assert os.path.getsize(striated_file) == 4 * Ns**3, f"corrupt grid file {striated_file}"
    assert os.path.getsize(turbulent_file) == 12 * Nt**3, f"corrupt grid file {turbulent_file}"
    striated = cr.Grid1f(cr.Vector3d(0.0), Ns, ds * cr.parsec)
    turbulent = cr.Grid3f(cr.Vector3d(0.0), Nt, dt * cr.parsec)
    cr.loadGrid(striated, striated_file)
    cr.loadGrid(turbulent, turbulent_file)
    return striated, turbulent


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
        "KST24",
    ]  # type of GMF models
    __Nmodels_UF23: int = 8  # number of models in UF23

    __UF23_models : typing.ClassVar[dict] = {
        "base" : 0,
        "neCL" : 1,
        "expX" : 2,
        "spur" : 3,
        "cre10" : 4,
        "synCG" : 5,
        "twistX" : 6,
        "nebCor" : 7
    }

    # number of consecutive samples traced through one field realisation
    _SAMPLES_PER_FIELD: int = 50

    def __init__(
        self: Self,
        data: Data,
        gmf_model: str = "JF12",
        turb_library_dir: str = None,
        n_turb_realisations: int = 50,
    ) -> None:
        """
        Class to simulate back propagation of UHECRs within a given dataset (simulated or real data).

        Parameters
        ----------
        data : Data
            object generated from fancy.interfaces.data
        gmf_model : str
            the GMF model considered for backpropagation.
        turb_library_dir : str, optional
            directory of the pre-generated turbulent realisations used by the batched
            backpropagation of turbulent models. Defaults to DEFAULT_TURB_LIBRARY_DIR.
        n_turb_realisations : int, default=50
            number of realisations to generate if the library does not exist yet.
        """
        self.gmf_model = gmf_model
        self.turb_library_dir = turb_library_dir
        self.n_turb_realisations = n_turb_realisations
        self.turb_library = None

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

        # if "UF23" in gmf_model:
        #     uf23_model = gmf_model.replace("UF23", "").replace("Turb", "")
        #     assert uf23_model in self.__UF23_models.keys() or uf23_model == "all", (
        #         f"GMF model {gmf_model} is not an available UF23 model."
        #     )
        # else:
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
        # same model for many independent fits at once (see _fit_kappa_gmf_batch)
        self.vMF_batch_model = CmdStanModel(
            stan_file=str(get_path_to_stan_file("vMF", "fit_from_vMF_batch.stan")),
            model_name="vMF_batch",
            stanc_options=stanc_options,
        )

    def _get_turb_library(self: Self) -> dict:
        """Index of the turbulent realisation library, generated on first use if missing."""
        if self.turb_library is None:
            if not os.path.exists(_turb_library_index_file(self.gmf_model, self.turb_library_dir)):
                generate_turb_realisation_library(
                    self.gmf_model, self.n_turb_realisations, self.turb_library_dir
                )
            self.turb_library = load_turb_realisation_library(self.gmf_model, self.turb_library_dir)
        return self.turb_library

    def _assign_fields(self: Self, Nsamples: int, Nsets: int, rng: np.random.Generator) -> np.ndarray:
        """
        Field assignment for the samples of ONE event, traced as `Nsets` sets of
        `Nsamples` samples (one set per rigidity grid node, or a single set).

        As in run_single_backpropagation, a field is set up for every block of
        `_SAMPLES_PER_FIELD` consecutive samples, and the UF23 model number of a block
        is that of its first sample. For turbulent models, every block of the event
        gets a different library realisation (drawn without replacement per UF23
        model), as long as the library is large enough.

        Returns int array (Nsets, Nsamples, 2): (UF23 model number, realisation or -1).
        """
        k0 = (np.arange(Nsamples) // self._SAMPLES_PER_FIELD) * self._SAMPLES_PER_FIELD
        if self.gmf_model.find("UF23all") != -1:
            mt = k0 // (Nsamples // self.__Nmodels_UF23)
        else:
            mt = np.zeros(Nsamples, dtype=int)
        fields = np.empty((Nsets, Nsamples, 2), dtype=int)
        fields[..., 0] = mt
        fields[..., 1] = -1
        if not _is_turbulent(self.gmf_model):
            return fields

        n_real = self._get_turb_library()["n_realisations"]
        block = np.arange(Nsamples) // self._SAMPLES_PER_FIELD
        block_mt = mt[:: self._SAMPLES_PER_FIELD]
        for m in np.unique(block_mt):
            blocks = np.flatnonzero(block_mt == m)
            n_draw = Nsets * len(blocks)
            if n_draw > n_real and not getattr(self, "_warned_library_size", False):
                print(
                    f"Warning: {n_draw} field blocks per event and model but only {n_real} "
                    "turbulent realisations, so some blocks of an event share a realisation."
                )
                self._warned_library_size = True
            draw = np.concatenate(
                [rng.permutation(n_real) for _ in range(-(-n_draw // n_real))]
            )[:n_draw].reshape(Nsets, len(blocks))
            per_block = np.full((Nsets, len(block_mt)), -1)
            per_block[:, blocks] = draw
            sel = block_mt[block] == m
            fields[:, sel, 1] = per_block[:, block[sel]]
        return fields

    def _backprop_batched(
        self: Self,
        uvs: np.ndarray,
        Rs: np.ndarray,
        fields: np.ndarray,
        nthreads: int,
        chunk: int = 200_000,
    ) -> tuple:
        """
        Backtrack M (arrival direction, rigidity) samples, grouped by field: for each
        field (UF23 model, realisation), all of its samples -- across all events and
        rigidities -- are traced in one OpenMP-parallel CRPropa call
        (ModuleList.run(CandidateVector)). Each turbulent realisation is loaded from
        the library once and shared by the UF23 models that use it.

        Parameters
        ----------
        uvs : np.ndarray
            sampled arrival directions, shape (M, 3).
        Rs : np.ndarray
            sampled rigidities in EV, shape (M,).
        fields : np.ndarray
            (UF23 model number, realisation or -1) per sample, shape (M, 2).
        nthreads : int
            number of OpenMP threads.

        Returns
        -------
        deflected directions at the GB (M, 3) and time delays in years (M,).
        """
        defl_uvs = np.full((len(Rs), 3), np.nan)
        time_delays = np.full(len(Rs), np.nan)

        pos_earth = cr.Vector3d(-8.5, 0, 0) * cr.kpc
        pid = -cr.nucleusId(1, 1)  # protons, charge via rigidity, negative to backtrack
        obs = cr.Observer()
        obs.add(cr.ObserverSurface(cr.Sphere(cr.Vector3d(0), 20 * cr.kpc)))
        rng = np.random.default_rng()

        prev_threads = _set_omp_num_threads(nthreads)
        try:
            realisations = np.unique(fields[:, 1])
            for r in tqdm(realisations, desc=f"Backpropagating ({len(realisations)} field realisations, {nthreads} threads)"):
                grids = None if r < 0 else load_turb_realisation(self._get_turb_library(), r)
                in_r = fields[:, 1] == r
                for mt in np.unique(fields[in_r, 0]):
                    sim = self.__setup_simulation(obs, int(mt), grids=grids)
                    # CRPropa schedules static blocks of 100 candidates, so shuffle to
                    # spread the slow low-rigidity trajectories over the threads
                    idx = rng.permutation(np.flatnonzero(in_r & (fields[:, 0] == mt)))
                    for s in range(0, len(idx), chunk):
                        cands = [
                            cr.Candidate(cr.ParticleState(pid, Rs[k] * cr.EeV, pos_earth, cr.Vector3d(*uvs[k])))
                            for k in idx[s : s + chunk]
                        ]
                        cv = cr.CandidateVector()
                        for c in cands:
                            cv.push_back(cr.CandidateRefPtr(c))
                        sim.run(cv, False)  # no secondaries in pure B-field backtracking
                        for k, c in zip(idx[s : s + chunk], cands):
                            d = c.current.getDirection()
                            defl_uvs[k] = (d.x, d.y, d.z)
                            time_delays[k] = self.__get_time_delay(c, pos_earth)
                        del cv, cands
                    del sim
                del grids
        finally:
            _restore_omp_num_threads(prev_threads)

        return defl_uvs, time_delays

    @staticmethod
    def _mean_direction(defl_uvs: np.ndarray) -> np.ndarray:
        """Normalised mean of the deflected directions over axis -2, ignoring NaN vectors."""
        ok = np.all(np.isfinite(defl_uvs), axis=-1, keepdims=True)
        mean = np.where(ok, defl_uvs, 0.0).sum(axis=-2)
        return mean / np.linalg.norm(mean, axis=-1, keepdims=True)

    # parallelize for each UHECR
    def run_backpropagation(
        self: Self,
        Nsamples: int = 500,
        njobs: int = 4,
        parallel: bool = True,
        batched: bool = True,
    ) -> None:
        """
        Run backpropagation for all UHECRs.

        Parameters
        ----------
        Nsamples : int
            number of samples to generate for each UHECR.
        njobs : int, default=4
            number of jobs (joblib) or OpenMP threads (batched) to run in parallel.
            not used if parallel=False
        parallel : bool
            flag whether to run in parallel or not.
        batched : bool, default=True
            trace all samples of all UHECRs grouped by field realisation in OpenMP-parallel
            CRPropa calls (`_backprop_batched`), with turbulent realisations taken from the
            pre-generated library. If False, use the original per-UHECR joblib path, which
            draws a fresh turbulent realisation every 50 samples.
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

        if batched:
            rng = np.random.default_rng()
            self.arr_sampled_uvs = np.array([uvs for _, uvs, _ in bt_args])
            Rs = np.array([Rs for _, _, Rs in bt_args])
            self.realisation_ids = np.array(
                [self._assign_fields(Nsamples, 1, rng)[0] for _ in range(self.Nuhecrs)]
            )
            defl_uvs, time_delays = self._backprop_batched(
                self.arr_sampled_uvs.reshape(-1, 3),
                Rs.ravel(),
                self.realisation_ids.reshape(-1, 2),
                nthreads=njobs if parallel else 1,
            )
            self.defl_sampled_uvs = defl_uvs.reshape(self.Nuhecrs, Nsamples, 3)
            self.time_delays = time_delays.reshape(self.Nuhecrs, Nsamples)
            self.defl_mean_uvs = self._mean_direction(self.defl_sampled_uvs)
            results = []
        # use joblib to run parallel jobs otherwise use serial
        elif parallel:
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

    def compute_kappa_gmf(self: Self, njobs : int = 4, batched: bool = True) -> None:
        """
        Compute kappa gmf & theta by fitting to vMF distribution pre-computed via stan.

        If batched, all UHECRs are fitted in a few Stan runs (`_fit_kappa_gmf_batch`)
        instead of one Stan run per UHECR.
        """
        if batched:
            self.kappa_gmfs = self._fit_kappa_gmf_batch(self.defl_sampled_uvs, self.defl_mean_uvs, njobs=njobs)
        else:
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

    def run_single_backpropagation_fixed_seed(self: Self, bt_arg: tuple, seed: int) -> tuple:
        """
        Same as `run_single_backpropagation`, but forces every field-setup
        call (every 50 samples, same cadence as the production path) to
        reuse the SAME explicit turbulent/striated field seed, instead of
        drawing fresh OS entropy each time.

        Test-only method for the turbulence-correlation diagnostic in
        new_uhecr_model/lnA_sensitivity/gmf_backpropagation/ -- not used by
        any production code path. Kept fully separate from
        `run_single_backpropagation` so the validated production behaviour
        is untouched by this addition.

        Parameters
        ----------
        bt_arg : tuple
            tuple containing the UHECR index, sampled arrival directions and sampled rigidities.
        seed : int
            explicit field seed, reused across every field-setup call in this run.

        Returns
        -------
        Same return shape as `run_single_backpropagation`.
        """
        uhecr_idx, uhecr_uvs, uhecr_Rs = bt_arg
        uhecr_defl_uvs = np.zeros_like(uhecr_uvs)
        uhecr_time_delays = np.zeros(uhecr_uvs.shape[0])

        pos_earth = cr.Vector3d(-8.5, 0, 0) * cr.kpc
        pid = -cr.nucleusId(1, 1)

        obs = cr.Observer()
        obs.add(cr.ObserverSurface(cr.Sphere(cr.Vector3d(0), 20 * cr.kpc)))

        uhecr_mean_defl_v3d = cr.Vector3d(0.0, 0.0, 0.0)

        for k, uhecr_uv in enumerate(uhecr_uvs):
            if k % 50 == 0:
                mt_num = int(np.floor(k / (len(uhecr_uvs) // self.__Nmodels_UF23)))
                sim = self.__setup_simulation(obs, mt_num, seed=seed)

            uhecr_vector3d = cr.Vector3d(*uhecr_uv)

            c = cr.Candidate(
                cr.ParticleState(pid, uhecr_Rs[k] * cr.EeV, pos_earth, uhecr_vector3d)
            )
            sim.run(c)

            uhecr_time_delays[k] = self.__get_time_delay(c, pos_earth)
            uhecr_defl_v3d = c.current.getDirection()

            uhecr_defl_uvs[k, :] = np.array(
                [uhecr_defl_v3d.x, uhecr_defl_v3d.y, uhecr_defl_v3d.z]
            )

            if np.any(np.isnan(uhecr_defl_v3d)):
                continue

            uhecr_mean_defl_v3d += uhecr_defl_v3d

        uhecr_mean_defl_v3d /= len(uhecr_Rs)
        uhecr_mean_defl_uv = np.array(
            [uhecr_mean_defl_v3d.x, uhecr_mean_defl_v3d.y, uhecr_mean_defl_v3d.z]
        )
        uhecr_mean_defl_uv /= np.linalg.norm(uhecr_mean_defl_uv)

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

    def __setup_simulation(self: Self, obs, mt_num: int, seed: int = None, grids: tuple = None):
        """
        Prepare the crpropa backtracking simulation.

        Parameters
        ----------
        obs : cr.Observer
            CRPropa observer object
        mt_num : int
            the montel number for the UF23 model
        seed : int, optional
            explicit seed for the turbulent/striated field realization. If
            None (default), a fresh seed is drawn from OS entropy every call,
            exactly as before -- this parameter is purely additive and does
            not change existing behaviour when omitted. Only needed to force
            multiple calls to reuse the SAME field realization (e.g. for the
            turbulence-correlation test in new_uhecr_model/lnA_sensitivity/).
        grids : tuple, optional
            pre-generated (striated Grid1f, turbulent Grid3f) realisation, set on the
            field instead of drawing new random grids (see load_turb_realisation).

        Returns
        -------
        cr.ModuleList
            the simulation object containing the observer and propagation model
        """
        sim = cr.ModuleList()
        rng = np.random.default_rng()

        def _resolve_seed():
            return seed if seed is not None else int(rng.integers(low=0, high=10000000))

        def _add_random_fields(gmf_cr):
            if grids is not None:
                gmf_cr.setStriatedGrid(grids[0])
                gmf_cr.setTurbulentGrid(grids[1])
            else:
                field_seed = _resolve_seed()
                gmf_cr.randomStriated(field_seed)
                gmf_cr.randomTurbulent(field_seed)

        # setup magnetic field
        if self.gmf_model == "JF12":
            gmf_cr = cr.JF12Field()
            _add_random_fields(gmf_cr)

        elif self.gmf_model == "UF23all":
            gmf_cr = cr.UF23Field(mt_num)

        elif self.gmf_model == "UF23allTurb":
            gmf_cr = cr.UF23Field(mt_num)
            _add_random_fields(gmf_cr)

        elif self.gmf_model.find("UF23") != -1:
            uf23_model = self.gmf_model.replace("UF23", "").replace("Turb", "")
            gmf_cr = cr.UF23Field(self.__UF23_models[uf23_model])

            if self.gmf_model.find("Turb") != -1:
                _add_random_fields(gmf_cr)

        elif self.gmf_model == "UF23baseTurb":
            gmf_cr = cr.UF23Field(0)
            _add_random_fields(gmf_cr)

        elif self.gmf_model == "PT11":
            gmf_cr = cr.PT11Field()
        elif self.gmf_model == "TF17":
            gmf_cr = cr.TF17Field()
        elif self.gmf_model == "KST24":
            gmf_cr = cr.KST24Field()
        else:
            raise NotImplementedError(
                f"GMF model {self.gmf_model} is not implemented yet."
            )

        # Propagation model, parameters: (B-field model, target error, min step, max step)
        sim.add(cr.PropagationCK(gmf_cr, 1e-3, 0.1 * cr.parsec, 100 * cr.parsec))

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
        return self._fit_kappa_gmf(
            self.defl_sampled_uvs[uhecr_idx, :, :],
            self.defl_mean_uvs[uhecr_idx, :],
        )

    def _fit_kappa_gmf(self: Self, defl_uvs: np.ndarray, defl_mean_uv: np.ndarray) -> float:
        """
        Fit kappa_GMF to a vMF distribution given deflected unit vectors and
        their mean direction directly (no dependence on `self.defl_sampled_uvs`
        / `self.defl_mean_uvs`), so this is safe to call from parallel workers
        indexed by anything other than a plain UHECR index (e.g. (event,
        rigidity) pairs in `RigidityResolvedGMFBackPropagation`).

        Parameters
        ----------
        defl_uvs : np.ndarray
            deflected unit vectors, shape (Nsamples, 3)
        defl_mean_uv : np.ndarray
            mean direction of the deflected unit vectors, shape (3,)
        """
        # nested function for conviencience
        rng_kgmf = np.random.default_rng()

        fit = self.vMF_model.sample(
            data={
                "n": defl_uvs,  # deflected unit vectors
                "N": defl_uvs.shape[0],  # shape of the deflected unit vectors
                "mu": defl_mean_uv,  # mean direction of deflected vectors
            },
            iter_warmup=1000,
            iter_sampling=2000,
            chains=2,  # fix number of chains since we dont need that many anyways
            seed=int(rng_kgmf.integers(low=1, high=10000)),
            show_progress=False,
        )

        return np.mean(fit.stan_variable("kappa"))

    def _fit_kappa_vmf_chunk(self: Self, N: np.ndarray, sum_cos: np.ndarray) -> np.ndarray:
        """Posterior mean kappa of len(N) independent fits in one Stan run."""
        rng_kgmf = np.random.default_rng()
        fit = self.vMF_batch_model.sample(
            data={"M": len(N), "N": N, "sum_cos": sum_cos},
            iter_warmup=1000,
            iter_sampling=2000,
            chains=2,  # same sampler settings as _fit_kappa_gmf
            seed=int(rng_kgmf.integers(low=1, high=10000)),
            show_progress=False,
        )
        return np.mean(fit.stan_variable("kappa"), axis=0)

    def _fit_kappa_gmf_batch(
        self: Self, defl_uvs: np.ndarray, defl_mean_uvs: np.ndarray, njobs: int = 4, chunk: int = 2000
    ) -> np.ndarray:
        """
        Same posterior mean kappa_GMF as `_fit_kappa_gmf` for many fits at once.

        The vMF likelihood of fit_from_vMF.stan depends on the samples only through
        the number of samples N and sum_i dot(n_i, mu), so fit_from_vMF_batch.stan
        loops over the fits with these sufficient statistics. The fits are split into
        chunks of `chunk` (to keep the Stan output small), run in parallel.

        Parameters
        ----------
        defl_uvs : np.ndarray
            deflected unit vectors, shape (..., Nsamples, 3)
        defl_mean_uvs : np.ndarray
            mean direction of each fit, shape (..., 3)

        Returns
        -------
        kappa_GMF, shape (...)
        """
        shape = defl_mean_uvs.shape[:-1]
        sum_cos = np.einsum("...ij,...j->...", defl_uvs, defl_mean_uvs).reshape(-1)
        N = np.full(sum_cos.shape, defl_uvs.shape[-2], dtype=int)
        starts = range(0, len(N), chunk)
        kappas = ParallelPbar(f"Calculating kappa_GMF ({len(N)} fits, {len(starts)} Stan runs): ")(
            n_jobs=min(njobs, len(starts))
        )(delayed(self._fit_kappa_vmf_chunk)(N[s : s + chunk], sum_cos[s : s + chunk]) for s in starts)
        return np.concatenate(kappas).reshape(shape)

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


class RigidityResolvedGMFBackPropagation(GMFBackPropagation):
    """
    GMFBackPropagation subclass that additionally computes kappa_GMF on a
    fixed rigidity grid (instead of only the rigidity-marginalised kappa_GMF
    from run_backpropagation/compute_kappa_gmf).

    The marginalised omega_det / kappa_GMF computed by the parent class are
    left completely untouched -- this only adds `kappa_gmf_grid` and
    `R_grid`, meant to be interpolated against an event's latent rigidity
    inside the Stan fit, for direct comparison against the marginalised
    kappa_GMF.
    """

    # rigidity grid (in EV) validated via new_uhecr_model/lnA_sensitivity/gmf_backpropagation:
    # log-spaced, densified below R=20 EV where kappa_GMF(R) and the interpolation
    # error both vary fastest.
    R_GRID_8NODE = np.array(
        [2.0, 2.517, 3.169, 3.988, 5.02, 7.96, 12.62, 20.0]
    )
    # 14 log-spaced nodes (2026-10-09): the mean deflected direction moves by
    # ~1 bubble width kappa_GMF(R)^-1/2 per node above 5 EV (2-3 widths with
    # R_GRID_8NODE), so a natural spline of omega_shift_grid / ln kappa_GMF(R)
    # interpolates to within the run-to-run (turbulent realisation) noise
    DEFAULT_R_GRID = np.geomspace(2.0, 20.0, 14)

    @staticmethod
    def omega_shift_table(omega: np.ndarray, R_means: np.ndarray) -> np.ndarray:
        """
        Tangent vectors at omega (N, 3) that carry it to the mean deflected
        direction at each rigidity node, R_means (N, Nr, 3): the logarithmic map
        v = theta * unit(m - cos(theta) omega), |v| = theta in radians. The Stan
        model rotates omega back along the interpolated v (exponential map).

        Returns (N, Nr, 3).
        """
        c = np.clip(np.einsum("irk,ik->ir", R_means, omega), -1.0, 1.0)
        perp = R_means - c[..., None] * omega[:, None, :]
        norm = np.linalg.norm(perp, axis=-1, keepdims=True)
        return np.arccos(c)[..., None] * np.where(norm > 0, perp / np.maximum(norm, 1e-300), 0.0)

    def run_backpropagation_rigidity_grid(
        self: Self,
        R_grid: np.ndarray = None,
        Nsamples_per_R: int = 300,
        njobs: int = 4,
        batched: bool = True,
        centre_on_marginal: bool = False,
    ) -> None:
        """
        Backpropagate at each fixed rigidity in R_grid (no lnA/rigidity
        marginalisation) and fit kappa_GMF at each grid point.

        Populates `self.R_grid` (shape [Nr]) and `self.kappa_gmf_grid`
        (shape [Nuhecrs, Nr]).

        Both the CRPropa backpropagation and the vMF kappa_GMF fit are
        dispatched as one flat pool of (event, rigidity) jobs each, instead
        of Nr sequential per-rigidity passes -- this is a pure scheduling
        change (better load-balancing across the fixed njobs worker pool,
        one parallel dispatch instead of Nr back-to-back ones) and does not
        share any CRPropa field/turbulence realization across rigidity grid
        points: every (event, rigidity) job still calls
        `run_single_backpropagation` independently, exactly as before, so
        the per-job physics/statistics are unchanged (see project memory --
        field-sharing across rigidities was tested separately and rejected
        for introducing real bias; this flattening is scheduling-only and
        does not revisit that).

        Parameters
        ----------
        R_grid : np.ndarray, optional
            fixed rigidities (in EV) to backpropagate at. Defaults to
            DEFAULT_R_GRID.
        Nsamples_per_R : int, default=300
            number of backpropagation samples per event per rigidity grid
            point (must be >=8 for UF23-family models).
        njobs : int, default=4
            number of parallel jobs for backpropagation (OpenMP threads if batched).
        batched : bool, default=True
            trace all (event, rigidity, sample) candidates grouped by field realisation
            in OpenMP-parallel CRPropa calls, with turbulent realisations from the
            pre-generated library (see `run_backpropagation`). Every block of 50 samples
            of an event still gets its own realisation, so no realisation is shared
            between the rigidity nodes of an event. The kappa_GMF(R) fits are done in a
            few Stan runs over all (event, rigidity) pairs (`_fit_kappa_gmf_batch`).
        centre_on_marginal : bool, default=False
            fit kappa_GMF(R) about the rigidity-marginalised mean direction
            (`self.defl_mean_uvs`, i.e. omega_det, from `run_backpropagation`)
            instead of about the mean direction at each rigidity, so that the
            offset of the rigidity-R deflections from omega_det widens kappa_GMF(R).
        """
        if centre_on_marginal:
            assert getattr(self, "defl_mean_uvs", None) is not None and len(self.defl_mean_uvs) == self.Nuhecrs, (
                "centre_on_marginal requires run_backpropagation first (omega_det)."
            )
        if R_grid is None:
            R_grid = self.DEFAULT_R_GRID
        R_grid = np.asarray(R_grid, dtype=float)

        Nsamples = Nsamples_per_R
        if self.gmf_model.find("UF23all") != -1:
            Nsamples = int(Nsamples * 8)  # number of models in UF23
        assert Nsamples >= 8, (
            "Nsamples_per_R must be >=8 for UF23-family models (mt_num division)."
        )

        Nr = len(R_grid)
        self.R_grid = R_grid
        self.kappa_gmf_grid = np.zeros((self.Nuhecrs, Nr))

        # --- build one flat list of (event, rigidity) backprop jobs ---
        # bt_args entries are still (i, uhecr_sampled_uvs, uhecr_fixed_Rs)
        # tuples, exactly what run_single_backpropagation expects; we just
        # additionally remember which (i, j) each job belongs to so results
        # can be scattered into grid-shaped buffers below.
        bt_args = []
        ij_index = []
        for j, R_fixed in enumerate(R_grid):
            for i in range(self.Nuhecrs):
                uhecr_sampled_uvs = sample_vMF(
                    self.uhecr_uv[i], self.kappa_det, num_samples=Nsamples
                )
                uhecr_fixed_Rs = np.full(Nsamples, R_fixed)
                bt_args.append((i, uhecr_sampled_uvs, uhecr_fixed_Rs))
                ij_index.append((i, j))

        if batched:
            rng = np.random.default_rng()
            uvs = np.empty((self.Nuhecrs, Nr, Nsamples, 3))
            for (i, j), (_, uhecr_sampled_uvs, _) in zip(ij_index, bt_args):
                uvs[i, j] = uhecr_sampled_uvs
            Rs = np.broadcast_to(R_grid[None, :, None], (self.Nuhecrs, Nr, Nsamples))
            self.realisation_ids_grid = np.array(
                [self._assign_fields(Nsamples, Nr, rng) for _ in range(self.Nuhecrs)]
            )
            defl_uvs, _ = self._backprop_batched(
                uvs.reshape(-1, 3), Rs.ravel(), self.realisation_ids_grid.reshape(-1, 2), nthreads=njobs
            )
            defl_uvs = defl_uvs.reshape(self.Nuhecrs, Nr, Nsamples, 3)
            defl_means = self._mean_direction(defl_uvs)
            results = [
                (i, uvs[i, j], defl_uvs[i, j], defl_means[i, j], None) for (i, j) in ij_index
            ]
        else:
            # --- flat parallel dispatch over all Nuhecrs * Nr backprop jobs ---
            results = ParallelPbar(
                f"Backpropagating on rigidity grid ({self.Nuhecrs} events x {Nr} rigidities): "
            )(n_jobs=njobs)(delayed(self.run_single_backpropagation)(arg) for arg in bt_args)

        # grid-shaped buffers (event, rigidity) instead of the single
        # per-pass scratch buffers the sequential-per-rigidity version reused
        # -- needed because results for all Nr grid points now coexist.
        defl_sampled_uvs_grid = np.zeros((self.Nuhecrs, Nr, Nsamples, 3))
        defl_mean_uvs_grid = np.zeros((self.Nuhecrs, Nr, 3))
        for (i, j), (uhecr_idx, ars, dls, dlm, td) in zip(ij_index, results):
            assert uhecr_idx == i  # bt_args order matches ij_index order 1:1
            nan_mask = np.isnan(dls)
            if nan_mask.any():
                a_sample_with_nonans = dls[~np.any(nan_mask, axis=1)][0:1]
                dls = np.where(nan_mask, a_sample_with_nonans, dls)
            defl_sampled_uvs_grid[i, j, ...] = dls
            defl_mean_uvs_grid[i, j, :] = dlm

        # kept for diagnostics (e.g. offsets between the rigidity-R and marginalised centres)
        self.defl_mean_uvs_grid = defl_mean_uvs_grid.copy()
        self.defl_sampled_uvs_grid = defl_sampled_uvs_grid
        if centre_on_marginal:
            defl_mean_uvs_grid[:] = self.defl_mean_uvs[:, None, :]

        # --- flat parallel dispatch over all Nuhecrs * Nr kappa_GMF fits ---
        # (this step was previously a serial `for i in range(self.Nuhecrs)`
        # loop repeated once per rigidity grid point -- unparallelized,
        # unlike the marginalised-path compute_kappa_gmf. _fit_kappa_gmf
        # takes its inputs directly rather than reading self.defl_sampled_uvs
        # / self.defl_mean_uvs by index, so this is safe under concurrent
        # (i, j) jobs.)
        if batched:
            self.kappa_gmf_grid = self._fit_kappa_gmf_batch(
                defl_sampled_uvs_grid, defl_mean_uvs_grid, njobs=njobs
            )
            return
        kappa_results = ParallelPbar(
            f"Calculating kappa_GMF on rigidity grid ({self.Nuhecrs} events x {Nr} rigidities): "
        )(n_jobs=njobs)(
            delayed(self._fit_kappa_gmf)(
                defl_sampled_uvs_grid[i, j, ...], defl_mean_uvs_grid[i, j, :]
            )
            for i in range(self.Nuhecrs)
            for j in range(Nr)
        )
        for (i, j), kappa in zip(
            ((i, j) for i in range(self.Nuhecrs) for j in range(Nr)), kappa_results
        ):
            self.kappa_gmf_grid[i, j] = kappa
