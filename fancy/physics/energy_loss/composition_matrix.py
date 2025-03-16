"""Container for composition weights"""

import os
import pickle as pickle

import astropy.units as u
import h5py
import numpy as np
from astropy.cosmology import WMAP9, Planck18, z_at_value
from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline
from scipy.optimize import Bounds, minimize
from tqdm import tqdm
from typing_extensions import Self  # change to typing for py>3.11

try:
    import prince_cr as pcr
    import prince_cr.config
    from prince_cr import core, cross_sections, photonfields
    from prince_cr import util as pru
    from prince_cr.cr_sources import CosmicRaySource
    from prince_cr.solvers import UHECRPropagationSolverBDF
except ImportError:
    pcr = None


class CompositionMatrixContainer:
    """
    Container to handle all composition loss computation performed by Prince.

    In principle this only needs to be accessed if one wants to re-compute the composition weights.
    """

    prince_cr.config.x_cut = 1e-4
    prince_cr.config.x_cut_proton = 1e-2
    prince_cr.config.tau_dec_threshold = np.inf
    prince_cr.config.linear_algebra_backend = "MKL"
    prince_cr.config.secondaries = False  # here we explicitly exclude secondaries
    prince_cr.config.ignore_particles = [  # noqa: RUF012
        0,
        11,
        12,
        13,
        14,
        15,
        16,
        20,
        21,
    ]  # as well as ignoring all secondary particles
    prince_cr.config.cosmic_ray_grid = (8, 12, 40)  # fixed to e/A from 1e8 to 1e12 eV

    # photon fields, combined CMB & EBL from Gilmore
    pf_gilmore = photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )

    # ID of nuclei used for injection, use all available mass ids possible
    massids = [  # noqa: RUF012
        101,
        201,
        301,
        302,
        402,
        603,
        703,
        704,
        904,
        1004,
        1005,
        1105,
        1206,
        1306,
        1406,
        1407,
        1507,
        1608,
        1708,
        1808,
        1909,
        2010,
        2110,
        2210,
        2211,
        2311,
        2412,
        2512,
        2612,
        2613,
        2713,
        2814,
        2914,
        3014,
        3115,
        3216,
        3316,
        3416,
        3516,
        3617,
        3718,
        3818,
        3919,
        4020,
        4120,
        4220,
        4320,
        4422,
        4521,
        4622,
        4722,
        4822,
        4923,
        5024,
        5124,
        5224,
        5325,
        5426,
        5526,
        5626,
    ]

    def __init__(
        self: Self,
        css: str = "PSB",
        dmin: float = 0.8,
        dmax: float = 110,
        Nds: int = 100,
        resources_path: str = os.path.dirname(os.path.realpath(__file__)),
        nthreads: int = 8,
    ) -> None:
        """
        Container to handle all composition loss computation performed by Prince.

        In principle this only needs to be accessed if one wants to re-compute the composition weights.

        Parameter
        ---------
        css: str
            cross section model used for prince computation. Valid models are ["TALYS", "PSB"]
        dmin : float
            minimum distance for distance grid in Mpc
        dmax : float
            maximum distance for distance grid in Mpc
        NDs : int
            number of grid points for distance grid
        resources_path: str
            path where resources (kernel, redshift_distance interpolator) is stored
        nthreads: int
            number of threads used for solver
        """
        self.css = css
        self.dmax = dmax
        self.distances = np.linspace(dmin, dmax, Nds)
        self.resources_path = resources_path

        if pcr is None:
            raise ImportError("Prince-CR needs to be installed for using this module!")

        # create directory if it doesnt exist yet
        if not os.path.exists(self.resources_path):
            os.mkdir(self.resources_path)

        # set mkl threads
        prince_cr.config.set_mkl_threads(nthreads)

        # get the kernel
        self.prince_run_datapath = os.path.join(
            self.resources_path, f"prince_run_{css}_mkl.ppo"
        )

        if not os.path.exists(self.prince_run_datapath):
            print("Pre-computing kernel")
            self.__create_kernel()

        self.prince_run = pickle.load(open(self.prince_run_datapath, "rb"))

        # similarly get the converstion from z <-> d
        self.redshift_distance_datapath = os.path.join(
            self.resources_path, "redshift_tables_Planck_WMAP.pkl"
        )

        if not os.path.exists(self.redshift_distance_datapath):
            print("Pre-computing redshift <-> distance table")
            self.__create_distance_tables()

        (_, self.spl_d_to_z_plk, _, _) = pickle.load(
            open(self.redshift_distance_datapath, "rb")
        )

        # initialise solver once to get utility functions to compute A and Z from massIDs
        solv = UHECRPropagationSolverBDF(
            initial_z=1.0, final_z=0.0, prince_run=self.prince_run
        )
        self.fA = lambda x: solv.spec_man.ncoid2sref[x].A
        self.fZ = lambda x: solv.spec_man.ncoid2sref[x].Z

        # A and Z
        self.As = np.array([self.fA(massid) for massid in self.massids])
        self.Zs = np.array([self.fZ(massid) for massid in self.massids])

        # genearte objects that we store later
        self.propa_matrix = None
        self.inj_eff_matrix = None
        self.source_mass_pdf = None

    def run_injection_solver(self: Self, reset: bool = False):
        """
        Run the injection solver.

        It will try to find the `sol_injection_solver.pkl` and load from it. Otherwise it will compute it.

        Parameter
        ----------
        reset: bool
            flag to reset the pre-computation or not.
            If True, then the matrix will be computed again.
        """
        solver_res_path = os.path.join(self.resources_path, "injection_solver")
        if not os.path.exists(solver_res_path):
            os.mkdir(solver_res_path)

        solver_res_files = [
            os.path.join(solver_res_path, f"sol_injection_solver_D{dinit:.2f}.pkl")
            for dinit in self.distances
        ]

        if reset:
            print("Computing the injection solver")

            # create linearly spaced grid for distances
            redshifts = self.spl_d_to_z_plk(self.distances)

            for i, dinit in enumerate(self.distances):
                # there is still some memory issue that needs to be solved over here...
                # very temporary and bad hack below for now
                # if dinit < 86:
                #     continue

                # initialise theinjection solver helper
                solver = InjectionSolverRunContainer(
                    self.distances[i], redshifts[i], self.massids, self.prince_run
                )
                solver_res_single = solver.run()
                # write to pickle file
                pickle.dump(
                    solver_res_single, open(solver_res_files[i], "wb"), protocol=-1
                )

                # delete the solver helper to avoid any memory issues
                del solver

            # currently parallelising over distances is not working too well, maybe because its contained within a class...
            # but using many threads can be as fast as parallelising over this so in principle its not necessary
            # TODO: investigate this further if necessary?
            # solver_res = Parallel(n_jobs=njobs)(delayed(self.run_single_injection_solver)(arg) for arg in run_args)

        else:
            print(
                "Using pre-computed solver results. Set reset == True to re-run the injection solver."
            )
            assert all([os.path.exists(f) for f in solver_res_files]), (
                "Files dont exist. Please run injection solver with reset=True."
            )

        solver_res = [pickle.load(open(f, "rb")) for f in solver_res_files]

        return solver_res

    def determine_mass_groups(self: Self) -> None:
        """Determine the masses within each mass group."""
        # first define the masses within each mass group
        mass_groups = [1, 2, 3, 4]
        mass_group_ids = []
        self.mass_group_idxlims = []

        for lnA_lower, lnA_upper in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            id_per_mg = []

            for im, massid in enumerate(self.massids):
                # compute A
                lnA = np.log(self.fA(massid))

                # geeq and lessthan sign as with Dembinski+2017
                if lnA >= lnA_lower and lnA < lnA_upper:
                    id_per_mg.append(massid)

            mass_group_ids.append(id_per_mg)

            # also calcualte the lower and upper limit in massid array
            # for computation convenience later
            mg_lower_idx = np.digitize(id_per_mg[0], self.massids, right=True)
            mg_upper_idx = np.digitize(id_per_mg[-1], self.massids, right=True)

            self.mass_group_idxlims.append([mg_lower_idx, mg_upper_idx])

            print(f"Masses (A) contained in mass group {lnA_upper}: ")
            print([self.fA(mid) for mid in id_per_mg])

    def compute_propagation_matrices(self: Self, solver_res: list) -> None:
        """
        Compute propagation matrices (weights) from the results from prince.

        Returns the fraction of UHECRs at Earth for each rigidity, source mass,
        distance, and arrival mass.

        Parameter
        ---------
        solver_res: list
            results from the solver
        """
        # define rigidity grid number here
        NRs = len(self.prince_run.cr_grid.grid)

        # propagation matrix resulting from propagation
        # this is defined for each arrival mass as well
        # shape is DIS x ASRC x AEARTH x RIGIDITY
        self.propa_matrix = np.zeros(
            (
                len(self.distances),
                len(self.massids),
                len(self.massids),
                NRs,
            )
        )

        # iterate for each distance & source mass
        for id in tqdm(
            range(len(self.distances)),
            desc="Iterating over all distances: ",
            total=len(self.distances),
        ):
            solver_res_per_d = solver_res[id]

            # computing the source and earth spectrum from prince
            for ims in range(len(self.massids)):
                res, src_spect = solver_res_per_d[ims]

                # get the earth spectrum.
                # we still store it for all arrival masses to
                # verify if our production efficiency produces
                # MG3 particles well later
                for ime, mid in enumerate(self.massids):
                    self.propa_matrix[id, ims, ime, :] = (
                        res.get_solution(mid)[1] / src_spect
                    )

    def compute_injection_matrices(
        self: Self,
        solver_res: list,
        damp_factor: float = 0.001,
        njobs: int = 4,
    ) -> None:
        """
        Compute injection matrices (weights) from the results from prince.

        This is done by optimising the cost function that maximises the
        production of each arrival mass from each source mass.

        We iterate over each distance and rigidity for the computation.

        Parameter
        ---------
        solver_res: list
            results from the solver
        damp_factor: float, default = 0.001
            The dampening factor to diminish the strong over-fitting for
            exclusion of other arrival masses, as this contribution is
            much stronger than the inclusion of the particular arrival mass.
            Default is 0.001, which is determined after playing around.
        n_cores : int, defualt = 4
            Number of cores used for parallelisation over distances.
            Note: each calculation takes ~ 10 GB of RAM per distance.
        """
        # prepare arguments to input in the parallelisation
        opt_args = [
            (idis, solver_res[idis], damp_factor) for idis in range(len(self.distances))
        ]

        # parallelise over each distance
        opt_results = Parallel(n_jobs=njobs)(
            delayed(self._run_single_optimisation)(arg) for arg in opt_args
        )

        # injection matrix resulting from optimisation of earth & source spectrum
        # this is defined for each arrival mass as well
        # shape is DIS x ASRC x AEARTH x RIGIDITY
        self.inj_eff_matrix = np.zeros(
            (
                len(self.distances),
                len(self.massids),
                len(self.massids),
                len(self.prince_run.cr_grid.grid),
            )
        )

        for idis, opt_res in opt_results:
            self.inj_eff_matrix[idis, ...] = opt_res

    def _run_single_optimisation(self: Self, args: tuple) -> np.ndarray:
        """
        Run single optimisation step for the cost function.

        Parameters
        ----------
        args: tuple
            dis_idx: int
                index of distance
            solver_res_per_d: list
                results from the solver for each source mass
            damp_factor: float
                dampening factor for the cost function
        """
        dis_idx, solver_res_per_d, damp_factor = args
        print(f"Current distance index: {dis_idx}")

        inj_eff_mat = np.zeros(
            (
                len(self.massids),
                len(self.massids),
                len(self.prince_run.cr_grid.grid),
            )
        )

        for ime, me in enumerate(self.massids):
            # store the earth spectrum & source spectrum to be used for optimisation
            earth_spects = np.zeros(
                (len(self.massids), len(self.prince_run.cr_grid.grid))
            )
            src_spects = np.zeros(
                (len(self.massids), len(self.prince_run.cr_grid.grid))
            )

            for ims in range(len(self.massids)):
                res, src_spect = solver_res_per_d[ims]
                src_spects[ims, :] = src_spect
                # get the earth spectrum at the mass group per source mass for all rigidity bins
                earth_spects[ims, :] = res.get_solution(me)[1]

            # set lower limit to avoid numerical issues
            src_spects[src_spects < 1e-60] = 1e-60
            earth_spects[earth_spects < 1e-60] = 1e-60

            # now we optimise over source masses, so loop over rigidities
            for iR in range(len(self.prince_run.cr_grid.grid)):
                # initial guess
                wA0s = np.ones(len(self.massids))

                # optimisation
                res = minimize(
                    cost_function,
                    x0=wA0s,
                    args=(
                        earth_spects[:, iR],
                        src_spects[:, iR],
                        ime,
                        damp_factor,
                        1.0,  # fixed for now, could be variable in future
                    ),
                    bounds=Bounds(0, 1),
                    method="L-BFGS-B",
                )

                inj_eff_mat[ime, :, iR] = res.x

        return dis_idx, inj_eff_mat

    def compute_source_mass_PDF(self: Self) -> None:
        """
        Compute the source mass PDF.

        This is done by normalising the injection efficiency matrix.
        """
        norma = np.sum(self.inj_eff_matrix, axis=1, keepdims=True)
        self.source_mass_pdf = self.inj_eff_matrix / norma

        # some limiters to avoid numerical issues
        self.source_mass_pdf[np.isnan(self.source_mass_pdf)] = 1e-40
        self.source_mass_pdf[self.source_mass_pdf < 1e-40] = 1e-40

    def save(self: Self, outfile: str) -> None:
        """
        Save data into h5py format.

        Parameters
        ----------
        outfile: str
            output file name
        """
        # compute rigidity grid, assuming constant mass-to-charge ratio
        # NB: rigidities are in GV!
        A_per_Z = 2  # assume constant A/Z ratio
        rigidities = self.prince_run.cr_grid.grid * A_per_Z
        rigidities_widths = np.diff(self.prince_run.cr_grid.bins) * A_per_Z

        with h5py.File(outfile, "w") as f:
            f.create_dataset("distances", data=self.distances)
            f.create_dataset("massids", data=self.massids)
            f.create_dataset("As", data=self.As)
            f.create_dataset("Zs", data=self.Zs)
            f.create_dataset("en_per_nucs", data=self.prince_run.cr_grid.grid)
            f.create_dataset("rigidities", data=rigidities)
            f.create_dataset("rigidities_widths", data=rigidities_widths)
            f.create_dataset("propa_matrix", data=self.propa_matrix)
            f.create_dataset("inj_eff_matrix", data=self.inj_eff_matrix)
            f.create_dataset("source_mass_pdf", data=self.source_mass_pdf)

    def __create_kernel(self: Self) -> None:
        """Create kernel and save the results if not yet done so."""
        if os.path.exists(self.prince_run_datapath):
            print("File already exists, no need for re-computation")
            return

        # cross section class, either use TALYS or PSB
        cs = cross_sections.CompositeCrossSection(
            [
                (0.0, cross_sections.TabulatedCrossSection, (self.css,)),
                (0.14, cross_sections.SophiaSuperposition, ()),
            ]
        )

        # generate kernel
        prince_run = core.PriNCeRun(
            max_mass=56, photon_field=self.pf_gilmore, cross_sections=cs
        )

        # pickle dump the results
        pickle.dump(prince_run, open(self.prince_run_datapath, "wb"), protocol=-1)

    def __create_distance_tables(self: Self) -> None:
        """Create conversion table from redshift to Mpc if not yet done so."""
        if os.path.exists(self.redshift_distance_datapath):
            print("File already exists, no need for re-computation")
            return

        distance_grid_mpc = np.logspace(-1, np.log10(5000), 1000)

        redshift_grid_plk = [
            z_at_value(Planck18.comoving_distance, d * u.Mpc) for d in distance_grid_mpc
        ]
        redshift_grid_wmap = [
            z_at_value(WMAP9.comoving_distance, d * u.Mpc) for d in distance_grid_mpc
        ]

        # Fix the zeros
        redshift_grid_plk.insert(0, 0.0)
        redshift_grid_wmap.insert(0, 0.0)
        distance_grid_mpc = np.hstack([[0], distance_grid_mpc])

        # computing both z-> d and d -> z
        spl_z_to_d_plk = UnivariateSpline(
            redshift_grid_plk, distance_grid_mpc, s=0, k=2
        )
        spl_d_to_z_plk = UnivariateSpline(
            distance_grid_mpc, redshift_grid_plk, s=0, k=2
        )
        spl_z_to_d_wmap = UnivariateSpline(
            redshift_grid_wmap, distance_grid_mpc, s=0, k=2
        )
        spl_d_to_z_wmap = UnivariateSpline(
            distance_grid_mpc, redshift_grid_wmap, s=0, k=2
        )

        # dump
        pickle.dump(
            (spl_z_to_d_plk, spl_d_to_z_plk, spl_z_to_d_wmap, spl_d_to_z_wmap),
            open(self.redshift_distance_datapath, "wb"),
        )


class InjectionSolverRunContainer:
    """Helper class to instantiate and generate a single run for each distance."""

    def __init__(
        self, dinit: float, zinit: float, massids: list, prince_run: core.PriNCeRun
    ) -> None:
        """Initialise the run container.

        Parameters
        ----------
        dinit : float
            the distance of the source in Mpc
        """
        self.dinit = dinit
        self.zinit = zinit
        self.massids = massids
        self.prince_run = prince_run

    def run(self):
        """Wrapper for paralleilisation of injection solver."""
        print(f"solving for dinit={self.dinit:.3f} Mpc")
        results_per_dinit = []

        for massid in tqdm(self.massids):
            # initialise solver
            solver = UniformInjectionSolver(
                initial_z=self.zinit,
                final_z=0.0,
                prince_run=self.prince_run,
                enable_pairprod_losses=True,
                enable_adiabatic_losses=True,
                enable_injection_jacobian=False,
                enable_partial_diff_jacobian=True,
            )

            solver.add_source_class(
                NoInjection(self.prince_run, params={})
            )  # no source model
            solver.set_initial_state(
                massid, self.zinit
            )  # set uniform injection as initial state
            solver.solve(
                dz=min(self.zinit / 200, 3e-4), verbose=False, progressbar=False
            )  # solve
            # append for each mass
            results_per_dinit.append(
                (
                    solver.res,
                    solver.initial_state[solver.spec_man.ncoid2sref[massid].sl],
                )
            )

        return results_per_dinit


class UniformInjectionSolver(UHECRPropagationSolverBDF):
    def _init_solver(self, dz):
        # print("_init_")
        # initial_state = np.zeros(self.dim_states)

        self._update_jacobian(self.initial_z)
        self.current_z_rates = self.initial_z

        # find the maximum injection and reduce the system by this
        self.red_idx = self.initial_state.max()

        # Convert csr_matrix from GPU to scipy
        try:
            sparsity = self.had_int_rates.get_hadr_jacobian(self.initial_z, 1.0).get()
        except AttributeError:
            sparsity = self.had_int_rates.get_hadr_jacobian(self.initial_z, 1.0)

        from prince_cr.util import PrinceBDF

        self.r = PrinceBDF(
            self.eqn_derivative,
            self.initial_z,
            self.initial_state,
            self.final_z,
            max_step=np.abs(dz),
            atol=self.atol,
            rtol=self.rtol,
            #  jac = self.eqn_jac,
            jac_sparsity=sparsity,
            vectorized=True,
        )

    def get_available_ncoids(self):
        """Get all available massids in this spectrum"""
        return list(self.spec_man.ncoid2sref.keys())

    def set_initial_state(self, nco_id, initial_z):
        import warnings

        ewidths = np.diff(self.ebins)
        self.initial_state = np.zeros(self.dim_states)
        inj_spec = self.spec_man.ncoid2sref[nco_id]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.initial_state[inj_spec.sl] = 1.0

        pru.info(
            2,
            "Normalization: {0:5.3e}".format(
                np.sum(self.initial_state[inj_spec.sl] * ewidths)
            ),
        )
        pru.info(
            2,
            f"Redshift: {initial_z:5.3e}",
        )

        # self._update_jacobian(initial_z)
        self.initial_z = initial_z
        self.current_z_rates = self.initial_z


class NoInjection(CosmicRaySource):
    """Zero injection class to solve autonomous equation."""

    def injection_spectrum(self, pid, energy, params):
        return np.zeros_like(energy)


def cost_function(wAs: np.ndarray, *args: tuple) -> np.ndarray:
    """
    Cost function to minimise weights per distance per rigidity for each mass group.

    The cost function is based on the MSE function in machine learning. But instead
    of subtracting the two values, we minimise the weights that optimise the production
    by multiplying by the weights with the exclusion of other arrival masses and
    multiplying by (1 - weights) for the production of the arrival mass we want.

    The logarithm is taken for stability.

    Parameters
    ----------
    wAs: np.ndarray
        weights that are optimised for maximal production for each arrival mass
    args: tuple
        earth_spect: np.ndarray
            earth spectrum
        src_spect: np.ndarray
            source spectrum
        ime: int
            index of arrival mass
        lmbda_1: float
            weight for production
        lmbda_2: float
            weight for survival
    """
    earth_spect, src_spect, ime, lmbda_1, lmbda_2 = args

    Lsrc = src_spect * wAs
    Learth = earth_spect[..., ime]
    Learth_ex = np.sum(np.delete(earth_spect, ime, axis=-1), axis=-1)

    return np.log10(np.mean(lmbda_1 * Lsrc * Learth_ex + lmbda_2 * (1 - Lsrc) * Learth))
