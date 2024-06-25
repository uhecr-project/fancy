"""Container for composition weights"""

import os
import numpy as np
import pickle as pickle
from tqdm import tqdm
import h5py

from joblib import delayed, Parallel


from astropy.cosmology import WMAP9, z_at_value, Planck18
import astropy.units as u
from scipy.interpolate import UnivariateSpline

try:
    import prince_cr as pcr
    import prince_cr.config
    from prince_cr import core, photonfields, cross_sections
    from prince_cr import util as pru
    from prince_cr.solvers import UHECRPropagationSolverBDF
    from prince_cr.cr_sources import CosmicRaySource
except ImportError:
    pcr = None


class CompositionMatrixContainer:
    """
    Container to handle all composition loss computation performed by Prince. In principle this only needs to be accessed if one wants to re-compute the composition weights.
    """

    prince_cr.config.x_cut = 1e-4
    prince_cr.config.x_cut_proton = 1e-2
    prince_cr.config.tau_dec_threshold = np.inf
    prince_cr.config.linear_algebra_backend = "MKL"
    prince_cr.config.secondaries = False
    prince_cr.config.ignore_particles = [
        0,
        11,
        12,
        13,
        14,
        15,
        16,
        20,
        21,
    ]
    prince_cr.config.cosmic_ray_grid = (8, 12, 40)

    # photon fields, combined CMB & EBL from Gilmore
    pf_gilmore = photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )

    # ID of nuclei used for injection, use all available mass ids possible
    massids = [
        101,
        201,
        302,
        402,
        904,
        1005,
        1105,
        1206,
        1306,
        1407,
        1507,
        1608,
        1708,
        1808,
        1909,
        2010,
        2110,
        2210,
        2311,
        2412,
        2512,
        2612,
        2713,
        2814,
        2914,
        3014,
        3115,
        3216,
        3316,
        3416,
        3517,
        3618,
        3718,
        3818,
        3919,
        4019,
        4119,
        4220,
        4320,
        4420,
        4521,
        4622,
        4722,
        4822,
        4922,
        5023,
        5123,
        5224,
        5324,
        5424,
        5525,
        5626,
    ]

    def __init__(
        self,
        css="PSB",
        dmin=0.8,
        dmax=110,
        Nds=100,
        resources_path: str = os.path.dirname(os.path.realpath(__file__)),
        nthreads=8,
    ):
        """
        Container to handle all composition loss computation performed by Prince. In principle this only needs to be accessed if one wants to re-compute the composition weights.

        :param css: cross section model used for prince computation. Valid models are ["TALYS", "PSB"]
        :param dmin,dmax,Nds: minimum / maximum distance and density for distance grid in Mpc
        :param resources_path: path where resources (kernel, redshift_distance interpolator) is stored
        :param nthreads: number of threads used for solver
        """
        self.css = css
        self.dmax = dmax
        self.distances = np.linspace(dmin, dmax, Nds)
        self.resources_path = resources_path

        if pcr == None:
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
            self._create_kernel()

        self.prince_run = pickle.load(open(self.prince_run_datapath, "rb"))

        # similarly get the converstion from z <-> d
        self.redshift_distance_datapath = os.path.join(
            self.resources_path, "redshift_tables_Planck_WMAP.pkl"
        )

        if not os.path.exists(self.redshift_distance_datapath):
            print("Pre-computing redshift <-> distance table")
            self._create_distance_tables()

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

    def run_single_injection_solver(self, run_args):
        """Wrapper for paralleilisation of injection solver"""
        dinit_idx, dinit, zinit, prince_run, massids = run_args
        print(f"solving for dinit={dinit:.3f} Mpc")
        results_per_dinit = []

        for massid in tqdm(massids):
            # initialise solver
            solver = UniformInjectionSolver(
                initial_z=zinit,
                final_z=0.0,
                prince_run=prince_run,
                enable_pairprod_losses=True,
                enable_adiabatic_losses=True,
                enable_injection_jacobian=False,
                enable_partial_diff_jacobian=True,
            )

            solver.add_source_class(
                NoInjection(prince_run, params={})
            )  # no source model
            solver.set_initial_state(
                massid, zinit
            )  # set uniform injection as initial state
            solver.solve(
                dz=min(zinit / 200, 3e-4), verbose=False, progressbar=False
            )  # solve
            # append for each mass
            results_per_dinit.append(
                (
                    solver.res,
                    solver.initial_state[solver.spec_man.ncoid2sref[massid].sl],
                )
            )

        return (dinit_idx, results_per_dinit)

    def run_injection_solver(self, njobs=4, reset=False):
        """
        Run injection solver. It will try to find the `sol_injection_solver.pkl` and load from it. otherwise
        it will compute it.

        :param njobs: number of cpus used for distance parallelisation
        :param reset: to reset the pre-computation or not
        """

        solver_res_path = os.path.join(self.resources_path, "sol_injection_solver.pkl")
        if not os.path.exists(solver_res_path) or reset == True:
            print("Computing the injection solver")

            # create linearly spaced grid for distances
            redshifts = self.spl_d_to_z_plk(self.distances)

            # iterate for each maximal distance <-> redshift & injection mass
            run_args = [
                (i, self.distances[i], redshifts[i], self.prince_run, self.massids)
                for i in range(len(self.distances))
            ]

            solver_res = [self.run_single_injection_solver(arg) for arg in run_args]

            # currently parallelising over distances is not working too well, maybe because its contained within a class...
            # but using many threads can be as fast as parallelising over this so in principle its not necessary
            # solver_res = Parallel(n_jobs=njobs)(delayed(self.run_single_injection_solver)(arg) for arg in run_args)

            solver_res.sort(key=lambda x: x[0])  # sort based on dinit index

            # write to pickle file
            pickle.dump(solver_res, open(solver_res_path, "wb"), protocol=-1)
        else:
            print("Using pre-computed solver. set reset=True to re-compute")
            solver_res = pickle.load(open(solver_res_path, "rb"))

        return solver_res

    def determine_mass_groups(self):
        """Determine the masses within each mass group"""

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

    def compute_weights(self, solver_res):
        """
        Compute composition weights, both the full version and also summed over each mass group

        :param solver_res: results from the solver
        """

        self.weights = np.zeros(
            (
                len(self.distances),
                len(self.massids),
                len(self.massids),
                len(self.prince_run.cr_grid.grid),
            )
        )

        # iterate for each distance & source mass
        for id in range(len(self.distances)):
            _, solver_res_per_d = solver_res[id]

            for ims in range(len(self.massids)):

                res, src_spect = solver_res_per_d[ims]

                for ime, massid_earth in enumerate(self.massids):
                    # get arrival spectrum for each arrival mass, as function of E/A
                    _, earth_spect = res.get_solution(massid_earth)

                    # sometimes the earth spectrum yields weird artefacts where its < 0.
                    # in this case, we just constrain it to some limiting value
                    earth_spect[earth_spect < 0] = 1e-100

                    # weights is the fraction of lost particles from the particular source mass
                    # at particular distance, as function of E/A
                    self.weights[id, ims, ime, :] = earth_spect / src_spect

        self.weights_mg = np.zeros(
            (
                len(self.distances),
                len(self.massids),
                4,
                len(self.prince_run.cr_grid.grid),
            )
        )

        self.determine_mass_groups()

        for img, (mg_lidx, mg_uidx) in enumerate(self.mass_group_idxlims):
            self.weights_mg[..., img, :] = np.sum(
                self.weights[..., mg_lidx : mg_uidx + 1, :], axis=2
            )

    def save(self, outfile):
        """Save data into h5py format"""

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

            weights_full = f.create_group("full")
            weights_full.create_dataset("weights", data=self.weights)

            weights_massgroup = f.create_group("mass_groups")
            weights_massgroup.create_dataset("mass_groups", data=[1, 2, 3, 4])
            weights_massgroup.create_dataset(
                "mass_group_idxlims", data=np.array(self.mass_group_idxlims, dtype=int)
            )
            weights_massgroup.create_dataset("weights_mg", data=self.weights_mg)

    def _create_kernel(self):
        """Create kernel and save the results if not yet done so"""

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

    def _create_distance_tables(self):
        """Create conversion table from redshift to Mpc if not yet done so"""
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
