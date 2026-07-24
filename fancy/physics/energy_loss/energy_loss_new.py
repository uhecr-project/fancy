"""Container for composition weights"""

import os
import pickle as pickle

import astropy.units as u
import h5py
import numpy as np
from tqdm import tqdm
from typing_extensions import (
    Self,
    Union,
    Tuple,
    Optional,
)  # change to typing for py>3.11

from fancy.interfaces.source import Source
from fancy.utils.package_data import (
    get_path_to_prince_config,
    get_path_to_injection_solvers,
)
from fancy.utils.helpers import source_spectrum

try:
    import prince_cr as pcr
    import prince_cr.config
    from prince_cr import core, cross_sections, photonfields
    from prince_cr import util as pru
    from prince_cr.cr_sources import CosmicRaySource
    from prince_cr.solvers import UHECRPropagationSolverBDF

    from .prince_helpers import (
        SingleInjectionSolver,
        TruncatedPropagationSolver,
        NoInjection,
        TruncatedInjectionSource,
        create_distance_tables,
        create_kernel,
    )
except ImportError:
    pcr = None


class EnergyLossModel:
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

    def __init__(
        self: Self,
        css: str = "CRP2_TALYS",
        alpha_grid: tuple = (-2.5, 5, 0.5),
        massids: list = [101, 402, 1407, 2814, 5626],
        nthreads: int = 8,
    ) -> None:
        """
        Container to handle all composition loss computation performed by Prince.

        In principle this only needs to be accessed if one wants to re-compute the composition weights.

        Parameter
        ---------
        css: str
            cross section model used for prince computation. Valid models are ["CRP2_TALYS", "PSB"]
        alpha_grid: tuple[float, float, float]
            grid of alpha values to use for the injection solver.
            The grid is defined as (alpha_min, alpha_max, alpha_step).
        massids: list[int]
            list of massids to use for the injection solver.
            The massids are defined using the usual convention in prince.
            Default is [101(H), 402(He), 1407(N), 2814(Ni), 5626(Fe)].
        resources_path: str
            path where resources (kernel, redshift_distance interpolator) is stored
        nthreads: int
            number of threads used for solver
        """
        self.css = css
        self.alphas = np.arange(
            alpha_grid[0], alpha_grid[1] + alpha_grid[2], alpha_grid[2]
        )
        self.massids = massids

        if pcr is None:
            raise ImportError("Prince-CR needs to be installed for using this module!")

        # set mkl threads, not needed if not actually computing
        prince_cr.config.set_mkl_threads(nthreads)

        # get the kernel
        if not os.path.exists(get_path_to_prince_config(f"prince_run_{css}.pkl")):
            print("Pre-computing kernel")
            create_kernel(css, get_path_to_prince_config(f"prince_run_{css}.pkl"))

        self.prince_run = pickle.load(
            open(get_path_to_prince_config(f"prince_run_{css}.pkl"), "rb")
        )

        # similarly get the converstion from z <-> d
        if not os.path.exists(
            get_path_to_prince_config("redshift_tables_Planck_WMAP.pkl")
        ):
            print("Pre-computing redshift <-> distance table")
            create_distance_tables(
                get_path_to_prince_config("redshift_tables_Planck_WMAP.pkl")
            )

        (_, self.spl_d_to_z_plk, _, _) = pickle.load(
            open(get_path_to_prince_config("redshift_tables_Planck_WMAP.pkl"), "rb")
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

        # other parameters that we set on the way
        self.distances = None
        self.dmins = None
        self.solvers_loaded = False

    def load_injection_solvers(
        self: Self,
        src_inj_config: dict = {
            "dinits": [4],
            "Rmax": 1.7,
        },
        bg_inj_config: dict = {
            "z_max": 3.0,
            "source_evo": "SFR",
            "Rmax": 1.7,
        },
    ) -> None:
        """
        Load the injection solvers.

        It will try to find the `sol_injection_solver.pkl` and load from it.
        Otherwise it will complain that the file does not exist, and you should
        run the `run_source_injection_solver` and `run_background_injection_solver`
        methods to compute the injection solvers.

        Parameters
        ----------
        src_inj_config : dict, default={'dinits': 'M82', 'Rmax': 1.7}
            configuration for the source injection solver.
            - dinits: list[float] or str, initial distances to inject from.
            - Rmax: float, maximal rigidity set in the source spectrum in EV.
        bg_inj_config : dict, default={'z_max': 3.0, 'source_evo': 'SFR', 'Rmax': 1.7}
            configuration for the background injection solver.
            - z_max: float, maximum redshift to start from.
            - source_evo: str, source evolution model to use.
            - Rmax: float, maximal rigidity set in the source spectrum in EV.
        """
        if type(src_inj_config["dinits"]) is str:
            source = Source()
            source.load_from_data_file(label=src_inj_config["dinits"])
            self.distances = source.distance
            source_solver_file = get_path_to_injection_solvers(
                f"src_injection_solver_{src_inj_config['dinits']}_Rmax{src_inj_config['Rmax']:.1f}.pkl"
            )

        elif type(src_inj_config["dinits"]) is list:
            self.distances = src_inj_config["dinits"]
            source_solver_file = get_path_to_injection_solvers(
                f"src_injection_solver_D{min(src_inj_config['dinits'])}_{max(src_inj_config['dinits'])}_Rmax{src_inj_config['Rmax']:.1f}.pkl"
            )
        else:
            raise ValueError("dinits must be either a list of floats or a string.")

        with open(source_solver_file, "rb") as f:
            solver_res_dict = pickle.load(f)
            self.solver_res_src = solver_res_dict["results"]
            self.distances = solver_res_dict["dinits"]
            assert np.all(solver_res_dict["alphas"] == self.alphas), (
                "alphas do not match!"
            )

            # do some sorting with the given mass ids and the
            # mass ids from the solver results
            if not np.all(np.sort(solver_res_dict["massids"]) == np.sort(self.massids)):
                for mid in self.massids:
                    if mid not in solver_res_dict["massids"]:
                        raise ValueError(f"massid {mid} not in solver results!")
                self.solver_res_src = [
                    [  # for each distance
                        [  # for each massid
                            solver_res_dict["results"][idist][
                                solver_res_dict["massids"].index(mid)
                            ][ia]
                            for ia in range(len(self.alphas))
                        ]
                        for mid in self.massids
                    ]
                    for idist in range(len(self.distances))
                ]
            # print(f"Loaded from {source_solver_file}")

        # load the background injection solver here too
        bg_solver_file = get_path_to_injection_solvers(
            f"bg_injection_solver_zmax{bg_inj_config['z_max']}_Rmax{bg_inj_config['Rmax']:.1f}_{bg_inj_config['source_evo']}.pkl"
        )
        with open(bg_solver_file, "rb") as f:
            solver_res_dict = pickle.load(f)
            self.solver_res_bg = solver_res_dict["results"]
            self.dmins = solver_res_dict["d_mins"]
            assert np.all(solver_res_dict["alphas"] == self.alphas), (
                "alphas do not match!"
            )

            # do some sorting with the given mass ids and the
            # mass ids from the solver results
            if not np.all(np.sort(solver_res_dict["massids"]) == np.sort(self.massids)):
                for mid in self.massids:
                    if mid not in solver_res_dict["massids"]:
                        raise ValueError(f"massid {mid} not in solver results!")
                self.solver_res_bg = [
                    [  # for each distance
                        [  # for each massid
                            solver_res_dict["results"][idist][
                                solver_res_dict["massids"].index(mid)
                            ][ia]
                            for ia in range(len(self.alphas))
                        ]
                        for mid in self.massids
                    ]
                    for idist in range(len(self.distances))
                ]
            # print(f"Loaded from {source_solver_file}")
            # print(f"Loaded {bg_solver_file}")

        self.solvers_loaded = True

    def compute_spectrum_and_lnA(
        self: Self,
        egrid: np.ndarray,
        egrid_widths: np.ndarray,
        egrid_lnA: np.ndarray,
        compute_src: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        """
        Compute the spectrum and lnA given an energy grid for energies and for lnA parameters.

        The energy grid must be provided in EV, as it is converted to GeV internally.

        Parameter
        ---------
        egrid : np.ndarray
            energy grid used for energy spectra in EV
        egrid_widths : np.ndarray
            spacing between energy bins in EV
        egrid_lnA : np.ndarray
            energy grid used for mean & var lnA in EV
        compute_src : bool, default=True
            whether to compute the source spectra as well.
            Default is True.
            If False, then only the background spectra will be computed.
            This is useful if one wants to compute only the background spectra
            for a given minimum distance.
        """
        espects = np.zeros(
            (
                len(egrid),
                len(self.alphas),
                len(self.massids),
                len(self.distances) + 1,
            )
        )
        lnA_params = np.zeros(
            (
                2,
                len(egrid_lnA),
                len(self.alphas),
                len(self.massids),
                len(self.distances) + 1,
            )
        )

        if compute_src:
            espects_src = np.zeros_like(espects)

        # choose the right background model to use based on its minimum distance
        dbg_idx = np.digitize(np.max(self.distances), self.dmins, right=True)
        print(f"Using background model with dmin={self.dmins[dbg_idx]:.2f} Mpc")
        solver_res_bg = self.solver_res_bg[dbg_idx]

        egrid_GeV = egrid * 1e9
        egrid_lnA_GeV = egrid_lnA * 1e9

        for ims, ia in np.ndindex((len(self.massids), len(self.alphas))):
            # iterate over all distances for sources
            for idis in range(len(self.distances)):
                res = self.solver_res_src[idis][ims][ia]
                # NB: internal conversion to GeV
                _, spect_from_src = res.get_solution_group(
                    "CR", egrid=egrid_GeV, epow=0
                )
                _, lnA_params[0, :, ia, ims, idis], lnA_params[1, :, ia, ims, idis] = (
                    res.get_lnA("CR", egrid=egrid_lnA_GeV)
                )

                espects[:, ia, ims, idis] = spect_from_src / np.sum(
                    spect_from_src * egrid_widths
                )

                if compute_src:
                    src_spect = source_spectrum(
                        egrid,
                        alpha=self.alphas[ia],
                        charge=self.Zs[ims],
                        Rmax=1.7,  # NB: here forced for now, in EV
                    )
                    espects_src[:, ia, ims, idis] = src_spect / np.sum(
                        src_spect * egrid_widths
                    )

            res, _ = solver_res_bg[ims][ia]
            # NB: internal conversion to GeV
            _, spect_from_bg = res.get_solution_group("CR", egrid=egrid_GeV, epow=0)
            _, lnA_params[0, :, ia, ims, -1], lnA_params[1, :, ia, ims, -1] = (
                res.get_lnA("CR", egrid=egrid_lnA_GeV)
            )

            espects[:, ia, ims, -1] = spect_from_bg / np.sum(
                spect_from_bg * egrid_widths
            )

        if compute_src:
            return espects, lnA_params, espects_src
        else:
            return espects, lnA_params

    def run_source_injection_solver(
        self: Self,
        dinits: Union[list, str] = "M82",
        Rmax: float = 1.7,  # in EV
    ) -> None:
        """
        Run the injection solver.

        It will try to find the `sol_injection_solver.pkl` and load from it. Otherwise it will compute it.

        Parameter
        ----------
        dinits : list[float] or str, default=M82
            the initial distances to inject from.
            Default is 3.8 Mpc (M82), but we can input as many as we wish.

            If string, then it will use a pre-defined set of distances within sourcedata.h5.
        Rmax : float, default=1.7  EV
            the maximal rigidity set in the source spectrum in EV.
            Default is 1.7 EV, which is the approximate value from
            Erhlet et al 2023.
        """
        if type(dinits) is str:
            source = Source()
            source.load_from_data_file(label=dinits)
            self.distances = source.distance
            source_solver_file = get_path_to_injection_solvers(
                f"src_injection_solver_{dinits}_Rmax{Rmax:.1f}.pkl"
            )

        elif type(dinits) is list:
            self.distances = dinits
            source_solver_file = get_path_to_injection_solvers(
                f"src_injection_solver_D{min(dinits)}_{max(dinits)}_Rmax{Rmax:.1f}.pkl"
            )
        else:
            raise ValueError("dinits must be either a list of floats or a string.")

        print("Computing the injection solver")

        solver_container = []

        # create linearly spaced grid for distances
        redshifts = self.spl_d_to_z_plk(self.distances)

        for i, dinit in enumerate(self.distances):
            print(f"Current distance: {dinit:.2f} Mpc")

            solvers_per_dinit = []
            zinit = redshifts[i]

            # looping over massids and alphas as well
            for massid in self.massids:
                solvers_per_massid = []
                for alpha in self.alphas:
                    solver = SingleInjectionSolver(
                        initial_z=zinit,
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
                        massid, zinit, alpha, Rmax=Rmax * 1e9
                    )  # set uniform injection as initial state

                    solver.solve(
                        dz=min(zinit / 200, 3e-4), verbose=False, progressbar=True
                    )  # solve

                    # append the results
                    solvers_per_massid.append(solver.res)
                # append the results
                solvers_per_dinit.append(solvers_per_massid)
            solver_container.append(solvers_per_dinit)

        # dump as pickle file to result file
        with open(source_solver_file, "wb") as f:
            pickle.dump(
                obj={
                    "results": solver_container,
                    "dinits": self.distances,
                    "alphas": self.alphas,
                    "massids": self.massids,
                },
                file=f,
                protocol=-1,
            )

    def run_background_injection_solver(
        self: Self,
        Rmax: float = 1.7,
        dmins: Union[None, list] = [4, 50],
        z_max: float = 3,
        source_evo: str = "SFR",
    ) -> None:
        """
        Run the background injection solver.

        Parameters
        ----------
        reset: bool
            flag to reset the pre-computation or not.
            If True, then the matrix will be computed again.
        Rmax : float, default=1.7  EV
            the maximal rigidity set in the source spectrum in EV.
            Default is 1.7 EV, which is the approximate value from
            Erhlet et al 2023.
        dmins : list[float], default=None
            the distances to stop injecting.
            Default is [4, 50] Mpc.
        z_max : float, default=3
            the maximum redshift to start from.
        source_evo : str, default=SFR
            the source evolution model to use.
            Default is SFR, but can be any of the following:
            - SFR
            - simple
            See the prince documentation for more details.

            By default the source evolution parameter m is set to zero.
            In the future this should be its own parameter that we can pass.
        """
        bg_solver_file = get_path_to_injection_solvers(
            f"bg_injection_solver_zmax{z_max}_Rmax{Rmax:.1f}_{source_evo}.pkl"
        )

        print("Computing the injection solver")

        solver_container = []

        for i, dmin in enumerate(dmins):
            print(f"Current distance: {dmin:.2f} Mpc")

            solvers_per_dmin = []
            zmin = self.spl_d_to_z_plk(dmin)

            # looping over massids and alphas as well
            for massid in self.massids:
                solvers_per_massid = []
                for alpha in self.alphas:
                    # initialise solution class first
                    inj_source = TruncatedInjectionSource(
                        self.prince_run,
                        params={massid: (alpha, Rmax, 1.0)},
                        m=(source_evo, 0.0),
                    )
                    # manually setting minimum redshift to which we
                    # stop injecting things anymore
                    inj_source.set_zmin(zmin)

                    solver = TruncatedPropagationSolver(
                        initial_z=z_max,
                        final_z=0.0,
                        prince_run=self.prince_run,
                        enable_pairprod_losses=True,
                        enable_adiabatic_losses=True,
                        enable_injection_jacobian=True,
                        enable_partial_diff_jacobian=True,
                    )

                    solver.add_source_class(inj_source)  # no source model

                    solver.solve(dz=1e-3, verbose=False, progressbar=True)  # solve

                    # append the results
                    solvers_per_massid.append(solver.res)
                # append the results
                solvers_per_dmin.append(solvers_per_massid)
            solver_container.append(solvers_per_dmin)

        # dump as pickle file to result file
        with open(bg_solver_file, "wb") as f:
            pickle.dump(
                obj={
                    "results": solver_container,
                    "dmins": dmins,
                    "alphas": self.alphas,
                    "massids": self.massids,
                },
                file=f,
                protocol=-1,
            )
