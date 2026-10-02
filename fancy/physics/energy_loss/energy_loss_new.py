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

    # injection solver pkls are always solved on a 0.1-spaced alpha grid;
    # requested alpha_grid steps must be an exact multiple of this and no finer.
    MIN_ALPHA_STEP = 0.1

    prince_cr.config.x_cut = 1e-4
    prince_cr.config.x_cut_proton = 1e-2
    prince_cr.config.tau_dec_threshold = np.inf
    prince_cr.config.linear_algebra_backend = "MKL"
    # prince_cr.config.cosmic_ray_grid = (7, 13, 40)
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
            alpha_step must be a multiple of `MIN_ALPHA_STEP` (0.1) and cannot
            be finer than it -- the injection solver pkls are only ever solved
            at 0.1 spacing, and `load_injection_solvers` slices that fixed grid
            down to whatever coarser spacing is requested here. alpha_min and
            alpha_max must match the solved grid's limits; only the spacing
            can change.
        massids: list[int]
            list of massids to use for the injection solver.
            The massids are defined using the usual convention in prince.
            Default is [101(H), 402(He), 1407(N), 2814(Si), 5626(Fe)].
        resources_path: str
            path where resources (kernel, redshift_distance interpolator) is stored
        nthreads: int
            number of threads used for solver
        """
        alpha_min, alpha_max, alpha_step = alpha_grid
        # guard against float remainder noise (e.g. 0.30000000000000004 % 0.1)
        step_ratio = alpha_step / self.MIN_ALPHA_STEP
        if alpha_step < self.MIN_ALPHA_STEP or not np.isclose(
            step_ratio, np.round(step_ratio)
        ):
            raise ValueError(
                f"alpha_grid step ({alpha_step}) must be a multiple of "
                f"{self.MIN_ALPHA_STEP} and cannot be smaller than it -- the "
                "injection solver pkls are only solved at this resolution."
            )
        # the step must also evenly divide the requested range, or the
        # requested grid can't land exactly on [alpha_min, alpha_max] --
        # e.g. step=0.3 over [-3.0, 5.0] (range 8.0) leaves a fractional
        # remainder and silently overshoots alpha_max via np.arange.
        n_intervals = (alpha_max - alpha_min) / alpha_step
        if not np.isclose(n_intervals, np.round(n_intervals)):
            raise ValueError(
                f"alpha_grid step ({alpha_step}) does not evenly divide the "
                f"range [{alpha_min}, {alpha_max}] -- the limits must stay "
                "exactly the same as the solved grid, so the step must "
                "divide (alpha_max - alpha_min) with no remainder."
            )

        self.css = css
        self.alpha_grid = alpha_grid
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
            alpha_idx = self._resolve_alpha_indices(solver_res_dict["alphas"])
            needs_alpha_slicing = alpha_idx is not None
            # None means the pkl's alpha grid already matches self.alphas
            # exactly -- resolve to an identity index list so the massid
            # re-sorting branch below (which always needs concrete indices
            # to iterate) works whether or not alpha slicing is also needed.
            if alpha_idx is None:
                alpha_idx = list(range(len(self.alphas)))

            # do some sorting with the given mass ids and the
            # mass ids from the solver results (and slice down to the
            # requested, possibly coarser/narrower, alpha grid)
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
                            for ia in alpha_idx
                        ]
                        for mid in self.massids
                    ]
                    for idist in range(len(self.distances))
                ]
            elif needs_alpha_slicing:
                self.solver_res_src = [
                    [  # for each distance
                        [solver_res_dict["results"][idist][ims][ia] for ia in alpha_idx]
                        for ims in range(len(self.massids))
                    ]
                    for idist in range(len(self.distances))
                ]
            # print(f"Loaded from {source_solver_file}")

        # load the background injection solver here too
        # bg_solver_file = get_path_to_injection_solvers(
        #     f"bg_injection_solver_zmax{bg_inj_config['z_max']:.0f}_Rmax{bg_inj_config['Rmax']:.1f}_{bg_inj_config['source_evo']}.pkl"
        # )
        bg_solver_file = get_path_to_injection_solvers(
            f"bg_injection_solver_zmax{bg_inj_config['z_max']:.1f}_Rmax{bg_inj_config['Rmax']:.1f}_{bg_inj_config['source_evo']}.pkl"
        )
        with open(bg_solver_file, "rb") as f:
            solver_res_dict = pickle.load(f)
            self.solver_res_bg = solver_res_dict["results"]
            self.dmins = solver_res_dict["dmins"]
            alpha_idx = self._resolve_alpha_indices(solver_res_dict["alphas"])
            needs_alpha_slicing = alpha_idx is not None
            if alpha_idx is None:
                alpha_idx = list(range(len(self.alphas)))

            # do some sorting with the given mass ids and the
            # mass ids from the solver results (and slice down to the
            # requested, possibly coarser/narrower, alpha grid)
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
                            for ia in alpha_idx
                        ]
                        for mid in self.massids
                    ]
                    for idist in range(len(self.distances))
                ]
            elif needs_alpha_slicing:
                self.solver_res_bg = [
                    [  # for each distance
                        [solver_res_dict["results"][idist][ims][ia] for ia in alpha_idx]
                        for ims in range(len(self.massids))
                    ]
                    for idist in range(len(self.distances))
                ]
            # print(f"Loaded from {source_solver_file}")
            # print(f"Loaded {bg_solver_file}")

        self.solvers_loaded = True

    def _resolve_alpha_indices(
        self: Self, pkl_alphas: np.ndarray
    ) -> Union[list, None]:
        """
        Map ``self.alphas`` (the requested, possibly coarser and/or narrower
        grid) onto index positions within ``pkl_alphas`` (the grid the
        injection solver pkl was actually solved at -- always 0.1 spacing).

        Returns ``None`` when the two grids already match exactly (the common
        case: requesting 0.1 spacing directly), so callers can skip re-slicing
        and keep using ``solver_res_dict["results"]`` as-is.

        Raises if the requested grid's limits fall outside the pkl's limits,
        or if the requested spacing doesn't land exactly on the pkl's grid
        points (which `__init__`'s multiple-of-`MIN_ALPHA_STEP` check should
        already guarantee, but the pkl's own limits are only known here).

        The requested [min, max] may be the pkl's own limits (the original,
        most common case: only the spacing coarsens) or a sub-interval of
        them (e.g. isolating whether a pileup at one edge is caused by that
        edge specifically, by narrowing the range away from it) -- both are
        accepted as long as every requested alpha value is an exact point on
        the pkl's grid. A requested range that extends beyond the pkl's
        limits is not supported (that needs the pkl regenerated over a wider
        range) and raises.
        """
        pkl_alphas = np.asarray(pkl_alphas)

        if np.array_equal(pkl_alphas, self.alphas):
            return None

        if (
            self.alphas.min() < pkl_alphas.min() - 1e-9
            or self.alphas.max() > pkl_alphas.max() + 1e-9
        ):
            raise ValueError(
                f"Requested alpha_grid limits [{self.alphas.min()}, "
                f"{self.alphas.max()}] fall outside the injection solver "
                f"pkl's limits [{pkl_alphas.min()}, {pkl_alphas.max()}]. "
                "The requested range must be the pkl's own limits or a "
                "sub-interval of them; a wider range needs the pkl "
                "regenerated."
            )

        # locate each requested alpha in the pkl's finer grid
        idx = np.searchsorted(pkl_alphas, self.alphas)
        matched = (idx < len(pkl_alphas)) & np.isclose(
            pkl_alphas[np.clip(idx, 0, len(pkl_alphas) - 1)], self.alphas
        )
        if not np.all(matched):
            missing = self.alphas[~matched]
            raise ValueError(
                f"Requested alpha grid values {missing} do not fall exactly "
                "on the injection solver pkl's grid; the requested step must "
                f"be an exact multiple of the pkl's step ({np.round(np.diff(pkl_alphas)[0], 6)})."
            )

        return idx.tolist()

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
                # each entry is [UHECRPropagationResult, array], matching the
                # background branch's (res, _) unpack below -- see
                # src_injection_solver_*.pkl's ["results"][idist][ims][ia]
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

            # res = solver_res_bg[ims][ia]
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
        
        Rmax_GV = Rmax * 1e9  # convert to GV for prince

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
                        params={massid: (alpha, Rmax_GV, 1.0)},
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

                    solver.add_source_class(inj_source)  # Auger Fit model

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
