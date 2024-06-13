import numpy as np
from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
from astropy import units as u
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm as progress_bar
from multiprocessing import Pool, cpu_count
from scipy.stats import bernoulli
import os

from ..interfaces.integration import ExposureIntegralTable
from ..interfaces.stan import Direction, convert_scale, coord_to_uv, uv_to_coord
from ..interfaces.data import Uhecr
from ..interfaces.utils import get_nucleartable, kappa_ex

from ..plotting import AllSkyMap
from ..propagation.proton_energy_loss import ProtonApproxEnergyLoss
from ..propagation.crpropa_energy_loss import CRPropaApproxEnergyLoss
from ..detector.vMF.vmf import sample_vMF, sample_sphere
from ..detector.exposure import m_dec

from fancy.interfaces.data import Data
from fancy.interfaces.stan import Model

from fancy.propagation.gmf import GMFDeflections

try:

    import crpropa

except ImportError:

    crpropa = None


class Analysis:
    """
    To manage the running of simulations and fits based on Data and Model objects.
    """

    def __init__(
        self,
        data: Data,
        model: Model,
        analysis_type=None,
        filename=None,
        summary=b"",
        energy_loss_approx: str = "loss_length",
    ):
        """
        To manage the running of simulations and fits based on Data and Model objects.

        :param data: a Data object
        :param model: a Model object
        :param analysis_type: type of analysis
        :param energy_loss_approx: Method used for energy loss approx,
            only relevant for ptype!="p". Can be "loss_length" or
            "mean_sim_energy"
        """

        self.data = data
        self.model = model
        self.filename = filename

        # Initialise file
        if self.filename:

            with h5py.File(self.filename, "w") as f:
                desc = f.create_group("description")
                desc.attrs["summary"] = summary

        self.simulation_input = None
        self.fit_input = None

        self.simulation = None
        self.fit = None

        if not model.ptype:

            raise ValueError("No particle type is defined!")

        # Energy loss calculations
        if model.ptype == "p":

            self.energy_loss = ProtonApproxEnergyLoss()

        else:

            self.energy_loss = CRPropaApproxEnergyLoss(
                ptype=model.ptype,
                method=energy_loss_approx,
            )

        # Simulation outputs
        self.source_labels = None
        self.E = None
        self.Earr = None
        self.Edet = None
        self.defl_plotvars = None

        self.arr_dir_type = "arrival_direction"
        self.E_loss_type = "energy_loss"
        self.joint_type = "joint"
        self.gmf_type = "joint_gmf"
        self.composition_type = "joint_composition"

        if analysis_type == None:
            analysis_type = self.arr_dir_type

        self.analysis_type = analysis_type

        if self.analysis_type == self.joint_type or self.analysis_type == self.gmf_type:

            # find lower energy threshold for the simulation, given Eth and Eerr
            self.model.Eth_sim = self.energy_loss.get_Eth_sim(
                self.data.detector.energy_uncertainty, self.model.Eth
            )

            # find correspsonding Eth_src
            self.Eth_src = self.energy_loss.get_Eth_src(
                self.model.Eth_sim, self.data.source.distance
            )

        if self.analysis_type == self.gmf_type or self.analysis_type == self.composition_type:

            # Set up gmf deflections
            self.gmf_deflections = GMFDeflections()

        # Set up integral tables
        params = self.data.detector.params
        varpi = self.data.source.unit_vector
        self.tables = ExposureIntegralTable(varpi=varpi, params=params)

        # table containing (A, Z) of each element
        self.nuc_table = get_nucleartable()

    def build_tables(
        self, num_points=100, sim_only=False, fit_only=False, parallel=True, nthreads=int(cpu_count() * 0.75), composition_file=None,
    ):
        """
        Build the necessary integral tables.
        """

        if sim_only:

            # kappa_true table for simulation
            if (
                self.analysis_type == self.arr_dir_type
                or self.analysis_type == self.E_loss_type
            ):
                kappa_true = self.model.kappa
                D_src = self.data.source.distance

            if self.analysis_type == self.joint_type:
                D_src = self.data.source.distance
                self.Eex = self.energy_loss.get_Eex(self.Eth_src, self.model.alpha)
                self.kappa_ex = self.energy_loss.get_kappa_ex(
                    self.Eex, self.model.B, D_src
                )
                kappa_true = self.kappa_ex

            if self.analysis_type == self.gmf_type:

                A, Z = self.nuc_table[self.model.ptype]

                # shift by 0.02 to get kappa_ex at g.b.
                D_src = self.data.source.distance - 0.02
                self.Eex = self.energy_loss.get_Eex(self.Eth_src, self.model.alpha)

                self.kappa_ex = self.energy_loss.get_kappa_ex(
                    self.Eex, self.model.B, D_src
                )

                # Find kappa_gmf for each source via lensing
                varpi = self.data.source.unit_vector
                kappa_gmf = []
                for v, k, e in zip(varpi, self.kappa_ex, self.Eex):
                    k_gmf = self.gmf_deflections.get_kappa_gmf_per_source(v, k, e, A, Z)
                    kappa_gmf.append(k_gmf)

                kappa_true = np.array(kappa_gmf)

            if parallel:
                self.tables.build_for_sim_parallel(
                    kappa_true, self.model.alpha, self.model.B, D_src, nthreads
                )
            else:
                self.tables.build_for_sim(
                    kappa_true, self.model.alpha, self.model.B, D_src
                )

        if fit_only:

            # logarithmically spcaed array with 60% of points between KAPPA_MIN and 100
            # kappa_first = np.logspace(
            #     np.log(1), np.log(10), int(num_points * 0.7), base=np.e
            # )
            # kappa_second = np.logspace(
            #     np.log(10), np.log(100), int(num_points * 0.2) + 1, base=np.e
            # )
            # kappa_third = np.logspace(
            #     np.log(100), np.log(1000), int(num_points * 0.1) + 1, base=np.e
            # )
            # kappa = np.concatenate(
            #     (kappa_first, kappa_second[1:], kappa_third[1:]), axis=0
            # )
            kappa = np.logspace(0, 5, num_points)

            # full table for fit
            if self.analysis_type == self.gmf_type:

                kappa_gmf = []
                alpha_approx = 2.5

                # in the future, read Rex from the files
                Eex = 2 ** (1 / (alpha_approx - 1)) * self.model.Eth

                A, Z = self.nuc_table[self.model.ptype]

                if parallel:
                    self.tables.build_for_fit_parallel_gmf(
                        kappa, Eex, A, Z, self.gmf_deflections, nthreads
                    )
                else:
                    self.tables.build_for_fit_gmf(
                        kappa, Eex, A, Z, self.gmf_deflections
                    )

            elif self.analysis_type == self.composition_type:
                kappa_gmf = []

                pass

            else:
                if parallel:
                    self.tables.build_for_fit_parallel(kappa, nthreads)
                else:
                    self.tables.build_for_fit(kappa)

    def build_energy_table(
        self,
        num_points=50,
        table_file=None,
        parallel=True,
        nthreads=int(cpu_count() * 0.75)
    ):
        """
        Build the energy interpolation tables.
        """

        self.E_grid = np.logspace(
            np.log(self.model.Eth), np.log(1.0e4), num_points, base=np.e
        )
        self.Earr_grid = []

        if parallel and not isinstance(self.energy_loss, CRPropaApproxEnergyLoss) and not np.isscalar(self.data.source.distance):

            args_list = [(self.E_grid, d) for d in self.data.source.distance]
            # parallelize for each source distance
            with Pool(nthreads) as mpool:
                results = list(
                    progress_bar(
                        mpool.imap(self.energy_loss.get_arrival_energy_vec, args_list),
                        total=len(args_list),
                        desc="Precomputing energy grids",
                    )
                )

                self.Earr_grid = results

        else:
            for i in progress_bar(
                range(len(self.data.source.distance)), desc="Precomputing energy grids"
            ):
                d = self.data.source.distance[i]
                self.Earr_grid.append(
                    [self.energy_loss.get_arrival_energy(e, d) for e in self.E_grid]
                )

        if table_file:
            with h5py.File(table_file, "a") as f:
                E_group = f.create_group("energy")
                E_group.create_dataset("E_grid", data=self.E_grid)
                E_group.create_dataset("Earr_grid", data=self.Earr_grid)

    def use_tables(self, input_filename, main_only=True):
        """
        Pass in names of integral tables that have already been made.
        Only the main table is read in by default, the simulation table
        must be recalculated every time the simulation parameters are
        changed.
        """

        if self.analysis_type == "joint_composition":
            '''Read from tables containing both spectrum & weighted exposures'''
            with h5py.File(input_filename, "r") as file:

                self.dmax_grid = file["dmax_grid"][()] # Mpc
                self.alpha_grid = file["alpha_grid"][()]
                self.log10_Bigmf_grid = file["log10_Bigmf_grid"][()] # log10(nG)
                self.log10_Rgrid = file["log10_rigidities"][()]

                self.dRs_grid = file["dRs_grid"][()]
                self.log10_arr_spect_grid = file["arr_spect_calc"]["log10_arrspect_grid"][()]  # log10(1/EV)
                self.log10_meanEinvs_grid = file["arr_spect_calc"]["log10_meanEinvs_grid"][()] # log10(1/EeV)

                self.log10_wexp_grid = file["Nex_calc"]["log10_wexp_grid"][()] # log10(km^2 yr)

        else:
            if main_only:
                input_table = ExposureIntegralTable(input_filename=input_filename)
                self.tables.table = input_table.table
                self.tables.kappa = input_table.kappa
                self.tables.alphas = input_table.alphas

                # uncomment once we figure out bug for single source case 
                # with h5py.File(input_filename, "r") as f:
                #     self.E_grid = f["energy/E_grid"][()]
                #     self.Earr_grid = f["energy/Earr_grid"][()]

            else:
                self.tables = ExposureIntegralTable(input_filename=input_filename)

    def _get_zenith_angle(self, c_icrs, loc, time):
        """
        Calculate the zenith angle of a known point
        in ICRS (equatorial coords) for a given
        location and time.
        """
        c_altaz = c_icrs.transform_to(AltAz(obstime=time, location=loc))
        return np.pi / 2 - c_altaz.alt.rad

    def _simulate_zenith_angles(self, start_year=2004):
        """
        Simulate zenith angles for a set of arrival_directions.

        :params: start_year: year in which measurements started.
        """

        if len(self.arrival_direction.d.icrs) == 1:
            c_icrs = self.arrival_direction.d.icrs[0]
        else:
            c_icrs = self.arrival_direction.d.icrs

        time = []
        zenith_angles = []
        stuck = []

        j = 0
        first = True
        for d in c_icrs:
            za = 99
            i = 0
            while za > self.data.detector.threshold_zenith_angle.rad:
                dt = np.random.exponential(1 / self.N)
                if first:
                    t = start_year + dt
                else:
                    t = time[-1] + dt
                tdy = Time(t, format="decimalyear")
                za = self._get_zenith_angle(d, self.data.detector.location, tdy)

                i += 1
                if i > 100:
                    za = self.data.detector.threshold_zenith_angle.rad
                    stuck.append(1)
            time.append(t)
            first = False
            zenith_angles.append(za)
            j += 1
            # print(j , za)

        if len(stuck) > 1:
            print(
                "Warning: % of zenith angles stuck is",
                len(stuck) / len(zenith_angles) * 100,
            )

        return zenith_angles

    def simulate(self, seed=None, Eth_sim=None):
        """
        Run a simulation.

        :param seed: seed for RNG
        :param Eth_sim: the minimun energy simulated
        :param gmf: enable galactic magnetic field deflections
        """

        eps = self.tables.sim_table

        # handle selected sources
        if self.data.source.N < len(eps):
            eps = [eps[i] for i in self.data.source.selection]

        # convert scale for sampling
        D = self.data.source.distance
        alpha_T = self.data.detector.alpha_T
        Q = self.model.Q
        F0 = self.model.F0
        D, alpha_T, eps, F0, Q = convert_scale(D, alpha_T, eps, F0, Q)

        if (
            self.analysis_type == self.joint_type
            or self.analysis_type == self.E_loss_type
            or self.analysis_type == self.gmf_type
        ):
            # find lower energy threshold for the simulation, given Eth and Eerr
            if Eth_sim:
                self.model.Eth_sim = Eth_sim

        # compile inputs from Model and Data
        self.simulation_input = {
            "kappa_d": self.data.detector.kappa_d,
            "Ns": len(self.data.source.distance),
            "varpi": self.data.source.unit_vector,
            "D": D,
            "A": self.data.detector.area,
            "a0": self.data.detector.location.lat.rad,
            "lon": self.data.detector.location.lon.rad,
            "theta_m": self.data.detector.threshold_zenith_angle.rad,
            "alpha_T": alpha_T,
            "eps": eps,
        }

        self.simulation_input["Q"] = Q
        self.simulation_input["F0"] = F0
        self.simulation_input["distance"] = self.data.source.distance

        if (
            self.analysis_type == self.arr_dir_type
            or self.analysis_type == self.E_loss_type
        ):

            self.simulation_input["kappa"] = self.model.kappa

        if self.analysis_type == self.E_loss_type:

            self.simulation_input["alpha"] = self.model.alpha
            self.simulation_input["Eth"] = self.model.Eth_sim
            self.simulation_input["Eerr"] = self.data.detector.energy_uncertainty

        if self.analysis_type == self.joint_type or self.analysis_type == self.gmf_type:

            self.simulation_input["B"] = self.model.B
            self.simulation_input["alpha"] = self.model.alpha
            self.simulation_input["Eth"] = self.model.Eth_sim
            self.simulation_input["Eerr"] = self.data.detector.energy_uncertainty

            # get particle type we intialize simulation with
            _, Z = self.nuc_table[self.model.ptype]
            self.simulation_input["Z"] = Z

        try:
            if self.data.source.flux:
                self.simulation_input["flux"] = self.data.source.flux
            else:
                self.simulation_input["flux"] = np.zeros(self.data.source.N)
        except:
            self.simulation_input["flux"] = np.zeros(self.data.source.N)

        # run simulation
        print("Running Stan simulation...")
        self.simulation = self.model.simulation.sample(
            data=self.simulation_input,
            iter_sampling=1,
            chains=1,
            fixed_param=True,
            seed=seed,
        )

        # extract output
        print("Extracting output...")

        self.Nex_sim = self.simulation.stan_variable("Nex_sim")[0]
        # source_labels: to which source label each UHECR is associated with
        self.source_labels = (self.simulation.stan_variable("lambda")[0]).astype(int)

        if self.analysis_type == self.arr_dir_type:
            arrival_direction = self.simulation.stan_variable("arrival_direction")[0]

        elif (
            self.analysis_type == self.joint_type
            or self.analysis_type == self.E_loss_type
            or self.analysis_type == self.gmf_type
        ):

            self.Earr = self.simulation.stan_variable("Earr")[0]  # arrival energy
            self.E = self.simulation.stan_variable("E")[0]  # sampled from spectrum

            # simulate with deflections with GMF
            if self.analysis_type == self.gmf_type:
                kappas = self.simulation.stan_variable("kappa")[0]
                print("Simulating deflections...")
                arrival_direction, self.Edet = self._simulate_deflections(kappas)

            else:
                arrival_direction = self.simulation.stan_variable("arrival_direction")[
                    0
                ]

                self.Edet = self.simulation.stan_variable("Edet")[0]

                # make cut on Eth
                inds = np.where(self.Edet >= self.model.Eth)
                self.Edet = self.Edet[inds]
                arrival_direction = arrival_direction[inds]
            # self.source_labels = self.source_labels[inds]

        # convert to Direction object
        self.arrival_direction = Direction(arrival_direction)
        self.N = len(self.arrival_direction.unit_vector)

        # simulate the zenith angles
        print("Simulating zenith angles...")
        self.zenith_angles = self._simulate_zenith_angles(self.data.detector.start_year)

        # Make uhecr object
        uhecr_properties = {}
        uhecr_properties["label"] = self.data.detector.label
        uhecr_properties["N"] = self.N
        uhecr_properties["unit_vector"] = self.arrival_direction.unit_vector
        uhecr_properties["energy"] = self.Edet
        uhecr_properties["zenith_angle"] = self.zenith_angles
        uhecr_properties["A"] = np.tile(self.data.detector.area, self.N)
        # uhecr_properties['source_labels'] = self.source_labels

        uhecr_properties["ptype"] = (
            self.model.ptype if self.analysis_type == self.gmf_type else "p"
        )

        new_uhecr = Uhecr()
        new_uhecr.from_simulation(uhecr_properties)

        # evaluate kappa_gmf manually here
        if self.analysis_type == self.gmf_type:
            print("Computing kappa_gmf...")
            (
                new_uhecr.kappa_gmf,
                omega_defl_kappa_gmf,
                omega_rand_kappa_gmf,
                _,
            ) = new_uhecr.eval_kappa_gmf(
                particle_type=self.model.ptype, Nrand=100, gmf="JF12", plot=False
            )

            self.defl_plotvars.update(
                {
                    "omega_rand_kappa_gmf": omega_rand_kappa_gmf,
                    "omega_defl_kappa_gmf": omega_defl_kappa_gmf,
                    "kappa_gmf": new_uhecr.kappa_gmf,
                }
            )

        self.data.uhecr = new_uhecr

        print("Done!")

    def _prepare_fit_inputs(self):
        """
        Gather inputs from Model, Data and IntegrationTables.
        """

        # prepare fit inputs
        self.fit_input = {
            "Ns": self.data.source.N,
            "varpi": self.data.source.coord.cartesian.xyz.value.T,
            "N": self.data.uhecr.N,
            "A": self.data.uhecr.A,
            "zenith_angle": self.data.uhecr.zenith_angle,
        }

        if self.analysis_type != self.composition_type:
            eps_fit = self.tables.table
            kappa_grid = self.tables.kappa
            # E_grid = self.E_grid
            # Earr_grid = list(self.Earr_grid)
            # temporary
            E_grid = np.zeros(50)
            Earr_grid = list(np.zeros((1, 50)))

            # convert scale for sampling
            D = self.data.source.distance
            alpha_T = self.data.detector.alpha_T
            if self.analysis_type != self.composition_type:
                D, alpha_T, eps_fit = convert_scale(D, alpha_T, eps_fit)

            # KW: due to multiprocessing appending,
            # collapse dimension from (1, 23, 50) -> (23, 50)
            eps_fit.resize(self.Earr_grid.shape)

            # handle selected sources
            if self.data.source.N < eps_fit.shape[1]:
                eps_fit = [eps_fit[i] for i in self.data.source.selection]
                Earr_grid = [Earr_grid[i] for i in self.data.source.selection]

            # add E interpolation for background component (possible extension with Dbg)
            Earr_grid.append([0 for e in E_grid])

            self.fit_input["arrival_direction"] = self.data.uhecr.unit_vector
            self.fit_input["eps"] = eps_fit
            self.fit_input["Ngrid"] = len(kappa_grid)
            self.fit_input["kappa_grid"] = kappa_grid
            self.fit_input["D"] = D
            self.fit_input["alpha_T"] = alpha_T

        if (
            self.analysis_type == self.joint_type
            or self.analysis_type == self.E_loss_type
            or self.analysis_type == self.gmf_type
        ):

            self.fit_input["Edet"] = self.data.uhecr.energy
            self.fit_input["Eth"] = self.data.detector.Eth
            self.fit_input["Eerr"] = self.data.detector.energy_uncertainty
            self.fit_input["E_grid"] = E_grid
            self.fit_input["Earr_grid"] = Earr_grid

            ptype = str(self.model.ptype)
            _, self.fit_input["Z"] = self.nuc_table[ptype]

        if self.analysis_type == self.gmf_type or self.analysis_type == self.composition_type:
            self.fit_input["kappa_gmfs"] = self.data.uhecr.kappa_gmfs
            # self.fit_input["kappa_gmf"] = np.ones_like(
            #     self.data.uhecr.kappa_gmf) * self.data.detector.kappa_d
        else:
            self.fit_input["kappa_d"] = self.data.detector.kappa_d

        if self.analysis_type == self.composition_type:
            
            # UHECR parameters
            self.fit_input["arrival_direction"] = self.data.uhecr.unit_vectors_gb
            # if we find rigidity in dataset, then use that, otherwise divide by meanZ of mass group
            if len(self.data.uhecr.rigidity) > 0:
                self.fit_input["Rdet"] = self.data.uhecr.rigidity
            else:
                self.fit_input["Rdet"] = self.data.uhecr.energy / self.data.detector.meanZ
            self.fit_input["exp_factors"] = self.data.uhecr.exposure

            # detector parameters
            self.fit_input["Rth"] = self.data.detector.Rth
            self.fit_input["Rth_max"] = self.data.detector.Rth_max
            self.fit_input["Rerr"] = self.data.detector.energy_uncertainty

            # arrival spectrum parameters
            self.fit_input["Ndmaxs"] = len(self.dmax_grid)
            self.fit_input["NRs"] = len(self.log10_Rgrid)
            self.fit_input["Nalphas"] = len(self.alpha_grid)
            self.fit_input["dmax_grid"] = self.dmax_grid
            self.fit_input["log10_Rgrid"] = self.log10_Rgrid
            self.fit_input["alpha_grid"] = self.alpha_grid
            self.fit_input["log_arr_spectrum_grid"] = np.log(10.0**self.log10_arr_spect_grid)

            # Nex / flux parameters
            self.fit_input["log10_meanEinvs_grid"] = self.log10_meanEinvs_grid
            self.fit_input["NBigmfs"] = len(self.log10_Bigmf_grid)
            self.fit_input["log10_Bigmf_grid"] = self.log10_Bigmf_grid
            self.fit_input["log10_wexp_grid"] = self.log10_wexp_grid

            



    def save(self):
        """
        Write the analysis to file.
        """

        with h5py.File(self.filename, "r+") as f:

            source_handle = f.create_group("source")
            if self.data.source:
                self.data.source.save(source_handle)

            uhecr_handle = f.create_group("uhecr")
            if self.data.uhecr:
                self.data.uhecr.save(uhecr_handle, self.analysis_type)

            detector_handle = f.create_group("detector")
            if self.data.detector:
                self.data.detector.save(detector_handle)

            model_handle = f.create_group("model")
            if self.model:
                self.model.save(model_handle)

            fit_handle = f.create_group("fit")
            if self.fit:

                # fit inputs
                fit_input_handle = fit_handle.create_group("input")
                for key, value in self.fit_input.items():
                    fit_input_handle.create_dataset(key, data=value)

                # samples
                samples = fit_handle.create_group("samples")
                for key, value in self.chain.items():
                    samples.create_dataset(key, data=value)

            else:
                plotvars_handle = f.create_group("plotvars")

                if self.analysis_type == self.gmf_type:
                    for key, value in list(self.defl_plotvars.items()):
                        plotvars_handle.create_dataset(key, data=value)

    def plot(self, type=None, cmap=None):
        """
        Plot the data associated with the analysis object.

        type == 'arrival direction':
        Plot the arrival directions on a skymap,
        with a colour scale describing which source
        the UHECR is from.

        type == 'energy'
        Plot the simulated energy spectrum from the
        source, to after propagation (arrival) and
        detection
        """

        # plot style
        if cmap == None:
            cmap = plt.cm.get_cmap("viridis")

        # plot arrival directions by default
        if type == None:
            type == "arrival_direction"

        if type == "arrival_direction":

            # figure
            fig, ax = plt.subplots()
            fig.set_size_inches((12, 6))

            # skymap
            skymap = AllSkyMap()

            self.data.source.plot(skymap)
            self.data.detector.draw_exposure_lim(skymap)
            self.data.uhecr.plot(skymap)

            # standard labels and background
            skymap.draw_standard_labels()

            # legend
            ax.legend(frameon=False, bbox_to_anchor=(0.85, 0.85))

        if type == "energy":

            bins = np.logspace(np.log(self.model.Eth), np.log(1e4), base=np.e)

            fig, ax = plt.subplots()

            if isinstance(self.E, (list, np.ndarray)):
                ax.hist(
                    self.E, bins=bins, alpha=0.7, label=r"$\tilde{E}$", color=cmap(0.0)
                )
            if isinstance(self.Earr, (list, np.ndarray)):
                ax.hist(self.Earr, bins=bins, alpha=0.7, label=r"$E$", color=cmap(0.5))

            ax.hist(
                self.data.uhecr.energy,
                bins=bins,
                alpha=0.7,
                label=r"$\hat{E}$",
                color=cmap(1.0),
            )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(frameon=False)

    def use_crpropa_data(self, energy, unit_vector):
        """
        Build fit inputs from the UHECR dataset.
        """

        self.N = len(energy)
        self.arrival_direction = Direction(unit_vector)

        # simulate the zenith angles
        print("Simulating zenith angles...")
        self.zenith_angles = self._simulate_zenith_angles()
        print("Done!")

        # Make Uhecr object
        uhecr_properties = {}
        uhecr_properties["label"] = "sim_uhecr"
        uhecr_properties["N"] = self.N
        uhecr_properties["unit_vector"] = self.arrival_direction.unit_vector
        uhecr_properties["energy"] = energy
        uhecr_properties["zenith_angle"] = self.zenith_angles
        uhecr_properties["A"] = np.tile(self.data.detector.area, self.N)

        new_uhecr = Uhecr()
        new_uhecr.from_properties(uhecr_properties)

        self.data.uhecr = new_uhecr

    def fit_model(
        self,
        iterations=1000,
        chains=4,
        seed=None,
        output_dir=None,
        warmup=None,
        show_progress=True,
        **kwargs,
    ):
        """
        Fit a model.

        :param iterations: number of iterations
        :param chains: number of chains
        :param seed: seed for RNG
        """

        # Prepare fit inputs
        self._prepare_fit_inputs()

        # fit
        print("Performing fitting...")
        self.fit = self.model.model.sample(
            data=self.fit_input,
            iter_sampling=iterations,
            chains=chains,
            seed=seed,
            show_progress=show_progress,
            output_dir=output_dir,
            iter_warmup=warmup,
            **kwargs
        )

        # Diagnositics
        print("Checking all diagnostics...")
        print(self.fit.diagnose())

        self.chain = self.fit.stan_variables()
        print("Done!")
        return self.fit
