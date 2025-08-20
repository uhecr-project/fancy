"""Helper functions from prince (classes) that we use to generate the tables."""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import astropy.units as u
from astropy.cosmology import WMAP9, Planck18, z_at_value
from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline
from scipy.optimize import Bounds, minimize

# simply try to import prince, if it doesnt exist and we
# just want to fit then these imports will simply be skipped
try:
    from prince_cr import util as pru
    from prince_cr.cr_sources import CosmicRaySource, AugerFitSource
    from prince_cr import core, cross_sections, photonfields
    from prince_cr.solvers import UHECRPropagationSolverBDF
except ImportError:
    pass

 # photon fields, combined CMB & EBL from Gilmore
pf_gilmore = photonfields.CombinedPhotonField(
    [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
)

class SingleInjectionSolver(UHECRPropagationSolverBDF):
    """
    Solver class for single "one-shot" injection. 

    i.e. we set the initial state to be the injection spectrum (here Auger)
    and then solve the propagation equations.
    """

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

    def set_initial_state(self, nco_id, initial_z, alpha_s, Rmax = 10**9.7):
        import warnings

        ewidths = np.diff(self.ebins)
        self.initial_state = np.zeros(self.dim_states)

        # calculate an initial state with power-law + exponential cutoff
        # injection spectrum

        # the mass & charge at this ID
        Ainj = self.spec_man.ncoid2sref[nco_id].A
        Zinj = self.spec_man.ncoid2sref[nco_id].Z
        Emax = Zinj * Rmax

        # converting to TOTAL energy here
        Egrid = self.egrid * Ainj

        # the energy at which the injection spectrum is cut off
        exp_cutoff = np.where(Egrid > Emax, np.exp(1 - Egrid / Emax), 1.0)
        inj_spec = self.spec_man.ncoid2sref[nco_id]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.initial_state[inj_spec.sl] = Ainj * (Egrid / 1e9) ** -alpha_s * exp_cutoff

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

class TruncatedPropagationSolver(UHECRPropagationSolverBDF):
    """
    Solver class for continuous homogeneous injection up to a threshold redshift. 

    i.e. we perform homogenous injection up to a given threshold (defined by the 
    source model) and we stop injecting after that.
    """

    def _init_solver(self, dz):
        # print("_init_")
        initial_state = np.zeros(self.dim_states)

        self._update_jacobian(self.initial_z)
        self.current_z_rates = self.initial_z

        # find the maximum injection and reduce the system by this
        # HACK: assuming only one source class, which is fine
        self.red_idx = self.injection(1.0, self.list_of_sources[0].zmin).max()

        # Convert csr_matrix from GPU to scipy
        try:
            sparsity = self.had_int_rates.get_hadr_jacobian(self.initial_z, 1.0).get()
        except AttributeError:
            sparsity = self.had_int_rates.get_hadr_jacobian(self.initial_z, 1.0)

        from prince_cr.util import PrinceBDF

        self.r = PrinceBDF(
            self.eqn_derivative,
            self.initial_z,
            initial_state,
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


class NoInjection(CosmicRaySource):
    """Zero injection class to solve autonomous equation."""

    def injection_spectrum(self, pid, energy, params):
        return np.zeros_like(energy)

class TruncatedInjectionSource(AugerFitSource):
    """Injection class that truncates the spectrum at a given redshift."""

    def set_zmin(self, zmin=1):
        """Set the minimum redshift in which we inject until."""
        self.zmin = zmin

    def injection_rate(self, z):
        if z > self.zmin:
            return super().injection_rate(z)
        else:
            return np.zeros_like(self.injection_grid)
        
def create_kernel(css: str, path_to_kernel: str) -> None:
    """
    Create kernel and save the results if not yet done so.
    
    Parameters
    ----------
    css : str
        Name of the cross section file.
        Available options are 'CRP2_TALYS' or 'PSB'.
    path_to_kernel : str
        Path to save the generated kernel.
    """

    # cross section class, either use TALYS or PSB
    cs = cross_sections.CompositeCrossSection(
        [
            (0.0, cross_sections.TabulatedCrossSection, (css,)),
            (0.14, cross_sections.SophiaSuperposition, ()),
        ]
    )

    # generate kernel
    prince_run = core.PriNCeRun(
        max_mass=56, photon_field=pf_gilmore, cross_sections=cs
    )

    # pickle dump the results
    pickle.dump(prince_run, open(path_to_kernel, "wb"), protocol=-1)

def create_distance_tables(path_to_distance_tables: str) -> None:
    """Create conversion table from redshift to Mpc if not yet done so."""

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
        open(path_to_distance_tables, "wb"),
    )