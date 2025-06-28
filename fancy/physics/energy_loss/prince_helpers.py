"""Helper functions from prince (classes) that we use to generate the tables."""

import numpy as np
import matplotlib.pyplot as plt

# simply try to import prince, if it doesnt exist and we
# just want to fit then these imports will simply be skipped
try:
    from prince_cr import util as pru
    from prince_cr.cr_sources import CosmicRaySource, AugerFitSource
    from prince_cr.solvers import UHECRPropagationSolverBDF
except ImportError:
    pass



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