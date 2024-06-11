import numpy as np
import os
from scipy import integrate
from matplotlib import pyplot as plt
from astropy.constants import c
from astropy import units as u
from typing import List, Tuple
from scipy import stats, optimize

from multiprocessing import Pool
from tqdm import tqdm
import h5py

from fancy import Data
from fancy.physics.energy_loss.energy_loss import EnergyLoss
from fancy.physics.energy_loss.cosmology import H0, Om, Ol, DH


class ProtonApproxEnergyLoss(EnergyLoss):
    """
    Semi-analytic approach to modelling proton energy losses.

    Wrapper class to update code below for new interface.
    """

    def __init__(self, data : Data, verbose=False):

        super().__init__(data, verbose)

        # raise exception if mass group != 1
        if self.mass_group != 1:
            raise ValueError(f"Mass Group {self.mass_group} not valid with this approach. Use Nuclei Energy Loss Model isntead.")
    
    def initialise_grid(
            self, 
            weights_dir : str = "./resources/composition_weights_PSB.h5", 
            alpha_min=-3, 
            alpha_max=10, 
            Nalphas=50,
            Emax=250,
            NEs = 50,
        ):
        '''
        Initalise our grid using composition weights
        
        :param weights_dir: directory to composition weights
        :param alpha_min, alpha_max, Nalphas: the min / max and density of spectral index grid
        :param Emax, NEs: the maximum energy (in EeV) and density of the log-spaced source energy grid
        '''
        super().initialise_grid(weights_dir, alpha_min, alpha_max, Nalphas)

        self.E_grid = np.logspace(
            np.log(self.Eth), np.log(Emax), NEs, base=np.e
        )
        self.Earr_grid = []

    def get_arrival_energy(self, Esrc: float, D: float):
        """
        Get the arrival energy for a given initial energy.
        Takes into account all propagation affects.

        :param Esrc: Source energy in EeV
        :param D: Source distance in Mpc
        """
        Esrc = Esrc * 1.0e18
        integrator = integrate.ode(_dEdr).set_integrator("lsoda", method="bdf")
        integrator.set_initial_value(Esrc, 0)
        r1 = D
        dr = min(D / 10, 10)

        while integrator.successful() and integrator.t < r1:
            integrator.integrate(integrator.t + dr)

        Earr = integrator.y / 1.0e18
        return Earr

    def get_arrival_energy_vec(self, args: Tuple):
        """
        Get the arrival energy for a given initial energy.
        Takes into account all propagation affects.
        Version for parallelisation.

        :param args: Tuple containing source energies in EeV
            and source distances in Mpc
        """
        Evec, D = args

        Earr_vec = []
        for E in Evec:
            E = E * 1.0e18
            integrator = integrate.ode(_dEdr).set_integrator("lsoda", method="bdf")
            integrator.set_initial_value(E, 0)
            r1 = D
            dr = min(D / 10, 10)

            while integrator.successful() and integrator.t < r1:
                integrator.integrate(integrator.t + dr)

            Earr = integrator.y / 1.0e18

            Earr_vec.append(Earr[0])

        return Earr_vec
    
    def p_gt_thresh(self, delta):
        """
        Probability that arrival energy is anove threshold.

        :param delta: Uncertainty in energy reconstruction (%)
        """
        delta = self.delta if delta == None else delta
        return 1 - np.array([stats.norm.cdf(self.Eth, Earr, delta * Earr) for Earr in self.Earr_grid])
    
    def p_gt_Eth(self, Earr: float, Eerr: float, Eth: float):
        """
        Probability that arrival energy is anove threshold.

        :param Earr: Arrival energy in EeV
        :param Eerr: Uncertainty in energy reconstruction (%)
        :param Eth: Threshold energy in EeV
        """

        return 1 - stats.norm.cdf(Eth, Earr, Eerr * Earr)

    def get_Eth_sim(self, Eerr: float, Eth: float):
        """
        Get a sensible threshold energy for the simulation
        given the threshold energy and uncertainty in the
        energy reconstruction.

        :param Eerr: Uncertainty in energy reconstruction (%)
        :param Eth: Threshold energy in EeV
        """

        E = optimize.fsolve(self.p_gt_Eth, Eth, args=(Eerr, Eth))

        return round(E[0])
    
    def compute_arrival_energies(self, parallel=True, njobs=4):
        '''
        Compute arrival energies to make the tables
        
        :param parallel: to enable parallel calculation or not (default True)
        :param njobs: number of jobs to parallelise (default 4)
        '''

        if parallel:

            args_list = [(self.E_grid, d) for d in self.distances.value]
            # parallelize for each source distance
            with Pool(njobs) as mpool:
                results = list(
                    tqdm(
                        mpool.imap(self.get_arrival_energy_vec, args_list),
                        total=len(args_list),
                        desc="Precomputing energy grids",
                    )
                )

                self.Earr_grid = results

        else:
            for i in tqdm(
                range(len(self.distances)), desc="Precomputing energy grids"
            ):
                d = self.distances[i].value
                self.Earr_grid.append(
                    [self.get_arrival_energy(e, d) for e in self.E_grid]
                )

    def compute_Eexs(self):
        '''Compute expected energies for all distance and energies'''
        for i, d in enumerate(self.distances.value):
            Eth_src = _proton_approx_get_source_threshold_energy(self.Eth, d)[0]
            self.Eexs[i,:] = 2 ** (1 / (self.alpha_grid - 1)) * Eth_src * u.EeV

        # limit to some smaller value
        self.Eexs[self.Eexs > np.max(self.E_grid) * u.EeV] = np.max(self.E_grid) * u.EeV

    def compute_Eth_srcs(self):
        '''Compute source threshold energies'''
        self.Eth_srcs = np.zeros(self.Ndistances)
        for i, d in enumerate(self.distances.value):
            self.Eth_srcs[i] = _proton_approx_get_source_threshold_energy(self.Eth, d)[0]

        
    def save(self, outfile):
        '''Save to h5py output file'''
        with h5py.File(outfile, "a") as f:
            config_label = f"{self.detector_type}_mg{self.mass_group}"
            if config_label in f.keys():
                del f[config_label]
            config_gr = f.create_group(config_label)

            config_gr.create_dataset("distances_grid", data=self.distances)
            config_gr.create_dataset("alpha_grid", data=self.alpha_grid)
            config_gr.create_dataset("E_grid", data=self.E_grid)
            config_gr.create_dataset("Earr_grid", data=self.Earr_grid)
            config_gr.create_dataset("Eth_src_grid", data = self.Eth_srcs)
            config_gr.create_dataset("log10_Eexs_grid", data=np.log10(self.Eexs.to_value(u.EeV)))

"""
Energy loss functions for propagation of UHECR protons. 
Based on the semi-analytical results of the continuous loss approximation, 
as described in detail in de Domenico and Insolia (2013) and also used by 
Khanin and Mortlock (2016).

The beta_* functions are loss rates for different processes. Units are as follows, unless otherwise stated:

beta_pi and beta_adi
--------------------
Energy, E [eV]
Redshift, z [dimensionless]
Loss rate, beta_* [yr^-1]

beta_bh
-------
Energy, E [EeV]
Redshift z, [dimnesionless]
Loss rate, beta_bh [s^-1]

NB: beta_bh is calculated in these units to avoid issues with numerical overflow.
NB: the forumla for beta_pi in de Domenico and Insolia (2013) is missing a minus sign 
    to have the correct for of Aexp(-B/E).

@author Francesca Capel
@date July 2018
"""

# ignore nan warnings
np.seterr(all="ignore")


def beta_pi(z, E):
    """
    GZK losses due to photomeson production.
    Originally parametrised by Anchordorqui et al. (1997)
    Based on Berezinsky and Grigor'eva (1988) suggestion of Aexp(-B/E) form.
    """

    p = [3.66e-8, (2.87e20 / 1e18), 2.42e-8]
    check = 6.86 * np.exp(-0.807 * z) * 1e20

    if E <= check:
        return p[0] * (1 + z) ** 3 * np.exp(-p[1] / ((1 + z) * (E / 1e18)))
    if E > check:
        return p[2] * (1 + z) ** 3


def beta_adi(z):
    """
    Losses due to adiabatic expansion.
    Makes use of the hubble constant converted into units of [yr^-1].
    """

    a = Om * (1 + z) ** 3
    # b = 1 - sum(lCDM) * (1 + z)**2
    b = 0
    return H0.to_value(1 / u.yr) * (a + Ol + b) ** (0.5)


def phi_inf(xi):
    """
    The approximation of the phi(xi) function as xi->inf.
    Used in calculation of beta_bh.
    Described in Chodorowski et al. (1992).
    """

    d = [-86.07, 50.96, -14.45, 8.0 / 3.0]
    sum_term = sum([d_i * np.log(xi) ** i for i, d_i in enumerate(d)])

    return xi * sum_term


def phi(xi):
    """
    Approximation of the phi(xi) function for different regimes
    of xi.
    Used in calculation of beta_bh.
    Described in Chodorowski et al. (1992).
    """

    if xi == 2:
        return np.pi / 12  # * (xi - 2)**4

    elif xi < 25:
        c = [0.8048, 0.1459, 1.137e-3, -3.879e-6]
        sum_term = 0

        for i in range(len(c)):
            sum_term += c[i] * (xi - 2) ** (i + 1)

        return (np.pi / 12) * (xi - 2) ** 4 / (1 + sum_term)

    elif xi >= 25:
        phi_inf_term = phi_inf(xi)
        f = [2.910, 78.35, 1837]
        sum_term = sum([(f_i * xi ** (-(i + 1))) for i, f_i in enumerate(f)])

        return phi_inf_term / (1 - sum_term)


def integrand(xi, E, z):
    """
    Integrand as a functional of phi(xi) used in calcultion of beta_bh.
    Described in de Domenico and Insolia (2013).
    """

    num = phi(xi)
    B = 1.02
    denom = np.exp((B * xi) / ((1 + z) * E)) - 1

    return num / denom


def beta_bh(z, E):
    """
    Losses due to Bethe-Heitler pair production.
    Described in de Domenico amnd Insolia (2013).
    """

    A = 3.44e-18
    integ, err = integrate.quad(integrand, 2, np.inf, args=(E, z))

    out = (A / E**3) * integ
    if out == np.nan or out == np.inf:
        out = 0

    return out


def Ltot(z, E):
    """
    The total energy loss length
    """

    c = 3.064e-7  # Mpc/yr

    bp = beta_pi(z, E)
    ba = beta_adi(z)
    bbh = 3.154e7 * beta_bh(z, E / 1.0e18)

    # print(bp, ba, bbh)
    # L = c / (bp + ba + bbh)
    L = dzdt(z) / (bp + ba + bbh)

    return L


def dzdt(z):
    """
    De Domenico & Insolia 2012 Equation 5.
    """

    numerator = ((Om * (1 + z) ** 3) + Ol) ** (-0.5)
    denominator = H0.to_value(1 / u.yr) * (1 + z)
    return (1 / (numerator / denominator)) * DH.to_value(u.Mpc)  # Mpc yr^-1


def make_energy_loss_plot(z, E):
    """
    Recreate figure 2 in de Domenico and Insolia (2013).
    """

    c = 3.064e-7  # Mpc/yr

    # adiabatic and GZK
    L_pi = [(dzdt(z) / beta_pi(z, e)) for e in E]
    L_adi = [(dzdt(z) / beta_adi(z)) for e in E]
    # pair production
    L_bh = [(dzdt(z) / (3.154e7 * beta_bh(z, e / 1e18))) for e in E]
    # total
    L_tot = [
        dzdt(z) / (beta_pi(z, e) + beta_adi(z) + (3.154e7 * beta_bh(z, e / 1e18)))
        for e in E
    ]

    # adiabatic and GZK
    # L_pi = [(c / beta_pi(z, e)) for e in E]
    # L_adi = [(c / beta_adi(z)) for e in E]
    # pair production
    # L_bh = [(c / (3.154e7 * beta_bh(z, e / 1e18))) for e in E]
    # total
    # L_tot = [c / (beta_pi(z, e) + beta_adi(z) + (3.154e7*beta_bh(z, e/1e18)) ) for e in E]

    plt.figure(figsize=(6, 6))

    plt.plot(E, L_pi, linewidth=5, alpha=0.7, label="Photomeson")
    plt.plot(E, L_adi, linewidth=5, alpha=0.7, label="Adiabatic")
    plt.plot(E, L_bh, linewidth=5, alpha=0.7, label="Pair")
    plt.plot(E, L_tot, "--", linewidth=5, alpha=0.7, label="Total")

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(5, 1e4)
    plt.xlim(1e18, 1e22)
    plt.xlabel("E [eV]")
    plt.ylabel("L [Mpc]")
    plt.legend(frameon=False, loc="best")


def _dEdr(r, E):
    """
    The ODE to solve for propagation energy losses.
    """
    z = r / DH.to_value(u.Mpc)
    return -E / Ltot(z, E)


def _dEdr_rev(r, E, D):
    """
    The ODE to solve for propagation energy losses.
    """
    r_rev = D - r
    z = r_rev / DH.to_value(u.Mpc)
    return E / Ltot(z, E)


def _proton_approx_get_source_threshold_energy(Eth, D):
    """
    Get the equivalent source energy for a given arrival energy.
    Takes into account all propagation affects.
    Solves the ODE dE/dr = E / Lloss.
    NB: input Eth and output E in EeV!
    """

    Eth = Eth * 1.0e18
    integrator = integrate.ode(_dEdr_rev).set_integrator("lsoda", method="bdf")
    integrator.set_initial_value(Eth, 0).set_f_params(D)
    r1 = D
    dr = 1

    while integrator.successful() and integrator.t < r1:
        integrator.integrate(integrator.t + dr)
        # print("%g %g" % (integrator.t, integrator.y))

    Eth_src = integrator.y / 1.0e18
    return Eth_src


def get_Eth_src(Eth, D):
    """
    Find the source threshold energies for all sources.
    NB: input Eth in EeV!
    """

    Eth_src = []
    for d in D:
        Eth_src.append(_proton_approx_get_source_threshold_energy(Eth, d)[0])

    return Eth_src
