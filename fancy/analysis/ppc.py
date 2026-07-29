import numpy as np
from typing_extensions import Union, Self, Tuple, List, Optional
from astropy.coordinates import SkyCoord
from astropy.coordinates import concatenate as concatenate_skycoords

from fancy import Data
from fancy.simulation.simulation import Simulation
import cmdstanpy

class PPC:
    def __init__(
        self: Self,
        fit: cmdstanpy.stanfit.mcmc.CmdStanMCMC,
        data: Data,
        gmf_model : str,
        simulation : Simulation
    ) -> None:
        """
        Generate PPCs (Posterior Predictive Checks) from (energy + spatial) fit results.

        Parameters
        ----------
        - fit : cmdstanpy.stanfit.mcmc.CmdStanMCMC
            The fit results from a CmdStanMCMC model.
        """
        self.fit = fit
        self.data = data
        self.gmf_model = gmf_model
        self.simulation = simulation

    def get_ppc(
        self: Self, N_ppc_samples: int = 100, seed: Union[int, None] = None, **grid_kwargs
    ) -> Tuple[List, List[SkyCoord], np.ndarray, np.ndarray]:
        """
        Calculate the poseterior predictive checks for the fit results.

        Parameters
        ----------
        - N_ppc_samples : int, optional
            Number of posterior predictive samples to generate. Default is 100.

        Returns
        -------
        - np.ndarray: Array of PPC values.
        """
        # TODO: add a limit such that it does not exceed the number of chaings * number of iterations
        rng = np.random.default_rng(seed)

        hyperparams = [
            "alphas",
            "mass_fracs",
            "Nex",
            "source_fraction",
            "beta_egmf"
        ]
        hyperparams_fit = [
            "alphas",
            "mass_fracs",
            "Nex",
            "src_frac",
            "log10_beta_egmf"
        ]

        energy_ppcs = []
        skycoord_gb_ppcs = []
        # mean_lnA_ppcs = np.zeros((N_ppc_samples, len(self.data.detector.lnA_logE_grid)))
        # var_lnA_ppcs = np.zeros((N_ppc_samples, len(self.data.detector.lnA_logE_grid)))
        mean_lnA_ppcs = []
        var_lnA_ppcs = []

        ippc = 0
        while ippc < N_ppc_samples:
            # get a realisation of the posterior sample
            pos_idx = rng.choice(
                self.fit.stan_variables()["alphas"].shape[0], size=1, replace=False
            )[0]

            # get the hyperparameters
            hyperparams_vals = {}
            for ihp, hp in enumerate(hyperparams_fit):
                param = self.fit.stan_variables()[hp][pos_idx]
                if hp == "Nex":
                    param = int(param)
                    hyperparams_vals[hyperparams[ihp]] = param
                elif hp == "mass_fracs":
                    print(param.shape)
                    param = param.T
                    hyperparams_vals[hyperparams[ihp]] = param
                elif hp == "log10_beta_egmf":
                    param = 10**param
                    hyperparams_vals[hyperparams[ihp]] = param
                else:
                    hyperparams_vals[hyperparams[ihp]] = param

            # generate a new instance of the simulation
            sim_ppc = Simulation(data=self.data, gmf_model=self.gmf_model)
            sim_ppc.initialise_grids(**grid_kwargs)
            # set the truths here
            sim_ppc.set_truths(**hyperparams_vals)

            # run the simulation
            try:
                _, skycoord_gb_truths, _, _ = sim_ppc.generate_samples(sampling_factor=20)
                _ = sim_ppc.get_skycoords_earth(skycoord_gb_truths)


                # get the forward folded PPC results
                mean_lnA_det, var_lnA_det = (
                    sim_ppc.apply_mass_response(
                        self.simulation.config["mean_lnA_stat"],
                        self.simulation.config["var_lnA_stat"],
                        self.simulation.config["mean_lnA_sys"],
                        self.simulation.config["var_lnA_sys"],
                    )
                )
                Edets, skycoord_earth_dets = sim_ppc.apply_energy_directional_response(
                    self.simulation.config["logE_stat"],
                    self.simulation.config["kappa_det"],
                    self.simulation.config["logE_sys"]
                )
                skycoord_gb_dets, _ = sim_ppc.backpropagate_events()
            except OverflowError:
                # if the simulation fails due to overflow, skip this sample
                print(f"Overflow error for sample {ippc}, skipping.")
                continue

            # append results
            energy_ppcs.append(Edets)  # use list append since the Nex can vary
            mean_lnA_ppcs.append(mean_lnA_det)
            var_lnA_ppcs.append(var_lnA_det)

            if self.simulation.gmf_model != "None":
                skycoord_gb_ppcs.append(skycoord_gb_dets)
            else:
                skycoord_gb_ppcs.append(skycoord_earth_dets)

            ippc += 1
        
        mean_lnA_ppcs = np.array(mean_lnA_ppcs)
        var_lnA_ppcs = np.array(var_lnA_ppcs)

        return energy_ppcs, skycoord_gb_ppcs, mean_lnA_ppcs, var_lnA_ppcs
    
    def compute_mean_and_std_energies(
        self : Self,
        energy_ppcs: list,
        bins: Union[int, np.ndarray] = 20,
        energy_truths : Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and standard deviation of the unbinned ppc results.

        Parameters
        ----------
        - energy_ppcs : list
            Unbinned posterior predictive checks for energy.
        - bins : int or np.ndarray, optional
            Number of bins or bin edges for the histogram. Default is 20.
        - energy_truths : np.ndarray, optional
            The true energies to also histogram. Default is None.

        Returns
        -------
        - energy_mean : np.ndarray
            Mean of the binned energy PPC results.
        - energy_std : np.ndarray
            Standard deviation of the binned energy PPC results.
        """
        # bin the energy ppc results
        Nbins = bins if isinstance(bins, int) else len(bins) - 1
        binned_energy_ppcs = np.zeros((len(energy_ppcs), Nbins))
        for i, eppc in enumerate(energy_ppcs):
            binned_energy_ppcs[i, :] = np.histogram(eppc, bins=bins, density=False)[0]

        # compute mean and std
        energy_mean = np.mean(binned_energy_ppcs, axis=0)
        energy_std = np.std(binned_energy_ppcs, axis=0)

        # if energy truths are provided, also compute their histogram
        if energy_truths is not None:
            binned_energy_truths = np.histogram(energy_truths, bins=bins, density=False)[0]
            return energy_mean, energy_std, binned_energy_truths
        
        return energy_mean, energy_std

    def compute_mean_dirs(
        self : Self,
        skycoord_gb_ppcs : List[SkyCoord],
    ) -> SkyCoord:
        """
        Compute the mean direction from each PPC sample for each UHECR.

        Parameters:
        ----------
        - skycoord_gb_ppcs : List[SkyCoord]
            the unbinned posterior predictive checks of the arrival direction at the Galactic boundary
        
        Returns:
        -------
        - mean_skycoord_gb : SkyCoord
            the mean direction per UHECR sample
        """
        mean_skycoord_gb_list = []
        for i, coord_ppc in enumerate(skycoord_gb_ppcs):
            # convert to cartesian
            coord_uv = coord_ppc.cartesian.xyz.value.T
            mean_uv = np.mean(coord_uv, axis=1) # TODO: CHECK axis!
            mean_skycoord = SkyCoord(mean_uv, frame="galactic",
                    representation_type="cartesian")
            mean_skycoord.representation_type = "unitspherical"
            mean_skycoord_gb_list.append(mean_skycoord)
        
        mean_skycoord_gb = concatenate_skycoords(mean_skycoord_gb_list)

        return mean_skycoord_gb


    def compute_ang_dis_to_source(
        self : Self,
        skycoord_gb_ppcs : List[SkyCoord],
        bins: Union[int, np.ndarray] = 20,
        skycoord_gb_truths : Optional[SkyCoord] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the angular distance to each source for each PPC sample.

        Parameters:
        -----------
        - skycoord_gb_ppcs : List[SkyCoord]
            the unbinned posterior predictive checks of the arrival direction at the Galactic boundary
        - bins : int or np.ndarray, optional
            Number of bins or bin edges for the histogram. Default is 20.

        Returns:
        ---------
        - ang_dist_mean : np.ndarray
            mean angular distance from each source for each UHECR
            Shape is given by (Nsrcs, Nbins)
        - ang_dist_std : np.ndarray
            standard deviation of the angular distance from each source for each UHECR
            Shape is given by (Nsrcs, Nbins)
        """
        # bin the energy ppc results
        Nbins = bins if isinstance(bins, int) else len(bins) - 1
        binned_ang_dis = np.zeros((self.simulation.Nsrcs, len(skycoord_gb_ppcs), Nbins))
        binned_ang_dis_truths = np.zeros((self.simulation.Nsrcs, Nbins))
        for k, src_uv in enumerate(self.simulation.source_uvs):
            if k == self.simulation.Nsrcs:
                print("done")
                break
            for i, coord_ppc in enumerate(skycoord_gb_ppcs):
                # convert to cartesian
                coord_uv = coord_ppc.cartesian.xyz.value.T
                # dis_from_src = np.linalg.norm(coord_uv - src_uv[np.newaxis,:], axis=1)**2
                ang_dis = np.arccos(np.clip(np.dot(coord_uv, src_uv), -1.0, 1.0)) * 180/np.pi
                binned_ang_dis[k, i, :] = np.histogram(ang_dis, bins=bins, density=False)[0]

            if skycoord_gb_truths is not None:
                # also compute the angular distance for the truths
                truth_uv = skycoord_gb_truths.cartesian.xyz.value.T
                ang_dis_truths = np.arccos(np.clip(np.dot(truth_uv, src_uv), -1.0, 1.0)) * 180/np.pi
                binned_ang_dis_truths[k,:] = np.histogram(ang_dis_truths, bins=bins, density=False)[0]
        # compute mean and std
        ang_dist_mean = np.mean(binned_ang_dis, axis=1)
        ang_dist_std = np.std(binned_ang_dis, axis=1)
        
        if skycoord_gb_truths is not None:
            return ang_dist_mean, ang_dist_std, binned_ang_dis_truths
        
        return ang_dist_mean, ang_dist_std

    def compute_mean_and_std_lnA(
        self : Self,
        mean_lnA_ppcs : np.ndarray,
        var_lnA_ppcs : np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and standard deviation of the mean_lnA and var_lnA PPC results.

        Parameters:
        -----------
        - mean_lnA_ppcs : np.ndarray
            Array of shape (N_ppc_samples, NElnAs) containing the mean_lnA PPC results.
        - var_lnA_ppcs : np.ndarray
            Array of shape (N_ppc_samples, NElnAs) containing the var_lnA PPC results.

        Returns:
        ---------
        - mean_of_mean_lnA : np.ndarray
            Mean of the mean_lnA PPC results across samples.
        - std_of_mean_lnA : np.ndarray
            Standard deviation of the mean_lnA PPC results across samples.
        - mean_of_var_lnA : np.ndarray
            Mean of the var_lnA PPC results across samples.
        - std_of_var_lnA : np.ndarray
            Standard deviation of the var_lnA PPC results across samples.
        """
        mean_of_mean_lnA = np.mean(mean_lnA_ppcs, axis=0)
        std_of_mean_lnA = np.std(mean_lnA_ppcs, axis=0)

        mean_of_var_lnA = np.mean(var_lnA_ppcs, axis=0)
        std_of_var_lnA = np.std(var_lnA_ppcs, axis=0)

        return mean_of_mean_lnA, std_of_mean_lnA, mean_of_var_lnA, std_of_var_lnA
    
    def compute_mean_dir_2d(
        self : Self,
        skycoord_gb_ppcs : List[SkyCoord],
    ):
        pass

    def compute_assos_prob(self : Self) -> np.ndarray:
        """
        Compute the association probability of each UHECR to each source.
        """
        logprob = self.fit.stan_variable("loglik_event").transpose(1, 2, 0)

        uhecr_assos_prob = np.zeros((self.simulation.Nsrcs, self.simulation.truths["Nex"]))
        for k in range(self.simulation.Nsrcs+1):
            lp_norm = np.sum(np.mean(np.exp(logprob[:, k, :]), axis=0), axis=1)
            uhecr_assos_prob[k, :] = np.mean(np.exp(logprob[:, k, :]), axis=0) / lp_norm

        return uhecr_assos_prob