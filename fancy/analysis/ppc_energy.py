import numpy as np
from typing_extensions import Union, Self, Tuple

from fancy.simulation.energy_simulation import EnergySimulation
import cmdstanpy


class PPCEnergy:
    def __init__(
        self: Self,
        fit: cmdstanpy.stanfit.mcmc.CmdStanMCMC,
        energy_simulation: EnergySimulation,
    ) -> None:
        """
        Generate PPCs (Posterior Predictive Checks) from energy fit results.

        Parameters
        ----------
        - fit : cmdstanpy.stanfit.mcmc.CmdStanMCMC
            The fit results from a CmdStanMCMC model.
        """
        self.fit = fit
        self.energy_simulation = energy_simulation

        # make sure that the grids of the energy simulation are already defined
        if (
            self.energy_simulation.energy_grid is None
            or self.energy_simulation.lnA_energy_grid is None
        ):
            raise ValueError(
                "Energy simulation grids are not defined. Please set them with energy_simulation.initialie_grid() before using PPC."
            )

    def get_ppc(
        self: Self, N_ppc_samples: int = 100, seed: Union[int, None] = None
    ) -> np.ndarray:
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
            "logE",
            "mean_lnA",
            "var_lnA",
        ]
        hyperparams_fit = [
            "alphas",
            "mass_fracs",
            "Nex",
            "src_frac",
            "delta_logE_sys",
            "delta_mulnA_sys",
            "delta_varlnA_sys",
        ]

        energy_ppcs = []
        mean_lnA_ppcs = np.zeros((N_ppc_samples, self.energy_simulation.NElnAs))
        var_lnA_ppcs = np.zeros((N_ppc_samples, self.energy_simulation.NElnAs))

        ippc = 0
        while ippc < N_ppc_samples:
            # get a realisation of the posterior sample
            pos_idx = rng.choice(
                self.fit.stan_variables()["alphas"].shape[0], size=1, replace=False
            )[0]

            # get the hyperparameters
            hyperparams_vals = {}
            sys_params = {}
            for ihp, hp in enumerate(hyperparams_fit):
                param = self.fit.stan_variables()[hp][pos_idx]
                if hp == "Nex":
                    param = int(param)
                    print(param)

                    # param /= 0.05076991387177731 
                    hyperparams_vals[hyperparams[ihp]] = param
                elif hp == "mass_fracs":
                    param = param.T
                    hyperparams_vals[hyperparams[ihp]] = param
                if "delta" in hp:
                    sys_params[hyperparams[ihp]] = param
                else:
                    hyperparams_vals[hyperparams[ihp]] = param

            # get the energy, mean lnA and var lnA
            self.energy_simulation.set_truths(**hyperparams_vals, sys_params=sys_params)

            # if self.energy_simulation.truths["Nex"] <= 10 or self.energy_simulation.truths["Nex"] > 300:
            #     # skip this sample if Nex is not in a reasonable range
            #     print(f"Skipping sample {ippc} with Nex={self.energy_simulation.truths['Nex']}.")
            #     continue

            # run the energy simulation
            try:
                self.energy_simulation.generate_samples()

                # get the forward foldede PPC results
                Edets, mean_lnA_det, var_lnA_det = (
                    self.energy_simulation.apply_detector_response(
                        self.energy_simulation.config["mean_lnA_unc"],
                        self.energy_simulation.config["var_lnA_unc"],
                        self.energy_simulation.config["energy_unc"],
                    )
                )
            except OverflowError:
                # if the simulation fails due to overflow, skip this sample
                print(f"Overflow error for sample {ippc}, skipping.")
                continue

            # append results
            energy_ppcs.append(Edets)  # use list append since the Nex can vary
            mean_lnA_ppcs[ippc, :] = mean_lnA_det
            var_lnA_ppcs[ippc, :] = var_lnA_det

            ippc += 1

        return energy_ppcs, mean_lnA_ppcs, var_lnA_ppcs
    
    def compute_energy_mean_and_std(
        self : Self,
        energy_ppcs: list,
        bins: Union[int, np.ndarray] = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and standard deviation of the unbinned ppc results.

        Parameters
        ----------
        - energy_ppcs : list
            Uninned posterior predictive checks for energy.
        - bins : int or np.ndarray, optional
            Number of bins or bin edges for the histogram. Default is 20.
        """
        # bin the energy ppc results
        Nbins = bins if isinstance(bins, int) else len(bins) - 1
        binned_energy_ppcs = np.zeros((len(energy_ppcs), Nbins))
        for i, eppc in enumerate(energy_ppcs):
            binned_energy_ppcs[i, :] = np.histogram(eppc, bins=bins, density=False)[0]

        # compute mean and std
        energy_mean = np.mean(binned_energy_ppcs, axis=0)
        energy_std = np.std(binned_energy_ppcs, axis=0)

        return energy_mean, energy_std
