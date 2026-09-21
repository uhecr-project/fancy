"""Category-based priors on beta_egmf (EGMF spread, in nG Mpc^1/2).

Maps a source's large-scale-structure environment (filament, void, cluster, ...)
to a (mean, sd) prior on log10(beta_egmf): a central field-strength value with
a width in dex (sigma_dex), directly specified per category below.
"""

import numpy as np

__all__ = ["EGMF_STRUCTURE_LOG10_PRIORS", "get_log10_beta_egmf_prior"]


def _log10_prior(mean_ng: float, sigma_dex: float) -> tuple:
    return np.log10(mean_ng), sigma_dex


# mean_ng: central beta_egmf value in nG Mpc^1/2; sigma_dex: width in dex (log10 units)
EGMF_STRUCTURE_LOG10_PRIORS = {
    "filament": _log10_prior(mean_ng=5.0, sigma_dex=1.0),
    "void": _log10_prior(mean_ng=0.1, sigma_dex=2.0),
}


def get_log10_beta_egmf_prior(structure: str) -> tuple:
    """
    Return (mean, sd) of the prior on log10(beta_egmf) for a structure category.

    Parameters
    ----------
    structure: str
        one of the keys in EGMF_STRUCTURE_LOG10_PRIORS (e.g. "filament", "void").
    """
    if structure not in EGMF_STRUCTURE_LOG10_PRIORS:
        raise KeyError(
            f"Unknown egmf_structure '{structure}'. "
            f"Known categories: {list(EGMF_STRUCTURE_LOG10_PRIORS)}"
        )
    return EGMF_STRUCTURE_LOG10_PRIORS[structure]
