"""Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps)."""

import typing

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from typing_extensions import Self

from fancy.utils.package_data import (
    get_path_to_lens,
)

try:
    import crpropa
except ImportError:
    crpropa = None


class GMFLensing:
    """Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps)."""

    __lens_names: typing.ClassVar[dict] = {
        "JF12": "JF12full_Gamale",
        "UF23all": "UF23_all",
        "UF23allTurb": "UF23Turb_all",
        "UF23base" : "UF23_base",
        "UF23baseTurb" : "UF23Turb_base",
    }
    __npix: int = 49152  # pixelisation of order 6

    def __init__(self: Self, gmf_model: str = "JF12") -> None:
        """
        Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps).

        Parameters
        ----------
        gmf_model: str, default JF12
            The desired GMF model for GMF lensing
        """
        if crpropa is None:
            raise ImportError("CRPropa must be installed to use this functionality.")

        self.gmf_model = gmf_model

        # read in GMF lens if we have GMF enabled
        if gmf_model in list(self.__lens_names.keys()):
            path_to_lens = str(get_path_to_lens(self.__lens_names[self.gmf_model]))
            self.gmf_lens = crpropa.MagneticLens(path_to_lens)
            self.disable_gmf = False
        elif gmf_model == "None":
            self.disable_gmf = True
        else:
            raise NotImplementedError(
                f"Lensing for GMF model {gmf_model} not yet implemented."
            )

    def apply_lens_with_particles(
        self: Self, rigidities: np.ndarray, coordinates: SkyCoord
    ) -> SkyCoord:
        """
        Apply GMF lensing by sampling & re-sampling. Returns same number of sampled events at earth as SkyCoord objects.

        Parameters
        ----------
        rigidities: np.ndarray
            rigidities from particle samples in EV
        coordinates: astropy.coordinates.SkyCoord
            arrival directions of samples at the Galacitc boundary in SkyCoord

        Returns
        -------
        astropy.coordinates.SkyCoord
            arrival directions of samples at Earth in Galactic coordinates
        """
        # now GMF lensing
        particle_map = crpropa.ParticleMapsContainer()
        Nsamples = coordinates.shape[0]

        for i in range(Nsamples):
            # make coordinate system consistent
            coord_gb_xyz = -1 * coordinates[i].cartesian.xyz.value
            vector3d_gb = crpropa.Vector3d(*coord_gb_xyz)

            # adding rigidities instead of energy
            particle_map.addParticle(
                crpropa.nucleusId(1, 1), rigidities[i] * crpropa.EeV, vector3d_gb
            )

        # lens
        if not self.disable_gmf:
            particle_map.applyLens(self.gmf_lens)

        # sample back same number of particles at earth
        _, _, glon_earth, glat_earth = particle_map.getRandomParticles(int(Nsamples))

        return SkyCoord(
            glon_earth * u.rad,
            glat_earth * u.rad,
            frame="galactic",
            representation_type="unitspherical",
        )

    def apply_lens_to_map(self: Self, weighted_map: np.ndarray, R: float) -> np.ndarray:
        """
        Apply GMF lensing from weighted healpy map.

        Parameters
        ----------
        weighted_map: healpix array
            map of normalised counts that represent an event distribution at each coordinate.
            Must be of Pixelisation order 6 (NPIX = 49152) following CRPropa conventions.
        R: float
            rigidity in EV

        Returns
        -------
        the lensed weighted map at Earth in np.ndarray
        """
        # make sure dimensionality is of order 6
        if len(weighted_map) != self.__npix:
            raise ValueError(
                "Dimension of weighted map must be of order 6 (NPIX = 49152)!"
            )

        # also make sure weighted map is normalised
        assert np.sum(weighted_map) < 1.01 and np.sum(weighted_map) > 0.99, (
            f"sum of unlensed weights = {np.sum(weighted_map)} != 1"
        )

        # create maps container and add weights to it
        particles = crpropa.ParticleMapsContainer()
        particles.addWeights(R * crpropa.EeV, weighted_map)

        # apply lensing
        if not self.disable_gmf:
            particles.applyLens(R * crpropa.EeV, self.gmf_lens)

        # obtain the lensed weights
        lensed_weighted_map = particles.getWeights(
            crpropa.nucleusId(1, 1), R * crpropa.EeV
        )

        # force nan values to be minimum value of probability
        lensed_weighted_map[np.isnan(lensed_weighted_map)] = 1 / self.__npix

        assert (
            np.sum(lensed_weighted_map) < 1.01 and np.sum(lensed_weighted_map) > 0.99
        ), f"sum of lensed weights = {np.sum(lensed_weighted_map)} != 1"

        return lensed_weighted_map
