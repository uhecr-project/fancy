"""Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps)."""

import typing

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
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

    _lens_cache: typing.ClassVar[dict] = {}  # class-level, shared across instances

    def __init__(
        self: Self, gmf_model: str = "JF12", lazy: bool = False
    ) -> None:
        """
        Class that handles forward simulations of GMF deflections (lensing / weighted vMF maps).

        Parameters
        ----------
        gmf_model: str, default JF12
            The desired GMF model for GMF lensing
        lazy: bool, default False
            If True, defer loading the (RAM-heavy) CRPropa magnetic lens map from
            disk until it is actually needed (i.e. on first call to
            `apply_lens_to_map` or `apply_lens_with_particles`).
            Useful when the caller may end up never using the lens, e.g. because
            cached results will be loaded from tables instead.
        """
        if crpropa is None:
            raise ImportError("CRPropa must be installed to use this functionality.")

        self.gmf_model = gmf_model
        self._lazy = lazy
        self.gmf_lens = None

        # read in GMF lens if we have GMF enabled
        if gmf_model in list(self.__lens_names.keys()):
            self.disable_gmf = False
            if not lazy:
                self.__load_lens()
        elif gmf_model == "None":
            self.disable_gmf = True
        else:
            raise NotImplementedError(
                f"Lensing for GMF model {gmf_model} not yet implemented."
            )

    def __load_lens(self):
        if self.gmf_lens is None:
            if self.gmf_model in self._lens_cache:
                self.gmf_lens = self._lens_cache[self.gmf_model]
            else:
                path_to_lens = str(get_path_to_lens(self.__lens_names[self.gmf_model]))
                self.gmf_lens = crpropa.MagneticLens(path_to_lens)
                self._lens_cache[self.gmf_model] = self.gmf_lens

    def apply_lens_with_particles(
        self: Self, rigidities: np.ndarray, coordinates: SkyCoord, return_mask: bool = False
    ) -> typing.Union[SkyCoord, typing.Tuple[SkyCoord, np.ndarray]]:
        """
        Apply GMF lensing particle by particle (crpropa MagneticLens.transformCosmicRay),
        so that each particle keeps its own rigidity: Earth direction i is the
        lensed direction of input particle i.

        A particle is lost (does not reach Earth) with the lens' probability for
        its Galactic-boundary pixel and rigidity; lost particles get NaN
        directions. Dropping them reproduces the lensed sky distribution that the
        previous map-based sampling (ParticleMapsContainer.getRandomParticles)
        drew from, which however re-drew particles at random and so lost the
        correspondence between each event's direction and its rigidity.

        Parameters
        ----------
        rigidities: np.ndarray
            rigidities from particle samples in EV
        coordinates: astropy.coordinates.SkyCoord
            arrival directions of samples at the Galacitc boundary in SkyCoord
        return_mask: bool, default False
            also return the boolean mask of particles that reached Earth

        Returns
        -------
        astropy.coordinates.SkyCoord
            arrival directions of samples at Earth in Galactic coordinates
            (NaN for particles that did not reach Earth)
        np.ndarray (only if return_mask)
            True for particles that reached Earth
        """
        Nsamples = coordinates.shape[0]
        coords_gb_xyz = coordinates.galactic.cartesian.xyz.value.T  # (Nsamples, 3)

        if self.disable_gmf:
            survived = np.ones(Nsamples, dtype=bool)
            coords_earth_xyz = coords_gb_xyz.copy()
        else:
            self.__load_lens()
            survived = np.zeros(Nsamples, dtype=bool)
            coords_earth_xyz = np.full((Nsamples, 3), np.nan)
            for i in range(Nsamples):
                # the lens works with momentum vectors, i.e. minus the arrival direction
                p = crpropa.Vector3d(*(-coords_gb_xyz[i]))
                if self.gmf_lens.transformCosmicRay(rigidities[i] * crpropa.EeV, p):
                    p_earth = np.array([p.x, p.y, p.z])
                    coords_earth_xyz[i] = -p_earth / np.linalg.norm(p_earth)
                    survived[i] = True

        coords_earth = SkyCoord(
            CartesianRepresentation(*coords_earth_xyz.T), frame="galactic"
        )
        coords_earth.representation_type = "unitspherical"
        if return_mask:
            return coords_earth, survived
        return coords_earth

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
            self.__load_lens()
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
