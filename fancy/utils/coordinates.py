"""Utility function for converting between different coordinate systems."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def get_coordinates(glon : np.ndarray, glat : np.ndarray, D : Union[np.ndarray, None]=None) -> SkyCoord:
    """
    Convert glon and glat to astropy SkyCoord.

    Add distance if possible (allows conversion to cartesian coords)

    Parameters
    ----------
    glon: float
        Galactic longitude in degrees
    glat: float
        Galactic latitude in degrees
    D: float, optional
        Distance in Mpc

    Returns
    -------
    astropy.coordinates.SkyCoord
    """
    if D:
        return SkyCoord(
            l=glon * u.degree,
            b=glat * u.degree,
            frame="galactic",
            distance=D * u.mpc,
        )
    else:
        return SkyCoord(l=glon * u.degree, b=glat * u.degree, frame="galactic")
    

def uv_to_coord(uv : np.ndarray) -> SkyCoord:
    """
    Convert unit vector array into SkyCoord object in the ICRS frame.

    Parameters
    ----------
    uv: np.ndarray
        array of 3D unit vectors
    
    Returns
    -------
    astropy SkyCoord object
    """
    transposed_uv = np.transpose(uv)
    x = transposed_uv[0]
    y = transposed_uv[1]
    z = transposed_uv[2]

    c = SkyCoord(x, y, z, unit="Mpc", representation_type="cartesian", frame="icrs")

    return c


def coord_to_uv(coord : SkyCoord) -> np.ndarray:
    """
    Convert SkyCoord object into array of unit vecotrs in the ICRS frame.

    Used for input into Stan programs.
    
    Parameters
    ----------
    coord: astropy SkyCoord object
        coordinates in skycoord format

    Returns
    -------
    an array of 3D unit vectors
    """
    c = coord.icrs
    ds = np.array([c.cartesian.x, c.cartesian.y, c.cartesian.z])
    # deal with single vector and multiple vector case
    if len(ds.shape) != 1:
        uv = (ds / np.linalg.norm(ds, axis=0)[np.newaxis, :]).T
    else:
        uv = ds / np.linalg.norm(ds)
    return uv