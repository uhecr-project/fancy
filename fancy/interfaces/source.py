import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import h5py
from typing_extensions import Self, Union

from fancy.utils.coordinates import get_coordinates, uv_to_coord, coord_to_uv
from fancy.utils.helpers import create_dataset_compressed
from fancy.utils.package_data import get_path_to_datafiles

from fancy.plotting import AllSkyMapCartopy as AllSkyMap

__all__ = ["Source"]


class Source:
    """Stores the data and parameters for sources."""

    def __init__(self: Self) -> None:
        """Initialise empty container."""
        self.label = None

        self.distance = None
        self.N = None
        self.coord = None
        self.label = None
        self.unit_vector = None
        self.names = None

    def load_from_data_file(self, label: str = "M82", filename: str = "sourcedata.h5") -> None:
        """
        Store the data and parameters for sources.

        Parameters
        ----------
        label: str
            identifier
        filename: str
            file containing source data
        """
        self.label = label
        path_to_source_data = get_path_to_datafiles(filename)

        with h5py.File(path_to_source_data, "r") as f:
            data = f[self.label]
            self.distance = data["D"][()]
            self.N = len(self.distance)
            glon = data["glon"][()]
            glat = data["glat"][()]
            self.coord = get_coordinates(glon, glat)
            self.names = data["name"][()]

        self.unit_vector = coord_to_uv(self.coord)

    def __get_properties(self: Self) -> dict:
        """Pack and return objects into dict."""
        properties = {}
        properties["label"] = self.label
        properties["N"] = self.N
        properties["unit_vector"] = self.unit_vector
        properties["distance"] = self.distance
        return properties

    def load_from_properties(self: Self, source_properties: dict) -> None:
        """
        Define sources from properties dict.

        Parameters
        ----------
        source_properties: dict
            dict containing source properties.
        """
        self.label = source_properties["label"]
        if isinstance(self.label, bytes):
            self.label = self.label.decode("UTF-8")

        self.N = source_properties["N"]
        self.unit_vector = source_properties["unit_vector"]
        self.distance = source_properties["distance"]

        self.coord = uv_to_coord(self.unit_vector)

    def plot_skymap(self, skymap: AllSkyMap, size : float=2.0, color : str="k") -> None:
        """
        Plot the sources on a map of the sky.

        Called by Data.plot_skymap()

        Parameters
        ----------
        skymap: AllSkyMapCartopy
            the AllSkyMap
        size: float
            tissot radius
        color : str
            color of the tissot circles
        """
        alpha_level = 0.9

        # plot the source locations
        write_label = True
        for lon, lat in np.nditer(
            [self.coord.galactic.l.deg, self.coord.galactic.b.deg]
        ):
            if write_label:
                skymap.tissot(
                    lon,
                    lat,
                    size,
                    npts=30,
                    color="k",
                    alpha=alpha_level,
                    label=self.label,
                )
                write_label = False
            else:
                skymap.tissot(lon, lat, size, npts=30, color="k", alpha=alpha_level)

    def save(self: Self, file_handle: h5py.File) -> None:
        """
        Save to the passed H5py file handle.

        i.e. something that cna be used with
        file_handle.create_dataset()

        file_handle: h5py.File
            h5py File handle object
        """
        properties = self.__get_properties()

        for key, value in properties.items():
            create_dataset_compressed(file_handle, key, value)

    def select_sources(self : Self, selection : list) -> None:
        """Select sources by providing certain indices from a list."""
        # store selection
        self.selection = selection

        # make selection
        self.unit_vector = [self.unit_vector[i] for i in selection]
        self.distance = [self.distance[i] for i in selection]

        self.N = len(self.distance)

        self.coord = self.coord[selection]

    def select_from_distance(self : Self, Dth : float) -> None:
        """
        Select sources with distance <= Dth.

        Dth should be eneterd in [Mpc].
        """
        selection = [i for i, d in enumerate(self.distance) if d <= Dth]
        self.selection = selection

        self.unit_vector = [self.unit_vector[i] for i in selection]
        self.distance = [self.distance[i] for i in selection]

        self.N = len(self.distance)

        self.coord = self.coord[selection]
