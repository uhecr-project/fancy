import h5py
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing_extensions import Self, Union

from fancy.plotting import AllSkyMapCartopy as AllSkyMap
from fancy.utils.package_data import get_path_to_meanlnA

from ..detector.detector import Detector
from .source import Source
from .uhecr import Uhecr


class Data:
    """A container for high level storage of data."""

    def __init__(self: Self) -> None:
        """Contain high level storage of data."""
        self._filename = None
        self._data = None

        # uhecr, source and detector objects
        self.uhecr = None
        self.source = None
        self.detector = None

    def add_source(self: Self, filename: str, label: str = "M82") -> None:
        """
        Add a source object to the data cotainer from file.

        Parameters
        ----------
        filename: str
            name of the file containing the object's data
        label: str
            reference label for the source object
        """
        new_source = Source()
        new_source.load_from_data_file(filename, label)

        # define source object
        self.source = new_source

    def add_uhecr(
        self: Self,
        filename: str,
        label: str = "TA2015",
        mass_model: str = "EPOS-LHC",
        gmf_model: str = "JF12",
    ) -> None:
        """
        Add a uhecr object to the data container from file.

        Parameters
        ----------
        filename: str
            name of the file containing the object's data
        label: str
            reference label for the uhecr dataset
        mass_model : str
            hadronic interaction model used to get the deflection information
        gmf_model : str
            GMF model used to get the deflection information
        """
        new_uhecr = Uhecr()
        new_uhecr.load_from_data_file(filename, label, mass_model, gmf_model=gmf_model)

        # define uhecr object
        self.uhecr = new_uhecr

    def add_detector(
        self: Self,
        label: str = "TA2015",
        mass_model: str = "EPOS-LHC",
        mean_lnA_file: str = "meanlnA_logE_fit.txt",
    ) -> None:
        """
        Add a detector object to complement the data.

        Parameters
        ----------
        label : str
            label of detector
        hadr_model : str
            hadronic interaction model used to get the composition information
        mean_lnA_file : str, default="meanlnA_logE_fit.txt"
            the file containing the mean lnA values
        """
        new_detector = Detector(label)
        new_detector.get_exposure_properties()
        new_detector.set_lnA_params(
            meanlnA_file=get_path_to_meanlnA(mean_lnA_file), mass_model=mass_model
        )

        # define detector
        self.detector = new_detector

    # TODO: move this to UHECR class
    def __generate_uhecr_colorbar(self, cm: matplotlib.colors.Colormap) -> None:
        """
        Add a colorbar normalised over all the Uhecr energies.

        Parameters
        ----------
        cmap: matplotlib colorbar object
            color map for the plot
        """
        max_energies = []
        min_energies = []
        # find the min and max uhecr energies
        max_energies.append(max(self.uhecr.energy))
        min_energies.append(min(self.uhecr.energy))

        max_energy = max(max_energies)
        min_energy = min(min_energies)

        norm_E = matplotlib.colors.Normalize(min_energy, max_energy)

        # colorbar
        cb_ax = plt.axes([0.25, 0, 0.5, 0.03], frameon=False)
        vals = np.linspace(min_energy, max_energy, 100)
        bar = matplotlib.colorbar.ColorbarBase(
            cb_ax,
            values=vals,
            norm=norm_E,
            cmap=cm,
            orientation="horizontal",
            drawedges=False,
            alpha=1,
        )
        # bar.ax.get_children()[1].set_linewidth(0)
        bar.set_label("UHECR Energy / EeV")

    def plot_skymap(
        self: Self,
        save: bool = False,
        file_path: Union[str, None] = None,
        cmap: str = "viridis",
    ) -> AllSkyMap:
        """
        Plot the data on a map of the sky.

        A quick way to check the data is loaded correctly.

        Parameters
        ----------
        save: bool
            flag to save figure or not
        file_path: str, default=None
            path to save the figure
        cmap: str, default="viridis"
            color map for the plot

        Returns
        -------
        a skymap object that we can add more information to
        """
        # plot style
        cm = plt.cm.get_cmap(cmap)

        # skymap
        skymap = AllSkyMap()
        skymap.fig.set_size_inches(12, 6)

        # uhecr object
        if self.uhecr:
            self.uhecr.plot_skymap(skymap)

        # source object
        if self.source:
            self.source.plot_skymap(skymap)

        # detector object
        if self.detector:
           self.detector.draw_exposure_lim(skymap)

        # standard labels and background
        skymap.set_gridlines()

        # legend
        leg = skymap.ax.legend(frameon=False, bbox_to_anchor=(0.85, 0.85))

        # add a colorbar if uhecr objects plotted
        if self.uhecr and self.uhecr.N != 1:
            self.__generate_uhecr_colorbar(cm)

        if save:
            skymap.fig.savefig(
                file_path,
                dpi=500,
                bbox_extra_artists=[leg],
                bbox_inches="tight",
                pad_inches=0.5,
            )

        return skymap

    def load_from_analysis_file(self: Self, filename: str) -> None:
        """
        Load data from an Analysis output file.

        Parameters
        ----------
        filename: str
            file name of the Analysis output file.
        """
        # Read out information on data and detector
        uhecr_properties = {}
        source_properties = {}
        detector_properties = {}
        with h5py.File(filename, "r") as f:
            uhecr = f["uhecr"]

            for key in uhecr:
                uhecr_properties[key] = uhecr[key][()]

            source = f["source"]

            for key in source:
                source_properties[key] = source[key][()]

            detector = f["detector"]

            for key in detector:
                detector_properties[key] = detector[key][()]

        uhecr = Uhecr()
        uhecr.load_from_properties(uhecr_properties)

        source = Source()
        source.load_from_properties(source_properties)

        detector = Detector(detector_properties)

        # Add to data object
        self.uhecr = uhecr
        self.source = source
        self.detector = detector


'''Below function is not used, but kept for backwards compatibility'''
class RawData:
    """Parses information for known data files in txt format."""

    def __init__(self, filename: str, filelayout: list) -> None:
        """
        Parse information for known data files in txt format.

        Parameters
        ----------
        filename: str
            name of the file to parse
        filelayout: list
            list of column names in file
        """
        self._filename = filename
        self._filelayout = filelayout
        self._data = self._parse()

    def _parse(self) -> dict:
        """
        Parse the data form the object's file.

        Returns
        -------
        dictionary of array for each column in the data file
        """
        output = pd.read_csv(
            self._filename, comment="#", delim_whitespace=True, names=self._filelayout
        )

        output_dict = output.to_dict()

        return output_dict

    def get_by_name(self, name: str) -> np.array:
        """
        Get data entries by name.

        Parameters
        ----------
        name: str
            name of the data as in self._filelayout

        Returns
        -------
        an array of data entries
        """
        try:
            selected_data = np.array(list(self._data[name].values()))

        except ValueError:
            print("No data of type", name)
            selected_data = []

        return selected_data
