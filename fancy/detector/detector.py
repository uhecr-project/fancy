import typing

import h5py
import matplotlib
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from matplotlib import pyplot as plt
from scipy import integrate
from typing_extensions import Self

from fancy.detector.exposure import m_dec, m_integrand
from fancy.plotting import AllSkyMapCartopy as AllSkyMap

__all__ = ["Detector"]


class Detector:
    """UHECR observatory information and instrument response."""

    __detector_labels : typing.ClassVar[tuple] = (
        "TA2015", "auger2022", "auger2014", "auger2010"
    )

    __hadr_models: typing.ClassVar[dict] = {
        "EPOS-LHC": 0,
        "SIBYLL2.3": 1,
    }

    __det_properties: typing.ClassVar[tuple] = (
        "label",
        "lat",
        "lon",
        "height",
        "theta_m",
        "kappa_d",
        "f_E",
        "A",
        "alpha_T",
        "start_year",
        "period_start",
        "Eth",
    )

    __view_options: typing.ClassVar[list] = ["map", "decplot"]

    def __init__(self: Self, label : str) -> None:
        """
        UHECR observatory information and instrument response.

        Parameters
        ----------
        label : str
            label of the detector
        """
        self.label = label
        self.properties = self.__get_detector_properties(label)

        # assert all keys are present in detector properties
        assert all(key in self.properties for key in self.__det_properties)

        self.label = self.properties["label"]

        # if read from h5 file, convert bytestr to str
        if isinstance(self.label, bytes):
            self.label = self.label.decode("UTF-8")

        # uncertainty information
        # See Equation 9 in Capel & Mortlock (2019)
        self.kappa_d = self.properties["kappa_d"]
        self.coord_uncertainty = np.sqrt(7552.0 / self.kappa_d)

        self.energy_uncertainty = self.properties["f_E"]
        self.Eth = self.properties["Eth"]
        self.hadr_model = None  # default hadronic interaction model
        self.lnA_params = None  # parameters for lnA fit

        # timing information
        self.start_year = self.properties["start_year"]
        self.period_start = self.properties["period_start"]

        # store other variables as none
        self.location = None
        self.threshold_zenith_angle = None
        self.area = None
        self.alpha_T = None
        self.params = None
        self.M = None
        self.declination = None
        self.exposure_max = None
        self.exposure_factor = None
        self.limiting_dec = None


    def __get_detector_properties(self : Self, label : str) -> dict:
        """
        Import the detector properties from the appropriate label.

        Parameters
        ----------
        label : str
            label of the detector

        Returns
        -------
        dict of the detector properties
        """
        # assert that the labels are within the defined labels
        assert label in self.__detector_labels, f"Detector label {label} not defined."

        if label == "TA2015":
            from fancy.detector.TA2015 import detector_properties
        elif label == "auger2022":
            from fancy.detector.auger2022 import detector_properties
        elif label == "auger2014":
            from fancy.detector.auger2014 import detector_properties
        elif label == "auger2010":
            from fancy.detector.auger2010 import detector_properties
        
        return detector_properties
        

    def get_exposure_properties(
        self: Self, num_points: int = 500
    ) -> None:
        """
        Calculate the exposure for a given detector location.

        Parameters
        ----------
        detector_properties: dict
            dictionary of detector properties
        num_points: int
            number of points to evaluate the exposure at
        """
        # location of detector
        lat = self.properties["lat"]  # radians
        lon = self.properties["lon"]  # radians
        height = self.properties["height"]  # metres

        self.location = EarthLocation(
            lat=lat * u.rad, lon=lon * u.rad, height=height * u.m
        )

        # in radians
        self.threshold_zenith_angle = self.properties["theta_m"] * u.rad
        print( self.threshold_zenith_angle)

        self.area = self.properties["A"]  # km^2
        self.alpha_T = self.properties["alpha_T"]  # km^2 sr yr

        self.params = [
            np.cos(self.location.lat.rad),
            np.sin(self.location.lat.rad),
            np.cos(self.threshold_zenith_angle.to_value("rad")),
        ]

        self.M, _ = integrate.quad(m_integrand, 0, np.pi, args=self.params)

        self.params.append(self.alpha_T)
        self.params.append(self.M)

        # define a range of declination to evaluate the
        # exposure at
        self.declination = np.linspace(-np.pi / 2, np.pi / 2, num_points)

        m = np.asarray([m_dec(d, self.params) for d in self.declination])

        self.exposure_max = np.max(m)

        # normalise to a maximum at 1
        # max value of exposure factor is normalization constant
        self.exposure_factor = m / self.exposure_max

        # find the point at which the exposure factor is 0
        # indexing value depends on TA or PAO
        # since TA only sees from dec ~ -10deg,
        # PAO only sees until dec ~ +45 deg
        declim_index = -1 if self.label.find("TA") != -1 else 0
        self.limiting_dec = (self.declination[m == 0])[declim_index] * u.rad

    def set_lnA_params(
        self: Self, meanlnA_file: str, hadr_model: str = "EPOS-LHC"
    ) -> None:
        """Set the fit parameters that fit mean lnA with logE."""
        self.hadr_model = hadr_model  # set this as the object
        self.lnA_params = np.zeros((2, 2))  # ((mean/sigma), (slope & intercept))

        self.lnA_params[0, :] = np.genfromtxt(meanlnA_file, usecols=(1, 2))[
            self.__hadr_models[hadr_model], :
        ]
        self.lnA_params[1, :] = np.array(
            [0, 0.5]
        )  # set it constant for now. TODO: We can also optionally read them from the resutls


    def save(self : Self, file_handle : h5py.File) -> None:
        """
        Save to the passed H5py file handle.

        i.e. something that cna be used with
        file_handle.create_dataset()

        file_handle: h5py.File
            file handle
        """
        for key, value in self.properties.items():
            if key == "period_start":
                continue
            file_handle.create_dataset(key, data=value)


    def plot_skymap(
        self : Self,
        view : str="map",
        coord: str = "gal",
        save: bool = False,
        file_path: typing.Union[str, None] = None,
        cmap: str = "viridis",
    ) -> None:
        """
        Make a plot of the detector's exposure.

        Parameters
        ----------
        view: a keyword describing how to show the plot
                     options are described by self._view_options
        save: boolean input, if True, the figure is saved
        savename: location to save to, required if save is
                         True
        """
        # plot style
        cm = plt.cm.get_cmap(cmap)

        if view not in self.__view_options:
            print("ERROR:", "view option", view, "is not defined")
            return

        # sky map
        if view == self.__view_options[0]:
            # skymap
            skymap = AllSkyMap()
            skymap.fig.set_size_inches(12, 6)

            # define RA and DEC over all coordinates
            rightascensions = np.linspace(-np.pi, np.pi, 500)
            declinations = self.declination

            norm_proj = matplotlib.colors.Normalize(
                self.exposure_factor.min(), self.exposure_factor.max()
            )

            # plot the exposure map
            # NB: use scatter as plot and pcolormesh have bugs in shiftdata methods
            for dec, proj in np.nditer([declinations, self.exposure_factor]):
                decs = np.tile(dec, 500)
                c = SkyCoord(ra=rightascensions * u.rad, dec=decs * u.rad, frame="icrs")

                if coord == "gal":
                    lon = c.galactic.l.deg
                    lat = c.galactic.b.deg
                elif coord == "eq":
                    lon = c.ra.degree
                    lat = c.dec.degree
                else:
                    raise Exception("Coordinate {0} is not defined.".format(coord))

                skymap.scatter(
                    lon,
                    lat,
                    linewidth=3,
                    color=cmap(norm_proj(proj)),
                    alpha=0.7,
                )

            # plot exposure boundary
            self.draw_exposure_lim(skymap, coord=coord)

            # add labels
            skymap.draw_standard_labels()

            # add colorbar
            self.__generate_exposure_colorbar(cm)

        # decplot
        elif view == self.__view_options[1]:
            # plot for all decs

            fig, ax = plt.subplots()
            ax.plot(self.declination, self.exposure_factor, linewidth=5, alpha=0.7)
            ax.set_xlabel("$\delta$")
            ax.set_ylabel("m($\delta$)")

        if save:
            fig.savefig(file_path, dpi=1000, bbox_inches="tight", pad_inches=0.5)

    def __generate_exposure_colorbar(self : Self, cm : matplotlib.colors.Colormap) -> None:
        """
        Plot a colorbar for the exposure map.

        cm: matplotlib cmap object
        """
        cb_ax = plt.axes([0.25, 0, 0.5, 0.03], frameon=False)
        vals = np.linspace(self.exposure_factor.min(), self.exposure_factor.max(), 100)

        norm_proj = matplotlib.colors.Normalize(
            self.exposure_factor.min(), self.exposure_factor.max()
        )

        bar = matplotlib.colorbar.ColorbarBase(
            cb_ax,
            values=vals,
            norm=norm_proj,
            cmap=cm,
            orientation="horizontal",
            drawedges=False,
            alpha=1,
        )

        bar.ax.get_children()[1].set_linewidth(0)
        bar.set_label("Relative exposure")

    def draw_exposure_lim(self : Self, skymap: AllSkyMap, coord: str = "gal") -> None:
        """
        Draw a line marking the edge of the detector's exposure.

        Parameters
        ----------
        skymap: an AllSkyMap instance.
        coord : str
            coordinate system to plot in
        """
        rightascensions = np.linspace(-180, 180, self.num_points)
        limiting_dec = self.limiting_dec.deg
        boundary_decs = np.tile(limiting_dec, self.num_points)
        c = SkyCoord(
            ra=rightascensions * u.degree, dec=boundary_decs * u.degree, frame="icrs"
        )
        if coord == "gal":
            lon = c.galactic.l.deg
            lat = c.galactic.b.deg
        elif coord == "eq":
            lon = c.ra.degree
            lat = c.dec.degree
        else:
            raise Exception("Coordinate {0} is not defined.".format(coord))

        skymap.scatter(
            lon,
            lat,
            s=8,
            color="grey",
            alpha=1,
            label="Limit of " + self.label[:-4] + "'s exposure",
            zorder=1,
        )


# if __name__ == "__main__":
#     # import auger2014 data
#     from fancy.detector.auger2014 import detector_properties

#     # create Detector object
#     detector = Detector(detector_properties)

#     # show the exposure skymap
#     detector.show(view="map", coord="gal")
