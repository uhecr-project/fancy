from datetime import date, timedelta

import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from typing_extensions import Self

from fancy.plotting import AllSkyMapCartopy as AllSkyMap
from fancy.utils.coordinates import get_coordinates, uv_to_coord
from fancy.utils.package_data import get_path_to_datafiles

__all__ = ["Uhecr"]


class Uhecr:
    """Stores the data and parameters for UHECRs."""

    def __init__(self: Self) -> None:
        """Initialise empty container that contains the UHECR information."""
        self.properties = None
        self.source_labels = None
        self.label = None

        # stubs for data
        self.year = None
        self.day = None
        self.zenith_angle = None
        self.energy = None
        self.N = None
        self.coord = None
        self.exposure = None
        self.unit_vector = None
        self.period = None
        self.A = None

        # stubs for gmf-related information
        self.mass_model = None
        self.coords_gb = None
        self.unit_vector_gb = None
        self.kappa_gmfs = None

    def __get_angular_uncertainty(self: Self) -> float:
        """Get angular reconstruction uncertainty from label."""
        if self.label == "TA2015":
            from fancy.detector.TA2015 import sig_omega
        elif self.label == "auger2014":
            from fancy.detector.auger2014 import sig_omega
        elif self.label == "auger2010":
            from fancy.detector.auger2010 import sig_omega
        else:
            raise Exception("Undefined detector type!")

        return np.deg2rad(sig_omega)

    def load_from_data_file(
        self: Self,
        label: str,
        mass_model: str = "EPOS-LHC",
        gmf_model: str = "JF12",
        filename: str = "UHECRdata.h5",
    ) -> None:
        """
        Define UHECR from data file of original information.

        Handles calculation of observation periods and
        effective areas assuming the UHECR are detected
        by the Pierre Auger Observatory or TA.

        label: str
            reference label for the UHECR data set
        hadr_model: str
            label for hadronic interaction model
        gmf_model: str
            label for GMF model
        filename: str
            name of the data file

        """
        self.label = label

        # use the default path for the datafiles if 
        # uhecr data is not simulation
        if filename.find("sim") < 0:
            path_to_data = get_path_to_datafiles(filename)
        else:
            path_to_data = filename

        with h5py.File(path_to_data, "r") as f:
            data = f[self.label]

            # timing & angle information
            self.year = data["year"][()]
            self.day = data["day"][()]
            self.zenith_angle = np.deg2rad(data["theta"][()])

            # energy information
            self.energy = data["energy"][()]

            # read in rigidity if it exists (e.g. from simulation)
            if "rigidity" in data:
                self.rigidity = data["rigidity"][()]
            self.N = len(self.energy)

            # arrival directions, read from glon / glat
            glon = data["glon"][()]
            glat = data["glat"][()]
            self.coord = get_coordinates(glon, glat)  # convert to skycoord

            # check if we can extract exposure of UHECR (auger2022 dataset)
            if "exposure" in data:
                self.exposure = data["exposure"][()]
            else:
                self.exposure = np.ones(self.N)

            # unit vector in cartesian coordinates
            self.unit_vector = self.coord.cartesian.xyz.value.T

            # period & effective area for observation.
            # kept only for backwards compatibility
            self.period = self.__find_period()
            self.A = self.__find_area()

            # represents the angular uncertainty per UHECR
            self.kappa_ds = data["kappa_ds"][()]

            self.mass_model = mass_model  # TODO: check why we need this
            # reading in GMF information
            # first check if
            if "gmf" in data and gmf_model != "None":
                # only read if data exists for both GMF model key and hadr model key
                config_key = f"{gmf_model}_{mass_model}"
                if config_key not in list(data["gmf"].keys()):
                    raise KeyError(
                        f"GMF data for configuration {gmf_model}, {mass_model} is not found."
                    )

                glons_gb = data["gmf"][config_key]["glons_gb"][()]
                glats_gb = data["gmf"][config_key]["glats_gb"][()]
                self.coords_gb = get_coordinates(glons_gb, glats_gb)
                self.unit_vector_gb = self.coords_gb.cartesian.xyz.value.T
                self.kappa_gmfs = data["gmf"][config_key]["kappa_gmf"][
                    ()
                ]  # deflection parameter

                # read the exposure factors
                if "exposure_factor" in data["gmf"][config_key]:
                    self.exposure = data["gmf"][config_key][
                        "exposure_factor"
                    ][()]

                # set the deflection parameter to the kappa_GMFs
                self.kappa_ds = self.kappa_gmfs

    def __get_properties(self, analysis_type: str) -> dict:
        """Pack all relevant UHECR object infomration to a dictionary."""
        properties = {}
        properties["label"] = self.label
        properties["N"] = self.N
        properties["unit_vector"] = self.unit_vector
        properties["energy"] = self.energy
        properties["A"] = self.A
        properties["zenith_angle"] = self.zenith_angle

        if analysis_type == "joint_gmf_composition":
            properties["hadr_model"] = self.hadr_model
            properties["kappa_gmf"] = self.kappa_gmfs
            properties["unit_vector_gb"] = self.unit_vector_gb

        # Only if simulated UHECRs
        # if isinstance(self.source_labels, (list, np.ndarray)):
        #     self.properties['source_labels'] = self.source_labels

        return properties

    def load_from_properties(self: Self, uhecr_properties: dict) -> None:
        """
        Define UHECR from properties dict.

        Parameters
        ----------
        uhecr_properties: dict
            dict containing UHECR properties.
        """
        self.label = uhecr_properties["label"]

        # Read from input dict
        self.unit_vector = uhecr_properties["unit_vector"]
        self.energy = uhecr_properties["energy"]
        self.N = len(self.energy)

        self.exposure = uhecr_properties["exposure"] if "exposure" in uhecr_properties else np.ones(self.N)

        self.zenith_angle = uhecr_properties["zenith_angle"]
        self.year = uhecr_properties["years"]
        self.day = uhecr_properties["days"]

        self.period = self.__find_period()
        self.A = self.__find_area()


        # Only if simulated UHECRs
        # try:
        #     self.source_labels = uhecr_properties['source_labels']
        # except:
        #     pass

        # Get SkyCoord from unit_vector
        self.coord = uv_to_coord(self.unit_vector)

    def plot_skymap(self: Self, skymap: AllSkyMap, size: int = 2) -> None:
        """
        Plot the Uhecr instance on a skymap.

        Called by Data.plot_skymap()

        Parameters
        ----------
        skymap: AllSkyMapCartopy
            the AllSkyMap
        size: float
            tissot radius
        """
        lons = self.coord.galactic.l.deg
        lats = self.coord.galactic.b.deg

        alpha_level = 0.7

        # If source labels are provided, plot with colour
        # indicating the source label.
        if isinstance(self.source_labels, (list, np.ndarray)):
            Nc = max(self.source_labels)

            # Use a continuous cmap
            cmap = plt.cm.get_cmap("plasma", Nc)

            write_label = True

            for lon, lat, lab in np.nditer([lons, lats, self.source_labels]):
                color = cmap(lab)
                if write_label:
                    skymap.tissot(
                        lon,
                        lat,
                        size,
                        npts=30,
                        color=color,
                        lw=0,
                        alpha=0.5,
                        label=self.label,
                    )
                    write_label = False
                else:
                    (
                        skymap.tissot(
                            lon, lat, size, npts=30, color=color, lw=0, alpha=0.5
                        ),
                    )

        # Otherwise, use the cmap to show the UHECR energy.
        else:
            # use colormap for energy
            norm_E = matplotlib.colors.Normalize(min(self.energy), max(self.energy))
            cmap = plt.cm.get_cmap("viridis", len(self.energy))

            write_label = True
            for E, lon, lat in np.nditer([self.energy, lons, lats]):
                color = cmap(norm_E(E))

                if write_label:
                    skymap.tissot(
                        lon,
                        lat,
                        size,
                        npts=30,
                        color=color,
                        lw=0,
                        alpha=alpha_level,
                        label=self.label,
                    )
                    write_label = False
                else:
                    skymap.tissot(
                        lon,
                        lat,
                        size,
                        npts=30,
                        color=color,
                        lw=0,
                        alpha=alpha_level,
                    )

    def save(self: Self, file_handle: h5py.File, analysis_type: str) -> None:
        """
        Save to the passed H5py file handle.

        i.e. something that can be used with
        file_handle.create_dataset()

        Parameters
        ----------
        file_handle: h5py.File
            h5py file handle to save data to.
        analysis_type: str
            type of analysis from Analysis object.
        """
        properties = self.__get_properties(analysis_type)

        for key, value in properties.items():
            file_handle.create_dataset(key, data=value)

    def __find_area(self: Self, exp_factor: float = 1.0) -> list:
        """
        Find the effective area of the observatory at the time of detection.

        Possible areas are calculated from the exposure reported
        in Abreu et al. (2010) or Collaboration et al. 2014.
        """
        if self.label == "auger2010":
            from ..detector.auger2010 import A1, A2, A3

            possible_areas = [A1, A2, A3]
            area = [possible_areas[i - 1] * exp_factor for i in self.period]

        elif self.label == "auger2014":
            from ..detector.auger2014 import (
                A1,
                A2,
                A3,
                A4,
                A1_incl,
                A2_incl,
                A3_incl,
                A4_incl,
            )

            possible_areas_vert = [A1, A2, A3, A4]
            possible_areas_incl = [A1_incl, A2_incl, A3_incl, A4_incl]

            # find area depending on period and incl
            area = []
            for i, p in enumerate(self.period):
                if self.zenith_angle[i] <= 60:
                    area.append(possible_areas_vert[p - 1] * exp_factor)
                if self.zenith_angle[i] > 60:
                    area.append(possible_areas_incl[p - 1] * exp_factor)

        elif "auger2022" in self.label:
            from ..detector.auger2022 import A, M, period_start

            # get period for each event - in years, taking into account days
            start_julianyear = period_start.year + period_start.day / 365.25
            deltats = (self.year + self.day / 365.25) - start_julianyear

            # very hacky, but only exists currently for backwards compatibility anyways
            if len(self.exposure) > 0:
                area = self.exposure / (M * deltats)
            else:
                area = np.tile(A, self.N)
            area = np.tile(A, self.N)

        elif "TA2015" in self.label:
            from ..detector.TA2015 import A1, A2

            possible_areas = [A1, A2]
            area = [possible_areas[i - 1] * exp_factor for i in self.period]

        else:
            print("Effective areas and periods not defined. Setting uniform.")
            area = np.ones(len(self.period))

        return area

    def __find_period(self: Self) -> list:
        """
        For a given year or day, find UHECR period.

        Dates are based on dates in table 1 in Abreu et al. (2010) or in Collaboration et al. 2014.
        """
        period = []
        if self.label == "auger2014":
            from ..detector.auger2014 import (
                period_1_end,
                period_1_start,
                period_2_end,
                period_2_start,
                period_3_end,
                period_3_start,
            )

            # check dates
            for y, d in np.nditer([self.year, self.day]):
                d = int(d)
                test_date = date(y, 1, 1) + timedelta(d)

                if period_1_start <= test_date <= period_1_end:
                    period.append(1)
                elif period_2_start <= test_date <= period_2_end:
                    period.append(2)
                elif period_3_start <= test_date <= period_3_end:
                    period.append(3)
                elif test_date >= period_3_end:
                    period.append(4)
                else:
                    print("Error: cannot determine period for year", y, "and day", d)

        elif self.label == "TA2015":
            from ..detector.TA2015 import (
                period_1_end,
                period_1_start,
                period_2_end,
                period_2_start,
            )

            for y, d in np.nditer([self.year, self.day]):
                d = int(d)
                test_date = date(
                    y, period_1_start.month, period_1_start.day
                ) + timedelta(d)

                if period_1_start <= test_date <= period_1_end:
                    period.append(1)
                elif period_2_start <= test_date <= period_2_end:
                    period.append(2)
                elif test_date >= period_2_end:
                    period.append(2)
                else:
                    print("Error: cannot determine period for year", y, "and day", d)

        else:
            print(f"no period data found for {self.label}. setting uniform")
            period = np.ones(len(self.year), dtype=int)

        return period

    def select_from_period(self: Self, period: list) -> None:
        """Select certain periods for analysis, other periods will be discarded."""
        # find selected periods
        if len(period) == 1:
            selection = np.where(np.asarray(self.period) == period[0])
        if len(period) == 2:
            selection = np.concatenate(
                [
                    np.where(np.asarray(self.period) == period[0]),
                    np.where(np.asarray(self.period) == period[1]),
                ],
                axis=1,
            )

        # keep things as lists
        selection = selection[0].tolist()

        # make selection
        self.A = [self.A[i] for i in selection]
        self.period = [self.period[i] for i in selection]
        self.energy = [self.energy[i] for i in selection]
        self.incidence_angle = [self.incidence_angle[i] for i in selection]
        self.unit_vector = [self.unit_vector[i] for i in selection]

        self.N = len(self.period)

        self.day = [self.day[i] for i in selection]
        self.year = [self.year[i] for i in selection]

        self.coord = self.coord[selection]

    def select_from_energy(self: Self, Eth: float) -> None:
        """Select out only UHECRs above a certain energy."""
        selection = np.where(np.asarray(self.energy) >= Eth)
        selection = selection[0].tolist()

        # make selection
        self.A = [self.A[i] for i in selection]
        self.period = [self.period[i] for i in selection]
        self.energy = [self.energy[i] for i in selection]
        self.incidence_angle = [self.incidence_angle[i] for i in selection]
        self.unit_vector = [self.unit_vector[i] for i in selection]

        self.N = len(self.period)

        self.day = [self.day[i] for i in selection]
        self.year = [self.year[i] for i in selection]

        self.coord = self.coord[selection]
