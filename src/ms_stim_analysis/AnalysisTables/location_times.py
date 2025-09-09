import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable

from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.position import PositionOutput
from spyglass.common import interval_list_contains_ind

import warnings

warnings.filterwarnings("ignore")

schema = dj.schema("ms_locations")


@schema
class LocationTimesParams(SpyglassMixin, dj.Manual):
    definition = """
    location_params_name: varchar(64)  # name of the location params
    ---
    centers: longblob  # centers of the locations
    radii: longblob # radii of the locations
    """


@schema
class LocationTimesSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> LocationTimesParams
    -> PositionOutput
    ---
    """


@schema
class LocationTimes(SpyglassMixin, dj.Computed):
    definition = """
    -> LocationTimesSelection
    ---
    intervals: longblob  # intervals of time spent in location
    """

    def make(self, key):
        pos_df = (PositionOutput & key).fetch1_dataframe()
        loc_params = (LocationTimesParams & key).fetch1()
        centers = loc_params["centers"]
        radii = loc_params["radii"]
        if not isinstance(centers, Iterable):
            centers = [centers]
        if not isinstance(radii, Iterable):
            radii = [radii]

        contained = (
            np.array(
                [
                    self._in_location(
                        pos_df.position_x, pos_df.position_y, center, radius
                    )
                    for center, radius in zip(centers, radii)
                ]
            )
            .any(axis=0)
            .astype(int)
        )

        enter_times = np.where(np.diff(contained) == 1)[0]
        exit_times = np.where(np.diff(contained) == -1)[0]

        if contained[0]:
            enter_times = np.concatenate([np.array([0]), enter_times])
        if contained[-1]:
            exit_times = np.concatenate([exit_times, np.array([len(contained) - 1])])

        time = pos_df.index.values
        intervals = np.array(
            [
                [time[i_enter], time[i_exit]]
                for i_enter, i_exit in zip(enter_times, exit_times)
            ]
        )
        self.insert1({**key, "intervals": intervals})

    @staticmethod
    def _in_location(x, y, center, radius):
        return np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) < radius

    def validate_location(self, key=dict()):
        key = (self & key).fetch1("KEY")
        pos_df = (PositionOutput & key).fetch1_dataframe()
        loc_params = (LocationTimesParams & key).fetch1()
        centers = loc_params["centers"]
        radii = loc_params["radii"]
        if not isinstance(centers, Iterable):
            centers = [centers]
        if not isinstance(radii, Iterable):
            radii = [radii]

        contained_intervals = (self & key).fetch1("intervals")

        fig, ax = plt.subplots()
        contained_indices = interval_list_contains_ind(
            contained_intervals,
            pos_df.index.values,
        )
        plt.scatter(pos_df.position_x, pos_df.position_y, c="cornflowerblue", s=10)
        plt.scatter(
            pos_df.position_x.iloc[contained_indices],
            pos_df.position_y.iloc[contained_indices],
            c="firebrick",
            s=10,
        )
        circles = [
            plt.Circle(center, radius, color="red", fill=True, alpha=0.3)
            for center, radius in zip(centers, radii)
        ]
        for circle in circles:
            ax.add_artist(circle)
