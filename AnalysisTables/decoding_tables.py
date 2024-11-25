import datajoint as dj
import pandas as pd
import numpy as np

from .place_fields import TrackCellCoverage

from non_local_detector.visualization import create_interactive_1D_decoding_figurl
from spyglass.common import AnalysisNwbfile, interval_list_intersect
from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1
from spyglass.utils import SpyglassMixin
from spyglass.ripple.v1 import RippleTimesV1
from spyglass.lfp.analysis.v1 import LFPBandV1


schema = dj.schema("ms_decoding")


# @schema
# class ClusterlessDecodingFigurl_1D(SpyglassMixin, dj.Computed):
#     definition = """
#     -> ClusterlessDecodingV1
#     ---
#     figurl: varchar(3000)
#     """

#     def make(self, key):
#         position_info = ClusterlessDecodingV1.fetch_linear_position_info(key)
#         decoding_results = (ClusterlessDecodingV1 & key).fetch_results()
#         results_time = decoding_results.acausal_posterior.isel(intervals=0).time.values
#         position_info = position_info.loc[results_time[0] : results_time[-1]]
#         spikes, _ = ClusterlessDecodingV1.fetch_spike_data(key)
#         figurl = create_interactive_1D_decoding_figurl(
#             position=position_info["linear_position"],
#             speed=position_info["speed"],
#             spike_times=spikes,
#             results=decoding_results.squeeze(),
#         )
#         key["figurl"] = figurl
#         self.insert1(key)


# @schema
# class SortedSpikesDecodingFigurl_1D(SpyglassMixin, dj.Computed):
#     definition = """
#     -> SortedSpikesDecodingV1
#     ---
#     figurl: varchar(3000)
#     """

#     def make(self, key):
#         position_info = SortedSpikesDecodingV1.fetch_linear_position_info(key)
#         decoding_results = (SortedSpikesDecodingV1 & key).fetch_results()
#         results_time = decoding_results.acausal_posterior.isel(intervals=0).time.values
#         position_info = position_info.loc[results_time[0] : results_time[-1]]
#         spikes = SortedSpikesDecodingV1.fetch_spike_data(key)
#         figurl = create_interactive_1D_decoding_figurl(
#             position=position_info["linear_position"],
#             speed=position_info["speed"],
#             spike_times=spikes,
#             results=decoding_results.squeeze(),
#         )
#         key["figurl"] = figurl
#         self.insert1(key)


@schema
class ClusterlessAheadBehindDistance(SpyglassMixin, dj.Computed):
    definition = """
    -> ClusterlessDecodingV1
    ---
    -> AnalysisNwbfile
    distance_object_id: varchar(128)
    """

    def make(self, key):
        distance = (ClusterlessDecodingV1() & key).get_ahead_behind_distance()
        results = (ClusterlessDecodingV1() & key).fetch_results()

        df = pd.DataFrame({"time": results.time, "decode_distance": distance})
        df = df.set_index("time")

        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key["analysis_file_name"] = analysis_file_name
        key["distance_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, df, "distance"
        )
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])

        self.insert1(key)

    def fetch1_dataframe(self) -> pd.DataFrame:
        if not len(nwb := self.fetch_nwb()):
            raise ValueError("fetch1_dataframe must be called on a single key")
        return nwb[0]["distance"]


@schema
class RippleClusterlessDecodeAnalysisSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> ClusterlessAheadBehindDistance
    -> RippleTimesV1
    acausal: bool
    ---
    """


@schema
class RippleClusterlessDecodeAnalysis(SpyglassMixin, dj.Computed):
    definition = """
    -> RippleClusterlessDecodeAnalysisSelection
    ---
    -> AnalysisNwbfile
    data_object_id: varchar(128)
    """

    def make(self, key):
        clusterless_key = (ClusterlessDecodingV1 & key).fetch1("KEY")
        ripple_key = (RippleTimesV1 & key).fetch1("KEY")
        ascausal = (RippleClusterlessDecodeAnalysisSelection & key).fetch1("acausal")
        key["acausal"] = ascausal

        # Fetch the data you need
        results = (ClusterlessDecodingV1() & clusterless_key).fetch_results()
        ripple_df = (RippleTimesV1() & ripple_key).fetch1_dataframe()
        ripple_intervals = np.array(
            [[st, en] for st, en in zip(ripple_df.start_time, ripple_df.end_time)]
        )
        if (RippleClusterlessDecodeAnalysisSelection & key).fetch1("acausal"):
            full_posterior = results.causal_posterior.unstack("state_bins")
        else:
            full_posterior = results.causal_posterior.unstack("state_bins")
        posterior = full_posterior.sum("state")[0]
        decode_pos = posterior.idxmax("position").values
        state_posterior = full_posterior.sum("position")[0]

        linear_df = ClusterlessDecodingV1().fetch_linear_position_info(clusterless_key)
        decode_distance = (
            (ClusterlessAheadBehindDistance() & clusterless_key).fetch1_dataframe()
        ).values[:, 0]

        band_df = (LFPBandV1() & ripple_key).fetch1_dataframe()

        ripple_data = []
        for interval in ripple_intervals:
            ind = np.logical_and(
                state_posterior.time.values > interval[0],
                state_posterior.time.values < interval[1],
            )
            ripple_distance = decode_distance[ind]
            interval_decode_pos = decode_pos[ind]

            ind_pos = np.logical_and(
                linear_df.index.values > interval[0],
                linear_df.index.values < interval[1],
            )
            pos = linear_df.linear_position.values[ind_pos]
            if np.any(np.isnan(pos)):
                # skip intervals with undefined position
                continue

            interval_state_posterior = state_posterior.sel(
                time=slice(interval[0], interval[1])
            ).values
            interval_pos_posterior = posterior.sel(
                time=slice(interval[0], interval[1])
            ).values

            ind_band = np.logical_and(
                band_df.index.values > interval[0], band_df.index.values < interval[1]
            )
            interval_band = band_df.values[ind_band]

            data_i = {
                "interval": interval,
                "distance": ripple_distance,
                "position": pos,
                "decode_position": interval_decode_pos,
                "state_posterior": interval_state_posterior,
                "pos_posterior": interval_pos_posterior,
                "ripple_band": interval_band,
            }
            ripple_data.append(data_i)

        ripple_data = pd.DataFrame(ripple_data)
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key["analysis_file_name"] = analysis_file_name
        key["data_object_id"] = AnalysisNwbfile().add_nwb_object(
            key["analysis_file_name"], ripple_data, "ripple_data"
        )
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
        self.insert1(key)

    def fetch1_dataframe(self) -> pd.DataFrame:
        if not len(nwb := self.fetch_nwb()):
            raise ValueError("fetch1_dataframe must be called on a single key")
        return nwb[0]["data"]

    def fetch_dataframe(self) -> pd.DataFrame:
        return pd.concat([nwb["data"] for nwb in self.fetch_nwb()])

    def classify_ripple_decode(
        self,
        key={},
        thresh_percent: float = 0.8,
        thresh_ripple_fraction=0.5,
        locality_threshold=40,
        class_names=[
            "Continuous",
            "Fragmented",
        ],
        valid_intervals=None,
        valid_interval_fraction=1.0,
    ):
        """Classify ripples based on decoding results

        Args:
            key (dict, optional): Restriction of results to analysze. Defaults to {}.
            thresh_percent (float, optional): percent threshold for a time to be defined as a given state. Defaults to 0.8.
            thresh_ripple_fraction (float, optional): fraction of a ripple that must be a defined state to categorize the ripple. Defaults to 0.5.
            locality_threshold (int, optional): ripples with a mean decode distance less than this classified as local. Defaults to 40.
            class_names (list, optional): names of the state variables. Defaults to [ "Continuous", "Fragmented", ].
            valid_intervals (np.ndarray, optional): interval list to restrict ripple analysis to. Defaults to None.
            valid_interval_fraction (float, optional): fraction of the interval that must be in the valid intervals to be included in the analysis. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        data_df = (self & key).fetch_dataframe()
        state_decode = data_df.state_posterior.values
        distance = data_df.distance.values
        ripple_intervals = data_df.interval.values
        ripple_intervals = np.array([r for r in ripple_intervals])
        if valid_intervals is not None:
            # computational efficiency for large number of ripples and intervals
            inter_range_start_list = (
                np.digitize(ripple_intervals[:, 0], valid_intervals[:, 0]) - 1
            )
            inter_range_start_list[inter_range_start_list < 0] = 0
            inter_range_end_list = (
                np.digitize(ripple_intervals[:, 1], valid_intervals[:, 1]) + 1
            )

        ripple_class = []

        for i, (interval, interval_distance, interval_posterior) in enumerate(
            zip(ripple_intervals, distance, state_decode)
        ):
            if valid_intervals is not None:
                # int_s = min(0, np.digitize(interval[0], valid_intervals[:, 0]) - 1)
                # int_e = np.digitize(interval[1], valid_intervals[:, 1]) + 1
                int_s = inter_range_start_list[i]
                int_e = inter_range_end_list[i]
                ripple_valid = interval_list_intersect(
                    np.array([interval]), valid_intervals[int_s:int_e]
                )
                ripple_length = interval[1] - interval[0]
                valid_length = (
                    np.sum(np.diff(ripple_valid, axis=1).squeeze())
                    if len(ripple_valid) > 0
                    else 0
                )
                if valid_length / ripple_length < valid_interval_fraction:
                    ripple_class.append("Invalid")
                    continue

            if np.all(np.isnan(interval_posterior)):
                ripple_class.append("NAN")
                continue
            if np.mean(interval_distance) < locality_threshold:
                ripple_class.append("Local")
                continue

            interval_posterior = interval_posterior > thresh_percent
            fract_time_state = np.mean(interval_posterior, axis=0)

            if np.any(fract_time_state > thresh_ripple_fraction):
                ripple_class.append(class_names[np.argmax(fract_time_state)])
            else:
                ripple_class.append("Mixed")

        return np.array(ripple_class)


@schema
class BadDecodes(dj.Computed):
    definition = """
    -> ClusterlessDecodingV1
    ---
    bad_decodes: bool
    """

    def make(self, key):
        result = (ClusterlessDecodingV1 & key).fetch_results()
        if np.mean(np.isnan(result.acausal_posterior.values)) > 0.5:
            key["bad_decodes"] = True
        else:
            key["bad_decodes"] = False
        self.insert1(key)
