import datajoint as dj
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr

from spyglass.common.common_interval import Interval
from spyglass.common import AnalysisNwbfile, IntervalList
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.utils.dj_mixin import SpyglassMixin


from ms_stim_analysis.Analysis.utils import get_running_valid_intervals, smooth


os.environ["JAX_PLATFORM_NAME"] = "cpu"

schema = dj.schema("sambray_compression_index")


@schema
class CompressionIndexParams(SpyglassMixin, dj.Lookup):
    definition = """
    compression_index_params_name: varchar(32)
    ---
    filter_speed: float
    min_running_spikes: int
    pf_bin_size: float  # cm
    pf_peak_ratio: float
    delay_range: int  # ms
    delay_smoothing: int  # ms
    min_coincident_spikes: int
    graph_distance = 0: bool  # whether to calculate the graph distance between place fields
    delay_distance = 80: int  # minimum distance between peaks in the delay histogram
    smoothing_sigma = NULL: float  # sigma for gaussian smoothing of the delay histogram defaults to 1/2 delay_smoothing
    """

    contents = [
        ("fast", 10, 100, 5, 10, 150, 33, 30, False, 100),
        ("default", 10, 100, 5, 10, 100, 33, 30, False, 80),
        ("slow", 10, 100, 5, 10, 1000, 250, 30, False, 1000),
        ("fast_graph_distance", 10, 100, 5, 10, 150, 33, 30, True, 100),
        ("default_graph_distance", 10, 100, 5, 10, 100, 33, 30, True, 80),
        ("slow_graph_distance", 10, 100, 5, 10, 1500, 250, 30, True, 2000),
        ("cross_method_graph_distance", 10, 100, 5, 1, 1000, 10, 30, True, 80),
    ]


@schema
class CompressionIndexSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SortedSpikesDecodingV1
    -> CompressionIndexParams
    ---
    """


@schema
class CompressionIndex(SpyglassMixin, dj.Computed):
    definition = """
    -> CompressionIndexSelection
    ---
    -> AnalysisNwbfile
    place_field_object_id: varchar(64)  # ID of the place field object
    delays_object_id: varchar(64)  # ID of the delays object

    """

    def make(self, key):
        # read params
        params = (CompressionIndexParams & key).fetch1()
        filter_speed = params["filter_speed"]
        min_running_spikes = params["min_running_spikes"]
        pf_bin_size = params["pf_bin_size"]
        pf_peak_ratio = params["pf_peak_ratio"]
        delay_range = params["delay_range"]
        delay_smoothing = params["delay_smoothing"]
        min_coincident_spikes = params["min_coincident_spikes"]
        graph_distance = params["graph_distance"]
        delay_distance = params["delay_distance"]
        smoothing_sigma = params["smoothing_sigma"]
        if smoothing_sigma is np.nan:
            smoothing_sigma = delay_smoothing / 2

        dlc_pos = "DLC" in key["position_group_name"]

        # fetch_data
        spikes, units = SortedSpikesGroup().fetch_spike_data(key, return_unit_ids=True)
        unit_ids = [f"{x['spikesorting_merge_id']}_{x['unit_id']}" for x in units]
        # Get running intervals
        pos_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["decoding_interval"],
        }
        running_intervals = get_running_valid_intervals(
            pos_key,
            filter_speed=filter_speed,
            filter_ports=True,
            seperate_optogenetics=False,
            dlc_pos=dlc_pos,
        )
        running_intervals = Interval(running_intervals)

        interval_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["encoding_interval"],
        }
        valid_interval = (IntervalList & interval_key).fetch_interval()
        running_intervals = running_intervals.intersect(valid_interval)

        # get position data
        pos_df = SortedSpikesDecodingV1().fetch_linear_position_info(key)

        # make place fields
        if graph_distance:
            # overides default pf_bin_size
            environment = SortedSpikesDecodingV1().fetch_environments(key)[0]
            pos_bins = np.squeeze(environment.place_bin_edges_)
        else:
            pos_bins = np.arange(
                pos_df.linear_position.min(),
                pos_df.linear_position.max() + 1,
                pf_bin_size,
            )
        occupancy = np.histogram(pos_df.linear_position, bins=pos_bins)[0]
        place_fields = {}
        for s, s_id in zip(spikes, unit_ids):
            s = running_intervals.contains(s)
            if len(s) < min_running_spikes:
                continue
            pos_s = pos_df.linear_position.iloc[np.digitize(s, pos_df.index.values) - 1]
            spikes_in_pos = np.histogram(pos_s, bins=pos_bins)[0]
            field = spikes_in_pos / occupancy
            field = field / np.nansum(field)

            # skip non-localized place fields
            if np.max(field) / np.median(field) < pf_peak_ratio:
                continue
            place_fields[s_id] = field

        # calculate needed delays

        delay_bins = np.arange(-2 * delay_range, 2 * delay_range, 1)
        ind_relevant = np.where(np.abs(delay_bins) <= delay_range)[0]
        # ind_relevant = np.where(np.logical_and(delay_bins >= 0, delay_bins <= delay_range))[0]
        results = []
        for s1, s_id_1 in zip(spikes, unit_ids):
            for s2, s_id_2 in zip(spikes, unit_ids):
                # skip if not good place fields
                if s_id_1 not in place_fields or s_id_2 not in place_fields:
                    continue
                # skip if same unit
                if s_id_1 == s_id_2:
                    continue
                # only need to calculate each pair once
                if s_id_1 < s_id_2:
                    continue

                field_1 = place_fields[s_id_1]
                field_2 = place_fields[s_id_2]
                field_loc_1 = np.average(pos_bins[:-1], weights=field_1)
                field_loc_2 = np.average(pos_bins[:-1], weights=field_2)
                if graph_distance:
                    # distance = self._get_field_graph_distance(
                    #     environment, field_1, field_2, pos_bins
                    # )
                    # print("HI")
                    field_loc_1 = pos_bins[np.argmax(field_1)]
                    field_loc_2 = pos_bins[np.argmax(field_2)]
                    distance = self._get_pos_graph_distance(
                        environment, field_loc_1, field_loc_2
                    )
                else:
                    distance = np.abs(field_loc_1 - field_loc_2)

                s1 = running_intervals.contains(s1)
                s2 = running_intervals.contains(s2)
                if len(s1) < min_coincident_spikes or len(s2) < min_coincident_spikes:
                    continue

                delays = np.subtract.outer(s1, s2) * 1000
                delays = np.ravel(delays)
                delays = delays[delays >= -delay_range * 2]
                delays = delays[delays <= delay_range * 2]
                if len(delays) < min_coincident_spikes:
                    continue
                cross_corr = np.histogram(delays, bins=delay_bins)[0]
                if (
                    n_coincident := cross_corr[ind_relevant].sum()
                ) < min_coincident_spikes:
                    continue
                cross_corr = smooth(
                    cross_corr, n=delay_smoothing, sigma=smoothing_sigma
                )

                # peak_delay = delay_bins[ind_relevant][np.argmax(cross_corr[ind_relevant])]
                peak_inds = find_peaks(
                    cross_corr[ind_relevant], distance=delay_distance
                )
                peak_times = delay_bins[1:][ind_relevant][peak_inds[0]]
                if len(peak_times) == 0:
                    continue
                peak_delay = peak_times[
                    np.argmin(np.abs(peak_times))
                ]  # This is the peak closest to zero

                results.append(
                    {
                        "s_id_1": s_id_1,
                        "s_id_2": s_id_2,
                        "field_loc_1": field_loc_1,
                        "field_loc_2": field_loc_2,
                        "distance": distance,
                        "cross_corr": cross_corr,
                        "peak_delay": peak_delay,
                        "n_coincidients": n_coincident,
                    }
                )

        results = pd.DataFrame(results)
        place_field_df = [
            {
                "unit_id": s_id,
                "place_field": place_fields[s_id],
            }
            for s_id in place_fields.keys()
        ]
        place_field_df = pd.DataFrame(place_field_df)

        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        place_field_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, place_field_df, "place_fields"
        )
        delays_object_id = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, results, "delays"
        )

        key["place_field_object_id"] = place_field_object_id
        key["delays_object_id"] = delays_object_id
        key["analysis_file_name"] = analysis_file_name

        AnalysisNwbfile().add(key["nwb_file_name"], analysis_file_name)
        self.insert1(key)

    def fetch_delays_dataframes(self):
        nwbs = self.fetch_nwb()
        dfs = [nwb["delays"] for nwb in nwbs]
        return pd.concat(dfs, ignore_index=True)

    def fetch_place_dataframes(self):
        nwbs = self.fetch_nwb()
        dfs = [nwb["place_field"] for nwb in nwbs]
        return pd.concat(dfs, ignore_index=True)

    def calculate_compression_index(
        self,
        min_coincident_spikes=None,
        max_field_distance=None,
        min_field_distance=None,
        return_vals=False,
    ):
        delays_df = self.fetch_delays_dataframes()
        # filter delays based on parameters
        if min_coincident_spikes is not None:
            delays_df = delays_df[delays_df["n_coincidients"] >= min_coincident_spikes]
        if max_field_distance is not None:
            delays_df = delays_df[delays_df["distance"] <= max_field_distance]
        if min_field_distance is not None:
            delays_df = delays_df[delays_df["distance"] >= min_field_distance]

        distances = delays_df["distance"].values
        peak_delays = np.abs(delays_df["peak_delay"].values)
        compression_index = pearsonr(distances, peak_delays)[0]
        if return_vals:
            return compression_index, distances, peak_delays
        return compression_index

    # --- helpers ---
    @staticmethod
    def make_distance_matrix(environment, pos_bins):
        """
        Create a distance matrix for the given position bins based on the environment's graph.
        """
        nodes_df = environment.nodes_df_
        edge_nodes_df = nodes_df[nodes_df["is_bin_edge"] == True]
        node_ids = [
            edge_nodes_df[edge_nodes_df.linear_position == x].node_id.values[0]
            for x in pos_bins
        ]

        distance_matrix = np.zeros((len(pos_bins), len(pos_bins)))

        for i, i in enumerate(node_ids):
            for j, j in enumerate(node_ids):
                distance_matrix[i, j] = environment.distance_between_nodes_[i][j]

        return distance_matrix

    def _get_field_graph_distance(self, environment, field_1, field_2, pos_bins):
        """
        Calculate the distance between two fields based on the environment's graph.
        """
        distance_score = 0
        for i, f1 in enumerate(field_1):
            for j, f2 in enumerate(field_2):
                distance_score += (
                    f1
                    * f2
                    * self._get_pos_graph_distance(
                        environment, pos_bins[i], pos_bins[j]
                    )
                )
        return distance_score

    def _get_pos_graph_distance(self, environment, pos_1, pos_2):
        """
        Calculate the distance between two positions based on the environment's distance matrix.
        """
        nodes_df = environment.nodes_df_
        edge_nodes_df = nodes_df[nodes_df["is_bin_edge"] == True]
        node_id_1 = edge_nodes_df[
            edge_nodes_df.linear_position == pos_1
        ].node_id.values[0]
        node_id_2 = edge_nodes_df[
            edge_nodes_df.linear_position == pos_2
        ].node_id.values[0]
        return environment.distance_between_nodes_[node_id_1][node_id_2]
