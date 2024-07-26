import datajoint as dj
import pandas as pd

from non_local_detector.visualization import create_interactive_1D_decoding_figurl
from spyglass.common import AnalysisNwbfile
from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1
from spyglass.utils import SpyglassMixin


schema = dj.schema("ms_decoding")


@schema
class ClusterlessDecodingFigurl_1D(SpyglassMixin, dj.Computed):
    definition = """
    -> ClusterlessDecodingV1
    ---
    figurl: varchar(3000)
    """

    def make(self, key):
        position_info = ClusterlessDecodingV1.fetch_linear_position_info(key)
        decoding_results = (ClusterlessDecodingV1 & key).fetch_results()
        results_time = decoding_results.acausal_posterior.isel(intervals=0).time.values
        position_info = position_info.loc[results_time[0] : results_time[-1]]
        spikes, _ = ClusterlessDecodingV1.fetch_spike_data(key)
        figurl = create_interactive_1D_decoding_figurl(
            position=position_info["linear_position"],
            speed=position_info["speed"],
            spike_times=spikes,
            results=decoding_results.squeeze(),
        )
        key["figurl"] = figurl
        self.insert1(key)


@schema
class SortedSpikesDecodingFigurl_1D(SpyglassMixin, dj.Computed):
    definition = """
    -> SortedSpikesDecodingV1
    ---
    figurl: varchar(3000)
    """

    def make(self, key):
        position_info = SortedSpikesDecodingV1.fetch_linear_position_info(key)
        decoding_results = (SortedSpikesDecodingV1 & key).fetch_results()
        results_time = decoding_results.acausal_posterior.isel(intervals=0).time.values
        position_info = position_info.loc[results_time[0] : results_time[-1]]
        spikes = SortedSpikesDecodingV1.fetch_spike_data(key)
        figurl = create_interactive_1D_decoding_figurl(
            position=position_info["linear_position"],
            speed=position_info["speed"],
            spike_times=spikes,
            results=decoding_results.squeeze(),
        )
        key["figurl"] = figurl
        self.insert1(key)


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
