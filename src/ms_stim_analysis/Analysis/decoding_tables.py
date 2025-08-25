from non_local_detector.visualization import create_interactive_1D_decoding_figurl
from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1
from spyglass.utils import SpyglassMixin
import datajoint as dj

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
