import datajoint as dj
from non_local_detector.model_checking import (
    get_HPD_spatial_coverage,
    get_highest_posterior_threshold,
)
import numpy as np
import pandas as pd

from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1
from spyglass.common import AnalysisNwbfile

from ms_opto_stim_protocol import OptoStimProtocol

schema = dj.schema("ms_credible_interval")


@schema
class CredibleIntervalParameters(SpyglassMixin, dj.Manual):
    definition = """
    # Parameters for credible interval calculation
    credible_params_name: varchar(80)  # name of the parameter set
    ---
    credible_interval_threshold: float
    acausal: bool
    """

    def insert_defaults(self):
        defaults = [
            {
                "credible_params_name": "default_causal",
                "credible_interval_threshold": 0.95,
                "acausal": False,
            },
            {
                "credible_params_name": "default_acausal",
                "credible_interval_threshold": 0.95,
                "acausal": True,
            },
        ]
        self.insert(defaults, skip_duplicates=True)


@schema
class CredibleIntervalSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> ClusterlessDecodingV1
    -> CredibleIntervalParameters
    ---
    """


@schema
class CredibleInterval(SpyglassMixin, dj.Computed):
    definition = """
    -> CredibleIntervalSelection
    ---
    -> AnalysisNwbfile
    data_object_id: varchar(64)
    """

    def make(self, key):
        # fetch the threshold
        threshold = (CredibleIntervalParameters & key).fetch1(
            "credible_interval_threshold"
        )

        # fetch the decoding results
        results = (ClusterlessDecodingV1() & key).fetch_results()
        if (CredibleIntervalParameters & key).fetch1("acausal"):
            posterior = results.acausal_posterior.squeeze()
        else:
            results.causal_posterior.squeeze()
        posterior = posterior.squeeze().unstack("state_bins").sum("state")

        # calculate the credible interval
        threshold = get_highest_posterior_threshold(posterior, threshold)
        spatial_coverage = get_HPD_spatial_coverage(posterior, threshold)

        # save the results
        credible_df = pd.DataFrame(
            spatial_coverage, index=results.time, columns=["coverage"]
        )

        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key["analysis_file_name"] = analysis_file_name
        key["data_object_id"] = AnalysisNwbfile().add_nwb_object(
            key["analysis_file_name"], credible_df, "credible_interval"
        )
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
        self.insert1(key)

    def alligned_response(self, key, marks, window=0.05):
        # fetch_data
        credible_df = self.fetch1_dataframe()
        # convert window from seconds to indices
        if not isinstance(window, list):
            window = [-window, window]
        delta_t = np.mean(np.diff(credible_df.index))
        window = [int(np.round(w / delta_t)) for w in window]

        # make the alligned values
        mark_inds = np.digitize(marks, credible_df.index)
        data = [credible_df.values[i + window[0] : i + window[1]] for i in mark_inds]
        return data

    def fetch1_dataframe(self) -> pd.DataFrame:
        if not len(nwb := self.fetch_nwb()):
            raise ValueError("fetch1_dataframe must be called on a single key")
        return nwb[0]["data"]
