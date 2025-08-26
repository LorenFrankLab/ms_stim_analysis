import numpy as np
import datajoint as dj

from spyglass.lfp.v1 import LFPV1
from spyglass.ripple.v1 import RippleTimesV1
from spyglass.common import IntervalList, get_electrode_indices
from spyglass.utils.dj_mixin import SpyglassMixin

from ms_stim_analysis.Analysis.lfp_analysis import get_ref_electrode_index

schema = dj.schema("ms_ripple")


@schema
class RippleIntervals(SpyglassMixin, dj.Computed):
    definition = """
    -> RippleTimesV1
    ---
    -> IntervalList().proj(ripple_interval_list_name="interval_list_name")
    """

    def make(self, key):
        ripple_df = (RippleTimesV1() & key).fetch1_dataframe()
        ripple_intervals = np.array(
            [[st, en] for st, en in zip(ripple_df.start_time, ripple_df.end_time)]
        )

        # insert into IntervalList
        import random
        import string

        interval_list_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["target_interval_list_name"]
            + "_ripple_times"
            + "".join(random.choices(string.ascii_uppercase + string.digits, k=10)),
            "valid_times": ripple_intervals,
            "pipeline": "ms_ripples",
        }
        IntervalList().insert1(interval_list_key)

        # insert into RippleIntervals
        key["ripple_interval_list_name"] = interval_list_key["interval_list_name"]
        self.insert1(key)


@schema
class MsLfpArtifact(SpyglassMixin, dj.Computed):
    definition = """
    -> LFPV1
    ---
    -> IntervalList().proj(artifact_interval_list_name="interval_list_name")
    """

    def make(self, key):
        ref_electrode, lfp_s_key = get_ref_electrode_index(key)

        lfp_eseries = (LFPV1 & key).fetch_nwb()[0][
            "lfp"
        ]  # (LFPOutput()).fetch_nwb(restriction=lfp_s_key)[0]["lfp"]
        lfp_elect_indeces = get_electrode_indices(lfp_eseries, [ref_electrode])
        if lfp_elect_indeces[0] > 1000:
            raise ValueError("lfp_elect_indeces: ", lfp_elect_indeces)
        time = lfp_eseries.timestamps
        lfp = lfp_eseries.data[:, lfp_elect_indeces]

        threshold = 2000  # 2 mV threshold, hardcoded for the project
        # ind_valid = np.where(np.abs(lfp) < threshold)[0]
        # interval_breaks = np.append([0], np.where(np.diff(ind_valid) > 1)[0] + 1)
        # interval_breaks = np.append(interval_breaks, [len(ind_valid) - 1])
        # valid_intervals = np.array(
        #     [
        #         [
        #             time[ind_valid[interval_breaks[i]]],
        #             time[ind_valid[interval_breaks[i + 1]]],
        #         ]
        #         for i in range(len(interval_breaks) - 1)
        #     ]
        # )
        lfp_valid = np.abs(lfp) < threshold
        if all(lfp_valid):
            valid_intervals = np.array([[time[0], time[-1]]])
        else:
            switch_valid = np.diff(lfp_valid.astype(int), axis=0)
            ind_start = np.where(switch_valid == 1)[0] + 1
            ind_end = np.where(switch_valid == -1)[0] + 1
            if ind_start[0] > ind_end[0]:
                ind_start = np.insert(ind_start, 0, 0)
            if ind_start[-1] > ind_end[-1]:
                ind_end = np.append(ind_end, len(lfp_valid) - 1)
            valid_intervals = np.array(
                [[time[ind_start[i]], time[ind_end[i]]] for i in range(len(ind_start))]
            )
        assert (
            np.diff(valid_intervals, axis=1) > 0
        ).all(), "Invalid intervals detected"
        assert (
            valid_intervals[1:, 0] > valid_intervals[:-1, 1]
        ).all(), "Invalid intervals detected"

        # insert into IntervalList
        interval_list_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["target_interval_list_name"]
            + "_ms_lfp_artifact_times",
            "valid_times": valid_intervals,
            "pipeline": "ms_lfp_artifact",
        }
        IntervalList().insert1(interval_list_key)

        # insert into MsLfpArtifact
        key["artifact_interval_list_name"] = interval_list_key["interval_list_name"]
        self.insert1(key)
