import numpy as np
import datajoint as dj

from spyglass.ripple.v1 import RippleTimesV1, RippleParameters
from spyglass.common import IntervalList

schema = dj.schema("ms_ripple")


@schema
class RippleIntervals(dj.Computed):
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
        interval_list_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["target_interval_list_name"] + "_ripple_times",
            "valid_times": ripple_intervals,
            "pipeline": "ms_ripples",
        }
        IntervalList().insert1(interval_list_key)

        # insert into RippleIntervals
        key["ripple_interval_list_name"] = interval_list_key["interval_list_name"]
        self.insert1(key)
