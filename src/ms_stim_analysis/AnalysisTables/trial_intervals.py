import datajoint as dj
import numpy as np


from ms_stim_analysis.AnalysisTables.ms_dio_event import ProcessedDioEvents
from ms_stim_analysis.AnalysisTables.ms_task_performance import (
    AlternationTaskPerformance,
)
from ms_stim_analysis.Analysis.position_analysis import filter_position_ports

from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.common import interval_list_intersect

import warnings

warnings.filterwarnings("ignore")

schema = dj.schema("ms_trial_intervals")


@schema
class TrialIntervals(SpyglassMixin, dj.Computed):
    definition = """
    -> AlternationTaskPerformance
    -> PositionOutput.proj(pos_merge_id="merge_id")
    ---
    inbound_trial_intervals: longblob
    outbound_trial_intervals: longblob
    """

    def make(self, key):
        # print(key)
        # outcomes and durations
        outcomes = (AlternationTaskPerformance() & key).fetch1("performance_outcomes")
        poke_times = ((ProcessedDioEvents().FirstUpPokes()) & key).fetch1(
            "dio_first_poke_times"
        )
        last_poke_times = ((ProcessedDioEvents().LastDownPokes()) & key).fetch1(
            "dio_last_poke_times"
        )
        # trial_durations.extend(np.diff(poke_times))
        inbound_trial = [("inbound" in x) for x in outcomes[1:]]
        accuracy = [("incorrect" in x) for x in outcomes[1:]]

        # durations
        # pos_key = {"merge_id": key["position_merge_id"]}
        # interval_list
        travel_intervals = np.array(filter_position_ports(key, dlc_pos=True))
        trial_durations = []
        trial_intervals = []
        for i in range(poke_times.size - 1):
            travel_ = interval_list_intersect(
                np.array([[poke_times[i], poke_times[i + 1]]]), travel_intervals
            )
            # if not len(travel_) == 1:
            #     raise ValueError("A trial should have exactly one interval")
            if not len(travel_):
                trial_intervals.append([np.nan, np.nan])
            else:
                trial_intervals.append([np.min(travel_), np.max(travel_)])

        # trial_durations = np.array(trial_durations)
        trial_intervals = np.array(trial_intervals)
        inbound_trial = np.array(inbound_trial).astype(bool)

        # inbound_trial_durations = trial_durations[inbound_trial]
        # outbound_trial_durations = trial_durations[~inbound_trial]
        inbound_trial_intervals = trial_intervals[inbound_trial]
        outbound_trial_intervals = trial_intervals[~inbound_trial]

        key["inbound_trial_intervals"] = inbound_trial_intervals
        key["outbound_trial_intervals"] = outbound_trial_intervals
        self.insert1(key)
