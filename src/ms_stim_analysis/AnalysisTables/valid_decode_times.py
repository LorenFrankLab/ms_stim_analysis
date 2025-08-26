import datajoint as dj
import numpy as np

from spyglass.common import interval_list_contains
from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1
from spyglass.utils.dj_mixin import SpyglassMixin


from ms_stim_analysis.Analysis.utils import get_running_valid_intervals
from ms_stim_analysis.AnalysisTables.ms_opto_stim_protocol import OptoStimProtocol

schema = dj.schema("ms_valid_decodes")


@schema
class ClusterlessValidDecodeParams(SpyglassMixin, dj.Manual):
    definition = """
    valid_decode_params_name: varchar(128)
    ---
    min_run_length : float
    min_speed : float
    dlc_pos : bool
    avg_decode_distance_threshold : float
    stim_validation_window : float
    invalid_stim_threshold_time : float
    """

    def insert_default(self):
        self.insert1(
            dict(
                valid_decode_params_name="default",
                min_run_length=0.5,
                min_speed=10,
                dlc_pos=True,
                avg_decode_distance_threshold=10,
                stim_validation_window=0.1,
                invalid_stim_threshold_time=0.01,
            ),
            skip_duplicates=True,
        )


@schema
class ClusterlessValidDecodeSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> ClusterlessValidDecodeParams
    -> ClusterlessDecodingV1
    -> OptoStimProtocol
    ---
    """


@schema
class ClusterlessValidDecodeTimes(SpyglassMixin, dj.Computed):
    definition = """
    -> ClusterlessValidDecodeSelection
    ---
    opto_valid_decode_times : longblob
    control_valid_decode_times : longblob
    valid_stim_times : longblob
    """

    def make(self, key):
        # load the parameters
        params = (ClusterlessValidDecodeParams & key).fetch1()
        min_run_length = params["min_run_length"]
        min_speed = params["min_speed"]
        dlc_pos = params["dlc_pos"]
        avg_decode_distance_threshold = params["avg_decode_distance_threshold"]
        stim_validation_window = params["stim_validation_window"]
        invalid_stim_threshold_time = params["invalid_stim_threshold_time"]

        # get the valid run intervals
        opto_run_interval, control_run_interval = get_running_valid_intervals(
            key.copy(), filter_speed=min_speed, dlc_pos=dlc_pos
        )
        # load the needed decode data
        decode_key = key
        decode_table = ClusterlessDecodingV1() & decode_key
        results = decode_table.fetch_results()
        ahead_behind = decode_table.get_ahead_behind_distance()
        decode_time = results.time.values

        for run_intervals, name in zip(
            [opto_run_interval, control_run_interval], ["opto", "control"]
        ):
            # filter for sufficiently long runs
            run_intervals = run_intervals[
                run_intervals[:, 1] - run_intervals[:, 0] > min_run_length
            ]
            # filter for runs with close average decode distance
            run_intervals = [
                interval
                for interval in run_intervals
                if self.valid_distance(
                    decode_time, ahead_behind, interval, avg_decode_distance_threshold
                )
            ]
            key[name + "_valid_decode_times"] = run_intervals

        # get the stim times
        stim, stim_time = (OptoStimProtocol() & key).get_stimulus(key)
        stim_time = stim_time[stim == 1]
        # restrict to stims in valid_runs
        run_intervals = key["opto_valid_decode_times"]
        valid_stims = interval_list_contains(np.array(run_intervals), stim_time)
        valid_stims = [
            s
            for s in valid_stims
            if self.include_stim(
                s,
                ahead_behind,
                decode_time,
                window=stim_validation_window,
                threshold_distance=avg_decode_distance_threshold,
                threshold_time=invalid_stim_threshold_time,
            )
        ]
        key["valid_stim_times"] = valid_stims
        self.insert1(key)

    @staticmethod
    def valid_distance(time, distance, interval, threshold):
        ind = np.logical_and(time > interval[0], time < interval[1])
        return np.mean(np.abs(distance[ind])) < threshold

    @staticmethod
    def include_stim(
        stim_time,
        ahead_behind,
        time,
        window=0.05,
        threshold_distance=50,
        threshold_time=0.01,
    ):
        ind = np.logical_and(time > stim_time - window, time < stim_time + window)
        dt = np.mean(np.diff(time[ind]))
        bad_decode = np.abs(ahead_behind[ind]) > threshold_distance
        return np.sum(bad_decode) * dt < threshold_time
