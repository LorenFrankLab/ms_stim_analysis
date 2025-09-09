import datajoint as dj
import spyglass.common as sgc
import numpy as np
import matplotlib.pyplot as plt
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("ms_opto_stim_protocol")


@schema
class OptoStimProtocolParams(SpyglassMixin, dj.Manual):
    """
    Parameters for Inferinng optogenetic stimulus from DIO data
    """

    definition = """
    opto_infer_params_name: varchar(80) # name for this set of parameters
    ---
    params: longblob #dict?
    """

    @classmethod
    def insert_default(cls, **kwargs):
        """
        Insert default parameter sets for parameter inference
        """
        params = {
            "optogenetic_protocol": "phase_targeting",  # type of protocol to extract
            "control_interval_threshold": 120,  # minimum off time to define a control interval
            "behavior_defined_off_threshold": 1,  # assume off periods greater than this are due to spatial position (unless big enogh to be control interval)
        }
        cls.insert1(
            {"opto_infer_params_name": "default_phase_targeting", "params": params},
            skip_duplicates=True,
        )
        params = {
            "optogenetic_protocol": "pulse_train",  # type of protocol to extract
            "control_interval_threshold": 120,  # minimum off time to define a control interval
            "behavior_defined_off_threshold": 1,  # assume off periods greater than this are due to spatial position (unless big enough to be control interval)
            "train_delay_threshold": 0.009,  # define any delay between pulses longer than this as a new train
        }
        cls.insert1(
            {"opto_infer_params_name": "default_pulse_train", "params": params},
            skip_duplicates=True,
        )

    @classmethod
    def get_default(cls):
        query = cls & {"trodes_pos_params_name": "default_entrainment"}
        if not len(query) > 0:
            cls().insert_default(skip_duplicates=True)
            default = (cls & {"trodes_pos_params_name": "default_entrainment"}).fetch1()
        else:
            default = query.fetch1()
        return default

    @classmethod
    def get_accepted_params(cls):
        default = cls.get_default()
        return list(default["params"].keys())


@schema
class OptoStimProtocolSelection(SpyglassMixin, dj.Manual):
    """
    Table to pair an interval with position data
    and position determination parameters
    """

    definition = """
    -> sgc.IntervalList
    -> sgc.DIOEvents
    -> OptoStimProtocolParams
    ---
    """


@schema
class OptoStimProtocol(SpyglassMixin, dj.Computed):
    """
    Table to calculate the position based on Trodes tracking
    """

    definition = """
    -> OptoStimProtocolSelection
    ---
    optogenetic_protocol: varchar(300) # name of optogenetic protocol type (e.g. pulse train)
    pulse_length_ms: int # length of individual pulse
    inter_pulse_interval_ms: int # time between pulses in a train (= np.nan if pulses_per_train=1 or phase-targeting feedback)
    inter_train_interval_ms: int # time between end of train and start of new one (= np.nan if phase-targeting feedback)
    period_ms: int # time between successive train starts  (= np.nan if phase-targeting feedback)
    pulses_per_train: int # number of pulses in a train
    number_trains = -1: float # mean number of train events observed between behavior-off intervals
    stim_on: bool # whether any optogenetic stimulus occured in this interval
    test_intervals: longblob  # numpy array with start and end times for each interval optogenetics is running
    control_intervals: longblob # numpy array with start and end times for each interval optogenetics is off
    """

    def make(self, key):
        print(f"Computing optogenetic protocols for: {key}")
        params = (OptoStimProtocolParams() & key).fetch1("params")

        # fetch epoch interval information
        interval = (sgc.IntervalList & key).fetch1("valid_times")[0]
        # gets the time and opt data for that epoch interval
        dio_info = sgc.DIOEvents() & {
            "nwb_file_name": key["nwb_file_name"],
            "dio_event_name": key["dio_event_name"],
        }
        dio_data = dio_info.fetch_nwb()[0]["dio"]
        time = dio_data.timestamps[:]
        epoch_index = np.where((time >= interval[0]) & (time <= interval[1]))[0]
        data = dio_data.data[epoch_index]
        time = time[epoch_index]
        control_intervals = []
        test_intervals = []
        # append an off event at the beginning and end of the epoch if necessary, indicates the session starts with probe off
        # TODO confrim this with abhilasha or with the data
        if len(data) == 0 or (data[0] == 1):
            data = np.append(
                [
                    0,
                ],
                data,
            )
            time = np.append([interval[0]], time)
        if len(data) == 0 or (data[-1] == 1):
            data = np.append(
                data,
                [
                    0,
                ],
            )
            time = np.append(
                time,
                [interval[-1]],
            )
            add_end_control_interval = False
        else:
            add_end_control_interval = True

        # check if there's any stim in this period. If not, call it all control and exit
        if data.sum() == 0:
            control_intervals.append(
                interval,
            )
            test_intervals = []
            self.insert1(
                {
                    **key,
                    "control_intervals": control_intervals,
                    "test_intervals": test_intervals,
                    "pulse_length_ms": -1,
                    "inter_pulse_interval_ms": -1,
                    "inter_train_interval_ms": -1,
                    "period_ms": -1,
                    "pulses_per_train": -1,
                    "number_trains": -1,
                    "stim_on": False,
                    "optogenetic_protocol": "None",
                }
            )
            return
        else:
            stim_on = True
        # find the times when the stim changes state and the duration of time between when it does
        t_switch_on = time[data > 0]
        t_switch_off = time[data == 0]
        duration_off = t_switch_on - t_switch_off[:-1]
        # define control intervals based on threshold
        index_control_list = np.where(
            duration_off > params["control_interval_threshold"]
        )[0]
        control_intervals.extend(
            [
                np.array([t_switch_off[index_control], t_switch_on[index_control]])
                for index_control in index_control_list
            ]
        )
        # check if it ends in a control interval:
        if add_end_control_interval:
            # if naturally ended in an off, check if control interval belongs there
            if interval[-1] - time[-1] > params["control_interval_threshold"]:
                control_intervals.append(np.array([time[-1], interval[1]]))
        # fill in the rest of the interval with test_intervals
        test_intervals = []
        for i in range(len(control_intervals)):
            if i == 0:
                if (
                    control_intervals[i][0] > interval[0]
                ):  # if epoch starts in test interval
                    test_intervals.append(
                        np.array(
                            [interval[0], control_intervals[0][0]],
                        )
                    )
            if i + 1 == len(control_intervals):
                # add final test interval if it doesn't end in a control
                if not (control_intervals[i][1] == interval[1]):
                    test_intervals.append(
                        np.array([control_intervals[i][1], interval[1]])
                    )
                break
            test_intervals.append(
                np.array([control_intervals[i][1], control_intervals[i + 1][0]])
            )
        if len(control_intervals) == 0:
            test_intervals.append(interval)
        # identify the pulse duration
        duration_on = t_switch_off[1:] - t_switch_on[:]
        pulse_length_ms = np.round(
            np.median(duration_on) * 1000,
        )  # units ms

        # calculate delay intervals, remove the control off period and spatially defined off events
        train_delays = duration_off[
            duration_off < params["behavior_defined_off_threshold"]
        ]
        if params["optogenetic_protocol"] == "phase_targeting":
            # only calculate the inter stimulus interval
            inter_pulse_interval_ms = np.round(np.median(train_delays) * 1000)
            inter_train_interval_ms = -1
            period_ms = -1
            pulses_per_train = -1
            number_trains = -1
        elif params["optogenetic_protocol"] == "pulse_train":
            # calculate within-train delays
            intra_train_delays = train_delays[
                train_delays < params["train_delay_threshold"]
            ]  # delays that are shorter than the new train threshold
            print("INTRATRAINS: ", len(intra_train_delays))
            if (
                len(intra_train_delays) <= 100
            ):  # only single pulse trains, (with a buffer for artifacts/short pauses)
                inter_pulse_interval_ms = -1
            else:
                inter_pulse_interval_ms = np.round(np.median(intra_train_delays) * 1000)
            # calculate delay between trains
            inter_train_interval_ms = np.round(
                np.median(train_delays[train_delays > params["train_delay_threshold"]])
                * 1000
            )
            # calculate number of pulses per train (= # pulses between super train threshold events)
            index_intertrain_delays = np.where(
                duration_off > params["train_delay_threshold"]
            )[0]
            pulses_per_train = np.median(
                index_intertrain_delays[1:] - index_intertrain_delays[:-1]
            )
            # calculate the period between the end of each train
            t_intertrain_delays = t_switch_off[
                index_intertrain_delays
            ]  # the times when intertrain delays begin
            intertrain_period = t_intertrain_delays[1:] - t_intertrain_delays[:-1]
            intertrain_period = intertrain_period[
                intertrain_period < params["behavior_defined_off_threshold"]
            ]  # filter delays including behaviorally define pauses
            period_ms = np.round(np.median(intertrain_period) * 1000)
            # calculate the observed number of trains between pauses
            index_interbehavior_delays = np.where(
                duration_off > params["behavior_defined_off_threshold"]
            )[0]
            number_trains = np.mean(
                (index_interbehavior_delays[1:] - index_interbehavior_delays[:-1])
                / pulses_per_train
            )
            if np.isnan(number_trains):
                number_trains = -1
            print(index_interbehavior_delays)
            print("TYPE:", type(float(number_trains)), number_trains)
        self.insert1(
            {
                **key,
                "control_intervals": control_intervals,
                "test_intervals": test_intervals,
                "pulse_length_ms": int(pulse_length_ms),
                "inter_pulse_interval_ms": int(inter_pulse_interval_ms),
                "inter_train_interval_ms": int(inter_train_interval_ms),
                "period_ms": int(period_ms),
                "pulses_per_train": int(pulses_per_train),
                "number_trains": float(number_trains),
                "stim_on": True,
                "optogenetic_protocol": params["optogenetic_protocol"],
            }
        )
        self.make_opto_intervals(key)

    def get_protocol_type(self, key):
        return (self & key).fetch("optogenetic_protocol")

    def validate_intervals(self, key):
        # Visual inspection to confirm control interval allocation
        data, time = self.get_stimulus(key, convert_to_plot_format=True)
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(time, data)
        for interval in (self & key).fetch1("control_intervals"):
            ax.fill_between(
                interval, [0, 0], [1, 1], facecolor="grey", alpha=0.3, label="control"
            )
        for interval in (self & key).fetch1("test_intervals"):
            ax.fill_between(
                interval, [0, 0], [1, 1], facecolor="firebrick", alpha=0.3, label="test"
            )
        plt.legend()
        return fig

    def get_control_intervals(self, key):
        return (self & key).fetch("control_intervals")

    def get_stimulus(self, key, initial_zero=True, convert_to_plot_format=False):
        # put stim info in if missing
        if "dio_event_name" not in key:
            key["dio_event_name"] = "stim"
        # gets the time and optognetic data for that epoch interval
        interval = (sgc.IntervalList & key).fetch1("valid_times")[0]
        dio_info = sgc.DIOEvents() & {
            "nwb_file_name": key["nwb_file_name"],
            "dio_event_name": key["dio_event_name"],
        }
        dio_data = dio_info.fetch_nwb()[0]["dio"]
        time = dio_data.timestamps[:]
        epoch_index = np.where((time >= interval[0]) & (time <= interval[1]))[0]
        data = dio_data.data[epoch_index]
        if len(data) == 0:
            return np.array([]), np.array([])
        time = time[epoch_index]
        # add a intial off state to indicate interval starts with stimulus off
        if initial_zero and not (data[0] == 0):
            data = np.append(
                [
                    0,
                ],
                data,
            )
            time = np.append([interval[0]], time)
        # add a final off state to indicate interval ends with stimulus off
        if initial_zero and not (data[-1] == 0):
            data = np.append(data, [0])
            time = np.append(
                time,
                [interval[-1]],
            )
        if convert_to_plot_format:
            data_plot = []
            time_plot = []
            for x, t in zip(data, time):
                if x == 1:
                    data_plot.extend([0, 1])
                if x == 0:
                    data_plot.extend([1, 0])
                time_plot.extend([t, t])
            data = np.array(data_plot)
            time = np.array(time_plot)
        return data, time

    def get_cylcle_begin_timepoints(self, key):
        # Returns a list of timepoints where stimulation resumes after behavior-defined delay
        key_list = (self & key).fetch("KEY")  # ensure full primary key is defined
        if len(key_list) > 1:
            print("please provide unique primary key")
            return None
        threshold = (OptoStimProtocolParams & key_list[0]).fetch1("params")[
            "behavior_defined_off_threshold"
        ]
        data, time = self.get_stimulus(
            key_list[0],
        )
        t_switch_off = time[data == 0]
        t_switch_on = time[data == 1]
        duration_off = t_switch_on - t_switch_off[:-1]
        ind_new_cycle = np.where(duration_off > threshold)[0]
        return t_switch_on[ind_new_cycle]

    @staticmethod
    def make_opto_intervals(key):
        test_interval = (OptoStimProtocol & key).fetch1("test_intervals")
        control_interval = (OptoStimProtocol & key).fetch1("control_intervals")

        test_interval_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["interval_list_name"] + "_opto_test_interval",
            "valid_times": test_interval,
        }
        sgc.IntervalList().insert1(test_interval_key, skip_duplicates=True)
        control_interval_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["interval_list_name"] + "_opto_control_interval",
            "valid_times": control_interval,
        }
        sgc.IntervalList().insert1(control_interval_key, skip_duplicates=True)

        # the stimulus on intervals
        stim, stim_time = (OptoStimProtocol() & key).get_stimulus(key)
        if stim.size == 0:
            return
        ind_on = np.where(stim == 1)[0]
        stim_intervals = [[stim_time[i], stim_time[i + 1]] for i in ind_on]
        stim_intervals = np.array(stim_intervals)
        stim_interval_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": key["interval_list_name"] + "_stimulus_on_interval",
            "valid_times": stim_intervals,
        }
        sgc.IntervalList().insert1(stim_interval_key, skip_duplicates=True)
        return


@schema
class OptoStimProtocolTransfected(dj.Manual):
    """
    Table to define whether an animal successfully showed transfection in histology
    """

    definition = """
    -> sgc.Session
    ---
    transfected: bool #whether an animal was successfully transfected
    """


@schema
class OptoStimProtocolLaser(dj.Manual):
    """
    Table to define whether an animal successfully showed transfection in histology
    """

    definition = """
    -> sgc.Session
    ---
    laser_power: decimal(10,1) #laser power in mW
    """


@schema
class OptoStimProtocolClosedLoop(dj.Manual):
    """
    Table to define parameters specific to closed-loop protocols
    """

    definition = """
    -> sgc.IntervalList
    ---
    targeted_phase: int #targeted phase of theta (in degrees)
    """
