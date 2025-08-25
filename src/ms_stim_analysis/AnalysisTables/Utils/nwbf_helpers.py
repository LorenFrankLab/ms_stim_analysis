import numpy as np
from spyglass.common import TaskEpoch, IntervalList, Session
from .point_process_helpers import event_times_in_intervals_bool


def get_epoch_valid_times(nwb_file_name, epoch):
    epoch_interval_list_name = (
        TaskEpoch() & {"nwb_file_name": nwb_file_name, "epoch": epoch}
    ).fetch1(
        "interval_list_name"
    )  # get interval list name for epoch
    return (
        IntervalList()
        & {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": epoch_interval_list_name,
        }
    ).fetch1(
        "valid_times"
    )  # get epoch valid times


def get_epoch_time_interval(nwb_file_name, epoch):
    epoch_valid_times = get_epoch_valid_times(nwb_file_name, epoch)
    return np.asarray([epoch_valid_times[0][0], epoch_valid_times[-1][-1]])


def events_in_epoch_bool(nwb_file_name, epoch, event_times):
    epoch_interval = get_epoch_time_interval(nwb_file_name, epoch)
    return event_times_in_intervals_bool(event_times, [epoch_interval])
