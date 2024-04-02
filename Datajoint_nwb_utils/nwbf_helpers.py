import numpy as np
from spyglass.common import TaskEpoch, IntervalList
from Utils.point_process_helpers import event_times_in_intervals_bool


def nwbf_session_id_to_file_name(nwbf):
    # Turn nwb file session_id into nwb file name
    # Assumes session_id has form: ratname_date, and nwb file name has form: ratnamedate_.nwb
    return f"{''.join(nwbf.fields['session_id'].split('_'))}_.nwb"


def get_nwb_file(nwb_file_name):
    """Return nwb file"""
    import pynwb
    from spyglass.common import Nwbfile
    return pynwb.NWBHDF5IO(Nwbfile().get_abs_path(nwb_file_name),'r').read()


def check_nwb_file_in_session_table(nwb_file_name):
    """Raise error if nwb file not in Session table"""
    from spyglass.common import Session
    if len((Session() & {'nwb_file_name': nwb_file_name})) == 0:
        raise Exception("nwb file not in Session table")


def nwbf_name_from_subject_id_date(subject_id,
                                   date):
    return f"{subject_id}{date}_.nwb"


def subject_id_date_from_nwbf_name(nwb_file_name):

    len_date = 8
    subject_id_date = nwb_file_name.split("_.nwb")[0]
    subject_id = subject_id_date[:-len_date]
    date = subject_id_date[-len_date:]

    return subject_id, date


def get_epoch_valid_times(nwb_file_name, epoch):
    epoch_interval_list_name = (TaskEpoch() & {"nwb_file_name": nwb_file_name,
                                               "epoch": epoch}).fetch1(
        "interval_list_name")  # get interval list name for epoch
    return (IntervalList() & {"nwb_file_name": nwb_file_name,
                               "interval_list_name": epoch_interval_list_name}).fetch1(
        "valid_times")  # get epoch valid times


def get_epoch_time_interval(nwb_file_name, epoch):
    epoch_valid_times = get_epoch_valid_times(nwb_file_name, epoch)
    return np.asarray([epoch_valid_times[0][0], epoch_valid_times[-1][-1]])


def events_in_epoch_bool(nwb_file_name, epoch, event_times):
    epoch_interval = get_epoch_time_interval(nwb_file_name, epoch)
    return event_times_in_intervals_bool(event_times, [epoch_interval])