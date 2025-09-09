import numpy as np


def event_times_in_intervals_bool(event_times,
                                  valid_time_intervals):
    """
    Filter event times for those within valid_intervals
    :param event_times: array-like with times of events
    :param valid_time_intervals: nested list with intervals for valid times
    :return: boolean indicating indices in event_times within valid_time_intervals
    """
    return np.sum(np.asarray([np.logical_and(event_times >= t1,
                                             event_times <= t2)
                              for t1, t2 in valid_time_intervals]), axis=0) > 0


def event_times_in_intervals(event_times,
                             valid_time_intervals):
    """
    Filter event times for those within valid_intervals
    :param event_times: array-like with times of events
    :param valid_time_intervals: nested list with intervals for valid times
    :return: array with indices (from np.where) in original event_times of valid event times
    :return: array with valid event times
    """
    event_times = np.asarray(event_times)
    valid_bool = event_times_in_intervals_bool(event_times,
                                               valid_time_intervals)
    return np.where(valid_bool)[0], event_times[valid_bool]


def not_small_diff_bool(x, diff_threshold):
    if len(x) == 0:
        return np.asarray([])
    x_diff = np.diff(x)
    valid_bool = x_diff > diff_threshold
    return (np.concatenate(([True], valid_bool)) *
                 np.concatenate((valid_bool, [True])))  # True if x value is NOT part of small gap pair

