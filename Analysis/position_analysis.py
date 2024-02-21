import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import os

import spyglass.common as sgc
from spyglass.common import (
    Session,
    IntervalList,
    LabMember,
    LabTeam,
    Raw,
    Session,
    Nwbfile,
    TaskEpoch,
    Electrode,
    ElectrodeGroup,
    LFP,
    LFPSelection,
    LFPBand,
    LFPBandSelection,
    get_electrode_indices,
)
from spyglass.position.v1 import TrodesPosV1
from spyglass.position import PositionOutput
from spyglass.linearization.v1 import LinearizedPositionV1, TrackGraph

import sys
from Metadata.ms_task_identification import TaskIdentification
from Time_and_trials.ms_interval import EpochIntervalListName


def get_running_intervals(
    nwb_file_name: str,
    epoch: int = None,
    interval_list_name: str = None,
    filter_speed: float = 10,
    **kwargs,
) -> list:
    """get list of interval times when rat is running

    Parameters
    ----------
    nwb_file_name : str
        nwb_file_name key
    epoch : int
        epoch number under Jen's convention
    pos_interval_name : str
        interval name for position data, if None, will look up the interval name for the epoch
    filter_speed : float, optional
        threshold speed to define running (in cm/s), by default 10

    Returns
    -------
    list
       time intervals when rat is running
    """
    trodes_pos_params_name = "single_led"
    key = {"nwb_file_name": nwb_file_name}
    key.update({"epoch": epoch})
    if interval_list_name is None:
        interval__list_name = (EpochIntervalListName() & key).fetch1(
            "interval_list_name"
        )
    key.update({"interval_list_name": interval_list_name})
    speed = (
        (TrodesPosV1() & key & {"trodes_pos_params_name": trodes_pos_params_name})
        .fetch_nwb()[0]["velocity"]["velocity"]
        .data[:, 2]
    )
    speed_time = (
        (TrodesPosV1() & key & {"trodes_pos_params_name": trodes_pos_params_name})
        .fetch_nwb()[0]["velocity"]["velocity"]
        .timestamps[:]
    )

    # make intervals where rat is running
    speed_binary = (speed > filter_speed).astype(int)
    speed_binary = np.append([0], speed_binary)
    if np.min(speed_binary) == 1:
        run_intervals = [(speed_time[0], speed_time[-1])]
    t_diff = np.diff(speed_binary)
    t_run_start = speed_time[np.where(t_diff == 1)[0]]
    t_run_stop = speed_time[np.where(t_diff == -1)[0]]
    run_intervals = [(start, stop) for start, stop in zip(t_run_start, t_run_stop)]
    return run_intervals


def lineartrack_position_filter(key: dict) -> list:
    """get list of interval times when rat is NOT at the ends of the linear track
    # 12.12.23: switch to using linearized position instead of x position

    Parameters
    ----------
    key : dict
        key for TrodesPosV1

    Returns
    -------
    list
        list of time intervals when rat is not at the ends of the linear track
    """

    # get the position data
    if not "pos_merge_id" in key:
        merge_id = (
            (PositionOutput.TrodesPosV1 & key)
            & "trodes_pos_params_name LIKE '%upsampled'"
        ).fetch1("merge_id")
        key["pos_merge_id"] = merge_id
    # get the linearized position data
    df_ = (
        LinearizedPositionV1() & key & "track_graph_name LIKE '%ms_lineartrack%'"
    ).fetch1_dataframe()
    x = np.asarray(df_["linear_position"])

    # get the linear limits
    track_key = {
        "track_graph_name": (
            LinearizedPositionV1() & key & "track_graph_name LIKE '%ms_lineartrack%'"
        ).fetch1("track_graph_name")
    }
    distance = (
        (TrackGraph() & track_key).get_networkx_track_graph().edges[[0, 1]]["distance"]
    )
    linear_limits = [10, distance - 10]

    # filter
    print("linear_limits", linear_limits)
    valid_pos = ((x > linear_limits[0]) & (x < linear_limits[1])).astype(int)
    valid_pos = np.append(
        [0],
        valid_pos,
    )
    interval_st = df_.index[np.where(np.diff(valid_pos) == 1)[0]]
    interval_end = df_.index[np.where(np.diff(valid_pos) == -1)[0]]
    valid_intervals = [[st, en] for st, en in zip(interval_st, interval_end)]
    for interval in valid_intervals:
        assert interval[0] < interval[1]
    return valid_intervals


def wtrack_position_filter(key: dict) -> list:
    """get list of interval times when rat is NOT at the ports of the w-track

    Parameters
    ----------
    key : dict
        key for TrodesPosV1

    Returns
    -------
    list
        list of time intervals when rat is not at the ports of the w-track
    """
    df_ = (
        (TrodesPosV1() & key) & "trodes_pos_params_name LIKE '%upsampled'"
    ).fetch1_dataframe()
    wtrack_limit = np.nanmin(np.asarray(df_["position_y"])) + 25
    print("wtrack_limit", wtrack_limit)
    valid_pos = (np.asarray(df_["position_y"]) > wtrack_limit).astype(int)
    valid_pos = np.append(
        [0],
        valid_pos,
    )
    interval_st = df_.index[np.where(np.diff(valid_pos) == 1)[0]]
    interval_end = df_.index[np.where(np.diff(valid_pos) == -1)[0]]
    return [[st, en] for st, en in zip(interval_st, interval_end)]


def filter_position_ports(key: dict) -> list:
    """filter position data to exclude times when rat is at the ports

    Parameters
    ----------
    key : dict
        key you want to filter

    Returns
    -------
    list
        list of time intervals when rat is not at the ports
    """
    task = ((TaskIdentification * EpochIntervalListName) & key).fetch1("contingency")
    if task in ["lineartrack", "Lineartrack"]:
        return lineartrack_position_filter(key)
    if task in ["wtrack", "w-track", "Wtrack", "W-track", "W-Track"]:
        return wtrack_position_filter(key)
    print(f"task {task} not recognized")
    return None
