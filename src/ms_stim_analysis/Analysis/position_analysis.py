import numpy as np
from spyglass.common import (
    IntervalList,
    TaskEpoch,
    PositionIntervalMap,
)
from spyglass.position.v1 import TrodesPosV1, DLCPosV1
from spyglass.position import PositionOutput
from spyglass.linearization.v1 import LinearizedPositionV1, TrackGraph
from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1


from ms_stim_analysis.AnalysisTables.ms_task_identification import TaskIdentification
from ms_stim_analysis.AnalysisTables.ms_interval import EpochIntervalListName


def get_running_intervals(
    nwb_file_name: str,
    epoch: int = None,
    interval_list_name: str = None,
    filter_speed: float = 10,
    dlc_pos: bool = False,
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
    dlc_pos : bool, optional
        whether to use dlc position data, by default False

    Returns
    -------
    list
       time intervals when rat is running
    """
    trodes_pos_params_name = "single_led"
    key = {"nwb_file_name": nwb_file_name}
    key.update({"epoch": epoch})
    if interval_list_name is None:
        interval_list_name = (EpochIntervalListName() & key).fetch1(
            "interval_list_name"
        )
    key.update({"interval_list_name": interval_list_name})
    if filter_speed == 0:
        return (IntervalList & key).fetch1("valid_times")

    if dlc_pos:
        if epoch is None:
            map_key = {
                "nwb_file_name": key["nwb_file_name"],
                "position_interval_name": key["interval_list_name"],
            }
            epoch = list(
                set(((PositionIntervalMap() & map_key) * TaskEpoch).fetch("epoch"))
            )
            if len(epoch) > 1:
                raise ValueError("More than one epoch found for", map_key)
            epoch = epoch[0]
            key["epoch"] = epoch
        df = (DLCPosV1() & key).fetch1_dataframe()
        speed = df["speed"].values
        speed_time = df.index.values

    else:
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


def lineartrack_position_filter(
    key: dict, buffer: float = 10, dlc_pos: bool = False, **kwargs
) -> list:
    """get list of interval times when rat is NOT at the ends of the linear track
    # 12.12.23: switch to using linearized position instead of x position

    Parameters
    ----------
    key : dict
        key for TrodesPosV1
    buffer : float, optional
        buffer zone around the ends of the linear track, by default 10

    Returns
    -------
    list
        list of time intervals when rat is not at the ends of the linear track
    """

    # get the position data
    if "pos_merge_id" not in key:
        if dlc_pos:
            map_key = {
                "nwb_file_name": key["nwb_file_name"],
                "position_interval_name": key["interval_list_name"],
            }
            epoch = list(
                set(((PositionIntervalMap() & map_key) * TaskEpoch).fetch("epoch"))
            )
            if len(epoch) > 1:
                raise ValueError("More than one epoch found for", map_key)
            dlc_key = {
                "nwb_file_name": key["nwb_file_name"],
                "epoch": epoch[0],
            }
            merge_id = (PositionOutput.DLCPosV1() & dlc_key).fetch1("merge_id")
            key["pos_merge_id"] = merge_id

        else:
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
    linear_limits = [buffer, distance - buffer]

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


def wtrack_position_filter(
    key: dict, buffer: float = 10, dlc_pos: bool = False, **kwargs
) -> list:
    """get list of interval times when rat is NOT at the ports of the w-track

    Parameters
    ----------
    key : dict
        key for TrodesPosV1
    buffer : float, optional
        buffer zone around the ports, by default 10
    dlc_pos : bool, optional
        whether to use dlc position data, by default False

    Returns
    -------
    list
        list of time intervals when rat is not at the ports of the w-track
    """
    # get the position data
    if "pos_merge_id" not in key:
        if dlc_pos:
            map_key = {
                "nwb_file_name": key["nwb_file_name"],
                "position_interval_name": key["interval_list_name"],
            }
            epoch = list(
                set(((PositionIntervalMap() & map_key) * TaskEpoch).fetch("epoch"))
            )
            if len(epoch) > 1:
                raise ValueError("More than one epoch found for", map_key)
            dlc_key = {
                "nwb_file_name": key["nwb_file_name"],
                "epoch": epoch[0],
            }
            merge_id = (PositionOutput.DLCPosV1() & dlc_key).fetch1("merge_id")
            key["pos_merge_id"] = merge_id

        else:
            merge_id = (
                (PositionOutput.TrodesPosV1 & key)
                & "trodes_pos_params_name LIKE '%upsampled'"
            ).fetch1("merge_id")
            key["pos_merge_id"] = merge_id

    # get the linearized position data
    lin_key = (
        LinearizedPositionV1() & key & "track_graph_name LIKE '%ms_wtrack%'"
    ).fetch1("KEY")
    df_ = (LinearizedPositionV1() & lin_key).fetch1_dataframe()
    x = np.asarray(df_["linear_position"])

    # get the info about the ports
    from spyglass.linearization.v1 import TrackGraph

    graph = (TrackGraph() & lin_key).get_networkx_track_graph()
    port_nodes = [0, 3, 5]  # port nodes we want to exclude
    exclude_right = [
        True,
        False,
        False,
    ]  # whether the aea to exclude is to the left or right of the port
    node_positions = (TrackGraph() & lin_key).fetch1("node_positions")[port_nodes]
    edge_order = (TrackGraph() & lin_key).fetch1("linear_edge_order")
    edge_spacing = (TrackGraph() & lin_key).fetch1("linear_edge_spacing")

    from track_linearization import get_linearized_position

    port_positions = get_linearized_position(
        node_positions, graph, edge_order=edge_order, edge_spacing=edge_spacing
    ).linear_position.values
    # define the linear position to exclude approaching the ports

    exclude_zone = [
        [x, x + buffer] if exclude_right[i] else [x - buffer, x]
        for i, x in enumerate(port_positions)
    ]

    # indexes to include
    valid_pos = np.ones(
        x.size,
    ).astype(bool)
    for exclude in exclude_zone:
        valid_pos = valid_pos & ((x < exclude[0]) | (x > exclude[1]))
    valid_pos = valid_pos.astype(int)
    valid_pos = np.append(
        [0],
        valid_pos,
    )
    assert valid_pos.sum()
    interval_st = df_.index[np.where(np.diff(valid_pos) == 1)[0]]
    interval_end = df_.index[np.where(np.diff(valid_pos) == -1)[0]]
    valid_intervals = [[st, en] for st, en in zip(interval_st, interval_end)]
    for interval in valid_intervals:
        assert interval[0] < interval[1]
    return valid_intervals


def filter_position_ports(key: dict, **kwargs) -> list:
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
    if key.get("epoch", -1) is None:
        key.pop("epoch")
    task = ((TaskIdentification * EpochIntervalListName) & key).fetch1("contingency")
    if task in ["lineartrack", "Lineartrack"]:
        return lineartrack_position_filter(key, **kwargs)
    if task in ["wtrack", "w-track", "Wtrack", "W-track", "W-Track"]:
        return wtrack_position_filter(key, **kwargs)
    print(f"task {task} not recognized")
    return None
