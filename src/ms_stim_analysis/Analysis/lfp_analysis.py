from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
from scipy import signal
from tqdm import tqdm
import pywt
import logging

import spyglass.common as sgc
from spyglass.common import (
    Session,
    Electrode,
    ElectrodeGroup,
    get_electrode_indices,
)
from spyglass.common.common_interval import interval_list_intersect
from spyglass.lfp.v1 import (
    LFPElectrodeGroup,
    LFPV1,
    LFPArtifactDetection,
)
from spyglass.position.v1 import TrodesPosV1
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.lfp.analysis.v1 import LFPBandV1

from .position_analysis import get_running_intervals, filter_position_ports

from ms_stim_analysis.AnalysisTables.ms_interval import EpochIntervalListName

from ms_stim_analysis.AnalysisTables.ms_opto_stim_protocol import OptoStimProtocol
from .utils import (
    weighted_quantile,
    convert_delta_marks_to_timestamp_values,
    filter_opto_data,
)
from .circular_shuffle import (
    normalize_by_index_wrapper,
    shuffled_trace_distribution,
    bootstrap,
    trace_median,
)
from ms_stim_analysis.Style.style_guide import interval_style

logger = logging.getLogger("example_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("/home/sambray/Desktop/wtrack_lfp.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


LFP_AMP_CUTOFF = 2000


################################################################################


def get_yaml_defined_reference_electrode(key: dict) -> int:
    electrode_group_name = (ElectrodeGroup & key & {"description": "reference"}).fetch(
        "electrode_group_name"
    )[0]
    logger.info(f"key: {key}")
    logger.info(f"electrode_group_name: {electrode_group_name}")
    logger.info("     ")
    e_group_name_list = (
        LFPV1 & key & {"electrode_group_name": electrode_group_name}
    ).fetch("lfp_electrode_group_name")
    targeted_e_group = list(
        set(
            [
                x
                for x in e_group_name_list
                if key["nwb_file_name"].split("_")[0][:-8] in x
            ]
        )
    )[0]
    key["lfp_electrode_group_name"] = targeted_e_group
    ref_electrode_id = (
        LFPElectrodeGroup().LFPElectrode()
        & key
        & {"electrode_group_name": electrode_group_name}
    ).fetch("electrode_id")[0]
    return ref_electrode_id


def get_ref_electrode_index(lfp_s_key: dict) -> Tuple[int, dict]:
    """find the reference electrode index and containing lfp_electrode_group for a
    given lfp selection key, returns the key with the reference electrode group
    name added

    Parameters
    ----------
    lfp_s_key : dict
        lfp selection key with nwb_file_name, lfp_electrode_group_name, filter_name, and filter_sampling_rate

    Returns
    -------
    Tuple[int,dict]
        reference electrode index, lfp selection key with lfp_electrode_group_name added
    """
    nwb_file_name = lfp_s_key["nwb_file_name"]

    e_group_name_list = (LFPV1 & lfp_s_key).fetch("lfp_electrode_group_name")
    if "full_probe" not in e_group_name_list:
        # for terode animals, use the reference electrode defined in the yaml file
        ref_electrode_id = get_yaml_defined_reference_electrode(lfp_s_key)
        return ref_electrode_id, lfp_s_key

    targeted_e_group = [
        x for x in e_group_name_list if nwb_file_name.split("_")[0][:-8] in x
    ]
    if len(targeted_e_group) == 1:
        lfp_s_key["lfp_electrode_group_name"] = targeted_e_group[0]
    elif len(targeted_e_group) > 1:
        print(f"Warning: multiple valid electrode groups for {lfp_s_key}")
        lfp_s_key["lfp_electrode_group_name"] = targeted_e_group[0]
    else:
        lfp_s_key["lfp_electrode_group_name"] = e_group_name_list[0]
    lfp_electrode_ids = [
        (Electrode & lfp_s_key).fetch("original_reference_electrode")[0],
    ]
    # # hack for Olive to fix incorrect reference electrode in nwb file
    # if "Olive" in nwb_file_name:
    #     lfp_electrode_ids = [48]
    return lfp_electrode_ids[0], lfp_s_key


def power_spectrum(
    data: np.ndarray,
    window_size: int,
    sampling_rate: float = 1000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """calculate the power spectrum density of a signal

    Parameters
    ----------
    data : np.ndarray
        signal to analyze
    window_size : int
        window size in index units
    sampling_rate : float, optional
        sampling rate of the signal, by default 1000

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        frequencies, power spectrum, weight

    """
    window_filt = signal.windows.hamming(window_size, sym=True)
    noverlap = window_size // 2
    frequencies, Pxx = signal.welch(
        data,
        fs=sampling_rate,
        window=window_filt,
        noverlap=noverlap,
        scaling="density",
        nfft=10000,
    )
    weight = np.floor(len(data) - window_size / (window_size - noverlap)) + 1
    return frequencies, Pxx, weight


def get_control_test_power_spectrum(
    nwb_file_name: str,
    epoch: int,
    filter_speed: float,
    filter_name: str,
    window: float,
    pos_interval_name: str = None,
    filter_ports: bool = False,
    return_stim_psd: bool = False,
    dlc_pos=False,
) -> Tuple[list, list, list, list, list]:
    """get the power spectrum for the control (no optogenetics) and test (optogenetics) intervals
    Filters out periods of excessve LFP amplitude
    Filters out periods when the rat is in the ports (optional)
    Filters out periods when the rat is not running (speed threshold)

    Parameters
    ----------
    nwb_file_name : str
        nwbf file name key
    epoch : int
        epoch number under Jen's convention
    filter_speed : float
        speed threshold for running (cm/s)
    filter_name : str
        the lfp filter name
    window : float
        window size in seconds for spectrum analysis
    pos_interval_name : _type_, optional
        interval name for position data, if None, will look up the interval name for the epoch, by default None
    filter_ports : bool, optional
        whether to filter out times when the rat is in the ports, by default False
    return_stim_psd : bool, optional
        whether to return the power spectrum of the driving stimulus, by default False

    Returns
    -------
    Tuple[list,list,list,list,list]
        opto_power_spectrum, control_power_spectrum, opto_weights, control_weights, frequencies.  Note that the weights are the number of windows that went into the power spectrum calculation
    """

    trodes_pos_params_name = "single_led"

    key = {"nwb_file_name": nwb_file_name}
    key.update({"epoch": epoch})
    if pos_interval_name is None:
        pos_interval_name = (EpochIntervalListName() & key).fetch1("interval_list_name")
    key.update({"interval_list_name": pos_interval_name})

    # make intervals where rat is running
    run_intervals = get_running_intervals(
        **key, filter_speed=filter_speed, dlc_pos=dlc_pos
    )
    # intersect with position-defined intervals
    if filter_ports:
        valid_position_intervals = filter_position_ports(
            {"nwb_file_name": nwb_file_name, "interval_list_name": pos_interval_name},
            dlc_pos=dlc_pos,
        )
        if len(valid_position_intervals) == 0:
            return (
                [],
                [],
                [],
                [],
                [],
            )
        run_intervals = interval_list_intersect(
            np.array(run_intervals), np.array(valid_position_intervals)
        )

    from .utils import get_running_valid_intervals

    optogenetic_run_interval, control_run_interval = get_running_valid_intervals(
        key, dlc_pos=dlc_pos
    )

    # Begin analysis
    basic_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_interval_name,
    }
    # get interval times
    interval = (sgc.IntervalList & basic_key).fetch1("valid_times")[0]

    # build keys
    lfp_eg_key = {
        "nwb_file_name": nwb_file_name,
        # "lfp_electrode_group_name": 'full_probe',
    }
    lfp_s_key = lfp_eg_key.copy()
    lfp_s_key["target_interval_list_name"] = pos_interval_name
    lfp_s_key["filter_name"] = filter_name
    lfp_s_key["filter_sampling_rate"] = 30000

    ref_electrode, lfp_s_key = get_ref_electrode_index(lfp_s_key)

    lfp_eseries = (LFPV1 & lfp_s_key).fetch_nwb()[0][
        "lfp"
    ]  # (LFPOutput()).fetch_nwb(restriction=lfp_s_key)[0]["lfp"]
    lfp_elect_indeces = get_electrode_indices(lfp_eseries, [ref_electrode])
    if lfp_elect_indeces[0] > 1000:
        return (
            [],
            [],
            [],
            [],
            [],
        )
    lfp_timestamps = np.asarray(lfp_eseries.timestamps)
    lfp_time_ind = np.where(
        np.logical_and(lfp_timestamps > interval[0], lfp_timestamps < interval[1])
    )[0]
    # filter out high amplitude noise
    # non_noise_intervals = np.array(
    #     noise_free_lfp_intervals(
    #         lfp_s_key,
    #     )
    # )
    non_noise_intervals = np.array(
        noise_free_lfp_intervals_NO_TABLE(
            lfp_timestamps, lfp_eseries.data, LFP_AMP_CUTOFF
        )
    )

    if len(optogenetic_run_interval) == 0 or len(control_run_interval) == 0:
        print(f"Warning: no runs found in either/or control/test test intervals")
        return (
            [],
            [],
            [],
            [],
            [],
        )
    optogenetic_run_interval = interval_list_intersect(
        optogenetic_run_interval, non_noise_intervals
    )
    control_run_interval = interval_list_intersect(
        control_run_interval, non_noise_intervals
    )

    window_size = int(np.round(window / np.mean(np.diff(lfp_timestamps))))
    # load the stimulus data if requested
    if return_stim_psd:
        stim_marks, stim_mark_timestamps = OptoStimProtocol().get_stimulus(basic_key)
        stim, stim_timestamps = convert_delta_marks_to_timestamp_values(
            stim_marks, stim_mark_timestamps, 100
        )
        stim_power_spectrum = []

    # initialize lists
    opto_power_spectrum = []
    opto_weights = []
    control_power_spectrum = []
    control_weights = []
    frequencies = []
    # loop through the run intervals, get the lfp data, and calculate the power spectrum
    for run_ in optogenetic_run_interval:
        lfp_st_ind = np.argmin(np.abs(lfp_timestamps[lfp_time_ind] - run_[0]))
        lfp_end_ind = np.argmin(np.abs(lfp_timestamps[lfp_time_ind] - run_[1]))
        lfp_sample = lfp_eseries.data[
            lfp_time_ind[lfp_st_ind] : lfp_time_ind[lfp_end_ind], lfp_elect_indeces
        ][:, 0]
        # skip if too short an interval
        if window_size > lfp_sample.size:
            continue
        # calculate the spectrum
        frequencies, Pxx, weight = power_spectrum(lfp_sample, window_size)
        opto_power_spectrum.append(Pxx)
        opto_weights.append(weight)
        # calculate the stimulus spectrum if requested
        if return_stim_psd:
            stim_st_ind = np.argmin(np.abs(stim_timestamps - run_[0]))
            stim_end_ind = np.argmin(np.abs(stim_timestamps - run_[1]))
            stim_sample = stim[stim_st_ind:stim_end_ind]
            frequencies, Pxx, weight = power_spectrum(
                stim_sample, int(window_size / 10), 100
            )

            stim_power_spectrum.append(Pxx)
    for run_ in control_run_interval:
        lfp_st_ind = np.argmin(np.abs(lfp_timestamps[lfp_time_ind] - run_[0]))
        lfp_end_ind = np.argmin(np.abs(lfp_timestamps[lfp_time_ind] - run_[1]))
        lfp_sample = lfp_eseries.data[
            lfp_time_ind[lfp_st_ind] : lfp_time_ind[lfp_end_ind], lfp_elect_indeces
        ][:, 0]
        # skip if too short an interval
        if window_size > lfp_sample.size:
            continue
        # calculate the spectrum
        frequencies, Pxx, weight = power_spectrum(lfp_sample, window_size)
        control_power_spectrum.append(Pxx)
        control_weights.append(weight)

    if return_stim_psd:
        return (
            opto_power_spectrum,
            control_power_spectrum,
            opto_weights,
            control_weights,
            frequencies,
            stim_power_spectrum,
        )
    return (
        opto_power_spectrum,
        control_power_spectrum,
        opto_weights,
        control_weights,
        frequencies,
    )


def get_control_test_power_spectrum_full_probe(
    nwb_file_name: str,
    epoch: int,
    filter_speed: float,
    filter_name: str,
    window: float,
    pos_interval_name: str = None,
    filter_ports: bool = False,
    return_stim_psd: bool = False,
) -> Tuple[list, list, list, list, list]:
    """get the power spectrum for the control (no optogenetics) and test (optogenetics) intervals
    Filters out periods of excessve LFP amplitude
    Filters out periods when the rat is in the ports (optional)
    Filters out periods when the rat is not running (speed threshold)

    Parameters
    ----------
    nwb_file_name : str
        nwbf file name key
    epoch : int
        epoch number under Jen's convention
    filter_speed : float
        speed threshold for running (cm/s)
    filter_name : str
        the lfp filter name
    window : float
        window size in seconds for spectrum analysis
    pos_interval_name : _type_, optional
        interval name for position data, if None, will look up the interval name for the epoch, by default None
    filter_ports : bool, optional
        whether to filter out times when the rat is in the ports, by default False
    return_stim_psd : bool, optional
        whether to return the power spectrum of the driving stimulus, by default False

    Returns
    -------
    Tuple[list,list,list,list,list]
        opto_power_spectrum, control_power_spectrum, opto_weights, control_weights, frequencies.  Note that the weights are the number of windows that went into the power spectrum calculation
    """

    trodes_pos_params_name = "single_led"

    key = {"nwb_file_name": nwb_file_name}
    key.update({"epoch": epoch})
    if pos_interval_name is None:
        pos_interval_name = (EpochIntervalListName() & key).fetch1("interval_list_name")
    key.update({"interval_list_name": pos_interval_name})

    # make intervals where rat is running
    run_intervals = get_running_intervals(**key, filter_speed=filter_speed)
    # intersect with position-defined intervals
    if filter_ports:
        valid_position_intervals = filter_position_ports(
            {"nwb_file_name": nwb_file_name, "interval_list_name": pos_interval_name}
        )
        if len(valid_position_intervals) == 0:
            return (
                [],
                [],
                [],
                [],
                [],
            )
        run_intervals = interval_list_intersect(
            np.array(run_intervals), np.array(valid_position_intervals)
        )

    # # determine if each interval is in the optogenetic control interval
    control_interval = (OptoStimProtocol() & key).fetch1("control_intervals")
    test_interval = (OptoStimProtocol() & key).fetch1("test_intervals")
    if len(control_interval) == 0 or len(test_interval) == 0:
        print(f"Warning: no optogenetic intervals found for {key}")
        return (
            [],
            [],
            [],
            [],
            [],
        )
    optogenetic_run_interval = interval_list_intersect(
        np.array(run_intervals), np.array(test_interval)
    )
    control_run_interval = interval_list_intersect(
        np.array(run_intervals), np.array(control_interval)
    )

    # Begin analysis
    basic_key = {
        "nwb_file_name": nwb_file_name,
        "interval_list_name": pos_interval_name,
    }
    # get interval times
    interval = (sgc.IntervalList & basic_key).fetch1("valid_times")[0]

    # build keys
    lfp_eg_key = {
        "nwb_file_name": nwb_file_name,
        # "lfp_electrode_group_name": 'full_probe',
    }
    lfp_s_key = lfp_eg_key.copy()
    lfp_s_key["target_interval_list_name"] = pos_interval_name
    lfp_s_key["filter_name"] = filter_name
    lfp_s_key["filter_sampling_rate"] = 30000

    ref_electrode, lfp_s_key = get_ref_electrode_index(lfp_s_key)

    lfp_eseries = (LFPOutput).fetch_nwb(restriction=lfp_s_key)[0]["lfp"]
    lfp_elect_indeces = get_electrode_indices(lfp_eseries, [ref_electrode])
    if lfp_elect_indeces[0] > 1000:
        return (
            [],
            [],
            [],
            [],
            [],
        )
    lfp_timestamps = np.asarray(lfp_eseries.timestamps)
    lfp_time_ind = np.where(
        np.logical_and(lfp_timestamps > interval[0], lfp_timestamps < interval[1])
    )[0]
    # filter out high amplitude noise
    non_noise_intervals = np.array(
        noise_free_lfp_intervals(
            lfp_s_key,
        )
    )
    if len(optogenetic_run_interval) == 0 or len(control_run_interval) == 0:
        print(f"Warning: no runs found in either/or control/test test intervals")
        return (
            [],
            [],
            [],
            [],
            [],
        )
    optogenetic_run_interval = interval_list_intersect(
        optogenetic_run_interval, non_noise_intervals
    )
    control_run_interval = interval_list_intersect(
        control_run_interval, non_noise_intervals
    )

    window_size = int(np.round(window / np.mean(np.diff(lfp_timestamps))))
    # load the stimulus data if requested
    if return_stim_psd:
        stim_marks, stim_mark_timestamps = OptoStimProtocol().get_stimulus(basic_key)
        stim, stim_timestamps = convert_delta_marks_to_timestamp_values(
            stim_marks, stim_mark_timestamps, 100
        )
        stim_power_spectrum = []

    # reset indecies to collect all, not just reference electrode
    lfp_elect_indeces = np.arange(lfp_eseries.data.shape[1])

    # initialize lists
    opto_power_spectrum = [[] for _ in lfp_elect_indeces]
    opto_weights = []
    control_power_spectrum = [[] for _ in lfp_elect_indeces]
    control_weights = []
    frequencies = []
    # loop through the run intervals, get the lfp data, and calculate the power spectrum
    for run_ in optogenetic_run_interval:
        lfp_st_ind = np.argmin(np.abs(lfp_timestamps[lfp_time_ind] - run_[0]))
        lfp_end_ind = np.argmin(np.abs(lfp_timestamps[lfp_time_ind] - run_[1]))
        lfp_sample = lfp_eseries.data[
            lfp_time_ind[lfp_st_ind] : lfp_time_ind[lfp_end_ind], lfp_elect_indeces
        ]  # [:, 0]
        # skip if too short an interval
        if window_size > lfp_sample.shape[0]:
            continue
        # calculate the spectrum
        for i in lfp_elect_indeces:
            frequencies, Pxx, weight = power_spectrum(lfp_sample[:, i], window_size)
            opto_power_spectrum[i].append(Pxx)
        opto_weights.append(weight)
        # calculate the stimulus spectrum if requested
        if return_stim_psd:
            stim_st_ind = np.argmin(np.abs(stim_timestamps - run_[0]))
            stim_end_ind = np.argmin(np.abs(stim_timestamps - run_[1]))
            stim_sample = stim[stim_st_ind:stim_end_ind]
            frequencies, Pxx, weight = power_spectrum(
                stim_sample, int(window_size / 10), 100
            )

            stim_power_spectrum.append(Pxx)
    for run_ in control_run_interval:
        lfp_st_ind = np.argmin(np.abs(lfp_timestamps[lfp_time_ind] - run_[0]))
        lfp_end_ind = np.argmin(np.abs(lfp_timestamps[lfp_time_ind] - run_[1]))
        lfp_sample = lfp_eseries.data[
            lfp_time_ind[lfp_st_ind] : lfp_time_ind[lfp_end_ind], lfp_elect_indeces
        ]
        # skip if too short an interval
        if window_size > lfp_sample.shape[0]:
            continue
        # print(window_size,lfp_sample.shape)
        # calculate the spectrum
        for i in lfp_elect_indeces:
            frequencies, Pxx, weight = power_spectrum(lfp_sample[:, i], window_size)
            control_power_spectrum[i].append(Pxx)
        control_weights.append(weight)

    if return_stim_psd:
        return (
            opto_power_spectrum,
            control_power_spectrum,
            opto_weights,
            control_weights,
            frequencies,
            stim_power_spectrum,
        )
    return (
        opto_power_spectrum,
        control_power_spectrum,
        opto_weights,
        control_weights,
        frequencies,
    )


def opto_spectrum_analysis(
    dataset_key: dict,
    filter_name: str = "LFP 0-400 Hz",
    filter_speed: float = 10.0,
    window: float = 1.0,
    return_distributions: bool = False,
    dlc_pos=False,
):
    """Generates a figure with the power spectrum for the control and test intervals, and the distribution of entrainment statistics
    Normalizes the spectrum power on a per-animal basis, using control interval peak as reference


    Parameters
    ----------
    dataset_key : dict
        key containing parameters which define the dataset to analyze
    filter_name : str, optional
        which lfp filter to analyze, by default "LFP 0-400 Hz"
    filter_speed : float, optional
        the minimum speed at which the animal moves for data to be included, by default 10.0
    window : float, optional
        window size for the PSD estimate through welch's method, by default 1.0
    return_distributions : bool, optional
        whether to return the entrainment spectrum distributions with the generated figure, by default False

    Returns
    -------
    matplotlib.figure.Figure
        figure of power spectrum and statistics
    Tuple
        distributions of entrainment statistics (optional)
    """
    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)

    # # compile the data
    # opto_power_spectrum = []
    # control_power_spectrum = []
    # opto_weight = []
    # control_weight = []
    # f = []
    # for nwb_file_name, interval_name in tqdm(
    #     zip(dataset.fetch("nwb_file_name"), dataset.fetch("interval_list_name"))
    # ):
    #     key = {"nwb_file_name": nwb_file_name, "interval_list_name": interval_name}
    #     if len(TrodesPosV1 & key) == 0:
    #         print("missing position:", key)
    #         continue
    #     lfp_key = {
    #         "nwb_file_name": nwb_file_name,
    #         "target_interval_list_name": interval_name,
    #     }
    #     if len(LFPV1 & lfp_key) == 0:
    #         print("missing lfp:", key)
    #         continue
    #     print(key)
    #     # epoch = (EpochIntervalListName() &{"nwb_file_name":nwb_file_name,"interval_list_name":interval_name}).fetch1("epoch")
    #     opto_, control_, opto_w_, control_w_, f_ = get_control_test_power_spectrum(
    #         nwb_file_name,
    #         None,
    #         filter_speed,
    #         filter_name,
    #         window,
    #         pos_interval_name=interval_name,
    #         filter_ports=1,
    #     )
    #     if len(f_) > 0:
    #         f = f_.copy()
    #     opto_power_spectrum.extend(opto_)
    #     control_power_spectrum.extend(control_)
    #     control_weight.extend(control_w_)
    #     opto_weight.extend(opto_w_)
    # # weight the data by sample lengths
    # control_power_spectrum = np.array(
    #     control_power_spectrum
    # )  # * (np.array(control_weight)[:,None] / np.mean(control_weight))
    # opto_power_spectrum = np.array(
    #     opto_power_spectrum
    # )  # * (np.array(opto_weight)[:,None] / np.mean(opto_weight))
    # if control_power_spectrum.size == 0 or opto_power_spectrum.size == 0:
    #     if return_distributions:
    #         return None, [[np.nan, np.nan], [np.nan, np.nan]]
    #     return

    # compile the data
    print("datasets:", len(dataset))
    opto_power_spectrum = {}
    control_power_spectrum = {}
    opto_weight = {}
    control_weight = {}
    f = []

    for nwb_file_name, interval_name in tqdm(
        zip(dataset.fetch("nwb_file_name"), dataset.fetch("interval_list_name"))
    ):
        key = {"nwb_file_name": nwb_file_name, "interval_list_name": interval_name}
        if len(TrodesPosV1 & key) == 0:
            print("missing position:", key)
            continue
        lfp_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": interval_name,
        }
        if len(LFPV1 & lfp_key) == 0:
            print("missing lfp:", key)
            continue
        print(key)
        # get animal name
        animal = (Session & key).fetch1("subject_id")
        if not animal in opto_power_spectrum:
            opto_power_spectrum[animal] = []
            control_power_spectrum[animal] = []
            opto_weight[animal] = []
            control_weight[animal] = []
        opto_, control_, opto_w_, control_w_, f_ = get_control_test_power_spectrum(
            nwb_file_name,
            None,
            filter_speed,
            filter_name,
            window,
            pos_interval_name=interval_name,
            filter_ports=1,
            dlc_pos=dlc_pos,
        )
        if len(f_) > 0:
            f = f_.copy()
        opto_power_spectrum[animal].extend(opto_)
        control_power_spectrum[animal].extend(control_)
        control_weight[animal].extend(control_w_)
        opto_weight[animal].extend(opto_w_)
    # normalize each animal by peak of control condition weighted median
    control_power_spectrum_combined = []
    opto_power_spectrum_combined = []
    opto_weight_combined = []
    control_weight_combined = []
    for animal in opto_power_spectrum.keys():
        x = np.array(control_power_spectrum[animal])
        if x.size == 0:
            continue
        if len(x.shape) == 1:
            x = x[:, None]
        w = control_weight[animal]
        animal_norm = np.max(
            [weighted_quantile(x[:, i], [0.5], w) for i in range(x.shape[1])]
        )
        control_power_spectrum_combined.extend(list(x.copy() / animal_norm))
        control_weight_combined.extend(list(w.copy()))
        x = np.array(opto_power_spectrum[animal])
        w = opto_weight[animal]
        # animal_norm = np.max([weighted_quantile(x[:,i],[.5],w) for i in range(x.shape[1])])
        opto_power_spectrum_combined.extend(list(x.copy() / animal_norm))
        opto_weight_combined.extend(list(w.copy()))
    control_power_spectrum = np.array(control_power_spectrum_combined)
    opto_power_spectrum = np.array(opto_power_spectrum_combined)
    control_weight = np.array(control_weight_combined)
    opto_weight = np.array(opto_weight_combined)
    if control_power_spectrum.size == 0 or opto_power_spectrum.size == 0:
        if return_distributions:
            return None, [[np.nan, np.nan], [np.nan, np.nan]]
        return

    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(
        1,
        3,
        width_ratios=[3, 1, 1],
    )

    # plot spectrums
    ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    quantiles = [0.5, 0.25, 0.75]
    opto_stats = np.array(
        [
            weighted_quantile(opto_power_spectrum[:, i], quantiles, opto_weight)
            for i in range(opto_power_spectrum.shape[1])
        ]
    )
    control_stats = np.array(
        [
            weighted_quantile(control_power_spectrum[:, i], quantiles, control_weight)
            for i in range(control_power_spectrum.shape[1])
        ]
    )
    n_control = len(control_power_spectrum)
    n_opto = len(opto_power_spectrum)
    c = interval_style["control"]
    ax[0].plot(f, control_stats[:, 0], c=c, label=f"stim protocol OFF, n={n_control}")
    ax[0].fill_between(
        f, control_stats[:, 1], control_stats[:, 2], alpha=0.4, facecolor=c
    )
    c = interval_style["test"]
    ax[0].plot(f, opto_stats[:, 0], c=c, label=f"stim protocol ON, n={n_opto}")
    ax[0].fill_between(f, opto_stats[:, 1], opto_stats[:, 2], alpha=0.4, facecolor=c)
    ax[0].set_xlim(0, 35)
    ax[0].set_xlabel("frequencies")
    ax[0].set_ylabel("power")
    ax[0].legend()

    # get entrainment statistics
    if "period_ms" in dataset_key:
        drive_freq = np.round(1000 / dataset_key["period_ms"], 2)
    else:
        drive_freq = np.nan  # case for closed-loop
    # plot entrainment statistic distributions
    scores = entrainment_statistics_distribution(
        opto_power_spectrum, control_power_spectrum, f, drive_freq
    )
    ax[1].violinplot(
        [*scores],
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    ax[1].scatter([1, 2], [np.mean(s) for s in scores], c="k")
    ax[1].set_xticks([1, 2])
    ax[1].set_xticklabels(["suppression", "entrainment"], rotation=-45, ha="left")
    ax[1].set_ylim(-1, 1)
    ax[1].set_yticks([-1, -0.5, 0, 0.5, 1])

    # Table with information about the dataset
    the_table = ax[2].table(
        cellText=[[str(drive_freq)], [len(dataset)]]
        + [[str(x)] for x in dataset_key.values()],
        rowLabels=["drive_freq", "number_epochs"]
        + [str(x) for x in dataset_key.keys()],
        loc="right",
        colWidths=[0.6, 0.6],
    )
    ax[2].spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    fig.canvas.draw()
    plt.rcParams["svg.fonttype"] = "none"

    if return_distributions:
        return fig, scores
    return fig


def opto_spectrum_analysis_full_probe(
    dataset_key: dict,
    filter_name: str = "LFP 0-400 Hz",
    filter_speed: float = 10.0,
    window: float = 1.0,
    return_distributions: bool = False,
):
    """Generates a figure with the power spectrum for the control and test intervals, and the distribution of entrainment statistics
    Normalizes the spectrum power on a per-animal basis, using control interval peak as reference


    Parameters
    ----------
    dataset_key : dict
        key containing parameters which define the dataset to analyze
    filter_name : str, optional
        which lfp filter to analyze, by default "LFP 0-400 Hz"
    filter_speed : float, optional
        the minimum speed at which the animal moves for data to be included, by default 10.0
    window : float, optional
        window size for the PSD estimate through welch's method, by default 1.0
    return_distributions : bool, optional
        whether to return the entrainment spectrum distributions with the generated figure, by default False

    Returns
    -------
    matplotlib.figure.Figure
        figure of power spectrum and statistics
    Tuple
        distributions of entrainment statistics (optional)
    """
    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)

    # compile the data
    print("datasets:", len(dataset))
    opto_power_spectrum = {}
    control_power_spectrum = {}
    opto_weight = {}
    control_weight = {}
    f = []

    for nwb_file_name, interval_name in tqdm(
        zip(dataset.fetch("nwb_file_name"), dataset.fetch("interval_list_name"))
    ):
        key = {"nwb_file_name": nwb_file_name, "interval_list_name": interval_name}
        if len(TrodesPosV1 & key) == 0:
            print("missing position:", key)
            continue
        lfp_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": interval_name,
        }
        if len(LFPV1 & lfp_key) == 0:
            print("missing lfp:", key)
            continue
        print(key)
        # get animal name
        animal = (Session & key).fetch1("subject_id")
        (
            opto_,
            control_,
            opto_w_,
            control_w_,
            f_,
        ) = get_control_test_power_spectrum_full_probe(
            nwb_file_name,
            None,
            filter_speed,
            filter_name,
            window,
            pos_interval_name=interval_name,
            filter_ports=1,
            dlc_pos=dlc_pos,
        )
        if len(f_) > 0:
            f = f_.copy()

        # add the results in for the animal
        if not animal in opto_power_spectrum:
            opto_power_spectrum[animal] = opto_
            control_power_spectrum[animal] = control_
            opto_weight[animal] = opto_w_
            control_weight[animal] = control_w_
        else:
            for i in range(len(opto_)):
                opto_power_spectrum[animal][i].extend(opto_[i])
                control_power_spectrum[animal][i].extend(control_[i])
            control_weight[animal].extend(control_w_)
            opto_weight[animal].extend(opto_w_)
        # get the probe position
        rel_position = (
            (LFPV1() & lfp_key)
            .fetch_nwb()[0]["lfp"]
            .electrodes.to_dataframe()
            .rel_y.values
        )

    if len(opto_power_spectrum) == 0:
        if return_distributions:
            return None, [[np.nan, np.nan], [np.nan, np.nan]]
        return
    # normalize each animal by peak of weighted median (CURRENTLY NORMED BY CHANNEL)
    control_power_spectrum_combined = [
        [] for i in range(len(opto_power_spectrum[animal]))
    ]
    opto_power_spectrum_combined = [[] for i in range(len(opto_power_spectrum[animal]))]
    opto_weight_combined = []
    control_weight_combined = []
    for animal in opto_power_spectrum.keys():
        for i in range(len(opto_power_spectrum[animal])):
            x = np.array(control_power_spectrum[animal][i])
            if x.size == 0:
                continue
            if len(x.shape) == 1:
                x = x[:, None]
            w = control_weight[animal]
            animal_norm = np.max(
                [weighted_quantile(x[:, ii], [0.5], w) for ii in range(x.shape[1])]
            )
            animal_norm = 1
            control_power_spectrum_combined[i].extend(list(x.copy() / animal_norm))
            control_weight_combined.extend(list(w.copy()))
            x = np.array(opto_power_spectrum[animal][i])
            w = opto_weight[animal]
            # animal_norm = np.max([weighted_quantile(x[:,i],[.5],w) for i in range(x.shape[1])])
            opto_power_spectrum_combined[i].extend(list(x.copy() / animal_norm))
            opto_weight_combined.extend(list(w.copy()))
    control_power_spectrum = [np.array(x) for x in control_power_spectrum_combined]
    opto_power_spectrum = [np.array(x) for x in opto_power_spectrum_combined]
    control_weight = np.array(control_weight_combined)
    opto_weight = np.array(opto_weight_combined)
    if control_power_spectrum[0].size == 0 or opto_power_spectrum[0].size == 0:
        if return_distributions:
            return None, [[np.nan, np.nan], [np.nan, np.nan]]
        return

    fig = plt.figure(
        figsize=(20, 5),
    )
    gs = GridSpec(
        2,
        4,
        width_ratios=[1, 1, 1, 1],
    )

    # plot spectrums
    ax = [
        fig.add_subplot(gs[:, 0]),
    ]
    ax.append(fig.add_subplot(gs[:, 1], sharey=ax[0]))
    quantiles = [0.5, 0.25, 0.75]

    # loop through electrodes
    for ii in range(len(opto_power_spectrum)):
        color = plt.cm.viridis(ii / len(opto_power_spectrum))

        opto_stats = np.array(
            [
                weighted_quantile(opto_power_spectrum[ii][:, i], quantiles, opto_weight)
                for i in range(opto_power_spectrum[ii].shape[1])
            ]
        )
        control_stats = np.array(
            [
                weighted_quantile(
                    control_power_spectrum[ii][:, i], quantiles, control_weight
                )
                for i in range(control_power_spectrum[ii].shape[1])
            ]
        )

        ax[0].plot(f, control_stats[:, 0], c=color, label="ii", lw=1)
        ax[0].fill_between(
            f, control_stats[:, 1], control_stats[:, 2], alpha=0.1, facecolor=color
        )
        ax[1].plot(f, opto_stats[:, 0], c=color, label="ii", lw=1)
        ax[1].fill_between(
            f, opto_stats[:, 1], opto_stats[:, 2], alpha=0.1, facecolor=color
        )
    for a in ax[:2]:
        a.set_xlim(0, 35)
        a.set_xlabel("frequencies")
    ax[0].set_ylabel("power")
    ax[1].set_yticklabels([])
    # ax[0].legend()

    # get entrainment statistics
    if "period_ms" in dataset_key:
        drive_freq = np.round(1000 / dataset_key["period_ms"], 2)
    else:
        drive_freq = np.nan  # case for closed-loop

    all_scores = None
    score_ax = [fig.add_subplot(gs[0, 2])]
    score_ax.append(fig.add_subplot(gs[1, 2], sharex=score_ax[0]))
    for ii in range(len(opto_power_spectrum)):
        color = plt.cm.viridis(ii / len(opto_power_spectrum))
        # plot entrainment statistic distributions
        scores = entrainment_statistics_distribution(
            opto_power_spectrum[ii], control_power_spectrum[ii], f, drive_freq
        )
        if all_scores is None:
            all_scores = [[] for score in scores]
        for i, score in enumerate(scores):
            all_scores[i].append(score)

        for a, score in zip(score_ax, scores):
            violin = a.violinplot(
                [*score],
                positions=[rel_position[ii]],
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for pc in violin["bodies"]:
                pc.set_facecolor(color)

            a.scatter([rel_position[ii]], [np.mean(score)], c=color)
            a.set_ylim(-1, 1)
    score_ax[0].set_ylabel("suppression")
    score_ax[1].set_ylabel("entrainment")
    score_ax[1].set_xlabel("relative position")
    score_ax[0].set_xticks([])

    # Table with information about the dataset
    table_ax = fig.add_subplot(gs[:, 3])
    the_table = table_ax.table(
        cellText=[[str(drive_freq)], [len(dataset)]]
        + [[str(x)] for x in dataset_key.values()],
        rowLabels=["drive_freq", "number_epochs"]
        + [str(x) for x in dataset_key.keys()],
        loc="right",
        colWidths=[0.6, 0.6],
    )
    table_ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    table_ax.set_xticks([])
    table_ax.set_yticks([])
    fig.canvas.draw()
    plt.rcParams["svg.fonttype"] = "none"

    if return_distributions:
        return fig, all_scores
    return fig


def noise_free_lfp_intervals(lfp_key: dict) -> list:
    """finds intervals where the lfp is below a threshold value. Use for filtering high-amplitude noise

    Parameters
    ----------
    lfp_key : dict
        selection key for the lfp artifact detection table

    Returns
    -------
    list
        intervals where amplitude artifacts are not detected
    """
    if not isinstance(lfp_key, dict):
        raise ValueError(
            "noise_free_lfp_intervals now uses the LFPArtifactDetection class, and requires a dictionary input",
            "Please update your code to use the new method signature",
        )
    return (LFPArtifactDetection & lfp_key).fetch1("artifact_removed_valid_times")


def noise_free_lfp_intervals_NO_TABLE(
    time: list, lfp: list, threshold=LFP_AMP_CUTOFF
) -> list:
    """finds intervals where the lfp is below a threshold value. Use for filtering high-amplitude noise
    This function is not used in the current pipeline, but is kept for reference
    Parameters
    ----------
    time : list
        lfp timestamps
    lfp : list
        lfp signal
    threshold : int, optional
        amplitude threshold value (in uV), by default 700

    Returns
    -------
    list
        intervals where signal amplitude is below threshold value
    """
    ind_valid = np.where(np.abs(lfp) < threshold)[0]
    interval_breaks = np.append([0], np.where(np.diff(ind_valid) > 1)[0] + 1)
    interval_breaks = np.append(interval_breaks, [len(ind_valid) - 1])
    valid_intervals = [
        [time[ind_valid[interval_breaks[i]]], time[ind_valid[interval_breaks[i + 1]]]]
        for i in range(len(interval_breaks) - 1)
    ]
    # print("valid intervals: ", valid_intervals)
    return valid_intervals


def entrainment_statistics_distribution(
    test_spectrums, control_spectrums, frequencies, target_frequency
):
    """Returns the distributions of entrainment statistics for a list of test and control spectrums

    Parameters
    ----------
    test_spectrums : _type_
        List of spectrums from STIM ON
    control_spectrums : _type_
        List of spectrums from STIM OFF
    frequencies : _type_
        Frequencies for the spectrum
    target_frequency : _type_
        Target frequency for entrainment

    Returns
    -------
    Tuple
        The distribution of scores for each statistic
    """
    statistics = []
    for test_spectrum in test_spectrums:
        for control_spectrum in control_spectrums:
            statistics.append(
                list(
                    entrainment_statistics(
                        test_spectrum, control_spectrum, frequencies, target_frequency
                    )
                )
            )
    statistics = np.asarray(statistics)
    statistics = tuple([statistics[:, i] for i in range(statistics.shape[1])])
    return statistics


def entrainment_statistics(
    test_spectrum, control_spectrum, frequencies, target_frequency
):
    """Calculate the entrainment statistics for a given test-control spectrum pair

    Parameters
    ----------
    test_spectrum : _type_
        A single spectrum from STIM ON
    control_spectrum : _type_
        A single spectrum from STIM OFF
    frequencies : _type_
        Frequencies for the spectrum
    target_frequency : _type_
        Target frequency for entrainment

    Returns
    -------
    Tuple
        the suppression score and entrainment score
    """
    return calculate_suppression_score(
        test_spectrum, control_spectrum, frequencies
    ), calculate_entrainment_score(
        test_spectrum, control_spectrum, frequencies, target_frequency
    )


def calculate_suppression_score(test_spectrum, control_spectrum, frequencies):
    """calculate the suppression score for a given frequency

    Parameters
    ----------
    test_spectrum : np.ndarray
        power spectrum for the test interval
    control_spectrum : np.ndarray
        power spectrum for the control interval

    Returns
    -------
    float
        suppression score
    """
    theta_rng = np.where(np.logical_and(frequencies > 4, frequencies < 12))[0]
    peak_freq = theta_rng[np.argmax(control_spectrum[theta_rng])]
    # print(peak_freq)
    # calculate the suppression score
    suppression_score = 1 - test_spectrum[peak_freq] / control_spectrum[peak_freq]
    suppression_score = (control_spectrum[peak_freq] - test_spectrum[peak_freq]) / (
        test_spectrum[peak_freq] + control_spectrum[peak_freq]
    )
    return suppression_score


def calculate_entrainment_score(
    test_spectrum, control_spectrum, frequencies, target_frequency
):
    """calculate the entrainment score for a given frequency

    Parameters
    ----------
    test_spectrum : np.ndarray
        power spectrum for the test interval
    control_spectrum : np.ndarray
        power spectrum for the control interval
    target_frequency : float
        target frequency for entrainment

    Returns
    -------
    float
        entrainment score
    """
    if np.isnan(target_frequency):
        return np.nan  # case for closed-loop

    target_ind = np.argmin(np.abs(frequencies - target_frequency))
    # calculate the entrainment score
    entrainment_score = (test_spectrum[target_ind] - control_spectrum[target_ind]) / (
        control_spectrum[target_ind] + test_spectrum[target_ind]
    )
    return entrainment_score


########################################################
"""Non-Spectrum Analysis Functions"""
########################################################


def lfp_per_pulse_analysis(
    dataset_key: dict,
    filter_name: str = "LFP 0-400 Hz",
    band_filter_name: str = "Theta 5-11 Hz",
    lfp_trace_window=(-int(0.125 * 1000), int(0.125 * 1000)),
    pulse_number_list=np.arange(10),
    fig=None,
    color="cornflowerblue",
    return_data=False,
    traces_only=False,
    circular_shuffle=False,
    limit_1_epoch=False,
):
    """Generates a figure characterizing the lfp around each stimulus pulse

    Parameters
    ----------
    dataset_key : dict
        key containing parameters which define the dataset to analyze
    filter_name : str, optional
        which lfp filter to analyze, by default "LFP 0-400 Hz"
    band_filter_name : str, optional
        which lfp band to analyze, by default "Theta 5-11 Hz"
    lfp_trace_window : tuple, optional
        time around a pulse to includ in analysis
    pulse_number_list : np.ndarray, optional
        which pulses to include in analysis (pulse 0 is first pulse in a cycle)
    fig : matplotlib.figure.Figure, optional
        figure to plot on, by default None which creates new figure
    color : str, optional
        color to plot this dataset, by default "cornflowerblue"
    return_data : bool, optional
        whether to return the lfp traces amplitudes and phases, by default False
    circular_shuffle : bool, optional
        whether to circularly shuffle the lfp traces, by default False
    limit_1_epoch : bool, optional
        whether to limit the analysis to 1 epoch per dataset, by default False
    Returns
    -------
    matplotlib.figure.Figure
        figure of power spectrum and statistics
    Tuple
        distributions of entrainment statistics (optional)
    """
    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)

    # Use the driving period lfp band if band filter not specified
    if band_filter_name is None:
        if "period_ms" not in dataset_key:
            raise ValueError("band_filter_name must be specified if period_ms is not")
        band_filter_name = f"ms_stim_{dataset_key['period_ms']}ms_period"

    #################################################################
    # make figure
    if fig is None:
        if traces_only:
            fig = plt.figure(tight_layout=True, figsize=(3, 6))
            ax = [fig.gca()]
        else:
            fig = plt.figure(tight_layout=True, figsize=(6, 9))
            ncol = len(pulse_number_list)
            gs_ = gridspec.GridSpec(4, ncol)
            ax = []
            ax.append(fig.add_subplot(gs_[:2, : ncol // 2]))
            ax.append(fig.add_subplot(gs_[:2, ncol // 2 :]))
            ax_horiz = fig.add_subplot(gs_[2, :])
            ax_rose = [fig.add_subplot(gs_[3, i], polar=True) for i in range(ncol)]
    else:
        if traces_only:
            ax = [fig.gca()]
        else:
            ax = fig.get_axes()[:2]
            ax_horiz = fig.get_axes()[2]
            ax_rose = fig.get_axes()[3:]
    nwb_file_name_list = dataset.fetch("nwb_file_name")
    interval_list_name_list = dataset.fetch("interval_list_name")
    if limit_1_epoch:
        nwb_file_name_list = nwb_file_name_list[:1]
        interval_list_name_list = interval_list_name_list[:1]

    # get the lfp traces for every relevant pulse
    lfp_traces = [[] for p in pulse_number_list]
    # get time-shuffled lfp_traces for every relevant pulse
    lfp_traces_shuffled = [[] for p in pulse_number_list]
    # get the lfp phase for every relevant pulse
    lfp_phase = [[] for p in pulse_number_list]
    # get the average lfp power within the window for every relevant pulse
    # lfp_power = [[] for p in pulse_number_list]
    count = 0
    for nwb_file_name, interval_list_name in zip(
        nwb_file_name_list, interval_list_name_list
    ):
        basic_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": interval_list_name,
            "filter_name": filter_name,
        }
        stim_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "dio_event_name": "stim",
        }
        band_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": interval_list_name,
            "filter_name": band_filter_name,
        }
        if len(LFPV1() & basic_key) == 0:
            print("missing LFP for: ", basic_key)
            continue
        print(basic_key)
        # get lfp band phase for reference electrode
        ref_elect, basic_key = get_ref_electrode_index(basic_key)  #
        # ref_elect = (Electrode() & basic_key).fetch("original_reference_electrode")[0]
        lfp_eseries = (LFPOutput & basic_key).fetch_nwb(restriction=basic_key)[0]["lfp"]
        ref_index = get_electrode_indices(lfp_eseries, [ref_elect])

        # get LFP series
        lfp_df = (LFPV1() & basic_key).fetch_nwb()[0]["lfp"]
        lfp_df = (LFPV1() & basic_key).fetch1_dataframe()
        lfp_timestamps = lfp_df.index
        lfp_ = np.array(lfp_df[ref_index]).astype(float)
        # nan out artifact intervals
        artifact_times = (LFPArtifactDetection() & basic_key).fetch1("artifact_times")
        for artifact in artifact_times:
            lfp_[
                np.logical_and(
                    lfp_timestamps > artifact[0], lfp_timestamps < artifact[1]
                )
            ] = np.nan

        ind = np.sort(np.unique(lfp_timestamps, return_index=True)[1])
        lfp_timestamps = lfp_timestamps[ind]
        lfp_ = lfp_[ind]
        try:
            assert np.all(np.diff(lfp_timestamps) > 0)
        except:
            continue
        # get phase information
        phase_df = (LFPBandV1() & band_key).compute_signal_phase(
            electrode_list=[ref_elect]
        )
        phase_time = phase_df.index
        phase_ = np.asarray(phase_df)[:, 0]
        # get power information
        # power_df = (LFPBandV1() & band_key).compute_signal_power(
        #     electrode_list=[ref_elect]
        # )
        # power_time = power_df.index
        # print("POWER", np.mean(np.diff(power_time)))
        # power_ = np.asarray(power_df)[:, 0]

        # loop through pulse numbers
        # get times of firt pulse in cylce
        t_mark_cycle = OptoStimProtocol().get_cylcle_begin_timepoints(stim_key)
        # get times of all stimulus
        stim, t_mark = OptoStimProtocol().get_stimulus(stim_key)
        t_mark = t_mark[stim == 1]
        ind_mark = np.digitize(t_mark, lfp_timestamps)
        ind_mark_phase = np.digitize(t_mark, phase_time)
        # label each pulse as its count in the cycle
        pulse_count = np.zeros_like(t_mark)
        mark_ind_cycle = [np.where(t_mark == t_)[0][0] for t_ in t_mark_cycle]
        pulse_count[mark_ind_cycle] = 1
        cycle_id = np.cumsum(pulse_count).astype(
            int
        )  # which pulse cycle each pulse is in
        count = 1
        for i in range(pulse_count.size):
            if pulse_count[i] == 1:
                count = 1
            pulse_count[i] = count
            count += 1
        pulse_count = pulse_count - 1  # 0 index the count
        # normalize each cycle by first pulse response
        lfp_norm = [np.nan for _ in range(np.max(cycle_id) + 1)]
        # find first pulses in each cycle
        ind_pulse_list = ind_mark[pulse_count == 0]
        # assert len(ind_pulse_list) == len(lfp_norm)
        for p, ind in enumerate(ind_pulse_list):
            dat = np.abs(lfp_[ind - 100 : ind + 500])
            dat = dat[dat < LFP_AMP_CUTOFF]
            lfp_norm[p] = np.nanmedian(dat)

        for p, pulse_number in enumerate(pulse_number_list):
            ind_pulse_list = ind_mark[pulse_count == pulse_number]
            ind_pulse_list_phase = ind_mark_phase[pulse_count == pulse_number]
            for lfp_ind, phase_ind, cycle in zip(
                ind_pulse_list, ind_pulse_list_phase, cycle_id
            ):
                if (lfp_ind + lfp_trace_window[0] < 0) or (
                    lfp_ind + lfp_trace_window[1] >= lfp_.size
                ):
                    continue

                # nan out segments with large noise
                if (
                    np.abs(
                        lfp_[
                            lfp_ind
                            + lfp_trace_window[0] : lfp_ind
                            + lfp_trace_window[1]
                        ]
                    ).max()
                    > LFP_AMP_CUTOFF
                ):
                    continue
                if np.isnan(lfp_norm[cycle]):
                    lfp_norm[cycle] = lfp_[
                        lfp_ind + lfp_trace_window[0] : lfp_ind + lfp_trace_window[1]
                    ].max()

                lfp_traces[p].append(
                    lfp_[lfp_ind + lfp_trace_window[0] : lfp_ind + lfp_trace_window[1]]
                    / lfp_norm[cycle]
                )
                # lfp phase
                lfp_phase[p].append(phase_[phase_ind])
                # # lfp power
                # ind_power = np.digitize(
                #     stim_timepoints_ref[int(i + pulse_number)], power_time
                # )
                # lfp_power[p].append(
                #     np.mean(
                #         power_[
                #             ind_power
                #             + lfp_trace_window[0] : ind_power
                #             + lfp_trace_window[1]
                #         ]
                #     )
                # )

            if circular_shuffle:
                # Get bootstrapped statistics for this pulse number and epoch
                marks = t_mark[pulse_count == pulse_number]
                mark_cycle = cycle_id[pulse_count == pulse_number]
                norm_func = normalize_by_index_wrapper(lfp_norm)
                if "period_ms" in dataset_key:
                    shuffle_window = dataset_key["period_ms"] / 1000.0
                else:
                    shuffle_window = 0.125
                lfp_traces_shuffled[p].extend(
                    shuffled_trace_distribution(
                        marks=marks,
                        signal=lfp_,
                        time=lfp_timestamps,
                        marks_id=mark_cycle,
                        shuffle_window=shuffle_window,
                        normalize_func=norm_func,
                        sample_window=lfp_trace_window,
                    )
                )

        lfp_time_seg = (
            lfp_timestamps[
                lfp_ind + lfp_trace_window[0] : lfp_ind + lfp_trace_window[1]
            ]
            - lfp_timestamps[lfp_ind]
        )

        # count += 1
        # if count > 3:
        #     break
    if len(lfp_traces[0]) == 0:
        if return_data:
            return fig, lfp_traces, [], []
        return fig

    # calculate statistics
    peak_rng = np.where((lfp_time_seg * 1000 > -125) & (lfp_time_seg * 1000 < 125))[0]
    # avg_trace
    PLOT_SCALING_FACTOR = 5
    n = [len(lfp_traces[i]) for i in range(len(pulse_number_list))]
    arr = [
        np.nanmedian(np.array(lfp_traces[i]), axis=0) / PLOT_SCALING_FACTOR
        for i in range(len(pulse_number_list))
    ]
    lo = [
        np.nanpercentile(np.array(lfp_traces[i]), 25, axis=0) / PLOT_SCALING_FACTOR
        for i in range(len(pulse_number_list))
    ]
    hi = [
        np.nanpercentile(np.array(lfp_traces[i]), 75, axis=0) / PLOT_SCALING_FACTOR
        for i in range(len(pulse_number_list))
    ]

    # peak_amplitudes = [np.max(np.squeeze(np.array(lfp_traces[i])[:,peak_rng]),axis=1) for i in pulse_number_list]
    peak_amplitudes = [
        np.nanpercentile(
            np.squeeze(np.abs(np.array(lfp_traces[i])[:, peak_rng])), 99, axis=1
        )
        for i in range(len(pulse_number_list))
    ]
    # peak_amplitudes = [x / np.median(lfp_power[0]) for x in lfp_power]

    for loc, i in enumerate(pulse_number_list):
        ax[0].plot(
            lfp_time_seg * 1000,
            arr[loc] - loc,
            c=color,
            alpha=0.9,
            lw=1,
            label="n trials = " + str(n[i]),
        )
        ax[0].fill_between(
            lfp_time_seg * 1000,
            np.squeeze(lo[loc]) - loc,
            np.squeeze(hi[loc]) - loc,
            facecolor=color,
            alpha=0.2,
        )
        if circular_shuffle:
            ref, ref_rng = bootstrap(
                np.squeeze(lfp_traces_shuffled[loc]),
                measurement=trace_median,
                n_samples=np.array(lfp_traces[loc]).shape[0],
                n_boot=1000,
            )
            ax[0].plot(
                lfp_time_seg * 1000,
                ref / PLOT_SCALING_FACTOR - loc,
                c="grey",
                alpha=1,
                lw=1,
            )
            ax[0].fill_between(
                lfp_time_seg * 1000,
                ref_rng[0] / PLOT_SCALING_FACTOR - loc,
                ref_rng[1] / PLOT_SCALING_FACTOR - loc,
                facecolor="grey",
                alpha=0.4,
            )

    if not traces_only:
        # plot peak amplitudes
        bins = np.linspace(0, 2, 30)  # np.linspace(100,1000,30)
        plot_x = (bins[1:] + bins[:-1]) / 2
        plot_base = np.zeros(bins.size - 1)
        for loc, i in enumerate(pulse_number_list):
            val, _ = np.histogram(peak_amplitudes[loc], bins)
            val = val / np.sum(val)
            val = val / np.max(val) * 1.1

            ax[1].fill_between(
                plot_x, plot_base - loc, val - loc, facecolor=color, alpha=0.4
            )
            ax[1].plot(plot_x, plot_base - loc, c=color, lw=1, alpha=0.5)
            ax[1].plot(plot_x, val - loc, c=color, lw=1, alpha=0.5)
            ax[1].scatter(np.nanmedian(peak_amplitudes[loc]), -loc, color=color)

            shift = 0
            ax_horiz.scatter(
                [loc + 0.1 * shift], np.nanmedian(peak_amplitudes[loc]), color=color
            )
            ax_horiz.plot(
                [loc + 0.1 * shift, loc + 0.1 * shift],
                [
                    np.nanpercentile(peak_amplitudes[loc], 25),
                    np.nanpercentile(peak_amplitudes[loc], 75),
                ],
                color=color,
            )
        # plot rose plots of stim phase
        for a, pulse_phase in zip(ax_rose, lfp_phase):
            freq, theta = np.histogram(pulse_phase, bins=50)
            freq = freq / freq.sum()
            theta = (theta[1:] + theta[:-1]) / 2
            width = np.radians(360 / len(theta))
            a.bar(
                theta,
                freq,
                width=width,
                facecolor=color,
                edgecolor=color,
                alpha=0.4,
                align="edge",
            )
            a.set_yticklabels([])
            a.set_xticklabels([])
            # ax.plot([0,mean_angle],[0,freq.max()*r])

    if len(pulse_number_list) > 1:
        ax[0].set_ylabel("train number")
        ax[0].set_yticks(
            -np.arange(len(pulse_number_list)),
        )
        ax[0].set_yticklabels(pulse_number_list)
    else:
        plt.ylabel("normalized LFP")
    ax[0].set_xlabel("time (ms)")  # relative to first train pulse (ms)',fontsize=8)
    ax[0].spines[["right", "top"]].set_visible(False)
    ax[0].set_xlim(lfp_time_seg[0] * 1000, lfp_time_seg[-1] * 1000)
    ax[0].set_ylim(-pulse_number_list.size - 0.1, 1.2)

    if not traces_only:
        ax_horiz.spines[["right", "top"]].set_visible(False)
        ax[1].spines[["right", "top", "left"]].set_visible(False)
        ax[1].set_xlabel("normalized peak LFP amplitude")

        ax_horiz.set_xlabel("pulse_number")
        ax_horiz.set_ylabel("normalized peak LFP amplitude")

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(left=0.5)
    plt.rcParams["svg.fonttype"] = "none"

    if return_data:
        return fig, lfp_traces, peak_amplitudes, lfp_phase
    return fig


def lfp_power_dynamics_pulse_hilbert(
    dataset_key: dict,
    filter_name: str = "LFP 0-400 Hz",
    band_filter_name: str = "Theta 5-11 Hz",
    lfp_trace_window=(-int(0.125 * 1000), int(1000)),
    fig=None,
    color="cornflowerblue",
    return_data=False,
    norm_window=None,
):
    """Generates a figure characterizing the lfp around each stimulus cycle

    Parameters
    ----------
    dataset_key : dict
        key containing parameters which define the dataset to analyze
    filter_name : str, optional
        which lfp filter to analyze, by default "LFP 0-400 Hz"
    band_filter_name : str, optional
        which lfp band to analyze, by default "Theta 5-11 Hz"
    lfp_trace_window : tuple, optional
        time around the first pulse
    pulse_number_list : np.ndarray, optional
        which pulses to include in analysis (pulse 0 is first pulse in a cycle)
    fig : matplotlib.figure.Figure, optional
        figure to plot on, by default None which creates new figure
    color : str, optional
        color to plot this dataset, by default "cornflowerblue"
    return_data : bool, optional
        whether to return the lfp traces amplitudes and phases, by default False

    Returns
    -------
    matplotlib.figure.Figure
        figure of power spectrum and statistics
    Tuple
        distributions of entrainment statistics (optional)
    """
    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)

    nwb_file_name_list = dataset.fetch("nwb_file_name")
    interval_list_name_list = dataset.fetch("interval_list_name")
    power_curves = []

    for nwb_file_name, interval_list_name in zip(
        nwb_file_name_list, interval_list_name_list
    ):
        basic_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": interval_list_name,
        }
        print(basic_key)
        stim_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "dio_event_name": "stim",
        }
        if len(LFPBandV1 & basic_key) == 0:
            continue
        # get analytic band power
        ref_elect_index, basic_key = get_ref_electrode_index(basic_key)
        power_df = (
            LFPBandV1 & basic_key & {"filter_name": band_filter_name}
        ).compute_signal_power([ref_elect_index])
        power_ = np.asarray(power_df[power_df.columns[0]])
        power_timestamps = power_df.index
        power_sampling_rate = int(np.round(1 / np.mean(np.diff(power_timestamps))))
        t0_list = OptoStimProtocol().get_cylcle_begin_timepoints(stim_key)
        for t0 in t0_list:
            ind = np.digitize(t0, power_timestamps)
            if ind + lfp_trace_window[0] < 0 or ind + lfp_trace_window[1] >= len(
                power_
            ):
                continue
            dat = (
                power_[ind + lfp_trace_window[0] : ind + lfp_trace_window[1]]
                # / power_[ind]
            )
            dat = dat / dat.max()
            power_curves.append(dat)
    if len(power_curves) == 0:
        return fig
    tp = np.arange(lfp_trace_window[0], lfp_trace_window[1]) / power_sampling_rate
    print(power_sampling_rate)
    # plot
    if fig is None:
        fig = plt.figure()
    power_curves = np.array(power_curves)
    if norm_window is not None:
        ind_norm = np.where((tp > norm_window[0]) & (tp < norm_window[1]))[0]
        power_curves = (
            power_curves / np.nanmean(power_curves[:, ind_norm], axis=1)[:, None]
        )

    print(power_curves.shape)
    plt.plot(tp, np.median(power_curves, axis=0), color=color)
    plt.fill_between(
        tp,
        np.percentile(power_curves, 25, axis=0),
        np.percentile(power_curves, 75, axis=0),
        facecolor=color,
        alpha=0.2,
    )
    fig.gca().spines[["right", "top"]].set_visible(False)
    fig.gca().set_xlabel("time (s)")
    fig.gca().set_ylabel("normalized power")
    fig.gca().set_xlim(tp[0], tp[-1])
    return fig


def lfp_power_dynamics_pulse_cwt(
    dataset_key: dict,
    filter_name: str = "LFP 0-400 Hz",
    # band_filter_name: str = "Theta 5-11 Hz",
    lfp_trace_window=(-int(0.125 * 1000), int(1000)),
    fig=None,
    color="cornflowerblue",
    frequencies=np.arange(5, 11, 0.5),
    wavelet="morl",
    return_data=False,
):
    """Generates a figure characterizing the lfp around each stimulus cycle

    Parameters
    ----------
    dataset_key : dict
        key containing parameters which define the dataset to analyze
    filter_name : str, optional
        which lfp filter to analyze, by default "LFP 0-400 Hz"
    band_filter_name : str, optional
        which lfp band to analyze, by default "Theta 5-11 Hz"
    lfp_trace_window : tuple, optional
        time around the first pulse
    pulse_number_list : np.ndarray, optional
        which pulses to include in analysis (pulse 0 is first pulse in a cycle)
    fig : matplotlib.figure.Figure, optional
        figure to plot on, by default None which creates new figure
    color : str, optional
        color to plot this dataset, by default "cornflowerblue"
    return_data : bool, optional
        whether to return the lfp traces amplitudes and phases, by default False

    Returns
    -------
    matplotlib.figure.Figure
        figure of power spectrum and statistics
    Tuple
        distributions of entrainment statistics (optional)
    """
    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)

    nwb_file_name_list = dataset.fetch("nwb_file_name")
    interval_list_name_list = dataset.fetch("interval_list_name")
    power_curves = []

    for nwb_file_name, interval_list_name in zip(
        nwb_file_name_list, interval_list_name_list
    ):
        basic_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": interval_list_name,
        }
        print(basic_key)
        stim_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "dio_event_name": "stim",
        }
        if len(LFPBandV1 & basic_key) == 0:
            continue

        # get lfp data
        ref_elect, basic_key = get_ref_electrode_index(basic_key)  #
        # ref_elect = (Electrode() & basic_key).fetch("original_reference_electrode")[0]
        lfp_eseries = (LFPOutput).fetch_nwb(restriction=basic_key)[0]["lfp"]
        ref_index = get_electrode_indices(lfp_eseries, [ref_elect])

        # get LFP series
        lfp_df = (LFPV1() & basic_key).fetch1_dataframe()
        lfp_timestamps = lfp_df.index
        lfp_ = np.array(lfp_df[ref_index])
        fs = (LFPV1() & basic_key).fetch1("lfp_sampling_rate")
        padding = int(1 / np.min(frequencies) * fs)

        # Define needed settings for cwt
        scale = pywt.frequency2scale(wavelet, frequencies / fs)

        # get analytic band power
        t0_list = OptoStimProtocol().get_cylcle_begin_timepoints(stim_key)
        for t0 in tqdm(t0_list):
            ind = np.digitize(t0, lfp_timestamps)
            # skip if can't fill window
            if ind + lfp_trace_window[0] - padding < 0 or ind + lfp_trace_window[
                1
            ] + padding >= len(lfp_):
                continue
            # skip the high amplitude noise
            if (
                np.max(
                    lfp_[
                        ind
                        + lfp_trace_window[0]
                        - padding : ind
                        + lfp_trace_window[1]
                        + padding
                    ]
                )
                > LFP_AMP_CUTOFF
            ):
                continue
            print(lfp_.shape)
            C, freq = pywt.cwt(
                lfp_[
                    ind
                    + lfp_trace_window[0]
                    - padding : ind
                    + lfp_trace_window[1]
                    + padding
                ],
                scale,
                sampling_period=1 / fs,
                wavelet=wavelet,
                axis=1,
            )
            # print((~np.isnan(C)).all())
            # print(np.max(C))
            dat = np.nansum(np.abs(C[:, padding:-padding]) ** 2, axis=0)[
                :, 0
            ]  # add up the power across scales
            # smooth with a boxcar filter
            if "period_ms" in dataset_key:
                filter_size = dataset_key["period_ms"]
            else:
                filter_size = 125
            dat = np.convolve(dat, np.ones(filter_size) / filter_size, mode="same")

            # print(np.max(dat))
            dat = dat / np.nanmax(dat)
            # print((~np.isnan(dat)).all())
            power_curves.append(dat)
            if len(power_curves) > 10000:
                break
    if len(power_curves) == 0:
        return fig
    tp = np.arange(lfp_trace_window[0], lfp_trace_window[1]) / fs
    # plot
    if fig is None:
        fig = plt.figure()
    power_curves = np.array(power_curves)
    print(power_curves.shape)
    plt.plot(tp, np.median(power_curves, axis=0), color=color)
    plt.fill_between(
        tp,
        np.percentile(power_curves, 25, axis=0),
        np.percentile(power_curves, 75, axis=0),
        facecolor=color,
        alpha=0.2,
    )
    fig.gca().spines[["right", "top"]].set_visible(False)
    fig.gca().set_xlabel("time (s)")
    fig.gca().set_ylabel("normalized power")
    fig.gca().set_xlim(tp[0], tp[-1])
    return fig  # , power_curves


def lfp_power_dynamics_pulse_cwt_spectrogram(
    dataset_key: dict,
    filter_name: str = "LFP 0-400 Hz",
    # band_filter_name: str = "Theta 5-11 Hz",
    lfp_trace_window=(-int(0.125 * 1000), int(1000)),
    fig=None,
    color="cornflowerblue",
    frequencies=np.arange(1, 20, 1),
    wavelet="morl",
    return_data=False,
    marks: str = "first_pulse",
):
    """Generates a figure characterizing the lfp around each stimulus cycle

    Parameters
    ----------
    dataset_key : dict
        key containing parameters which define the dataset to analyze
    filter_name : str, optional
        which lfp filter to analyze, by default "LFP 0-400 Hz"
    band_filter_name : str, optional
        which lfp band to analyze, by default "Theta 5-11 Hz"
    lfp_trace_window : tuple, optional
        time around the first pulse
    pulse_number_list : np.ndarray, optional
        which pulses to include in analysis (pulse 0 is first pulse in a cycle)
    fig : matplotlib.figure.Figure, optional
        figure to plot on, by default None which creates new figure
    color : str, optional
        color to plot this dataset, by default "cornflowerblue"
    return_data : bool, optional
        whether to return the lfp traces amplitudes and phases, by default False
    marks : str, optional
        How to define t0 allignment. Options are "first_pulse" (default, allignment to the first pulse in the cycle),
        "position_test" (allignment to exit of the reward port during the opto test interval),
        "position_control" (allignment to exit of the reward port during the opto control interval)

    Returns
    -------
    matplotlib.figure.Figure
        figure of power spectrum and statistics
    Tuple
        distributions of entrainment statistics (optional)
    """
    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)

    nwb_file_name_list = dataset.fetch("nwb_file_name")
    interval_list_name_list = dataset.fetch("interval_list_name")
    spectrograms = []

    for nwb_file_name, interval_list_name in zip(
        nwb_file_name_list, interval_list_name_list
    ):
        basic_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": interval_list_name,
        }
        print(basic_key)
        stim_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "dio_event_name": "stim",
        }
        # if len(LFPBandV1 & basic_key) == 0:
        #     continue

        # get lfp data
        ref_elect, basic_key = get_ref_electrode_index(basic_key)  #
        # ref_elect = (Electrode() & basic_key).fetch("original_reference_electrode")[0]
        lfp_eseries = (LFPOutput & basic_key).fetch_nwb(restriction=basic_key)[0]["lfp"]
        ref_index = get_electrode_indices(lfp_eseries, [ref_elect])

        # get LFP series
        lfp_df = (LFPV1() & basic_key).fetch1_dataframe()
        lfp_timestamps = lfp_df.index
        lfp_ = np.array(lfp_df[ref_index]).astype(float)
        assert all(np.isfinite(lfp_))
        # nan out segments with large noise
        # artifacts = (LFPArtifactDetection() & basic_key).fetch1("artifact_times")
        # for artifact in artifacts:
        #     lfp_[
        #         np.logical_and(
        #             lfp_timestamps > artifact[0], lfp_timestamps < artifact[1]
        #         )
        #     ] = np.nan

        fs = (LFPV1() & basic_key).fetch1("lfp_sampling_rate")
        padding = int(1 / np.min(frequencies) * fs)

        # Define needed settings for cwt
        scale = pywt.frequency2scale(wavelet, frequencies / fs)

        # get analytic band power
        if marks == "first_pulse":
            t0_list = OptoStimProtocol().get_cylcle_begin_timepoints(stim_key)
        elif marks in [
            "position_test",
            "position_control",
            "first_pulse_position_restricted",
        ]:
            # get t0 from filtered port interval
            valid_intervals = np.array(filter_position_ports(stim_key, buffer=20))
            if marks in ["position_test", "first_pulse_position_restricted"]:
                restrict_intervals = (OptoStimProtocol & stim_key).fetch1(
                    "test_intervals"
                )
            else:
                restrict_intervals = (OptoStimProtocol & stim_key).fetch1(
                    "control_intervals"
                )
            valid_intervals = interval_list_intersect(
                valid_intervals, np.array(restrict_intervals)
            )
            valid_intervals = [
                interval
                for interval in valid_intervals
                if interval[1] - interval[0] > 0.5
            ]
            t0_list = [interval[0] for interval in valid_intervals]
        else:
            raise ValueError(
                "marks must be 'first_pulse', 'position_test' or 'position_control'"
            )

        if marks == "first_pulse_position_restricted":
            stim_list = OptoStimProtocol().get_cylcle_begin_timepoints(stim_key)
            t0_list = [
                s for s in stim_list if np.min(np.abs(s - np.array(t0_list))) < 0.3
            ]

        for t0 in tqdm(t0_list):
            ind = np.digitize(t0, lfp_timestamps)
            # skip if can't fill window
            if ind + lfp_trace_window[0] - padding < 0 or ind + lfp_trace_window[
                1
            ] + padding >= len(lfp_):
                continue
            # skip the high amplitude noise
            if (
                # np.max(
                #     lfp_[
                #         ind
                #         + lfp_trace_window[0]
                #         - padding : ind
                #         + lfp_trace_window[1]
                #         + padding
                #     ]
                # )
                np.percentile(
                    lfp_[
                        ind
                        + lfp_trace_window[0]
                        - padding : ind
                        + lfp_trace_window[1]
                        + padding
                    ],
                    99,
                )
                > LFP_AMP_CUTOFF
            ):
                continue
            C, freq = pywt.cwt(
                lfp_[
                    ind
                    + lfp_trace_window[0]
                    - padding : ind
                    + lfp_trace_window[1]
                    + padding,
                    0,
                ],
                scale,
                sampling_period=1 / fs,
                wavelet=wavelet,
                axis=1,
                method="fft",
            )
            for i, f in enumerate(frequencies):
                width = fs / f  # / 2
                C[i, :] = np.convolve(
                    np.abs(C[i, :]), np.ones(int(width)) / width, mode="same"
                )

            spectrograms.append(C[:, padding:-padding])

    if len(spectrograms) == 0:
        if return_data:
            return fig, [], []
        return fig
    tp = np.arange(lfp_trace_window[0], lfp_trace_window[1]) / fs
    # plot
    if fig is None:
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    spectrograms = np.array(spectrograms)
    avg_spectrogram = np.nanmean(np.abs(spectrograms), axis=0)
    ax[0].matshow(
        avg_spectrogram,
        cmap="plasma",
        extent=[tp[0], tp[-1], freq[0], freq[-1]],
        aspect="auto",
        origin="lower",
    )
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("frequency (Hz)")
    ax[0].vlines(0, freq[0], freq[-1], color="white", ls="--")
    if "period_ms" in dataset_key:
        drive_freq = 1000 / dataset_key["period_ms"]
        ax[0].hlines(drive_freq, 0, tp[-1], color="white", ls="--")
        ax[0].hlines(drive_freq, 0, tp[-1], color="thistle", ls="--")

    track_freq = [
        (6, 10),
        (14, 18),
        (20, 24),
    ]
    if "period_ms" in dataset_key:
        central = 1000.0 / dataset_key["period_ms"]
        track_freq.append((central - 0.5, central + 0.5))

    return_traces = []
    for i, f in enumerate(track_freq):
        ind_freq = np.where((freq >= f[0]) & (freq <= f[1]))[0]
        c = plt.cm.Set1(i / len(track_freq))

        power_traces = np.nanmean(np.abs(spectrograms)[:, ind_freq, :], axis=1)
        ax[1].plot(
            tp,
            np.nanmedian(power_traces, axis=0),
            label=f"{np.round(f[0],2)}-{np.round(f[1],2)} Hz",
            color=c,
        )
        ax[1].fill_between(
            tp,
            np.nanpercentile(power_traces, 25, axis=0),
            np.nanpercentile(power_traces, 75, axis=0),
            alpha=0.1,
            facecolor=c,
        )
        return_traces.append(power_traces)

    ax[1].legend()
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("Frequency power")

    peak_freq = freq[np.nanargmax(avg_spectrogram, axis=0)]
    ax[0].plot(tp, peak_freq, color="limegreen", ls="dashdot", lw=2.5)

    if return_data:
        return fig, spectrograms, return_traces
    return fig
