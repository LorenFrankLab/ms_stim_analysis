from typing import Tuple
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
import os
from scipy import signal
from tqdm import tqdm
import pywt

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
from spyglass.common.common_interval import interval_list_intersect
from spyglass.lfp.v1 import LFPElectrodeGroup, LFPSelection, LFPV1, LFPArtifactDetection
from spyglass.position.v1 import TrodesPosV1
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.lfp.analysis.v1 import LFPBandV1

from .position_analysis import get_running_intervals, filter_position_ports
from .utils import convert_delta_marks_to_timestamp_values

import sys

sys.path.append("/home/sambray/Documents/MS_analysis_samsplaying/")
os.chdir("/home/sambray/Documents/MS_analysis_samsplaying/")
from ms_opto_stim_protocol import (
    OptoStimProtocol,
    OptoStimProtocolLaser,
    OptoStimProtocolTransfected,
    OptoStimProtocolClosedLoop,
)

from Analysis.utils import (
    filter_animal,
    weighted_quantile,
    filter_task,
    convert_delta_marks_to_timestamp_values,
    filter_opto_data,
)
from Analysis.lfp_analysis import get_ref_electrode_index
from Style.style_guide import animal_style, transfection_style

LFP_AMP_CUTOFF = 2000


def individual_lfp_traces(
    dataset_key: dict,
    filter_name: str = "LFP 0-400 Hz",
    # band_filter_name: str = "Theta 5-11 Hz",
    lfp_trace_window=(-int(0.125 * 1000), int(1 * 1000)),
    fig=None,
    color="cornflowerblue",
    n_plot=10,
):
    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)
    nwb_file_name_list = dataset.fetch("nwb_file_name")
    interval_list_name_list = dataset.fetch("interval_list_name")
    # get the color for display
    if "animal" in dataset_key:
        color = animal_style.loc[dataset_key["animal"]]["color"]
    elif "transfected" in dataset_key:
        if dataset_key["transfected"]:
            color = transfection_style["transfected"]
        else:
            color = transfection_style["control"]
    # get the lfp traces for every relevant pulse
    lfp_traces = []
    marks = []
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
        if len(LFPV1() & basic_key) == 0:
            print("missing LFP for: ", basic_key)
            continue
        print(basic_key)
        # get lfp band phase for reference electrode
        ref_elect, basic_key = get_ref_electrode_index(basic_key)  #
        # ref_elect = (Electrode() & basic_key).fetch("original_reference_electrode")[0]
        lfp_eseries = LFPOutput().fetch_nwb(restriction=basic_key)[0]["lfp"]
        ref_index = get_electrode_indices(lfp_eseries, [ref_elect])

        # get LFP series
        lfp_df = (LFPV1() & basic_key).fetch_nwb()[0]["lfp"]
        lfp_df = (LFPV1() & basic_key).fetch1_dataframe()
        lfp_timestamps = lfp_df.index
        lfp_ = np.array(lfp_df[ref_index])

        ind = np.sort(np.unique(lfp_timestamps, return_index=True)[1])
        lfp_timestamps = lfp_timestamps[ind]
        lfp_ = lfp_[ind]
        # nan out artifact intervals
        artifact_times = (LFPArtifactDetection() & basic_key).fetch1("artifact_times")
        for artifact in artifact_times:
            lfp_[
                np.logical_and(
                    lfp_timestamps > artifact[0], lfp_timestamps < artifact[1]
                )
            ] = np.nan

        try:
            assert np.all(np.diff(lfp_timestamps) > 0)
        except:
            continue
        # get stim times
        t_mark_cycle = OptoStimProtocol().get_cylcle_begin_timepoints(stim_key)
        ind = np.digitize(t_mark_cycle, lfp_timestamps)
        stim, t_mark = OptoStimProtocol().get_stimulus(stim_key)
        t_mark = t_mark[stim == 1]
        ind_mark = np.digitize(t_mark, lfp_timestamps)

        for i in ind:
            lfp_traces.append(lfp_[i + lfp_trace_window[0] : i + lfp_trace_window[1]])
            marks.append(
                ind_mark[
                    (ind_mark >= i + lfp_trace_window[0])
                    & (ind_mark < i + lfp_trace_window[1])
                ]
                - (i + lfp_trace_window[0])
            )

        if len(lfp_traces) > 100:
            break

    fig, ax = plt.subplots(nrows=n_plot, figsize=(10, n_plot), sharex=True, sharey=True)
    tp = np.linspace(lfp_trace_window[0], lfp_trace_window[1], lfp_traces[0].shape[0])
    for a in ax:
        i = np.random.randint(len(lfp_traces))
        a.plot(tp, lfp_traces[i], color=color)
        a.spines[["top", "right", "bottom"]].set_visible(False)
        # if "period_ms" in dataset_key:
        #     loc = 0
        #     while loc < tp[-1]:
        #         a.axvline(loc, color="thistle", linestyle="--")
        #         loc += dataset_key["period_ms"]
        for m in marks[i]:
            a.axvline(tp[m], color="thistle", linestyle="--")

    # for a in ax[:-2]:
    #     a.set_xticks([])
    return fig
