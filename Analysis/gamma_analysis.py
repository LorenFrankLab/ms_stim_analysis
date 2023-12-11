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
from spyglass.lfp.v1 import (
    LFPElectrodeGroup,
    LFPSelection,
    LFPV1,
)
from spyglass.position.v1 import TrodesPosV1
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.lfp.analysis.v1 import LFPBandV1

from .position_analysis import get_running_intervals, filter_position_ports
from .utils import convert_delta_marks_to_timestamp_values

os.chdir("/home/sambray/Documents/MS_analysis_Jen/")
from ms_interval import EpochIntervalListName

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
from Analysis.circular_shuffle import (
    normalize_by_index_wrapper,
    normalize_by_peak,
    shuffled_trace_distribution,
    bootstrap,
    trace_median,
)

from Analysis.lfp_analysis import get_ref_electrode_index


LFP_AMP_CUTOFF = 2000

################################################################################


def gamma_theta_nesting(
    dataset_key: dict,
    phase_filter_name: str = "Theta 5-11 Hz",
    filter_speed: float = 10.0,
    window: float = 1.0,
    return_distributions: bool = False,
):
    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)

    nwb_file_name_list = dataset.fetch("nwb_file_name")
    interval_list_name_list = dataset.fetch("interval_list_name")
    power = []
    phase = []

    if len(set((dataset * Session).fetch("subject_id"))) > 1:
        raise NotImplementedError("Only one subject allowed")

    for nwb_file_name, interval_list_name in zip(
        nwb_file_name_list, interval_list_name_list
    ):
        basic_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": interval_list_name,
        }
        print(basic_key)

        # define the key for the band used to define phase
        phase_key = {**basic_key, "filter_name": phase_filter_name}
        # define the key for the band used to define amplitude
        power_filter = "Slow Gamma 25-55 Hz"
        power_filter = "Fast Gamma 65-100 Hz"
        power_key = {**basic_key, "filter_name": power_filter}
        print(power_key)
        print(phase_key)

        # get analytic band power
        ref_elect_index, basic_key = get_ref_electrode_index(basic_key)
        power_df = (LFPBandV1 & power_key).compute_signal_power([ref_elect_index])
        power_ = np.asarray(power_df[power_df.columns[0]])
        power_timestamps = power_df.index

        # get phase
        if not (LFPBandV1 & phase_key) or not (LFPBandV1 & power_key):
            continue
        phase_df = (LFPBandV1 & phase_key).compute_signal_phase([ref_elect_index])
        phase_timestamps = phase_df.index
        phase_ = np.asarray(phase_df)[:, 0]

        # append to the list of samples
        # ind = np.digitize(power_timestamps, phase_timestamps)
        # print(power_[np.logical_and(ind > 0, ind < len(phase_timestamps))].size)
        # print(
        #     phase_[ind[np.logical_and(ind > 0, ind < len(phase_timestamps))] - 1].size
        # )
        # power.extend(power_[np.logical_and(ind > 0, ind < len(phase_timestamps))])
        # phase.extend(
        #     phase_[ind[np.logical_and(ind > 0, ind < len(phase_timestamps))] - 1]
        # )
        ind = np.digitize(phase_timestamps, power_timestamps)
        # print(power_[np.logical_and(ind > 0, ind < len(phase_timestamps))].size)
        # print(
        #     phase_[ind[np.logical_and(ind > 0, ind < len(phase_timestamps))] - 1].size
        # )
        power.extend(
            power_[ind[np.logical_and(ind > 0, ind < len(phase_timestamps))] - 1]
        )
        phase.extend(phase_[np.logical_and(ind > 0, ind < len(phase_timestamps))])

    if len(phase) == 0:
        return None, None
    H, xedges, yedges = np.histogram2d(
        phase, power, bins=[np.linspace(0, 2 * np.pi, 100), np.linspace(0, 1000, 100)]
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.figure(figsize=(8, 6))
    plt.imshow(
        H.T,
        origin="lower",
        # extent=extent,
        cmap=plt.cm.plasma,
        interpolation="nearest",
        aspect=1,
    )
    plt.colorbar(label="Density")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Density Heatmap")
    plt.show()
