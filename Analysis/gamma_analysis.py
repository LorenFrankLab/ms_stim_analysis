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
import scipy.stats

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
from spyglass.common.common_interval import (
    interval_list_intersect,
    interval_list_contains,
)
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
from Time_and_trials.ms_interval import EpochIntervalListName

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
    get_running_valid_intervals,
    bootstrap,
)

# from Analysis.circular_shuffle import (
#     normalize_by_index_wrapper,
#     normalize_by_peak,
#     shuffled_trace_distribution,
#     bootstrap,
#     trace_median,
# )

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
    # Use the driving period lfp band if band filter not specified
    if phase_filter_name is None:
        if "period_ms" not in dataset_key:
            raise ValueError("band_filter_name must be specified if period_ms is not")
        phase_filter_name = f"ms_stim_{dataset_key['period_ms']}ms_period"

    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)

    nwb_file_name_list = dataset.fetch("nwb_file_name")
    interval_list_name_list = dataset.fetch("interval_list_name")
    power = []
    phase = []

    if len(set((dataset * Session).fetch("subject_id"))) > 1:
        raise NotImplementedError("Only one subject allowed")

    # make the figure
    fig, ax_all = plt.subplots(
        2, 4, figsize=(20, 6), gridspec_kw={"width_ratios": [3, 3, 3, 1]}, sharex="col"
    )
    for power_filter, ax in zip(
        ["Slow Gamma 25-55 Hz", "Fast Gamma 65-100 Hz"], ax_all
    ):
        power_opto = []
        power_control = []
        phase_opto = []
        phase_control = []

        for nwb_file_name, interval_list_name in zip(
            nwb_file_name_list, interval_list_name_list
        ):
            basic_key = {
                "nwb_file_name": nwb_file_name,
                "target_interval_list_name": interval_list_name,
            }
            print(basic_key)

            # define the key for the band used to define phase
            phase_key = {
                **basic_key,
                "filter_name": phase_filter_name,
                "filter_sampling_rate": 1000,
            }
            # define the key for the band used to define amplitude
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

            # get test and control run intervals
            pos_key = {
                **basic_key,
                "interval_list_name": basic_key["target_interval_list_name"],
            }
            opto_run_intervals, control_run_intervals = get_running_valid_intervals(
                pos_key, filter_speed=filter_speed, seperate_optogenetics=True
            )

            for intervals, power, phase in zip(
                [opto_run_intervals, control_run_intervals],
                [power_opto, power_control],
                [phase_opto, phase_control],
            ):
                valid_times = interval_list_contains(intervals, phase_timestamps)
                ind_power = np.digitize(valid_times, power_timestamps)
                power.extend(power_[ind_power - 1])
                ind_phase = np.digitize(valid_times, phase_timestamps)
                phase.extend(phase_[ind_phase - 1])

        if len(phase_opto) == 0 or len(phase_control) == 0:
            return None, None, None, None

        for a, phase, power, color, name in zip(
            ax[:2],
            [phase_opto, phase_control],
            [power_opto, power_control],
            ["firebrick", "cornflowerblue"],
            ["opto", "control"],
        ):
            H, xedges, yedges = np.histogram2d(
                phase,
                np.log10(power),
                bins=[np.linspace(0, 2 * np.pi, 100), np.linspace(0, 5, 30)],
            )
            # H, xedges, yedges = np.histogram2d(
            #     phase,
            #     power,
            #     bins=[np.linspace(0, 2 * np.pi, 100), np.linspace(0, 1e5, 100)],
            # )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            H = H / H.sum(axis=1)[:, None]
            a.imshow(
                H.T,
                origin="lower",
                extent=extent,
                cmap=plt.cm.plasma,
                interpolation="nearest",
                aspect=float(np.diff(extent)[0]) / np.diff(extent)[-1],
            )
            a.set_xlabel("Phase")
            a.set_ylabel("log10 Power")
            a.set_title(name)

            # val, bins, ind = scipy.stats.binned_statistic(phase, np.log10(power), bins=100)
            # # val, bins, ind = scipy.stats.binned_statistic(phase, power, bins=100)
            # bin_centers = bins[:-1] + np.diff(bins) / 2

            bins = np.linspace(0, 2 * np.pi, 65)
            labels = np.digitize(phase, bins)
            val, rng_lo, rng_hi = bootstrap_binned(labels, np.log10(power), n_boot=1000)
            bin_centers = bins[:-1] + np.diff(bins) / 2

            ax[2].plot(bin_centers, val, color=color, label=name)
            ax[2].fill_between(bin_centers, rng_lo, rng_hi, facecolor=color, alpha=0.3)
            ax[2].set_xlabel("Phase")
            ax[2].set_ylabel(f"log10 Power {power_filter}")

        ax[2].spines[["top", "right"]].set_visible(False)
        ax[2].set_xlim([0, 2 * np.pi])
        ax[2].legend()

    # Table with information about the dataset
    the_table = ax[3].table(
        cellText=[[len(dataset)], [phase_filter_name]]
        + [[str(x)] for x in dataset_key.values()],
        rowLabels=["number_epochs", "phase_filter_name"]
        + [str(x) for x in dataset_key.keys()],
        loc="right",
        colWidths=[1, 1],
    )

    for a in ax_all[:, 3]:
        a.spines[["top", "right", "left", "bottom"]].set_visible(False)
        a.set_xticks([])
        a.set_yticks([])

    fig.suptitle(f"{dataset_key['animal']}: {dataset_key['period_ms']}ms period")
    return phase_opto, phase_control, power_opto, power_control


def bootstrap_binned(labels, values, n_boot=1000):
    unique_labels = np.unique(labels)
    bootstrap_dist = []

    for label in unique_labels:
        ind = labels == label
        samples = values[ind]
        boot_samples = np.random.choice(samples, (n_boot, len(samples)))
        boot_means = np.nanmean(boot_samples, axis=1)
        bootstrap_dist.append(boot_means)

    bootstrap_dist = np.array(bootstrap_dist)
    return (
        np.mean(bootstrap_dist, axis=1),
        np.percentile(bootstrap_dist, 0.5, axis=1),
        np.percentile(bootstrap_dist, 99.5, axis=1),
    )


# def bootstrap_binned(labels, values, n_boot=1000):
#     bootstrap_dist = []
#     for label in np.unique(labels):
#         bootstrap_dist.append([])
#         ind = labels == label
#         for _ in range(n_boot):
#             bootstrap_dist[-1].append(
#                 np.nanmean(np.random.choice(values[ind], len(values[ind])))
#             )
#     bootstrap_dist = np.array(bootstrap_dist)
#     return (
#         np.mean(bootstrap_dist, axis=1),
#         np.percentile(bootstrap_dist, 0.5, axis=1),
#         np.percentile(bootstrap_dist, 99.5, axis=1),
#     )


# def gamma_stim_nesting(
#     dataset_key: dict,
#     phase_filter_name: str = "Theta 5-11 Hz",
#     filter_speed: float = 10.0,
#     window: float = 1.0,
#     return_distributions: bool = False,
# ):
#     # Define the dataset (epochs included in this analyusis)
#     dataset = filter_opto_data(dataset_key)

#     nwb_file_name_list = dataset.fetch("nwb_file_name")
#     interval_list_name_list = dataset.fetch("interval_list_name")
#     power = []
#     phase = []

#     if len(set((dataset * Session).fetch("subject_id"))) > 1:
#         raise NotImplementedError("Only one subject allowed")

#     power_opto = []
#     power_control = []
#     phase_opto = []
#     phase_control = []

#     for nwb_file_name, interval_list_name in zip(
#         nwb_file_name_list, interval_list_name_list
#     ):
#         basic_key = {
#             "nwb_file_name": nwb_file_name,
#             "target_interval_list_name": interval_list_name,
#         }
#         print(basic_key)

#         # define the key for the band used to define phase
#         phase_key = {**basic_key, "filter_name": phase_filter_name}
#         # define the key for the band used to define amplitude
#         power_filter = "Slow Gamma 25-55 Hz"
#         # power_filter = "Fast Gamma 65-100 Hz"
#         power_key = {**basic_key, "filter_name": power_filter}
#         print(power_key)
#         print(phase_key)

#         # get analytic band power
#         ref_elect_index, basic_key = get_ref_electrode_index(basic_key)
#         power_df = (LFPBandV1 & power_key).compute_signal_power([ref_elect_index])
#         power_ = np.asarray(power_df[power_df.columns[0]])
#         power_timestamps = power_df.index

#         # get stimulus phase
#         # if not (LFPBandV1 & phase_key) or not (LFPBandV1 & power_key):
#         #     continue
#         # phase_df = (LFPBandV1 & phase_key).compute_signal_phase([ref_elect_index])
#         # phase_timestamps = phase_df.index
#         # phase_ = np.asarray(phase_df)[:, 0]
#         pos_key = {
#             **basic_key,
#             "interval_list_name": basic_key["target_interval_list_name"],
#         }
#         stim, time = (OptoStimProtocol() & pos_key).get_stimulus(pos_key)
#         pulse_timestamps = time[stim == 1]
#         phase_ = np.array(
#             [
#                 t - pulse_timestamps[np.argmin(np.abs(pulse_timestamps - t))]
#                 for t in power_timestamps
#             ]
#         )
#         phase_timestamps = power_timestamps

#         # get test and control run intervals
#         opto_run_intervals, control_run_intervals = get_running_valid_intervals(
#             pos_key, filter_speed=filter_speed, seperate_optogenetics=True
#         )

#         for intervals, power, phase in zip(
#             [opto_run_intervals, control_run_intervals],
#             [power_opto, power_control],
#             [phase_opto, phase_control],
#         ):
#             valid_times = interval_list_contains(intervals, phase_timestamps)
#             ind_power = np.digitize(valid_times, power_timestamps)
#             power.extend(power_[ind_power - 1])
#             ind_phase = np.digitize(valid_times, phase_timestamps)
#             phase.extend(phase_[ind_power - 1])

#     ind = np.logical_and(
#         np.array(phase_opto) >= -dataset_key["period_ms"],
#         np.array(phase_opto) <= dataset_key["period_ms"],
#     )
#     phase_opto = np.array(phase_opto)[ind]
#     power_opto = np.array(power_opto)[ind]

#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))

#     if len(phase_opto) == 0 or len(phase_control) == 0:
#         return None, None

#     for a, phase, power in zip(
#         ax, [phase_opto, phase_control], [power_opto, power_control]
#     ):
#         H, xedges, yedges = np.histogram2d(
#             phase,
#             power,
#             bins=[
#                 np.linspace(-dataset_key["period_ms"], dataset_key["period_ms"], 100),
#                 np.linspace(0, 1000, 100),
#             ],
#         )
#         extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#         H = H / H.sum(axis=1)
#         a.imshow(
#             H.T,
#             origin="lower",
#             # extent=extent,
#             cmap=plt.cm.plasma,
#             interpolation="nearest",
#             aspect=1,
#         )
#         a.set_xlabel("Phase")
#         a.set_ylabel("Power")
#         a.set_title("Density Heatmap")
#     fig.suptitle(f"{dataset_key['animal']}: {dataset_key['period_ms']}ms period")
#     return phase_opto, phase_control, power_opto, power_control
