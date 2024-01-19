from typing import Tuple
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib as mpl
import os
from scipy import signal
from tqdm import tqdm

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
    convert_epoch_interval_name_to_position_interval_name,
    PositionIntervalMap,
    interval_list_contains,
)
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.position.v1 import TrodesPosV1
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
    generate_alligned_binned_spike_func,
    shuffled_spiking_distribution,
    discrete_KL_divergence,
    bootstrap,
    stacked_marks_to_kl,
)


from spiking_analysis_tables import BinnedSpiking
from spyglass.spikesorting import CuratedSpikeSorting
import scipy.signal


##################################################################################3
def opto_spiking_dynamics(
    dataset_key: dict,
    plot_rng: np.ndarray = np.arange(-0.08, 0.2, 0.002),
    # first_pulse_only: bool = False,
    marks="first_pulse",
    return_data: bool = False,
):
    # get the filtered data
    dataset = filter_opto_data(dataset_key)

    # compile the data
    spike_counts = []
    spike_counts_shuffled = []
    for nwb_file_name, position_interval_name in tqdm(
        zip(dataset.fetch("nwb_file_name")[:3], dataset.fetch("interval_list_name")[:3])
    ):
        interval_name = (
            (
                PositionIntervalMap()
                & {
                    "nwb_file_name": nwb_file_name,
                    "position_interval_name": position_interval_name,
                }
            )
            * TaskEpoch()
        ).fetch1("interval_list_name")
        basic_key = {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": interval_name,
        }
        print(basic_key)
        # get BinnedSpiking keys for this interval
        key_list = (BinnedSpiking & basic_key).get_current_curation_key_list()
        # check if spiking data exists
        if len(key_list) == 0:
            print("no spiking data for", basic_key)
            continue
        pos_interval_name = convert_epoch_interval_name_to_position_interval_name(
            {"nwb_file_name": nwb_file_name, "interval_list_name": interval_name}
        )
        opto_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": pos_interval_name,
        }
        if marks == "first_pulse":
            pulse_timepoints = (
                OptoStimProtocol() & opto_key
            ).get_cylcle_begin_timepoints(opto_key)
        elif marks == "all_pulses":
            stim, time = (OptoStimProtocol() & opto_key).get_stimulus(opto_key)
            pulse_timepoints = time[stim == 1]
        elif marks == "odd_pulses":
            # get times of firt pulse in cylce
            t_mark_cycle = OptoStimProtocol().get_cylcle_begin_timepoints(opto_key)
            # get times of all stimulus
            stim, t_mark = OptoStimProtocol().get_stimulus(opto_key)
            t_mark = t_mark[stim == 1]
            # label each pulse as its count in the cycle
            pulse_count = np.zeros_like(t_mark)
            mark_ind_cycle = [np.where(t_mark == t_)[0][0] for t_ in t_mark_cycle]
            pulse_count[mark_ind_cycle] = 1
            count = 1
            for i in range(pulse_count.size):
                if pulse_count[i] == 1:
                    count = 1
                pulse_count[i] = count
                count += 1
            pulse_count = pulse_count - 1  # 0 index the count
            pulse_timepoints = t_mark[pulse_count % 2 == 1]

        elif marks == "theta_peaks":
            # get all running theta peaks
            band_key = {
                "nwb_file_name": nwb_file_name,
                "target_interval_list_name": interval_name,
            }
            pulse_timepoints = get_theta_peaks(band_key)
            # subset to peaks within 1second of a cycle start
            cycle_start = (OptoStimProtocol() & opto_key).get_cylcle_begin_timepoints(
                opto_key
            )
            cycle_start_intervals = []
            for t in cycle_start:
                cycle_start_intervals.append([t, t + 1])
            pulse_timepoints = interval_list_contains(
                cycle_start_intervals, pulse_timepoints
            )
        elif "dummy_cycle" in marks:
            dummy_freq = int(marks.split("=")[-1])
            cycle_start = (OptoStimProtocol() & opto_key).get_cylcle_begin_timepoints(
                opto_key
            )
            pulse_timepoints = []
            for t in cycle_start:
                pulse_timepoints.append(t)
                for dummy_count in range(10):
                    pulse_timepoints.append(t + dummy_count * 1 / dummy_freq)
            pulse_timepoints = np.asarray(pulse_timepoints)

        else:
            raise ValueError(
                "marks must be in [first_pulse, all_pulses, theta_peaks, dummy_cycle]"
            )

        period = (OptoStimProtocol() & opto_key).fetch1("period_ms")
        # get time-binned spike counts
        for bin_key in key_list:
            spikes_group = (BinnedSpiking() & bin_key).mark_alligned_binned_spikes(
                bin_key, marks=pulse_timepoints, rng=[plot_rng[0], plot_rng[-1]]
            )
            if not (spikes_group is None or spikes_group.shape[0] == 0):
                spike_counts.append(spikes_group.sum(axis=0))

                # shuffle the spike counts
                alligned_binned_spike_func = generate_alligned_binned_spike_func(
                    bin_key, [plot_rng[0], plot_rng[-1]]
                )
                if "period_ms" in dataset_key:
                    shuffle_window = dataset_key["period_ms"] / 1000.0
                else:
                    shuffle_window = 0.125
                n_shuffles = 100
                spike_counts_shuffled.extend(
                    shuffled_spiking_distribution(
                        marks=pulse_timepoints,
                        alligned_binned_spike_func=alligned_binned_spike_func,
                        n_shuffles=n_shuffles,
                        shuffle_window=shuffle_window,
                    )
                )
                # if len(spike_counts) > 3:
                #     break
                # # break  # TODO: remove this break

    if len(spike_counts) == 0 or len(pulse_timepoints) == 0:
        if return_data:
            return None, [], [], []
        return
    spike_counts = np.concatenate(spike_counts, axis=0)  # shape = (units,bins)
    # spike_counts_shuffled = np.concatenate(
    #     spike_counts_shuffled, axis=0
    # )  # shape = (units, marks, bins)
    # spike_counts = np.sum(spike_counts, axis=0)
    ind = spike_counts.sum(axis=1) > 1e1
    spike_counts = spike_counts[ind]
    spike_counts_shuffled = np.array(spike_counts_shuffled)[ind]
    # print("opto", opto_key)
    # print("key_list", key_list)

    if len(spike_counts) == 0:
        if return_data:
            return None, [], [], []
        return

    # calculate KL divergence
    KL = [
        discrete_KL_divergence(s, q="uniform", laplace_smooth=True)
        for s in spike_counts
    ]

    # calculate the bootstrap statistics of the null distribution for the
    # measurement on each unit
    unit_measurement_null_dist_mean = []
    unit_measurement_null_dist_rng = []
    unit_sig_modulated = []
    for i in range(spike_counts_shuffled.shape[0]):
        x, rng = bootstrap(
            spike_counts_shuffled[i],
            measurement=stacked_marks_to_kl,
            n_samples=int(spike_counts_shuffled[i].shape[0] / n_shuffles),
            n_boot=1000,
        )
        unit_measurement_null_dist_mean.append(x)
        unit_measurement_null_dist_rng.append(rng)
        # print(x, rng, KL[i])
        if KL[i] > rng[1]:
            unit_sig_modulated.append(True)
        else:
            unit_sig_modulated.append(False)

    # make figure
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 16)
    ax = [
        fig.add_subplot(gs[:, :5]),
        fig.add_subplot(gs[:, 6:11]),
        fig.add_subplot(gs[:, 11]),
        fig.add_subplot(gs[:, 13:14]),
        fig.add_subplot(gs[:, 14:16]),
    ]
    # fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    # plot traces

    tp = np.linspace(plot_rng[0], plot_rng[-1], spike_counts.shape[1]) * 1000
    plot_spikes = (spike_counts[:].T).copy()
    mua = np.sum(plot_spikes, axis=1)
    plot_spikes = plot_spikes / plot_spikes[:].mean(axis=0)
    mua = mua / mua[:].mean()
    plot_spikes = smooth(plot_spikes, 5)
    mua = smooth(mua[:, None], 5)

    ax[0].plot(
        tp,
        np.log10(plot_spikes),
        alpha=min(5.0 / plot_spikes.shape[1], 0.4),
        c="cornflowerblue",
    )
    ax[0].plot(
        tp,
        np.log10(np.nanmean((plot_spikes), axis=1)),
        c="cornflowerblue",
        linewidth=3,
        label="multi unit activity",
    )

    if marks == "first_pulse" or period == -1:
        ax[0].fill_between(
            [0, 0 + 40], [-1, -1], [1, 1], facecolor="thistle", alpha=0.3
        )
    elif marks in ["all_pulses", "odd_pulses"] and period is not None:
        if period is not None:
            t = 0
            while t < tp.max():
                ax[0].fill_between(
                    [t, t + 40], [-1, -1], [1, 1], facecolor="thistle", alpha=0.5
                )
                t += period
            t = -period
            while t + 40 > tp.min():
                ax[0].fill_between(
                    [t, t + 40], [-1, -1], [1, 1], facecolor="thistle", alpha=0.5
                )
                t -= period

    ax[0].set_ylim(-1, 1)
    ax[0].set_xlim(tp[0], tp[-1])
    ax[0].set_xlabel("time (ms)")
    ax[0].set_ylabel("log10 Normalized firing rate ")
    ax[0].spines[["top", "right"]].set_visible(False)
    ax[0].legend()
    # ax[0].set_title(dataset)

    # plot heatmap of normalized firing rate
    ind_peak = np.arange(tp.size // 2)  # np.where((tp > -10) & (tp < period))[0]
    peak_time = np.argmax(plot_spikes[ind_peak], axis=0)
    peak_order = np.argsort(peak_time)
    sig_unit = [i for i in peak_order if unit_sig_modulated[i]]
    not_sig_unit = [i for i in peak_order if not unit_sig_modulated[i]]
    peak_order = sig_unit + not_sig_unit

    ax[1].matshow(
        np.log10(plot_spikes[:, peak_order].T),
        cmap="RdBu_r",
        origin="lower",
        clim=(-0.5, 0.5),
        extent=(tp[0], tp[-1], 0, plot_spikes.shape[1]),
        aspect="auto",
    )

    num_sig = np.sum(unit_sig_modulated)
    ax[1].fill_between(
        [tp[0], tp[-1]],
        [num_sig, num_sig],
        [plot_spikes.shape[1], plot_spikes.shape[1]],
        facecolor="grey",
        alpha=0.1,
    )
    ax[1].fill_between(
        [tp[0], tp[-1]],
        [num_sig, num_sig],
        [plot_spikes.shape[1], plot_spikes.shape[1]],
        facecolor="none",
        alpha=0.7,
        hatch="/",
        edgecolor="grey",
    )

    ax[1].plot(
        [
            0,
            0,
        ],
        [0, plot_spikes.shape[1]],
        ls="--",
        c="k",
        lw=2,
    )
    ax[1].plot(
        [
            40,
            40,
        ],
        [0, plot_spikes.shape[1]],
        ls="--",
        c="k",
        lw=2,
    )

    # violinplot of kl divergence across units
    ax[3].violinplot(
        KL,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    ax[3].scatter([1], [np.nanmean(KL)], c="k", s=50)
    ax[3].set_ylabel("KL divergence")
    ax[3].set_xticks([])
    ax[3].spines[["top", "right", "bottom"]].set_visible(False)

    # Table with information about the dataset
    the_table = ax[4].table(
        cellText=[[len(dataset)], [marks]] + [[str(x)] for x in dataset_key.values()],
        rowLabels=["number_epochs", "marks"] + [str(x) for x in dataset_key.keys()],
        loc="right",
        colWidths=[0.6, 0.6],
    )
    ax[4].spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax[4].set_xticks([])
    ax[4].set_yticks([])

    # colorbar
    # ax[] = fig.add_subplot(gs[:,-1])
    plt.colorbar(
        cm.ScalarMappable(mpl.colors.Normalize(-0.5, 0.5), cmap="RdBu_r"),
        cax=ax[2],
        label="log10 Normalized firing rate",
    )

    ax[1].set_xlabel("time (ms)")
    ax[1].set_ylabel("Unit #")
    ax[1].set_xlim(tp[0], tp[-1])
    ax[2].set_ylabel("log 10 normalized firing rate")

    fig.canvas.draw()
    plt.rcParams["svg.fonttype"] = "none"

    print(num_sig, len(unit_sig_modulated))
    if return_data:
        return fig, spike_counts, tp, KL
    return fig


###################################################################
# Theta distribution analysis
def spiking_theta_distribution(
    dataset_key: dict,
    n_bins: int = 20,
    band_filter_name: str = "Theta 5-11 Hz",
):
    # define datasets
    dataset = filter_opto_data(dataset_key)

    # loop through datasets and get relevant results
    spike_phase_list = [[], []]  # control, test
    for nwb_file_name, position_interval_name in tqdm(
        zip(dataset.fetch("nwb_file_name"), dataset.fetch("interval_list_name"))
    ):
        interval_name = (
            PositionIntervalMap()
            & {
                "nwb_file_name": nwb_file_name,
                "position_interval_name": position_interval_name,
            }
        ).fetch1("interval_list_name")
        basic_key = {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": interval_name,
        }
        band_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": position_interval_name,
            "filter_name": band_filter_name,
        }
        pos_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": position_interval_name,
        }

        # check if spiking data exists
        if len(CuratedSpikeSorting() & basic_key) == 0:
            print("no spiking data for", basic_key)
            continue
        if len(TrodesPosV1() & pos_key) == 0:
            print("no position data for", pos_key)
            continue
        print(basic_key)

        # get times when rat is running and not in port
        # make intervals where rat is running
        filter_speed = 4
        filter_ports = True
        run_intervals = get_running_intervals(**pos_key, filter_speed=filter_speed)
        # intersect with position-defined intervals
        if filter_ports:
            valid_position_intervals = filter_position_ports(pos_key)
            run_intervals = interval_list_intersect(
                np.array(run_intervals), np.array(valid_position_intervals)
            )

        # define the test and control intervals
        spike_df = []
        restrict_interval_list = [
            (OptoStimProtocol() & pos_key).fetch1("control_intervals"),
            (OptoStimProtocol() & pos_key).fetch1("test_intervals"),
        ]
        restrict_interval_list = [
            interval_list_intersect(np.array(restrict_interval), run_intervals)
            for restrict_interval in restrict_interval_list
        ]

        # get phase information
        ref_elect, basic_key = get_ref_electrode_index(basic_key)  #
        phase_df = (LFPBandV1() & band_key).compute_signal_phase(
            electrode_list=[ref_elect]
        )
        phase_time = phase_df.index
        phase_ = np.asarray(phase_df)[:, 0]

        # get the spike and position dat for each
        for sort_group in set(
            (CuratedSpikeSorting() & basic_key).fetch("sort_group_id")
        ):
            key = {"sort_group_id": sort_group}
            cur_id = np.max(
                (CuratedSpikeSorting() & basic_key & key).fetch("curation_id")
            )
            key["curation_id"] = cur_id
            tetrode_df = (CuratedSpikeSorting & basic_key & key).fetch_nwb()[0]
            if "units" in tetrode_df:
                tetrode_df = tetrode_df["units"]
                tetrode_df = tetrode_df[tetrode_df.label == ""]
                spike_df.append(tetrode_df)
        spike_df = pd.concat(spike_df)

        # determin the phase for each spike
        for ii, restrict_interval in enumerate(restrict_interval_list):
            spike_phase = []
            for spikes in tqdm(spike_df.spike_times):
                # find phase time bin of each spike
                spikes = interval_list_contains(restrict_interval, spikes)
                spikes = interval_list_contains(
                    [[phase_time[0], phase_time[-1]]], spikes
                )
                spike_ind = np.digitize(spikes, phase_time, right=False)
                spike_phase.append(phase_[spike_ind])
            spike_phase_list[ii].append(spike_phase)

    # make figure
    spike_phase_list = [np.concatenate(spike_phase) for spike_phase in spike_phase_list]
    fig, ax = plt.subplots(ncols=4, figsize=(15, 5))
    bins = np.linspace(0, 2 * np.pi, n_bins)
    kl_list = [[], []]
    for i, spike_phase in enumerate(spike_phase_list):
        phase_density = []
        for spikes in spike_phase:
            if spikes.size < 10:
                continue
            yy, _ = np.histogram(
                spikes,
                bins=bins,
            )
            phase_density.append(yy / np.mean(yy))
            kl_list[i].append(discrete_KL_divergence(yy, q="uniform", pool_bins=1))

        phase_density = np.asarray(phase_density)
        peak_time = np.argmax(phase_density, axis=1)
        peak_order = np.argsort(peak_time)

        ax[i].imshow(
            np.log10(np.asarray(phase_density[peak_order])),
            cmap="RdBu_r",
            origin="lower",
            clim=(-0.5, 0.5),
        )
    ax[2].violinplot(
        kl_list,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for a in ax[:2]:
        a.set_xticks([0, n_bins / 2, n_bins])
        a.set_xticklabels(["0", "$\pi$", "$2\pi$"])
        a.set_yticks([])
        a.set_xlabel("phase")
        a.set_ylabel("unit #")
    ax[2].set_xticks([1, 2])
    ax[2].set_xticklabels(["control", "test"])
    ax[2].set_ylabel("KL divergence")

    # Table with information about the dataset
    the_table = ax[3].table(
        cellText=[
            [len(dataset)],
        ]
        + [[str(x)] for x in dataset_key.values()],
        rowLabels=[
            "number_epochs",
        ]
        + [str(x) for x in dataset_key.keys()],
        loc="right",
        colWidths=[0.6, 0.6],
    )
    ax[3].spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax[3].set_xticks([])
    ax[3].set_yticks([])


###################################################################
# Place Fields


def place_field_analysis(
    dataset_key: dict,
    plot_rng: np.ndarray = np.arange(-0.08, 0.2, 0.002),
    first_pulse_only: bool = False,
):
    # define datasets
    dataset = filter_opto_data(dataset_key)

    # loop through datasets and get relevant results
    place_fields_list = [[], []]
    spatial_information_rate_list = [[], []]
    for nwb_file_name, position_interval_name in tqdm(
        zip(dataset.fetch("nwb_file_name"), dataset.fetch("interval_list_name"))
    ):
        interval_name = (
            PositionIntervalMap()
            & {
                "nwb_file_name": nwb_file_name,
                "position_interval_name": position_interval_name,
            }
        ).fetch1("interval_list_name")
        basic_key = {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": interval_name,
        }
        pos_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": position_interval_name,
        }
        # check if spiking data exists
        if len(CuratedSpikeSorting() & basic_key) == 0:
            print("no spiking data for", basic_key)
            continue
        if len(TrodesPosV1() & pos_key) == 0:
            print("no position data for", pos_key)
            continue

        # define the test and control intervals
        spike_df = []
        restrict_interval_list = [
            (OptoStimProtocol() & pos_key).fetch1("control_intervals"),
            (OptoStimProtocol() & pos_key).fetch1("test_intervals"),
        ]
        # get the spike and position dat for each
        for sort_group in set(
            (CuratedSpikeSorting() & basic_key).fetch("sort_group_id")
        ):
            key = {"sort_group_id": sort_group}
            cur_id = np.max(
                (CuratedSpikeSorting() & basic_key & key).fetch("curation_id")
            )
            key["curation_id"] = cur_id
            tetrode_df = (CuratedSpikeSorting & basic_key & key).fetch_nwb()[0]
            if "units" in tetrode_df:
                tetrode_df = tetrode_df["units"]
                tetrode_df = tetrode_df[tetrode_df.label == ""]
                spike_df.append(tetrode_df)
        spike_df = pd.concat(spike_df)
        pos_df = (TrodesPosV1() & pos_key).fetch1_dataframe()
        pos_time = np.asarray(pos_df.index)
        # determin the position fo each spike
        spike_pos_list = []
        for restrict_interval in restrict_interval_list:
            spike_pos = []
            for ii, spikes in tqdm(enumerate(spike_df.spike_times)):
                # find position time bin of each spike
                spikes = interval_list_contains(restrict_interval, spikes)
                spike_ind = np.digitize(
                    spikes,
                    pos_time,
                )
                spike_pos.append(pos_df.position_x.iloc[spike_ind].values)
            spike_pos_list.append(spike_pos)

        crop = 10
        rng = np.linspace(
            np.min(pos_df.position_x) + crop, np.max(pos_df.position_x) - crop, 100
        )  # TODO: define more consistently
        occupancy_list = [
            np.histogram(
                pos_df.position_x[
                    interval_list_contains(restrict_interval, pos_df.index)
                ],
                bins=rng,
            )[0]
            for restrict_interval in restrict_interval_list
        ]
        occupancy_list = [
            smooth(occupancy, int(0.1 * rng.size)) for occupancy in occupancy_list
        ]

        for i in range(len(spike_pos)):
            val_list = []
            sir = []
            keep = False
            sufficient_count = True
            for spike_pos, restrict_interval, occupancy in zip(
                spike_pos_list, restrict_interval_list, occupancy_list
            ):
                val = np.histogram(spike_pos[i], bins=rng)[0]
                if val.sum() < 100:
                    sufficient_count = False
                val = smooth(val, int(0.1 * rng.size))
                sir.append(spatial_information_rate(val, occupancy))

                val = (val / occupancy) / (
                    val.sum() / occupancy.sum()
                )  # p(spike|pos)/p(spike)
                # val = np.log(val)
                val_list.append(val)
                if np.nanmax(val) > 3:
                    keep = True

            if keep and sufficient_count:
                # print(i)
                for ii, val in enumerate(val_list):
                    place_fields_list[ii].append(val)
                    spatial_information_rate_list[ii].append(sir[ii])
    if len(place_fields_list[0]) == 0:
        return
    # fig = plt.figure(figsize=(5, 10))
    fig, ax = plt.subplots(ncols=3, figsize=(13, 5))
    if len(place_fields_list[0]) == 1:
        sort_peak = [0]
    else:
        peak = np.nanargmax(place_fields_list[0], axis=1)
        sort_peak = np.argsort(peak)
    plot_count = 0
    shift = 10
    labels = ["control", "test"]
    for i in sort_peak:
        # print("i",i)
        # rng[:-1]
        # print(place_fields_list[0][0].shape)#[int(i)]
        # labels[0]
        ax[0].plot(
            rng[:-1],
            place_fields_list[0][i] + shift * plot_count,
            c="cornflowerblue",
        )  # label=labels[0])
        ax[0].plot(
            rng[:-1],
            place_fields_list[1][i] + shift * plot_count,
            c="firebrick",
            label=labels[1],
        )
        plot_count += 1
        labels = [None, None]

    ax[0].set_ylabel("p(spike|pos)/p(spike)")
    ax[0].set_xlabel("position (cm)")
    ax[0].spines[["top", "right"]].set_visible(False)
    # ax[0].set_title(f"place fields {basic_key}")
    ax[0].legend()

    # plot information rates
    violin = ax[1].violinplot(
        spatial_information_rate_list[0],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc in violin["bodies"]:
        pc.set_facecolor("cornflowerblue")
        pc.set_alpha(0.5)
    violin = ax[1].violinplot(
        spatial_information_rate_list[1],
        positions=[2],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc in violin["bodies"]:
        pc.set_facecolor("firebrick")
        pc.set_alpha(0.5)
    violin = ax[1].violinplot(
        np.array(spatial_information_rate_list[1])
        - np.array(spatial_information_rate_list[0]),
        positions=[3],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc in violin["bodies"]:
        pc.set_facecolor("grey")
        pc.set_alpha(0.5)
    ax[1].set_xticks([1, 2, 3])
    ax[1].set_xticklabels(["control", "test", "test-control"])
    ax[1].set_ylabel("spatial information rate (bits/spike)")
    ax[1].spines[["top", "right"]].set_visible(False)

    # table of experiment information
    the_table = ax[2].table(
        cellText=[[len(dataset)]] + [[str(x)] for x in dataset_key.values()],
        rowLabels=["number_epochs"] + [str(x) for x in dataset_key.keys()],
        loc="right",
        colWidths=[0.6, 0.6],
    )
    ax[2].spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    return fig


"""
UTILS
"""


def get_spikecount_per_time_bin(spike_times, time):
    spike_times = spike_times[
        np.logical_and(spike_times >= time[0], spike_times <= time[-1])
    ]
    return np.bincount(
        np.digitize(spike_times, time[1:-1]),
        minlength=time.shape[0],
    )


def gkern(l: int = 5, sig: float = 1.0):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return gauss / np.sum(gauss)
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def smooth(data, n=5, sigma=None):
    """smooths data with gaussian kernel of size n"""
    if n % 2 == 0:
        n += 1  # make sure n is odd
    if sigma is None:
        sigma = n / 2
    kernel = gkern(n, sigma)[:, None]
    if len(data.shape) == 1:
        pad = np.ones(((n - 1) // 2, 1))
        return np.squeeze(
            scipy.signal.convolve2d(
                np.concatenate(
                    [pad * data[:, None][0], data[:, None], pad * data[:, None][-1]],
                    axis=0,
                ),
                kernel,
                mode="valid",
            )
        )
    else:
        pad = np.ones(((n - 1) // 2, data.shape[1]))
        return scipy.signal.convolve2d(
            np.concatenate([pad * data[0], data, pad * data[-1]], axis=0),
            kernel,
            mode="valid",
        )


def spatial_information_rate(spike_counts, occupancy):
    """
    Calculates the spatial information rate of units firing
    Formula from:
    Experience-Dependent Increase in CA1 Place Cell Spatial Information, But Not Spatial Reproducibility,
    Is Dependent on the Autophosphorylation of the α-Isoform of the Calcium/Calmodulin-Dependent Protein Kinase II
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2680063/
    """

    spike_rate = spike_counts / occupancy
    p_loc = occupancy / occupancy.sum()
    total_rate = spike_counts.sum() / occupancy.sum()
    return np.nansum(p_loc * spike_rate / total_rate * np.log2(spike_rate / total_rate))

    """Calculates the spatial information rate of units firing
    Formula from:
    Experience-Dependent Increase in CA1 Place Cell Spatial Information, But Not Spatial Reproducibility,
    Is Dependent on the Autophosphorylation of the α-Isoform of the Calcium/Calmodulin-Dependent Protein Kinase II
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2680063/

    Parameters
    ----------
    spike_counts : np.ndarray
        counts in each spatial bin
    occupancy : np.ndarray
        occupancy in each spatial bin

    Returns
    -------
    np.ndarray
        Spatial information rate
    """
    # spike_rate = spike_counts / occupancy
    # p_loc = occupancy / occupancy.sum()
    # total_rate = spike_counts.sum() / occupancy.sum()
    # return np.nansum(p_loc * spike_rate * np.log2(spike_rate / total_rate))
    # return np.nansum(spike_rate * np.log2(spike_rate))


os.chdir("/home/sambray/Documents/MS_analysis_samsplaying/")
from Analysis.lfp_analysis import get_ref_electrode_index
from spyglass.lfp.analysis.v1 import LFPBandV1
from spyglass.common import interval_list_intersect


def get_theta_peaks(key):
    if not LFPBandV1 & key:
        map_key = key.copy()
        map_key["interval_list_name"] = key["target_interval_list_name"]
        pos_interval = (PositionIntervalMap() & map_key).fetch1(
            "position_interval_name"
        )
        key["target_interval_list_name"] = pos_interval
        if not LFPBandV1 & key:
            print("no theta band for", key)
            return []
    ref_elect, basic_key = get_ref_electrode_index(key)  # get reference electrode
    filter_key = {"filter_name": "Theta 5-11 Hz"}
    # get phase information
    phase_df = (LFPBandV1() & key & filter_key).compute_signal_phase(
        electrode_list=[ref_elect]
    )
    phase_time = phase_df.index
    phase_ = np.asarray(phase_df)[:, 0]

    # find positive zero crossings
    target_phase = np.pi  # TODO:pick this
    phase_ -= target_phase
    pos_zero_crossings = np.where(np.diff(np.sign(phase_)) > 0)[0]
    marks = phase_time[pos_zero_crossings]

    # filter against times rat is running and not in port
    pos_key = {"nwb_file_name": key["nwb_file_name"]}
    pos_key["interval_list_name"] = key["target_interval_list_name"]
    if not (TrodesPosV1 & pos_key):
        # try converting to position interval name
        pos_interval = (PositionIntervalMap() & key).fetch1("position_interval_name")
        pos_key2 = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": pos_interval,
        }
        if not (TrodesPosV1 & pos_key2):
            print("no position data for", key)
            return []
        else:
            pos_key = pos_key2

    # make intervals where rat is running
    filter_speed = 10
    run_intervals = get_running_intervals(**pos_key, filter_speed=filter_speed)
    # print("run", run_intervals)
    # intersect with position-defined intervals
    valid_position_intervals = filter_position_ports(pos_key)
    if len(valid_position_intervals) == 0:
        return []
    # print("position", valid_position_intervals)
    run_intervals = interval_list_intersect(
        np.array(run_intervals), np.array(valid_position_intervals)
    )

    # return the theta marks from these intervals
    return interval_list_contains(run_intervals, marks)
