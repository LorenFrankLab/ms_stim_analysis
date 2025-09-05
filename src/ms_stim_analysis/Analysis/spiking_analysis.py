import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from spyglass.common import (
    IntervalList,
    PositionIntervalMap,
    TaskEpoch,
    convert_epoch_interval_name_to_position_interval_name,
    interval_list_contains,
    interval_list_intersect,
)
from spyglass.common.common_interval import Interval
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1
from spyglass.lfp.analysis.v1 import LFPBandV1
from spyglass.position.v1 import TrodesPosV1
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.spikesorting.v0 import CuratedSpikeSorting

from .circular_shuffle import shuffled_spiking_distribution
from .lfp_analysis import get_ref_electrode_index
from .position_analysis import filter_position_ports, get_running_intervals
from .utils import smooth, filter_opto_data
from .spiking_place_fields import decoding_place_fields
from ms_stim_analysis.AnalysisTables.ms_opto_stim_protocol import OptoStimProtocol

from .circular_shuffle import discrete_KL_divergence, stacked_marks_to_kl, bootstrap


##################################################################################
def bin_spikes_around_marks(spikes, marks, bins):
    delays = np.subtract.outer(spikes, marks)
    delays = delays[delays < bins[-1]]
    delays = delays[delays >= bins[0]]
    vals, _ = np.histogram(delays, bins=bins)
    vals = vals + 1e-9
    return vals


def opto_spiking_dynamics(
    dataset_key: dict,
    plot_rng: np.ndarray = np.arange(-0.08, 0.2, 0.002),
    marks="first_pulse",
    return_data: bool = False,
    limit_1_epoch: bool = False,
    neuron_type: str = None,
):
    """Function to plot the spiking dynamics around opto stimulations

    Parameters
    ----------
    dataset_key : dict
        restriction defining what data to analyze
    plot_rng : np.ndarray, optional
        time bins to analyze around the stim, by default np.arange(-0.08, 0.2, 0.002)
    marks : str, optional
        how to allign the data,. Valid options are first_pulse, all_pulses, theta_peaks, dummy_pulse=x, by default "first_pulse"
    return_data : bool, optional
        whether to return the , by default False
    limit_1_epoch : bool, optional
        if True only analysze the first matching epoch in the data, by default False
    neuron_type : str, optional
        if not None, only analyze putative neurons of this type as defined by firing rate, by default None

    Returns
    -------
    fig
        plotted figure
    spike_counts
        array of spike counts for each unit
    tp
        time points for the spike counts
    KL
        KL divergence for each unit
    """
    # get the filtered data
    dataset = filter_opto_data(dataset_key)
    nwb_file_name_list = dataset.fetch("nwb_file_name")
    position_interval_name_list = dataset.fetch("interval_list_name")
    if limit_1_epoch:
        nwb_file_name_list = nwb_file_name_list[:1]
        position_interval_name_list = position_interval_name_list[:1]

    if len(plot_rng) < 3:
        raise ValueError("plot_rng is the histogram bins for plotting")
    # compile the data
    spike_counts = []
    spike_counts_shuffled = []
    for nwb_file_name, position_interval_name in tqdm(
        zip(nwb_file_name_list, position_interval_name_list)
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
            "sorted_spikes_group_name": interval_name,
        }
        print(basic_key)
        # get spike times for this interval
        if not SortedSpikesGroup() & basic_key:
            print("no compiled spiking data for", basic_key)
            continue
        sorted_group_key = (SortedSpikesGroup() & basic_key).fetch1("KEY")
        spikes = SortedSpikesGroup().fetch_spike_data(sorted_group_key)
        # filter based on overall firing rate
        sort_interval = (
            IntervalList
            & {"nwb_file_name": nwb_file_name, "interval_list_name": interval_name}
        ).fetch1("valid_times")
        sort_time = np.sum([e[1] - e[0] for e in sort_interval])
        rate = np.array([len(s) for s in spikes]) / sort_time
        if neuron_type == "pyramidal":
            spikes = [s for s, r in zip(spikes, rate) if r < 5]
        elif neuron_type == "interneuron":
            spikes = [s for s, r in zip(spikes, rate) if r > 5]

        pos_interval_name = convert_epoch_interval_name_to_position_interval_name(
            {"nwb_file_name": nwb_file_name, "interval_list_name": interval_name}
        )
        opto_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": pos_interval_name,
        }

        # Define what marks we're alligning to
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

        interval_restrict = np.array(
            [[tp + plot_rng[0], tp + plot_rng[-1]] for tp in pulse_timepoints]
        )

        interval_restrict = (
            Interval(interval_restrict, no_overlap=True).consolidate().times
        )

        period = (OptoStimProtocol() & opto_key).fetch1("period_ms")
        if "period_ms" in dataset_key:
            shuffle_window = dataset_key["period_ms"] / 1000.0
        else:
            shuffle_window = 0.125
        n_shuffles = 10
        interval_restrict_shuffle = np.array(
            [
                [tp + plot_rng[0] - shuffle_window, tp + plot_rng[-1] + shuffle_window]
                for tp in pulse_timepoints
            ]
        )
        interval_restrict_shuffle = (
            Interval(interval_restrict_shuffle, no_overlap=True).consolidate().times
        )

        # get histogram spike counts
        for unit_spikes in spikes:
            unit_spikes_restricted = np.unique(
                interval_list_contains(interval_restrict, unit_spikes)
            )
            vals = bin_spikes_around_marks(
                unit_spikes_restricted, pulse_timepoints, plot_rng
            )
            spike_counts.append(vals)
            # if gauss_smooth:
            #     vals = smooth(
            #         vals, int(gauss_smooth / np.mean(np.diff(histogram_bins)))
            #     )
            # break
            unit_spikes_restricted = np.unique(
                interval_list_contains(interval_restrict_shuffle, unit_spikes)
            )

            def alligned_binned_spike_func(marks):
                return np.array(
                    [
                        bin_spikes_around_marks(unit_spikes_restricted, m, plot_rng)
                        for m in marks
                    ]
                )[None, :, :]

            spike_counts_shuffled.extend(
                shuffled_spiking_distribution(
                    marks=pulse_timepoints,
                    alligned_binned_spike_func=alligned_binned_spike_func,
                    n_shuffles=n_shuffles,
                    shuffle_window=shuffle_window,
                )
            )

    if len(spike_counts) == 0 or len(pulse_timepoints) == 0:
        if return_data:
            return None, [], [], []
        return
    spike_counts = np.array(spike_counts)  # shape = (units,bins)
    spike_counts_shuffled = np.array(spike_counts_shuffled, dtype=object)
    ind = spike_counts.sum(axis=1) > 1e1
    spike_counts = spike_counts[ind]
    spike_counts_shuffled = np.array(spike_counts_shuffled, dtype=object)[ind]

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
        cellText=[[len(dataset)], [marks], [neuron_type]]
        + [[str(x)] for x in dataset_key.values()],
        rowLabels=["number_epochs", "marks", "neuron_type"]
        + [str(x) for x in dataset_key.keys()],
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
# Place Field + Opto Stimulation
def opto_spiking_dynamics_place_dependence(
    dataset_key: dict,
    plot_rng: np.ndarray = np.arange(-0.08, 0.2, 0.002),
    marks="first_pulse",
    return_data: bool = False,
    place_field_ranges: list = [
        [0, 5],
        [10, 99999],
    ],  # distance from place field center in cm
    normalize_rates: bool = True,
):
    # get the filtered data
    dataset = filter_opto_data(dataset_key)
    n_place = len(place_field_ranges)

    # compile the data
    spike_counts_list = [[] for _ in range(n_place)]
    spike_counts_shuffled_list = [[] for _ in range(n_place)]
    marks_counts_list = [[] for _ in range(n_place)]
    for nwb_file_name, position_interval_name in tqdm(
        zip(dataset.fetch("nwb_file_name")[:1], dataset.fetch("interval_list_name")[:1])
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
            "sorted_spikes_group_name": interval_name,
        }
        print(basic_key)

        # get place field data
        if not SortedSpikesDecodingV1() & basic_key:
            print("no place field data for", basic_key)
            continue
        place_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": position_interval_name,
        }
        place_fields, place_bins = decoding_place_fields(
            dataset_key=place_key,
            return_place_fields=True,
            plot=False,
            return_correlations=False,
            filter_specificity=False,
            min_rate=0,
            return_place_field_centers=True,
        )
        place_bins = np.array(place_bins)
        # define the position of the center of each place field
        place_field_centers = place_bins[np.argmax(place_fields[1], axis=1)]

        # get position info for this interval
        decode_key = (
            SortedSpikesDecodingV1()
            & basic_key
            & {"encoding_interval": position_interval_name + "_opto_test_interval"}
            & {"position_group_name": position_interval_name}
        ).fetch1("KEY")
        pos_df = SortedSpikesDecodingV1().fetch_linear_position_info(decode_key)

        # get spike times for this interval
        # if not SortedSpikesGroup() & basic_key:
        #     print("no compiled spiking data for", basic_key)
        #     continue
        # sorted_group_key = (SortedSpikesGroup() & basic_key).fetch1("KEY")
        # spikes = SortedSpikesGroup().fetch_spike_data(sorted_group_key)
        spikes = SortedSpikesDecodingV1().fetch_spike_data(decode_key)

        pos_interval_name = convert_epoch_interval_name_to_position_interval_name(
            {"nwb_file_name": nwb_file_name, "interval_list_name": interval_name}
        )
        opto_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": pos_interval_name,
        }

        # Define what marks we're alligning to
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
        n_marks = len(pulse_timepoints)
        # get position info of pulse_timepoints
        pos_ind = np.digitize(pulse_timepoints, pos_df.index.values)
        pulse_pos = pos_df.linear_position.iloc[pos_ind].values
        # general restriction on spike times considered
        interval_restrict = np.array(
            [[tp + plot_rng[0], tp + plot_rng[-1]] for tp in pulse_timepoints]
        )
        period = (OptoStimProtocol() & opto_key).fetch1("period_ms")
        if "period_ms" in dataset_key:
            shuffle_window = dataset_key["period_ms"] / 1000.0
        else:
            shuffle_window = 0.125
        n_shuffles = 10
        interval_restrict_shuffle = np.array(
            [
                [tp + plot_rng[0] - shuffle_window, tp + plot_rng[-1] + shuffle_window]
                for tp in pulse_timepoints
            ]
        )
        spikes = [s for s in spikes if len(s) > 0]

        for n_pos, pos_range in enumerate(place_field_ranges):
            # get histogram spike counts
            for n_spike, unit_spikes in enumerate(spikes[:]):
                # define what pulses happen in the right position relative to unit's place field
                center_loc = place_field_centers[n_spike]
                pulse_distance = np.abs(pulse_pos - center_loc)
                pulse_ind = np.where(
                    (pulse_distance > pos_range[0]) & (pulse_distance < pos_range[1])
                )[0]
                if len(pulse_ind) == 0:
                    spike_counts_list[n_pos].append(np.ones(plot_rng.size - 1) * np.nan)
                    spike_counts_shuffled_list[n_pos].extend(
                        np.ones((1, 10, plot_rng.size - 1)) * np.nan
                    )
                    marks_counts_list[n_pos].append(len(pulse_ind))
                    continue

                unit_spikes_restricted = interval_list_contains(
                    interval_restrict, unit_spikes
                )
                vals = bin_spikes_around_marks(
                    unit_spikes_restricted, pulse_timepoints[pulse_ind], plot_rng
                )
                spike_counts_list[n_pos].append(vals)  # / float(len(pulse_ind)))
                marks_counts_list[n_pos].append(float(len(pulse_ind)))
                # if gauss_smooth:
                #     vals = smooth(
                #         vals, int(gauss_smooth / np.mean(np.diff(histogram_bins)))
                #     )
                # break
                unit_spikes_restricted = interval_list_contains(
                    interval_restrict_shuffle, unit_spikes
                )

                def alligned_binned_spike_func(marks):
                    return np.array(
                        [
                            bin_spikes_around_marks(unit_spikes_restricted, m, plot_rng)
                            for m in marks
                        ]
                    )[None, :, :]

                spike_counts_shuffled_list[n_pos].extend(
                    shuffled_spiking_distribution(
                        marks=pulse_timepoints[pulse_ind],
                        alligned_binned_spike_func=alligned_binned_spike_func,
                        n_shuffles=n_shuffles,
                        shuffle_window=shuffle_window,
                    )
                )
        # break

    # if len(spike_counts_list) == 0 or len(pulse_timepoints) == 0:
    #     if return_data:
    #         return None, [], [], []
    #     return
    spike_counts_list = [
        np.array(x) for x in spike_counts_list
    ]  # list[shape = (units,bins)]
    spike_counts_shuffled_list = [
        np.array(x) for x in spike_counts_shuffled_list
    ]  # list[shape = (units, marks, bins)]
    marks_counts_list = [
        np.array(x) for x in marks_counts_list
    ]  # list[shape = (units)]

    """""" ""
    # make figure
    fig = plt.figure(figsize=(18, 5 * len(place_field_ranges)))
    gs = gridspec.GridSpec(len(place_field_ranges), 16)
    ax_list = [
        [
            fig.add_subplot(gs[i, :5]),
            fig.add_subplot(gs[i, 6:11]),
            fig.add_subplot(gs[i, 11]),
            fig.add_subplot(gs[i, 13:14]),
            fig.add_subplot(gs[i, 14:16]),
        ]
        for i in range(len(place_field_ranges))
    ]

    ind = None
    peak_order = None
    mua_rng = None
    for i in range(len(place_field_ranges)):
        ax = ax_list[i]
        spike_counts = spike_counts_list[i]
        spike_counts_shuffled = spike_counts_shuffled_list[i]
        track_range = place_field_ranges[i]
        mark_counts = marks_counts_list[i]
        if spike_counts.size == 0:
            continue

        if ind is None:
            ind = spike_counts.sum(axis=1) > 1e1
            # ind = np.ones(spike_counts.shape[0], dtype=bool)
        spike_counts = spike_counts[ind]
        spike_counts_shuffled = np.array(spike_counts_shuffled)[ind]
        mark_counts = mark_counts[ind]
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

        # fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
        # plot traces

        tp = np.linspace(plot_rng[0], plot_rng[-1], spike_counts.shape[1]) * 1000
        plot_spikes = (spike_counts[:].T).copy() / mark_counts
        mua = np.sum(plot_spikes / mark_counts, axis=1)
        if normalize_rates:
            plot_spikes = plot_spikes / plot_spikes[:].mean(axis=0)
            mua = mua / mua[:].mean()
        else:
            plot_spikes = plot_spikes - plot_spikes[:].mean(axis=0)  # /mark_counts
            mua = mua - mua[:].mean()
        plot_spikes = smooth(plot_spikes, 5)
        mua = smooth(mua[:, None], 5)

        if normalize_rates:
            plot_traces = np.log10(plot_spikes)
            mua_traces = np.log10(np.nanmean((plot_spikes), axis=1))
        else:
            plot_traces = plot_spikes / np.mean(np.diff(tp)) * 1000  # /mark_counts
            mua_traces = np.nansum((plot_spikes), axis=1)
            mua_traces = (
                (mua_traces - np.mean(mua_traces)) / np.mean(np.diff(tp)) * 1000
            )
        ax[0].plot(
            tp,
            plot_traces,
            alpha=min(5.0 / plot_spikes.shape[1], 0.4),
            c="cornflowerblue",
        )
        mua_ax = ax[0].twinx()
        c_mua = "darkolivegreen"
        mua_ax.plot(
            tp,
            mua_traces,
            c=c_mua,
            linewidth=3,
            label="multi unit activity",
        )
        mua_ax.set_ylabel("$\Delta$ MUA firing rate (Hz)", color=c_mua)
        if mua_rng is None:
            mua_scale = np.nanmax(np.abs(mua_traces)) * 1.1
            mua_rng = (-mua_scale, mua_scale)
        mua_ax.set_ylim(*mua_rng)

        fill_rng = 100
        if marks == "first_pulse" or period == -1:
            ax[0].fill_between(
                [0, 0 + 40],
                [-fill_rng, -fill_rng],
                [fill_rng, fill_rng],
                facecolor="thistle",
                alpha=0.3,
            )
        elif marks in ["all_pulses", "odd_pulses"] and period is not None:
            if period is not None:
                t = 0
                while t < tp.max():
                    ax[0].fill_between(
                        [t, t + 40],
                        [-fill_rng, -fill_rng],
                        [fill_rng, fill_rng],
                        facecolor="thistle",
                        alpha=0.5,
                    )
                    t += period
                t = -period
                while t + 40 > tp.min():
                    ax[0].fill_between(
                        [t, t + 40],
                        [-fill_rng, -fill_rng],
                        [fill_rng, fill_rng],
                        facecolor="thistle",
                        alpha=0.5,
                    )
                    t -= period

        if normalize_rates:
            ax[0].set_ylabel("log10 Normalized firing rate ")
            clim = (-1, 1)
        else:
            ax[0].set_ylabel("$\Delta$ firing rate (Hz)")
            clim = (-40, 40)
        ax[0].spines[["top", "right"]].set_visible(False)
        ax[0].set_ylim(*clim)
        ax[0].set_xlim(tp[0], tp[-1])
        ax[0].set_xlabel("time (ms)")
        ax[0].legend()
        # ax[0].set_title(dataset)

        # plot heatmap of normalized firing rate
        if peak_order is None:
            ind_peak = np.arange(
                tp.size // 2, tp.size
            )  # np.where((tp > -10) & (tp < period))[0]
            peak_time = np.argmin(plot_spikes[ind_peak], axis=0)
            peak_order = np.argsort(peak_time)
            sig_unit = [i for i in peak_order if unit_sig_modulated[i]]
            not_sig_unit = [i for i in peak_order if not unit_sig_modulated[i]]
            peak_order = sig_unit + not_sig_unit

        if normalize_rates:
            plot_matrix = np.log10(plot_spikes[:, peak_order].T)
            clim = (-0.5, 0.5)
        else:
            plot_matrix = plot_spikes[:, peak_order].T / np.mean(np.diff(tp)) * 1000
            clim = (-20, 20)
        ax[1].matshow(
            # np.log10(plot_spikes[:, peak_order].T),
            plot_matrix,
            cmap="RdBu_r",
            origin="lower",
            clim=clim,
            extent=(tp[0], tp[-1], 0, plot_spikes.shape[1]),
            aspect="auto",
        )

        # num_sig = np.sum(unit_sig_modulated)
        # ax[1].fill_between(
        #     [tp[0], tp[-1]],
        #     [num_sig, num_sig],
        #     [plot_spikes.shape[1], plot_spikes.shape[1]],
        #     facecolor="grey",
        #     alpha=0.1,
        # )
        # ax[1].fill_between(
        #     [tp[0], tp[-1]],
        #     [num_sig, num_sig],
        #     [plot_spikes.shape[1], plot_spikes.shape[1]],
        #     facecolor="none",
        #     alpha=0.7,
        #     hatch="/",
        #     edgecolor="grey",
        # )
        num_sig = np.sum(unit_sig_modulated)
        for i_sig, sig in enumerate(np.array(unit_sig_modulated)[np.array(peak_order)]):
            if sig:
                continue
            ax[1].fill_between(
                [tp[0], tp[-1]],
                [i_sig, i_sig],
                [i_sig + 1, i_sig + 1],
                facecolor="grey",
                alpha=0.1,
            )
            ax[1].fill_between(
                [tp[0], tp[-1]],
                [i_sig, i_sig],
                [i_sig + 1, i_sig + 1],
                facecolor="none",
                alpha=0.7,
                hatch="/",
                # edgecolor="grey",
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
            cellText=[[len(dataset)], [marks], [n_marks]]
            + [[str(x)] for x in dataset_key.values()],
            rowLabels=["number_epochs", "marks", "total marks"]
            + [str(x) for x in dataset_key.keys()],
            loc="right",
            colWidths=[0.6, 0.6],
        )
        ax[4].spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax[4].set_xticks([])
        ax[4].set_yticks([])
        ax[0].set_title(f"Distance from place field center: {track_range} cm")

        # colorbar
        # ax[] = fig.add_subplot(gs[:,-1])
        plt.colorbar(
            cm.ScalarMappable(mpl.colors.Normalize(*clim), cmap="RdBu_r"),
            cax=ax[2],
            label="log10 Normalized firing rate",
        )

        ax[1].set_xlabel("time (ms)")
        ax[1].set_ylabel("Unit #")
        ax[1].set_xlim(tp[0], tp[-1])
        if normalize_rates:
            ax[2].set_ylabel("log 10 normalized firing rate")
        else:
            ax[2].set_ylabel("$\Delta$ firing rate")

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


def spatial_information_rate(
    spike_counts=None, occupancy=None, spike_rate=None, p_loc=None
):
    """
    Calculates the spatial information rate of units firing
    Formula from:
    Experience-Dependent Increase in CA1 Place Cell Spatial Information, But Not Spatial Reproducibility,
    Is Dependent on the Autophosphorylation of the α-Isoform of the Calcium/Calmodulin-Dependent Protein Kinase II
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2680063/
    """
    if spike_counts is not None and occupancy is not None:
        spike_rate = spike_counts / occupancy
        p_loc = occupancy / occupancy.sum()
        total_rate = spike_counts.sum() / occupancy.sum()
    elif spike_rate is not None and p_loc is not None:
        total_rate = (spike_rate * p_loc).sum()
    else:
        raise ValueError(
            "spike_counts and occupancy or spike_rate and p_loc must be provided"
        )
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
