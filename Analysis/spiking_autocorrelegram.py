from spyglass.common import convert_epoch_interval_name_to_position_interval_name
from spyglass.spikesorting.v0 import CuratedSpikeSorting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spyglass.common import (
    interval_list_contains,
    interval_list_intersect,
    interval_list_contains_ind,
    PositionIntervalMap,
    TaskEpoch,
)
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1
import os

os.chdir("/home/sambray/Documents/MS_analysis_samsplaying/")
from ms_opto_stim_protocol import OptoStimProtocol
from Analysis.spiking_analysis import smooth

from Analysis.utils import filter_opto_data, get_running_valid_intervals
from Style.style_guide import interval_style


def autocorrelegram(
    dataset_key: dict,
    filter_speed: float = 10,
    min_spikes: int = 300,
    min_run_time: float = 0.5,
    return_periodicity_results: bool = False,
    return_auto_corr: bool = False,
):
    """Function that calculates autocorrelegrams and periodicity of sorted units under optogenetic stimulation

    Args:
        dataset_key (dict): restriction for the dataset
        filter_speed (float, optional): minimum running speed. Defaults to 10.
        min_spikes (int, optional): minimum number of spikes to include a unit. Defaults to 300.
        min_run_time (float, optional): minimum time rat must be running to include interval for analysis, seconds. Defaults to 0.5.
        return_periodicity_results (bool, optional): whether to periodicity results, used in plot_periodicity_dependence(). Defaults to False.
        return_auto_corr (bool, optional): whether to return the autocorrelegrams. Used for development. Defaults to False.
    Returns:
       fig: subplot figure of results
        periodicity_results (optional): list of periodicity outputs from autocorrelegram()
    """
    histogram_bins = np.arange(-0.5, 0.5, 0.002)

    # get the matching epochs
    dataset = filter_opto_data(dataset_key)
    nwb_file_names = dataset.fetch("nwb_file_name")
    pos_interval_names = dataset.fetch("interval_list_name")

    # get the autocorrelegrams
    results = [[], []]
    counts = []
    stim_results = []
    for nwb_file_name, pos_interval in zip(nwb_file_names, pos_interval_names):
        interval_name = (
            (
                PositionIntervalMap()
                & {
                    "nwb_file_name": nwb_file_name,
                    "position_interval_name": pos_interval,
                }
            )
            * TaskEpoch
        ).fetch1(
            "interval_list_name"
        )  # [0]
        basic_key = {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": interval_name,
            "sorter": "mountainsort4",
            "curation_id": 1,
        }
        pos_key = {
            "nwb_file_name": basic_key["nwb_file_name"],
            "interval_list_name": pos_interval,
        }
        # get the opto intervals
        test_interval, control_interval = (OptoStimProtocol & pos_key).fetch1(
            "test_intervals", "control_intervals"
        )
        # get the spike and position dat for each
        decode_key = {
            "nwb_file_name": nwb_file_name,
            "encoding_interval": pos_interval,
        }
        if not SortedSpikesDecodingV1 & decode_key:
            continue
        decode_key = (SortedSpikesDecodingV1 & decode_key).fetch1("KEY")
        spike_df = SortedSpikesDecodingV1().fetch_spike_data(
            decode_key, filter_by_interval=False
        )

        run_intervals = get_running_valid_intervals(
            pos_key, seperate_optogenetics=False, filter_speed=filter_speed
        )
        run_intervals = [
            interval
            for interval in run_intervals
            if interval[1] - interval[0] > min_run_time
        ]

        for spikes in spike_df:
            spikes = interval_list_contains(
                run_intervals,
                spikes,
            )
            if spikes.size < min_spikes:
                continue
            # counts.append(spikes.size)
            for i, interval in enumerate(
                [
                    control_interval,
                    test_interval,
                ]
            ):
                x = interval_list_contains(interval, spikes)
                bins = histogram_bins[:-1]
                absolute_bin_times = np.add.outer(x, bins).ravel()
                absolute_bin_index = np.array(
                    [np.arange(bins.size) for _ in range(x.size)]
                ).ravel()
                valid_interval = interval_list_intersect(
                    np.array(interval), np.array(run_intervals)
                )
                valid_bin_index = absolute_bin_index[
                    interval_list_contains_ind(valid_interval, absolute_bin_times)
                ]
                valid_bin_count = np.bincount(valid_bin_index, minlength=bins.size)

                # if x.size<300:
                #     results[i].append(np.zeros_like(histogram_bins[:-1]))
                #     if i==1:
                #         counts.append(0)
                #     continue
                if i == 1:
                    counts.append(x.size)
                delays = np.subtract.outer(x, x)
                # delays = delays[np.tril_indices_from(delays, k=0)]
                delays = np.ravel(delays)
                # delays = delays[delays != 0]
                delays = delays[
                    np.logical_and(
                        delays < histogram_bins[-1], delays > histogram_bins[0]
                    )
                ]
                # delays = delays[np.abs(delays) > 0]
                vals, bins = np.histogram(delays, bins=histogram_bins)
                vals = vals + 1e-9
                vals = vals / valid_bin_count
                sigma = int(0.015 / np.mean(np.diff(histogram_bins)))
                vals = smooth(vals, 3 * sigma, sigma)
                bins = bins[:-1] + np.diff(bins) / 2
                vals = vals / vals.sum()
                results[i].append(vals)

        stim, stim_time = OptoStimProtocol().get_stimulus(pos_key)
        stim_time = stim_time[stim == 1]
        delays = np.subtract.outer(stim_time, stim_time)
        delays = delays[np.tril_indices_from(delays, k=0)]
        delays = delays[delays < histogram_bins[-1]]
        vals, bins = np.histogram(delays, bins=histogram_bins)
        vals = vals + 1e-9
        vals = smooth(vals, int(0.015 / np.mean(np.diff(histogram_bins))))
        bins = bins[:-1] + np.diff(bins) / 2
        # vals = vals / vals.sum()
        stim_results.append(vals)

    results = [np.array(r) for r in results]
    stim_results = np.array(stim_results) * np.nan

    ## plot the results
    fig, ax = plt.subplots(
        1,
        6,
        figsize=(25, 4),
        width_ratios=[1, 1, 2, 1, 1, 0.5],
    )  # sharex=True,sharey=True)
    ind = np.argsort(counts)
    period = dataset_key["period_ms"] if "period_ms" in dataset_key else 0
    periods = [125, period, period]
    periodicity_results = []
    for i, (color, label, data) in enumerate(
        zip(
            [interval_style["control"], interval_style["test"], "purple"],
            ["control", "test", "stim"],
            [*results, stim_results],
        )
    ):
        if i < 2:
            ax[i].imshow(
                np.log10(results[i][ind, 1:]),
                aspect="auto",
                origin="lower",
                extent=[bins[0], bins[-1], 0, len(results[i])],
                clim=[-2.5, -2],
                interpolation="none",
            )
            ax[i].imshow(
                results[i][ind,],
                aspect="auto",
                origin="lower",
                extent=[bins[0], bins[-1], 0, len(results[i])],
                clim=[0.001, 0.004],
                #  cmap='bone_r'
            )
            # ax[i].imshow(results[i][1:],aspect='auto',origin='lower',
            #              extent=[bins[0],bins[-1],0,len(results[i])],
            #             #  clim=[-3,-2],
            #              )
        marks = [
            periods[i] / 1000 * n
            for n in range(1, 10)
            if periods[i] / 1000 * n < bins[-1]
        ]
        ax[i].vlines(marks, 0, len(data), color="w", ls=":")

        ax[2].plot(bins, np.median(data.T, axis=1), color=color)
        ax[2].fill_between(
            bins,
            np.quantile(data.T, 0.25, axis=1),
            np.quantile(data.T, 0.75, axis=1),
            alpha=0.3,
            facecolor=color,
        )
        marks = np.array(
            [
                periods[i] / 1000 * n
                for n in np.arange(0, 10, 1)
                if periods[i] / 1000 * n < bins[-1]
            ]
        )
        if i >= 2:
            continue
        if periods[i] < 99:
            ax[2].vlines(marks[::2], 0, len(data), ls="--", color=color, alpha=0.4)
            ax[2].vlines(marks[1::2], 0, len(data), ls=":", color=color, alpha=0.2)
        else:
            ax[2].vlines(marks, 0, len(data), ls="--", color=color, alpha=0.4)

        # get the autocorrellegram periodicity timescale
        width = 0.02
        periodicity = [
            estimate_periodicity_timescale(np.log10(x), bins, width=width) for x in data
        ]
        periodicity = np.array(periodicity)
        if periodicity.size == 0:
            continue
        periodicity = periodicity[~np.isnan(periodicity)]
        periodicity_results.append(periodicity)
        violin = ax[3].violinplot(
            periodicity, positions=[i], showmeans=False, showextrema=False
        )
        ax[3].scatter(
            np.random.normal(0, 0.1, periodicity.size) + i,
            periodicity,
            color=color,
            alpha=0.2,
        )
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.3)
        y = np.mean(results[i].T, axis=1)
        periodicity = estimate_periodicity_timescale(
            np.log10(y)[: int(bins.size)], bins, width=width
        )
        ax[3].scatter([i], [periodicity], color="k", alpha=1)

    # get the rhythmicity score for test and control
    rhythmicity_score = [rhythmicity(data, tau=bins * 1000) for data in results[:2]]
    delta_rhythmicity = (
        rhythmicity_score[1] - rhythmicity_score[0]
    ) / rhythmicity_score[0]
    ax[4].violinplot(
        delta_rhythmicity, positions=[0], showmeans=False, showextrema=False
    )
    ax[4].set_ylabel("Rhythmicity (test-control)/control")
    ax[4].set_xticks([])
    # periodicity_results.append(rhythmicity_score)
    periodicity_results.append(delta_rhythmicity)

    # label and clean up axes
    y = np.median(results[1].T, axis=1)
    ax[2].set_ylim(y.min() * 0.3, y.max() * 1.1)
    # ax[2].set_yscale("log")

    ax[1].set_yticks([])
    ax[0].set_title("Optogenetic Control")
    ax[1].set_title("Optogenetic Test")
    ax[0].set_ylabel("Unit #")
    ax[2].set_ylabel("AutoCorrellegram")
    for a in ax[:3]:
        a.set_xlabel("Time (s)")
    for a in ax[2:4]:
        a.spines[["top", "right"]].set_visible(False)
    ax[3].set_xticks([0, 1], labels=["control", "test"])
    ax[3].set_ylabel("Periodicity (s)")

    # table of experiment information
    the_table = ax[-1].table(
        cellText=[[len(dataset)]] + [[str(x)] for x in dataset_key.values()],
        rowLabels=["number_epochs"] + [str(x) for x in dataset_key.keys()],
        loc="right",
        colWidths=[0.6, 0.6],
    )
    ax[-1].spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])

    plt.rcParams["svg.fonttype"] = "none"
    fig.suptitle(f"{dataset_key['animal']}: {period}ms opto stim")
    if return_periodicity_results:
        if return_auto_corr:
            return fig, periodicity_results, results
        return fig, periodicity_results
    return fig


from scipy.signal import find_peaks


def estimate_periodicity_timescale(autocorr, time, width=0.01):
    """
    Estimates the timescale of periodic peaks in the autocorrelation function.

    :param autocorr: NumPy array containing the autocorrelation values.
    :param time: NumPy array containing the time values corresponding to the autocorr data.

    :return: Estimated timescale of the periodicity.

    """
    # translate width from seconds to indices
    width = int(width / np.mean(np.diff(time)))

    # Find peaks in the autocorrelation function
    peaks, _ = find_peaks(autocorr, width=width)

    # Calculate the times of the peaks
    peak_times = time[peaks]

    # return peak_times
    # Calculate the differences between successive peak times
    peak_intervals = np.diff(peak_times)

    # Estimate the timescale as the mean of the intervals
    timescale = np.mean(peak_intervals)

    return timescale


def plot_periodicity_dependence(periodicities, driving_periods):
    """Function to summary plot of periodicity dependence on driving frequency

    Args:
        periodicities: list of periodicity outputs from autocorrelegram()
        driving_periods : list of driving periods

    Returns:
       fig: subplot figure of results
    """
    periods = driving_periods

    fig, ax = plt.subplots(nrows=2, sharex=True)
    for cond, a in enumerate(ax):
        for n, (period, periodicity) in enumerate(zip(periods, periodicities)):
            for i, color in enumerate(["cornflowerblue", "firebrick"]):
                violin = a.violinplot(
                    periodicity[i] - cond * (0.125 if i == 0 else period / 1000),
                    positions=[n + i * 0.3],
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                    widths=0.3,
                    points=100,
                    bw_method=0.5,
                    vert=True,
                )
                for pc in violin["bodies"]:
                    pc.set_facecolor(color)
                    # pc.set_edgecolor('black')
                    pc.set_alpha(0.3)
                violin["cmedians"].set_color(color)

                # show comparison to the half frequency
                if period < 99 and i == 1 and cond == 1:
                    violin = a.violinplot(
                        periodicity[i] - (2 * period / 1000),
                        positions=[n + i * 0.3],
                        showmeans=False,
                        showmedians=True,
                        showextrema=False,
                        widths=0.3,
                        points=100,
                        bw_method=0.5,
                        vert=True,
                    )
                    for pc in violin["bodies"]:
                        pc.set_facecolor("grey")
                        pc.set_alpha(0.3)
                    violin["cmedians"].set_color("grey")
                # show comparison to the double frequency
                if period > 130 and i == 1 and cond == 1:
                    violin = a.violinplot(
                        periodicity[i] - (period / 1000 / 2),
                        positions=[n + i * 0.3],
                        showmeans=False,
                        showmedians=True,
                        showextrema=False,
                        widths=0.3,
                        points=100,
                        bw_method=0.5,
                        vert=True,
                    )
                    for pc in violin["bodies"]:
                        pc.set_facecolor("grey")
                        pc.set_alpha(0.3)
                    violin["cmedians"].set_color("grey")

    ax[1].set_xticks(range(len(periods)), periods)
    ax[1].hlines(0, -0.4, len(periods) - 0.1, linestyle="--", color="k", alpha=0.3)
    ax[0].spines[["top", "right"]].set_visible(False)
    ax[1].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylim(0.05, 0.2)
    ax[1].set_ylim([-0.1, 0.1])
    ax[0].set_ylabel("Periodicity")
    ax[1].set_ylabel("Periodicity - Reference Frequency")
    ax[1].set_xlabel("Stimulation Period (ms)")
    return fig


def rhythmicity(data, tau=None, window=10):
    """calculates rhythmicity of autocorrelogram
    Uses method from: "The medial septum controls hippocampal supra-theta oscillations" Nature Comm. 2023

    Parameters
    ----------
    data : np.array
        autocorrelogram, shape (n_neurons, n_bins)
    tau : _type_, optional
        lag times, by default None assumes 1ms bins
    window : int, optional
        window to average for peak and trough values, by default 10

    Returns
    -------
    rhythmicity score: np.array
        shape (n_neurons,)
    """

    if tau is None:
        tau = np.arange(data.shape[1])  # assume 1ms bins
    rhythmicity = []
    for x in data:
        peak = np.argmax(x[np.logical_and(tau > 50, tau < 200)]) + 50
        hi_val = np.mean(x[peak - window : peak + window])
        lo_vals = [
            np.mean(
                x[
                    np.logical_and(
                        tau > int(peak * 1.5) - window, tau < int(peak * 1.5 + window)
                    )
                ]
            ),
            np.mean(
                x[
                    np.logical_and(
                        tau > int(peak * 0.5) - window, tau < int(peak * 0.5 + window)
                    )
                ]
            ),
        ]
        lo_vals = np.array(lo_vals)

        rhythmicity.append(np.mean((hi_val - lo_vals) / hi_val))
    return np.array(rhythmicity)
