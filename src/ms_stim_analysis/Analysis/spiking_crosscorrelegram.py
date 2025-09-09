import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from spyglass.common import (
    PositionIntervalMap,
    TaskEpoch,
    interval_list_contains,
    interval_list_contains_ind,
    interval_list_intersect,
)
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1
from spyglass.spikesorting.v0 import CuratedSpikeSorting
from tqdm import tqdm

from .spiking_analysis import smooth
from .utils import filter_opto_data, get_running_valid_intervals, violin_scatter
from ms_stim_analysis.AnalysisTables.ms_opto_stim_protocol import OptoStimProtocol
from ms_stim_analysis.Style.style_guide import interval_style


def overlap_place_fields_crosscorrelegram(
    dataset_key: dict,
    max_place_distance: int = 8,
    min_place_distance: int = 1,
    time_window: float = 0.2,
    gauss_smooth: float = 0.010,
    min_spikes: int = 1000,
    closest_spike_only=False,
):
    """get the cross correlegram of pairs of units that have place fields that
    are a certain distance apart

    Args:
        dataset_key (dict): restriction of datasets to include
        max_place_distance (int, optional): min number of bins place fields are to include. Defaults to 8.
        min_place_distance (int, optional): max number of bins place fields are to include. Defaults to 1.
        time_window (float, optional): +- window size of correlegram, seconds. Defaults to .2.
        gauss_smooth (float, optional): sigma of gauss smooth filter, seconds. Defaults to .010.
        min_spikes (int, optional): minimum number of spikes on a unit to be included. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    raise Warning(
        "Not used in final publication analyses in favor of peak-delay quantification"
    )

    full_day_sort = False
    # get the matching epochs
    dataset = filter_opto_data(dataset_key)
    nwb_file_name_list = dataset.fetch("nwb_file_name", limit=100)
    interval_list = dataset.fetch("interval_list_name")

    results = [[], []]
    time = []
    rates = [[], []]
    distances = []
    # loop through epochs
    for nwb_file_name, interval in tqdm(zip(nwb_file_name_list, interval_list)):

        sort_interval = (
            (
                PositionIntervalMap()
                & {"nwb_file_name": nwb_file_name}
                & {"position_interval_name": interval}
            )
            * TaskEpoch
        ).fetch1("interval_list_name")
        if full_day_sort:
            sort_interval = "manual_full_day"

        key = {
            "nwb_file_name": nwb_file_name,
            "encoding_interval": interval,
            "sorted_spikes_group_name": sort_interval,
            "position_group_name": interval,
        }
        if not SortedSpikesDecodingV1() & key:
            continue
        # get the place field locations
        result = (SortedSpikesDecodingV1() & key).fetch_model()
        for k in list(result.encoding_model_.keys()):
            print(k)
            place_fields = result.encoding_model_[k]["place_fields"]
            print("\n")
        # get the pairs of units that are correct distance apart
        included_pairs = []
        pair_distance = []
        place_peak = np.argmax(place_fields, axis=1)
        for i in range(place_fields.shape[0]):
            for j in range(place_fields.shape[0]):
                # if j >= i:
                #     break
                # if np.abs(place_peak[i] - place_peak[j]) < 10:
                if np.allclose(place_peak[i], place_peak[j]):
                    continue
                if (
                    np.abs(place_peak[i] - place_peak[j]) <= max_place_distance
                    and np.abs(place_peak[i] - place_peak[j]) >= min_place_distance
                ):
                    included_pairs.append((i, j))
                    pair_distance.append(place_peak[i] - place_peak[j])

        # get the cross correlegram of these pairs in this epoch
        opto_key = {"nwb_file_name": nwb_file_name, "interval_list_name": interval}
        results_, time_, rates_, pair_ids = crosscorrelegram(
            opto_key,
            min_spikes=min_spikes,
            gauss_smooth=gauss_smooth,
            window=time_window,
            analyze_pairs=included_pairs,
            closest_spike_only=closest_spike_only,
            return_ids=True,
        )
        time = time_
        for i in range(2):
            results[i].extend(results_[i])
            rates[i].extend(rates_[i])
        distances.extend([place_peak[i] - place_peak[j] for (i, j) in pair_ids])

    results = [np.array(results[0]), np.array(results[1])]
    rates = [np.array(rates[0]), np.array(rates[1])]

    fig, ax = plt.subplots(1, 3, figsize=(10, 5), width_ratios=[2, 2, 1])
    ax_rhythm = ax[0]
    # Calculate rhythmicity
    peak_width = int(0.1 / np.diff(time).mean())  # look for peaks that are 100ms wide
    # peak_width = int(0.05 / np.diff(time).mean())
    avg_window = int(
        0.01 / np.diff(time).mean()
    )  # take the average of the 10ms around the peak
    rhythmicity = [
        np.array(
            [cross_correlation_rhythmicity(x, peak_width, avg_window) for x in data]
        )
        for data in results
    ]
    ind_rhythmic = np.where(rhythmicity[0] > 0.05)[0]

    rhythmicity_plot = [data[ind_rhythmic] for data in rhythmicity]
    for i, (data, label) in enumerate(zip(rhythmicity_plot, ["control", "test"])):
        violin = ax_rhythm.violinplot(
            data, positions=[i], showmedians=False, showextrema=False
        )
        violin["bodies"][0].set_facecolor(interval_style[label])
        ax_rhythm.scatter([i] * len(data), data, alpha=0.5, color=interval_style[label])
    for x1, x2 in zip(rhythmicity_plot[0], rhythmicity_plot[1]):
        ax_rhythm.plot([0, 1], [x1, x2], "grey", alpha=0.5, lw=0.3, zorder=-1)
    ax_rhythm.set_xticks([0, 1], ["control", "test"])
    ax_rhythm.set_title("crosscorrelegram rhythmicity")
    # ax_rhythm.set_title(dataset_key)

    # Calculate and plot peak cross correlegram times
    ax_peak = ax[1]
    window = 0.05
    peak_window = np.logical_and(time > -1 * window, time < window)[:-1]

    # # regress out the linear trend in the peak time
    # def remove_linear_trend(data):
    #     x = np.arange(data.shape[1])
    #     for i in range(len(data)):
    #         data[i] = data[i] - np.polyval(np.polyfit(x, data[i], 1), x)
    #     return data

    # peak_time = [
    #     time[:-1][peak_window][
    #         remove_linear_trend(data[:, peak_window].copy()).argmax(axis=1)
    #     ]
    #     for data in results
    # ]

    # peak_time = [time[:-1][data.argmax(axis=1)] for data in results]

    filtered_corr = [boxcar_filter_set(data.T, 100, "divide").T for data in results]
    # peak_time = [time[:-1][peak_window][data[:,peak_window].argmax(axis=1)] for data in filtered_corr]

    all_peaks = [
        [find_peaks(x, distance=peak_width)[0] for x in data] for data in filtered_corr
    ]

    def closest_peak(x):
        if len(x):
            return time[x[np.argmin(np.abs(time[x]))]]
        else:
            return 0

    peak_time = [[closest_peak(x) for x in data] for data in all_peaks]

    x_pos = []
    for i, (data, interval) in enumerate(zip(peak_time, ["control", "test"])):
        x_pos.append(
            violin_scatter(
                data,
                pos=i,
                color=interval_style[interval],
                bw_method=0.1,
                ax=ax_peak,
                return_locs=True,
            )[0]
        )
    for i in range(len(x_pos[0])):
        ax_peak.plot(
            [x_pos[0][i], x_pos[1][i]],
            [peak_time[0][i], peak_time[1][i]],
            "grey",
            alpha=0.5,
            lw=0.3,
            zorder=-1,
        )

    ax_peak.set_xticks([0, 1], ["control", "test"])
    ax_peak.set_title("peak autocorrelegram time (s)")

    # table of experiment information
    the_table = ax[-1].table(
        cellText=[[len(dataset)], [min_place_distance], [max_place_distance]]
        + [[str(x)] for x in dataset_key.values()],
        rowLabels=["number_epochs", "min_place_distance", "max_place_distance"]
        + [str(x) for x in dataset_key.keys()],
        loc="right",
        colWidths=[0.6, 0.6],
    )
    ax[-1].spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])

    return results, time, rates, rhythmicity, peak_time, distances


def crosscorrelegram(
    dataset_key: dict,
    filter_speed: float = 10,
    min_spikes: int = 300,
    min_run_time: float = 0.5,
    return_periodicity_results: bool = False,
    gauss_smooth: float = 0.003,
    analyze_pairs: list = None,
    window: float = 0.1,
    closest_spike_only=False,
    return_ids=False,
):
    """Returns the cross correlegram of spike times between unit pairs in a dataset

    Args:
        dataset_key (dict): restriction of datasets to include
        filter_speed (float, optional): Only consider times with speeds above this. Defaults to 10.
        min_spikes (int, optional): Only consider units with at least this many spikes. Defaults to 300.
        min_run_time (float, optional): Minimum time rat above filter speed to include an interval. Defaults to 0.5.
        return_periodicity_results (bool, optional): return periodicity estimates. Defaults to False.
        gauss_smooth (float, optional): sigma of gausian smoothing, seconds. Defaults to 0.003.
        analyze_pairs (list, optional): If provided, only run analysis on pairs of neurons listed here. Defaults to None.
        window (float, optional): +-time window to analyze correlagram. Defaults to 0.1.

    Returns:
        results, histogram_bins, rates: cross correlegrams (Hz), bins, firing rates (of individual neuron, Hz)
    """
    full_day_sort = False

    # get the matching epochs
    dataset = filter_opto_data(dataset_key)
    if len(dataset) > 1 and analyze_pairs is not None:
        raise ValueError(
            "If analyzing specific pairs, must restrict to one dataset at a time"
        )
    nwb_file_names = dataset.fetch("nwb_file_name")
    pos_interval_names = dataset.fetch("interval_list_name")

    # get the cross-correlegrams
    results = [[], []]
    rates = [[], []]
    pair_id = []

    if len(nwb_file_names) > 1:
        print(
            "WARNING: more than one nwb file found for this dataset. Only the first will be used."
        )
        print(
            "Full list: ", [(a, b) for a, b in zip(nwb_file_names, pos_interval_names)]
        )
    nwb_file_name = nwb_file_names[0]
    pos_interval = pos_interval_names[0]
    print(f"Processing {nwb_file_name} with {pos_interval}")

    ###########
    ###########
    ###########

    for nwb_file_name, pos_interval in zip(nwb_file_names, pos_interval_names):

        interval_name = (
            (
                PositionIntervalMap()
                & {"nwb_file_name": nwb_file_name}
                & {"position_interval_name": pos_interval}
            )
            * TaskEpoch
        ).fetch1("interval_list_name")
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
        from spyglass.common import IntervalList

        # test_interval = (
        #     IntervalList
        #     & {
        #         "nwb_file_name": nwb_file_name,
        #         "interval_list_name": pos_interval + "_stimulus_on_interval",
        #     }
        # ).fetch1("valid_times")

        spike_df = []
        decode_key = {
            "nwb_file_name": nwb_file_name,
            "encoding_interval": pos_interval,
            "sorted_spikes_group_name": interval_name,
            "position_group_name": pos_interval,
        }
        decode_key = (SortedSpikesDecodingV1 & decode_key).fetch1("KEY")
        spike_df = SortedSpikesDecodingV1().fetch_spike_data(decode_key)

        # define what intervals to use
        run_intervals = get_running_valid_intervals(
            pos_key, seperate_optogenetics=False, filter_speed=filter_speed
        )
        run_intervals = [
            interval
            for interval in run_intervals
            if interval[1] - interval[0] > min_run_time
        ]

        histogram_bins = np.arange(-window, window, 0.0005)
        print("number_units", len(spike_df))
        # loop through unit pairs
        for n_s1, spikes in enumerate(spike_df):
            spikes = interval_list_contains(
                run_intervals,
                spikes,
            )
            if spikes.size < min_spikes:
                continue

            valid_bin_count = []
            # print("spikes", spikes.size)
            for interval in [control_interval, test_interval]:
                bins = histogram_bins[:-1] + np.diff(histogram_bins) / 2
                x = interval_list_contains(interval, spikes)
                absolute_bin_times = np.add.outer(x, bins).ravel()
                absolute_bin_index = np.array(
                    [np.arange(bins.size) for _ in range(x.size)]
                ).ravel()
                valid_interval = interval_list_intersect(
                    np.array(interval), np.array(run_intervals)
                )
                valid_interval = interval
                valid_bin_index = absolute_bin_index[
                    interval_list_contains_ind(valid_interval, absolute_bin_times)
                ]
                valid_bin_count.append(
                    np.bincount(valid_bin_index, minlength=bins.size)
                )

            repeat_count = np.zeros(histogram_bins.size - 1)
            # bin_centers = histogram_bins[:-1] + np.diff(histogram_bins) / 2
            for n_s2, spikes_2 in enumerate(spike_df):
                # skip if have a specific list of pairs and this pair is not in it
                if (
                    (analyze_pairs is not None)
                    and (n_s1, n_s2) not in analyze_pairs
                    and (n_s2, n_s1) not in analyze_pairs
                ):
                    continue
                # skip auto correlegrams
                if n_s1 == n_s2:
                    continue
                spikes_2 = interval_list_contains(
                    run_intervals,
                    spikes_2,
                )
                if spikes_2.size < min_spikes:
                    continue

                for i, interval in enumerate(
                    [
                        control_interval,
                        test_interval,
                    ]
                ):
                    x = interval_list_contains(interval, spikes)
                    x2 = interval_list_contains(interval, spikes_2)

                    # get the correlegram count
                    delays = np.subtract.outer(x, x2)
                    if closest_spike_only:
                        closest_spike = np.argmin(np.abs(delays), axis=1)
                        delays = np.array(
                            [
                                delays[ii, closest_spike[ii]]
                                for ii in range(closest_spike.size)
                            ]
                        )
                    delays = np.ravel(delays)
                    delays = delays[delays < histogram_bins[-1]]
                    delays = delays[delays >= histogram_bins[0]]
                    vals, bins = np.histogram(delays, bins=histogram_bins)
                    # vals = vals + 1e-1
                    if gauss_smooth:
                        sigma = int(
                            gauss_smooth / np.mean(np.diff(histogram_bins))
                        )  # turn gauss_smooth from seconds to bins
                        # print(sigma)
                        vals = smooth(vals, 3 * sigma, sigma)
                    bins = bins[:-1] + np.diff(bins) / 2

                    # vals = vals / vals.sum() # normalize
                    vals = vals / (
                        np.diff(bins).mean() * valid_bin_count[i]
                    )  # normalize into rate (Hz)

                    results[i].append(vals)
                    rates[i].append(
                        x2.size / np.sum([e - s for s, e in interval])
                    )  # overall rate of the neuron
                pair_id.append([n_s1, n_s2])

    results = [np.array(r) for r in results]
    rates = [np.array(r) for r in rates]

    if return_ids:
        return results, histogram_bins, rates, pair_id
    return results, histogram_bins, rates


def crosscorrelegram_stimulus_only(
    dataset_key: dict,
    filter_speed: float = 10,
    min_spikes: int = 300,
    min_run_time: float = 0.5,
    return_periodicity_results: bool = False,
    gauss_smooth: float = 0.003,
):
    # get the matching epochs
    dataset = filter_opto_data(dataset_key)
    nwb_file_names = dataset.fetch("nwb_file_name")
    pos_interval_names = dataset.fetch("interval_list_name")

    # get the autocorrelegrams
    results = [[], []]
    rates = [[], []]

    if len(nwb_file_names) > 1:
        print(
            "WARNING: more than one nwb file found for this dataset. Only the first will be used."
        )
        print(
            "Full list: ", [(a, b) for a, b in zip(nwb_file_names, pos_interval_names)]
        )
    nwb_file_name = nwb_file_names[0]
    pos_interval = pos_interval_names[0]
    print(f"Processing {nwb_file_name} with {pos_interval}")

    ###########
    ###########
    ###########

    for nwb_file_name, pos_interval in zip(nwb_file_names, pos_interval_names):

        interval_name = (
            (
                PositionIntervalMap()
                & {"nwb_file_name": nwb_file_name}
                & {"position_interval_name": pos_interval}
            )
            * TaskEpoch
        ).fetch1("interval_list_name")
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
        spike_df = []
        for sort_group in set(
            (CuratedSpikeSorting() & basic_key).fetch("sort_group_id")
        ):
            key = {"sort_group_id": sort_group}
            cur_id = np.max(
                (CuratedSpikeSorting() & basic_key & key).fetch("curation_id")
            )
            key["curation_id"] = cur_id
            cur_id = 1

            tetrode_df = (CuratedSpikeSorting & basic_key & key).fetch_nwb()[0]
            if "units" in tetrode_df:
                tetrode_df = tetrode_df["units"]
                tetrode_df = tetrode_df[tetrode_df.label == ""]
                spike_df.append(tetrode_df)
        if len(spike_df) == 0:
            continue
        spike_df = pd.concat(spike_df)

        run_intervals = get_running_valid_intervals(
            pos_key, seperate_optogenetics=False, filter_speed=filter_speed
        )
        run_intervals = [
            interval
            for interval in run_intervals
            if interval[1] - interval[0] > min_run_time
        ]
        # restrict to stimulus only
        stim, stim_time = OptoStimProtocol().get_stimulus(pos_key)
        t_on = stim_time[stim == 1]
        t_off = stim_time[stim == 0][1:]
        stimulating_intervals = np.array(
            [[s - 0.02, e + 0.02] for s, e in zip(t_on, t_off)]
        )
        run_intervals_stim_only = interval_list_intersect(
            np.array(run_intervals), stimulating_intervals
        )

        histogram_bins = np.arange(-0.1, 0.1, 0.0001)
        print("number_units", len(spike_df.spike_times.values))

        for n_s1, spikes in enumerate(spike_df.spike_times.values):
            spikes = interval_list_contains(
                run_intervals_stim_only,
                spikes,
            )
            if spikes.size < min_spikes:
                continue

            for n_s2, spikes_2 in enumerate(spike_df.spike_times.values):
                if n_s1 == n_s2:
                    continue
                spikes_2 = interval_list_contains(
                    run_intervals_stim_only,
                    spikes_2,
                )
                if spikes_2.size < min_spikes:
                    continue

                for i, interval in enumerate(
                    [
                        control_interval,
                        test_interval,
                    ]
                ):
                    x = interval_list_contains(interval, spikes)
                    x2 = interval_list_contains(interval, spikes_2)

                    delays = np.subtract.outer(x, x2)
                    delays = delays[delays < histogram_bins[-1]]
                    delays = delays[delays >= histogram_bins[0]]
                    vals, bins = np.histogram(delays, bins=histogram_bins)
                    # if vals.sum() < 100:
                    #     continue
                    vals = vals + 1e-9
                    if gauss_smooth:
                        vals = smooth(
                            vals, int(gauss_smooth / np.mean(np.diff(histogram_bins)))
                        )
                    bins = bins[:-1] + np.diff(bins) / 2
                    vals = vals / vals.sum()
                    results[i].append(vals)
                    rates[i].append(
                        x.size
                        / np.sum(
                            [
                                e - s
                                for s, e in interval_list_intersect(
                                    np.array(interval), run_intervals_stim_only
                                )
                            ]
                        )
                    )

        # stim, stim_time = OptoStimProtocol().get_stimulus(pos_key)
        # stim_time = stim_time[stim == 1]
        # delays = np.subtract.outer(stim_time, stim_time)
        # delays = delays[np.tril_indices_from(delays, k=0)]
        # delays = delays[delays < histogram_bins[-1]]
        # vals, bins = np.histogram(delays, bins=histogram_bins)
        # vals = vals + 1e-9
        # vals = smooth(vals, int(0.015 / np.mean(np.diff(histogram_bins))))
        # bins = bins[:-1] + np.diff(bins) / 2
        # vals = vals / vals.sum()
        # stim_results.append(vals)

    results = [np.array(r) for r in results]
    rates = [np.array(r) for r in rates]
    # stim_results = np.array(stim_results)

    return results, histogram_bins, rates


#############################################################################
# spike lag vs position


def spike_lag_vs_position(
    dataset_key: dict,
    filter_speed: float = 10,
    min_spikes: int = 300,
    min_run_time: float = 0.5,
    return_periodicity_results: bool = False,
    gauss_smooth: float = 0.003,
    analyze_pairs: list = None,
    field_centers: list = None,
):
    assert len(field_centers) == len(analyze_pairs)
    # get the matching epochs
    dataset = filter_opto_data(dataset_key)
    nwb_file_names = dataset.fetch("nwb_file_name")
    pos_interval_names = dataset.fetch("interval_list_name")

    # get the autocorrelegrams
    lags = [[], []]
    delta_positions = [[], []]

    if len(nwb_file_names) > 1:
        print(
            "WARNING: more than one nwb file found for this dataset. Only the first will be used."
        )
        print(
            "Full list: ", [(a, b) for a, b in zip(nwb_file_names, pos_interval_names)]
        )
    nwb_file_name = nwb_file_names[0]
    pos_interval = pos_interval_names[0]
    print(f"Processing {nwb_file_name} with {pos_interval}")

    ###########
    ###########
    ###########

    for nwb_file_name, pos_interval in zip(nwb_file_names, pos_interval_names):

        interval_name = (
            (
                PositionIntervalMap()
                & {"nwb_file_name": nwb_file_name}
                & {"position_interval_name": pos_interval}
            )
            * TaskEpoch
        ).fetch1("interval_list_name")
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
        # # get the spike and position dat for each
        # spike_df = []
        # for sort_group in set(
        #     (CuratedSpikeSorting() & basic_key).fetch("sort_group_id")
        # ):
        #     key = {"sort_group_id": sort_group}
        #     cur_id = np.max(
        #         (CuratedSpikeSorting() & basic_key & key).fetch("curation_id")
        #     )
        #     key["curation_id"] = cur_id
        #     cur_id = 1

        #     tetrode_df = (CuratedSpikeSorting & basic_key & key).fetch_nwb()[0]
        #     if "units" in tetrode_df:
        #         tetrode_df = tetrode_df["units"]
        #         tetrode_df = tetrode_df[tetrode_df.label == ""]
        #         spike_df.append(tetrode_df)
        # if len(spike_df) == 0:
        #     continue
        # spike_df = pd.concat(spike_df)
        # get the spike and position dat for each
        spike_df = []
        decode_key = {
            "nwb_file_name": nwb_file_name,
            "encoding_interval": pos_interval,
        }
        decode_key = (SortedSpikesDecodingV1 & decode_key).fetch1("KEY")
        spike_df = SortedSpikesDecodingV1().fetch_spike_data(decode_key)
        pos_df = SortedSpikesDecodingV1().fetch_linear_position_info(decode_key)

        # define what intervals to use
        run_intervals = get_running_valid_intervals(
            pos_key, seperate_optogenetics=False, filter_speed=filter_speed
        )
        run_intervals = [
            interval
            for interval in run_intervals
            if interval[1] - interval[0] > min_run_time
        ]

        histogram_bins = np.arange(-0.2, 0.2, 0.0005)
        print("number_units", len(spike_df))
        # loop through unit pairs
        for n_s1, spikes in enumerate(spike_df):
            spikes = interval_list_contains(
                run_intervals,
                spikes,
            )
            if spikes.size < min_spikes:
                continue

            for n_s2, spikes_2 in enumerate(spike_df):
                # skip if have a specific list of pairs and this pair is not in it
                if (
                    (analyze_pairs is not None)
                    and (n_s1, n_s2) not in analyze_pairs
                    and (n_s2, n_s1) not in analyze_pairs
                ):
                    continue
                # skip auto correlegrams
                if n_s1 == n_s2:
                    continue
                spikes_2 = interval_list_contains(
                    run_intervals,
                    spikes_2,
                )
                if spikes_2.size < min_spikes:
                    continue

                for i, interval in enumerate(
                    [
                        control_interval,
                        test_interval,
                    ]
                ):
                    x = interval_list_contains(interval, spikes)
                    x2 = interval_list_contains(interval, spikes_2)

                    delays = np.subtract.outer(x, x2)
                    ref_times = np.subtract.outer(x, np.zeros_like(x2))

                    ref_times = ref_times[delays < histogram_bins[-1]]
                    delays = delays[delays < histogram_bins[-1]]
                    ref_times = ref_times[delays >= histogram_bins[0]]
                    delays = delays[delays >= histogram_bins[0]]
                    ref_times = ref_times[delays != 0]
                    delays = delays[delays != 0]

                    # get the position for each spike
                    pos_index = np.digitize(ref_times, pos_df.index)
                    try:
                        ref_ind = analyze_pairs.index((n_s1, n_s2))
                    except:
                        ref_ind = analyze_pairs.index((n_s2, n_s1))
                    ref_place = field_centers[ref_ind]
                    position = pos_df.linear_position.values[pos_index] - ref_place

                    # save the results
                    lags[i].append(delays)
                    delta_positions[i].append(position)

                    # vals, bins = np.histogram(delays, bins=histogram_bins)
                    # # if vals.sum() < 100:
                    # #     continue
                    # vals = vals + 1e-9
                    # if gauss_smooth:
                    #     vals = smooth(
                    #         vals, int(gauss_smooth / np.mean(np.diff(histogram_bins)))
                    #     )
                    # bins = bins[:-1] + np.diff(bins) / 2
                    # vals = vals / vals.sum()
                    # results[i].append(vals)
                    # # rates[i].append(
                    # #     x.size
                    # #     / np.sum(
                    # #         [
                    # #             e - s
                    # #             for s, e in interval_list_intersect(
                    # #                 np.array(interval), np.array(run_intervals)
                    # #             )
                    # #         ]
                    # #     )
                    # # )
                    # rates[i].append(x.size / np.sum([e - s for s, e in interval]))

        # stim, stim_time = OptoStimProtocol().get_stimulus(pos_key)
        # stim_time = stim_time[stim == 1]
        # delays = np.subtract.outer(stim_time, stim_time)
        # delays = delays[np.tril_indices_from(delays, k=0)]
        # delays = delays[delays < histogram_bins[-1]]
        # vals, bins = np.histogram(delays, bins=histogram_bins)
        # vals = vals + 1e-9
        # vals = smooth(vals, int(0.015 / np.mean(np.diff(histogram_bins))))
        # bins = bins[:-1] + np.diff(bins) / 2
        # vals = vals / vals.sum()
        # stim_results.append(vals)

    # lags = [np.array(r) for r in lags]
    # delta_positions = [np.array(r) for r in delta_positions]
    # stim_results = np.array(stim_results)

    return lags, delta_positions


def cross_correlation_rhythmicity(x, width=150, avg_window=10):
    peak_inds = find_peaks(x, distance=width)[0]
    peak_vals = np.nanmean(
        [np.nanmean(x[i - avg_window // 2 : i + avg_window // 2]) for i in peak_inds]
    )
    trough_inds = find_peaks(-x, distance=width)[0]
    # trough_inds = (peak_inds[1:]+peak_inds[:-1])//2

    trough_vals = np.nanmean(
        [np.nanmean(x[i - avg_window // 2 : i + avg_window // 2]) for i in trough_inds]
    )
    rhythmicity = (peak_vals - trough_vals) / peak_vals
    return rhythmicity


# def peak_corr_offset(x, width=150,):


from scipy.signal import welch


def get_fft_rhythmicities(
    cross_correlegrams, lag_times, freq_range=(5, 11), nperseg=None
):
    """
    returns a rhythmicity score based on relative theta-rng power

    Parameters
    ----------
    cross_correlegrams : np.array
        the cross-correlegram
    lag_times : List
        the lag times of cross-correlegram points
    freq_range : tuple, optional
        frequency range from which to extract max power, by default (5,11)
    nperseg : int, optional
        welch fft parameter, by default None

    Returns
    -------
    List
        the rhythmicity scores
    """

    fft_rhythmicity = []
    for x in cross_correlegrams:
        if np.max(x) > 1e9:
            print(x)
            raise
        xx = x / np.nanmax(x)
        fs = 1 / np.mean(np.diff(lag_times))

        # Use Welch's method with zero padding via nfft
        frequencies, power_spectrum = welch(
            xx, fs=fs, nperseg=nperseg, nfft=10000, scaling="density"
        )
        rhythm_power = np.nanmax(
            power_spectrum[
                np.logical_and(frequencies > freq_range[0], frequencies < freq_range[1])
            ]
        )
        fft_rhythmicity.append(rhythm_power)
    print(frequencies)
    return np.array(fft_rhythmicity)  # [np.isfinite(fft_rhythmicity)]


def boxcar_filter(x, size=100, method="subtract"):
    filter = np.ones(size) / size
    if method == "subtract":
        return x - np.convolve(x, filter, mode="same")
    elif method == "divide":
        return x / np.convolve(x, filter, mode="same")
    else:
        raise NotImplementedError(f"method {method} not defined for boxcar filter")


def boxcar_filter_set(data, size, method="subtract"):
    result = []
    for i in range(data.shape[1]):
        result.append(boxcar_filter(data[:, i], size, method))
    return np.array(result).T
