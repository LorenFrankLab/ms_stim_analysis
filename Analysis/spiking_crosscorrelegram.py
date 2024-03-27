from tqdm import tqdm

from spyglass.common import convert_epoch_interval_name_to_position_interval_name
from spyglass.spikesorting.v0 import CuratedSpikeSorting
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spyglass.common import (
    interval_list_contains,
    interval_list_intersect,
    PositionIntervalMap,
    TaskEpoch,
)


import os
os.chdir("/home/sambray/Documents/MS_analysis_samsplaying/")
from ms_opto_stim_protocol import OptoStimProtocol
from Analysis.spiking_analysis import smooth
from Analysis.position_analysis import get_running_intervals
from Analysis.utils import filter_opto_data, get_running_valid_intervals

def overlap_place_fields_crosscorrelegram(dataset_key:dict,
                                          max_place_distance:int = 8,
                                          min_place_distance:int = 1,
                                          time_window:float = .2,
                                          gauss_smooth:float = .010,
                                          min_spikes:int = 1000):
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
    
    # get the matching epochs
    dataset = filter_opto_data(dataset_key)
    nwb_file_name_list = dataset.fetch("nwb_file_name")
    interval_list = dataset.fetch("interval_list_name")
    
    results = [[],[]]
    time = []
    rates = [[],[]]
    #loop through epochs
    for nwb_file_name,interval in tqdm(zip(nwb_file_name_list,interval_list)):
        key = {"nwb_file_name":nwb_file_name,"encoding_interval":interval}
        if not SortedSpikesDecodingV1() & key:
            continue
        #get the place field locations
        result = (SortedSpikesDecodingV1() & key).fetch_model()
        for k in list(result.encoding_model_.keys()):
            print(k)
            place_fields = result.encoding_model_[k]['place_fields']
            print("\n")
        # get the pairs of units that are correct distance apart
        included_pairs = []
        place_peak = np.argmax(place_fields,axis=1)
        for i in range(place_fields.shape[0]):
            for j in range(place_fields.shape[0]):
                if j >= i:
                    break
                # if np.abs(place_peak[i] - place_peak[j]) < 10:
                if (np.abs(place_peak[i] - place_peak[j]) <= max_place_distance and np.abs(place_peak[i] - place_peak[j]) > min_place_distance):
                    included_pairs.append((i,j))
        # get the cross correlegram of these pairs in this epoch
        opto_key = {"nwb_file_name":nwb_file_name,"interval_list_name":interval}
        results_, time_, rates_  = crosscorrelegram(opto_key,min_spikes = min_spikes,gauss_smooth=gauss_smooth,
                                                    window=time_window,
                                                    analyze_pairs=included_pairs)
        time=time_
        for i in range(2):
            results[i].extend(results_[i])
            rates[i].extend(rates_[i])
            
    results = [np.array(results[0]),np.array(results[1])]
    rates = [np.array(rates[0]),np.array(rates[1])]
    
    return results, time, rates




def crosscorrelegram(
    dataset_key: dict,
    filter_speed: float = 10,
    min_spikes: int = 300,
    min_run_time: float = 0.5,
    return_periodicity_results: bool = False,
    gauss_smooth: float = 0.003,
    analyze_pairs: list = None,
    window: float = 0.1,
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
    # get the matching epochs
    dataset = filter_opto_data(dataset_key)
    if len(dataset) > 1 and analyze_pairs is not None:
        raise ValueError(
            "If analyzing specific pairs, must restrict to one dataset at a time"
        )
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
        # histogram_bins = np.arange(-0.3, 0.3, 0.0005)
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
                    delays = delays[delays < histogram_bins[-1]]
                    delays = delays[delays >= histogram_bins[0]]
                    # delays = delays[delays != 0]
                    vals, bins = np.histogram(delays, bins=histogram_bins)
                    # if vals.sum() < 100:
                    #     continue
                    vals = vals + 1e-9
                    if gauss_smooth:
                        sigma = int(gauss_smooth / np.mean(np.diff(histogram_bins))) #turn gauss_smooth from seconds to bins
                        vals = smooth(vals, 3 * sigma, sigma)
                    bins = bins[:-1] + np.diff(bins) / 2
                    # vals = vals / vals.sum() # normalize
                    vals = vals/(np.diff(bins).mean() *x.size) # normalize into rate (Hz)
                    
                    
                    results[i].append(vals)
                    # rates[i].append(
                    #     x.size
                    #     / np.sum(
                    #         [
                    #             e - s
                    #             for s, e in interval_list_intersect(
                    #                 np.array(interval), np.array(run_intervals)
                    #             )
                    #         ]
                    #     )
                    # )
                    rates[i].append(x.size / np.sum([e - s for s, e in interval])) # overall rate of the neuron

    results = [np.array(r) for r in results]
    rates = [np.array(r) for r in rates]

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
