from spyglass.common import convert_epoch_interval_name_to_position_interval_name
from spyglass.spikesorting import CuratedSpikeSorting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spyglass.common import interval_list_contains, PositionIntervalMap, TaskEpoch

import os

import spyglass.spikesorting as sgs
from spyglass.spikesorting.merge import SpikeSortingOutput
from spyglass.decoding.v1.sorted_spikes import SortedSpikesGroup
from spyglass.decoding.v1.clusterless import PositionGroup
from spyglass.position import PositionOutput
from spyglass.decoding.v1.sorted_spikes import (
    SortedSpikesDecodingV1,
    SortedSpikesDecodingSelection,
)

os.chdir("/home/sambray/Documents/MS_analysis_samsplaying/")
from ms_opto_stim_protocol import OptoStimProtocol
from Analysis.spiking_analysis import smooth
from Analysis.position_analysis import get_running_intervals
from Analysis.utils import filter_opto_data, get_running_valid_intervals


def decoding_place_fields(dataset_key: dict, return_correlations=False):
    # get the matching epochs
    dataset = filter_opto_data(dataset_key)
    nwb_file_names = dataset.fetch("nwb_file_name")
    pos_interval_names = dataset.fetch("interval_list_name")

    # get the autocorrelegrams
    place_field_list = [[], []]
    mean_rates_list = [[], []]
    for nwb_file_name, pos_interval in zip(nwb_file_names, pos_interval_names):
        interval_name = (
            (
                PositionIntervalMap()
                & {"nwb_file_name": nwb_file_name}
                & {"position_interval_name": pos_interval}
            )
            * TaskEpoch
        ).fetch1("interval_list_name")
        key = {
            "nwb_file_name": nwb_file_name,
            "sorted_spikes_group_name": interval_name,
            "position_group_name": pos_interval,
        }
        if not len(SortedSpikesDecodingV1 & key) == 3:
            continue
        for i, encoding in enumerate(["_opto_control_interval", "_opto_test_interval"]):
            fit_model = (
                SortedSpikesDecodingV1
                & key
                & {"encoding_interval": pos_interval + encoding}
            ).load_model()

            place_field = list(fit_model.encoding_model_.values())[0]["place_fields"]
            norm_place_field = place_field / np.sum(place_field, axis=1, keepdims=True)
            place_field_list[i].extend(norm_place_field)

            encode_interval = (
                SortedSpikesDecodingV1
                & key
                & {"encoding_interval": pos_interval + encoding}
            ).fetch1("encoding_interval")
            from spyglass.common import IntervalList

            encode_times = (
                IntervalList & key & {"interval_list_name": encode_interval}
            ).fetch1("valid_times")
            n_bins = (
                np.sum([x[1] - x[0] for x in encode_times]) * 500
            )  # TODO: don't hardcode sampling rate
            mean_rates_list[i].extend(
                np.array(list(fit_model.encoding_model_.values())[0]["mean_rates"])
                * n_bins
            )
    # only consider units with a minimum number of events in each condition
    min_rate = 100
    ind_valid = np.logical_and(
        np.array(mean_rates_list[0]) > min_rate, np.array(mean_rates_list[1]) > min_rate
    )
    place_field_list = [np.array(x)[ind_valid] for x in place_field_list]
    # only consider units with a mimum of place specificity in the control condition #TODO: consider other measures of specificity
    min_peak = 2 / place_field_list[0][0].size
    print("min_peak", min_peak)
    ind_valid = np.max(place_field_list[0], axis=1) > min_peak
    place_field_list = [np.array(x)[ind_valid] for x in place_field_list]
    place_field_list = np.array(place_field_list)
    if place_field_list[0].size == 0:
        return
    ind_sort = np.argsort(np.argmax(place_field_list[0], axis=1))

    # plot the results
    fig, ax = plt.subplots(1, 4, figsize=(18, 5), width_ratios=[1, 1, 1, 0.3])
    # heatmap of the place fields
    ax[0].imshow(
        place_field_list[0][ind_sort], aspect="auto", cmap="bone_r", origin="lower"
    )
    ax[0].set_title("Control")
    ax[1].imshow(
        place_field_list[1][ind_sort], aspect="auto", cmap="bone_r", origin="lower"
    )
    ax[1].set_title("Test")
    ax[0].set_xlabel("Position (cm)")
    ax[1].set_xlabel("Position (cm)")
    ax[0].set_ylabel("Cell #")
    ax[1].set_yticks([])
    # correlation of placefields between conditions
    var_list = [x - x.mean(axis=1, keepdims=True) for x in place_field_list]
    var_list = [x / np.linalg.norm(x, axis=1, keepdims=True) for x in var_list]
    cond_correlation = (var_list[0] * var_list[1]).sum(axis=1)
    ax[2].violinplot(cond_correlation, showextrema=False)
    ax[2].scatter(
        np.random.normal(1, 0.04, size=len(cond_correlation)),
        cond_correlation,
        alpha=0.3,
    )
    ax[2].set_ylabel(
        "Correlation ($place\ field_{\ control}$, $place\ field_{\ test}$)"
    )
    ax[2].set_xticks([])
    ax[2].spines[["top", "right", "bottom"]].set_visible(False)
    # table of experiment information
    the_table = ax[3].table(
        cellText=[[len(dataset)]] + [[str(x)] for x in dataset_key.values()],
        rowLabels=["number_epochs"] + [str(x) for x in dataset_key.keys()],
        loc="right",
        colWidths=[0.6, 0.6],
    )
    ax[3].spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    # title
    plt.rcParams["svg.fonttype"] = "none"
    if "period_ms" in dataset_key:
        period = dataset_key["period_ms"]
        fig.suptitle(f"{dataset_key['animal']}: {period}ms opto stim")
    elif "targeted_phase" in dataset_key:
        phase = dataset_key["targeted_phase"]
        fig.suptitle(f"{dataset_key['animal']}: {phase} phase opto stim")
