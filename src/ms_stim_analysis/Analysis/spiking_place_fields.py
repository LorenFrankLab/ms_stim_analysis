import numpy as np
import matplotlib.pyplot as plt
from spyglass.common import PositionIntervalMap, TaskEpoch

from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1

from .utils import filter_opto_data, violin_scatter


def decoding_place_fields(
    dataset_key: dict,
    return_correlations=False,
    plot=True,
    return_place_fields=False,
    min_rate=100,
    filter_specificity=True,
    return_place_field_centers=False,
    full_day_sort=False,
    interpolate_field=False,
    dlc_position=False,
):
    # get the matching epochs
    dataset = filter_opto_data(dataset_key)
    nwb_file_names = dataset.fetch("nwb_file_name")
    pos_interval_names = dataset.fetch("interval_list_name")

    # get the autocorrelegrams
    place_field_list = [[], [], []]
    raw_place_field_list = [[], [], []]
    mean_rates_list = [[], [], []]
    information_rates_list = [[], [], []]
    spikes = []
    place_bin_centers = None
    for nwb_file_name, pos_interval in zip(nwb_file_names, pos_interval_names):
        interval_name = (
            (
                PositionIntervalMap()
                & {"nwb_file_name": nwb_file_name}
                & {"position_interval_name": pos_interval}
            )
            * TaskEpoch
        ).fetch1("interval_list_name")
        if full_day_sort:
            sort_interval = "manual_full_day"
        else:
            sort_interval = interval_name
        key = {
            "nwb_file_name": nwb_file_name,
            "sorted_spikes_group_name": sort_interval,
            "position_group_name": (
                f"DLC {pos_interval}" if dlc_position else pos_interval
            ),
        }
        print(key)
        if not len(SortedSpikesDecodingV1 & key) == 4:
            print("Not all decoding intervals are present")
            continue
        for i, encoding in enumerate(
            ["_opto_control_interval", "_opto_test_interval", "_stimulus_on_interval"]
        ):

            decode_key = {**key, "encoding_interval": pos_interval + encoding}
            decode_key = (SortedSpikesDecodingV1 & decode_key).fetch1("KEY")
            if i == 0:
                spikes.extend((SortedSpikesDecodingV1).fetch_spike_data(decode_key))
            fit_model = (
                SortedSpikesDecodingV1
                & key
                & {"encoding_interval": pos_interval + encoding}
            ).fetch_model()

            # get place fields
            place_field = list(fit_model.encoding_model_.values())[0]["place_fields"]
            norm_place_field = place_field / np.sum(place_field, axis=1, keepdims=True)
            place_field_list[i].extend(norm_place_field)
            raw_place_field_list[i].extend(place_field)

            encode_interval = (
                SortedSpikesDecodingV1
                & key
                & {"encoding_interval": pos_interval + encoding}
            ).fetch1("encoding_interval")
            from spyglass.common import IntervalList

            # get mean rates
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
            # get information rates
            encoding = fit_model.encoding_model_
            encoding = encoding[list(encoding.keys())[0]]
            p_loc = encoding["occupancy"]
            p_loc = p_loc / p_loc.sum()
            from ms_stim_analysis.Analysis.spiking_analysis import (
                spatial_information_rate,
            )

            place = list(
                (
                    SortedSpikesDecodingV1
                    & key
                    & {"encoding_interval": pos_interval + "_opto_test_interval"}
                )
                .fetch_model()
                .encoding_model_.values()
            )[0]["environment"].place_bin_centers_
            place = [float(x) for x in place]
            if place_bin_centers is None:
                place_bin_centers = place
            elif not interpolate_field:
                assert np.all(place_bin_centers == place), "Place bins don't match"
    if interpolate_field:
        from scipy.interpolate import interp1d

        x_interp = np.linspace(0, 1, 100)
        for i in range(len(place_field_list)):
            for j in range(len(place_field_list[i])):
                x_dat = np.linspace(0, 1, place_field_list[i][j].size)
                f = interp1d(x_dat, place_field_list[i][j], kind="cubic")
                place_field_list[i][j] = f(x_interp)
                f = interp1d(x_dat, raw_place_field_list[i][j], kind="cubic")
                raw_place_field_list[i][j] = f(x_interp)
        place_bin_centers = x_interp

    # only consider units with a minimum number of events in each condition
    ind_valid = np.logical_and(
        np.array(mean_rates_list[0]) > min_rate, np.array(mean_rates_list[1]) > min_rate
    )
    ind_valid = np.array([len(s) for s in spikes]) > min_rate

    place_field_list = [np.array(x)[ind_valid] for x in place_field_list]
    raw_place_field_list = [np.array(x)[ind_valid] for x in raw_place_field_list]
    # only consider units with a mimum of place specificity in the control condition #TODO: consider other measures of specificity
    if filter_specificity:
        min_peak = 0.05  #
        min_peak = 4 / place_field_list[0][0].size
        # min_peak = 0.005
        print("min_peak", min_peak)
    else:
        min_peak = 0
    ind_valid = np.max(place_field_list[0], axis=1) > min_peak
    place_field_list = [np.array(x)[ind_valid] for x in place_field_list]
    raw_place_field_list = [np.array(x)[ind_valid] for x in raw_place_field_list]
    # place_field_list = np.array(place_field_list)
    # if place_field_list[0].size == 0:
    #     return
    ind_sort = np.argsort(np.argmax(place_field_list[0], axis=1))
    place_field_list = [np.array(x) for x in place_field_list]

    if plot:
        clim = (0, 0.1)
        clim = None
        # plot the results
        fig, ax = plt.subplots(1, 6, figsize=(25, 5), width_ratios=[1, 1, 1, 1, 1, 0.3])
        # heatmap of the place fields
        ax[0].imshow(
            place_field_list[0][ind_sort],
            aspect="auto",
            cmap="bone_r",
            origin="lower",
            clim=clim,
        )
        ax[0].set_title("Control")
        ax[1].imshow(
            place_field_list[1][ind_sort],
            aspect="auto",
            cmap="bone_r",
            origin="lower",
            clim=clim,
        )
        ax[1].set_title("Test")
        ax[2].imshow(
            place_field_list[2][ind_sort],
            aspect="auto",
            cmap="bone_r",
            origin="lower",
            clim=clim,
        )
        ax[2].set_title("Stim-on")

        for a in ax[:3]:
            a.set_xticks(
                np.linspace(0, place_field_list[0].shape[1], 5),
                labels=np.round(
                    np.linspace(place_bin_centers[0], place_bin_centers[-1], 5), 2
                ),
            )

        ax[0].set_xlabel("Position (cm)")
        ax[1].set_xlabel("Position (cm)")
        ax[0].set_ylabel("Cell #")
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        # correlation of placefields between conditions
        var_list = [x - x.mean(axis=1, keepdims=True) for x in place_field_list]
        var_list = [x / np.linalg.norm(x, axis=1, keepdims=True) for x in var_list]
        cond_correlation = (var_list[0] * var_list[1]).sum(axis=1)
        violin_scatter(cond_correlation, ax=ax[-3])
        ax[-3].set_title(
            "Correlation ($place\ field_{\ control}$, $place\ field_{\ test}$)"
        )
        ax[-3].set_xticks([])
        ax[-3].spines[["top", "right", "bottom"]].set_visible(False)

        ax[-2].set_xticks([0, 1, 2], ["control", "test", "stim-on"])

        # table of experiment information
        the_table = ax[3].table(
            cellText=[[len(dataset)]] + [[str(x)] for x in dataset_key.values()],
            rowLabels=["number_epochs"] + [str(x) for x in dataset_key.keys()],
            loc="right",
            colWidths=[0.6, 0.6],
        )
        ax[-1].spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
        # title
        plt.rcParams["svg.fonttype"] = "none"
        if "period_ms" in dataset_key:
            period = dataset_key["period_ms"]
            fig.suptitle(f"{dataset_key['animal']}: {period}ms opto stim")
        elif "targeted_phase" in dataset_key:
            phase = dataset_key["targeted_phase"]
            if "animal" in dataset_key:
                fig.suptitle(f"{dataset_key['animal']}: {phase} phase opto stim")
            elif "transfected" in dataset_key:
                fig.suptitle(
                    f"pooled_transfected={dataset_key['transfected']}: {phase} phase opto stim"
                )

    return_values = []
    if plot:
        return_values.append(fig)
    if return_correlations:
        return_values.append(cond_correlation)
    if return_place_fields:
        return_values.append(place_field_list)
    if return_place_field_centers:
        return_values.append(place_bin_centers)
    return tuple(return_values)
