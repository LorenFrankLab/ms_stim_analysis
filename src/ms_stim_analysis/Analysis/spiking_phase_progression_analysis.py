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
from spyglass.lfp.analysis.v1 import LFPBandV1
from .position_analysis import get_running_intervals, filter_position_ports
from .utils import convert_delta_marks_to_timestamp_values, filter_opto_data


import sys

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
)
from Analysis.circular_shuffle import (
    generate_alligned_binned_spike_func,
    shuffled_spiking_distribution,
    discrete_KL_divergence,
    bootstrap,
    stacked_marks_to_kl,
)

from Analysis.lfp_analysis import get_ref_electrode_index

from spiking_analysis_tables import BinnedSpiking
from spyglass.spikesorting import CuratedSpikeSorting
import scipy.signal


def phase_progression_analysis(
    dataset_key: dict,
    plot_rng: np.ndarray = np.arange(-0.08, 0.2, 0.002),
    first_pulse_only: bool = False,
    band_filter_name: str = "Theta 5-11 Hz",
):
    # define datasets
    dataset = filter_opto_data(dataset_key)

    # loop through datasets and get relevant results
    spike_pos_list = [[], []]
    spike_phase_list = [[], []]
    spike_velocity_list = [[], []]
    crop_rng = [np.nan, np.nan]

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
        band_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": position_interval_name,
            "filter_name": band_filter_name,
        }
        print(basic_key)
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

        ### POSITION ###
        pos_df = (TrodesPosV1() & pos_key).fetch1_dataframe()
        crop_rng[0] = np.nanmin([crop_rng[0], np.nanmin(pos_df.position_x)])
        crop_rng[1] = np.nanmax([crop_rng[1], np.nanmax(pos_df.position_x)])
        pos_time = np.asarray(pos_df.index)
        # determin the position fo each spike
        for i, restrict_interval in enumerate(restrict_interval_list):
            spike_pos = []
            for ii, spikes in tqdm(enumerate(spike_df.spike_times)):
                # find position time bin of each spike
                spikes = interval_list_contains(restrict_interval, spikes)
                spike_ind = np.digitize(
                    spikes,
                    pos_time,
                )
                spike_pos.append(pos_df.position_x.iloc[spike_ind].values)
                spike_velocity_list[i].append(pos_df.velocity_x.iloc[spike_ind].values)
            spike_pos_list[i].extend(spike_pos)

        ### PHASE ###
        # get phase information
        ref_elect, basic_key = get_ref_electrode_index(basic_key)  #
        phase_df = (LFPBandV1() & band_key).compute_signal_phase(
            electrode_list=[ref_elect]
        )
        phase_time = phase_df.index
        phase_ = np.asarray(phase_df)[:, 0]

        # determin the phase for each spike
        for i, restrict_interval in enumerate(restrict_interval_list):
            spike_phase = []
            for spikes in tqdm(spike_df.spike_times):
                # find phase time bin of each spike
                spikes = interval_list_contains(restrict_interval, spikes)
                spikes = interval_list_contains(
                    [[phase_time[0], phase_time[-1]]], spikes
                )
                spike_ind = np.digitize(spikes, phase_time, right=False)
                spike_phase.append((phase_[spike_ind] + np.pi) % (2 * np.pi))
            spike_phase_list[i].extend(spike_phase)

    ### PLOT ###
    # nrows = 10
    # fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(5, 10))
    # crop = 20
    # crop_rng = [crop_rng[0] + crop, crop_rng[1] - crop]

    # ax_count = 0
    # for i in range(len(spike_pos_list[0])):
    #     skip = False
    #     ind_list = []
    #     for interval, color in enumerate(["cornflowerblue", "firebrick"]):
    #         ind = np.where(
    #             np.logical_and(
    #                 np.logical_and(
    #                     spike_pos_list[interval][i] > crop_rng[0],
    #                     spike_pos_list[interval][i] < crop_rng[1],
    #                 ),
    #                 spike_velocity_list[interval][i] < -10,
    #             )
    #         )[0]
    #         if ind.size < 100:
    #             skip = True
    #         ind_list.append(ind)
    #     if skip:
    #         continue
    #     for interval, (ind, color) in enumerate(
    #         zip(ind_list, ["cornflowerblue", "firebrick"])
    #     ):
    #         ax[ax_count].scatter(
    #             spike_pos_list[interval][i][ind],
    #             spike_phase_list[interval][i][ind],
    #             s=10,
    #             alpha=0.3,
    #             color=color,
    #         )
    #     ax[ax_count].set_yticks([0, np.pi, 2 * np.pi])
    #     ax[ax_count].set_yticklabels(["0", "$\pi$", "$2\pi$"])
    #     ax_count += 1
    #     if ax_count >= len(ax):
    #         break

    # ax[-1].set_xlabel("position")
    # ax[-1].set_ylabel("phase")

    ### PLOT heatmap ###
    nrows = 3
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(5, 10), sharey=True)
    crop = 20
    crop_rng = [crop_rng[0] + crop, crop_rng[1] - crop]

    ax_count = 0
    for i in range(len(spike_pos_list[0])):
        skip = False
        ind_list = []
        for interval, color in enumerate(["cornflowerblue", "firebrick"]):
            ind = np.where(
                np.logical_and(
                    np.logical_and(
                        spike_pos_list[interval][i] > crop_rng[0],
                        spike_pos_list[interval][i] < crop_rng[1],
                    ),
                    spike_velocity_list[interval][i] > 10,
                )
            )[0]
            if ind.size < 100:
                skip = True
            ind_list.append(ind)
        if skip:
            continue
        for interval, (ind, color) in enumerate(
            zip(ind_list, ["cornflowerblue", "firebrick"])
        ):
            counts = (
                np.histogram2d(
                    spike_pos_list[interval][i][ind],
                    spike_phase_list[interval][i][ind],
                    bins=(50, 80),
                )[0]
                + 0.01  # laplace smoothing
            )

            counts = scipy.signal.convolve2d(counts, np.ones((10, 10)), mode="same")
            counts = counts / np.sum(counts, axis=0)
            ax[ax_count, interval].imshow(
                counts,
                origin="lower",
                cmap="plasma",
                aspect="auto",
                extent=[crop_rng[0], crop_rng[1], 0, 2 * np.pi],
            )

            # ax[ax_count, interval].scatter(
            #     spike_pos_list[interval][i][ind],
            #     spike_phase_list[interval][i][ind],
            #     s=10,
            #     alpha=0.3,
            #     color=color,
            # )
            ax[ax_count, 0].set_yticks([0, np.pi, 2 * np.pi])
            ax[ax_count, 0].set_yticklabels(["0", "$\pi$", "$2\pi$"])
        ax_count += 1
        if ax_count >= len(ax):
            break
    ax[-1, 0].set_xlabel("position")
    ax[-1, 1].set_xlabel("position")
    ax[-1, 0].set_ylabel("phase")
    # ax[-1].set_xlabel("position")
    # ax[-1].set_ylabel("phase")

    # spike_pos = np.concatenate(spike_pos_list[i])
    # spike_phase = np.concatenate(spike_phase_list[i])
    # # plot the results
    # ax[ax_count].hist2d(spike_pos, spike_phase, bins=plot_rng)
    # ax[ax_count].set_title("spike position vs phase")
    # ax[ax_count].set_xlabel("position")
    # ax[ax_count].set_ylabel("phase")
    # ax_count += 1
    # ax[ax_count].hist(spike_pos, bins=100)
    # ax[ax_count].set_title("spike position")
    # ax_count += 1
    # ax[ax_count].hist(spike_phase, bins=100)
    # ax[ax_count].set_title("spike phase")
    # ax_count += 1
