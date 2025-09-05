import numpy as np
import matplotlib.pyplot as plt
from spyglass.common import (
    get_electrode_indices,
)
from spyglass.lfp.v1 import LFPElectrodeGroup, LFPV1
from spyglass.lfp.lfp_merge import LFPOutput
from spyglass.lfp.analysis.v1 import LFPBandV1
from ms_stim_analysis.AnalysisTables.ms_opto_stim_protocol import OptoStimProtocol
from .utils import filter_opto_data
from .lfp_analysis import get_ref_electrode_index
from ms_stim_analysis.Style.style_guide import animal_style, transfection_style

LFP_AMP_CUTOFF = 2000


def individual_lfp_traces(
    dataset_key: dict,
    filter_name: str = "LFP 0-400 Hz",
    band_filter_name: str = "Theta 5-11 Hz",
    lfp_trace_window=(-int(0.125 * 1000), int(1 * 1000)),
    fig=None,
    color="cornflowerblue",
    n_plot=10,
    electrode_group=None,
):
    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)
    nwb_file_name_list = dataset.fetch("nwb_file_name")
    interval_list_name_list = dataset.fetch("interval_list_name")
    # get the color for display
    if "animal" in dataset_key:
        color = animal_style.loc[dataset_key["animal"]]["color"]
    elif "transfected" in dataset_key:
        if dataset_key["transfected"]:
            color = transfection_style["transfected"]
        else:
            color = transfection_style["control"]
    # get the lfp traces for every relevant pulse
    lfp_traces = []
    marks = []
    band_traces = []
    for nwb_file_name, interval_list_name in zip(
        nwb_file_name_list, interval_list_name_list
    ):
        basic_key = {
            "nwb_file_name": nwb_file_name,
            "target_interval_list_name": interval_list_name,
            "filter_name": filter_name,
        }
        stim_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": interval_list_name,
            "dio_event_name": "stim",
        }
        if len(LFPV1() & basic_key) == 0:
            print("missing LFP for: ", basic_key)
            continue
        print(basic_key)
        # get lfp band phase for reference electrode
        ref_elect, basic_key = get_ref_electrode_index(basic_key)  #
        if electrode_group is not None:
            ref_elect = (
                LFPElectrodeGroup.LFPElectrode
                & basic_key
                & {"electrode_group_name": electrode_group}
            ).fetch("electrode_id")[0]

        # ref_elect = (Electrode() & basic_key).fetch("original_reference_electrode")[0]
        lfp_eseries = LFPOutput().fetch_nwb(restriction=basic_key)[0]["lfp"]
        ref_index = get_electrode_indices(lfp_eseries, [ref_elect])

        # get LFP series
        lfp_df = (LFPV1() & basic_key).fetch_nwb()[0]["lfp"]
        lfp_df = (LFPV1() & basic_key).fetch1_dataframe()
        lfp_timestamps = lfp_df.index
        lfp_ = np.array(lfp_df[ref_index])

        ind = np.sort(np.unique(lfp_timestamps, return_index=True)[1])
        lfp_timestamps = lfp_timestamps[ind]
        lfp_ = lfp_[ind].astype(float)
        # # nan out artifact intervals
        # artifact_times = (LFPArtifactDetection() & basic_key).fetch1("artifact_times")
        # for artifact in artifact_times:
        #     lfp_[
        #         np.logical_and(
        #             lfp_timestamps > artifact[0], lfp_timestamps < artifact[1]
        #         )
        #     ] = np.nan
        try:
            assert np.all(np.diff(lfp_timestamps) > 0)
        except:
            continue
        if band_filter_name is not None:
            band_key = basic_key.copy()
            band_key["filter_name"] = "Theta 5-11 Hz"
            band_key = (LFPBandV1() & band_key).fetch("KEY")[
                0
            ]  # account for different artifact filters in database
            band_df = (LFPBandV1() & band_key).fetch1_dataframe()
            band_ = np.array(band_df[ref_index])
            band_timestamps = band_df.index
            time_ratio = (
                np.diff(lfp_timestamps).mean() / np.diff(band_timestamps).mean()
            )
            band_trace_window = [int(i * time_ratio) for i in lfp_trace_window]

        # get stim times
        t_mark_cycle = OptoStimProtocol().get_cylcle_begin_timepoints(stim_key)
        ind = np.digitize(t_mark_cycle, lfp_timestamps)

        stim, t_mark = OptoStimProtocol().get_stimulus(stim_key)
        t_mark = t_mark[stim == 1]
        ind_mark = np.digitize(t_mark, lfp_timestamps)

        for i in ind:
            lfp_traces.append(lfp_[i + lfp_trace_window[0] : i + lfp_trace_window[1]])
            marks.append(
                ind_mark[
                    (ind_mark >= i + lfp_trace_window[0])
                    & (ind_mark < i + lfp_trace_window[1])
                ]
                - (i + lfp_trace_window[0])
            )

        if band_filter_name is not None:
            ind_band = np.digitize(t_mark_cycle, band_timestamps)
            for i in ind_band:
                band_traces.append(
                    band_[i + band_trace_window[0] : i + band_trace_window[1]]
                )

        if len(lfp_traces) > 100:
            break

    fig, ax = plt.subplots(nrows=n_plot, figsize=(10, n_plot), sharex=True, sharey=True)
    tp = np.linspace(lfp_trace_window[0], lfp_trace_window[1], lfp_traces[0].shape[0])
    np.random.seed(0)
    sampled = np.random.randint(0, len(lfp_traces), len(ax))
    for i, a in zip(sampled, ax):
        a.plot(tp, lfp_traces[i], color=color)
        a.spines[["top", "right", "bottom"]].set_visible(False)
        # if "period_ms" in dataset_key:
        #     loc = 0
        #     while loc < tp[-1]:
        #         a.axvline(loc, color="thistle", linestyle="--")
        #         loc += dataset_key["period_ms"]
        for m in marks[i]:
            a.axvline(tp[m], color="thistle", linestyle="--")

    if not band_filter_name is None:
        tp = np.linspace(
            band_trace_window[0] / time_ratio,
            band_trace_window[1] / time_ratio,
            band_traces[0].shape[0],
        )
        for i, a in zip(sampled, ax):
            a.plot(tp, band_traces[i], color="grey")

    for a in ax:
        a.set_ylim(-600, 600)
    return fig
