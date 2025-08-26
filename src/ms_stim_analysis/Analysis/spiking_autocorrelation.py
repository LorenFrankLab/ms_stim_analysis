import numpy as np
import matplotlib.pyplot as plt

from spyglass.common import PositionIntervalMap
from spyglass.common import interval_list_contains_ind


from ms_stim_analysis.AnalysisTables.ms_opto_stim_protocol import OptoStimProtocol
from .position_analysis import get_running_intervals, filter_position_ports
from .utils import get_running_valid_intervals, autocorr2d, filter_opto_data, smooth


def spiking_autocorrelation(
    dataset_key: dict,
    filter_speed: float = 10,
    filter_ports: bool = True,
    auto_corr_window: int = 500,
):
    raise Warning(
        "This function not used in final publication in favor of table-based analyses"
    )
    dataset = filter_opto_data(dataset_key)
    C_all = [[], []]
    for nwb_file_name, pos_interval_name in zip(
        dataset.fetch("nwb_file_name"), dataset.fetch("interval_list_name")
    ):
        sort_interval_name = (
            PositionIntervalMap()
            & {
                "nwb_file_name": nwb_file_name,
                "position_interval_name": pos_interval_name,
            }
        ).fetch1("interval_list_name")

        basic_key = {
            "nwb_file_name": nwb_file_name,
            "sort_interval_name": sort_interval_name,
        }

        interval_key = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": sort_interval_name,
        }

        pos_key = {
            "nwb_file_name": basic_key["nwb_file_name"],
            "interval_list_name": pos_interval_name,
        }

        try:
            (
                optogenetic_run_interval,
                control_run_interval,
            ) = get_running_valid_intervals(
                pos_key,
                filter_speed=filter_speed,
                filter_ports=filter_ports,
                seperate_optogenetics=True,
            )
        except:
            continue

        # Get binned spiking data
        key_list = BinnedSpiking().get_current_curation_key_list(basic_key)
        spike_counts = []
        times = []
        for key in key_list:
            group_counts = (BinnedSpiking & key).fetch1("binned_spiking")
            if len(group_counts) > 0:
                spike_counts.append(group_counts)
                times.append((BinnedSpiking & key).fetch1("time_bins"))

        C = [[[] for _ in spike_counts], [[] for _ in spike_counts]]
        for cond, interval_list in enumerate(
            [control_run_interval, optogenetic_run_interval]
        ):
            for interval in interval_list:
                for i, (t, counts) in enumerate(zip(times, spike_counts)):
                    ind = interval_list_contains_ind(np.array([interval]), t)
                    if ind.size >= auto_corr_window:
                        counts_interval = smooth(counts[:, ind].T, 3)
                        C[cond][i].append(
                            autocorr2d(counts_interval)[:auto_corr_window]
                        )
                    else:
                        C[cond][i].append(
                            np.zeros((auto_corr_window, counts.shape[0])) * np.nan
                        )
        if len(C[0]) == 0 or len(C[1]) == 0:
            continue
        if len(C[0]) > 1:
            C = [np.concatenate(c, axis=-1) for c in C]
        print(C[0].shape, C[1].shape)
        print(len(C), len(C_all))
        C_all[0].append(np.nanmean(C[0], axis=(0)))
        C_all[1].append(np.nanmean(C[1], axis=(0)))
        period_ms = (OptoStimProtocol() & pos_key).fetch1("period_ms")
    if len(C_all[0]) == 0 or len(C_all[1]) == 0:
        return
    C_all = [np.concatenate(c, axis=-1) for c in C_all]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    t = np.arange(auto_corr_window) * 2
    for cond, color in enumerate(["cornflowerblue", "firebrick"]):
        ax[cond].plot(
            t,
            C_all[cond],
            color=color,
            alpha=min(0.15, 30 / np.array(C_all[cond]).shape[1]),
        )
        ax[cond].plot(t, np.nanmean(C_all[cond], axis=(1,)), color=color, alpha=1)
    ax[0].set_yscale("log")
    ax[0].set_ylim([1e-3, 1])
    # ax[0].set_xscale('log')
    ax[0].set_xlim([-3, 300])

    ax[0].set_ylabel("Autocorrelation")
    ax[0].set_xlabel("Time (ms)")
    ax[1].set_xlabel("Time (ms)")
    ax[0].set_title("Control Interval")
    ax[1].set_title("Optogenetic Interval")

    for i in range(10):
        ax[0].vlines(
            x=period_ms * i, ymin=0, ymax=1, color="purple", linestyle="dotted"
        )
        ax[1].vlines(
            x=period_ms * i, ymin=0, ymax=1, color="purple", linestyle="dotted"
        )

    for i in range(10):
        ax[0].vlines(x=125 * i, ymin=0, ymax=1, color="k", linestyle="--")
        ax[1].vlines(x=125 * i, ymin=0, ymax=1, color="k", linestyle="--")

    fig.suptitle(f"{dataset_key['animal']}: {dataset_key['period_ms']}ms")
