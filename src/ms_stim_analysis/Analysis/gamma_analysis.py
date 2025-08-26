import numpy as np
import matplotlib.pyplot as plt

from spyglass.common import Session

from spyglass.common.common_interval import interval_list_contains
from spyglass.lfp.analysis.v1 import LFPBandV1

from .utils import (
    filter_opto_data,
    get_running_valid_intervals,
)

from .lfp_analysis import get_ref_electrode_index


LFP_AMP_CUTOFF = 2000

################################################################################


def gamma_theta_nesting(
    dataset_key: dict,
    phase_filter_name: str = "Theta 5-11 Hz",
    filter_speed: float = 10.0,
    window: float = 1.0,
    return_distributions: bool = False,
):
    # Use the driving period lfp band if band filter not specified
    if phase_filter_name is None:
        if "period_ms" not in dataset_key:
            raise ValueError("band_filter_name must be specified if period_ms is not")
        phase_filter_name = f"ms_stim_{dataset_key['period_ms']}ms_period"

    # Define the dataset (epochs included in this analyusis)
    dataset = filter_opto_data(dataset_key)

    nwb_file_name_list = dataset.fetch("nwb_file_name")
    interval_list_name_list = dataset.fetch("interval_list_name")
    power = []
    phase = []

    if len(set((dataset * Session).fetch("subject_id"))) > 1:
        raise NotImplementedError("Only one subject allowed")

    # make the figure
    fig, ax_all = plt.subplots(
        2, 4, figsize=(20, 6), gridspec_kw={"width_ratios": [3, 3, 3, 1]}, sharex="col"
    )
    for power_filter, ax in zip(
        ["Slow Gamma 25-55 Hz", "Fast Gamma 65-100 Hz"], ax_all
    ):
        power_opto = []
        power_control = []
        phase_opto = []
        phase_control = []

        for nwb_file_name, interval_list_name in zip(
            nwb_file_name_list, interval_list_name_list
        ):
            basic_key = {
                "nwb_file_name": nwb_file_name,
                "target_interval_list_name": interval_list_name,
            }
            print(basic_key)

            # define the key for the band used to define phase
            phase_key = {
                **basic_key,
                "filter_name": phase_filter_name,
                "filter_sampling_rate": 1000,
            }
            # define the key for the band used to define amplitude
            power_key = {**basic_key, "filter_name": power_filter}
            print(power_key)
            print(phase_key)

            # get analytic band power
            ref_elect_index, basic_key = get_ref_electrode_index(basic_key)
            power_df = (LFPBandV1 & power_key).compute_signal_power([ref_elect_index])
            power_ = np.asarray(power_df[power_df.columns[0]])
            power_timestamps = power_df.index

            # get phase
            if not (LFPBandV1 & phase_key) or not (LFPBandV1 & power_key):
                continue
            phase_df = (LFPBandV1 & phase_key).compute_signal_phase([ref_elect_index])
            phase_timestamps = phase_df.index
            phase_ = np.asarray(phase_df)[:, 0]

            # get test and control run intervals
            pos_key = {
                **basic_key,
                "interval_list_name": basic_key["target_interval_list_name"],
            }
            opto_run_intervals, control_run_intervals = get_running_valid_intervals(
                pos_key, filter_speed=filter_speed, seperate_optogenetics=True
            )

            for intervals, power, phase in zip(
                [opto_run_intervals, control_run_intervals],
                [power_opto, power_control],
                [phase_opto, phase_control],
            ):
                valid_times = interval_list_contains(intervals, phase_timestamps)
                ind_power = np.digitize(valid_times, power_timestamps)
                power.extend(power_[ind_power - 1])
                ind_phase = np.digitize(valid_times, phase_timestamps)
                phase.extend(phase_[ind_phase - 1])

        if len(phase_opto) == 0 or len(phase_control) == 0:
            return None, None, None, None

        for a, phase, power, color, name in zip(
            ax[:2],
            [phase_opto, phase_control],
            [power_opto, power_control],
            ["firebrick", "cornflowerblue"],
            ["opto", "control"],
        ):
            H, xedges, yedges = np.histogram2d(
                phase,
                np.log10(power),
                bins=[np.linspace(0, 2 * np.pi, 100), np.linspace(0, 5, 30)],
            )
            # H, xedges, yedges = np.histogram2d(
            #     phase,
            #     power,
            #     bins=[np.linspace(0, 2 * np.pi, 100), np.linspace(0, 1e5, 100)],
            # )
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            H = H / H.sum(axis=1)[:, None]
            a.imshow(
                H.T,
                origin="lower",
                extent=extent,
                cmap=plt.cm.plasma,
                interpolation="nearest",
                aspect=float(np.diff(extent)[0]) / np.diff(extent)[-1],
            )
            a.set_xlabel("Phase")
            a.set_ylabel("log10 Power")
            a.set_title(name)

            # val, bins, ind = scipy.stats.binned_statistic(phase, np.log10(power), bins=100)
            # # val, bins, ind = scipy.stats.binned_statistic(phase, power, bins=100)
            # bin_centers = bins[:-1] + np.diff(bins) / 2

            bins = np.linspace(0, 2 * np.pi, 65)
            labels = np.digitize(phase, bins)
            val, rng_lo, rng_hi = bootstrap_binned(labels, np.log10(power), n_boot=1000)
            bin_centers = bins[:-1] + np.diff(bins) / 2

            ax[2].plot(bin_centers, val, color=color, label=name)
            ax[2].fill_between(bin_centers, rng_lo, rng_hi, facecolor=color, alpha=0.3)
            ax[2].set_xlabel("Phase")
            ax[2].set_ylabel(f"log10 Power {power_filter}")

        ax[2].spines[["top", "right"]].set_visible(False)
        ax[2].set_xlim([0, 2 * np.pi])
        ax[2].legend()

    # Table with information about the dataset
    the_table = ax[3].table(
        cellText=[[len(dataset)], [phase_filter_name]]
        + [[str(x)] for x in dataset_key.values()],
        rowLabels=["number_epochs", "phase_filter_name"]
        + [str(x) for x in dataset_key.keys()],
        loc="right",
        colWidths=[1, 1],
    )

    for a in ax_all[:, 3]:
        a.spines[["top", "right", "left", "bottom"]].set_visible(False)
        a.set_xticks([])
        a.set_yticks([])

    fig.suptitle(f"{dataset_key['animal']}: {dataset_key['period_ms']}ms period")
    return phase_opto, phase_control, power_opto, power_control


def bootstrap_binned(labels, values, n_boot=1000):
    unique_labels = np.unique(labels)
    bootstrap_dist = []

    for label in unique_labels:
        ind = labels == label
        samples = values[ind]
        boot_samples = np.random.choice(samples, (n_boot, len(samples)))
        boot_means = np.nanmean(boot_samples, axis=1)
        bootstrap_dist.append(boot_means)

    bootstrap_dist = np.array(bootstrap_dist)
    return (
        np.mean(bootstrap_dist, axis=1),
        np.percentile(bootstrap_dist, 0.5, axis=1),
        np.percentile(bootstrap_dist, 99.5, axis=1),
    )
