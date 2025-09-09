import datajoint as dj
import numpy as np
import pandas as pd
from tqdm import tqdm

from spyglass.common import (
    AnalysisNwbfile,
    IntervalList,
    interval_list_contains,
    interval_list_contains_ind,
    interval_list_intersect,
    convert_epoch_interval_name_to_position_interval_name,
)
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from spyglass.utils.dj_mixin import SpyglassMixin
from ms_stim_analysis.Analysis.utils import get_running_valid_intervals, parse_unit_ids
from ms_stim_analysis.Analysis.spiking_analysis import smooth

schema = dj.schema("ms_cross_correlations")


@schema
class CrossCorrelogramParameters(SpyglassMixin, dj.Manual):
    definition = """
    cross_corr_params_name: varchar(128)
    ---
    interval_buffer = NULL: float
    filter_speed = 10: float
    filter_port = 1: bool
    min_run_time = .5: float
    gauss_smooth = 0.003: float
    max_lag = .5: int
    closest_spike_only = 0: bool
    dlc_position = 0: bool
    exclude_simultaneous = 0: bool
    """

    def insert_default(self):
        self.insert1(
            {
                "cross_corr_params_name": "default",
                "interval_buffer": 0,
                "filter_speed": 10,
                "min_run_time": 0.5,
                "gauss_smooth": 0.003,
                "max_lag": 0.5,
                "closest_spike_only": False,
                "dlc_position": False,
            },
            skip_duplicates=True,
        )

        self.insert1(
            {
                "cross_corr_params_name": "default_dlc",
                "interval_buffer": 0,
                "filter_speed": 10,
                "min_run_time": 0.5,
                "gauss_smooth": 0.003,
                "max_lag": 0.5,
                "closest_spike_only": False,
                "dlc_position": True,
            },
            skip_duplicates=True,
        )

        self.insert1(
            {
                "cross_corr_params_name": "all_times",
                "interval_buffer": 0,
                "filter_speed": 0,
                "min_run_time": 0,
                "gauss_smooth": 0.003,
                "max_lag": 0.5,
                "closest_spike_only": False,
                "dlc_position": False,
                "filter_port": False,
            },
            skip_duplicates=True,
        )


@schema
class CrossCorrelogramSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> SortedSpikesGroup
    -> IntervalList
    -> CrossCorrelogramParameters
    """


@schema
class CrossCorrelogram(SpyglassMixin, dj.Computed):
    definition = """
    -> CrossCorrelogramSelection
    ---
    -> AnalysisNwbfile
    corr_object_id: varchar(128)
    """

    def make(self, key):
        # get the parameters
        filter_speed = (CrossCorrelogramParameters & key).fetch1("filter_speed")
        min_run_time = (CrossCorrelogramParameters & key).fetch1("min_run_time")
        gauss_smooth = (CrossCorrelogramParameters & key).fetch1("gauss_smooth")
        filter_ports = (CrossCorrelogramParameters & key).fetch1("filter_port")
        max_lag = (CrossCorrelogramParameters & key).fetch1("max_lag")
        interval_buffer = (CrossCorrelogramParameters & key).fetch1("interval_buffer")
        closest_spike_only = (CrossCorrelogramParameters & key).fetch1(
            "closest_spike_only"
        )
        dlc_position = (CrossCorrelogramParameters & key).fetch1("dlc_position")
        exclude_simultaneous = (CrossCorrelogramParameters & key).fetch1(
            "exclude_simultaneous"
        )

        # get spike data
        spikes_list, unit_ids = (SortedSpikesGroup & key).fetch_spike_data(
            key, return_unit_ids=True
        )
        unit_ids = parse_unit_ids(unit_ids)

        # define what intervals to use
        if filter_speed or filter_ports:
            # get the running intervals
            pos_interval = convert_epoch_interval_name_to_position_interval_name(
                (IntervalList & key).fetch1("KEY"), populate_missing=False
            )
            if not pos_interval or pos_interval is None:
                pos_interval = key["interval_list_name"].split("_")[0]

            pos_key = {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": pos_interval,
            }

            run_intervals = get_running_valid_intervals(
                pos_key,
                seperate_optogenetics=False,
                filter_speed=filter_speed,
                dlc_pos=dlc_position,
            )
        else:
            run_intervals = (IntervalList & key).fetch1("valid_times")
        run_intervals = np.array(
            [
                interval
                for interval in run_intervals
                if interval[1] - interval[0] > min_run_time
            ]
        )
        entry_interval = np.array((IntervalList & key).fetch1("valid_times"))
        valid_interval = interval_list_intersect(run_intervals, entry_interval)

        results = []
        histogram_bins = np.arange(-max_lag, max_lag, 0.0005)
        print("number_units", len(spikes_list))
        # loop through unit pairs
        for n_s1, (spikes, unit_id_1) in tqdm(enumerate(zip(spikes_list, unit_ids))):
            spikes = interval_list_contains(
                run_intervals,
                spikes,
            )

            # print("spikes", spikes.size)
            # for interval in [control_interval, test_interval]:
            bins = histogram_bins[:-1] + np.diff(histogram_bins) / 2
            x = interval_list_contains(valid_interval, spikes)
            if x.size == 0:
                continue
            absolute_bin_times = np.add.outer(x, bins).ravel()
            absolute_bin_index = np.array(
                [np.arange(bins.size) for _ in range(x.size)]
            ).ravel()
            valid_bin_index = absolute_bin_index[
                interval_list_contains_ind(valid_interval, absolute_bin_times)
            ]
            valid_bin_count = np.bincount(valid_bin_index, minlength=bins.size)

            for n_s2, (spikes_2, unit_id_2) in enumerate(zip(spikes_list, unit_ids)):
                # skip auto correlograms
                spikes_2 = interval_list_contains(
                    valid_interval,
                    spikes_2,
                )

                # rename from adapting older code
                x = spikes
                x2 = spikes_2

                # get the correlogram count
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
                if exclude_simultaneous:
                    delays = delays[delays != 0]
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

                vals = vals / (
                    np.diff(bins).mean() * valid_bin_count
                )  # normalize into rate (Hz)

                results.append(
                    {
                        "unit_id_1": unit_id_1,
                        "unit_id_2": unit_id_2,
                        "correlogram": vals,
                        "bins": bins,
                        "counts_1": len(spikes),
                        "counts_2": len(spikes_2),
                    }
                )

        results = pd.DataFrame(results)
        nwb_file_name = key["nwb_file_name"]
        analysis_file_name = AnalysisNwbfile().create(nwb_file_name)
        key["analysis_file_name"] = analysis_file_name
        key["corr_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, results, "cross_correlogram"
        )
        AnalysisNwbfile().add(nwb_file_name, analysis_file_name)
        self.insert1(key)

    def fetch_dataframe(self, min_spike_count=None) -> pd.DataFrame:
        # query = "counts_1 > @min_spike and counts_2 > @min_spike_count"
        return pd.concat([data["corr"] for data in self.fetch_nwb()])

    def fetch_auto_correlograms(self) -> pd.DataFrame:
        corr_df = self.fetch_dataframe()
        return corr_df.query("unit_id_1 == unit_id_2")
