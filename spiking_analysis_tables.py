from typing import Tuple
import numpy as np
import pandas as pd

from spyglass.spikesorting.v0 import CuratedSpikeSorting
import datajoint as dj

schema = dj.schema("sambray_spiking")


@schema
class BinnedSpikingParams(dj.Manual):
    definition = """
    # Parameters for binning spiking data
    binned_spike_params_name: varchar(80) # Name of the binned spike params
    ---
    bin_width : float # Bin width in seconds
    """

    @classmethod
    def insert_default(cls, **kwargs):
        cls.insert1(
            {"binned_spike_params_name": "default", "bin_width": 0.002},
            skip_duplicates=True,
        )


@schema
class BinnedSpikingSelection(dj.Manual):
    definition = """
    # Parameters for binning spiking data
    -> BinnedSpikingParams
    -> CuratedSpikeSorting
    ---
    """


@schema
class BinnedSpiking(dj.Computed):
    definition = """
    # Binned spiking data
    -> BinnedSpikingSelection
    ---
    binned_spiking : longblob
    time_bins : longblob
    """

    def make(self, key):
        # get spike data
        tetrode_df = (CuratedSpikeSorting & key).fetch_nwb()[0]
        spike_df = []
        if "units" in tetrode_df:
            tetrode_df = tetrode_df["units"]
            tetrode_df = tetrode_df[tetrode_df.label == ""]
            spike_df.append(tetrode_df)

        # if no good units insert an empty array
        if (len(spike_df) == 0) or (len(spike_df) == 1 and len(spike_df[0]) == 0):
            self.insert1(
                {
                    **key,
                    "binned_spiking": np.array([]),
                    "time_bins": np.array([]),
                }
            )
            return
        # make the full dataframe
        spike_df = pd.concat(spike_df)
        # get the bin width
        bin_width = (BinnedSpikingParams & key).fetch1("bin_width")
        # get the bin edges
        t_min = np.min([np.min(spikes) for spikes in spike_df.spike_times])
        t_max = np.max([np.max(spikes) for spikes in spike_df.spike_times])
        bin_edges = np.arange(t_min, t_max, bin_width)
        # bin the spikes
        binned_spiking = np.array(
            [
                get_spikecount_per_time_bin(spikes, bin_edges)
                for spikes in spike_df.spike_times
            ]
        )
        # insert results
        self.insert1(
            {
                **key,
                "binned_spiking": binned_spiking,
                "time_bins": bin_edges,
            }
        )

    def mark_alligned_binned_spikes(
        self, key: dict, marks: list, rng: Tuple[float, float]
    ) -> np.ndarray:
        """return binned spikes alligned to marks

        Parameters
        ----------
        key : dict
            key for BinnedSpiking table
        marks : list
            mark times to allign bins to
        rng : Tuple[float, float]
            range of time around marks to return

        Returns
        -------
        np.ndarray
            allign binned spikes, shape = (n_marks, n_units, n_bins)
        """
        # find mark locations
        mark_ind = np.digitize(marks, (self & key).fetch1("time_bins"))
        # get spikes
        binned_spiking = (self & key).fetch1("binned_spiking")
        if len(binned_spiking) == 0:
            return
        # translate rng from time to index
        rng = (np.array(rng) / np.diff((self & key).fetch1("time_bins")[0:2])).astype(
            int
        )
        # get alligned spikes
        alligned_spiking = np.array(
            [
                binned_spiking[:, ind + rng[0] : ind + rng[1]]
                for ind in mark_ind
                if (ind + rng[0] >= 0 and ind + rng[1] < binned_spiking.shape[1])
            ]
        )
        return alligned_spiking  # shape = (n_marks, n_units, n_bins)

    def get_current_curation_key_list(self, key: dict = {}) -> list:
        """get list of all curation keys that have been used to curate this table

        Parameters
        ----------
        key : dict
            key for BinnedSpiking table

        Returns
        -------
        list
            list of keys for the most recent curation of each dataset that matches the key
        """
        sub_table = self & key
        key_list = []
        for nwb_file_name in set(sub_table.fetch("nwb_file_name")):
            key = dict(nwb_file_name=nwb_file_name)
            for sort_group in set((sub_table & key).fetch("sort_group_id")):
                key["sort_group_id"] = sort_group
                for sort_interval_name in set(
                    (sub_table & key).fetch("sort_interval_name")
                ):
                    key["sort_interval_name"] = sort_interval_name
                    cur_id = np.max((sub_table & key).fetch("curation_id"))
                    key["curation_id"] = cur_id
                    key_list.append((sub_table & key).fetch1("KEY"))
        return key_list


def get_spikecount_per_time_bin(spike_times, time):
    spike_times = spike_times[
        np.logical_and(spike_times >= time[0], spike_times <= time[-1])
    ]
    return np.bincount(
        np.digitize(spike_times, time[1:-1]),
        minlength=time.shape[0],
    )
