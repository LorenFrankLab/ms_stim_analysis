import os
import random
import string

import datajoint as dj
from non_local_detector.model_checking import (
    get_HPD_spatial_coverage,
    get_highest_posterior_threshold,
)
import numpy as np
import pandas as pd
from spyglass.common import (
    AnalysisNwbfile,
    IntervalList,
    interval_list_contains,
)
import xarray as xr

from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1
from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1
from spyglass.utils.dj_mixin import SpyglassMixin, SpyglassMixinPart

schema = dj.schema("ms_place_fields")


@schema
class SortedDecodingGroup(SpyglassMixin, dj.Manual):
    definition = """
    -> Session
    decode_group_name: varchar(128)
    """

    class ControlEncoding(SpyglassMixinPart):
        definition = """
        -> master
        -> SortedSpikesDecodingV1
        """

    class TestEncoding(SpyglassMixinPart):
        definition = """
        -> master
        -> SortedSpikesDecodingV1
        """

    class StimulusEncoding(SpyglassMixinPart):
        definition = """
        -> master
        -> SortedSpikesDecodingV1
        """

    def create_group(
        self,
        nwb_file_name,
        decode_group_name,
        control_decodes,
        test_decodes,
        stimulus_decodes,
    ):
        if not len(control_decodes) == len(test_decodes) == len(stimulus_decodes):
            raise ValueError("All decode sets must have the same length")
        key = {"nwb_file_name": nwb_file_name, "decode_group_name": decode_group_name}
        if self & key:
            raise ValueError("Group already exists")
        self.insert1(key)
        for decodes in [control_decodes, test_decodes, stimulus_decodes]:
            for decode in decodes:
                decode["decode_group_name"] = decode_group_name
        self.ControlEncoding().insert(control_decodes)
        self.TestEncoding().insert(test_decodes)
        self.StimulusEncoding().insert(stimulus_decodes)
        return


@schema
class OptoPlaceField(SpyglassMixin, dj.Computed):
    definition = """
    -> SortedDecodingGroup
    ---
    -> AnalysisNwbfile
    place_object_id: varchar(255)
    """

    def make(self, key):
        datasets = [
            (table & key).fetch1("KEY")
            for table in [
                SortedDecodingGroup.ControlEncoding,
                SortedDecodingGroup.TestEncoding,
                SortedDecodingGroup.StimulusEncoding,
            ]
        ]
        spikes, unit_ids = SortedSpikesDecodingV1.fetch_spike_data(
            datasets[0], return_unit_ids=True
        )  # should be same spike data for all three
        if type(unit_ids[0]) is dict:
            unit_ids = [
                f"{x['spikesorting_merge_id']}_{x['unit_id']}" for x in unit_ids
            ]

        # data we're compiling
        place_field_list = []
        raw_place_field_list = []
        encoding_spike_counts = []
        information_rates_list = []

        # calculate the values
        for data_key in datasets:
            fit_model = (SortedSpikesDecodingV1 & data_key).fetch_model()

            # get place fields
            place_field = list(fit_model.encoding_model_.values())[0]["place_fields"]
            norm_place_field = place_field / np.sum(place_field, axis=1, keepdims=True)
            place_field_list.append(norm_place_field)
            raw_place_field_list.append(place_field)

            encode_interval = (SortedSpikesDecodingV1 & data_key).fetch1(
                "encoding_interval"
            )

            # get mean rates
            encode_times = (
                IntervalList & data_key & {"interval_list_name": encode_interval}
            ).fetch1("valid_times")
            encoding_spike_counts.append(
                [len(interval_list_contains(encode_times, s)) for s in spikes]
            )

            # get information rates
            encoding = fit_model.encoding_model_
            encoding = encoding[list(encoding.keys())[0]]
            p_loc = encoding["occupancy"]
            p_loc = p_loc / p_loc.sum()
            if not p_loc.size == place_field.shape[1]:
                print("Place field and occupancy are not the same size")
                information_rates_list.append([np.nan for _ in place_field])
            else:
                information_rate = [
                    self.spatial_information_rate(spike_rate=field, p_loc=p_loc)
                    for field in place_field
                ]
                information_rates_list.append(information_rate)

        # save the results
        place_field_list = np.array([x for x in place_field_list])
        raw_place_field_list = np.array([x for x in raw_place_field_list])
        encoding_spike_counts = np.array([x for x in encoding_spike_counts])
        information_rates_list = np.array([x for x in information_rates_list])
        # compile the dataframe object
        df = []
        for i, condition in enumerate(["control", "test", "stimulus"]):
            for j, unit in enumerate(unit_ids):
                df.append(
                    {
                        "unit_id": unit,
                        "condition": condition,
                        "place_field": place_field_list[i][j],
                        "raw_place_field": raw_place_field_list[i][j],
                        "encoding_spike_count": encoding_spike_counts[i][j],
                        "information_rate": information_rates_list[i][j],
                    }
                )
        df = pd.DataFrame(df)

        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key["analysis_file_name"] = analysis_file_name
        key["place_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, df, "place_fields"
        )
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])

        self.insert1(key)

    @staticmethod
    def spatial_information_rate(
        spike_counts=None, occupancy=None, spike_rate=None, p_loc=None
    ):
        """
        Calculates the spatial information rate of units firing
        Formula from:
        Experience-Dependent Increase in CA1 Place Cell Spatial Information, But Not Spatial Reproducibility,
        Is Dependent on the Autophosphorylation of the α-Isoform of the Calcium/Calmodulin-Dependent Protein Kinase II
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2680063/
        """
        if spike_counts is not None and occupancy is not None:
            spike_rate = spike_counts / occupancy
            p_loc = occupancy / occupancy.sum()
            total_rate = spike_counts.sum() / occupancy.sum()
        elif spike_rate is not None and p_loc is not None:
            total_rate = (spike_rate * p_loc).sum()
        else:
            raise ValueError(
                "spike_counts and occupancy or spike_rate and p_loc must be provided"
            )
        return np.nansum(
            p_loc * spike_rate / total_rate * np.log2(spike_rate / total_rate)
        )

    def fetch_dataframe(self) -> pd.DataFrame:
        data = [data["place"] for data in self.fetch_nwb()]
        if len(data) > 1:
            return pd.concat(data)
        return data[0]


@schema
class PlaceFieldCoverageParams(SpyglassMixin, dj.Manual):
    definition = """
    unit_coverage_params_name: varchar(64)
    ---
    spatial_coverage_threshold: float
    """

    def insert_default(self):
        key = {
            "unit_coverage_params_name": "default",
            "spatial_coverage_threshold": 0.95,
        }
        self.insert1(key, skip_duplicates=True)


@schema
class PlaceFieldCoverageSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> OptoPlaceField
    -> PlaceFieldCoverageParams
    ---
    """


@schema
class PlaceFieldCoverage(SpyglassMixin, dj.Computed):
    definition = """
    -> PlaceFieldCoverageSelection
    ---
    -> AnalysisNwbfile
    coverage_object_id: varchar(255)
    """

    def make(self, key):
        # fetch the threshold
        threshold = (PlaceFieldCoverageParams & key).fetch1(
            "spatial_coverage_threshold"
        )

        # fetch the place fields
        spike_df = (OptoPlaceField() & key).fetch_dataframe()
        fields = spike_df.place_field.values
        # fields[np.isnan(fields)] = 0
        bad_fields = []
        for i, field in enumerate(fields):
            fields[i][np.isnan(field)] = 0
            if fields[i].sum() == 0:
                bad_fields.append(i)
                fields[i][0] = 1
        fields = np.array([f / max(np.sum(f), 1e-8) for f in fields])

        # calculate the coverage
        fields_xr = xr.DataArray(
            fields,
            dims=["unit", "position"],
        )
        thresholds = get_highest_posterior_threshold(fields_xr, threshold)
        coverage = get_HPD_spatial_coverage(fields_xr, thresholds)
        coverage[bad_fields] = 1e9
        # save the results
        coverage_df = pd.DataFrame(
            {
                "coverage": coverage,
                "unit_id": spike_df.unit_id,
                "condition": spike_df.condition,
            }
        )

        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key["analysis_file_name"] = analysis_file_name
        key["coverage_object_id"] = AnalysisNwbfile().add_nwb_object(
            key["analysis_file_name"], coverage_df, "place_field_coverage"
        )
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
        self.insert1(key)

    def fetch_dataframe(self) -> pd.DataFrame:
        return pd.concat([data["coverage"] for data in self.fetch_nwb()])


@schema
class TrackCellCoverageParams(SpyglassMixin, dj.Manual):
    definition = """
    track_coverage_params_name: varchar(64)
    ---
    place_field_coverage_specificity: float # how localized a unit's firing distribution must be for it to be considered a place cell, unit = bins
    place_field_percentile_valid: float # What percent of a unit's distribution counts as track coverage
    min_unit_coverage: int # minimum number of units that must be active at a position for it to be considered covered
    condition: enum('control', 'test', 'stimulus') # which condition to use for the track coverage
    """

    def insert_default(self):
        key = {
            "track_coverage_params_name": "default",
            "place_field_coverage_specificity": 50,
            "place_field_percentile_valid": 0.5,
            "min_unit_coverage": 2,
            "condition": "test",
        }
        self.insert1(key, skip_duplicates=True)


@schema
class TrackCellCoverageSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> PlaceFieldCoverage
    -> TrackCellCoverageParams
    ---
    """


@schema
class TrackCellCoverage(SpyglassMixin, dj.Computed):
    definition = """
    -> TrackCellCoverageSelection
    ---
    -> AnalysisNwbfile
    -> IntervalList.proj(coverage_interval_name="interval_list_name")
    coverage_object_id: varchar(255)
    """

    def make(self, key):
        # parameters
        # pf_coverage_specificity = 50
        # pf_percentile_valid = 0.5
        # min_unit_coverage = 2
        # condition = "test"
        pf_coverage_specificity, pf_percentile_valid, min_unit_coverage, condition = (
            TrackCellCoverageParams & key
        ).fetch1(
            "place_field_coverage_specificity",
            "place_field_percentile_valid",
            "min_unit_coverage",
            "condition",
        )

        # load the place fields and coverage specificity info
        pfc_df = (PlaceFieldCoverage & key).fetch_dataframe()
        pf_df = (OptoPlaceField & key).fetch_dataframe()
        pf_df = pd.merge(pf_df, pfc_df, on=["unit_id", "condition"])
        # select the place fields that are specific enough
        pf_df = pf_df[pf_df["coverage"] < pf_coverage_specificity]
        # select the place fields for the condition
        cond_df = pf_df[pf_df["condition"] == condition]
        fields = cond_df.place_field.values
        fields = np.array([f / np.sum(f) for f in fields])

        # calculate the positions covered by these good place fields
        fields_xr = xr.DataArray(
            fields,
            dims=["unit", "position"],
        )
        thresholds = get_highest_posterior_threshold(fields_xr, pf_percentile_valid)
        valid_coverage = np.array([f > thresh for f, thresh in zip(fields, thresholds)])
        # get the total coverage at each position and threshold
        position_coverage = valid_coverage.sum(axis=0)
        covered_positions = np.where(position_coverage >= min_unit_coverage)[0]
        # load the linearized position data
        decoding_entry = (
            SortedDecodingGroup().TestEncoding  # same pos group for all parts tables
            & (OptoPlaceField() & key)
        )
        decoding_table = SortedSpikesDecodingV1() & decoding_entry
        decoding_key = decoding_table.fetch1("KEY")
        pos_df = decoding_table.fetch_linear_position_info(decoding_key)
        # assign each position timepoint to a a bin on the track
        results = decoding_table.fetch_results()
        position_bins = results.position.values
        pos_bin_ind = np.argmin(
            np.abs(np.subtract.outer(pos_df.linear_position.to_numpy(), position_bins)),
            axis=1,
        )
        # determine if each timepoint happens in a covered position
        covered_time_ind = np.array(
            [x in covered_positions for x in pos_bin_ind]
        ).astype(int)
        # make an interval list of the covered timepoints
        covered_start = np.where(np.diff(covered_time_ind) == 1)[0]
        if covered_time_ind[0] == 1:
            covered_start = np.concatenate([[0], covered_start])
        covered_stop = np.where(np.diff(covered_time_ind) == -1)[0]
        if covered_time_ind[-1] == 1:
            covered_stop = np.concatenate([covered_stop, [len(covered_time_ind) - 1]])
        assert len(covered_start) == len(covered_stop)
        covered_intervals = np.array(
            [
                pos_df.iloc[covered_start].index.to_numpy(),
                pos_df.iloc[covered_stop].index.to_numpy(),
            ]
        ).T

        # save the results
        id = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
        interval_list_key = {
            "nwb_file_name": key["nwb_file_name"],
            "interval_list_name": f"{key['decode_group_name']}_track_cell_coverage_{id}",
            "valid_times": covered_intervals,
            "pipeline": self.full_table_name,
        }
        IntervalList().insert1(interval_list_key)
        key["coverage_interval_name"] = interval_list_key["interval_list_name"]

        track_df = pd.DataFrame(
            {
                "unit_coverage": position_coverage,
                "good_coverage": position_coverage >= min_unit_coverage,
            }
        )
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key["analysis_file_name"] = analysis_file_name
        key["coverage_object_id"] = AnalysisNwbfile().add_nwb_object(
            key["analysis_file_name"], track_df, "track_cell_coverage"
        )
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
        self.insert1(key)

    def fetch_good_coverage_times(self, key={}):
        """fetch times when the animal is on a section of track that is covered by a
        sufficient number of place fields

        Args:
            key (dict, optional): restriction on the table. Defaults to {}.

        Returns:
            np.ndarray: Nx2 array of start and stop times of the covered intervals
        """
        interval_key = (
            (self & key)
            .proj(interval_list_name="coverage_interval_name")
            .fetch("nwb_file_name", "interval_list_name", as_dict=True)
        )
        return (IntervalList & interval_key).fetch1("valid_times")

    def fetch1_dataframe(self) -> pd.DataFrame:
        assert len(nwb := self.fetch_nwb()) == 1
        return nwb[0]["coverage"]


@schema
class DecodesToCoveredTrackSelection(SpyglassMixin, dj.Manual):
    definition = """
    -> ClusterlessDecodingV1
    -> TrackCellCoverage
    ---
    """


@schema
class DecodesToCoveredTrack(SpyglassMixin, dj.Computed):
    # Defines whether the decoded positions maps to a track location with sufficient
    # place field coverage at each timepoint

    definition = """
    -> DecodesToCoveredTrackSelection
    ---
    -> AnalysisNwbfile
    object_id: varchar(255)
    """

    def make(self, key):
        # get track coverage data
        track_df = (TrackCellCoverage() & key).fetch1_dataframe()
        # get clusterless decoding results
        results = (ClusterlessDecodingV1() & key).fetch_results()
        full_posterior = results.causal_posterior.unstack("state_bins")
        posterior = full_posterior.sum("state")[0]
        decode_pos_bin = np.argmax(np.array(posterior), axis=1)
        # map decoded positions to track coverage
        decode_to_covered = track_df.loc[decode_pos_bin].good_coverage.values

        # save the results
        df = pd.DataFrame(
            {"decode_to_covered": decode_to_covered, "time": results.time}
        )
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        key["analysis_file_name"] = analysis_file_name
        key["object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, df, "decode_to_covered_track"
        )
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
        self.insert1(key)

    def fetch1_dataframe(self) -> pd.DataFrame:
        assert len(nwb := self.fetch_nwb()) == 1
        return nwb[0]["object_id"]

    def fetch_intervals(self, key={}):
        df = (self & key).fetch1_dataframe()
        good = df.values.astype(int)[:, 0]
        change = np.diff(good, axis=0)

        start = np.where(change == 1)[0]
        end = np.where(change == -1)[0]
        if good[0] == 1:
            start = np.concatenate([[0], start])
        if good[-1] == 1:
            end = np.concatenate([end, [good.shape[0] - 1]])

        time = df.time.values
        intervals = []
        for s, e in zip(start, end):
            intervals.append([time[s], time[e]])
        return np.array(intervals)
