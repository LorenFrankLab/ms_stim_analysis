import os

import datajoint as dj
import numpy as np
import pandas as pd
from spyglass.common import (
    AnalysisNwbfile,
    IntervalList,
    Session,
    interval_list_contains,
)
from spyglass.decoding.v1.sorted_spikes import SortedSpikesDecodingV1
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
        return pd.concat([data["place"] for data in self.fetch_nwb()])
