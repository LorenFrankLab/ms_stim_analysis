import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spyglass as nd
from spyglass.common import Nwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb
from spyglass.utils.dj_mixin import SpyglassMixin

from .Utils.datajoint_table_helpers import (
    insert_analysis_table_entry,
    fetch1_dataframe,
    get_table_object_id_name,
)
from .Utils.df_helpers import df_pop
from .Utils.nwbf_helpers import events_in_epoch_bool
from .Utils.plot_helpers import format_ax
from .Utils.vector_helpers import remove_repeat_elements

schema = dj.schema("ms_dio_event")


@schema
class DioEvents(SpyglassMixin, dj.Computed):
    definition = """
    # DIO events recorded at full sampling rate
    -> TaskIdentification
    ---
    -> nd.common.AnalysisNwbfile
    dio_events_object_id : varchar(40)
    """

    def make(self, key):
        # Get DIO events for this epoch from nwb file
        nwbf = (Nwbfile & key).fetch_nwb[0]
        nwbf_dios = nwbf.fields["processing"]["behavior"]["behavioral_events"].fields[
            "time_series"
        ]
        dio_event_values_list, dio_event_times_list, dio_descriptions = tuple(
            [[] for _ in range(3)]
        )
        for dio_name, dios in nwbf_dios.items():  # for each DIO type
            # Filter for events in epoch
            dio_event_times = np.asarray(dios.fields["timestamps"])
            valid_bool = events_in_epoch_bool(
                nwb_file_name=key["nwb_file_name"],
                epoch=key["epoch"],
                event_times=dio_event_times,
            )
            dio_event_values_list.append(
                np.asarray(dios.fields["data"])[valid_bool]
            )  # DIO event values
            dio_event_times_list.append(dio_event_times[valid_bool])  # DIO event times
            dio_descriptions.append(dios.fields["description"])
        dio_event_df = pd.DataFrame.from_dict(
            {
                "dio_name": list(nwbf_dios.keys()),
                "dio_description": dio_descriptions,
                "dio_int": [convert_dio_description(x) for x in dio_descriptions],
                "dio_event_times": dio_event_times_list,
                "dio_event_values": dio_event_values_list,
            }
        )
        # Store
        insert_analysis_table_entry(self, [dio_event_df], key)

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self,
            (nd.common.AnalysisNwbfile, "analysis_file_abs_path"),
            *attrs,
            **kwargs,
        )

    def fetch1_dataframe(self):
        return fetch1_dataframe(self, "dio_events").set_index("dio_int")

    def plot_dios(self, nwb_file_name, epoch):
        df = (
            self & {"nwb_file_name": nwb_file_name, "epoch": epoch}
        ).fetch1_dataframe()
        fig, axes = plt.subplots(len(df), 1, sharex=True, figsize=(15, 2 * len(df)))
        fig.tight_layout()
        for (_, df_row), ax in zip(df.iterrows(), axes):
            ax.plot(df_row["dio_event_times"], df_row["dio_event_values"], ".")
            format_ax(ax, title=f"{df_row['dio_name']} {df_row['dio_description']}")

    def get_object_id_name(
        self, leave_out_object_id=False, unpack_single_object_id=True
    ):
        return get_table_object_id_name(
            self, leave_out_object_id, unpack_single_object_id
        )


@schema
class ProcessedDioEvents(dj.Computed):
    definition = """
    # Processed DIO events
    -> DioEvents
    ---
    -> nd.common.AnalysisNwbfile
    processed_dio_events_object_id : varchar(40)
    """

    class Stim(dj.Part):
        definition = """
        # Opto stim events
        -> ProcessedDioEvents
        ---
        -> nd.common.AnalysisNwbfile
        stim_object_id : varchar(40)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self,
                (nd.common.AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )

        def fetch1_dataframe(self):
            return fetch1_dataframe(self, object_name="stim").set_index(
                "dio_event_times"
            )

        def get_object_id_name(
            self, leave_out_object_id=False, unpack_single_object_id=True
        ):
            return get_table_object_id_name(
                self, leave_out_object_id, unpack_single_object_id
            )

    class StimUp(dj.Part):
        definition = """
        # Opto stim up events
        -> ProcessedDioEvents
        ---
        -> nd.common.AnalysisNwbfile
        stim_up_object_id : varchar(40)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self,
                (nd.common.AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )

        def fetch1_dataframe(self):
            return fetch1_dataframe(self, object_name="stim_up").set_index(
                "dio_event_times"
            )

        def get_object_id_name(
            self, leave_out_object_id=False, unpack_single_object_id=True
        ):
            return get_table_object_id_name(
                self, leave_out_object_id, unpack_single_object_id
            )

    class StimDown(dj.Part):
        definition = """
        # Opto stim up events
        -> ProcessedDioEvents
        ---
        -> nd.common.AnalysisNwbfile
        stim_down_object_id : varchar(40)
        """

        def fetch_nwb(self, *attrs, **kwargs):
            return fetch_nwb(
                self,
                (nd.common.AnalysisNwbfile, "analysis_file_abs_path"),
                *attrs,
                **kwargs,
            )

        def fetch1_dataframe(self):
            return fetch1_dataframe(self, object_name="stim_down").set_index(
                "dio_event_times"
            )

        def get_object_id_name(
            self, leave_out_object_id=False, unpack_single_object_id=True
        ):
            return get_table_object_id_name(
                self, leave_out_object_id, unpack_single_object_id
            )

    class Pokes(dj.Part):
        definition = """
        # DIO well poke events
        -> ProcessedDioEvents
        ---
        dio_poke_names : blob
        dio_poke_times : blob
        dio_poke_values : blob
        """

    class FirstUpPokes(dj.Part):
        definition = """
        # DIO well poke events with consecutive up pokes (after first) at same well removed
        -> ProcessedDioEvents.Pokes
        ---
        dio_first_poke_names : blob
        dio_first_poke_times : blob
        dio_first_poke_values : blob
        """

    class LastDownPokes(dj.Part):
        definition = """
        # DIO well poke events with consecutive down pokes (until last) at same well removed
        -> ProcessedDioEvents.Pokes
        ---
        dio_last_poke_names : blob
        dio_last_poke_times : blob
        dio_last_poke_values : blob
        """

    class Pumps(dj.Part):
        definition = """
        # DIO pump events
        -> ProcessedDioEvents
        ---
        dio_pump_names : blob
        dio_pump_times : blob
        dio_pump_values : blob
        """

    def make(self, key):

        # Put all processed dio events into a single df
        dio_df = (DioEvents & key).fetch1_dataframe()
        dio_names = dio_df["dio_name"]  # DIO event names
        dict_temp = {
            "dio_event_names": [],
            "dio_event_times": [],
            "dio_event_values": [],
        }  # initialize dictionary to store all dio events across epochs
        for dio_name in dio_names:  # for each dio poke event
            dio_event_times = df_pop(dio_df, {"dio_name": dio_name}, "dio_event_times")
            dio_event_values = df_pop(
                dio_df, {"dio_name": dio_name}, "dio_event_values"
            )
            dict_temp["dio_event_names"] += [dio_name] * len(dio_event_times)
            dict_temp["dio_event_times"] += list(dio_event_times)
            dict_temp["dio_event_values"] += list(dio_event_values)
        all_dio_df = pd.DataFrame.from_dict(dict_temp)
        all_dio_df.set_index("dio_event_times", inplace=True)
        all_dio_df.sort_index(inplace=True)

        # Insert into table
        all_dio_df_reset_index = (
            all_dio_df.reset_index()
        )  # index does not save out to analysis nwbf so must reset
        insert_analysis_table_entry(self, [all_dio_df_reset_index], key)

        # Now prepare to populate subtables for stim events
        # Filter for stim events
        poke_dio_names = [
            dio_name for dio_name in dio_names if "stim" in dio_name
        ]  # get names of stim dio events
        dio_stim_df = all_dio_df[all_dio_df["dio_event_names"].isin(poke_dio_names)]
        dio_stim_ups_df = dio_stim_df[
            dio_stim_df["dio_event_values"] == 1
        ]  # filter for up events
        dio_stim_downs_df = dio_stim_df[
            dio_stim_df["dio_event_values"] == 0
        ]  # filter for down events

        # Now prepare to populate subtables for pokes
        # Filter for dio poke events
        poke_dio_names = [
            dio_name for dio_name in dio_names if "poke" in dio_name
        ]  # get names of poke dio events
        dio_pokes_df = all_dio_df[all_dio_df["dio_event_names"].isin(poke_dio_names)]

        # Get first dio up event in series of consecutive up events at same well
        dio_pokes_ups_df = dio_pokes_df[
            dio_pokes_df["dio_event_values"] == 1
        ]  # filter for up events
        _, idxs = remove_repeat_elements(
            dio_pokes_ups_df["dio_event_names"], keep_first=True
        )  # find consecutive pokes at same well (after first)
        dio_pokes_first_ups_df = dio_pokes_ups_df.iloc[
            idxs
        ]  # remove consecutive pokes at same well

        # Get last DIO down events in series of consecutive down events at same well
        # Only consider DIO down events that happen after first DIO up event
        dio_pokes_downs_df = dio_pokes_df[
            dio_pokes_df["dio_event_values"] == 0
        ]  # filter for down events
        # Initialize variable for first up time
        if len(dio_pokes_downs_df) > 0:  # if down events
            first_up_time = dio_pokes_downs_df.index[
                -1
            ]  # initialize variable for first up time to last down event
        else:
            first_up_time = -1  # value will not matter since no down pokes to filter
        if len(dio_pokes_first_ups_df) > 0:  # if up events
            first_up_time = dio_pokes_first_ups_df.index[
                0
            ]  # get time of first up event
        dio_pokes_downs_df = dio_pokes_downs_df[
            dio_pokes_downs_df.index > first_up_time
        ]  # filter for down events after first up event
        _, idxs = remove_repeat_elements(
            dio_pokes_downs_df["dio_event_names"], keep_first=False
        )  # find consecutive pokes at same well (before last)
        dio_pokes_last_downs_df = dio_pokes_downs_df.iloc[
            idxs
        ]  # remove consecutive pokes at same well

        # Check that same well visits found for first dio ups and last dio downs, tolerating having one less down than
        # up event (since recording can be stopped during a dio up)
        if len(dio_pokes_first_ups_df) - len(dio_pokes_last_downs_df) not in [0, 1]:
            raise Exception(
                f"Should have found either zero or one more dio up events than dio down events, but found "
                f"{len(dio_pokes_first_ups_df) - len(dio_pokes_last_downs_df)}"
            )
        if not all(
            dio_pokes_last_downs_df["dio_event_names"].values
            == dio_pokes_first_ups_df["dio_event_names"]
            .iloc[: len(dio_pokes_last_downs_df)]
            .values
        ):
            raise Exception(
                f"Not all well identities the same for first dio ups and last dio downs"
            )
        # Check that each dio down after same index dio up and before next index dio up
        if not np.logical_and(
            all(
                (
                    dio_pokes_last_downs_df["dio_event_names"].index[:-1]
                    - dio_pokes_first_ups_df["dio_event_names"]
                    .iloc[: len(dio_pokes_last_downs_df)]
                    .index[1:]
                )
                < 0
            ),
            all(
                (
                    dio_pokes_last_downs_df["dio_event_names"].index
                    - dio_pokes_first_ups_df["dio_event_names"]
                    .iloc[: len(dio_pokes_last_downs_df)]
                    .index
                )
                > 0
            ),
        ):
            raise Exception(
                f"At least one dio down is not after same index dio up and next index dio up"
            )

        # Populate subtable for stim events
        insert_analysis_table_entry(self.Stim(), [dio_stim_df], key, reset_index=True)

        # Populate subtable for stim up events
        insert_analysis_table_entry(
            self.StimUp(), [dio_stim_ups_df], key, reset_index=True
        )

        # Populate subtable for stim down events
        insert_analysis_table_entry(
            self.StimDown(), [dio_stim_downs_df], key, reset_index=True
        )

        # Populate subtable for well pokes
        ProcessedDioEvents.Pokes.insert1(
            {
                **key,
                **{
                    "dio_poke_names": dio_pokes_df["dio_event_names"].to_numpy(),
                    "dio_poke_times": dio_pokes_df.index.to_numpy(),
                    "dio_poke_values": dio_pokes_df["dio_event_values"].to_numpy(),
                },
            }
        )
        print(
            "Populated ProcessedDioEvents.Pokes for file {nwb_file_name}, epoch {epoch}".format(
                **key
            )
        )

        # Populate subtable for first well up pokes
        ProcessedDioEvents.FirstUpPokes.insert1(
            {
                **key,
                **{
                    "dio_first_poke_names": dio_pokes_first_ups_df[
                        "dio_event_names"
                    ].to_numpy(),
                    "dio_first_poke_times": dio_pokes_first_ups_df.index.to_numpy(),
                    "dio_first_poke_values": dio_pokes_first_ups_df[
                        "dio_event_values"
                    ].to_numpy(),
                },
            }
        )
        print(
            "Populated ProcessedDioEvents.FirstPokes for file {nwb_file_name}, epoch {epoch}".format(
                **key
            )
        )

        # Populate subtable for last well down pokes
        ProcessedDioEvents.LastDownPokes.insert1(
            {
                **key,
                **{
                    "dio_last_poke_names": dio_pokes_last_downs_df[
                        "dio_event_names"
                    ].to_numpy(),
                    "dio_last_poke_times": dio_pokes_last_downs_df.index.to_numpy(),
                    "dio_last_poke_values": dio_pokes_last_downs_df[
                        "dio_event_values"
                    ].to_numpy(),
                },
            }
        )
        print(
            "Populated ProcessedDioEvents.LastDownPokes for file {nwb_file_name}, epoch {epoch}".format(
                **key
            )
        )

        # Populate subtable for pump events
        pump_dio_names = [
            dio_name for dio_name in dio_names if "pump" in dio_name
        ]  # filter for pump dio events
        dio_pumps_df = all_dio_df[all_dio_df["dio_event_names"].isin(pump_dio_names)]
        # Only consider dio down events that happen after first dio up event
        valid_idxs = []  # initialize valid idxs to empty list
        if np.sum(dio_pumps_df["dio_event_values"] == 1) > 0:  # if dio up events
            idx_first_up = np.where(dio_pumps_df["dio_event_values"] == 1)[0][0]
            valid_idxs = np.arange(idx_first_up, len(dio_pumps_df))
        dio_pumps_df = dio_pumps_df.iloc[valid_idxs]
        ProcessedDioEvents.Pumps.insert1(
            {
                **key,
                **{
                    "dio_pump_names": dio_pumps_df["dio_event_names"].to_numpy(),
                    "dio_pump_times": dio_pumps_df.index.to_numpy(),
                    "dio_pump_values": dio_pumps_df["dio_event_values"].to_numpy(),
                },
            }
        )
        print(
            "Populated ProcessedDioEvents.Pumps for file {nwb_file_name}, epoch {epoch}".format(
                **key
            )
        )

    def fetch_nwb(self, *attrs, **kwargs):
        return fetch_nwb(
            self,
            (nd.common.AnalysisNwbfile, "analysis_file_abs_path"),
            *attrs,
            **kwargs,
        )

    def fetch1_dataframe(self):
        return fetch1_dataframe(self, object_name="processed_dio_events").set_index(
            "dio_event_times"
        )

    def get_object_id_name(
        self, leave_out_object_id=False, unpack_single_object_id=True
    ):
        return get_table_object_id_name(
            self, leave_out_object_id, unpack_single_object_id
        )


# Checked that the following assignments hold in metadata files (at /cumulus/amankili/{animal_name}/metadata) for:
# Winnie20220719.yml, Winnie20220720.yml
def get_poke_map():
    # Laterality assessed from top view
    return {"poke3": "right_poke", "poke4": "center_poke", "poke5": "left_poke"}


def get_poke_name(metadata_poke_name):
    return get_poke_map()[metadata_poke_name]


def convert_dio_description(dio_description, convert_to_type=None):
    """
    Convert dio description to/from "Dio{x}" (string) and x (int)
    :param dio_description:
    :param convert_to_type:
    :return:
    """
    # Check inputs
    valid_types = ["int", "string", None]
    if convert_to_type not in valid_types:
        raise Exception(f"convert_to_type must be in {valid_types}")
    # Get digital in as integer (helpful for all cases below)
    if isinstance(dio_description, str):
        dio_description_int = int(dio_description.split("Din")[-1].split("Dout")[-1])
    else:
        dio_description_int = int(dio_description)
    # Return in desired form
    if convert_to_type == "int" or (
        convert_to_type is None and isinstance(dio_description, str)
    ):
        return dio_description_int
    elif convert_to_type == "string" or (
        convert_to_type is None and isinstance(dio_description, int)
    ):
        return f"Din{dio_description_int}"
    else:
        raise Exception(f"No valid conditions met to convert digital input")
