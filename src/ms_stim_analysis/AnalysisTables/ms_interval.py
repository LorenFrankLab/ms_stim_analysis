import datajoint as dj
import numpy as np
from .ms_task_identification import TaskIdentification
from spyglass.common import TaskEpoch, IntervalList
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("ms_interval")  # define custom schema

# Do not remove these tables
TaskIdentification


@schema
class EpochIntervalListName(SpyglassMixin, dj.Computed):
    definition = """
    # Map between epochs and interval list names
    -> TaskIdentification
    -> IntervalList
    """

    # Note that tried having IntervalList as upstream table but for it to not influence primary key;
    # This led to mismatched epochs and interval list names.
    # Takes a long time to populate because consider all pairs of epochs and interval list names
    # for an nwb file.

    def make(self, key):
        # Find correspondence between pos valid times names and epochs
        # Use epsilon to tolerate small differences in epoch boundaries across epoch/pos intervals

        # *** HARD CODED VALUES ***
        epsilon = 0.11  # tolerated time difference in epoch boundaries across epoch/pos intervals
        # *************************

        # Unpack key
        nwb_file_name = key["nwb_file_name"]

        # Get pos interval list names
        pos_interval_list_names = get_pos_interval_list_names(nwb_file_name)

        # Skip populating if no pos interval list names
        if len(pos_interval_list_names) == 0:
            print(f"NO POS INTERVALS FOR {key}; CANNOT POPULATE EpochIntervalListName")
            return

        # Get epoch number and corresponding interval list name
        x = (TaskEpoch & {"nwb_file_name": nwb_file_name}).fetch(
            "epoch", "interval_list_name"
        )
        epochs, epoch_interval_list_names = x[0], x[1]
        epoch_pos_valid_time_dict = {
            epoch: [] for epoch in epochs
        }  # store correspondence between epoch number and pos x valid time interval names
        for epoch, epoch_interval_list_name in zip(
            epochs, epoch_interval_list_names
        ):  # for each epoch
            epoch_valid_times = (
                IntervalList
                & {
                    "nwb_file_name": nwb_file_name,
                    "interval_list_name": epoch_interval_list_name,
                }
            ).fetch1(
                "valid_times"
            )  # get epoch valid times
            epoch_time_interval = [
                epoch_valid_times[0][0],
                epoch_valid_times[-1][-1],
            ]  # [epoch start, epoch end]
            epoch_time_interval_widened = np.asarray(
                [epoch_time_interval[0] - epsilon, epoch_time_interval[1] + epsilon]
            )  # widen to tolerate small differences in epoch boundaries across epoch/pos intervals
            for (
                pos_interval_list_name
            ) in pos_interval_list_names:  # for each pos valid time interval list
                pos_valid_times = (
                    IntervalList
                    & {
                        "nwb_file_name": nwb_file_name,
                        "interval_list_name": pos_interval_list_name,
                    }
                ).fetch1(
                    "valid_times"
                )  # get interval valid times
                pos_time_interval = np.asarray(
                    [pos_valid_times[0][0], pos_valid_times[-1][-1]]
                )  # [pos valid time interval start, pos valid time interval end]
                if np.logical_and(
                    epoch_time_interval_widened[0] < pos_time_interval[0],
                    epoch_time_interval_widened[1] > pos_time_interval[1],
                ):  # if pos valid time interval within epoch interval
                    epoch_pos_valid_time_dict[epoch].append(
                        pos_interval_list_name
                    )  # match pos valid time interval to epoch

        # Check that each pos interval was matched to only one epoch
        import itertools

        matched_pos_interval_list_names = list(
            itertools.chain.from_iterable(epoch_pos_valid_time_dict.values())
        )
        if len(np.unique(matched_pos_interval_list_names)) != len(
            matched_pos_interval_list_names
        ):
            raise Exception(
                "At least one pos interval list name was matched with more than one epoch"
            )
        # Unpack matching pos interval lists from array
        epoch_pos_valid_time_dict = {
            k: v[0] for k, v in epoch_pos_valid_time_dict.items() if len(v) > 0
        }

        # Insert into table if epoch matches interval list name
        # ...Exit function if epoch not in epoch_pos_valid_time_dict
        if key["epoch"] not in epoch_pos_valid_time_dict:
            return
        if epoch_pos_valid_time_dict[key["epoch"]] == key["interval_list_name"]:
            self.insert1(key)
            print(
                "Populated EpochIntervalListName for {nwb_file_name}, {epoch}".format(
                    **key
                )
            )

    def get_interval_list_name(self, nwb_file_name, epoch):
        return (self & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1(
            "interval_list_name"
        )


def get_pos_interval_list_names(nwb_file_name):
    return [
        interval_list_name
        for interval_list_name in (
            IntervalList & {"nwb_file_name": nwb_file_name}
        ).fetch("interval_list_name")
        if np.logical_and(
            interval_list_name.split(" ")[0] == "pos",
            " ".join(interval_list_name.split(" ")[2:]) == "valid times",
        )
    ]


def get_epoch_interval_list_names(nwb_file_name):
    return (TaskEpoch & {"nwb_file_name": nwb_file_name}).fetch("interval_list_name")
