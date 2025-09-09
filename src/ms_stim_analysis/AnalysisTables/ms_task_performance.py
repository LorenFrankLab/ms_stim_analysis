import datajoint as dj
import numpy as np

from .Utils.list_helpers import check_lists_same_length
from .ms_dio_event import get_poke_name, ProcessedDioEvents
from .ms_task_identification import TaskIdentification
from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("ms_task_performance")


@schema
class AlternationTaskPerformanceSel(SpyglassMixin, dj.Manual):
    definition = """
    # Selection from upstream tables for AlternationTaskPerformance
    -> ProcessedDioEvents
    """

    def insert_defaults(self):
        keys = [
            x
            for x in ProcessedDioEvents().fetch("KEY")
            if (TaskIdentification & x).fetch1("contingency") == "wtrack"
        ]
        for key in keys:
            self.insert1(key, skip_duplicates=True)


@schema
class AlternationTaskPerformance(SpyglassMixin, dj.Computed):
    definition = """
    # Mapping of well visits to reward and performance outcome
    -> AlternationTaskPerformanceSel
    ---
    previous_side_wells : blob
    previous_wells : blob
    current_wells : blob
    reward_outcomes : blob
    performance_outcomes : blob
    """

    def make(self, key):

        def _get_current_wells(well_pokes):
            return np.asarray(well_pokes)

        def _get_previous_wells(well_pokes, previous_well="none"):
            return np.asarray([previous_well] + well_pokes[:-1])

        def _get_previous_side_wells(well_pokes):
            previous_side_well = "none"
            previous_side_wells = [
                previous_side_well
            ]  # initialize list for previous side wells
            for well in well_pokes[:-1]:  # for each well visit
                if well in ["right_well", "left_well"]:  # if side well
                    previous_side_well = well  # update previous side well
                previous_side_wells.append(
                    previous_side_well
                )  # append previous side well
            return np.asarray(previous_side_wells)

        well_pokes = (ProcessedDioEvents.FirstUpPokes() & key).fetch1(
            "dio_first_poke_names"
        )
        well_pokes = [
            get_poke_name(x) for x in well_pokes
        ]  # replace with more informative name (e.g. poke3 --> well name)
        well_pokes = [
            x.replace("poke", "well") for x in well_pokes
        ]  # replace poke with well

        previous_side_wells = _get_previous_side_wells(well_pokes)
        previous_wells = _get_previous_wells(well_pokes)
        current_wells = _get_current_wells(well_pokes)
        reward_outcomes, performance_outcomes = zip(
            *[
                (
                    AlternationTaskRule
                    & {
                        "previous_side_well": previous_side_well,
                        "previous_well": previous_well,
                        "current_well": current_well,
                    }
                ).fetch1("reward_outcome", "performance_outcome")
                for previous_side_well, previous_well, current_well in zip(
                    previous_side_wells, previous_wells, current_wells
                )
            ]
        )

        # Populate parent table
        check_lists_same_length(
            [
                previous_side_wells,
                previous_wells,
                current_wells,
                reward_outcomes,
                performance_outcomes,
            ]
        )
        self.insert1(
            {
                **key,
                **{
                    "previous_side_wells": previous_side_wells,
                    "previous_wells": previous_wells,
                    "current_wells": current_wells,
                    "reward_outcomes": np.asarray(reward_outcomes),
                    "performance_outcomes": np.asarray(performance_outcomes),
                },
            },
        )


@schema
class AlternationTaskRule(SpyglassMixin, dj.Manual):
    definition = """
    # Mapping from well visits to reward and performance outcomes for alternation task on three-arm maze
    previous_side_well : varchar(40)
    previous_well : varchar(40)
    current_well : varchar(40)
    ---
    reward_outcome : varchar(40)
    performance_outcome : varchar(40)
    """

    def insert_defaults(self):
        """
        Populate AlternationTaskRule table in ms_task schema
        NOTE: for the purposes of processing downstream, all performance outcomes should start with "correct",
        "incorrect", or "neutral".
        """

        condition_list = (
            []
        )  # initialize list for tuples with alternation task rule conditions

        def _append_condition_list(
            condition_list,
            past_side_well,
            past_well,
            current_well,
            reward_outcome,
            performance_outcome,
        ):
            """Function for appending to list what will be a row in the table"""
            # Check that performance_outcome starts with "correct", "incorrect", or "neutral". Important for downstream
            # processing.
            if performance_outcome.split("_")[0] not in [
                "correct",
                "incorrect",
                "neutral",
            ]:
                raise Exception(
                    f"Performance outcome must start with correct, incorrect, or neutral,"
                    f" but starts with: {performance_outcome}"
                )
            condition_list.append(
                (
                    past_side_well,
                    past_well,
                    current_well,
                    reward_outcome,
                    performance_outcome,
                )
            )
            return condition_list

        side_wells = ["right_well", "left_well"]

        # Poke at home
        current_well = "center_well"
        # First poke ever
        condition_list = _append_condition_list(
            condition_list, "none", "none", current_well, "reward", "neutral"
        )
        # Correct inbound
        for past_well in side_wells:
            condition_list = _append_condition_list(
                condition_list,
                past_well,
                past_well,
                current_well,
                "reward",
                "correct_inbound",
            )

        # Poke at side well
        for current_well in side_wells:
            # First poke at side well after poke at home or no pokes
            for past_well in ["none", "center_well"]:
                condition_list = _append_condition_list(
                    condition_list, "none", past_well, current_well, "reward", "neutral"
                )
            # After poke at home
            past_well = "center_well"
            # Correct outbound
            condition_list = _append_condition_list(
                condition_list,
                [side_well for side_well in side_wells if side_well != current_well][0],
                past_well,
                current_well,
                "reward",
                "correct_outbound",
            )
            # Incorrect outbound
            condition_list = _append_condition_list(
                condition_list,
                current_well,
                past_well,
                current_well,
                "no_reward",
                "incorrect_outbound",
            )
            # After poke at other side well
            past_well = [
                side_well for side_well in side_wells if side_well != current_well
            ][0]
            condition_list = _append_condition_list(
                condition_list,
                past_well,
                past_well,
                current_well,
                "no_reward",
                "incorrect_inbound",
            )

        # Insert into table
        self.insert(condition_list, skip_duplicates=True)

    def outcomes_by_keyword(self, keywords):
        performance_outcomes = set(self.fetch("performance_outcome"))
        return {
            keyword: [x for x in performance_outcomes if keyword in x.split("_")]
            for keyword in keywords
        }


def performance_outcomes_to_int(performance_outcomes):
    performance_outcomes_int_map = {
        "neutral": np.nan,
        "correct_inbound": 1,
        "correct_outbound": 1,
        "incorrect_inbound": 0,
        "incorrect_outbound": 0,
    }
    return [performance_outcomes_int_map[x] for x in performance_outcomes]
