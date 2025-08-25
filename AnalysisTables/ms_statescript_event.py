import numpy as np
import datajoint as dj
from spyglass.common import StateScriptFile
from spyglass.utils.dj_mixin import SpyglassMixin

from .ms_task_identification import TaskIdentification

TaskIdentification

schema = dj.schema("ms_statescript_event")


@schema
class StatescriptEvents(SpyglassMixin, dj.Computed):
    definition = """
    # DIO events in statescript log
    -> TaskIdentification  # use this instead of TaskEpoch to limit table to JAG recordings
    -> StateScriptFile
    ---
    statescript_event_names : blob
    statescript_event_times_trodes : blob
    """

    def make(self, key):
        # Get statescript file for this epoch and split into lines
        ss_file_entry = (StateScriptFile & key).fetch_nwb()
        # Find statescript printouts (lines that are not comments and have content)
        state_script_printouts = [
            x
            for x in [
                z
                for z in ss_file_entry[0]["file"].fields["content"].split("\n")
                if len(z) > 0
            ]
            if x[0] != "#"
        ]  # note that must first find lines with content to enbale search for hash indicating comment
        # Parse printouts into event times and event (printouts have form "time event")
        event_times, event_names = zip(
            *[
                (int(line.split(" ")[0]), " ".join(line.split(" ")[1:]))
                for line in state_script_printouts
            ]
        )
        key.update(
            {
                "statescript_event_names": np.asarray(event_names),
                "statescript_event_times_trodes": np.asarray(event_times),
            }
        )
        self.insert1(key)
