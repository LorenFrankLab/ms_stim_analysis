import datajoint as dj

from spyglass.common import TaskEpoch

from spyglass.utils.dj_mixin import SpyglassMixin

schema = dj.schema("ms_metadata")


@schema
class TaskIdentification(SpyglassMixin, dj.Computed):
    definition = """
    # Epoch environment and reward contingency
    -> TaskEpoch
    ---
    contingency : varchar(40)
    task_environment : varchar(40)
    """

    def make(self, key):
        task_epoch_entry = (TaskEpoch & key).fetch1()
        # Assumes task_name has either form: "trackName_contingencyName_environmentName_delay.sc" or "sleep" or "home"
        if len(task_epoch_entry["task_name"].split("_")) == 1:
            key["contingency"] = task_epoch_entry["task_name"]
        else:
            key["contingency"] = task_epoch_entry["task_name"].split("_")[1]
        key["task_environment"] = task_epoch_entry["task_environment"]
        self.insert1(key)  # insert into table
        print(
            "Populated TaskIdentification for file {nwb_file_name}, epoch {epoch}".format(
                **key
            )
        )

    def get_contingency(self, nwb_file_name, epoch):
        return (self & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1(
            "contingency"
        )

    def get_environment(self, nwb_file_name, epoch):
        return (self & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch1(
            "task_environment"
        )
