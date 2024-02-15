from spyglass.common import TaskEpoch


def get_ms_rat_names():
    return ["Banner", "Winnie", "Totoro", "Yoshi"]


def get_ms_nwbf_epoch_keys():
    ms_rat_names = get_ms_rat_names()
    nwb_file_names, epochs = TaskEpoch.fetch("nwb_file_name", "epoch")
    valid_bool = [any([ms_rat_name in nwb_file_name for ms_rat_name in ms_rat_names])
                  for nwb_file_name in nwb_file_names]
    return [{"nwb_file_name": nwb_file_name,
             "epoch": epoch} for nwb_file_name, epoch in
            zip(nwb_file_names[valid_bool], epochs[valid_bool])]
