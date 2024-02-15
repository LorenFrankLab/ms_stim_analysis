import numpy as np
import pandas as pd
import spyglass as nd
import copy
from spyglass.common import (IntervalList,
                             Session)

from Time_and_trials.ms_interval import EpochIntervalListName
from Utils.string_helpers import snake_to_camel_case
from Utils.vector_helpers import unpack_single_element


def fetch1_tolerate_no_entry(table,
                             attribute):
    """
    Fetch a single entry from a datajoint table; return None if no
    :param table: datajoint table (or table subset)
    :param attribute: column for datajoint table to retrieve
    :return: table column
    """
    if len(table) > 0:  # if entry in table
        return table.fetch1(attribute)
    else:
        return None


def fetch1_dataframe_tolerate_no_entry(table_subset, object_name=None):
    """
    Fetch a single entry from a datajoint table subset; return None if no entry
    :param table_subset: subset of datajoint table
    :param object_name is necessary for fetch1_dataframe method used by tables that have analysis nwb files,
            but not those that dont.
    :return: single table entry or None if no table entries
    """
    if len(table_subset) > 0:  # if entry in table
        if object_name is None:
            return table_subset.fetch1_dataframe()
        return table_subset.fetch1_dataframe(object_name)
    else:
        return None


def get_pos_valid_times_interval_names(nwb_file_name):
    from spyglass.common import IntervalList
    pos_valid_times_interval_list_names = []
    for n in (IntervalList & {"nwb_file_name": nwb_file_name}).fetch("interval_list_name"):
        if len(np.asarray(n.split(" "))) == 4:
            if all(np.asarray(n.split(" "))[[0, 2, 3]] == np.asarray(["pos", "valid", "times"])):
                pos_valid_times_interval_list_names.append(n)
    return pos_valid_times_interval_list_names


def create_analysis_nwbf(key, nwb_objects, nwb_object_names):
    # Make copy of key to avoid altering key outside function
    key = copy.deepcopy(key)
    key['analysis_file_name'] = nd.common.AnalysisNwbfile().create(key['nwb_file_name'])
    nwb_analysis_file = nd.common.AnalysisNwbfile()
    # Check that objects all dfs (code currently assumes this in defining table_name)
    if not all([isinstance(x, pd.DataFrame) for x in nwb_objects]):
        raise Exception(f"create_analysis_nwbf currently assumes all objects dfs")
    for nwb_object_name, nwb_object in zip(nwb_object_names, nwb_objects):
        key[nwb_object_name] = nwb_analysis_file.add_nwb_object(
            analysis_file_name=key['analysis_file_name'],
            nwb_object=nwb_object,
            table_name=f"pandas_table_{nwb_object_name}")
    nwb_analysis_file.add(nwb_file_name=key['nwb_file_name'], analysis_file_name=key['analysis_file_name'])
    return key


def fetch1_dataframe(obj, object_name, restore_empty_nwb_object=True):
    entries = obj.fetch_nwb()
    if len(entries) != 1:
        raise Exception(f"Should have found exactly one entry but found {len(entries)}")
    df = entries[0][object_name]
    if restore_empty_nwb_object:  # restore altered empty nwb_object
        return handle_empty_nwb_object(df, from_empty=False)
    return df


def fetch_dataframe(table_subset, table_column_names=None):
    """
    Return multiple rows in datajoint table within single dataframe
    :param table_subset: datajoint table
    :return: dataframe
    """
    # Define inputs if not passed
    if table_column_names is None:
        table_column_names = return_datajoint_table_column_names(table_subset)  # all columns of table
    return pd.DataFrame({k: v for k, v in zip(table_column_names,
                                       list(zip(*fetch_iterable_array(table_subset, table_column_names))))})


def make_param_name(param_values):
    return "_".join([str(x) for x in param_values])


def handle_empty_nwb_object(nwb_object, from_empty):
    # Cannot save empty df out to analysis nwb file. Handle these cases below.
    # DATAFRAME
    # Empty df to one that can be saved: add a row with nans, and add a column to signal empty dataframe
    # Saved df restored to empty df: remove row with nans, remove column that had signaled empty dataframe.
    if isinstance(nwb_object, pd.DataFrame):
        empty_df_tag = "EMPTY_DF"  # add as column to empty df to be able to identify altered previously empty dfs
        # FROM EMPTY DF
        if from_empty and len(nwb_object) == 0:  # convert empty df
            for column in nwb_object.columns:
                nwb_object[column] = [np.nan]  # add row with nan
            nwb_object[empty_df_tag] = [np.nan]  # add flag so can know that empty df
        # TO EMPTY DF
        else:  # restore empty df
            # Identify altered empty df as one that has empty_df_tag as column, and has a single row with all nans
            if empty_df_tag in nwb_object and len(nwb_object) == 1 and all(np.isnan(nwb_object.iloc[0].values)):
                nwb_object.drop(labels=["EMPTY_DF"], axis=1, inplace=True)
                nwb_object.drop([0], inplace=True)
    else:
        raise Exception(f"Need to write code to handle empty nwb object of type {type(nwb_object)}")
    return nwb_object


def insert_analysis_table_entry(table, nwb_objects, key, nwb_object_names=None, convert_empty_nwb_object=True,
                                reset_index=False, replace_none_col_names=None):
    # Reset index in any dfs in nwb_objects if indicated (useful because currently index does not get stored
    # in analysis nwb file). Default is not True because if reset index when there is none, adds column called "index"
    if reset_index:
        nwb_objects = [x.reset_index() if isinstance(x, pd.DataFrame) else x for x in nwb_objects]
    # Convert None to "none" in specified df cols since None cannot be stored in analysis nwb file currently
    if replace_none_col_names is not None:
        for nwb_object in nwb_objects:
            if isinstance(nwb_object, pd.DataFrame):
                for col_name in replace_none_col_names:
                    nwb_object[col_name] = none_to_string_none(nwb_object[col_name])
    # Get nwb object names if not passed
    if nwb_object_names is None:
        nwb_object_names = table.get_object_id_name(unpack_single_object_id=False)
    # Convert nwb objects that are empty pandas dfs to something that can be saved out if indicated
    if convert_empty_nwb_object:
        nwb_objects = [handle_empty_nwb_object(x, from_empty=True) for x in nwb_objects]
    # Insert into table
    key = create_analysis_nwbf(key=key, nwb_objects=nwb_objects, nwb_object_names=nwb_object_names)
    table.insert1(key, skip_duplicates=True)  # insert into table
    print(f'Populated {table.table_name} for {key}')


def return_epoch_interval(nwb_file_name, epoch):
    interval_list_name = (EpochIntervalListName & {"nwb_file_name": nwb_file_name,
                                                   "epoch": epoch}).fetch1("interval_list_name")
    epoch_valid_times = (IntervalList & {"nwb_file_name": nwb_file_name,
                                         "interval_list_name": interval_list_name}).fetch1("valid_times")
    return [epoch_valid_times[0][0], epoch_valid_times[-1][-1]]


def check_nwb_file_inserted(nwb_file_name):
    if len((Session & {'nwb_file_name': nwb_file_name})) == 0:
        raise Exception("nwb file not in Session table")


def check_nwb_files_inserted(nwb_file_names):
    for nwb_file_name in nwb_file_names:
        check_nwb_file_inserted(nwb_file_name)


def intervals_within_epoch_bool(key, start_times, end_times):
    # Exclude trials with start or end time outside epoch
    epoch_start_time, epoch_end_time = return_epoch_interval(key["nwb_file_name"],
                                                             key["epoch"])
    return np.logical_and(start_times > epoch_start_time,
                                      end_times < epoch_end_time)


def format_sort_group_unit_id(sort_group, unit_id):
    return f"{sort_group}_{unit_id}"


def unformat_sort_group_unit_id(formated_sort_group_unit_id):
    return list(map(int, formated_sort_group_unit_id.split("_")))


def unformat_sort_group_unit_ids(formated_sort_group_unit_ids):
    return list(zip(*[unformat_sort_group_unit_id(x) for x in formated_sort_group_unit_ids]))


def format_nwb_file_name(nwb_file_name):
    return nwb_file_name.split("_")[0]


def get_table_object_id_name(table, leave_out_object_id=False, unpack_single_object_id=True, tolerate_none=False):
    object_id_names = [x for x in get_table_column_names(table) if "object_id" in x]
    if len(object_id_names) == 0 and tolerate_none:
        return None
    if leave_out_object_id:
        object_id_names = [x.replace("_object_id", "") for x in object_id_names]
    if len(object_id_names) == 1 and unpack_single_object_id:
        return unpack_single_element(object_id_names)
    return object_id_names


def get_table_column_names(table):
    return list(table.fetch().dtype.fields.keys())


def insert1_print(table, key):
    table.insert1(key, skip_duplicates=True)
    print(f"Added entry to {get_table_name(table)} for {key}")


def get_table_name(table):
    return snake_to_camel_case(table.table_name)

