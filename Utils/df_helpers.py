import numpy as np
import pandas as pd
from Utils.point_process_helpers import event_times_in_intervals_bool


def df_filter_columns(df, key, column_and=True):
    if column_and:
        return df[np.asarray([df[k] == v for k, v in key.items()]).sum(axis=0) == len(key)]
    else:
        return df[np.asarray([df[k] == v for k, v in key.items()]).sum(axis=0) > 0]

def df_filter_columns_greater_than(df, key, column_and=True):
    num_valid_columns = np.asarray([df[k] > v for k, v in key.items()]).sum(axis=0)  # num columns in key meeting less than condition
    if column_and:
        return df[num_valid_columns == len(key)]
    else:
        return df[num_valid_columns > 0]


def df_filter_columns_less_than(df, key, column_and=True):
    num_valid_columns = np.asarray([df[k] < v for k, v in key.items()]).sum(axis=0)  # num columns in key meeting less than condition
    if column_and:
        return df[num_valid_columns == len(key)]
    else:
        return df[num_valid_columns > 0]


def df_filter1_columns(df, key, tolerate_no_entry=False):
    df_subset = df_filter_columns(df, key)
    if np.logical_or(len(df_subset) > 1,
                     not tolerate_no_entry and len(df_subset) == 0):
        raise Exception(f"Should have found exactly one entry in df for key, but found {len(df_subset)}")
    return df_subset


def df_pop(df, key, column, tolerate_no_entry=False):
    df_subset = df_filter1_columns(df, key, tolerate_no_entry)
    if len(df_subset) == 0:  # empty df
        return df_subset
    return df_subset.iloc[0][column]


def df_filter_columns_isin(df, key):
    if len(key) == 0:  # if empty key
        return df
    return df[np.sum(np.asarray([df[k].isin(v) for k, v in key.items()]), axis=0) == len(key)]
    # Alternate code: df[df[list(df_filter)].isin(df_filter).all(axis=1)]


def df_filter_columns_contains(df, target_column, target_str):
    return df[df[target_column].str.contains(target_str)]


def zip_df_columns(df, column_names=None):
    if column_names is None:
        column_names = df.columns
    return zip(*[df[column_name] for column_name in column_names])


def unpack_df_columns(df, column_names):
    # TODO: Turning series into tuple seems to get rid of series. So should find alternative strategy for this function.
    # For dfs with single row, this works: df_subset[["test_fold_lengths", "y_test", "y_test_predicted"]].to_numpy()[0]
    # Should see if can change to make work with any number of rows.
    return tuple([np.asarray(x) for x in zip(*zip_df_columns(df, column_names))])


def df_filter_index(df, valid_intervals):
    return df[event_times_in_intervals_bool(event_times=df.index,
                                            valid_time_intervals=valid_intervals)]


def df_from_data_list(data_list, entry_names):
    return pd.DataFrame.from_dict({k: v for k, v in zip(entry_names, zip(*data_list))})
