import numpy as np
import pandas as pd


def df_filter_columns(df, key, column_and=True):
    if column_and:
        return df[
            np.asarray([df[k] == v for k, v in key.items()]).sum(axis=0) == len(key)
        ]
    else:
        return df[np.asarray([df[k] == v for k, v in key.items()]).sum(axis=0) > 0]


def df_filter_columns_greater_than(df, key, column_and=True):
    num_valid_columns = np.asarray([df[k] > v for k, v in key.items()]).sum(
        axis=0
    )  # num columns in key meeting less than condition
    if column_and:
        return df[num_valid_columns == len(key)]
    else:
        return df[num_valid_columns > 0]


def df_filter_columns_less_than(df, key, column_and=True):
    num_valid_columns = np.asarray([df[k] < v for k, v in key.items()]).sum(
        axis=0
    )  # num columns in key meeting less than condition
    if column_and:
        return df[num_valid_columns == len(key)]
    else:
        return df[num_valid_columns > 0]


def df_filter1_columns(df, key, tolerate_no_entry=False):
    df_subset = df_filter_columns(df, key)
    if np.logical_or(len(df_subset) > 1, not tolerate_no_entry and len(df_subset) == 0):
        raise Exception(
            f"Should have found exactly one entry in df for key, but found {len(df_subset)}"
        )
    return df_subset


def df_pop(df, key, column, tolerate_no_entry=False):
    df_subset = df_filter1_columns(df, key, tolerate_no_entry)
    if len(df_subset) == 0:  # empty df
        return df_subset
    return df_subset.iloc[0][column]


def df_from_data_list(data_list, entry_names):
    return pd.DataFrame.from_dict({k: v for k, v in zip(entry_names, zip(*data_list))})
