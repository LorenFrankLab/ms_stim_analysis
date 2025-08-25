import numpy as np


def check_lists_same_length(lists, lists_description="Lists"):
    var_lengths = np.unique(list(map(len, lists)))
    if len(var_lengths) != 1:
        raise Exception(
            f"{lists_description} must all have same length, but set of lengths is: {var_lengths}"
        )
