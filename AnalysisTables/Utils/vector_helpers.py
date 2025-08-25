import numpy as np


def check_length(x, expected_length):
    if len(x) != expected_length:
        raise Exception(f"Vector length is {len(x)}, but should be length {expected_length}")


def unpack_single_element(x, tolerate_no_entry=False, return_no_entry=None):
    if tolerate_no_entry:
        if len(x) == 0:
            return return_no_entry
        return unpack_single_element(x, tolerate_no_entry=False)
    check_length(x, expected_length=1)  # first check only one element
    return x[0]


def remove_repeat_elements(x, keep_first=True):
    """
    Remove consecutive elements in array
    :param x: array-like.
    :param keep_first: default True. If True, keep first element in a stretch of repeats. Otherwise, keep last.
    :return: x after removing consecutive elements.
    :return keep_x_idxs: kept indices in x after removing consecutive elements.
    """
    x = np.asarray(x)  # ensure array
    if len(x) == 0:  # if no values in array
        return x, []
    if len(x.shape) != 1:  # check that 1D array-like passed
        raise Exception(f"Must pass 1D array-like. Shape of passed item: {x.shape}")
    if keep_first:  # keep first element in repeats
        keep_x_idxs = np.concatenate((np.array([0]),
                                  np.where(x[1:] != x[:-1])[0] + 1))
    else:  # keep last element in repeats
        keep_x_idxs = np.concatenate((np.where(x[1:] != x[:-1])[0],
                              np.array([len(x) - 1])))
    return x[keep_x_idxs], keep_x_idxs


def none_to_string_none(x):
    return np.asarray(["none" if x_i is None else x_i for x_i in x])
