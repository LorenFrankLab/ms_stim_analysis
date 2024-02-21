import numpy as np
import itertools


def check_single_element(x):
    """
    Check list for existence of a single unique element
    :param x: list with elements
    """
    unique_elements = np.unique(list(x))
    if len(unique_elements) != 1:
        raise Exception(f"Should have found one unique element in list but found {len(unique_elements)}")


def check_return_single_element(x):
    """
    Check list for existence of a single unique element and return if exists, otherwise raise error
    :param x: list with elements
    :return: single unique element (if list had one)
    """
    unique_elements = np.unique(list(x))
    if len(unique_elements) != 1:
        raise Exception(f"Should have found one unique element in list but found {len(unique_elements)}")
    return unique_elements[0]


def check_lists_same_length(lists, lists_description="Lists"):
    var_lengths = np.unique(list(map(len, lists)))
    if len(var_lengths) != 1:
        raise Exception(f"{lists_description} must all have same length, but set of lengths is: {var_lengths}")


def duplicate_inside_elements(x, num_duplicates=2):
    return ([x[0]]
            + list(itertools.chain.from_iterable([[x_i]*num_duplicates
                                                  for idx, x_i in enumerate(x)
                                                  if idx not in [0, len(x) - 1]]))
            + [x[-1]])


def duplicate_elements(x, num_duplicates=2):
    return list(itertools.chain.from_iterable([[x_i]*num_duplicates for x_i in x]))


def unzip_as_array(list_tuples):
    x, y = zip(*list_tuples)
    return np.asarray(x), np.asarray(y)


def unzip_as_list(list_tuples):
    x, y = zip(*list_tuples)
    return list(x), list(y)


def remove_duplicate_combinations(x):
    return list(set(map(tuple, map(sorted, x))))


def return_n_empty_lists(n, as_array=False):
    if as_array:
        return tuple([np.asarray([]) for _ in np.arange(0, n)])
    return tuple([[] for _ in np.arange(0, n)])


def zip_adjacent_elements(x):
    return list(zip(x[:-1], x[1:]))


def append_multiple_lists(variables, lists):
    # Check inputs
    if len(variables) != len(lists):
        raise Exception(f"number of lists to be appended must be same as number of variables")
    appended_lists = []
    for variable, list_ in zip(variables, lists):
        list_.append(variable)
        appended_lists.append(list_)
    return appended_lists


def check_alternating_elements(x, element_1, element_2):
    # Check inputs
    if element_1 == element_2:
        raise Exception(f"elements 1 and 2 cannot be the same")
    x = np.asarray(x)  # convert to array
    x1_idxs = np.where(x == element_1)[0]  # where array has first element
    x2_idxs = np.where(x == element_2)[0]  # where array has second element
    # Check that x contains only elements 1 and 2
    if len(x1_idxs) + len(x2_idxs) != len(x):
        raise Exception(f"passed x contains elements other than {element_1} and {element_2}")
    # Return if one element in x
    if len(x) == 1:
        return
    # Check that idxs alternating
    # To do so, must index into idxs lists above. Use shortest length across these lists to avoid
    # indexing error.
    end_idx = np.min(list(map(len, [x1_idxs, x2_idxs])))
    if abs(check_return_single_element(x1_idxs[:end_idx] - x2_idxs[:end_idx])) != 1:
        raise Exception(f"List does not contain alternating {element_1} and {element_2}")
    # Check that last idx not same as second to last (escapes check above)
    if len(x) > 1:
        if x[-1] == x[-2]:
            raise Exception(f"List does not contain alternating {element_1} and {element_2}")


def element_frequency(x, elements=None):
    if elements is None:
        elements = set(x)
    return {element: np.sum(x == element) for element in elements}