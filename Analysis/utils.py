import datajoint as dj
from spyglass.common import Session, interval_list_contains_ind, interval_list_intersect
from datajoint.user_tables import UserTable
import numpy as np
from typing import Tuple

import os
import scipy
import matplotlib.pyplot as plt
from Metadata.ms_task_identification import TaskIdentification
from Time_and_trials.ms_interval import EpochIntervalListName

os.chdir("/home/sambray/Documents/MS_analysis_samsplaying/")
from ms_opto_stim_protocol import (
    OptoStimProtocol,
    OptoStimProtocolTransfected,
    OptoStimProtocolLaser,
    OptoStimProtocolClosedLoop,
)
from Analysis.position_analysis import get_running_intervals, filter_position_ports


def filter_animal(table: UserTable, animal: str) -> UserTable:
    """filter table with all sessions for an animal

    Parameters
    ----------
    table : UserTable
        table to filter
    animal : str
        animal to include

    Returns
    -------
    UserTable
        filtered table
    """
    if len(animal) == 0:
        return table
    return table & ((table * Session) & {"subject_id": animal}).fetch("KEY")


def filter_task(table: UserTable, task: str) -> UserTable:
    """filter table with all epochs for a given task type (e.g. "lineartrack)

    Parameters
    ----------
    table : UserTable
        table to filter
    task : str
        task type to include

    Returns
    -------
    UserTable
        filtered table
    """
    wtrack_aliases = ["wtrack", "w-track", "w track", "W-track", "W track", "Wtrack"]
    lineartrack_aliases = [
        "lineartrack",
        "linear-track",
        "linear track",
        "Linear-track",
        "Linear track",
        "Lineartrack",
    ]
    alias_sets = [wtrack_aliases, lineartrack_aliases]

    if len(task) == 0:
        return table
    for alias_set in alias_sets:
        if task in alias_set:
            keys = []
            for alias in alias_set:
                keys.extend(
                    (
                        table * EpochIntervalListName * TaskIdentification
                        & {"contingency": alias}
                    ).fetch("KEY")
                )
            return table & keys

    return table & (
        table * EpochIntervalListName * TaskIdentification & {"contingency": task}
    ).fetch("KEY")


def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def convert_delta_marks_to_timestamp_values(
    marks: list, mark_timestamps: list, sampling_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """convert delta marks to data values at regularly sampled timestamps

    Parameters
    ----------
    marks : list
        values of delta marks
    mark_timestamps : list
        when the marks occur
    sampling_rate : int
        desired sampling rate

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        data, timestamps
    """
    timestamps = np.arange(
        mark_timestamps[0],
        mark_timestamps[-1],
        1 / sampling_rate,
    )
    data = np.zeros(len(timestamps))

    ind = np.argmin(np.abs(timestamps - mark_timestamps[0]))
    data[:ind] = 1 - marks[0]
    for i in range(len(marks) - 1):
        ind_new = np.argmin(np.abs(timestamps - mark_timestamps[i]))
        data[ind:ind_new] = 1 - marks[i + 1]
        ind = ind_new.copy()
    return data, timestamps


def bootstrap(x, n_bootstraps, func=np.mean):
    """Bootstrap a function func by sampling with replacement from x."""
    n = len(x)
    idx = np.random.randint(0, n, (n_bootstraps, n))
    return func(np.array(x)[idx], axis=1)


def get_running_valid_intervals(
    pos_key: dict,
    filter_speed: float = 10,
    filter_ports: bool = True,
    seperate_optogenetics: bool = True,
):
    """Find intervals where rat is running and not in a port.  if seperate_optogenetics, then also separate into intervals where optogenetics are and ar not running

    Args:
        pos_key (dict): key to find the position data
        filter_speed (float, optional): speed threshold for running. Defaults to 10.
        filter_ports (bool, optional): whether to filter out port intervals. Defaults to True.
        seperate_optogenetics (bool, optional): whether to seperate into optogenetic and control intervals. Defaults to True.

    Returns:
        if not seperate_optogenetics:
        run_intervals (list): intervals where rat is running
        if seperate_optogenetics:
        optogenetic_run_interval (list): intervals where rat is running and in optogenetic interval
        control_run_interval (list): intervals where rat is running and in control interval
    """
    # make intervals where rat is running
    run_intervals = get_running_intervals(**pos_key, filter_speed=filter_speed)
    # intersect with position-defined intervals
    if filter_ports:
        valid_position_intervals = filter_position_ports(pos_key)
        run_intervals = interval_list_intersect(
            np.array(run_intervals), np.array(valid_position_intervals)
        )
    if not seperate_optogenetics:
        return run_intervals

    # determine if each interval is in the optogenetic control interval
    control_interval = (OptoStimProtocol() & pos_key).fetch1("control_intervals")
    test_interval = (OptoStimProtocol() & pos_key).fetch1("test_intervals")
    if len(control_interval) == 0 or len(test_interval) == 0:
        print(f"Warning: no optogenetic intervals found for {pos_key}")
        return np.array([]), np.array([])
    optogenetic_run_interval = interval_list_intersect(
        np.array(run_intervals), np.array(test_interval)
    )
    control_run_interval = interval_list_intersect(
        np.array(run_intervals), np.array(control_interval)
    )
    return optogenetic_run_interval, control_run_interval


def autocorr2d(x):
    """Efficiently compute autocorrelation along 1 axis of a 2D array

    Args:
        x (np.array): data to compute autocorrelation of (n_samples, n_features)

    Returns:
        corr (np.array): autocorrelation of x along axis 0 (n_samples, n_features)
    """
    n = x.shape[0]
    # Zero-pad the array for FFT-based convolution
    padded_x = np.pad(x, ((0, n), (0, 0)), "constant")

    # Compute FFT and its complex conjugate
    X_f = np.fft.fft(padded_x, axis=0)
    result = np.fft.ifft(X_f * np.conj(X_f), axis=0).real

    # Return the positive lags
    return result[:n] / result[0]


def filter_opto_data(dataset_key: dict):
    """filter optogenetic data based on the dataset key

    Args:
        dataset_key (dict): restriction to filter by

    Returns:
        Table: filtered table
    """
    # define datasets
    dataset_table = OptoStimProtocol
    if "transfected" in dataset_key:
        dataset_table = dataset_table * OptoStimProtocolTransfected
    if "laser_power" in dataset_key:
        dataset_table = dataset_table * OptoStimProtocolLaser
    if "targeted_phase" in dataset_key:
        dataset_table = dataset_table * OptoStimProtocolClosedLoop
    dataset = dataset_table & dataset_key
    if "animal" in dataset_key:
        dataset = filter_animal(dataset, dataset_key["animal"])
    if "track_type" in dataset_key:
        dataset = filter_task(dataset, dataset_key["track_type"])
    if "min_pulse_length" in dataset_key:
        dataset = dataset & f"pulse_length_ms>{dataset_key['min_pulse_length']}"
    if "max_pulse_length" in dataset_key:
        dataset = dataset & f"pulse_length_ms<{dataset_key['max_pulse_length']}"
    print("datasets:", len(dataset))
    return dataset


def smooth(data, n=5, sigma=None):
    """smooths data with gaussian kernel of size n"""
    if n % 2 == 0:
        n += 1  # make sure n is odd
    if sigma is None:
        sigma = n / 2
    kernel = gkern(n, sigma)[:, None]
    if len(data.shape) == 1:
        pad = np.ones(((n - 1) // 2, 1))
        return np.squeeze(
            scipy.signal.convolve2d(
                np.concatenate(
                    [pad * data[:, None][0], data[:, None], pad * data[:, None][-1]],
                    axis=0,
                ),
                kernel,
                mode="valid",
            )
        )
    else:
        pad = np.ones(((n - 1) // 2, data.shape[1]))
        return scipy.signal.convolve2d(
            np.concatenate([pad * data[0], data, pad * data[-1]], axis=0),
            kernel,
            mode="valid",
        )


def gkern(l: int = 5, sig: float = 1.0):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return gauss / np.sum(gauss)
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def violin_scatter(data,pos=0,color="cornflowerblue",bw_method=None, ax=None, return_locs=False):
    """plot a violin plot with scatter points jiittered around the width of the violin plot"""
    if ax is None:
        ax = plt.gca()
    vp = ax.violinplot(data,positions=[pos],showmedians=False,showextrema=False,points=1000,bw_method=bw_method)
    body = vp['bodies'][0]
    body.set_facecolor(color)
    path = body.get_paths()[0].vertices
    x_data, y_data = path[:, 0], path[:, 1]
    y_data = y_data[x_data.size//2:]#-pos
    x_data = x_data[x_data.size//2:]-pos
    print(y_data.min())
    width = x_data[np.digitize(data,y_data,right=False)]
    x_pos = np.random.normal(0,.3,len(data))*width+pos
    ax.scatter(x_pos,data,alpha=.5,color=color)
    if return_locs:
        return x_pos, data
    return


def get_slope(data,time):
    from scipy.stats import linregress
    slope = []
    for i in range(data.shape[0]):
        slope.append(linregress(time,data[i]).slope)
    return np.array(slope)